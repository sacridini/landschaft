// SPDX-License-Identifier: GPL-3.0-or-later
#include "cost/frontal_lu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <queue>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <Eigen/Core>

#if defined(_M_X64) || defined(__x86_64__) || defined(__SSE2__)
#include <xmmintrin.h>
#define COST_HAVE_MXCSR 1
#endif

namespace cost {

namespace {

using Block = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>;
using ConstBlock = Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>>;

constexpr std::int64_t kBlock = 64;      // column block of the dense kernel
constexpr std::int64_t kTileCols = 128;  // tiles of the trailing update
constexpr std::int64_t kTileRows = 256;
constexpr std::int64_t kParallel = 256;  // fronts from which the kernel runs in parallel

// Flush-to-zero and denormals-are-zero on this thread while in scope: the
// Schur complements of I - W decay into subnormal numbers, whose arithmetic
// is two orders of magnitude slower; below 1e-308 they do not matter here.
struct FlushDenormals {
#if defined(COST_HAVE_MXCSR)
    unsigned int saved = _mm_getcsr();
    FlushDenormals() { _mm_setcsr(saved | 0x8040u); }
    ~FlushDenormals() { _mm_setcsr(saved); }
#endif
};

bool in_parallel() {
#if defined(_OPENMP)
    return omp_in_parallel() != 0;
#else
    return true;
#endif
}

// Partial LU without pivoting of the front F (m x m, column-major): the
// first np columns (and rows) eliminated, blocked right-looking (the
// diagonal block unblocked, the panels below and to the right of it by
// triangular solves, the trailing block by products in fixed tiles, in
// parallel for large fronts); the trailing (m - np) block ends as the Schur
// complement. The tiles do not depend on the number of threads, nor do the
// results.
bool partial_lu(double* F, std::int64_t m, std::int64_t np) {
    const FlushDenormals ftz;
    const bool parallel = m >= kParallel && !in_parallel();
    (void)parallel;
    for (std::int64_t k0 = 0; k0 < np; k0 += kBlock) {
        const std::int64_t kb = std::min(kBlock, np - k0), k1 = k0 + kb;
        for (std::int64_t k = k0; k < k1; ++k) {  // the diagonal block
            const double piv = F[k * m + k];
            if (piv == 0 || !std::isfinite(piv)) return false;
            double* lk = F + k * m;
            for (std::int64_t i = k + 1; i < k1; ++i) lk[i] /= piv;
            for (std::int64_t j = k + 1; j < k1; ++j) {
                double* fj = F + j * m;
                const double ukj = fj[k];
                for (std::int64_t i = k + 1; i < k1; ++i) fj[i] -= lk[i] * ukj;
            }
        }
        if (k1 >= m) continue;
        const std::int64_t nr = m - k1;
        const ConstBlock D(F + k0 * m + k0, kb, kb, Eigen::OuterStride<>(m));
        const std::int64_t rchunks = (nr + kTileRows - 1) / kTileRows, cchunks = (nr + kTileCols - 1) / kTileCols;
        const std::int64_t panels = rchunks + cchunks;
        // L below the block: X U_D = F[k1:m, k0:k1]; U to its right: L_D X = F[k0:k1, k1:m]
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) if (parallel && panels > 1)
#endif
        for (std::int64_t c = 0; c < panels; ++c) {
            const FlushDenormals ftz_thread;
            if (c < rchunks) {
                const std::int64_t i0 = k1 + c * kTileRows, ni = std::min(kTileRows, m - i0);
                Block P(F + k0 * m + i0, ni, kb, Eigen::OuterStride<>(m));
                D.triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheRight>(P);
            } else {
                const std::int64_t j0 = k1 + (c - rchunks) * kTileCols, nj = std::min(kTileCols, m - j0);
                Block U(F + j0 * m + k0, kb, nj, Eigen::OuterStride<>(m));
                D.triangularView<Eigen::UnitLower>().solveInPlace(U);
            }
        }
        // the trailing block, tile by tile
        const std::int64_t tiles = rchunks * cchunks;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) if (parallel && tiles > 1)
#endif
        for (std::int64_t tt = 0; tt < tiles; ++tt) {
            const FlushDenormals ftz_thread;
            const std::int64_t i0 = k1 + (tt % rchunks) * kTileRows, ni = std::min(kTileRows, m - i0);
            const std::int64_t j0 = k1 + (tt / rchunks) * kTileCols, nj = std::min(kTileCols, m - j0);
            const ConstBlock Li(F + k0 * m + i0, ni, kb, Eigen::OuterStride<>(m));
            const ConstBlock Uj(F + j0 * m + k0, kb, nj, Eigen::OuterStride<>(m));
            Block T(F + j0 * m + i0, ni, nj, Eigen::OuterStride<>(m));
            T.noalias() -= Li * Uj;
        }
    }
    return true;
}

}  // namespace

FrontalLU::FrontalLU(std::shared_ptr<const circuit::FrontalSymbolic> sym) : sym_(std::move(sym)) {
    const auto& nodes = sym_->nodes;
    lpos_.resize(nodes.size());
    upos_.resize(nodes.size());
    std::int64_t l = 0, u = 0;
    for (std::size_t t = 0; t < nodes.size(); ++t) {
        const std::int64_t np = nodes[t].npiv, nu = nodes[t].nupd;
        lpos_[t] = l;
        upos_[t] = u;
        l += (np + nu) * np;
        u += np * nu;
    }
    l_.resize(static_cast<std::size_t>(l));
    u_.resize(static_cast<std::size_t>(u));
}

bool FrontalLU::factorize(const double* diag, const double* val, const std::int64_t* rev) {
    const circuit::FrontalSymbolic& s = *sym_;
    const std::int32_t nn = static_cast<std::int32_t>(s.nodes.size());
    int threads = 1;
#if defined(_OPENMP)
    threads = omp_in_parallel() ? 1 : omp_get_max_threads();
#endif
    // Parts: subtrees factorised in parallel, each by one thread (planned as
    // the circuit module's Cholesky plans them); then the nodes above them,
    // by heights: several in parallel, or a few large ones with parallel kernels.
    std::vector<double> work(static_cast<std::size_t>(nn), 0.0);
    std::vector<std::int32_t> begin(static_cast<std::size_t>(nn)), height(static_cast<std::size_t>(nn), 0);
    std::vector<std::uint8_t> has_parent(static_cast<std::size_t>(nn), 0);
    for (std::int32_t t = 0; t < nn; ++t) {
        const auto& node = s.nodes[t];
        const double m = static_cast<double>(node.npiv) + node.nupd;
        work[t] += static_cast<double>(node.npiv) * m * m + m;
        begin[t] = node.child_begin < node.child_end ? begin[s.children[node.child_begin]] : t;
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const std::int32_t c = s.children[ci];
            work[t] += work[c];
            has_parent[c] = 1;
            height[t] = std::max(height[t], height[c] + 1);
        }
    }
    std::vector<std::int32_t> roots;
    {
        auto heavier = [&](std::int32_t a, std::int32_t b) { return work[a] < work[b] || (work[a] == work[b] && a > b); };
        std::priority_queue<std::int32_t, std::vector<std::int32_t>, decltype(heavier)> heap(heavier);
        double total = 0.0;
        for (std::int32_t t = 0; t < nn; ++t)
            if (!has_parent[t]) {
                heap.push(t);
                total += work[t];
            }
        const int parts = threads > 1 ? 4 * threads : 1;
        while (static_cast<int>(heap.size()) < parts) {
            const std::int32_t t = heap.top();
            const auto& node = s.nodes[t];
            if (node.child_begin == node.child_end || work[t] * 4 * parts <= total) break;
            heap.pop();
            for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) heap.push(s.children[ci]);
        }
        while (!heap.empty()) {
            roots.push_back(heap.top());
            heap.pop();
        }
    }
    std::vector<std::uint8_t> in_part(static_cast<std::size_t>(nn), 0);
    for (const std::int32_t r : roots)
        for (std::int32_t t = begin[r]; t <= r; ++t) in_part[t] = 1;
    std::vector<std::vector<double>> updates(static_cast<std::size_t>(nn));  // Schur complements, until used
    bool failed = false;

    auto one = [&](std::int32_t t) {
        const circuit::FrontalSymbolic::Node& node = s.nodes[t];
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        std::vector<double> front(static_cast<std::size_t>(m * m), 0.0);
        double* F = front.data();
        for (std::int32_t p = node.first; p < node.first + node.npiv; ++p) {
            const std::int64_t j = p - node.first;
            F[j * m + j] += diag[static_cast<std::size_t>(s.value_index(p))];
            for (std::int64_t k = s.entry_begin(p); k < s.entry_end(p); ++k) {
                const std::int32_t a = s.amap[static_cast<std::size_t>(k)];
                if (a < 0) continue;
                F[a * m + j] += val[k];       // row p, a later column: U
                F[j * m + a] += val[rev[k]];  // its reverse, a later row: L
            }
        }
        // extend-add the children's Schur complements, in order
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const std::int32_t c = s.children[ci];
            const std::int64_t u = s.nodes[c].nupd;
            const std::int32_t* rel = s.relmap.data() + s.nodes[c].upd;
            const double* U = updates[c].data();
            for (std::int64_t jj = 0; jj < u; ++jj) {
                double* col = F + static_cast<std::int64_t>(rel[jj]) * m;
                const double* ucol = U + jj * u;
                for (std::int64_t ii = 0; ii < u; ++ii) col[rel[ii]] += ucol[ii];
            }
            std::vector<double>().swap(updates[c]);
        }
        if (!partial_lu(F, m, np)) return false;
        double* L = l_.data() + lpos_[t];
        for (std::int64_t j = 0; j < np; ++j) std::copy(F + j * m, F + j * m + m, L + j * m);
        double* U12 = u_.data() + upos_[t];
        for (std::int64_t j = 0; j < nu; ++j)
            for (std::int64_t i = 0; i < np; ++i) U12[j * np + i] = F[(np + j) * m + i];
        updates[t].resize(static_cast<std::size_t>(nu * nu));
        double* S = updates[t].data();
        for (std::int64_t j = 0; j < nu; ++j) std::copy(F + (np + j) * m + np, F + (np + j) * m + m, S + j * nu);
        return true;
    };

    const bool profile = std::getenv("LANDSCHAFT_PROFILE") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    const std::int64_t n_roots = static_cast<std::int64_t>(roots.size());
#pragma omp parallel for schedule(dynamic, 1) if (threads > 1 && n_roots > 1)
    for (std::int64_t k = 0; k < n_roots; ++k) {
        const std::int32_t r = roots[static_cast<std::size_t>(k)];
        for (std::int32_t t = begin[r]; t <= r; ++t)
            if (!one(t)) {
#pragma omp critical(cost_frontal_lu_failed)
                failed = true;
                break;
            }
    }
    if (failed) return false;
    const double t_parts = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    t0 = std::chrono::steady_clock::now();
    std::int32_t hmax = 0;
    for (std::int32_t t = 0; t < nn; ++t)
        if (!in_part[t]) hmax = std::max(hmax, height[t]);
    std::vector<std::vector<std::int32_t>> by_height(static_cast<std::size_t>(hmax) + 1);
    for (std::int32_t t = 0; t < nn; ++t)
        if (!in_part[t]) by_height[height[t]].push_back(t);
    for (const auto& nodes : by_height) {
        const std::int64_t count = static_cast<std::int64_t>(nodes.size());
        if (count == 0) continue;
        if (count >= 2 && threads > 1 && count * 2 >= threads) {
#pragma omp parallel for schedule(dynamic, 1)
            for (std::int64_t i = 0; i < count; ++i) {
                if (!one(nodes[static_cast<std::size_t>(i)])) {
#pragma omp critical(cost_frontal_lu_failed)
                    failed = true;
                }
            }
        } else {
            for (const std::int32_t node : nodes)  // few, large fronts: parallel dense kernels
                if (!one(node)) failed = true;
        }
        if (failed) return false;
    }
    if (profile) {
        const double t_top = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[lu]  %lld parts %.1f ms, fronts above %.1f ms%c", static_cast<long long>(n_roots),
                     t_parts, t_top, 10);
    }
    return true;
}

void FrontalLU::solve(double* x, bool transposed) const {
    const FlushDenormals ftz;
    const circuit::FrontalSymbolic& s = *sym_;
    std::int64_t max_upd = 1;
    for (const auto& node : s.nodes) max_upd = std::max<std::int64_t>(max_upd, node.nupd);
    std::vector<double> tmp(static_cast<std::size_t>(max_upd));
    const std::size_t nn = s.nodes.size();
    // forward: L (A x = b) or U' (A' x = b), node by node
    for (std::size_t t = 0; t < nn; ++t) {
        const auto& node = s.nodes[t];
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        if (np == 0) continue;
        const ConstBlock L(l_.data() + lpos_[t], m, np, Eigen::OuterStride<>(m));
        const ConstBlock U12(u_.data() + upos_[t], np, nu, Eigen::OuterStride<>(std::max<std::int64_t>(np, 1)));
        Eigen::Map<Eigen::VectorXd> xp(x + node.first, np);
        Eigen::Map<Eigen::VectorXd> tv(tmp.data(), nu);
        if (!transposed) {
            L.topRows(np).triangularView<Eigen::UnitLower>().solveInPlace(xp);
            if (nu > 0) tv.noalias() = L.bottomRows(nu) * xp;
        } else {
            L.topRows(np).triangularView<Eigen::Upper>().transpose().solveInPlace(xp);
            if (nu > 0) tv.noalias() = U12.transpose() * xp;
        }
        const std::int32_t* rows = s.rows.data() + node.upd;
        for (std::int64_t i = 0; i < nu; ++i) x[rows[i]] -= tmp[static_cast<std::size_t>(i)];
    }
    // backward: U (A x = b) or L' (A' x = b), nodes in reverse
    for (std::size_t tt = nn; tt-- > 0;) {
        const auto& node = s.nodes[tt];
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        if (np == 0) continue;
        const ConstBlock L(l_.data() + lpos_[tt], m, np, Eigen::OuterStride<>(m));
        const ConstBlock U12(u_.data() + upos_[tt], np, nu, Eigen::OuterStride<>(std::max<std::int64_t>(np, 1)));
        Eigen::Map<Eigen::VectorXd> xp(x + node.first, np);
        Eigen::Map<Eigen::VectorXd> tv(tmp.data(), nu);
        const std::int32_t* rows = s.rows.data() + node.upd;
        for (std::int64_t i = 0; i < nu; ++i) tmp[static_cast<std::size_t>(i)] = x[rows[i]];
        if (!transposed) {
            if (nu > 0) xp.noalias() -= U12 * tv;
            L.topRows(np).triangularView<Eigen::Upper>().solveInPlace(xp);
        } else {
            if (nu > 0) xp.noalias() -= L.bottomRows(nu).transpose() * tv;
            L.topRows(np).triangularView<Eigen::UnitLower>().transpose().solveInPlace(xp);
        }
    }
}

}  // namespace cost
