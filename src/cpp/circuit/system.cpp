// SPDX-License-Identifier: GPL-3.0-or-later

#include "circuit/system.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <utility>

namespace circuit {

namespace {

constexpr i64 kLeafCells = 16;

using Block = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>;
using ConstBlock = Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>>;

struct TreeBuilder {
    const std::uint8_t* mark;
    i64 stride;
    std::vector<i64> cells;
    std::vector<FrontalSymbolic::Node> nodes;
    std::vector<std::vector<std::int32_t>> kids;

    std::int32_t add(std::int32_t first, std::vector<std::int32_t> ch) {
        FrontalSymbolic::Node n;
        n.first = first;
        n.npiv = static_cast<std::int32_t>(cells.size()) - first;
        nodes.push_back(n);
        kids.push_back(std::move(ch));
        return static_cast<std::int32_t>(nodes.size()) - 1;
    }

    // Shrinks the box to its marked cells, cuts it by its middle row or column
    // (a single line separates 4- and 8-neighbour grids) and orders both halves
    // before the separator; small boxes are leaves in row order. Returns the
    // subtree's root, or -1 for an empty box.
    std::int32_t build(i64 r0, i64 r1, i64 c0, i64 c1) {
        i64 top = r1, bottom = r0 - 1, left = c1, right = c0 - 1;
        for (i64 r = r0; r < r1; ++r)
            for (i64 c = c0; c < c1; ++c)
                if (mark[r * stride + c]) {
                    top = std::min(top, r);
                    bottom = std::max(bottom, r);
                    left = std::min(left, c);
                    right = std::max(right, c);
                }
        if (bottom < top) return -1;
        r0 = top;
        r1 = bottom + 1;
        c0 = left;
        c1 = right + 1;
        const i64 h = r1 - r0, w = c1 - c0;
        if (h * w <= kLeafCells) {
            const std::int32_t first = static_cast<std::int32_t>(cells.size());
            for (i64 r = r0; r < r1; ++r)
                for (i64 c = c0; c < c1; ++c)
                    if (mark[r * stride + c]) cells.push_back(r * stride + c);
            return add(first, {});
        }
        std::vector<std::int32_t> ch;
        const std::int32_t first_sep = [&] {
            if (h >= w) {
                const i64 m = r0 + h / 2;
                const std::int32_t a = build(r0, m, c0, c1);
                const std::int32_t b = build(m + 1, r1, c0, c1);
                if (a >= 0) ch.push_back(a);
                if (b >= 0) ch.push_back(b);
                const std::int32_t f = static_cast<std::int32_t>(cells.size());
                for (i64 c = c0; c < c1; ++c)
                    if (mark[m * stride + c]) cells.push_back(m * stride + c);
                return f;
            }
            const i64 m = c0 + w / 2;
            const std::int32_t a = build(r0, r1, c0, m);
            const std::int32_t b = build(r0, r1, m + 1, c1);
            if (a >= 0) ch.push_back(a);
            if (b >= 0) ch.push_back(b);
            const std::int32_t f = static_cast<std::int32_t>(cells.size());
            for (i64 r = r0; r < r1; ++r)
                if (mark[r * stride + m]) cells.push_back(r * stride + m);
            return f;
        }();
        if (static_cast<std::int32_t>(cells.size()) == first_sep && ch.size() == 1) return ch[0];
        return add(first_sep, std::move(ch));
    }
};

}  // namespace

std::shared_ptr<const FrontalSymbolic> frontal_analyze(i64 bx, i64 by, const std::uint8_t* unknown,
                                                       bool four_neighbors) {
    auto sym = std::make_shared<FrontalSymbolic>();
    FrontalSymbolic& s = *sym;
    s.bx = bx;
    s.by = by;
    s.four = four_neighbors;
    TreeBuilder tb{unknown, bx, {}, {}, {}};
    tb.build(0, by, 0, bx);
    s.cells = std::move(tb.cells);
    s.nodes = std::move(tb.nodes);
    const i64 n = static_cast<i64>(s.cells.size());
    s.unknown_of.assign(static_cast<std::size_t>(bx * by), -1);
    for (i64 k = 0; k < n; ++k) s.unknown_of[static_cast<std::size_t>(s.cells[k])] = static_cast<std::int32_t>(k);

    const Offset* nb = four_neighbors ? kNeighbours4 : kNeighbours8;
    const int nd = four_neighbors ? 4 : 8;
    s.nbr.assign(static_cast<std::size_t>(n * nd), -1);
    for (i64 k = 0; k < n; ++k) {
        const i64 y = s.cells[k] / bx, x = s.cells[k] % bx;
        for (int d = 0; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            if (yy < 0 || yy >= by || xx < 0 || xx >= bx) continue;
            s.nbr[static_cast<std::size_t>(k * nd + d)] = s.unknown_of[static_cast<std::size_t>(yy * bx + xx)];
        }
    }

    // Front structures, bottom up.
    s.amap.assign(static_cast<std::size_t>(n * nd), -1);
    std::vector<std::int32_t> stamp(static_cast<std::size_t>(n), -1), loc(static_cast<std::size_t>(n), -1);
    std::vector<std::int32_t> upd;
    std::int64_t lpos = 0, stack = 0, peak = 0;
    const std::int32_t n_nodes = static_cast<std::int32_t>(s.nodes.size());
    for (std::int32_t t = 0; t < n_nodes; ++t) {
        FrontalSymbolic::Node& node = s.nodes[t];
        node.child_begin = static_cast<std::int32_t>(s.children.size());
        for (std::int32_t c : tb.kids[t]) s.children.push_back(c);
        node.child_end = static_cast<std::int32_t>(s.children.size());
        const std::int32_t end = node.first + node.npiv;
        upd.clear();
        for (std::int32_t p = node.first; p < end; ++p)
            for (int d = 0; d < nd; ++d) {
                const std::int32_t q = s.nbr[static_cast<std::size_t>(p) * nd + d];
                if (q >= end && stamp[q] != t) {
                    stamp[q] = t;
                    upd.push_back(q);
                }
            }
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const FrontalSymbolic::Node& c = s.nodes[s.children[ci]];
            for (std::int32_t i = 0; i < c.nupd; ++i) {
                const std::int32_t r = s.rows[static_cast<std::size_t>(c.upd + i)];
                if (r >= end && stamp[r] != t) {
                    stamp[r] = t;
                    upd.push_back(r);
                }
            }
        }
        std::sort(upd.begin(), upd.end());
        node.upd = static_cast<std::int32_t>(s.rows.size());
        node.nupd = static_cast<std::int32_t>(upd.size());
        s.rows.insert(s.rows.end(), upd.begin(), upd.end());
        s.relmap.resize(s.rows.size(), -1);
        for (std::int32_t p = node.first; p < end; ++p) loc[p] = p - node.first;
        for (std::int32_t i = 0; i < node.nupd; ++i) loc[upd[i]] = node.npiv + i;
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const FrontalSymbolic::Node& c = s.nodes[s.children[ci]];
            for (std::int32_t i = 0; i < c.nupd; ++i)
                s.relmap[static_cast<std::size_t>(c.upd + i)] = loc[s.rows[static_cast<std::size_t>(c.upd + i)]];
        }
        for (std::int32_t p = node.first; p < end; ++p)
            for (int d = 0; d < nd; ++d) {
                const std::int32_t q = s.nbr[static_cast<std::size_t>(p) * nd + d];
                if (q > p) s.amap[static_cast<std::size_t>(p) * nd + d] = loc[q];
            }
        const std::int64_t m = node.npiv + node.nupd;
        node.lpos = lpos;
        lpos += m * node.npiv;
        s.max_front = std::max(s.max_front, m * m);
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const std::int64_t u = s.nodes[s.children[ci]].nupd;
            stack -= u * u;
        }
        stack += static_cast<std::int64_t>(node.nupd) * node.nupd;
        peak = std::max(peak, stack);
    }
    s.lsize = lpos;
    s.stack_size = peak;
    return sym;
}

void CellSystem::analyze(i64 bx, i64 by, const std::uint8_t* unknown, bool four_neighbors) {
    share(frontal_analyze(bx, by, unknown, four_neighbors));
}

void CellSystem::share(std::shared_ptr<const FrontalSymbolic> symbolic) {
    sym_ = std::move(symbolic);
    const std::size_t n = sym_->cells.size();
    diag_.assign(n, 1.0);
    off_.assign(n * (sym_->four ? 4 : 8), 0.0);
}

void CellSystem::assemble(const double* g, const std::uint8_t* present, const double* extra_diag, double shift) {
    const FrontalSymbolic& s = *sym_;
    const Offset* nb = s.four ? kNeighbours4 : kNeighbours8;
    const int nd = s.four ? 4 : 8;
    const i64 n = size();
    for (i64 j = 0; j < n; ++j) {
        const i64 c = s.cells[j];
        double* off = off_.data() + j * nd;
        if (!present[c]) {
            std::fill(off, off + nd, 0.0);
            diag_[j] = 1.0;
            continue;
        }
        const i64 y = c / s.bx, x = c % s.bx;
        const double gc = g[c];
        double diag = 0.0;
        for (int d = 0; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            if (yy < 0 || yy >= s.by || xx < 0 || xx >= s.bx || !present[yy * s.bx + xx]) {
                off[d] = 0.0;
                continue;
            }
            const double w = edge_weight(gc, g[yy * s.bx + xx], nb[d].diagonal);
            diag += w;
            off[d] = -w;
        }
        if (extra_diag) diag += extra_diag[c];
        diag_[j] = diag + shift;
    }
}

bool CellSystem::factorize() { return factorize(nullptr); }

bool CellSystem::factorize_solve(const double* b, double* x) {
    const i64 n = size();
    double* y = rhs_.get(static_cast<std::size_t>(n));
    std::copy(b, b + n, y);
    if (!factorize(y)) return false;
    backward(y, 1);
    std::copy(y, y + n, x);
    return true;
}

bool CellSystem::factorize(double* y) {
    const FrontalSymbolic& s = *sym_;
    const int nd = s.four ? 4 : 8;
    double* stack = stack_.get(static_cast<std::size_t>(s.stack_size));
    // Every front is built where its columns of L go: its first npiv columns
    // stay as L, its trailing ones (the update matrix, copied to the stack)
    // spill over the space of the fronts that follow, hence the margin.
    double* lstore = l_.get(static_cast<std::size_t>(s.lsize + s.max_front));
    std::int64_t top = 0;
    for (const FrontalSymbolic::Node& node : s.nodes) {
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        double* F = lstore + node.lpos;
        for (std::int64_t j = 0; j < m; ++j) std::fill(F + j * m + j, F + j * m + m, 0.0);
        for (std::int32_t p = node.first; p < node.first + node.npiv; ++p) {
            const std::int64_t j = p - node.first;
            F[j * m + j] += diag_[p];
            for (int d = 0; d < nd; ++d) {
                const std::int32_t a = s.amap[static_cast<std::size_t>(p) * nd + d];
                if (a >= 0) F[j * m + a] += off_[static_cast<std::size_t>(p) * nd + d];
            }
        }
        // Extend-add the children's update matrices (the top of the stack, in order).
        std::int64_t below = top;
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const std::int64_t u = s.nodes[s.children[ci]].nupd;
            below -= u * u;
        }
        std::int64_t at = below;
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const FrontalSymbolic::Node& c = s.nodes[s.children[ci]];
            const std::int64_t u = c.nupd;
            const std::int32_t* rel = s.relmap.data() + c.upd;
            const double* U = stack + at;
            for (std::int64_t jj = 0; jj < u; ++jj) {
                double* col = F + static_cast<std::int64_t>(rel[jj]) * m;
                const double* ucol = U + jj * u;
                for (std::int64_t ii = jj; ii < u; ++ii) col[rel[ii]] += ucol[ii];
            }
            at += u * u;
        }
        top = below;
        if (np > 0) {
            Block F11(F, np, np, Eigen::OuterStride<>(m));
            Eigen::LLT<Eigen::Ref<Eigen::MatrixXd, 0, Eigen::OuterStride<>>> llt(F11);
            if (llt.info() != Eigen::Success) return false;
            if (nu > 0) {
                Block F21(F + np, nu, np, Eigen::OuterStride<>(m));
                F11.triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheRight>(F21);
                Block F22(F + np * m + np, nu, nu, Eigen::OuterStride<>(m));
                F22.selfadjointView<Eigen::Lower>().rankUpdate(F21, -1.0);
            }
            if (y) {
                // As forward() with one right-hand side.
                const ConstBlock L(F, m, np, Eigen::OuterStride<>(m));
                Eigen::Map<Eigen::VectorXd> yp(y + node.first, np);
                L.topRows(np).triangularView<Eigen::Lower>().solveInPlace(yp);
                if (nu > 0) {
                    double* tmp = tmp_.get(static_cast<std::size_t>(nu));
                    Eigen::Map<Eigen::MatrixXd> t(tmp, nu, 1);
                    t.col(0).noalias() = L.bottomRows(nu) * yp.col(0);
                    const std::int32_t* rows = s.rows.data() + node.upd;
                    for (std::int64_t i = 0; i < nu; ++i) y[rows[i]] -= tmp[i];
                }
            }
        }
        // Push the update matrix (lower triangle, nu × nu).
        double* U = stack + top;
        const double* F22 = F + np * m + np;
        for (std::int64_t j = 0; j < nu; ++j) std::copy(F22 + j * m + j, F22 + j * m + nu, U + j * nu + j);
        top += nu * nu;
    }
    return true;
}

void CellSystem::solve(const double* b, double* x) {
    const std::size_t n = sym_->cells.size();
    if (x != b) std::copy(b, b + n, x);
    solve_many(x, 1);
}

void CellSystem::solve_many(double* xout, i64 k) {
    const i64 n = size();
    // Solved in an aligned copy (see Buffer).
    double* x = rhs_.get(static_cast<std::size_t>(n * k));
    std::copy(xout, xout + n * k, x);
    forward(x, k);
    backward(x, k);
    std::copy(x, x + n * k, xout);
}

namespace {

// The right-hand sides of one node's pivots: a vector for one (the same
// kernels as the forward substitution fused into factorize), a block for more.
template <class F>
void with_pivots(double* x, std::int64_t first, std::int64_t np, i64 n, i64 k, F&& f) {
    if (k == 1) {
        Eigen::Map<Eigen::VectorXd> xp(x + first, np);
        f(xp);
    } else {
        Block xp(x + first, np, k, Eigen::OuterStride<>(n));
        f(xp);
    }
}

}  // namespace

void CellSystem::forward(double* x, i64 k) {
    const FrontalSymbolic& s = *sym_;
    const i64 n = size();
    const double* lstore = l_.p.get();
    std::int64_t max_upd = 0;
    for (const FrontalSymbolic::Node& node : s.nodes) max_upd = std::max<std::int64_t>(max_upd, node.nupd);
    double* tmp = tmp_.get(static_cast<std::size_t>(max_upd * k));
    for (const FrontalSymbolic::Node& node : s.nodes) {
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        if (np == 0) continue;
        const ConstBlock L(lstore + node.lpos, m, np, Eigen::OuterStride<>(m));
        with_pivots(x, node.first, np, n, k, [&](auto& xp) {
            L.topRows(np).template triangularView<Eigen::Lower>().solveInPlace(xp);
            if (nu > 0) {
                Eigen::Map<Eigen::MatrixXd> t(tmp, nu, k);
                if (k == 1) t.col(0).noalias() = L.bottomRows(nu) * xp.col(0);
                else t.noalias() = L.bottomRows(nu) * xp;
                const std::int32_t* rows = s.rows.data() + node.upd;
                for (i64 c = 0; c < k; ++c)
                    for (std::int64_t i = 0; i < nu; ++i) x[c * n + rows[i]] -= t(i, c);
            }
        });
    }
}

void CellSystem::backward(double* x, i64 k) {
    const FrontalSymbolic& s = *sym_;
    const i64 n = size();
    const double* lstore = l_.p.get();
    std::int64_t max_upd = 0;
    for (const FrontalSymbolic::Node& node : s.nodes) max_upd = std::max<std::int64_t>(max_upd, node.nupd);
    double* tmp = tmp_.get(static_cast<std::size_t>(max_upd * k));
    for (auto it = s.nodes.rbegin(); it != s.nodes.rend(); ++it) {
        const FrontalSymbolic::Node& node = *it;
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        if (np == 0) continue;
        const ConstBlock L(lstore + node.lpos, m, np, Eigen::OuterStride<>(m));
        with_pivots(x, node.first, np, n, k, [&](auto& xp) {
            if (nu > 0) {
                Eigen::Map<Eigen::MatrixXd> t(tmp, nu, k);
                const std::int32_t* rows = s.rows.data() + node.upd;
                for (i64 c = 0; c < k; ++c)
                    for (std::int64_t i = 0; i < nu; ++i) t(i, c) = x[c * n + rows[i]];
                if (k == 1) xp.col(0).noalias() -= L.bottomRows(nu).transpose() * t.col(0);
                else xp.noalias() -= L.bottomRows(nu).transpose() * t;
            }
            L.topRows(np).template triangularView<Eigen::Lower>().transpose().solveInPlace(xp);
        });
    }
}

}  // namespace circuit
