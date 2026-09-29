// SPDX-License-Identifier: GPL-3.0-or-later
#include "cost/rsp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include "circuit/system.hpp"
#include "lsm/rmath.hpp"

namespace cost {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Prints the duration of each stage to stderr when LANDSCHAFT_PROFILE is set.
class StageTimer {
public:
    StageTimer() : enabled_(std::getenv("LANDSCHAFT_PROFILE") != nullptr), last_(Clock::now()) {}
    void operator()(const char* stage) {
        if (!enabled_) return;
        const auto now = Clock::now();
        std::fprintf(stderr, "[rsp] %-22s %9.1f ms\n", stage,
                     std::chrono::duration<double, std::milli>(now - last_).count());
        last_ = now;
    }

private:
    using Clock = std::chrono::steady_clock;
    bool enabled_;
    Clock::time_point last_;
};

// The graph of the walk: nodes (cells), their outgoing connections in
// increasing order (CSR), T and the reference walk's W on every connection.
struct RspGraph {
    std::vector<i64> cell;
    std::vector<std::int32_t> node_of;
    std::vector<std::int64_t> ptr;
    std::vector<std::int32_t> adj;
    std::vector<std::int8_t> dir;  // direction index (asymmetric layout) of every connection
    std::vector<double> t, w, k;   // conductance, W, and s_k W_kl (symmetric scaling)
    std::vector<double> s;         // row scale making s_k W_kl symmetric
    bool symmetric = false;
    // I - W scaled is symmetric positive definite: a symmetric transition with
    // method 1 (rows of W sum to at most 1; method 2's P^theta can exceed 1)
    bool spd = false;
};

RspGraph rsp_graph(const Layout& t, const RspOptions& o, const std::vector<i64>& extra) {
    const Grid& g = *t.grid;
    const i64 ncell = g.ncell();
    const auto off = offsets(t.directions, t.symmetric);
    const auto all = offsets(t.directions, false);
    const int nd = static_cast<int>(off.size());
    auto index_of = [&](Offset x) {
        for (int k = 0; k < static_cast<int>(all.size()); ++k)
            if (all[k].dr == x.dr && all[k].dc == x.dc) return k;
        return -1;
    };
    // direction index (asymmetric layout) of each stored direction and of its reverse
    std::vector<int> fwd(static_cast<std::size_t>(nd)), bwd(static_cast<std::size_t>(nd));
    for (int k = 0; k < nd; ++k) {
        fwd[k] = index_of(off[k]);
        bwd[k] = index_of(Offset{-off[k].dr, -off[k].dc});
    }
    // outgoing connections of a cell, ascending
    auto out = [&](i64 a, i64* b, double* v, std::int8_t* d) {
        const i64 r = a / g.nx, c = a % g.nx;
        int m = 0;
        auto add = [&](i64 nb, double x, int dd) {
            int p = m++;
            while (p > 0 && b[p - 1] > nb) {
                b[p] = b[p - 1];
                v[p] = v[p - 1];
                d[p] = d[p - 1];
                --p;
            }
            b[p] = nb;
            v[p] = x;
            d[p] = static_cast<std::int8_t>(dd);
        };
        for (int k = 0; k < nd; ++k) {
            const double x = t.values[k * ncell + a];
            if (x != 0) {
                const i64 nb = g.neighbour(r, c, off[k]);
                if (nb >= 0) add(nb, x, fwd[k]);
            }
            if (t.symmetric) {
                const i64 prev = g.neighbour(r, c, Offset{-off[k].dr, -off[k].dc});
                if (prev >= 0) {
                    const double y = t.values[k * ncell + prev];
                    if (y != 0) add(prev, y, bwd[k]);
                }
            }
        }
        return m;
    };
    // a connection into the cell (asymmetric layouts; symmetric ones: the same as out)
    auto has_in = [&](i64 a) {
        const i64 r = a / g.nx, c = a % g.nx;
        for (int k = 0; k < nd; ++k) {
            const i64 prev = g.neighbour(r, c, Offset{-off[k].dr, -off[k].dc});
            if (prev >= 0 && t.values[k * ncell + prev] != 0) return true;
        }
        return false;
    };
    RspGraph G;
    G.symmetric = t.symmetric;
    G.spd = t.symmetric && o.method == 1;
    std::vector<char> keep(static_cast<std::size_t>(ncell), 0);
#pragma omp parallel for schedule(static)
    for (i64 a = 0; a < ncell; ++a) {
        i64 b[16];
        double v[16];
        std::int8_t d[16];
        const int m = out(a, b, v, d);
        double s = 0;
        for (int q = 0; q < m; ++q) s += v[q];
        if (o.solidify) {
            keep[a] = s / static_cast<double>(ncell) > 1e-300;  // rowMeans > 1e-300
        } else {
            keep[a] = m > 0 || has_in(a);
        }
    }
    if (!o.solidify)
        for (const i64 c : extra)
            if (c >= 0 && c < ncell) keep[c] = 1;
    G.node_of.assign(static_cast<std::size_t>(ncell), -1);
    for (i64 a = 0; a < ncell; ++a)
        if (keep[a]) {
            G.node_of[a] = static_cast<std::int32_t>(G.cell.size());
            G.cell.push_back(a);
        }
    const std::int64_t n = static_cast<std::int64_t>(G.cell.size());
    G.ptr.assign(static_cast<std::size_t>(n + 1), 0);
    // connections to kept cells, per node
#pragma omp parallel for schedule(static)
    for (std::int64_t u = 0; u < n; ++u) {
        i64 b[16];
        double v[16];
        std::int8_t d[16];
        const int m = out(G.cell[u], b, v, d);
        std::int64_t c = 0;
        for (int q = 0; q < m; ++q) c += G.node_of[b[q]] >= 0;
        G.ptr[u + 1] = c;
    }
    for (std::int64_t u = 0; u < n; ++u) G.ptr[u + 1] += G.ptr[u];
    const std::size_t ne = static_cast<std::size_t>(G.ptr[n]);
    G.adj.resize(ne);
    G.t.resize(ne);
    G.dir.resize(ne);
    G.w.resize(ne);
    G.k.resize(ne);
    G.s.resize(static_cast<std::size_t>(n));
    const double theta = o.theta;
#pragma omp parallel for schedule(static)
    for (std::int64_t u = 0; u < n; ++u) {
        i64 b[16];
        double v[16];
        std::int8_t d[16];
        const int m = out(G.cell[u], b, v, d);
        std::int64_t e = G.ptr[u];
        double s = 0;  // rowSums of the (solidified) matrix, ascending
        for (int q = 0; q < m; ++q) {
            const std::int32_t nb = G.node_of[b[q]];
            if (nb < 0) continue;
            G.adj[e] = nb;
            G.t[e] = v[q];
            G.dir[e] = d[q];
            s += v[q];
            ++e;
        }
        const double rinv = s > 0 ? 1.0 / s : 0.0;  // rs[rs > 0] <- 1 / rs
        G.s[u] = o.method == 1 ? s : lsm::r_pow(s, theta);
        for (std::int64_t f = G.ptr[u]; f < G.ptr[u + 1]; ++f) {
            const double tv = G.t[f];
            const double p = tv * rinv;
            if (o.method == 1) {
                const double ex = lsm::r_exp(-theta * (1.0 / tv));
                G.w[f] = ex * p;
                G.k[f] = ex * tv;
            } else {
                G.w[f] = lsm::r_exp(-theta * -lsm::r_log(p));
                G.k[f] = lsm::r_exp(theta * lsm::r_log(tv));
            }
        }
    }
    return G;
}

// A = I - Ij W over the nodes, the rows with Ij = 0 fixed (identity), solved
// for A x = b and A' y = e.
class RspSystem {
public:
    RspSystem(const RspGraph& g, const std::vector<double>& ij, bool geometric, i64 nx)
        : g_(g), ij_(ij), n_(static_cast<std::int32_t>(g.cell.size())) {
        local_.assign(static_cast<std::size_t>(n_), -1);
        for (std::int32_t u = 0; u < n_; ++u)
            if (ij_[u] != 0) {
                local_[u] = static_cast<std::int32_t>(rows_.size());
                rows_.push_back(u);
            }
        if (g.spd && factor_cholesky(geometric, nx)) return;
        chol_.reset();
        factor_lu(geometric, nx);
    }

    // A x = b (all nodes).
    void solve(const double* b, double* x) const {
        const bool scaled = chol_ != nullptr;
        const std::size_t m = rows_.size();
        for (std::int32_t u = 0; u < n_; ++u)
            if (ij_[u] == 0) x[u] = b[u];
        std::vector<double> rhs(m);
        for (std::size_t i = 0; i < m; ++i) {
            const std::int32_t u = rows_[i];
            double f = 0;  // the fixed nodes' values moved to the right-hand side
            for (std::int64_t e = g_.ptr[u]; e < g_.ptr[u + 1]; ++e) {
                const std::int32_t l = g_.adj[e];
                if (ij_[l] == 0) f += (scaled ? g_.k[e] : ij_[u] * g_.w[e]) * b[l];
            }
            rhs[i] = scaled ? g_.s[u] / ij_[u] * b[u] + f : b[u] + f;
        }
        std::vector<double> y = solve_rows(rhs, false);
        for (std::size_t i = 0; i < m; ++i) x[rows_[i]] = y[i];
    }

    // A' y = e (all nodes).
    void solve_t(const double* e, double* y) const {
        const bool scaled = chol_ != nullptr;
        const std::size_t m = rows_.size();
        std::vector<double> rhs(m);
        for (std::size_t i = 0; i < m; ++i) rhs[i] = e[rows_[i]];
        std::vector<double> z = solve_rows(rhs, true);
        for (std::int32_t u = 0; u < n_; ++u) y[u] = ij_[u] == 0 ? e[u] : 0.0;
        for (std::size_t i = 0; i < m; ++i) {
            const std::int32_t u = rows_[i];
            y[u] = scaled ? g_.s[u] / ij_[u] * z[i] : z[i];
        }
        // y_F = e_F + sum_r Ij_r W_rf y_r
        for (std::size_t i = 0; i < m; ++i) {
            const std::int32_t u = rows_[i];
            for (std::int64_t k = g_.ptr[u]; k < g_.ptr[u + 1]; ++k) {
                const std::int32_t l = g_.adj[k];
                if (ij_[l] == 0) y[l] += ij_[u] * g_.w[k] * y[u];
            }
        }
    }

private:
    // The rows scaled: M = diag(s / Ij) - K, symmetric; false when M is not
    // positive definite.
    bool factor_cholesky(bool geometric, i64 nx) {
        const RspGraph& g = g_;
        const std::size_t m = rows_.size();
        i64 x0 = std::numeric_limits<i64>::max(), x1 = -1, y0 = std::numeric_limits<i64>::max(), y1 = -1;
        if (geometric)
            for (const std::int32_t u : rows_) {
                const i64 c = g.cell[u];
                x0 = std::min(x0, c % nx);
                x1 = std::max(x1, c % nx);
                y0 = std::min(y0, c / nx);
                y1 = std::max(y1, c / nx);
            }
        const i64 bx = geometric && x1 >= x0 ? x1 - x0 + 1 : 1, by = geometric && y1 >= y0 ? y1 - y0 + 1 : 1;
        std::vector<i64> cell(m, -1);
        std::vector<std::int64_t> ptr(m + 1, 0);
        std::vector<std::int32_t> adj;
        std::vector<double> off, diag(m);
        for (std::size_t i = 0; i < m; ++i) {
            const std::int32_t u = rows_[i];
            if (geometric) cell[i] = (g.cell[u] / nx - y0) * bx + (g.cell[u] % nx - x0);
            for (std::int64_t e = g.ptr[u]; e < g.ptr[u + 1]; ++e) {
                const std::int32_t l = local_[g.adj[e]];
                if (l < 0) continue;
                adj.push_back(l);
                off.push_back(-g.k[e]);
            }
            ptr[i + 1] = static_cast<std::int64_t>(adj.size());
            diag[i] = g.s[u] / ij_[u];
        }
        StageTimer timer;
        chol_ = std::make_unique<circuit::CellSystem>();
        chol_->share(circuit::frontal_analyze_graph(bx, by, cell, ptr, adj));
        timer("  analyse");
        chol_->set_values(diag.data(), off.data());
        if (!chol_->factorize()) return false;
        timer("  factorise (numeric)");
        const auto& perm = chol_->perm();
        iperm_.assign(m, -1);
        for (std::size_t q = 0; q < m; ++q) iperm_[static_cast<std::size_t>(perm[q])] = static_cast<std::int32_t>(q);
        return true;
    }

    // Sparse LU of A_RR (asymmetric transitions, or method 2), its rows and
    // columns in the nested-dissection order of the (symmetrised) graph, which
    // fills far less than Eigen's COLAMD on grids.
    void factor_lu(bool geometric, i64 nx) {
        const RspGraph& g = g_;
        const std::size_t m = rows_.size();
        // the symmetrised pattern, for the ordering
        std::vector<std::vector<std::int32_t>> nb(m);
        for (std::size_t i = 0; i < m; ++i) {
            const std::int32_t u = rows_[i];
            for (std::int64_t e = g.ptr[u]; e < g.ptr[u + 1]; ++e) {
                const std::int32_t l = local_[g.adj[e]];
                if (l < 0) continue;
                nb[i].push_back(l);
                nb[static_cast<std::size_t>(l)].push_back(static_cast<std::int32_t>(i));
            }
        }
        std::vector<std::int64_t> ptr(m + 1, 0);
        std::vector<std::int32_t> adj;
        for (std::size_t i = 0; i < m; ++i) {
            std::sort(nb[i].begin(), nb[i].end());
            nb[i].erase(std::unique(nb[i].begin(), nb[i].end()), nb[i].end());
            adj.insert(adj.end(), nb[i].begin(), nb[i].end());
            ptr[i + 1] = static_cast<std::int64_t>(adj.size());
        }
        nb.clear();
        std::vector<i64> cell(m, -1);
        i64 bx = 1, by = 1;
        if (geometric) {
            i64 x0 = std::numeric_limits<i64>::max(), x1 = -1, y0 = std::numeric_limits<i64>::max(), y1 = -1;
            for (const std::int32_t u : rows_) {
                const i64 c = g.cell[u];
                x0 = std::min(x0, c % nx);
                x1 = std::max(x1, c % nx);
                y0 = std::min(y0, c / nx);
                y1 = std::max(y1, c / nx);
            }
            bx = x1 - x0 + 1;
            by = y1 - y0 + 1;
            for (std::size_t i = 0; i < m; ++i) cell[i] = (g.cell[rows_[i]] / nx - y0) * bx + (g.cell[rows_[i]] % nx - x0);
        }
        const auto sym = circuit::frontal_analyze_graph(bx, by, cell, ptr, adj);
        order_.assign(sym->perm.begin(), sym->perm.end());  // new position -> row
        std::vector<int> pos(m);
        for (std::size_t q = 0; q < m; ++q) pos[static_cast<std::size_t>(order_[q])] = static_cast<int>(q);
        std::vector<Eigen::Triplet<double>> tri;
        tri.reserve(adj.size() + m);
        for (std::size_t i = 0; i < m; ++i) {
            const std::int32_t u = rows_[i];
            tri.emplace_back(pos[i], pos[i], 1.0);
            for (std::int64_t e = g.ptr[u]; e < g.ptr[u + 1]; ++e) {
                const std::int32_t l = local_[g.adj[e]];
                if (l >= 0) tri.emplace_back(pos[i], pos[static_cast<std::size_t>(l)], -ij_[u] * g.w[e]);
            }
        }
        Eigen::SparseMatrix<double> a(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(m));
        a.setFromTriplets(tri.begin(), tri.end());
        a.makeCompressed();
        lu_ = std::make_unique<Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::NaturalOrdering<int>>>();
        lu_->compute(a);
        if (lu_->info() != Eigen::Success) throw std::runtime_error("I - W is singular");
    }

    // M^-1 rhs (Cholesky) or A_RR^-1 rhs / A_RR'^-1 rhs (LU).
    std::vector<double> solve_rows(const std::vector<double>& rhs, bool transposed) const {
        const std::size_t m = rows_.size();
        std::vector<double> out(m);
        if (m == 0) return out;
        if (chol_) {
            std::vector<double> b(m), x(m);
            for (std::size_t i = 0; i < m; ++i) b[static_cast<std::size_t>(iperm_[i])] = rhs[i];
            chol_->solve(b.data(), x.data());
            for (std::size_t i = 0; i < m; ++i) out[i] = x[static_cast<std::size_t>(iperm_[i])];
            return out;
        }
        Eigen::VectorXd b(static_cast<Eigen::Index>(m));
        for (std::size_t q = 0; q < m; ++q) b[static_cast<Eigen::Index>(q)] = rhs[static_cast<std::size_t>(order_[q])];
        Eigen::VectorXd x = transposed ? Eigen::VectorXd(lu_->transpose().solve(b)) : Eigen::VectorXd(lu_->solve(b));
        for (std::size_t q = 0; q < m; ++q) out[static_cast<std::size_t>(order_[q])] = x[static_cast<Eigen::Index>(q)];
        return out;
    }

    const RspGraph& g_;
    std::vector<double> ij_;
    std::int32_t n_;
    std::vector<std::int32_t> rows_, local_, iperm_;
    std::unique_ptr<circuit::CellSystem> chol_;
    std::vector<std::int32_t> order_;  // LU: position -> row
    std::unique_ptr<Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::NaturalOrdering<int>>> lu_;
};

std::vector<std::int32_t> nodes_of(const RspGraph& g, const std::vector<i64>& cells) {
    std::vector<std::int32_t> out;
    for (const i64 c : cells) out.push_back(c >= 0 && c < static_cast<i64>(g.node_of.size()) ? g.node_of[c] : -1);
    return out;
}

bool geometric_of(const Layout& t) { return t.directions != 16 && !t.grid->wrap; }

}  // namespace

std::vector<double> rsp_passages(const Layout& t, const std::vector<i64>& origins, const std::vector<i64>& goals,
                                 const RspOptions& o, bool edges) {
    StageTimer timer;
    const RspGraph g = rsp_graph(t, o, {});
    timer("graph");
    const i64 ncell = t.grid->ncell();
    const auto ci = nodes_of(g, origins), cj = nodes_of(g, goals);
    for (const std::int32_t v : ci)
        if (v < 0) throw std::invalid_argument("an origin is not on the transition's graph");
    for (const std::int32_t v : cj)
        if (v < 0) throw std::invalid_argument("a goal is not on the transition's graph");
    const std::int32_t n = static_cast<std::int32_t>(g.cell.size());
    std::vector<double> ij(static_cast<std::size_t>(n), 1.0);
    for (const std::int32_t v : cj) ij[v] = 1.0 - 1.0 / static_cast<double>(cj.size());
    const RspSystem sys(g, ij, geometric_of(t), t.grid->nx);
    timer("factorise");
    std::vector<double> ei(static_cast<std::size_t>(n), 0.0), ej(static_cast<std::size_t>(n), 0.0);
    for (const std::int32_t v : ci) ei[v] = 1.0 / static_cast<double>(ci.size());
    for (const std::int32_t v : cj) ej[v] = 1.0 / static_cast<double>(cj.size());
    std::vector<double> zci(static_cast<std::size_t>(n)), zcj(static_cast<std::size_t>(n));
    sys.solve_t(ei.data(), zci.data());
    sys.solve(ej.data(), zcj.data());
    timer("solves");
    double zcij = 0;
    for (std::int32_t u = 0; u < n; ++u) zcij += ei[u] * zcj[u];
    const std::size_t nd = offsets(t.directions, false).size();
    if (edges) {
        std::vector<double> out(nd * static_cast<std::size_t>(ncell), 0.0);
        if (zcij < 1e-300) return out;
        // N_kl = zci_k Wj_kl zcj_l / zcij; net: max(N_kl - N_lk, 0)
        std::vector<double> nkl(g.t.size());
        for (std::int32_t u = 0; u < n; ++u)
            for (std::int64_t e = g.ptr[u]; e < g.ptr[u + 1]; ++e)
                nkl[e] = zci[u] * (ij[u] * g.w[e]) * zcj[g.adj[e]] / zcij;
        for (std::int32_t u = 0; u < n; ++u) {
            for (std::int64_t e = g.ptr[u]; e < g.ptr[u + 1]; ++e) {
                double val = nkl[e];
                if (o.net) {
                    const std::int32_t l = g.adj[e];
                    double back = 0;
                    for (std::int64_t f = g.ptr[l]; f < g.ptr[l + 1]; ++f)
                        if (g.adj[f] == u) back = nkl[f];
                    val = std::max(val - back, 0.0);
                }
                out[static_cast<std::size_t>(g.dir[e]) * static_cast<std::size_t>(ncell) +
                    static_cast<std::size_t>(g.cell[u])] = val;
            }
        }
        return out;
    }
    std::vector<double> out(static_cast<std::size_t>(ncell), kNaN);
    if (zcij < 1e-300) {
        std::fill(out.begin(), out.end(), 0.0);
        return out;
    }
    std::vector<double> nkl(g.t.size());
    for (std::int32_t u = 0; u < n; ++u)
        for (std::int64_t e = g.ptr[u]; e < g.ptr[u + 1]; ++e)
            nkl[e] = zci[u] * (ij[u] * g.w[e]) * zcj[g.adj[e]] / zcij;
    std::vector<double> rows(static_cast<std::size_t>(n), 0.0), cols(static_cast<std::size_t>(n), 0.0);
    for (std::int32_t u = 0; u < n; ++u) {
        for (std::int64_t e = g.ptr[u]; e < g.ptr[u + 1]; ++e) {
            const std::int32_t l = g.adj[e];
            double val = nkl[e];
            if (o.net) {  // |N - N'| / 2: each connection pair once from each side
                double back = 0;
                for (std::int64_t f = g.ptr[l]; f < g.ptr[l + 1]; ++f)
                    if (g.adj[f] == u) back = nkl[f];
                val = std::fabs(val - back) / 2;
                rows[u] += val;
                // a connection without its reverse: the reverse entry of |N - N'|
                bool has_back = false;
                for (std::int64_t f = g.ptr[l]; f < g.ptr[l + 1]; ++f)
                    if (g.adj[f] == u) has_back = true;
                if (!has_back) rows[l] += val;
            } else {
                rows[u] += val;
                cols[l] += val;
            }
        }
    }
    for (std::int32_t u = 0; u < n; ++u) out[static_cast<std::size_t>(g.cell[u])] = o.net ? rows[u] : std::max(rows[u], cols[u]);
    if (o.net) {  // n[c(ci, cj)] <- 2 * n[c(ci, cj)]: once per cell
        std::vector<std::int32_t> ends(ci);
        ends.insert(ends.end(), cj.begin(), cj.end());
        for (const std::int32_t v : ends) out[static_cast<std::size_t>(g.cell[v])] = 2 * rows[v];
    }
    return out;
}

std::vector<double> rsp_distances(const Layout& t, const std::vector<i64>& from, const std::vector<i64>& to,
                                  const RspOptions& o) {
    std::vector<i64> extra(from);
    extra.insert(extra.end(), to.begin(), to.end());
    const RspGraph g = rsp_graph(t, o, extra);
    const auto ci = nodes_of(g, from), cj = nodes_of(g, to);
    const std::int32_t n = static_cast<std::int32_t>(g.cell.size());
    const std::size_t nf = ci.size(), nt = cj.size();
    std::vector<double> out(nf * nt, kNaN);
    // reverse connection of every connection (net passages)
    std::vector<std::int64_t> rev(g.t.size(), -1);
    for (std::int32_t u = 0; u < n; ++u)
        for (std::int64_t e = g.ptr[u]; e < g.ptr[u + 1]; ++e) {
            const std::int32_t l = g.adj[e];
            for (std::int64_t f = g.ptr[l]; f < g.ptr[l + 1]; ++f)
                if (g.adj[f] == u) rev[e] = f;
        }
    const bool geometric = geometric_of(t);
    int threads = 1;
#if defined(_OPENMP)
    threads = std::max(1, std::min(omp_get_max_threads(), static_cast<int>(nt)));
#endif
    bool failed = false;
#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (std::int64_t jj = 0; jj < static_cast<std::int64_t>(nt); ++jj) {
        const std::int32_t j = cj[static_cast<std::size_t>(jj)];
        if (j < 0) continue;
        try {
            std::vector<double> ij(static_cast<std::size_t>(n), 1.0);
            ij[j] = 0.0;  // the goal absorbs
            const RspSystem sys(g, ij, geometric, t.grid->nx);
            std::vector<double> ej(static_cast<std::size_t>(n), 0.0), zcj(static_cast<std::size_t>(n));
            ej[j] = 1.0;
            sys.solve(ej.data(), zcj.data());
            std::vector<double> ei(static_cast<std::size_t>(n), 0.0), zci(static_cast<std::size_t>(n)),
                nkl(g.t.size());
            for (std::size_t ii = 0; ii < nf; ++ii) {
                const std::int32_t i = ci[ii];
                if (i < 0) continue;
                ei[i] = 1.0;
                sys.solve_t(ei.data(), zci.data());
                ei[i] = 0.0;
                const double zcij = zcj[i];
                for (std::int32_t u = 0; u < n; ++u)
                    for (std::int64_t e = g.ptr[u]; e < g.ptr[u + 1]; ++e)
                        nkl[e] = zci[u] * (ij[u] * g.w[e]) * zcj[g.adj[e]] / zcij;
                double dsum = 0;
                for (std::size_t e = 0; e < nkl.size(); ++e) {
                    double val = nkl[e];
                    if (o.net) val = std::max(val - (rev[e] >= 0 ? nkl[rev[e]] : 0.0), 0.0);
                    dsum += (1.0 / g.t[e]) * val;
                }
                out[ii * nt + static_cast<std::size_t>(jj)] = dsum;
            }
        } catch (...) {
#pragma omp critical(cost_rsp_failed)
            failed = true;
        }
    }
    if (failed) throw std::runtime_error("I - W is singular for some goal (theta = 0 on a graph in several parts?)");
    return out;
}

}  // namespace cost
