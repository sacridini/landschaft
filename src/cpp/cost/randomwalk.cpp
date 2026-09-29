// SPDX-License-Identifier: GPL-3.0-or-later
#include "cost/randomwalk.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "circuit/system.hpp"

namespace cost {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// A component's grounded Laplacian, factorised: its nodes but the ground.
struct ComponentSystem {
    circuit::CellSystem sys;
    std::vector<std::int32_t> nodes;   // local -> graph node (the unknowns, ascending)
    std::vector<std::int32_t> iperm;   // local -> unknown (elimination order)
    std::int32_t ground = -1;
};

void build(const WalkGraph& g, const std::vector<std::int32_t>& members, bool geometric, i64 nx,
           ComponentSystem& cs, std::vector<std::int32_t>& local) {
    cs.ground = members.back();
    cs.nodes.assign(members.begin(), members.end() - 1);
    const std::size_t n = cs.nodes.size();
    for (std::size_t i = 0; i < n; ++i) local[cs.nodes[i]] = static_cast<std::int32_t>(i);
    i64 x0 = std::numeric_limits<i64>::max(), x1 = -1, y0 = std::numeric_limits<i64>::max(), y1 = -1;
    if (geometric) {
        for (const std::int32_t v : cs.nodes) {
            const i64 c = g.cell[v];
            x0 = std::min(x0, c % nx);
            x1 = std::max(x1, c % nx);
            y0 = std::min(y0, c / nx);
            y1 = std::max(y1, c / nx);
        }
    }
    const i64 bx = geometric && x1 >= x0 ? x1 - x0 + 1 : 1, by = geometric && y1 >= y0 ? y1 - y0 + 1 : 1;
    std::vector<i64> cell(n, -1);
    std::vector<std::int64_t> ptr(n + 1, 0);
    std::vector<std::int32_t> adj;
    std::vector<double> off, diag(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t v = cs.nodes[i];
        if (geometric) cell[i] = (g.cell[v] / nx - y0) * bx + (g.cell[v] % nx - x0);
        for (std::int64_t e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
            const std::int32_t u = g.adj[static_cast<std::size_t>(e)];
            if (u == cs.ground) continue;
            adj.push_back(local[u]);
            off.push_back(-g.w[static_cast<std::size_t>(e)]);
        }
        ptr[i + 1] = static_cast<std::int64_t>(adj.size());
        diag[i] = g.degree[v];
    }
    for (const std::int32_t v : cs.nodes) local[v] = -1;
    cs.sys.share(circuit::frontal_analyze_graph(bx, by, cell, ptr, adj));
    cs.sys.set_values(diag.data(), off.data());
    if (!cs.sys.factorize()) throw std::runtime_error("the Laplacian is not positive definite");
    const std::vector<std::int32_t>& perm = cs.sys.perm();
    cs.iperm.assign(n, -1);
    for (std::size_t u = 0; u < n; ++u) cs.iperm[static_cast<std::size_t>(perm[u])] = static_cast<std::int32_t>(u);
}

std::vector<std::vector<std::int32_t>> members_of(const WalkGraph& g) {
    std::vector<std::vector<std::int32_t>> m(static_cast<std::size_t>(g.n_components));
    for (std::size_t v = 0; v < g.cell.size(); ++v) m[static_cast<std::size_t>(g.component[v])].push_back(
        static_cast<std::int32_t>(v));
    return m;
}

int threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace

WalkGraph walk_graph(const Layout& t) {
    const Grid& grid = *t.grid;
    const i64 ncell = grid.ncell();
    const auto off = offsets(t.directions, t.symmetric);
    const int nd = static_cast<int>(off.size());
    WalkGraph g;
    // Row sums in the Matrix package's order (ascending neighbours, double),
    // needed for transitionSolidify (rowMeans > 1e-300) and the degrees.
    Csc m = to_csc(grid, t.directions, t.symmetric, t.values);
    (void)nd;
    std::vector<double> colsum(static_cast<std::size_t>(ncell), 0.0);
    for (i64 j = 0; j < ncell; ++j) {
        double s = 0.0;
        for (i64 k = m.p[j]; k < m.p[j + 1]; ++k) s += m.x[k];
        colsum[j] = s;
    }
    g.node_of.assign(static_cast<std::size_t>(ncell), -1);
    for (i64 c = 0; c < ncell; ++c) {
        if (colsum[c] / static_cast<double>(ncell) > 1e-300) {  // symmetric: rowMeans = colMeans
            g.node_of[c] = static_cast<std::int32_t>(g.cell.size());
            g.cell.push_back(c);
        }
    }
    const std::size_t n = g.cell.size();
    g.ptr.assign(n + 1, 0);
    g.degree.resize(n);
    for (std::size_t v = 0; v < n; ++v) {
        const i64 c = g.cell[v];
        for (i64 k = m.p[c]; k < m.p[c + 1]; ++k) {
            const std::int32_t u = g.node_of[m.i[k]];
            if (u < 0) continue;
            g.adj.push_back(u);
            g.w.push_back(m.x[k]);
        }
        g.ptr[v + 1] = static_cast<std::int64_t>(g.adj.size());
        g.degree[v] = colsum[c];
    }
    // sum(transitionMatrix): Matrix sums the stored values column by column
    double vol = 0.0;
    for (std::size_t k = 0; k < m.x.size(); ++k) vol += m.x[k];
    g.volume = vol;
    // components
    g.component.assign(n, -1);
    std::vector<std::int32_t> stack;
    for (std::size_t s = 0; s < n; ++s) {
        if (g.component[s] >= 0) continue;
        const std::int32_t id = g.n_components++;
        g.component_volume.push_back(0.0);
        stack.push_back(static_cast<std::int32_t>(s));
        g.component[s] = id;
        while (!stack.empty()) {
            const std::int32_t v = stack.back();
            stack.pop_back();
            for (std::int64_t e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
                const std::int32_t u = g.adj[static_cast<std::size_t>(e)];
                if (g.component[u] < 0) {
                    g.component[u] = id;
                    stack.push_back(u);
                }
            }
        }
    }
    for (std::size_t v = 0; v < n; ++v) g.component_volume[static_cast<std::size_t>(g.component[v])] += g.degree[v];
    return g;
}

std::vector<double> effective_resistance(const WalkGraph& g, const std::vector<std::int32_t>& nodes, bool geometric,
                                         i64 nx) {
    const std::size_t k = nodes.size();
    std::vector<double> out(k * k, kInf);
    for (std::size_t a = 0; a < k; ++a) {
        for (std::size_t b = 0; b < k; ++b) {
            if (nodes[a] < 0 || nodes[b] < 0) out[a * k + b] = kNaN;
        }
        if (nodes[a] >= 0) out[a * k + a] = 0.0;
    }
    const auto members = members_of(g);
    std::vector<std::int32_t> local(g.cell.size(), -1);
    for (std::int32_t comp = 0; comp < g.n_components; ++comp) {
        // the requested nodes of this component, distinct
        std::vector<std::int32_t> want;
        for (const std::int32_t v : nodes)
            if (v >= 0 && g.component[v] == comp) want.push_back(v);
        std::sort(want.begin(), want.end());
        want.erase(std::unique(want.begin(), want.end()), want.end());
        if (want.size() < 2) continue;
        ComponentSystem cs;
        build(g, members[static_cast<std::size_t>(comp)], geometric, nx, cs, local);
        const std::size_t nu = cs.nodes.size();
        // G = L_g^-1 on the wanted nodes (the ground: 0)
        std::vector<std::int32_t> col(want.size(), -1);  // unknown of each wanted node
        for (std::size_t a = 0; a < want.size(); ++a) {
            if (want[a] == cs.ground) continue;
            const auto it = std::lower_bound(cs.nodes.begin(), cs.nodes.end(), want[a]);
            col[a] = cs.iperm[static_cast<std::size_t>(it - cs.nodes.begin())];
        }
        const std::size_t m = want.size();
        std::vector<double> G(m * m, 0.0);
        const std::size_t block = std::max<std::size_t>(1, std::min<std::size_t>(64, 20000000 / std::max<std::size_t>(1, nu)));
        const std::size_t nblocks = (m + block - 1) / block;
        const int nt = std::max(1, std::min(threads(), static_cast<int>(nblocks)));
        std::vector<circuit::SolveScratch> scratch(static_cast<std::size_t>(nt));
#pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (std::int64_t bi = 0; bi < static_cast<std::int64_t>(nblocks); ++bi) {
            int tid = 0;
#if defined(_OPENMP)
            tid = omp_get_thread_num();
#endif
            const std::size_t c0 = static_cast<std::size_t>(bi) * block, c1 = std::min(m, c0 + block);
            std::vector<double> x(nu * (c1 - c0), 0.0);
            for (std::size_t a = c0; a < c1; ++a)
                if (col[a] >= 0) x[(a - c0) * nu + static_cast<std::size_t>(col[a])] = 1.0;
            cs.sys.solve_many(x.data(), static_cast<i64>(c1 - c0), scratch[static_cast<std::size_t>(tid)]);
            for (std::size_t a = c0; a < c1; ++a) {
                for (std::size_t b = 0; b < m; ++b) {
                    G[b * m + a] = col[b] >= 0 ? x[(a - c0) * nu + static_cast<std::size_t>(col[b])] : 0.0;
                }
            }
        }
        // resistances between the wanted nodes, then the requested order
        for (std::size_t a = 0; a < k; ++a) {
            if (nodes[a] < 0 || g.component[nodes[a]] != comp) continue;
            const std::size_t ia = static_cast<std::size_t>(std::lower_bound(want.begin(), want.end(), nodes[a]) - want.begin());
            for (std::size_t b = 0; b < k; ++b) {
                if (nodes[b] < 0 || g.component[nodes[b]] != comp) continue;
                const std::size_t ib =
                    static_cast<std::size_t>(std::lower_bound(want.begin(), want.end(), nodes[b]) - want.begin());
                const double gab = (G[ia * m + ib] + G[ib * m + ia]) / 2;
                out[a * k + b] = ia == ib ? 0.0 : G[ia * m + ia] + G[ib * m + ib] - 2 * gab;
            }
        }
    }
    return out;
}

std::vector<double> random_walk_passages(const WalkGraph& g, const std::vector<std::int32_t>& origins,
                                         const std::vector<std::int32_t>& goals, bool quirk, bool geometric, i64 nx,
                                         i64 ncell) {
    std::vector<double> out(static_cast<std::size_t>(ncell), 0.0);
    if (origins.empty() || goals.empty()) throw std::invalid_argument("origins and goals are needed");
    const std::int32_t comp = g.component[origins[0]];
    for (const std::int32_t v : origins)
        if (g.component[v] != comp) throw std::invalid_argument("the origins and goals are not all connected");
    for (const std::int32_t v : goals)
        if (g.component[v] != comp) throw std::invalid_argument("the origins and goals are not all connected");
    const auto members = members_of(g);
    const auto& mem = members[static_cast<std::size_t>(comp)];
    std::vector<std::int32_t> local(g.cell.size(), -1);
    ComponentSystem cs;
    build(g, mem, geometric, nx, cs, local);
    const std::size_t nu = cs.nodes.size();
    // e[from] = 1/lf, then e[to] = -1/lt (gdistance's .currentR, without its scale C)
    std::vector<double> b(nu, 0.0), v(nu);
    auto unknown = [&](std::int32_t node) -> std::int64_t {
        if (node == cs.ground) return -1;
        const auto it = std::lower_bound(cs.nodes.begin(), cs.nodes.end(), node);
        return cs.iperm[static_cast<std::size_t>(it - cs.nodes.begin())];
    };
    const double ef = 1.0 / static_cast<double>(origins.size()), et = 1.0 / static_cast<double>(goals.size());
    for (const std::int32_t o : origins) {
        const std::int64_t u = unknown(o);
        if (u >= 0) b[static_cast<std::size_t>(u)] = ef;
    }
    for (const std::int32_t d : goals) {
        const std::int64_t u = unknown(d);
        if (u >= 0) b[static_cast<std::size_t>(u)] = -et;
    }
    cs.sys.solve(b.data(), v.data());
    // potentials by graph node (the other components: unused)
    std::vector<double> pot(g.cell.size(), 0.0);
    for (std::size_t i = 0; i < nu; ++i) pot[static_cast<std::size_t>(cs.nodes[i])] = v[static_cast<std::size_t>(cs.iperm[i])];
    pot[static_cast<std::size_t>(cs.ground)] = quirk && nu > 0 ? pot[static_cast<std::size_t>(cs.nodes.back())] : 0.0;
    // half the absolute current through each node: colSums(|V_i - V_j| g_ij) / 2
    for (const std::int32_t j : mem) {
        double s = 0.0;
        bool diag_done = false;
        for (std::int64_t e = g.ptr[j]; e < g.ptr[j + 1]; ++e) {
            const std::int32_t i = g.adj[static_cast<std::size_t>(e)];
            if (!diag_done && i > j) {
                s += 0.0 * -g.degree[j];  // the diagonal entry: |V_j - V_j| (-L_jj)
                diag_done = true;
            }
            s += std::fabs(pot[i] - pot[j]) * g.w[static_cast<std::size_t>(e)];
        }
        if (!diag_done) s += 0.0 * -g.degree[j];
        out[static_cast<std::size_t>(g.cell[j])] = s / 2;
    }
    for (const std::int32_t o : origins) out[static_cast<std::size_t>(g.cell[o])] = 1.0;
    for (const std::int32_t d : goals) out[static_cast<std::size_t>(g.cell[d])] = 1.0;
    return out;
}

}  // namespace cost
