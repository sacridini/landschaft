// SPDX-License-Identifier: GPL-3.0-or-later

#include "circuit/circuitscape.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "circuit/system.hpp"

namespace circuit {

namespace {

constexpr double kSqrt2 = 1.4142135623730951;
constexpr double kShift = 10 * std::numeric_limits<double>::epsilon();  // CHOLMOD's (construct_cholesky_factor)

// raster/graph.jl: cond_avg, weird_avg, res_avg, weirder_avg.
double average(double x, double y, bool diagonal, bool resistances) {
    if (resistances) return diagonal ? 1 / (kSqrt2 * (1 / x + 1 / y) / 2) : 1 / ((1 / x + 1 / y) / 2);
    return diagonal ? (x + y) / kTwoSqrt2 : (x + y) / 2;
}

}  // namespace

RasterGraph raster_graph(const double* gmap, const std::int64_t* polymap, i64 nx, i64 ny, bool avg_resistances,
                         bool four_neighbors) {
    RasterGraph g;
    g.nx = nx;
    g.ny = ny;
    const i64 n = nx * ny;
    // construct_node_map: habitat cells numbered in column-major order ...
    std::vector<i64> label(static_cast<std::size_t>(n), 0);
    i64 next = 0;
    for (i64 x = 0; x < nx; ++x)
        for (i64 y = 0; y < ny; ++y)
            if (gmap[y * nx + x] > 0) label[y * nx + x] = ++next;
    // ... every cell of a region with habitat takes the label of its first habitat cell ...
    if (polymap) {
        std::unordered_map<std::int64_t, i64> first;
        for (i64 x = 0; x < nx; ++x)
            for (i64 y = 0; y < ny; ++y) {
                const i64 c = y * nx + x;
                if (polymap[c] != 0 && gmap[c] > 0) first.emplace(polymap[c], label[c]);
            }
        for (i64 c = 0; c < n; ++c) {
            if (polymap[c] == 0) continue;
            const auto it = first.find(polymap[c]);
            if (it != first.end()) label[c] = it->second;
        }
    }
    // ... and relabel!: nodes by the rank of their label.
    std::vector<i64> labels;
    for (i64 c = 0; c < n; ++c)
        if (label[c] != 0) labels.push_back(label[c]);
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    g.n_nodes = static_cast<std::int32_t>(labels.size());
    g.nodemap.assign(static_cast<std::size_t>(n), -1);
    std::vector<std::int32_t> cells_per_node(labels.size(), 0);
    g.node_cell.assign(labels.size(), -1);
    for (i64 c = 0; c < n; ++c) {
        if (label[c] == 0) continue;
        const std::int32_t v = static_cast<std::int32_t>(std::lower_bound(labels.begin(), labels.end(), label[c]) -
                                                         labels.begin());
        g.nodemap[c] = v;
        if (cells_per_node[v]++ == 0) g.node_cell[v] = c;
    }
    for (std::int32_t v = 0; v < g.n_nodes; ++v)
        if (cells_per_node[v] > 1) g.node_cell[v] = -1;

    // construct_graph: edges in column-major cell order, to the right, below
    // and (8 neighbours) below right and above right; both orientations.
    struct Edge {
        std::int32_t a, b;
        double w;
    };
    std::vector<Edge> edges;
    auto emit = [&](i64 c, i64 d, bool diagonal) {
        const std::int32_t a = g.nodemap[c], b = g.nodemap[d];
        if (a < 0 || b < 0) return;
        const double w = average(gmap[c], gmap[d], diagonal, avg_resistances);
        if (a == b) return;  // within a merged node: nothing in the Laplacian
        edges.push_back({a, b, w});
        edges.push_back({b, a, w});
    };
    for (i64 x = 0; x < nx; ++x)
        for (i64 y = 0; y < ny; ++y) {
            const i64 c = y * nx + x;
            if (g.nodemap[c] < 0) continue;
            if (x + 1 < nx) emit(c, c + 1, false);
            if (y + 1 < ny) emit(c, c + nx, false);
            if (!four_neighbors) {
                if (y + 1 < ny && x + 1 < nx) emit(c, c + nx + 1, true);
                if (y > 0 && x + 1 < nx) emit(c, c - nx + 1, true);
            }
        }
    // Duplicates (merged nodes) summed in the order met, as sparse! does.
    std::stable_sort(edges.begin(), edges.end(),
                     [](const Edge& p, const Edge& q) { return std::tie(p.a, p.b) < std::tie(q.a, q.b); });
    g.ptr.assign(static_cast<std::size_t>(g.n_nodes) + 1, 0);
    for (std::size_t k = 0; k < edges.size();) {
        std::size_t e = k;
        double w = 0.0;
        bool first = true;
        while (e < edges.size() && edges[e].a == edges[k].a && edges[e].b == edges[k].b) {
            w = first ? edges[e].w : w + edges[e].w;
            first = false;
            ++e;
        }
        g.adj.push_back(edges[k].b);
        g.w.push_back(w);
        ++g.ptr[static_cast<std::size_t>(edges[k].a) + 1];
        k = e;
    }
    for (std::int32_t v = 0; v < g.n_nodes; ++v) g.ptr[v + 1] += g.ptr[v];
    // laplacian!: the diagonal sums the column's off-diagonal entries (rows ascending).
    g.degree.assign(static_cast<std::size_t>(g.n_nodes), 0.0);
    for (std::int32_t v = 0; v < g.n_nodes; ++v) {
        double d = 0.0;
        for (std::int64_t k = g.ptr[v]; k < g.ptr[v + 1]; ++k) d += g.w[static_cast<std::size_t>(k)];
        g.degree[v] = d;
    }
    // connected_components: edges of weight 0 do not connect.
    g.comp.assign(static_cast<std::size_t>(g.n_nodes), -1);
    std::vector<std::int32_t> stack;
    for (std::int32_t u = 0; u < g.n_nodes; ++u) {
        if (g.comp[u] >= 0) continue;
        g.comp[u] = g.n_comp;
        stack.push_back(u);
        while (!stack.empty()) {
            const std::int32_t v = stack.back();
            stack.pop_back();
            for (std::int64_t k = g.ptr[v]; k < g.ptr[v + 1]; ++k) {
                const std::int32_t t = g.adj[static_cast<std::size_t>(k)];
                if (g.w[static_cast<std::size_t>(k)] == 0 || g.comp[t] >= 0) continue;
                g.comp[t] = g.n_comp;
                stack.push_back(t);
            }
        }
        ++g.n_comp;
    }
    return g;
}

std::vector<double> node_sums(const RasterGraph& g, const double* cells) {
    std::vector<double> s(static_cast<std::size_t>(g.n_nodes), 0.0);
    for (i64 x = 0; x < g.nx; ++x)
        for (i64 y = 0; y < g.ny; ++y) {
            const i64 c = y * g.nx + x;
            const std::int32_t v = g.nodemap[c];
            if (v >= 0 && cells[c] != 0) s[v] += cells[c];
        }
    return s;
}

NodeSolution raster_advanced(const RasterGraph& g, std::vector<double> src, std::vector<double> gnd,
                             ConflictPolicy policy, int max_threads) {
    const double inf = std::numeric_limits<double>::infinity();
    const std::int32_t n = g.n_nodes;
    NodeSolution out;
    out.voltage.assign(static_cast<std::size_t>(n), 0.0);
    out.current.assign(static_cast<std::size_t>(n), 0.0);
    // resolve_conflicts: finite grounds are taken before the policy applies.
    std::vector<double> finite(static_cast<std::size_t>(n), 0.0);
    bool any_finite = false;
    for (std::int32_t v = 0; v < n; ++v) {
        finite[v] = gnd[v] < inf ? gnd[v] : 0.0;
        if (finite[v] != 0) any_finite = true;
    }
    for (std::int32_t v = 0; v < n; ++v) {
        if (!(src[v] != 0 && gnd[v] != 0)) continue;
        if (policy == ConflictPolicy::RemoveSource || policy == ConflictPolicy::RemoveAll) src[v] = 0.0;
        else if (policy == ConflictPolicy::RemoveGround) gnd[v] = 0.0;
    }
    for (std::int32_t v = 0; v < n; ++v)
        if (gnd[v] == inf && src[v] > 0) gnd[v] = 0.0;

    // Components in order, their nodes ascending.
    std::vector<std::int64_t> start(static_cast<std::size_t>(g.n_comp) + 1, 0);
    for (std::int32_t v = 0; v < n; ++v) ++start[static_cast<std::size_t>(g.comp[v]) + 1];
    for (std::int32_t k = 0; k < g.n_comp; ++k) start[k + 1] += start[k];
    std::vector<std::int32_t> members(static_cast<std::size_t>(n));
    {
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (std::int32_t v = 0; v < n; ++v) members[static_cast<std::size_t>(fill[g.comp[v]]++)] = v;
    }
    std::vector<std::int32_t> solvable;
    for (std::int32_t k = 0; k < g.n_comp; ++k) {
        double s = 0.0, t = 0.0;
        for (std::int64_t p = start[k]; p < start[k + 1]; ++p) {
            s += src[members[p]];
            t += gnd[members[p]];
        }
        if (s != 0 && t != 0) solvable.push_back(k);
    }
    out.solved = !solvable.empty();

    bool failed = false;
    const int n_solvable = static_cast<int>(solvable.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(max_threads > 0 ? max_threads : omp_get_max_threads()) if (n_solvable > 1)
#else
    (void)max_threads;
#endif
    for (int t = 0; t < n_solvable; ++t) {
        const std::int32_t k = solvable[t];
        const std::int64_t p0 = start[k], p1 = start[k + 1];
        // The unknowns: the component's nodes without a direct ground.
        std::unordered_map<std::int32_t, std::int32_t> local;
        std::vector<std::int32_t> nodes;
        i64 x0 = g.nx, x1 = -1, y0 = g.ny, y1 = -1;
        for (std::int64_t p = p0; p < p1; ++p) {
            const std::int32_t v = members[p];
            if (gnd[v] == inf) continue;
            local.emplace(v, static_cast<std::int32_t>(nodes.size()));
            nodes.push_back(v);
            const i64 c = g.node_cell[v];
            if (c >= 0) {
                x0 = std::min(x0, c % g.nx);
                x1 = std::max(x1, c % g.nx);
                y0 = std::min(y0, c / g.nx);
                y1 = std::max(y1, c / g.nx);
            }
        }
        std::vector<double> x(nodes.size(), 0.0);
        if (!nodes.empty()) {
            const i64 bx = x1 >= x0 ? x1 - x0 + 1 : 1, by = y1 >= y0 ? y1 - y0 + 1 : 1;
            std::vector<i64> cell(nodes.size());
            std::vector<std::int64_t> ptr(nodes.size() + 1, 0);
            std::vector<std::int32_t> adj;
            std::vector<double> off, diag(nodes.size());
            for (std::size_t i = 0; i < nodes.size(); ++i) {
                const std::int32_t v = nodes[i];
                const i64 c = g.node_cell[v];
                cell[i] = c >= 0 ? (c / g.nx - y0) * bx + (c % g.nx - x0) : -1;
                for (std::int64_t e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
                    const auto it = local.find(g.adj[static_cast<std::size_t>(e)]);
                    if (it == local.end()) continue;
                    adj.push_back(it->second);
                    off.push_back(-g.w[static_cast<std::size_t>(e)]);
                }
                ptr[i + 1] = static_cast<std::int64_t>(adj.size());
                // multiple_solver: the Laplacian plus the finite grounds, and
                // CHOLMOD's shift.
                diag[i] = (any_finite ? g.degree[v] + finite[v] : g.degree[v]) + kShift;
            }
            CellSystem sys;
            sys.share(frontal_analyze_graph(bx, by, cell, ptr, adj));
            sys.set_values(diag.data(), off.data());
            const std::vector<std::int32_t>& perm = sys.perm();
            std::vector<double> b(nodes.size());
            for (std::size_t i = 0; i < nodes.size(); ++i) b[i] = src[nodes[perm[i]]];
            std::vector<double> y(nodes.size());
            if (!sys.factorize_solve(b.data(), y.data())) {
#pragma omp critical(circuitscape_failed)
                failed = true;
                continue;
            }
            for (std::size_t i = 0; i < nodes.size(); ++i) out.voltage[nodes[perm[i]]] = y[i];
        }
        // get_node_currents over the component (direct grounds at 0 V).
        const std::vector<double>& v = out.voltage;
        double maxin = -inf, maxout = -inf;
        for (std::int64_t p = p0; p < p1; ++p) {
            const std::int32_t c = members[p];
            for (std::int64_t e = g.ptr[c]; e < g.ptr[c + 1]; ++e) {
                const std::int32_t r = g.adj[static_cast<std::size_t>(e)];
                const double b = std::abs(g.w[static_cast<std::size_t>(e)]) * (v[r] - v[c]);
                if (r < c) {
                    if (b > maxin) maxin = b;
                } else if (b > maxout) {
                    maxout = b;
                }
            }
        }
        for (std::int64_t p = p0; p < p1; ++p) {
            const std::int32_t c = members[p];
            double inflow = 0.0, outflow = 0.0;
            for (std::int64_t e = g.ptr[c]; e < g.ptr[c + 1]; ++e) {
                const std::int32_t r = g.adj[static_cast<std::size_t>(e)];
                const double b = std::abs(g.w[static_cast<std::size_t>(e)]) * (v[r] - v[c]);
                if (b > 0) {
                    if (!(std::abs(b / maxin) < 1e-8)) inflow += b;
                } else if (b < 0) {
                    if (!(std::abs(b / maxout) < 1e-8)) outflow -= b;
                }
            }
            if (any_finite) {
                const double f = finite[c] * v[c];
                if (f < 0) inflow -= f;
                else if (f > 0) outflow += f;
            }
            out.current[c] = inflow > outflow ? inflow : outflow;
        }
    }
    if (failed) throw std::runtime_error("a circuit component is not positive definite");
    return out;
}

}  // namespace circuit
