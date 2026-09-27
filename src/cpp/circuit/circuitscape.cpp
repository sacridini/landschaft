// SPDX-License-Identifier: GPL-3.0-or-later

#include "circuit/circuitscape.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <Eigen/Cholesky>
#include <Eigen/Core>

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

// get_node_currents over the nodes `members` (one component, ascending),
// voltages v by node, finite grounds by node (nullptr: none): cur[node].
void graph_node_currents(const RasterGraph& g, const std::int32_t* members, std::int64_t count, const double* v,
                         const double* finite, double* cur) {
    const double inf = std::numeric_limits<double>::infinity();
    double maxin = -inf, maxout = -inf;
    for (std::int64_t p = 0; p < count; ++p) {
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
    for (std::int64_t p = 0; p < count; ++p) {
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
        if (finite) {
            const double f = finite[c] * v[c];
            if (f < 0) inflow -= f;
            else if (f > 0) outflow += f;
        }
        cur[c] = inflow > outflow ? inflow : outflow;
    }
}

// The Laplacian of one component (its nodes ascending), plus `finite` on the
// diagonal when given and CHOLMOD's shift, without the nodes `drop` rejects
// (at 0 V), factorised. `nodes[k]` is the node of unknown k in the caller's
// order (use sys.perm() for the elimination order).
struct ComponentSystem {
    CellSystem sys;
    std::vector<std::int32_t> nodes;
};

// Node -> unknown of the component being assembled, -1 elsewhere: one table
// per thread, reset after use.
std::vector<std::int32_t>& local_index(std::int32_t n_nodes) {
    static thread_local std::vector<std::int32_t> table;
    if (table.size() < static_cast<std::size_t>(n_nodes)) table.resize(static_cast<std::size_t>(n_nodes), -1);
    return table;
}

template <class Drop>
bool factor_component(const RasterGraph& g, const std::int32_t* members, std::int64_t count, const double* finite,
                      Drop drop, ComponentSystem& cs) {
    cs.nodes.clear();
    std::vector<std::int32_t>& local = local_index(g.n_nodes);
    i64 x0 = g.nx, x1 = -1, y0 = g.ny, y1 = -1;
    for (std::int64_t p = 0; p < count; ++p) {
        const std::int32_t v = members[p];
        if (drop(v)) continue;
        local[v] = static_cast<std::int32_t>(cs.nodes.size());
        cs.nodes.push_back(v);
        const i64 c = g.node_cell[v];
        if (c >= 0) {
            x0 = std::min(x0, c % g.nx);
            x1 = std::max(x1, c % g.nx);
            y0 = std::min(y0, c / g.nx);
            y1 = std::max(y1, c / g.nx);
        }
    }
    if (cs.nodes.empty()) return true;
    struct Reset {
        std::vector<std::int32_t>& t;
        const std::vector<std::int32_t>& nodes;
        ~Reset() {
            for (std::int32_t v : nodes) t[v] = -1;
        }
    } reset{local, cs.nodes};
    const i64 bx = x1 >= x0 ? x1 - x0 + 1 : 1, by = y1 >= y0 ? y1 - y0 + 1 : 1;
    std::vector<i64> cell(cs.nodes.size());
    std::vector<std::int64_t> ptr(cs.nodes.size() + 1, 0);
    std::vector<std::int32_t> adj;
    std::vector<double> off, diag(cs.nodes.size());
    for (std::size_t i = 0; i < cs.nodes.size(); ++i) {
        const std::int32_t v = cs.nodes[i];
        const i64 c = g.node_cell[v];
        cell[i] = c >= 0 ? (c / g.nx - y0) * bx + (c % g.nx - x0) : -1;
        for (std::int64_t e = g.ptr[v]; e < g.ptr[v + 1]; ++e) {
            const std::int32_t u = local[g.adj[static_cast<std::size_t>(e)]];
            if (u < 0) continue;
            adj.push_back(u);
            off.push_back(-g.w[static_cast<std::size_t>(e)]);
        }
        ptr[i + 1] = static_cast<std::int64_t>(adj.size());
        // multiple_solver / construct_cholesky_factor: finite grounds, shift.
        diag[i] = (finite ? g.degree[v] + finite[v] : g.degree[v]) + kShift;
    }
    cs.sys.share(frontal_analyze_graph(bx, by, cell, ptr, adj));
    cs.sys.set_values(diag.data(), off.data());
    return cs.sys.factorize();
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

RasterGraph network_graph(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b,
                          const std::vector<double>& w, std::int32_t m) {
    RasterGraph g;
    g.n_nodes = m;
    g.node_cell.assign(static_cast<std::size_t>(m), -1);
    struct Entry {
        std::int32_t r, c;
        double w;
    };
    // sparse(a, b, w): duplicates summed in input order (column-major order).
    std::vector<Entry> in(a.size());
    for (std::size_t k = 0; k < a.size(); ++k) {
        if (a[k] < 0 || b[k] < 0 || a[k] >= m || b[k] >= m) throw std::invalid_argument("node index out of range");
        in[k] = {a[k], b[k], w[k]};
    }
    auto by_column = [m](std::vector<Entry>& v) {
        // stable: counting sort by column, then insertion sort by row within each column
        std::vector<std::int64_t> start(static_cast<std::size_t>(m) + 1, 0);
        for (const Entry& e : v) ++start[static_cast<std::size_t>(e.c) + 1];
        for (std::int32_t c = 0; c < m; ++c) start[c + 1] += start[c];
        std::vector<Entry> out(v.size());
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (const Entry& e : v) out[static_cast<std::size_t>(fill[e.c]++)] = e;
        for (std::int32_t c = 0; c < m; ++c) {
            Entry* b = out.data() + start[c];
            Entry* e = out.data() + start[c + 1];
            for (Entry* x = b + 1; x < e; ++x) {
                Entry k = *x;
                Entry* y = x;
                for (; y > b && (y - 1)->r > k.r; --y) *y = *(y - 1);
                *y = k;
            }
        }
        v.swap(out);
    };
    by_column(in);
    std::vector<Entry> A;
    for (std::size_t k = 0; k < in.size();) {
        Entry e = in[k];
        std::size_t t = k + 1;
        for (; t < in.size() && in[t].r == e.r && in[t].c == e.c; ++t) e.w += in[t].w;
        A.push_back(e);
        k = t;
    }
    // A + A': both orientations, (r, c) = A[r, c] + A[c, r]; zeros dropped; self loops left out.
    std::vector<Entry> both;
    both.reserve(2 * A.size());
    for (const Entry& e : A) {
        if (e.r == e.c) continue;
        both.push_back({e.r, e.c, e.w});
        both.push_back({e.c, e.r, e.w});
    }
    by_column(both);
    g.ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    for (std::size_t k = 0; k < both.size();) {
        // at most two entries per position: A[r, c] and A[c, r]
        double v = both[k].w;
        std::size_t t = k + 1;
        if (t < both.size() && both[t].r == both[k].r && both[t].c == both[k].c) v = both[k].w + both[t++].w;
        if (v != 0) {
            // column c holds row r: node c's neighbour r
            g.adj.push_back(both[k].r);
            g.w.push_back(v);
            ++g.ptr[static_cast<std::size_t>(both[k].c) + 1];
        }
        k = t;
    }
    for (std::int32_t v = 0; v < m; ++v) g.ptr[v + 1] += g.ptr[v];
    g.degree.assign(static_cast<std::size_t>(m), 0.0);
    for (std::int32_t v = 0; v < m; ++v) {
        double d = 0.0;
        for (std::int64_t k = g.ptr[v]; k < g.ptr[v + 1]; ++k) d += g.w[static_cast<std::size_t>(k)];
        g.degree[v] = d;
    }
    g.comp.assign(static_cast<std::size_t>(m), -1);
    std::vector<std::int32_t> stack;
    for (std::int32_t u = 0; u < m; ++u) {
        if (g.comp[u] >= 0) continue;
        g.comp[u] = g.n_comp;
        stack.push_back(u);
        while (!stack.empty()) {
            const std::int32_t v = stack.back();
            stack.pop_back();
            for (std::int64_t k = g.ptr[v]; k < g.ptr[v + 1]; ++k) {
                const std::int32_t t = g.adj[static_cast<std::size_t>(k)];
                if (g.comp[t] >= 0) continue;
                g.comp[t] = g.n_comp;
                stack.push_back(t);
            }
        }
        ++g.n_comp;
    }
    return g;
}

void network_currents(const RasterGraph& g, const std::int32_t* members, std::int64_t count, const double* v,
                      const double* finite, double* node_cur, std::vector<std::pair<std::int64_t, double>>& branch) {
    graph_node_currents(g, members, count, v, finite, node_cur);
    branch.clear();
    double maxcur = -std::numeric_limits<double>::infinity();
    for (std::int64_t p = 0; p < count; ++p) {
        const std::int32_t c = members[p];
        for (std::int64_t e = g.ptr[c]; e < g.ptr[c + 1]; ++e) {
            const std::int32_t r = g.adj[static_cast<std::size_t>(e)];
            if (r >= c) continue;
            const double b = std::abs(g.w[static_cast<std::size_t>(e)]) * (v[r] - v[c]);
            maxcur = std::max(maxcur, b);
            branch.emplace_back(e, b);
        }
    }
    for (auto& eb : branch) {
        double& b = eb.second;
        if (std::abs(b / maxcur) < 1e-8) b = 0.0;
        b = std::abs(b);
    }
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
                             ConflictPolicy policy, int max_threads, std::int32_t only_node) {
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
    for (std::int32_t v = 0; v < n; ++v)
        if (src[v] != 0 || gnd[v] != 0) out.active.push_back(v);

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
        if (only_node == -2 || (only_node >= 0 && g.comp[only_node] != k)) continue;
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
        ComponentSystem cs;
        if (!factor_component(g, members.data() + p0, p1 - p0, any_finite ? finite.data() : nullptr,
                              [&](std::int32_t v) { return gnd[v] == inf; }, cs)) {
#pragma omp critical(circuitscape_failed)
            failed = true;
            continue;
        }
        if (!cs.nodes.empty()) {
            const std::vector<std::int32_t>& perm = cs.sys.perm();
            std::vector<double> b(cs.nodes.size()), y(cs.nodes.size());
            for (std::size_t i = 0; i < cs.nodes.size(); ++i) b[i] = src[cs.nodes[perm[i]]];
            cs.sys.solve(b.data(), y.data());
            for (std::size_t i = 0; i < cs.nodes.size(); ++i) out.voltage[cs.nodes[perm[i]]] = y[i];
        }
        graph_node_currents(g, members.data() + p0, p1 - p0, out.voltage.data(), any_finite ? finite.data() : nullptr,
                            out.current.data());
    }
    if (failed) throw std::runtime_error("a circuit component is not positive definite");
    return out;
}

}  // namespace circuit

namespace circuit {

PairwiseResult raster_pairwise(const RasterGraph& g, const double* gmap, const std::vector<std::int32_t>& points,
                               const std::vector<std::uint8_t>& excluded, const PairwiseOptions& o,
                               const std::function<void(PairMaps&&)>& emit, int max_threads) {
    const std::int32_t np = static_cast<std::int32_t>(points.size());
    const i64 ncell = g.nx * g.ny;
    PairwiseResult out;
    out.resistances.assign(static_cast<std::size_t>(np) * np, -1.0);
    auto R = [&](std::int32_t a, std::int32_t b) -> double& { return out.resistances[static_cast<std::size_t>(a) * np + b]; };
    bool any_excluded = false;
    for (std::uint8_t e : excluded) any_excluded = any_excluded || e;
    const bool shortcut = !o.network && !o.maps() && !any_excluded;
    std::vector<double> voltmatrix, shortcut_res;
    if (shortcut) {
        voltmatrix.assign(static_cast<std::size_t>(np) * np, 0.0);
        shortcut_res.assign(static_cast<std::size_t>(np) * np, -1.0);
    }
    if (o.network) {
        if (o.maps()) {
            out.cum.assign(static_cast<std::size_t>(g.n_nodes), 0.0);
            out.cum_branch.assign(g.adj.size(), 0.0);
        }
    } else {
        if (o.cum) out.cum.assign(static_cast<std::size_t>(ncell), 0.0);
        if (o.max) out.max.assign(static_cast<std::size_t>(ncell), -9999.0);
    }
    const int threads =
#if defined(_OPENMP)
        max_threads > 0 ? max_threads : omp_get_max_threads();
#else
        1;
    (void)max_threads;
#endif

    // Components in order; their nodes ascending.
    std::vector<std::int64_t> start(static_cast<std::size_t>(g.n_comp) + 1, 0);
    for (std::int32_t v = 0; v < g.n_nodes; ++v) ++start[static_cast<std::size_t>(g.comp[v]) + 1];
    for (std::int32_t k = 0; k < g.n_comp; ++k) start[k + 1] += start[k];
    std::vector<std::int32_t> members(static_cast<std::size_t>(g.n_nodes));
    {
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (std::int32_t v = 0; v < g.n_nodes; ++v) members[static_cast<std::size_t>(fill[g.comp[v]]++)] = v;
    }
    // Focal indices by node (positions), in order.
    std::unordered_map<std::int32_t, std::vector<std::int32_t>> positions;
    for (std::int32_t i = 0; i < np; ++i)
        if (points[i] >= 0) positions[points[i]].push_back(i);

    for (std::int32_t k = 0; k < g.n_comp; ++k) {
        const std::int64_t p0 = start[k], p1 = start[k + 1];
        // component_points: distinct focal nodes, in first-appearance order.
        std::vector<std::int32_t> csub;
        {
            std::unordered_set<std::int32_t> seen;
            for (std::int32_t i = 0; i < np; ++i) {
                const std::int32_t v = points[i];
                if (v >= 0 && g.comp[v] == k && seen.insert(v).second) csub.push_back(v);
            }
        }
        if (csub.empty()) continue;
        // Focal points sharing a node: resistance 0.
        for (std::int32_t v : csub) {
            const auto& pos = positions[v];
            for (std::size_t a = 0; a < pos.size(); ++a)
                for (std::size_t b = a + 1; b < pos.size(); ++b) R(pos[a], pos[b]) = R(pos[b], pos[a]) = 0.0;
        }
        // The jobs: pairs of focal nodes, not every combination excluded.
        struct Job {
            std::int32_t src, dst;
        };
        std::vector<Job> jobs;
        const std::size_t nsrc = shortcut ? std::min<std::size_t>(1, csub.size()) : csub.size();
        for (std::size_t si = 0; si < nsrc; ++si)
            for (std::size_t di = si + 1; di < csub.size(); ++di) {
                bool any = false;
                for (std::int32_t ci : positions[csub[si]])
                    for (std::int32_t cj : positions[csub[di]]) any = any || !excluded[static_cast<std::size_t>(ci) * np + cj];
                if (any) jobs.push_back({csub[si], csub[di]});
            }
        if (jobs.empty()) continue;
        // One node grounded (the component's last): exact voltage differences.
        const std::int32_t ref = members[p1 - 1];
        ComponentSystem cs;
        if (!factor_component(g, members.data() + p0, p1 - p0, nullptr, [&](std::int32_t v) { return v == ref; }, cs))
            throw std::runtime_error("a circuit component is not positive definite");
        const std::size_t m = cs.nodes.size();
        const std::vector<std::int32_t>& perm = cs.sys.perm();
        std::vector<std::int32_t> elim(static_cast<std::size_t>(p1 - p0), -1);  // member -> unknown in elimination order
        // members of a component are ascending: a node's member index by binary search
        const std::int32_t* mb = members.data() + p0;
        const std::int32_t* me = members.data() + p1;
        auto member_index = [mb, me](std::int32_t v) {
            return static_cast<std::int32_t>(std::lower_bound(mb, me, v) - mb);
        };
        for (std::size_t e = 0; e < m; ++e) elim[static_cast<std::size_t>(member_index(cs.nodes[perm[e]]))] = static_cast<std::int32_t>(e);

        // Batches of 8 right-hand sides per thread; results handled in job order.
        // With maps every job keeps the voltages of the whole component (batches
        // capped at about 256 MB of them), else those of its focal nodes only.
        constexpr std::size_t kCols = 8;
        const std::int64_t count = p1 - p0;
        const bool full = o.maps();
        std::unordered_map<std::int32_t, std::int32_t> focal_slot;  // focal node -> position in csub
        for (std::size_t k2 = 0; k2 < csub.size(); ++k2) focal_slot.emplace(csub[k2], static_cast<std::int32_t>(k2));
        const std::size_t stride = full ? static_cast<std::size_t>(count) : csub.size();
        std::size_t per_batch = kCols * static_cast<std::size_t>(threads) * 4;
        if (full) per_batch = std::max(kCols, std::min(per_batch, (std::size_t(1) << 25) / std::max<std::size_t>(stride, 1)));
        std::vector<SolveScratch> scratch(static_cast<std::size_t>(threads));
        for (std::size_t b0 = 0; b0 < jobs.size(); b0 += per_batch) {
            const std::size_t nb = std::min(per_batch, jobs.size() - b0);
            // Voltages (relative to the source) by member, or by focal node, one column per job.
            std::vector<double> volt(stride * nb, 0.0);
            const long long ngroups = static_cast<long long>((nb + kCols - 1) / kCols);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
            for (long long gi = 0; gi < ngroups; ++gi) {
                const int tid =
#if defined(_OPENMP)
                    omp_get_thread_num();
#else
                    0;
#endif
                const std::size_t j0 = static_cast<std::size_t>(gi) * kCols, j1 = std::min(nb, j0 + kCols);
                const i64 kk = static_cast<i64>(j1 - j0);
                std::vector<double> rhs(m * static_cast<std::size_t>(kk), 0.0);
                for (std::size_t j = j0; j < j1; ++j) {
                    const Job& job = jobs[b0 + j];
                    double* col = rhs.data() + (j - j0) * m;
                    const std::int32_t ei = elim[static_cast<std::size_t>(member_index(job.src))];
                    const std::int32_t ej = elim[static_cast<std::size_t>(member_index(job.dst))];
                    if (ei >= 0) col[ei] = -1.0;
                    if (ej >= 0) col[ej] = 1.0;
                }
                cs.sys.solve_many(rhs.data(), kk, scratch[static_cast<std::size_t>(tid)]);
                for (std::size_t j = j0; j < j1; ++j) {
                    const double* col = rhs.data() + (j - j0) * m;
                    double* vcol = volt.data() + j * stride;
                    if (full) {
                        for (std::int64_t q = 0; q < count; ++q) {
                            const std::int32_t e = elim[static_cast<std::size_t>(q)];
                            vcol[q] = e >= 0 ? col[e] : 0.0;
                        }
                        const double vs = vcol[member_index(jobs[b0 + j].src)];
                        for (std::int64_t q = 0; q < count; ++q) vcol[q] -= vs;
                    } else {
                        for (std::size_t k2 = 0; k2 < csub.size(); ++k2) {
                            const std::int32_t e = elim[static_cast<std::size_t>(member_index(csub[k2]))];
                            vcol[k2] = e >= 0 ? col[e] : 0.0;
                        }
                        const double vs = vcol[focal_slot.at(jobs[b0 + j].src)];
                        for (std::size_t k2 = 0; k2 < csub.size(); ++k2) vcol[k2] -= vs;
                    }
                }
            }
            // handle_pair, in job order.
            std::vector<double> vnode, cnode;
            std::vector<std::pair<std::int64_t, double>> branch;
            if (o.maps()) {
                vnode.assign(static_cast<std::size_t>(g.n_nodes), 0.0);
                cnode.assign(static_cast<std::size_t>(g.n_nodes), 0.0);
            }
            for (std::size_t j = 0; j < nb; ++j) {
                const Job& job = jobs[b0 + j];
                const double* vcol = volt.data() + j * stride;
                auto volt_at = [&](std::int32_t node) {
                    return full ? vcol[member_index(node)] : vcol[focal_slot.at(node)];
                };
                const double resistance = volt_at(job.dst) - volt_at(job.src);
                if (o.maps()) {
                    for (std::int64_t q = 0; q < count; ++q) vnode[members[p0 + q]] = vcol[q];
                    if (o.network)
                        network_currents(g, members.data() + p0, count, vnode.data(), nullptr, cnode.data(), branch);
                    else
                        graph_node_currents(g, members.data() + p0, count, vnode.data(), nullptr, cnode.data());
                }
                for (std::int32_t ci : positions[job.src])
                    for (std::int32_t cj : positions[job.dst]) {
                        if (excluded[static_cast<std::size_t>(ci) * np + cj]) continue;
                        R(ci, cj) = R(cj, ci) = resistance;
                        if (shortcut) {
                            // update_voltmatrix!: focal points 2..n (Circuitscape skips the first).
                            for (std::int32_t i = 1; i < np; ++i) {
                                const std::int32_t v = points[i];
                                if (v < 0 || g.comp[v] != k) continue;
                                voltmatrix[static_cast<std::size_t>(i) * np + cj] = 1 - (volt_at(v) / resistance);
                            }
                            continue;
                        }
                        if (!o.maps()) continue;
                        PairMaps pm;
                        pm.ci = ci;
                        pm.cj = cj;
                        if (o.network) {
                            // postprocess: the voltages when asked, the currents with any map option
                            pm.nodes.assign(members.begin() + p0, members.begin() + p1);
                            if (o.volt_maps) pm.voltage.assign(vcol, vcol + count);
                            pm.current.resize(static_cast<std::size_t>(count));
                            for (std::int64_t q = 0; q < count; ++q) {
                                const std::int32_t v = members[static_cast<std::size_t>(p0 + q)];
                                pm.current[static_cast<std::size_t>(q)] = cnode[v];
                                out.cum[v] += cnode[v];
                            }
                            for (const auto& eb : branch) out.cum_branch[static_cast<std::size_t>(eb.first)] += eb.second;
                            pm.branch = branch;
                            emit(std::move(pm));
                            continue;
                        }
                        if (o.volt_maps) {
                            pm.voltage.assign(static_cast<std::size_t>(ncell), 0.0);
                            for (i64 c = 0; c < ncell; ++c) {
                                const std::int32_t v = g.nodemap[c];
                                if (v >= 0 && g.comp[v] == k) pm.voltage[c] = vnode[v];
                                if (o.null_voltages && gmap[c] == 0) pm.voltage[c] = -9999.0;
                            }
                        }
                        std::vector<double> cmap(static_cast<std::size_t>(ncell), 0.0);
                        for (i64 c = 0; c < ncell; ++c) {
                            const std::int32_t v = g.nodemap[c];
                            if (v >= 0 && g.comp[v] == k) {
                                cmap[c] = cnode[v];
                                if (o.focal_zero && (v == job.src || v == job.dst)) cmap[c] = 0.0;
                            }
                            if (o.log_transform) cmap[c] = cmap[c] > 0 ? std::log10(cmap[c]) : -9999.0;
                            if (o.null_currents && gmap[c] == 0) cmap[c] = -9999.0;
                        }
                        if (o.cum)
                            for (i64 c = 0; c < ncell; ++c) out.cum[c] += cmap[c];
                        if (o.max)
                            for (i64 c = 0; c < ncell; ++c) out.max[c] = std::max(out.max[c], cmap[c]);
                        if (o.cur_maps) pm.current = std::move(cmap);
                        if (o.volt_maps || o.cur_maps) emit(std::move(pm));
                    }
            }
        }
        if (shortcut) {
            // update_shortcut_resistances!, anchored at the first focal point of csub[0].
            const std::int32_t anchor = positions[csub[0]].front();
            auto S = [&](std::int32_t a, std::int32_t b) -> double& { return shortcut_res[static_cast<std::size_t>(a) * np + b]; };
            constexpr double kInvalid = -777;
            std::vector<std::uint8_t> check(static_cast<std::size_t>(np), 0);
            for (std::int32_t i = 0; i < np; ++i) check[i] = points[i] >= 0 && g.comp[points[i]] == k;
            for (std::int32_t x = 0; x < np; ++x) {
                if (!check[x]) continue;
                const double r1x = R(anchor, x);
                if (r1x == -1) continue;
                S(x, anchor) = S(anchor, x) = r1x;
                for (std::int32_t y = x; y < np; ++y) {
                    if (!check[y]) continue;
                    const double r12 = R(anchor, y);
                    if (r12 == -1) continue;
                    if (r1x != kInvalid) {
                        S(anchor, y) = S(y, anchor) = r12;
                        const double vx = voltmatrix[static_cast<std::size_t>(x) * np + y];
                        const double r2x = 2 * r12 * vx + r1x - r12;
                        if (S(y, x) != kInvalid) S(y, x) = S(x, y) = r2x;
                    } else {
                        for (std::int32_t t = 0; t < np; ++t) S(x, t) = S(t, x) = kInvalid;
                    }
                }
            }
        }
    }
    if (shortcut) out.resistances = std::move(shortcut_res);
    for (std::int32_t i = 0; i < np; ++i) out.resistances[static_cast<std::size_t>(i) * np + i] = 0.0;
    if (o.cum && !o.network)
        for (double& c : out.cum) c = c < -9999 ? -9999 : c;
    if (o.max && !o.network)
        for (double& c : out.max) c = c < -9999 ? -9999 : c;
    return out;
}

}  // namespace circuit

namespace circuit {

std::vector<std::int64_t> region_polymap(const std::int64_t* polymap, i64 n, const std::vector<i64>& cells,
                                         const std::vector<std::int64_t>& ids, std::int64_t pt1, std::int64_t pt2) {
    std::vector<std::int64_t> poly(static_cast<std::size_t>(n), 0);
    if (!polymap) {
        for (std::size_t x = 0; x < ids.size(); ++x)
            if (ids[x] == pt1) poly[static_cast<std::size_t>(cells[x])] = pt1;
        for (std::size_t x = 0; x < ids.size(); ++x)
            if (ids[x] == pt2) poly[static_cast<std::size_t>(cells[x])] = pt2;
        return poly;
    }
    std::copy(polymap, polymap + n, poly.begin());
    std::int64_t k = *std::max_element(polymap, polymap + n);
    for (std::int64_t pt : {pt1, pt2}) {
        std::vector<i64> idx;
        for (std::size_t x = 0; x < ids.size(); ++x)
            if (ids[x] == pt) idx.push_back(cells[x]);
        if (idx.size() == 1) continue;
        // A focal region becomes one node with every polygon it overlaps.
        ++k;
        std::vector<std::int64_t> vals;
        for (i64 c : idx)
            if (poly[static_cast<std::size_t>(c)] != 0) vals.push_back(poly[static_cast<std::size_t>(c)]);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        if (!vals.empty())
            for (auto& v : poly)
                if (std::binary_search(vals.begin(), vals.end(), v)) v = k;
        for (i64 c : idx) poly[static_cast<std::size_t>(c)] = k;
    }
    return poly;
}

PairwiseResult raster_pairwise_regions(const double* gmap, const std::int64_t* polymap, i64 nx, i64 ny,
                                       bool avg_resistances, bool four_neighbors, const std::vector<i64>& cells,
                                       const std::vector<std::int64_t>& ids, const std::vector<std::uint8_t>& excluded,
                                       const PairwiseOptions& o, const std::function<void(PairMaps&&)>& emit,
                                       int max_threads) {
    std::vector<std::int64_t> pts(ids);
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    const std::int32_t np = static_cast<std::int32_t>(pts.size());
    const i64 ncell = nx * ny;
    PairwiseResult out;
    out.resistances.assign(static_cast<std::size_t>(np) * np, -1.0);
    if (o.cum) out.cum.assign(static_cast<std::size_t>(ncell), 0.0);
    if (o.max) out.max.assign(static_cast<std::size_t>(ncell), -9999.0);
    std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
    for (std::int32_t i = 0; i < np; ++i)
        for (std::int32_t j = i + 1; j < np; ++j)
            if (!excluded[static_cast<std::size_t>(i) * np + j] && !excluded[static_cast<std::size_t>(j) * np + i])
                pairs.emplace_back(i, j);
    const int threads =
#if defined(_OPENMP)
        max_threads > 0 ? max_threads : omp_get_max_threads();
#else
        1;
    (void)max_threads;
#endif
    struct Done {
        PairwiseResult r;
        std::vector<PairMaps> maps;
    };
    const std::size_t batch = static_cast<std::size_t>(threads) * 2;
    bool failed = false;
    std::string error;
    for (std::size_t b0 = 0; b0 < pairs.size(); b0 += batch) {
        const std::size_t nb = std::min(batch, pairs.size() - b0);
        std::vector<Done> done(nb);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
        for (long long q = 0; q < static_cast<long long>(nb); ++q) {
            try {
                const auto [i, j] = pairs[b0 + static_cast<std::size_t>(q)];
                const std::vector<std::int64_t> poly = region_polymap(polymap, ncell, cells, ids, pts[i], pts[j]);
                const RasterGraph g = raster_graph(gmap, poly.data(), nx, ny, avg_resistances, four_neighbors);
                // The node of each region's first cell.
                std::vector<std::int32_t> nodes(2, -1);
                for (int e = 0; e < 2; ++e) {
                    const std::int64_t pt = pts[e == 0 ? i : j];
                    for (std::size_t x = 0; x < ids.size(); ++x)
                        if (ids[x] == pt) {
                            nodes[e] = g.nodemap[static_cast<std::size_t>(cells[x])];
                            break;
                        }
                }
                Done& d = done[static_cast<std::size_t>(q)];
                d.r = raster_pairwise(g, gmap, nodes, std::vector<std::uint8_t>(4, 0), o,
                                      [&](PairMaps&& m) { d.maps.push_back(std::move(m)); }, 1);
                for (PairMaps& m : d.maps) {
                    m.ci = m.ci == 0 ? i : j;
                    m.cj = m.cj == 0 ? i : j;
                }
            } catch (const std::exception& e) {
#pragma omp critical(regions_error)
                {
                    failed = true;
                    error = e.what();
                }
            }
        }
        if (failed) throw std::runtime_error(error);
        for (std::size_t q = 0; q < nb; ++q) {
            const auto [i, j] = pairs[b0 + q];
            Done& d = done[q];
            const double r = d.r.resistances[1];
            out.resistances[static_cast<std::size_t>(i) * np + j] = out.resistances[static_cast<std::size_t>(j) * np + i] = r;
            if (o.cum)
                for (i64 c = 0; c < ncell; ++c) out.cum[c] += d.r.cum[c];
            if (o.max)
                for (i64 c = 0; c < ncell; ++c) out.max[c] = std::max(out.max[c], d.r.max[c]);
            for (PairMaps& m : d.maps) emit(std::move(m));
        }
    }
    for (std::int32_t i = 0; i < np; ++i) out.resistances[static_cast<std::size_t>(i) * np + i] = 0.0;
    if (o.cum)
        for (double& c : out.cum) c = c < -9999 ? -9999 : c;
    return out;
}

}  // namespace circuit

namespace circuit {

namespace {

// resolve_conflicts for one problem: the sources and grounds after the policy,
// the finite grounds taken before it, and whether any node has one.
struct Resolved {
    std::vector<double> src, gnd, finite;
    bool any_finite = false;
};

Resolved resolve(std::vector<double> src, std::vector<double> gnd, ConflictPolicy policy) {
    const double inf = std::numeric_limits<double>::infinity();
    Resolved r;
    const std::size_t n = src.size();
    r.finite.assign(n, 0.0);
    for (std::size_t v = 0; v < n; ++v) {
        r.finite[v] = gnd[v] < inf ? gnd[v] : 0.0;
        if (r.finite[v] != 0) r.any_finite = true;
    }
    for (std::size_t v = 0; v < n; ++v) {
        if (!(src[v] != 0 && gnd[v] != 0)) continue;
        if (policy == ConflictPolicy::RemoveSource || policy == ConflictPolicy::RemoveAll) src[v] = 0.0;
        else if (policy == ConflictPolicy::RemoveGround) gnd[v] = 0.0;
    }
    for (std::size_t v = 0; v < n; ++v)
        if (gnd[v] == inf && src[v] > 0) gnd[v] = 0.0;
    r.src = std::move(src);
    r.gnd = std::move(gnd);
    return r;
}

}  // namespace

std::vector<NodeSolution> raster_advanced_many(const RasterGraph& g, const std::vector<std::vector<double>>& sources,
                                               const std::vector<std::vector<double>>& grounds,
                                               const std::vector<std::int32_t>& only_nodes, ConflictPolicy policy,
                                               int max_threads) {
    const double inf = std::numeric_limits<double>::infinity();
    const std::size_t np = sources.size();
    const std::int32_t n = g.n_nodes;
    const int threads =
#if defined(_OPENMP)
        max_threads > 0 ? max_threads : omp_get_max_threads();
#else
        1;
    (void)max_threads;
#endif
    std::vector<Resolved> res(np);
    std::vector<std::uint8_t> special(static_cast<std::size_t>(n), 0);
    for (std::size_t p = 0; p < np; ++p) {
        res[p] = resolve(sources[p], grounds[p], policy);
        for (std::int32_t v = 0; v < n; ++v)
            if (res[p].src[v] != 0 || res[p].gnd[v] != 0 || res[p].finite[v] != 0) special[v] = 1;
    }
    std::vector<NodeSolution> out(np);
    for (std::size_t p = 0; p < np; ++p) {
        out[p].voltage.assign(static_cast<std::size_t>(n), 0.0);
        out[p].current.assign(static_cast<std::size_t>(n), 0.0);
        for (std::int32_t v = 0; v < n; ++v)
            if (res[p].src[v] != 0 || res[p].gnd[v] != 0) out[p].active.push_back(v);
    }
    // Components, their nodes ascending.
    std::vector<std::int64_t> start(static_cast<std::size_t>(g.n_comp) + 1, 0);
    for (std::int32_t v = 0; v < n; ++v) ++start[static_cast<std::size_t>(g.comp[v]) + 1];
    for (std::int32_t k = 0; k < g.n_comp; ++k) start[k + 1] += start[k];
    std::vector<std::int32_t> members(static_cast<std::size_t>(n));
    {
        std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
        for (std::int32_t v = 0; v < n; ++v) members[static_cast<std::size_t>(fill[g.comp[v]]++)] = v;
    }
    // The problems to solve in every component (advanced_kernel's tests).
    std::vector<std::vector<std::size_t>> todo(static_cast<std::size_t>(g.n_comp));
    for (std::size_t p = 0; p < np; ++p) {
        const std::int32_t only = only_nodes[p];
        for (std::int32_t k = 0; k < g.n_comp; ++k) {
            if (only == -2 || (only >= 0 && g.comp[only] != k)) continue;
            double sum_s = 0.0, sum_g = 0.0;
            for (std::int64_t q = start[k]; q < start[k + 1]; ++q) {
                sum_s += res[p].src[members[q]];
                sum_g += res[p].gnd[members[q]];
            }
            if (sum_s != 0 && sum_g != 0) {
                todo[static_cast<std::size_t>(k)].push_back(p);
                out[p].solved = true;
            }
        }
    }

    for (std::int32_t k = 0; k < g.n_comp; ++k) {
        const std::vector<std::size_t>& probs = todo[static_cast<std::size_t>(k)];
        if (probs.empty()) continue;
        const std::int64_t p0 = start[k], p1 = start[k + 1];
        const std::int32_t* mem = members.data() + p0;
        const std::int64_t count = p1 - p0;
        std::vector<std::int32_t> F;
        for (std::int64_t q = 0; q < count; ++q)
            if (special[mem[q]]) F.push_back(mem[q]);
        const std::size_t nf = F.size();
        bool any_finite = false;
        for (std::size_t p : probs) any_finite = any_finite || res[p].any_finite;
        if (nf > 2048 || probs.size() < 2) {
            // Few problems or many special nodes: factorise every problem.
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) if (probs.size() > 1)
#endif
            for (long long t = 0; t < static_cast<long long>(probs.size()); ++t) {
                const std::size_t p = probs[static_cast<std::size_t>(t)];
                const Resolved& r = res[p];
                ComponentSystem cs;
                if (!factor_component(g, mem, count, r.any_finite ? r.finite.data() : nullptr,
                                      [&](std::int32_t v) { return r.gnd[v] == inf; }, cs))
                    throw std::runtime_error("a circuit component is not positive definite");
                if (!cs.nodes.empty()) {
                    const std::vector<std::int32_t>& perm = cs.sys.perm();
                    std::vector<double> b(cs.nodes.size()), y(cs.nodes.size());
                    for (std::size_t i = 0; i < cs.nodes.size(); ++i) b[i] = r.src[cs.nodes[perm[i]]];
                    cs.sys.solve(b.data(), y.data());
                    for (std::size_t i = 0; i < cs.nodes.size(); ++i) out[p].voltage[cs.nodes[perm[i]]] = y[i];
                }
                graph_node_currents(g, mem, count, out[p].voltage.data(), r.any_finite ? r.finite.data() : nullptr,
                                    out[p].current.data());
            }
            continue;
        }
        // One factorisation without the special nodes (the shift as CHOLMOD's
        // on every per-problem matrix), and the Schur complement on them.
        ComponentSystem cs;
        if (!factor_component(g, mem, count, nullptr, [&](std::int32_t v) { return special[v] != 0; }, cs))
            throw std::runtime_error("a circuit component is not positive definite");
        const std::size_t na = cs.nodes.size();
        std::unordered_map<std::int32_t, std::int32_t> uidx;
        {
            const std::vector<std::int32_t>& perm = cs.sys.perm();
            for (std::size_t e = 0; e < na; ++e) uidx.emplace(cs.nodes[perm[e]], static_cast<std::int32_t>(e));
        }
        std::unordered_map<std::int32_t, std::int32_t> fidx;
        for (std::size_t f = 0; f < nf; ++f) fidx.emplace(F[f], static_cast<std::int32_t>(f));
        // Z = A⁻¹ B (B: the edges from the special nodes into A, -w), by columns.
        Eigen::MatrixXd Z(static_cast<Eigen::Index>(na), static_cast<Eigen::Index>(nf));
        Z.setZero();
        for (std::size_t f = 0; f < nf; ++f)
            for (std::int64_t e = g.ptr[F[f]]; e < g.ptr[F[f] + 1]; ++e) {
                const auto it = uidx.find(g.adj[static_cast<std::size_t>(e)]);
                if (it != uidx.end()) Z(it->second, static_cast<Eigen::Index>(f)) -= g.w[static_cast<std::size_t>(e)];
            }
        Eigen::MatrixXd B = Z;  // keep B for the Schur complement
        if (na > 0) {
            std::vector<SolveScratch> scratch(static_cast<std::size_t>(threads));
            const long long groups = static_cast<long long>((nf + 7) / 8);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
            for (long long gi = 0; gi < groups; ++gi) {
                const int tid =
#if defined(_OPENMP)
                    omp_get_thread_num();
#else
                    0;
#endif
                const Eigen::Index c0 = static_cast<Eigen::Index>(gi * 8);
                const Eigen::Index kk = std::min<Eigen::Index>(8, static_cast<Eigen::Index>(nf) - c0);
                cs.sys.solve_many(Z.data() + c0 * static_cast<Eigen::Index>(na), kk,
                                  scratch[static_cast<std::size_t>(tid)]);
            }
        }
        // S = L_FF + shift - Bᵀ Z.
        Eigen::MatrixXd S = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(nf), static_cast<Eigen::Index>(nf));
        for (std::size_t f = 0; f < nf; ++f) {
            S(static_cast<Eigen::Index>(f), static_cast<Eigen::Index>(f)) = g.degree[F[f]];
            for (std::int64_t e = g.ptr[F[f]]; e < g.ptr[F[f] + 1]; ++e) {
                const auto it = fidx.find(g.adj[static_cast<std::size_t>(e)]);
                if (it != fidx.end()) S(static_cast<Eigen::Index>(f), it->second) -= g.w[static_cast<std::size_t>(e)];
            }
        }
        if (na > 0) S.noalias() -= B.transpose() * Z;
        for (std::size_t f = 0; f < nf; ++f) S(static_cast<Eigen::Index>(f), static_cast<Eigen::Index>(f)) += kShift;
        // Every problem: the free special nodes (not direct grounds) solve the
        // Schur system (plus their finite grounds); the rest follows from Z.
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
        for (long long t = 0; t < static_cast<long long>(probs.size()); ++t) {
            const std::size_t p = probs[static_cast<std::size_t>(t)];
            const Resolved& r = res[p];
            std::vector<Eigen::Index> U;
            for (std::size_t f = 0; f < nf; ++f)
                if (r.gnd[F[f]] != inf) U.push_back(static_cast<Eigen::Index>(f));
            const Eigen::Index nu = static_cast<Eigen::Index>(U.size());
            Eigen::MatrixXd Suu(nu, nu);
            Eigen::VectorXd rhs(nu);
            for (Eigen::Index a = 0; a < nu; ++a) {
                for (Eigen::Index b = 0; b < nu; ++b) Suu(a, b) = S(U[a], U[b]);
                if (r.any_finite) Suu(a, a) += r.finite[F[U[a]]];
                rhs(a) = r.src[F[U[a]]];
            }
            Eigen::VectorXd y = Suu.llt().solve(rhs);
            std::vector<double>& v = out[p].voltage;
            Eigen::VectorXd yf = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nf));
            for (Eigen::Index a = 0; a < nu; ++a) {
                v[F[U[a]]] = y(a);
                yf(U[a]) = y(a);
            }
            if (na > 0) {
                const Eigen::VectorXd x = -(Z * yf);
                const std::vector<std::int32_t>& perm = cs.sys.perm();
                for (std::size_t e = 0; e < na; ++e) v[cs.nodes[perm[e]]] = x(static_cast<Eigen::Index>(e));
            }
            graph_node_currents(g, mem, count, v.data(), r.any_finite ? r.finite.data() : nullptr,
                                out[p].current.data());
        }
    }
    return out;
}

}  // namespace circuit
