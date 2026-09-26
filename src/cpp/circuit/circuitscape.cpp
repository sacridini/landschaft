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
    std::unordered_map<std::int32_t, std::int32_t> local;
};

template <class Drop>
bool factor_component(const RasterGraph& g, const std::int32_t* members, std::int64_t count, const double* finite,
                      Drop drop, ComponentSystem& cs) {
    cs.nodes.clear();
    cs.local.clear();
    i64 x0 = g.nx, x1 = -1, y0 = g.ny, y1 = -1;
    for (std::int64_t p = 0; p < count; ++p) {
        const std::int32_t v = members[p];
        if (drop(v)) continue;
        cs.local.emplace(v, static_cast<std::int32_t>(cs.nodes.size()));
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
            const auto it = cs.local.find(g.adj[static_cast<std::size_t>(e)]);
            if (it == cs.local.end()) continue;
            adj.push_back(it->second);
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
    const bool shortcut = !o.maps() && !any_excluded;
    std::vector<double> voltmatrix, shortcut_res;
    if (shortcut) {
        voltmatrix.assign(static_cast<std::size_t>(np) * np, 0.0);
        shortcut_res.assign(static_cast<std::size_t>(np) * np, -1.0);
    }
    if (o.cum) out.cum.assign(static_cast<std::size_t>(ncell), 0.0);
    if (o.max) out.max.assign(static_cast<std::size_t>(ncell), -9999.0);
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
        std::unordered_map<std::int32_t, std::int32_t> member_of;
        for (std::int64_t p = p0; p < p1; ++p) member_of.emplace(members[p], static_cast<std::int32_t>(p - p0));
        for (std::size_t e = 0; e < m; ++e) elim[static_cast<std::size_t>(member_of[cs.nodes[perm[e]]])] = static_cast<std::int32_t>(e);

        // Batches of 8 right-hand sides per thread; results handled in job order.
        constexpr std::size_t kCols = 8;
        const std::size_t per_batch = kCols * static_cast<std::size_t>(threads) * 4;
        std::vector<SolveScratch> scratch(static_cast<std::size_t>(threads));
        const std::int64_t count = p1 - p0;
        for (std::size_t b0 = 0; b0 < jobs.size(); b0 += per_batch) {
            const std::size_t nb = std::min(per_batch, jobs.size() - b0);
            // Voltages by member (relative to the source), one column per job.
            std::vector<double> volt(static_cast<std::size_t>(count) * nb, 0.0);
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
                    const std::int32_t ei = elim[static_cast<std::size_t>(member_of.at(job.src))];
                    const std::int32_t ej = elim[static_cast<std::size_t>(member_of.at(job.dst))];
                    if (ei >= 0) col[ei] = -1.0;
                    if (ej >= 0) col[ej] = 1.0;
                }
                cs.sys.solve_many(rhs.data(), kk, scratch[static_cast<std::size_t>(tid)]);
                for (std::size_t j = j0; j < j1; ++j) {
                    const double* col = rhs.data() + (j - j0) * m;
                    double* vcol = volt.data() + j * static_cast<std::size_t>(count);
                    for (std::int64_t q = 0; q < count; ++q) {
                        const std::int32_t e = elim[static_cast<std::size_t>(q)];
                        vcol[q] = e >= 0 ? col[e] : 0.0;
                    }
                    const double vs = vcol[member_of.at(jobs[b0 + j].src)];
                    for (std::int64_t q = 0; q < count; ++q) vcol[q] -= vs;
                }
            }
            // handle_pair, in job order.
            std::vector<double> vnode, cnode;
            if (o.maps()) {
                vnode.assign(static_cast<std::size_t>(g.n_nodes), 0.0);
                cnode.assign(static_cast<std::size_t>(g.n_nodes), 0.0);
            }
            for (std::size_t j = 0; j < nb; ++j) {
                const Job& job = jobs[b0 + j];
                const double* vcol = volt.data() + j * static_cast<std::size_t>(count);
                const double resistance = vcol[member_of.at(job.dst)] - vcol[member_of.at(job.src)];
                if (o.maps()) {
                    for (std::int64_t q = 0; q < count; ++q) vnode[members[p0 + q]] = vcol[q];
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
                                voltmatrix[static_cast<std::size_t>(i) * np + cj] = 1 - (vcol[member_of.at(v)] / resistance);
                            }
                            continue;
                        }
                        if (!o.maps()) continue;
                        PairMaps pm;
                        pm.ci = ci;
                        pm.cj = cj;
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
    if (o.cum)
        for (double& c : out.cum) c = c < -9999 ? -9999 : c;
    if (o.max)
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
