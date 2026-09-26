// SPDX-License-Identifier: GPL-3.0-or-later
#include "conn/graph.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "lsm/parallel.hpp"

namespace conn {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr std::int32_t kUnreached = -1;

int thread_count() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Compressed rows from an edge list (each edge once, stored both ways).
Csr build_csr(std::int32_t n, const std::vector<std::int32_t>& ei, const std::vector<std::int32_t>& ej,
              const std::vector<double>* cost) {
    Csr g;
    g.n = n;
    g.start.assign(static_cast<std::size_t>(n) + 1, 0);
    for (std::size_t e = 0; e < ei.size(); ++e) {
        ++g.start[static_cast<std::size_t>(ei[e]) + 1];
        ++g.start[static_cast<std::size_t>(ej[e]) + 1];
    }
    for (std::int32_t v = 0; v < n; ++v) g.start[v + 1] += g.start[v];
    g.to.resize(ei.size() * 2);
    g.edge.resize(ei.size() * 2);
    g.eu = ei;
    g.ev = ej;
    if (cost) g.cost.resize(ei.size() * 2);
    std::vector<i64> fill(g.start.begin(), g.start.end() - 1);
    for (std::size_t e = 0; e < ei.size(); ++e) {
        const i64 a = fill[ei[e]]++;
        const i64 b = fill[ej[e]]++;
        g.to[a] = ej[e];
        g.to[b] = ei[e];
        g.edge[a] = g.edge[b] = static_cast<i64>(e);
        if (cost) g.cost[a] = g.cost[b] = (*cost)[e];
    }
    // Rows sorted by neighbour (edges are then found by binary search). Pairs
    // sorted by (i, j) already give sorted rows.
    lsm::parallel_for(n > 4096, n, [&](i64 v) {
        const i64 b = g.start[v], e = g.start[v + 1];
        if (std::is_sorted(g.to.begin() + b, g.to.begin() + e)) return;
        struct Slot {
            std::int32_t to;
            double cost;
            i64 edge;
        };
        std::vector<Slot> row;
        for (i64 x = b; x < e; ++x) row.push_back({g.to[x], cost ? g.cost[x] : 0.0, g.edge[x]});
        std::sort(row.begin(), row.end(), [](const Slot& p, const Slot& q) { return p.to < q.to; });
        for (i64 x = b; x < e; ++x) {
            g.to[x] = row[x - b].to;
            g.edge[x] = row[x - b].edge;
            if (cost) g.cost[x] = row[x - b].cost;
        }
    });
    return g;
}

// Position of the edge j-i in the (sorted) row of j, or -1.
inline i64 find_edge(const Csr& g, std::int32_t j, std::int32_t i) {
    const auto first = g.to.begin() + g.start[j], last = g.to.begin() + g.start[j + 1];
    const auto it = std::lower_bound(first, last, i);
    return (it != last && *it == i) ? static_cast<i64>(it - g.to.begin()) : -1;
}

inline double log2_ceil(i64 x) { return std::log2(static_cast<double>(x) + 1.0); }

void check_pairs(std::int32_t n, const Pairs& p) {
    if (p.i.size() != p.j.size() || p.i.size() != p.value.size()) throw std::invalid_argument("pair arrays differ in length");
    for (std::size_t e = 0; e < p.i.size(); ++e) {
        if (p.i[e] < 0 || p.j[e] < 0 || p.i[e] >= n || p.j[e] >= n || p.i[e] == p.j[e]) {
            throw std::invalid_argument("pairs must join two different nodes");
        }
    }
}

// Per-thread buffers of a source's shortest-path tree and its repairs.
// The K cheapest ways into every node (cost through each expanded
// neighbour), kept while the tree is built: a node cut off by a removal
// usually restarts from one of them instead of scanning all its neighbours.
constexpr int kCandidates = 4;

template <class D>
struct Workspace {
    std::vector<D> dist, repaired;
    std::vector<std::int32_t> parent, order, tin, tout, euler, child_start, child;
    std::vector<std::int32_t> sorted, affected, level_count;
    std::vector<double> bw_sigma, bw_d1, bw_d2;
    std::vector<char> flag;
    std::vector<double> cand_cost;
    std::vector<std::int32_t> cand_node;
    std::vector<std::int8_t> cand_len;
    Workspace(std::int32_t n, bool candidates)
        : dist(n), repaired(n), parent(n, kUnreached), tin(n, kUnreached), tout(n), child_start(static_cast<std::size_t>(n) + 1),
          flag(n, 0) {
        order.reserve(n);
        euler.reserve(n);
        child.reserve(n);
        if (candidates) {
            cand_cost.resize(static_cast<std::size_t>(n) * kCandidates);
            cand_node.resize(static_cast<std::size_t>(n) * kCandidates);
            cand_len.assign(n, 0);
        }
    }
    void clear() {
        for (const std::int32_t v : order) {
            parent[v] = kUnreached;
            tin[v] = kUnreached;
        }
        if (!cand_len.empty()) {
            for (const std::int32_t v : order) cand_len[v] = 0;
        }
        order.clear();
    }
    void add_candidate(std::int32_t u, std::int32_t via, double c) {
        const std::size_t base = static_cast<std::size_t>(u) * kCandidates;
        double* cc = cand_cost.data() + base;
        std::int32_t* cn = cand_node.data() + base;
        const int len = cand_len[u];
        if (len == kCandidates && !(c < cc[kCandidates - 1])) return;
        int p = len < kCandidates ? len : kCandidates - 1;
        if (len < kCandidates) cand_len[u] = static_cast<std::int8_t>(len + 1);
        while (p > 0 && cc[p - 1] > c) {
            cc[p] = cc[p - 1];
            cn[p] = cn[p - 1];
            --p;
        }
        cc[p] = c;
        cn[p] = via;
    }
    // Pre-order of the tree rooted at `root` over the settled nodes: the
    // subtree of v is euler[tin[v] .. tout[v]).
    void number_tree(std::int32_t root) {
        const std::size_t m = order.size();
        for (const std::int32_t v : order) child_start[v] = 0;
        // Counts go in child_start[v] of the settled nodes; a prefix sum over
        // `order` turns them into offsets into `child`.
        for (const std::int32_t v : order) {
            if (v != root) ++child_start[parent[v]];
        }
        std::int32_t sum = 0;
        for (const std::int32_t v : order) {
            const std::int32_t c = child_start[v];
            child_start[v] = sum;
            sum += c;
        }
        child.resize(m);
        // Fill children (tin doubles as the fill cursor, then is reset).
        for (const std::int32_t v : order) tin[v] = child_start[v];
        for (const std::int32_t v : order) {
            if (v != root) child[tin[parent[v]]++] = v;
        }
        // Children of v: child[child_start[v] .. tin[v]).
        for (const std::int32_t v : order) tout[v] = tin[v];  // end of the child list
        euler.clear();
        stack.clear();
        stack.push_back(root);
        while (!stack.empty()) {
            const std::int32_t v = stack.back();
            stack.pop_back();
            if (v < 0) {
                tout[~v] = static_cast<std::int32_t>(euler.size());
                continue;
            }
            const std::int32_t first = child_start[v];
            const std::int32_t last = tout[v];
            tin[v] = static_cast<std::int32_t>(euler.size());
            euler.push_back(v);
            stack.push_back(~v);
            for (std::int32_t c = last; c > first; --c) stack.push_back(child[c - 1]);
        }
    }
    std::vector<std::int32_t> stack;
};

// Contributions of one source to the per-node sums (merged in source order so
// the result does not depend on the threads).
enum ContributionKind : std::uint8_t { kNode = 0, kEdge = 1, kBetween = 2 };

struct Contribution {
    i64 index;  // a node, or an edge for kEdge
    std::uint8_t kind;
    double first, second;
};

template <class Body, class Merge>
void over_sources(std::int32_t n, bool importance, Body&& body, Merge&& merge) {
    const bool parallel = n > 64;
    if (!importance) {
        lsm::parallel_for_dynamic(parallel, n, [&](i64 s) {
            std::vector<Contribution> none;
            body(static_cast<std::int32_t>(s), none);
        });
        return;
    }
    const std::int32_t block = std::max<std::int32_t>(64, 16 * thread_count());
    std::vector<std::vector<Contribution>> out(static_cast<std::size_t>(block));
    for (std::int32_t b0 = 0; b0 < n; b0 += block) {
        const std::int32_t count = std::min(block, n - b0);
        lsm::parallel_for_dynamic(parallel, count, [&](i64 s) {
            out[s].clear();
            body(b0 + static_cast<std::int32_t>(s), out[s]);
        });
        for (std::int32_t s = 0; s < count; ++s) merge(b0 + s, out[s]);
    }
}

}  // namespace

double pair_length(double v, const Scenario& s, bool links, bool cut) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (std::isnan(v)) return nan;
    if (links) {
        bool link = false;
        switch (s.kind) {
            case Kind::Distance: link = s.conefor ? v <= s.threshold : v < s.threshold; break;
            case Kind::Probability: link = v >= s.threshold; break;
            case Kind::Link: link = v != 0; break;
        }
        return link ? 1.0 : nan;
    }
    if (s.kind == Kind::Link) throw std::invalid_argument("probabilities need distances or probabilities");
    double p = v;
    if (s.kind == Kind::Distance) {
        const double log_p = std::log(s.probability);
        const double k = log_p / s.threshold;
        p = s.conefor ? std::exp(k * v) : std::exp((v * log_p) / s.threshold);
    }
    const bool keep = !cut || (s.conefor ? p > s.pij_min : (p >= s.pij_min && p > 0));
    return keep && p > 0 ? -std::log(p) : nan;
}

Csr make_link_graph(std::int32_t n, const Pairs& pairs, const Scenario& s) {
    check_pairs(n, pairs);
    std::vector<std::int32_t> ei, ej;
    for (std::size_t e = 0; e < pairs.i.size(); ++e) {
        const double v = pairs.value[e];
        if (std::isnan(v)) continue;
        bool link = false;
        switch (s.kind) {
            case Kind::Distance: link = s.conefor ? v <= s.threshold : v < s.threshold; break;
            case Kind::Probability: link = v >= s.threshold; break;
            case Kind::Link: link = v != 0; break;
        }
        if (link) {
            ei.push_back(pairs.i[e]);
            ej.push_back(pairs.j[e]);
        }
    }
    return build_csr(n, ei, ej, nullptr);
}

Csr make_prob_graph(std::int32_t n, const Pairs& pairs, const Scenario& s, const double* area,
                    std::vector<double>* f_row, std::vector<double>* awf_row) {
    check_pairs(n, pairs);
    if (s.kind == Kind::Link) throw std::invalid_argument("probabilities need distances or probabilities");
    const double log_p = std::log(s.probability);
    const double k = log_p / s.threshold;  // Conefor's constant
    if (f_row) f_row->assign(n, 0.0);
    if (awf_row) awf_row->assign(n, 0.0);
    std::vector<std::int32_t> ei, ej;
    std::vector<double> cost;
    for (std::size_t e = 0; e < pairs.i.size(); ++e) {
        const double v = pairs.value[e];
        if (std::isnan(v)) continue;
        double p = v;
        if (s.kind == Kind::Distance) p = s.conefor ? std::exp(k * v) : std::exp((v * log_p) / s.threshold);
        const std::int32_t a = pairs.i[e], b = pairs.j[e];
        if (f_row) {
            (*f_row)[a] += p;
            (*f_row)[b] += p;
        }
        if (awf_row) {
            (*awf_row)[a] += area[b] * p;
            (*awf_row)[b] += area[a] * p;
        }
        const bool keep = s.conefor ? p > s.pij_min : (p >= s.pij_min && p > 0);
        if (keep) {
            ei.push_back(a);
            ej.push_back(b);
            cost.push_back(-std::log(p));
        }
    }
    return build_csr(n, ei, ej, &cost);
}

LinkResult analyse_links(const Csr& g, const double* area, bool importance, bool edges, bool betweenness) {
    const std::int32_t n = g.n;
    LinkResult r;
    r.nl = g.n_edges();
    r.component.assign(n, kUnreached);

    // Components and articulation points (iterative Tarjan): the pieces a
    // component splits into without each node, with their attribute sums.
    std::vector<std::int32_t> disc(n, -1), low(n), dfs_parent(n, kUnreached), sep_count(n, 0);
    std::vector<i64> cursor(n);
    std::vector<double> sub(n), sep_area(n, 0.0), sep_sq(n, 0.0), comp_area;
    std::vector<std::int32_t> stack;
    std::int32_t time = 0;
    for (std::int32_t root = 0; root < n; ++root) {
        if (disc[root] >= 0) continue;
        const std::int32_t c = r.nc++;
        disc[root] = low[root] = time++;
        cursor[root] = g.start[root];
        sub[root] = area[root];
        r.component[root] = c;
        stack.push_back(root);
        while (!stack.empty()) {
            const std::int32_t v = stack.back();
            if (cursor[v] < g.start[v + 1]) {
                const std::int32_t u = g.to[cursor[v]++];
                if (disc[u] < 0) {
                    dfs_parent[u] = v;
                    disc[u] = low[u] = time++;
                    cursor[u] = g.start[u];
                    sub[u] = area[u];
                    r.component[u] = c;
                    stack.push_back(u);
                } else if (u != dfs_parent[v]) {
                    low[v] = std::min(low[v], disc[u]);
                }
            } else {
                stack.pop_back();
                const std::int32_t p = dfs_parent[v];
                if (p != kUnreached) {
                    low[p] = std::min(low[p], low[v]);
                    sub[p] += sub[v];
                    if (low[v] >= disc[p]) {
                        ++sep_count[p];
                        sep_area[p] += sub[v];
                        sep_sq[p] += sub[v] * sub[v];
                    }
                }
            }
        }
        comp_area.push_back(sub[root]);
    }
    for (const double s : comp_area) r.comp_sq += s * s;

    if (importance) {
        r.degree.resize(n);
        r.pieces.resize(n);
        r.pieces_sq.resize(n);
        for (std::int32_t v = 0; v < n; ++v) {
            r.degree[v] = g.start[v + 1] - g.start[v];
            if (dfs_parent[v] == kUnreached) {  // a root: every child is a piece
                r.pieces[v] = sep_count[v];
                r.pieces_sq[v] = sep_sq[v];
            } else {
                const double rest = comp_area[r.component[v]] - area[v] - sep_area[v];
                r.pieces[v] = sep_count[v] + 1;
                r.pieces_sq[v] = sep_sq[v] + rest * rest;
            }
        }
        r.iic_row.assign(n, 0.0);
        r.unreach_a.assign(n, 0.0);
        r.h_row.assign(n, 0.0);
        r.iic_conn.assign(n, 0.0);
        r.h_conn.assign(n, 0.0);
    }
    if (edges) {
        r.edge_iic.assign(static_cast<std::size_t>(g.n_edges()), 0.0);
        r.edge_h.assign(static_cast<std::size_t>(g.n_edges()), 0.0);
    }
    if (betweenness) {
        r.bc.assign(n, 0.0);
        r.bc_iic.assign(n, 0.0);
    }

    double total = 0;
    for (std::int32_t v = 0; v < n; ++v) total += area[v];
    std::vector<double> row(n), hrow(n), direct(n), reach(n);

    std::vector<Workspace<std::int32_t>*> spaces(static_cast<std::size_t>(thread_count()), nullptr);
    auto workspace = [&]() -> Workspace<std::int32_t>& {
#if defined(_OPENMP)
        const int t = omp_in_parallel() ? omp_get_thread_num() : 0;
#else
        const int t = 0;
#endif
        if (!spaces[t]) spaces[t] = new Workspace<std::int32_t>(n, false);
        return *spaces[t];
    };
    struct Cleanup {
        std::vector<Workspace<std::int32_t>*>& s;
        ~Cleanup() {
            for (auto* w : s) delete w;
        }
    } cleanup{spaces};

    using Heap = std::priority_queue<std::pair<std::int32_t, std::int32_t>, std::vector<std::pair<std::int32_t, std::int32_t>>,
                                     std::greater<>>;
    constexpr std::int32_t none = std::numeric_limits<std::int32_t>::max();

    // New hop counts of the subtree positions [lo, hi) once `blocked` (a node,
    // or the edge block_node - block_j when block_j >= 0) is removed; returns
    // the changes of sum a_j / (1 + nl) and of sum 1 / nl over the subtree.
    // A node keeps its level when a neighbour that keeps its own sits one
    // level up; the others are found again by a Dijkstra seeded from their
    // untouched neighbours.
    auto repair = [&](Workspace<std::int32_t>& w, Heap& heap, std::int32_t lo, std::int32_t hi, std::int32_t block_node,
                      std::int32_t block_j) -> std::pair<double, double> {
        auto blocked = [&](std::int32_t j, std::int32_t u) { return u == block_node && (block_j < 0 || j == block_j); };
        std::int32_t base = none, top = 0;
        for (std::int32_t pos = lo; pos < hi; ++pos) {
            base = std::min(base, w.dist[w.euler[pos]]);
            top = std::max(top, w.dist[w.euler[pos]]);
        }
        std::int32_t* count = w.level_count.data();
        for (std::int32_t pos = lo; pos < hi; ++pos) ++count[w.dist[w.euler[pos]] - base + 1];
        for (std::int32_t l = 1; l <= top - base + 1; ++l) count[l] += count[l - 1];
        w.sorted.resize(static_cast<std::size_t>(hi - lo));
        for (std::int32_t pos = lo; pos < hi; ++pos) {
            const std::int32_t j = w.euler[pos];
            w.sorted[count[w.dist[j] - base]++] = j;
        }
        std::fill(count, count + (top - base + 2), 0);
        w.affected.clear();
        for (const std::int32_t j : w.sorted) {
            const std::int32_t up = w.dist[j] - 1;
            bool keeps = false;
            for (i64 e = g.start[j]; e < g.start[j + 1]; ++e) {
                const std::int32_t u = g.to[e];
                if (w.dist[u] == up && !w.flag[u] && !blocked(j, u)) {
                    keeps = true;
                    break;
                }
            }
            if (!keeps) {
                w.flag[j] = 1;
                w.affected.push_back(j);
            }
        }
        if (w.affected.empty()) return {0.0, 0.0};
        for (const std::int32_t j : w.affected) {
            std::int32_t best = none;
            for (i64 e = g.start[j]; e < g.start[j + 1]; ++e) {
                const std::int32_t u = g.to[e];
                if (w.flag[u] || blocked(j, u)) continue;
                best = std::min(best, w.dist[u] + 1);
            }
            w.repaired[j] = best;
            if (best != none) heap.emplace(best, j);
        }
        while (!heap.empty()) {
            const auto [d, j] = heap.top();
            heap.pop();
            if (d != w.repaired[j]) continue;
            for (i64 e = g.start[j]; e < g.start[j + 1]; ++e) {
                const std::int32_t u = g.to[e];
                if (!w.flag[u] || w.repaired[u] <= d + 1) continue;
                w.repaired[u] = d + 1;
                heap.emplace(d + 1, u);
            }
        }
        double d_iic = 0, d_h = 0;
        for (const std::int32_t j : w.affected) {
            w.flag[j] = 0;
            const std::int32_t before = w.dist[j], after = w.repaired[j];
            if (after == before) continue;
            d_iic += area[j] * (1.0 / (1.0 + before) - (after == none ? 0.0 : 1.0 / (1.0 + after)));
            d_h += 1.0 / before - (after == none ? 0.0 : 1.0 / after);
        }
        return {d_iic, d_h};
    };

    auto body = [&](std::int32_t src, std::vector<Contribution>& out) {
        Workspace<std::int32_t>& w = workspace();
        w.clear();
        w.dist[src] = 0;
        w.parent[src] = src;
        w.order.push_back(src);
        for (std::size_t q = 0; q < w.order.size(); ++q) {
            const std::int32_t v = w.order[q];
            for (i64 e = g.start[v]; e < g.start[v + 1]; ++e) {
                const std::int32_t u = g.to[e];
                if (w.parent[u] == kUnreached) {
                    w.parent[u] = v;
                    w.dist[u] = w.dist[v] + 1;
                    w.order.push_back(u);
                }
            }
        }
        double s_row = area[src], s_h = 0, s_direct = 0, s_reach = area[src];
        for (std::size_t q = 1; q < w.order.size(); ++q) {
            const std::int32_t v = w.order[q];
            const std::int32_t nl = w.dist[v];
            s_row += area[v] / (1.0 + nl);
            s_h += 1.0 / nl;
            if (nl == 1) s_direct += area[v];
            s_reach += area[v];
        }
        row[src] = s_row;
        hrow[src] = s_h;
        direct[src] = s_direct;
        reach[src] = s_reach;
        if (betweenness && w.order.size() > 2) {
            // Brandes: shortest-path counts, then dependencies back from the far
            // nodes, unweighted (BC) and weighted by a_s a_t / (1 + nl) (BC(IIC)).
            std::vector<double>& sigma = w.bw_sigma;
            std::vector<double>& d1 = w.bw_d1;
            std::vector<double>& d2 = w.bw_d2;
            if (sigma.size() < static_cast<std::size_t>(n)) {
                sigma.assign(n, 0.0);
                d1.assign(n, 0.0);
                d2.assign(n, 0.0);
            }
            for (const std::int32_t v : w.order) {
                sigma[v] = v == src ? 1.0 : 0.0;
                d1[v] = d2[v] = 0.0;
            }
            for (std::size_t q = 1; q < w.order.size(); ++q) {
                const std::int32_t v = w.order[q];
                for (i64 e = g.start[v]; e < g.start[v + 1]; ++e) {
                    const std::int32_t u = g.to[e];
                    if (w.dist[u] == w.dist[v] - 1) sigma[v] += sigma[u];
                }
            }
            for (std::size_t q = w.order.size(); q-- > 1;) {
                const std::int32_t v = w.order[q];
                const double own = area[src] * area[v] / (1.0 + w.dist[v]);
                for (i64 e = g.start[v]; e < g.start[v + 1]; ++e) {
                    const std::int32_t u = g.to[e];
                    if (w.dist[u] != w.dist[v] - 1 || u == src) continue;
                    const double c = sigma[u] / sigma[v];
                    d1[u] += c * (1.0 + d1[v]);
                    d2[u] += c * (own + d2[v]);
                }
                if (d1[v] != 0 || d2[v] != 0) out.push_back({v, kBetween, d1[v], d2[v]});
            }
        }
        if ((!importance && !edges) || w.order.size() < 2) return;

        w.number_tree(src);
        Heap heap;
        if (w.level_count.size() < w.order.size() + 2) w.level_count.assign(w.order.size() + 2, 0);
        if (importance) {
            for (const std::int32_t k : w.order) {
                if (k == src) continue;
                const std::int32_t lo = w.tin[k], hi = w.tout[k];
                if (hi - lo <= 1) continue;  // a leaf: no path runs through it
                const auto [d_iic, d_h] = repair(w, heap, lo + 1, hi, k, -1);
                if (d_iic != 0 || d_h != 0) out.push_back({k, kNode, area[src] * d_iic, d_h});
            }
        }
        if (edges) {
            for (const std::int32_t c : w.order) {
                if (c == src) continue;
                const std::int32_t p = w.parent[c];
                const auto [d_iic, d_h] = repair(w, heap, w.tin[c], w.tout[c], p, c);
                if (d_iic != 0 || d_h != 0) out.push_back({g.edge[find_edge(g, c, p)], kEdge, area[src] * d_iic, d_h});
            }
        }
    };
    over_sources(n, importance || edges || betweenness, body, [&](std::int32_t, const std::vector<Contribution>& c) {
        for (const Contribution& x : c) {
            if (x.kind == kEdge) {
                r.edge_iic[x.index] += x.first;
                r.edge_h[x.index] += x.second;
            } else if (x.kind == kBetween) {
                r.bc[x.index] += x.first;
                r.bc_iic[x.index] += x.second;
            } else {
                r.iic_conn[x.index] += x.first;
                r.h_conn[x.index] += x.second;
            }
        }
    });

    for (std::int32_t v = 0; v < n; ++v) {
        r.iic_reach += area[v] * row[v];
        r.h += hrow[v];
        r.iic_direct += area[v] * direct[v] / 2;
        r.unreach_aa += area[v] * (total - reach[v]);
    }
    r.h /= 2;
    if (importance) {
        for (std::int32_t v = 0; v < n; ++v) {
            r.iic_row[v] = row[v];
            r.h_row[v] = hrow[v];
            r.unreach_a[v] = total - reach[v];
            r.h_conn[v] /= 2;
        }
    }
    for (double& h : r.edge_h) h /= 2;
    return r;
}

ProbResult analyse_probabilities(const Csr& g, const double* area, double expand_limit, bool importance, bool edges,
                                 bool betweenness) {
    const std::int32_t n = g.n;
    ProbResult r;
    std::vector<double> row(n), direct(n);
    if (importance) r.pc_conn.assign(n, 0.0);
    if (edges) r.edge_pc.assign(static_cast<std::size_t>(g.n_edges()), 0.0);
    if (betweenness) r.bc_pc.assign(n, 0.0);
    const bool repairs = importance || edges;

    std::vector<Workspace<double>*> spaces(static_cast<std::size_t>(thread_count()), nullptr);
    auto workspace = [&]() -> Workspace<double>& {
#if defined(_OPENMP)
        const int t = omp_in_parallel() ? omp_get_thread_num() : 0;
#else
        const int t = 0;
#endif
        if (!spaces[t]) spaces[t] = new Workspace<double>(n, repairs);  // candidates only for repairs
        return *spaces[t];
    };
    struct Cleanup {
        std::vector<Workspace<double>*>& s;
        ~Cleanup() {
            for (auto* w : s) delete w;
        }
    } cleanup{spaces};

    using Heap = std::priority_queue<std::pair<double, std::int32_t>, std::vector<std::pair<double, std::int32_t>>,
                                     std::greater<>>;

    // New costs of the subtree positions [lo, hi) once `blocked` (a node, or
    // the edge block_node - block_j) is removed: seeded from the K best ways
    // into each node that avoid it (or all neighbours when they do not), then
    // a Dijkstra inside the subtree. Returns the change of sum a_j p*_j.
    auto repair = [&](Workspace<double>& w, Heap& heap, std::int32_t lo, std::int32_t hi, std::int32_t block_node,
                      std::int32_t block_j) -> double {
        auto blocked = [&](std::int32_t j, std::int32_t u) { return u == block_node && (block_j < 0 || j == block_j); };
        auto inside = [&](std::int32_t v) { return w.tin[v] >= lo && w.tin[v] < hi; };
        const std::int32_t size = hi - lo;
        for (std::int32_t pos = lo; pos < hi; ++pos) {
            const std::int32_t j = w.euler[pos];
            double best = kInf;
            const std::size_t cb = static_cast<std::size_t>(j) * kCandidates;
            const int len = w.cand_len[j];
            bool found = false;
            for (int c = 0; c < len; ++c) {
                const std::int32_t u = w.cand_node[cb + c];
                if (blocked(j, u) || inside(u)) continue;
                best = w.cand_cost[cb + c];
                found = true;
                break;
            }
            if (!found && len == kCandidates) {  // all K cut off: every neighbour
                for (i64 e = g.start[j]; e < g.start[j + 1]; ++e) {
                    const std::int32_t u = g.to[e];
                    if (blocked(j, u) || inside(u) || w.dist[u] > expand_limit) continue;
                    best = std::min(best, w.dist[u] + g.cost[e]);
                }
            }
            w.repaired[j] = best;
            if (best < kInf) heap.emplace(best, j);
        }
        while (!heap.empty()) {
            const auto [d, j] = heap.top();
            heap.pop();
            if (d != w.repaired[j] || d > expand_limit) continue;
            const i64 degree = g.start[j + 1] - g.start[j];
            if (static_cast<double>(size) * log2_ceil(degree) < static_cast<double>(degree)) {
                // Few nodes cut off: look their edges up in the row of j.
                for (std::int32_t pos = lo; pos < hi; ++pos) {
                    const std::int32_t u = w.euler[pos];
                    if (u == j) continue;
                    const i64 e = find_edge(g, j, u);
                    if (e < 0) continue;
                    const double nd = d + g.cost[e];
                    if (nd < w.repaired[u]) {
                        w.repaired[u] = nd;
                        heap.emplace(nd, u);
                    }
                }
                continue;
            }
            for (i64 e = g.start[j]; e < g.start[j + 1]; ++e) {
                const std::int32_t u = g.to[e];
                if (!inside(u)) continue;
                const double nd = d + g.cost[e];
                if (nd < w.repaired[u]) {
                    w.repaired[u] = nd;
                    heap.emplace(nd, u);
                }
            }
        }
        double change = 0;
        for (std::int32_t pos = lo; pos < hi; ++pos) {
            const std::int32_t j = w.euler[pos];
            if (w.repaired[j] == w.dist[j]) continue;
            change += area[j] * (std::exp(-w.dist[j]) - std::exp(-w.repaired[j]));
        }
        return change;
    };

    auto body = [&](std::int32_t src, std::vector<Contribution>& out) {
        Workspace<double>& w = workspace();
        w.clear();
        // Dijkstra over -log p. Tentative costs live in `repaired` until a node
        // is settled (tin = 0 marks settled nodes, parent reached ones); every
        // reached node is settled, so clear() resets them all through `order`.
        Heap heap;
        w.repaired[src] = 0;
        w.parent[src] = src;
        heap.emplace(0.0, src);
        while (!heap.empty()) {
            const auto [d, v] = heap.top();
            heap.pop();
            if (w.tin[v] != kUnreached || d != w.repaired[v]) continue;
            w.tin[v] = 0;  // settled
            w.dist[v] = d;
            w.order.push_back(v);
            if (d > expand_limit) continue;
            for (i64 e = g.start[v]; e < g.start[v + 1]; ++e) {
                const std::int32_t u = g.to[e];
                const double nd = d + g.cost[e];
                if (repairs) w.add_candidate(u, v, nd);
                if (w.tin[u] != kUnreached) continue;
                if (w.parent[u] != kUnreached && nd >= w.repaired[u]) continue;
                w.parent[u] = v;
                w.repaired[u] = nd;
                heap.emplace(nd, u);
            }
        }
        double s_row = 0, s_direct = 0;
        for (const std::int32_t v : w.order) s_row += area[v] * std::exp(-w.dist[v]);
        for (i64 e = g.start[src]; e < g.start[src + 1]; ++e) s_direct += area[g.to[e]] * std::exp(-g.cost[e]);
        row[src] = s_row;
        direct[src] = s_direct;
        if (betweenness && w.order.size() > 2) {
            // BC(PC): along the most probable path Dijkstra kept (its parents),
            // from the far nodes back towards the source.
            std::vector<double>& d = w.bw_d1;
            if (d.size() < static_cast<std::size_t>(n)) d.assign(n, 0.0);
            for (const std::int32_t v : w.order) d[v] = 0.0;
            for (std::size_t q = w.order.size(); q-- > 1;) {
                const std::int32_t v = w.order[q];
                const std::int32_t p = w.parent[v];
                if (p != src) d[p] += area[src] * area[v] * std::exp(-w.dist[v]) + d[v];
                if (d[v] != 0) out.push_back({v, kBetween, d[v], 0.0});
            }
        }
        if (!repairs || w.order.size() < 2) return;

        w.number_tree(src);
        if (importance) {
            for (const std::int32_t k : w.order) {
                if (k == src) continue;
                const std::int32_t lo = w.tin[k], hi = w.tout[k];
                if (hi - lo <= 1) continue;
                const double change = repair(w, heap, lo + 1, hi, k, -1);
                if (change != 0) out.push_back({k, kNode, area[src] * change, 0.0});
            }
        }
        if (edges) {
            for (const std::int32_t c : w.order) {
                if (c == src) continue;
                const std::int32_t p = w.parent[c];
                const double change = repair(w, heap, w.tin[c], w.tout[c], p, c);
                if (change != 0) out.push_back({g.edge[find_edge(g, c, p)], kEdge, area[src] * change, 0.0});
            }
        }
    };
    over_sources(n, repairs || betweenness, body, [&](std::int32_t, const std::vector<Contribution>& c) {
        for (const Contribution& x : c) {
            if (x.kind == kEdge) r.edge_pc[x.index] += x.first;
            else if (x.kind == kBetween) r.bc_pc[x.index] += x.first;
            else r.pc_conn[x.index] += x.first;
        }
    });

    for (std::int32_t v = 0; v < n; ++v) {
        r.pc_num += area[v] * row[v];
        r.pc_direct += area[v] * direct[v];
    }
    if (importance) r.pc_row = std::move(row);
    return r;
}

std::vector<double> all_pairs(const Csr& g, bool hops, double expand_limit) {
    const std::int32_t n = g.n;
    std::vector<double> out(static_cast<std::size_t>(n) * n, kInf);
    lsm::parallel_for_dynamic(n > 64, n, [&](i64 s) {
        double* d = out.data() + s * n;
        std::vector<char> done(n, 0);
        using Item = std::pair<double, std::int32_t>;
        std::priority_queue<Item, std::vector<Item>, std::greater<>> heap;
        d[s] = 0;
        heap.emplace(0.0, static_cast<std::int32_t>(s));
        while (!heap.empty()) {
            const auto [dv, v] = heap.top();
            heap.pop();
            if (done[v] || dv != d[v]) continue;
            done[v] = 1;
            if (dv > expand_limit) continue;
            for (i64 e = g.start[v]; e < g.start[v + 1]; ++e) {
                const std::int32_t u = g.to[e];
                const double nd = dv + (hops ? 1.0 : g.cost[e]);
                if (!done[u] && nd < d[u]) {
                    d[u] = nd;
                    heap.emplace(nd, u);
                }
            }
        }
    });
    return out;
}

namespace {

// f of a path length: 1 / (1 + hops) for links, exp(-cost) for probabilities (0 without a path).
inline double gain_f(double x, bool hops) {
    if (!(x < kInf)) return 0.0;
    return hops ? 1.0 / (1.0 + x) : std::exp(-x);
}

}  // namespace

Gains addition_gains(const std::vector<double>& D, std::int32_t n, const double* area, const i64* cand_start,
                     const std::int32_t* cand_node, const double* cand_len, std::int32_t n_cand, const double* cand_area,
                     bool hops) {
    Gains out;
    out.num.assign(n_cand, 0.0);
    out.h.assign(n_cand, 0.0);
    lsm::parallel_for_dynamic(n_cand > 4, n_cand, [&](i64 k) {
        // Lengths from the new node through its edges.
        std::vector<double> dk(n, kInf);
        for (i64 e = cand_start[k]; e < cand_start[k + 1]; ++e) {
            const double* row = D.data() + static_cast<std::size_t>(cand_node[e]) * n;
            const double w = hops ? 1.0 : cand_len[e];
            for (std::int32_t j = 0; j < n; ++j) dk[j] = std::min(dk[j], w + row[j]);
        }
        const double ak = cand_area[k];
        double own = 0, own_h = 0;
        for (std::int32_t j = 0; j < n; ++j) {
            own += area[j] * gain_f(dk[j], hops);
            if (hops && dk[j] < kInf) own_h += 1.0 / dk[j];
        }
        double pairs = 0, pairs_h = 0;
        for (std::int32_t i = 0; i < n; ++i) {
            if (!(dk[i] < kInf)) continue;
            const double* row = D.data() + static_cast<std::size_t>(i) * n;
            double acc = 0, acc_h = 0;
            for (std::int32_t j = 0; j < n; ++j) {
                const double t = dk[i] + dk[j];
                if (t < row[j]) {
                    acc += area[j] * (gain_f(t, hops) - gain_f(row[j], hops));
                    if (hops && j != i) acc_h += 1.0 / t - (row[j] < kInf ? 1.0 / row[j] : 0.0);
                }
            }
            pairs += area[i] * acc;
            pairs_h += acc_h;
        }
        out.num[k] = ak * ak + 2 * ak * own + pairs;
        out.h[k] = own_h + pairs_h / 2;
    });
    return out;
}

Gains edge_gains(const std::vector<double>& D, std::int32_t n, const double* area, const std::int32_t* ea,
                 const std::int32_t* eb, const double* elen, std::int32_t n_edges, bool hops) {
    Gains out;
    out.num.assign(n_edges, 0.0);
    out.h.assign(n_edges, 0.0);
    lsm::parallel_for_dynamic(n_edges > 4, n_edges, [&](i64 k) {
        const std::int32_t a = ea[k], b = eb[k];
        const double w = hops ? 1.0 : elen[k];
        const double* ra = D.data() + static_cast<std::size_t>(a) * n;
        const double* rb = D.data() + static_cast<std::size_t>(b) * n;
        double num = 0, h = 0;
        for (std::int32_t i = 0; i < n; ++i) {
            const double* row = D.data() + static_cast<std::size_t>(i) * n;
            const double via_a = row[a] + w, via_b = row[b] + w;
            if (!(via_a < kInf) && !(via_b < kInf)) continue;
            double acc = 0, acc_h = 0;
            for (std::int32_t j = 0; j < n; ++j) {
                const double t = std::min(via_a + rb[j], via_b + ra[j]);
                if (t < row[j]) {
                    acc += area[j] * (gain_f(t, hops) - gain_f(row[j], hops));
                    if (hops && j != i) acc_h += 1.0 / t - (row[j] < kInf ? 1.0 / row[j] : 0.0);
                }
            }
            num += area[i] * acc;
            h += acc_h;
        }
        out.num[k] = num;
        out.h[k] = h / 2;
    });
    return out;
}

}  // namespace conn

namespace conn {

namespace {

// igraph_cmp_epsilon: equal within a relative tolerance (igraph's shortest
// paths use 1e-10).
int cmp_epsilon(double a, double b, double eps) {
    if (a == b) return 0;
    const double diff = a - b;
    const double abs_diff = std::fabs(diff);
    const double sum = std::fabs(a) + std::fabs(b);
    if (a == 0 || b == 0 || sum < std::numeric_limits<double>::min()) {
        return abs_diff < eps * std::numeric_limits<double>::min() ? 0 : (diff < 0 ? -1 : 1);
    }
    if (!std::isfinite(sum)) return abs_diff < eps * std::fabs(a) + eps * std::fabs(b) ? 0 : (diff < 0 ? -1 : 1);
    return abs_diff / sum < eps ? 0 : (diff < 0 ? -1 : 1);
}

}  // namespace

Centrality centrality(const Csr& g, bool unit) {
    const std::int32_t n = g.n;
    constexpr double eps = 1e-10;  // IGRAPH_SHORTEST_PATH_EPSILON
    Centrality out;
    out.betweenness.assign(n, 0.0);
    out.closeness_sum.assign(n, 0.0);
    out.reached.assign(n, 0);
    std::vector<double> closeness(n, 0.0);
    std::vector<std::int32_t> reached(n, 0);
    const std::int32_t block = std::max<std::int32_t>(64, 16 * thread_count());
    std::vector<std::vector<std::pair<std::int32_t, double>>> parts(static_cast<std::size_t>(block));
    for (std::int32_t b0 = 0; b0 < n; b0 += block) {
        const std::int32_t count = std::min(block, n - b0);
        lsm::parallel_for_dynamic(n > 64, count, [&](i64 k) {
            const std::int32_t src = b0 + static_cast<std::int32_t>(k);
            auto& part = parts[static_cast<std::size_t>(k)];
            part.clear();
            // igraph's weighted single-source search: distances + 1, parents of
            // equal (within eps) paths, geodesic counts.
            std::vector<double> dist(n, 0.0), nrgeo(n, 0.0), delta(n, 0.0);
            std::vector<std::vector<std::int32_t>> parents(n);
            std::vector<std::int32_t> stack;
            using Item = std::pair<double, std::int32_t>;
            std::priority_queue<Item, std::vector<Item>, std::greater<>> queue;
            std::vector<char> done(n, 0);
            dist[src] = 1.0;
            nrgeo[src] = 1.0;
            queue.emplace(1.0, src);
            while (!queue.empty()) {
                const auto [d, v] = queue.top();
                queue.pop();
                if (done[v] || d != dist[v]) continue;
                done[v] = 1;
                stack.push_back(v);
                for (i64 e = g.start[v]; e < g.start[v + 1]; ++e) {
                    const std::int32_t to = g.to[e];
                    const double alt = d + (unit ? 1.0 : g.cost[e]);
                    const double cur = dist[to];
                    if (cur == 0) {
                        parents[to].assign(1, v);
                        nrgeo[to] = nrgeo[v];
                        dist[to] = alt;
                        queue.emplace(alt, to);
                    } else {
                        const int c = cmp_epsilon(alt, cur, eps);
                        if (c < 0) {
                            parents[to].assign(1, v);
                            nrgeo[to] = nrgeo[v];
                            dist[to] = alt;
                            queue.emplace(alt, to);
                        } else if (c == 0) {  // as igraph, even for a node already visited
                            parents[to].push_back(v);
                            nrgeo[to] += nrgeo[v];
                        }
                    }
                }
            }
            double sum = 0;
            for (const std::int32_t v : stack) {
                if (v != src) sum += dist[v] - 1.0;
            }
            closeness[src] = sum;
            reached[src] = static_cast<std::int32_t>(stack.size());
            for (std::size_t q = stack.size(); q-- > 0;) {
                const std::int32_t w = stack[q];
                for (const std::int32_t p : parents[w]) delta[p] += nrgeo[p] / nrgeo[w] * (1.0 + delta[w]);
                if (w != src && delta[w] != 0) part.emplace_back(w, delta[w]);
            }
        });
        for (std::int32_t k = 0; k < count; ++k) {
            for (const auto& [w, d] : parts[static_cast<std::size_t>(k)]) out.betweenness[w] += d;
        }
    }
    for (double& b : out.betweenness) b /= 2;  // undirected: each pair counted from both ends
    out.closeness_sum = std::move(closeness);
    out.reached = std::move(reached);
    return out;
}

Csr make_length_graph(std::int32_t n, const std::vector<std::int32_t>& ei, const std::vector<std::int32_t>& ej,
                      const std::vector<double>& length) {
    return build_csr(n, ei, ej, &length);
}

}  // namespace conn
