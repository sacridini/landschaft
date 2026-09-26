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
    if (cost) g.cost.resize(ei.size() * 2);
    std::vector<i64> fill(g.start.begin(), g.start.end() - 1);
    for (std::size_t e = 0; e < ei.size(); ++e) {
        const i64 a = fill[ei[e]]++;
        const i64 b = fill[ej[e]]++;
        g.to[a] = ej[e];
        g.to[b] = ei[e];
        if (cost) g.cost[a] = g.cost[b] = (*cost)[e];
    }
    // Rows sorted by neighbour (edges are then found by binary search). Pairs
    // sorted by (i, j) already give sorted rows.
    lsm::parallel_for(n > 4096, n, [&](i64 v) {
        const i64 b = g.start[v], e = g.start[v + 1];
        if (std::is_sorted(g.to.begin() + b, g.to.begin() + e)) return;
        std::vector<std::pair<std::int32_t, double>> row;
        for (i64 x = b; x < e; ++x) row.emplace_back(g.to[x], cost ? g.cost[x] : 0.0);
        std::sort(row.begin(), row.end());
        for (i64 x = b; x < e; ++x) {
            g.to[x] = row[x - b].first;
            if (cost) g.cost[x] = row[x - b].second;
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
struct Contribution {
    std::int32_t node;
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

LinkResult analyse_links(const Csr& g, const double* area, bool importance) {
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
        if (!importance || w.order.size() < 3) return;

        w.number_tree(src);
        Heap heap;
        constexpr std::int32_t none = std::numeric_limits<std::int32_t>::max();
        if (w.level_count.size() < w.order.size() + 2) w.level_count.assign(w.order.size() + 2, 0);
        for (const std::int32_t k : w.order) {
            if (k == src) continue;
            const std::int32_t lo = w.tin[k], hi = w.tout[k];
            if (hi - lo <= 1) continue;  // a leaf: no path runs through it
            // The subtree of k by level (counting sort).
            const std::int32_t base = w.dist[k] + 1;
            std::int32_t top = base;
            for (std::int32_t pos = lo + 1; pos < hi; ++pos) top = std::max(top, w.dist[w.euler[pos]]);
            std::int32_t* count = w.level_count.data();
            for (std::int32_t pos = lo + 1; pos < hi; ++pos) ++count[w.dist[w.euler[pos]] - base + 1];
            for (std::int32_t l = 1; l <= top - base + 1; ++l) count[l] += count[l - 1];
            w.sorted.resize(static_cast<std::size_t>(hi - lo - 1));
            for (std::int32_t pos = lo + 1; pos < hi; ++pos) {
                const std::int32_t j = w.euler[pos];
                w.sorted[count[w.dist[j] - base]++] = j;
            }
            std::fill(count, count + (top - base + 2), 0);
            // A node keeps its level when a neighbour that keeps its own sits
            // one level up; the others (`affected`) are found again below.
            w.affected.clear();
            for (const std::int32_t j : w.sorted) {
                const std::int32_t up = w.dist[j] - 1;
                bool keeps = false;
                for (i64 e = g.start[j]; e < g.start[j + 1]; ++e) {
                    const std::int32_t u = g.to[e];
                    if (u != k && w.dist[u] == up && !w.flag[u]) {
                        keeps = true;
                        break;
                    }
                }
                if (!keeps) {
                    w.flag[j] = 1;
                    w.affected.push_back(j);
                }
            }
            if (w.affected.empty()) continue;
            for (const std::int32_t j : w.affected) {
                std::int32_t best = none;
                for (i64 e = g.start[j]; e < g.start[j + 1]; ++e) {
                    const std::int32_t u = g.to[e];
                    if (u == k || w.flag[u]) continue;
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
            if (d_iic != 0 || d_h != 0) out.push_back({k, area[src] * d_iic, d_h});
        }
    };
    over_sources(n, importance, body, [&](std::int32_t, const std::vector<Contribution>& c) {
        for (const Contribution& x : c) {
            r.iic_conn[x.node] += x.first;
            r.h_conn[x.node] += x.second;
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
    return r;
}

ProbResult analyse_probabilities(const Csr& g, const double* area, double expand_limit, bool importance) {
    const std::int32_t n = g.n;
    ProbResult r;
    std::vector<double> row(n), direct(n);
    if (importance) r.pc_conn.assign(n, 0.0);

    std::vector<Workspace<double>*> spaces(static_cast<std::size_t>(thread_count()), nullptr);
    auto workspace = [&]() -> Workspace<double>& {
#if defined(_OPENMP)
        const int t = omp_in_parallel() ? omp_get_thread_num() : 0;
#else
        const int t = 0;
#endif
        if (!spaces[t]) spaces[t] = new Workspace<double>(n, importance);
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
                if (importance) w.add_candidate(u, v, nd);
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
        if (!importance || w.order.size() < 3) return;

        w.number_tree(src);
        for (const std::int32_t k : w.order) {
            if (k == src) continue;
            const std::int32_t lo = w.tin[k], hi = w.tout[k];
            if (hi - lo <= 1) continue;
            auto inside = [&](std::int32_t v) { return w.tin[v] > lo && w.tin[v] < hi; };
            const std::int32_t size = hi - lo - 1;
            for (std::int32_t pos = lo + 1; pos < hi; ++pos) {
                const std::int32_t j = w.euler[pos];
                double best = kInf;
                const std::size_t cb = static_cast<std::size_t>(j) * kCandidates;
                const int len = w.cand_len[j];
                bool found = false;
                for (int c = 0; c < len; ++c) {
                    const std::int32_t u = w.cand_node[cb + c];
                    if (u == k || inside(u)) continue;
                    best = w.cand_cost[cb + c];
                    found = true;
                    break;
                }
                if (!found && len == kCandidates) {  // all K cut off: every neighbour
                    for (i64 e = g.start[j]; e < g.start[j + 1]; ++e) {
                        const std::int32_t u = g.to[e];
                        if (u == k || inside(u) || w.dist[u] > expand_limit) continue;
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
                    for (std::int32_t pos = lo + 1; pos < hi; ++pos) {
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
                    if (u == k || !inside(u)) continue;
                    const double nd = d + g.cost[e];
                    if (nd < w.repaired[u]) {
                        w.repaired[u] = nd;
                        heap.emplace(nd, u);
                    }
                }
            }
            double change = 0;
            for (std::int32_t pos = lo + 1; pos < hi; ++pos) {
                const std::int32_t j = w.euler[pos];
                if (w.repaired[j] == w.dist[j]) continue;
                change += area[j] * (std::exp(-w.dist[j]) - std::exp(-w.repaired[j]));
            }
            if (change != 0) out.push_back({k, area[src] * change, 0.0});
        }
    };
    over_sources(n, importance, body, [&](std::int32_t, const std::vector<Contribution>& c) {
        for (const Contribution& x : c) r.pc_conn[x.node] += x.first;
    });

    for (std::int32_t v = 0; v < n; ++v) {
        r.pc_num += area[v] * row[v];
        r.pc_direct += area[v] * direct[v];
    }
    if (importance) r.pc_row = std::move(row);
    return r;
}

}  // namespace conn
