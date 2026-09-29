// SPDX-License-Identifier: GPL-3.0-or-later
#include "cost/shortest.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace cost {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The out-neighbours of every cell, in increasing index, with the cost of the
// step (1 / T[a, b]); igraph's incidence order.
class Steps {
public:
    explicit Steps(const Layout& t) : t_(t), off_(offsets(t.directions, t.symmetric)), n_(t.grid->ncell()) {
        for (i64 k = 0; k < static_cast<i64>(off_.size()) * n_; ++k) {
            if (t.values[k] < 0) throw std::invalid_argument("negative conductances: costs must not be negative");
        }
    }

    // Fills b[], c[] (sorted by b) and returns their number.
    int of(i64 a, i64* b, double* c) const {
        const Grid& g = *t_.grid;
        const i64 r = a / g.nx, col = a % g.nx;
        int m = 0;
        auto add = [&](i64 k, double v) {
            int p = m++;
            while (p > 0 && b[p - 1] > k) {
                b[p] = b[p - 1];
                c[p] = c[p - 1];
                --p;
            }
            b[p] = k;
            c[p] = 1.0 / v;  // R: E(g)$weight <- 1 / E(g)$weight
        };
        const int nd = static_cast<int>(off_.size());
        for (int d = 0; d < nd; ++d) {
            const double v = t_.values[d * n_ + a];
            if (v != 0) {
                const i64 nb = g.neighbour(r, col, off_[d]);
                if (nb >= 0) add(nb, v);
            }
            if (t_.symmetric) {
                const i64 prev = g.neighbour(r, col, Offset{-off_[d].dr, -off_[d].dc});
                if (prev >= 0) {
                    const double w = t_.values[d * n_ + prev];
                    if (w != 0) add(prev, w);
                }
            }
        }
        return m;
    }

    i64 n() const { return n_; }

    // The same steps in no particular order (for distances only), calling f(b, cost).
    template <class F>
    void each(i64 a, F&& f) const {
        const Grid& g = *t_.grid;
        const i64 r = a / g.nx, col = a - r * g.nx;
        const int nd = static_cast<int>(off_.size());
        for (int d = 0; d < nd; ++d) {
            const double v = t_.values[d * n_ + a];
            if (v != 0) {
                const i64 nb = g.neighbour(r, col, off_[d]);
                if (nb >= 0) f(nb, 1.0 / v);
            }
            if (t_.symmetric) {
                const i64 prev = g.neighbour(r, col, Offset{-off_[d].dr, -off_[d].dc});
                if (prev >= 0) {
                    const double w = t_.values[d * n_ + prev];
                    if (w != 0) f(prev, 1.0 / w);
                }
            }
        }
    }

private:
    const Layout& t_;
    std::vector<Offset> off_;
    i64 n_;
};

// igraph's two-way indexed heap (core/indheap.c): a max-heap of -distance;
// index2[v] = position + 2, 1 = removed (deactivated), 0 = never pushed.
class TwoWayHeap {
public:
    explicit TwoWayHeap(i64 n) : index2_(static_cast<std::size_t>(n), 0) {}

    bool empty() const { return data_.empty(); }
    i64 max_index() const { return index_[0]; }
    bool has_elem(i64 v) const { return index2_[v] != 0; }
    bool has_active(i64 v) const { return index2_[v] > 1; }
    double get(i64 v) const { return data_[index2_[v] - 2]; }

    void push(i64 v, double x) {
        const i64 size = static_cast<i64>(data_.size());
        data_.push_back(x);
        index_.push_back(v);
        index2_[v] = size + 2;
        shift_up(size);
    }

    // Removes the top; `removed` is 0 (delete) or 1 (deactivate).
    double pop(i64 removed) {
        const double top = data_[0];
        const i64 v = index_[0];
        swap(0, static_cast<i64>(data_.size()) - 1);
        data_.pop_back();
        index_.pop_back();
        index2_[v] = removed;
        sink(0);
        return top;
    }

    void modify(i64 v, double x) {
        const i64 pos = index2_[v] - 2;
        data_[pos] = x;
        sink(pos);
        shift_up(pos);
    }

    void clear() {
        for (const i64 v : index_) index2_[v] = 0;
        for (const i64 v : touched_) index2_[v] = 0;
        touched_.clear();
        data_.clear();
        index_.clear();
    }

    void remember(i64 v) { touched_.push_back(v); }

private:
    static i64 parent(i64 x) { return (x + 1) / 2 - 1; }
    static i64 left(i64 x) { return (x + 1) * 2 - 1; }
    static i64 right(i64 x) { return (x + 1) * 2; }

    void swap(i64 e1, i64 e2) {
        if (e1 == e2) return;
        std::swap(data_[e1], data_[e2]);
        const i64 t1 = index_[e1], t2 = index_[e2];
        index2_[t1] = e2 + 2;
        index2_[t2] = e1 + 2;
        index_[e1] = t2;
        index_[e2] = t1;
    }

    void shift_up(i64 e) {
        while (!(e == 0 || data_[e] < data_[parent(e)])) {
            swap(e, parent(e));
            e = parent(e);
        }
    }

    void sink(i64 head) {
        const i64 size = static_cast<i64>(data_.size());
        for (;;) {
            if (left(head) >= size) return;
            i64 child;
            if (right(head) == size || data_[left(head)] >= data_[right(head)]) {
                child = left(head);
            } else {
                child = right(head);
            }
            if (!(data_[head] < data_[child])) return;
            swap(head, child);
            head = child;
        }
    }

    std::vector<double> data_;
    std::vector<i64> index_, index2_, touched_;
};

// A 4-ary min-heap of (distance, cell) without decrease-key: stale entries are
// skipped when popped. Faster than the indexed heap; for distances only, whose
// bits do not depend on the order of equal keys.
class LazyHeap {
public:
    bool empty() const { return h_.empty(); }
    void push(double d, i64 v) {
        h_.push_back({d, v});
        std::size_t i = h_.size() - 1;
        while (i > 0) {
            const std::size_t p = (i - 1) / 4;
            if (!(h_[i].d < h_[p].d)) break;
            std::swap(h_[i], h_[p]);
            i = p;
        }
    }
    std::pair<double, i64> pop() {
        const Item top = h_[0];
        h_[0] = h_.back();
        h_.pop_back();
        const std::size_t n = h_.size();
        std::size_t i = 0;
        for (;;) {
            const std::size_t c0 = 4 * i + 1;
            if (c0 >= n) break;
            std::size_t best = c0;
            const std::size_t c1 = std::min(c0 + 4, n);
            for (std::size_t c = c0 + 1; c < c1; ++c) {
                if (h_[c].d < h_[best].d) best = c;
            }
            if (!(h_[best].d < h_[i].d)) break;
            std::swap(h_[i], h_[best]);
            i = best;
        }
        return {top.d, top.v};
    }
    void clear() { h_.clear(); }

private:
    struct Item {
        double d;
        i64 v;
    };
    std::vector<Item> h_;
};

// Dijkstra's distances from the sources (at 0); settle(v, d) is called in
// order of distance and returns false to stop.
struct Workspace {
    explicit Workspace(i64 n) : dist(static_cast<std::size_t>(n), kInf), done(static_cast<std::size_t>(n), 0) {}
    std::vector<double> dist;
    std::vector<char> done;
    std::vector<i64> touched;
    LazyHeap q;
};

template <class Settle>
void fast_dijkstra(const Steps& steps, Workspace& w, const std::vector<i64>& sources, Settle&& settle) {
    for (const i64 v : sources) {
        if (w.dist[v] == kInf) {
            w.touched.push_back(v);
            w.dist[v] = 0.0;
            w.q.push(0.0, v);
        }
    }
    while (!w.q.empty()) {
        const auto [du, u] = w.q.pop();
        if (w.done[u] || du != w.dist[u]) continue;  // stale entry
        w.done[u] = 1;
        if (!settle(u, du)) break;
        steps.each(u, [&](i64 v, double c) {
            if (w.done[v] || c == kInf) return;
            const double alt = du + c;
            if (alt < w.dist[v]) {
                if (w.dist[v] == kInf) w.touched.push_back(v);
                w.dist[v] = alt;
                w.q.push(alt, v);
            }
        });
    }
    w.q.clear();
    for (const i64 v : w.touched) {
        w.dist[v] = kInf;
        w.done[v] = 0;
    }
    w.touched.clear();
}

int thread_count() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace

std::vector<double> accumulated_cost(const Layout& t, const std::vector<i64>& sources) {
    const Steps steps(t);
    std::vector<double> out(static_cast<std::size_t>(steps.n()), kInf);
    if (sources.empty()) return out;
    Workspace w(steps.n());
    fast_dijkstra(steps, w, sources, [&](i64 v, double d) {
        out[v] = d;
        return true;
    });
    return out;
}

std::vector<double> cost_matrix(const Layout& t, const std::vector<i64>& from, const std::vector<i64>& to) {
    const Steps steps(t);
    const i64 nf = static_cast<i64>(from.size()), nt = static_cast<i64>(to.size());
    std::vector<double> out(static_cast<std::size_t>(nf * nt), kInf);
    // distinct targets and where each goes
    std::vector<i64> uniq(to);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    const i64 nu = static_cast<i64>(uniq.size());
#pragma omp parallel num_threads(std::max(1, std::min<int>(thread_count(), static_cast<int>(nf))))
    {
        Workspace w(steps.n());
        std::vector<double> found(static_cast<std::size_t>(nu));
#pragma omp for schedule(dynamic)
        for (i64 i = 0; i < nf; ++i) {
            std::fill(found.begin(), found.end(), kInf);
            i64 reached = 0;
            fast_dijkstra(steps, w, {from[i]}, [&](i64 v, double d) {
                const auto it = std::lower_bound(uniq.begin(), uniq.end(), v);
                if (it != uniq.end() && *it == v) {
                    found[it - uniq.begin()] = d;
                    if (++reached == nu) return false;
                }
                return true;
            });
            for (i64 j = 0; j < nt; ++j) {
                out[i * nt + j] = found[std::lower_bound(uniq.begin(), uniq.end(), to[j]) - uniq.begin()];
            }
        }
    }
    return out;
}

std::vector<std::vector<i64>> shortest_paths(const Layout& t, i64 origin, const std::vector<i64>& goals) {
    // igraph_get_shortest_paths_dijkstra
    const Steps steps(t);
    const i64 n = steps.n();
    TwoWayHeap q(n);
    std::vector<double> dist(static_cast<std::size_t>(n), -1.0);
    std::vector<i64> parent(static_cast<std::size_t>(n), -1);
    std::vector<char> is_target(static_cast<std::size_t>(n), 0);
    i64 to_reach = static_cast<i64>(goals.size());
    for (const i64 g : goals) {
        if (!is_target[g]) is_target[g] = 1;
        else --to_reach;
    }
    dist[origin] = 0.0;
    q.push(origin, 0);
    i64 b[16];
    double c[16];
    while (!q.empty() && to_reach > 0) {
        const i64 u = q.max_index();
        const double du = -q.pop(0);
        if (is_target[u]) {
            is_target[u] = 0;
            --to_reach;
        }
        const int m = steps.of(u, b, c);
        for (int k = 0; k < m; ++k) {
            const i64 v = b[k];
            const double alt = du + c[k];
            const double cur = dist[v];
            if (cur < 0) {
                dist[v] = alt;
                parent[v] = u;
                q.push(v, -alt);
            } else if (alt < cur) {
                dist[v] = alt;
                parent[v] = u;
                q.modify(v, -alt);
            }
        }
    }
    std::vector<std::vector<i64>> out;
    out.reserve(goals.size());
    for (const i64 g : goals) {
        std::vector<i64> path;
        if (g == origin || parent[g] >= 0) {
            for (i64 v = g; v != origin; v = parent[v]) path.push_back(v);
            path.push_back(origin);
            std::reverse(path.begin(), path.end());
        }
        out.push_back(std::move(path));
    }
    return out;
}

}  // namespace cost
