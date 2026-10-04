// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/maxflow.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace opt {

MaxFlow::MaxFlow(int n, const std::vector<int>& i, const std::vector<int>& j, const std::vector<double>& cap_ij,
                 const std::vector<double>& cap_ji)
    : n_(n) {
    const std::size_t m = i.size();
    if (j.size() != m || cap_ij.size() != m || cap_ji.size() != m)
        throw std::invalid_argument("inconsistent edge arrays");
    first_.assign(static_cast<std::size_t>(n) + 1, 0);
    for (std::size_t k = 0; k < m; ++k) {
        if (i[k] < 0 || i[k] >= n || j[k] < 0 || j[k] >= n || i[k] == j[k])
            throw std::invalid_argument("an edge between nodes that do not exist or a node and itself");
        if (cap_ij[k] < 0 || cap_ji[k] < 0) throw std::invalid_argument("negative edge capacity");
        first_[i[k] + 1]++;
        first_[j[k] + 1]++;
    }
    for (int v = 0; v < n; ++v) first_[v + 1] += first_[v];
    head_.resize(2 * m);
    sister_.resize(2 * m);
    cap_.resize(2 * m);
    std::vector<std::int64_t> at(first_.begin(), first_.end() - 1);
    for (std::size_t k = 0; k < m; ++k) {
        const auto a = static_cast<int>(at[i[k]]++), b = static_cast<int>(at[j[k]]++);
        head_[a] = j[k];
        head_[b] = i[k];
        sister_[a] = b;
        sister_[b] = a;
        cap_[a] = cap_ij[k];
        cap_[b] = cap_ji[k];
    }
    rcap_.resize(2 * m);
    tr_.resize(n);
    parent_.resize(n);
    tree_.resize(n);
    dist_.resize(n);
    ts_.resize(n);
    active_.resize(n);
}

void MaxFlow::activate(int v) {
    if (active_[v]) return;
    active_[v] = 1;
    queue_.push_back(v);
}

int MaxFlow::next_active() {
    while (queue_head_ < queue_.size()) {
        const int v = queue_[queue_head_++];
        active_[v] = 0;
        if (parent_[v] != kNone) return v;
    }
    queue_.clear();
    queue_head_ = 0;
    return -1;
}

void MaxFlow::set_orphan(int v) {
    parent_[v] = kOrphan;
    orphans_.push_back(v);
}

double MaxFlow::augment(int a) {
    // a: an arc from a node of the source's tree to one of the sink's
    double f = rcap_[a];
    for (int v = tail(a);;) {  // up the source's tree: the flow goes from the parent to v
        const int pa = parent_[v];
        if (pa == kTerminal) {
            f = std::min(f, tr_[v]);
            break;
        }
        f = std::min(f, rcap_[sister_[pa]]);
        v = head_[pa];
    }
    for (int v = head_[a];;) {  // down the sink's tree: from v to the parent
        const int pa = parent_[v];
        if (pa == kTerminal) {
            f = std::min(f, -tr_[v]);
            break;
        }
        f = std::min(f, rcap_[pa]);
        v = head_[pa];
    }
    rcap_[a] -= f;
    rcap_[sister_[a]] += f;
    for (int v = tail(a);;) {
        const int pa = parent_[v];
        if (pa == kTerminal) {
            tr_[v] -= f;
            if (tr_[v] <= 0) {
                tr_[v] = 0;
                set_orphan(v);
            }
            break;
        }
        const int up = head_[pa];
        rcap_[pa] += f;
        rcap_[sister_[pa]] -= f;
        if (rcap_[sister_[pa]] <= 0) {
            rcap_[sister_[pa]] = 0;
            set_orphan(v);
        }
        v = up;
    }
    for (int v = head_[a];;) {
        const int pa = parent_[v];
        if (pa == kTerminal) {
            tr_[v] += f;
            if (tr_[v] >= 0) {
                tr_[v] = 0;
                set_orphan(v);
            }
            break;
        }
        const int up = head_[pa];
        rcap_[sister_[pa]] += f;
        rcap_[pa] -= f;
        if (rcap_[pa] <= 0) {
            rcap_[pa] = 0;
            set_orphan(v);
        }
        v = up;
    }
    return f;
}

int MaxFlow::origin_distance(int v) {
    constexpr int kInf = std::numeric_limits<int>::max();
    int d = 0;
    int u = v;
    for (;;) {
        if (ts_[u] == time_) {
            d += dist_[u];
            break;
        }
        const int pa = parent_[u];
        ++d;
        if (pa == kTerminal) {
            ts_[u] = time_;
            dist_[u] = 1;
            break;
        }
        if (pa < 0) return -1;  // an orphan (or free) on the way: no valid origin
        u = head_[pa];
    }
    if (d == kInf) return -1;
    // the nodes on the way get the distances (and the timestamp): later walks stop at them
    int dd = d;
    for (u = v; ts_[u] != time_; u = head_[parent_[u]]) {
        ts_[u] = time_;
        dist_[u] = dd--;
    }
    return d;
}

void MaxFlow::adopt(int v) {
    const std::uint8_t t = tree_[v];
    int best = kNone, dmin = std::numeric_limits<int>::max();
    for (std::int64_t a = first_[v]; a < first_[v + 1]; ++a) {
        const int u = head_[a];
        // the flow comes from u into v in the source's tree, goes from v to u in the sink's
        const double r = t == kSourceTree ? rcap_[sister_[a]] : rcap_[a];
        if (!(r > 0) || tree_[u] != t || parent_[u] == kNone || parent_[u] == kOrphan) continue;
        const int d = origin_distance(u);
        if (d >= 0 && d < dmin) {
            dmin = d;
            best = static_cast<int>(a);
        }
    }
    if (best != kNone) {
        parent_[v] = best;
        ts_[v] = time_;
        dist_[v] = dmin + 1;
        return;
    }
    // no parent: v leaves its tree; its children become orphans, its neighbours that could reach it active
    for (std::int64_t a = first_[v]; a < first_[v + 1]; ++a) {
        const int u = head_[a];
        if (tree_[u] != t || parent_[u] == kNone) continue;
        const double r = t == kSourceTree ? rcap_[sister_[a]] : rcap_[a];
        if (r > 0) activate(u);
        const int pu = parent_[u];
        if (pu >= 0 && head_[pu] == v) set_orphan(u);
    }
    parent_[v] = kNone;
}

double MaxFlow::solve(const std::vector<double>& source, const std::vector<double>& sink) {
    if (static_cast<int>(source.size()) != n_ || static_cast<int>(sink.size()) != n_)
        throw std::invalid_argument("one source and one sink capacity per node");
    std::copy(cap_.begin(), cap_.end(), rcap_.begin());
    queue_.clear();
    queue_head_ = 0;
    orphans_.clear();
    time_ = 0;
    double flow = 0;
    for (int v = 0; v < n_; ++v) {
        if (source[v] < 0 || sink[v] < 0) throw std::invalid_argument("negative terminal capacity");
        flow += std::min(source[v], sink[v]);
        tr_[v] = source[v] - sink[v];
        active_[v] = 0;
        ts_[v] = 0;
        dist_[v] = 1;
        if (tr_[v] > 0) {
            tree_[v] = kSourceTree;
            parent_[v] = kTerminal;
            activate(v);
        } else if (tr_[v] < 0) {
            tree_[v] = kSinkTree;
            parent_[v] = kTerminal;
            activate(v);
        } else {
            tree_[v] = 0;
            parent_[v] = kNone;
        }
    }
    int current = -1;
    for (;;) {
        int v = current;
        if (v < 0 || parent_[v] == kNone) {
            v = next_active();
            if (v < 0) break;
        }
        current = -1;
        // growth
        int path = -1;
        if (tree_[v] == kSourceTree) {
            for (std::int64_t a = first_[v]; a < first_[v + 1]; ++a) {
                if (!(rcap_[a] > 0)) continue;
                const int u = head_[a];
                if (parent_[u] == kNone) {
                    tree_[u] = kSourceTree;
                    parent_[u] = sister_[a];
                    ts_[u] = ts_[v];
                    dist_[u] = dist_[v] + 1;
                    activate(u);
                } else if (tree_[u] == kSinkTree) {
                    path = static_cast<int>(a);
                    break;
                } else if (ts_[u] <= ts_[v] && dist_[u] > dist_[v]) {
                    parent_[u] = sister_[a];  // a shorter way to the source
                    ts_[u] = ts_[v];
                    dist_[u] = dist_[v] + 1;
                }
            }
        } else {
            for (std::int64_t a = first_[v]; a < first_[v + 1]; ++a) {
                const int b = sister_[a];  // the arc from the neighbour into v
                if (!(rcap_[b] > 0)) continue;
                const int u = head_[a];
                if (parent_[u] == kNone) {
                    tree_[u] = kSinkTree;
                    parent_[u] = b;
                    ts_[u] = ts_[v];
                    dist_[u] = dist_[v] + 1;
                    activate(u);
                } else if (tree_[u] == kSourceTree) {
                    path = b;
                    break;
                } else if (ts_[u] <= ts_[v] && dist_[u] > dist_[v]) {
                    parent_[u] = b;
                    ts_[u] = ts_[v];
                    dist_[u] = dist_[v] + 1;
                }
            }
        }
        if (path < 0) continue;  // v is done (inactive) until a neighbour activates it again
        current = v;             // v may have more paths
        ++time_;
        flow += augment(path);
        // adoption
        while (!orphans_.empty()) {
            const int o = orphans_.back();
            orphans_.pop_back();
            adopt(o);
        }
    }
    return flow;
}

}  // namespace opt
