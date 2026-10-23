// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/patch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace opt {

void PatchProblem::check() const {
    const auto nn = static_cast<std::size_t>(n);
    if (n < 0) throw std::invalid_argument("negative number of units");
    if (cost.size() != nn) throw std::invalid_argument("one cost per unit");
    if (start.size() != nn + 1 || start.front() != 0 || static_cast<std::size_t>(start.back()) != feature.size() ||
        feature.size() != amount.size())
        throw std::invalid_argument("the units' features must be in compressed rows");
    if (adj_start.size() != nn + 1 || adj_start.front() != 0 || static_cast<std::size_t>(adj_start.back()) != adj.size())
        throw std::invalid_argument("the neighbours must be in compressed rows");
    for (const int f : feature)
        if (f < 0 || static_cast<std::size_t>(f) >= target.size()) throw std::invalid_argument("a feature is out of range");
    for (const int v : adj)
        if (v < 0 || v >= n) throw std::invalid_argument("a neighbour is out of range");
    for (const double c : cost)
        if (!(c >= 0.0) || !std::isfinite(c)) throw std::invalid_argument("the costs must be finite and >= 0");
    if ((!locked_in.empty() && locked_in.size() != nn) || (!locked_out.empty() && locked_out.size() != nn))
        throw std::invalid_argument("one lock per unit");
}

namespace {

using Clock = std::chrono::steady_clock;

// splitmix64: the same numbers with every compiler and library
std::uint64_t mix(std::uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// a number from [0, 1) for a key and a unit
double unit_draw(std::uint64_t key, std::uint64_t v) {
    return static_cast<double>(mix(key ^ mix(v)) >> 11) * 0x1.0p-53;
}

// A selection and what it holds, with the steps of the heuristic.
class Search {
  public:
    Search(const PatchProblem& p, Clock::time_point deadline)
        : p_(p), deadline_(deadline), sel_(p.n, 0), held_(p.target.size(), 0.0), tol_(p.target.size()) {
        for (std::size_t f = 0; f < p.target.size(); ++f) tol_[f] = 1.0e-9 * std::max(1.0, std::abs(p.target[f]));
    }

    bool out_of_time() const { return Clock::now() >= deadline_; }
    bool locked_in(int v) const { return !p_.locked_in.empty() && p_.locked_in[v]; }
    bool locked_out(int v) const { return !p_.locked_out.empty() && p_.locked_out[v]; }

    void set(int v, bool on) {
        if ((sel_[v] != 0) == on) return;
        sel_[v] = on;
        const double s = on ? 1.0 : -1.0;
        for (i64 q = p_.start[v]; q < p_.start[v + 1]; ++q) held_[p_.feature[q]] += s * p_.amount[q];
        total_ += s * p_.cost[v];
    }

    void restore(const std::vector<std::uint8_t>& s) {
        for (int v = 0; v < p_.n; ++v) set(v, s[v] != 0);
    }

    // a selection with the locks applied
    void load(const std::vector<std::uint8_t>& s) {
        for (int v = 0; v < p_.n; ++v) set(v, locked_in(v) || (s[v] != 0 && !locked_out(v)));
    }

    double total() const { return total_; }
    const std::vector<std::uint8_t>& selection() const { return sel_; }

    bool meets() const {
        for (std::size_t f = 0; f < held_.size(); ++f)
            if (held_[f] < p_.target[f] - tol_[f]) return false;
        return true;
    }

    // whether the targets stay met without v
    bool spare(int v) const {
        for (i64 q = p_.start[v]; q < p_.start[v + 1]; ++q) {
            const int f = p_.feature[q];
            if (held_[f] - p_.amount[q] < p_.target[f] - tol_[f]) return false;
        }
        return true;
    }

    // the shares of the targets' shortfall that v would make up (weighted by `weight_` when set)
    double gain(int v) const {
        double g = 0.0;
        for (i64 q = p_.start[v]; q < p_.start[v + 1]; ++q) {
            const int f = p_.feature[q];
            const double need = p_.target[f] - held_[f];
            if (need > tol_[f])
                g += (weight_.empty() ? 1.0 : weight_[f]) * std::min(p_.amount[q], need) /
                     std::max(std::abs(p_.target[f]), 1e-300);
        }
        return g;
    }

    // the selection's patches: a label per selected unit (-1 elsewhere); returns their number
    int label(std::vector<int>& lab) const {
        lab.assign(p_.n, -1);
        int k = 0;
        std::vector<int> stack;
        for (int s = 0; s < p_.n; ++s) {
            if (!sel_[s] || lab[s] >= 0) continue;
            lab[s] = k;
            stack.assign(1, s);
            while (!stack.empty()) {
                const int u = stack.back();
                stack.pop_back();
                for (i64 q = p_.adj_start[u]; q < p_.adj_start[u + 1]; ++q) {
                    const int w = p_.adj[q];
                    if (sel_[w] && lab[w] < 0) {
                        lab[w] = k;
                        stack.push_back(w);
                    }
                }
            }
            ++k;
        }
        return k;
    }

    // joins the patches into one by the cheapest paths from the largest; false if some cannot be reached
    bool connect() {
        std::vector<int> lab;
        for (;;) {
            const int k = label(lab);
            if (k <= 1) return true;
            std::vector<int> size(k, 0);
            for (int v = 0; v < p_.n; ++v)
                if (lab[v] >= 0) ++size[lab[v]];
            const int main = static_cast<int>(std::max_element(size.begin(), size.end()) - size.begin());
            paths(lab, main);
            int hit = -1;
            double d = std::numeric_limits<double>::infinity();
            for (int v = 0; v < p_.n; ++v)
                if (lab[v] >= 0 && lab[v] != main && dist_[v] < d) {
                    d = dist_[v];
                    hit = v;
                }
            if (hit < 0) return false;
            for (int v = from_[hit]; v >= 0 && !sel_[v]; v = from_[v]) set(v, true);
        }
    }

    // grows a patch from `seed` (and the locked units): every step adds the unit with the greatest gain per
    // cost of the cheapest path to it from the patch, with the path; false if the targets cannot be met
    bool grow(int seed, double noise = 0.0, std::uint64_t key = 0) {
        for (int v = 0; v < p_.n; ++v) set(v, locked_in(v));
        if (locked_out(seed)) return false;
        set(seed, true);
        return regrow(noise, key);
    }

    // joins the selection's patches, then grows it as grow() until the targets are met; with `noise` > 0 every
    // unit's score is multiplied by 1 + noise x a number from [0, 1) drawn from `key` and the unit
    bool regrow(double noise = 0.0, std::uint64_t key = 0) {
        if (!connect()) return false;
        // with noise, the features weighed from 1 - noise / 2 to 1 + noise / 2
        weight_.clear();
        if (noise > 0.0) {
            weight_.resize(held_.size());
            for (std::size_t f = 0; f < held_.size(); ++f)
                weight_[f] = 1.0 + noise * (unit_draw(~key, static_cast<std::uint64_t>(f)) - 0.5);
        }
        std::vector<int> lab;
        while (!meets()) {
            if (out_of_time()) return false;
            lab.assign(p_.n, -1);
            for (int v = 0; v < p_.n; ++v)
                if (sel_[v]) lab[v] = 0;
            paths(lab, 0);
            int best = -1;
            double score = 0.0;
            for (int v = 0; v < p_.n; ++v) {
                if (sel_[v] || !std::isfinite(dist_[v])) continue;
                const double g = gain(v);
                if (g <= 0.0) continue;
                double sc = dist_[v] > 0.0 ? g / dist_[v] : std::numeric_limits<double>::infinity();
                if (noise > 0.0) sc *= 1.0 + noise * unit_draw(key, static_cast<std::uint64_t>(v));
                if (sc > score) {
                    score = sc;
                    best = v;
                }
            }
            if (best < 0) {
                weight_.clear();
                return false;
            }
            for (int v = best; v >= 0 && !sel_[v]; v = from_[v]) set(v, true);
        }
        weight_.clear();
        return true;
    }

    // removes a piece of the selection: up to `size` units found breadth-first from `centre` through it
    void cut_piece(int centre, int size) {
        std::vector<int> queue{centre};
        std::vector<std::uint8_t> seen(p_.n, 0);
        seen[centre] = 1;
        for (std::size_t h = 0; h < queue.size() && static_cast<int>(h) < size; ++h) {
            const int u = queue[h];
            for (i64 q = p_.adj_start[u]; q < p_.adj_start[u + 1]; ++q) {
                const int w = p_.adj[q];
                if (sel_[w] && !seen[w]) {
                    seen[w] = 1;
                    queue.push_back(w);
                }
            }
        }
        for (std::size_t h = 0; h < queue.size() && static_cast<int>(h) < size; ++h)
            if (!locked_in(queue[h])) set(queue[h], false);
    }

    // the cut vertices of the selection (Tarjan, iteratively)
    void cut_vertices(std::vector<std::uint8_t>& cut) const {
        cut.assign(p_.n, 0);
        std::vector<int> disc(p_.n, -1), low(p_.n, 0), parent(p_.n, -1);
        std::vector<i64> it(p_.n);
        int time = 0;
        std::vector<int> stack;
        for (int r = 0; r < p_.n; ++r) {
            if (!sel_[r] || disc[r] >= 0) continue;
            int children = 0;
            disc[r] = low[r] = time++;
            it[r] = p_.adj_start[r];
            stack.assign(1, r);
            while (!stack.empty()) {
                const int u = stack.back();
                if (it[u] < p_.adj_start[u + 1]) {
                    const int w = p_.adj[it[u]++];
                    if (!sel_[w]) continue;
                    if (disc[w] < 0) {
                        parent[w] = u;
                        disc[w] = low[w] = time++;
                        it[w] = p_.adj_start[w];
                        stack.push_back(w);
                        if (u == r) ++children;
                    } else if (w != parent[u]) {
                        low[u] = std::min(low[u], disc[w]);
                    }
                } else {
                    stack.pop_back();
                    const int pu = parent[u];
                    if (pu >= 0) {
                        low[pu] = std::min(low[pu], low[u]);
                        if (pu != r && low[u] >= disc[pu]) cut[pu] = 1;
                    }
                }
            }
            if (children > 1) cut[r] = 1;
        }
    }

    // drops units, the costliest first, while the targets stay met and the patch one; with `noise` > 0 the
    // order is by the costs multiplied by 1 + noise x a number from [0, 1) drawn from `key` and the unit
    void prune(double noise = 0.0, std::uint64_t key = 0) {
        std::vector<int> order;
        std::vector<double> rank(p_.n, 0.0);
        for (int v = 0; v < p_.n; ++v)
            if (sel_[v] && !locked_in(v) && p_.cost[v] > 0.0) {
                order.push_back(v);
                rank[v] = p_.cost[v] * (noise > 0.0 ? 1.0 + noise * unit_draw(key, static_cast<std::uint64_t>(v)) : 1.0);
            }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return rank[a] > rank[b]; });
        std::vector<std::uint8_t> cut;
        bool dropped = true;
        while (dropped) {
            dropped = false;
            cut_vertices(cut);
            for (const int v : order) {
                if (!sel_[v] || cut[v] || !spare(v)) continue;
                set(v, false);
                dropped = true;
                break;
            }
        }
    }

    // adds units by the greatest gain per cost (not `tabu`) until the targets are met
    bool repair(int tabu) {
        while (!meets()) {
            int best = -1;
            double score = 0.0;
            for (int v = 0; v < p_.n; ++v) {
                if (sel_[v] || v == tabu || locked_out(v)) continue;
                const double g = gain(v);
                if (g <= 0) continue;
                const double sc = g / std::max(p_.cost[v], 1e-12);
                if (sc > score) {
                    score = sc;
                    best = v;
                }
            }
            if (best < 0) return false;
            set(best, true);
        }
        return true;
    }

  private:
    // Dijkstra from the units labelled `from_label` (distance 0) over the others that are not locked out, a
    // unit's cost paid on entering it; units of other labels are reached but not passed through
    void paths(const std::vector<int>& lab, int from_label) {
        dist_.assign(p_.n, std::numeric_limits<double>::infinity());
        from_.assign(p_.n, -1);
        using Item = std::pair<double, int>;
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
        for (int v = 0; v < p_.n; ++v)
            if (lab[v] == from_label) {
                dist_[v] = 0.0;
                heap.emplace(0.0, v);
            }
        while (!heap.empty()) {
            const auto [d, u] = heap.top();
            heap.pop();
            if (d > dist_[u]) continue;
            if (lab[u] >= 0 && lab[u] != from_label) continue;
            for (i64 q = p_.adj_start[u]; q < p_.adj_start[u + 1]; ++q) {
                const int w = p_.adj[q];
                if (lab[w] == from_label || (locked_out(w) && !sel_[w])) continue;
                const double nd = d + (sel_[w] ? 0.0 : p_.cost[w]);
                if (nd < dist_[w]) {
                    dist_[w] = nd;
                    from_[w] = u;
                    heap.emplace(nd, w);
                }
            }
        }
    }

    const PatchProblem& p_;
    Clock::time_point deadline_;
    std::vector<std::uint8_t> sel_;
    std::vector<double> held_;
    std::vector<double> tol_;
    double total_ = 0.0;
    std::vector<double> dist_;
    std::vector<int> from_;
    std::vector<double> weight_;  // the features' weights in gain() (empty: 1)
};

// moves from a patch until none pays or the time is up; returns the moves made (s ends on the best)
int local_search(const PatchProblem& p, Search& s, std::vector<std::uint8_t>& best, double& best_cost) {
    int moves = 0;
    const double eps = 1e-9 * std::max(1.0, std::abs(best_cost));
    bool improved = true;
    while (improved && !s.out_of_time()) {
        improved = false;
        // a neighbour enters, then the patch is pruned
        std::vector<int> frontier;
        for (int v = 0; v < p.n; ++v) {
            if (best[v] || (!p.locked_out.empty() && p.locked_out[v])) continue;
            for (i64 q = p.adj_start[v]; q < p.adj_start[v + 1]; ++q)
                if (best[p.adj[q]]) {
                    frontier.push_back(v);
                    break;
                }
        }
        std::stable_sort(frontier.begin(), frontier.end(), [&](int a, int b) { return p.cost[a] < p.cost[b]; });
        for (const int u : frontier) {
            if (s.out_of_time()) break;
            s.restore(best);
            s.set(u, true);
            s.prune();
            if (s.total() < best_cost - eps) {
                best = s.selection();
                best_cost = s.total();
                ++moves;
                improved = true;
                break;
            }
        }
        if (improved) continue;
        // a unit leaves: the shortfall repaired, the patches joined again, then pruned
        std::vector<int> inside;
        for (int v = 0; v < p.n; ++v)
            if (best[v] && (p.locked_in.empty() || !p.locked_in[v])) inside.push_back(v);
        std::stable_sort(inside.begin(), inside.end(), [&](int a, int b) { return p.cost[a] > p.cost[b]; });
        for (const int v : inside) {
            if (s.out_of_time()) break;
            s.restore(best);
            s.set(v, false);
            if (!s.repair(v) || !s.connect()) continue;
            s.prune();
            if (s.total() < best_cost - eps) {
                best = s.selection();
                best_cost = s.total();
                ++moves;
                improved = true;
                break;
            }
        }
    }
    s.restore(best);
    return moves;
}

// a perturbation of the patch `from` (number `k`): a piece of 10 to 30 % of it removed around a unit and the
// patch grown back with noise on the scores (from 0.25 to 1.5), or (one in four) a patch grown with noise from a
// unit drawn from all; then pruned (in an order with the same noise on the costs) and improved by the moves;
// false if it could not be grown
bool perturb(const PatchProblem& p, Search& s, const std::vector<std::uint8_t>& from, std::uint64_t k,
             std::vector<std::uint8_t>& out, double& cost) {
    const std::uint64_t a = mix(k), b = mix(a), c = mix(b);
    const double noise = 0.25 + 1.25 * static_cast<double>(c >> 11) * 0x1.0p-53;
    if (k % 4 == 3) {
        if (!s.grow(static_cast<int>(a % static_cast<std::uint64_t>(p.n)), noise, mix(b))) return false;
    } else {
        std::vector<int> inside;
        for (int v = 0; v < p.n; ++v)
            if (from[v]) inside.push_back(v);
        if (inside.size() < 2) return false;
        const int centre = inside[a % inside.size()];
        const double share = 0.1 + 0.2 * static_cast<double>(b >> 11) * 0x1.0p-53;
        s.restore(from);
        s.cut_piece(centre, std::max(1, static_cast<int>(share * static_cast<double>(inside.size()))));
        if (!s.regrow(noise, mix(b))) return false;
    }
    s.prune(noise, mix(c));
    out = s.selection();
    cost = s.total();
    local_search(p, s, out, cost);
    return true;
}

}  // namespace

PatchResult improve_patch(const PatchProblem& p, const std::vector<std::uint8_t>& start, double time_limit,
                          int threads, int max_seeds, int stall_rounds) {
    p.check();
    if (!start.empty() && start.size() != static_cast<std::size_t>(p.n))
        throw std::invalid_argument("one start value per unit");
    if (threads < 1) throw std::invalid_argument("threads must be at least 1");
    const auto t0 = Clock::now();
    const auto deadline = t0 + std::chrono::duration_cast<Clock::duration>(
                                   std::chrono::duration<double>(std::clamp(time_limit, 0.0, 1.0e7)));
    PatchResult r;
    std::vector<std::vector<std::uint8_t>> tops;
    std::vector<double> top_cost;
    if (!start.empty()) {
        Search s(p, deadline);
        s.load(start);
        if (s.repair(-1) && s.connect()) {
            s.prune();
            tops.push_back(s.selection());
            top_cost.push_back(s.total());
        }
    }
    // seeds: the units that hold the most of the targets per cost first
    std::vector<int> seeds;
    std::vector<double> value(p.n, 0.0);
    for (int v = 0; v < p.n; ++v) {
        if (!p.locked_out.empty() && p.locked_out[v]) continue;
        for (i64 q = p.start[v]; q < p.start[v + 1]; ++q)
            value[v] += p.amount[q] / std::max(std::abs(p.target[p.feature[q]]), 1e-300);
        seeds.push_back(v);
    }
    std::stable_sort(seeds.begin(), seeds.end(), [&](int a, int b) {
        return value[a] / std::max(p.cost[a], 1e-12) > value[b] / std::max(p.cost[b], 1e-12);
    });
    if (max_seeds >= 0 && static_cast<int>(seeds.size()) > max_seeds) seeds.resize(max_seeds);
    const int ns = static_cast<int>(seeds.size());
    std::vector<double> grown(ns, std::numeric_limits<double>::infinity());
    std::vector<std::vector<std::uint8_t>> patches(ns);
#pragma omp parallel num_threads(threads)
    {
        Search s(p, deadline);
#pragma omp for schedule(dynamic, 1)
        for (int k = 0; k < ns; ++k) {
            if (s.out_of_time() || !s.grow(seeds[k])) continue;
            s.prune();
            grown[k] = s.total();
            patches[k] = s.selection();
        }
    }
    // the cheapest grown patches (distinct) join the start's; each improved by the moves
    std::vector<int> order(ns);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return grown[a] < grown[b]; });
    const std::size_t keep = tops.size() + static_cast<std::size_t>(std::max(threads, 4));
    for (const int k : order) {
        if (!std::isfinite(grown[k]) || tops.size() >= keep) break;
        if (std::find(tops.begin(), tops.end(), patches[k]) == tops.end()) {
            tops.push_back(std::move(patches[k]));
            top_cost.push_back(grown[k]);
        }
    }
    const int nt = static_cast<int>(tops.size());
    std::vector<int> moves(nt, 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int k = 0; k < nt; ++k) {
        Search s(p, deadline);
        s.restore(tops[k]);
        top_cost[k] = s.total();
        moves[k] = local_search(p, s, tops[k], top_cost[k]);
    }
    int best = -1;
    for (int k = 0; k < nt; ++k)
        if (best < 0 || top_cost[k] < top_cost[best]) best = k;
    if (best < 0) {
        r.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        return r;
    }
    std::vector<std::uint8_t> sel = std::move(tops[best]);
    double sel_cost = top_cost[best];
    r.moves = moves[best];
    // rounds of perturbations of the best patch, a fixed number per round (the same with any number of
    // threads), until the time is up: the cheapest of a round replaces the best if it is cheaper
    constexpr int kPerRound = 16;
    std::vector<std::vector<std::uint8_t>> tried(kPerRound);
    std::vector<double> tried_cost(kPerRound);
    std::vector<std::uint8_t> ok(kPerRound);
    int stalled = 0;
    for (std::uint64_t round = 0; Clock::now() < deadline && stalled < stall_rounds; ++round) {
#pragma omp parallel num_threads(threads)
        {
            Search s(p, deadline);
#pragma omp for schedule(dynamic, 1)
            for (int j = 0; j < kPerRound; ++j)
                ok[j] = perturb(p, s, sel, round * kPerRound + j, tried[j], tried_cost[j]) ? 1 : 0;
        }
        if (Clock::now() >= deadline) break;  // a round cut short by the time is not taken
        int j_best = -1;
        for (int j = 0; j < kPerRound; ++j)
            if (ok[j] && (j_best < 0 || tried_cost[j] < tried_cost[j_best])) j_best = j;
        if (j_best >= 0 && tried_cost[j_best] < sel_cost - 1e-9 * std::max(1.0, std::abs(sel_cost))) {
            sel = std::move(tried[j_best]);
            sel_cost = tried_cost[j_best];
            ++r.moves;
            stalled = 0;
        } else {
            ++stalled;
        }
        ++r.rounds;
    }
    r.selection = std::move(sel);
    r.cost = sel_cost;
    r.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    return r;
}

}  // namespace opt
