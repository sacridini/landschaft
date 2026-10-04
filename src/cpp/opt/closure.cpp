// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/closure.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>

#include "opt/bundle.hpp"
#include "opt/maxflow.hpp"

namespace opt::closure {

namespace {

using Clock = std::chrono::steady_clock;
using Heap = std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>,
                                 std::greater<std::pair<double, int>>>;

constexpr double kBigArc = 1e100;  // the requirements' arcs: never cut

double relative_gap(double upper, double lower) {
    if (!std::isfinite(upper) || !std::isfinite(lower)) return kInf;
    const double g = upper - lower;
    if (g <= 0) return 0.0;
    return g / std::max(std::abs(upper), 1e-10);
}

double row_tol(double b) { return 1e-9 * std::max(1.0, std::abs(b)); }

// Adjacency lists by node.
struct Adj {
    std::vector<i64> start;
    std::vector<int> node;
    std::vector<double> w;

    Adj(int n, const std::vector<int>& from, const std::vector<int>& to, const std::vector<double>* weight)
        : start(static_cast<std::size_t>(n) + 1, 0), node(from.size()), w(weight ? from.size() : 0) {
        for (const int v : from) ++start[static_cast<std::size_t>(v) + 1];
        for (std::size_t v = 0; v < static_cast<std::size_t>(n); ++v) start[v + 1] += start[v];
        std::vector<i64> next(start.begin(), start.end() - 1);
        for (std::size_t k = 0; k < from.size(); ++k) {
            const auto q = static_cast<std::size_t>(next[static_cast<std::size_t>(from[k])]++);
            node[q] = to[k];
            if (weight) w[q] = (*weight)[k];
        }
    }
    i64 begin(int v) const { return start[static_cast<std::size_t>(v)]; }
    i64 end(int v) const { return start[static_cast<std::size_t>(v) + 1]; }
};

// The repair of selections into feasible ones (see closure.hpp).
class Search {
public:
    Search(const Problem& p, const Adj& req_out, const Adj& req_in, const Adj& pen_out, const Adj& pen_in,
           const std::vector<std::int8_t>& fix)
        : p_(p), req_out_(req_out), req_in_(req_in), pen_out_(pen_out), pen_in_(pen_in), fix_(fix),
          mark_(static_cast<std::size_t>(p.n), 0), ga_(static_cast<std::size_t>(p.m), 0.0),
          act_(static_cast<std::size_t>(p.m), 0.0) {}

    // Makes z feasible and improves it; the objective, or +inf when no feasible selection was found.
    double run(std::vector<std::uint8_t>& z, const std::vector<double>& weight) {
        z_.swap(z);
        weight_ = &weight;
        const double v = repair();
        z_.swap(z);
        return v;
    }

    double objective_of(const std::vector<std::uint8_t>& z) const {
        double obj = 0;
        for (int v = 0; v < p_.n; ++v)
            if (z[static_cast<std::size_t>(v)]) {
                obj += p_.cost[static_cast<std::size_t>(v)];
                for (i64 k = pen_out_.begin(v); k < pen_out_.end(v); ++k)
                    if (!z[static_cast<std::size_t>(pen_out_.node[static_cast<std::size_t>(k)])]) obj += pen_out_.w[static_cast<std::size_t>(k)];
            }
        return obj;
    }

private:
    const Problem& p_;
    const Adj &req_out_, &req_in_, &pen_out_, &pen_in_;
    const std::vector<std::int8_t>& fix_;
    const std::vector<double>* weight_ = nullptr;
    std::vector<std::uint8_t> z_;
    std::vector<int> mark_;
    int stamp_ = 0;
    std::vector<int> group_, stack_, touched_;
    std::vector<double> ga_;   // the group's row amounts (dense, rows in touched_)
    std::vector<double> act_;  // row activities
    double obj_ = 0;

    bool in_group(int v) const { return mark_[static_cast<std::size_t>(v)] == stamp_; }

    void new_stamp() {
        if (++stamp_ == 0x7fffffff) {
            std::fill(mark_.begin(), mark_.end(), 0);
            stamp_ = 1;
        }
    }

    // v and the unselected nodes it needs (add) or the selected nodes that need it (remove), into
    // group_ (marked); false when that touches a node fixed the other way
    bool build(int v, bool add) {
        new_stamp();
        group_.clear();
        stack_.assign(1, v);
        mark_[static_cast<std::size_t>(v)] = stamp_;
        const Adj& next = add ? req_out_ : req_in_;
        while (!stack_.empty()) {
            const int u = stack_.back();
            stack_.pop_back();
            if (fix_[static_cast<std::size_t>(u)] == (add ? 0 : 1)) return false;
            group_.push_back(u);
            for (i64 k = next.begin(u); k < next.end(u); ++k) {
                const int w = next.node[static_cast<std::size_t>(k)];
                if (mark_[static_cast<std::size_t>(w)] == stamp_ || z_[static_cast<std::size_t>(w)] != (add ? 0 : 1)) continue;
                mark_[static_cast<std::size_t>(w)] = stamp_;
                stack_.push_back(w);
            }
        }
        // the group's row amounts
        for (const int r : touched_) ga_[static_cast<std::size_t>(r)] = 0.0;
        touched_.clear();
        for (const int u : group_)
            for (i64 k = p_.col_start[static_cast<std::size_t>(u)]; k < p_.col_start[static_cast<std::size_t>(u) + 1]; ++k) {
                const int r = p_.col_row[static_cast<std::size_t>(k)];
                if (ga_[static_cast<std::size_t>(r)] == 0.0) touched_.push_back(r);
                ga_[static_cast<std::size_t>(r)] += p_.col_val[static_cast<std::size_t>(k)];
            }
        return true;
    }

    // the objective's change when the group flips
    double delta(bool add) const {
        double d = 0;
        for (const int u : group_) {
            d += add ? p_.cost[static_cast<std::size_t>(u)] : -p_.cost[static_cast<std::size_t>(u)];
            for (i64 k = pen_out_.begin(u); k < pen_out_.end(u); ++k) {
                const int j = pen_out_.node[static_cast<std::size_t>(k)];
                if (!z_[static_cast<std::size_t>(j)] && !in_group(j)) d += add ? pen_out_.w[static_cast<std::size_t>(k)] : -pen_out_.w[static_cast<std::size_t>(k)];
            }
            for (i64 k = pen_in_.begin(u); k < pen_in_.end(u); ++k) {
                const int i = pen_in_.node[static_cast<std::size_t>(k)];
                if (z_[static_cast<std::size_t>(i)] && !in_group(i)) d += add ? -pen_in_.w[static_cast<std::size_t>(k)] : pen_in_.w[static_cast<std::size_t>(k)];
            }
        }
        return d;
    }

    void apply(bool add, double d) {
        for (const int u : group_) z_[static_cast<std::size_t>(u)] = add ? 1 : 0;
        for (const int r : touched_) act_[static_cast<std::size_t>(r)] += add ? ga_[static_cast<std::size_t>(r)] : -ga_[static_cast<std::size_t>(r)];
        obj_ += d;
    }

    // >= rows short of their target / <= rows above their bound, weighted
    double cover(bool geq) const {
        double c = 0;
        for (const int r : touched_) {
            const auto q = static_cast<std::size_t>(r);
            if ((p_.geq[q] != 0) != geq) continue;
            const double gap = geq ? p_.rhs[q] - act_[q] : act_[q] - p_.rhs[q];
            if (gap > row_tol(p_.rhs[q])) c += (*weight_)[q] * std::min(ga_[q], gap);
        }
        return c;
    }

    bool fits() const {  // adding the group keeps the <= rows
        for (const int r : touched_) {
            const auto q = static_cast<std::size_t>(r);
            if (!p_.geq[q] && act_[q] + ga_[q] > p_.rhs[q] + row_tol(p_.rhs[q])) return false;
        }
        return true;
    }

    bool keeps_targets() const {  // removing the group keeps the >= rows
        for (const int r : touched_) {
            const auto q = static_cast<std::size_t>(r);
            if (p_.geq[q] && act_[q] - ga_[q] < p_.rhs[q] - row_tol(p_.rhs[q])) return false;
        }
        return true;
    }

    // the share of the <= rows the group uses (0 without them)
    double usage() const {
        double u = 0;
        for (const int r : touched_) {
            const auto q = static_cast<std::size_t>(r);
            if (!p_.geq[q]) u += ga_[q] / std::max(std::abs(p_.rhs[q]), 1e-12);
        }
        return u;
    }

    int violated(bool geq) const {
        int c = 0;
        for (int r = 0; r < p_.m; ++r) {
            const auto q = static_cast<std::size_t>(r);
            if ((p_.geq[q] != 0) != geq) continue;
            const double gap = geq ? p_.rhs[q] - act_[q] : act_[q] - p_.rhs[q];
            if (gap > row_tol(p_.rhs[q])) ++c;
        }
        return c;
    }

    // the nodes whose groups may have got cheaper after `changed` flipped
    template <class F>
    void neighbours(const std::vector<int>& changed, bool added, F&& f) {
        for (const int u : changed) {
            const Adj& dep = added ? req_in_ : req_out_;
            for (i64 k = dep.begin(u); k < dep.end(u); ++k) f(dep.node[static_cast<std::size_t>(k)]);
            for (i64 k = pen_out_.begin(u); k < pen_out_.end(u); ++k) f(pen_out_.node[static_cast<std::size_t>(k)]);
            for (i64 k = pen_in_.begin(u); k < pen_in_.end(u); ++k) f(pen_in_.node[static_cast<std::size_t>(k)]);
        }
    }

    static bool later(const Heap& h, double key) {
        return !h.empty() && key > h.top().first + 1e-12 * std::max(1.0, std::abs(key));
    }

    double repair() {
        const int n = p_.n;
        // fixings, then no node without what it needs: zeros down the dependents, ones up the requirements
        for (int v = 0; v < n; ++v)
            if (fix_[static_cast<std::size_t>(v)] >= 0) z_[static_cast<std::size_t>(v)] = static_cast<std::uint8_t>(fix_[static_cast<std::size_t>(v)]);
        for (int v = 0; v < n; ++v)
            if (!z_[static_cast<std::size_t>(v)]) {
                stack_.assign(1, v);
                while (!stack_.empty()) {
                    const int u = stack_.back();
                    stack_.pop_back();
                    for (i64 k = req_in_.begin(u); k < req_in_.end(u); ++k) {
                        const int w = req_in_.node[static_cast<std::size_t>(k)];
                        if (z_[static_cast<std::size_t>(w)]) {
                            z_[static_cast<std::size_t>(w)] = 0;
                            stack_.push_back(w);
                        }
                    }
                }
            }
        for (int v = 0; v < n; ++v)
            if (z_[static_cast<std::size_t>(v)]) {
                stack_.assign(1, v);
                while (!stack_.empty()) {
                    const int u = stack_.back();
                    stack_.pop_back();
                    for (i64 k = req_out_.begin(u); k < req_out_.end(u); ++k) {
                        const int w = req_out_.node[static_cast<std::size_t>(k)];
                        if (!z_[static_cast<std::size_t>(w)]) {
                            z_[static_cast<std::size_t>(w)] = 1;
                            stack_.push_back(w);
                        }
                    }
                }
            }
        std::fill(act_.begin(), act_.end(), 0.0);
        for (int v = 0; v < n; ++v)
            if (z_[static_cast<std::size_t>(v)])
                for (i64 k = p_.col_start[static_cast<std::size_t>(v)]; k < p_.col_start[static_cast<std::size_t>(v) + 1]; ++k)
                    act_[static_cast<std::size_t>(p_.col_row[static_cast<std::size_t>(k)])] += p_.col_val[static_cast<std::size_t>(k)];
        obj_ = objective_of(z_);

        // 1. the <= rows: drop the groups that cost least per amount freed
        if (violated(false)) {
            Heap h;
            auto key = [&](int v, double& k) {
                if (!z_[static_cast<std::size_t>(v)] || !build(v, false)) return false;
                const double freed = cover(false);
                if (freed <= 0) return false;
                k = delta(false) / freed;
                return true;
            };
            double k;
            for (int v = 0; v < n; ++v)
                if (key(v, k)) h.push({k, v});
            while (violated(false)) {
                if (h.empty()) return kInf;
                const int v = h.top().second;
                h.pop();
                if (!key(v, k)) continue;
                if (later(h, k)) {
                    h.push({k, v});
                    continue;
                }
                apply(false, delta(false));
            }
        }
        // 2. the >= rows: add the groups that cover most per cost, keeping the <= rows
        std::vector<int> changed;
        if (violated(true)) {
            Heap h;
            auto key = [&](int v, double& k) {
                if (z_[static_cast<std::size_t>(v)] || !build(v, true)) return false;
                const double c = cover(true);
                if (c <= 0 || !fits()) return false;
                k = delta(true) / c;
                return true;
            };
            double k;
            for (int v = 0; v < n; ++v)
                if (key(v, k)) h.push({k, v});
            while (violated(true)) {
                if (h.empty()) return kInf;
                const int v = h.top().second;
                h.pop();
                if (!key(v, k)) continue;
                if (later(h, k)) {
                    h.push({k, v});
                    continue;
                }
                apply(true, delta(true));
                changed = group_;
                neighbours(changed, true, [&](int u) {
                    double ku;
                    if (key(u, ku)) h.push({ku, u});
                });
            }
        }
        // 3. add the groups that improve the objective, the most per share of the <= rows first
        {
            Heap h;
            auto key = [&](int v, double& k) {
                if (z_[static_cast<std::size_t>(v)] || !build(v, true)) return false;
                const double d = delta(true);
                if (d >= -1e-12 * std::max(1.0, std::abs(obj_)) || !fits()) return false;
                const double u = usage();
                k = u > 0 ? d / u : d * 1e30;
                return true;
            };
            double k;
            for (int v = 0; v < n; ++v)
                if (key(v, k)) h.push({k, v});
            while (!h.empty()) {
                const int v = h.top().second;
                h.pop();
                if (!key(v, k)) continue;
                if (later(h, k)) {
                    h.push({k, v});
                    continue;
                }
                apply(true, delta(true));
                changed = group_;
                neighbours(changed, true, [&](int u) {
                    double ku;
                    if (key(u, ku)) h.push({ku, u});
                });
            }
        }
        // 4. drop the groups that do not pay for themselves, the largest saving first, keeping the >= rows
        {
            Heap h;
            auto key = [&](int v, double& k) {
                if (!z_[static_cast<std::size_t>(v)] || !build(v, false)) return false;
                k = delta(false);
                return k < -1e-12 * std::max(1.0, std::abs(obj_)) && keeps_targets();
            };
            double k;
            for (int v = 0; v < n; ++v)
                if (key(v, k)) h.push({k, v});
            while (!h.empty()) {
                const int v = h.top().second;
                h.pop();
                if (!key(v, k)) continue;
                if (later(h, k)) {
                    h.push({k, v});
                    continue;
                }
                apply(false, k);
                changed = group_;
                neighbours(changed, false, [&](int u) {
                    double ku;
                    if (key(u, ku)) h.push({ku, u});
                });
            }
        }
        if (violated(true) || violated(false)) return kInf;
        obj_ = objective_of(z_);
        return obj_;
    }
};

// r = c - A'(s y): every node's reduced cost (s = +1 on >= rows, -1 on <= rows)
void reduced_costs(const Problem& p, const std::vector<double>& y, std::vector<double>& r, int threads) {
    const int n = p.n;
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int v = 0; v < n; ++v) {
        double c = p.cost[static_cast<std::size_t>(v)];
        for (i64 k = p.col_start[static_cast<std::size_t>(v)]; k < p.col_start[static_cast<std::size_t>(v) + 1]; ++k) {
            const auto q = static_cast<std::size_t>(p.col_row[static_cast<std::size_t>(k)]);
            c -= (p.geq[q] ? y[q] : -y[q]) * p.col_val[static_cast<std::size_t>(k)];
        }
        r[static_cast<std::size_t>(v)] = c;
    }
}

// The inner minimum of star requirements without pair terms, group by group: a root (a unit) is in
// when it is forced or its reduced cost plus its negative dependents' is below 0. gval: the group's value.
void star_inner(const std::vector<int>& roots, const Adj& req_in, const std::vector<std::int8_t>& fix,
                const std::vector<double>& r, std::vector<std::uint8_t>& z, std::vector<double>& gval, int threads) {
    const auto nr = static_cast<int>(roots.size());
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int q = 0; q < nr; ++q) {
        const int u = roots[static_cast<std::size_t>(q)];
        const std::int8_t fu = fix[static_cast<std::size_t>(u)];
        double open = r[static_cast<std::size_t>(u)];
        bool forced = fu == 1;
        for (i64 k = req_in.begin(u); k < req_in.end(u); ++k) {
            const int a = req_in.node[static_cast<std::size_t>(k)];
            const std::int8_t fa = fix[static_cast<std::size_t>(a)];
            if (fa == 1) {
                open += r[static_cast<std::size_t>(a)];
                forced = true;
            } else if (fa < 0 && r[static_cast<std::size_t>(a)] < 0) {
                open += r[static_cast<std::size_t>(a)];
            }
        }
        const bool in = fu != 0 && (forced || open < 0);
        z[static_cast<std::size_t>(u)] = in;
        for (i64 k = req_in.begin(u); k < req_in.end(u); ++k) {
            const int a = req_in.node[static_cast<std::size_t>(k)];
            const std::int8_t fa = fix[static_cast<std::size_t>(a)];
            z[static_cast<std::size_t>(a)] = in && (fa == 1 || (fa < 0 && r[static_cast<std::size_t>(a)] < 0));
        }
        gval[static_cast<std::size_t>(u)] = in ? open : 0.0;
    }
}

}  // namespace

void Problem::check() const {
    const auto nn = static_cast<std::size_t>(n);
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::invalid_argument(what);
    };
    need(cost.size() == nn && fixed.size() == nn, "closure: one cost and fixing per node");
    need(req_from.size() == req_to.size(), "closure: requirements");
    need(pen_i.size() == pen_j.size() && pen_w.size() == pen_i.size(), "closure: penalties");
    need(rhs.size() == static_cast<std::size_t>(m) && geq.size() == static_cast<std::size_t>(m), "closure: rows");
    need(col_start.size() == nn + 1 && col_row.size() == col_val.size() &&
             static_cast<std::size_t>(col_start.back()) == col_row.size(),
         "closure: row entries");
    for (std::size_t k = 0; k < req_from.size(); ++k)
        need(req_from[k] >= 0 && req_from[k] < n && req_to[k] >= 0 && req_to[k] < n, "closure: requirement nodes");
    for (std::size_t k = 0; k < pen_i.size(); ++k)
        need(pen_i[k] >= 0 && pen_i[k] < n && pen_j[k] >= 0 && pen_j[k] < n && pen_w[k] >= 0,
             "closure: penalties need nodes and weights >= 0");
    for (std::size_t k = 0; k < col_row.size(); ++k)
        need(col_row[k] >= 0 && col_row[k] < m && col_val[k] >= 0, "closure: row entries need coefficients >= 0");
}

Result solve(const Problem& p, const Options& o, const std::vector<std::uint8_t>& initial) {
    p.check();
    const Clock::time_point t0 = Clock::now();
    const Clock::time_point deadline =
        o.time_limit >= 0 ? t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(o.time_limit))
                          : Clock::time_point::max();
    auto seconds = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    const int n = p.n, m = p.m;
    Result res;

    const Adj req_out(n, p.req_from, p.req_to, nullptr), req_in(n, p.req_to, p.req_from, nullptr);
    const Adj pen_out(n, p.pen_i, p.pen_j, &p.pen_w), pen_in(n, p.pen_j, p.pen_i, &p.pen_w);

    // the fixings closed under the requirements: ones up, zeros down
    std::vector<std::int8_t> fix(p.fixed);
    {
        std::vector<int> stack;
        for (int v = 0; v < n; ++v) {
            if (fix[static_cast<std::size_t>(v)] != 1) continue;
            stack.assign(1, v);
            while (!stack.empty()) {
                const int u = stack.back();
                stack.pop_back();
                for (i64 k = req_out.begin(u); k < req_out.end(u); ++k) {
                    const int w = req_out.node[static_cast<std::size_t>(k)];
                    if (fix[static_cast<std::size_t>(w)] == 0) {
                        res.status = "infeasible";
                        res.seconds = seconds();
                        return res;
                    }
                    if (fix[static_cast<std::size_t>(w)] != 1) {
                        fix[static_cast<std::size_t>(w)] = 1;
                        stack.push_back(w);
                    }
                }
            }
        }
        for (int v = 0; v < n; ++v) {
            if (fix[static_cast<std::size_t>(v)] != 0) continue;
            stack.assign(1, v);
            while (!stack.empty()) {
                const int u = stack.back();
                stack.pop_back();
                for (i64 k = req_in.begin(u); k < req_in.end(u); ++k) {
                    const int w = req_in.node[static_cast<std::size_t>(k)];
                    if (fix[static_cast<std::size_t>(w)] == 1) {
                        res.status = "infeasible";
                        res.seconds = seconds();
                        return res;
                    }
                    if (fix[static_cast<std::size_t>(w)] != 0) {
                        fix[static_cast<std::size_t>(w)] = 0;
                        stack.push_back(w);
                    }
                }
            }
        }
    }
    // rows that cannot hold
    {
        std::vector<double> reach(static_cast<std::size_t>(m), 0.0), least(static_cast<std::size_t>(m), 0.0);
        for (int v = 0; v < n; ++v)
            for (i64 k = p.col_start[static_cast<std::size_t>(v)]; k < p.col_start[static_cast<std::size_t>(v) + 1]; ++k) {
                const auto r = static_cast<std::size_t>(p.col_row[static_cast<std::size_t>(k)]);
                if (fix[static_cast<std::size_t>(v)] != 0) reach[r] += p.col_val[static_cast<std::size_t>(k)];
                if (fix[static_cast<std::size_t>(v)] == 1) least[r] += p.col_val[static_cast<std::size_t>(k)];
            }
        for (int r = 0; r < m; ++r) {
            const auto q = static_cast<std::size_t>(r);
            if (p.geq[q] ? reach[q] < p.rhs[q] - row_tol(p.rhs[q]) : least[q] > p.rhs[q] + row_tol(p.rhs[q])) {
                res.status = "infeasible";
                res.seconds = seconds();
                return res;
            }
        }
    }

    // the requirements' shape: stars (every node needs at most one node, which needs none) without pair
    // terms are solved in closed form, anything else by a minimum cut
    bool star = p.pen_i.empty();
    for (int v = 0; v < n && star; ++v) {
        const i64 d = req_out.end(v) - req_out.begin(v);
        if (d > 1 || (d == 1 && req_out.end(req_out.node[static_cast<std::size_t>(req_out.begin(v))]) >
                                    req_out.begin(req_out.node[static_cast<std::size_t>(req_out.begin(v))])))
            star = false;
    }
    std::vector<int> roots;
    if (star)
        for (int v = 0; v < n; ++v)
            if (req_out.end(v) == req_out.begin(v)) roots.push_back(v);
    std::unique_ptr<MaxFlow> flow;
    if (!star) {
        std::vector<int> ei, ej;
        std::vector<double> cij, cji;
        for (std::size_t k = 0; k < p.pen_i.size(); ++k) {
            ei.push_back(p.pen_i[k]);
            ej.push_back(p.pen_j[k]);
            cij.push_back(p.pen_w[k]);
            cji.push_back(0.0);
        }
        for (std::size_t k = 0; k < p.req_from.size(); ++k) {
            ei.push_back(p.req_from[k]);
            ej.push_back(p.req_to[k]);
            cij.push_back(kBigArc);
            cji.push_back(0.0);
        }
        flow = std::make_unique<MaxFlow>(n, ei, ej, cij, cji);
    }
    double wsum = 0;
    for (const double w : p.pen_w) wsum += w;

    Search search(p, req_out, req_in, pen_out, pen_in, fix);
    double best = kInf;
    std::vector<double> weight(static_cast<std::size_t>(m), 1.0);
    auto take = [&](std::vector<std::uint8_t> z) {
        const double v = search.run(z, weight);
        if (v < best) {
            best = v;
            res.z = std::move(z);
        }
    };
    auto set_weights = [&](const std::vector<double>& y) {
        double price = 0, amount = 0;
        for (int r = 0; r < m; ++r)
            if (y[static_cast<std::size_t>(r)] > 0) {
                price += y[static_cast<std::size_t>(r)] * std::abs(p.rhs[static_cast<std::size_t>(r)]);
                amount += std::abs(p.rhs[static_cast<std::size_t>(r)]);
            }
        const double fallback = amount > 0 && price > 0 ? price / amount : 1.0;
        for (int r = 0; r < m; ++r)
            weight[static_cast<std::size_t>(r)] = y[static_cast<std::size_t>(r)] > 0 ? y[static_cast<std::size_t>(r)] : fallback;
    };
    if (!initial.empty()) {
        if (static_cast<int>(initial.size()) != n) throw std::invalid_argument("closure: one initial value per node");
        take(initial);
    }

    std::vector<double> r(static_cast<std::size_t>(n)), src(r.size()), snk(r.size()), gval(r.size());
    std::vector<std::uint8_t> z(static_cast<std::size_t>(n));
    std::vector<double> act(static_cast<std::size_t>(m));
    int evaluations = 0;
    auto reduced = [&](const std::vector<double>& y) { reduced_costs(p, y, r, o.threads); };
    // the inner minimum for the reduced costs r: z and its value
    auto inner = [&]() {
        double e = 0;
        if (star) {
            star_inner(roots, req_in, fix, r, z, gval, o.threads);
            for (const int u : roots) e += gval[static_cast<std::size_t>(u)];
        } else {
            double big = 1.0 + wsum;
            for (const double v : r) big += std::abs(v);
            for (int v = 0; v < n; ++v) {
                const auto q = static_cast<std::size_t>(v);
                src[q] = snk[q] = 0;
                if (fix[q] == 1) src[q] = big;
                else if (fix[q] == 0) snk[q] = big;
                else if (r[q] > 0) snk[q] = r[q];
                else src[q] = -r[q];
            }
            flow->solve(src, snk);
            for (int v = 0; v < n; ++v) {
                z[static_cast<std::size_t>(v)] = flow->source_side(v);
                if (z[static_cast<std::size_t>(v)]) e += r[static_cast<std::size_t>(v)];
            }
            for (std::size_t k = 0; k < p.pen_i.size(); ++k)
                if (z[static_cast<std::size_t>(p.pen_i[k])] && !z[static_cast<std::size_t>(p.pen_j[k])]) e += p.pen_w[k];
        }
        return e;
    };
    auto lagrangian = [&](const std::vector<double>& y, std::vector<double>& g) {
        reduced(y);
        double l = inner();
        std::fill(act.begin(), act.end(), 0.0);
        for (int v = 0; v < n; ++v)
            if (z[static_cast<std::size_t>(v)])
                for (i64 k = p.col_start[static_cast<std::size_t>(v)]; k < p.col_start[static_cast<std::size_t>(v) + 1]; ++k)
                    act[static_cast<std::size_t>(p.col_row[static_cast<std::size_t>(k)])] += p.col_val[static_cast<std::size_t>(k)];
        g.assign(static_cast<std::size_t>(m), 0.0);
        for (int k = 0; k < m; ++k) {
            const auto q = static_cast<std::size_t>(k);
            const double s = p.geq[q] ? 1.0 : -1.0;
            l += s * y[q] * p.rhs[q];
            g[q] = s * (p.rhs[q] - act[q]);
        }
        ++evaluations;
        set_weights(y);
        take(z);
        return l;
    };

    // the starting multipliers: a >= row's cost per amount in a greedy cover of it alone; a <= row's
    // gain per amount where a greedy filling of it alone stops
    std::vector<double> y(static_cast<std::size_t>(m), 0.0);
    {
        std::vector<double> eff(static_cast<std::size_t>(n));
        for (int v = 0; v < n; ++v) {
            double c = std::max(0.0, p.cost[static_cast<std::size_t>(v)]);
            for (i64 k = req_out.begin(v); k < req_out.end(v); ++k)
                c += std::max(0.0, p.cost[static_cast<std::size_t>(req_out.node[static_cast<std::size_t>(k)])]);
            eff[static_cast<std::size_t>(v)] = c;
        }
        std::vector<std::vector<std::pair<double, double>>> by(static_cast<std::size_t>(m));
        for (int v = 0; v < n; ++v) {
            if (fix[static_cast<std::size_t>(v)] == 0) continue;
            for (i64 k = p.col_start[static_cast<std::size_t>(v)]; k < p.col_start[static_cast<std::size_t>(v) + 1]; ++k) {
                const auto q = static_cast<std::size_t>(p.col_row[static_cast<std::size_t>(k)]);
                const double a = p.col_val[static_cast<std::size_t>(k)];
                if (a <= 0) continue;
                if (p.geq[q])
                    by[q].emplace_back(eff[static_cast<std::size_t>(v)] / a, a);
                else if (p.cost[static_cast<std::size_t>(v)] < 0)
                    by[q].emplace_back(p.cost[static_cast<std::size_t>(v)] / a, a);  // minus the gain per amount
            }
        }
        for (int k = 0; k < m; ++k) {
            auto& v = by[static_cast<std::size_t>(k)];
            std::sort(v.begin(), v.end());
            const double t = p.rhs[static_cast<std::size_t>(k)];
            double held = 0, spent = 0;
            if (p.geq[static_cast<std::size_t>(k)]) {
                for (const auto& [ratio, a] : v) {
                    if (held >= t) break;
                    held += a;
                    spent += ratio * a;
                }
                y[static_cast<std::size_t>(k)] = t > 0 && held > 0 ? spent / held : 0.0;
            } else {
                for (const auto& [ratio, a] : v) {
                    held += a;
                    if (held > t) {
                        y[static_cast<std::size_t>(k)] = -ratio;
                        break;
                    }
                }
            }
        }
    }

    // integral costs and weights: every objective value is an integer, and so the bound can be rounded up
    bool integral = true;
    auto whole = [](double v) { return std::abs(v - std::round(v)) <= 1e-9 * std::max(1.0, std::abs(v)); };
    for (const double c : p.cost) integral = integral && whole(c);
    for (const double w : p.pen_w) integral = integral && whole(w);
    auto rounded = [&](double b) {
        return integral && std::isfinite(b) ? std::ceil(b - 1e-9 * std::max(1.0, std::abs(b))) : b;
    };
    std::vector<double> g;
    const BundleResult br = bundle_maximize(
        y, lagrangian, [&](double bound) { return relative_gap(best, rounded(bound)) <= o.gap; }, evaluations,
        o.max_evaluations, deadline);
    const double bound = rounded(br.bound);
    // primal search: the multipliers scaled (more covered, less spent: less to repair)
    for (const double theta : {1.02, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 0.98, 0.95, 0.9}) {
        if (relative_gap(best, bound) <= o.gap || Clock::now() >= deadline) break;
        std::vector<double> yt(br.centre);
        for (double& v : yt) v *= theta;
        lagrangian(yt, g);
    }
    // every node's value in the inner minimum at the best multipliers and what forcing it the other way
    // adds to the bound; the nodes no better solution flips (the bound with the node flipped is above the
    // best solution)
    if (star && std::isfinite(best)) {
        reduced(br.centre);
        const double base = br.centre_value;
        const double tol = 1e-9 * std::max(1.0, std::abs(best));
        res.centre_z.assign(static_cast<std::size_t>(n), 0);
        res.flip.assign(static_cast<std::size_t>(n), kInf);
        for (const int u : roots) {
            const std::int8_t fu = fix[static_cast<std::size_t>(u)];
            double open = r[static_cast<std::size_t>(u)];
            bool forced = fu == 1;
            for (i64 k = req_in.begin(u); k < req_in.end(u); ++k) {
                const int a = req_in.node[static_cast<std::size_t>(k)];
                const std::int8_t fa = fix[static_cast<std::size_t>(a)];
                if (fa == 1) {
                    open += r[static_cast<std::size_t>(a)];
                    forced = true;
                } else if (fa < 0 && r[static_cast<std::size_t>(a)] < 0) {
                    open += r[static_cast<std::size_t>(a)];
                }
            }
            const bool in = fu != 0 && (forced || open < 0);
            const double gmin = in ? open : 0.0;
            res.centre_z[static_cast<std::size_t>(u)] = in;
            if (fu < 0 && !forced) res.flip[static_cast<std::size_t>(u)] = in ? -open : open;
            for (i64 k = req_in.begin(u); k < req_in.end(u); ++k) {
                const int a = req_in.node[static_cast<std::size_t>(k)];
                const std::int8_t fa = fix[static_cast<std::size_t>(a)];
                const double ra = r[static_cast<std::size_t>(a)];
                res.centre_z[static_cast<std::size_t>(a)] = in && (fa == 1 || (fa < 0 && ra < 0));
                if (fa >= 0) continue;
                if (in)
                    res.flip[static_cast<std::size_t>(a)] = std::abs(ra);
                else if (fu != 0)
                    res.flip[static_cast<std::size_t>(a)] = open - std::min(0.0, ra) + ra - gmin;  // the group opened
            }
        }
        res.fixable.assign(static_cast<std::size_t>(n), -1);
        for (int v = 0; v < n; ++v) {
            const double f = res.flip[static_cast<std::size_t>(v)];
            if (fix[static_cast<std::size_t>(v)] < 0 && std::isfinite(f) && rounded(base + f) >= best - tol)
                res.fixable[static_cast<std::size_t>(v)] = static_cast<std::int8_t>(res.centre_z[static_cast<std::size_t>(v)]);
        }
    }
    res.objective = best;
    res.bound = bound;
    res.gap = relative_gap(best, bound);
    res.evaluations = evaluations;
    if (!std::isfinite(best))
        res.status = "no_solution";
    else
        res.status = res.gap <= o.gap ? "optimal" : (br.why == "time" ? "time_limit" : "stalled");
    res.seconds = seconds();
    if (o.verbose)
        std::fprintf(stderr, "closure engine: %d evaluations, best %.10g, bound %.10g, gap %.3g (%s), %.2f s\n",
                     evaluations, best, bound, res.gap, br.why.c_str(), res.seconds);
    return res;
}

}  // namespace opt::closure
