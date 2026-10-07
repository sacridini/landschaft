// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/cut.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>

#include "opt/bundle.hpp"
#include "opt/maxflow.hpp"
#include "opt/milp.hpp"

namespace opt::cut {

namespace {

using Clock = std::chrono::steady_clock;

// the bundle's decisions see the repairs of the evaluations up to this many back (at most this many are
// pending): a constant, so the result does not depend on the number of threads
constexpr std::size_t kRepairLag = 8;

double relative_gap(double upper, double lower) {
    if (!std::isfinite(upper) || !std::isfinite(lower)) return kInf;
    const double g = upper - lower;
    if (g <= 0) return 0.0;
    return g / std::max(std::abs(upper), 1e-10);
}

}  // namespace

Result solve(const marxan::Data& d, const Options& o, const std::vector<int>& initial) {
    const Clock::time_point t0 = Clock::now();
    const Clock::time_point deadline =
        o.time_limit >= 0 ? t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(o.time_limit))
                          : Clock::time_point::max();
    auto seconds = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    const int n = d.npu, m = d.nsp;
    Result res;

    // the pairs (each once) and the units' own terms (cost and fixed boundary)
    std::vector<int> pi, pj;
    std::vector<double> pw;
    for (int i = 0; i < n; ++i)
        for (std::int64_t k = d.nb_start[i]; k < d.nb_start[i + 1]; ++k) {
            if (d.nb_cost[k] < 0) throw std::invalid_argument("the cut engine needs boundaries >= 0");
            if (i < d.nb[k]) {
                pi.push_back(i);
                pj.push_back(d.nb[k]);
                pw.push_back(d.nb_cost[k]);
            }
        }
    double wsum = 0;
    for (const double w : pw) wsum += w;
    std::vector<double> c(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) c[i] = d.cost[i] + d.fixedcost[i];

    // targets that the units not locked out cannot meet
    std::vector<double> reach(static_cast<std::size_t>(m), 0.0);
    for (int i = 0; i < n; ++i)
        if (d.status[i] != 3)
            for (std::int64_t k = d.offset[i]; k < d.offset[i + 1]; ++k) reach[d.spindex[k]] += d.amount[k];
    for (int k = 0; k < m; ++k)
        if (reach[k] < d.target[k] - 1e-9 * std::max(1.0, std::abs(d.target[k]))) {
            res.status = "infeasible";
            res.seconds = seconds();
            return res;
        }

    MaxFlow flow(n, pi, pj, pw, pw);
    const marxan::Repair repair(d, 1.0);
    double best = kInf;
    // the repairs, in evaluation order: each on a thread of its own (at most o.threads at a time), their
    // results taken in order once more than kRepairLag are pending (or all, at the end)
    struct Pending {
        std::shared_ptr<std::vector<int>> R;
        std::future<double> value;
    };
    std::deque<Pending> pending;
    const std::size_t running_max = static_cast<std::size_t>(std::max(1, o.threads));
    auto settle = [&](std::size_t keep) {
        while (pending.size() > keep) {
            Pending& p = pending.front();
            const double v = p.value.get();
            if (v < best) {
                best = v;
                res.x.assign(static_cast<std::size_t>(n), 0);
                for (int i = 0; i < n; ++i) res.x[i] = (*p.R)[i] == 1 || (*p.R)[i] == 2;
            }
            pending.pop_front();
        }
    };
    auto take = [&](const std::vector<int>& R) {
        auto copy = std::make_shared<std::vector<int>>(R);
        if (running_max == 1) {
            std::promise<double> done;
            done.set_value(repair(*copy));
            pending.push_back({copy, done.get_future()});
        } else {
            // no more than o.threads running: wait for the oldest still running
            std::size_t running = 0;
            for (const Pending& p : pending)
                running += p.value.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
            for (Pending& p : pending) {
                if (running < running_max) break;
                if (p.value.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                    p.value.wait();
                    --running;
                }
            }
            pending.push_back({copy, std::async(std::launch::async, [&repair, copy] { return repair(*copy); })});
        }
        settle(kRepairLag);
    };
    std::vector<int> R(static_cast<std::size_t>(n));
    if (!initial.empty()) {
        if (static_cast<int>(initial.size()) != n) throw std::invalid_argument("one initial value per unit");
        for (int i = 0; i < n; ++i) R[i] = d.status[i] >= 2 ? d.status[i] : (initial[i] != 0);
        take(R);
    }

    std::vector<double> u(static_cast<std::size_t>(n)), src(u.size()), snk(u.size());
    int evaluations = 0;
    // L(y) and its subgradient t - A x(y); the cut's selection repaired on the way
    auto lagrangian = [&](const std::vector<double>& y, std::vector<double>& g) {
        double big = 1.0 + wsum;
        for (int i = 0; i < n; ++i) {
            double v = c[i];
            for (std::int64_t k = d.offset[i]; k < d.offset[i + 1]; ++k) v -= y[d.spindex[k]] * d.amount[k];
            u[i] = v;
            big += std::abs(v);
        }
        for (int i = 0; i < n; ++i) {
            src[i] = snk[i] = 0;
            if (d.status[i] == 2) src[i] = big;
            else if (d.status[i] == 3) snk[i] = big;
            else if (u[i] > 0) snk[i] = u[i];
            else src[i] = -u[i];
        }
        flow.solve(src, snk);
        double e = 0;
        g.assign(d.target.begin(), d.target.end());
        for (int i = 0; i < n; ++i) {
            const bool in = flow.source_side(i);
            R[i] = d.status[i] >= 2 ? d.status[i] : static_cast<int>(in);
            if (!in) continue;
            e += u[i];
            for (std::int64_t k = d.offset[i]; k < d.offset[i + 1]; ++k) g[d.spindex[k]] -= d.amount[k];
        }
        for (std::size_t k = 0; k < pw.size(); ++k)
            if (flow.source_side(pi[k]) != flow.source_side(pj[k])) e += pw[k];
        double l = e;
        for (int k = 0; k < m; ++k) l += y[k] * d.target[k];
        ++evaluations;
        take(R);
        return l;
    };

    // the starting multipliers: every feature's cost per amount in a greedy cover of its target alone
    std::vector<double> y(static_cast<std::size_t>(m), 0.0);
    {
        std::vector<std::vector<std::pair<double, double>>> by(static_cast<std::size_t>(m));  // (ratio, amount)
        for (int i = 0; i < n; ++i)
            if (d.status[i] != 3)
                for (std::int64_t k = d.offset[i]; k < d.offset[i + 1]; ++k)
                    if (d.amount[k] > 0) by[d.spindex[k]].emplace_back(std::max(c[i], 0.0) / d.amount[k], d.amount[k]);
        for (int k = 0; k < m; ++k) {
            auto& v = by[k];
            std::sort(v.begin(), v.end());
            double held = 0, spent = 0;
            for (const auto& [ratio, a] : v) {
                if (held >= d.target[k]) break;
                held += a;
                spent += ratio * a;
            }
            y[k] = d.target[k] > 0 && held > 0 ? spent / held : 0.0;
        }
    }

    // the multipliers by the box-step bundle method (opt/bundle.hpp)
    std::vector<double> g;
    const BundleResult br = bundle_maximize(
        y, lagrangian, [&](double bound) { return relative_gap(best, bound) <= o.gap; }, evaluations,
        o.max_evaluations, deadline);
    double bound = br.bound;
    const std::string& why = br.why;
    settle(0);
    // primal search: the cuts of the best multipliers scaled up (more units in, less to repair)
    const double bound_value = bound;
    std::vector<double> best_y = br.centre;
    for (const double theta : {1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0}) {
        if (relative_gap(best, bound) <= o.gap || Clock::now() >= deadline) break;
        std::vector<double> yt(best_y);
        for (double& v : yt) v *= theta;
        lagrangian(yt, g);
    }
    settle(0);
    bound = bound_value;
    res.objective = best;
    res.bound = bound;
    res.gap = relative_gap(best, bound);
    res.evaluations = evaluations;
    res.status = res.gap <= o.gap ? "optimal" : (why == "time" ? "time_limit" : "stalled");
    res.seconds = seconds();
    if (o.verbose)
        std::fprintf(stderr, "cut engine: %d evaluations, best %.10g, bound %.10g, gap %.3g (%s), %.2f s\n",
                     evaluations, best, bound, res.gap, why.c_str(), res.seconds);
    return res;
}

}  // namespace opt::cut
