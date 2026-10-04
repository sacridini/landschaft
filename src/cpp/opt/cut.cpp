// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/cut.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "opt/maxflow.hpp"
#include "opt/milp.hpp"

namespace opt::cut {

namespace {

using Clock = std::chrono::steady_clock;

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
    auto take = [&](std::vector<int>& R) {
        const double v = repair(R);
        if (v < best) {
            best = v;
            res.x.assign(static_cast<std::size_t>(n), 0);
            for (int i = 0; i < n; ++i) res.x[i] = R[i] == 1 || R[i] == 2;
        }
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

    // box-step bundle (opt/wide.cpp's): master LP max t s.t. t <= L_j + g_j'(y - y_j), y >= 0 in a box
    struct Cut {
        double value;
        std::vector<double> y, g;
    };
    std::vector<Cut> cuts;
    std::vector<double> g;
    std::vector<double> centre = y;
    double centre_value = lagrangian(centre, g);
    std::vector<double> centre_g = g;
    double bound = centre_value;
    cuts.push_back({centre_value, centre, g});
    double scale = 0;
    for (const double v : y) scale = std::max(scale, std::abs(v));
    double box = scale > 0 ? 0.5 * scale : 1.0;
    const double box_min = 1e-12 * std::max(1.0, scale);
    std::string why = "evaluations";
    std::vector<double> history;
    while (evaluations < o.max_evaluations) {
        if (relative_gap(best, bound) <= o.gap) {
            why = "gap";
            break;
        }
        if (Clock::now() >= deadline) {
            why = "time";
            break;
        }
        if (box <= box_min) {
            why = "box";
            break;
        }
        Model master;
        master.ncol = m + 1;
        master.maximize = true;
        master.obj.assign(static_cast<std::size_t>(m + 1), 0.0);
        master.obj[m] = 1.0;
        for (int k = 0; k < m; ++k) {
            master.lb.push_back(std::max(0.0, centre[k] - box));
            master.ub.push_back(centre[k] + box);
        }
        master.lb.push_back(-kInf);
        master.ub.push_back(kInf);
        master.integer.assign(static_cast<std::size_t>(m + 1), 0);
        master.start.push_back(0);
        for (const Cut& ct : cuts) {
            double rhs = ct.value;
            for (int k = 0; k < m; ++k) {
                rhs -= ct.g[k] * ct.y[k];
                if (ct.g[k] != 0.0) {
                    master.index.push_back(k);
                    master.value.push_back(-ct.g[k]);
                }
            }
            master.index.push_back(m);
            master.value.push_back(1.0);
            master.start.push_back(static_cast<i64>(master.index.size()));
            master.row_lo.push_back(-kInf);
            master.row_hi.push_back(rhs);
        }
        master.nrow = static_cast<i64>(master.row_lo.size());
        SolveOptions so;
        so.threads = 1;
        const SolveResult mr = solve_highs(master, so);
        if (mr.status != "optimal") {
            why = "master " + mr.status;
            break;
        }
        std::vector<double> trial(mr.x.begin(), mr.x.begin() + m);
        for (double& v : trial) v = std::max(0.0, v);
        const double predicted = mr.x[static_cast<std::size_t>(m)] - centre_value;
        if (predicted <= 1e-10 * std::max(1.0, std::abs(centre_value))) {
            why = "converged";
            break;
        }
        const double value = lagrangian(trial, g);
        bound = std::max(bound, value);
        cuts.push_back({value, trial, g});
        if (value >= centre_value + 0.1 * predicted) {  // serious step
            centre = trial;
            centre_value = value;
            centre_g = g;
            box *= 2.0;
        } else {
            box *= 0.5;
        }
        if (cuts.size() > static_cast<std::size_t>(std::max(200, 4 * m))) {
            std::vector<Cut> kept;
            kept.push_back({centre_value, centre, centre_g});
            kept.insert(kept.end(), cuts.end() - static_cast<std::ptrdiff_t>(2 * m), cuts.end());
            cuts = std::move(kept);
        }
        history.push_back(bound);
        if (history.size() > 30 && bound - history[history.size() - 31] <= 1e-6 * std::max(1.0, std::abs(bound))) {
            why = "stalled";
            break;
        }
    }
    // primal search: the cuts of the best multipliers scaled up (more units in, less to repair)
    const double bound_value = bound;
    std::vector<double> best_y = centre;
    for (const double theta : {1.02, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0}) {
        if (relative_gap(best, bound) <= o.gap || Clock::now() >= deadline) break;
        std::vector<double> yt(best_y);
        for (double& v : yt) v *= theta;
        lagrangian(yt, g);
    }
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
