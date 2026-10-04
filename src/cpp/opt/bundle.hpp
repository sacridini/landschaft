// SPDX-License-Identifier: GPL-3.0-or-later
//
// A box-step bundle method for Lagrangian duals: maximises a concave function L(y) over y >= 0 given its
// value and a supergradient at any point. Every step solves the master LP
//
//     max t  subject to  t <= L(y_j) + g_j'(y - y_j) for the cuts j,  y >= 0 within a box around the centre,
//
// on HiGHS; the trial point becomes the centre when it gains at least a tenth of what the master
// predicted (the box doubles), otherwise the box halves. Used by the cut engine (opt/cut.cpp) and the
// closure engine (opt/closure.cpp).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include "opt/milp.hpp"

namespace opt {

struct BundleResult {
    std::vector<double> centre;  // the best multipliers found
    double centre_value = 0.0;   // L there
    double bound = 0.0;          // the best value of L seen
    std::string why;             // "gap" (stop() said so), "time", "box", "master ...", "converged", "stalled",
                                 // "evaluations"
};

// eval(y, g) -> L(y), g := a supergradient (counted in `evaluations`); stop(bound) -> end now.
template <class Eval, class Stop>
BundleResult bundle_maximize(std::vector<double> y, Eval&& eval, Stop&& stop, int& evaluations, int max_evaluations,
                             std::chrono::steady_clock::time_point deadline) {
    const int m = static_cast<int>(y.size());
    struct Cut {
        double value;
        std::vector<double> y, g;
    };
    std::vector<Cut> cuts;
    std::vector<double> g;
    BundleResult res;
    res.centre = y;
    res.centre_value = eval(res.centre, g);
    std::vector<double> centre_g = g;
    double bound = res.centre_value;
    cuts.push_back({res.centre_value, res.centre, g});
    double scale = 0;
    for (const double v : y) scale = std::max(scale, std::abs(v));
    double box = scale > 0 ? 0.5 * scale : 1.0;
    const double box_min = 1e-12 * std::max(1.0, scale);
    res.why = "evaluations";
    std::vector<double> history;
    while (evaluations < max_evaluations) {
        if (stop(bound)) {
            res.why = "gap";
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            res.why = "time";
            break;
        }
        if (box <= box_min) {
            res.why = "box";
            break;
        }
        Model master;
        master.ncol = m + 1;
        master.maximize = true;
        master.obj.assign(static_cast<std::size_t>(m + 1), 0.0);
        master.obj[static_cast<std::size_t>(m)] = 1.0;
        for (int k = 0; k < m; ++k) {
            master.lb.push_back(std::max(0.0, res.centre[static_cast<std::size_t>(k)] - box));
            master.ub.push_back(res.centre[static_cast<std::size_t>(k)] + box);
        }
        master.lb.push_back(-kInf);
        master.ub.push_back(kInf);
        master.integer.assign(static_cast<std::size_t>(m + 1), 0);
        master.start.push_back(0);
        for (const Cut& ct : cuts) {
            double rhs = ct.value;
            for (int k = 0; k < m; ++k) {
                rhs -= ct.g[static_cast<std::size_t>(k)] * ct.y[static_cast<std::size_t>(k)];
                if (ct.g[static_cast<std::size_t>(k)] != 0.0) {
                    master.index.push_back(k);
                    master.value.push_back(-ct.g[static_cast<std::size_t>(k)]);
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
            res.why = "master " + mr.status;
            break;
        }
        std::vector<double> trial(mr.x.begin(), mr.x.begin() + m);
        for (double& v : trial) v = std::max(0.0, v);
        const double predicted = mr.x[static_cast<std::size_t>(m)] - res.centre_value;
        if (predicted <= 1e-10 * std::max(1.0, std::abs(res.centre_value))) {
            res.why = "converged";
            break;
        }
        const double value = eval(trial, g);
        bound = std::max(bound, value);
        cuts.push_back({value, trial, g});
        if (value >= res.centre_value + 0.1 * predicted) {  // serious step
            res.centre = trial;
            res.centre_value = value;
            centre_g = g;
            box *= 2.0;
        } else {
            box *= 0.5;
        }
        if (cuts.size() > static_cast<std::size_t>(std::max(200, 4 * m))) {
            std::vector<Cut> kept;
            kept.push_back({res.centre_value, res.centre, centre_g});
            kept.insert(kept.end(), cuts.end() - static_cast<std::ptrdiff_t>(2 * m), cuts.end());
            cuts = std::move(kept);
        }
        history.push_back(bound);
        if (history.size() > 30 && bound - history[history.size() - 31] <= 1e-6 * std::max(1.0, std::abs(bound))) {
            res.why = "stalled";
            break;
        }
    }
    res.bound = bound;
    return res;
}

}  // namespace opt
