// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/milp.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include "Highs.h"

namespace opt {

void Model::check() const {
    const auto nr = static_cast<std::size_t>(nrow);
    const auto nc = static_cast<std::size_t>(ncol);
    if (nrow < 0 || ncol < 0) throw std::invalid_argument("negative model size");
    if (start.size() != nr + 1 || start.front() != 0)
        throw std::invalid_argument("the row starts must have nrow + 1 entries, from 0");
    if (index.size() != value.size() || static_cast<i64>(index.size()) != start.back())
        throw std::invalid_argument("the matrix's indices and values do not match its row starts");
    if (obj.size() != nc || lb.size() != nc || ub.size() != nc || integer.size() != nc)
        throw std::invalid_argument("obj, lb, ub and integer must have one entry per column");
    if (row_lo.size() != nr || row_hi.size() != nr)
        throw std::invalid_argument("row_lo and row_hi must have one entry per row");
    for (i64 r = 0; r < nrow; ++r)
        if (start[r + 1] < start[r]) throw std::invalid_argument("the row starts must not decrease");
    for (const i64 j : index)
        if (j < 0 || j >= ncol) throw std::invalid_argument("a column index is out of range");
}

namespace {

using Clock = std::chrono::steady_clock;

constexpr i64 kMaxHighsInt = 2147483647;  // HiGHS is built with 32-bit indices

HighsInt to_highs(i64 v, const char* what) {
    if (v > kMaxHighsInt) throw std::invalid_argument(std::string("the model has too many ") + what);
    return static_cast<HighsInt>(v);
}

std::string model_size(const Model& m) {
    return std::to_string(m.nrow) + " rows, " + std::to_string(m.ncol) + " columns";
}

HighsLp highs_lp(const Model& m) {
    HighsLp lp;
    lp.num_col_ = to_highs(m.ncol, "columns");
    lp.num_row_ = to_highs(m.nrow, "rows");
    lp.sense_ = m.maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
    lp.offset_ = m.offset;
    lp.col_cost_ = m.obj;
    lp.col_lower_ = m.lb;
    lp.col_upper_ = m.ub;
    lp.row_lower_ = m.row_lo;
    lp.row_upper_ = m.row_hi;
    auto& a = lp.a_matrix_;
    a.format_ = MatrixFormat::kRowwise;
    a.num_col_ = lp.num_col_;
    a.num_row_ = lp.num_row_;
    to_highs(m.start.back(), "non-zeros");
    a.start_.assign(m.start.begin(), m.start.end());
    a.index_.assign(m.index.begin(), m.index.end());
    a.value_ = m.value;
    bool any = false;
    for (const auto t : m.integer) any = any || t != 0;
    if (any) {
        lp.integrality_.resize(m.integer.size());
        for (std::size_t j = 0; j < m.integer.size(); ++j)
            lp.integrality_[j] = m.integer[j] ? HighsVarType::kInteger : HighsVarType::kContinuous;
    }
    return lp;
}

// HiGHS's task scheduler belongs to the calling thread and keeps the thread
// count of its first use: start a new one when this call wants another.
thread_local int scheduler_threads = 0;

void use_threads(int threads) {
    if (scheduler_threads != 0 && scheduler_threads != threads) Highs::resetGlobalScheduler(true);
    scheduler_threads = threads;
}

template <class T>
void set(Highs& h, const std::string& name, const T& value) {
    if (h.setOptionValue(name, value) != HighsStatus::kOk)
        throw std::runtime_error("HiGHS rejected the option " + name);
}

}  // namespace

void use_highs_threads(int threads) { use_threads(threads); }

SolveResult solve_highs(const Model& model, const SolveOptions& options) {
    model.check();
    if (options.threads < 1) throw std::invalid_argument("threads must be at least 1");
    if (!options.start.empty() && options.start.size() != static_cast<std::size_t>(model.ncol))
        throw std::invalid_argument("the start solution must have one value per column");

    const auto t0 = Clock::now();
    if (options.time_limit <= 0) {  // no time at all (HiGHS rejects a negative limit)
        SolveResult r;
        r.status = "time_limit";
        return r;
    }
    use_threads(options.threads);
    Highs h;
    set(h, "output_flag", options.verbose);
    set(h, "threads", static_cast<HighsInt>(options.threads));
    set(h, "mip_rel_gap", options.gap);
    set(h, "presolve", std::string(options.presolve == Presolve::off  ? "off"
                                   : options.presolve == Presolve::on ? "on"
                                                                      : "choose"));
    if (std::isfinite(options.time_limit)) set(h, "time_limit", options.time_limit);
    if (options.relaxation) set(h, "solve_relaxation", true);
    if (std::isfinite(options.objective_target)) set(h, "objective_target", options.objective_target);

    if (h.passModel(highs_lp(model)) == HighsStatus::kError) {
        // say what HiGHS objects to
        std::string why;
        for (i64 j = 0; j < model.ncol && why.empty(); ++j) {
            if (std::isnan(model.obj[j]) || std::isinf(model.obj[j])) why = "objective of column " + std::to_string(j);
            else if (std::isnan(model.lb[j]) || std::isnan(model.ub[j]) || model.lb[j] > model.ub[j])
                why = "bounds of column " + std::to_string(j) + ": [" + std::to_string(model.lb[j]) + ", " +
                      std::to_string(model.ub[j]) + "]";
        }
        for (i64 r = 0; r < model.nrow && why.empty(); ++r)
            if (std::isnan(model.row_lo[r]) || std::isnan(model.row_hi[r]) || model.row_lo[r] > model.row_hi[r])
                why = "bounds of row " + std::to_string(r) + ": [" + std::to_string(model.row_lo[r]) + ", " +
                      std::to_string(model.row_hi[r]) + "]";
        for (std::size_t k = 0; k < model.value.size() && why.empty(); ++k)
            if (!std::isfinite(model.value[k])) why = "a coefficient of row-wise entry " + std::to_string(k);
        throw std::runtime_error("HiGHS rejected the model (" + model_size(model) + (why.empty() ? "" : "; " + why) + ")");
    }
    if (!options.start.empty()) {
        HighsSolution s;
        s.col_value = options.start;
        s.value_valid = true;
        h.setSolution(s);
    }

    // our clock: interrupt the simplex, the interior point method and the MIP at the deadline
    const bool timed = std::isfinite(options.time_limit);
    const auto deadline = t0 + std::chrono::duration_cast<Clock::duration>(
                                   std::chrono::duration<double>(timed ? options.time_limit : 0.0));
    bool timed_out = false;
    if (timed) {
        h.setCallback([&](int, const std::string&, const HighsCallbackOutput*, HighsCallbackInput* in, void*) {
            if (Clock::now() >= deadline) {
                timed_out = true;
                in->user_interrupt = true;
            }
        });
        h.startCallback(kCallbackSimplexInterrupt);
        h.startCallback(kCallbackIpmInterrupt);
        h.startCallback(kCallbackMipInterrupt);
    }

    const HighsStatus run = h.run();
    SolveResult r;
    r.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    const HighsModelStatus ms = h.getModelStatus();
    const HighsInfo& info = h.getInfo();
    switch (ms) {
        case HighsModelStatus::kOptimal: r.status = "optimal"; break;
        case HighsModelStatus::kTimeLimit: r.status = "time_limit"; break;
        case HighsModelStatus::kObjectiveTarget: r.status = "target"; break;
        case HighsModelStatus::kInterrupt: r.status = timed_out ? "time_limit" : "error: interrupted"; break;
        case HighsModelStatus::kInfeasible: r.status = "infeasible"; break;
        case HighsModelStatus::kUnbounded: r.status = "unbounded"; break;
        case HighsModelStatus::kUnboundedOrInfeasible: r.status = "infeasible_or_unbounded"; break;
        default: r.status = "error: " + h.modelStatusToString(ms);
    }
    if (run == HighsStatus::kError && r.status == "optimal") r.status = "error: HiGHS failed";
    r.has_solution = info.primal_solution_status == kSolutionStatusFeasible;
    if (r.has_solution) {
        r.x = h.getSolution().col_value;
        r.objective = info.objective_function_value;
    }
    const bool mip = h.getLp().isMip() && !options.relaxation;
    if (mip) {
        r.bound = info.mip_dual_bound;
        r.gap = info.mip_gap;
        r.nodes = static_cast<i64>(info.mip_node_count);
    } else if (r.status == "optimal") {
        r.bound = r.objective;
        r.gap = 0.0;
        if (info.dual_solution_status == kSolutionStatusFeasible) r.row_dual = h.getSolution().row_dual;
    }
    return r;
}

}  // namespace opt
