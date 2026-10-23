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

struct WatchScope::State {
    Watch watch;
    Clock::time_point t0;
    Clock::time_point told;
    bool stopped = false;
};

namespace {
thread_local WatchScope::State* watching = nullptr;
}

WatchScope::WatchScope(Watch watch)
    : state_(new State{std::move(watch), Clock::now(), Clock::now() - std::chrono::milliseconds(200)}),
      previous_(watching) {
    watching = state_;
}

WatchScope::~WatchScope() {
    watching = previous_;
    delete state_;
}

bool WatchScope::stopped() const { return state_->stopped; }

bool watched() { return watching != nullptr && watching->watch; }

bool stop_asked(double best, double bound, double gap) {
    WatchScope::State* s = watching;
    if (s == nullptr || !s->watch) return false;
    if (s->stopped) return true;
    const auto now = Clock::now();
    if (now - s->told < std::chrono::milliseconds(200)) return false;
    s->told = now;
    if (!s->watch(best, bound, gap, std::chrono::duration<double>(now - s->t0).count())) s->stopped = true;
    return s->stopped;
}

double objective_steps(const Model& model) {
    if (model.offset != 0.0) return 0.0;
    for (i64 j = 0; j < model.ncol; ++j)
        if (!model.integer[static_cast<std::size_t>(j)] && model.obj[static_cast<std::size_t>(j)] != 0.0) return 0.0;
    for (const double d : {1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 20.0, 25.0, 50.0, 100.0}) {
        bool all = true;
        for (i64 j = 0; j < model.ncol && all; ++j) {
            const double s = model.obj[static_cast<std::size_t>(j)] * d;
            all = std::abs(s - std::round(s)) <= 1e-9 * std::max(1.0, std::abs(s));
        }
        if (all) return d;
    }
    return 0.0;
}

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
    const bool separating = options.separator && !options.relaxation;
    set(h, "presolve", std::string(options.presolve == Presolve::off || separating ? "off"
                                   : options.presolve == Presolve::on              ? "on"
                                                                                   : "choose"));
    if (separating) h.setUserSeparator(options.separator);
    for (const auto& [name, value] : options.highs_options)
        if (h.setOptionValue(name, value) != HighsStatus::kOk)
            throw std::invalid_argument("HiGHS rejected the option " + name + " = " + value);
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

    // our clock: interrupt the simplex, the interior point method and the MIP at the deadline; the watch told how
    // the solve goes a few times a second, and asked whether to go on
    const bool timed = std::isfinite(options.time_limit);
    const auto deadline = t0 + std::chrono::duration_cast<Clock::duration>(
                                   std::chrono::duration<double>(timed ? options.time_limit : 0.0));
    bool timed_out = false;
    if (timed || watched()) {
        h.setCallback([&](int type, const std::string&, const HighsCallbackOutput* out, HighsCallbackInput* in, void*) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const bool mip = type == kCallbackMipInterrupt && out != nullptr;
            const double best = mip && std::isfinite(out->mip_primal_bound) ? out->mip_primal_bound : nan;
            const double bound = mip ? out->mip_dual_bound : nan;
            const double gap = mip && std::isfinite(best) ? out->mip_gap : nan;
            if ((timed && Clock::now() >= deadline) || stop_asked(best, bound, gap)) {
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
        // every objective value a multiple of 1 / d: so is the optimum, and the bound rounds to one
        const double d = objective_steps(model);
        if (d > 0.0 && std::isfinite(r.bound)) {
            const double s = r.bound * d, tol = 1e-9 * std::max(1.0, std::abs(s));
            r.bound = (model.maximize ? std::floor(s + tol) : std::ceil(s - tol)) / d;
            if (r.has_solution) {
                const double g = model.maximize ? r.bound - r.objective : r.objective - r.bound;
                r.gap = g <= 0 ? 0.0 : g / std::max(std::abs(r.objective), 1e-10);
                if (r.status == "time_limit" && r.gap <= options.gap) r.status = "optimal";
            }
        }
    } else if (r.status == "optimal") {
        r.bound = r.objective;
        r.gap = 0.0;
        if (info.dual_solution_status == kSolutionStatusFeasible) r.row_dual = h.getSolution().row_dual;
    }
    return r;
}

RootCuts separate_root(const Model& model, const Separator& separator, const SolveOptions& options, double min_rise,
                       int stall_rounds) {
    model.check();
    if (!separator) throw std::invalid_argument("separate_root needs a separator");
    if (options.threads < 1) throw std::invalid_argument("threads must be at least 1");
    const auto t0 = Clock::now();
    RootCuts out;
    const bool timed = std::isfinite(options.time_limit);
    const auto deadline = t0 + std::chrono::duration_cast<Clock::duration>(
                                   std::chrono::duration<double>(timed ? std::max(options.time_limit, 0.0) : 0.0));
    auto left = [&]() { return std::chrono::duration<double>(deadline - Clock::now()).count(); };
    if (timed && left() <= 0) {
        out.status = "time_limit";
        return out;
    }
    use_threads(options.threads);
    Highs h;
    set(h, "output_flag", options.verbose);
    set(h, "threads", static_cast<HighsInt>(options.threads));
    set(h, "presolve", std::string("off"));
    HighsLp lp = highs_lp(model);
    lp.integrality_.clear();
    if (h.passModel(std::move(lp)) == HighsStatus::kError)
        throw std::runtime_error("HiGHS rejected the model (" + model_size(model) + ")");
    bool timed_out = false;
    if (timed || watched()) {
        h.setCallback([&](int, const std::string&, const HighsCallbackOutput*, HighsCallbackInput* in, void*) {
            if ((timed && Clock::now() >= deadline) || stop_asked()) {
                timed_out = true;
                in->user_interrupt = true;
            }
        });
        h.startCallback(kCallbackSimplexInterrupt);
        h.startCallback(kCallbackIpmInterrupt);
    }
    const double sign = model.maximize ? -1.0 : 1.0;
    double last = -kInf;
    int stalled = 0;
    std::vector<int> st, idx;
    std::vector<double> val, up, lo;
    for (;;) {
        h.run();
        const HighsModelStatus ms = h.getModelStatus();
        if (ms != HighsModelStatus::kOptimal) {
            if (timed_out || ms == HighsModelStatus::kTimeLimit) out.status = "time_limit";
            else if (ms == HighsModelStatus::kInfeasible) out.status = "infeasible";
            else out.status = "error: " + h.modelStatusToString(ms);
            break;
        }
        out.status = "optimal";
        const double obj = h.getInfo().objective_function_value;
        out.bound = obj;
        const double v = sign * obj;
        if (std::isfinite(last)) {
            stalled = v - last < min_rise * std::max(std::abs(v), 1.0) ? stalled + 1 : 0;
            if (stalled >= stall_rounds) break;
        }
        last = std::max(last, v);
        if (timed && left() <= 0) {
            out.status = "time_limit";
            break;
        }
        st.assign(1, 0);
        idx.clear();
        val.clear();
        up.clear();
        separator(h.getSolution().col_value, st, idx, val, up);
        if (up.empty()) break;
        ++out.rounds;
        lo.assign(up.size(), -kInf);
        if (h.addRows(static_cast<HighsInt>(up.size()), lo.data(), up.data(), static_cast<HighsInt>(idx.size()),
                      st.data(), idx.data(), val.data()) == HighsStatus::kError)
            throw std::runtime_error("HiGHS rejected a cut");
        const int base = out.start.back();
        for (std::size_t k = 1; k < st.size(); ++k) out.start.push_back(base + st[k]);
        out.index.insert(out.index.end(), idx.begin(), idx.end());
        out.value.insert(out.value.end(), val.begin(), val.end());
        out.upper.insert(out.upper.end(), up.begin(), up.end());
    }
    out.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    return out;
}

}  // namespace opt
