// SPDX-License-Identifier: GPL-3.0-or-later
//
// Mixed-integer linear models of the MILP prioritization and their solution
// on HiGHS (built into the module). The model is the compiler's output:
//
//     min / max  obj' x + offset
//     subject to row_lo <= A x <= row_hi,  lb <= x <= ub,  x_j integer where integer[j]
//
// with A in compressed rows. The time limit is kept by our own clock (the
// interrupt callbacks), not only by HiGHS's option: HiGHS does not look at
// its limit inside some steps on wide models (see IDEAS.md, phase 0).
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace opt {

using i64 = std::int64_t;

inline constexpr double kInf = std::numeric_limits<double>::infinity();

struct Model {
    i64 nrow = 0;
    i64 ncol = 0;
    std::vector<i64> start;  // nrow + 1
    std::vector<i64> index;  // column of each non-zero
    std::vector<double> value;
    std::vector<double> obj, lb, ub;      // ncol
    std::vector<double> row_lo, row_hi;   // nrow (-inf / +inf when free)
    std::vector<std::uint8_t> integer;    // ncol, 1 = integer (binary when its bounds are 0 and 1)
    bool maximize = false;
    double offset = 0.0;

    void check() const;  // sizes and indices; throws std::invalid_argument
};

enum class Presolve { choose, off, on };

struct SolveOptions {
    double gap = 0.1;                     // relative MIP gap
    double time_limit = kInf;             // seconds, kept by our clock
    int threads = 1;                      // HiGHS's threads in this call
    Presolve presolve = Presolve::choose;
    bool verbose = false;                 // HiGHS's log on the console
    bool relaxation = false;              // solve the LP relaxation only
    std::vector<double> start;            // a solution to start from (empty: none)
    double objective_target = -kInf;     // MIP: stop at a solution better than it (HiGHS's objective_target)
};

// What the solves of a thread are told to say how they go: the best solution's objective, the bound, the gap
// (NaN when not known) and the seconds; false when they are to stop.
using Watch = std::function<bool(double best, double bound, double gap, double seconds)>;

// While a WatchScope lives, the solves on its thread (HiGHS's, and the loops of the covering, wide and zones
// engines around theirs) ask its watch a few times a second how to go on; once it says no, they stop as at a
// deadline: their status is "time_limit", and `stopped()` says why.
class WatchScope {
  public:
    struct State;  // the watch and how it went (milp.cpp)
    explicit WatchScope(Watch watch);
    ~WatchScope();
    WatchScope(const WatchScope&) = delete;
    WatchScope& operator=(const WatchScope&) = delete;
    bool stopped() const;

  private:
    State* state_;
    State* previous_;
};

// Whether the solves of this thread were told to stop: asks the watch of the scope, if any, at most five times a
// second (told what is known of the solve: NaN when nothing).
bool stop_asked(double best = std::numeric_limits<double>::quiet_NaN(),
                double bound = std::numeric_limits<double>::quiet_NaN(),
                double gap = std::numeric_limits<double>::quiet_NaN());

// Whether a watch is asked on this thread.
bool watched();

struct SolveResult {
    // "optimal" (within the gap), "time_limit" (also when stopped by a WatchScope's watch), "target"
    // (objective_target reached), "infeasible", "unbounded", "infeasible_or_unbounded" or "error: <HiGHS's status>"
    std::string status;
    bool has_solution = false;
    std::vector<double> x;  // ncol, when has_solution
    double objective = std::numeric_limits<double>::quiet_NaN();
    double bound = std::numeric_limits<double>::quiet_NaN();  // dual bound (the LP's objective for an LP)
    double gap = std::numeric_limits<double>::quiet_NaN();
    i64 nodes = 0;
    double seconds = 0.0;
    std::vector<double> row_dual;  // nrow, for an LP solved to optimality (HiGHS's signs: c - A'y)
};

// Solves on HiGHS in the calling thread. HiGHS keeps one task scheduler per
// calling thread, sized by the first call; a call with another thread count
// in the same thread replaces it.
SolveResult solve_highs(const Model& model, const SolveOptions& options);

// Makes HiGHS's scheduler of the calling thread use `threads` threads (for code that calls HiGHS itself).
void use_highs_threads(int threads);

}  // namespace opt
