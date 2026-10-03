// SPDX-License-Identifier: GPL-3.0-or-later
//
// Covering problems (prioritizr's minimum set and its kin):
//
//     min c' x   subject to   A x >= b,  lb <= x <= ub (finite),  some x integer
//
// with few rows (one per feature) and up to millions of columns (planning
// units). HiGHS's MIP stalls on such wide models (IDEAS.md, phase 0); here:
//
// 1. presolve: fixed columns and empty columns leave the model, rows already
//    met leave it (when their row has no negative coefficient), and rows that
//    cannot be met make the model infeasible;
// 2. the LP relaxation: on wide models, near-optimal duals first (the LP on a
//    2% sample of the columns, then a proximal bundle method on the
//    Lagrangian, whose dimension is the number of rows); then sifting: HiGHS
//    solves the LP on a working set of columns, the other columns sit at the
//    bound their reduced cost points to, every column is priced with the
//    duals (OpenMP, fixed blocks) and the ones whose bound breaks optimality
//    join the working set, until none does: then the solution is optimal for
//    the whole LP and its objective a valid lower bound;
// 3. integer columns: a basic LP solution has at most one fractional column
//    per row; they are rounded up (A >= 0 keeps the rows met), and then the
//    selected columns that are not needed are dropped, the most expensive
//    first;
// 4. if the gap between that solution and the LP bound is wider than asked
//    for, HiGHS's MIP goes on from it with the time left.
//
// Every step is deterministic: the result does not depend on the threads.
#pragma once

#include "opt/milp.hpp"

namespace opt {

// A minimisation with only ">=" rows (row_hi infinite, row_lo finite) and finite bounds.
bool is_covering(const Model& model);

// Solves a covering model (see above); options as solve_highs (threads: the
// pricing's threads; HiGHS runs on one).
SolveResult solve_covering(const Model& model, const SolveOptions& options);

}  // namespace opt
