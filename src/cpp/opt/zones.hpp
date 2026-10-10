// SPDX-License-Identifier: GPL-3.0-or-later
//
// Covering problems with management zones (prioritizr's minimum set with zones):
//
//     min c' x   subject to   A x >= b             (the targets, A >= 0)
//                             sum_z x_zi <= 1      (one row per unit; = 1 with mandatory allocation)
//                             lb <= x <= ub,  binary or continuous unit columns
//
// over the columns z * n + i (unit i in zone z, zone after zone) and nothing else. The GUB rows make
// the model no covering model, and HiGHS's MIP stalls on it from 1e5 units (IDEAS.md, zones, phase 0).
// Here, as in opt/cover.cpp with the GUB rows kept out of the Lagrangian:
//
// 1. the Lagrangian of the targets decomposes by unit: every unit takes the zones of least reduced cost
//    (a continuous knapsack over its zones: the best zone, or none when no reduced cost is negative,
//    with 0 / 1 bounds), so L(y) is cheap and equals the LP bound at its maximum (the per-unit
//    polytopes are integral); a box-step bundle method (opt/bundle.hpp) finds near-optimal duals;
// 2. sifting over units: HiGHS solves the LP on a working set of units (their columns and GUB rows),
//    the other units at their per-unit optimum; units whose optimum the duals change join, until none
//    does: the LP is then optimal, its objective the bound;
// 3. binary columns: every fractional unit takes its largest share, then the targets left unmet are
//    repaired greedily (moves of a unit into a zone, by amount gained per cost) and the units not
//    needed leave, or move to a cheaper zone, the most expensive first;
// 4. if the gap is still wider than asked for, HiGHS's MIP goes on from that solution on the model
//    left after Lagrangian fixing (a unit-zone column whose forcing raises the bound beyond the
//    incumbent keeps its value), within the time left and the size HiGHS keeps time on.
//
// Every step is deterministic: the result does not depend on the threads.
#pragma once

#include "opt/milp.hpp"

namespace opt {

// Whether the model has the form above (n_units units, the GUB rows recognised by their entries).
bool is_zone_covering(const Model& model, i64 n_units);

// Solves it (see above); options as solve_highs (threads: the pricing's; HiGHS runs on one).
SolveResult solve_zone_covering(const Model& model, i64 n_units, const SolveOptions& options);

}  // namespace opt
