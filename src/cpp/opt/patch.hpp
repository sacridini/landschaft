// SPDX-License-Identifier: GPL-3.0-or-later
//
// A primal heuristic for minimum sets with contiguity (one patch):
//
// 1. patches grown from many seeds (in parallel): every step adds the unit with the greatest share of the
//    targets' shortfall per cost of the cheapest path to it from the patch (Dijkstra over the units, a unit's
//    cost paid on entering it), with the path; and from a given selection, its patches joined by the
//    cheapest paths from the largest (a node-weighted Steiner tree heuristic);
// 2. every patch pruned: the costliest units first, a unit leaves while the targets stay met and the patch
//    stays one (it is not a cut vertex of the patch);
// 3. the cheapest few improved by moves until no move pays or the time is up: a neighbouring unit enters
//    and the patch is pruned again (kept if cheaper), or a unit leaves, the shortfall is repaired, the
//    patches joined again and pruned;
// 4. the best then perturbed until the time is up (an iterated local search): in rounds of 16 (the same
//    with any number of threads), a piece of 10 to 30 % of the patch around a unit is removed and the
//    patch grown back with noise on the growth's scores (and the features' weights), or (one in four) a new patch grown with noise
//    from a unit drawn at random; each pruned and improved by the moves; the cheapest of a round replaces
//    the best if cheaper; until the time is up or `stall_rounds` rounds in a row found nothing cheaper.
//
// Every patch it returns is feasible (one patch, targets met, locked units kept).
#pragma once

#include <cstdint>
#include <vector>

#include "opt/milp.hpp"

namespace opt {

struct PatchProblem {
    int n = 0;                         // units
    std::vector<double> cost;          // n, >= 0
    std::vector<i64> start;            // n + 1: every unit's features (CSR by unit)
    std::vector<int> feature;
    std::vector<double> amount;        // >= 0
    std::vector<double> target;        // per feature (rows >=)
    std::vector<i64> adj_start;        // n + 1: every unit's neighbours (both directions)
    std::vector<int> adj;
    std::vector<std::uint8_t> locked_in, locked_out;  // n (empty: none)

    void check() const;  // throws std::invalid_argument
};

struct PatchResult {
    std::vector<std::uint8_t> selection;  // n; empty when no patch was found
    double cost = 0.0;
    int moves = 0;                        // improving moves made
    int rounds = 0;                       // rounds of perturbations
    double seconds = 0.0;
};

// The cheapest patch found within `time_limit` seconds: from `start` (n, 1 = selected; empty: none; repaired
// and joined if need be) and from every seed (the units holding the most of the targets per cost first; at most
// `max_seeds`, -1: all) grown on `threads` threads, the cheapest few then improved by the moves.
// The perturbations stop after `stall_rounds` rounds in a row that found nothing cheaper.
PatchResult improve_patch(const PatchProblem& p, const std::vector<std::uint8_t>& start, double time_limit,
                          int threads = 1, int max_seeds = -1, int stall_rounds = 50);

}  // namespace opt
