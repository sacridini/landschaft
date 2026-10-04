// SPDX-License-Identifier: GPL-3.0-or-later
//
// A port of Marxan 4.0.6 (Ball, Possingham & Watts; MIT licence, Copyright
// (c) 2020 Hugh Possingham, Ian Ball, Matthew Watts): simulated annealing,
// iterative improvement and the greedy heuristics on the Marxan objective
//
//     cost + BLM x boundary + sum over features of SPF x penalty x shortfall,
//
// the features' penalties computed as Marxan does (a greedy cover of each
// target). Same algorithms and floating-point steps, so the same runs, unit by
// unit, as the Marxan executable built with gcc: its std::mt19937 is the
// standard's, and the distributions are libstdc++'s (Lemire's integers,
// generate_canonical's reals, std::shuffle's two swaps per draw), written out
// here so that the results do not depend on the compiler's library.
//
// Supported: run modes 0, 1, 2 and 6 (penalties computed), annealing types 1
// (fixed schedule), 2 (Connolly) and 3 (adaptive), heuristics 0 (richness)
// and 1 (greedy), iterative improvement types 0 and 3, PROP initial
// reserves, cost thresholds, the features' proportion targets, symmetric
// boundaries ("connectivity edge" or "in"). Not: clumping or separation
// targets, probabilities, asymmetric connectivity, user penalties.
//
// One correction (implemented only in its correct form): Marxan's Connolly
// and adaptive initialisations score random reserves with the features'
// amounts of a different reserve (in the first run, memory readSpecies leaves
// uninitialised); here the amounts are those of the reserve scored.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace opt::marxan {

struct Data {
    int npu = 0, nsp = 0;
    std::vector<double> cost;   // per unit
    std::vector<int> status;    // 0 available, 1 initially in, 2 locked in, 3 locked out
    // features in every unit, units in order, features by index within a unit
    std::vector<std::int64_t> offset;  // npu + 1
    std::vector<int> spindex;
    std::vector<double> amount;
    std::vector<double> target, prop, spf;  // per feature (prop > 0: target = prop x total)
    std::vector<int> targetocc;
    // boundaries: every unit's external (fixed) boundary and its neighbours, in Marxan's order
    std::vector<double> fixedcost;          // npu
    std::vector<std::int64_t> nb_start;     // npu + 1
    std::vector<int> nb;
    std::vector<double> nb_cost;
};

struct Options {
    double blm = 0.0, prop = 0.0;
    long long iterations = 1000000;
    double tinit = -1.0;  // < 0: annealing type -tinit + 1 (-1 Connolly, -2 adaptive)
    double tcool = 0.0;
    long titns = 10000;
    long repeats = 10;
    double costthresh = 0.0, tpf1 = 0.0, tpf2 = 0.0;
    double misslevel = 1.0;
    int runmode = 1, heurtype = 1, itimptype = 0;
    int connectivity_in = 0;
    std::uint32_t seed = 0;
    int threads = 1;
    double start_dec_thresh = 0.7, end_dec_thresh = 0.3, start_dec_mult = 3.0, end_dec_mult = 1.0;
};

struct Score {
    double total = 0.0, cost = 0.0, connection = 0.0, penalty = 0.0, shortfall = 0.0;
    double threshpen = 0.0;
    int pus = 0, missing = 0;
};

struct Run {
    std::vector<int> R;  // 0/1 (and 2/3 for locked units)
    Score score;         // computed afresh for the final reserve (Marxan picks the best run by it)
    Score incremental;   // as updated change by change (the summary's figures)
    double connectivity = 0.0;  // the boundary length of the reserve (BLM 1)
    double min_prop_met = 1.0;  // MPM
    int missing = 0;            // features under misslevel x target
    double shortfall = 0.0;
};

struct Result {
    std::vector<double> penalty;  // per feature
    std::vector<double> target;   // per feature, after the proportion targets
    int unmet = 0;                // features that cannot meet their target
    std::vector<Run> runs;
    int best = 0;                 // the run with the lowest score (the first on ties)
};

Result solve(const Data& data, const Options& options);

}  // namespace opt::marxan
