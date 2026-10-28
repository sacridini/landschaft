// SPDX-License-Identifier: GPL-3.0-or-later
//
// Species dispersal and metapopulations:
//
// - migclim(): MigClim 1.6's cellular automaton (Engler and Guisan 2009;
//   Engler et al. 2012) on the habitat suitability of successive periods,
//   drawing from Microsoft's C runtime rand() as MigClim's Windows builds do,
//   so a seed gives MigClim's run;
// - metapopulation(): Hanski's incidence function model as R's MetaLandSim
//   (spom) iterates it, drawing from R's runif after set.seed.
#pragma once

#include <cstdint>
#include <vector>

namespace sdm {

using i64 = std::int64_t;

// MigClim's no data.
constexpr std::int32_t MIGCLIM_NO_DATA = -9999;

struct MigClimSettings {
    int steps = 1;                    // dispersal steps per period (dispSteps)
    std::vector<double> kernel;       // probability at 1, 2... cells (dispKernel)
    int ini_maturity = 1;             // iniMatAge
    int full_maturity = 1;            // fullMatAge
    std::vector<double> propagules;   // propaguleProd
    int threshold = 0;                // rcThreshold (0: none)
    double ldd_frequency = 0.0;       // lddFreq
    int ldd_min = 10, ldd_max = 15;   // lddMinDist, lddMaxDist
    bool weak = false;                // barrierType: weak (all five lines) or strong (two)
    std::uint32_t seed = 1;
};

struct MigClimRun {
    std::vector<std::int32_t> fin;     // the final state
    std::vector<std::int32_t> steps;   // the state after every step (n_steps x ny x nx)
    std::vector<std::int32_t> ids;     // every step's id (period x 100 + step)
    // per step: envChgStep, dispStep, stepID, univDispersal, noDispersal,
    // occupied, absent, stepColonized, stepDecolonized, stepLDDsuccess
    std::vector<std::int32_t> stats;
};

// mcMigrate: an initial distribution (1 present, 0 absent, no data) over
// periods of suitability (0 to 1000; n_periods x ny x nx), barriers (1) or
// null.
MigClimRun migclim(i64 ny, i64 nx, const std::int32_t* initial, const std::int32_t* suitability, i64 n_periods,
                   const std::int32_t* barriers, const MigClimSettings& s);

struct IfmModel {
    int kernel = 0;         // 0 exp(-alpha d), 1 1 / (1 + alpha d^beta)
    int connectivity = 0;   // 0 plain, 1 times the patch's area^c1
    int colonisation = 0;   // 0 S^2/(S^2+y^2), 1 1-exp(-y S), 2 S^z/(S^z+1/c2)
    int extinction = 0;     // 0 min(1, e/A^x), 1 1-exp(-e/A^x), 2 min(1, e/A^x (1-C)^r)
    double alpha = 0, x = 0, y = 0, e = 0, b = 1;
    double beta = 1, c1 = 0, c2 = 1, z = 1, r = 1;
};

// spom iterated `steps` times from `occupied`: the occupancy after every step
// (steps x n, 0 or 1).
std::vector<std::uint8_t> metapopulation(const double* x, const double* y, const double* area,
                                         const std::uint8_t* occupied, i64 n, i64 steps, const IfmModel& m,
                                         std::int32_t seed);

}  // namespace sdm
