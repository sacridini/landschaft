// SPDX-License-Identifier: GPL-3.0-or-later
//
// FUTURES (Meentemeyer et al. 2013; r.futures.simulation of the GRASS GIS
// addons, GPL >= 2, by Anna Petrasova and Vaclav Petras): urban growth by
// patches grown from seeds drawn on a development probability, as the GRASS
// module runs it, with GRASS's portable drand48. The arithmetic follows the
// module: probabilities, pressure and predictors in floats, candidates'
// rankings in doubles.
#pragma once

#include <climits>
#include <cstdint>
#include <vector>

namespace sim {

constexpr int kFuturesNull = INT_MIN;  // GRASS's CELL null

struct FuturesSettings {
    std::int64_t rows = 0, cols = 0;
    int nregions = 0;          // subregions (indices 0..n-1, in order of first appearance)
    int npot = 0;              // potential subregions
    // potential: per potential subregion
    std::vector<double> intercept, devpressure_coef;
    // development pressure
    int neighborhood = 0;      // n_dev_neighbourhood
    int approach = 1;          // 0 occurrence, 1 gravity, 2 kernel
    float gamma = 1.5f, scaling = 1.0f;
    // patches
    std::vector<std::vector<int>> patch_sizes;  // per region (or one list)
    bool single_column = true;
    int max_patch_size = 1;
    float compactness_mean = 0.4f, compactness_range = 0.05f;
    int num_neighbors = 4;
    bool seed_probability = true;  // seed search: probability (true) or random
    // incentive (empty: exponent 1)
    std::vector<float> incentive;
    // zoning: effects (id, region index or -1 for all, effect); stringency per region
    bool use_zone = false, user_effects = false;
    std::vector<int> zone_ids, zone_regions;
    std::vector<float> zone_effects, stringency;
    bool use_weight = false;
    // demand: nregions x nsteps cells (floats, rounded with lroundf)
    std::vector<float> demand;
    int nsteps = 0;
    std::int64_t seed = 0;
};

struct FuturesResult {
    std::vector<int> developed;          // -1 undeveloped, 0 at the start, k developed in step k; null
    std::vector<std::vector<int>> series;  // developed after each step
    std::vector<std::int64_t> converted; // nsteps x nregions cells developed in each region (inside it)
    std::vector<std::int64_t> overflow;  // nsteps x nregions cells carried to the next step
    std::vector<int> warnings;           // nsteps x nregions: bit 1 not enough cells, bit 2 too many failed tries
};

// developed: -1 / 0 / null (kFuturesNull) per cell, already null wherever an
// input is; region, pot_region: indices; devpressure: float (NaN null);
// aggregated: the sum of the predictor terms (floats, as read_predictors);
// weight, zone: optional per cell.
FuturesResult futures_run(const FuturesSettings& s, const int* developed, const int* region, const int* pot_region,
                          const float* devpressure, const float* aggregated, const float* weight, const int* zone);

// r.mfilter with a filter of r.futures.devpressure: the interior cells get
// sum(w * v) over the window in row order (double; any null in the window
// gives null), the cells within `size` of the edges keep their input value;
// the result is stored as float. matrix: (2 size + 1)^2 weights.
std::vector<float> futures_mfilter(const double* input, std::int64_t rows, std::int64_t cols, const double* matrix,
                                   int size);

}  // namespace sim
