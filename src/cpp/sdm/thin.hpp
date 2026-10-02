// SPDX-License-Identifier: GPL-3.0-or-later
//
// Spatial thinning of occurrence records: spThin's randomised heuristic
// (Aiello-Lammens et al. 2015), the same records kept as spThin::thin for the
// same set.seed().
#pragma once

#include <cstdint>
#include <vector>

namespace sdm {

using i64 = std::int64_t;

// Neighbours closer than `max_km` (fields::rdist.earth, R = 6378.388 km, the
// same bits as R on Windows): for each point the sorted indices of its
// neighbours, in CSR form. Points with a missing coordinate have none.
struct Neighbours {
    std::vector<i64> start;  // n + 1
    std::vector<std::int32_t> index;
};

Neighbours earth_neighbours(const double* lon, const double* lat, i64 n, double max_km);

// spThin::thin.algorithm(): `reps` repetitions on one R stream seeded with
// set.seed(seed). keep[r * n + i] = 1 when repetition r keeps point i.
struct ThinResult {
    std::vector<std::uint8_t> keep;
    std::vector<i64> kept;  // records kept by each repetition
};

ThinResult thin_spthin(const double* lon, const double* lat, i64 n, double max_km, int reps, std::int32_t seed);

}  // namespace sdm
