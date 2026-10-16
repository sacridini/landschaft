// SPDX-License-Identifier: GPL-3.0-or-later
//
// Ecosystem services of InVEST (natcap.invest, Apache-2.0), reimplemented: carbon
// storage and sequestration, habitat quality. The arithmetic follows InVEST's own
// (float32 rasters, the order of its numpy expressions), so that the maps are the
// same bits where the maths permits.
#pragma once

#include <cstdint>
#include <vector>

namespace serv {

using i64 = std::int64_t;

// Index of the class of every cell: -1 where the cell is missing (valid[i] == 0),
// -2 where its code is not in ``codes`` (sorted, unique). Returns the first cell
// with an unknown code, or -1.
i64 classify(const i64* lulc, const std::uint8_t* valid, i64 n, const i64* codes, int k, std::int32_t* out,
             int threads);

// Carbon: the four pools (k x 4 densities, float32, by class index) mapped to the
// cells, and their float32 sum ((above + below) + soil) + dead. NaN where missing.
// ``out`` holds 5 arrays of n floats: above, below, soil, dead, total.
void carbon_pools(const std::int32_t* cls, i64 n, const float* pools, float* const out[5], int threads);

// Sum of the non-missing values in float64, in a fixed order (independent of the
// thread count).
double total(const float* x, i64 n, int threads);

// The Euclidean distance transform of pygeoprocessing (Meijster et al. 2002, in
// float32, as distance_transform_edt computes it): the distance in pixels of every
// cell to the nearest region cell (region[i] != 0). Without any region cell every
// distance is +infinity (pygeoprocessing's numerical stand-in for infinity there,
// about the half perimeter, is not kept).
void distance_transform(const std::uint8_t* region, int ny, int nx, float* dist, int threads);

// One threat's degradation: decay of the distance (linear or exponential, InVEST's
// float32 expressions with ``max_px`` the maximum distance in pixels), times the
// sensitivity of each class to the threat, the normalized weight and the access,
// rounded to float32 as InVEST does. Added (in float32) to ``total``. ``given``: the
// decay already computed (numpy's float32 exp, for the exponential decay), else null.
void threat_degradation(const float* dist, const float* given, const std::int32_t* cls, const float* sens,
                        const float* access, double max_px, bool exponential, double weight, i64 n, float* deg,
                        float* total, int threads);

// Habitat quality from the total degradation and the habitat of each class:
// h * (1 - d^z / (d^z + k^z)) in float32 (z = 2.5).
void quality(const float* total, const std::int32_t* cls, const float* habitat, double ksq, i64 n, float* out,
             int threads);

// Annual water yield (Budyko curve of Fu and Zhang, InVEST's fractp_op): the actual
// evapotranspiration as a fraction of the precipitation, in T (InVEST's numpy type:
// float32 unless an input is float64), stored as float32. kc, root and veg by class
// index (float32); eto, precip, soil (depth to the root restricting layer) and pawc
// by cell, NaN where missing. Missing also where precip is 0. NaN out where missing.
template <typename T>
void water_fraction(const std::int32_t* cls, const float* kc, const float* root, const float* veg, const T* eto,
                    const T* precip, const T* soil, const T* pawc, double z, i64 n, float* fractp, int threads);

// Water yield (1 - fractp) * precip and actual evapotranspiration fractp * precip, in U
// (float32 unless precip is float64), NaN where fractp or precip is missing.
template <typename U>
void water_yield(const float* fractp, const U* precip, i64 n, U* wyield, U* aet, int threads);

// Potential evapotranspiration eto * kc, in V (float32 unless eto is float64).
template <typename V>
void potential_et(const std::int32_t* cls, const float* kc, const V* eto, i64 n, V* pet, int threads);

}  // namespace serv
