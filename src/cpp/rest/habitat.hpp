// SPDX-License-Identifier: GPL-3.0-or-later
//
// Inputs of restoration planning, after plangea (IIS; GPL >= 3): the original vegetation of the
// converted land (harmonize_oa) and the area of habitat of species (harmonize_bd; Brooks et al.
// 2019). R's arithmetic is kept (sequential double sums of the classes, R's LDOUBLE sum(), mean()
// and rowSums()), so the results are plangea's bits.
#pragma once

#include <cstdint>
#include <vector>

namespace rest {

using i64 = std::int64_t;

struct Vegetation {
    std::vector<double> original;  // k x n: the share of every natural class in the original cover
    std::vector<double> restore;   // k x n: the shares of the natural classes restored land returns to
    std::vector<std::uint8_t> source;  // n: 0 past cover, 1 the ecoregion's, 2 the failsafe shares
    std::vector<double> fill;      // k: the failsafe shares (0 for the classes not filled)
};

// The original vegetation of n pixels with k natural classes (current shares `natural`, k x n), m
// anthropic ones (`anthropic`, m x n), the past shares of the natural classes (`past`, k x n, or
// null: the current ones) and the ecoregion's shares (`ecoregion`, k x n, or null: equal shares).
// `domain` (n, or null: every pixel) marks the pixels whose shares make the failsafe means.
// `fill` (k): the classes the failsafe fills. `restore_current`: restored land returns to the
// current natural cover where there is some (plangea's restoration with conversion), else to the
// original one.
Vegetation original_vegetation(const double* natural, int k, const double* anthropic, int m, i64 n,
                               const double* past, const double* ecoregion, const std::uint8_t* domain,
                               const std::uint8_t* fill, double threshold, bool restore_current, int threads);

// Current and potential area of habitat of every species: the sum, over the pixels of its range
// (range_idx[range_ptr[s] .. range_ptr[s + 1]), increasing), of the shares of its suitable classes
// (suitable, ns x k; the classes in their order) times the pixel's area.
void habitat_areas(const double* current, const double* original, const double* area, i64 n, int k,
                   const i64* range_ptr, const i64* range_idx, const std::uint8_t* suitable, i64 ns,
                   double* now, double* potential, int threads);

// The habitat a species gains per unit of a pixel's area restored: the pixel's area times the
// restored shares of its suitable classes, for the pixels of its range in `domain` (null: all).
// CSR rows by species: indptr (ns + 1), the pixels and the gains.
void habitat_gain(const double* restore, const double* area, i64 n, int k, const i64* range_ptr,
                  const i64* range_idx, const std::uint8_t* suitable, i64 ns, const std::uint8_t* domain,
                  std::vector<i64>& indptr, std::vector<i64>& index, std::vector<double>& gain, int threads);

}  // namespace rest
