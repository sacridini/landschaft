// SPDX-License-Identifier: GPL-3.0-or-later
//
// Crop pollination (InVEST's pollination model, natcap.invest 3.20): the
// exponential decay kernel of pygeoprocessing, its block-wise FFT convolution
// (numpy's pocketfft, the same transforms) and the model's cell by cell
// expressions, in the float32 / float64 types InVEST computes them in.
#pragma once

#include <cstdint>
#include <vector>

#include "serv/services.hpp"

namespace serv {

// pygeoprocessing.kernels.exponential_decay_kernel(max_distance, expected_distance =
// alpha), normalized: a float32 square of side 2 floor(max_distance) + 1, exp(-d / alpha)
// out to max_distance (pixels), divided by its float32 sum (numpy's, tile by tile of 256).
std::vector<float> exponential_kernel(double alpha, double max_distance, i64& size);

// pygeoprocessing.convolve_2d(signal, kernel, ignore_nodata_and_edges=True,
// mask_nodata=True, normalize_kernel=False): the kernel over the signal (ny x nx,
// float32, NaN missing), in 256 x 256 blocks of the signal and of the kernel, each
// pair by numpy's rfftn / irfftn, added up in float32 as pygeoprocessing writes them;
// then divided by the same convolution of the valid cells and multiplied by the
// kernel's sum. Values of a block's convolution within 1e-8 of 0 are set to 0 (what
// pygeoprocessing documents; its code zeroes those within 1e-8 of 1e-8). ``out``
// (float64): NaN where the signal is missing.
void convolve_2d(const float* signal, i64 ny, i64 nx, const float* kernel, i64 ksize, double* out, int threads);

// The species' local maps: habitat nesting HN = max_n (N_n ns_n) (float64 products,
// stored float32), foraged flowers FF_j = RA_j fa_j (float32) of every season and the
// local foraging effectiveness FE = sum_j FF_j (float32). ``nesting``: nsub maps,
// ``floral``: nseason maps, NaN missing; ``ff``: nseason outputs (or null).
void pollination_local(const float* const* nesting, const double* ns, int nsub, const float* const* floral,
                       const double* fa, int nseason, i64 n, float* hn, float* const* ff, float* fe, int threads);

// Pollinator supply PS = sa HN FR (float64, stored float32).
void pollination_supply(const float* hn, const double* fr, double sa, i64 n, float* ps, int threads);

// Pollinator abundance PA = FF / FR * conv(PS) (0 where FR is 0; float64, stored
// float32), added in float32 to ``total`` (NaN where no species has a value yet).
void pollination_abundance(const float* ff, const double* fr, const double* cps, i64 n, float* pa, float* total,
                           int threads);

// On-farm pollinator abundance of a season, (pat (1 - h)) / (h (1 - 2 pat) + pat) in
// float32 (h: the half saturation of the farms of that season), added in float32
// to ``total`` (NaN where no season has a value yet); ``fp`` may be null.
void pollination_farm(const float* half_sat, const float* abundance, i64 n, float* fp, float* total, int threads);

// Total and wild pollinator yield: min(mp + FP, 1) and max(PYT - mp, 0) (float32).
void pollination_yield(const float* managed, const float* farm, i64 n, float* pyt, float* pyw, int threads);

}  // namespace serv
