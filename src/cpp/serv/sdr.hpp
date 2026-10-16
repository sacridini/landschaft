// SPDX-License-Identifier: GPL-3.0-or-later
//
// The non-algebraic parts of InVEST's sediment delivery ratio model.
#pragma once

#include <cstdint>

namespace serv {

using i64 = std::int64_t;

// pygeoprocessing's calculate_slope: percent slope from the 3 x 3 window (Zevenbergen &
// Thorne with the diagonals, missing neighbours replaced as pygeoprocessing does), the
// cell sizes signed as in the geotransform (ycell < 0 north up); float32, NaN where the
// elevation is missing (NaN).
void pgp_slope(const double* z, int ny, int nx, double xcell, double ycell, float* out, int threads);

// InVEST's calculate_sediment_deposition: the sediment deposited on each cell (t) and its
// flux (f), from E' and the SDR (NaN missing), down pygeoprocessing's MFD (packed, 0
// missing) or D8 (0..7, 128 missing) directions; the seeds taken block by block. NaN
// where a cell is not reached.
void sediment_deposition_mfd(const std::int32_t* dir, const float* e_prime, const float* sdr, int ny, int nx, int block,
                             float* dep, float* flux);
void sediment_deposition_d8(const std::uint8_t* dir, const float* e_prime, const float* sdr, int ny, int nx, int block,
                            float* dep, float* flux);

}  // namespace serv
