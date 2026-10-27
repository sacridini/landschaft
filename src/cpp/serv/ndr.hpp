// SPDX-License-Identifier: GPL-3.0-or-later
//
// The non-algebraic part of InVEST's nutrient delivery ratio model.
#pragma once

#include <cstdint>

namespace serv {

using i64 = std::int64_t;

// InVEST's calculate_retention (ndr/retention.h): the effective retention of each cell
// on its way down to a stream, from the maximum retention efficiency `eff` and the
// critical length `crit` (metres) of its land use (NaN missing), down pygeoprocessing's
// MFD (packed, 0 missing) or D8 (0..7, 128 missing) directions to the `stream` cells
// (1; 0 not, 255 missing). The drains are taken block by block as InVEST does, the
// values kept in float64 until they are written (float32, NaN where not reached).
void ndr_retention_mfd(const std::int32_t* dir, const std::uint8_t* stream, const float* eff, const float* crit,
                       int ny, int nx, double cell_size, int block, float* out);
void ndr_retention_d8(const std::uint8_t* dir, const std::uint8_t* stream, const float* eff, const float* crit, int ny,
                      int nx, double cell_size, int block, float* out);

}  // namespace serv
