// SPDX-License-Identifier: GPL-3.0-or-later
//
// The routing of InVEST's seasonal water yield model (natcap.invest 3.20, Apache-2.0,
// seasonal_water_yield/swy.h): the local recharge and the baseflow, down pygeoprocessing's
// MFD (eight 4-bit weights packed, 0 missing) or D8 (0..7, 128 missing) directions.
#pragma once

#include <cstdint>

namespace serv {

using i64 = std::int64_t;

// The twelve months of a cell's inputs, month-major (month m of cell i at m * n + i), NaN
// where missing (a missing precipitation or quickflow counts 0, a missing Kc or ET0 no
// potential evapotranspiration, as InVEST); the precipitation and ET0 in float32 or
// float64 (InVEST reads every raster as double).
template <class P, class E>
struct SwyMonths {
    const P* precip;
    const E* et0;
    const float* quickflow;
    const float* kc;
    const float* alpha;  // the twelve alpha_m
};

// InVEST's calculate_local_recharge (eqs. 100-104): L_sum_avail from upslope, the annual
// AET, P, L and L_avail of every cell with a flow direction, the seeds taken block by block
// (pygeoprocessing's 256 x 256 tiles). Outputs in double (InVEST keeps them so until its
// cache writes them, float32), NaN where not reached. L_sum_avail is eq. 103, the shares
// of the upslope flow summed (InVEST takes their mean weighted by the raw weights).
// Defined for P, E in {float, double}.
template <class P, class E>
void swy_local_recharge_mfd(const std::int32_t* dir, int ny, int nx, int block, const SwyMonths<P, E>& in,
                            float beta, float gamma, double* l, double* l_avail, double* l_sum_avail, double* aet,
                            double* p_annual);
template <class P, class E>
void swy_local_recharge_d8(const std::uint8_t* dir, int ny, int nx, int block, const SwyMonths<P, E>& in,
                           float beta, float gamma, double* l, double* l_avail, double* l_sum_avail, double* aet,
                           double* p_annual);

// InVEST's route_baseflow_sum (eqs. 107-110): B_sum and B from the outlets up, from L,
// L_avail (as InVEST reads them back, float32), L_sum (float64) and the streams (0, 1,
// 255 missing; a missing stream cell, like the flow off the raster, ends the path as a
// stream does). NaN where not reached.
void swy_baseflow_mfd(const std::int32_t* dir, int ny, int nx, int block, const double* l, const double* l_avail,
                      const double* l_sum, const std::uint8_t* stream, double* b, double* b_sum);
void swy_baseflow_d8(const std::uint8_t* dir, int ny, int nx, int block, const double* l, const double* l_avail,
                     const double* l_sum, const std::uint8_t* stream, double* b, double* b_sum);

}  // namespace serv
