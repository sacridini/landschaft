// SPDX-License-Identifier: GPL-3.0-or-later
//
// The carbon of land change over a series of maps (the IPCC's stock-difference
// method), as Terraflux's carbon.emissions computes it: every cell's stock at every
// date from its class (a stock per class, or a curve of regrowth from the date the
// class appeared in the cell), and the emissions and removals between the dates.
#pragma once

#include <cstdint>

namespace serv {

using i64 = std::int64_t;

// Regrowth forms of a class.
enum : int { kNoRegrowth = 0, kChapmanRichards = 1, kLinear = 2 };

// The stocks of ``t`` maps of ``n`` cells (``cls``: a class index per cell and date,
// -1 missing; t x n), tC/ha, into ``out`` (t x n, NaN where missing). By class
// index: ``stock`` (NaN: none), ``form``, ``rmax``, ``rate``, ``shape`` of its
// regrowth. A class of regrowth grows from the date it appeared in the cell (the
// first date of the series or after a missing date: not grown, its stock, else its
// curve's max). Every class must have a stock or a regrowth (checked by the caller).
void emission_stocks(const std::int32_t* cls, int t, i64 n, const double* times, const double* stock,
                     const int* form, const double* rmax, const double* rate, const double* shape, double* out,
                     int threads);

// The flows between consecutive dates: ``net`` (n; the sum of every period's fall of
// stock times ``factor``, NaN for a cell never valid at two consecutive dates),
// ``periods`` ((t - 1) x 2: emissions, removals in t, the falls and rises times
// ``factor`` and ``ha``) and ``trans`` ((t - 1) x k x k x 3: cells, emissions,
// removals by pair of classes). The sums go by fixed blocks of cells, so they do
// not depend on the thread count.
void emission_flows(const double* stocks, const std::int32_t* cls, int t, i64 n, int k, double factor, double ha,
                    double* net, double* periods, double* trans, int threads);

}  // namespace serv
