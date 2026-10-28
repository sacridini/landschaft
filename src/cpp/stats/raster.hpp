// SPDX-License-Identifier: GPL-3.0-or-later
//
// Raster tools of ls.stats and ls.surface, as Terraflux's stack.reduce,
// raster.local_correlation and raster.viewshed: a statistic of a stack's
// layers at every cell (with the layer of the largest or smallest value),
// the correlation and least squares line of two maps in a moving window
// (terra's focalPairs / focalReg), and the observers seeing every cell of an
// elevation map (Franklin and Ray's R2 rays).
#pragma once

#include <cstdint>

namespace stats {

// The statistics of cell_stats: the codes of ls.zonal_stats, and 14 argmax,
// 15 argmin (the layer, 1.., of the largest / smallest value; a tie: the first).
enum CellStat : int {
    kCount = 0, kNodata = 1, kSum = 2, kMean = 3, kMin = 4, kMax = 5, kRange = 6, kStd = 7, kVar = 8,
    kMedian = 9, kMajority = 10, kMinority = 11, kUnique = 12, kPercentile = 13, kArgmax = 14, kArgmin = 15,
};

// layers: k x n (row-major); out: n. NaN is missing; a cell without a value
// in any layer gets NaN (0 for count and unique, k for nodata).
void cell_stats(const double* layers, std::int64_t k, std::int64_t n, int stat, double q, double* out, int threads);

// x, y: ny x nx; out: 4 x ny x nx (correlation, slope, intercept, r2). The
// window: the cells within `radius` rows and columns (circle: within the
// radius' circle) where both maps have a value, at least `min_cells`.
void local_correlation(const double* x, const double* y, int ny, int nx, int radius, bool circle, int min_cells,
                       double* out, int threads);

// dem: ny x nx (NaN missing); observers: their row and column; rw, rh: the
// cells' width and height in map units. out: how many observers see each
// cell (NaN where the dem has none).
void viewshed(const double* dem, int ny, int nx, double rw, double rh, const std::int64_t* rows,
              const std::int64_t* cols, std::int64_t n_obs, double eye, double target, double reach, bool curvature,
              double* out, int threads);

}  // namespace stats
