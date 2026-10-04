// SPDX-License-Identifier: GPL-3.0-or-later
//
// Coverage-weighted sums of raster values over polygons, as exactextractr's
// exact_extract(x, polygons, fun = "sum") (exactextractr 0.10.0) computes
// them, bit for bit: for every feature, the grid cropped to the feature's
// box, in pieces of at most max_cells cells, the exact covered fraction of
// every cell (float, exactextract's algorithm: io/exact/coverage.hpp), then
// sum += value * fraction over the cells (row by row; NaN values skipped) in
// double, given as a float.
#pragma once

#include <cstdint>
#include <vector>

#include "io/exact/coverage.hpp"

namespace lsio {

struct ExactGrid {
    double xmin, ymin, xmax, ymax, dx, dy;
    std::int64_t rows, cols;
};

// values: layers x rows x cols (row-major, NaN = no data); features: each a list of polygons.
// Returns features x layers (0 where a feature misses the grid).
std::vector<double> exact_sums(const double* values, std::int64_t layers, const ExactGrid& grid,
                               const std::vector<std::vector<exactextract::Polygon>>& features,
                               std::int64_t max_cells, int threads);

// The covered fraction of every cell by one feature: (row offset, col offset, rows, cols, fractions).
struct ExactCoverage {
    std::int64_t row0 = 0, col0 = 0, rows = 0, cols = 0;
    std::vector<float> fraction;
};
ExactCoverage exact_coverage(const ExactGrid& grid, const std::vector<exactextract::Polygon>& feature);

}  // namespace lsio
