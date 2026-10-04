// SPDX-License-Identifier: GPL-3.0-or-later

#include "io/exact.hpp"

#include <cmath>
#include <exception>
#include <stdexcept>

namespace lsio {

namespace {

using exactextract::Box;
using exactextract::bounded_extent;
using Grid = exactextract::Grid<bounded_extent>;

Grid grid_of(const ExactGrid& g) {
    Grid grid({g.xmin, g.ymin, g.xmax, g.ymax}, g.dx, g.dy);
    if (static_cast<std::int64_t>(grid.rows()) != g.rows || static_cast<std::int64_t>(grid.cols()) != g.cols)
        throw std::invalid_argument("the grid's extent and resolution do not give its rows and columns");
    return grid;
}

// GEOSGeom_getXMin...: the box of every coordinate of the feature (its exterior rings hold them)
Box feature_box(const std::vector<exactextract::Polygon>& feature) {
    Box b = Box::make_empty();
    bool first = true;
    for (const auto& p : feature)
        for (const auto& ring : p.rings) {
            if (ring.empty()) continue;
            const Box r = exactextract::ring_box(ring);
            b = first ? r : Box{std::min(b.xmin, r.xmin), std::min(b.ymin, r.ymin), std::max(b.xmax, r.xmax),
                                std::max(b.ymax, r.ymax)};
            first = false;
        }
    return b;
}

}  // namespace

std::vector<double> exact_sums(const double* values, std::int64_t layers, const ExactGrid& g,
                               const std::vector<std::vector<exactextract::Polygon>>& features,
                               std::int64_t max_cells, int threads) {
    const Grid grid = grid_of(g);
    const auto nf = static_cast<std::int64_t>(features.size());
    std::vector<double> out(static_cast<std::size_t>(nf * layers), 0.0);
    const std::int64_t plane = g.rows * g.cols;
    // exceptions must not leave the OpenMP region: the first one is thrown again afterwards
    std::exception_ptr error;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (std::int64_t f = 0; f < nf; ++f) try {
        const auto& feature = features[static_cast<std::size_t>(f)];
        std::vector<double> sum(static_cast<std::size_t>(layers), 0.0);
        const Box bbox = feature_box(feature);
        if (!feature.empty() && bbox.intersects(grid.extent())) {
            const Grid cropped = grid.crop(bbox);
            for (const auto& subgrid : exactextract::subdivide(cropped, static_cast<std::size_t>(max_cells))) {
                const exactextract::Coverage cov = exactextract::raster_cell_intersection(subgrid, feature);
                if (cov.grid.empty()) continue;
                // the values of the window (S4RasterSource::read_box: the grid cropped to the coverage's
                // extent), seen on the coverage's grid as RasterStats::process does through a RasterView
                // when the two grids differ (a polygon within one cell gives a coverage grid a few ulps
                // off the window's): cell (i, j) of the coverage is (i + y_off, j + x_off) of the window
                const Grid window = grid.crop(cov.grid.extent());
                if (window.empty()) continue;
                if (window.dx() != cov.grid.dx() || window.dy() != cov.grid.dy())
                    throw std::runtime_error("coverage grid at another resolution than the raster");
                std::int64_t x_off = 0, y_off = 0;
                if (!(window == cov.grid)) {
                    x_off = static_cast<std::int64_t>(std::round((cov.grid.xmin() - window.xmin()) / cov.grid.dx()));
                    y_off = static_cast<std::int64_t>(std::round((window.ymax() - cov.grid.ymax()) / cov.grid.dy()));
                }
                const auto i0 = static_cast<std::int64_t>(window.row_offset(grid));
                const auto j0 = static_cast<std::int64_t>(window.col_offset(grid));
                const auto wrows = static_cast<std::int64_t>(window.rows());
                const auto wcols = static_cast<std::int64_t>(window.cols());
                const auto rows = static_cast<std::int64_t>(cov.fraction.rows());
                const auto cols = static_cast<std::int64_t>(cov.fraction.cols());
                for (std::int64_t l = 0; l < layers; ++l) {
                    const double* v = values + l * plane;
                    double& s = sum[static_cast<std::size_t>(l)];
                    for (std::int64_t i = 0; i < rows; ++i) {
                        const std::int64_t wi = i + y_off;
                        if (wi < 0 || wi >= wrows) continue;
                        for (std::int64_t j = 0; j < cols; ++j) {
                            const std::int64_t wj = j + x_off;
                            if (wj < 0 || wj >= wcols) continue;
                            const float pct = cov.fraction(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
                            const double val = v[(i0 + wi) * g.cols + (j0 + wj)];
                            if (pct > 0 && !std::isnan(val)) s += val * static_cast<double>(pct);
                        }
                    }
                }
            }
        }
        for (std::int64_t l = 0; l < layers; ++l)
            out[static_cast<std::size_t>(f * layers + l)] = static_cast<float>(sum[static_cast<std::size_t>(l)]);
    } catch (...) {
#pragma omp critical(lsio_exact_error)
        if (!error) error = std::current_exception();
    }
    if (error) std::rethrow_exception(error);
    return out;
}

ExactCoverage exact_coverage(const ExactGrid& g, const std::vector<exactextract::Polygon>& feature) {
    const Grid grid = grid_of(g);
    ExactCoverage r;
    const Box bbox = feature_box(feature);
    if (feature.empty() || !bbox.intersects(grid.extent())) return r;
    const exactextract::Coverage cov = exactextract::raster_cell_intersection(grid.crop(bbox), feature);
    if (cov.grid.empty()) return r;
    r.row0 = static_cast<std::int64_t>(cov.grid.row_offset(grid));
    r.col0 = static_cast<std::int64_t>(cov.grid.col_offset(grid));
    r.rows = static_cast<std::int64_t>(cov.fraction.rows());
    r.cols = static_cast<std::int64_t>(cov.fraction.cols());
    r.fraction.resize(static_cast<std::size_t>(r.rows * r.cols));
    for (std::int64_t i = 0; i < r.rows; ++i)
        for (std::int64_t j = 0; j < r.cols; ++j)
            r.fraction[static_cast<std::size_t>(i * r.cols + j)] =
                cov.fraction(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
    return r;
}

}  // namespace lsio
