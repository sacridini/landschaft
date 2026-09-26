// SPDX-License-Identifier: GPL-3.0-or-later
//
// Habitat-graph nodes from a raster of patch ids, and Euclidean distances
// between nodes, as Makurhini's nodesfile / distancefile compute them:
// centroids are the means of the raster package's cell centres (R's
// colMeans, summed in 80-bit precision), edge distances the gap between the
// closest cells of two patches (the distance GEOS measures between the
// polygons of terra::as.polygons).
#pragma once

#include <cstdint>
#include <limits>
#include <vector>

namespace conn {

using i64 = std::int64_t;

struct RasterGrid {
    i64 nx = 0, ny = 0;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;  // extent as terra reads it
};

struct RasterNodes {
    std::vector<std::int32_t> ids;  // the distinct patch ids, increasing
    std::vector<i64> cells;         // cells per node
    std::vector<double> cx, cy;     // centroids (when asked)
    std::vector<std::int32_t> node; // node index of every cell, -1 outside the nodes (when asked)
};

// `na` may be null. Cells with a value and na == 0 belong to the node of
// their value.
RasterNodes raster_nodes(const std::int32_t* values, const std::uint8_t* na, const RasterGrid& grid, bool centroids,
                         bool keep_node_index);

struct PairList {
    std::vector<std::int32_t> i, j;  // i < j, sorted by (i, j)
    std::vector<double> distance;
};

constexpr double kNoLimit = std::numeric_limits<double>::infinity();

// Distances between points (GEOS: sqrt(dx*dx + dy*dy)), pairs up to max_distance.
PairList point_distances(const double* x, const double* y, std::int32_t n, double max_distance);

// Edge-to-edge distances between the patches of a node index raster (cells
// of size resx × resy): the smallest gap between two of their cells, 0 when
// they touch. Pairs up to max_distance.
PairList raster_edge_distances(const std::int32_t* node, std::int32_t n_nodes, i64 nx, i64 ny, double resx, double resy,
                               double max_distance);

// Areas of polygons as GDAL computes them (OGR get_Area, which sf's st_area
// calls): each ring by the shoelace sum x_i (y_{i+1} - y_{i-1}) accumulated in
// order, a polygon as its exterior minus its holes, a multipolygon as the
// sum of its parts. Rings are closed coordinate runs [ring_start[r],
// ring_start[r + 1]); polygon p owns rings [poly_start[p], poly_start[p + 1])
// (exterior first), geometry g polygons [geom_start[g], geom_start[g + 1]).
std::vector<double> gdal_areas(const double* x, const double* y, const i64* ring_start, const i64* poly_start,
                               const i64* geom_start, i64 n_geoms);

}  // namespace conn
