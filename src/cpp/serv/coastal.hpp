// SPDX-License-Identifier: GPL-3.0-or-later
//
// Coastal vulnerability (InVEST's coastal_vulnerability model, natcap.invest 3.20):
// the fetch rays cast from the shore points against the landmass, the bathymetry
// sampled along them and the raster values aggregated within a radius of the
// points, as InVEST computes them (GEOS's segment intersection, GEOS's
// interpolation along a line, numpy's float32 / float64 means).
#pragma once

#include <cstdint>

#include "serv/services.hpp"

namespace serv {

// The fetch rays: from each of the n points (x, y), ndir rays start at
// p + delta_k (delta: ndir unit vectors (dx, dy), offset 1 m) and run max_fetch
// further; a ray stops at the nearest intersection with the m segments (x0, y0,
// x1, y1) of the landmass' coastline, computed by GEOS's LineIntersector (double-
// double arithmetic). Rays whose origin is on land (on_land, n x ndir) have length
// 0 and end NaN. ``fdist`` (n x ndir): the length sqrt(dx dx + dy dy) of the ray as
// OGR measures it; ``ends`` (n x ndir x 2): its end.
void coastal_rays(const double* points, i64 n, const std::uint8_t* on_land, const double* deltas, int ndir,
                  double max_fetch, const double* segments, i64 m, double* fdist, double* ends, int threads);

// InVEST's extract_bathymetry_along_ray for r rays (starts and ends, r x 2) over a
// raster (ny x nx, geotransform gt): ceil(length / gt[1]) + 1 points along the ray
// (GEOS's interpolation), the cell of each (truncated), its value kept unless NaN or
// numpy.isclose to ``nodata`` (when has_nodata), and their numpy mean (float32 sums
// for float rasters); with none, the mean of the valid cells of the smallest square
// around the cell of the end (cut at the raster's edges) that holds one (exact NoData
// test there, as pygeoprocessing's array_equals_nodata). Returns -1, or the first ray
// with a point off the raster (-2 - ray: no valid cell in the whole raster).
template <class T>
i64 coastal_ray_depth(const double* starts, const double* ends, i64 r, const T* raster, i64 ny, i64 nx,
                      const double* gt, bool has_nodata, double nodata, double* depth, int threads);

// InVEST's _aggregate_raster_values_in_radius: for each point, the window of the
// kernel (ksize x ksize, centred on the cell (truncated) of the point, cut at the
// raster's edges as InVEST cuts it), the cells of the kernel that hold a value (not
// NaN, not NoData: numpy.isclose for float rasters, == for integer ones); their sum
// (numpy's pairwise sum, in T) and count, and whether any is not 0. count = -1 where
// the window is off the raster.
template <class T>
void coastal_aggregate(const double* points, i64 n, const T* raster, i64 ny, i64 nx, const double* gt,
                       bool has_nodata, double nodata, bool exact_nodata, const std::uint8_t* kernel, i64 ksize,
                       i64 pixel_dist, double* sum, i64* count, std::uint8_t* nonzero, int threads);

}  // namespace serv
