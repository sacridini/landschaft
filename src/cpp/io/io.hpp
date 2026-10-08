// SPDX-License-Identifier: GPL-3.0-or-later
//
// Raster values by zone and at points: the kernels of ls.zonal_stats and
// ls.extract.
#pragma once

#include <cstdint>
#include <vector>

namespace lsio {

using i64 = std::int64_t;

// The cells of every zone as runs of one row, [x0, x1) in row y, grouped by
// zone: zone z owns runs offsets[z] .. offsets[z + 1].
struct Runs {
    std::vector<i64> offsets{0};
    std::vector<std::int32_t> y, x0, x1;
    i64 zones() const { return static_cast<i64>(offsets.size()) - 1; }
};

// The cells of polygons on an nx x ny grid (GDAL geotransform gt, no
// rotation), as GDAL's rasterizer burns them (centre rule or all-touched).
// Geometry g has the rings ring_offsets[geom_rings[g]] .. ring_offsets[geom_rings[g + 1]]
// (even-odd over all of them), ring r the points ring_offsets[r] .. ring_offsets[r + 1]
// of xy (x, y pairs). Polygons may overlap.
Runs polygon_runs(const double* xy, const i64* ring_offsets, const i64* geom_rings, i64 ngeoms, const double* gt,
                  int nx, int ny, bool all_touched);

// The cells of every distinct label of a raster of zones (NaN and nodata
// left out); labels: the zones' labels, sorted.
template <class L>
Runs label_runs(const L* labels, int ny, int nx, bool has_nodata, double nodata, std::vector<double>* zone_labels);

// Statistics.
enum Stat : int {
    COUNT = 0,     // cells with data
    NODATA = 1,    // cells without data
    SUM = 2,
    MEAN = 3,
    MIN = 4,
    MAX = 5,
    RANGE = 6,
    STD = 7,       // population standard deviation
    VAR = 8,       // population variance
    MEDIAN = 9,
    MAJORITY = 10, // the most frequent value (ties: the smallest)
    MINORITY = 11, // the least frequent value (ties: the smallest)
    UNIQUE = 12,   // the number of distinct values
    PERCENTILE = 13,
};

// stats[k] for every zone and band: (zones, bands, stats) values, row-major;
// percentiles[k] is the percentile (0-100, linear interpolation) of a
// PERCENTILE stat. Values: (bands, ny, nx), row-major; NaN and nodata are
// no data. Zones without data: NaN (count 0).
template <class T>
std::vector<double> zonal_stats(const T* values, int bands, int ny, int nx, bool has_nodata, double nodata,
                                const Runs& runs, const std::vector<int>& stats,
                                const std::vector<double>& percentiles);

// The values of every band at points (x, y in map units): (n, bands)
// row-major; NaN outside the grid or on no data. Bilinear: the four cell
// centres around the point, weights renormalised over those with data.
// values holds the rows [row0, row0 + ny) of a grid of ny_total rows (gt is the
// grid's): the points must fall in rows the block holds (with one row around
// for bilinear); positions are computed on the whole grid, so a block gives
// the same values as the whole raster.
template <class T>
std::vector<double> extract(const T* values, int bands, int ny, int nx, bool has_nodata, double nodata,
                            const double* gt, const double* x, const double* y, i64 n, bool bilinear, int row0,
                            int ny_total);

// Burns value[z] into the cells of zone z of out (ny x nx), zone after zone (the
// later zones win where they overlap), or adds them up (add).
template <class T>
void burn_runs(T* out, int ny, int nx, const Runs& runs, const double* value, bool add);

// The mean of every f x f block of a raster (the cells with data; NaN for a block
// without), as float32: (ny / f) x (nx / f), the incomplete blocks left out.
template <class T>
std::vector<float> block_mean(const T* values, int ny, int nx, int f, bool has_nodata, double nodata);

// Polygons cut into triangles (earcut, for WebGL): polygon p has the rings
// polygon_rings[p] .. polygon_rings[p + 1], ring r the points ring_offsets[r] .. ring_offsets[r + 1]
// of xy. The triangles' vertex indices, three per triangle, counted from the first point of
// their polygon, polygon after polygon; counts: the indices of each polygon.
// Where every pixel of a W x H view comes from in an h x w grid (north up: gt = x0, dx, y0, dy):
// the source coordinates X, Y of the pixels' centres known on a coarse grid (columns ci, rows cj,
// ncj x nci values), linear between; out: the cell's index (row * w + col), -1 outside the grid.
void view_index(const double* X, const double* Y, const i64* ci, i64 nci, const i64* cj, i64 ncj, i64 W, i64 H,
                const double* gt, i64 w, i64 h, std::int32_t* out);

// out[p] = src[index[p]] (channels bytes a pixel), 0 where the index is -1.
void gather_pixels(const std::uint8_t* src, i64 channels, const std::int32_t* index, i64 n, std::uint8_t* out);

std::vector<std::uint32_t> triangulate(const double* xy, const i64* ring_offsets, const i64* polygon_rings,
                                       i64 npolygons, std::vector<i64>* counts);

// The rules of ls.reclassify, resolved: a cell is `missing` where it is NaN or
// nodata, else the value of the interval [breaks[i], breaks[i + 1]) it falls in
// when break_set[i] (the ranges), else key_out[k] where it equals keys[k]
// (sorted), else its own value (keep) or `other`. Values are compared as doubles.
template <class Out>
struct ReclassRules {
    bool has_nodata = false;
    double nodata = 0.0;
    const double* keys = nullptr;
    const Out* key_out = nullptr;
    i64 n_keys = 0;
    const double* breaks = nullptr;  // n_breaks breaks, n_breaks - 1 intervals
    const Out* break_out = nullptr;
    const std::uint8_t* break_set = nullptr;
    i64 n_breaks = 0;
    bool keep = true;
    Out other{};
    Out missing{};
};

// out[i] = the new value of in[i], for n cells.
template <class T, class Out>
void reclassify(const T* in, Out* out, i64 n, const ReclassRules<Out>& rules);

}  // namespace lsio
