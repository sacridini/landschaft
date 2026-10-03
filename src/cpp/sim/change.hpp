// SPDX-License-Identifier: GPL-3.0-or-later
//
// Cross-tabulating categorical maps: the classes of a map, the transitions
// between two maps and the number of changes along a series of maps. A cell
// is missing when its value is NaN or the map's no-data value.
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

// A map's values, of any of the numeric types numpy gives.
enum class DType { u8, i8, u16, i16, u32, i32, i64, f32, f64 };

struct MapView {
    const void* data = nullptr;
    DType type = DType::f64;
    std::int64_t n = 0;
    bool has_nodata = false;
    double nodata = 0.0;
};

// The distinct values of the cells that are not missing, sorted.
std::vector<double> map_classes(const MapView& map);

// Cells of each class of `a` (rows, `classes_a`) that hold each class of `b`
// (columns, `classes_b`), row-major; cells missing in either map, or whose
// value is not in the given classes, are left out.
std::vector<std::int64_t> crosstab(const MapView& a, const MapView& b, const std::vector<double>& classes_a,
                                   const std::vector<double>& classes_b);

// Per cell, how many consecutive maps of the series differ there (raster's
// overlay of x != y, summed); -1 where any map is missing.
std::vector<std::int32_t> change_count(const std::vector<MapView>& maps);

}  // namespace sim
