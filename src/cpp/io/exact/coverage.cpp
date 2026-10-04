// SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-or-later
//
// Port of exactextract's raster_cell_intersection.cpp and floodfill.{h,cpp}
// (Copyright (c) 2018-2022 ISciences, LLC; Apache License 2.0, see LICENSE in
// this folder) to rings: same traversal, same rounding, same fill.

#include "coverage.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>

#include "cell.h"
#include "measures.h"

namespace exactextract {

Box ring_box(const Ring& ring) {
    double xmin = std::numeric_limits<double>::max(), ymin = std::numeric_limits<double>::max();
    double xmax = std::numeric_limits<double>::lowest(), ymax = std::numeric_limits<double>::lowest();
    for (const auto& c : ring) {
        xmin = std::min(xmin, c.x);
        ymin = std::min(ymin, c.y);
        xmax = std::max(xmax, c.x);
        ymax = std::max(ymax, c.y);
    }
    return {xmin, ymin, xmax, ymax};
}

namespace {

// GEOSCoordSeq_isCCW for a valid ring: the sign of its (shoelace) area
bool is_ccw(const Ring& ring) {
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < ring.size(); ++i) s += ring[i].x * ring[i + 1].y - ring[i + 1].x * ring[i].y;
    return s > 0;
}

// GEOSPreparedContains(polygon of the ring, point): strictly inside (on the ring: false)
bool ring_contains(const Ring& ring, double x, double y) {
    bool inside = false;
    for (std::size_t i = 0, j = ring.size() - 1; i < ring.size(); j = i++) {
        const Coordinate& a = ring[i];
        const Coordinate& b = ring[j];
        // on the segment
        const double cross = (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
        if (cross == 0 && std::min(a.x, b.x) <= x && x <= std::max(a.x, b.x) && std::min(a.y, b.y) <= y &&
            y <= std::max(a.y, b.y))
            return false;
        if ((a.y > y) != (b.y > y)) {
            const double xc = a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (x < xc) inside = !inside;
        }
    }
    return inside;
}

template <typename T>
struct fill_values;

template <>
struct fill_values<float> {
    static constexpr float EXTERIOR{0.0f};   // Cell is known to be entirely outside the polygon
    static constexpr float INTERIOR{1.0f};   // Cell is known to be entirely within the polygon
    static constexpr float FILLABLE{-1.0f};  // Cell location relative to polygon unknown, but can be determined by fill.
    static constexpr float UNKNOWN{-2.0f};   // Cell location relative to polygon unknown and cannot be filled
};

class FloodFill {
public:
    FloodFill(const Ring& ring, const Grid<bounded_extent>& extent) : m_extent{extent}, m_ring{ring} {}

    bool cell_is_inside(size_t i, size_t j) const {
        return ring_contains(m_ring, m_extent.x_for_col(j), m_extent.y_for_row(i));
    }

    template <typename T>
    void flood(Matrix<T>& arr) const {
        for (size_t i = 0; i < arr.rows(); i++) {
            for (size_t j = 0; j < arr.cols(); j++) {
                if (arr(i, j) == fill_values<T>::UNKNOWN) {
                    throw std::runtime_error("Cell with unknown position encountered.");
                } else if (arr(i, j) == fill_values<T>::FILLABLE) {
                    if (cell_is_inside(i, j)) {
                        flood_from_pixel(arr, i, j, fill_values<T>::INTERIOR);
                    } else {
                        flood_from_pixel(arr, i, j, fill_values<T>::EXTERIOR);
                    }
                }
            }
        }
    }

private:
    template <typename T>
    static void flood_from_pixel(Matrix<T>& arr, size_t i, size_t j, T fill_value) {
        std::queue<std::pair<size_t, size_t>> locations;
        locations.emplace(i, j);
        while (!locations.empty()) {
            i = locations.front().first;
            j = locations.front().second;
            locations.pop();
            if (arr(i, j) == fill_value) continue;
            if (j > 0 && arr(i, j - 1) == fill_values<T>::FILLABLE) locations.emplace(i, j - 1);
            auto j0 = j;
            for (; j < arr.cols() && arr(i, j) == fill_values<T>::FILLABLE; j++) arr(i, j) = fill_value;
            auto j1 = j;
            if (i > 0)
                for (j = j0; j < j1; j++)
                    if (arr(i - 1, j) == fill_values<T>::FILLABLE) locations.emplace(i - 1, j);
            if (i < arr.rows() - 1)
                for (j = j0; j < j1; j++)
                    if (arr(i + 1, j) == fill_values<T>::FILLABLE) locations.emplace(i + 1, j);
        }
    }

    Grid<bounded_extent> m_extent;
    const Ring& m_ring;
};

Cell* get_cell(Matrix<std::unique_ptr<Cell>>& cells, const Grid<infinite_extent>& ex, size_t row, size_t col) {
    if (cells(row, col) == nullptr) cells(row, col) = std::make_unique<Cell>(grid_cell(ex, row, col));
    return cells(row, col).get();
}

Box processing_region(const Box& raster_extent, const std::vector<Box>& component_boxes) {
    Box ret = Box::make_empty();
    for (const auto& box : component_boxes) {
        if (ret == raster_extent) return ret;  // no more expansion is possible
        if (!box.intersects(raster_extent)) continue;
        Box isect = raster_extent.intersection(box);
        if (ret.empty()) {
            ret = isect;
        } else if (!ret.contains(isect)) {
            ret = ret.expand_to_include(isect);
        }
    }
    return ret;
}

Grid<infinite_extent> get_box_grid(const Box& box, const Grid<infinite_extent>& geometry_grid) {
    Box cropped_ring_extent = geometry_grid.extent().intersection(box);
    return geometry_grid.shrink_to_fit(cropped_ring_extent);
}

class RasterCellIntersection {
public:
    RasterCellIntersection(const Grid<bounded_extent>& raster_grid, const std::vector<Polygon>& polygons)
        : m_geometry_grid{geometry_grid(raster_grid, polygons)},
          m_results{std::make_unique<Matrix<float>>(m_geometry_grid.rows() - 2, m_geometry_grid.cols() - 2)} {
        if (!m_geometry_grid.empty())
            for (const Polygon& p : polygons) {
                if (p.rings.empty()) continue;
                process_line(p.rings[0], true);
                for (std::size_t i = 1; i < p.rings.size(); ++i) process_line(p.rings[i], false);
            }
    }

    Grid<infinite_extent> m_geometry_grid;
    std::unique_ptr<Matrix<float>> m_results;

private:
    // GEOSGetNumGeometries: one box per polygon, the box of its coordinates (its exterior ring's)
    static Grid<infinite_extent> geometry_grid(const Grid<bounded_extent>& raster_grid,
                                               const std::vector<Polygon>& polygons) {
        std::vector<Box> boxes;
        bool any = false;
        for (const Polygon& p : polygons)
            if (!p.rings.empty() && !p.rings[0].empty()) {
                boxes.push_back(ring_box(p.rings[0]));
                any = true;
            }
        if (!any) throw std::invalid_argument("Can't get statistics for empty geometry");
        Box region = processing_region(raster_grid.extent(), boxes);
        if (!region.empty()) return make_infinite(raster_grid.shrink_to_fit(region));
        return Grid<infinite_extent>::make_empty();
    }

    void process_rectangular_ring(const Box& box, bool exterior_ring) {
        if (!box.intersects(m_geometry_grid.extent())) return;
        auto ring_grid = get_box_grid(box, m_geometry_grid);
        auto row_min = ring_grid.get_row(box.ymax);
        auto row_max = ring_grid.get_row(box.ymin);
        auto col_min = ring_grid.get_column(box.xmin);
        auto col_max = ring_grid.get_column(box.xmax);
        Matrix<float> areas(ring_grid.rows() - 2, ring_grid.cols() - 2);
        if (row_min > 0 && col_min > 0) {  // upper-left
            auto ul = grid_cell(ring_grid, row_min, col_min);
            areas(row_min - 1, col_min - 1) = static_cast<float>(ul.intersection(box).area() / ul.area());
        }
        if (row_min > 0 && col_max < ring_grid.cols() - 1) {  // upper-right
            auto ur = grid_cell(ring_grid, row_min, col_max);
            auto frac = ur.intersection(box).area() / ur.area();
            areas(row_min - 1, col_max - 1) = static_cast<float>(frac);
        }
        if (row_max < ring_grid.rows() - 1 && col_min > 0) {  // lower-left
            auto ll = grid_cell(ring_grid, row_max, col_min);
            areas(row_max - 1, col_min - 1) = static_cast<float>(ll.intersection(box).area() / ll.area());
        }
        if (row_max < ring_grid.rows() - 1 && col_max < ring_grid.cols() - 1) {  // lower-right
            auto lr = grid_cell(ring_grid, row_max, col_max);
            areas(row_max - 1, col_max - 1) = static_cast<float>(lr.intersection(box).area() / lr.area());
        }
        if (col_min > 0) {  // left
            auto left = grid_cell(ring_grid, row_min + 1, col_min);
            auto frac = left.intersection(box).area() / left.area();
            for (size_t row = row_min + 1; row < row_max; row++) areas(row - 1, col_min - 1) = static_cast<float>(frac);
        }
        if (col_max < ring_grid.cols() - 1) {  // right
            auto right = grid_cell(ring_grid, row_min + 1, col_max);
            auto frac = right.intersection(box).area() / right.area();
            for (size_t row = row_min + 1; row < row_max; row++) areas(row - 1, col_max - 1) = static_cast<float>(frac);
        }
        if (row_min > 0) {  // top
            auto top = grid_cell(ring_grid, row_min, col_min + 1);
            auto frac = top.intersection(box).area() / top.area();
            for (size_t col = col_min + 1; col < col_max; col++) areas(row_min - 1, col - 1) = static_cast<float>(frac);
        }
        if (row_max < ring_grid.rows() - 1) {  // bottom
            auto bottom = grid_cell(ring_grid, row_max, col_min + 1);
            auto frac = bottom.intersection(box).area() / bottom.area();
            for (size_t col = col_min + 1; col < col_max; col++) areas(row_max - 1, col - 1) = static_cast<float>(frac);
        }
        for (size_t row = row_min + 1; row < row_max; row++)  // interior
            for (size_t col = col_min + 1; col < col_max; col++) areas(row - 1, col - 1) = 1.0f;
        add_ring_results(ring_grid.row_offset(m_geometry_grid), ring_grid.col_offset(m_geometry_grid), areas,
                         exterior_ring);
    }

    void process_line(const Ring& ring, bool exterior_ring) {
        auto geom_box = ring_box(ring);
        if (!geom_box.intersects(m_geometry_grid.extent())) return;
        std::vector<Coordinate> coords = ring;
        if (coords.size() == 5 && area(coords) == geom_box.area()) {
            process_rectangular_ring(geom_box, exterior_ring);
            return;
        }
        Grid<infinite_extent> ring_grid = get_box_grid(geom_box, m_geometry_grid);
        size_t rows = ring_grid.rows();
        size_t cols = ring_grid.cols();
        // short circuit for small geometries entirely contained within a single grid cell
        if (rows == (1 + 2 * infinite_extent::padding) && cols == (1 + 2 * infinite_extent::padding) &&
            grid_cell(ring_grid, 1, 1).contains(geom_box)) {
            size_t i0 = ring_grid.row_offset(m_geometry_grid);
            size_t j0 = ring_grid.col_offset(m_geometry_grid);
            auto ring_area = area(coords) / grid_cell(ring_grid, 1, 1).area();
            m_results->increment(i0, j0, static_cast<float>(exterior_ring ? ring_area : -1 * ring_area));
            return;
        }
        Matrix<std::unique_ptr<Cell>> cells(rows, cols);
        if (!is_ccw(ring)) std::reverse(coords.begin(), coords.end());

        size_t pos = 0;
        size_t row = ring_grid.get_row(coords.front().y);
        size_t col = ring_grid.get_column(coords.front().x);
        const Coordinate* last_exit = nullptr;
        while (pos < coords.size()) {
            Cell& cell = *get_cell(cells, ring_grid, row, col);
            while (pos < coords.size()) {
                const Coordinate* next_coord = last_exit ? last_exit : &coords[pos];
                const Coordinate* prev_coord = pos > 0 ? &coords[pos - 1] : nullptr;
                cell.take(*next_coord, prev_coord);
                if (cell.last_traversal().exited()) {
                    // only push our exit coordinate if it's not the same as the coordinate we just took
                    const Coordinate& exc = cell.last_traversal().exit_coordinate();
                    if (exc != *next_coord) last_exit = &exc;
                    break;
                } else {
                    if (last_exit) {
                        last_exit = nullptr;
                    } else {
                        pos++;
                    }
                }
            }
            cell.force_exit();
            if (cell.last_traversal().exited()) {
                // started in the middle of a cell: reprocess the incomplete traversal's coordinates at the end
                if (!cell.last_traversal().traversed())
                    for (const auto& coord : cell.last_traversal().coords()) coords.push_back(coord);
                switch (cell.last_traversal().exit_side()) {
                    case Side::TOP: row--; break;
                    case Side::BOTTOM: row++; break;
                    case Side::LEFT: col--; break;
                    case Side::RIGHT: col++; break;
                    default: throw std::runtime_error("Invalid traversal");
                }
            }
        }

        Matrix<float> areas(rows - 2, cols - 2, fill_values<float>::FILLABLE);
        FloodFill ff(ring, make_finite(ring_grid));
        for (size_t i = 1; i <= areas.rows(); i++) {
            for (size_t j = 1; j <= areas.cols(); j++) {
                if (cells(i, j) != nullptr) {
                    // a processed cell with zero covered fraction may be inside the polygon: test it
                    auto frac = static_cast<float>(cells(i, j)->covered_fraction());
                    if (frac == 0) {
                        areas(i - 1, j - 1) =
                            ff.cell_is_inside(i - 1, j - 1) ? fill_values<float>::INTERIOR : fill_values<float>::EXTERIOR;
                    } else {
                        areas(i - 1, j - 1) = frac;
                    }
                }
            }
        }
        ff.flood(areas);
        add_ring_results(ring_grid.row_offset(m_geometry_grid), ring_grid.col_offset(m_geometry_grid), areas,
                         exterior_ring);
    }

    void add_ring_results(size_t i0, size_t j0, const Matrix<float>& areas, bool exterior_ring) {
        int factor = exterior_ring ? 1 : -1;
        for (size_t i = 0; i < areas.rows(); i++)
            for (size_t j = 0; j < areas.cols(); j++) m_results->increment(i0 + i, j0 + j, factor * areas(i, j));
    }
};

}  // namespace

Coverage raster_cell_intersection(const Grid<bounded_extent>& raster_grid, const std::vector<Polygon>& polygons) {
    RasterCellIntersection rci(raster_grid, polygons);
    return {std::move(*rci.m_results), make_finite(rci.m_geometry_grid)};
}

}  // namespace exactextract
