// SPDX-License-Identifier: Apache-2.0 AND GPL-3.0-or-later
//
// exactextract's raster_cell_intersection (ISciences, Apache-2.0; the
// version bundled in exactextractr 0.10.0, exactextract 8f6753c) for
// polygons given as rings instead of GEOS geometries: the fraction of every
// raster cell a polygon covers, computed exactly from where its rings cross
// the cell edges. The algorithm and its floating-point steps are
// exactextract's; GEOS is replaced by the ring's bounding box, its
// orientation (shoelace sign) and a point-in-polygon test, which give the
// same answers for valid rings.
#pragma once

#include <vector>

#include "coordinate.h"
#include "grid.h"
#include "matrix.h"

namespace exactextract {

using Ring = std::vector<Coordinate>;

// One polygon: rings[0] is the exterior, the others its holes (closed rings: first == last).
struct Polygon {
    std::vector<Ring> rings;
};

struct Coverage {
    Matrix<float> fraction;        // the covered fraction of every cell of `grid`
    Grid<bounded_extent> grid;     // a window of the raster's grid (empty if the polygons miss it)
};

// The coverage of the raster grid's cells by a (multi)polygon, on the smallest
// window of the grid that holds the parts of the polygons within its extent.
Coverage raster_cell_intersection(const Grid<bounded_extent>& raster_grid, const std::vector<Polygon>& polygons);

// The bounding box of the coordinates of a ring.
Box ring_box(const Ring& ring);

}  // namespace exactextract
