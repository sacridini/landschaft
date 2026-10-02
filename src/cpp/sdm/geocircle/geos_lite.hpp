// SPDX-License-Identifier: GPL-3.0-or-later
//
// A small, faithful port of the parts of GEOS 3.12.2 that terra 1.9-50 runs
// when it splits a lon/lat buffer at the dateline (polygon ∩ rectangle) and
// when cc_fixed.R dissolves the two copies of a dateline circle (union of
// two disjoint polygons). The aim is the bits GEOS produces, vertex order
// included: the ring starts where GEOS's OverlayNG graph walk starts it.
//
// Derived from GEOS (https://libgeos.org), LGPL-2.1-or-later, Copyright (C)
// 2001-2023 the GEOS contributors and Vivid Solutions (JTS); used under
// section 3 of the LGPL 2.1, which allows the code to be distributed under
// the GNU GPL (here GPL-3.0-or-later). Ported: math/DD (double-double),
// CGAlgorithmsDD::orientationIndex / intersection, LineIntersector (point
// intersections), Orientation::isCCW, Octant / SegmentPointComparator /
// SegmentNode / SegmentNodeList (noding), RingClipper, OverlayUtil's clipping
// envelope, RobustClipEnvelopeComputer, OverlayEdge::addCoordinates and the
// ring walk of PolygonBuilder / MaximalEdgeRing / OverlayEdgeRing.
//
// Scope: two single-ring polygons (shells, no holes) in floating precision,
// OverlayNG's first (floating noder) attempt. Edge labels are computed
// geometrically instead of by OverlayLabeller, which gives the same labels
// for these inputs. Configurations this port does not cover throw
// std::runtime_error instead of returning something GEOS would not:
// collinear (overlapping) segments, nodes with more than one result edge
// leaving them, result holes.
#pragma once

#include <vector>

namespace sdm {
namespace geos_lite {

struct Coord {
    double x;
    double y;
};

// A closed ring (first coordinate repeated at the end).
using Ring = std::vector<Coord>;

enum class OverlayOp { Intersection, Union };

// GEOS OverlayNG overlay of the polygons with shells `a` and `b`
// (GEOSIntersection_r / GEOSUnion_r with floating precision). Returns the
// shells of the result polygons in GEOS's output order.
std::vector<Ring> overlay(const Ring& a, const Ring& b, OverlayOp op);

// GEOS UnaryUnion of a multipolygon with the shells `polys` (as
// terra::aggregate(dissolve = TRUE) runs it): CascadedPolygonUnion orders
// the polygons with an STR tree, then unions them pairwise. Only one or two
// polygons are supported.
std::vector<Ring> unary_union(const std::vector<Ring>& polys);

// GEOS Orientation::isCCW of a closed ring.
bool is_ccw(const Ring& ring);

}  // namespace geos_lite
}  // namespace sdm
