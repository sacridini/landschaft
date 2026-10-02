// SPDX-License-Identifier: GPL-3.0-or-later
//
// Polygons for "a circle of radius d metres around a point" exactly as the R
// packages that CoordinateCleaner 3.0.1 relies on build them (bit-identical
// vertices on x86-64 Windows, R 4.4 / Rtools 4.4):
//
//  * geosphere_ring: geosphere::destPoint(p, b = seq(0, 360, by = 5), d),
//    the 73-vertex polygon of cc_cap/cc_cen/cc_gbif/cc_inst/cc_aohi with
//    geod = TRUE (GeographicLib C++ 2.7, WGS84).
//  * geod_polys_fixed: the corrected form of that polygon (bug B10, the
//    `.geod_polys` of cc_fixed.R): circles containing a pole are closed
//    along the pole, circles crossing ±180 become two copies, one shifted
//    by 360 degrees, dissolved with GEOS as terra::aggregate does.
//  * terra_point_buffer: terra::buffer(vect(p, crs = "+proj=longlat
//    +datum=WGS84"), width = d) for one point (terra 1.9-50
//    SpatVector::point_buffer: 4 * quadsegs vertices from GeographicLib C
//    1.52, pole closure, dateline split with GEOS 3.12.2).
//  * geos_circle: the planar GEOS buffer of a point (the corrected cc_zero).
//
// Rings are returned as interleaved coordinates x0, y0, x1, y1, ... and are
// closed (the first vertex is repeated at the end). Functions returning
// several rings return one ring per polygon part (no holes occur), in the
// order of terra::geom()'s `part` column.
//
// The bits depend on the elementary functions: see geocircle_libm.hpp.
#pragma once

#include <vector>

namespace sdm {

// One closed ring, x/y (lon/lat) interleaved.
using GeoRing = std::vector<double>;

// The parts of a (multi)polygon, each one closed ring.
using GeoParts = std::vector<GeoRing>;

// --- geodesics as the R packages compute them (WGS84) ---------------------

// geosphere 1.6-8 (GeographicLib C++ 2.7): Geodesic(a, f).Direct and the
// distance of Inverse (geosphere::distGeo). Degrees and metres.
void geosphere_direct(double lon1, double lat1, double azi1, double s12, double& lon2, double& lat2);
double geosphere_distance(double lon1, double lat1, double lon2, double lat2);

// terra 1.9-50 (GeographicLib C 1.52): geod_direct and the distance of
// geod_inverse.
void terra_direct(double lon1, double lat1, double azi1, double s12, double& lon2, double& lat2);
double terra_distance(double lon1, double lat1, double lon2, double lat2);

// --- polygons ---------------------------------------------------------------

// The 73 vertices (lon, lat interleaved; 146 numbers) of
// geosphere::destPoint(c(lon, lat), b = seq(0, 360, by = 5), d) in bearing
// order 0, 5, ..., 360. Longitudes are normalised to [-180, 180] as
// destPoint returns them. Bearing 360 gives the same point as bearing 0, so
// the ring is closed.
std::vector<double> geosphere_ring(double lon, double lat, double d);

// cc_fixed.R `.geod_polys(cbind(lon, lat), d)`, the corrected geod = TRUE
// polygon, as terra::geom() returns it:
//  * if the point is less than d from a pole (geosphere::distGeo along its
//    meridian; the north pole is tested first): one ring, the 72 distinct
//    vertices sorted by longitude (stably) and closed through
//    (180, pole) and (-180, pole);
//  * else if the longitudes span more than 180 degrees (the circle crosses
//    ±180): the vertices unwrapped around lon and a copy shifted by -360
//    (lon > 0) or +360 (lon <= 0), dissolved by GEOS (two parts, ordered
//    and rotated as GEOS's union outputs them);
//  * else the destPoint ring itself.
GeoParts geod_polys_fixed(double lon, double lat, double d);

// terra::buffer() of the lon/lat point (lon, lat) with width d (metres) and
// quadsegs segments per quadrant, as terra::geom() returns it:
//  * 4 * quadsegs vertices at bearings i * (360 / (4 * quadsegs)) from
//    geod_direct, closed with the first vertex;
//  * if the point is less than d from one pole: the distinct vertices sorted
//    by (lon, lat) and closed through (±180, pole);
//  * if the longitudes span more than 180 degrees: split at ±180 into two
//    parts (GEOS intersection with [-1, 180] and [180, 361], the second part
//    shifted by -360), unless the shifted ring would be wider;
//  * d > 20003931: the whole globe.
// An empty result for NaN coordinates or |lat| > 90. Throws
// std::invalid_argument for d <= 0 (terra's error) and std::domain_error
// when the circle contains both poles (d of about 10000 km or more; terra
// then erases the circle from the globe with GEOS, which is not ported).
GeoParts terra_point_buffer(double lon, double lat, double d, unsigned quadsegs = 10);

// The GEOS buffer of the planar point (cx, cy) with radius r and quadsegs
// segments per quadrant (terra::buffer of a point without CRS, GEOS
// OffsetSegmentGenerator::createCircle): (cx + r, cy), then clockwise
// (cx + r cos(a_i), cy + r sin(a_i)) with a_i = -i * 2 pi / (4 quadsegs),
// i = 1 .. 4 quadsegs - 1, then closed. Vertices closer than r * 1e-6 to the
// previous one are dropped as GEOS drops them. Empty for r <= 0.
GeoRing geos_circle(double cx, double cy, double r, int quadsegs = 10);

}  // namespace sdm
