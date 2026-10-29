// SPDX-License-Identifier: GPL-3.0-or-later
//
// Distances and directions between consecutive locations as move2 computes
// them (R, Windows build), the same bits:
//
// - lon / lat distance: sf::st_distance with s2 on, i.e. s2::s2_distance:
//   each point is S2's PlateCarreeProjection(180).Unproject (lng wrapped by
//   remainder(lng, 360), both scaled by M_PI / 180, S2LatLng::ToPoint), the
//   chord angle min(4, |a - b|^2) and 2 * asin(sqrt(len2) / 2), times
//   s2_earth_radius_meters() = 6371010 m. sin and cos are mingw-w64's (x87),
//   asin the UCRT's (std::asin on MSVC), sqrt correctly rounded.
// - projected / no-CRS distance: sf's sqrt((x1 - x2)^2 + (y1 - y2)^2), R's
//   vector sqrt (x87 fsqrt, lsm::r_sqrt).
// - lon / lat azimuth: lwgeom::st_geod_azimuth, liblwgeom's
//   lwgeom_azumith_spheroid: the degrees through radians and back
//   (deg2rad, the normalisations, * 180 / M_PI), PROJ's geod_inverse (Spheroid)
//   and azi1 * M_PI / 180; NaN for points within 1e-14 degrees (lwgeodetic.h's FP_TOLERANCE).
// - no-CRS azimuth: R's atan2(dx, dy) (mingw-w64's x87 fpatan).
#pragma once

namespace move {

constexpr double kS2EarthRadius = 6371010.0;  // s2::s2_earth_radius_meters()

double s2_distance(double lon1, double lat1, double lon2, double lat2);
double planar_distance(double x1, double y1, double x2, double y2);

// PROJ 9.4.1's GeographicLib C 2.1 (geodesic_proj.cpp) on a spheroid given as
// sf passes it to lwgeom: semi-major axis and inverse flattening.
class Spheroid {
public:
    Spheroid(double a, double inv_flattening);
    ~Spheroid();
    Spheroid(const Spheroid&) = delete;
    Spheroid& operator=(const Spheroid&) = delete;
    // geod_inverse's azi1 (degrees)
    double azimuth_deg(double lat1, double lon1, double lat2, double lon2) const;

private:
    void* impl_ = nullptr;
};

double lwgeom_azimuth(const Spheroid& s, double lon1, double lat1, double lon2, double lat2);
double planar_azimuth(double x1, double y1, double x2, double y2);

// s2::s2_interpolate_normalized on the two-vertex line (lon0, lat0)-(lon1, lat1)
// (S2Polyline::Interpolate: S2::InterpolateAtDistance along the arc, the
// vertices at the ends), back to degrees as s2 writes them.
void s2_interpolate(double lon0, double lat0, double lon1, double lat1, double fraction, double& lon, double& lat);

// s2::s2_centroid of points (the normalised sum of their unit vectors), in degrees.
void s2_centroid(const double* lon, const double* lat, long long n, double& clon, double& clat);

// sf::st_line_sample(line, sample = fraction) of a two-vertex line in a plane:
// GDAL's OGRSimpleCurve::Value at fraction * the line's length.
void planar_sample(double x0, double y0, double x1, double y1, double fraction, double& x, double& y);

}  // namespace move
