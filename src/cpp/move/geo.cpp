// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/geo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "lsm/rmath.hpp"

namespace move {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Point3 {
    double x, y, z;
};

// s2geography's lnglat projection: S2::PlateCarreeProjection(180).Unproject
Point3 s2_point(double lng, double lat) {
    constexpr double to_radians = kPi / 180.0;
    const double phi = to_radians * lat;
    const double theta = to_radians * std::remainder(lng, 360.0);
    const double cosphi = lsm::r_cos(phi);
    return {lsm::r_cos(theta) * cosphi, lsm::r_sin(theta) * cosphi, lsm::r_sin(phi)};
}

// liblwgeom's latitude_radians_normalize / longitude_radians_normalize
double lat_normalize(double lat) {
    if (lat > 2.0 * kPi) lat = std::remainder(lat, 2.0 * kPi);
    if (lat < -2.0 * kPi) lat = std::remainder(lat, -2.0 * kPi);
    if (lat > kPi) lat = kPi - lat;
    if (lat < -1.0 * kPi) lat = -1.0 * kPi - lat;
    if (lat > kPi / 2) lat = kPi - lat;
    if (lat < -1.0 * kPi / 2) lat = -1.0 * kPi - lat;
    return lat;
}

double lon_normalize(double lon) {
    if (lon == -1.0 * kPi) return kPi;
    if (lon == -2.0 * kPi) return 0.0;
    if (lon > 2.0 * kPi) lon = std::remainder(lon, 2.0 * kPi);
    if (lon < -2.0 * kPi) lon = std::remainder(lon, -2.0 * kPi);
    if (lon > kPi) lon = -2.0 * kPi + lon;
    if (lon < -1.0 * kPi) lon = 2.0 * kPi + lon;
    if (lon == -2.0 * kPi) lon *= -1.0;
    return lon;
}

}  // namespace

double s2_distance(double lon1, double lat1, double lon2, double lat2) {
    if (std::isnan(lon1) || std::isnan(lat1) || std::isnan(lon2) || std::isnan(lat2)) return kNaN;
    const Point3 a = s2_point(lon1, lat1), b = s2_point(lon2, lat2);
    const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    const double length2 = std::min(4.0, dx * dx + dy * dy + dz * dz);
    return (2 * std::asin(0.5 * std::sqrt(length2))) * kS2EarthRadius;
}

double planar_distance(double x1, double y1, double x2, double y2) {
    const double dx = x1 - x2, dy = y1 - y2;
    return lsm::r_sqrt(dx * dx + dy * dy);
}

double lwgeom_azimuth(const Spheroid& s, double lon1, double lat1, double lon2, double lat2) {
    if (std::isnan(lon1) || std::isnan(lat1) || std::isnan(lon2) || std::isnan(lat2)) return kNaN;
    if (std::fabs(lon1 - lon2) <= 1e-14 && std::fabs(lat1 - lat2) <= 1e-14) return kNaN;
    // geographic_point_init: deg2rad(d) = M_PI * d / 180, normalised
    const double glat1 = lat_normalize(kPi * lat1 / 180.0), glon1 = lon_normalize(kPi * lon1 / 180.0);
    const double glat2 = lat_normalize(kPi * lat2 / 180.0), glon2 = lon_normalize(kPi * lon2 / 180.0);
    const double azi1 = s.azimuth_deg(glat1 * 180.0 / kPi, glon1 * 180.0 / kPi, glat2 * 180.0 / kPi,
                                      glon2 * 180.0 / kPi);
    return azi1 * kPi / 180.0;
}

double planar_azimuth(double x1, double y1, double x2, double y2) {
    if (std::isnan(x1) || std::isnan(y1) || std::isnan(x2) || std::isnan(y2)) return kNaN;
    return lsm::r_atan2(x2 - x1, y2 - y1);
}

namespace {

Point3 add(const Point3& a, const Point3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Point3 sub(const Point3& a, const Point3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Point3 scale(double k, const Point3& a) { return {k * a.x, k * a.y, k * a.z}; }
double dot(const Point3& a, const Point3& b) {
    double s = 0;
    s += a.x * b.x;
    s += a.y * b.y;
    s += a.z * b.z;
    return s;
}
double norm(const Point3& a) { return std::sqrt(dot(a, a)); }
Point3 cross(const Point3& a, const Point3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Point3 normalize(const Point3& a) {
    double n = norm(a);
    if (n != 0.0) n = 1.0 / n;
    return {a.x * n, a.y * n, a.z * n};
}
// Vector3::Angle: atan2(|a x b|, a . b)
double angle(const Point3& a, const Point3& b) { return lsm::r_atan2(norm(cross(a, b)), dot(a, b)); }

// S2::Ortho
Point3 ortho(const Point3& a) {
    const double ax = std::fabs(a.x), ay = std::fabs(a.y), az = std::fabs(a.z);
    // LargestAbsComponent, then (k + 1) % 3 is the component set to 1
    const int k = (ax > ay) ? ((ax > az) ? 0 : 2) : ((ay > az) ? 1 : 2);
    Point3 temp{0.012, 0.0053, 0.00457};
    const int j = (k + 1) % 3;
    if (j == 0) temp.x = 1; else if (j == 1) temp.y = 1; else temp.z = 1;
    return normalize(cross(a, temp));
}

// S2LatLng(p) in degrees, as PlateCarreeProjection(180).FromLatLng writes it
void to_lnglat(const Point3& p, double& lon, double& lat) {
    constexpr double from_radians = 180.0 / kPi;
    lat = from_radians * lsm::r_atan2(p.z, std::sqrt(p.x * p.x + p.y * p.y));
    lon = from_radians * lsm::r_atan2(p.y, p.x);
}

}  // namespace

void s2_interpolate(double lon0, double lat0, double lon1, double lat1, double fraction, double& lon, double& lat) {
    const Point3 a = s2_point(lon0, lat0), b = s2_point(lon1, lat1);
    Point3 r;
    if (fraction <= 0) {
        r = a;
    } else {
        const double length = 0.0 + angle(a, b);
        const double target = fraction * length;
        if (target < length) {
            // S2::InterpolateAtDistance with S2::RobustCrossProd
            Point3 normal = cross(add(b, a), sub(b, a));
            if (normal.x == 0 && normal.y == 0 && normal.z == 0) normal = ortho(a);
            const Point3 tangent = cross(normal, a);
            r = normalize(add(scale(lsm::r_cos(target), a), scale(lsm::r_sin(target) / norm(tangent), tangent)));
        } else {
            r = b;
        }
    }
    to_lnglat(r, lon, lat);
}

void s2_centroid(const double* lon, const double* lat, long long n, double& clon, double& clat) {
    Point3 c{0, 0, 0};
    for (long long i = 0; i < n; ++i) c = add(c, s2_point(lon[i], lat[i]));
    to_lnglat(normalize(c), clon, clat);
}

void planar_sample(double x0, double y0, double x1, double y1, double fraction, double& x, double& y) {
    const double dx = x1 - x0, dy = y1 - y0;
    const double length = std::sqrt(dx * dx + dy * dy);
    const double distance = fraction * length;
    if (distance <= 0) {
        x = x0;
        y = y0;
        return;
    }
    if (distance >= length) {
        x = x1;
        y = y1;
        return;
    }
    const double ratio = distance / length;
    x = x0 * (1 - ratio) + x1 * ratio;
    y = y0 * (1 - ratio) + y1 * ratio;
}

}  // namespace move
