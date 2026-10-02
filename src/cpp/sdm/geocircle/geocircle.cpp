// SPDX-License-Identifier: GPL-3.0-or-later
#include "sdm/geocircle/geocircle.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "sdm/geocircle/geocircle_libm.hpp"
#include "sdm/geocircle/geos_lite.hpp"

namespace sdm {

namespace {

using geos_lite::Coord;
using geos_lite::Ring;

GeoRing to_interleaved(const Ring& r) {
    GeoRing out;
    out.reserve(2 * r.size());
    for (const Coord& c : r) {
        out.push_back(c.x);
        out.push_back(c.y);
    }
    return out;
}

GeoRing to_interleaved(const std::vector<double>& x, const std::vector<double>& y) {
    GeoRing out;
    out.reserve(2 * x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out.push_back(x[i]);
        out.push_back(y[i]);
    }
    return out;
}

// terra's vmin / vmax with narm = false: NaN if any value is NaN.
double vmin(const std::vector<double>& v) {
    double m = v[0];
    for (double x : v) {
        if (std::isnan(x)) return x;
        if (x < m) m = x;
    }
    return m;
}

double vmax(const std::vector<double>& v) {
    double m = v[0];
    for (double x : v) {
        if (std::isnan(x)) return x;
        if (x > m) m = x;
    }
    return m;
}

// terra's split_dateline (distVector.cpp): intersect the ring (longitudes
// already in [0, 360)) with the rectangles [-1, 180] x [-91, 91] and
// [180, 361] x [-91, 91], as SpatVector(SpatExtent) builds them, and shift
// the second piece by -360.
GeoParts split_dateline(const std::vector<double>& x, const std::vector<double>& y) {
    Ring circle(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) circle[i] = Coord{x[i], y[i]};
    // SpatVector(SpatExtent e): x = {xmin, xmin, xmax, xmax, xmin},
    // y = {ymin, ymax, ymax, ymin, ymin}
    const Ring west{{-1, -91}, {-1, 91}, {180, 91}, {180, -91}, {-1, -91}};
    const Ring east{{180, -91}, {180, 91}, {361, 91}, {361, -91}, {180, -91}};
    const std::vector<Ring> w = geos_lite::overlay(circle, west, geos_lite::OverlayOp::Intersection);
    std::vector<Ring> e = geos_lite::overlay(circle, east, geos_lite::OverlayOp::Intersection);
    if (w.empty() || e.empty()) throw std::runtime_error("terra_point_buffer: dateline split without two pieces");
    GeoParts out;
    for (const Ring& r : w) out.push_back(to_interleaved(r));
    for (Ring& r : e) {
        for (Coord& c : r) c.x = c.x + -360.0;  // SpatVector::shift(-360, 0)
        out.push_back(to_interleaved(r));
    }
    return out;
}

// terra's fix_date_line: x is the closed ring's longitudes.
GeoParts fix_date_line(std::vector<double> x, const std::vector<double>& y) {
    const std::vector<double> x0 = x;
    const double minx = vmin(x), maxx = vmax(x);
    if ((maxx - minx) > 180) {
        for (double& v : x)
            if (v < 0) v += 360;
        const double minx2 = vmin(x), maxx2 = vmax(x);
        if ((maxx - minx) < (maxx2 - minx2)) return {to_interleaved(x0, y)};
        return split_dateline(x, y);
    }
    return {to_interleaved(x0, y)};
}

// terra's sort_unique_2d: sort the vertices by (x, y), drop exact duplicates.
void sort_unique_2d(std::vector<double>& x, std::vector<double>& y) {
    std::vector<std::pair<double, double>> v(x.size());
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = {x[i], y[i]};
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    x.resize(v.size());
    y.resize(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        x[i] = v[i].first;
        y[i] = v[i].second;
    }
}

}  // namespace

std::vector<double> geosphere_ring(double lon, double lat, double d) {
    std::vector<double> out(2 * 73);
    for (int j = 0; j <= 72; ++j) {
        // seq(0, 360, by = 5) is 0 + (0:72) * 5: exact
        geosphere_direct(lon, lat, 5.0 * j, d, out[2 * j], out[2 * j + 1]);
    }
    return out;
}

GeoParts geod_polys_fixed(double lon, double lat, double d) {
    std::vector<double> m = geosphere_ring(lon, lat, d);
    const std::size_t n = m.size() / 2;  // 73
    const double to_pole_n = geosphere_distance(lon, lat, lon, 90);
    const double to_pole_s = geosphere_distance(lon, lat, lon, -90);
    if (to_pole_n < d || to_pole_s < d) {
        // polar cap: drop the closing vertex, order() by lon (stable), close
        // along the pole edge
        const double pole = to_pole_n < d ? 90 : -90;
        std::vector<std::pair<double, double>> v(n - 1);
        for (std::size_t i = 0; i + 1 < n; ++i) v[i] = {m[2 * i], m[2 * i + 1]};
        std::stable_sort(v.begin(), v.end(), [](const std::pair<double, double>& a,
                                                const std::pair<double, double>& b) { return a.first < b.first; });
        GeoRing r;
        r.reserve(2 * (v.size() + 5));
        r.insert(r.end(), {-180.0, v.front().second});
        for (const auto& p : v) r.insert(r.end(), {p.first, p.second});
        r.insert(r.end(), {180.0, v.back().second, 180.0, pole, -180.0, pole, -180.0, v.front().second});
        return {r};
    }
    double xmin = m[0], xmax = m[0];
    for (std::size_t i = 0; i < n; ++i) {
        xmin = std::min(xmin, m[2 * i]);
        xmax = std::max(xmax, m[2 * i]);
    }
    if (xmax - xmin > 180) {
        // unwrap around the centre, add the copy shifted by -/+360, and
        // dissolve the two (terra::aggregate -> GEOS UnaryUnion)
        Ring r1(n), r2(n);
        const double s = lon > 0 ? -360 : 360;
        for (std::size_t i = 0; i < n; ++i) {
            double x = m[2 * i];
            const double dx = x - lon;
            if (dx > 180) x = x - 360;
            if (dx < -180) x = x + 360;
            r1[i] = Coord{x, m[2 * i + 1]};
            r2[i] = Coord{x + s, m[2 * i + 1]};
        }
        GeoParts out;
        for (const Ring& r : geos_lite::unary_union({r1, r2})) out.push_back(to_interleaved(r));
        return out;
    }
    return {m};
}

GeoParts terra_point_buffer(double lon, double lat, double d, unsigned quadsegs) {
    if (!(d > 0)) throw std::invalid_argument("terra_point_buffer: a negative buffer is only meaningful with polygons");
    if (std::isnan(lon) || std::isnan(lat) || lat > 90 || lat < -90) return {};
    if (d > 20003931) {
        // the whole globe
        return {GeoRing{-180, 90, 0, 90, 180, 90, 180, 0, 180, -90, 0, -90, -180, -90, -180, 0, -180, 90}};
    }
    const std::size_t n = static_cast<std::size_t>(quadsegs) * 4;
    const double step = 360.0 / n;
    std::vector<double> ptx, pty;
    ptx.reserve(n + 5);
    pty.reserve(n + 5);
    for (std::size_t j = 0; j < n; ++j) {
        double lon2, lat2;
        terra_direct(lon, lat, j * step, d, lon2, lat2);
        ptx.push_back(lon2);
        pty.push_back(lat2);
    }
    const bool npole = terra_distance(lon, lat, lon, 90) < d;
    const bool spole = terra_distance(lon, lat, lon, -90) < d;
    if (npole && spole) {
        throw std::domain_error("terra_point_buffer: circle containing both poles is not supported");
    }
    if (npole || spole) {
        const double pole = npole ? 90 : -90;
        sort_unique_2d(ptx, pty);
        if (ptx.back() < 180) {
            ptx.push_back(180);
            pty.push_back(pty.back());
        }
        ptx.push_back(180);
        pty.push_back(pole);
        ptx.push_back(-180);
        pty.push_back(pole);
        if (ptx[0] > -180) {
            ptx.push_back(-180);
            pty.push_back(pty[0]);
        }
        ptx.push_back(ptx[0]);
        pty.push_back(pty[0]);
        return {to_interleaved(ptx, pty)};
    }
    ptx.push_back(ptx[0]);
    pty.push_back(pty[0]);
    return fix_date_line(ptx, pty);
}

GeoRing geos_circle(double cx, double cy, double r, int quadsegs) {
    if (!(r > 0)) return {};
    constexpr double MATH_PI = 3.14159265358979323846;
    const int quadSegs = quadsegs < 1 ? 1 : quadsegs;
    const double filletAngleQuantum = MATH_PI / 2.0 / quadSegs;
    const double minimumVertexDistance = r * 1.0E-6;  // CURVE_VERTEX_SNAP_DISTANCE_FACTOR
    GeoRing out;
    // OffsetSegmentString::addPt: drop a point closer than the minimum
    // vertex distance to the previous one
    auto addPt = [&out, minimumVertexDistance](double x, double y) {
        if (!out.empty()) {
            const double dx = x - out[out.size() - 2], dy = y - out[out.size() - 1];
            if (std::sqrt(dx * dx + dy * dy) < minimumVertexDistance) return;
        }
        out.push_back(x);
        out.push_back(y);
    };
    // OffsetSegmentGenerator::createCircle -> addDirectedFillet(p, 0, 2 pi,
    // CLOCKWISE, r)
    addPt(cx + r, cy);
    const double startAngle = 0.0;
    const double totalAngle = std::fabs(startAngle - 2.0 * MATH_PI);
    const int nSegs = static_cast<int>(totalAngle / filletAngleQuantum + 0.5);
    if (nSegs >= 1) {
        const double angleInc = totalAngle / nSegs;
        const int directionFactor = -1;
        for (int i = 0; i < nSegs; ++i) {
            const double angle = startAngle + directionFactor * i * angleInc;
            addPt(cx + r * geocircle_libm::cos(angle), cy + r * geocircle_libm::sin(angle));
        }
    }
    // closeRing
    if (out[0] != out[out.size() - 2] || out[1] != out[out.size() - 1]) {
        const double x0 = out[0], y0 = out[1];
        out.push_back(x0);
        out.push_back(y0);
    }
    return out;
}

}  // namespace sdm
