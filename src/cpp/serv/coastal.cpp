// SPDX-License-Identifier: GPL-3.0-or-later
//
// Coastal vulnerability kernels, see coastal.hpp. The segment intersection is a
// port of GEOS 3.14 (LGPL-2.1-or-later, Copyright (C) 2001-2023 the GEOS
// contributors and Vivid Solutions): math/DD, algorithm/CGAlgorithmsDD,
// algorithm/LineIntersector (floating precision model), with GEOS's names.
#include "serv/coastal.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace serv {

namespace {

struct Coord {
    double x, y;
};

bool equals2D(const Coord& a, const Coord& b) { return a.x == b.x && a.y == b.y; }

// ---------------------------------------------------------------------------
// math/DD.cpp: double-double arithmetic

struct DD {
    double hi;
    double lo;
    explicit DD(double v) : hi(v), lo(0.0) {}
    DD(double h, double l) : hi(h), lo(l) {}
};

constexpr double SPLIT = 134217729.0;  // 2^27 + 1

void selfAdd(DD& a, double yhi, double ylo) {
    double H, h, T, t, S, s, e, f;
    S = a.hi + yhi;
    T = a.lo + ylo;
    e = S - a.hi;
    f = T - a.lo;
    s = S - e;
    t = T - f;
    s = (yhi - e) + (a.hi - s);
    t = (ylo - f) + (a.lo - t);
    e = s + T;
    H = S + e;
    h = e + (S - H);
    e = t + h;
    const double zhi = H + e;
    const double zlo = e + (H - zhi);
    a.hi = zhi;
    a.lo = zlo;
}

DD operator+(const DD& lhs, const DD& rhs) {
    DD rv(lhs.hi, lhs.lo);
    selfAdd(rv, rhs.hi, rhs.lo);
    return rv;
}

DD operator-(const DD& lhs, const DD& rhs) {
    DD rv(lhs.hi, lhs.lo);
    selfAdd(rv, -1 * rhs.hi, -1 * rhs.lo);
    return rv;
}

DD operator*(const DD& lhs, const DD& rhs) {
    const double yhi = rhs.hi, ylo = rhs.lo;
    double hx, tx, hy, ty, C, c;
    C = SPLIT * lhs.hi;
    hx = C - lhs.hi;
    c = SPLIT * yhi;
    hx = C - hx;
    tx = lhs.hi - hx;
    hy = c - yhi;
    C = lhs.hi * yhi;
    hy = c - hy;
    ty = yhi - hy;
    c = ((((hx * hy - C) + hx * ty) + tx * hy) + tx * ty) + (lhs.hi * ylo + lhs.lo * yhi);
    const double zhi = C + c;
    hx = C - zhi;
    const double zlo = c + hx;
    return DD(zhi, zlo);
}

DD operator/(const DD& lhs, const DD& rhs) {
    const double yhi = rhs.hi, ylo = rhs.lo;
    double hc, tc, hy, ty, C, c, U, u;
    C = lhs.hi / yhi;
    c = SPLIT * C;
    hc = c - C;
    u = SPLIT * yhi;
    hc = c - hc;
    tc = C - hc;
    hy = u - yhi;
    U = C * yhi;
    hy = u - hy;
    ty = yhi - hy;
    u = (((hc * hy - U) + hc * ty) + tc * hy) + tc * ty;
    c = ((((lhs.hi - U) - u) + lhs.lo) - C * ylo) / yhi;
    u = C + c;
    return DD(u, (C - u) + c);
}

// ---------------------------------------------------------------------------
// algorithm/CGAlgorithmsDD.cpp

int orientation(double x) {
    if (x < 0) return -1;
    if (x > 0) return 1;
    return 0;
}

int orientationIndexFilter(double pax, double pay, double pbx, double pby, double pcx, double pcy) {
    constexpr double DP_SAFE_EPSILON = 1e-15;
    double detsum;
    const double detleft = (pax - pcx) * (pby - pcy);
    const double detright = (pay - pcy) * (pbx - pcx);
    const double det = detleft - detright;
    if (detleft > 0.0) {
        if (detright <= 0.0) return orientation(det);
        detsum = detleft + detright;
    } else if (detleft < 0.0) {
        if (detright >= 0.0) return orientation(det);
        detsum = -detleft - detright;
    } else {
        return orientation(det);
    }
    const double errbound = DP_SAFE_EPSILON * detsum;
    if ((det >= errbound) || (-det >= errbound)) return orientation(det);
    return 2;
}

int orientationIndex(const Coord& p1, const Coord& p2, const Coord& q) {
    const int index = orientationIndexFilter(p1.x, p1.y, p2.x, p2.y, q.x, q.y);
    if (index <= 1) return index;
    const DD dx1 = DD(p2.x) + DD(-p1.x);
    const DD dy1 = DD(p2.y) + DD(-p1.y);
    const DD dx2 = DD(q.x) + DD(-p2.x);
    const DD dy2 = DD(q.y) + DD(-p2.y);
    const DD d = (dx1 * dy2) - (dy1 * dx2);
    if (d.hi < 0.0 || (d.hi == 0.0 && d.lo < 0.0)) return -1;
    if (d.hi > 0.0 || (d.hi == 0.0 && d.lo > 0.0)) return 1;
    return 0;
}

bool intersectionDD(const Coord& p1, const Coord& p2, const Coord& q1, const Coord& q2, Coord& out) {
    const DD q1x(q1.x), q1y(q1.y), q2x(q2.x), q2y(q2.y);
    const DD p1x(p1.x), p1y(p1.y), p2x(p2.x), p2y(p2.y);
    const DD px = p1y - p2y;
    const DD py = p2x - p1x;
    const DD pw = (p1x * p2y) - (p2x * p1y);
    const DD qx = q1y - q2y;
    const DD qy = q2x - q1x;
    const DD qw = (q1x * q2y) - (q2x * q1y);
    const DD x = (py * qw) - (qy * pw);
    const DD y = (qx * pw) - (px * qw);
    const DD w = (px * qy) - (qx * py);
    const DD xd = x / w, yd = y / w;
    const double xInt = xd.hi + xd.lo;
    const double yInt = yd.hi + yd.lo;
    if (!std::isfinite(xInt) || !std::isfinite(yInt)) return false;
    out = Coord{xInt, yInt};
    return true;
}

// ---------------------------------------------------------------------------
// algorithm/LineIntersector.cpp

bool envelopeContains(const Coord& p1, const Coord& p2, const Coord& q) {
    return ((q.x >= (p1.x < p2.x ? p1.x : p2.x)) && (q.x <= (p1.x > p2.x ? p1.x : p2.x))) &&
           ((q.y >= (p1.y < p2.y ? p1.y : p2.y)) && (q.y <= (p1.y > p2.y ? p1.y : p2.y)));
}

bool segmentEnvelopesIntersect(const Coord& p1, const Coord& p2, const Coord& q1, const Coord& q2) {
    double minq = std::min(q1.x, q2.x), maxq = std::max(q1.x, q2.x);
    double minp = std::min(p1.x, p2.x), maxp = std::max(p1.x, p2.x);
    if (minp > maxq || maxp < minq) return false;
    minq = std::min(q1.y, q2.y);
    maxq = std::max(q1.y, q2.y);
    minp = std::min(p1.y, p2.y);
    maxp = std::max(p1.y, p2.y);
    return !(minp > maxq || maxp < minq);
}

double distance(const Coord& a, const Coord& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

double pointToSegment(const Coord& p, const Coord& A, const Coord& B) {
    if (equals2D(A, B)) return distance(p, A);
    const double len2 = (B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y);
    const double r = ((p.x - A.x) * (B.x - A.x) + (p.y - A.y) * (B.y - A.y)) / len2;
    if (r <= 0.0) return distance(p, A);
    if (r >= 1.0) return distance(p, B);
    const double s = ((A.y - p.y) * (B.x - A.x) - (A.x - p.x) * (B.y - A.y)) / len2;
    return std::fabs(s) * std::sqrt(len2);
}

Coord nearestEndpoint(const Coord& p1, const Coord& p2, const Coord& q1, const Coord& q2) {
    Coord nearestPt = p1;
    double minDist = pointToSegment(p1, q1, q2);
    double dist = pointToSegment(p2, q1, q2);
    if (dist < minDist) {
        minDist = dist;
        nearestPt = p2;
    }
    dist = pointToSegment(q1, p1, p2);
    if (dist < minDist) {
        minDist = dist;
        nearestPt = q1;
    }
    dist = pointToSegment(q2, p1, p2);
    if (dist < minDist) nearestPt = q2;
    return nearestPt;
}

// LineIntersector::computeIntersect of segments p1-p2 and q1-q2: 0 none, 1 a point
// (pt[0]), 2 collinear (pt[0], pt[1]).
int computeIntersect(const Coord& p1, const Coord& p2, const Coord& q1, const Coord& q2, Coord pt[2]) {
    if (!segmentEnvelopesIntersect(p1, p2, q1, q2)) return 0;
    const int Pq1 = orientationIndex(p1, p2, q1);
    const int Pq2 = orientationIndex(p1, p2, q2);
    if ((Pq1 > 0 && Pq2 > 0) || (Pq1 < 0 && Pq2 < 0)) return 0;
    const int Qp1 = orientationIndex(q1, q2, p1);
    const int Qp2 = orientationIndex(q1, q2, p2);
    if ((Qp1 > 0 && Qp2 > 0) || (Qp1 < 0 && Qp2 < 0)) return 0;
    if (Pq1 == 0 && Pq2 == 0 && Qp1 == 0 && Qp2 == 0) {
        const bool q1inP = envelopeContains(p1, p2, q1);
        const bool q2inP = envelopeContains(p1, p2, q2);
        const bool p1inQ = envelopeContains(q1, q2, p1);
        const bool p2inQ = envelopeContains(q1, q2, p2);
        if (q1inP && q2inP) { pt[0] = q1; pt[1] = q2; return 2; }
        if (p1inQ && p2inQ) { pt[0] = p1; pt[1] = p2; return 2; }
        if (q1inP && p1inQ) { pt[0] = q1; pt[1] = p1; return equals2D(q1, p1) && !q2inP && !p2inQ ? 1 : 2; }
        if (q1inP && p2inQ) { pt[0] = q1; pt[1] = p2; return equals2D(q1, p2) && !q2inP && !p1inQ ? 1 : 2; }
        if (q2inP && p1inQ) { pt[0] = q2; pt[1] = p1; return equals2D(q2, p1) && !q1inP && !p2inQ ? 1 : 2; }
        if (q2inP && p2inQ) { pt[0] = q2; pt[1] = p2; return equals2D(q2, p2) && !q1inP && !p1inQ ? 1 : 2; }
        return 0;
    }
    Coord p;
    if (Pq1 == 0 || Pq2 == 0 || Qp1 == 0 || Qp2 == 0) {
        // the intersection is an endpoint: copied, exact
        if (equals2D(p1, q1)) p = p1;
        else if (equals2D(p1, q2)) p = p1;
        else if (equals2D(p2, q1)) p = p2;
        else if (equals2D(p2, q2)) p = p2;
        else if (Pq1 == 0) p = q1;
        else if (Pq2 == 0) p = q2;
        else if (Qp1 == 0) p = p1;
        else p = p2;
    } else {
        Coord ptInt;
        if (!intersectionDD(p1, p2, q1, q2, ptInt)) ptInt = nearestEndpoint(p1, p2, q1, q2);
        if (!(envelopeContains(p1, p2, ptInt) && envelopeContains(q1, q2, ptInt)))
            ptInt = nearestEndpoint(p1, p2, q1, q2);
        p = ptInt;
    }
    pt[0] = p;
    return 1;
}

// ---------------------------------------------------------------------------
// numpy's pairwise sum (np.add.reduce of a contiguous array, accumulators in T)

template <class T>
T pairwise(const T* a, i64 n) {
    if (n < 8) {
        T res = -T(0);
        for (i64 i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        T r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        i64 i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    i64 n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

// np.sum: its identity 0 plus the pairwise sum.
template <class T>
T np_sum(const T* a, i64 n) {
    return T(0) + pairwise(a, n);
}

// np.mean of n values (n > 0): the sum divided by n in T.
template <class T>
double np_mean(const std::vector<T>& v) {
    const T s = np_sum(v.data(), static_cast<i64>(v.size()));
    return static_cast<double>(s / static_cast<T>(v.size()));
}

// numpy.isclose(v, nodata, equal_nan=True), computed in float64.
bool isclose(double v, double nd) {
    if (std::isnan(v) && std::isnan(nd)) return true;
    if (v == nd) return true;
    return std::isfinite(nd) && std::fabs(v - nd) <= 1e-8 + 1e-5 * std::fabs(nd);
}

// ---------------------------------------------------------------------------
// a uniform grid over the segments' envelopes

struct SegmentGrid {
    double x0 = 0, y0 = 0, cell = 1;
    i64 nx = 0, ny = 0;
    std::vector<i64> start, ids;

    SegmentGrid(const double* seg, i64 m) {
        if (m == 0) return;
        double minx = std::numeric_limits<double>::infinity(), miny = minx, maxx = -minx, maxy = -minx;
        double total = 0;
        for (i64 i = 0; i < m; ++i) {
            const double* s = seg + 4 * i;
            minx = std::min({minx, s[0], s[2]});
            maxx = std::max({maxx, s[0], s[2]});
            miny = std::min({miny, s[1], s[3]});
            maxy = std::max({maxy, s[1], s[3]});
            total += std::hypot(s[2] - s[0], s[3] - s[1]);
        }
        const double w = std::max(maxx - minx, 1e-9), h = std::max(maxy - miny, 1e-9);
        cell = std::max(2.0 * total / static_cast<double>(m), std::sqrt(w * h / (4.0 * static_cast<double>(m))));
        cell = std::max(cell, std::max(w, h) / 4096.0);
        const double pad = 1e-6 * cell + 1e-9 * std::max(std::fabs(minx), std::fabs(miny));
        x0 = minx - 2 * pad;
        y0 = miny - 2 * pad;
        nx = static_cast<i64>((w + 4 * pad) / cell) + 1;
        ny = static_cast<i64>((h + 4 * pad) / cell) + 1;
        std::vector<i64> counts(static_cast<std::size_t>(nx * ny) + 1, 0);
        auto range = [&](const double* s, i64& cx0, i64& cx1, i64& cy0, i64& cy1) {
            cx0 = clampx(static_cast<i64>(std::floor((std::min(s[0], s[2]) - pad - x0) / cell)));
            cx1 = clampx(static_cast<i64>(std::floor((std::max(s[0], s[2]) + pad - x0) / cell)));
            cy0 = clampy(static_cast<i64>(std::floor((std::min(s[1], s[3]) - pad - y0) / cell)));
            cy1 = clampy(static_cast<i64>(std::floor((std::max(s[1], s[3]) + pad - y0) / cell)));
        };
        i64 cx0, cx1, cy0, cy1;
        for (i64 i = 0; i < m; ++i) {
            range(seg + 4 * i, cx0, cx1, cy0, cy1);
            for (i64 cy = cy0; cy <= cy1; ++cy)
                for (i64 cx = cx0; cx <= cx1; ++cx) ++counts[static_cast<std::size_t>(cy * nx + cx) + 1];
        }
        for (std::size_t k = 1; k < counts.size(); ++k) counts[k] += counts[k - 1];
        start = counts;
        ids.resize(static_cast<std::size_t>(counts.back()));
        for (i64 i = 0; i < m; ++i) {
            range(seg + 4 * i, cx0, cx1, cy0, cy1);
            for (i64 cy = cy0; cy <= cy1; ++cy)
                for (i64 cx = cx0; cx <= cx1; ++cx) ids[static_cast<std::size_t>(counts[static_cast<std::size_t>(cy * nx + cx)]++)] = i;
        }
    }
    i64 clampx(i64 v) const { return std::min(std::max(v, i64(0)), nx - 1); }
    i64 clampy(i64 v) const { return std::min(std::max(v, i64(0)), ny - 1); }
};

// The nearest intersection of the ray a-b with the segments (parameter t along a-b).
bool nearestHit(const SegmentGrid& g, const double* seg, const Coord& a, const Coord& b, std::vector<i64>& stamp,
                i64 tag, Coord& hit) {
    if (g.nx == 0) return false;
    const double dx = b.x - a.x, dy = b.y - a.y;
    const double dd = dx * dx + dy * dy;
    // the ray within the grid's box
    double t0 = 0.0, t1 = 1.0;
    const double bx0 = g.x0, bx1 = g.x0 + g.cell * static_cast<double>(g.nx);
    const double by0 = g.y0, by1 = g.y0 + g.cell * static_cast<double>(g.ny);
    auto slab = [&](double o, double d, double lo, double hi) {
        if (d == 0.0) return o >= lo && o <= hi;
        double ta = (lo - o) / d, tb = (hi - o) / d;
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
        return t0 <= t1;
    };
    if (!slab(a.x, dx, bx0, bx1) || !slab(a.y, dy, by0, by1)) return false;
    const double sx = a.x + dx * t0, sy = a.y + dy * t0;
    i64 cx = g.clampx(static_cast<i64>(std::floor((sx - g.x0) / g.cell)));
    i64 cy = g.clampy(static_cast<i64>(std::floor((sy - g.y0) / g.cell)));
    const i64 stepx = dx > 0 ? 1 : (dx < 0 ? -1 : 0), stepy = dy > 0 ? 1 : (dy < 0 ? -1 : 0);
    const double inf = std::numeric_limits<double>::infinity();
    auto next_t = [&](i64 c, i64 step, double o, double d, double origin) {
        if (step == 0) return inf;
        const double edge = origin + g.cell * static_cast<double>(step > 0 ? c + 1 : c);
        return (edge - o) / d;
    };
    double tmx = next_t(cx, stepx, a.x, dx, g.x0), tmy = next_t(cy, stepy, a.y, dy, g.y0);
    const double tdx = stepx ? g.cell / std::fabs(dx) : inf, tdy = stepy ? g.cell / std::fabs(dy) : inf;
    double entry = t0, best = inf;
    bool found = false;
    const Coord p1 = a, p2 = b;
    for (;;) {
        if (entry > best + 1e-9) break;
        const std::size_t c = static_cast<std::size_t>(cy * g.nx + cx);
        for (i64 k = g.start[c]; k < g.start[c + 1]; ++k) {
            const i64 id = g.ids[static_cast<std::size_t>(k)];
            if (stamp[static_cast<std::size_t>(id)] == tag) continue;
            stamp[static_cast<std::size_t>(id)] = tag;
            const double* s = seg + 4 * id;
            const Coord q1{s[0], s[1]}, q2{s[2], s[3]};
            Coord pt[2];
            const int kind = computeIntersect(p1, p2, q1, q2, pt);
            for (int j = 0; j < kind; ++j) {
                const double t = ((pt[j].x - a.x) * dx + (pt[j].y - a.y) * dy) / dd;
                if (t < best) {
                    best = t;
                    hit = pt[j];
                    found = true;
                }
            }
        }
        if (tmx < tmy) {
            cx += stepx;
            entry = tmx;
            tmx += tdx;
            if (cx < 0 || cx >= g.nx) break;
        } else {
            if (tmy == inf) break;
            cy += stepy;
            entry = tmy;
            tmy += tdy;
            if (cy < 0 || cy >= g.ny) break;
        }
        if (entry > t1) break;
    }
    return found;
}

int n_threads(int threads) { return threads > 0 ? threads : 1; }

}  // namespace

void coastal_rays(const double* points, i64 n, const std::uint8_t* on_land, const double* deltas, int ndir,
                  double max_fetch, const double* segments, i64 m, double* fdist, double* ends, int threads) {
    const SegmentGrid grid(segments, m);
    const double nan = std::numeric_limits<double>::quiet_NaN();
#pragma omp parallel num_threads(n_threads(threads))
    {
        std::vector<i64> stamp(static_cast<std::size_t>(m), -1);
#pragma omp for schedule(dynamic, 16)
        for (i64 i = 0; i < n; ++i) {
            for (int k = 0; k < ndir; ++k) {
                const i64 o = i * ndir + k;
                if (on_land[o]) {
                    fdist[o] = 0.0;
                    ends[2 * o] = ends[2 * o + 1] = nan;
                    continue;
                }
                const double ddx = deltas[2 * k], ddy = deltas[2 * k + 1];
                const Coord a{points[2 * i] + ddx * 1, points[2 * i + 1] + ddy * 1};
                const Coord b{a.x + ddx * max_fetch, a.y + ddy * max_fetch};
                Coord e = b, hit{};
                if (nearestHit(grid, segments, a, b, stamp, o, hit)) e = hit;
                const double dx = e.x - a.x, dy = e.y - a.y;
                fdist[o] = std::sqrt(dx * dx + dy * dy);
                ends[2 * o] = e.x;
                ends[2 * o + 1] = e.y;
            }
        }
    }
}

template <class T>
i64 coastal_ray_depth(const double* starts, const double* ends, i64 r, const T* raster, i64 ny, i64 nx,
                      const double* gt, bool has_nodata, double nodata, double* depth, int threads) {
    std::atomic<i64> failed{std::numeric_limits<i64>::max()};
    std::atomic<i64> empty{std::numeric_limits<i64>::max()};
#pragma omp parallel num_threads(n_threads(threads))
    {
        std::vector<T> vals;
#pragma omp for schedule(dynamic, 64)
        for (i64 j = 0; j < r; ++j) {
            const Coord p0{starts[2 * j], starts[2 * j + 1]}, p1{ends[2 * j], ends[2 * j + 1]};
            const double dx = p1.x - p0.x, dy = p1.y - p0.y;
            const double length = std::sqrt(dx * dx + dy * dy);  // OGR's and GEOS's length
            const i64 npts = static_cast<i64>(std::ceil(length / gt[1]));
            vals.clear();
            i64 ix = 0, iy = 0;
            bool off = false;
            for (i64 i = 0; i <= npts; ++i) {
                // GEOS's Interpolate(d = i / npts * length): LengthLocationMap, LinearLocation
                const double d = (static_cast<double>(i) / static_cast<double>(npts)) * length;
                Coord p = p1;
                if (d <= 0.0) {
                    p = p0;
                } else if (0.0 + length > d) {
                    const double frac = (d - 0.0) / length;
                    if (frac <= 0.0) p = p0;
                    else if (frac >= 1.0) p = p1;
                    else p = Coord{(p1.x - p0.x) * frac + p0.x, (p1.y - p0.y) * frac + p0.y};
                }
                ix = static_cast<i64>((p.x - gt[0]) / gt[1]);
                iy = static_cast<i64>((p.y - gt[3]) / gt[5]);
                if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) {
                    off = true;
                    break;
                }
                const T v = raster[iy * nx + ix];
                if (std::isnan(static_cast<double>(v))) continue;
                if (has_nodata && isclose(static_cast<double>(v), nodata)) continue;
                vals.push_back(v);
            }
            if (off) {
                i64 cur = failed.load();
                while (j < cur && !failed.compare_exchange_weak(cur, j)) {
                }
                depth[j] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            if (!vals.empty()) {
                depth[j] = np_mean(vals);
                continue;
            }
            // no valid cell along the ray: the smallest square around the end with one
            for (i64 k = 0;; ++k) {
                const i64 x0 = std::max(ix - k, i64(0)), y0 = std::max(iy - k, i64(0));
                const i64 x1 = std::min(ix + k + 1, nx), y1 = std::min(iy + k + 1, ny);
                if (x0 == 0 && y0 == 0 && x1 == nx && y1 == ny) {
                    i64 cur = empty.load();
                    while (j < cur && !empty.compare_exchange_weak(cur, j)) {
                    }
                    depth[j] = std::numeric_limits<double>::quiet_NaN();
                    break;
                }
                const i64 a0 = std::max(ix - k - 1, i64(0)), b0 = std::max(iy - k - 1, i64(0));
                const i64 a1 = std::min(ix + k + 2, nx), b1 = std::min(iy + k + 2, ny);
                vals.clear();
                for (i64 y = b0; y < b1; ++y)
                    for (i64 x = a0; x < a1; ++x) {
                        const T v = raster[y * nx + x];
                        if (std::isnan(static_cast<double>(v))) continue;
                        if (has_nodata && isclose(static_cast<double>(v), nodata)) continue;
                        vals.push_back(v);
                    }
                if (!vals.empty()) {
                    depth[j] = np_mean(vals);
                    break;
                }
            }
        }
    }
    if (failed.load() != std::numeric_limits<i64>::max()) return failed.load();
    if (empty.load() != std::numeric_limits<i64>::max()) return -2 - empty.load();
    return -1;
}

template <class T>
void coastal_aggregate(const double* points, i64 n, const T* raster, i64 ny, i64 nx, const double* gt,
                       bool has_nodata, double nodata, bool exact_nodata, const std::uint8_t* kernel, i64 ksize,
                       i64 pixel_dist, double* sum, i64* count, std::uint8_t* nonzero, int threads) {
#pragma omp parallel num_threads(n_threads(threads))
    {
        std::vector<T> vals;
#pragma omp for schedule(dynamic, 64)
        for (i64 i = 0; i < n; ++i) {
            i64 win_x = ksize, win_y = ksize;
            i64 px = static_cast<i64>((points[2 * i] - gt[0]) / gt[1]) - pixel_dist;
            i64 py = static_cast<i64>((points[2 * i + 1] - gt[3]) / gt[5]) - pixel_dist;
            i64 kx0 = 0, ky0 = 0;  // first column and row of the kernel in the window
            if (px < 0) {
                win_x += px;
                px = 0;
                kx0 = ksize - win_x;  // kernel_mask[:, -win_x:]
            }
            if (py < 0) {
                win_y += py;
                py = 0;
                ky0 = ksize - win_y;
            }
            if (px + win_x > nx) win_x -= px + win_x - nx;
            if (py + win_y > ny) win_y -= py + win_y - ny;
            if (win_x <= 0 || win_y <= 0) {
                sum[i] = 0.0;
                count[i] = -1;
                nonzero[i] = 0;
                continue;
            }
            vals.clear();
            bool any = false;
            for (i64 y = 0; y < win_y; ++y)
                for (i64 x = 0; x < win_x; ++x) {
                    if (!kernel[(ky0 + y) * ksize + kx0 + x]) continue;
                    const T v = raster[(py + y) * nx + px + x];
                    const double dv = static_cast<double>(v);
                    if (std::isnan(dv)) continue;
                    if (has_nodata && (exact_nodata ? dv == nodata : isclose(dv, nodata))) continue;
                    vals.push_back(v);
                    if (dv != 0.0) any = true;
                }
            sum[i] = static_cast<double>(np_sum(vals.data(), static_cast<i64>(vals.size())));
            count[i] = static_cast<i64>(vals.size());
            nonzero[i] = any ? 1 : 0;
        }
    }
}

template i64 coastal_ray_depth<float>(const double*, const double*, i64, const float*, i64, i64, const double*, bool,
                                      double, double*, int);
template i64 coastal_ray_depth<double>(const double*, const double*, i64, const double*, i64, i64, const double*, bool,
                                       double, double*, int);
template void coastal_aggregate<float>(const double*, i64, const float*, i64, i64, const double*, bool, double, bool,
                                       const std::uint8_t*, i64, i64, double*, i64*, std::uint8_t*, int);
template void coastal_aggregate<double>(const double*, i64, const double*, i64, i64, const double*, bool, double,
                                        bool, const std::uint8_t*, i64, i64, double*, i64*, std::uint8_t*, int);

}  // namespace serv
