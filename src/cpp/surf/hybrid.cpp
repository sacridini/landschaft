// SPDX-License-Identifier: GPL-3.0-or-later
//
// geodiv's hybrid parameters (sdq.R, surfacearea.R): sdq, sdq6, surface_area,
// flatsa, sdr. Cells are visited in R's order (zshift: row by row), sums are
// R's 80-bit sums, R's vector sqrt is the x87 one (a single value: sqrtsd).

#include <cmath>
#include <limits>
#include <vector>

#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"
#include "surf/surface.hpp"

namespace surf {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// R's sum(a, b, na.rm = TRUE) of two sums: do_summary adds each argument's
// rsum into a double.
inline double sum2(double a, double b) { return a + b; }

// zshift(..., scale = TRUE): (z - min) / (max - min), min and max without NA.
std::vector<double> rescaled(const Grid& g) {
    const i64 n = g.size();
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (i64 i = 0; i < n; ++i) {
        const double v = g.z[i];
        if (std::isnan(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    const double range = hi - lo;
    std::vector<double> out(static_cast<std::size_t>(n));
#pragma omp parallel for schedule(static) if (n > 65536)
    for (i64 i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = (g.z[i] - lo) / range;
    return out;
}

// The triangulated area of every 2 x 2 block (surface_area / flatsa): the mean
// of the two triangulations, over the blocks whose top-left value is not NA,
// row by row; blocks with another NA are left out of the sum.
double triangulated_area(const std::vector<double>& z, int ny, int nx, double dx, double dy) {
    const double dx2 = dx * dx, dy2 = dy * dy;
    std::vector<double> akl;
    akl.reserve(static_cast<std::size_t>(ny > 1 && nx > 1 ? (ny - 1) * static_cast<i64>(nx - 1) : 0));
    for (int r = 0; r + 1 < ny; ++r)
        for (int c = 0; c + 1 < nx; ++c)
            if (!std::isnan(z[static_cast<std::size_t>(static_cast<i64>(r) * nx + c)])) akl.push_back(0.0);
    const bool scalar = akl.size() == 1;  // R's sqrt of one value: sqrtsd
    auto S = [scalar](double v) { return lsm::r_sqrt(v, scalar); };
    const i64 m = static_cast<i64>(akl.size());
    // the blocks in order: prefix counts per row for the parallel pass
    std::vector<i64> start(static_cast<std::size_t>(ny > 0 ? ny : 1), 0);
    {
        i64 k = 0;
        for (int r = 0; r + 1 < ny; ++r) {
            start[static_cast<std::size_t>(r)] = k;
            for (int c = 0; c + 1 < nx; ++c)
                if (!std::isnan(z[static_cast<std::size_t>(static_cast<i64>(r) * nx + c)])) ++k;
        }
    }
#pragma omp parallel for schedule(static) if (m > 16384)
    for (int r = 0; r < ny - 1; ++r) {
        i64 k = start[static_cast<std::size_t>(r)];
        const double* row = z.data() + static_cast<i64>(r) * nx;
        const double* below = row + nx;
        for (int c = 0; c + 1 < nx; ++c) {
            const double a = row[c];
            if (std::isnan(a)) continue;
            const double ypl = below[c], xpl = row[c + 1], xy = below[c + 1];
            const double s1 = S(dy2 + (ypl - a) * (ypl - a));
            const double s2 = S(dx2 + (xpl - a) * (xpl - a));
            const double s3 = S(dy2 + (xy - xpl) * (xy - xpl));
            const double s4 = S(dx2 + (xy - ypl) * (xy - ypl));
            const double akl1 = (0.5 * (s1 * s2)) + (0.5 * (s3 * s4));
            const double akl2 = (0.5 * (s1 * s4)) + (0.5 * (s3 * s2));
            akl[static_cast<std::size_t>(k++)] = 0.5 * (akl1 + akl2);
        }
    }
    return lsm::sum_of(0, m, [&](i64 i) { return akl[static_cast<std::size_t>(i)]; }, true);
}

// geodiv's cell spacing: x 1, y the ratio of the resolutions
// (res / mean(deltax, deltay), where mean's second argument is `trim`).
void geodiv_spacing(const Grid& g, double& dx, double& dy) {
    const double rx = (g.xmax - g.xmin) / g.nx, ry = (g.ymax - g.ymin) / g.ny;
    dx = rx / rx;
    dy = ry / rx;
}

}  // namespace

double sdq(const Grid& g) {
    if (g.ny < 2 || g.nx < 2) return NaN;
    const i64 m = static_cast<i64>(g.ny - 1) * (g.nx - 1);
    const int nx = g.nx;
    const double* z = g.z;
    return std::sqrt(lsm::sum_of(0, m, [&](i64 k) {
               const i64 r = k / (nx - 1), c = k % (nx - 1), i = r * nx + c;
               const double ex = z[i] - z[i + 1], ey = z[i] - z[i + nx];
               return (ex * ex) + (ey * ey);
           }, true) / static_cast<double>(m));
}

double sdq6(const Grid& g, bool iso) {
    const int ny = g.ny, nx = g.nx;
    const double* z = g.z;
    const int r0 = 3, r1 = ny - 3, c0 = 3, c1 = nx - 3;  // rows / columns [r0, r1)
    const i64 w = c1 > c0 ? c1 - c0 : 0, h = r1 > r0 ? r1 - r0 : 0;
    const double k = 1.0 / (60.0 * 1.0);
    auto at = [&](i64 r, i64 c) { return z[r * nx + c]; };
    auto px = [&](i64 j) {
        const i64 r = r0 + j / w, c = c0 + j % w;
        double t = at(r, c + 3) - (9 * at(r, c + 2));
        t = t + (45 * at(r, c + 1));
        t = t - (45 * at(r, c - 1));
        t = iso ? t + (9 * at(r, c - 2)) : t - (9 * at(r, c + 2));
        t = t - at(r, c - 3);
        const double p = k * t;
        return p * p;
    };
    auto py = [&](i64 j) {
        const i64 r = r0 + j / w, c = c0 + j % w;
        double t = at(r + 3, c) - (9 * at(r + 2, c));
        t = t + (45 * at(r + 1, c));
        t = t - (45 * at(r - 1, c));
        t = iso ? t + (9 * at(r - 2, c)) : t - (9 * at(r + 2, c));
        t = t - at(r - 3, c);
        const double p = k * t;
        return p * p;
    };
    const i64 m = w * h;
    const double sx = m ? lsm::sum_of(0, m, px, true) : 0.0;
    const double sy = m ? lsm::sum_of(0, m, py, true) : 0.0;
    // geodiv divides by length(x), which is 1 for a SpatRaster; ISO by the
    // number of cells the slopes are taken at
    const double n = iso ? static_cast<double>(m) : 1.0;
    return std::sqrt((1.0 / n) * sum2(sx, sy));
}

double surface_area(const Grid& g, bool iso) {
    if (iso) {
        const std::vector<double> z(g.z, g.z + g.size());
        return triangulated_area(z, g.ny, g.nx, (g.xmax - g.xmin) / g.nx, (g.ymax - g.ymin) / g.ny);
    }
    double dx, dy;
    geodiv_spacing(g, dx, dy);
    return triangulated_area(rescaled(g), g.ny, g.nx, dx, dy);
}

double flat_area(const Grid& g, bool iso) {
    double dx, dy;
    if (iso) {
        dx = (g.xmax - g.xmin) / g.nx;
        dy = (g.ymax - g.ymin) / g.ny;
    } else {
        geodiv_spacing(g, dx, dy);
    }
    // triangulated_area of the heights set to 0: the same area for every
    // complete block (R's vector sqrt, or sqrtsd for a single block)
    const int ny = g.ny, nx = g.nx;
    i64 blocks = 0;
    for (int r = 0; r + 1 < ny; ++r)
        for (int c = 0; c + 1 < nx; ++c) blocks += !std::isnan(g.z[static_cast<i64>(r) * nx + c]);
    const bool scalar = blocks == 1;
    const double sy = lsm::r_sqrt(dy * dy + 0.0 * 0.0, scalar), sx = lsm::r_sqrt(dx * dx + 0.0 * 0.0, scalar);
    const double akl1 = (0.5 * (sy * sx)) + (0.5 * (sy * sx));
    const double akl = 0.5 * (akl1 + akl1);
    auto complete = [&](i64 i) {
        return !std::isnan(g.z[i + 1]) && !std::isnan(g.z[i + nx]) && !std::isnan(g.z[i + nx + 1]);
    };
    // the blocks in R's order (top-left value present), NaN where another is missing
    const i64 m = static_cast<i64>(ny > 1 ? ny - 1 : 0) * (nx > 1 ? nx - 1 : 0);
    return lsm::sum_of(0, m, [&](i64 k) {
        const i64 r = k / (nx - 1), c = k % (nx - 1), i = r * nx + c;
        return std::isnan(g.z[i]) || !complete(i) ? NaN : akl;
    }, true);
}

double sdr(const Grid& g, bool iso) {
    const double flat = flat_area(g, iso);
    const double area = surface_area(g, iso);
    return ((area - flat) / flat) * 100;
}

}  // namespace surf
