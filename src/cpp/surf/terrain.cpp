// SPDX-License-Identifier: GPL-3.0-or-later
//
// Terrain indices of a DEM (the R package spatialEco's tpi, tri, curvature,
// hsp, on terra's terrain and focal), corrected where spatialEco errs:
// - slope and aspect as terra::terrain (Horn's 3 x 3 or 4 neighbours; on
//   longitude/latitude rasters the cell width in metres per row, GeographicLib
//   geodesics); atan2 as mingw-w64's x87 libm, atan as the UCRT's;
// - focal statistics over a window mask, NA where any masked cell is NA or
//   beyond the edge (terra's focal with na.rm = FALSE), each window summed in
//   row order (terra slides running sums, whose rounding depends on its
//   blocks);
// - curvature: Zevenbergen & Thorne's second derivatives over L^2 (spatialEco
//   divides by 2 L^2), NA at the edges (spatialEco pads with 0), no rounding,
//   the cell height for the y derivatives; McNab's mean difference over the 8
//   neighbours (spatialEco: / 9 - 1); Bolstad's with the cell width (spatialEco
//   fails);
// - hsp: z-scores (tpi - mean) / sd per scale (spatialEco: tpi - mean / sd).

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cost/geodesic.hpp"
#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"
#include "surf/surface.hpp"

namespace surf {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.141592653589793238462643383279502884;
using Index = std::size_t;

// atan as terra gets it on Windows (the UCRT's, not an x87 one).
inline double r_atan(double x) { return std::atan(x); }

inline double dmod(double x, double n) { return x - n * std::floor(x / n); }

}  // namespace

std::vector<double> terrain(const Grid& g, bool lonlat, int neighbors, bool degrees, bool aspect) {
    const int nr = g.ny, nc = g.nx;
    std::vector<double> out(static_cast<Index>(g.size()), NaN);
    if (nr < 3 || nc < 3) return out;
    const double xres = (g.xmax - g.xmin) / nc, yres = (g.ymax - g.ymin) / nr;
    double dy = yres;
    std::vector<double> ddx;
    if (lonlat) {
        const cost::Geodesic geod;
        dy = geod.distance(0, 0, 0, yres);
        ddx.resize(static_cast<Index>(nr));
        for (int r = 0; r < nr; ++r) {
            const double lat = g.ymax - ((r + 0.5) * yres);
            ddx[static_cast<Index>(r)] = geod.distance(-xres, lat, xres, lat) / 2;
        }
    }
    const double* d = g.z;
    const double twoPI = 2 * kPi, halfPI = kPi / 2;
#pragma omp parallel for schedule(static) if (g.size() > 65536)
    for (int row = 1; row < nr - 1; ++row) {
        double xw[6], yw[6];
        if (neighbors == 4) {
            const double xi[2] = {-1, 1}, yi[2] = {-1, 1};
            for (int k = 0; k < 2; ++k) {
                xw[k] = lonlat ? xi[k] / (-2 * ddx[static_cast<Index>(row)]) : xi[k] / (-2 * xres);
                yw[k] = yi[k] / (2 * dy);
            }
        } else {
            const double xi[6] = {-1, -2, -1, 1, 2, 1}, yi[6] = {-1, 1, -2, 2, -1, 1};
            for (int k = 0; k < 6; ++k) {
                xw[k] = lonlat ? xi[k] / (-8 * ddx[static_cast<Index>(row)]) : xi[k] / (-8 * xres);
                yw[k] = yi[k] / (8 * dy);
            }
        }
        for (int col = 1; col < nc - 1; ++col) {
            const i64 i = static_cast<i64>(row) * nc + col;
            double zx, zy;
            if (neighbors == 4) {
                zx = d[i - 1] * xw[0] + d[i + 1] * xw[1];
                zy = d[i - nc] * yw[0] + d[i + nc] * yw[1];
            } else {
                zx = d[i - 1 - nc] * xw[0] + d[i - 1] * xw[1] + d[i - 1 + nc] * xw[2] + d[i + 1 - nc] * xw[3] +
                     d[i + 1] * xw[4] + d[i + 1 + nc] * xw[5];
                zy = d[i - 1 - nc] * yw[0] + d[i - 1 + nc] * yw[1] + d[i - nc] * yw[2] + d[i + nc] * yw[3] +
                     d[i + 1 - nc] * yw[4] + d[i + 1 + nc] * yw[5];
            }
            double v;
            if (aspect) {
                const double a = lsm::r_atan2(zy, zx);
                v = (neighbors == 4 && lonlat) ? std::fmod(halfPI - a, twoPI) : dmod(halfPI - a, twoPI);
            } else {
                v = r_atan(std::sqrt(zy * zy + zx * zx));
            }
            out[static_cast<Index>(i)] = degrees ? v * (180 / kPi) : v;
        }
    }
    return out;
}

std::vector<double> focal(const Grid& g, int wr, int wc, const std::vector<char>& mask, FocalStat stat) {
    const int nr = g.ny, nc = g.nx, hr = wr / 2, hc = wc / 2;
    std::vector<double> out(static_cast<Index>(g.size()), NaN);
    i64 k = 0;
    for (char m : mask) k += m != 0;
#pragma omp parallel
    {
        std::vector<double> v(static_cast<Index>(k));
#pragma omp for schedule(static)
        for (int r = 0; r < nr; ++r)
            for (int c = 0; c < nc; ++c) {
                bool na = false;
                Index n = 0;
                for (int a = 0; a < wr && !na; ++a)
                    for (int b = 0; b < wc; ++b) {
                        if (!mask[static_cast<Index>(a) * wc + b]) continue;
                        const int rr = r - hr + a, cc = c - hc + b;
                        const double x = rr >= 0 && rr < nr && cc >= 0 && cc < nc ? g.z[static_cast<i64>(rr) * nc + cc] : NaN;
                        if (std::isnan(x)) {
                            na = true;
                            break;
                        }
                        v[n++] = x;
                    }
                if (na || n == 0) continue;
                double res;
                switch (stat) {
                    case kFocalMin: res = *std::min_element(v.begin(), v.end()); break;
                    case kFocalMax: res = *std::max_element(v.begin(), v.end()); break;
                    default: {
                        double s = 0.0;
                        for (double x : v) s = s + x;
                        if (stat == kFocalSum) {
                            res = s;
                        } else if (stat == kFocalMean) {
                            res = s / static_cast<double>(n);
                        } else {  // sample sd: two passes
                            const double m = s / static_cast<double>(n);
                            double ss = 0.0;
                            for (double x : v) ss = ss + (x - m) * (x - m);
                            res = lsm::r_sqrt(ss / (static_cast<double>(n) - 1));
                        }
                    }
                }
                out[static_cast<Index>(r) * nc + c] = res;
            }
    }
    return out;
}

std::vector<double> tri(const Grid& g, int wr, int wc) {
    // sqrt(sum((z - z_i)^2)) over the neighbours with a value (spatialEco's
    // exact tri: R's sum in 80 bits, the sqrt of one number)
    const int nr = g.ny, nc = g.nx, hr = wr / 2, hc = wc / 2;
    std::vector<double> out(static_cast<Index>(g.size()), NaN);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < nr; ++r)
        for (int c = 0; c < nc; ++c) {
            const double zc = g.z[static_cast<i64>(r) * nc + c];
            if (std::isnan(zc)) continue;
            lsm::Extended s(0.0);
            int n = 0;
            for (int a = 0; a < wr; ++a)
                for (int b = 0; b < wc; ++b) {
                    if (a == hr && b == hc) continue;
                    const int rr = r - hr + a, cc = c - hc + b;
                    if (rr < 0 || rr >= nr || cc < 0 || cc >= nc) continue;
                    const double x = g.z[static_cast<i64>(rr) * nc + cc];
                    if (std::isnan(x)) continue;
                    const double dv = zc - x;
                    s += dv * dv;
                    ++n;
                }
            if (n) out[static_cast<Index>(r) * nc + c] = std::sqrt(static_cast<double>(s));
        }
    return out;
}

std::vector<double> curvature(const Grid& g, int type) {
    const int nr = g.ny, nc = g.nx;
    const double lx = (g.xmax - g.xmin) / nc, ly = (g.ymax - g.ymin) / nr;
    std::vector<double> out(static_cast<Index>(g.size()), NaN);
    const double* z = g.z;
    if (type == kCurvBolstad) {
        std::vector<char> mask(9, 1);
        const std::vector<double> mean = focal(g, 3, 3, mask, kFocalMean);
        for (i64 i = 0; i < g.size(); ++i)
            out[static_cast<Index>(i)] = 10000 * (((z[i] - mean[static_cast<Index>(i)]) / 1000) / (lx + lx / 2));
        return out;
    }
    if (type == kCurvMcnab) {
        // mean of (z - z_i) over the 8 neighbours with a value, / the largest
        double hi = -std::numeric_limits<double>::infinity();
        for (int r = 0; r < nr; ++r)
            for (int c = 0; c < nc; ++c) {
                const i64 i = static_cast<i64>(r) * nc + c;
                if (std::isnan(z[i])) continue;
                lsm::Extended s(0.0);
                for (int a = -1; a <= 1; ++a)
                    for (int b = -1; b <= 1; ++b) {
                        if (a == 0 && b == 0) continue;
                        const int rr = r + a, cc = c + b;
                        if (rr < 0 || rr >= nr || cc < 0 || cc >= nc) continue;
                        const double x = z[static_cast<i64>(rr) * nc + cc];
                        if (!std::isnan(x)) s += z[i] - x;
                    }
                const double v = static_cast<double>(s) / 8;
                out[static_cast<Index>(i)] = v;
                hi = std::max(hi, v);
            }
        for (double& v : out) v = v / hi;
        return out;
    }
    const bool scalar = static_cast<i64>(nr - 2) * (nc - 2) == 1;
#pragma omp parallel for schedule(static) if (g.size() > 65536)
    for (int r = 1; r < nr - 1; ++r)
        for (int c = 1; c < nc - 1; ++c) {
            auto m = [&](int k) {  // 1..9, row order around (r, c)
                const int dr = (k - 1) / 3 - 1, dc = (k - 1) % 3 - 1;
                return z[static_cast<i64>(r + dr) * nc + (c + dc)];
            };
            const double p = (m(6) - m(4)) / (2 * lx);
            const double q = (m(2) - m(8)) / (2 * ly);
            const double rr = ((m(4) + m(6)) - (2 * m(5))) / (lx * lx);
            const double s = (((m(3) + m(7)) - m(1)) - m(9)) / ((4 * lx) * ly);
            const double t = ((m(2) + m(8)) - (2 * m(5))) / (ly * ly);
            const double p2 = p * p, q2 = q * q;
            const double root = lsm::r_sqrt((1 + p2) + q2, scalar);
            const double plan = -(((q2 * rr) - (((2 * p) * q) * s)) + (p2 * t)) / ((p2 + q2) * root);
            const double prof = -(((p2 * rr) + (((2 * p) * q) * s)) + (q2 * t)) / ((p2 + q2) * lsm::r_pow(root, 3.0));
            double v = type == kCurvPlanform ? plan : type == kCurvProfile ? prof : plan + prof;
            out[static_cast<Index>(r) * nc + c] = v;
        }
    return out;
}

std::vector<double> tpi(const Grid& g, int wr, int wc, const std::vector<char>& mask, bool normalize) {
    const std::vector<double> mean = focal(g, wr, wc, mask, kFocalMean);
    std::vector<double> out(static_cast<Index>(g.size()));
    for (i64 i = 0; i < g.size(); ++i) out[static_cast<Index>(i)] = g.z[i] - mean[static_cast<Index>(i)];
    if (normalize) {
        const std::vector<double> sd = focal(g, wr, wc, mask, kFocalSd);
        for (Index i = 0; i < out.size(); ++i) out[i] = out[i] / sd[i];
    }
    return out;
}

std::vector<double> hsp(const Grid& g, const std::vector<int>& rows, const std::vector<int>& cols,
                        const std::vector<std::vector<char>>& masks, bool normalize) {
    // sum over the scales (largest first) of 100 z-scores of the tpi
    std::vector<double> acc(static_cast<Index>(g.size()), 0.0);
    bool first = true;
    for (Index k = rows.size(); k-- > 0;) {
        std::vector<double> t = tpi(g, rows[k], cols[k], masks[k], false);
        std::vector<double> v;
        v.reserve(t.size());
        for (double x : t)
            if (!std::isnan(x)) v.push_back(x);
        const i64 n = static_cast<i64>(v.size());
        const double m = lsm::r_mean(v.data(), n), s = lsm::r_sd(v.data(), n, m);
        for (Index i = 0; i < t.size(); ++i) {
            const double z = 100 * ((t[i] - m) / s);
            acc[i] = first ? z : acc[i] + z;
        }
        first = false;
    }
    if (normalize) {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        for (double x : acc)
            if (!std::isnan(x)) {
                lo = std::min(lo, x);
                hi = std::max(hi, x);
            }
        for (double& x : acc) x = (x - lo) / (hi - lo);
    }
    return acc;
}

}  // namespace surf
