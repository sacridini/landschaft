// SPDX-License-Identifier: GPL-3.0-or-later
//
// Surface metrics in a moving window: each cell gets the metrics of the
// size x size window around it (NA beyond the edges; with `circle`, NA beyond
// size / 2 cells from the centre), computed by the same functions as the
// whole-raster metrics, on a raster whose extent is the window's. Rows of
// cells run in parallel, each thread with its own window.
//
// Where the window's value is the same whatever the window holds, no window
// is built: sv and sph take the running minimum / maximum along rows, then
// along columns, and sds counts the raster's peaks with a summed-area table
// (the same values).

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "surf/surface.hpp"

namespace surf {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
using Index = std::size_t;

bool needs(const std::vector<int>& ids, std::initializer_list<int> any) {
    for (int m : ids)
        for (int a : any)
            if (m == a) return true;
    return false;
}

// The minimum or maximum over every run of `w` values of v (NaN skipped;
// NaN where a run has no value), centred: out[i] covers v[i - h .. i + h]
// with the values beyond the ends missing.
template <class Better>
void running_extreme(const double* v, int n, int w, double* out, Better better, double worst) {
    const int h = w / 2;
    for (int i = 0; i < n; ++i) {
        double best = worst;
        bool any = false;
        const int a = std::max(0, i - h), b = std::min(n - 1, i + h);
        for (int k = a; k <= b; ++k) {
            const double x = v[k];
            if (std::isnan(x)) continue;
            if (!any || better(x, best)) best = x;
            any = true;
        }
        out[i] = any ? best : NaN;
    }
}

}  // namespace

std::vector<std::vector<double>> window_metrics(const Grid& g, bool lonlat, int size, bool circle,
                                                const std::vector<int>& ids) {
    const int ny = g.ny, nx = g.nx, h = size / 2, w = 2 * h + 1;
    const Index n = static_cast<Index>(g.size());
    std::vector<std::vector<double>> out(ids.size(), std::vector<double>(n, NaN));
    const double xres = (g.xmax - g.xmin) / nx, yres = (g.ymax - g.ymin) / ny;

    // the window mask: circle of radius h (in cells)
    std::vector<char> inside(static_cast<Index>(w) * w, 1);
    if (circle)
        for (int dr = -h; dr <= h; ++dr)
            for (int dc = -h; dc <= h; ++dc)
                inside[static_cast<Index>(dr + h) * w + (dc + h)] = dr * dr + dc * dc <= h * h;

    // shortcuts on square windows: extremes and summit counts
    std::vector<int> general;
    std::vector<Index> general_slot;
    for (Index k = 0; k < ids.size(); ++k) {
        const int m = ids[k];
        if (!circle && (m == kWSv || m == kWSph)) {
            // running min / max along rows, then along columns
            const bool is_max = m == kWSph;
            std::vector<double> rows(n), col(static_cast<Index>(ny)), res(static_cast<Index>(ny));
            auto better = [is_max](double a, double b) { return is_max ? a > b : a < b; };
#pragma omp parallel for schedule(static)
            for (int r = 0; r < ny; ++r)
                running_extreme(g.z + static_cast<i64>(r) * nx, nx, w, rows.data() + static_cast<Index>(r) * nx, better, 0.0);
#pragma omp parallel for schedule(static)
            for (int c = 0; c < nx; ++c) {
                std::vector<double> cv(static_cast<Index>(ny)), rv(static_cast<Index>(ny));
                for (int r = 0; r < ny; ++r) cv[static_cast<Index>(r)] = rows[static_cast<Index>(r) * nx + c];
                running_extreme(cv.data(), ny, w, rv.data(), better, 0.0);
                for (int r = 0; r < ny; ++r) out[k][static_cast<Index>(r) * nx + c] = std::fabs(rv[static_cast<Index>(r)]);
            }
            // abs(max(z)) / abs(min(z)) of a window without values: Inf, as R
            for (Index i = 0; i < n; ++i)
                if (std::isnan(out[k][i])) out[k][i] = std::numeric_limits<double>::infinity();
            continue;
        }
        if (!circle && m == kWSds) {
            // peaks of the raster; in a window, those off its border
            const std::vector<i64> peaks = find_peaks(g);
            const Index W = static_cast<Index>(nx) + 1;
            std::vector<i64> sat(W * (static_cast<Index>(ny) + 1), 0), mark(n, 0);
            for (i64 p : peaks) mark[static_cast<Index>(p)] = 1;
            for (int r = 0; r < ny; ++r)
                for (int c = 0; c < nx; ++c)
                    sat[(r + 1) * W + c + 1] = sat[r * W + c + 1] + sat[(r + 1) * W + c] - sat[r * W + c] +
                                               mark[static_cast<Index>(r) * nx + c];
            const double denom = (static_cast<double>(w) - 1) * (static_cast<double>(w) - 1);
#pragma omp parallel for schedule(static)
            for (int r = 0; r < ny; ++r)
                for (int c = 0; c < nx; ++c) {
                    // interior of the window: rows r-h+1 .. r+h-1, clipped to the raster
                    const int r0 = std::max(0, r - h + 1), r1 = std::min(ny, r + h);
                    const int c0 = std::max(0, c - h + 1), c1 = std::min(nx, c + h);
                    const i64 cnt = r1 > r0 && c1 > c0 ? sat[r1 * W + c1] - sat[r0 * W + c1] - sat[r1 * W + c0] + sat[r0 * W + c0]
                                                       : 0;
                    out[k][static_cast<Index>(r) * nx + c] = static_cast<double>(cnt) / denom;
                }
            continue;
        }
        general.push_back(m);
        general_slot.push_back(k);
    }
    if (general.empty()) return out;

    unsigned parts = 0;
    for (int m : general) {
        if (m == kWSa) parts |= kSa;
        if (m == kWSq || m == kWSbi) parts |= kSq;
        if (m == kWSsk) parts |= kSsk;
        if (m == kWSku) parts |= kSku;
        if (m == kWSv || m == kWSph) parts |= kExtremes;
        if (m == kWSmean) parts |= kSmean;
    }
    const bool hybrid = needs(general, {kWSdq, kWSdq6, kWSdr});
    const bool features = needs(general, {kWSds, kWSsc, kWS10z});
    const bool bearing = needs(general, {kWSbi, kWSci, kWSvi, kWSk, kWSpk, kWSvk});
    const bool lengths = needs(general, {kWSal, kWStr});
    const bool direction = needs(general, {kWStd, kWStdi});
    const bool wavelength = needs(general, {kWSrw, kWSrwi, kWShw});
    const bool fractal = needs(general, {kWSfd});

#pragma omp parallel
    {
        std::vector<double> buf(static_cast<Index>(w) * w);
#pragma omp for schedule(dynamic, 1)
        for (int r = 0; r < ny; ++r) {
            for (int c = 0; c < nx; ++c) {
                for (int dr = -h; dr <= h; ++dr)
                    for (int dc = -h; dc <= h; ++dc) {
                        const int rr = r + dr, cc = c + dc;
                        const Index b = static_cast<Index>(dr + h) * w + (dc + h);
                        buf[b] = rr >= 0 && rr < ny && cc >= 0 && cc < nx && inside[b]
                                     ? g.z[static_cast<i64>(rr) * nx + cc]
                                     : NaN;
                    }
                Grid wg{};
                wg.z = buf.data();
                wg.ny = w;
                wg.nx = w;
                wg.xmin = g.xmin + (c - h) * xres;
                wg.xmax = wg.xmin + w * xres;
                wg.ymax = g.ymax - (r - h) * yres;
                wg.ymin = wg.ymax - w * yres;
                const Index cell = static_cast<Index>(r) * nx + c;
                Amplitude a{};
                if (parts) a = amplitude(buf.data(), static_cast<i64>(buf.size()), parts);
                double sdq_v = NaN, sdq6_v = NaN, sdr_v = NaN;
                if (hybrid) {
                    sdq_v = sdq(wg);
                    sdq6_v = sdq6(wg);
                    const double flat = flat_area(wg), area = surface_area(wg);
                    sdr_v = ((area - flat) / flat) * 100;
                }
                double sds_v = NaN, ssc_v = NaN, s10z_v = NaN;
                if (features) {
                    sds_v = sds(wg);
                    ssc_v = ssc(wg);
                    s10z_v = s10z(wg);
                }
                BearingParameters bp{NaN, NaN, NaN, NaN, NaN, NaN};
                if (bearing) {
                    const Bearing bc = bearing_curve(buf.data(), static_cast<i64>(buf.size()));
                    if (!bc.s.empty()) bp = bearing_parameters(bc, a.sq);
                }
                std::vector<double> len, dir, wav;
                if (lengths) len = scl(wg, lonlat, {0.2, 1 / std::exp(1.0)});
                if (direction) dir = texture_direction(wg, lonlat);
                if (wavelength) wav = radial_wavelength(wg, lonlat);
                const double fd = fractal ? sfd(wg) : NaN;
                for (Index k = 0; k < general.size(); ++k) {
                    double v = NaN;
                    switch (general[k]) {
                        case kWSa: v = a.sa; break;
                        case kWSq: v = a.sq; break;
                        case kWSsk: v = a.ssk; break;
                        case kWSku: v = a.sku; break;
                        case kWSv: v = a.sv; break;
                        case kWSph: v = a.sph; break;
                        case kWSmean: v = a.smean; break;
                        case kWSdq: v = sdq_v; break;
                        case kWSdq6: v = sdq6_v; break;
                        case kWSdr: v = sdr_v; break;
                        case kWSds: v = sds_v; break;
                        case kWSsc: v = ssc_v; break;
                        case kWS10z: v = s10z_v; break;
                        case kWSbi: v = bp.sbi; break;
                        case kWSci: v = bp.sci; break;
                        case kWSvi: v = bp.svi; break;
                        case kWSk: v = bp.sk; break;
                        case kWSpk: v = bp.spk; break;
                        case kWSvk: v = bp.svk; break;
                        case kWSal: v = len[0]; break;
                        case kWStr: v = len[0] / len[2]; break;
                        case kWStd: v = dir[0]; break;
                        case kWStdi: v = dir[1]; break;
                        case kWSrw: v = wav[0]; break;
                        case kWSrwi: v = wav[1]; break;
                        case kWShw: v = wav[2]; break;
                        case kWSfd: v = fd; break;
                        default: break;
                    }
                    out[general_slot[k]][cell] = v;
                }
            }
        }
    }
    return out;
}

}  // namespace surf
