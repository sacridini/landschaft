// SPDX-License-Identifier: GPL-3.0-or-later
//
// The pixels of a view on a web map, looked up in a grid (ls.plot's detail as a map is zoomed in):
// where every pixel comes from, from source coordinates known on a coarse grid of the pixels and
// linear between them (as GDAL's approximate transformer, but closer: every few pixels), and the
// lookup of an image's pixels through it.

#include <cmath>
#include <cstdint>
#include <vector>

#include "io/io.hpp"

namespace lsio {

void view_index(const double* X, const double* Y, const i64* ci, i64 nci, const i64* cj, i64 ncj, i64 W, i64 H,
                const double* gt, i64 w, i64 h, std::int32_t* out) {
    // for each column of the view: its interval of the coarse grid and the weight of its right end
    std::vector<i64> kc(static_cast<std::size_t>(W));
    std::vector<double> sc(static_cast<std::size_t>(W));
    i64 k = 0;
    for (i64 i = 0; i < W; ++i) {
        while (k + 1 < nci - 1 && ci[k + 1] <= i) ++k;
        kc[static_cast<std::size_t>(i)] = k;
        sc[static_cast<std::size_t>(i)] = nci > 1 ? static_cast<double>(i - ci[k]) / static_cast<double>(ci[k + 1] - ci[k])
                                                  : 0.0;
    }
    const double c0 = gt[0], a = gt[1], f0 = gt[2], e = gt[3];
#pragma omp parallel for schedule(static)
    for (i64 j = 0; j < H; ++j) {
        i64 r0 = 0;
        while (r0 + 1 < ncj - 1 && cj[r0 + 1] <= j) ++r0;
        const i64 r1 = ncj > 1 ? r0 + 1 : r0;
        const double t = ncj > 1 ? static_cast<double>(j - cj[r0]) / static_cast<double>(cj[r1] - cj[r0]) : 0.0;
        const double* x0 = X + r0 * nci;
        const double* x1 = X + r1 * nci;
        const double* y0 = Y + r0 * nci;
        const double* y1 = Y + r1 * nci;
        std::int32_t* row = out + j * W;
        for (i64 i = 0; i < W; ++i) {
            const i64 q0 = kc[static_cast<std::size_t>(i)], q1 = nci > 1 ? q0 + 1 : q0;
            const double s = sc[static_cast<std::size_t>(i)];
            const double x = (1 - t) * ((1 - s) * x0[q0] + s * x0[q1]) + t * ((1 - s) * x1[q0] + s * x1[q1]);
            const double y = (1 - t) * ((1 - s) * y0[q0] + s * y0[q1]) + t * ((1 - s) * y1[q0] + s * y1[q1]);
            const double c = std::floor((x - c0) / a), r = std::floor((y - f0) / e);
            row[i] = (c >= 0 && c < static_cast<double>(w) && r >= 0 && r < static_cast<double>(h))
                         ? static_cast<std::int32_t>(static_cast<i64>(r) * w + static_cast<i64>(c))
                         : -1;
        }
    }
}

void gather_pixels(const std::uint8_t* src, i64 channels, const std::int32_t* index, i64 n, std::uint8_t* out) {
#pragma omp parallel for schedule(static)
    for (i64 p = 0; p < n; ++p) {
        const std::int32_t k = index[p];
        for (i64 b = 0; b < channels; ++b) out[p * channels + b] = k < 0 ? 0 : src[static_cast<i64>(k) * channels + b];
    }
}

}  // namespace lsio
