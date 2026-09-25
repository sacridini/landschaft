// SPDX-License-Identifier: GPL-3.0-or-later
#include "entropy.hpp"

#include <algorithm>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gtb {

namespace {

const double ENT_MAX = 1.5;
const double AL2 = std::log(2.0);
const double NORM_FAC = 100.0 / (AL2 * ENT_MAX);

// 8 neighbor directions
const i64 DX[8] = {1, -1, 0, 0, 1, -1, 1, -1};
const i64 DY[8] = {0, 0, 1, -1, 1, -1, -1, 1};

ImageU8 make_extended_encoded(const ImageU8& im) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 ext_nx = nx + 2;
    const i64 ext_ny = ny + 2;

    ImageU8 ext(ext_nx, ext_ny, 0);

    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const u8 v = im.at(x, y);
            u8 code = 0;
            if (v == 2) code = 7;
            else if (v == 1) code = 3;
            ext.at(x + 1, y + 1) = code;
        }
    }

    // Border replication
    for (i64 y = 0; y < ny; ++y) {
        ext.at(0, y + 1) = ext.at(1, y + 1);
        ext.at(ext_nx - 1, y + 1) = ext.at(nx, y + 1);
    }
    for (i64 x = 0; x < ext_nx; ++x) {
        ext.at(x, 0) = ext.at(x, 1);
        ext.at(x, ext_ny - 1) = ext.at(x, ny);
    }

    return ext;
}

double entropy_from_hist(const std::uint64_t* counts, std::size_t size) {
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < size; ++i) {
        total += counts[i];
    }
    if (total == 0) return 0.0;

    double h = 0.0;
    const double inv_total = 1.0 / static_cast<double>(total);
    for (std::size_t i = 0; i < size; ++i) {
        if (counts[i] > 0) {
            const double p = counts[i] * inv_total;
            h -= p * std::log(p);
        }
    }

    double normalized = h * NORM_FAC;
    if (normalized > 100.0) normalized = 100.0;
    if (normalized < 0.0) normalized = 0.0;
    return normalized;
}

}  // namespace

SpatialEntropyResultCpp compute_spatial_entropy(const ImageU8& im) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const ImageU8 ext = make_extended_encoded(im);

    std::vector<std::uint64_t> counts(256, 0);

#if defined(_OPENMP)
    // Local histograms per thread to avoid contention
    const int max_th = omp_get_max_threads();
    std::vector<std::vector<std::uint64_t>> local_counts(max_th, std::vector<std::uint64_t>(256, 0));

#pragma omp parallel for schedule(static) if (nx * ny >= kParallelThreshold)
    for (i64 y = 0; y < ny; ++y) {
        const int tid = omp_get_thread_num();
        for (i64 x = 0; x < nx; ++x) {
            const u8 c0 = ext.at(x + 1, y + 1);
            for (int k = 0; k < 8; ++k) {
                const u8 c1 = ext.at(x + 1 + DX[k], y + 1 + DY[k]);
                const u8 diff = static_cast<u8>(c0 - c1);
                local_counts[tid][diff] += 1;
            }
        }
    }

    for (int t = 0; t < max_th; ++t) {
        for (int i = 0; i < 256; ++i) {
            counts[i] += local_counts[t][i];
        }
    }
#else
    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const u8 c0 = ext.at(x + 1, y + 1);
            for (int k = 0; k < 8; ++k) {
                const u8 c1 = ext.at(x + 1 + DX[k], y + 1 + DY[k]);
                const u8 diff = static_cast<u8>(c0 - c1);
                counts[diff] += 1;
            }
        }
    }
#endif

    SpatialEntropyResultCpp res;
    res.histogram = counts;
    res.entropy_index = entropy_from_hist(counts.data(), counts.size());
    return res;
}

Image<float> compute_spatial_entropy_map(const ImageU8& im, int window_size) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 pad = window_size / 2;
    const ImageU8 ext = make_extended_encoded(im);

    Image<float> out(nx, ny, 0.0f);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (nx * ny >= kParallelThreshold)
#endif
    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const i64 y0 = std::max<i64>(0, y - pad);
            const i64 y1 = std::min<i64>(ny, y + pad + 1);
            const i64 x0 = std::max<i64>(0, x - pad);
            const i64 x1 = std::min<i64>(nx, x + pad + 1);

            std::uint64_t counts[256] = {0};
            for (i64 wy = y0; wy < y1; ++wy) {
                for (i64 wx = x0; wx < x1; ++wx) {
                    const u8 c0 = ext.at(wx + 1, wy + 1);
                    for (int k = 0; k < 8; ++k) {
                        const u8 c1 = ext.at(wx + 1 + DX[k], wy + 1 + DY[k]);
                        const u8 diff = static_cast<u8>(c0 - c1);
                        counts[diff] += 1;
                    }
                }
            }

            out.at(x, y) = static_cast<float>(entropy_from_hist(counts, 256));
        }
    }

    return out;
}

}  // namespace gtb
