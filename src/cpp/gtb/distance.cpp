// SPDX-License-Identifier: GPL-3.0-or-later
#include "distance.hpp"

#include <algorithm>
#include <cmath>

#include "primitives.hpp"

namespace gtb {

EuclideanDistanceResultCpp compute_euclidean_distance(const ImageU8& im) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 ext_nx = nx + 2;
    const i64 ext_ny = ny + 2;

    // Build extended arrays with replicated borders (matching GTB's exact padding)
    ImageU8 ext_fg(ext_nx, ext_ny, 0);
    ImageU8 ext_bg(ext_nx, ext_ny, 0);

    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const u8 v = im.at(x, y);
            if (v == 2) ext_fg.at(x + 1, y + 1) = 1;
            if (v == 1) ext_bg.at(x + 1, y + 1) = 1;
        }
    }

    // Border replication
    for (i64 y = 0; y < ny; ++y) {
        ext_fg.at(0, y + 1) = ext_fg.at(1, y + 1);
        ext_fg.at(ext_nx - 1, y + 1) = ext_fg.at(nx, y + 1);
        ext_bg.at(0, y + 1) = ext_bg.at(1, y + 1);
        ext_bg.at(ext_nx - 1, y + 1) = ext_bg.at(nx, y + 1);
    }
    for (i64 x = 0; x < ext_nx; ++x) {
        ext_fg.at(x, 0) = ext_fg.at(x, 1);
        ext_fg.at(x, ext_ny - 1) = ext_fg.at(x, ny);
        ext_bg.at(x, 0) = ext_bg.at(x, 1);
        ext_bg.at(x, ext_ny - 1) = ext_bg.at(x, ny);
    }

    // Compute squared Euclidean distance transforms
    const std::vector<std::uint32_t> sq_fg = sqedt(ext_fg);
    const std::vector<std::uint32_t> sq_bg = sqedt(ext_bg);

    EuclideanDistanceResultCpp res;
    res.signed_distance.resize(static_cast<std::size_t>(nx * ny), 0.0f);
    res.byte_map = ImageU8(nx, ny, 0);

    double sum_fg_dist = 0.0;
    double max_fg_dist = 0.0;
    i64 fg_count = 0;

    double sum_bg_dist = 0.0;
    double max_bg_dist = 0.0;
    i64 bg_count = 0;

    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const i64 idx = y * nx + x;
            const i64 ext_idx = (y + 1) * ext_nx + (x + 1);
            const u8 orig = im.at(x, y);

            if (orig == 2) {
                const double d = std::sqrt(static_cast<double>(sq_fg[static_cast<std::size_t>(ext_idx)]));
                res.signed_distance[static_cast<std::size_t>(idx)] = static_cast<float>(d);

                const i64 d_round = static_cast<i64>(std::round(d));
                const u8 fo = static_cast<u8>(std::min<i64>(d_round, 120)) + 130;
                res.byte_map[idx] = (fo > 130) ? fo : 0;

                sum_fg_dist += d;
                if (d > max_fg_dist) max_fg_dist = d;
                fg_count += 1;
            } else if (orig == 1) {
                const double d = std::sqrt(static_cast<double>(sq_bg[static_cast<std::size_t>(ext_idx)]));
                res.signed_distance[static_cast<std::size_t>(idx)] = -static_cast<float>(d);

                const i64 d_round = static_cast<i64>(std::round(d));
                const u8 bo = 130 - static_cast<u8>(std::min<i64>(d_round, 120));
                res.byte_map[idx] = (bo < 130) ? bo : 0;

                sum_bg_dist += d;
                if (d > max_bg_dist) max_bg_dist = d;
                bg_count += 1;
            } else {
                res.signed_distance[static_cast<std::size_t>(idx)] = 0.0f;
                res.byte_map[idx] = 0;
            }
        }
    }

    res.stats.fg_pixel_count = fg_count;
    res.stats.bg_pixel_count = bg_count;
    res.stats.mean_fg_distance = (fg_count > 0) ? (sum_fg_dist / static_cast<double>(fg_count)) : 0.0;
    res.stats.max_fg_distance = max_fg_dist;
    res.stats.mean_bg_distance = (bg_count > 0) ? (sum_bg_dist / static_cast<double>(bg_count)) : 0.0;
    res.stats.max_bg_distance = max_bg_dist;

    return res;
}

}  // namespace gtb
