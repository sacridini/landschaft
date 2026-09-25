// SPDX-License-Identifier: GPL-3.0-or-later
#include "mosaic.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gtb {

// Exact recode lookup table from GTB recodelm103.sav
const u8 kRecode103To19[256] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 4, 4, 4, 4, 4, 46, 47, 48, 49, 50, 5, 5, 5, 5, 5, 56, 57, 58, 59,
    60, 6, 6, 6, 6, 6, 66, 67, 68, 69, 70, 7, 7, 7, 7, 7, 76, 77, 78, 79,
    80, 8, 8, 8, 8, 8, 86, 87, 88, 89, 90, 9, 9, 9, 9, 9, 96, 97, 98, 99,
    100, 10, 10, 10, 10, 105, 106, 107, 108, 109, 110, 11, 11, 11, 11, 115, 116, 117, 118, 119,
    120, 12, 12, 12, 12, 125, 126, 127, 128, 129, 130, 13, 13, 13, 13, 13, 136, 137, 138, 139,
    140, 14, 14, 14, 14, 14, 146, 147, 148, 149, 150, 15, 15, 15, 15, 15, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 17, 3, 3, 173, 174, 175, 176, 177, 178, 179,
    18, 1, 1, 183, 184, 185, 186, 187, 188, 189, 19, 2, 2, 193, 194, 195, 196, 197, 198, 199,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};

namespace {

struct IntegralImage2D {
    i64 nx = 0;
    i64 ny = 0;
    std::vector<std::uint64_t> sat;

    IntegralImage2D(i64 nx_, i64 ny_)
        : nx(nx_), ny(ny_), sat(static_cast<std::size_t>((nx_ + 1) * (ny_ + 1)), 0) {}

    inline std::uint64_t at(i64 x, i64 y) const {
        return sat[static_cast<std::size_t>(y * (nx + 1) + x)];
    }

    inline void set(i64 x, i64 y, std::uint64_t val) {
        sat[static_cast<std::size_t>(y * (nx + 1) + x)] = val;
    }

    // Sum over [x0, x1) x [y0, y1)
    inline std::uint64_t rect_sum(i64 x0, i64 y0, i64 x1, i64 y1) const {
        if (x0 >= x1 || y0 >= y1) return 0;
        return at(x1, y1) - at(x0, y1) - at(x1, y0) + at(x0, y0);
    }
};

// spatcon.c rule 7: float proportions compared with double constants
inline u8 classify_lm_103(float p_agr, float p_for, float p_dev) {
    constexpr double kEps = FLT_EPSILON;  // spatcon.c 1.3.4: const float EPSILON = FLT_EPSILON

    // Check corners (pure 100% classes)
    if (std::fabs(p_for - 1.0) < kEps) return 170;
    if (std::fabs(p_dev - 1.0) < kEps) return 190;
    if (std::fabs(p_agr - 1.0) < kEps) return 180;

    if (p_for < 0.1) {
        if (p_agr < 0.1) {
            return (p_dev >= 0.9) ? 191 : 192;
        }
        if (p_agr < 0.2) {
            return (p_dev >= 0.8) ? 71 : 72;
        }
        if (p_agr < 0.3) {
            return (p_dev >= 0.7) ? 73 : 74;
        }
        if (p_agr < 0.4) {
            return (p_dev >= 0.6) ? 75 : 131;
        }
        if (p_agr < 0.5) {
            return (p_dev >= 0.5) ? 132 : 133;
        }
        if (p_agr < 0.6) {
            return (p_dev >= 0.4) ? 134 : 135;
        }
        if (p_agr < 0.7) {
            return (p_dev >= 0.3) ? 45 : 44;
        }
        if (p_agr < 0.8) {
            return (p_dev >= 0.2) ? 43 : 42;
        }
        if (p_agr < 0.9) {
            return (p_dev >= 0.1) ? 41 : 182;
        }
        return 181;
    }

    if (p_for < 0.2) {
        if (p_agr < 0.1) {
            return (p_dev >= 0.8) ? 61 : 62;
        }
        if (p_agr < 0.2) {
            return (p_dev >= 0.7) ? 111 : 112;
        }
        if (p_agr < 0.3) {
            return (p_dev >= 0.6) ? 114 : 200;
        }
        if (p_agr < 0.4) {
            return (p_dev >= 0.5) ? 201 : 202;
        }
        if (p_agr < 0.5) {
            return (p_dev >= 0.4) ? 203 : 204;
        }
        if (p_agr < 0.6) {
            return (p_dev >= 0.3) ? 205 : 206;
        }
        if (p_agr < 0.7) {
            return (p_dev >= 0.2) ? 103 : 102;
        }
        if (p_agr < 0.8) {
            return (p_dev >= 0.1) ? 101 : 52;
        }
        return 51;
    }

    if (p_for < 0.3) {
        if (p_agr < 0.1) {
            return (p_dev >= 0.7) ? 63 : 64;
        }
        if (p_agr < 0.2) {
            return (p_dev >= 0.6) ? 113 : 222;
        }
        if (p_agr < 0.3) {
            return (p_dev >= 0.5) ? 223 : 224;
        }
        if (p_agr < 0.4) {
            return (p_dev >= 0.4) ? 225 : 226;
        }
        if (p_agr < 0.5) {
            return (p_dev >= 0.3) ? 227 : 228;
        }
        if (p_agr < 0.6) {
            return (p_dev >= 0.2) ? 207 : 208;
        }
        if (p_agr < 0.7) {
            return (p_dev >= 0.1) ? 104 : 54;
        }
        return 53;
    }

    if (p_for < 0.4) {
        if (p_agr < 0.1) {
            return (p_dev >= 0.6) ? 65 : 155;
        }
        if (p_agr < 0.2) {
            return (p_dev >= 0.5) ? 221 : 220;
        }
        if (p_agr < 0.3) {
            return (p_dev >= 0.4) ? 235 : 234;
        }
        if (p_agr < 0.4) {
            return (p_dev >= 0.3) ? 236 : 230;
        }
        if (p_agr < 0.5) {
            return (p_dev >= 0.2) ? 229 : 210;
        }
        if (p_agr < 0.6) {
            return (p_dev >= 0.1) ? 209 : 141;
        }
        return 55;
    }

    if (p_for < 0.5) {
        if (p_agr < 0.1) {
            return (p_dev >= 0.5) ? 154 : 153;
        }
        if (p_agr < 0.2) {
            return (p_dev >= 0.4) ? 219 : 218;
        }
        if (p_agr < 0.3) {
            return (p_dev >= 0.3) ? 233 : 232;
        }
        if (p_agr < 0.4) {
            return (p_dev >= 0.2) ? 231 : 212;
        }
        if (p_agr < 0.5) {
            return (p_dev >= 0.1) ? 211 : 143;
        }
        return 142;
    }

    if (p_for < 0.6) {
        if (p_agr < 0.1) {
            return (p_dev >= 0.4) ? 152 : 151;
        }
        if (p_agr < 0.2) {
            return (p_dev >= 0.3) ? 217 : 216;
        }
        if (p_agr < 0.3) {
            return (p_dev >= 0.2) ? 215 : 214;
        }
        if (p_agr < 0.4) {
            return (p_dev >= 0.1) ? 213 : 145;
        }
        return 144;
    }

    if (p_for < 0.7) {
        if (p_agr < 0.1) {
            return (p_dev >= 0.3) ? 95 : 94;
        }
        if (p_agr < 0.2) {
            return (p_dev >= 0.2) ? 124 : 122;
        }
        if (p_agr < 0.3) {
            return (p_dev >= 0.1) ? 123 : 84;
        }
        return 85;
    }

    if (p_for < 0.8) {
        if (p_agr < 0.1) {
            return (p_dev >= 0.2) ? 93 : 92;
        }
        if (p_agr < 0.2) {
            return (p_dev >= 0.1) ? 121 : 82;
        }
        return 83;
    }

    if (p_for < 0.9) {
        if (p_agr < 0.1) {
            return (p_dev >= 0.1) ? 91 : 172;
        }
        return 81;
    }

    return 171;
}

}  // namespace

ImageU8 compute_landscape_mosaic(
    const ImageU8& im,
    int window_size,
    int output_classes
) {
    if (window_size < 3 || window_size % 2 == 0) {
        throw std::invalid_argument("window_size must be an odd integer >= 3");
    }
    if (output_classes != 19 && output_classes != 103) {
        throw std::invalid_argument("output_classes must be 19 or 103");
    }

    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 b = (window_size - 1) / 2;

    const ImageU8 pad = add_frame(im, b, static_cast<u8>(0));
    const i64 pnx = pad.nx;
    const i64 pny = pad.ny;

    IntegralImage2D sat_agr(pnx, pny);
    IntegralImage2D sat_for(pnx, pny);
    IntegralImage2D sat_dev(pnx, pny);

    for (i64 y = 0; y < pny; ++y) {
        std::uint64_t row_agr = 0;
        std::uint64_t row_for = 0;
        std::uint64_t row_dev = 0;
        for (i64 x = 0; x < pnx; ++x) {
            const u8 v = pad.at(x, y);
            if (v == 1) row_agr += 1;
            else if (v == 2) row_for += 1;
            else if (v == 3) row_dev += 1;

            sat_agr.set(x + 1, y + 1, sat_agr.at(x + 1, y) + row_agr);
            sat_for.set(x + 1, y + 1, sat_for.at(x + 1, y) + row_for);
            sat_dev.set(x + 1, y + 1, sat_dev.at(x + 1, y) + row_dev);
        }
    }

    ImageU8 out(nx, ny);

    #pragma omp parallel for schedule(static)
    for (i64 y = 0; y < ny; ++y) {
        const i64 y0 = y;
        const i64 y1 = y + window_size;

        for (i64 x = 0; x < nx; ++x) {
            // Missing pixels in original input stay missing (0)
            if (im.at(x, y) == 0) {
                out.at(x, y) = 0;
                continue;
            }

            const i64 x0 = x;
            const i64 x1 = x + window_size;

            const std::uint64_t n_agr = sat_agr.rect_sum(x0, y0, x1, y1);
            const std::uint64_t n_for = sat_for.rect_sum(x0, y0, x1, y1);
            const std::uint64_t n_dev = sat_dev.rect_sum(x0, y0, x1, y1);

            const std::uint64_t valid = n_agr + n_for + n_dev;
            u8 code103 = 0;

            if (valid > 0) {
                const float temp_float = static_cast<float>(1.0 / (1.0 * static_cast<double>(valid)));
                const float p_agr = static_cast<float>(1.0 * static_cast<double>(n_agr) * temp_float);
                const float p_for = static_cast<float>(1.0 * static_cast<double>(n_for) * temp_float);
                const float p_dev = static_cast<float>(1.0 * static_cast<double>(n_dev) * temp_float);

                code103 = classify_lm_103(p_agr, p_for, p_dev);
            }

            if (output_classes == 19) {
                out.at(x, y) = kRecode103To19[code103];
            } else {
                out.at(x, y) = code103;
            }
        }
    }

    return out;
}

LandscapeMosaicStats compute_landscape_mosaic_stats(
    const ImageU8& im103
) {
    std::vector<std::uint64_t> counts(256, 0);
    const i64 total_px = im103.size();

    for (i64 i = 0; i < total_px; ++i) {
        counts[im103[i]]++;
    }

    // Total non-missing pixels
    std::uint64_t valid_total = 0;
    for (int c = 1; c < 256; ++c) {
        valid_total += counts[static_cast<std::size_t>(c)];
    }

    LandscapeMosaicStats stats;
    for (int c = 1; c < 256; ++c) {
        if (counts[static_cast<std::size_t>(c)] > 0) {
            stats.class_codes.push_back(c);
            stats.pixel_counts.push_back(counts[static_cast<std::size_t>(c)]);
            const double pct = (valid_total > 0)
                ? (100.0 * static_cast<double>(counts[static_cast<std::size_t>(c)]) / static_cast<double>(valid_total))
                : 0.0;
            stats.percentages.push_back(pct);
        }
    }

    return stats;
}

}  // namespace gtb
