// SPDX-License-Identifier: GPL-3.0-or-later
#include "contagion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gtb {

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

inline u8 gtb_rescale_byte(u8 spat_val, u8 orig_val, bool apply_mask) {
    u8 out_val = 0;
    if (spat_val == 255) {
        out_val = 100;
    } else if (spat_val == 0) {
        out_val = 0;
    } else {
        double temp = (static_cast<double>(spat_val) - 1.0) / 254.0;
        if (temp > 0.994999) temp = 0.994999;
        if (temp < 0.0) temp = 0.0;
        out_val = static_cast<u8>(std::round(temp * 100.0));
    }

    if (apply_mask) {
        if (orig_val == 1) {
            out_val = 101; // background
        } else if (orig_val == 0) {
            out_val = 102; // missing
        }
    }
    return out_val;
}

}  // namespace

ImageU8 compute_contagion_pff(
    const ImageU8& im,
    int window_size,
    bool apply_gtb_mask
) {
    if (window_size < 3 || window_size % 2 == 0) {
        throw std::invalid_argument("window_size must be an odd integer >= 3");
    }

    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 b = (window_size - 1) / 2;

    // Pad image with 0 (missing value)
    const ImageU8 pad = add_frame(im, b, static_cast<u8>(0));
    const i64 pnx = pad.nx;
    const i64 pny = pad.ny;

    // Integral images for horizontal edges
    // Size is pnx x pny
    IntegralImage2D sat_h_eff(pnx, pny);
    IntegralImage2D sat_h_ef_star(pnx, pny);

    for (i64 y = 0; y < pny; ++y) {
        std::uint64_t row_h_eff = 0;
        std::uint64_t row_h_ef_star = 0;
        for (i64 x = 0; x < pnx - 1; ++x) {
            const u8 v1 = pad.at(x, y);
            const u8 v2 = pad.at(x + 1, y);
            if (v1 == 2 && v2 == 2) {
                row_h_eff += 1;
            }
            if ((v1 == 2 && v2 >= 1) || (v1 >= 1 && v2 == 2)) {
                row_h_ef_star += 1;
            }
            sat_h_eff.set(x + 1, y + 1, sat_h_eff.at(x + 1, y) + row_h_eff);
            sat_h_ef_star.set(x + 1, y + 1, sat_h_ef_star.at(x + 1, y) + row_h_ef_star);
        }
        // Last column of horizontal edges has no edge starting at pnx - 1
        sat_h_eff.set(pnx, y + 1, sat_h_eff.at(pnx, y) + row_h_eff);
        sat_h_ef_star.set(pnx, y + 1, sat_h_ef_star.at(pnx, y) + row_h_ef_star);
    }

    // Integral images for vertical edges
    IntegralImage2D sat_v_eff(pnx, pny);
    IntegralImage2D sat_v_ef_star(pnx, pny);

    for (i64 y = 0; y < pny; ++y) {
        std::uint64_t row_v_eff = 0;
        std::uint64_t row_v_ef_star = 0;
        if (y < pny - 1) {
            for (i64 x = 0; x < pnx; ++x) {
                const u8 v1 = pad.at(x, y);
                const u8 v2 = pad.at(x, y + 1);
                if (v1 == 2 && v2 == 2) {
                    row_v_eff += 1;
                }
                if ((v1 == 2 && v2 >= 1) || (v1 >= 1 && v2 == 2)) {
                    row_v_ef_star += 1;
                }
                sat_v_eff.set(x + 1, y + 1, sat_v_eff.at(x + 1, y) + row_v_eff);
                sat_v_ef_star.set(x + 1, y + 1, sat_v_ef_star.at(x + 1, y) + row_v_ef_star);
            }
        } else {
            // Last row has no vertical edge starting at pny - 1
            for (i64 x = 0; x < pnx; ++x) {
                sat_v_eff.set(x + 1, pny, sat_v_eff.at(x + 1, pny - 1));
                sat_v_ef_star.set(x + 1, pny, sat_v_ef_star.at(x + 1, pny - 1));
            }
        }
    }

    ImageU8 out(nx, ny);

    #pragma omp parallel for schedule(static)
    for (i64 y = 0; y < ny; ++y) {
        const i64 y0 = y;
        const i64 y1 = y + window_size;

        for (i64 x = 0; x < nx; ++x) {
            const i64 x0 = x;
            const i64 x1 = x + window_size;

            // Horizontal edges in window: x in [x0, x1 - 1), y in [y0, y1)
            const std::uint64_t n_h_eff = sat_h_eff.rect_sum(x0, y0, x1 - 1, y1);
            const std::uint64_t n_h_ef_star = sat_h_ef_star.rect_sum(x0, y0, x1 - 1, y1);

            // Vertical edges in window: x in [x0, x1), y in [y0, y1 - 1)
            const std::uint64_t n_v_eff = sat_v_eff.rect_sum(x0, y0, x1, y1 - 1);
            const std::uint64_t n_v_ef_star = sat_v_ef_star.rect_sum(x0, y0, x1, y1 - 1);

            const std::uint64_t eff = n_h_eff + n_v_eff;
            const std::uint64_t ef_star = n_h_ef_star + n_v_ef_star;

            u8 spat_val = 0;
            if (ef_star > 0) {
                const float p = (1.0f * static_cast<float>(eff)) / (1.0f * static_cast<float>(ef_star));
                const auto temp_int = static_cast<long int>(static_cast<double>(p) * 254.0) + 1;
                spat_val = static_cast<u8>(temp_int);
            }

            out.at(x, y) = gtb_rescale_byte(spat_val, im.at(x, y), apply_gtb_mask);
        }
    }

    return out;
}

Image<float> compute_contagion_pff_float(
    const ImageU8& im,
    int window_size,
    bool apply_gtb_mask
) {
    if (window_size < 3 || window_size % 2 == 0) {
        throw std::invalid_argument("window_size must be an odd integer >= 3");
    }

    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 b = (window_size - 1) / 2;

    const ImageU8 pad = add_frame(im, b, static_cast<u8>(0));
    const i64 pnx = pad.nx;
    const i64 pny = pad.ny;

    IntegralImage2D sat_h_eff(pnx, pny);
    IntegralImage2D sat_h_ef_star(pnx, pny);

    for (i64 y = 0; y < pny; ++y) {
        std::uint64_t row_h_eff = 0;
        std::uint64_t row_h_ef_star = 0;
        for (i64 x = 0; x < pnx - 1; ++x) {
            const u8 v1 = pad.at(x, y);
            const u8 v2 = pad.at(x + 1, y);
            if (v1 == 2 && v2 == 2) row_h_eff += 1;
            if ((v1 == 2 && v2 >= 1) || (v1 >= 1 && v2 == 2)) row_h_ef_star += 1;
            sat_h_eff.set(x + 1, y + 1, sat_h_eff.at(x + 1, y) + row_h_eff);
            sat_h_ef_star.set(x + 1, y + 1, sat_h_ef_star.at(x + 1, y) + row_h_ef_star);
        }
        sat_h_eff.set(pnx, y + 1, sat_h_eff.at(pnx, y) + row_h_eff);
        sat_h_ef_star.set(pnx, y + 1, sat_h_ef_star.at(pnx, y) + row_h_ef_star);
    }

    IntegralImage2D sat_v_eff(pnx, pny);
    IntegralImage2D sat_v_ef_star(pnx, pny);

    for (i64 y = 0; y < pny; ++y) {
        std::uint64_t row_v_eff = 0;
        std::uint64_t row_v_ef_star = 0;
        if (y < pny - 1) {
            for (i64 x = 0; x < pnx; ++x) {
                const u8 v1 = pad.at(x, y);
                const u8 v2 = pad.at(x, y + 1);
                if (v1 == 2 && v2 == 2) row_v_eff += 1;
                if ((v1 == 2 && v2 >= 1) || (v1 >= 1 && v2 == 2)) row_v_ef_star += 1;
                sat_v_eff.set(x + 1, y + 1, sat_v_eff.at(x + 1, y) + row_v_eff);
                sat_v_ef_star.set(x + 1, y + 1, sat_v_ef_star.at(x + 1, y) + row_v_ef_star);
            }
        } else {
            for (i64 x = 0; x < pnx; ++x) {
                sat_v_eff.set(x + 1, pny, sat_v_eff.at(x + 1, pny - 1));
                sat_v_ef_star.set(x + 1, pny, sat_v_ef_star.at(x + 1, pny - 1));
            }
        }
    }

    Image<float> out(nx, ny);
    const float nan_val = std::numeric_limits<float>::quiet_NaN();

    #pragma omp parallel for schedule(static)
    for (i64 y = 0; y < ny; ++y) {
        const i64 y0 = y;
        const i64 y1 = y + window_size;

        for (i64 x = 0; x < nx; ++x) {
            const u8 orig = im.at(x, y);
            if (apply_gtb_mask && (orig == 0 || orig == 1)) {
                out.at(x, y) = nan_val;
                continue;
            }

            const i64 x0 = x;
            const i64 x1 = x + window_size;

            const std::uint64_t n_h_eff = sat_h_eff.rect_sum(x0, y0, x1 - 1, y1);
            const std::uint64_t n_h_ef_star = sat_h_ef_star.rect_sum(x0, y0, x1 - 1, y1);

            const std::uint64_t n_v_eff = sat_v_eff.rect_sum(x0, y0, x1, y1 - 1);
            const std::uint64_t n_v_ef_star = sat_v_ef_star.rect_sum(x0, y0, x1, y1 - 1);

            const std::uint64_t eff = n_h_eff + n_v_eff;
            const std::uint64_t ef_star = n_h_ef_star + n_v_ef_star;

            if (ef_star == 0) {
                out.at(x, y) = 0.0f;
            } else {
                out.at(x, y) = static_cast<float>(eff) / static_cast<float>(ef_star);
            }
        }
    }

    return out;
}

ImageU8 compute_clustering_fac(
    const ImageU8& im,
    int window_size,
    bool apply_gtb_mask
) {
    if (window_size < 3 || window_size % 2 == 0) {
        throw std::invalid_argument("window_size must be an odd integer >= 3");
    }

    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 b = (window_size - 1) / 2;

    const ImageU8 pad = add_frame(im, b, static_cast<u8>(0));
    const i64 pnx = pad.nx;
    const i64 pny = pad.ny;

    IntegralImage2D sat_h_eff(pnx, pny);
    IntegralImage2D sat_h_nm(pnx, pny);

    for (i64 y = 0; y < pny; ++y) {
        std::uint64_t row_h_eff = 0;
        std::uint64_t row_h_nm = 0;
        for (i64 x = 0; x < pnx - 1; ++x) {
            const u8 v1 = pad.at(x, y);
            const u8 v2 = pad.at(x + 1, y);
            if (v1 == 2 && v2 == 2) row_h_eff += 1;
            if (v1 >= 1 && v2 >= 1) row_h_nm += 1;
            sat_h_eff.set(x + 1, y + 1, sat_h_eff.at(x + 1, y) + row_h_eff);
            sat_h_nm.set(x + 1, y + 1, sat_h_nm.at(x + 1, y) + row_h_nm);
        }
        sat_h_eff.set(pnx, y + 1, sat_h_eff.at(pnx, y) + row_h_eff);
        sat_h_nm.set(pnx, y + 1, sat_h_nm.at(pnx, y) + row_h_nm);
    }

    IntegralImage2D sat_v_eff(pnx, pny);
    IntegralImage2D sat_v_nm(pnx, pny);

    for (i64 y = 0; y < pny; ++y) {
        std::uint64_t row_v_eff = 0;
        std::uint64_t row_v_nm = 0;
        if (y < pny - 1) {
            for (i64 x = 0; x < pnx; ++x) {
                const u8 v1 = pad.at(x, y);
                const u8 v2 = pad.at(x, y + 1);
                if (v1 == 2 && v2 == 2) row_v_eff += 1;
                if (v1 >= 1 && v2 >= 1) row_v_nm += 1;
                sat_v_eff.set(x + 1, y + 1, sat_v_eff.at(x + 1, y) + row_v_eff);
                sat_v_nm.set(x + 1, y + 1, sat_v_nm.at(x + 1, y) + row_v_nm);
            }
        } else {
            for (i64 x = 0; x < pnx; ++x) {
                sat_v_eff.set(x + 1, pny, sat_v_eff.at(x + 1, pny - 1));
                sat_v_nm.set(x + 1, pny, sat_v_nm.at(x + 1, pny - 1));
            }
        }
    }

    ImageU8 out(nx, ny);

    #pragma omp parallel for schedule(static)
    for (i64 y = 0; y < ny; ++y) {
        const i64 y0 = y;
        const i64 y1 = y + window_size;

        for (i64 x = 0; x < nx; ++x) {
            const i64 x0 = x;
            const i64 x1 = x + window_size;

            const std::uint64_t n_h_eff = sat_h_eff.rect_sum(x0, y0, x1 - 1, y1);
            const std::uint64_t n_h_nm = sat_h_nm.rect_sum(x0, y0, x1 - 1, y1);

            const std::uint64_t n_v_eff = sat_v_eff.rect_sum(x0, y0, x1, y1 - 1);
            const std::uint64_t n_v_nm = sat_v_nm.rect_sum(x0, y0, x1, y1 - 1);

            const std::uint64_t eff = n_h_eff + n_v_eff;
            const std::uint64_t nm = n_h_nm + n_v_nm;

            u8 spat_val = 0;
            if (nm > 0) {
                const float temp_float_3 = 1.0f / (1.0f * static_cast<float>(nm));
                const float p = static_cast<float>(eff) * temp_float_3;
                const auto temp_int = static_cast<long int>(static_cast<double>(p) * 254.0) + 1;
                spat_val = static_cast<u8>(temp_int);
            }

            out.at(x, y) = gtb_rescale_byte(spat_val, im.at(x, y), apply_gtb_mask);
        }
    }

    return out;
}

Image<float> compute_clustering_fac_float(
    const ImageU8& im,
    int window_size,
    bool apply_gtb_mask
) {
    if (window_size < 3 || window_size % 2 == 0) {
        throw std::invalid_argument("window_size must be an odd integer >= 3");
    }

    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 b = (window_size - 1) / 2;

    const ImageU8 pad = add_frame(im, b, static_cast<u8>(0));
    const i64 pnx = pad.nx;
    const i64 pny = pad.ny;

    IntegralImage2D sat_h_eff(pnx, pny);
    IntegralImage2D sat_h_nm(pnx, pny);

    for (i64 y = 0; y < pny; ++y) {
        std::uint64_t row_h_eff = 0;
        std::uint64_t row_h_nm = 0;
        for (i64 x = 0; x < pnx - 1; ++x) {
            const u8 v1 = pad.at(x, y);
            const u8 v2 = pad.at(x + 1, y);
            if (v1 == 2 && v2 == 2) row_h_eff += 1;
            if (v1 >= 1 && v2 >= 1) row_h_nm += 1;
            sat_h_eff.set(x + 1, y + 1, sat_h_eff.at(x + 1, y) + row_h_eff);
            sat_h_nm.set(x + 1, y + 1, sat_h_nm.at(x + 1, y) + row_h_nm);
        }
        sat_h_eff.set(pnx, y + 1, sat_h_eff.at(pnx, y) + row_h_eff);
        sat_h_nm.set(pnx, y + 1, sat_h_nm.at(pnx, y) + row_h_nm);
    }

    IntegralImage2D sat_v_eff(pnx, pny);
    IntegralImage2D sat_v_nm(pnx, pny);

    for (i64 y = 0; y < pny; ++y) {
        std::uint64_t row_v_eff = 0;
        std::uint64_t row_v_nm = 0;
        if (y < pny - 1) {
            for (i64 x = 0; x < pnx; ++x) {
                const u8 v1 = pad.at(x, y);
                const u8 v2 = pad.at(x, y + 1);
                if (v1 == 2 && v2 == 2) row_v_eff += 1;
                if (v1 >= 1 && v2 >= 1) row_v_nm += 1;
                sat_v_eff.set(x + 1, y + 1, sat_v_eff.at(x + 1, y) + row_v_eff);
                sat_v_nm.set(x + 1, y + 1, sat_v_nm.at(x + 1, y) + row_v_nm);
            }
        } else {
            for (i64 x = 0; x < pnx; ++x) {
                sat_v_eff.set(x + 1, pny, sat_v_eff.at(x + 1, pny - 1));
                sat_v_nm.set(x + 1, pny, sat_v_nm.at(x + 1, pny - 1));
            }
        }
    }

    Image<float> out(nx, ny);
    const float nan_val = std::numeric_limits<float>::quiet_NaN();

    #pragma omp parallel for schedule(static)
    for (i64 y = 0; y < ny; ++y) {
        const i64 y0 = y;
        const i64 y1 = y + window_size;

        for (i64 x = 0; x < nx; ++x) {
            const u8 orig = im.at(x, y);
            if (apply_gtb_mask && (orig == 0 || orig == 1)) {
                out.at(x, y) = nan_val;
                continue;
            }

            const i64 x0 = x;
            const i64 x1 = x + window_size;

            const std::uint64_t n_h_eff = sat_h_eff.rect_sum(x0, y0, x1 - 1, y1);
            const std::uint64_t n_h_nm = sat_h_nm.rect_sum(x0, y0, x1 - 1, y1);

            const std::uint64_t n_v_eff = sat_v_eff.rect_sum(x0, y0, x1, y1 - 1);
            const std::uint64_t n_v_nm = sat_v_nm.rect_sum(x0, y0, x1, y1 - 1);

            const std::uint64_t eff = n_h_eff + n_v_eff;
            const std::uint64_t nm = n_h_nm + n_v_nm;

            if (nm == 0) {
                out.at(x, y) = 0.0f;
            } else {
                out.at(x, y) = static_cast<float>(eff) / static_cast<float>(nm);
            }
        }
    }

    return out;
}

}  // namespace gtb
