// SPDX-License-Identifier: GPL-3.0-or-later
#include "fad.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gtb {

namespace {

struct IntegralImage2D {
    i64 nx = 0;
    i64 ny = 0;
    std::vector<std::uint64_t> sat;

    IntegralImage2D(i64 nx_, i64 ny_) : nx(nx_), ny(ny_), sat(static_cast<std::size_t>((nx_ + 1) * (ny_ + 1)), 0) {}

    inline std::uint64_t at(i64 x, i64 y) const {
        return sat[static_cast<std::size_t>(y * (nx + 1) + x)];
    }

    inline void set(i64 x, i64 y, std::uint64_t val) {
        sat[static_cast<std::size_t>(y * (nx + 1) + x)] = val;
    }

    inline std::uint64_t rect_sum(i64 x0, i64 y0, i64 x1, i64 y1) const {
        return at(x1, y1) - at(x0, y1) - at(x1, y0) + at(x0, y0);
    }
};

void build_integral_images(
    const ImageU8& im,
    IntegralImage2D& sat_valid,
    IntegralImage2D& sat_fg
) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;

    for (i64 y = 0; y < ny; ++y) {
        std::uint64_t row_valid = 0;
        std::uint64_t row_fg = 0;
        for (i64 x = 0; x < nx; ++x) {
            const u8 v = im.at(x, y);
            if (v != 0 && v != 4) row_valid += 1;  // GTB sets 4 to missing for spatcon
            if (v == 2) row_fg += 1;
            sat_valid.set(x + 1, y + 1, sat_valid.at(x + 1, y) + row_valid);
            sat_fg.set(x + 1, y + 1, sat_fg.at(x + 1, y) + row_fg);
        }
    }
}

// spatcon rule 81 (area density of code 2, missing not included) followed by
// GTB's rescaling to percent: spatcon computes the proportion in float and
// maps it to 1 + (int)(p * 254.) (255 only for p = 1); GTB then takes
// byte(round(((b - 1) / 254.0 < 0.994999) * 100)), 255 -> 100.
inline u8 fad_percent(std::uint64_t fg, std::uint64_t non_missing) {
    if (non_missing == 0) return 0;
    const float p = static_cast<float>(fg) / static_cast<float>(non_missing);
    const int spatcon_byte = static_cast<int>(static_cast<double>(p) * 254.) + 1;
    if (spatcon_byte >= 255) return 100;
    float val = static_cast<float>(spatcon_byte - 1) / 254.0f;
    val = std::max(0.0f, std::min(val, 0.994999f));
    return static_cast<u8>(std::round(val * 100.0f));
}

// GTB codes of the non-foreground pixels in FAD maps (0 = foreground).
inline u8 fad_code(u8 orig) {
    switch (orig) {
        case 0: return 102;  // missing
        case 1: return 101;  // background
        case 3: return 105;  // special background
        case 4: return 106;  // special background, non-fragmenting
        default: return 0;
    }
}

}  // namespace

ImageU8 compute_fad_density(
    const ImageU8& im,
    int window_size,
    bool apply_gtb_mask
) {
    if (window_size < 3 || window_size % 2 == 0) {
        throw std::invalid_argument("window_size must be an odd integer >= 3");
    }

    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 pad = (window_size - 1) / 2;

    IntegralImage2D sat_valid(nx, ny);
    IntegralImage2D sat_fg(nx, ny);
    build_integral_images(im, sat_valid, sat_fg);

    ImageU8 out(nx, ny);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (nx * ny >= kParallelThreshold)
#endif
    for (i64 y = 0; y < ny; ++y) {
        const i64 y0 = std::max<i64>(0, y - pad);
        const i64 y1 = std::min<i64>(ny, y + pad + 1);

        for (i64 x = 0; x < nx; ++x) {
            const i64 x0 = std::max<i64>(0, x - pad);
            const i64 x1 = std::min<i64>(nx, x + pad + 1);

            const std::uint64_t non_missing = sat_valid.rect_sum(x0, y0, x1, y1);
            const std::uint64_t fg = sat_fg.rect_sum(x0, y0, x1, y1);

            const u8 byte_val = fad_percent(fg, non_missing);
            const u8 code = apply_gtb_mask ? fad_code(im.at(x, y)) : 0;
            out.at(x, y) = code ? code : byte_val;
        }
    }

    return out;
}

FADMultiscaleResultCpp compute_fad_multiscale(
    const ImageU8& im,
    const std::vector<int>& window_sizes
) {
    if (window_sizes.empty()) {
        throw std::invalid_argument("window_sizes cannot be empty");
    }

    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const std::size_t num_scales = window_sizes.size();

    IntegralImage2D sat_valid(nx, ny);
    IntegralImage2D sat_fg(nx, ny);
    build_integral_images(im, sat_valid, sat_fg);

    FADMultiscaleResultCpp res;
    res.window_sizes = window_sizes;
    res.single_scale_maps.resize(num_scales);
    res.raw_densities.resize(num_scales);
    res.stats.resize(num_scales + 1); // +1 for multiscale

    // Find foreground pixel count
    i64 fg_total = 0;
    for (i64 i = 0; i < nx * ny; ++i) {
        if (im[i] == 2) fg_total += 1;
    }

    std::vector<std::uint64_t> sum_densities(static_cast<std::size_t>(nx * ny), 0);

    for (std::size_t s = 0; s < num_scales; ++s) {
        const int w = window_sizes[s];
        const i64 pad = (w - 1) / 2;

        ImageU8 gtb_map(nx, ny);
        ImageU8 raw_map(nx, ny);

        i64 c_intact = 0;
        i64 c_interior = 0;
        i64 c_dominant = 0;
        i64 c_transitional = 0;
        i64 c_patchy = 0;
        i64 c_rare = 0;
        double sum_fad_fg = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (nx * ny >= kParallelThreshold) reduction(+:c_intact, c_interior, c_dominant, c_transitional, c_patchy, c_rare, sum_fad_fg)
#endif
        for (i64 y = 0; y < ny; ++y) {
            const i64 y0 = std::max<i64>(0, y - pad);
            const i64 y1 = std::min<i64>(ny, y + pad + 1);

            for (i64 x = 0; x < nx; ++x) {
                const i64 x0 = std::max<i64>(0, x - pad);
                const i64 x1 = std::min<i64>(nx, x + pad + 1);

                const std::uint64_t non_missing = sat_valid.rect_sum(x0, y0, x1, y1);
                const std::uint64_t fg = sat_fg.rect_sum(x0, y0, x1, y1);

                const u8 byte_val = fad_percent(fg, non_missing);
                raw_map.at(x, y) = byte_val;
                sum_densities[static_cast<std::size_t>(y * nx + x)] += byte_val;

                const u8 code = fad_code(im.at(x, y));
                if (code) {
                    gtb_map.at(x, y) = code;
                } else {
                    gtb_map.at(x, y) = byte_val;
                    sum_fad_fg += byte_val;

                    if (byte_val == 100) c_intact += 1;
                    else if (byte_val >= 90) c_interior += 1;
                    else if (byte_val >= 60) c_dominant += 1;
                    else if (byte_val >= 40) c_transitional += 1;
                    else if (byte_val >= 10) c_patchy += 1;
                    else c_rare += 1;
                }
            }
        }

        res.single_scale_maps[s] = std::move(gtb_map);
        res.raw_densities[s] = std::move(raw_map);

        auto& st = res.stats[s];
        st.scale = w;
        if (fg_total > 0) {
            const double inv = 100.0 / static_cast<double>(fg_total);
            st.intact_pct = c_intact * inv;
            st.interior_pct = c_interior * inv;
            st.dominant_pct = c_dominant * inv;
            st.transitional_pct = c_transitional * inv;
            st.patchy_pct = c_patchy * inv;
            st.rare_pct = c_rare * inv;
            st.average_fad = sum_fad_fg / static_cast<double>(fg_total);
        }
    }

    // Multiscale average map
    ImageU8 multi_map(nx, ny);
    i64 m_intact = 0;
    i64 m_interior = 0;
    i64 m_dominant = 0;
    i64 m_transitional = 0;
    i64 m_patchy = 0;
    i64 m_rare = 0;
    double m_sum_fad_fg = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (nx * ny >= kParallelThreshold) reduction(+:m_intact, m_interior, m_dominant, m_transitional, m_patchy, m_rare, m_sum_fad_fg)
#endif
    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const i64 idx = y * nx + x;
            const u8 code = fad_code(im.at(x, y));

            if (code) {
                multi_map.at(x, y) = code;
            } else {
                // IDL: byte(round(sum / 5.0)) in single precision
                const float avg = static_cast<float>(sum_densities[static_cast<std::size_t>(idx)]) / static_cast<float>(num_scales);
                const u8 byte_val = static_cast<u8>(std::round(avg));
                multi_map.at(x, y) = byte_val;
                m_sum_fad_fg += byte_val;

                if (byte_val == 100) m_intact += 1;
                else if (byte_val >= 90) m_interior += 1;
                else if (byte_val >= 60) m_dominant += 1;
                else if (byte_val >= 40) m_transitional += 1;
                else if (byte_val >= 10) m_patchy += 1;
                else m_rare += 1;
            }
        }
    }

    res.multiscale_map = std::move(multi_map);

    auto& mst = res.stats[num_scales];
    mst.scale = 0; // 0 indicates multiscale average
    if (fg_total > 0) {
        const double inv = 100.0 / static_cast<double>(fg_total);
        mst.intact_pct = m_intact * inv;
        mst.interior_pct = m_interior * inv;
        mst.dominant_pct = m_dominant * inv;
        mst.transitional_pct = m_transitional * inv;
        mst.patchy_pct = m_patchy * inv;
        mst.rare_pct = m_rare * inv;
        mst.average_fad = m_sum_fad_fg / static_cast<double>(fg_total);
    }

    return res;
}

}  // namespace gtb
