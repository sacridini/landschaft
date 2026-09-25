// SPDX-License-Identifier: GPL-3.0-or-later
#include "change.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

#include "primitives.hpp"

namespace gtb {

namespace {

// Perform flat 3x3 binary erosion on an 8-bit mask (0/1)
ImageU8 erode_3x3(const ImageU8& im) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    ImageU8 out(nx, ny, 0);

    for (i64 y = 1; y < ny - 1; ++y) {
        const i64 r_off = y * nx;
        for (i64 x = 1; x < nx - 1; ++x) {
            if (im.px[static_cast<std::size_t>(r_off + x)] == 0) continue;
            bool all_one = true;
            for (i64 dy = -1; dy <= 1 && all_one; ++dy) {
                const i64 n_off = (y + dy) * nx;
                for (i64 dx = -1; dx <= 1 && all_one; ++dx) {
                    if (im.px[static_cast<std::size_t>(n_off + (x + dx))] == 0) {
                        all_one = false;
                    }
                }
            }
            if (all_one) {
                out.px[static_cast<std::size_t>(r_off + x)] = 1;
            }
        }
    }
    return out;
}

// Perform flat 3x3 binary dilation on an 8-bit mask (0/1)
ImageU8 dilate_3x3(const ImageU8& im) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    ImageU8 out(nx, ny, 0);

    for (i64 y = 0; y < ny; ++y) {
        const i64 r_off = y * nx;
        for (i64 x = 0; x < nx; ++x) {
            if (im.px[static_cast<std::size_t>(r_off + x)] == 1) {
                for (i64 dy = -1; dy <= 1; ++dy) {
                    const i64 ny_pos = y + dy;
                    if (ny_pos < 0 || ny_pos >= ny) continue;
                    const i64 n_off = ny_pos * nx;
                    for (i64 dx = -1; dx <= 1; ++dx) {
                        const i64 nx_pos = x + dx;
                        if (nx_pos < 0 || nx_pos >= nx) continue;
                        out.px[static_cast<std::size_t>(n_off + nx_pos)] = 1;
                    }
                }
            }
        }
    }
    return out;
}

// Flat 3x3 binary closing: dilation followed by erosion
ImageU8 close_3x3(const ImageU8& im) {
    return erode_3x3(dilate_3x3(im));
}

// Reconstruct marker inside mask using 8-connected dilation
ImageU8 reconstruct_by_dilation(const ImageU8& marker, const ImageU8& mask) {
    const i64 nx = marker.nx;
    const i64 ny = marker.ny;
    const i64 npix = nx * ny;

    ImageU8 out(nx, ny, 0);
    std::queue<i64> q;

    for (i64 i = 0; i < npix; ++i) {
        if (marker.px[static_cast<std::size_t>(i)] > 0 && mask.px[static_cast<std::size_t>(i)] > 0) {
            out.px[static_cast<std::size_t>(i)] = 1;
            q.push(i);
        }
    }

    const i64 dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const i64 dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    while (!q.empty()) {
        const i64 curr = q.front();
        q.pop();

        const i64 cx = curr % nx;
        const i64 cy = curr / nx;

        for (int k = 0; k < 8; ++k) {
            const i64 nx_pos = cx + dx[k];
            const i64 ny_pos = cy + dy[k];
            if (nx_pos < 0 || nx_pos >= nx || ny_pos < 0 || ny_pos >= ny) continue;

            const i64 n_idx = ny_pos * nx + nx_pos;
            if (out.px[static_cast<std::size_t>(n_idx)] == 0 && mask.px[static_cast<std::size_t>(n_idx)] > 0) {
                out.px[static_cast<std::size_t>(n_idx)] = 1;
                q.push(n_idx);
            }
        }
    }
    return out;
}

int fos_to_class_idx(u8 fos) {
    if (fos < 10) return 0;       // Rare [0, 9]
    if (fos < 40) return 1;       // Patchy [10, 39]
    if (fos < 60) return 2;       // Transitional [40, 59]
    if (fos < 90) return 3;       // Dominant [60, 89]
    if (fos < 100) return 4;      // Interior [90, 99]
    return 5;                     // Intact [100]
}

}  // namespace

SimpleChangeResult compute_simple_change(
    const ImageU8& im1,
    const ImageU8& im2
) {
    if (im1.nx != im2.nx || im1.ny != im2.ny) {
        throw std::invalid_argument("im1 and im2 dimensions mismatch");
    }

    const i64 nx = im1.nx;
    const i64 ny = im1.ny;
    const i64 npix = nx * ny;

    SimpleChangeResult res;
    res.change_map = ImageU8(nx, ny, 255);

    for (i64 i = 0; i < npix; ++i) {
        const u8 v1 = im1.px[static_cast<std::size_t>(i)];
        const u8 v2 = im2.px[static_cast<std::size_t>(i)];

        if (v1 == 0 || v2 == 0) {
            res.change_map.px[static_cast<std::size_t>(i)] = 255;
            res.missing++;
        } else if (v1 == 1 && v2 == 1) {
            res.change_map.px[static_cast<std::size_t>(i)] = 11;
            res.stable_bg++;
            res.total_valid++;
        } else if (v1 == 1 && v2 == 2) {
            res.change_map.px[static_cast<std::size_t>(i)] = 12;
            res.gain++;
            res.total_valid++;
        } else if (v1 == 2 && v2 == 1) {
            res.change_map.px[static_cast<std::size_t>(i)] = 21;
            res.loss++;
            res.total_valid++;
        } else if (v1 == 2 && v2 == 2) {
            res.change_map.px[static_cast<std::size_t>(i)] = 22;
            res.stable_fg++;
            res.total_valid++;
        } else {
            res.change_map.px[static_cast<std::size_t>(i)] = 255;
            res.missing++;
        }
    }

    return res;
}

MorphologicalChangeResult compute_morphological_change(
    const ImageU8& im1,
    const ImageU8& im2
) {
    if (im1.nx != im2.nx || im1.ny != im2.ny) {
        throw std::invalid_argument("im1 and im2 dimensions mismatch");
    }

    const i64 nx = im1.nx;
    const i64 ny = im1.ny;
    const i64 npix = nx * ny;

    // Step 1: Binary FG and BG masks
    ImageU8 fg1(nx, ny, 0);
    ImageU8 bg1(nx, ny, 0);
    ImageU8 fg2(nx, ny, 0);
    ImageU8 bg2(nx, ny, 0);

    i64 fg_area_1 = 0;
    i64 fg_area_2 = 0;

    for (i64 i = 0; i < npix; ++i) {
        const u8 v1 = im1.px[static_cast<std::size_t>(i)];
        const u8 v2 = im2.px[static_cast<std::size_t>(i)];

        if (v1 == 2) { fg1.px[static_cast<std::size_t>(i)] = 1; ++fg_area_1; }
        else if (v1 == 1) { bg1.px[static_cast<std::size_t>(i)] = 1; }

        if (v2 == 2) { fg2.px[static_cast<std::size_t>(i)] = 1; ++fg_area_2; }
        else if (v2 == 1) { bg2.px[static_cast<std::size_t>(i)] = 1; }
    }

    // Core FG and BG (distance > 2, which corresponds to 2 successive 3x3 erosions)
    const ImageU8 core_fg1 = erode_3x3(erode_3x3(fg1));
    const ImageU8 core_bg1 = erode_3x3(erode_3x3(bg1));
    const ImageU8 core_fg2 = erode_3x3(erode_3x3(fg2));
    const ImageU8 core_bg2 = erode_3x3(erode_3x3(bg2));

    // Step 2: Seed masks for core-to-core transitions
    ImageU8 seed_loss(nx, ny, 0);
    ImageU8 seed_gain(nx, ny, 0);
    ImageU8 mask_loss(nx, ny, 0);
    ImageU8 mask_gain(nx, ny, 0);

    for (i64 i = 0; i < npix; ++i) {
        const u8 v1 = im1.px[static_cast<std::size_t>(i)];
        const u8 v2 = im2.px[static_cast<std::size_t>(i)];

        if (v1 == 2 && v2 == 1) mask_loss.px[static_cast<std::size_t>(i)] = 1;
        if (v1 == 1 && v2 == 2) mask_gain.px[static_cast<std::size_t>(i)] = 1;

        if (core_fg1.px[static_cast<std::size_t>(i)] == 1 && core_bg2.px[static_cast<std::size_t>(i)] == 1) {
            seed_loss.px[static_cast<std::size_t>(i)] = 1;
        }
        if (core_bg1.px[static_cast<std::size_t>(i)] == 1 && core_fg2.px[static_cast<std::size_t>(i)] == 1) {
            seed_gain.px[static_cast<std::size_t>(i)] = 1;
        }
    }

    // Step 3: Geodesic reconstruction of seeds inside change candidate masks
    const ImageU8 rec_loss = reconstruct_by_dilation(seed_loss, mask_loss);
    const ImageU8 rec_gain = reconstruct_by_dilation(seed_gain, mask_gain);

    // Step 4: Morphological closing (3x3)
    const ImageU8 cl_loss = close_3x3(rec_loss);
    const ImageU8 cl_gain = close_3x3(rec_gain);

    // Step 5: Final classification
    MorphologicalChangeResult res;
    res.change_map = ImageU8(nx, ny, 255);
    res.fg_area_t1 = fg_area_1;
    res.fg_area_t2 = fg_area_2;
    res.net_change = fg_area_2 - fg_area_1;

    for (i64 i = 0; i < npix; ++i) {
        const u8 v1 = im1.px[static_cast<std::size_t>(i)];
        const u8 v2 = im2.px[static_cast<std::size_t>(i)];

        if (v1 == 0 || v2 == 0) {
            res.change_map.px[static_cast<std::size_t>(i)] = 255;
            res.missing++;
            continue;
        }

        const bool has_gain = (cl_gain.px[static_cast<std::size_t>(i)] == 1);
        const bool has_loss = (cl_loss.px[static_cast<std::size_t>(i)] == 1);

        if (has_gain && has_loss) {
            // Resolution in GTB line 20786: conflict resolved to gain (12)
            res.change_map.px[static_cast<std::size_t>(i)] = 12;
            res.mcd_gain++;
        } else if (has_gain) {
            res.change_map.px[static_cast<std::size_t>(i)] = 12;
            res.mcd_gain++;
        } else if (has_loss) {
            res.change_map.px[static_cast<std::size_t>(i)] = 21;
            res.mcd_loss++;
        } else if (v1 == 1 && v2 == 1) {
            res.change_map.px[static_cast<std::size_t>(i)] = 11;
            res.stable_bg++;
        } else if (v1 == 2 && v2 == 2) {
            res.change_map.px[static_cast<std::size_t>(i)] = 22;
            res.stable_fg++;
        } else {
            // Edge chatter / spurious change
            res.change_map.px[static_cast<std::size_t>(i)] = 176;
            res.spurious++;
        }
    }

    // Step 6: Landscape Elasticity
    const ImageU8 interior1 = erode_3x3(fg1);
    const ImageU8 interior2 = erode_3x3(fg2);

    i64 fia = 0;
    i64 fib = 0;
    for (i64 i = 0; i < npix; ++i) {
        if (interior1.px[static_cast<std::size_t>(i)] > 0) ++fia;
        if (interior2.px[static_cast<std::size_t>(i)] > 0) ++fib;
    }

    double ptot = 0.0;
    if (fg_area_1 > 0) {
        ptot = (static_cast<double>(fg_area_2 - fg_area_1) / static_cast<double>(fg_area_1)) * 100.0;
    }
    double pitot = 0.0;
    if (fia > 0) {
        pitot = (static_cast<double>(fib - fia) / static_cast<double>(fia)) * 100.0;
    }

    if (std::abs(ptot) > 1e-6) {
        res.elasticity = pitot / ptot;
    } else {
        res.elasticity = 0.0;
    }

    return res;
}

FOSChangeResult compute_fos_change(
    const ImageU8& fos1,
    const ImageU8& fos2
) {
    if (fos1.nx != fos2.nx || fos1.ny != fos2.ny) {
        throw std::invalid_argument("fos1 and fos2 dimensions mismatch");
    }

    const i64 nx = fos1.nx;
    const i64 ny = fos1.ny;
    const i64 npix = nx * ny;

    FOSChangeResult res;
    res.change_map = ImageU8(nx, ny, 254);
    res.matrix_6x6 = std::vector<std::vector<i64>>(6, std::vector<i64>(6, 0));

    i64 above_diag = 0;
    i64 below_diag = 0;

    for (i64 i = 0; i < npix; ++i) {
        const u8 v1 = fos1.px[static_cast<std::size_t>(i)];
        const u8 v2 = fos2.px[static_cast<std::size_t>(i)];

        if (v1 == 102 || v2 == 102 || v1 == 255 || v2 == 255 || v1 == 0 || v2 == 0) {
            // Missing
            res.change_map.px[static_cast<std::size_t>(i)] = 254;
            continue;
        }

        if (v1 == 105 || v2 == 105) {
            // Special water class
            res.change_map.px[static_cast<std::size_t>(i)] = 253;
            continue;
        }

        if (v1 == 101 && v2 == 101) {
            // Stable Background
            res.change_map.px[static_cast<std::size_t>(i)] = 252;
            res.stable_bg++;
            continue;
        }

        if (v1 == 101 && v2 <= 100) {
            // Foreground Gain
            res.change_map.px[static_cast<std::size_t>(i)] = 250;
            res.fg_gain++;
            continue;
        }

        if (v1 <= 100 && v2 == 101) {
            // Foreground Loss
            res.change_map.px[static_cast<std::size_t>(i)] = 251;
            res.fg_loss++;
            continue;
        }

        if (v1 <= 100 && v2 <= 100) {
            // Common foreground at both times
            res.forcom++;
            // Delta FOS shifted by +100
            const int delta = static_cast<int>(v1) + 100 - static_cast<int>(v2);
            res.change_map.px[static_cast<std::size_t>(i)] = static_cast<u8>(std::clamp(delta, 0, 200));

            const int c1 = fos_to_class_idx(v1);
            const int c2 = fos_to_class_idx(v2);
            res.matrix_6x6[static_cast<std::size_t>(c1)][static_cast<std::size_t>(c2)]++;

            if (c1 == c2) {
                res.diag_same_class++;
            } else {
                res.diff_class++;
                if (c1 < c2) {
                    above_diag++; // Fragmentation decrease
                } else {
                    below_diag++; // Fragmentation increase
                }
            }
        }
    }

    if (res.diff_class > 0) {
        res.frag_decrease_pct = (static_cast<double>(above_diag) / static_cast<double>(res.diff_class)) * 100.0;
        res.frag_increase_pct = (static_cast<double>(below_diag) / static_cast<double>(res.diff_class)) * 100.0;
    }

    return res;
}

}  // namespace gtb
