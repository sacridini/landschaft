// SPDX-License-Identifier: GPL-3.0-or-later
//
// Port of the 'spa' event of guidostoolbox.pro (GuidosToolbox, (C) European
// Union (JRC), Peter Vogt, GPL-3.0-or-later).
#include "spa.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "primitives.hpp"

namespace gtb {

namespace {

// 3x3 binary erosion / dilation of a 0/1 image (IDL ERODE / DILATE with a
// 3x3 structuring element): erosion clears the pixels on the image edge,
// dilation ignores the pixels outside. Separable: rows, then columns.
ImageU8 morph3x3(const ImageU8& im, bool erode) {
    const i64 nx = im.nx, ny = im.ny;
    ImageU8 tmp(nx, ny), out(nx, ny);
    auto op = [erode](u8 a, u8 b) { return erode ? std::min(a, b) : std::max(a, b); };
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (im.size() >= kParallelThreshold)
#endif
    for (i64 y = 0; y < ny; ++y) {
        const u8* r = im.data() + y * nx;
        u8* t = tmp.data() + y * nx;
        for (i64 x = 0; x < nx; ++x) {
            const u8 l = x > 0 ? r[x - 1] : (erode ? u8{0} : r[x]);
            const u8 rr = x + 1 < nx ? r[x + 1] : (erode ? u8{0} : r[x]);
            t[x] = op(op(l, r[x]), rr);
        }
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (im.size() >= kParallelThreshold)
#endif
    for (i64 y = 0; y < ny; ++y) {
        const u8* c = tmp.data() + y * nx;
        const u8* up = y > 0 ? tmp.data() + (y - 1) * nx : nullptr;
        const u8* dn = y + 1 < ny ? tmp.data() + (y + 1) * nx : nullptr;
        u8* o = out.data() + y * nx;
        for (i64 x = 0; x < nx; ++x) {
            const u8 a = up ? up[x] : (erode ? u8{0} : c[x]);
            const u8 b = dn ? dn[x] : (erode ? u8{0} : c[x]);
            o[x] = op(op(a, c[x]), b);
        }
    }
    return out;
}

// max(label_region(mask eq value, all_neighbors = graph == 8)): the number of
// components, the image edge being left unlabelled as IDL does.
i64 count_regions(const ImageU8& im, u8 value, int graph) {
    ImageU8 m(im.nx, im.ny);
    for (i64 i = 0; i < im.size(); ++i) m[i] = im[i] == value;
    set_frame<u8>(m, 1, 0);
    Image<std::int32_t> labels;
    std::vector<i64> areas;
    return label_regions(m, 1, graph, false, labels, areas);
}

}  // namespace

SPAResult compute_spa(const ImageU8& im, int variant) {
    if (variant != 2 && variant != 3 && variant != 5 && variant != 6) throw std::invalid_argument("SPA variant must be 2, 3, 5 or 6");
    const i64 nx = im.nx, ny = im.ny, n = im.size();
    constexpr i64 kB = 2;
    SPAResult res;

    bool has_missing = false;
    for (i64 i = 0; i < n && !has_missing; ++i) has_missing = im[i] == 0;

    // ext1 = image in a frame of background, then count the FG objects
    ImageU8 ext = add_frame<u8>(im, kB, 1);
    const i64 ex = ext.nx, ey = ext.ny, en = ext.size();
    res.nr_fg = count_regions(ext, 2, 8);
    // replicate the image outline into the inner frame ring (edges and corners)
    for (i64 x = kB; x < ex - kB; ++x) {
        ext.at(x, 1) = ext.at(x, 2);
        ext.at(x, ey - 2) = ext.at(x, ey - 3);
    }
    for (i64 y = kB; y < ey - kB; ++y) {
        ext.at(1, y) = ext.at(2, y);
        ext.at(ex - 2, y) = ext.at(ex - 3, y);
    }
    ext.at(1, 1) = ext.at(2, 2);
    ext.at(1, ey - 2) = ext.at(2, ey - 3);
    ext.at(ex - 2, 1) = ext.at(ex - 3, 2);
    ext.at(ex - 2, ey - 2) = ext.at(ex - 3, ey - 3);

    ImageU8 fg(ex, ey);
    for (i64 i = 0; i < en; ++i) fg[i] = ext[i] == 2;
    if (has_missing) {  // extend the foreground into adjacent missing pixels (dilation of the inner image)
        ImageU8 inner = sub_frame(fg, kB);
        const ImageU8 d = morph3x3(inner, false);
        for (i64 y = 0; y < ny; ++y) {
            for (i64 x = 0; x < nx; ++x) {
                const i64 i = y * nx + x;
                if (im[i] == 0 && d[i]) fg.at(x + kB, y + kB) = 1;
            }
        }
    }

    if (variant == 2) {  // small linear features: FG minus its opening
        const ImageU8 open = morph3x3(morph3x3(fg, true), false);
        ImageU8 out(nx, ny);
        for (i64 y = 0; y < ny; ++y) {
            for (i64 x = 0; x < nx; ++x) {
                const u8 f = fg.at(x + kB, y + kB), o = open.at(x + kB, y + kB);
                out.at(x, y) = f ? (o ? 17 : 1) : 0;
                if (im.at(x, y) == 0) out.at(x, y) = 129;
            }
        }
        res.nr_cores = count_regions(out, 17, 8);
        res.nr_margin = count_regions(out, 1, 8);
        res.map = std::move(out);
        return res;
    }

    const ImageU8 cores = morph3x3(fg, true);
    // holes: 4-connected non-core regions other than the outer one (label 1)
    ImageU8 notcore(ex, ey);
    for (i64 i = 0; i < en; ++i) notcore[i] = !cores[i];
    set_frame<u8>(notcore, 1, 0);
    Image<std::int32_t> hl;
    std::vector<i64> ha;
    label_regions(notcore, 1, 4, false, hl, ha);

    ImageU8 map(ex, ey, 0);
    if (variant == 3) {
        for (i64 i = 0; i < en; ++i) {
            u8 v = cores[i] ? 17 : 0;
            if (hl[i] > 1) v = 100;
            if (fg[i] && !cores[i]) v = 1;  // margin
            map[i] = v;
        }
    } else {
        ImageU8 islets;
        if (variant == 6) {  // FG objects without core: FG minus its reconstruction from the cores
            ImageU8 rec = cores;
            reconstruct_dilation(rec, fg, 8);
            islets = ImageU8(ex, ey);
            for (i64 i = 0; i < en; ++i) islets[i] = fg[i] && !rec[i];
        }
        const ImageU8 dil = morph3x3(cores, /*erode=*/false);
        for (i64 i = 0; i < en; ++i) {
            const bool boundary = dil[i] && fg[i] && !cores[i];
            u8 v = cores[i] ? 17 : 0;
            if (hl[i] > 1) v = 100;
            if (boundary) v = v == 100 ? 5 : static_cast<u8>(v + 3);  // perforation (103 -> 5) or edge
            if (fg[i] && !cores[i] && !boundary) v = 1;  // margin
            if (variant == 6 && islets[i]) v = 9;
            map[i] = v;
        }
    }
    // statistics on the extended map: frame to background, missing back in
    set_frame<u8>(map, kB, 0);
    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            if (im.at(x, y) == 0) map.at(x + kB, y + kB) = 129;
        }
    }
    res.nr_holes = count_regions(map, 100, 4);
    res.nr_cores = count_regions(map, 17, 8);
    if (variant == 3) {
        res.nr_margin = count_regions(map, 1, 8);
    } else {
        res.nr_edge = count_regions(map, 3, 8);
        res.nr_perf = count_regions(map, 5, 8);
        res.nr_margin = count_regions(map, 1, 8);
        if (variant == 6) res.nr_islet = count_regions(map, 9, 8);
        ImageU8 contiguous(ex, ey);
        for (i64 i = 0; i < en; ++i) contiguous[i] = map[i] == 17 || map[i] == 3 || map[i] == 5;
        res.nr_contiguous = count_regions(contiguous, 1, 8);
    }
    res.map = sub_frame(map, kB);
    return res;
}

}  // namespace gtb
