// SPDX-License-Identifier: GPL-3.0-or-later
//
// NLMR's Voronoi models, nlm_mosaictess() and nlm_mosaicgibbs(): the germs
// (sf::st_sample's runif() draws, spatstat's rSSI()) and the values drawn in
// R's order, and the rasterisation of the Voronoi tessellation (the cell
// centre takes the value of its nearest germ, as terra::rasterize() of the
// Voronoi polygons takes the polygon containing it). The order of GEOS's
// Voronoi cells, which decides which value goes to which germ in NLMR, is
// found in Python (landschaft.nlm._tess).

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "lsm/parallel.hpp"
#include "nlm/models.hpp"

namespace nlm {

GermDraws tess_germs(RRng& rng, i64 germs, double width, double height) {
    // sf::st_sample(square, germs): runif(size, xmin, xmax), runif(size,
    // ymin, ymax); then runif(germs) for the values
    GermDraws g;
    g.x.resize(static_cast<std::size_t>(germs));
    g.y.resize(static_cast<std::size_t>(germs));
    g.v.resize(static_cast<std::size_t>(germs));
    rng.runif(g.x.data(), germs, 0.0, width);
    rng.runif(g.y.data(), germs, 0.0, height);
    rng.runif(g.v.data(), germs, 0.0, 1.0);
    return g;
}

GermDraws gibbs_germs(RRng& rng, i64 germs, double r, int patch_classes, double width, double height, int giveup) {
    // spatstat.random's rSSIengine.owin(): proposals in blocks of `germs`
    // (runifrect: n x's, then n y's), accepted when no earlier point is
    // within r; then sample(rep(1:patch_classes, length.out = germs)).
    GermDraws g;
    const double r2 = r * r;
    std::vector<double> xprop, yprop;
    std::size_t next = 0;
    int ntries = 0;
    const i64 n = germs;
    while (ntries < giveup && n > 0) {
        ++ntries;
        if (next == xprop.size()) {
            xprop.assign(static_cast<std::size_t>(n), 0.0);
            yprop.assign(static_cast<std::size_t>(n), 0.0);
            rng.runif(xprop.data(), n, 0.0, width);
            rng.runif(yprop.data(), n, 0.0, height);
            next = 0;
        }
        const double xn = xprop[next], yn = yprop[next];
        ++next;
        bool close = false;
        for (std::size_t k = 0; k < g.x.size() && !close; ++k) {
            const double dx = xn - g.x[k], dy = yn - g.y[k];
            close = dx * dx + dy * dy <= r2;
        }
        if (!close) {
            g.x.push_back(xn);
            g.y.push_back(yn);
            ntries = 0;
        }
        if (static_cast<i64>(g.x.size()) >= n) break;
    }
    // m <- sample(rep(1:patch_classes, length.out = germs))
    std::vector<double> rep(static_cast<std::size_t>(germs));
    for (i64 i = 0; i < germs; ++i) rep[static_cast<std::size_t>(i)] = static_cast<double>(i % patch_classes + 1);
    if (germs == 1) {
        g.v = {static_cast<double>(rng.sample_int(1, 1, false)[0])};
    } else {
        const std::vector<i64> idx = rng.sample_int(germs, germs, false);
        g.v.resize(static_cast<std::size_t>(germs));
        for (i64 i = 0; i < germs; ++i)
            g.v[static_cast<std::size_t>(i)] = rep[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)] - 1)];
    }
    return g;
}

std::vector<double> nearest_germ_raster(const std::vector<double>& gx, const std::vector<double>& gy,
                                        const std::vector<double>& value, i64 nrow, i64 ncol) {
    // cell centres (c + 0.5, nrow - r - 0.5); the germs bucketed on a grid of
    // about one germ per bucket, searched in growing rings
    const i64 n = static_cast<i64>(gx.size());
    std::vector<double> out(static_cast<std::size_t>(nrow * ncol), std::numeric_limits<double>::quiet_NaN());
    if (n == 0) return out;
    const double w = static_cast<double>(ncol), h = static_cast<double>(nrow);
    const double side = std::max(std::sqrt(w * h / static_cast<double>(n)), 1e-9);
    const i64 bx = std::max<i64>(1, static_cast<i64>(std::ceil(w / side)));
    const i64 by = std::max<i64>(1, static_cast<i64>(std::ceil(h / side)));
    auto bucket_of = [&](double x, double y, i64& ix, i64& iy) {
        ix = std::min(bx - 1, std::max<i64>(0, static_cast<i64>(x / side)));
        iy = std::min(by - 1, std::max<i64>(0, static_cast<i64>(y / side)));
    };
    std::vector<i64> start(static_cast<std::size_t>(bx * by + 1), 0), items(static_cast<std::size_t>(n));
    std::vector<i64> where(static_cast<std::size_t>(n));
    for (i64 k = 0; k < n; ++k) {
        i64 ix, iy;
        bucket_of(gx[static_cast<std::size_t>(k)], gy[static_cast<std::size_t>(k)], ix, iy);
        where[static_cast<std::size_t>(k)] = iy * bx + ix;
        ++start[static_cast<std::size_t>(iy * bx + ix + 1)];
    }
    for (std::size_t b = 1; b < start.size(); ++b) start[b] += start[b - 1];
    {
        std::vector<i64> fill(start.begin(), start.end() - 1);
        for (i64 k = 0; k < n; ++k) items[static_cast<std::size_t>(fill[static_cast<std::size_t>(where[static_cast<std::size_t>(k)])]++)] = k;
    }
    lsm::parallel_for_dynamic(nrow * ncol >= 4096, nrow, [&](i64 r) {
        const double y = h - (static_cast<double>(r) + 0.5);
        for (i64 c = 0; c < ncol; ++c) {
            const double x = static_cast<double>(c) + 0.5;
            i64 ix, iy;
            bucket_of(x, y, ix, iy);
            double best = std::numeric_limits<double>::infinity();
            i64 besti = -1;
            for (i64 ring = 0;; ++ring) {
                for (i64 jy = iy - ring; jy <= iy + ring; ++jy) {
                    if (jy < 0 || jy >= by) continue;
                    for (i64 jx = ix - ring; jx <= ix + ring; ++jx) {
                        if (jx < 0 || jx >= bx) continue;
                        if (std::max(std::llabs(jx - ix), std::llabs(jy - iy)) != ring) continue;
                        const i64 b = jy * bx + jx;
                        for (i64 t = start[static_cast<std::size_t>(b)]; t < start[static_cast<std::size_t>(b + 1)]; ++t) {
                            const i64 k = items[static_cast<std::size_t>(t)];
                            const double dx = x - gx[static_cast<std::size_t>(k)], dy = y - gy[static_cast<std::size_t>(k)];
                            const double d = dx * dx + dy * dy;
                            if (d < best || (d == best && k < besti)) {
                                best = d;
                                besti = k;
                            }
                        }
                    }
                }
                // every germ outside the rings searched is farther than `ring` buckets
                const double reach = static_cast<double>(ring) * side;
                if (besti >= 0 && reach * reach >= best) break;
                if (ring > bx + by) break;
            }
            out[static_cast<std::size_t>(r * ncol + c)] = value[static_cast<std::size_t>(besti)];
        }
    });
    return out;
}

}  // namespace nlm
