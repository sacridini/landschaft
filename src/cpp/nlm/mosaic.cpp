// SPDX-License-Identifier: GPL-3.0-or-later
//
// nlm_mosaicfield() of NLMR: spatstat.random's rMosaicField() of a Poisson
// line tessellation (rpoislinetess(4) on the unit square), summed over
// iterations, and the log intensity of rLGCP("exp", 4, var = 1,
// scale = 0.2) for `infinit`. spatstat.random 3.4 / spatstat.geom 3.7.
//
// chop.tess() splits every tile by each line into the part below and the
// part above it, in that order, and intersect.tess() then keeps the tiles
// that meet the unit square. The tiles are therefore the cells of the line
// arrangement inside the square, ordered lexicographically by their side of
// line 1, line 2, ... (below first); as.im.tess() gives each pixel the
// tile containing its centre, and rMosaicField() a normal value per tile.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "nlm/models.hpp"
#include "nlm/rfft.hpp"

namespace nlm {

namespace {

constexpr double kPi = 3.141592653589793238462643383280;

struct Line {
    int kind;  // 0: y = a + b x, 1: horizontal y = h, 2: vertical x = v
    double a, b, c;
    // 1 on the second tile of the split (above, or right of a vertical line)
    bool side(double x, double y) const {
        if (kind == 0) return y > a + b * x;
        if (kind == 1) return y > c;
        return x > c;
    }
    // signed value: < 0 on the first tile's side
    double f(double x, double y) const {
        if (kind == 0) return y - (a + b * x);
        if (kind == 1) return y - c;
        return x - c;
    }
};

using Poly = std::vector<std::array<double, 2>>;

double area(const Poly& p) {
    double s = 0;
    for (std::size_t i = 0, n = p.size(); i < n; ++i) {
        const auto& u = p[i];
        const auto& w = p[(i + 1) % n];
        s += u[0] * w[1] - w[0] * u[1];
    }
    return 0.5 * s;
}

// The part of a convex polygon with sign * line.f <= 0.
Poly clip(const Poly& p, const Line& line, double sign) {
    Poly out;
    const std::size_t n = p.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& u = p[i];
        const auto& w = p[(i + 1) % n];
        const double fu = sign * line.f(u[0], u[1]);
        const double fw = sign * line.f(w[0], w[1]);
        if (fu <= 0) out.push_back(u);
        if ((fu < 0 && fw > 0) || (fu > 0 && fw < 0)) {
            const double t = fu / (fu - fw);
            out.push_back({u[0] + t * (w[0] - u[0]), u[1] + t * (w[1] - u[1])});
        }
    }
    return out;
}

using Key = std::vector<std::uint64_t>;

// Pixel centres of spatstat's as.mask(dimyx): seq(lo + step/2, hi - step/2,
// length.out = n).
std::vector<double> centres(i64 n) {
    std::vector<double> v(static_cast<std::size_t>(n));
    const double step = (1.0 - 0.0) / static_cast<double>(n);
    const double from = 0.0 + step / 2, to = 1.0 - step / 2;
    if (n == 1) {
        v[0] = from;
        return v;
    }
    const double by = (to - from) / static_cast<double>(n - 1);
    v[0] = from;
    for (i64 k = 1; k + 1 < n; ++k) v[static_cast<std::size_t>(k)] = from + static_cast<double>(k) * by;
    v[static_cast<std::size_t>(n - 1)] = to;
    return v;
}

}  // namespace

std::vector<double> mosaic_tile_field(RRng& rng, i64 nrow, i64 ncol, double mean, double sd) {
    // rpoislinetess(4): lines through the disc around the unit square
    const double rmax = std::sqrt(1.0 * 1.0 + 1.0 * 1.0) / 2;
    const double xmid = 0.5, ymid = 0.5;
    const double n_lines = rng.rpois(4.0 * 2 * kPi * rmax);
    std::vector<Line> lines;
    if (n_lines > 0) {
        const i64 n = static_cast<i64>(n_lines);
        std::vector<double> theta(static_cast<std::size_t>(n)), p(static_cast<std::size_t>(n));
        rng.runif(theta.data(), n, 0.0, 2 * kPi);
        rng.runif(p.data(), n, 0.0, rmax);
        for (i64 i = 0; i < n; ++i) {
            double& pi_ = p[static_cast<std::size_t>(i)];
            const double t = theta[static_cast<std::size_t>(i)];
            pi_ = (pi_ + xmid * lsm::r_cos(t)) + ymid * lsm::r_sin(t);
        }
        // infline(p = p, theta = theta)
        for (i64 i = 0; i < n; ++i) {
            double t = theta[static_cast<std::size_t>(i)];  // theta %% (2 pi): unchanged in [0, 2 pi)
            double pp = p[static_cast<std::size_t>(i)];
            if (t >= kPi) {
                t = t - kPi;
                pp = -pp;
            }
            const double co = lsm::r_cos(t), si = lsm::r_sin(t);
            if (t == 0)
                lines.push_back({2, 0, 0, pp});
            else if (co == 0)
                lines.push_back({1, 0, 0, pp});
            else
                lines.push_back({0, pp / si, -co / si, 0});
        }
    }
    // tiles: cells of the arrangement in the unit square, in chop.tess order
    const std::size_t words = (lines.size() + 63) / 64 + 1;
    struct Tile {
        Poly poly;
        Key key;
    };
    std::vector<Tile> tiles{{Poly{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, Key(words, 0)}};
    for (std::size_t li = 0; li < lines.size(); ++li) {
        std::vector<Tile> next;
        next.reserve(tiles.size() * 2);
        for (const Tile& t : tiles) {
            for (int s = 0; s < 2; ++s) {
                Poly part = clip(t.poly, lines[li], s == 0 ? 1.0 : -1.0);
                if (part.size() < 3 || !(std::fabs(area(part)) > 0)) continue;
                Key k = t.key;
                if (s) k[li / 64] |= std::uint64_t{1} << (li % 64);
                next.push_back({std::move(part), std::move(k)});
            }
        }
        tiles.swap(next);
    }
    std::vector<Key> keys;
    keys.reserve(tiles.size());
    for (Tile& t : tiles) keys.push_back(std::move(t.key));
    // lexicographic order of the sides = order of the keys compared from line 1
    auto less = [](const Key& x, const Key& y) {
        for (std::size_t w = 0; w < x.size(); ++w) {
            if (x[w] == y[w]) continue;
            const std::uint64_t diff = x[w] ^ y[w];
            std::uint64_t low = diff & (~diff + 1);  // first differing line
            return (x[w] & low) == 0;
        }
        return false;
    };
    std::vector<double> values(keys.size());
    rng.rnorm(values.data(), static_cast<i64>(values.size()), mean, sd);

    const std::vector<double> xs = centres(ncol), ys = centres(nrow);
    std::vector<double> out(static_cast<std::size_t>(nrow * ncol), std::numeric_limits<double>::quiet_NaN());
    lsm::parallel_for(nrow * ncol >= 4096, nrow, [&](i64 r) {
        Key k(words, 0);
        for (i64 c = 0; c < ncol; ++c) {
            std::fill(k.begin(), k.end(), 0);
            const double x = xs[static_cast<std::size_t>(c)], y = ys[static_cast<std::size_t>(r)];
            for (std::size_t li = 0; li < lines.size(); ++li)
                if (lines[li].side(x, y)) k[li / 64] |= std::uint64_t{1} << (li % 64);
            const auto it = std::lower_bound(keys.begin(), keys.end(), k, less);
            if (it != keys.end() && *it == k)
                out[static_cast<std::size_t>(r * ncol + c)] = values[static_cast<std::size_t>(it - keys.begin())];
        }
    });
    return out;
}

std::vector<double> lgcp_log_field(RRng& rng, i64 nrow, i64 ncol, double mu, double var, double scale) {
    // rGRFexpo() by rCircEmbedEngine() (spatstat.random): grid.prep with
    // ext = 2, covariance.prep(wrap = TRUE), generate.normals(); then
    // log(exp(Z + mu)) as nlm_mosaicfield takes log(attr(X, "Lambda")).
    const i64 M = ncol, N = nrow, Mx = 2 * M, Nx = 2 * N;
    if (Mx * Nx > 0x7fffffff) throw std::invalid_argument("raster too large");
    const double cw = (1.0 - 0.0) / static_cast<double>(M), ch = (1.0 - 0.0) / static_cast<double>(N);
    auto cens = [](i64 n, double step) {
        std::vector<double> v(static_cast<std::size_t>(n));
        for (i64 k = 0; k < n; ++k) v[static_cast<std::size_t>(k)] = (0.0 + static_cast<double>(k) * step) + 0.5 * step;
        return v;
    };
    const std::vector<double> mc = cens(Mx, cw), nc = cens(Nx, ch);
    const double Rx = static_cast<double>(2.0 * static_cast<double>(M)) * cw;
    const double Ry = static_cast<double>(2.0 * static_cast<double>(N)) * ch;
    std::vector<double> md(static_cast<std::size_t>(Mx)), nd(static_cast<std::size_t>(Nx));
    for (i64 k = 0; k < Mx; ++k) {
        const double a = std::fabs(mc[0] - mc[static_cast<std::size_t>(k)]);
        md[static_cast<std::size_t>(k)] = std::min(a, Rx - a);
    }
    for (i64 k = 0; k < Nx; ++k) {
        const double a = std::fabs(nc[0] - nc[static_cast<std::size_t>(k)]);
        nd[static_cast<std::size_t>(k)] = std::min(a, Ry - a);
    }
    const i64 ncell = Mx * Nx;
    std::vector<double> z(2 * static_cast<std::size_t>(ncell));
    lsm::parallel_for(ncell >= 16384, Nx, [&](i64 j) {
        for (i64 i = 0; i < Mx; ++i) {
            const double dx = md[static_cast<std::size_t>(i)], dy = nd[static_cast<std::size_t>(j)];
            const double d = lsm::r_sqrt(dx * dx + dy * dy);
            const std::size_t k = static_cast<std::size_t>(j * Mx + i);
            z[2 * k] = var * lsm::r_exp(-d / scale);
            z[2 * k + 1] = 0.0;
        }
    });
    r_fft2(z, static_cast<int>(Mx), static_cast<int>(Nx), true, true);
    std::vector<double> sq(static_cast<std::size_t>(ncell));
    double mn = std::numeric_limits<double>::infinity();
    for (i64 k = 0; k < ncell; ++k) mn = std::min(mn, z[2 * static_cast<std::size_t>(k)]);
    lsm::parallel_for(ncell >= 16384, ncell, [&](i64 k) {
        double v = z[2 * static_cast<std::size_t>(k)];
        if (mn < 0 && v < 0) v = 0;  // pmax(0, refft)
        sq[static_cast<std::size_t>(k)] = lsm::r_sqrt(v);
    });
    const double sqrtncell = std::sqrt(static_cast<double>(ncell));
    std::vector<double> noise(static_cast<std::size_t>(ncell));
    rng.rnorm(noise.data(), ncell);
    for (i64 k = 0; k < ncell; ++k) {
        z[2 * static_cast<std::size_t>(k)] = noise[static_cast<std::size_t>(k)];
        z[2 * static_cast<std::size_t>(k) + 1] = 0.0;
    }
    r_fft2(z, static_cast<int>(Mx), static_cast<int>(Nx), false, true);
    lsm::parallel_for(ncell >= 16384, ncell, [&](i64 k) {
        const std::size_t i = static_cast<std::size_t>(k);
        const double s = sq[i], re = z[2 * i], im = z[2 * i + 1];
        z[2 * i] = (s * re - 0.0 * im) / sqrtncell;
        z[2 * i + 1] = (s * im + 0.0 * re) / sqrtncell;
    });
    r_fft2(z, static_cast<int>(Mx), static_cast<int>(Nx), true, true);
    // realisation[i + j * Mx] -> image row j (y), column i (x)
    std::vector<double> out(static_cast<std::size_t>(N * M));
    lsm::parallel_for(N * M >= 16384, N, [&](i64 j) {
        for (i64 i = 0; i < M; ++i) {
            const double v = z[2 * static_cast<std::size_t>(j * Mx + i)] / sqrtncell;
            out[static_cast<std::size_t>(j * M + i)] = lsm::r_log(lsm::r_exp(v + mu));
        }
    });
    return out;
}

MosaicFieldResult mosaic_field(RRng& rng, i64 nrow, i64 ncol, int n, bool nlmr_compat, double mean, double sd,
                               bool collect, bool infinit) {
    // nlm_mosaicfield(): the first field, then `for (i in 2:n)` more (2:1
    // is c(2, 1) in R: three fields for n = 1), summed in order; `collect`
    // keeps the i-th field divided by sqrt(i) (the loop index: n = 1 writes
    // field 3 over field 1). Images are flipped to rows from the top.
    MosaicFieldResult res;
    auto flip = [&](std::vector<double>&& v) {
        std::vector<double> out(v.size());
        for (i64 r = 0; r < nrow; ++r)
            std::copy(v.begin() + (nrow - 1 - r) * ncol, v.begin() + (nrow - r) * ncol, out.begin() + r * ncol);
        return out;
    };
    if (n > 0) {
        std::vector<std::vector<double>> list;
        auto put = [&](int i, const std::vector<double>& f) {
            if (!collect) return;
            if (static_cast<int>(list.size()) < i) list.resize(static_cast<std::size_t>(i));
            list[static_cast<std::size_t>(i - 1)] = f;
        };
        std::vector<double> sum = mosaic_tile_field(rng, nrow, ncol, mean, sd);
        put(1, sum);
        std::vector<int> idx;
        if (n >= 2)
            for (int i = 2; i <= n; ++i) idx.push_back(i);
        else if (nlmr_compat)
            idx = {2, 1};
        for (int i : idx) {
            const std::vector<double> f = mosaic_tile_field(rng, nrow, ncol, mean, sd);
            for (std::size_t k = 0; k < sum.size(); ++k) sum[k] = sum[k] + f[k];
            put(i, f);
        }
        res.sum = flip(std::move(sum));
        for (std::size_t i = 0; i < list.size(); ++i) {
            const double s = std::sqrt(static_cast<double>(i + 1));
            std::vector<double>& f = list[i];
            for (double& v : f) v = v / s;
            res.steps.push_back(flip(std::move(f)));
        }
    }
    if (infinit) res.inf = flip(lgcp_log_field(rng, nrow, ncol, 4.0, 1.0, 0.2));
    return res;
}

}  // namespace nlm
