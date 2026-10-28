// SPDX-License-Identifier: GPL-3.0-or-later
//
// The trend of every cell and the spatial autocorrelation of a map (see stats.hpp).

#include "stats/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "stats/special.hpp"

namespace stats {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

inline double sq(double x) { return x * x; }

// numpy's median of values (the mean of the two middle ones), NaN without any; a
// zero positive (numpy's partition does not tell -0 from 0).
double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end(), [](double a, double b) {
        // total order (no NaNs here)
        return a < b || (a == b && std::signbit(a) && !std::signbit(b));
    });
    const std::size_t n = v.size();
    if (n == 0) return kNaN;
    if (n % 2 == 1) return v[n / 2] + 0.0;
    return (v[n / 2 - 1] + v[n / 2]) / 2.0 + 0.0;
}

}  // namespace

MannKendall mann_kendall(const double* series, int t, double alpha) {
    std::vector<double> x;
    x.reserve(static_cast<std::size_t>(t));
    for (int k = 0; k < t; ++k)
        if (!std::isnan(series[k])) x.push_back(series[k]);
    const i64 n = static_cast<i64>(x.size());
    i64 s = 0;
    for (i64 k = 0; k + 1 < n; ++k)
        for (i64 j = k + 1; j < n; ++j) s += (x[j] > x[k] ? 1 : 0) - (x[j] < x[k] ? 1 : 0);
    std::vector<double> sorted = x;
    std::sort(sorted.begin(), sorted.end());
    std::vector<double> ties;
    for (i64 k = 0; k < n;) {
        i64 j = k;
        while (j < n && sorted[j] == sorted[k]) ++j;
        ties.push_back(static_cast<double>(j - k));
        k = j;
    }
    const double nf = static_cast<double>(n);
    const double base = static_cast<double>(n * (n > 0 ? n - 1 : 0) * (2 * n + 5));
    double var_s;
    if (static_cast<i64>(ties.size()) == n) {
        var_s = base / 18.0;
    } else {
        std::vector<double> tt(ties.size());
        for (std::size_t k = 0; k < ties.size(); ++k) tt[k] = ties[k] * (ties[k] - 1.0) * (2.0 * ties[k] + 5.0);
        var_s = (base - np_sum(tt)) / 18.0;
    }
    MannKendall r{};
    r.s = static_cast<double>(s);
    r.var_s = var_s;
    r.tau = r.s / (0.5 * nf * (nf - 1.0));
    r.z = r.s > 0.0 ? (r.s - 1.0) / std::sqrt(var_s) : (r.s < 0.0 ? (r.s + 1.0) / std::sqrt(var_s) : 0.0);
    r.p = 2.0 * (1.0 - ndtr(std::fabs(r.z)));
    const bool h = std::fabs(r.z) > ndtri(1.0 - alpha / 2.0);
    r.h = h ? 1.0 : 0.0;
    r.trend = h && r.z < 0.0 ? -1.0 : (h && r.z > 0.0 ? 1.0 : 0.0);
    // Sen's slope over the positions of the series as it is (pairs with a missing value left out)
    std::vector<double> d;
    d.reserve(static_cast<std::size_t>(t) * static_cast<std::size_t>(t > 0 ? t - 1 : 0) / 2);
    for (int i = 0; i + 1 < t; ++i)
        for (int j = i + 1; j < t; ++j) {
            const double v = (series[j] - series[i]) / static_cast<double>(j - i);
            if (!std::isnan(v)) d.push_back(v);
        }
    r.slope = median(d);
    std::vector<double> pos;
    for (int i = 0; i < t; ++i)
        if (!std::isnan(series[i])) pos.push_back(static_cast<double>(i));
    std::vector<double> valid = x;
    r.intercept = median(valid) - median(pos) * r.slope;
    return r;
}

void trend(const double* layers, int t, i64 n, double alpha, int min_values, double* out, int threads) {
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> series(static_cast<std::size_t>(t));
#pragma omp for schedule(dynamic, 1024)
        for (i64 i = 0; i < n; ++i) {
            int valid = 0;
            for (int k = 0; k < t; ++k) {
                series[k] = layers[static_cast<std::size_t>(k) * n + i];
                if (!std::isnan(series[k])) ++valid;
            }
            double v[6] = {kNaN, kNaN, kNaN, kNaN, kNaN, kNaN};
            if (valid >= min_values) {
                const MannKendall m = mann_kendall(series.data(), t, alpha);
                v[0] = m.slope;
                v[1] = m.intercept;
                v[2] = m.p;
                v[3] = m.z;
                v[4] = m.tau;
                v[5] = m.trend;
            }
            for (int k = 0; k < 6; ++k) out[static_cast<std::size_t>(k) * n + i] = v[k];
        }
    }
}

Lattice lattice(const double* data, i64 ny, i64 nx, bool queen) {
    Lattice l;
    std::vector<i64> index(static_cast<std::size_t>(ny * nx), -1);
    for (i64 i = 0; i < ny * nx; ++i)
        if (!std::isnan(data[i])) {
            index[i] = static_cast<i64>(l.cells.size());
            l.cells.push_back(i);
        }
    l.start.reserve(l.cells.size() + 1);
    l.start.push_back(0);
    for (const i64 i : l.cells) {
        const i64 r = i / nx, c = i % nx;
        for (i64 dr = -1; dr <= 1; ++dr)
            for (i64 dc = -1; dc <= 1; ++dc) {
                if ((dr == 0 && dc == 0) || (!queen && dr != 0 && dc != 0)) continue;
                const i64 rr = r + dr, cc = c + dc;
                if (rr < 0 || cc < 0 || rr >= ny || cc >= nx) continue;
                const i64 k = index[rr * nx + cc];
                if (k >= 0) l.nb.push_back(k);
            }
        l.start.push_back(static_cast<i64>(l.nb.size()));
    }
    return l;
}

namespace {

// The spatial lag of row-standardised weights: the mean of every cell's neighbours (0 without any).
double lag_at(const Lattice& l, const double* z, i64 k) {
    const i64 c = l.count(k);
    if (c == 0) return 0.0;
    double s = 0.0;
    for (i64 j = l.start[k]; j < l.start[k + 1]; ++j) s += z[l.nb[j]];
    return s / static_cast<double>(c);
}

double weight(const Lattice& l, i64 k) {
    const i64 c = l.count(k);
    return c == 0 ? 0.0 : 1.0 / static_cast<double>(c);
}

// A two-sided p of a z (esda's: the tail it falls in, doubled).
double two_sided(double z) { return 2.0 * (z > 0.0 ? ndtr(-z) : ndtr(z)); }

// Moran's I of values whose deviations are z (inum / z2ss scaled), sums in a fixed order.
double moran_i_of(const double* y, const Lattice& l, double s0, int threads) {
    const i64 n = l.size();
    const double mean = np_sum(y, n) / static_cast<double>(n);
    std::vector<double> z(static_cast<std::size_t>(n)), z2(static_cast<std::size_t>(n)),
        prod(static_cast<std::size_t>(n));
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 k = 0; k < n; ++k) {
        z[k] = y[k] - mean;
        z2[k] = z[k] * z[k];
    }
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 k = 0; k < n; ++k) prod[k] = z[k] * lag_at(l, z.data(), k);
    double inum = 0.0;
    for (i64 k = 0; k < n; ++k) inum += prod[k];
    return static_cast<double>(n) / s0 * inum / np_sum(z2);
}

double s0_of(const Lattice& l) {
    double s0 = 0.0;
    for (i64 k = 0; k < l.size(); ++k) s0 += weight(l, k) * static_cast<double>(l.count(k));
    return s0;
}

}  // namespace

GlobalMoran global_moran(const double* y, const Lattice& l) {
    const i64 nn = l.size();
    const double n = static_cast<double>(nn);
    const double mean = np_sum(y, nn) / n;
    std::vector<double> z(static_cast<std::size_t>(nn)), z2(static_cast<std::size_t>(nn));
    for (i64 k = 0; k < nn; ++k) {
        z[k] = y[k] - mean;
        z2[k] = z[k] * z[k];
    }
    const double z2ss = np_sum(z2);
    const double s0 = s0_of(l);
    double s1 = 0.0;
    std::vector<double> col(static_cast<std::size_t>(nn), 0.0);
    for (i64 k = 0; k < nn; ++k)
        for (i64 j = l.start[k]; j < l.start[k + 1]; ++j) {
            const double a = weight(l, k) + weight(l, l.nb[j]);
            s1 += a * a;
            col[l.nb[j]] += weight(l, k);
        }
    s1 = s1 / 2.0;
    double s2 = 0.0;
    for (i64 k = 0; k < nn; ++k) s2 += sq(weight(l, k) * static_cast<double>(l.count(k)) + col[k]);
    double inum = 0.0;
    for (i64 k = 0; k < nn; ++k) inum += z[k] * lag_at(l, z.data(), k);
    GlobalMoran g{};
    g.i = n / s0 * inum / z2ss;
    g.ei = -1.0 / (n - 1.0);
    const double n2 = n * n, s02 = s0 * s0;
    g.vi_norm = (n2 * s1 - n * s2 + 3.0 * s02) / ((n - 1.0) * (n + 1.0) * s02) - sq(1.0 / (n - 1.0));
    double m4 = 0.0, m2 = 0.0;
    for (i64 k = 0; k < nn; ++k) m4 += sq(sq(z[k]));
    for (i64 k = 0; k < nn; ++k) m2 += z[k] * z[k];
    const double kurt = (m4 / n) / sq(m2 / n);
    const double a = n * ((n2 - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s02);
    const double b = kurt * ((n2 - n) * s1 - 2.0 * n * s2 + 6.0 * s02);
    g.vi_rand = (a - b) / ((n - 1.0) * (n - 2.0) * (n - 3.0) * s02) - g.ei * g.ei;
    g.z_norm = (g.i - g.ei) / std::sqrt(g.vi_norm);
    g.z_rand = (g.i - g.ei) / std::sqrt(g.vi_rand);
    g.p_norm = two_sided(g.z_norm);
    g.p_rand = two_sided(g.z_rand);
    return g;
}

double moran_p_sim(const double* y, const Lattice& l, double observed, int perms, u64 seed, int threads) {
    if (perms <= 0) return kNaN;
    NumpyRng rng(seed);
    std::vector<double> shuffled(y, y + l.size());
    const double s0 = s0_of(l);
    int larger = 0;
    for (int k = 0; k < perms; ++k) {
        rng.shuffle(shuffled.data(), static_cast<i64>(shuffled.size()));
        if (moran_i_of(shuffled.data(), l, s0, threads) >= observed) ++larger;
    }
    return (static_cast<double>(std::min(larger, perms - larger)) + 1.0) / (static_cast<double>(perms) + 1.0);
}

void local_moran(const double* y, const Lattice& l, int perms, u64 seed, double* li, double* quadrant, double* ei,
                 double* vi, double* lz, double* p, int threads) {
    const i64 nn = l.size();
    const double n = static_cast<double>(nn);
    const double mean = np_sum(y, nn) / n;
    std::vector<double> dev(static_cast<std::size_t>(nn)), tmp(static_cast<std::size_t>(nn));
    for (i64 k = 0; k < nn; ++k) {
        dev[k] = y[k] - mean;
        tmp[k] = dev[k] * dev[k];
    }
    const double sd = std::sqrt(np_sum(tmp) / n);
    std::vector<double> z(static_cast<std::size_t>(nn));
    for (i64 k = 0; k < nn; ++k) {
        z[k] = dev[k] / sd;
        tmp[k] = z[k] * z[k];
    }
    const double den = np_sum(tmp);
    const double m2 = np_sum(tmp) / n;
    for (i64 k = 0; k < nn; ++k) tmp[k] = sq(sq(z[k]));
    const double m4 = np_sum(tmp) / n;
    const double b2 = m4 / (m2 * m2);
    const double n1 = n - 1.0;
    // the permutations' statistic: deviations (not standardised) over their sum of squares
    double den_p = 0.0;
    for (i64 k = 0; k < nn; ++k) den_p += dev[k] * dev[k];
#pragma omp parallel for num_threads(threads) schedule(dynamic, 256)
    for (i64 k = 0; k < nn; ++k) {
        const double zl = lag_at(l, z.data(), k);
        li[k] = (n - 1.0) * z[k] * zl / den;
        const bool hi = z[k] > 0.0, nhi = zl > 0.0;
        quadrant[k] = hi && nhi ? 1.0 : (!hi && nhi ? 2.0 : (!hi && !nhi ? 3.0 : 4.0));
        const i64 c = l.count(k);
        const double wi = c == 0 ? 0.0 : 1.0, wi2 = c == 0 ? 0.0 : 1.0 / static_cast<double>(c);
        ei[k] = -wi / n1;
        vi[k] = wi2 * (n - b2) / n1 + (wi * wi - wi2) * (2.0 * b2 - n) / (n1 * (n - 2.0)) - sq(-wi / n1);
        lz[k] = (li[k] - ei[k]) / std::sqrt(vi[k]);
        if (c == 0 || perms <= 0) {
            p[k] = kNaN;
            continue;
        }
        auto stat = [&](auto&& at) {
            double s = 0.0;
            for (i64 j = 0; j < c; ++j) s += dev[at(j)];
            return (n - 1.0) * dev[k] * (s / static_cast<double>(c)) / den_p;
        };
        const double observed = stat([&](i64 j) { return l.nb[l.start[k] + j]; });
        NumpyRng rng(seed * 0x9E3779B97F4A7C15ull + static_cast<u64>(k));
        int larger = 0;
        for (int r = 0; r < perms; ++r) {
            const std::vector<u64> pick = rng.choice_distinct(static_cast<u64>(nn - 1), static_cast<u64>(c));
            const double v = stat([&](i64 j) {
                const i64 q = static_cast<i64>(pick[j]);
                return q >= k ? q + 1 : q;
            });
            if (v >= observed) ++larger;
        }
        p[k] = (static_cast<double>(std::min(larger, perms - larger)) + 1.0) / (static_cast<double>(perms) + 1.0);
    }
}

void getis_ord(const double* y, const Lattice& l, double* g, double* z, double* p, double* spots, int threads) {
    const i64 nn = l.size();
    const double n = static_cast<double>(nn);
    const double total = np_sum(y, nn);
    const double mean = total / n;
    std::vector<double> y2(static_cast<std::size_t>(nn));
    for (i64 k = 0; k < nn; ++k) y2[k] = y[k] * y[k];
    const double squares = np_sum(y2) / n;
    const double var = squares - mean * mean;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 k = 0; k < nn; ++k) {
        const double w = 1.0 / static_cast<double>(l.count(k) + 1);
        double s = 0.0;
        for (i64 j = l.start[k]; j < l.start[k + 1]; ++j) s += y[l.nb[j]] * w;
        const double gi = (y[k] * w + s) / total;
        const double card = 1.0;
        const double ev = card / n;
        double vg = card * (n - card);
        vg /= n - 1.0;
        vg *= 1.0 / (n * n);
        vg *= var / (mean * mean);
        g[k] = gi;
        z[k] = (gi - ev) / std::sqrt(vg);
        p[k] = ndtr(-std::fabs(z[k]));
        const double q = 2.0 * p[k];
        const double level = q < 0.01 ? 3.0 : (q < 0.05 ? 2.0 : (q < 0.10 ? 1.0 : 0.0));
        const double sign = std::isnan(z[k]) ? kNaN : (std::signbit(z[k]) ? -1.0 : 1.0);
        spots[k] = level * sign;
    }
}

}  // namespace stats
