// SPDX-License-Identifier: GPL-3.0-or-later
//
// Interpolation of points as R's gstat (see interp.hpp).

#include "stats/interp.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>

namespace stats {

namespace {

using i64 = std::int64_t;

// The semivariance of a unit sill at a distance (gstat's fn_*).
double unit(Shape s, double h, double r) {
    if (h == 0.0) return 0.0;
    switch (s) {
        case Shape::kSpherical: {
            if (h >= r) return 1.0;
            const double hr = h / r;
            return hr * (1.5 - 0.5 * hr * hr);
        }
        case Shape::kExponential:
            return 1.0 - std::exp(-h / r);
        case Shape::kGaussian: {
            const double hr = h / r;
            return 1.0 - std::exp(-(hr * hr));
        }
    }
    return 0.0;
}

// Its derivative by the range (gstat's da_fn_*).
double d_range(Shape s, double h, double r) {
    switch (s) {
        case Shape::kSpherical: {
            if (h > r) return 0.0;
            const double hr2 = h / (r * r);
            return 1.5 * hr2 * (-1.0 + h * hr2);
        }
        case Shape::kExponential: {
            const double hr = -h / r;
            return (hr / r) * std::exp(hr);
        }
        case Shape::kGaussian: {
            const double hr = h / r;
            return (-hr / r) * std::exp(-(hr * hr));
        }
    }
    return 0.0;
}

// Cholesky's lower factor in place; false when not positive definite.
bool cholesky(double* a, i64 n) {
    for (i64 j = 0; j < n; ++j) {
        double d = a[j * n + j];
        for (i64 k = 0; k < j; ++k) d -= a[j * n + k] * a[j * n + k];
        if (d <= 0.0 || std::isnan(d)) return false;
        d = std::sqrt(d);
        a[j * n + j] = d;
        for (i64 i = j + 1; i < n; ++i) {
            double s = a[i * n + j];
            for (i64 k = 0; k < j; ++k) s -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = s / d;
        }
    }
    return true;
}

// L y = b in place.
void forward(const double* l, i64 n, double* y) {
    for (i64 i = 0; i < n; ++i) {
        for (i64 k = 0; k < i; ++k) y[i] -= l[i * n + k] * y[k];
        y[i] /= l[i * n + i];
    }
}

// L^T x = y in place.
void backward(const double* l, i64 n, double* y) {
    for (i64 i = n - 1; i >= 0; --i) {
        for (i64 k = i + 1; k < n; ++k) y[i] -= l[k * n + i] * y[k];
        y[i] /= l[i * n + i];
    }
}

// The small system of a fit's step (LAPACK's dpotrs: the back substitution
// subtracts the last unknowns first).
bool cholesky_solve(std::vector<double> a, std::vector<double>& b, i64 n) {
    if (!cholesky(a.data(), n)) return false;
    forward(a.data(), n, b.data());
    for (i64 i = n - 1; i >= 0; --i) {
        for (i64 k = n - 1; k > i; --k) b[i] -= a[k * n + i] * b[k];
        b[i] /= a[i * n + i];
    }
    return true;
}

double get(const VarioModel& m, int k) { return k == 0 ? m.nugget : (k == 1 ? m.psill : m.range); }

void set(VarioModel& m, int k, double v) {
    if (k == 0) m.nugget = v;
    else if (k == 1) m.psill = v;
    else m.range = v;
}

double sserr_of(const SampleVariogram& s, const VarioModel& m) {
    double e = 0.0;
    for (std::size_t i = 0; i < s.dist.size(); ++i) {
        const double r = s.gamma[i] - m.gamma(s.dist[i]);
        e += r * r * (s.np[i] / (s.dist[i] * s.dist[i]));
    }
    return e;
}

// gstat's wls_fit: Gauss-Newton steps of the parameters fitted, a step whose
// squared length passes 0.2 of the parameters' shrunk by that ratio, until the
// sum of squares falls by less than 1e-5 of itself after a free step, or 200
// iterations.
VarioFit wls_fit(const SampleVariogram& s, VarioModel m, bool fit_nugget, bool fit_psill, bool fit_range) {
    std::vector<int> fitted;
    if (fit_nugget) fitted.push_back(0);
    if (fit_psill) fitted.push_back(1);
    if (fit_range) fitted.push_back(2);
    if (fit_nugget && m.nugget == 0.0) m.nugget = 1.0;
    if (fit_psill && m.psill == 0.0) m.psill = 1.0;
    const i64 p = static_cast<i64>(fitted.size());
    const std::size_t nd = s.dist.size();
    if (p == 0 || nd == 0) return {m, sserr_of(s, m), nd == 0, true};
    std::vector<double> w(nd);
    for (std::size_t i = 0; i < nd; ++i) w[i] = std::sqrt(s.np[i] / (s.dist[i] * s.dist[i]));
    double old = DBL_MAX;
    std::vector<double> start(p), a(p * p), b(p), x(p);
    for (int iter = 1; iter <= 200; ++iter) {
        for (i64 j = 0; j < p; ++j) start[j] = get(m, fitted[j]);
        std::fill(a.begin(), a.end(), 0.0);
        std::fill(b.begin(), b.end(), 0.0);
        if (static_cast<i64>(nd) < p) return {m, sserr_of(s, m), true, false};
        for (std::size_t i = 0; i < nd; ++i) {
            const double h = s.dist[i];
            const double y = (s.gamma[i] - m.gamma(h)) * w[i];
            for (i64 j = 0; j < p; ++j) {
                const int k = fitted[j];
                const double d = k == 0 ? (h == 0.0 ? 0.0 : 1.0)
                                        : (k == 1 ? unit(m.shape, h, m.range) : m.psill * d_range(m.shape, h, m.range));
                x[j] = w[i] * d;
            }
            for (i64 r = 0; r < p; ++r) {
                b[r] += x[r] * y;
                for (i64 c = 0; c < p; ++c) a[r * p + c] += x[r] * x[c];
            }
        }
        if (!cholesky_solve(a, b, p)) return {m, sserr_of(s, m), true, false};
        bool bounded = false;
        if (fit_range) {
            // (gstat's v_norm2: the sum of the squares)
            double nb = 0.0, ns = 0.0;
            for (i64 j = 0; j < p; ++j) nb += b[j] * b[j];
            for (i64 j = 0; j < p; ++j) ns += start[j] * start[j];
            const double ratio = nb / ns;
            if (ratio > 0.2) {
                for (i64 j = 0; j < p; ++j) b[j] *= 0.2 / ratio;
                bounded = true;
            }
        }
        for (i64 j = 0; j < p; ++j) set(m, fitted[j], start[j] + b[j]);
        const double e = sserr_of(s, m);
        double step = old - e;
        if (e > DBL_EPSILON * 10.0) step /= e;
        old = e;
        if (step >= 0.0 && step < 1e-5 && !bounded) return {m, e, false, true};
        if (iter == 200) return {m, e, false, false};
    }
    return {m, old, false, false};
}

double d2_of(const double* x, const double* y, i64 i, double px, double py) {
    const double dx = x[i] - px, dy = y[i] - py;
    return dx * dx + dy * dy;
}

// The indices of the nmax points nearest a place, nearest first (ties by index).
void nearest(const double* x, const double* y, i64 n, double px, double py, i64 nmax, std::vector<i64>& idx,
             std::vector<double>& d2) {
    idx.resize(n);
    d2.resize(n);
    std::iota(idx.begin(), idx.end(), i64{0});
    for (i64 i = 0; i < n; ++i) d2[i] = d2_of(x, y, i, px, py);
    std::partial_sort(idx.begin(), idx.begin() + nmax, idx.end(), [&](i64 a, i64 b) {
        return d2[a] < d2[b] || (d2[a] == d2[b] && a < b);
    });
    idx.resize(nmax);
}

double idw_over(const double* x, const double* y, const double* z, const i64* sel, i64 k, double px, double py,
                double power) {
    if (k == 1) return z[sel[0]];
    double sw = 0.0, sv = 0.0;
    for (i64 t = 0; t < k; ++t) {
        const i64 j = sel[t];
        const double d2 = d2_of(x, y, j, px, py);
        if (d2 == 0.0) return z[j];
        const double w = power == 2.0 ? 1.0 / d2 : std::pow(d2, -0.5 * power);
        sw += w;
        sv += w * z[j];
    }
    return sv / sw;
}

double dist(double ax, double ay, double bx, double by) {
    const double dx = ax - bx, dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

// Ordinary kriging of points with a model, ready for every place.
struct Kriger {
    std::vector<double> l, ci1, r;
    double s = 0.0, mean = 0.0;

    bool build(const double* x, const double* y, const double* z, i64 n, const VarioModel& m) {
        l.assign(static_cast<std::size_t>(n * n), 0.0);
        for (i64 i = 0; i < n; ++i)
            for (i64 j = 0; j < n; ++j) l[i * n + j] = m.covariance(dist(x[i], y[i], x[j], y[j]));
        if (!cholesky(l.data(), n)) return false;
        ci1.assign(static_cast<std::size_t>(n), 1.0);
        forward(l.data(), n, ci1.data());
        backward(l.data(), n, ci1.data());
        s = 0.0;
        for (i64 i = 0; i < n; ++i) s += ci1[i];
        double cz = 0.0;
        for (i64 i = 0; i < n; ++i) cz += ci1[i] * z[i];
        mean = cz / s;
        r.resize(static_cast<std::size_t>(n));
        for (i64 i = 0; i < n; ++i) r[i] = z[i] - mean;
        forward(l.data(), n, r.data());
        backward(l.data(), n, r.data());
        return true;
    }

    void at(const double* x, const double* y, i64 n, const VarioModel& m, double px, double py, std::vector<double>& c0,
            std::vector<double>& t, double& pred, double& var) const {
        c0.resize(static_cast<std::size_t>(n));
        for (i64 i = 0; i < n; ++i) c0[i] = m.covariance(dist(x[i], y[i], px, py));
        double cr = 0.0;
        for (i64 i = 0; i < n; ++i) cr += c0[i] * r[i];
        pred = mean + cr;
        t = c0;
        forward(l.data(), n, t.data());
        double cc = 0.0;
        for (i64 i = 0; i < n; ++i) cc += t[i] * t[i];
        double c1 = 0.0;
        for (i64 i = 0; i < n; ++i) c1 += ci1[i] * c0[i];
        const double u = 1.0 - c1;
        var = m.covariance(0.0) - cc + u * u / s;
    }
};

}  // namespace

double VarioModel::gamma(double h) const {
    if (shape == Shape::kSpherical && h > std::max(range, 0.0)) return nugget + psill;
    const double nug = h == 0.0 ? 0.0 : 1.0;
    return nugget * nug + psill * unit(shape, h, range);
}

double default_cutoff(const double* x, const double* y, i64 n) {
    double x0 = INFINITY, x1 = -INFINITY, y0 = INFINITY, y1 = -INFINITY;
    for (i64 i = 0; i < n; ++i) {
        x0 = std::min(x0, x[i]);
        x1 = std::max(x1, x[i]);
        y0 = std::min(y0, y[i]);
        y1 = std::max(y1, y[i]);
    }
    const double dx = x1 - x0, dy = y1 - y0;
    return std::sqrt(dx * dx + dy * dy) / 3.0;
}

SampleVariogram sample_variogram(const double* x, const double* y, const double* z, i64 n, double cutoff,
                                 double width) {
    const auto n_est = static_cast<std::size_t>(std::max(std::ceil(cutoff / width), 1.0));
    std::vector<i64> nh(n_est, 0);
    std::vector<double> dsum(n_est, 0.0), gsum(n_est, 0.0);
    for (i64 i = 0; i < n; ++i) {
        for (i64 j = 0; j < i; ++j) {
            const double d = dist(x[i], y[i], x[j], y[j]);
            if (d > cutoff) continue;
            const double frac = d / width;
            std::size_t k = d > 0.0 && frac == std::floor(frac) ? static_cast<std::size_t>(frac) - 1
                                                                : static_cast<std::size_t>(std::floor(frac));
            k = std::min(k, n_est - 1);
            const double dz = z[i] - z[j];
            gsum[k] += dz * dz;
            dsum[k] += d;
            ++nh[k];
        }
    }
    SampleVariogram s;
    for (std::size_t k = 0; k < n_est; ++k) {
        if (nh[k] == 0) continue;
        const double c = static_cast<double>(nh[k]);
        s.np.push_back(c);
        s.dist.push_back(dsum[k] / c);
        s.gamma.push_back(gsum[k] / (2.0 * c));
    }
    return s;
}

VarioFit fit_variogram(const SampleVariogram& s, VarioModel init, bool fit_nugget, bool fit_psill, bool fit_range) {
    const VarioFit f = wls_fit(s, init, fit_nugget, fit_psill, fit_range);
    if (f.singular || (f.model.nugget >= 0.0 && f.model.psill >= 0.0)) return f;
    VarioModel m = f.model;
    // (R's fit.sills = model$psill > 0: the positive sills fitted again)
    const bool fn = m.nugget > 0.0, fp = m.psill > 0.0;
    m.nugget = std::max(m.nugget, 0.0);
    m.psill = std::max(m.psill, 0.0);
    m.range = init.range;
    return fit_variogram(s, m, fn, fp, fit_range);
}

void idw(const double* x, const double* y, const double* z, i64 n, const double* px, const double* py, i64 m,
         double power, i64 nmax, double* out, int threads) {
    const bool all = nmax <= 0 || nmax >= n;
    std::vector<i64> every;
    if (all) {
        every.resize(static_cast<std::size_t>(n));
        std::iota(every.begin(), every.end(), i64{0});
    }
#pragma omp parallel num_threads(threads)
    {
        std::vector<i64> idx;
        std::vector<double> d2;
#pragma omp for schedule(dynamic, 256)
        for (i64 c = 0; c < m; ++c) {
            if (all) {
                out[c] = idw_over(x, y, z, every.data(), n, px[c], py[c], power);
            } else {
                nearest(x, y, n, px[c], py[c], nmax, idx, d2);
                out[c] = idw_over(x, y, z, idx.data(), nmax, px[c], py[c], power);
            }
        }
    }
}

bool krige(const double* x, const double* y, const double* z, i64 n, const VarioModel& model, const double* px,
           const double* py, i64 m, i64 nmax, double* pred, double* var, int threads) {
    if (nmax <= 0 || nmax >= n) {
        Kriger k;
        if (!k.build(x, y, z, n, model)) return false;
#pragma omp parallel num_threads(threads)
        {
            std::vector<double> c0, t;
#pragma omp for schedule(dynamic, 64)
            for (i64 c = 0; c < m; ++c) k.at(x, y, n, model, px[c], py[c], c0, t, pred[c], var[c]);
        }
        return true;
    }
    std::atomic<bool> ok{true};
#pragma omp parallel num_threads(threads)
    {
        std::vector<i64> idx;
        std::vector<double> d2, sx, sy, sz, c0, t;
        Kriger k;
#pragma omp for schedule(dynamic, 64)
        for (i64 c = 0; c < m; ++c) {
            if (!ok.load(std::memory_order_relaxed)) continue;
            nearest(x, y, n, px[c], py[c], nmax, idx, d2);
            sx.resize(static_cast<std::size_t>(nmax));
            sy.resize(static_cast<std::size_t>(nmax));
            sz.resize(static_cast<std::size_t>(nmax));
            for (i64 j = 0; j < nmax; ++j) {
                sx[j] = x[idx[j]];
                sy[j] = y[idx[j]];
                sz[j] = z[idx[j]];
            }
            if (!k.build(sx.data(), sy.data(), sz.data(), nmax, model)) {
                ok.store(false, std::memory_order_relaxed);
                continue;
            }
            k.at(sx.data(), sy.data(), nmax, model, px[c], py[c], c0, t, pred[c], var[c]);
        }
    }
    return ok.load();
}

}  // namespace stats
