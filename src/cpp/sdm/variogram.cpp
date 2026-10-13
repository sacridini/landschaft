// SPDX-License-Identifier: GPL-3.0-or-later
//
// The pair loop of gstat's semivariogram (sem.c) for blockCV's
// cv_spatial_autocor, without storing the pairs.

#include "sdm/variogram.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>

namespace sdm {

namespace {

// gstat_gcdist (gcdist.c, from sp): great-circle distance in km.
double gcdist(double lon1, double lon2, double lat1, double lat2) {
    if (lon1 == lon2 && lat1 == lat2) return 0.0;
    const double rad = 3.14159265358979323846 / 180.0;
    const double a = 6378.137, f = 1.0 / 298.257223563;
    const double F = (lat1 * rad + lat2 * rad) / 2.0;
    const double G = (lat1 * rad - lat2 * rad) / 2.0;
    const double L = (lon1 * rad - lon2 * rad) / 2.0;
    const double sg = std::sin(G), cg = std::cos(G), sf = std::sin(F), cf = std::cos(F), sl = std::sin(L),
                 cl = std::cos(L);
    const double sinG2 = sg * sg, cosG2 = cg * cg, sinF2 = sf * sf, cosF2 = cf * cf, sinL2 = sl * sl,
                 cosL2 = cl * cl;
    const double S = sinG2 * cosL2 + cosF2 * sinL2;
    const double C = cosG2 * cosL2 + sinF2 * sinL2;
    const double w = std::atan(std::sqrt(S / C));
    const double R = std::sqrt(S * C) / w;
    const double D = 2 * w * a;
    const double H1 = (3 * R - 1) / (2 * C);
    const double H2 = (3 * R + 1) / (2 * S);
    return D * (1 + f * H1 * sinF2 * cosG2 - f * H2 * cosF2 * sinG2);
}

}  // namespace

SampleVariogram sample_variogram(const double* x, const double* y, const double* z, i64 n,
                                 const std::vector<double>& bounds, bool lonlat) {
    const std::size_t nb = bounds.size();
    // per-row sums (rows of pairs i > j), added in row order afterwards
    std::vector<double> rnp(static_cast<std::size_t>(n) * nb, 0.0), rd(rnp.size(), 0.0), rg(rnp.size(), 0.0);
#pragma omp parallel for schedule(dynamic, 32)
    for (i64 i = 1; i < n; ++i) {
        double* np_ = &rnp[static_cast<std::size_t>(i) * nb];
        double* d_ = &rd[static_cast<std::size_t>(i) * nb];
        double* g_ = &rg[static_cast<std::size_t>(i) * nb];
        for (i64 j = 0; j < i; ++j) {
            const double d = lonlat ? gcdist(x[i], x[j], y[i], y[j]) : std::hypot(x[i] - x[j], y[i] - y[j]);
            const std::size_t k = static_cast<std::size_t>(std::lower_bound(bounds.begin(), bounds.end(), d) -
                                                           bounds.begin());
            if (k >= nb) continue;
            const double dz = z[i] - z[j];
            np_[k] += 1.0;
            d_[k] += d;
            g_[k] += 0.5 * dz * dz;
        }
    }
    SampleVariogram v{std::vector<double>(nb, 0.0), std::vector<double>(nb, 0.0), std::vector<double>(nb, 0.0)};
    for (i64 i = 1; i < n; ++i)
        for (std::size_t k = 0; k < nb; ++k) {
            v.np[k] += rnp[static_cast<std::size_t>(i) * nb + k];
            v.dist[k] += rd[static_cast<std::size_t>(i) * nb + k];
            v.gamma[k] += rg[static_cast<std::size_t>(i) * nb + k];
        }
    for (std::size_t k = 0; k < nb; ++k)
        if (v.np[k] > 0) {
            v.dist[k] /= v.np[k];
            v.gamma[k] /= v.np[k];
        }
    return v;
}

namespace {

#if defined(__cpp_lib_math_special_functions)
double bessel_k(double nu, double x) { return std::cyl_bessel_k(nu, x); }
#else
// Apple's libc++ has none of C++17's special functions: libstdc++'s
// cyl_bessel_k (__bessel_ik, Temme's method), for K only, nu >= 0.
double bessel_k(double nu, double x) {
    if (std::isnan(nu) || std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
    if (nu < 0.0 || x < 0.0) return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0) return std::numeric_limits<double>::infinity();
    const double pi = 3.14159265358979323846;
    const double eps = std::numeric_limits<double>::epsilon();
    const int max_iter = 15000;
    const double x_min = 2.0;
    const int nl = static_cast<int>(nu + 0.5);
    const double mu = nu - nl;
    const double mu2 = mu * mu;
    const double xi = 1.0 / x;
    const double xi2 = 2.0 * xi;
    double kmu, knu1;
    if (x < x_min) {
        const double x2 = x / 2.0;
        const double pimu = pi * mu;
        const double fact = (std::abs(pimu) < eps ? 1.0 : pimu / std::sin(pimu));
        double d = -std::log(x2);
        double e = mu * d;
        const double fact2 = (std::abs(e) < eps ? 1.0 : std::sinh(e) / e);
        // __gamma_temme
        const double gampl = 1.0 / std::tgamma(1.0 + mu);
        const double gammi = 1.0 / std::tgamma(1.0 - mu);
        const double gam1 =
            std::abs(mu) < std::numeric_limits<double>::min() ? -0.5772156649015328606 : (gammi - gampl) / (2.0 * mu);
        const double gam2 = (gammi + gampl) / 2.0;
        double ff = fact * (gam1 * std::cosh(e) + gam2 * fact2 * d);
        double sum = ff;
        e = std::exp(e);
        double p = e / (2.0 * gampl);
        double q = 1.0 / (2.0 * e * gammi);
        double c = 1.0;
        d = x2 * x2;
        double sum1 = p;
        for (int i = 1; i <= max_iter; ++i) {
            ff = (i * ff + p + q) / (i * i - mu2);
            c *= d / i;
            p /= i - mu;
            q /= i + mu;
            const double del = c * ff;
            sum += del;
            const double del1 = c * (p - i * ff);
            sum1 += del1;
            if (std::abs(del) < eps * std::abs(sum)) break;
        }
        kmu = sum;
        knu1 = sum1 * xi2;
    } else {
        double b = 2.0 * (1.0 + x);
        double d = 1.0 / b;
        double delh = d;
        double h = d;
        double q1 = 0.0;
        double q2 = 1.0;
        const double a1 = 0.25 - mu2;
        double c = a1;
        double q = a1;
        double a = -a1;
        double s = 1.0 + q * delh;
        for (int i = 2; i <= max_iter; ++i) {
            a -= 2 * (i - 1);
            c = -a * c / i;
            const double qnew = (q1 - b * q2) / a;
            q1 = q2;
            q2 = qnew;
            q += c * qnew;
            b += 2.0;
            d = 1.0 / (b + a * d);
            delh = (b * d - 1.0) * delh;
            h += delh;
            const double dels = q * delh;
            s += dels;
            if (std::abs(dels / s) < eps) break;
        }
        h = a1 * h;
        kmu = std::sqrt(pi / (2.0 * x)) * std::exp(-x) / s;
        knu1 = kmu * (mu + x + 0.5 - h) * xi;
    }
    for (int i = 1; i <= nl; ++i) {
        const double knutemp = (mu + i) * xi2 * knu1 + kmu;
        kmu = knu1;
        knu1 = knutemp;
    }
    return kmu;
}
#endif

enum class Model { Sph, Exp, Gau, Ste };

// gstat's unit semivariances (vario_fn.c), 0 at distance 0
double unit(Model m, double h, double r, double kappa) {
    if (h == 0.0) return 0.0;
    switch (m) {
        case Model::Sph: {
            if (h >= r) return 1.0;
            const double hr = h / r;
            return hr * (1.5 - 0.5 * hr * hr);
        }
        case Model::Exp:
            return 1.0 - std::exp(-h / r);
        case Model::Gau: {
            const double hr = h / r;
            return 1.0 - std::exp(-(hr * hr));
        }
        default: {  // fn_matern2: Matern, M. Stein's parameterisation
            const double arg = 2.0 * std::sqrt(kappa) * (h / r);
            const double bes = bessel_k(kappa, arg);
            if (!std::isfinite(bes)) return 0.0;
            if (bes == 0.0) return 1.0;
            const double mult = std::pow(2.0, 1.0 - kappa) / std::tgamma(kappa) * std::pow(arg, kappa);
            if (!std::isfinite(mult)) return 1.0;
            return 1.0 - bes * mult;
        }
    }
}

// d semivariance / d range: gstat's analytic derivatives, numerical (da_general) for Ste
double d_range(Model m, double h, double r, double kappa, double sill) {
    switch (m) {
        case Model::Sph: {
            if (h > r) return 0.0;
            const double hr2 = h / (r * r);
            return sill * (1.5 * hr2 * (-1.0 + h * hr2));
        }
        case Model::Exp: {
            const double hr = -h / r;
            return sill * ((hr / r) * std::exp(hr));
        }
        case Model::Gau: {
            const double hr = h / r;
            return sill * ((-hr / r) * std::exp(-(hr * hr)));
        }
        default: {
            const double rr = std::max(1e-20, r);
            const double lo = unit(m, h, rr * 1.001, kappa), hi = unit(m, h, rr * 0.999, kappa);
            return sill * (lo - hi) / (2.0 * rr * 0.001);
        }
    }
}

// One Gauss-Newton run over the free parameters (nugget, psill, range).
void gauss_newton(Model m, double kappa, const double* dist, const double* gamma, const double* w, i64 n,
                  double p[3], const bool free[3], double* sse_out) {
    auto sserr = [&](const double* q) {
        double s = 0.0;
        for (i64 i = 0; i < n; ++i) {
            const double res = gamma[i] - ((dist[i] > 0 ? q[0] : 0.0) + q[1] * unit(m, dist[i], q[2], kappa));
            s += res * res * w[i];
        }
        return s;
    };
    int idx[3], nf = 0;
    for (int k = 0; k < 3; ++k)
        if (free[k]) idx[nf++] = k;
    double old = std::numeric_limits<double>::infinity();
    for (int it = 0; it < 200; ++it) {
        // weighted normal equations X'WX beta = X'W res
        double A[3][3] = {}, b[3] = {};
        for (i64 i = 0; i < n; ++i) {
            const double u = unit(m, dist[i], p[2], kappa);
            const double res = gamma[i] - ((dist[i] > 0 ? p[0] : 0.0) + p[1] * u);
            const double full[3] = {dist[i] > 0 ? 1.0 : 0.0, u, d_range(m, dist[i], p[2], kappa, p[1])};
            double x[3] = {};
            for (int a = 0; a < nf; ++a) x[a] = full[idx[a]];
            for (int a = 0; a < nf; ++a) {
                b[a] += w[i] * x[a] * res;
                for (int k = 0; k < nf; ++k) A[a][k] += w[i] * x[a] * x[k];
            }
        }
        // Cholesky: gstat (meschach) stops the fit when the matrix is not positive definite
        double L[3][3] = {};
        for (int a = 0; a < nf; ++a) {
            for (int k = 0; k <= a; ++k) {
                double s = A[a][k];
                for (int q = 0; q < k; ++q) s -= L[a][q] * L[k][q];
                if (a == k) {
                    if (!(s > 0.0)) {
                        *sse_out = std::isfinite(old) ? old : sserr(p);
                        return;
                    }
                    L[a][a] = std::sqrt(s);
                } else {
                    L[a][k] = s / L[k][k];
                }
            }
        }
        double yv[3] = {}, beta[3] = {};
        for (int a = 0; a < nf; ++a) {
            double s = b[a];
            for (int q = 0; q < a; ++q) s -= L[a][q] * yv[q];
            yv[a] = s / L[a][a];
        }
        for (int a = nf - 1; a >= 0; --a) {
            double s = yv[a];
            for (int q = a + 1; q < nf; ++q) s -= L[q][a] * beta[q];
            beta[a] = s / L[a][a];
        }
        double bb = 0.0, pp = 0.0;
        for (int a = 0; a < nf; ++a) {
            if (!std::isfinite(beta[a])) {
                *sse_out = std::isfinite(old) ? old : sserr(p);
                return;
            }
            bb += beta[a] * beta[a];
            pp += p[idx[a]] * p[idx[a]];
        }
        const double s = bb / pp;  // gstat's v_norm2 is the sum of squares
        const bool bounded = s > 0.2;
        for (int a = 0; a < nf; ++a) p[idx[a]] += bounded ? beta[a] * (0.2 / s) : beta[a];
        const double sse = sserr(p);
        double step = old - sse;
        if (sse > 0) step /= sse;
        old = sse;
        if (step < 1e-5 && step >= 0 && !bounded) break;
    }
    *sse_out = old;
}

}  // namespace

VariogramFit fit_variogram(const char* model, double kappa, const double* dist, const double* gamma,
                           const double* np, i64 n, double nugget, double psill, double range) {
    Model m;
    if (std::strcmp(model, "Sph") == 0) m = Model::Sph;
    else if (std::strcmp(model, "Exp") == 0) m = Model::Exp;
    else if (std::strcmp(model, "Gau") == 0) m = Model::Gau;
    else if (std::strcmp(model, "Ste") == 0) m = Model::Ste;
    else return {0, 0, 0, 0, false};
    std::vector<double> w(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) w[static_cast<std::size_t>(i)] = np[i] / (dist[i] * dist[i]);
    double p[3] = {nugget, psill, range};
    bool free[3] = {true, true, true};
    for (;;) {
        double sse = 0.0;
        gauss_newton(m, kappa, dist, gamma, w.data(), n, p, free, &sse);
        bool neg = false;
        for (int k = 0; k < 2; ++k)
            if (free[k] && p[k] < 0) {
                neg = true;
                p[k] = 0.0;
                free[k] = false;
            }
        if (!neg) return {p[0], p[1], p[2], sse, true};
        p[2] = range;
    }
}

}  // namespace sdm
