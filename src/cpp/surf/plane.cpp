// SPDX-License-Identifier: GPL-3.0-or-later
//
// Trend surfaces as geodiv's fitplane / bestfitplane compute them: the least
// squares polynomial of spatial::surf.ls (B. D. Ripley's krc.c: coordinates
// scaled to [-1, 1], a Householder QR, back substitution) fitted to the cell
// centres of the values that are not NA, predicted in every cell.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "lsm/rmath.hpp"
#include "surf/surface.hpp"

namespace surf {

namespace {

// krc.c's powi: x^i by repeated multiplication.
inline double powi(double x, int i) {
    double t = 1.0;
    for (int j = 1; j <= i; ++j) t *= x;
    return t;
}

// VR_frset + dscale: (x, y) scaled to [-1, 1] on the range of the data.
struct Scale {
    double xl, xu, yl, yu;
    void operator()(double xo, double yo, double& xs, double& ys) const {
        const double xl2 = (xu + xl) / 2;
        const double yl2 = (yu + yl) / 2;
        xs = (xo - xl2) / (xu - xl2);
        ys = (yo - yl2) / (yu - yl2);
    }
};

// krc.c's householder: F (n x m, by columns) reduced to R (upper triangle,
// packed by columns), the reflections kept in nu and b.
int householder(std::vector<double>& f, std::vector<double>& nu, double* b, double* r, i64 n, int m) {
    for (int k = 0; k < m; ++k) {
        const i64 k1 = static_cast<i64>(k) * n;
        const i64 k3 = k1 + k;
        double c2 = std::fabs(f[k3]);
        for (i64 i = k + 1; i < n; ++i) c2 = std::max(c2, std::fabs(f[i + k1]));
        if (c2 < 1e-6) return k + 1;
        double c1 = 0.0;
        for (i64 i = k; i < n; ++i) {
            const i64 k2 = k1 + i;
            nu[k2] = f[k2] / c2;
            c1 += nu[k2] * nu[k2];
        }
        c1 = std::sqrt(c1);
        b[k] = c1 * (c1 + std::fabs(nu[k3]));
        if (nu[k3] >= 0.0)
            nu[k3] += c1;
        else
            nu[k3] -= c1;
        int i1 = (k + 1) * (k + 2) / 2;
        int i2 = k + 1;
        for (int i = k; i < m; ++i) {
            const i64 i3 = static_cast<i64>(i) * n;
            c1 = 0.0;
            for (i64 j = k; j < n; ++j) c1 += nu[k1 + j] * f[i3 + j];
            c1 /= b[k];
            r[i1 - 1] = f[i3 + k] - c1 * nu[k3];
            for (i64 j = k; j < n; ++j) f[i3 + j] -= c1 * nu[k1 + j];
            i1 += i2;
            ++i2;
        }
    }
    return 0;
}

// krc.c's bsolv: U x = y, U upper triangular packed by columns.
void bsolv(double* x, const double* y, int n, const double* u) {
    int i1 = n * (n + 1) / 2 - 1;
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        double sum = 0.0;
        int ic = i1;
        for (int j = i + 1; j < n; ++j) {
            ic += j;
            sum += x[j] * u[ic];
        }
        x[i] = (x[i] - sum) / u[i1];
        i1 -= i + 1;
    }
}

// krc.c's house_rhs: the reflections applied to z, then R beta = Q'z.
void house_rhs(const std::vector<double>& nu, const double* b, const double* r, i64 n, int m, const double* z,
               double* beta) {
    std::vector<double> w(z, z + n);
    for (int k = 0; k < m; ++k) {
        double sum = 0.0;
        const i64 k1 = static_cast<i64>(k) * n;
        for (i64 i = k; i < n; ++i) sum += nu[i + k1] * w[i];
        sum /= b[k];
        for (i64 i = k; i < n; ++i) w[i] -= sum * nu[i + k1];
    }
    bsolv(beta, w.data(), m, r);
}

}  // namespace

std::vector<double> fit_plane(const Grid& g, int order, int* ifail) {
    const int np = order;
    const int npar = (np + 1) * (np + 2) / 2;
    // the cell centres of the values (row order), as crds(x)[!is.na(z)]
    std::vector<double> xs, ys, zs;
    const i64 size = g.size();
    xs.reserve(static_cast<std::size_t>(size));
    ys.reserve(static_cast<std::size_t>(size));
    zs.reserve(static_cast<std::size_t>(size));
    std::vector<double> cx(static_cast<std::size_t>(g.nx)), cy(static_cast<std::size_t>(g.ny));
    for (int c = 0; c < g.nx; ++c) cx[static_cast<std::size_t>(c)] = g.x(c);
    for (int r = 0; r < g.ny; ++r) cy[static_cast<std::size_t>(r)] = g.y(r);
    for (int r = 0; r < g.ny; ++r)
        for (int c = 0; c < g.nx; ++c) {
            const double v = g.z[static_cast<i64>(r) * g.nx + c];
            if (std::isnan(v)) continue;
            xs.push_back(cx[static_cast<std::size_t>(c)]);
            ys.push_back(cy[static_cast<std::size_t>(r)]);
            zs.push_back(v);
        }
    const i64 n = static_cast<i64>(zs.size());
    // surf.ls: range(x), range(y) (NA without values, as R's range)
    Scale scale{};
    if (n > 0) {
        scale.xl = *std::min_element(xs.begin(), xs.end());
        scale.xu = *std::max_element(xs.begin(), xs.end());
        scale.yl = *std::min_element(ys.begin(), ys.end());
        scale.yu = *std::max_element(ys.begin(), ys.end());
    }
    // VR_fmat: the regression matrix, by columns x^j y^i
    std::vector<double> f(static_cast<std::size_t>(n * npar));
    {
        std::vector<double> sx(static_cast<std::size_t>(n)), sy(static_cast<std::size_t>(n));
        for (i64 k = 0; k < n; ++k) scale(xs[k], ys[k], sx[k], sy[k]);
        i64 k1 = 0;
        for (int i = 0; i <= np; ++i)
            for (int j = 0; j <= np - i; ++j)
                for (i64 k = 0; k < n; ++k) f[static_cast<std::size_t>(k1++)] = powi(sx[k], j) * powi(sy[k], i);
    }
    // VR_ls
    std::array<double, 28> b{};
    std::vector<double> rr(static_cast<std::size_t>(npar * (npar + 1) / 2));
    std::vector<double> beta(static_cast<std::size_t>(npar), 0.0);
    std::vector<double> nu(static_cast<std::size_t>(n * npar), 0.0);
    const int fail = householder(f, nu, b.data(), rr.data(), n, npar);
    if (fail == 0) house_rhs(nu, b.data(), rr.data(), n, npar, zs.data(), beta.data());
    if (ifail) *ifail = fail;
    // predict.trls (VR_valn) in every cell
    std::vector<double> out(static_cast<std::size_t>(size));
#pragma omp parallel for schedule(static) if (size > 65536)
    for (int r = 0; r < g.ny; ++r) {
        for (int c = 0; c < g.nx; ++c) {
            double sx, sy;
            scale(cx[static_cast<std::size_t>(c)], cy[static_cast<std::size_t>(r)], sx, sy);
            double sum = 0.0;
            int i1 = 0;
            for (int i = 0; i <= np; ++i)
                for (int j = 0; j <= np - i; ++j) sum += beta[static_cast<std::size_t>(i1++)] * powi(sx, j) * powi(sy, i);
            out[static_cast<std::size_t>(static_cast<i64>(r) * g.nx + c)] = sum;
        }
    }
    return out;
}

std::vector<double> best_fit_plane(const Grid& g, int* order) {
    std::array<std::vector<double>, 4> fits;
    std::array<double, 4> err{};
#pragma omp parallel for schedule(dynamic, 1)
    for (int k = 0; k < 4; ++k) {
        fits[static_cast<std::size_t>(k)] = fit_plane(g, k);
        // abs(mean(xmat - fit, na.rm = TRUE)): the matrix by columns
        const auto& fit = fits[static_cast<std::size_t>(k)];
        std::vector<double> e;
        e.reserve(fit.size());
        for (int c = 0; c < g.nx; ++c)
            for (int r = 0; r < g.ny; ++r) {
                const i64 i = static_cast<i64>(r) * g.nx + c;
                if (!std::isnan(g.z[i])) e.push_back(g.z[i] - fit[static_cast<std::size_t>(i)]);
            }
        err[static_cast<std::size_t>(k)] = std::fabs(lsm::r_mean(e.data(), static_cast<i64>(e.size())));
    }
    // which(meanerr == min(meanerr, na.rm = TRUE)), the first one
    double lo = std::numeric_limits<double>::infinity();
    for (double e : err)
        if (e < lo) lo = e;
    int best = 0;
    while (best < 3 && !(err[static_cast<std::size_t>(best)] == lo)) ++best;
    if (order) *order = best;
    return std::move(fits[static_cast<std::size_t>(best)]);
}

}  // namespace surf
