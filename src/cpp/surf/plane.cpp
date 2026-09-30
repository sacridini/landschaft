// SPDX-License-Identifier: GPL-3.0-or-later
//
// Trend surfaces as geodiv's fitplane / bestfitplane compute them: the least
// squares polynomial of spatial::surf.ls (B. D. Ripley's krc.c: coordinates
// scaled to [-1, 1], a Householder QR, back substitution) fitted to the cell
// centres of the values that are not NA, predicted in every cell.
//
// The same operations in the same order as krc.c, arranged for speed: the
// scaled coordinates per row and column (dscale is separable), the regression
// matrix filled in parallel, and in each Householder step the columns'
// products (each summed in krc.c's order) and updates in parallel; the
// reflections are kept in the columns they came from (krc.c's `nu`).

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "lsm/rmath.hpp"
#include "surf/surface.hpp"

namespace surf {

namespace {

using Index = std::size_t;

// krc.c's powi: x^i by repeated multiplication.
inline double powi(double x, int i) {
    double t = 1.0;
    for (int j = 1; j <= i; ++j) t *= x;
    return t;
}

// VR_frset + dscale: a coordinate scaled to [-1, 1] on the range of the data.
inline double dscale(double o, double lo, double hi) {
    const double mid = (hi + lo) / 2;
    return (o - mid) / (hi - mid);
}

// The index in R's packed upper triangle (by columns) of krc.c's r[i1 - 1] at
// step k, column i >= k: i1 starts at (k+1)(k+2)/2 and grows by k+1, k+2, ...
inline Index packed(int k, int i) {
    const Index d = static_cast<Index>(i - k);
    return static_cast<Index>((k + 1) * (k + 2) / 2 - 1) + d * static_cast<Index>(k + 1) + d * (d - 1) / 2;
}

// krc.c's householder on F (n x m, by columns). On return column k of f holds
// the reflection nu of step k (rows k..n-1); b and r as krc.c.
int householder(std::vector<double>& f, double* b, double* r, i64 n, int m) {
    std::vector<double> nu(static_cast<Index>(n));
    for (int k = 0; k < m; ++k) {
        double* fk = f.data() + static_cast<i64>(k) * n;
        double c2 = std::fabs(fk[k]);
        for (i64 i = k + 1; i < n; ++i) c2 = std::max(c2, std::fabs(fk[i]));
        if (c2 < 1e-6) return k + 1;
        double c1 = 0.0;
        for (i64 i = k; i < n; ++i) {
            nu[static_cast<Index>(i)] = fk[i] / c2;
            c1 += nu[static_cast<Index>(i)] * nu[static_cast<Index>(i)];
        }
        c1 = std::sqrt(c1);
        b[k] = c1 * (c1 + std::fabs(nu[static_cast<Index>(k)]));
        if (nu[static_cast<Index>(k)] >= 0.0)
            nu[static_cast<Index>(k)] += c1;
        else
            nu[static_cast<Index>(k)] -= c1;
        const double* v = nu.data();
        const double nuk = v[k], bk = b[k];
        // every column i >= k: its product with nu (in order), r, the update;
        // column k itself is only needed for its r
#pragma omp parallel for schedule(dynamic, 1) if (n > 32768 && m - k > 1)
        for (int i = k; i < m; ++i) {
            double* fi = f.data() + static_cast<i64>(i) * n;
            double c = 0.0;
            for (i64 j = k; j < n; ++j) c += v[j] * fi[j];
            c /= bk;
            r[packed(k, i)] = fi[k] - c * nuk;
            if (i == k) continue;
            for (i64 j = k; j < n; ++j) fi[j] -= c * v[j];
        }
        std::copy(nu.begin() + k, nu.end(), fk + k);
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

// krc.c's house_rhs: the reflections (columns of `nu`) applied to z, then
// R beta = Q'z.
void house_rhs(const std::vector<double>& nu, const double* b, const double* r, i64 n, int m, const double* z,
               double* beta) {
    std::vector<double> w(z, z + n);
    for (int k = 0; k < m; ++k) {
        const double* v = nu.data() + static_cast<i64>(k) * n;
        double sum = 0.0;
        for (i64 i = k; i < n; ++i) sum += v[i] * w[static_cast<Index>(i)];
        sum /= b[k];
#pragma omp parallel for schedule(static) if (n > 65536)
        for (i64 i = k; i < n; ++i) w[static_cast<Index>(i)] -= sum * v[i];
    }
    bsolv(beta, w.data(), m, r);
}

}  // namespace

std::vector<double> fit_plane(const Grid& g, int order, int* ifail) {
    const int np = order;
    const int npar = (np + 1) * (np + 2) / 2;
    const int ny = g.ny, nx = g.nx;
    // the cells with a value (row order), as crds(x)[!is.na(z)], and the rows
    // and columns that have one (for range(x), range(y))
    std::vector<i64> first(static_cast<Index>(ny) + 1, 0);
    std::vector<char> row_used(static_cast<Index>(ny), 0), col_used(static_cast<Index>(nx), 0);
    for (int r = 0; r < ny; ++r) {
        i64 k = 0;
        for (int c = 0; c < nx; ++c)
            if (!std::isnan(g.z[static_cast<i64>(r) * nx + c])) {
                ++k;
                col_used[static_cast<Index>(c)] = 1;
            }
        row_used[static_cast<Index>(r)] = k > 0;
        first[static_cast<Index>(r) + 1] = first[static_cast<Index>(r)] + k;
    }
    const i64 n = first[static_cast<Index>(ny)];
    std::vector<double> cx(static_cast<Index>(nx)), cy(static_cast<Index>(ny));
    for (int c = 0; c < nx; ++c) cx[static_cast<Index>(c)] = g.x(c);
    for (int r = 0; r < ny; ++r) cy[static_cast<Index>(r)] = g.y(r);
    // surf.ls: range(x), range(y) of the cells with a value
    double xl = std::numeric_limits<double>::infinity(), xu = -xl, yl = xl, yu = -xl;
    for (int c = 0; c < nx; ++c)
        if (col_used[static_cast<Index>(c)]) {
            xl = std::min(xl, cx[static_cast<Index>(c)]);
            xu = std::max(xu, cx[static_cast<Index>(c)]);
        }
    for (int r = 0; r < ny; ++r)
        if (row_used[static_cast<Index>(r)]) {
            yl = std::min(yl, cy[static_cast<Index>(r)]);
            yu = std::max(yu, cy[static_cast<Index>(r)]);
        }
    std::vector<double> sx(static_cast<Index>(nx)), sy(static_cast<Index>(ny));
    for (int c = 0; c < nx; ++c) sx[static_cast<Index>(c)] = dscale(cx[static_cast<Index>(c)], xl, xu);
    for (int r = 0; r < ny; ++r) sy[static_cast<Index>(r)] = dscale(cy[static_cast<Index>(r)], yl, yu);
    // VR_fmat: the regression matrix, by columns x^j y^i; and the values
    std::vector<double> f(static_cast<Index>(n) * static_cast<Index>(npar));
    std::vector<double> zs(static_cast<Index>(n));
#pragma omp parallel for schedule(static) if (n > 16384)
    for (int r = 0; r < ny; ++r) {
        i64 k = first[static_cast<Index>(r)];
        const double ys = sy[static_cast<Index>(r)];
        for (int c = 0; c < nx; ++c) {
            const double v = g.z[static_cast<i64>(r) * nx + c];
            if (std::isnan(v)) continue;
            zs[static_cast<Index>(k)] = v;
            const double xs = sx[static_cast<Index>(c)];
            i64 col = 0;
            for (int i = 0; i <= np; ++i)
                for (int j = 0; j <= np - i; ++j) f[static_cast<Index>(col++ * n + k)] = powi(xs, j) * powi(ys, i);
            ++k;
        }
    }
    // VR_ls
    std::array<double, 28> b{};
    std::vector<double> rr(static_cast<Index>(npar * (npar + 1) / 2));
    std::vector<double> beta(static_cast<Index>(npar), 0.0);
    const int fail = householder(f, b.data(), rr.data(), n, npar);
    if (fail == 0) house_rhs(f, b.data(), rr.data(), n, npar, zs.data(), beta.data());
    if (ifail) *ifail = fail;
    f = std::vector<double>();
    // predict.trls (VR_valn) in every cell
    std::vector<double> out(static_cast<Index>(g.size()));
#pragma omp parallel for schedule(static) if (g.size() > 16384)
    for (int r = 0; r < ny; ++r) {
        const double ys = sy[static_cast<Index>(r)];
        for (int c = 0; c < nx; ++c) {
            const double xs = sx[static_cast<Index>(c)];
            double sum = 0.0;
            int i1 = 0;
            for (int i = 0; i <= np; ++i)
                for (int j = 0; j <= np - i; ++j) sum += beta[static_cast<Index>(i1++)] * powi(xs, j) * powi(ys, i);
            out[static_cast<Index>(static_cast<i64>(r) * nx + c)] = sum;
        }
    }
    return out;
}

std::vector<double> best_fit_plane(const Grid& g, int* order) {
    std::array<std::vector<double>, 4> fits;
    std::array<double, 4> err{};
    for (int k = 0; k < 4; ++k) {
        fits[static_cast<Index>(k)] = fit_plane(g, k);
        // abs(mean(xmat - fit, na.rm = TRUE)): the matrix by columns
        const auto& fit = fits[static_cast<Index>(k)];
        std::vector<double> e;
        e.reserve(fit.size());
        for (int c = 0; c < g.nx; ++c)
            for (int r = 0; r < g.ny; ++r) {
                const i64 i = static_cast<i64>(r) * g.nx + c;
                if (!std::isnan(g.z[i])) e.push_back(g.z[i] - fit[static_cast<Index>(i)]);
            }
        err[static_cast<Index>(k)] = std::fabs(lsm::r_mean(e.data(), static_cast<i64>(e.size())));
    }
    // which(meanerr == min(meanerr, na.rm = TRUE)), the first one
    double lo = std::numeric_limits<double>::infinity();
    for (double e : err)
        if (e < lo) lo = e;
    int best = 0;
    while (best < 3 && !(err[static_cast<Index>(best)] == lo)) ++best;
    if (order) *order = best;
    return std::move(fits[static_cast<Index>(best)]);
}

}  // namespace surf
