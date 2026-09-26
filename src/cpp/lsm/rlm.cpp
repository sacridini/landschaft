// SPDX-License-Identifier: GPL-3.0-or-later
//
// The slope of R's lm(y ~ x), operation for operation: lm.fit calls LINPACK's
// dqrls (dqrdc2: Householder QR with R's limited pivoting, tol = 1e-7; dqrsl
// with job 1110), which call R's reference BLAS (ddot unrolled by 5, daxpy by
// 4, dscal by 5, LAPACK 3.12's dnrm2). gfortran compiles them without fused
// multiply-adds, so every step here rounds as in R (src/appl/dqrdc2.f,
// dqrsl.f, dqrls.f and src/extra/blas of R 4.4).
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "lsm/landscape.hpp"

namespace lsm {

namespace {

double ddot(int n, const double* dx, const double* dy) {
    double dtemp = 0.0;
    if (n <= 0) return 0.0;
    const int m = n % 5;
    for (int i = 0; i < m; ++i) dtemp = dtemp + dx[i] * dy[i];
    if (n < 5) return dtemp;
    for (int i = m; i < n; i += 5) {
        dtemp = dtemp + dx[i] * dy[i] + dx[i + 1] * dy[i + 1] + dx[i + 2] * dy[i + 2] + dx[i + 3] * dy[i + 3] +
                dx[i + 4] * dy[i + 4];
    }
    return dtemp;
}

void daxpy(int n, double da, const double* dx, double* dy) {
    if (n <= 0 || da == 0.0) return;
    const int m = n % 4;
    for (int i = 0; i < m; ++i) dy[i] = dy[i] + da * dx[i];
    if (n < 4) return;
    for (int i = m; i < n; i += 4) {
        dy[i] = dy[i] + da * dx[i];
        dy[i + 1] = dy[i + 1] + da * dx[i + 1];
        dy[i + 2] = dy[i + 2] + da * dx[i + 2];
        dy[i + 3] = dy[i + 3] + da * dx[i + 3];
    }
}

void dscal(int n, double da, double* dx) {
    if (n <= 0 || da == 1.0) return;
    const int m = n % 5;
    for (int i = 0; i < m; ++i) dx[i] = da * dx[i];
    if (n < 5) return;
    for (int i = m; i < n; i += 5) {
        dx[i] = da * dx[i];
        dx[i + 1] = da * dx[i + 1];
        dx[i + 2] = da * dx[i + 2];
        dx[i + 3] = da * dx[i + 3];
        dx[i + 4] = da * dx[i + 4];
    }
}

// LAPACK 3.12's DNRM2 (Blue's scaling), for incx = 1.
double dnrm2(int n, const double* x) {
    if (n <= 0) return 0.0;
    const double tsml = std::ldexp(1.0, -511);
    const double tbig = std::ldexp(1.0, 486);
    const double ssml = std::ldexp(1.0, 537);
    const double sbig = std::ldexp(1.0, -538);
    const double maxn = std::numeric_limits<double>::max();
    bool notbig = true;
    double asml = 0.0, amed = 0.0, abig = 0.0;
    for (int i = 0; i < n; ++i) {
        const double ax = std::fabs(x[i]);
        if (ax > tbig) {
            abig = abig + (ax * sbig) * (ax * sbig);
            notbig = false;
        } else if (ax < tsml) {
            if (notbig) asml = asml + (ax * ssml) * (ax * ssml);
        } else {
            amed = amed + ax * ax;
        }
    }
    double scl = 1.0, sumsq = 0.0;
    if (abig > 0.0) {
        if (amed > 0.0 || amed > maxn || amed != amed) abig = abig + (amed * sbig) * sbig;
        scl = 1.0 / sbig;
        sumsq = abig;
    } else if (asml > 0.0) {
        if (amed > 0.0 || amed > maxn || amed != amed) {
            amed = std::sqrt(amed);
            asml = std::sqrt(asml) / ssml;
            const double ymin = asml > amed ? amed : asml;
            const double ymax = asml > amed ? asml : amed;
            scl = 1.0;
            sumsq = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = 1.0 / ssml;
            sumsq = asml;
        }
    } else {
        scl = 1.0;
        sumsq = amed;
    }
    return scl * std::sqrt(sumsq);
}

}  // namespace

// Slope of R's lm(y ~ x) for n points: NaN when x is (numerically) constant
// (rank 1, R's NA).
double r_lm_slope(const double* xv, const double* yv, int n) {
    constexpr int p = 2;
    const double tol = 1e-7;
    // model matrix (column-major, ldx = n): intercept, x
    std::vector<double> x(static_cast<std::size_t>(n) * p);
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0;
        x[n + i] = xv[i];
    }
    auto X = [&](int i, int j) -> double& { return x[static_cast<std::size_t>(j) * n + i]; };  // 0-based
    double qraux[p], work1[p], work2[p];
    int jpvt[p] = {1, 2};
    // dqrdc2
    for (int j = 0; j < p; ++j) {
        qraux[j] = dnrm2(n, &X(0, j));
        work1[j] = qraux[j];
        work2[j] = qraux[j];
        if (work2[j] == 0.0) work2[j] = 1.0;
    }
    const int lup = std::min(n, p);
    int k = p + 1;
    for (int l = 1; l <= lup; ++l) {
        // move nearly collinear columns to the end
        while (!(l >= k || qraux[l - 1] >= work2[l - 1] * tol)) {
            const int lp1 = l + 1;
            for (int i = 0; i < n; ++i) {
                const double t = X(i, l - 1);
                for (int j = lp1; j <= p; ++j) X(i, j - 2) = X(i, j - 1);
                X(i, p - 1) = t;
            }
            const int ii = jpvt[l - 1];
            const double t = qraux[l - 1], tt = work1[l - 1], ttt = work2[l - 1];
            for (int j = lp1; j <= p; ++j) {
                jpvt[j - 2] = jpvt[j - 1];
                qraux[j - 2] = qraux[j - 1];
                work1[j - 2] = work1[j - 1];
                work2[j - 2] = work2[j - 1];
            }
            jpvt[p - 1] = ii;
            qraux[p - 1] = t;
            work1[p - 1] = tt;
            work2[p - 1] = ttt;
            k = k - 1;
        }
        if (l == n) continue;
        double nrmxl = dnrm2(n - l + 1, &X(l - 1, l - 1));
        if (nrmxl == 0.0) continue;
        if (X(l - 1, l - 1) != 0.0) nrmxl = std::copysign(std::fabs(nrmxl), X(l - 1, l - 1));
        dscal(n - l + 1, 1.0 / nrmxl, &X(l - 1, l - 1));
        X(l - 1, l - 1) = 1.0 + X(l - 1, l - 1);
        const int lp1 = l + 1;
        for (int j = lp1; j <= p; ++j) {
            const double t = -ddot(n - l + 1, &X(l - 1, l - 1), &X(l - 1, j - 1)) / X(l - 1, l - 1);
            daxpy(n - l + 1, t, &X(l - 1, l - 1), &X(l - 1, j - 1));
            if (qraux[j - 1] == 0.0) continue;
            const double r = std::fabs(X(l - 1, j - 1)) / qraux[j - 1];
            double tt = 1.0 - r * r;
            tt = std::max(tt, 0.0);
            if (std::fabs(tt) < 1e-6) {
                qraux[j - 1] = dnrm2(n - l, &X(l, j - 1));
                work1[j - 1] = qraux[j - 1];
            } else {
                qraux[j - 1] = qraux[j - 1] * std::sqrt(tt);
            }
        }
        qraux[l - 1] = X(l - 1, l - 1);
        X(l - 1, l - 1) = -nrmxl;
    }
    k = std::min(k - 1, n);
    if (k < p) return std::numeric_limits<double>::quiet_NaN();  // rank deficient: R gives NA
    // dqrsl, job 1110: qty, b
    std::vector<double> qty(yv, yv + n);
    const int ju = std::min(k, n - 1);
    for (int j = 1; j <= ju; ++j) {
        if (qraux[j - 1] == 0.0) continue;
        const double temp = X(j - 1, j - 1);
        X(j - 1, j - 1) = qraux[j - 1];
        const double t = -ddot(n - j + 1, &X(j - 1, j - 1), &qty[j - 1]) / X(j - 1, j - 1);
        daxpy(n - j + 1, t, &X(j - 1, j - 1), &qty[j - 1]);
        X(j - 1, j - 1) = temp;
    }
    double b[p] = {qty[0], qty[1]};
    for (int jj = 1; jj <= k; ++jj) {
        const int j = k - jj + 1;
        if (X(j - 1, j - 1) == 0.0) return std::numeric_limits<double>::quiet_NaN();
        b[j - 1] = b[j - 1] / X(j - 1, j - 1);
        if (j == 1) continue;
        daxpy(j - 1, -b[j - 1], &X(0, j - 1), b);
    }
    // unpivot: jpvt is the identity at full rank
    return jpvt[1] == 2 ? b[1] : b[0];
}

}  // namespace lsm
