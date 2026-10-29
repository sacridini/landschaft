// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/distr.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "lsm/rmath.hpp"
#include "move/rnmath.hpp"
#include "move/track.hpp"
#include "nlm/rng.hpp"

namespace move {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.141592653589793238462643383280;
constexpr double kBig = 1.0e+35;  // nmmin's "very large number"

// fminfn(): R evaluates fn at p * parscale (parscale 1) and refuses
// non-finite parameters.
double call(const Objective& fn, const std::vector<double>& p) {
    for (double v : p)
        if (!std::isfinite(v)) throw std::runtime_error("non-finite value supplied by optim");
    return fn(p);
}

}  // namespace

NelderMead nelder_mead(const std::vector<double>& start, const Objective& fn, int maxit) {
    // R's appl/optim.c nmmin(), with alpha 1, beta 0.5, gamma 2, abstol -Inf,
    // intol = reltol = sqrt(DBL_EPSILON); trace off
    const double alpha = 1.0, bet = 0.5, gamm = 2.0;
    const double abstol = -std::numeric_limits<double>::infinity();
    const double intol = 1.490116119384765625e-08;
    const int n = static_cast<int>(start.size());
    NelderMead out;
    std::vector<double> Bvec = start;
    if (maxit <= 0) {
        out.value = call(fn, Bvec);
        out.par = Bvec;
        return out;
    }
    const int n1 = n + 1, C = n + 2;
    std::vector<std::vector<double>> P(static_cast<std::size_t>(n1), std::vector<double>(static_cast<std::size_t>(n + 2)));
    int fail = 0, funcount = 0, H, L = 0;
    double f = call(fn, Bvec);
    if (!std::isfinite(f)) throw std::runtime_error("function cannot be evaluated at initial parameters");
    funcount = 1;
    const double convtol = intol * (std::fabs(f) + intol);
    P[n1 - 1][0] = f;
    for (int i = 0; i < n; i++) P[i][0] = Bvec[i];
    L = 1;
    double size = 0.0, step = 0.0, trystep, oldsize, temp, VH, VL, VR;
    for (int i = 0; i < n; i++) {
        if (0.1 * std::fabs(Bvec[i]) > step) step = 0.1 * std::fabs(Bvec[i]);
    }
    if (step == 0.0) step = 0.1;
    for (int j = 2; j <= n1; j++) {
        for (int i = 0; i < n; i++) P[i][j - 1] = Bvec[i];
        trystep = step;
        while (P[j - 2][j - 1] == Bvec[j - 2]) {
            P[j - 2][j - 1] = Bvec[j - 2] + trystep;
            trystep *= 10;
        }
        size += trystep;
    }
    oldsize = size;
    bool calcvert = true;
    do {
        if (calcvert) {
            for (int j = 0; j < n1; j++) {
                if (j + 1 != L) {
                    for (int i = 0; i < n; i++) Bvec[i] = P[i][j];
                    f = call(fn, Bvec);
                    if (!std::isfinite(f)) f = kBig;
                    funcount++;
                    P[n1 - 1][j] = f;
                }
            }
            calcvert = false;
        }
        VL = P[n1 - 1][L - 1];
        VH = VL;
        H = L;
        for (int j = 1; j <= n1; j++) {
            if (j != L) {
                f = P[n1 - 1][j - 1];
                if (f < VL) {
                    L = j;
                    VL = f;
                }
                if (f > VH) {
                    H = j;
                    VH = f;
                }
            }
        }
        if (VH <= VL + convtol || VL <= abstol) break;
        for (int i = 0; i < n; i++) {
            temp = -P[i][H - 1];
            for (int j = 0; j < n1; j++) temp += P[i][j];
            P[i][C - 1] = temp / n;
        }
        for (int i = 0; i < n; i++) Bvec[i] = (1.0 + alpha) * P[i][C - 1] - alpha * P[i][H - 1];
        f = call(fn, Bvec);
        if (!std::isfinite(f)) f = kBig;
        funcount++;
        VR = f;
        if (VR < VL) {
            P[n1 - 1][C - 1] = f;
            for (int i = 0; i < n; i++) {
                f = gamm * Bvec[i] + (1 - gamm) * P[i][C - 1];
                P[i][C - 1] = Bvec[i];
                Bvec[i] = f;
            }
            f = call(fn, Bvec);
            if (!std::isfinite(f)) f = kBig;
            funcount++;
            if (f < VR) {
                for (int i = 0; i < n; i++) P[i][H - 1] = Bvec[i];
                P[n1 - 1][H - 1] = f;
            } else {
                for (int i = 0; i < n; i++) P[i][H - 1] = P[i][C - 1];
                P[n1 - 1][H - 1] = VR;
            }
        } else {
            if (VR < VH) {
                for (int i = 0; i < n; i++) P[i][H - 1] = Bvec[i];
                P[n1 - 1][H - 1] = VR;
            }
            for (int i = 0; i < n; i++) Bvec[i] = (1 - bet) * P[i][H - 1] + bet * P[i][C - 1];
            f = call(fn, Bvec);
            if (!std::isfinite(f)) f = kBig;
            funcount++;
            if (f < P[n1 - 1][H - 1]) {
                for (int i = 0; i < n; i++) P[i][H - 1] = Bvec[i];
                P[n1 - 1][H - 1] = f;
            } else {
                if (VR >= VH) {
                    calcvert = true;
                    size = 0.0;
                    for (int j = 0; j < n1; j++) {
                        if (j + 1 != L) {
                            for (int i = 0; i < n; i++) {
                                P[i][j] = bet * (P[i][j] - P[i][L - 1]) + P[i][L - 1];
                                size += std::fabs(P[i][j] - P[i][L - 1]);
                            }
                        }
                    }
                    if (size < oldsize) {
                        oldsize = size;
                    } else {
                        fail = 10;
                        break;
                    }
                }
            }
        }
    } while (funcount <= maxit);
    out.value = P[n1 - 1][L - 1];
    out.par.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; i++) out.par[i] = P[i][L - 1];
    if (funcount > maxit) fail = 1;
    out.fail = fail;
    out.fncount = funcount;
    return out;
}

std::vector<double> optim_hessian(const std::vector<double>& par, const Objective& fn, double ndeps) {
    // stats/src/optim.c optimhess() with fmingr()'s numerical gradient
    // (fnscale and parscale 1)
    const int n = static_cast<int>(par.size());
    std::vector<double> dpar = par, df1(static_cast<std::size_t>(n)), df2(static_cast<std::size_t>(n));
    std::vector<double> ans(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    auto fmingr = [&](const std::vector<double>& p, std::vector<double>& df) {
        std::vector<double> x = p;
        for (int i = 0; i < n; i++) {
            const double eps = ndeps;
            x[i] = (p[i] + eps) * 1.0;
            const double val1 = fn(x);
            x[i] = (p[i] - eps) * 1.0;
            const double val2 = fn(x);
            df[i] = (val1 - val2) / (2 * eps);
            if (!std::isfinite(df[i]))
                throw std::runtime_error("non-finite finite-difference value [" + std::to_string(i + 1) + "]");
            x[i] = p[i] * 1.0;
        }
    };
    for (int i = 0; i < n; i++) {
        const double eps = ndeps / 1.0;
        dpar[i] = dpar[i] + eps;
        fmingr(dpar, df1);
        dpar[i] = dpar[i] - 2 * eps;
        fmingr(dpar, df2);
        for (int j = 0; j < n; j++) ans[i * n + j] = 1.0 * (df1[j] - df2[j]) / (2 * eps * 1.0 * 1.0);
        dpar[i] = dpar[i] + eps;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            const double tmp = 0.5 * (ans[i * n + j] + ans[j * n + i]);
            ans[i * n + j] = ans[j * n + i] = tmp;
        }
    // R's matrix is column-major: ans[i * n + j] is row j, column i
    std::vector<double> rowmajor(ans.size());
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) rowmajor[j * n + i] = ans[i * n + j];
    return rowmajor;
}

namespace {

// The 2-norm of a column (the classic scaled sum; only the rank is used)
double nrm2(int n, const double* x) {
    double scale = 0.0, ssq = 1.0;
    for (int i = 0; i < n; i++) {
        if (x[i] != 0.0) {
            const double absxi = std::fabs(x[i]);
            if (scale < absxi) {
                ssq = 1.0 + ssq * (scale / absxi) * (scale / absxi);
                scale = absxi;
            } else {
                ssq += (absxi / scale) * (absxi / scale);
            }
        }
    }
    return scale * std::sqrt(ssq);
}

}  // namespace

int qr_rank(const std::vector<double>& a, int n, double tol) {
    // LINPACK dqrdc2 (R's qr()), column-major copy, n x n
    const int p = n;
    std::vector<double> x(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) x[j * n + i] = a[i * n + j];
    auto X = [&](int i, int j) -> double& { return x[(j - 1) * n + (i - 1)]; };
    std::vector<double> qraux(static_cast<std::size_t>(p) + 1), w1(static_cast<std::size_t>(p) + 1),
        w2(static_cast<std::size_t>(p) + 1);
    for (int j = 1; j <= p; j++) {
        qraux[j] = nrm2(n, &X(1, j));
        w1[j] = qraux[j];
        w2[j] = qraux[j];
        if (w2[j] == 0.0) w2[j] = 1.0;
    }
    const int lup = std::min(n, p);
    int k = p + 1;
    for (int l = 1; l <= lup; l++) {
        while (!(l >= k || qraux[l] >= w2[l] * tol)) {
            for (int i = 1; i <= n; i++) {
                const double t = X(i, l);
                for (int j = l + 1; j <= p; j++) X(i, j - 1) = X(i, j);
                X(i, p) = t;
            }
            const double t = qraux[l], tt = w1[l], ttt = w2[l];
            for (int j = l + 1; j <= p; j++) {
                qraux[j - 1] = qraux[j];
                w1[j - 1] = w1[j];
                w2[j - 1] = w2[j];
            }
            qraux[p] = t;
            w1[p] = tt;
            w2[p] = ttt;
            k = k - 1;
        }
        if (l == n) break;
        double nrmxl = nrm2(n - l + 1, &X(l, l));
        if (nrmxl == 0.0) continue;
        if (X(l, l) != 0.0) nrmxl = std::copysign(std::fabs(nrmxl), X(l, l));
        for (int i = l; i <= n; i++) X(i, l) *= 1.0 / nrmxl;
        X(l, l) = 1.0 + X(l, l);
        for (int j = l + 1; j <= p; j++) {
            double dot = 0.0;
            for (int i = l; i <= n; i++) dot += X(i, l) * X(i, j);
            const double t = -dot / X(l, l);
            for (int i = l; i <= n; i++) X(i, j) += t * X(i, l);
            if (qraux[j] == 0.0) continue;
            double tt = 1.0 - (std::fabs(X(l, j)) / qraux[j]) * (std::fabs(X(l, j)) / qraux[j]);
            tt = std::max(tt, 0.0);
            if (std::fabs(tt) < 1e-6) {
                qraux[j] = nrm2(n - l, &X(l + 1, j));
                w1[j] = qraux[j];
            } else {
                qraux[j] = qraux[j] * std::sqrt(tt);
            }
        }
        qraux[l] = X(l, l);
        X(l, l) = -nrmxl;
    }
    return std::min(k - 1, n);
}

namespace {

// Reference LAPACK's dgetrf2 (recursive LU with partial pivoting) on the
// column-major m x n block at a (leading dimension lda); ipiv 1-based.
int dgetrf2(int m, int n, double* a, int lda, int* ipiv) {
    auto A = [&](int i, int j) -> double& { return a[(j - 1) * lda + (i - 1)]; };
    if (m == 0 || n == 0) return 0;
    if (m == 1) {
        ipiv[0] = 1;
        return A(1, 1) == 0.0 ? 1 : 0;
    }
    if (n == 1) {
        const double sfmin = DBL_MIN;
        int imax = 1;
        double vmax = std::fabs(A(1, 1));
        for (int i = 2; i <= m; i++)
            if (std::fabs(A(i, 1)) > vmax) {
                vmax = std::fabs(A(i, 1));
                imax = i;
            }
        ipiv[0] = imax;
        if (A(imax, 1) != 0.0) {
            if (imax != 1) std::swap(A(1, 1), A(imax, 1));
            if (std::fabs(A(1, 1)) >= sfmin) {
                const double r = 1.0 / A(1, 1);
                for (int i = 2; i <= m; i++) A(i, 1) = r * A(i, 1);
            } else {
                for (int i = 1; i <= m - 1; i++) A(1 + i, 1) = A(1 + i, 1) / A(1, 1);
            }
            return 0;
        }
        return 1;
    }
    const int n1 = std::min(m, n) / 2, n2 = n - n1;
    int info = dgetrf2(m, n1, a, lda, ipiv);
    // dlaswp on [A12; A22]
    for (int i = 1; i <= n1; i++) {
        const int ip = ipiv[i - 1];
        if (ip != i)
            for (int j = n1 + 1; j <= n; j++) std::swap(A(i, j), A(ip, j));
    }
    // dtrsm('L', 'L', 'N', 'U'): A12 = L11^-1 A12
    for (int j = n1 + 1; j <= n; j++)
        for (int k = 1; k <= n1; k++)
            if (A(k, j) != 0.0)
                for (int i = k + 1; i <= n1; i++) A(i, j) = A(i, j) - A(k, j) * A(i, k);
    // dgemm: A22 = A22 - A21 A12
    for (int j = n1 + 1; j <= n; j++)
        for (int l = 1; l <= n1; l++) {
            const double temp = -1.0 * A(l, j);
            for (int i = n1 + 1; i <= m; i++) A(i, j) = A(i, j) + temp * A(i, l);
        }
    const int iinfo = dgetrf2(m - n1, n2, &A(n1 + 1, n1 + 1), lda, ipiv + n1);
    if (info == 0 && iinfo > 0) info = iinfo + n1;
    for (int i = n1 + 1; i <= std::min(m, n); i++) ipiv[i - 1] += n1;
    // dlaswp on [A11; A21] with the second half's pivots
    for (int i = n1 + 1; i <= std::min(m, n); i++) {
        const int ip = ipiv[i - 1];
        if (ip != i)
            for (int j = 1; j <= n1; j++) std::swap(A(i, j), A(ip, j));
    }
    return info;
}

}  // namespace

bool r_inverse(const std::vector<double>& a, int n, std::vector<double>& inv) {
    // dgesv: dgetrf (= dgetrf2 below the block size) and dgetrs on B = I
    std::vector<double> lu(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) lu[j * n + i] = a[i * n + j];
    std::vector<int> ipiv(static_cast<std::size_t>(n));
    if (dgetrf2(n, n, lu.data(), n, ipiv.data()) != 0) return false;
    std::vector<double> b(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; i++) b[i * n + i] = 1.0;
    auto A = [&](int i, int j) -> double& { return lu[(j - 1) * n + (i - 1)]; };
    auto B = [&](int i, int j) -> double& { return b[(j - 1) * n + (i - 1)]; };
    for (int i = 1; i <= n; i++) {
        const int ip = ipiv[i - 1];
        if (ip != i)
            for (int j = 1; j <= n; j++) std::swap(B(i, j), B(ip, j));
    }
    for (int j = 1; j <= n; j++) {
        // L (unit lower)
        for (int k = 1; k <= n; k++)
            if (B(k, j) != 0.0)
                for (int i = k + 1; i <= n; i++) B(i, j) = B(i, j) - B(k, j) * A(i, k);
        // U (upper, non-unit)
        for (int k = n; k >= 1; k--)
            if (B(k, j) != 0.0) {
                B(k, j) = B(k, j) / A(k, k);
                for (int i = 1; i <= k - 1; i++) B(i, j) = B(i, j) - B(k, j) * A(i, k);
            }
    }
    inv.assign(b.size(), 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) inv[i * n + j] = b[j * n + i];
    return true;
}

GammaFit fit_gamma(const double* x_in, i64 n) {
    if (n < 2) throw std::invalid_argument("fitting a gamma distribution needs at least two step lengths");
    std::vector<double> x(x_in, x_in + n);
    GammaFit out;
    bool zero = false;
    double sl_min = std::numeric_limits<double>::infinity();
    for (double v : x) {
        if (v < 0 || !std::isfinite(v)) throw std::invalid_argument("step lengths must be finite and >= 0");
        if (v == 0) zero = true;
        else sl_min = std::min(sl_min, v);
    }
    if (zero) {
        if (!std::isfinite(sl_min)) throw std::invalid_argument("all step lengths are 0");
        for (double& v : x)
            if (v == 0) v = sl_min;
        out.zeros_replaced = true;
        out.zero_replacement = sl_min;
    }
    // closed-form starting values: t1 <- n * sum(x * log(x)) - sum(log(x)) * sum(x)
    std::vector<double> lx(x.size()), xlx(x.size());
    for (std::size_t i = 0; i < x.size(); i++) {
        lx[i] = lsm::r_log(x[i]);
        xlx[i] = x[i] * lx[i];
    }
    const double nn = static_cast<double>(n);
    const double sum_x = lsm::r_sum(x.data(), n);
    const double t1 = nn * lsm::r_sum(xlx.data(), n) - lsm::r_sum(lx.data(), n) * sum_x;
    const double shape_closed = (nn * sum_x) / t1;
    const double scale_closed = 1 / (nn * nn) * t1;
    // fitdistrplus' minus mean log likelihood of (scale, shape)
    std::vector<double> t(x.size());
    Objective fn = [&](const std::vector<double>& par) {
        bool inf = false;
        for (std::size_t i = 0; i < x.size(); i++) {
            t[i] = r_dgamma(x[i], par[1], par[0], true);
            if (std::isinf(t[i])) inf = true;
        }
        if (inf) return 2147483647.0;
        return -lsm::r_mean(t.data(), n);
    };
    const NelderMead opt = nelder_mead({scale_closed, shape_closed}, fn);
    out.scale = opt.par[0];
    out.shape = opt.par[1];
    out.fail = opt.fail;
    out.fncount = opt.fncount;
    out.loglik = -opt.value * nn;
    const std::vector<double> hess = optim_hessian(opt.par, fn);
    bool finite = true;
    for (double v : hess)
        if (std::isnan(v)) finite = false;
    std::vector<double> inv;
    if (finite && qr_rank(hess, 2) == 2 && r_inverse(hess, 2, inv)) {
        for (double& v : inv) v = v / nn;
        out.vcov = inv;
    }
    return out;
}

namespace {

// fitdistrplus' fnobj: .Machine$integer.max when a log density is infinite,
// else minus their mean
double minus_mean_log(std::vector<double>& t, i64 n) {
    for (double v : t)
        if (std::isinf(v)) return 2147483647.0;
    return -lsm::r_mean(t.data(), n);
}

void check_positive(const double* x, i64 n, const char* what) {
    if (n < 2) throw std::invalid_argument(std::string("fitting ") + what + " needs at least two values");
    for (i64 i = 0; i < n; i++)
        if (!(x[i] > 0) || !std::isfinite(x[i]))
            throw std::invalid_argument(std::string("fitting ") + what + ": values must be finite and > 0");
}

}  // namespace

double fit_exp(const double* x, i64 n) {
    check_positive(x, n, "an exponential distribution");
    return 1 / lsm::r_mean(x, n);
}

std::vector<double> fit_lnorm(const double* x, i64 n) {
    check_positive(x, n, "a log-normal distribution");
    std::vector<double> lx(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; i++) lx[i] = lsm::r_log(x[i]);
    const double ml = lsm::r_mean(lx.data(), n);
    const double nn = static_cast<double>(n);
    // sd0 <- sqrt((n - 1) / n) * sd(lx): R's scalar sqrt (correctly rounded)
    const double sd0 = std::sqrt((nn - 1) / nn) * lsm::r_sd(lx.data(), n, ml);
    std::vector<double> t(static_cast<std::size_t>(n));
    Objective fn = [&](const std::vector<double>& par) {
        const double meanlog = par[0], sdlog = par[1];
        // R's dlnorm(x, meanlog, sdlog, log = TRUE) for x > 0
        for (i64 i = 0; i < n; i++) {
            if (std::isnan(meanlog) || std::isnan(sdlog)) {
                t[i] = x[i] + meanlog + sdlog;
            } else if (sdlog < 0) {
                t[i] = kNaN;
            } else if (sdlog == 0) {
                t[i] = lsm::r_log(x[i]) == meanlog ? std::numeric_limits<double>::infinity()
                                                   : -std::numeric_limits<double>::infinity();
            } else {
                const double y = (lsm::r_log(x[i]) - meanlog) / sdlog;
                t[i] = -(0.918938533204672741780329736406 + 0.5 * y * y + lsm::r_log(x[i] * sdlog));
            }
        }
        return minus_mean_log(t, n);
    };
    return nelder_mead({ml, sd0}, fn).par;
}

namespace {

// circular's A1inv(x)
double a1inv(double x) {
    if (0 <= x && x < 0.53) return 2 * x + lsm::r_pow(x, 3) + (5 * lsm::r_pow(x, 5)) / 6;
    if (x < 0.85) return -0.4 + 1.39 * x + 0.43 / (1 - x);
    return 1 / (lsm::r_pow(x, 3) - 4 * (x * x) + 3 * x);
}

// circular's A1(kappa): besselI(kappa, 1, TRUE) / besselI(kappa, 0, TRUE)
double a1(double kappa) {
    return r_bessel_i_scaled(kappa, 1) / r_bessel_i_scaled(kappa, 0);
}

}  // namespace

VonMisesFit fit_vonmises(const double* x_in, i64 n) {
    if (n < 1) throw std::invalid_argument("fitting a von Mises distribution needs turn angles");
    // conversion.circular(modulo = "2pi"): x %% (2 * pi)
    std::vector<double> x(static_cast<std::size_t>(n)), s(x.size()), c(x.size());
    for (i64 i = 0; i < n; i++) {
        x[i] = r_fmod(x_in[i], 2 * kPi);
        s[i] = lsm::r_sin(x[i]);
        c[i] = lsm::r_cos(x[i]);
    }
    const double sinr = lsm::r_sum(s.data(), n), cosr = lsm::r_sum(c.data(), n);
    VonMisesFit out;
    out.mu = lsm::r_atan2(sinr, cosr);
    for (i64 i = 0; i < n; i++) c[i] = lsm::r_cos(x[i] - out.mu);
    const double V = lsm::r_mean(c.data(), n);
    out.kappa = V > 0 ? a1inv(V) : 0;
    const double a = a1(out.kappa);
    const double nn = static_cast<double>(n);
    out.se_mu = std::sqrt(1 / (nn * out.kappa * a));
    out.se_kappa = std::sqrt(1 / (nn * (1 - a / out.kappa - a * a)));
    return out;
}

void rvonmises(nlm::RRng& rng, i64 n, double mu, double kappa, double* out) {
    if (kappa == 0) {
        // stats::runif(n, 0, 2 * pi)
        rng.runif(out, n, 0.0, 2 * kPi);
        return;
    }
    // circular's src/rvonmises.c rvm()
    const double a = 1 + std::sqrt(1 + 4 * kappa * kappa);
    const double b = (a - std::sqrt(2 * a)) / (2 * kappa);
    const double r = (1 + b * b) / (2 * b);
    i64 i = 0;
    do {
        const double U1 = rng.unif_rand();
        const double z = lsm::r_cos(kPi * U1);
        const double f = (1. + r * z) / (r + z);
        const double c = kappa * (r - f);
        const double U2 = rng.unif_rand();
        if (c * (2 - c) - U2 > 0 || lsm::r_log(c / U2) + 1 - c >= 0.) {
            const double U3 = rng.unif_rand();
            out[i] = (U3 > 0.50) ? std::acos(f) + mu : -std::acos(f) + mu;
            i++;
        }
    } while (i < n);
    // RvonmisesRad: .C("rvm", ...)[[1]] %% (2 * pi)
    for (i64 j = 0; j < n; j++) out[j] = r_fmod(out[j], 2 * kPi);
}

void random_numbers(nlm::RRng& rng, const char* name, const std::vector<double>& params, i64 n, double* out) {
    const std::string d(name);
    auto need = [&](std::size_t k) {
        if (params.size() != k) throw std::invalid_argument("random_numbers: " + d + " takes " + std::to_string(k) + " parameters");
    };
    if (d == "gamma") {
        need(2);
        for (i64 i = 0; i < n; i++) out[i] = r_rgamma(rng, params[0], params[1]);
    } else if (d == "exp") {
        need(1);
        const double scale = 1 / params[0];  // rexp(n, rate) is .Call(C_rexp, n, 1 / rate)
        for (i64 i = 0; i < n; i++) out[i] = scale * rng.exp_rand();
    } else if (d == "lnorm") {
        need(2);
        for (i64 i = 0; i < n; i++) out[i] = lsm::r_exp(rng.rnorm(params[0], params[1]));
    } else if (d == "hnorm") {
        need(1);
        for (i64 i = 0; i < n; i++) out[i] = std::fabs(rng.rnorm(0.0, params[0]));
    } else if (d == "unif") {
        need(2);
        rng.runif(out, n, params[0], params[1]);
    } else if (d == "vonmises") {
        if (params.size() != 1) need(2);
        // circular's rvonmises takes mu modulo 2 pi (amt's mu: 0)
        const double mu = params.size() == 2 ? r_fmod(params[1], 2 * kPi) : 0.0;
        rvonmises(rng, n, mu, params[0], out);
        // amt: x %% (2 * pi), then x - 2 pi where x > pi
        for (i64 i = 0; i < n; i++) {
            const double v = r_fmod(out[i], 2 * kPi);
            out[i] = v > kPi ? v - (2 * kPi) : v;
        }
    } else {
        throw std::invalid_argument("unknown distribution " + d);
    }
}

}  // namespace move
