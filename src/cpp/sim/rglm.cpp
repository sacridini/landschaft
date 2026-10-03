// SPDX-License-Identifier: GPL-3.0-or-later
//
// R 4.4's glm.fit for family = binomial(link = "logit") with unit weights and
// no offset (src/library/stats/R/glm.R, family.c): the start from
// mustart = (y + 0.5) / 2, IRLS with C_Cdqrls (LINPACK dqrls, tol =
// min(1e-7, epsilon / 1000)), the C link functions with their +-30
// thresholds, the deviance summed as R sums (80 bits), exp and log of the
// Windows build (x87, lsm/rmath.hpp), vector sqrt (x87), x %*% beta by the
// reference BLAS dgemv (column by column). The step-halving branches of
// glm.fit are reproduced for a non-finite deviance.

#include "sim/rglm.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "lsm/landscape.hpp"
#include "lsm/rmath.hpp"

namespace sim {

namespace {

constexpr double kThresh = 30.0;
constexpr double kMThresh = -30.0;
constexpr double kDblEps = std::numeric_limits<double>::epsilon();
constexpr double kInvEps = 1.0 / kDblEps;

double linkinv(double eta) {
    const double tmp = eta < kMThresh ? kDblEps : (eta > kThresh ? kInvEps : lsm::r_exp(eta));
    return tmp / (1.0 + tmp);
}

double mu_eta(double eta) {
    const double opexp = 1.0 + lsm::r_exp(eta);
    return (eta > kThresh || eta < kMThresh) ? kDblEps : lsm::r_exp(eta) / (opexp * opexp);
}

double y_log_y(double y, double mu) { return y != 0.0 ? y * lsm::r_log(y / mu) : 0.0; }

double deviance(const std::vector<double>& y, const std::vector<double>& mu) {
    std::vector<double> d(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) d[i] = 2.0 * 1.0 * (y_log_y(y[i], mu[i]) + y_log_y(1.0 - y[i], 1.0 - mu[i]));
    return lsm::r_sum(d.data(), static_cast<lsm::i64>(d.size()));
}

// reference BLAS dgemv, trans = "N", alpha = 1, beta = 0
void gemv(const double* x, std::int64_t n, int p, const double* b, const int* cols, int ncols, double* out) {
    for (std::int64_t i = 0; i < n; ++i) out[i] = 0.0;
    for (int c = 0; c < ncols; ++c) {
        const int j = cols ? cols[c] : c;
        const double temp = 1.0 * b[j];
        const double* col = x + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
        for (std::int64_t i = 0; i < n; ++i) out[i] = out[i] + temp * col[i];
    }
    (void)p;
}

}  // namespace

GlmFit glm_logit(const std::vector<double>& x, const std::vector<double>& y, int n, int p, int maxit, double epsilon) {
    if (static_cast<std::size_t>(n) * static_cast<std::size_t>(p) != x.size() || y.size() != static_cast<std::size_t>(n))
        throw std::invalid_argument("x must be n x p and y of length n");
    for (double v : y)
        if (!(v == 0.0 || v == 1.0)) throw std::invalid_argument("y must be 0 or 1");
    GlmFit out;
    std::vector<double> eta(n), mu(n), start(p, 0.0), coefold(p, 0.0);
    for (int i = 0; i < n; ++i) {
        const double mustart = (1.0 * y[i] + 0.5) / (1.0 + 1.0);
        eta[i] = lsm::r_log(mustart / (1.0 - mustart));
        mu[i] = linkinv(eta[i]);
    }
    double devold = deviance(y, mu);
    const double tol = std::min(1e-7, epsilon / 1000.0);
    std::vector<double> xw(static_cast<std::size_t>(n) * p), zw(n), b(p);
    std::vector<int> jpvt(p);
    std::vector<int> good_index;
    bool conv = false;
    int rank = 0;
    bool have_coef = false;
    for (int iter = 1; iter <= maxit; ++iter) {
        out.iterations = iter;
        good_index.clear();
        std::vector<double> mev(n);
        for (int i = 0; i < n; ++i) {
            mev[i] = mu_eta(eta[i]);
            if (mev[i] != 0.0) good_index.push_back(i);
        }
        const int ng = static_cast<int>(good_index.size());
        if (ng == 0) throw std::runtime_error("glm: no observations informative at the first iteration");
        xw.assign(static_cast<std::size_t>(ng) * p, 0.0);
        zw.assign(ng, 0.0);
        std::vector<double> w(ng), z(ng), wsq(ng);
        for (int g = 0; g < ng; ++g) {
            const int i = good_index[g];
            const double varmu = mu[i] * (1.0 - mu[i]);
            z[g] = (eta[i] - 0.0) + (y[i] - mu[i]) / mev[i];
            wsq[g] = (1.0 * (mev[i] * mev[i])) / varmu;
        }
        for (int g = 0; g < ng; ++g) w[g] = lsm::r_sqrt(wsq[g]);
        for (int j = 0; j < p; ++j)
            for (int g = 0; g < ng; ++g)
                xw[static_cast<std::size_t>(j) * ng + g] = x[static_cast<std::size_t>(j) * n + good_index[g]] * w[g];
        for (int g = 0; g < ng; ++g) zw[g] = z[g] * w[g];
        rank = lsm::r_dqrls(xw.data(), zw.data(), ng, p, tol, b.data(), jpvt.data());
        for (int j = 0; j < p; ++j)
            if (!std::isfinite(b[j])) throw std::runtime_error("glm: non-finite coefficients");
        for (int j = 0; j < p; ++j) start[jpvt[j] - 1] = b[j];
        gemv(x.data(), n, p, start.data(), nullptr, p, eta.data());
        for (int i = 0; i < n; ++i) mu[i] = linkinv(eta[i] = eta[i] + 0.0);
        double dev = deviance(y, mu);
        out.boundary = false;
        if (!std::isfinite(dev)) {
            if (!have_coef) throw std::runtime_error("glm: no valid set of coefficients has been found");
            int ii = 1;
            while (!std::isfinite(dev)) {
                if (ii > maxit) throw std::runtime_error("glm: inner loop 1; cannot correct step size");
                ++ii;
                for (int j = 0; j < p; ++j) start[j] = (start[j] + coefold[j]) / 2.0;
                gemv(x.data(), n, p, start.data(), nullptr, p, eta.data());
                for (int i = 0; i < n; ++i) mu[i] = linkinv(eta[i]);
                dev = deviance(y, mu);
            }
            out.boundary = true;
        }
        if (std::fabs(dev - devold) / (std::fabs(dev) + 0.1) < epsilon) {
            conv = true;
            out.deviance = dev;
            break;
        }
        devold = dev;
        coefold = start;
        have_coef = true;
        out.deviance = dev;
    }
    out.converged = conv;
    out.rank = rank;
    out.pivot = jpvt;
    out.coefficients = start;
    for (int j = rank; j < p; ++j) out.coefficients[jpvt[j] - 1] = std::numeric_limits<double>::quiet_NaN();
    // null deviance: the intercept-only fit, wtdmu = sum(w * y) / sum(w)
    {
        const double ybar = lsm::r_sum(y.data(), n) / static_cast<double>(n);
        std::vector<double> m(n, ybar);
        out.null_deviance = deviance(y, m);
    }
    return out;
}

std::vector<double> glm_logit_predict(const double* x, std::int64_t n, int p, const GlmFit& fit) {
    std::vector<int> cols(fit.rank);
    for (int c = 0; c < fit.rank; ++c) cols[c] = fit.pivot[c] - 1;
    std::vector<double> eta(static_cast<std::size_t>(n));
    gemv(x, n, p, fit.coefficients.data(), cols.data(), fit.rank, eta.data());
    for (std::int64_t i = 0; i < n; ++i) eta[i] = linkinv(eta[i]);
    return eta;
}

}  // namespace sim
