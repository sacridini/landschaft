// SPDX-License-Identifier: GPL-3.0-or-later
//
// The distributions of steps of ls.move (amt 0.3.1's fit_distr and
// random_numbers), with R's numbers: R's optim (Nelder-Mead, optimhess),
// fitdistrplus 1.2-6's maximum likelihood fit, circular 0.5-2's
// mle.vonmises and rvonmises, R's solve() of small matrices (LAPACK's
// dgetrf2 / dgetrs as reference LAPACK computes them).
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace nlm {
class RRng;
}

namespace move {

using i64 = std::int64_t;

// R's optim(par, fn, method = "Nelder-Mead") with the default control
// (alpha 1, beta 0.5, gamma 2, reltol sqrt(eps), abstol -Inf, maxit 500):
// appl/optim.c's nmmin. `fail`: 0, 1 (maxit reached) or 10 (degenerate).
struct NelderMead {
    std::vector<double> par;
    double value = 0;
    int fncount = 0;
    int fail = 0;
};
using Objective = std::function<double(const std::vector<double>&)>;
NelderMead nelder_mead(const std::vector<double>& start, const Objective& fn, int maxit = 500);

// optimHess(par, fn) with ndeps 1e-3 (stats/src/optim.c's optimhess):
// central differences of central differences, symmetrised. Row-major.
std::vector<double> optim_hessian(const std::vector<double>& par, const Objective& fn, double ndeps = 1e-3);

// qr(a)$rank (LINPACK dqrdc2, tol 1e-7) of an n x n row-major matrix.
int qr_rank(const std::vector<double>& a, int n, double tol = 1e-7);

// solve(a): the inverse of an n x n row-major matrix by reference LAPACK's
// dgetrf2 and dgetrs (R's dgesv). False when a pivot is exactly zero.
bool r_inverse(const std::vector<double>& a, int n, std::vector<double>& inv);

// amt::fit_distr(x, "gamma"): steps of length 0 set to the smallest other,
// closed-form starting values, fitdistrplus::fitdist(x, "gamma", start =
// list(scale, shape)) by Nelder-Mead on minus the mean log density, and
// vcov = solve(hessian) / n (empty when the hessian is not of full rank).
struct GammaFit {
    double shape = 0, scale = 0;
    std::vector<double> vcov;  // 2 x 2 row-major, (scale, shape) order as fitdistrplus
    int fail = 0;
    int fncount = 0;
    double loglik = 0;
    bool zeros_replaced = false;
    double zero_replacement = 0;
};
GammaFit fit_gamma(const double* x, i64 n);

// amt::fit_distr(x, "exp"), corrected: the maximum likelihood rate 1 /
// mean(x). amt fits it with fitdistrplus by BFGS from that value, whose
// numerical gradient (steps of 1e-3) fails for rates below 1e-3 (mean steps
// over 1000 m), and otherwise moves it only by the optimizer's noise.
double fit_exp(const double* x, i64 n);

// amt::fit_distr(x, "lnorm"): fitdistrplus::fitdist(x, "lnorm") by
// Nelder-Mead from meanlog = mean(log(x)), sdlog = sqrt((n - 1) / n) *
// sd(log(x)); (meanlog, sdlog).
std::vector<double> fit_lnorm(const double* x, i64 n);

// circular::mle.vonmises(x) (mu and kappa estimated, bias FALSE) after
// as.circular(modulo = "asis") and the conversion to [0, 2 pi): mu =
// atan2(sum sin, sum cos), kappa = A1inv(mean(cos(x - mu))).
struct VonMisesFit {
    double mu = 0, kappa = 0, se_mu = 0, se_kappa = 0;
};
VonMisesFit fit_vonmises(const double* x, i64 n);

// circular's rvonmises(n, mu, kappa) (Best & Fisher's rejection sampler, C
// code rvm) %% (2 pi), or runif(n, 0, 2 pi) for kappa = 0.
void rvonmises(nlm::RRng& rng, i64 n, double mu, double kappa, double* out);

// amt's random_numbers(): n draws of a distribution of step lengths or
// turn angles from `rng`, as amt draws them (rgamma, rexp, rlnorm, abs of
// rnorm, runif; rvonmises with mu 0 wrapped to [-pi, pi)). `name`: "gamma"
// (shape, scale), "exp" (rate), "lnorm" (meanlog, sdlog), "hnorm" (sd),
// "unif" (min, max), "vonmises" (kappa, optionally mu; amt: 0).
void random_numbers(nlm::RRng& rng, const char* name, const std::vector<double>& params, i64 n, double* out);

}  // namespace move
