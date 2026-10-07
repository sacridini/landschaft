// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/proxirr.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "lsm/extended.hpp"

namespace opt::proxirr {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double one_alpha(double local, double global, double target, bool triage) {
    if (std::isnan(local) || std::isnan(global) || std::isnan(target)) return kNaN;
    if (global == 0.0 && local == 0.0) return 0.0;  // nothing of the feature anywhere
    if (target == 0.0) return 0.0;
    if (triage && target > global) return 0.0;
    if (target >= global) return local > 0.0 ? 1.0 : 0.0;
    return std::min(1.0, local / (global - target));
}

// R's prod() of the non-NaN values: LDOUBLE s = 1; s *= x[i]; then rounded to double (+-Inf beyond).
double r_prod(const double* x, i64 n, bool skip_ones) {
    lsm::Extended s = 1.0;
    for (i64 i = 0; i < n; ++i) {
        const double v = x[i];
        if (std::isnan(v)) continue;
        if (skip_ones && v == 0.0) continue;  // 1 - alpha with alpha == 1: gamma counts those apart
        s *= lsm::Extended(v);
    }
    const double d = static_cast<double>(s);
    return d;
}

}  // namespace

std::vector<double> alpha(const std::vector<double>& local, const std::vector<double>& global,
                          const std::vector<double>& target, bool triage, int threads) {
    const i64 n = static_cast<i64>(local.size());
    if (global.size() != local.size() || target.size() != local.size())
        throw std::invalid_argument("local, global and target must have the same length");
    for (i64 i = 0; i < n; ++i) {
        if (local[i] < 0) throw std::invalid_argument("local amounts must be non-negative numbers");
        if (global[i] < 0) throw std::invalid_argument("global amounts must be non-negative numbers");
        if (target[i] < 0) throw std::invalid_argument("targets must be non-negative numbers");
        if (local[i] > global[i]) throw std::invalid_argument("a local amount is greater than its global amount");
    }
    std::vector<double> out(static_cast<std::size_t>(n));
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) out[i] = one_alpha(local[i], global[i], target[i], triage);
    return out;
}

void beta_gamma(const std::vector<i64>& start, const std::vector<double>& alphas, int threads,
                std::vector<double>& beta, std::vector<double>& gamma) {
    const i64 nu = static_cast<i64>(start.size()) - 1;
    if (nu < 0 || start.back() != static_cast<i64>(alphas.size()))
        throw std::invalid_argument("start must hold one offset per unit and the number of alphas");
    beta.assign(static_cast<std::size_t>(nu), 0.0);
    gamma.assign(static_cast<std::size_t>(nu), 0.0);
    std::vector<double> q(alphas.size());
    for (std::size_t k = 0; k < alphas.size(); ++k) {
        const double a = alphas[k];
        if (!std::isnan(a) && (a < 0.0 || a > 1.0)) throw std::invalid_argument("an alpha is not between 0 and 1");
        q[k] = std::isnan(a) ? kNaN : 1.0 - a;  // R: 1 - alphas, in double
    }
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1024)
    for (i64 u = 0; u < nu; ++u) {
        const i64 b = start[u], e = start[u + 1];
        beta[u] = 1.0 - r_prod(q.data() + b, e - b, false);
        // gamma: sum(alphas == 1) + .beta(alphas[alphas != 1]) (an integer plus a double)
        i64 ones = 0;
        for (i64 k = b; k < e; ++k) ones += alphas[k] == 1.0;
        gamma[u] = static_cast<double>(ones) + (1.0 - r_prod(q.data() + b, e - b, true));
    }
}

}  // namespace opt::proxirr
