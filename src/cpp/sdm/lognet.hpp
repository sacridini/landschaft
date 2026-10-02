// SPDX-License-Identifier: GPL-3.0-or-later
//
// glmnet 4.1-10's lasso path of the two-class logistic regression (lognet),
// as maxnet 0.1.4 calls it: dense x, alpha = 1, standardize = FALSE,
// intercept, type.logistic = "Newton", a user lambda sequence, thresh 1e-7,
// maxit 1e5 and glmnet.control(pmin = 1e-8). The same bits as R's: glmnet's
// C++ (glmnetpp) operation by operation, Eigen's SSE2 order of every sum and
// dot product, R's x87 exp() and log() (lsm/rmath.hpp).
#pragma once

#include <cstdint>
#include <vector>

namespace sdm {

struct LognetResult {
    std::vector<double> beta;  // coefficients at the last lambda (one per column)
    std::int64_t n_passes = 0;  // coordinate-descent passes over the whole path (glmnet's npasses)
    int n_lambda = 0;           // lambdas completed (fewer than asked: maxit reached)
};

// Fits the path. `x` (n x ni, column-major) is centred in place, as glmnet
// does; `y` is 1 for a presence and 0 otherwise, `weights` the observation
// weights, `penalty` the penalty factors, `lambda` the decreasing sequence.
// Throws std::invalid_argument for glmnet's fatal errors (all penalty factors
// zero, a response proportion below pmin).
LognetResult lognet_path(double* x, std::int64_t n, std::int64_t ni, const double* y, const double* weights,
                         const double* penalty, const double* lambda, int n_lambda, double thresh = 1e-7,
                         std::int64_t maxit = 100000, double pmin = 1e-8);

}  // namespace sdm
