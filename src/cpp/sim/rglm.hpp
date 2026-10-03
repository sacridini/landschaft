// SPDX-License-Identifier: GPL-3.0-or-later
//
// R's glm(family = binomial) fit by glm.fit (logit link, prior weights 1),
// and its predictions on the response scale, operation for operation.
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

struct GlmFit {
    std::vector<double> coefficients;  // NaN for aliased columns (R's NA)
    std::vector<int> pivot;            // 1-based, R's qr pivot
    int rank = 0;
    int iterations = 0;
    bool converged = false;
    double deviance = 0.0;
    double null_deviance = 0.0;
    bool boundary = false;
};

// x: the model matrix, column-major n x p (the intercept column included); y: 0 / 1.
GlmFit glm_logit(const std::vector<double>& x, const std::vector<double>& y, int n, int p, int maxit = 25,
                 double epsilon = 1e-8);

// predict(type = "response"): plogis of x %*% coef over the pivoted columns of the rank (predict.lm).
std::vector<double> glm_logit_predict(const double* x, std::int64_t n, int p, const GlmFit& fit);

}  // namespace sim
