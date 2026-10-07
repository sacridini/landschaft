// SPDX-License-Identifier: GPL-3.0-or-later
//
// ProxIRR (Baisero, Schuster & Plumptre 2022, "Redefining and mapping global irreplaceability"), the
// arithmetic of the CRAN package proxirr 0.5 (MIT): alpha per feature and unit, beta and gamma per
// unit. beta = 1 - prod(1 - alpha) is R's prod() (80-bit accumulation), so the bits are proxirr's.
#pragma once

#include <cstdint>
#include <vector>

namespace opt::proxirr {

using i64 = std::int64_t;

// proxirr's .alpha for every entry: 0 when the target is 0 (or, with triage, above the global amount);
// when the target reaches the global amount, 1 if the unit holds any, else 0; otherwise
// min(1, local / (global - target)). NaN in any input gives NaN (proxirr's na.allow). A global amount of
// 0 gives 0 (proxirr stops with an error). Throws on local < 0, global < 0, target < 0 or local > global.
std::vector<double> alpha(const std::vector<double>& local, const std::vector<double>& global,
                          const std::vector<double>& target, bool triage, int threads);

// beta and gamma of every unit u from the alphas start[u] .. start[u + 1] (in the order of R's vector):
// beta = 1 - prod(1 - alpha), gamma = count(alpha == 1) + beta of the others; NaN alphas left out
// (na.rm = TRUE).
void beta_gamma(const std::vector<i64>& start, const std::vector<double>& alphas, int threads,
                std::vector<double>& beta, std::vector<double>& gamma);

}  // namespace opt::proxirr
