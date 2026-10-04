// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace opt {

FeatureMatrix::FeatureMatrix(std::int64_t layers)
    : totals_(static_cast<std::size_t>(layers), 0.0),
      index_(static_cast<std::size_t>(layers)),
      value_(static_cast<std::size_t>(layers)) {
    if (layers < 0) throw std::invalid_argument("layers must be >= 0");
}

void FeatureMatrix::add(const double* values, std::int64_t n, const std::int64_t* unit, int threads) {
    const std::int64_t nl = layers();
    constexpr double big = std::numeric_limits<double>::max();
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : 1)
    for (std::int64_t l = 0; l < nl; ++l) {
        const double* v = values + l * n;
        double total = totals_[static_cast<std::size_t>(l)];
        auto& index = index_[static_cast<std::size_t>(l)];
        auto& value = value_[static_cast<std::size_t>(l)];
        for (std::int64_t i = 0; i < n; ++i) {
            const double x = v[i];
            if (std::isfinite(x)) total += x;
            const std::int64_t u = unit[i];
            if (u < 0 || std::isnan(x) || x == 0.0) continue;
            index.push_back(u);
            value.push_back(std::isinf(x) ? (x > 0 ? big : -big) : x);
        }
        totals_[static_cast<std::size_t>(l)] = total;
    }
}

void FeatureMatrix::csr(std::vector<std::int64_t>& start, std::vector<std::int64_t>& index,
                        std::vector<double>& value) const {
    const auto nl = static_cast<std::size_t>(layers());
    start.assign(nl + 1, 0);
    for (std::size_t l = 0; l < nl; ++l) start[l + 1] = start[l] + static_cast<std::int64_t>(index_[l].size());
    index.resize(static_cast<std::size_t>(start[nl]));
    value.resize(static_cast<std::size_t>(start[nl]));
    for (std::size_t l = 0; l < nl; ++l) {
        std::copy(index_[l].begin(), index_[l].end(), index.begin() + start[l]);
        std::copy(value_[l].begin(), value_[l].end(), value.begin() + start[l]);
    }
}

}  // namespace opt
