// SPDX-License-Identifier: GPL-3.0-or-later
#include "sim/molusce.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace sim {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kMax = std::numeric_limits<double>::max();

// numpy's nan_to_num(v, nan=nan_value): infinities to the largest finite values.
double nan_to_num(double v, double nan_value) {
    if (std::isnan(v)) return nan_value;
    if (v == kInf) return kMax;
    if (v == -kInf) return -kMax;
    return v;
}

// MOLUSCE's sigmoid: the argument clipped to [-100, 100].
double sigmoid(double x) {
    x = std::max(-100.0, std::min(x, 100.0));
    return 1.0 / (1.0 + std::exp(-x));
}

}  // namespace

std::vector<double> molusce_categories(const double* values, std::int64_t n, const std::vector<double>& bounds) {
    std::vector<double> out(values, values + n);
    if (bounds.empty()) return out;
    for (std::int64_t i = 0; i < n; ++i) {
        if (std::isnan(out[i])) continue;
        const auto k = std::upper_bound(bounds.begin(), bounds.end(), out[i]) - bounds.begin();
        out[i] = static_cast<double>(k) + 1.0;
    }
    return out;
}

std::vector<double> molusce_total(const double* factors, int nf, std::int64_t n,
                                  const std::vector<std::vector<MolusceFactorWeights>>& weights) {
    const auto ncode = static_cast<std::int64_t>(weights.size());
    std::vector<double> out(static_cast<std::size_t>(ncode * n), 0.0);
    for (std::int64_t c = 0; c < ncode; ++c) {
        if (static_cast<int>(weights[c].size()) != nf) throw std::invalid_argument("one set of weights per factor");
        double* total = out.data() + c * n;
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < n; ++i) {
            double t = 0.0;
            bool nan = false;
            for (int f = 0; f < nf; ++f) {
                const double v = factors[static_cast<std::int64_t>(f) * n + i];
                const auto& w = weights[c][f];
                double x = 0.0;
                for (std::size_t k = 0; k < w.categories.size(); ++k) {
                    if (v == w.categories[k]) x = w.weights[k];  // the last category of that value (numpy's
                }                                                  // repeated assignments)
                nan = nan || std::isnan(v);
                t = f == 0 ? x : t + x;
            }
            total[i] = nan ? kNaN : t;
        }
    }
    return out;
}

MolusceForecast molusce_prediction_woe(const double* state, std::int64_t n, const std::vector<double>& classes,
                                       const std::vector<std::int64_t>& codes, const double* totals) {
    const auto m = static_cast<std::int64_t>(classes.size());
    MolusceForecast r{std::vector<double>(static_cast<std::size_t>(n), kNaN),
                      std::vector<double>(static_cast<std::size_t>(n), kNaN)};
    // per class, the codes i * m + j given, in the order of j, with their row of totals
    std::vector<std::vector<std::pair<std::int64_t, const double*>>> cand(static_cast<std::size_t>(m));
    for (std::int64_t i = 0; i < m; ++i)
        for (std::int64_t j = 0; j < m; ++j)
            for (std::size_t k = 0; k < codes.size(); ++k)
                if (codes[k] == i * m + j) {
                    cand[i].emplace_back(codes[k], totals + static_cast<std::int64_t>(k) * n);
                    break;
                }
#pragma omp parallel for schedule(static)
    for (std::int64_t p = 0; p < n; ++p) {
        const double s = state[p];
        std::int64_t i = -1;
        for (std::int64_t k = 0; k < m; ++k)
            if (s == classes[k]) {
                i = k;
                break;
            }
        if (i < 0) continue;
        double best = -1000.0, second = -1000.0;
        std::int64_t arg = -1;
        for (const auto& [code, t] : cand[i]) {
            const double w = t[p];
            if (std::isnan(w)) continue;
            if (w > best) {
                second = best;
                best = w;
                arg = code;
            } else if (w > second) {
                second = w;
            }
        }
        r.prediction[p] = arg >= 0 ? classes[arg % m] : kNaN;
        r.confidence[p] = std::trunc(100.0 * (sigmoid(best) - sigmoid(second)));
    }
    return r;
}

MolusceForecast molusce_prediction_classes(const double* state, std::int64_t n, int m, const double* layers,
                                           const std::vector<double>& classes) {
    if (static_cast<int>(classes.size()) != m) throw std::invalid_argument("one class per layer");
    MolusceForecast r{std::vector<double>(static_cast<std::size_t>(n), kNaN),
                      std::vector<double>(static_cast<std::size_t>(n), kNaN)};
#pragma omp parallel for schedule(static)
    for (std::int64_t p = 0; p < n; ++p) {
        if (std::isnan(state[p])) continue;
        // numpy's stable argsort of -layers: the largest first, the first class among equals
        int top = 0;
        double vtop = -kInf;
        for (int k = 0; k < m; ++k) {
            const double v = nan_to_num(layers[static_cast<std::int64_t>(k) * n + p], -kInf);
            if (k == 0 || v > vtop) {
                vtop = v;
                top = k;
            }
        }
        double sec = 0.0;
        if (m > 1) {
            sec = -kInf;
            bool found = false;
            for (int k = 0; k < m; ++k) {
                if (k == top) continue;
                const double v = nan_to_num(layers[static_cast<std::int64_t>(k) * n + p], -kInf);
                if (!found || v > sec) {
                    sec = v;
                    found = true;
                }
            }
        }
        if (!std::isfinite(vtop)) {
            r.confidence[p] = 0.0;
            continue;
        }
        r.prediction[p] = classes[top];
        r.confidence[p] = std::trunc(100.0 * (vtop - (std::isfinite(sec) ? sec : 0.0)));
    }
    return r;
}

MolusceStep molusce_ca_step(const double* state, std::int64_t n, const double* prediction, const double* confidence,
                            const std::vector<MolusceTransition>& counts, NumpyRandomState& rng) {
    MolusceStep r;
    r.state.assign(state, state + n);
    std::vector<double> present;
    for (std::int64_t p = 0; p < n; ++p)
        if (!std::isnan(state[p])) present.push_back(state[p]);
    std::sort(present.begin(), present.end());
    present.erase(std::unique(present.begin(), present.end()), present.end());
    std::vector<double> c(static_cast<std::size_t>(n));
    std::vector<std::int64_t> idx;
    for (const double f : present) {
        for (const double t : present) {
            if (f == t) continue;
            std::int64_t want = 0;
            for (const auto& x : counts)
                if (x.from == f && x.to == t) want = x.cells;  // the last one given
            if (want == 0) continue;
            std::int64_t places = 0;
            for (std::int64_t p = 0; p < n; ++p) places += (state[p] == f && prediction[p] == t) ? 1 : 0;
            std::int64_t k = std::min(want, places);
            if (k > 0) {
                idx.clear();
                for (std::int64_t p = 0; p < n; ++p) {
                    const double conf = nan_to_num(confidence[p], 0.0);
                    const double u = rng.random_sample();
                    const bool here = state[p] == f && prediction[p] == t;
                    c[p] = here ? (conf + u / 1000.0) * 1.0 : 0.0;
                    if (c[p] > 0.0) idx.push_back(p);
                }
                k = std::min(k, static_cast<std::int64_t>(idx.size()));
                if (k > 0) {
                    // the k largest; among equal values the later cells (the last k of a stable ascending order)
                    auto later = [&](std::int64_t a, std::int64_t b) {
                        return c[a] > c[b] || (c[a] == c[b] && a > b);
                    };
                    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), later);
                    for (std::int64_t q = 0; q < k; ++q) r.state[idx[q]] = t;
                }
            }
            r.changed.push_back({f, t, k});
        }
    }
    return r;
}

}  // namespace sim
