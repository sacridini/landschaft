// SPDX-License-Identifier: GPL-3.0-or-later
#include "grayspatcon.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace gtb {

namespace {

constexpr float kMissingFloat = -0.01f;

enum class MetricType {
    Mean,
    Median,
    Majority,
    Min,
    Max,
    Range,
    StdDev,
    ShannonDiversity,
    ShannonEvenness,
    DiagonalContagion,
    Unknown
};

MetricType parse_metric(const std::string& name) {
    if (name == "mean") return MetricType::Mean;
    if (name == "median") return MetricType::Median;
    if (name == "majority") return MetricType::Majority;
    if (name == "min") return MetricType::Min;
    if (name == "max") return MetricType::Max;
    if (name == "range") return MetricType::Range;
    if (name == "stddev") return MetricType::StdDev;
    if (name == "shannon_diversity") return MetricType::ShannonDiversity;
    if (name == "shannon_evenness") return MetricType::ShannonEvenness;
    if (name == "diagonal_contagion") return MetricType::DiagonalContagion;
    return MetricType::Unknown;
}

float eval_from_histogram(
    const int freq[101],
    int total_valid,
    MetricType mtype
) {
    if (total_valid == 0) return kMissingFloat;

    switch (mtype) {
        case MetricType::Mean: {
            double sum = 0.0;
            for (int k = 0; k <= 100; ++k) {
                if (freq[k] > 0) sum += static_cast<double>(k) * freq[k];
            }
            return static_cast<float>(sum / total_valid);
        }
        case MetricType::Median: {
            const int target = (total_valid + 1) / 2;
            int cum = 0;
            for (int k = 0; k <= 100; ++k) {
                cum += freq[k];
                if (cum >= target) return static_cast<float>(k);
            }
            return 100.0f;
        }
        case MetricType::Majority: {
            int max_f = -1;
            int best_k = 0;
            for (int k = 0; k <= 100; ++k) {
                if (freq[k] > max_f) {
                    max_f = freq[k];
                    best_k = k;
                }
            }
            return static_cast<float>(best_k);
        }
        case MetricType::Min: {
            for (int k = 0; k <= 100; ++k) {
                if (freq[k] > 0) return static_cast<float>(k);
            }
            return 0.0f;
        }
        case MetricType::Max: {
            for (int k = 100; k >= 0; --k) {
                if (freq[k] > 0) return static_cast<float>(k);
            }
            return 0.0f;
        }
        case MetricType::Range: {
            int mn = 0, mx = 0;
            for (int k = 0; k <= 100; ++k) {
                if (freq[k] > 0) { mn = k; break; }
            }
            for (int k = 100; k >= 0; --k) {
                if (freq[k] > 0) { mx = k; break; }
            }
            return static_cast<float>(mx - mn);
        }
        case MetricType::StdDev: {
            double sum = 0.0;
            double sum_sq = 0.0;
            for (int k = 0; k <= 100; ++k) {
                if (freq[k] > 0) {
                    const double kv = static_cast<double>(k);
                    sum += kv * freq[k];
                    sum_sq += kv * kv * freq[k];
                }
            }
            const double mean = sum / total_valid;
            const double var = (sum_sq / total_valid) - (mean * mean);
            return static_cast<float>(std::sqrt(std::max(0.0, var)));
        }
        case MetricType::ShannonDiversity: {
            double entropy = 0.0;
            const double da = static_cast<double>(total_valid);
            for (int k = 0; k <= 100; ++k) {
                if (freq[k] > 0) {
                    const double p = static_cast<double>(freq[k]) / da;
                    entropy -= p * std::log(p);
                }
            }
            return static_cast<float>(entropy);
        }
        case MetricType::ShannonEvenness: {
            double entropy = 0.0;
            int num_classes = 0;
            const double da = static_cast<double>(total_valid);
            for (int k = 0; k <= 100; ++k) {
                if (freq[k] > 0) {
                    ++num_classes;
                    const double p = static_cast<double>(freq[k]) / da;
                    entropy -= p * std::log(p);
                }
            }
            if (num_classes <= 1) return 0.0f;
            const double max_h = std::log(static_cast<double>(num_classes));
            return static_cast<float>(entropy / max_h);
        }
        default:
            return 0.0f;
    }
}

}  // namespace

Image<float> compute_gray_convolution(
    const ImageU8& im,
    const std::string& metric,
    int window_size,
    bool mask_missing,
    int n_threads
) {
    const MetricType mtype = parse_metric(metric);
    if (mtype == MetricType::Unknown) {
        throw std::invalid_argument("Unknown GraySpatCon metric: " + metric);
    }
    if (window_size < 3 || (window_size % 2) == 0) {
        throw std::invalid_argument("window_size must be an odd positive integer >= 3");
    }

    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const i64 hw = window_size / 2;

    Image<float> out(nx, ny, kMissingFloat);

#if defined(_OPENMP)
    if (n_threads > 0) omp_set_num_threads(n_threads);
#pragma omp parallel for schedule(dynamic)
#endif
    for (i64 y = 0; y < ny; ++y) {
        const i64 y0 = std::max(i64{0}, y - hw);
        const i64 y1 = std::min(ny - 1, y + hw);

        int freq[101] = {0};
        int total_valid = 0;

        // Initialize window for x = 0
        const i64 init_x1 = std::min(nx - 1, hw);
        for (i64 wy = y0; wy <= y1; ++wy) {
            const i64 r_off = wy * nx;
            for (i64 wx = 0; wx <= init_x1; ++wx) {
                const u8 val = im.px[static_cast<std::size_t>(r_off + wx)];
                if (val <= 100) {
                    freq[val]++;
                    total_valid++;
                }
            }
        }

        for (i64 x = 0; x < nx; ++x) {
            const i64 c_idx = y * nx + x;
            const u8 center_val = im.px[static_cast<std::size_t>(c_idx)];

            if (mask_missing && center_val > 100) {
                out.px[static_cast<std::size_t>(c_idx)] = kMissingFloat;
            } else {
                if (mtype == MetricType::DiagonalContagion) {
                    // 2-neighbor adjacencies within window
                    i64 diag_matches = 0;
                    i64 total_pairs = 0;
                    for (i64 wy = y0; wy <= y1; ++wy) {
                        const i64 r_off = wy * nx;
                        const i64 nr_off = (wy + 1) * nx;
                        for (i64 wx = std::max(i64{0}, x - hw); wx <= std::min(nx - 1, x + hw); ++wx) {
                            const u8 v = im.px[static_cast<std::size_t>(r_off + wx)];
                            if (v > 100) continue;
                            // Horizontal neighbor
                            if (wx + 1 <= std::min(nx - 1, x + hw)) {
                                const u8 vh = im.px[static_cast<std::size_t>(r_off + (wx + 1))];
                                if (vh <= 100) {
                                    total_pairs++;
                                    if (v == vh) diag_matches++;
                                }
                            }
                            // Vertical neighbor
                            if (wy + 1 <= y1) {
                                const u8 vv = im.px[static_cast<std::size_t>(nr_off + wx)];
                                if (vv <= 100) {
                                    total_pairs++;
                                    if (v == vv) diag_matches++;
                                }
                            }
                        }
                    }
                    if (total_pairs > 0) {
                        out.px[static_cast<std::size_t>(c_idx)] = static_cast<float>(diag_matches) / static_cast<float>(total_pairs);
                    } else {
                        out.px[static_cast<std::size_t>(c_idx)] = kMissingFloat;
                    }
                } else {
                    out.px[static_cast<std::size_t>(c_idx)] = eval_from_histogram(freq, total_valid, mtype);
                }
            }

            // Slide window to x + 1:
            // Remove column x - hw
            const i64 rem_x = x - hw;
            if (rem_x >= 0) {
                for (i64 wy = y0; wy <= y1; ++wy) {
                    const u8 v = im.px[static_cast<std::size_t>(wy * nx + rem_x)];
                    if (v <= 100) {
                        freq[v]--;
                        total_valid--;
                    }
                }
            }
            // Add column x + hw + 1
            const i64 add_x = x + hw + 1;
            if (add_x < nx) {
                for (i64 wy = y0; wy <= y1; ++wy) {
                    const u8 v = im.px[static_cast<std::size_t>(wy * nx + add_x)];
                    if (v <= 100) {
                        freq[v]++;
                        total_valid++;
                    }
                }
            }
        }
    }

    return out;
}

}  // namespace gtb
