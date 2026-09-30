// SPDX-License-Identifier: GPL-3.0-or-later
//
// Reading the feature layers as Zonation 5 does (statistics.cpp, layerdata.cpp):
// value extraction, NoData, thresholds, the maxima, and the quantization of each
// value to round(v / max * (2^bits - 1)); per-layer sums in double accumulated in
// row-major order within each of Zonation's blocks of rows (flush() ends one) and
// added block by block.

#include "prio/prio.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace prio {

namespace {

// The value Zonation works with and whether it is NoData: the extraction first
// (other values become the band's NoData, or 0 when it has none), then NoData
// (the NoData value or NaN) becomes 0.
inline float read_value(const ReadRules& r, int l, float raw, bool& nodata) {
    const float ev = r.extract[static_cast<size_t>(l)];
    const float nd = r.nodata[static_cast<size_t>(l)];
    if (!std::isnan(ev) && !(raw == ev)) raw = std::isnan(nd) ? 0.0f : nd;
    nodata = raw == nd || std::isnan(raw);
    return nodata ? 0.0f : raw;
}

void check_rules(const ReadRules& r) {
    const size_t n = static_cast<size_t>(r.layers);
    if (r.layers <= 0 || r.nodata.size() != n || r.extract.size() != n || r.thresholds.size() != n)
        throw std::invalid_argument("read rules: one nodata, extract and threshold value per layer");
}

}  // namespace

Stats::Stats(ReadRules r) : rules(std::move(r)) {
    check_rules(rules);
    const size_t n = static_cast<size_t>(rules.layers);
    zeros.assign(n, 0);
    negative.assign(n, 0);
    nodata.assign(n, 0);
    min_values.assign(n, std::numeric_limits<float>::max());
    max_values.assign(n, std::numeric_limits<float>::lowest());
    orig_max_values.assign(n, std::numeric_limits<float>::lowest());
}

void Stats::add_block(const float* values, i64 h, i64 w, const std::uint8_t* area) {
    const int L = rules.layers;
    const i64 plane = h * w;
    std::vector<float> f(static_cast<size_t>(L));
    std::vector<std::uint8_t> nd(static_cast<size_t>(L));
    for (i64 c = 0; c < plane; ++c) {
        ++pixels;
        for (int l = 0; l < L; ++l) {
            bool isnd;
            f[l] = read_value(rules, l, values[l * plane + c], isnd);
            nd[l] = isnd;
            // origMaxValues = features.max(0).max(origMaxValues)
            const float v0 = std::max(f[l], 0.0f);
            if (orig_max_values[l] < v0) orig_max_values[l] = v0;
        }
        if (area && !area[c]) {
            for (int l = 0; l < L; ++l) ++nodata[l];
            continue;
        }
        int nnd = 0;
        for (int l = 0; l < L; ++l) {
            nodata[l] += nd[l];
            nnd += nd[l];
        }
        if (nnd == L) continue;
        ++data_cells;
        for (int l = 0; l < L; ++l) {
            negative[l] += f[l] < 0.0f;
            const float clipped = static_cast<float>(f[l] >= rules.thresholds[l]) * f[l];
            zeros[l] += !nd[l] && std::fabs(clipped) == 0.0f;
            if (f[l] < min_values[l]) min_values[l] = f[l];
            if (max_values[l] < clipped) max_values[l] = clipped;
        }
    }
}

Data::Data(ReadRules r, std::vector<float> max_values, std::vector<std::uint8_t> keep, int b)
    : rules(std::move(r)), bits(b), keep_zeros(std::move(keep)) {
    check_rules(rules);
    const size_t n = static_cast<size_t>(rules.layers);
    if (max_values.size() != n || keep_zeros.size() != n)
        throw std::invalid_argument("data: one maximum and keep-zeros flag per layer");
    if (bits != 8 && bits != 16) throw std::invalid_argument("bits must be 8 or 16");
    if (rules.layers > 65535) throw std::invalid_argument("at most 65535 feature layers");
    storage_scale = bits == 8 ? 255.0 : 65535.0;
    this->max_values = std::move(max_values);
    inv_max.resize(n);
    for (size_t l = 0; l < n; ++l) {
        const float m = this->max_values[l];
        inv_max[l] = m > 0.0f ? 1.0f / m : 0.0f;
    }
    max_norm_sums.assign(n, 0.0);
    nq_sums.assign(n, 0.0);
    condition_sums.assign(n, 0.0);
    quantized_zeros.assign(n, 0);
    block_sums_.assign(n, 0.0);
    block_nq_.assign(n, 0.0);
    block_cond_.assign(n, 0.0);
}

void Data::add_block(const float* values, i64 h, i64 w, const std::uint8_t* area, const float* cond) {
    const int L = rules.layers;
    const bool conditioned = cond && !condition.empty();
    const i64 plane = h * w;
    const float scale = static_cast<float>(storage_scale);
    const double scale_r = 1.0 / storage_scale;
    std::vector<std::uint16_t> q(static_cast<size_t>(L));
    std::vector<std::uint8_t> skip(static_cast<size_t>(L));
    for (i64 c = 0; c < plane; ++c) {
        const i64 index = pixels++;
        if (area && !area[c]) continue;
        bool has_data = false;
        for (int l = 0; l < L; ++l) {
            bool nd;
            const float f = read_value(rules, l, values[l * plane + c], nd);
            const float clipped = static_cast<float>(f >= rules.thresholds[l]) * f;
            const float normalized = clipped * inv_max[l];
            const float scaled_nq = normalized * scale;
            q[l] = static_cast<std::uint16_t>(std::round(scaled_nq));
            skip[l] = nd || (!keep_zeros[l] && scaled_nq == 0.0f);
            has_data |= !skip[l];
            block_nq_[l] += static_cast<double>(normalized);
            block_sums_[l] += static_cast<double>(q[l]) * scale_r;
        }
        if (!has_data) continue;
        cell_index.push_back(index);
        for (int l = 0; l < L; ++l) {
            if (skip[l]) continue;
            if (conditioned && condition[l] >= 0) {
                // StorageType *= float: the product truncated
                const float cv = cond[static_cast<i64>(condition[l]) * plane + c];
                q[l] = static_cast<std::uint16_t>(static_cast<float>(q[l]) * cv);
            }
            entry_layer.push_back(static_cast<std::uint16_t>(l));
            entry_value.push_back(q[l]);
            quantized_zeros[l] += q[l] == 0;
            block_cond_[l] += static_cast<double>(q[l]) * scale_r;
        }
        cell_ptr.push_back(static_cast<i64>(entry_layer.size()));
    }
}

void Data::flush() {
    for (int l = 0; l < rules.layers; ++l) {
        max_norm_sums[l] += block_sums_[l];
        nq_sums[l] += block_nq_[l];
        condition_sums[l] += block_cond_[l];
    }
    std::fill(block_sums_.begin(), block_sums_.end(), 0.0);
    std::fill(block_nq_.begin(), block_nq_.end(), 0.0);
    std::fill(block_cond_.begin(), block_cond_.end(), 0.0);
}

void Data::finish() {
    flush();
    const size_t n = static_cast<size_t>(rules.layers);
    layer_scale.assign(n, 0.0);
    initial_levels.assign(n, 0.0);
    quantization_errors.assign(n, 0.0);
    orig_abs_sums.assign(n, 0.0);
    for (size_t l = 0; l < n; ++l) {
        const double s = max_norm_sums[l];
        orig_abs_sums[l] = s * static_cast<double>(max_values[l]);
        initial_levels[l] = s > 0 ? condition_sums[l] / s : 0.0;
        layer_scale[l] = s > 0 ? 1.0 / (s * storage_scale) : 0.0;
        if (l < renormalize.size() && renormalize[l] && condition[l] >= 0 && initial_levels[l] > 0) {
            layer_scale[l] /= initial_levels[l];
            initial_levels[l] = 1.0;
        }
        quantization_errors[l] = nq_sums[l] > 0 ? std::fabs(nq_sums[l] - s) / nq_sums[l] * 100.0 : 0.0;
    }
    cell_index.shrink_to_fit();
    cell_ptr.shrink_to_fit();
    entry_layer.shrink_to_fit();
    entry_value.shrink_to_fit();
}

i64 Data::retain(int layer, int kind, const float* values, float fraction) {
    if (layer < 0 || layer >= rules.layers) throw std::out_of_range("layer");
    const float scale = static_cast<float>(storage_scale);
    i64 clipped = 0;
    for (i64 c = 0; c < cells(); ++c)
        for (i64 e = cell_ptr[c]; e < cell_ptr[c + 1]; ++e) {
            if (entry_layer[e] != layer) continue;
            std::uint16_t& p = entry_value[e];
            const float v = static_cast<float>(p);
            float b;
            if (kind == 0) {
                // scaleToLayerStorage: the absolute value in the layer's quantized units
                b = values[c] * inv_max[layer] * scale;
                if (b > v) {
                    b = v;
                    ++clipped;
                }
            } else {
                b = (kind == 1 ? values[c] : fraction) * v;
            }
            p = static_cast<std::uint16_t>(v - b);
            break;
        }
    return clipped;
}

std::vector<float> Data::preprocessing(const float* mult, const float* weights, bool unscaled) const {
    const int L = rules.layers;
    std::vector<float> out(static_cast<size_t>(cells()));
#pragma omp parallel
    {
        std::vector<float> f(static_cast<size_t>(L));
#pragma omp for schedule(static)
        for (i64 c = 0; c < cells(); ++c) {
            std::fill(f.begin(), f.end(), 0.0f);
            for (i64 e = cell_ptr[c]; e < cell_ptr[c + 1]; ++e) {
                const int l = entry_layer[e];
                f[l] = static_cast<float>(static_cast<double>(static_cast<float>(entry_value[e])) * layer_scale[l]);
            }
            for (int l = 0; l < L; ++l) {
                float v = weights ? weights[l] * f[l] : f[l];
                if (unscaled) v = static_cast<float>(static_cast<double>(v) * orig_abs_sums[l]);
                f[l] = v * mult[l];
            }
            out[c] = eigen_avx_sum(f.data(), L, 0);
        }
    }
    return out;
}

double Data::replace_layer(int layer, std::vector<float> packed, double level) {
    if (layer < 0 || layer >= rules.layers) throw std::out_of_range("layer");
    if (static_cast<i64>(packed.size()) != cells()) throw std::invalid_argument("one value per data cell");
    const float fscale = static_cast<float>(storage_scale);
    const double scale_r = 1.0 / storage_scale;
    float maxv = 0.0f;
    for (float v : packed) maxv = std::max(v, maxv);
    double sum = 0.0, nqsum = 0.0;
    if (maxv > 0) {
        const float m = fscale / maxv;
        for (float& v : packed) v = std::min(v * m, fscale);
        for (float v : packed) nqsum += static_cast<double>(v) * scale_r;
        for (float& v : packed) v = std::min(std::round(v), fscale);
        for (float v : packed) sum += static_cast<double>(static_cast<std::uint16_t>(v)) * scale_r;
    }
    layer_scale[layer] = sum > 0 ? level / (sum * storage_scale) : 0.0;
    initial_levels[layer] = level;
    for (i64 c = 0; c < cells(); ++c)
        for (i64 e = cell_ptr[c]; e < cell_ptr[c + 1]; ++e)
            if (entry_layer[e] == layer) {
                entry_value[e] = static_cast<std::uint16_t>(packed[c]);
                break;
            }
    return nqsum > 0 ? std::fabs(nqsum - sum) / nqsum * 100.0 : 0.0;
}

double eigen_avx_sum(const double* v, i64 n, i64 offset) {
    // Packet4d: predux adds the halves (k + k+2), then 0 + 1
    if (n <= 0) return 0.0;
    const i64 P = 4;
    const i64 start = std::min<i64>((P - offset % P) % P, n);
    const i64 size2 = ((n - start) / (2 * P)) * (2 * P);
    const i64 size1 = ((n - start) / P) * P;
    const i64 end2 = start + size2, end1 = start + size1;
    double res;
    if (size1) {
        double r0[4], r1[4];
        for (int k = 0; k < 4; ++k) r0[k] = v[start + k];
        if (size1 > P) {
            for (int k = 0; k < 4; ++k) r1[k] = v[start + P + k];
            for (i64 i = start + 2 * P; i < end2; i += 2 * P)
                for (int k = 0; k < 4; ++k) {
                    r0[k] = r0[k] + v[i + k];
                    r1[k] = r1[k] + v[i + P + k];
                }
            for (int k = 0; k < 4; ++k) r0[k] = r0[k] + r1[k];
            if (end1 > end2)
                for (int k = 0; k < 4; ++k) r0[k] = r0[k] + v[end2 + k];
        }
        res = (r0[0] + r0[2]) + (r0[1] + r0[3]);
        for (i64 i = 0; i < start; ++i) res = res + v[i];
        for (i64 i = end1; i < n; ++i) res = res + v[i];
    } else {
        res = v[0];
        for (i64 i = 1; i < n; ++i) res = res + v[i];
    }
    return res;
}

float eigen_avx_sum(const float* v, i64 n, i64 offset) {
    // Redux.h, LinearVectorizedTraversal: packets of 8 from the first 32-byte
    // aligned element, two accumulators, predux (lanes k + k+4, then (0+2)+(1+3)),
    // then the elements before and after the packets.
    if (n <= 0) return 0.0f;
    const i64 P = 8;
    const i64 start = std::min<i64>((P - offset % P) % P, n);
    const i64 size2 = ((n - start) / (2 * P)) * (2 * P);
    const i64 size1 = ((n - start) / P) * P;
    const i64 end2 = start + size2, end1 = start + size1;
    float res;
    if (size1) {
        float r0[8], r1[8];
        for (int k = 0; k < 8; ++k) r0[k] = v[start + k];
        if (size1 > P) {
            for (int k = 0; k < 8; ++k) r1[k] = v[start + P + k];
            for (i64 i = start + 2 * P; i < end2; i += 2 * P) {
                for (int k = 0; k < 8; ++k) {
                    r0[k] = r0[k] + v[i + k];
                    r1[k] = r1[k] + v[i + P + k];
                }
            }
            for (int k = 0; k < 8; ++k) r0[k] = r0[k] + r1[k];
            if (end1 > end2)
                for (int k = 0; k < 8; ++k) r0[k] = r0[k] + v[end2 + k];
        }
        float b[4];
        for (int k = 0; k < 4; ++k) b[k] = r0[k] + r0[k + 4];
        res = (b[0] + b[2]) + (b[1] + b[3]);
        for (i64 i = 0; i < start; ++i) res = res + v[i];
        for (i64 i = end1; i < n; ++i) res = res + v[i];
    } else {
        res = v[0];
        for (i64 i = 1; i < n; ++i) res = res + v[i];
    }
    return res;
}

}  // namespace prio
