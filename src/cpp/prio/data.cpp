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
#include <cstring>
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

// In parallel over the layers (counts, minima and maxima do not depend on the order).
void Stats::add_block(const float* values, i64 h, i64 w, const std::uint8_t* area) {
    const int L = rules.layers;
    const i64 plane = h * w;
    // cells with data: in the area and not NoData in every layer
    std::vector<std::uint8_t> valid(static_cast<size_t>(plane), 0);
    i64 valid_cells = 0;
#pragma omp parallel for schedule(static) reduction(+ : valid_cells)
    for (i64 c = 0; c < plane; ++c) {
        if (area && !area[c]) continue;
        int nnd = 0;
        for (int l = 0; l < L; ++l) {
            bool isnd;
            read_value(rules, l, values[l * plane + c], isnd);
            nnd += isnd;
        }
        valid[c] = nnd < L;
        valid_cells += nnd < L;
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (int l = 0; l < L; ++l) {
        const float* v = values + l * plane;
        i64 nd_count = 0, neg = 0, zero = 0;
        float mn = min_values[l], mx = max_values[l], omx = orig_max_values[l];
        for (i64 c = 0; c < plane; ++c) {
            bool isnd;
            const float f = read_value(rules, l, v[c], isnd);
            // origMaxValues = features.max(0).max(origMaxValues)
            const float v0 = std::max(f, 0.0f);
            if (omx < v0) omx = v0;
            if (area && !area[c]) {
                ++nd_count;
                continue;
            }
            nd_count += isnd;
            if (!valid[c]) continue;
            neg += f < 0.0f;
            const float clipped = static_cast<float>(f >= rules.thresholds[l]) * f;
            zero += !isnd && std::fabs(clipped) == 0.0f;
            if (f < mn) mn = f;
            if (mx < clipped) mx = clipped;
        }
        nodata[l] += nd_count;
        negative[l] += neg;
        zeros[l] += zero;
        min_values[l] = mn;
        max_values[l] = mx;
        orig_max_values[l] = omx;
    }
    pixels += plane;
    data_cells += valid_cells;
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

// The layers are quantized in parallel, each summed in cell order as Zonation
// does; the runs of the cells are written in parallel chunks, then joined.
void Data::add_block(const float* values, i64 h, i64 w, const std::uint8_t* area, const float* cond) {
    const int L = rules.layers;
    const bool conditioned = cond && !condition.empty();
    const i64 plane = h * w;
    const float scale = static_cast<float>(storage_scale);
    const double scale_r = 1.0 / storage_scale;
    std::vector<std::uint16_t> q(static_cast<size_t>(L) * static_cast<size_t>(plane));
    std::vector<std::uint8_t> skip(static_cast<size_t>(L) * static_cast<size_t>(plane), 1);
#pragma omp parallel for schedule(dynamic, 1)
    for (int l = 0; l < L; ++l) {
        const float* v = values + l * plane;
        std::uint16_t* ql = q.data() + l * plane;
        std::uint8_t* sl = skip.data() + l * plane;
        double nq = block_nq_[l], sum = block_sums_[l];
        for (i64 c = 0; c < plane; ++c) {
            if (area && !area[c]) continue;
            bool nd;
            const float f = read_value(rules, l, v[c], nd);
            const float clipped = static_cast<float>(f >= rules.thresholds[l]) * f;
            const float normalized = clipped * inv_max[l];
            const float scaled_nq = normalized * scale;
            ql[c] = static_cast<std::uint16_t>(std::round(scaled_nq));
            sl[c] = nd || (!keep_zeros[l] && scaled_nq == 0.0f);
            nq += static_cast<double>(normalized);
            sum += static_cast<double>(ql[c]) * scale_r;
        }
        block_nq_[l] = nq;
        block_sums_[l] = sum;
    }
    // the cells with data: conditions, runs
    std::vector<std::uint8_t> has(static_cast<size_t>(plane), 0);
    const int T = std::max(1, std::min(64, static_cast<int>(plane / 4096) + 1));
    std::vector<std::vector<std::uint8_t>> bytes(static_cast<size_t>(T));
    std::vector<std::vector<i64>> index(static_cast<size_t>(T)), ends(static_cast<size_t>(T));
    std::vector<std::vector<i64>> qzeros(static_cast<size_t>(T), std::vector<i64>(static_cast<size_t>(L), 0));
    const int vb = value_bytes();
#pragma omp parallel for schedule(dynamic, 1)
    for (int t = 0; t < T; ++t) {
        runs::Writer writer(vb);
        auto& out = bytes[t];
        i64* qz = qzeros[t].data();
        for (i64 c = plane * t / T; c < plane * (t + 1) / T; ++c) {
            if (area && !area[c]) continue;
            bool any = false;
            for (int l = 0; l < L && !any; ++l) any = !skip[static_cast<size_t>(l) * plane + c];
            if (!any) continue;
            has[c] = 1;
            for (int l = 0; l < L; ++l) {
                const size_t k = static_cast<size_t>(l) * plane + c;
                if (skip[k]) continue;
                if (conditioned && condition[l] >= 0) {
                    // StorageType *= float: the product truncated
                    const float cv = cond[static_cast<i64>(condition[l]) * plane + c];
                    q[k] = static_cast<std::uint16_t>(static_cast<float>(q[k]) * cv);
                }
                writer.add(l, q[k] ? runs::VALUE : runs::ZERO, q[k]);
                qz[l] += q[k] == 0;
            }
            writer.finish(out);
            index[t].push_back(pixels + c);
            ends[t].push_back(static_cast<i64>(out.size()));
        }
    }
    for (int t = 0; t < T; ++t) {
        const i64 base = static_cast<i64>(runs.size());
        runs.append(bytes[t].data(), bytes[t].size());
        cell_index.append(index[t].data(), index[t].size());
        for (i64 e : ends[t]) cell_ptr.push_back(base + e);
        for (int l = 0; l < L; ++l) quantized_zeros[l] += qzeros[t][l];
    }
    // the sums of the (conditioned) values, per layer in cell order
#pragma omp parallel for schedule(dynamic, 1)
    for (int l = 0; l < L; ++l) {
        const std::uint16_t* ql = q.data() + l * plane;
        const std::uint8_t* sl = skip.data() + l * plane;
        double sum = block_cond_[l];
        for (i64 c = 0; c < plane; ++c)
            if (has[c] && !sl[c]) sum += static_cast<double>(ql[c]) * scale_r;
        block_cond_[l] = sum;
    }
    pixels += plane;
}

void Data::set_storage(const std::string& dir) {
    if (cells() || pixels) throw std::logic_error("the storage is set before the data");
    cell_index.set_storage(dir);
    cell_ptr.set_storage(dir);
    cell_ptr.push_back(0);
    runs.set_storage(dir);
}

long Data::value_at(i64 c, int layer) const {
    const std::uint8_t* p = row(c);
    const std::uint8_t* end = row_end(c);
    const int vb = value_bytes();
    i64 l = 0;
    while (p < end) {
        unsigned kind;
        i64 len;
        p = runs::read_run(p, kind, len);
        if (layer < l + len) {
            if (kind == runs::VALUE) return static_cast<long>(runs::read_value(p + (layer - l) * vb, vb));
            return kind == runs::ZERO ? 0 : -1;
        }
        if (kind == runs::VALUE) p += len * vb;
        l += len;
    }
    return -1;
}

i64 Data::entries() const {
    i64 n = 0;
    const int vb = value_bytes();
#pragma omp parallel for schedule(static) reduction(+ : n)
    for (i64 c = 0; c < cells(); ++c) {
        const std::uint8_t* p = row(c);
        const std::uint8_t* end = row_end(c);
        while (p < end) {
            unsigned kind;
            i64 len;
            p = runs::read_run(p, kind, len);
            if (kind != runs::NODATA) n += len;
            if (kind == runs::VALUE) p += len * vb;
        }
    }
    return n;
}

void Data::update_layer(int layer, const std::function<unsigned(i64, unsigned)>& f) {
    if (layer < 0 || layer >= rules.layers) throw std::out_of_range("layer");
    const i64 N = cells();
    const int vb = value_bytes();
    // the runs of zeros that get a value: their header (offset in the runs), first
    // layer and length, the value, and the bytes the runs replacing them add
    struct Split {
        i64 at = -1;
        i64 start = 0, len = 0;
        unsigned q = 0;
        i64 extra = 0;
    };
    std::vector<Split> split(static_cast<size_t>(N));
    int any = 0;
#pragma omp parallel for schedule(static) reduction(| : any)
    for (i64 c = 0; c < N; ++c) {
        std::uint8_t* base = runs.data();
        const std::uint8_t* p = base + cell_ptr[c];
        const std::uint8_t* end = base + cell_ptr[c + 1];
        i64 l = 0;
        while (p < end) {
            const std::uint8_t* h = p;
            unsigned kind;
            i64 len;
            p = runs::read_run(p, kind, len);
            if (layer < l + len) {
                if (kind == runs::VALUE) {
                    std::uint8_t* v = base + (p - base) + (layer - l) * vb;
                    runs::write_value(v, f(c, runs::read_value(v, vb)), vb);
                } else if (kind == runs::ZERO) {
                    const unsigned q = f(c, 0u);
                    if (q) {
                        const i64 left = layer - l, right = l + len - 1 - layer;
                        const i64 bytes = (left ? runs::header_bytes(left) : 0) + runs::header_bytes(1) + vb +
                                          (right ? runs::header_bytes(right) : 0);
                        split[c] = {h - base, l, len, q, bytes - (p - h)};
                        any = 1;
                    }
                }
                break;
            }
            if (kind == runs::VALUE) p += len * vb;
            l += len;
        }
    }
    if (!any) return;
    // rewrite the runs: the split runs replaced, the rest copied
    Buffer<i64> ptr;
    ptr.set_storage(cell_ptr.storage());
    ptr.resize(static_cast<size_t>(N) + 1);
    for (i64 c = 0; c < N; ++c) ptr[c + 1] = ptr[c] + (cell_ptr[c + 1] - cell_ptr[c]) + split[c].extra;
    Buffer<std::uint8_t> out;
    out.set_storage(runs.storage());
    out.resize(static_cast<size_t>(ptr[N]));
#pragma omp parallel for schedule(static)
    for (i64 c = 0; c < N; ++c) {
        const std::uint8_t* src = runs.data() + cell_ptr[c];
        std::uint8_t* dst = out.data() + ptr[c];
        const i64 n = cell_ptr[c + 1] - cell_ptr[c];
        const Split& s = split[c];
        if (s.at < 0) {
            std::memcpy(dst, src, static_cast<size_t>(n));
            continue;
        }
        const i64 head = s.at - cell_ptr[c];  // bytes before the run
        std::memcpy(dst, src, static_cast<size_t>(head));
        std::uint8_t* d = dst + head;
        const i64 left = layer - s.start, right = s.start + s.len - 1 - layer;
        if (left) d = runs::write_run(d, runs::ZERO, left);
        d = runs::write_run(d, runs::VALUE, 1);
        d = runs::write_value(d, s.q, vb);
        if (right) d = runs::write_run(d, runs::ZERO, right);
        const i64 old = runs::header_bytes(s.len);
        std::memcpy(d, src + head + old, static_cast<size_t>(n - head - old));
    }
    runs.swap(out);
    cell_ptr.swap(ptr);
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
    runs.shrink_to_fit();
}

i64 Data::retain(int layer, int kind, const float* values, float fraction) {
    if (layer < 0 || layer >= rules.layers) throw std::out_of_range("layer");
    const float scale = static_cast<float>(storage_scale);
    std::vector<std::uint8_t> clip(static_cast<size_t>(cells()), 0);
    update_layer(layer, [&](i64 c, unsigned q) {
        const float v = static_cast<float>(q);
        float b;
        if (kind == 0) {
            // scaleToLayerStorage: the absolute value in the layer's quantized units
            b = values[c] * inv_max[layer] * scale;
            if (b > v) {
                b = v;
                clip[c] = 1;
            }
        } else {
            b = (kind == 1 ? values[c] : fraction) * v;
        }
        return static_cast<unsigned>(static_cast<std::uint16_t>(v - b));
    });
    i64 clipped = 0;
    for (std::uint8_t k : clip) clipped += k;
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
            each<false>(c, [&](int l, unsigned q) {
                f[l] = static_cast<float>(static_cast<double>(static_cast<float>(q)) * layer_scale[l]);
            });
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
    update_layer(layer,
                 [&](i64 c, unsigned) { return static_cast<unsigned>(static_cast<std::uint16_t>(packed[c])); });
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
