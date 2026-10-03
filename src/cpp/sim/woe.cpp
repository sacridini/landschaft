// SPDX-License-Identifier: GPL-3.0-or-later
//
// Weights of evidence on rasters (see woe.hpp). The counts are summed per
// thread and added in thread order: integers, the same totals with any
// thread count.

#include "sim/woe.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "lsm/parallel.hpp"

namespace sim {

namespace {

using i64 = std::int64_t;
constexpr i64 kParallelCells = 1 << 15;
constexpr double kEdge = 1e-14;  // a value this close below a limit belongs above it (Dinamica EGO)

template <class T>
bool is_missing(T v, const MapView& m) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(v)) return true;
    }
    return m.has_nodata && static_cast<double>(v) == m.nodata;
}

template <class F>
auto with_values(const MapView& m, F&& f) {
    switch (m.type) {
        case DType::u8: return f(static_cast<const std::uint8_t*>(m.data));
        case DType::i8: return f(static_cast<const std::int8_t*>(m.data));
        case DType::u16: return f(static_cast<const std::uint16_t*>(m.data));
        case DType::i16: return f(static_cast<const std::int16_t*>(m.data));
        case DType::u32: return f(static_cast<const std::uint32_t*>(m.data));
        case DType::i32: return f(static_cast<const std::int32_t*>(m.data));
        case DType::i64: return f(static_cast<const std::int64_t*>(m.data));
        case DType::f32: return f(static_cast<const float*>(m.data));
        case DType::f64: return f(static_cast<const double*>(m.data));
    }
    throw std::invalid_argument("unsupported map type");
}

int threads(bool parallel) {
#ifdef _OPENMP
    return parallel ? omp_get_max_threads() : 1;
#else
    (void)parallel;
    return 1;
#endif
}

std::int32_t class_index(const std::vector<double>& classes, double v) {
    auto it = std::lower_bound(classes.begin(), classes.end(), v);
    return it != classes.end() && *it == v ? static_cast<std::int32_t>(it - classes.begin()) : -1;
}

}  // namespace

std::vector<std::int32_t> woe_buckets(const MapView& var, bool categorical, const std::vector<double>& classes,
                                      double start, double inc, i64 n_buckets) {
    if (!categorical && !(inc > 0)) throw std::invalid_argument("the increment must be > 0");
    std::vector<std::int32_t> out(static_cast<std::size_t>(var.n), -1);
    const bool parallel = var.n >= kParallelCells;
    with_values(var, [&](auto* v) {
        lsm::parallel_for(parallel, var.n, [&](i64 i) {
            const auto x = v[i];
            if (is_missing(x, var)) return;
            const double d = static_cast<double>(x);
            if (categorical) {
                out[i] = class_index(classes, d);
                return;
            }
            double q = std::floor((d - start) / inc);
            if (!(q >= -1) || q > static_cast<double>(n_buckets)) return;
            i64 k = static_cast<i64>(q);
            while (k + 1 < n_buckets && d >= start + static_cast<double>(k + 1) * inc - kEdge) ++k;
            while (k > 0 && d < start + static_cast<double>(k) * inc - kEdge) --k;
            if (k >= 0 && k < n_buckets) out[i] = static_cast<std::int32_t>(k);
        });
        return 0;
    });
    return out;
}

std::vector<std::int64_t> woe_counts(const MapView& t0, const MapView& t1, const std::vector<std::int32_t>& buckets,
                                     i64 n_buckets, double from, const std::vector<double>& final_classes) {
    if (t0.n != t1.n || t0.n != static_cast<i64>(buckets.size())) throw std::invalid_argument("the maps differ in size");
    const std::size_t nc = final_classes.size();
    const bool parallel = t0.n >= kParallelCells;
    const int nt = threads(parallel);
    std::vector<std::vector<std::int64_t>> part(nt);
    const i64 chunk = (t0.n + nt - 1) / std::max(nt, 1);
    with_values(t0, [&](auto* a) {
        return with_values(t1, [&](auto* b) {
            lsm::parallel_for(parallel && nt > 1, nt, [&](i64 t) {
                auto& c = part[t];
                c.assign(static_cast<std::size_t>(n_buckets) * nc, 0);
                const i64 lo = t * chunk, hi = std::min(t0.n, lo + chunk);
                for (i64 i = lo; i < hi; ++i) {
                    const std::int32_t k = buckets[i];
                    if (k < 0 || is_missing(a[i], t0) || is_missing(b[i], t1)) continue;
                    if (static_cast<double>(a[i]) != from) continue;
                    const std::int32_t j = class_index(final_classes, static_cast<double>(b[i]));
                    if (j >= 0) ++c[static_cast<std::size_t>(k) * nc + static_cast<std::size_t>(j)];
                }
            });
            return 0;
        });
    });
    std::vector<std::int64_t> out(static_cast<std::size_t>(n_buckets) * nc, 0);
    for (const auto& c : part)
        for (std::size_t k = 0; k < c.size(); ++k) out[k] += c[k];
    return out;
}

std::vector<double> woe_probability(const MapView& landscape, double from, const std::vector<const std::int32_t*>& buckets,
                                    const std::vector<std::vector<double>>& weights, i64 n) {
    if (buckets.size() != weights.size()) throw std::invalid_argument("one weight table per variable");
    std::vector<double> out(static_cast<std::size_t>(n), std::numeric_limits<double>::quiet_NaN());
    const bool parallel = n >= kParallelCells;
    with_values(landscape, [&](auto* a) {
        lsm::parallel_for(parallel, n, [&](i64 i) {
            if (is_missing(a[i], landscape) || static_cast<double>(a[i]) != from) return;
            double s = 0.0;
            for (std::size_t v = 0; v < buckets.size(); ++v) {
                const std::int32_t k = buckets[v][i];
                if (k < 0 || static_cast<std::size_t>(k) >= weights[v].size()) continue;  // missing: weight 0
                s += weights[v][static_cast<std::size_t>(k)];
            }
            out[i] = s;
        });
        return 0;
    });
    return out;
}

}  // namespace sim
