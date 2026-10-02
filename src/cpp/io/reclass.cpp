// SPDX-License-Identifier: GPL-3.0-or-later
//
// ls.reclassify: the new value of every cell in one pass, in parallel. 8- and
// 16-bit rasters go through a table of every value of their type; the others
// look every value up in the sorted rules.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "io/io.hpp"

namespace lsio {

namespace {

template <class Out, class T>
constexpr bool holds_every_value() {
    using In = std::numeric_limits<T>;
    using O = std::numeric_limits<Out>;
    return std::is_integral_v<T> && static_cast<long double>(O::min()) <= static_cast<long double>(In::min()) &&
           static_cast<long double>(In::max()) <= static_cast<long double>(O::max());
}

// v as Out, when the value is kept: exact where Out holds every value of T, else
// through double, truncated toward zero through int32 (8- to 32-bit signed and
// 8-, 16-bit unsigned types) or int64, out of its range its minimum: NumPy's
// casts on x86 (the old float64 path), without undefined behaviour.
template <class Out, class T>
inline Out keep_cast(T v) {
    if constexpr (std::is_floating_point_v<Out>) {
        return static_cast<Out>(static_cast<double>(v));  // through double, as NumPy did
    } else if constexpr (holds_every_value<Out, T>()) {
        return static_cast<Out>(v);
    } else {
        using Via = std::conditional_t<(sizeof(Out) < 4 || std::is_same_v<Out, std::int32_t>), std::int32_t,
                                       std::int64_t>;
        const double x = static_cast<double>(v);
        constexpr double lim = static_cast<double>(std::numeric_limits<Via>::max()) + 1.0;  // 2^31 or 2^63
        if (!(x > -lim - 1.0 && x < lim)) return static_cast<Out>(std::numeric_limits<Via>::min());
        return static_cast<Out>(static_cast<Via>(x));
    }
}

// The number of the n sorted values a below x (or, upper, not above x), without
// branches to mispredict on rasters whose values change from cell to cell.
template <bool upper>
inline i64 rank(const double* a, i64 n, double x) {
    const double* base = a;
    while (n > 1) {
        const i64 half = n / 2;
        base = (upper ? !(x < base[half]) : base[half] < x) ? base + half : base;
        n -= half;
    }
    return static_cast<i64>(base - a) + (upper ? !(x < *base) : *base < x);
}

template <class T, class Out>
inline Out classify(T v, const ReclassRules<Out>& r) {
    const double x = static_cast<double>(v);
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(x)) return r.missing;
    }
    if (r.has_nodata && x == r.nodata) return r.missing;
    if (r.n_breaks > 1) {  // the ranges first: they win over the single values
        const i64 i = rank<true>(r.breaks, r.n_breaks, x) - 1;
        if (i >= 0 && i < r.n_breaks - 1 && r.break_set[i]) return r.break_out[i];
    }
    if (r.n_keys > 0) {
        const i64 k = rank<false>(r.keys, r.n_keys, x);
        if (k < r.n_keys && r.keys[k] == x) return r.key_out[k];
    }
    return r.keep ? keep_cast<Out>(v) : r.other;
}

constexpr i64 kParallelMin = i64{1} << 15;  // fewer cells: one thread

// out[i] = t[in[i] - lo] for the cells
template <class T, class Out>
void map_table(const T* in, Out* out, i64 n, const Out* t, i64 lo) {
#pragma omp parallel for schedule(static) if (n >= kParallelMin)
    for (i64 i = 0; i < n; ++i) out[i] = t[static_cast<i64>(in[i]) - lo];
}

}  // namespace

template <class T, class Out>
void reclassify(const T* in, Out* out, i64 n, const ReclassRules<Out>& r) {
    if constexpr (std::is_integral_v<T> && sizeof(T) <= 2) {
        constexpr i64 lo = std::numeric_limits<T>::min();
        constexpr i64 size = static_cast<i64>(std::numeric_limits<T>::max()) - lo + 1;
        if (n >= size / 4) {  // a table of every value of the type
            std::vector<Out> lut(static_cast<std::size_t>(size));
            for (i64 k = 0; k < size; ++k) lut[static_cast<std::size_t>(k)] = classify(static_cast<T>(k + lo), r);
            map_table(in, out, n, lut.data(), lo);
            return;
        }
    } else if constexpr (std::is_integral_v<T>) {
        if (n >= kParallelMin) {  // a table of the values from the raster's minimum to its maximum, if few
            T lo = in[0], hi = in[0];
#pragma omp parallel
            {
                T tlo = in[0], thi = in[0];
#pragma omp for schedule(static) nowait
                for (i64 i = 0; i < n; ++i) {
                    tlo = std::min(tlo, in[i]);
                    thi = std::max(thi, in[i]);
                }
#pragma omp critical(lsio_reclass_range)
                {
                    lo = std::min(lo, tlo);
                    hi = std::max(hi, thi);
                }
            }
            const std::uint64_t span = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo);
            if (span < static_cast<std::uint64_t>(std::min<i64>(n / 2, i64{1} << 22))) {
                const i64 size = static_cast<i64>(span) + 1;
                std::vector<Out> lut(static_cast<std::size_t>(size));
                Out* t = lut.data();
#pragma omp parallel for schedule(static) if (size >= kParallelMin)
                for (i64 k = 0; k < size; ++k) t[k] = classify(static_cast<T>(lo + static_cast<T>(k)), r);
                // in[i] - lo as i64: exact, the span is small
                map_table(in, out, n, t, static_cast<i64>(lo));
                return;
            }
        }
    }
#pragma omp parallel for schedule(static) if (n >= kParallelMin)
    for (i64 i = 0; i < n; ++i) out[i] = classify(in[i], r);
}

#define LSIO_RECLASS_OUT(T, Out) template void reclassify<T, Out>(const T*, Out*, i64, const ReclassRules<Out>&);
#define LSIO_RECLASS(T)                    \
    LSIO_RECLASS_OUT(T, std::uint8_t)      \
    LSIO_RECLASS_OUT(T, std::int8_t)       \
    LSIO_RECLASS_OUT(T, std::uint16_t)     \
    LSIO_RECLASS_OUT(T, std::int16_t)      \
    LSIO_RECLASS_OUT(T, std::uint32_t)     \
    LSIO_RECLASS_OUT(T, std::int32_t)      \
    LSIO_RECLASS_OUT(T, std::int64_t)      \
    LSIO_RECLASS_OUT(T, float)             \
    LSIO_RECLASS_OUT(T, double)

LSIO_RECLASS(std::uint8_t)
LSIO_RECLASS(std::int8_t)
LSIO_RECLASS(std::uint16_t)
LSIO_RECLASS(std::int16_t)
LSIO_RECLASS(std::uint32_t)
LSIO_RECLASS(std::int32_t)
LSIO_RECLASS(std::int64_t)
LSIO_RECLASS(float)
LSIO_RECLASS(double)

#undef LSIO_RECLASS
#undef LSIO_RECLASS_OUT

}  // namespace lsio
