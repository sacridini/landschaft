// SPDX-License-Identifier: GPL-3.0-or-later
//
// Cross-tabulating categorical maps (see change.hpp). Values are mapped to
// class indices through a look-up table when the classes are integers in a
// short range (the usual land use map) and by binary search otherwise; the
// counts are summed per thread and then added in thread order (integers, so
// the totals do not depend on the thread count).

#include "sim/change.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "lsm/parallel.hpp"

namespace sim {

namespace {

using i64 = std::int64_t;

constexpr i64 kParallelCells = 1 << 15;
constexpr double kMaxLutRange = 1 << 24;

template <class T>
bool missing(T v, const MapView& m) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(v)) return true;
    }
    return m.has_nodata && static_cast<double>(v) == m.nodata;
}

template <class F>
auto dispatch(const MapView& m, F&& f) {
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

int thread_count(bool parallel) {
#ifdef _OPENMP
    return parallel ? omp_get_max_threads() : 1;
#else
    (void)parallel;
    return 1;
#endif
}

int thread_id() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// Value -> class index (-1: not a class) for one map.
struct ClassIndex {
    std::vector<double> classes;
    bool lut = false;
    double lo = 0;
    std::vector<std::int32_t> table;  // lut: index of value lo + k

    explicit ClassIndex(const std::vector<double>& c) : classes(c) {
        if (classes.empty()) return;
        bool integral = std::all_of(classes.begin(), classes.end(), [](double v) { return v == std::floor(v); });
        const double range = classes.back() - classes.front();
        if (integral && range < kMaxLutRange) {
            lut = true;
            lo = classes.front();
            table.assign(static_cast<std::size_t>(range) + 1, -1);
            for (std::size_t k = 0; k < classes.size(); ++k)
                table[static_cast<std::size_t>(classes[k] - lo)] = static_cast<std::int32_t>(k);
        }
    }

    std::int32_t operator()(double v) const {
        if (lut) {
            const double k = v - lo;
            if (!(k >= 0) || k >= static_cast<double>(table.size()) || v != std::floor(v)) return -1;
            return table[static_cast<std::size_t>(k)];
        }
        auto it = std::lower_bound(classes.begin(), classes.end(), v);
        return it != classes.end() && *it == v ? static_cast<std::int32_t>(it - classes.begin()) : -1;
    }
};

}  // namespace

std::vector<double> map_classes(const MapView& m) {
    return dispatch(m, [&](auto* v) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(v)>>;
        const bool parallel = m.n >= kParallelCells;
        const int nt = thread_count(parallel);
        // pass 1: range and integrality of the valid values
        std::vector<double> lo(nt, std::numeric_limits<double>::infinity());
        std::vector<double> hi(nt, -std::numeric_limits<double>::infinity());
        std::vector<char> integral(nt, 1);
        const i64 chunk = (m.n + nt - 1) / std::max(nt, 1);
        lsm::parallel_for(parallel && nt > 1, nt, [&](i64 t) {
            const i64 b = t * chunk, e = std::min(m.n, b + chunk);
            double l = lo[t], h = hi[t];
            bool in = true;
            for (i64 i = b; i < e; ++i) {
                const T x = v[i];
                if (missing(x, m)) continue;
                const double d = static_cast<double>(x);
                l = std::min(l, d);
                h = std::max(h, d);
                if constexpr (std::is_floating_point_v<T>) in = in && d == std::floor(d) && std::isfinite(d);
            }
            lo[t] = l, hi[t] = h, integral[t] = in;
        });
        double l = std::numeric_limits<double>::infinity(), h = -l;
        bool in = true;
        for (int t = 0; t < nt; ++t) l = std::min(l, lo[t]), h = std::max(h, hi[t]), in = in && integral[t];
        std::vector<double> out;
        if (!(l <= h)) return out;
        if (in && h - l < kMaxLutRange) {
            // pass 2: which values of the range occur (a byte per value and thread, OR-ed)
            const std::size_t width = static_cast<std::size_t>(h - l) + 1;
            std::vector<std::vector<std::uint8_t>> seen(nt);
            lsm::parallel_for(parallel && nt > 1, nt, [&](i64 t) {
                auto& s = seen[t];
                s.assign(width, 0);
                const i64 b = t * chunk, e = std::min(m.n, b + chunk);
                for (i64 i = b; i < e; ++i) {
                    const T x = v[i];
                    if (!missing(x, m)) s[static_cast<std::size_t>(static_cast<double>(x) - l)] = 1;
                }
            });
            for (std::size_t k = 0; k < width; ++k) {
                bool any = false;
                for (int t = 0; t < nt && !any; ++t) any = seen[t][k] != 0;
                if (any) out.push_back(l + static_cast<double>(k));
            }
            return out;
        }
        out.reserve(static_cast<std::size_t>(m.n));
        for (i64 i = 0; i < m.n; ++i)
            if (!missing(v[i], m)) out.push_back(static_cast<double>(v[i]));
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    });
}

std::vector<std::int64_t> crosstab(const MapView& a, const MapView& b, const std::vector<double>& classes_a,
                                   const std::vector<double>& classes_b) {
    if (a.n != b.n) throw std::invalid_argument("the maps differ in size");
    const ClassIndex ia(classes_a), ib(classes_b);
    const std::size_t na = classes_a.size(), nb = classes_b.size();
    const bool parallel = a.n >= kParallelCells;
    const int nt = thread_count(parallel);
    std::vector<std::vector<std::int64_t>> part(nt);
    const i64 chunk = (a.n + nt - 1) / std::max(nt, 1);
    dispatch(a, [&](auto* va) {
        return dispatch(b, [&](auto* vb) {
            lsm::parallel_for(parallel && nt > 1, nt, [&](i64 t) {
                auto& c = part[t];
                c.assign(na * nb, 0);
                const i64 lo = t * chunk, hi = std::min(a.n, lo + chunk);
                for (i64 i = lo; i < hi; ++i) {
                    if (missing(va[i], a) || missing(vb[i], b)) continue;
                    const std::int32_t r = ia(static_cast<double>(va[i]));
                    const std::int32_t s = ib(static_cast<double>(vb[i]));
                    if (r >= 0 && s >= 0) ++c[static_cast<std::size_t>(r) * nb + static_cast<std::size_t>(s)];
                }
            });
            return 0;
        });
    });
    std::vector<std::int64_t> out(na * nb, 0);
    for (const auto& c : part)
        for (std::size_t k = 0; k < c.size(); ++k) out[k] += c[k];
    return out;
}

std::vector<std::int32_t> change_count(const std::vector<MapView>& maps) {
    if (maps.empty()) return {};
    const i64 n = maps.front().n;
    for (const auto& m : maps)
        if (m.n != n) throw std::invalid_argument("the maps differ in size");
    std::vector<std::int32_t> out(static_cast<std::size_t>(n), 0);
    const bool parallel = n >= kParallelCells;
    // a pair at a time: the map's missing cells mark the cell, differences add one
    std::vector<char> gone(static_cast<std::size_t>(n), 0);
    for (std::size_t k = 0; k < maps.size(); ++k) {
        dispatch(maps[k], [&](auto* v) {
            lsm::parallel_for(parallel, n, [&](i64 i) {
                if (missing(v[i], maps[k])) gone[i] = 1;
            });
            if (k == 0) return 0;
            return dispatch(maps[k - 1], [&](auto* u) {
                lsm::parallel_for(parallel, n, [&](i64 i) {
                    if (!gone[i] && static_cast<double>(u[i]) != static_cast<double>(v[i])) ++out[i];
                });
                return 0;
            });
        });
    }
    lsm::parallel_for(parallel, n, [&](i64 i) {
        if (gone[i]) out[i] = -1;
    });
    return out;
}

}  // namespace sim
