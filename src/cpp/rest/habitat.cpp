// SPDX-License-Identifier: GPL-3.0-or-later

#include "rest/habitat.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"

namespace rest {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// R's Reduce('+', layers) at pixel i: the layers added one after the other, in their order.
inline double reduce_sum(const double* layers, int count, i64 n, i64 i) {
    double s = layers[i];
    for (int c = 1; c < count; ++c) s += layers[static_cast<i64>(c) * n + i];
    return s;
}

// R's rowSums() of the k layers at pixel i: an LDOUBLE accumulator, rounded once.
inline double row_sum(const double* layers, int k, i64 n, i64 i) {
    lsm::Extended s = 0.0;
    for (int c = 0; c < k; ++c) s += layers[static_cast<i64>(c) * n + i];
    return static_cast<double>(s);
}

}  // namespace

Vegetation original_vegetation(const double* natural, int k, const double* anthropic, int m, i64 n,
                               const double* past, const double* ecoregion, const std::uint8_t* domain,
                               const std::uint8_t* fill, double threshold, bool restore_current, int threads) {
    if (k < 1) throw std::invalid_argument("at least one natural class is needed");
    if (m < 1) throw std::invalid_argument("at least one anthropic class is needed");
    const double* p = past ? past : natural;  // plangea: without past maps, the current natural cover
    const std::size_t kn = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
    std::vector<double> share(kn);  // p_nat_vals
    Vegetation out;
    out.source.assign(static_cast<std::size_t>(n), 0);
    std::vector<double> anth(static_cast<std::size_t>(n));
    const double flat = 1.0 / static_cast<double>(k);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        const double a = reduce_sum(anthropic, m, n, i);
        anth[i] = a;
        const double total = reduce_sum(p, k, n, i);
        // no past natural cover, or nearly all the pixel converted: the ecoregion's shares
        const bool corr = total == 0.0 || a > threshold;
        for (int c = 0; c < k; ++c) {
            const i64 at = static_cast<i64>(c) * n + i;
            share[at] = corr ? (ecoregion ? ecoregion[at] : flat) : p[at] / total;
        }
        if (corr) out.source[i] = 1;
    }
    // the failsafe: the classes' mean shares over the domain (R's mean(na.rm = TRUE)), as fractions
    // of their sum
    out.fill.assign(static_cast<std::size_t>(k), 0.0);
    std::vector<double> means;
    std::vector<int> filled;
    for (int c = 0; c < k; ++c) {
        if (!fill[c]) continue;
        std::vector<double> v;
        v.reserve(static_cast<std::size_t>(n));
        for (i64 i = 0; i < n; ++i) {
            const double x = share[static_cast<i64>(c) * n + i];
            if ((!domain || domain[i]) && !std::isnan(x)) v.push_back(x);
        }
        means.push_back(lsm::r_mean(v.data(), static_cast<i64>(v.size())));
        filled.push_back(c);
    }
    if (!means.empty()) {
        const double total = lsm::r_sum(means.data(), static_cast<i64>(means.size()));
        for (std::size_t q = 0; q < means.size(); ++q) out.fill[filled[q]] = means[q] / total;
    }
    out.original.resize(kn);
    out.restore.resize(kn);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        const double s = reduce_sum(share.data(), k, n, i);
        if (s == 0.0 || std::isnan(s)) {  // still nothing: the failsafe shares (the other classes none)
            for (int c = 0; c < k; ++c) share[static_cast<i64>(c) * n + i] = out.fill[c];
            out.source[i] = 2;
        }
        for (int c = 0; c < k; ++c) {
            const i64 at = static_cast<i64>(c) * n + i;
            out.original[at] = natural[at] + share[at] * anth[i];
        }
        const double* basis = out.original.data();
        if (restore_current && row_sum(natural, k, n, i) != 0.0) basis = natural;
        const double t = row_sum(basis, k, n, i);
        for (int c = 0; c < k; ++c) {
            const i64 at = static_cast<i64>(c) * n + i;
            out.restore[at] = t != 0.0 ? basis[at] / t : kNaN;
        }
    }
    return out;
}

void habitat_areas(const double* current, const double* original, const double* area, i64 n, int k,
                   const i64* range_ptr, const i64* range_idx, const std::uint8_t* suitable, i64 ns,
                   double* now, double* potential, int threads) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (i64 s = 0; s < ns; ++s) {
        const std::uint8_t* suit = suitable + s * k;
        bool any = false;
        for (int c = 0; c < k; ++c) any = any || suit[c];
        if (!any) {  // no suitable natural class: no habitat
            now[s] = 0.0;
            potential[s] = 0.0;
            continue;
        }
        const i64 b = range_ptr[s], e = range_ptr[s + 1];
        // R: sum(range * Reduce('+', the suitable classes) * area) over every pixel
        auto habitat = [&](const double* layers) {
            return lsm::sum_of(b, e, [&](i64 q) {
                const i64 i = range_idx[q];
                double v = 0.0;
                bool first = true;
                for (int c = 0; c < k; ++c) {
                    if (!suit[c]) continue;
                    const double x = layers[static_cast<i64>(c) * n + i];
                    v = first ? x : v + x;
                    first = false;
                }
                return v * area[i];
            });
        };
        now[s] = habitat(current);
        potential[s] = habitat(original);
    }
}

void habitat_gain(const double* restore, const double* area, i64 n, int k, const i64* range_ptr,
                  const i64* range_idx, const std::uint8_t* suitable, i64 ns, const std::uint8_t* domain,
                  std::vector<i64>& indptr, std::vector<i64>& index, std::vector<double>& gain, int threads) {
    std::vector<std::vector<i64>> idx(static_cast<std::size_t>(ns));
    std::vector<std::vector<double>> val(static_cast<std::size_t>(ns));
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (i64 s = 0; s < ns; ++s) {
        const std::uint8_t* suit = suitable + s * k;
        for (i64 q = range_ptr[s]; q < range_ptr[s + 1]; ++q) {
            const i64 i = range_idx[q];
            if (domain && !domain[i]) continue;
            // plangea's prop_restore %*% the species' classes (0 / 1): the suitable shares in order
            double v = 0.0;
            for (int c = 0; c < k; ++c)
                if (suit[c]) v += restore[static_cast<i64>(c) * n + i];
            v = area[i] * v;
            if (v != 0.0) {
                idx[s].push_back(i);
                val[s].push_back(v);
            }
        }
    }
    indptr.assign(static_cast<std::size_t>(ns + 1), 0);
    for (i64 s = 0; s < ns; ++s) indptr[s + 1] = indptr[s] + static_cast<i64>(idx[s].size());
    index.resize(static_cast<std::size_t>(indptr[ns]));
    gain.resize(static_cast<std::size_t>(indptr[ns]));
    for (i64 s = 0; s < ns; ++s) {
        std::copy(idx[s].begin(), idx[s].end(), index.begin() + indptr[s]);
        std::copy(val[s].begin(), val[s].end(), gain.begin() + indptr[s]);
    }
}

}  // namespace rest
