// SPDX-License-Identifier: GPL-3.0-or-later

#include "rest/plan.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rest {

namespace {

// Fill `room` of the row from the units `items` (a > 0), best c / a first: weighted quickselect, the
// units better than the split taken in full, the split's own density in the units' order. Returns the
// room left (0 unless the units run out).
double fill(std::vector<i64>& items, const double* c, const double* a, const double* lb, const double* ub,
            double room, double* x) {
    auto density = [&](i64 j) { return c[j] / a[j]; };
    auto weight = [&](i64 j) { return a[j] * (ub[j] - lb[j]); };
    std::size_t lo = 0, hi = items.size();
    while (lo < hi && room > 0) {
        // the median of three densities as the pivot
        const std::size_t mid = lo + (hi - lo) / 2;
        double d[3] = {density(items[lo]), density(items[mid]), density(items[hi - 1])};
        std::sort(d, d + 3);
        const double pivot = d[1];
        // [lo, p1): above the pivot, [p1, p2): at it, [p2, hi): below
        auto p1 = std::partition(items.begin() + lo, items.begin() + hi, [&](i64 j) { return density(j) > pivot; });
        auto p2 = std::partition(p1, items.begin() + hi, [&](i64 j) { return density(j) == pivot; });
        const std::size_t i1 = static_cast<std::size_t>(p1 - items.begin());
        const std::size_t i2 = static_cast<std::size_t>(p2 - items.begin());
        double above = 0.0;
        for (std::size_t k = lo; k < i1; ++k) above += weight(items[k]);
        if (above >= room) {
            hi = i1;  // the split is among the better ones
            continue;
        }
        for (std::size_t k = lo; k < i1; ++k) x[items[k]] = ub[items[k]];
        room -= above;
        std::sort(items.begin() + i1, items.begin() + i2);  // ties in the units' order
        for (std::size_t k = i1; k < i2 && room > 0; ++k) {
            const i64 j = items[k];
            const double w = weight(j);
            if (w <= room) {
                x[j] = ub[j];
                room -= w;
            } else {
                x[j] = lb[j] + room / a[j];
                room = 0.0;
            }
        }
        lo = i2;
    }
    return room;
}

}  // namespace

bool knapsack(const double* c, const double* a, const double* lb, const double* ub, i64 n, double cap,
              const std::string& sense, double* x) {
    if (sense != "<=" && sense != ">=" && sense != "=") throw std::invalid_argument("sense: '<=', '>=' or '='");
    double used = 0.0;
    std::vector<i64> good, bad;
    for (i64 j = 0; j < n; ++j) {
        if (a[j] < 0) throw std::invalid_argument("the row's coefficients must be at least 0");
        x[j] = lb[j];
        used += a[j] * lb[j];
        if (!(ub[j] > lb[j])) continue;
        if (a[j] == 0) {
            if (c[j] > 0) x[j] = ub[j];  // free and worth it
        } else if (c[j] > 0) {
            good.push_back(j);
        } else {
            bad.push_back(j);
        }
    }
    double room = cap - used;
    if (sense == ">=") {
        // every unit worth it, in full: then the row is met or the least bad ones fill it
        for (i64 j : good) {
            room -= a[j] * (ub[j] - lb[j]);
            x[j] = ub[j];
        }
        if (room <= 0) return true;
        return fill(bad, c, a, lb, ub, room, x) <= 1e-9 * std::max(1.0, std::fabs(cap));
    }
    if (room < -1e-9 * std::max(1.0, std::fabs(cap))) return false;
    room = fill(good, c, a, lb, ub, std::max(room, 0.0), x);
    if (sense == "<=" || room <= 0) return true;
    return fill(bad, c, a, lb, ub, room, x) <= 1e-9 * std::max(1.0, std::fabs(cap));
}

template <class P, class I>
void csr_matvec(const P* indptr, const I* indices, const double* data, i64 nrow, const double* x, double* out,
                int threads) {
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
    for (i64 r = 0; r < nrow; ++r) {
        double s = 0.0;
        for (P q = indptr[r]; q < indptr[r + 1]; ++q) s += data[q] * x[indices[q]];
        out[r] = s;
    }
}

template <class P, class I>
void csr_rmatvec(const P* indptr, const I* indices, const double* data, i64 nrow, i64 ncol, const double* y,
                 double* out, int threads) {
    // threads that each sum a band of rows into their own copy of the columns, added in order
    const i64 nnz = static_cast<i64>(indptr[nrow]);
    int t_use = std::max(1, std::min<int>(threads, static_cast<int>(std::max<i64>(1, nnz / std::max<i64>(ncol, 1)))));
    std::vector<std::vector<double>> part(static_cast<std::size_t>(t_use));
    // the bands: equal shares of the products
    std::vector<i64> cut(static_cast<std::size_t>(t_use) + 1, nrow);
    cut[0] = 0;
    for (int t = 1; t < t_use; ++t) {
        const P want = static_cast<P>(nnz / t_use * t);
        cut[static_cast<std::size_t>(t)] = static_cast<i64>(std::lower_bound(indptr, indptr + nrow + 1, want) - indptr);
    }
#pragma omp parallel for num_threads(t_use) schedule(static, 1)
    for (int t = 0; t < t_use; ++t) {
        std::vector<double>& acc = part[static_cast<std::size_t>(t)];
        acc.assign(static_cast<std::size_t>(ncol), 0.0);
        for (i64 r = cut[static_cast<std::size_t>(t)]; r < cut[static_cast<std::size_t>(t) + 1]; ++r) {
            const double v = y[r];
            if (v == 0.0) continue;
            for (P q = indptr[r]; q < indptr[r + 1]; ++q) acc[static_cast<std::size_t>(indices[q])] += data[q] * v;
        }
    }
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 j = 0; j < ncol; ++j) {
        double s = 0.0;
        for (int t = 0; t < t_use; ++t) s += part[static_cast<std::size_t>(t)][static_cast<std::size_t>(j)];
        out[j] = s;
    }
}

template <class P, class I>
void csr_select_columns(const P* indptr, const I* indices, const double* data, i64 nrow, const i64* pos,
                        std::vector<i64>& out_indptr, std::vector<std::int32_t>& out_indices,
                        std::vector<double>& out_data, int threads) {
    out_indptr.assign(static_cast<std::size_t>(nrow) + 1, 0);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
    for (i64 r = 0; r < nrow; ++r) {
        i64 k = 0;
        for (P q = indptr[r]; q < indptr[r + 1]; ++q) k += pos[indices[q]] >= 0;
        out_indptr[static_cast<std::size_t>(r) + 1] = k;
    }
    for (i64 r = 0; r < nrow; ++r) out_indptr[static_cast<std::size_t>(r) + 1] += out_indptr[static_cast<std::size_t>(r)];
    out_indices.resize(static_cast<std::size_t>(out_indptr[static_cast<std::size_t>(nrow)]));
    out_data.resize(out_indices.size());
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
    for (i64 r = 0; r < nrow; ++r) {
        i64 k = out_indptr[static_cast<std::size_t>(r)];
        for (P q = indptr[r]; q < indptr[r + 1]; ++q) {
            const i64 j = pos[indices[q]];
            if (j < 0) continue;
            out_indices[static_cast<std::size_t>(k)] = static_cast<std::int32_t>(j);
            out_data[static_cast<std::size_t>(k)] = data[q];
            ++k;
        }
    }
}

#define REST_SELECT(P, I)                                                                                      template void csr_select_columns<P, I>(const P*, const I*, const double*, i64, const i64*,                                                       std::vector<i64>&, std::vector<std::int32_t>&, std::vector<double>&, int);
REST_SELECT(std::int32_t, std::int32_t)
REST_SELECT(std::int64_t, std::int32_t)
REST_SELECT(std::int64_t, std::int64_t)
#undef REST_SELECT

template void csr_matvec<std::int32_t, std::int32_t>(const std::int32_t*, const std::int32_t*, const double*, i64,
                                                     const double*, double*, int);
template void csr_matvec<std::int64_t, std::int32_t>(const std::int64_t*, const std::int32_t*, const double*, i64,
                                                     const double*, double*, int);
template void csr_matvec<std::int64_t, std::int64_t>(const std::int64_t*, const std::int64_t*, const double*, i64,
                                                     const double*, double*, int);
template void csr_rmatvec<std::int32_t, std::int32_t>(const std::int32_t*, const std::int32_t*, const double*, i64,
                                                      i64, const double*, double*, int);
template void csr_rmatvec<std::int64_t, std::int32_t>(const std::int64_t*, const std::int32_t*, const double*, i64,
                                                      i64, const double*, double*, int);
template void csr_rmatvec<std::int64_t, std::int64_t>(const std::int64_t*, const std::int64_t*, const double*, i64,
                                                      i64, const double*, double*, int);

}  // namespace rest
