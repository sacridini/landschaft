// SPDX-License-Identifier: GPL-3.0-or-later
//
// Zonal statistics: the cells of zones (polygons burnt as GDAL does, or the
// labels of a raster) as runs of rows, then the statistics of every zone and
// band, in parallel over the zones, or over the runs of a zone when there are
// few zones.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "io/io.hpp"
#include "lsm/landscape.hpp"

namespace lsio {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// The runs of a mask of a window (wx x wy at ox, oy) of the grid.
void mask_runs(const std::uint8_t* mask, int ox, int oy, int wx, int wy, std::vector<std::int32_t>& ry,
               std::vector<std::int32_t>& rx0, std::vector<std::int32_t>& rx1) {
    for (int y = 0; y < wy; ++y) {
        const std::uint8_t* row = mask + static_cast<std::size_t>(y) * wx;
        int x = 0;
        while (x < wx) {
            while (x < wx && !row[x]) ++x;
            if (x == wx) break;
            const int start = x;
            while (x < wx && row[x]) ++x;
            ry.push_back(oy + y);
            rx0.push_back(ox + start);
            rx1.push_back(ox + x);
        }
    }
}

// Streaming statistics (Welford's mean and variance).
struct Acc {
    i64 n = 0, missing = 0;
    double sum = 0.0, mean = 0.0, m2 = 0.0;
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    void add(double v) {
        ++n;
        sum += v;
        const double d = v - mean;
        mean += d / static_cast<double>(n);
        m2 += d * (v - mean);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    void merge(const Acc& o) {
        if (o.n == 0) {
            missing += o.missing;
            return;
        }
        if (n == 0) {
            const i64 keep = missing;
            *this = o;
            missing += keep;
            return;
        }
        const double total = static_cast<double>(n + o.n);
        const double d = o.mean - mean;
        mean += d * static_cast<double>(o.n) / total;
        m2 += o.m2 + d * d * static_cast<double>(n) * static_cast<double>(o.n) / total;
        n += o.n;
        missing += o.missing;
        sum += o.sum;
        lo = std::min(lo, o.lo);
        hi = std::max(hi, o.hi);
    }
};

template <class T>
inline bool no_data(T v, bool has_nodata, double nodata) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(v)) return true;
    }
    return has_nodata && static_cast<double>(v) == nodata;
}

// The statistics of the runs [r0, r1) of one band into acc (and the values, when
// the order statistics need them).
template <class T>
void scan(const T* band, int nx, const Runs& runs, i64 r0, i64 r1, bool has_nodata, double nodata, bool keep,
          Acc& acc, std::vector<double>& vals) {
    for (i64 r = r0; r < r1; ++r) {
        const T* row = band + static_cast<std::size_t>(runs.y[r]) * nx;
        for (int x = runs.x0[r]; x < runs.x1[r]; ++x) {
            const T v = row[x];
            if (no_data(v, has_nodata, nodata)) {
                ++acc.missing;
                continue;
            }
            const double d = static_cast<double>(v);
            acc.add(d);
            if (keep) vals.push_back(d);
        }
    }
}

// NumPy's linear interpolation between two order statistics (its rounding too).
double lerp(double a, double b, double t) {
    const double d = b - a;
    return t >= 0.5 ? b - d * (1.0 - t) : a + d * t;
}

// The q-th percentile of vals (NumPy's "linear" method); sorted: vals is in order,
// else it is partially reordered (nth_element, linear time).
double percentile(std::vector<double>& vals, double q, bool sorted) {
    const std::size_t n = vals.size();
    if (n == 0) return kNaN;
    const double pos = q / 100.0 * static_cast<double>(n - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const double t = pos - static_cast<double>(lo);
    if (sorted) return lerp(vals[lo], vals[std::min(lo + 1, n - 1)], t);
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(lo), vals.end());
    const double a = vals[lo];
    if (lo + 1 >= n) return a;
    const double b = *std::min_element(vals.begin() + static_cast<std::ptrdiff_t>(lo) + 1, vals.end());
    return lerp(a, b, t);
}

void finish(const Acc& a, std::vector<double>& vals, const std::vector<int>& stats,
            const std::vector<double>& percentiles, double* out) {
    bool counts = false;  // the frequency statistics need the values in order
    for (int s : stats) counts |= s == MAJORITY || s == MINORITY || s == UNIQUE;
    double majority = kNaN, minority = kNaN, unique = 0.0;
    if (counts && !vals.empty()) {
        std::sort(vals.begin(), vals.end());
        i64 best = 0, worst = std::numeric_limits<i64>::max();
        for (std::size_t i = 0; i < vals.size();) {
            std::size_t j = i;
            while (j < vals.size() && vals[j] == vals[i]) ++j;
            const auto c = static_cast<i64>(j - i);
            if (c > best) {
                best = c;
                majority = vals[i];
            }
            if (c < worst) {
                worst = c;
                minority = vals[i];
            }
            unique += 1.0;
            i = j;
        }
    }
    const bool empty = a.n == 0;
    for (std::size_t k = 0; k < stats.size(); ++k) {
        double v = kNaN;
        switch (stats[k]) {
            case COUNT: v = static_cast<double>(a.n); break;
            case NODATA: v = static_cast<double>(a.missing); break;
            case SUM: v = empty ? kNaN : a.sum; break;
            case MEAN: v = empty ? kNaN : a.mean; break;
            case MIN: v = empty ? kNaN : a.lo; break;
            case MAX: v = empty ? kNaN : a.hi; break;
            case RANGE: v = empty ? kNaN : a.hi - a.lo; break;
            case VAR: v = empty ? kNaN : a.m2 / static_cast<double>(a.n); break;
            case STD: v = empty ? kNaN : std::sqrt(a.m2 / static_cast<double>(a.n)); break;
            case MEDIAN: v = percentile(vals, 50.0, counts); break;
            case MAJORITY: v = majority; break;
            case MINORITY: v = minority; break;
            case UNIQUE: v = unique; break;
            case PERCENTILE: v = percentile(vals, percentiles[k], counts); break;
            default: throw std::invalid_argument("unknown statistic");
        }
        out[k] = v;
    }
}

}  // namespace

Runs polygon_runs(const double* xy, const i64* ring_offsets, const i64* geom_rings, i64 ngeoms, const double* gt,
                  int nx, int ny, bool all_touched) {
    if (gt[2] != 0.0 || gt[4] != 0.0 || gt[1] == 0.0 || gt[5] == 0.0) {
        throw std::invalid_argument("zonal statistics: rotated or degenerate geotransform");
    }
    // GDALInvGeoTransform without rotation, then GDALApplyGeoTransform (as lsm::rasterize_all_touched)
    const double inv[6] = {-gt[0] / gt[1], 1.0 / gt[1], 0.0, -gt[3] / gt[5], 0.0, 1.0 / gt[5]};
    std::vector<std::vector<std::int32_t>> ys(static_cast<std::size_t>(ngeoms)), x0s(ys.size()), x1s(ys.size());
#pragma omp parallel
    {
        std::vector<int> part_size;
        std::vector<double> px, py;
        std::vector<std::uint8_t> mask;
#pragma omp for schedule(dynamic, 1)
        for (i64 g = 0; g < ngeoms; ++g) {
            part_size.clear();
            px.clear();
            py.clear();
            double xmin = std::numeric_limits<double>::infinity(), xmax = -xmin, ymin = xmin, ymax = -xmin;
            for (i64 r = geom_rings[g]; r < geom_rings[g + 1]; ++r) {
                part_size.push_back(static_cast<int>(ring_offsets[r + 1] - ring_offsets[r]));
                for (i64 p = ring_offsets[r]; p < ring_offsets[r + 1]; ++p) {
                    const double X = xy[2 * p], Y = xy[2 * p + 1];
                    const double cx = inv[0] + X * inv[1] + Y * inv[2];
                    const double cy = inv[3] + X * inv[4] + Y * inv[5];
                    px.push_back(cx);
                    py.push_back(cy);
                    xmin = std::min(xmin, cx);
                    xmax = std::max(xmax, cx);
                    ymin = std::min(ymin, cy);
                    ymax = std::max(ymax, cy);
                }
            }
            if (px.empty() || !(xmax >= 0.0) || !(ymax >= 0.0) || xmin > nx || ymin > ny) continue;
            // the cells any rule can burn lie within the bounding box, one cell around
            const int ox = std::max(0, static_cast<int>(std::floor(xmin)) - 1);
            const int oy = std::max(0, static_cast<int>(std::floor(ymin)) - 1);
            const int ex = std::min(nx, static_cast<int>(std::ceil(xmax)) + 1);
            const int ey = std::min(ny, static_cast<int>(std::ceil(ymax)) + 1);
            if (ex <= ox || ey <= oy) continue;
            const int wx = ex - ox, wy = ey - oy;
            mask.assign(static_cast<std::size_t>(wx) * wy, 0);
            lsm::burn_polygon(part_size, px, py, nx, ny, all_touched, ox, oy, wx, wy, mask.data());
            const auto gi = static_cast<std::size_t>(g);
            mask_runs(mask.data(), ox, oy, wx, wy, ys[gi], x0s[gi], x1s[gi]);
        }
    }
    Runs runs;
    runs.offsets.resize(static_cast<std::size_t>(ngeoms) + 1);
    runs.offsets[0] = 0;
    for (i64 g = 0; g < ngeoms; ++g) {
        runs.offsets[g + 1] = runs.offsets[g] + static_cast<i64>(ys[static_cast<std::size_t>(g)].size());
    }
    runs.y.reserve(static_cast<std::size_t>(runs.offsets.back()));
    runs.x0.reserve(runs.y.capacity());
    runs.x1.reserve(runs.y.capacity());
    for (std::size_t g = 0; g < ys.size(); ++g) {
        runs.y.insert(runs.y.end(), ys[g].begin(), ys[g].end());
        runs.x0.insert(runs.x0.end(), x0s[g].begin(), x0s[g].end());
        runs.x1.insert(runs.x1.end(), x1s[g].begin(), x1s[g].end());
    }
    return runs;
}

template <class L>
Runs label_runs(const L* labels, int ny, int nx, bool has_nodata, double nodata, std::vector<double>* zone_labels) {
    // runs of equal labels, row by row (rows in parallel)
    std::vector<std::vector<std::int32_t>> rx0(static_cast<std::size_t>(ny)), rx1(rx0.size());
    std::vector<std::vector<double>> rlab(rx0.size());
#pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; ++y) {
        const L* row = labels + static_cast<std::size_t>(y) * nx;
        const auto yi = static_cast<std::size_t>(y);
        int x = 0;
        while (x < nx) {
            const L v = row[x];
            int e = x + 1;
            while (e < nx && row[e] == v) ++e;
            if (!no_data(v, has_nodata, nodata)) {
                rx0[yi].push_back(x);
                rx1[yi].push_back(e);
                rlab[yi].push_back(static_cast<double>(v));
            }
            x = e;
        }
    }
    // the labels, sorted, and every run's zone
    std::vector<double> labs;
    for (const auto& r : rlab) labs.insert(labs.end(), r.begin(), r.end());
    std::sort(labs.begin(), labs.end());
    labs.erase(std::unique(labs.begin(), labs.end()), labs.end());
    std::unordered_map<double, i64> zone_of;
    zone_of.reserve(labs.size() * 2);
    for (std::size_t i = 0; i < labs.size(); ++i) zone_of.emplace(labs[i], static_cast<i64>(i));
    // counting sort of the runs by zone (rows in order inside a zone)
    Runs runs;
    runs.offsets.assign(labs.size() + 1, 0);
    for (const auto& r : rlab) {
        for (double v : r) ++runs.offsets[static_cast<std::size_t>(zone_of[v]) + 1];
    }
    for (std::size_t z = 0; z < labs.size(); ++z) runs.offsets[z + 1] += runs.offsets[z];
    const auto total = static_cast<std::size_t>(runs.offsets.back());
    runs.y.resize(total);
    runs.x0.resize(total);
    runs.x1.resize(total);
    std::vector<i64> next(runs.offsets.begin(), runs.offsets.end() - 1);
    for (int y = 0; y < ny; ++y) {
        const auto yi = static_cast<std::size_t>(y);
        for (std::size_t k = 0; k < rlab[yi].size(); ++k) {
            const auto at = static_cast<std::size_t>(next[static_cast<std::size_t>(zone_of[rlab[yi][k]])]++);
            runs.y[at] = y;
            runs.x0[at] = rx0[yi][k];
            runs.x1[at] = rx1[yi][k];
        }
    }
    *zone_labels = std::move(labs);
    return runs;
}

template <class T>
std::vector<double> zonal_stats(const T* values, int bands, int ny, int nx, bool has_nodata, double nodata,
                                const Runs& runs, const std::vector<int>& stats,
                                const std::vector<double>& percentiles) {
    const i64 nz = runs.zones();
    const auto ns = static_cast<i64>(stats.size());
    bool keep = false;
    for (int s : stats) keep |= s >= MEDIAN;
    std::vector<double> out(static_cast<std::size_t>(nz * bands * ns), kNaN);
    const std::size_t plane = static_cast<std::size_t>(ny) * nx;
    // Every zone in blocks of runs of about kBlockCells cells: the blocks depend on
    // the zones only, so the result does not depend on the number of threads.
    constexpr i64 kBlockCells = i64{1} << 16;
    std::vector<i64> block_start{0}, block_zone, zone_first(static_cast<std::size_t>(nz) + 1, 0);
    for (i64 z = 0; z < nz; ++z) {
        zone_first[static_cast<std::size_t>(z)] = static_cast<i64>(block_zone.size());
        i64 cells = 0;
        const i64 r1 = runs.offsets[z + 1];
        for (i64 r = runs.offsets[z]; r < r1; ++r) {
            cells += runs.x1[r] - runs.x0[r];
            if (cells >= kBlockCells || r + 1 == r1) {
                block_start.push_back(r + 1);
                block_zone.push_back(z);
                cells = 0;
            }
        }
        if (runs.offsets[z] == r1) {  // a zone without cells: one empty block
            block_start.push_back(r1);
            block_zone.push_back(z);
        }
    }
    zone_first[static_cast<std::size_t>(nz)] = static_cast<i64>(block_zone.size());
    // block_start[k] .. block_start[k + 1] are the runs of block k; the first block of a zone starts at its offset
    const auto nb = static_cast<i64>(block_zone.size());
    for (i64 z = 0; z < nz; ++z) {
        if (zone_first[static_cast<std::size_t>(z)] < nb) block_start[static_cast<std::size_t>(zone_first[z])] = runs.offsets[z];
    }
    std::vector<Acc> accs(static_cast<std::size_t>(nb));
    std::vector<std::vector<double>> parts(keep ? static_cast<std::size_t>(nb) : 0);
    for (int b = 0; b < bands; ++b) {
        const T* band = values + plane * b;
#pragma omp parallel for schedule(dynamic, 1)
        for (i64 k = 0; k < nb; ++k) {
            const auto ki = static_cast<std::size_t>(k);
            accs[ki] = Acc{};
            static thread_local std::vector<double> none;
            std::vector<double>& vals = keep ? parts[ki] : none;
            vals.clear();
            scan(band, nx, runs, block_start[ki], block_start[ki + 1], has_nodata, nodata, keep, accs[ki], vals);
        }
#pragma omp parallel
        {
            std::vector<double> vals;
#pragma omp for schedule(dynamic, 1)
            for (i64 z = 0; z < nz; ++z) {
                const i64 k0 = zone_first[static_cast<std::size_t>(z)], k1 = zone_first[static_cast<std::size_t>(z) + 1];
                Acc acc;
                vals.clear();
                for (i64 k = k0; k < k1; ++k) {
                    acc.merge(accs[static_cast<std::size_t>(k)]);
                    if (keep) {
                        auto& p = parts[static_cast<std::size_t>(k)];
                        vals.insert(vals.end(), p.begin(), p.end());
                        std::vector<double>().swap(p);
                    }
                }
                finish(acc, vals, stats, percentiles, &out[static_cast<std::size_t>((z * bands + b) * ns)]);
            }
        }
    }
    return out;
}

template <class T>
std::vector<double> extract(const T* values, int bands, int ny, int nx, bool has_nodata, double nodata,
                            const double* gt, const double* x, const double* y, i64 n, bool bilinear, int row0,
                            int ny_total) {
    if (gt[2] != 0.0 || gt[4] != 0.0 || gt[1] == 0.0 || gt[5] == 0.0) {
        throw std::invalid_argument("extract: rotated or degenerate geotransform");
    }
    std::vector<double> out(static_cast<std::size_t>(n * bands), kNaN);
    const std::size_t plane = static_cast<std::size_t>(ny) * nx;
#pragma omp parallel for schedule(static)
    for (i64 i = 0; i < n; ++i) {
        const double fx = (x[i] - gt[0]) / gt[1];  // column, from the left edge
        const double fy = (y[i] - gt[3]) / gt[5];  // row, from the top edge
        double* o = &out[static_cast<std::size_t>(i * bands)];
        if (!(fx >= 0.0 && fx <= nx && fy >= 0.0 && fy <= ny_total)) continue;  // outside (or NaN)
        if (!bilinear) {
            const int c = std::min(static_cast<int>(fx), nx - 1);
            const int r = std::min(static_cast<int>(fy), ny_total - 1) - row0;
            if (r < 0 || r >= ny) continue;  // not in this block
            for (int b = 0; b < bands; ++b) {
                const T v = values[plane * b + static_cast<std::size_t>(r) * nx + c];
                if (!no_data(v, has_nodata, nodata)) o[b] = static_cast<double>(v);
            }
            continue;
        }
        // the four cell centres around the point
        const double cx = fx - 0.5, cy = fy - 0.5;
        const int c0 = static_cast<int>(std::floor(cx)), r0 = static_cast<int>(std::floor(cy));
        const double wx1 = cx - c0, wy1 = cy - r0;
        const int cs[2] = {c0, c0 + 1}, rs[2] = {r0, r0 + 1};
        const double wxs[2] = {1.0 - wx1, wx1}, wys[2] = {1.0 - wy1, wy1};
        for (int b = 0; b < bands; ++b) {
            double sum = 0.0, wsum = 0.0;
            for (int j = 0; j < 2; ++j) {
                if (rs[j] < 0 || rs[j] >= ny_total || wys[j] == 0.0) continue;
                const int r = rs[j] - row0;
                if (r < 0 || r >= ny) continue;  // not in this block
                for (int k = 0; k < 2; ++k) {
                    if (cs[k] < 0 || cs[k] >= nx || wxs[k] == 0.0) continue;
                    const T v = values[plane * b + static_cast<std::size_t>(r) * nx + cs[k]];
                    if (no_data(v, has_nodata, nodata)) continue;
                    const double w = wys[j] * wxs[k];
                    sum += w * static_cast<double>(v);
                    wsum += w;
                }
            }
            if (wsum > 0.0) o[b] = sum / wsum;
        }
    }
    return out;
}

#define LSIO_INSTANTIATE(T)                                                                                          \
    template Runs label_runs<T>(const T*, int, int, bool, double, std::vector<double>*);                            \
    template std::vector<double> zonal_stats<T>(const T*, int, int, int, bool, double, const Runs&,                \
                                                const std::vector<int>&, const std::vector<double>&);              \
    template std::vector<double> extract<T>(const T*, int, int, int, bool, double, const double*, const double*,   \
                                            const double*, i64, bool, int, int);

LSIO_INSTANTIATE(std::uint8_t)
LSIO_INSTANTIATE(std::int8_t)
LSIO_INSTANTIATE(std::uint16_t)
LSIO_INSTANTIATE(std::int16_t)
LSIO_INSTANTIATE(std::uint32_t)
LSIO_INSTANTIATE(std::int32_t)
LSIO_INSTANTIATE(std::int64_t)
LSIO_INSTANTIATE(float)
LSIO_INSTANTIATE(double)

#undef LSIO_INSTANTIATE

}  // namespace lsio
