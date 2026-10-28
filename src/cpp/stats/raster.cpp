// SPDX-License-Identifier: GPL-3.0-or-later
//
// Raster tools: cell statistics of a stack, local correlation, viewshed (see raster.hpp).

#include "stats/raster.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stats {

namespace {

using i64 = std::int64_t;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// numpy's linear percentile of sorted values
double percentile(const std::vector<double>& v, double q) {
    const std::size_t n = v.size();
    const double pos = q / 100.0 * static_cast<double>(n - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, n - 1);
    const double t = pos - static_cast<double>(lo);
    const double a = v[lo], b = v[hi], d = b - a;
    return t >= 0.5 ? b - d * (1.0 - t) : a + d * t;
}

// A statistic of the values of a cell with one (sorted when the statistic needs it).
double statistic(int stat, std::vector<double>& v, i64 missing, double q) {
    const std::size_t n = v.size();
    if (n == 0) {
        if (stat == kCount || stat == kUnique) return 0.0;
        if (stat == kNodata) return static_cast<double>(missing);
        return kNaN;
    }
    const double cn = static_cast<double>(n);
    auto mean = [&] {
        double s = 0.0;
        for (double x : v) s += x;
        return s / cn;
    };
    auto var = [&] {
        const double m = mean();
        double s = 0.0;
        for (double x : v) s += (x - m) * (x - m);
        return s / cn;
    };
    switch (stat) {
        case kCount: return cn;
        case kNodata: return static_cast<double>(missing);
        case kSum: {
            double s = 0.0;
            for (double x : v) s += x;
            return s;
        }
        case kMean: return mean();
        case kMin: return *std::min_element(v.begin(), v.end());
        case kMax: return *std::max_element(v.begin(), v.end());
        case kRange: return *std::max_element(v.begin(), v.end()) - *std::min_element(v.begin(), v.end());
        case kStd: return std::sqrt(var());
        case kVar: return var();
        default: break;
    }
    std::sort(v.begin(), v.end());
    if (stat == kMedian) return percentile(v, 50.0);
    if (stat == kPercentile) return percentile(v, q);
    // the values and how many times each comes, increasing (a tie: the smallest)
    double majority = kNaN, minority = kNaN, unique = 0.0;
    i64 best = 0, worst = std::numeric_limits<i64>::max();
    for (std::size_t i = 0; i < n;) {
        std::size_t j = i;
        while (j < n && v[j] == v[i]) ++j;
        const auto c = static_cast<i64>(j - i);
        if (c > best) {
            best = c;
            majority = v[i];
        }
        if (c < worst) {
            worst = c;
            minority = v[i];
        }
        unique += 1.0;
        i = j;
    }
    if (stat == kMajority) return majority;
    if (stat == kMinority) return minority;
    return unique;
}

}  // namespace

void cell_stats(const double* layers, i64 k, i64 n, int stat, double q, double* out, int threads) {
    if (stat < kCount || stat > kArgmin) throw std::invalid_argument("unknown statistic");
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> v;
        v.reserve(static_cast<std::size_t>(k));
#pragma omp for schedule(static)
        for (i64 c = 0; c < n; ++c) {
            if (stat == kArgmax || stat == kArgmin) {
                i64 at = -1;
                double b = 0.0;
                for (i64 l = 0; l < k; ++l) {
                    const double x = layers[l * n + c];
                    if (std::isnan(x)) continue;
                    if (at < 0 || (stat == kArgmax ? x > b : x < b)) {
                        at = l;
                        b = x;
                    }
                }
                out[c] = at < 0 ? kNaN : static_cast<double>(at + 1);
                continue;
            }
            v.clear();
            for (i64 l = 0; l < k; ++l) {
                const double x = layers[l * n + c];
                if (!std::isnan(x)) v.push_back(x);
            }
            out[c] = statistic(stat, v, k - static_cast<i64>(v.size()), q);
        }
    }
}

void local_correlation(const double* x, const double* y, int ny, int nx, int radius, bool circle, int min_cells,
                       double* out, int threads) {
    std::vector<std::pair<int, int>> offsets;
    for (int dr = -radius; dr <= radius; ++dr)
        for (int dc = -radius; dc <= radius; ++dc)
            if (!circle || dr * dr + dc * dc <= radius * radius) offsets.emplace_back(dr, dc);
    const int least = std::max(min_cells, 2);
    const i64 n = static_cast<i64>(ny) * nx;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int r = 0; r < ny; ++r) {
        for (int c = 0; c < nx; ++c) {
            const i64 at = static_cast<i64>(r) * nx + c;
            const double x0 = x[at], y0 = y[at];
            double res[4] = {kNaN, kNaN, kNaN, kNaN};
            if (!std::isnan(x0) && !std::isnan(y0)) {
                // the sums about the centre's values (fewer digits lost)
                int cnt = 0;
                double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
                for (const auto& [dr, dc] : offsets) {
                    const int rr = r + dr, cc = c + dc;
                    if (rr < 0 || cc < 0 || rr >= ny || cc >= nx) continue;
                    const i64 i = static_cast<i64>(rr) * nx + cc;
                    const double a0 = x[i], b0 = y[i];
                    if (std::isnan(a0) || std::isnan(b0)) continue;
                    const double a = a0 - x0, b = b0 - y0;
                    ++cnt;
                    sx += a;
                    sy += b;
                    sxx += a * a;
                    syy += b * b;
                    sxy += a * b;
                }
                if (cnt >= least) {
                    const double k = cnt;
                    const double vxx = sxx - sx * sx / k;
                    const double vyy = syy - sy * sy / k;
                    const double vxy = sxy - sx * sy / k;
                    const double rho = vxx > 0.0 && vyy > 0.0 ? vxy / std::sqrt(vxx * vyy) : kNaN;
                    const double slope = vxx > 0.0 ? vxy / vxx : kNaN;
                    res[0] = rho;
                    res[1] = slope;
                    res[2] = (y0 + sy / k) - slope * (x0 + sx / k);
                    res[3] = rho * rho;
                }
            }
            for (int p = 0; p < 4; ++p) out[p * n + at] = res[p];
        }
    }
}

void viewshed(const double* dem, int ny, int nx, double rw, double rh, const i64* rows, const i64* cols, i64 n_obs,
              double eye, double target, double reach, bool curvature, double* out, int threads) {
    const i64 n = static_cast<i64>(ny) * nx;
    auto drop = [&](double d) { return curvature ? 0.85714 * d * d / (2.0 * 6371008.8) : 0.0; };
    std::vector<std::int32_t> count(static_cast<std::size_t>(n), 0);
#pragma omp parallel num_threads(threads)
    {
        std::vector<std::int32_t> mine(static_cast<std::size_t>(n), 0);
        std::vector<char> seen, visited;
#pragma omp for schedule(dynamic, 1)
        for (i64 o = 0; o < n_obs; ++o) {
            const i64 orow = rows[o], ocol = cols[o];
            if (orow < 0 || ocol < 0 || orow >= ny || ocol >= nx || std::isnan(dem[orow * nx + ocol])) continue;
            const double z0 = dem[orow * nx + ocol] + eye;
            const auto kr = static_cast<i64>(std::min(std::ceil(reach / rh), static_cast<double>(ny)));
            const auto kc = static_cast<i64>(std::min(std::ceil(reach / rw), static_cast<double>(nx)));
            const i64 r0 = std::max<i64>(orow - kr, 0), r1 = std::min<i64>(orow + kr, ny - 1);
            const i64 c0 = std::max<i64>(ocol - kc, 0), c1 = std::min<i64>(ocol + kc, nx - 1);
            const i64 ww = c1 - c0 + 1;
            // the observer's window
            seen.assign(static_cast<std::size_t>((r1 - r0 + 1) * ww), 0);
            visited.assign(seen.size(), 0);
            auto local = [&](i64 rr, i64 cc) { return static_cast<std::size_t>((rr - r0) * ww + (cc - c0)); };
            seen[local(orow, ocol)] = 1;
            visited[local(orow, ocol)] = 1;
            auto dist = [&](i64 rr, i64 cc) {
                const double a = static_cast<double>(rr - orow) * rh, b = static_cast<double>(cc - ocol) * rw;
                return std::sqrt(a * a + b * b);
            };
            auto ray = [&](i64 er, i64 ec) {
                const i64 dr = er - orow, dc = ec - ocol;
                const i64 steps = std::max(std::llabs(dr), std::llabs(dc));
                double steepest = -INFINITY;
                for (i64 k = 1; k <= steps; ++k) {
                    const i64 rr = orow + static_cast<i64>(std::round(static_cast<double>(dr) * static_cast<double>(k) /
                                                                      static_cast<double>(steps)));
                    const i64 cc = ocol + static_cast<i64>(std::round(static_cast<double>(dc) * static_cast<double>(k) /
                                                                      static_cast<double>(steps)));
                    double z = dem[rr * nx + cc];
                    if (std::isnan(z)) continue;
                    const double d = dist(rr, cc);
                    if (d > reach) break;
                    visited[local(rr, cc)] = 1;
                    z -= drop(d);
                    if ((z + target - z0) / d >= steepest) seen[local(rr, cc)] = 1;
                    steepest = std::max(steepest, (z - z0) / d);
                }
            };
            // the rays to every cell of the edge of the observer's square
            for (i64 cc = c0; cc <= c1; ++cc) {
                ray(r0, cc);
                ray(r1, cc);
            }
            for (i64 rr = r0; rr <= r1; ++rr) {
                ray(rr, c0);
                ray(rr, c1);
            }
            // the cells no ray passed through: their own line of sight
            for (i64 rr = r0; rr <= r1; ++rr) {
                for (i64 cc = c0; cc <= c1; ++cc) {
                    const i64 i = rr * nx + cc;
                    const double d = dist(rr, cc);
                    if (visited[local(rr, cc)] || std::isnan(dem[i]) || d > reach) continue;
                    const i64 dr = rr - orow, dc = cc - ocol;
                    const i64 steps = std::max(std::llabs(dr), std::llabs(dc));
                    double steepest = -INFINITY;
                    for (i64 k = 1; k < steps; ++k) {
                        const i64 r = orow + static_cast<i64>(std::round(static_cast<double>(dr) *
                                                                         static_cast<double>(k) /
                                                                         static_cast<double>(steps)));
                        const i64 c = ocol + static_cast<i64>(std::round(static_cast<double>(dc) *
                                                                         static_cast<double>(k) /
                                                                         static_cast<double>(steps)));
                        const double z = dem[r * nx + c];
                        if (!std::isnan(z)) {
                            const double e = dist(r, c);
                            steepest = std::max(steepest, (z - drop(e) - z0) / e);
                        }
                    }
                    seen[local(rr, cc)] = (dem[i] - drop(d) + target - z0) / d >= steepest;
                }
            }
            for (i64 rr = r0; rr <= r1; ++rr)
                for (i64 cc = c0; cc <= c1; ++cc)
                    if (seen[local(rr, cc)]) ++mine[static_cast<std::size_t>(rr * nx + cc)];
        }
#pragma omp critical
        for (i64 i = 0; i < n; ++i) count[static_cast<std::size_t>(i)] += mine[static_cast<std::size_t>(i)];
    }
    for (i64 i = 0; i < n; ++i) out[i] = std::isnan(dem[i]) ? kNaN : static_cast<double>(count[static_cast<std::size_t>(i)]);
}

}  // namespace stats
