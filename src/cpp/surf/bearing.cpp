// SPDX-License-Identifier: GPL-3.0-or-later
//
// The bearing area (Abbott-Firestone) curve parameters (geodiv's
// bearingarea.R, bestfit.R, simpsons.R): heights by R's quantile(type = 7)
// of z, seq(), Simpson's rule for the areas above the curve, and find_flat's
// core line: slopes of the curve by differences, their mean absolute value
// over every 40 % window, and lm() through the flattest window.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "lsm/landscape.hpp"
#include "lsm/rmath.hpp"
#include "surf/surface.hpp"

namespace surf {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
using Index = std::size_t;

// Sorts v ascending: chunks in parallel, then merged pairwise.
void parallel_sort(std::vector<double>& v) {
    const i64 n = static_cast<i64>(v.size());
    int parts = 1;
#if defined(_OPENMP)
    if (n > (1 << 20)) {
        parts = 1;
        while (parts * 2 <= omp_get_max_threads() && parts < 64) parts *= 2;
    }
#endif
    if (parts == 1) {
        std::sort(v.begin(), v.end());
        return;
    }
    std::vector<i64> cut(static_cast<Index>(parts) + 1);
    for (int p = 0; p <= parts; ++p) cut[static_cast<Index>(p)] = n * p / parts;
#pragma omp parallel for schedule(static) if (!omp_in_parallel())
    for (int p = 0; p < parts; ++p)
        std::sort(v.begin() + cut[static_cast<Index>(p)], v.begin() + cut[static_cast<Index>(p) + 1]);
    for (int width = 1; width < parts; width *= 2) {
#pragma omp parallel for schedule(static) if (!omp_in_parallel())
        for (int p = 0; p < parts; p += 2 * width) {
            const int q = std::min(p + width, parts), e = std::min(p + 2 * width, parts);
            std::inplace_merge(v.begin() + cut[static_cast<Index>(p)], v.begin() + cut[static_cast<Index>(q)],
                               v.begin() + cut[static_cast<Index>(e)]);
        }
    }
}

// seq(from, to, length.out = len) of R for len > 2, from < to (doubles).
std::vector<double> r_seq_len(double from, double to, int len) {
    const double n1 = static_cast<double>(len) - 1;
    const double by = (to - from) / n1;
    std::vector<double> x(static_cast<Index>(len));
    x[0] = from;
    for (int k = 1; k < len - 1; ++k) x[static_cast<Index>(k)] = from + static_cast<double>(k) * by;
    x[static_cast<Index>(len) - 1] = to;
    return x;
}

}  // namespace

Bearing bearing_curve(const double* z, i64 n) {
    Bearing b;
    b.s.reserve(static_cast<Index>(n));
    for (i64 i = 0; i < n; ++i)
        if (!std::isnan(z[i])) b.s.push_back(z[i]);
    parallel_sort(b.s);
    // the curve: distinct heights from the highest, and the share of the
    // cells at or above each
    const i64 m = static_cast<i64>(b.s.size());
    const double dn = static_cast<double>(m);
    i64 above = 0;
    for (i64 j = m; j > 0;) {
        i64 i = j - 1;
        while (i > 0 && b.s[static_cast<Index>(i - 1)] == b.s[static_cast<Index>(j - 1)]) --i;
        above += j - i;
        b.heights.push_back(b.s[static_cast<Index>(j - 1)]);
        b.bearing.push_back(static_cast<double>(above) / dn);
        j = i;
    }
    return b;
}

// quantile(z, probs = q, type = 7) of R on the sorted sample.
double Bearing::quantile(double q) const {
    const i64 n = static_cast<i64>(s.size());
    if (n == 0 || std::isnan(q)) return NaN;
    q = std::max(0.0, std::min(1.0, q));
    const double index = 1 + std::max(static_cast<double>(n) - 1, 0.0) * q;
    const double lo = std::floor(index), hi = std::ceil(index);
    double qs = s[static_cast<Index>(lo) - 1];
    const double xhi = s[static_cast<Index>(hi) - 1];
    if (index > lo && xhi != qs) {
        const double h = index - lo;
        qs = (1 - h) * qs + h * xhi;
    }
    return qs;
}

// The share of the cells at or above height v.
double Bearing::share_above(double v) const {
    if (std::isnan(v) || s.empty()) return NaN;
    const auto it = std::lower_bound(s.begin(), s.end(), v);
    return static_cast<double>(s.end() - it) / static_cast<double>(s.size());
}

// The area between the curve and its top over bearings [a, b]: max(y) (b - a)
// minus Simpson's rule on n intervals (n even). geodiv subtracts min(y) (b -
// a) as well unless the curve dips below 0, and drops the curve's last point.
double area_above(const Bearing& b, double a, double bb, int n) {
    if (b.s.empty()) return NaN;
    const double h = (bb - a) / n;
    const std::vector<double> x = r_seq_len(a, bb, n + 1);
    std::vector<double> y(x.size());
    for (Index k = 0; k < x.size(); ++k) y[k] = b.height(x[k]);
    double yhi = -std::numeric_limits<double>::infinity();
    for (double v : y) yhi = std::max(yhi, v);
    // y[seq(2, n, 2)] and y[seq(3, n - 1, 2)] (1-based)
    const double s4 = lsm::sum_of(0, n / 2, [&](i64 k) { return 4 * y[static_cast<Index>(2 * k + 1)]; });
    const double s2 = lsm::sum_of(0, n / 2 - 1, [&](i64 k) { return 2 * y[static_cast<Index>(2 * k + 2)]; });
    const double s = (h / 3) * (((y[0] + s4) + s2) + y[static_cast<Index>(n)]);
    return (yhi * (x.back() - x.front())) - s;
}

CoreLine find_flat(const Bearing& b) {
    CoreLine out{NaN, NaN, NaN, NaN};
    if (b.s.empty()) return out;
    const int len = 10000;
    const std::vector<double> x = r_seq_len(0, 1, len);
    // the slopes of the curve: central differences, one-sided at the ends
    const double h = 0.001;
    const double space = 1.0 / len;
    double end_length = h / space;
    if (end_length < 1) end_length = 1;
    const i64 begin = static_cast<i64>(std::ceil(1 + end_length));  // 1-based
    const i64 end = static_cast<i64>(std::floor(len - end_length));
    std::vector<double> absolute(static_cast<Index>(len));
#pragma omp parallel for schedule(static) if (!omp_in_parallel())
    for (int i = 1; i <= len; ++i) {
        const double xi = x[static_cast<Index>(i) - 1];
        double v;
        if (i <= begin - 1) v = (b.height(xi + h) - b.height(xi)) / h;
        else if (i >= end + 1) v = (b.height(xi) - b.height(xi - h)) / h;
        else v = (b.height(xi + h) - b.height(xi - h)) / (2 * h);
        absolute[static_cast<Index>(i) - 1] = std::fabs(v);
    }
    // the first 40 % window (4001 slopes) with the least mean(abs(slope)):
    // means from prefix sums (relative error ~1e-15) pick the candidates within
    // 1e-9 of the least, then R's mean() of those decides, in order (the same
    // window as R's mean of every window, 6000 x 4001 80-bit additions)
    const double length = static_cast<double>(len) * 0.4;
    const i64 end_ind = static_cast<i64>(len - length);
    const i64 width = static_cast<i64>(length) + 1;
    for (double v : absolute)
        if (std::isnan(v)) return out;
    std::vector<double> prefix(static_cast<Index>(len) + 1, 0.0);
    for (int i = 0; i < len; ++i) prefix[static_cast<Index>(i) + 1] = prefix[static_cast<Index>(i)] + absolute[static_cast<Index>(i)];
    std::vector<double> approx(static_cast<Index>(end_ind));
    double amin = std::numeric_limits<double>::infinity();
    for (i64 i = 0; i < end_ind; ++i) {
        approx[static_cast<Index>(i)] = (prefix[static_cast<Index>(i + width)] - prefix[static_cast<Index>(i)]) / width;
        amin = std::min(amin, approx[static_cast<Index>(i)]);
    }
    const double margin = std::fabs(amin) * 1e-9 + prefix.back() * 1e-13 + 1e-300;
    double lo = std::numeric_limits<double>::infinity();
    i64 w = -1;
    for (i64 i = 0; i < end_ind; ++i) {
        if (approx[static_cast<Index>(i)] > amin + margin) continue;
        const double m = lsm::r_mean(absolute.data() + i, width);
        if (m < lo) {
            lo = m;
            w = i;
        }
    }
    if (w < 0) return out;
    const double xstart = x[static_cast<Index>(w)];
    const double xend = x[static_cast<Index>(w + width) - 1];
    // lm(y ~ x) through the curve's points in the window
    std::vector<double> px, py;
    for (Index k = 0; k < b.heights.size(); ++k) {
        const double xv = b.bearing[k];
        if (xv >= xstart && xv <= xend) {
            px.push_back(xv);
            py.push_back(b.heights[k]);
        }
    }
    if (px.empty()) return out;
    double b0, b1;
    lsm::r_lm_coef(px.data(), py.data(), static_cast<int>(px.size()), &b0, &b1);
    out.high = std::isnan(b1) ? b0 : (0.0 + b0 * 1.0) + b1 * 0.0;
    out.low = std::isnan(b1) ? b0 : (0.0 + b0 * 1.0) + b1 * 1.0;
    out.smr1 = b.share_above(out.high);
    out.smr2 = b.share_above(out.low);
    return out;
}

BearingParameters bearing_parameters(const Bearing& b, double sq) {
    BearingParameters p{};
    p.sbi = sq / b.height(0.05);
    p.svi = area_above(b, 0.8, 1, 500);
    p.sci = area_above(b, 0.05, 1, 1000) - p.svi;
    const CoreLine line = find_flat(b);
    p.sk = std::fabs(line.high - line.low);
    p.spk = std::fabs(b.height(0) - b.height(line.smr1));
    p.svk = std::fabs(b.height(1) - b.height(line.smr2));
    return p;
}

}  // namespace surf
