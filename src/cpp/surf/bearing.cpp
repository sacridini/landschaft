// SPDX-License-Identifier: GPL-3.0-or-later
//
// geodiv's bearing area (Abbott-Firestone) curve parameters (bearingarea.R,
// bestfit.R, simpsons.R): the ecdf of 1 - z and R's quantile(type = 7) of it,
// seq(), Simpson's rule as area_above, and find_flat's core line: slopes of
// the curve by central differences, their mean absolute value over every 40 %
// window, and lm() through the flattest window.

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
#pragma omp parallel for schedule(static)
    for (int p = 0; p < parts; ++p)
        std::sort(v.begin() + cut[static_cast<Index>(p)], v.begin() + cut[static_cast<Index>(p) + 1]);
    for (int width = 1; width < parts; width *= 2) {
#pragma omp parallel for schedule(static)
        for (int p = 0; p < parts; p += 2 * width) {
            const int q = std::min(p + width, parts), e = std::min(p + 2 * width, parts);
            std::inplace_merge(v.begin() + cut[static_cast<Index>(p)], v.begin() + cut[static_cast<Index>(q)],
                               v.begin() + cut[static_cast<Index>(e)]);
        }
    }
}

// seq(from, to, by = by) of R (seq.default, doubles).
std::vector<double> r_seq_by(double from, double to, double by) {
    const double del = to - from;
    if (del == 0 && to == 0) return {to};
    const double n = del / by;
    if (std::fabs(del) / std::max(std::fabs(to), std::fabs(from)) < 100 * std::numeric_limits<double>::epsilon())
        return {from};
    const int m = static_cast<int>(n + 1e-10);
    std::vector<double> x(static_cast<Index>(m) + 1);
    for (int k = 0; k <= m; ++k) x[static_cast<Index>(k)] = std::min(from + static_cast<double>(k) * by, to);
    return x;
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
        if (!std::isnan(z[i])) b.s.push_back(1 - z[i]);
    parallel_sort(b.s);
    // ecdf: the distinct values and cumsum(counts) / n
    const i64 m = static_cast<i64>(b.s.size());
    const double dn = static_cast<double>(m);
    for (i64 i = 0; i < m;) {
        i64 j = i + 1;
        while (j < m && b.s[static_cast<Index>(j)] == b.s[static_cast<Index>(i)]) ++j;
        b.vals.push_back(b.s[static_cast<Index>(i)]);
        b.cum.push_back(static_cast<double>(j) / dn);
        i = j;
    }
    return b;
}

// quantile(x, probs = p, type = 7) of R on the sorted sample.
double Bearing::quantile(double p) const {
    const i64 n = static_cast<i64>(s.size());
    if (n == 0 || std::isnan(p)) return NaN;
    p = std::max(0.0, std::min(1.0, p));
    const double index = 1 + std::max(static_cast<double>(n) - 1, 0.0) * p;
    const double lo = std::floor(index), hi = std::ceil(index);
    double qs = s[static_cast<Index>(lo) - 1];
    const double xhi = s[static_cast<Index>(hi) - 1];
    if (index > lo && xhi != qs) {
        const double h = index - lo;
        qs = (1 - h) * qs + h * xhi;
    }
    return qs;
}

// The ecdf at v: approxfun(vals, cum, "constant", yleft = 0, yright = 1, f = 0).
double Bearing::ecdf(double v) const {
    if (std::isnan(v) || vals.empty()) return NaN;
    if (v < vals.front()) return 0.0;
    if (v > vals.back()) return 1.0;
    const auto it = std::upper_bound(vals.begin(), vals.end(), v);
    return cum[static_cast<Index>(it - vals.begin()) - 1];
}

// geodiv's area_above(f, a, b, n): the area of the box around the curve of
// heights 1 - quantile(f, x) over x in seq(a, b, (b - a) / n), minus the area
// under it by Simpson's rule (as geodiv writes it: y[[n]] where the last point
// is y[[n + 1]]).
double area_above(const Bearing& b, double a, double bb, int n) {
    if (b.s.empty()) return NaN;
    const double h = (bb - a) / n;
    const std::vector<double> x = r_seq_by(a, bb, h);
    std::vector<double> y(x.size());
    for (Index k = 0; k < x.size(); ++k) y[k] = 1 - b.quantile(x[k]);
    double ylo = std::numeric_limits<double>::infinity(), yhi = -ylo;
    for (double v : y) {
        ylo = std::min(ylo, v);
        yhi = std::max(yhi, v);
    }
    if (ylo < 0) {
        const double shift = std::fabs(ylo);
        for (double& v : y) v = v + shift;
        ylo = std::numeric_limits<double>::infinity();
        yhi = -ylo;
        for (double v : y) {
            ylo = std::min(ylo, v);
            yhi = std::max(yhi, v);
        }
    }
    // y[seq(2, n - 1, by = 2)] and y[seq(3, n - 1, by = 2)] (1-based)
    const std::vector<double> even = r_seq_by(2, n - 1, 2), odd = r_seq_by(3, n - 1, 2);
    const double s4 = lsm::sum_of(0, static_cast<i64>(even.size()),
                                  [&](i64 k) { return 4 * y[static_cast<Index>(even[static_cast<Index>(k)]) - 1]; });
    const double s2 = lsm::sum_of(0, static_cast<i64>(odd.size()),
                                  [&](i64 k) { return 2 * y[static_cast<Index>(odd[static_cast<Index>(k)]) - 1]; });
    const double s = (h / 3) * (((y[0] + s4) + s2) + y[static_cast<Index>(n) - 1]);
    double xlo = std::numeric_limits<double>::infinity(), xhi = -xlo;
    for (double v : x) {
        xlo = std::min(xlo, v);
        xhi = std::max(xhi, v);
    }
    return ((yhi - ylo) * (xhi - xlo)) - s;
}

CoreLine find_flat(const Bearing& b, double perc) {
    (void)perc;  // geodiv's slopemeans uses its default l = 0.4 whatever perc is
    CoreLine out{NaN, NaN, NaN, NaN};
    if (b.s.empty()) return out;
    const int len = 10000;
    const std::vector<double> x = r_seq_len(0, 1, len);
    // slopecalc(x, h = 0.001, f)
    const double h = 0.001;
    const double space = 1.0 / len;
    double end_length = h / space;
    if (end_length < 1) end_length = 1;
    const i64 begin = static_cast<i64>(std::ceil(1 + end_length));  // 1-based
    const i64 end = static_cast<i64>(std::floor(len - end_length));
    std::vector<double> slope(static_cast<Index>(len));
#pragma omp parallel for schedule(static)
    for (int i = 1; i <= len; ++i) {
        const double xi = x[static_cast<Index>(i) - 1];
        double pos, neg;
        if (i <= begin - 1) {
            pos = 1 - b.quantile(xi + h);
            neg = 1 - b.quantile(xi);
        } else if (i >= end + 1) {
            pos = 1 - b.quantile(xi);
            neg = 1 - b.quantile(xi - h);
        } else {
            pos = 1 - b.quantile(xi + h);
            neg = 1 - b.quantile(xi - h);
        }
        slope[static_cast<Index>(i) - 1] = (pos - neg) / (2 * h);
    }
    // slopemeans(slopes, l = 0.4): mean(abs(slope)) of every window
    const double l = 0.4;
    const double length = static_cast<double>(len) * l;
    const i64 end_ind = static_cast<i64>(len - length);
    const i64 begin_ind = static_cast<i64>(1 + length);
    const i64 width = len - begin_ind + 1;  // iend[i] - istart[i] + 1 = begin_ind
    (void)width;
    std::vector<double> absolute(slope.size());
    for (Index i = 0; i < slope.size(); ++i) absolute[i] = std::fabs(slope[i]);
    std::vector<double> means(static_cast<Index>(end_ind));
#pragma omp parallel for schedule(static)
    for (i64 i = 0; i < end_ind; ++i)
        means[static_cast<Index>(i)] = lsm::r_mean(absolute.data() + i, begin_ind);
    // the first window with the least mean (min() without na.rm: NA if any is)
    double lo = std::numeric_limits<double>::infinity();
    for (double m : means) {
        if (std::isnan(m)) return out;
        lo = std::min(lo, m);
    }
    i64 w = 0;
    while (w < end_ind && !(means[static_cast<Index>(w)] == lo)) ++w;
    const double xstart = x[static_cast<Index>(w)];
    const double xend = x[static_cast<Index>(w + begin_ind) - 1];
    // lm(y ~ x) through the curve's points in the window
    std::vector<double> px, py;
    for (Index k = 0; k < b.vals.size(); ++k) {
        const double xv = b.cum[k];
        if (xv >= xstart && xv <= xend) {
            px.push_back(xv);
            py.push_back(1 - b.vals[k]);
        }
    }
    if (px.empty()) return out;
    double b0, b1;
    lsm::r_lm_coef(px.data(), py.data(), static_cast<int>(px.size()), &b0, &b1);
    // predict.lm at x = 0 and x = 1 (dgemv: 0 + b0 * 1, then + b1 * x)
    out.high = std::isnan(b1) ? b0 : (0.0 + b0 * 1.0) + b1 * 0.0;
    out.low = std::isnan(b1) ? b0 : (0.0 + b0 * 1.0) + b1 * 1.0;
    out.smr1 = b.ecdf(1 - out.high);
    out.smr2 = b.ecdf(1 - out.low);
    return out;
}

BearingParameters bearing_parameters(const Bearing& b, double sq) {
    BearingParameters p{};
    const double z05 = 1 - b.quantile(0.05);
    p.sbi = sq / z05;
    p.svi = area_above(b, 0.8, 1, 500);
    p.sci = area_above(b, 0.05, 1, 1000) - p.svi;
    const CoreLine line = find_flat(b, 0.4);
    p.sk = std::fabs(line.high - line.low);
    p.spk = std::fabs((1 - b.quantile(0)) - (1 - b.quantile(line.smr1)));
    p.svk = std::fabs((1 - b.quantile(1)) - (1 - b.quantile(line.smr2)));
    return p;
}

}  // namespace surf
