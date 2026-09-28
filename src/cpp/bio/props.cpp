// SPDX-License-Identifier: GPL-3.0-or-later
//
// Statistics objects of the property, numeric label and matrix indices (see
// props.hpp).

#include "bio/props.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "bio/perlmath.hpp"

namespace bio {

namespace {
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
enum Key { COUNT, SUMW, SUM, MEAN, MIN, MAX, RANGE, MEDIAN, SD, SKEW, KURT, IQR, HMEAN, GMEAN };

// PDL sum: a double accumulator from 0
template <class F>
double pdl_sum(std::size_t n, F f) {
    double t = 0;
    for (std::size_t i = 0; i < n; ++i) t += f(i);
    return t;
}
}  // namespace

// ---------------------------------------------------------------------------------------------
// Statistics::Descriptive::PDL::SampleWeighted

void WStats::set(std::vector<double> data, std::vector<std::int64_t> weights) {
    d_ = std::move(data);
    w_ = std::move(weights);
    sorted_ = false;
    have_cum_ = false;
    cache_.clear();
    pct_.clear();
}

template <class F>
double WStats::cached(int key, F f) {
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    if (d_.empty()) return NaN;  // no ndarray: undef, not cached
    const double v = f();
    cache_.emplace(key, v);
    return v;
}

void WStats::sort() {
    if (sorted_) return;
    std::vector<std::size_t> idx(d_.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        const double x = d_[a], y = d_[b];
        if (std::isnan(x)) return false;
        if (std::isnan(y)) return true;
        return x < y;
    });
    std::vector<double> d(d_.size());
    std::vector<std::int64_t> w(w_.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
        d[i] = d_[idx[i]];
        w[i] = w_[idx[i]];
    }
    d_ = std::move(d);
    w_ = std::move(w);
    sorted_ = true;
    have_cum_ = false;
}

const std::vector<std::int64_t>& WStats::cumsum() {
    if (!have_cum_) {
        cum_.resize(w_.size());
        std::int64_t t = 0;
        for (std::size_t i = 0; i < w_.size(); ++i) cum_[i] = t += w_[i];
        have_cum_ = true;
    }
    return cum_;
}

double WStats::count() {
    return cached(COUNT, [&] { return static_cast<double>(std::accumulate(w_.begin(), w_.end(), std::int64_t(0))); });
}

double WStats::sum_weights() {
    return cached(SUMW, [&] { return static_cast<double>(std::accumulate(w_.begin(), w_.end(), std::int64_t(0))); });
}

double WStats::sum() {
    return cached(SUM, [&] { return pdl_sum(d_.size(), [&](std::size_t i) { return d_[i] * static_cast<double>(w_[i]); }); });
}

double WStats::mean() {
    return cached(MEAN, [&] {
        const double s = pdl_sum(d_.size(), [&](std::size_t i) { return d_[i] * static_cast<double>(w_[i]); });
        return s / static_cast<double>(std::accumulate(w_.begin(), w_.end(), std::int64_t(0)));
    });
}

double WStats::min() {
    return cached(MIN, [&] { return *std::min_element(d_.begin(), d_.end()); });
}

double WStats::max() {
    return cached(MAX, [&] { return *std::max_element(d_.begin(), d_.end()); });
}

double WStats::sample_range() {
    return cached(RANGE, [&] { return max() - min(); });
}

double WStats::median() {
    return cached(MEDIAN, [&] {
        sort();
        const auto& cum = cumsum();
        const double target = sum_weights() * 0.5;
        std::size_t idx = 0;
        while (idx < cum.size() && static_cast<double>(cum[idx]) < target) ++idx;
        if (idx < cum.size() && target == static_cast<double>(cum[idx]) && idx + 1 < d_.size())
            return (d_[idx] + d_[idx + 1]) / 2;
        return idx < d_.size() ? d_[idx] : NaN;
    });
}

double WStats::sd() {
    return cached(SD, [&] {
        const std::int64_t n = std::accumulate(w_.begin(), w_.end(), std::int64_t(0));
        if (n == 1) return 0.0;
        const double s2 = pdl_sum(d_.size(), [&](std::size_t i) { return perl_pow(d_[i], 2) * static_cast<double>(w_[i]); });
        const double var = s2 - static_cast<double>(n) * perl_pow(mean(), 2);
        return var > 0 ? std::sqrt(var / static_cast<double>(n - 1)) : 0.0;
    });
}

double WStats::skewness() {
    return cached(SKEW, [&] {
        const double n = sum_weights();
        if (n < 3) return NaN;
        const double m = mean(), s = sd();
        const double s3 = pdl_sum(d_.size(), [&](std::size_t i) {
            return static_cast<double>(w_[i]) * perl_pow((d_[i] - m) / s, 3);
        });
        const double correction = n / ((n - 1) * (n - 2));
        return correction * s3;
    });
}

double WStats::kurtosis() {
    return cached(KURT, [&] {
        const double n = sum_weights();
        if (n <= 3) return NaN;
        const double m = mean(), s = sd();
        const double s4 = pdl_sum(d_.size(), [&](std::size_t i) {
            return static_cast<double>(w_[i]) * perl_pow((d_[i] - m) / s, 4);
        });
        const double c1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3));
        const double c2 = (3 * perl_pow(n - 1, 2)) / ((n - 2) * (n - 3));
        return (c1 * s4) - c2;
    });
}

double WStats::percentile(double p) {
    if (d_.empty()) return NaN;
    if (p == 50) return median();
    auto it = pct_.find(p);
    if (it != pct_.end()) return it->second;
    sort();
    const auto& cum = cumsum();
    const double wt_sum = sum_weights();
    const double target = (p / 100) * (wt_sum - 1) + 1;
    const double k = std::floor(target);
    const double d = target - k;
    std::size_t idx = 0;
    while (idx < cum.size() && static_cast<double>(cum[idx]) < k) ++idx;
    double val = NaN;
    if (idx < d_.size()) {
        const double fraction = target - static_cast<double>(cum[idx]);
        if (fraction > 0 && fraction < 1 && idx + 1 < d_.size()) {
            const double lower = d_[idx], upper = d_[idx + 1];
            val = lower + d * (upper - lower);
        } else {
            val = d_[idx];
        }
    }
    if (std::trunc(p) == p) pct_[p] = val;
    return val;
}

double WStats::iqr() {
    return cached(IQR, [&] { return percentile(75) - percentile(25); });
}

double WStats::harmonic_mean() {
    return cached(HMEAN, [&] {
        for (double x : d_)
            if (x == 0) return NaN;
        const double hs = pdl_sum(d_.size(), [&](std::size_t i) { return (1 / d_[i]) * static_cast<double>(w_[i]); });
        return hs != 0 ? count() / hs : NaN;
    });
}

double WStats::geometric_mean() {
    return cached(GMEAN, [&] {
        for (double x : d_)
            if (x < 0) return NaN;
        const double s = pdl_sum(d_.size(), [&](std::size_t i) { return perl_log(d_[i]) * static_cast<double>(w_[i]); });
        return perl_exp(s / static_cast<double>(std::accumulate(w_.begin(), w_.end(), std::int64_t(0))));
    });
}

// ---------------------------------------------------------------------------------------------
// Biodiverse::Statistics

void DStats::add(const std::vector<double>& v) {
    if (v.empty()) return;
    // a single add_data (the only use): sums with List::Util, starting from 0
    double s = v[0], sq = perl_pow(v[0], 2);
    double mx = v[0], mn = v[0];
    for (std::size_t i = 1; i < v.size(); ++i) {
        s += v[i];
        sq += perl_pow(v[i], 2);
    }
    for (double x : v) {
        if (!(x < mx)) mx = x;
        if (x < mn) mn = x;
    }
    sum_ = 0 + s;
    sumsq_ = 0 + sq;
    max_ = mx;
    min_ = mn;
    data_ = v;
    sorted_ = false;
}

double DStats::mean() const { return data_.empty() ? NaN : sum_ / count(); }

double DStats::variance() const {
    const double n = count();
    if (n == 0) return NaN;
    if (n == 1) return 0;
    double var = sumsq_ - n * perl_pow(mean(), 2);
    if (var < 0)
        var = 0;
    else
        var /= n - 1;
    return var;
}

double DStats::sd() const {
    const double v = variance();
    return std::isnan(v) ? NaN : perl_sqrt(v);
}

void DStats::sort_data() {
    if (sorted_) return;
    std::stable_sort(data_.begin(), data_.end());
    sorted_ = true;
}

double DStats::median() {
    const std::size_t c = data_.size();
    if (c == 0) return NaN;
    sort_data();
    if (c % 2) return data_[(c - 1) / 2];
    return (data_[c / 2] + data_[(c - 2) / 2]) / 2;
}

double DStats::percentile(double p) {
    const double c = count();
    if (c == 0) return NaN;
    p = p > 100 ? 100 : (p < 0 ? 0 : p);
    sort_data();
    const double num = (c - 1) * p / 100;
    const auto index = static_cast<std::size_t>(std::trunc(num + 0.5));
    return data_[index];
}

double DStats::skewness() {
    const double n = count();
    const double s = sd();
    if (std::isnan(s) || s == 0 || !(n > 2)) return NaN;
    const double m = mean();
    double sum3 = 0;
    for (double x : data_) sum3 += perl_pow((x - m) / s, 3);
    const double correction = n / ((n - 1) * (n - 2));
    return correction * sum3;
}

double DStats::kurtosis() {
    const double n = count();
    const double s = sd();
    if (std::isnan(s) || s == 0 || !(n > 3)) return NaN;
    const double m = mean();
    double sum4 = 0;
    for (double x : data_) sum4 += perl_pow((x - m) / s, 4);
    const double c1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3));
    const double c2 = (3 * perl_pow(n - 1, 2)) / ((n - 2) * (n - 3));
    return (c1 * sum4) - c2;
}

// ---------------------------------------------------------------------------------------------

double gistar(const GlobalSummary& g, WStats& local) {
    const double sum = local.sum();
    if (std::isnan(sum)) return NaN;
    const double n = g.count;
    const double w = local.count();
    const double s1 = w;
    const double expected = w * g.mean;
    const double numerator = sum - expected;
    if (w == 0) return NaN;
    const double denominator = g.sd * perl_sqrt(((n * s1) - perl_pow(w, 2)) / (n - 1));
    return denominator != 0 ? numerator / denominator : 0.0;
}

}  // namespace bio
