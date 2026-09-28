// SPDX-License-Identifier: GPL-3.0-or-later
//
// The statistics objects of Biodiverse's element property, numeric label and
// matrix indices.
//
// WStats is Statistics::Descriptive::PDL::SampleWeighted as those indices use
// it: values (hash keys, in Perl's key order) with integer weights, cached
// results, and a sort that happens the first time a median or percentile is
// asked for, after which sums run in sorted order. One object is shared by
// all the calculations of a group, so results depend on the order they run in.
// PDL arithmetic: plain double sums from 0, pow for integer exponents by
// squaring, sqrt correctly rounded.
//
// DStats is Biodiverse::Statistics (Statistics::Descriptive::Full).

#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace bio {

class WStats {
public:
    void set(std::vector<double> data, std::vector<std::int64_t> weights);
    bool empty() const { return d_.empty(); }
    double count();
    double sum_weights();
    double sum();
    double mean();
    double min();
    double max();
    double sample_range();
    double median();
    double sd();
    double skewness();
    double kurtosis();
    double iqr();
    double percentile(double p);
    double harmonic_mean();
    double geometric_mean();
    const std::vector<double>& data() const { return d_; }
    const std::vector<std::int64_t>& weights() const { return w_; }

private:
    std::vector<double> d_;
    std::vector<std::int64_t> w_;
    bool sorted_ = false;
    std::vector<std::int64_t> cum_;
    bool have_cum_ = false;
    std::map<int, double> cache_;
    std::map<double, double> pct_;
    void sort();
    const std::vector<std::int64_t>& cumsum();
    template <class F>
    double cached(int key, F f);
};

class DStats {
public:
    void add(const std::vector<double>& v);
    double count() const { return static_cast<double>(data_.size()); }
    double sum() const { return sum_; }
    double mean() const;
    double min() const { return min_; }
    double max() const { return max_; }
    double sample_range() const { return data_.empty() ? std::numeric_limits<double>::quiet_NaN() : max_ - min_; }
    double variance() const;
    double sd() const;
    double median();
    double percentile(double p);
    double skewness();
    double kurtosis();
    void sort_data();
    const std::vector<double>& data() const { return data_; }

private:
    std::vector<double> data_;
    double sum_ = 0, sumsq_ = 0, min_ = std::numeric_limits<double>::quiet_NaN(),
           max_ = std::numeric_limits<double>::quiet_NaN();
    bool sorted_ = false;
};

// Getis-Ord Gi* (Biodiverse::Indices::_get_gistar_score) from global summary
// statistics and a group's statistics object.
struct GlobalSummary {
    double mean = std::numeric_limits<double>::quiet_NaN(), sum = std::numeric_limits<double>::quiet_NaN(),
           sd = std::numeric_limits<double>::quiet_NaN(), count = std::numeric_limits<double>::quiet_NaN();
};
double gistar(const GlobalSummary& g, WStats& local);

}  // namespace bio
