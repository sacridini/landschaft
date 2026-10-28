// SPDX-License-Identifier: GPL-3.0-or-later
//
// Numerical pieces shared by the statistics of maps: scipy.special's normal
// distribution (Cephes' ndtr, ndtri, erf, erfc: the same bits), numpy's pairwise
// sum, and numpy's default generator (np.random.default_rng(seed): PCG64 seeded by
// a SeedSequence, with its integers, random, shuffle and choice without replacement).
#pragma once

#include <cstdint>
#include <vector>

namespace stats {

using i64 = std::int64_t;
using u64 = std::uint64_t;
using u32 = std::uint32_t;

double erf(double x);
double erfc(double a);
// The standard normal's distribution function (scipy's norm.cdf).
double ndtr(double a);
// Its quantile (scipy's norm.ppf).
double ndtri(double y0);

// numpy's pairwise sum of a contiguous array of doubles (np.sum).
double np_sum(const double* a, i64 n);
inline double np_sum(const std::vector<double>& v) { return np_sum(v.data(), static_cast<i64>(v.size())); }

// numpy's default generator.
class NumpyRng {
public:
    explicit NumpyRng(u64 seed);
    // bit_generator.random_raw
    u64 next_u64();
    // integers(n): [0, n), Lemire's method as numpy
    u64 integers(u64 n);
    // random(): [0, 1)
    double random();
    // shuffle of a 1-D array (Fisher-Yates from the end, numpy's random_interval)
    template <class T>
    void shuffle(T* v, i64 n) {
        for (i64 i = n - 1; i >= 1; --i) {
            const i64 j = static_cast<i64>(interval(static_cast<u64>(i)));
            T t = v[i];
            v[i] = v[j];
            v[j] = t;
        }
    }
    // choice(pop, size, replace=False): numpy's order
    std::vector<u64> choice_distinct(u64 pop, u64 size);

private:
    u32 next_u32();
    u64 interval(u64 max);
    void step();
    u64 state_hi_ = 0, state_lo_ = 0, inc_hi_ = 0, inc_lo_ = 0;
    bool has_buffer_ = false;
    u32 buffer_ = 0;
};

}  // namespace stats
