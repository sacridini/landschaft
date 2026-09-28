// SPDX-License-Identifier: GPL-3.0-or-later
//
// The two random number generators behind NLMR's neutral landscape models,
// reproduced bit for bit so that the same seed gives the same landscape:
//
// - RRng: R's default generator (Mersenne-Twister seeded by set.seed(),
//   norm_rand() by inversion, sample() by rejection sampling), from R 4.4's
//   src/main/RNG.c, random.c, unique.c and src/nmath/qnorm.c;
// - StdMt19937 and the std_* functions: std::mt19937 with the distributions
//   of libstdc++ (GCC 13, the Rtools compiler that builds NLMR's C++ code),
//   whose algorithms the C++ standard leaves to the implementation.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace nlm {

using i64 = std::int64_t;

// -- R ------------------------------------------------------------------------

class RRng {
public:
    // set.seed(seed) with the default kinds (Mersenne-Twister, Inversion,
    // Rejection).
    explicit RRng(std::int32_t seed = 0) { set_seed(seed); }
    void set_seed(std::int32_t seed);

    // unif_rand(): (0, 1), never 0 or 1.
    double unif_rand() { return fixup(genrand()); }
    // norm_rand() by inversion.
    double norm_rand();
    // R_unif_index(dn): an integer in [0, dn) as a double (rejection sampling).
    double unif_index(double dn);
    // runif(1, a, b) and rnorm(1, mu, sigma).
    double runif(double a, double b);
    double rnorm(double mu, double sigma);

    // Vectors: runif(n, a, b) and rnorm(n, mu, sigma). rnorm draws the
    // uniforms in sequence and inverts them with `n_threads` threads.
    void runif(double* out, i64 n, double a = 0.0, double b = 1.0);
    void rnorm(double* out, i64 n, double mu = 0.0, double sigma = 1.0);

    // exp_rand() (Ahrens & Dieter 1972) and rpois(mu) (Ahrens & Dieter 1982).
    double exp_rand();
    double rpois(double mu);

    // sample.int(n, size, replace, prob) (1-based results), including the
    // hashed sample2() that sample.int() uses for n > 1e7 without
    // replacement or prob and size <= n / 2.
    std::vector<i64> sample_int(i64 n, i64 size, bool replace, const std::vector<double>* prob = nullptr);

private:
    std::uint32_t mt_[624];
    int mti_ = 625;

    // rpois()'s parameters kept between calls with the same mu (as R's
    // static variables; they only save work)
    struct PoisState {
        int l = 0, m = 0;
        double b1 = 0, b2 = 0, c = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        double pp[36] = {}, p0 = 0, p = 0, q = 0, s = 0, d = 0, omega = 0;
        double big_l = 0, muprev = 0, muprev2 = 0;
    } pois_;

    static double fixup(double x);
    double genrand();
    double rbits(int bits);
    std::vector<i64> sample2(i64 n, i64 size);
};

// R's qnorm(p, 0, 1) for p in (0, 1) (lower tail, not log), as norm_rand()
// calls it.
double r_qnorm(double p);

// R's revsort(): heapsort into descending order, `ib` alongside.
void r_revsort(double* a, int* ib, int n);

// -- libstdc++ ------------------------------------------------------------------

// std::mt19937 (the standard fixes the engine; only the seeding by an
// integer is used).
class StdMt19937 {
public:
    explicit StdMt19937(std::uint32_t seed = 5489u) { seed_with(seed); }
    void seed_with(std::uint32_t seed);
    std::uint32_t operator()();

private:
    std::uint32_t mt_[624];
    int mti_ = 624;
};

// std::generate_canonical<double, 53>: two 32-bit words.
double std_canonical(StdMt19937& g);
// std::uniform_real_distribution<double>(a, b)(g).
inline double std_uniform_real(StdMt19937& g, double a, double b) { return std_canonical(g) * (b - a) + a; }
// std::uniform_int_distribution<int or unsigned>(a, b)(g) for b - a < 2^32 - 1
// (Lemire's nearly divisionless method with 64-bit products).
std::int64_t std_uniform_int(StdMt19937& g, std::int64_t a, std::int64_t b);

// std::normal_distribution<double>: Marsaglia's polar method, which yields
// two deviates and keeps the second for the next call.
class StdNormal {
public:
    StdNormal(double mean = 0.0, double stddev = 1.0) : mean_(mean), stddev_(stddev) {}
    double operator()(StdMt19937& g);

private:
    double mean_, stddev_;
    double saved_ = 0.0;
    bool saved_available_ = false;
};

// std::shuffle(first, first + n, g) for a random-access range, with the
// two-positions-per-draw path of libstdc++ (n^2 < 2^32).
template <class T>
void std_shuffle(T* first, i64 n, StdMt19937& g);

// The positions std_shuffle swaps: for i = 1 .. n-1, element i is swapped
// with element pos[i - 1].
void std_shuffle_positions(i64 n, StdMt19937& g, std::vector<i64>& pos);

template <class T>
void std_shuffle(T* first, i64 n, StdMt19937& g) {
    if (n <= 0) return;
    std::vector<i64> pos;
    std_shuffle_positions(n, g, pos);
    for (i64 i = 1; i < n; ++i) std::swap(first[i], first[pos[static_cast<std::size_t>(i - 1)]]);
}

}  // namespace nlm
