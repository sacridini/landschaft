// SPDX-License-Identifier: GPL-3.0-or-later
//
// Cephes' normal distribution as scipy.special has it, numpy's pairwise sum and
// numpy's default generator (see special.hpp).

#include "stats/special.hpp"

#include <cmath>
#include <limits>
#include <utility>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace stats {

namespace {

template <std::size_t N>
double polevl(double x, const double (&c)[N]) {
    double a = c[0];
    for (std::size_t i = 1; i < N; ++i) a = a * x + c[i];
    return a;
}

template <std::size_t N>
double p1evl(double x, const double (&c)[N]) {
    double a = x + c[0];
    for (std::size_t i = 1; i < N; ++i) a = a * x + c[i];
    return a;
}

}  // namespace

double ndtri(double y0) {
    static const double P0[] = {-5.99633501014107895267E1, 9.80010754185999661536E1, -5.66762857469070293439E1,
                                1.39312609387279679503E1, -1.23916583867381258016E0};
    static const double Q0[] = {1.95448858338141759834E0,  4.67627912898881538453E0,  8.63602421390890590575E1,
                                -2.25462687854119370527E2, 2.00260212380060660359E2,  -8.20372256168333339912E1,
                                1.59056225126211695515E1,  -1.18331621121330003142E0};
    static const double P1[] = {4.05544892305962419923E0,  3.15251094599893866154E1,  5.71628192246421288162E1,
                                4.40805073893200834700E1,  1.46849561928858024014E1,  2.18663306850790267539E0,
                                -1.40256079171354495875E-1, -3.50424626827848203418E-2, -8.57456785154685413611E-4};
    static const double Q1[] = {1.57799883256466749731E1,  4.53907635128879210584E1,  4.13172038254672030440E1,
                                1.50425385692907503408E1,  2.50464946208309415979E0,  -1.42182922854787788574E-1,
                                -3.80806407691578277194E-2, -9.33259480895457427372E-4};
    static const double P2[] = {3.23774891776946035970E0, 6.91522889068984211695E0, 3.93881025292474443415E0,
                                1.33303460815807542389E0, 2.01485389549179081538E-1, 1.23716634817820021358E-2,
                                3.01581553508235416007E-4, 2.65806974686737550832E-6, 6.23974539184983293730E-9};
    static const double Q2[] = {6.02427039364742014255E0, 3.67983563856160859403E0, 1.37702099489081330271E0,
                                2.16236993594496635890E-1, 1.34204006088543189037E-2, 3.28014464682127739104E-4,
                                2.89247864745380683936E-6, 6.79019408009981274425E-9};
    const double s2pi = 2.50662827463100050242E0;
    const double expm2 = 0.13533528323661269189;
    if (y0 == 0.0) return -std::numeric_limits<double>::infinity();
    if (y0 == 1.0) return std::numeric_limits<double>::infinity();
    if (!(y0 >= 0.0 && y0 <= 1.0)) return std::numeric_limits<double>::quiet_NaN();
    double y = y0;
    bool code = true;
    if (y > 1.0 - expm2) {
        y = 1.0 - y;
        code = false;
    }
    if (y > expm2) {
        y -= 0.5;
        const double y2 = y * y;
        const double x = y + y * (y2 * polevl(y2, P0) / p1evl(y2, Q0));
        return x * s2pi;
    }
    const double x = std::sqrt(-2.0 * std::log(y));
    const double x0 = x - std::log(x) / x;
    const double z = 1.0 / x;
    const double x1 = x < 8.0 ? z * polevl(z, P1) / p1evl(z, Q1) : z * polevl(z, P2) / p1evl(z, Q2);
    const double r = x0 - x1;
    return code ? -r : r;
}

double erf(double x) {
    static const double T[] = {9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3,
                               7.00332514112805075473E3, 5.55923013010394962768E4};
    static const double U[] = {3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3,
                               2.26290000613890934246E4, 4.92673942608635921086E4};
    if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
    if (std::fabs(x) > 1.0) return 1.0 - erfc(x);
    const double z = x * x;
    return x * polevl(z, T) / p1evl(z, U);
}

double erfc(double a) {
    static const double P[] = {2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0,
                               4.86371970985681366614E1,  1.96520832956077098242E2, 5.26445194995477358631E2,
                               9.34528527171957607540E2,  1.02755188689515710272E3, 5.57535335369399327526E2};
    static const double Q[] = {1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2,
                               9.75708501743205489753E2, 1.82390916687909736289E3, 2.24633760818710981792E3,
                               1.65666309194161350182E3, 5.57535340817727675546E2};
    static const double R[] = {5.64189583547755073984E-1, 1.27536670759978104416E0, 5.01905042251180477414E0,
                               6.16021097993053585195E0,  7.40974269950448939160E0, 2.97886665372100240670E0};
    static const double S[] = {2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1,
                               1.70814450747565897222E1, 9.60896809063285878198E0, 3.36907645100081516050E0};
    const double maxlog = 7.09782712893383996843E2;
    if (std::isnan(a)) return std::numeric_limits<double>::quiet_NaN();
    const double x = std::fabs(a);
    if (x < 1.0) return 1.0 - erf(a);
    double z = -a * a;
    const double under = a < 0.0 ? 2.0 : 0.0;
    if (z < -maxlog) return under;
    z = std::exp(z);
    double p, q;
    if (x < 8.0) {
        p = polevl(x, P);
        q = p1evl(x, Q);
    } else {
        p = polevl(x, R);
        q = p1evl(x, S);
    }
    double y = (z * p) / q;
    if (a < 0.0) y = 2.0 - y;
    return y == 0.0 ? under : y;
}

double ndtr(double a) {
    if (std::isnan(a)) return std::numeric_limits<double>::quiet_NaN();
    const double frac_1_sqrt_2 = 0.70710678118654752440;
    const double x = a * frac_1_sqrt_2;
    const double z = std::fabs(x);
    if (z < frac_1_sqrt_2) return 0.5 + 0.5 * erf(x);
    const double y = 0.5 * erfc(z);
    return x > 0.0 ? 1.0 - y : y;
}

namespace {

double pairwise(const double* a, i64 n) {
    if (n < 8) {
        double res = 0.0;
        for (i64 i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        i64 i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    i64 n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

// 64 x 64 -> 128 bits
inline void mul64(u64 a, u64 b, u64& hi, u64& lo) {
#if defined(__SIZEOF_INT128__)
    const unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
    hi = static_cast<u64>(p >> 64);
    lo = static_cast<u64>(p);
#elif defined(_MSC_VER) && defined(_M_X64)
    lo = _umul128(a, b, &hi);
#else
    const u64 a0 = a & 0xFFFFFFFFu, a1 = a >> 32, b0 = b & 0xFFFFFFFFu, b1 = b >> 32;
    const u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const u64 mid = (p00 >> 32) + (p01 & 0xFFFFFFFFu) + (p10 & 0xFFFFFFFFu);
    lo = (p00 & 0xFFFFFFFFu) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
}

// numpy's SeedSequence(entropy).generate_state(4, uint64), as 8 words of 32 bits.
void seed_sequence(u64 entropy, u32 out[8]) {
    const u32 init_a = 0x43b0d7e5u, mult_a = 0x931e8875u, init_b = 0x8b51f9ddu, mult_b = 0x58f38dedu;
    const u32 mix_l = 0xca01f9ddu, mix_r = 0x4973f715u;
    std::vector<u32> words;
    for (u64 e = entropy; e > 0; e >>= 32) words.push_back(static_cast<u32>(e));
    if (words.empty()) words.push_back(0);
    u32 hash_const = init_a;
    auto hashmix = [&](u32 value) {
        u32 v = value ^ hash_const;
        hash_const *= mult_a;
        v *= hash_const;
        return v ^ (v >> 16);
    };
    auto mix = [&](u32 x, u32 y) {
        const u32 r = mix_l * x - mix_r * y;
        return r ^ (r >> 16);
    };
    u32 pool[4];
    for (int i = 0; i < 4; ++i) pool[i] = hashmix(i < static_cast<int>(words.size()) ? words[i] : 0u);
    for (int src = 0; src < 4; ++src)
        for (int dst = 0; dst < 4; ++dst)
            if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src]));
    for (std::size_t w = 4; w < words.size(); ++w)
        for (u32& p : pool) p = mix(p, hashmix(words[w]));
    u32 hc = init_b;
    for (int i = 0; i < 8; ++i) {
        u32 v = pool[i % 4] ^ hc;
        hc *= mult_b;
        v *= hc;
        out[i] = v ^ (v >> 16);
    }
}

const u64 kMultHi = 2549297995355413924ull, kMultLo = 4865540595714422341ull;

}  // namespace

double np_sum(const double* a, i64 n) { return 0.0 + pairwise(a, n); }

NumpyRng::NumpyRng(u64 seed) {
    u32 w[8];
    seed_sequence(seed, w);
    auto word = [&](int i) { return (static_cast<u64>(w[2 * i + 1]) << 32) | w[2 * i]; };
    const u64 init_hi = word(0), init_lo = word(1), seq_hi = word(2), seq_lo = word(3);
    // inc = (initseq << 1) | 1
    inc_hi_ = (seq_hi << 1) | (seq_lo >> 63);
    inc_lo_ = (seq_lo << 1) | 1u;
    state_hi_ = state_lo_ = 0;
    step();
    // state += initstate
    const u64 lo = state_lo_ + init_lo;
    state_hi_ = state_hi_ + init_hi + (lo < state_lo_ ? 1u : 0u);
    state_lo_ = lo;
    step();
}

void NumpyRng::step() {
    // state = state * mult + inc (mod 2^128)
    u64 hi, lo;
    mul64(state_lo_, kMultLo, hi, lo);
    hi += state_lo_ * kMultHi + state_hi_ * kMultLo;
    const u64 l2 = lo + inc_lo_;
    hi += inc_hi_ + (l2 < lo ? 1u : 0u);
    state_hi_ = hi;
    state_lo_ = l2;
}

u64 NumpyRng::next_u64() {
    step();
    const u64 x = state_hi_ ^ state_lo_;
    const unsigned rot = static_cast<unsigned>(state_hi_ >> 58);
    return rot == 0 ? x : (x >> rot) | (x << (64 - rot));
}

u32 NumpyRng::next_u32() {
    if (has_buffer_) {
        has_buffer_ = false;
        return buffer_;
    }
    const u64 next = next_u64();
    buffer_ = static_cast<u32>(next >> 32);
    has_buffer_ = true;
    return static_cast<u32>(next);
}

u64 NumpyRng::integers(u64 n) {
    const u64 rng = n - 1;
    if (rng == 0) return 0;
    if (rng <= 0xFFFFFFFFull) {
        if (rng == 0xFFFFFFFFull) return next_u32();
        const u32 excl = static_cast<u32>(rng) + 1u;
        u64 m = static_cast<u64>(next_u32()) * excl;
        u32 leftover = static_cast<u32>(m);
        if (leftover < excl) {
            const u32 threshold = (0xFFFFFFFFu - static_cast<u32>(rng)) % excl;
            while (leftover < threshold) {
                m = static_cast<u64>(next_u32()) * excl;
                leftover = static_cast<u32>(m);
            }
        }
        return m >> 32;
    }
    if (rng == ~0ull) return next_u64();
    const u64 excl = rng + 1;
    u64 hi, lo;
    mul64(next_u64(), excl, hi, lo);
    if (lo < excl) {
        const u64 threshold = (~0ull - rng) % excl;
        while (lo < threshold) mul64(next_u64(), excl, hi, lo);
    }
    return hi;
}

u64 NumpyRng::interval(u64 max) {
    if (max == 0) return 0;
    u64 mask = max;
    for (int s : {1, 2, 4, 8, 16, 32}) mask |= mask >> s;
    for (;;) {
        const u64 v = max <= 0xFFFFFFFFull ? (static_cast<u64>(next_u32()) & mask) : (next_u64() & mask);
        if (v <= max) return v;
    }
}

double NumpyRng::random() { return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0); }

std::vector<u64> NumpyRng::choice_distinct(u64 pop, u64 size) {
    std::vector<u64> idx;
    if (pop > 10000 && size > pop / 50) {
        idx.resize(pop);
        for (u64 i = 0; i < pop; ++i) idx[i] = i;
        const u64 first = pop - size > 1 ? pop - size : 1;
        for (u64 i = pop - 1; i >= first; --i) {
            const u64 j = integers(i + 1);
            std::swap(idx[i], idx[j]);
            if (i == 0) break;
        }
        idx.erase(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(pop - size));
        return idx;
    }
    idx.assign(size, 0);
    u64 mask = static_cast<u64>(1.2 * static_cast<double>(size));
    for (int s : {1, 2, 4, 8, 16, 32}) mask |= mask >> s;
    std::vector<u64> set(mask + 1, ~0ull);
    for (u64 j = pop - size; j < pop; ++j) {
        const u64 val = integers(j + 1);
        u64 loc = val & mask;
        while (set[loc] != ~0ull && set[loc] != val) loc = (loc + 1) & mask;
        const u64 k = j + size - pop;
        if (set[loc] == ~0ull) {
            set[loc] = val;
            idx[k] = val;
        } else {
            loc = j & mask;
            while (set[loc] != ~0ull) loc = (loc + 1) & mask;
            set[loc] = j;
            idx[k] = j;
        }
    }
    for (u64 i = size; i-- > 1;) {
        const u64 j = integers(i + 1);
        std::swap(idx[i], idx[j]);
    }
    return idx;
}

}  // namespace stats
