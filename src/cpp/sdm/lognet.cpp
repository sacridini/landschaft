// SPDX-License-Identifier: GPL-3.0-or-later
//
// glmnet's two-class logistic lasso path (glmnet 4.1-10, glmnetpp:
// ElnetDriver<binomial> -> ElnetPath -> ElnetPointInternal<binomial,
// two_class>) with maxnet's settings, reproducing R's bits on x86-64.
//
// Three things make the bits the same as R's (the solver stops at thresholds,
// so one last bit can flip a branch and move a coefficient by 1e-3):
//  * every scalar expression in glmnet's order, no FMA contraction;
//  * every reduction (dot products, sums) in the order of Eigen 3.4's
//    redux_impl with SSE2 packets of two doubles, which RcppEigen uses on
//    CRAN: four interleaved chains (index mod 4), combined pairwise, plus the
//    odd tail (`redux` below);
//  * R's exp() and log(), mingw-w64's x87 code (lsm::r_exp, lsm::r_log).
//
// What is faster than glmnet without changing a bit:
//  * the gradient checks (KKT) of the columns outside the strong set, ~60 %
//    of glmnet's time, only need the side of a threshold |r.x_k| is on. It
//    is decided from a bound when possible (|r.x_k| is within
//    ||r - r_ref|| ||x_k|| of its value at a reference residual), else from a
//    16-bit copy of the column with a rigorous error bound (a quarter of the
//    memory traffic), and only for the rare values within ~1e-5 of the
//    threshold computed exactly in glmnet's order; in parallel over columns,
//    four columns per sweep of the residual (glmnet: 434k dot products of
//    the whole column on the 10k x 2190 case, here 128k 16-bit ones and 119
//    exact ones);
//  * in the coordinate passes, the residual update of a coordinate is fused
//    with the dot products of the next ones, and the dot products of the
//    zero coordinates that follow are computed together (most stay zero, so
//    the residual they were computed with is still current); with AVX2 when
//    the processor has it (one 4-lane accumulator holds Eigen's two 2-lane
//    ones: the same additions in the same order);
//  * the IRLS step (linear predictor, x87 exp, weights, residual) runs in
//    parallel over rows and the weighted squares xv in parallel over columns.
// Every value is still computed by one thread in a fixed order, so the
// result does not depend on the number of threads.

#include "sdm/lognet.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__) || defined(__SSE2__)
#include <emmintrin.h>
#define SDM_LOGNET_SSE2 1
#endif
// AVX2 versions of the coordinate-pass kernels, chosen at run time: one
// 4-lane accumulator holds Eigen's two 2-lane ones (lanes 0-1 and 2-3), so
// the order of every addition is the same. Explicit multiply then add: no FMA.
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#define SDM_LOGNET_AVX2 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define SDM_AVX2_TARGET
#else
#define SDM_AVX2_TARGET __attribute__((target("avx2")))
#endif
#endif

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"

namespace sdm {

namespace {

using i64 = std::int64_t;

// -- Pairs of doubles: Eigen's SSE2 packets (elementwise operations, the
// same bits with or without SIMD) -------------------------------------------

#if defined(SDM_LOGNET_SSE2)
struct P2 {
    __m128d v;
};
inline P2 load(const double* p) { return {_mm_loadu_pd(p)}; }
inline void store(double* p, P2 a) { _mm_storeu_pd(p, a.v); }
inline P2 splat(double d) { return {_mm_set1_pd(d)}; }
inline P2 operator+(P2 a, P2 b) { return {_mm_add_pd(a.v, b.v)}; }
inline P2 operator-(P2 a, P2 b) { return {_mm_sub_pd(a.v, b.v)}; }
inline P2 operator*(P2 a, P2 b) { return {_mm_mul_pd(a.v, b.v)}; }
// Eigen's predux: lane 0 + lane 1
inline double hsum(P2 a) { return _mm_cvtsd_f64(a.v) + _mm_cvtsd_f64(_mm_unpackhi_pd(a.v, a.v)); }
#else
struct P2 {
    double a, b;
};
inline P2 load(const double* p) { return {p[0], p[1]}; }
inline void store(double* p, P2 x) {
    p[0] = x.a;
    p[1] = x.b;
}
inline P2 splat(double d) { return {d, d}; }
inline P2 operator+(P2 x, P2 y) { return {x.a + y.a, x.b + y.b}; }
inline P2 operator-(P2 x, P2 y) { return {x.a - y.a, x.b - y.b}; }
inline P2 operator*(P2 x, P2 y) { return {x.a * y.a, x.b * y.b}; }
inline double hsum(P2 x) { return x.a + x.b; }
#endif

// Eigen 3.4's sum of f(0..n-1) (redux_impl<sum, LinearVectorizedTraversal,
// NoUnrolling>, packet size 2, first aligned index 0): two packet
// accumulators over blocks of four, combined, the last full packet, the
// horizontal sum, then the odd element. `pk(i)` gives f(i), f(i + 1) as a
// packet, `sc(i)` f(i); each index is evaluated exactly once.
template <class Pk, class Sc>
inline double redux(i64 n, Pk&& pk, Sc&& sc) {
    const i64 n1 = n & ~i64(1), n2 = n & ~i64(3);
    if (n1 == 0) return n == 0 ? 0.0 : sc(0);
    P2 a = pk(0);
    if (n1 > 2) {
        P2 b = pk(2);
        for (i64 i = 4; i < n2; i += 4) {
            a = a + pk(i);
            b = b + pk(i + 2);
        }
        a = a + b;
        if (n1 > n2) a = a + pk(n2);
    }
    double res = hsum(a);
    for (i64 i = n1; i < n; ++i) res += sc(i);
    return res;
}

// a.dot(b)
inline double dot(const double* a, const double* b, i64 n) {
    return redux(
        n, [&](i64 i) { return load(a + i) * load(b + i); }, [&](i64 i) { return a[i] * b[i]; });
}

// v.dot(x.cwiseAbs2()): glmnet's xv
inline double dot_sq(const double* v, const double* x, i64 n) {
    return redux(
        n,
        [&](i64 i) {
            const P2 xi = load(x + i);
            return load(v + i) * (xi * xi);
        },
        [&](i64 i) { return v[i] * (x[i] * x[i]); });
}

// -- The residual, updated lazily ---------------------------------------------
//
// A change of coordinate k by diff makes glmnet update the residual,
// r -= diff * (v .* x_k), and a change d of the intercept r -= d * v. The
// update is kept pending and done in the same sweep as the next dot products
// with r (each element is updated before it is used, so the values are the
// same).

enum class Pending { None, Coef, Intercept };

struct Update {
    Pending kind = Pending::None;
    const double* x = nullptr;  // the column (Coef)
    double s = 0;               // diff or d
};

template <Pending U>
struct ResidualAccess {
    double* r;
    const double* v;
    const double* x;
    double s;
    P2 ps;
    ResidualAccess(double* r_, const double* v_, const Update& u) : r(r_), v(v_), x(u.x), s(u.s), ps(splat(u.s)) {}
    inline P2 pair(i64 i) const {
        P2 ri = load(r + i);
        if (U == Pending::Coef) {
            ri = ri - ps * (load(v + i) * load(x + i));
            store(r + i, ri);
        } else if (U == Pending::Intercept) {
            ri = ri - ps * load(v + i);
            store(r + i, ri);
        }
        return ri;
    }
    inline double one(i64 i) const {
        double ri = r[i];
        if (U == Pending::Coef) {
            ri = ri - s * (v[i] * x[i]);
            r[i] = ri;
        } else if (U == Pending::Intercept) {
            ri = ri - s * v[i];
            r[i] = ri;
        }
        return ri;
    }
};

// Applies the pending update and returns r.dot(x_c) for K columns, each in
// Eigen's order (the same structure as `redux`, K chains at once).
template <int K, Pending U>
void grads_impl(i64 n, double* r, const double* v, const Update& u, const double* const* xs, double* out) {
    const ResidualAccess<U> ra(r, v, u);
    const i64 n1 = n & ~i64(1), n2 = n & ~i64(3);
    if (n1 == 0) {
        if (n == 1) {
            const double ri = ra.one(0);
            for (int c = 0; c < K; ++c) out[c] = ri * xs[c][0];
        } else {
            for (int c = 0; c < K; ++c) out[c] = 0.0;
        }
        return;
    }
    constexpr int KK = K > 0 ? K : 1;
    P2 a[KK], b[KK];
    {
        const P2 r0 = ra.pair(0);
        for (int c = 0; c < K; ++c) a[c] = r0 * load(xs[c]);
    }
    if (n1 > 2) {
        const P2 r2 = ra.pair(2);
        for (int c = 0; c < K; ++c) b[c] = r2 * load(xs[c] + 2);
        for (i64 i = 4; i < n2; i += 4) {
            const P2 ri = ra.pair(i), rj = ra.pair(i + 2);
            for (int c = 0; c < K; ++c) {
                a[c] = a[c] + ri * load(xs[c] + i);
                b[c] = b[c] + rj * load(xs[c] + i + 2);
            }
        }
        for (int c = 0; c < K; ++c) a[c] = a[c] + b[c];
        if (n1 > n2) {
            const P2 ri = ra.pair(n2);
            for (int c = 0; c < K; ++c) a[c] = a[c] + ri * load(xs[c] + n2);
        }
    }
    double res[KK];
    for (int c = 0; c < K; ++c) res[c] = hsum(a[c]);
    for (i64 i = n1; i < n; ++i) {
        const double ri = ra.one(i);
        for (int c = 0; c < K; ++c) res[c] += ri * xs[c][i];
    }
    for (int c = 0; c < K; ++c) out[c] = res[c];
}

constexpr int kMaxBatch = 4;    // columns per sweep of the residual in the KKT checks
constexpr int kPassBatch = 4;   // zero coordinates whose dot products a pass computes together (8: no faster)

// sum_i r_i q_ik for K 16-bit columns (any order: only an approximation,
// whose rounding the callers' bounds cover)
template <int K>
void approx_dots_impl(i64 n, const double* r, const std::int16_t* const* qs, double* out) {
    i64 i = 0;
#if defined(SDM_LOGNET_SSE2)
    __m128d a[K], b[K];
    for (int c = 0; c < K; ++c) a[c] = b[c] = _mm_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        const __m128d r0 = _mm_loadu_pd(r + i), r1 = _mm_loadu_pd(r + i + 2);
        const __m128d r2 = _mm_loadu_pd(r + i + 4), r3 = _mm_loadu_pd(r + i + 6);
        for (int c = 0; c < K; ++c) {
            const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs[c] + i));
            const __m128i lo = _mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16);
            const __m128i hi = _mm_srai_epi32(_mm_unpackhi_epi16(v, v), 16);
            a[c] = _mm_add_pd(a[c], _mm_mul_pd(r0, _mm_cvtepi32_pd(lo)));
            b[c] = _mm_add_pd(b[c], _mm_mul_pd(r1, _mm_cvtepi32_pd(_mm_shuffle_epi32(lo, 0x4E))));
            a[c] = _mm_add_pd(a[c], _mm_mul_pd(r2, _mm_cvtepi32_pd(hi)));
            b[c] = _mm_add_pd(b[c], _mm_mul_pd(r3, _mm_cvtepi32_pd(_mm_shuffle_epi32(hi, 0x4E))));
        }
    }
    for (int c = 0; c < K; ++c) out[c] = hsum({_mm_add_pd(a[c], b[c])});
#else
    for (int c = 0; c < K; ++c) out[c] = 0.0;
#endif
    for (; i < n; ++i)
        for (int c = 0; c < K; ++c) out[c] += r[i] * static_cast<double>(qs[c][i]);
}


#if defined(SDM_LOGNET_AVX2)
bool cpu_has_avx2() {
#if defined(_MSC_VER) && !defined(__clang__)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    if ((r[2] & (1 << 27)) == 0 || (r[2] & (1 << 28)) == 0) return false;  // OSXSAVE, AVX
    if ((_xgetbv(0) & 0x6) != 0x6) return false;  // the OS saves the YMM registers
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#endif
}

const bool kAvx2 = cpu_has_avx2() && std::getenv("LANDSCHAFT_NO_AVX2") == nullptr;

// The residual at i..i+3 after the pending update (stored back), into `dst`.
#define SDM_LOGNET_R4(i, dst)                                                                                      \
    do {                                                                                                           \
        __m256d rv_ = _mm256_loadu_pd(r + (i));                                                                    \
        if (U == Pending::Coef) {                                                                                  \
            rv_ = _mm256_sub_pd(rv_, _mm256_mul_pd(s4, _mm256_mul_pd(_mm256_loadu_pd(v + (i)),                     \
                                                                     _mm256_loadu_pd(xu + (i)))));                 \
            _mm256_storeu_pd(r + (i), rv_);                                                                        \
        } else if (U == Pending::Intercept) {                                                                      \
            rv_ = _mm256_sub_pd(rv_, _mm256_mul_pd(s4, _mm256_loadu_pd(v + (i))));                                 \
            _mm256_storeu_pd(r + (i), rv_);                                                                        \
        }                                                                                                          \
        dst = rv_;                                                                                                 \
    } while (0)

// grads_impl with 4-lane vectors (n >= 4).
template <int K, Pending U>
SDM_AVX2_TARGET void grads_avx2(i64 n, double* r, const double* v, const Update& u, const double* const* xs,
                                double* out) {
    const i64 n1 = n & ~i64(1), n2 = n & ~i64(3);
    const double s = u.s;
    const double* xu = u.x;
    const __m256d s4 = _mm256_set1_pd(s);
    constexpr int KK = K > 0 ? K : 1;
    __m256d acc[KK];
    {
        __m256d r0;
        SDM_LOGNET_R4(0, r0);
        for (int c = 0; c < K; ++c) acc[c] = _mm256_mul_pd(r0, _mm256_loadu_pd(xs[c]));
    }
    for (i64 i = 4; i < n2; i += 4) {
        __m256d ri;
        SDM_LOGNET_R4(i, ri);
        for (int c = 0; c < K; ++c) acc[c] = _mm256_add_pd(acc[c], _mm256_mul_pd(ri, _mm256_loadu_pd(xs[c] + i)));
    }
    __m128d a[KK];
    for (int c = 0; c < K; ++c) a[c] = _mm_add_pd(_mm256_castpd256_pd128(acc[c]), _mm256_extractf128_pd(acc[c], 1));
    if (n1 > n2) {
        __m128d r2 = _mm_loadu_pd(r + n2);
        if (U == Pending::Coef) {
            r2 = _mm_sub_pd(r2, _mm_mul_pd(_mm_set1_pd(s), _mm_mul_pd(_mm_loadu_pd(v + n2), _mm_loadu_pd(xu + n2))));
            _mm_storeu_pd(r + n2, r2);
        } else if (U == Pending::Intercept) {
            r2 = _mm_sub_pd(r2, _mm_mul_pd(_mm_set1_pd(s), _mm_loadu_pd(v + n2)));
            _mm_storeu_pd(r + n2, r2);
        }
        for (int c = 0; c < K; ++c) a[c] = _mm_add_pd(a[c], _mm_mul_pd(r2, _mm_loadu_pd(xs[c] + n2)));
    }
    double res[KK];
    for (int c = 0; c < K; ++c) res[c] = _mm_cvtsd_f64(a[c]) + _mm_cvtsd_f64(_mm_unpackhi_pd(a[c], a[c]));
    for (i64 i = n1; i < n; ++i) {
        double ri = r[i];
        if (U == Pending::Coef) {
            ri = ri - s * (v[i] * xu[i]);
            r[i] = ri;
        } else if (U == Pending::Intercept) {
            ri = ri - s * v[i];
            r[i] = ri;
        }
        for (int c = 0; c < K; ++c) res[c] += ri * xs[c][i];
    }
    for (int c = 0; c < K; ++c) out[c] = res[c];
}
#undef SDM_LOGNET_R4

template <int K>
SDM_AVX2_TARGET void approx_dots_avx2(i64 n, const double* r, const std::int16_t* const* qs, double* out) {
    __m256d a[K], b[K];
    for (int c = 0; c < K; ++c) a[c] = b[c] = _mm256_setzero_pd();
    i64 i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256d r0 = _mm256_loadu_pd(r + i), r1 = _mm256_loadu_pd(r + i + 4);
        for (int c = 0; c < K; ++c) {
            const __m256i q = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(qs[c] + i)));
            a[c] = _mm256_add_pd(a[c], _mm256_mul_pd(r0, _mm256_cvtepi32_pd(_mm256_castsi256_si128(q))));
            b[c] = _mm256_add_pd(b[c], _mm256_mul_pd(r1, _mm256_cvtepi32_pd(_mm256_extracti128_si256(q, 1))));
        }
    }
    for (int c = 0; c < K; ++c) {
        const __m256d t = _mm256_add_pd(a[c], b[c]);
        const __m128d h = _mm_add_pd(_mm256_castpd256_pd128(t), _mm256_extractf128_pd(t, 1));
        out[c] = _mm_cvtsd_f64(h) + _mm_cvtsd_f64(_mm_unpackhi_pd(h, h));
    }
    for (; i < n; ++i)
        for (int c = 0; c < K; ++c) out[c] += r[i] * static_cast<double>(qs[c][i]);
}
#endif

template <int K>
void approx_dots_one(i64 n, const double* r, const std::int16_t* const* qs, double* out) {
#if defined(SDM_LOGNET_AVX2)
    if (kAvx2) {
        approx_dots_avx2<K>(n, r, qs, out);
        return;
    }
#endif
    approx_dots_impl<K>(n, r, qs, out);
}

void approx_dots(int k, i64 n, const double* r, const std::int16_t* const* qs, double* out) {
    switch (k) {
        case 1: approx_dots_one<1>(n, r, qs, out); break;
        case 2: approx_dots_one<2>(n, r, qs, out); break;
        case 3: approx_dots_one<3>(n, r, qs, out); break;
        default: approx_dots_one<4>(n, r, qs, out); break;
    }
}

template <int K, Pending U>
inline void grads_one(i64 n, double* r, const double* v, const Update& u, const double* const* xs, double* out) {
#if defined(SDM_LOGNET_AVX2)
    if (kAvx2 && n >= 4) {
        grads_avx2<K, U>(n, r, v, u, xs, out);
        return;
    }
#endif
    grads_impl<K, U>(n, r, v, u, xs, out);
}

template <Pending U>
void grads_k(int k, i64 n, double* r, const double* v, const Update& u, const double* const* xs, double* out) {
    switch (k) {
        case 0: grads_one<0, U>(n, r, v, u, xs, out); break;
        case 1: grads_one<1, U>(n, r, v, u, xs, out); break;
        case 2: grads_one<2, U>(n, r, v, u, xs, out); break;
        case 3: grads_one<3, U>(n, r, v, u, xs, out); break;
        default: grads_one<4, U>(n, r, v, u, xs, out); break;
    }
}

// The pending update (cleared), then r.dot(x_c) for the k <= 4 columns xs.
void grads(int k, i64 n, double* r, const double* v, Update& u, const double* const* xs, double* out) {
    switch (u.kind) {
        case Pending::None: grads_k<Pending::None>(k, n, r, v, u, xs, out); break;
        case Pending::Coef: grads_k<Pending::Coef>(k, n, r, v, u, xs, out); break;
        case Pending::Intercept: grads_k<Pending::Intercept>(k, n, r, v, u, xs, out); break;
    }
    u.kind = Pending::None;
}

// The pending update (cleared), then r.sum().
template <Pending U>
double sum_after_impl(i64 n, double* r, const double* v, const Update& u) {
    const ResidualAccess<U> ra(r, v, u);
    return redux(
        n, [&](i64 i) { return ra.pair(i); }, [&](i64 i) { return ra.one(i); });
}

double sum_after(i64 n, double* r, const double* v, Update& u) {
    double s = 0;
    switch (u.kind) {
        case Pending::None: s = sum_after_impl<Pending::None>(n, r, v, u); break;
        case Pending::Coef: s = sum_after_impl<Pending::Coef>(n, r, v, u); break;
        case Pending::Intercept: s = sum_after_impl<Pending::Intercept>(n, r, v, u); break;
    }
    u.kind = Pending::None;
    return s;
}

void apply(i64 n, double* r, const double* v, Update& u) {
    if (u.kind != Pending::None) grads(0, n, r, v, u, nullptr, nullptr);
}

struct MaxitReached {};

// Time and count of the solver's stages, printed to stderr when
// LANDSCHAFT_PROFILE is set.
struct Profile {
    using Clock = std::chrono::steady_clock;
    enum Stage { kSetup, kScreen, kXv, kPass, kIrls, kKktNorms, kCount };
    bool on = std::getenv("LANDSCHAFT_PROFILE") != nullptr;
    double seconds[kCount] = {};
    i64 dots_approx = 0, dots_exact = 0, dots_pass = 0, residual_updates = 0, irls = 0;
    Clock::time_point t0 = Clock::now();
    void lap(Stage s) {
        if (!on) return;
        const Clock::time_point t = Clock::now();
        seconds[s] += std::chrono::duration<double>(t - t0).count();
        t0 = t;
    }
    ~Profile() {
        if (!on) return;
        std::fprintf(stderr,
                     "lognet: setup %.3f s, strong/KKT dot products %.3f s (%lld 16-bit, %lld exact), xv %.3f s, "
                     "passes %.3f s (%lld dot products, %lld residual updates), IRLS %.3f s (%lld), KKT norms %.3f s\n",
                     seconds[kSetup], seconds[kScreen], static_cast<long long>(dots_approx),
                     static_cast<long long>(dots_exact), seconds[kXv], seconds[kPass],
                     static_cast<long long>(dots_pass), static_cast<long long>(residual_updates), seconds[kIrls],
                     static_cast<long long>(irls), seconds[kKktNorms]);
    }
};

}  // namespace

LognetResult lognet_path(double* x, i64 n, i64 ni, const double* y01, const double* weights, const double* penalty,
                         const double* lambda, int n_lambda, double thresh, i64 maxit, double pmin) {
    const double big = 9.9e35;
    Profile prof;
    LognetResult out;
    if (n < 1 || ni < 1) throw std::invalid_argument("glmnet: no observations or no variables");
    auto col = [&](i64 k) { return x + k * n; };

    // -- driver (elnet_driver/binomial.hpp, base.hpp) --
    // penalty factors: vq = max(vp, 0) * ni / sum(vq)
    std::vector<double> vq(penalty, penalty + ni);
    {
        double mx = vq[0];
        for (double v : vq) mx = std::max(mx, v);
        if (!(mx > 0)) throw std::invalid_argument("glmnet: all penalty factors are <= 0 (error 10000)");
        for (double& v : vq) v = std::max(v, 0.0);
        const double f = static_cast<double>(ni) / redux(
            ni, [&](i64 i) { return load(vq.data() + i); }, [&](i64 i) { return vq[i]; });
        for (double& v : vq) v *= f;
    }
    // ju: columns that are not constant (chkvars)
    std::vector<char> ju(static_cast<std::size_t>(ni));
    const bool par_cols = n * ni >= 200000;
    lsm::parallel_for(par_cols, ni, [&](i64 j) {
        const double* xj = col(j);
        const double t = xj[0];
        bool any = false;
        for (i64 i = 1; i < n && !any; ++i) any = xj[i] != t;
        ju[static_cast<std::size_t>(j)] = any;
    });
    // y = (w (p == 1), w (p == 0)) per row, divided by its row sum: y_i = p_i,
    // ww_i = w_i; then the weights sum to one
    std::vector<double> w(static_cast<std::size_t>(n)), y(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) {
        double c0 = (y01[i] == 1.0 ? 1.0 : 0.0) * weights[i];
        const double c1 = (y01[i] == 1.0 ? 0.0 : 1.0) * weights[i];
        const double ww = c0 + c1;
        if (ww != 0) c0 /= ww;
        w[i] = ww;
        y[i] = c0;
    }
    const double sw = redux(
        n, [&](i64 i) { return load(w.data() + i); }, [&](i64 i) { return w[i]; });
    for (double& v : w) v /= sw;
    // LStandardize1 (isd = 0, intr = 1): x centred on the weighted means
    lsm::parallel_for(par_cols, ni, [&](i64 j) {
        if (!ju[static_cast<std::size_t>(j)]) return;
        double* xj = col(j);
        const double xm = dot(w.data(), xj, n);
        for (i64 i = 0; i < n; ++i) xj[i] -= xm;
    });

    // -- point solver construction (internal/binomial_base.hpp) --
    std::vector<double> b(static_cast<std::size_t>(ni + 1), 0.0), bs(static_cast<std::size_t>(ni + 1), 0.0);
    std::vector<double> xv(static_cast<std::size_t>(ni), 0.0), ga(static_cast<std::size_t>(ni), 0.0);
    std::vector<double> q(static_cast<std::size_t>(n)), v(static_cast<std::size_t>(n)), r(static_cast<std::size_t>(n));
    std::vector<i64> ia;  // active set, in order of entry
    ia.reserve(static_cast<std::size_t>(std::min<i64>(ni, 1024)));
    std::vector<char> active(static_cast<std::size_t>(ni), 0), strong(static_cast<std::size_t>(ni), 0);
    std::vector<i64> strong_list;  // the strong set, ascending
    const double fmax = lsm::r_log(1.0 / pmin - 1.0), fmin = -fmax;
    const double vmin = (1.0 + pmin) * pmin * (1.0 - pmin);
    const double q0 = dot(w.data(), y.data(), n);
    if (q0 <= pmin) throw std::invalid_argument("glmnet: the proportion of presences is below pmin (error 8001)");
    if (q0 >= 1.0 - pmin)
        throw std::invalid_argument("glmnet: the proportion of background points is below pmin (error 9001)");
    const double bz = lsm::r_log(q0 / (1.0 - q0));
    const double vi = q0 * (1.0 - q0);
    b[0] = bz;
    for (i64 i = 0; i < n; ++i) {
        v[i] = vi * w[i];
        r[i] = w[i] * (y[i] - q0);
        q[i] = q0;
    }
    double xmz = vi;
    const double dev1 = -(bz * q0 + lsm::r_log(1.0 - q0));
    // dev0 adds w y log(y) and w (1 - y) log(1 - y), zero for a 0/1 response
    const double dev0 = dev1 + 0.0;
    const double thr = thresh * dev0;

    // -- gradients of the columns outside the strong set, decided lazily --
    // ga[k] = |rk.x_k| is used only in the comparisons ga[k] > t of the
    // strong rule and of the KKT check, always at the residual rk of the last
    // check, so it is enough to know on which side of t it is:
    //  1. from a bound without touching x_k: with a reference residual rref,
    //     |rk.x_k| <= |rref.x_k| + ||rk - rref|| ||x_k||;
    //  2. else from a 16-bit copy of the column, x~_k = off_k + s_k q_k with
    //     ||x_k - x~_k|| <= qerr_k: |rk.x~_k| is within ||rk|| qerr_k of it;
    //  3. else exactly, in glmnet's order (rare: values within ~1e-5 of t).
    // Every bound also covers the rounding errors of both dot products.
    std::vector<double> rk(r), rref(r);
    std::vector<double> xnorm(static_cast<std::size_t>(ni), 0.0), gref(static_cast<std::size_t>(ni), 0.0);
    std::vector<double> grerr(static_cast<std::size_t>(ni), 0.0), gerr(static_cast<std::size_t>(ni), 0.0);
    std::vector<char> fresh(static_cast<std::size_t>(ni), 0);  // ga[k] (within gerr[k]) is |rk.x_k|
    std::vector<std::int16_t> xq(static_cast<std::size_t>(n * ni));
    std::vector<double> qoff(static_cast<std::size_t>(ni), 0.0), qscale(static_cast<std::size_t>(ni), 0.0);
    std::vector<double> qerr(static_cast<std::size_t>(ni), 0.0), qnorm(static_cast<std::size_t>(ni), 0.0);
    // the 16-bit columns: q = round((x - off) / s), s = range / 65534; qerr
    // also covers the rounding of x - (off + s q) (by the eps_rel terms)
    lsm::parallel_for(par_cols, ni, [&](i64 j) {
        const auto ju_ = static_cast<std::size_t>(j);
        if (!ju[ju_]) return;
        const double* xj = col(j);
        double mn = xj[0], mx = xj[0], s2 = 0;
        for (i64 i = 0; i < n; ++i) {
            mn = std::min(mn, xj[i]);
            mx = std::max(mx, xj[i]);
            s2 += xj[i] * xj[i];
        }
        xnorm[ju_] = std::sqrt(s2);
        const double off = 0.5 * mn + 0.5 * mx;
        double sc = (mx - mn) / 65534.0;
        if (!(sc > 0)) sc = 1.0;
        std::int16_t* qj = xq.data() + j * n;
        double e2 = 0, q2 = 0;
        for (i64 i = 0; i < n; ++i) {
            const double qi = std::max(-32767.0, std::min(32767.0, std::nearbyint((xj[i] - off) / sc)));
            qj[i] = static_cast<std::int16_t>(qi);
            const double e = xj[i] - (off + sc * qi);
            e2 += e * e;
            q2 += qi * qi;
        }
        qoff[ju_] = off;
        qscale[ju_] = sc;
        qerr[ju_] = std::sqrt(e2) * (1.0 + 1e-6);
        qnorm[ju_] = std::fabs(off) * std::sqrt(static_cast<double>(n)) + sc * std::sqrt(q2);
    });
    auto residual_stats = [&](double& nrm, double& sum) {
        double s2 = 0, s1 = 0;
        for (i64 i = 0; i < n; ++i) {
            s2 += rk[i] * rk[i];
            s1 += rk[i];
        }
        nrm = std::sqrt(s2);
        sum = s1;
    };
    double rk_norm = 0, rk_sum = 0;
    residual_stats(rk_norm, rk_sum);
    double rref_norm = rk_norm, dr = 0.0;
    // relative size of the rounding errors of the dot products and the norms
    const double eps_rel = 1e-9 + 1e-15 * static_cast<double>(n);
    std::vector<i64> todo;
    const bool par_kkt = n >= 2048;
    // ga[k] ~ |rk.x~_k|, within gerr[k], for the columns in `todo`
    auto approx_grads = [&]() {
        const i64 m = static_cast<i64>(todo.size());
        prof.dots_approx += m;
        const i64 nb = (m + kMaxBatch - 1) / kMaxBatch;
        lsm::parallel_for_dynamic(par_kkt && nb > 1, nb, [&](i64 bi) {
            const i64 lo = bi * kMaxBatch, cnt = std::min<i64>(kMaxBatch, m - lo);
            const std::int16_t* qs[kMaxBatch];
            double g[kMaxBatch];
            for (i64 c = 0; c < cnt; ++c) qs[c] = xq.data() + todo[static_cast<std::size_t>(lo + c)] * n;
            approx_dots(static_cast<int>(cnt), n, rk.data(), qs, g);
            for (i64 c = 0; c < cnt; ++c) {
                const auto k = static_cast<std::size_t>(todo[static_cast<std::size_t>(lo + c)]);
                ga[k] = std::fabs(qoff[k] * rk_sum + qscale[k] * g[c]);
                gerr[k] = rk_norm * (qerr[k] + eps_rel * (xnorm[k] + qnorm[k]));
                fresh[k] = 1;
            }
        });
    };
    // ga[k] = |rk.x_k| exactly (gerr[k] = 0) for the columns in `todo`
    auto exact_grads = [&]() {
        const i64 m = static_cast<i64>(todo.size());
        prof.dots_exact += m;
        const i64 nb = (m + kMaxBatch - 1) / kMaxBatch;
        lsm::parallel_for_dynamic(par_kkt && nb > 1, nb, [&](i64 bi) {
            const i64 lo = bi * kMaxBatch, cnt = std::min<i64>(kMaxBatch, m - lo);
            const double* xs[kMaxBatch];
            double g[kMaxBatch];
            for (i64 c = 0; c < cnt; ++c) xs[c] = col(todo[static_cast<std::size_t>(lo + c)]);
            Update none;
            grads(static_cast<int>(cnt), n, rk.data(), nullptr, none, xs, g);
            for (i64 c = 0; c < cnt; ++c) {
                const auto k = static_cast<std::size_t>(todo[static_cast<std::size_t>(lo + c)]);
                ga[k] = std::fabs(g[c]);
                gerr[k] = 0.0;
                fresh[k] = 1;
            }
        });
    };
    // For every non-strong column: strong[k] |= ga[k] > scale * vq[k]; true if any was added
    auto screen = [&](double scale) {
        todo.clear();
        for (i64 k = 0; k < ni; ++k) {
            const auto ku = static_cast<std::size_t>(k);
            if (strong[ku] || !ju[ku] || fresh[ku]) continue;
            const double t = scale * vq[ku];
            const double bound =
                gref[ku] + grerr[ku] + dr * xnorm[ku] + eps_rel * (rk_norm + rref_norm + dr) * xnorm[ku];
            if (!(bound <= t)) todo.push_back(k);
        }
        approx_grads();
        todo.clear();
        for (i64 k = 0; k < ni; ++k) {
            const auto ku = static_cast<std::size_t>(k);
            if (strong[ku] || !ju[ku] || !fresh[ku] || gerr[ku] == 0.0) continue;
            const double t = scale * vq[ku];
            if (!(ga[ku] - gerr[ku] > t) && !(ga[ku] + gerr[ku] <= t)) todo.push_back(k);
        }
        exact_grads();
        bool added = false;
        for (i64 k = 0; k < ni; ++k) {
            const auto ku = static_cast<std::size_t>(k);
            if (strong[ku] || !ju[ku] || !fresh[ku]) continue;
            if (ga[ku] - gerr[ku] > scale * vq[ku]) {
                strong[ku] = 1;
                added = true;
            }
        }
        if (added) {
            strong_list.clear();
            for (i64 k = 0; k < ni; ++k)
                if (strong[static_cast<std::size_t>(k)]) strong_list.push_back(k);
        }
        return added;
    };
    // the reference gradients at rref = rk, for every non-strong column
    auto new_reference = [&]() {
        rref = rk;
        rref_norm = rk_norm;
        dr = 0.0;
        todo.clear();
        for (i64 k = 0; k < ni; ++k)
            if (!strong[static_cast<std::size_t>(k)] && ju[static_cast<std::size_t>(k)]) todo.push_back(k);
        approx_grads();
        for (const i64 k : todo) {
            gref[static_cast<std::size_t>(k)] = ga[static_cast<std::size_t>(k)];
            grerr[static_cast<std::size_t>(k)] = gerr[static_cast<std::size_t>(k)];
        }
    };
    new_reference();  // glmnet's initial gradients

    prof.lap(Profile::kSetup);
    i64 nlp = 0;
    double dlx = 0.0;
    const bool par_rows = n >= 8192;

    // One coordinate-descent pass (elnet_point/base.hpp) over `cols`
    // (the strong set, ascending, or the active set, in order of entry).
    // Returns true when converged.
    Update pending;
    auto pass = [&](const std::vector<i64>& cols, bool full, double ab, double dem) {
        ++nlp;
        dlx = 0.0;
        const i64 m = static_cast<i64>(cols.size());
        i64 pos = 0;
        while (pos < m) {
            // a coordinate at zero usually stays there and leaves r unchanged:
            // the dot products of a run of them are computed together
            int nb = 1;
            if (full && b[static_cast<std::size_t>(cols[static_cast<std::size_t>(pos)] + 1)] == 0.0) {
                while (nb < kPassBatch && pos + nb < m &&
                       b[static_cast<std::size_t>(cols[static_cast<std::size_t>(pos + nb)] + 1)] == 0.0)
                    ++nb;
            }
            const double* xs[kPassBatch];
            double g[kPassBatch];
            for (int c = 0; c < nb; ++c) xs[c] = col(cols[static_cast<std::size_t>(pos + c)]);
            grads(nb, n, r.data(), v.data(), pending, xs, g);
            prof.dots_pass += nb;
            int t = 0;
            for (; t < nb; ++t) {
                const i64 k = cols[static_cast<std::size_t>(pos + t)];
                const auto ku = static_cast<std::size_t>(k);
                const double old = b[ku + 1];
                const double u = g[t] + old * xv[ku];
                const double vv = std::fabs(u) - vq[ku] * ab;
                double an = 0.0;
                if (vv > 0.0) an = std::max(-big, std::min(big, std::copysign(vv, u) / (xv[ku] + vq[ku] * dem)));
                b[ku + 1] = an;
                if (old == an) continue;
                if (full && !active[ku]) {
                    active[ku] = 1;
                    ia.push_back(k);
                }
                const double diff = an - old;
                dlx = std::max(xv[ku] * diff * diff, dlx);
                ++prof.residual_updates;
                pending.kind = Pending::Coef;
                pending.x = col(k);
                pending.s = diff;
                break;  // r changes: the next dot products need it
            }
            pos += t < nb ? t + 1 : nb;
        }
        // the intercept (gaussian_base.hpp update_intercept)
        const double d = sum_after(n, r.data(), v.data(), pending) / xmz;
        if (d != 0.0) {
            b[0] += d;
            dlx = std::max(xmz * d * d, dlx);
            pending.kind = Pending::Intercept;
            pending.s = d;
        }
        if (dlx < thr) return true;
        if (nlp > maxit) throw MaxitReached{};
        return false;
    };

    std::vector<double> f(static_cast<std::size_t>(n));
    out.beta.assign(static_cast<std::size_t>(ni), 0.0);
    double lmda_curr = 0.0;
    try {
        for (int m = 0; m < n_lambda; ++m) {
            const double alm0 = lmda_curr, alm = lambda[m];
            lmda_curr = alm;
            const double ab = alm * 1.0, dem = alm * (1.0 - 1.0);
            // sequential strong rule
            const double tlam = 1.0 * (2.0 * alm - alm0);
            prof.lap(Profile::kPass);
            screen(tlam);
            prof.lap(Profile::kScreen);
            while (true) {  // IRLS
                if (nlp > maxit) throw MaxitReached{};
                bs[0] = b[0];
                for (const i64 k : ia) bs[static_cast<std::size_t>(k + 1)] = b[static_cast<std::size_t>(k + 1)];
                {
                    const i64 ns = static_cast<i64>(strong_list.size());
                    lsm::parallel_for_dynamic(ns * n >= 200000 && ns > 1, ns, [&](i64 s) {
                        const i64 k = strong_list[static_cast<std::size_t>(s)];
                        xv[static_cast<std::size_t>(k)] = dot_sq(v.data(), col(k), n);
                    });
                }
                prof.lap(Profile::kXv);
                // weighted least squares: full passes until the active set is stable
                while (true) {
                    if (pass(strong_list, true, ab, dem)) break;
                    while (!pass(ia, false, ab, dem)) {
                    }
                }
                apply(n, r.data(), v.data(), pending);
                prof.lap(Profile::kPass);
                ++prof.irls;
                // IRLS update (two_class.hpp update_irls)
                const i64 nblocks = (n + 1023) / 1024;
                const i64 na = static_cast<i64>(ia.size());
                lsm::parallel_for(par_rows && nblocks > 1, nblocks, [&](i64 bi) {
                    const i64 lo = bi * 1024, hi = std::min(n, lo + 1024);
                    const double b0 = b[0] + 0.0;
                    for (i64 i = lo; i < hi; ++i) f[i] = b0;
                    for (i64 l = 0; l < na; ++l) {
                        const i64 k = ia[static_cast<std::size_t>(l)];
                        const double c = b[static_cast<std::size_t>(k + 1)];
                        const double* xk = col(k);
                        for (i64 i = lo; i < hi; ++i) f[i] += c * xk[i];
                    }
                    for (i64 i = lo; i < hi; ++i) {
                        const double fi = f[i];
                        const double qi = fi < fmin ? 0.0 : (fi > fmax ? 1.0 : 1.0 / (1.0 + lsm::r_exp(-fi)));
                        q[i] = qi;
                        f[i] = w[i] * qi * (1.0 - qi);  // the new v
                    }
                });
                const double xmz_new = redux(
                    n, [&](i64 i) { return load(f.data() + i); }, [&](i64 i) { return f[i]; });
                v.swap(f);
                xmz = xmz_new;
                if (xmz <= vmin) break;  // total variance too small
                for (i64 i = 0; i < n; ++i) r[i] = w[i] * (y[i] - q[i]);
                // convergence (with the new xmz, the xv of this iteration)
                const double d0 = b[0] - bs[0];
                bool ix = xmz * d0 * d0 > thr;
                if (!ix) {
                    for (const i64 k : ia) {
                        const auto ku = static_cast<std::size_t>(k);
                        const double d = b[ku + 1] - bs[ku + 1];
                        if (xv[ku] * d * d > thr) {
                            ix = true;
                            break;
                        }
                    }
                }
                prof.lap(Profile::kIrls);
                if (ix) continue;
                // KKT check of the columns outside the strong set, at this r
                std::copy(r.begin(), r.end(), rk.begin());
                residual_stats(rk_norm, rk_sum);
                {
                    double sd = 0;
                    for (i64 i = 0; i < n; ++i) {
                        const double di = rk[i] - rref[i];
                        sd += di * di;
                    }
                    dr = std::sqrt(sd);
                }
                for (i64 k = 0; k < ni; ++k)
                    if (!strong[static_cast<std::size_t>(k)]) fresh[static_cast<std::size_t>(k)] = 0;
                prof.lap(Profile::kKktNorms);
                // a new reference when the bound has become loose (0.05: the fewest 16-bit dot products)
                if (dr > 0.05 * rref_norm) new_reference();
                const bool added = screen(ab);
                prof.lap(Profile::kScreen);
                if (!added) break;
            }
            for (const i64 k : ia) out.beta[static_cast<std::size_t>(k)] = b[static_cast<std::size_t>(k + 1)];
            out.n_lambda = m + 1;
        }
    } catch (const MaxitReached&) {
    }
    out.n_passes = nlp;
    if (prof.on)
        std::fprintf(stderr, "lognet: %lld strong columns, %lld active, %lld passes\n",
                     static_cast<long long>(strong_list.size()), static_cast<long long>(ia.size()),
                     static_cast<long long>(nlp));
    return out;
}

}  // namespace sdm
