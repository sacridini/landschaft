// SPDX-License-Identifier: GPL-3.0-or-later
//
// Elementary functions with the bits R gives them on x86-64 Windows, where
// landscapemetrics' reference values come from. R's arithmetic runs through
// the mingw-w64 math library: log() and sqrt() are computed on the x87 unit
// in extended precision and then rounded to double (so they can differ in
// the last bit from a correctly rounded result or from MSVC's library), and
// x^y with y other than 2 is powl(). On other processors the standard
// functions are used.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include "lsm/extended.hpp"

#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#define LSM_X87_MASM 1
extern "C" double lsm_x87_log(double x);
extern "C" double lsm_x87_sqrt(double x);
extern "C" double lsm_x87_sum(const double* x, std::int64_t n);
extern "C" void lsm_x87_accumulate(const double* x, std::int64_t n, void* acc);
extern "C" double lsm_x87_to_double(const void* acc);
extern "C" double lsm_x87_mean(const double* x, std::int64_t n);
extern "C" double lsm_x87_ssd(const double* x, std::int64_t n, double mean, std::int64_t divisor);
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define LSM_X87_INLINE 1
#endif

namespace lsm {

#if defined(LSM_X87_INLINE)
namespace detail {
struct X87Precision {
    unsigned short saved;
    X87Precision() {
        unsigned short cw;
        __asm__ volatile("fnstcw %0" : "=m"(saved));
        cw = static_cast<unsigned short>((saved & 0xF0FF) | 0x0300);
        __asm__ volatile("fldcw %0" : : "m"(cw));
    }
    ~X87Precision() { __asm__ volatile("fldcw %0" : : "m"(saved)); }
};
}  // namespace detail
#endif

// R's log(x) (glibc's x87 __ieee754_log: fyl2xp1 near 1, fyl2x elsewhere).
inline double r_log(double x) {
    if (!(x > 0) || !std::isfinite(x)) return std::log(x);
#if defined(LSM_X87_MASM)
    return lsm_x87_log(x);
#elif defined(LSM_X87_INLINE)
    detail::X87Precision precision;
    long double v = x;
    long double r;
    const long double xm1 = v - 1.0L;
    if (std::fabs(xm1) <= static_cast<long double>(0.29)) {
        __asm__("fldln2\n\tfxch\n\tfyl2xp1" : "=t"(r) : "0"(xm1) : "st(1)");
    } else {
        __asm__("fldln2\n\tfxch\n\tfyl2x" : "=t"(r) : "0"(v) : "st(1)");
    }
    return static_cast<double>(r);
#else
    return std::log(x);
#endif
}

// R's sqrt() of a vector (x87 fsqrt in extended precision, rounded to
// double). R's byte-code interpreter computes the sqrt of a single number
// inline instead, correctly rounded: r_sqrt(x, true), or std::sqrt.
inline double r_sqrt(double x, bool scalar = false) {
    if (scalar) return std::sqrt(x);
    if (!(x >= 0) || !std::isfinite(x)) return std::sqrt(x);
#if defined(LSM_X87_MASM)
    return lsm_x87_sqrt(x);
#elif defined(LSM_X87_INLINE)
    detail::X87Precision precision;
    long double v = x;
    __asm__("fsqrt" : "+t"(v));
    return static_cast<double>(v);
#else
    return std::sqrt(x);
#endif
}

// -- R's sum(), mean() and sd() (80-bit accumulation) ------------------------

using i64 = std::int64_t;

// R's mean(): sum, divide, then add the mean residual, all in R's LDOUBLE.
inline double r_mean(const double* x, i64 n) {
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
#if defined(LSM_X87_MASM)
    return lsm_x87_mean(x, n);
#endif
    Extended s = 0.0;
    for (i64 i = 0; i < n; ++i) s += x[i];
    s /= Extended(n);
    if (std::isfinite(static_cast<double>(s))) {
        Extended t = 0.0;
        for (i64 i = 0; i < n; ++i) t += Extended(x[i]) - s;
        s += t / Extended(n);
    }
    return static_cast<double>(s);
}

// R's sd(): sqrt(var()), var as R's cov.c (mean rounded to double, then the
// sum of squared deviations in LDOUBLE); NA for fewer than 2 values. `mean`
// is r_mean(x, n).
// With `population`, the standard deviation over n instead of n - 1
// (FRAGSTATS' and pylandstats'; 0 for one value).
inline double r_sd(const double* x, i64 n, double mean, bool population = false) {
    if (n < (population ? 1 : 2)) return std::numeric_limits<double>::quiet_NaN();
#if defined(LSM_X87_MASM)
    return r_sqrt(lsm_x87_ssd(x, n, mean, population ? n : n - 1), true);
#endif
    const Extended m = mean;
    Extended ss = 0.0;
    for (i64 i = 0; i < n; ++i) {
        const Extended d = Extended(x[i]) - m;
        ss += d * d;
    }
    return r_sqrt(static_cast<double>(ss / Extended(population ? n : n - 1)), true);  // sqrt(var(x)): one number
}

// R's sum() of doubles.
inline double r_sum(const double* x, i64 n) {
#if defined(LSM_X87_MASM)
    return lsm_x87_sum(x, n);
#endif
    Extended s = 0.0;
    for (i64 i = 0; i < n; ++i) s += x[i];
    return static_cast<double>(s);
}

// R's sum() of f(i) for i in [begin, end), NaN skipped when na_rm.
template <class F>
inline double sum_of(i64 begin, i64 end, F&& f, bool na_rm = false) {
#if defined(LSM_X87_MASM)
    // the terms in blocks, added on the x87 unit
    alignas(16) unsigned char acc[16] = {};  // an 80-bit +0
    double buf[256];
    int m = 0;
    for (i64 i = begin; i < end; ++i) {
        const double v = f(i);
        if (na_rm && std::isnan(v)) continue;
        buf[m++] = v;
        if (m == 256) {
            lsm_x87_accumulate(buf, m, acc);
            m = 0;
        }
    }
    lsm_x87_accumulate(buf, m, acc);
    return lsm_x87_to_double(acc);
#endif
    Extended s = 0.0;
    for (i64 i = begin; i < end; ++i) {
        const double v = f(i);
        if (na_rm && std::isnan(v)) continue;
        s += v;
    }
    return static_cast<double>(s);
}

// R's x^-1: powl(x, -1), the reciprocal rounded to extended, then to double.
inline double r_reciprocal(double x) { return static_cast<double>(Extended(1.0) / Extended(x)); }

}  // namespace lsm
