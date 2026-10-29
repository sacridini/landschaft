// SPDX-License-Identifier: GPL-3.0-or-later
//
// Elementary functions with the bits R gives them on x86-64 Windows, where
// landscapemetrics' reference values come from. R's arithmetic runs through
// the mingw-w64 math library: log(), sqrt(), sin() and cos() are computed on the x87 unit
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
extern "C" double lsm_x87_log1p(double x);
extern "C" double lsm_x87_f2xm1(double x);
extern "C" double lsm_x87_sqrt(double x);
extern "C" double lsm_x87_sin(double x);
extern "C" double lsm_x87_cos(double x);
extern "C" double lsm_x87_atan2(double y, double x);
extern "C" double lsm_x87_exp(double x);
extern "C" double lsm_x87_pow_frac(double x, double y);
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

// R's log1p(x): mingw-w64's (fyl2xp1 when |x| < 0.29, else fyl2x of 1 + x,
// in extended precision), which R's nmath calls (ebd0, log1pmx, lgamma1p).
inline double r_log1p(double x) {
    if (!(x > -1) || !std::isfinite(x)) return std::log1p(x);
#if defined(LSM_X87_MASM)
    return lsm_x87_log1p(x);
#elif defined(LSM_X87_INLINE)
    detail::X87Precision precision;
    long double v = x;
    long double r;
    if (std::fabs(v) < static_cast<long double>(0.29)) {
        __asm__("fldln2\n\tfxch\n\tfyl2xp1" : "=t"(r) : "0"(v) : "st(1)");
    } else {
        v += 1.0L;
        __asm__("fldln2\n\tfxch\n\tfyl2x" : "=t"(r) : "0"(v) : "st(1)");
    }
    return static_cast<double>(r);
#else
    return std::log1p(x);
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

// R's sin(x) and cos(x): mingw-w64 runs fsin / fcos in extended precision
// (the same bits for landscapemetrics' circular plots).
inline double r_sin(double x) {
    if (!(std::fabs(x) < 9.2e18)) return std::sin(x);
#if defined(LSM_X87_MASM)
    return lsm_x87_sin(x);
#elif defined(LSM_X87_INLINE)
    detail::X87Precision precision;
    long double v = x;
    __asm__("fsin" : "+t"(v));
    return static_cast<double>(v);
#else
    return std::sin(x);
#endif
}

inline double r_cos(double x) {
    if (!(std::fabs(x) < 9.2e18)) return std::cos(x);
#if defined(LSM_X87_MASM)
    return lsm_x87_cos(x);
#elif defined(LSM_X87_INLINE)
    detail::X87Precision precision;
    long double v = x;
    __asm__("fcos" : "+t"(v));
    return static_cast<double>(v);
#else
    return std::cos(x);
#endif
}

// atan2(y, x) as mingw-w64 computes it (fpatan in extended precision),
// e.g. in terra's haversine distances.
inline double r_atan2(double y, double x) {
    if (!std::isfinite(y) || !std::isfinite(x)) return std::atan2(y, x);
#if defined(LSM_X87_MASM)
    return lsm_x87_atan2(y, x);
#elif defined(LSM_X87_INLINE)
    detail::X87Precision precision;
    long double r;
    const long double ly = y, lx = x;
    __asm__("fpatan" : "=t"(r) : "0"(lx), "u"(ly) : "st(1)");
    return static_cast<double>(r);
#else
    return std::atan2(y, x);
#endif
}

// hypot(x, y) as the UCRT's _hypot computes it (R's on Windows: mingw-w64's
// hypot calls it, and R's Mod() of a complex): sqrt(x * x + y * y) in double,
// rescaled by a power of two (exactly) where the squares would overflow or
// fall below the normal range. The same bits on every platform; glibc's
// hypot is closer to correctly rounded, hence different in the last bit.
inline double ucrt_hypot(double x, double y) {
    if (std::isinf(x) || std::isinf(y)) return std::numeric_limits<double>::infinity();
    if (std::isnan(x) || std::isnan(y)) return std::numeric_limits<double>::quiet_NaN();
    x = std::fabs(x);
    y = std::fabs(y);
    const double m = x > y ? x : y;
    const double s = x > y ? y : x;
    if (m == 0.0) return 0.0;
    constexpr double lo = 0x1p-500, hi = 0x1p+500;
    if (m > lo && m < hi && (s > lo || s == 0.0)) return std::sqrt(x * x + y * y);
    int e = 0;
    std::frexp(m, &e);
    const double xs = std::ldexp(x, -e), ys = std::ldexp(y, -e);
    return std::ldexp(std::sqrt(xs * xs + ys * ys), e);
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

// R's exp(x): mingw-w64 v11's exp (x87, exp.def.h): 2^f with f = x log2(e) -
// trunc(x log2(e)) from log2(e) split in two parts, then fscale.
inline double r_exp(double x) {
    if (std::isnan(x)) return x;
    if (std::isinf(x)) return x < 0 ? 0.0 : x;
    if (x == 0) return 1.0;
    if (x > 7.09782712893383996843E2) return std::numeric_limits<double>::infinity();
    if (x < -7.45133219101941108420E2) return 0.0;
#if defined(LSM_X87_MASM)
    return lsm_x87_exp(x);
#elif defined(LSM_X87_INLINE)
    detail::X87Precision precision;
    static const long double c0 = 1.44268798828125L;
    static const long double c1 = 7.05260771340735992468e-6L;
    long double res;
    unsigned short cw_saved, cw_chop;
    // mingw-w64's __expl_internal, with its AT&T operand order
    __asm__ volatile(
        "fldl2e\n\t"
        "fmul %%st(1),%%st\n\t"
        "fnstcw %[cws]\n\t"
        "movzwl %[cws], %%eax\n\t"
        "orb $12, %%ah\n\t"
        "movw %%ax, %[cwc]\n\t"
        "fldcw %[cwc]\n\t"
        "frndint\n\t"
        "fld %%st(1)\n\t"
        "frndint\n\t"
        "fldcw %[cws]\n\t"
        "fld %%st(1)\n\t"
        "fldt %[c0]\n\t"
        "fld %%st(2)\n\t"
        "fmul %%st(1),%%st\n\t"
        "fsubp %%st,%%st(2)\n\t"
        "fld %%st(4)\n\t"
        "fsub %%st(3),%%st\n\t"
        "fmulp %%st,%%st(1)\n\t"
        "faddp %%st,%%st(1)\n\t"
        "fldt %[c1]\n\t"
        "fmul %%st(4),%%st\n\t"
        "faddp %%st,%%st(1)\n\t"
        "f2xm1\n\t"
        "fld1\n\t"
        "faddp\n\t"
        "fstp %%st(1)\n\t"
        "fscale\n\t"
        "fstp %%st(1)\n\t"
        "fstp %%st(1)\n\t"
        : "=t"(res), [cws] "=m"(cw_saved), [cwc] "=m"(cw_chop)
        : "0"(static_cast<long double>(x)), [c0] "m"(c0), [c1] "m"(c1)
        : "ax");
    return static_cast<double>(res);
#else
    return std::exp(x);
#endif
}

// R's expm1(x): mingw-w64's (math/x86/expm1.def.h): f2xm1 of x / log(2) in
// extended precision when |x| < log(2), else exp(x) - 1 (r_exp).
inline double r_expm1(double x) {
    if (std::isnan(x) || x == 0) return x;
    if (std::isinf(x)) return x < 0 ? -1.0 : x;
    constexpr double kLn2 = 6.93147180559945309417E-1;
    if (std::fabs(x) < kLn2) {
        const double t = x / kLn2;
#if defined(LSM_X87_MASM)
        return lsm_x87_f2xm1(t);
#elif defined(LSM_X87_INLINE)
        detail::X87Precision precision;
        long double v = t;
        __asm__("f2xm1" : "+t"(v));
        return static_cast<double>(v);
#else
        return std::expm1(x);
#endif
    }
    return r_exp(x) - 1.0;
}

// R's x^y (R_pow: powl on 64-bit Windows) with mingw-w64 v11's powl:
// integer exponents by repeated squaring in extended precision (a
// reciprocal for negative ones), 0.5 by fsqrt, others exp2l(y log2l(|x|)).
inline double r_pow(double x, double y) {
    if (y == 2.0) return x * x;
    if (x == 1. || y == 0.) return 1.;
    if (x == 0.) {
        if (y > 0.) return 0.;
        if (y < 0.) return std::numeric_limits<double>::infinity();
        return y;
    }
    if (!std::isfinite(x) || !std::isfinite(y)) return std::pow(x, y);
    const double yi = std::trunc(y);
    if (y != yi) {
        if (x < 0) return -std::numeric_limits<double>::quiet_NaN();
        if (y == 0.5) return r_sqrt(x);
#if defined(LSM_X87_MASM)
        return lsm_x87_pow_frac(x, y);
#elif defined(LSM_X87_INLINE)
        detail::X87Precision precision;
        long double t;
        const long double lx = x;
        if (std::fabs(lx - 1.0L) <= static_cast<long double>(0.29)) {
            __asm__("fld1\n\tfxch\n\tfyl2xp1" : "=t"(t) : "0"(lx - 1.0L) : "st(1)");
        } else {
            __asm__("fld1\n\tfxch\n\tfyl2x" : "=t"(t) : "0"(lx) : "st(1)");
        }
        t *= static_cast<long double>(y);
        long double r;
        unsigned short cw_saved, cw_chop;
        __asm__ volatile(
            "fld %%st\n\t"
            "fnstcw %[cws]\n\t"
            "movzwl %[cws], %%eax\n\t"
            "orb $12, %%ah\n\t"
            "movw %%ax, %[cwc]\n\t"
            "fldcw %[cwc]\n\t"
            "frndint\n\t"
            "fldcw %[cws]\n\t"
            "fsubr %%st,%%st(1)\n\t"
            "fxch\n\t"
            "f2xm1\n\t"
            "fld1\n\t"
            "faddp\n\t"
            "fscale\n\t"
            "fstp %%st(1)\n\t"
            : "=t"(r), [cws] "=m"(cw_saved), [cwc] "=m"(cw_chop)
            : "0"(t)
            : "ax");
        return static_cast<double>(r);
#else
        return std::pow(x, y);
#endif
    }
    if (yi <= 2147483647.0 && yi >= -2147483648.0) {
        // __powil(x, (int) y)
        int n = static_cast<int>(yi);
        const bool odd = (n & 1) != 0;
        const bool recip = n < 0;
        unsigned int u = recip ? 0u - static_cast<unsigned int>(n) : static_cast<unsigned int>(n);
        Extended d = std::fabs(x);
        Extended r;
        if (u == 1) {
            r = d;
        } else {
            r = (u & 1) ? d : Extended(1.0);
            u >>= 1;
            do {
                d = d * d;
                if (u & 1) r = r * d;
                u >>= 1;
            } while (u > 0);
            if (recip && std::isinf(static_cast<double>(r)) && std::fabs(x) > 1.0) {
                // overflow before the reciprocal: start from 1 / |x|
                Extended e = Extended(1.0) / Extended(std::fabs(x));
                unsigned int v = 0u - static_cast<unsigned int>(n);
                r = (v & 1) ? e : Extended(1.0);
                v >>= 1;
                do {
                    e = e * e;
                    if (v & 1) r = r * e;
                    v >>= 1;
                } while (v > 0);
                double out = static_cast<double>(r);
                return (x < 0 && odd) ? -out : out;
            }
        }
        if (recip) r = Extended(1.0) / r;
        double out = static_cast<double>(r);
        return (x < 0 && odd) ? -out : out;
    }
    return std::pow(x, y);  // |y| >= 2^31: not reached by the models
}

}  // namespace lsm
