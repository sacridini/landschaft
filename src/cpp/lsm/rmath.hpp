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
#include <limits>

#include "lsm/extended.hpp"

#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#define LSM_X87_MASM 1
extern "C" double lsm_x87_log(double x);
extern "C" double lsm_x87_sqrt(double x);
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

// R's x^-1: powl(x, -1), the reciprocal rounded to extended, then to double.
inline double r_reciprocal(double x) { return static_cast<double>(Extended(1.0) / Extended(x)); }

}  // namespace lsm
