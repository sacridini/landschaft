// SPDX-License-Identifier: GPL-3.0-or-later
//
// R's as.character(<double>): the text by which R groups numbers in
// as.factor(), merge() keys and paste(). Ported from R 4.4.2 (GPL >= 2):
//   src/main/coerce.c      coerceToString(): R_print.digits = DBL_DIG (15)
//   src/main/printutils.c  StringFromReal(), EncodeRealDrop0()
//   src/main/format.c      scientific(), formatReal()
//   src/extra/trio/trio.c  TrioWriteDouble(): R for Windows prints with trio
//                          (trioremap.h), not with the C library
//
// Both steps are done in R's 80-bit long double, which is not correctly
// rounded decimal conversion:
// - scientific() scales |x| to 15 integer digits in long double and rounds
//   with nearbyintl(); a scaled value that lands on a half rounds to even and
//   can drop a digit ("9.5577270258218" for 9.557727025821805, where "%.15g"
//   gives "9.55772702582181"). Its table of powers of ten holds double
//   literals (1e23 is not 10^23).
// - trio makes the digits with long double floor / fmod and R's own
//   round-half-even fudges, so near a tie it can round the other way from the
//   C library, and past 19 digits it prints noise (91837940248660738058 for
//   91837940248660738048).
// The long double operations are done with lsm::Extended (the x87 format,
// emulated with correct rounding where the compiler has no such type) and the
// library functions are replaced by exact equivalents: floorl, fmodl and the
// truncating casts are exact anyway, powl(10, n) is mingw-w64's __powil
// (repeated squaring), and log10l is only ever floored, which is done exactly
// (it can differ from the x87 result only for an x within ~1e-17 of a power
// of ten, which prints as "1e+k" either way).

#include "sdm/rformat.hpp"

#include <array>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "lsm/extended.hpp"

namespace sdm {

namespace {

using lsm::Extended;

constexpr int kDigits = 15;  // R_print.digits = DBL_DIG in coerceToString()
constexpr int kKpMax = 27;   // KP_MAX with long doubles
constexpr int kScipen = 0;   // options(scipen = 0)

// -- long double helpers -----------------------------------------------------------

// a - b as a double: its sign is exact (no comparison here is between values
// closer than the smallest normal double without being equal).
double diff(const Extended& a, const Extended& b) { return static_cast<double>(a - b); }
bool lt(const Extended& a, const Extended& b) { return diff(a, b) < 0.0; }
bool gt(const Extended& a, const Extended& b) { return diff(a, b) > 0.0; }
bool ne(const Extended& a, const Extended& b) { return diff(a, b) != 0.0; }

// floorl(v): exact. With hi = (double) v, v - floor(hi) is exact in 64 bits and
// is itself a double.
Extended floorl_(const Extended& v) {
    const double hi = static_cast<double>(v);
    if (!std::isfinite(hi)) return v;
    const double c = std::floor(hi);
    const double r = static_cast<double>(v - Extended(c));
    return Extended(c) + Extended(std::floor(r));
}

// fmodl(v, 10) for v >= 0: exact.
Extended fmod10(const Extended& v) {
    const double hi = static_cast<double>(v);
    if (hi >= 9223372036854775808.0) {  // >= 2^63: an integer, split into two doubles
        const double lo = static_cast<double>(v - Extended(hi));
        double r = std::fmod(hi, 10.0) + std::fmod(lo, 10.0);  // integers: exact
        r = std::fmod(r, 10.0);
        if (r < 0) r += 10.0;
        return Extended(r);
    }
    // v - 10 floor(v / 10), corrected if the rounded quotient crossed an integer
    const Extended q = floorl_(v / Extended(10.0));
    Extended r = v - q * Extended(10.0);
    if (static_cast<double>(r) < 0.0) r = r + Extended(10.0);
    if (static_cast<double>(r) >= 10.0) r = r - Extended(10.0);
    return r;
}

// (int) v: truncation; out of range gives the x87 "integer indefinite" INT_MIN.
int to_int(const Extended& v) {
    const bool neg = static_cast<double>(v) < 0.0;
    const Extended t = neg ? Extended(0.0) - floorl_(Extended(0.0) - v) : floorl_(v);
    const double tv = static_cast<double>(t);  // an integer; exact when in range
    if (!(tv >= -2147483648.0 && tv <= 2147483647.0)) return INT_MIN;
    return static_cast<int>(tv);
}

// powl(10, n), n integer, as mingw-w64's powl computes it (__powil): repeated
// squaring in extended precision, the reciprocal for n < 0.
Extended powl10(int n) {
    unsigned int u = n < 0 ? 0u - static_cast<unsigned int>(n) : static_cast<unsigned int>(n);
    Extended d = 10.0;
    Extended r;
    if (u == 0) {
        r = 1.0;
    } else if (u == 1) {
        r = d;
    } else {
        r = (u & 1) ? d : Extended(1.0);
        u >>= 1;
        do {
            d = d * d;
            if (u & 1) r = r * d;
            u >>= 1;
        } while (u > 0);
    }
    return n < 0 ? Extended(1.0) / r : r;
}

// format.c's tbl[]: the DOUBLE literals 1e00 .. 1e27 in a long double array.
const std::array<Extended, kKpMax + 1>& format_tbl() {
    static const std::array<Extended, kKpMax + 1> tbl = [] {
        static const double lit[kKpMax + 1] = {1e00, 1e01, 1e02, 1e03, 1e04, 1e05, 1e06, 1e07, 1e08, 1e09,
                                               1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
                                               1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27};
        std::array<Extended, kKpMax + 1> t;
        for (int i = 0; i <= kKpMax; ++i) t[static_cast<std::size_t>(i)] = Extended(lit[i]);
        return t;
    }();
    return tbl;
}

// Exact 10^0 .. 10^27 (all fit the 64-bit mantissa).
const std::array<Extended, 28>& exact_pow10() {
    static const std::array<Extended, 28> t = [] {
        std::array<Extended, 28> p;
        p[0] = Extended(1.0);
        for (std::size_t i = 1; i < p.size(); ++i) p[i] = p[i - 1] * Extended(10.0);
        return p;
    }();
    return t;
}

// Sign of x - 10^k for a finite double x > 0, exact (multi-word integers).
int cmp_pow10(double x, int k) {
    using Big = std::vector<std::uint32_t>;  // little-endian words
    auto mul_small = [](Big& a, std::uint32_t f) {
        std::uint64_t carry = 0;
        for (auto& w : a) {
            const std::uint64_t t = static_cast<std::uint64_t>(w) * f + carry;
            w = static_cast<std::uint32_t>(t);
            carry = t >> 32;
        }
        if (carry) a.push_back(static_cast<std::uint32_t>(carry));
    };
    auto shl = [](Big& a, int bits) {
        const int words = bits / 32, rem = bits % 32;
        if (rem) {
            std::uint32_t carry = 0;
            for (auto& w : a) {
                const std::uint32_t nw = (w << rem) | carry;
                carry = w >> (32 - rem);
                w = nw;
            }
            if (carry) a.push_back(carry);
        }
        a.insert(a.begin(), static_cast<std::size_t>(words), 0u);
    };
    auto cmp = [](Big a, Big b) {
        while (!a.empty() && a.back() == 0) a.pop_back();
        while (!b.empty() && b.back() == 0) b.pop_back();
        if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
        for (std::size_t i = a.size(); i-- > 0;)
            if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
        return 0;
    };
    int e2;
    const double f = std::frexp(x, &e2);  // x = f 2^e2, 0.5 <= f < 1
    const std::uint64_t m = static_cast<std::uint64_t>(std::ldexp(f, 53));
    const int e = e2 - 53;  // x = m 2^e
    Big a{static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(m >> 32)};
    Big b{1u};
    const int ak = k < 0 ? -k : k;
    if (k >= 0) {  // m 2^e  vs  5^k 2^k
        for (int i = 0; i < ak; ++i) mul_small(b, 5);
        if (e - k >= 0)
            shl(a, e - k);
        else
            shl(b, k - e);
    } else {  // m 5^|k| 2^(e + |k|)  vs  1
        for (int i = 0; i < ak; ++i) mul_small(a, 5);
        if (e + ak >= 0)
            shl(a, e + ak);
        else
            shl(b, -(e + ak));
    }
    return cmp(a, b);
}

// floor(log10(x)) of a finite double x > 0, exact.
int floor_log10(double x) {
    const double l = std::log10(x);
    int k = static_cast<int>(std::floor(l));
    const double frac = l - std::floor(l);
    if (frac > 1e-9 && frac < 1 - 1e-9) return k;
    while (cmp_pow10(x, k) < 0) --k;
    while (cmp_pow10(x, k + 1) >= 0) ++k;
    return k;
}

// floor(log10(v)) of an integer-valued 1 <= v < 1e27, exact.
int floor_log10_int(const Extended& v) {
    const auto& p = exact_pow10();
    int k = 0;
    while (k + 1 < static_cast<int>(p.size()) && !lt(v, p[static_cast<std::size_t>(k) + 1])) ++k;
    return k;
}

// trio's TrioPower(10, n): 10 * 1E(n-1)L for 0 <= n <= 9, else powl.
Extended trio_power10(int n) {
    if (n == 0) return Extended(10.0) * (Extended(1.0) / Extended(10.0));  // 10 * 1E-1L
    if (n >= 1 && n <= 9) return exact_pow10()[static_cast<std::size_t>(n)];  // 10 * 1E+(n-1)L: exact
    return powl10(n);
}

// -- format.c ------------------------------------------------------------------------

// nearbyintl(v) (round half to even) of an extended 0 <= v < 2^52, as a double.
double nearbyint_ext(const Extended& v) {
    const Extended fl = floorl_(v);
    const double c = static_cast<double>(fl);  // < 2^52: exact
    const double f = static_cast<double>(v - fl);
    if (f > 0.5 || (f == 0.5 && std::fmod(c, 2.0) != 0.0)) return c + 1.0;
    return c;
}

struct Scientific {
    int neg = 0;
    int kpower = 0;  // |x| = alpha 10^kpower, 1 <= alpha < 10
    int nsig = 1;    // significant digits, at most 15
    bool roundingwidens = false;
};

// scientific(), the HAVE_LONG_DOUBLE branch with HAVE_POWL.
Scientific scientific(double x) {
    Scientific s;
    if (x == 0.0) return s;
    double r;
    if (x < 0.0) {
        s.neg = 1;
        r = -x;
    } else {
        r = x;
    }
    const auto& tbl = format_tbl();
    int kp = static_cast<int>(std::floor(std::log10(r))) - kDigits + 1;
    Extended r_prec = r;
    if (std::abs(kp) <= kKpMax) {
        if (kp > 0)
            r_prec = r_prec / tbl[static_cast<std::size_t>(kp)];
        else if (kp < 0)
            r_prec = r_prec * tbl[static_cast<std::size_t>(-kp)];
    } else {
        r_prec = r_prec / powl10(kp);
    }
    if (lt(r_prec, tbl[kDigits - 1])) {
        r_prec = r_prec * Extended(10.0);
        --kp;
    }
    double alpha = nearbyint_ext(r_prec);
    s.nsig = kDigits;
    for (int j = 1; j <= kDigits; ++j) {
        alpha /= 10.0;
        if (alpha == std::floor(alpha))
            --s.nsig;
        else
            break;
    }
    if (s.nsig == 0) {
        s.nsig = 1;
        kp += 1;
    }
    s.kpower = kp + kDigits - 1;
    int rgt = kDigits - s.kpower;
    rgt = rgt < 0 ? 0 : rgt > kKpMax ? kKpMax : rgt;
    const double fuzz = 0.5 / static_cast<double>(tbl[static_cast<std::size_t>(rgt)]);
    s.roundingwidens = s.kpower > 0 && s.kpower <= kKpMax &&
                       lt(Extended(r), tbl[static_cast<std::size_t>(s.kpower)] - Extended(fuzz));
    return s;
}

// -- trio.c ----------------------------------------------------------------------------

// TrioWriteDouble() for "%w.de" / "%#w.de" (e_format) and "%w.df": base 10,
// a double argument (epsilon = DBL_EPSILON), no flags but '#'.
std::string trio_write_double(double x, bool e_format, bool alternative, int width, int precision) {
    const Extended epsilon = DBL_EPSILON;
    const Extended dblBase = 10.0;
    const bool isNegative = std::signbit(x);
    Extended number = isNegative ? -x : x;
    const double absx = isNegative ? -x : x;

    int exponent = 0;
    unsigned int uExponent = 0;
    bool isExponentNegative = false;
    Extended fAdjust = 0.5;
    if (e_format) {
        if (absx == 0.0) {  // log10l(0) = -Inf
            exponent = 0;
        } else {
            exponent = floor_log10(absx);
            number = number * powl10(-exponent);  // A * 10^-B
            isExponentNegative = exponent < 0;
            uExponent = isExponentNegative ? 0u - static_cast<unsigned int>(exponent) : static_cast<unsigned int>(exponent);
        }
    }

    Extended integerNumber = floorl_(number);
    Extended fractionNumber = number - integerNumber;
    int integerDigits = 1;
    if (gt(integerNumber, epsilon)) integerDigits += floor_log10_int(integerNumber);
    int fractionDigits = precision;
    Extended dblFractionBase = trio_power10(fractionDigits);

    // This where rounding up occurs (R's modification: round half to even)
    Extended workNumber = fmod10(number * dblFractionBase);
    if (to_int(workNumber) % 2 == 0) fAdjust = Extended(0.5) * (Extended(1.0) - Extended(5.0) * epsilon);
    workNumber = number + fAdjust / dblFractionBase;

    if (ne(floorl_(number), floorl_(workNumber))) {
        if (e_format) {
            const int workDigits = 1 + floor_log10_int(floorl_(workNumber));
            if (integerDigits == workDigits) {
                // Adjust if the same number of digits are used
                number = number + Extended(0.5) / dblFractionBase;
                integerNumber = floorl_(number);
                fractionNumber = number - integerNumber;
            } else {
                // Adjust if number was rounded up one digit (ie. 0.99 to 1.00)
                exponent++;
                isExponentNegative = exponent < 0;
                uExponent = isExponentNegative ? 0u - static_cast<unsigned int>(exponent) : static_cast<unsigned int>(exponent);
                workNumber = (number + Extended(0.5) / dblFractionBase) / dblBase;
                integerNumber = floorl_(workNumber);
                fractionNumber = workNumber - integerNumber;
            }
        } else {
            if (gt(workNumber, Extended(1.0))) {
                // Adjust if number was rounded up one digit (ie. 99 to 100)
                integerNumber = floorl_(workNumber);
                fractionNumber = 0.0;
                integerDigits = gt(integerNumber, epsilon) ? 1 + floor_log10_int(integerNumber) : 1;
            } else {
                integerNumber = floorl_(workNumber);
                fractionNumber = workNumber - integerNumber;
            }
        }
    }

    const Extended integerAdjust = 0.5;
    Extended fractionAdjust = fAdjust / dblFractionBase;
    const bool keepDecimalPoint = alternative || precision != 0;  // trailing zeroes are kept

    int expectedWidth = integerDigits + fractionDigits;
    if (keepDecimalPoint) expectedWidth += 1;
    if (isNegative) expectedWidth += 1;
    int exponentDigits = 0;
    if (e_format) {
        // ceil(log10(uExponent + 1)): the number of digits
        exponentDigits = 1;
        for (unsigned int t = uExponent; t >= 10; t /= 10) ++exponentDigits;
    }
    const bool requireTwoDigitExponent = exponentDigits == 1;
    if (exponentDigits > 0) expectedWidth += exponentDigits + (requireTwoDigitExponent ? 3 : 2);

    std::string out;
    for (int i = expectedWidth; i < width; i++) out += ' ';
    if (isNegative) out += '-';

    // the integer part
    for (int i = 0; i < integerDigits; i++) {
        const Extended p = trio_power10(integerDigits - i - 1);
        if (i == integerDigits - 1 && fractionDigits == 0) {  // integerAdjust == 0.5: round half to even
            const int ii = to_int(floorl_(integerNumber / p));
            if (ii % 2 == 0)
                workNumber = floorl_((integerNumber + integerAdjust - Extended(1e-7)) / p);
            else
                workNumber = floorl_((integerNumber + integerAdjust) / p);
        } else {
            workNumber = floorl_((integerNumber + integerAdjust) / p);
        }
        out += static_cast<char>('0' + to_int(fmod10(workNumber)));
    }

    if (keepDecimalPoint) out += '.';

    // the fraction
    int trailingZeroes = 0;
    for (int i = 0; i < fractionDigits; i++) {
        fractionNumber = fractionNumber * dblBase;
        fractionAdjust = fractionAdjust * dblBase;
        if (i == fractionDigits - 1) {
            const int ii = to_int(floorl_(fractionNumber));
            if (ii % 2 == 0)
                workNumber = floorl_(fractionNumber + fractionAdjust * Extended(1 - 1e-14));
            else
                workNumber = floorl_(fractionNumber + fractionAdjust);
        } else {
            workNumber = floorl_(fractionNumber + fractionAdjust);
        }
        if (gt(workNumber, fractionNumber)) {
            // fractionNumber should never become negative
            fractionNumber = 0.0;
            fractionAdjust = 0.0;
        } else {
            fractionNumber = fractionNumber - workNumber;
        }
        const int offset = to_int(fmod10(workNumber));
        if (offset == 0) {
            trailingZeroes++;
        } else {
            for (; trailingZeroes > 0; trailingZeroes--) out += '0';
            out += static_cast<char>('0' + offset);
        }
    }
    for (; trailingZeroes > 0; trailingZeroes--) out += '0';

    if (exponentDigits > 0) {
        out += 'e';
        out += isExponentNegative ? '-' : '+';
        if (requireTwoDigitExponent) out += '0';
        unsigned int exponentBase = 1;
        for (int i = 1; i < exponentDigits; ++i) exponentBase *= 10;
        for (int i = 0; i < exponentDigits; i++) {
            out += static_cast<char>('0' + (uExponent / exponentBase) % 10);
            exponentBase /= 10;
        }
    }
    return out;
}

bool is_r_na(double x) {
    if (!std::isnan(x)) return false;
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return static_cast<std::uint32_t>(bits) == 1954u;  // R's NA_real_: a NaN with low word 1954
}

}  // namespace

std::string r_as_character(double x) {
    if (std::isnan(x)) return is_r_na(x) ? "NA" : "NaN";
    if (std::isinf(x)) return x > 0 ? "Inf" : "-Inf";

    // formatReal(&x, 1, &w, &d, &e, 0)
    const Scientific s = scientific(x);
    int left = s.kpower + 1;
    if (s.roundingwidens) --left;
    const int sleft = s.neg + ((left <= 0) ? 1 : left);
    int rgt = s.nsig - left;  // digits right of '.'
    const int mxl = left, mnl = left, mxns = s.nsig;
    int mxsl = sleft;
    if (mxl < 0) mxsl = 1 + s.neg;
    if (rgt < 0) rgt = 0;
    const int wF = mxsl + rgt + (rgt != 0);
    int e = (mxl > 100 || mnl <= -99) ? 2 : 1;
    int d = mxns - 1;
    int w = s.neg + (d > 0) + d + 4 + e;
    if (wF <= w + kScipen) {
        e = 0;
        d = rgt;
        w = wF;
    }

    // EncodeRealDrop0(x, w, d, e, "."): "%#w.de" (d > 0) / "%w.de" / "%w.df"
    // through trio, then the trailing zeros of the fraction dropped
    if (x == 0.0) x = 0.0;  // no "-0"
    std::string buff = trio_write_double(x, e != 0, e != 0 && d != 0, w, d);
    const std::size_t dot = buff.find('.');
    if (dot != std::string::npos) {
        std::size_t p = dot + 1, replace = dot;
        while (p < buff.size() && buff[p] >= '0' && buff[p] <= '9')
            if (buff[p++] != '0') replace = p;
        if (replace != p) buff.erase(replace, p - replace);
    }
    return buff;
}

}  // namespace sdm
