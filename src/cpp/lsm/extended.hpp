// SPDX-License-Identifier: GPL-3.0-or-later
//
// R's LDOUBLE: the x87 80-bit extended type (64-bit mantissa) that R on
// x86-64 (gcc / mingw-w64) uses in sum(), mean() and var(). MSVC has no such
// type (its long double is a double), so it is emulated here with correctly
// rounded (round half to even) integer arithmetic; where the compiler's long
// double already has a 64-bit mantissa it is used directly. Either way the
// results are the same bits as R's.
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define LSM_FORCE_INLINE __forceinline
#else
#define LSM_FORCE_INLINE inline __attribute__((always_inline))
#endif

namespace lsm {

class SoftExtended {
public:
    SoftExtended() = default;
    SoftExtended(double d) { from_double(d); }  // NOLINT: implicit, like long double
    SoftExtended(std::int64_t n) { from_int(n); }
    SoftExtended(int n) { from_int(n); }

    explicit operator double() const;

    friend SoftExtended operator+(const SoftExtended& a, const SoftExtended& b) { return add(a, b, false); }
    friend SoftExtended operator-(const SoftExtended& a, const SoftExtended& b) { return add(a, b, true); }
    friend SoftExtended operator*(const SoftExtended& a, const SoftExtended& b);
    friend SoftExtended operator/(const SoftExtended& a, const SoftExtended& b);
    SoftExtended& operator+=(const SoftExtended& b) { return *this = *this + b; }
    // Accumulating a double: fast path for a same-sign addend no larger
    // than the sum (the common case of sums of positive values).
    SoftExtended& operator+=(double d) {
        std::uint64_t bits;
        std::memcpy(&bits, &d, sizeof bits);
        const int ex = static_cast<int>((bits >> 52) & 0x7FF);
        const bool neg = (bits >> 63) != 0;
        if (kind_ == kFinite && ex != 0 && ex != 0x7FF && neg == neg_) {
            const std::int32_t xe = ex - 1075 - 11;
            const std::int64_t shift = static_cast<std::int64_t>(e_) - xe;
            if (shift >= 0 && shift < 64) {
                constexpr std::uint64_t kTop = std::uint64_t(1) << 63;
                const std::uint64_t xm = ((bits & ((std::uint64_t(1) << 52) - 1)) | (std::uint64_t(1) << 52)) << 11;
                std::uint64_t lo = shift ? xm << (64 - shift) : 0;
                std::uint64_t hi = m_ + (xm >> shift);
                std::int32_t e = e_;
                if (hi < m_) {  // carry: shift right one, keeping a sticky bit
                    lo = (lo >> 1) | (hi << 63) | (lo & 1);
                    hi = (hi >> 1) | kTop;
                    ++e;
                }
                if (lo > kTop || (lo == kTop && (hi & 1))) {
                    if (++hi == 0) {
                        hi = kTop;
                        ++e;
                    }
                }
                m_ = hi;
                e_ = e;
                return *this;
            }
        }
        return *this = *this + SoftExtended(d);
    }
    SoftExtended& operator-=(const SoftExtended& b) { return *this = *this - b; }
    SoftExtended& operator*=(const SoftExtended& b) { return *this = *this * b; }
    SoftExtended& operator/=(const SoftExtended& b) { return *this = *this / b; }

private:
    enum Kind : std::uint8_t { kZero, kFinite, kInf, kNan };

    // value = m * 2^e, m normalised (bit 63 set) when finite
    std::uint64_t m_ = 0;
    std::int32_t e_ = 0;
    bool neg_ = false;
    Kind kind_ = kZero;

    struct U128 {
        std::uint64_t hi, lo;
    };

    static int clz64(std::uint64_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
        unsigned long idx;
        _BitScanReverse64(&idx, v);
        return 63 - static_cast<int>(idx);
#else
        return __builtin_clzll(v);
#endif
    }
    static U128 mul64(std::uint64_t a, std::uint64_t b) {
#if defined(_MSC_VER) && !defined(__clang__)
        std::uint64_t hi;
        const std::uint64_t lo = _umul128(a, b, &hi);
        return {hi, lo};
#else
        const unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
        return {static_cast<std::uint64_t>(p >> 64), static_cast<std::uint64_t>(p)};
#endif
    }
    // (hi:lo) / d with hi < d
    static std::uint64_t div128(std::uint64_t hi, std::uint64_t lo, std::uint64_t d, std::uint64_t* rem) {
#if defined(_MSC_VER) && !defined(__clang__)
        return _udiv128(hi, lo, d, rem);
#else
        const unsigned __int128 n = (static_cast<unsigned __int128>(hi) << 64) | lo;
        *rem = static_cast<std::uint64_t>(n % d);
        return static_cast<std::uint64_t>(n / d);
#endif
    }

    static SoftExtended special(Kind k, bool neg) {
        SoftExtended r;
        r.kind_ = k;
        r.neg_ = neg;
        return r;
    }

    // Round the 128-bit value (hi:lo) * 2^e (lo already holding a sticky
    // bit for anything below it) to 64 bits, half to even.
    LSM_FORCE_INLINE static SoftExtended round(std::uint64_t hi, std::uint64_t lo, std::int32_t e, bool neg) {
        if (hi == 0 && lo == 0) return special(kZero, false);
        if (hi == 0) {
            hi = lo;
            lo = 0;
            e -= 64;
        }
        const int s = clz64(hi);
        if (s) {
            hi = (hi << s) | (lo >> (64 - s));
            lo <<= s;
            e -= s;
        }
        constexpr std::uint64_t kHalf = std::uint64_t(1) << 63;
        if (lo > kHalf || (lo == kHalf && (hi & 1))) {
            if (++hi == 0) {
                hi = kHalf;
                ++e;
            }
        }
        SoftExtended r;
        r.kind_ = kFinite;
        r.neg_ = neg;
        r.m_ = hi;
        r.e_ = e + 64;
        return r;
    }

    LSM_FORCE_INLINE void from_double(double d) {
        std::uint64_t bits;
        std::memcpy(&bits, &d, sizeof bits);
        neg_ = (bits >> 63) != 0;
        const int ex = static_cast<int>((bits >> 52) & 0x7FF);
        const std::uint64_t frac = bits & ((std::uint64_t(1) << 52) - 1);
        if (ex == 0x7FF) {
            kind_ = frac ? kNan : kInf;
            return;
        }
        std::uint64_t mant = ex ? (frac | (std::uint64_t(1) << 52)) : frac;
        if (mant == 0) {
            kind_ = kZero;
            return;
        }
        std::int32_t e = (ex ? ex : 1) - 1075;
        const int s = clz64(mant);
        mant <<= s;
        e -= s;
        kind_ = kFinite;
        m_ = mant;
        e_ = e;
    }

    void from_int(std::int64_t n) {
        if (n == 0) {
            kind_ = kZero;
            neg_ = false;
            return;
        }
        neg_ = n < 0;
        std::uint64_t v = neg_ ? (~static_cast<std::uint64_t>(n) + 1) : static_cast<std::uint64_t>(n);
        const int s = clz64(v);
        kind_ = kFinite;
        m_ = v << s;
        e_ = -s;
    }

    LSM_FORCE_INLINE static SoftExtended add(const SoftExtended& a, SoftExtended b, bool subtract) {
        if (subtract) b.neg_ = !b.neg_;
        if (a.kind_ == kNan || b.kind_ == kNan) return special(kNan, false);
        if (a.kind_ == kInf || b.kind_ == kInf) {
            if (a.kind_ == kInf && b.kind_ == kInf && a.neg_ != b.neg_) return special(kNan, false);
            return a.kind_ == kInf ? a : b;
        }
        if (a.kind_ == kZero) {
            if (b.kind_ == kZero) return special(kZero, a.neg_ && b.neg_);
            return b;
        }
        if (b.kind_ == kZero) return a;
        // x = larger magnitude
        const bool swap = b.e_ > a.e_ || (b.e_ == a.e_ && b.m_ > a.m_);
        const SoftExtended& x = swap ? b : a;
        const SoftExtended& y = swap ? a : b;
        const std::int64_t d = static_cast<std::int64_t>(x.e_) - y.e_;
        std::uint64_t yh, yl;
        if (d == 0) {
            yh = y.m_;
            yl = 0;
        } else if (d < 64) {
            yh = y.m_ >> d;
            yl = y.m_ << (64 - d);
        } else if (d == 64) {
            yh = 0;
            yl = y.m_;
        } else if (d < 128) {
            yh = 0;
            yl = (y.m_ >> (d - 64)) | ((y.m_ << (128 - d)) != 0 ? 1 : 0);
        } else {
            yh = 0;
            yl = 1;  // sticky
        }
        const std::int32_t e = x.e_ - 64;  // exponent of the low limb
        if (x.neg_ == y.neg_) {
            std::uint64_t lo = yl;  // x's low limb is 0
            std::uint64_t hi = x.m_ + yh;
            if (hi < x.m_) {  // carry out of bit 127: shift right one
                const std::uint64_t sticky = lo & 1;
                lo = (lo >> 1) | (hi << 63) | sticky;
                hi = (hi >> 1) | (std::uint64_t(1) << 63);
                return round(hi, lo, e + 1, x.neg_);
            }
            return round(hi, lo, e, x.neg_);
        }
        const std::uint64_t lo = 0 - yl;
        const std::uint64_t hi = x.m_ - yh - (yl != 0 ? 1 : 0);
        if (hi == 0 && lo == 0) return special(kZero, false);
        return round(hi, lo, e, x.neg_);
    }

};

inline SoftExtended::operator double() const {
    switch (kind_) {
        case kZero:
            return neg_ ? -0.0 : 0.0;
        case kInf:
            return neg_ ? -HUGE_VAL : HUGE_VAL;
        case kNan:
            return std::numeric_limits<double>::quiet_NaN();
        default:
            break;
    }
    std::uint64_t m = m_ >> 11;
    const std::uint64_t rest = m_ & 0x7FF;
    std::int32_t e = e_ + 11;
    if (rest > 0x400 || (rest == 0x400 && (m & 1))) {
        if (++m == (std::uint64_t(1) << 53)) {
            m >>= 1;
            ++e;
        }
    }
    const double v = std::ldexp(static_cast<double>(m), e);
    return neg_ ? -v : v;
}

LSM_FORCE_INLINE SoftExtended operator*(const SoftExtended& a, const SoftExtended& b) {
    const bool neg = a.neg_ != b.neg_;
    if (a.kind_ == SoftExtended::kNan || b.kind_ == SoftExtended::kNan) return SoftExtended::special(SoftExtended::kNan, false);
    if (a.kind_ == SoftExtended::kInf || b.kind_ == SoftExtended::kInf) {
        if (a.kind_ == SoftExtended::kZero || b.kind_ == SoftExtended::kZero) return SoftExtended::special(SoftExtended::kNan, false);
        return SoftExtended::special(SoftExtended::kInf, neg);
    }
    if (a.kind_ == SoftExtended::kZero || b.kind_ == SoftExtended::kZero) return SoftExtended::special(SoftExtended::kZero, neg);
    const SoftExtended::U128 p = SoftExtended::mul64(a.m_, b.m_);
    return SoftExtended::round(p.hi, p.lo, a.e_ + b.e_, neg);
}

inline SoftExtended operator/(const SoftExtended& a, const SoftExtended& b) {
    const bool neg = a.neg_ != b.neg_;
    if (a.kind_ == SoftExtended::kNan || b.kind_ == SoftExtended::kNan) return SoftExtended::special(SoftExtended::kNan, false);
    if (a.kind_ == SoftExtended::kInf) {
        return b.kind_ == SoftExtended::kInf ? SoftExtended::special(SoftExtended::kNan, false)
                                         : SoftExtended::special(SoftExtended::kInf, neg);
    }
    if (b.kind_ == SoftExtended::kInf) return SoftExtended::special(SoftExtended::kZero, neg);
    if (b.kind_ == SoftExtended::kZero) {
        return a.kind_ == SoftExtended::kZero ? SoftExtended::special(SoftExtended::kNan, false)
                                          : SoftExtended::special(SoftExtended::kInf, neg);
    }
    if (a.kind_ == SoftExtended::kZero) return SoftExtended::special(SoftExtended::kZero, neg);
    // a.m / b.m in [1/2, 2): 128 quotient bits below the binary point plus a
    // leading bit, then a sticky bit from the remainder
    std::uint64_t rem = a.m_;
    const bool top = rem >= b.m_;
    if (top) rem -= b.m_;
    std::uint64_t r1, r2;
    const std::uint64_t q1 = SoftExtended::div128(rem, 0, b.m_, &r1);
    const std::uint64_t q2 = SoftExtended::div128(r1, 0, b.m_, &r2);
    std::int32_t e = a.e_ - b.e_ - 128;
    std::uint64_t hi = q1, lo = q2 | (r2 != 0 ? 1 : 0);
    if (top) {
        const std::uint64_t sticky = lo & 1;
        lo = (lo >> 1) | (hi << 63) | sticky;
        hi = (hi >> 1) | (std::uint64_t(1) << 63);
        ++e;
    }
    return SoftExtended::round(hi, lo, e, neg);
}

#if LDBL_MANT_DIG == 64
using Extended = long double;
#else
using Extended = SoftExtended;
#endif

}  // namespace lsm
