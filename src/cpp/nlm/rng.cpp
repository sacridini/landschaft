// SPDX-License-Identifier: GPL-3.0-or-later
//
// R's and libstdc++'s random number generators (see rng.hpp).

#include "nlm/rng.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"

namespace nlm {

namespace {

constexpr int kN = 624;
constexpr int kM = 397;
constexpr std::uint32_t kMatrixA = 0x9908b0dfu;
constexpr std::uint32_t kUpper = 0x80000000u;
constexpr std::uint32_t kLower = 0x7fffffffu;

// The Mersenne-Twister recurrence over the whole state, as in R's
// MT_genrand() and libstdc++'s _M_gen_rand().
void twist(std::uint32_t* mt) {
    int kk = 0;
    for (; kk < kN - kM; ++kk) {
        const std::uint32_t y = (mt[kk] & kUpper) | (mt[kk + 1] & kLower);
        mt[kk] = mt[kk + kM] ^ (y >> 1) ^ ((y & 1u) ? kMatrixA : 0u);
    }
    for (; kk < kN - 1; ++kk) {
        const std::uint32_t y = (mt[kk] & kUpper) | (mt[kk + 1] & kLower);
        mt[kk] = mt[kk + (kM - kN)] ^ (y >> 1) ^ ((y & 1u) ? kMatrixA : 0u);
    }
    const std::uint32_t y = (mt[kN - 1] & kUpper) | (mt[0] & kLower);
    mt[kN - 1] = mt[kM - 1] ^ (y >> 1) ^ ((y & 1u) ? kMatrixA : 0u);
}

inline std::uint32_t temper(std::uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

constexpr double kI2_32m1 = 2.328306437080797e-10;  // 1 / (2^32 - 1)
constexpr double kBig = 134217728;                  // 2^27

}  // namespace

// -- R ------------------------------------------------------------------------

void RRng::set_seed(std::int32_t seed_in) {
    // RNG_Init(): initial scrambling, then 625 values of which the first is
    // mti (reset to 624 by FixupSeeds).
    std::uint32_t seed = static_cast<std::uint32_t>(seed_in);
    for (int j = 0; j < 50; ++j) seed = 69069u * seed + 1u;
    seed = 69069u * seed + 1u;  // i_seed[0], then overwritten
    for (int j = 0; j < kN; ++j) {
        seed = 69069u * seed + 1u;
        mt_[j] = seed;
    }
    mti_ = kN;
}

double RRng::fixup(double x) {
    if (x <= 0.0) return 0.5 * kI2_32m1;
    if ((1.0 - x) <= 0.0) return 1.0 - 0.5 * kI2_32m1;
    return x;
}

double RRng::genrand() {
    if (mti_ >= kN) {
        twist(mt_);
        mti_ = 0;
    }
    return static_cast<double>(temper(mt_[mti_++])) * 2.3283064365386963e-10;
}

double RRng::norm_rand() {
    double u1 = unif_rand();
    u1 = static_cast<int>(kBig * u1) + unif_rand();
    return r_qnorm(u1 / kBig);
}

double RRng::rbits(int bits) {
    std::int64_t v = 0;
    for (int n = 0; n <= bits; n += 16) {
        const int v1 = static_cast<int>(std::floor(unif_rand() * 65536));
        v = 65536 * v + v1;
    }
    return static_cast<double>(v & ((std::int64_t{1} << bits) - 1));
}

double RRng::unif_index(double dn) {
    if (dn <= 0) return 0.0;
    const int bits = static_cast<int>(std::ceil(std::log2(dn)));
    double dv;
    do {
        dv = rbits(bits);
    } while (dn <= dv);
    return dv;
}

double RRng::runif(double a, double b) {
    if (a == b) return a;
    double u;
    do {
        u = unif_rand();
    } while (u <= 0 || u >= 1);
    return a + (b - a) * u;
}

double RRng::rnorm(double mu, double sigma) {
    if (sigma == 0. || !std::isfinite(mu)) return mu;
    return mu + sigma * norm_rand();
}

double RRng::exp_rand() {
    // sexp.c
    static const double q[] = {0.6931471805599453, 0.9333736875190459, 0.9888777961838675, 0.9984959252914960,
                               0.9998292811061389, 0.9999833164100727, 0.9999985691438767, 0.9999998906925558,
                               0.9999999924734159, 0.9999999995283275, 0.9999999999728814, 0.9999999999985598,
                               0.9999999999999289, 0.9999999999999968, 0.9999999999999999, 1.0000000000000000};
    double a = 0.;
    double u = unif_rand();
    while (u <= 0. || u >= 1.) u = unif_rand();
    for (;;) {
        u += u;
        if (u > 1.) break;
        a += q[0];
    }
    u -= 1.;
    if (u <= q[0]) return a + u;
    int i = 0;
    double ustar = unif_rand(), umin = ustar;
    do {
        ustar = unif_rand();
        if (umin > ustar) umin = ustar;
        i++;
    } while (u > q[i]);
    return a + umin * q[0];
}

namespace {

// mingw-w64's pow() of doubles for an integer exponent (__powi: repeated
// squaring in double), as rpois() calls it.
double mingw_powi(double x, int y) {
    if (y == 0 || x == 1.0) return 1.0;
    const bool odd = (y & 1) != 0;
    const bool recip = y < 0;
    unsigned int u = recip ? 0u - static_cast<unsigned int>(y) : static_cast<unsigned int>(y);
    double d = std::fabs(x), r;
    if (u == 1) {
        r = d;
    } else {
        r = (u & 1) ? d : 1.0;
        u >>= 1;
        do {
            d *= d;
            if (u & 1) r *= d;
            u >>= 1;
        } while (u > 0);
    }
    if (recip) r = 1.0 / r;
    return (x < 0 && odd) ? -r : r;
}

}  // namespace

double RRng::rpois(double mu) {
    // nmath/rpois.c of R 4.4 (Ahrens & Dieter 1982)
    constexpr double a0 = -0.5, a1 = 0.3333333, a2 = -0.2500068, a3 = 0.2000118, a4 = -0.1661269, a5 = 0.1421878,
                     a6 = -0.1384794, a7 = 0.1250060;
    constexpr double one_7 = 0.1428571428571428571, one_12 = 0.0833333333333333333,
                     one_24 = 0.0416666666666666667;
    constexpr double M_1_SQRT_2PI_ = 0.398942280401432677939946059934;
    static const double fact[10] = {1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880.};
    PoisState& st = pois_;
    double del, difmuk = 0., E = 0., fk = 0., fx, fy, g, px, py, t, u = 0., v, x;
    double pois = -1.;
    int k, kflag, big_mu, new_big_mu = 0;

    if (!std::isfinite(mu) || mu < 0) return std::numeric_limits<double>::quiet_NaN();
    if (mu <= 0.) return 0.;

    big_mu = mu >= 10.;
    if (big_mu) new_big_mu = 0;

    if (!(big_mu && mu == st.muprev)) {
        if (big_mu) {
            new_big_mu = 1;
            st.muprev = mu;
            st.s = std::sqrt(mu);
            st.d = 6. * mu * mu;
            st.big_l = std::floor(mu - 1.1484);
        } else {
            if (mu != st.muprev) {
                st.muprev = mu;
                st.m = std::max(1, static_cast<int>(mu));
                st.l = 0;
                st.q = st.p0 = st.p = lsm::r_exp(-mu);
            }
            for (;;) {
                u = unif_rand();
                if (u <= st.p0) return 0.;
                if (st.l != 0) {
                    for (k = (u <= 0.458) ? 1 : std::min(st.l, st.m); k <= st.l; k++)
                        if (u <= st.pp[k]) return static_cast<double>(k);
                    if (st.l == 35) continue;
                }
                st.l++;
                for (k = st.l; k <= 35; k++) {
                    st.p *= mu / k;
                    st.q += st.p;
                    st.pp[k] = st.q;
                    if (u <= st.q) {
                        st.l = k;
                        return static_cast<double>(k);
                    }
                }
                st.l = 35;
            }
        }
    }

    g = mu + st.s * norm_rand();
    if (g >= 0.) {
        pois = std::floor(g);
        if (pois >= st.big_l) return pois;
        fk = pois;
        difmuk = mu - fk;
        u = unif_rand();
        if (st.d * u >= difmuk * difmuk * difmuk) return pois;
    }

    if (new_big_mu || mu != st.muprev2) {
        st.muprev2 = mu;
        st.omega = M_1_SQRT_2PI_ / st.s;
        st.b1 = one_24 / mu;
        st.b2 = 0.3 * st.b1 * st.b1;
        st.c3 = one_7 * st.b1 * st.b2;
        st.c2 = st.b2 - 15. * st.c3;
        st.c1 = st.b1 - 6. * st.b2 + 45. * st.c3;
        st.c0 = 1. - st.b1 + 3. * st.b2 - 15. * st.c3;
        st.c = 0.1069 / mu;
    }

    if (g >= 0.) {
        kflag = 0;
        goto Step_F;
    }

    for (;;) {
        E = exp_rand();
        u = 2 * unif_rand() - 1.;
        t = 1.8 + ((u >= 0) ? std::fabs(E) : -std::fabs(E));  // fsign(E, u)
        if (t > -0.6744) {
            pois = std::floor(mu + st.s * t);
            fk = pois;
            difmuk = mu - fk;
            kflag = 1;

        Step_F:
            if (pois < 10) {
                px = -mu;
                py = mingw_powi(mu, static_cast<int>(pois)) / fact[static_cast<int>(pois)];
            } else {
                del = one_12 / fk;
                del = del * (1. - 4.8 * del * del);
                v = difmuk / fk;
                if (std::fabs(v) <= 0.25)
                    px = fk * v * v * (((((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v + a2) * v + a1) * v + a0) -
                         del;
                else
                    px = fk * lsm::r_log(1. + v) - difmuk - del;
                py = M_1_SQRT_2PI_ / std::sqrt(fk);
            }
            x = (0.5 - difmuk) / st.s;
            x *= x;
            fx = -0.5 * x;
            fy = st.omega * (((st.c3 * x + st.c2) * x + st.c1) * x + st.c0);
            if (kflag > 0) {
                if (st.c * std::fabs(u) <= py * lsm::r_exp(px + E) - fy * lsm::r_exp(fx + E)) break;
            } else if (fy - u * fy <= py * lsm::r_exp(px - fx)) {
                break;
            }
        }
    }
    return pois;
}

void RRng::runif(double* out, i64 n, double a, double b) {
    for (i64 i = 0; i < n; ++i) out[i] = runif(a, b);
}

void RRng::rnorm(double* out, i64 n, double mu, double sigma) {
    if (sigma == 0. || !std::isfinite(mu)) {
        for (i64 i = 0; i < n; ++i) out[i] = mu;
        return;
    }
    // The generator runs in sequence; the inversion does not depend on it.
    for (i64 i = 0; i < n; ++i) {
        double u1 = unif_rand();
        u1 = static_cast<int>(kBig * u1) + unif_rand();
        out[i] = u1 / kBig;
    }
    lsm::parallel_for(n > 20000, n, [&](i64 i) { out[i] = mu + sigma * r_qnorm(out[i]); });
}

std::vector<i64> RRng::sample2(i64 n, i64 size) {
    // do_sample2(): up to 100 draws until the value is new.
    std::vector<i64> out(static_cast<std::size_t>(size));
    std::unordered_set<i64> seen;
    seen.reserve(static_cast<std::size_t>(size) * 2);
    const double dn = static_cast<double>(n);
    for (i64 i = 0; i < size; ++i) {
        i64 v = 0;
        for (int j = 0; j < 100; ++j) {
            v = static_cast<i64>(unif_index(dn)) + 1;
            if (seen.insert(v).second) break;
        }
        out[static_cast<std::size_t>(i)] = v;
    }
    return out;
}

namespace {

// ProbSampleReplace(): perm[j] for the first j < n - 1 with rU <= the
// cumulative p[j], else perm[n - 1]. Rounding is monotone, so the cumulative
// sums never decrease and bisection finds the j of R's linear scan.
void prob_sample_replace(int n, double* p, int* perm, i64 nans, i64* ans, RRng& rng, bool reference) {
    const int nm1 = n - 1;
    for (int i = 0; i < n; ++i) perm[i] = i + 1;
    r_revsort(p, perm, n);
    for (int i = 1; i < n; ++i) p[i] += p[i - 1];
    for (i64 i = 0; i < nans; ++i) {
        const double rU = rng.unif_rand();
        int j;
        if (reference) {
            for (j = 0; j < nm1; ++j) {
                if (rU <= p[j]) break;
            }
        } else {
            j = static_cast<int>(std::lower_bound(p, p + nm1, rU) - p);
        }
        ans[i] = perm[j];
    }
}

void walker_prob_sample_replace(int n, const double* p, i64 nans, i64* ans, RRng& rng) {
    std::vector<int> HL(static_cast<std::size_t>(n)), a(static_cast<std::size_t>(n), 0);
    std::vector<double> q(static_cast<std::size_t>(n));
    int* H = HL.data() - 1;
    int* L = HL.data() + n;
    for (int i = 0; i < n; ++i) {
        q[i] = p[i] * n;
        if (q[i] < 1.)
            *++H = i;
        else
            *--L = i;
    }
    if (H >= HL.data() && L < HL.data() + n) {
        for (int k = 0; k < n - 1; ++k) {
            const int i = HL[k];
            const int j = *L;
            a[i] = j;
            q[j] += q[i] - 1;
            if (q[j] < 1.) L++;
            if (L >= HL.data() + n) break;
        }
    }
    for (int i = 0; i < n; ++i) q[i] += i;
    for (i64 i = 0; i < nans; ++i) {
        const int k = static_cast<int>(rng.unif_index(n));
        const double rU = k + rng.unif_rand();
        ans[i] = (rU < q[k]) ? k + 1 : a[k] + 1;
    }
}

// R's ProbSampleNoReplace() as written: for every draw a scan of the
// descending probabilities (mass += p[j] until rT <= mass) and a shift of the
// arrays over the drawn element, O(n) per draw. Kept to test the fast
// version below against.
void prob_sample_no_replace_reference(int n, double* p, int* perm, i64 nans, i64* ans, RRng& rng) {
    for (int i = 0; i < n; ++i) perm[i] = i + 1;
    r_revsort(p, perm, n);
    double totalmass = 1;
    int n1 = n - 1;
    for (i64 i = 0; i < nans; ++i, --n1) {
        const double rT = totalmass * rng.unif_rand();
        double mass = 0;
        int j;
        for (j = 0; j < n1; ++j) {
            mass += p[j];
            if (rT <= mass) break;
        }
        ans[i] = perm[j];
        totalmass -= p[j];
        for (int k = j; k < n1; ++k) {
            p[k] = p[k + 1];
            perm[k] = perm[k + 1];
        }
    }
}

// The same draws in O(runs scanned + log n) per draw. Equal probabilities
// are contiguous after the sort and form runs (value w, number of elements
// not drawn yet); the scan adds w run by run, each run in closed form
// (repeated_add()), so it computes every rounded mass R's scan computes and
// stops at the same element. The masses at the ends of the runs are kept
// between draws as far as the last draw left them unchanged, and drawn
// elements are flagged off in a Fenwick tree over the sorted positions
// instead of being shifted out.

// Fenwick tree of 0/1 flags over positions 0 .. n-1, all set at first.
class AliveTree {
public:
    explicit AliveTree(int n) : n_(n), t_(static_cast<std::size_t>(n) + 1, 0) {
        for (int i = 1; i <= n; ++i) {
            t_[i] += 1;
            const int j = i + (i & -i);
            if (j <= n) t_[j] += t_[i];
        }
        while (top_ * 2 <= n) top_ *= 2;
    }
    void clear(int pos) {
        for (int i = pos + 1; i <= n_; i += i & -i) --t_[i];
    }
    // The number of set flags at positions < pos.
    int count_before(int pos) const {
        int k = 0;
        for (int i = pos; i > 0; i -= i & -i) k += t_[i];
        return k;
    }
    // The position of the set flag of 0-based rank k (k < number set).
    int kth(int k) const {
        int pos = 0;
        for (int step = top_; step > 0; step >>= 1) {
            if (pos + step <= n_ && t_[pos + step] <= k) {
                pos += step;
                k -= t_[pos];
            }
        }
        return pos;
    }

private:
    int n_;
    int top_ = 1;
    std::vector<int> t_;
};

constexpr i64 kTwo53 = i64(1) << 53;

}  // namespace

// Let u be the spacing of the doubles at mass (2^(e-52) for mass in
// [2^e, 2^(e+1)); 2^-1074 for mass < 2^-1021, where the spacing is the same
// from 0), mass = a*u with an integer a < 2^53, and w = (q + f)*u with an
// integer q and 0 <= f < 1. While a + q < 2^53 the exact sum is below
// 2^53*u, where the spacing is still u, so mass + w rounds to
// (a + q + d)*u: d = 1 for f > 1/2, 0 for f < 1/2 and, for a tie f = 1/2,
// 1 if a + q is odd (ties to even). For an even a a tie adds q rounded up to
// even and a stays even, so (after one ordinary addition from an odd a) every
// addition adds the same inc = q + d until the binade ends, mass after s of
// them is (a + s*inc)*u, and the first s with rT <= mass follows from
// integers. At the end of a binade, and where these do not apply (mass = 0,
// w at least 2^53*u, mass of 2^53 or more), the additions are made one by one.
i64 repeated_add(double& mass, double w, i64 c, double rT) {
    i64 done = 0;
    while (done < c) {
        std::uint64_t bits;
        std::memcpy(&bits, &mass, sizeof bits);
        const std::uint64_t e = bits >> 52;  // the biased exponent (mass >= 0)
        // rT > mass: no addition can stop the scan while mass does not grow
        if (c - done >= 4 && mass > 0.0 && rT > mass && e <= 1075) {
            // u = 2^(e-1075), a subnormal for e < 53; 1/u (>= 1 for
            // e <= 1075) is a double only for e >= 52
            const std::uint64_t ubits = e >= 53 ? (e - 52) << 52 : std::uint64_t(1) << (e > 0 ? e - 1 : 0);
            const std::uint64_t ibits = (2098 - (e >= 53 ? e : 53)) << 52;
            double u, inv;
            std::memcpy(&u, &ubits, sizeof u);
            std::memcpy(&inv, &ibits, sizeof inv);
            const auto scale = [&](double x) { return e >= 53 ? x * inv : x / u; };
            const double top = static_cast<double>(kTwo53) * u;
            if (w < top) {
                // all exact: scalings by powers of 2 within range, and the
                // fractional part of a double
                const double wu = scale(w);
                const double qd = std::floor(wu);
                const double f = wu - qd;
                const i64 a = static_cast<i64>(scale(mass));
                const i64 q = static_cast<i64>(qd);
                const bool tie = f == 0.5;
                if (a + q < kTwo53 && !(tie && (a & 1))) {
                    const i64 inc = q + ((f > 0.5 || (tie && (q & 1))) ? 1 : 0);
                    // additions from a + s*inc for s = 0 .. len-1 keep a + q < 2^53
                    i64 len = c - done;
                    if (inc > 0) len = std::min(len, (kTwo53 - 1 - q - a) / inc + 1);
                    if (inc > 0 && rT <= top) {
                        // rT/u in (a, 2^53]: a + s*inc >= ceil(rT/u) <=> mass >= rT
                        const i64 at = static_cast<i64>(std::ceil(scale(rT)));
                        const i64 s = (at - a + inc - 1) / inc;
                        if (s <= len) {
                            mass = static_cast<double>(a + s * inc) * u;
                            return done + s;
                        }
                    }
                    mass = static_cast<double>(a + len * inc) * u;
                    done += len;
                    continue;
                }
            }
        }
        mass += w;
        ++done;
        if (rT <= mass) return done;
    }
    return 0;
}

namespace {

void prob_sample_no_replace(int n, double* p, int* perm, i64 nans, i64* ans, RRng& rng) {
    for (int i = 0; i < n; ++i) perm[i] = i + 1;
    r_revsort(p, perm, n);
    // runs of equal probabilities: value, first sorted position, elements
    // not drawn yet
    std::vector<double> w;
    std::vector<int> start, count;
    for (int i = 0; i < n; ++i) {
        if (i == 0 || p[i] != p[i - 1]) {
            w.push_back(p[i]);
            start.push_back(i);
            count.push_back(0);
        }
        ++count.back();
    }
    const int nr = static_cast<int>(w.size());
    // end[r]: the scan's mass after run r, known for r < valid. A draw from
    // run h changes the masses from run h on only, so the next scan resumes
    // from the known ones (by bisection: they never decrease). The last
    // non-empty run (tail) is never cached: R's scan does not test the last
    // element left, so its tested count changes with every draw.
    std::vector<double> end(static_cast<std::size_t>(nr));
    int valid = 0;
    int tail = nr - 1;
    AliveTree alive(n);
    double totalmass = 1;
    int m = n;  // elements not drawn
    for (i64 i = 0; i < nans; ++i, --m) {
        const double rT = totalmass * rng.unif_rand();
        int r = valid;
        if (valid > 0 && rT <= end[valid - 1])
            r = static_cast<int>(std::lower_bound(end.data(), end.data() + valid, rT) - end.data());
        double mass = r > 0 ? end[r - 1] : 0.0;
        int hit = tail;
        i64 s = 0;
        for (; r <= tail; ++r) {
            // R's scan tests the first m - 1 elements; the last one is drawn
            // when none of them ends it
            const i64 c = r < tail ? count[r] : count[r] - 1;
            if (c < 4) {
                for (i64 t = 1; t <= c; ++t) {
                    mass += w[r];
                    if (rT <= mass) {
                        s = t;
                        break;
                    }
                }
            } else {
                s = repeated_add(mass, w[r], c, rT);
            }
            if (s > 0) {
                hit = r;
                break;
            }
            if (r < tail) {
                end[r] = mass;
                if (r >= valid) valid = r + 1;
            }
        }
        const int pos = s > 0 ? alive.kth(alive.count_before(start[hit]) + static_cast<int>(s) - 1) : alive.kth(m - 1);
        ans[i] = perm[pos];
        totalmass -= p[pos];
        alive.clear(pos);
        --count[hit];
        while (tail > 0 && count[tail] == 0) --tail;
        valid = std::min(valid, std::min(hit, tail));
    }
}

// sample.int(n, size, replace, prob): FixupProb() and R's choice of sampler.
std::vector<i64> prob_sample(RRng& rng, i64 n, i64 size, bool replace, const std::vector<double>& prob,
                             bool reference) {
    if (static_cast<i64>(prob.size()) != n) throw std::invalid_argument("incorrect number of probabilities");
    if (n > INT_MAX) throw std::invalid_argument("sample_int: n too large with prob");
    const int ni = static_cast<int>(n);
    std::vector<double> p(prob);
    // FixupProb()
    double sum = 0.0;
    int npos = 0;
    for (int i = 0; i < ni; ++i) {
        if (!std::isfinite(p[i])) throw std::invalid_argument("NA in probability vector");
        if (p[i] < 0.0) throw std::invalid_argument("negative probability");
        if (p[i] > 0.0) {
            ++npos;
            sum += p[i];
        }
    }
    if (npos == 0 || (!replace && size > npos)) throw std::invalid_argument("too few positive probabilities");
    for (int i = 0; i < ni; ++i) p[i] /= sum;
    std::vector<i64> out(static_cast<std::size_t>(size));
    std::vector<int> perm(static_cast<std::size_t>(ni));
    if (replace) {
        int nc = 0;
        for (int i = 0; i < ni; ++i)
            if (ni * p[i] > 0.1) ++nc;
        if (nc > 200)
            walker_prob_sample_replace(ni, p.data(), size, out.data(), rng);
        else
            prob_sample_replace(ni, p.data(), perm.data(), size, out.data(), rng, reference);
    } else if (reference) {
        prob_sample_no_replace_reference(ni, p.data(), perm.data(), size, out.data(), rng);
    } else {
        prob_sample_no_replace(ni, p.data(), perm.data(), size, out.data(), rng);
    }
    return out;
}

}  // namespace

std::vector<i64> sample_int_reference(RRng& rng, i64 size, bool replace, const std::vector<double>& prob) {
    if (size < 0) throw std::invalid_argument("sample_int: invalid n or size");
    const i64 n = static_cast<i64>(prob.size());
    if (!replace && size > n)
        throw std::invalid_argument("cannot take a sample larger than the population when 'replace = FALSE'");
    return prob_sample(rng, n, size, replace, prob, true);
}

std::vector<i64> RRng::sample_int(i64 n, i64 size, bool replace, const std::vector<double>* prob) {
    if (n < 0 || size < 0) throw std::invalid_argument("sample_int: invalid n or size");
    if (!replace && size > n)
        throw std::invalid_argument("cannot take a sample larger than the population when 'replace = FALSE'");
    // sample.int()'s useHash.
    if (static_cast<double>(n) > 1e7 && !replace && prob == nullptr && static_cast<double>(size) <= n / 2.0)
        return sample2(n, size);
    if (prob != nullptr) return prob_sample(*this, n, size, replace, *prob, false);
    std::vector<i64> out(static_cast<std::size_t>(size));
    const double dn = static_cast<double>(n);
    if (replace || size < 2) {
        for (i64 i = 0; i < size; ++i) out[static_cast<std::size_t>(i)] = static_cast<i64>(unif_index(dn)) + 1;
        return out;
    }
    std::vector<i64> x(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) x[static_cast<std::size_t>(i)] = i;
    i64 m = n;
    for (i64 i = 0; i < size; ++i) {
        const i64 j = static_cast<i64>(unif_index(static_cast<double>(m)));
        out[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(j)] + 1;
        x[static_cast<std::size_t>(j)] = x[static_cast<std::size_t>(--m)];
    }
    return out;
}

double r_qnorm(double p) {
    // qnorm5(p, 0, 1, TRUE, FALSE) of R 4.4 (Wichura's AS 241); C's sqrt is
    // correctly rounded, log is mingw's (x87).
    if (std::isnan(p)) return p;
    if (p <= 0) return p == 0 ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
    if (p >= 1) return p == 1 ? std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
    const double q = p - 0.5;
    double r, val;
    if (std::fabs(q) <= .425) {
        r = .180625 - q * q;
        val = q *
              (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r +
                   45921.953931549871457) * r +
                  13731.693765509461125) * r +
                 1971.5909503065514427) * r +
                133.14166789178437745) * r +
               3.387132872796366608) /
              (((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r +
                   21213.794301586595867) * r +
                  5394.1960214247511077) * r +
                 687.1870074920579083) * r +
                42.313330701600911252) * r +
               1.);
        return val;
    }
    const double lp = lsm::r_log((q > 0) ? (0.5 - p + 0.5) : p);
    r = std::sqrt(-lp);
    if (r <= 5.) {
        r += -1.6;
        val = (((((((r * 7.7454501427834140764e-4 + .0227238449892691845833) * r + .24178072517745061177) * r +
                   1.27045825245236838258) * r +
                  3.64784832476320460504) * r +
                 5.7694972214606914055) * r +
                4.6303378461565452959) * r +
               1.42343711074968357734) /
              (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + .0151986665636164571966) * r +
                   .14810397642748007459) * r +
                  .68976733498510000455) * r +
                 1.6763848301838038494) * r +
                2.05319162663775882187) * r +
               1.);
    } else if (r <= 27) {
        r += -5.;
        val = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + .0012426609473880784386) * r +
                   .026532189526576123093) * r +
                  .29656057182850489123) * r +
                 1.7848265399172913358) * r +
                5.4637849111641143699) * r +
               6.6579046435011037772) /
              (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r +
                   7.868691311456132591e-4) * r +
                  .0148753612908506148525) * r +
                 .13692988092273580531) * r +
                .59983220655588793769) * r +
               1.);
    } else {
        // Not reached from norm_rand() (p >= 2^-60); asymptotic expansions.
        const double two_pi = 6.283185307179586476925286766559;
        if (r >= 6.4e8) {
            val = r * 1.414213562373095048801688724210;
        } else {
            const double s2 = -std::ldexp(lp, 1);
            double x2 = s2 - lsm::r_log(two_pi * s2);
            if (r < 36000.) {
                x2 = s2 - lsm::r_log(two_pi * x2) - 2. / (2. + x2);
                if (r < 840.) {
                    x2 = s2 - lsm::r_log(two_pi * x2) + 2 * std::log1p(-(1 - 1 / (4 + x2)) / (2. + x2));
                    if (r < 109.) {
                        x2 = s2 - lsm::r_log(two_pi * x2) +
                             2 * std::log1p(-(1 - (1 - 5 / (6 + x2)) / (4. + x2)) / (2. + x2));
                        if (r < 55.) {
                            x2 = s2 - lsm::r_log(two_pi * x2) +
                                 2 * std::log1p(-(1 - (1 - (5 - 9 / (8. + x2)) / (6. + x2)) / (4. + x2)) / (2. + x2));
                        }
                    }
                }
            }
            val = std::sqrt(x2);
        }
    }
    if (q < 0.0) val = -val;
    return val;
}

void r_revsort(double* a, int* ib, int n) {
    if (n <= 1) return;
    a--;
    ib--;
    int l = (n >> 1) + 1;
    int ir = n;
    double ra;
    int ii;
    for (;;) {
        if (l > 1) {
            l = l - 1;
            ra = a[l];
            ii = ib[l];
        } else {
            ra = a[ir];
            ii = ib[ir];
            a[ir] = a[1];
            ib[ir] = ib[1];
            if (--ir == 1) {
                a[1] = ra;
                ib[1] = ii;
                return;
            }
        }
        int i = l;
        int j = l << 1;
        while (j <= ir) {
            if (j < ir && a[j] > a[j + 1]) ++j;
            if (ra > a[j]) {
                a[i] = a[j];
                ib[i] = ib[j];
                j += (i = j);
            } else {
                j = ir + 1;
            }
        }
        a[i] = ra;
        ib[i] = ii;
    }
}

// -- libstdc++ ------------------------------------------------------------------

void StdMt19937::seed_with(std::uint32_t seed) {
    mt_[0] = seed;
    for (int i = 1; i < kN; ++i) mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<std::uint32_t>(i);
    mti_ = kN;
}

std::uint32_t StdMt19937::operator()() {
    if (mti_ >= kN) {
        twist(mt_);
        mti_ = 0;
    }
    return temper(mt_[mti_++]);
}

double std_canonical(StdMt19937& g) {
    // generate_canonical<double, 53>: m = 2 words, sum = w1 + w2 * 2^32 in
    // double, divided by 2^64; 1 is replaced by the largest double below it.
    double sum = static_cast<double>(g());
    sum += static_cast<double>(g()) * 4294967296.0;
    double ret = sum / 18446744073709551616.0;
    if (ret >= 1.0) ret = std::nextafter(1.0, 0.0);
    return ret;
}

std::int64_t std_uniform_int(StdMt19937& g, std::int64_t a, std::int64_t b) {
    const std::uint64_t urange = static_cast<std::uint64_t>(b - a);
    if (urange >= 0xffffffffull) {
        if (urange == 0xffffffffull) return a + static_cast<std::int64_t>(g());
        throw std::invalid_argument("std_uniform_int: range wider than 32 bits");
    }
    const std::uint32_t range = static_cast<std::uint32_t>(urange + 1);
    std::uint64_t product = static_cast<std::uint64_t>(g()) * range;
    std::uint32_t low = static_cast<std::uint32_t>(product);
    if (low < range) {
        const std::uint32_t threshold = static_cast<std::uint32_t>(0u - range) % range;
        while (low < threshold) {
            product = static_cast<std::uint64_t>(g()) * range;
            low = static_cast<std::uint32_t>(product);
        }
    }
    return a + static_cast<std::int64_t>(product >> 32);
}

double StdNormal::operator()(StdMt19937& g) {
    double ret;
    if (saved_available_) {
        saved_available_ = false;
        ret = saved_;
    } else {
        double x, y, r2;
        do {
            x = 2.0 * std_canonical(g) - 1.0;
            y = 2.0 * std_canonical(g) - 1.0;
            r2 = x * x + y * y;
        } while (r2 > 1.0 || r2 == 0.0);
        const double mult = std::sqrt(-2 * lsm::r_log(r2) / r2);
        saved_ = x * mult;
        saved_available_ = true;
        ret = y * mult;
    }
    return ret * stddev_ + mean_;
}

void std_shuffle_positions(i64 n, StdMt19937& g, std::vector<i64>& pos) {
    pos.assign(n > 1 ? static_cast<std::size_t>(n - 1) : 0, 0);
    if (n <= 1) return;
    const std::uint64_t urange = static_cast<std::uint64_t>(n);
    if (0xffffffffull / urange >= urange) {
        i64 i = 1;
        if (urange % 2 == 0) pos[static_cast<std::size_t>(i++ - 1)] = std_uniform_int(g, 0, 1);
        while (i != n) {
            const std::int64_t b0 = i + 1;  // swap range
            const std::int64_t b1 = b0 + 1;
            const std::int64_t x = std_uniform_int(g, 0, b0 * b1 - 1);
            pos[static_cast<std::size_t>(i - 1)] = x / b1;
            pos[static_cast<std::size_t>(i)] = x % b1;
            i += 2;
        }
        return;
    }
    for (i64 i = 1; i < n; ++i) pos[static_cast<std::size_t>(i - 1)] = std_uniform_int(g, 0, i);
}

}  // namespace nlm
