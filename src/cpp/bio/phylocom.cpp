// SPDX-License-Identifier: GPL-3.0-or-later
//
// PhyloCom indices (see phylocom.hpp), following Biodiverse::Indices::PhyloCom
// and the expected values of Biodiverse::Tree (Tsirogiannis et al. 2012).

#include "bio/phylocom.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <set>
#include <stdexcept>

#include "bio/perlmath.hpp"
#include "lsm/parallel.hpp"

namespace bio {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------------------------
// Perl integer arithmetic (IV/UV until a product overflows, then NV)

struct PNum {
    bool is_int = true;
    bool neg = false;
    std::uint64_t mag = 0;
    double d = 0;
    static PNum of(std::int64_t v) {
        PNum p;
        p.neg = v < 0;
        p.mag = p.neg ? static_cast<std::uint64_t>(-(v + 1)) + 1 : static_cast<std::uint64_t>(v);
        return p;
    }
    double value() const {
        if (!is_int) return d;
        const double m = static_cast<double>(mag);
        return neg ? -m : m;
    }
};

PNum pmul(const PNum& a, const PNum& b) {
    if (a.is_int && b.is_int) {
        const bool over = a.mag != 0 && b.mag > UINT64_MAX / a.mag;
        const std::uint64_t m = a.mag * b.mag;
        const bool neg = (a.neg != b.neg) && m != 0;
        if (!over && (!neg || m <= (std::uint64_t(1) << 63))) {
            PNum r;
            r.neg = neg;
            r.mag = m;
            return r;
        }
    }
    PNum r;
    r.is_int = false;
    r.d = a.value() * b.value();
    return r;
}

// List::Util max (x, 0) and min (a, b)
inline double max0(double x) { return 0 < x ? x : 0.0; }
inline double lu_min(double a, double b) { return b < a ? b : a; }

}  // namespace

// ---------------------------------------------------------------------------------------------
// Math::Random::MT::Auto, 64-bit

namespace {
constexpr std::uint64_t MIX_HI = 0xFFFFFFFF80000000ULL, MIX_LO = 0x7FFFFFFFULL;
inline std::uint64_t twist(std::uint64_t u, std::uint64_t v) {
    return (((u & MIX_HI) | (v & MIX_LO)) >> 1) ^ ((v & 1ULL) ? 0xB5026F5AA96619E9ULL : 0ULL);
}
}  // namespace

MT64::MT64(const std::vector<std::uint64_t>& seed) {
    std::uint64_t* st = state_;
    st[0] = 19650218ULL;
    for (int ii = 1; ii < N; ++ii)
        st[ii] = 6364136223846793005ULL * (st[ii - 1] ^ (st[ii - 1] >> 62)) + static_cast<std::uint64_t>(ii);
    const int len = static_cast<int>(seed.size());
    int ii = 1, jj = 0;
    for (int kk = (N > len ? N : len); kk; --kk) {
        st[ii] = (st[ii] ^ ((st[ii - 1] ^ (st[ii - 1] >> 62)) * 3935559000370003845ULL)) + seed[static_cast<std::size_t>(jj)] +
                 static_cast<std::uint64_t>(jj);
        if (++ii >= N) {
            st[0] = st[N - 1];
            ii = 1;
        }
        if (++jj >= len) jj = 0;
    }
    for (int kk = N - 1; kk; --kk) {
        st[ii] = (st[ii] ^ ((st[ii - 1] ^ (st[ii - 1] >> 62)) * 2862933555777941757ULL)) - static_cast<std::uint64_t>(ii);
        if (++ii >= N) {
            st[0] = st[N - 1];
            ii = 1;
        }
    }
    st[0] = 1ULL << 63;
    left_ = 1;
    next_ = 0;
}

std::uint64_t MT64::algo() {
    std::uint64_t* st = state_;
    std::uint64_t* sn = &state_[2];
    std::uint64_t* sx = &state_[M];
    std::uint64_t n0 = state_[0], n1 = state_[1];
    int kk;
    for (kk = N - M + 1; --kk; n0 = n1, n1 = *sn++) *st++ = *sx++ ^ twist(n0, n1);
    sx = state_;
    for (kk = M; --kk; n0 = n1, n1 = *sn++) *st++ = *sx++ ^ twist(n0, n1);
    n1 = state_[0];
    *st = *sx ^ twist(n0, n1);
    next_ = 1;
    left_ = N;
    return n1;
}

namespace {
// _rand: (0, 1); _tan; _ln_gamma (Numerical Recipes), as in MRMA.xs
inline double mt_rand_open(std::uint64_t x) {
    return static_cast<double>(x >> 12) * 2.220446049250313080847263336181640625e-16 +
           1.1102230246251565404236316680908203125e-16;
}
inline double mt_rand_neg1(std::uint64_t x) {
    return static_cast<double>(static_cast<std::int64_t>(x) >> 11) * 2.220446049250313080847263336181640625e-16 +
           1.1102230246251565404236316680908203125e-16;
}
double mt_ln_gamma(double x) {
    double qq = x + 4.5;
    qq -= (x - 0.5) * perl_log(qq);
    const double ser = 1.000000000190015 + (76.18009172947146 / x) - (86.50532032941677 / (x + 1.0)) +
                       (24.01409824083091 / (x + 2.0)) - (1.231739572450155 / (x + 3.0)) +
                       (1.208650973866179e-3 / (x + 4.0)) - (5.395239384953e-6 / (x + 5.0));
    return perl_log(2.5066282746310005 * ser) - qq;
}
}  // namespace

std::int64_t MT64::binomial(double prob, std::int64_t trials) {
    if (prob < 0.0 || prob > 1.0 || trials < 0) throw std::invalid_argument("Invalid argument(s) to 'binomial'");
    const double p = (prob <= 0.5) ? prob : 1.0 - prob;
    std::int64_t ret;
    if (trials < 25) {
        ret = 0;
        for (std::int64_t ii = 1; ii <= trials; ++ii)
            if (mt_rand_open(irand()) < p) ++ret;
    } else {
        const double mean = p * static_cast<double>(trials);
        if (mean < 1.0) {
            double bound = 1.0;
            const double limit = perl_exp(-mean);
            for (ret = 0; ret < trials; ++ret) {
                bound *= mt_rand_open(irand());
                if (bound < limit) break;
            }
        } else {
            const double en = static_cast<double>(trials);
            const double pc = 1.0 - p;
            const double sq = std::sqrt(2.0 * mean * pc);
            if (trials != b_trials_) {
                b_trials_ = trials;
                b_term_ = mt_ln_gamma(en + 1.0);
            }
            if (p != b_prob_) {
                b_prob_ = p;
                b_plog_ = perl_log(p);
                b_pclog_ = perl_log(pc);
            }
            double em, bound, tang;
            do {
                do {
                    double x2, y2;
                    do {
                        const std::uint64_t x1 = irand();
                        const std::uint64_t y1 = irand();
                        x2 = mt_rand_neg1(x1);
                        y2 = mt_rand_neg1(y1);
                    } while (x2 * x2 + y2 * y2 > 1.0);
                    tang = y2 / x2;
                    em = (sq * tang) + mean;
                } while (em < 0.0 || em >= (en + 1.0));
                em = std::floor(em);
                bound = 1.2 * sq * (1.0 + tang * tang) *
                        perl_exp(b_term_ - mt_ln_gamma(em + 1.0) - mt_ln_gamma(en - em + 1.0) + em * b_plog_ +
                                 (en - em) * b_pclog_);
            } while (mt_rand_open(irand()) > bound);
            ret = static_cast<std::int64_t>(em);
        }
    }
    if (p < prob) ret = trials - ret;
    return ret;
}

std::vector<std::uint64_t> MT64::get_state() const {
    std::vector<std::uint64_t> s(state_, state_ + N);
    s.push_back(static_cast<std::uint64_t>(next_));
    s.push_back(static_cast<std::uint64_t>(left_));
    return s;
}

void MT64::set_state(const std::vector<std::uint64_t>& s) {
    if (s.size() != static_cast<std::size_t>(N) + 2) throw std::invalid_argument("invalid Mersenne Twister state");
    std::copy(s.begin(), s.begin() + N, state_);
    next_ = static_cast<int>(s[N]);
    left_ = static_cast<std::int64_t>(s[N + 1]);
}

std::uint64_t MT64::irand() {
    std::uint64_t x = (--left_ == 0) ? algo() : state_[next_++];
    x ^= (x >> 29) & 0x0000000555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);
    return x;
}

// ---------------------------------------------------------------------------------------------

PhyloCom::PhyloCom(const TreeView& t, bool vpd, std::uint64_t seed) : t_(t), vpd_(vpd), prng_({seed}) {
    const i32 n = t.n;
    tcount_.assign(static_cast<std::size_t>(n), 0);
    std::vector<i32> order(static_cast<std::size_t>(n));
    for (i32 i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
    std::stable_sort(order.begin(), order.end(), [&](i32 a, i32 b) {
        return t.depth[static_cast<std::size_t>(a)] > t.depth[static_cast<std::size_t>(b)];
    });
    for (i32 node : order) {
        auto& c = tcount_[static_cast<std::size_t>(node)];
        if (t.terminal[static_cast<std::size_t>(node)]) c = 1;
        for (i64 k = t.child_ptr[static_cast<std::size_t>(node)]; k < t.child_ptr[static_cast<std::size_t>(node) + 1]; ++k)
            c += tcount_[static_cast<std::size_t>(t.child_idx[static_cast<std::size_t>(k)])];
    }
    for (i32 i = 0; i < n; ++i)
        if (t.terminal[static_cast<std::size_t>(i)]) terminals_.push_back(i);
    s_ = static_cast<i32>(terminals_.size());

    // is_ultrametric: path sums of the terminals (name order) within 1e-3 of the first
    ultrametric = true;
    double len1 = 0;
    for (std::size_t k = 0; k < terminals_.size(); ++k) {
        i32 node = terminals_[k];
        double sum = t.length[static_cast<std::size_t>(node)];
        node = t.parent[static_cast<std::size_t>(node)];
        while (node >= 0) {
            sum += t.length[static_cast<std::size_t>(node)];
            node = t.parent[static_cast<std::size_t>(node)];
        }
        if (k == 0) {
            len1 = sum;
        } else if (std::fabs(sum - len1) > 1e-03) {
            ultrametric = false;
            break;
        }
    }

    // children sorted by terminal count (Sort::Key::nkeysort, stable)
    sib_rank_.assign(static_cast<std::size_t>(n), -1);
    for (i32 node = 0; node < n; ++node) {
        const i64 b = t.child_ptr[static_cast<std::size_t>(node)], e = t.child_ptr[static_cast<std::size_t>(node) + 1];
        if (e - b > 2) multifurcating = true;
        std::vector<i32> kids(t.child_idx.begin() + b, t.child_idx.begin() + e);
        std::stable_sort(kids.begin(), kids.end(), [&](i32 x, i32 y) {
            return tcount_[static_cast<std::size_t>(x)] < tcount_[static_cast<std::size_t>(y)];
        });
        for (std::size_t k = 0; k < kids.size(); ++k) sib_rank_[static_cast<std::size_t>(kids[k])] = static_cast<i32>(k);
    }
    um_fill_.assign(static_cast<std::size_t>(n), NaN);
}

double PhyloCom::path(i32 na, i32 nb) {
    const auto& depth = t_.depth;
    const auto& parent = t_.parent;
    const auto& length = t_.length;
    i32 a = na, b = nb, ca = -1, cb = -1;
    double sa = 0, sb = 0;
    while (depth[static_cast<std::size_t>(a)] > depth[static_cast<std::size_t>(b)]) {
        sa += length[static_cast<std::size_t>(a)];
        ca = a;
        a = parent[static_cast<std::size_t>(a)];
    }
    while (depth[static_cast<std::size_t>(b)] > depth[static_cast<std::size_t>(a)]) {
        sb += length[static_cast<std::size_t>(b)];
        cb = b;
        b = parent[static_cast<std::size_t>(b)];
    }
    while (a != b) {
        sa += length[static_cast<std::size_t>(a)];
        ca = a;
        a = parent[static_cast<std::size_t>(a)];
        sb += length[static_cast<std::size_t>(b)];
        cb = b;
        b = parent[static_cast<std::size_t>(b)];
    }
    if (!ultrametric) return sa + sb;
    const i32 lca = a;
    if (frozen_) return um_fill_[static_cast<std::size_t>(lca)];
    double& fill = um_fill_[static_cast<std::size_t>(lca)];
    if (!std::isnan(fill) && ca >= 0 && cb >= 0 && t_.terminal[static_cast<std::size_t>(na)] &&
        t_.terminal[static_cast<std::size_t>(nb)] &&
        std::abs(sib_rank_[static_cast<std::size_t>(ca)] - sib_rank_[static_cast<std::size_t>(cb)]) == 1)
        return fill;
    // twice one side, which then fills the pairs across adjacent children
    const double v = sa * 2;
    fill = v;
    return v;
}

void PhyloCom::touch(const std::vector<i32>& nodes) {
    const i32 n = t_.n;
    if (stamp_.size() != static_cast<std::size_t>(n)) {
        stamp_.assign(static_cast<std::size_t>(n), 0);
        visit_child_.assign(static_cast<std::size_t>(n), -1);
        visit_cum_.assign(static_cast<std::size_t>(n), 0);
    }
    ++stamp_id_;
    for (i32 a : nodes) {
        double run = 0;
        i32 prev = -1;
        for (i32 x = a; x >= 0; x = t_.parent[static_cast<std::size_t>(x)]) {
            const auto ux = static_cast<std::size_t>(x);
            if (stamp_[ux] == stamp_id_) {
                // reached from another child: the first visitor's path, doubled
                if (prev != visit_child_[ux] && std::isnan(um_fill_[ux])) um_fill_[ux] = visit_cum_[ux] * 2;
                break;
            }
            stamp_[ux] = stamp_id_;
            visit_child_[ux] = prev;
            visit_cum_[ux] = run;
            run += t_.length[ux];
            prev = x;
        }
    }
}

MpdResult PhyloCom::mpd_mntd(const std::vector<i32>& nodes, const std::vector<double>& wts, bool use_wts, int mode,
                             std::size_t n_on_tree) {
    MpdResult res;
    if (n_on_tree <= 1) {
        // default_mpd_mntd_results (no variance)
        res.s[0].n = 0;
        res.s[1].n = 0;
        return res;
    }
    std::vector<double> mpd, mntd, mpd_w, mntd_w;
    std::vector<double> here;
    const std::size_t k = nodes.size();
    // lengths do not depend on which label comes first (unless the cache can still change)
    const bool symmetric = !ultrametric || frozen_;
    std::vector<double> d;
    if (symmetric) {
        d.assign(k * k, 0);
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = i + 1; j < k; ++j) d[i * k + j] = d[j * k + i] = path(nodes[i], nodes[j]);
    }
    for (std::size_t i = 0; i < k; ++i) {
        here.clear();
        for (std::size_t j = 0; j < k; ++j)
            if (j != i) here.push_back(symmetric ? d[i * k + j] : path(nodes[i], nodes[j]));
        if (here.empty()) continue;
        mpd.insert(mpd.end(), here.begin(), here.end());
        double mn = here[0];
        for (std::size_t j = 1; j < here.size(); ++j) mn = lu_min(mn, here[j]);
        mntd.push_back(mn);
        if (use_wts) {
            for (std::size_t j = 0; j < k; ++j)
                if (j != i) mpd_w.push_back(wts[j] * wts[i]);
            mntd_w.push_back(wts[i]);
        }
    }
    auto lsum = [](const std::vector<double>& v) {
        if (v.empty()) return NaN;
        double s = v[0];
        for (std::size_t i = 1; i < v.size(); ++i) s += v[i];
        return s;
    };
    for (int p = 0; p < 2; ++p) {
        const std::vector<double>& path_v = p == 0 ? mntd : mpd;
        const std::vector<double>& w = p == 0 ? mntd_w : mpd_w;
        MpdStats& st = res.s[p];
        double n, mean;
        if (use_wts) {
            n = lsum(w);
            std::vector<double> prod(path_v.size());
            for (std::size_t i = 0; i < path_v.size(); ++i) prod[i] = path_v[i] * w[i];
            const double s = lsum(prod);
            mean = (std::isnan(n) || n == 0 || std::isnan(s)) ? NaN : s / n;
        } else {
            n = static_cast<double>(path_v.size());
            mean = n == 0 ? NaN : lsum(path_v) / n;
        }
        st.mean = mean;
        if (mode == 1) continue;
        double rmsd = NaN, variance = NaN;
        if (!std::isnan(n) && n != 0) {
            std::vector<double> sq(path_v.size());
            for (std::size_t i = 0; i < path_v.size(); ++i)
                sq[i] = use_wts ? perl_pow(path_v[i], 2) * w[i] : perl_pow(path_v[i], 2);
            const double sumsq = lsum(sq);
            rmsd = perl_sqrt(sumsq / n);
            variance = max0(sumsq / n - perl_pow(mean, 2));
        }
        st.variance = variance;
        if (mode == 2) continue;
        st.rmsd = rmsd;
        st.n = n;
        if (!path_v.empty()) {
            double mn = path_v[0], mx = path_v[0];
            for (std::size_t i = 1; i < path_v.size(); ++i) {
                mn = lu_min(mn, path_v[i]);
                if (!(path_v[i] < mx)) mx = path_v[i];
            }
            st.min = mn;
            st.max = mx;
        }
    }
    return res;
}

// ---------------------------------------------------------------------------------------------
// expected values

const std::vector<double>& PhyloCom::lgamma(i32 n) {
    for (i32 i = static_cast<i32>(lgamma_.size()); i <= n; ++i)
        lgamma_.push_back(lgamma_[static_cast<std::size_t>(i) - 1] + perl_log(static_cast<double>(i)));
    return lgamma_;
}

double PhyloCom::nri_expected_mean() {
    if (nri_mean_) return nri_mean_;
    double sum = 0;
    for (i32 node = 0; node < t_.n; ++node) {
        const double tc = tcount_[static_cast<std::size_t>(node)];
        sum += t_.length[static_cast<std::size_t>(node)] * tc * (s_ - tc);
    }
    nri_mean_ = sum * 2 / pmul(PNum::of(s_), PNum::of(s_ - 1)).value();
    return nri_mean_;
}

double PhyloCom::nri_expected_sd(i32 r) {
    if (r == 1) return NaN;
    const i32 s = s_;
    if (r == s) return 0;
    if (r > s) throw std::runtime_error("Cannot estimate MPD SD: more labels than tree terminals");
    const i32 n = t_.n;
    if (!have_tce_) {
        // last common ancestor of all tips (skipping single-child nodes below the root)
        std::vector<std::uint8_t> skip(static_cast<std::size_t>(n), 0);
        i32 lca = t_.root;
        auto nkids = [&](i32 node) {
            return t_.child_ptr[static_cast<std::size_t>(node) + 1] - t_.child_ptr[static_cast<std::size_t>(node)];
        };
        while (nkids(lca) == 1) {
            skip[static_cast<std::size_t>(lca)] = 1;
            lca = t_.child_idx[static_cast<std::size_t>(t_.child_ptr[static_cast<std::size_t>(lca)])];
        }
        // descendants of the LCA (name order) for get_nri_all_weights
        std::vector<std::uint8_t> below(static_cast<std::size_t>(n), 0);
        std::vector<i32> stack{lca}, pre;
        while (!stack.empty()) {
            const i32 node = stack.back();
            stack.pop_back();
            pre.push_back(node);
            for (i64 k = t_.child_ptr[static_cast<std::size_t>(node)]; k < t_.child_ptr[static_cast<std::size_t>(node) + 1]; ++k) {
                below[static_cast<std::size_t>(t_.child_idx[static_cast<std::size_t>(k)])] = 1;
                stack.push_back(t_.child_idx[static_cast<std::size_t>(k)]);
            }
        }
        double all_weights = NaN;
        for (i32 node = 0; node < n; ++node) {
            if (!below[static_cast<std::size_t>(node)]) continue;
            const double v = t_.length[static_cast<std::size_t>(node)] * tcount_[static_cast<std::size_t>(node)];
            all_weights = std::isnan(all_weights) ? v : all_weights + v;
        }
        // _calc_nri_tce_score: ancestor sums top-down, off-path sums bottom-up
        std::vector<double> anc1(static_cast<std::size_t>(n), 0), anc2(static_cast<std::size_t>(n), 0),
            off(static_cast<std::size_t>(n), 0), ret(static_cast<std::size_t>(n), 0), tce(static_cast<std::size_t>(n), NaN);
        for (i32 node : pre)
            for (i64 k = t_.child_ptr[static_cast<std::size_t>(node)]; k < t_.child_ptr[static_cast<std::size_t>(node) + 1]; ++k) {
                const i32 c = t_.child_idx[static_cast<std::size_t>(k)];
                const double ch_len = t_.length[static_cast<std::size_t>(c)];
                const i32 sl = tcount_[static_cast<std::size_t>(c)];
                anc1[static_cast<std::size_t>(c)] = anc1[static_cast<std::size_t>(node)] + ch_len * (s - sl);
                anc2[static_cast<std::size_t>(c)] = anc2[static_cast<std::size_t>(node)] + ch_len * sl;
            }
        for (auto it = pre.rbegin(); it != pre.rend(); ++it) {
            const i32 node = *it;
            double sum_off = 0;
            for (i64 k = t_.child_ptr[static_cast<std::size_t>(node)]; k < t_.child_ptr[static_cast<std::size_t>(node) + 1]; ++k)
                sum_off += ret[static_cast<std::size_t>(t_.child_idx[static_cast<std::size_t>(k)])];
            const i32 se = tcount_[static_cast<std::size_t>(node)];
            const double so = (s - se) * sum_off;
            const double sa = se * anc1[static_cast<std::size_t>(node)];
            const double si = se * (all_weights - anc2[static_cast<std::size_t>(node)] - sum_off);
            tce[static_cast<std::size_t>(node)] = so + sa + si;
            ret[static_cast<std::size_t>(node)] = sum_off + t_.length[static_cast<std::size_t>(node)] * se;
        }
        sum_tce_ = 0;
        sum_tcuu_ = 0;
        for (i32 node = 0; node < n; ++node) {
            if (skip[static_cast<std::size_t>(node)]) continue;
            const double v = tce[static_cast<std::size_t>(node)];
            sum_tce_ += v * t_.length[static_cast<std::size_t>(node)];
            if (t_.terminal[static_cast<std::size_t>(node)]) sum_tcuu_ += perl_pow(v, 2);
        }
        have_tce_ = true;
    }
    auto I = [](i64 v) { return PNum::of(v); };
    const PNum d1 = pmul(pmul(pmul(pmul(pmul(I(r), I(r - 1)), I(s)), I(s - 1)), I(s - 2)), I(s - 3));
    const PNum d2 = pmul(pmul(pmul(pmul(I(r), I(r - 1)), I(s)), I(s - 1)), I(s - 2));
    const PNum d3 = pmul(pmul(pmul(I(r), I(r - 1)), I(s)), I(s - 1));
    if (d1.value() == 0 || d2.value() == 0 || d3.value() == 0) throw std::runtime_error("Illegal division by zero");
    const double c1 = pmul(pmul(I(4), I(r - 2)), I(r - 3)).value() / d1.value();
    const double c2 = pmul(I(4), I(r - 2)).value() / d2.value();
    const double c3 = 4.0 / d3.value();
    const double c21 = c2 - c1;
    const double c123 = c1 - 2 * c2 + c3;
    const double mean = nri_expected_mean();
    const double tc = mean * pmul(I(s), I(s - 1)).value() / 2;
    const double variance = c1 * perl_pow(tc, 2) + c21 * sum_tcuu_ + c123 * sum_tce_ - perl_pow(mean, 2);
    if (variance < 0) return 0;
    return perl_sqrt(variance);
}

double PhyloCom::nti_expected_mean(i32 r) {
    auto it = nti_mean_cache_.find(r);
    if (it != nti_mean_cache_.end()) return it->second;
    const i32 s = s_;
    const auto& lg = lgamma(s);
    const double bnok_sr = lg[static_cast<std::size_t>(s)] - (lg[static_cast<std::size_t>(r)] + lg[static_cast<std::size_t>(s - r)]);
    const double exp_bnok_sr = perl_exp(-bnok_sr);
    const i32 sr1 = s - r + 1;
    auto cb1 = [&](i32 se) {
        if (se < sr1)
            return perl_exp(lg[static_cast<std::size_t>(s - se)] -
                            (lg[static_cast<std::size_t>(r - 1)] + lg[static_cast<std::size_t>(sr1 - se)]) - bnok_sr);
        if (se == sr1) return exp_bnok_sr;
        return 0.0;
    };
    double sum = 0;
    for (i32 node = 0; node < t_.n; ++node) {
        const i32 se = tcount_[static_cast<std::size_t>(node)];
        sum += t_.length[static_cast<std::size_t>(node)] * se * cb1(se);
    }
    const double expected = (2.0 / r) * sum;
    nti_mean_cache_[r] = expected;
    return expected;
}

double PhyloCom::nti_expected_sd(i32 r) {
    const i32 s = s_;
    if (r == s) return 0;
    const double exp_mean = nti_expected_mean(r);
    const i32 n = t_.n;
    const auto& lg = lgamma(s);
    const double bnok_sr = lg[static_cast<std::size_t>(s)] - (lg[static_cast<std::size_t>(r)] + lg[static_cast<std::size_t>(s - r)]);
    const double exp_bnok_sr = perl_exp(-bnok_sr);
    const i32 sr1 = s - r + 1, sr2 = s - r + 2, rminus2 = r - 2;
    auto cb1 = [&](i32 se) {
        if (se < sr1)
            return perl_exp(lg[static_cast<std::size_t>(s - se)] -
                            (lg[static_cast<std::size_t>(r - 1)] + lg[static_cast<std::size_t>(sr1 - se)]) - bnok_sr);
        if (se == sr1) return exp_bnok_sr;
        return 0.0;
    };
    auto cb2 = [&](i32 se, i32 sl) {
        const i32 sesl = se + sl;
        if (sr2 < sesl) return 0.0;
        if (sr2 == sesl) return exp_bnok_sr;
        return perl_exp(lg[static_cast<std::size_t>(s - sesl)] -
                        (lg[static_cast<std::size_t>(rminus2)] + lg[static_cast<std::size_t>(sr2 - sesl)]) - bnok_sr);
    };

    // nodes but the root, name order; by descending depth (stable)
    std::vector<i32> refs;
    for (i32 node = 0; node < n; ++node)
        if (node != t_.root) refs.push_back(node);
    std::map<i32, double> by_se;
    for (i32 node : refs) {
        const i32 te = tcount_[static_cast<std::size_t>(node)];
        auto f = by_se.find(te);
        const double v = t_.length[static_cast<std::size_t>(node)] * te;
        if (f == by_se.end())
            by_se.emplace(te, v);
        else
            f->second += v;
    }
    std::vector<i32> by_depth = refs;
    std::stable_sort(by_depth.begin(), by_depth.end(), [&](i32 a, i32 b) {
        return t_.depth[static_cast<std::size_t>(a)] > t_.depth[static_cast<std::size_t>(b)];
    });
    // _get_len_sum_by_tip_count_hash of every node (bottom-up)
    std::vector<std::map<i32, double>> lsh(static_cast<std::size_t>(n));
    {
        std::vector<i32> all(static_cast<std::size_t>(n));
        for (i32 i = 0; i < n; ++i) all[static_cast<std::size_t>(i)] = i;
        std::stable_sort(all.begin(), all.end(), [&](i32 a, i32 b) {
            return t_.depth[static_cast<std::size_t>(a)] > t_.depth[static_cast<std::size_t>(b)];
        });
        for (i32 node : all) {
            auto& h = lsh[static_cast<std::size_t>(node)];
            h[tcount_[static_cast<std::size_t>(node)]] = t_.length[static_cast<std::size_t>(node)];
            for (i64 k = t_.child_ptr[static_cast<std::size_t>(node)]; k < t_.child_ptr[static_cast<std::size_t>(node) + 1]; ++k)
                for (const auto& kv : lsh[static_cast<std::size_t>(t_.child_idx[static_cast<std::size_t>(k)])]) {
                    auto f = h.find(kv.first);
                    if (f == h.end())
                        h.emplace(kv.first, kv.second);
                    else
                        f->second += kv.second;
                }
        }
    }
    // keys of a tip-count hash in Perl's (string) order
    auto string_order = [](const std::map<i32, double>& h) {
        std::vector<std::pair<std::string, i32>> keys;
        for (const auto& kv : h) keys.emplace_back(std::to_string(kv.first), kv.first);
        std::sort(keys.begin(), keys.end());
        std::vector<i32> out;
        for (const auto& k : keys) out.push_back(k.second);
        return out;
    };

    double sum_subtree = 0, sum_subtract = 0, sum_self = 0, sum_self_third_case = 0, sum_third_case = 0,
           sum_same_class_third_case = 0;
    std::vector<double> sum_pr(static_cast<std::size_t>(n), 0);
    for (i32 node : by_depth) {
        const double length = t_.length[static_cast<std::size_t>(node)];
        const i32 se = tcount_[static_cast<std::size_t>(node)];
        const double mhyperg = cb1(se);
        const double mhyperg2 = cb2(se, se);
        sum_pr[static_cast<std::size_t>(node)] = length * se;
        for (i64 k = t_.child_ptr[static_cast<std::size_t>(node)]; k < t_.child_ptr[static_cast<std::size_t>(node) + 1]; ++k) {
            const i32 child = t_.child_idx[static_cast<std::size_t>(k)];
            const double spr = sum_pr[static_cast<std::size_t>(child)];
            sum_subtree += length * spr * mhyperg;
            sum_pr[static_cast<std::size_t>(node)] += spr;
            const auto& h = lsh[static_cast<std::size_t>(child)];
            for (i32 sl : string_order(h)) sum_subtract += se * length * sl * h.at(sl) * cb2(se, sl);
        }
        const double len2 = perl_pow(length, 2);
        const double se2 = static_cast<double>(pmul(PNum::of(se), PNum::of(se)).value());
        sum_subtree += len2 * se * mhyperg;
        sum_subtract += mhyperg2 * len2 * se2;
        sum_self += se * len2 * mhyperg;
        sum_self_third_case += len2 * se2 * mhyperg2;
    }
    std::vector<i32> keys;
    for (const auto& kv : by_se) keys.push_back(kv.first);
    for (std::size_t jj = 0; jj < keys.size(); ++jj) {
        const i32 se = keys[jj];
        if (s - se - se >= 0) sum_same_class_third_case += perl_pow(by_se[se], 2) * cb2(se, se);
        if (jj == 0) continue;
        for (std::size_t jji = 0; jji < jj; ++jji) {
            const i32 sl = keys[jji];
            sum_third_case += by_se[se] * by_se[sl] * cb2(se, sl);
        }
    }
    sum_same_class_third_case = (sum_same_class_third_case - sum_self_third_case) / 2 + sum_self_third_case;
    sum_third_case += sum_same_class_third_case;
    const double total_sum = 2 * (sum_third_case - sum_subtract + sum_subtree) - sum_self;
    const double expected = 4 * total_sum / pmul(PNum::of(r), PNum::of(r)).value() - perl_pow(exp_mean, 2);
    return expected < 0 ? 0.0 : perl_sqrt(expected);
}

double PhyloCom::mean_nearest_neighbour_distance() {
    if (have_nn_) return mean_nn_;
    const i32 n = t_.n;
    auto nkids = [&](i32 node) {
        return t_.child_ptr[static_cast<std::size_t>(node) + 1] - t_.child_ptr[static_cast<std::size_t>(node)];
    };
    i32 root = t_.root;
    while (nkids(root) == 1) root = t_.child_idx[static_cast<std::size_t>(t_.child_ptr[static_cast<std::size_t>(root)])];
    // get_shortest_path_length_to_terminals, bottom-up
    std::vector<double> shortest(static_cast<std::size_t>(n), NaN);
    {
        std::vector<i32> all(static_cast<std::size_t>(n));
        for (i32 i = 0; i < n; ++i) all[static_cast<std::size_t>(i)] = i;
        std::stable_sort(all.begin(), all.end(), [&](i32 a, i32 b) {
            return t_.depth[static_cast<std::size_t>(a)] > t_.depth[static_cast<std::size_t>(b)];
        });
        for (i32 node : all) {
            double v = t_.length[static_cast<std::size_t>(node)];
            if (!t_.terminal[static_cast<std::size_t>(node)]) {
                const i64 b = t_.child_ptr[static_cast<std::size_t>(node)], e = t_.child_ptr[static_cast<std::size_t>(node) + 1];
                double mn = shortest[static_cast<std::size_t>(t_.child_idx[static_cast<std::size_t>(b)])];
                for (i64 k = b + 1; k < e; ++k) mn = lu_min(mn, shortest[static_cast<std::size_t>(t_.child_idx[static_cast<std::size_t>(k)])]);
                v += mn;
            }
            shortest[static_cast<std::size_t>(node)] = v;
        }
    }
    std::vector<double> sib_min(static_cast<std::size_t>(n), NaN);
    std::vector<std::uint8_t> have_sib(static_cast<std::size_t>(n), 0);
    double total = NaN;
    for (i32 term : terminals_) {
        // cumulative path lengths from the terminal (index 0) to the root
        std::vector<double> cum;
        double run = 0;
        bool first = true;
        for (i32 x = term; x >= 0; x = t_.parent[static_cast<std::size_t>(x)]) {
            run = first ? t_.length[static_cast<std::size_t>(x)] : run + t_.length[static_cast<std::size_t>(x)];
            first = false;
            cum.push_back(run);
        }
        std::size_t idx = static_cast<std::size_t>(-1);  // incremented before use
        double min_dist = NaN;
        i32 node = term;
        while (node != root) {
            ++idx;
            if (!have_sib[static_cast<std::size_t>(node)]) {
                const i32 par = t_.parent[static_cast<std::size_t>(node)];
                bool any = false;
                double mn = NaN;
                for (i64 k = t_.child_ptr[static_cast<std::size_t>(par)]; k < t_.child_ptr[static_cast<std::size_t>(par) + 1]; ++k) {
                    const i32 sib = t_.child_idx[static_cast<std::size_t>(k)];
                    if (sib == node) continue;
                    const double v = shortest[static_cast<std::size_t>(sib)];
                    mn = any ? lu_min(mn, v) : v;
                    any = true;
                }
                if (!any) {
                    node = par;
                    continue;
                }
                sib_min[static_cast<std::size_t>(node)] = mn;
                have_sib[static_cast<std::size_t>(node)] = 1;
            }
            const double cand = sib_min[static_cast<std::size_t>(node)] + cum[idx];
            min_dist = std::isnan(min_dist) ? cand : lu_min(min_dist, cand);
            if (idx + 1 < cum.size() && min_dist < cum[idx + 1]) break;
            node = t_.parent[static_cast<std::size_t>(node)];
        }
        total = std::isnan(total) ? min_dist : total + min_dist;
    }
    mean_nn_ = total / static_cast<double>(terminals_.size());
    have_nn_ = true;
    return mean_nn_;
}

NriNti PhyloCom::compute_expected(std::size_t kk) {
    const i32 k = static_cast<i32>(kk);
    NriNti r;
    r.nri_mean = nri_expected_mean();
    r.nri_sd = nri_expected_sd(k);
    bool do_mntd = true;
    const bool do_vpd = k > 2 && vpd_;
    if (ultrametric) {
        r.nti_mean = nti_expected_mean(k);
        r.nti_sd = nti_expected_sd(k);
        r.n = 0;
        if (!do_vpd) return r;
        do_mntd = false;
    }
    if (do_mntd && k == s_) {
        r.nti_mean = mean_nearest_neighbour_distance();
        r.nti_sd = 0;
        r.n = 0;
        if (!do_vpd) return r;
        do_mntd = false;
    }
    // min (4999, C(s, k))
    double iterations = 1;
    for (i32 i = 1; i <= k && iterations <= 4999; ++i) iterations = iterations * (s_ - k + i) / i;
    iterations = std::min(4999.0, iterations);

    std::set<std::vector<i32>> seen;
    std::deque<double> conv[4];  // mntd mean, mntd sd, vpd mean, vpd sd
    double mntd_sum = 0, mntd_sq = 0, mntd_mean = NaN, mntd_sd = NaN;
    double vpd_sum = 0, vpd_sq = 0, vpd_mean = NaN, vpd_sd = NaN;
    const std::vector<double> ones(static_cast<std::size_t>(k), 1.0);
    std::size_t n = 0;
    // Draws in batches: the sets drawn one after the other, their distances in
    // parallel (lengths are fixed unless the tree is ultrametric), then taken in
    // order; the generator goes back to where Biodiverse stops drawing.
    const bool par = parallel_sampling && (!ultrametric || frozen_);
    const std::size_t batch = par ? 64 : 1;
    std::vector<std::vector<i32>> targets(batch);
    std::vector<MT64> after(batch, prng_);
    std::vector<MpdResult> vals(batch);
    std::size_t next = batch;  // position in the current batch
    bool stop = false;
    while (!stop && static_cast<double>(n) < iterations) {
        if (next == batch) {
            for (std::size_t b = 0; b < batch; ++b) {
                std::vector<i32> shuffled = terminals_;
                prng_.shuffle(shuffled);
                targets[b].assign(shuffled.begin(), shuffled.begin() + k);
                std::sort(targets[b].begin(), targets[b].end());
                after[b] = prng_;
            }
            if (par) {
                lsm::parallel_for_dynamic(true, static_cast<std::int64_t>(batch), [&](std::int64_t b) {
                    vals[static_cast<std::size_t>(b)] =
                        mpd_mntd(targets[static_cast<std::size_t>(b)], ones, false, do_vpd ? 2 : 1, terminals_.size());
                });
            }
            next = 0;
        }
        const std::size_t b = next++;
        if (!seen.insert(targets[b]).second) continue;
        ++n;
        const MpdResult m = par ? vals[b] : mpd_mntd(targets[b], ones, false, do_vpd ? 2 : 1, terminals_.size());
        const double dn = static_cast<double>(n);
        if (do_mntd) {
            const double val = m.s[0].mean;
            mntd_sum += val;
            mntd_sq += perl_pow(val, 2);
            mntd_mean = mntd_sum / dn;
            mntd_sd = perl_sqrt(max0(mntd_sq / dn - perl_pow(mntd_mean, 2)));
            conv[0].push_back(mntd_mean);
            conv[1].push_back(mntd_sd);
        }
        if (do_vpd) {
            const double val = m.s[1].variance;
            vpd_sum += val;
            vpd_sq += perl_pow(val, 2);
            vpd_mean = vpd_sum / dn;
            vpd_sd = perl_sqrt(max0(vpd_sq / dn - perl_pow(vpd_mean, 2)));
            conv[2].push_back(vpd_mean);
            conv[3].push_back(vpd_sd);
        }
        if (n > 100) {
            bool converged = true;
            for (auto& a : conv) {
                if (a.empty()) continue;
                a.pop_front();
                if (a.empty()) continue;
                const auto mm = std::minmax_element(a.begin(), a.end());
                const double mn = *mm.first, mx = *mm.second;
                const double ratio = mx != 0 ? 1 - mn / mx : 0;
                if (ratio > 0.005) converged = false;
            }
            if (converged) stop = true;
        }
        if (stop || static_cast<double>(n) >= iterations) prng_ = after[b];
    }
    if (do_mntd) {
        r.nti_mean = mntd_mean;
        r.nti_sd = mntd_sd;
        r.n = static_cast<double>(n);
    }
    if (do_vpd) {
        r.vpd_mean = vpd_mean;
        r.vpd_sd = vpd_sd;
        r.vpd_n = static_cast<double>(n);
    }
    return r;
}

const NriNti* PhyloCom::expected_step(std::size_t k) {
    if (k < 2) return nullptr;
    auto it = cache_.find(k);
    const bool recalc = it == cache_.end() || (vpd_ && std::isnan(it->second.vpd_n)) ||
                        (!vpd_ && std::isnan(it->second.nri_mean));
    if (recalc) {
        const NriNti w = compute_expected(k);
        if (it == cache_.end()) {
            it = cache_.emplace(k, w).first;
        } else {
            // keep existing values
            NriNti& c = it->second;
            auto keep = [](double& a, double b) {
                if (std::isnan(a)) a = b;
            };
            keep(c.nri_mean, w.nri_mean);
            keep(c.nri_sd, w.nri_sd);
            keep(c.nti_mean, w.nti_mean);
            keep(c.nti_sd, w.nti_sd);
            keep(c.n, w.n);
            keep(c.vpd_mean, w.vpd_mean);
            keep(c.vpd_sd, w.vpd_sd);
            keep(c.vpd_n, w.vpd_n);
        }
    }
    return &it->second;
}

const NriNti* PhyloCom::cached(std::size_t k) const {
    if (k < 2) return nullptr;
    auto it = cache_.find(k);
    return it == cache_.end() ? nullptr : &it->second;
}

}  // namespace bio
