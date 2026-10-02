// SPDX-License-Identifier: GPL-3.0-or-later
//
// maxnet 0.1.4 (R/maxnet.R, maxnet.formula.R, hinge.R, thresholds.R,
// categorical.R, maxnet.default.regularization.R, predict.maxnet.R) in C++.
// R's semantics where they decide bits: sum(), mean() and sd() accumulate
// in 80-bit long double, exp() and log() are mingw-w64's x87 code (all in
// lsm/rmath.hpp), seq() and approx() as R computes them, 10^seq(4, 0,
// length.out = 200) from a table (R uses mingw's powl), and the knots that
// predict.maxnet() reads back from the column names (as.character(): 15
// significant digits).

#include "sdm/maxnet.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "sdm/lognet.hpp"

namespace sdm {
namespace maxnet {

namespace {

// 10^seq(4, 0, length.out = 200) as R computes it (R_pow -> mingw-w64's powl),
// written by R with sprintf("%a").
constexpr double kPow10[200] = {
    0x1.388p+13, 0x1.2a5dba9a15fe7p+13, 0x1.1cdf1b3ce6568p+13, 0x1.0ffcbb1bbaf73p+13,
    0x1.03af891cbe1a9p+13, 0x1.efe18bf16f6dfp+12, 0x1.d974010f578e7p+12, 0x1.c40a2470a39d4p+12,
    0x1.af983760eaf0cp+12, 0x1.9c130328cd45dp+12, 0x1.896fd2e769312p+12, 0x1.77a46db309469p+12,
    0x1.66a710fdcf71cp+12, 0x1.566e6b3b4b3d1p+12, 0x1.46f196c40b5e2p+12, 0x1.382814f45ce84p+12,
    0x1.2a09c9838aeafp+12, 0x1.1c8ef611104dap+12, 0x1.0fb035e54b4aap+12, 0x1.036679e35e299p+12,
    0x1.ef56095407b38p+11, 0x1.d8eecdc271869p+11, 0x1.c38af765618b4p+11, 0x1.af1ecad6559f3p+11,
    0x1.9b9f148591cadp+11, 0x1.8901229551adcp+11, 0x1.773abefc1c96ap+11, 0x1.664229eb04ac8p+11,
    0x1.560e1474bec1ap+11, 0x1.46959b72a3fd8p+11, 0x1.37d042a4d084fp+11, 0x1.29b5f00ab2a9ep+11,
    0x1.1c3ee7717d2abp+11, 0x1.0f63c6360c935p+11, 0x1.031d7f37ec07cp+11, 0x1.eecaadf681721p+10,
    0x1.d869bfeef9e83p+10, 0x1.c30bee21a8a76p+10, 0x1.aea5807504185p+10, 0x1.9b2b468011568p+10,
    0x1.88929167501cp+10, 0x1.76d12e00b54f6p+10, 0x1.65dd5f3b7d8fep+10, 0x1.55add8c8c6266p+10,
    0x1.4639ba01fe643p+10, 0x1.3778890a6560ep+10, 0x1.29622e28e8588p+10, 0x1.1beeef57d4f8bp+10,
    0x1.0f176c07f04f4p+10, 0x1.02d499149f537p+10, 0x1.ee3f79cdd1cfcp+9, 0x1.d7e4d78a65b41p+9,
    0x1.c28d089b68048p+9, 0x1.ae2c583359fbfp+9, 0x1.9ab7990f1ed0dp+9, 0x1.88241f54a1a1fp+9,
    0x1.7667bab876079p+9, 0x1.6578b0e73d8dp+9, 0x1.554db82fc1554p+9, 0x1.45ddf26ad2c72p+9,
    0x1.3720e81e27fc2p+9, 0x1.290e83d788f2ep+9, 0x1.1b9f0dbdc18b8p+9, 0x1.0ecb2754e9b0ap+9,
    0x1.028bc773b14c1p+9, 0x1.edb46ccef10dep+8, 0x1.d760148a2ce21p+8, 0x1.c20e46c8918a8p+8,
    0x1.adb35207bd9e1p+8, 0x1.9a440c298fb58p+8, 0x1.87b5cc5485dcap+8, 0x1.75fe651b03b09p+8,
    0x1.65141ee64a544p+8, 0x1.54edb2a2125d6p+8, 0x1.458244a5db666p+8, 0x1.36c95fd926cb6p+8,
    0x1.28baf10ff352ep+8, 0x1.1b4f429cee7f9p+8, 0x1.0e7ef816ed9d2p+8, 0x1.02430a4f5cd06p+8,
    0x1.ed2986eeda883p+7, 0x1.d6db76e3ca60dp+7, 0x1.c18fa89f19f58p+7, 0x1.ad3a6de898059p+7,
    0x1.99d09fc63c14ap+7, 0x1.8747985e3edf9p+7, 0x1.75952d2005958p+7, 0x1.64afa930abd49p+7,
    0x1.548dc8181d723p+7, 0x1.4526b0abd48e3p+7, 0x1.3671f03472376p+7, 0x1.286775cb882fep+7,
    0x1.1aff8def0938fp+7, 0x1.0e32de47f2ae7p+7, 0x1.01fa61a1de5efp+7, 0x1.ec9ec8228cb41p+6,
    0x1.d656fe8cbc145p+6, 0x1.c1112e14f8d4fp+6, 0x1.acc1abcc54ed8p+6, 0x1.995d53dbfe91fp+6,
    0x1.86d983691133cp+6, 0x1.752c12bf255aap+6, 0x1.644b4fbe6c3acp+6, 0x1.542df88a48ec3p+6,
    0x1.44cb36757c962p+6, 0x1.361a99291c9c8p+6, 0x1.28141203aa1e4p+6, 0x1.1aafefadc0e34p+6,
    0x1.0de6d9e1f1304p+6, 0x1.01b1cd6574157p+6, 0x1.ec14305f091efp+5, 0x1.d5d2ab7a82d5ap+5,
    0x1.c092d720288afp+5, 0x1.ac490ba962c16p+5, 0x1.98ea2861b463bp+5, 0x1.866b8d6c43d6ap+5,
    0x1.74c315f00efccp+5, 0x1.63e7128797f11p+5, 0x1.53ce43f0fd462p+5, 0x1.446fd5fb93e04p+5,
    0x1.35c35ab03a49ap+5, 0x1.27c0c5b1bd8ecp+5, 0x1.1a6067d2c670fp+5, 0x1.0d9aeadee320fp+5,
    0x1.01694d945db05p+5, 0x1.eb89bf99546fbp+4, 0x1.d54e7da2a271cp+4, 0x1.c014a3b6a64b6p+4,
    0x1.abd08d76329e6p+4, 0x1.98771d4e3d525p+4, 0x1.85fdb65f203b4p+4, 0x1.745a36aa70d06p+4,
    0x1.6382f1843d9e5p+4, 0x1.536eaa44a51ep+4, 0x1.44148f36dcd8cp+4, 0x1.356c34c2e1815p+4,
    0x1.276d90cf28ce1p+4, 0x1.1a10f657cc9acp+4, 0x1.0d4f1138c4308p+4, 0x1.0120e228dc8ap+4,
    0x1.eaff75c67662bp+3, 0x1.d4ca74faa1a8cp+3, 0x1.bf9693ce721b2p+3, 0x1.ab5831293852p+3,
    0x1.980432987bb77p+3, 0x1.858ffe38f246ep+3, 0x1.73f174e5fb814p+3, 0x1.631eecac6e257p+3,
    0x1.530f2b7dad33dp+3, 0x1.43b962201bf54p+3, 0x1.3515275a2a76cp+3, 0x1.271a73555403ep+3,
    0x1.19c19b3687df7p+3, 0x1.0d034ce991c06p+3, 0x1.00d88b1d339adp+3, 0x1.ea7552db79ca9p+2,
    0x1.d44691780a2cap+2, 0x1.bf18a75d8ecf6p+2, 0x1.aadff6b8ea598p+2, 0x1.97916837547d2p+2,
    0x1.852264f108521p+2, 0x1.7388d09a62113p+2, 0x1.62bb03f83ca4ep+2, 0x1.52afc79484695p+2,
    0x1.435e4eb017b46p+2, 0x1.34be326f2f4fp+2, 0x1.26c76d3da933dp+2, 0x1.19725668ae831p+2,
    0x1.0cb79deb4ae3p+2, 0x1.0090486ba7787p+2, 0x1.e9eb56cd6c8f9p+1, 0x1.d3c2d31068a26p+1,
    0x1.be9ade5a020cap+1, 0x1.aa67de1bc1e1p+1, 0x1.971ebe21af1d6p+1, 0x1.84b4ea7eb3279p+1,
    0x1.732049bf59d9p+1, 0x1.6257375fbe75bp+1, 0x1.52507e819bc13p+1, 0x1.430354df989cfp+1,
    0x1.346755fb0c1fep+1, 0x1.26747e81943abp+1, 0x1.192327e7f88e8p+1, 0x1.0c6c0437f05afp+1,
    0x1.00481a0e7e553p+1, 0x1.e96181915fadfp+0, 0x1.d33f39b94c9e3p+0, 0x1.be1d38b9d4461p+0,
    0x1.a9efe7483ac2ap+0, 0x1.96ac344e75a16p+0, 0x1.84478ed946039p+0, 0x1.72b7e04c9a85ap+0,
    0x1.61f386db0b2b4p+0, 0x1.51f1503d665e9p+0, 0x1.42a874a7693dap+0, 0x1.341091f6deef7p+0,
    0x1.2621a71a82cfap+0, 0x1.18d40fae1fceep+0, 0x1.0c207fc9849bp+0, 0x1p+0,
};

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// seq(from, to, length.out = len)[i] (seq.default: from + i * ((to - from) / (len - 1)),
// the ends exact).
double r_seq_at(double from, double to, int len, int i) {
    if (from == to || i == 0) return from;
    if (i == len - 1) return to;
    const double n1 = static_cast<double>(len - 1);
    return from + static_cast<double>(i) * ((to - from) / n1);
}

// approx(x, y, v, rule = 2)$y (R's approx1, linear)
double r_approx(const std::vector<double>& x, const std::vector<double>& y, double v) {
    std::size_t i = 0, j = x.size() - 1;
    if (v < x[i]) return y[i];
    if (v > x[j]) return y[j];
    while (i < j - 1) {
        const std::size_t ij = (i + j) / 2;
        if (v < x[ij]) {
            j = ij;
        } else {
            i = ij;
        }
    }
    if (v == x[j]) return y[j];
    if (v == x[i]) return y[i];
    return y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]));
}

// hingeval(x, min, max) = pmin(1, pmax(0, (x - min) / (max - min)))
inline double hingeval(double x, double a, double b) { return std::min(1.0, std::max(0.0, (x - a) / (b - a))); }

// Rows of a table: out[t] is about row rows[t] (or t).
struct Rows {
    const Table& d;
    const i64* rows;
    i64 row(i64 t) const { return rows ? rows[t] : t; }
    double x(std::int32_t v, i64 t) const { return d.cont[static_cast<i64>(v) * d.n + row(t)]; }
    std::int32_t c(std::int32_t v, i64 t) const { return d.cat[static_cast<i64>(v) * d.n + row(t)]; }
};

// The value of a feature in one row; `rounded`: with the knots of the names
// (predict.maxnet), else the exact ones (model.matrix in maxnet()).
inline double feature_value(const Feature& f, double xa, double xb, std::int32_t code, bool rounded) {
    switch (f.kind) {
        case Kind::Linear: return xa;
        case Kind::Quadratic: return xa * xa;  // R's x^2 is x * x
        case Kind::Hinge: return rounded ? hingeval(xa, f.pk1, f.pk2) : hingeval(xa, f.k1, f.k2);
        case Kind::Threshold: return xa >= (rounded ? f.pk1 : f.k1) ? 1.0 : 0.0;
        case Kind::Categorical: return code == f.var2 ? 1.0 : 0.0;
        case Kind::Product: return xa * xb;
    }
    return 0.0;
}

// -- R's as.character() of doubles ----------------------------------------

#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
#define SDM_MAXNET_TO_CHARS 1
#endif

std::string format_double(double x, bool scientific, int precision) {
    char buf[512];
#if defined(SDM_MAXNET_TO_CHARS)
    const auto res = std::to_chars(buf, buf + sizeof buf, x,
                                   scientific ? std::chars_format::scientific : std::chars_format::fixed, precision);
    return std::string(buf, res.ptr);
#else
    std::snprintf(buf, sizeof buf, scientific ? "%.*e" : "%.*f", precision, x);
    return std::string(buf);
#endif
}

double parse_double(const std::string& s) {
#if defined(SDM_MAXNET_TO_CHARS)
    double v = 0;
    std::from_chars(s.data(), s.data() + s.size(), v);
    return v;
#else
    return std::strtod(s.c_str(), nullptr);
#endif
}

}  // namespace

// R's formatReal() with 15 digits: the fewest significant digits (<= 15)
// that give the value rounded to 15 digits, then fixed notation unless the
// scientific one is narrower.
std::string r_as_character(double x) {
    if (std::isnan(x)) return "NaN";
    if (std::isinf(x)) return x > 0 ? "Inf" : "-Inf";
    if (x == 0) return "0";
    const std::string e15 = format_double(x, true, 14);  // [-]d.dddddddddddddde[+-]x
    const std::size_t epos = e15.find('e');
    const int kp = std::atoi(e15.c_str() + epos + 1);
    std::string digits;
    for (std::size_t i = 0; i < epos; ++i)
        if (e15[i] >= '0' && e15[i] <= '9') digits.push_back(e15[i]);
    int nsig = static_cast<int>(digits.size());
    while (nsig > 1 && digits[static_cast<std::size_t>(nsig - 1)] == '0') --nsig;
    const int neg = x < 0 ? 1 : 0;
    const int rgt = std::max(0, nsig - kp - 1);
    const int left = kp >= 0 ? kp + 1 : 1;
    const int w_fixed = neg + left + rgt + (rgt > 0 ? 1 : 0);
    const int e = (kp >= 100 || kp <= -99) ? 2 : 1;
    const int w_sci = neg + (nsig > 1 ? 1 : 0) + (nsig - 1) + 4 + e;
    if (w_fixed <= w_sci) return format_double(x, false, rgt);
    return format_double(x, true, nsig - 1);
}

namespace {

void set_knot_names(Feature& f, int count) {
    f.s1 = r_as_character(f.k1);
    f.pk1 = parse_double(f.s1);
    if (count > 1) {
        f.s2 = r_as_character(f.k2);
        f.pk2 = parse_double(f.s2);
    }
}

std::uint64_t mix(std::uint64_t h, std::uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    return h * 0xFF51AFD7ED558CCDULL;
}

std::uint64_t value_bits(double x) {
    if (x == 0) x = 0.0;  // -0 == 0 for identical()
    std::uint64_t b;
    std::memcpy(&b, &x, sizeof b);
    return b;
}

// Link (mm %*% betas + alpha) for the rows `rows` (or all) of `d`: R's dgemv,
// y_i = y_i + b_t f_t(i) in the order of the betas, then + alpha.
void link_rows(const Predictor& m, const Table& d, const i64* rows, i64 n, bool clamp, double* out) {
    const std::size_t nb = m.betas.size();
    // the variables the formula uses: R drops rows where they are missing
    std::vector<char> used_cont(static_cast<std::size_t>(d.n_cont), 0), used_cat(static_cast<std::size_t>(d.n_cat), 0);
    for (const Feature& f : m.features) {
        if (f.kind == Kind::Categorical) {
            used_cat[static_cast<std::size_t>(f.var)] = 1;
        } else {
            used_cont[static_cast<std::size_t>(f.var)] = 1;
            if (f.kind == Kind::Product) used_cont[static_cast<std::size_t>(f.var2)] = 1;
        }
    }
    constexpr i64 kBlock = 256;
    const i64 nblocks = (n + kBlock - 1) / kBlock;
    const Rows src{d, rows};
    lsm::parallel_for(n * static_cast<i64>(nb + 1) >= 20000 && nblocks > 1, nblocks, [&](i64 bi) {
        const i64 lo = bi * kBlock, hi = std::min(n, lo + kBlock), len = hi - lo;
        double y[kBlock];
        double xa[kBlock], xb[kBlock];
        std::int32_t code[kBlock];
        char bad[kBlock];
        for (i64 t = 0; t < len; ++t) {
            y[t] = 0.0;
            xa[t] = xb[t] = 0.0;
            code[t] = 0;
            bad[t] = 0;
        }
        for (std::int32_t v = 0; v < d.n_cont; ++v) {
            if (!used_cont[static_cast<std::size_t>(v)]) continue;
            for (i64 t = 0; t < len; ++t)
                if (std::isnan(src.x(v, lo + t))) bad[t] = 1;
        }
        for (std::int32_t v = 0; v < d.n_cat; ++v) {
            if (!used_cat[static_cast<std::size_t>(v)]) continue;
            for (i64 t = 0; t < len; ++t)
                if (src.c(v, lo + t) == -1) bad[t] = 1;
        }
        for (std::size_t j = 0; j < nb; ++j) {
            const Feature& f = m.features[j];
            const double bj = m.betas[j];
            if (f.kind == Kind::Categorical) {
                for (i64 t = 0; t < len; ++t) code[t] = src.c(f.var, lo + t);
            } else {
                const double vmin = clamp ? m.var_min[static_cast<std::size_t>(f.var)] : 0.0;
                const double vmax = clamp ? m.var_max[static_cast<std::size_t>(f.var)] : 0.0;
                for (i64 t = 0; t < len; ++t) {
                    double xv = src.x(f.var, lo + t);
                    if (clamp) xv = std::min(std::max(xv, vmin), vmax);  // pmin(pmax(x, varmin), varmax)
                    xa[t] = xv;
                }
                if (f.kind == Kind::Product) {
                    const double wmin = clamp ? m.var_min[static_cast<std::size_t>(f.var2)] : 0.0;
                    const double wmax = clamp ? m.var_max[static_cast<std::size_t>(f.var2)] : 0.0;
                    for (i64 t = 0; t < len; ++t) {
                        double xv = src.x(f.var2, lo + t);
                        if (clamp) xv = std::min(std::max(xv, wmin), wmax);
                        xb[t] = xv;
                    }
                }
            }
            const double fmin = m.feature_min[j], fmax = m.feature_max[j];
            for (i64 t = 0; t < len; ++t) {
                double fv = feature_value(f, xa[t], xb[t], code[t], true);
                if (clamp) fv = std::min(std::max(fv, fmin), fmax);
                y[t] = y[t] + bj * fv;
            }
        }
        for (i64 t = 0; t < len; ++t) out[lo + t] = bad[t] ? kNaN : y[t] + m.alpha;
    });
}

}  // namespace

Predictor predictor(const Model& m) {
    Predictor p;
    for (std::size_t t = 0; t < m.beta_index.size(); ++t) {
        const auto j = static_cast<std::size_t>(m.beta_index[t]);
        p.features.push_back(m.features[j]);
        p.betas.push_back(m.betas[t]);
        p.feature_min.push_back(m.feature_min[j]);
        p.feature_max.push_back(m.feature_max[j]);
    }
    p.var_min = m.var_min;
    p.var_max = m.var_max;
    p.alpha = m.alpha;
    p.entropy = m.entropy;
    return p;
}

void predict(const Predictor& m, const Table& d, bool clamp, Output type, double* out) {
    const i64 n = d.n;
    if (static_cast<i64>(m.var_min.size()) != d.n_cont || static_cast<i64>(m.var_max.size()) != d.n_cont)
        throw std::invalid_argument("maxnet predict: the number of continuous variables differs from the model's");
    link_rows(m, d, nullptr, n, clamp, out);
    if (type == Output::Link) return;
    const double ent = m.entropy;
    lsm::parallel_for(n >= 20000, (n + 1023) / 1024, [&](i64 bi) {
        const i64 lo = bi * 1024, hi = std::min(n, lo + 1024);
        for (i64 i = lo; i < hi; ++i) {
            const double link = out[i];
            if (std::isnan(link)) continue;
            switch (type) {
                case Output::Exponential: out[i] = lsm::r_exp(link); break;
                case Output::Cloglog: out[i] = 1.0 - lsm::r_exp(0.0 - lsm::r_exp(ent + link)); break;
                case Output::Logistic: out[i] = 1.0 / (1.0 + lsm::r_exp(-ent - link)); break;
                default: break;
            }
        }
    });
}

Model fit(const Table& d, const double* p, const Settings& s) {
    const i64 n0 = d.n;
    if (n0 < 2) throw std::invalid_argument("maxnet: need presences and background points");
    if (static_cast<std::int32_t>(d.n_levels.size()) != d.n_cat)
        throw std::invalid_argument("maxnet: n_levels must give the levels of every factor");
    i64 np = 0;
    for (i64 i = 0; i < n0; ++i) {
        if (p[i] != 0.0 && p[i] != 1.0) throw std::invalid_argument("maxnet: p must be 1 (presence) or 0 (background)");
        np += p[i] == 1.0;
    }
    if (np < 2) throw std::invalid_argument("maxnet: need at least two presences");
    if (np == n0) throw std::invalid_argument("maxnet: need background points (p == 0)");
    for (i64 k = 0; k < n0 * d.n_cont; ++k)
        if (std::isnan(d.cont[k])) throw std::invalid_argument("maxnet: NA values in data table. Please remove them and rerun.");
    for (std::int32_t v = 0; v < d.n_cat; ++v)
        for (i64 i = 0; i < n0; ++i) {
            const std::int32_t c = d.cat[static_cast<i64>(v) * n0 + i];
            if (c < 0 || c >= d.n_levels[static_cast<std::size_t>(v)])
                throw std::invalid_argument("maxnet: NA values in data table. Please remove them and rerun.");
        }
    for (char c : s.classes)
        if (s.classes != "default" && std::string("lqpht").find(c) == std::string::npos)
            throw std::invalid_argument("maxnet: classes must be 'default' or letters among l, q, p, h, t");

    const auto t_start = std::chrono::steady_clock::now();
    Model m;
    // -- addsamplestobackground: presences not already among the background
    // rows are appended to it (exact comparison of the values) --
    std::vector<i64> rows(static_cast<std::size_t>(n0));
    std::iota(rows.begin(), rows.end(), i64(0));
    std::vector<double> pa(p, p + n0);
    if (s.add_samples_to_background) {
        std::vector<std::uint64_t> h(static_cast<std::size_t>(n0));
        for (i64 i = 0; i < n0; ++i) {
            std::uint64_t x = 0x84222325CBF29CE4ULL;
            for (std::int32_t v = 0; v < d.n_cont; ++v) x = mix(x, value_bits(d.cont[static_cast<i64>(v) * n0 + i]));
            for (std::int32_t v = 0; v < d.n_cat; ++v)
                x = mix(x, static_cast<std::uint64_t>(d.cat[static_cast<i64>(v) * n0 + i]));
            h[static_cast<std::size_t>(i)] = x;
        }
        std::vector<std::pair<std::uint64_t, i64>> bg;
        for (i64 i = 0; i < n0; ++i)
            if (p[i] == 0.0) bg.emplace_back(h[static_cast<std::size_t>(i)], i);
        std::sort(bg.begin(), bg.end());
        auto same = [&](i64 a, i64 b) {
            for (std::int32_t v = 0; v < d.n_cont; ++v)
                if (!(d.cont[static_cast<i64>(v) * n0 + a] == d.cont[static_cast<i64>(v) * n0 + b])) return false;
            for (std::int32_t v = 0; v < d.n_cat; ++v)
                if (d.cat[static_cast<i64>(v) * n0 + a] != d.cat[static_cast<i64>(v) * n0 + b]) return false;
            return true;
        };
        for (i64 i = 0; i < n0; ++i) {
            if (p[i] != 1.0) continue;
            const std::uint64_t hi = h[static_cast<std::size_t>(i)];
            auto it = std::lower_bound(bg.begin(), bg.end(), std::make_pair(hi, std::numeric_limits<i64>::min()));
            bool found = false;
            for (; it != bg.end() && it->first == hi && !found; ++it) found = same(i, it->second);
            if (!found) {
                rows.push_back(i);
                pa.push_back(0.0);
            }
        }
    }
    const i64 n = static_cast<i64>(rows.size());
    m.n_added = n - n0;
    m.n_presences = np;
    m.n_rows = n;
    const Rows src{d, rows.data()};

    // -- maxnet.formula: the feature classes and the columns of model.matrix --
    std::string classes = s.classes;
    if (classes == "default") classes = np < 10 ? "l" : np < 15 ? "lq" : np < 80 ? "lqh" : "lqph";
    m.classes = classes;
    auto has = [&](char c) { return classes.find(c) != std::string::npos; };
    const std::int32_t nc = d.n_cont;
    m.var_min.resize(static_cast<std::size_t>(nc));
    m.var_max.resize(static_cast<std::size_t>(nc));
    for (std::int32_t v = 0; v < nc; ++v) {
        const double* xv = d.cont + static_cast<i64>(v) * n0;
        double mn = xv[0], mx = xv[0];
        for (i64 i = 1; i < n0; ++i) {
            mn = std::min(mn, xv[i]);
            mx = std::max(mx, xv[i]);
        }
        m.var_min[static_cast<std::size_t>(v)] = mn;
        m.var_max[static_cast<std::size_t>(v)] = mx;
    }
    std::vector<Feature>& feats = m.features;
    auto add = [&](Kind k, std::int32_t a, std::int32_t b) {
        Feature f;
        f.kind = k;
        f.var = a;
        f.var2 = b;
        feats.push_back(f);
        return &feats.back();
    };
    if (nc > 0) {
        if (has('l'))
            for (std::int32_t v = 0; v < nc; ++v) add(Kind::Linear, v, 0);
        if (has('q'))
            for (std::int32_t v = 0; v < nc; ++v) add(Kind::Quadratic, v, 0);
        if (has('h')) {
            for (std::int32_t v = 0; v < nc; ++v) {
                const double mn = m.var_min[static_cast<std::size_t>(v)], mx = m.var_max[static_cast<std::size_t>(v)];
                if (mn == mx)
                    throw std::invalid_argument("maxnet: a constant variable cannot have hinge features "
                                                "(glmnet: x has missing values)");
                // hinge(): seq(min, max, length = 50); 49 "left" ramps from each
                // knot to max, then 49 ramps from min to each knot
                for (int i = 0; i < 49; ++i) {
                    Feature* f = add(Kind::Hinge, v, 0);
                    f->k1 = r_seq_at(mn, mx, 50, i);
                    f->k2 = mx;
                    set_knot_names(*f, 2);
                }
                for (int i = 1; i < 50; ++i) {
                    Feature* f = add(Kind::Hinge, v, 1);
                    f->k1 = mn;
                    f->k2 = r_seq_at(mn, mx, 50, i);
                    set_knot_names(*f, 2);
                }
            }
        }
        if (has('t')) {
            // thresholds(): seq(min, max, length = 52)[2:50 + 1]
            for (std::int32_t v = 0; v < nc; ++v) {
                const double mn = m.var_min[static_cast<std::size_t>(v)], mx = m.var_max[static_cast<std::size_t>(v)];
                for (int t = 2; t <= 50; ++t) {
                    Feature* f = add(Kind::Threshold, v, 0);
                    f->k1 = r_seq_at(mn, mx, 52, t);
                    set_knot_names(*f, 1);
                }
            }
        }
    }
    for (std::int32_t v = 0; v < d.n_cat; ++v)
        for (std::int32_t l = 0; l < d.n_levels[static_cast<std::size_t>(v)]; ++l) add(Kind::Categorical, v, l);
    // products last (terms sorted by degree): m[lower.tri(m)] of outer(cont, cont)
    if (nc > 0 && has('p'))
        for (std::int32_t j = 0; j < nc; ++j)
            for (std::int32_t i = j + 1; i < nc; ++i) add(Kind::Product, j, i);
    const i64 ni = static_cast<i64>(feats.size());
    if (ni == 0) throw std::invalid_argument("maxnet: no features (no variables for these classes)");

    // -- the model matrix, its ranges and maxnet.default.regularization --
    std::vector<i64> prow;  // presence rows (the original presences, in order)
    for (i64 t = 0; t < n; ++t)
        if (pa[static_cast<std::size_t>(t)] == 1.0) prow.push_back(t);
    bool any_q = false, any_p = false;
    for (const Feature& f : feats) {
        any_q |= f.kind == Kind::Quadratic;
        any_p |= f.kind == Kind::Product;
    }
    const std::vector<double> lx{0, 10, 30, 100}, ly{1, 1, 0.2, 0.05};
    const std::vector<double> qx{0, 10, 17, 30, 100}, qy{1.3, 0.8, 0.5, 0.25, 0.05};
    const std::vector<double> px{0, 10, 17, 30, 100}, py{2.6, 1.6, 0.9, 0.55, 0.05};
    const std::vector<double> hx{0, 1}, hy{0.5, 0.5}, cx{0, 10, 17}, cy{0.65, 0.5, 0.25}, tx{0, 100}, ty{2.0, 1.0};
    const std::vector<double>& lqx = any_p ? px : any_q ? qx : lx;
    const std::vector<double>& lqy = any_p ? py : any_q ? qy : ly;
    const double dnp = static_cast<double>(np);
    const double sqnp = std::sqrt(dnp);
    const double reg_lqp = r_approx(lqx, lqy, dnp) / sqnp, reg_h = r_approx(hx, hy, dnp) / sqnp;
    const double reg_c = r_approx(cx, cy, dnp) / sqnp, reg_t = r_approx(tx, ty, dnp) / sqnp;

    std::unique_ptr<double[]> X(new double[static_cast<std::size_t>(n * ni)]);
    m.feature_min.resize(static_cast<std::size_t>(ni));
    m.feature_max.resize(static_cast<std::size_t>(ni));
    m.penalty.resize(static_cast<std::size_t>(ni));
    lsm::parallel_for_dynamic(n * ni >= 100000, ni, [&](i64 j) {
        const Feature& f = feats[static_cast<std::size_t>(j)];
        double* xj = X.get() + j * n;
        if (f.kind == Kind::Categorical) {
            for (i64 t = 0; t < n; ++t) xj[t] = src.c(f.var, t) == f.var2 ? 1.0 : 0.0;
        } else if (f.kind == Kind::Product) {
            for (i64 t = 0; t < n; ++t) xj[t] = src.x(f.var, t) * src.x(f.var2, t);
        } else {
            for (i64 t = 0; t < n; ++t) xj[t] = feature_value(f, src.x(f.var, t), 0.0, 0, false);
        }
        double mn = xj[0], mx = xj[0];
        for (i64 t = 1; t < n; ++t) {
            mn = std::min(mn, xj[t]);
            mx = std::max(mx, xj[t]);
        }
        const auto ju = static_cast<std::size_t>(j);
        m.feature_min[ju] = mn;
        m.feature_max[ju] = mx;
        std::vector<double> pv(prow.size());
        for (std::size_t t = 0; t < prow.size(); ++t) pv[t] = xj[prow[t]];
        const i64 npi = static_cast<i64>(pv.size());
        const double sd = lsm::r_sd(pv.data(), npi, lsm::r_mean(pv.data(), npi));
        double classreg = reg_lqp, hmindev = 0.0, tmindev = 0.0;
        if (f.kind == Kind::Hinge) {
            classreg = reg_h;
            // more regularisation for extreme hinges
            const double sdh = std::max(sd, 1.0 / sqnp);
            hmindev = sdh * .5 / sqnp;
        } else if (f.kind == Kind::Categorical) {
            classreg = reg_c;
        } else if (f.kind == Kind::Threshold) {
            classreg = reg_t;
            // and for thresholds that are all 0 or all 1 on the presences
            const double sum = lsm::r_sum(pv.data(), npi);
            if (sum == 0.0 || sum == static_cast<double>(npi)) tmindev = 1.0;
        }
        double reg = 0.001 * (mx - mn);
        reg = std::max(reg, hmindev);
        reg = std::max(reg, tmindev);
        reg = std::max(reg, sd * classreg);
        m.penalty[ju] = reg * s.regmult;
    });

    // -- weights, lambda, glmnet --
    std::vector<double> weights(static_cast<std::size_t>(n));
    for (i64 t = 0; t < n; ++t) {
        const double pt = pa[static_cast<std::size_t>(t)];
        weights[static_cast<std::size_t>(t)] = pt + (1.0 - pt) * 100.0;
    }
    {
        const double S = lsm::r_sum(m.penalty.data(), ni);
        const double P = lsm::r_sum(pa.data(), n);
        const double W = lsm::r_sum(weights.data(), n);
        m.lambda.resize(200);
        for (int k = 0; k < 200; ++k)
            m.lambda[static_cast<std::size_t>(k)] = kPow10[k] * S / static_cast<double>(ni) * P / W;
    }
    const bool profile = std::getenv("LANDSCHAFT_PROFILE") != nullptr;
    const auto t_features = std::chrono::steady_clock::now();
    if (profile)
        std::fprintf(stderr, "maxnet: %lld rows x %lld features, model matrix and regularisation %.3f s\n",
                     static_cast<long long>(n), static_cast<long long>(ni),
                     std::chrono::duration<double>(t_features - t_start).count());
    LognetResult path = lognet_path(X.get(), n, ni, pa.data(), weights.data(), m.penalty.data(), m.lambda.data(), 200,
                                    1e-7, s.maxit, 1e-8);
    X.reset();
    m.n_passes = path.n_passes;
    if (path.n_lambda < 200) {
        std::string msg = "Error: glmnet failed to complete regularization path.  Model may be infeasible.";
        if (!s.add_samples_to_background) msg += "  Try re-running with addsamplestobackground=T.";
        throw std::runtime_error(msg);
    }
    for (i64 j = 0; j < ni; ++j) {
        const double bj = path.beta[static_cast<std::size_t>(j)];
        if (bj != 0.0) {
            m.beta_index.push_back(static_cast<std::int32_t>(j));
            m.betas.push_back(bj);
        }
    }

    // -- alpha and entropy: the exponential output over the background rows
    // (with the appended presences), unclamped, made to sum to one --
    {
        Predictor pr = predictor(m);
        pr.alpha = 0.0;
        std::vector<i64> bg;
        for (i64 t = 0; t < n; ++t)
            if (pa[static_cast<std::size_t>(t)] == 0.0) bg.push_back(rows[static_cast<std::size_t>(t)]);
        const i64 nbg = static_cast<i64>(bg.size());
        std::vector<double> rr(static_cast<std::size_t>(nbg));
        link_rows(pr, d, bg.data(), nbg, false, rr.data());
        for (double& v : rr) v = lsm::r_exp(v);
        const double S = lsm::r_sum(rr.data(), nbg);
        const double ent = lsm::sum_of(0, nbg, [&](i64 t) {
            const double raw = rr[static_cast<std::size_t>(t)] / S;
            return raw * lsm::r_log(raw);
        });
        m.entropy = -ent;
        m.alpha = -lsm::r_log(S);
    }
    if (profile)
        std::fprintf(stderr, "maxnet: glmnet and alpha %.3f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t_features).count());

    // -- samplemeans: means of the continuous variables and the majority
    // level of each factor over the presences --
    {
        std::vector<double> pv(prow.size());
        for (std::int32_t v = 0; v < nc; ++v) {
            for (std::size_t t = 0; t < prow.size(); ++t) pv[t] = src.x(v, prow[t]);
            m.sample_means.push_back(lsm::r_mean(pv.data(), static_cast<i64>(pv.size())));
        }
        for (std::int32_t v = 0; v < d.n_cat; ++v) {
            std::vector<i64> count(static_cast<std::size_t>(d.n_levels[static_cast<std::size_t>(v)]), 0);
            for (const i64 t : prow) ++count[static_cast<std::size_t>(src.c(v, t))];
            m.majority.push_back(
                static_cast<std::int32_t>(std::max_element(count.begin(), count.end()) - count.begin()));
        }
    }
    return m;
}

}  // namespace maxnet
}  // namespace sdm
