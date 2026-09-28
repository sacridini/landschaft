// SPDX-License-Identifier: GPL-3.0-or-later
//
// Biodiverse's indices (see indices.hpp). Each calc_* below follows the Perl
// of Biodiverse::Indices::* line by line; comments point out where Perl's
// behaviour needs care (undef, eval around divisions, hash order).

#include "bio/indices.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

#include "bio/perlmath.hpp"
#include "lsm/parallel.hpp"

namespace bio {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

inline bool defined(double v) { return !std::isnan(v); }
// Perl's truth of a number (undef -> false)
inline bool truthy(double v) { return defined(v) && v != 0; }
// Perl's numeric value of a possibly undefined scalar
inline double num(double v) { return defined(v) ? v : 0.0; }
// eval { a / b }: undef when dividing by zero (or by undef)
inline double safe_div(double a, double b) { return (num(b) == 0) ? NaN : num(a) / num(b); }

// Perl's stringification of a number (%.15g), for the order of numeric hash keys.
std::string perl_str(double v) {
    if (v == 0) return "0";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.15g", v);
    return buf;
}

// Distinct values of `vals` in the order Perl iterates a hash keyed by them
// (sorted keys, as strings), with their multiplicities.
std::vector<std::pair<double, double>> numeric_key_counts(const std::vector<double>& vals) {
    std::map<std::string, std::pair<double, double>> m;
    for (double v : vals) {
        auto& e = m[perl_str(v)];
        e.first = v;
        e.second += 1;
    }
    std::vector<std::pair<double, double>> out;
    out.reserve(m.size());
    for (auto& kv : m) out.push_back(kv.second);
    return out;
}

// Statistics::Descriptive-like accumulators (Biodiverse::Statistics).
struct Stats {
    std::vector<double> data;
    double sum = 0, sumsq = 0;
    void add(const std::vector<double>& v) {
        for (double x : v) sum += x;
        for (double x : v) sumsq += x * x;
        data.insert(data.end(), v.begin(), v.end());
    }
    double count() const { return static_cast<double>(data.size()); }
    double mean() const { return data.empty() ? NaN : sum / count(); }
    double variance() const {
        const double n = count();
        if (n == 0) return NaN;
        if (n == 1) return 0;
        const double m = mean();
        double var = sumsq - n * (m * m);
        if (var < 0)
            var = 0;
        else
            var /= n - 1;
        return var;
    }
    double sd() const {
        const double v = variance();
        return defined(v) ? perl_sqrt(v) : NaN;
    }
    std::vector<double> percentiles(const std::vector<double>& ps) {
        std::vector<double> out;
        if (data.empty()) return out;
        std::vector<double> sorted = data;
        std::stable_sort(sorted.begin(), sorted.end());
        const double n = count();
        for (double p : ps) {
            p = p > 100 ? 100 : (p < 0 ? 0 : p);
            const auto idx = static_cast<i64>(std::trunc(0.5 + (n - 1) * p / 100));
            out.push_back(sorted[static_cast<std::size_t>(idx)]);
        }
        return out;
    }
};

// ---------------------------------------------------------------------------------------------
// registry

using K = KeyKind;

std::vector<CalcInfo> make_registry() {
    std::vector<CalcInfo> r;
    auto add = [&](std::string name, std::vector<std::string> scalars,
                   std::vector<std::pair<std::string, KeyKind>> lists, i32 nbr, bool proc = false) {
        CalcInfo c;
        c.name = std::move(name);
        c.scalars = std::move(scalars);
        c.lists = std::move(lists);
        c.uses_nbr_lists = nbr;
        c.uses_processing_element = proc;
        r.push_back(std::move(c));
    };
    // Lists and counts
    add("calc_richness", {"RICHNESS_ALL", "RICHNESS_SET1", "RICHNESS_SET2"}, {}, 1);
    add("calc_redundancy", {"REDUNDANCY_ALL", "REDUNDANCY_SET1", "REDUNDANCY_SET2"}, {}, 1);
    add("calc_abc_counts", {"ABC_A", "ABC_ABC", "ABC_B", "ABC_C"}, {}, 2);
    add("calc_d", {"ABC_D"}, {}, 1);
    add("calc_elements_used", {"EL_COUNT_ALL", "EL_COUNT_SET1", "EL_COUNT_SET2"}, {}, 1);
    add("calc_nonempty_elements_used", {"EL_COUNT_NONEMPTY_ALL", "EL_COUNT_NONEMPTY_SET1", "EL_COUNT_NONEMPTY_SET2"},
        {}, 1);
    add("calc_element_lists_used", {},
        {{"EL_LIST_ALL", K::GroupArray}, {"EL_LIST_SET1", K::Group}, {"EL_LIST_SET2", K::Group}}, 1);
    add("calc_element_lists_used_as_arrays", {},
        {{"EL_ARRAY_ALL", K::GroupArray}, {"EL_ARRAY_SET1", K::GroupArray}, {"EL_ARRAY_SET2", K::GroupArray}}, 1);
    add("calc_label_count_quantile_position", {}, {{"LABEL_COUNT_RANK_PCT", K::Label}}, 1, true);
    add("calc_local_range_lists", {},
        {{"ABC2_LABELS_ALL", K::Label}, {"ABC2_LABELS_SET1", K::Label}, {"ABC2_LABELS_SET2", K::Label}}, 1);
    add("calc_local_range_stats",
        {"ABC2_MEAN_ALL", "ABC2_MEAN_SET1", "ABC2_MEAN_SET2", "ABC2_SD_ALL", "ABC2_SD_SET1", "ABC2_SD_SET2"}, {}, 1);
    add("calc_local_sample_count_lists", {},
        {{"ABC3_LABELS_ALL", K::Label}, {"ABC3_LABELS_SET1", K::Label}, {"ABC3_LABELS_SET2", K::Label}}, 1);
    add("calc_local_sample_count_quantiles", {},
        {{"ABC3_QUANTILES_ALL", K::Text}, {"ABC3_QUANTILES_SET1", K::Text}, {"ABC3_QUANTILES_SET2", K::Text}}, 1);
    add("calc_local_sample_count_stats",
        {"ABC3_MEAN_ALL", "ABC3_MEAN_SET1", "ABC3_MEAN_SET2", "ABC3_SD_ALL", "ABC3_SD_SET1", "ABC3_SD_SET2",
         "ABC3_SUM_ALL", "ABC3_SUM_SET1", "ABC3_SUM_SET2"},
        {}, 1);
    // Taxonomic dissimilarity and comparison
    add("calc_beta_diversity", {"BETA_2"}, {}, 2);
    add("calc_bray_curtis", {"BC_A", "BC_B", "BC_W", "BRAY_CURTIS"}, {}, 2);
    add("calc_bray_curtis_norm_by_gp_counts", {"BCN_A", "BCN_B", "BCN_W", "BRAY_CURTIS_NORM"}, {}, 2);
    add("calc_jaccard", {"JACCARD"}, {}, 2);
    add("calc_kulczynski2", {"KULCZYNSKI2"}, {}, 2);
    add("calc_nestedness_resultant", {"NEST_RESULTANT"}, {}, 2);
    add("calc_rw_turnover", {"RW_TURNOVER", "RW_TURNOVER_A", "RW_TURNOVER_B", "RW_TURNOVER_C"}, {}, 2);
    add("calc_s2", {"S2"}, {}, 2);
    add("calc_simpson_shannon", {"SHANNON_E", "SHANNON_H", "SHANNON_HMAX", "SIMPSON_D"}, {}, 1);
    add("calc_sorenson", {"SORENSON"}, {}, 2);
    add("calc_tx_rao_qe", {"TX_RAO_QE", "TX_RAO_TN"}, {{"TX_RAO_TLABELS", K::Label}}, 1);
    // Endemism
    add("calc_endemism_central", {"ENDC_CWE", "ENDC_RICHNESS", "ENDC_SINGLE", "ENDC_WE"}, {}, 1);
    add("calc_endemism_central_lists", {}, {{"ENDC_RANGELIST", K::Label}, {"ENDC_WTLIST", K::Label}}, 1);
    add("calc_endemism_central_normalised", {"ENDC_CWE_NORM", "ENDC_WE_NORM"}, {}, 1);
    add("calc_endemism_whole", {"ENDW_CWE", "ENDW_RICHNESS", "ENDW_SINGLE", "ENDW_WE"}, {}, 1);
    add("calc_endemism_whole_lists", {}, {{"ENDW_RANGELIST", K::Label}, {"ENDW_WTLIST", K::Label}}, 1);
    add("calc_endemism_whole_normalised", {"ENDW_CWE_NORM", "ENDW_WE_NORM"}, {}, 1);
    add("calc_endemism_absolute", {"END_ABS1", "END_ABS1_P", "END_ABS2", "END_ABS2_P", "END_ABS_ALL", "END_ABS_ALL_P"},
        {}, 1);
    add("calc_endemism_absolute_lists", {},
        {{"END_ABS1_LIST", K::Label}, {"END_ABS2_LIST", K::Label}, {"END_ABS_ALL_LIST", K::Label}}, 1);
    // Rarity
    add("calc_rarity_central", {"RAREC_CWE", "RAREC_RICHNESS", "RAREC_WE"}, {}, 1);
    add("calc_rarity_central_lists", {}, {{"RAREC_RANGELIST", K::Label}, {"RAREC_WTLIST", K::Label}}, 1);
    add("calc_rarity_whole", {"RAREW_CWE", "RAREW_RICHNESS", "RAREW_WE"}, {}, 1);
    add("calc_rarity_whole_lists", {}, {{"RAREW_RANGELIST", K::Label}, {"RAREW_WTLIST", K::Label}}, 1);
    // Richness estimators
    add("calc_chao1",
        {"CHAO1_CI_LOWER", "CHAO1_CI_UPPER", "CHAO1_ESTIMATE", "CHAO1_F1_COUNT", "CHAO1_F2_COUNT", "CHAO1_SE",
         "CHAO1_UNDETECTED", "CHAO1_VARIANCE"},
        {{"CHAO1_META", K::Text}}, 1);
    add("calc_chao2",
        {"CHAO2_CI_LOWER", "CHAO2_CI_UPPER", "CHAO2_ESTIMATE", "CHAO2_Q1_COUNT", "CHAO2_Q2_COUNT", "CHAO2_SE",
         "CHAO2_UNDETECTED", "CHAO2_VARIANCE"},
        {{"CHAO2_META", K::Text}}, 1);
    add("calc_ace",
        {"ACE_CI_LOWER", "ACE_CI_UPPER", "ACE_ESTIMATE", "ACE_ESTIMATE_USED_CHAO", "ACE_INFREQUENT_COUNT", "ACE_SE",
         "ACE_UNDETECTED", "ACE_VARIANCE"},
        {}, 1);
    add("calc_ice",
        {"ICE_CI_LOWER", "ICE_CI_UPPER", "ICE_ESTIMATE", "ICE_ESTIMATE_USED_CHAO", "ICE_INFREQUENT_COUNT", "ICE_SE",
         "ICE_UNDETECTED", "ICE_VARIANCE"},
        {}, 1);
    add("calc_hurlbert_es", {}, {{"HURLBERT_ES", K::Number}}, 1);
    // Hierarchical labels (lists per label level are added at run time)
    add("calc_endemism_central_hier_part", {}, {}, 1);
    add("calc_endemism_whole_hier_part", {}, {}, 1);
    add("calc_hierarchical_label_ratios", {},
        {{"HIER_A", K::Number}, {"HIER_ARAT", K::Number}, {"HIER_ASUM", K::Number}, {"HIER_ASUMRAT", K::Number},
         {"HIER_B", K::Number}, {"HIER_BRAT", K::Number}, {"HIER_C", K::Number}, {"HIER_CRAT", K::Number}},
        2);
    return r;
}

// ---------------------------------------------------------------------------------------------
// per-group aggregates

struct Abc {
    std::vector<i32> g1, g2;           // neighbour groups (ascending)
    bool has_set2 = false;             // a second neighbour set exists (maybe empty)
    std::vector<i32> l1, l2, lall;     // labels (ascending)
    std::vector<std::uint8_t> in1, in2;
    std::vector<i32> c1, c2;           // abc2: groups per label
    std::vector<double> s1, s2;        // abc3: samples per label
    i64 A = 0, B = 0, C = 0, ABC = 0;

    void init(i64 n_labels) {
        in1.assign(static_cast<std::size_t>(n_labels), 0);
        in2.assign(static_cast<std::size_t>(n_labels), 0);
        c1.assign(static_cast<std::size_t>(n_labels), 0);
        c2.assign(static_cast<std::size_t>(n_labels), 0);
        s1.assign(static_cast<std::size_t>(n_labels), 0);
        s2.assign(static_cast<std::size_t>(n_labels), 0);
    }
    void clear() {
        for (i32 l : l1) {
            in1[l] = 0;
            c1[l] = 0;
            s1[l] = 0;
        }
        for (i32 l : l2) {
            in2[l] = 0;
            c2[l] = 0;
            s2[l] = 0;
        }
        l1.clear();
        l2.clear();
        lall.clear();
    }
    double all_count(i32 l) const { return static_cast<double>(c1[l] + c2[l]); }
    double all_samples(i32 l) const {
        if (in1[l] && in2[l]) return s1[l] + s2[l];
        return in1[l] ? s1[l] : s2[l];
    }
};

void aggregate(const BaseDataView& bd, const std::vector<i32>& groups, std::vector<i32>& labels,
               std::vector<std::uint8_t>& in, std::vector<i32>& cnt, std::vector<double>& smp) {
    for (i32 g : groups) {
        for (i64 k = bd.indptr[g]; k < bd.indptr[g + 1]; ++k) {
            const i32 l = bd.indices[static_cast<std::size_t>(k)];
            const double v = bd.counts[static_cast<std::size_t>(k)];
            if (!in[l]) {
                in[l] = 1;
                labels.push_back(l);
                cnt[l] = 1;
                smp[l] = v;
            } else {
                ++cnt[l];
                smp[l] += v;
            }
        }
    }
    std::sort(labels.begin(), labels.end());
}

// ---------------------------------------------------------------------------------------------
// the calculations

struct Out {
    // scalars in the order of CalcInfo::scalars
    std::vector<double>* scalars = nullptr;
    // list entries per list of the calc
    std::vector<std::vector<std::pair<double, double>>>* lists = nullptr;
    void set(std::size_t k, double v) { (*scalars)[k] = v; }
    void put(std::size_t list, double key, double value) { (*lists)[list].emplace_back(key, value); }
};

struct Ctx {
    const BaseDataView& bd;
    const Abc& abc;
    i32 proc;
    const std::vector<std::string>& texts;
    std::function<i32(const std::string&)> text_id;
};

struct Endemism {
    double cwe = NaN, we = NaN, richness = 0, single = NaN;
    std::vector<std::pair<i32, double>> wts, ranges;
};

// _calc_endemism: local ranges from abc2 (endemism) or abc3 (rarity)
Endemism calc_endemism(const Ctx& c, bool central, bool rarity) {
    Endemism e;
    const auto& labels = central ? c.abc.l1 : c.abc.lall;
    double endemism = NaN, rosauer = NaN;
    for (i32 l : labels) {
        const double range = rarity ? c.bd.label_abundance[l] : c.bd.label_range[l];
        const double local = rarity ? c.abc.all_samples(l) : c.abc.all_count(l);
        const double wt = local / range;
        endemism = num(endemism) + wt;
        e.wts.emplace_back(l, wt);
        e.ranges.emplace_back(l, range);
        rosauer = num(rosauer) + 1 / range;
    }
    e.richness = static_cast<double>(labels.size());
    e.we = endemism;
    e.single = rosauer;
    e.cwe = safe_div(endemism, e.richness);
    return e;
}

double stats_or_undef(const std::vector<double>& v, int what) {
    if (v.empty()) return NaN;
    Stats s;
    s.add(v);
    return what == 0 ? s.mean() : (what == 1 ? s.sd() : s.sum);
}

// chao confidence intervals (_calc_chao_confidence_intervals)
constexpr double kZ = 1.959964;

std::pair<double, double> chao_ci(double f1, double f2, double chao, double richness, double variance,
                                  const std::vector<double>& values) {
    double lower = NaN, upper = NaN;
    if ((truthy(f1) && truthy(f2)) || f1 > 1) {
        const double T = chao - richness;
        // eval: log of a non-positive number or a division by zero leaves them undef
        const double denom = perl_pow(T, 2);
        if (denom != 0) {
            const double arg = 1 + num(variance) / denom;
            if (arg > 0) {
                const double s = perl_log(arg);
                if (s >= 0) {
                    const double Kf = perl_exp(kZ * perl_sqrt(s));
                    if (Kf != 0) {
                        lower = richness + T / Kf;
                        upper = richness + T * Kf;
                    }
                }
            }
        }
    } else {
        double P = 0;
        const auto sums = numeric_key_counts(values);
        const bool only_ones = sums.size() == 1 && sums[0].first == 1;
        if (truthy(richness) && !only_ones) {
            for (const auto& fs : sums) P += fs.second * perl_exp(-fs.first);
            P /= richness;
            const double part1 = richness / (1 - P);
            const double part2 = kZ * perl_sqrt(num(variance)) / (1 - P);
            lower = std::max(richness, part1 - part2);
            upper = part1 + part2;
        }
    }
    return {lower, upper};
}

struct ChaoResult {
    double estimate, f1, f2, se, variance, undetected, lo, hi;
    double variance_formula, chao_formula, ci_formula;
};

// calc_chao1 (values = sample counts per label) and calc_chao2 (group counts, R groups)
ChaoResult chao(const std::vector<double>& values, bool chao2, double R) {
    double f1 = 0, f2 = 0, n = 0;
    for (double a : values) {
        if (a == 1)
            f1 += 1;
        else if (a == 2)
            f2 += 1;
        n += a;
    }
    const double richness = static_cast<double>(values.size());
    double cn1;
    if (chao2)
        cn1 = truthy(R) ? (R - 1) / R : 1;
    else
        cn1 = truthy(n) ? (n - 1) / n : 1;
    const double cn2 = perl_pow(cn1, 2);
    double chao_formula = chao2 ? 4 : 2;
    double partial = 0;
    double variance = NaN;
    bool eq8 = !truthy(f1);
    bool eq7 = false;
    if (truthy(f1)) {
        if (truthy(f2)) {
            partial = perl_pow(f1, 2) / (2 * f2);
            const double r = f1 / f2;
            if (chao2)
                variance = f2 * (perl_pow(r, 2) * cn1 / 2 + perl_pow(r, 3) * cn2 + perl_pow(r, 4) * cn2 / 4);
            else
                variance = f2 * ((cn1 * perl_pow(r, 2)) / 2 + cn2 * perl_pow(r, 3) + (cn2 * perl_pow(r, 4)) / 4);
        } else if (f1 > 1) {
            partial = f1 * (f1 - 1) / 2;
            eq7 = true;
        } else {
            eq8 = true;
            chao_formula = 0;
        }
    }
    partial *= cn1;
    const double chao_score = richness + partial;
    if (eq7) {
        if (chao2)
            variance = cn1 * (f1 * (f1 - 1)) / 2 + cn2 * (f1 * perl_pow(2 * f1 - 1, 2)) / 4 -
                       cn2 * perl_pow(f1, 4) / 4 / chao_score;
        else
            variance = cn1 * (f1 * (f1 - 1)) / 2 + cn2 * f1 * perl_pow(2 * f1 - 1, 2) / 4 -
                       cn2 * perl_pow(f1, 4) / 4 / chao_score;
    } else if (eq8) {
        double part1 = NaN, part2 = NaN;
        for (const auto& is : numeric_key_counts(values)) {
            const double i = is.first, f = is.second;
            part1 = num(part1) + f * (perl_exp(-i) - perl_exp(-2 * i));
            part2 = num(part2) + i * perl_exp(-i) * f;
        }
        const double denom = chao2 ? R : n;
        variance = truthy(denom) ? num(part1) - perl_pow(num(part2), 2) / denom : 0;
        chao_formula = 0;
    }
    if (chao2) {
        if (defined(variance)) variance = std::max(0.0, variance);
    } else {
        variance = std::max(0.0, num(variance));
    }
    const auto ci = chao_ci(f1, f2, chao_score, richness, variance, values);
    ChaoResult r;
    r.estimate = chao_score;
    r.f1 = f1;
    r.f2 = f2;
    r.variance = variance;
    r.se = perl_sqrt(num(variance));
    r.undetected = partial;
    r.lo = ci.first;
    r.hi = ci.second;
    r.variance_formula = eq7 ? (chao2 ? 11 : 7) : (eq8 ? (chao2 ? 12 : 8) : (chao2 ? 10 : 6));
    r.chao_formula = chao_formula;
    r.ci_formula = eq8 ? 14 : 13;
    return r;
}

// ---- ACE / ICE ------------------------------------------------------------------------------

struct AceArgs {
    std::map<double, double> F;  // frequency -> count (all taxa)
    std::map<double, double> f_rare;
    double cv, n_rare, C_rare, S_rare, s_estimate, t;
};

// sum over keys (string order) of f(key)
double sum_over_keys(const std::vector<std::pair<double, double>>& keys, const std::function<bool(double)>& keep,
                     const std::function<double(double, double)>& f, bool& any) {
    double s = 0;
    any = false;
    for (const auto& kv : keys) {
        if (!keep(kv.first)) continue;
        s += f(kv.first, kv.second);
        any = true;
    }
    return s;
}

double get_or0(const std::map<double, double>& m, double k) {
    auto it = m.find(k);
    return it == m.end() ? 0.0 : it->second;
}

double ace_differential(const AceArgs& a, double f, const std::vector<std::pair<double, double>>& Fkeys) {
    const double k = 10;
    if (f > k) return 1;
    const double cv = a.cv;
    double n_rare = a.n_rare;
    const double D_rare = a.S_rare;
    double si;
    {
        double s = NaN;
        for (int i = 2; i <= 10; ++i) {
            const double Fi = get_or0(a.F, i);
            if (!truthy(Fi)) continue;
            s = num(s) + i * (i - 1) * Fi;
        }
        si = s;
    }
    (void)Fkeys;
    const double f1 = get_or0(a.F, 1);
    double d;
    const double cov_est = 1 - f1 / n_rare;
    if (cv != 0) {
        const double n_rare_1 = n_rare - 1;
        const double si_v = num(si);
        if (f == 1) {
            d = (cov_est + D_rare * (n_rare - f1) / perl_pow(n_rare, 2)) / perl_pow(cov_est, 2) +
                (perl_pow(cov_est, 2) * n_rare * n_rare_1 * (D_rare * si_v + f1 * si_v) -
                 f1 * D_rare * si_v *
                     (-2 * cov_est * (n_rare - f1) / perl_pow(n_rare, 2) * n_rare * n_rare_1 +
                      perl_pow(cov_est, 2) * (2 * n_rare - 1))) /
                    perl_pow(cov_est, 4) / perl_pow(n_rare, 2) / perl_pow(n_rare_1, 2) -
                (cov_est + f1 * (n_rare - f1) / perl_pow(n_rare, 2)) / perl_pow(cov_est, 2);
        } else {
            d = (cov_est - D_rare * f * f1 / perl_pow(n_rare, 2)) / perl_pow(cov_est, 2) +
                (perl_pow(cov_est, 2) * n_rare * n_rare_1 * f1 * (si_v + D_rare * f * (f - 1)) -
                 f1 * D_rare * si_v *
                     (2 * cov_est * f1 * f / perl_pow(n_rare, 2) * n_rare * n_rare_1 +
                      perl_pow(cov_est, 2) * f * n_rare_1 + perl_pow(cov_est, 2) * n_rare * f)) /
                    perl_pow(cov_est, 4) / perl_pow(n_rare, 2) / perl_pow(n_rare_1, 2) +
                (f * perl_pow(f1, 2) / perl_pow(n_rare, 2)) / perl_pow(cov_est, 2);
        }
    } else {
        if (f == 1)
            d = (cov_est + D_rare * (n_rare - f1) / perl_pow(n_rare, 2)) / perl_pow(cov_est, 2);
        else
            d = (cov_est - D_rare * f * f1 / perl_pow(n_rare, 2)) / perl_pow(cov_est, 2);
    }
    return d;
}

double ice_differential(const AceArgs& a, double q) {
    const double k = 10;
    if (q > k) return 1;
    const double CV = a.cv;
    const double n_infreq = a.n_rare;
    const double C = a.C_rare;
    const double D = a.S_rare;
    const double t = a.t;
    double si_s = NaN;
    for (int i = 2; i <= 10; ++i) {
        const double Qi = get_or0(a.f_rare, i);
        if (!truthy(Qi)) continue;
        si_s = num(si_s) + i * (i - 1) * Qi;
    }
    const double si = num(si_s);
    const double Q1 = get_or0(a.f_rare, 1), Q2 = get_or0(a.f_rare, 2);
    double d, dc;
    const double t_1 = t - 1;
    if (CV != 0) {
        const double n1 = n_infreq - 1;
        if (q == 1) {
            dc = -1 *
                 (n_infreq * (t_1 * Q1 + 2 * Q2) * 2 * Q1 * t_1 -
                  t_1 * perl_pow(Q1, 2) * (t_1 * (Q1 + n_infreq) + 2 * Q2)) /
                 perl_pow(n_infreq * (t_1 * Q1 + 2 * Q2), 2);
            d = (C - D * dc) / perl_pow(C, 2) +
                t / t_1 *
                    (perl_pow(C, 2) * n_infreq * n1 * (D * si + Q1 * si) -
                     Q1 * D * si * (2 * C * dc * n_infreq * n1 + perl_pow(C, 2) * n1 + perl_pow(C, 2) * n_infreq)) /
                    perl_pow(C, 4) / perl_pow(n_infreq, 2) / perl_pow(n1, 2) -
                (C - Q1 * dc) / perl_pow(C, 2);
        } else if (q == 2) {
            dc = -(-t_1 * perl_pow(Q1, 2) * (2 * t_1 * Q1 + 2 * (n_infreq + 2 * Q2))) /
                 perl_pow(n_infreq * (t_1 * Q1 + 2 * Q2), 2);
            d = (C - D * dc) / perl_pow(C, 2) +
                t / t_1 *
                    (perl_pow(C, 2) * n_infreq * n1 * Q1 * (si + 2 * D) -
                     Q1 * D * si * (2 * C * dc * n_infreq * n1 + perl_pow(C, 2) * 2 * n1 + perl_pow(C, 2) * n_infreq * 2)) /
                    perl_pow(C, 4) / perl_pow(n_infreq, 2) / perl_pow(n1, 2) -
                (-Q1 * dc) / perl_pow(C, 2);
        } else {
            dc = -(-t_1 * perl_pow(Q1, 2) * (t_1 * Q1 * q + 2 * Q2 * q)) / perl_pow(n_infreq * (t_1 * Q1 + 2 * Q2), 2);
            d = (C - D * dc) / perl_pow(C, 2) +
                t / t_1 *
                    (perl_pow(C, 2) * n_infreq * n1 * Q1 * (si + q * (q - 1) * D) -
                     Q1 * D * si * (2 * C * dc * n_infreq * n1 + perl_pow(C, 2) * q * n1 + perl_pow(C, 2) * n_infreq * q)) /
                    perl_pow(C, 4) / perl_pow(n_infreq, 2) / perl_pow(n1, 2) -
                (-Q1 * dc) / perl_pow(C, 2);
        }
    } else {
        if (q == 1)
            dc = -1 *
                 (n_infreq * (t_1 * Q1 + 2 * Q2) * 2 * Q1 * t_1 -
                  t_1 * perl_pow(Q1, 2) * (t_1 * (Q1 + n_infreq) + 2 * Q2)) /
                 perl_pow(n_infreq * (t_1 * Q1 + 2 * Q2), 2);
        else if (q == 2)
            dc = -1 * (-1 * t_1 * perl_pow(Q1, 2) * (2 * t_1 * Q1 + 2 * (n_infreq + 2 * Q2))) /
                 perl_pow(n_infreq * (t_1 * Q1 + 2 * Q2), 2);
        else
            dc = -1 * (-1 * t_1 * perl_pow(Q1, 2) * (t_1 * Q1 * q + 2 * Q2 * q)) /
                 perl_pow(n_infreq * (t_1 * Q1 + 2 * Q2), 2);
        d = (C - D * dc) / perl_pow(C, 2);
    }
    return d;
}

// calc_ace / calc_ice. `values` are sample counts (ACE) or group counts (ICE).
std::vector<double> ace(const std::vector<double>& values, bool ice, double t,
                        const std::function<ChaoResult()>& chao_fallback) {
    // order of the output: CI_LOWER CI_UPPER ESTIMATE ESTIMATE_USED_CHAO INFREQUENT_COUNT SE UNDETECTED VARIANCE
    std::vector<double> r(8, NaN);
    r[3] = 0;
    double S_abund = 0, S_rare = 0, f1 = 0, n_rare = 0;
    const double richness = static_cast<double>(values.size());
    std::map<double, double> all_freqs, f_rare;
    for (double f : values) {
        all_freqs[f] += 1;
        if (f <= 10) {
            f_rare[f] += 1;
            n_rare += f;
            S_rare += 1;
            if (f == 1) f1 += 1;
        } else {
            S_abund += 1;
        }
    }
    if (!truthy(richness) || (ice && t <= 1) || f1 == richness) {
        r[4] = S_rare;
        r[2] = richness;
        r[3] = 0;
        return r;
    }
    if (!truthy(n_rare) || !truthy(f1) || f1 == n_rare) {
        r[4] = S_rare;
        r[3] = 1;
        const ChaoResult c = chao_fallback();
        r[0] = c.lo;
        r[1] = c.hi;
        r[2] = c.estimate;
        r[5] = c.se;
        r[6] = c.undetected;
        r[7] = c.variance;
        return r;
    }
    double C_ace = 1 - f1 / n_rare;
    double a1 = NaN, a2 = NaN;
    for (int i = 1; i <= 10; ++i) {
        const double fi = get_or0(f_rare, i);
        if (!truthy(fi)) continue;
        a1 = num(a1) + i * (i - 1) * fi;
        a2 = num(a2) + i * fi;
    }
    a1 = num(a1);
    a2 = num(a2);
    double gamma;
    if (ice) {
        double A = 1;
        const double fr1 = get_or0(f_rare, 1), fr2 = get_or0(f_rare, 2);
        if (truthy(fr1)) {
            if (truthy(fr2))
                A = 2 * fr2 / ((t - 1) * fr1 + 2 * fr2);
            else
                A = 2 / ((t - 1) * (fr1 - 1) + 2);
        }
        C_ace = 1 - (f1 / n_rare) * (1 - A);
        gamma = C_ace == 0 ? 0 : (S_rare / C_ace) * t / (t - 1) * a1 / a2 / (a2 - 1) - 1;
    } else {
        gamma = C_ace == 0 ? 0 : (S_rare / C_ace) * a1 / a2 / (a2 - 1) - 1;
    }
    gamma = std::max(gamma, 0.0);
    const double S_ace = S_abund + S_rare / C_ace + f1 / C_ace * gamma;
    const double cv = perl_sqrt(gamma);
    AceArgs aa;
    aa.F = all_freqs;
    aa.f_rare = f_rare;
    aa.cv = cv;
    aa.n_rare = n_rare;
    aa.C_rare = C_ace;
    aa.S_rare = S_rare;
    aa.s_estimate = S_ace;
    aa.t = t;
    // _get_ice_variance: keys sorted numerically
    std::vector<double> sorted;
    for (const auto& kv : all_freqs) sorted.push_back(kv.first);
    std::vector<double> diff(sorted.size());
    double var = 0;
    auto cov = [&](double i, double j) {
        const double Qi = all_freqs[i], Qj = all_freqs[j];
        return i == j ? Qi * (1 - Qi / S_ace) : -1 * Qi * Qj / S_ace;
    };
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        const double v1 = sorted[i];
        diff[i] = ice ? ice_differential(aa, v1) : ace_differential(aa, v1, {});
        for (std::size_t j = 0; j < i; ++j) var += 2 * diff[i] * diff[j] * cov(v1, sorted[j]);
        var += perl_pow(diff[i], 2) * cov(v1, v1);
    }
    const double variance = var == 0 ? NaN : var;  // $var ||= undef
    const double se = perl_sqrt(num(variance));
    // _calc_ace_confidence_intervals
    double lower = NaN, upper = NaN;
    if ((S_ace - richness) >= 0.00001) {
        const double T = S_ace - richness;
        const double Kf = perl_exp(kZ * perl_sqrt(perl_log(1 + num(variance) / perl_pow(T, 2))));
        lower = richness + T / Kf;
        upper = richness + T * Kf;
    } else {
        double part1 = 0, part2 = 0, P = 0;
        std::vector<double> keys_vals;
        for (const auto& kv : all_freqs) keys_vals.push_back(kv.first);
        // foreach my ($f, $val) (%$freq_counts): keys in string order
        std::map<std::string, std::pair<double, double>> ordered;
        for (const auto& kv : all_freqs) ordered[perl_str(kv.first)] = kv;
        for (const auto& e : ordered) {
            const double f = e.second.first, val = e.second.second;
            part1 += val * (perl_exp(-f) - perl_exp(-2 * f));
            part2 += f * perl_exp(-f) * val;
            P += val * perl_exp(-f) / richness;
        }
        double n = NaN;
        for (double v : values) n = num(n) + v;
        const double var_obs = part1 - perl_pow(part2, 2) / n;
        lower = std::max(richness, richness / (1 - P) - kZ * perl_sqrt(var_obs) / (1 - P));
        upper = richness / (1 - P) + kZ * perl_sqrt(var_obs) / (1 - P);
    }
    r[0] = lower;
    r[1] = upper;
    r[2] = S_ace;
    r[4] = S_rare;
    r[5] = se;
    r[6] = S_ace - richness;
    r[7] = variance;
    return r;
}

std::vector<double> lgamma_table(i64 n) {
    std::vector<double> t(static_cast<std::size_t>(std::max<i64>(n, 1) + 1), 0.0);
    for (i64 i = 2; i <= n; ++i) t[static_cast<std::size_t>(i)] = t[static_cast<std::size_t>(i - 1)] + perl_log(static_cast<double>(i));
    return t;
}

// ---------------------------------------------------------------------------------------------

struct Engine {
    const BaseDataView& bd;
    const std::vector<CalcInfo>& reg;
    std::vector<i32> calc_ids;
    std::vector<std::vector<std::size_t>> scalar_slot;  // per calc, per scalar -> output column
    std::vector<std::vector<std::size_t>> list_slot;
    std::vector<std::string> texts;
    std::unordered_map<std::string, i32> text_ids;

    i32 text_id(const std::string& s) {
        auto it = text_ids.find(s);
        if (it != text_ids.end()) return it->second;
        const i32 id = static_cast<i32>(texts.size());
        texts.push_back(s);
        text_ids.emplace(s, id);
        return id;
    }
};

}  // namespace

const std::vector<CalcInfo>& calculation_registry() {
    static const std::vector<CalcInfo> reg = make_registry();
    return reg;
}

IndexResult run_indices(const BaseDataView& bd, const std::vector<i32>& procs, const Csr& set1, const Csr* set2,
                        const std::vector<std::string>& calcs, bool parallel) {
    const auto& reg = calculation_registry();
    IndexResult res;
    std::vector<i32> ids;
    for (const auto& name : calcs) {
        auto it = std::find_if(reg.begin(), reg.end(), [&](const CalcInfo& c) { return c.name == name; });
        if (it == reg.end()) throw std::invalid_argument("unknown calculation " + name);
        ids.push_back(static_cast<i32>(it - reg.begin()));
    }
    std::vector<std::vector<std::size_t>> sslot(ids.size()), lslot(ids.size());
    for (std::size_t c = 0; c < ids.size(); ++c) {
        const CalcInfo& ci = reg[static_cast<std::size_t>(ids[c])];
        for (const auto& s : ci.scalars) {
            sslot[c].push_back(res.scalar_names.size());
            res.scalar_names.push_back(s);
        }
        for (const auto& l : ci.lists) {
            lslot[c].push_back(res.list_names.size());
            res.list_names.push_back(l.first);
            res.list_kinds.push_back(l.second);
        }
        if (ci.name == "calc_endemism_central_hier_part" || ci.name == "calc_endemism_whole_hier_part") {
            // PREFIX_i, PREFIX_C_i, PREFIX_E_i, PREFIX_OME_i for every label level
            const std::string pfx = ci.name == "calc_endemism_central_hier_part" ? "ENDC_HPART_" : "ENDW_HPART_";
            for (const char* part : {"", "C_", "E_", "OME_"})
                for (i32 i = 0; i < bd.n_label_levels; ++i) {
                    lslot[c].push_back(res.list_names.size());
                    res.list_names.push_back(pfx + part + std::to_string(i));
                    res.list_kinds.push_back(K::Text);
                }
        }
    }
    // fixed text keys (so that threads never add any)
    std::unordered_map<std::string, i32> text_ids;
    auto text_id = [&](const std::string& s) {
        auto it = text_ids.find(s);
        if (it != text_ids.end()) return it->second;
        const i32 id = static_cast<i32>(res.texts.size());
        res.texts.push_back(s);
        text_ids.emplace(s, id);
        return id;
    };
    std::vector<i32> quantile_keys;
    for (int q = 0; q <= 100; q += 5) {
        char buf[8];
        std::snprintf(buf, sizeof buf, "Q%03d", q);
        quantile_keys.push_back(text_id(buf));
    }
    const i32 key_vf = text_id("VARIANCE_FORMULA"), key_cf = text_id("CHAO_FORMULA"), key_ci = text_id("CI_FORMULA");
    std::vector<i32> prefix_text(bd.prefix_names.size());
    for (std::size_t i = 0; i < bd.prefix_names.size(); ++i) prefix_text[i] = text_id(bd.prefix_names[i]);
    const i32 levels = bd.n_label_levels;

    const i64 np = static_cast<i64>(procs.size());
    const std::size_t ns = res.scalar_names.size(), nl = res.list_names.size();
    res.scalars.assign(static_cast<std::size_t>(np) * ns, NaN);
    // per-row list entries, merged after the loop
    // per-row list entries (list, key, value), merged after the loop
    struct Entry {
        i32 list;
        double key, value;
    };
    std::vector<std::vector<Entry>> rows(static_cast<std::size_t>(np));

    // global label count (calc_d) and maximum sample sum (hurlbert)
    const double n_labels = static_cast<double>(bd.n_labels);
    const bool have_set2 = set2 != nullptr;

    std::string error;
    bool failed = false;
    lsm::parallel_for_dynamic(parallel && np > 1, np, [&](i64 r) {
        if (failed) return;
        try {
            thread_local Abc abc;
            if (abc.in1.size() != static_cast<std::size_t>(bd.n_labels)) abc.init(bd.n_labels);
            abc.clear();
            abc.g1.assign(set1.indices.begin() + set1.indptr[r], set1.indices.begin() + set1.indptr[r + 1]);
            abc.g2.clear();
            abc.has_set2 = have_set2;
            if (have_set2) abc.g2.assign(set2->indices.begin() + set2->indptr[r], set2->indices.begin() + set2->indptr[r + 1]);
            aggregate(bd, abc.g1, abc.l1, abc.in1, abc.c1, abc.s1);
            aggregate(bd, abc.g2, abc.l2, abc.in2, abc.c2, abc.s2);
            std::set_union(abc.l1.begin(), abc.l1.end(), abc.l2.begin(), abc.l2.end(), std::back_inserter(abc.lall));
            abc.ABC = static_cast<i64>(abc.lall.size());
            abc.A = static_cast<i64>(abc.l1.size() + abc.l2.size()) - abc.ABC;
            abc.B = abc.ABC - static_cast<i64>(abc.l2.size());
            abc.C = abc.ABC - static_cast<i64>(abc.l1.size());
            const double A = static_cast<double>(abc.A), B = static_cast<double>(abc.B), C = static_cast<double>(abc.C),
                         ABC = static_cast<double>(abc.ABC);
            const double e1 = static_cast<double>(abc.g1.size()), e2 = static_cast<double>(abc.g2.size());
            const double e_all = e1 + e2;
            const i32 proc = procs[static_cast<std::size_t>(r)];

            std::vector<Entry> entries;
            double* out = res.scalars.data() + static_cast<std::size_t>(r) * ns;

            // lazily computed shared pieces
            bool have_endw = false, have_endc = false, have_rarew = false, have_rarec = false;
            Endemism endw, endc, rarew, rarec;
            auto get_endw = [&]() -> const Endemism& {
                if (!have_endw) {
                    endw = calc_endemism(Ctx{bd, abc, proc, res.texts, nullptr}, false, false);
                    have_endw = true;
                }
                return endw;
            };
            auto get_endc = [&]() -> const Endemism& {
                if (!have_endc) {
                    endc = calc_endemism(Ctx{bd, abc, proc, res.texts, nullptr}, true, false);
                    have_endc = true;
                }
                return endc;
            };
            auto get_rarew = [&]() -> const Endemism& {
                if (!have_rarew) {
                    rarew = calc_endemism(Ctx{bd, abc, proc, res.texts, nullptr}, false, true);
                    have_rarew = true;
                }
                return rarew;
            };
            auto get_rarec = [&]() -> const Endemism& {
                if (!have_rarec) {
                    rarec = calc_endemism(Ctx{bd, abc, proc, res.texts, nullptr}, true, true);
                    have_rarec = true;
                }
                return rarec;
            };
            auto nonempty = [&](const std::vector<i32>& gs) {
                double n = 0;
                for (i32 g : gs)
                    if (bd.indptr[g + 1] > bd.indptr[g]) n += 1;
                return n;
            };
            auto values_count = [&](const std::vector<i32>& ls, int which) {
                std::vector<double> v;
                v.reserve(ls.size());
                for (i32 l : ls) v.push_back(which == 1 ? abc.c1[l] : (which == 2 ? abc.c2[l] : abc.all_count(l)));
                return v;
            };
            auto values_samples = [&](const std::vector<i32>& ls, int which) {
                std::vector<double> v;
                v.reserve(ls.size());
                for (i32 l : ls) v.push_back(which == 1 ? abc.s1[l] : (which == 2 ? abc.s2[l] : abc.all_samples(l)));
                return v;
            };

            for (std::size_t c = 0; c < ids.size(); ++c) {
                const std::string& name = reg[static_cast<std::size_t>(ids[c])].name;
                const auto& S = sslot[c];
                const auto& L = lslot[c];
                auto set = [&](std::size_t k, double v) { out[S[k]] = v; };
                auto put = [&](std::size_t k, double key, double v) {
                    entries.push_back(Entry{static_cast<i32>(L[k]), key, v});
                };

                if (name == "calc_richness") {
                    set(0, ABC);
                    set(1, A + B);
                    set(2, A + C);
                } else if (name == "calc_redundancy") {
                    double sa = NaN, s1v = NaN, s2v = NaN;
                    for (i32 l : abc.lall) {
                        sa = num(sa) + abc.all_samples(l);
                        if (abc.in1[l]) s1v = num(s1v) + abc.s1[l];
                        if (abc.in2[l]) s2v = num(s2v) + abc.s2[l];
                    }
                    auto red = [&](double n, double s) { return num(s) == 0 ? NaN : 1 - (n / num(s)); };
                    set(0, red(ABC, sa));
                    set(1, red(static_cast<double>(abc.l1.size()), s1v));
                    set(2, red(static_cast<double>(abc.l2.size()), s2v));
                } else if (name == "calc_abc_counts") {
                    set(0, A);
                    set(1, ABC);
                    set(2, B);
                    set(3, C);
                } else if (name == "calc_d") {
                    set(0, n_labels - ABC);
                } else if (name == "calc_elements_used") {
                    set(0, e_all);
                    set(1, e1);
                    set(2, e2);
                } else if (name == "calc_nonempty_elements_used") {
                    const double n1 = nonempty(abc.g1);
                    const double n2 = have_set2 ? nonempty(abc.g2) : NaN;
                    set(0, n1 + num(n2));
                    set(1, n1);
                    set(2, n2);
                } else if (name == "calc_element_lists_used" || name == "calc_element_lists_used_as_arrays") {
                    std::vector<i32> all;
                    std::set_union(abc.g1.begin(), abc.g1.end(), abc.g2.begin(), abc.g2.end(), std::back_inserter(all));
                    for (i32 g : all) put(0, g, 1);
                    for (i32 g : abc.g1) put(1, g, 1);
                    for (i32 g : abc.g2) put(2, g, 1);
                } else if (name == "calc_label_count_quantile_position") {
                    // For each label of the neighbourhood or the processing group: the share of
                    // the other neighbours whose count of it is below the processing group's.
                    // One pass over the neighbours' rows: absent labels count 0.
                    std::vector<i32> proc_labels(bd.indices.begin() + bd.indptr[proc], bd.indices.begin() + bd.indptr[proc + 1]);
                    std::vector<i32> merged;
                    std::set_union(abc.lall.begin(), abc.lall.end(), proc_labels.begin(), proc_labels.end(),
                                   std::back_inserter(merged));
                    thread_local std::vector<double> target, below, present;
                    if (target.size() != static_cast<std::size_t>(bd.n_labels)) {
                        target.assign(static_cast<std::size_t>(bd.n_labels), 0.0);
                        below.assign(static_cast<std::size_t>(bd.n_labels), 0.0);
                        present.assign(static_cast<std::size_t>(bd.n_labels), 0.0);
                    }
                    for (i64 k = bd.indptr[proc]; k < bd.indptr[proc + 1]; ++k)
                        target[bd.indices[static_cast<std::size_t>(k)]] = bd.counts[static_cast<std::size_t>(k)];
                    double n = 0;
                    auto scan = [&](const std::vector<i32>& gs) {
                        for (i32 g : gs) {
                            if (g == proc) continue;
                            n += 1;
                            for (i64 k = bd.indptr[g]; k < bd.indptr[g + 1]; ++k) {
                                const i32 l = bd.indices[static_cast<std::size_t>(k)];
                                present[l] += 1;
                                if (bd.counts[static_cast<std::size_t>(k)] < target[l]) below[l] += 1;
                            }
                        }
                    };
                    scan(abc.g1);
                    scan(abc.g2);
                    for (i32 l : merged) {
                        // groups without the label hold 0, below the target when it is positive
                        const double pos = below[l] + (0 < target[l] ? n - present[l] : 0.0);
                        put(0, l, n > 0 ? 100 * pos / n : NaN);
                    }
                    for (i32 l : merged) {
                        target[l] = 0;
                        below[l] = 0;
                        present[l] = 0;
                    }
                    // labels of neighbours outside `merged` cannot exist (merged holds every label seen)
                } else if (name == "calc_local_range_lists") {
                    for (i32 l : abc.lall) put(0, l, abc.all_count(l));
                    for (i32 l : abc.l1) put(1, l, abc.c1[l]);
                    for (i32 l : abc.l2) put(2, l, abc.c2[l]);
                } else if (name == "calc_local_range_stats") {
                    set(0, stats_or_undef(values_count(abc.lall, 0), 0));
                    set(1, stats_or_undef(values_count(abc.l1, 1), 0));
                    set(2, stats_or_undef(values_count(abc.l2, 2), 0));
                    set(3, stats_or_undef(values_count(abc.lall, 0), 1));
                    set(4, stats_or_undef(values_count(abc.l1, 1), 1));
                    set(5, stats_or_undef(values_count(abc.l2, 2), 1));
                } else if (name == "calc_local_sample_count_lists") {
                    for (i32 l : abc.lall) put(0, l, abc.all_samples(l));
                    for (i32 l : abc.l1) put(1, l, abc.s1[l]);
                    for (i32 l : abc.l2) put(2, l, abc.s2[l]);
                } else if (name == "calc_local_sample_count_stats") {
                    const auto va = values_samples(abc.lall, 0), v1 = values_samples(abc.l1, 1),
                               v2 = values_samples(abc.l2, 2);
                    set(0, stats_or_undef(va, 0));
                    set(1, stats_or_undef(v1, 0));
                    set(2, stats_or_undef(v2, 0));
                    set(3, stats_or_undef(va, 1));
                    set(4, stats_or_undef(v1, 1));
                    set(5, stats_or_undef(v2, 1));
                    set(6, stats_or_undef(va, 2));
                    set(7, stats_or_undef(v1, 2));
                    set(8, stats_or_undef(v2, 2));
                } else if (name == "calc_local_sample_count_quantiles") {
                    std::vector<double> ps;
                    for (int q = 0; q <= 100; q += 5) ps.push_back(q);
                    auto emit = [&](std::size_t k, const std::vector<double>& v) {
                        if (v.empty()) return;
                        Stats s;
                        s.add(v);
                        const auto pv = s.percentiles(ps);
                        for (std::size_t i = 0; i < pv.size(); ++i) put(k, quantile_keys[i], pv[i]);
                    };
                    const auto va = values_samples(abc.lall, 0);
                    emit(0, va);
                    if (!abc.l2.empty()) {
                        emit(1, values_samples(abc.l1, 1));
                        emit(2, values_samples(abc.l2, 2));
                    } else {
                        emit(1, va);
                    }
                } else if (name == "calc_kulczynski2") {
                    double v = NaN;
                    if (truthy(A) || (truthy(B) && truthy(C))) {
                        if (A + B != 0 && A + C != 0) v = 1 - 0.5 * (A / (A + B) + A / (A + C));
                    }
                    set(0, v);
                } else if (name == "calc_sorenson") {
                    double v = NaN;
                    if (truthy(A) || (truthy(B) && truthy(C))) v = safe_div(2 * A, A + ABC);
                    set(0, defined(v) ? 1 - v : NaN);
                } else if (name == "calc_jaccard") {
                    double v = NaN;
                    if (truthy(A) || (truthy(B) && truthy(C))) {
                        const double q = safe_div(A, ABC);
                        v = defined(q) ? 1 - q : NaN;
                    }
                    set(0, v);
                } else if (name == "calc_nestedness_resultant") {
                    double score;
                    if (!truthy(A) && truthy(B) && truthy(C)) {
                        score = 0;
                    } else if (!truthy(A) && !(truthy(B) && truthy(C))) {
                        score = NaN;
                    } else {
                        // eval'd divisions: undef parts multiply as 0
                        const double part1 = safe_div(std::fabs(B - C), A + ABC);
                        const double part2 = safe_div(A, A + std::min(B, C));
                        score = num(part1) * num(part2);
                    }
                    set(0, score);
                } else if (name == "calc_bray_curtis" || name == "calc_bray_curtis_norm_by_gp_counts") {
                    const bool norm = name == "calc_bray_curtis_norm_by_gp_counts";
                    double Asum = 0, Bsum = 0, W = 0;
                    for (i32 l : abc.lall) {
                        double v1 = abc.in1[l] ? abc.s1[l] : 0, v2 = abc.in2[l] ? abc.s2[l] : 0;
                        if (norm) {
                            v1 = num(safe_div(v1, e1));
                            v2 = num(safe_div(v2, e2));
                        }
                        W += v1 < v2 ? v1 : v2;
                        Asum += v1;
                        Bsum += v2;
                    }
                    const double q = safe_div(2 * W, Asum + Bsum);
                    set(0, Asum);
                    set(1, Bsum);
                    set(2, W);
                    set(3, defined(q) ? 1 - q : NaN);
                } else if (name == "calc_beta_diversity") {
                    const double q = safe_div(ABC, std::max(A + B, A + C));
                    set(0, defined(q) ? q - 1 : NaN);
                } else if (name == "calc_s2") {
                    double v = NaN;
                    if (truthy(ABC)) {
                        const double q = safe_div(A, A + std::min(B, C));
                        v = defined(q) ? 1 - q : NaN;
                    }
                    set(0, v);
                } else if (name == "calc_simpson_shannon") {
                    if (truthy(ABC)) {
                        double n = 0;
                        for (i32 l : abc.lall) n += abc.all_samples(l);
                        double simpson = NaN, shannon = NaN;
                        for (i32 l : abc.lall) {
                            const double p = abc.all_samples(l) / n;
                            simpson = num(simpson) + perl_pow(p, 2);
                            shannon = num(shannon) + p * perl_log(p);
                        }
                        shannon = num(shannon) * -1;
                        const double hmax = perl_log(ABC);
                        set(0, hmax == 0 ? NaN : shannon / hmax);
                        set(1, shannon);
                        set(2, hmax);
                        set(3, 1 - num(simpson));
                    }
                } else if (name == "calc_tx_rao_qe") {
                    if (!abc.lall.empty()) {
                        double n = 0;
                        bool first = true;
                        for (i32 l : abc.lall) {
                            if (first) {
                                n = abc.all_samples(l);
                                first = false;
                            } else {
                                n += abc.all_samples(l);
                            }
                        }
                        std::vector<double> p(abc.lall.size());
                        for (std::size_t i = 0; i < abc.lall.size(); ++i) p[i] = abc.all_samples(abc.lall[i]) / n;
                        double qe = 0;
                        for (std::size_t i = 0; i < abc.lall.size(); ++i)
                            for (std::size_t j = i; j < abc.lall.size(); ++j) {
                                const double value = i == j ? 0.0 : 1.0;
                                qe += 2 * value * p[i] * p[j];
                            }
                        set(0, qe);
                        set(1, perl_pow(static_cast<double>(abc.lall.size()), 2));
                        for (std::size_t i = 0; i < abc.lall.size(); ++i) put(0, abc.lall[i], p[i]);
                    } else {
                        set(1, 0);
                    }
                } else if (name == "calc_rw_turnover") {
                    const Endemism& e = get_endw();
                    // weights by label (ENDW_WTLIST)
                    double aa = 0, bb = 0, cc = 0;
                    std::size_t wi = 0;
                    auto weight = [&](i32 l) {
                        while (wi < e.wts.size() && e.wts[wi].first < l) ++wi;
                        return (wi < e.wts.size() && e.wts[wi].first == l) ? e.wts[wi].second : 0.0;
                    };
                    for (i32 l : abc.l1) {
                        if (abc.in2[l])
                            aa += weight(l);
                        else
                            bb += weight(l);
                    }
                    wi = 0;
                    for (i32 l : abc.l2)
                        if (!abc.in1[l]) cc += weight(l);
                    const bool valid = (truthy(aa) || truthy(bb)) && (truthy(aa) || truthy(cc));
                    set(0, valid ? 1 - (aa / (aa + bb + cc)) : NaN);
                    set(1, aa);
                    set(2, bb);
                    set(3, cc);
                } else if (name == "calc_endemism_central" || name == "calc_endemism_whole" ||
                           name == "calc_rarity_central" || name == "calc_rarity_whole") {
                    const bool rare = name.find("rarity") != std::string::npos;
                    const bool central = name.find("central") != std::string::npos;
                    const Endemism& e = rare ? (central ? get_rarec() : get_rarew()) : (central ? get_endc() : get_endw());
                    if (rare) {
                        set(0, e.cwe);
                        set(1, e.richness);
                        set(2, e.we);
                    } else {
                        set(0, e.cwe);
                        set(1, e.richness);
                        set(2, e.single);
                        set(3, e.we);
                    }
                } else if (name == "calc_endemism_central_lists" || name == "calc_endemism_whole_lists" ||
                           name == "calc_rarity_central_lists" || name == "calc_rarity_whole_lists") {
                    const bool rare = name.find("rarity") != std::string::npos;
                    const bool central = name.find("central") != std::string::npos;
                    const Endemism& e = rare ? (central ? get_rarec() : get_rarew()) : (central ? get_endc() : get_endw());
                    for (const auto& kv : e.ranges) put(0, kv.first, kv.second);
                    for (const auto& kv : e.wts) put(1, kv.first, kv.second);
                } else if (name == "calc_endemism_central_normalised" || name == "calc_endemism_whole_normalised") {
                    const Endemism& e = name.find("central") != std::string::npos ? get_endc() : get_endw();
                    if (truthy(e_all) && defined(e.we)) {
                        set(0, e.cwe / e_all);
                        set(1, e.we / e_all);
                    }
                } else if (name == "calc_endemism_absolute" || name == "calc_endemism_absolute_lists") {
                    double end1 = 0, end2 = 0, end_all = 0;
                    const bool lists_wanted = name == "calc_endemism_absolute_lists";
                    for (i32 l : abc.lall) {
                        const double range = bd.label_range[l];
                        const double local = abc.all_count(l);
                        if (range > local) continue;
                        end_all += 1;
                        if (lists_wanted) put(2, l, local);
                        if (abc.in1[l] && range <= abc.c1[l]) {
                            end1 += 1;
                            if (lists_wanted) put(0, l, local);
                        }
                        if (abc.in2[l] && range <= abc.c2[l]) {
                            end2 += 1;
                            if (lists_wanted) put(1, l, local);
                        }
                    }
                    if (!lists_wanted) {
                        set(0, end1);
                        set(1, safe_div(end1, static_cast<double>(abc.l1.size())));
                        set(2, end2);
                        set(3, safe_div(end2, static_cast<double>(abc.l2.size())));
                        set(4, end_all);
                        set(5, safe_div(end_all, static_cast<double>(abc.lall.size())));
                    }
                } else if (name == "calc_chao1" || name == "calc_chao2") {
                    const bool c2 = name == "calc_chao2";
                    const ChaoResult cr = c2 ? chao(values_count(abc.lall, 0), true, e_all)
                                             : chao(values_samples(abc.lall, 0), false, 0);
                    set(0, cr.lo);
                    set(1, cr.hi);
                    set(2, cr.estimate);
                    set(3, cr.f1);
                    set(4, cr.f2);
                    set(5, cr.se);
                    set(6, cr.undetected);
                    set(7, cr.variance);
                    put(0, key_ci, cr.ci_formula);
                    put(0, key_cf, cr.chao_formula);
                    put(0, key_vf, cr.variance_formula);
                } else if (name == "calc_ace" || name == "calc_ice") {
                    const bool ice = name == "calc_ice";
                    const double t = ice ? nonempty(abc.g1) + (have_set2 ? nonempty(abc.g2) : 0.0) : NaN;
                    std::vector<double> values = ice ? values_count(abc.lall, 0) : values_samples(abc.lall, 0);
                    const auto rr = ace(values, ice, t, [&]() {
                        return ice ? chao(values, true, e_all) : chao(values, false, 0);
                    });
                    for (std::size_t k = 0; k < 8; ++k) set(k, rr[k]);
                } else if (name == "calc_hurlbert_es") {
                    double N = 0;
                    for (i32 l : abc.lall) N += abc.all_samples(l);
                    const i64 Ni = static_cast<i64>(N);
                    const std::vector<double> lg = lgamma_table(Ni);
                    auto lgv = [&](double x) { return lg[static_cast<std::size_t>(static_cast<i64>(x))]; };
                    const double targets[] = {5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000};
                    for (double tn : targets) {
                        if (N < tn) break;
                        double score = NaN;
                        const double lg_tn_N = lgv(N - tn) - lgv(N);
                        for (i32 l : abc.lall) {
                            const double ni = abc.all_samples(l);
                            if (N - ni >= tn) {
                                score = num(score) + (1 - perl_exp(lgv(N - ni) + lg_tn_N - lgv(N - ni - tn)));
                            } else if (N >= tn) {
                                score = num(score) + 1;
                            }
                        }
                        put(0, tn, score);
                    }
                } else if (name == "calc_endemism_central_hier_part" || name == "calc_endemism_whole_hier_part") {
                    // _calc_endemism_hier_part: each label's share of WE, summed up its label hierarchy
                    const Endemism& e = name == "calc_endemism_central_hier_part" ? get_endc() : get_endw();
                    const i32 depth = levels - 1;
                    const double total = static_cast<double>(e.wts.size());
                    std::vector<std::map<i32, double>> obs(static_cast<std::size_t>(levels));
                    std::vector<std::map<i32, double>> cnt(static_cast<std::size_t>(levels));
                    for (const auto& lw : e.wts) {
                        if (!(num(e.we) != 0)) throw std::runtime_error("Illegal division by zero");
                        const double contribution = lw.second / e.we;
                        for (i32 i = depth; i >= 0; --i) {
                            const i32 node = bd.label_prefix[static_cast<std::size_t>(lw.first) * levels + i];
                            obs[i][node] += contribution;
                            cnt[i][node] += 1;
                        }
                    }
                    for (i32 i = 0; i < levels; ++i) {
                        if (e.wts.empty()) break;
                        // node names in string order (prefix ids are numbered in that order)
                        for (const auto& kv : cnt[i]) {
                            const double expected = kv.second / total;
                            const i32 key = prefix_text[static_cast<std::size_t>(kv.first)];
                            put(static_cast<std::size_t>(i), key, obs[i][kv.first]);
                            put(static_cast<std::size_t>(levels + i), key, kv.second);
                            put(static_cast<std::size_t>(2 * levels + i), key, expected);
                            put(static_cast<std::size_t>(3 * levels + i), key, expected - obs[i][kv.first]);
                        }
                    }
                } else if (name == "calc_hierarchical_label_ratios") {
                    // Biodiverse computes the same A, B, C for every level (it never uses
                    // the per-level basedatas it builds): reproduced as is.
                    if (levels > 1) {
                        double asum = 0;
                        for (i32 l : abc.lall)
                            if (abc.in1[l] && abc.in2[l]) asum += abc.all_samples(l);
                        for (i32 i = 0; i < levels; ++i) {
                            put(0, i, A);
                            put(4, i, B);
                            put(6, i, C);
                            put(2, i, asum);
                            if (i == 0) continue;
                            put(1, i, safe_div(A, A));
                            put(5, i, safe_div(B, B));
                            put(7, i, safe_div(C, C));
                            put(3, i, 1 - num(safe_div(asum, asum)));
                        }
                    }
                } else {
                    throw std::invalid_argument("calculation not implemented: " + name);
                }
            }
            rows[static_cast<std::size_t>(r)] = std::move(entries);
        } catch (const std::exception& e) {
#pragma omp critical(bio_indices_error)
            {
                failed = true;
                error = e.what();
            }
        }
    });
    if (failed) throw std::runtime_error(error);

    res.lists.resize(nl);
    // counts per list and row, then a single fill pass
    std::vector<i64> count(nl * static_cast<std::size_t>(np), 0);
    for (i64 r = 0; r < np; ++r)
        for (const Entry& e : rows[static_cast<std::size_t>(r)]) ++count[static_cast<std::size_t>(e.list) * np + r];
    std::vector<std::vector<i64>> pos(nl);
    for (std::size_t k = 0; k < nl; ++k) {
        ListOut& lo = res.lists[k];
        lo.indptr.assign(static_cast<std::size_t>(np) + 1, 0);
        for (i64 r = 0; r < np; ++r)
            lo.indptr[static_cast<std::size_t>(r) + 1] = lo.indptr[static_cast<std::size_t>(r)] + count[k * np + r];
        lo.keys.resize(static_cast<std::size_t>(lo.indptr.back()));
        lo.values.resize(static_cast<std::size_t>(lo.indptr.back()));
        pos[k].assign(lo.indptr.begin(), lo.indptr.end() - 1);
    }
    for (i64 r = 0; r < np; ++r) {
        for (const Entry& e : rows[static_cast<std::size_t>(r)]) {
            ListOut& lo = res.lists[static_cast<std::size_t>(e.list)];
            const i64 at = pos[static_cast<std::size_t>(e.list)][static_cast<std::size_t>(r)]++;
            lo.keys[static_cast<std::size_t>(at)] = e.key;
            lo.values[static_cast<std::size_t>(at)] = e.value;
        }
        std::vector<Entry>().swap(rows[static_cast<std::size_t>(r)]);
    }
    return res;
}

}  // namespace bio
