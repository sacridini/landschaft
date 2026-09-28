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
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "bio/perlmath.hpp"
#include "bio/phylocom.hpp"
#include "bio/props.hpp"
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
                   std::vector<std::pair<std::string, KeyKind>> lists, i32 nbr, bool proc = false, bool tree = false) {
        CalcInfo c;
        c.name = std::move(name);
        c.scalars = std::move(scalars);
        c.lists = std::move(lists);
        c.uses_nbr_lists = nbr;
        c.uses_processing_element = proc;
        c.uses_tree = tree;
        r.push_back(std::move(c));
    };
    auto addt = [&](std::string name, std::vector<std::string> scalars,
                    std::vector<std::pair<std::string, KeyKind>> lists, i32 nbr) {
        add(std::move(name), std::move(scalars), std::move(lists), nbr, false, true);
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
    // Phylogenetic indices
    addt("calc_count_labels_on_tree", {"PHYLO_LABELS_ON_TREE_COUNT"}, {}, 1);
    addt("calc_labels_not_on_tree", {"PHYLO_LABELS_NOT_ON_TREE_N", "PHYLO_LABELS_NOT_ON_TREE_P"},
         {{"PHYLO_LABELS_NOT_ON_TREE", K::Label}}, 1);
    addt("calc_labels_on_tree", {}, {{"PHYLO_LABELS_ON_TREE", K::Label}}, 1);
    addt("calc_last_shared_ancestor_props",
         {"LAST_SHARED_ANCESTOR_DEPTH", "LAST_SHARED_ANCESTOR_DIST_TO_ROOT", "LAST_SHARED_ANCESTOR_DIST_TO_TIP",
          "LAST_SHARED_ANCESTOR_LENGTH", "LAST_SHARED_ANCESTOR_POS_REL"},
         {}, 1);
    addt("calc_pd", {"PD", "PD_P", "PD_P_per_taxon", "PD_per_taxon"}, {}, 1);
    addt("calc_pd_clade_contributions", {},
         {{"PD_CLADE_CONTR", K::Node}, {"PD_CLADE_CONTR_P", K::Node}, {"PD_CLADE_SCORE", K::Node}}, 1);
    addt("calc_pd_clade_loss", {},
         {{"PD_CLADE_LOSS_CONTR", K::Node}, {"PD_CLADE_LOSS_CONTR_P", K::Node}, {"PD_CLADE_LOSS_SCORE", K::Node}}, 1);
    addt("calc_pd_clade_loss_ancestral", {}, {{"PD_CLADE_LOSS_ANC", K::Node}, {"PD_CLADE_LOSS_ANC_P", K::Node}}, 1);
    addt("calc_pd_endemism", {"PD_ENDEMISM", "PD_ENDEMISM_P"}, {{"PD_ENDEMISM_WTS", K::Node}}, 1);
    addt("calc_pd_local", {"PD_LOCAL", "PD_LOCAL_P"}, {}, 1);
    addt("calc_pd_node_list", {}, {{"PD_INCLUDED_NODE_LIST", K::Node}}, 1);
    addt("calc_pd_terminal_node_count", {"PD_INCLUDED_TERMINAL_NODE_COUNT"}, {}, 1);
    addt("calc_pd_terminal_node_list", {}, {{"PD_INCLUDED_TERMINAL_NODE_LIST", K::Node}}, 1);
    addt("calc_pe", {"PE_WE", "PE_WE_P"}, {}, 1);
    addt("calc_pe_central", {"PEC_WE", "PEC_WE_P"}, {}, 1);
    addt("calc_pe_central_cwe", {"PEC_CWE", "PEC_CWE_PD"}, {}, 1);
    addt("calc_pe_central_lists", {},
         {{"PEC_LOCAL_RANGELIST", K::Node}, {"PEC_RANGELIST", K::Node}, {"PEC_WTLIST", K::Node}}, 1);
    addt("calc_pe_clade_contributions", {},
         {{"PE_CLADE_CONTR", K::Node}, {"PE_CLADE_CONTR_P", K::Node}, {"PE_CLADE_SCORE", K::Node}}, 1);
    addt("calc_pe_clade_loss", {},
         {{"PE_CLADE_LOSS_CONTR", K::Node}, {"PE_CLADE_LOSS_CONTR_P", K::Node}, {"PE_CLADE_LOSS_SCORE", K::Node}}, 1);
    addt("calc_pe_clade_loss_ancestral", {}, {{"PE_CLADE_LOSS_ANC", K::Node}, {"PE_CLADE_LOSS_ANC_P", K::Node}}, 1);
    addt("calc_pe_lists", {},
         {{"PE_LOCAL_RANGELIST", K::Node}, {"PE_RANGELIST", K::Node}, {"PE_WTLIST", K::Node}}, 1);
    addt("calc_pe_single", {"PE_WE_SINGLE", "PE_WE_SINGLE_P"}, {}, 1);
    addt("calc_phylo_abc", {"PHYLO_A", "PHYLO_ABC", "PHYLO_B", "PHYLO_C"}, {}, 2);
    addt("calc_phylo_abundance", {"PHYLO_ABUNDANCE"}, {{"PHYLO_ABUNDANCE_BRANCH_HASH", K::Node}}, 1);
    addt("calc_phylo_aed", {},
         {{"PHYLO_AED_LIST", K::Label}, {"PHYLO_ED_LIST", K::Label}, {"PHYLO_ES_LIST", K::Label}}, 1);
    addt("calc_phylo_aed_t", {"PHYLO_AED_T"}, {}, 1);
    addt("calc_phylo_aed_t_wtlists", {}, {{"PHYLO_AED_T_WTLIST", K::Label}, {"PHYLO_AED_T_WTLIST_P", K::Label}}, 1);
    addt("calc_phylo_corrected_weighted_endemism", {"PE_CWE"}, {}, 1);
    addt("calc_phylo_corrected_weighted_rarity", {"PHYLO_RARITY_CWR"}, {}, 1);
    addt("calc_phylo_jaccard", {"PHYLO_JACCARD"}, {}, 2);
    addt("calc_phylo_rw_turnover", {"PHYLO_RW_TURNOVER", "PHYLO_RW_TURNOVER_A", "PHYLO_RW_TURNOVER_B",
                                    "PHYLO_RW_TURNOVER_C"}, {}, 2);
    addt("calc_phylo_s2", {"PHYLO_S2"}, {}, 2);
    addt("calc_phylo_sorenson", {"PHYLO_SORENSON"}, {}, 2);
    // Phylogenetic indices (relative)
    addt("calc_labels_on_trimmed_tree", {}, {{"PHYLO_LABELS_ON_TRIMMED_TREE", K::Label}}, 1);
    addt("calc_labels_not_on_trimmed_tree", {"PHYLO_LABELS_NOT_ON_TRIMMED_TREE_N", "PHYLO_LABELS_NOT_ON_TRIMMED_TREE_P"},
         {{"PHYLO_LABELS_NOT_ON_TRIMMED_TREE", K::Label}}, 1);
    addt("calc_phylo_rpd1", {"PHYLO_RPD1", "PHYLO_RPD_DIFF1", "PHYLO_RPD_NULL1"}, {}, 1);
    addt("calc_phylo_rpd2", {"PHYLO_RPD2", "PHYLO_RPD_DIFF2", "PHYLO_RPD_NULL2"}, {}, 1);
    addt("calc_phylo_rpe1", {"PHYLO_RPE1", "PHYLO_RPE_DIFF1", "PHYLO_RPE_NULL1"}, {}, 1);
    addt("calc_phylo_rpe2", {"PHYLO_RPE2", "PHYLO_RPE_DIFF2", "PHYLO_RPE_NULL2"}, {}, 1);
    addt("calc_phylo_rpe_central", {"PHYLO_RPEC", "PHYLO_RPE_DIFFC", "PHYLO_RPE_NULLC"}, {}, 1);
    // Element properties (the per-property lists are added in run_indices)
    add("calc_lbprop_lists", {}, {}, 1);
    add("calc_lbprop_data", {}, {}, 1);
    add("calc_lbprop_hashes", {}, {}, 1);
    add("calc_lbprop_stats", {}, {{"LBPROP_STATS", K::Text}}, 1);
    add("calc_lbprop_quantiles", {}, {{"LBPROP_QUANTILES", K::Text}}, 1);
    add("calc_lbprop_gistar", {}, {{"LBPROP_GISTAR_LIST", K::Text}}, 1);
    add("calc_lbprop_hashes_abc2", {}, {}, 1);
    add("calc_lbprop_stats_abc2", {}, {{"LBPROP_STATS_ABC2", K::Text}}, 1);
    add("calc_lbprop_quantiles_abc2", {}, {{"LBPROP_QUANTILES_ABC2", K::Text}}, 1);
    add("calc_lbprop_gistar_abc2", {}, {{"LBPROP_GISTAR_LIST_ABC2", K::Text}}, 1);
    add("calc_gpprop_lists", {}, {}, 1);
    add("calc_gpprop_hashes", {}, {}, 1);
    add("calc_gpprop_stats", {}, {{"GPPROP_STATS_LIST", K::Text}}, 1);
    add("calc_gpprop_quantiles", {}, {{"GPPROP_QUANTILE_LIST", K::Text}}, 1);
    add("calc_gpprop_gistar", {}, {{"GPPROP_GISTAR_LIST", K::Text}}, 1);
    // Numeric labels
    add("calc_numeric_label_stats",
        {"NUM_CV", "NUM_KURT", "NUM_MAX", "NUM_MEAN", "NUM_MIN", "NUM_N", "NUM_RANGE", "NUM_SD", "NUM_SKEW"}, {}, 1);
    add("calc_numeric_label_other_means", {"NUM_GMEAN", "NUM_HMEAN"}, {}, 1);
    {
        std::vector<std::string> q;
        for (int v = 5; v <= 95; v += 5) {
            char buf[16];
            std::snprintf(buf, sizeof buf, "NUM_Q%03d", v);
            q.push_back(buf);
        }
        add("calc_numeric_label_quantiles", q, {}, 1);
    }
    add("calc_numeric_label_data", {}, {{"NUM_DATA_ARRAY", K::NumberArray}}, 1);
    add("calc_numeric_label_dissimilarity", {"NUMD_ABSMEAN", "NUMD_COUNT", "NUMD_VARIANCE"}, {}, 2);
    add("calc_num_labels_gistar", {"NUM_GISTAR"}, {}, 1);
    // Inter-event intervals (BiodiverseX::Indices::IEI)
    add("calc_iei_stats",
        {"IEI_CV", "IEI_GMEAN", "IEI_KURT", "IEI_MAX", "IEI_MEAN", "IEI_MIN", "IEI_N", "IEI_RANGE", "IEI_SD", "IEI_SKEW"},
        {}, 1);
    add("calc_iei_data", {}, {{"IEI_DATA_ARRAY", K::NumberArray}, {"IEI_DATA_HASH", K::Number}}, 1);
    // Matrix
    add("calc_matrix_stats",
        {"MX_KURT", "MX_MAXVALUE", "MX_MEAN", "MX_MEDIAN", "MX_MINVALUE", "MX_N", "MX_PCT05", "MX_PCT25", "MX_PCT75",
         "MX_PCT95", "MX_RANGE", "MX_SD", "MX_SKEW"},
        {{"MX_LABELS", K::Label}, {"MX_VALUES", K::NumberArray}}, 1);
    add("calc_compare_dissim_matrix_values", {"MXD_COUNT", "MXD_MEAN", "MXD_VARIANCE"},
        {{"MXD_LIST1", K::Label}, {"MXD_LIST2", K::Label}}, 2);
    add("calc_mx_rao_qe", {"MX_RAO_QE", "MX_RAO_TN"}, {{"MX_RAO_TLABELS", K::Label}}, 1);
    // PhyloCom
    for (const char* n : {"1", "2", "3"}) {
        const std::string d = n, a = "PMPD" + d, b = "PNTD" + d;
        addt("calc_phylo_mpd_mntd" + d,
             {a + "_MAX", a + "_MEAN", a + "_MIN", a + "_N", a + "_RMSD", a + "_VARIANCE", b + "_MAX", b + "_MEAN",
              b + "_MIN", b + "_N", b + "_RMSD", b + "_VARIANCE"},
             {}, 1);
        addt("calc_nri_nti" + d, {"PHYLO_NRI" + d, "PHYLO_NTI" + d}, {}, 1);
    }
    addt("calc_nri_nti_expected_values",
         {"PHYLO_NRI_NTI_SAMPLE_N", "PHYLO_NRI_SAMPLE_MEAN", "PHYLO_NRI_SAMPLE_SD", "PHYLO_NTI_SAMPLE_MEAN",
          "PHYLO_NTI_SAMPLE_SD"},
         {}, 1);
    addt("calc_net_vpd", {"PHYLO_NET_VPD"}, {}, 1);
    addt("calc_vpd_expected_values", {"PHYLO_NET_VPD_SAMPLE_MEAN", "PHYLO_NET_VPD_SAMPLE_N", "PHYLO_NET_VPD_SAMPLE_SD"},
         {}, 1);
    addt("calc_rwibald", {},
         {{"RWIBALD_CODES", K::Node}, {"RWIBALD_CODE_COUNTS", K::Number}, {"RWIBALD_DIFFS", K::Node},
          {"RWIBALD_METADATA", K::Text}, {"RWIBALD_RR_DIFFS", K::Node}},
         1);
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
// phylogenetic quantities of one neighbourhood, computed on demand

struct PhyloScratch {
    PathUnion pu;
    std::vector<std::vector<i32>> st_children;  // sub tree as hash (children in insertion order)
    std::vector<std::uint8_t> in_st;
    std::vector<i32> st_nodes;
    std::vector<double> lr;                     // local ranges (PE)
    std::vector<i32> lr_touched;
};

struct PhyloNbhd {
    const TreeView& t;
    const PhyloGlobals& g;
    const BaseDataView& bd;
    const Abc& abc;
    PhyloScratch& sc;

    // labels on the tree (label indices, ascending)
    bool have_lot = false;
    std::vector<i32> lot;
    const std::vector<i32>& labels_on_tree() {
        if (!have_lot) {
            for (i32 l : abc.lall)
                if (t.label_node[static_cast<std::size_t>(l)] >= 0) lot.push_back(l);
            have_lot = true;
        }
        return lot;
    }

    // PD: nodes on the paths of the labels on the tree (ascending) and their length sum
    bool have_pd = false;
    std::vector<i32> pd_nodes;
    double pd = NaN;
    void pd_union() {
        if (have_pd) return;
        sc.pu.reset(t.n);
        for (i32 l : labels_on_tree()) sc.pu.add_path(t, t.label_node[static_cast<std::size_t>(l)]);
        pd_nodes = sc.pu.nodes;
        std::sort(pd_nodes.begin(), pd_nodes.end());
        bool first = true;
        for (i32 node : pd_nodes) {
            const double len = t.length[static_cast<std::size_t>(node)];
            if (first) {
                pd = len;
                first = false;
            } else {
                pd += len;
            }
        }
        have_pd = true;
    }

    // union of the paths of the labels in a label list (keys only), ascending
    std::vector<i32> path_of(const std::vector<i32>& labels) {
        sc.pu.reset(t.n);
        for (i32 l : labels) {
            const i32 node = t.label_node[static_cast<std::size_t>(l)];
            if (node >= 0 && t.in_trimmed[static_cast<std::size_t>(node)]) sc.pu.add_path(t, node);
        }
        std::vector<i32> out = sc.pu.nodes;
        std::sort(out.begin(), out.end());
        return out;
    }

    // get_sub_tree_as_hash
    bool have_st = false;
    void sub_tree() {
        if (have_st) return;
        if (sc.in_st.size() != static_cast<std::size_t>(t.n)) {
            sc.in_st.assign(static_cast<std::size_t>(t.n), 0);
            sc.st_children.assign(static_cast<std::size_t>(t.n), {});
        }
        for (i32 node : sc.st_nodes) {
            sc.in_st[static_cast<std::size_t>(node)] = 0;
            sc.st_children[static_cast<std::size_t>(node)].clear();
        }
        sc.st_nodes.clear();
        for (i32 l : labels_on_tree()) {
            const i32 label = t.label_node[static_cast<std::size_t>(l)];
            if (t.parent[static_cast<std::size_t>(label)] < 0) continue;  // the root has no parent
            if (!sc.in_st[static_cast<std::size_t>(label)]) {
                sc.in_st[static_cast<std::size_t>(label)] = 1;
                sc.st_nodes.push_back(label);
            }
            sc.st_children[static_cast<std::size_t>(label)].clear();  // $subtree{$label} = []
            i32 node = label;
            i32 parent = t.parent[static_cast<std::size_t>(label)];
            while (parent >= 0) {
                const bool last = sc.in_st[static_cast<std::size_t>(parent)] != 0;
                if (!last) {
                    sc.in_st[static_cast<std::size_t>(parent)] = 1;
                    sc.st_nodes.push_back(parent);
                }
                sc.st_children[static_cast<std::size_t>(parent)].push_back(node);
                if (last) break;
                node = parent;
                parent = t.parent[static_cast<std::size_t>(node)];
            }
        }
        std::sort(sc.st_nodes.begin(), sc.st_nodes.end());
        have_st = true;
    }
    const std::vector<i32>& st_kids(i32 node) const { return sc.st_children[static_cast<std::size_t>(node)]; }

    // get_last_shared_ancestor_from_subtree
    i32 lsa() {
        sub_tree();
        i32 current = t.root;
        if (!sc.st_nodes.empty()) {
            while (st_kids(current).size() == 1) current = st_kids(current)[0];
        }
        return current;
    }

    // PE over the neighbourhood's groups
    bool have_pe = false;
    double pe_we = NaN;
    std::vector<i32> pe_nodes;       // ascending
    std::vector<double> pe_lr, pe_wt;
    void pe_for(const std::vector<i32>& groups, std::vector<i32>& nodes, std::vector<double>& lrs,
                std::vector<double>& wts, double* we) {
        if (sc.lr.size() != static_cast<std::size_t>(t.n)) sc.lr.assign(static_cast<std::size_t>(t.n), 0.0);
        sc.lr_touched.clear();
        double sum = NaN;
        for (i32 gi : groups) {
            const double s = g.gp_score[static_cast<std::size_t>(gi)];
            if (!std::isnan(s)) sum = num(sum) + s;
            for (i64 k = g.gp_ptr[static_cast<std::size_t>(gi)]; k < g.gp_ptr[static_cast<std::size_t>(gi) + 1]; ++k) {
                const i32 node = g.gp_nodes[static_cast<std::size_t>(k)];
                if (sc.lr[static_cast<std::size_t>(node)] == 0) sc.lr_touched.push_back(node);
                sc.lr[static_cast<std::size_t>(node)] += 1;
            }
        }
        std::sort(sc.lr_touched.begin(), sc.lr_touched.end());
        nodes = sc.lr_touched;
        lrs.clear();
        wts.clear();
        for (i32 node : nodes) {
            const double lr = sc.lr[static_cast<std::size_t>(node)];
            lrs.push_back(lr);
            wts.push_back(num(g.rw[static_cast<std::size_t>(node)]) * lr);
            sc.lr[static_cast<std::size_t>(node)] = 0;
        }
        if (we) *we = sum;
    }
    void pe() {
        if (have_pe) return;
        std::vector<i32> all;
        std::set_union(abc.g1.begin(), abc.g1.end(), abc.g2.begin(), abc.g2.end(), std::back_inserter(all));
        pe_for(all, pe_nodes, pe_lr, pe_wt, &pe_we);
        have_pe = true;
    }
    double pe_we_p() {
        pe();
        return t.trimmed_total_length == 0 ? NaN : num(pe_we) / t.trimmed_total_length;
    }
    double wt_of(i32 node) {
        pe();
        auto it = std::lower_bound(pe_nodes.begin(), pe_nodes.end(), node);
        return (it != pe_nodes.end() && *it == node) ? pe_wt[static_cast<std::size_t>(it - pe_nodes.begin())] : NaN;
    }

    // _calc_phylo_abc_lists: nodes (lengths) only in set 1, both, only in set 2
    bool have_abcl = false;
    std::vector<i32> a_list, b_list, c_list;
    void abc_lists() {
        if (have_abcl) return;
        const std::vector<i32> p1 = path_of(abc.l1);
        if (abc.C == 0) {
            b_list = p1;
        } else {
            const std::vector<i32> p2 = path_of(abc.l2);
            std::set_intersection(p1.begin(), p1.end(), p2.begin(), p2.end(), std::back_inserter(a_list));
            std::set_difference(p1.begin(), p1.end(), p2.begin(), p2.end(), std::back_inserter(b_list));
            std::set_difference(p2.begin(), p2.end(), p1.begin(), p1.end(), std::back_inserter(c_list));
        }
        have_abcl = true;
    }

    // clade contributions of a weight list (nodes of the sub tree)
    struct Clade {
        std::vector<i32> nodes;  // ascending
        std::vector<double> score, contr, contr_p;
    };
    Clade clade(const std::function<double(i32)>& weight, double p_score, double total) {
        sub_tree();
        Clade c;
        c.nodes = sc.st_nodes;
        const std::size_t m = c.nodes.size();
        c.score.assign(m, NaN);
        c.contr.assign(m, NaN);
        c.contr_p.assign(m, NaN);
        // by depth, deepest first; names in order within each depth
        std::vector<std::size_t> order(m);
        for (std::size_t i = 0; i < m; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return t.depth[static_cast<std::size_t>(c.nodes[a])] > t.depth[static_cast<std::size_t>(c.nodes[b])];
        });
        auto pos = [&](i32 node) {
            return static_cast<std::size_t>(std::lower_bound(c.nodes.begin(), c.nodes.end(), node) - c.nodes.begin());
        };
        for (std::size_t oi : order) {
            const i32 node = c.nodes[oi];
            double wt = weight(node);
            for (i32 kid : st_kids(node)) wt = num(wt) + num(c.score[pos(kid)]);
            c.contr[oi] = truthy(p_score) ? std::trunc(1e11 * num(wt) / p_score) / 1e11 : NaN;
            c.contr_p[oi] = truthy(total) ? std::trunc(1e11 * num(wt) / total) / 1e11 : NaN;
            c.score[oi] = wt;
        }
        return c;
    }
    // _calc_pd_pe_clade_loss
    struct Loss {
        std::vector<double> score, contr, contr_p;
    };
    Loss clade_loss(const Clade& c) {
        const std::size_t m = c.nodes.size();
        Loss L;
        L.score.assign(m, NaN);
        L.contr.assign(m, NaN);
        L.contr_p.assign(m, NaN);
        std::vector<std::uint8_t> done(m, 0);
        auto pos = [&](i32 node) {
            return static_cast<std::size_t>(std::lower_bound(c.nodes.begin(), c.nodes.end(), node) - c.nodes.begin());
        };
        for (std::size_t i = 0; i < m; ++i) {
            if (done[i]) continue;
            std::vector<std::size_t> anc{i};
            i32 parent = t.parent[static_cast<std::size_t>(c.nodes[i])];
            while (parent >= 0) {
                if (st_kids(parent).size() > 1) break;
                anc.push_back(pos(parent));
                parent = t.parent[static_cast<std::size_t>(parent)];
            }
            const std::size_t last = anc.back();
            for (std::size_t a : anc) {
                L.contr[a] = c.contr[last];
                L.score[a] = c.score[last];
                L.contr_p[a] = c.contr_p[last];
                done[a] = 1;
            }
        }
        return L;
    }
};

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
                        const std::vector<std::string>& calcs, bool parallel, const TreeView* tree) {
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
        {
            // one list per property
            auto per_prop = [&](const std::vector<BaseDataView::Property>& props, const std::string& pfx,
                                const std::string& sfx, KeyKind kind) {
                for (const auto& p : props) {
                    lslot[c].push_back(res.list_names.size());
                    res.list_names.push_back(pfx + p.name + sfx);
                    res.list_kinds.push_back(kind);
                }
            };
            const std::string& n = ci.name;
            if (n == "calc_lbprop_lists") per_prop(bd.label_props, "LBPROP_LIST_", "", K::Label);
            if (n == "calc_lbprop_data") per_prop(bd.label_props, "LBPROP_STATS_", "_DATA", K::NumberArray);
            if (n == "calc_lbprop_hashes") per_prop(bd.label_props, "LBPROP_STATS_", "_HASH", K::Text);
            if (n == "calc_lbprop_hashes_abc2") per_prop(bd.label_props, "LBPROP_STATS_", "_HASH2", K::Text);
            if (n == "calc_gpprop_lists") per_prop(bd.group_props, "GPPROP_STATS_", "_DATA", K::NumberArray);
            if (n == "calc_gpprop_hashes") per_prop(bd.group_props, "GPPROP_STATS_", "_HASH", K::Text);
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

    // phylogenetic globals
    PhyloGlobals pg;
    {
        bool need_tree = false, need_pe = false, need_aed = false;
        for (i32 id : ids) {
            const CalcInfo& ci = reg[static_cast<std::size_t>(id)];
            if (!ci.uses_tree) continue;
            need_tree = true;
            const std::string& nm = ci.name;
            if (nm.find("pe") != std::string::npos || nm == "calc_pd_endemism" || nm == "calc_phylo_rw_turnover" ||
                nm == "calc_phylo_corrected_weighted_endemism" || nm.find("rpe") != std::string::npos)
                need_pe = true;
            if (nm.find("aed") != std::string::npos || nm == "calc_phylo_corrected_weighted_rarity") need_aed = true;
        }
        if (need_tree) {
            if (!tree) throw std::invalid_argument("these calculations need a tree");
            build_phylo_globals(bd, *tree, pg, need_pe, need_aed, parallel);
            for (i32 id : ids)
                if (reg[static_cast<std::size_t>(id)].name == "calc_rwibald") build_rwibald(*tree, pg);
        }
    }
    const i32 key_rt = text_id("range_thresh"), key_nt = text_id("neg_len_thresh"), key_pt = text_id("pos_len_thresh");

    // element properties: text keys and global summaries (Gi*)
    static const char* lb_stats[] = {"count", "mean", "min", "max", "median", "sum", "skewness", "kurtosis", "sd", "iqr"};
    static const char* gp_stats[] = {"count", "mean", "min", "max", "median", "sum", "sd", "iqr"};
    static const char* prop_q[] = {"05", "10", "20", "30", "40", "50", "60", "70", "80", "90", "95"};
    auto upper = [](std::string s) {
        for (auto& ch : s) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        return s;
    };
    struct PropTexts {
        std::vector<i32> stat, quant;
        std::unordered_map<std::string, i32> by_str;  // hash keys (Perl strings of the values)
        i32 name = -1;
    };
    auto prop_texts = [&](const std::vector<BaseDataView::Property>& props, bool group) {
        std::vector<PropTexts> out;
        for (const auto& p : props) {
            PropTexts t;
            t.name = text_id(p.name);
            if (group) {
                for (const char* st : gp_stats) t.stat.push_back(text_id(p.name + "_" + upper(st)));
                for (const char* q : prop_q) t.quant.push_back(text_id("GPPROP_STATS_" + p.name + "_Q" + q));
            } else {
                for (const char* st : lb_stats) t.stat.push_back(text_id(p.name + "_" + upper(st)));
                for (const char* q : prop_q) t.quant.push_back(text_id(p.name + "_Q" + q));
            }
            for (double v : p.values) {
                const std::string k = perl_str(v);
                t.by_str.emplace(k, text_id(k));
            }
            out.push_back(std::move(t));
        }
        return out;
    };
    const auto lb_texts = prop_texts(bd.label_props, false);
    const auto gp_texts = prop_texts(bd.group_props, true);
    // get_element_properties_summary_stats: values weighted by 1 (or the label ranges)
    auto global_summary = [&](const BaseDataView::Property& p, const std::vector<double>* weight) {
        std::vector<double> acc(p.keys.size(), 0);
        std::vector<std::uint8_t> used(p.keys.size(), 0);
        for (std::size_t e = 0; e < p.element_key.size(); ++e) {
            const i32 k = p.element_key[e];
            if (k < 0) continue;
            acc[static_cast<std::size_t>(k)] += weight ? (*weight)[e] : 1.0;
            used[static_cast<std::size_t>(k)] = 1;
        }
        std::vector<double> d;
        std::vector<std::int64_t> w;
        for (std::size_t k = 0; k < acc.size(); ++k)
            if (used[k]) {
                d.push_back(p.values[k]);
                w.push_back(static_cast<std::int64_t>(acc[k]));
            }
        GlobalSummary g;
        bool any = !d.empty();
        WStats s;
        s.set(std::move(d), std::move(w));
        if (any) {
            g.mean = s.mean();
            g.sum = s.sum();
            g.sd = s.sd();
            g.count = s.count();
        }
        return std::make_pair(any, g);
    };
    std::vector<std::pair<bool, GlobalSummary>> lb_global, lb_global_rw, gp_global;
    for (const auto& p : bd.label_props) {
        lb_global.push_back(global_summary(p, nullptr));
        lb_global_rw.push_back(global_summary(p, &bd.label_range));
    }
    for (const auto& p : bd.group_props) gp_global.push_back(global_summary(p, nullptr));
    // _get_num_label_global_summary_stats
    GlobalSummary num_global;
    if (bd.labels_numeric) {
        double sumx = 0, sumxx = 0, n = 0;
        bool first = true;
        for (i64 l = 0; l < bd.n_labels; ++l) {
            const double count = bd.label_sample_count[static_cast<std::size_t>(l)];
            const double x = bd.label_values[static_cast<std::size_t>(l)];
            if (first) {
                sumx = count * x;
                sumxx = count * perl_pow(x, 2);
                n = count;
                first = false;
            } else {
                sumx += count * x;
                sumxx += count * perl_pow(x, 2);
                n += count;
            }
        }
        if (n != 0) {
            const double mean = sumx / n;
            double variance = sumxx - n * perl_pow(mean, 2);
            variance /= n - 1;
            variance = 0 < variance ? variance : 0.0;
            num_global.mean = mean;
            num_global.sd = perl_sqrt(variance);
        }
        num_global.sum = sumx;
        num_global.count = n;
    }

    // PhyloCom: path caches and expected values are shared across groups
    std::unique_ptr<PhyloCom> pc;
    bool pc_has_expected = false, pc_has_mpd = false;
    if (tree && !tree->pc_steps.empty()) {
        pc.reset(new PhyloCom(*tree, tree->pc_vpd, tree->pc_seed));
        for (i32 st : tree->pc_steps) (st == 0 ? pc_has_expected : pc_has_mpd) = true;
    }
    // On ultrametric trees path lengths depend on the groups processed before.
    // With two children per node they are settled by a pass in processing order
    // (with the expected values, which draw from one random stream); otherwise
    // groups are processed one after the other.
    bool internal_labels = false;
    if (pc)
        for (i32 node : tree->label_node)
            if (node >= 0 && !tree->terminal[static_cast<std::size_t>(node)]) internal_labels = true;
    const bool pc_stateful = pc && pc->ultrametric && (pc_has_mpd || (tree->pc_vpd && pc_has_expected));
    const bool pc_sequential = pc_stateful && (pc->multifurcating || internal_labels);
    if (pc_sequential) parallel = false;
    if (pc) pc->parallel_sampling = parallel;
    if (pc && !pc_sequential && (pc_has_expected || pc_stateful)) {
        Abc pa;
        pa.init(bd.n_labels);
        std::vector<i32> lot, nodes;
        for (i64 r = 0; r < np; ++r) {
            pa.clear();
            pa.g1.assign(set1.indices.begin() + set1.indptr[r], set1.indices.begin() + set1.indptr[r + 1]);
            pa.g2.clear();
            if (set2) pa.g2.assign(set2->indices.begin() + set2->indptr[r], set2->indices.begin() + set2->indptr[r + 1]);
            aggregate(bd, pa.g1, pa.l1, pa.in1, pa.c1, pa.s1);
            aggregate(bd, pa.g2, pa.l2, pa.in2, pa.c2, pa.s2);
            std::set_union(pa.l1.begin(), pa.l1.end(), pa.l2.begin(), pa.l2.end(), std::back_inserter(pa.lall));
            lot.clear();
            for (i32 l : pa.lall)
                if (tree->label_node[static_cast<std::size_t>(l)] >= 0) lot.push_back(l);
            for (i32 st : tree->pc_steps) {
                if (st == 0) {
                    pc->expected_step(lot.size());
                    continue;
                }
                if (!pc->ultrametric || lot.size() <= 1) continue;
                nodes.clear();
                for (i32 l : lot) {
                    const double v = st == 1 ? 1.0 : (st == 2 ? pa.all_count(l) : pa.all_samples(l));
                    if (v != 0) nodes.push_back(tree->label_node[static_cast<std::size_t>(l)]);
                }
                pc->touch(nodes);
            }
        }
        if (pc->ultrametric) pc->freeze();
    }

    std::string error;
    bool failed = false;
    lsm::parallel_for_dynamic(parallel && np > 1, np, [&](i64 r) {
        if (failed) return;
        try {
            thread_local Abc abc;
            thread_local PhyloScratch phylo_scratch;
            if (abc.in1.size() != static_cast<std::size_t>(bd.n_labels)) abc.init(bd.n_labels);
            abc.clear();
            abc.g1.assign(set1.indices.begin() + set1.indptr[r], set1.indices.begin() + set1.indptr[r + 1]);
            abc.g2.clear();
            abc.has_set2 = have_set2;
            if (have_set2) abc.g2.assign(set2->indices.begin() + set2->indptr[r], set2->indices.begin() + set2->indptr[r + 1]);
            if (bd.has_row_labels) {
                for (i64 k = bd.row_indptr[static_cast<std::size_t>(r)]; k < bd.row_indptr[static_cast<std::size_t>(r) + 1]; ++k) {
                    const i32 l = bd.row_labels[static_cast<std::size_t>(k)];
                    abc.l1.push_back(l);
                    abc.in1[l] = 1;
                    abc.c1[l] = static_cast<i32>(bd.row_counts[static_cast<std::size_t>(k)]);
                    abc.s1[l] = bd.row_samples[static_cast<std::size_t>(k)];
                }
            } else {
                aggregate(bd, abc.g1, abc.l1, abc.in1, abc.c1, abc.s1);
            }
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
            // phylogenetic pieces, built on first use
            std::unique_ptr<PhyloNbhd> ph;
            auto P = [&]() -> PhyloNbhd& {
                if (!ph) ph.reset(new PhyloNbhd{*tree, pg, bd, abc, phylo_scratch});
                return *ph;
            };
            auto abc_any_value = [&](i32 l) {
                return bd.abc_any_variant == 3 ? abc.all_samples(l)
                                               : (bd.abc_any_variant == 2 ? abc.all_count(l) : 1.0);
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

            // PhyloCom steps, in Biodiverse's order
            MpdResult pc_mpd[4];
            NriNti pc_exp;
            bool pc_exp_defined = false;
            if (pc) {
                for (i32 st : tree->pc_steps) {
                    const auto& lot = P().labels_on_tree();
                    if (st == 0) {
                        const NriNti* e = pc_sequential ? pc->expected_step(lot.size()) : pc->cached(lot.size());
                        if (e) {
                            pc_exp = *e;
                            pc_exp_defined = true;
                        }
                        continue;
                    }
                    auto value = [&](i32 l) {
                        return st == 1 ? 1.0 : (st == 2 ? abc.all_count(l) : abc.all_samples(l));
                    };
                    bool all_one = true;
                    for (i32 l : abc.lall)
                        if (value(l) != 1) all_one = false;
                    std::vector<i32> nodes;
                    std::vector<double> wts;
                    for (i32 l : lot) {
                        const double v = value(l);
                        if (v != 0) {
                            nodes.push_back(tree->label_node[static_cast<std::size_t>(l)]);
                            wts.push_back(v);
                        }
                    }
                    pc_mpd[st] = pc->mpd_mntd(nodes, wts, st != 1 && !all_one, 0, lot.size());
                }
            }
            auto zscore = [](double obs, double mean, double sd) {
                if (std::isnan(sd) || sd == 0) return NaN;  // division by zero in an eval
                return (num(obs) - num(mean)) / sd;
            };

            // statistics objects shared by the calculations of this group
            std::vector<WStats> lb_obj, lb2_obj, gp_obj;
            WStats num_obj;
            bool have_num = false;
            auto build_props = [&](const std::vector<BaseDataView::Property>& props, std::vector<WStats>& objs,
                                   int variant) {
                objs.assign(props.size(), WStats());
                for (std::size_t pi = 0; pi < props.size(); ++pi) {
                    const auto& p = props[pi];
                    std::vector<double> acc(p.keys.size(), 0);
                    std::vector<std::uint8_t> used(p.keys.size(), 0);
                    auto add = [&](i32 e, double w) {
                        const i32 k = p.element_key[static_cast<std::size_t>(e)];
                        if (k < 0) return;
                        acc[static_cast<std::size_t>(k)] += w;
                        used[static_cast<std::size_t>(k)] = 1;
                    };
                    if (variant == 0) {
                        // groups of both sets, once each
                        std::vector<i32> all;
                        std::set_union(abc.g1.begin(), abc.g1.end(), abc.g2.begin(), abc.g2.end(), std::back_inserter(all));
                        for (i32 g : all) add(g, 1);
                    } else {
                        for (i32 l : abc.lall) add(l, variant == 1 ? 1.0 : abc.all_count(l));
                    }
                    std::vector<double> d;
                    std::vector<std::int64_t> w;
                    for (std::size_t k = 0; k < acc.size(); ++k)
                        if (used[k]) {
                            d.push_back(p.values[k]);
                            w.push_back(static_cast<std::int64_t>(acc[k]));
                        }
                    objs[pi].set(std::move(d), std::move(w));
                }
            };
            auto lb_objects = [&]() -> std::vector<WStats>& {
                if (lb_obj.size() != bd.label_props.size() || lb_obj.empty()) build_props(bd.label_props, lb_obj, 1);
                return lb_obj;
            };
            auto lb2_objects = [&]() -> std::vector<WStats>& {
                if (lb2_obj.size() != bd.label_props.size() || lb2_obj.empty()) build_props(bd.label_props, lb2_obj, 2);
                return lb2_obj;
            };
            auto gp_objects = [&]() -> std::vector<WStats>& {
                if (gp_obj.size() != bd.group_props.size() || gp_obj.empty()) build_props(bd.group_props, gp_obj, 0);
                return gp_obj;
            };
            auto num_object = [&]() -> WStats& {
                if (!have_num) {
                    std::vector<double> d;
                    std::vector<std::int64_t> w;
                    for (i32 l : abc.lall) {
                        d.push_back(bd.label_values[static_cast<std::size_t>(l)]);
                        w.push_back(static_cast<std::int64_t>(abc.all_samples(l)));
                    }
                    num_obj.set(std::move(d), std::move(w));
                    have_num = true;
                }
                return num_obj;
            };
            // matrix lookups (get_value with the self similarity; get_defined_value_aa)
            auto mx_defined = [&](i32 l1, i32 l2) {
                const i32 a = bd.label_mx[static_cast<std::size_t>(l1)], b = bd.label_mx[static_cast<std::size_t>(l2)];
                return bd.mx_values[static_cast<std::size_t>(a) * static_cast<std::size_t>(bd.mx_n) + static_cast<std::size_t>(b)];
            };
            auto mx_value = [&](i32 l1, i32 l2) {
                const double v = mx_defined(l1, l2);
                return (std::isnan(v) && l1 == l2) ? bd.mx_self_similarity : v;
            };

            for (std::size_t cc = 0; cc < ids.size(); ++cc) {
                const std::size_t c = bd.exec_order.empty() ? cc : static_cast<std::size_t>(bd.exec_order[cc]);
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
                } else if (name == "calc_labels_on_tree") {
                    for (i32 l : P().labels_on_tree()) put(0, l, 1);
                } else if (name == "calc_count_labels_on_tree") {
                    set(0, static_cast<double>(P().labels_on_tree().size()));
                } else if (name == "calc_labels_not_on_tree") {
                    bool any_off = false;
                    for (std::size_t l = 0; l < static_cast<std::size_t>(bd.n_labels); ++l)
                        if (tree->label_node[l] < 0) {
                            any_off = true;
                            break;
                        }
                    if (!any_off) {
                        set(0, 0);
                        set(1, 0);
                    } else {
                        double off = 0;
                        for (i32 l : abc.lall)
                            if (tree->label_node[static_cast<std::size_t>(l)] < 0) {
                                off += 1;
                                put(0, l, abc_any_value(l));
                            }
                        const double q = safe_div(off, static_cast<double>(abc.lall.size()));
                        set(0, off);
                        set(1, truthy(q) ? q : 0.0);
                    }
                } else if (name == "calc_pd" || name == "calc_phylo_corrected_weighted_endemism" ||
                           name == "calc_phylo_corrected_weighted_rarity") {
                    PhyloNbhd& p = P();
                    p.pd_union();
                    const double pd = p.pd;
                    if (name == "calc_pd") {
                        const double pdp = truthy(pd) ? pd / tree->total_length : NaN;
                        const double rich = static_cast<double>(p.labels_on_tree().size());
                        set(0, pd);
                        set(1, pdp);
                        set(2, safe_div(pdp, rich));
                        set(3, safe_div(pd, rich));
                    } else if (name == "calc_phylo_corrected_weighted_endemism") {
                        p.pe();
                        set(0, truthy(pd) ? num(p.pe_we) / pd : NaN);
                    } else {
                        double aed_t = NaN;
                        for (i32 l : abc.lall) {
                            if (tree->label_node[static_cast<std::size_t>(l)] < 0 || std::isnan(pg.ed[static_cast<std::size_t>(l)])) continue;
                            aed_t = num(aed_t) + abc.all_samples(l) * num(pg.aed[static_cast<std::size_t>(l)]);
                        }
                        set(0, truthy(pd) ? num(aed_t) / pd : NaN);
                    }
                } else if (name == "calc_pd_node_list") {
                    PhyloNbhd& p = P();
                    p.pd_union();
                    for (i32 node : p.pd_nodes) put(0, node, tree->length[static_cast<std::size_t>(node)]);
                } else if (name == "calc_pd_terminal_node_list" || name == "calc_pd_terminal_node_count") {
                    PhyloNbhd& p = P();
                    p.pd_union();
                    double count = 0;
                    for (i32 node : p.pd_nodes) {
                        if (!tree->terminal[static_cast<std::size_t>(node)]) continue;
                        count += 1;
                        if (name == "calc_pd_terminal_node_list") put(0, node, tree->length[static_cast<std::size_t>(node)]);
                    }
                    if (name == "calc_pd_terminal_node_count") set(0, count);
                } else if (name == "calc_pd_local") {
                    PhyloNbhd& p = P();
                    p.pd_union();
                    double pd = p.pd;
                    double pdp = truthy(pd) ? pd / tree->total_length : NaN;
                    if (truthy(pd)) {
                        i32 anc = p.lsa();
                        if (!tree->terminal[static_cast<std::size_t>(anc)]) {
                            double sum = 0;
                            while (anc >= 0) {
                                sum += tree->length[static_cast<std::size_t>(anc)];
                                anc = tree->parent[static_cast<std::size_t>(anc)];
                            }
                            pd = pd - sum;
                            pdp = truthy(pd) ? (pd / tree->total_length) : 0.0;
                        }
                    }
                    set(0, pd);
                    set(1, pdp);
                } else if (name == "calc_last_shared_ancestor_props") {
                    PhyloNbhd& p = P();
                    const i32 anc0 = p.lsa();
                    auto path_sum = [&](i32 node) {
                        // sum (0, values %path_lengths): nodes of the path in name order
                        std::vector<i32> path;
                        while (node >= 0) {
                            path.push_back(node);
                            node = tree->parent[static_cast<std::size_t>(node)];
                        }
                        std::sort(path.begin(), path.end());
                        double s = 0;
                        for (i32 nd : path) s += tree->length[static_cast<std::size_t>(nd)];
                        return s;
                    };
                    const double depth = tree->depth[static_cast<std::size_t>(anc0)];
                    const double length = tree->length[static_cast<std::size_t>(anc0)];
                    double tips = 0;
                    if (!tree->terminal[static_cast<std::size_t>(anc0)]) {
                        const double to_root = anc0 == tree->root ? 0.0 : path_sum(anc0);
                        for (i32 l : abc.lall) {
                            const i32 node = tree->label_node[static_cast<std::size_t>(l)];
                            if (node < 0 || !p.sc.in_st[static_cast<std::size_t>(node)]) continue;
                            tips = std::max(tips, path_sum(node));
                        }
                        tips -= to_root;
                    }
                    double to_root_walk = 0;
                    if (anc0 != tree->root) {
                        i32 a = anc0;
                        while (a >= 0) {
                            to_root_walk += tree->length[static_cast<std::size_t>(a)];
                            a = tree->parent[static_cast<std::size_t>(a)];
                        }
                    }
                    const double rel = (truthy(to_root_walk) || truthy(tips)) ? to_root_walk / (to_root_walk + tips) : 0.0;
                    set(0, depth);
                    set(1, to_root_walk);
                    set(2, tips);
                    set(3, length);
                    set(4, rel);
                } else if (name == "calc_pe") {
                    PhyloNbhd& p = P();
                    p.pe();
                    set(0, p.pe_we);
                    set(1, p.pe_we_p());
                } else if (name == "calc_pe_lists") {
                    PhyloNbhd& p = P();
                    p.pe();
                    for (std::size_t i = 0; i < p.pe_nodes.size(); ++i) {
                        const i32 node = p.pe_nodes[i];
                        put(0, node, p.pe_lr[i]);
                        put(1, node, pg.range[static_cast<std::size_t>(node)]);
                        put(2, node, p.pe_wt[i]);
                    }
                } else if (name == "calc_pe_single") {
                    PhyloNbhd& p = P();
                    p.pe();
                    double single = NaN;
                    for (i32 node : p.pe_nodes) {
                        const double r = pg.range[static_cast<std::size_t>(node)];
                        if (num(r) == 0) throw std::runtime_error("Illegal division by zero");
                        single = num(single) + tree->length[static_cast<std::size_t>(node)] / r;
                    }
                    set(0, single);
                    set(1, defined(single) ? single / tree->trimmed_total_length : NaN);
                } else if (name == "calc_pd_endemism") {
                    PhyloNbhd& p = P();
                    p.pe();
                    double pde = NaN;
                    for (std::size_t i = 0; i < p.pe_nodes.size(); ++i) {
                        const i32 node = p.pe_nodes[i];
                        if (pg.range[static_cast<std::size_t>(node)] != p.pe_lr[i]) continue;
                        pde = num(pde) + p.pe_wt[i];
                        put(0, node, p.pe_wt[i]);
                    }
                    set(0, pde);
                    set(1, (defined(pde) && truthy(tree->trimmed_total_length)) ? pde / tree->trimmed_total_length : NaN);
                } else if (name == "calc_pe_central" || name == "calc_pe_central_lists" || name == "calc_pe_central_cwe") {
                    PhyloNbhd& p = P();
                    p.pe();
                    p.abc_lists();
                    // PEC_WE: PE minus the weights of nodes only in set 2
                    double sub = 0;
                    for (i32 node : p.c_list) sub += num(p.wt_of(node));
                    const double pec = num(p.pe_we) - sub;
                    const double pec_we = (defined(p.pe_we) || sub != 0) ? pec : NaN;
                    // PEC lists: without the set 2 only nodes when there are some
                    std::vector<std::size_t> keep;
                    if (!p.c_list.empty()) {
                        std::vector<i32> ab;
                        std::set_union(p.a_list.begin(), p.a_list.end(), p.b_list.begin(), p.b_list.end(),
                                       std::back_inserter(ab));
                        for (std::size_t i = 0; i < p.pe_nodes.size(); ++i)
                            if (std::binary_search(ab.begin(), ab.end(), p.pe_nodes[i])) keep.push_back(i);
                    } else {
                        for (std::size_t i = 0; i < p.pe_nodes.size(); ++i) keep.push_back(i);
                    }
                    if (name == "calc_pe_central") {
                        set(0, pec_we);
                        set(1, truthy(pec_we) ? pec_we / tree->trimmed_total_length : NaN);
                    } else if (name == "calc_pe_central_lists") {
                        for (std::size_t i : keep) {
                            const i32 node = p.pe_nodes[i];
                            put(0, node, p.pe_lr[i]);
                            put(1, node, pg.range[static_cast<std::size_t>(node)]);
                            put(2, node, p.pe_wt[i]);
                        }
                    } else {
                        p.pd_union();
                        double pdsum = NaN;
                        for (std::size_t i : keep) {
                            const i32 node = p.pe_nodes[i];
                            const bool in_pd = std::binary_search(p.pd_nodes.begin(), p.pd_nodes.end(), node);
                            pdsum = num(pdsum) + (in_pd ? tree->length[static_cast<std::size_t>(node)] : 0.0);
                        }
                        set(0, truthy(pdsum) ? num(pec_we) / pdsum : NaN);
                        set(1, pdsum);
                    }
                } else if (name == "calc_pd_clade_contributions" || name == "calc_pd_clade_loss" ||
                           name == "calc_pd_clade_loss_ancestral" || name == "calc_pe_clade_contributions" ||
                           name == "calc_pe_clade_loss" || name == "calc_pe_clade_loss_ancestral") {
                    PhyloNbhd& p = P();
                    const bool is_pe = name.find("_pe_") != std::string::npos;
                    PhyloNbhd::Clade c;
                    if (is_pe) {
                        p.pe();
                        c = p.clade([&](i32 node) { return p.wt_of(node); }, num(p.pe_we), tree->trimmed_total_length);
                    } else {
                        p.pd_union();
                        c = p.clade(
                            [&](i32 node) {
                                return std::binary_search(p.pd_nodes.begin(), p.pd_nodes.end(), node)
                                           ? tree->length[static_cast<std::size_t>(node)]
                                           : NaN;
                            },
                            num(p.pd), tree->total_length);
                    }
                    if (name.find("contributions") != std::string::npos) {
                        for (std::size_t i = 0; i < c.nodes.size(); ++i) {
                            put(0, c.nodes[i], c.contr[i]);
                            put(1, c.nodes[i], c.contr_p[i]);
                            put(2, c.nodes[i], c.score[i]);
                        }
                    } else {
                        const PhyloNbhd::Loss L = p.clade_loss(c);
                        if (name.find("ancestral") == std::string::npos) {
                            for (std::size_t i = 0; i < c.nodes.size(); ++i) {
                                put(0, c.nodes[i], L.contr[i]);
                                put(1, c.nodes[i], L.contr_p[i]);
                                put(2, c.nodes[i], L.score[i]);
                            }
                        } else {
                            for (std::size_t i = 0; i < c.nodes.size(); ++i) {
                                const double score = num(L.score[i]) - num(c.score[i]);
                                put(0, c.nodes[i], score);
                                put(1, c.nodes[i], truthy(L.score[i]) ? score / L.score[i] : 0.0);
                            }
                        }
                    }
                } else if (name == "calc_phylo_abc" || name == "calc_phylo_sorenson" || name == "calc_phylo_jaccard" ||
                           name == "calc_phylo_s2") {
                    PhyloNbhd& p = P();
                    const std::vector<i32> l1 = abc.g1.empty() ? std::vector<i32>{} : p.path_of(abc.l1);
                    const std::vector<i32> l2 = abc.g2.empty() ? std::vector<i32>{} : p.path_of(abc.l2);
                    double aa = 0, bb = 0, cc = 0;
                    if (bd.pairwise_mode) {
                        // totals of each set minus the shared length
                        auto total = [&](const std::vector<i32>& l) {
                            double s = 0;
                            bool first = true;
                            for (i32 node : l) {
                                const double v = tree->length[static_cast<std::size_t>(node)];
                                s = first ? v : s + v;
                                first = false;
                            }
                            return s;
                        };
                        for (i32 node : l1)
                            if (std::binary_search(l2.begin(), l2.end(), node)) aa += tree->length[static_cast<std::size_t>(node)];
                        bb = total(l1) - aa;
                        cc = total(l2) - aa;
                    } else {
                        for (i32 node : l1) {
                            if (std::binary_search(l2.begin(), l2.end(), node))
                                aa += tree->length[static_cast<std::size_t>(node)];
                            else
                                bb += tree->length[static_cast<std::size_t>(node)];
                        }
                        for (i32 node : l2)
                            if (!std::binary_search(l1.begin(), l1.end(), node))
                                cc += tree->length[static_cast<std::size_t>(node)];
                    }
                    const double abcs = aa + bb + cc;
                    if (name == "calc_phylo_abc") {
                        set(0, aa);
                        set(1, abcs);
                        set(2, bb);
                        set(3, cc);
                    } else {
                        double v = NaN;
                        if (truthy(aa) || (truthy(bb) && truthy(cc))) {
                            if (name == "calc_phylo_sorenson") {
                                const double q = safe_div(2 * aa, aa + abcs);
                                v = defined(q) ? 1 - q : NaN;
                            } else if (name == "calc_phylo_jaccard") {
                                const double q = safe_div(aa, abcs);
                                v = defined(q) ? 1 - q : NaN;
                            } else {
                                const double q = safe_div(aa, aa + std::min(bb, cc));
                                v = defined(q) ? 1 - q : NaN;
                            }
                        }
                        set(0, v);
                    }
                } else if (name == "calc_phylo_rw_turnover") {
                    PhyloNbhd& p = P();
                    std::vector<i32> n1, n2;
                    std::vector<double> lr1, lr2, w1, w2;
                    p.pe_for(abc.g1, n1, lr1, w1, nullptr);
                    p.pe_for(abc.g2, n2, lr2, w2, nullptr);
                    if (n1.empty() || n2.empty()) {
                        set(0, NaN);
                        set(1, NaN);
                        set(2, NaN);
                        set(3, NaN);
                    } else {
                        double aa = 0, bb = 0, cc = 0;
                        for (std::size_t i = 0; i < n1.size(); ++i) {
                            auto it = std::lower_bound(n2.begin(), n2.end(), n1[i]);
                            if (it != n2.end() && *it == n1[i])
                                aa += (w1[i] + w2[static_cast<std::size_t>(it - n2.begin())]);
                            else
                                bb += w1[i];
                        }
                        for (std::size_t j = 0; j < n2.size(); ++j)
                            if (!std::binary_search(n1.begin(), n1.end(), n2[j])) cc += w2[j];
                        double turnover = NaN;
                        if ((truthy(aa) || truthy(bb)) && (truthy(aa) || truthy(cc))) {
                            const double v = 1 - (aa / (aa + bb + cc));
                            turnover = truthy(v) ? v : 0.0;
                        }
                        if (truthy(turnover) && turnover < 0) turnover = 0;
                        set(0, turnover);
                        set(1, aa);
                        set(2, bb);
                        set(3, cc);
                    }
                } else if (name == "calc_phylo_abundance") {
                    PhyloNbhd& p = P();
                    std::map<i32, double> branch;
                    double total = NaN;
                    for (i32 l : p.labels_on_tree()) {
                        std::vector<i32> path;
                        i32 node = tree->label_node[static_cast<std::size_t>(l)];
                        while (node >= 0) {
                            path.push_back(node);
                            node = tree->parent[static_cast<std::size_t>(node)];
                        }
                        std::sort(path.begin(), path.end());
                        const double ab = abc.all_samples(l);
                        for (i32 nd : path) {
                            const double val = ab * tree->length[static_cast<std::size_t>(nd)];
                            branch[nd] += val;
                            total = num(total) + val;
                        }
                    }
                    set(0, total);
                    for (const auto& kv : branch) put(0, kv.first, kv.second);
                } else if (name == "calc_phylo_aed") {
                    for (i32 l : abc.lall) {
                        const double a = pg.aed[static_cast<std::size_t>(l)];
                        if (tree->label_node[static_cast<std::size_t>(l)] < 0 || std::isnan(pg.ed[static_cast<std::size_t>(l)])) continue;
                        put(0, l, a);
                        put(1, l, pg.ed[static_cast<std::size_t>(l)]);
                        put(2, l, pg.es[static_cast<std::size_t>(l)]);
                    }
                } else if (name == "calc_phylo_aed_t" || name == "calc_phylo_aed_t_wtlists") {
                    double aed_t = NaN;
                    std::vector<std::pair<i32, double>> wts;
                    for (i32 l : abc.lall) {
                        if (tree->label_node[static_cast<std::size_t>(l)] < 0 || std::isnan(pg.ed[static_cast<std::size_t>(l)])) continue;
                        const double w = abc.all_samples(l) * num(pg.aed[static_cast<std::size_t>(l)]);
                        wts.emplace_back(l, w);
                        aed_t = num(aed_t) + w;
                    }
                    if (name == "calc_phylo_aed_t") {
                        set(0, aed_t);
                    } else {
                        for (const auto& lw : wts) put(0, lw.first, lw.second);
                        for (const auto& lw : wts) put(1, lw.first, lw.second / aed_t);
                    }
                } else if (name == "calc_labels_on_trimmed_tree" || name == "calc_labels_not_on_trimmed_tree") {
                    double off = 0;
                    for (i32 l : abc.lall) {
                        const i32 node = tree->label_node[static_cast<std::size_t>(l)];
                        const bool on = node >= 0 && tree->in_trimmed[static_cast<std::size_t>(node)];
                        if (name == "calc_labels_on_trimmed_tree") {
                            if (on) put(0, l, 1);
                        } else if (!on) {
                            off += 1;
                            put(0, l, abc_any_value(l));
                        }
                    }
                    if (name == "calc_labels_not_on_trimmed_tree") {
                        const double q = safe_div(off, static_cast<double>(abc.lall.size()));
                        set(0, off);
                        set(1, truthy(q) ? q : 0.0);
                    }
                } else if (name == "calc_phylo_rpd1" || name == "calc_phylo_rpd2") {
                    PhyloNbhd& p = P();
                    p.pd_union();
                    const double pdp = truthy(p.pd) ? p.pd / tree->total_length : NaN;
                    double null_;
                    if (name == "calc_phylo_rpd1") {
                        null_ = safe_div(static_cast<double>(p.labels_on_tree().size()), tree->n_terminals);
                    } else {
                        double eq = 0;
                        for (i32 node : p.pd_nodes)
                            if (tree->zero_length_count == 0 || tree->length[static_cast<std::size_t>(node)] != 0) eq += 1;
                        null_ = safe_div(eq, tree->eq_full_total);
                    }
                    set(0, safe_div(pdp, null_));
                    set(1, tree->total_length * (num(pdp) - num(null_)));
                    set(2, null_);
                } else if (name == "calc_phylo_rpe1") {
                    PhyloNbhd& p = P();
                    p.pe();
                    const Endemism& e = get_endw();
                    double we = NaN;
                    std::size_t wi = 0;
                    for (i32 l : abc.lall) {
                        const i32 node = tree->label_node[static_cast<std::size_t>(l)];
                        if (node < 0 || !tree->in_trimmed[static_cast<std::size_t>(node)]) continue;
                        while (wi < e.wts.size() && e.wts[wi].first < l) ++wi;
                        if (wi < e.wts.size() && e.wts[wi].first == l) we = num(we) + e.wts[wi].second;
                    }
                    const double pep = p.pe_we_p();
                    const double null_ = safe_div(we, tree->n_trimmed_terminals);
                    set(0, safe_div(pep, null_));
                    set(1, tree->trimmed_total_length * (num(pep) - num(null_)));
                    set(2, null_);
                } else if (name == "calc_phylo_rpe2" || name == "calc_phylo_rpe_central") {
                    PhyloNbhd& p = P();
                    p.pe();
                    auto ri = [&](i32 node) {
                        const double r = pg.range[static_cast<std::size_t>(node)];
                        if (std::isnan(r)) return NaN;
                        return r != 0 ? (tree->length[static_cast<std::size_t>(node)] != 0 ? 1.0 : 0.0) / r : 0.0;
                    };
                    const bool central_is_rpe2 = abc.g2.empty() || abc.C == 0;
                    double pep = NaN, pe_null = NaN;
                    if (name == "calc_phylo_rpe2" || central_is_rpe2) {
                        pep = p.pe_we_p();
                        std::vector<i32> all;
                        std::set_union(abc.g1.begin(), abc.g1.end(), abc.g2.begin(), abc.g2.end(), std::back_inserter(all));
                        for (i32 gi : all) {
                            double gs = NaN;
                            for (i64 k = pg.gp_ptr[static_cast<std::size_t>(gi)]; k < pg.gp_ptr[static_cast<std::size_t>(gi) + 1]; ++k)
                                gs = num(gs) + num(ri(pg.gp_nodes[static_cast<std::size_t>(k)]));
                            if (truthy(gs)) gs *= tree->default_eq_len;
                            if (defined(gs)) pe_null = num(pe_null) + gs;
                        }
                        if (central_is_rpe2 && name == "calc_phylo_rpe_central") {
                            // Biodiverse passes the central PE here (equal when nothing is only in set 2)
                            p.abc_lists();
                            double sub = 0;
                            for (i32 node : p.c_list) sub += num(p.wt_of(node));
                            const double pec = num(p.pe_we) - sub;
                            pep = truthy(pec) ? pec / tree->trimmed_total_length : NaN;
                        }
                    } else {
                        // central: nodes of the central PE lists, with their local ranges
                        p.abc_lists();
                        double sub = 0;
                        for (i32 node : p.c_list) sub += num(p.wt_of(node));
                        const double pec = num(p.pe_we) - sub;
                        pep = truthy(pec) ? pec / tree->trimmed_total_length : NaN;
                        std::vector<i32> ab;
                        std::set_union(p.a_list.begin(), p.a_list.end(), p.b_list.begin(), p.b_list.end(),
                                       std::back_inserter(ab));
                        const bool single = (abc.g1.size() + abc.g2.size()) == 1;
                        for (std::size_t i = 0; i < p.pe_nodes.size(); ++i) {
                            const i32 node = p.pe_nodes[i];
                            if (!std::binary_search(ab.begin(), ab.end(), node)) continue;
                            pe_null = num(pe_null) + (single ? num(ri(node)) : num(ri(node)) * p.pe_lr[i]);
                        }
                        if (truthy(pe_null)) pe_null *= tree->default_eq_len;
                    }
                    if (name == "calc_phylo_rpe2" && !defined(pep)) {
                        set(0, NaN);
                        set(1, NaN);
                        set(2, NaN);
                    } else {
                        const double null_ = safe_div(pe_null, tree->eq_trimmed_total);
                        set(0, safe_div(pep, null_));
                        set(1, tree->trimmed_total_length * (num(pep) - num(null_)));
                        set(2, null_);
                    }
                } else if (name == "calc_lbprop_lists") {
                    for (std::size_t pi = 0; pi < bd.label_props.size(); ++pi) {
                        const auto& p = bd.label_props[pi];
                        for (i32 l : abc.lall) {
                            const i32 k = p.element_key[static_cast<std::size_t>(l)];
                            if (k >= 0)
                                put(pi, l, p.values[static_cast<std::size_t>(k)]);
                            else if (k == -2)
                                put(pi, l, NaN);  // set but undefined
                        }
                    }
                } else if (name == "calc_lbprop_data" || name == "calc_gpprop_lists") {
                    auto& objs = name == "calc_lbprop_data" ? lb_objects() : gp_objects();
                    for (std::size_t pi = 0; pi < objs.size(); ++pi) {
                        const auto& d = objs[pi].data();
                        const auto& w = objs[pi].weights();
                        for (std::size_t i = 0; i < d.size(); ++i)
                            for (std::int64_t r = 0; r < w[i]; ++r) put(pi, d[i], 0);
                    }
                } else if (name == "calc_lbprop_hashes" || name == "calc_lbprop_hashes_abc2" || name == "calc_gpprop_hashes") {
                    const bool gp = name == "calc_gpprop_hashes";
                    auto& objs = gp ? gp_objects() : (name == "calc_lbprop_hashes" ? lb_objects() : lb2_objects());
                    for (std::size_t pi = 0; pi < objs.size(); ++pi) {
                        // @h{@data} = @wts: a later value wins for the same key
                        std::map<std::string, std::pair<double, double>> h;
                        const auto& d = objs[pi].data();
                        const auto& w = objs[pi].weights();
                        for (std::size_t i = 0; i < d.size(); ++i) h[perl_str(d[i])] = {d[i], static_cast<double>(w[i])};
                        const auto& by_str = (gp ? gp_texts : lb_texts)[pi].by_str;
                        for (const auto& kv : h) {
                            auto f = by_str.find(kv.first);
                            if (f != by_str.end()) put(pi, f->second, kv.second.second);
                        }
                    }
                } else if (name == "calc_lbprop_stats" || name == "calc_lbprop_stats_abc2" || name == "calc_gpprop_stats") {
                    const bool gp = name == "calc_gpprop_stats";
                    auto& objs = gp ? gp_objects() : (name == "calc_lbprop_stats" ? lb_objects() : lb2_objects());
                    for (std::size_t pi = 0; pi < objs.size(); ++pi) {
                        WStats& s = objs[pi];
                        const auto& t = (gp ? gp_texts : lb_texts)[pi].stat;
                        const std::size_t ns_ = gp ? 8 : 10;
                        for (std::size_t k = 0; k < ns_; ++k) {
                            const std::string st = gp ? gp_stats[k] : lb_stats[k];
                            double v = NaN;
                            if (st == "count") v = s.count();
                            else if (st == "mean") v = s.mean();
                            else if (st == "min") v = s.min();
                            else if (st == "max") v = s.max();
                            else if (st == "median") v = s.median();
                            else if (st == "sum") v = s.sum();
                            else if (st == "skewness") v = s.skewness();
                            else if (st == "kurtosis") v = s.kurtosis();
                            else if (st == "sd") v = s.sd();
                            else if (st == "iqr") v = s.iqr();
                            put(0, t[k], v);
                        }
                    }
                } else if (name == "calc_lbprop_quantiles" || name == "calc_lbprop_quantiles_abc2" ||
                           name == "calc_gpprop_quantiles") {
                    const bool gp = name == "calc_gpprop_quantiles";
                    auto& objs = gp ? gp_objects() : (name == "calc_lbprop_quantiles" ? lb_objects() : lb2_objects());
                    for (std::size_t pi = 0; pi < objs.size(); ++pi) {
                        const auto& t = (gp ? gp_texts : lb_texts)[pi].quant;
                        for (std::size_t k = 0; k < 11; ++k) put(0, t[k], objs[pi].percentile(std::atof(prop_q[k])));
                    }
                } else if (name == "calc_lbprop_gistar" || name == "calc_lbprop_gistar_abc2" || name == "calc_gpprop_gistar") {
                    const bool gp = name == "calc_gpprop_gistar";
                    auto& objs = gp ? gp_objects() : (name == "calc_lbprop_gistar" ? lb_objects() : lb2_objects());
                    const auto& glob = gp ? gp_global : (name == "calc_lbprop_gistar" ? lb_global : lb_global_rw);
                    for (std::size_t pi = 0; pi < objs.size(); ++pi) {
                        if (!glob[pi].first) continue;
                        put(0, (gp ? gp_texts : lb_texts)[pi].name, gistar(glob[pi].second, objs[pi]));
                    }
                } else if (name == "calc_numeric_label_stats") {
                    WStats& s = num_object();
                    const double n = s.count();
                    if (!std::isnan(n) && n != 0) {
                        const double sd_ = s.sd(), mean_ = s.mean();
                        const double cv = mean_ != 0 ? sd_ / mean_ : NaN;
                        set(3, s.mean());
                        set(7, s.sd());
                        set(5, n);
                        set(6, s.sample_range());
                        set(8, n > 2 ? s.skewness() : NaN);
                        set(1, n > 3 ? s.kurtosis() : NaN);
                        set(0, cv);
                        set(4, s.min());
                        set(2, s.max());
                    } else {
                        set(5, n);
                    }
                } else if (name == "calc_numeric_label_other_means") {
                    WStats& s = num_object();
                    const double n = s.count();
                    if (!std::isnan(n) && n != 0) {
                        const double g = s.geometric_mean();
                        const double h = s.harmonic_mean();
                        set(0, g);
                        set(1, h);
                    }
                } else if (name == "calc_numeric_label_quantiles") {
                    WStats& s = num_object();
                    for (int k = 0; k < 19; ++k) set(static_cast<std::size_t>(k), s.percentile(5.0 * (k + 1)));
                } else if (name == "calc_numeric_label_data") {
                    WStats& s = num_object();
                    s.median();
                    const auto& d = s.data();
                    const auto& w = s.weights();
                    for (std::size_t i = 0; i < d.size(); ++i)
                        for (std::int64_t r = 0; r < w[i]; ++r) put(0, d[i], 0);
                } else if (name == "calc_iei_stats" || name == "calc_iei_data") {
                    // intervals between the labels (event times) of each group, groups in name order
                    std::vector<i32> groups;
                    std::set_union(abc.g1.begin(), abc.g1.end(), abc.g2.begin(), abc.g2.end(), std::back_inserter(groups));
                    std::vector<double> data;
                    std::map<std::string, std::pair<double, double>> hash;  // key -> (interval, count)
                    auto bump = [&](double iv, double by) {
                        auto& e = hash[perl_str(iv)];
                        e.first = iv;
                        e.second += by;
                    };
                    auto zeros = [&](double count) {
                        if (!(count > 1)) return;
                        const double z = count - 1;
                        for (i64 k = 0; k < static_cast<i64>(z); ++k) data.push_back(0.0);
                        bump(0.0, z);
                    };
                    for (i32 g : groups) {
                        std::vector<std::pair<i32, double>> ls;
                        for (i64 k = bd.indptr[g]; k < bd.indptr[g + 1]; ++k)
                            ls.push_back({bd.indices[static_cast<std::size_t>(k)], bd.counts[static_cast<std::size_t>(k)]});
                        if (ls.size() < 2) continue;
                        std::stable_sort(ls.begin(), ls.end(), [&](const auto& a, const auto& b) {
                            return bd.label_values[static_cast<std::size_t>(a.first)] <
                                   bd.label_values[static_cast<std::size_t>(b.first)];
                        });
                        zeros(ls[0].second);
                        for (std::size_t i = 1; i < ls.size(); ++i) {
                            const double iv = bd.label_values[static_cast<std::size_t>(ls[i].first)] -
                                              bd.label_values[static_cast<std::size_t>(ls[i - 1].first)];
                            data.push_back(iv);
                            bump(iv, 1);
                            zeros(ls[i].second);
                        }
                    }
                    DStats s;
                    s.add(data);
                    s.sort_data();
                    if (name == "calc_iei_data") {
                        for (double v : s.data()) put(0, v, 0);
                        for (const auto& kv : hash) put(1, kv.second.first, kv.second.second);
                    } else {
                        const double n = s.count();
                        set(6, n);
                        if (n != 0) {
                            const double sd = s.sd(), mean = s.mean();
                            set(0, (!std::isnan(sd) && mean != 0) ? sd / mean : NaN);
                            double gm = 1;
                            const double ex = 1 / n;
                            for (double v : s.data()) {
                                if (v < 0) {
                                    gm = NaN;
                                    break;
                                }
                                gm *= perl_pow(v, ex);
                            }
                            set(1, gm);
                            set(4, mean);
                            set(8, sd);
                            set(7, s.sample_range());
                            set(9, s.skewness());
                            set(2, s.kurtosis());
                            set(5, s.min());
                            set(3, s.max());
                        }
                    }
                } else if (name == "calc_num_labels_gistar") {
                    set(0, gistar(num_global, num_object()));
                } else if (name == "calc_numeric_label_dissimilarity") {
                    // label_hash1/2 of calc_abc3 (sample counts per set)
                    const std::vector<i32>* la = &abc.l1;
                    const std::vector<i32>* lbv = &abc.l2;
                    const std::vector<double>* sa = &abc.s1;
                    const std::vector<double>* sb = &abc.s2;
                    if (la->empty() || lbv->empty()) {
                        set(0, NaN);
                        set(1, NaN);
                        set(2, NaN);
                    } else {
                        if (la->size() > lbv->size()) {
                            std::swap(la, lbv);
                            std::swap(sa, sb);
                        }
                        auto numeric_sorted = [&](const std::vector<i32>& ls) {
                            std::vector<i32> v = ls;  // string order, then a stable numeric sort
                            std::stable_sort(v.begin(), v.end(), [&](i32 a, i32 b) {
                                return bd.label_values[static_cast<std::size_t>(a)] < bd.label_values[static_cast<std::size_t>(b)];
                            });
                            return v;
                        };
                        const auto v1 = numeric_sorted(*la), v2 = numeric_sorted(*lbv);
                        auto val = [&](i32 l) { return bd.label_values[static_cast<std::size_t>(l)]; };
                        double ssq_v2 = 0, sum_v2 = 0, sum_wt2 = 0;
                        std::vector<double> cum_sum, cum_wt2;
                        for (i32 l2 : v2) {
                            const double wt2 = (*sb)[static_cast<std::size_t>(l2)];
                            const double x2 = val(l2);
                            sum_v2 += x2 * wt2;
                            ssq_v2 += perl_pow(x2, 2) * wt2;
                            sum_wt2 += wt2;
                            cum_sum.push_back(sum_v2);
                            cum_wt2.push_back(sum_wt2);
                        }
                        const double v2min = val(v2.front()), v2max = val(v2.back());
                        const double v2mean = sum_v2 / sum_wt2;
                        double sum_abs = NaN, sum_sq = NaN, sum_wts = 0;
                        auto acc = [](double& a, double v) { a = std::isnan(a) ? v : a + v; };
                        i64 i = 0;
                        const i64 last = static_cast<i64>(v2.size()) - 1;
                        auto v2at = [&](i64 k) { return (k >= 0 && k <= last) ? val(v2[static_cast<std::size_t>(k)]) : 0.0; };
                        auto at = [&](const std::vector<double>& a, i64 k) {
                            return (k >= 0 && k < static_cast<i64>(a.size())) ? a[static_cast<std::size_t>(k)] : 0.0;
                        };
                        for (i32 l1 : v1) {
                            const double wt1 = (*sa)[static_cast<std::size_t>(l1)];
                            const double x1 = val(l1);
                            acc(sum_sq, wt1 * (ssq_v2 - 2 * x1 * sum_v2 + sum_wt2 * perl_pow(x1, 2)));
                            sum_wts += wt1 * sum_wt2;
                            if (x1 < v2min || x1 > v2max) {
                                acc(sum_abs, wt1 * sum_wt2 * std::fabs(x1 - v2mean));
                                continue;
                            }
                            const i64 j = std::min(i, last);
                            for (i64 it = j; it <= last; ++it) {
                                if (x1 < v2at(i)) {
                                    --i;
                                    break;
                                }
                                if (x1 == v2at(i)) break;
                                ++i;
                            }
                            const double cum_sum_i = at(cum_sum, i);
                            const double n_le = at(cum_wt2, i);
                            const double n_gt = sum_wt2 - n_le;
                            const double mean_lt = at(cum_sum, i) / (n_le != 0 ? n_le : 1);
                            const double mean_gt = (sum_v2 - cum_sum_i) / (n_gt != 0 ? n_gt : 1);
                            const double diff_lt = std::fabs(x1 - mean_lt) * n_le;
                            const double diff_gt = std::fabs(x1 - mean_gt) * n_gt;
                            acc(sum_abs, wt1 * (diff_lt + diff_gt));
                        }
                        set(0, sum_wts != 0 ? num(sum_abs) / sum_wts : NaN);
                        set(2, sum_wts != 0 ? num(sum_sq) / sum_wts : NaN);
                        set(1, sum_wts);
                    }
                } else if (name == "calc_matrix_stats") {
                    std::vector<i32> in;
                    for (i32 l : abc.lall)
                        if (bd.label_mx[static_cast<std::size_t>(l)] >= 0) in.push_back(l);
                    std::vector<double> values;
                    for (std::size_t a = 0; a < in.size(); ++a)
                        for (std::size_t b = a; b < in.size(); ++b) {
                            const double v = mx_value(in[a], in[b]);
                            if (!std::isnan(v)) values.push_back(v);
                        }
                    DStats s;
                    s.add(values);
                    s.sort_data();
                    for (double v : s.data()) put(1, v, 0);
                    for (i32 l : in) put(0, l, 1);
                    set(9, s.percentile(95));
                    set(6, s.percentile(5));
                    set(7, s.percentile(25));
                    set(8, s.percentile(75));
                    set(2, s.mean());
                    set(11, s.sd());
                    set(5, s.count());
                    set(3, s.median());
                    set(10, s.sample_range());
                    set(4, s.min());
                    set(1, s.max());
                    set(12, s.skewness());
                    set(0, s.kurtosis());
                } else if (name == "calc_compare_dissim_matrix_values") {
                    auto in_mx = [&](const std::vector<i32>& ls) {
                        std::vector<i32> out;
                        for (i32 l : ls)
                            if (bd.label_mx[static_cast<std::size_t>(l)] >= 0) out.push_back(l);
                        return out;
                    };
                    const auto a = in_mx(abc.l1), b = in_mx(abc.l2);
                    double sx = NaN, sxx = NaN, count = 0;
                    for (i32 l1 : a)
                        for (i32 l2 : b) {
                            double v = mx_value(l1, l2);
                            if (std::isnan(v) || l1 == l2) v = 0;  // self_similarity argument
                            sx = std::isnan(sx) ? v : sx + v;
                            sxx = std::isnan(sxx) ? perl_pow(v, 2) : sxx + perl_pow(v, 2);
                            count += 1;
                        }
                    set(1, count != 0 ? num(sx) / count : NaN);
                    set(2, count != 0 ? num(sxx) / count : NaN);
                    set(0, count);
                    for (i32 l1 : a) put(0, l1, 1);
                    if (!a.empty())
                        for (i32 l2 : b) put(1, l2, static_cast<double>(a.size()));
                } else if (name == "calc_mx_rao_qe") {
                    std::vector<i32> in;
                    for (i32 l : abc.lall)
                        if (bd.label_mx[static_cast<std::size_t>(l)] >= 0) in.push_back(l);
                    const double n = static_cast<double>(in.size());
                    double qe = NaN;
                    for (std::size_t a = 0; a < in.size(); ++a) {
                        const double p1 = 1 / n;
                        for (std::size_t b = a; b < in.size(); ++b) {
                            const double p2 = 1 / n;
                            double v = mx_defined(in[a], in[b]);
                            if (std::isnan(v)) v = 0;  // self_similarity argument
                            const double t = 2 * v * p1 * p2;
                            qe = std::isnan(qe) ? t : qe + t;
                        }
                    }
                    set(0, qe);
                    set(1, perl_pow(n, 2));
                    for (i32 l : in) put(0, l, 1 / n);
                } else if (name.rfind("calc_phylo_mpd_mntd", 0) == 0) {
                    const MpdResult& m = pc_mpd[name.back() - '0'];
                    for (int p = 0; p < 2; ++p) {
                        const MpdStats& st = m.s[p == 0 ? 1 : 0];
                        const std::size_t o = static_cast<std::size_t>(p) * 6;
                        set(o + 0, st.max);
                        set(o + 1, st.mean);
                        set(o + 2, st.min);
                        set(o + 3, st.n);
                        set(o + 4, st.rmsd);
                        set(o + 5, st.variance);
                    }
                } else if (name == "calc_nri_nti1" || name == "calc_nri_nti2" || name == "calc_nri_nti3") {
                    const MpdResult& m = pc_mpd[name.back() - '0'];
                    set(0, zscore(m.s[1].mean, pc_exp.nri_mean, pc_exp.nri_sd));
                    set(1, zscore(m.s[0].mean, pc_exp.nti_mean, pc_exp.nti_sd));
                } else if (name == "calc_nri_nti_expected_values") {
                    if (pc_exp_defined) {
                        set(0, pc_exp.n);
                        set(1, pc_exp.nri_mean);
                        set(2, pc_exp.nri_sd);
                        set(3, pc_exp.nti_mean);
                        set(4, pc_exp.nti_sd);
                    }
                } else if (name == "calc_net_vpd") {
                    set(0, zscore(pc_mpd[1].s[1].variance, pc_exp.vpd_mean, pc_exp.vpd_sd));
                } else if (name == "calc_vpd_expected_values") {
                    set(0, pc_exp.vpd_mean);
                    set(1, num(pc_exp.vpd_n));
                    set(2, pc_exp.vpd_sd);
                } else if (name == "calc_rwibald") {
                    PhyloNbhd& p = P();
                    p.pd_union();
                    double counts[4] = {0, 0, 0, 0};
                    for (i32 node : p.pd_nodes) {
                        const i32 code = pg.rwibald_code[static_cast<std::size_t>(node)];
                        put(0, node, code >= 0 ? code : NaN);
                        if (code >= 0) counts[code] += 1;
                        put(2, node, pg.rwibald_diff[static_cast<std::size_t>(node)]);
                        put(4, node, pg.rwibald_is_rr[static_cast<std::size_t>(node)] ? pg.rwibald_diff[static_cast<std::size_t>(node)] : NaN);
                    }
                    for (int c = 0; c < 4; ++c) put(1, c, counts[c]);
                    put(3, key_nt, pg.rwibald_neg);
                    put(3, key_pt, pg.rwibald_pos);
                    put(3, key_rt, pg.rwibald_range_thresh);
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
