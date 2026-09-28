// SPDX-License-Identifier: GPL-3.0-or-later
//
// Biodiverse's indices over neighbourhoods of groups.
//
// For each processing group the labels of its neighbour sets are aggregated
// the way Biodiverse's _calc_abc does it (presence, groups per label, sample
// counts per label summed group by group in list order), then the requested
// calculations run on these aggregates, each a line-by-line port of the Perl
// with its arithmetic (x87 libm, Perl's `**`) and its hash iteration in
// sorted key order (the reference runs' order; labels and groups are
// numbered in that order). Undefined results are NaN.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bio/conditions.hpp"

namespace bio {

// How a list index keys its entries.
enum class KeyKind : i32 {
    Label = 0,   // label index
    Group = 1,   // group index
    Text = 2,    // index into the engine's text table
    Number = 3,  // a number (Perl stringifies it)
    GroupArray = 4,  // an array of groups (keys in order, values unused)
};

struct CalcInfo {
    std::string name;
    std::vector<std::string> scalars;
    std::vector<std::pair<std::string, KeyKind>> lists;
    i32 uses_nbr_lists = 1;
    bool uses_processing_element = false;
};

// Every calculation the engine knows.
const std::vector<CalcInfo>& calculation_registry();

struct BaseDataView {
    i64 n_groups = 0, n_labels = 0;
    std::vector<i64> indptr;       // groups x labels CSR, labels ascending
    std::vector<i32> indices;
    std::vector<double> counts;
    std::vector<double> label_range;      // Biodiverse get_range (property or variety)
    std::vector<double> label_abundance;  // get_label_abundance
    // label hierarchy from the label axes (BaseData::to_tree): for each label
    // and level (0 = first axis) the index of its prefix in prefix_names
    i32 n_label_levels = 1;
    std::vector<i32> label_prefix;
    std::vector<std::string> prefix_names;
};

struct ListOut {
    // rows are processing groups; entries (key, value)
    std::vector<i64> indptr;
    std::vector<double> keys;   // label/group/text indices or numbers
    std::vector<double> values;
};

struct IndexResult {
    std::vector<std::string> scalar_names;
    std::vector<double> scalars;          // n_procs x n_scalars, row major
    std::vector<std::string> list_names;
    std::vector<KeyKind> list_kinds;
    std::vector<ListOut> lists;
    std::vector<std::string> texts;       // text keys (Text kind)
};

// Run `calcs` (names from the registry) for every processing group.
// set1/set2 rows follow `procs`; set2 may be empty (one neighbour set).
IndexResult run_indices(const BaseDataView& bd, const std::vector<i32>& procs, const Csr& set1, const Csr* set2,
                        const std::vector<std::string>& calcs, bool parallel);

}  // namespace bio
