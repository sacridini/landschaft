// SPDX-License-Identifier: GPL-3.0-or-later
//
// Agglomerative clustering of groups (Biodiverse::Cluster): dissimilarity
// matrices indexed by value as Biodiverse::Matrix does (buckets of the
// values printed with %.2g, ties by the values Perl prints the same), the
// matrix build over spatial conditions, and the joining loop with its
// linkage functions and tie handling. The tree is finished in Python.

#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "bio/conditions.hpp"
#include "bio/phylocom.hpp"

namespace bio {

struct ClusterInput {
    std::vector<std::string> names;  // elements (groups passing the definition query), name order
    // per spatial condition: each element's neighbours (element indices)
    std::vector<std::vector<std::vector<i32>>> nbrs;
    bool trim_processed = false;  // one always_true condition: each pair once
    // index values: direct (Sorenson, Jaccard, S2 from label sets) or given per ordered pair
    std::string direct;                         // "SORENSON", "JACCARD", "S2" or ""
    std::vector<std::vector<i32>> labels;       // per element, sorted label ids (direct)
    std::vector<i64> pair_keys;                 // e1 * n + e2 (given values)
    std::vector<double> pair_values;
    std::string linkage = "link_average";
    bool can_lump_zeroes = false;
    double min_poss = 1e10;
    // tie breakers: keys ("random" only here) with "maximise"/"minimise"
    bool have_tie_breaker = false;
    std::vector<std::string> breaker_keys, breaker_minmax;
    bool objective_max = false;  // get_max_value (else get_min_value)
    double max_poss = std::numeric_limits<double>::quiet_NaN();
    // link_recalculate: the index of (node1 and node2) against each check node
    std::function<std::vector<double>(const std::string&, const std::string&, const std::string&,
                                      const std::vector<std::string>&)>
        recalculate;
};

struct ClusterJoin {
    i32 join_number;
    std::string name, child1, child2;
    double value;
    i32 matrix_iter;
};

struct ClusterOutput {
    std::vector<std::string> terminals;  // elements of the matrix used for the nodes
    std::vector<ClusterJoin> joins;
    i32 join_number = -1;
    double min_value = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::uint64_t> rng_state;
};

ClusterOutput cluster_groups(const ClusterInput& in, MT64& rng);

}  // namespace bio
