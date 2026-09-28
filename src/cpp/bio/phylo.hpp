// SPDX-License-Identifier: GPL-3.0-or-later
//
// Phylogenetic parts of the index engine: the tree as arrays (nodes numbered
// in the sorted order of their names, so that iterating the indices follows
// the reference runs' hash order), the trimmed tree Biodiverse builds (the
// tree pruned to the basedata's labels), node ranges, per-group phylogenetic
// endemism and the global evolutionary distinctiveness scores.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "bio/conditions.hpp"

namespace bio {

struct BaseDataView;

struct TreeView {
    i32 n = 0;
    std::vector<i32> parent;        // -1 for the root
    std::vector<double> length;
    std::vector<i64> child_ptr;     // children in the tree's order (CSR)
    std::vector<i32> child_idx;
    std::vector<i32> depth;         // root 0
    std::vector<std::uint8_t> terminal;       // terminal in the full tree
    std::vector<std::uint8_t> in_trimmed;     // kept in the trimmed tree
    std::vector<i32> trimmed_child_count;
    double total_length = 0;          // full tree, summed in name order
    double trimmed_total_length = 0;  // trimmed tree
    i32 root = -1;
    std::vector<i32> label_node;      // per basedata label: node index or -1
    std::vector<i32> node_label;      // per node: basedata label index or -1
};

// Things computed once per analysis.
struct PhyloGlobals {
    std::vector<double> range;        // node range on the trimmed tree (NaN: not in it)
    std::vector<double> rw;           // length / range (0 when range 0; NaN: not in the trimmed tree)
    // per group: nodes on the paths of its labels (full tree), ascending, and the PE score
    std::vector<i64> gp_ptr;
    std::vector<i32> gp_nodes;
    std::vector<double> gp_score;     // NaN when the group has no label on the tree
    // evolutionary distinctiveness per label (NaN when not a trimmed-tree terminal)
    std::vector<double> es, ed, aed;
    std::vector<double> node_abundance;  // trimmed tree, sum of terminals' sample counts
    std::vector<i32> terminal_count;     // terminals below each trimmed-tree node
    bool have_pe = false, have_aed = false;
};

void build_phylo_globals(const BaseDataView& bd, const TreeView& tree, PhyloGlobals& g, bool pe, bool aed,
                         bool parallel);

// Scratch for unions of paths to the root.
struct PathUnion {
    std::vector<i32> stamp;
    i32 current = 0;
    std::vector<i32> nodes;  // the union, unsorted until sorted()
    void reset(i32 n) {
        if (static_cast<i32>(stamp.size()) != n) {
            stamp.assign(static_cast<std::size_t>(n), 0);
            current = 0;
        }
        if (++current == 0x7fffffff) {
            std::fill(stamp.begin(), stamp.end(), 0);
            current = 1;
        }
        nodes.clear();
    }
    bool has(i32 node) const { return stamp[static_cast<std::size_t>(node)] == current; }
    void add_path(const TreeView& t, i32 node) {
        while (node >= 0 && stamp[static_cast<std::size_t>(node)] != current) {
            stamp[static_cast<std::size_t>(node)] = current;
            nodes.push_back(node);
            node = t.parent[static_cast<std::size_t>(node)];
        }
    }
};

}  // namespace bio
