// SPDX-License-Identifier: GPL-3.0-or-later
//
// Global phylogenetic quantities (see phylo.hpp), following
// Biodiverse::Indices::Phylogenetic (get_node_range_hash,
// get_inverse_range_weighted_path_lengths, _calc_pe's per-group cache,
// get_global_node_abundance_hash, get_aed_scores).

#include "bio/phylo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "bio/indices.hpp"
#include "lsm/parallel.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace bio {

namespace {
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
}

void build_phylo_globals(const BaseDataView& bd, const TreeView& t, PhyloGlobals& g, bool pe, bool aed, bool parallel) {
    const i32 n = t.n;
    const i64 ng = bd.n_groups;
    auto trimmed_terminal = [&](i32 node) {
        return t.in_trimmed[static_cast<std::size_t>(node)] && t.trimmed_child_count[static_cast<std::size_t>(node)] == 0;
    };

    // node ranges: groups whose trimmed-tree terminal labels lie below each node
    std::vector<i64> counts(static_cast<std::size_t>(n), 0);
#pragma omp parallel if (parallel && ng > 256)
    {
        std::vector<i64> local(static_cast<std::size_t>(n), 0);
        PathUnion pu;
#pragma omp for schedule(dynamic, 64)
        for (i64 gi = 0; gi < ng; ++gi) {
            pu.reset(n);
            for (i64 k = bd.indptr[gi]; k < bd.indptr[gi + 1]; ++k) {
                const i32 node = t.label_node[static_cast<std::size_t>(bd.indices[static_cast<std::size_t>(k)])];
                if (node >= 0 && trimmed_terminal(node)) pu.add_path(t, node);
            }
            for (i32 node : pu.nodes) ++local[static_cast<std::size_t>(node)];
        }
#pragma omp critical(bio_phylo_ranges)
        for (i32 i = 0; i < n; ++i) counts[static_cast<std::size_t>(i)] += local[static_cast<std::size_t>(i)];
    }
    g.range.assign(static_cast<std::size_t>(n), NaN);
    g.rw.assign(static_cast<std::size_t>(n), NaN);
    for (i32 i = 0; i < n; ++i) {
        if (!t.in_trimmed[static_cast<std::size_t>(i)]) continue;
        const double r = static_cast<double>(counts[static_cast<std::size_t>(i)]);
        g.range[static_cast<std::size_t>(i)] = r;
        g.rw[static_cast<std::size_t>(i)] = r != 0 ? t.length[static_cast<std::size_t>(i)] / r : 0.0;
    }

    if (pe) {
        // each group's nodes (paths of its labels on the full tree) and PE score
        std::vector<std::vector<i32>> rows(static_cast<std::size_t>(ng));
        g.gp_score.assign(static_cast<std::size_t>(ng), NaN);
#pragma omp parallel if (parallel && ng > 256)
        {
            PathUnion pu;
#pragma omp for schedule(dynamic, 64)
            for (i64 gi = 0; gi < ng; ++gi) {
                pu.reset(n);
                for (i64 k = bd.indptr[gi]; k < bd.indptr[gi + 1]; ++k) {
                    const i32 node = t.label_node[static_cast<std::size_t>(bd.indices[static_cast<std::size_t>(k)])];
                    if (node >= 0) pu.add_path(t, node);
                }
                auto& row = rows[static_cast<std::size_t>(gi)];
                row = pu.nodes;
                std::sort(row.begin(), row.end());
                if (!row.empty()) {
                    // List::Util::sum over values, an undefined weight counting 0
                    double s = 0;
                    bool first = true;
                    for (i32 node : row) {
                        const double w = g.rw[static_cast<std::size_t>(node)];
                        const double v = std::isnan(w) ? 0.0 : w;
                        if (first) {
                            s = v;
                            first = false;
                        } else {
                            s += v;
                        }
                    }
                    g.gp_score[static_cast<std::size_t>(gi)] = s;
                }
            }
        }
        g.gp_ptr.assign(static_cast<std::size_t>(ng) + 1, 0);
        for (i64 gi = 0; gi < ng; ++gi)
            g.gp_ptr[static_cast<std::size_t>(gi) + 1] = g.gp_ptr[static_cast<std::size_t>(gi)] + static_cast<i64>(rows[static_cast<std::size_t>(gi)].size());
        g.gp_nodes.resize(static_cast<std::size_t>(g.gp_ptr.back()));
        for (i64 gi = 0; gi < ng; ++gi)
            std::copy(rows[static_cast<std::size_t>(gi)].begin(), rows[static_cast<std::size_t>(gi)].end(),
                      g.gp_nodes.begin() + g.gp_ptr[static_cast<std::size_t>(gi)]);
        g.have_pe = true;
    }

    if (aed) {
        // terminals below each trimmed-tree node, in name order
        std::vector<std::vector<i32>> below(static_cast<std::size_t>(n));
        std::vector<i32> order(static_cast<std::size_t>(n));
        for (i32 i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
        std::sort(order.begin(), order.end(), [&](i32 a, i32 b) {
            return t.depth[static_cast<std::size_t>(a)] > t.depth[static_cast<std::size_t>(b)];
        });
        for (i32 node : order) {
            if (!t.in_trimmed[static_cast<std::size_t>(node)]) continue;
            auto& lst = below[static_cast<std::size_t>(node)];
            if (trimmed_terminal(node)) lst.push_back(node);
            for (i64 k = t.child_ptr[static_cast<std::size_t>(node)]; k < t.child_ptr[static_cast<std::size_t>(node) + 1]; ++k) {
                const i32 c = t.child_idx[static_cast<std::size_t>(k)];
                if (!t.in_trimmed[static_cast<std::size_t>(c)]) continue;
                const auto& cl = below[static_cast<std::size_t>(c)];
                lst.insert(lst.end(), cl.begin(), cl.end());
            }
            std::sort(lst.begin(), lst.end());
        }
        g.terminal_count.assign(static_cast<std::size_t>(n), 0);
        g.node_abundance.assign(static_cast<std::size_t>(n), NaN);
        for (i32 node = 0; node < n; ++node) {
            if (!t.in_trimmed[static_cast<std::size_t>(node)]) continue;
            const auto& lst = below[static_cast<std::size_t>(node)];
            g.terminal_count[static_cast<std::size_t>(node)] = static_cast<i32>(lst.size());
            double a = 0;
            for (i32 term : lst) {
                const i32 lab = t.node_label[static_cast<std::size_t>(term)];
                a += lab >= 0 ? bd.label_sample_count[static_cast<std::size_t>(lab)] : 0.0;
            }
            g.node_abundance[static_cast<std::size_t>(node)] = a;
        }
        const auto nl = static_cast<std::size_t>(bd.n_labels);
        g.es.assign(nl, NaN);
        g.ed.assign(nl, NaN);
        g.aed.assign(nl, NaN);
        for (std::size_t lab = 0; lab < nl; ++lab) {
            const i32 node0 = t.label_node[lab];
            if (node0 < 0 || !trimmed_terminal(node0)) continue;
            const double len = t.length[static_cast<std::size_t>(node0)];
            double es_sum = len, ed_sum = len;
            const double ab0 = g.node_abundance[static_cast<std::size_t>(node0)];
            double aed_sum = ab0 != 0 ? len / ab0 : NaN;  // eval { ... }
            double es_wt = 1;
            i32 node = t.parent[static_cast<std::size_t>(node0)];
            while (node >= 0) {
                const double nl_len = t.length[static_cast<std::size_t>(node)];
                es_wt /= static_cast<double>(t.trimmed_child_count[static_cast<std::size_t>(node)]);
                const double ed_wt = 1 / static_cast<double>(g.terminal_count[static_cast<std::size_t>(node)]);
                const double aed_wt = 1 / g.node_abundance[static_cast<std::size_t>(node)];
                es_sum += nl_len * es_wt;
                ed_sum += nl_len * ed_wt;
                aed_sum = (std::isnan(aed_sum) ? 0.0 : aed_sum) + nl_len * aed_wt;
                node = t.parent[static_cast<std::size_t>(node)];
            }
            g.es[lab] = es_sum;
            g.ed[lab] = ed_sum;
            g.aed[lab] = aed_sum;
        }
        g.have_aed = true;
    }
}

}  // namespace bio
