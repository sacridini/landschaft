// SPDX-License-Identifier: GPL-3.0-or-later
//
// Circuitscape 6.1's raster circuits (raster/graph.jl, core.jl, advanced.jl):
// the node map (cells with a conductance > 0, numbered in column-major order;
// the cells of a short-circuit region with habitat merged into one node, its
// other cells included), the node graph (edges to the right, lower and, with
// 8 neighbours, both right-hand diagonal cells, weighted by the average
// conductance or the average resistance of the two cells, summed between
// merged nodes), its Laplacian and its connected components (edges of weight
// 0 do not connect).
#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "circuit/grid.hpp"

namespace circuit {

struct RasterGraph {
    i64 nx = 0, ny = 0;
    std::int32_t n_nodes = 0;
    std::vector<std::int32_t> nodemap;   // cell -> node (-1: none)
    std::vector<i64> node_cell;          // node -> its cell when it is a single cell, else -1
    std::vector<std::int64_t> ptr;       // node -> first neighbour in adj / w
    std::vector<std::int32_t> adj;       // neighbours, ascending
    std::vector<double> w;               // edge conductances (> 0, or 0 with average resistances)
    std::vector<double> degree;          // Laplacian diagonal: the node's edges summed in adj order
    std::vector<std::int32_t> comp;      // node -> component
    std::int32_t n_comp = 0;
};

// `gmap`: conductance per cell (row-major ny × nx; > 0: habitat); `polymap`:
// short-circuit region ids (0: none) or nullptr.
RasterGraph raster_graph(const double* gmap, const std::int64_t* polymap, i64 nx, i64 ny, bool avg_resistances,
                         bool four_neighbors);

// Circuitscape's network graph (network/problem.jl): edges (a[k], b[k]) of
// conductance w[k] over the nodes 0..m-1. A = sparse(a, b, w) (duplicates
// summed in input order), A + Aᵀ (entries that add up to 0 are dropped, as
// sparse `+` does), its Laplacian (self loops left out) and its components.
// node_cell is -1 everywhere: the solver orders the graph by its own nested
// dissection.
RasterGraph network_graph(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b,
                          const std::vector<double>& w, std::int32_t m);

// Node currents (get_node_currents: max of inflow and outflow, finite grounds
// by node or nullptr) and branch currents (_get_branch_currents: every edge
// r < c of the nodes `members` (ascending), column by column, |w| (v_r - v_c)
// zeroed below 1e-8 of the largest, then made absolute) over `members`.
// `node_cur` is by node; `branch` gets (adjacency entry of r in c's list, value).
void network_currents(const RasterGraph& g, const std::int32_t* members, std::int64_t count, const double* v,
                      const double* finite, double* node_cur, std::vector<std::pair<std::int64_t, double>>& branch);

// Circuitscape's remove_src_or_gnd.
enum class ConflictPolicy { KeepAll, RemoveSource, RemoveGround, RemoveAll };

// Advanced mode (advanced_kernel with CHOLMOD) on a raster graph: source and
// ground (conductance to ground, +Inf: direct) per node. Every component with
// a source and a ground is solved; the others keep voltage and current 0.
struct NodeSolution {
    std::vector<double> voltage, current;  // per node
    bool solved = false;
    std::vector<std::int32_t> active;      // nodes with a source or a ground after the policy
};
// With only_node >= 0 only its component is solved (check_node of the
// one-to-all driver); only_node == -2 solves nothing (a focal point off the graph).
NodeSolution raster_advanced(const RasterGraph& g, std::vector<double> source, std::vector<double> ground,
                             ConflictPolicy policy, int max_threads, std::int32_t only_node = -1);

// Many advanced-mode problems on one graph (the one-to-all driver), each
// with its own sources, grounds and check node (-1: every component; -2:
// nothing). When the nodes that carry a source or a ground in any problem
// are few, every component is factorised once without them and each problem
// is solved through the dense Schur complement on them (exact; one more
// triangular solve per problem); otherwise every problem is factorised. The
// problems run in parallel.
std::vector<NodeSolution> raster_advanced_many(const RasterGraph& g, const std::vector<std::vector<double>>& sources,
                                               const std::vector<std::vector<double>>& grounds,
                                               const std::vector<std::int32_t>& only_nodes, ConflictPolicy policy,
                                               int max_threads);

// Per-node sums of a cell map through the node map (column-major, as
// Circuitscape's fill_sources_and_grounds!).
std::vector<double> node_sums(const RasterGraph& g, const double* cells);

// Pairwise mode (single_ground_all_pairs, CHOLMOD). `points`: the node of
// every focal point (-1: none), in the order of their ids; `excluded`: n × n,
// pairs of focal points not to solve. Every component is factorised once (one
// node grounded: the voltage differences are those of Circuitscape's shifted
// singular solve) and its pairs solved in batches. Without maps and exclusions
// Circuitscape's shortcut applies: only the pairs of the component's first
// focal node are solved, the others follow from their voltages. Resistances:
// -1 where not solved, 0 between focal points on one node and on the diagonal.
struct PairwiseOptions {
    bool volt_maps = false;      // write_volt_maps: every pair's voltage map
    bool cur_maps = false;       // every pair's current map (write_cur_maps without write_cum_cur_map_only)
    bool cum = false;            // write_cur_maps or write_cum_cur_map_only
    bool max = false;            // write_max_cur_maps
    bool focal_zero = false;     // set_focal_node_currents_to_zero
    bool null_currents = false;  // set_null_currents_to_nodata
    bool null_voltages = false;  // set_null_voltages_to_nodata
    bool log_transform = false;  // log_transform_maps
    // A network (no grid): no shortcut; with any map option every pair gives
    // its node voltages, node currents and branch currents (component nodes),
    // and the node and branch currents are summed over the pairs.
    bool network = false;
    bool maps() const { return volt_maps || cur_maps || cum || max; }
};
struct PairMaps {
    int ci = -1, cj = -1;                  // focal point indices (their ids name the maps)
    std::vector<double> voltage, current;  // grids, when asked for (networks: per node of `nodes`)
    std::vector<std::int32_t> nodes;       // networks: the component's nodes
    std::vector<std::pair<std::int64_t, double>> branch;  // networks: branch currents (see network_currents)
};
struct PairwiseResult {
    std::vector<double> resistances;  // n × n
    std::vector<double> cum, max;     // grids (empty unless asked for); networks: cum by node
    std::vector<double> cum_branch;   // networks: summed branch currents by adjacency entry
};
PairwiseResult raster_pairwise(const RasterGraph& g, const double* gmap, const std::vector<std::int32_t>& points,
                               const std::vector<std::uint8_t>& excluded, const PairwiseOptions& options,
                               const std::function<void(PairMaps&&)>& emit, int max_threads);

// Pairwise mode over focal regions (pairwise_regions): every pair of distinct
// ids (i < j, not excluded) gets its own graph, the cells of its two regions
// made short-circuit regions (create_new_polymap: merged with the polygons
// they overlap), and one solve between the nodes of their first cells.
// `cells` (row-major) and `ids` (sorted) list the focal cells; `excluded` is
// over the distinct ids, n × n. Pairs run in parallel, summed in order.
PairwiseResult raster_pairwise_regions(const double* gmap, const std::int64_t* polymap, i64 nx, i64 ny,
                                       bool avg_resistances, bool four_neighbors, const std::vector<i64>& cells,
                                       const std::vector<std::int64_t>& ids, const std::vector<std::uint8_t>& excluded,
                                       const PairwiseOptions& options, const std::function<void(PairMaps&&)>& emit,
                                       int max_threads);

// create_new_polymap without a point map, for the focal ids pt1 and pt2.
std::vector<std::int64_t> region_polymap(const std::int64_t* polymap, i64 n, const std::vector<i64>& cells,
                                         const std::vector<std::int64_t>& ids, std::int64_t pt1, std::int64_t pt2);

}  // namespace circuit
