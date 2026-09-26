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

// Circuitscape's remove_src_or_gnd.
enum class ConflictPolicy { KeepAll, RemoveSource, RemoveGround, RemoveAll };

// Advanced mode (advanced_kernel with CHOLMOD) on a raster graph: source and
// ground (conductance to ground, +Inf: direct) per node. Every component with
// a source and a ground is solved; the others keep voltage and current 0.
struct NodeSolution {
    std::vector<double> voltage, current;  // per node
    bool solved = false;
};
NodeSolution raster_advanced(const RasterGraph& g, std::vector<double> source, std::vector<double> ground,
                             ConflictPolicy policy, int max_threads);

// Per-node sums of a cell map through the node map (column-major, as
// Circuitscape's fill_sources_and_grounds!).
std::vector<double> node_sums(const RasterGraph& g, const double* cells);

}  // namespace circuit
