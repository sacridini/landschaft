// SPDX-License-Identifier: GPL-3.0-or-later
//
// Raster circuits as Circuitscape 6.1 builds them (raster/graph.jl): every
// cell with a conductance > 0 is a node joined to its 8 neighbours (4 with
// connect_four_neighbors_only) by the average conductance of the two cells,
// (x + y) / 2, or (x + y) / (2√2) on the diagonals. Circuitscape numbers the
// nodes in column-major order (Julia arrays); that order decides the sums of
// the Laplacian diagonal and which branch currents fall under its 1e-8
// threshold, so every loop over neighbours here follows it.
#pragma once

#include <cstdint>
#include <vector>

namespace circuit {

using i64 = std::int64_t;

// 2√2 in double, exactly as Julia evaluates `2 * √2`.
inline constexpr double kTwoSqrt2 = 2.8284271247461903;

struct Offset {
    int dy, dx;
    bool diagonal;
};

// Neighbours in increasing column-major index: the left column top to bottom,
// the cell above, the cell below, the right column top to bottom.
inline constexpr Offset kNeighbours8[8] = {{-1, -1, true}, {0, -1, false}, {1, -1, true}, {-1, 0, false},
                                           {1, 0, false},  {-1, 1, true},  {0, 1, false}, {1, 1, true}};
inline constexpr Offset kNeighbours4[4] = {{0, -1, false}, {-1, 0, false}, {1, 0, false}, {0, 1, false}};
// Of the neighbours above, the first 4 (8-neighbour) / 2 (4-neighbour) have a
// smaller column-major index than the cell itself.
inline constexpr int kLower8 = 4;
inline constexpr int kLower4 = 2;

// Circuitscape's edge weight between two cells.
inline double edge_weight(double a, double b, bool diagonal) {
    return diagonal ? (a + b) / kTwoSqrt2 : (a + b) / 2;
}

// Nested-dissection elimination order of the marked cells (mark != 0) of a
// row-major grid with `stride` columns inside rows [r0, r1) and columns
// [c0, c1): the box is shrunk to its marked cells and cut by its middle row
// or column (a single line separates both 4- and 8-neighbour grids), the two
// halves are ordered first and the separator last. Small boxes keep row order.
void nested_dissection(const std::uint8_t* mark, i64 stride, i64 r0, i64 r1, i64 c0, i64 c1,
                       std::vector<i64>& order);

// Circuitscape's advanced mode on a raster (compute_omniscape_current):
// `conductance`, `source` (amps) and `ground` (conductance to ground, +Inf =
// direct ground) are row-major ny × nx. Where a cell is both a source and a
// ground its source is dropped; each connected component with a source and a
// ground is solved (finite grounds on the diagonal, direct grounds removed at
// 0 V). `current` is the node current max(inflow, outflow) of every cell (0
// where not solved), `voltage` the node voltages (0 where not solved, NaN
// where there is no node). Components are solved in parallel.
struct AdvancedResult {
    std::vector<double> current, voltage;
};
AdvancedResult advanced(const double* conductance, const double* source, const double* ground, i64 nx, i64 ny,
                        bool four_neighbors, int max_threads);

}  // namespace circuit
