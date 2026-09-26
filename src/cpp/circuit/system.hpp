// SPDX-License-Identifier: GPL-3.0-or-later
//
// The grounded Laplacian of a set of grid cells with a fixed sparsity pattern:
// the cells are numbered in a nested-dissection order and analysed once
// (Eigen's SimplicialLDLT: elimination tree and column counts); after that
// every new set of conductances only refills the values and runs the numeric
// factorization. Cells that are not part of a circuit get an identity row, so
// windows with NoData keep the pattern of the full window.
#pragma once

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <cstdint>
#include <vector>

#include "circuit/grid.hpp"

namespace circuit {

class CellSystem {
public:
    // The unknowns are `cells` (row-major indices of a bx × by box), in that
    // elimination order; neighbours as in kNeighbours8 / kNeighbours4.
    void analyze(i64 bx, i64 by, std::vector<i64> cells, bool four_neighbors);

    i64 size() const { return static_cast<i64>(cells_.size()); }
    i64 box_nx() const { return bx_; }
    i64 box_ny() const { return by_; }
    bool four_neighbors() const { return four_; }
    const std::vector<i64>& cells() const { return cells_; }
    // Box cell -> unknown (-1 for cells that are not unknowns).
    const std::vector<std::int32_t>& unknown_of() const { return unknown_of_; }

    // Circuitscape's Laplacian over the box cells with present[cell] != 0
    // (conductance `g`, row-major over the box): the diagonal of an unknown is
    // the sum, in column-major neighbour order, of its edges to present cells
    // (unknowns or not: a direct ground is present but not an unknown), plus
    // extra_diag[cell] when given, plus `shift`; unknowns that are not present
    // get an identity row.
    void assemble(const double* g, const std::uint8_t* present, const double* extra_diag, double shift);

    // False when the matrix is not positive definite.
    bool factorize();
    // x = A⁻¹ b over the unknowns (b and x may alias).
    void solve(const double* b, double* x);

private:
    using Matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    using Solver = Eigen::SimplicialLDLT<Matrix, Eigen::Upper, Eigen::NaturalOrdering<int>>;

    i64 bx_ = 0, by_ = 0;
    bool four_ = false;
    std::vector<i64> cells_;
    std::vector<std::int32_t> unknown_of_;
    Matrix a_;                    // upper triangle, column j holds rows i <= j
    std::vector<int> diag_slot_;  // unknown -> value index of its diagonal
    std::vector<int> edge_slot_;  // unknown × direction -> value index of the edge (-1: no unknown there)
    Solver solver_;
    Eigen::VectorXd work_;
};

}  // namespace circuit
