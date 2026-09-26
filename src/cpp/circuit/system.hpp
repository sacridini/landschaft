// SPDX-License-Identifier: GPL-3.0-or-later
//
// The grounded Laplacian of a set of grid cells, factorised by a multifrontal
// Cholesky (LLᵀ) over a geometric nested dissection of the grid: the box of
// unknowns is cut by its middle row or column, the halves recursively, and
// every separator (or small leaf box) is one front, a dense matrix whose
// pivot block is factorised with dense kernels and whose Schur complement is
// passed to the parent front. The symbolic part (tree, front structures,
// assembly maps) depends only on the shape of the set of cells and is shared
// read-only between threads; a factorisation only refills the values. Cells
// that are not part of a circuit get an identity row, so windows with NoData
// keep the structure of the full window.
#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <new>
#include <vector>

#include "circuit/grid.hpp"

namespace circuit {

struct FrontalSymbolic {
    struct Node {
        std::int32_t first = 0, npiv = 0;  // pivots: unknowns [first, first + npiv)
        std::int32_t nupd = 0;             // update rows (unknowns after the pivots)
        std::int32_t upd = 0;              // offset of the update rows in `rows` / `relmap`
        std::int32_t child_begin = 0, child_end = 0;  // children, in `children`
        std::int64_t lpos = 0;             // offset of the (npiv + nupd) × npiv block of L
    };
    i64 bx = 0, by = 0;
    bool four = false;
    std::vector<i64> cells;                // unknown -> box cell, in elimination order (-1: a merged node)
    std::vector<std::int32_t> unknown_of;  // box cell -> unknown (-1: none)
    // The matrix entries: grids (frontal_analyze) have unknown k's at
    // [k nd, (k + 1) nd), one per neighbour direction; general graphs
    // (frontal_analyze_graph) the caller's adjacency of node perm[k] at
    // [adj_ptr[perm[k]], adj_ptr[perm[k] + 1]), and their values (diagonal,
    // entries) are indexed by the caller's nodes.
    std::vector<std::int32_t> perm;        // unknown -> caller's node (empty: the identity)
    std::vector<std::int64_t> adj_ptr;     // caller's node -> first entry (empty: grid)
    std::vector<std::int32_t> nbr;         // entry -> neighbouring unknown (-1: none)
    std::vector<std::int32_t> amap;        // entry -> front row of that edge (-1: not stored)
    std::int64_t entry_begin(std::int32_t k) const {
        return adj_ptr.empty() ? static_cast<std::int64_t>(k) * (four ? 4 : 8) : adj_ptr[perm[k]];
    }
    std::int64_t entry_end(std::int32_t k) const {
        return adj_ptr.empty() ? static_cast<std::int64_t>(k + 1) * (four ? 4 : 8) : adj_ptr[perm[k] + 1];
    }
    std::int32_t value_index(std::int32_t k) const { return perm.empty() ? k : perm[k]; }
    std::vector<Node> nodes;               // post-order
    std::vector<std::int32_t> children;
    std::vector<std::int32_t> rows;        // update rows of every node (unknowns, ascending)
    std::vector<std::int32_t> relmap;      // update row of a node -> row in its parent's front
    std::int64_t lsize = 0;                // doubles of L
    std::int64_t stack_size = 0;           // doubles of the update-matrix stack
    std::int64_t max_front = 0;            // largest front, m² (room past the last L block)
};

// The structure for the box cells with unknown[cell] != 0 (row-major bx × by).
std::shared_ptr<const FrontalSymbolic> frontal_analyze(i64 bx, i64 by, const std::uint8_t* unknown,
                                                       bool four_neighbors);

// The structure of a general graph of n nodes (adjacency in CSR: node c's
// neighbours adj[adj_ptr[c] .. adj_ptr[c + 1]), -1 ignored; symmetric). Nodes
// that are one cell (cell[c] >= 0, row-major in a bx × by box) are ordered by
// the grid's nested dissection; merged nodes (cell[c] < 0, e.g. Circuitscape's
// short-circuit regions) are eliminated last, in one front at the root. Cells
// that are neighbours on the grid must be the only edges between cell nodes.
std::shared_ptr<const FrontalSymbolic> frontal_analyze_graph(i64 bx, i64 by, const std::vector<i64>& cell,
                                                             const std::vector<std::int64_t>& adj_ptr,
                                                             const std::vector<std::int32_t>& adj);

// Scratch that keeps its capacity and is not zeroed when resized, aligned to
// 64 bytes: Eigen's vector kernels split their loops by the alignment of the
// data, so a fixed alignment keeps the results the same from run to run.
struct AlignedBuffer {
    struct Free {
        void operator()(double* p) const { ::operator delete[](p, std::align_val_t{64}); }
    };
    std::unique_ptr<double[], Free> p;
    std::size_t capacity = 0;
    double* get(std::size_t n) {
        if (n > capacity) {
            p.reset(static_cast<double*>(::operator new[](n * sizeof(double), std::align_val_t{64})));
            capacity = n;
        }
        return p.get();
    }
};

// Workspace of the triangular solves, one per thread when several threads
// solve with one factorization.
struct SolveScratch {
    AlignedBuffer rhs, tmp;
};

class CellSystem {
public:
    void analyze(i64 bx, i64 by, const std::uint8_t* unknown, bool four_neighbors);
    // Use a structure built by frontal_analyze (shared between systems).
    void share(std::shared_ptr<const FrontalSymbolic> symbolic);

    i64 size() const { return static_cast<i64>(sym_->cells.size()); }
    const std::vector<i64>& cells() const { return sym_->cells; }
    const std::vector<std::int32_t>& unknown_of() const { return sym_->unknown_of; }

    // Circuitscape's Laplacian over the box cells with present[cell] != 0
    // (conductance `g`, row-major over the box): the diagonal of an unknown is
    // the sum, in column-major neighbour order, of its edges to present cells
    // (unknowns or not: a direct ground is present but not an unknown), plus
    // extra_diag[cell] when given, plus `shift`; unknowns that are not present
    // get an identity row.
    void assemble(const double* g, const std::uint8_t* present, const double* extra_diag, double shift);

    // The matrix of a general graph: diag[c] and the entries off[k] (in the
    // caller's adjacency) of frontal_analyze_graph.
    void set_values(const double* diag, const double* off);
    // Unknown -> caller's node (general graphs).
    const std::vector<std::int32_t>& perm() const { return sym_->perm; }

    // False when the matrix is not positive definite.
    bool factorize();
    // factorize() and x = A⁻¹ b in one go: the forward substitution runs
    // front by front while each front is still in cache.
    bool factorize_solve(const double* b, double* x);
    // x = A⁻¹ b over the unknowns (b and x may alias).
    void solve(const double* b, double* x);
    // X = A⁻¹ X for k right-hand sides (column-major, size() × k).
    void solve_many(double* x, i64 k);
    // The same with the caller's workspace: safe from several threads at once.
    void solve_many(double* x, i64 k, SolveScratch& scratch) const;

private:
    std::shared_ptr<const FrontalSymbolic> sym_;
    bool factorize(double* y);  // with y: the forward substitution of y too
    void forward(double* x, i64 k, AlignedBuffer& tmp) const;
    void backward(double* x, i64 k, AlignedBuffer& tmp) const;
    std::vector<double> diag_, off_;  // the matrix: diagonal, and unknown × direction edge values
    AlignedBuffer l_, stack_;
    SolveScratch scratch_;
};

}  // namespace circuit
