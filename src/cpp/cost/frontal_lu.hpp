// SPDX-License-Identifier: GPL-3.0-or-later
//
// Multifrontal LU without pivoting for sparse matrices with a symmetric
// pattern and unsymmetric values, on the circuit module's symbolic analysis
// (nested dissection, fronts): the randomised-shortest-path systems I - W of
// asymmetric transitions, which are M-matrices (row diagonally dominant with
// a positive diagonal), so elimination without pivoting is stable. Every
// front keeps its first npiv columns of L (unit lower, with U11 above the
// diagonal) and its npiv rows of U12; the trailing Schur complement goes on
// a stack, as in the Cholesky.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "circuit/system.hpp"

namespace cost {

class FrontalLU {
public:
    // The structure (frontal_analyze_graph of the caller's symmetric pattern).
    explicit FrontalLU(std::shared_ptr<const circuit::FrontalSymbolic> sym);

    // Factorises the matrix with diagonal diag[node] and off-diagonal entries
    // val[k] (row node, column adj[k] of the caller's adjacency); rev[k] is the
    // entry of the reverse connection. False on a zero pivot.
    bool factorize(const double* diag, const double* val, const std::int64_t* rev);

    // A x = b, or A' x = b with `transposed`, in place, over the unknowns
    // (elimination order: unknown u is the caller's node perm()[u]).
    void solve(double* x, bool transposed) const;

    const std::vector<std::int32_t>& perm() const { return sym_->perm; }
    std::int64_t size() const { return static_cast<std::int64_t>(sym_->cells.size()); }

private:
    std::shared_ptr<const circuit::FrontalSymbolic> sym_;
    std::vector<std::int64_t> lpos_, upos_;  // per node: its L (m x npiv) and U12 (npiv x nupd) blocks
    std::vector<double> l_, u_;
};

}  // namespace cost
