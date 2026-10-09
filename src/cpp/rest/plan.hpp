// SPDX-License-Identifier: GPL-3.0-or-later
//
// Pieces of the restoration planner's exact method at scale: the vertex of a set of plans with one
// row (a fractional knapsack, by weighted selection instead of a linear program) and the products of
// the species x units gain matrix (CSR) with a plan and with the species' slopes, in parallel.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rest {

using i64 = std::int64_t;

// max c x subject to a x (sense) cap and lb <= x <= ub, a >= 0: the units by c / a, best first (ties
// in the units' order). sense "<=": the best units up to cap; ">=": every unit of c > 0, then the
// least bad ones (c <= 0) until cap; "=": the best up to cap, then the least bad ones. Returns false
// when no plan meets the row.
bool knapsack(const double* c, const double* a, const double* lb, const double* ub, i64 n, double cap,
              const std::string& sense, double* x);

// out = G x (rows of G), G in CSR.
template <class P, class I>
void csr_matvec(const P* indptr, const I* indices, const double* data, i64 nrow, const double* x, double* out,
                int threads);

// out = G' y (columns of G, ncol of them), G in CSR: thread-local sums added in a fixed order.
template <class P, class I>
void csr_rmatvec(const P* indptr, const I* indices, const double* data, i64 nrow, i64 ncol, const double* y,
                 double* out, int threads);

// The rows of a CSR matrix with only the columns whose `pos` is >= 0, renumbered to pos (increasing
// with the column): out_indptr (nrow + 1), out_indices, out_data.
template <class P, class I>
void csr_select_columns(const P* indptr, const I* indices, const double* data, i64 nrow, const i64* pos,
                        std::vector<i64>& out_indptr, std::vector<std::int32_t>& out_indices,
                        std::vector<double>& out_data, int threads);

}  // namespace rest
