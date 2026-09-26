// SPDX-License-Identifier: GPL-3.0-or-later

#include "circuit/system.hpp"

#include <algorithm>
#include <utility>

namespace circuit {

void CellSystem::analyze(i64 bx, i64 by, std::vector<i64> cells, bool four_neighbors) {
    bx_ = bx;
    by_ = by;
    four_ = four_neighbors;
    cells_ = std::move(cells);
    const i64 n = size();
    unknown_of_.assign(static_cast<std::size_t>(bx * by), -1);
    for (i64 k = 0; k < n; ++k) unknown_of_[static_cast<std::size_t>(cells_[k])] = static_cast<std::int32_t>(k);

    const Offset* nb = four_ ? kNeighbours4 : kNeighbours8;
    const int nd = four_ ? 4 : 8;
    // Column j of the upper triangle: the neighbour unknowns i < j, then j.
    std::vector<int> colptr(static_cast<std::size_t>(n) + 1, 0);
    std::vector<std::int32_t> nbr(static_cast<std::size_t>(n * nd), -1);
    for (i64 j = 0; j < n; ++j) {
        const i64 y = cells_[j] / bx, x = cells_[j] % bx;
        int count = 1;
        for (int d = 0; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            if (yy < 0 || yy >= by || xx < 0 || xx >= bx) continue;
            const std::int32_t i = unknown_of_[static_cast<std::size_t>(yy * bx + xx)];
            nbr[static_cast<std::size_t>(j * nd + d)] = i;
            if (i >= 0 && i < j) ++count;
        }
        colptr[static_cast<std::size_t>(j) + 1] = colptr[static_cast<std::size_t>(j)] + count;
    }
    const int nnz = colptr[static_cast<std::size_t>(n)];
    std::vector<int> rows(static_cast<std::size_t>(nnz));
    diag_slot_.assign(static_cast<std::size_t>(n), -1);
    edge_slot_.assign(static_cast<std::size_t>(n * nd), -1);
    std::vector<std::pair<int, int>> col;  // (row, direction or -1 for the diagonal)
    for (i64 j = 0; j < n; ++j) {
        col.clear();
        for (int d = 0; d < nd; ++d) {
            const std::int32_t i = nbr[static_cast<std::size_t>(j * nd + d)];
            if (i >= 0 && i < j) col.emplace_back(i, d);
        }
        col.emplace_back(static_cast<int>(j), -1);
        std::sort(col.begin(), col.end());
        int p = colptr[static_cast<std::size_t>(j)];
        for (const auto& [i, d] : col) {
            rows[static_cast<std::size_t>(p)] = i;
            if (d < 0) {
                diag_slot_[static_cast<std::size_t>(j)] = p;
            } else {
                // The same slot seen from both ends of the edge.
                edge_slot_[static_cast<std::size_t>(j * nd + d)] = p;
                const i64 yi = cells_[static_cast<std::size_t>(i)] / bx, xi = cells_[static_cast<std::size_t>(i)] % bx;
                const i64 yj = cells_[j] / bx, xj = cells_[j] % bx;
                for (int e = 0; e < nd; ++e)
                    if (yi + nb[e].dy == yj && xi + nb[e].dx == xj)
                        edge_slot_[static_cast<std::size_t>(i) * nd + e] = p;
            }
            ++p;
        }
    }
    a_ = Matrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    a_.resizeNonZeros(nnz);
    std::copy(colptr.begin(), colptr.end(), a_.outerIndexPtr());
    std::copy(rows.begin(), rows.end(), a_.innerIndexPtr());
    std::fill(a_.valuePtr(), a_.valuePtr() + nnz, 0.0);
    for (i64 j = 0; j < n; ++j) a_.valuePtr()[diag_slot_[static_cast<std::size_t>(j)]] = 1.0;
    solver_.analyzePattern(a_);
    work_.resize(static_cast<Eigen::Index>(n));
}

void CellSystem::assemble(const double* g, const std::uint8_t* present, const double* extra_diag, double shift) {
    const Offset* nb = four_ ? kNeighbours4 : kNeighbours8;
    const int nd = four_ ? 4 : 8;
    double* v = a_.valuePtr();
    const i64 n = size();
    for (i64 j = 0; j < n; ++j) {
        const i64 c = cells_[j];
        const std::size_t base = static_cast<std::size_t>(j * nd);
        if (!present[c]) {
            for (int d = 0; d < nd; ++d) {
                const int s = edge_slot_[base + d];
                if (s >= 0) v[s] = 0.0;
            }
            v[diag_slot_[static_cast<std::size_t>(j)]] = 1.0;
            continue;
        }
        const i64 y = c / bx_, x = c % bx_;
        const double gc = g[c];
        double diag = 0.0;
        for (int d = 0; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            const int s = edge_slot_[base + d];
            if (yy < 0 || yy >= by_ || xx < 0 || xx >= bx_ || !present[yy * bx_ + xx]) {
                if (s >= 0) v[s] = 0.0;
                continue;
            }
            const double w = edge_weight(gc, g[yy * bx_ + xx], nb[d].diagonal);
            diag += w;
            if (s >= 0) v[s] = -w;
        }
        if (extra_diag) diag += extra_diag[c];
        v[diag_slot_[static_cast<std::size_t>(j)]] = diag + shift;
    }
}

bool CellSystem::factorize() {
    solver_.factorize(a_);
    return solver_.info() == Eigen::Success;
}

void CellSystem::solve(const double* b, double* x) {
    const Eigen::Index n = static_cast<Eigen::Index>(size());
    work_ = Eigen::Map<const Eigen::VectorXd>(b, n);
    Eigen::Map<Eigen::VectorXd>(x, n) = solver_.solve(work_);
}

}  // namespace circuit
