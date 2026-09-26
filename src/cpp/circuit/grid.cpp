// SPDX-License-Identifier: GPL-3.0-or-later

#include "circuit/grid.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "circuit/system.hpp"

namespace circuit {

namespace {

constexpr i64 kLeafCells = 16;

}  // namespace

void nested_dissection(const std::uint8_t* mark, i64 stride, i64 r0, i64 r1, i64 c0, i64 c1,
                       std::vector<i64>& order) {
    // Shrink the box to its marked cells.
    i64 top = r1, bottom = r0 - 1, left = c1, right = c0 - 1;
    for (i64 r = r0; r < r1; ++r)
        for (i64 c = c0; c < c1; ++c)
            if (mark[r * stride + c]) {
                if (r < top) top = r;
                if (r > bottom) bottom = r;
                if (c < left) left = c;
                if (c > right) right = c;
            }
    if (bottom < top) return;
    r0 = top;
    r1 = bottom + 1;
    c0 = left;
    c1 = right + 1;
    const i64 h = r1 - r0, w = c1 - c0;
    if (h * w <= kLeafCells) {
        for (i64 r = r0; r < r1; ++r)
            for (i64 c = c0; c < c1; ++c)
                if (mark[r * stride + c]) order.push_back(r * stride + c);
        return;
    }
    if (h >= w) {
        const i64 m = r0 + h / 2;
        nested_dissection(mark, stride, r0, m, c0, c1, order);
        nested_dissection(mark, stride, m + 1, r1, c0, c1, order);
        for (i64 c = c0; c < c1; ++c)
            if (mark[m * stride + c]) order.push_back(m * stride + c);
    } else {
        const i64 m = c0 + w / 2;
        nested_dissection(mark, stride, r0, r1, c0, m, order);
        nested_dissection(mark, stride, r0, r1, m + 1, c1, order);
        for (i64 r = r0; r < r1; ++r)
            if (mark[r * stride + m]) order.push_back(r * stride + m);
    }
}

AdvancedResult advanced(const double* conductance, const double* source, const double* ground, i64 nx, i64 ny,
                        bool four_neighbors, int max_threads) {
    const i64 n = nx * ny;
    const Offset* nb = four_neighbors ? kNeighbours4 : kNeighbours8;
    const int nd = four_neighbors ? 4 : 8;
    const int lower = four_neighbors ? kLower4 : kLower8;
    const double inf = std::numeric_limits<double>::infinity();
    AdvancedResult out;
    out.current.assign(static_cast<std::size_t>(n), 0.0);
    out.voltage.assign(static_cast<std::size_t>(n), std::numeric_limits<double>::quiet_NaN());

    // Components of the node cells (conductance > 0), labelled by a flood fill.
    std::vector<std::int32_t> comp(static_cast<std::size_t>(n), -1);
    std::vector<i64> stack;
    std::int32_t n_comp = 0;
    for (i64 c = 0; c < n; ++c) {
        if (!(conductance[c] > 0) || comp[c] >= 0) continue;
        comp[c] = n_comp;
        stack.push_back(c);
        while (!stack.empty()) {
            const i64 p = stack.back();
            stack.pop_back();
            const i64 y = p / nx, x = p % nx;
            for (int d = 0; d < nd; ++d) {
                const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
                if (yy < 0 || yy >= ny || xx < 0 || xx >= nx) continue;
                const i64 q = yy * nx + xx;
                if (conductance[q] > 0 && comp[q] < 0) {
                    comp[q] = n_comp;
                    stack.push_back(q);
                }
            }
        }
        ++n_comp;
    }
    for (i64 c = 0; c < n; ++c)
        if (comp[c] >= 0) out.voltage[c] = 0.0;

    // Sources and grounds on the nodes (resolve_conflicts with rmvsrc: a
    // source on a ground is dropped). Finite grounds go on the diagonal,
    // unless no node has one.
    std::vector<double> src(static_cast<std::size_t>(n), 0.0), gnd(static_cast<std::size_t>(n), 0.0),
        finite(static_cast<std::size_t>(n), 0.0);
    bool any_finite = false;
    for (i64 c = 0; c < n; ++c) {
        if (comp[c] < 0) continue;
        src[c] = source ? source[c] : 0.0;
        gnd[c] = ground ? ground[c] : 0.0;
        if (gnd[c] < inf) finite[c] = gnd[c];
        if (finite[c] != 0) any_finite = true;
        if (src[c] != 0 && gnd[c] != 0) src[c] = 0.0;
    }

    // Cells of every component, in column-major order (Circuitscape's node order).
    std::vector<i64> start(static_cast<std::size_t>(n_comp) + 1, 0);
    for (i64 c = 0; c < n; ++c)
        if (comp[c] >= 0) ++start[static_cast<std::size_t>(comp[c]) + 1];
    for (std::int32_t k = 0; k < n_comp; ++k) start[k + 1] += start[k];
    std::vector<i64> members(static_cast<std::size_t>(start[n_comp]));
    {
        std::vector<i64> fill(start.begin(), start.end() - 1);
        for (i64 x = 0; x < nx; ++x)
            for (i64 y = 0; y < ny; ++y) {
                const i64 c = y * nx + x;
                if (comp[c] >= 0) members[static_cast<std::size_t>(fill[comp[c]]++)] = c;
            }
    }
    std::vector<std::int32_t> solvable;
    for (std::int32_t k = 0; k < n_comp; ++k) {
        double s = 0.0, g = 0.0;
        for (i64 p = start[k]; p < start[k + 1]; ++p) {
            s += src[members[p]];
            g += gnd[members[p]];
        }
        if (s != 0 && g != 0) solvable.push_back(k);
    }

    const int n_solvable = static_cast<int>(solvable.size());
    bool failed = false;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(max_threads > 0 ? max_threads : omp_get_max_threads()) if (n_solvable > 1)
#else
    (void)max_threads;
#endif
    for (int t = 0; t < n_solvable; ++t) {
        const std::int32_t k = solvable[t];
        const i64 p0 = start[k], p1 = start[k + 1];
        i64 x0 = nx, x1 = -1, y0 = ny, y1 = -1;
        for (i64 p = p0; p < p1; ++p) {
            const i64 y = members[p] / nx, x = members[p] % nx;
            if (x < x0) x0 = x;
            if (x > x1) x1 = x;
            if (y < y0) y0 = y;
            if (y > y1) y1 = y;
        }
        const i64 bx = x1 - x0 + 1, by = y1 - y0 + 1;
        auto box = [&](i64 c) { return (c / nx - y0) * bx + (c % nx - x0); };
        // Unknowns: the component's cells that are not direct grounds.
        std::vector<std::uint8_t> present(static_cast<std::size_t>(bx * by), 0), unknown(present.size(), 0);
        std::vector<double> g(present.size(), 0.0), fg(present.size(), 0.0), b(present.size(), 0.0);
        for (i64 p = p0; p < p1; ++p) {
            const i64 c = members[p], q = box(c);
            present[q] = 1;
            unknown[q] = gnd[c] != inf;
            g[q] = conductance[c];
            fg[q] = finite[c];
            b[q] = src[c];
        }
        std::vector<i64> order;
        order.reserve(static_cast<std::size_t>(p1 - p0));
        nested_dissection(unknown.data(), bx, 0, by, 0, bx, order);
        std::vector<double> v(present.size(), 0.0);
        if (!order.empty()) {
            CellSystem sys;
            sys.analyze(bx, by, order, four_neighbors);
            // CHOLMOD factorizes with a 10ε shift (construct_cholesky_factor).
            sys.assemble(g.data(), present.data(), any_finite ? fg.data() : nullptr,
                         10 * std::numeric_limits<double>::epsilon());
            if (!sys.factorize()) {
#pragma omp critical(circuit_failed)
                failed = true;
                continue;
            }
            std::vector<double> rhs(order.size()), x(order.size());
            for (std::size_t i = 0; i < order.size(); ++i) rhs[i] = b[order[i]];
            sys.solve(rhs.data(), x.data());
            for (std::size_t i = 0; i < order.size(); ++i) v[order[i]] = x[i];
        }

        // Node currents (get_node_currents): branch currents b = w (v_n - v_c)
        // into each cell; those under 1e-8 of the largest one of their
        // direction are left out.
        double maxin = -inf, maxout = -inf;
        for (i64 p = p0; p < p1; ++p) {
            const i64 q = box(members[p]);
            const i64 y = q / bx, x = q % bx;
            for (int d = 0; d < nd; ++d) {
                const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
                if (yy < 0 || yy >= by || xx < 0 || xx >= bx || !present[yy * bx + xx]) continue;
                const i64 r = yy * bx + xx;
                const double bc = edge_weight(g[q], g[r], nb[d].diagonal) * (v[r] - v[q]);
                if (d < lower) {
                    if (bc > maxin) maxin = bc;
                } else if (bc > maxout) {
                    maxout = bc;
                }
            }
        }
        for (i64 p = p0; p < p1; ++p) {
            const i64 c = members[p], q = box(c);
            const i64 y = q / bx, x = q % bx;
            double inflow = 0.0, outflow = 0.0;
            for (int d = 0; d < nd; ++d) {
                const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
                if (yy < 0 || yy >= by || xx < 0 || xx >= bx || !present[yy * bx + xx]) continue;
                const i64 r = yy * bx + xx;
                const double bc = edge_weight(g[q], g[r], nb[d].diagonal) * (v[r] - v[q]);
                if (bc > 0) {
                    if (!(std::abs(bc / maxin) < 1e-8)) inflow += bc;
                } else if (bc < 0) {
                    if (!(std::abs(bc / maxout) < 1e-8)) outflow -= bc;
                }
            }
            if (any_finite) {
                const double f = fg[q] * v[q];
                if (f < 0) inflow -= f;
                else if (f > 0) outflow += f;
            }
            out.current[c] = inflow > outflow ? inflow : outflow;
            out.voltage[c] = v[q];
        }
    }
    if (failed) throw std::runtime_error("a circuit component is not positive definite");
    return out;
}

}  // namespace circuit
