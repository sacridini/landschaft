// SPDX-License-Identifier: GPL-3.0-or-later

#include "circuit/grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "circuit/system.hpp"

namespace circuit {

void node_currents(i64 w, i64 h, const double* g, const std::uint8_t* present, const double* v,
                   const double* finite, bool four_neighbors, double* cur, std::vector<double>& edges) {
    const Offset* nb = four_neighbors ? kNeighbours4 : kNeighbours8;
    const int nd = four_neighbors ? 4 : 8;
    const int lower = four_neighbors ? kLower4 : kLower8;
    const int nf = nd - lower;  // neighbours after the cell
    const i64 n = w * h;
    edges.resize(static_cast<std::size_t>(n * nf));
    double* e = edges.data();
    // Every edge once, from its earlier cell q to the later r: b = w (v_r - v_q)
    // is the current into q in column q, row r (the lower triangle: maxout);
    // seen from r it is -b (the upper triangle: maxin), exactly.
    const double inf = std::numeric_limits<double>::infinity();
    double maxin = -inf, maxout = -inf;
    for (i64 q = 0; q < n; ++q) {
        if (!present[q]) continue;
        const i64 y = q / w, x = q % w;
        for (int d = lower; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
            const i64 r = yy * w + xx;
            if (!present[r]) continue;
            const double b = edge_weight(g[q], g[r], nb[d].diagonal) * (v[r] - v[q]);
            e[q * nf + (d - lower)] = b;
            if (b > maxout) maxout = b;
            if (-b > maxin) maxin = -b;
        }
    }
    for (i64 q = 0; q < n; ++q) {
        if (!present[q]) continue;
        const i64 y = q / w, x = q % w;
        double inflow = 0.0, outflow = 0.0;
        for (int d = 0; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
            const i64 r = yy * w + xx;
            if (!present[r]) continue;
            // The opposite of direction d is nd - 1 - d.
            const double b = d < lower ? -e[r * nf + (nd - 1 - d - lower)] : e[q * nf + (d - lower)];
            if (b > 0) {
                if (!(std::abs(b / maxin) < 1e-8)) inflow += b;
            } else if (b < 0) {
                if (!(std::abs(b / maxout) < 1e-8)) outflow -= b;
            }
        }
        if (finite) {
            const double f = finite[q] * v[q];
            if (f < 0) inflow -= f;
            else if (f > 0) outflow += f;
        }
        cur[q] = inflow > outflow ? inflow : outflow;
    }
}

AdvancedResult advanced(const double* conductance, const double* source, const double* ground, i64 nx, i64 ny,
                        bool four_neighbors, int max_threads) {
    const i64 n = nx * ny;
    const Offset* nb = four_neighbors ? kNeighbours4 : kNeighbours8;
    const int nd = four_neighbors ? 4 : 8;
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
        std::vector<double> v(present.size(), 0.0);
        if (std::find(unknown.begin(), unknown.end(), std::uint8_t{1}) != unknown.end()) {
            CellSystem sys;
            sys.analyze(bx, by, unknown.data(), four_neighbors);
            const std::vector<i64>& order = sys.cells();
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

        std::vector<double> cur(present.size(), 0.0), edges;
        node_currents(bx, by, g.data(), present.data(), v.data(), any_finite ? fg.data() : nullptr, four_neighbors,
                      cur.data(), edges);
        for (i64 p = p0; p < p1; ++p) {
            const i64 c = members[p], q = box(c);
            out.current[c] = cur[q];
            out.voltage[c] = v[q];
        }
    }
    if (failed) throw std::runtime_error("a circuit component is not positive definite");
    return out;
}

}  // namespace circuit
