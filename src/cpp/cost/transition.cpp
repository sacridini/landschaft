// SPDX-License-Identifier: GPL-3.0-or-later
#include "cost/transition.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "cost/geodesic.hpp"
#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"

namespace cost {

namespace {

constexpr Offset kAll16[16] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, -1}, {-1, 0}, {-1, 1}, {-1, 2}, {0, -1},
                               {0, 1},   {1, -2}, {1, -1},  {1, 0},   {1, 1},  {1, 2},  {2, -1}, {2, 1}};

bool in_set(Offset o, int directions) {
    const int ar = std::abs(o.dr), ac = std::abs(o.dc);
    if (directions == 4) return ar + ac == 1;
    if (directions == 8) return ar <= 1 && ac <= 1;
    return true;
}

// Neighbours of a cell with the value of each connection, sorted by the
// neighbour's index (the Matrix package's order within a row or column).
struct Entries {
    std::pair<i64, double> e[16];
    int n = 0;
    void add(i64 k, double v) {
        int m = n++;
        while (m > 0 && e[m - 1].first > k) {
            e[m] = e[m - 1];
            --m;
        }
        e[m] = {k, v};
    }
};

// The entries of row `cell` (margin 0: T[cell, j]) or column `cell` (T[i, cell]).
void entries_of(const Grid& g, const std::vector<Offset>& off, bool symmetric, const double* values, i64 cell,
                int margin, Entries& out) {
    const i64 n = g.ncell();
    const i64 r = cell / g.nx, c = cell % g.nx;
    const int nd = static_cast<int>(off.size());
    out.n = 0;
    for (int d = 0; d < nd; ++d) {
        // the stored connection (cell, d) ...
        const i64 b = g.neighbour(r, c, off[d]);
        if (b >= 0) {
            const double v = values[d * n + cell];
            if (v != 0 && (symmetric || margin == 0)) out.add(b, v);
        }
        // ... and (a, d) with cell as its neighbour
        const i64 a = g.neighbour(r, c, Offset{-off[d].dr, -off[d].dc});
        if (a >= 0) {
            const double v = values[d * n + a];
            if (v != 0 && (symmetric || margin == 1)) out.add(a, v);
        }
    }
}

double planar(double x1, double y1, double x2, double y2, bool scalar) {
    const double dx = x1 - x2, dy = y1 - y2;
    return lsm::r_sqrt(dx * dx + dy * dy, scalar);
}

}  // namespace

std::vector<Offset> offsets(int directions, bool symmetric) {
    if (directions != 4 && directions != 8 && directions != 16)
        throw std::invalid_argument("directions must be 4, 8 or 16");
    std::vector<Offset> out;
    for (const Offset o : kAll16) {
        if (!in_set(o, directions)) continue;
        if (symmetric && !(o.dr > 0 || (o.dr == 0 && o.dc > 0))) continue;
        out.push_back(o);
    }
    return out;
}

void neighbours(const Grid& g, int directions, bool symmetric, i64* out) {
    const auto off = offsets(directions, symmetric);
    const i64 n = g.ncell();
    const i64 nd = static_cast<i64>(off.size());
#pragma omp parallel for schedule(static)
    for (i64 r = 0; r < g.ny; ++r) {
        for (i64 d = 0; d < nd; ++d) {
            for (i64 c = 0; c < g.nx; ++c) out[d * n + r * g.nx + c] = g.neighbour(r, c, off[d]);
        }
    }
}

void builtin_values(const double* x, const Grid& g, int directions, bool symmetric, Builtin f, double* out) {
    const auto off = offsets(directions, symmetric);
    const i64 n = g.ncell();
    const i64 nd = static_cast<i64>(off.size());
#pragma omp parallel for schedule(static)
    for (i64 r = 0; r < g.ny; ++r) {
        for (i64 d = 0; d < nd; ++d) {
            double* o = out + d * n + r * g.nx;
            for (i64 c = 0; c < g.nx; ++c) {
                const i64 cell = r * g.nx + c;
                const i64 b = g.neighbour(r, c, off[d]);
                o[c] = 0;
                if (b < 0 || !std::isfinite(x[cell]) || !std::isfinite(x[b])) continue;
                double p[2];
                if (symmetric && b < cell) {
                    p[0] = x[b];
                    p[1] = x[cell];
                } else {
                    p[0] = x[cell];
                    p[1] = x[b];
                }
                double v = 0;
                switch (f) {
                    case Builtin::Mean: v = lsm::r_mean(p, 2); break;
                    case Builtin::Min: v = p[1] < p[0] ? p[1] : p[0]; break;
                    case Builtin::Max: v = p[1] > p[0] ? p[1] : p[0]; break;
                    case Builtin::Sum: v = lsm::r_sum(p, 2); break;
                    case Builtin::From: v = p[0]; break;
                    case Builtin::To: v = p[1]; break;
                    case Builtin::Diff: v = p[1] - p[0]; break;
                    case Builtin::Harmonic: {
                        const double q[2] = {1.0 / p[0], 1.0 / p[1]};
                        v = 1.0 / lsm::r_mean(q, 2);
                        break;
                    }
                    case Builtin::Geometric: {
                        const double q[2] = {lsm::r_log(p[0]), lsm::r_log(p[1])};
                        v = lsm::r_exp(lsm::r_mean(q, 2));
                        break;
                    }
                    case Builtin::Product:
                        v = static_cast<double>(lsm::Extended(p[0]) * lsm::Extended(p[1]));
                        break;
                }
                o[c] = v;
            }
        }
    }
}

double scale_value(const Grid& g, bool lonlat) {
    const double xs[2] = {g.xmin, g.xmax}, ys[2] = {g.ymin, g.ymax};
    const double mx = lsm::r_mean(xs, 2), my = lsm::r_mean(ys, 2);
    const double x2 = mx + g.xres(), y2 = my + 0.0;
    if (lonlat) return Geodesic().distance(mx, my, x2, y2);
    return planar(mx, my, x2, y2, true);  // one number: R's inline sqrt
}

void geo_correction(const Grid& g, int directions, bool symmetric, bool lonlat, char type, bool resistance,
                    bool scale, bool factors, double* values) {
    if (type != 'c' && type != 'r') throw std::invalid_argument("type must be 'c' or 'r'");
    const auto off = offsets(directions, symmetric);
    const i64 n = g.ncell();
    const i64 nd = static_cast<i64>(off.size());
    const double s = scale ? scale_value(g, lonlat) : 1.0;
    // raster's .planedist on a single pair runs R's scalar (inline) sqrt
    i64 count = 0;
    for (i64 k = 0; k < nd * n && count < 2; ++k) count += values[k] != 0;
    const bool scalar = count == 1;
    const double deg = 3.141592653589793238462643383280 / 180;  // R's pi / 180
    const Geodesic geod;
#pragma omp parallel for schedule(dynamic, 16)
    for (i64 r = 0; r < g.ny; ++r) {
        for (i64 d = 0; d < nd; ++d) {
            for (i64 c = 0; c < g.nx; ++c) {
                const i64 cell = r * g.nx + c;
                double& v = values[d * n + cell];
                if (v == 0) continue;
                const i64 b = g.neighbour(r, c, off[d]);
                // symmetric: the upper triangle, from the lower index
                i64 p = cell, q = b;
                if (symmetric && b < cell) std::swap(p, q);
                const i64 pr = p / g.nx, pc = p % g.nx, qr = q / g.nx, qc = q % g.nx;
                const double x1 = g.x(pc), y1 = g.y(pr), x2 = g.x(qc), y2 = g.y(qr);
                const double dist = lonlat ? geod.distance(x1, y1, x2, y2) : planar(x1, y1, x2, y2, scalar);
                double f = resistance ? dist / s : 1.0 / (dist / s);
                if (lonlat && type == 'r' && pr != qr) {
                    const double m = static_cast<double>((lsm::Extended(y1) + lsm::Extended(y2)) / lsm::Extended(2.0));
                    const double cs = lsm::r_cos(deg * m);
                    f = f * (resistance ? 1.0 / cs : cs);
                }
                v = factors ? f : f * v;
            }
        }
    }
}

void margin_sums(const Grid& g, int directions, bool symmetric, const double* values, int margin, bool count,
                 double* out) {
    const auto off = offsets(directions, symmetric);
    const i64 n = g.ncell();
#pragma omp parallel for schedule(static)
    for (i64 cell = 0; cell < n; ++cell) {
        Entries e;
        entries_of(g, off, symmetric, values, cell, margin, e);
        double s = 0.0;
        for (int k = 0; k < e.n; ++k) s += count ? 1.0 : e.e[k].second;
        out[cell] = s;
    }
}

Csc to_csc(const Grid& g, int directions, bool symmetric, const double* values) {
    const auto off = offsets(directions, symmetric);
    const i64 n = g.ncell();
    Csc m;
    m.p.assign(static_cast<std::size_t>(n + 1), 0);
#pragma omp parallel for schedule(static)
    for (i64 cell = 0; cell < n; ++cell) {
        Entries e;
        entries_of(g, off, symmetric, values, cell, 1, e);
        m.p[cell + 1] = e.n;
    }
    for (i64 j = 0; j < n; ++j) m.p[j + 1] += m.p[j];
    m.i.resize(static_cast<std::size_t>(m.p[n]));
    m.x.resize(static_cast<std::size_t>(m.p[n]));
#pragma omp parallel for schedule(static)
    for (i64 cell = 0; cell < n; ++cell) {
        Entries e;
        entries_of(g, off, symmetric, values, cell, 1, e);
        for (int k = 0; k < e.n; ++k) {
            m.i[m.p[cell] + k] = e.e[k].first;
            m.x[m.p[cell] + k] = e.e[k].second;
        }
    }
    return m;
}

}  // namespace cost
