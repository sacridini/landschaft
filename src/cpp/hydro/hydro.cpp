// SPDX-License-Identifier: GPL-3.0-or-later
//
// Terrain hydrology (see hydro.hpp). The flow directions and accumulations follow
// WhiteboxTools 2.4 (MIT; J. Lindsay) cell for cell, the flat areas Barnes, Lehman &
// Mulla (2014), the filling Barnes, Lehman & Mulla (2014) Priority-Flood.

#include "hydro/hydro.hpp"

#include <cmath>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hydro {

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr std::int8_t INFLOWING[8] = {4, 5, 6, 7, 0, 1, 2, 3};   // the direction that points back

struct Cells {
    int ny, nx;
    bool inside(i64 r, i64 c) const { return r >= 0 && c >= 0 && r < ny && c < nx; }
};

std::vector<double> lengths(const Grid& g) {
    const double diag = std::sqrt(g.dx * g.dx + g.dy * g.dy);
    return {diag, g.dx, diag, g.dy, diag, g.dx, diag, g.dy};
}

// A missing cell, or a cell outside the map.
inline bool missing_at(const Grid& g, i64 r, i64 c) {
    return r < 0 || c < 0 || r >= g.ny || c >= g.nx || std::isnan(g.z[r * g.nx + c]);
}

// A cell on the map's edge or next to a missing cell: water leaves the map there.
bool outlet(const Grid& g, i64 r, i64 c) {
    for (int k = 0; k < 8; ++k)
        if (missing_at(g, r + DROW[k], c + DCOL[k])) return true;
    return false;
}

}  // namespace

// ------------------------------------------------------------------------------- filling

std::vector<double> fill(const Grid& g, bool epsilon) {
    const i64 n = static_cast<i64>(g.ny) * g.nx;
    std::vector<double> z(g.z, g.z + n);
    std::vector<std::uint8_t> closed(static_cast<std::size_t>(n), 0);
    struct Item {   // equal elevations in cell order: the same result on every run
        double z;
        i64 cell;
        bool operator>(const Item& o) const { return z > o.z || (z == o.z && cell > o.cell); }
    };
    std::vector<Item> storage;
    storage.reserve(static_cast<std::size_t>(2 * (g.ny + g.nx)));
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open(std::greater<Item>(), std::move(storage));
    std::deque<i64> pits;
    for (i64 r = 0; r < g.ny; ++r)
        for (i64 c = 0; c < g.nx; ++c) {
            const i64 i = r * g.nx + c;
            if (std::isnan(z[i]) || !outlet(g, r, c)) continue;
            closed[i] = 1;
            open.push({z[i], i});
        }
    const double inf = std::numeric_limits<double>::infinity();
    while (!pits.empty() || !open.empty()) {
        i64 i;
        if (!pits.empty()) {
            i = pits.front();
            pits.pop_front();
        } else {
            i = open.top().cell;
            open.pop();
        }
        const i64 r = i / g.nx, c = i % g.nx;
        const double zc = z[i];
        const double up = epsilon ? std::nextafter(zc, inf) : zc;
        for (int k = 0; k < 8; ++k) {
            const i64 rn = r + DROW[k], cn = c + DCOL[k];
            if (rn < 0 || cn < 0 || rn >= g.ny || cn >= g.nx) continue;
            const i64 j = rn * g.nx + cn;
            if (closed[j] || std::isnan(z[j])) continue;
            closed[j] = 1;
            if (z[j] <= up) {
                z[j] = up;
                pits.push_back(j);
            } else {
                open.push({z[j], j});
            }
        }
    }
    return z;
}

// ------------------------------------------------------------------------------- D8

std::vector<std::int8_t> d8(const Grid& g, bool resolve_flats, int n_threads) {
    const i64 n = static_cast<i64>(g.ny) * g.nx;
    const std::vector<double> len = lengths(g);
    std::vector<std::int8_t> dir(static_cast<std::size_t>(n), MISSING);
#pragma omp parallel for schedule(static) num_threads(n_threads) if (n_threads > 1 && n > 65536)
    for (i64 r = 0; r < g.ny; ++r) {
        for (i64 c = 0; c < g.nx; ++c) {
            const double z = g.z[r * g.nx + c];
            if (std::isnan(z)) continue;
            int best = -1;
            double max_slope = std::numeric_limits<double>::lowest();
            for (int k = 0; k < 8; ++k) {
                const i64 rn = r + DROW[k], cn = c + DCOL[k];
                if (missing_at(g, rn, cn)) continue;
                const double s = (z - g.z[rn * g.nx + cn]) / len[k];
                if (s > max_slope && s > 0.0) {
                    max_slope = s;
                    best = k;
                }
            }
            dir[r * g.nx + c] = static_cast<std::int8_t>(best);
        }
    }
    if (resolve_flats) {
        const std::vector<std::int8_t> flats = flat_directions(g, dir);
        for (i64 i = 0; i < n; ++i)
            if (flats[i] >= 0) dir[i] = flats[i];
    }
    return dir;
}

std::vector<std::int8_t> flat_directions(const Grid& g, const std::vector<std::int8_t>& dir) {
    const i64 n = static_cast<i64>(g.ny) * g.nx;
    const Cells cells{g.ny, g.nx};
    std::vector<std::int8_t> out(static_cast<std::size_t>(n), NO_FLOW);
    // flat: no lower neighbour, not draining off the map
    std::vector<std::uint8_t> flat(static_cast<std::size_t>(n), 0), drains_off(static_cast<std::size_t>(n), 0);
    for (i64 r = 0; r < g.ny; ++r)
        for (i64 c = 0; c < g.nx; ++c) {
            const i64 i = r * g.nx + c;
            if (dir[i] != NO_FLOW) continue;
            if (outlet(g, r, c)) drains_off[i] = 1;
            else flat[i] = 1;
        }
    auto equal_flat_neighbour = [&](i64 r, i64 c, double z) {
        for (int k = 0; k < 8; ++k) {
            const i64 rn = r + DROW[k], cn = c + DCOL[k];
            if (!cells.inside(rn, cn)) continue;
            const i64 j = rn * g.nx + cn;
            if (flat[j] && g.z[j] == z) return true;
        }
        return false;
    };
    // low edges: cells that drain (or leave the map) next to a flat cell of their elevation;
    // high edges: flat cells next to higher terrain
    std::vector<i64> low, high;
    for (i64 r = 0; r < g.ny; ++r)
        for (i64 c = 0; c < g.nx; ++c) {
            const i64 i = r * g.nx + c;
            const double z = g.z[i];
            if (std::isnan(z)) continue;
            if ((dir[i] >= 0 || drains_off[i]) && equal_flat_neighbour(r, c, z)) low.push_back(i);
            if (flat[i]) {
                for (int k = 0; k < 8; ++k) {
                    const i64 rn = r + DROW[k], cn = c + DCOL[k];
                    if (missing_at(g, rn, cn)) continue;
                    if (g.z[rn * g.nx + cn] > z) {
                        high.push_back(i);
                        break;
                    }
                }
            }
        }
    if (low.empty()) return out;
    // label each flat (with its low edges) from its low edges
    std::vector<std::int32_t> label(static_cast<std::size_t>(n), 0);
    std::int32_t labels = 0;
    std::deque<i64> q;
    for (i64 s : low) {
        if (label[s]) continue;
        label[s] = ++labels;
        const double z = g.z[s];
        q.push_back(s);
        while (!q.empty()) {
            const i64 i = q.front();
            q.pop_front();
            const i64 r = i / g.nx, c = i % g.nx;
            for (int k = 0; k < 8; ++k) {
                const i64 rn = r + DROW[k], cn = c + DCOL[k];
                if (!cells.inside(rn, cn)) continue;
                const i64 j = rn * g.nx + cn;
                if (label[j] || g.z[j] != z) continue;
                if (!flat[j] && !(dir[j] >= 0 || drains_off[j])) continue;
                label[j] = labels;
                q.push_back(j);
            }
        }
    }
    // away from higher terrain: breadth-first steps from the high edges
    std::vector<std::int32_t> away(static_cast<std::size_t>(n), 0), towards(static_cast<std::size_t>(n), 0);
    std::vector<std::int32_t> height(static_cast<std::size_t>(labels) + 1, 0);
    auto spread = [&](const std::vector<i64>& seeds, std::vector<std::int32_t>& step, bool from_low) {
        std::vector<i64> now, next;
        for (i64 s : seeds)
            if (label[s] && !step[s]) {
                step[s] = 1;
                now.push_back(s);
            }
        std::int32_t loops = 1;
        while (!now.empty()) {
            ++loops;
            next.clear();
            for (i64 i : now) {
                if (!from_low) height[label[i]] = std::max(height[label[i]], step[i]);
                const i64 r = i / g.nx, c = i % g.nx;
                for (int k = 0; k < 8; ++k) {
                    const i64 rn = r + DROW[k], cn = c + DCOL[k];
                    if (!cells.inside(rn, cn)) continue;
                    const i64 j = rn * g.nx + cn;
                    if (!flat[j] || label[j] != label[i] || step[j]) continue;
                    step[j] = loops;
                    next.push_back(j);
                }
            }
            now.swap(next);
        }
    };
    spread(high, away, false);
    spread(low, towards, true);
    // the mask: twice the steps to lower terrain, plus the steps left to higher terrain
    std::vector<std::int64_t> mask(static_cast<std::size_t>(n), 0);
    for (i64 i = 0; i < n; ++i) {
        if (!label[i] || !towards[i]) continue;
        mask[i] = 2 * static_cast<std::int64_t>(towards[i]) + (away[i] ? height[label[i]] - away[i] : 0);
    }
    for (i64 r = 0; r < g.ny; ++r)
        for (i64 c = 0; c < g.nx; ++c) {
            const i64 i = r * g.nx + c;
            if (!flat[i] || !label[i] || !towards[i]) continue;
            std::int64_t best = mask[i];
            int to = -1;
            for (int k = 0; k < 8; ++k) {
                const i64 rn = r + DROW[k], cn = c + DCOL[k];
                if (!cells.inside(rn, cn)) continue;
                const i64 j = rn * g.nx + cn;
                if (label[j] != label[i] || !towards[j]) continue;
                if (mask[j] < best) {
                    best = mask[j];
                    to = k;
                }
            }
            out[i] = static_cast<std::int8_t>(to);
        }
    return out;
}

std::vector<double> d8_accumulation(const std::int8_t* dir, const double* weight, int ny, int nx) {
    const i64 n = static_cast<i64>(ny) * nx;
    const Cells cells{ny, nx};
    std::vector<double> acc(static_cast<std::size_t>(n), std::numeric_limits<double>::quiet_NaN());
    std::vector<std::int8_t> inflow(static_cast<std::size_t>(n), -1);
    std::vector<i64> stack;
    stack.reserve(static_cast<std::size_t>(n));
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const i64 i = r * nx + c;
            if (dir[i] == MISSING) continue;
            acc[i] = weight ? weight[i] : 1.0;
            std::int8_t count = 0;
            for (int k = 0; k < 8; ++k) {
                const i64 rn = r + DROW[k], cn = c + DCOL[k];
                if (cells.inside(rn, cn) && dir[rn * nx + cn] == INFLOWING[k]) ++count;
            }
            inflow[i] = count;
            if (count == 0) stack.push_back(i);
        }
    while (!stack.empty()) {
        const i64 i = stack.back();
        stack.pop_back();
        const int d = dir[i];
        if (d < 0) continue;
        const i64 rn = i / nx + DROW[d], cn = i % nx + DCOL[d];
        if (!cells.inside(rn, cn)) continue;
        const i64 j = rn * nx + cn;
        if (dir[j] == MISSING) continue;
        acc[j] += acc[i];
        if (--inflow[j] == 0) stack.push_back(j);
    }
    return acc;
}

// ------------------------------------------------------------------------------- D-infinity

std::vector<double> dinf(const Grid& g, const std::int8_t* flats, int n_threads) {
    const i64 n = static_cast<i64>(g.ny) * g.nx;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> out(static_cast<std::size_t>(n), nan);
    const double grid_res = (g.dx + g.dy) / 2.0;
    const double diag = std::sqrt(g.dx * g.dx + g.dy * g.dy);
    static constexpr double AC[8] = {0, 1, 1, 2, 2, 3, 3, 4};
    static constexpr double AF[8] = {1, -1, 1, -1, 1, -1, 1, -1};
    static constexpr int E1C[8] = {1, 0, 0, -1, -1, 0, 0, 1};
    static constexpr int E1R[8] = {0, -1, -1, 0, 0, 1, 1, 0};
    static constexpr int E2C[8] = {1, 1, -1, -1, -1, -1, 1, 1};
    static constexpr int E2R[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
    // the azimuth of each D8 direction, for the flat cells
    static constexpr double AZIMUTH[8] = {45, 90, 135, 180, 225, 270, 315, 360};
    const double atan1 = std::atan(1.0);
    const double half_pi = PI / 2.0;
#pragma omp parallel for schedule(static) num_threads(n_threads) if (n_threads > 1 && n > 65536)
    for (i64 r = 0; r < g.ny; ++r) {
        for (i64 c = 0; c < g.nx; ++c) {
            const i64 i = r * g.nx + c;
            const double e0 = g.z[i];
            if (std::isnan(e0)) continue;
            double dir = 360.0, max_slope = std::numeric_limits<double>::lowest();
            for (int k = 0; k < 8; ++k) {
                const i64 r1 = r + E1R[k], c1 = c + E1C[k], r2 = r + E2R[k], c2 = c + E2C[k];
                if (missing_at(g, r1, c1) || missing_at(g, r2, c2)) continue;
                const double e1 = g.z[r1 * g.nx + c1], e2 = g.z[r2 * g.nx + c2];
                double rr, s;
                if (e0 > e1 && e0 > e2) {
                    const double s1 = (e0 - e1) / grid_res, s2 = (e1 - e2) / grid_res;
                    rr = s1 != 0.0 ? std::atan(s2 / s1) : PI / 2.0;
                    s = std::sqrt(s1 * s1 + s2 * s2);
                    if (rr < 0.0 || rr > atan1) {
                        if (rr < 0.0) {
                            rr = 0.0;
                            s = s1;
                        } else {
                            rr = atan1;
                            s = (e0 - e2) / diag;
                        }
                    }
                } else if (e0 > e1 || e0 > e2) {
                    if (e0 > e1) {
                        rr = 0.0;
                        s = (e0 - e1) / grid_res;
                    } else {
                        rr = atan1;
                        s = (e0 - e2) / diag;
                    }
                } else {
                    continue;
                }
                if (s >= max_slope) {
                    max_slope = s;
                    dir = AF[k] * rr + AC[k] * half_pi;
                }
            }
            if (max_slope > 0.0) {
                double deg = 360.0 - dir * (180.0 / PI) + 90.0;
                if (deg > 360.0) deg -= 360.0;
                out[i] = deg;
            } else if (flats && flats[i] >= 0) {
                out[i] = AZIMUTH[flats[i]];
            } else {
                out[i] = -1.0;
            }
        }
    }
    return out;
}

namespace {

// The two cells (by D8 index) that receive the flow of a D-infinity azimuth, and their shares.
inline void dinf_split(double a, int& k1, double& p1, int& k2, double& p2) {
    // sectors of 45 degrees clockwise from north: N-NE, NE-E, E-SE, ...
    static constexpr int FIRST[8] = {7, 0, 1, 2, 3, 4, 5, 6};
    static constexpr int SECOND[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int s = 7;
    for (int k = 1; k <= 7; ++k)
        if (a < 45.0 * k) {
            s = k - 1;
            break;
        }
    const double hi = s < 7 ? 45.0 * (s + 1) : 360.0;
    p1 = (hi - a) / 45.0;
    p2 = (a - (hi - 45.0)) / 45.0;
    k1 = FIRST[s];
    k2 = SECOND[s];
}

}  // namespace

std::vector<double> dinf_accumulation(const double* angle, const double* weight, int ny, int nx) {
    const i64 n = static_cast<i64>(ny) * nx;
    const Cells cells{ny, nx};
    std::vector<double> acc(static_cast<std::size_t>(n), std::numeric_limits<double>::quiet_NaN());
    std::vector<std::int8_t> inflow(static_cast<std::size_t>(n), -1);
    // which neighbour's azimuth points into a cell: open sectors, as WhiteboxTools counts them
    static constexpr double START[8] = {180, 225, 270, 315, 0, 45, 90, 135};
    static constexpr double END[8] = {270, 315, 360, 45, 90, 135, 180, 225};
    std::vector<i64> stack;
    stack.reserve(static_cast<std::size_t>(n));
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const i64 i = r * nx + c;
            if (std::isnan(angle[i])) continue;
            acc[i] = weight ? weight[i] : 1.0;
            std::int8_t count = 0;
            for (int k = 0; k < 8; ++k) {
                const i64 rn = r + DROW[k], cn = c + DCOL[k];
                if (!cells.inside(rn, cn)) continue;
                const double a = angle[rn * nx + cn];
                if (!(a >= 0.0)) continue;
                if (k != 3 ? (a > START[k] && a < END[k]) : (a > START[k] || a < END[k])) ++count;
            }
            inflow[i] = count;
            if (count == 0) stack.push_back(i);
        }
    auto pass = [&](i64 i, int k, double share) {
        const i64 rn = i / nx + DROW[k], cn = i % nx + DCOL[k];
        if (!cells.inside(rn, cn)) return;
        const i64 j = rn * nx + cn;
        if (std::isnan(angle[j])) return;
        acc[j] += acc[i] * share;
        if (--inflow[j] == 0) stack.push_back(j);
    };
    while (!stack.empty()) {
        const i64 i = stack.back();
        stack.pop_back();
        const double a = angle[i];
        if (!(a >= 0.0)) continue;
        int k1, k2;
        double p1, p2;
        dinf_split(a, k1, p1, k2, p2);
        if (p1 > 0.0) pass(i, k1, p1);
        if (p2 > 0.0) pass(i, k2, p2);
    }
    return acc;
}

// ------------------------------------------------------------------------------- MFD

std::vector<double> mfd_accumulation(const Grid& g, double exponent, bool quinn, const std::int8_t* flats,
                                     const double* weight) {
    const i64 n = static_cast<i64>(g.ny) * g.nx;
    const Cells cells{g.ny, g.nx};
    const std::vector<double> len = lengths(g);
    static constexpr double CONTOUR[8] = {0.354, 0.5, 0.354, 0.5, 0.354, 0.5, 0.354, 0.5};
    std::vector<double> acc(static_cast<std::size_t>(n), std::numeric_limits<double>::quiet_NaN());
    std::vector<std::int8_t> inflow(static_cast<std::size_t>(n), -1);
    std::vector<i64> stack;
    stack.reserve(static_cast<std::size_t>(n));
    for (i64 r = 0; r < g.ny; ++r)
        for (i64 c = 0; c < g.nx; ++c) {
            const i64 i = r * g.nx + c;
            const double z = g.z[i];
            if (std::isnan(z)) continue;
            acc[i] = weight ? weight[i] : 1.0;
            std::int8_t count = 0;
            for (int k = 0; k < 8; ++k) {
                const i64 rn = r + DROW[k], cn = c + DCOL[k];
                if (!cells.inside(rn, cn)) continue;
                const i64 j = rn * g.nx + cn;
                const double zn = g.z[j];
                if (std::isnan(zn)) continue;
                // a higher neighbour sends here; so does a flat neighbour pointing here
                if (zn > z || (flats && flats[j] == INFLOWING[k] && zn == z)) ++count;
            }
            inflow[i] = count;
            if (count == 0) stack.push_back(i);
        }
    while (!stack.empty()) {
        const i64 i = stack.back();
        stack.pop_back();
        const i64 r = i / g.nx, c = i % g.nx;
        const double z = g.z[i];
        double w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        bool down[8] = {false, false, false, false, false, false, false, false};
        double total = 0.0;
        for (int k = 0; k < 8; ++k) {
            const i64 rn = r + DROW[k], cn = c + DCOL[k];
            if (missing_at(g, rn, cn)) continue;
            const double zn = g.z[rn * g.nx + cn];
            if (zn < z) {
                const double s = (z - zn) / len[k];
                w[k] = std::pow(s, exponent) * (quinn ? CONTOUR[k] : 1.0);
                total += w[k];
                down[k] = true;
            }
        }
        if (total <= 0.0 && flats && flats[i] >= 0) {
            w[flats[i]] = 1.0;
            down[flats[i]] = true;
            total = 1.0;
        }
        if (total <= 0.0) continue;
        const double fa = acc[i];
        for (int k = 0; k < 8; ++k) {
            if (!down[k]) continue;
            const i64 j = (r + DROW[k]) * g.nx + (c + DCOL[k]);
            acc[j] += fa * (w[k] / total);
            if (--inflow[j] == 0) stack.push_back(j);
        }
    }
    return acc;
}

// ------------------------------------------------------------------------------- HAND

std::vector<double> hand(const Grid& g, const std::int8_t* dir, const std::uint8_t* streams) {
    const i64 n = static_cast<i64>(g.ny) * g.nx;
    const Cells cells{g.ny, g.nx};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> out(static_cast<std::size_t>(n), nan);
    std::vector<std::uint8_t> done(static_cast<std::size_t>(n), 0);
    // from each stream cell up the cells that drain into it, carrying its elevation
    std::vector<std::pair<i64, double>> stack;
    for (i64 i = 0; i < n; ++i) {
        if (std::isnan(g.z[i])) {
            done[i] = 1;
            continue;
        }
        if (streams[i]) {
            out[i] = 0.0;
            done[i] = 1;
            stack.emplace_back(i, g.z[i]);
        }
    }
    while (!stack.empty()) {
        const auto [i, zs] = stack.back();
        stack.pop_back();
        const i64 r = i / g.nx, c = i % g.nx;
        for (int k = 0; k < 8; ++k) {
            const i64 rn = r + DROW[k], cn = c + DCOL[k];
            if (!cells.inside(rn, cn)) continue;
            const i64 j = rn * g.nx + cn;
            if (done[j] || dir[j] != INFLOWING[k]) continue;
            done[j] = 1;
            out[j] = g.z[j] - zs;
            stack.emplace_back(j, zs);
        }
    }
    return out;
}

}  // namespace hydro
