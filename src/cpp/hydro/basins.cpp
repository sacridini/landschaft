// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stream links and watersheds along D8 directions (see hydro.hpp): the segments of a
// stream network between sources, junctions and outlets, and the area draining to each
// of a set of seed cells.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "hydro/hydro.hpp"

namespace hydro {

namespace {

constexpr std::int8_t BACK[8] = {4, 5, 6, 7, 0, 1, 2, 3};   // the direction that points back

// The cell a cell drains to, or -1 (no direction, off the map or into a missing cell).
inline i64 downstream(const std::int8_t* dir, int ny, int nx, i64 i) {
    const int d = dir[i];
    if (d < 0) return -1;
    const i64 r = i / nx + DROW[d], c = i % nx + DCOL[d];
    if (r < 0 || c < 0 || r >= ny || c >= nx) return -1;
    const i64 j = r * nx + c;
    return dir[j] == MISSING ? -1 : j;
}

}  // namespace

StreamLinks stream_links(const std::int8_t* dir, const std::uint8_t* stream, const std::uint8_t* breaks, int ny,
                         int nx) {
    const i64 n = static_cast<i64>(ny) * nx;
    auto on = [&](i64 i) { return i >= 0 && stream[i] != 0 && dir[i] != MISSING; };
    // stream cells draining into each stream cell, and the last one of them
    std::vector<std::int8_t> inflows(static_cast<std::size_t>(n), 0);
    std::vector<i64> from(static_cast<std::size_t>(n), -1);
    for (i64 i = 0; i < n; ++i) {
        if (!on(i)) continue;
        const i64 j = downstream(dir, ny, nx, i);
        if (on(j)) {
            ++inflows[j];
            from[j] = i;
        }
    }
    // a link starts at a source, below a junction and below a break (a forced outlet)
    auto starts = [&](i64 i) { return inflows[i] != 1 || (breaks && breaks[from[i]] != 0); };
    StreamLinks out;
    out.link.assign(static_cast<std::size_t>(n), 0);
    for (i64 i = 0; i < n; ++i) {
        if (!on(i) || !starts(i)) continue;
        const int id = static_cast<int>(out.start.size()) + 1;
        out.start.push_back(i);
        i64 c = i;
        for (;;) {
            out.link[c] = id;
            const i64 j = downstream(dir, ny, nx, c);
            if ((breaks && breaks[c] != 0) || !on(j) || starts(j)) break;
            c = j;
        }
        out.end.push_back(c);
    }
    out.down.assign(out.start.size(), 0);
    for (std::size_t k = 0; k < out.end.size(); ++k) {
        const i64 j = downstream(dir, ny, nx, out.end[k]);
        if (on(j)) out.down[k] = out.link[j];
    }
    return out;
}

std::vector<std::int32_t> watersheds(const std::int8_t* dir, const std::int32_t* seeds, int ny, int nx) {
    const i64 n = static_cast<i64>(ny) * nx;
    std::vector<std::int32_t> label(static_cast<std::size_t>(n), 0);
    std::vector<i64> stack;
    for (i64 i = 0; i < n; ++i)
        if (seeds[i] != 0 && dir[i] != MISSING) {
            label[i] = seeds[i];
            stack.push_back(i);
        }
    while (!stack.empty()) {
        const i64 i = stack.back();
        stack.pop_back();
        const i64 r = i / nx, c = i % nx;
        for (int k = 0; k < 8; ++k) {
            const i64 rn = r + DROW[k], cn = c + DCOL[k];
            if (rn < 0 || cn < 0 || rn >= ny || cn >= nx) continue;
            const i64 j = rn * nx + cn;
            if (dir[j] != BACK[k] || seeds[j] != 0) continue;
            label[j] = label[i];
            stack.push_back(j);
        }
    }
    return label;
}

std::vector<double> flow_distance(const std::int8_t* dir, const std::uint8_t* targets, int ny, int nx, double dx,
                                  double dy) {
    const i64 n = static_cast<i64>(ny) * nx;
    const double diag = std::sqrt(dx * dx + dy * dy);
    const double step[8] = {diag, dx, diag, dy, diag, dx, diag, dy};
    std::vector<double> dist(static_cast<std::size_t>(n), std::numeric_limits<double>::quiet_NaN());
    std::vector<i64> stack;
    for (i64 i = 0; i < n; ++i)
        if (targets[i] != 0 && dir[i] != MISSING) {
            dist[i] = 0.0;
            stack.push_back(i);
        }
    while (!stack.empty()) {
        const i64 i = stack.back();
        stack.pop_back();
        const i64 r = i / nx, c = i % nx;
        for (int k = 0; k < 8; ++k) {
            const i64 rn = r + DROW[k], cn = c + DCOL[k];
            if (rn < 0 || cn < 0 || rn >= ny || cn >= nx) continue;
            const i64 j = rn * nx + cn;
            if (dir[j] != BACK[k] || targets[j] != 0) continue;
            dist[j] = dist[i] + step[dir[j]];
            stack.push_back(j);
        }
    }
    return dist;
}

}  // namespace hydro
