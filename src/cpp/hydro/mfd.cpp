// SPDX-License-Identifier: GPL-3.0-or-later
//
// The multiple flow direction routing of pygeoprocessing 2.4 (Apache-2.0; Natural
// Capital Project), which the InVEST models (SDR, NDR, seasonal water yield) use: the
// directions packed as eight 4-bit weights, the accumulation pulled from upstream and
// the mean distance to a channel. Ported cell for cell; the cells that start the search
// of a plateau are taken block by block as pygeoprocessing iterates a raster's blocks,
// so the distances across plateaus are summed in the same order.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

#include "hydro/hydro.hpp"

namespace hydro {

namespace {

// pygeoprocessing's neighbours: 0 E, 1 NE, 2 N, 3 NW, 4 W, 5 SW, 6 S, 7 SE
constexpr int ROW[8] = {0, -1, -1, -1, 0, 1, 1, 1};
constexpr int COL[8] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int INFLOW[8] = {4, 5, 6, 7, 0, 1, 2, 3};
constexpr double SQRT2 = 1.4142135623730951;
constexpr double SQRT2_INV = 1.0 / 1.4142135623730951;

inline int pack(const double* w, double sum) {
    int out = 0;
    for (int k = 0; k < 8; ++k) out |= static_cast<int>(0.5 + w[k] / sum * 0xF) << (k * 4);
    return out;
}

}  // namespace

std::vector<std::int32_t> mfd_directions(const Grid& g, int block) {
    const i64 ny = g.ny, nx = g.nx, n = ny * nx;
    const double* z = g.z;
    auto inside = [&](i64 r, i64 c) { return r >= 0 && c >= 0 && r < ny && c < nx; };
    std::vector<std::int32_t> dir(static_cast<std::size_t>(n), 0);
    std::vector<std::uint8_t> flat(static_cast<std::size_t>(n), 0), drain_mask(static_cast<std::size_t>(n), 0);
    const double far = static_cast<double>(static_cast<unsigned long>(nx * ny));   // pygeoprocessing's nodata distance
    std::vector<double> distance(static_cast<std::size_t>(n), far);
    std::deque<i64> search, dist_q, nodata_q, direction_q;
    std::deque<int> nodata_dirs;
    if (block <= 0) block = static_cast<int>(std::max(ny, nx));
    for (i64 by = 0; by < ny; by += block)
        for (i64 bx = 0; bx < nx; bx += block)
            for (i64 r = by; r < std::min(ny, by + block); ++r)
                for (i64 c = bx; c < std::min(nx, bx + block); ++c) {
                    const i64 root = r * nx + c;
                    const double h = z[root];
                    if (std::isnan(h) || dir[root] != 0) continue;
                    // phase 1: the local downhill slopes
                    double w[8], sum = 0.0;
                    for (int k = 0; k < 8; ++k) {
                        w[k] = 0.0;
                        const i64 rn = r + ROW[k], cn = c + COL[k];
                        if (!inside(rn, cn)) continue;
                        const double hn = z[rn * nx + cn];
                        if (std::isnan(hn)) continue;
                        double s = h - hn;
                        if (s > 0.0) {
                            if (k & 1) s *= SQRT2_INV;
                            w[k] = s;
                            sum += s;
                        }
                    }
                    if (sum > 0.0) {
                        dir[root] = pack(w, sum);
                        continue;
                    }
                    // phase 2: the drains of the plateau, downhill ones first, else missing cells
                    search.push_back(root);
                    flat[root] = 1;
                    while (!search.empty()) {
                        const i64 q = search.front();
                        search.pop_front();
                        const i64 rq = q / nx, cq = q % nx;
                        double down[8], miss[8], sum_down = 0.0, sum_miss = 0.0;
                        for (int k = 0; k < 8; ++k) {
                            down[k] = miss[k] = 0.0;
                            const i64 rn = rq + ROW[k], cn = cq + COL[k];
                            const double hn = inside(rn, cn) ? z[rn * nx + cn] : std::numeric_limits<double>::quiet_NaN();
                            if (std::isnan(hn)) {
                                const double s = (k & 1) ? SQRT2_INV : 1.0;
                                sum_miss += s;
                                miss[k] = s;
                                continue;
                            }
                            double s = h - hn;
                            if (s < 0) continue;
                            if (s == 0.0) {
                                const i64 j = rn * nx + cn;
                                if (!flat[j]) {
                                    search.push_back(j);
                                    flat[j] = 1;
                                }
                                continue;
                            }
                            if (k & 1) s *= SQRT2_INV;
                            down[k] = s;
                            sum_down += down[k];
                        }
                        if (sum_down > 0.0) {
                            dir[q] = pack(down, sum_down);
                            distance[q] = 0.0;
                            drain_mask[q] = 1;
                            dist_q.push_back(q);
                        } else if (sum_miss > 0.0) {
                            nodata_q.push_back(q);
                            nodata_dirs.push_back(pack(miss, sum_miss));
                        }
                    }
                    if (dist_q.empty()) {
                        while (!nodata_q.empty()) {
                            const i64 q = nodata_q.front();
                            dir[q] = nodata_dirs.front();
                            distance[q] = 0.0;
                            drain_mask[q] = 1;
                            dist_q.push_back(q);
                            nodata_dirs.pop_front();
                            nodata_q.pop_front();
                        }
                    } else {
                        nodata_q.clear();
                        nodata_dirs.clear();
                    }
                    direction_q.assign(dist_q.begin(), dist_q.end());
                    // phase 3: the distance of each plateau cell to its nearest drain
                    while (!dist_q.empty()) {
                        const i64 q = dist_q.front();
                        dist_q.pop_front();
                        const i64 rq = q / nx, cq = q % nx;
                        const double d = distance[q];
                        for (int k = 0; k < 8; ++k) {
                            const i64 rn = rq + ROW[k], cn = cq + COL[k];
                            if (!inside(rn, cn)) continue;
                            const i64 j = rn * nx + cn;
                            const double dn = d + ((k & 1) ? SQRT2 : 1.0);
                            if (z[j] == h && distance[j] > dn) {
                                distance[j] = dn;
                                dist_q.push_back(j);
                            }
                        }
                    }
                    // phase 4: plateau cells flow to their neighbours nearer a drain
                    while (!direction_q.empty()) {
                        const i64 q = direction_q.front();
                        direction_q.pop_front();
                        const i64 rq = q / nx, cq = q % nx;
                        const double d = distance[q];
                        double wd[8], s = 0.0;
                        for (int k = 0; k < 8; ++k) {
                            wd[k] = 0.0;
                            const i64 rn = rq + ROW[k], cn = cq + COL[k];
                            if (!inside(rn, cn)) continue;
                            const i64 j = rn * nx + cn;
                            if (z[j] != h) continue;
                            const double dn = distance[j];
                            if (dn == far) continue;
                            if (dn < d) {
                                wd[k] = (k & 1) ? SQRT2_INV : 1.0;
                                s += wd[k];
                            } else if (!drain_mask[j]) {
                                direction_q.push_back(j);
                                drain_mask[j] = 1;
                            }
                        }
                        if (s == 0) continue;
                        dir[q] = pack(wd, s);
                    }
                }
    return dir;
}

std::vector<double> mfd_packed_accumulation(const std::int32_t* dir, const double* weight, int ny_, int nx_) {
    const i64 ny = ny_, nx = nx_, n = ny * nx;
    auto inside = [&](i64 r, i64 c) { return r >= 0 && c >= 0 && r < ny && c < nx; };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> acc(static_cast<std::size_t>(n), nan);
    std::vector<std::uint8_t> visited(static_cast<std::size_t>(n), 0);
    struct Item {
        i64 cell;
        int last;
        double value;
    };
    std::vector<Item> stack;
    auto weight_of = [&](i64 i) { return weight ? (std::isnan(weight[i]) ? 0.0 : weight[i]) : 1.0; };
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const i64 i = r * nx + c;
            const std::int32_t d = dir[i];
            if (d == 0) continue;
            // a drain: it flows (at least partly) into a cell without flow or off the map
            for (int k = 0; k < 8; ++k) {
                if (((d >> (k * 4)) & 0xF) == 0) continue;
                const i64 rn = r + ROW[k], cn = c + COL[k];
                if (!inside(rn, cn) || dir[rn * nx + cn] == 0) {
                    if (!visited[i]) {
                        stack.push_back({i, 0, weight_of(i)});
                        visited[i] = 1;
                    }
                    break;
                }
            }
            while (!stack.empty()) {
                Item it = stack.back();
                stack.pop_back();
                const i64 ri = it.cell / nx, ci = it.cell % nx;
                bool preempted = false;
                for (int k = it.last; k < 8; ++k) {
                    const i64 rn = ri + ROW[k], cn = ci + COL[k];
                    if (!inside(rn, cn)) continue;
                    const i64 j = rn * nx + cn;
                    const std::int32_t up = dir[j];
                    const int wup = (up >> (INFLOW[k] * 4)) & 0xF;
                    if (wup == 0) continue;
                    const double accum = acc[j];
                    if (std::isnan(accum) && !visited[j]) {
                        it.last = k;
                        stack.push_back(it);
                        stack.push_back({j, 0, weight_of(j)});
                        visited[j] = 1;
                        preempted = true;
                        break;
                    }
                    int total = 0;
                    for (int m = 0; m < 8; ++m) total += (up >> (m * 4)) & 0xF;
                    it.value += accum * wup / static_cast<double>(static_cast<float>(total));
                }
                if (!preempted) acc[it.cell] = it.value;
            }
        }
    return acc;
}

std::vector<double> mfd_distance_to_channel(const std::int32_t* dir, const std::uint8_t* channel,
                                            const double* weight, int ny_, int nx_) {
    const i64 ny = ny_, nx = nx_, n = ny * nx;
    auto inside = [&](i64 r, i64 c) { return r >= 0 && c >= 0 && r < ny && c < nx; };
    constexpr double NODATA = -1.0;
    std::vector<double> dist(static_cast<std::size_t>(n), NODATA);
    std::vector<std::uint8_t> visited(static_cast<std::size_t>(n), 0);
    struct Item {
        i64 cell;
        int last;
        double sum_weights, sum_distances;
    };
    std::vector<Item> stack;
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const i64 root = r * nx + c;
            if (channel[root] == 1) {
                dist[root] = 0.0;
                continue;
            }
            if (dir[root] == 0) continue;
            if (!visited[root]) {
                visited[root] = 1;
                stack.push_back({root, 0, 0.0, 0.0});
            }
            while (!stack.empty()) {
                Item p = stack.back();
                stack.pop_back();
                if (channel[p.cell] == 1) {
                    dist[p.cell] = 0.0;
                    continue;
                }
                const std::int32_t d = dir[p.cell];
                const i64 rp = p.cell / nx, cp = p.cell % nx;
                bool preempted = false;
                for (int k = p.last; k < 8; ++k) {
                    const int w = 0xF & (d >> (k * 4));
                    if (w == 0) continue;
                    const i64 rn = rp + ROW[k], cn = cp + COL[k];
                    if (!inside(rn, cn)) continue;
                    const i64 j = rn * nx + cn;
                    if (!visited[j]) {
                        visited[j] = 1;
                        preempted = true;
                        p.last = k;
                        stack.push_back(p);
                        stack.push_back({j, 0, 0.0, 0.0});
                        break;
                    }
                    const double dn = dist[j];
                    if (dn == NODATA) continue;
                    double wv;
                    if (weight) wv = std::isnan(weight[j]) ? 0.0 : weight[j];
                    else wv = (k % 2) ? SQRT2 : 1.0;
                    p.sum_weights += w;
                    p.sum_distances += w * (wv + dn);
                }
                if (preempted) continue;
                if (p.sum_weights != 0) dist[p.cell] = p.sum_distances / p.sum_weights;
            }
        }
    for (double& v : dist)
        if (v == NODATA) v = std::numeric_limits<double>::quiet_NaN();
    return dist;
}

std::vector<std::uint8_t> mfd_streams(const double* acc, const std::int32_t* dir, int ny_, int nx_, double threshold,
                                      double trace_proportion, int block) {
    const i64 ny = ny_, nx = nx_, n = ny * nx;
    auto inside = [&](i64 r, i64 c) { return r >= 0 && c >= 0 && r < ny && c < nx; };
    constexpr std::uint8_t NONE = 255;
    std::vector<std::uint8_t> stream(static_cast<std::size_t>(n), NONE);
    const double trace = trace_proportion * threshold;
    std::deque<i64> open, back;
    if (block <= 0) block = static_cast<int>(std::max(ny, nx));
    for (i64 by = 0; by < ny; by += block)
        for (i64 bx = 0; bx < nx; bx += block)
            for (i64 r = by; r < std::min(ny, by + block); ++r)
                for (i64 c = bx; c < std::min(nx, bx + block); ++c) {
                    const i64 root = r * nx + c;
                    if (std::isnan(acc[root]) || stream[root] != NONE) continue;
                    stream[root] = 0;
                    if (acc[root] < threshold) continue;
                    const std::int32_t d = dir[root];
                    bool outlet = false;
                    for (int k = 0; k < 8; ++k) {
                        if (((d >> (k * 4)) & 0xF) == 0) continue;
                        const i64 rn = r + ROW[k], cn = c + COL[k];
                        if (!inside(rn, cn) || std::isnan(acc[rn * nx + cn])) {
                            outlet = true;
                            break;
                        }
                    }
                    if (outlet) {
                        open.push_back(root);
                        stream[root] = 1;
                    }
                    while (!open.empty()) {
                        const i64 q = open.front();
                        open.pop_front();
                        const i64 rq = q / nx, cq = q % nx;
                        for (int k = 0; k < 8; ++k) {
                            const i64 rs = rq + ROW[k], cs = cq + COL[k];
                            if (!inside(rs, cs)) continue;
                            const i64 j = rs * nx + cs;
                            const std::int32_t up = dir[j];
                            if (up == 0) continue;
                            if (((up >> (INFLOW[k] * 4)) & 0xF) == 0) continue;
                            if (stream[j] == 1 || stream[j] == 2) continue;
                            if (acc[j] >= threshold) {
                                stream[j] = 1;
                                open.push_back(j);
                                back.push_back(j);
                                while (!back.empty()) {   // confirm the tentative cells downstream
                                    const i64 b = back.front();
                                    back.pop_front();
                                    const std::int32_t db = dir[b];
                                    const i64 rb = b / nx, cb = b % nx;
                                    for (int m = 0; m < 8; ++m) {
                                        if (((db >> (m * 4)) & 0xF) == 0) continue;
                                        const i64 rm = rb + ROW[m], cm = cb + COL[m];
                                        if (!inside(rm, cm)) continue;
                                        if (stream[rm * nx + cm] == 2) {
                                            stream[rm * nx + cm] = 1;
                                            back.push_back(rm * nx + cm);
                                        }
                                    }
                                }
                            } else if (acc[j] >= trace) {
                                stream[j] = 2;
                                open.push_back(j);
                            }
                        }
                    }
                }
    for (auto& v : stream)
        if (v == 2) v = 0;
    return stream;
}

// ------------------------------------------------------------------------------- pygeoprocessing's D8

std::vector<std::uint8_t> pgp_d8_directions(const Grid& g, int block) {
    const i64 ny = g.ny, nx = g.nx, n = ny * nx;
    const double* z = g.z;
    auto inside = [&](i64 r, i64 c) { return r >= 0 && c >= 0 && r < ny && c < nx; };
    constexpr std::uint8_t NONE = 128;
    std::vector<std::uint8_t> dir(static_cast<std::size_t>(n), NONE);
    std::vector<std::uint8_t> flat(static_cast<std::size_t>(n), 0);
    const double far = static_cast<double>(static_cast<unsigned long>(nx * ny));
    std::vector<double> distance(static_cast<std::size_t>(n), far);
    std::deque<i64> search, drains, nodata_drains;
    std::deque<int> nodata_dirs;
    if (block <= 0) block = static_cast<int>(std::max(ny, nx));
    for (i64 by = 0; by < ny; by += block)
        for (i64 bx = 0; bx < nx; bx += block)
            for (i64 r = by; r < std::min(ny, by + block); ++r)
                for (i64 c = bx; c < std::min(nx, bx + block); ++c) {
                    const i64 root = r * nx + c;
                    const double h = z[root];
                    if (std::isnan(h) || dir[root] != NONE) continue;
                    int best = -1;
                    double largest = 0.0;
                    for (int k = 0; k < 8; ++k) {
                        const i64 rn = r + ROW[k], cn = c + COL[k];
                        if (!inside(rn, cn)) continue;
                        const double hn = z[rn * nx + cn];
                        if (std::isnan(hn)) continue;
                        double s = h - hn;
                        if (k & 1) s *= SQRT2_INV;
                        if (s > largest) {
                            best = k;
                            largest = s;
                        }
                    }
                    if (best >= 0) {
                        dir[root] = static_cast<std::uint8_t>(best);
                        continue;
                    }
                    search.push_back(root);
                    flat[root] = 1;
                    while (!search.empty()) {
                        const i64 q = search.front();
                        search.pop_front();
                        const i64 rq = q / nx, cq = q % nx;
                        int qbest = -1;
                        double qlargest = 0.0;
                        int diagonal_nodata = 1;
                        for (int k = 0; k < 8; ++k) {
                            const i64 rn = rq + ROW[k], cn = cq + COL[k];
                            const double hn = inside(rn, cn) ? z[rn * nx + cn] : std::numeric_limits<double>::quiet_NaN();
                            if (std::isnan(hn)) {
                                if (diagonal_nodata && qlargest == 0.0) {
                                    qbest = k;
                                    diagonal_nodata = k & 1;
                                }
                                continue;
                            }
                            double s = h - hn;
                            if (s < 0) continue;
                            if (s == 0.0) {
                                const i64 j = rn * nx + cn;
                                if (!flat[j]) {
                                    search.push_back(j);
                                    flat[j] = 1;
                                }
                                continue;
                            }
                            if (k & 1) s *= SQRT2_INV;
                            if (s > qlargest) {
                                qlargest = s;
                                qbest = k;
                            }
                        }
                        if (qbest >= 0) {
                            if (qlargest > 0.0) {
                                dir[q] = static_cast<std::uint8_t>(qbest);
                                distance[q] = 0.0;
                                drains.push_back(q);
                            } else {
                                nodata_drains.push_back(q);
                                nodata_dirs.push_back(qbest);
                            }
                        }
                    }
                    if (drains.empty()) {
                        while (!nodata_drains.empty()) {
                            const i64 q = nodata_drains.front();
                            dir[q] = static_cast<std::uint8_t>(nodata_dirs.front());
                            distance[q] = 0.0;
                            drains.push_back(q);
                            nodata_dirs.pop_front();
                            nodata_drains.pop_front();
                        }
                    } else {
                        nodata_drains.clear();
                        nodata_dirs.clear();
                    }
                    while (!drains.empty()) {
                        const i64 q = drains.front();
                        drains.pop_front();
                        const i64 rq = q / nx, cq = q % nx;
                        const double d = distance[q];
                        for (int k = 0; k < 8; ++k) {
                            const i64 rn = rq + ROW[k], cn = cq + COL[k];
                            if (!inside(rn, cn)) continue;
                            const i64 j = rn * nx + cn;
                            const double dn = d + ((k & 1) ? SQRT2 : 1.0);
                            if (z[j] == h && distance[j] > dn) {
                                dir[j] = static_cast<std::uint8_t>(INFLOW[k]);
                                distance[j] = dn;
                                drains.push_back(j);
                            }
                        }
                    }
                }
    return dir;
}

std::vector<double> pgp_d8_accumulation(const std::uint8_t* dir, const double* weight, int ny_, int nx_) {
    const i64 ny = ny_, nx = nx_, n = ny * nx;
    auto inside = [&](i64 r, i64 c) { return r >= 0 && c >= 0 && r < ny && c < nx; };
    constexpr std::uint8_t NONE = 128;
    std::vector<double> acc(static_cast<std::size_t>(n), std::numeric_limits<double>::quiet_NaN());
    struct Item {
        i64 cell;
        int last;
        double value;
    };
    std::vector<Item> stack;
    auto weight_of = [&](i64 i) { return weight ? (std::isnan(weight[i]) ? 0.0 : weight[i]) : 1.0; };
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const i64 i = r * nx + c;
            const int d = dir[i];
            if (d == NONE) continue;
            const i64 rn = r + ROW[d], cn = c + COL[d];
            if (!inside(rn, cn) || dir[rn * nx + cn] == NONE) stack.push_back({i, 0, weight_of(i)});
            while (!stack.empty()) {
                Item it = stack.back();
                stack.pop_back();
                const i64 ri = it.cell / nx, ci = it.cell % nx;
                bool remain = false;
                for (int k = it.last; k < 8; ++k) {
                    const i64 ru = ri + ROW[k], cu = ci + COL[k];
                    if (!inside(ru, cu)) continue;
                    const i64 j = ru * nx + cu;
                    const int up = dir[j];
                    if (up == NONE || up != INFLOW[k]) continue;
                    if (std::isnan(acc[j])) {
                        it.last = k;
                        stack.push_back(it);
                        stack.push_back({j, 0, weight_of(j)});
                        remain = true;
                        break;
                    }
                    it.value += acc[j];
                }
                if (!remain) acc[it.cell] = it.value;
            }
        }
    return acc;
}

std::vector<double> pgp_d8_distance_to_channel(const std::uint8_t* dir, const std::uint8_t* channel,
                                               const double* weight, int ny_, int nx_) {
    const i64 ny = ny_, nx = nx_, n = ny * nx;
    auto inside = [&](i64 r, i64 c) { return r >= 0 && c >= 0 && r < ny && c < nx; };
    std::vector<double> dist(static_cast<std::size_t>(n), std::numeric_limits<double>::quiet_NaN());
    std::vector<std::pair<i64, double>> stack;
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            if (channel[r * nx + c] != 1) continue;
            stack.emplace_back(r * nx + c, 0.0);
            while (!stack.empty()) {
                const auto [q, value] = stack.back();
                stack.pop_back();
                dist[q] = value;
                const i64 rq = q / nx, cq = q % nx;
                for (int k = 0; k < 8; ++k) {
                    const i64 rn = rq + ROW[k], cn = cq + COL[k];
                    if (!inside(rn, cn)) continue;
                    const i64 j = rn * nx + cn;
                    if (channel[j] == 1) continue;
                    if (dir[j] == INFLOW[k]) {
                        double w;
                        if (weight) w = std::isnan(weight[j]) ? 0.0 : weight[j];
                        else w = (k % 2) ? SQRT2 : 1.0;
                        stack.emplace_back(j, w + value);
                    }
                }
            }
        }
    return dist;
}

}  // namespace hydro
