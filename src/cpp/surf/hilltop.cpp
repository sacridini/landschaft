// SPDX-License-Identifier: GPL-3.0-or-later
//
// Hilltops as permanent protection areas (APP de topo de morro, Brazil's Forest
// Code, Lei 12.651/2012 art. 4 IX): every summit of a DEM with its base, height
// and slope, and the upper part of the hills that qualify.
//
// The DEM's cells are swept from the highest down (ties: row order), joining
// each cell to the regions of its 8 neighbours already swept (a union-find).
// A cell with no swept neighbour starts the region of a new summit. Where
// regions meet, the cell is a saddle: the regions of lower summits end there
// (their base is the saddle's elevation: the key col of topographic
// prominence) and join the region of the highest. A region that reaches a
// cell of the plain mask (plains, water bodies) gets that cell's elevation as
// its base (the plain or water adjacent to the elevation); the region of the
// highest summit of a connected area that meets neither gets its lowest
// elevation (its floor, whose cell nearest the summit is the base cell). Each
// summit's height is its elevation minus its base; its slope the mean slope of
// the cells of its region above the base (joined to the summit above it: flat
// ground at the base level and regions meeting only there left out), or the
// angle of the line from the summit to its foot, the nearest cell at or below
// the base.
//
// The upper part of a qualifying hill is the region, at the sweep's level
// base + fraction x height, that holds its summit: the cells joined to the
// summit through cells at least that high. The joins are kept as a forest with
// the cell whose sweep made each link (union by size: depth O(log n)), so a
// cell belongs to that region when its path up the forest reaches the
// region's root through links made at or above that level (their levels only
// fall along the path).
//
// Speed: the sweep's order by a radix sort of the elevations' bits; 32-bit
// indices when they fit; a bitset of the swept cells (small enough for the
// cache) so that unswept neighbours cost no random access; each region's cells
// above its floor counted during the sweep.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "surf/surface.hpp"

namespace surf {

namespace {

constexpr int kDr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
constexpr int kDc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
constexpr double kDeg = 180.0 / 3.14159265358979323846;

struct Stats {
    double sum = 0;  // slopes (finite ones) and their count; cells
    i64 count = 0;
    i64 cells = 0;
    void add(double s) {
        if (std::isfinite(s)) {
            sum += s;
            ++count;
        }
        ++cells;
    }
    void add(const Stats& o) {
        sum += o.sum;
        count += o.count;
        cells += o.cells;
    }
};

struct Region {      // the data of a region, kept at its root's slot
    i64 summit = -1;
    i64 hill = -1;
    Stats all;       // its cells
    Stats above;     // its cells above its floor joined to its summit through cells above the floor
    double floor_z = 0;  // its lowest elevation
};

// The cells of finite elevation from the highest, ties in row order: a stable LSD radix sort (11-bit
// digits) of keys that order like the elevations, descending.
template <typename I>
std::vector<I> descending(const double* z, i64 n) {
    std::vector<std::uint64_t> key;
    std::vector<I> idx;
    key.reserve(static_cast<std::size_t>(n));
    idx.reserve(static_cast<std::size_t>(n));
    for (i64 c = 0; c < n; ++c) {
        const double v = z[c];
        if (!std::isfinite(v)) continue;
        std::uint64_t b;
        std::memcpy(&b, &v, sizeof b);
        b = (b >> 63) ? ~b : (b | 0x8000000000000000ull);  // ascending order of the values
        if (v == 0) b = 0x8000000000000000ull;               // -0 == +0
        key.push_back(~b);                                   // descending
        idx.push_back(static_cast<I>(c));
    }
    const std::size_t m = key.size();
    std::vector<std::uint64_t> key2(m);
    std::vector<I> idx2(m);
    constexpr int kBits = 11;
    constexpr std::size_t kBuckets = std::size_t{1} << kBits;
    std::vector<std::size_t> count(kBuckets);
    for (int shift = 0; shift < 64; shift += kBits) {
        std::fill(count.begin(), count.end(), 0);
        for (std::size_t i = 0; i < m; ++i) ++count[(key[i] >> shift) & (kBuckets - 1)];
        if (m == 0 || count[(key[0] >> shift) & (kBuckets - 1)] == m) continue;  // one bucket: no move
        std::size_t sum = 0;
        for (auto& c : count) {
            const std::size_t t = c;
            c = sum;
            sum += t;
        }
        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t d = count[(key[i] >> shift) & (kBuckets - 1)]++;
            key2[d] = key[i];
            idx2[d] = idx[i];
        }
        key.swap(key2);
        idx.swap(idx2);
    }
    return idx;
}

template <typename I>
Hilltops sweep(const double* z, const double* slope, const std::uint8_t* plain, int ny, int nx, double dx, double dy,
               double min_height, double min_slope, bool line_slope, double fraction) {
    const i64 n = static_cast<i64>(ny) * nx;
    const std::vector<I> order = descending<I>(z, n);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto N = static_cast<std::size_t>(n);
    std::vector<std::uint64_t> swept((N + 63) / 64, 0);
    std::vector<I> uf(N);      // compressed: the current root
    std::vector<I> parent(N);  // the forest of joins
    std::vector<I> link(N);    // the cell whose sweep linked each node to its parent (its level: z[link])
    std::vector<I> size(N);
    std::vector<std::int32_t> slot(N, -1);  // a root's region (summits only)
    std::vector<Region> regions;
    auto R = [&](I x) -> Region& { return regions[static_cast<std::size_t>(slot[static_cast<std::size_t>(x)])]; };
    auto find = [&](I c) {
        I r = c;
        while (uf[r] != r) r = uf[r];
        while (uf[c] != r) {
            const I next = uf[c];
            uf[c] = r;
            c = next;
        }
        return r;
    };

    Hilltops out;
    auto& hills = out.hills;
    auto record = [&](const Region& g, const Stats& st, double base, i64 base_cell, int kind) {
        Hill& h = hills[static_cast<std::size_t>(g.hill)];
        h.base = base;
        h.base_cell = base_cell;
        h.kind = kind;
        h.height = h.top - base;
        h.mean_slope = st.count > 0 ? st.sum / static_cast<double>(st.count) : nan;
        h.cells = st.cells;
        h.fixed = true;
    };
    // a region's cells above a base at `level` (its floor or lower): those above its floor when that is the
    // level (flat ground at the base left out), else all
    auto upto = [](const Region& g, double level) { return g.floor_z == level ? g.above : g.all; };
    auto fixed = [&](const Region& g) { return hills[static_cast<std::size_t>(g.hill)].fixed; };
    auto higher = [&](i64 a, i64 b) { return z[a] > z[b] || (z[a] == z[b] && a < b); };  // summits

    I roots[8];
    for (const I c : order) {
        const i64 r = static_cast<i64>(c) / nx, k = static_cast<i64>(c) % nx;
        int nr = 0;
        for (int w = 0; w < 8; ++w) {
            const i64 rr = r + kDr[w], kk = k + kDc[w];
            if (rr < 0 || rr >= ny || kk < 0 || kk >= nx) continue;
            const auto nb = static_cast<std::size_t>(rr * nx + kk);
            if (!((swept[nb >> 6] >> (nb & 63)) & 1u)) continue;
            const I root = find(static_cast<I>(nb));
            bool seen = false;
            for (int q = 0; q < nr; ++q) seen = seen || roots[q] == root;
            if (!seen) roots[nr++] = root;
        }
        const auto cs = static_cast<std::size_t>(c);
        swept[cs >> 6] |= std::uint64_t{1} << (cs & 63);
        uf[cs] = c;
        parent[cs] = c;
        link[cs] = c;
        size[cs] = 1;
        const bool is_plain = plain != nullptr && plain[cs] != 0;
        if (nr == 0) {  // a new summit
            slot[cs] = static_cast<std::int32_t>(regions.size());
            regions.emplace_back();
            Region& g = regions.back();
            g.summit = c;
            g.hill = static_cast<i64>(hills.size());
            Hill h;
            h.summit = c;
            h.top = z[cs];
            hills.push_back(h);
            g.all.add(slope[cs]);
            g.floor_z = z[cs];
            if (is_plain) record(g, Stats{}, z[cs], c, 1);  // a summit on the plain: height 0
            continue;
        }
        // the region of the highest summit goes on; the others end at this saddle
        I win = roots[0];
        for (int q = 1; q < nr; ++q)
            if (higher(R(roots[q]).summit, R(win).summit)) win = roots[q];
        for (int q = 0; q < nr; ++q)
            if (roots[q] != win && !fixed(R(roots[q]))) record(R(roots[q]), upto(R(roots[q]), z[cs]), z[cs], c, 0);
        if (is_plain && !fixed(R(win)) && z[cs] < z[R(win).summit]) record(R(win), upto(R(win), z[cs]), z[cs], c, 1);
        // the merged data: all cells; above the new floor (this cell's elevation: no swept cell is lower), the
        // winner's cells joined above it (regions joined here meet only at the floor)
        Region merged = R(win);
        if (merged.floor_z != z[cs]) merged.above = merged.all;
        for (int q = 0; q < nr; ++q)
            if (roots[q] != win) merged.all.add(R(roots[q]).all);
        merged.all.add(slope[cs]);
        merged.floor_z = z[cs];
        // join: the largest tree is the root (the forest's depth stays O(log n)); it takes the winner's data
        I top = roots[0];
        for (int q = 1; q < nr; ++q)
            if (size[roots[q]] > size[top]) top = roots[q];
        for (int q = 0; q < nr; ++q) {
            const I x = roots[q];
            if (x == top) continue;
            parent[x] = top;
            link[x] = c;
            uf[x] = top;
            size[top] += size[x];
        }
        parent[cs] = top;
        uf[cs] = top;
        size[top] += 1;
        R(win) = merged;
        slot[top] = slot[win];
    }

    // the highest summits of each connected area: based at its floor, their slope and area from the cells above
    // it; the base cell is the floor's cell nearest the summit
    std::vector<I> last;
    double highest_floor = -std::numeric_limits<double>::infinity();
    for (const I c : order) {
        if (parent[c] != c || fixed(R(c))) continue;
        last.push_back(c);
        highest_floor = std::max(highest_floor, R(c).floor_z);
    }
    if (!last.empty()) {
        std::vector<i64> base_cell(last.size(), -1);
        std::vector<double> near(last.size(), std::numeric_limits<double>::infinity());
        std::vector<std::int32_t> which(regions.size(), -1);
        for (std::size_t q = 0; q < last.size(); ++q) which[static_cast<std::size_t>(slot[last[q]])] = static_cast<std::int32_t>(q);
        for (auto it = order.rbegin(); it != order.rend() && z[*it] <= highest_floor; ++it) {
            const I c = *it;
            const I root = find(c);
            const std::int32_t q = which[static_cast<std::size_t>(slot[root])];
            if (q < 0 || z[c] != R(root).floor_z) continue;
            const i64 sm = R(root).summit;
            const double ry = static_cast<double>(static_cast<i64>(c) / nx - sm / nx) * dy;
            const double rx = static_cast<double>(static_cast<i64>(c) % nx - sm % nx) * dx;
            const double d2 = rx * rx + ry * ry;
            if (d2 < near[static_cast<std::size_t>(q)] ||
                (d2 == near[static_cast<std::size_t>(q)] && c < base_cell[static_cast<std::size_t>(q)])) {
                near[static_cast<std::size_t>(q)] = d2;
                base_cell[static_cast<std::size_t>(q)] = c;
            }
        }
        for (std::size_t q = 0; q < last.size(); ++q) {
            const Region& g = R(last[q]);
            record(g, g.above, g.floor_z, base_cell[q], 2);
        }
    }

    // the slope of the line from each summit to its foot: the nearest cell at or below its base (rings of cells
    // around the summit until no nearer one can be found)
    const i64 nh = static_cast<i64>(hills.size());
#pragma omp parallel for schedule(dynamic, 16) if (nh > 64)
    for (i64 q = 0; q < nh; ++q) {
        Hill& h = hills[static_cast<std::size_t>(q)];
        if (!(h.height > 0)) {
            h.line_slope = 0;
            continue;
        }
        const i64 r0 = h.summit / nx, c0 = h.summit % nx;
        const double step = std::min(dx, dy);
        double best = std::numeric_limits<double>::infinity();
        const i64 kmax = std::max<i64>(ny, nx);
        for (i64 k = 1; k <= kmax && static_cast<double>(k) * step < best; ++k) {
            for (i64 r = r0 - k; r <= r0 + k; ++r) {
                if (r < 0 || r >= ny) continue;
                const bool edge = r == r0 - k || r == r0 + k;
                for (i64 c = c0 - k; c <= c0 + k; c += edge ? 1 : 2 * k) {
                    if (c < 0 || c >= nx) continue;
                    const double v = z[r * nx + c];
                    if (!(v <= h.base)) continue;
                    const double ry = static_cast<double>(r - r0) * dy, rx = static_cast<double>(c - c0) * dx;
                    best = std::min(best, std::sqrt(rx * rx + ry * ry));
                }
            }
        }
        h.line_slope = std::isfinite(best) ? std::atan2(h.height, best) * kDeg : 90.0;
    }

    // the upper parts: for each qualifying hill, the root of its summit's region at its threshold
    std::vector<std::pair<I, std::int32_t>> marks;  // (node, hill)
    for (std::size_t h = 0; h < hills.size(); ++h) {
        Hill& x = hills[h];
        const double s = line_slope ? x.line_slope : x.mean_slope;
        x.qualifies = x.height >= min_height && s > min_slope;
        if (!x.qualifies) continue;
        x.threshold = x.base + fraction * x.height;
        auto node = static_cast<I>(x.summit);
        while (parent[node] != node && z[link[node]] >= x.threshold) node = parent[node];
        marks.emplace_back(node, static_cast<std::int32_t>(h));
    }
    // grouped by node, the highest threshold first
    std::sort(marks.begin(), marks.end(), [&](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        const double ta = hills[static_cast<std::size_t>(a.second)].threshold;
        const double tb = hills[static_cast<std::size_t>(b.second)].threshold;
        return ta > tb || (ta == tb && a.second < b.second);
    });
    std::vector<std::int32_t> first(N, -1);
    for (std::size_t m = marks.size(); m-- > 0;) first[static_cast<std::size_t>(marks[m].first)] = static_cast<std::int32_t>(m);
    out.label.assign(N, -1);
    if (marks.empty()) return out;
    // a cell is in a hill's upper part when its path up the forest reaches the hill's node through links made at
    // or above the threshold; its label: the satisfied hill of the highest threshold (the innermost)
#pragma omp parallel for schedule(static) if (n > 65536)
    for (i64 c = 0; c < n; ++c) {
        if (!((swept[static_cast<std::size_t>(c) >> 6] >> (c & 63)) & 1u)) continue;
        auto node = static_cast<I>(c);
        double in = std::numeric_limits<double>::infinity();  // the lowest link level on the way into `node`
        std::int32_t label = -1;
        double best = -std::numeric_limits<double>::infinity();
        for (;;) {
            for (std::int32_t m = first[static_cast<std::size_t>(node)];
                 m >= 0 && static_cast<std::size_t>(m) < marks.size() && marks[static_cast<std::size_t>(m)].first == node;
                 ++m) {
                const std::int32_t h = marks[static_cast<std::size_t>(m)].second;
                const double t = hills[static_cast<std::size_t>(h)].threshold;
                if (in >= t) {
                    if (t > best) {
                        best = t;
                        label = h;
                    }
                    break;  // the others here have lower thresholds
                }
            }
            if (parent[node] == node) break;
            in = std::min(in, z[link[node]]);
            node = parent[node];
        }
        out.label[static_cast<std::size_t>(c)] = label;
    }
    return out;
}

}  // namespace

Hilltops hilltops(const double* z, const double* slope, const std::uint8_t* plain, int ny, int nx, double dx,
                  double dy, double min_height, double min_slope, bool line_slope, double fraction) {
    if (!(fraction >= 0 && fraction <= 1)) throw std::invalid_argument("the fraction must be between 0 and 1");
    const i64 n = static_cast<i64>(ny) * nx;
    if (n < std::numeric_limits<std::int32_t>::max())
        return sweep<std::int32_t>(z, slope, plain, ny, nx, dx, dy, min_height, min_slope, line_slope, fraction);
    return sweep<i64>(z, slope, plain, ny, nx, dx, dy, min_height, min_slope, line_slope, fraction);
}

}  // namespace surf
