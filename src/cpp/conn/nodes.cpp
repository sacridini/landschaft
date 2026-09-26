// SPDX-License-Identifier: GPL-3.0-or-later
#include "conn/nodes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "lsm/extended.hpp"
#include "lsm/parallel.hpp"

namespace conn {

namespace {

inline bool valid(const std::uint8_t* na, i64 c) { return na == nullptr || na[c] == 0; }

// Pair lists built per first node in parallel, then concatenated in order.
PairList concatenate(std::vector<PairList>& parts) {
    PairList out;
    std::size_t total = 0;
    for (const PairList& p : parts) total += p.i.size();
    out.i.reserve(total);
    out.j.reserve(total);
    out.distance.reserve(total);
    for (PairList& p : parts) {
        out.i.insert(out.i.end(), p.i.begin(), p.i.end());
        out.j.insert(out.j.end(), p.j.begin(), p.j.end());
        out.distance.insert(out.distance.end(), p.distance.begin(), p.distance.end());
        p = PairList();
    }
    return out;
}

// Boundary cells of one patch by row: rows[k] holds cols[row_start[k] .. row_start[k + 1]).
struct Rows {
    std::vector<std::int32_t> rows, row_start, cols;
    std::int32_t rmin = 0, rmax = -1, cmin = 0, cmax = -1;
};

// Smallest squared gap (in map units) between two patches, searched from
// the cells of `a` into the rows of `b`; stops at `floor2` (the bounding
// boxes' gap: nothing can be closer) and ignores anything beyond `limit2`.
double min_gap2(const Rows& a, const Rows& b, double resx, double resy, double floor2, double limit2) {
    double best2 = limit2;
    const auto gap = [](std::int32_t d) { return static_cast<double>(std::max<std::int32_t>(0, std::abs(d) - 1)); };
    const std::size_t nb = b.rows.size();
    for (std::size_t ra = 0; ra < a.rows.size(); ++ra) {
        const std::int32_t r = a.rows[ra];
        // Rows of b nearest to r first, in both directions.
        const std::size_t mid = static_cast<std::size_t>(std::lower_bound(b.rows.begin(), b.rows.end(), r) - b.rows.begin());
        for (std::int32_t k = a.row_start[ra]; k < a.row_start[ra + 1]; ++k) {
            const std::int32_t c = a.cols[k];
            auto visit = [&](std::size_t rb) -> bool {
                const double dy = gap(b.rows[rb] - r) * resy;
                const double dy2 = dy * dy;
                if (dy2 > best2) return false;
                const auto first = b.cols.begin() + b.row_start[rb], last = b.cols.begin() + b.row_start[rb + 1];
                const auto it = std::lower_bound(first, last, c);
                if (it != last) {
                    const double dx = gap(*it - c) * resx;
                    best2 = std::min(best2, dx * dx + dy2);
                }
                if (it != first) {
                    const double dx = gap(*(it - 1) - c) * resx;
                    best2 = std::min(best2, dx * dx + dy2);
                }
                return true;
            };
            for (std::size_t rb = mid; rb < nb; ++rb) {
                if (!visit(rb)) break;
            }
            for (std::size_t rb = mid; rb-- > 0;) {
                if (!visit(rb)) break;
            }
            if (best2 <= floor2) return best2;
        }
    }
    return best2;
}

}  // namespace

RasterNodes raster_nodes(const std::int32_t* values, const std::uint8_t* na, const RasterGrid& grid, bool centroids,
                         bool keep_node_index) {
    const i64 nx = grid.nx, ny = grid.ny, ncell = nx * ny;
    RasterNodes out;
    // Distinct ids: a direct table over the value range when it is small,
    // else a sorted copy.
    std::int64_t lo = std::numeric_limits<std::int64_t>::max(), hi = std::numeric_limits<std::int64_t>::min();
    for (i64 c = 0; c < ncell; ++c) {
        if (!valid(na, c)) continue;
        lo = std::min<std::int64_t>(lo, values[c]);
        hi = std::max<std::int64_t>(hi, values[c]);
    }
    if (lo > hi) return out;
    const std::int64_t range = hi - lo + 1;
    std::vector<std::int32_t> table;
    const bool direct = range <= 4 * ncell + (1 << 20);
    if (direct) {
        table.assign(static_cast<std::size_t>(range), -1);
        for (i64 c = 0; c < ncell; ++c) {
            if (valid(na, c)) table[static_cast<std::size_t>(values[c] - lo)] = 0;
        }
        for (std::int64_t v = 0; v < range; ++v) {
            if (table[static_cast<std::size_t>(v)] == 0) {
                table[static_cast<std::size_t>(v)] = static_cast<std::int32_t>(out.ids.size());
                out.ids.push_back(static_cast<std::int32_t>(v + lo));
            }
        }
    } else {
        for (i64 c = 0; c < ncell; ++c) {
            if (valid(na, c)) out.ids.push_back(values[c]);
        }
        std::sort(out.ids.begin(), out.ids.end());
        out.ids.erase(std::unique(out.ids.begin(), out.ids.end()), out.ids.end());
    }
    const auto n = static_cast<std::int32_t>(out.ids.size());
    auto index_of = [&](std::int32_t v) -> std::int32_t {
        if (direct) return table[static_cast<std::size_t>(v - lo)];
        return static_cast<std::int32_t>(std::lower_bound(out.ids.begin(), out.ids.end(), v) - out.ids.begin());
    };
    std::vector<std::int32_t> node(static_cast<std::size_t>(ncell), -1);
    lsm::parallel_for(ncell > (1 << 16), ny, [&](i64 r) {
        for (i64 c = r * nx; c < (r + 1) * nx; ++c) {
            if (valid(na, c)) node[c] = index_of(values[c]);
        }
    });
    out.cells.assign(n, 0);
    for (i64 c = 0; c < ncell; ++c) {
        if (node[c] >= 0) ++out.cells[node[c]];
    }
    if (centroids) {
        // raster::xyFromCell: (col + 0.5) * xres + xmin, ymax - (row + 0.5) * yres,
        // averaged per node in cell order as R's colMeans does (80-bit sums).
        const double xres = (grid.xmax - grid.xmin) / static_cast<double>(nx);
        const double yres = (grid.ymax - grid.ymin) / static_cast<double>(ny);
        std::vector<i64> start(static_cast<std::size_t>(n) + 1, 0);
        for (std::int32_t v = 0; v < n; ++v) start[v + 1] = start[v] + out.cells[v];
        std::vector<i64> fill(start.begin(), start.end() - 1), cells(static_cast<std::size_t>(start[n]));
        for (i64 c = 0; c < ncell; ++c) {
            if (node[c] >= 0) cells[fill[node[c]]++] = c;
        }
        out.cx.resize(n);
        out.cy.resize(n);
        lsm::parallel_for_dynamic(n > 64, n, [&](i64 v) {
            lsm::Extended sx = 0.0, sy = 0.0;
            for (i64 k = start[v]; k < start[v + 1]; ++k) {
                const double row = static_cast<double>(cells[k] / nx);
                const double col = static_cast<double>(cells[k] % nx);
                sx += (col + 0.5) * xres + grid.xmin;
                sy += grid.ymax - (row + 0.5) * yres;
            }
            const lsm::Extended count(static_cast<std::int64_t>(out.cells[v]));
            out.cx[v] = static_cast<double>(sx / count);
            out.cy[v] = static_cast<double>(sy / count);
        });
    }
    if (keep_node_index) out.node = std::move(node);
    return out;
}

PairList point_distances(const double* x, const double* y, std::int32_t n, double max_distance) {
    const bool limited = std::isfinite(max_distance);
    // With a limit, nodes sorted by x are swept within the limit.
    std::vector<std::int32_t> by_x(n);
    for (std::int32_t v = 0; v < n; ++v) by_x[v] = v;
    std::vector<std::int32_t> rank(n);
    if (limited) {
        std::stable_sort(by_x.begin(), by_x.end(), [&](std::int32_t a, std::int32_t b) { return x[a] < x[b]; });
        for (std::int32_t k = 0; k < n; ++k) rank[by_x[k]] = k;
    }
    std::vector<PairList> parts(n);
    lsm::parallel_for_dynamic(n > 256, n, [&](i64 iv) {
        const auto i = static_cast<std::int32_t>(iv);
        PairList& p = parts[i];
        auto add = [&](std::int32_t j) {
            const double dx = x[i] - x[j], dy = y[i] - y[j];
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d <= max_distance) {
                p.i.push_back(i);
                p.j.push_back(j);
                p.distance.push_back(d);
            }
        };
        if (!limited) {
            for (std::int32_t j = i + 1; j < n; ++j) add(j);
            return;
        }
        std::vector<std::int32_t> found;
        for (std::int32_t k = rank[i] + 1; k < n && x[by_x[k]] - x[i] <= max_distance; ++k) {
            if (by_x[k] > i) found.push_back(by_x[k]);
        }
        for (std::int32_t k = rank[i]; k-- > 0 && x[i] - x[by_x[k]] <= max_distance;) {
            if (by_x[k] > i) found.push_back(by_x[k]);
        }
        std::sort(found.begin(), found.end());
        for (const std::int32_t j : found) add(j);
    });
    return concatenate(parts);
}

PairList raster_edge_distances(const std::int32_t* node, std::int32_t n_nodes, i64 nx, i64 ny, double resx, double resy,
                               double max_distance) {
    // Boundary cells (a 4-neighbour outside the raster or in another node) by
    // node, in row-major order.
    std::vector<Rows> rows(n_nodes);
    auto boundary = [&](i64 r, i64 c) {
        const std::int32_t v = node[r * nx + c];
        return r == 0 || c == 0 || r == ny - 1 || c == nx - 1 || node[(r - 1) * nx + c] != v ||
               node[(r + 1) * nx + c] != v || node[r * nx + c - 1] != v || node[r * nx + c + 1] != v;
    };
    for (i64 r = 0; r < ny; ++r) {
        for (i64 c = 0; c < nx; ++c) {
            const std::int32_t v = node[r * nx + c];
            if (v < 0 || !boundary(r, c)) continue;
            Rows& b = rows[v];
            const auto rr = static_cast<std::int32_t>(r), cc = static_cast<std::int32_t>(c);
            if (b.rows.empty() || b.rows.back() != rr) {
                b.rows.push_back(rr);
                b.row_start.push_back(static_cast<std::int32_t>(b.cols.size()));
            }
            b.cols.push_back(cc);
            if (b.rmax < b.rmin) {
                b.rmin = b.rmax = rr;
                b.cmin = b.cmax = cc;
            }
            b.rmax = rr;
            b.cmin = std::min(b.cmin, cc);
            b.cmax = std::max(b.cmax, cc);
        }
    }
    for (Rows& b : rows) b.row_start.push_back(static_cast<std::int32_t>(b.cols.size()));

    const bool limited = std::isfinite(max_distance);
    // Squared limit, a hair above so the final `d <= max_distance` decides.
    const double limit2 = limited ? max_distance * max_distance * (1 + 1e-12) : kNoLimit;
    auto box_gap2 = [&](const Rows& a, const Rows& b) {
        const double gc = std::max(0, std::max(b.cmin - a.cmax, a.cmin - b.cmax) - 1) * resx;
        const double gr = std::max(0, std::max(b.rmin - a.rmax, a.rmin - b.rmax) - 1) * resy;
        return gc * gc + gr * gr;
    };
    // Candidate partners of each node: all later nodes, or (with a limit)
    // those whose columns come within reach, found by a sweep over cmin.
    std::vector<std::int32_t> by_c(n_nodes);
    for (std::int32_t v = 0; v < n_nodes; ++v) by_c[v] = v;
    std::int32_t reach = 0;
    if (limited) {
        std::stable_sort(by_c.begin(), by_c.end(), [&](std::int32_t a, std::int32_t b) { return rows[a].cmin < rows[b].cmin; });
        const double cells = std::floor(max_distance / resx) + 2;
        reach = cells > 2e9 ? std::numeric_limits<std::int32_t>::max() / 2 : static_cast<std::int32_t>(cells);
    }
    std::vector<PairList> parts(n_nodes);
    lsm::parallel_for_dynamic(n_nodes > 16, n_nodes, [&](i64 iv) {
        const auto i = static_cast<std::int32_t>(iv);
        if (rows[i].cols.empty()) return;
        std::vector<std::int32_t> partners;
        if (!limited) {
            for (std::int32_t j = i + 1; j < n_nodes; ++j) partners.push_back(j);
        } else {
            // Nodes starting at most `reach` columns after i ends, and those
            // starting before i (they end within reach or are skipped by the box test).
            for (std::int32_t k = 0; k < n_nodes; ++k) {
                const std::int32_t j = by_c[k];
                if (rows[j].cmin > rows[i].cmax + reach) break;
                if (j > i) partners.push_back(j);
            }
            std::sort(partners.begin(), partners.end());
        }
        PairList& p = parts[i];
        for (const std::int32_t j : partners) {
            if (rows[j].cols.empty()) continue;
            const double floor2 = box_gap2(rows[i], rows[j]);
            if (floor2 > limit2) continue;
            const bool i_small = rows[i].cols.size() <= rows[j].cols.size();
            const double d2 = i_small ? min_gap2(rows[i], rows[j], resx, resy, floor2, limit2)
                                      : min_gap2(rows[j], rows[i], resx, resy, floor2, limit2);
            const double d = std::sqrt(d2);
            if (d <= max_distance) {
                p.i.push_back(i);
                p.j.push_back(j);
                p.distance.push_back(d);
            }
        }
    });
    return concatenate(parts);
}

std::vector<double> gdal_areas(const double* x, const double* y, const i64* ring_start, const i64* poly_start,
                               const i64* geom_start, i64 n_geoms) {
    auto ring = [&](i64 r) {
        const i64 b = ring_start[r], n = ring_start[r + 1] - b;
        if (n < 2) return 0.0;
        const double* rx = x + b;
        const double* ry = y + b;
        double sum = rx[0] * (ry[1] - ry[n - 1]);
        for (i64 i = 1; i < n - 1; ++i) sum += rx[i] * (ry[i + 1] - ry[i - 1]);
        sum += rx[n - 1] * (ry[0] - ry[n - 2]);
        return 0.5 * std::fabs(sum);
    };
    std::vector<double> out(static_cast<std::size_t>(n_geoms));
    lsm::parallel_for(n_geoms > 1024, n_geoms, [&](i64 g) {
        double area = 0;
        for (i64 p = geom_start[g]; p < geom_start[g + 1]; ++p) {
            double a = 0;
            if (poly_start[p] < poly_start[p + 1]) {
                a = ring(poly_start[p]);
                for (i64 r = poly_start[p] + 1; r < poly_start[p + 1]; ++r) a -= ring(r);
            }
            area += a;
        }
        out[g] = area;
    });
    return out;
}

}  // namespace conn
