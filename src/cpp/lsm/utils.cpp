// SPDX-License-Identifier: GPL-3.0-or-later
//
// landscapemetrics' utility functions: get_centroids, get_nearestneighbour,
// get_circumscribingcircle (the circles are in geometry.cpp),
// get_boundaries, get_adjacencies, and R's cor() for calculate_correlation.
// Each follows landscapemetrics' (and R's) arithmetic so the values are the
// same bits.
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "lsm/landscape.hpp"
#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace lsm {

namespace {

constexpr i64 kParallel = i64(1) << 16;

int threads() {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int thread_id() {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

}  // namespace

// Cells of every patch in raster (row-major) order: offsets into `cell`.
PatchCells Landscape::patch_cells() {
    if (tables_only_) throw std::logic_error("the cells of an out-of-core landscape are not kept");
    const Patches& pt = patches();
    PatchCells out;
    out.offset.assign(static_cast<std::size_t>(pt.n) + 1, 0);
    for (i64 i = 0; i < pt.n; ++i) out.offset[i + 1] = out.offset[i] + pt.cells[i];
    out.cell.resize(static_cast<std::size_t>(out.offset[pt.n]));
    const i64 n_strips = static_cast<i64>(pt.strip_offset.size()) - 1;
    const i64 n_local = pt.strip_offset[n_strips];
    // cells per strip piece, then where each piece starts in its patch
    std::vector<i64> piece(static_cast<std::size_t>(n_local), 0);
    const bool parallel = nx_ * ny_ >= kParallel && n_strips > 1;
    parallel_for_dynamic(parallel, n_strips, [&](i64 s) {
        const i64 y1 = std::min(ny_, (s + 1) * pt.strip_rows);
        for (i64 p = s * pt.strip_rows * nx_; p < y1 * nx_; ++p) {
            if (pt.code[p] != -1) ++piece[pt.strip_offset[s] + (-static_cast<i64>(pt.code[p]) - 2)];
        }
    });
    std::vector<i64> cursor(out.offset.begin(), out.offset.end() - 1);
    for (i64 l = 0; l < n_local; ++l) {
        const i64 cells = piece[l];
        piece[l] = cursor[pt.local_to_patch[l]];
        cursor[pt.local_to_patch[l]] += cells;
    }
    parallel_for_dynamic(parallel, n_strips, [&](i64 s) {
        const i64 y1 = std::min(ny_, (s + 1) * pt.strip_rows);
        for (i64 p = s * pt.strip_rows * nx_; p < y1 * nx_; ++p) {
            if (pt.code[p] != -1) {
                out.cell[piece[pt.strip_offset[s] + (-static_cast<i64>(pt.code[p]) - 2)]++] = p;
            }
        }
    });
    return out;
}

// get_centroids: the mean() of the cell centres of every patch (terra's
// coordinates, cells in raster order); with cell_center, the patch cells
// nearest to it (several on ties, as landscapemetrics returns them).
CentroidTable Landscape::centroids(double xmin, double xmax, double ymin, double ymax, bool cell_center) {
    const Patches& pt = patches();
    const PatchCells pc = patch_cells();
    const double xr = (xmax - xmin) / static_cast<double>(nx_);
    const double yr = (ymax - ymin) / static_cast<double>(ny_);
    auto X = [&](i64 p) { return xmin + ((static_cast<double>(p % nx_) + 0.5) * xr); };
    auto Y = [&](i64 p) { return ymax - ((static_cast<double>(p / nx_) + 0.5) * yr); };
    const i64 n = pt.n;
    std::vector<double> cx(static_cast<std::size_t>(n)), cy(static_cast<std::size_t>(n));
    std::vector<std::vector<i64>> nearest(cell_center ? n : 0);
    // a class with a single cell takes R's scalar sqrt
    const std::vector<i64>& per_class = class_cells_;
#pragma omp parallel if (n >= 64 && nx_ * ny_ >= kParallel)
    {
        std::vector<double> buf, dist;
#pragma omp for schedule(dynamic, 16)
        for (i64 i = 0; i < n; ++i) {
            const i64 b = pc.offset[i];
            const i64 m = pc.offset[i + 1] - b;
            buf.resize(static_cast<std::size_t>(m));
            for (i64 k = 0; k < m; ++k) buf[k] = X(pc.cell[b + k]);
            cx[i] = r_mean(buf.data(), m);
            for (i64 k = 0; k < m; ++k) buf[k] = Y(pc.cell[b + k]);
            cy[i] = r_mean(buf.data(), m);
            if (!cell_center) continue;
            dist.resize(static_cast<std::size_t>(m));
            const bool scalar = per_class[pt.cls[i]] == 1;
            double best = std::numeric_limits<double>::infinity();
            for (i64 k = 0; k < m; ++k) {
                const double dx = X(pc.cell[b + k]) - cx[i];
                const double dy = Y(pc.cell[b + k]) - cy[i];
                dist[k] = r_sqrt(dx * dx + dy * dy, scalar);
                best = std::min(best, dist[k]);
            }
            for (i64 k = 0; k < m; ++k) {
                if (dist[k] == best) nearest[i].push_back(pc.cell[b + k]);
            }
        }
    }
    if (cell_center) order_ties_as_r(nearest);
    CentroidTable out;
    for (i64 i = 0; i < n; ++i) {
        if (!cell_center) {
            out.patch.push_back(static_cast<std::int32_t>(i));
            out.x.push_back(cx[i]);
            out.y.push_back(cy[i]);
            continue;
        }
        for (i64 p : nearest[i]) {
            out.patch.push_back(static_cast<std::int32_t>(i));
            out.x.push_back(X(p));
            out.y.push_back(Y(p));
        }
    }
    return out;
}

// landscapemetrics finds the cells nearest to the centroid in the result
// of merge(points, centroids, by = "id"): within a patch its rows come in
// the order R's merge() leaves them, from a Shell sort (isort_with_index,
// not stable) of the patch keys of all cells of the class in raster order.
// Put the tied cells of each patch in that order.
void Landscape::order_ties_as_r(std::vector<std::vector<i64>>& nearest) {
    const Patches& pt = patches();
    const int k = n_classes();
    std::vector<char> needs(static_cast<std::size_t>(k), 0);
    bool any = false;
    for (i64 i = 0; i < pt.n; ++i) {
        if (nearest[i].size() > 1) {
            needs[pt.cls[i]] = 1;
            any = true;
        }
    }
    if (!any) return;
    for (int c = 0; c < k; ++c) {
        if (!needs[c]) continue;
        // the class cells in raster order, keyed by the (1-based) row of the
        // first cell of their patch: R's match(bx, bxy)
        std::vector<i64> cells;
        std::vector<int> key;
        std::vector<int> first(static_cast<std::size_t>(pt.n), 0);
        for (i64 y = 0; y < ny_; ++y) {
            for (i64 x = 0; x < nx_; ++x) {
                const i64 p = y * nx_ + x;
                if (cls_[p] != c) continue;
                const std::int32_t id = pt.patch(p, y);
                cells.push_back(p);
                if (first[id] == 0) first[id] = static_cast<int>(cells.size());
                key.push_back(first[id]);
            }
        }
        const int m = static_cast<int>(cells.size());
        std::vector<int> ix(static_cast<std::size_t>(m));
        std::iota(ix.begin(), ix.end(), 1);
        int h = 1;
        while (h <= m / 9) h = 3 * h + 1;
        for (; h > 0; h /= 3) {
            for (int i = h; i < m; ++i) {
                const int v = key[i];
                const int iv = ix[i];
                int j = i;
                while (j >= h && key[j - h] > v) {
                    key[j] = key[j - h];
                    ix[j] = ix[j - h];
                    j -= h;
                }
                key[j] = v;
                ix[j] = iv;
            }
        }
        std::vector<int> rank(static_cast<std::size_t>(m));
        for (int r = 0; r < m; ++r) rank[ix[r] - 1] = r;
        // position of each class cell in raster order
        std::vector<i64> sorted_cells = cells;  // already in raster order
        for (i64 i = 0; i < pt.n; ++i) {
            if (pt.cls[i] != c || nearest[i].size() < 2) continue;
            std::vector<i64>& t = nearest[i];
            std::sort(t.begin(), t.end(), [&](i64 a, i64 b) {
                const auto ra = rank[std::lower_bound(sorted_cells.begin(), sorted_cells.end(), a) - sorted_cells.begin()];
                const auto rb = rank[std::lower_bound(sorted_cells.begin(), sorted_cells.end(), b) - sorted_cells.begin()];
                return ra < rb;
            });
        }
    }
}

// get_nearestneighbour on the patches of each class, as landscapemetrics'
// enn does. Its distance per patch is the enn metric. With return_id, the
// patches reached: landscapemetrics sorts the boundary cells of a class
// (cells with a 4-neighbour of another class, NA or the map edge) stably by
// x from raster order, i.e. by (x, y), and gives each cell the first cell of
// another patch at its nearest distance that rcpp_get_nearest_neighbor's
// sweep meets: the last one before it in that order, else the first one
// after it. A patch reaches the patches its cells meet at the patch's
// distance. Here the candidates come from a (x, y) range search around each
// cell, so the cost does not grow with the length of the columns.
NeighbourTable Landscape::nearest_neighbours(bool return_id) {
    const Patches& pt = patches();
    const std::vector<double>& enn_dist = patch_base("enn");
    NeighbourTable out;
    if (!return_id) {
        for (i64 i = 0; i < pt.n; ++i) {
            out.patch.push_back(static_cast<std::int32_t>(i));
            out.dist.push_back(enn_dist[i]);
            out.neighbour.push_back(-1);
        }
        return out;
    }
    const i64 nx = nx_;
    const i64 ny = ny_;
    const int k = n_classes();
    auto boundary = [&](i64 x, i64 y) {
        const i64 p = y * nx + x;
        const std::int32_t c = cls_[p];
        return x == 0 || y == 0 || x == nx - 1 || y == ny - 1 || cls_[p - 1] != c || cls_[p + 1] != c ||
               cls_[p - nx] != c || cls_[p + nx] != c;
    };
    // boundary cells of each class, sorted by (column, row) = by x, then
    // by y (the stable sort by x of raster order)
    std::vector<i64> count(static_cast<std::size_t>(k) + 1, 0);
    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const std::int32_t c = cls_[y * nx + x];
            if (c >= 0 && boundary(x, y)) ++count[c + 1];
        }
    }
    for (int c = 0; c < k; ++c) count[c + 1] += count[c];
    const i64 m = count[k];
    std::vector<std::int32_t> col(static_cast<std::size_t>(m)), row(static_cast<std::size_t>(m));
    std::vector<std::int32_t> sid(static_cast<std::size_t>(m));
    {
        std::vector<i64> at(count.begin(), count.end() - 1);
        for (i64 x = 0; x < nx; ++x) {
            for (i64 y = 0; y < ny; ++y) {
                const std::int32_t c = cls_[y * nx + x];
                if (c < 0 || !boundary(x, y)) continue;
                const i64 j = at[c]++;
                col[j] = static_cast<std::int32_t>(x);
                row[j] = static_cast<std::int32_t>(y);
                sid[j] = pt.patch(y * nx + x, y);
            }
        }
    }
    // first cell of each column within each class
    std::vector<i64> col_start;  // indices where a new (class, column) begins, plus m
    std::vector<i64> col_class_first(static_cast<std::size_t>(k) + 1);
    for (int c = 0; c < k; ++c) {
        col_class_first[c] = static_cast<i64>(col_start.size());
        for (i64 j = count[c]; j < count[c + 1]; ++j) {
            if (j == count[c] || col[j] != col[j - 1]) col_start.push_back(j);
        }
    }
    col_class_first[k] = static_cast<i64>(col_start.size());
    col_start.push_back(m);
    std::vector<int> class_of(static_cast<std::size_t>(m));
    for (int c = 0; c < k; ++c) {
        for (i64 j = count[c]; j < count[c + 1]; ++j) class_of[j] = c;
    }
    const double rx = resx_;
    const double ry = resy_;
    std::vector<std::int32_t> reached(static_cast<std::size_t>(m), -1);
#pragma omp parallel for schedule(dynamic, 256) if (m >= 4096)
    for (i64 i = 0; i < m; ++i) {
        const double target = enn_dist[sid[i]];
        if (std::isnan(target)) continue;
        const int c = class_of[i];
        const double xi = static_cast<double>(col[i] + 1) * rx;
        const double yi = static_cast<double>(row[i] + 1) * ry;
        // cells within the target distance (one cell of slack for rounding)
        const i64 cx = static_cast<i64>(std::ceil(target / rx)) + 1;
        const i64 cy = static_cast<i64>(std::ceil(target / ry)) + 1;
        const i64 c0 = std::max<i64>(0, col[i] - cx);
        const i64 c1 = col[i] + cx;
        const i64 r0 = std::max<i64>(0, row[i] - cy);
        const i64 r1 = row[i] + cy;
        // columns of the class in [c0, c1]
        const auto cols_begin = col_start.begin() + col_class_first[c];
        const auto cols_end = col_start.begin() + col_class_first[c + 1];
        auto it = std::lower_bound(cols_begin, cols_end, c0, [&](i64 start, i64 v) { return col[start] < v; });
        // the smallest squared distance, and its cells: the last one before
        // i and the first one after i (the sweep keeps the first it meets)
        double best = std::numeric_limits<double>::infinity();
        i64 before = -1;
        i64 after = -1;
        for (; it != cols_end && col[*it] <= c1; ++it) {
            const i64 s = *it;
            const i64 e = *(it + 1);
            i64 j = std::lower_bound(row.begin() + s, row.begin() + e, static_cast<std::int32_t>(r0)) - row.begin();
            for (; j < e && row[j] <= r1; ++j) {
                if (sid[j] == sid[i]) continue;
                const double dx = static_cast<double>(col[j] + 1) * rx - xi;
                const double dy = static_cast<double>(row[j] + 1) * ry - yi;
                const double d = dx * dx + dy * dy;
                if (d > best) continue;
                if (d < best) {
                    best = d;
                    before = after = -1;
                }
                if (j < i) {
                    before = std::max(before, j);
                } else if (after < 0 || j < after) {
                    after = j;
                }
            }
        }
        if (std::sqrt(best) == target) reached[i] = sid[before >= 0 ? before : after];
    }
    std::vector<std::vector<std::int32_t>> per_patch(static_cast<std::size_t>(pt.n));
    for (i64 i = 0; i < m; ++i) {
        if (reached[i] >= 0) per_patch[sid[i]].push_back(reached[i]);
    }
    for (i64 i = 0; i < pt.n; ++i) {
        std::vector<std::int32_t>& r = per_patch[i];
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        if (r.empty()) {
            out.patch.push_back(static_cast<std::int32_t>(i));
            out.dist.push_back(enn_dist[i]);
            out.neighbour.push_back(-1);
        }
        for (std::int32_t q : r) {
            out.patch.push_back(static_cast<std::int32_t>(i));
            out.dist.push_back(enn_dist[i]);
            out.neighbour.push_back(q);
        }
    }
    return out;
}

// get_boundaries of the patches of every class (each class on its own, as
// on get_patches' rasters): 1 when a cell is within edge_depth cells
// (city-block) of another class, NA or, unless consider_boundary, the map
// edge; 0 inside; -1 for NA.
std::vector<std::int8_t> Landscape::patch_boundaries(bool consider_boundary, int edge_depth) {
    if (tables_only_) throw std::logic_error("the boundaries of an out-of-core landscape are not kept");
    constexpr std::int32_t kFar = std::numeric_limits<std::int32_t>::max() / 2;
    const std::int32_t outside = consider_boundary ? kFar : 1;
    const std::vector<std::int32_t> d = city_block_distance(cls_.data(), nx_, ny_, outside, outside, outside);
    const i64 n = nx_ * ny_;
    const std::int32_t depth = std::max(1, edge_depth);
    std::vector<std::int8_t> out(static_cast<std::size_t>(n));
    parallel_for(n >= kParallel, n, [&](i64 p) { out[p] = cls_[p] < 0 ? -1 : (d[p] <= depth ? 1 : 0); });
    return out;
}

// Directed cell pairs between classes for the neighbours at the given
// (column, row) offsets (rcpp_get_coocurrence_matrix): k x k, row = class of
// the cell, column = class of the neighbour. NA cells are left out.
std::vector<i64> Landscape::cooccurrence(const std::vector<std::pair<int, int>>& offsets) {
    if (tables_only_) throw std::logic_error("the raster of an out-of-core landscape is not kept");
    const int k = n_classes();
    const std::size_t cells = static_cast<std::size_t>(k) * k;
    const i64 n = nx_ * ny_;
    const int nt = (n >= kParallel && cells * threads() <= (std::size_t(1) << 24)) ? threads() : 1;
    std::vector<std::vector<i64>> part(nt, std::vector<i64>(cells, 0));
#pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (i64 y = 0; y < ny_; ++y) {
        i64* acc = part[nt > 1 ? thread_id() : 0].data();
        for (i64 x = 0; x < nx_; ++x) {
            const std::int32_t a = cls_[y * nx_ + x];
            if (a < 0) continue;
            for (const auto& o : offsets) {
                const i64 xx = x + o.first;
                const i64 yy = y + o.second;
                if (xx < 0 || yy < 0 || xx >= nx_ || yy >= ny_) continue;
                const std::int32_t b = cls_[yy * nx_ + xx];
                if (b >= 0) ++acc[static_cast<std::size_t>(a) * k + b];
            }
        }
    }
    for (int t = 1; t < nt; ++t) {
        for (std::size_t i = 0; i < cells; ++i) part[0][i] += part[t][i];
    }
    return part[0];
}

// The distinct non-NA values, sorted: the value range per chunk, then a
// presence table over the range (per thread, OR-ed), or for wide ranges the
// sorted distinct values of each chunk, merged.
std::vector<std::int32_t> unique_values(const std::int32_t* values, const std::uint8_t* na_mask, i64 n) {
    const bool parallel = n >= kParallel;
    const i64 n_chunks = parallel ? std::min<i64>(256, n / 65536 + 1) : 1;
    std::vector<std::int32_t> lo(static_cast<std::size_t>(n_chunks), std::numeric_limits<std::int32_t>::max());
    std::vector<std::int32_t> hi(static_cast<std::size_t>(n_chunks), std::numeric_limits<std::int32_t>::min());
    std::vector<std::uint8_t> any(static_cast<std::size_t>(n_chunks), 0);
    parallel_for(parallel, n_chunks, [&](i64 c) {
        std::int32_t a = lo[c], b = hi[c];
        bool seen = false;
        for (i64 p = n * c / n_chunks, p1 = n * (c + 1) / n_chunks; p < p1; ++p) {
            if (na_mask && na_mask[p]) continue;
            a = std::min(a, values[p]);
            b = std::max(b, values[p]);
            seen = true;
        }
        lo[c] = a;
        hi[c] = b;
        any[c] = seen;
    });
    if (std::find(any.begin(), any.end(), 1) == any.end()) return {};
    const std::int32_t vmin = *std::min_element(lo.begin(), lo.end());
    const std::int32_t vmax = *std::max_element(hi.begin(), hi.end());
    const i64 range = static_cast<i64>(vmax) - vmin + 1;
    std::vector<std::int32_t> out;
    if (range <= (i64(1) << 22)) {
        std::vector<std::uint8_t> present(static_cast<std::size_t>(range), 0);
        auto mark = [&](i64 p0, i64 p1, std::uint8_t* f) {
            for (i64 p = p0; p < p1; ++p) {
                if (!(na_mask && na_mask[p])) f[values[p] - static_cast<i64>(vmin)] = 1;
            }
        };
        if (parallel && range <= n / 4) {
#pragma omp parallel
            {
                std::vector<std::uint8_t> local(static_cast<std::size_t>(range), 0);
#pragma omp for schedule(static)
                for (i64 c = 0; c < n_chunks; ++c) mark(n * c / n_chunks, n * (c + 1) / n_chunks, local.data());
#pragma omp critical
                for (i64 v = 0; v < range; ++v) present[v] |= local[v];
            }
        } else {
            mark(0, n, present.data());
        }
        for (i64 v = 0; v < range; ++v) {
            if (present[v]) out.push_back(static_cast<std::int32_t>(v + vmin));
        }
        return out;
    }
    std::vector<std::vector<std::int32_t>> distinct(static_cast<std::size_t>(n_chunks));
    parallel_for(parallel, n_chunks, [&](i64 c) {
        std::vector<std::int32_t>& d = distinct[c];
        for (i64 p = n * c / n_chunks, p1 = n * (c + 1) / n_chunks; p < p1; ++p) {
            if (!(na_mask && na_mask[p])) d.push_back(values[p]);
        }
        std::sort(d.begin(), d.end());
        d.erase(std::unique(d.begin(), d.end()), d.end());
    });
    for (auto& d : distinct) {
        out.insert(out.end(), d.begin(), d.end());
        std::vector<std::int32_t>().swap(d);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// landscapemetrics' get_boundaries on the raster values themselves
// (rcpp_get_boundaries with 4 directions): a cell is a boundary when the
// cell below has another value (or NA), or the cell above, left or right is
// NA; the map edge counts as NA unless consider_boundary. edge_depth > 1
// peels the boundary off and repeats. Returns 0/1 (or the cell's value with
// patch_id), NA as `na`.
std::vector<std::int32_t> raster_boundaries(const std::int32_t* values, const std::uint8_t* na_mask, i64 nx, i64 ny,
                                            bool consider_boundary, int edge_depth, bool patch_id, std::int32_t na) {
    // the grid landscapemetrics works on: an NA ring unless consider_boundary
    const i64 pad = consider_boundary ? 0 : 1;
    const i64 gx = nx + 2 * pad;
    const i64 gy = ny + 2 * pad;
    std::vector<std::int32_t> v(static_cast<std::size_t>(gx * gy));
    std::vector<std::uint8_t> isna(static_cast<std::size_t>(gx * gy), 1);
    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const i64 p = y * nx + x;
            const i64 g = (y + pad) * gx + (x + pad);
            v[g] = values[p];
            isna[g] = na_mask && na_mask[p];
        }
    }
    auto boundaries = [&](const std::vector<std::int32_t>& val, const std::vector<std::uint8_t>& nan) {
        std::vector<std::uint8_t> b(static_cast<std::size_t>(gx * gy), 0);
        parallel_for(gx * gy >= kParallel, gy, [&](i64 y) {
            for (i64 x = 0; x < gx; ++x) {
                const i64 g = y * gx + x;
                if (nan[g]) continue;
                bool edge = false;
                if (y + 1 < gy) edge = nan[g + gx] || val[g + gx] != val[g];
                if (!edge && y > 0) edge = nan[g - gx];
                if (!edge && x > 0) edge = nan[g - 1];
                if (!edge && x + 1 < gx) edge = nan[g + 1];
                b[g] = edge;
            }
        });
        return b;
    };
    std::vector<std::uint8_t> b = boundaries(v, isna);
    if (edge_depth > 1) {
        std::vector<std::uint8_t> nan = isna;
        std::vector<std::uint8_t> cur = b;
        const std::vector<std::int32_t> zeros(v.size(), 0);
        for (int i = 1; i < edge_depth; ++i) {
            for (std::size_t g = 0; g < cur.size(); ++g) nan[g] = nan[g] || cur[g];
            cur = boundaries(zeros, nan);
            for (std::size_t g = 0; g < cur.size(); ++g) b[g] = b[g] || cur[g];
        }
    }
    std::vector<std::int32_t> out(static_cast<std::size_t>(nx * ny));
    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const i64 p = y * nx + x;
            const i64 g = (y + pad) * gx + (x + pad);
            if (isna[g]) {
                out[p] = na;
            } else if (b[g]) {
                out[p] = patch_id ? values[p] : 1;
            } else {
                out[p] = 0;
            }
        }
    }
    return out;
}

namespace {

// R's rank(ties.method = "average")
std::vector<double> r_rank(const double* x, i64 n) {
    std::vector<i64> idx(static_cast<std::size_t>(n));
    std::iota(idx.begin(), idx.end(), i64(0));
    std::stable_sort(idx.begin(), idx.end(), [&](i64 a, i64 b) { return x[a] < x[b]; });
    std::vector<double> r(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n;) {
        i64 j = i;
        while (j + 1 < n && x[idx[j + 1]] == x[idx[i]]) ++j;
        const double avg = (static_cast<double>(i + 1) + static_cast<double>(j + 1)) / 2;
        for (i64 t = i; t <= j; ++t) r[idx[t]] = avg;
        i = j + 1;
    }
    return r;
}

double sign(double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); }

}  // namespace

// R's cor(x, method) of the columns of the n x m matrix x (column-major):
// cov.c's cov_na_1 (two-pass means and cross products in LDOUBLE, gcc's
// sqrt), Kendall's tau from sign products, Spearman as Pearson on ranks.
// NA in a column gives NA correlations; a constant column NA (R warns).
std::vector<double> r_cor(const std::vector<double>& x_in, i64 n, i64 m, const std::string& method) {
    if (method != "pearson" && method != "kendall" && method != "spearman") {
        throw std::invalid_argument("method must be 'pearson', 'kendall' or 'spearman'");
    }
    std::vector<double> x = x_in;
    std::vector<char> has_na(static_cast<std::size_t>(m), 0);
    for (i64 j = 0; j < m; ++j) {
        for (i64 i = 0; i < n; ++i) has_na[j] = has_na[j] || std::isnan(x[j * n + i]);
        if (method == "spearman" && !has_na[j]) {
            const std::vector<double> r = r_rank(x.data() + j * n, n);
            std::copy(r.begin(), r.end(), x.begin() + j * n);
        }
    }
    std::vector<double> ans(static_cast<std::size_t>(m * m), kNaN);
    if (n <= 1) return ans;
    const bool kendall = method == "kendall";
    std::vector<double> xm(static_cast<std::size_t>(m), kNaN);
    if (!kendall) {
        for (i64 j = 0; j < m; ++j) {
            if (!has_na[j]) xm[j] = r_mean(x.data() + j * n, n);
        }
    }
    for (i64 i = 0; i < m; ++i) {
        if (has_na[i]) continue;
        const double* xx = x.data() + i * n;
        for (i64 j = 0; j <= i; ++j) {
            if (has_na[j]) continue;
            const double* yy = x.data() + j * n;
            Extended sum = 0.0;
            if (!kendall) {
                const Extended xxm = xm[i];
                const Extended yym = xm[j];
                for (i64 k = 0; k < n; ++k) sum += (Extended(xx[k]) - xxm) * (Extended(yy[k]) - yym);
                ans[j * m + i] = ans[i * m + j] = static_cast<double>(sum / Extended(n - 1));
            } else {
                for (i64 k = 0; k < n; ++k) {
                    for (i64 l = 0; l < n; ++l) sum += sign(xx[k] - xx[l]) * sign(yy[k] - yy[l]);
                }
                ans[j * m + i] = ans[i * m + j] = static_cast<double>(sum);
            }
        }
    }
    std::vector<double> sd(static_cast<std::size_t>(m), kNaN);
    for (i64 i = 0; i < m; ++i) {
        if (!has_na[i]) sd[i] = std::sqrt(ans[i * m + i]);
    }
    for (i64 i = 0; i < m; ++i) {
        if (has_na[i]) continue;
        for (i64 j = 0; j < i; ++j) {
            if (has_na[j]) continue;
            if (sd[i] == 0 || sd[j] == 0) {
                ans[j * m + i] = ans[i * m + j] = kNaN;
            } else {
                const double r = ans[i * m + j] / (sd[i] * sd[j]);
                ans[j * m + i] = ans[i * m + j] = r >= 1. ? 1. : (r <= -1. ? -1. : r);
            }
        }
    }
    for (i64 i = 0; i < m; ++i) ans[i * m + i] = 1.0;  // as cov.c, also for NA columns
    return ans;
}

}  // namespace lsm
