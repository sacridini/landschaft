// SPDX-License-Identifier: GPL-3.0-or-later
#include "lsm/landscape.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace lsm {

namespace {

constexpr i64 kParallelThreshold = i64(1) << 16;
constexpr i64 kStripRows = 64;

int max_threads() {
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

// Union-find whose roots are the smallest index of their set, so that
// parent[p] <= p holds throughout.
std::int32_t uf_find(std::int32_t* parent, std::int32_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

void uf_unite(std::int32_t* parent, std::int32_t a, std::int32_t b) {
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return;
    if (a < b) {
        parent[b] = a;
    } else {
        parent[a] = b;
    }
}

// Indices 0..n-1 ordered by key (LSD radix sort; keys move with the indices
// so every pass reads sequentially).
std::vector<i64> sort_by_key(std::vector<std::uint64_t> key) {
    const std::size_t n = key.size();
    std::vector<i64> order(n);
    std::iota(order.begin(), order.end(), i64(0));
    if (n < 512) {  // small maps (moving windows, sample plots)
        std::sort(order.begin(), order.end(), [&](i64 a, i64 b) { return key[a] < key[b]; });
        return order;
    }
    std::vector<i64> order_tmp(n);
    std::vector<std::uint64_t> key_tmp(n);
    std::uint64_t all = 0;
    for (std::uint64_t k : key) all |= k;
    constexpr int kBits = 11;
    constexpr std::uint64_t kMask = (std::uint64_t(1) << kBits) - 1;
    std::vector<std::size_t> count(std::size_t(1) << kBits);
    for (int shift = 0; shift < 64 && (all >> shift) != 0; shift += kBits) {
        std::fill(count.begin(), count.end(), 0);
        for (std::size_t i = 0; i < n; ++i) ++count[(key[i] >> shift) & kMask];
        std::size_t sum = 0;
        for (auto& c : count) {
            const std::size_t v = c;
            c = sum;
            sum += v;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t j = count[(key[i] >> shift) & kMask]++;
            key_tmp[j] = key[i];
            order_tmp[j] = order[i];
        }
        key.swap(key_tmp);
        order.swap(order_tmp);
    }
    return order;
}

// Patch pieces of one strip and their statistics.
struct StripStats {
    std::vector<std::int32_t> cls;
    std::vector<i64> cells, edge_h, edge_v, adj_orth, adj_diag, sum_x, sum_y, first_cm;
};

// Label the rows [y0, y1): pixels become -(local index) - 2 (NA: -1) in
// `code`, which serves meanwhile as the union-find parent array.
void label_strip(const std::vector<std::int32_t>& cls, std::int32_t* code, i64 nx, i64 ny, i64 y0, i64 y1,
                 int directions, StripStats& st) {
    for (i64 p = y0 * nx; p < y1 * nx; ++p) code[p] = static_cast<std::int32_t>(p);
    for (i64 y = y0; y < y1; ++y) {
        const i64 row = y * nx;
        for (i64 x = 0; x < nx; ++x) {
            const i64 p = row + x;
            const std::int32_t c = cls[p];
            if (c < 0) continue;
            const auto P = static_cast<std::int32_t>(p);
            if (x > 0 && cls[p - 1] == c) uf_unite(code, P, P - 1);
            if (y == y0) continue;
            if (cls[p - nx] == c) uf_unite(code, P, static_cast<std::int32_t>(p - nx));
            if (directions == 8) {
                if (x > 0 && cls[p - nx - 1] == c) uf_unite(code, P, static_cast<std::int32_t>(p - nx - 1));
                if (x + 1 < nx && cls[p - nx + 1] == c) uf_unite(code, P, static_cast<std::int32_t>(p - nx + 1));
            }
        }
    }
    // Forward pass: parents (earlier pixels, already resolved) become local
    // ids, and the statistics of each piece are accumulated.
    i64 n_local = 0;
    for (i64 y = y0; y < y1; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const i64 p = y * nx + x;
            const std::int32_t c = cls[p];
            if (c < 0) {
                code[p] = -1;
                continue;
            }
            i64 k;
            const std::int32_t up = code[p];
            if (up == static_cast<std::int32_t>(p)) {
                k = n_local++;
                code[p] = static_cast<std::int32_t>(-(k + 2));
                st.cls.push_back(c);
                for (auto* v : {&st.cells, &st.edge_h, &st.edge_v, &st.adj_orth, &st.adj_diag, &st.sum_x, &st.sum_y}) {
                    v->push_back(0);
                }
                st.first_cm.push_back(x * ny + y);
            } else {
                code[p] = code[up];
                k = -static_cast<i64>(code[p]) - 2;
            }
            st.cells[k] += 1;
            st.sum_x[k] += x;
            st.sum_y[k] += y;
            st.first_cm[k] = std::min(st.first_cm[k], x * ny + y);
            const bool left = x > 0 && cls[p - 1] == c;
            const bool right = x + 1 < nx && cls[p + 1] == c;
            const bool above = y > 0 && cls[p - nx] == c;
            const bool below = y + 1 < ny && cls[p + nx] == c;
            st.edge_h[k] += !left + !right;
            st.edge_v[k] += !above + !below;
            st.adj_orth[k] += right + below;
            // With 8 directions, diagonal cells of one class are one patch.
            if (directions == 8 && y > 0) {
                st.adj_diag[k] += (x > 0 && cls[p - nx - 1] == c) + (x + 1 < nx && cls[p - nx + 1] == c);
            }
        }
    }
}

Patches label_patches(const std::vector<std::int32_t>& cls, i64 nx, i64 ny, int directions) {
    const i64 n = nx * ny;
    Patches out;
    out.strip_rows = kStripRows;
    const i64 n_strips = (ny + kStripRows - 1) / kStripRows;
    out.code.resize(static_cast<std::size_t>(n));
    std::int32_t* code = out.code.data();
    std::vector<StripStats> strips(static_cast<std::size_t>(n_strips));
#pragma omp parallel for schedule(dynamic) if (n >= kParallelThreshold && n_strips > 1)
    for (i64 s = 0; s < n_strips; ++s) {
        label_strip(cls, code, nx, ny, s * kStripRows, std::min(ny, (s + 1) * kStripRows), directions, strips[s]);
    }
    out.strip_offset.assign(static_cast<std::size_t>(n_strips + 1), 0);
    for (i64 s = 0; s < n_strips; ++s) {
        out.strip_offset[s + 1] = out.strip_offset[s] + static_cast<i64>(strips[s].cls.size());
    }
    const i64 n_local = out.strip_offset[n_strips];

    // Join the pieces that touch across strip seams.
    std::vector<std::int32_t> parent(static_cast<std::size_t>(n_local));
    std::iota(parent.begin(), parent.end(), 0);
    for (i64 s = 1; s < n_strips; ++s) {
        const i64 y = s * kStripRows;
        for (i64 x = 0; x < nx; ++x) {
            const i64 p = y * nx + x;
            const std::int32_t c = cls[p];
            if (c < 0) continue;
            const auto a = static_cast<std::int32_t>(out.local(p, y));
            auto join = [&](i64 q) {
                if (cls[q] == c) uf_unite(parent.data(), a, static_cast<std::int32_t>(out.local(q, y - 1)));
            };
            join(p - nx);
            if (directions == 8) {
                if (x > 0) join(p - nx - 1);
                if (x + 1 < nx) join(p - nx + 1);
            }
        }
    }

    // Patches: the sums of their pieces.
    std::vector<std::int32_t> gid(static_cast<std::size_t>(n_local), -1);
    std::vector<i64> first_cm;
    Patches tmp;
    for (i64 s = 0; s < n_strips; ++s) {
        const StripStats& st = strips[s];
        for (std::size_t k = 0; k < st.cls.size(); ++k) {
            const auto l = static_cast<std::int32_t>(out.strip_offset[s] + static_cast<i64>(k));
            const std::int32_t root = uf_find(parent.data(), l);
            if (gid[root] < 0) {
                gid[root] = static_cast<std::int32_t>(tmp.n++);
                tmp.cls.push_back(st.cls[k]);
                for (auto* v : {&tmp.cells, &tmp.edge_h, &tmp.edge_v, &tmp.adj_orth, &tmp.adj_diag, &tmp.sum_x,
                                &tmp.sum_y}) {
                    v->push_back(0);
                }
                first_cm.push_back(st.first_cm[k]);
            }
            const std::int32_t g = gid[root];
            gid[l] = g;
            tmp.cells[g] += st.cells[k];
            tmp.edge_h[g] += st.edge_h[k];
            tmp.edge_v[g] += st.edge_v[k];
            tmp.adj_orth[g] += st.adj_orth[k];
            tmp.adj_diag[g] += st.adj_diag[k];
            tmp.sum_x[g] += st.sum_x[k];
            tmp.sum_y[g] += st.sum_y[k];
            first_cm[g] = std::min(first_cm[g], st.first_cm[k]);
        }
    }
    strips = {};
    parent = {};

    // landscapemetrics order: by class, then by first cell in column-major order.
    std::vector<std::uint64_t> key(static_cast<std::size_t>(tmp.n));
    for (i64 i = 0; i < tmp.n; ++i) {
        key[i] = static_cast<std::uint64_t>(tmp.cls[i]) * static_cast<std::uint64_t>(n) +
                 static_cast<std::uint64_t>(first_cm[i]);
    }
    const std::vector<i64> order = sort_by_key(std::move(key));
    out.n = tmp.n;
    auto permute = [&](const auto& src, auto& dst) {
        dst.resize(src.size());
        for (i64 i = 0; i < tmp.n; ++i) dst[i] = src[order[i]];
    };
    permute(tmp.cls, out.cls);
    permute(tmp.cells, out.cells);
    permute(tmp.edge_h, out.edge_h);
    permute(tmp.edge_v, out.edge_v);
    permute(tmp.adj_orth, out.adj_orth);
    permute(tmp.adj_diag, out.adj_diag);
    permute(tmp.sum_x, out.sum_x);
    permute(tmp.sum_y, out.sum_y);
    permute(first_cm, out.first_cm);
    std::vector<std::int32_t> rank(static_cast<std::size_t>(tmp.n));
    for (i64 i = 0; i < tmp.n; ++i) rank[order[i]] = static_cast<std::int32_t>(i);
    out.local_to_patch.resize(static_cast<std::size_t>(n_local));
    for (i64 l = 0; l < n_local; ++l) out.local_to_patch[l] = rank[gid[l]];

    // With 4 directions diagonal cells of one class may or may not be one
    // patch: count the pairs once the patches are known.
    if (directions == 4) {
        std::vector<std::vector<i64>> counts(static_cast<std::size_t>(n_strips));
#pragma omp parallel for schedule(dynamic) if (n >= kParallelThreshold && n_strips > 1)
        for (i64 s = 0; s < n_strips; ++s) {
            const i64 y0 = std::max<i64>(1, s * kStripRows);
            const i64 y1 = std::min(ny, (s + 1) * kStripRows);
            std::vector<i64>& cnt = counts[s];
            cnt.assign(static_cast<std::size_t>(out.strip_offset[s + 1] - out.strip_offset[s]), 0);
            for (i64 y = y0; y < y1; ++y) {
                for (i64 x = 0; x < nx; ++x) {
                    const i64 p = y * nx + x;
                    const std::int32_t c = cls[p];
                    if (c < 0) continue;
                    const std::int32_t me = out.patch(p, y);
                    const i64 k = -static_cast<i64>(code[p]) - 2;
                    if (x > 0 && cls[p - nx - 1] == c && out.patch(p - nx - 1, y - 1) == me) ++cnt[k];
                    if (x + 1 < nx && cls[p - nx + 1] == c && out.patch(p - nx + 1, y - 1) == me) ++cnt[k];
                }
            }
        }
        for (i64 s = 0; s < n_strips; ++s) {
            for (std::size_t k = 0; k < counts[s].size(); ++k) {
                out.adj_diag[out.local_to_patch[out.strip_offset[s] + static_cast<i64>(k)]] += counts[s][k];
            }
        }
    }
    return out;
}

}  // namespace

Landscape::Landscape(const std::int32_t* values, const std::uint8_t* na_mask, i64 nx, i64 ny, double resx,
                     double resy, Options options)
    : nx_(nx), ny_(ny), resx_(resx), resy_(resy), options_(std::move(options)) {
    if (nx < 0 || ny < 0) throw std::invalid_argument("negative raster size");
    if (options_.directions != 4 && options_.directions != 8) throw std::invalid_argument("directions must be 4 or 8");
    if (options_.neighbourhood != 4 && options_.neighbourhood != 8) {
        throw std::invalid_argument("neighbourhood must be 4 or 8");
    }
    if (options_.base != "log2" && options_.base != "log" && options_.base != "log10") {
        throw std::invalid_argument("base must be 'log2', 'log' or 'log10'");
    }
    if (options_.edge_depth < 0) throw std::invalid_argument("edge_depth must be >= 0");
    const i64 n = nx * ny;
    if (n >= (i64(1) << 31) - 2) throw std::invalid_argument("raster too large for in-memory metrics");

    // Class values in increasing order, mapped to dense indices.
    std::int32_t vmin = std::numeric_limits<std::int32_t>::max();
    std::int32_t vmax = std::numeric_limits<std::int32_t>::min();
    for (i64 p = 0; p < n; ++p) {
        if (na_mask && na_mask[p]) continue;
        vmin = std::min(vmin, values[p]);
        vmax = std::max(vmax, values[p]);
    }
    cls_.assign(static_cast<std::size_t>(n), -1);
    if (vmin > vmax) return;  // all NA

    const i64 range = static_cast<i64>(vmax) - vmin + 1;
    if (range <= (i64(1) << 24)) {
        std::vector<std::int32_t> lut(static_cast<std::size_t>(range), -1);
        for (i64 p = 0; p < n; ++p) {
            if (na_mask && na_mask[p]) continue;
            lut[static_cast<std::size_t>(values[p] - static_cast<i64>(vmin))] = 0;
        }
        for (i64 v = 0; v < range; ++v) {
            if (lut[v] == 0) {
                lut[v] = static_cast<std::int32_t>(class_values_.size());
                class_values_.push_back(static_cast<std::int32_t>(v + vmin));
            }
        }
#pragma omp parallel for schedule(static) if (n >= kParallelThreshold)
        for (i64 p = 0; p < n; ++p) {
            if (!(na_mask && na_mask[p])) cls_[p] = lut[static_cast<std::size_t>(values[p] - static_cast<i64>(vmin))];
        }
    } else {
        std::vector<std::int32_t> vals;
        for (i64 p = 0; p < n; ++p) {
            if (!(na_mask && na_mask[p])) vals.push_back(values[p]);
        }
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        class_values_ = vals;
#pragma omp parallel for schedule(static) if (n >= kParallelThreshold)
        for (i64 p = 0; p < n; ++p) {
            if (na_mask && na_mask[p]) continue;
            cls_[p] = static_cast<std::int32_t>(std::lower_bound(vals.begin(), vals.end(), values[p]) - vals.begin());
        }
    }
    class_cells_.assign(class_values_.size(), 0);
    for (i64 p = 0; p < n; ++p) {
        if (cls_[p] >= 0) ++class_cells_[cls_[p]];
    }
    valid_cells_ = std::accumulate(class_cells_.begin(), class_cells_.end(), i64(0));
}

const Patches& Landscape::patches() {
    if (!patches_) patches_ = label_patches(cls_, nx_, ny_, options_.directions);
    return *patches_;
}

std::vector<std::int32_t> Landscape::labels() {
    const Patches& pt = patches();
    const i64 n = nx_ * ny_;
    std::vector<std::int32_t> out(static_cast<std::size_t>(n));
#pragma omp parallel for schedule(static) if (n >= kParallelThreshold)
    for (i64 y = 0; y < ny_; ++y) {
        for (i64 x = 0; x < nx_; ++x) {
            const i64 p = y * nx_ + x;
            out[p] = pt.patch(p, y) + 1;
        }
    }
    return out;
}

const Core& Landscape::core() {
    if (core_) return *core_;
    const Patches& pt = patches();
    const i64 nx = nx_;
    const i64 ny = ny_;
    const i64 n = nx * ny;
    const bool parallel = n >= kParallelThreshold;
    // landscapemetrics' get_boundaries always marks one ring of edge cells.
    const i64 depth = std::max(1, options_.edge_depth);
    constexpr std::int32_t kFar = std::numeric_limits<std::int32_t>::max() / 2;
    const std::int32_t outside = options_.consider_boundary ? kFar : 1;
    const std::int32_t* cls = cls_.data();

    // City-block distance to the nearest cell of another class (or NA):
    // along rows, then along columns. A path through a cell of another class
    // would pass a nearer source, so a neighbour of another class is at 1.
    std::vector<std::int32_t> dist(static_cast<std::size_t>(n));
    std::int32_t* d = dist.data();
#pragma omp parallel for schedule(static) if (parallel)
    for (i64 y = 0; y < ny; ++y) {
        const i64 row = y * nx;
        for (i64 x = 0; x < nx; ++x) {
            const i64 p = row + x;
            if (cls[p] < 0) continue;
            d[p] = x == 0 ? outside : (cls[p - 1] != cls[p] ? 1 : std::min(kFar, d[p - 1] + 1));
        }
        for (i64 x = nx - 1; x >= 0; --x) {
            const i64 p = row + x;
            if (cls[p] < 0) continue;
            const std::int32_t r = x == nx - 1 ? outside : (cls[p + 1] != cls[p] ? 1 : std::min(kFar, d[p + 1] + 1));
            d[p] = std::min(d[p], r);
        }
    }
    constexpr i64 kBlock = 256;
    const i64 n_blocks = (nx + kBlock - 1) / kBlock;
#pragma omp parallel for schedule(dynamic) if (parallel && n_blocks > 1)
    for (i64 b = 0; b < n_blocks; ++b) {
        const i64 x0 = b * kBlock;
        const i64 x1 = std::min(nx, x0 + kBlock);
        for (i64 y = 0; y < ny; ++y) {
            for (i64 x = x0; x < x1; ++x) {
                const i64 p = y * nx + x;
                if (cls[p] < 0) continue;
                const std::int32_t u = y == 0 ? outside : (cls[p - nx] != cls[p] ? 1 : std::min(kFar, d[p - nx] + 1));
                d[p] = std::min(d[p], u);
            }
        }
        for (i64 y = ny - 1; y >= 0; --y) {
            for (i64 x = x0; x < x1; ++x) {
                const i64 p = y * nx + x;
                if (cls[p] < 0) continue;
                const std::int32_t v = y == ny - 1 ? outside : (cls[p + nx] != cls[p] ? 1 : std::min(kFar, d[p + nx] + 1));
                d[p] = std::min(d[p], v);
            }
        }
    }

    // Core cells per patch, summed per strip piece and then per patch.
    std::vector<std::int32_t> core_cls(static_cast<std::size_t>(n), -1);
    const i64 n_strips = static_cast<i64>(pt.strip_offset.size()) - 1;
    std::vector<std::vector<i64>> pieces(static_cast<std::size_t>(n_strips));
#pragma omp parallel for schedule(dynamic) if (parallel && n_strips > 1)
    for (i64 s = 0; s < n_strips; ++s) {
        std::vector<i64>& acc = pieces[s];
        acc.assign(static_cast<std::size_t>(pt.strip_offset[s + 1] - pt.strip_offset[s]), 0);
        const i64 y1 = std::min(ny, (s + 1) * pt.strip_rows);
        for (i64 y = s * pt.strip_rows; y < y1; ++y) {
            for (i64 x = 0; x < nx; ++x) {
                const i64 p = y * nx + x;
                if (cls[p] < 0 || d[p] <= depth) continue;
                core_cls[p] = cls[p];
                ++acc[-static_cast<i64>(pt.code[p]) - 2];
            }
        }
    }
    dist = {};
    Core out;
    out.cells.assign(static_cast<std::size_t>(pt.n), 0);
    out.areas.assign(static_cast<std::size_t>(pt.n), 0);
    for (i64 s = 0; s < n_strips; ++s) {
        for (std::size_t k = 0; k < pieces[s].size(); ++k) {
            out.cells[pt.local_to_patch[pt.strip_offset[s] + static_cast<i64>(k)]] += pieces[s][k];
        }
    }
    pieces = {};

    // Disjunct core areas: core cells labelled like patches; each lies in one
    // patch (cores of different patches cannot touch).
    const Patches cores = label_patches(core_cls, nx, ny, options_.directions);
    for (i64 i = 0; i < cores.n; ++i) {
        const i64 x = cores.first_cm[i] / ny;
        const i64 y = cores.first_cm[i] % ny;
        ++out.areas[pt.patch(y * nx + x, y)];
    }
    core_ = std::move(out);
    return *core_;
}

const Adjacency& Landscape::adjacency() {
    if (adjacency_) return *adjacency_;
    const int k = n_classes();
    const int dim = k + 2;
    const int na = k;
    const int out = k + 1;
    const bool diag = options_.neighbourhood == 8;
    const std::size_t cells = static_cast<std::size_t>(dim) * dim;
    const i64 n = nx_ * ny_;
    // Thread-local tables unless they would be large.
    const int n_threads = (n >= kParallelThreshold && cells * 3 * max_threads() <= (std::size_t(1) << 25)) ? max_threads() : 1;
    std::vector<std::vector<i64>> h(n_threads, std::vector<i64>(cells, 0));
    std::vector<std::vector<i64>> v(n_threads, std::vector<i64>(cells, 0));
    std::vector<std::vector<i64>> d(diag ? n_threads : 0, std::vector<i64>(cells, 0));
    const i64 nx = nx_;
    const i64 ny = ny_;
#pragma omp parallel for schedule(static) num_threads(n_threads) if (n_threads > 1)
    for (i64 y = 0; y < ny; ++y) {
        const int t = n_threads > 1 ? thread_id() : 0;
        i64* hh = h[t].data();
        i64* vv = v[t].data();
        for (i64 x = 0; x < nx; ++x) {
            const i64 p = y * nx + x;
            const int a = cls_[p];
            if (a < 0) continue;
            const i64 row = static_cast<i64>(a) * dim;
            auto nb = [&](i64 q) { return cls_[q] < 0 ? na : cls_[q]; };
            ++hh[row + (x > 0 ? nb(p - 1) : out)];
            ++hh[row + (x + 1 < nx ? nb(p + 1) : out)];
            ++vv[row + (y > 0 ? nb(p - nx) : out)];
            ++vv[row + (y + 1 < ny ? nb(p + nx) : out)];
            if (diag) {
                i64* dd = d[t].data();
                for (int dy = -1; dy <= 1; dy += 2) {
                    const i64 yy = y + dy;
                    if (yy < 0 || yy >= ny) continue;
                    for (int dx = -1; dx <= 1; dx += 2) {
                        const i64 xx = x + dx;
                        if (xx < 0 || xx >= nx) continue;
                        const int b = cls_[yy * nx + xx];
                        if (b >= 0) ++dd[row + b];
                    }
                }
            }
        }
    }
    Adjacency adj;
    adj.dim = dim;
    adj.h = std::move(h[0]);
    adj.v = std::move(v[0]);
    for (int t = 1; t < n_threads; ++t) {
        for (std::size_t i = 0; i < cells; ++i) {
            adj.h[i] += h[t][i];
            adj.v[i] += v[t][i];
        }
    }
    if (diag) {
        adj.diag = std::move(d[0]);
        for (int t = 1; t < n_threads; ++t) {
            for (std::size_t i = 0; i < cells; ++i) adj.diag[i] += d[t][i];
        }
    }
    adjacency_ = std::move(adj);
    return *adjacency_;
}

}  // namespace lsm
