// SPDX-License-Identifier: GPL-3.0-or-later
#include "lsm/landscape.hpp"
#include "lsm/parallel.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace lsm {

namespace {

constexpr i64 kParallelThreshold = i64(1) << 16;

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
// so every pass reads sequentially). Large inputs sort in parallel: every
// thread counts the digits of its chunk and scatters it to its own offsets,
// which keeps the sort stable.
std::vector<i64> sort_by_key(std::vector<std::uint64_t> key) {
    const std::size_t n = key.size();
    std::vector<i64> order(n);
    std::iota(order.begin(), order.end(), i64(0));
    if (n < 512) {  // small maps (moving windows, sample plots)
        std::sort(order.begin(), order.end(), [&](i64 a, i64 b) { return key[a] < key[b]; });
        return order;
    }
    const int nt = n >= (std::size_t(1) << 20) ? max_threads() : 1;
    if (nt > 1) {
        std::vector<i64> order_tmp(n);
        std::vector<std::uint64_t> key_tmp(n);
        std::uint64_t all = 0;
        for (std::uint64_t k : key) all |= k;
        constexpr int kBits = 11;
        constexpr std::size_t kBuckets = std::size_t(1) << kBits;
        std::vector<std::size_t> count(static_cast<std::size_t>(nt) * kBuckets);
        const std::size_t chunk = (n + nt - 1) / nt;
        for (int shift = 0; shift < 64 && (all >> shift) != 0; shift += kBits) {
#pragma omp parallel num_threads(nt)
            {
                const int t = thread_id();
                std::size_t* c = count.data() + static_cast<std::size_t>(t) * kBuckets;
                std::fill(c, c + kBuckets, 0);
                const std::size_t b = std::min(n, t * chunk);
                const std::size_t e = std::min(n, b + chunk);
                for (std::size_t i = b; i < e; ++i) ++c[(key[i] >> shift) & (kBuckets - 1)];
#pragma omp barrier
#pragma omp single
                {
                    std::size_t sum = 0;
                    for (std::size_t d = 0; d < kBuckets; ++d) {
                        for (int u = 0; u < nt; ++u) {
                            std::size_t& v = count[static_cast<std::size_t>(u) * kBuckets + d];
                            const std::size_t x = v;
                            v = sum;
                            sum += x;
                        }
                    }
                }
                for (std::size_t i = b; i < e; ++i) {
                    const std::size_t j = c[(key[i] >> shift) & (kBuckets - 1)]++;
                    key_tmp[j] = key[i];
                    order_tmp[j] = order[i];
                }
            }
            key.swap(key_tmp);
            order.swap(order_tmp);
        }
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
void label_strip(const std::int32_t* cls, std::int32_t* code, i64 nx, i64 ny, i64 y0, i64 y1,
                 int directions, StripStats& st) {
    // room for the pieces of small strips at once (one allocation per table)
    const auto guess = static_cast<std::size_t>(std::min<i64>(nx * (y1 - y0), 256));
    for (auto* v : {&st.cells, &st.edge_h, &st.edge_v, &st.adj_orth, &st.adj_diag, &st.sum_x, &st.sum_y, &st.first_cm}) {
        v->reserve(guess);
    }
    st.cls.reserve(guess);
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

}  // namespace

// With 4 directions diagonal cells of one class may or may not be one
// patch: count the pairs once the patches are known.
static void count_diagonal_pairs(const std::int32_t* cls, i64 nx, i64 ny, Patches& out, bool parallel) {
    const std::int32_t* code = out.code.data();
    const i64 n_strips = static_cast<i64>(out.strip_offset.size()) - 1;
    std::vector<std::vector<i64>> counts(static_cast<std::size_t>(n_strips));
    parallel_for_dynamic(parallel, n_strips, [&](i64 s) {
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
    });
    for (i64 s = 0; s < n_strips; ++s) {
        for (std::size_t k = 0; k < counts[s].size(); ++k) {
            out.adj_diag[out.local_to_patch[out.strip_offset[s] + static_cast<i64>(k)]] += counts[s][k];
        }
    }
}

Patches label_patches(const std::int32_t* cls, i64 nx, i64 ny, int directions) {
    const i64 n = nx * ny;
    Patches out;
    out.strip_rows = kStripRows;
    const i64 n_strips = (ny + kStripRows - 1) / kStripRows;
    out.code.resize(static_cast<std::size_t>(n));
    std::int32_t* code = out.code.data();
    std::vector<StripStats> strips(static_cast<std::size_t>(n_strips));
    const bool parallel = n >= kParallelThreshold && n_strips > 1;
    parallel_for_dynamic(parallel, n_strips, [&](i64 s) {
        label_strip(cls, code, nx, ny, s * kStripRows, std::min(ny, (s + 1) * kStripRows), directions, strips[s]);
    });
    out.strip_offset.assign(static_cast<std::size_t>(n_strips + 1), 0);
    for (i64 s = 0; s < n_strips; ++s) {
        out.strip_offset[s + 1] = out.strip_offset[s] + static_cast<i64>(strips[s].cls.size());
    }
    const i64 n_local = out.strip_offset[n_strips];

    if (n_strips == 1) {
        // One strip (moving windows, most sample plots): its pieces are the
        // patches; only their order is left.
        StripStats& st = strips[0];
        const i64 m = n_local;
        std::vector<std::uint64_t> key(static_cast<std::size_t>(m));
        for (i64 i = 0; i < m; ++i) {
            key[i] = static_cast<std::uint64_t>(st.cls[i]) * static_cast<std::uint64_t>(n) +
                     static_cast<std::uint64_t>(st.first_cm[i]);
        }
        const std::vector<i64> order = sort_by_key(std::move(key));
        auto take = [&](auto& src, auto& dst) {
            dst.resize(static_cast<std::size_t>(m));
            for (i64 i = 0; i < m; ++i) dst[i] = src[order[i]];
        };
        out.n = m;
        take(st.cls, out.cls);
        take(st.cells, out.cells);
        take(st.edge_h, out.edge_h);
        take(st.edge_v, out.edge_v);
        take(st.adj_orth, out.adj_orth);
        take(st.adj_diag, out.adj_diag);
        take(st.sum_x, out.sum_x);
        take(st.sum_y, out.sum_y);
        take(st.first_cm, out.first_cm);
        out.local_to_patch.resize(static_cast<std::size_t>(m));
        for (i64 i = 0; i < m; ++i) out.local_to_patch[order[i]] = static_cast<std::int32_t>(i);
        if (directions == 4) count_diagonal_pairs(cls, nx, ny, out, false);
        return out;
    }

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

    // Patches: one per root piece, numbered in piece order, with the sums of
    // their pieces. The roots take their own piece's statistics in parallel;
    // the other pieces (those joined across a seam) are added afterwards.
    std::vector<std::int32_t> gid(static_cast<std::size_t>(n_local));
    std::vector<i64> roots_before(static_cast<std::size_t>(n_strips) + 1, 0);
    parallel_for(parallel, n_strips, [&](i64 s) {
        i64 roots = 0;
        for (i64 l = out.strip_offset[s]; l < out.strip_offset[s + 1]; ++l) {
            // find without path compression: safe in parallel
            std::int32_t r = static_cast<std::int32_t>(l);
            while (parent[r] != r) r = parent[r];
            gid[l] = r;  // the root, for now
            roots += r == l;
        }
        roots_before[s + 1] = roots;
    });
    for (i64 s = 0; s < n_strips; ++s) roots_before[s + 1] += roots_before[s];
    Patches tmp;
    tmp.n = roots_before[n_strips];
    for (auto* v : {&tmp.cells, &tmp.edge_h, &tmp.edge_v, &tmp.adj_orth, &tmp.adj_diag, &tmp.sum_x, &tmp.sum_y}) {
        v->resize(static_cast<std::size_t>(tmp.n));
    }
    tmp.cls.resize(static_cast<std::size_t>(tmp.n));
    std::vector<i64> first_cm(static_cast<std::size_t>(tmp.n));
    std::vector<std::int32_t> root_id(static_cast<std::size_t>(n_local), -1);
    parallel_for(parallel, n_strips, [&](i64 s) {
        const StripStats& st = strips[s];
        std::int32_t g = static_cast<std::int32_t>(roots_before[s]);
        for (std::size_t k = 0; k < st.cls.size(); ++k) {
            const i64 l = out.strip_offset[s] + static_cast<i64>(k);
            if (gid[l] != l) continue;
            root_id[l] = g;
            tmp.cls[g] = st.cls[k];
            tmp.cells[g] = st.cells[k];
            tmp.edge_h[g] = st.edge_h[k];
            tmp.edge_v[g] = st.edge_v[k];
            tmp.adj_orth[g] = st.adj_orth[k];
            tmp.adj_diag[g] = st.adj_diag[k];
            tmp.sum_x[g] = st.sum_x[k];
            tmp.sum_y[g] = st.sum_y[k];
            first_cm[g] = st.first_cm[k];
            ++g;
        }
    });
    for (i64 s = 0; s < n_strips; ++s) {
        const StripStats& st = strips[s];
        for (std::size_t k = 0; k < st.cls.size(); ++k) {
            const i64 l = out.strip_offset[s] + static_cast<i64>(k);
            if (gid[l] == l) continue;
            const std::int32_t g = root_id[gid[l]];
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
    parallel_for(parallel, n_local, [&](i64 l) { gid[l] = root_id[gid[l]]; });
    root_id = {};
    strips = {};
    parent = {};

    // landscapemetrics order: by class, then by first cell in column-major order.
    std::vector<std::uint64_t> key(static_cast<std::size_t>(tmp.n));
    parallel_for(parallel, tmp.n, [&](i64 i) {
        key[i] = static_cast<std::uint64_t>(tmp.cls[i]) * static_cast<std::uint64_t>(n) +
                 static_cast<std::uint64_t>(first_cm[i]);
    });
    const std::vector<i64> order = sort_by_key(std::move(key));
    out.n = tmp.n;
    auto permute = [&](const auto& src, auto& dst) {
        dst.resize(src.size());
        parallel_for(parallel, tmp.n, [&](i64 i) { dst[i] = src[order[i]]; });
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
    parallel_for(parallel, tmp.n, [&](i64 i) { rank[order[i]] = static_cast<std::int32_t>(i); });
    out.local_to_patch.resize(static_cast<std::size_t>(n_local));
    parallel_for(parallel, n_local, [&](i64 l) { out.local_to_patch[l] = rank[gid[l]]; });

    if (directions == 4) count_diagonal_pairs(cls, nx, ny, out, parallel);
    return out;
}

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

    // Class values in increasing order, mapped to dense indices: the value
    // range and a histogram of the values (per thread), then the map.
    const bool parallel = n >= kParallelThreshold;
    const i64 n_chunks = parallel ? std::min<i64>(256, n / 65536 + 1) : 1;
    std::vector<std::int32_t> chunk_min(static_cast<std::size_t>(n_chunks), std::numeric_limits<std::int32_t>::max());
    std::vector<std::int32_t> chunk_max(static_cast<std::size_t>(n_chunks), std::numeric_limits<std::int32_t>::min());
    parallel_for(parallel, n_chunks, [&](i64 c) {
        const i64 p0 = n * c / n_chunks;
        const i64 p1 = n * (c + 1) / n_chunks;
        std::int32_t lo = std::numeric_limits<std::int32_t>::max();
        std::int32_t hi = std::numeric_limits<std::int32_t>::min();
        if (na_mask) {
            for (i64 p = p0; p < p1; ++p) {
                if (na_mask[p]) continue;
                lo = std::min(lo, values[p]);
                hi = std::max(hi, values[p]);
            }
        } else {
            for (i64 p = p0; p < p1; ++p) {
                lo = std::min(lo, values[p]);
                hi = std::max(hi, values[p]);
            }
        }
        chunk_min[c] = lo;
        chunk_max[c] = hi;
    });
    const std::int32_t vmin = *std::min_element(chunk_min.begin(), chunk_min.end());
    const std::int32_t vmax = *std::max_element(chunk_max.begin(), chunk_max.end());
    cls_.resize(static_cast<std::size_t>(n));
    if (vmin > vmax) {  // all NA
        parallel_for(parallel, n, [&](i64 p) { cls_[p] = -1; });
        return;
    }
    if (vmin == std::numeric_limits<std::int32_t>::min()) {
        throw std::invalid_argument("class values must fit in a 32-bit integer (above -2147483648)");
    }

    const i64 range = static_cast<i64>(vmax) - vmin + 1;
    std::int32_t* cls = cls_.data();
    if (range <= (i64(1) << 16)) {
        std::vector<i64> count(static_cast<std::size_t>(range), 0);
        auto histogram = [&](i64 p0, i64 p1, i64* h) {
            for (i64 p = p0; p < p1; ++p) {
                if (!(na_mask && na_mask[p])) ++h[values[p] - static_cast<i64>(vmin)];
            }
        };
        if (parallel) {
#pragma omp parallel
            {
                std::vector<i64> local(static_cast<std::size_t>(range), 0);
#pragma omp for schedule(static)
                for (i64 c = 0; c < n_chunks; ++c) histogram(n * c / n_chunks, n * (c + 1) / n_chunks, local.data());
#pragma omp critical
                for (i64 v = 0; v < range; ++v) count[v] += local[v];
            }
        } else {
            histogram(0, n, count.data());
        }
        std::vector<std::int32_t> lut(static_cast<std::size_t>(range), -1);
        for (i64 v = 0; v < range; ++v) {
            if (count[v] == 0) continue;
            lut[v] = static_cast<std::int32_t>(class_values_.size());
            class_values_.push_back(static_cast<std::int32_t>(v + vmin));
            class_cells_.push_back(count[v]);
        }
        parallel_for(parallel, n, [&](i64 p) {
            cls[p] = na_mask && na_mask[p] ? -1 : lut[static_cast<std::size_t>(values[p] - static_cast<i64>(vmin))];
        });
    } else {
        // wide range: the distinct values of each chunk, merged
        std::vector<std::vector<std::int32_t>> distinct(static_cast<std::size_t>(n_chunks));
        parallel_for(parallel, n_chunks, [&](i64 c) {
            std::vector<std::int32_t>& d = distinct[c];
            for (i64 p = n * c / n_chunks, p1 = n * (c + 1) / n_chunks; p < p1; ++p) {
                if (!(na_mask && na_mask[p])) d.push_back(values[p]);
            }
            std::sort(d.begin(), d.end());
            d.erase(std::unique(d.begin(), d.end()), d.end());
        });
        std::vector<std::int32_t> vals;
        for (auto& d : distinct) {
            vals.insert(vals.end(), d.begin(), d.end());
            std::vector<std::int32_t>().swap(d);
        }
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        class_values_ = vals;
        parallel_for(parallel, n, [&](i64 p) {
            cls[p] = na_mask && na_mask[p]
                         ? -1
                         : static_cast<std::int32_t>(std::lower_bound(vals.begin(), vals.end(), values[p]) - vals.begin());
        });
        const auto k = static_cast<i64>(vals.size());
        class_cells_.assign(static_cast<std::size_t>(k), 0);
        if (parallel && k <= (i64(1) << 16)) {
#pragma omp parallel
            {
                std::vector<i64> local(static_cast<std::size_t>(k), 0);
#pragma omp for schedule(static)
                for (i64 p = 0; p < n; ++p) {
                    if (cls[p] >= 0) ++local[cls[p]];
                }
#pragma omp critical
                for (i64 c = 0; c < k; ++c) class_cells_[c] += local[c];
            }
        } else {
            for (i64 p = 0; p < n; ++p) {
                if (cls[p] >= 0) ++class_cells_[cls[p]];
            }
        }
    }
    valid_cells_ = std::accumulate(class_cells_.begin(), class_cells_.end(), i64(0));
}

Landscape::Landscape(i64 nx, i64 ny, double resx, double resy, Options options, std::vector<std::int32_t> class_values,
                     std::vector<i64> class_cells, Patches patches, Adjacency adjacency)
    : nx_(nx), ny_(ny), resx_(resx), resy_(resy), options_(std::move(options)),
      class_values_(std::move(class_values)), class_cells_(std::move(class_cells)), tables_only_(true),
      patches_(std::move(patches)), adjacency_(std::move(adjacency)) {
    valid_cells_ = std::accumulate(class_cells_.begin(), class_cells_.end(), i64(0));
}

void Landscape::set_patch_values(const std::string& name, std::vector<double> values) {
    if (static_cast<i64>(values.size()) != patches().n) throw std::invalid_argument("one value per patch expected");
    patch_cache_[name] = std::move(values);
    stat_done_.fill(false);
    class_cache_.clear();
    landscape_cache_.clear();
}

const Patches& Landscape::patches() {
    if (!patches_) patches_ = label_patches(cls_.data(), nx_, ny_, options_.directions);
    return *patches_;
}

std::vector<std::int32_t> Landscape::labels() {
    if (tables_only_) throw std::logic_error("patch labels of an out-of-core landscape are not kept");
    const Patches& pt = patches();
    const i64 n = nx_ * ny_;
    std::vector<std::int32_t> out(static_cast<std::size_t>(n));
    parallel_for(n >= kParallelThreshold, ny_, [&](i64 y) {
        for (i64 x = 0; x < nx_; ++x) {
            const i64 p = y * nx_ + x;
            out[p] = pt.patch(p, y) + 1;
        }
    });
    return out;
}

std::vector<std::int32_t> city_block_distance(const std::int32_t* cls, i64 nx, i64 ny, std::int32_t top,
                                              std::int32_t bottom, std::int32_t sides) {
    const i64 n = nx * ny;
    const bool parallel = n >= kParallelThreshold;
    constexpr std::int32_t kFar = std::numeric_limits<std::int32_t>::max() / 2;
    // Along rows, then along columns. A path through a cell of another class
    // would pass a nearer source, so a neighbour of another class is at 1.
    std::vector<std::int32_t> dist(static_cast<std::size_t>(n));
    std::int32_t* d = dist.data();
    parallel_for(parallel, ny, [&](i64 y) {
        const i64 row = y * nx;
        for (i64 x = 0; x < nx; ++x) {
            const i64 p = row + x;
            if (cls[p] < 0) continue;
            d[p] = x == 0 ? sides : (cls[p - 1] != cls[p] ? 1 : std::min(kFar, d[p - 1] + 1));
        }
        for (i64 x = nx - 1; x >= 0; --x) {
            const i64 p = row + x;
            if (cls[p] < 0) continue;
            const std::int32_t r = x == nx - 1 ? sides : (cls[p + 1] != cls[p] ? 1 : std::min(kFar, d[p + 1] + 1));
            d[p] = std::min(d[p], r);
        }
    });
    constexpr i64 kBlock = 256;
    const i64 n_blocks = (nx + kBlock - 1) / kBlock;
    parallel_for_dynamic(parallel && n_blocks > 1, n_blocks, [&](i64 b) {
        const i64 x0 = b * kBlock;
        const i64 x1 = std::min(nx, x0 + kBlock);
        for (i64 y = 0; y < ny; ++y) {
            for (i64 x = x0; x < x1; ++x) {
                const i64 p = y * nx + x;
                if (cls[p] < 0) continue;
                const std::int32_t u = y == 0 ? top : (cls[p - nx] != cls[p] ? 1 : std::min(kFar, d[p - nx] + 1));
                d[p] = std::min(d[p], u);
            }
        }
        for (i64 y = ny - 1; y >= 0; --y) {
            for (i64 x = x0; x < x1; ++x) {
                const i64 p = y * nx + x;
                if (cls[p] < 0) continue;
                const std::int32_t v = y == ny - 1 ? bottom : (cls[p + nx] != cls[p] ? 1 : std::min(kFar, d[p + nx] + 1));
                d[p] = std::min(d[p], v);
            }
        }
    });
    return dist;
}

const Core& Landscape::core() {
    if (core_) return *core_;
    if (tables_only_) throw std::logic_error("core areas of an out-of-core landscape need a detail pass");
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
    std::vector<std::int32_t> dist = city_block_distance(cls, nx, ny, outside, outside, outside);
    const std::int32_t* d = dist.data();

    // Core cells per patch, summed per strip piece and then per patch.
    std::vector<std::int32_t> core_cls(static_cast<std::size_t>(n), -1);
    const i64 n_strips = static_cast<i64>(pt.strip_offset.size()) - 1;
    std::vector<std::vector<i64>> pieces(static_cast<std::size_t>(n_strips));
    parallel_for_dynamic(parallel && n_strips > 1, n_strips, [&](i64 s) {
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
    });
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
    const Patches cores = label_patches(core_cls.data(), nx, ny, options_.directions);
    for (i64 i = 0; i < cores.n; ++i) {
        const i64 x = cores.first_cm[i] / ny;
        const i64 y = cores.first_cm[i] % ny;
        ++out.areas[pt.patch(y * nx + x, y)];
    }
    out.area_cells = cores.cells;
    out.area_class = cores.cls;
    out.has_area_cells = true;
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
    auto count_row = [&](i64 y) {
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
    };
    if (n_threads > 1) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (i64 y = 0; y < ny; ++y) count_row(y);
    } else {
        for (i64 y = 0; y < ny; ++y) count_row(y);
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
