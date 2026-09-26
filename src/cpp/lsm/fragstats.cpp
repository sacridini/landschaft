// SPDX-License-Identifier: GPL-3.0-or-later
//
// FRAGSTATS metrics that landscapemetrics does not have: proximity (PROX)
// and similarity (SIMI) indices, connectance (CONNECT) and the edge-contrast
// metrics (ECON, CWED, TECI), following the FRAGSTATS documentation.
// Distances between patches are the shortest distances between their cell
// centres, as FRAGSTATS (and the enn metric) measure them.
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "lsm/landscape.hpp"
#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"

namespace lsm {

namespace {

constexpr i64 kParallel = i64(1) << 16;

struct PairHit {
    std::int32_t b;
    double dd;
};

// Smallest value per key (patch index), open addressing, cleared cheaply.
class FlatMin {
public:
    FlatMin() : keys_(64, -1), vals_(64) {}
    void clear() {
        for (std::size_t slot : used_) keys_[slot] = -1;
        used_.clear();
    }
    void update(std::int32_t key, double v) {
        if (used_.size() * 2 >= keys_.size()) grow();
        std::size_t slot = hash(key);
        while (keys_[slot] != -1 && keys_[slot] != key) slot = (slot + 1) & (keys_.size() - 1);
        if (keys_[slot] == -1) {
            keys_[slot] = key;
            vals_[slot] = v;
            used_.push_back(slot);
        } else if (v < vals_[slot]) {
            vals_[slot] = v;
        }
    }
    void collect(std::vector<PairHit>& out) const {
        for (std::size_t slot : used_) out.push_back({keys_[slot], vals_[slot]});
    }

private:
    std::size_t hash(std::int32_t key) const {
        return (static_cast<std::uint32_t>(key) * 2654435761u) & (keys_.size() - 1);
    }
    void grow() {
        std::vector<PairHit> items;
        collect(items);
        keys_.assign(keys_.size() * 2, -1);
        vals_.resize(keys_.size());
        used_.clear();
        for (const PairHit& h : items) update(h.b, h.dd);
    }
    std::vector<std::int32_t> keys_;
    std::vector<double> vals_;
    std::vector<std::size_t> used_;
};

}  // namespace

// Dense class x class matrix of a (class a, class b, weight) table; a pair
// given in one order counts for both. Missing pairs are NaN, except the
// diagonal, which is `diagonal`.
std::vector<double> Landscape::class_weights(const std::vector<std::array<double, 3>>& table, double diagonal) const {
    const int k = n_classes();
    std::vector<double> w(static_cast<std::size_t>(k) * k, kNaN);
    for (int c = 0; c < k; ++c) w[static_cast<std::size_t>(c) * k + c] = diagonal;
    auto index = [&](double v) -> int {
        const auto it = std::lower_bound(class_values_.begin(), class_values_.end(), static_cast<std::int32_t>(v));
        if (it == class_values_.end() || *it != v) return -1;
        return static_cast<int>(it - class_values_.begin());
    };
    for (const auto& t : table) {
        const int a = index(t[0]);
        const int b = index(t[1]);
        if (a < 0 || b < 0) continue;
        w[static_cast<std::size_t>(a) * k + b] = t[2];
        if (std::isnan(w[static_cast<std::size_t>(b) * k + a]) || a == b) w[static_cast<std::size_t>(b) * k + a] = t[2];
    }
    // an explicit (b, a) wins over the mirrored (a, b)
    for (const auto& t : table) {
        const int a = index(t[0]);
        const int b = index(t[1]);
        if (a >= 0 && b >= 0) w[static_cast<std::size_t>(a) * k + b] = t[2];
    }
    return w;
}

// Lengths of a left/right and of a top/bottom cell side, as the patch
// perimeter uses them (landscapemetrics' convention or the physical one).
std::pair<double, double> Landscape::side_lengths() const {
    if (options_.fragstats) return {resy_, resx_};
    if (res_all_equal()) return {resx_, resx_};
    return {resx_, resy_};
}

// ECON: contrast-weighted perimeter over perimeter, in %. Sides towards
// another class weigh the contrast between the classes, sides towards NA or
// the map edge `boundary_contrast`.
std::vector<double> Landscape::econ() {
    const Patches& pt = patches();
    if (options_.contrast.empty()) return std::vector<double>(static_cast<std::size_t>(pt.n), kNaN);
    const int k = n_classes();
    const std::vector<double> d = class_weights(options_.contrast, 0.0);
    const double db = options_.boundary_contrast;
    const i64 nx = nx_;
    const i64 ny = ny_;
    const i64 n_strips = static_cast<i64>(pt.strip_offset.size()) - 1;
    std::vector<std::vector<double>> wh(static_cast<std::size_t>(n_strips)), wv(static_cast<std::size_t>(n_strips));
    parallel_for_dynamic(nx * ny >= kParallel && n_strips > 1, n_strips, [&](i64 s) {
        const i64 off = pt.strip_offset[s];
        wh[s].assign(static_cast<std::size_t>(pt.strip_offset[s + 1] - off), 0.0);
        wv[s].assign(static_cast<std::size_t>(pt.strip_offset[s + 1] - off), 0.0);
        const i64 y1 = std::min(ny, (s + 1) * pt.strip_rows);
        for (i64 y = s * pt.strip_rows; y < y1; ++y) {
            for (i64 x = 0; x < nx; ++x) {
                const i64 p = y * nx + x;
                const std::int32_t c = cls_[p];
                if (c < 0) continue;
                const i64 piece = -static_cast<i64>(pt.code[p]) - 2;
                auto w = [&](i64 xx, i64 yy) {
                    if (xx < 0 || yy < 0 || xx >= nx || yy >= ny) return db;
                    const std::int32_t b = cls_[yy * nx + xx];
                    if (b < 0) return db;
                    return b == c ? 0.0 : d[static_cast<std::size_t>(c) * k + b];
                };
                wh[s][piece] += w(x - 1, y) + w(x + 1, y);
                wv[s][piece] += w(x, y - 1) + w(x, y + 1);
            }
        }
    });
    std::vector<double> h(static_cast<std::size_t>(pt.n), 0.0), v(static_cast<std::size_t>(pt.n), 0.0);
    for (i64 s = 0; s < n_strips; ++s) {
        for (std::size_t j = 0; j < wh[s].size(); ++j) {
            const std::int32_t patch = pt.local_to_patch[pt.strip_offset[s] + static_cast<i64>(j)];
            h[patch] += wh[s][j];
            v[patch] += wv[s][j];
        }
    }
    const auto [lh, lv] = side_lengths();
    const std::vector<double>& perim = patch_base("perim");
    std::vector<double> out(static_cast<std::size_t>(pt.n));
    for (i64 i = 0; i < pt.n; ++i) out[i] = (h[i] * lh + v[i] * lv) / perim[i] * 100;
    return out;
}

// Pairs of patches (a < b) whose cells come within `radius` of each other
// (cell centre to cell centre), with that shortest distance: of the same
// class only, or of any classes. Found from the boundary cells, sorted by
// (column, row), with a range search around each.
const PatchPairs& Landscape::patch_pairs(double radius, bool same_class) {
    const std::string key = std::to_string(radius) + (same_class ? "/same" : "/all");
    if (auto it = pair_cache_.find(key); it != pair_cache_.end()) return it->second;
    const Patches& pt = patches();
    const i64 nx = nx_;
    const i64 ny = ny_;
    auto boundary = [&](i64 x, i64 y) {
        const i64 p = y * nx + x;
        const std::int32_t c = cls_[p];
        return x == 0 || y == 0 || x == nx - 1 || y == ny - 1 || cls_[p - 1] != c || cls_[p + 1] != c ||
               cls_[p - nx] != c || cls_[p + nx] != c;
    };
    // boundary cells by column, then row
    std::vector<std::int32_t> col, row, sid;
    std::vector<i64> col_start(static_cast<std::size_t>(nx) + 1, 0);
    for (i64 x = 0; x < nx; ++x) {
        col_start[x] = static_cast<i64>(col.size());
        for (i64 y = 0; y < ny; ++y) {
            if (cls_[y * nx + x] < 0 || !boundary(x, y)) continue;
            col.push_back(static_cast<std::int32_t>(x));
            row.push_back(static_cast<std::int32_t>(y));
            sid.push_back(pt.patch(y * nx + x, y));
        }
    }
    col_start[nx] = static_cast<i64>(col.size());
    const i64 m = static_cast<i64>(col.size());
    const double rx = resx_;
    const double ry = resy_;
    const i64 cx = static_cast<i64>(std::ceil(radius / rx));
    const i64 cy = static_cast<i64>(std::ceil(radius / ry));
    const double r2 = radius * radius;
    // boundary cells grouped by patch
    std::vector<i64> by_patch_start(static_cast<std::size_t>(pt.n) + 1, 0);
    for (i64 i = 0; i < m; ++i) ++by_patch_start[sid[i] + 1];
    for (i64 i = 0; i < pt.n; ++i) by_patch_start[i + 1] += by_patch_start[i];
    std::vector<i64> by_patch(static_cast<std::size_t>(m));
    {
        std::vector<i64> at(by_patch_start.begin(), by_patch_start.end() - 1);
        for (i64 i = 0; i < m; ++i) by_patch[at[sid[i]]++] = i;
    }
    std::vector<std::vector<PairHit>> found(static_cast<std::size_t>(pt.n));
#pragma omp parallel if (m >= 4096)
    {
        FlatMin nearest;  // patch -> smallest squared distance, for one focal patch
#pragma omp for schedule(dynamic, 64)
        for (i64 a = 0; a < pt.n; ++a) {
            nearest.clear();
            for (i64 t = by_patch_start[a]; t < by_patch_start[a + 1]; ++t) {
                const i64 i = by_patch[t];
                const double xi = static_cast<double>(col[i] + 1) * rx;
                const double yi = static_cast<double>(row[i] + 1) * ry;
                const i64 x0 = std::max<i64>(0, col[i] - cx);
                const i64 x1 = std::min<i64>(nx - 1, col[i] + cx);
                for (i64 x = x0; x <= x1; ++x) {
                    const i64 s = col_start[x];
                    const i64 e = col_start[x + 1];
                    i64 j = std::lower_bound(row.begin() + s, row.begin() + e,
                                             static_cast<std::int32_t>(row[i] - cy)) - row.begin();
                    const double dx = static_cast<double>(x + 1) * rx - xi;
                    const double dx2 = dx * dx;
                    if (dx2 > r2) continue;
                    for (; j < e && row[j] <= row[i] + cy; ++j) {
                        const std::int32_t b = sid[j];
                        if (b <= a) continue;  // each pair once, from its smaller patch
                        if (same_class && pt.cls[a] != pt.cls[b]) continue;
                        const double dy = static_cast<double>(row[j] + 1) * ry - yi;
                        const double dd = dx2 + dy * dy;
                        if (dd <= r2) nearest.update(b, dd);
                    }
                }
            }
            nearest.collect(found[a]);
        }
    }
    PatchPairs out;
    for (i64 a = 0; a < pt.n; ++a) {
        std::sort(found[a].begin(), found[a].end(), [](const PairHit& u, const PairHit& v) { return u.b < v.b; });
        for (const PairHit& h : found[a]) {
            out.a.push_back(static_cast<std::int32_t>(a));
            out.b.push_back(h.b);
            out.dist.push_back(std::sqrt(h.dd));
        }
    }
    return pair_cache_.emplace(key, std::move(out)).first->second;
}

// PROX (similarity = false): sum over the other patches of the class within
// search_radius of area (m2) / distance^2. SIMI: the same over the patches
// of every class, each weighted by the similarity of its class to the
// patch's class (1 on the diagonal; SIMI = PROX without a table).
std::vector<double> Landscape::proximity(bool similarity) {
    const Patches& pt = patches();
    const double radius = options_.search_radius;
    std::vector<double> out(static_cast<std::size_t>(pt.n), kNaN);
    if (!(radius > 0)) return out;
    std::fill(out.begin(), out.end(), 0.0);
    const int k = n_classes();
    std::vector<double> sim;
    if (similarity) {
        sim = options_.similarity.empty() ? std::vector<double>(static_cast<std::size_t>(k) * k, 0.0)
                                          : class_weights(options_.similarity, 1.0);
        for (int c = 0; c < k; ++c) sim[static_cast<std::size_t>(c) * k + c] = 1.0;
    }
    const PatchPairs& pairs = patch_pairs(radius, !similarity);
    const double cell = resx_ * resy_;
    for (std::size_t i = 0; i < pairs.a.size(); ++i) {
        const std::int32_t a = pairs.a[i];
        const std::int32_t b = pairs.b[i];
        const double h2 = pairs.dist[i] * pairs.dist[i];
        double wab = 1.0, wba = 1.0;
        if (similarity) {
            wab = sim[static_cast<std::size_t>(pt.cls[a]) * k + pt.cls[b]];
            wba = sim[static_cast<std::size_t>(pt.cls[b]) * k + pt.cls[a]];
        }
        out[a] += static_cast<double>(pt.cells[b]) * cell * wab / h2;
        out[b] += static_cast<double>(pt.cells[a]) * cell * wba / h2;
    }
    return out;
}

// CONNECT: the share (%) of the pairs of patches of a class that are within
// connect_threshold of each other; per class, or over all classes.
std::vector<double> Landscape::connectance(bool by_class) {
    const Patches& pt = patches();
    const int k = n_classes();
    const double threshold = options_.connect_threshold;
    std::vector<double> joined(static_cast<std::size_t>(k), 0.0);
    std::vector<double> npatch(static_cast<std::size_t>(k), 0.0);
    for (i64 i = 0; i < pt.n; ++i) ++npatch[pt.cls[i]];
    if (!(threshold >= 0)) return std::vector<double>(by_class ? k : 1, kNaN);
    const PatchPairs& pairs = patch_pairs(threshold, true);
    for (std::size_t i = 0; i < pairs.a.size(); ++i) {
        if (pairs.dist[i] <= threshold) ++joined[pt.cls[pairs.a[i]]];
    }
    if (by_class) {
        std::vector<double> out(static_cast<std::size_t>(k));
        for (int c = 0; c < k; ++c) {
            out[c] = npatch[c] < 2 ? kNaN : joined[c] / (npatch[c] * (npatch[c] - 1) / 2) * 100;
        }
        return out;
    }
    double j = 0, all = 0;
    for (int c = 0; c < k; ++c) {
        j += joined[c];
        all += npatch[c] * (npatch[c] - 1) / 2;
    }
    return {all > 0 ? j / all * 100 : kNaN};
}

// CWED (m per ha) and TECI (%) per class: contrast-weighted edge lengths
// between classes (from the class adjacencies); the edges to NA and the map
// edge weigh boundary_contrast and count in CWED with count_boundary, in
// TECI always (FRAGSTATS' E*).
std::vector<double> Landscape::contrast_edges(bool teci, bool by_class) {
    const Adjacency& adj = adjacency();
    const int k = n_classes();
    if (options_.contrast.empty()) return std::vector<double>(by_class ? k : 1, kNaN);
    const std::vector<double> d = class_weights(options_.contrast, 0.0);
    const double db = options_.boundary_contrast;
    const double ta = total_area();
    auto e = [&](int a, int b) { return edge_length(adj.h[a * adj.dim + b], adj.v[a * adj.dim + b]); };
    auto boundary = [&](int a) {
        return edge_length(adj.h[a * adj.dim + k] + adj.h[a * adj.dim + k + 1],
                           adj.v[a * adj.dim + k] + adj.v[a * adj.dim + k + 1]);
    };
    const bool with_boundary = teci || options_.count_boundary;
    if (by_class) {
        std::vector<double> out(static_cast<std::size_t>(k));
        for (int a = 0; a < k; ++a) {
            double weighted = 0, total = 0;
            for (int b = 0; b < k; ++b) {
                if (b == a) continue;
                weighted += e(a, b) * d[static_cast<std::size_t>(a) * k + b];
                total += e(a, b);
            }
            if (with_boundary) {
                weighted += boundary(a) * db;
                total += boundary(a);
            }
            out[a] = teci ? (total > 0 ? weighted / total * 100 : kNaN) : weighted / ta;
        }
        return out;
    }
    double weighted = 0, total = 0;
    for (int a = 0; a < k; ++a) {
        for (int b = 0; b < a; ++b) {
            weighted += e(a, b) * d[static_cast<std::size_t>(a) * k + b];
            total += e(a, b);
        }
        if (with_boundary) {
            weighted += boundary(a) * db;
            total += boundary(a);
        }
    }
    return {teci ? (total > 0 ? weighted / total * 100 : kNaN) : weighted / ta};
}

// FRAGSTATS' DCORE statistics (convention = "fragstats"): over the sizes
// (ha) of the disjunct core areas of the class (or the landscape), the
// area-weighted mean weighted by those sizes. NaN without core areas.
std::vector<double> Landscape::dcore_sizes(const std::string& stat, bool by_class) {
    const Core& core = this->core();
    if (!core.has_area_cells) throw std::logic_error("disjunct core area sizes are not kept out of core");
    const int k = n_classes();
    const double cell = resx_ * resy_ / 10000;
    std::vector<double> x(core.area_cells.size());
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(core.area_cells[i]) * cell;
    std::vector<i64> start(static_cast<std::size_t>(k) + 1, 0);
    for (std::int32_t c : core.area_class) ++start[c + 1];
    for (int c = 0; c < k; ++c) start[c + 1] += start[c];
    if (!by_class) start = {0, static_cast<i64>(x.size())};
    const i64 groups = static_cast<i64>(start.size()) - 1;
    std::vector<double> out(static_cast<std::size_t>(groups));
    for (i64 g = 0; g < groups; ++g) {
        const double* v = x.data() + start[g];
        const i64 n = start[g + 1] - start[g];
        const double mean = r_mean(v, n);
        const double sd = r_sd(v, n, mean, true);
        if (stat == "mn") {
            out[g] = mean;
        } else if (stat == "sd") {
            out[g] = sd;
        } else if (stat == "cv") {
            out[g] = sd / mean * 100;
        } else {
            out[g] = n == 0 ? kNaN : reduce_statistic(stat, v, n, v);
        }
    }
    return out;
}

}  // namespace lsm
