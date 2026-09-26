// SPDX-License-Identifier: GPL-3.0-or-later
//
// Patch metrics that need the position of the cells, not only per-patch
// sums: the Euclidean nearest-neighbour distance (enn, landscapemetrics'
// lsm_p_enn) and the related circumscribing circle (circle, lsm_p_circle).
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lsm/landscape.hpp"

namespace lsm {

namespace {

constexpr i64 kParallelThreshold = i64(1) << 16;
constexpr double kPi = 3.14159265358979323846;

// -- nearest neighbour --------------------------------------------------------

// Interleaved bits of x (even) and y (odd): the Morton (Z-order) code.
std::uint64_t morton(std::uint32_t x, std::uint32_t y) {
    auto spread = [](std::uint64_t v) {
        v = (v | (v << 16)) & 0x0000FFFF0000FFFFULL;
        v = (v | (v << 8)) & 0x00FF00FF00FF00FFULL;
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0FULL;
        v = (v | (v << 2)) & 0x3333333333333333ULL;
        return (v | (v << 1)) & 0x5555555555555555ULL;
    };
    return spread(x) | (spread(y) << 1);
}

std::uint32_t morton_part(std::uint64_t v) {
    v &= 0x5555555555555555ULL;
    v = (v | (v >> 1)) & 0x3333333333333333ULL;
    v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0FULL;
    v = (v | (v >> 4)) & 0x00FF00FF00FF00FFULL;
    v = (v | (v >> 8)) & 0x0000FFFF0000FFFFULL;
    return static_cast<std::uint32_t>((v | (v >> 16)) & 0xFFFFFFFFULL);
}

int highest_bit(std::uint64_t v) {
    int b = 0;
    for (int s = 32; s > 0; s >>= 1) {
        if (v >> s) {
            v >>= s;
            b += s;
        }
    }
    return b;
}

// Edge cells of all classes, sorted by class and then by Morton code.
struct EdgeCells {
    std::int64_t n = 0;
    std::unique_ptr<std::uint64_t[]> key;  // Morton code (freed once the trees are built)
    std::unique_ptr<std::int32_t[]> patch;
    std::unique_ptr<std::int32_t[]> x, y;
};

// Tree over the edge cells [begin, end) of one class, split like a quadtree:
// every node splits its range where the highest differing bit of the Morton
// codes changes (the cells do not move). Nodes whose cells all belong to one
// patch are skipped when searching the neighbours of that patch, so a search
// never walks through the cells of its own patch.
class CellTree {
public:
    struct Node {
        std::int32_t x0, x1, y0, y1;  // bounding box
        std::int32_t patch;           // patch of all its cells, -1 if several
        std::int32_t begin, end;
        std::int32_t left, right;     // -1: leaf
    };
    using Job = std::array<std::int32_t, 2>;  // cell range of a subtree

    CellTree() = default;
    // Builds the top of the tree over the cells [begin, end) and appends the
    // ranges of the subtrees below it to `jobs`; build_job builds those (on
    // any thread) and finish joins them.
    CellTree(const EdgeCells& cells, std::int32_t begin, std::int32_t end, const std::vector<double>& X,
             const std::vector<double>& Y, std::vector<Job>& jobs)
        : cells_(&cells), X_(&X), Y_(&Y), first_job_(static_cast<std::int64_t>(jobs.size())) {
        top(begin, end, jobs);
        n_jobs_ = static_cast<std::int64_t>(jobs.size()) - first_job_;
    }

    std::vector<Node> build_job(const Job& job) const {
        std::vector<Node> out;
        out.reserve(static_cast<std::size_t>((job[1] - job[0]) / (kLeafSize / 2) + 1));
        build(job[0], job[1], out);
        return out;
    }

    // Joins the subtrees of this tree's jobs (sub[i] for job i of all trees);
    // the root ends up at node 0.
    void finish(std::vector<std::vector<Node>>& sub) {
        const auto n_top = static_cast<std::int32_t>(nodes_.size());
        std::vector<std::int32_t> offset(static_cast<std::size_t>(n_jobs_));
        std::size_t total = nodes_.size();
        for (std::int64_t j = 0; j < n_jobs_; ++j) {
            offset[j] = static_cast<std::int32_t>(total);
            total += sub[first_job_ + j].size();
        }
        nodes_.resize(total);
        for (std::int64_t j = 0; j < n_jobs_; ++j) {
            Node* out = nodes_.data() + offset[j];
            for (Node nd : sub[first_job_ + j]) {
                if (nd.left >= 0) {
                    nd.left += offset[j];
                    nd.right += offset[j];
                }
                *out++ = nd;
            }
            sub[first_job_ + j] = {};
        }
        // children of top nodes come after them: finish the top bottom-up
        for (std::int32_t i = n_top - 1; i >= 0; --i) {
            Node& nd = nodes_[i];
            if (nd.left < -1) nd.left = offset[-nd.left - 2];
            if (nd.right < -1) nd.right = offset[-nd.right - 2];
            join(nd, nodes_[nd.left], nodes_[nd.right]);
        }
    }

    static constexpr std::int32_t kLeafSize = 32;

    // Appends the cell ranges of the leaves.
    void leaves(std::int32_t tree, std::vector<std::array<std::int32_t, 3>>& out) const {
        for (const Node& nd : nodes_) {
            if (nd.left < 0) out.push_back({tree, nd.begin, nd.end});
        }
    }

    // Lowers best2 to the smallest squared distance from the cells idx[0..count)
    // of patch `patch` (bounding box g) to a cell of another patch.
    void nearest(std::int32_t patch, const std::int32_t* idx, int count, const std::array<std::int32_t, 4>& g,
                 double& best2) const {
        const double* X = X_->data();
        const double* Y = Y_->data();
        const std::int32_t* cx = cells_->x.get();
        const std::int32_t* cy = cells_->y.get();
        const std::int32_t* cp = cells_->patch.get();
        std::int32_t stack[160];  // depth <= 64: one pending sibling per level
        int top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const Node& nd = nodes_[stack[--top]];
            if (nd.patch == patch || bound2(nd, g) >= best2) continue;
            if (nd.left < 0) {
                for (std::int32_t i = nd.begin; i < nd.end; ++i) {
                    if (cp[i] == patch) continue;
                    for (int a = 0; a < count; ++a) {
                        // as landscapemetrics' compute_d2
                        const double dx = X[cx[i]] - X[cx[idx[a]]];
                        const double dy = Y[cy[i]] - Y[cy[idx[a]]];
                        const double d2 = dx * dx + dy * dy;
                        if (d2 < best2) best2 = d2;
                    }
                }
                continue;
            }
            // nearer child on top of the stack
            const bool left_first = bound2(nodes_[nd.left], g) <= bound2(nodes_[nd.right], g);
            stack[top++] = left_first ? nd.right : nd.left;
            stack[top++] = left_first ? nd.left : nd.right;
        }
    }

private:
    static constexpr std::int32_t kParallelBuild = 1 << 14;  // cells of a subtree built by one thread

    // Lower bound of the squared distance between cells in box g (x0, x1,
    // y0, y1) and the node's cells; made of the same coordinate differences
    // as the distances, so it never exceeds them after rounding.
    double bound2(const Node& nd, const std::array<std::int32_t, 4>& g) const {
        const double* X = X_->data();
        const double* Y = Y_->data();
        const double dx = g[1] < nd.x0 ? X[nd.x0] - X[g[1]] : (g[0] > nd.x1 ? X[g[0]] - X[nd.x1] : 0.0);
        const double dy = g[3] < nd.y0 ? Y[nd.y0] - Y[g[3]] : (g[2] > nd.y1 ? Y[g[2]] - Y[nd.y1] : 0.0);
        return dx * dx + dy * dy;
    }

    // First cell of the upper part of a range: where the highest bit in
    // which its (distinct, sorted) codes differ becomes 1.
    std::int32_t split(std::int32_t begin, std::int32_t end) const {
        const std::uint64_t* key = cells_->key.get();
        const int hb = highest_bit(key[begin] ^ key[end - 1]);
        const std::uint64_t target = ((key[begin] >> hb) | 1) << hb;
        return static_cast<std::int32_t>(std::lower_bound(key + begin, key + end, target) - key);
    }

    static void join(Node& nd, const Node& a, const Node& b) {
        nd.x0 = std::min(a.x0, b.x0);
        nd.x1 = std::max(a.x1, b.x1);
        nd.y0 = std::min(a.y0, b.y0);
        nd.y1 = std::max(a.y1, b.y1);
        nd.patch = a.patch == b.patch ? a.patch : -1;
    }

    // Subtree over cells [begin, end) appended to `out`; returns its root.
    std::int32_t build(std::int32_t begin, std::int32_t end, std::vector<Node>& out) const {
        const auto idx = static_cast<std::int32_t>(out.size());
        out.emplace_back();
        Node nd;
        nd.begin = begin;
        nd.end = end;
        if (end - begin <= kLeafSize) {
            nd.left = nd.right = -1;
            const std::int32_t* cx = cells_->x.get();
            const std::int32_t* cy = cells_->y.get();
            const std::int32_t* cp = cells_->patch.get();
            nd.x0 = nd.y0 = std::numeric_limits<std::int32_t>::max();
            nd.x1 = nd.y1 = -1;
            nd.patch = cp[begin];
            for (std::int32_t i = begin; i < end; ++i) {
                nd.x0 = std::min(nd.x0, cx[i]);
                nd.x1 = std::max(nd.x1, cx[i]);
                nd.y0 = std::min(nd.y0, cy[i]);
                nd.y1 = std::max(nd.y1, cy[i]);
                if (cp[i] != nd.patch) nd.patch = -1;
            }
        } else {
            const std::int32_t mid = split(begin, end);
            nd.left = build(begin, mid, out);
            nd.right = build(mid, end, out);
            join(nd, out[nd.left], out[nd.right]);
        }
        out[idx] = nd;
        return idx;
    }

    // Top nodes (ranges of at least kParallelBuild cells) in pre-order; the
    // smaller ranges below them become jobs, referenced as -(job of this
    // tree + 2).
    std::int32_t top(std::int32_t begin, std::int32_t end, std::vector<Job>& jobs) {
        if (end - begin < kParallelBuild) {
            jobs.push_back({begin, end});
            return -static_cast<std::int32_t>(static_cast<std::int64_t>(jobs.size()) - first_job_) - 1;
        }
        const auto idx = static_cast<std::int32_t>(nodes_.size());
        nodes_.emplace_back();
        const std::int32_t mid = split(begin, end);
        const std::int32_t left = top(begin, mid, jobs);
        const std::int32_t right = top(mid, end, jobs);
        nodes_[idx].begin = begin;
        nodes_[idx].end = end;
        nodes_[idx].left = left;
        nodes_[idx].right = right;
        return idx;
    }

    const EdgeCells* cells_ = nullptr;
    std::vector<Node> nodes_;
    std::int64_t first_job_ = 0;
    std::int64_t n_jobs_ = 0;
    const std::vector<double>* X_ = nullptr;
    const std::vector<double>* Y_ = nullptr;
};

// One stable pass of an LSD radix sort of (key, patch) pairs on the digit
// digit(key, patch) in [0, n_buckets), counting and scattering blocks of
// pairs in parallel.
template <class Digit>
void radix_pass(std::int64_t n, int n_buckets, bool parallel, std::unique_ptr<std::uint64_t[]>& key,
                std::unique_ptr<std::int32_t[]>& patch, std::unique_ptr<std::uint64_t[]>& key_tmp,
                std::unique_ptr<std::int32_t[]>& patch_tmp, Digit digit) {
    // blocks of at least 32768 pairs, and at most 2^22 counters in all
    const std::int64_t n_blocks =
        parallel ? std::max<std::int64_t>(1, std::min({std::int64_t(256), n / 32768, (std::int64_t(1) << 22) / n_buckets}))
                 : 1;
    const std::int64_t block = (n + n_blocks - 1) / n_blocks;
    std::vector<std::int64_t> count(static_cast<std::size_t>(n_blocks * n_buckets));
#pragma omp parallel for schedule(static) if (n_blocks > 1)
    for (std::int64_t b = 0; b < n_blocks; ++b) {
        std::int64_t* cnt = count.data() + b * n_buckets;
        std::fill(cnt, cnt + n_buckets, 0);
        const std::int64_t end = std::min(n, (b + 1) * block);
        for (std::int64_t i = b * block; i < end; ++i) ++cnt[digit(key[i], patch[i])];
    }
    std::int64_t sum = 0;
    for (int d = 0; d < n_buckets; ++d) {
        for (std::int64_t b = 0; b < n_blocks; ++b) {
            std::int64_t& c = count[b * n_buckets + d];
            const std::int64_t v = c;
            c = sum;
            sum += v;
        }
    }
#pragma omp parallel for schedule(static) if (n_blocks > 1)
    for (std::int64_t b = 0; b < n_blocks; ++b) {
        std::int64_t* cnt = count.data() + b * n_buckets;
        const std::int64_t end = std::min(n, (b + 1) * block);
        for (std::int64_t i = b * block; i < end; ++i) {
            const std::int64_t j = cnt[digit(key[i], patch[i])]++;
            key_tmp[j] = key[i];
            patch_tmp[j] = patch[i];
        }
    }
    key.swap(key_tmp);
    patch.swap(patch_tmp);
}

std::uint64_t double_bits(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

double bits_double(std::uint64_t b) {
    double v;
    std::memcpy(&v, &b, sizeof v);
    return v;
}

// -- smallest enclosing circle ----------------------------------------------
//
// Welzl's algorithm as in landscapemetrics (Project Nayuki's library), with a
// fixed per-patch shuffle instead of a random one so results are repeatable.

struct Point {
    double x, y;
};

struct Circle {
    Point c;
    double r;
};

constexpr Circle kInvalid{{0, 0}, -1};
constexpr double kMultiplicativeEpsilon = 1 + 1e-14;

double distance(const Point& a, const Point& b) { return std::hypot(a.x - b.x, a.y - b.y); }

double cross(const Point& a, const Point& b) { return a.x * b.y - a.y * b.x; }

Point subtract(const Point& a, const Point& b) { return Point{a.x - b.x, a.y - b.y}; }

bool contains(const Circle& c, const Point& p) { return distance(c.c, p) <= c.r * kMultiplicativeEpsilon; }

Circle diameter(const Point& a, const Point& b) {
    const Point c{(a.x + b.x) / 2, (a.y + b.y) / 2};
    return Circle{c, std::max(distance(c, a), distance(c, b))};
}

Circle circumcircle(const Point& a, const Point& b, const Point& c) {
    const double ox = (std::min(std::min(a.x, b.x), c.x) + std::max(std::min(a.x, b.x), c.x)) / 2;
    const double oy = (std::min(std::min(a.y, b.y), c.y) + std::max(std::min(a.y, b.y), c.y)) / 2;
    const double ax = a.x - ox, ay = a.y - oy;
    const double bx = b.x - ox, by = b.y - oy;
    const double cx = c.x - ox, cy = c.y - oy;
    const double d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2;
    if (d == 0) return kInvalid;
    const double x = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
    const double y = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
    const Point p{ox + x, oy + y};
    return Circle{p, std::max(std::max(distance(p, a), distance(p, b)), distance(p, c))};
}

Circle circle_two_points(const std::vector<Point>& pts, std::size_t end, const Point& p, const Point& q) {
    const Circle circ = diameter(p, q);
    Circle left = kInvalid;
    Circle right = kInvalid;
    const Point pq = subtract(q, p);
    for (std::size_t i = 0; i < end; ++i) {
        const Point& r = pts[i];
        if (contains(circ, r)) continue;
        const double cr = cross(pq, subtract(r, p));
        const Circle c = circumcircle(p, q, r);
        if (c.r < 0) continue;
        if (cr > 0 && (left.r < 0 || cross(pq, subtract(c.c, p)) > cross(pq, subtract(left.c, p)))) {
            left = c;
        } else if (cr < 0 && (right.r < 0 || cross(pq, subtract(c.c, p)) < cross(pq, subtract(right.c, p)))) {
            right = c;
        }
    }
    if (left.r < 0 && right.r < 0) return circ;
    if (left.r < 0) return right;
    if (right.r < 0) return left;
    return left.r <= right.r ? left : right;
}

Circle circle_one_point(const std::vector<Point>& pts, std::size_t end, const Point& p) {
    Circle c{p, 0};
    for (std::size_t i = 0; i < end; ++i) {
        const Point& q = pts[i];
        if (!contains(c, q)) c = c.r == 0 ? diameter(p, q) : circle_two_points(pts, i + 1, p, q);
    }
    return c;
}

std::uint64_t splitmix64(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

Circle smallest_circle(std::vector<Point>& pts, std::uint64_t seed) {
    for (std::size_t i = pts.size(); i > 1; --i) {
        std::swap(pts[i - 1], pts[splitmix64(seed) % i]);
    }
    Circle c = kInvalid;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        if (c.r < 0 || !contains(c, pts[i])) c = circle_one_point(pts, i + 1, pts[i]);
    }
    return c;
}

}  // namespace

// Distance between the centres of the nearest cells of the patch and of
// another patch of its class (NaN for a class with a single patch). The
// nearest cells always lie on the patch edges (cells with a 4-neighbour
// outside the patch or on the map edge), which landscapemetrics searches.
//
// The edge cells are sorted by class and Morton code, each class gets a tree
// over its range, and the cells search the nearest cell of another patch leaf
// by leaf (in parallel), lowering their patch's best distance atomically;
// cells of one patch are close in Morton order, so the searches start from a
// tight bound.
std::vector<double> Landscape::enn() {
    const Patches& pt = patches();
    const i64 n = pt.n;
    const i64 nx = nx_;
    const i64 ny = ny_;
    const int k = n_classes();
    std::vector<i64> class_np(k, 0);
    for (i64 i = 0; i < n; ++i) ++class_np[pt.cls[i]];
    std::vector<double> out(static_cast<std::size_t>(n), kNaN);
    // classes with several patches, numbered 0..G-1
    std::vector<std::int32_t> group(k, -1);
    std::int32_t n_groups = 0;
    for (int c = 0; c < k; ++c) {
        if (class_np[c] >= 2) group[c] = n_groups++;
    }
    if (n_groups == 0) return out;

    // Edge cells in Morton order. A strip (a power of two of rows) is a row
    // of square tiles: the cells are counted per tile, written straight into
    // their tile's place (tiles in Morton order) and sorted within each tile.
    const std::int32_t* cls = cls_.data();
    const i64 n_strips = static_cast<i64>(pt.strip_offset.size()) - 1;
    const i64 tile = pt.strip_rows;
    if ((tile & (tile - 1)) != 0) throw std::logic_error("enn: strip height must be a power of two");
    const bool parallel = nx * ny >= kParallelThreshold;
    auto is_edge = [&](i64 p, i64 x, i64 y) {
        const std::int32_t c = cls[p];
        return c >= 0 && group[c] >= 0 &&
               (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || cls[p - 1] != c || cls[p + 1] != c ||
                cls[p - nx] != c || cls[p + nx] != c);
    };
    const i64 ntx = (nx + tile - 1) / tile;
    const i64 n_tiles = ntx * n_strips;
    std::vector<i64> tile_count(static_cast<std::size_t>(n_tiles), 0);
#pragma omp parallel for schedule(dynamic) if (parallel && n_strips > 1)
    for (i64 s = 0; s < n_strips; ++s) {
        i64* count = tile_count.data() + s * ntx;
        const i64 y1 = std::min(ny, (s + 1) * tile);
        for (i64 y = s * tile; y < y1; ++y) {
            for (i64 x = 0; x < nx; ++x) count[x / tile] += is_edge(y * nx + x, x, y);
        }
    }
    std::vector<i64> tile_start(static_cast<std::size_t>(n_tiles));
    {
        std::vector<std::uint64_t> code(static_cast<std::size_t>(n_tiles));
        for (i64 t = 0; t < n_tiles; ++t) {
            code[t] = morton(static_cast<std::uint32_t>(t % ntx), static_cast<std::uint32_t>(t / ntx));
        }
        std::vector<i64> order(static_cast<std::size_t>(n_tiles));
        std::iota(order.begin(), order.end(), i64(0));
        std::sort(order.begin(), order.end(), [&](i64 a, i64 b) { return code[a] < code[b]; });
        i64 sum = 0;
        for (const i64 t : order) {
            tile_start[t] = sum;
            sum += tile_count[t];
        }
    }
    EdgeCells cells;
    const i64 m = std::accumulate(tile_count.begin(), tile_count.end(), i64(0));
    cells.n = m;
    // uninitialised buffers, first touched by the parallel loops
    cells.key.reset(new std::uint64_t[m]);
    cells.patch.reset(new std::int32_t[m]);
#pragma omp parallel for schedule(dynamic) if (parallel && n_strips > 1)
    for (i64 s = 0; s < n_strips; ++s) {
        std::vector<i64> pos(tile_start.begin() + s * ntx, tile_start.begin() + (s + 1) * ntx);
        const i64 y1 = std::min(ny, (s + 1) * tile);
        for (i64 y = s * tile; y < y1; ++y) {
            for (i64 x = 0; x < nx; ++x) {
                const i64 p = y * nx + x;
                if (!is_edge(p, x, y)) continue;
                const i64 j = pos[x / tile]++;
                cells.key[j] = morton(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y));
                cells.patch[j] = pt.patch(p, y);
            }
        }
        // sort every tile of the strip by Morton code
        std::vector<std::pair<std::uint64_t, std::int32_t>> buf;
        for (i64 tx = 0; tx < ntx; ++tx) {
            const i64 b = tile_start[s * ntx + tx];
            const i64 e = b + tile_count[s * ntx + tx];
            buf.clear();
            for (i64 j = b; j < e; ++j) buf.emplace_back(cells.key[j], cells.patch[j]);
            std::sort(buf.begin(), buf.end());
            for (i64 j = b; j < e; ++j) {
                cells.key[j] = buf[j - b].first;
                cells.patch[j] = buf[j - b].second;
            }
        }
    }
    tile_count = {};
    tile_start = {};
    // Then (stable) by class.
    if (n_groups > 1) {
        std::unique_ptr<std::uint64_t[]> key_tmp(new std::uint64_t[m]);
        std::unique_ptr<std::int32_t[]> patch_tmp(new std::int32_t[m]);
        const std::int32_t* pc = pt.cls.data();
        radix_pass(m, n_groups, m >= kParallelThreshold, cells.key, cells.patch, key_tmp, patch_tmp,
                   [&](std::uint64_t, std::int32_t p) { return static_cast<int>(group[pc[p]]); });
    }
    auto group_of = [&](i64 i) { return group[pt.cls[cells.patch[i]]]; };
    std::vector<i64> first(static_cast<std::size_t>(n_groups + 1), m);
    for (std::int32_t g = 0; g < n_groups; ++g) {
        i64 lo = 0, hi = m;  // first cell of group >= g
        while (lo < hi) {
            const i64 mid = lo + (hi - lo) / 2;
            if (group_of(mid) < g) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        first[g] = lo;
    }
    cells.x.reset(new std::int32_t[m]);
    cells.y.reset(new std::int32_t[m]);
#pragma omp parallel for schedule(static) if (m >= kParallelThreshold)
    for (i64 i = 0; i < m; ++i) {
        cells.x[i] = static_cast<std::int32_t>(morton_part(cells.key[i]));
        cells.y[i] = static_cast<std::int32_t>(morton_part(cells.key[i] >> 1));
    }

    std::vector<double> X(static_cast<std::size_t>(nx)), Y(static_cast<std::size_t>(ny));
    for (i64 x = 0; x < nx; ++x) X[x] = static_cast<double>(x + 1) * resx_;
    for (i64 y = 0; y < ny; ++y) Y[y] = static_cast<double>(y + 1) * resy_;

    // One tree per class: their tops here, all subtrees in one parallel loop,
    // then each tree joined.
    std::vector<CellTree> trees;
    trees.reserve(static_cast<std::size_t>(n_groups));
    std::vector<CellTree::Job> jobs;
    std::vector<std::int32_t> job_tree;
    for (std::int32_t g = 0; g < n_groups; ++g) {
        trees.emplace_back(cells, static_cast<std::int32_t>(first[g]), static_cast<std::int32_t>(first[g + 1]), X, Y,
                           jobs);
        job_tree.resize(jobs.size(), g);
    }
    {
        const auto n_jobs = static_cast<i64>(jobs.size());
        std::vector<std::vector<CellTree::Node>> sub(jobs.size());
#pragma omp parallel for schedule(dynamic) if (m >= kParallelThreshold)
        for (i64 j = 0; j < n_jobs; ++j) sub[j] = trees[job_tree[j]].build_job(jobs[j]);
#pragma omp parallel for schedule(dynamic) if (m >= kParallelThreshold && n_groups > 1)
        for (std::int32_t g = 0; g < n_groups; ++g) trees[g].finish(sub);
    }
    cells.key.reset();

    // Searches from every edge cell; positive doubles order like their bits.
    std::unique_ptr<std::atomic<std::uint64_t>[]> best(new std::atomic<std::uint64_t>[static_cast<std::size_t>(n)]);
    const std::uint64_t inf = double_bits(std::numeric_limits<double>::infinity());
#pragma omp parallel for schedule(static) if (parallel)
    for (i64 i = 0; i < n; ++i) best[i].store(inf, std::memory_order_relaxed);

    // Two shortcuts, both exact. (1) Smallest distance two patches of one
    // class can have (offsets of at most 2 cells, minus those joining them
    // into one patch), as computed anywhere on the map: a patch that reached
    // it needs no more searches. (2) The nearest cell within kWindow cells
    // (a raster look-up) is the nearest overall when it is nearer than any
    // cell outside the window can be.
    constexpr int kWindow = 2;
    auto min_step2 = [](const std::vector<double>& C, int a) {
        double m = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i + a < C.size(); ++i) m = std::min(m, (C[i + a] - C[i]) * (C[i + a] - C[i]));
        return a == 0 ? 0.0 : m;
    };
    double floor2 = std::numeric_limits<double>::infinity();
    for (int a = 0; a <= 2; ++a) {
        for (int b = 0; b <= 2; ++b) {
            const bool joined = options_.directions == 8 ? (a <= 1 && b <= 1) : (a + b <= 1);
            if (!joined) floor2 = std::min(floor2, min_step2(X, a) + min_step2(Y, b));
        }
    }
    const std::uint64_t floor_bits = double_bits(floor2);
    const double reach = (kWindow + 1) * std::min(resx_, resy_);
    const double certain2 = reach * reach * (1 - 1e-9);

    // The searches go leaf by leaf: the cells of one patch in a leaf search
    // together, pruning with the distance between boxes.
    std::vector<std::array<std::int32_t, 3>> work;  // tree, first cell, end of each leaf
    for (std::int32_t g = 0; g < n_groups; ++g) trees[g].leaves(g, work);
    const auto n_work = static_cast<i64>(work.size());
#pragma omp parallel for schedule(dynamic, 64) if (m >= kParallelThreshold)
    for (i64 w = 0; w < n_work; ++w) {
        const std::int32_t begin = work[w][1];
        const std::int32_t end = work[w][2];
        constexpr int kMax = CellTree::kLeafSize;
        double near[kMax];
        bool done[kMax];
        for (std::int32_t i = begin; i < end; ++i) {
            const std::int32_t p = cells.patch[i];
            done[i - begin] = best[p].load(std::memory_order_relaxed) <= floor_bits;
            if (done[i - begin]) continue;
            const std::int32_t x = cells.x[i];
            const std::int32_t y = cells.y[i];
            const std::int32_t c = pt.cls[p];
            double near2 = std::numeric_limits<double>::infinity();
            for (i64 yy = std::max<i64>(0, y - kWindow); yy <= std::min<i64>(ny - 1, y + kWindow); ++yy) {
                for (i64 xx = std::max<i64>(0, x - kWindow); xx <= std::min<i64>(nx - 1, x + kWindow); ++xx) {
                    const i64 q = yy * nx + xx;
                    if (cls[q] != c || pt.patch(q, yy) == p) continue;
                    const double dx = X[xx] - X[x];
                    const double dy = Y[yy] - Y[y];
                    near2 = std::min(near2, dx * dx + dy * dy);
                }
            }
            near[i - begin] = near2;
        }
        for (std::int32_t i = begin; i < end; ++i) {
            if (done[i - begin]) continue;
            const std::int32_t p = cells.patch[i];
            std::atomic<std::uint64_t>& b = best[p];
            std::uint64_t cur = b.load(std::memory_order_relaxed);
            double best2 = bits_double(cur);
            // the cells of patch p in this leaf; those not settled by their window search together
            std::int32_t idx[kMax];
            int count = 0;
            std::array<std::int32_t, 4> box{std::numeric_limits<std::int32_t>::max(), -1,
                                            std::numeric_limits<std::int32_t>::max(), -1};
            for (std::int32_t j = i; j < end; ++j) {
                if (done[j - begin] || cells.patch[j] != p) continue;
                done[j - begin] = true;
                best2 = std::min(best2, near[j - begin]);
                if (near[j - begin] < certain2) continue;
                idx[count++] = j;
                box[0] = std::min(box[0], cells.x[j]);
                box[1] = std::max(box[1], cells.x[j]);
                box[2] = std::min(box[2], cells.y[j]);
                box[3] = std::max(box[3], cells.y[j]);
            }
            if (count > 0) trees[work[w][0]].nearest(p, idx, count, box, best2);
            const std::uint64_t mine = double_bits(best2);
            while (mine < cur && !b.compare_exchange_weak(cur, mine, std::memory_order_relaxed)) {
            }
        }
    }
#pragma omp parallel for schedule(static) if (parallel)
    for (i64 i = 0; i < n; ++i) {
        if (group[pt.cls[i]] >= 0) out[i] = std::sqrt(bits_double(best[i].load(std::memory_order_relaxed)));
    }
    return out;
}

// 1 - patch area / area of the smallest circle around the patch. As in
// landscapemetrics, the circle encloses the corners of the patch cells in
// its first and last rows and columns.
std::vector<double> Landscape::circle() {
    const Patches& pt = patches();
    const std::vector<double>& area = patch_base("area");
    const i64 n = pt.n;
    const i64 nx = nx_;
    const i64 ny = ny_;
    const bool parallel = nx * ny >= kParallelThreshold;

    // Bounding boxes per strip piece, then per patch. A piece that does not
    // reach a strip seam is a whole patch and is written directly; the pieces
    // at the seams are joined afterwards.
    struct Box {
        std::int32_t x0, x1, y0, y1;
    };
    const Box empty{static_cast<std::int32_t>(nx), -1, static_cast<std::int32_t>(ny), -1};
    auto grow = [](Box& b, const Box& o) {
        b.x0 = std::min(b.x0, o.x0);
        b.x1 = std::max(b.x1, o.x1);
        b.y0 = std::min(b.y0, o.y0);
        b.y1 = std::max(b.y1, o.y1);
    };
    const i64 n_strips = static_cast<i64>(pt.strip_offset.size()) - 1;
    std::vector<Box> box(static_cast<std::size_t>(n));
    std::vector<std::vector<std::pair<std::int32_t, Box>>> seam(static_cast<std::size_t>(n_strips));
#pragma omp parallel if (parallel && n_strips > 1)
    {
        std::vector<Box> acc;
#pragma omp for schedule(dynamic)
        for (i64 s = 0; s < n_strips; ++s) {
            const i64 off = pt.strip_offset[s];
            acc.assign(static_cast<std::size_t>(pt.strip_offset[s + 1] - off), empty);
            const i64 y0 = s * pt.strip_rows;
            const i64 y1 = std::min(ny, (s + 1) * pt.strip_rows);
            for (i64 y = y0; y < y1; ++y) {
                const auto Y = static_cast<std::int32_t>(y);
                for (i64 x = 0; x < nx; ++x) {
                    const std::int32_t c = pt.code[y * nx + x];
                    if (c == -1) continue;
                    const auto X = static_cast<std::int32_t>(x);
                    grow(acc[-static_cast<i64>(c) - 2], Box{X, X, Y, Y});
                }
            }
            for (std::size_t k = 0; k < acc.size(); ++k) {
                const std::int32_t patch = pt.local_to_patch[off + static_cast<i64>(k)];
                const Box& b = acc[k];
                if ((s > 0 && b.y0 == y0) || (s + 1 < n_strips && b.y1 == y1 - 1)) {
                    seam[s].emplace_back(patch, b);
                } else {
                    box[patch] = b;
                }
            }
        }
    }
    for (const auto& list : seam) {
        for (const auto& piece : list) box[piece.first] = empty;
    }
    for (const auto& list : seam) {
        for (const auto& piece : list) grow(box[piece.first], piece.second);
    }
    seam = {};

    const double rx = resx_;
    const double ry = resy_;
    std::vector<double> out(static_cast<std::size_t>(n));
#pragma omp parallel if (parallel)
    {
        std::vector<Point> pts;  // reused by the patches of one thread
#pragma omp for schedule(dynamic, 64)
        for (i64 i = 0; i < n; ++i) {
            const Box& b = box[i];
            const auto me = static_cast<std::int32_t>(i);
            pts.clear();
            // corners as landscapemetrics computes them: col * res (+ res)
            for (i64 x = b.x0; x <= b.x1; ++x) {
                const double X = static_cast<double>(x) * rx;
                if (pt.patch(b.y0 * nx + x, b.y0) == me) {
                    const double Y = static_cast<double>(b.y0) * ry;
                    pts.push_back({X, Y});
                    pts.push_back({X + rx, Y});
                }
                if (pt.patch(b.y1 * nx + x, b.y1) == me) {
                    const double Y = static_cast<double>(b.y1) * ry + ry;
                    pts.push_back({X, Y});
                    pts.push_back({X + rx, Y});
                }
            }
            for (i64 y = b.y0; y <= b.y1; ++y) {
                const double Y = static_cast<double>(y) * ry;
                if (pt.patch(y * nx + b.x0, y) == me) {
                    const double X = static_cast<double>(b.x0) * rx;
                    pts.push_back({X, Y});
                    pts.push_back({X, Y + ry});
                }
                if (pt.patch(y * nx + b.x1, y) == me) {
                    const double X = static_cast<double>(b.x1) * rx + rx;
                    pts.push_back({X, Y});
                    pts.push_back({X, Y + ry});
                }
            }
            const Circle c = smallest_circle(pts, static_cast<std::uint64_t>(i));
            out[i] = 1 - (area[i] * 10000) / (kPi * std::pow(c.r, 2));
        }
    }
    return out;
}

}  // namespace lsm
