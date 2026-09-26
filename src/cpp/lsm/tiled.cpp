// SPDX-License-Identifier: GPL-3.0-or-later
//
// Out-of-core landscapes: the raster is read by blocks of rows. Every block
// is labelled with the in-memory engine; the pieces of the patches cut by
// the block seams are joined with a union-find, and the per-patch statistics
// that the block labelling got wrong at the seams (cell sides and cell pairs
// across them) are corrected. Blocks start at multiples of kStripRows rows,
// so they are labelled in the same strips as the whole raster in memory and
// the floating-point sums (gyrate) are made in the same order.
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "lsm/landscape.hpp"

namespace lsm {

namespace {

constexpr i64 kParallelThreshold = i64(1) << 16;
constexpr std::int32_t kFar = std::numeric_limits<std::int32_t>::max() / 2;
// Cells per block: the per-piece counts (cells, cell sides, cell pairs) of
// a block then fit in 32 bits.
constexpr i64 kMaxBlockCells = i64(1) << 28;

template <class T>
T uf_find(std::vector<T>& parent, T x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

template <class T>
void uf_unite(std::vector<T>& parent, T a, T b) {
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return;
    if (a < b) {
        parent[b] = a;
    } else {
        parent[a] = b;
    }
}

struct Box {
    std::int32_t x0, x1, y0, y1;
};

// Bounding box (rows local to the block) of every patch of a block labelling.
std::vector<Box> patch_boxes(const Patches& pt, i64 nx, i64 ny) {
    const Box empty{std::numeric_limits<std::int32_t>::max(), -1, std::numeric_limits<std::int32_t>::max(), -1};
    const i64 n_strips = static_cast<i64>(pt.strip_offset.size()) - 1;
    std::vector<std::vector<Box>> pieces(static_cast<std::size_t>(n_strips));
#pragma omp parallel for schedule(dynamic) if (nx * ny >= kParallelThreshold && n_strips > 1)
    for (i64 s = 0; s < n_strips; ++s) {
        std::vector<Box>& acc = pieces[s];
        acc.assign(static_cast<std::size_t>(pt.strip_offset[s + 1] - pt.strip_offset[s]), empty);
        const i64 y1 = std::min(ny, (s + 1) * pt.strip_rows);
        for (i64 y = s * pt.strip_rows; y < y1; ++y) {
            for (i64 x = 0; x < nx; ++x) {
                const std::int32_t c = pt.code[y * nx + x];
                if (c == -1) continue;
                Box& b = acc[-static_cast<i64>(c) - 2];
                b.x0 = std::min(b.x0, static_cast<std::int32_t>(x));
                b.x1 = std::max(b.x1, static_cast<std::int32_t>(x));
                b.y0 = std::min(b.y0, static_cast<std::int32_t>(y));
                b.y1 = std::max(b.y1, static_cast<std::int32_t>(y));
            }
        }
    }
    std::vector<Box> out(static_cast<std::size_t>(pt.n), empty);
    for (i64 s = 0; s < n_strips; ++s) {
        for (std::size_t k = 0; k < pieces[s].size(); ++k) {
            Box& b = out[pt.local_to_patch[pt.strip_offset[s] + static_cast<i64>(k)]];
            const Box& o = pieces[s][k];
            b.x0 = std::min(b.x0, o.x0);
            b.x1 = std::max(b.x1, o.x1);
            b.y0 = std::min(b.y0, o.y0);
            b.y1 = std::max(b.y1, o.y1);
        }
    }
    return out;
}

}  // namespace

struct TiledLandscape::State {
    i64 nx = 0, ny = 0;
    double resx = 1, resy = 1;
    Options opt;
    std::vector<std::int32_t> class_values;
    std::int32_t vmin = 0;
    std::vector<std::int32_t> lut;  // value - vmin -> class index (-2 = absent), if the range is small

    // pass 1: patch pieces (one per block labelling patch), in block order.
    // Counts of a piece fit in 32 bits (one block); patches sum them in 64.
    i64 next_y = 0;
    std::vector<i64> block_y0;
    std::vector<i64> block_offset;  // first piece of each block
    std::vector<std::int32_t> cls, cells, edge_h, edge_v, adj_orth, adj_diag;
    std::vector<i64> sum_x, sum_y, first_cm;
    std::vector<Box> box;
    std::vector<std::int32_t> parent;
    std::vector<std::pair<std::int32_t, std::int32_t>> diag_pending;  // 4 directions: diagonal pairs of pieces
    std::vector<std::int32_t> prev_cls;                               // last row of the previous block
    std::vector<std::int32_t> prev_piece;
    std::vector<i64> class_cells;
    Adjacency adj;

    // after finish()
    bool finished = false;
    std::vector<std::int32_t> piece_patch;
    std::vector<Box> patch_box;
    std::optional<Landscape> land;

    // detail pass
    bool d_core = false, d_gyrate = false, d_circle = false, d_enn = false;
    std::size_t d_block = 0;
    std::vector<double> gx, gy;               // centroids
    std::vector<double> gsum;
    std::vector<double> best;                  // cell_center: nearest distance (6 digits)
    std::vector<std::vector<std::pair<double, double>>> ties;
    std::vector<i64> flag_offset;             // circle: cells on the bounding box edges
    std::vector<std::uint8_t> flags;
    std::vector<std::pair<std::int32_t, std::int32_t>> row_ext;  // fragstats circle: first/last x per row
    std::vector<std::int32_t> group;          // enn: class -> group (-1: single patch)
    std::vector<std::vector<std::uint64_t>> enn_key;  // edge cells, per block
    std::vector<std::vector<std::int32_t>> enn_patch;
    std::vector<i64> core_cells;
    std::vector<i64> core_parent;
    std::vector<std::int32_t> core_owner;
    std::vector<std::int32_t> prev_core_cls;
    std::vector<i64> prev_core_piece;
    bool tie_pass = false;
    std::size_t t_block = 0;
    std::vector<double> tsum;

    std::vector<std::int32_t> dense(const std::int32_t* values, const std::uint8_t* na, i64 n) const {
        std::vector<std::int32_t> out(static_cast<std::size_t>(n));
        int missing = 0;
#pragma omp parallel for schedule(static) if (n >= kParallelThreshold) reduction(+ : missing)
        for (i64 p = 0; p < n; ++p) {
            if (na && na[p]) {
                out[p] = -1;
                continue;
            }
            std::int32_t c;
            if (!lut.empty()) {
                const i64 off = static_cast<i64>(values[p]) - vmin;
                c = off >= 0 && off < static_cast<i64>(lut.size()) ? lut[off] : -2;
            } else {
                auto it = std::lower_bound(class_values.begin(), class_values.end(), values[p]);
                c = it != class_values.end() && *it == values[p] ? static_cast<std::int32_t>(it - class_values.begin())
                                                                 : -2;
            }
            if (c < 0) missing += 1;
            out[p] = c;
        }
        if (missing > 0) throw std::invalid_argument("a block holds a class value missing from the class list");
        return out;
    }

    void check_block(i64 y0, i64 rows, i64 above, i64 below, i64 halo) const {
        if (rows <= 0 || y0 < 0 || y0 + rows > ny) throw std::invalid_argument("block outside the raster");
        if (above < std::min(halo, y0) || below < std::min(halo, ny - y0 - rows) || above > y0 ||
            below > ny - y0 - rows) {
            throw std::invalid_argument("block context rows do not match");
        }
    }

    // Block labelling of rows [y0, y0 + rows) of an extended block.
    Patches label_block(const std::vector<std::int32_t>& ext, i64 above, i64 rows) const {
        return label_patches(ext.data() + above * nx, nx, rows, opt.directions);
    }
};

TiledLandscape::TiledLandscape(i64 nx, i64 ny, double resx, double resy, Options options,
                               std::vector<std::int32_t> class_values)
    : s_(std::make_unique<State>()) {
    State& s = *s_;
    if (nx <= 0 || ny <= 0) throw std::invalid_argument("empty raster");
    if (options.directions != 4 && options.directions != 8) throw std::invalid_argument("directions must be 4 or 8");
    if (options.neighbourhood != 4 && options.neighbourhood != 8) {
        throw std::invalid_argument("neighbourhood must be 4 or 8");
    }
    if (options.base != "log2" && options.base != "log" && options.base != "log10") {
        throw std::invalid_argument("base must be 'log2', 'log' or 'log10'");
    }
    if (options.edge_depth < 0) throw std::invalid_argument("edge_depth must be >= 0");
    if (!std::is_sorted(class_values.begin(), class_values.end()) ||
        std::adjacent_find(class_values.begin(), class_values.end()) != class_values.end()) {
        throw std::invalid_argument("class values must be sorted and unique");
    }
    s.nx = nx;
    s.ny = ny;
    s.resx = resx;
    s.resy = resy;
    s.opt = std::move(options);
    s.class_values = std::move(class_values);
    const int k = static_cast<int>(s.class_values.size());
    if (k > 0) {
        s.vmin = s.class_values.front();
        const i64 range = static_cast<i64>(s.class_values.back()) - s.vmin + 1;
        if (range <= (i64(1) << 24)) {
            s.lut.assign(static_cast<std::size_t>(range), -2);
            for (int c = 0; c < k; ++c) s.lut[static_cast<i64>(s.class_values[c]) - s.vmin] = c;
        }
    }
    s.class_cells.assign(static_cast<std::size_t>(k), 0);
    s.adj.dim = k + 2;
    s.adj.h.assign(static_cast<std::size_t>(s.adj.dim) * s.adj.dim, 0);
    s.adj.v.assign(s.adj.h.size(), 0);
    if (s.opt.neighbourhood == 8) s.adj.diag.assign(s.adj.h.size(), 0);
    s.prev_cls.assign(static_cast<std::size_t>(nx), -1);
    s.prev_piece.assign(static_cast<std::size_t>(nx), -1);
}

TiledLandscape::~TiledLandscape() = default;

i64 TiledLandscape::nx() const { return s_->nx; }
i64 TiledLandscape::ny() const { return s_->ny; }
int TiledLandscape::n_threads() const { return s_->opt.n_threads; }

void TiledLandscape::add_block(const std::int32_t* values, const std::uint8_t* na_mask, i64 y0, i64 rows, i64 above,
                               i64 below) {
    State& s = *s_;
    if (s.finished) throw std::logic_error("the first pass is finished");
    if (y0 != s.next_y) throw std::invalid_argument("blocks must come in order, without gaps");
    s.check_block(y0, rows, above, below, 1);
    if (y0 + rows < s.ny && rows % kStripRows != 0) {
        throw std::invalid_argument("blocks (but the last) must have a multiple of 64 rows");
    }
    if (rows * s.nx > kMaxBlockCells) throw std::invalid_argument("block too large: use fewer rows per block");
    const i64 nx = s.nx;
    const i64 ny = s.ny;
    const i64 ext_rows = above + rows + below;
    const std::vector<std::int32_t> ext = s.dense(values, na_mask, ext_rows * nx);
    const std::int32_t* E = ext.data() + above * nx;  // first row of the block
    const Patches bp = s.label_block(ext, above, rows);
    const i64 offset = static_cast<i64>(s.cls.size());
    if (offset + bp.n > std::numeric_limits<std::int32_t>::max()) throw std::length_error("too many patch pieces");
    s.block_y0.push_back(y0);
    s.block_offset.push_back(offset);

    // pieces, in bulk; capacity for the total projected from the rows read
    const std::vector<Box> boxes = patch_boxes(bp, nx, rows);
    const i64 total = offset + bp.n;
    if (static_cast<std::size_t>(total) > s.cls.capacity()) {
        const auto projected = static_cast<std::size_t>(static_cast<double>(total) * ny / (y0 + rows) * 1.05) + 1024;
        for (auto* v : {&s.cls, &s.cells, &s.edge_h, &s.edge_v, &s.adj_orth, &s.adj_diag, &s.parent}) {
            v->reserve(projected);
        }
        for (auto* v : {&s.sum_x, &s.sum_y, &s.first_cm}) v->reserve(projected);
        s.box.reserve(projected);
    }
    for (auto* v : {&s.cls, &s.cells, &s.edge_h, &s.edge_v, &s.adj_orth, &s.adj_diag, &s.parent}) {
        v->resize(static_cast<std::size_t>(total));
    }
    for (auto* v : {&s.sum_x, &s.sum_y, &s.first_cm}) v->resize(static_cast<std::size_t>(total));
    s.box.resize(static_cast<std::size_t>(total));
#pragma omp parallel for schedule(static) if (bp.n >= kParallelThreshold)
    for (i64 k = 0; k < bp.n; ++k) {
        const i64 i = offset + k;
        s.cls[i] = bp.cls[k];
        s.cells[i] = static_cast<std::int32_t>(bp.cells[k]);
        s.edge_h[i] = static_cast<std::int32_t>(bp.edge_h[k]);
        s.edge_v[i] = static_cast<std::int32_t>(bp.edge_v[k]);
        s.adj_orth[i] = static_cast<std::int32_t>(bp.adj_orth[k]);
        s.adj_diag[i] = static_cast<std::int32_t>(bp.adj_diag[k]);
        s.sum_x[i] = bp.sum_x[k];
        s.sum_y[i] = bp.sum_y[k] + y0 * bp.cells[k];
        s.first_cm[i] = (bp.first_cm[k] / rows) * ny + y0 + bp.first_cm[k] % rows;
        Box b = boxes[k];
        b.y0 += static_cast<std::int32_t>(y0);
        b.y1 += static_cast<std::int32_t>(y0);
        s.box[i] = b;
        s.parent[i] = static_cast<std::int32_t>(i);
    }
    for (i64 k = 0; k < bp.n; ++k) s.class_cells[bp.cls[k]] += bp.cells[k];

    // 4 directions: diagonal pairs of one class in different pieces of the
    // block may still be one patch
    if (s.opt.directions == 4) {
        for (i64 y = 1; y < rows; ++y) {
            for (i64 x = 0; x < nx; ++x) {
                const i64 p = y * nx + x;
                const std::int32_t c = E[p];
                if (c < 0) continue;
                const std::int32_t me = bp.patch(p, y);
                for (int dx = -1; dx <= 1; dx += 2) {
                    const i64 xx = x + dx;
                    if (xx < 0 || xx >= nx || E[p - nx + dx] != c) continue;
                    const std::int32_t other = bp.patch(p - nx + dx, y - 1);
                    if (other != me) {
                        s.diag_pending.emplace_back(static_cast<std::int32_t>(offset + me),
                                                    static_cast<std::int32_t>(offset + other));
                    }
                }
            }
        }
    }

    // the seam with the previous block: the block labelling took the row
    // above for the landscape boundary
    if (y0 > 0) {
        for (i64 x = 0; x < nx; ++x) {
            const std::int32_t c = E[x];
            if (c < 0) continue;
            const auto me = static_cast<std::int32_t>(offset + bp.patch(x, 0));
            if (s.prev_cls[x] == c) {
                const std::int32_t up = s.prev_piece[x];
                uf_unite(s.parent, me, up);
                s.edge_v[me] -= 1;
                s.edge_v[up] -= 1;
                s.adj_orth[up] += 1;
            }
            for (int dx = -1; dx <= 1; dx += 2) {
                const i64 xx = x + dx;
                if (xx < 0 || xx >= nx || s.prev_cls[xx] != c) continue;
                if (s.opt.directions == 8) {
                    uf_unite(s.parent, me, s.prev_piece[xx]);
                    s.adj_diag[me] += 1;
                } else {
                    s.diag_pending.emplace_back(me, s.prev_piece[xx]);
                }
            }
        }
    }
    for (i64 x = 0; x < nx; ++x) {
        const i64 p = (rows - 1) * nx + x;
        s.prev_cls[x] = E[p];
        s.prev_piece[x] = E[p] < 0 ? -1 : static_cast<std::int32_t>(offset + bp.patch(p, rows - 1));
    }

    // class adjacencies of the block's cells (context rows as neighbours)
    const int k = static_cast<int>(s.class_values.size());
    const int dim = s.adj.dim;
    const int na = k;
    const int out = k + 1;
    const bool diag = s.opt.neighbourhood == 8;
    std::vector<i64>& H = s.adj.h;
    std::vector<i64>& V = s.adj.v;
    std::vector<i64>& D = s.adj.diag;
#pragma omp parallel if (rows * nx >= kParallelThreshold)
    {
        std::vector<i64> h(H.size(), 0), v(V.size(), 0), d(diag ? D.size() : 0, 0);
#pragma omp for schedule(static)
        for (i64 yl = 0; yl < rows; ++yl) {
            const i64 y = y0 + yl;
            for (i64 x = 0; x < nx; ++x) {
                const i64 p = yl * nx + x;
                const int a = E[p];
                if (a < 0) continue;
                const i64 row = static_cast<i64>(a) * dim;
                auto nb = [&](i64 q) { return E[q] < 0 ? na : E[q]; };
                ++h[row + (x > 0 ? nb(p - 1) : out)];
                ++h[row + (x + 1 < nx ? nb(p + 1) : out)];
                ++v[row + (y > 0 ? nb(p - nx) : out)];
                ++v[row + (y + 1 < ny ? nb(p + nx) : out)];
                if (diag) {
                    for (int dy = -1; dy <= 1; dy += 2) {
                        if (y + dy < 0 || y + dy >= ny) continue;
                        for (int dx = -1; dx <= 1; dx += 2) {
                            if (x + dx < 0 || x + dx >= nx) continue;
                            const int b = E[p + dy * nx + dx];
                            if (b >= 0) ++d[row + b];
                        }
                    }
                }
            }
        }
#pragma omp critical(lsm_tiled_adjacency)
        {
            for (std::size_t i = 0; i < H.size(); ++i) {
                H[i] += h[i];
                V[i] += v[i];
            }
            for (std::size_t i = 0; i < d.size(); ++i) D[i] += d[i];
        }
    }
    s.next_y = y0 + rows;
}

void TiledLandscape::finish() {
    State& s = *s_;
    if (s.finished) return;
    if (s.next_y != s.ny) throw std::logic_error("the first pass has not covered the raster");
    const auto n_pieces = static_cast<i64>(s.cls.size());
    std::vector<std::int32_t> root(static_cast<std::size_t>(n_pieces));
    for (i64 i = 0; i < n_pieces; ++i) root[i] = uf_find(s.parent, static_cast<std::int32_t>(i));
    s.parent = {};
    for (const auto& pr : s.diag_pending) {
        if (root[pr.first] == root[pr.second]) s.adj_diag[pr.first] += 1;
    }
    s.diag_pending = {};
    // landscapemetrics' patch order: by class, then first cell in column-major order
    for (i64 i = 0; i < n_pieces; ++i) {
        if (root[i] != i) s.first_cm[root[i]] = std::min(s.first_cm[root[i]], s.first_cm[i]);
    }
    std::vector<std::int32_t> roots;
    for (i64 i = 0; i < n_pieces; ++i) {
        if (root[i] == i) roots.push_back(static_cast<std::int32_t>(i));
    }
    std::sort(roots.begin(), roots.end(), [&](std::int32_t a, std::int32_t b) {
        return s.cls[a] != s.cls[b] ? s.cls[a] < s.cls[b] : s.first_cm[a] < s.first_cm[b];
    });
    const auto n = static_cast<i64>(roots.size());
    {
        std::vector<std::int32_t> rank(static_cast<std::size_t>(n_pieces), -1);
        for (i64 i = 0; i < n; ++i) rank[roots[i]] = static_cast<std::int32_t>(i);
        s.piece_patch.resize(static_cast<std::size_t>(n_pieces));
        for (i64 i = 0; i < n_pieces; ++i) s.piece_patch[i] = rank[root[i]];
    }
    root = {};

    // patch tables: sums of the pieces (each piece field freed once used)
    Patches pt;
    pt.n = n;
    auto sum_into = [&](auto& src, std::vector<i64>& dst) {
        dst.assign(static_cast<std::size_t>(n), 0);
        for (i64 i = 0; i < n_pieces; ++i) dst[s.piece_patch[i]] += src[i];
        src = {};
    };
    pt.cls.resize(static_cast<std::size_t>(n));
    pt.first_cm.resize(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) {
        pt.cls[i] = s.cls[roots[i]];
        pt.first_cm[i] = s.first_cm[roots[i]];
    }
    s.cls = {};
    s.first_cm = {};
    roots = {};
    sum_into(s.cells, pt.cells);
    sum_into(s.edge_h, pt.edge_h);
    sum_into(s.edge_v, pt.edge_v);
    sum_into(s.adj_orth, pt.adj_orth);
    sum_into(s.adj_diag, pt.adj_diag);
    sum_into(s.sum_x, pt.sum_x);
    sum_into(s.sum_y, pt.sum_y);
    const Box empty{std::numeric_limits<std::int32_t>::max(), -1, std::numeric_limits<std::int32_t>::max(), -1};
    s.patch_box.assign(static_cast<std::size_t>(n), empty);
    for (i64 i = 0; i < n_pieces; ++i) {
        Box& b = s.patch_box[s.piece_patch[i]];
        b.x0 = std::min(b.x0, s.box[i].x0);
        b.x1 = std::max(b.x1, s.box[i].x1);
        b.y0 = std::min(b.y0, s.box[i].y0);
        b.y1 = std::max(b.y1, s.box[i].y1);
    }
    s.box = {};
    s.prev_cls = {};
    s.prev_piece = {};
    s.land.emplace(s.nx, s.ny, s.resx, s.resy, s.opt, s.class_values, s.class_cells, std::move(pt), std::move(s.adj));
    s.finished = true;
}

Landscape& TiledLandscape::landscape() {
    if (!s_->finished) throw std::logic_error("the first pass is not finished");
    return *s_->land;
}

i64 TiledLandscape::detail_halo(bool core, bool enn) const {
    if (core) return std::max<i64>(1, s_->opt.edge_depth);
    return enn ? 1 : 0;
}

void TiledLandscape::begin_detail(bool core, bool gyrate, bool circle, bool enn) {
    State& s = *s_;
    if (!s.finished) throw std::logic_error("the first pass is not finished");
    const Patches& pt = s.land->patches();
    const i64 n = pt.n;
    s.d_core = core;
    s.d_gyrate = gyrate;
    s.d_circle = circle;
    s.d_enn = enn;
    s.d_block = 0;
    if (gyrate) {
        s.gx.resize(static_cast<std::size_t>(n));
        s.gy.resize(static_cast<std::size_t>(n));
        for (i64 i = 0; i < n; ++i) {
            const double c = static_cast<double>(pt.cells[i]);
            s.gx[i] = static_cast<double>(pt.sum_x[i] + pt.cells[i]) / c * s.resx;
            s.gy[i] = static_cast<double>(pt.sum_y[i] + pt.cells[i]) / c * s.resy;
        }
        if (s.opt.cell_center) {
            s.best.assign(static_cast<std::size_t>(n), std::numeric_limits<double>::infinity());
            s.ties.assign(static_cast<std::size_t>(n), {});
        } else {
            s.gsum.assign(static_cast<std::size_t>(n), 0.0);
        }
    }
    if (circle && s.opt.fragstats) {
        // the first and last cell of each patch in each of its rows
        s.flag_offset.assign(static_cast<std::size_t>(n + 1), 0);
        for (i64 i = 0; i < n; ++i) {
            const Box& b = s.patch_box[i];
            s.flag_offset[i + 1] = s.flag_offset[i] + (b.y1 - b.y0 + 1);
        }
        s.row_ext.assign(static_cast<std::size_t>(s.flag_offset[n]), {std::numeric_limits<std::int32_t>::max(), -1});
    } else if (circle) {
        s.flag_offset.assign(static_cast<std::size_t>(n + 1), 0);
        for (i64 i = 0; i < n; ++i) {
            const Box& b = s.patch_box[i];
            s.flag_offset[i + 1] = s.flag_offset[i] + 2 * (b.x1 - b.x0 + 1) + 2 * (b.y1 - b.y0 + 1);
        }
        s.flags.assign(static_cast<std::size_t>(s.flag_offset[n]), 0);
    }
    if (enn) {
        const int k = s.land->n_classes();
        std::vector<i64> class_np(static_cast<std::size_t>(k), 0);
        for (i64 i = 0; i < n; ++i) ++class_np[pt.cls[i]];
        s.group.assign(static_cast<std::size_t>(k), -1);
        std::int32_t g = 0;
        for (int c = 0; c < k; ++c) {
            if (class_np[c] >= 2) s.group[c] = g++;
        }
        s.enn_key.clear();
        s.enn_patch.clear();
    }
    if (core) {
        s.core_cells.assign(static_cast<std::size_t>(n), 0);
        s.core_parent.clear();
        s.core_owner.clear();
        s.prev_core_cls.assign(static_cast<std::size_t>(s.nx), -1);
        s.prev_core_piece.assign(static_cast<std::size_t>(s.nx), -1);
    }
}

void TiledLandscape::detail_block(const std::int32_t* values, const std::uint8_t* na_mask, i64 y0, i64 rows,
                                  i64 above, i64 below) {
    State& s = *s_;
    if (s.d_block >= s.block_y0.size() || s.block_y0[s.d_block] != y0) {
        throw std::invalid_argument("the detail pass must use the blocks of the first pass");
    }
    const i64 y_end = s.d_block + 1 < s.block_y0.size() ? s.block_y0[s.d_block + 1] : s.ny;
    if (rows != y_end - y0) throw std::invalid_argument("the detail pass must use the blocks of the first pass");
    s.check_block(y0, rows, above, below, detail_halo(s.d_core, s.d_enn));
    const i64 nx = s.nx;
    const i64 ny = s.ny;
    const i64 ext_rows = above + rows + below;
    const std::vector<std::int32_t> ext = s.dense(values, na_mask, ext_rows * nx);
    const std::int32_t* E = ext.data() + above * nx;
    const Patches bp = s.label_block(ext, above, rows);
    const i64 offset = s.block_offset[s.d_block];
    const i64 n_local = (s.d_block + 1 < s.block_offset.size() ? s.block_offset[s.d_block + 1]
                                                                : static_cast<i64>(s.piece_patch.size())) -
                        offset;
    if (bp.n != n_local) throw std::invalid_argument("the block differs from the first pass");
    std::vector<std::int32_t> local_patch(static_cast<std::size_t>(bp.n));
    for (i64 k = 0; k < bp.n; ++k) local_patch[k] = s.piece_patch[offset + k];
    const bool parallel = rows * nx >= kParallelThreshold;
    const i64 n_strips = static_cast<i64>(bp.strip_offset.size()) - 1;

    if (s.d_gyrate && !s.opt.cell_center) {
        // as Landscape::gyrate(): sums per strip piece, then per patch in strip order
        std::vector<std::vector<double>> pieces(static_cast<std::size_t>(n_strips));
#pragma omp parallel for schedule(dynamic) if (parallel && n_strips > 1)
        for (i64 st = 0; st < n_strips; ++st) {
            const i64 off = bp.strip_offset[st];
            std::vector<double>& acc = pieces[st];
            acc.assign(static_cast<std::size_t>(bp.strip_offset[st + 1] - off), 0.0);
            const i64 y1 = std::min(rows, (st + 1) * bp.strip_rows);
            for (i64 yl = st * bp.strip_rows; yl < y1; ++yl) {
                for (i64 x = 0; x < nx; ++x) {
                    const std::int32_t c = bp.code[yl * nx + x];
                    if (c == -1) continue;
                    const i64 k = -static_cast<i64>(c) - 2;
                    const std::int32_t pid = local_patch[bp.local_to_patch[off + k]];
                    const double dx = (x + 1) * s.resx - s.gx[pid];
                    const double dy = (y0 + yl + 1) * s.resy - s.gy[pid];
                    acc[k] += std::sqrt(dx * dx + dy * dy);
                }
            }
        }
        for (i64 st = 0; st < n_strips; ++st) {
            for (std::size_t k = 0; k < pieces[st].size(); ++k) {
                s.gsum[local_patch[bp.local_to_patch[bp.strip_offset[st] + static_cast<i64>(k)]]] += pieces[st][k];
            }
        }
    }
    if (s.d_gyrate && s.opt.cell_center) {
        // nearest cells to the centroid (6 significant digits), in row order
        for (i64 yl = 0; yl < rows; ++yl) {
            for (i64 x = 0; x < nx; ++x) {
                const std::int32_t lp = bp.patch(yl * nx + x, yl);
                if (lp < 0) continue;
                const std::int32_t k = local_patch[lp];
                const double X = (x + 1) * s.resx;
                const double Y = (y0 + yl + 1) * s.resy;
                const double d = r_signif(std::sqrt((X - s.gx[k]) * (X - s.gx[k]) + (Y - s.gy[k]) * (Y - s.gy[k])), 6);
                if (d < s.best[k]) {
                    s.best[k] = d;
                    s.ties[k].clear();
                    s.ties[k].emplace_back(X, Y);
                } else if (d == s.best[k]) {
                    s.ties[k].emplace_back(X, Y);
                }
            }
        }
    }
    if (s.d_circle && s.opt.fragstats) {
        // one row of a patch is only seen by the thread of that row
#pragma omp parallel for schedule(static) if (parallel)
        for (i64 yl = 0; yl < rows; ++yl) {
            const i64 y = y0 + yl;
            for (i64 x = 0; x < nx; ++x) {
                const std::int32_t lp = bp.patch(yl * nx + x, yl);
                if (lp < 0) continue;
                const std::int32_t k = local_patch[lp];
                auto& e = s.row_ext[s.flag_offset[k] + y - s.patch_box[k].y0];
                e.first = std::min(e.first, static_cast<std::int32_t>(x));
                e.second = std::max(e.second, static_cast<std::int32_t>(x));
            }
        }
    } else if (s.d_circle) {
        // cells on the edges of their patch's bounding box (distinct flags per cell)
#pragma omp parallel for schedule(static) if (parallel)
        for (i64 yl = 0; yl < rows; ++yl) {
            const auto y = static_cast<std::int32_t>(y0 + yl);
            for (i64 x = 0; x < nx; ++x) {
                const std::int32_t lp = bp.patch(yl * nx + x, yl);
                if (lp < 0) continue;
                const std::int32_t k = local_patch[lp];
                const Box& b = s.patch_box[k];
                const i64 w = b.x1 - b.x0 + 1;
                const i64 h = b.y1 - b.y0 + 1;
                std::uint8_t* f = s.flags.data() + s.flag_offset[k];
                const auto X = static_cast<std::int32_t>(x);
                if (y == b.y0) f[X - b.x0] = 1;                 // top row
                if (y == b.y1) f[w + X - b.x0] = 1;             // bottom row
                if (X == b.x0) f[2 * w + y - b.y0] = 1;         // left column
                if (X == b.x1) f[2 * w + h + y - b.y0] = 1;     // right column
            }
        }
    }
    if (s.d_enn) {
        std::vector<std::vector<std::pair<std::uint64_t, std::int32_t>>> found(static_cast<std::size_t>(rows));
#pragma omp parallel for schedule(static) if (parallel)
        for (i64 yl = 0; yl < rows; ++yl) {
            const i64 y = y0 + yl;
            for (i64 x = 0; x < nx; ++x) {
                const i64 p = yl * nx + x;
                const std::int32_t c = E[p];
                if (c < 0 || s.group[c] < 0) continue;
                if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || E[p - 1] != c || E[p + 1] != c ||
                    E[p - nx] != c || E[p + nx] != c) {
                    found[yl].emplace_back(morton_code(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y)),
                                           local_patch[bp.patch(p, yl)]);
                }
            }
        }
        std::size_t count = 0;
        for (const auto& f : found) count += f.size();
        std::vector<std::uint64_t> keys;
        std::vector<std::int32_t> patches;
        keys.reserve(count);
        patches.reserve(count);
        for (auto& f : found) {
            for (const auto& e : f) {
                keys.push_back(e.first);
                patches.push_back(e.second);
            }
            f = {};
        }
        s.enn_key.push_back(std::move(keys));
        s.enn_patch.push_back(std::move(patches));
    }
    if (s.d_core) {
        // city-block distance on the extended block: the context rows are
        // at least edge_depth deep, so only the raster edges count as edges
        const i64 depth = std::max(1, s.opt.edge_depth);
        const std::int32_t outside = s.opt.consider_boundary ? kFar : 1;
        const std::int32_t top = y0 - above == 0 ? outside : kFar;
        const std::int32_t bottom = y0 + rows + below == ny ? outside : kFar;
        const std::vector<std::int32_t> dist = city_block_distance(ext.data(), nx, ext_rows, top, bottom, outside);
        const std::int32_t* d = dist.data() + above * nx;
        std::vector<std::int32_t> core_cls(static_cast<std::size_t>(rows * nx), -1);
        for (i64 p = 0; p < rows * nx; ++p) {
            if (E[p] < 0 || d[p] <= depth) continue;
            core_cls[p] = E[p];
            ++s.core_cells[local_patch[bp.patch(p, p / nx)]];
        }
        // disjunct core areas: core cells labelled like patches, joined across seams
        const Patches cp = label_patches(core_cls.data(), nx, rows, s.opt.directions);
        const auto core_offset = static_cast<i64>(s.core_parent.size());
        for (i64 j = 0; j < cp.n; ++j) {
            s.core_parent.push_back(core_offset + j);
            const i64 x = cp.first_cm[j] / rows;
            const i64 yl = cp.first_cm[j] % rows;
            s.core_owner.push_back(local_patch[bp.patch(yl * nx + x, yl)]);
        }
        if (y0 > 0) {
            for (i64 x = 0; x < nx; ++x) {
                const std::int32_t c = core_cls[x];
                if (c < 0) continue;
                const i64 me = core_offset + cp.patch(x, 0);
                if (s.prev_core_cls[x] == c) uf_unite(s.core_parent, me, s.prev_core_piece[x]);
                if (s.opt.directions == 8) {
                    for (int dx = -1; dx <= 1; dx += 2) {
                        const i64 xx = x + dx;
                        if (xx >= 0 && xx < nx && s.prev_core_cls[xx] == c) {
                            uf_unite(s.core_parent, me, s.prev_core_piece[xx]);
                        }
                    }
                }
            }
        }
        for (i64 x = 0; x < nx; ++x) {
            const i64 p = (rows - 1) * nx + x;
            s.prev_core_cls[x] = core_cls[p];
            s.prev_core_piece[x] = core_cls[p] < 0 ? -1 : core_offset + cp.patch(p, rows - 1);
        }
    }
    ++s.d_block;
}

bool TiledLandscape::end_detail() {
    State& s = *s_;
    if (s.d_block != s.block_y0.size()) throw std::logic_error("the detail pass has not covered the raster");
    Landscape& land = *s.land;
    const Patches& pt = land.patches();
    const i64 n = pt.n;
    if (s.d_gyrate && !s.opt.cell_center) {
        std::vector<double> out(static_cast<std::size_t>(n));
        for (i64 i = 0; i < n; ++i) out[i] = static_cast<double>(s.gsum[i] / pt.cells[i]);
        land.set_patch_values("gyrate", std::move(out));
        s.gsum = {};
    }
    if (s.d_circle) {
        // the corners of Landscape::patch_circles(), then the same circle
        const double factor_ha = s.resx * s.resy / 10000;
        std::vector<double> out(static_cast<std::size_t>(n));
#pragma omp parallel if (n >= 1024)
        {
            std::vector<std::pair<i64, i64>> corners;
#pragma omp for schedule(dynamic, 64)
            for (i64 i = 0; i < n; ++i) {
                const Box& b = s.patch_box[i];
                corners.clear();
                if (s.opt.fragstats) {
                    for (i64 r = s.flag_offset[i]; r < s.flag_offset[i + 1]; ++r) {
                        const auto [x0, x1] = s.row_ext[r];
                        if (x1 < 0) continue;
                        const i64 y = b.y0 + (r - s.flag_offset[i]);
                        corners.emplace_back(x0, y);
                        corners.emplace_back(x0, y + 1);
                        corners.emplace_back(x1 + 1, y);
                        corners.emplace_back(x1 + 1, y + 1);
                    }
                } else {
                    const i64 w = b.x1 - b.x0 + 1;
                    const i64 h = b.y1 - b.y0 + 1;
                    const std::uint8_t* f = s.flags.data() + s.flag_offset[i];
                    for (i64 x = b.x0; x <= b.x1; ++x) {
                        if (f[x - b.x0]) {
                            corners.emplace_back(x, b.y0);
                            corners.emplace_back(x + 1, b.y0);
                        }
                        if (f[w + x - b.x0]) {
                            corners.emplace_back(x, b.y1 + 1);
                            corners.emplace_back(x + 1, b.y1 + 1);
                        }
                    }
                    for (i64 y = b.y0; y <= b.y1; ++y) {
                        if (f[2 * w + y - b.y0]) {
                            corners.emplace_back(b.x0, y);
                            corners.emplace_back(b.x0, y + 1);
                        }
                        if (f[2 * w + h + y - b.y0]) {
                            corners.emplace_back(b.x1 + 1, y);
                            corners.emplace_back(b.x1 + 1, y + 1);
                        }
                    }
                }
                const EnclosingCircle c = grid_circle(corners, s.resx, s.resy, static_cast<std::uint64_t>(i));
                out[i] = 1 - (static_cast<double>(pt.cells[i]) * factor_ha * 10000) / (3.14159265358979323846 * std::pow(c.r, 2));
            }
        }
        land.set_patch_values("circle", std::move(out));
        s.flags = {};
        s.row_ext = {};
        s.flag_offset = {};
    }
    if (s.d_enn) {
        // one array, filled block by block (each block freed once copied)
        i64 m = 0;
        for (const auto& k : s.enn_key) m += static_cast<i64>(k.size());
        std::unique_ptr<std::uint64_t[]> key(new std::uint64_t[m]);
        std::unique_ptr<std::int32_t[]> patch(new std::int32_t[m]);
        i64 at = 0;
        for (std::size_t b = 0; b < s.enn_key.size(); ++b) {
            std::copy(s.enn_key[b].begin(), s.enn_key[b].end(), key.get() + at);
            std::copy(s.enn_patch[b].begin(), s.enn_patch[b].end(), patch.get() + at);
            at += static_cast<i64>(s.enn_key[b].size());
            s.enn_key[b] = {};
            s.enn_patch[b] = {};
        }
        s.enn_key = {};
        s.enn_patch = {};
        land.set_patch_values("enn", enn_from_edge_cells(std::move(key), std::move(patch), m, pt.cls,
                                                         land.n_classes(), s.nx, s.ny, s.resx, s.resy,
                                                         s.opt.directions));
    }
    if (s.d_core) {
        Core core;
        core.cells = std::move(s.core_cells);
        core.areas.assign(static_cast<std::size_t>(n), 0);
        for (i64 j = 0; j < static_cast<i64>(s.core_parent.size()); ++j) {
            if (uf_find(s.core_parent, j) == j) ++core.areas[s.core_owner[j]];
        }
        land.set_core(std::move(core));
        s.core_parent = {};
        s.core_owner = {};
    }
    s.tie_pass = s.d_gyrate && s.opt.cell_center;
    if (s.tie_pass) {
        s.t_block = 0;
        s.tsum.assign(static_cast<std::size_t>(n), 0.0);
    }
    return s.tie_pass;
}

void TiledLandscape::tie_block(const std::int32_t* values, const std::uint8_t* na_mask, i64 y0, i64 rows, i64 above,
                               i64 below) {
    State& s = *s_;
    if (!s.tie_pass) throw std::logic_error("no tie pass is due");
    if (s.t_block >= s.block_y0.size() || s.block_y0[s.t_block] != y0) {
        throw std::invalid_argument("the tie pass must use the blocks of the first pass");
    }
    s.check_block(y0, rows, above, below, 0);
    const i64 nx = s.nx;
    const std::vector<std::int32_t> ext = s.dense(values, na_mask, (above + rows + below) * nx);
    const Patches bp = s.label_block(ext, above, rows);
    const i64 offset = s.block_offset[s.t_block];
    for (i64 yl = 0; yl < rows; ++yl) {
        for (i64 x = 0; x < nx; ++x) {
            const std::int32_t lp = bp.patch(yl * nx + x, yl);
            if (lp < 0) continue;
            const std::int32_t k = s.piece_patch[offset + lp];
            const double X = (x + 1) * s.resx;
            const double Y = (y0 + yl + 1) * s.resy;
            for (const auto& c : s.ties[k]) {
                s.tsum[k] += std::sqrt((X - c.first) * (X - c.first) + (Y - c.second) * (Y - c.second));
            }
        }
    }
    ++s.t_block;
}

void TiledLandscape::end_ties() {
    State& s = *s_;
    if (!s.tie_pass || s.t_block != s.block_y0.size()) throw std::logic_error("the tie pass has not covered the raster");
    const Patches& pt = s.land->patches();
    std::vector<double> out(static_cast<std::size_t>(pt.n));
    for (i64 i = 0; i < pt.n; ++i) {
        out[i] = static_cast<double>(s.tsum[i] / (static_cast<double>(pt.cells[i]) * s.ties[i].size()));
    }
    s.land->set_patch_values("gyrate", std::move(out));
    s.tie_pass = false;
    s.tsum = {};
    s.ties = {};
    s.best = {};
}

}  // namespace lsm
