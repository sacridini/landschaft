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

    // Index of the block of the first pass starting at row y0 (rows checked).
    std::size_t block_index(i64 y0, i64 rows, const char* what) const {
        const auto it = std::lower_bound(block_y0.begin(), block_y0.end(), y0);
        const auto b = static_cast<std::size_t>(it - block_y0.begin());
        const i64 y_end = b + 1 < block_y0.size() ? block_y0[b + 1] : ny;
        if (it == block_y0.end() || *it != y0 || rows != y_end - y0) throw std::invalid_argument(what);
        return b;
    }

    // Patch of every patch of a block labelling (its pieces of the first pass).
    std::vector<std::int32_t> local_patches(std::size_t b, const Patches& bp) const {
        const i64 offset = block_offset[b];
        const i64 n_local =
            (b + 1 < block_offset.size() ? block_offset[b + 1] : static_cast<i64>(piece_patch.size())) - offset;
        if (bp.n != n_local) throw std::invalid_argument("the block differs from the first pass");
        return {piece_patch.begin() + offset, piece_patch.begin() + offset + n_local};
    }
};

// What a block prepared hands to its merge.
struct TiledBlock::Data {
    i64 y0 = 0, rows = 0;
    // pass 1: the block's patches (pieces) and their boxes, diagonal pairs of
    // pieces (4 directions), its first and last rows (class, piece; -1: NA)
    // and its class adjacencies
    Patches bp;
    std::vector<Box> boxes;
    std::vector<std::pair<std::int32_t, std::int32_t>> diag;
    std::vector<std::int32_t> first_cls, first_piece, last_cls, last_piece;
    std::vector<i64> h, v, dg;
    // detail pass
    std::size_t block = 0;
    std::vector<std::pair<std::int32_t, double>> gyrate_sums;  // (patch, sum of a strip piece), in strip order
    std::vector<std::int32_t> local_patch;                    // cell_center: kept with bp for the merge
    std::vector<std::uint64_t> enn_key;
    std::vector<std::int32_t> enn_patch;
    std::vector<std::pair<std::int32_t, i64>> core_cells;  // (patch, core cells)
    std::vector<std::int32_t> core_owner;                   // patch of each core piece
    std::vector<std::int32_t> core_first_cls, core_first_piece, core_last_cls, core_last_piece;
};

TiledBlock::TiledBlock() = default;
TiledBlock::TiledBlock(TiledBlock&&) noexcept = default;
TiledBlock& TiledBlock::operator=(TiledBlock&&) noexcept = default;
TiledBlock::~TiledBlock() = default;

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
    if (s_->finished) throw std::logic_error("the first pass is finished");
    if (y0 != s_->next_y) throw std::invalid_argument("blocks must come in order, without gaps");
    TiledBlock block = prepare_block(values, na_mask, y0, rows, above, below);
    merge_block(block);
}

TiledBlock TiledLandscape::prepare_block(const std::int32_t* values, const std::uint8_t* na_mask, i64 y0, i64 rows,
                                         i64 above, i64 below) const {
    const State& s = *s_;
    if (s.finished) throw std::logic_error("the first pass is finished");
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
    TiledBlock out;
    out.d = std::make_unique<TiledBlock::Data>();
    TiledBlock::Data& b = *out.d;
    b.y0 = y0;
    b.rows = rows;
    b.bp = s.label_block(ext, above, rows);
    const Patches& bp = b.bp;
    b.boxes = patch_boxes(bp, nx, rows);

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
                    if (other != me) b.diag.emplace_back(me, other);
                }
            }
        }
    }

    // the first and last rows, for the seams with the blocks around
    b.first_cls.assign(E, E + nx);
    b.last_cls.assign(E + (rows - 1) * nx, E + rows * nx);
    b.first_piece.resize(static_cast<std::size_t>(nx));
    b.last_piece.resize(static_cast<std::size_t>(nx));
    for (i64 x = 0; x < nx; ++x) {
        b.first_piece[x] = E[x] < 0 ? -1 : bp.patch(x, 0);
        const i64 p = (rows - 1) * nx + x;
        b.last_piece[x] = E[p] < 0 ? -1 : bp.patch(p, rows - 1);
    }

    // class adjacencies of the block's cells (context rows as neighbours)
    const int k = static_cast<int>(s.class_values.size());
    const int dim = s.adj.dim;
    const int na = k;
    const int edge = k + 1;
    const bool diag = s.opt.neighbourhood == 8;
    const auto cells = static_cast<std::size_t>(dim) * dim;
    b.h.assign(cells, 0);
    b.v.assign(cells, 0);
    b.dg.assign(diag ? cells : 0, 0);
    auto count_rows = [&](i64 r0, i64 r1, i64* hh, i64* vv, i64* dd) {
        for (i64 yl = r0; yl < r1; ++yl) {
            const i64 y = y0 + yl;
            for (i64 x = 0; x < nx; ++x) {
                const i64 p = yl * nx + x;
                const int a = E[p];
                if (a < 0) continue;
                const i64 row = static_cast<i64>(a) * dim;
                auto nb = [&](i64 q) { return E[q] < 0 ? na : E[q]; };
                ++hh[row + (x > 0 ? nb(p - 1) : edge)];
                ++hh[row + (x + 1 < nx ? nb(p + 1) : edge)];
                ++vv[row + (y > 0 ? nb(p - nx) : edge)];
                ++vv[row + (y + 1 < ny ? nb(p + nx) : edge)];
                if (diag) {
                    for (int dy = -1; dy <= 1; dy += 2) {
                        if (y + dy < 0 || y + dy >= ny) continue;
                        for (int dx = -1; dx <= 1; dx += 2) {
                            if (x + dx < 0 || x + dx >= nx) continue;
                            const int c = E[p + dy * nx + dx];
                            if (c >= 0) ++dd[row + c];
                        }
                    }
                }
            }
        }
    };
    if (rows * nx >= kParallelThreshold) {
#pragma omp parallel
        {
            std::vector<i64> h(cells, 0), v(cells, 0), d(diag ? cells : 0, 0);
#pragma omp for schedule(static)
            for (i64 yl = 0; yl < rows; ++yl) count_rows(yl, yl + 1, h.data(), v.data(), d.data());
#pragma omp critical(lsm_tiled_adjacency)
            {
                for (std::size_t i = 0; i < cells; ++i) {
                    b.h[i] += h[i];
                    b.v[i] += v[i];
                }
                for (std::size_t i = 0; i < d.size(); ++i) b.dg[i] += d[i];
            }
        }
    } else {
        count_rows(0, rows, b.h.data(), b.v.data(), b.dg.data());
    }
    // the merge needs the tables of the pieces only
    b.bp.code = {};
    b.bp.local_to_patch = {};
    return out;
}

void TiledLandscape::merge_block(TiledBlock& block) {
    State& s = *s_;
    if (s.finished) throw std::logic_error("the first pass is finished");
    if (!block.d) throw std::invalid_argument("the block was merged already");
    TiledBlock::Data& b = *block.d;
    if (b.y0 != s.next_y) throw std::invalid_argument("blocks must come in order, without gaps");
    const i64 nx = s.nx;
    const i64 ny = s.ny;
    const i64 y0 = b.y0;
    const i64 rows = b.rows;
    const Patches& bp = b.bp;
    const i64 offset = static_cast<i64>(s.cls.size());
    if (offset + bp.n > std::numeric_limits<std::int32_t>::max()) throw std::length_error("too many patch pieces");
    s.block_y0.push_back(y0);
    s.block_offset.push_back(offset);

    // pieces, in bulk; capacity for the total projected from the rows read
    const std::vector<Box>& boxes = b.boxes;
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
        Box bx = boxes[k];
        bx.y0 += static_cast<std::int32_t>(y0);
        bx.y1 += static_cast<std::int32_t>(y0);
        s.box[i] = bx;
        s.parent[i] = static_cast<std::int32_t>(i);
    }
    for (i64 k = 0; k < bp.n; ++k) s.class_cells[bp.cls[k]] += bp.cells[k];
    for (const auto& pr : b.diag) {
        s.diag_pending.emplace_back(static_cast<std::int32_t>(offset + pr.first),
                                    static_cast<std::int32_t>(offset + pr.second));
    }

    // the seam with the previous block: the block labelling took the row
    // above for the landscape boundary
    if (y0 > 0) {
        for (i64 x = 0; x < nx; ++x) {
            const std::int32_t c = b.first_cls[x];
            if (c < 0) continue;
            const auto me = static_cast<std::int32_t>(offset + b.first_piece[x]);
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
        s.prev_cls[x] = b.last_cls[x];
        s.prev_piece[x] = b.last_cls[x] < 0 ? -1 : static_cast<std::int32_t>(offset + b.last_piece[x]);
    }

    // class adjacencies
    for (std::size_t i = 0; i < s.adj.h.size(); ++i) {
        s.adj.h[i] += b.h[i];
        s.adj.v[i] += b.v[i];
    }
    for (std::size_t i = 0; i < b.dg.size(); ++i) s.adj.diag[i] += b.dg[i];
    s.next_y = y0 + rows;
    block.d.reset();
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
    const State& s = *s_;
    if (s.d_block >= s.block_y0.size() || s.block_y0[s.d_block] != y0) {
        throw std::invalid_argument("the detail pass must use the blocks of the first pass");
    }
    TiledBlock block = prepare_detail(values, na_mask, y0, rows, above, below);
    merge_detail(block);
}

TiledBlock TiledLandscape::prepare_detail(const std::int32_t* values, const std::uint8_t* na_mask, i64 y0, i64 rows,
                                          i64 above, i64 below) const {
    State& s = *s_;  // the circle marks of the block's rows are written here
    if (!s.finished) throw std::logic_error("the first pass is not finished");
    const std::size_t block = s.block_index(y0, rows, "the detail pass must use the blocks of the first pass");
    s.check_block(y0, rows, above, below, detail_halo(s.d_core, s.d_enn));
    const i64 nx = s.nx;
    const i64 ny = s.ny;
    const i64 ext_rows = above + rows + below;
    const std::vector<std::int32_t> ext = s.dense(values, na_mask, ext_rows * nx);
    const std::int32_t* E = ext.data() + above * nx;
    TiledBlock out;
    out.d = std::make_unique<TiledBlock::Data>();
    TiledBlock::Data& b = *out.d;
    b.y0 = y0;
    b.rows = rows;
    b.block = block;
    Patches bp = s.label_block(ext, above, rows);
    std::vector<std::int32_t> local_patch = s.local_patches(block, bp);
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
                b.gyrate_sums.emplace_back(local_patch[bp.local_to_patch[bp.strip_offset[st] + static_cast<i64>(k)]],
                                           pieces[st][k]);
            }
        }
    }
    if (s.d_circle && s.opt.fragstats) {
        // one row of a patch is only seen by the thread of that row (and the block of that row)
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
        // cells on the edges of their patch's bounding box (distinct flags per
        // cell, so per row and block)
#pragma omp parallel for schedule(static) if (parallel)
        for (i64 yl = 0; yl < rows; ++yl) {
            const auto y = static_cast<std::int32_t>(y0 + yl);
            for (i64 x = 0; x < nx; ++x) {
                const std::int32_t lp = bp.patch(yl * nx + x, yl);
                if (lp < 0) continue;
                const std::int32_t k = local_patch[lp];
                const Box& bx = s.patch_box[k];
                const i64 w = bx.x1 - bx.x0 + 1;
                const i64 h = bx.y1 - bx.y0 + 1;
                std::uint8_t* f = s.flags.data() + s.flag_offset[k];
                const auto X = static_cast<std::int32_t>(x);
                if (y == bx.y0) f[X - bx.x0] = 1;                 // top row
                if (y == bx.y1) f[w + X - bx.x0] = 1;             // bottom row
                if (X == bx.x0) f[2 * w + y - bx.y0] = 1;         // left column
                if (X == bx.x1) f[2 * w + h + y - bx.y0] = 1;     // right column
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
        b.enn_key.reserve(count);
        b.enn_patch.reserve(count);
        for (auto& f : found) {
            for (const auto& e : f) {
                b.enn_key.push_back(e.first);
                b.enn_patch.push_back(e.second);
            }
            f = {};
        }
    }
    if (s.d_core) {
        // city-block distance on the extended block: the context rows are
        // at least edge_depth deep, so only the raster edges count as edges
        const i64 depth = std::max(1, s.opt.edge_depth);
        const std::int32_t outside = s.opt.consider_boundary ? kFar : 1;
        std::vector<std::int32_t> dist;
        if (depth > 1) {
            const std::int32_t top = y0 - above == 0 ? outside : kFar;
            const std::int32_t bottom = y0 + rows + below == ny ? outside : kFar;
            dist = city_block_distance(ext.data(), nx, ext_rows, top, bottom, outside);
        }
        const std::int32_t* d = depth > 1 ? dist.data() + above * nx : nullptr;
        // at depth 1 (as Landscape::core()): the four neighbours, the context
        // rows giving those across the block seams
        const bool inside_ok = s.opt.consider_boundary;
        auto is_core = [&](i64 p, i64 x, i64 y) {
            if (depth > 1) return d[p] > depth;
            const std::int32_t c = E[p];
            return (x > 0 ? E[p - 1] == c : inside_ok) && (x + 1 < nx ? E[p + 1] == c : inside_ok) &&
                   (y > 0 ? E[p - nx] == c : inside_ok) && (y + 1 < ny ? E[p + nx] == c : inside_ok);
        };
        std::vector<std::int32_t> core_cls(static_cast<std::size_t>(rows * nx), -1);
        std::vector<i64> per_piece(static_cast<std::size_t>(bp.n), 0);
        for (i64 yl = 0; yl < rows; ++yl) {
            for (i64 x = 0; x < nx; ++x) {
                const i64 p = yl * nx + x;
                if (E[p] < 0 || !is_core(p, x, y0 + yl)) continue;
                core_cls[p] = E[p];
                ++per_piece[bp.patch(p, yl)];
            }
        }
        for (i64 k = 0; k < bp.n; ++k) {
            if (per_piece[k] > 0) b.core_cells.emplace_back(local_patch[k], per_piece[k]);
        }
        // disjunct core areas: core cells labelled like patches (class,
        // cells and first cell only), joined across seams by the merge
        Patches cp;
        label_patches(core_cls.data(), nx, rows, s.opt.directions, cp, false);
        b.core_owner.resize(static_cast<std::size_t>(cp.n));
        for (i64 j = 0; j < cp.n; ++j) {
            const i64 x = cp.first_cm[j] / rows;
            const i64 yl = cp.first_cm[j] % rows;
            b.core_owner[j] = local_patch[bp.patch(yl * nx + x, yl)];
        }
        b.core_first_cls.assign(core_cls.begin(), core_cls.begin() + nx);
        b.core_last_cls.assign(core_cls.end() - nx, core_cls.end());
        b.core_first_piece.resize(static_cast<std::size_t>(nx));
        b.core_last_piece.resize(static_cast<std::size_t>(nx));
        for (i64 x = 0; x < nx; ++x) {
            b.core_first_piece[x] = core_cls[x] < 0 ? -1 : cp.patch(x, 0);
            const i64 p = (rows - 1) * nx + x;
            b.core_last_piece[x] = core_cls[p] < 0 ? -1 : cp.patch(p, rows - 1);
        }
    }
    if (s.d_gyrate && s.opt.cell_center) {
        // the nearest cells go in row order over all blocks: the merge does it
        b.bp = std::move(bp);
        b.local_patch = std::move(local_patch);
    }
    return out;
}

void TiledLandscape::merge_detail(TiledBlock& block) {
    State& s = *s_;
    if (!block.d) throw std::invalid_argument("the block was merged already");
    TiledBlock::Data& b = *block.d;
    if (b.block != s.d_block) throw std::invalid_argument("the detail pass must merge the blocks in order");
    const i64 nx = s.nx;
    const i64 y0 = b.y0;
    const i64 rows = b.rows;
    for (const auto& [pid, sum] : b.gyrate_sums) s.gsum[pid] += sum;
    if (s.d_gyrate && s.opt.cell_center) {
        // nearest cells to the centroid (6 significant digits), in row order
        const Patches& bp = b.bp;
        for (i64 yl = 0; yl < rows; ++yl) {
            for (i64 x = 0; x < nx; ++x) {
                const std::int32_t lp = bp.patch(yl * nx + x, yl);
                if (lp < 0) continue;
                const std::int32_t k = b.local_patch[lp];
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
    if (s.d_enn) {
        s.enn_key.push_back(std::move(b.enn_key));
        s.enn_patch.push_back(std::move(b.enn_patch));
    }
    if (s.d_core) {
        for (const auto& [pid, n] : b.core_cells) s.core_cells[pid] += n;
        const auto core_offset = static_cast<i64>(s.core_parent.size());
        for (std::size_t j = 0; j < b.core_owner.size(); ++j) {
            s.core_parent.push_back(core_offset + static_cast<i64>(j));
            s.core_owner.push_back(b.core_owner[j]);
        }
        if (y0 > 0) {
            for (i64 x = 0; x < nx; ++x) {
                const std::int32_t c = b.core_first_cls[x];
                if (c < 0) continue;
                const i64 me = core_offset + b.core_first_piece[x];
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
            s.prev_core_cls[x] = b.core_last_cls[x];
            s.prev_core_piece[x] = b.core_last_cls[x] < 0 ? -1 : core_offset + b.core_last_piece[x];
        }
    }
    ++s.d_block;
    block.d.reset();
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

std::vector<std::int32_t> TiledLandscape::block_labels(const std::int32_t* values, const std::uint8_t* na_mask, i64 y0,
                                                       i64 rows, i64 above, i64 below) const {
    const State& s = *s_;
    if (!s.finished) throw std::logic_error("the first pass is not finished");
    const std::size_t b = s.block_index(y0, rows, "the labels must be read by the blocks of the first pass");
    s.check_block(y0, rows, above, below, 0);
    const i64 nx = s.nx;
    const std::vector<std::int32_t> ext = s.dense(values, na_mask, (above + rows + below) * nx);
    const Patches bp = s.label_block(ext, above, rows);
    const std::vector<std::int32_t> local_patch = s.local_patches(b, bp);
    std::vector<std::int32_t> out(static_cast<std::size_t>(rows * nx));
#pragma omp parallel for schedule(static) if (rows * nx >= kParallelThreshold)
    for (i64 yl = 0; yl < rows; ++yl) {
        for (i64 x = 0; x < nx; ++x) {
            const i64 p = yl * nx + x;
            const std::int32_t lp = bp.patch(p, yl);
            out[p] = lp < 0 ? 0 : local_patch[lp] + 1;
        }
    }
    return out;
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
