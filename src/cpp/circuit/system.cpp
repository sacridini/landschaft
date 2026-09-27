// SPDX-License-Identifier: GPL-3.0-or-later

#include "circuit/system.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <algorithm>
#include <queue>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace circuit {

namespace {

constexpr i64 kLeafCells = 16;

using Block = Eigen::Map<Eigen::MatrixXd, 0, Eigen::OuterStride<>>;
using ConstBlock = Eigen::Map<const Eigen::MatrixXd, 0, Eigen::OuterStride<>>;

struct TreeBuilder {
    const std::uint8_t* mark;
    i64 stride;
    std::vector<i64> cells;
    std::vector<FrontalSymbolic::Node> nodes;
    std::vector<std::vector<std::int32_t>> kids;

    std::int32_t add(std::int32_t first, std::vector<std::int32_t> ch) {
        FrontalSymbolic::Node n;
        n.first = first;
        n.npiv = static_cast<std::int32_t>(cells.size()) - first;
        nodes.push_back(n);
        kids.push_back(std::move(ch));
        return static_cast<std::int32_t>(nodes.size()) - 1;
    }

    // Shrinks the box to its marked cells, cuts it by its middle row or column
    // (a single line separates 4- and 8-neighbour grids) and orders both halves
    // before the separator; small boxes are leaves in row order. Returns the
    // subtree's root, or -1 for an empty box.
    std::int32_t build(i64 r0, i64 r1, i64 c0, i64 c1) {
        i64 top = r1, bottom = r0 - 1, left = c1, right = c0 - 1;
        for (i64 r = r0; r < r1; ++r)
            for (i64 c = c0; c < c1; ++c)
                if (mark[r * stride + c]) {
                    top = std::min(top, r);
                    bottom = std::max(bottom, r);
                    left = std::min(left, c);
                    right = std::max(right, c);
                }
        if (bottom < top) return -1;
        r0 = top;
        r1 = bottom + 1;
        c0 = left;
        c1 = right + 1;
        const i64 h = r1 - r0, w = c1 - c0;
        if (h * w <= kLeafCells) {
            const std::int32_t first = static_cast<std::int32_t>(cells.size());
            for (i64 r = r0; r < r1; ++r)
                for (i64 c = c0; c < c1; ++c)
                    if (mark[r * stride + c]) cells.push_back(r * stride + c);
            return add(first, {});
        }
        std::vector<std::int32_t> ch;
        const std::int32_t first_sep = [&] {
            if (h >= w) {
                const i64 m = r0 + h / 2;
                const std::int32_t a = build(r0, m, c0, c1);
                const std::int32_t b = build(m + 1, r1, c0, c1);
                if (a >= 0) ch.push_back(a);
                if (b >= 0) ch.push_back(b);
                const std::int32_t f = static_cast<std::int32_t>(cells.size());
                for (i64 c = c0; c < c1; ++c)
                    if (mark[m * stride + c]) cells.push_back(m * stride + c);
                return f;
            }
            const i64 m = c0 + w / 2;
            const std::int32_t a = build(r0, r1, c0, m);
            const std::int32_t b = build(r0, r1, m + 1, c1);
            if (a >= 0) ch.push_back(a);
            if (b >= 0) ch.push_back(b);
            const std::int32_t f = static_cast<std::int32_t>(cells.size());
            for (i64 r = r0; r < r1; ++r)
                if (mark[r * stride + m]) cells.push_back(r * stride + m);
            return f;
        }();
        if (static_cast<std::int32_t>(cells.size()) == first_sep && ch.size() == 1) return ch[0];
        return add(first_sep, std::move(ch));
    }
};

// Nested dissection of a general graph (no geometry): a set of nodes is split
// by a level of a breadth-first search from a pseudo-peripheral node (edges
// only join consecutive levels), the smallest level that leaves at least a
// fifth of the nodes on either side, trimmed of the nodes that touch no node
// beyond it; both sides first, the separator last. Disconnected sets give one
// subtree per component. The sets of one depth are disjoint and split in
// parallel (the result does not depend on the threads); the tree is then laid
// out in post-order: `order` receives the nodes in elimination order.
struct GraphDissection {
    static constexpr std::size_t kLeaf = 32;
    const std::vector<std::int64_t>& ptr;
    const std::vector<std::int32_t>& adj;
    std::vector<std::int32_t> order;
    std::vector<FrontalSymbolic::Node> nodes;
    std::vector<std::vector<std::int32_t>> kids;
    std::vector<std::int32_t> mark, level;  // mark: the part being split (its index)

    struct Part {
        std::vector<std::int32_t> set, piv, sub;  // nodes; pivots; child parts
        bool pass = false;                         // components only: no front of its own
    };
    std::vector<Part> parts;

    GraphDissection(std::int32_t n, const std::vector<std::int64_t>& p, const std::vector<std::int32_t>& a)
        : ptr(p), adj(a), mark(static_cast<std::size_t>(n), -1), level(static_cast<std::size_t>(n), -1) {}

    std::int32_t add(std::vector<std::int32_t>& piv, std::vector<std::int32_t> ch) {
        std::sort(piv.begin(), piv.end());
        FrontalSymbolic::Node node;
        node.first = static_cast<std::int32_t>(order.size());
        node.npiv = static_cast<std::int32_t>(piv.size());
        order.insert(order.end(), piv.begin(), piv.end());
        nodes.push_back(node);
        kids.push_back(std::move(ch));
        return static_cast<std::int32_t>(nodes.size()) - 1;
    }

    // BFS over the nodes marked t from `s`: levels, the nodes in BFS order.
    void bfs(std::int32_t s, std::int32_t t, std::vector<std::int32_t>& seen) {
        seen.clear();
        seen.push_back(s);
        level[s] = 0;
        for (std::size_t h = 0; h < seen.size(); ++h) {
            const std::int32_t v = seen[h];
            for (std::int64_t e = ptr[v]; e < ptr[v + 1]; ++e) {
                const std::int32_t w = adj[static_cast<std::size_t>(e)];
                if (w < 0 || mark[w] != t || level[w] >= 0) continue;
                level[w] = level[v] + 1;
                seen.push_back(w);
            }
        }
    }

    void clear_levels(const std::vector<std::int32_t>& vs) {
        for (std::int32_t v : vs) level[v] = -1;
    }

    // Splits part t (its nodes marked t): its pivots, or the node sets of its children.
    void split(Part& p, std::int32_t t, std::vector<std::vector<std::int32_t>>& children) {
        std::vector<std::int32_t>& set = p.set;
        if (set.size() <= kLeaf) {
            p.piv = std::move(set);
            return;
        }
        std::vector<std::int32_t> seen;
        // Components of the set, from its smallest node.
        bfs(*std::min_element(set.begin(), set.end()), t, seen);
        if (seen.size() < set.size()) {
            children.push_back(seen);
            for (std::int32_t v : set) {
                if (level[v] >= 0) continue;
                std::vector<std::int32_t> c;
                bfs(v, t, c);
                children.push_back(std::move(c));
            }
            for (auto& c : children) clear_levels(c);
            p.pass = true;
            return;
        }
        // A pseudo-peripheral start: from the farthest node, while the depth grows.
        std::int32_t depth = level[seen.back()];
        for (int it = 0; it < 2; ++it) {
            std::int32_t far = seen.back();
            for (std::int32_t v : seen)
                if (level[v] == depth && v < far) far = v;
            clear_levels(seen);
            bfs(far, t, seen);
            const std::int32_t d = level[seen.back()];
            if (d <= depth && it > 0) break;
            depth = d;
        }
        depth = level[seen.back()];
        if (depth < 2) {  // no level separates two nonempty sides
            clear_levels(seen);
            p.piv = std::move(set);
            return;
        }
        std::vector<std::int64_t> count(static_cast<std::size_t>(depth) + 1, 0);
        for (std::int32_t v : seen) ++count[static_cast<std::size_t>(level[v])];
        const std::int64_t n = static_cast<std::int64_t>(seen.size());
        std::int32_t best = -1, half = -1;
        std::int64_t before = count[0];
        for (std::int32_t l = 1; l < depth; ++l) {
            const std::int64_t after = n - before - count[l];
            if (half < 0 && before + count[l] >= n / 2) half = l;
            if (5 * before >= n && 5 * after >= n && (best < 0 || count[l] < count[best])) best = l;
            before += count[l];
        }
        const std::int32_t sl = best >= 0 ? best : (half >= 0 ? half : depth / 2);
        std::vector<std::int32_t> a, b;
        for (std::int32_t v : seen) {
            const std::int32_t l = level[v];
            if (l < sl) {
                a.push_back(v);
            } else if (l > sl) {
                b.push_back(v);
            } else {
                bool touches_b = false;
                for (std::int64_t e = ptr[v]; e < ptr[v + 1] && !touches_b; ++e) {
                    const std::int32_t w = adj[static_cast<std::size_t>(e)];
                    touches_b = w >= 0 && mark[w] == t && level[w] == sl + 1;
                }
                (touches_b ? p.piv : a).push_back(v);
            }
        }
        clear_levels(seen);
        if (!a.empty()) children.push_back(std::move(a));
        if (!b.empty()) children.push_back(std::move(b));
        std::vector<std::int32_t>().swap(set);
    }

    // The subtree roots of `all`, appended to `roots`.
    void dissect(std::vector<std::int32_t> all, std::vector<std::int32_t>& roots) {
        if (all.empty()) return;
        parts.push_back({});
        parts.back().set = std::move(all);
        std::vector<std::int32_t> frontier{0};
        while (!frontier.empty()) {
            const int nf = static_cast<int>(frontier.size());
            std::vector<std::vector<std::vector<std::int32_t>>> children(frontier.size());
#pragma omp parallel for schedule(dynamic, 1) if (nf > 1)
            for (int i = 0; i < nf; ++i)
                for (std::int32_t v : parts[frontier[i]].set) mark[v] = frontier[i];
#pragma omp parallel for schedule(dynamic, 1) if (nf > 1)
            for (int i = 0; i < nf; ++i) split(parts[frontier[i]], frontier[i], children[i]);
            std::vector<std::int32_t> next;
            for (int i = 0; i < nf; ++i)
                for (auto& c : children[i]) {
                    const std::int32_t k = static_cast<std::int32_t>(parts.size());
                    parts.push_back({});
                    parts.back().set = std::move(c);
                    parts[frontier[i]].sub.push_back(k);
                    next.push_back(k);
                }
            frontier = std::move(next);
        }
        emit(0, roots);
    }

    // Post-order layout of part k: its fronts' roots appended to `roots`.
    void emit(std::int32_t k, std::vector<std::int32_t>& roots) {
        Part& p = parts[k];
        if (p.pass) {
            for (std::int32_t c : p.sub) emit(c, roots);
            return;
        }
        std::vector<std::int32_t> ch;
        for (std::int32_t c : p.sub) emit(c, ch);
        if (p.piv.empty() && ch.size() == 1) {
            roots.push_back(ch[0]);
            return;
        }
        roots.push_back(add(p.piv, std::move(ch)));
    }
};

}  // namespace

namespace {

// Front structures, bottom up, of a tree whose nodes' children are `kids`.
void build_fronts(FrontalSymbolic& s, const std::vector<std::vector<std::int32_t>>& kids) {
    const std::int32_t n = static_cast<std::int32_t>(s.cells.size());
    s.amap.assign(s.nbr.size(), -1);
    std::vector<std::int32_t> stamp(static_cast<std::size_t>(n), -1), loc(static_cast<std::size_t>(n), -1);
    std::vector<std::int32_t> upd;
    std::int64_t lpos = 0, stack = 0, peak = 0;
    const std::int32_t n_nodes = static_cast<std::int32_t>(s.nodes.size());
    for (std::int32_t t = 0; t < n_nodes; ++t) {
        FrontalSymbolic::Node& node = s.nodes[t];
        node.child_begin = static_cast<std::int32_t>(s.children.size());
        for (std::int32_t c : kids[t]) s.children.push_back(c);
        node.child_end = static_cast<std::int32_t>(s.children.size());
        const std::int32_t end = node.first + node.npiv;
        upd.clear();
        for (std::int32_t p = node.first; p < end; ++p)
            for (std::int64_t k = s.entry_begin(p); k < s.entry_end(p); ++k) {
                const std::int32_t q = s.nbr[static_cast<std::size_t>(k)];
                if (q >= end && stamp[q] != t) {
                    stamp[q] = t;
                    upd.push_back(q);
                }
            }
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const FrontalSymbolic::Node& c = s.nodes[s.children[ci]];
            for (std::int32_t i = 0; i < c.nupd; ++i) {
                const std::int32_t r = s.rows[static_cast<std::size_t>(c.upd + i)];
                if (r >= end && stamp[r] != t) {
                    stamp[r] = t;
                    upd.push_back(r);
                }
            }
        }
        std::sort(upd.begin(), upd.end());
        node.upd = static_cast<std::int32_t>(s.rows.size());
        node.nupd = static_cast<std::int32_t>(upd.size());
        s.rows.insert(s.rows.end(), upd.begin(), upd.end());
        s.relmap.resize(s.rows.size(), -1);
        for (std::int32_t p = node.first; p < end; ++p) loc[p] = p - node.first;
        for (std::int32_t i = 0; i < node.nupd; ++i) loc[upd[i]] = node.npiv + i;
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const FrontalSymbolic::Node& c = s.nodes[s.children[ci]];
            for (std::int32_t i = 0; i < c.nupd; ++i)
                s.relmap[static_cast<std::size_t>(c.upd + i)] = loc[s.rows[static_cast<std::size_t>(c.upd + i)]];
        }
        for (std::int32_t p = node.first; p < end; ++p)
            for (std::int64_t k = s.entry_begin(p); k < s.entry_end(p); ++k) {
                const std::int32_t q = s.nbr[static_cast<std::size_t>(k)];
                if (q > p) s.amap[static_cast<std::size_t>(k)] = loc[q];
            }
        const std::int64_t m = node.npiv + node.nupd;
        node.lpos = lpos;
        lpos += m * node.npiv;
        s.max_front = std::max(s.max_front, m * m);
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const std::int64_t u = s.nodes[s.children[ci]].nupd;
            stack -= u * u;
        }
        stack += static_cast<std::int64_t>(node.nupd) * node.nupd;
        peak = std::max(peak, stack);
    }
    s.lsize = lpos;
    s.stack_size = peak;
}

}  // namespace

std::shared_ptr<const FrontalSymbolic> frontal_analyze(i64 bx, i64 by, const std::uint8_t* unknown,
                                                       bool four_neighbors) {
    auto sym = std::make_shared<FrontalSymbolic>();
    FrontalSymbolic& s = *sym;
    s.bx = bx;
    s.by = by;
    s.four = four_neighbors;
    TreeBuilder tb{unknown, bx, {}, {}, {}};
    tb.build(0, by, 0, bx);
    s.cells = std::move(tb.cells);
    s.nodes = std::move(tb.nodes);
    const i64 n = static_cast<i64>(s.cells.size());
    s.unknown_of.assign(static_cast<std::size_t>(bx * by), -1);
    for (i64 k = 0; k < n; ++k) s.unknown_of[static_cast<std::size_t>(s.cells[k])] = static_cast<std::int32_t>(k);

    const Offset* nb = four_neighbors ? kNeighbours4 : kNeighbours8;
    const int nd = four_neighbors ? 4 : 8;
    s.nbr.assign(static_cast<std::size_t>(n * nd), -1);
    for (i64 k = 0; k < n; ++k) {
        const i64 y = s.cells[k] / bx, x = s.cells[k] % bx;
        for (int d = 0; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            if (yy < 0 || yy >= by || xx < 0 || xx >= bx) continue;
            s.nbr[static_cast<std::size_t>(k * nd + d)] = s.unknown_of[static_cast<std::size_t>(yy * bx + xx)];
        }
    }
    build_fronts(s, tb.kids);
    return sym;
}

std::shared_ptr<const FrontalSymbolic> frontal_analyze_graph(i64 bx, i64 by, const std::vector<i64>& cell,
                                                             const std::vector<std::int64_t>& adj_ptr,
                                                             const std::vector<std::int32_t>& adj) {
    auto sym = std::make_shared<FrontalSymbolic>();
    FrontalSymbolic& s = *sym;
    s.bx = bx;
    s.by = by;
    const std::int32_t n = static_cast<std::int32_t>(cell.size());
    if (std::none_of(cell.begin(), cell.end(), [](i64 c) { return c >= 0; })) {
        // No geometry (a network): the graph's own nested dissection.
        GraphDissection gd(n, adj_ptr, adj);
        std::vector<std::int32_t> all(static_cast<std::size_t>(n)), roots;
        for (std::int32_t c = 0; c < n; ++c) all[c] = c;
        gd.dissect(std::move(all), roots);
        if (roots.size() > 1) {
            std::vector<std::int32_t> none;
            gd.add(none, roots);
        }
        s.perm = std::move(gd.order);
        s.nodes = std::move(gd.nodes);
        s.cells.assign(static_cast<std::size_t>(n), -1);
        std::vector<std::int32_t> iperm(static_cast<std::size_t>(n), -1);
        for (std::int32_t k = 0; k < n; ++k) iperm[static_cast<std::size_t>(s.perm[k])] = k;
        s.unknown_of.assign(static_cast<std::size_t>(bx * by), -1);
        s.adj_ptr = adj_ptr;
        s.nbr.resize(adj.size());
        for (std::size_t k = 0; k < adj.size(); ++k) s.nbr[k] = adj[k] >= 0 ? iperm[static_cast<std::size_t>(adj[k])] : -1;
        build_fronts(s, gd.kids);
        return sym;
    }
    std::vector<std::uint8_t> mark(static_cast<std::size_t>(bx * by), 0);
    std::vector<std::int32_t> node_of_cell(static_cast<std::size_t>(bx * by), -1);
    for (std::int32_t c = 0; c < n; ++c)
        if (cell[c] >= 0) {
            mark[static_cast<std::size_t>(cell[c])] = 1;
            node_of_cell[static_cast<std::size_t>(cell[c])] = c;
        }
    TreeBuilder tb{mark.data(), bx, {}, {}, {}};
    const std::int32_t root = tb.build(0, by, 0, bx);
    s.cells = std::move(tb.cells);
    s.nodes = std::move(tb.nodes);
    s.perm.reserve(static_cast<std::size_t>(n));
    for (i64 b : s.cells) s.perm.push_back(node_of_cell[static_cast<std::size_t>(b)]);
    // Merged nodes: one front at the root.
    const std::int32_t first_merged = static_cast<std::int32_t>(s.perm.size());
    for (std::int32_t c = 0; c < n; ++c)
        if (cell[c] < 0) {
            s.perm.push_back(c);
            s.cells.push_back(-1);
        }
    if (static_cast<std::int32_t>(s.perm.size()) > first_merged) {
        FrontalSymbolic::Node top;
        top.first = first_merged;
        top.npiv = static_cast<std::int32_t>(s.perm.size()) - first_merged;
        s.nodes.push_back(top);
        tb.kids.push_back(root >= 0 ? std::vector<std::int32_t>{root} : std::vector<std::int32_t>{});
    }
    std::vector<std::int32_t> iperm(static_cast<std::size_t>(n), -1);
    for (std::int32_t k = 0; k < n; ++k) iperm[static_cast<std::size_t>(s.perm[k])] = k;
    s.unknown_of.assign(static_cast<std::size_t>(bx * by), -1);
    for (std::int32_t k = 0; k < n; ++k)
        if (s.cells[k] >= 0) s.unknown_of[static_cast<std::size_t>(s.cells[k])] = k;
    s.adj_ptr = adj_ptr;
    s.nbr.resize(adj.size());
    for (std::size_t k = 0; k < adj.size(); ++k) s.nbr[k] = adj[k] >= 0 ? iperm[static_cast<std::size_t>(adj[k])] : -1;
    build_fronts(s, tb.kids);
    return sym;
}

void CellSystem::analyze(i64 bx, i64 by, const std::uint8_t* unknown, bool four_neighbors) {
    share(frontal_analyze(bx, by, unknown, four_neighbors));
}

void CellSystem::share(std::shared_ptr<const FrontalSymbolic> symbolic) {
    sym_ = std::move(symbolic);
    const std::size_t n = sym_->cells.size();
    diag_.assign(n, 1.0);
    off_.assign(sym_->nbr.size(), 0.0);
}

void CellSystem::set_values(const double* diag, const double* off) {
    std::copy(diag, diag + diag_.size(), diag_.begin());
    std::copy(off, off + off_.size(), off_.begin());
}

void CellSystem::assemble(const double* g, const std::uint8_t* present, const double* extra_diag, double shift) {
    const FrontalSymbolic& s = *sym_;
    const Offset* nb = s.four ? kNeighbours4 : kNeighbours8;
    const int nd = s.four ? 4 : 8;
    const i64 n = size();
    for (i64 j = 0; j < n; ++j) {
        const i64 c = s.cells[j];
        double* off = off_.data() + j * nd;
        if (!present[c]) {
            std::fill(off, off + nd, 0.0);
            diag_[j] = 1.0;
            continue;
        }
        const i64 y = c / s.bx, x = c % s.bx;
        const double gc = g[c];
        double diag = 0.0;
        for (int d = 0; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            if (yy < 0 || yy >= s.by || xx < 0 || xx >= s.bx || !present[yy * s.bx + xx]) {
                off[d] = 0.0;
                continue;
            }
            const double w = edge_weight(gc, g[yy * s.bx + xx], nb[d].diagonal);
            diag += w;
            off[d] = -w;
        }
        if (extra_diag) diag += extra_diag[c];
        diag_[j] = diag + shift;
    }
}

bool CellSystem::factorize() { return factorize(nullptr); }

namespace {

// Structures with more doubles of L than this are factorised apart: every
// front built in an aligned workspace and only its L copied out (so the
// arithmetic does not depend on where a front is built), subtrees in
// parallel, large fronts by blocked kernels in parallel. The blocks are fixed,
// so the result is the same with any number of threads.
constexpr std::int64_t kLargeL = std::int64_t(1) << 22;
constexpr std::int64_t kBigFront = 128;  // pivots from which a front uses the blocked kernels
constexpr std::int64_t kBlock = 128;     // column block of the blocked kernels
constexpr std::int64_t kRowChunk = 256;  // row chunk of the panel solves

int team_size() {
#if defined(_OPENMP)
    return omp_in_parallel() ? 1 : omp_get_max_threads();
#else
    return 1;
#endif
}

// Partial Cholesky of the front F (m × m, column-major, lower triangle): the
// first np columns factorised, the trailing block updated by their Schur
// complement. Fronts of kBigFront pivots or more: right-looking, by column
// blocks of kBlock (LLT of the diagonal block, the panel below solved in row
// chunks, the trailing lower part updated block column by block column; the
// strictly upper parts of the diagonal blocks get garbage, never read).
bool dense_front(double* F, std::int64_t m, std::int64_t np, bool parallel) {
    const std::int64_t nu = m - np;
    if (np == 0) return true;
    if (np < kBigFront) {
        Block F11(F, np, np, Eigen::OuterStride<>(m));
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd, 0, Eigen::OuterStride<>>> llt(F11);
        if (llt.info() != Eigen::Success) return false;
        if (nu > 0) {
            Block F21(F + np, nu, np, Eigen::OuterStride<>(m));
            F11.triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheRight>(F21);
            Block F22(F + np * m + np, nu, nu, Eigen::OuterStride<>(m));
            F22.selfadjointView<Eigen::Lower>().rankUpdate(F21, -1.0);
        }
        return true;
    }
    (void)parallel;
    for (std::int64_t k0 = 0; k0 < np; k0 += kBlock) {
        const std::int64_t kb = std::min(kBlock, np - k0);
        Block D(F + k0 * m + k0, kb, kb, Eigen::OuterStride<>(m));
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd, 0, Eigen::OuterStride<>>> llt(D);
        if (llt.info() != Eigen::Success) return false;
        const std::int64_t r0 = k0 + kb, nr = m - r0;
        if (nr == 0) continue;
        const long long chunks = static_cast<long long>((nr + kRowChunk - 1) / kRowChunk);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) if (parallel && chunks > 1)
#endif
        for (long long c = 0; c < chunks; ++c) {
            const std::int64_t i0 = r0 + c * kRowChunk, ni = std::min(kRowChunk, m - i0);
            Block P(F + k0 * m + i0, ni, kb, Eigen::OuterStride<>(m));
            D.triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheRight>(P);
        }
        // Tiles: column block [j0, j0 + kBlock) by row chunk [i0, i0 + kRowChunk), i0 >= j0.
        const std::int64_t blocks = (nr + kBlock - 1) / kBlock;
        std::int64_t tiles = 0;
        for (std::int64_t bb = 0; bb < blocks; ++bb) tiles += (m - (r0 + bb * kBlock) + kRowChunk - 1) / kRowChunk;
        const long long n_tiles = static_cast<long long>(tiles);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) if (parallel && n_tiles > 1)
#endif
        for (long long tt = 0; tt < n_tiles; ++tt) {
            std::int64_t rest = tt, bb = 0;
            for (;; ++bb) {
                const std::int64_t c = (m - (r0 + bb * kBlock) + kRowChunk - 1) / kRowChunk;
                if (rest < c) break;
                rest -= c;
            }
            const std::int64_t j0 = r0 + bb * kBlock, nj = std::min(kBlock, m - j0);
            const std::int64_t i0 = j0 + rest * kRowChunk, ni = std::min(kRowChunk, m - i0);
            const ConstBlock Pi(F + k0 * m + i0, ni, kb, Eigen::OuterStride<>(m));
            const ConstBlock Pj(F + k0 * m + j0, nj, kb, Eigen::OuterStride<>(m));
            Block C(F + j0 * m + i0, ni, nj, Eigen::OuterStride<>(m));
            C.noalias() -= Pi * Pj.transpose();
        }
    }
    return true;
}

// The subtrees factorised in parallel: nodes are in post-order, so node t's
// subtree is the node range [begin[t], t]. From the roots, the subtree of
// most work is replaced by its children (and goes "above") while there are
// fewer than `parts` subtrees and it holds more than a 1/(4 parts) of the work.
struct Plan {
    std::vector<std::int32_t> begin, subtree;  // node -> first node of its subtree; -> its part (-1: above)
    std::vector<std::int32_t> roots;           // part roots, heaviest first
    std::vector<std::int64_t> peak;            // per part: its update-stack peak (doubles)
};

Plan make_plan(const FrontalSymbolic& s, int parts) {
    const std::int32_t n = static_cast<std::int32_t>(s.nodes.size());
    Plan p;
    p.begin.resize(static_cast<std::size_t>(n));
    std::vector<double> work(static_cast<std::size_t>(n), 0.0);
    std::vector<std::uint8_t> has_parent(static_cast<std::size_t>(n), 0);
    for (std::int32_t t = 0; t < n; ++t) {
        const FrontalSymbolic::Node& node = s.nodes[t];
        const double m = static_cast<double>(node.npiv) + node.nupd;
        work[t] += static_cast<double>(node.npiv) * m * m + m;
        p.begin[t] = node.child_begin < node.child_end ? p.begin[s.children[node.child_begin]] : t;
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            work[t] += work[s.children[ci]];
            has_parent[s.children[ci]] = 1;
        }
    }
    auto heavier = [&](std::int32_t a, std::int32_t b) { return work[a] < work[b] || (work[a] == work[b] && a > b); };
    std::priority_queue<std::int32_t, std::vector<std::int32_t>, decltype(heavier)> heap(heavier);
    double total = 0.0;
    for (std::int32_t t = 0; t < n; ++t)
        if (!has_parent[t]) {
            heap.push(t);
            total += work[t];
        }
    while (static_cast<int>(heap.size()) < parts) {
        const std::int32_t t = heap.top();
        const FrontalSymbolic::Node& node = s.nodes[t];
        if (node.child_begin == node.child_end || work[t] * 4 * parts <= total) break;
        heap.pop();
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) heap.push(s.children[ci]);
    }
    while (!heap.empty()) {
        p.roots.push_back(heap.top());
        heap.pop();
    }
    p.subtree.assign(static_cast<std::size_t>(n), -1);
    p.peak.assign(p.roots.size(), 0);
    for (std::size_t k = 0; k < p.roots.size(); ++k) {
        const std::int32_t r = p.roots[k];
        std::int64_t top = 0, peak = 0;
        for (std::int32_t t = p.begin[r]; t <= r; ++t) {
            p.subtree[t] = static_cast<std::int32_t>(k);
            const FrontalSymbolic::Node& node = s.nodes[t];
            for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
                const std::int64_t u = s.nodes[s.children[ci]].nupd;
                top -= u * u;
            }
            top += static_cast<std::int64_t>(node.nupd) * node.nupd;
            peak = std::max(peak, top);
        }
        p.peak[k] = peak;
    }
    return p;
}

}  // namespace

// The large-structure factorisation (see kLargeL). The parts (subtrees) run in
// parallel; the fronts above them by heights, several fronts of one height in
// parallel or a lone one with parallel kernels. The forward substitution of y
// (as factorize(y) fuses it) runs inside the parts for their own unknowns; their
// contributions to other unknowns are recorded and, with the forward steps of
// the fronts above, applied in a final pass in post-order: y gets the same
// operations in the same order as in a single thread.
bool CellSystem::factorize_large(double* y) {
    const FrontalSymbolic& s = *sym_;
    const std::int32_t n_nodes = static_cast<std::int32_t>(s.nodes.size());
    const int threads = team_size();
    const Plan plan = make_plan(s, threads > 1 ? 2 * threads : 1);
    double* lstore = l_.get(static_cast<std::size_t>(s.lsize));
    std::vector<std::vector<double>> held(static_cast<std::size_t>(n_nodes));  // updates of part roots and above
    const int n_parts = static_cast<int>(plan.roots.size());
    std::vector<std::vector<std::pair<std::int32_t, double>>> deferred(static_cast<std::size_t>(n_parts));

    // Forward step of node t: y's own pivots, then the contributions to its update
    // rows, applied when in [lo, hi) and recorded in `out` otherwise.
    auto forward_step = [&](std::int32_t t, AlignedBuffer& tmpbuf, std::int32_t lo, std::int32_t hi,
                            std::vector<std::pair<std::int32_t, double>>* out) {
        const FrontalSymbolic::Node& node = s.nodes[t];
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        if (np == 0) return;
        const ConstBlock Lb(lstore + node.lpos, m, np, Eigen::OuterStride<>(m));
        Eigen::Map<Eigen::VectorXd> yp(y + node.first, np);
        Lb.topRows(np).triangularView<Eigen::Lower>().solveInPlace(yp);
        if (nu == 0) return;
        double* tmp = tmpbuf.get(static_cast<std::size_t>(nu));
        Eigen::Map<Eigen::MatrixXd> tm(tmp, nu, 1);
        tm.col(0).noalias() = Lb.bottomRows(nu) * yp.col(0);
        const std::int32_t* rows = s.rows.data() + node.upd;
        for (std::int64_t i = 0; i < nu; ++i) {
            if (rows[i] >= lo && rows[i] < hi) y[rows[i]] -= tmp[i];
            else out->emplace_back(rows[i], tmp[i]);
        }
    };

    // One front: assembled in F (m × m, aligned), children's updates from `child(ci)`,
    // factorised, L copied to its place, the update matrix handed to `keep`.
    auto front = [&](std::int32_t t, AlignedBuffer& work, bool parallel, auto&& child, auto&& keep) -> bool {
        const FrontalSymbolic::Node& node = s.nodes[t];
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        double* F = work.get(static_cast<std::size_t>(std::max<std::int64_t>(m * m, 1)));
        for (std::int64_t j = 0; j < m; ++j) std::fill(F + j * m + j, F + j * m + m, 0.0);
        for (std::int32_t p = node.first; p < node.first + node.npiv; ++p) {
            const std::int64_t j = p - node.first;
            F[j * m + j] += diag_[static_cast<std::size_t>(s.value_index(p))];
            for (std::int64_t k = s.entry_begin(p); k < s.entry_end(p); ++k) {
                const std::int32_t a = s.amap[static_cast<std::size_t>(k)];
                if (a >= 0) F[j * m + a] += off_[static_cast<std::size_t>(k)];
            }
        }
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const FrontalSymbolic::Node& c = s.nodes[s.children[ci]];
            const std::int64_t u = c.nupd;
            const std::int32_t* rel = s.relmap.data() + c.upd;
            const double* U = child(ci);
            for (std::int64_t jj = 0; jj < u; ++jj) {
                double* col = F + static_cast<std::int64_t>(rel[jj]) * m;
                const double* ucol = U + jj * u;
                for (std::int64_t ii = jj; ii < u; ++ii) col[rel[ii]] += ucol[ii];
            }
        }
        if (!dense_front(F, m, np, parallel)) return false;
        double* L = lstore + node.lpos;
        for (std::int64_t j = 0; j < np; ++j) std::copy(F + j * m + j, F + j * m + m, L + j * m + j);
        keep(F + np * m + np, m, nu);
        return true;
    };

    // A part: its subtree on one thread, its updates on a stack of its own.
    auto run_part = [&](int k, AlignedBuffer& work, AlignedBuffer& stackbuf, AlignedBuffer& tmpbuf,
                        bool parallel) -> bool {
        const std::int32_t r = plan.roots[static_cast<std::size_t>(k)];
        const std::int32_t b = plan.begin[r];
        const std::int32_t lo = s.nodes[b].first, hi = s.nodes[r].first + s.nodes[r].npiv;
        double* stack = stackbuf.get(static_cast<std::size_t>(std::max<std::int64_t>(plan.peak[k], 1)));
        std::int64_t top = 0;
        for (std::int32_t t = b; t <= r; ++t) {
            const FrontalSymbolic::Node& node = s.nodes[t];
            std::int64_t below = top;
            for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
                const std::int64_t u = s.nodes[s.children[ci]].nupd;
                below -= u * u;
            }
            std::int64_t at = below;
            auto child = [&](std::int32_t ci) {
                const double* U = stack + at;
                const std::int64_t u = s.nodes[s.children[ci]].nupd;
                at += u * u;
                return U;
            };
            auto keep = [&](const double* F22, std::int64_t m, std::int64_t nu) {
                double* U;
                if (t == r) {
                    held[t].resize(static_cast<std::size_t>(nu * nu));
                    U = held[t].data();
                } else {
                    U = stack + below;
                }
                for (std::int64_t j = 0; j < nu; ++j) std::copy(F22 + j * m + j, F22 + j * m + nu, U + j * nu + j);
                top = below + (t == r ? 0 : nu * nu);
            };
            if (!front(t, work, parallel, child, keep)) return false;
            if (y) forward_step(t, tmpbuf, lo, hi, &deferred[static_cast<std::size_t>(k)]);
        }
        return true;
    };

    bool failed = false;
    if (n_parts > 1) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
        {
            AlignedBuffer work, stackbuf, tmpbuf;
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 1)
#endif
            for (int k = 0; k < n_parts; ++k) {
                if (!run_part(k, work, stackbuf, tmpbuf, false)) {
#if defined(_OPENMP)
#pragma omp critical(circuit_factorize_failed)
#endif
                    failed = true;
                }
            }
        }
    } else if (n_parts == 1) {
        AlignedBuffer stackbuf;
        failed = !run_part(0, work_, stackbuf, scratch_.tmp, threads > 1);
    }
    if (failed) return false;

    // Above the parts, by heights (a front's height: 1 + its highest child above, 0 without one).
    std::vector<std::int32_t> height(static_cast<std::size_t>(n_nodes), -1);
    std::int32_t max_height = -1;
    for (std::int32_t t = 0; t < n_nodes; ++t) {
        if (plan.subtree[t] >= 0) continue;
        const FrontalSymbolic::Node& node = s.nodes[t];
        std::int32_t h = 0;
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci)
            if (plan.subtree[s.children[ci]] < 0) h = std::max(h, height[s.children[ci]] + 1);
        height[t] = h;
        max_height = std::max(max_height, h);
    }
    for (std::int32_t h = 0; h <= max_height; ++h) {
        std::vector<std::int32_t> level;
        for (std::int32_t t = 0; t < n_nodes; ++t)
            if (height[t] == h) level.push_back(t);
        const int n_level = static_cast<int>(level.size());
        auto one = [&](int i, AlignedBuffer& work, bool parallel) -> bool {
            const std::int32_t t = level[static_cast<std::size_t>(i)];
            const FrontalSymbolic::Node& node = s.nodes[t];
            auto child = [&](std::int32_t ci) { return held[s.children[ci]].data(); };
            auto keep = [&](const double* F22, std::int64_t m, std::int64_t nu) {
                for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci)
                    std::vector<double>().swap(held[s.children[ci]]);
                held[t].resize(static_cast<std::size_t>(nu * nu));
                double* U = held[t].data();
                for (std::int64_t j = 0; j < nu; ++j) std::copy(F22 + j * m + j, F22 + j * m + nu, U + j * nu + j);
            };
            return front(t, work, parallel, child, keep);
        };
        if (threads > 1 && n_level > 1) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
            {
                AlignedBuffer work;
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 1)
#endif
                for (int i = 0; i < n_level; ++i) {
                    if (!one(i, work, false)) {
#if defined(_OPENMP)
#pragma omp critical(circuit_factorize_failed)
#endif
                        failed = true;
                    }
                }
            }
        } else {
            for (int i = 0; i < n_level && !failed; ++i) failed = !one(i, work_, threads > 1);
        }
        if (failed) return false;
    }

    // y: the parts' recorded contributions at their roots, the fronts above in turn.
    if (y) {
        std::vector<std::pair<std::int32_t, double>> none;
        const std::int32_t hi = static_cast<std::int32_t>(s.cells.size());
        for (std::int32_t t = 0; t < n_nodes; ++t) {
            const std::int32_t k = plan.subtree[t];
            if (k >= 0) {
                if (t == plan.roots[static_cast<std::size_t>(k)])
                    for (const auto& rv : deferred[static_cast<std::size_t>(k)]) y[rv.first] -= rv.second;
                continue;
            }
            forward_step(t, scratch_.tmp, 0, hi, &none);
        }
    }
    return true;
}

bool CellSystem::factorize_solve(const double* b, double* x) {
    const i64 n = size();
    double* y = scratch_.rhs.get(static_cast<std::size_t>(n));
    std::copy(b, b + n, y);
    if (!factorize(y)) return false;
    backward(y, 1, scratch_.tmp);
    std::copy(y, y + n, x);
    return true;
}

bool CellSystem::factorize(double* y) {
    const FrontalSymbolic& s = *sym_;
    if (s.lsize >= kLargeL) return factorize_large(y);
    double* stack = stack_.get(static_cast<std::size_t>(s.stack_size));
    // Every front is built where its columns of L go: its first npiv columns
    // stay as L, its trailing ones (the update matrix, copied to the stack)
    // spill over the space of the fronts that follow, hence the margin.
    double* lstore = l_.get(static_cast<std::size_t>(s.lsize + s.max_front));
    std::int64_t top = 0;
    for (const FrontalSymbolic::Node& node : s.nodes) {
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        double* F = lstore + node.lpos;
        for (std::int64_t j = 0; j < m; ++j) std::fill(F + j * m + j, F + j * m + m, 0.0);
        for (std::int32_t p = node.first; p < node.first + node.npiv; ++p) {
            const std::int64_t j = p - node.first;
            F[j * m + j] += diag_[static_cast<std::size_t>(s.value_index(p))];
            for (std::int64_t k = s.entry_begin(p); k < s.entry_end(p); ++k) {
                const std::int32_t a = s.amap[static_cast<std::size_t>(k)];
                if (a >= 0) F[j * m + a] += off_[static_cast<std::size_t>(k)];
            }
        }
        // Extend-add the children's update matrices (the top of the stack, in order).
        std::int64_t below = top;
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const std::int64_t u = s.nodes[s.children[ci]].nupd;
            below -= u * u;
        }
        std::int64_t at = below;
        for (std::int32_t ci = node.child_begin; ci < node.child_end; ++ci) {
            const FrontalSymbolic::Node& c = s.nodes[s.children[ci]];
            const std::int64_t u = c.nupd;
            const std::int32_t* rel = s.relmap.data() + c.upd;
            const double* U = stack + at;
            for (std::int64_t jj = 0; jj < u; ++jj) {
                double* col = F + static_cast<std::int64_t>(rel[jj]) * m;
                const double* ucol = U + jj * u;
                for (std::int64_t ii = jj; ii < u; ++ii) col[rel[ii]] += ucol[ii];
            }
            at += u * u;
        }
        top = below;
        if (np > 0) {
            Block F11(F, np, np, Eigen::OuterStride<>(m));
            Eigen::LLT<Eigen::Ref<Eigen::MatrixXd, 0, Eigen::OuterStride<>>> llt(F11);
            if (llt.info() != Eigen::Success) return false;
            if (nu > 0) {
                Block F21(F + np, nu, np, Eigen::OuterStride<>(m));
                F11.triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheRight>(F21);
                Block F22(F + np * m + np, nu, nu, Eigen::OuterStride<>(m));
                F22.selfadjointView<Eigen::Lower>().rankUpdate(F21, -1.0);
            }
            if (y) {
                // As forward() with one right-hand side.
                const ConstBlock L(F, m, np, Eigen::OuterStride<>(m));
                Eigen::Map<Eigen::VectorXd> yp(y + node.first, np);
                L.topRows(np).triangularView<Eigen::Lower>().solveInPlace(yp);
                if (nu > 0) {
                    double* tmp = scratch_.tmp.get(static_cast<std::size_t>(nu));
                    Eigen::Map<Eigen::MatrixXd> t(tmp, nu, 1);
                    t.col(0).noalias() = L.bottomRows(nu) * yp.col(0);
                    const std::int32_t* rows = s.rows.data() + node.upd;
                    for (std::int64_t i = 0; i < nu; ++i) y[rows[i]] -= tmp[i];
                }
            }
        }
        // Push the update matrix (lower triangle, nu × nu).
        double* U = stack + top;
        const double* F22 = F + np * m + np;
        for (std::int64_t j = 0; j < nu; ++j) std::copy(F22 + j * m + j, F22 + j * m + nu, U + j * nu + j);
        top += nu * nu;
    }
    return true;
}

void CellSystem::solve(const double* b, double* x) {
    const std::size_t n = sym_->cells.size();
    if (x != b) std::copy(b, b + n, x);
    solve_many(x, 1);
}

void CellSystem::solve_many(double* xout, i64 k) { solve_many(xout, k, scratch_); }

void CellSystem::solve_many(double* xout, i64 k, SolveScratch& scratch) const {
    const i64 n = size();
    // Solved in an aligned copy (see AlignedBuffer).
    double* x = scratch.rhs.get(static_cast<std::size_t>(n * k));
    std::copy(xout, xout + n * k, x);
    forward(x, k, scratch.tmp);
    backward(x, k, scratch.tmp);
    std::copy(x, x + n * k, xout);
}

namespace {

// The right-hand sides of one node's pivots: a vector for one (the same
// kernels as the forward substitution fused into factorize), a block for more.
template <class F>
void with_pivots(double* x, std::int64_t first, std::int64_t np, i64 n, i64 k, F&& f) {
    if (k == 1) {
        Eigen::Map<Eigen::VectorXd> xp(x + first, np);
        f(xp);
    } else {
        Block xp(x + first, np, k, Eigen::OuterStride<>(n));
        f(xp);
    }
}

}  // namespace

void CellSystem::forward(double* x, i64 k, AlignedBuffer& scratch) const {
    const FrontalSymbolic& s = *sym_;
    const i64 n = size();
    const double* lstore = l_.p.get();
    std::int64_t max_upd = 0;
    for (const FrontalSymbolic::Node& node : s.nodes) max_upd = std::max<std::int64_t>(max_upd, node.nupd);
    double* tmp = scratch.get(static_cast<std::size_t>(max_upd * k));
    for (const FrontalSymbolic::Node& node : s.nodes) {
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        if (np == 0) continue;
        const ConstBlock L(lstore + node.lpos, m, np, Eigen::OuterStride<>(m));
        with_pivots(x, node.first, np, n, k, [&](auto& xp) {
            L.topRows(np).template triangularView<Eigen::Lower>().solveInPlace(xp);
            if (nu > 0) {
                Eigen::Map<Eigen::MatrixXd> t(tmp, nu, k);
                if (k == 1) t.col(0).noalias() = L.bottomRows(nu) * xp.col(0);
                else t.noalias() = L.bottomRows(nu) * xp;
                const std::int32_t* rows = s.rows.data() + node.upd;
                for (i64 c = 0; c < k; ++c)
                    for (std::int64_t i = 0; i < nu; ++i) x[c * n + rows[i]] -= t(i, c);
            }
        });
    }
}

void CellSystem::backward(double* x, i64 k, AlignedBuffer& scratch) const {
    const FrontalSymbolic& s = *sym_;
    const i64 n = size();
    const double* lstore = l_.p.get();
    std::int64_t max_upd = 0;
    for (const FrontalSymbolic::Node& node : s.nodes) max_upd = std::max<std::int64_t>(max_upd, node.nupd);
    double* tmp = scratch.get(static_cast<std::size_t>(max_upd * k));
    for (auto it = s.nodes.rbegin(); it != s.nodes.rend(); ++it) {
        const FrontalSymbolic::Node& node = *it;
        const std::int64_t np = node.npiv, nu = node.nupd, m = np + nu;
        if (np == 0) continue;
        const ConstBlock L(lstore + node.lpos, m, np, Eigen::OuterStride<>(m));
        with_pivots(x, node.first, np, n, k, [&](auto& xp) {
            if (nu > 0) {
                Eigen::Map<Eigen::MatrixXd> t(tmp, nu, k);
                const std::int32_t* rows = s.rows.data() + node.upd;
                for (i64 c = 0; c < k; ++c)
                    for (std::int64_t i = 0; i < nu; ++i) t(i, c) = x[c * n + rows[i]];
                if (k == 1) xp.col(0).noalias() -= L.bottomRows(nu).transpose() * t.col(0);
                else xp.noalias() -= L.bottomRows(nu).transpose() * t;
            }
            L.topRows(np).template triangularView<Eigen::Lower>().transpose().solveInPlace(xp);
        });
    }
}

}  // namespace circuit
