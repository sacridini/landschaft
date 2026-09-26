// SPDX-License-Identifier: GPL-3.0-or-later

#include "circuit/omniscape.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "circuit/system.hpp"

namespace circuit {

namespace {

constexpr double kShift = 10 * std::numeric_limits<double>::epsilon();  // CHOLMOD's, as in grid.cpp

// A window: h × w cells with the ground (the target) at (gy, gx).
struct Shape {
    i64 h, w, gy, gx;
    bool operator==(const Shape& o) const { return h == o.h && w == o.w && gy == o.gy && gx == o.gx; }
    bool operator<(const Shape& o) const { return std::tie(h, w, gy, gx) < std::tie(o.h, o.w, o.gy, o.gx); }
};

// The unknowns of a window in nested-dissection order: the cells within
// `radius` of the ground (every cell when radius < 0), without the ground.
std::vector<i64> window_order(const Shape& s, i64 radius) {
    std::vector<std::uint8_t> mark(static_cast<std::size_t>(s.h * s.w), 0);
    for (i64 y = 0; y < s.h; ++y)
        for (i64 x = 0; x < s.w; ++x) {
            const i64 dy = y - s.gy, dx = x - s.gx;
            mark[y * s.w + x] = radius < 0 || dy * dy + dx * dx <= radius * radius;
        }
    mark[s.gy * s.w + s.gx] = 0;
    std::vector<i64> order;
    nested_dissection(mark.data(), s.w, 0, s.h, 0, s.w, order);
    return order;
}

struct Workspace {
    std::vector<std::uint8_t> present;
    std::vector<i64> stack;
    std::vector<double> v, rhs, x;
    void reserve(std::size_t cells) {
        present.resize(cells);
        v.resize(cells);
    }
};

// Circuitscape's compute_omniscape_current on one window with its direct
// ground at (gy, gx): only the ground's component can be solved. `sys` is
// analysed for the window's unknowns; with `factored` its factorization is
// that of `g` already. Node currents (0 elsewhere) go to `cur`.
void solve_window(CellSystem& sys, bool& factored, const double* g, const double* src, const Shape& s, bool four,
                  double* cur, Workspace& ws) {
    const i64 n = s.h * s.w;
    std::fill(cur, cur + n, 0.0);
    const i64 ground = s.gy * s.w + s.gx;
    if (!(g[ground] > 0)) return;  // the ground has no node
    ws.reserve(static_cast<std::size_t>(n));
    std::uint8_t* present = ws.present.data();
    std::fill(present, present + n, std::uint8_t{0});
    const Offset* nb = four ? kNeighbours4 : kNeighbours8;
    const int nd = four ? 4 : 8;
    const int lower = four ? kLower4 : kLower8;
    present[ground] = 1;
    ws.stack.assign(1, ground);
    double total = 0.0;
    while (!ws.stack.empty()) {
        const i64 p = ws.stack.back();
        ws.stack.pop_back();
        const i64 y = p / s.w, x = p % s.w;
        for (int d = 0; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            if (yy < 0 || yy >= s.h || xx < 0 || xx >= s.w) continue;
            const i64 q = yy * s.w + xx;
            if (!present[q] && g[q] > 0) {
                present[q] = 1;
                total += src[q];
                ws.stack.push_back(q);
            }
        }
    }
    if (total == 0) return;  // no source in the ground's component (the ground's own is dropped)

    const std::vector<i64>& cells = sys.cells();
    const std::size_t m = cells.size();
    if (!factored) {
        sys.assemble(g, present, nullptr, kShift);
        if (!sys.factorize()) throw std::runtime_error("an Omniscape window is not positive definite");
        factored = true;
    }
    ws.rhs.resize(m);
    ws.x.resize(m);
    for (std::size_t k = 0; k < m; ++k) ws.rhs[k] = present[cells[k]] ? src[cells[k]] : 0.0;
    sys.solve(ws.rhs.data(), ws.x.data());
    double* v = ws.v.data();
    std::fill(v, v + n, 0.0);
    for (std::size_t k = 0; k < m; ++k)
        if (present[cells[k]]) v[cells[k]] = ws.x[k];

    // Node currents as in grid.cpp (get_node_currents).
    const double inf = std::numeric_limits<double>::infinity();
    double maxin = -inf, maxout = -inf;
    for (i64 q = 0; q < n; ++q) {
        if (!present[q]) continue;
        const i64 y = q / s.w, x = q % s.w;
        for (int d = 0; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            if (yy < 0 || yy >= s.h || xx < 0 || xx >= s.w) continue;
            const i64 r = yy * s.w + xx;
            if (!present[r]) continue;
            const double b = edge_weight(g[q], g[r], nb[d].diagonal) * (v[r] - v[q]);
            if (d < lower) {
                if (b > maxin) maxin = b;
            } else if (b > maxout) {
                maxout = b;
            }
        }
    }
    for (i64 q = 0; q < n; ++q) {
        if (!present[q]) continue;
        const i64 y = q / s.w, x = q % s.w;
        double inflow = 0.0, outflow = 0.0;
        for (int d = 0; d < nd; ++d) {
            const i64 yy = y + nb[d].dy, xx = x + nb[d].dx;
            if (yy < 0 || yy >= s.h || xx < 0 || xx >= s.w) continue;
            const i64 r = yy * s.w + xx;
            if (!present[r]) continue;
            const double b = edge_weight(g[q], g[r], nb[d].diagonal) * (v[r] - v[q]);
            if (b > 0) {
                if (!(std::abs(b / maxin) < 1e-8)) inflow += b;
            } else if (b < 0) {
                if (!(std::abs(b / maxout) < 1e-8)) outflow -= b;
            }
        }
        cur[q] = inflow > outflow ? inflow : outflow;
    }
}

struct Target {
    i64 x, y;
    double amps;
    Shape shape;
    i64 y0, x0;  // window corner on the map
};

// The systems of one window shape, for one thread.
struct ShapeSystems {
    Shape shape{-1, -1, -1, -1};
    CellSystem disk;  // cells within radius + buffer: the landscape
    CellSystem rect;  // every cell, conductance 1: the flow potential
    bool rect_ready = false, rect_factored = false;
};

class ShapeCache {
public:
    ShapeSystems& get(const Shape& s, i64 radius, bool four) {
        for (auto& e : entries_)
            if (e && e->shape == s) return *e;
        auto e = std::make_unique<ShapeSystems>();
        e->shape = s;
        e->disk.analyze(s.w, s.h, window_order(s, radius), four);
        std::rotate(entries_.rbegin(), entries_.rbegin() + 1, entries_.rend());
        entries_[0] = std::move(e);
        return *entries_[0];
    }
    ShapeSystems& get_rect(const Shape& s, i64 radius, bool four) {
        ShapeSystems& e = get(s, radius, four);
        if (!e.rect_ready) {
            e.rect.analyze(s.w, s.h, window_order(s, -1), four);
            e.rect_ready = true;
        }
        return e;
    }

private:
    std::array<std::unique_ptr<ShapeSystems>, 3> entries_;
};

int thread_count(int max_threads) {
#if defined(_OPENMP)
    return max_threads > 0 ? max_threads : omp_get_max_threads();
#else
    (void)max_threads;
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

std::vector<double> omniscape_correction(const OmniscapeParams& p, i64 nx, i64 ny) {
    const i64 R = p.radius, B = p.buffer, D = R + B, W = 2 * D + 1;
    const i64 bs = p.block_size, br = (bs - 1) / 2;
    const Shape s{W, W, D, D};
    const std::size_t n = static_cast<std::size_t>(W * W);
    // Conductance 1 within radius + buffer.
    std::vector<double> g(n, -9999.0);
    std::vector<double> null_src(n, 0.0), block_src(n, 0.0);
    double n_sources = 0;
    for (i64 y = 0; y < W; ++y)
        for (i64 x = 0; x < W; ++x) {
            const i64 d2 = (y - D) * (y - D) + (x - D) * (x - D);
            if (d2 <= D * D) g[y * W + x] = 1.0;
            if (d2 <= R * R) {
                null_src[y * W + x] = 1.0;
                block_src[y * W + x] = 1.0;
                n_sources += 1;
            }
        }
    // One source per cell but the ground's, 1 / (n - 1) each.
    null_src[D * W + D] = 0.0;
    for (auto& v : null_src)
        if (v != 0.0) v = 1 / (n_sources - 1);
    // get_source on a map of ones: the block zeroed, bs² amps over the rest.
    for (i64 y = D - br; y <= std::min(D + br, ny - 1); ++y)
        for (i64 x = D - br; x <= std::min(D + br, nx - 1); ++x)
            if (y >= 0 && y < W && x >= 0 && x < W) block_src[y * W + x] = 0.0;
    double sum = 0.0;
    for (i64 x = 0; x < W; ++x)
        for (i64 y = 0; y < W; ++y)
            if (block_src[y * W + x] > 0) sum += block_src[y * W + x];
    const double amps = static_cast<double>(bs * bs);
    for (auto& v : block_src)
        if (v > 0) v = (v * amps) / sum;

    CellSystem sys;
    sys.analyze(W, W, window_order(s, D), p.four_neighbors);
    bool factored = false;
    Workspace ws;
    std::vector<double> block_cur(n), null_cur(n);
    solve_window(sys, factored, g.data(), block_src.data(), s, p.four_neighbors, block_cur.data(), ws);
    solve_window(sys, factored, g.data(), null_src.data(), s, p.four_neighbors, null_cur.data(), ws);

    // null_current_total: the null current of every target of a block, as
    // Omniscape sums it (i, then j), cut to the window.
    const i64 T = 2 * D + bs;
    std::vector<double> total(static_cast<std::size_t>(T * T), 0.0);
    for (i64 i = 0; i < bs; ++i)
        for (i64 j = 0; j < bs; ++j)
            for (i64 y = 0; y < W; ++y)
                for (i64 x = 0; x < W; ++x) total[(i + y) * T + (j + x)] += null_cur[y * W + x];
    double top = -std::numeric_limits<double>::infinity();
    for (double v : block_cur) top = std::max(top, v);
    const double floor = 1e-6 * top;
    std::vector<double> corr(n);
    for (i64 y = 0; y < W; ++y)
        for (i64 x = 0; x < W; ++x) {
            double b = block_cur[y * W + x];
            double t = total[(y + br) * T + (x + br)];
            if (b <= floor) b = 0.0;
            if (b == 0.0) t = 0.0;
            if (t == 0.0) t = 1.0;
            if (b == 0.0) b = 1.0;
            corr[y * W + x] = t / b;
        }
    return corr;
}

OmniscapeResult omniscape(const double* resistance, const double* source, i64 nx, i64 ny, const OmniscapeParams& p,
                          int max_threads) {
    if (p.radius < 0 || p.buffer < 0 || p.block_size < 1 || p.block_size % 2 == 0)
        throw std::invalid_argument("radius and buffer must be >= 0 and block_size odd");
    const i64 R = p.radius, B = p.buffer, D = R + B;
    const i64 bs = p.block_size, br = (bs - 1) / 2;
    const bool fp = p.flow_potential;
    const bool correct = p.correct_artifacts && bs > 1;
    OmniscapeResult out;
    out.current.assign(static_cast<std::size_t>(nx * ny), 0.0);
    if (fp) out.flow_potential.assign(static_cast<std::size_t>(nx * ny), 0.0);

    // Targets: block centres with a positive sum of sources (column-major sum).
    std::vector<Target> targets;
    for (i64 x = br; x + br < nx; x += bs)
        for (i64 y = br; y + br < ny; y += bs) {
            double amps = 0.0;
            for (i64 xx = x - br; xx <= x + br; ++xx)
                for (i64 yy = y - br; yy <= y + br; ++yy) {
                    const double v = source[yy * nx + xx];
                    if (!std::isnan(v)) amps += v;
                }
            if (!(amps > 0)) continue;
            Target t{x, y, amps, {}, std::max<i64>(y - D, 0), std::max<i64>(x - D, 0)};
            const i64 y1 = std::min(y + D, ny - 1), x1 = std::min(x + D, nx - 1);
            t.shape = Shape{y1 - t.y0 + 1, x1 - t.x0 + 1, y - t.y0, x - t.x0};
            targets.push_back(t);
        }
    // Grouped by window shape so that a thread reuses its analysis.
    std::stable_sort(targets.begin(), targets.end(),
                     [](const Target& a, const Target& b) { return a.shape < b.shape; });
    out.n_targets = static_cast<i64>(targets.size());
    if (targets.empty()) return out;

    std::vector<double> corr;
    const i64 W = 2 * D + 1;
    if (correct) corr = omniscape_correction(p, nx, ny);

    const int threads = thread_count(max_threads);
    const std::size_t window = static_cast<std::size_t>(W * W);
    // Windows solved together, then added to the maps in target order.
    const std::size_t per_target = window * (fp ? 2 : 1) * sizeof(double);
    std::size_t batch = static_cast<std::size_t>(threads) * 8;
    const std::size_t budget = std::size_t{512} << 20;
    batch = std::max<std::size_t>(static_cast<std::size_t>(threads), std::min(batch, budget / per_target));
    batch = std::min(batch, targets.size());
    std::vector<std::vector<double>> cur(batch, std::vector<double>(window)), fpcur;
    if (fp) fpcur.assign(batch, std::vector<double>(window));
    std::vector<ShapeCache> caches(static_cast<std::size_t>(threads));
    std::vector<Workspace> spaces(static_cast<std::size_t>(threads));
    std::vector<std::vector<double>> gbuf(static_cast<std::size_t>(threads), std::vector<double>(window)),
        sbuf(static_cast<std::size_t>(threads), std::vector<double>(window)),
        ones(static_cast<std::size_t>(threads), std::vector<double>(window, 1.0));
    const std::size_t n_targets = targets.size();
    bool failed = false;
    std::string error;

    for (std::size_t start = 0; start < n_targets; start += batch) {
        const std::size_t count = std::min(batch, n_targets - start);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
        for (long long k = 0; k < static_cast<long long>(count); ++k) {
            try {
                const Target& t = targets[start + static_cast<std::size_t>(k)];
                const int tid = thread_id();
                const Shape& s = t.shape;
                double* g = gbuf[tid].data();
                double* src = sbuf[tid].data();
                // Conductance within radius + buffer (clip), NoData outside.
                for (i64 y = 0; y < s.h; ++y)
                    for (i64 x = 0; x < s.w; ++x) {
                        const i64 dy = y - s.gy, dx = x - s.gx;
                        const double r = resistance[(t.y0 + y) * nx + (t.x0 + x)];
                        double c = -9999.0;
                        if (dy * dy + dx * dx <= D * D && !std::isnan(r)) c = p.resistance_is_conductance ? r : 1 / r;
                        g[y * s.w + x] = c;
                    }
                // Sources within radius; the target block zeroed at radius +
                // buffer from the corner (get_source); amps shared out.
                for (i64 y = 0; y < s.h; ++y)
                    for (i64 x = 0; x < s.w; ++x) {
                        const i64 dy = y - s.gy, dx = x - s.gx;
                        double v = 0.0;
                        if (dy * dy + dx * dx <= R * R) {
                            v = source[(t.y0 + y) * nx + (t.x0 + x)];
                            if (std::isnan(v)) v = 0.0;
                        }
                        src[y * s.w + x] = v;
                    }
                for (i64 y = D - br; y <= std::min(D + br, ny - 1); ++y)
                    for (i64 x = D - br; x <= std::min(D + br, nx - 1); ++x)
                        if (y >= 0 && y < s.h && x >= 0 && x < s.w) src[y * s.w + x] = 0.0;
                double sum = 0.0;
                for (i64 x = 0; x < s.w; ++x)
                    for (i64 y = 0; y < s.h; ++y)
                        if (src[y * s.w + x] > 0) sum += src[y * s.w + x];
                for (i64 q = 0; q < s.h * s.w; ++q)
                    if (src[q] > 0) src[q] = (src[q] * t.amps) / sum;

                ShapeCache& cache = caches[tid];
                ShapeSystems& sys = fp ? cache.get_rect(s, D, p.four_neighbors) : cache.get(s, D, p.four_neighbors);
                bool factored = false;
                double* c = cur[static_cast<std::size_t>(k)].data();
                solve_window(sys.disk, factored, g, src, s, p.four_neighbors, c, spaces[tid]);
                double* f = fp ? fpcur[static_cast<std::size_t>(k)].data() : nullptr;
                if (fp)
                    solve_window(sys.rect, sys.rect_factored, ones[tid].data(), src, s, p.four_neighbors, f,
                                 spaces[tid]);
                if (correct) {
                    // The part of the correction under the (clipped) window:
                    // Omniscape keeps its first columns when the window is cut
                    // on the right (on both sides too), its last when on the left.
                    const i64 cx0 = t.x < D && t.x + D <= nx - 1 ? W - s.w : 0;
                    const i64 cy0 = t.y < D && t.y + D <= ny - 1 ? W - s.h : 0;
                    for (i64 y = 0; y < s.h; ++y)
                        for (i64 x = 0; x < s.w; ++x) {
                            const double k2 = corr[(cy0 + y) * W + (cx0 + x)];
                            c[y * s.w + x] *= k2;
                            if (fp) f[y * s.w + x] *= k2;
                        }
                }
            } catch (const std::exception& e) {
#pragma omp critical(omniscape_error)
                {
                    failed = true;
                    error = e.what();
                }
            }
        }
        if (failed) throw std::runtime_error(error);
        // Add the windows row by row, each row in target order.
        i64 row0 = ny, row1 = -1;
        for (std::size_t k = 0; k < count; ++k) {
            const Target& t = targets[start + k];
            row0 = std::min(row0, t.y0);
            row1 = std::max(row1, t.y0 + t.shape.h - 1);
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (long long row = row0; row <= row1; ++row) {
            for (std::size_t k = 0; k < count; ++k) {
                const Target& t = targets[start + k];
                const i64 y = row - t.y0;
                if (y < 0 || y >= t.shape.h) continue;
                const double* c = cur[k].data() + y * t.shape.w;
                double* dst = out.current.data() + row * nx + t.x0;
                for (i64 x = 0; x < t.shape.w; ++x) dst[x] += c[x];
                if (fp) {
                    const double* f = fpcur[k].data() + y * t.shape.w;
                    double* fdst = out.flow_potential.data() + row * nx + t.x0;
                    for (i64 x = 0; x < t.shape.w; ++x) fdst[x] += f[x];
                }
            }
        }
    }
    return out;
}

}  // namespace circuit
