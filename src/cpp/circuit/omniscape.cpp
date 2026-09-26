// SPDX-License-Identifier: GPL-3.0-or-later

#include "circuit/omniscape.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
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

// The unknowns of a window: the cells within `radius` of the ground (every
// cell when radius < 0), without the ground.
std::vector<std::uint8_t> window_mark(const Shape& s, i64 radius) {
    std::vector<std::uint8_t> mark(static_cast<std::size_t>(s.h * s.w), 0);
    for (i64 y = 0; y < s.h; ++y)
        for (i64 x = 0; x < s.w; ++x) {
            const i64 dy = y - s.gy, dx = x - s.gx;
            mark[y * s.w + x] = radius < 0 || dy * dy + dx * dx <= radius * radius;
        }
    mark[s.gy * s.w + s.gx] = 0;
    return mark;
}

std::shared_ptr<const FrontalSymbolic> window_symbolic(const Shape& s, i64 radius, bool four) {
    const std::vector<std::uint8_t> mark = window_mark(s, radius);
    return frontal_analyze(s.w, s.h, mark.data(), four);
}

struct Workspace {
    std::vector<std::uint8_t> present;
    std::vector<i64> stack;
    std::vector<double> v, rhs, x, edges, ones, multi;
    std::vector<std::uint8_t> all;
    std::vector<int> which;
    std::vector<double> cond;
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
    ws.rhs.resize(m);
    ws.x.resize(m);
    for (std::size_t k = 0; k < m; ++k) ws.rhs[k] = present[cells[k]] ? src[cells[k]] : 0.0;
    if (!factored) {
        sys.assemble(g, present, nullptr, kShift);
        if (!sys.factorize_solve(ws.rhs.data(), ws.x.data()))
            throw std::runtime_error("an Omniscape window is not positive definite");
        factored = true;
    } else {
        sys.solve(ws.rhs.data(), ws.x.data());
    }
    double* v = ws.v.data();
    std::fill(v, v + n, 0.0);
    for (std::size_t k = 0; k < m; ++k)
        if (present[cells[k]]) v[cells[k]] = ws.x[k];

    node_currents(s.w, s.h, g, present, v, nullptr, four, cur, ws.edges);
}

// The flow potential of k windows of one shape: conductance 1 on every cell,
// so the whole window is one component and its factorization is computed once
// (`factored`); the k right-hand sides are solved together.
void flow_batch(CellSystem& sys, bool& factored, const Shape& s, bool four, double* const* src, double* const* cur,
                int k, Workspace& ws) {
    const i64 n = s.h * s.w;
    const i64 ground = s.gy * s.w + s.gx;
    if (ws.ones.size() < static_cast<std::size_t>(n)) {
        ws.ones.assign(static_cast<std::size_t>(n), 1.0);
        ws.all.assign(static_cast<std::size_t>(n), std::uint8_t{1});
    }
    if (!factored) {
        sys.assemble(ws.ones.data(), ws.all.data(), nullptr, kShift);
        if (!sys.factorize()) throw std::runtime_error("an Omniscape window is not positive definite");
        factored = true;
    }
    const std::vector<i64>& cells = sys.cells();
    const std::size_t m = cells.size();
    ws.multi.resize(m * static_cast<std::size_t>(k));
    int cols = 0;
    std::vector<int>& which = ws.which;
    which.resize(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j) {
        std::fill(cur[j], cur[j] + n, 0.0);
        double total = 0.0;
        for (std::size_t i = 0; i < m; ++i) total += src[j][cells[i]];
        if (total == 0) continue;  // no source (the ground's own is dropped)
        double* col = ws.multi.data() + static_cast<std::size_t>(cols) * m;
        for (std::size_t i = 0; i < m; ++i) col[i] = src[j][cells[i]];
        which[static_cast<std::size_t>(cols++)] = j;
    }
    if (cols == 0) return;
    sys.solve_many(ws.multi.data(), cols);
    ws.v.resize(static_cast<std::size_t>(n));
    double* v = ws.v.data();
    for (int c = 0; c < cols; ++c) {
        const double* col = ws.multi.data() + static_cast<std::size_t>(c) * m;
        v[ground] = 0.0;
        for (std::size_t i = 0; i < m; ++i) v[cells[i]] = col[i];
        node_currents(s.w, s.h, ws.ones.data(), ws.all.data(), v, nullptr, four, cur[which[static_cast<std::size_t>(c)]],
                      ws.edges);
    }
}

// The target block's value of a condition: the median ("within") or the mode
// ("equal") of its future values in column-major order, NaN left out.
double block_condition(const OmniscapeCondition& c, i64 nx, i64 ny, i64 x, i64 y, i64 br, std::vector<double>& buf) {
    const double* f = c.future ? c.future : c.present;
    buf.clear();
    for (i64 xx = x - br; xx <= std::min(x + br, nx - 1); ++xx)
        for (i64 yy = y - br; yy <= std::min(y + br, ny - 1); ++yy) {
            const double v = f[yy * nx + xx];
            if (!std::isnan(v)) buf.push_back(v);
        }
    if (buf.empty()) throw std::runtime_error("a target block has no value of a condition (all NoData)");
    const std::size_t n = buf.size();
    if (!c.equal) {
        // Statistics.median: middle(a, b) = a/2 + b/2 for an even count.
        if (n % 2 == 1) {
            std::nth_element(buf.begin(), buf.begin() + n / 2, buf.end());
            return buf[n / 2];
        }
        std::nth_element(buf.begin(), buf.begin() + n / 2, buf.end());
        const double b = buf[n / 2];
        const double a = *std::max_element(buf.begin(), buf.begin() + n / 2);
        return a / 2 + b / 2;
    }
    // StatsBase.mode: counts by value (isequal: 0.0 and -0.0 apart), the
    // first value to exceed the running highest count wins.
    std::vector<std::pair<std::uint64_t, int>> counts;
    double best = buf[0];
    int top = 0;
    for (double v : buf) {
        std::uint64_t bits;
        std::memcpy(&bits, &v, sizeof bits);
        int* cnt = nullptr;
        for (auto& e : counts)
            if (e.first == bits) {
                cnt = &e.second;
                break;
            }
        if (!cnt) {
            counts.emplace_back(bits, 0);
            cnt = &counts.back().second;
        }
        if (++*cnt > top) {
            top = *cnt;
            best = v;
        }
    }
    return best;
}

// Sources of a window whose present condition does not match `value` set to 0.
void match_condition(const OmniscapeCondition& c, double value, i64 nx, i64 x0, i64 y0, i64 h, i64 w, double* src) {
    for (i64 y = 0; y < h; ++y)
        for (i64 x = 0; x < w; ++x) {
            const double v = c.present[(y0 + y) * nx + (x0 + x)];
            if (std::isnan(v)) continue;
            if (c.equal) {
                if (v != value) src[y * w + x] = 0.0;
            } else {
                const double d = v - value;
                if (d > c.upper || d < c.lower) src[y * w + x] = 0.0;
            }
        }
}

struct Target {
    i64 x, y;
    double amps;
    Shape shape;
    i64 y0, x0;  // window corner on the map
};

// The symbolic structures of the window shapes, shared by the threads. Targets
// come grouped by shape, so a few recent shapes are enough.
class SymbolicCache {
public:
    SymbolicCache(i64 radius, bool four, std::size_t capacity) : radius_(radius), four_(four), capacity_(capacity) {}

    // kind 0: the landscape (cells within radius + buffer); 1: the whole
    // window (the flow potential).
    std::shared_ptr<const FrontalSymbolic> get(int kind, const Shape& s) {
        const Key key{kind, s};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& e : entries_)
                if (e.first == key) return e.second;
        }
        auto sym = window_symbolic(s, kind == 0 ? radius_ : -1, four_);
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& e : entries_)
            if (e.first == key) return e.second;
        if (entries_.size() >= capacity_) entries_.pop_front();
        entries_.emplace_back(key, sym);
        return sym;
    }

private:
    struct Key {
        int kind;
        Shape shape;
        bool operator==(const Key& o) const { return kind == o.kind && shape == o.shape; }
    };
    i64 radius_;
    bool four_;
    std::size_t capacity_;
    std::mutex mutex_;
    std::deque<std::pair<Key, std::shared_ptr<const FrontalSymbolic>>> entries_;
};

// A thread's systems: the landscape system, refactorised for every window,
// and the flow-potential systems of its last shapes, factorised once each.
struct ThreadSystems {
    CellSystem disk;
    Shape disk_shape{-1, -1, -1, -1};
    struct Rect {
        Shape shape{-1, -1, -1, -1};
        CellSystem sys;
        bool factored = false;
    };
    std::array<std::unique_ptr<Rect>, 2> rects;

    CellSystem& landscape(SymbolicCache& cache, const Shape& s) {
        if (!(disk_shape == s)) {
            disk.share(cache.get(0, s));
            disk_shape = s;
        }
        return disk;
    }
    Rect& flow(SymbolicCache& cache, const Shape& s) {
        for (auto& r : rects)
            if (r && r->shape == s) return *r;
        std::swap(rects[0], rects[1]);
        if (!rects[0]) rects[0] = std::make_unique<Rect>();
        rects[0]->shape = s;
        rects[0]->sys.share(cache.get(1, s));
        rects[0]->factored = false;
        return *rects[0];
    }
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
    sys.share(window_symbolic(s, D, p.four_neighbors));
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

    int threads = thread_count(max_threads);
    const std::size_t window = static_cast<std::size_t>(W * W);
    SymbolicCache symbolic(D, p.four_neighbors, static_cast<std::size_t>(4 * threads + 4));
    // Memory of one thread, from the full window (every other shape is smaller):
    // the factor with its margin and update stack, the flow potential's two
    // cached factors, and the thread's window buffers.
    {
        const Shape full{W, W, D, D};
        auto bytes = [](const FrontalSymbolic& f) {
            const double n = static_cast<double>(f.cells.size());
            return 8.0 * (static_cast<double>(f.lsize + f.max_front + f.stack_size) + 10 * n);
        };
        double per_thread = bytes(*symbolic.get(0, full)) + 8.0 * static_cast<double>(window) * 16;
        if (fp) per_thread += 2 * bytes(*symbolic.get(1, full));
        if (p.memory_limit > 0) {
            const double room = p.memory_limit / 2;  // the other half: batch buffers and the maps
            const int fit = static_cast<int>(std::max(1.0, std::floor(room / per_thread)));
            threads = std::min(threads, fit);
        }
    }
    out.threads = threads;
    // Windows solved together, then added to the maps in target order. Each
    // thread takes chunks of consecutive targets (mostly of one shape), whose
    // flow potentials are solved together; about twelve chunks per thread and
    // batch, within a memory budget.
    const std::size_t per_target = window * (fp ? 2 : 1) * sizeof(double);
    std::size_t budget = std::size_t{768} << 20;
    if (p.memory_limit > 0) budget = std::min(budget, static_cast<std::size_t>(p.memory_limit / 4));
    const std::size_t nt = static_cast<std::size_t>(threads);
    // Chunks are 8 consecutive targets of the global order, whatever the
    // threads: the flow potentials solved together (and so the rounding of
    // every one) do not depend on the threads or the memory.
    constexpr std::size_t chunk = 8;
    const std::size_t fit = budget / per_target;
    std::size_t batch = std::max<std::size_t>(chunk, std::min(fit, 12 * nt * chunk) / chunk * chunk);
    batch = std::min(batch, targets.size());
    std::vector<std::vector<double>> cur(batch, std::vector<double>(window)), fpcur;
    if (fp) fpcur.assign(batch, std::vector<double>(window));
    std::vector<ThreadSystems> systems(static_cast<std::size_t>(threads));
    std::vector<Workspace> spaces(static_cast<std::size_t>(threads));
    std::vector<std::vector<double>> gbuf(static_cast<std::size_t>(threads), std::vector<double>(window)),
        sbuf(static_cast<std::size_t>(threads), std::vector<double>(window * chunk));
    const std::size_t n_targets = targets.size();
    bool failed = false;
    std::string error;

    for (std::size_t start = 0; start < n_targets; start += batch) {
        const std::size_t count = std::min(batch, n_targets - start);
        const long long n_chunks = static_cast<long long>((count + chunk - 1) / chunk);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
        for (long long ch = 0; ch < n_chunks; ++ch) {
            try {
                const int tid = thread_id();
                ThreadSystems& sys = systems[tid];
                const std::size_t k0 = static_cast<std::size_t>(ch) * chunk;
                const std::size_t k1 = std::min(count, k0 + chunk);
                for (std::size_t k = k0; k < k1; ++k) {
                    const Target& t = targets[start + k];
                    const Shape& s = t.shape;
                    double* g = gbuf[tid].data();
                    double* src = sbuf[tid].data() + (k - k0) * window;
                    // Conductance within radius + buffer (clip), NoData outside.
                    for (i64 y = 0; y < s.h; ++y)
                        for (i64 x = 0; x < s.w; ++x) {
                            const i64 dy = y - s.gy, dx = x - s.gx;
                            const double r = resistance[(t.y0 + y) * nx + (t.x0 + x)];
                            double c = -9999.0;
                            if (dy * dy + dx * dx <= D * D && !std::isnan(r))
                                c = p.resistance_is_conductance ? r : 1 / r;
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
                    for (const OmniscapeCondition* cond : {&p.condition1, &p.condition2}) {
                        if (!cond->present) continue;
                        const double value = block_condition(*cond, nx, ny, t.x, t.y, br, spaces[tid].cond);
                        match_condition(*cond, value, nx, t.x0, t.y0, s.h, s.w, src);
                    }
                    bool factored = false;
                    solve_window(sys.landscape(symbolic, s), factored, g, src, s, p.four_neighbors, cur[k].data(),
                                 spaces[tid]);
                }
                if (fp) {
                    // Runs of one shape.
                    for (std::size_t a = k0; a < k1;) {
                        std::size_t b = a + 1;
                        while (b < k1 && targets[start + b].shape == targets[start + a].shape) ++b;
                        const Shape& s = targets[start + a].shape;
                        double* srcs[8];
                        double* curs[8];
                        for (std::size_t k = a; k < b; ++k) {
                            srcs[k - a] = sbuf[tid].data() + (k - k0) * window;
                            curs[k - a] = fpcur[k].data();
                        }
                        ThreadSystems::Rect& rect = sys.flow(symbolic, s);
                        flow_batch(rect.sys, rect.factored, s, p.four_neighbors, srcs, curs, static_cast<int>(b - a),
                                   spaces[tid]);
                        a = b;
                    }
                }
                if (correct) {
                    for (std::size_t k = k0; k < k1; ++k) {
                        const Target& t = targets[start + k];
                        const Shape& s = t.shape;
                        double* c = cur[k].data();
                        double* f = fp ? fpcur[k].data() : nullptr;
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
