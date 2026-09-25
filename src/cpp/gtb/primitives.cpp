// SPDX-License-Identifier: GPL-3.0-or-later
//
// Ports of the miallib routines used by MSPA. See primitives.hpp.
// Original algorithms: Copyright (C) 2000-2022 European Union (JRC),
// authors Pierre Soille et al., released under GPL-3.0-or-later.

#include "primitives.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gtb {

namespace {

void check_same_shape(const ImageU8& a, const ImageU8& b) {
    if (a.nx != b.nx || a.ny != b.ny) throw std::invalid_argument("images have different shapes");
}

}  // namespace

// ---------------------------------------------------------------------------
// Pointwise operations
// ---------------------------------------------------------------------------

ImageU8 negation(const ImageU8& im) {
    ImageU8 out(im.nx, im.ny);
    const u8* a = im.data();
    u8* o = out.data();
    parallel_chunks(im.size(), [=](i64 b, i64 e) {
        for (i64 i = b; i < e; ++i) o[i] = a[i] ? 0 : 1;
    });
    return out;
}

// Kernels with restrict-qualified pointers so that the compiler vectorises.
template <class Op>
static void binary_kernel(u8* __restrict p, const u8* __restrict q, i64 n, Op op) {
    for (i64 i = 0; i < n; ++i) p[i] = op(p[i], q[i]);
}

template <class Op>
static void unary_kernel(u8* __restrict p, i64 n, Op op) {
    for (i64 i = 0; i < n; ++i) p[i] = op(p[i]);
}

// Applies p[i] = op(p[i], q[i]) over two images of equal shape.
template <class Op>
static void binary_op(ImageU8& a, const ImageU8& b, Op op) {
    check_same_shape(a, b);
    u8* p = a.data();
    const u8* q = b.data();
    parallel_chunks(a.size(), [=](i64 lo, i64 hi) { binary_kernel(p + lo, q + lo, hi - lo, op); });
}

// Applies p[i] = op(p[i]) over an image.
template <class Op>
static void unary_op(ImageU8& a, Op op) {
    u8* p = a.data();
    parallel_chunks(a.size(), [=](i64 lo, i64 hi) { unary_kernel(p + lo, hi - lo, op); });
}

void bit_or(ImageU8& a, const ImageU8& b) {
    binary_op(a, b, [](u8 x, u8 y) { return static_cast<u8>(x | y); });
}

void bit_and(ImageU8& a, const ImageU8& b) {
    binary_op(a, b, [](u8 x, u8 y) { return static_cast<u8>(x & y); });
}

void sub_sat(ImageU8& a, const ImageU8& b) {
    binary_op(a, b, [](u8 x, u8 y) { return static_cast<u8>(x > y ? x - y : 0); });
}

void add_sat(ImageU8& a, const ImageU8& b) {
    binary_op(a, b, [](u8 x, u8 y) {
        const int s = int(x) + int(y);
        return static_cast<u8>(s > 255 ? 255 : s);
    });
}

void inf(ImageU8& a, const ImageU8& b) {
    binary_op(a, b, [](u8 x, u8 y) { return std::min(x, y); });
}

void sup(ImageU8& a, const ImageU8& b) {
    binary_op(a, b, [](u8 x, u8 y) { return std::max(x, y); });
}

void shift_left(ImageU8& a, int k) {
    unary_op(a, [k](u8 x) { return static_cast<u8>(x << k); });
}

void set_level(ImageU8& a, u8 t1, u8 t2, u8 value) {
    unary_op(a, [=](u8 x) { return (x >= t1 && x <= t2) ? value : x; });
}

void threshold(ImageU8& a, u8 t1, u8 t2, u8 bg, u8 fg) {
    unary_op(a, [=](u8 x) { return (x >= t1 && x <= t2) ? fg : bg; });
}

bool is_empty(const ImageU8& a) {
    return std::all_of(a.px.begin(), a.px.end(), [](u8 v) { return v == 0; });
}

// ---------------------------------------------------------------------------
// uc_sqedt
// ---------------------------------------------------------------------------

namespace {
using u64 = std::uint64_t;
inline u64 squ(i64 v) { return static_cast<u64>(v) * static_cast<u64>(v); }
inline u64 normu(i64 a, i64 b) { return squ(a) + squ(b); }
}  // namespace

std::vector<std::uint32_t> sqedt(const ImageU8& im) {
    const i64 m = im.nx;
    const i64 n = im.ny;
    const std::uint32_t bigval = std::numeric_limits<std::uint32_t>::max() / 2;
    const u8* b = im.data();
    std::vector<std::uint32_t> g(static_cast<std::size_t>(m * n));
    std::vector<std::uint32_t> dt(static_cast<std::size_t>(m * n));

    // Scans 1 and 2: vertical distances. Columns are independent; they are
    // swept in blocks so that the inner loop runs along contiguous memory.
    constexpr i64 kBlock = 512;
    const i64 nblocks = (m + kBlock - 1) / kBlock;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (m * n >= kParallelThreshold)
#endif
    for (i64 blk = 0; blk < nblocks; ++blk) {
        const i64 x0 = blk * kBlock;
        const i64 x1 = std::min(m, x0 + kBlock);
        for (i64 x = x0; x < x1; ++x) g[x] = b[x] == 0 ? 0 : bigval;
        for (i64 y = 1; y < n; ++y) {
            const i64 row = y * m;
            for (i64 x = x0; x < x1; ++x) {
                const i64 off = row + x;
                g[off] = b[off] == 0 ? 0 : 1 + g[off - m];
            }
        }
        for (i64 y = n - 2; y >= 0; --y) {
            const i64 row = y * m;
            for (i64 x = x0; x < x1; ++x) {
                const i64 off = row + x;
                if (g[off + m] < g[off]) g[off] = 1 + g[off + m];
            }
        }
    }

    // Scans 3 and 4: lower envelope of parabolas along every row.
#if defined(_OPENMP)
#pragma omp parallel if (m * n >= kParallelThreshold)
#endif
    {
        std::vector<i64> s(static_cast<std::size_t>(m));
        std::vector<i64> t(static_cast<std::size_t>(m));
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (i64 y = 0; y < n; ++y) {
            const std::uint32_t* gy = g.data() + y * m;
            std::uint32_t* dty = dt.data() + y * m;
            i64 q = 0;
            s[0] = 0;
            t[0] = 0;
            for (i64 x = 1; x < m; ++x) {
                while (q >= 0 && normu(t[q] - s[q], gy[s[q]]) > normu(t[q] - x, gy[x])) --q;
                if (q < 0) {
                    q = 0;
                    s[0] = x;
                } else {
                    // Unsigned arithmetic exactly as in efedt.c.
                    const u64 num = squ(x) - squ(s[q]) + squ(gy[x]) - squ(gy[s[q]]);
                    const u64 den = static_cast<u64>(2 * (x - s[q]));
                    const i64 sep = static_cast<i64>(num / den);
                    const i64 w = 1 + sep;
                    if (w < m) {
                        ++q;
                        s[q] = x;
                        t[q] = w;
                    }
                }
            }
            for (i64 x = m - 1; x >= 0; --x) {
                dty[x] = static_cast<std::uint32_t>(normu(x - s[q], gy[s[q]]));
                if (x == t[q]) --q;
            }
        }
    }
    return dt;
}

// ---------------------------------------------------------------------------
// Thresholded distances
// ---------------------------------------------------------------------------

std::int64_t squared_threshold(float s) {
    if (!(s >= 0.0f)) return -1;
    if (s > 4000.0f) throw std::invalid_argument("squared_threshold: distance too large");
    // edt_value is exact-monotone below 2^24; start near s^2 and adjust.
    std::int64_t t = static_cast<std::int64_t>(static_cast<double>(s) * static_cast<double>(s));
    while (t > 0 && edt_value(static_cast<std::uint32_t>(t)) > s) --t;
    while (edt_value(static_cast<std::uint32_t>(t + 1)) <= s) ++t;
    return t;
}

namespace {
std::int64_t isqrt(std::int64_t v) {
    auto r = static_cast<std::int64_t>(std::sqrt(static_cast<double>(v)));
    while (r * r > v) --r;
    while ((r + 1) * (r + 1) <= v) ++r;
    return r;
}
}  // namespace

ImageU8 dilate_disk(const ImageU8& src, std::int64_t T) {
    const i64 nx = src.nx;
    const i64 ny = src.ny;
    ImageU8 out(nx, ny, 0);
    if (T < 0) return out;
    const std::int64_t R = isqrt(T);
    if (R >= 65535) throw std::invalid_argument("dilate_disk: radius too large");
    const int far = static_cast<int>(R + 1);

    // Vertical distance to the nearest source in the same column, clipped at
    // R + 1 (anything farther cannot satisfy d^2 <= T).
    std::vector<std::uint16_t> v(static_cast<std::size_t>(nx * ny));
    const u8* s = src.data();
    constexpr i64 kBlock = 1024;
    const i64 nblocks = (nx + kBlock - 1) / kBlock;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (nx * ny >= kParallelThreshold && nblocks > 1)
#endif
    for (i64 blk = 0; blk < nblocks; ++blk) {
        const i64 x0 = blk * kBlock;
        const i64 x1 = std::min(nx, x0 + kBlock);
        for (i64 x = x0; x < x1; ++x) v[x] = s[x] ? 0 : static_cast<std::uint16_t>(far);
        for (i64 y = 1; y < ny; ++y) {
            std::uint16_t* vy = v.data() + y * nx;
            const std::uint16_t* va = vy - nx;
            const u8* sy = s + y * nx;
            for (i64 x = x0; x < x1; ++x) {
                const int up = std::min(int(va[x]) + 1, far);
                vy[x] = sy[x] ? 0 : static_cast<std::uint16_t>(up);
            }
        }
        for (i64 y = ny - 2; y >= 0; --y) {
            std::uint16_t* vy = v.data() + y * nx;
            const std::uint16_t* vb = vy + nx;
            for (i64 x = x0; x < x1; ++x) {
                const int down = int(vb[x]) + 1;
                if (down < vy[x]) vy[x] = static_cast<std::uint16_t>(down);
            }
        }
    }

    // Half-width of the disk at vertical offset g.
    std::vector<std::int64_t> W(static_cast<std::size_t>(R + 1));
    for (std::int64_t g = 0; g <= R; ++g) W[g] = isqrt(T - g * g);

    // Pixel x is covered iff some column s has |x - s| <= W[v_s].
    u8* o = out.data();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (nx * ny >= kParallelThreshold)
#endif
    for (i64 y = 0; y < ny; ++y) {
        const std::uint16_t* vy = v.data() + y * nx;
        u8* oy = o + y * nx;
        i64 reach = -1;
        for (i64 x = 0; x < nx; ++x) {
            if (vy[x] <= R) reach = std::max(reach, x + W[vy[x]]);
            oy[x] = reach >= x ? 1 : 0;
        }
        i64 lo = nx;
        for (i64 x = nx - 1; x >= 0; --x) {
            if (vy[x] <= R) lo = std::min(lo, x - W[vy[x]]);
            if (lo <= x) oy[x] = 1;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// ced
// ---------------------------------------------------------------------------

namespace {

inline long sgn(long v) { return v < 0 ? -1 : (v > 0 ? 1 : 0); }

// inmaskp (ced.c): Bresenham walk from the pixel at `base` along (x2, y2);
// false as soon as a pixel of `mask` along the way is zero.
bool inmaskp(const u8* mask, i64 ncol, i64 base, long x2, long y2) {
    const u8* p = mask + base;
    long dx = std::labs(x2);
    long dy = std::labs(y2);
    const long s1 = sgn(x2);
    const long s2 = sgn(y2);
    bool interchange = true;
    if (dy > dx) {
        std::swap(dx, dy);
    } else {
        interchange = false;
    }
    long d = 2 * dy - dx;
    const long incr1 = 2 * dy;
    const long incr2 = 2 * (dy - dx);
    long x = 0;
    long y = 0;
    const long der = dx;
    while (std::labs(interchange ? y : x) < der) {
        if (interchange)
            y += s2;
        else
            x += s1;
        if (d < 0) {
            d += incr1;
        } else {
            d += incr2;
            if (interchange)
                x += s1;
            else
                y += s2;
        }
        if (p[x + y * ncol] == 0) return false;
    }
    return true;
}

}  // namespace

std::vector<float> ced(const ImageU8& ref_in, const ImageU8& mask_in) {
    check_same_shape(ref_in, mask_in);
    constexpr std::int16_t kShortMax = 0x7FFF;
    constexpr u8 kInQueue = 2;

    const ImageU8 ref = add_frame<u8>(ref_in, 1, 0);
    ImageU8 pm = add_frame<u8>(mask_in, 1, 0);  // working copy, marks queued pixels
    const i64 nx = ref.nx;
    const i64 npix = ref.size();
    const i64 shft[4] = {-nx, -1, nx, 1};  // set_seq_shift(graph = 4)
    const int dx[4] = {0, 1, 0, -1};
    const int dy[4] = {1, 0, -1, 0};

    std::vector<i64> cur;
    cur.reserve(static_cast<std::size_t>(npix / 100 + 1024));
    for (i64 i = 0; i < npix; ++i) {
        if (ref[i] && pm[i]) {
            pm[i] = 3;
            for (int k = 0; k < 4; ++k) {
                const i64 o = i + shft[k];
                if (pm[o] == 1 && ref[o] == 0) {
                    cur.push_back(o);
                    pm[o] = kInQueue;
                }
            }
        }
    }

    // (px, py): vector from the current base pixel; pb: geodesic distance of
    // that base pixel.
    std::vector<std::int16_t> px(static_cast<std::size_t>(npix));
    std::vector<std::int16_t> py(static_cast<std::size_t>(npix));
    std::vector<float> pb(static_cast<std::size_t>(npix), 0.0f);
    for (i64 i = 0; i < npix; ++i) {
        const std::int16_t v = pm[i] <= 2 ? kShortMax : 0;
        px[i] = v;
        py[i] = v;
    }

    auto sq = [](i64 v) { return v * v; };
    const u8* mask = pm.data();
    std::vector<i64> next;
    next.reserve(cur.capacity());
    while (!cur.empty()) {
        // Look pass: compute a tentative distance for every queued pixel.
        float dmin = FLT_MAX;
        for (const i64 ofs : cur) {
            if (px[ofs] != kShortMax) continue;
            float dp = FLT_MAX;
            for (int k = 0; k < 4; ++k) {
                const i64 o = ofs + shft[k];
                if (px[o] == kShortMax) continue;
                const i64 base = o - px[o] - i64(py[o]) * nx;
                bool base_flag;
                float dcrt;
                if (inmaskp(mask, nx, base, px[o] + dx[k], py[o] + dy[k])) {
                    base_flag = false;
                    dcrt = static_cast<float>(
                        std::sqrt(static_cast<double>(sq(i64(px[o]) + dx[k]) + sq(i64(py[o]) + dy[k]))) +
                        static_cast<double>(pb[o]));
                } else {
                    base_flag = true;
                    dcrt = static_cast<float>(std::sqrt(static_cast<double>(sq(px[o]) + sq(py[o]))) +
                                              static_cast<double>(pb[o]) + 1.0);
                }
                if (dcrt < dp) {
                    if (base_flag) {
                        px[ofs] = static_cast<std::int16_t>(dx[k]);
                        py[ofs] = static_cast<std::int16_t>(dy[k]);
                        pb[ofs] = dcrt - 1.0f;
                    } else {
                        px[ofs] = static_cast<std::int16_t>(px[o] + dx[k]);
                        py[ofs] = static_cast<std::int16_t>(py[o] + dy[k]);
                        pb[ofs] = pb[o];
                    }
                    dp = dcrt;
                }
            }
            if (dmin > dp) dmin = dp;
        }
        // Remove pass: pixels at the current minimum propagate, others wait.
        next.clear();
        for (const i64 ofs : cur) {
            const double d = std::sqrt(static_cast<double>(sq(px[ofs]) + sq(py[ofs]))) + static_cast<double>(pb[ofs]);
            if (d > static_cast<double>(dmin)) {
                next.push_back(ofs);
            } else {
                for (int k = 0; k < 4; ++k) {
                    const i64 o = ofs + shft[k];
                    if (pm[o] == 1) {
                        next.push_back(o);
                        pm[o] = kInQueue;
                    }
                }
            }
        }
        cur.swap(next);
    }

    Image<float> out(nx, ref.ny);
    for (i64 i = 0; i < npix; ++i) {
        float v = pb[i];
        if (px[i] != kShortMax) v += static_cast<float>(std::sqrt(static_cast<double>(sq(px[i]) + sq(py[i]))));
        out[i] = v;
    }
    return sub_frame(out, 1).px;
}

// ---------------------------------------------------------------------------
// Binary reconstructions
// ---------------------------------------------------------------------------

namespace {

// Connected components of a binary predicate, computed on horizontal runs
// with a union-find: much faster than pixel flooding and exact.
class RunComponents {
public:
    struct Run {
        std::int32_t y, xs, xe;  // pixels [xs, xe) of row y
    };

    // Runs of pixels where pred(offset) holds.
    template <class Pred>
    RunComponents(i64 nx, i64 ny, int graph, Pred pred) : ny_(ny) {
        row_begin_.resize(static_cast<std::size_t>(ny + 1));
        for (i64 y = 0; y < ny; ++y) {
            row_begin_[y] = static_cast<i64>(runs_.size());
            i64 x = 0;
            while (x < nx) {
                while (x < nx && !pred(y * nx + x)) ++x;
                if (x == nx) break;
                const i64 xs = x;
                while (x < nx && pred(y * nx + x)) ++x;
                add_run(y, xs, x);
            }
        }
        link(graph);
    }

    // Runs of pixels of a byte image that are non-zero (`nonzero`) or zero,
    // scanning 8 bytes at a time over uniform stretches.
    RunComponents(const u8* img, i64 nx, i64 ny, int graph, bool nonzero) : ny_(ny) {
        row_begin_.resize(static_cast<std::size_t>(ny + 1));
        for (i64 y = 0; y < ny; ++y) {
            row_begin_[y] = static_cast<i64>(runs_.size());
            const u8* row = img + y * nx;
            i64 x = 0;
            while (x < nx) {
                x = nonzero ? next_nonzero(row, x, nx) : next_zero(row, x, nx);
                if (x == nx) break;
                const i64 xs = x;
                x = nonzero ? next_zero(row, x, nx) : next_nonzero(row, x, nx);
                add_run(y, xs, x);
            }
        }
        link(graph);
    }

    const std::vector<Run>& runs() const { return runs_; }
    // Runs of row y are [row_begin(y), row_begin(y + 1)).
    i64 row_begin(i64 y) const { return row_begin_[y]; }
    // Root (component id) of run i; ids are run indices.
    i64 root(i64 i) const { return parent_[i]; }

private:
    static std::uint64_t load8(const u8* p) {
        std::uint64_t w;
        std::memcpy(&w, p, 8);
        return w;
    }
    // Word has at least one zero byte (classic bit trick).
    static bool has_zero_byte(std::uint64_t w) {
        return ((w - 0x0101010101010101ULL) & ~w & 0x8080808080808080ULL) != 0;
    }
    static i64 next_nonzero(const u8* row, i64 x, i64 nx) {
        while (x + 8 <= nx && load8(row + x) == 0) x += 8;
        while (x < nx && row[x] == 0) ++x;
        return x;
    }
    static i64 next_zero(const u8* row, i64 x, i64 nx) {
        while (x + 8 <= nx && !has_zero_byte(load8(row + x))) x += 8;
        while (x < nx && row[x] != 0) ++x;
        return x;
    }

    void add_run(i64 y, i64 xs, i64 xe) {
        runs_.push_back({static_cast<std::int32_t>(y), static_cast<std::int32_t>(xs), static_cast<std::int32_t>(xe)});
    }

    void link(int graph) {
        if (graph != 4 && graph != 8) throw std::invalid_argument("connectivity must be 4 or 8");
        row_begin_[ny_] = static_cast<i64>(runs_.size());
        const i64 ny = ny_;

        parent_.resize(runs_.size());
        for (std::size_t i = 0; i < parent_.size(); ++i) parent_[i] = static_cast<i64>(i);
        const i64 tol = graph == 8 ? 1 : 0;
        for (i64 y = 1; y < ny; ++y) {
            i64 i = row_begin_[y - 1];
            i64 j = row_begin_[y];
            const i64 iend = row_begin_[y];
            const i64 jend = row_begin_[y + 1];
            while (i < iend && j < jend) {
                const Run& a = runs_[i];
                const Run& b = runs_[j];
                if (a.xs <= b.xe - 1 + tol && b.xs <= a.xe - 1 + tol) unite(i, j);
                if (a.xe < b.xe)
                    ++i;
                else
                    ++j;
            }
        }
        for (std::size_t i = 0; i < parent_.size(); ++i) parent_[i] = find(static_cast<i64>(i));
    }

    i64 find(i64 i) {
        while (parent_[i] != i) {
            parent_[i] = parent_[parent_[i]];
            i = parent_[i];
        }
        return i;
    }
    void unite(i64 a, i64 b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (a < b)
            parent_[b] = a;  // keep the earliest run as root
        else
            parent_[a] = b;
    }

    i64 ny_;
    std::vector<Run> runs_;
    std::vector<i64> row_begin_;
    std::vector<i64> parent_;
};

}  // namespace

ImageU8 fill_holes(const ImageU8& im, int graph) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const u8* p = im.data();
    const RunComponents cc(p, nx, ny, graph, /*nonzero=*/false);
    const auto& runs = cc.runs();
    std::vector<u8> touches_border(runs.size(), 0);
    for (std::size_t r = 0; r < runs.size(); ++r) {
        const auto& run = runs[r];
        if (run.y == 0 || run.y == ny - 1 || run.xs == 0 || run.xe == nx) touches_border[cc.root(r)] = 1;
    }
    ImageU8 out(nx, ny, 1);
    for (std::size_t r = 0; r < runs.size(); ++r) {
        if (!touches_border[cc.root(r)]) continue;
        const auto& run = runs[r];
        std::fill(out.data() + run.y * nx + run.xs, out.data() + run.y * nx + run.xe, u8(0));
    }
    return out;
}

void reconstruct_dilation(ImageU8& marker, const ImageU8& mask, int graph) {
    check_same_shape(marker, mask);
    const i64 nx = mask.nx;
    const u8* m = mask.data();
    const u8* k = marker.data();
    const RunComponents cc(m, nx, mask.ny, graph, /*nonzero=*/true);
    const auto& runs = cc.runs();
    std::vector<u8> marked(runs.size(), 0);
    for (std::size_t r = 0; r < runs.size(); ++r) {
        const auto& run = runs[r];
        const u8* row = k + run.y * nx;
        for (i64 x = run.xs; x < run.xe; ++x) {
            if (row[x]) {
                marked[cc.root(r)] = 1;
                break;
            }
        }
    }
    ImageU8 out(nx, mask.ny, 0);
    for (std::size_t r = 0; r < runs.size(); ++r) {
        if (!marked[cc.root(r)]) continue;
        const auto& run = runs[r];
        const i64 off = run.y * nx;
        std::copy(m + off + run.xs, m + off + run.xe, out.data() + off + run.xs);
    }
    marker = std::move(out);
}

// ---------------------------------------------------------------------------
// Labelling, watershed, region range
// ---------------------------------------------------------------------------

void label_components(Image<std::uint32_t>& im, int graph) {
    set_frame<std::uint32_t>(im, 1, 0);
    const i64 nx = im.nx;
    const std::uint32_t* p = im.data();
    const RunComponents cc(nx, im.ny, graph, [p](i64 i) { return p[i] == 1; });
    const auto& runs = cc.runs();
    // Roots are the first run of each component in raster order, so numbering
    // roots in run order reproduces u32_label's raster-order labels (from 2).
    std::vector<std::uint32_t> label_of(runs.size(), 0);
    std::uint32_t lbl = 1;
    for (std::size_t r = 0; r < runs.size(); ++r) {
        if (cc.root(static_cast<i64>(r)) == static_cast<i64>(r)) {
            if (lbl == std::numeric_limits<std::uint32_t>::max()) lbl = 1;
            label_of[r] = ++lbl;
        }
    }
    for (std::size_t r = 0; r < runs.size(); ++r) {
        const auto& run = runs[r];
        std::fill(im.data() + run.y * nx + run.xs, im.data() + run.y * nx + run.xe, label_of[cc.root(r)]);
    }
}

std::uint32_t label_regions(const ImageU8& im, u8 value, int graph, bool bottom_up,
                            Image<std::int32_t>& labels, std::vector<i64>& areas) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    const u8* p = im.data();
    const RunComponents cc(nx, ny, graph, [p, value](i64 i) { return p[i] == value; });
    const auto& runs = cc.runs();

    // Number the components in the order their first run is met.
    std::vector<std::int32_t> label_of(runs.size(), 0);
    std::int32_t n = 0;
    areas.assign(1, 0);
    for (i64 k = 0; k < ny; ++k) {
        const i64 y = bottom_up ? ny - 1 - k : k;
        for (i64 r = cc.row_begin(y); r < cc.row_begin(y + 1); ++r) {
            const i64 root = cc.root(r);
            if (label_of[root] == 0) {
                if (n == std::numeric_limits<std::int32_t>::max()) throw std::overflow_error("too many regions");
                label_of[root] = ++n;
                areas.push_back(0);
            }
            areas[static_cast<std::size_t>(label_of[root])] += runs[r].xe - runs[r].xs;
        }
    }

    labels = Image<std::int32_t>(nx, ny, 0);
    const i64 nruns = static_cast<i64>(runs.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (im.size() >= kParallelThreshold)
#endif
    for (i64 r = 0; r < nruns; ++r) {
        const auto& run = runs[r];
        std::fill(labels.data() + run.y * nx + run.xs, labels.data() + run.y * nx + run.xe, label_of[cc.root(r)]);
    }
    return static_cast<std::uint32_t>(n);
}

void wsfah(Image<std::uint32_t>& iml, const ImageU8& imr, int graph, int maxfl) {
    if (iml.nx != imr.nx || iml.ny != imr.ny) throw std::invalid_argument("wsfah: shape mismatch");
    constexpr std::uint32_t kLabelMax = 0x7FFFFFFF;
    constexpr std::uint32_t kLabelMsb = 0x80000000;
    const i64 nx = iml.nx;
    std::vector<i64> shft;
    if (graph == 4) {
        shft = {-nx, -1, nx, 1};  // set_shift(graph = 4)
    } else if (graph == 8) {
        shft = {-nx, -1, nx, 1, -nx - 1, -nx + 1, nx - 1, nx + 1};  // set_shift(graph = 8)
    } else {
        throw std::invalid_argument("connectivity must be 4 or 8");
    }
    const int nshft = graph;
    set_frame<std::uint32_t>(iml, 1, std::numeric_limits<std::uint32_t>::max());

    ++maxfl;
    std::vector<std::vector<i64>> fah(static_cast<std::size_t>(maxfl));
    std::uint32_t* pl = iml.data();
    const u8* pr = imr.data();
    const i64 npix = iml.size();

    auto init_pass = [&](int k0, int k1) {
        for (i64 i = 0; i < npix; ++i) {
            if (pl[i] > 0 && pl[i] < kLabelMax) {
                for (int k = k0; k < k1; ++k) {
                    const i64 o = i + shft[k];
                    if (pl[o] == 0) {
                        if (pr[o] < maxfl) {
                            pl[o] = pl[i] | kLabelMsb;
                            fah[pr[o]].push_back(o);
                        } else {
                            pl[o] = kLabelMax;
                        }
                    }
                }
            }
        }
    };
    if (nshft == 8) {
        init_pass(0, 4);
        init_pass(4, 8);
    } else {
        init_pass(0, nshft);
    }
    for (i64 i = 0; i < npix; ++i) {
        if (pl[i] > kLabelMax) pl[i] ^= kLabelMsb;
    }

    for (int lev = 0; lev < maxfl; ++lev) {
        for (std::size_t h = 0; h < fah[lev].size(); ++h) {
            const i64 ofs = fah[lev][h];
            for (int k = 0; k < nshft; ++k) {
                const i64 o = ofs + shft[k];
                if (pl[o] == 0) {
                    if (pr[o] < maxfl) {
                        pl[o] = pl[ofs];
                        fah[std::max<int>(lev, pr[o])].push_back(o);
                    } else {
                        pl[o] = kLabelMax;
                    }
                }
            }
        }
        std::vector<i64>().swap(fah[lev]);
    }
    set_frame<std::uint32_t>(iml, 1, 0);
}

}  // namespace gtb
