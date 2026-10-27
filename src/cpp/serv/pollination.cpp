// SPDX-License-Identifier: GPL-3.0-or-later
//
// Crop pollination (InVEST), see pollination.hpp.

#include "serv/pollination.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <tuple>

#define POCKETFFT_NO_MULTITHREADING
#define POCKETFFT_NO_VECTORS  // numpy's build on MSVC; gcc's vectors give each line the same operations
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include "serv/pocketfft/pocketfft_hdronly.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace serv {

namespace {

using cplx = pocketfft::detail::cmplx<double>;
using RPlan = pocketfft::detail::pocketfft_r<double>;
using CPlan = pocketfft::detail::pocketfft_c<double>;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr float kNaNf = std::numeric_limits<float>::quiet_NaN();
constexpr i64 kLargestBlock = i64{1} << 16;  // pygeoprocessing's _LARGEST_ITERBLOCK
constexpr i64 kTile = 256;                   // its GTiff tiles (BLOCKXSIZE, BLOCKYSIZE)

// numpy's pairwise sum of a contiguous float32 array (np.sum, float32 accumulators).
float pairwise(const float* a, i64 n) {
    if (n < 8) {
        float res = -0.0f;
        for (i64 i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        float r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        i64 i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    i64 n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

struct Block {
    i64 x, y, w, h;
};

// pygeoprocessing.iterblocks of a raster of 256 x 256 tiles: blocks widened, then
// heightened, to about 2^16 cells, by rows of blocks.
std::vector<Block> iterblocks(i64 nx, i64 ny) {
    i64 cols = kTile, rows = kTile, area = cols * rows;
    if (kLargestBlock / area > 0) {
        cols *= kLargestBlock / area;
        if (cols > nx) cols = nx;
        area = cols * rows;
    }
    if (kLargestBlock / area > 0) {
        rows *= kLargestBlock / area;
        if (rows > ny) rows = ny;
    }
    std::vector<Block> out;
    for (i64 y = 0; y < ny; y += rows)
        for (i64 x = 0; x < nx; x += cols) out.push_back({x, y, std::min(cols, nx - x), std::min(rows, ny - y)});
    return out;
}

// scipy's (and pygeoprocessing's) _next_regular: the next 5-smooth number >= base.
i64 next_regular(i64 base) {
    if (base <= 6) return base;
    if (!(base & (base - 1))) return base;
    i64 match = std::numeric_limits<i64>::max();
    i64 p5 = 1;
    while (p5 < base) {
        i64 p35 = p5;
        while (p35 < base) {
            const i64 quotient = (base + p35 - 1) / p35;
            i64 p2 = 1;
            while (p2 < quotient) p2 <<= 1;  // 2 ** (quotient - 1).bit_length()
            const i64 N = p2 * p35;
            if (N == base) return N;
            if (N < match) match = N;
            p35 *= 3;
            if (p35 == base) return p35;
        }
        if (p35 < match) match = p35;
        p5 *= 5;
        if (p5 == base) return p5;
    }
    if (p5 < match) match = p5;
    return match;
}

// numpy.fft.rfftn(x, (fh, fw)) of an h x w block: the real transform of every row
// (numpy's rfft_impl: FFTPACK order unpacked), then the complex one of every column
// padded to fh. ``out``: fh x (fw / 2 + 1), by rows.
void rfft2(const double* x, i64 h, i64 w, i64 fh, i64 fw, std::vector<cplx>& out) {
    const i64 nc = fw / 2 + 1;
    out.assign(static_cast<std::size_t>(fh * nc), cplx(0.0, 0.0));
    RPlan rp(static_cast<std::size_t>(fw));
    std::vector<double> buf(static_cast<std::size_t>(2 * nc));
    for (i64 r = 0; r < h; ++r) {
        double* b = buf.data() + 1;
        for (i64 i = 0; i < w; ++i) b[i] = x[r * w + i];
        for (i64 i = w; i < 2 * nc - 1; ++i) b[i] = 0.0;
        rp.exec(b, 1.0, pocketfft::FORWARD);
        cplx* o = out.data() + r * nc;
        o[0] = cplx(buf[1], 0.0);
        for (i64 k = 1; k < nc; ++k) o[k] = cplx(buf[static_cast<std::size_t>(2 * k)],
                                                 buf[static_cast<std::size_t>(2 * k + 1)]);
    }
    CPlan cp(static_cast<std::size_t>(fh));
    std::vector<cplx> col(static_cast<std::size_t>(fh));
    for (i64 k = 0; k < nc; ++k) {
        for (i64 r = 0; r < fh; ++r) col[static_cast<std::size_t>(r)] = out[static_cast<std::size_t>(r * nc + k)];
        cp.exec(col.data(), 1.0, pocketfft::FORWARD);
        for (i64 r = 0; r < fh; ++r) out[static_cast<std::size_t>(r * nc + k)] = col[static_cast<std::size_t>(r)];
    }
}

// numpy.fft.irfftn(z, (fh, fw))[:sh, :sw]: the inverse complex transform of every
// column (1 / fh), then the inverse real one of the rows kept (numpy's irfft_loop,
// 1 / fw). ``z`` is overwritten.
void irfft2(std::vector<cplx>& z, i64 fh, i64 fw, i64 sh, i64 sw, std::vector<double>& out) {
    const i64 nc = fw / 2 + 1;
    CPlan cp(static_cast<std::size_t>(fh));
    std::vector<cplx> col(static_cast<std::size_t>(fh));
    const double fct_h = 1.0 / static_cast<double>(fh);
    for (i64 k = 0; k < nc; ++k) {
        for (i64 r = 0; r < fh; ++r) col[static_cast<std::size_t>(r)] = z[static_cast<std::size_t>(r * nc + k)];
        cp.exec(col.data(), fct_h, pocketfft::BACKWARD);
        for (i64 r = 0; r < fh; ++r) z[static_cast<std::size_t>(r * nc + k)] = col[static_cast<std::size_t>(r)];
    }
    RPlan rp(static_cast<std::size_t>(fw));
    const double fct_w = 1.0 / static_cast<double>(fw);
    std::vector<double> buf(static_cast<std::size_t>(fw));
    out.resize(static_cast<std::size_t>(sh * sw));
    for (i64 r = 0; r < sh; ++r) {
        const cplx* in = z.data() + r * nc;
        buf[0] = in[0].r;
        const i64 ncopy = std::min(nc - 1, (fw - 1) / 2);
        i64 i = 0;
        for (; i < ncopy; ++i) {
            buf[static_cast<std::size_t>(1 + 2 * i)] = in[1 + i].r;
            buf[static_cast<std::size_t>(2 + 2 * i)] = in[1 + i].i;
        }
        for (; i < (fw - 1) / 2; ++i) buf[static_cast<std::size_t>(1 + 2 * i)] = buf[static_cast<std::size_t>(2 + 2 * i)] = 0.0;
        if (fw % 2 == 0) buf[static_cast<std::size_t>(fw - 1)] = fw / 2 >= nc ? 0.0 : in[fw / 2].r;
        rp.exec(buf.data(), fct_w, pocketfft::BACKWARD);
        std::copy(buf.begin(), buf.begin() + sw, out.begin() + r * sw);
    }
}

// numpy's complex multiply (its AVX2 / FMA3 loop: fmaddsub of a.re b and a.im b swapped).
inline cplx cmul(const cplx& a, const cplx& b) {
    return cplx(std::fma(a.r, b.r, -(a.i * b.i)), std::fma(a.r, b.i, a.i * b.r));
}

struct Pair {
    int s, k;          // signal and kernel block
    i64 fh, fw;        // the FFT shape
    i64 left, top, right, bottom;  // in the raster
    i64 lres, tres, rres, bres;    // in the block's convolution
};

}  // namespace

std::vector<float> exponential_kernel(double alpha, double max_distance, i64& size) {
    const i64 apothem = static_cast<i64>(std::floor(max_distance));
    size = apothem * 2 + 1;
    std::vector<float> k(static_cast<std::size_t>(size * size), 0.0f);
    float running = 0.0f;
    for (const Block& b : iterblocks(size, size)) {
        std::vector<float> tile(static_cast<std::size_t>(b.w * b.h), 0.0f);
        for (i64 r = 0; r < b.h; ++r)
            for (i64 c = 0; c < b.w; ++c) {
                const double y = static_cast<double>(b.y - apothem + r), x = static_cast<double>(b.x - apothem + c);
                const double d = std::sqrt(y * y + x * x);  // numpy.hypot of the integer grid
                if (d <= max_distance) tile[static_cast<std::size_t>(r * b.w + c)] = static_cast<float>(std::exp(-d / alpha));
            }
        running = running + pairwise(tile.data(), b.w * b.h);
        for (i64 r = 0; r < b.h; ++r)
            for (i64 c = 0; c < b.w; ++c)
                k[static_cast<std::size_t>((b.y + r) * size + b.x + c)] = tile[static_cast<std::size_t>(r * b.w + c)];
    }
    for (float& v : k) v = v / running;
    return k;
}

void convolve_2d(const float* signal, i64 ny, i64 nx, const float* kernel, i64 ksize, double* out, int threads) {
    const std::vector<Block> sblocks = iterblocks(nx, ny), kblocks = iterblocks(ksize, ksize);
    const i64 half = ksize / 2;
    // the kernel's sum, tile by tile in float32 (pygeoprocessing's kernel_sum: a numpy float32)
    float ksum = 0.0f;
    std::vector<std::vector<double>> ktiles(kblocks.size());
    for (std::size_t b = 0; b < kblocks.size(); ++b) {
        const Block& kb = kblocks[b];
        std::vector<float> t(static_cast<std::size_t>(kb.w * kb.h));
        ktiles[b].resize(t.size());
        for (i64 r = 0; r < kb.h; ++r)
            for (i64 c = 0; c < kb.w; ++c) {
                const float v = kernel[(kb.y + r) * ksize + kb.x + c];
                t[static_cast<std::size_t>(r * kb.w + c)] = v;
                ktiles[b][static_cast<std::size_t>(r * kb.w + c)] = v;
            }
        ksum = ksum + pairwise(t.data(), kb.w * kb.h);
    }
    // the pairs of blocks that touch the raster, in pygeoprocessing's order
    std::vector<Pair> pairs;
    for (std::size_t s = 0; s < sblocks.size(); ++s)
        for (std::size_t k = 0; k < kblocks.size(); ++k) {
            const Block &sb = sblocks[s], &kb = kblocks[k];
            Pair p{};
            p.s = static_cast<int>(s);
            p.k = static_cast<int>(k);
            p.left = sb.x - half + kb.x;
            p.right = p.left + sb.w + kb.w - 1;
            p.top = sb.y - half + kb.y;
            p.bottom = p.top + sb.h + kb.h - 1;
            if (p.right < 0 || p.bottom < 0 || p.left > nx || p.top > ny) continue;
            p.fh = next_regular(sb.h + kb.h - 1);
            p.fw = next_regular(sb.w + kb.w - 1);
            p.lres = 0;
            p.rres = sb.w + kb.w - 1;
            p.tres = 0;
            p.bres = sb.h + kb.h - 1;
            if (p.left < 0) { p.lres = -p.left; p.left = 0; }
            if (p.top < 0) { p.tres = -p.top; p.top = 0; }
            if (p.right > nx) { p.rres -= p.right - nx; p.right = nx; }
            if (p.bottom > ny) { p.bres -= p.bottom - ny; p.bottom = ny; }
            pairs.push_back(p);
        }
    // the kernel tiles' transforms, once for every FFT shape
    std::map<std::tuple<int, i64, i64>, std::vector<cplx>> kfft;
    for (const Pair& p : pairs) kfft[{p.k, p.fh, p.fw}];
    {
        std::vector<std::pair<const std::tuple<int, i64, i64>, std::vector<cplx>>*> todo;
        for (auto& e : kfft) todo.push_back(&e);
        const int nt = static_cast<int>(todo.size());
#pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (int t = 0; t < nt; ++t) {
            const auto& key = todo[static_cast<std::size_t>(t)]->first;
            const Block& kb = kblocks[static_cast<std::size_t>(std::get<0>(key))];
            rfft2(ktiles[static_cast<std::size_t>(std::get<0>(key))].data(), kb.h, kb.w, std::get<1>(key),
                  std::get<2>(key), todo[static_cast<std::size_t>(t)]->second);
        }
    }
    const i64 n = ny * nx;
    std::vector<float> acc(static_cast<std::size_t>(n), 0.0f), macc(static_cast<std::size_t>(n), 0.0f);
    // the block convolutions in batches, in parallel; added up in their order
    const int batch = std::max(1, threads) * 2;
    const int np = static_cast<int>(pairs.size());
    std::vector<std::vector<double>> res(static_cast<std::size_t>(batch)), mres(static_cast<std::size_t>(batch));
    for (int start = 0; start < np; start += batch) {
        const int stop = std::min(np, start + batch);
#pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (int q = start; q < stop; ++q) {
            const Pair& p = pairs[static_cast<std::size_t>(q)];
            const Block &sb = sblocks[static_cast<std::size_t>(p.s)], &kb = kblocks[static_cast<std::size_t>(p.k)];
            std::vector<double> sig(static_cast<std::size_t>(sb.w * sb.h)), msk(sig.size());
            for (i64 r = 0; r < sb.h; ++r)
                for (i64 c = 0; c < sb.w; ++c) {
                    const float v = signal[(sb.y + r) * nx + sb.x + c];
                    const bool ok = !std::isnan(v);
                    sig[static_cast<std::size_t>(r * sb.w + c)] = ok ? static_cast<double>(v) : 0.0;
                    msk[static_cast<std::size_t>(r * sb.w + c)] = ok ? 1.0 : 0.0;
                }
            const std::vector<cplx>& kf = kfft.at({p.k, p.fh, p.fw});
            const i64 sh = sb.h + kb.h - 1, sw = sb.w + kb.w - 1;
            std::vector<cplx> f;
            rfft2(sig.data(), sb.h, sb.w, p.fh, p.fw, f);
            for (std::size_t i = 0; i < f.size(); ++i) f[i] = cmul(f[i], kf[i]);
            std::vector<double>& r = res[static_cast<std::size_t>(q - start)];
            irfft2(f, p.fh, p.fw, sh, sw, r);
            for (double& v : r)
                if (std::fabs(v) <= 1e-8) v = 0.0;
            rfft2(msk.data(), sb.h, sb.w, p.fh, p.fw, f);
            for (std::size_t i = 0; i < f.size(); ++i) f[i] = cmul(f[i], kf[i]);
            irfft2(f, p.fh, p.fw, sh, sw, mres[static_cast<std::size_t>(q - start)]);
        }
        for (int q = start; q < stop; ++q) {
            const Pair& p = pairs[static_cast<std::size_t>(q)];
            const Block &sb = sblocks[static_cast<std::size_t>(p.s)], &kb = kblocks[static_cast<std::size_t>(p.k)];
            const i64 sw = sb.w + kb.w - 1;
            const std::vector<double>& r = res[static_cast<std::size_t>(q - start)];
            const std::vector<double>& m = mres[static_cast<std::size_t>(q - start)];
            const i64 rows = p.bottom - p.top;
#pragma omp parallel for num_threads(threads) schedule(static)
            for (i64 y = 0; y < rows; ++y) {
                const i64 ry = p.tres + y;
                for (i64 x = 0; x < p.right - p.left; ++x) {
                    const i64 cell = (p.top + y) * nx + p.left + x;
                    if (std::isnan(signal[cell])) continue;
                    const std::size_t j = static_cast<std::size_t>(ry * sw + p.lres + x);
                    acc[static_cast<std::size_t>(cell)] =
                        static_cast<float>(r[j] + static_cast<double>(acc[static_cast<std::size_t>(cell)]));
                    macc[static_cast<std::size_t>(cell)] =
                        static_cast<float>(m[j] + static_cast<double>(macc[static_cast<std::size_t>(cell)]));
                }
            }
        }
    }
    const double ks = static_cast<double>(ksum);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        if (std::isnan(signal[i])) {
            out[i] = kNaN;
            continue;
        }
        const double t = static_cast<double>(acc[static_cast<std::size_t>(i)]);
        const double m = static_cast<double>(macc[static_cast<std::size_t>(i)]);
        out[i] = m > 0 ? (t / m) * ks : t;
    }
}

void pollination_local(const float* const* nesting, const double* ns, int nsub, const float* const* floral,
                       const double* fa, int nseason, i64 n, float* hn, float* const* ff, float* fe, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        bool ok = true;
        double best = -std::numeric_limits<double>::infinity();
        for (int s = 0; s < nsub; ++s) {
            const float v = nesting[s][i];
            if (std::isnan(v)) {
                ok = false;
                break;
            }
            best = std::max(best, static_cast<double>(v) * ns[s]);
        }
        hn[i] = ok && nsub > 0 ? static_cast<float>(best) : kNaNf;
        float sum = 0.0f;
        bool any = false;
        for (int j = 0; j < nseason; ++j) {
            const float v = floral[j][i];
            float f = kNaNf;
            if (!std::isnan(v)) {
                f = static_cast<float>(static_cast<double>(v) * fa[j]);
                sum = sum + f;
                any = true;
            }
            if (ff) ff[j][i] = f;
        }
        fe[i] = any ? sum : kNaNf;
    }
}

void pollination_supply(const float* hn, const double* fr, double sa, i64 n, float* ps, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        if (std::isnan(hn[i]) || std::isnan(fr[i])) {
            ps[i] = kNaNf;
            continue;
        }
        ps[i] = static_cast<float>((sa * static_cast<double>(hn[i])) * fr[i]);
    }
}

void pollination_abundance(const float* ff, const double* fr, const double* cps, i64 n, float* pa, float* total,
                           int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        float v = kNaNf;
        if (!std::isnan(ff[i]) && !std::isnan(fr[i]) && !std::isnan(cps[i]))
            v = fr[i] == 0.0 ? 0.0f : static_cast<float>((static_cast<double>(ff[i]) / fr[i]) * cps[i]);
        if (pa) pa[i] = v;
        if (!std::isnan(v)) total[i] = (std::isnan(total[i]) ? 0.0f : total[i]) + v;
    }
}

void pollination_farm(const float* half_sat, const float* abundance, i64 n, float* fp, float* total, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        float v = kNaNf;
        const float h = half_sat[i], pat = abundance[i];
        if (!std::isnan(h) && !std::isnan(pat)) {
            const float num = pat * (1.0f - h);
            const float twice = 2.0f * pat;
            const float den = h * (1.0f - twice) + pat;
            v = num / den;
        }
        if (fp) fp[i] = v;
        if (!std::isnan(v)) total[i] = (std::isnan(total[i]) ? 0.0f : total[i]) + v;
    }
}

void pollination_yield(const float* managed, const float* farm, i64 n, float* pyt, float* pyw, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) {
        const float mp = managed[i], fp = farm[i];
        if (std::isnan(mp) || std::isnan(fp)) {
            pyt[i] = pyw[i] = kNaNf;
            continue;
        }
        const float sum = mp + fp;
        const float t = sum <= 1.0f || std::isnan(sum) ? sum : 1.0f;  // numpy.minimum, numpy.maximum
        pyt[i] = t;
        const float d = t - mp;
        pyw[i] = d >= 0.0f || std::isnan(d) ? d : 0.0f;
    }
}

}  // namespace serv
