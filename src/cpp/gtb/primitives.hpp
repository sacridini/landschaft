// SPDX-License-Identifier: GPL-3.0-or-later
//
// Morphological primitives needed by MSPA.
//
// Every function here reproduces the observable behaviour of the miallib
// routine named in its comment (https://github.com/ec-jrc/jeolib-miallib),
// including integer/float arithmetic, neighbour visiting order and queue
// order wherever those influence the result. Deviating from them breaks
// bit-exact agreement with the original GTB/GWB `mspa` executable.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "image.hpp"

namespace gtb {

// Below this many pixels, data-parallel loops stay single-threaded: short
// OpenMP regions cost more than they save (idle workers spin afterwards and
// slow down the sequential stages that follow).
constexpr i64 kParallelThreshold = i64(1) << 24;

// Calls f(begin, end) on consecutive chunks of [0, n), in parallel for large
// n. Keeping the per-pixel loop inside `f` lets the compiler vectorise it.
template <class F>
inline void parallel_chunks(i64 n, F&& f) {
    constexpr i64 kChunk = i64(1) << 16;
    const i64 nchunks = (n + kChunk - 1) / kChunk;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n >= kParallelThreshold)
#endif
    for (i64 c = 0; c < nchunks; ++c) f(c * kChunk, std::min(n, (c + 1) * kChunk));
}

// ---------------------------------------------------------------------------
// Pointwise operations on 8-bit images (miallib pointop.c semantics).
// ---------------------------------------------------------------------------

// negation(): 0 -> 1, anything else -> 0.
ImageU8 negation(const ImageU8& im);
// a |= b, a &= b
void bit_or(ImageU8& a, const ImageU8& b);
void bit_and(ImageU8& a, const ImageU8& b);
// generic_arith SUB_op: a = max(a - b, 0)
void sub_sat(ImageU8& a, const ImageU8& b);
// generic_arith ADD_op: a = min(a + b, 255)
void add_sat(ImageU8& a, const ImageU8& b);
// generic_arith INF_op / SUP_op
void inf(ImageU8& a, const ImageU8& b);
void sup(ImageU8& a, const ImageU8& b);
// generic_shift(im, -k): a <<= k
void shift_left(ImageU8& a, int k);
// generic_setlevel: pixels in [t1, t2] -> value
void set_level(ImageU8& a, u8 t1, u8 t2, u8 value);
// generic_thresh: pixels in [t1, t2] -> fg, others -> bg
void threshold(ImageU8& a, u8 t1, u8 t2, u8 bg, u8 fg);
// true when every pixel is zero (miallib volume(...) == 0)
bool is_empty(const ImageU8& a);

// ---------------------------------------------------------------------------
// Distances
// ---------------------------------------------------------------------------

// uc_sqedt (efedt.c): squared Euclidean distance of every non-zero pixel to
// the nearest zero pixel (Meijster, Roerdink & Hesselink), with the exact
// integer types of the original (result truncated to 32 bits).
std::vector<std::uint32_t> sqedt(const ImageU8& im);

// uint32_to_float followed by imsqrt, as used after sqedt.
inline float edt_value(std::uint32_t d2) {
    const float f = static_cast<float>(d2);
    return static_cast<float>(std::sqrt(static_cast<double>(f)));
}

// Largest integer T with edt_value(T) <= s (edt_value is monotone), i.e.
// "edt_value(d2) <= s" is equivalent to "d2 <= T" for d2 < 2^24.
// Returns -1 when no squared distance qualifies (s < 0).
std::int64_t squared_threshold(float s);

// Pixels whose squared Euclidean distance to the nearest non-zero pixel of
// `src` is <= T (a dilation by the digital disk x^2 + y^2 <= T). Exact.
ImageU8 dilate_disk(const ImageU8& src, std::int64_t T);

// ced (ced.c): constrained (geodesic) Euclidean distance from `ref` inside
// `mask`, using the original FIFO propagation and Bresenham visibility test.
std::vector<float> ced(const ImageU8& ref, const ImageU8& mask);

// ---------------------------------------------------------------------------
// Reconstruction / connectivity (binary images with values 0/1)
// ---------------------------------------------------------------------------

// uc_fillhole (fsp.c): fill background regions not `graph`-connected to the
// image border.
ImageU8 fill_holes(const ImageU8& im, int graph);

// uc_rdil(marker, mask, graph, 1): binary reconstruction by dilation.
void reconstruct_dilation(ImageU8& marker, const ImageU8& mask, int graph);

// u32_label (label.c) with a 3x3 structuring element of connectivity
// `graph`: pixels equal to 1 are labelled 2, 3, ... in raster order; the
// 1-pixel image border is cleared first.
void label_components(Image<std::uint32_t>& im, int graph);

// wsfah (wsfah.c) for UINT32 labels and UCHAR reference image.
void wsfah(Image<std::uint32_t>& labels, const ImageU8& ref, int graph, int maxfl);

// binOIthin_FIFO (skel.c): anchored order-independent homotopic thinning.
// Both images get their 2-pixel border cleared, as in the original.
void thin_anchored(ImageU8& im, int stype, ImageU8& anchor);

}  // namespace gtb
