// SPDX-License-Identifier: GPL-3.0-or-later
//
// Port of fsp() from GTB's external_sources/fsp/fsp.c (MSPA v2.3,
// Pierre Soille and Peter Vogt, Copyright (C) 2008-2022 European Union),
// run the way GuidosToolbox runs it (`mspa -disk ...`).
//
// Function names follow fsp.c so the two can be read side by side.

#include "mspa.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <future>
#include <utility>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "primitives.hpp"

namespace gtb {

namespace {

// DEF_PEDU in frag.h, passed to fsp() as a float.
const float kEdu = static_cast<float>(std::sqrt(2.0));

// Prints the duration of each stage to stderr when LANDSCHAFT_PROFILE is set.
class StageTimer {
public:
    StageTimer() : enabled_(std::getenv("LANDSCHAFT_PROFILE") != nullptr), last_(Clock::now()) {}
    void operator()(const char* stage) {
        if (!enabled_) return;
        const auto now = Clock::now();
        std::fprintf(stderr, "[mspa] %-22s %9.1f ms\n", stage,
                     std::chrono::duration<double, std::milli>(now - last_).count());
        last_ = now;
    }

private:
    using Clock = std::chrono::steady_clock;
    bool enabled_;
    Clock::time_point last_;
};

// Runs `f` on its own thread when `parallel`, otherwise lazily on get().
// Task threads use single-threaded OpenMP kernels to avoid oversubscription.
template <class F>
auto spawn(bool parallel, F&& f) -> std::future<decltype(f())> {
    if (!parallel) return std::async(std::launch::deferred, std::forward<F>(f));
    return std::async(std::launch::async, [fn = std::forward<F>(f)]() mutable {
#if defined(_OPENMP)
        omp_set_num_threads(1);
#endif
        return fn();
    });
}

// The distance thresholds below have two implementations:
//  * literal: full squared EDT (sqedt) + float threshold, as fsp.c;
//  * fast: exact integer test d^2 <= T via dilate_disk.
// They agree whenever the EDT has at least one source pixel and every true
// squared distance fits the original 32-bit/float arithmetic, i.e. images
// smaller than kFastMaxSide. Otherwise sqedt overflows in the original and
// the literal version reproduces that.
constexpr i64 kFastMaxSide = 46000;

bool fast_distances(const ImageU8& im) { return std::max(im.nx, im.ny) < kFastMaxSide; }

bool has_zero(const ImageU8& im) {
    return std::any_of(im.px.begin(), im.px.end(), [](u8 v) { return v == 0; });
}

// getexternalboundary: pixels outside `im` within Euclidean distance
// size*edu of it.
ImageU8 external_boundary(const ImageU8& im, float size, float edu) {
    const float s = size * edu;
    if (fast_distances(im) && !is_empty(im)) {
        ImageU8 out = dilate_disk(im, squared_threshold(s));
        const u8* a = im.data();
        u8* o = out.data();
        parallel_chunks(out.size(), [=](i64 b, i64 e) {
            for (i64 i = b; i < e; ++i) o[i] = a[i] ? 0 : o[i];
        });
        return out;
    }
    const ImageU8 nim = negation(im);
    const Buffer<std::uint32_t> d2 = sqedt(nim);
    ImageU8 out(im.nx, im.ny);
    const u8* n = nim.data();
    u8* o = out.data();
    parallel_chunks(out.size(), [&](i64 b, i64 e) {
        for (i64 i = b; i < e; ++i) {
            const float r = edt_value(d2[i]);
            o[i] = (r >= 0.0f && r <= s) ? n[i] : 0;
        }
    });
    return out;
}

// getcore: foreground pixels farther than size*edu from the background.
ImageU8 core_mask(const ImageU8& im, float size, float edu) {
    const float s = size * edu;
    if (fast_distances(im) && has_zero(im)) {
        ImageU8 out = dilate_disk(negation(im), squared_threshold(s));
        const u8* a = im.data();
        u8* o = out.data();
        parallel_chunks(out.size(), [=](i64 b, i64 e) {
            for (i64 i = b; i < e; ++i) o[i] = (a[i] && !o[i]) ? 1 : 0;
        });
        return out;
    }
    const Buffer<std::uint32_t> d2 = sqedt(im);
    ImageU8 out(im.nx, im.ny);
    u8* o = out.data();
    parallel_chunks(out.size(), [&](i64 b, i64 e) {
        for (i64 i = b; i < e; ++i) {
            const float r = edt_value(d2[i]);
            o[i] = (r > s && r < 65535.0f) ? 1 : 0;
        }
    });
    return out;
}

// getexternalboundarygeodesic
ImageU8 geodesic_boundary(const ImageU8& ref, const ImageU8& mask, float size, float edu) {
    const float s = size * edu;
    // ced gives 0 on `ref` and at least 1 elsewhere (the length of a non-zero
    // integer vector plus a non-negative base distance), so no pixel lies in
    // [0.0001, s] when s < 1: edge width 1 needs no geodesic distance.
    if (s < 1.0f) return ImageU8(ref.nx, ref.ny, 0);
    return ced_band(ref, mask, 0.0001f, s);  // pixels at distance 0 are below 0.0001
}

// fm_preproc: foreground plus the missing pixels within size*edu of it.
ImageU8 fm_preproc(const ImageU8& fm, float size, float edu, bool par) {
    auto dda_task = spawn(par, [&] {
        ImageU8 da = fm;
        threshold(da, 1, 255, 0, 1);
        return external_boundary(da, size, edu);
    });
    ImageU8 fa = fm;
    threshold(fa, 2, 255, 0, 1);
    ImageU8 dfa = external_boundary(fa, size, edu);
    bit_or(dfa, fa);
    const ImageU8 dda = dda_task.get();
    inf(dfa, dda);
    sup(dfa, fa);
    return dfa;
}

// fm_preproc2: add a frame of width s replicating the outermost rows and
// columns. As in the original, the last frame column and row stay 0.
ImageU8 fm_preproc2(const ImageU8& im, i64 s) {
    const i64 nx = im.nx;
    const i64 ny = im.ny;
    ImageU8 out = add_frame<u8>(im, s, 0);
    for (i64 y = 0; y < ny; ++y) {
        const u8 left = im.at(0, y);
        const u8 right = im.at(nx - 1, y);
        for (i64 x = 0; x < s; ++x) out.at(x, y + s) |= left;
        for (i64 x = nx + s - 1; x < nx + 2 * s - 1; ++x) out.at(x, y + s) |= right;
    }
    for (i64 x = 0; x < nx; ++x) {
        const u8 top = im.at(x, 0);
        const u8 bottom = im.at(x, ny - 1);
        for (i64 y = 0; y < s; ++y) out.at(x + s, y) |= top;
        for (i64 y = ny + s - 1; y < ny + 2 * s - 1; ++y) out.at(x + s, y) |= bottom;
    }
    return out;
}

// setedges: returns {edges, perforations}. `core` is getcore(im) and
// `core_boundary` gives getexternalboundary(core), which the openings share.
std::pair<ImageU8, ImageU8> set_edges(ImageU8 core, float size, int graphfg, int graphbg, float edu,
                                      std::shared_future<ImageU8>& core_boundary, bool par) {
    const ImageU8 core1 = core;
    ImageU8 corefill = fill_holes(core, graphbg);
    ImageU8 loecher = corefill;  // holes of the core
    sub_sat(loecher, core);
    ImageU8 edges(core.nx, core.ny, 0);

    while (!is_empty(corefill)) {
        auto boundary = spawn(par, [&] { return external_boundary(corefill, size, edu); });
        const ImageU8 i0 = fill_holes(loecher, graphfg);
        bit_or(edges, boundary.get());
        sub_sat(corefill, i0);
        sub_sat(core, corefill);
        corefill = fill_holes(core, graphbg);
        loecher = corefill;
        sub_sat(loecher, core);
    }

    ImageU8 outer = core_boundary.get();
    bit_and(edges, negation(core1));
    sub_sat(outer, edges);
    return {std::move(edges), std::move(outer)};
}

// getconnector2core. With `-disk` the core is re-read from disk here, so the
// 2-pixel border that the thinning clears in its anchor stays local.
ImageU8 connector_to_core(const ImageU8& core_in, const ImageU8& opening, const ImageU8& residues, float size,
                          int oitype, int graphfg, float edu, StageTimer& timer) {
    ImageU8 sk = opening;
    bit_or(sk, residues);
    ImageU8 core = core_in;
    thin_anchored(sk, oitype, core);
    sub_sat(sk, core);
    timer("thinning");

    ImageU8 i0 = opening;
    sub_sat(i0, core);
    bit_or(i0, residues);

    ImageU8 connector = geodesic_boundary(sk, i0, size - 1.0f, edu);
    timer("geodesic distance");
    bit_or(connector, sk);
    bit_and(connector, i0);
    reconstruct_dilation(sk, i0, graphfg);
    bit_and(connector, sk);
    return connector;
}

Image<std::uint32_t> to_u32(const ImageU8& im) {
    Image<std::uint32_t> out(im.nx, im.ny);
    const u8* a = im.data();
    std::uint32_t* o = out.data();
    parallel_chunks(im.size(), [=](i64 b, i64 e) {
        for (i64 i = b; i < e; ++i) o[i] = a[i];
    });
    return out;
}

// getcorridor: connectors joining two different cores (bridges).
// `lbl` holds the labelled cores (label_components of the core image).
ImageU8 get_corridor(const ImageU8& connector, Image<std::uint32_t> lbl, const ImageU8& opening, int graph,
                     bool par, StageTimer& timer) {
    auto cor_task = spawn(par, [&] {  // the connectors' components, independent of the watershed
        Image<std::uint32_t> cor = to_u32(connector);
        label_components(cor, graph);
        return cor;
    });
    ImageU8 imref = opening;
    bit_or(imref, connector);
    set_level(imref, 0, 0, 255);
    wsfah(lbl, imref, graph, 254);
    timer("corridor: watershed");
    Image<std::uint32_t> cor = cor_task.get();
    timer("corridor: labels");

    // set_regions(cor, lbl, 20) gives every connector component (and the
    // background, label 0) max(lbl) - min(lbl) over it; then
    // i32_setlevel(INT_MAX -> 0) and i32_thresh(1, INT_MAX). All values are
    // below 2^31, so the int32/double arithmetic of the original reduces to
    // this integer version.
    std::uint32_t max_label = 0;
    for (const std::uint32_t c : cor.px) max_label = std::max(max_label, c);
    Buffer<std::uint32_t> lo(static_cast<std::size_t>(max_label) + 1, std::numeric_limits<std::uint32_t>::max());
    Buffer<std::uint32_t> hi(static_cast<std::size_t>(max_label) + 1, 0);
    for (i64 i = 0; i < cor.size(); ++i) {
        const std::uint32_t c = cor[i];
        const std::uint32_t v = lbl[i];
        if (v < lo[c]) lo[c] = v;
        if (v > hi[c]) hi[c] = v;
    }
    Buffer<u8> is_bridge(lo.size(), 0);
    for (std::size_t c = 0; c < lo.size(); ++c) {
        if (lo[c] > hi[c]) continue;  // label not present
        const std::uint32_t range = hi[c] - lo[c];
        is_bridge[c] = (range >= 1 && range != static_cast<std::uint32_t>(INT_MAX)) ? 1 : 0;
    }
    ImageU8 out(cor.nx, cor.ny);
    for (i64 i = 0; i < cor.size(); ++i) out[i] = is_bridge[cor[i]];
    return out;
}

void check_input(const ImageU8& input, const MspaParams& p) {
    if (p.connectivity != 4 && p.connectivity != 8) throw std::invalid_argument("connectivity must be 4 or 8");
    if (p.edge_width < 1 || p.edge_width > 255) throw std::invalid_argument("edge_width must be in [1, 255]");
    if (input.nx < 1 || input.ny < 1) throw std::invalid_argument("input image is empty");
    if (max_value(input) > 2)
        throw std::invalid_argument("input values must be 0 (missing), 1 (background) or 2 (foreground)");
}

}  // namespace

ImageU8 mspa(const ImageU8& input, const MspaParams& p) {
    check_input(input, p);
    const PoolScope pool_scope;  // reuse the buffers of the many intermediate images
    StageTimer timer;

    const float eew = static_cast<float>(p.edge_width);
    const float size = static_cast<float>((static_cast<double>(eew) + 0.98) / std::sqrt(2.0));
    const float edu = kEdu;
    const int graphfg = p.connectivity;
    const int graphbg = graphfg == 8 ? 4 : 8;
    const int oitype = graphfg == 8 ? 0 : 1;
    const int bufsize = static_cast<int>((size * 1.5) + 0.5);
    const i64 frame = bufsize + 1;  // (int)(bufsize + 1.5)
    // fsp.c has extra processing for size == 1, which cannot happen for an
    // integer edge width >= 1 (size = (eew + 0.98) / sqrt(2)).

    // Independent stages run as parallel tasks when more than one thread is
    // allowed; the result does not depend on it.
    const bool par = p.threads != 1;

    ImageU8 all_holes;
    ImageU8 i0;
    {
        const ImageU8 im = add_frame<u8>(input, frame, 0);
        auto holes = spawn(par && p.internal, [&] {
            ImageU8 h;
            if (p.internal) {
                const ImageU8 i1 = fm_preproc(im, 1.0f, edu, false);
                h = fill_holes(i1, graphbg);
                sub_sat(h, i1);
            }
            return h;
        });
        i0 = fm_preproc(im, size, edu, par);
        all_holes = holes.get();
    }
    i0 = fm_preproc2(sub_frame(i0, frame), frame);
    ImageU8 out = i0;
    timer("preprocessing");

    const ImageU8 core = core_mask(i0, size, edu);
    auto patch_task = spawn(par, [&] {  // getpatch
        ImageU8 reached = core;
        reconstruct_dilation(reached, i0, graphfg);
        ImageU8 patch = i0;
        sub_sat(patch, reached);
        return patch;
    });
    std::shared_future<ImageU8> core_boundary = spawn(par, [&] { return external_boundary(core, size, edu); }).share();
    auto labels_task = spawn(par, [&] {
        Image<std::uint32_t> lbl = to_u32(core);
        label_components(lbl, graphfg);
        return lbl;
    });
    // Core-openings (+100) and border-openings (220), which only need the core.
    auto internal_task = spawn(par && p.internal, [&] {
        std::pair<ImageU8, ImageU8> r;
        if (p.internal) {
            ImageU8 tmp = fill_holes(core, graphbg);
            ImageU8 core_holes = tmp;
            sub_sat(core_holes, core);
            ImageU8 border_openings = all_holes;
            bit_and(border_openings, negation(core_holes));
            set_level(border_openings, 1, 1, mspa_class::kBorderOpening);
            sub_sat(tmp, core);
            ImageU8 core_openings = fill_holes(tmp, graphbg);
            set_level(core_openings, 1, 1, mspa_class::kCoreOpening);
            r = {std::move(core_openings), std::move(border_openings)};
        }
        return r;
    });

    auto [edges, perforation] = set_edges(core, size, graphfg, graphbg, edu, core_boundary, par);
    ImageU8 patch = patch_task.get();  // before i0 is modified below
    timer("core/patch/edges");

    ImageU8& residues = i0;
    sub_sat(residues, core);
    sub_sat(residues, patch);
    shift_left(patch, 3);
    bit_or(out, patch);
    sub_sat(residues, perforation);
    shift_left(perforation, 2);
    bit_or(out, perforation);
    sub_sat(residues, edges);
    shift_left(edges, 1);
    bit_or(out, edges);

    ImageU8 opening = core_boundary.get();  // getopening: boundary + core
    add_sat(opening, core);
    timer("residues");
    ImageU8 connector = connector_to_core(core, opening, residues, size, oitype, graphfg, edu, timer);
    timer("connector");
    ImageU8 corridor = get_corridor(connector, labels_task.get(), opening, graphfg, par, timer);
    timer("corridor");
    {
        ImageU8 core_bits = core;
        shift_left(core_bits, 4);
        bit_or(out, core_bits);
    }
    ImageU8& shortcut = connector;
    sub_sat(shortcut, corridor);
    shift_left(corridor, 5);
    shift_left(shortcut, 6);
    bit_or(out, corridor);
    bit_or(out, shortcut);

    if (p.internal) {
        auto [core_openings, border_openings] = internal_task.get();
        add_sat(out, core_openings);
        sup(out, border_openings);
    }

    timer("internal");
    out = sub_frame(out, frame);
    fill_where_zero(out, input, mspa_class::kMissing);
    // GuidosToolbox post-processing for EdgeWidth == 1: value 2 -> 0.
    if (p.edge_width == 1) set_level(out, 2, 2, 0);
    return out;
}

}  // namespace gtb
