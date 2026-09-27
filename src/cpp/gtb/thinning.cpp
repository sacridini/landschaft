// SPDX-License-Identifier: GPL-3.0-or-later
//
// Port of binOIthin_FIFO (miallib skel.c): anchored, order-independent
// homotopic thinning of binary images.
// Original algorithm: Marcin Iwanowski and Pierre Soille,
// Copyright (C) 2005-2020 European Union (JRC), GPL-3.0-or-later.

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "primitives.hpp"

namespace gtb {

namespace {

// Simple-pixel lookup tables, verbatim from skel.c (htab), indexed by
//   stype 0: simple g=8, g'=4       stype 3: w-simple g=8, g'=4
//   stype 1: simple g=4, g'=8       stype 4: w-simple g=4, g'=8
//   stype 2: simple g=4, g'=4       stype 5: w-simple g=4, g'=4
const unsigned char kSimpleTable[6][256] = {
    {
        0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,1,1,0,
        1,1,0,0,0,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,1,0,
        1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,1,0,1,1,0,
        0,0,0,0,0,1,0,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,1,0,1,1,0,
        1,0,1,0,0,0,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,
        0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,1,0,
        0,0,1,0,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,1,0,
        0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,
    },
    {
        0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,
        0,1,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,0,0,0,1,0,0,
        0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,
        0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,1,
        0,1,1,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0,
        0,1,1,0,1,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,
        0,1,1,0,1,0,1,0,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,1,
        0,1,1,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,
    },
    {
        0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
        0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,
        0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,
        0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,
        0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,
    },
    {
        1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,1,1,0,
        1,1,0,0,0,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,1,0,
        1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,1,0,1,1,0,
        0,0,0,0,0,1,0,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,1,0,1,1,0,
        1,0,1,0,0,0,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,
        0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,1,0,
        0,0,1,0,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,1,0,
        0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,
    },
    {
        1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,
        1,1,1,0,1,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,1,0,0,0,1,0,0,
        1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,
        1,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,1,1,1,1,0,0,0,1,0,1,
        1,1,1,0,1,0,0,0,1,0,1,0,0,0,0,0,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0,
        1,1,1,0,1,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,
        1,1,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,0,1,0,0,0,1,1,
        1,1,1,0,1,0,1,0,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,
    },
    {
        1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
        1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,
        1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,
        1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,
        0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,
    }
};

const int kSimplePair2[6][4] = {
    {0, 1, 1, 1}, {1, 1, 1, 0}, {0, 1, 1, 0}, {1, 1, 1, 1}, {1, 1, 1, 0}, {1, 1, 1, 0},
};

const int kSimplePair4[6][16] = {
    {0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0},
    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, {1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
    {1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0}, {1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
};

class Thinner {
public:
    Thinner(u8* img, const u8* anchor, int stype, i64 nx) : img_(img), anchor_(anchor), stype_(stype), nx_(nx) {
        // getngbshift8
        sh_[0] = -1;
        sh_[1] = 1;
        sh_[2] = -nx;
        sh_[3] = nx;
        sh_[4] = -nx - 1;
        sh_[5] = nx - 1;
        sh_[6] = -nx + 1;
        sh_[7] = nx + 1;
        sh_[8] = 0;
    }

    bool simple_pixel(i64 pos) const {
        const u8* p = img_ + pos;
        if (*p == 0) return false;
        int code = (p[-1] == 1);
        code |= (p[1] == 1) << 1;
        code |= (p[-nx_] == 1) << 2;
        code |= (p[nx_] == 1) << 3;
        code |= (p[-nx_ - 1] == 1) << 4;
        code |= (p[nx_ - 1] == 1) << 5;
        code |= (p[-nx_ + 1] == 1) << 6;
        code |= (p[nx_ + 1] == 1) << 7;
        return kSimpleTable[stype_][code] != 0;
    }

    // test_anchor with atype == 1
    bool not_anchor(i64 pos) const { return anchor_[pos] == 0; }

    bool indep_simple(i64 pos) const {
        bool allindep = true;
        int ngbnum = 0;
        i64 a = 0, b = 0, c = 0, d = 0, aa = 0, bb = 0;
        int ct = 0;
        int conftype = 0;
        for (int i = 0; i < 8 && allindep; ++i) {
            if (testsimple(pos, sh_[i])) {
                ++ngbnum;
                if (!simple_pair(pos, i, conftype, a, b, c, d)) {
                    allindep = false;
                    aa = a;
                    bb = b;
                    ct = conftype;
                }
            }
        }
        if (allindep) ct = conftype;
        if (!allindep && (stype_ == 1 || stype_ == 2 || stype_ == 4 || stype_ == 5)) {
            if (ct == 4 && testsimple(pos, aa) && testsimple(pos, bb)) {  // edge configuration
                int zerocount = 0;
                for (int i = 0; i < 4; ++i)
                    if (img_[pos + sh_[i]] < img_[pos]) ++zerocount;
                if (zerocount == 2) {
                    allindep = true;
                    if (stype_ < 3) {
                        bool founddiff = false;
                        for (int i = 0; i < 4 && !founddiff; ++i)
                            if (testsimple(pos, sh_[i]) && num_sngb(pos + sh_[i], 4) != 2) founddiff = true;
                        if (!founddiff) allindep = false;
                    }
                }
            }
            return allindep;
        }
        if (allindep && stype_ < 3) {
            // does p belong to a triple or quadruple of isolated simple pixels?
            if ((ngbnum == 2 && (ct == 2 || ct == 3 || ct == 6 || ct == 7 || ct == 9 || ct == 13)) ||
                (ngbnum == 3 && (ct == 4 || ct == 8 || ct == 17))) {
                bool founddiff = false;
                for (int i = 0; i < 8 && !founddiff; ++i)
                    if (testsimple(pos, sh_[i]) && num_sngb(pos + sh_[i], 8) != ngbnum) founddiff = true;
                if (!founddiff) allindep = false;
            }
        }
        return allindep;
    }

    const i64* shifts() const { return sh_; }

private:
    bool testsimple(i64 pos, i64 ngbp) const {
        return not_anchor(pos + ngbp) && simple_pixel(pos + ngbp) && img_[pos] == img_[pos + ngbp];
    }

    int num_sngb(i64 pos, int ngbnum) const {
        int sum = 0;
        for (int i = 0; i < ngbnum; ++i)
            if (img_[pos + sh_[i]] >= img_[pos]) ++sum;
        return sum;
    }

    bool simple_pair(i64 pos, int ngbpos, int& conftype, i64& a, i64& b, i64& c, i64& d) const {
        const u8 v = img_[pos];
        if (ngbpos < 4) {  // non-diagonal neighbour
            switch (ngbpos) {
                case 0: a = sh_[2]; b = sh_[4]; c = sh_[3]; d = sh_[5]; break;
                case 1: a = sh_[2]; b = sh_[6]; c = sh_[3]; d = sh_[7]; break;
                case 2: a = sh_[0]; b = sh_[4]; c = sh_[1]; d = sh_[6]; break;
                default: a = sh_[0]; b = sh_[5]; c = sh_[1]; d = sh_[7]; break;
            }
            int code = (img_[pos + d] >= v);
            code |= (img_[pos + c] >= v) << 1;
            code |= (img_[pos + b] >= v) << 2;
            code |= (img_[pos + a] >= v) << 3;
            conftype = code + 5;
            return kSimplePair4[stype_][code] != 0;
        }
        switch (ngbpos) {  // diagonal neighbour
            case 4: a = sh_[0]; b = sh_[2]; break;
            case 5: a = sh_[3]; b = sh_[0]; break;
            case 6: a = sh_[2]; b = sh_[1]; break;
            default: a = sh_[1]; b = sh_[3]; break;
        }
        int code = (img_[pos + b] >= v);
        code |= (img_[pos + a] >= v) << 1;
        conftype = code + 1;
        return kSimplePair2[stype_][code] != 0;
    }

    u8* img_;
    const u8* anchor_;
    int stype_;
    i64 nx_;
    i64 sh_[9];
};

}  // namespace

void thin_anchored(ImageU8& im, int stype, ImageU8& anchor) {
    if (stype < 0 || stype > 5) throw std::invalid_argument("thin_anchored: invalid stype");
    if (im.nx != anchor.nx || im.ny != anchor.ny) throw std::invalid_argument("thin_anchored: shape mismatch");
    set_frame<u8>(anchor, 2, 0);
    set_frame<u8>(im, 2, 0);

    const i64 nx = im.nx;
    const i64 npix = im.size();
    const i64 start = 2 * nx + 2;
    const i64 end = npix - 2 * nx - 2;
    Thinner th(im.data(), anchor.data(), stype, nx);
    const i64* sh = th.shifts();
    u8* img = im.data();
    Buffer<u8> is_simple(static_cast<std::size_t>(npix), 0);

    // Initial scan: read-only, so it runs in parallel; per-chunk candidate
    // lists are concatenated in chunk order to keep the raster queue order.
    constexpr i64 kChunk = i64(1) << 16;
    const i64 nchunks = end > start ? (end - start + kChunk - 1) / kChunk : 0;
    Buffer<Buffer<i64>> found(static_cast<std::size_t>(nchunks));
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic) if (npix >= kHeavyScanThreshold)
#endif
    for (i64 c = 0; c < nchunks; ++c) {
        const i64 lo = start + c * kChunk;
        const i64 hi = std::min(end, lo + kChunk);
        for (i64 i = lo; i < hi; ++i) {
            if (img[i] && th.simple_pixel(i) && th.not_anchor(i) && th.indep_simple(i)) found[c].push_back(i);
        }
    }
    Buffer<i64> current, next, removed;
    for (const auto& f : found) {
        for (const i64 i : f) {
            is_simple[i] = 1;
            current.push_back(i);
        }
    }
    bool deleted = true;
    while (deleted) {
        deleted = false;
        removed.clear();
        for (const i64 i : current) {
            deleted = true;
            is_simple[i] = 0;
            img[i] = 0;
            removed.push_back(i);
        }
        current.clear();
        next.clear();
        for (const i64 i : removed) {
            for (int j = 0; j < 8; ++j) {
                const i64 n = i + sh[j];
                if (th.not_anchor(n) && th.simple_pixel(n) && is_simple[n] == 0 && th.indep_simple(n)) {
                    is_simple[n] = 1;
                    next.push_back(n);
                }
            }
        }
        current.swap(next);
    }
}

}  // namespace gtb
