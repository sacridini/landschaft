// SPDX-License-Identifier: GPL-3.0-or-later
//
// Port of the 'distance_influence' / 'distance_proximity' event of
// guidostoolbox.pro (GuidosToolbox, (C) European Union (JRC), Peter Vogt,
// GPL-3.0-or-later), with ports of the IDL routines it calls.
#include "influence.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "primitives.hpp"

namespace gtb {

namespace {

template <class T>
Image<T> flip_rows(const Image<T>& im) {
    Image<T> out(im.nx, im.ny);
    for (i64 y = 0; y < im.ny; ++y) {
        std::copy_n(im.data() + (im.ny - 1 - y) * im.nx, im.nx, out.data() + y * im.nx);
    }
    return out;
}

template <class T>
Image<T> crop(const Image<T>& im, i64 b) {
    return sub_frame(im, b);
}

// Offsets of the 8 (graph 8) or 4 neighbours, in raster order.
struct Neighbours {
    int n = 0;
    std::array<int, 8> dx{}, dy{};
    explicit Neighbours(int graph) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                if ((x == 0 && y == 0) || (graph == 4 && x != 0 && y != 0)) continue;
                dx[static_cast<std::size_t>(n)] = x;
                dy[static_cast<std::size_t>(n)] = y;
                ++n;
            }
        }
    }
};

// IDL ROUND of a single-precision value: half away from zero.
inline i64 idl_round(float v) { return static_cast<i64>(std::round(v)); }

}  // namespace

// ---------------------------------------------------------------------------
// IDL routines
// ---------------------------------------------------------------------------

std::vector<float> morph_distance(const ImageU8& mask, bool background, DistanceMetric metric) {
    const i64 nx = mask.nx, ny = mask.ny, n = mask.size();
    // The pixels whose distance is measured are the "set" pixels; the others are the sources.
    ImageU8 set(nx, ny);
    for (i64 i = 0; i < n; ++i) set[i] = background ? (mask[i] == 0) : (mask[i] != 0);
    std::vector<float> out(static_cast<std::size_t>(n), 0.0f);

    if (metric == DistanceMetric::Euclidean) {
        const std::vector<std::uint32_t> d2 = sqedt(set);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n >= kParallelThreshold)
#endif
        for (i64 i = 0; i < n; ++i) {
            if (set[i]) out[static_cast<std::size_t>(i)] = static_cast<float>(std::sqrt(static_cast<double>(d2[i])));
        }
        return out;
    }

    // Quasi-Euclidean chamfer distance (orthogonal steps 1, diagonal steps sqrt(2)) in single
    // precision, as one propagation pass per direction in the order that best reproduces IDL's
    // MORPH_DISTANCE(NEIGHBOR_SAMPLING=3) on GTB outputs (image rows as GTB stores them, i.e.
    // flipped): right, down, up, down-right, down-left, left, up-right, up-left. IDL still differs
    // at some pixels next to the boundaries between the zones of different sources.
    const float inf = std::numeric_limits<float>::infinity();
    const float diag = std::sqrt(2.0f);
    std::vector<float>& d = out;
    for (i64 i = 0; i < n; ++i) d[static_cast<std::size_t>(i)] = set[i] ? inf : 0.0f;
    constexpr int kDirs[8][2] = {{0, 1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {0, -1}, {-1, 1}, {-1, -1}};  // (dy, dx)
    for (const auto& dir : kDirs) {
        const int dy = dir[0], dx = dir[1];
        const float w = (dy != 0 && dx != 0) ? diag : 1.0f;
        if (dy == 0) {  // along the rows, column by column
            for (i64 k = 1; k < nx; ++k) {
                const i64 x = dx > 0 ? k : nx - 1 - k;
                for (i64 y = 0; y < ny; ++y) {
                    float& v = d[static_cast<std::size_t>(y * nx + x)];
                    v = std::min(v, d[static_cast<std::size_t>(y * nx + x - dx)] + w);
                }
            }
        } else {  // row by row from the previous row
            for (i64 k = 1; k < ny; ++k) {
                const i64 y = dy > 0 ? k : ny - 1 - k;
                const float* prev = d.data() + (y - dy) * nx;
                float* row = d.data() + y * nx;
                for (i64 x = 0; x < nx; ++x) {
                    const i64 px = x - dx;
                    if (px < 0 || px >= nx) continue;
                    row[x] = std::min(row[x], prev[px] + w);
                }
            }
        }
    }
    return out;
}

ImageU8 hist_equal(const std::vector<float>& im, i64 nx, i64 ny) {
    const i64 n = nx * ny;
    if (n == 0) return ImageU8(nx, ny);
    const auto [lo, hi] = std::minmax_element(im.begin(), im.end());
    const double minv = *lo, maxv = *hi;
    if (!(minv < maxv)) throw std::invalid_argument("hist_equal: MINV must be less than MAXV");
    // hist = HISTOGRAM(img, BINSIZE=bs, MIN=minv, MAX=maxv) & hist[0] = 0
    const double bs = (maxv - minv) / 5000.0;
    const i64 nbins = static_cast<i64>(std::floor((maxv - minv) / bs)) + 1;
    std::vector<double> cumu(static_cast<std::size_t>(nbins), 0.0);
    for (const float v : im) {
        const i64 b = static_cast<i64>(std::floor((static_cast<double>(v) - minv) / bs));
        if (b >= 0 && b < nbins) cumu[static_cast<std::size_t>(b)] += 1.0;
    }
    cumu[0] = 0.0;
    for (std::size_t b = 1; b < cumu.size(); ++b) cumu[b] += cumu[b - 1];
    // BYTSCL(cumu, TOP=255): byte((TOP + 0.9999) * (x - min) / (max - min))
    const double cmin = *std::min_element(cumu.begin(), cumu.end());
    const double cmax = *std::max_element(cumu.begin(), cumu.end());
    std::vector<u8> lut(cumu.size(), 0);
    if (cmax > cmin) {
        for (std::size_t b = 0; b < cumu.size(); ++b) {
            lut[b] = static_cast<u8>((255.0 + 0.9999) * (cumu[b] - cmin) / (cmax - cmin));
        }
    }
    ImageU8 out(nx, ny);
    for (i64 i = 0; i < n; ++i) {
        // bytsc[img / bs] (MINV = 0) or bytsc[((img > minv) - minv) / bs]
        const double v = std::max(static_cast<double>(im[static_cast<std::size_t>(i)]), minv);
        i64 b = static_cast<i64>((v - minv) / bs);
        b = std::min(b, nbins - 1);
        out[i] = lut[static_cast<std::size_t>(b)];
    }
    return out;
}

Image<std::int32_t> watershed(const ImageU8& im, int graph, std::int32_t& nregions) {
    if (graph != 4 && graph != 8) throw std::invalid_argument("connectivity must be 4 or 8");
    const i64 nx = im.nx, ny = im.ny, n = im.size();
    constexpr std::int32_t kMask = -2, kInit = -1, kWshed = 0;
    constexpr i64 kFictitious = -1;
    const Neighbours nb(graph);

    // pixels sorted by grey level, raster order within a level (counting sort)
    std::array<i64, 257> start{};
    for (i64 i = 0; i < n; ++i) ++start[static_cast<std::size_t>(im[i]) + 1];
    for (std::size_t h = 1; h < start.size(); ++h) start[h] += start[h - 1];
    std::vector<i64> order(static_cast<std::size_t>(n));
    {
        std::array<i64, 256> pos{};
        std::copy(start.begin(), start.end() - 1, pos.begin());
        for (i64 i = 0; i < n; ++i) order[static_cast<std::size_t>(pos[im[i]]++)] = i;
    }

    Image<std::int32_t> lab(nx, ny, kInit);
    std::vector<std::int32_t> dist(static_cast<std::size_t>(n), 0);
    std::deque<i64> fifo;
    std::int32_t curlab = 0;

    auto for_neighbours = [&](i64 p, auto&& f) {
        const i64 x = p % nx, y = p / nx;
        for (int k = 0; k < nb.n; ++k) {
            const i64 qx = x + nb.dx[static_cast<std::size_t>(k)], qy = y + nb.dy[static_cast<std::size_t>(k)];
            if (qx < 0 || qy < 0 || qx >= nx || qy >= ny) continue;
            f(qy * nx + qx);
        }
    };

    for (int h = 0; h < 256; ++h) {
        const i64 b = start[static_cast<std::size_t>(h)], e = start[static_cast<std::size_t>(h) + 1];
        if (b == e) continue;
        for (i64 k = b; k < e; ++k) {
            const i64 p = order[static_cast<std::size_t>(k)];
            lab[p] = kMask;
            bool touches = false;
            for_neighbours(p, [&](i64 q) { touches = touches || lab[q] > 0 || lab[q] == kWshed; });
            if (touches) {
                dist[static_cast<std::size_t>(p)] = 1;
                fifo.push_back(p);
            }
        }
        std::int32_t curdist = 1;
        fifo.push_back(kFictitious);
        while (true) {
            i64 p = fifo.front();
            fifo.pop_front();
            if (p == kFictitious) {
                if (fifo.empty()) break;
                fifo.push_back(kFictitious);
                ++curdist;
                p = fifo.front();
                fifo.pop_front();
            }
            for_neighbours(p, [&](i64 q) {
                if (dist[static_cast<std::size_t>(q)] < curdist && (lab[q] > 0 || lab[q] == kWshed)) {
                    if (lab[q] > 0) {
                        if (lab[p] == kMask || lab[p] == kWshed) {
                            lab[p] = lab[q];
                        } else if (lab[p] != lab[q]) {
                            lab[p] = kWshed;
                        }
                    } else if (lab[p] == kMask) {
                        lab[p] = kWshed;
                    }
                } else if (lab[q] == kMask && dist[static_cast<std::size_t>(q)] == 0) {
                    dist[static_cast<std::size_t>(q)] = curdist + 1;
                    fifo.push_back(q);
                }
            });
        }
        // new minima at level h
        for (i64 k = b; k < e; ++k) {
            const i64 p = order[static_cast<std::size_t>(k)];
            dist[static_cast<std::size_t>(p)] = 0;
            if (lab[p] == kMask) {
                ++curlab;
                fifo.push_back(p);
                lab[p] = curlab;
                while (!fifo.empty()) {
                    const i64 q = fifo.front();
                    fifo.pop_front();
                    for_neighbours(q, [&](i64 r) {
                        if (lab[r] == kMask) {
                            fifo.push_back(r);
                            lab[r] = curlab;
                        }
                    });
                }
            }
        }
    }
    nregions = curlab;
    return lab;
}

// ---------------------------------------------------------------------------
// Influence Zones / Proximity
// ---------------------------------------------------------------------------

InfluenceResult compute_influence(const ImageU8& input, i64 min_area, int core_buffer, int buffer, bool proximity,
                                  DistanceMetric metric) {
    constexpr i64 kPad = 5;
    // GTB works on the image as displayed: rows flipped (rotate(im, 7)).
    const ImageU8 orig = flip_rows(input);
    const i64 nx = orig.nx, ny = orig.ny;

    // ext = bytarr(sz + 10) + 1b & ext[5:..] = im
    ImageU8 ext = add_frame<u8>(orig, kPad, 1);
    const i64 ex = ext.nx, ey = ext.ny, en = ext.size();

    if (core_buffer > 0) {  // core zones: drop the foreground within core_buffer of its edge
        ImageU8 fg(ex, ey);
        for (i64 i = 0; i < en; ++i) fg[i] = ext[i] == 2;
        const std::vector<float> bo = morph_distance(fg, /*background=*/false, metric);
        const float limit = static_cast<float>(core_buffer) + 0.5f;
        for (i64 i = 0; i < en; ++i) {
            if (ext[i] == 2 && bo[static_cast<std::size_t>(i)] < limit) ext[i] = 1;
        }
    }

    // b) remove small objects: keep the 8-connected objects of >= min_area pixels
    Image<std::int32_t> lx;
    std::vector<i64> h;
    label_regions(ext, 2, 8, /*bottom_up=*/false, lx, h);
    h[0] = en - std::accumulate(h.begin() + 1, h.end(), i64(0));
    std::vector<std::int32_t> qok;
    for (std::size_t k = 0; k < h.size(); ++k) {
        if (h[k] >= min_area) qok.push_back(static_cast<std::int32_t>(k));
    }
    if (qok.size() <= 1) throw std::invalid_argument("the area threshold is too large: no object is kept");
    std::vector<u8> keep(h.size(), 0);
    for (std::size_t k = 1; k < qok.size(); ++k) keep[static_cast<std::size_t>(qok[k])] = 1;  // GTB skips qok[0]
    ImageU8 obj(ex, ey, 0);
    for (i64 i = 0; i < en; ++i) obj[i] = keep[static_cast<std::size_t>(lx[i])] ? 2 : 0;

    // a) close holes: 4-connected non-object regions other than the outer one
    //    (IDL label_region leaves the array edge unlabelled, label 1 is the outer region)
    ImageU8 bg(ex, ey, 0);
    for (i64 i = 0; i < en; ++i) bg[i] = obj[i] != 2;
    set_frame<u8>(bg, 1, 0);
    Image<std::int32_t> bl;
    std::vector<i64> bla;
    label_regions(bg, 1, 4, false, bl, bla);
    std::vector<u8> hole(static_cast<std::size_t>(en), 0);
    for (i64 i = 0; i < en; ++i) {
        if (bl[i] > 1) {
            hole[static_cast<std::size_t>(i)] = 1;
            obj[i] = 2;
        }
    }

    // zones: watershed of the (histogram-equalised) distance to the objects
    std::vector<float> ws_dist;
    {
        ImageU8 fgm(ex, ey);
        for (i64 i = 0; i < en; ++i) fgm[i] = obj[i] == 2;
        ws_dist = morph_distance(fgm, /*background=*/true, metric);
    }
    Image<std::int32_t> nw_conns;
    if (proximity) {
        nw_conns = Image<std::int32_t>(ex, ey);
        for (i64 i = 0; i < en; ++i) nw_conns[i] = static_cast<std::int32_t>(idl_round(ws_dist[static_cast<std::size_t>(i)] * 2.0f));
    }
    std::int32_t nregions = 0;
    Image<std::int32_t> ws = watershed(hist_equal(ws_dist, ex, ey), 8, nregions);
    for (i64 i = 0; i < en; ++i) {
        if (ws[i] == 0) obj[i] = kInfluenceLine;
    }

    // colour the objects (without their holes)
    Image<std::int32_t> nw_ids;
    std::vector<i64> ids_area;
    label_regions(obj, 2, 8, false, nw_ids, ids_area);
    InfluenceResult res;
    const std::int32_t max_id = static_cast<std::int32_t>(ids_area.size()) - 1;
    res.n_objects = max_id;
    res.object_area.assign(ids_area.size(), 0);
    for (i64 i = 0; i < en; ++i) {
        const std::int32_t z = hole[static_cast<std::size_t>(i)] ? 0 : nw_ids[i];
        ++res.object_area[static_cast<std::size_t>(z)];
        if (z > 0) {
            obj[i] = proximity ? kProximityObject : kInfluenceObjectColors[static_cast<std::size_t>(z % 12)];
        }
    }

    // cut back to the original size
    ImageU8 cls = crop(obj, kPad);
    Image<std::int32_t> ids = crop(nw_ids, kPad);
    ws = crop(ws, kPad);
    if (proximity) nw_conns = crop(nw_conns, kPad);
    const i64 n = cls.size();

    std::vector<u8> neglect(static_cast<std::size_t>(n), 0);
    for (i64 i = 0; i < n; ++i) {
        neglect[static_cast<std::size_t>(i)] = orig[i] == 2 && cls[i] <= 2;
        if (orig[i] == 1 && cls[i] == 2) {  // holes of the objects
            cls[i] = kInfluenceHole;
            ids[i] = -1;
        }
    }
    std::vector<u8> buffer_edge;
    if (!proximity && buffer > 0) {  // background buffer around the objects
        ImageU8 c(nx, ny);
        for (i64 i = 0; i < n; ++i) c[i] = cls[i] > 4;
        const std::vector<float> cd = morph_distance(c, /*background=*/true, metric);
        const float a1 = static_cast<float>(buffer) - 0.5f, a2 = static_cast<float>(buffer) + 0.5f;
        buffer_edge.assign(static_cast<std::size_t>(n), 0);
        for (i64 i = 0; i < n; ++i) {
            const float v = cd[static_cast<std::size_t>(i)];
            buffer_edge[static_cast<std::size_t>(i)] = v >= a1 && v < a2;
            if (v < a1 && v > 0.0f) {
                cls[i] = kInfluenceBuffer;
                ids[i] = -6;
            }
        }
    }
    for (i64 i = 0; i < n; ++i) {
        if (neglect[static_cast<std::size_t>(i)]) {
            cls[i] = kInfluenceNeglected;
            ids[i] = -2;
        }
        if (!buffer_edge.empty() && buffer_edge[static_cast<std::size_t>(i)]) {
            cls[i] = kInfluenceBufferEdge;
            ids[i] = -5;
        }
        if (orig[i] == 0) {
            cls[i] = kInfluenceMissing;
            ids[i] = -4;
        }
    }
    bool any_line = false;
    for (i64 i = 0; i < n; ++i) {
        if (ws[i] == 0) {
            cls[i] = kInfluenceLine;
            ids[i] = -3;
            any_line = true;
        }
    }

    if (proximity && any_line) {
        i64 proxmin = std::numeric_limits<i64>::max();
        for (i64 i = 0; i < n; ++i) {
            if (cls[i] == kInfluenceLine) proxmin = std::min<i64>(proxmin, nw_conns[i]);
        }
        const i64 markmax = std::max<i64>(buffer, proxmin);
        std::vector<i64> at_min;
        for (i64 i = 0; i < n; ++i) {
            if (cls[i] == kInfluenceLine && nw_conns[i] == proxmin) at_min.push_back(i);
        }
        for (i64 i = 0; i < n; ++i) {
            if (cls[i] == kInfluenceLine && nw_conns[i] > 0 && nw_conns[i] <= markmax) cls[i] = kProximityNear;
        }
        for (const i64 i : at_min) cls[i] = kProximityMin;
        res.prox_min = proxmin;
        res.prox_min_count = static_cast<i64>(at_min.size());

        res.cag = Image<std::int64_t>(nx, ny, 0);
        for (i64 i = 0; i < n; ++i) {
            if (ws[i] != 0) continue;
            const i64 x = i % nx, y = i / nx;
            std::vector<std::int32_t> p;
            for (i64 yy = std::max<i64>(y - 1, 0); yy <= std::min<i64>(y + 1, ny - 1); ++yy) {
                for (i64 xx = std::max<i64>(x - 1, 0); xx <= std::min<i64>(x + 1, nx - 1); ++xx) {
                    p.push_back(ws.at(xx, yy));
                }
            }
            std::sort(p.begin(), p.end());
            p.erase(std::unique(p.begin(), p.end()), p.end());
            p.erase(p.begin());  // the watershed line itself (0)
            const i64 nrp = static_cast<i64>(p.size());
            if (nrp <= 1) continue;  // only at the image boundary: not reported
            auto area_of = [&](std::int32_t z) -> i64 {
                return z >= 0 && z <= max_id ? res.object_area[static_cast<std::size_t>(z)] : 0;
            };
            double cag = 0.0;
            for (i64 a = 0; a < nrp - 1; ++a) {
                for (i64 b = a + 1; b < nrp; ++b) {
                    cag += static_cast<double>(area_of(p[static_cast<std::size_t>(a)]) *
                                               area_of(p[static_cast<std::size_t>(b)]));
                }
            }
            const i64 cag_r = static_cast<i64>(std::round(std::sqrt(2.0 * cag)));
            res.cag[i] = cag_r;
            i64 d = nw_conns[i];
            if (nrp > 2) {
                d = idl_round(static_cast<float>(d) / 2.0f * static_cast<float>(nrp));
                nw_conns[i] = static_cast<std::int32_t>(d);
                if (d <= markmax) cls[i] = kProximityNear;
                if (d == proxmin) cls[i] = kProximityMin;
            }
            ProximityLine line;
            line.x = x + 1;
            line.y = ny - y;
            line.nbr = nrp;
            line.clength = d;
            line.cag = cag_r;
            line.ids = p;
            for (const std::int32_t z : p) line.areas.push_back(area_of(z));
            res.lines.push_back(std::move(line));
        }
        res.cag = flip_rows(res.cag);
    }

    res.classes = flip_rows(cls);
    res.ids = flip_rows(ids);
    res.zones = flip_rows(ws);
    if (proximity) res.distance = flip_rows(nw_conns);
    return res;
}

}  // namespace gtb
