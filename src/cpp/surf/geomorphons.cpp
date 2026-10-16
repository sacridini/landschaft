// SPDX-License-Identifier: GPL-3.0-or-later
//
// Geomorphons (Jasiewicz & Stepinski 2013): the landform of every cell from the
// lines of sight in its 8 directions, as GRASS GIS's r.geomorphon computes it
// (8.4, GPL-2.0-or-later, ported). Along each direction, from the cell `skip + 1`
// steps away while the distance stays below the search distance, the highest
// (zenith) and lowest (nadir) angle of the terrain seen from the cell; the
// direction is higher (+1), lower (-1) or level (0) by the comparison of the
// two angles with the flatness threshold; the counts of + and - name one of 10
// forms, and the 8 signs, up to rotation and reflection, one of 498 ternary
// patterns.
//
// The DEM is read as float32 (GRASS's FCELL); heights, distances and angles in
// double as GRASS has them, with the mingw-w64 atan2 (x87 fpatan) of its Windows
// build and the cell coordinates computed as Rast_row_to_northing does.
//
// Corrections of r.geomorphon (the correct form only, see the Python docstring):
// lines of sight reach the DEM's last row (r.geomorphon's row buffer leaves it
// out for the cells of the bottom `search` rows); the binary positive and
// negative patterns set bit i for direction i (r.geomorphon adds i to a byte);
// the extended correction's second, shorter search starts next to the cell
// (r.geomorphon keeps the skip, though it sets the skip distance to 0); the
// shape of a geomorphon with no visible point in its first direction keeps its
// other points (r.geomorphon tests the first direction's distance for all).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "lsm/rmath.hpp"
#include "surf/surface.hpp"

namespace surf {

namespace {

constexpr int kDirs = 8;
// 3|2|1 / 4|0|8 / 5|6|7 (NE, N, NW, W, SW, S, SE, E)
constexpr int kNextR[kDirs] = {-1, -1, -1, 0, 1, 1, 1, 0};
constexpr int kNextC[kDirs] = {1, 0, -1, -1, -1, 0, 1, 1};
constexpr double kSins[kDirs] = {0.7071067812, 0, -0.7071067812, -1, -0.7071067812, 0, 0.7071067812, 1};
constexpr double kCoss[kDirs] = {0.7071067812, 1, 0.7071067812, 0, -0.7071067812, -1, -0.7071067812, 0};

enum Form : std::int8_t { kNull = 0, FL, PK, RI, SH, SP, SL, HL, FS, VL, PT, ER };

const Form kForms[9][9] = {
    /* minus \ plus: 0   1   2   3   4   5   6   7   8 */
    /* 0 */ {FL, FL, FL, FS, FS, VL, VL, VL, PT},
    /* 1 */ {FL, FL, FS, FS, FS, VL, VL, VL, ER},
    /* 2 */ {FL, SH, SL, SL, HL, HL, VL, ER, ER},
    /* 3 */ {SH, SH, SL, SL, SL, HL, ER, ER, ER},
    /* 4 */ {SH, SH, SP, SL, SL, ER, ER, ER, ER},
    /* 5 */ {RI, RI, SP, SP, ER, ER, ER, ER, ER},
    /* 6 */ {RI, RI, RI, ER, ER, ER, ER, ER, ER},
    /* 7 */ {RI, RI, ER, ER, ER, ER, ER, ER, ER},
    /* 8 */ {PK, ER, ER, ER, ER, ER, ER, ER, ER},
};

Form form_of(int num_minus, int num_plus) { return kForms[num_minus][num_plus]; }

// The smallest code of the pattern's rotations and reflections (r.geomorphon's
// ternary_rotate).
unsigned ternary_rotate(unsigned value) {
    unsigned char pattern[kDirs], rev[kDirs];
    unsigned code = 10000, rev_code = 10000;
    for (int i = 0; i < kDirs; ++i) {
        pattern[i] = static_cast<unsigned char>(value % 3);
        rev[7 - i] = static_cast<unsigned char>(value % 3);
        value /= 3;
    }
    for (int j = 0; j < kDirs; ++j) {
        unsigned power = 1, c = 0, rc = 0;
        for (int i = 0; i < kDirs; ++i) {
            const int k = (i - j) < 0 ? j - kDirs : j;
            c += pattern[i - k] * power;
            rc += rev[i - k] * power;
            power *= 3;
        }
        code = std::min(c, code);
        rev_code = std::min(rc, rev_code);
    }
    return std::min(code, rev_code);
}

const std::vector<std::int32_t>& ternary_table() {
    static const std::vector<std::int32_t> table = [] {
        std::vector<std::int32_t> t(6561);
        for (unsigned i = 0; i < 6561; ++i) t[i] = static_cast<std::int32_t>(ternary_rotate(i));
        return t;
    }();
    return table;
}

int rotate_binary(unsigned char binary) {
    unsigned char result = 255;
    for (int i = 0; i < kDirs; ++i) {
        const unsigned char test =
            i == 0 ? binary : static_cast<unsigned char>((binary << i) | (binary >> (kDirs - i)));
        result = std::min(result, test);
    }
    return result;
}

struct Pattern {
    int num_positives = 0, num_negatives = 0;
    unsigned char positives = 0, negatives = 0;
    unsigned char positive_sum = 0, negative_sum = 0;  // r.geomorphon's (direction numbers added)
    int pattern[kDirs] = {};
    float elevation[kDirs] = {};
    double distance[kDirs] = {};
};

int compare_multi(double nadir, double zenith, double nadir_thr, double zenith_thr, double nadir_dist,
                  double zenith_dist) {
    const bool nadir_over = nadir > nadir_thr, zenith_over = zenith > zenith_thr;
    if (!nadir_over && !zenith_over) return 0;
    if (!nadir_over && zenith_over) return 1;
    if (nadir_over && !zenith_over) return -1;
    if (nadir < zenith) return 1;
    if (nadir > zenith) return -1;
    if (nadir_dist < zenith_dist) return 1;
    if (nadir_dist > zenith_dist) return -1;
    return 1;
}

// The zenith (or nadir) of a line of sight, kept by its ratio height / distance: atan2
// grows with the ratio, so a step whose ratio is clearly greater (smaller) has the
// greater (smaller) angle and the angle is only computed at the end. Within the margin
// (1e-9, relative: far above the rounding of the ratios, far below the spacing of the
// angles for ratios up to 1e4) the two angles are computed and compared as GRASS does,
// so its choice among equal angles (the first) is kept.
struct Extreme {
    double height = 0, distance = 0, ratio = 0, angle = 0;
    bool set = false, known = false;

    double angle_of() {
        if (!known) {
            angle = lsm::r_atan2(height, distance);
            known = true;
        }
        return angle;
    }
    void take(double h, double d, double q) {
        height = h;
        distance = d;
        ratio = q;
        set = true;
        known = false;
    }
    // step (h, d, q) against this extreme: sign +1 keeps the greatest angle, -1 the least
    void offer(double h, double d, double q, int sign) {
        if (!set) {
            take(h, d, q);
            return;
        }
        const double margin = 1e-9 * std::max(std::fabs(q), std::fabs(ratio)) + 1e-300;
        const double diff = sign * (q - ratio);
        if (std::fabs(q) <= 1e4 && std::fabs(ratio) <= 1e4) {
            if (diff > margin) {
                take(h, d, q);
                return;
            }
            if (diff < -margin) return;
        }
        const double a = lsm::r_atan2(h, d);
        if (sign * (a - angle_of()) > 0) {
            take(h, d, q);
            angle = a;
            known = true;
        }
    }
};

// GRASS's G_distance in a projected location: hypot (msvcrt's _hypot, sqrt(x x + y y)
// for map coordinates, as lsm::ucrt_hypot's ordinary range).
inline double distance(double dx, double dy) { return std::sqrt(dx * dx + dy * dy); }

struct Context {
    const float* z;
    int ny, nx;
    const GeomorphonParams* p;
    double pi2;
    int pad;                       // rows / columns of coordinates kept beyond the DEM
    std::vector<double> north_of;  // Rast_row_to_northing(r + 0.5), r from -pad
    std::vector<double> east_of;   // Rast_col_to_easting(c + 0.5), c from -pad

    Context(const float* z_, int ny_, int nx_, const GeomorphonParams* p_, double pi2_, int pad_)
        : z(z_), ny(ny_), nx(nx_), p(p_), pi2(pi2_), pad(pad_) {
        north_of.resize(static_cast<std::size_t>(ny) + 2 * pad + 2);
        east_of.resize(static_cast<std::size_t>(nx) + 2 * pad + 2);
        for (std::size_t k = 0; k < north_of.size(); ++k)
            north_of[k] = p->north - (static_cast<int>(k) - pad + 0.5) * p->ns_res;
        for (std::size_t k = 0; k < east_of.size(); ++k)
            east_of[k] = p->west + (static_cast<int>(k) - pad + 0.5) * p->ew_res;
    }

    float at(int r, int c) const { return z[static_cast<i64>(r) * nx + c]; }
    // the centre of row r / column c (r, c may lie up to `pad` outside the DEM)
    double northing(int r) const { return north_of[static_cast<std::size_t>(r + pad)]; }
    double easting(int c) const { return east_of[static_cast<std::size_t>(c + pad)]; }

    // r.geomorphon's calc_pattern: the pattern of the cell, the number of directions seen.
    int pattern(Pattern& pt, int row, int col, int skip_cells, double search_distance, double flat_distance) const {
        const double cur_n = northing(row), cur_e = easting(col);
        const double center = at(row, col);
        pt = Pattern{};
        int size = 0;
        double zenith_height = 0, nadir_height = 0, zenith_distance = 0, nadir_distance = 0;
        for (int i = 0; i < kDirs; ++i) {
            int j = skip_cells + 1;
            double zenith_angle = -pi2, nadir_angle = pi2;
            if (row + j * kNextR[i] < 0 || row + j * kNextR[i] > ny - 1 || col + j * kNextC[i] < 0 ||
                col + j * kNextC[i] > nx - 1)
                continue;  // the cell is at the end of the DEM
            if (std::isnan(at(row + kNextR[i], col + kNextC[i]))) continue;  // no line of sight
            ++size;
            double dist = distance(cur_e - easting(col + j * kNextC[i]), cur_n - northing(row + j * kNextR[i]));
            Extreme zen, nad;
            while (dist < search_distance) {
                const int rr = row + j * kNextR[i], cc = col + j * kNextC[i];
                if (rr < 0 || rr > ny - 1 || cc < 0 || cc > nx - 1) break;
                const double height = static_cast<double>(at(rr, cc)) - center;
                const double q = height / dist;
                if (q == q) {  // a null cell further on: no angle (GRASS's atan2 of NaN changes nothing)
                    zen.offer(height, dist, q, 1);
                    nad.offer(height, dist, q, -1);
                }
                ++j;
                const int nr = row + j * kNextR[i], nc = col + j * kNextC[i];
                if (nr < -pad || nr > ny - 1 + pad || nc < -pad || nc > nx - 1 + pad) break;  // beyond any search
                dist = distance(cur_e - easting(nc), cur_n - northing(nr));
            }
            if (zen.set) {
                zenith_angle = zen.angle_of();
                zenith_height = zen.height;
                zenith_distance = zen.distance;
            }
            if (nad.set) {
                nadir_angle = nad.angle_of();
                nadir_height = nad.height;
                nadir_distance = nad.distance;
            }
            const double zenith_thr = (flat_distance > 0 && flat_distance < zenith_distance)
                                          ? lsm::r_atan2(p->flat_height, zenith_distance)
                                          : p->flat_threshold;
            const double nadir_thr = (flat_distance > 0 && flat_distance < nadir_distance)
                                         ? lsm::r_atan2(p->flat_height, nadir_distance)
                                         : p->flat_threshold;
            if (zenith_angle > zenith_thr) {
                pt.positives = static_cast<unsigned char>(pt.positives | (1u << i));
                pt.positive_sum = static_cast<unsigned char>(pt.positive_sum + i);
            }
            if (nadir_angle < -nadir_thr) {
                pt.negatives = static_cast<unsigned char>(pt.negatives | (1u << i));
                pt.negative_sum = static_cast<unsigned char>(pt.negative_sum + i);
            }
            if (p->comparison != 0) {
                pt.pattern[i] = compare_multi(std::fabs(nadir_angle), std::fabs(zenith_angle), nadir_thr, zenith_thr,
                                              p->comparison == 2 ? nadir_distance : 0,
                                              p->comparison == 2 ? zenith_distance : 0);
                if (pt.pattern[i] == 1) {
                    pt.elevation[i] = static_cast<float>(zenith_height);
                    pt.distance[i] = zenith_distance;
                    ++pt.num_positives;
                } else if (pt.pattern[i] == -1) {
                    pt.elevation[i] = static_cast<float>(nadir_height);
                    pt.distance[i] = nadir_distance;
                    ++pt.num_negatives;
                } else {
                    pt.distance[i] = search_distance;
                }
                continue;
            }
            if (std::fabs(zenith_angle) > zenith_thr || std::fabs(nadir_angle) > nadir_thr) {
                if (std::fabs(nadir_angle) < std::fabs(zenith_angle)) {
                    pt.pattern[i] = 1;
                    pt.elevation[i] = static_cast<float>(zenith_height);
                    pt.distance[i] = zenith_distance;
                    ++pt.num_positives;
                }
                if (std::fabs(nadir_angle) > std::fabs(zenith_angle)) {
                    pt.pattern[i] = -1;
                    pt.elevation[i] = static_cast<float>(nadir_height);
                    pt.distance[i] = nadir_distance;
                    ++pt.num_negatives;
                }
            } else {
                pt.distance[i] = search_distance;
            }
        }
        return size;
    }
};

float intensity(const Pattern& pt, int size) {
    float s = 0.f;
    for (int i = 0; i < kDirs; ++i) s -= pt.elevation[i];
    return s / static_cast<float>(size);
}

float exposition(const Pattern& pt) {
    float m = pt.elevation[0];
    for (int i = 1; i < kDirs; ++i)
        if (std::fabs(pt.elevation[i]) > std::fabs(m)) m = pt.elevation[i];
    return -m;
}

float range_of(const Pattern& pt) {
    float mx = pt.elevation[0], mn = pt.elevation[0];
    for (int i = 1; i < kDirs; ++i) {
        mx = std::max(pt.elevation[i], mx);
        mn = std::min(pt.elevation[i], mn);
    }
    return mx - mn;
}

float variance_of(const Pattern& pt, int size) {
    float s = 0.f, v = 0.f;
    for (int i = 0; i < kDirs; ++i) s += pt.elevation[i];
    s /= static_cast<float>(size);
    for (int i = 0; i < kDirs; ++i) v += (s - pt.elevation[i]) * (s - pt.elevation[i]);
    return v / static_cast<float>(size);
}

struct Shape {
    float azimuth, elongation, width, extend;
};

Shape shape_of(const Pattern& pt, int size, double pi2, double rad2deg, bool grass) {
    double x[kDirs], y[kDirs];
    for (int i = 0; i < kDirs; ++i) {
        if (pt.distance[grass ? 0 : i] > 0.) {
            x[i] = pt.distance[i] * kSins[i];
            y[i] = pt.distance[i] * kCoss[i];
        } else {
            x[i] = 0.;
            y[i] = 0.;
        }
    }
    double ax = 0, ay = 0, axy = 0, axx = 0;
    for (int i = 0; i < kDirs; ++i) {
        ay += y[i];
        ax += x[i];
        axx += x[i] * x[i];
        axy += x[i] * y[i];
    }
    const float fs = static_cast<float>(size);
    ay /= fs;
    ax /= fs;
    axy /= fs;
    axx /= fs;
    double result = (axy - ax * ay) / (axx - ax * ax);
    result = std::atan(result);
    Shape s{};
    s.azimuth = static_cast<float>((pi2 - result) * rad2deg);
    const double sine = lsm::r_sin(result), cosine = lsm::r_cos(result);
    double rxmin = x[0] * cosine - y[0] * sine, rxmax = rxmin;
    double rymin = x[0] * sine + y[0] * cosine, rymax = rymin;
    for (int i = 1; i < kDirs; ++i) {
        const double rx = x[i] * cosine - y[i] * sine, ry = x[i] * sine + y[i] * cosine;
        rxmin = std::min(rx, rxmin);
        rxmax = std::max(rx, rxmax);
        rymin = std::min(ry, rymin);
        rymax = std::max(ry, rymax);
    }
    const double rx = rxmax - rxmin, ry = rymax - rymin;
    s.elongation = rx > ry ? static_cast<float>(static_cast<float>(rx) / ry)
                           : static_cast<float>(static_cast<float>(ry) / rx);
    s.width = static_cast<float>(std::min(rx, ry));
    float area = 0;
    for (int i = 0, j = 1; i < kDirs; ++i, ++j) {
        j = j < kDirs ? j : 0;
        area = static_cast<float>(area + (x[i] * y[j] - x[j] * y[i]));
    }
    s.extend = static_cast<float>(std::fabs(area) / 2.);
    return s;
}

}  // namespace

Geomorphons geomorphons(const float* z, int ny, int nx, const GeomorphonParams& p) {
    if (ny <= 0 || nx <= 0) throw std::invalid_argument("empty DEM");
    if (p.search_cells < 1) throw std::invalid_argument("the search radius must cover at least 1 cell");
    if (p.skip_cells < 0 || p.skip_cells >= p.search_cells)
        throw std::invalid_argument("the skip radius must be at least 1 cell less than the search radius");
    const auto& table = ternary_table();
    const i64 n = static_cast<i64>(ny) * nx;
    Geomorphons g;
    const float fnan = std::numeric_limits<float>::quiet_NaN();
    const unsigned o = p.outputs;
    if (o & kGmForms) g.forms.assign(n, 0);
    if (o & kGmTernary) g.ternary.assign(n, -1);
    if (o & kGmPositive) g.positive.assign(n, -1);
    if (o & kGmNegative) g.negative.assign(n, -1);
    if (o & kGmIntensity) g.intensity.assign(n, fnan);
    if (o & kGmExposition) g.exposition.assign(n, fnan);
    if (o & kGmRange) g.range.assign(n, fnan);
    if (o & kGmVariance) g.variance.assign(n, fnan);
    if (o & kGmElongation) g.elongation.assign(n, fnan);
    if (o & kGmAzimuth) g.azimuth.assign(n, fnan);
    if (o & kGmExtend) g.extend.assign(n, fnan);
    if (o & kGmWidth) g.width.assign(n, fnan);
    const double pi = 4 * std::atan(1.0);
    // a line of sight stops at the DEM's edge, so the coordinates of one cell beyond it suffice
    Context ctx(z, ny, nx, &p, 2 * std::atan(1.0), 1);
    const double rad2deg = 180 / pi;
    const double octagon = 4 * (p.search_distance * p.search_distance) * lsm::r_sin(45. / (180 / pi));
    const bool shape = (o & (kGmElongation | kGmAzimuth | kGmExtend | kGmWidth)) != 0;
    const int skip = p.skip_cells;
    const bool grass = (o & kGmGrassErrors) != 0;

#pragma omp parallel for schedule(dynamic, 4) if (n > 4096)
    for (int row = 0; row < ny; ++row) {
        Pattern pt, small;
        for (int col = 0; col < nx; ++col) {
            const i64 k = static_cast<i64>(row) * nx + col;
            // on the borders forms are usually unnatural
            if (row < skip + 1 || row > ny - (skip + 2) || col < skip + 1 || col > nx - (skip + 2) ||
                std::isnan(ctx.at(row, col)))
                continue;
            const int size = ctx.pattern(pt, row, col, skip, p.search_distance, p.flat_distance);
            Form form = form_of(pt.num_negatives, pt.num_positives);
            if (p.extended && p.search_distance > 10 * p.max_resolution &&
                (form == SH || form == FS || form == PK || form == RI)) {
                const double s2 = (p.search_distance / 2. < 4 * p.max_resolution) ? 4 * p.max_resolution
                                                                                   : p.search_distance / 2.;
                ctx.pattern(small, row, col, grass ? skip : 0, s2, 0.0);
                const Form small_form = form_of(small.num_negatives, small.num_positives);
                if (form == SH || form == FS) form = (small_form == FL) ? FL : form;
                else form = small_form;
            }
            if (o & kGmForms) g.forms[k] = form;
            if (o & kGmTernary) {
                unsigned code = 0, power = 1;
                for (int i = 0; i < kDirs; ++i, power *= 3) code += static_cast<unsigned>(pt.pattern[i] + 1) * power;
                g.ternary[k] = table[code];
            }
            if (o & kGmPositive)
                g.positive[k] = static_cast<std::int16_t>(rotate_binary(grass ? pt.positive_sum : pt.positives));
            if (o & kGmNegative)
                g.negative[k] = static_cast<std::int16_t>(rotate_binary(grass ? pt.negative_sum : pt.negatives));
            if (o & kGmIntensity) g.intensity[k] = intensity(pt, size);
            if (o & kGmExposition) g.exposition[k] = exposition(pt);
            if (o & kGmRange) g.range[k] = range_of(pt);
            if (o & kGmVariance) g.variance[k] = variance_of(pt, size);
            if (shape) {
                const Shape s = shape_of(pt, size, ctx.pi2, rad2deg, grass);
                if (o & kGmAzimuth) g.azimuth[k] = s.azimuth;
                if (o & kGmElongation) g.elongation[k] = s.elongation;
                if (o & kGmWidth) g.width[k] = s.width;
                if (o & kGmExtend) g.extend[k] = static_cast<float>(s.extend / octagon);
            }
        }
    }
    return g;
}

}  // namespace surf
