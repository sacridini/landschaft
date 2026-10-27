// SPDX-License-Identifier: GPL-3.0-or-later
//
// The routing of InVEST's seasonal water yield model (natcap.invest 3.20, Apache-2.0,
// seasonal_water_yield/swy.h), ported cell for cell: the local recharge, from the ridges
// down, and the baseflow, from the outlets up, over pygeoprocessing's flow directions
// (ManagedRaster.h's neighbour iterators). The arithmetic is InVEST's, in its types (float
// shares and parameters, double sums); the values stay in double, as in InVEST's block
// cache, until Python stores them in float32.
//
// Fixed here (and in the patched InVEST of tools/make_reference_invest_swy.py): L_sum_avail
// sums the upslope terms by their shares of the flow (eq. 103; InVEST takes their mean
// weighted by the raw MFD weights); a cell without a flow direction is never visited (a D8
// direction can point into a hole of the DEM, whose own direction InVEST then reads past
// its offset tables); and the flow off the raster ends the baseflow path as the flow into a
// hole does (eq. 108; InVEST drops it, giving B_sum = 0 at the raster's edge).

#include "serv/swy.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace serv {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// pygeoprocessing's neighbours: 0 E, 1 NE, 2 N, 3 NW, 4 W, 5 SW, 6 S, 7 SE
constexpr int ROW[8] = {0, -1, -1, -1, 0, 1, 1, 1};
constexpr int COL[8] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int REVERSE[8] = {4, 5, 6, 7, 0, 1, 2, 3};

struct Neighbor {
    int x, y;
    float proportion;
};

struct MFD {
    const std::int32_t* dir;
    int ny, nx;
    bool inside(i64 x, i64 y) const { return x >= 0 && y >= 0 && x < nx && y < ny; }
    bool missing(i64 i) const { return dir[i] == 0; }
    int val(i64 x, i64 y) const { return dir[y * nx + x]; }

    bool high_point(int x, int y) const {
        for (int k = 0; k < 8; ++k) {
            const i64 xj = x + COL[k], yj = y + ROW[k];
            if (!inside(xj, yj)) continue;
            if (0xF & (val(xj, yj) >> (4 * REVERSE[k]))) return false;
        }
        return true;
    }
    // UpslopeNeighbors: the share of each upslope cell's flow that comes here (float)
    void upslope(int x, int y, std::vector<Neighbor>& out) const {
        out.clear();
        for (int k = 0; k < 8; ++k) {
            const i64 xj = x + COL[k], yj = y + ROW[k];
            if (!inside(xj, yj)) continue;
            const int fj = val(xj, yj);
            const int fji = 0xF & (fj >> (4 * REVERSE[k]));
            if (!fji) continue;
            long sum = 0;
            for (int m = 0; m < 8; ++m) sum += (fj >> (m * 4)) & 0xF;
            out.push_back({static_cast<int>(xj), static_cast<int>(yj),
                           static_cast<float>(fji) / static_cast<float>(sum)});
        }
    }
    // DownslopeNeighbors (inside the raster) or DownslopeNeighborsNoSkip: the raw weights
    void downslope(int x, int y, bool skip_outside, std::vector<Neighbor>& out) const {
        out.clear();
        const int v = val(x, y);
        for (int k = 0; k < 8; ++k) {
            const i64 xj = x + COL[k], yj = y + ROW[k];
            if (skip_outside && !inside(xj, yj)) continue;
            const int flow = (v >> (k * 4)) & 0xF;
            if (flow) out.push_back({static_cast<int>(xj), static_cast<int>(yj), static_cast<float>(flow)});
        }
    }
};

struct D8 {
    const std::uint8_t* dir;
    int ny, nx;
    bool inside(i64 x, i64 y) const { return x >= 0 && y >= 0 && x < nx && y < ny; }
    bool missing(i64 i) const { return dir[i] == 128; }
    int val(i64 x, i64 y) const { return dir[y * nx + x]; }

    bool high_point(int x, int y) const {
        for (int k = 0; k < 8; ++k) {
            const i64 xj = x + COL[k], yj = y + ROW[k];
            if (!inside(xj, yj)) continue;
            if (val(xj, yj) == REVERSE[k]) return false;
        }
        return true;
    }
    void upslope(int x, int y, std::vector<Neighbor>& out) const {
        out.clear();
        for (int k = 0; k < 8; ++k) {
            const i64 xj = x + COL[k], yj = y + ROW[k];
            if (!inside(xj, yj)) continue;
            if (val(xj, yj) == REVERSE[k]) out.push_back({static_cast<int>(xj), static_cast<int>(yj), 1.0f});
        }
    }
    void downslope(int x, int y, bool skip_outside, std::vector<Neighbor>& out) const {
        out.clear();
        const int v = val(x, y);
        if (v < 0 || v > 7) return;
        const i64 xj = x + COL[v], yj = y + ROW[v];
        if (skip_outside && !inside(xj, yj)) return;
        out.push_back({static_cast<int>(xj), static_cast<int>(yj), 1.0f});
    }
};

template <class T, class P, class E>
void local_recharge(const T& fl, int block, const SwyMonths<P, E>& in, float beta, float gamma, double* l,
                    double* l_avail, double* lsa, double* aet, double* p_annual) {
    const int ny = fl.ny, nx = fl.nx;
    const i64 n = static_cast<i64>(ny) * nx;
    std::fill(l, l + n, kNaN);
    std::fill(l_avail, l_avail + n, kNaN);
    std::fill(lsa, lsa + n, kNaN);
    std::fill(aet, aet + n, kNaN);
    std::fill(p_annual, p_annual + n, kNaN);
    float ab[12];
    for (int m = 0; m < 12; ++m) ab[m] = in.alpha[m] * beta;  // alpha_values[m] * beta_i, in float
    std::deque<i64> queue;
    std::vector<Neighbor> up, dn;
    const int nbx = (nx + block - 1) / block, nby = (ny + block - 1) / block;
    for (int by = 0; by < nby; ++by) {
        const int y0 = by * block, y1 = std::min(ny, y0 + block);
        for (int bx = 0; bx < nbx; ++bx) {
            const int x0 = bx * block, x1 = std::min(nx, x0 + block);
            for (int ys = y0; ys < y1; ++ys)
                for (int xs = x0; xs < x1; ++xs) {
                    const i64 seed = static_cast<i64>(ys) * nx + xs;
                    if (fl.missing(seed)) continue;
                    if (fl.high_point(xs, ys)) queue.push_back(seed);
                    while (!queue.empty()) {
                        const i64 i = queue.front();
                        queue.pop_front();
                        if (fl.missing(i) || !std::isnan(lsa[i])) continue;
                        const int xi = static_cast<int>(i % nx), yi = static_cast<int>(i / nx);
                        // eq. 103
                        bool defined = true;
                        double s = 0.0;
                        fl.upslope(xi, yi, up);
                        for (const Neighbor& nb : up) {
                            const i64 j = static_cast<i64>(nb.y) * nx + nb.x;
                            if (std::isnan(lsa[j])) {
                                defined = false;
                                break;
                            }
                            s += (lsa[j] + l_avail[j]) * nb.proportion;
                        }
                        if (!defined) continue;  // reached again from its last upslope cell
                        lsa[i] = s;
                        double aet_i = 0, p_i = 0, qf_i = 0;
                        for (int m = 0; m < 12; ++m) {
                            const i64 k = m * n + i;
                            double p_m = static_cast<double>(in.precip[k]);
                            if (!std::isnan(p_m)) p_i += p_m;
                            else p_m = 0;
                            double qf_m = static_cast<double>(in.quickflow[k]);
                            if (!std::isnan(qf_m)) qf_i += qf_m;
                            else qf_m = 0;
                            const double kc_m = static_cast<double>(in.kc[k]), et0_m = static_cast<double>(in.et0[k]);
                            double pet_m = 0;
                            if (!(std::isnan(kc_m) || std::isnan(et0_m))) pet_m = kc_m * et0_m;  // eq. 102
                            aet_i += std::min(pet_m, p_m - qf_m + ab[m] * s);                   // eq. 101
                        }
                        const double l_i = p_i - qf_i - aet_i;
                        p_annual[i] = p_i;
                        aet[i] = aet_i;
                        l[i] = l_i;
                        l_avail[i] = std::min(gamma * l_i, l_i);  // eq. 104
                        fl.downslope(xi, yi, true, dn);
                        for (const Neighbor& nb : dn) queue.push_back(static_cast<i64>(nb.y) * nx + nb.x);
                    }
                }
        }
    }
}

template <class T>
void baseflow(const T& fl, int block, const double* l, const double* l_avail, const double* l_sum,
              const std::uint8_t* stream, double* b, double* b_sum) {
    const int ny = fl.ny, nx = fl.nx;
    const i64 n = static_cast<i64>(ny) * nx;
    std::fill(b, b + n, kNaN);
    std::fill(b_sum, b_sum + n, kNaN);
    std::vector<i64> stack;
    std::vector<Neighbor> up, dn;
    const int nbx = (nx + block - 1) / block, nby = (ny + block - 1) / block;
    for (int by = 0; by < nby; ++by) {
        const int y0 = by * block, y1 = std::min(ny, y0 + block);
        for (int bx = 0; bx < nbx; ++bx) {
            const int x0 = bx * block, x1 = std::min(nx, x0 + block);
            for (int ys = y0; ys < y1; ++ys)
                for (int xs = x0; xs < x1; ++xs) {
                    const i64 root = static_cast<i64>(ys) * nx + xs;
                    if (fl.missing(root)) continue;
                    // an outlet: no downslope cell inside the raster with a stream value
                    bool outlet = true;
                    fl.downslope(xs, ys, true, dn);
                    for (const Neighbor& nb : dn)
                        if (stream[static_cast<i64>(nb.y) * nx + nb.x] != 255) {
                            outlet = false;
                            break;
                        }
                    if (!outlet) continue;
                    stack.push_back(root);
                    while (!stack.empty()) {
                        const i64 i = stack.back();
                        stack.pop_back();
                        if (!std::isnan(b_sum[i])) continue;
                        const int xi = static_cast<int>(i % nx), yi = static_cast<int>(i / nx);
                        double bs = 0;
                        bool defined = true;
                        long dir_sum = 0;
                        fl.downslope(xi, yi, false, dn);
                        for (const Neighbor& nb : dn) {
                            dir_sum += static_cast<long>(nb.proportion);
                            if (!fl.inside(nb.x, nb.y)) {
                                bs += nb.proportion;  // off the raster: the path ends, as into a hole
                                continue;
                            }
                            const i64 j = static_cast<i64>(nb.y) * nx + nb.x;
                            if (stream[j]) {  // a stream, or no stream value (255)
                                bs += nb.proportion;
                                continue;
                            }
                            const double bs_j = b_sum[j];
                            if (std::isnan(bs_j)) {
                                defined = false;
                                break;
                            }
                            const double l_j = l[j], la_j = l_avail[j], ls_j = l_sum[j];
                            if (ls_j != 0 && (ls_j - l_j) != 0)
                                bs += nb.proportion * ((1 - la_j / ls_j) * (bs_j / (ls_j - l_j)));  // eq. 107
                            else
                                bs += nb.proportion;
                        }
                        if (!defined) continue;
                        const double l_i = l[i], ls_i = l_sum[i];
                        if (dir_sum > 0) bs = ls_i * bs / dir_sum;
                        const double b_i = ls_i != 0 ? std::max(bs * l_i / ls_i, 0.0) : 0.0;  // eq. 110
                        b[i] = b_i;
                        b_sum[i] = bs;
                        fl.upslope(xi, yi, up);
                        for (const Neighbor& nb : up) stack.push_back(static_cast<i64>(nb.y) * nx + nb.x);
                    }
                }
        }
    }
}

}  // namespace

template <class P, class E>
void swy_local_recharge_mfd(const std::int32_t* dir, int ny, int nx, int block, const SwyMonths<P, E>& in,
                            float beta, float gamma, double* l, double* l_avail, double* l_sum_avail, double* aet,
                            double* p_annual) {
    local_recharge(MFD{dir, ny, nx}, block, in, beta, gamma, l, l_avail, l_sum_avail, aet, p_annual);
}

template <class P, class E>
void swy_local_recharge_d8(const std::uint8_t* dir, int ny, int nx, int block, const SwyMonths<P, E>& in,
                           float beta, float gamma, double* l, double* l_avail, double* l_sum_avail, double* aet,
                           double* p_annual) {
    local_recharge(D8{dir, ny, nx}, block, in, beta, gamma, l, l_avail, l_sum_avail, aet, p_annual);
}

#define SWY_INSTANTIATE(P, E)                                                                                        \
    template void swy_local_recharge_mfd<P, E>(const std::int32_t*, int, int, int, const SwyMonths<P, E>&, float,   \
                                               float, double*, double*, double*, double*, double*);                \
    template void swy_local_recharge_d8<P, E>(const std::uint8_t*, int, int, int, const SwyMonths<P, E>&, float,    \
                                              float, double*, double*, double*, double*, double*);
SWY_INSTANTIATE(float, float)
SWY_INSTANTIATE(float, double)
SWY_INSTANTIATE(double, float)
SWY_INSTANTIATE(double, double)
#undef SWY_INSTANTIATE

void swy_baseflow_mfd(const std::int32_t* dir, int ny, int nx, int block, const double* l, const double* l_avail,
                      const double* l_sum, const std::uint8_t* stream, double* b, double* b_sum) {
    baseflow(MFD{dir, ny, nx}, block, l, l_avail, l_sum, stream, b, b_sum);
}

void swy_baseflow_d8(const std::uint8_t* dir, int ny, int nx, int block, const double* l, const double* l_avail,
                     const double* l_sum, const std::uint8_t* stream, double* b, double* b_sum) {
    baseflow(D8{dir, ny, nx}, block, l, l_avail, l_sum, stream, b, b_sum);
}

}  // namespace serv
