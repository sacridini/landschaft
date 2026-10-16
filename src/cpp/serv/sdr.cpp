// SPDX-License-Identifier: GPL-3.0-or-later
//
// The parts of InVEST's sediment delivery ratio model (natcap.invest 3.20, Apache-2.0) that
// are not raster algebra: pygeoprocessing's percent slope (geoprocessing_core.calculate_slope)
// and the sediment deposition and flux (sdr/sediment_deposition.h), ported cell for cell.

#include "serv/sdr.hpp"

#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace serv {

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// pygeoprocessing's neighbours: 0 E, 1 NE, 2 N, 3 NW, 4 W, 5 SW, 6 S, 7 SE
constexpr int ROW[8] = {0, -1, -1, -1, 0, 1, 1, 1};
constexpr int COL[8] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int REVERSE[8] = {4, 5, 6, 7, 0, 1, 2, 3};

// ManagedRaster.h's is_close
inline bool is_close(double x, double y) {
    if (std::isnan(x) && std::isnan(y)) return true;
    return std::abs(x - y) <= (std::pow(10.0, -8) + std::pow(10.0, -5) * std::abs(y));
}

}  // namespace

void pgp_slope(const double* z, int ny, int nx, double xcell, double ycell, float* out, int threads) {
    const double r2 = std::pow(2.0, 0.5);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int row = 0; row < ny; ++row) {
        auto at = [&](int r, int c) -> double {
            if (r < 0 || c < 0 || r >= ny || c >= nx) return std::numeric_limits<double>::quiet_NaN();
            return z[static_cast<i64>(r) * nx + c];
        };
        for (int col = 0; col < nx; ++col) {
            const double e = at(row, col);
            float* o = out + static_cast<i64>(row) * nx + col;
            if (std::isnan(e)) {
                *o = kNaN;
                continue;
            }
            const double a = at(row - 1, col - 1), b = at(row - 1, col), c = at(row - 1, col + 1);
            const double d = at(row, col - 1), f = at(row, col + 1);
            const double g = at(row + 1, col - 1), h = at(row + 1, col), i = at(row + 1, col + 1);
            const bool av = !std::isnan(a), bv = !std::isnan(b), cv = !std::isnan(c), dv = !std::isnan(d);
            const bool fv = !std::isnan(f), gv = !std::isnan(g), hv = !std::isnan(h), iv = !std::isnan(i);
            double dzdx = 0.0, dzdy = 0.0;
            int xd = 0, yd = 0;
            // a - c
            if (av && cv) { dzdx += a - c; xd += 2; }
            else if (av && bv) { dzdx += a - b; xd += 1; }
            else if (bv && cv) { dzdx += b - c; xd += 1; }
            else if (av) { dzdx += (a - e) * r2; xd += 1; }
            else if (cv) { dzdx += (e - c) * r2; xd += 1; }
            // d - f
            if (dv && fv) { dzdx += 2 * (d - f); xd += 4; }
            else if (dv) { dzdx += 2 * (d - e); xd += 2; }
            else if (fv) { dzdx += 2 * (e - f); xd += 2; }
            // g - i
            if (gv && iv) { dzdx += g - i; xd += 2; }
            else if (gv && hv) { dzdx += g - h; xd += 1; }
            else if (hv && iv) { dzdx += h - i; xd += 1; }
            else if (gv) { dzdx += (g - e) * r2; xd += 1; }
            else if (iv) { dzdx += (e - i) * r2; xd += 1; }
            // a - g
            if (av && gv) { dzdy += a - g; yd += 2; }
            else if (av && dv) { dzdy += a - d; yd += 1; }
            else if (dv && gv) { dzdy += d - g; yd += 1; }
            else if (av) { dzdy += (a - e) * r2; yd += 1; }
            else if (gv) { dzdy += (e - g) * r2; yd += 1; }
            // b - h
            if (bv && hv) { dzdy += 2 * (b - h); yd += 4; }
            else if (bv) { dzdy += 2 * (b - e); yd += 2; }
            else if (hv) { dzdy += 2 * (e - h); yd += 2; }
            // c - i
            if (cv && iv) { dzdy += c - i; yd += 2; }
            else if (cv && fv) { dzdy += c - f; yd += 1; }
            else if (fv && iv) { dzdy += f - i; yd += 1; }
            else if (cv) { dzdy += (c - e) * r2; yd += 1; }
            else if (iv) { dzdy += (e - i) * r2; yd += 1; }
            const double gx = xd != 0 ? dzdx / (xd * xcell) : 0.0;
            const double gy = yd != 0 ? dzdy / (yd * ycell) : 0.0;
            *o = static_cast<float>(100 * std::sqrt(gx * gx + gy * gy));
        }
    }
}

namespace {

// The flow directions of the deposition: pygeoprocessing's MFD (eight 4-bit weights,
// 0 missing) or D8 (0..7, 128 missing).
struct MFD {
    const std::int32_t* dir;
    static constexpr double nodata = 0;
    int val(i64 i) const { return dir[i]; }
    bool missing(i64 i) const { return static_cast<double>(dir[i]) == nodata; }
};

struct D8 {
    const std::uint8_t* dir;
    static constexpr double nodata = 128;
    int val(i64 i) const { return dir[i]; }
    bool missing(i64 i) const { return static_cast<double>(dir[i]) == nodata; }
};

struct Neighbor {
    int direction;
    int x, y;
    float proportion;
};

template <class T>
struct Flow {
    T d;
    int ny, nx;

    bool inside(i64 x, i64 y) const { return x >= 0 && y >= 0 && x < nx && y < ny; }

    bool high_point(int x, int y) const;
    void upslope(int x, int y, std::vector<Neighbor>& out) const;
    void downslope(int x, int y, std::vector<Neighbor>& out) const;
};

template <>
bool Flow<MFD>::high_point(int x, int y) const {
    for (int k = 0; k < 8; ++k) {
        const i64 xj = x + COL[k], yj = y + ROW[k];
        if (!inside(xj, yj)) continue;
        const int fj = d.val(yj * nx + xj);
        if (0xF & (fj >> (4 * REVERSE[k]))) return false;
    }
    return true;
}

template <>
bool Flow<D8>::high_point(int x, int y) const {
    for (int k = 0; k < 8; ++k) {
        const i64 xj = x + COL[k], yj = y + ROW[k];
        if (!inside(xj, yj)) continue;
        if (d.val(yj * nx + xj) == REVERSE[k]) return false;
    }
    return true;
}

template <>
void Flow<MFD>::upslope(int x, int y, std::vector<Neighbor>& out) const {
    out.clear();
    for (int k = 0; k < 8; ++k) {
        const i64 xj = x + COL[k], yj = y + ROW[k];
        if (!inside(xj, yj)) continue;
        const int fj = d.val(yj * nx + xj);
        const int fji = 0xF & (fj >> (4 * REVERSE[k]));
        if (!fji) continue;
        long sum = 0;
        for (int m = 0; m < 8; ++m) sum += (fj >> (m * 4)) & 0xF;
        out.push_back({k, static_cast<int>(xj), static_cast<int>(yj),
                       static_cast<float>(fji) / static_cast<float>(sum)});
    }
}

template <>
void Flow<D8>::upslope(int x, int y, std::vector<Neighbor>& out) const {
    out.clear();
    for (int k = 0; k < 8; ++k) {
        const i64 xj = x + COL[k], yj = y + ROW[k];
        if (!inside(xj, yj)) continue;
        if (d.val(yj * nx + xj) == REVERSE[k]) out.push_back({k, static_cast<int>(xj), static_cast<int>(yj), 1.0f});
    }
}

template <>
void Flow<MFD>::downslope(int x, int y, std::vector<Neighbor>& out) const {
    out.clear();
    const int v = d.val(static_cast<i64>(y) * nx + x);
    for (int k = 0; k < 8; ++k) {
        const i64 xj = x + COL[k], yj = y + ROW[k];
        if (!inside(xj, yj)) continue;
        const int flow = (v >> (k * 4)) & 0xF;
        if (flow) out.push_back({k, static_cast<int>(xj), static_cast<int>(yj), static_cast<float>(flow)});
    }
}

template <>
void Flow<D8>::downslope(int x, int y, std::vector<Neighbor>& out) const {
    out.clear();
    const int v = d.val(static_cast<i64>(y) * nx + x);
    if (v < 0 || v > 7) return;  // missing: InVEST would index past its offset tables
    const i64 xj = x + COL[v], yj = y + ROW[v];
    if (!inside(xj, yj)) return;
    out.push_back({v, static_cast<int>(xj), static_cast<int>(yj), 1.0f});
}

template <class T>
void deposition(const Flow<T>& fl, const float* e_prime, const float* sdr, int block, float* dep_out,
                float* flux_out) {
    const int ny = fl.ny, nx = fl.nx;
    const i64 n = static_cast<i64>(ny) * nx;
    constexpr double target_nodata = -1;
    std::vector<double> dep(static_cast<std::size_t>(n), target_nodata), flux(static_cast<std::size_t>(n), target_nodata);
    std::vector<i64> stack;
    std::vector<Neighbor> up, dn, up2;
    auto sdr_at = [&](i64 i) { return std::isnan(sdr[i]) ? target_nodata : static_cast<double>(sdr[i]); };
    const int nbx = (nx + block - 1) / block, nby = (ny + block - 1) / block;
    for (int by = 0; by < nby; ++by) {
        const int y0 = by * block, y1 = std::min(ny, y0 + block);
        for (int bx = 0; bx < nbx; ++bx) {
            const int x0 = bx * block, x1 = std::min(nx, x0 + block);
            for (int ys = y0; ys < y1; ++ys)
                for (int xs = x0; xs < x1; ++xs) {
                    const i64 seed = static_cast<i64>(ys) * nx + xs;
                    if (fl.d.missing(seed)) continue;
                    if (fl.high_point(xs, ys) && is_close(dep[static_cast<std::size_t>(seed)], target_nodata))
                        stack.push_back(seed);
                    while (!stack.empty()) {
                        const i64 flat = stack.back();
                        stack.pop_back();
                        const int gr = static_cast<int>(flat / nx), gc = static_cast<int>(flat % nx);
                        double f_sum = 0;
                        fl.upslope(gc, gr, up);
                        for (const Neighbor& nb : up) {
                            const double fj = flux[static_cast<std::size_t>(static_cast<i64>(nb.y) * nx + nb.x)];
                            if (is_close(fj, target_nodata)) continue;
                            f_sum += nb.proportion * fj;
                        }
                        double dn_sum = 0;
                        long dir_sum = 0;
                        fl.downslope(gc, gr, dn);
                        for (const Neighbor& nb : dn) {
                            dir_sum += static_cast<long>(nb.proportion);
                            const i64 j = static_cast<i64>(nb.y) * nx + nb.x;
                            double sdr_j = sdr_at(j);
                            if (is_close(sdr_j, target_nodata)) continue;
                            if (sdr_j == 0) sdr_j = 1;  // a stream: the last step that retains sediment
                            dn_sum += sdr_j * nb.proportion;
                            bool ready = true;
                            fl.upslope(nb.x, nb.y, up2);
                            for (const Neighbor& nn : up2) {
                                if (REVERSE[nn.direction] == nb.direction) continue;  // this cell
                                if (is_close(dep[static_cast<std::size_t>(static_cast<i64>(nn.y) * nx + nn.x)],
                                             target_nodata)) {
                                    ready = false;
                                    break;
                                }
                            }
                            if (ready) stack.push_back(j);
                        }
                        const double sdr_i = sdr_at(flat);
                        if (is_close(sdr_i, target_nodata)) continue;
                        if (std::isnan(e_prime[flat])) continue;
                        const double e_i = e_prime[flat];
                        if (dir_sum) dn_sum /= dir_sum;
                        if (dn_sum < sdr_i) dn_sum = sdr_i;
                        double dr = sdr_i == 1 ? 1.0 : (dn_sum - sdr_i) / (1 - sdr_i);
                        double t = dr * f_sum;
                        double f = (1 - dr) * f_sum + e_i;
                        if (dr < 0) dr = 0;
                        if (t < 0) t = 0;
                        if (f < 0) f = 0;
                        dep[static_cast<std::size_t>(flat)] = t;
                        flux[static_cast<std::size_t>(flat)] = f;
                    }
                }
        }
    }
    for (i64 i = 0; i < n; ++i) {
        const double t = dep[static_cast<std::size_t>(i)], f = flux[static_cast<std::size_t>(i)];
        dep_out[i] = t == target_nodata ? kNaN : static_cast<float>(t);
        flux_out[i] = f == target_nodata ? kNaN : static_cast<float>(f);
    }
}

}  // namespace

void sediment_deposition_mfd(const std::int32_t* dir, const float* e_prime, const float* sdr, int ny, int nx, int block,
                             float* dep, float* flux) {
    deposition(Flow<MFD>{MFD{dir}, ny, nx}, e_prime, sdr, block, dep, flux);
}

void sediment_deposition_d8(const std::uint8_t* dir, const float* e_prime, const float* sdr, int ny, int nx, int block,
                            float* dep, float* flux) {
    deposition(Flow<D8>{D8{dir}, ny, nx}, e_prime, sdr, block, dep, flux);
}

}  // namespace serv
