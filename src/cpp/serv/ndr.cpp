// SPDX-License-Identifier: GPL-3.0-or-later
//
// The part of InVEST's nutrient delivery ratio model (natcap.invest 3.20, Apache-2.0) that is
// not raster algebra: the effective retention down the flow to a stream
// (ndr/ndr_core.pyx and ndr/retention.h), ported cell for cell.

#include "serv/ndr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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

struct Step {
    int direction;
    i64 x, y;
    float proportion;
};

// pygeoprocessing's packed MFD directions (eight 4-bit weights, 0 missing)
struct MFD {
    const std::int32_t* dir;
    int ny, nx;
    static constexpr double nodata = 0;

    int val(i64 i) const { return dir[i]; }
    bool inside(i64 x, i64 y) const { return x >= 0 && y >= 0 && x < nx && y < ny; }
    // ndr_core's _mfd_to_flow_dir_op: a bit for every direction with flow
    int outflow_bits(i64 i) const {
        int out = 0;
        for (int k = 0; k < 8; ++k)
            if (((dir[i] >> (k * 4)) & 0xF) > 0) out |= 1 << k;
        return out;
    }
    // DownslopeNeighborsNoSkip: the cells off the map are kept
    void downslope(i64 x, i64 y, std::vector<Step>& out) const {
        out.clear();
        const int v = dir[y * nx + x];
        for (int k = 0; k < 8; ++k) {
            const int flow = (v >> (k * 4)) & 0xF;
            if (flow) out.push_back({k, x + COL[k], y + ROW[k], static_cast<float>(flow)});
        }
    }
    void upslope(i64 x, i64 y, std::vector<Step>& out) const {
        out.clear();
        for (int k = 0; k < 8; ++k) {
            const i64 xj = x + COL[k], yj = y + ROW[k];
            if (!inside(xj, yj)) continue;
            if (0xF & (dir[yj * nx + xj] >> (4 * REVERSE[k]))) out.push_back({k, xj, yj, 0.0f});
        }
    }
};

// pygeoprocessing's D8 (0..7, 128 missing)
struct D8 {
    const std::uint8_t* dir;
    int ny, nx;
    static constexpr double nodata = 128;

    int val(i64 i) const { return dir[i]; }
    bool inside(i64 x, i64 y) const { return x >= 0 && y >= 0 && x < nx && y < ny; }
    int outflow_bits(i64 i) const { return dir[i] < 8 ? 1 << dir[i] : 0; }
    void downslope(i64 x, i64 y, std::vector<Step>& out) const {
        out.clear();
        const int v = dir[y * nx + x];
        if (v > 7) return;
        out.push_back({v, x + COL[v], y + ROW[v], 1.0f});
    }
    void upslope(i64 x, i64 y, std::vector<Step>& out) const {
        out.clear();
        for (int k = 0; k < 8; ++k) {
            const i64 xj = x + COL[k], yj = y + ROW[k];
            if (!inside(xj, yj)) continue;
            if (dir[yj * nx + xj] == REVERSE[k]) out.push_back({k, xj, yj, 1.0f});
        }
    }
};

template <class T>
void retention(const T& fl, const std::uint8_t* stream, const float* eff, const float* crit, double cell_size,
               int block, float* out) {
    const i64 ny = fl.ny, nx = fl.nx, n = ny * nx;
    constexpr double retention_nodata = -1;
    constexpr double stream_retention = 0;
    std::vector<double> ret(static_cast<std::size_t>(n), retention_nodata);
    std::vector<int> to_process(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) to_process[static_cast<std::size_t>(i)] = fl.outflow_bits(i);
    std::vector<i64> stack;
    std::vector<Step> down, up;
    const double sqrt_2 = std::sqrt(2.0);
    if (block <= 0) block = static_cast<int>(std::max(ny, nx));
    for (i64 by = 0; by < ny; by += block)
        for (i64 bx = 0; bx < nx; bx += block) {
            // the drains of the block: cells that flow off the map or into a cell without flow
            for (i64 y = by; y < std::min(ny, by + block); ++y)
                for (i64 x = bx; x < std::min(nx, bx + block); ++x) {
                    const i64 i = y * nx + x;
                    int outflow = to_process[static_cast<std::size_t>(i)];
                    bool seed = false;
                    for (int k = 0; k < 8; ++k) {
                        const int mask = 1 << k;
                        if (!(outflow & mask)) continue;
                        const i64 xn = x + COL[k], yn = y + ROW[k];
                        if (!fl.inside(xn, yn) || to_process[static_cast<std::size_t>(yn * nx + xn)] == 0) {
                            seed = true;
                            outflow &= ~mask;
                        }
                    }
                    if (seed) {
                        to_process[static_cast<std::size_t>(i)] = outflow;
                        stack.push_back(i);
                    }
                }
            while (!stack.empty()) {
                const i64 flat = stack.back();
                stack.pop_back();
                const i64 yi = flat / nx, xi = flat % nx;
                const double crit_i = crit[flat], eff_i = eff[flat];
                const int dir_i = fl.val(flat);
                if (stream[flat] == 1) {
                    ret[static_cast<std::size_t>(flat)] = stream_retention;
                } else if (std::isnan(crit_i) || std::isnan(eff_i) || is_close(dir_i, T::nodata)) {
                    ret[static_cast<std::size_t>(flat)] = retention_nodata;
                } else {
                    double r = 0;
                    long dir_sum = 0;
                    fl.downslope(xi, yi, down);
                    for (const Step& j : down) {
                        dir_sum += static_cast<long>(j.proportion);
                        if (!fl.inside(j.x, j.y)) continue;
                        const double ret_j = ret[static_cast<std::size_t>(j.y * nx + j.x)];
                        if (is_close(ret_j, retention_nodata)) continue;
                        const double step_length = j.direction % 2 == 1 ? cell_size * sqrt_2 : cell_size;
                        // a critical length of 0 is guarded
                        const double step_factor = crit_i > 0 ? std::exp(-5 * step_length / crit_i) : 0;
                        double between;
                        if (ret_j == stream_retention)  // the next cell is a stream
                            between = eff_i * (1 - step_factor);
                        else if (eff_i > ret_j)
                            between = (ret_j * step_factor) + (eff_i * (1 - step_factor));
                        else
                            between = ret_j;
                        r += between * j.proportion;
                    }
                    // InVEST stops on a cell without outflow; none is ever pushed
                    if (!down.empty()) ret[static_cast<std::size_t>(flat)] = r / dir_sum;
                }
                // the upslope cells whose every downslope neighbour is done go on the stack
                fl.upslope(xi, yi, up);
                for (const Step& k : up) {
                    const int mask = 1 << REVERSE[k.direction];
                    const std::size_t kk = static_cast<std::size_t>(k.y * nx + k.x);
                    int todo = to_process[kk];
                    if (todo == 0 || (todo & mask) == 0) continue;
                    todo &= ~mask;
                    to_process[kk] = todo;
                    if (todo == 0) stack.push_back(k.y * nx + k.x);
                }
            }
        }
    for (i64 i = 0; i < n; ++i) {
        const double v = ret[static_cast<std::size_t>(i)];
        out[i] = v == retention_nodata ? kNaN : static_cast<float>(v);
    }
}

}  // namespace

void ndr_retention_mfd(const std::int32_t* dir, const std::uint8_t* stream, const float* eff, const float* crit,
                       int ny, int nx, double cell_size, int block, float* out) {
    retention(MFD{dir, ny, nx}, stream, eff, crit, cell_size, block, out);
}

void ndr_retention_d8(const std::uint8_t* dir, const std::uint8_t* stream, const float* eff, const float* crit, int ny,
                      int nx, double cell_size, int block, float* out) {
    retention(D8{dir, ny, nx}, stream, eff, crit, cell_size, block, out);
}

}  // namespace serv
