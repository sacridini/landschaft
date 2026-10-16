// SPDX-License-Identifier: GPL-3.0-or-later
//
// Least-cost depression breaching, as WhiteboxTools 2.4's BreachDepressionsLeastCost
// (MIT; J. Lindsay) computes it, with its candidate cells taken in the same order: its
// min-heap is Rust's BinaryHeap, reproduced below, so equal costs are resolved alike.
// Differs from it (only the correct form): a channel that reaches the map's edge is
// cut (WhiteboxTools starts walking back from the cell beyond the edge, which has no
// way back, and leaves the channel uncut), and the pits left are filled with the edge
// and the missing cells as outlets (see fill).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "hydro/hydro.hpp"

namespace hydro {

namespace {

struct Candidate {
    i64 row, col;
    double priority;
};

// Rust's BinaryHeap<GridCell> with GridCell ordered by the reverse of its priority:
// a min-heap whose equal elements come out in the same order as Rust's.
class RustMinHeap {
public:
    void clear() { data_.clear(); }
    bool empty() const { return data_.empty(); }
    void push(const Candidate& c) {
        data_.push_back(c);
        sift_up(0, data_.size() - 1);
    }
    Candidate pop() {
        Candidate item = data_.back();
        data_.pop_back();
        if (!data_.empty()) {
            std::swap(item, data_[0]);
            sift_down_to_bottom(0);
        }
        return item;
    }

private:
    // a <= b in the element's order: b.priority <= a.priority (partial_cmp reversed)
    static bool le(const Candidate& a, const Candidate& b) { return b.priority <= a.priority; }

    void sift_up(std::size_t start, std::size_t pos) {
        const Candidate e = data_[pos];
        while (pos > start) {
            const std::size_t parent = (pos - 1) / 2;
            if (le(e, data_[parent])) break;
            data_[pos] = data_[parent];
            pos = parent;
        }
        data_[pos] = e;
    }

    void sift_down_to_bottom(std::size_t pos) {
        const std::size_t end = data_.size(), start = pos;
        const Candidate e = data_[pos];
        std::size_t child = 2 * pos + 1;
        const std::size_t last = end >= 2 ? end - 2 : 0;
        while (child <= last && end >= 2) {
            child += le(data_[child], data_[child + 1]) ? 1 : 0;
            data_[pos] = data_[child];
            pos = child;
            child = 2 * pos + 1;
        }
        if (child == end - 1) {
            data_[pos] = data_[child];
            pos = child;
        }
        data_[pos] = e;
        sift_up(start, pos);
    }

    std::vector<Candidate> data_;
};

// WhiteboxTools' automatic flat increment: from the number of digits of the largest
// elevation's integer part and the diagonal of a cell.
double auto_increment(const Grid& g) {
    double maximum = -std::numeric_limits<double>::infinity();
    const i64 n = static_cast<i64>(g.ny) * g.nx;
    for (i64 i = 0; i < n; ++i)
        if (!std::isnan(g.z[i])) maximum = std::max(maximum, g.z[i]);
    if (!std::isfinite(maximum)) return 0.0;
    const int digits = static_cast<int>(std::to_string(static_cast<std::int32_t>(maximum)).size());
    const double multiplier = std::pow(10.0, 9 - digits);
    const double diag = std::sqrt(g.dx * g.dx + g.dy * g.dy);
    return 1.0 / multiplier * std::ceil(diag);
}

}  // namespace

std::vector<double> breach(const Grid& g, int max_length, double max_cost, bool min_dist, double small_num,
                           bool fill_rest) {
    const i64 ny = g.ny, nx = g.nx, n = ny * nx;
    if (!(small_num > 0.0)) small_num = auto_increment(g);
    const double diag = std::sqrt(g.dx * g.dx + g.dy * g.dy);
    const double cost_dist[8] = {diag, g.dx, diag, g.dy, diag, g.dx, diag, g.dy};
    constexpr int BACK[8] = {4, 5, 6, 7, 0, 1, 2, 3};
    std::vector<double> out(g.z, g.z + n);
    auto inside = [&](i64 r, i64 c) { return r >= 0 && c >= 0 && r < ny && c < nx; };
    auto value = [&](i64 r, i64 c) { return inside(r, c) ? out[r * nx + c] : std::numeric_limits<double>::quiet_NaN(); };

    // the pits (no lower neighbour, none missing), raised to just below their lowest neighbour
    struct Pit {
        i64 row, col;
        double z;
    };
    std::vector<Pit> pits;
    for (i64 r = 0; r < ny; ++r)
        for (i64 c = 0; c < nx; ++c) {
            const double z = g.z[r * nx + c];
            if (std::isnan(z)) continue;
            bool pit = true;
            double min_zn = std::numeric_limits<double>::infinity();
            for (int k = 0; k < 8; ++k) {
                const i64 rn = r + DROW[k], cn = c + DCOL[k];
                const double zn = inside(rn, cn) ? g.z[rn * nx + cn] : std::numeric_limits<double>::quiet_NaN();
                if (std::isnan(zn) || zn < z) {
                    pit = false;
                    break;
                }
                min_zn = std::min(min_zn, zn);
            }
            if (pit) {
                out[r * nx + c] = min_zn - small_num;
                pits.push_back({r, c, z});
            }
        }
    // from the lowest pit up (a stable sort from the highest, taken from the back)
    std::stable_sort(pits.begin(), pits.end(), [](const Pit& a, const Pit& b) { return a.z > b.z; });

    std::vector<std::int8_t> backlink(static_cast<std::size_t>(n), -1), encountered(static_cast<std::size_t>(n), 0);
    std::vector<std::int32_t> length(static_cast<std::size_t>(n), 0);
    std::vector<i64> scanned;
    RustMinHeap heap;
    bool unsolved = false;
    while (!pits.empty()) {
        const Pit pit = pits.back();
        pits.pop_back();
        const i64 row = pit.row, col = pit.col;
        const double z = out[row * nx + col];
        bool still_pit = true;
        for (int k = 0; k < 8; ++k) {
            const double zn = value(row + DROW[k], col + DCOL[k]);
            if (zn < z) {   // NaN compares false
                still_pit = false;
                break;
            }
        }
        if (!still_pit) continue;
        encountered[row * nx + col] = 1;
        heap.clear();
        heap.push({row, col, 0.0});
        scanned.push_back(row * nx + col);
        bool searching = true;
        while (!heap.empty() && searching) {
            const Candidate cell = heap.pop();
            const double accum = cell.priority;
            if (accum > max_cost) {
                unsolved = true;
                searching = false;
                break;
            }
            const std::int32_t len = length[cell.row * nx + cell.col];
            const double cost1 = out[cell.row * nx + cell.col] - z + len * small_num;
            for (int k = 0; k < 8; ++k) {
                const i64 rn = cell.row + DROW[k], cn = cell.col + DCOL[k];
                const bool in = inside(rn, cn);
                const i64 j = in ? rn * nx + cn : -1;
                if (in && encountered[j] == 1) continue;
                const std::int32_t len_n = len + 1;
                if (in) {
                    scanned.push_back(j);
                    length[j] = len_n;
                    backlink[j] = static_cast<std::int8_t>(BACK[k]);
                }
                const double zn = in ? out[j] : std::numeric_limits<double>::quiet_NaN();
                const double zout = z - len_n * small_num;
                if (zn > zout) {   // a cell to cut through (NaN compares false)
                    const double cost2 = zn - zout;
                    const double new_cost = min_dist ? accum + (cost1 + cost2) / 2.0 * cost_dist[k] : accum + cost2;
                    encountered[j] = 1;
                    if (len_n <= max_length) heap.push({rn, cn, new_cost});
                } else {
                    // a cell to drain to: cut the channel back to the pit
                    i64 r = rn, c = cn;
                    if (!in) {   // off the map: the channel ends at this cell, which is cut too
                        r = cell.row;
                        c = cell.col;
                        const double zo = z - len * small_num;
                        if (out[r * nx + c] > zo) out[r * nx + c] = zo;
                    }
                    while (backlink[r * nx + c] > -1) {
                        const int b = backlink[r * nx + c];
                        r += DROW[b];
                        c += DCOL[b];
                        const double zo = z - length[r * nx + c] * small_num;
                        if (out[r * nx + c] > zo) out[r * nx + c] = zo;
                    }
                    searching = false;
                    break;
                }
            }
        }
        for (i64 j : scanned) {
            backlink[j] = -1;
            encountered[j] = 0;
            length[j] = 0;
        }
        scanned.clear();
        if (searching) unsolved = true;   // no cell to drain to within reach
    }
    if (fill_rest && unsolved) {
        Grid filled = g;
        filled.z = out.data();
        out = fill(filled, false);
    }
    return out;
}

}  // namespace hydro
