// SPDX-License-Identifier: GPL-3.0-or-later
//
// The patches of an observed transition (see observed.hpp), the same numbers
// as landschaft's former scipy.ndimage code.

#include "sim/observed.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace sim {

namespace {

using i64 = std::int64_t;

// numpy's pairwise summation of doubles (np.add.reduce of a contiguous array):
// blocks of up to 128 values in 8 interleaved sums, halves above.
double pairwise_sum(const double* a, i64 n) {
    if (n < 8) {
        double res = 0.0;
        for (i64 i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        i64 i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    i64 n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

double np_sum(const std::vector<double>& v) { return 0.0 + pairwise_sum(v.data(), static_cast<i64>(v.size())); }

// numpy's mean and var (ddof 0) of the sizes, (1, 0) without any.
void stats(const std::vector<double>& s, double& mean, double& var) {
    if (s.empty()) {
        mean = 1.0;
        var = 0.0;
        return;
    }
    const double n = static_cast<double>(s.size());
    mean = np_sum(s) / n;
    std::vector<double> d(s.size());
    for (std::size_t k = 0; k < s.size(); ++k) {
        const double x = s[k] - mean;
        d[k] = x * x;
    }
    var = np_sum(d) / n;
}

i64 find(std::vector<i64>& parent, i64 x) {
    while (parent[static_cast<std::size_t>(x)] != x) {
        parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
        x = parent[static_cast<std::size_t>(x)];
    }
    return x;
}

}  // namespace

ObservedPatches observed_patches(const double* a, const double* b, std::int64_t ny, std::int64_t nx, double from,
                                 double to) {
    const i64 n = ny * nx;
    std::vector<char> changed(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) changed[static_cast<std::size_t>(i)] = a[i] == from && b[i] == to;

    // 8-connected components: union with the neighbours already visited
    std::vector<i64> parent(static_cast<std::size_t>(n));
    std::iota(parent.begin(), parent.end(), i64{0});
    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const i64 i = y * nx + x;
            if (!changed[static_cast<std::size_t>(i)]) continue;
            const i64 nb[4][2] = {{y, x - 1}, {y - 1, x - 1}, {y - 1, x}, {y - 1, x + 1}};
            for (const auto& q : nb) {
                if (q[0] < 0 || q[1] < 0 || q[1] >= nx) continue;
                const i64 j = q[0] * nx + q[1];
                if (!changed[static_cast<std::size_t>(j)]) continue;
                const i64 ri = find(parent, i), rj = find(parent, j);
                if (ri != rj) parent[static_cast<std::size_t>(std::max(ri, rj))] = std::min(ri, rj);
            }
        }
    }
    // patch numbers in the order of their first cell; their sizes and whether they touch the target class
    std::vector<i64> label(static_cast<std::size_t>(n), -1);
    std::vector<double> sizes;
    std::vector<char> touching;
    auto target_near = [&](i64 y, i64 x) {
        for (i64 dy = -1; dy <= 1; ++dy)
            for (i64 dx = -1; dx <= 1; ++dx) {
                const i64 yy = y + dy, xx = x + dx;
                if (yy >= 0 && yy < ny && xx >= 0 && xx < nx && a[yy * nx + xx] == to) return true;
            }
        return false;
    };
    ObservedPatches out;
    for (i64 y = 0; y < ny; ++y) {
        for (i64 x = 0; x < nx; ++x) {
            const i64 i = y * nx + x;
            if (!changed[static_cast<std::size_t>(i)]) continue;
            const i64 r = find(parent, i);
            if (label[static_cast<std::size_t>(r)] < 0) {
                label[static_cast<std::size_t>(r)] = static_cast<i64>(sizes.size());
                sizes.push_back(0.0);
                touching.push_back(0);
            }
            const auto k = static_cast<std::size_t>(label[static_cast<std::size_t>(r)]);
            sizes[k] += 1.0;
            if (!touching[k] && target_near(y, x)) touching[k] = 1;
            ++out.cells;
        }
    }
    std::vector<double> expanding, fresh;
    for (std::size_t k = 0; k < sizes.size(); ++k) (touching[k] ? expanding : fresh).push_back(sizes[k]);
    const double total = np_sum(sizes);
    out.expansion = total != 0.0 ? np_sum(expanding) / total : 0.0;
    out.expansion_patches = static_cast<i64>(expanding.size());
    out.new_patches = static_cast<i64>(fresh.size());
    stats(expanding, out.expansion_mean, out.expansion_variance);
    stats(fresh, out.new_mean, out.new_variance);
    return out;
}

}  // namespace sim
