// SPDX-License-Identifier: GPL-3.0-or-later
//
// spThin's thinning (spThin/R/thin.algorithm.R) without its n x n matrices:
//
// - the neighbours come from a grid over the unit vectors of the points, the
//   distance of each candidate pair evaluated as fields::rdist.earth does
//   (x87 sin / cos of (deg * pi) / 180, the dot product summed left to
//   right as R's reference BLAS dgemm, clamped to +-1, the C library's acos,
//   times 6378.388 km);
// - "which(S == max(S))" and "sample(cand, 1)" through a segment tree holding
//   the maximum neighbour count, how many points have it and, by descending
//   it, the k-th of them in index order: one removal costs O(degree log n)
//   instead of R's O(n^2).
//
// The removals and the draws are R's, in R's order: one unif_index(#ties)
// per removal, only when the maximum is tied.

#include "sdm/thin.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <unordered_map>

#include "lsm/rmath.hpp"
#include "nlm/rng.hpp"

namespace sdm {

namespace {

constexpr double kEarthKm = 6378.388;  // fields::rdist.earth(miles = FALSE)
constexpr double kPi = 3.141592653589793;  // R's pi

struct Unit {
    double x, y, z;
    bool ok;
};

std::vector<Unit> unit_vectors(const double* lon, const double* lat, i64 n) {
    std::vector<Unit> a(static_cast<std::size_t>(n));
#pragma omp parallel for schedule(static)
    for (i64 i = 0; i < n; ++i) {
        Unit& u = a[static_cast<std::size_t>(i)];
        u.ok = std::isfinite(lon[i]) && std::isfinite(lat[i]);
        if (!u.ok) {
            u.x = u.y = u.z = 0.0;
            continue;
        }
        const double rlat = (lat[i] * kPi) / 180.0, rlon = (lon[i] * kPi) / 180.0;
        const double coslat = lsm::r_cos(rlat), sinlat = lsm::r_sin(rlat);
        const double coslon = lsm::r_cos(rlon), sinlon = lsm::r_sin(rlon);
        u.x = coslat * coslon;
        u.y = coslat * sinlon;
        u.z = sinlat;
    }
    return a;
}

// rdist.earth's distance between two points (km).
inline double earth_km(const Unit& a, const Unit& b) {
    double pp = (b.x * a.x + b.y * a.y) + b.z * a.z;
    if (std::fabs(pp) > 1.0) pp = pp > 0.0 ? 1.0 : -1.0;
    return kEarthKm * std::acos(pp);
}

// Segment tree over the neighbour counts: maximum, multiplicity, k-th position.
class MaxTree {
public:
    explicit MaxTree(i64 n) : n_(n) {
        size_ = 1;
        while (size_ < n) size_ <<= 1;
        max_.assign(static_cast<std::size_t>(2 * size_), -1);
        cnt_.assign(static_cast<std::size_t>(2 * size_), 0);
    }
    void build(const std::vector<std::int32_t>& v) {
        for (i64 i = 0; i < size_; ++i) {
            max_[static_cast<std::size_t>(size_ + i)] = i < n_ ? v[static_cast<std::size_t>(i)] : -1;
            cnt_[static_cast<std::size_t>(size_ + i)] = i < n_ ? 1 : 0;
        }
        for (i64 p = size_ - 1; p >= 1; --p) pull(p);
    }
    void set(i64 i, std::int32_t value) {
        i64 p = size_ + i;
        max_[static_cast<std::size_t>(p)] = value;
        for (p >>= 1; p >= 1; p >>= 1) pull(p);
    }
    std::int32_t max() const { return max_[1]; }
    i64 count() const { return cnt_[1]; }
    // Position of the k-th (1-based) point holding the maximum, in index order.
    i64 kth(i64 k) const {
        const std::int32_t m = max_[1];
        i64 p = 1;
        while (p < size_) {
            const i64 l = 2 * p;
            if (max_[static_cast<std::size_t>(l)] == m) {
                if (cnt_[static_cast<std::size_t>(l)] >= k) {
                    p = l;
                    continue;
                }
                k -= cnt_[static_cast<std::size_t>(l)];
            }
            p = l + 1;
        }
        return p - size_;
    }

private:
    void pull(i64 p) {
        const std::size_t l = static_cast<std::size_t>(2 * p), r = l + 1;
        const std::int32_t m = std::max(max_[l], max_[r]);
        max_[static_cast<std::size_t>(p)] = m;
        cnt_[static_cast<std::size_t>(p)] = (max_[l] == m ? cnt_[l] : 0) + (max_[r] == m ? cnt_[r] : 0);
    }
    i64 n_, size_;
    std::vector<std::int32_t> max_;
    std::vector<i64> cnt_;
};

}  // namespace

Neighbours earth_neighbours(const double* lon, const double* lat, i64 n, double max_km) {
    const std::vector<Unit> a = unit_vectors(lon, lat, n);
    std::vector<std::vector<std::int32_t>> lists(static_cast<std::size_t>(n));
    const double theta = max_km / kEarthKm;  // the angle of max_km
    if (max_km > 0.0) {
        // candidate pairs: unit vectors within the chord of a slightly larger angle
        const bool everything = theta >= 3.0;
        const double chord = everything ? 2.0 : 2.0 * std::sin(std::min(theta * (1.0 + 1e-9) + 1e-12, kPi) / 2.0);
        const double h = std::max(chord, 1e-9);
        auto cell = [&](double v) { return static_cast<std::int64_t>(std::floor((v + 1.0) / h)); };
        std::unordered_map<std::int64_t, std::vector<std::int32_t>> grid;
        const std::int64_t span = cell(1.0) + 3;
        auto key = [&](std::int64_t cx, std::int64_t cy, std::int64_t cz) { return (cx * span + cy) * span + cz; };
        for (i64 i = 0; i < n; ++i) {
            const Unit& u = a[static_cast<std::size_t>(i)];
            if (u.ok) grid[key(cell(u.x), cell(u.y), cell(u.z))].push_back(static_cast<std::int32_t>(i));
        }
#pragma omp parallel for schedule(dynamic, 256)
        for (i64 i = 0; i < n; ++i) {
            const Unit& u = a[static_cast<std::size_t>(i)];
            if (!u.ok) continue;
            std::vector<std::int32_t>& out = lists[static_cast<std::size_t>(i)];
            const std::int64_t cx = cell(u.x), cy = cell(u.y), cz = cell(u.z);
            for (std::int64_t dx = -1; dx <= 1; ++dx)
                for (std::int64_t dy = -1; dy <= 1; ++dy)
                    for (std::int64_t dz = -1; dz <= 1; ++dz) {
                        const auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
                        if (it == grid.end()) continue;
                        for (const std::int32_t j : it->second) {
                            if (j == i) continue;
                            if (earth_km(u, a[static_cast<std::size_t>(j)]) < max_km) out.push_back(j);
                        }
                    }
            std::sort(out.begin(), out.end());
        }
    }
    Neighbours nb;
    nb.start.assign(static_cast<std::size_t>(n + 1), 0);
    for (i64 i = 0; i < n; ++i)
        nb.start[static_cast<std::size_t>(i + 1)] =
            nb.start[static_cast<std::size_t>(i)] + static_cast<i64>(lists[static_cast<std::size_t>(i)].size());
    nb.index.resize(static_cast<std::size_t>(nb.start.back()));
#pragma omp parallel for schedule(static)
    for (i64 i = 0; i < n; ++i)
        std::copy(lists[static_cast<std::size_t>(i)].begin(), lists[static_cast<std::size_t>(i)].end(),
                  nb.index.begin() + nb.start[static_cast<std::size_t>(i)]);
    return nb;
}

ThinResult thin_spthin(const double* lon, const double* lat, i64 n, double max_km, int reps, std::int32_t seed) {
    const Neighbours nb = earth_neighbours(lon, lat, n, max_km);
    std::vector<std::int32_t> s0(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i)
        s0[static_cast<std::size_t>(i)] =
            static_cast<std::int32_t>(nb.start[static_cast<std::size_t>(i + 1)] - nb.start[static_cast<std::size_t>(i)]);
    ThinResult res;
    res.keep.assign(static_cast<std::size_t>(reps) * static_cast<std::size_t>(n), 1);
    res.kept.assign(static_cast<std::size_t>(reps), n);
    nlm::RRng rng(seed);
    MaxTree tree(std::max<i64>(n, 1));
    std::vector<std::int32_t> s;
    for (int r = 0; r < reps; ++r) {
        std::uint8_t* keep = &res.keep[static_cast<std::size_t>(r) * static_cast<std::size_t>(n)];
        if (n == 0) continue;
        s = s0;
        tree.build(s);
        i64 kept = n;
        while (tree.max() > 0) {
            const i64 ties = tree.count();
            const i64 pick = ties > 1 ? static_cast<i64>(rng.unif_index(static_cast<double>(ties))) + 1 : 1;
            const i64 k = tree.kth(pick);
            for (i64 e = nb.start[static_cast<std::size_t>(k)]; e < nb.start[static_cast<std::size_t>(k + 1)]; ++e) {
                const std::int32_t j = nb.index[static_cast<std::size_t>(e)];
                if (!keep[j]) continue;
                tree.set(j, --s[static_cast<std::size_t>(j)]);
            }
            s[static_cast<std::size_t>(k)] = 0;
            tree.set(k, 0);
            keep[k] = 0;
            --kept;
        }
        res.kept[static_cast<std::size_t>(r)] = kept;
    }
    return res;
}

}  // namespace sdm
