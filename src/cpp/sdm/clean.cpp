// SPDX-License-Identifier: GPL-3.0-or-later
//
// CoordinateCleaner's outlier statistic without its n x n matrix: each row of
// distances is computed on the fly (in parallel over the rows) and reduced as
// R reduces it (apply(dist, 1, mean, na.rm = TRUE): the non-missing values in
// column order, R's 80-bit mean).

#include "sdm/clean.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "lsm/rmath.hpp"

namespace sdm {

namespace {

constexpr double kPi = 3.141592653589793;  // R's pi
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kToRad = kPi / 180.0;

struct Rad {
    double lon, lat, coslat;
};

std::vector<Rad> radians(const double* lon, const double* lat, i64 n) {
    std::vector<Rad> p(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) {
        Rad& r = p[static_cast<std::size_t>(i)];
        r.lon = lon[i] * kToRad;
        r.lat = lat[i] * kToRad;
        r.coslat = lsm::r_cos(r.lat);
    }
    return p;
}

// distHaversine(p1, p2) / 1000 on points already in radians.
inline double haversine(const Rad& p1, const Rad& p2) {
    const double dlat = p2.lat - p1.lat;
    const double dlon = p2.lon - p1.lon;
    const double s1 = lsm::r_sin(dlat / 2.0);
    const double s2 = lsm::r_sin(dlon / 2.0);
    double a = s1 * s1 + (p1.coslat * p2.coslat) * (s2 * s2);
    if (a > 1.0) a = 1.0;  // pmin(a, 1)
    const double d = (2.0 * lsm::r_atan2(lsm::r_sqrt(a), lsm::r_sqrt(1.0 - a))) * 6378137.0;
    return d / 1000.0;
}

// R's round(x) (digits 0): half to even.
inline double round_half_even(double x) { return std::nearbyint(x); }

}  // namespace

double haversine_km(double lon1, double lat1, double lon2, double lat2) {
    Rad a{lon1 * kToRad, lat1 * kToRad, 0.0}, b{lon2 * kToRad, lat2 * kToRad, 0.0};
    a.coslat = lsm::r_cos(a.lat);
    b.coslat = lsm::r_cos(b.lat);
    return haversine(a, b);
}

std::vector<double> outlier_statistic(const double* lon, const double* lat, i64 n, bool use_min) {
    const std::vector<Rad> p = radians(lon, lat, n);
    std::vector<double> out(static_cast<std::size_t>(n), kNaN);
#pragma omp parallel
    {
        std::vector<double> row;
        row.reserve(static_cast<std::size_t>(n));
#pragma omp for schedule(dynamic, 16)
        for (i64 i = 0; i < n; ++i) {
            row.clear();
            double best = std::numeric_limits<double>::infinity();
            for (i64 j = 0; j < n; ++j) {
                if (j == i) continue;
                const Rad& hi = p[static_cast<std::size_t>(i > j ? i : j)];
                const Rad& lo = p[static_cast<std::size_t>(i > j ? j : i)];
                const double d = haversine(hi, lo);
                if (d == 0.0 || std::isnan(d)) continue;  // dist[dist == 0] <- NA
                if (use_min) {
                    if (d < best) best = d;
                } else {
                    row.push_back(d);
                }
            }
            if (use_min) {
                out[static_cast<std::size_t>(i)] = best;  // min(numeric(0), na.rm = TRUE) = Inf
            } else if (!row.empty()) {
                out[static_cast<std::size_t>(i)] = lsm::r_mean(row.data(), static_cast<i64>(row.size()));
            }
        }
    }
    return out;
}

std::vector<double> outlier_statistic_weighted(const double* lon, const double* lat, const double* count, i64 n,
                                               double within, bool use_min) {
    const std::vector<Rad> p = radians(lon, lat, n);
    std::vector<double> out(static_cast<std::size_t>(n), kNaN);
#pragma omp parallel for schedule(dynamic, 16)
    for (i64 i = 0; i < n; ++i) {
        // rowSums of integers (exact in any order) over rowSums of the weights
        double sd = 0.0, sw = 0.0, best = std::numeric_limits<double>::infinity();
        for (i64 j = 0; j < n; ++j) {
            const Rad& hi = p[static_cast<std::size_t>(i > j ? i : j)];
            const Rad& lo = p[static_cast<std::size_t>(i > j ? j : i)];
            double d = j == i ? 0.0 : haversine(hi, lo);
            if (d == 0.0) d = within;
            const double w = count[i] * count[j];
            const double r = round_half_even(d * w);
            sd += r;
            sw += w;
            if (r < best) best = r;
        }
        out[static_cast<std::size_t>(i)] = use_min ? best : sd / sw;
    }
    return out;
}

}  // namespace sdm
