// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/track.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "lsm/extended.hpp"
#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "move/geo.hpp"
#include "nlm/rng.hpp"

namespace move {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 2.220446049250313e-16;  // R's c_eps (DBL_EPSILON)

bool r_finite(double x) { return std::isfinite(x); }

// floorl() of an extended value
lsm::Extended floor_ext(const lsm::Extended& v) {
    const double d = static_cast<double>(v);
    if (!std::isfinite(d)) return v;
    double f = std::floor(d);
    if (static_cast<double>(v - lsm::Extended(f)) < 0) f -= 1.0;
    return lsm::Extended(f);
}

}  // namespace

double r_fmod(double x1, double x2) {
    if (x2 == 0.0) return kNaN;
    if (std::fabs(x2) * kEps > 1 && r_finite(x1) && std::fabs(x1) <= std::fabs(x2)) {
        return (std::fabs(x1) == std::fabs(x2)) ? 0 : ((x1 < 0 && x2 > 0) || (x2 < 0 && x1 > 0)) ? x1 + x2 : x1;
    }
    const double q = x1 / x2;
    const lsm::Extended tmp = lsm::Extended(x1) - lsm::Extended(std::floor(q)) * lsm::Extended(x2);
    return static_cast<double>(tmp - floor_ext(tmp / lsm::Extended(x2)) * lsm::Extended(x2));
}

double r_fdiv(double x1, double x2) {
    const double q = x1 / x2;
    if (x2 == 0.0 || std::fabs(q) * kEps > 1 || !r_finite(q)) return q;
    if (std::fabs(q) < 1) return (q < 0) ? -1 : ((x1 < 0 && x2 > 0) || (x1 > 0 && x2 < 0) ? -1 : 0);
    const lsm::Extended tmp = lsm::Extended(x1) - lsm::Extended(std::floor(q)) * lsm::Extended(x2);
    return static_cast<double>(lsm::Extended(std::floor(q)) + floor_ext(tmp / lsm::Extended(x2)));
}

void distances(const double* x, const double* y, const i64* track, i64 n, Coords coords, int threads, double* out) {
    lsm::parallel_for(threads > 1 && n > 20000, n, [&](i64 i) {
        if (i + 1 >= n || track[i + 1] != track[i]) {
            out[i] = kNaN;
        } else if (coords == Coords::LonLat) {
            out[i] = s2_distance(x[i + 1], y[i + 1], x[i], y[i]);
        } else {
            out[i] = planar_distance(x[i + 1], y[i + 1], x[i], y[i]);
        }
    });
}

void azimuths(const double* x, const double* y, const i64* track, i64 n, Coords coords, double a,
              double inv_flattening, int threads, double* out) {
    const Spheroid* s = nullptr;
    std::unique_ptr<Spheroid> spheroid;
    if (coords == Coords::LonLat) {
        spheroid = std::make_unique<Spheroid>(a, inv_flattening);
        s = spheroid.get();
    }
    lsm::parallel_for(threads > 1 && n > 5000, n, [&](i64 i) {
        if (i + 1 >= n || track[i + 1] != track[i]) {
            out[i] = kNaN;
        } else if (s != nullptr) {
            out[i] = lwgeom_azimuth(*s, x[i], y[i], x[i + 1], y[i + 1]);
        } else {
            out[i] = planar_azimuth(x[i], y[i], x[i + 1], y[i + 1]);
        }
    });
}

void turn_angles(const double* azimuth, i64 n, int threads, double* out) {
    lsm::parallel_for(threads > 1 && n > 20000, n, [&](i64 i) {
        if (i == 0) {
            out[i] = kNaN;
            return;
        }
        const double d = azimuth[i] - azimuth[i - 1];
        if (std::isnan(d)) {
            out[i] = kNaN;
            return;
        }
        // ta <- ((az + pi) %% (pi * 2)) - pi; t <- (ta + pi * 3) %/% (pi * 2); ta + pi * 2 where !t.
        // The azimuths carry units (rad): the units package (0.8) computes e1 %% e2 as
        // e1 - round(e1 / e2) * e2, while %/% of equal units drops them and is R's.
        const double s = d + kPi;
        const double ta = (s - std::nearbyint(s / (kPi * 2)) * (kPi * 2)) - kPi;
        const double t = r_fdiv(ta + kPi * 3, kPi * 2);
        out[i] = (t == 0) ? ta + kPi * 2 : ta;
    });
}

void brownian(void* rng_ptr, const double* t, i64 n, const double* sigma, i64 n_sigma, double x0, double y0,
              double* x, double* y) {
    if (n < 1) return;
    if (n_sigma != 1 && n_sigma != n - 1) throw std::invalid_argument("sigma: one value or one per step");
    auto& rng = *static_cast<nlm::RRng*>(rng_ptr);
    // sigma <- sqrt((sigma^2L) * dt), R's vector sqrt
    std::vector<double> sd(static_cast<std::size_t>(n > 1 ? n - 1 : 0));
    for (i64 i = 0; i + 1 < n; ++i) {
        const double s = sigma[n_sigma == 1 ? 0 : i];
        sd[i] = lsm::r_sqrt((s * s) * (t[i + 1] - t[i]));
    }
    // x <- cumsum(c(start[1], rnorm(n - 1, 0, sigma))); y likewise, drawn after x
    for (int axis = 0; axis < 2; ++axis) {
        double* out = axis == 0 ? x : y;
        lsm::Extended sum = 0.0;
        sum += axis == 0 ? x0 : y0;
        out[0] = static_cast<double>(sum);
        for (i64 i = 1; i < n; ++i) {
            sum += rng.rnorm(0.0, sd[i - 1]);
            out[i] = static_cast<double>(sum);
        }
    }
}

}  // namespace move
