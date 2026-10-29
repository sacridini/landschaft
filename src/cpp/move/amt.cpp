// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/amt.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "lsm/rmath.hpp"
#include "move/geo.hpp"
#include "move/track.hpp"
#include "sdm/geocircle/geocircle.hpp"

namespace move {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.141592653589793238462643383280;

}  // namespace

void amt_steps(const double* x, const double* y, const i64* group, i64 n, bool lonlat, double* sl, double* dir,
               double* ta) {
    for (i64 i = 0; i < n; i++) {
        if (i + 1 >= n || group[i + 1] != group[i]) {
            sl[i] = dir[i] = ta[i] = kNaN;
            continue;
        }
        // data.table::shift(x, n = -1) - x
        const double dx = x[i + 1] - x[i], dy = y[i + 1] - y[i];
        double a;
        if (lonlat) {
            sl[i] = s2_distance(x[i], y[i], x[i + 1], y[i + 1]);
            a = (90 - sdm::geosphere_bearing(x[i], y[i], x[i + 1], y[i + 1])) * kPi / 180;
        } else {
            sl[i] = lsm::r_sqrt(dx * dx + dy * dy);
            a = lsm::r_atan2(dy, dx);
        }
        if (dx == 0 && dy == 0) a = kNaN;
        // ifelse(a < 0, 2 * pi + a, a); zero_dir "E"; not clockwise;
        // ifelse(a > pi, (2 * pi - a) * -1, a)
        if (a < 0) a = 2 * kPi + a;
        if (a > kPi) a = (2 * kPi - a) * -1;
        dir[i] = a;
    }
    // direction_rel: c(NA, diff(p)) %% (2 * pi); ifelse(p > pi, p - 2 * pi, p)
    for (i64 i = 0; i < n; i++) {
        if (std::isnan(sl[i]) || i == 0 || group[i - 1] != group[i]) {
            ta[i] = kNaN;
            continue;
        }
        const double p = r_fmod(dir[i] - dir[i - 1], 2 * kPi);
        ta[i] = p > kPi ? p - 2 * kPi : p;
    }
}

void amt_resample(const double* t, const i64* group, i64 n, double rate, double tolerance, i64 start, i64* burst) {
    i64 g0 = 0;
    while (g0 < n) {
        i64 g1 = g0 + 1;
        while (g1 < n && group[g1] == group[g0]) g1++;
        // mk_reg(t1 = t[g0:g1], time_dist = rate, time_tol = tolerance, start), 0-based
        const double* t1 = t + g0;
        i64* out = burst + g0;
        const i64 m = g1 - g0;
        for (i64 j = 0; j < m; j++) out[j] = 0;
        if (start >= m) {
            for (i64 j = 0; j < m; j++) out[j] = -1;
            g0 = g1;
            continue;
        }
        i64 k = 1;
        for (i64 j = 0; j < start; j++) out[j] = -1;
        out[start] = 1;
        i64 i = start;
        while (i != m - 1) {
            const double t_min = t1[i] + rate - tolerance;
            const double t_max = t1[i] + rate + tolerance;
            i64 j = i + 1;
            while (j < m - 1 && t1[j] < t_min) {
                out[j] = -1;
                j++;
            }
            i = j;
            if (j == m - 1 && t1[j] < t_min) {
                out[j] = -1;
            } else if (t1[j] >= t_min && t1[j] <= t_max) {
                out[j] = k;
            } else {
                k++;
                out[j] = k;
            }
        }
        g0 = g1;
    }
}

void amt_controls(const double* x1, const double* y1, const double* x0, const double* y0, const double* angle,
                  const double* sl, const double* ta, i64 n, double* x2, double* y2, double* ta_out) {
    for (i64 i = 0; i < n; i++) {
        // random_steps.numeric: new.x <- x[1] + slr * cos(tar + angle)
        const double a = ta[i] + angle[i];
        x2[i] = x1[i] + sl[i] * lsm::r_cos(a);
        y2[i] = y1[i] + sl[i] * lsm::r_sin(a);
        // abs.dir1 <- atan2(y1_ - y0_, x1_ - x0_); abs.dir2 <- atan2(y2_ - y1_, x2_ - x1_)
        const double d1 = lsm::r_atan2(y1[i] - y0[i], x1[i] - x0[i]);
        const double d2 = lsm::r_atan2(y2[i] - y1[i], x2[i] - x1[i]);
        double rel = d2 - d1;
        if (rel <= -kPi) rel = rel + 2 * kPi;
        if (rel >= kPi) rel = rel - 2 * kPi;
        ta_out[i] = rel;
    }
}

namespace {

// suncalc's .getSetJ (with .hourAngle, .approxTransit and .solarTransitJ)
double sun_set_j(double h, double lw, double phi, double dec, double nc, double M, double L) {
    const double w =
        std::acos((lsm::r_sin(h) - lsm::r_sin(phi) * lsm::r_sin(dec)) / (lsm::r_cos(phi) * lsm::r_cos(dec)));
    const double a = 0.0009 + (w + lw) / (2 * kPi) + nc;
    return 2451545 + a + 0.0053 * lsm::r_sin(M) - 0.0069 * lsm::r_sin(2 * L);
}

// suncalc's .fromJulian: the time in seconds, floored to the second
double from_julian(double j) { return std::floor((j + 0.5 - 2440588) * 86400); }

}  // namespace

void sun_times(double lon, double lat, double noon, double* sunrise, double* sunset, double* dawn, double* dusk) {
    const double rad = kPi / 180;
    const double lw = rad * -lon;
    const double phi = rad * lat;
    // .toDays(date) = .toJulian(date) - J2000
    const double d = ((noon / 86400) - 0.5 + 2440588) - 2451545;
    const double nc = std::nearbyint(d - 0.0009 - lw / (2 * kPi));
    const double ds = 0.0009 + (0 + lw) / (2 * kPi) + nc;
    const double M = (kPi / 180) * (357.5291 + 0.98560028 * ds);
    const double C = (kPi / 180) * (1.9148 * lsm::r_sin(M) + 0.02 * lsm::r_sin(2 * M) + 0.0003 * lsm::r_sin(3 * M));
    const double P = (kPi / 180) * 102.9372;
    const double L = M + C + P + kPi;
    const double e = (kPi / 180) * 23.4397;
    const double dec = std::asin(lsm::r_sin(0.0) * lsm::r_cos(e) + lsm::r_cos(0.0) * lsm::r_sin(e) * lsm::r_sin(L));
    const double jnoon = 2451545 + ds + 0.0053 * lsm::r_sin(M) - 0.0069 * lsm::r_sin(2 * L);
    const double set833 = sun_set_j(-0.833 * rad, lw, phi, dec, nc, M, L);
    const double set6 = sun_set_j(-6 * rad, lw, phi, dec, nc, M, L);
    *sunrise = from_julian(jnoon - (set833 - jnoon));
    *sunset = from_julian(set833);
    *dawn = from_julian(jnoon - (set6 - jnoon));
    *dusk = from_julian(set6);
}

double r_sum_of(const double* x, i64 n) { return lsm::r_sum(x, n); }

double r_mean_of(const double* x, i64 n) { return lsm::r_mean(x, n); }

double amt_tac(const double* ta, i64 n_steps) {
    // ta <- x$ta_[-1]; 1 / nrow(x) * sum(diff(cos(ta))^2 + diff(sin(ta))^2, na.rm = TRUE)
    std::vector<double> terms;
    for (i64 i = 2; i < n_steps; i++) {
        const double dc = lsm::r_cos(ta[i]) - lsm::r_cos(ta[i - 1]);
        const double ds = lsm::r_sin(ta[i]) - lsm::r_sin(ta[i - 1]);
        const double v = dc * dc + ds * ds;
        if (!std::isnan(v)) terms.push_back(v);
    }
    const double s = terms.empty() ? 0.0 : lsm::r_sum(terms.data(), static_cast<i64>(terms.size()));
    return 1 / static_cast<double>(n_steps) * s;
}

double amt_sinuosity(const double* sl, const double* ta, i64 n_steps) {
    const double p = lsm::r_mean(sl, n_steps);
    const double cv = lsm::r_sd(sl, n_steps, p) / p;
    std::vector<double> c, s;
    for (i64 i = 0; i < n_steps; i++) {
        if (std::isnan(ta[i])) continue;
        c.push_back(lsm::r_cos(ta[i]));
        s.push_back(lsm::r_sin(ta[i]));
    }
    const double mc = lsm::r_mean(c.data(), static_cast<i64>(c.size()));
    const double ms = lsm::r_mean(s.data(), static_cast<i64>(s.size()));
    const double inner = p * ((1 - mc * mc - ms * ms) / ((1 - mc) * (1 - mc) + ms * ms) + cv * cv);
    return 2 * lsm::r_pow(inner, -0.5);
}

double amt_msd(const double* x, const double* y, i64 n) {
    const double mx = lsm::r_mean(x, n), my = lsm::r_mean(y, n);
    std::vector<double> d(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; i++) {
        const double ex = x[i] - mx, ey = y[i] - my;
        d[i] = ex * ex + ey * ey;
    }
    return lsm::r_mean(d.data(), n);
}

}  // namespace move
