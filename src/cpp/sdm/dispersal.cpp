// SPDX-License-Identifier: GPL-3.0-or-later
//
// MigClim's automaton and MetaLandSim's spom (see dispersal.hpp).

#include "sdm/dispersal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "nlm/rng.hpp"

namespace sdm {

namespace {

using i32 = std::int32_t;

// rand() of Microsoft's C runtime (msvcrt and the UCRT): a linear
// congruential generator of 15 bits.
struct CrtRand {
    std::uint32_t state;
    int rand() {
        state = state * 214013u + 2531011u;
        return static_cast<int>((state >> 16) & 0x7fffu);
    }
    // MigClim's UNIF01: rand() / RAND_MAX.
    double unif01() { return static_cast<double>(rand()) / 32767.0; }
};

struct Automaton {
    i64 ny, nx;
    const MigClimSettings& s;
    CrtRand rng;
    bool use_barrier;

    i64 idx(i64 r, i64 c) const { return r * nx + c; }

    // A line from the sink to the source, offset by (ox, oy), crossing a
    // barrier (mcIntersectsBarrier).
    bool hit(i64 snk_x, i64 snk_y, i64 dist_max, double dx, double dy, double ox, double oy, bool centred,
             const i32* bar) const {
        const double dm = static_cast<double>(dist_max);
        const double sx = static_cast<double>(snk_x), sy = static_cast<double>(snk_y);
        for (i64 k = 1; k <= dist_max; ++k) {
            const double i = static_cast<double>(k);
            double px, py;
            if (centred) {
                px = std::round(sx + ox + (((i - 1.0) / dm * dx) + ((1.0 / dm * dx) / 2.0)));
                py = std::round(sy + oy + (((i - 1.0) / dm * dy) + ((1.0 / dm * dy) / 2.0)));
            } else {
                px = std::round(sx + ox + (1.0 * i / dm * dx));
                py = std::round(sy + oy + (1.0 * i / dm * dy));
            }
            if (bar[idx(static_cast<i64>(px), static_cast<i64>(py))] == 1) return true;
        }
        return false;
    }

    bool intersects_barrier(i64 snk_x, i64 snk_y, i64 src_x, i64 src_y, const i32* bar) const {
        const i64 dst_x = src_x - snk_x, dst_y = src_y - snk_y;
        const i64 dist_max = std::max(std::llabs(dst_x), std::llabs(dst_y));
        const double dx = static_cast<double>(dst_x), dy = static_cast<double>(dst_y);
        static constexpr double off[4][2] = {{-0.49, -0.49}, {0.49, -0.49}, {-0.49, 0.49}, {0.49, 0.49}};
        if (s.weak) {
            if (!hit(snk_x, snk_y, dist_max, dx, dy, 0.0, 0.0, false, bar)) return false;
            for (const auto& o : off)
                if (!hit(snk_x, snk_y, dist_max, dx, dy, o[0], o[1], false, bar)) return false;
            return true;
        }
        int count = hit(snk_x, snk_y, dist_max, dx, dy, 0.0, 0.0, false, bar) ? 1 : 0;
        for (const auto& o : off) {
            count += hit(snk_x, snk_y, dist_max, dx, dy, o[0], o[1], true, bar) ? 1 : 0;
            if (count > 1) return true;
        }
        return false;
    }

    // mcSrcCell: a source within the dispersal distance colonising the sink,
    // its draws in MigClim's order.
    bool source_cell(i64 i, i64 j, const i32* cur, const i32* age, i32 loop_id, i32 suit, const i32* bar) {
        const i64 d = static_cast<i64>(s.kernel.size());
        for (i64 k = i - d; k <= i + d; ++k) {
            for (i64 l = j - d; l <= j + d; ++l) {
                if (k < 0 || k >= ny || l < 0 || l >= nx) continue;
                const i64 at = idx(k, l);
                if (cur[at] <= 0 || cur[at] == loop_id || age[at] < s.ini_maturity) continue;
                const i64 real =
                    static_cast<i64>(std::round(std::sqrt(static_cast<double>((k - i) * (k - i) + (l - j) * (l - j)))));
                if (real == 0 || real > d) continue;
                const double kernel = s.kernel[static_cast<std::size_t>(real - 1)] * 1.0;
                const double p = age[at] >= s.full_maturity
                                     ? kernel * (suit / 1000.0)
                                     : kernel * s.propagules[static_cast<std::size_t>(age[at] - s.ini_maturity)] *
                                           (suit / 1000.0);
                const double rnd = rng.unif01();
                if ((rnd < p || p == 1.0) && !(use_barrier && intersects_barrier(i, j, k, l, bar))) return true;
            }
        }
        return false;
    }
};

}  // namespace

MigClimRun migclim(i64 ny, i64 nx, const i32* initial, const i32* suitability, i64 n_periods, const i32* barriers,
                   const MigClimSettings& s) {
    const i64 n = ny * nx;
    Automaton a{ny, nx, s, CrtRand{s.seed}, barriers != nullptr};
    std::vector<i32> cur(initial, initial + n);
    std::vector<i32> bar = barriers ? std::vector<i32>(barriers, barriers + n) : std::vector<i32>(n, 0);
    // mcFilterMatrix(barriers, currentState, true, false, true) and, with
    // barriers, (currentState, barriers, ones)
    for (i64 k = 0; k < n; ++k) {
        if (bar[k] < 0) bar[k] = 0;
        if (cur[k] == MIGCLIM_NO_DATA) bar[k] = MIGCLIM_NO_DATA;
    }
    if (a.use_barrier)
        for (i64 k = 0; k < n; ++k)
            if (bar[k] == 1) cur[k] = 0;
    std::vector<i32> age(n);
    for (i64 k = 0; k < n; ++k) age[k] = cur[k] == 1 ? s.full_maturity : 0;
    std::vector<i32> no_disp = cur;
    const i32 initial_count = static_cast<i32>(std::count(cur.begin(), cur.end(), 1));
    i32 absent = static_cast<i32>(std::count(cur.begin(), cur.end(), 0));
    i32 colonized = initial_count, no_disp_count = initial_count, univ = initial_count;
    MigClimRun run;
    run.stats = {0, 0, 1, univ, no_disp_count, colonized, absent, 0, 0, 0};
    std::vector<i32> hs(n);
    for (i64 e = 0; e < n_periods; ++e) {
        const i32 env = static_cast<i32>(e) + 1;
        std::copy(suitability + e * n, suitability + (e + 1) * n, hs.begin());
        if (s.threshold > 0)
            for (auto& v : hs) v = v < s.threshold ? 0 : 1000;
        for (i64 k = 0; k < n; ++k) {
            if (hs[k] < 0) hs[k] = 0;
            if (bar[k] == 1) hs[k] = 0;
            if (bar[k] == MIGCLIM_NO_DATA) hs[k] = MIGCLIM_NO_DATA;
        }
        i32 loop_id = env * 100;
        univ = static_cast<i32>(std::count_if(hs.begin(), hs.end(), [](i32 v) { return v > 0; }));
        if (no_disp_count > 0)
            for (i64 k = 0; k < n; ++k)
                if (no_disp[k] == 1 && hs[k] == 0) {
                    no_disp[k] = 0;
                    --no_disp_count;
                }
        i32 step_decolonized = 0;
        for (i64 k = 0; k < n; ++k)
            if (hs[k] == 0 && cur[k] > 0) {
                cur[k] = 29900;
                ++step_decolonized;
            }
        for (int d = 1; d <= s.steps; ++d) {
            ++loop_id;
            i32 step_colonized = 0, step_ldd = 0;
            if (d > 1) step_decolonized = 0;
            for (i64 i = 0; i < ny; ++i)
                for (i64 j = 0; j < nx; ++j) {
                    const i64 k = i * nx + j;
                    if (hs[k] > 0 && cur[k] <= 0 &&
                        a.source_cell(i, j, cur.data(), age.data(), loop_id, hs[k], bar.data())) {
                        cur[k] = loop_id;
                        ++step_colonized;
                        age[k] = 0;
                    }
                }
            if (s.ldd_frequency > 0.0) {
                for (i64 i = 0; i < ny; ++i)
                    for (i64 j = 0; j < nx; ++j) {
                        const i64 k = i * nx + j;
                        if (cur[k] <= 0 || cur[k] == loop_id || age[k] < s.ini_maturity) continue;
                        const double p =
                            age[k] >= s.full_maturity
                                ? s.ldd_frequency
                                : s.ldd_frequency * s.propagules[static_cast<std::size_t>(age[k] - s.ini_maturity)];
                        if (!(a.rng.unif01() < p || p == 1.0)) continue;
                        // mcRandomPixel (MigClim's 2 pi, to six decimals)
                        const double dist =
                            (a.rng.unif01() * static_cast<double>(s.ldd_max - s.ldd_min)) + s.ldd_min;
                        const double angle = a.rng.unif01() * 6.283185;
                        const i64 r = static_cast<i64>(dist * std::cos(angle)) + i;
                        const i64 c = static_cast<i64>(dist * std::sin(angle)) + j;
                        // mcSinkCellCheck
                        if (r < 0 || r >= ny || c < 0 || c >= nx) continue;
                        const i64 t = r * nx + c;
                        if (cur[t] > 0 || hs[t] == 0) continue;
                        if (a.rng.unif01() * 1000.0 > hs[t]) continue;
                        cur[t] = loop_id;
                        ++step_colonized;
                        ++step_ldd;
                        age[t] = 0;
                    }
            }
            for (i64 k = 0; k < n; ++k) {
                if (cur[k] > 0) ++age[k];
                if (cur[k] >= 29900) ++cur[k];
            }
            colonized = colonized + step_colonized - step_decolonized;
            absent = absent - step_colonized + step_decolonized;
            const i32 row[10] = {env,        d,         loop_id,        univ,             no_disp_count,
                                 colonized, absent, step_colonized, step_decolonized, step_ldd};
            run.stats.insert(run.stats.end(), row, row + 10);
            run.ids.push_back(loop_id);
            run.steps.insert(run.steps.end(), cur.begin(), cur.end());
        }
        // the temporarily resilient cells at the end of the period (MigClim's
        // dispSteps - loopID - 1)
        for (i64 k = 0; k < n; ++k)
            if (cur[k] >= 29900) {
                cur[k] = s.steps - loop_id - 1;
                age[k] = 0;
            }
    }
    for (i64 k = 0; k < n; ++k)
        if (hs[k] > 0 && cur[k] <= 0) cur[k] = 30000;
    run.fin = std::move(cur);
    return run;
}

namespace {

// R's `^` (R_POW: a square multiplied, a power of 1 or of a base 1 exactly 1).
double rpow(double x, double y) {
    if (y == 2.0) return x * x;
    if (x == 1.0 || y == 0.0) return 1.0;
    return std::pow(x, y);
}

}  // namespace

std::vector<std::uint8_t> metapopulation(const double* x, const double* y, const double* area,
                                         const std::uint8_t* occupied, i64 n, i64 steps, const IfmModel& m,
                                         std::int32_t seed) {
    nlm::RRng rng(seed);
    std::vector<double> ab(n), ax(n), ac(n), c(n), e(n), u(n);
    for (i64 i = 0; i < n; ++i) {
        ab[i] = rpow(area[i], m.b);
        ax[i] = rpow(area[i], m.x);
        ac[i] = m.connectivity == 1 ? rpow(area[i], m.c1) : 1.0;
    }
    std::vector<std::uint8_t> occ(occupied, occupied + n), out;
    out.reserve(static_cast<std::size_t>(steps * n));
    for (i64 t = 0; t < steps; ++t) {
        for (i64 i = 0; i < n; ++i) {
            double s = 0.0;
            for (i64 j = 0; j < n; ++j) {
                if (!occ[j] || j == i) continue;
                const double dx = x[i] - x[j], dy = y[i] - y[j];
                const double d = std::sqrt(dx * dx + dy * dy);
                const double k = m.kernel == 0 ? std::exp(-m.alpha * d) : 1.0 / (1.0 + m.alpha * rpow(d, m.beta));
                s += k * ab[j];
            }
            if (m.connectivity == 1) s *= ac[i];
            double ci;
            if (m.colonisation == 0)
                ci = rpow(s, 2.0) / (rpow(s, 2.0) + rpow(m.y, 2.0));
            else if (m.colonisation == 1)
                ci = 1.0 - std::exp(-m.y * s);
            else
                ci = rpow(s, m.z) / (rpow(s, m.z) + (1.0 / m.c2));
            double ei;
            if (m.extinction == 0)
                ei = std::min(m.e / ax[i], 1.0);
            else if (m.extinction == 1)
                ei = 1.0 - std::exp((-m.e) / ax[i]);
            else
                ei = std::min((m.e / ax[i]) * rpow(1.0 - ci, m.r), 1.0);
            c[i] = ci;
            e[i] = ei;
        }
        // spom: an occupied patch lost with probability (1 - C) E, an empty
        // one colonised with C
        rng.runif(u.data(), n, 0.0, 1.0);
        for (i64 i = 0; i < n; ++i) {
            const double cond = occ[i] ? (1.0 - c[i]) * e[i] : c[i];
            if (u[i] < cond) occ[i] = !occ[i];
        }
        out.insert(out.end(), occ.begin(), occ.end());
    }
    return out;
}

}  // namespace sdm
