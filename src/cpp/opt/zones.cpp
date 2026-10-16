// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/zones.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "opt/bundle.hpp"

namespace opt {

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kFeasTol = 1e-9;   // row activity vs right-hand side (relative to max(1, |rhs|))
constexpr double kDualTol = 1e-7;   // reduced costs (HiGHS's dual feasibility tolerance)
constexpr int kMaxSiftRounds = 500;
constexpr i64 kWideMip = 100000;    // columns above which HiGHS's MIP ignores the time limit for seconds
constexpr i64 kBlock = 1 << 13;     // units per block of the parallel sums (fixed: the same bits on any threads)

double row_tol(double rhs) { return kFeasTol * std::max(1.0, std::abs(rhs)); }

double relative_gap(double upper, double lower) {
    if (!std::isfinite(upper) || !std::isfinite(lower)) return kInf;
    const double g = upper - lower;
    if (g <= 0) return 0.0;
    return g / std::max(std::abs(upper), 1e-10);
}

// The model by columns: the target rows over the unit columns, and the units' GUB rows.
struct ZoneCover {
    i64 n = 0, nz = 0, ncol = 0, nrow = 0;   // units, zones, unit columns, target rows kept
    std::vector<i64> col_start, row_index;    // CSC of the target rows
    std::vector<double> value;
    std::vector<double> c, lb, ub, rhs;
    std::vector<std::uint8_t> integer;
    std::vector<i64> row_of;                  // model row of every kept target row
    std::vector<i64> gub_of;                  // model row of every unit's GUB row
    bool mandatory = false;                   // GUB rows "= 1"
    bool binary = true;                       // every column binary (0 / 1)
    double offset = 0.0;
};

// The classification of the model's rows; false when the model has another form.
bool classify(const Model& m, i64 n, ZoneCover* out) {
    if (m.maximize || n <= 0 || m.ncol % n != 0 || m.ncol == 0) return false;
    const i64 nz = m.ncol / n;
    for (i64 j = 0; j < m.ncol; ++j)
        if (!std::isfinite(m.lb[j]) || !std::isfinite(m.ub[j]) || m.lb[j] < 0 || m.ub[j] > 1) return false;
    std::vector<i64> gub(static_cast<std::size_t>(n), -1);
    std::vector<i64> targets;
    int mand = -1;  // -1 unknown, 0 "<=", 1 "="
    for (i64 r = 0; r < m.nrow; ++r) {
        const i64 a = m.start[r], b = m.start[r + 1];
        const bool upper_one = m.row_hi[r] == 1.0 && (std::isinf(m.row_lo[r]) || m.row_lo[r] == 1.0);
        if (upper_one && b - a == nz) {  // a GUB row: the nz columns of one unit, all 1
            const i64 i = m.index[a] % n;
            std::vector<std::uint8_t> seen(static_cast<std::size_t>(nz), 0);
            bool ok = true;
            for (i64 k = a; k < b && ok; ++k) {
                const i64 j = m.index[k];
                ok = m.value[k] == 1.0 && j % n == i && !seen[j / n];
                if (ok) seen[j / n] = 1;
            }
            if (ok && gub[i] < 0) {
                const int mm = std::isinf(m.row_lo[r]) ? 0 : 1;
                if (mand >= 0 && mm != mand) return false;
                mand = mm;
                gub[i] = r;
                continue;
            }
        }
        if (!std::isinf(m.row_hi[r]) || !std::isfinite(m.row_lo[r])) return false;
        for (i64 k = a; k < b; ++k)
            if (m.value[k] < 0) return false;
        targets.push_back(r);
    }
    for (i64 i = 0; i < n; ++i)
        if (gub[i] < 0) return false;
    if (out == nullptr) return true;
    ZoneCover& p = *out;
    p.n = n;
    p.nz = nz;
    p.ncol = m.ncol;
    p.mandatory = mand == 1;
    p.gub_of = gub;
    p.c = m.obj;
    p.lb = m.lb;
    p.ub = m.ub;
    p.integer = m.integer;
    p.offset = m.offset;
    for (i64 j = 0; j < m.ncol; ++j) {
        if (!m.integer[j]) p.binary = false;
        else if (!(m.lb[j] == 0.0 || m.lb[j] == 1.0) || !(m.ub[j] == 0.0 || m.ub[j] == 1.0)) return false;
    }
    // the target rows with something to meet (rhs <= 0 rows hold with A >= 0, x >= 0)
    std::vector<i64> keep;
    for (const i64 r : targets)
        if (m.row_lo[r] > row_tol(m.row_lo[r])) keep.push_back(r);
    p.nrow = static_cast<i64>(keep.size());
    p.row_of = keep;
    for (const i64 r : keep) p.rhs.push_back(m.row_lo[r]);
    p.col_start.assign(static_cast<std::size_t>(m.ncol + 1), 0);
    for (i64 q = 0; q < p.nrow; ++q) {
        const i64 r = keep[q];
        for (i64 k = m.start[r]; k < m.start[r + 1]; ++k) ++p.col_start[m.index[k] + 1];
    }
    for (i64 j = 0; j < m.ncol; ++j) p.col_start[j + 1] += p.col_start[j];
    p.row_index.resize(static_cast<std::size_t>(p.col_start.back()));
    p.value.resize(p.row_index.size());
    std::vector<i64> pos(p.col_start.begin(), p.col_start.end() - 1);
    for (i64 q = 0; q < p.nrow; ++q) {
        const i64 r = keep[q];
        for (i64 k = m.start[r]; k < m.start[r + 1]; ++k) {
            const i64 j = m.index[k];
            p.row_index[pos[j]] = q;
            p.value[pos[j]++] = m.value[k];
        }
    }
    return true;
}

// The unit's best shares at reduced costs d (its zones' columns): the least sum d_z x_z with
// lb <= x <= ub and sum x <= 1 (= 1 when mandatory), filled from the lower bounds by increasing d
// (only d < 0 without mandatory allocation); ties by zone. Returns the value, or +inf when the unit
// cannot be allocated (mandatory, the upper bounds summing to less than 1, or lower bounds above 1).
double unit_choice(const ZoneCover& p, i64 i, const double* d, double* x) {
    const i64 nz = p.nz, n = p.n;
    double cap = 1.0, value = 0.0;
    for (i64 z = 0; z < nz; ++z) {
        const i64 j = z * n + i;
        x[z] = p.lb[j];
        cap -= p.lb[j];
        value += d[z] * p.lb[j];
    }
    if (cap < -1e-12) return kInf;
    // zones by increasing reduced cost (insertion sort: few zones), ties by zone
    i64 order[64];
    i64 k = 0;
    for (i64 z = 0; z < nz; ++z) {
        i64 q = k++;
        while (q > 0 && d[order[q - 1]] > d[z]) {
            order[q] = order[q - 1];
            --q;
        }
        order[q] = z;
    }
    for (i64 q = 0; q < nz && cap > 1e-12; ++q) {
        const i64 z = order[q];
        if (!p.mandatory && d[z] >= 0) break;
        const i64 j = z * n + i;
        const double t = std::min(p.ub[j] - x[z], cap);
        if (t <= 0) continue;
        x[z] += t;
        cap -= t;
        value += d[z] * t;
    }
    if (p.mandatory && cap > 1e-9) return kInf;
    return value;
}

struct Engine {
    const ZoneCover& p;
    int threads;
    bool verbose = false;
    bool out_of_time = false;
    std::vector<double> y;          // the targets' duals
    std::vector<double> x;          // the LP's solution (unit columns)
    double lp_objective = std::nan("");
    double lagrangian = -kInf;      // the best Lagrangian bound (without the offset)
    bool optimal = false, infeasible = false;
    bool good_enough = false;       // a solution within the gap of the Lagrangian bound: no LP needed
    // called with the Lagrangian bound and the units' shares at the duals: true ends the run
    std::function<bool(double, const std::vector<double>&)> enough;

    Engine(const ZoneCover& zc, int n_threads) : p(zc), threads(n_threads) {}

    // reduced costs of the unit's columns at duals yy
    void reduced(const std::vector<double>& yy, i64 i, double* d) const {
        for (i64 z = 0; z < p.nz; ++z) {
            const i64 j = z * p.n + i;
            double s = 0.0;
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) s += yy[p.row_index[k]] * p.value[k];
            d[z] = p.c[j] - s;
        }
    }

    // L(yy) = b'yy + sum_i (best shares of unit i), and the supergradient g = b - A x(yy); also the
    // shares in xx when given. +inf units (cannot be allocated) make the model infeasible: -inf.
    double lagrangian_at(const std::vector<double>& yy, std::vector<double>& g, std::vector<double>* xx = nullptr) const {
        const i64 n = p.n, m = p.nrow, nz = p.nz;
        const i64 nb = (n + kBlock - 1) / kBlock;
        std::vector<double> part(static_cast<std::size_t>(nb * (m + 1)), 0.0);
        if (xx != nullptr) xx->assign(static_cast<std::size_t>(p.ncol), 0.0);
        int bad = 0;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) reduction(+ : bad)
        for (i64 b = 0; b < nb; ++b) {
            double* acc = &part[static_cast<std::size_t>(b * (m + 1))];
            std::vector<double> d(static_cast<std::size_t>(nz)), s(static_cast<std::size_t>(nz));
            const i64 i1 = std::min(n, (b + 1) * kBlock);
            for (i64 i = b * kBlock; i < i1; ++i) {
                reduced(yy, i, d.data());
                const double v = unit_choice(p, i, d.data(), s.data());
                if (!std::isfinite(v)) {
                    ++bad;
                    continue;
                }
                acc[m] += v;
                for (i64 z = 0; z < nz; ++z) {
                    if (s[z] == 0.0) continue;
                    const i64 j = z * n + i;
                    if (xx != nullptr) (*xx)[j] = s[z];
                    for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) acc[p.row_index[k]] += p.value[k] * s[z];
                }
            }
        }
        g.assign(p.rhs.begin(), p.rhs.end());
        if (bad > 0) return -kInf;
        double value = 0.0;
        for (i64 r = 0; r < m; ++r) value += p.rhs[r] * yy[r];
        for (i64 b = 0; b < nb; ++b) {
            const double* acc = &part[static_cast<std::size_t>(b * (m + 1))];
            for (i64 r = 0; r < m; ++r) g[r] -= acc[r];
            value += acc[m];
        }
        return value;
    }

    // The LP on the working units, the others fixed at `fixed`; on success the duals in y and the
    // working units' shares in x.
    SolveResult solve_restricted(const std::vector<i64>& work, double time_left) {
        const i64 n = p.n, nz = p.nz, m = p.nrow;
        std::vector<double> rhs(p.rhs);
        std::vector<std::uint8_t> in(static_cast<std::size_t>(n), 0);
        for (const i64 i : work) in[i] = 1;
        for (i64 j = 0; j < p.ncol; ++j) {
            if (in[j % n] || x[j] == 0.0) continue;
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) rhs[p.row_index[k]] -= p.value[k] * x[j];
        }
        const i64 w = static_cast<i64>(work.size());
        Model sub;
        sub.nrow = m + w;
        sub.ncol = w * nz;
        // rows by CSR: count the target entries, then one GUB row per working unit
        std::vector<i64> cnt(static_cast<std::size_t>(m), 0);
        for (const i64 i : work)
            for (i64 z = 0; z < nz; ++z) {
                const i64 j = z * n + i;
                for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) ++cnt[p.row_index[k]];
            }
        sub.start.assign(static_cast<std::size_t>(sub.nrow + 1), 0);
        for (i64 r = 0; r < m; ++r) sub.start[r + 1] = sub.start[r] + cnt[r];
        for (i64 q = 0; q < w; ++q) sub.start[m + q + 1] = sub.start[m + q] + nz;
        sub.index.resize(static_cast<std::size_t>(sub.start.back()));
        sub.value.resize(sub.index.size());
        std::vector<i64> pos(sub.start.begin(), sub.start.begin() + m);
        for (i64 q = 0; q < w; ++q) {
            const i64 i = work[q];
            for (i64 z = 0; z < nz; ++z) {
                const i64 j = z * n + i, col = q * nz + z;
                for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) {
                    const i64 r = p.row_index[k];
                    sub.index[pos[r]] = col;
                    sub.value[pos[r]++] = p.value[k];
                }
                sub.index[sub.start[m + q] + z] = col;
                sub.value[sub.start[m + q] + z] = 1.0;
                sub.obj.push_back(p.c[j]);
                sub.lb.push_back(p.lb[j]);
                sub.ub.push_back(p.ub[j]);
            }
        }
        sub.integer.assign(static_cast<std::size_t>(sub.ncol), 0);
        sub.row_lo = rhs;
        sub.row_hi.assign(static_cast<std::size_t>(m), kInf);
        for (i64 q = 0; q < w; ++q) {
            sub.row_lo.push_back(p.mandatory ? 1.0 : -kInf);
            sub.row_hi.push_back(1.0);
        }
        SolveOptions o;
        o.threads = 1;
        o.time_limit = time_left;
        o.presolve = Presolve::off;  // HiGHS's LP presolve on restricted LPs costs more than it saves (cover.cpp)
        SolveResult out = solve_highs(sub, o);
        if (out.status == "time_limit") out_of_time = true;
        if (out.status == "optimal") {
            if (out.row_dual.size() != static_cast<std::size_t>(sub.nrow))
                throw std::runtime_error("the restricted LP gave no duals");
            for (i64 r = 0; r < m; ++r) y[r] = std::max(out.row_dual[r], 0.0);
            for (i64 q = 0; q < w; ++q)
                for (i64 z = 0; z < nz; ++z) x[z * n + work[q]] = out.x[q * nz + z];
        } else if (out.status != "infeasible" && out.status != "time_limit") {
            throw std::runtime_error("the restricted LP failed: " + out.status);
        }
        return out;
    }

    // how much the unit's fixed shares lose to its best shares at the duals y (>= 0), and its margin
    // between its best and its second-best option (small: the LP may split it)
    void price_unit(i64 i, double* d, double* best, double* loss, double* margin) const {
        reduced(y, i, d);
        const double v = unit_choice(p, i, d, best);
        double cur = 0.0;
        for (i64 z = 0; z < p.nz; ++z) cur += d[z] * x[z * p.n + i];
        *loss = std::isfinite(v) ? cur - v : kInf;
        // the margin: the least |reduced cost| of the free columns, against the unit's best value
        double mg = kInf;
        for (i64 z = 0; z < p.nz; ++z) {
            const i64 j = z * p.n + i;
            if (p.ub[j] <= p.lb[j]) continue;
            mg = std::min(mg, std::abs(d[z] - (p.mandatory ? v : std::min(v, 0.0))));
        }
        *margin = mg;
    }

    // every unit's loss and margin (price_unit) at the duals y
    void price_units(std::vector<double>& loss, std::vector<double>& margin) const {
        const i64 n = p.n, nz = p.nz;
#pragma omp parallel for schedule(static) num_threads(threads)
        for (i64 i = 0; i < n; ++i) {
            std::vector<double> d(static_cast<std::size_t>(nz)), b(static_cast<std::size_t>(nz));
            price_unit(i, d.data(), b.data(), &loss[i], &margin[i]);
        }
    }

    void run(Clock::time_point deadline) {
        const i64 n = p.n, nz = p.nz, m = p.nrow;
        auto time_left = [&] {
            return deadline == Clock::time_point::max()
                       ? kInf : std::chrono::duration<double>(deadline - Clock::now()).count();
        };
        y.assign(static_cast<std::size_t>(m), 0.0);
        // first duals: the bundle method on the Lagrangian (from y = the cheapest cost per amount of every row)
        for (i64 j = 0; j < p.ncol; ++j)
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) {
                const i64 r = p.row_index[k];
                const double ratio = p.c[j] > 0 && p.value[k] > 0 ? p.c[j] / p.value[k] : 0.0;
                if (ratio > 0 && (y[r] == 0.0 || ratio < y[r])) y[r] = ratio;
            }
        std::vector<double> g;
        if (m > 0) {
            int evaluations = 0;
            auto eval = [&](const std::vector<double>& yy, std::vector<double>& gg) {
                ++evaluations;
                const double v = lagrangian_at(yy, gg);
                return std::isfinite(v) ? v : -1e300;
            };
            const BundleResult br = bundle_maximize(y, eval, [](double) { return false; }, evaluations, 300, deadline);
            y = br.centre;
            if (verbose) std::fprintf(stderr, "zones bundle: %d evaluations (%s), L = %.12g\n", evaluations,
                                      br.why.c_str(), br.bound);
        }
        lagrangian = lagrangian_at(y, g, &x);
        if (!std::isfinite(lagrangian)) {
            infeasible = true;
            return;
        }
        if (enough && enough(lagrangian, x)) {
            good_enough = true;
            return;
        }
        if (m == 0) {  // nothing to meet: every unit at its cheapest shares
            optimal = true;
            lp_objective = lagrangian;
            return;
        }
        // sifting over units: the ones with the smallest margins work, the others keep their best shares
        i64 size = std::min(n, std::max<i64>(1000, 20 * m));
        std::vector<double> loss(static_cast<std::size_t>(n)), margin(static_cast<std::size_t>(n));
        auto price_all = [&] { price_units(loss, margin); };
        price_all();
        std::vector<std::uint8_t> working(static_cast<std::size_t>(n), 0);
        {
            std::vector<i64> order(static_cast<std::size_t>(n));
            std::iota(order.begin(), order.end(), i64{0});
            std::stable_sort(order.begin(), order.end(), [&](i64 a, i64 b) { return margin[a] < margin[b]; });
            for (i64 k = 0; k < size; ++k) working[order[k]] = 1;
        }
        i64 widen = size;
        for (int round = 0; round < kMaxSiftRounds; ++round) {
            const double left = time_left();
            if (left <= 0) {
                out_of_time = true;
                return;
            }
            std::vector<i64> work;
            for (i64 i = 0; i < n; ++i)
                if (working[i]) work.push_back(i);
            const SolveResult r = solve_restricted(work, left);
            if (out_of_time) return;
            if (verbose)
                std::fprintf(stderr, "zones sifting %d: %zu working units, LP %s\n", round, work.size(), r.status.c_str());
            if (r.status != "optimal") {  // infeasible: more units work
                std::vector<std::pair<double, i64>> rest;
                for (i64 i = 0; i < n; ++i)
                    if (!working[i]) rest.emplace_back(margin[i], i);
                if (rest.empty()) {
                    infeasible = true;
                    return;
                }
                const std::size_t add = std::min(rest.size(), static_cast<std::size_t>(widen));
                std::partial_sort(rest.begin(), rest.begin() + static_cast<std::ptrdiff_t>(add), rest.end());
                for (std::size_t k = 0; k < add; ++k) working[rest[k].second] = 1;
                widen *= 2;
                continue;
            }
            lp_objective = 0.0;
            for (i64 j = 0; j < p.ncol; ++j) lp_objective += p.c[j] * x[j];
            price_all();
            std::vector<std::pair<double, i64>> join;
            for (i64 i = 0; i < n; ++i) {
                if (working[i]) continue;
                double cmax = 0.0;
                for (i64 z = 0; z < nz; ++z) cmax = std::max(cmax, std::abs(p.c[z * n + i]));
                if (loss[i] > kDualTol * std::max(1.0, cmax)) join.emplace_back(-loss[i], i);
            }
            {  // the Lagrangian at the LP's duals: a valid bound whatever the round
                std::vector<double> gg, xl;
                const double l = lagrangian_at(y, gg, &xl);
                lagrangian = std::max(lagrangian, l);
                if (enough && enough(lagrangian, xl)) {
                    good_enough = true;
                    return;
                }
            }
            if (join.empty()) {
                optimal = true;
                return;
            }
            widen = size;
            const std::size_t add = std::min(join.size(), std::max(static_cast<std::size_t>(size), join.size() / 2));
            std::partial_sort(join.begin(), join.begin() + static_cast<std::ptrdiff_t>(add), join.end());
            for (std::size_t k = 0; k < add; ++k) working[join[k].second] = 1;
            // keep the working set near its size: units clearly at their best shares leave
            i64 nw = 0;
            for (i64 i = 0; i < n; ++i) nw += working[i];
            if (nw > 2 * size) {
                std::vector<std::pair<double, i64>> out;
                for (i64 i = 0; i < n; ++i)
                    if (working[i] && loss[i] <= 0 && margin[i] > kDualTol) out.emplace_back(-margin[i], i);
                const std::size_t drop = std::min(out.size(), static_cast<std::size_t>(nw - size));
                std::partial_sort(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(drop), out.end());
                for (std::size_t k = 0; k < drop; ++k) working[out[k].second] = 0;
            }
            size += size / 10;
        }
    }
};

// Activity of the target rows for unit columns x.
std::vector<double> activity(const ZoneCover& p, const std::vector<double>& x) {
    std::vector<double> act(static_cast<std::size_t>(p.nrow), 0.0);
    for (i64 j = 0; j < p.ncol; ++j)
        if (x[j] != 0.0)
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) act[p.row_index[k]] += p.value[k] * x[j];
    return act;
}

// Binary columns: every unit's zone (-1: none) from the LP's shares (the largest, ties by cost then zone;
// a locked column wins), then the targets left unmet repaired, then the units not needed out (or moved to
// a cheaper zone), the most expensive first. Empty when the repair fails.
std::vector<double> round_repair_trim(const ZoneCover& p, const std::vector<double>& lp) {
    const i64 n = p.n, nz = p.nz;
    std::vector<i64> zone(static_cast<std::size_t>(n), -1);
    for (i64 i = 0; i < n; ++i) {
        i64 best = -1;
        double bx = 0.0;
        for (i64 z = 0; z < nz; ++z) {
            const i64 j = z * n + i;
            if (p.ub[j] <= 0) continue;
            if (p.lb[j] >= 1.0) {  // locked in
                best = z;
                break;
            }
            const double v = lp[j];
            if (v > bx + 1e-12 || (best >= 0 && std::abs(v - bx) <= 1e-12 && v > 0 && p.c[j] < p.c[best * n + i])) {
                best = z;
                bx = v;
            }
        }
        if (best < 0 && p.mandatory) {  // the cheapest open zone
            for (i64 z = 0; z < nz; ++z) {
                const i64 j = z * n + i;
                if (p.ub[j] > 0 && (best < 0 || p.c[j] < p.c[best * n + i])) best = z;
            }
        }
        if (best >= 0 && lp[best * n + i] <= 1e-9 && !p.mandatory && p.lb[best * n + i] < 1.0) best = -1;
        zone[i] = best;
    }
    std::vector<double> x(static_cast<std::size_t>(p.ncol), 0.0);
    for (i64 i = 0; i < n; ++i)
        if (zone[i] >= 0) x[zone[i] * n + i] = 1.0;
    std::vector<double> act = activity(p, x);
    auto unmet = [&](i64 r) { return act[r] < p.rhs[r] - row_tol(p.rhs[r]); };
    auto locked = [&](i64 i) {
        for (i64 z = 0; z < nz; ++z)
            if (p.lb[z * n + i] >= 1.0) return true;
        return false;
    };
    // the change of the rows when unit i moves from its zone to zone z (z = -1: none)
    auto move_gain = [&](i64 i, i64 z, double* cost) -> double {
        const i64 from = zone[i];
        *cost = (z >= 0 ? p.c[z * n + i] : 0.0) - (from >= 0 ? p.c[from * n + i] : 0.0);
        double gain = 0.0;
        // rows of the new zone up, of the old one down; a met row must stay met
        static thread_local std::vector<double> delta;
        static thread_local std::vector<i64> touched;
        delta.resize(static_cast<std::size_t>(p.nrow), 0.0);
        touched.clear();
        if (z >= 0)
            for (i64 k = p.col_start[z * n + i]; k < p.col_start[z * n + i + 1]; ++k) {
                if (delta[p.row_index[k]] == 0.0) touched.push_back(p.row_index[k]);
                delta[p.row_index[k]] += p.value[k];
            }
        if (from >= 0)
            for (i64 k = p.col_start[from * n + i]; k < p.col_start[from * n + i + 1]; ++k) {
                if (delta[p.row_index[k]] == 0.0) touched.push_back(p.row_index[k]);
                delta[p.row_index[k]] -= p.value[k];
            }
        bool ok = true;
        for (const i64 r : touched) {
            const double dv = delta[r];
            const double left = p.rhs[r] - act[r];
            if (dv > 0 && left > row_tol(p.rhs[r])) gain += std::min(dv, left) / std::max(p.rhs[r], 1e-300);
            if (dv < 0 && act[r] + dv < p.rhs[r] - row_tol(p.rhs[r]) && !(act[r] < p.rhs[r] - row_tol(p.rhs[r])))
                ok = false;
            delta[r] = 0.0;
        }
        return ok ? gain : -1.0;
    };
    auto apply = [&](i64 i, i64 z) {
        const i64 from = zone[i];
        if (from >= 0) {
            x[from * n + i] = 0.0;
            for (i64 k = p.col_start[from * n + i]; k < p.col_start[from * n + i + 1]; ++k)
                act[p.row_index[k]] -= p.value[k];
        }
        if (z >= 0) {
            x[z * n + i] = 1.0;
            for (i64 k = p.col_start[z * n + i]; k < p.col_start[z * n + i + 1]; ++k) act[p.row_index[k]] += p.value[k];
        }
        zone[i] = z;
    };
    i64 open = 0;
    for (i64 r = 0; r < p.nrow; ++r) open += unmet(r);
    if (open > 0) {  // repair: the move with the most amount gained per cost (free moves first)
        using Item = std::pair<double, i64>;  // key, move (i * nz + z)
        auto key_of = [&](i64 i, i64 z) {
            double cost;
            const double gain = move_gain(i, z, &cost);
            if (gain <= 0) return kInf;
            return cost <= 0 ? -gain : cost / gain;
        };
        std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
        for (i64 i = 0; i < n; ++i) {
            if (locked(i)) continue;
            for (i64 z = 0; z < nz; ++z) {
                if (z == zone[i] || p.ub[z * n + i] <= 0) continue;
                const double k = key_of(i, z);
                if (std::isfinite(k)) heap.push({k, i * nz + z});
            }
        }
        while (open > 0) {
            if (heap.empty()) return {};
            const auto [k0, mv] = heap.top();
            heap.pop();
            const i64 i = mv / nz, z = mv % nz;
            if (zone[i] == z) continue;
            const double k = key_of(i, z);
            if (!std::isfinite(k)) continue;
            // stale key (relative to the next one's magnitude: keys of free moves are negative)
            if (!heap.empty() && k > heap.top().first + 1e-12 * std::abs(heap.top().first) + 1e-300) {
                heap.push({k, mv});
                continue;
            }
            apply(i, z);
            open = 0;
            for (i64 r = 0; r < p.nrow; ++r) open += unmet(r);
            for (i64 q = 0; q < nz; ++q)  // the unit's other moves, from its new zone
                if (q != z && p.ub[q * n + i] > 0) {
                    const double kk = key_of(i, q);
                    if (std::isfinite(kk)) heap.push({kk, i * nz + q});
                }
        }
    }
    // trim: units out (or into a cheaper zone) while every row stays met, the largest saving first; repeated
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<std::pair<double, i64>> cand;  // -saving, move
        for (i64 i = 0; i < n; ++i) {
            if (zone[i] < 0 || locked(i)) continue;
            const double here = p.c[zone[i] * n + i];
            if (!p.mandatory && here > 0) cand.emplace_back(-here, i * (nz + 1) + nz);  // out
            for (i64 z = 0; z < nz; ++z) {
                const i64 j = z * n + i;
                if (z != zone[i] && p.ub[j] > 0 && p.c[j] < here) cand.emplace_back(p.c[j] - here, i * (nz + 1) + z);
            }
        }
        std::stable_sort(cand.begin(), cand.end());
        bool changed = false;
        for (const auto& [s, mv] : cand) {
            const i64 i = mv / (nz + 1), q = mv % (nz + 1);
            const i64 z = q == nz ? -1 : q;
            if (zone[i] < 0 && z < 0) continue;
            const double here = zone[i] >= 0 ? p.c[zone[i] * n + i] : 0.0;
            const double there = z >= 0 ? p.c[z * n + i] : 0.0;
            if (there >= here || zone[i] == z) continue;
            double cost;
            if (move_gain(i, z, &cost) < 0) continue;  // a row would be left unmet
            apply(i, z);
            changed = true;
        }
        if (!changed) break;
    }
    for (i64 r = 0; r < p.nrow; ++r)
        if (unmet(r)) return {};
    return x;
}

// Whether unit columns x meet every target and GUB row and their bounds.
bool feasible(const ZoneCover& p, const std::vector<double>& x) {
    if (x.size() != static_cast<std::size_t>(p.ncol)) return false;
    for (i64 j = 0; j < p.ncol; ++j)
        if (x[j] < p.lb[j] - 1e-9 || x[j] > p.ub[j] + 1e-9) return false;
    const std::vector<double> act = activity(p, x);
    for (i64 r = 0; r < p.nrow; ++r)
        if (act[r] < p.rhs[r] - row_tol(p.rhs[r])) return false;
    for (i64 i = 0; i < p.n; ++i) {
        double s = 0.0;
        for (i64 z = 0; z < p.nz; ++z) s += x[z * p.n + i];
        if (s > 1.0 + 1e-9 || (p.mandatory && s < 1.0 - 1e-9)) return false;
    }
    return true;
}

double objective_of(const ZoneCover& p, const std::vector<double>& x) {
    double v = 0.0;
    for (i64 j = 0; j < p.ncol; ++j) v += p.c[j] * x[j];
    return v;
}

// The caller's start, if it is a feasible binary allocation of the model.
std::vector<double> start_solution(const ZoneCover& p, const Model& m, const std::vector<double>& start) {
    if (start.size() != static_cast<std::size_t>(p.ncol)) return {};
    std::vector<double> x(start);
    for (i64 j = 0; j < p.ncol; ++j) {
        double v = x[j];
        if (!std::isfinite(v)) return {};
        if (p.integer[j]) {
            if (std::abs(v - std::round(v)) > 1e-6) return {};
            v = std::round(v);
        }
        if (v < p.lb[j] - 1e-9 || v > p.ub[j] + 1e-9) return {};
        x[j] = std::clamp(v, p.lb[j], p.ub[j]);
    }
    for (i64 r = 0; r < m.nrow; ++r) {
        double a = 0.0;
        for (i64 k = m.start[r]; k < m.start[r + 1]; ++k) a += m.value[k] * x[m.index[k]];
        if (a < m.row_lo[r] - row_tol(m.row_lo[r]) || a > m.row_hi[r] + row_tol(m.row_hi[r])) return {};
    }
    return x;
}

}  // namespace

bool is_zone_covering(const Model& model, i64 n_units) { return classify(model, n_units, nullptr); }

SolveResult solve_zone_covering(const Model& model, i64 n_units, const SolveOptions& options) {
    model.check();
    ZoneCover p;
    if (!classify(model, n_units, &p)) throw std::invalid_argument("not a covering model with management zones");
    const auto t0 = Clock::now();
    const bool timed = std::isfinite(options.time_limit);
    const auto deadline = timed ? t0 + std::chrono::duration_cast<Clock::duration>(
                                           std::chrono::duration<double>(options.time_limit))
                                : Clock::time_point::max();
    auto seconds = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    SolveResult res;
    Engine e(p, std::max(1, options.threads));
    e.verbose = options.verbose;
    std::vector<double> early;  // the repaired Lagrangian solution that ended the run
    double early_obj = kInf;
    if (p.binary && !options.relaxation) {
        e.enough = [&](double bound_now, const std::vector<double>& xl) {
            std::vector<double> xr = round_repair_trim(p, xl);
            if (xr.empty()) return false;
            const double v = objective_of(p, xr);
            if (v < early_obj) {
                early_obj = v;
                early = std::move(xr);
            }
            return relative_gap(early_obj + p.offset, bound_now + p.offset) <= options.gap;
        };
    }
    e.run(deadline);
    if (e.infeasible) {
        res.status = "infeasible";
        res.seconds = seconds();
        return res;
    }
    const double lower = e.optimal ? e.lp_objective : e.lagrangian;
    double bound = lower;
    if (e.good_enough) {  // within the gap of the Lagrangian bound
        res.status = "optimal";
        res.has_solution = true;
        res.x = early;
        res.objective = early_obj + p.offset;
        res.bound = e.lagrangian + p.offset;
        res.gap = relative_gap(res.objective, res.bound);
        res.seconds = seconds();
        return res;
    }
    const bool any_integer = !options.relaxation && p.binary;
    std::vector<double> best;
    double best_obj = kInf;
    auto consider = [&](std::vector<double> x) {
        if (x.empty()) return;
        const double v = objective_of(p, x);
        if (v < best_obj) {
            best_obj = v;
            best = std::move(x);
        }
    };
    consider(start_solution(p, model, options.start));
    consider(early);
    if (!any_integer) {
        if (feasible(p, e.x)) consider(e.x);
    } else {
        if (!std::isnan(e.lp_objective) || e.optimal) consider(round_repair_trim(p, e.x));
        // the Lagrangian solution at the final duals, repaired (often better on wide models)
        std::vector<double> g, xl;
        e.lagrangian_at(e.y, g, &xl);
        consider(round_repair_trim(p, xl));
    }
    bool mip_optimal = false;
    if (!e.optimal) {  // out of time
        res.status = "time_limit";
    } else if (!any_integer) {
        mip_optimal = true;
        bound = e.lp_objective;
    } else if (relative_gap(best_obj + p.offset, bound + p.offset) > options.gap && !(timed && Clock::now() >= deadline) && !stop_asked()) {
        // Lagrangian fixing: forcing unit-zone column j to the other value raises L(y) by at least the change of
        // its unit's best shares; when that exceeds best - L, j keeps its value (exact)
        const i64 n = p.n, nz = p.nz;
        // L at the final duals (the fixings below are measured from it)
        std::vector<double> gy;
        const double ly = e.lagrangian_at(e.y, gy);
        bound = std::max(bound, ly);
        const double slack = std::isfinite(best_obj) ? best_obj - ly : kInf;
        std::vector<double> fixed(static_cast<std::size_t>(p.ncol), std::nan(""));
        {
            std::vector<double> d(static_cast<std::size_t>(nz)), s(static_cast<std::size_t>(nz));
            std::vector<double> lb2(p.lb), ub2(p.ub);
            for (i64 i = 0; i < n; ++i) {
                e.reduced(e.y, i, d.data());
                const double v = unit_choice(p, i, d.data(), s.data());
                for (i64 z = 0; z < nz; ++z) {
                    const i64 j = z * n + i;
                    if (p.lb[j] == p.ub[j]) continue;
                    // the unit's best shares with column j forced to the value it does not take
                    ZoneCover& q = const_cast<ZoneCover&>(p);
                    const double keep_lb = q.lb[j], keep_ub = q.ub[j];
                    const double other = s[z] > 0.5 ? 0.0 : 1.0;
                    q.lb[j] = other;
                    q.ub[j] = other;
                    std::vector<double> s2(static_cast<std::size_t>(nz));
                    const double v2 = unit_choice(q, i, d.data(), s2.data());
                    q.lb[j] = keep_lb;
                    q.ub[j] = keep_ub;
                    if (v2 - v > slack * (1 + 1e-9) + 1e-9) fixed[j] = s[z] > 0.5 ? 1.0 : 0.0;
                }
            }
        }
        // the sub-model: the free columns, the targets less the fixed columns' amounts, the GUB rows
        std::vector<i64> keep;
        for (i64 j = 0; j < p.ncol; ++j)
            if (std::isnan(fixed[j])) keep.push_back(j);
        const bool fits = !timed || static_cast<i64>(keep.size()) <= kWideMip;
        const double tl = timed ? std::chrono::duration<double>(deadline - Clock::now()).count() : kInf;
        if (fits && tl > 0) {
            std::vector<i64> colpos(static_cast<std::size_t>(p.ncol), -1);
            for (std::size_t w = 0; w < keep.size(); ++w) colpos[keep[w]] = static_cast<i64>(w);
            Model sub;
            sub.ncol = static_cast<i64>(keep.size());
            sub.nrow = model.nrow;
            sub.start.push_back(0);
            double fixed_obj = 0.0;
            for (i64 j = 0; j < p.ncol; ++j)
                if (!std::isnan(fixed[j])) fixed_obj += p.c[j] * fixed[j];
            for (i64 r = 0; r < model.nrow; ++r) {
                double shift = 0.0;
                for (i64 k = model.start[r]; k < model.start[r + 1]; ++k) {
                    const i64 j = model.index[k];
                    if (colpos[j] >= 0) {
                        sub.index.push_back(colpos[j]);
                        sub.value.push_back(model.value[k]);
                    } else {
                        shift += model.value[k] * fixed[j];
                    }
                }
                sub.start.push_back(static_cast<i64>(sub.index.size()));
                sub.row_lo.push_back(model.row_lo[r] - shift);
                sub.row_hi.push_back(model.row_hi[r] - shift);
            }
            std::vector<double> start;
            for (const i64 j : keep) {
                sub.obj.push_back(p.c[j]);
                sub.lb.push_back(p.lb[j]);
                sub.ub.push_back(p.ub[j]);
                sub.integer.push_back(p.integer[j]);
                if (!best.empty()) start.push_back(best[j]);
            }
            sub.offset = fixed_obj;
            SolveOptions o = options;
            o.start = start;
            o.time_limit = tl;
            o.objective_target = -kInf;
            if (sub.ncol > kWideMip) o.presolve = Presolve::off;
            const SolveResult mip = solve_highs(sub, o);
            res.nodes = mip.nodes;
            if (mip.has_solution && mip.objective < best_obj) {
                best_obj = mip.objective;
                best.assign(p.ncol, 0.0);
                for (i64 j = 0; j < p.ncol; ++j)
                    best[j] = colpos[j] >= 0 ? mip.x[colpos[j]] : fixed[j];
            }
            if (std::isfinite(mip.bound)) bound = std::max(bound, mip.bound);
            mip_optimal = mip.status == "optimal" || mip.status == "infeasible";
            if (mip.status == "infeasible") bound = std::max(bound, best_obj);
        }
    }
    res.has_solution = !best.empty();
    res.bound = bound + p.offset;
    if (res.has_solution) {
        res.x = best;
        res.objective = best_obj + p.offset;
        res.gap = relative_gap(res.objective, res.bound);
    }
    if (res.status.empty())
        res.status = res.has_solution && (res.gap <= options.gap || mip_optimal) ? "optimal" : "time_limit";
    res.seconds = seconds();
    return res;
}

}  // namespace opt
