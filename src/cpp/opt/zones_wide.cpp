// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/zones_wide.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace opt {

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kFeasTol = 1e-9;
constexpr double kDualTol = 1e-7;
constexpr int kMaxSiftRounds = 500;
constexpr i64 kWideMip = 100000;
constexpr i64 kMaxOther = 20000;

double tol_of(double v) { return kFeasTol * std::max(1.0, std::abs(v)); }

// The model in minimisation form, by columns, its GUB rows apart.
struct ZW {
    i64 n = 0, nz = 0, nu = 0, ncol = 0, nrow = 0;  // units, zones, unit columns, all columns, other rows
    std::vector<i64> col_start, row_index;          // CSC of the other rows over every column
    std::vector<double> value;
    std::vector<double> c, lb, ub;                  // c: minimisation form
    std::vector<std::uint8_t> integer;
    std::vector<double> lo, hi;                     // the other rows' bounds
    std::vector<i64> row_of;                        // their model rows
    std::vector<std::uint8_t> unit_only;            // rows without other columns
    std::vector<i64> gub_of;
    bool mandatory = false, maximize = false, binary_units = true;
    double offset = 0.0;
};

bool classify(const Model& m, i64 n, ZW* out) {
    if (n <= 0 || m.ncol < n) return false;
    std::vector<i64> gub(static_cast<std::size_t>(n), -1);
    std::vector<std::uint8_t> is_gub(static_cast<std::size_t>(m.nrow), 0);
    i64 nz = -1;
    int mand = -1;
    for (i64 r = 0; r < m.nrow; ++r) {
        const i64 a = m.start[r], b = m.start[r + 1];
        if (m.row_hi[r] != 1.0 || !(std::isinf(m.row_lo[r]) || m.row_lo[r] == 1.0) || b == a) continue;
        const i64 i = m.index[a] % n;
        const i64 k = b - a;
        bool ok = (nz < 0 || k == nz) && k * n <= m.ncol;
        std::vector<std::uint8_t> seen(static_cast<std::size_t>(ok ? k : 0), 0);
        for (i64 q = a; q < b && ok; ++q) {
            const i64 j = m.index[q];
            ok = m.value[q] == 1.0 && j % n == i && j / n < k && !seen[j / n];
            if (ok) seen[j / n] = 1;
        }
        if (!ok) continue;
        if (gub[i] >= 0) return false;  // two such rows of one unit: ambiguous
        const int mm = std::isinf(m.row_lo[r]) ? 0 : 1;
        if (mand >= 0 && mm != mand) return false;
        mand = mm;
        nz = k;
        gub[i] = r;
        is_gub[r] = 1;
    }
    if (nz < 1) return false;
    for (i64 i = 0; i < n; ++i)
        if (gub[i] < 0) return false;
    const i64 nu = nz * n;
    if (m.ncol - nu > kMaxOther) return false;
    for (i64 j = 0; j < nu; ++j)
        if (!std::isfinite(m.lb[j]) || !std::isfinite(m.ub[j]) || m.lb[j] < 0 || m.ub[j] > 1) return false;
    for (i64 r = 0; r < m.nrow; ++r)
        if (!is_gub[r] && !std::isfinite(m.row_lo[r]) && !std::isfinite(m.row_hi[r])) return false;
    if (out == nullptr) return true;
    ZW& p = *out;
    p.n = n;
    p.nz = nz;
    p.nu = nu;
    p.ncol = m.ncol;
    p.mandatory = mand == 1;
    p.maximize = m.maximize;
    p.gub_of = gub;
    p.c = m.obj;
    if (m.maximize)
        for (double& v : p.c) v = -v;
    p.lb = m.lb;
    p.ub = m.ub;
    p.integer = m.integer;
    p.offset = m.offset;
    for (i64 j = 0; j < nu; ++j) {
        if (!m.integer[j]) p.binary_units = false;
        else if (!(m.lb[j] == 0.0 || m.lb[j] == 1.0) || !(m.ub[j] == 0.0 || m.ub[j] == 1.0)) return false;
    }
    std::vector<i64> keep;
    for (i64 r = 0; r < m.nrow; ++r)
        if (!is_gub[r]) keep.push_back(r);
    p.nrow = static_cast<i64>(keep.size());
    p.row_of = keep;
    p.unit_only.assign(keep.size(), 1);
    p.col_start.assign(static_cast<std::size_t>(m.ncol + 1), 0);
    for (i64 q = 0; q < p.nrow; ++q) {
        const i64 r = keep[q];
        p.lo.push_back(m.row_lo[r]);
        p.hi.push_back(m.row_hi[r]);
        for (i64 k = m.start[r]; k < m.start[r + 1]; ++k) {
            ++p.col_start[m.index[k] + 1];
            if (m.index[k] >= nu) p.unit_only[q] = 0;
        }
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

// The unit's best shares at reduced costs d (as opt/zones.cpp: a continuous knapsack over its zones).
double unit_choice(const ZW& p, i64 i, const double* d, double* x) {
    const i64 nz = p.nz, n = p.n;
    double cap = 1.0, value = 0.0;
    for (i64 z = 0; z < nz; ++z) {
        const i64 j = z * n + i;
        x[z] = p.lb[j];
        cap -= p.lb[j];
        value += d[z] * p.lb[j];
    }
    if (cap < -1e-12) return kInf;
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

double objective_of(const ZW& p, const std::vector<double>& x) {
    double v = 0.0;
    for (i64 j = 0; j < p.ncol; ++j) v += p.c[j] * x[j];
    return v;
}

struct Engine {
    const ZW& p;
    int threads;
    bool verbose = false;
    bool out_of_time = false, optimal = false, infeasible = false;
    std::vector<double> x, y;  // all columns; the other rows' duals (HiGHS's signs: d = c - A'y)
    double lp_objective = std::nan("");

    Engine(const ZW& zw, int t) : p(zw), threads(t) {}

    void reduced(const std::vector<double>& yy, i64 i, double* d) const {
        for (i64 z = 0; z < p.nz; ++z) {
            const i64 j = z * p.n + i;
            double s = 0.0;
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) s += yy[p.row_index[k]] * p.value[k];
            d[z] = p.c[j] - s;
        }
    }

    // L(y) in minimisation form: the rows' terms, every unit's best shares, every other column at its best
    // bound (-inf when a reduced cost points to an infinite bound)
    double lagrangian(const std::vector<double>& yy) const {
        double v = 0.0;
        for (i64 r = 0; r < p.nrow; ++r) {
            if (yy[r] > 0) {
                if (!std::isfinite(p.lo[r])) return -kInf;
                v += yy[r] * p.lo[r];
            } else if (yy[r] < 0) {
                if (!std::isfinite(p.hi[r])) return -kInf;
                v += yy[r] * p.hi[r];
            }
        }
        std::vector<double> d(static_cast<std::size_t>(p.nz)), s(static_cast<std::size_t>(p.nz));
        for (i64 i = 0; i < p.n; ++i) {
            reduced(yy, i, d.data());
            const double u = unit_choice(p, i, d.data(), s.data());
            if (!std::isfinite(u)) return -kInf;
            v += u;
        }
        for (i64 j = p.nu; j < p.ncol; ++j) {
            double sum = 0.0;
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) sum += yy[p.row_index[k]] * p.value[k];
            const double dj = p.c[j] - sum;
            if (dj > 0) v += std::isfinite(p.lb[j]) ? dj * p.lb[j] : -kInf;
            else if (dj < 0) v += std::isfinite(p.ub[j]) ? dj * p.ub[j] : -kInf;
            if (!std::isfinite(v)) return -kInf;
        }
        return v;
    }

    SolveResult solve_restricted(const std::vector<i64>& work, double time_left, const std::vector<double>* scale = nullptr) {
        const i64 n = p.n, nz = p.nz, m = p.nrow, no = p.ncol - p.nu;
        std::vector<double> shift(static_cast<std::size_t>(m), 0.0);
        std::vector<std::uint8_t> in(static_cast<std::size_t>(n), 0);
        for (const i64 i : work) in[i] = 1;
        for (i64 j = 0; j < p.nu; ++j) {
            if (in[j % n] || x[j] == 0.0) continue;
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) shift[p.row_index[k]] += p.value[k] * x[j];
        }
        const i64 w = static_cast<i64>(work.size());
        std::vector<i64> cols;  // model columns of the restricted LP: working units, then the others
        for (const i64 i : work)
            for (i64 z = 0; z < nz; ++z) cols.push_back(z * n + i);
        for (i64 j = p.nu; j < p.ncol; ++j) cols.push_back(j);
        Model sub;
        sub.ncol = static_cast<i64>(cols.size());
        sub.nrow = m + w;
        std::vector<i64> cnt(static_cast<std::size_t>(m), 0);
        for (const i64 j : cols)
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) ++cnt[p.row_index[k]];
        sub.start.assign(static_cast<std::size_t>(sub.nrow + 1), 0);
        for (i64 r = 0; r < m; ++r) sub.start[r + 1] = sub.start[r] + cnt[r];
        for (i64 q = 0; q < w; ++q) sub.start[m + q + 1] = sub.start[m + q] + nz;
        sub.index.resize(static_cast<std::size_t>(sub.start.back()));
        sub.value.resize(sub.index.size());
        std::vector<i64> pos(sub.start.begin(), sub.start.begin() + m);
        for (std::size_t s = 0; s < cols.size(); ++s) {
            const i64 j = cols[s];
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) {
                const i64 r = p.row_index[k];
                sub.index[pos[r]] = static_cast<i64>(s);
                sub.value[pos[r]++] = p.value[k];
            }
            sub.obj.push_back(p.c[j]);
            sub.lb.push_back(p.lb[j]);
            sub.ub.push_back(p.ub[j]);
        }
        for (i64 q = 0; q < w; ++q)
            for (i64 z = 0; z < nz; ++z) {
                sub.index[sub.start[m + q] + z] = q * nz + z;
                sub.value[sub.start[m + q] + z] = 1.0;
            }
        sub.integer.assign(static_cast<std::size_t>(sub.ncol), 0);
        for (i64 r = 0; r < m; ++r) {
            const double f = scale != nullptr ? (*scale)[r] : 1.0;
            sub.row_lo.push_back(p.lo[r] * f - shift[r]);
            sub.row_hi.push_back(p.hi[r] * f - shift[r]);
        }
        for (i64 q = 0; q < w; ++q) {
            sub.row_lo.push_back(p.mandatory ? 1.0 : -kInf);
            sub.row_hi.push_back(1.0);
        }
        (void)no;
        SolveOptions o;
        o.threads = 1;
        o.time_limit = time_left;
        o.presolve = Presolve::off;
        SolveResult out = solve_highs(sub, o);
        if (out.status == "time_limit") out_of_time = true;
        if (out.status == "optimal") {
            if (out.row_dual.size() != static_cast<std::size_t>(sub.nrow))
                throw std::runtime_error("the restricted LP gave no duals");
            for (i64 r = 0; r < m; ++r) y[r] = out.row_dual[r];
            if (scale == nullptr)
                for (std::size_t s = 0; s < cols.size(); ++s) x[cols[s]] = out.x[s];
        } else if (out.status != "infeasible" && out.status != "time_limit") {
            throw std::runtime_error("the restricted LP failed: " + out.status);
        }
        return out;
    }

    // every unit's loss (its shares against its best shares at the duals) and margin (the least distance of
    // a reduced cost from the unit's best value: small, the LP may change the unit)
    void price(std::vector<double>& loss, std::vector<double>* margin = nullptr) const {
        const i64 n = p.n, nz = p.nz;
#pragma omp parallel for schedule(static) num_threads(threads)
        for (i64 i = 0; i < n; ++i) {
            std::vector<double> d(static_cast<std::size_t>(nz)), b(static_cast<std::size_t>(nz));
            reduced(y, i, d.data());
            const double v = unit_choice(p, i, d.data(), b.data());
            double cur = 0.0;
            for (i64 z = 0; z < nz; ++z) cur += d[z] * x[z * n + i];
            loss[i] = std::isfinite(v) ? cur - v : kInf;
            if (margin != nullptr) {
                double mg = kInf;
                for (i64 z = 0; z < nz; ++z) {
                    const i64 j = z * n + i;
                    if (p.ub[j] > p.lb[j]) mg = std::min(mg, std::abs(d[z] - (p.mandatory ? v : std::min(v, 0.0))));
                }
                (*margin)[i] = mg;
            }
        }
    }

    void run(Clock::time_point deadline) {
        const i64 n = p.n, nz = p.nz, m = p.nrow;
        auto time_left = [&] {
            return deadline == Clock::time_point::max()
                       ? kInf : std::chrono::duration<double>(deadline - Clock::now()).count();
        };
        x.assign(static_cast<std::size_t>(p.ncol), 0.0);
        y.assign(static_cast<std::size_t>(m), 0.0);
        // units outside the working set start at their lower bounds (with mandatory allocation: their
        // cheapest zones); the other columns are always in the LP
        {
            std::vector<double> d(static_cast<std::size_t>(nz)), s(static_cast<std::size_t>(nz));
            for (i64 i = 0; i < n; ++i) {
                for (i64 z = 0; z < nz; ++z) d[z] = p.mandatory ? p.c[z * n + i] : 1.0;
                unit_choice(p, i, d.data(), s.data());
                for (i64 z = 0; z < nz; ++z) x[z * n + i] = s[z];
            }
        }
        i64 size = std::min(n, std::max<i64>(1000, 20 * m));
        std::vector<std::uint8_t> working(static_cast<std::size_t>(n), 0);
        std::vector<double> loss(static_cast<std::size_t>(n), 0.0), margin(static_cast<std::size_t>(n), 0.0);
        if (n > 4 * size) {
            // wide: first duals from the LP on an evenly spaced sample of 2% of the units, its rows' bounds
            // scaled by the share of their amounts the sample holds; the working set: the units of smallest margin
            const i64 ns = std::min(n, std::max<i64>(size, n / 50));
            std::vector<i64> sample;
            for (i64 k = 0; k < ns; ++k) sample.push_back(k * n / ns);
            std::vector<double> all(static_cast<std::size_t>(m), 0.0), in(all);
            std::vector<std::uint8_t> sampled(static_cast<std::size_t>(n), 0);
            for (const i64 i : sample) sampled[i] = 1;
            for (i64 j = 0; j < p.nu; ++j)
                for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) {
                    const double v = std::abs(p.value[k]) * p.ub[j];
                    all[p.row_index[k]] += v;
                    if (sampled[j % n]) in[p.row_index[k]] += v;
                }
            std::vector<double> scale(static_cast<std::size_t>(m), 1.0);
            for (i64 r = 0; r < m; ++r)
                if (all[r] > 0) scale[r] = in[r] / all[r];
            const SolveResult rs = solve_restricted(sample, time_left(), &scale);
            if (out_of_time) return;
            if (rs.status != "optimal") std::fill(y.begin(), y.end(), 0.0);
            price(loss, &margin);
            std::vector<i64> order(static_cast<std::size_t>(n));
            std::iota(order.begin(), order.end(), i64{0});
            std::stable_sort(order.begin(), order.end(), [&](i64 a, i64 b) { return margin[a] < margin[b]; });
            // outside the working set, every unit at its best shares for these duals
            {
                std::vector<double> d(static_cast<std::size_t>(nz)), sh(static_cast<std::size_t>(nz));
                for (i64 i = 0; i < n; ++i) {
                    reduced(y, i, d.data());
                    if (!std::isfinite(unit_choice(p, i, d.data(), sh.data()))) continue;
                    for (i64 z = 0; z < nz; ++z) x[z * n + i] = sh[z];
                }
            }
            for (i64 k = 0; k < size; ++k) working[order[k]] = 1;
        } else {
            for (i64 k = 0; k < size; ++k) working[k * n / size] = 1;  // evenly spaced
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
            if (verbose) std::fprintf(stderr, "zones wide sifting %d: %zu working units, LP %s\n", round, work.size(),
                                      r.status.c_str());
            if (r.status != "optimal") {
                // more units work: first those held in (their shares may break the rows), the smallest
                // margins first, then any
                std::vector<std::pair<double, i64>> held, rest;
                for (i64 i = 0; i < n; ++i) {
                    if (working[i]) continue;
                    bool in = false;
                    for (i64 z = 0; z < nz && !in; ++z) in = x[z * n + i] != p.lb[z * n + i];
                    (in ? held : rest).emplace_back(margin[i], i);
                }
                if (held.empty() && rest.empty()) {
                    infeasible = true;
                    return;
                }
                auto& pool = held.empty() ? rest : held;
                const std::size_t add = std::min(pool.size(), static_cast<std::size_t>(widen));
                std::partial_sort(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(add), pool.end());
                for (std::size_t k = 0; k < add; ++k) working[pool[k].second] = 1;
                widen *= 2;
                continue;
            }
            lp_objective = objective_of(p, x);
            price(loss, &margin);
            std::vector<std::pair<double, i64>> join;
            for (i64 i = 0; i < n; ++i) {
                if (working[i]) continue;
                double cmax = 0.0;
                for (i64 z = 0; z < nz; ++z) cmax = std::max(cmax, std::abs(p.c[z * n + i]));
                if (loss[i] > kDualTol * std::max(1.0, cmax)) join.emplace_back(-loss[i], i);
            }
            if (join.empty()) {
                optimal = true;
                return;
            }
            widen = size;
            const std::size_t add = std::min(join.size(), static_cast<std::size_t>(size));
            std::partial_sort(join.begin(), join.begin() + static_cast<std::ptrdiff_t>(add), join.end());
            for (std::size_t k = 0; k < add; ++k) working[join[k].second] = 1;
            i64 nw = 0;
            for (i64 i = 0; i < n; ++i) nw += working[i];
            if (nw > 2 * size) {  // units the LP leaves at their best shares go back out
                std::vector<i64> out;
                for (i64 i = 0; i < n; ++i)
                    if (working[i] && loss[i] <= 0) {
                        bool integral = true;
                        for (i64 z = 0; z < nz && integral; ++z) {
                            const double v = x[z * n + i];
                            integral = v == p.lb[z * n + i] || v == p.ub[z * n + i];
                        }
                        if (integral) out.push_back(i);
                    }
                const i64 drop = std::min<i64>(static_cast<i64>(out.size()), nw - size);
                for (i64 k = 0; k < drop; ++k) working[out[k]] = 0;
            }
            size += size / 10;
        }
    }
};

// The other columns re-optimised for the units' values xu (a MIP on them alone); the full solution, or
// empty when the rows cannot be met.
std::vector<double> reoptimise(const ZW& p, std::vector<double> xu, double time_left) {
    const i64 m = p.nrow, no = p.ncol - p.nu;
    std::vector<double> shift(static_cast<std::size_t>(m), 0.0);
    for (i64 j = 0; j < p.nu; ++j)
        if (xu[j] != 0.0)
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) shift[p.row_index[k]] += p.value[k] * xu[j];
    for (i64 r = 0; r < m; ++r)  // rows of the units alone: met as they are, or not at all
        if (p.unit_only[r] && (shift[r] < p.lo[r] - tol_of(p.lo[r]) || shift[r] > p.hi[r] + tol_of(p.hi[r])))
            return {};
    if (no == 0) return xu;
    Model sub;
    sub.ncol = no;
    std::vector<i64> rows;
    for (i64 r = 0; r < m; ++r)
        if (!p.unit_only[r]) rows.push_back(r);
    sub.nrow = static_cast<i64>(rows.size());
    std::vector<i64> rpos(static_cast<std::size_t>(m), -1);
    for (std::size_t q = 0; q < rows.size(); ++q) rpos[rows[q]] = static_cast<i64>(q);
    std::vector<i64> cnt(rows.size() + 1, 0);
    for (i64 j = p.nu; j < p.ncol; ++j)
        for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) ++cnt[rpos[p.row_index[k]] + 1];
    for (std::size_t q = 0; q < rows.size(); ++q) cnt[q + 1] += cnt[q];
    sub.start = cnt;
    sub.index.resize(static_cast<std::size_t>(cnt.back()));
    sub.value.resize(sub.index.size());
    std::vector<i64> pos(cnt.begin(), cnt.end() - 1);
    for (i64 j = p.nu; j < p.ncol; ++j) {
        for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) {
            const i64 q = rpos[p.row_index[k]];
            sub.index[pos[q]] = j - p.nu;
            sub.value[pos[q]++] = p.value[k];
        }
        sub.obj.push_back(p.c[j]);
        sub.lb.push_back(p.lb[j]);
        sub.ub.push_back(p.ub[j]);
        sub.integer.push_back(p.integer[j]);
    }
    for (const i64 r : rows) {
        sub.row_lo.push_back(p.lo[r] - shift[r]);
        sub.row_hi.push_back(p.hi[r] - shift[r]);
    }
    SolveOptions o;
    o.threads = 1;
    o.gap = 0.0;
    o.time_limit = time_left;
    const SolveResult r = solve_highs(sub, o);
    if (!r.has_solution) return {};
    for (i64 j = 0; j < no; ++j) xu[p.nu + j] = r.x[j];
    return xu;
}

}  // namespace

bool is_zone_wide(const Model& model, i64 n_units) { return classify(model, n_units, nullptr); }

SolveResult solve_zone_wide(const Model& model, i64 n_units, const SolveOptions& options) {
    model.check();
    ZW p;
    if (!classify(model, n_units, &p)) throw std::invalid_argument("not a wide model with management zones");
    const auto t0 = Clock::now();
    const bool timed = std::isfinite(options.time_limit);
    const auto deadline = timed ? t0 + std::chrono::duration_cast<Clock::duration>(
                                           std::chrono::duration<double>(options.time_limit))
                                : Clock::time_point::max();
    auto seconds = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    auto left = [&] { return timed ? std::chrono::duration<double>(deadline - Clock::now()).count() : kInf; };
    const double sgn = p.maximize ? -1.0 : 1.0;  // original objective = sgn * (minimisation form) + offset
    SolveResult res;
    Engine e(p, std::max(1, options.threads));
    e.verbose = options.verbose;
    e.run(deadline);
    if (e.infeasible) {
        res.status = "infeasible";
        res.seconds = seconds();
        return res;
    }
    const i64 n = p.n, nz = p.nz;
    double bound = e.optimal ? e.lp_objective : -kInf;
    const bool any_integer = !options.relaxation &&
                             std::any_of(p.integer.begin(), p.integer.end(), [](std::uint8_t t) { return t != 0; });
    std::vector<double> best;
    double best_obj = kInf;
    auto consider = [&](std::vector<double> xx) {
        if (xx.empty()) return;
        const double v = objective_of(p, xx);
        if (v < best_obj) {
            best_obj = v;
            best = std::move(xx);
        }
    };
    if (e.optimal && !any_integer) {
        consider(e.x);
    } else if (!std::isnan(e.lp_objective)) {
        // the fractional units out (mandatory: their largest share), the other columns re-optimised
        std::vector<double> xu(e.x.begin(), e.x.end());
        std::fill(xu.begin() + p.nu, xu.end(), 0.0);
        if (p.binary_units) {
            for (i64 i = 0; i < n; ++i) {
                bool frac = false;
                i64 top = -1;
                for (i64 z = 0; z < nz; ++z) {
                    const double v = xu[z * n + i];
                    if (v > 1e-9 && v < 1 - 1e-9) frac = true;
                    if (top < 0 || v > xu[top * n + i]) top = z;
                }
                if (!frac) {
                    for (i64 z = 0; z < nz; ++z) xu[z * n + i] = std::round(xu[z * n + i]);
                    continue;
                }
                for (i64 z = 0; z < nz; ++z) xu[z * n + i] = p.lb[z * n + i];
                if (p.mandatory) xu[top * n + i] = 1.0;
            }
        }
        std::vector<double> down = reoptimise(p, xu, std::max(1.0, left()));
        consider(down);
        if (p.binary_units) {  // rounded up: every fractional unit into its largest share
            std::vector<double> xup(e.x.begin(), e.x.end());
            std::fill(xup.begin() + p.nu, xup.end(), 0.0);
            for (i64 i = 0; i < n; ++i) {
                i64 top = -1;
                bool frac = false;
                for (i64 z = 0; z < nz; ++z) {
                    const double v = xup[z * n + i];
                    if (v > 1e-9 && v < 1 - 1e-9) frac = true;
                    if (top < 0 || v > xup[top * n + i]) top = z;
                }
                for (i64 z = 0; z < nz; ++z)
                    xup[z * n + i] = frac ? (z == top ? 1.0 : p.lb[z * n + i]) : std::round(xup[z * n + i]);
            }
            consider(reoptimise(p, xup, std::max(1.0, left())));
        }
        // then units in, the most attractive reduced cost first, while the rows of units alone allow it
        if (p.binary_units && !p.mandatory) {
            std::vector<double> act(static_cast<std::size_t>(p.nrow), 0.0);
            for (i64 j = 0; j < p.nu; ++j)
                if (xu[j] != 0.0)
                    for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) act[p.row_index[k]] += p.value[k] * xu[j];
            std::vector<std::pair<double, i64>> cand;
            std::vector<double> d(static_cast<std::size_t>(nz));
            std::vector<std::uint8_t> used(static_cast<std::size_t>(n), 0);
            for (i64 i = 0; i < n; ++i) {
                for (i64 z = 0; z < nz; ++z) used[i] |= xu[z * n + i] > 0.5;
                if (used[i]) continue;
                e.reduced(e.y, i, d.data());
                for (i64 z = 0; z < nz; ++z)
                    if (d[z] < 0 && p.ub[z * n + i] > 0) cand.emplace_back(d[z], z * n + i);
            }
            std::stable_sort(cand.begin(), cand.end());
            bool added = false;
            for (const auto& [dv, j] : cand) {
                const i64 i = j % n;
                if (used[i]) continue;
                bool ok = true;
                for (i64 k = p.col_start[j]; k < p.col_start[j + 1] && ok; ++k) {
                    const i64 r = p.row_index[k];
                    if (!p.unit_only[r]) continue;
                    const double a = act[r] + p.value[k];
                    ok = a <= p.hi[r] + tol_of(p.hi[r]) && (p.value[k] >= 0 || a >= p.lo[r] - tol_of(p.lo[r]));
                }
                if (!ok) continue;
                for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) act[p.row_index[k]] += p.value[k];
                xu[j] = 1.0;
                used[i] = 1;
                added = true;
            }
            if (added) consider(reoptimise(p, xu, std::max(1.0, left())));
        }
    }
    bool mip_optimal = !any_integer && e.optimal;
    if (any_integer && e.optimal &&
        (!std::isfinite(best_obj) || std::abs(best_obj - bound) > options.gap * std::max(std::abs(best_obj), 1e-10)) &&
        (!timed || left() > 0)) {
        // Lagrangian fixing of unit columns at the LP's duals, then HiGHS's MIP on the rest
        const double ly = e.lagrangian(e.y);
        if (std::isfinite(ly)) bound = std::max(bound, ly);
        const double slack = std::isfinite(ly) ? best_obj - ly : kInf;
        std::vector<double> fixed(static_cast<std::size_t>(p.ncol), std::nan(""));
        if (std::isfinite(slack) && p.binary_units) {
            ZW& q = const_cast<ZW&>(p);
            std::vector<double> d(static_cast<std::size_t>(nz)), s(static_cast<std::size_t>(nz)),
                s2(static_cast<std::size_t>(nz));
            for (i64 i = 0; i < n; ++i) {
                e.reduced(e.y, i, d.data());
                const double v = unit_choice(p, i, d.data(), s.data());
                for (i64 z = 0; z < nz; ++z) {
                    const i64 j = z * n + i;
                    if (p.lb[j] == p.ub[j]) continue;
                    const double other = s[z] > 0.5 ? 0.0 : 1.0;
                    const double klb = q.lb[j], kub = q.ub[j];
                    q.lb[j] = other;
                    q.ub[j] = other;
                    const double v2 = unit_choice(q, i, d.data(), s2.data());
                    q.lb[j] = klb;
                    q.ub[j] = kub;
                    if (v2 - v > slack * (1 + 1e-9) + 1e-9) fixed[j] = s[z] > 0.5 ? 1.0 : 0.0;
                }
            }
        }
        std::vector<i64> keep;
        for (i64 j = 0; j < p.ncol; ++j)
            if (std::isnan(fixed[j])) keep.push_back(j);
        const double tl = left();
        if ((!timed || static_cast<i64>(keep.size()) <= kWideMip) && tl > 0) {
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
                std::vector<double> xx(static_cast<std::size_t>(p.ncol), 0.0);
                for (i64 j = 0; j < p.ncol; ++j) xx[j] = colpos[j] >= 0 ? mip.x[colpos[j]] : fixed[j];
                best = std::move(xx);
            }
            if (std::isfinite(mip.bound)) bound = std::max(bound, mip.bound);
            mip_optimal = mip.status == "optimal" || mip.status == "infeasible";
            if (mip.status == "infeasible") bound = std::max(bound, best_obj);
        }
    } else if (any_integer && e.optimal && std::isfinite(best_obj)) {
        mip_optimal = true;  // within the gap of the LP bound
    }
    res.has_solution = !best.empty();
    if (res.has_solution) {
        res.x = best;
        res.objective = sgn * best_obj + p.offset;
    }
    res.bound = std::isfinite(bound) ? sgn * bound + p.offset : sgn * -kInf;
    if (res.has_solution && std::isfinite(bound)) {
        const double g = best_obj - bound;
        res.gap = g <= 0 ? 0.0 : g / std::max(std::abs(best_obj + sgn * p.offset), 1e-10);
    } else {
        res.gap = kInf;
    }
    res.status = !e.optimal ? "time_limit"
                 : res.has_solution && (res.gap <= options.gap || mip_optimal) ? "optimal" : "time_limit";
    res.seconds = seconds();
    return res;
}

}  // namespace opt
