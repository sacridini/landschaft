// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/cover.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace opt {

bool is_covering(const Model& m) {
    if (m.maximize) return false;
    for (i64 r = 0; r < m.nrow; ++r)
        if (!std::isinf(m.row_hi[r]) || !std::isfinite(m.row_lo[r])) return false;
    for (i64 j = 0; j < m.ncol; ++j)
        if (!std::isfinite(m.lb[j]) || !std::isfinite(m.ub[j])) return false;
    return true;
}

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kFeasTol = 1e-9;   // row activity vs right-hand side (relative to max(1, |rhs|))
constexpr double kDualTol = 1e-7;   // reduced costs (HiGHS's dual feasibility tolerance)
constexpr int kMaxSiftRounds = 500;
constexpr i64 kWideMip = 100000;    // columns above which HiGHS's MIP ignores the time limit for seconds

// The model after presolve, by columns, with the map back to the original.
struct Reduced {
    i64 nrow = 0, ncol = 0;
    std::vector<i64> col_start, row_index;  // CSC
    std::vector<double> value;
    std::vector<double> c, lb, ub, rhs;
    std::vector<std::uint8_t> integer;
    std::vector<i64> col_of;  // original column of each reduced column
    std::vector<i64> row_of;  // original row of each reduced row
    bool nonnegative = true;  // every coefficient >= 0
    std::vector<double> x_fixed;  // original columns' values fixed by presolve (NaN: kept)
    double offset = 0.0;          // objective of the fixed columns + the model's offset
    bool infeasible = false;
};

double row_tol(double rhs) { return kFeasTol * std::max(1.0, std::abs(rhs)); }

Reduced presolve(const Model& m) {
    Reduced p;
    const i64 n = m.ncol, nr = m.nrow;
    std::vector<i64> count(static_cast<std::size_t>(n), 0);
    for (const i64 j : m.index) ++count[j];
    p.x_fixed.assign(static_cast<std::size_t>(n), std::nan(""));
    p.offset = m.offset;
    std::vector<double> rhs(m.row_lo);
    for (i64 j = 0; j < n; ++j) {
        if (m.lb[j] > m.ub[j]) {
            p.infeasible = true;
            return p;
        }
        if (m.lb[j] == m.ub[j]) p.x_fixed[j] = m.lb[j];
        else if (count[j] == 0) p.x_fixed[j] = m.obj[j] >= 0 ? m.lb[j] : m.ub[j];
        if (!std::isnan(p.x_fixed[j])) p.offset += m.obj[j] * p.x_fixed[j];
    }
    // rows: subtract the fixed columns; the bounds of the activity of the others
    std::vector<std::uint8_t> keep_row(static_cast<std::size_t>(nr), 1);
    for (i64 r = 0; r < nr; ++r) {
        double lo = 0.0, hi = 0.0;
        for (i64 k = m.start[r]; k < m.start[r + 1]; ++k) {
            const i64 j = m.index[k];
            const double a = m.value[k];
            if (!std::isnan(p.x_fixed[j])) {
                rhs[r] -= a * p.x_fixed[j];
                continue;
            }
            lo += a > 0 ? a * m.lb[j] : a * m.ub[j];
            hi += a > 0 ? a * m.ub[j] : a * m.lb[j];
        }
        if (hi < rhs[r] - row_tol(rhs[r])) {
            p.infeasible = true;
            return p;
        }
        if (lo >= rhs[r]) keep_row[r] = 0;  // met whatever the other columns do
    }
    // reduced columns and rows
    std::vector<i64> new_row(static_cast<std::size_t>(nr), -1);
    for (i64 r = 0; r < nr; ++r)
        if (keep_row[r]) {
            new_row[r] = p.nrow++;
            p.row_of.push_back(r);
            p.rhs.push_back(rhs[r]);
        }
    std::vector<i64> new_col(static_cast<std::size_t>(n), -1);
    for (i64 j = 0; j < n; ++j)
        if (std::isnan(p.x_fixed[j])) {
            new_col[j] = p.ncol++;
            p.col_of.push_back(j);
            p.c.push_back(m.obj[j]);
            p.lb.push_back(m.lb[j]);
            p.ub.push_back(m.ub[j]);
            p.integer.push_back(m.integer[j]);
        }
    // transpose the kept part into columns (row order within each column: increasing)
    p.col_start.assign(static_cast<std::size_t>(p.ncol + 1), 0);
    for (i64 r = 0; r < nr; ++r) {
        if (new_row[r] < 0) continue;
        for (i64 k = m.start[r]; k < m.start[r + 1]; ++k)
            if (new_col[m.index[k]] >= 0 && m.value[k] != 0.0) ++p.col_start[new_col[m.index[k]] + 1];
    }
    for (i64 j = 0; j < p.ncol; ++j) p.col_start[j + 1] += p.col_start[j];
    p.row_index.resize(static_cast<std::size_t>(p.col_start.back()));
    p.value.resize(p.row_index.size());
    std::vector<i64> pos(p.col_start.begin(), p.col_start.end() - 1);
    for (i64 r = 0; r < nr; ++r) {
        if (new_row[r] < 0) continue;
        for (i64 k = m.start[r]; k < m.start[r + 1]; ++k) {
            const i64 j = new_col[m.index[k]];
            if (j < 0 || m.value[k] == 0.0) continue;
            p.row_index[pos[j]] = new_row[r];
            p.value[pos[j]++] = m.value[k];
            if (m.value[k] < 0) p.nonnegative = false;
        }
    }
    return p;
}

// Bound fixing: with binary columns, costs >= 0 and coefficients >= 0, a greedy cover of the rows left
// costs U >= the optimum, so a column dearer than U is in no optimal solution: fixed at 0. Cheap when
// the rows are short of little (re-solves with most units fixed: the replacement importance), and
// skipped when the greedy cover would take more than max_picks columns.
void prune_by_bound(Reduced& p, i64 max_picks = 1000) {
    if (p.infeasible || p.nrow == 0 || p.ncol == 0 || !p.nonnegative) return;
    for (i64 j = 0; j < p.ncol; ++j)
        if (p.c[j] < 0 || !p.integer[j] || p.lb[j] != 0.0 || p.ub[j] != 1.0) return;
    std::vector<double> left(p.rhs);
    auto unmet = [&](i64 r) { return left[r] > row_tol(p.rhs[r]); };
    auto key = [&](i64 j) {
        double cover = 0;
        for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) {
            const i64 r = p.row_index[k];
            if (unmet(r)) cover += std::min(p.value[k], left[r]) / std::max(p.rhs[r], 1e-300);
        }
        return cover > 0 ? p.c[j] / cover : kInf;
    };
    using Item = std::pair<double, i64>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    for (i64 j = 0; j < p.ncol; ++j) {
        const double k = key(j);
        if (std::isfinite(k)) heap.push({k, j});
    }
    i64 open = 0;
    for (i64 r = 0; r < p.nrow; ++r) open += unmet(r);
    double bound = 0;
    i64 picks = 0;
    std::vector<std::uint8_t> taken(static_cast<std::size_t>(p.ncol), 0);
    while (open > 0) {
        if (heap.empty() || ++picks > max_picks) return;
        const i64 j = heap.top().second;
        heap.pop();
        if (taken[j]) continue;
        const double k = key(j);
        if (!std::isfinite(k)) continue;
        if (!heap.empty() && k > heap.top().first * (1 + 1e-12)) {
            heap.push({k, j});
            continue;
        }
        taken[j] = 1;
        bound += p.c[j];
        for (i64 q = p.col_start[j]; q < p.col_start[j + 1]; ++q) {
            const i64 r = p.row_index[q];
            const bool was = unmet(r);
            left[r] -= p.value[q];
            if (was && !unmet(r)) --open;
        }
    }
    const double cut = bound * (1 + 1e-9) + 1e-12;
    i64 keep = 0;
    for (i64 j = 0; j < p.ncol; ++j) keep += p.c[j] <= cut;
    if (keep == p.ncol) return;
    // rebuild without the columns above the bound (fixed at 0)
    Reduced q;
    q.nrow = p.nrow;
    q.rhs = p.rhs;
    q.row_of = p.row_of;
    q.nonnegative = p.nonnegative;
    q.offset = p.offset;
    q.x_fixed = p.x_fixed;
    q.col_start.push_back(0);
    for (i64 j = 0; j < p.ncol; ++j) {
        if (p.c[j] > cut) {
            q.x_fixed[static_cast<std::size_t>(p.col_of[j])] = 0.0;
            continue;
        }
        for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) {
            q.row_index.push_back(p.row_index[k]);
            q.value.push_back(p.value[k]);
        }
        q.col_start.push_back(static_cast<i64>(q.row_index.size()));
        q.col_of.push_back(p.col_of[j]);
        q.c.push_back(p.c[j]);
        q.lb.push_back(p.lb[j]);
        q.ub.push_back(p.ub[j]);
        q.integer.push_back(p.integer[j]);
        ++q.ncol;
    }
    p = std::move(q);
}

enum class At : std::uint8_t { lower, upper, work };

struct Sifting {
    const Reduced& p;
    int threads;
    bool verbose = false;
    bool out_of_time = false;
    std::vector<At> at;
    std::vector<double> x, y, d;  // columns' values, rows' duals, reduced costs
    double lp_objective = std::nan("");   // of the reduced model (without the offset)
    double lagrangian = -kInf;            // best valid lower bound found (reduced model)
    bool optimal = false;
    bool infeasible = false;

    Sifting(const Reduced& red, int n_threads) : p(red), threads(n_threads) {}

    // reduced costs of every column and the Lagrangian bound b'y + sum_j min(d_j lb_j, d_j ub_j)
    void price() {
        const i64 n = p.ncol;
        d.resize(static_cast<std::size_t>(n));
#pragma omp parallel for schedule(static) num_threads(threads)
        for (i64 j = 0; j < n; ++j) {
            double s = 0.0;
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) s += y[p.row_index[k]] * p.value[k];
            d[j] = p.c[j] - s;
        }
        double bound = 0.0;
        for (i64 r = 0; r < p.nrow; ++r) bound += p.rhs[r] * y[r];
        for (i64 j = 0; j < n; ++j) bound += std::min(d[j] * p.lb[j], d[j] * p.ub[j]);
        lagrangian = std::max(lagrangian, bound);
    }

    // The Lagrangian at y: L(y) = b'y + sum_j min(d_j lb_j, d_j ub_j), with d = c - A'y, and a
    // supergradient g = b - A x(y), x_j(y) at the bound that gives the minimum. Sums over fixed
    // blocks of columns, added in block order: the same bits whatever the number of threads.
    double lagrangian_at(const std::vector<double>& yy, std::vector<double>& g) const {
        constexpr i64 kBlock = 1 << 15;
        const i64 n = p.ncol, m = p.nrow;
        const i64 nb = (n + kBlock - 1) / kBlock;
        std::vector<double> part(static_cast<std::size_t>(nb * (m + 1)), 0.0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (i64 b = 0; b < nb; ++b) {
            double* acc = &part[static_cast<std::size_t>(b * (m + 1))];  // A x over the block, then sum min(.)
            const i64 j1 = std::min(n, (b + 1) * kBlock);
            for (i64 j = b * kBlock; j < j1; ++j) {
                double s = 0.0;
                for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) s += yy[p.row_index[k]] * p.value[k];
                const double dj = p.c[j] - s;
                const double xj = dj < 0 ? p.ub[j] : p.lb[j];
                acc[m] += dj * xj;
                if (xj != 0.0)
                    for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) acc[p.row_index[k]] += p.value[k] * xj;
            }
        }
        g.assign(p.rhs.begin(), p.rhs.end());
        double value = 0.0;
        for (i64 r = 0; r < m; ++r) value += p.rhs[r] * yy[r];
        for (i64 b = 0; b < nb; ++b) {
            const double* acc = &part[static_cast<std::size_t>(b * (m + 1))];
            for (i64 r = 0; r < m; ++r) g[r] -= acc[r];
            value += acc[m];
        }
        return value;
    }

    // Near-optimal duals by a proximal (box-step) bundle method on the Lagrangian, from y: the
    // cutting-plane model min_k L_k + g_k'(y - y_k) maximised in a box around the best point. Leaves
    // the best point in y and its value in `lagrangian` (a valid lower bound).
    void dual_bundle(Clock::time_point deadline) {
        const i64 m = p.nrow;
        struct Cut {
            double value;
            std::vector<double> y, g;
        };
        std::vector<Cut> cuts;
        std::vector<double> g;
        std::vector<double> best = y;
        double best_value = lagrangian_at(best, g);
        std::vector<double> best_g = g;
        cuts.push_back({best_value, best, g});
        double scale = 0.0;
        for (const double v : best) scale = std::max(scale, v);
        double box = scale > 0 ? 0.5 * scale : 1.0;
        const double box_min = 1e-12 * std::max(1.0, scale);
        int it = 0;
        for (; it < 1000 && box > box_min; ++it) {
            if (Clock::now() >= deadline || stop_asked()) break;
            // master LP: max t  s.t.  t - g_k'y <= L_k - g_k'y_k,  best - box <= y <= best + box,  y >= 0
            Model master;
            master.ncol = m + 1;
            master.nrow = static_cast<i64>(cuts.size());
            master.maximize = true;
            master.obj.assign(static_cast<std::size_t>(m + 1), 0.0);
            master.obj[m] = 1.0;
            for (i64 r = 0; r < m; ++r) {
                master.lb.push_back(std::max(0.0, best[r] - box));
                master.ub.push_back(best[r] + box);
            }
            master.lb.push_back(-kInf);
            master.ub.push_back(kInf);
            master.integer.assign(static_cast<std::size_t>(m + 1), 0);
            master.start.push_back(0);
            for (const Cut& c : cuts) {
                double rhs = c.value;
                for (i64 r = 0; r < m; ++r) {
                    rhs -= c.g[r] * c.y[r];
                    if (c.g[r] != 0.0) {
                        master.index.push_back(r);
                        master.value.push_back(-c.g[r]);
                    }
                }
                master.index.push_back(m);
                master.value.push_back(1.0);
                master.start.push_back(static_cast<i64>(master.index.size()));
                master.row_lo.push_back(-kInf);
                master.row_hi.push_back(rhs);
            }
            SolveOptions o;
            o.threads = 1;
            const SolveResult mr = solve_highs(master, o);
            if (mr.status != "optimal") break;
            const std::vector<double> trial(mr.x.begin(), mr.x.begin() + m);
            const double predicted = mr.x[static_cast<std::size_t>(m)] - best_value;
            // the model (above L) gains nothing on a box around the best point: L is concave, so the best
            // point is a maximum
            if (predicted <= 1e-10 * std::max(1.0, std::abs(best_value))) break;
            const double value = lagrangian_at(trial, g);
            cuts.push_back({value, trial, g});
            if (value >= best_value + 0.1 * predicted) {  // serious step
                best = trial;
                best_value = value;
                best_g = g;
                box *= 2.0;
            } else {  // null step: the new cut sharpens the model; a smaller box
                box *= 0.5;
            }
            if (cuts.size() > static_cast<std::size_t>(std::max<i64>(200, 4 * m))) {
                // keep the best point's cut and the most recent ones
                std::vector<Cut> kept;
                kept.push_back({best_value, best, best_g});
                kept.insert(kept.end(), cuts.end() - static_cast<std::ptrdiff_t>(2 * m), cuts.end());
                cuts = std::move(kept);
            }
        }
        y = best;
        lagrangian = std::max(lagrangian, best_value);
        if (verbose) std::fprintf(stderr, "bundle: %d iterations, %zu cuts, L = %.12g\n", it, cuts.size(), best_value);
    }

    // The LP on the columns `cols` with right-hand sides `rhs`; its status, and on success the duals in y.
    SolveResult solve_lp(const std::vector<i64>& cols, const std::vector<double>& rhs, double time_left) {
        Model sub;
        sub.nrow = p.nrow;
        sub.ncol = static_cast<i64>(cols.size());
        sub.start.assign(static_cast<std::size_t>(p.nrow + 1), 0);
        for (const i64 j : cols)
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) ++sub.start[p.row_index[k] + 1];
        for (i64 r = 0; r < p.nrow; ++r) sub.start[r + 1] += sub.start[r];
        sub.index.resize(static_cast<std::size_t>(sub.start.back()));
        sub.value.resize(sub.index.size());
        std::vector<i64> pos(sub.start.begin(), sub.start.end() - 1);
        for (std::size_t w = 0; w < cols.size(); ++w) {
            const i64 j = cols[w];
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) {
                const i64 r = p.row_index[k];
                sub.index[pos[r]] = static_cast<i64>(w);
                sub.value[pos[r]++] = p.value[k];
            }
            sub.obj.push_back(p.c[j]);
            sub.lb.push_back(p.lb[j]);
            sub.ub.push_back(p.ub[j]);
        }
        sub.integer.assign(cols.size(), 0);
        sub.row_lo = rhs;
        sub.row_hi.assign(static_cast<std::size_t>(p.nrow), kInf);
        SolveOptions o;
        o.threads = 1;
        o.time_limit = time_left;
        SolveResult out = solve_highs(sub, o);
        if (out.status == "time_limit") out_of_time = true;
        if (out.status == "optimal") {
            if (out.row_dual.size() != static_cast<std::size_t>(p.nrow))
                throw std::runtime_error("the restricted LP gave no duals");
            y = out.row_dual;
            for (double& v : y) v = std::max(v, 0.0);  // ">=" rows of a minimisation: y >= 0 (HiGHS may give -1e-17)
        } else if (out.status != "infeasible" && out.status != "time_limit") {
            throw std::runtime_error("the restricted LP failed: " + out.status);
        }
        return out;
    }

    // The LP on the working columns, the others at their bound; false when it is infeasible.
    bool solve_restricted(double time_left) {
        std::vector<i64> work;
        std::vector<double> rhs(p.rhs);
        for (i64 j = 0; j < p.ncol; ++j) {
            if (at[j] == At::work) {
                work.push_back(j);
                continue;
            }
            const double v = at[j] == At::upper ? p.ub[j] : p.lb[j];
            if (v == 0.0) continue;
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) rhs[p.row_index[k]] -= p.value[k] * v;
        }
        const SolveResult out = solve_lp(work, rhs, time_left);
        if (out.status != "optimal") return false;
        for (std::size_t w = 0; w < work.size(); ++w) x[work[w]] = out.x[w];
        return true;
    }

    // First duals: the LP on an evenly spaced sample of `size` columns, every right-hand side scaled by
    // the share of its row's amount the sample holds (y = 0 if that LP fails).
    void sample_duals(i64 size, double time_left) {
        const i64 n = p.ncol;
        std::vector<i64> cols;
        for (i64 k = 0; k < size; ++k) cols.push_back(k * n / size);
        std::vector<double> all(static_cast<std::size_t>(p.nrow), 0.0), in(all);
        std::vector<std::uint8_t> sampled(static_cast<std::size_t>(n), 0);
        for (const i64 j : cols) sampled[j] = 1;
        for (i64 j = 0; j < n; ++j)
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) {
                const double v = p.value[k] * (p.ub[j] - p.lb[j]);
                all[p.row_index[k]] += v;
                if (sampled[j]) in[p.row_index[k]] += v;
            }
        std::vector<double> rhs(static_cast<std::size_t>(p.nrow));
        for (i64 r = 0; r < p.nrow; ++r) rhs[r] = all[r] > 0 ? p.rhs[r] * in[r] / all[r] : 0.0;
        const SolveResult out = solve_lp(cols, rhs, time_left);
        if (out.status != "optimal") std::fill(y.begin(), y.end(), 0.0);
    }

    // The `size` columns with the smallest |reduced cost| work; the others go to the bound their
    // reduced cost points to (ties by column, for determinism).
    void partition(i64 size) {
        const i64 n = p.ncol;
        std::vector<i64> order(static_cast<std::size_t>(n));
        std::iota(order.begin(), order.end(), i64{0});
        if (size < n) {
            std::nth_element(order.begin(), order.begin() + size, order.end(), [&](i64 a, i64 b) {
                const double da = std::abs(d[a]), db = std::abs(d[b]);
                return da < db || (da == db && a < b);
            });
        }
        for (i64 j = 0; j < n; ++j) at[j] = d[j] < 0 ? At::upper : At::lower;
        for (i64 k = 0; k < std::min(size, n); ++k) at[order[k]] = At::work;
        for (i64 j = 0; j < n; ++j)
            if (at[j] != At::work) x[j] = at[j] == At::upper ? p.ub[j] : p.lb[j];
    }

    // deadline: Clock::time_point::max() when there is no time limit
    void run(Clock::time_point deadline) {
        const i64 n = p.ncol;
        at.assign(static_cast<std::size_t>(n), At::work);
        x.assign(p.lb.begin(), p.lb.end());
        y.assign(static_cast<std::size_t>(p.nrow), 0.0);
        if (n == 0) {
            infeasible = p.nrow > 0;
            optimal = !infeasible;
            lp_objective = 0.0;
            return;
        }
        auto time_left = [&] {
            return deadline == Clock::time_point::max()
                       ? kInf : std::chrono::duration<double>(deadline - Clock::now()).count();
        };
        i64 size = std::max<i64>(2000, 20 * p.nrow);  // the working set's size
        if (n > 4 * size) {  // wide: start from the duals of a 2% sample; narrow: one LP on every column
            sample_duals(std::min(n, std::max<i64>(size, n / 50)), time_left());
            if (out_of_time) return;
            dual_bundle(deadline);
            price();
            partition(size);
        }
        i64 widen = size;  // columns to add when the restricted LP is infeasible (doubles each time)
        for (int round = 0; round < kMaxSiftRounds; ++round) {
            const double left = time_left();
            if (left <= 0) return;
            const auto t_lp = Clock::now();
            const bool feasible = solve_restricted(left);
            if (out_of_time) return;
            if (verbose) {
                i64 w = 0, u = 0;
                for (i64 j = 0; j < n; ++j) {
                    w += at[j] == At::work;
                    u += at[j] == At::upper;
                }
                std::fprintf(stderr, "sifting %d: %lld working, %lld at upper, LP %s in %.3f s\n", round,
                             static_cast<long long>(w), static_cast<long long>(u), feasible ? "solved" : "infeasible",
                             std::chrono::duration<double>(Clock::now() - t_lp).count());
            }
            if (!feasible) {
                // more room: the columns at the lower bound with the smallest reduced costs join the working set
                std::vector<std::pair<double, i64>> lower;
                for (i64 j = 0; j < n; ++j)
                    if (at[j] == At::lower) lower.emplace_back(d.empty() ? p.c[j] : d[j], j);
                if (!p.nonnegative)  // a column at its upper bound may be what makes it infeasible
                    for (i64 j = 0; j < n; ++j)
                        if (at[j] == At::upper) at[j] = At::work;
                if (lower.empty()) {
                    if (p.nonnegative || std::none_of(at.begin(), at.end(), [](At a) { return a != At::work; })) {
                        infeasible = true;
                        return;
                    }
                    continue;
                }
                const std::size_t add = std::min(lower.size(), static_cast<std::size_t>(widen));
                std::partial_sort(lower.begin(), lower.begin() + static_cast<std::ptrdiff_t>(add), lower.end());
                for (std::size_t k = 0; k < add; ++k) at[lower[k].second] = At::work;
                widen *= 2;
                continue;
            }
            lp_objective = 0.0;
            for (i64 j = 0; j < n; ++j) lp_objective += p.c[j] * x[j];
            price();
            // columns at a bound whose reduced cost is against it
            std::vector<std::pair<double, i64>> raise;  // at the lower bound with d < 0
            bool wrong = false;
            for (i64 j = 0; j < n; ++j) {
                const double tol = kDualTol * std::max(1.0, std::abs(p.c[j]));
                if (at[j] == At::lower && d[j] < -tol) {
                    raise.emplace_back(d[j], j);
                } else if (at[j] == At::upper && d[j] > tol) {
                    at[j] = At::work;
                    wrong = true;
                }
            }
            if (raise.empty() && !wrong) {  // optimal for the whole LP
                optimal = true;
                return;
            }
            // partial pricing: the most negative (a working set's worth, or half of them) join it, the others stay
            // at their lower bound until a later round prices them again
            widen = size;
            const std::size_t add = std::min(raise.size(), std::max(static_cast<std::size_t>(size), raise.size() / 2));
            std::partial_sort(raise.begin(), raise.begin() + static_cast<std::ptrdiff_t>(add), raise.end());
            for (std::size_t k = 0; k < add; ++k) at[raise[k].second] = At::work;
            // a working column at the bound its reduced cost points to can leave without changing the
            // restricted LP's optimum: keep the working set near its size
            i64 working = 0;
            for (i64 j = 0; j < n; ++j) working += at[j] == At::work;
            if (working > 2 * size) {
                std::vector<std::pair<double, i64>> out;
                for (i64 j = 0; j < n; ++j) {
                    if (at[j] != At::work) continue;
                    if ((x[j] == p.lb[j] && d[j] > 0) || (x[j] == p.ub[j] && d[j] < 0))
                        out.emplace_back(-std::abs(d[j]), j);
                }
                const std::size_t drop = std::min(out.size(), static_cast<std::size_t>(working - size));
                std::partial_sort(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(drop), out.end());
                for (std::size_t k = 0; k < drop; ++k) {
                    const i64 j = out[k].second;
                    at[j] = x[j] == p.ub[j] ? At::upper : At::lower;
                }
            }
            size += size / 10;  // a slowly growing working set: the rounds end
        }
    }
};

// Rounds the integer columns of an LP solution up and drops the selected columns not needed,
// the most expensive first. Needs every coefficient >= 0.
std::vector<double> round_and_trim(const Reduced& p, std::vector<double> x) {
    const i64 n = p.ncol;
    for (i64 j = 0; j < n; ++j)
        if (p.integer[j]) x[j] = std::min(p.ub[j], std::ceil(x[j] - 1e-9));
    std::vector<double> act(static_cast<std::size_t>(p.nrow), 0.0);
    for (i64 j = 0; j < n; ++j)
        for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) act[p.row_index[k]] += p.value[k] * x[j];
    std::vector<i64> cand;
    for (i64 j = 0; j < n; ++j)
        if (p.integer[j] && x[j] > p.lb[j] && p.c[j] > 0) cand.push_back(j);
    std::stable_sort(cand.begin(), cand.end(), [&](i64 a, i64 b) { return p.c[a] > p.c[b]; });
    for (const i64 j : cand) {
        while (x[j] > p.lb[j]) {  // one unit at a time (binary columns: once)
            bool ok = true;
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1] && ok; ++k) {
                const i64 r = p.row_index[k];
                ok = act[r] - p.value[k] >= p.rhs[r] - row_tol(p.rhs[r]);
            }
            if (!ok) break;
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) act[p.row_index[k]] -= p.value[k];
            x[j] -= 1.0;
        }
    }
    return x;
}

// The caller's start solution on the reduced model, or nothing when it does not fit it: within the
// bounds, integral where it must be (to 1e-6, then rounded) and meeting every kept row. The columns
// presolve fixed are in no row or forced by their bounds, so they take their fixed value: the start
// stays feasible and gets no worse.
std::vector<double> start_solution(const Reduced& p, const std::vector<double>& start) {
    if (start.empty()) return {};
    std::vector<double> x(static_cast<std::size_t>(p.ncol));
    for (i64 j = 0; j < p.ncol; ++j) {
        double v = start[p.col_of[j]];
        if (!std::isfinite(v)) return {};
        if (p.integer[j]) {
            if (std::abs(v - std::round(v)) > 1e-6) return {};
            v = std::round(v);
        }
        const double tol = 1e-9 * std::max(1.0, std::abs(v));
        if (v < p.lb[j] - tol || v > p.ub[j] + tol) return {};
        x[j] = std::clamp(v, p.lb[j], p.ub[j]);
    }
    std::vector<double> act(static_cast<std::size_t>(p.nrow), 0.0);
    for (i64 j = 0; j < p.ncol; ++j)
        for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) act[p.row_index[k]] += p.value[k] * x[j];
    for (i64 r = 0; r < p.nrow; ++r)
        if (act[r] < p.rhs[r] - row_tol(p.rhs[r])) return {};
    return x;
}

double objective_of(const Reduced& p, const std::vector<double>& x) {
    double obj = 0.0;
    for (i64 j = 0; j < p.ncol; ++j) obj += p.c[j] * x[j];
    return obj;
}

double relative_gap(double upper, double lower) {
    if (!std::isfinite(upper) || !std::isfinite(lower)) return kInf;
    const double g = upper - lower;
    if (g <= 0) return 0.0;
    return g / std::max(std::abs(upper), 1e-10);
}

}  // namespace

SolveResult solve_covering(const Model& model, const SolveOptions& options) {
    model.check();
    if (!is_covering(model)) throw std::invalid_argument("not a covering model");
    if (!options.start.empty() && options.start.size() != static_cast<std::size_t>(model.ncol))
        throw std::invalid_argument("the start solution must have one value per column");
    const auto t0 = Clock::now();
    const bool timed = std::isfinite(options.time_limit);
    const auto deadline = timed ? t0 + std::chrono::duration_cast<Clock::duration>(
                                           std::chrono::duration<double>(options.time_limit))
                                : Clock::time_point::max();
    auto seconds = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };

    SolveResult res;
    Reduced pr = presolve(model);
    prune_by_bound(pr);
    const Reduced& p = pr;
    if (p.infeasible) {
        res.status = "infeasible";
        res.seconds = seconds();
        return res;
    }
    auto full = [&](const std::vector<double>& xr) {
        std::vector<double> x(p.x_fixed);
        for (i64 j = 0; j < p.ncol; ++j) x[p.col_of[j]] = xr[j];
        return x;
    };

    // the caller's start (trimmed: units not needed dropped, the most expensive first) when it beats `x`
    const std::vector<double> given = [&] {
        std::vector<double> g = start_solution(p, options.start);
        return !g.empty() && p.nonnegative ? round_and_trim(p, std::move(g)) : g;
    }();
    auto take_start = [&](std::vector<double>& x, double& obj) {
        if (given.empty()) return;
        const double g = objective_of(p, given);
        if (g < obj) {
            x = given;
            obj = g;
        }
    };

    Sifting lp(p, std::max(1, options.threads));
    lp.verbose = options.verbose;
    lp.run(deadline);
    if (lp.infeasible) {
        res.status = "infeasible";
        res.seconds = seconds();
        return res;
    }
    const double lower = lp.optimal ? lp.lp_objective : lp.lagrangian;
    res.bound = lower + p.offset;
    bool any_integer = !options.relaxation &&
                       std::any_of(p.integer.begin(), p.integer.end(), [](std::uint8_t t) { return t != 0; });
    if (!lp.optimal) {  // out of time: the working LP's solution is feasible, the bound the best Lagrangian one
        res.status = "time_limit";
        std::vector<double> xr;
        double obj = kInf;
        if (!std::isnan(lp.lp_objective) && (!any_integer || p.nonnegative)) {
            xr = any_integer ? round_and_trim(p, lp.x) : lp.x;
            obj = objective_of(p, xr);
        }
        take_start(xr, obj);
        if (!xr.empty()) {
            res.has_solution = true;
            res.x = full(xr);
            res.objective = obj + p.offset;
            res.gap = relative_gap(res.objective, res.bound);
        }
        res.seconds = seconds();
        return res;
    }
    if (!any_integer) {
        res.status = "optimal";
        res.has_solution = true;
        res.x = full(lp.x);
        res.objective = lp.lp_objective + p.offset;
        res.bound = res.objective;
        res.gap = 0.0;
        res.seconds = seconds();
        return res;
    }

    // integer columns: round the LP solution, then HiGHS's MIP if the gap is still too wide
    std::vector<double> best;
    double best_obj = kInf;
    if (p.nonnegative) {
        best = round_and_trim(p, lp.x);
        best_obj = objective_of(p, best);
    }
    take_start(best, best_obj);
    double bound = lower;
    bool mip_optimal = false;  // HiGHS proved the gap (by its own tolerances)
    if (relative_gap(best_obj + p.offset, bound + p.offset) > options.gap) {
        // reduced-cost fixing (exact): a column whose reduced cost exceeds the gap keeps its LP bound
        const double slack = std::isfinite(best_obj) ? best_obj - lower : kInf;
        Model sub;
        sub.nrow = p.nrow;
        std::vector<i64> keep;
        std::vector<double> rhs(p.rhs);
        double fixed_obj = 0.0;
        std::vector<double> fixed_x(static_cast<std::size_t>(p.ncol), std::nan(""));
        for (i64 j = 0; j < p.ncol; ++j) {
            double v = std::nan("");
            if (p.integer[j] && lp.d[j] > slack && lp.x[j] == p.lb[j]) v = p.lb[j];
            else if (p.integer[j] && -lp.d[j] > slack && lp.x[j] == p.ub[j]) v = p.ub[j];
            if (std::isnan(v)) {
                keep.push_back(j);
                continue;
            }
            fixed_x[j] = v;
            fixed_obj += p.c[j] * v;
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) rhs[p.row_index[k]] -= p.value[k] * v;
        }
        sub.ncol = static_cast<i64>(keep.size());
        sub.start.assign(static_cast<std::size_t>(p.nrow + 1), 0);
        for (const i64 j : keep)
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) ++sub.start[p.row_index[k] + 1];
        for (i64 r = 0; r < p.nrow; ++r) sub.start[r + 1] += sub.start[r];
        sub.index.resize(static_cast<std::size_t>(sub.start.back()));
        sub.value.resize(sub.index.size());
        std::vector<i64> pos(sub.start.begin(), sub.start.end() - 1);
        std::vector<double> start;
        for (std::size_t w = 0; w < keep.size(); ++w) {
            const i64 j = keep[w];
            for (i64 k = p.col_start[j]; k < p.col_start[j + 1]; ++k) {
                const i64 r = p.row_index[k];
                sub.index[pos[r]] = static_cast<i64>(w);
                sub.value[pos[r]++] = p.value[k];
            }
            sub.obj.push_back(p.c[j]);
            sub.lb.push_back(p.lb[j]);
            sub.ub.push_back(p.ub[j]);
            sub.integer.push_back(p.integer[j]);
            if (!best.empty()) start.push_back(best[j]);
        }
        sub.row_lo = rhs;
        sub.row_hi.assign(static_cast<std::size_t>(p.nrow), kInf);
        sub.offset = fixed_obj;
        SolveOptions o = options;
        o.start = start;
        o.time_limit = timed ? std::chrono::duration<double>(deadline - Clock::now()).count() : kInf;
        // HiGHS's own presolve stalls on very wide models (and does not look at the time there)
        if (sub.ncol > kWideMip) o.presolve = Presolve::off;
        // and so does its MIP's root (feasibility jump, randomized rounding: no interrupt reaches them;
        // a 1 s limit took 2.5 s at 2e5 columns, 6 s at 4e5, 20 s at 1e6): with a time limit, a wider
        // sub-model is not sent to it, and the gap asked for is not proven in time
        const bool fits = !timed || sub.ncol <= kWideMip;
        if (o.time_limit > 0 && fits) {
            const SolveResult mip = solve_highs(sub, o);
            res.nodes = mip.nodes;
            if (mip.has_solution && mip.objective < best_obj) {
                best_obj = mip.objective;
                best.assign(fixed_x.begin(), fixed_x.end());
                for (std::size_t w = 0; w < keep.size(); ++w) best[keep[w]] = mip.x[w];
            }
            if (std::isfinite(mip.bound)) bound = std::max(bound, mip.bound);
            // the fixings only exclude solutions no better than `best`: a proven sub-model proves `best`
            mip_optimal = mip.status == "optimal" || mip.status == "infeasible";
            if (mip.status == "infeasible") bound = std::max(bound, best_obj);  // nothing better than `best`
        }
    }
    res.has_solution = !best.empty();
    res.bound = bound + p.offset;
    if (res.has_solution) {
        res.x = full(best);
        res.objective = best_obj + p.offset;
        res.gap = relative_gap(res.objective, res.bound);
    }
    res.status = res.has_solution && (res.gap <= options.gap || mip_optimal) ? "optimal" : "time_limit";
    res.seconds = seconds();
    return res;
}

}  // namespace opt
