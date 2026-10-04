// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/wide.hpp"

#include "Highs.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace opt {

namespace {

using Clock = std::chrono::steady_clock;

constexpr i64 kMaxOther = 20000;   // columns other than the units
constexpr double kDualTol = 1e-7;  // reduced costs (HiGHS's dual feasibility tolerance)
constexpr double kFeasTol = 1e-9;  // row activities (relative to max(1, |bound|))
constexpr int kMaxSiftRounds = 500;
constexpr i64 kWideMip = 100000;   // as cover.cpp: wider MIPs ignore the time limit in HiGHS's root
constexpr int kGreedyTries = 400;  // re-optimisations of the other columns in the greedy step
constexpr int kGreedyStale = 60;   // ... stopping after this many without improvement

double row_tol(double b) { return kFeasTol * std::max(1.0, std::abs(b)); }

double relative_gap(double upper, double lower) {
    if (!std::isfinite(upper) || !std::isfinite(lower)) return kInf;
    const double g = upper - lower;
    if (g <= 0) return 0.0;
    return g / std::max(std::abs(upper), 1e-10);
}

enum class At : std::uint8_t { lower, upper, work };

struct Wide {
    const Model& m;
    const i64 n, nrow, ncol;
    int threads;
    bool verbose;
    std::vector<double> c;  // the objective as a minimisation
    std::vector<i64> col_start, row_index;  // every column, by columns
    std::vector<double> value;
    std::vector<std::uint8_t> row_other;  // rows with entries in the other columns

    Wide(const Model& model, i64 n_units, int n_threads, bool log)
        : m(model), n(n_units), nrow(model.nrow), ncol(model.ncol), threads(n_threads), verbose(log) {
        c.resize(static_cast<std::size_t>(ncol));
        for (i64 j = 0; j < ncol; ++j) c[j] = m.maximize ? -m.obj[j] : m.obj[j];
        col_start.assign(static_cast<std::size_t>(ncol + 1), 0);
        for (const i64 j : m.index) ++col_start[j + 1];
        for (i64 j = 0; j < ncol; ++j) col_start[j + 1] += col_start[j];
        row_index.resize(m.index.size());
        value.resize(m.index.size());
        std::vector<i64> pos(col_start.begin(), col_start.end() - 1);
        row_other.assign(static_cast<std::size_t>(nrow), 0);
        for (i64 r = 0; r < nrow; ++r)
            for (i64 k = m.start[r]; k < m.start[r + 1]; ++k) {
                const i64 j = m.index[k];
                row_index[pos[j]] = r;
                value[pos[j]++] = m.value[k];
                if (j >= n) row_other[r] = 1;
            }
    }

    // The model on the columns `cols` (other columns after the units, as given), rows within [lo, hi],
    // integer columns kept integer when `mip`.
    Model restrict(const std::vector<i64>& cols, const std::vector<double>& lo, const std::vector<double>& hi,
                   bool mip, const std::vector<double>* lb = nullptr, const std::vector<double>* ub = nullptr) const {
        Model s;
        s.nrow = nrow;
        s.ncol = static_cast<i64>(cols.size());
        s.start.assign(static_cast<std::size_t>(nrow + 1), 0);
        for (const i64 j : cols)
            for (i64 k = col_start[j]; k < col_start[j + 1]; ++k) ++s.start[row_index[k] + 1];
        for (i64 r = 0; r < nrow; ++r) s.start[r + 1] += s.start[r];
        s.index.resize(static_cast<std::size_t>(s.start.back()));
        s.value.resize(s.index.size());
        std::vector<i64> pos(s.start.begin(), s.start.end() - 1);
        for (std::size_t w = 0; w < cols.size(); ++w) {
            const i64 j = cols[w];
            for (i64 k = col_start[j]; k < col_start[j + 1]; ++k) {
                s.index[pos[row_index[k]]] = static_cast<i64>(w);
                s.value[pos[row_index[k]]++] = value[k];
            }
            s.obj.push_back(c[j]);
            s.lb.push_back(lb ? (*lb)[j] : m.lb[j]);
            s.ub.push_back(ub ? (*ub)[j] : m.ub[j]);
            s.integer.push_back(mip ? m.integer[j] : 0);
        }
        s.row_lo = lo;
        s.row_hi = hi;
        return s;
    }

    // Row bounds less the activity of the units outside `cols` at the values `x`.
    void moved_bounds(const std::vector<double>& x, const std::vector<std::uint8_t>& kept, std::vector<double>& lo,
                      std::vector<double>& hi) const {
        lo = m.row_lo;
        hi = m.row_hi;
        for (i64 j = 0; j < n; ++j) {
            if (kept[j] || x[j] == 0.0) continue;
            for (i64 k = col_start[j]; k < col_start[j + 1]; ++k) {
                lo[row_index[k]] -= value[k] * x[j];
                hi[row_index[k]] -= value[k] * x[j];
            }
        }
    }

    std::vector<i64> others() const {
        std::vector<i64> o(static_cast<std::size_t>(ncol - n));
        std::iota(o.begin(), o.end(), n);
        return o;
    }
};


// The restricted LP of the sifting, kept in one HiGHS instance from round to round: the units that
// join the working set are added as columns, those that leave it deleted, the row bounds changed;
// HiGHS goes on from its last basis, a few pivots instead of a whole LP.
struct WarmLP {
    const Wide& w;
    Highs h;
    std::vector<i64> cols;  // LP column -> model column
    std::vector<i64> pos;   // unit -> LP column (-1: not in the LP)
    Clock::time_point deadline = Clock::time_point::max();
    bool interrupted = false;

    explicit WarmLP(const Wide& wide) : w(wide), pos(static_cast<std::size_t>(wide.n), -1) {
        h.setOptionValue("output_flag", false);
        h.setOptionValue("threads", 1);
        h.setOptionValue("presolve", std::string("off"));
        // the other columns, then rows with placeholder bounds
        HighsLp lp;
        lp.num_row_ = static_cast<HighsInt>(w.nrow);
        lp.num_col_ = 0;
        lp.row_lower_.assign(static_cast<std::size_t>(w.nrow), -kInf);
        lp.row_upper_.assign(static_cast<std::size_t>(w.nrow), kInf);
        lp.a_matrix_.format_ = MatrixFormat::kColwise;
        lp.a_matrix_.num_row_ = lp.num_row_;
        lp.a_matrix_.start_.assign(1, 0);
        if (h.passModel(lp) == HighsStatus::kError) throw std::runtime_error("HiGHS rejected the restricted LP");
        std::vector<i64> others;
        for (i64 j = w.n; j < w.ncol; ++j) others.push_back(j);
        add(others);
        h.setCallback([this](int, const std::string&, const HighsCallbackOutput*, HighsCallbackInput* in, void*) {
            if (Clock::now() >= deadline) {
                interrupted = true;
                in->user_interrupt = true;
            }
        });
        h.startCallback(kCallbackSimplexInterrupt);
    }

    void add(const std::vector<i64>& js) {
        if (js.empty()) return;
        std::vector<double> cost, lower, upper, values;
        std::vector<HighsInt> starts, index;
        for (const i64 j : js) {
            starts.push_back(static_cast<HighsInt>(index.size()));
            for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) {
                index.push_back(static_cast<HighsInt>(w.row_index[k]));
                values.push_back(w.value[k]);
            }
            cost.push_back(w.c[j]);
            lower.push_back(w.m.lb[j]);
            upper.push_back(w.m.ub[j]);
        }
        if (h.addCols(static_cast<HighsInt>(js.size()), cost.data(), lower.data(), upper.data(),
                      static_cast<HighsInt>(index.size()), starts.data(), index.data(), values.data()) ==
            HighsStatus::kError)
            throw std::runtime_error("HiGHS rejected the columns of the restricted LP");
        for (const i64 j : js) {
            if (j < w.n) pos[j] = static_cast<i64>(cols.size());
            cols.push_back(j);
        }
    }

    // the LP's units become the working ones
    void sync(const std::vector<At>& at) {
        std::vector<HighsInt> mask(cols.size(), 0);
        bool any = false;
        for (std::size_t k = 0; k < cols.size(); ++k)
            if (cols[k] < w.n && at[cols[k]] != At::work) {
                mask[k] = 1;
                any = true;
            }
        if (any) {
            if (h.deleteCols(mask.data()) == HighsStatus::kError) throw std::runtime_error("HiGHS could not delete columns");
            std::vector<i64> kept;
            for (std::size_t k = 0; k < cols.size(); ++k) {
                const i64 j = cols[k];
                if (mask[k] < 0) {
                    if (j < w.n) pos[j] = -1;
                } else {
                    if (j < w.n) pos[j] = static_cast<i64>(kept.size());
                    kept.push_back(j);
                }
            }
            cols = std::move(kept);
        }
        std::vector<i64> join;
        for (i64 j = 0; j < w.n; ++j)
            if (at[j] == At::work && pos[j] < 0) join.push_back(j);
        add(join);
    }

    // "optimal", "infeasible" or "time_limit"; on success x (LP columns) and the row duals
    std::string solve(const std::vector<double>& lo, const std::vector<double>& hi, Clock::time_point until,
                      std::vector<double>& x, std::vector<double>& y) {
        if (h.changeRowsBounds(0, static_cast<HighsInt>(w.nrow - 1), lo.data(), hi.data()) == HighsStatus::kError)
            throw std::runtime_error("HiGHS rejected the row bounds of the restricted LP");
        deadline = until;
        interrupted = false;
        use_highs_threads(1);
        h.run();
        const HighsModelStatus ms = h.getModelStatus();
        if (ms == HighsModelStatus::kOptimal) {
            x = h.getSolution().col_value;
            y = h.getSolution().row_dual;
            return "optimal";
        }
        if (ms == HighsModelStatus::kInfeasible) return "infeasible";
        if (ms == HighsModelStatus::kInterrupt || ms == HighsModelStatus::kTimeLimit) return "time_limit";
        throw std::runtime_error("the restricted LP failed: " + h.modelStatusToString(ms));
    }
};

struct Sifting {
    Wide& w;
    std::vector<At> at;
    std::vector<double> x, y, d;  // every column's value, rows' duals, units' reduced costs
    std::unique_ptr<WarmLP> warm;  // the restricted LP, from round to round
    Clock::time_point until = Clock::time_point::max();
    double lp_objective = std::nan("");
    double lagrangian = -kInf;  // best valid lower bound found
    bool optimal = false, infeasible = false, out_of_time = false, feasible_x = false;

    explicit Sifting(Wide& wide) : w(wide) {}

    // y_r >= 0 is allowed when row r has a finite lower bound, y_r <= 0 when it has a finite upper one
    double dual_lo(i64 r) const { return std::isfinite(w.m.row_hi[r]) ? -kInf : 0.0; }
    double dual_hi(i64 r) const { return std::isfinite(w.m.row_lo[r]) ? kInf : 0.0; }

    // The value at the bound a reduced cost points to (a finite one when it is ~0).
    static double at_bound(double dj, double lb, double ub) {
        if (dj < 0 && std::isfinite(ub)) return ub;
        if (dj > 0 && std::isfinite(lb)) return lb;
        return std::isfinite(lb) ? lb : (std::isfinite(ub) ? ub : 0.0);
    }

    // The Lagrangian at yy (dual-feasible for the columns with an infinite bound):
    // L(y) = sum_r y_r b_r(y) + sum_j d_j x_j(y), d = c - A'y, x_j(y) at the bound d_j points to,
    // b_r(y) the row's lower bound when y_r > 0, its upper one when y_r < 0; a supergradient
    // g = b(y) - A x(y). Sums over fixed blocks of columns, added in block order: the same bits
    // whatever the number of threads.
    double lagrangian_at(const std::vector<double>& yy, std::vector<double>& g) const {
        constexpr i64 kBlock = 1 << 15;
        const i64 n = w.ncol, m = w.nrow;
        const i64 nb = (n + kBlock - 1) / kBlock;
        std::vector<double> part(static_cast<std::size_t>(nb * (m + 1)), 0.0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(w.threads)
        for (i64 b = 0; b < nb; ++b) {
            double* acc = &part[static_cast<std::size_t>(b * (m + 1))];
            const i64 j1 = std::min(n, (b + 1) * kBlock);
            for (i64 j = b * kBlock; j < j1; ++j) {
                double s = 0.0;
                for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) s += yy[w.row_index[k]] * w.value[k];
                const double dj = w.c[j] - s;
                const double xj = at_bound(dj, w.m.lb[j], w.m.ub[j]);
                acc[m] += dj * xj;
                if (xj != 0.0)
                    for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) acc[w.row_index[k]] += w.value[k] * xj;
            }
        }
        g.assign(static_cast<std::size_t>(m), 0.0);
        double value = 0.0;
        for (i64 r = 0; r < m; ++r) {
            const double lo = w.m.row_lo[r], hi = w.m.row_hi[r];
            const double b = yy[r] > 0 ? lo : yy[r] < 0 ? hi : (std::isfinite(lo) ? lo : (std::isfinite(hi) ? hi : 0.0));
            if (yy[r] != 0.0) value += yy[r] * b;
            g[r] = b;
        }
        for (i64 bk = 0; bk < nb; ++bk) {
            const double* acc = &part[static_cast<std::size_t>(bk * (m + 1))];
            for (i64 r = 0; r < m; ++r) g[r] -= acc[r];
            value += acc[m];
        }
        return value;
    }

    // Near-optimal duals by a proximal (box-step) bundle method on the Lagrangian from y (as
    // cover.cpp), the duals kept feasible for the columns with an infinite bound by linear
    // constraints in the master. Leaves the best point in y, its value in `lagrangian`.
    void dual_bundle(Clock::time_point deadline) {
        const i64 m = w.nrow;
        // the columns with an infinite bound: d_j >= 0 (ub = inf) or d_j <= 0 (lb = -inf)
        std::vector<i64> inf_cols;
        for (i64 j = w.n; j < w.ncol; ++j)
            if (!std::isfinite(w.m.ub[j]) || !std::isfinite(w.m.lb[j])) inf_cols.push_back(j);
        for (i64 r = 0; r < m; ++r) y[r] = std::clamp(y[r], dual_lo(r), dual_hi(r));
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
        for (const double v : best) scale = std::max(scale, std::abs(v));
        double box = scale > 0 ? 0.5 * scale : 1.0;
        const double box_min = 1e-12 * std::max(1.0, scale);
        int it = 0;
        for (; it < 1000 && box > box_min; ++it) {
            if (Clock::now() >= deadline) break;
            // master LP: max t  s.t.  t - g_k'y <= L_k - g_k'y_k, the box around the best point, the
            // signs of y, and A_j'y <= c_j (ub_j = inf) / A_j'y >= c_j (lb_j = -inf)
            Model master;
            master.ncol = m + 1;
            master.maximize = true;
            master.obj.assign(static_cast<std::size_t>(m + 1), 0.0);
            master.obj[m] = 1.0;
            for (i64 r = 0; r < m; ++r) {
                master.lb.push_back(std::max(dual_lo(r), best[r] - box));
                master.ub.push_back(std::min(dual_hi(r), best[r] + box));
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
            for (const i64 j : inf_cols) {
                for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) {
                    master.index.push_back(w.row_index[k]);
                    master.value.push_back(w.value[k]);
                }
                master.start.push_back(static_cast<i64>(master.index.size()));
                master.row_lo.push_back(std::isfinite(w.m.lb[j]) ? -kInf : w.c[j]);
                master.row_hi.push_back(std::isfinite(w.m.ub[j]) ? kInf : w.c[j]);
            }
            master.nrow = static_cast<i64>(master.row_lo.size());
            SolveOptions o;
            o.threads = 1;
            const SolveResult mr = solve_highs(master, o);
            if (mr.status != "optimal") break;
            // HiGHS's point may leave the bounds by its tolerance: a dual of the wrong sign would meet an
            // infinite row bound in the Lagrangian
            std::vector<double> trial(mr.x.begin(), mr.x.begin() + m);
            for (i64 r = 0; r < m; ++r) trial[r] = std::clamp(trial[r], dual_lo(r), dual_hi(r));
            const double predicted = mr.x[static_cast<std::size_t>(m)] - best_value;
            if (predicted <= 1e-10 * std::max(1.0, std::abs(best_value))) break;
            const double value = lagrangian_at(trial, g);
            cuts.push_back({value, trial, g});
            if (value >= best_value + 0.1 * predicted) {  // serious step
                best = trial;
                best_value = value;
                best_g = g;
                box *= 2.0;
            } else {  // null step: a sharper model, a smaller box
                box *= 0.5;
            }
            if (cuts.size() > static_cast<std::size_t>(std::max<i64>(200, 4 * m))) {
                std::vector<Cut> kept;
                kept.push_back({best_value, best, best_g});
                kept.insert(kept.end(), cuts.end() - static_cast<std::ptrdiff_t>(2 * m), cuts.end());
                cuts = std::move(kept);
            }
        }
        y = best;
        lagrangian = std::max(lagrangian, best_value);
        if (w.verbose) std::fprintf(stderr, "wide bundle: %d iterations, %zu cuts, L = %.12g\n", it, cuts.size(), best_value);
    }

    void price() {
        const i64 n = w.n;
        d.resize(static_cast<std::size_t>(n));
#pragma omp parallel for schedule(static) num_threads(w.threads)
        for (i64 j = 0; j < n; ++j) {
            double s = 0.0;
            for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) s += y[w.row_index[k]] * w.value[k];
            d[j] = w.c[j] - s;
        }
    }

    // the LP on the working units and the other columns; false when infeasible (or out of time)
    bool solve_restricted(double time_left, const std::vector<double>* lo_in = nullptr,
                          const std::vector<double>* hi_in = nullptr, const std::vector<std::uint8_t>* sample = nullptr) {
        std::vector<i64> cols;
        std::vector<std::uint8_t> kept(static_cast<std::size_t>(w.n), 0);
        std::vector<double> xb(static_cast<std::size_t>(w.n), 0.0);
        for (i64 j = 0; j < w.n; ++j) {
            const bool in = sample ? (*sample)[j] != 0 : at[j] == At::work;
            if (in) {
                cols.push_back(j);
                kept[j] = 1;
            } else if (!sample) {
                xb[j] = at[j] == At::upper ? w.m.ub[j] : w.m.lb[j];
            }
        }
        const i64 nwork = static_cast<i64>(cols.size());
        for (i64 j = w.n; j < w.ncol; ++j) cols.push_back(j);
        std::vector<double> lo, hi;
        if (lo_in) {
            lo = *lo_in;
            hi = *hi_in;
        } else {
            w.moved_bounds(xb, kept, lo, hi);
        }
        if (!sample) {  // the warm restricted LP
            if (!warm) warm = std::make_unique<WarmLP>(w);
            warm->sync(at);
            std::vector<double> xl, yl;
            const std::string st = warm->solve(lo, hi, until, xl, yl);
            if (st == "time_limit") out_of_time = true;
            if (st != "optimal") return false;
            y = std::move(yl);
            for (i64 j = 0; j < w.n; ++j) x[j] = xb[j];
            for (std::size_t k = 0; k < warm->cols.size(); ++k) x[warm->cols[k]] = xl[k];
            feasible_x = true;
            return true;
        }
        const Model sub = w.restrict(cols, lo, hi, false);
        SolveOptions o;
        o.threads = 1;
        o.time_limit = time_left;
        o.presolve = Presolve::off;
        const SolveResult r = solve_highs(sub, o);
        if (r.status == "time_limit") out_of_time = true;
        if (r.status != "optimal") {
            if (r.status != "infeasible" && r.status != "time_limit" && r.status != "infeasible_or_unbounded")
                throw std::runtime_error("the restricted LP failed: " + r.status);
            return false;
        }
        if (r.row_dual.size() != static_cast<std::size_t>(w.nrow))
            throw std::runtime_error("the restricted LP gave no duals");
        y = r.row_dual;
        if (!sample) {
            for (i64 j = 0; j < w.n; ++j) x[j] = xb[j];
            for (std::size_t k = 0; k < cols.size(); ++k) x[cols[k]] = r.x[k];
            feasible_x = true;
        }
        (void)nwork;
        return true;
    }

    void run(Clock::time_point deadline) {
        until = deadline;
        const i64 n = w.n;
        x.assign(static_cast<std::size_t>(w.ncol), 0.0);
        y.assign(static_cast<std::size_t>(w.nrow), 0.0);
        at.assign(static_cast<std::size_t>(n), At::work);
        auto time_left = [&] {
            return deadline == Clock::time_point::max() ? kInf
                                                        : std::chrono::duration<double>(deadline - Clock::now()).count();
        };
        const i64 size = std::max<i64>(5000, 50 * w.nrow);
        if (n > 4 * size) {
            // first duals: the LP on an evenly spaced sample, its rows scaled by the share of their amount it holds
            const i64 ns = std::max(size, n / 50);
            std::vector<std::uint8_t> sample(static_cast<std::size_t>(n), 0);
            for (i64 k = 0; k < ns; ++k) sample[k * n / ns] = 1;
            std::vector<double> all(static_cast<std::size_t>(w.nrow), 0.0), in(all);
            for (i64 j = 0; j < n; ++j)
                for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) {
                    const double v = std::abs(w.value[k]) * (w.m.ub[j] - w.m.lb[j]);
                    all[w.row_index[k]] += v;
                    if (sample[j]) in[w.row_index[k]] += v;
                }
            std::vector<double> lo(w.m.row_lo), hi(w.m.row_hi);
            for (i64 r = 0; r < w.nrow; ++r) {
                const double s = all[r] > 0 ? in[r] / all[r] : 1.0;
                if (std::isfinite(lo[r])) lo[r] *= s;
                if (std::isfinite(hi[r])) hi[r] *= s;
            }
            if (!solve_restricted(time_left(), &lo, &hi, &sample)) std::fill(y.begin(), y.end(), 0.0);
            if (out_of_time) return;
            // the bundle gets at most 40% of the time left: the sifting needs the rest
            dual_bundle(deadline == Clock::time_point::max()
                            ? deadline
                            : Clock::now() + std::chrono::duration_cast<Clock::duration>(0.4 * (deadline - Clock::now())));
            price();
            std::vector<i64> order(static_cast<std::size_t>(n));
            std::iota(order.begin(), order.end(), i64{0});
            std::nth_element(order.begin(), order.begin() + size, order.end(), [&](i64 a, i64 b) {
                const double da = std::abs(d[a]), db = std::abs(d[b]);
                return da < db || (da == db && a < b);
            });
            for (i64 j = 0; j < n; ++j) at[j] = d[j] < 0 ? At::upper : At::lower;
            for (i64 k = 0; k < size; ++k) at[order[k]] = At::work;
        }
        i64 widen = size;
        for (int round = 0; round < kMaxSiftRounds; ++round) {
            const double left = time_left();
            if (left <= 0) {
                out_of_time = true;
                return;
            }
            const bool feasible = solve_restricted(left);
            if (out_of_time) return;
            if (!feasible) {
                // room: every unit at its upper bound and the cheapest at the lower one join the working set
                std::vector<std::pair<double, i64>> lower;
                bool moved = false;
                for (i64 j = 0; j < n; ++j) {
                    if (at[j] == At::upper) {
                        at[j] = At::work;
                        moved = true;
                    } else if (at[j] == At::lower) {
                        lower.emplace_back(d.empty() ? w.c[j] : d[j], j);
                    }
                }
                if (lower.empty() && !moved) {
                    infeasible = true;
                    return;
                }
                const std::size_t add = std::min(lower.size(), static_cast<std::size_t>(widen));
                std::partial_sort(lower.begin(), lower.begin() + static_cast<std::ptrdiff_t>(add), lower.end());
                for (std::size_t k = 0; k < add; ++k) at[lower[k].second] = At::work;
                widen *= 2;
                continue;
            }
            lp_objective = 0.0;
            for (i64 j = 0; j < w.ncol; ++j) lp_objective += w.c[j] * x[j];
            price();
            std::vector<std::pair<double, i64>> raise;
            bool wrong = false;
            for (i64 j = 0; j < n; ++j) {
                const double tol = kDualTol * std::max(1.0, std::abs(w.c[j]));
                if (at[j] == At::lower && d[j] < -tol) {
                    raise.emplace_back(d[j], j);
                } else if (at[j] == At::upper && d[j] > tol) {
                    at[j] = At::work;
                    wrong = true;
                }
            }
            if (w.verbose) {
                i64 nw = 0;
                for (const At a : at) nw += a == At::work;
                std::fprintf(stderr, "wide sifting %d: %lld working, LP %.12g, %zu to add\n", round,
                             static_cast<long long>(nw), lp_objective, raise.size());
            }
            if (raise.empty() && !wrong) {
                optimal = true;
                return;
            }
            widen = size;
            const std::size_t add = std::min(raise.size(), static_cast<std::size_t>(4 * size));
            std::partial_sort(raise.begin(), raise.begin() + static_cast<std::ptrdiff_t>(add), raise.end());
            for (std::size_t k = 0; k < add; ++k) at[raise[k].second] = At::work;
            // keep the working set near its size: units at the bound their reduced cost points to leave it
            i64 working = 0;
            for (i64 j = 0; j < n; ++j) working += at[j] == At::work;
            if (working > 8 * size) {
                std::vector<std::pair<double, i64>> out;
                for (i64 j = 0; j < n; ++j) {
                    if (at[j] != At::work) continue;
                    const double tol = kDualTol * std::max(1.0, std::abs(w.c[j]));
                    if ((d[j] > tol && x[j] == w.m.lb[j]) || (d[j] < -tol && x[j] == w.m.ub[j]))
                        out.emplace_back(-std::abs(d[j]), j);
                }
                const std::size_t drop = std::min(out.size(), static_cast<std::size_t>(working - 4 * size));
                std::partial_sort(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(drop), out.end());
                for (std::size_t k = 0; k < drop; ++k) {
                    const i64 j = out[k].second;
                    at[j] = d[j] > 0 ? At::lower : At::upper;
                }
            }
        }
    }
};

// The activity of every row from the units' values xu, and their objective.
void unit_activity(const Wide& w, const std::vector<double>& xu, std::vector<double>& act, double& unit_obj) {
    act.assign(static_cast<std::size_t>(w.nrow), 0.0);
    unit_obj = 0.0;
    for (i64 j = 0; j < w.n; ++j) {
        if (xu[j] == 0.0) continue;
        unit_obj += w.c[j] * xu[j];
        for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) act[w.row_index[k]] += w.value[k] * xu[j];
    }
}

// The other columns at their best for the units' row activity `act`: false when no values make the
// rows hold. xo: the other columns' values; other_obj: their objective.
bool best_others(const Wide& w, const std::vector<double>& act, double time_left, std::vector<double>& xo,
                 double& other_obj, bool check_unit_rows = true) {
    std::vector<double> lo(w.m.row_lo), hi(w.m.row_hi);
    for (i64 r = 0; r < w.nrow; ++r) {
        lo[r] -= act[r];
        hi[r] -= act[r];
        if (w.row_other[r]) continue;
        // rows of units only must hold as they are
        if (check_unit_rows && (lo[r] > row_tol(w.m.row_lo[r]) || hi[r] < -row_tol(w.m.row_hi[r]))) return false;
        lo[r] = -kInf;
        hi[r] = kInf;
    }
    xo.clear();
    other_obj = 0.0;
    if (w.ncol == w.n) return true;
    const Model sub = w.restrict(w.others(), lo, hi, true);
    SolveOptions o;
    o.threads = 1;
    o.gap = 0.0;
    o.time_limit = time_left;
    const SolveResult r = solve_highs(sub, o);
    if (!r.has_solution || r.status != "optimal") return false;
    xo = r.x;
    for (i64 j = w.n; j < w.ncol; ++j) other_obj += w.c[j] * xo[j - w.n];
    return true;
}


// How far an activity is outside a row's bounds.
double violation(double a, double lo, double hi) {
    return a < lo ? lo - a : (a > hi ? a - hi : 0.0);
}

// Drops the integer units the rows do not need, with the other columns at xo: a unit goes when every
// row with other columns stays within its bounds and no row of units only gets further outside them;
// those that weigh most on the rows of units only (the budget) first, then the costliest, then by
// column. act: the units' row activity, kept up to date.
void trim(const Wide& w, std::vector<double>& xu, std::vector<double>& act, const std::vector<double>& xo) {
    std::vector<double> oc(static_cast<std::size_t>(w.nrow), 0.0);  // the other columns' activity
    for (i64 j = w.n; j < w.ncol; ++j) {
        const double v = xo[j - w.n];
        if (v != 0.0)
            for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) oc[w.row_index[k]] += w.value[k] * v;
    }
    std::vector<std::pair<double, i64>> cand;
    for (i64 j = 0; j < w.n; ++j) {
        if (!w.m.integer[j] || xu[j] <= w.m.lb[j]) continue;
        double weight = 0.0;
        for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k)
            if (!w.row_other[w.row_index[k]]) weight += std::abs(w.value[k]);
        cand.emplace_back(-(weight + 1e-12 * w.c[j]), j);
    }
    std::sort(cand.begin(), cand.end());
    for (const auto& [key, j] : cand) {
        while (xu[j] > w.m.lb[j]) {
            bool ok = true;
            for (i64 k = w.col_start[j]; k < w.col_start[j + 1] && ok; ++k) {
                const i64 r = w.row_index[k];
                const double lo = w.m.row_lo[r] - oc[r], hi = w.m.row_hi[r] - oc[r];
                const double now = act[r], next = act[r] - w.value[k];
                if (w.row_other[r])
                    ok = next >= lo - row_tol(w.m.row_lo[r]) && next <= hi + row_tol(w.m.row_hi[r]);
                else
                    ok = violation(next, lo, hi) <= violation(now, lo, hi);
            }
            if (!ok) break;
            for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) act[w.row_index[k]] -= w.value[k];
            xu[j] -= 1.0;
        }
    }
}

}  // namespace

bool is_wide(const Model& model, i64 n_units) {
    if (n_units < 0 || n_units > model.ncol || model.ncol - n_units > kMaxOther) return false;
    for (i64 j = 0; j < n_units; ++j)
        if (!std::isfinite(model.lb[j]) || !std::isfinite(model.ub[j])) return false;
    return true;
}

SolveResult solve_wide(const Model& model, i64 n_units, const SolveOptions& options) {
    model.check();
    if (!is_wide(model, n_units)) throw std::invalid_argument("not a wide model");
    if (!options.start.empty() && options.start.size() != static_cast<std::size_t>(model.ncol))
        throw std::invalid_argument("the start solution must have one value per column");
    const auto t0 = Clock::now();
    const bool timed = std::isfinite(options.time_limit);
    const auto deadline = timed ? t0 + std::chrono::duration_cast<Clock::duration>(
                                           std::chrono::duration<double>(options.time_limit))
                                : Clock::time_point::max();
    auto seconds = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    auto time_left = [&] { return timed ? std::chrono::duration<double>(deadline - Clock::now()).count() : kInf; };
    const double sign = model.maximize ? -1.0 : 1.0;  // objective in the model's sense = sign * minimisation's

    Wide w(model, n_units, std::max(1, options.threads), options.verbose);
    SolveResult res;
    auto finish = [&](const std::vector<double>& x, double obj_min, double bound_min, const std::string& status) {
        res.status = status;
        if (!x.empty()) {
            res.has_solution = true;
            res.x = x;
            res.objective = sign * obj_min + model.offset;
            res.gap = relative_gap(obj_min, bound_min);
        }
        res.bound = std::isfinite(bound_min) ? sign * bound_min + model.offset : sign * -kInf;
        res.seconds = seconds();
        return res;
    };

    Sifting lp(w);
    lp.run(deadline);
    if (lp.infeasible) return finish({}, kInf, kInf, "infeasible");
    const bool any_integer = !options.relaxation &&
                             std::any_of(model.integer.begin(), model.integer.end(), [](std::uint8_t t) { return t; });
    // out of time before the LP was solved: the bound is the best Lagrangian one, and the restricted
    // LP's solution (feasible: the other units at their bounds) is rounded like the LP's
    const bool lp_done = lp.optimal;
    if (!lp_done && !lp.feasible_x) {
        // out of time before any restricted LP: the units at their lower bounds, if the rows allow it
        std::vector<double> xu(model.lb.begin(), model.lb.begin() + w.n), act, xo;
        double unit_obj, other_obj;
        unit_activity(w, xu, act, unit_obj);
        if (!best_others(w, act, 0.5, xo, other_obj)) return finish({}, kInf, lp.lagrangian, "time_limit");
        xu.insert(xu.end(), xo.begin(), xo.end());
        return finish(xu, unit_obj + other_obj, std::min(lp.lagrangian, unit_obj + other_obj), "time_limit");
    }
    const double lower = lp_done ? lp.lp_objective : lp.lagrangian;
    if (!any_integer) return finish(lp.x, lp.lp_objective, lower, lp_done ? "optimal" : "time_limit");
    // the re-optimisations of the few other columns take milliseconds: they run even at the deadline
    auto aux_time = [&] { return std::max(time_left(), 0.5); };

    // integer units: round, re-optimise the other columns, add units greedily
    std::vector<double> best, best_act;  // best: every column; best_act: its units' row activity
    double best_obj = kInf;
    auto consider = [&](const std::vector<double>& xu, const std::vector<double>& act, double unit_obj) {
        std::vector<double> xo;
        double other_obj;
        if (!best_others(w, act, aux_time(), xo, other_obj) || !(unit_obj + other_obj < best_obj)) return false;
        best.assign(xu.begin(), xu.begin() + w.n);
        best.insert(best.end(), xo.begin(), xo.end());
        best_act = act;
        best_obj = unit_obj + other_obj;
        return true;
    };
    auto consider_units = [&](const std::vector<double>& xu) {
        std::vector<double> act;
        double unit_obj;
        unit_activity(w, xu, act, unit_obj);
        return consider(xu, act, unit_obj);
    };
    {  // round down
        std::vector<double> xu(lp.x.begin(), lp.x.begin() + w.n);
        for (i64 j = 0; j < w.n; ++j)
            if (model.integer[j]) xu[j] = std::max(model.lb[j], std::floor(xu[j] + 1e-9));
        consider_units(xu);
    }
    {  // round up, set the other columns for those units (the rows of units only aside), drop the units not needed
        std::vector<double> xu(lp.x.begin(), lp.x.begin() + w.n);
        for (i64 j = 0; j < w.n; ++j)
            if (model.integer[j]) xu[j] = std::min(model.ub[j], std::ceil(xu[j] - 1e-9));
        std::vector<double> act, xo;
        double unit_obj, other_obj;
        unit_activity(w, xu, act, unit_obj);
        if (best_others(w, act, aux_time(), xo, other_obj, false)) {
            trim(w, xu, act, xo);
            consider_units(xu);
        }
    }
    // The cheapest LP solution with the other columns within [olo, oup] (the least weight on the rows of
    // units only, those rows left open), rounded up and trimmed, then offered as a solution: true when
    // that gives a feasible one.
    auto cheapest_round = [&](const std::vector<double>& olo, const std::vector<double>& oup, double& lp_cost) {
        Model m2 = model;
        m2.maximize = false;
        m2.offset = 0.0;
        bool any_budget = false;
        for (i64 r = 0; r < w.nrow; ++r)
            if (!w.row_other[r]) {
                m2.row_lo[r] = -kInf;
                m2.row_hi[r] = kInf;
            }
        for (i64 j = 0; j < w.ncol; ++j) {
            double v = 0.0;
            if (j < w.n) {
                for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) {
                    const i64 r = w.row_index[k];
                    if (!w.row_other[r] && std::isfinite(model.row_hi[r])) v += w.value[k];
                }
            } else {
                m2.lb[j] = olo[j - w.n];
                m2.ub[j] = oup[j - w.n];
            }
            m2.obj[j] = v;
            any_budget = any_budget || v != 0.0;
        }
        lp_cost = kInf;
        if (!any_budget) return false;
        Wide w2(m2, w.n, w.threads, false);
        Sifting lp2(w2);
        lp2.run(deadline);
        if (!lp2.optimal) return false;
        lp_cost = lp2.lp_objective;
        std::vector<double> xu(lp2.x.begin(), lp2.x.begin() + w.n);
        for (i64 j = 0; j < w.n; ++j)
            if (model.integer[j]) xu[j] = std::min(model.ub[j], std::ceil(xu[j] - 1e-9));
        std::vector<double> act, xo;
        double unit_obj, other_obj;
        unit_activity(w, xu, act, unit_obj);
        if (!best_others(w, act, aux_time(), xo, other_obj, false)) return false;
        trim(w, xu, act, xo);
        unit_activity(w, xu, act, unit_obj);
        if (!best_others(w, act, aux_time(), xo, other_obj)) return false;
        consider(xu, act, unit_obj);
        return true;
    };
    // the LP's optimum may spend the whole budget (the units cost nothing in min shortfall's objective, for
    // one): the cheapest LP solution with the other columns at their LP values leaves room to round up
    if (lp_done && relative_gap(best_obj, lower) > options.gap && time_left() > 0) {
        std::vector<double> olo(lp.x.begin() + w.n, lp.x.end());
        double cost;
        cheapest_round(olo, olo, cost);
        if (options.verbose)
            std::fprintf(stderr, "wide: cheapest LP %.12g, rounded up and trimmed: %.12g\n", cost, best_obj);
    }
    // binary other columns (targets met, features present): a dive. They are ranked by their LP value
    // (then their objective, then column); the first k are fixed at 1, the others at 0, and the cheapest
    // LP for that is rounded up and trimmed; a binary search finds the largest k that fits
    std::vector<i64> flags;
    for (i64 j = w.n; j < w.ncol; ++j)
        if (model.integer[j] && model.lb[j] == 0.0 && model.ub[j] == 1.0) flags.push_back(j);
    if (lp_done && !flags.empty() && relative_gap(best_obj, lower) > options.gap && time_left() > 0) {
        std::stable_sort(flags.begin(), flags.end(), [&](i64 a, i64 b) {
            if (lp.x[a] != lp.x[b]) return lp.x[a] > lp.x[b];
            return w.c[a] < w.c[b];
        });
        auto probe = [&](i64 k) {
            std::vector<double> olo(model.lb.begin() + w.n, model.lb.end()), oup(model.ub.begin() + w.n, model.ub.end());
            for (std::size_t q = 0; q < flags.size(); ++q)
                olo[flags[q] - w.n] = oup[flags[q] - w.n] = static_cast<i64>(q) < k ? 1.0 : 0.0;
            double cost;
            const bool ok = cheapest_round(olo, oup, cost);
            if (options.verbose) std::fprintf(stderr, "wide dive: %lld fixed, %s, best %.12g\n",
                                              static_cast<long long>(k), ok ? "fits" : "no", best_obj);
            return ok;
        };
        i64 lo_k = 0, hi_k = static_cast<i64>(flags.size());
        while (lo_k < hi_k && time_left() > 0) {
            const i64 mid = (lo_k + hi_k + 1) / 2;
            if (probe(mid)) lo_k = mid;
            else hi_k = mid - 1;
        }
    }
    if (!options.start.empty()) consider_units(std::vector<double>(options.start.begin(), options.start.begin() + w.n));
    if (!best.empty() && lp.d.size() == static_cast<std::size_t>(w.n)) {
        // the units not selected, the most attractive reduced cost first
        std::vector<std::pair<double, i64>> cand;
        for (i64 j = 0; j < w.n; ++j)
            if (model.integer[j] && best[j] + 1.0 <= model.ub[j] + 1e-9 && lp.d[j] < 0) cand.emplace_back(lp.d[j], j);
        std::sort(cand.begin(), cand.end());
        double best_unit_obj = 0.0;
        for (i64 i = 0; i < w.n; ++i) best_unit_obj += w.c[i] * best[i];
        int tries = 0, stale = 0;
        for (const auto& [dj, j] : cand) {
            if (tries >= kGreedyTries || stale >= kGreedyStale || (timed && Clock::now() >= deadline)) break;
            std::vector<double> act(best_act);
            bool fits = true;
            for (i64 k = w.col_start[j]; k < w.col_start[j + 1]; ++k) {
                const i64 r = w.row_index[k];
                act[r] += w.value[k];
                if (!w.row_other[r])
                    fits = fits && act[r] <= model.row_hi[r] + row_tol(model.row_hi[r]) &&
                           act[r] >= model.row_lo[r] - row_tol(model.row_lo[r]);
            }
            if (!fits) continue;
            ++tries;
            std::vector<double> xo;
            double other_obj;
            const double unit_obj = best_unit_obj + w.c[j];
            if (best_others(w, act, time_left(), xo, other_obj) && unit_obj + other_obj < best_obj) {
                best[j] += 1.0;
                std::copy(xo.begin(), xo.end(), best.begin() + w.n);
                best_act = std::move(act);
                best_unit_obj = unit_obj;
                best_obj = unit_obj + other_obj;
                stale = 0;
            } else {
                ++stale;
            }
        }
    }
    if (options.verbose)
        std::fprintf(stderr, "wide: LP %.12g, rounded and greedy %.12g\n", lower, best_obj);

    double bound = lower;
    bool mip_optimal = false;
    if (lp_done && relative_gap(best_obj, bound) > options.gap && time_left() > 0) {
        // reduced-cost fixing (exact): a unit whose reduced cost exceeds the gap keeps its LP bound
        const double slack = std::isfinite(best_obj) ? best_obj - lower : kInf;
        std::vector<i64> keep;
        std::vector<std::uint8_t> kept(static_cast<std::size_t>(w.n), 0);
        std::vector<double> xfix(static_cast<std::size_t>(w.n), 0.0);
        double fixed_obj = 0.0;
        for (i64 j = 0; j < w.n; ++j) {
            const double tol = kDualTol * std::max(1.0, std::abs(w.c[j]));
            double v = std::nan("");
            if (model.integer[j] && lp.d[j] > slack + tol && lp.x[j] == model.lb[j]) v = model.lb[j];
            else if (model.integer[j] && -lp.d[j] > slack + tol && lp.x[j] == model.ub[j]) v = model.ub[j];
            if (std::isnan(v)) {
                keep.push_back(j);
                kept[j] = 1;
            } else {
                xfix[j] = v;
                fixed_obj += w.c[j] * v;
            }
        }
        const i64 nkeep = static_cast<i64>(keep.size());
        for (i64 j = w.n; j < w.ncol; ++j) keep.push_back(j);
        std::vector<double> lo, hi;
        w.moved_bounds(xfix, kept, lo, hi);
        Model sub = w.restrict(keep, lo, hi, true);
        sub.offset = fixed_obj;
        if (options.verbose)
            std::fprintf(stderr, "wide: %lld units left after reduced-cost fixing (slack %.6g)\n",
                         static_cast<long long>(nkeep), slack);
        SolveOptions o = options;
        o.start.clear();
        if (!best.empty()) {
            // the incumbent agrees with the fixings unless it was given by the caller: then start without it
            bool agrees = true;
            for (i64 j = 0; j < w.n && agrees; ++j)
                if (!kept[j]) agrees = best[j] == xfix[j];
            if (agrees)
                for (const i64 j : keep) o.start.push_back(best[j]);
        }
        o.time_limit = time_left();
        if (sub.ncol > kWideMip) o.presolve = Presolve::off;
        const bool fits = !timed || nkeep <= kWideMip;
        if (o.time_limit > 0 && fits) {
            const SolveResult mip = solve_highs(sub, o);
            res.nodes = mip.nodes;
            if (mip.has_solution && mip.objective < best_obj) {
                best.assign(static_cast<std::size_t>(w.ncol), 0.0);
                for (i64 j = 0; j < w.n; ++j) best[j] = xfix[j];
                for (std::size_t k = 0; k < keep.size(); ++k) best[keep[k]] = mip.x[k];
                best_obj = mip.objective;
            }
            if (std::isfinite(mip.bound)) bound = std::max(bound, mip.bound);
            mip_optimal = mip.status == "optimal" || mip.status == "infeasible";
            if (mip.status == "infeasible") bound = std::max(bound, best_obj);
        }
    }
    if (best.empty()) return finish({}, kInf, bound, mip_optimal ? "infeasible" : "time_limit");
    const bool ok = relative_gap(best_obj, bound) <= options.gap || mip_optimal;
    return finish(best, best_obj, std::min(bound, best_obj), ok ? "optimal" : "time_limit");
}

}  // namespace opt
