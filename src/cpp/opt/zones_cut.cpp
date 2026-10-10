// SPDX-License-Identifier: GPL-3.0-or-later

#include "opt/zones_cut.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <stdexcept>
#include <utility>

#include "opt/bundle.hpp"
#include "opt/maxflow.hpp"

namespace opt::zcut {

namespace {

using Clock = std::chrono::steady_clock;

double relative_gap(double upper, double lower) {
    if (!std::isfinite(upper) || !std::isfinite(lower)) return kInf;
    const double g = upper - lower;
    if (g <= 0) return 0.0;
    return g / std::max(std::abs(upper), 1e-10);
}

double tol_of(double v) { return 1e-9 * std::max(1.0, std::abs(v)); }

// Solutions as every unit's zone (-1: none): their objective, repair and local search.
struct Search {
    const Input& in;
    i64 n, nz, ncol, m;
    std::vector<i64> adj_start;  // the pairs touching every column
    std::vector<int> adj_col;
    std::vector<double> adj_w;
    std::vector<int> lock;       // every unit's locked zone (-1: none)

    explicit Search(const Input& input) : in(input), n(input.n), nz(input.nz), ncol(input.n * input.nz),
                                          m(static_cast<i64>(input.target.size())) {
        adj_start.assign(static_cast<std::size_t>(ncol + 1), 0);
        for (std::size_t k = 0; k < in.pi.size(); ++k) {
            ++adj_start[in.pi[k] + 1];
            ++adj_start[in.pj[k] + 1];
        }
        for (i64 j = 0; j < ncol; ++j) adj_start[j + 1] += adj_start[j];
        adj_col.resize(static_cast<std::size_t>(adj_start.back()));
        adj_w.resize(adj_col.size());
        std::vector<i64> pos(adj_start.begin(), adj_start.end() - 1);
        for (std::size_t k = 0; k < in.pi.size(); ++k) {
            adj_col[pos[in.pi[k]]] = in.pj[k];
            adj_w[pos[in.pi[k]]++] = in.pw[k];
            adj_col[pos[in.pj[k]]] = in.pi[k];
            adj_w[pos[in.pj[k]]++] = in.pw[k];
        }
        lock.assign(static_cast<std::size_t>(n), -1);
        for (i64 j = 0; j < ncol; ++j)
            if (in.lb[j] >= 1.0) lock[j % n] = static_cast<int>(j / n);
    }

    bool open(i64 i, int z) const { return in.ub[z * n + i] > 0; }

    double objective(const std::vector<int>& zone) const {
        double v = 0.0;
        for (i64 i = 0; i < n; ++i)
            if (zone[i] >= 0) v += in.cost[zone[i] * n + i];
        for (std::size_t k = 0; k < in.pi.size(); ++k) {
            const bool a = zone[in.pi[k] % n] == in.pi[k] / n, b = zone[in.pj[k] % n] == in.pj[k] / n;
            if (a != b) v += in.pw[k];
        }
        return v;
    }

    // the change of the objective when unit i moves from its zone to zone b (-1: none)
    double delta(const std::vector<int>& zone, i64 i, int b) const {
        const int a = zone[i];
        if (a == b) return 0.0;
        const i64 ca = a >= 0 ? a * n + i : -1, cb = b >= 0 ? b * n + i : -1;
        double d = (b >= 0 ? in.cost[cb] : 0.0) - (a >= 0 ? in.cost[ca] : 0.0);
        auto x_now = [&](i64 j) { return zone[j % n] == j / n; };
        auto x_new = [&](i64 j) { return j == cb ? true : j == ca ? false : x_now(j); };
        for (const i64 c : {ca, cb}) {
            if (c < 0) continue;
            for (i64 k = adj_start[c]; k < adj_start[c + 1]; ++k) {
                const i64 o = adj_col[k];
                if ((o == ca || o == cb) && o < c) continue;  // a pair between the two columns: once
                const bool old_cut = x_now(c) != x_now(o), new_cut = x_new(c) != x_new(o);
                d += adj_w[k] * (static_cast<double>(new_cut) - static_cast<double>(old_cut));
            }
        }
        return d;
    }

    // gain toward the unmet targets of moving unit i to zone b; -1 if a met target would become unmet
    double gain(const std::vector<int>& zone, const std::vector<double>& act, i64 i, int b,
                std::vector<double>& dv, std::vector<i64>& touched) const {
        const int a = zone[i];
        touched.clear();
        auto add = [&](i64 c, double s) {
            for (i64 k = in.t_start[c]; k < in.t_start[c + 1]; ++k) {
                const i64 r = in.t_index[k];
                if (dv[r] == 0.0) touched.push_back(r);
                dv[r] += s * in.t_value[k];
                if (dv[r] == 0.0) dv[r] = 1e-300;  // keep it marked
            }
        };
        if (b >= 0) add(b * n + i, 1.0);
        if (a >= 0) add(a * n + i, -1.0);
        double g = 0.0;
        bool ok = true;
        for (const i64 r : touched) {
            const double v = dv[r] == 1e-300 ? 0.0 : dv[r];
            const double left = in.target[r] - act[r];
            const bool unmet = left > tol_of(in.target[r]);
            if (v > 0 && unmet) g += std::min(v, left) / std::max(in.target[r], 1e-300);
            if (v < 0 && !unmet && act[r] + v < in.target[r] - tol_of(in.target[r])) ok = false;
            dv[r] = 0.0;
        }
        return ok ? g : -1.0;
    }

    void apply(std::vector<int>& zone, std::vector<double>& act, i64 i, int b) const {
        const int a = zone[i];
        if (a >= 0)
            for (i64 k = in.t_start[a * n + i]; k < in.t_start[a * n + i + 1]; ++k) act[in.t_index[k]] -= in.t_value[k];
        if (b >= 0)
            for (i64 k = in.t_start[b * n + i]; k < in.t_start[b * n + i + 1]; ++k) act[in.t_index[k]] += in.t_value[k];
        zone[i] = b;
    }

    std::vector<double> activity(const std::vector<int>& zone) const {
        std::vector<double> act(static_cast<std::size_t>(m), 0.0);
        for (i64 i = 0; i < n; ++i)
            if (zone[i] >= 0)
                for (i64 k = in.t_start[zone[i] * n + i]; k < in.t_start[zone[i] * n + i + 1]; ++k)
                    act[in.t_index[k]] += in.t_value[k];
        return act;
    }

    // repair the unmet targets, then improve; false when the targets cannot be met
    bool repair_and_improve(std::vector<int>& zone) const {
        std::vector<double> act = activity(zone);
        std::vector<double> dv(static_cast<std::size_t>(m), 0.0);
        std::vector<i64> touched;
        auto unmet_count = [&] {
            i64 c = 0;
            for (i64 r = 0; r < m; ++r) c += act[r] < in.target[r] - tol_of(in.target[r]);
            return c;
        };
        i64 open_rows = unmet_count();
        if (open_rows > 0) {
            auto key_of = [&](i64 i, int b) {
                const double g = gain(zone, act, i, b, dv, touched);
                if (g <= 0) return kInf;
                const double d = delta(zone, i, b);
                return d <= 0 ? -g : d / g;
            };
            using Item = std::pair<double, i64>;
            std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
            for (i64 i = 0; i < n; ++i) {
                if (lock[i] >= 0) continue;
                for (int b = 0; b < nz; ++b) {
                    if (b == zone[i] || !open(i, b)) continue;
                    const double k = key_of(i, b);
                    if (std::isfinite(k)) heap.push({k, i * nz + b});
                }
            }
            while (open_rows > 0) {
                if (heap.empty()) return false;
                const auto [k0, mv] = heap.top();
                heap.pop();
                const i64 i = mv / nz;
                const int b = static_cast<int>(mv % nz);
                if (zone[i] == b) continue;
                const double k = key_of(i, b);
                if (!std::isfinite(k)) continue;
                if (!heap.empty() && k > heap.top().first + 1e-12 * std::abs(heap.top().first) + 1e-300) {
                    heap.push({k, mv});
                    continue;
                }
                apply(zone, act, i, b);
                open_rows = unmet_count();
                for (int q = 0; q < nz; ++q)
                    if (q != b && open(i, q)) {
                        const double kk = key_of(i, q);
                        if (std::isfinite(kk)) heap.push({kk, i * nz + q});
                    }
            }
        }
        // local search: every unit's best move that lowers the objective and keeps the targets met
        double scale = 0.0;
        for (const double c : in.cost) scale = std::max(scale, std::abs(c));
        const double eps = 1e-12 * std::max(1.0, scale);
        for (int pass = 0; pass < 20; ++pass) {
            bool changed = false;
            for (i64 i = 0; i < n; ++i) {
                if (lock[i] >= 0) continue;
                int best_b = zone[i];
                double best_d = -eps;
                for (int b = in.mandatory ? 0 : -1; b < nz; ++b) {
                    if (b == zone[i] || (b >= 0 && !open(i, b))) continue;
                    const double d = delta(zone, i, b);
                    if (d >= best_d) continue;
                    if (gain(zone, act, i, b, dv, touched) < 0) continue;
                    best_d = d;
                    best_b = b;
                }
                if (best_b != zone[i]) {
                    apply(zone, act, i, best_b);
                    changed = true;
                }
            }
            if (!changed) break;
        }
        return unmet_count() == 0;
    }
};

// Expansion moves (alpha-expansion, Boykov, Veksler & Zabih 2001): for every zone z in turn (and none, without
// mandatory allocation) every unit keeps its zone or moves to z, by one minimum cut over the units of the
// objective plus the targets' Lagrangian penalty at y (pair terms that the move makes non-submodular are
// truncated: a heuristic, the result is evaluated exactly); the targets then repaired and the units improved,
// the move kept when the objective falls. Rounds until one changes nothing (or the deadline).
void expansions(const Search& s, const std::vector<double>& y, std::vector<int>& zone, double& value,
                Clock::time_point deadline, bool verbose) {
    const Input& in = s.in;
    const i64 n = s.n, nz = s.nz, ncol = s.ncol;
    std::vector<double> ay(static_cast<std::size_t>(ncol));
    for (i64 j = 0; j < ncol; ++j) {
        double v = in.cost[j];
        for (i64 k = in.t_start[j]; k < in.t_start[j + 1]; ++k) v -= y[in.t_index[k]] * in.t_value[k];
        ay[j] = v;
    }
    auto label_cost = [&](i64 i, int z) { return z >= 0 ? ay[z * n + i] : 0.0; };
    // the units next to a unit in zone z (by the pairs between their columns)
    auto near_zone = [&](int z, std::vector<std::uint8_t>& near) {
        near.assign(static_cast<std::size_t>(n), 0);
        for (std::size_t k = 0; k < in.pi.size(); ++k) {
            const i64 ua = in.pi[k] % n, ub = in.pj[k] % n;
            if (zone[ua] == z) near[ub] = 1;
            if (zone[ub] == z) near[ua] = 1;
        }
    };
    std::vector<std::uint8_t> near;
    for (int round = 0; round < 10; ++round) {
        bool improved = false;
        // moves of every unit; of the selected units only (zones swapped, units out); of those and the units next
        // to the zone (it grows by its edge): the Lagrangian penalty does not saturate at the targets, so the
        // wide move often overshoots and the narrow ones find what it misses
        for (int mode = 0; mode < 3; ++mode)
        for (int z = in.mandatory ? 0 : -1; z < nz; ++z) {
            if (Clock::now() >= deadline) return;
            if (mode == 2 && z >= 0) near_zone(z, near);
            std::vector<std::uint8_t> movable(static_cast<std::size_t>(n), 0);
            bool any = false;
            for (i64 i = 0; i < n; ++i) {
                bool ok = s.lock[i] < 0 && zone[i] != z && (z < 0 || s.open(i, z));
                if (mode == 1) ok = ok && zone[i] >= 0;
                if (mode == 2) ok = ok && (zone[i] >= 0 || (z >= 0 && near[i]));
                movable[i] = ok;
                any |= movable[i] != 0;
            }
            if (!any) continue;
            // unary: cost of moving (t = 1) minus cost of staying
            std::vector<double> a1(static_cast<std::size_t>(n), 0.0);
            for (i64 i = 0; i < n; ++i)
                if (movable[i]) a1[i] = label_cost(i, z) - label_cost(i, zone[i]);
            // pairs between the units: their 2 x 2 energies, as unary terms and one edge each
            std::vector<int> ei, ej;
            std::vector<double> cij, cji;
            auto x_of = [&](i64 c, i64 u, int t) {
                const int lab = t && movable[u] ? z : zone[u];
                return lab == c / n;
            };
            for (std::size_t k = 0; k < in.pi.size(); ++k) {
                const i64 ca = in.pi[k], cb = in.pj[k], ua = ca % n, ub = cb % n;
                const double w = in.pw[k];
                if (!movable[ua] && !movable[ub]) continue;
                double e[2][2];
                for (int ta = 0; ta < 2; ++ta)
                    for (int tb = 0; tb < 2; ++tb)
                        e[ta][tb] = w * static_cast<double>(x_of(ca, ua, ta) != x_of(cb, ub, tb));
                if (ua == ub) {  // two columns of one unit: a unary term
                    a1[ua] += e[1][1] - e[0][0];
                    continue;
                }
                if (!movable[ub]) {
                    a1[ua] += e[1][0] - e[0][0];
                    continue;
                }
                if (!movable[ua]) {
                    a1[ub] += e[0][1] - e[0][0];
                    continue;
                }
                const double A = e[0][0], B = e[0][1], C = e[1][0], D = e[1][1];
                a1[ua] += C - A;
                a1[ub] += D - C;
                const double lam = B + C - A - D;  // paid when ua stays and ub moves: the edge ub -> ua
                if (lam > 0) {
                    ei.push_back(static_cast<int>(ua));
                    ej.push_back(static_cast<int>(ub));
                    cij.push_back(0.0);
                    cji.push_back(lam);
                }
            }
            MaxFlow flow(static_cast<int>(n), ei, ej, cij, cji);
            std::vector<double> src(static_cast<std::size_t>(n), 0.0), snk(src);
            for (i64 i = 0; i < n; ++i) {
                if (!movable[i]) {
                    snk[i] = kInf;
                    continue;
                }
                if (a1[i] > 0) snk[i] = a1[i];
                else src[i] = -a1[i];
            }
            for (i64 i = 0; i < n; ++i)
                if (!std::isfinite(snk[i])) snk[i] = 1e300;
            flow.solve(src, snk);
            std::vector<int> trial(zone);
            bool moved = false;
            for (i64 i = 0; i < n; ++i)
                if (movable[i] && flow.source_side(static_cast<int>(i))) {
                    trial[i] = z;
                    moved = true;
                }
            i64 nmoved = 0;
            for (i64 i = 0; i < n; ++i) nmoved += trial[i] != zone[i];
            const double before_repair = s.objective(trial);
            if (!moved || !s.repair_and_improve(trial)) {
                if (verbose) std::fprintf(stderr, "expansion %d: %lld moved, repair failed\n", z, static_cast<long long>(nmoved));
                continue;
            }
            const double v = s.objective(trial);
            if (verbose)
                std::fprintf(stderr, "expansion %d (round %d): %lld moved, %.6g before repair, %.6g after (best %.6g)\n", z,
                             round, static_cast<long long>(nmoved), before_repair, v, value);
            if (v < value - 1e-12 * std::max(1.0, std::abs(value))) {
                value = v;
                zone = std::move(trial);
                improved = true;
            }
        }
        if (!improved) return;
    }
}

}  // namespace

Result solve(const Input& in, const Options& o, const std::vector<double>& start) {
    const Clock::time_point t0 = Clock::now();
    const Clock::time_point deadline =
        o.time_limit >= 0 ? t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(o.time_limit))
                          : Clock::time_point::max();
    auto seconds = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    const i64 n = in.n, nz = in.nz, ncol = n * nz, m = static_cast<i64>(in.target.size());
    if (static_cast<i64>(in.cost.size()) != ncol || static_cast<i64>(in.t_start.size()) != ncol + 1)
        throw std::invalid_argument("zones cut: sizes do not match");
    for (const double w : in.pw)
        if (w < 0) throw std::invalid_argument("the zones cut engine needs pair weights >= 0");
    Result res;
    // targets the open columns cannot meet even without the GUB rows
    {
        std::vector<double> reach(static_cast<std::size_t>(m), 0.0);
        for (i64 j = 0; j < ncol; ++j)
            if (in.ub[j] > 0)
                for (i64 k = in.t_start[j]; k < in.t_start[j + 1]; ++k) reach[in.t_index[k]] += in.t_value[k];
        for (i64 r = 0; r < m; ++r)
            if (reach[r] < in.target[r] - tol_of(in.target[r])) {
                res.status = "infeasible";
                res.seconds = seconds();
                return res;
            }
    }
    const Search search(in);
    double wsum = 0.0;
    for (const double w : in.pw) wsum += w;
    MaxFlow flow(static_cast<int>(ncol), in.pi, in.pj, in.pw, in.pw);
    double best = kInf;
    std::vector<int> best_zone;
    auto consider = [&](std::vector<int> zone) {
        if (!search.repair_and_improve(zone)) return;
        const double v = search.objective(zone);
        if (v < best) {
            best = v;
            best_zone = std::move(zone);
        }
    };
    if (!start.empty()) {
        if (static_cast<i64>(start.size()) != ncol) throw std::invalid_argument("one start value per column");
        std::vector<int> zone(static_cast<std::size_t>(n), -1);
        for (i64 j = 0; j < ncol; ++j)
            if (start[j] > 0.5 && in.ub[j] > 0 && zone[j % n] < 0) zone[j % n] = static_cast<int>(j / n);
        for (i64 i = 0; i < n; ++i)
            if (search.lock[i] >= 0) zone[i] = search.lock[i];
        consider(std::move(zone));
    }

    std::vector<double> u(static_cast<std::size_t>(ncol)), src(u.size()), snk(u.size());
    std::vector<double> ay(static_cast<std::size_t>(ncol));  // c - A'y of every column
    std::vector<double> mu(static_cast<std::size_t>(n), 0.0);  // the GUB rows' multipliers (kept between calls)
    std::vector<std::uint8_t> cut(static_cast<std::size_t>(ncol), 0), best_cut(cut);
    int evaluations = 0;
    double best_l = -kInf;
    bool force = false;  // a solution from the next cut whatever its bound
    constexpr int kInner = 10;  // subgradient steps on the GUB multipliers per evaluation
    // L(y, mu) = y't - sum_i mu_i (+ mu_i for "= 1" rows) + min cut of c - A'y + mu: valid for any y >= 0 and
    // mu >= 0 (mu free with mandatory allocation); for given y, a few subgradient steps on mu from the last mu
    auto cut_value = [&](double& e) {
        double big = 1.0 + wsum;
        for (i64 j = 0; j < ncol; ++j) {
            u[j] = ay[j] + mu[j % n];
            big += std::abs(u[j]);
        }
        for (i64 j = 0; j < ncol; ++j) {
            src[j] = snk[j] = 0;
            if (in.lb[j] >= 1.0) src[j] = big;
            else if (in.ub[j] <= 0.0) snk[j] = big;
            else if (u[j] > 0) snk[j] = u[j];
            else src[j] = -u[j];
        }
        flow.solve(src, snk);
        e = 0.0;
        for (i64 j = 0; j < ncol; ++j) {
            cut[j] = static_cast<std::uint8_t>(flow.source_side(static_cast<int>(j)));
            if (cut[j]) e += u[j];
        }
        for (std::size_t k = 0; k < in.pw.size(); ++k)
            if (cut[in.pi[k]] != cut[in.pj[k]]) e += in.pw[k];
        for (i64 i = 0; i < n; ++i) e -= mu[i];
    };
    auto lagrangian = [&](const std::vector<double>& y, std::vector<double>& g) {
        for (i64 j = 0; j < ncol; ++j) {
            double v = in.cost[j];
            for (i64 k = in.t_start[j]; k < in.t_start[j + 1]; ++k) v -= y[in.t_index[k]] * in.t_value[k];
            ay[j] = v;
        }
        double yt = 0.0;
        for (i64 r = 0; r < m; ++r) yt += y[r] * in.target[r];
        double l_best = -kInf;
        std::vector<double> mu_best(mu);
        std::vector<double> sub(static_cast<std::size_t>(n));
        for (int it = 0; it < kInner; ++it) {
            double e;
            cut_value(e);
            const double l = e + yt;
            if (l > l_best) {
                l_best = l;
                mu_best = mu;
                best_cut = cut;
            }
            if (it + 1 == kInner) break;
            // subgradient of the GUB rows: sum_z x_zi - 1; Polyak's step toward the best solution (or 10% up)
            double norm = 0.0;
            for (i64 i = 0; i < n; ++i) {
                double sx = 0.0;
                for (int z = 0; z < nz; ++z) sx += cut[z * n + i];
                sub[i] = sx - 1.0;
                if (!in.mandatory && mu[i] <= 0 && sub[i] < 0) sub[i] = 0.0;  // projected (mu >= 0)
                norm += sub[i] * sub[i];
            }
            if (norm == 0.0) break;  // the cut meets every GUB row: L(y, mu) is L(y) of the full problem
            const double goal = std::isfinite(best) ? best : l + 0.1 * std::max(1.0, std::abs(l));
            const double step = 0.5 * std::max(goal - l, 1e-9 * std::max(1.0, std::abs(l))) / norm;
            for (i64 i = 0; i < n; ++i) {
                mu[i] += step * sub[i];
                if (!in.mandatory) mu[i] = std::max(mu[i], 0.0);
            }
        }
        // the bound and the supergradient of the best step; mu goes on from the last step (warm start)
        cut = best_cut;
        g.assign(in.target.begin(), in.target.end());
        for (i64 j = 0; j < ncol; ++j)
            if (cut[j])
                for (i64 k = in.t_start[j]; k < in.t_start[j + 1]; ++k) g[in.t_index[k]] -= in.t_value[k];
        const double l = l_best;
        ++evaluations;
        if (force || l > best_l || evaluations % 4 == 1) {
            std::vector<int> zone(static_cast<std::size_t>(n), -1);
            for (i64 i = 0; i < n; ++i) {
                if (search.lock[i] >= 0) {
                    zone[i] = search.lock[i];
                    continue;
                }
                double bu = kInf;
                for (int z = 0; z < nz; ++z) {
                    const i64 j = z * n + i;
                    const bool pick = cut[j] || in.mandatory;
                    if (pick && in.ub[j] > 0 && ay[j] < bu) {
                        bu = ay[j];
                        zone[i] = z;
                    }
                }
            }
            consider(std::move(zone));
        }
        best_l = std::max(best_l, l);
        return l;
    };
    // the starting multipliers: every target's cost per amount in a greedy cover of it alone
    std::vector<double> y(static_cast<std::size_t>(m), 0.0);
    {
        std::vector<std::vector<std::pair<double, double>>> by(static_cast<std::size_t>(m));
        for (i64 j = 0; j < ncol; ++j)
            if (in.ub[j] > 0)
                for (i64 k = in.t_start[j]; k < in.t_start[j + 1]; ++k)
                    if (in.t_value[k] > 0)
                        by[in.t_index[k]].emplace_back(std::max(in.cost[j], 0.0) / in.t_value[k], in.t_value[k]);
        for (i64 r = 0; r < m; ++r) {
            auto& v = by[r];
            std::sort(v.begin(), v.end());
            double held = 0, spent = 0;
            for (const auto& [ratio, a] : v) {
                if (held >= in.target[r]) break;
                held += a;
                spent += ratio * a;
            }
            y[r] = in.target[r] > 0 && held > 0 ? spent / held : 0.0;
        }
    }
    const BundleResult br = bundle_maximize(
        y, lagrangian, [&](double bound) { return relative_gap(best, bound) <= o.gap; }, evaluations,
        o.max_evaluations, deadline);
    const double bound = std::max(br.bound, best_l);
    // primal search: the cuts of the best multipliers scaled up (more units in, less to repair)
    std::vector<double> g;
    for (const double theta : {1.02, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 3.0}) {
        if (relative_gap(best, bound) <= o.gap || Clock::now() >= deadline) break;
        std::vector<double> yt(br.centre);
        for (double& v : yt) v *= theta;
        force = true;  // a solution from every one of these cuts
        lagrangian(yt, g);
        force = false;
    }
    if (!best_zone.empty() && relative_gap(best, bound) > o.gap && Clock::now() < deadline)
        for (const double theta : {1.0, 1.25, 1.5, 2.0, 3.0}) {  // the targets' penalty scaled up in turn
            if (relative_gap(best, bound) <= o.gap || Clock::now() >= deadline) break;
            std::vector<double> yt(br.centre);
            for (double& v : yt) v *= theta;
            expansions(search, yt, best_zone, best, deadline, o.verbose);
        }
    if (!best_zone.empty()) {
        res.x.assign(static_cast<std::size_t>(ncol), 0.0);
        for (i64 i = 0; i < n; ++i)
            if (best_zone[i] >= 0) res.x[best_zone[i] * n + i] = 1.0;
    }
    res.objective = best;
    res.bound = bound;
    res.gap = relative_gap(best, bound);
    res.evaluations = evaluations;
    res.status = best_zone.empty() ? (br.why == "time" ? "time_limit" : "stalled")
                 : res.gap <= o.gap ? "optimal" : (br.why == "time" ? "time_limit" : "stalled");
    res.seconds = seconds();
    if (o.verbose)
        std::fprintf(stderr, "zones cut engine: %d evaluations, best %.10g, bound %.10g, gap %.3g (%s), %.2f s\n",
                     evaluations, best, bound, res.gap, br.why.c_str(), res.seconds);
    return res;
}

}  // namespace opt::zcut
