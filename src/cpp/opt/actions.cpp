// SPDX-License-Identifier: GPL-3.0-or-later
//
// The arithmetic follows prioriactions's C++ (and landschaft's former Python port of it) operation by
// operation: the sums in the same order, so that compile() is bit for bit getModelInfo()'s model.

#include "opt/actions.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "opt/closure.hpp"

namespace opt::actions {

namespace {

void need(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(what);
}

// Rows, columns and entries of a model being built, in prioriactions's order.
struct Builder {
    std::vector<i64> ri, ci;
    std::vector<double> vi;
    std::vector<double> rhs;
    std::vector<std::int8_t> sense;
    std::vector<double> obj, lb, ub;
    std::vector<std::uint8_t> integer;

    i64 nrow() const { return static_cast<i64>(rhs.size()); }
    i64 ncol() const { return static_cast<i64>(obj.size()); }
    void entry(i64 r, i64 c, double v) {
        ri.push_back(r);
        ci.push_back(c);
        vi.push_back(v);
    }
    void row(double b, Sense s) {
        rhs.push_back(b);
        sense.push_back(s);
    }
    void column(double o, bool binary, double l = 0.0, double u = 1.0) {
        obj.push_back(o);
        lb.push_back(l);
        ub.push_back(u);
        integer.push_back(binary ? 1 : 0);
    }

    // compressed rows, columns ascending, duplicates summed in the order they came
    Compiled finish(bool maximize) {
        Compiled out;
        Model& m = out.model;
        m.nrow = nrow();
        m.ncol = ncol();
        const std::size_t nnz = ri.size();
        std::vector<i64> count(static_cast<std::size_t>(m.nrow) + 1, 0);
        for (const i64 r : ri) ++count[static_cast<std::size_t>(r) + 1];
        for (std::size_t r = 0; r < static_cast<std::size_t>(m.nrow); ++r) count[r + 1] += count[r];
        std::vector<i64> order(nnz);
        {
            std::vector<i64> next(count.begin(), count.end() - 1);
            for (std::size_t k = 0; k < nnz; ++k) order[static_cast<std::size_t>(next[static_cast<std::size_t>(ri[k])]++)] =
                static_cast<i64>(k);
        }
        m.start.assign(1, 0);
        for (i64 r = 0; r < m.nrow; ++r) {
            auto b = order.begin() + count[static_cast<std::size_t>(r)];
            auto e = order.begin() + count[static_cast<std::size_t>(r) + 1];
            std::stable_sort(b, e, [&](i64 x, i64 y) { return ci[static_cast<std::size_t>(x)] < ci[static_cast<std::size_t>(y)]; });
            for (auto it = b; it != e; ++it) {
                const i64 c = ci[static_cast<std::size_t>(*it)];
                const double v = vi[static_cast<std::size_t>(*it)];
                if (static_cast<i64>(m.index.size()) > m.start.back() && m.index.back() == c)
                    m.value.back() += v;
                else {
                    m.index.push_back(c);
                    m.value.push_back(v);
                }
            }
            m.start.push_back(static_cast<i64>(m.index.size()));
        }
        m.obj = std::move(obj);
        m.lb = std::move(lb);
        m.ub = std::move(ub);
        m.integer = std::move(integer);
        m.maximize = maximize;
        m.row_lo.resize(static_cast<std::size_t>(m.nrow));
        m.row_hi.resize(static_cast<std::size_t>(m.nrow));
        for (std::size_t r = 0; r < static_cast<std::size_t>(m.nrow); ++r) {
            m.row_lo[r] = sense[r] == kLe ? -kInf : rhs[r];
            m.row_hi[r] = sense[r] == kGe ? kInf : rhs[r];
        }
        out.sense = std::move(sense);
        out.rhs = std::move(rhs);
        return out;
    }
};

// The action of every (unit, threat), -1 when none.
std::vector<i64> action_table(const Input& in) {
    std::vector<i64> t(static_cast<std::size_t>(in.n_units) * static_cast<std::size_t>(in.n_threats), -1);
    for (i64 a = 0; a < in.n_actions(); ++a)
        t[static_cast<std::size_t>(in.act_unit[a] * in.n_threats + in.act_threat[a])] = a;
    return t;
}

// Every action's boundary to the units with the same threat: sum_{u2 != u1} M[u1, u2] over the
// units of its threat, ascending (prioriactions's dist_threats_extended column), with the pairs.
template <class F>
void action_neighbours(const Input& in, const Boundary& bm, const std::vector<i64>& table, i64 a, F&& f) {
    const i64 u1 = in.act_unit[a], t = in.act_threat[a];
    for (i64 k = bm.start[static_cast<std::size_t>(u1)]; k < bm.start[static_cast<std::size_t>(u1) + 1]; ++k) {
        const i64 u2 = bm.index[static_cast<std::size_t>(k)];
        const double v = bm.value[static_cast<std::size_t>(k)];
        if (u2 == u1 || v == 0.0) continue;
        const i64 a2 = table[static_cast<std::size_t>(u2 * in.n_threats + t)];
        if (a2 >= 0) f(a2, v);
    }
}

}  // namespace

void Input::check() const {
    const auto n = static_cast<std::size_t>(n_units), nf = static_cast<std::size_t>(n_features),
               nt = static_cast<std::size_t>(n_threats);
    need(monitoring_cost.size() == n && unit_status.size() == n, "one cost and status per unit");
    need(target_recovery.size() == nf && target_conservation.size() == nf, "two targets per feature");
    need(df_feature.size() == df_unit.size() && df_amount.size() == df_unit.size(), "dist_features columns");
    need(blm_actions.size() == nt, "one blm_actions per threat");
    const std::size_t na = act_unit.size();
    need(act_threat.size() == na && act_amount.size() == na && act_cost.size() == na && act_status.size() == na,
         "dist_threats columns");
    need(sens_threat.size() == sens_feature.size() && sens_delta.size() == 4 * sens_feature.size(),
         "sensitivity columns");
    need(bnd_id2.size() == bnd_id1.size() && bnd_value.size() == bnd_id1.size(), "boundary columns");
    for (std::size_t k = 0; k < df_unit.size(); ++k)
        need(df_unit[k] >= 0 && df_unit[k] < n_units && df_feature[k] >= 0 && df_feature[k] < n_features,
             "dist_features ids");
    for (std::size_t k = 0; k < na; ++k)
        need(act_unit[k] >= 0 && act_unit[k] < n_units && act_threat[k] >= 0 && act_threat[k] < n_threats,
             "dist_threats ids");
    for (std::size_t k = 0; k < sens_feature.size(); ++k)
        need(sens_feature[k] >= 0 && sens_feature[k] < n_features && sens_threat[k] >= 0 &&
                 sens_threat[k] < n_threats,
             "sensitivity ids");
    for (std::size_t k = 0; k < bnd_id1.size(); ++k)
        need(bnd_id1[k] >= 0 && bnd_id1[k] < n_units && bnd_id2[k] >= 0 && bnd_id2[k] < n_units, "boundary ids");
}

Pairs pairs(const Input& in) {
    const i64 nf = in.n_features, nt = in.n_threats, n = in.n_units;
    Pairs p;
    const std::size_t np = in.df_unit.size();
    std::vector<i64> order(np);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](i64 x, i64 y) {
        const auto a = static_cast<std::size_t>(x), b = static_cast<std::size_t>(y);
        return in.df_feature[a] != in.df_feature[b] ? in.df_feature[a] < in.df_feature[b] : in.df_unit[a] < in.df_unit[b];
    });
    // the threats of every unit, ascending, with their actions
    std::vector<i64> ustart(static_cast<std::size_t>(n) + 1, 0), uaction(static_cast<std::size_t>(in.n_actions()));
    for (i64 a = 0; a < in.n_actions(); ++a) ++ustart[static_cast<std::size_t>(in.act_unit[a]) + 1];
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) ustart[i + 1] += ustart[i];
    {
        std::vector<i64> next(ustart.begin(), ustart.end() - 1);
        for (i64 a = 0; a < in.n_actions(); ++a) uaction[static_cast<std::size_t>(next[static_cast<std::size_t>(in.act_unit[a])]++)] = a;
        for (i64 i = 0; i < n; ++i)
            std::sort(uaction.begin() + ustart[static_cast<std::size_t>(i)], uaction.begin() + ustart[static_cast<std::size_t>(i) + 1],
                      [&](i64 x, i64 y) { return in.act_threat[static_cast<std::size_t>(x)] < in.act_threat[static_cast<std::size_t>(y)]; });
    }
    // the response's parameters (later sensitivity rows win)
    const auto ft = static_cast<std::size_t>(nf * nt);
    std::vector<double> params(4 * ft, 0.0);
    std::vector<std::uint8_t> harms(ft, 0);
    const std::size_t ns = in.sens_feature.size();
    for (std::size_t k = 0; k < ns; ++k) {
        const auto st = static_cast<std::size_t>(in.sens_feature[k] * nt + in.sens_threat[k]);
        for (std::size_t q = 0; q < 4; ++q) params[q * ft + st] = in.sens_delta[q * ns + k];
        harms[st] = 1;
    }
    p.feature.resize(np);
    p.unit.resize(np);
    p.amount.resize(np);
    p.threatened.assign(np, 0);
    p.sum_alpha.assign(np, 0.0);
    p.start.assign(1, 0);
    std::vector<double> alphas, vars;
    std::vector<i64> acts;
    for (std::size_t k = 0; k < np; ++k) {
        const auto r = static_cast<std::size_t>(order[k]);
        const i64 s = in.df_feature[r], i = in.df_unit[r];
        p.feature[k] = s;
        p.unit[k] = i;
        p.amount[k] = in.df_amount[r];
        alphas.clear();
        vars.clear();
        acts.clear();
        double total = 0.0;
        for (i64 q = ustart[static_cast<std::size_t>(i)]; q < ustart[static_cast<std::size_t>(i) + 1]; ++q) {
            const i64 a = uaction[static_cast<std::size_t>(q)];
            const auto st = static_cast<std::size_t>(s * nt + in.act_threat[static_cast<std::size_t>(a)]);
            if (!harms[st]) continue;
            p.threatened[k] = 1;
            const double x = in.act_amount[static_cast<std::size_t>(a)];
            const double pa = params[st], pb = params[ft + st], pc = params[2 * ft + st], pd = params[3 * ft + st];
            double var, cst;
            if (x <= pa) {
                var = 0.0;
                cst = pd;
            } else if (x >= pb) {
                var = pd - pc;
                cst = pc;
            } else {
                var = ((pa - x) * (pc - pd)) / (pb - pa);
                cst = (pc * (x - pa) - pd * (x - pb)) / (pb - pa);
            }
            const double alpha = 1 - cst;
            total = total + alpha;
            alphas.push_back(alpha);
            vars.push_back(var);
            acts.push_back(a);
        }
        p.sum_alpha[k] = total;
        if (total != 0.0)
            for (std::size_t q = 0; q < acts.size(); ++q) {
                const double v = (vars[q] * alphas[q]) / total;
                p.action.push_back(acts[q]);
                p.coef.push_back(-1 * (vars[q] * alphas[q]) / total);
                p.contribution.push_back(v);
                p.action_status.push_back(in.act_status[static_cast<std::size_t>(acts[q])]);
            }
        p.start.push_back(static_cast<i64>(p.action.size()));
    }
    return p;
}

double Boundary::at(i64 i, i64 j) const {
    const auto b = index.begin() + start[static_cast<std::size_t>(i)], e = index.begin() + start[static_cast<std::size_t>(i) + 1];
    const auto it = std::lower_bound(b, e, j);
    return it != e && *it == j ? value[static_cast<std::size_t>(it - index.begin())] : 0.0;
}

Boundary boundary_matrix(const Input& in) {
    Boundary bm;
    if (!in.has_boundary || in.bnd_id1.empty()) return bm;
    const i64 n = in.n_units;
    // both directions, unless the pair is already there (then only (id1, id2) changes)
    std::unordered_map<i64, double> vals;
    vals.reserve(2 * in.bnd_id1.size());
    for (std::size_t k = 0; k < in.bnd_id1.size(); ++k) {
        const i64 a = in.bnd_id1[k], b = in.bnd_id2[k];
        const double v = in.bnd_value[k];
        auto it = vals.find(a * n + b);
        if (it == vals.end() || it->second == 0.0) {
            vals[a * n + b] = v;
            vals[b * n + a] = v;
        } else {
            it->second = v;
        }
    }
    std::vector<std::pair<i64, double>> e;
    e.reserve(vals.size());
    for (const auto& [key, v] : vals)
        if (v != 0.0) e.emplace_back(key, v);  // arma keeps no zeros
    std::sort(e.begin(), e.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    bm.start.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto& [key, v] : e) {
        ++bm.start[static_cast<std::size_t>(key / n) + 1];
        bm.index.push_back(key % n);
        bm.value.push_back(v);
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) bm.start[i + 1] += bm.start[i];
    return bm;
}

Benefit benefit(const Input& in, const Pairs& p, const std::vector<double>& solution) {
    const auto nf = static_cast<std::size_t>(in.n_features);
    const bool sol = !solution.empty();
    if (sol) need(static_cast<i64>(solution.size()) == in.n_units + in.n_actions(), "one value per unit and action");
    Benefit b;
    b.dist.assign(nf, 0.0);
    b.dist_threatened.assign(nf, 0.0);
    b.conservation.assign(nf, 0.0);
    b.recovery.assign(nf, 0.0);
    const i64 n = in.n_units;
    for (i64 k = 0; k < p.size(); ++k) {
        const auto s = static_cast<std::size_t>(p.feature[static_cast<std::size_t>(k)]);
        const i64 i = p.unit[static_cast<std::size_t>(k)];
        b.dist[s] += 1;
        if (in.unit_status[static_cast<std::size_t>(i)] == 3) continue;
        if (p.threatened[static_cast<std::size_t>(k)]) b.dist_threatened[s] += 1;
        const double amount = p.amount[static_cast<std::size_t>(k)];
        if (p.sum_alpha[static_cast<std::size_t>(k)] != 0.0) {
            for (i64 j = p.start[static_cast<std::size_t>(k)]; j < p.start[static_cast<std::size_t>(k) + 1]; ++j) {
                if (p.action_status[static_cast<std::size_t>(j)] == 3) continue;
                const double v = p.contribution[static_cast<std::size_t>(j)] * amount;
                if (sol)
                    b.recovery[s] = b.recovery[s] + solution[static_cast<std::size_t>(n + p.action[static_cast<std::size_t>(j)])] * v;
                else
                    b.recovery[s] = b.recovery[s] + v;
            }
        } else {
            if (sol)
                b.conservation[s] = b.conservation[s] + solution[static_cast<std::size_t>(i)] * amount;
            else
                b.conservation[s] = b.conservation[s] + amount;
        }
    }
    return b;
}

Compiled compile(const Input& in, const Pairs& p, const Settings& s, bool disaggregate, bool curves) {
    const i64 n = in.n_units, na = in.n_actions(), nf = in.n_features, np = p.size();
    Builder b;
    // --- benefit rows: b_is - sum coef x_ik <= 0, or b_is - w_i <= 0 (no harmful threat with a response)
    for (i64 k = 0; k < np; ++k) {
        const auto q = static_cast<std::size_t>(k);
        b.entry(k, n + na + k, 1.0);
        if (p.sum_alpha[q] != 0.0) {
            for (i64 j = p.start[q]; j < p.start[q + 1]; ++j)
                b.entry(k, n + p.action[static_cast<std::size_t>(j)], p.coef[static_cast<std::size_t>(j)]);
        } else {
            b.entry(k, p.unit[q], -1.0);
        }
        b.row(0.0, kLe);
    }
    // --- activation: sum_k x_ik - |K_i| w_i <= 0 per unit, or x_ik - w_i <= 0 per action
    if (disaggregate) {
        for (i64 a = 0; a < na; ++a) {
            const i64 r = b.nrow();
            b.entry(r, n + a, 1.0);
            b.entry(r, in.act_unit[static_cast<std::size_t>(a)], -1.0);
            b.row(0.0, kLe);
        }
    } else {
        std::vector<std::vector<i64>> by_unit(static_cast<std::size_t>(n));
        for (i64 a = 0; a < na; ++a) by_unit[static_cast<std::size_t>(in.act_unit[static_cast<std::size_t>(a)])].push_back(a);
        for (i64 i = 0; i < n; ++i) {
            auto& acts = by_unit[static_cast<std::size_t>(i)];
            std::stable_sort(acts.begin(), acts.end(), [&](i64 x, i64 y) {
                return in.act_threat[static_cast<std::size_t>(x)] < in.act_threat[static_cast<std::size_t>(y)];
            });
            const i64 r = b.nrow();
            for (const i64 a : acts) b.entry(r, n + a, 1.0);
            b.entry(r, i, -1.0 * static_cast<double>(acts.size()));
            b.row(0.0, kLe);
        }
    }
    // --- locks: units, then actions
    for (i64 i = 0; i < n; ++i) {
        const double st = in.unit_status[static_cast<std::size_t>(i)];
        if (st == 0) continue;
        b.entry(b.nrow(), i, 1.0);
        b.row(st == 2 ? 1.0 : 0.0, kEq);
    }
    for (i64 a = 0; a < na; ++a) {
        const double st = in.act_status[static_cast<std::size_t>(a)];
        if (st == 0) continue;
        b.entry(b.nrow(), n + a, 1.0);
        b.row(st == 2 ? 1.0 : 0.0, kEq);
    }
    // --- objective
    const bool maximize = s.maximize;
    const Boundary bm = boundary_matrix(in);
    const bool units_bnd = !bm.empty() && s.blm != 0;
    std::vector<double> conn_units(static_cast<std::size_t>(n), 0.0);
    if (units_bnd)
        for (i64 c = 0; c < n; ++c)
            for (i64 k = bm.start[static_cast<std::size_t>(c)]; k < bm.start[static_cast<std::size_t>(c) + 1]; ++k)
                if (bm.index[static_cast<std::size_t>(k)] != c)
                    conn_units[static_cast<std::size_t>(c)] = conn_units[static_cast<std::size_t>(c)] + bm.value[static_cast<std::size_t>(k)];
    for (i64 i = 0; i < n; ++i) {
        const double cu = conn_units[static_cast<std::size_t>(i)];
        b.column(maximize ? -s.blm * cu : s.blm * cu + in.monitoring_cost[static_cast<std::size_t>(i)], true);
    }
    const std::vector<i64> table = action_table(in);
    std::vector<double> conn_actions(static_cast<std::size_t>(na), 0.0);
    std::vector<std::pair<std::pair<i64, i64>, double>> pair_terms;
    for (i64 a = 0; a < na; ++a) {
        const i64 t = in.act_threat[static_cast<std::size_t>(a)];
        if (bm.empty() || in.blm_actions[static_cast<std::size_t>(t)] == 0) continue;
        action_neighbours(in, bm, table, a, [&](i64 a2, double v) {
            conn_actions[static_cast<std::size_t>(a)] = conn_actions[static_cast<std::size_t>(a)] + v;
            pair_terms.push_back({{a, a2}, v});
        });
    }
    for (i64 a = 0; a < na; ++a) {
        const double bt = in.blm_actions[static_cast<std::size_t>(in.act_threat[static_cast<std::size_t>(a)])];
        const double ca = conn_actions[static_cast<std::size_t>(a)];
        b.column(maximize ? -bt * ca : bt * ca + in.act_cost[static_cast<std::size_t>(a)], true);
    }
    const int curve = s.curve;
    for (i64 k = 0; k < np; ++k) b.column(maximize && curve == 1 ? p.amount[static_cast<std::size_t>(k)] : 0.0, false);
    Compiled out;
    out.n_pairs = np;
    if (curve != 1) {
        out.first_pow = b.ncol();
        if (maximize) {
            const i64 r = b.nrow();  // prioriactions puts explicit zeros here (the next row's index)
            for (i64 k = 0; k < np; ++k) b.entry(r, out.first_pow + k, 0.0);
        }
        for (i64 k = 0; k < np; ++k) b.column(maximize ? p.amount[static_cast<std::size_t>(k)] : 0.0, false);
    }
    auto triple = [&](i64 col, i64 c1, i64 c2) {  // y <= z1, y <= z2, y >= z1 + z2 - 1
        const i64 r = b.nrow();
        b.entry(r, col, 1.0);
        b.entry(r, c1, -1.0);
        b.entry(r + 1, col, 1.0);
        b.entry(r + 1, c2, -1.0);
        b.entry(r + 2, col, 1.0);
        b.entry(r + 2, c1, -1.0);
        b.entry(r + 2, c2, -1.0);
        b.row(0.0, kLe);
        b.row(0.0, kLe);
        b.row(-1.0, kGe);
    };
    // y[i1, i2]: every directed pair of the boundary matrix (by row c, columns r ascending)
    if (units_bnd)
        for (i64 c = 0; c < n; ++c)
            for (i64 k = bm.start[static_cast<std::size_t>(c)]; k < bm.start[static_cast<std::size_t>(c) + 1]; ++k) {
                const i64 r = bm.index[static_cast<std::size_t>(k)];
                if (r == c) continue;
                const double v = bm.value[static_cast<std::size_t>(k)];
                const i64 col = b.ncol();
                b.column(maximize ? s.blm * (1 * v) : s.blm * (-1 * v), true);
                triple(col, r, c);
            }
    // p[i1, i2, k]: the boundary between two units' actions against the same threat
    for (const auto& [aa, v] : pair_terms) {
        const double bt = in.blm_actions[static_cast<std::size_t>(in.act_threat[static_cast<std::size_t>(aa.first)])];
        const i64 col = b.ncol();
        b.column(bt * (maximize ? v : -1 * v), true);
        triple(col, n + aa.first, n + aa.second);
    }
    // --- targets (two rows per feature: conservation, recovery) or the budget
    if (!maximize) {
        const i64 row0 = b.nrow();
        const i64 tcol = n + na + (curve == 1 ? 0 : np);
        i64 k = 0;
        for (i64 f = 0; f < nf; ++f) {
            const i64 r = row0 + 2 * f;
            bool cons = false, rec = false;
            for (; k < np && p.feature[static_cast<std::size_t>(k)] == f; ++k) {
                const bool th = p.threatened[static_cast<std::size_t>(k)] != 0;
                b.entry(th ? r + 1 : r, tcol + k, p.amount[static_cast<std::size_t>(k)]);
                (th ? rec : cons) = true;
            }
            if (!cons) b.entry(r, 1, 0.0);
            b.row(cons ? in.target_conservation[static_cast<std::size_t>(f)] : 0.0, kGe);
            if (!rec) b.entry(r + 1, 1, 0.0);
            b.row(rec ? in.target_recovery[static_cast<std::size_t>(f)] : 0.0, kGe);
        }
    } else {
        const i64 r = b.nrow();
        for (i64 i = 0; i < n; ++i) b.entry(r, i, in.monitoring_cost[static_cast<std::size_t>(i)]);
        for (i64 a = 0; a < na; ++a) b.entry(r, n + a, in.act_cost[static_cast<std::size_t>(a)]);
        b.row(s.budget, kLe);
    }
    // --- the response curve as a piecewise-linear function (Gurobi's genconpow in prioriactions:
    // FuncPieces 1, FuncPieceLength round(1 / segments, 1), points on the curve), incremental form:
    // x = sum d_k, 0 <= d_k <= L_k, d_k >= L_k z_k, d_{k+1} <= L_{k+1} z_k (z binary), y = sum slope_k d_k
    if (curve != 1 && curves) {
        need(s.segments >= 1, "segments");
        const double step = s.step > 0 ? s.step : 1.0 / s.segments;
        std::vector<double> pts{0.0};
        while (pts.back() + step < 1.0 - 1e-12) pts.push_back(pts.back() + step);
        pts.push_back(1.0);
        const std::size_t K = pts.size() - 1;
        std::vector<double> len(K), slope(K);
        for (std::size_t q = 0; q < K; ++q) {
            len[q] = pts[q + 1] - pts[q];
            slope[q] = (std::pow(pts[q + 1], curve) - std::pow(pts[q], curve)) / len[q];
        }
        const i64 d0 = b.ncol();
        const auto Ki = static_cast<i64>(K);
        for (i64 k = 0; k < np; ++k)
            for (std::size_t q = 0; q < K; ++q) b.column(0.0, false, 0.0, len[q]);
        const i64 z0 = b.ncol();
        for (i64 k = 0; k < np; ++k)
            for (std::size_t q = 0; q + 1 < K; ++q) b.column(0.0, true);
        for (i64 k = 0; k < np; ++k) {
            const i64 dk = d0 + k * Ki, zk = z0 + k * (Ki - 1);
            i64 r = b.nrow();
            b.entry(r, n + na + k, 1.0);
            for (i64 q = 0; q < Ki; ++q) b.entry(r, dk + q, -1.0);
            b.row(0.0, kEq);
            r = b.nrow();
            b.entry(r, out.first_pow + k, 1.0);
            for (i64 q = 0; q < Ki; ++q) b.entry(r, dk + q, -slope[static_cast<std::size_t>(q)]);
            b.row(0.0, kEq);
            for (i64 q = 0; q + 1 < Ki; ++q) {
                r = b.nrow();
                b.entry(r, dk + q, 1.0);
                b.entry(r, zk + q, -len[static_cast<std::size_t>(q)]);
                b.row(0.0, kGe);
                r = b.nrow();
                b.entry(r, dk + q + 1, 1.0);
                b.entry(r, zk + q, -len[static_cast<std::size_t>(q) + 1]);
                b.row(0.0, kLe);
            }
        }
    }
    Compiled c = b.finish(maximize);
    c.n_pairs = out.n_pairs;
    c.first_pow = out.first_pow;
    return c;
}

Compiled equivalent(const Input& in, const Pairs& p, const Settings& s) {
    if (s.curve != 1) return {};
    const i64 n = in.n_units, na = in.n_actions(), nf = in.n_features, np = p.size();
    for (i64 k = 0; k < np; ++k) {
        double sum = 0.0;
        for (i64 j = p.start[static_cast<std::size_t>(k)]; j < p.start[static_cast<std::size_t>(k) + 1]; ++j) {
            const double v = p.contribution[static_cast<std::size_t>(j)];
            if (v < 0) return {};
            sum += v;
        }
        if (sum > 1.0 + 1e-12) return {};
    }
    const bool maximize = s.maximize;
    Builder b;
    Compiled out;
    // the closure engine's view: every unit's and action's own term (a minimisation), the pair terms
    out.node_cost.assign(static_cast<std::size_t>(n + na), 0.0);
    if (!maximize) {
        for (i64 i = 0; i < n; ++i) out.node_cost[static_cast<std::size_t>(i)] = in.monitoring_cost[static_cast<std::size_t>(i)];
        for (i64 a = 0; a < na; ++a) out.node_cost[static_cast<std::size_t>(n + a)] = in.act_cost[static_cast<std::size_t>(a)];
    }
    // columns: units, actions (the locks as bounds)
    const Boundary bm = boundary_matrix(in);
    const bool units_bnd = !bm.empty() && s.blm != 0;
    std::vector<double> conn_units(static_cast<std::size_t>(n), 0.0);
    if (units_bnd)
        for (i64 c = 0; c < n; ++c)
            for (i64 k = bm.start[static_cast<std::size_t>(c)]; k < bm.start[static_cast<std::size_t>(c) + 1]; ++k)
                if (bm.index[static_cast<std::size_t>(k)] != c)
                    conn_units[static_cast<std::size_t>(c)] += bm.value[static_cast<std::size_t>(k)];
    for (i64 i = 0; i < n; ++i) {
        const double st = in.unit_status[static_cast<std::size_t>(i)];
        const double cu = conn_units[static_cast<std::size_t>(i)];
        b.column(maximize ? -s.blm * cu : s.blm * cu + in.monitoring_cost[static_cast<std::size_t>(i)], true,
                 st == 2 ? 1.0 : 0.0, st == 3 ? 0.0 : 1.0);
    }
    const std::vector<i64> table = action_table(in);
    std::vector<std::pair<std::pair<i64, i64>, double>> pair_terms;
    for (i64 a = 0; a < na; ++a) {
        const i64 t = in.act_threat[static_cast<std::size_t>(a)];
        const double bt = in.blm_actions[static_cast<std::size_t>(t)];
        double ca = 0.0;
        if (!bm.empty() && bt != 0)
            action_neighbours(in, bm, table, a, [&](i64 a2, double v) {
                ca += v;
                pair_terms.push_back({{a, a2}, v});
            });
        const double st = in.act_status[static_cast<std::size_t>(a)];
        b.column(maximize ? -bt * ca : bt * ca + in.act_cost[static_cast<std::size_t>(a)], true, st == 2 ? 1.0 : 0.0,
                 st == 3 ? 0.0 : 1.0);
    }
    if (maximize)  // the benefits' amounts on the units and actions they come from
        for (i64 k = 0; k < np; ++k) {
            const auto q = static_cast<std::size_t>(k);
            if (p.sum_alpha[q] != 0.0)
                for (i64 j = p.start[q]; j < p.start[q + 1]; ++j) {
                    const auto c = static_cast<std::size_t>(n + p.action[static_cast<std::size_t>(j)]);
                    const double v = p.contribution[static_cast<std::size_t>(j)] * p.amount[q];
                    b.obj[c] += v;
                    out.node_cost[c] -= v;
                }
            else {
                b.obj[static_cast<std::size_t>(p.unit[q])] += p.amount[q];
                out.node_cost[static_cast<std::size_t>(p.unit[q])] -= p.amount[q];
            }
        }
    // activation, one row per action
    for (i64 a = 0; a < na; ++a) {
        const i64 r = b.nrow();
        b.entry(r, n + a, 1.0);
        b.entry(r, in.act_unit[static_cast<std::size_t>(a)], -1.0);
        b.row(0.0, kLe);
    }
    auto triple = [&](i64 col, i64 c1, i64 c2) {
        const i64 r = b.nrow();
        b.entry(r, col, 1.0);
        b.entry(r, c1, -1.0);
        b.entry(r + 1, col, 1.0);
        b.entry(r + 1, c2, -1.0);
        b.entry(r + 2, col, 1.0);
        b.entry(r + 2, c1, -1.0);
        b.entry(r + 2, c2, -1.0);
        b.row(0.0, kLe);
        b.row(0.0, kLe);
        b.row(-1.0, kGe);
    };
    if (units_bnd)
        for (i64 c = 0; c < n; ++c)
            for (i64 k = bm.start[static_cast<std::size_t>(c)]; k < bm.start[static_cast<std::size_t>(c) + 1]; ++k) {
                const i64 r = bm.index[static_cast<std::size_t>(k)];
                if (r == c) continue;
                const double v = bm.value[static_cast<std::size_t>(k)];
                const i64 col = b.ncol();
                b.column(maximize ? s.blm * v : -s.blm * v, true);
                triple(col, r, c);
                out.prod_a.push_back(r);
                out.prod_b.push_back(c);
                out.pen_i.push_back(c);  // blm v w_c (1 - w_r): unit c's boundary term
                out.pen_j.push_back(r);
                out.pen_w.push_back(s.blm * v);
            }
    for (const auto& [aa, v] : pair_terms) {
        const double bt = in.blm_actions[static_cast<std::size_t>(in.act_threat[static_cast<std::size_t>(aa.first)])];
        const i64 col = b.ncol();
        b.column(maximize ? bt * v : -bt * v, true);
        triple(col, n + aa.first, n + aa.second);
        out.prod_a.push_back(n + aa.first);
        out.prod_b.push_back(n + aa.second);
        out.pen_i.push_back(n + aa.first);
        out.pen_j.push_back(n + aa.second);
        out.pen_w.push_back(bt * v);
    }
    out.side_row0 = b.nrow();
    if (!maximize) {
        const i64 row0 = b.nrow();
        i64 k = 0;
        for (i64 f = 0; f < nf; ++f) {
            const i64 r = row0 + 2 * f;
            bool cons = false, rec = false;
            for (; k < np && p.feature[static_cast<std::size_t>(k)] == f; ++k) {
                const auto q = static_cast<std::size_t>(k);
                const bool th = p.threatened[q] != 0;
                (th ? rec : cons) = true;
                const i64 row = th ? r + 1 : r;
                if (p.sum_alpha[q] != 0.0) {
                    for (i64 j = p.start[q]; j < p.start[q + 1]; ++j)
                        b.entry(row, n + p.action[static_cast<std::size_t>(j)],
                                p.amount[q] * p.contribution[static_cast<std::size_t>(j)]);
                } else {
                    b.entry(row, p.unit[q], p.amount[q]);
                }
            }
            b.row(cons ? in.target_conservation[static_cast<std::size_t>(f)] : 0.0, kGe);
            b.row(rec ? in.target_recovery[static_cast<std::size_t>(f)] : 0.0, kGe);
        }
    } else {
        const i64 r = b.nrow();
        for (i64 i = 0; i < n; ++i) b.entry(r, i, in.monitoring_cost[static_cast<std::size_t>(i)]);
        for (i64 a = 0; a < na; ++a) b.entry(r, n + a, in.act_cost[static_cast<std::size_t>(a)]);
        b.row(s.budget, kLe);
    }
    Compiled c = b.finish(maximize);
    c.n_pairs = np;
    c.node_cost = std::move(out.node_cost);
    c.side_row0 = out.side_row0;
    c.prod_a = std::move(out.prod_a);
    c.prod_b = std::move(out.prod_b);
    c.pen_i = std::move(out.pen_i);
    c.pen_j = std::move(out.pen_j);
    c.pen_w = std::move(out.pen_w);
    return c;
}

std::vector<double> local_benefit(const Pairs& p, i64 n_units, const std::vector<double>& x) {
    std::vector<double> out(static_cast<std::size_t>(p.size()));
    for (i64 k = 0; k < p.size(); ++k) {
        const auto q = static_cast<std::size_t>(k);
        double v = 0.0;
        if (p.sum_alpha[q] != 0.0)
            for (i64 j = p.start[q]; j < p.start[q + 1]; ++j)
                v += p.contribution[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(n_units + p.action[static_cast<std::size_t>(j)])];
        else
            v = x[static_cast<std::size_t>(p.unit[q])];
        out[q] = std::min(1.0, v);
    }
    return out;
}

std::vector<double> costs(const Input& in, const std::vector<double>& units, const std::vector<double>& actions) {
    need(static_cast<i64>(units.size()) == in.n_units && static_cast<i64>(actions.size()) == in.n_actions(),
         "one value per unit and action");
    std::vector<double> out(static_cast<std::size_t>(in.n_threats) + 1, 0.0);
    double cu = 0.0;
    for (i64 i = 0; i < in.n_units; ++i) cu = cu + in.monitoring_cost[static_cast<std::size_t>(i)] * units[static_cast<std::size_t>(i)];
    out[0] = cu;
    // every threat's column, units ascending
    std::vector<i64> order(static_cast<std::size_t>(in.n_actions()));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](i64 x, i64 y) {
        const auto a = static_cast<std::size_t>(x), b = static_cast<std::size_t>(y);
        return in.act_threat[a] != in.act_threat[b] ? in.act_threat[a] < in.act_threat[b] : in.act_unit[a] < in.act_unit[b];
    });
    for (const i64 a : order) {
        const double v = in.act_cost[static_cast<std::size_t>(a)] * actions[static_cast<std::size_t>(a)];
        auto& t = out[static_cast<std::size_t>(in.act_threat[static_cast<std::size_t>(a)]) + 1];
        if (v != 0) t = t + v;
    }
    return out;
}

std::vector<double> connectivity(const Input& in, const std::vector<double>& units, const std::vector<double>& actions) {
    need(static_cast<i64>(units.size()) == in.n_units && static_cast<i64>(actions.size()) == in.n_actions(),
         "one value per unit and action");
    const Boundary bm = boundary_matrix(in);
    const i64 n = in.n_units, na = in.n_actions();
    std::vector<double> out(static_cast<std::size_t>(in.n_threats) + 1, 0.0);
    if (bm.empty()) return out;
    std::vector<double> acc(static_cast<std::size_t>(n), 0.0);
    for (i64 c = 0; c < n; ++c) {
        double conn = 0.0;
        for (i64 k = bm.start[static_cast<std::size_t>(c)]; k < bm.start[static_cast<std::size_t>(c) + 1]; ++k)
            if (bm.index[static_cast<std::size_t>(k)] != c) conn = conn + bm.value[static_cast<std::size_t>(k)];
        acc[static_cast<std::size_t>(c)] = conn * units[static_cast<std::size_t>(c)];
    }
    for (i64 c = 0; c < n; ++c)
        for (i64 k = bm.start[static_cast<std::size_t>(c)]; k < bm.start[static_cast<std::size_t>(c) + 1]; ++k) {
            const i64 r = bm.index[static_cast<std::size_t>(k)];
            if (r != c && units[static_cast<std::size_t>(r)] > 0.99 && units[static_cast<std::size_t>(c)] > 0.99)
                acc[static_cast<std::size_t>(r)] = acc[static_cast<std::size_t>(r)] + (-1 * bm.value[static_cast<std::size_t>(k)]);
        }
    double total = 0.0;
    for (const double v : acc) total = total + v;
    out[0] = total;
    const std::vector<i64> table = action_table(in);
    std::vector<double> per(static_cast<std::size_t>(na), 0.0);
    for (i64 a = 0; a < na; ++a) {
        double sum = 0.0, acc_a = 0.0;
        const double sa = actions[static_cast<std::size_t>(a)];
        action_neighbours(in, bm, table, a, [&](i64 a2, double v) {
            sum = sum + v;
            if (sa > 0.99 && actions[static_cast<std::size_t>(a2)] > 0.99) acc_a = acc_a + (-1 * v);
        });
        per[static_cast<std::size_t>(a)] = acc_a + sum * sa;
    }
    std::vector<i64> order(static_cast<std::size_t>(na));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](i64 x, i64 y) {
        const auto a = static_cast<std::size_t>(x), b = static_cast<std::size_t>(y);
        return in.act_threat[a] != in.act_threat[b] ? in.act_threat[a] < in.act_threat[b] : in.act_unit[a] < in.act_unit[b];
    });
    for (const i64 a : order) {
        const double v = per[static_cast<std::size_t>(a)];
        auto& t = out[static_cast<std::size_t>(in.act_threat[static_cast<std::size_t>(a)]) + 1];
        if (v != 0) t = t + v;
    }
    return out;
}


namespace {

// HiGHS's MIP continues the engine only up to this many free columns when there is a time limit (it
// does not look at the clock in its root heuristics on wide models: IDEAS.md, phase 0)
constexpr i64 kHighsColumns = 100000;
// the core MIPs: the units and actions whose flip costs the bound least, the others at the best solution's
// values (a restriction: its solutions are feasible)
constexpr i64 kCoreMin = 1000, kCoreColumns = 50000;

double gap_of(double upper, double lower) {
    if (!std::isfinite(upper) || !std::isfinite(lower)) return kInf;
    const double g = upper - lower;
    if (g <= 0) return 0.0;
    return g / std::max(std::abs(upper), 1e-10);
}

// The full solution of the equivalent model from its units and actions (the products after them).
std::vector<double> expand(const Compiled& c, std::vector<double> x) {
    const std::size_t base = x.size();
    x.resize(static_cast<std::size_t>(c.model.ncol));
    for (std::size_t k = 0; k < c.prod_a.size(); ++k)
        x[base + k] = x[static_cast<std::size_t>(c.prod_a[k])] * x[static_cast<std::size_t>(c.prod_b[k])];
    return x;
}

// The engine's problem: nodes = the units and actions; requirements = action -> unit; side rows = the
// equivalent model's rows from side_row0 on (>= rows with a right side <= 0 left out: always met).
closure::Problem closure_problem(const Input& in, const Compiled& c) {
    const i64 n = in.n_units, na = in.n_actions(), nn = n + na;
    const Model& m = c.model;
    closure::Problem q;
    q.n = static_cast<int>(nn);
    q.cost = c.node_cost;
    q.fixed.assign(static_cast<std::size_t>(nn), -1);
    for (i64 v = 0; v < nn; ++v) {
        if (m.lb[static_cast<std::size_t>(v)] > 0.5) q.fixed[static_cast<std::size_t>(v)] = 1;
        else if (m.ub[static_cast<std::size_t>(v)] < 0.5) q.fixed[static_cast<std::size_t>(v)] = 0;
    }
    for (i64 a = 0; a < na; ++a) {
        q.req_from.push_back(static_cast<int>(n + a));
        q.req_to.push_back(static_cast<int>(in.act_unit[static_cast<std::size_t>(a)]));
    }
    for (std::size_t k = 0; k < c.pen_i.size(); ++k) {
        q.pen_i.push_back(static_cast<int>(c.pen_i[k]));
        q.pen_j.push_back(static_cast<int>(c.pen_j[k]));
        q.pen_w.push_back(c.pen_w[k]);
    }
    std::vector<std::vector<std::pair<int, double>>> by(static_cast<std::size_t>(nn));
    for (i64 r = c.side_row0; r < m.nrow; ++r) {
        const auto rr = static_cast<std::size_t>(r);
        const bool geq = c.sense[rr] == kGe;
        if (geq && c.rhs[rr] <= 0) continue;
        const int row = q.m++;
        q.rhs.push_back(c.rhs[rr]);
        q.geq.push_back(geq ? 1 : 0);
        for (i64 k = m.start[rr]; k < m.start[rr + 1]; ++k) {
            const i64 col = m.index[static_cast<std::size_t>(k)];
            const double v = m.value[static_cast<std::size_t>(k)];
            if (v != 0.0) by[static_cast<std::size_t>(col)].emplace_back(row, v);
        }
    }
    q.col_start.assign(1, 0);
    for (const auto& col : by) {
        for (const auto& [row, v] : col) {
            q.col_row.push_back(row);
            q.col_val.push_back(v);
        }
        q.col_start.push_back(static_cast<i64>(q.col_row.size()));
    }
    return q;
}

}  // namespace

SolveResult solve(const Input& in, const Pairs& p, const Settings& s, const SolveOptions& o, bool engine) {
    using Clock = std::chrono::steady_clock;
    const Clock::time_point t0 = Clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    const i64 n = in.n_units, na = in.n_actions(), nn = n + na;
    Compiled c = equivalent(in, p, s);
    SolveResult res;
    if (c.model.ncol == 0) {  // the full model (response curves, or coefficients the substitution cannot take)
        const Compiled full = compile(in, p, s, true, true);
        res = solve_highs(full.model, o);
        if (res.has_solution) {
            std::vector<double> x(static_cast<std::size_t>(nn + p.size()));
            for (i64 v = 0; v < nn; ++v) x[static_cast<std::size_t>(v)] = std::round(res.x[static_cast<std::size_t>(v)]);
            for (i64 k = 0; k < p.size(); ++k)
                x[static_cast<std::size_t>(nn + k)] = res.x[static_cast<std::size_t>(nn + k)];
            res.x = std::move(x);
        }
        res.seconds = elapsed();
        return res;
    }
    const Model& m = c.model;
    const double sign = s.maximize ? -1.0 : 1.0;  // the engine minimises
    bool usable = engine;
    for (const double w : c.pen_w) usable = usable && w >= 0;
    std::vector<double> best;  // units and actions
    double best_obj = kInf, bound = -kInf;  // the engine's sense (a minimisation)
    std::vector<std::int8_t> fixable;
    std::vector<double> flip, centre;
    std::string status;
    if (usable) {
        closure::Options eo;
        eo.gap = o.gap;
        eo.time_limit = std::isfinite(o.time_limit) ? o.time_limit : -1.0;
        eo.threads = std::max(1, o.threads);
        eo.verbose = o.verbose;
        std::vector<std::uint8_t> init;
        if (!o.start.empty()) {
            if (static_cast<i64>(o.start.size()) < nn) throw std::invalid_argument("the start needs the units and actions");
            init.resize(static_cast<std::size_t>(nn));
            for (i64 v = 0; v < nn; ++v) init[static_cast<std::size_t>(v)] = o.start[static_cast<std::size_t>(v)] > 0.5;
        }
        const closure::Result er = closure::solve(closure_problem(in, c), eo, init);
        if (er.status == "infeasible") {
            res.status = "infeasible";
            res.seconds = elapsed();
            return res;
        }
        bound = er.bound;
        if (std::isfinite(er.objective)) {
            best_obj = er.objective;
            best.assign(er.z.begin(), er.z.end());
        }
        fixable = er.fixable;
        flip = er.flip;
        centre.assign(er.centre_z.begin(), er.centre_z.end());
        status = er.status;
    }
    if (best.empty() || gap_of(best_obj, bound) > o.gap) {
        double left = o.time_limit - elapsed();
        Model mm = m;
        i64 free_cols = mm.ncol;
        if (!fixable.empty())
            for (i64 v = 0; v < nn; ++v) {
                const std::int8_t f = fixable[static_cast<std::size_t>(v)];
                if (f < 0 || (!best.empty() && best[static_cast<std::size_t>(v)] != f)) continue;  // keep the start
                mm.lb[static_cast<std::size_t>(v)] = mm.ub[static_cast<std::size_t>(v)] = f;
                --free_cols;
            }
        // core MIPs on HiGHS, for a better solution (the bound stays the engine's): free the units and
        // actions where the best solution and the inner minimum at the best multipliers differ, and those
        // whose flip costs the bound least (the near ties of the Lagrangian), the others at their value in
        // the inner minimum; a twentieth of the free columns first, doubled while the gap stays open
        // HiGHS may stop at any solution within the gap of the engine's bound (in the model's sense)
        auto target = [&] {
            if (!std::isfinite(bound)) return -kInf;
            const double b = bound, g = std::min(o.gap, 0.5);
            const double t = (b >= 0 ? b / (1 - g) : b / (1 + g)) + 1e-9 * std::max(1.0, std::abs(b));
            return sign * t;
        };
        i64 size = std::max(kCoreMin, free_cols / 20);
        while (!flip.empty() && !best.empty() && size <= kCoreColumns && left > 0 && gap_of(best_obj, bound) > o.gap) {
            std::vector<std::pair<double, i64>> order;
            i64 differ = 0;
            for (i64 v = 0; v < nn; ++v) {
                const auto q = static_cast<std::size_t>(v);
                if (mm.lb[q] >= mm.ub[q]) continue;
                if (best[q] != centre[q]) {
                    order.emplace_back(-1.0, v);
                    ++differ;
                } else {
                    order.emplace_back(flip[q], v);
                }
            }
            const auto keep = static_cast<std::size_t>(std::max(size, differ));
            if (keep >= order.size()) break;  // no smaller than the whole model
            std::sort(order.begin(), order.end());
            // the best solution differs from the inner minimum only inside the core: it is the core's start
            Model core = mm;
            for (std::size_t k = keep; k < order.size(); ++k) {
                const auto q = static_cast<std::size_t>(order[k].second);
                core.lb[q] = core.ub[q] = centre[q];
            }
            SolveOptions co = o;
            co.time_limit = std::isfinite(left) ? 0.5 * left : 60.0;
            co.start = expand(c, best);
            co.objective_target = target();
            const SolveResult cr = solve_highs(core, co);
            if (o.verbose)
                std::fprintf(stderr, "multi-action: core MIP of %lld columns: %s, %.10g, %.2f s\n",
                             static_cast<long long>(keep), cr.status.c_str(), cr.objective, cr.seconds);
            if (cr.has_solution && sign * cr.objective < best_obj) {
                best_obj = sign * cr.objective;
                best.assign(cr.x.begin(), cr.x.begin() + nn);
                for (double& x : best) x = std::round(x);
            }
            if (cr.status != "optimal" && cr.status != "target") break;
            size = 2 * static_cast<i64>(keep);
            left = o.time_limit - elapsed();
        }
        left = o.time_limit - elapsed();
        if (o.verbose)
            std::fprintf(stderr, "multi-action: %lld of %lld columns left free for HiGHS\n",
                         static_cast<long long>(free_cols), static_cast<long long>(mm.ncol));
        if (gap_of(best_obj, bound) <= o.gap) {
            status = "optimal";
        } else
        if (left > 0 && (!std::isfinite(o.time_limit) || free_cols <= kHighsColumns || best.empty())) {
            SolveOptions ho = o;
            ho.time_limit = left;
            ho.objective_target = target();
            ho.start.clear();
            if (!best.empty()) ho.start = expand(c, best);
            const SolveResult hr = solve_highs(mm, ho);
            if (hr.has_solution) {
                const double v = sign * hr.objective;
                if (v < best_obj) {
                    best_obj = v;
                    best.assign(hr.x.begin(), hr.x.begin() + nn);
                    for (double& x : best) x = std::round(x);
                }
            }
            // HiGHS's bound holds for the model with the fixings: the optimum is above it or is the engine's
            // solution, which no fixed node can improve on
            if (std::isfinite(hr.bound)) {
                const double hb = sign * hr.bound;
                bound = std::max(bound, fixable.empty() ? hb : std::min(hb, best_obj));
            }
            if (hr.status == "infeasible" && best.empty()) {
                res.status = "infeasible";
                res.seconds = elapsed();
                return res;
            }
            status = gap_of(best_obj, bound) <= o.gap ? "optimal" : hr.status;
        } else if (!status.empty() && status != "optimal") {
            status = "time_limit";  // out of time, or too wide for HiGHS's MIP to keep the limit
        }
    }
    res.has_solution = !best.empty();
    res.bound = sign * bound;
    res.gap = gap_of(best_obj, bound);
    res.status = res.has_solution ? (res.gap <= o.gap ? "optimal" : status) : (status.empty() ? "no_solution" : status);
    if (res.has_solution) {
        res.objective = sign * best_obj;
        std::vector<double> x(best);
        const std::vector<double> local = local_benefit(p, n, x);
        x.insert(x.end(), local.begin(), local.end());
        res.x = std::move(x);
    }
    res.seconds = elapsed();
    return res;
}

}  // namespace opt::actions
