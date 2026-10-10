// SPDX-License-Identifier: GPL-3.0-or-later
//
// The ordered allocation (see ordered.hpp): the steps of landschaft's numpy
// port of lulcc's OrderedModel, the candidates ranked by a stable sort and
// drawn with R's generator in the same order.

#include "sim/ordered.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace sim {

namespace {

using i64 = std::int64_t;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// The candidates with a suitability, ranked by a stable sort on the key
// (decreasing suitability when `decreasing`, else increasing).
void rank(std::vector<i64>& cand, std::vector<double>& p, bool decreasing) {
    std::vector<std::size_t> o(cand.size());
    std::iota(o.begin(), o.end(), std::size_t{0});
    std::vector<double> key(cand.size());
    for (std::size_t k = 0; k < cand.size(); ++k) key[k] = std::isnan(p[k]) ? kInf : (decreasing ? -p[k] : p[k]);
    std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return key[a] < key[b]; });
    std::vector<i64> c2;
    std::vector<double> p2;
    c2.reserve(cand.size());
    p2.reserve(cand.size());
    for (const auto k : o) {
        if (std::isnan(p[k])) continue;
        c2.push_back(cand[k]);
        p2.push_back(p[k]);
    }
    cand.swap(c2);
    p.swap(p2);
}

// lulcc's draws: up to 1001 rounds of runif() over the ranked cells until at
// least n pass (`grow`: p >= u, else p < u); the cells of the last round
// otherwise. Without `stochastic` every ranked cell passes.
std::vector<i64> draw(const std::vector<double>& p, i64 n, bool stochastic, bool grow, nlm::RRng& rng) {
    std::vector<i64> sel;
    if (!stochastic) {
        sel.resize(p.size());
        std::iota(sel.begin(), sel.end(), i64{0});
        return sel;
    }
    std::vector<double> u(p.size());
    for (int counter = 1;; ++counter) {
        rng.runif(u.data(), static_cast<i64>(u.size()));
        sel.clear();
        for (std::size_t k = 0; k < p.size(); ++k)
            if (grow ? p[k] >= u[k] : p[k] < u[k]) sel.push_back(static_cast<i64>(k));
        if (static_cast<i64>(sel.size()) >= n || counter > 1000) return sel;
    }
}

}  // namespace

OrderedResult ordered_allocate(const double* tprob_in, const double* x, std::int64_t ncell, const double* demand,
                               const double* classes, int ncode, const double* order, bool stochastic,
                               nlm::RRng& rng) {
    std::vector<double> sorted_order(order, order + ncode), sorted_classes(classes, classes + ncode);
    std::sort(sorted_order.begin(), sorted_order.end());
    std::sort(sorted_classes.begin(), sorted_classes.end());
    if (sorted_order != sorted_classes) throw std::invalid_argument("'order' must contain the model's classes, each once");

    OrderedResult out;
    const std::size_t nc = static_cast<std::size_t>(ncode);
    double gap = 0.0;
    for (int j = 0; j < ncode; ++j) {
        i64 area = 0;
        for (i64 i = 0; i < ncell; ++i) area += x[i] == classes[j];
        gap += std::fabs(demand[j] - static_cast<double>(area));
    }
    out.allocation.assign(x, x + ncell);
    if (gap == 0) return out;

    std::vector<double> tprob(tprob_in, tprob_in + static_cast<std::size_t>(ncell) * nc);
    auto& x1 = out.allocation;
    auto forbid_rows = [&](const std::vector<double>& m, double cat) {
        for (i64 i = 0; i < ncell; ++i)
            if (m[static_cast<std::size_t>(i)] == cat)
                std::fill_n(tprob.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(i) * nc), ncode, kNaN);
    };
    const std::vector<double> x0(x, x + ncell);
    i64 short_cells = 0;
    for (int o = 0; o < ncode; ++o) {
        const double cat = order[o];
        int ix = 0;
        while (classes[ix] != cat) ++ix;
        i64 have = 0;
        for (i64 i = 0; i < ncell; ++i) have += x1[static_cast<std::size_t>(i)] == cat;
        const i64 n = static_cast<i64>(demand[ix] - static_cast<double>(have));  // int(): towards zero
        if (n == 0) {
            forbid_rows(x0, cat);
            continue;
        }
        const bool grow = n > 0;
        std::vector<i64> cand;
        std::vector<double> p;
        for (i64 i = 0; i < ncell; ++i) {
            const bool take = grow ? x1[static_cast<std::size_t>(i)] != cat : x0[static_cast<std::size_t>(i)] == cat;
            if (!take) continue;
            cand.push_back(i);
            p.push_back(tprob[static_cast<std::size_t>(i) * nc + static_cast<std::size_t>(ix)]);
        }
        rank(cand, p, grow);
        const i64 want = grow ? n : -n;
        auto sel = draw(p, want, stochastic, grow, rng);
        if (static_cast<i64>(sel.size()) > want) sel.resize(static_cast<std::size_t>(want));
        short_cells += want - static_cast<i64>(sel.size());
        for (const auto k : sel) x1[static_cast<std::size_t>(cand[static_cast<std::size_t>(k)])] = grow ? cat : -1.0;
        forbid_rows(x1, cat);
    }
    for (i64 i = 0; i < ncell; ++i)  // released by a shrinking class and taken by none served after it: they stay
        if (x1[static_cast<std::size_t>(i)] == -1.0) x1[static_cast<std::size_t>(i)] = x0[static_cast<std::size_t>(i)];
    out.iterations = 1;
    out.converged = short_cells == 0;
    out.short_cells = short_cells;
    return out;
}

}  // namespace sim
