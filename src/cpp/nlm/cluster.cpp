// SPDX-License-Identifier: GPL-3.0-or-later
//
// NLMR's nlm_randomcluster() and nlm_curds(), written in R there (terra,
// dplyr): the same draws of R's generator in the same order.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "lsm/parallel.hpp"
#include "nlm/models.hpp"

namespace nlm {

namespace {

// terra::patches(directions, allowGaps = FALSE) of the cells with
// `on[i] != 0`: labels 1.. in the order of the patches' first cells in a
// row-major scan (terra keeps the smallest label of merged patches and then
// renumbers them in order), 0 elsewhere. Returns the number of patches.
std::int64_t label_patches(const std::vector<std::uint8_t>& on, i64 nrow, i64 ncol, int directions,
                           std::vector<std::int64_t>& label) {
    const std::size_t n = static_cast<std::size_t>(nrow * ncol);
    label.assign(n, 0);
    std::vector<std::int64_t> parent(1, 0);
    auto find = [&](std::int64_t a) {
        while (parent[static_cast<std::size_t>(a)] != a) {
            parent[static_cast<std::size_t>(a)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(a)])];
            a = parent[static_cast<std::size_t>(a)];
        }
        return a;
    };
    auto unite = [&](std::int64_t a, std::int64_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (a < b)
            parent[static_cast<std::size_t>(b)] = a;
        else
            parent[static_cast<std::size_t>(a)] = b;
    };
    for (i64 r = 0; r < nrow; ++r) {
        for (i64 c = 0; c < ncol; ++c) {
            const std::size_t i = static_cast<std::size_t>(r * ncol + c);
            if (!on[i]) continue;
            std::int64_t chosen = 0;
            auto consider = [&](i64 rr, i64 cc) {
                if (rr < 0 || cc < 0 || cc >= ncol) return;
                const std::int64_t l = label[static_cast<std::size_t>(rr * ncol + cc)];
                if (l == 0) return;
                if (chosen == 0)
                    chosen = l;
                else
                    unite(chosen, l);
            };
            consider(r, c - 1);
            consider(r - 1, c);
            if (directions == 8) {
                consider(r - 1, c - 1);
                consider(r - 1, c + 1);
            }
            if (chosen == 0) {
                chosen = static_cast<std::int64_t>(parent.size());
                parent.push_back(chosen);
            }
            label[i] = chosen;
        }
    }
    // roots in increasing order = patches in order of their first cell
    std::vector<std::int64_t> rank(parent.size(), 0);
    std::int64_t count = 0;
    for (std::size_t l = 1; l < parent.size(); ++l)
        if (find(static_cast<std::int64_t>(l)) == static_cast<std::int64_t>(l)) rank[l] = ++count;
    for (std::size_t i = 0; i < n; ++i)
        if (label[i]) label[i] = rank[static_cast<std::size_t>(find(label[i]))];
    return count;
}

}  // namespace

std::vector<double> random_cluster(RRng& rng, i64 nrow, i64 ncol, double p, const std::vector<double>& ai,
                                   int neighbourhood) {
    const std::size_t n = static_cast<std::size_t>(nrow * ncol);
    // A: percolation map, runif() by column
    std::vector<std::uint8_t> on(n);
    for (i64 c = 0; c < ncol; ++c)
        for (i64 r = 0; r < nrow; ++r) on[static_cast<std::size_t>(r * ncol + c)] = rng.runif(0.0, 1.0) < p ? 1 : 0;
    // B: clusters
    std::vector<std::int64_t> label;
    const std::int64_t numclu = label_patches(on, nrow, ncol, neighbourhood, label);
    // C: a type per cluster, sample(seq_along(ai), numclu, replace = TRUE, prob = ai)
    const i64 k = static_cast<i64>(ai.size());
    std::vector<i64> clutyp;
    if (numclu > 0) clutyp = rng.sample_int(k, numclu, true, &ai);
    std::vector<double> map(n, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < n; ++i)
        if (label[i]) map[i] = static_cast<double>(clutyp[static_cast<std::size_t>(label[i] - 1)]);
    // D: every empty cell takes the most frequent type among its 8
    // neighbours in the map of step C (ties: sample(), none: sample(prob =
    // ai)), in the order of the cells. Neighbours are counted in parallel;
    // the draws follow in order.
    std::vector<i64> gaps;
    for (std::size_t i = 0; i < n; ++i)
        if (std::isnan(map[i])) gaps.push_back(static_cast<i64>(i));
    struct Choice {
        std::uint8_t n = 0;   // candidates (0: no neighbour)
        double cand[8] = {};  // tied types, ascending
    };
    std::vector<Choice> choice(gaps.size());
    lsm::parallel_for(gaps.size() >= 4096, static_cast<i64>(gaps.size()), [&](i64 g) {
        const i64 i = gaps[static_cast<std::size_t>(g)];
        const i64 r = i / ncol, c = i % ncol;
        double vals[8];
        int nv = 0;
        for (i64 dr = -1; dr <= 1; ++dr)
            for (i64 dc = -1; dc <= 1; ++dc) {
                if (!dr && !dc) continue;
                const i64 rr = r + dr, cc = c + dc;
                if (rr < 0 || rr >= nrow || cc < 0 || cc >= ncol) continue;
                const double v = map[static_cast<std::size_t>(rr * ncol + cc)];
                if (!std::isnan(v)) vals[nv++] = v;
            }
        Choice& ch = choice[static_cast<std::size_t>(g)];
        if (nv == 0) return;
        std::sort(vals, vals + nv);
        int best = 0;
        for (int a = 0; a < nv;) {
            int b = a;
            while (b < nv && vals[b] == vals[a]) ++b;
            if (b - a > best) {
                best = b - a;
                ch.n = 0;
            }
            if (b - a == best) ch.cand[ch.n++] = vals[a];
            a = b;
        }
    });
    std::vector<double> filled(map);
    for (std::size_t g = 0; g < gaps.size(); ++g) {
        const Choice& ch = choice[g];
        double v;
        if (ch.n == 0)
            v = static_cast<double>(rng.sample_int(k, 1, false, &ai)[0]);
        else if (ch.n == 1)
            v = ch.cand[0];
        else
            v = ch.cand[rng.sample_int(ch.n, 1, false)[0] - 1];
        filled[static_cast<std::size_t>(gaps[g])] = v;
    }
    return filled;
}

std::vector<double> curds(RRng& rng, const std::vector<double>& curds_p, const std::vector<std::int64_t>& steps,
                          const std::vector<double>* wheyes, i64& size) {
    // A 1 x 1 FALSE raster; per step: disaggregate, set a share 1 - curds[i]
    // of the FALSE cells to TRUE (dplyr::sample_frac: sample.int(n,
    // round(n * size))), then a share wheyes[i] of the TRUE cells back to
    // FALSE; finally invert.
    std::vector<std::uint8_t> v(1, 0);
    size = 1;
    std::vector<i64> ids;
    for (std::size_t s = 0; s < steps.size(); ++s) {
        const i64 f = steps[s];
        if (f < 1) throw std::invalid_argument("recursion steps must be positive");
        const i64 nsize = size * f;
        std::vector<std::uint8_t> w(static_cast<std::size_t>(nsize * nsize));
        lsm::parallel_for(nsize * nsize >= 65536, nsize, [&](i64 r) {
            for (i64 c = 0; c < nsize; ++c)
                w[static_cast<std::size_t>(r * nsize + c)] = v[static_cast<std::size_t>((r / f) * size + c / f)];
        });
        v.swap(w);
        size = nsize;
        auto flip = [&](std::uint8_t from, double share) {
            ids.clear();
            for (std::size_t i = 0; i < v.size(); ++i)
                if (v[i] == from) ids.push_back(static_cast<i64>(i));
            const i64 m = static_cast<i64>(ids.size());
            const double k = std::nearbyint(static_cast<double>(m) * share);
            if (share > 1) throw std::invalid_argument("`size` of sampled fraction must be less or equal to one.");
            const std::vector<i64> pick = rng.sample_int(m, static_cast<i64>(k), false);
            for (i64 j : pick) v[static_cast<std::size_t>(ids[static_cast<std::size_t>(j - 1)])] = from ? 0 : 1;
        };
        flip(0, 1 - curds_p[s]);
        if (wheyes) flip(1, (*wheyes)[s]);
    }
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = v[i] ? 0.0 : 1.0;
    return out;
}

}  // namespace nlm
