// SPDX-License-Identifier: GPL-3.0-or-later
//
// The Patcher of a continuous state: see continuous.hpp.

#include "sim/continuous.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "sim/rng.hpp"
#include "sim/weighted.hpp"

namespace sim {

namespace {

// lognormal of the given mean and variance, rounded (at least 1); the mean when the variance is 0
std::int64_t patch_size(Xoshiro& rng, double mean, double var) {
    if (!(mean > 0)) return 1;
    if (!(var > 0)) return std::max<std::int64_t>(1, std::llround(mean));
    const double s2 = std::log(1 + var / (mean * mean));
    const double mu = std::log(mean) - s2 / 2;
    return std::max<std::int64_t>(1, std::llround(std::exp(mu + std::sqrt(s2) * rng.normal())));
}

class Continuous {
public:
    Continuous(std::int64_t ny, std::int64_t nx, const double* stock, const double* prob, int wr, int wc,
               double iso, std::uint64_t seed)
        : ny_(ny), nx_(nx), stock_(stock), prob_(prob), wr_(wr), wc_(wc), taken_(ny * nx, 0.0),
          patch_(ny * nx, 0), rng_(seed) {
        pow_.resize(static_cast<std::size_t>(wr) * wc + 1);
        for (std::size_t k = 0; k < pow_.size(); ++k) pow_[k] = std::pow(iso, static_cast<double>(k));
    }

    bool eligible(std::int64_t c) const {
        return patch_[c] == 0 && stock_[c] > 0 && prob_[c] > 0;  // NaN fails both comparisons
    }

    template <typename F>
    void neighbours(std::int64_t c, F&& f) const {
        const std::int64_t r = c / nx_, k = c % nx_;
        for (int a = -wr_ / 2; a <= wr_ / 2; ++a) {
            const std::int64_t rr = r + a;
            if (rr < 0 || rr >= ny_) continue;
            for (int b = -wc_ / 2; b <= wc_ / 2; ++b) {
                if (a == 0 && b == 0) continue;
                const std::int64_t kk = k + b;
                if (kk < 0 || kk >= nx_) continue;
                f(rr * nx_ + kk);
            }
        }
    }

    ContinuousResult run(double quota, double mean, double var, double prune) {
        ContinuousResult out;
        const std::int64_t n = ny_ * nx_;
        std::vector<std::int64_t> cand;
        for (std::int64_t c = 0; c < n; ++c)
            if (eligible(c)) cand.push_back(c);
        // ranked from the highest probability (ties: the first cell first); only the head is sorted, in
        // blocks that double until they hold the first cells whose summed stock is at most prune x quota
        // (at least one)
        auto better = [&](std::int64_t a, std::int64_t b) {
            return prob_[a] > prob_[b] || (prob_[a] == prob_[b] && a < b);
        };
        const double cap = prune * quota;
        double mean_stock = 0;
        for (const auto c : cand) mean_stock += stock_[c];
        mean_stock = cand.empty() ? 1 : mean_stock / static_cast<double>(cand.size());
        std::size_t step = static_cast<std::size_t>(std::min(1e15, std::max(1024.0, 1.25 * cap / mean_stock)));
        double sum = 0;
        std::size_t keep = 0, lo = 0;
        bool full = false;
        while (!full && lo < cand.size()) {
            const std::size_t hi = std::min(cand.size(), lo + step);
            if (hi < cand.size()) std::nth_element(cand.begin() + lo, cand.begin() + hi, cand.end(), better);
            std::sort(cand.begin() + lo, cand.begin() + hi, better);
            for (keep = lo; keep < hi; ++keep) {
                sum += stock_[cand[keep]];
                if (sum > cap && keep > 0) {
                    full = true;
                    break;
                }
            }
            lo = hi;
            step *= 2;
        }
        cand.resize(keep);
        std::vector<double> w(cand.size());
        for (std::size_t i = 0; i < cand.size(); ++i) w[i] = prob_[cand[i]];
        Weighted draw(w);
        double left = quota;
        while (left > 0 && draw.left() > 0) {
            const double total = draw.total();
            if (!(total > 0)) break;
            const std::size_t i = draw.draw(rng_.uniform() * total);
            draw.zero(i);
            const std::int64_t s = cand[i];
            if (!eligible(s)) continue;  // taken by an earlier patch
            ++out.patches;
            left = grow(s, patch_size(rng_, mean, var), left, static_cast<std::int32_t>(out.patches), out.cells);
        }
        out.remaining = left;
        out.taken = std::move(taken_);
        out.patch = std::move(patch_);
        return out;
    }

private:
    void take(std::int64_t c, std::int32_t id, double& left, std::int64_t& cells) {
        taken_[c] = stock_[c];
        patch_[c] = id;
        left -= stock_[c];
        ++cells;
    }

    // grows a patch from its seed up to `size` cells or until the quota is met; the quota left
    double grow(std::int64_t seed, std::int64_t size, double left, std::int32_t id, std::int64_t& cells) {
        take(seed, id, left, cells);
        std::int64_t grown = 1;
        std::unordered_map<std::int64_t, std::size_t> at;  // front cell -> its index
        std::vector<std::int64_t> front;                    // first-touched order
        std::vector<int> count;                             // patch cells in each front cell's window
        std::vector<double> weight;
        double total = 0;
        auto touch = [&](std::int64_t c) {
            neighbours(c, [&](std::int64_t nb) {
                if (!eligible(nb)) return;
                auto it = at.find(nb);
                if (it == at.end()) {
                    at.emplace(nb, front.size());
                    front.push_back(nb);
                    count.push_back(1);
                    weight.push_back(prob_[nb] * pow_[1]);
                } else {
                    const std::size_t i = it->second;
                    ++count[i];
                    weight[i] = prob_[nb] * pow_[count[i]];
                }
            });
        };
        touch(seed);
        while (grown < size && left > 0) {
            total = 0;
            for (std::size_t i = 0; i < front.size(); ++i) total += weight[i];
            if (!(total > 0)) break;
            double u = rng_.uniform() * total;
            std::size_t pick = front.size();
            for (std::size_t i = 0; i < front.size(); ++i) {
                if (weight[i] <= 0) continue;
                pick = i;
                if (u < weight[i]) break;
                u -= weight[i];
            }
            const std::int64_t c = front[pick];
            take(c, id, left, cells);
            ++grown;
            // remove from the front keeping the first-touched order
            at.erase(c);
            front.erase(front.begin() + static_cast<std::ptrdiff_t>(pick));
            count.erase(count.begin() + static_cast<std::ptrdiff_t>(pick));
            weight.erase(weight.begin() + static_cast<std::ptrdiff_t>(pick));
            for (std::size_t i = pick; i < front.size(); ++i) at[front[i]] = i;
            touch(c);
        }
        return left;
    }

    std::int64_t ny_, nx_;
    const double* stock_;
    const double* prob_;
    int wr_, wc_;
    std::vector<double> taken_;
    std::vector<std::int32_t> patch_;
    std::vector<double> pow_;
    Xoshiro rng_;
};

}  // namespace

ContinuousResult continuous_patches(std::int64_t ny, std::int64_t nx, const double* stock, const double* probability,
                                    double quota, double mean, double variance, double isometry,
                                    double prune_factor, int win_rows, int win_cols, std::uint64_t seed) {
    if (win_rows < 1 || win_cols < 1 || win_rows % 2 == 0 || win_cols % 2 == 0)
        throw std::invalid_argument("the neighbour window needs odd sides");
    if (!(prune_factor > 0)) throw std::invalid_argument("the prune factor must be positive");
    if (!(isometry >= 0) || !std::isfinite(isometry)) throw std::invalid_argument("the isometry must be >= 0");
    if (!(mean > 0) || !std::isfinite(mean)) throw std::invalid_argument("the mean patch size must be positive");
    if (!(variance >= 0) || !std::isfinite(variance)) throw std::invalid_argument("the variance must be >= 0");
    if (!std::isfinite(quota)) throw std::invalid_argument("the quota must be finite");
    Continuous a(ny, nx, stock, probability, win_rows, win_cols, isometry, seed);
    if (!(quota > 0)) {
        ContinuousResult out;
        out.taken.assign(ny * nx, 0.0);
        out.patch.assign(ny * nx, 0);
        out.remaining = quota;
        return out;
    }
    return a.run(quota, mean, variance, prune_factor);
}

}  // namespace sim
