// SPDX-License-Identifier: GPL-3.0-or-later
//
// CARS (PLUS): see plus.hpp.

#include "sim/plus.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "lsm/parallel.hpp"

namespace sim {

namespace {

constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ull;

inline std::uint64_t mix(std::uint64_t z) {  // SplitMix64's finaliser
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// The random numbers of a cell in an iteration: a counter-based stream.
class CellRandom {
public:
    CellRandom(std::uint64_t seed, std::int64_t iter, std::int64_t cell)
        : base_(mix(mix(seed + kGolden * static_cast<std::uint64_t>(iter + 1)) ^
                    (static_cast<std::uint64_t>(cell) * 0xD1B54A32D192ED03ull))) {}
    double uniform(std::uint64_t slot) const { return static_cast<double>(mix(base_ + kGolden * (slot + 1)) >> 11) * 0x1.0p-53; }
    std::uint64_t bits(std::uint64_t slot) const { return mix(base_ + kGolden * (slot + 1)); }

private:
    std::uint64_t base_;
};

constexpr double kTwoPi = 6.283185307179586;

class Cars {
public:
    Cars(std::int64_t ny, std::int64_t nx, int k, const int* land, const std::uint8_t* fixed, const int* region,
         int nreg, const double* prob, const double* weights, const double* seeds, const std::uint8_t* allowed,
         const std::int64_t* demand, const CarsSettings& s)
        : ny_(ny), nx_(nx), n_(ny * nx), k_(k), m_(static_cast<std::size_t>(nreg) * k), land_(land, land + ny * nx),
          region_(region ? std::vector<int>(region, region + ny * nx) : std::vector<int>(ny * nx, 0)), prob_(prob),
          w_(weights, weights + k), mu_(seeds, seeds + k),
          allowed_(allowed, allowed + static_cast<std::size_t>(k) * k), demand_(demand, demand + m_), s_(s),
          half_(s.window / 2), denom_(static_cast<double>(s.window) * s.window - 1) {
        count_.assign(m_, 0);
        for (std::int64_t i = 0; i < n_; ++i)
            if (land_[i] >= 0 && region_[i] >= 0) ++count_[slot(i, land_[i])];
        any_target_.assign(static_cast<std::size_t>(k), 0);
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b)
                if (a != b && allowed_[static_cast<std::size_t>(a) * k + b]) any_target_[a] = 1;
        for (std::int64_t i = 0; i < n_; ++i)
            if (land_[i] >= 0 && region_[i] >= 0 && !(fixed && fixed[i])) cells_.push_back(i);
        nb_.assign(static_cast<std::size_t>(n_) * k, 0);
        lsm::parallel_for(n_ * s.window * s.window >= (1 << 18), ny_, [&](std::int64_t r) {
            for (std::int64_t c = 0; c < nx_; ++c) {
                std::uint16_t* out = &nb_[static_cast<std::size_t>(r * nx_ + c) * k_];
                neighbours(r * nx_ + c, [&](std::int64_t j) {
                    if (land_[j] >= 0) ++out[land_[j]];
                });
            }
        });
    }

    CarsResult run() {
        CarsResult res;
        std::vector<double> d(m_, 1.0);
        std::vector<std::int64_t> g2(m_), g1(m_);
        for (std::size_t c = 0; c < m_; ++c) g1[c] = demand_[c] - count_[c];
        res.trace.insert(res.trace.end(), count_.begin(), count_.end());
        std::int64_t gap = total(g1);
        std::int64_t l = 0;
        std::vector<int> proposal(cells_.size());
        std::vector<std::pair<std::uint64_t, std::int64_t>> todo;  // (random priority, index into cells_)
        std::vector<std::int64_t> changed;
        std::vector<int> old_class;
        std::int64_t it = 0;
        while (gap > 0 && it < s_.max_iter) {
            const double level = std::pow(s_.decay, static_cast<double>(l));
            propose(it, d, level, proposal);
            todo.clear();
            for (std::size_t q = 0; q < cells_.size(); ++q)
                if (proposal[q] >= 0) todo.emplace_back(CellRandom(s_.seed, it, cells_[q]).bits(1u << 20), q);
            std::sort(todo.begin(), todo.end());
            changed.clear();
            old_class.clear();
            for (const auto& [key, q] : todo) {
                const int c = proposal[q];
                const std::int64_t i = cells_[q];
                const std::size_t to_slot = slot(i, c), from_slot = slot(i, land_[i]);
                if (count_[to_slot] >= demand_[to_slot]) continue;  // the class has its demand
                if (count_[from_slot] <= demand_[from_slot]) continue;  // nor does it lose cells below it
                const int from = land_[i];
                land_[i] = c;
                ++count_[to_slot];
                --count_[from_slot];
                changed.push_back(i);
                old_class.push_back(from);
            }
            for (std::size_t m = 0; m < changed.size(); ++m) {  // the neighbourhoods of the next iteration
                const int from = old_class[m], to = land_[changed[m]];
                neighbours(changed[m], [&](std::int64_t j) {
                    std::uint16_t* v = &nb_[static_cast<std::size_t>(j) * k_];
                    --v[from];
                    ++v[to];
                });
            }
            ++it;
            g2 = g1;
            for (std::size_t c = 0; c < m_; ++c) g1[c] = demand_[c] - count_[c];
            const std::int64_t gap_now = total(g1);
            if (static_cast<double>(gap - gap_now) < s_.step) ++l;  // Eq. 6
            gap = gap_now;
            for (std::size_t c = 0; c < m_; ++c) {  // Eq. 4
                const double a = static_cast<double>(g1[c]), b = static_cast<double>(g2[c]);
                if (std::abs(a) <= std::abs(b)) continue;
                if (0 > b && b > a) d[c] *= b / a;
                else if (a > b && b > 0) d[c] *= a / b;
            }
            res.trace.insert(res.trace.end(), count_.begin(), count_.end());
            res.trace_decays.push_back(l);
            res.trace_d.insert(res.trace_d.end(), d.begin(), d.end());
        }
        res.landscape = land_;
        res.iterations = it;
        res.converged = gap == 0;
        res.decays = l;
        return res;
    }

private:
    std::size_t slot(std::int64_t i, int c) const { return static_cast<std::size_t>(region_[i]) * k_ + c; }

    template <typename F>
    void neighbours(std::int64_t i, F&& f) const {
        const std::int64_t r = i / nx_, c = i % nx_;
        for (std::int64_t a = std::max<std::int64_t>(0, r - half_); a <= std::min(ny_ - 1, r + half_); ++a)
            for (std::int64_t b = std::max<std::int64_t>(0, c - half_); b <= std::min(nx_ - 1, c + half_); ++b)
                if (a != r || b != c) f(a * nx_ + b);
    }

    static std::int64_t total(const std::vector<std::int64_t>& g) {
        std::int64_t s = 0;
        for (auto v : g) s += v < 0 ? -v : v;
        return s;
    }

    // r of the threshold: normal of mean 1, standard deviation 1/3, drawn again outside [0, 2]
    static double threshold_r(const CellRandom& rnd, int k) {
        for (std::uint64_t j = 0;; ++j) {
            const double u1 = rnd.uniform(static_cast<std::uint64_t>(k) + 1 + 2 * j);
            const double u2 = rnd.uniform(static_cast<std::uint64_t>(k) + 2 + 2 * j);
            if (!(u1 > 0)) continue;
            const double z = std::sqrt(-2 * std::log(u1)) * std::cos(kTwoPi * u2);
            const double r = 1 + z / 3;
            if (r >= 0 && r <= 2) return r;
        }
    }

    void propose(std::int64_t it, const std::vector<double>& d, double level, std::vector<int>& proposal) const {
        const auto ncell = static_cast<std::int64_t>(cells_.size());
        lsm::parallel_for(ncell * k_ >= (1 << 16), ncell, [&](std::int64_t q) {
            proposal[q] = -1;
            const std::int64_t i = cells_[q];
            const int cur = land_[i];
            if (!any_target_[cur]) return;
            const CellRandom rnd(s_.seed, it, i);
            const std::uint16_t* nb = &nb_[static_cast<std::size_t>(i) * k_];
            const double* dr = &d[slot(i, 0)];
            double op[64];
            std::vector<double> big;
            double* o = op;
            if (k_ > 64) {
                big.resize(static_cast<std::size_t>(k_));
                o = big.data();
            }
            double sum = 0;
            for (int c = 0; c < k_; ++c) {
                const double p = prob_[static_cast<std::size_t>(c) * n_ + i];
                double v = 0;
                if (p > 0) {
                    const double omega = nb[c] / denom_ * w_[c];
                    if (omega > 0) {
                        v = p * omega * dr[c];
                    } else {
                        const double r = rnd.uniform(static_cast<std::uint64_t>(c) + 1);
                        if (r < p) v = p * (r * mu_[c]) * dr[c];
                    }
                }
                o[c] = v;
                sum += v;
            }
            if (!(sum > 0)) return;
            const double u = rnd.uniform(0) * sum;
            int pick = -1;
            double acc = 0;
            for (int c = 0; c < k_; ++c) {
                if (!(o[c] > 0)) continue;
                pick = c;
                acc += o[c];
                if (u < acc) break;
            }
            if (pick < 0 || pick == cur) return;
            if (!allowed_[static_cast<std::size_t>(cur) * k_ + pick]) return;
            const double tau = level * threshold_r(rnd, k_);
            if (prob_[static_cast<std::size_t>(pick) * n_ + i] > tau) proposal[q] = pick;
        });
    }

    std::int64_t ny_, nx_, n_;
    int k_;
    std::size_t m_;  // regions x classes
    std::vector<int> land_, region_;
    const double* prob_;
    std::vector<double> w_, mu_;
    std::vector<std::uint8_t> allowed_;
    std::vector<std::int64_t> demand_;
    CarsSettings s_;
    std::int64_t half_;
    double denom_;
    std::vector<std::int64_t> count_;
    std::vector<std::uint8_t> any_target_;
    std::vector<std::int64_t> cells_;
    std::vector<std::uint16_t> nb_;  // (cell, class): cells of the class in the window, the centre left out
};

}  // namespace

CarsResult cars_run(std::int64_t ny, std::int64_t nx, int k, const int* land, const std::uint8_t* fixed,
                    const int* region, int nreg, const double* prob, const double* weights, const double* seeds, const std::uint8_t* allowed,
                    const std::int64_t* demand, const CarsSettings& s) {
    if (k < 1) throw std::invalid_argument("CARS needs at least one class");
    if (s.window < 3 || s.window % 2 == 0 || s.window > 255)
        throw std::invalid_argument("the neighbourhood must be odd, from 3 to 255 cells");
    if (!(s.decay > 0 && s.decay <= 1)) throw std::invalid_argument("the decay must be in (0, 1]");
    for (int c = 0; c < k; ++c)
        if (!(seeds[c] >= 0 && seeds[c] <= 1)) throw std::invalid_argument("the patch generation thresholds must be in [0, 1]");
    if (nreg < 1) throw std::invalid_argument("CARS needs at least one region");
    for (std::int64_t i = 0; i < ny * nx; ++i) {
        if (land[i] >= k) throw std::invalid_argument("a cell has a class index out of range");
        if (region && region[i] >= nreg) throw std::invalid_argument("a cell has a region index out of range");
    }
    Cars c(ny, nx, k, land, fixed, region, nreg, prob, weights, seeds, allowed, demand, s);
    return c.run();
}

}  // namespace sim
