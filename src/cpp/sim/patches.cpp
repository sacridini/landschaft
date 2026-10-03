// SPDX-License-Identifier: GPL-3.0-or-later
//
// Expander and Patcher: see patches.hpp.

#include "sim/patches.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace sim {

namespace {

class Xoshiro {
public:
    explicit Xoshiro(std::uint64_t seed) {
        for (auto& v : s_) {
            seed += 0x9E3779B97F4A7C15ull;
            std::uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            v = z ^ (z >> 31);
        }
    }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
    double normal() {  // Marsaglia's polar method (one of the pair)
        double u, v, s;
        do {
            u = 2 * uniform() - 1;
            v = 2 * uniform() - 1;
            s = u * u + v * v;
        } while (s >= 1 || s == 0);
        return u * std::sqrt(-2 * std::log(s) / s);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t next() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }
    std::uint64_t s_[4];
};

// Weighted draws without replacement (a Fenwick tree of weights).
class Weighted {
public:
    explicit Weighted(const std::vector<double>& w) : n_(w.size()), tree_(w.size() + 1, 0.0), w_(w) {
        for (std::size_t i = 0; i < n_; ++i) {
            add(i, w[i]);
            if (w[i] > 0) ++left_;
        }
    }
    std::size_t left() const { return left_; }  // weights not yet drawn (the tree's sum keeps rounding residue)
    double total() const { return sum(n_); }
    void zero(std::size_t i) {
        if (w_[i] != 0) {
            add(i, -w_[i]);
            w_[i] = 0;
            --left_;
        }
    }
    std::size_t draw(double u) const {  // u in [0, total)
        std::size_t pos = 0;
        std::size_t step = 1;
        while (step * 2 <= n_) step *= 2;
        double rem = u;
        for (; step > 0; step /= 2) {
            if (pos + step <= n_ && tree_[pos + step] <= rem) {
                pos += step;
                rem -= tree_[pos];
            }
        }
        // pos is the count of elements whose prefix sum <= u: the next one holds u
        std::size_t i = std::min(pos, n_ - 1);
        while (w_[i] == 0 && i + 1 < n_) ++i;  // guard against rounding at zero weights
        while (w_[i] == 0 && i > 0) --i;
        return i;
    }

private:
    void add(std::size_t i, double v) {
        for (std::size_t k = i + 1; k <= n_; k += k & (~k + 1)) tree_[k] += v;
    }
    double sum(std::size_t k) const {
        double s = 0;
        for (; k > 0; k -= k & (~k + 1)) s += tree_[k];
        return s;
    }
    std::size_t n_;
    std::vector<double> tree_;
    std::vector<double> w_;
    std::size_t left_ = 0;
};

struct Grid {
    std::int64_t ny, nx;
    int wr, wc;
    template <typename F>
    void neighbours(std::int64_t c, F&& f) const {
        const std::int64_t r = c / nx, k = c % nx;
        for (int a = -wr / 2; a <= wr / 2; ++a) {
            const std::int64_t rr = r + a;
            if (rr < 0 || rr >= ny) continue;
            for (int b = -wc / 2; b <= wc / 2; ++b) {
                if (a == 0 && b == 0) continue;
                const std::int64_t kk = k + b;
                if (kk < 0 || kk >= nx) continue;
                f(rr * nx + kk);
            }
        }
    }
};

std::int64_t patch_size(Xoshiro& rng, double mean, double var) {
    if (!(mean > 0)) return 1;
    if (!(var > 0)) return std::max<std::int64_t>(1, std::llround(mean));
    const double s2 = std::log(1 + var / (mean * mean));
    const double mu = std::log(mean) - s2 / 2;
    return std::max<std::int64_t>(1, std::llround(std::exp(mu + std::sqrt(s2) * rng.normal())));
}

class Allocator {
public:
    Allocator(std::int64_t ny, std::int64_t nx, const int* land, std::vector<PatchTransition>& t, double prune,
              int wr, int wc, std::uint64_t seed)
        : g_{ny, nx, wr, wc}, land_(land, land + ny * nx), t_(t), prune_(prune), rng_(seed),
          half_((wr * wc + 1) / 2) {}

    PatchResult run() {
        const std::size_t nt = t_.size();
        PatchResult out;
        out.expanded.assign(nt, 0);
        out.patched.assign(nt, 0);
        out.patches.assign(nt, 0);
        std::vector<std::int64_t> left(nt, 0);
        for (std::size_t k = 0; k < nt; ++k) {
            const auto done = phase(k, true, t_[k].expand, out.patches[k]);
            out.expanded[k] = done;
            left[k] = t_[k].expand - done;
        }
        for (std::size_t k = 0; k < nt; ++k) out.patched[k] = phase(k, false, t_[k].patch + left[k], out.patches[k]);
        out.landscape = land_;
        return out;
    }

private:
    int count_to(std::int64_t c, int to) const {
        int n = 0;
        g_.neighbours(c, [&](std::int64_t nb) {
            if (land_[nb] == to) ++n;
        });
        return n;
    }

    // corrodes the cell: no transition may use it again
    void change(std::int64_t c, int to) {
        land_[c] = to;
        for (auto& tr : t_) tr.probability[c] = 0;
    }

    // The cells of positive (expander: modified) probability ranked from the highest, the first
    // prune_factor x n kept. The Expander draws them in proportion to P', each tried once and becoming
    // a seed with probability P' (what is left is returned, as Dinamica EGO's Expander does); the
    // Patcher draws seeds in proportion to P until its changes are made.
    std::int64_t phase(std::size_t k, bool expander, std::int64_t n, std::int64_t& patches) {
        if (expander) return walk(k, true, n, patches);
        std::int64_t done = 0;  // the Patcher ranks the cells again when its list runs out
        while (done < n) {
            const std::int64_t got = walk(k, false, n - done, patches);
            if (got == 0) break;
            done += got;
        }
        return done;
    }

    std::int64_t walk(std::size_t k, bool expander, std::int64_t n, std::int64_t& patches) {
        if (n <= 0) return 0;
        auto& tr = t_[k];
        const std::int64_t ncell = g_.ny * g_.nx;
        std::vector<std::int64_t> cand;
        std::vector<double> weight;
        for (std::int64_t c = 0; c < ncell; ++c) {
            if (land_[c] != tr.from) continue;
            const double p = tr.probability[c];
            if (!(p > 0)) continue;
            double w = p;
            if (expander) {
                const int nj = count_to(c, tr.to);
                w = nj >= half_ ? p : p * nj / static_cast<double>(half_);
            }
            if (w > 0) {
                cand.push_back(c);
                weight.push_back(w);
            }
        }
        std::vector<std::size_t> order(cand.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return weight[a] > weight[b]; });
        const std::size_t keep = std::min<std::size_t>(order.size(), static_cast<std::size_t>(
                                                                        std::ceil(prune_ * static_cast<double>(n))));
        order.resize(keep);
        const double mean = expander ? tr.e_mean : tr.p_mean;
        const double var = expander ? tr.e_var : tr.p_var;
        const double iso = expander ? tr.e_iso : tr.p_iso;
        std::int64_t done = 0;
        if (expander) {  // drawn in proportion to P', each tried once: a seed with probability P'
            std::vector<double> we(order.size());
            for (std::size_t i = 0; i < order.size(); ++i) we[i] = weight[order[i]];
            Weighted draw(we);
            while (done < n) {
                if (draw.left() == 0) break;
                const double total = draw.total();
                if (!(total > 0)) break;
                const std::size_t i = draw.draw(rng_.uniform() * total);
                draw.zero(i);
                const std::int64_t seed = cand[order[i]];
                if (land_[seed] != tr.from || !(tr.probability[seed] > 0)) continue;
                if (!(rng_.uniform() < weight[order[i]])) continue;
                const std::int64_t size = std::min(patch_size(rng_, mean, var), n - done);
                done += grow(seed, tr, size, iso);
                ++patches;
            }
            return done;
        }
        std::vector<double> w(order.size());
        for (std::size_t i = 0; i < order.size(); ++i) w[i] = weight[order[i]];
        Weighted draw(w);  // seeds drawn in proportion to P, until done
        while (done < n) {
            if (draw.left() == 0) break;
            const double total = draw.total();
            if (!(total > 0)) break;
            const std::size_t i = draw.draw(rng_.uniform() * total);
            draw.zero(i);
            const std::int64_t seed = cand[order[i]];
            if (land_[seed] != tr.from || !(tr.probability[seed] > 0)) continue;
            const std::int64_t size = std::min(patch_size(rng_, mean, var), n - done);
            done += grow(seed, tr, size, iso);
            ++patches;
        }
        return done;
    }

    std::int64_t grow(std::int64_t seed, PatchTransition& tr, std::int64_t size, double iso) {
        change(seed, tr.to);
        std::int64_t grown = 1;
        std::unordered_map<std::int64_t, int> front;  // candidate -> patch cells in its window
        std::vector<std::int64_t> order;              // first-touched order (deterministic iteration)
        auto touch = [&](std::int64_t c) {
            g_.neighbours(c, [&](std::int64_t nb) {
                if (land_[nb] != tr.from || !(tr.probability[nb] > 0)) return;
                auto it = front.find(nb);
                if (it == front.end()) {
                    front.emplace(nb, 1);
                    order.push_back(nb);
                } else {
                    ++it->second;
                }
            });
        };
        touch(seed);
        std::vector<double> wts;
        while (grown < size) {
            wts.assign(order.size(), 0.0);
            double total = 0;
            for (std::size_t i = 0; i < order.size(); ++i) {
                const std::int64_t c = order[i];
                if (land_[c] != tr.from || !(tr.probability[c] > 0)) continue;
                const double wv = tr.probability[c] * std::pow(iso, front[c]);
                wts[i] = wv;
                total += wv;
            }
            if (!(total > 0)) break;
            double u = rng_.uniform() * total;
            std::size_t pick = order.size();
            for (std::size_t i = 0; i < order.size(); ++i) {
                if (wts[i] <= 0) continue;
                pick = i;
                if (u < wts[i]) break;
                u -= wts[i];
            }
            const std::int64_t c = order[pick];
            change(c, tr.to);
            ++grown;
            front.erase(c);
            order.erase(order.begin() + static_cast<std::ptrdiff_t>(pick));
            touch(c);
        }
        return grown;
    }

    Grid g_;
    std::vector<int> land_;
    std::vector<PatchTransition>& t_;
    double prune_;
    Xoshiro rng_;
    int half_;  // n_j from which the Expander keeps the full probability: half the window, centre included
};

}  // namespace

PatchResult allocate_patches(std::int64_t ny, std::int64_t nx, const int* landscape,
                             std::vector<PatchTransition>& transitions, double prune_factor, int win_rows,
                             int win_cols, std::uint64_t seed) {
    if (win_rows < 1 || win_cols < 1 || win_rows % 2 == 0 || win_cols % 2 == 0)
        throw std::invalid_argument("the neighbour window needs odd sides");
    if (!(prune_factor >= 1)) throw std::invalid_argument("the prune factor must be at least 1");
    for (const auto& t : transitions)
        if (static_cast<std::int64_t>(t.probability.size()) != ny * nx)
            throw std::invalid_argument("a probability map differs in size from the landscape");
    Allocator a(ny, nx, landscape, transitions, prune_factor, win_rows, win_cols, seed);
    return a.run();
}

}  // namespace sim
