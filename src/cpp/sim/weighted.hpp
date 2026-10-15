// SPDX-License-Identifier: GPL-3.0-or-later
//
// Weighted draws without replacement over a fixed list (the Expander, Patcher
// and continuous Patcher's seeds).
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace sim {

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

}  // namespace sim
