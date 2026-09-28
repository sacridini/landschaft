// SPDX-License-Identifier: GPL-3.0-or-later
//
// Basedata randomisations (see randomise.hpp), following Biodiverse::Randomise.

#include "bio/randomise.hpp"

#include "bio/perlmath.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace bio {

namespace {

// List::Unique::DeterministicOrder: keys with positions; deleting moves the
// last key into the freed position.
class DetOrder {
public:
    DetOrder() = default;
    explicit DetOrder(const std::vector<i32>& data) {
        for (i32 k : data) push(k);
    }
    std::size_t size() const { return arr_.size(); }
    bool empty() const { return arr_.empty(); }
    bool exists(i32 k) const { return pos_.count(k) != 0; }
    void push(i32 k) {
        if (exists(k)) return;
        arr_.push_back(k);
        pos_[k] = arr_.size() - 1;
    }
    i32 at(std::size_t p) const { return arr_[p]; }
    std::size_t pos(i32 k) const { return pos_.at(k); }
    void erase(i32 k) {
        auto it = pos_.find(k);
        if (it == pos_.end()) return;
        const std::size_t p = it->second;
        pos_.erase(it);
        const i32 last = arr_.back();
        arr_.pop_back();
        if (p == arr_.size()) return;
        pos_[last] = p;
        arr_[p] = last;
    }
    i32 erase_at(std::size_t p) {
        const i32 k = arr_[p];
        const i32 move = arr_.back();
        arr_.pop_back();
        pos_.erase(k);
        if (p == arr_.size()) return k;
        pos_[move] = p;
        arr_[p] = move;
        return k;
    }

private:
    std::vector<i32> arr_;
    std::unordered_map<i32, std::size_t> pos_;
};

// Positions 0..n-1 that are still available, with the j-th available one found in O(log n).
class Fenwick {
public:
    void reset(const std::vector<std::uint8_t>& on) {
        n_ = on.size();
        t_.assign(n_ + 1, 0);
        count_ = 0;
        for (std::size_t i = 0; i < n_; ++i) {
            t_[i + 1] += on[i];
            count_ += on[i];
            const std::size_t j = (i + 1) + ((i + 1) & (~(i + 1) + 1));
            if (j <= n_) t_[j] += t_[i + 1];
        }
        step_ = 1;
        while (step_ * 2 <= n_) step_ *= 2;
    }
    i64 count() const { return count_; }
    // the (k+1)-th available position (k from 0)
    std::size_t select(i64 k) const {
        std::size_t pos = 0;
        for (std::size_t s = step_; s; s >>= 1)
            if (pos + s <= n_ && t_[pos + s] <= k) {
                pos += s;
                k -= t_[pos];
            }
        return pos;
    }
    void remove(std::size_t i) {
        --count_;
        for (std::size_t j = i + 1; j <= n_; j += j & (~j + 1)) --t_[j];
    }

private:
    std::size_t n_ = 0, step_ = 1;
    std::vector<i64> t_;
    i64 count_ = 0;
};

void bremove(std::vector<i32>& v, i32 x) {
    auto it = std::lower_bound(v.begin(), v.end(), x);
    if (it != v.end() && *it == x) v.erase(it);
}

void binsert(std::vector<i32>& v, i32 x) {
    auto it = std::lower_bound(v.begin(), v.end(), x);
    v.insert(it, x);
}

// Statistics::Sampler::Multinomial::Indexed: a sum tree over integer weights
class IndexedSampler {
public:
    explicit IndexedSampler(std::vector<double> data) : data_(std::move(data)) {
        const auto n = static_cast<double>(data_.size());
        const int max_depth = 1 + static_cast<int>(perl_log(n) / perl_log(2.0));
        index_.assign(static_cast<std::size_t>(max_depth) + 1, {});
        index_[static_cast<std::size_t>(max_depth)] = data_;
        for (int level = max_depth - 1; level >= 0; --level) {
            const auto& below = index_[static_cast<std::size_t>(level) + 1];
            auto& cur = index_[static_cast<std::size_t>(level)];
            cur.assign((below.size() + 1) / 2, 0.0);
            for (std::size_t i = 0; i < below.size(); ++i) cur[i / 2] += below[i];
        }
    }
    std::size_t draw(MT64& rng) const {
        if (data_.size() == 1) return 0;
        const double norm = index_[0][0];
        double r = rng.rand() * norm;
        std::size_t level = 1, left = 0, right = 1;
        const std::size_t last = index_.size() - 1;
        while (level < last) {
            const auto& lv = index_[level];
            const double v = left < lv.size() ? lv[left] : 0.0;
            if (r <= v) {
                left *= 2;
                right = left + 1;
            } else {
                r -= v;
                left = right * 2;
                right = left + 1;
            }
            ++level;
        }
        const double d = left < data_.size() ? data_[left] : 0.0;
        return r > d ? right : left;
    }
    void update(std::size_t iter, double value) {
        const double diff = value - data_[iter];
        data_[iter] = value;
        index_.back()[iter] = value;
        std::size_t idx = iter >> 1;
        for (std::size_t level = index_.size() - 1; level-- > 0;) {
            index_[level][idx] += diff;
            idx >>= 1;
        }
    }

private:
    std::vector<double> data_;
    std::vector<std::vector<double>> index_;
};

CountsCsr from_label_maps(i64 ng, i64 nl, const std::vector<std::map<i32, double>>& by_label) {
    std::vector<std::map<i32, double>> by_group(static_cast<std::size_t>(ng));
    for (i64 l = 0; l < nl; ++l)
        for (const auto& kv : by_label[static_cast<std::size_t>(l)])
            by_group[static_cast<std::size_t>(kv.first)][static_cast<i32>(l)] = kv.second;
    CountsCsr out;
    out.n_groups = ng;
    out.n_labels = nl;
    out.indptr.assign(static_cast<std::size_t>(ng) + 1, 0);
    for (i64 g = 0; g < ng; ++g) {
        for (const auto& kv : by_group[static_cast<std::size_t>(g)]) {
            out.indices.push_back(kv.first);
            out.counts.push_back(kv.second);
        }
        out.indptr[static_cast<std::size_t>(g) + 1] = static_cast<i64>(out.indices.size());
    }
    return out;
}

// int $rand->rand (n)
inline std::size_t rand_index(MT64& rng, std::size_t n) {
    return static_cast<std::size_t>(std::trunc(rng.rand() * static_cast<double>(n)));
}

}  // namespace

CountsCsr rand_independent_swaps(const CountsCsr& bd, MT64& rng, const std::vector<double>& label_range,
                                 const SwapOptions& opt, bool modified) {
    const i64 ng = bd.n_groups, nl = bd.n_labels;
    std::vector<std::map<i32, double>> gp_hash(static_cast<std::size_t>(nl));  // label -> group -> count
    std::vector<i32> richness(static_cast<std::size_t>(ng), 0), variety(static_cast<std::size_t>(nl), 0);
    for (i64 g = 0; g < ng; ++g)
        for (i64 k = bd.indptr[g]; k < bd.indptr[g + 1]; ++k) {
            const i32 l = bd.indices[static_cast<std::size_t>(k)];
            gp_hash[static_cast<std::size_t>(l)][static_cast<i32>(g)] = bd.counts[static_cast<std::size_t>(k)];
            ++richness[static_cast<std::size_t>(g)];
            ++variety[static_cast<std::size_t>(l)];
        }
    std::vector<i32> groups, labels;  // not empty, name order
    for (i64 g = 0; g < ng; ++g)
        if (richness[static_cast<std::size_t>(g)]) groups.push_back(static_cast<i32>(g));
    for (i64 l = 0; l < nl; ++l)
        if (variety[static_cast<std::size_t>(l)]) labels.push_back(static_cast<i32>(l));
    std::vector<std::uint8_t> max_range(static_cast<std::size_t>(nl), 0);
    std::int64_t nonzero = 0;
    for (i32 l : labels) {
        nonzero += variety[static_cast<std::size_t>(l)];
        if (label_range[static_cast<std::size_t>(l)] == static_cast<double>(groups.size()))
            max_range[static_cast<std::size_t>(l)] = 1;
    }
    const std::int64_t target = opt.swap_count > 0 ? opt.swap_count : 2 * nonzero;
    const std::int64_t max_attempts = opt.max_swap_attempts > 0 ? opt.max_swap_attempts : 100 * target;
    std::int64_t swaps = 0, attempts = 0, moved = 0;
    std::set<std::pair<i32, i32>> moved_pairs;  // (label, group)
    const auto orig = gp_hash;
    auto track = [&](i32 l, i32 g) {
        if (!opt.stop_on_all_swapped) return;
        if (orig[static_cast<std::size_t>(l)].count(g) && moved_pairs.insert({l, g}).second) ++moved;
    };
    auto truthy = [&](i32 l, i32 g) {
        const auto& m = gp_hash[static_cast<std::size_t>(l)];
        auto it = m.find(g);
        return it != m.end() && it->second != 0;
    };
    // $gp_hash{l1}{g2} = delete $gp_hash{l2}{g2}; $gp_hash{l2}{g1} = delete $gp_hash{l1}{g1}
    auto swap_cells = [&](i32 l1, i32 g1, i32 l2, i32 g2) {
        auto& h1 = gp_hash[static_cast<std::size_t>(l1)];
        auto& h2 = gp_hash[static_cast<std::size_t>(l2)];
        auto i2 = h2.find(g2);
        const double c22 = i2 != h2.end() ? i2->second : 0.0;
        if (i2 != h2.end()) h2.erase(i2);
        h1[g2] = c22;
        auto i1 = h1.find(g1);
        const double c11 = i1 != h1.end() ? i1->second : 0.0;
        if (i1 != h1.end()) h1.erase(i1);
        h2[g1] = c11;
    };

    if (!modified) {
        while (swaps < target && attempts < max_attempts && moved < nonzero) {
            ++attempts;
            i32 l1 = labels[rand_index(rng, labels.size())];
            if (max_range[static_cast<std::size_t>(l1)]) continue;
            i32 l2 = labels[rand_index(rng, labels.size())];
            bool stop = false;
            while (l1 == l2) {
                if (groups.size() == 1) {
                    stop = true;
                    break;
                }
                l2 = labels[rand_index(rng, labels.size())];
            }
            if (stop) break;
            if (max_range[static_cast<std::size_t>(l2)]) continue;
            const i32 g1 = groups[rand_index(rng, groups.size())];
            i32 g2 = groups[rand_index(rng, groups.size())];
            while (g1 == g2) g2 = groups[rand_index(rng, groups.size())];
            if (!(truthy(l1, g1) && truthy(l2, g2))) {
                std::swap(l1, l2);
                if (!(truthy(l1, g1) && truthy(l2, g2))) continue;
            }
            if (truthy(l1, g2) || truthy(l2, g1)) continue;
            track(l1, g1);
            track(l2, g2);
            swap_cells(l1, g1, l2, g2);
            ++swaps;
        }
        return from_label_maps(ng, nl, gp_hash);
    }

    // modified: labels drawn by range, target groups by richness among the groups without the label
    std::vector<double> ranges;
    for (i32 l : labels) ranges.push_back(static_cast<double>(variety[static_cast<std::size_t>(l)]));
    IndexedSampler label_sampler(ranges);
    std::vector<DetOrder> gp_list(static_cast<std::size_t>(nl)), gp_shadow(static_cast<std::size_t>(nl));
    std::vector<DetOrder> lb_list(static_cast<std::size_t>(ng));
    std::vector<std::unique_ptr<IndexedSampler>> shadow_sampler(static_cast<std::size_t>(nl));
    for (i32 l : labels) {
        std::vector<i32> with, without;
        for (const auto& kv : gp_hash[static_cast<std::size_t>(l)]) with.push_back(kv.first);
        for (i32 g : groups)
            if (!gp_hash[static_cast<std::size_t>(l)].count(g)) without.push_back(g);
        gp_list[static_cast<std::size_t>(l)] = DetOrder(with);
        gp_shadow[static_cast<std::size_t>(l)] = DetOrder(without);
        if (!without.empty()) {
            std::vector<double> w;
            for (i32 g : without) w.push_back(static_cast<double>(richness[static_cast<std::size_t>(g)]));
            shadow_sampler[static_cast<std::size_t>(l)].reset(new IndexedSampler(w));
        }
    }
    for (i32 g : groups) {
        std::vector<i32> ls;
        for (i64 k = bd.indptr[g]; k < bd.indptr[g + 1]; ++k) ls.push_back(bd.indices[static_cast<std::size_t>(k)]);
        lb_list[static_cast<std::size_t>(g)] = DetOrder(ls);
    }
    while (swaps < target && attempts < max_attempts && moved < nonzero) {
        ++attempts;
        const i32 l1 = labels[label_sampler.draw(rng)];
        if (max_range[static_cast<std::size_t>(l1)]) continue;
        auto& gl1 = gp_list[static_cast<std::size_t>(l1)];
        const i32 g1 = gl1.at(rand_index(rng, gl1.size()));
        const std::size_t it = shadow_sampler[static_cast<std::size_t>(l1)]->draw(rng);
        const i32 g2 = gp_shadow[static_cast<std::size_t>(l1)].at(it);
        auto& lb2 = lb_list[static_cast<std::size_t>(g2)];
        const std::size_t key_count = lb2.size();
        i32 l2 = lb2.at(rand_index(rng, key_count));
        std::set<i32> checked;
        bool skip = false;
        while (lb_list[static_cast<std::size_t>(g1)].exists(l2) || max_range[static_cast<std::size_t>(l2)]) {
            checked.insert(l2);
            if (checked.size() >= key_count) {
                skip = true;
                break;
            }
            l2 = lb2.at(rand_index(rng, key_count));
        }
        if (skip) continue;
        // (Biodiverse never fills the original pairs here, so no pair counts as moved)
        swap_cells(l1, g1, l2, g2);
        auto& gl2 = gp_list[static_cast<std::size_t>(l2)];
        gl2.erase(g2);
        gl1.push(g2);
        gl1.erase(g1);
        gl2.push(g1);
        lb2.erase(l2);
        lb_list[static_cast<std::size_t>(g1)].push(l2);
        lb_list[static_cast<std::size_t>(g1)].erase(l1);
        lb2.push(l1);
        auto& s1 = gp_shadow[static_cast<std::size_t>(l1)];
        auto& s2 = gp_shadow[static_cast<std::size_t>(l2)];
        s1.push(g1);
        s2.push(g2);
        s1.erase(g2);
        s2.erase(g1);
        shadow_sampler[static_cast<std::size_t>(l1)]->update(s1.pos(g1), richness[static_cast<std::size_t>(g1)]);
        shadow_sampler[static_cast<std::size_t>(l2)]->update(s2.pos(g2), richness[static_cast<std::size_t>(g2)]);
        ++swaps;
    }
    return from_label_maps(ng, nl, gp_hash);
}

CountsCsr rand_curveball(const CountsCsr& bd, MT64& rng, const SwapOptions& opt) {
    const i64 ng = bd.n_groups, nl = bd.n_labels;
    std::vector<std::map<i32, double>> lb_hash(static_cast<std::size_t>(ng));  // group -> label -> count
    std::vector<i32> variety(static_cast<std::size_t>(nl), 0);
    for (i64 g = 0; g < ng; ++g)
        for (i64 k = bd.indptr[g]; k < bd.indptr[g + 1]; ++k) {
            const i32 l = bd.indices[static_cast<std::size_t>(k)];
            lb_hash[static_cast<std::size_t>(g)][l] = bd.counts[static_cast<std::size_t>(k)];
            ++variety[static_cast<std::size_t>(l)];
        }
    std::size_t n_labels = 0;
    for (i32 v : variety)
        if (v) ++n_labels;
    std::vector<i32> groups;
    std::int64_t nonzero = 0;
    for (i64 g = 0; g < ng; ++g) {
        const auto r = lb_hash[static_cast<std::size_t>(g)].size();
        if (!r) continue;
        if (r == n_labels) continue;  // full groups cannot swap
        nonzero += static_cast<std::int64_t>(r);
        groups.push_back(static_cast<i32>(g));
    }
    const auto orig = lb_hash;
    const std::int64_t target = opt.swap_count > 0 ? opt.swap_count : 2 * nonzero;
    const std::int64_t max_attempts = opt.max_swap_attempts > 0 ? opt.max_swap_attempts : 100 * target;
    std::int64_t swaps = 0, attempts = 0, moved = 0;
    std::set<std::pair<i32, i32>> moved_pairs;  // (group, label)
    const std::size_t n = groups.size();
    while (swaps < target && attempts < max_attempts && moved < nonzero && n > 1) {
        ++attempts;
        const i32 g1 = groups[static_cast<std::size_t>(rng.irand() % n)];
        i32 g2 = groups[static_cast<std::size_t>(rng.irand() % n)];
        while (g1 == g2) g2 = groups[static_cast<std::size_t>(rng.irand() % n)];
        auto& labels1 = lb_hash[static_cast<std::size_t>(g1)];
        auto& labels2 = lb_hash[static_cast<std::size_t>(g2)];
        std::vector<i32> from1, from2;
        for (const auto& kv : labels1)
            if (!labels2.count(kv.first)) from1.push_back(kv.first);
        for (const auto& kv : labels2)
            if (!labels1.count(kv.first)) from2.push_back(kv.first);
        const std::size_t n1 = std::min(from1.size(), from2.size()), n2 = std::max(from1.size(), from2.size());
        if (!n1) continue;
        std::vector<i32> swap1, swap2;
        const double ratio = static_cast<double>(n1) / static_cast<double>(n2);
        if (opt.use_max_swap) {
            if (from1.size() > n1) rng.shuffle(from1);
            from1.resize(n1);
            if (from2.size() > n1) rng.shuffle(from2);
            from2.resize(n1);
            swap1 = from1;
            swap2 = from2;
        } else if (n1 == 1 || ratio <= 0.05) {
            std::int64_t nswaps = 0;
            if (n1 == 1)
                nswaps = (rng.rand() > 1 / (1 + static_cast<double>(n2))) ? 1 : 0;
            else
                nswaps = static_cast<std::int64_t>(n1) - rng.binomial(ratio, static_cast<std::int64_t>(n1));
            if (!nswaps) continue;
            if (nswaps == 1) {
                swap1.push_back(from1[static_cast<std::size_t>(rng.irand() % from1.size())]);
                swap2.push_back(from2[static_cast<std::size_t>(rng.irand() % from2.size())]);
            } else {
                rng.shuffle(from1);
                from1.resize(static_cast<std::size_t>(nswaps));
                rng.shuffle(from2);
                from2.resize(static_cast<std::size_t>(nswaps));
                swap1 = from1;
                swap2 = from2;
            }
        } else {
            std::vector<i32> shuffled = from1;
            shuffled.insert(shuffled.end(), from2.begin(), from2.end());
            rng.shuffle(shuffled);
            std::size_t s_count = 0, i = 0;
            while (s_count != n1 && i < from1.size()) {
                if (labels2.count(shuffled[i])) {
                    swap2.push_back(shuffled[i]);
                    ++s_count;
                }
                ++i;
            }
            i = from1.size();
            while (s_count != 0 && i < shuffled.size()) {
                if (labels1.count(shuffled[i])) {
                    swap1.push_back(shuffled[i]);
                    --s_count;
                }
                ++i;
            }
            if (swap1.empty()) continue;
        }
        if (opt.stop_on_all_swapped) {
            for (std::size_t i = 0; i < swap1.size(); ++i) {
                if (orig[static_cast<std::size_t>(g1)].count(swap1[i]) && moved_pairs.insert({g1, swap1[i]}).second)
                    ++moved;
                if (orig[static_cast<std::size_t>(g2)].count(swap2[i]) && moved_pairs.insert({g2, swap2[i]}).second)
                    ++moved;
            }
        }
        for (i32 l : swap1) {
            labels2[l] = labels1.at(l);
            labels1.erase(l);
        }
        for (i32 l : swap2) {
            labels1[l] = labels2.at(l);
            labels2.erase(l);
        }
        swaps += static_cast<std::int64_t>(swap1.size());
    }
    CountsCsr out;
    out.n_groups = ng;
    out.n_labels = nl;
    out.indptr.assign(static_cast<std::size_t>(ng) + 1, 0);
    for (i64 g = 0; g < ng; ++g) {
        for (const auto& kv : lb_hash[static_cast<std::size_t>(g)]) {
            out.indices.push_back(kv.first);
            out.counts.push_back(kv.second);
        }
        out.indptr[static_cast<std::size_t>(g) + 1] = static_cast<i64>(out.indices.size());
    }
    return out;
}

SpatialAlloc::SpatialAlloc(std::string order_, std::string backtracking_, double reseed_prob_,
                           std::vector<std::vector<std::vector<i32>>> nbrs, bool same,
                           std::vector<std::vector<double>> coords, bool have_seeds_, std::vector<i32> seeds_)
    : order(std::move(order_)),
      backtracking(std::move(backtracking_)),
      reseed_prob(reseed_prob_),
      have_seeds(have_seeds_),
      seeds(std::move(seeds_)),
      nbrs_(std::move(nbrs)),
      same_(same),
      coords_(std::move(coords)) {
    std::sort(seeds.begin(), seeds.end());
    cache_.assign(nbrs_.size(), {});
    cached_.assign(nbrs_.size(), 0);
}

// get_sp_alloc_nbr_list
const std::vector<std::vector<i32>>& SpatialAlloc::lists(i32 group, MT64& rng) {
    const auto g = static_cast<std::size_t>(group);
    if (cached_[g]) return cache_[g];
    std::vector<std::vector<i32>> out = nbrs_[g];
    if (order == "proximity") {
        // by distance to the group, then a random number, then name
        for (auto& list : out) {
            struct Item {
                i32 g;
                double d, r;
            };
            std::vector<Item> items;
            for (i32 e : list) {
                double dist = 0;
                const auto& a = coords_[static_cast<std::size_t>(e)];
                const auto& b = coords_[g];
                for (std::size_t i = 0; i < b.size(); ++i) dist += perl_pow(a[i] - b[i], 2);
                items.push_back({e, perl_sqrt(dist), rng.rand()});
            }
            std::stable_sort(items.begin(), items.end(), [](const Item& x, const Item& y) {
                if (x.d != y.d) return x.d < y.d;
                if (x.r != y.r) return x.r < y.r;
                return x.g < y.g;
            });
            for (std::size_t i = 0; i < items.size(); ++i) list[i] = items[i].g;
        }
    }
    cache_[g] = out;
    cached_[g] = 1;
    if (same_) {
        // always_same / always_true: this list for every group
        for (std::size_t k = 0; k < cache_.size(); ++k) {
            cache_[k] = out;
            cached_[k] = 1;
        }
    }
    return cache_[g];
}

CountsCsr rand_structured(const CountsCsr& bd, MT64& rng, double multiplier, double addition, SpatialAlloc* sp) {
    const i64 ng = bd.n_groups, nl = bd.n_labels;
    // the basedata as labels -> groups -> count (destructively sampled)
    std::vector<std::map<i32, double>> cloned(static_cast<std::size_t>(nl));
    std::vector<i32> richness(static_cast<std::size_t>(ng), 0);
    for (i64 g = 0; g < ng; ++g)
        for (i64 k = bd.indptr[g]; k < bd.indptr[g + 1]; ++k) {
            cloned[static_cast<std::size_t>(bd.indices[static_cast<std::size_t>(k)])][static_cast<i32>(g)] =
                bd.counts[static_cast<std::size_t>(k)];
            ++richness[static_cast<std::size_t>(g)];
        }
    std::size_t n_cloned = 0;
    for (const auto& m : cloned)
        if (!m.empty()) ++n_cloned;

    std::vector<i32> label_order(static_cast<std::size_t>(nl));
    for (i64 l = 0; l < nl; ++l) label_order[static_cast<std::size_t>(l)] = static_cast<i32>(l);
    rng.shuffle(label_order);

    std::vector<double> target(static_cast<std::size_t>(ng));
    std::map<i32, double> filled;  // group -> richness reached (0 for empty targets)
    std::vector<std::uint8_t> is_filled(static_cast<std::size_t>(ng), 0);  // its keys
    std::set<i32> unfilled;
    for (i64 g = 0; g < ng; ++g) {
        const double t = std::floor(richness[static_cast<std::size_t>(g)] * multiplier + addition);
        target[static_cast<std::size_t>(g)] = t;
        if (t != 0) {
            unfilled.insert(static_cast<i32>(g));
        } else {
            filled[static_cast<i32>(g)] = 0;
            is_filled[static_cast<std::size_t>(g)] = 1;
        }
    }
    // groups still unfilled (in name order): the targets of each label
    std::vector<std::uint8_t> is_unfilled(static_cast<std::size_t>(ng), 0);
    for (i32 g : unfilled) is_unfilled[static_cast<std::size_t>(g)] = 1;
    Fenwick target_groups;

    // the new basedata: groups -> labels -> count
    std::vector<std::map<i32, double>> nw(static_cast<std::size_t>(ng));
    std::vector<i32> new_richness(static_cast<std::size_t>(ng), 0);

    const bool random_prop = sp && (sp->order == "random_walk" || sp->order == "diffusion");
    const bool diffusion = sp && sp->order == "diffusion";
    const bool shuffle_lists = sp && sp->order.rfind("random", 0) == 0;
    if (sp && sp->seeded.size() != static_cast<std::size_t>(nl)) sp->seeded.assign(static_cast<std::size_t>(nl), 0);
    std::vector<std::uint8_t> tmask;
    // per-label sets of groups (assigned, cleared, queued) as stamps: member when == the label's pass
    std::vector<i32> assigned_at(static_cast<std::size_t>(ng), -1), cleared_at(static_cast<std::size_t>(ng), -1),
        queued_at(static_cast<std::size_t>(ng), -1);
    i32 pass = -1;
    for (i32 label : label_order) {
        auto& gp_hash = cloned[static_cast<std::size_t>(label)];
        if (gp_hash.empty()) continue;
        ++pass;
        auto assigned = [&](i32 g) { return assigned_at[static_cast<std::size_t>(g)] == pass; };
        auto cleared = [&](i32 g) { return cleared_at[static_cast<std::size_t>(g)] == pass; };
        auto queue = [&](const std::vector<i32>& v) {
            for (i32 g : v) queued_at[static_cast<std::size_t>(g)] = pass;
        };
        tmask = is_unfilled;
        target_groups.reset(tmask);
        auto remove_target = [&](i32 g) {
            if (tmask[static_cast<std::size_t>(g)]) {
                tmask[static_cast<std::size_t>(g)] = 0;
                target_groups.remove(static_cast<std::size_t>(g));
            }
        };
        std::vector<i32> from_order;
        for (const auto& kv : gp_hash) from_order.push_back(kv.first);
        rng.shuffle(from_order);
        std::size_t from_pos = 0;
        std::deque<i32> to_groups;
        std::vector<i32> seed_targets;
        bool new_seed = false, done = false;
        auto filter = [&](const std::vector<i32>& list, i32 self) {
            std::vector<i32> sub;
            for (i32 g : list) {
                if (is_filled[static_cast<std::size_t>(g)] || assigned(g) || g == self) continue;
                if (diffusion && queued_at[static_cast<std::size_t>(g)] == pass) continue;
                sub.push_back(g);
            }
            return sub;
        };
        while (!done && from_pos < from_order.size() && target_groups.count() > 0) {
            if (to_groups.empty() || new_seed) {
                to_groups.clear();
                new_seed = false;
                i32 to = -1;
                if (sp && sp->have_seeds) {
                    if (!sp->seeded[static_cast<std::size_t>(label)]) {
                        // first lookup of this label's seed groups (cached thereafter)
                        sp->seeded[static_cast<std::size_t>(label)] = 1;
                        for (i32 g : sp->seeds)
                            if (!is_filled[static_cast<std::size_t>(g)]) seed_targets.push_back(g);
                    }
                    i32 seed = -1;
                    while (seed < 0 && !seed_targets.empty()) {
                        const auto jj = static_cast<std::size_t>(rng.irand() % seed_targets.size());
                        seed = seed_targets[jj];
                        seed_targets.erase(seed_targets.begin() + static_cast<std::ptrdiff_t>(jj));
                        auto f = nw[static_cast<std::size_t>(seed)].find(label);
                        if (f != nw[static_cast<std::size_t>(seed)].end() && f->second != 0) seed = -1;
                    }
                    if (seed >= 0) {
                        // its index among the targets, or -1 (the last target) when gone
                        to = tmask[static_cast<std::size_t>(seed)]
                                 ? seed
                                 : static_cast<i32>(target_groups.select(target_groups.count() - 1));
                    }
                }
                if (to < 0) {
                    const i64 j = static_cast<i64>(rng.irand() % static_cast<std::uint64_t>(target_groups.count()));
                    to = static_cast<i32>(target_groups.select(j));
                }
                remove_target(to);
                to_groups.push_back(to);
                cleared_at[static_cast<std::size_t>(to)] = pass;
                if (sp) {
                    for (const auto& list : sp->lists(to, rng)) {
                        auto sub = filter(list, to);
                        if (sub.empty()) continue;
                        queue(sub);
                        if (shuffle_lists) rng.shuffle(sub);
                        to_groups.insert(to_groups.end(), sub.begin(), sub.end());
                    }
                }
            }
            if (to_groups.empty()) break;
            while (!to_groups.empty()) {
                const i32 to = to_groups.front();
                to_groups.pop_front();
                if (from_pos >= from_order.size()) {
                    done = true;
                    break;
                }
                if (random_prop && assigned(to)) break;  // next BY_GROUP
                const i32 from = from_order[from_pos++];
                auto it = gp_hash.find(from);
                const double count = it->second;
                gp_hash.erase(it);
                if (gp_hash.empty()) --n_cloned;
                if (!cleared(to)) {
                    remove_target(to);
                    cleared_at[static_cast<std::size_t>(to)] = pass;
                }
                nw[static_cast<std::size_t>(to)][label] = count;
                assigned_at[static_cast<std::size_t>(to)] = pass;
                const i32 r = ++new_richness[static_cast<std::size_t>(to)];
                if (r >= target[static_cast<std::size_t>(to)]) {
                    filled[to] = r;
                    is_filled[static_cast<std::size_t>(to)] = 1;
                    unfilled.erase(to);
                    is_unfilled[static_cast<std::size_t>(to)] = 0;
                }
                if (sp && sp->reseed_prob != 0 && rng.rand() < sp->reseed_prob) {
                    new_seed = true;
                    break;
                }
                if (random_prop) {
                    const auto& sets = sp->lists(to, rng);
                    std::vector<std::size_t> idx(sets.size());
                    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
                    if (sp->backtracking != "from_start") std::reverse(idx.begin(), idx.end());
                    std::size_t valid = 0;
                    for (std::size_t i : idx) {
                        auto sub = filter(sets[i], to);
                        if (sub.empty()) continue;
                        valid += sub.size();
                        queue(sub);
                        if (shuffle_lists) rng.shuffle(sub);
                        if (sp->backtracking == "from_start")
                            to_groups.insert(to_groups.end(), sub.begin(), sub.end());
                        else
                            to_groups.insert(to_groups.begin(), sub.begin(), sub.end());
                    }
                    if (diffusion || (!valid && sp->backtracking == "random")) {
                        if (!diffusion) {
                            std::set<i32> seen;
                            std::deque<i32> u;
                            for (i32 g : to_groups)
                                if (seen.insert(g).second) u.push_back(g);
                            to_groups.swap(u);
                        }
                        if (to_groups.size() > 1) {
                            const auto k = static_cast<std::size_t>(rng.irand() % to_groups.size());
                            const i32 g = to_groups[k];
                            to_groups.erase(to_groups.begin() + static_cast<std::ptrdiff_t>(k));
                            to_groups.push_front(g);
                        }
                    }
                }
                if (target_groups.count() == 0) {
                    done = true;
                    break;
                }
            }
        }
    }

    // swap_to_reach_richness_targets
    if (n_cloned) {
        std::map<i32, i64> labels_in_unfilled;
        std::map<i32, DetOrder> unfilled_without_label;  // label -> unfilled groups without it
        std::map<i32, std::set<i32>> unfilled_without_label_by_gp;
        for (i32 gp : unfilled) {
            const auto& list = nw[static_cast<std::size_t>(gp)];
            for (i32 l = 0; l < nl; ++l) {
                if (list.count(l)) {
                    ++labels_in_unfilled[l];
                } else {
                    unfilled_without_label[l].push(gp);
                    unfilled_without_label_by_gp[gp].insert(l);
                }
            }
        }
        bool target_has_empty = false;
        for (const auto& kv : filled)
            if (kv.second == 0) target_has_empty = true;
        std::map<i32, DetOrder> groups_without_label;
        std::map<i32, std::vector<i32>> cloned_groups_with_label, orig_groups_with_label;
        std::map<i32, std::vector<i32>> labels_in_gp_array;
        std::vector<i32> cloned_label_arr;
        std::set<i32> cloned_label_set;
        for (i32 l = 0; l < nl; ++l)
            if (!cloned[static_cast<std::size_t>(l)].empty()) {
                cloned_label_arr.push_back(l);
                cloned_label_set.insert(l);
            }
        auto fill_group = [&](i32 gp, i32 r) {
            filled[gp] = r;
            is_filled[static_cast<std::size_t>(gp)] = 1;
            unfilled.erase(gp);
            auto bg = unfilled_without_label_by_gp.find(gp);
            if (bg != unfilled_without_label_by_gp.end()) {
                for (i32 l : bg->second) unfilled_without_label[l].erase(gp);
                unfilled_without_label_by_gp.erase(bg);
            }
        };

        while (!unfilled.empty()) {
            if (!n_cloned) break;
            const i32 add_label =
                cloned_label_arr[static_cast<std::size_t>(rng.irand() % cloned_label_arr.size())];
            auto& from_hash = cloned[static_cast<std::size_t>(add_label)];
            auto fa = cloned_groups_with_label.find(add_label);
            if (fa == cloned_groups_with_label.end() || fa->second.empty()) {
                std::vector<i32> v;
                for (const auto& kv : from_hash) v.push_back(kv.first);
                fa = cloned_groups_with_label.insert_or_assign(add_label, v).first;
            }
            std::vector<i32>& from_arr = fa->second;
            const i32 from = from_arr[static_cast<std::size_t>(rng.irand() % from_arr.size())];
            const double add_count = from_hash.at(from);
            from_hash.erase(from);
            if (from_hash.empty()) --n_cloned;
            bremove(from_arr, from);
            if (from_arr.empty()) cloned_groups_with_label.erase(add_label);
            if (from_hash.empty()) {
                bremove(cloned_label_arr, add_label);
                cloned_label_set.erase(add_label);
            }
            auto ta = groups_without_label.find(add_label);
            if (ta == groups_without_label.end()) {
                std::vector<i32> v;
                for (i64 g = 0; g < ng; ++g) {
                    if (nw[static_cast<std::size_t>(g)].count(add_label)) continue;
                    if (target_has_empty) {
                        auto f = filled.find(static_cast<i32>(g));
                        if (f == filled.end() || f->second == 0) continue;
                    }
                    v.push_back(static_cast<i32>(g));
                }
                ta = groups_without_label.emplace(add_label, DetOrder(v)).first;
            }
            if (ta->second.empty()) throw std::runtime_error("rand_structured: no group can take the label (Illegal modulus zero)");
            const i32 tg = ta->second.at(static_cast<std::size_t>(rng.irand() % ta->second.size()));
            auto& tg_labels = nw[static_cast<std::size_t>(tg)];
            const auto tg_richness = static_cast<double>(tg_labels.size());
            if (tg_richness >= target[static_cast<std::size_t>(tg)]) {
                auto la = labels_in_gp_array.find(tg);
                if (la == labels_in_gp_array.end()) {
                    std::vector<i32> v;
                    for (const auto& kv : tg_labels) v.push_back(kv.first);
                    la = labels_in_gp_array.emplace(tg, v).first;
                }
                std::vector<i32> use;
                for (i32 l : la->second)
                    if (!labels_in_unfilled.count(l)) use.push_back(l);
                if (use.empty()) use = la->second;
                rng.shuffle(use);
                i32 remove_label = use[0];
                bool to_unfilled = false;
                for (i32 l : use) {
                    auto x = unfilled_without_label.find(l);
                    if (x == unfilled_without_label.end() || x->second.empty()) continue;
                    remove_label = l;
                    to_unfilled = true;
                    break;
                }
                const double removed_count = tg_labels.at(remove_label);
                tg_labels.erase(remove_label);
                bremove(la->second, remove_label);
                auto gw = groups_without_label.find(remove_label);
                if (gw != groups_without_label.end()) gw->second.push(tg);
                if (!to_unfilled) {
                    auto og = orig_groups_with_label.find(remove_label);
                    if (og == orig_groups_with_label.end()) {
                        std::vector<i32> v;
                        for (i64 g = 0; g < ng; ++g)
                            for (i64 k = bd.indptr[g]; k < bd.indptr[g + 1]; ++k)
                                if (bd.indices[static_cast<std::size_t>(k)] == remove_label) v.push_back(static_cast<i32>(g));
                        og = orig_groups_with_label.emplace(remove_label, v).first;
                    }
                    auto& rl_hash = cloned[static_cast<std::size_t>(remove_label)];
                    i32 old_gp = -1;
                    for (i32 g : og->second)
                        if (!rl_hash.count(g)) {
                            old_gp = g;
                            break;
                        }
                    if (rl_hash.empty()) ++n_cloned;
                    rl_hash[old_gp] = removed_count;
                    binsert(cloned_groups_with_label[remove_label], old_gp);
                    if (!cloned_label_set.count(remove_label)) {
                        binsert(cloned_label_arr, remove_label);
                        cloned_label_set.insert(remove_label);
                    }
                } else {
                    DetOrder& ul = unfilled_without_label[remove_label];
                    const i32 return_gp = ul.erase_at(static_cast<std::size_t>(rng.irand() % ul.size()));
                    nw[static_cast<std::size_t>(return_gp)][remove_label] = removed_count;
                    auto ra = labels_in_gp_array.find(return_gp);
                    if (ra != labels_in_gp_array.end()) binsert(ra->second, remove_label);
                    const auto nr = static_cast<i32>(nw[static_cast<std::size_t>(return_gp)].size());
                    ++labels_in_unfilled[remove_label];
                    auto bg = unfilled_without_label_by_gp.find(return_gp);
                    if (bg != unfilled_without_label_by_gp.end()) bg->second.erase(remove_label);
                    if (nr >= target[static_cast<std::size_t>(return_gp)]) {
                        fill_group(return_gp, nr);
                        for (const auto& kv : nw[static_cast<std::size_t>(return_gp)]) {
                            auto li = labels_in_unfilled.find(kv.first);
                            if (li == labels_in_unfilled.end()) continue;
                            if (li->second <= 1)
                                labels_in_unfilled.erase(li);
                            else
                                --li->second;
                        }
                    }
                }
            }
            tg_labels[add_label] += add_count;
            auto ga = labels_in_gp_array.find(tg);
            if (ga != labels_in_gp_array.end()) binsert(ga->second, add_label);
            auto gw = groups_without_label.find(add_label);
            if (gw != groups_without_label.end()) gw->second.erase(tg);
            if (unfilled.count(tg)) {
                auto bg = unfilled_without_label_by_gp.find(tg);
                if (bg != unfilled_without_label_by_gp.end()) bg->second.erase(add_label);
                unfilled_without_label[add_label].erase(tg);
            }
            const auto nr = static_cast<double>(tg_labels.size());
            if (nr != tg_richness && nr >= target[static_cast<std::size_t>(tg)]) fill_group(tg, static_cast<i32>(nr));
        }
    }

    CountsCsr out;
    out.n_groups = ng;
    out.n_labels = nl;
    out.indptr.assign(static_cast<std::size_t>(ng) + 1, 0);
    for (i64 g = 0; g < ng; ++g) {
        for (const auto& kv : nw[static_cast<std::size_t>(g)]) {
            out.indices.push_back(kv.first);
            out.counts.push_back(kv.second);
        }
        out.indptr[static_cast<std::size_t>(g) + 1] = static_cast<i64>(out.indices.size());
    }
    return out;
}

}  // namespace bio
