// SPDX-License-Identifier: GPL-3.0-or-later
//
// Basedata randomisations (see randomise.hpp), following Biodiverse::Randomise.

#include "bio/randomise.hpp"

#include <algorithm>
#include <cmath>
#include <map>
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

}  // namespace

CountsCsr rand_structured(const CountsCsr& bd, MT64& rng, double multiplier, double addition) {
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
    std::set<i32> unfilled;
    for (i64 g = 0; g < ng; ++g) {
        const double t = std::floor(richness[static_cast<std::size_t>(g)] * multiplier + addition);
        target[static_cast<std::size_t>(g)] = t;
        if (t != 0)
            unfilled.insert(static_cast<i32>(g));
        else
            filled[static_cast<i32>(g)] = 0;
    }
    // groups still unfilled (in name order): the targets of each label
    std::vector<std::uint8_t> is_unfilled(static_cast<std::size_t>(ng), 0);
    for (i32 g : unfilled) is_unfilled[static_cast<std::size_t>(g)] = 1;
    Fenwick target_groups;

    // the new basedata: groups -> labels -> count
    std::vector<std::map<i32, double>> nw(static_cast<std::size_t>(ng));
    std::vector<i32> new_richness(static_cast<std::size_t>(ng), 0);

    for (i32 label : label_order) {
        auto& gp_hash = cloned[static_cast<std::size_t>(label)];
        if (gp_hash.empty()) continue;
        target_groups.reset(is_unfilled);
        std::vector<i32> from_order;
        for (const auto& kv : gp_hash) from_order.push_back(kv.first);
        rng.shuffle(from_order);
        std::size_t from_pos = 0;
        while (from_pos < from_order.size() && target_groups.count() > 0) {
            const i64 j = static_cast<i64>(rng.irand() % static_cast<std::uint64_t>(target_groups.count()));
            const auto pos = target_groups.select(j);
            const i32 to = static_cast<i32>(pos);
            target_groups.remove(pos);
            const i32 from = from_order[from_pos++];
            auto it = gp_hash.find(from);
            const double count = it->second;
            gp_hash.erase(it);
            if (gp_hash.empty()) --n_cloned;
            nw[static_cast<std::size_t>(to)][label] = count;
            const i32 r = ++new_richness[static_cast<std::size_t>(to)];
            if (r >= target[static_cast<std::size_t>(to)]) {
                filled[to] = r;
                unfilled.erase(to);
                is_unfilled[static_cast<std::size_t>(to)] = 0;
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
