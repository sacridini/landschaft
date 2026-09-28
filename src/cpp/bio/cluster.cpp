// SPDX-License-Identifier: GPL-3.0-or-later
//
// Agglomerative clustering (see cluster.hpp), following Biodiverse::Cluster
// and Biodiverse::Matrix.

#include "bio/cluster.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace bio {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

std::string perl_string(double v) {
    if (v == 0) return "0";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.15g", v);
    return buf;
}

std::string index_key(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.2g", v);
    return buf;
}

inline std::uint64_t pkey(i32 a, i32 b) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b);
}

// Biodiverse::Matrix: values by (stored) element pair and by value (bucketed by %.2g)
class Mx {
public:
    struct Item {
        double v;
        i32 a, b;
        bool operator<(const Item& o) const {
            if (v != o.v) return v < o.v;
            if (a != o.a) return a < o.a;
            return b < o.b;
        }
    };
    struct Bucket {
        std::string key;
        std::set<Item> items;  // by value
    };
    using BKey = std::pair<double, std::string>;
    using BIt = std::map<BKey, Bucket>::iterator;
    struct Entry {
        double v;
        BIt bucket;
    };

    int exists(i32 a, i32 b) const {
        if (by_el_.count(pkey(a, b))) return 1;
        if (by_el_.count(pkey(b, a))) return 2;
        return 0;
    }
    double defined(i32 a, i32 b) const {
        auto it = by_el_.find(pkey(a, b));
        if (it != by_el_.end()) return it->second.v;
        it = by_el_.find(pkey(b, a));
        return it != by_el_.end() ? it->second.v : NaN;
    }
    void add(i32 a, i32 b, double v) {
        const BIt bi = bucket_for(v);
        bi->second.items.insert({v, a, b});
        by_el_[pkey(a, b)] = Entry{v, bi};
        ++elements_[a];
        ++elements_[b];
    }
    bool del(i32 a, i32 b) {
        const int e = exists(a, b);
        if (!e) return false;
        if (e == 2) std::swap(a, b);
        auto it = by_el_.find(pkey(a, b));
        const Entry en = it->second;
        by_el_.erase(it);
        en.bucket->second.items.erase({en.v, a, b});
        if (en.bucket->second.items.empty()) {
            lookup_.erase(en.bucket->second.key);
            by_val_.erase(en.bucket);
        }
        for (i32 x : {a, b}) {
            auto ei = elements_.find(x);
            if (--ei->second == 0) elements_.erase(ei);
        }
        return true;
    }
    std::size_t element_count() const { return elements_.size(); }
    std::vector<i32> elements() const {
        std::vector<i32> out;
        out.reserve(elements_.size());
        for (const auto& kv : elements_) out.push_back(kv.first);
        return out;
    }
    // get_min_value / get_max_value
    double extreme(bool max) const {
        if (by_val_.empty()) return NaN;
        if (!max) {
            const auto& b = *by_val_.begin();
            if (b.first.second == "0") return 0;
            return b.second.items.begin()->v;
        }
        return by_val_.rbegin()->second.items.rbegin()->v;
    }
    // get_elements_with_value: pairs of the value's bucket that Perl prints as the value
    std::vector<std::pair<i32, i32>> pairs_with(double v) const {
        std::vector<std::pair<i32, i32>> out;
        auto li = lookup_.find(index_key(v));
        if (li == lookup_.end()) return out;
        const auto& items = li->second->second.items;
        const std::string s = perl_string(v);
        // values Perl prints alike are contiguous: start at the value and scan both ways
        auto it = items.lower_bound({v, std::numeric_limits<i32>::min(), std::numeric_limits<i32>::min()});
        for (auto f = it; f != items.end() && (f->v == v || perl_string(f->v) == s); ++f) out.push_back({f->a, f->b});
        for (auto b = it; b != items.begin();) {
            --b;
            if (b->v != v && perl_string(b->v) != s) break;
            out.push_back({b->a, b->b});
        }
        return out;
    }

private:
    BIt bucket_for(double v) {
        const std::string k = index_key(v);
        auto li = lookup_.find(k);
        if (li != lookup_.end()) return li->second;
        BIt bi = by_val_.emplace(BKey(std::strtod(k.c_str(), nullptr), k), Bucket{k, {}}).first;
        lookup_.emplace(k, bi);
        return bi;
    }
    std::unordered_map<std::uint64_t, Entry> by_el_;
    std::map<BKey, Bucket> by_val_;
    std::unordered_map<std::string, BIt> lookup_;
    std::unordered_map<i32, i64> elements_;
};

}  // namespace

ClusterOutput cluster_groups(const ClusterInput& in, MT64& rng) {
    const i32 n = static_cast<i32>(in.names.size());
    std::vector<std::string> names = in.names;
    auto by_name = [&](i32 a, i32 b) { return names[static_cast<std::size_t>(a)] < names[static_cast<std::size_t>(b)]; };

    // index values
    std::unordered_map<std::int64_t, double> given;
    for (std::size_t i = 0; i < in.pair_keys.size(); ++i) given.emplace(in.pair_keys[i], in.pair_values[i]);
    auto index_value = [&](i32 e1, i32 e2) -> double {
        if (in.direct.empty()) {
            auto it = given.find(static_cast<std::int64_t>(e1) * n + e2);
            return it == given.end() ? NaN : it->second;
        }
        const auto& h1 = in.labels[static_cast<std::size_t>(e1)];
        const auto& h2 = in.labels[static_cast<std::size_t>(e2)];
        std::size_t aa = 0;
        std::size_t i = 0, j = 0;
        while (i < h1.size() && j < h2.size()) {
            if (h1[i] < h2[j])
                ++i;
            else if (h2[j] < h1[i])
                ++j;
            else {
                ++aa;
                ++i;
                ++j;
            }
        }
        const double a = static_cast<double>(aa), b = static_cast<double>(h1.size() - aa),
                     c = static_cast<double>(h2.size() - aa);
        if (!(aa || (b != 0 && c != 0))) return NaN;
        if (in.direct == "SORENSON") return 1 - (2 * a) / (2 * a + b + c);
        if (in.direct == "JACCARD") return 1 - a / (a + b + c);
        return 1 - a / (a + std::min(b, c));  // S2
    };

    // build the matrices (one per condition, plus a shadow matrix of all pairs)
    const std::size_t nm = in.nbrs.size();
    std::vector<Mx> mxs(nm);
    Mx shadow;
    const bool have_shadow = nm > 1;
    std::vector<std::uint8_t> processed(static_cast<std::size_t>(n), 0);
    for (i32 e1 = 0; e1 < n; ++e1) {
        std::vector<std::vector<i32>> nb(nm);
        for (std::size_t m = 0; m < nm; ++m) {
            for (i32 e2 : in.nbrs[m][static_cast<std::size_t>(e1)]) {
                if (e2 == e1) continue;
                if (in.trim_processed && processed[static_cast<std::size_t>(e2)]) continue;
                nb[m].push_back(e2);
            }
            std::sort(nb[m].begin(), nb[m].end(), by_name);
            nb[m].erase(std::unique(nb[m].begin(), nb[m].end()), nb[m].end());
        }
        for (std::size_t m = 0; m < nm; ++m) {
            std::vector<Mx*> arr{&mxs[m]};
            if (have_shadow) arr.push_back(&shadow);
            for (i32 e2 : nb[m]) {
                if (!in.trim_processed) {
                    std::size_t found = 0;
                    std::vector<std::size_t> missing;
                    double value = NaN;
                    for (std::size_t k = 0; k < arr.size(); ++k) {
                        value = arr[k]->defined(e1, e2);
                        if (!std::isnan(value))
                            ++found;
                        else
                            missing.push_back(k);
                    }
                    if (found == arr.size()) continue;
                    if (found) {
                        // the value of the last matrix checked (undefined: not added)
                        if (!std::isnan(value))
                            for (std::size_t k : missing) arr[k]->add(e1, e2, value);
                        continue;
                    }
                }
                const double v = index_value(e1, e2);
                if (std::isnan(v)) continue;
                for (Mx* mx : arr) mx->add(e1, e2, v);
            }
        }
        processed[static_cast<std::size_t>(e1)] = 1;
    }

    ClusterOutput out;
    Mx& for_nodes = have_shadow ? shadow : mxs[0];
    {
        auto el = for_nodes.elements();
        std::sort(el.begin(), el.end(), by_name);
        for (i32 e : el) out.terminals.push_back(names[static_cast<std::size_t>(e)]);
    }
    std::vector<i64> tcount(static_cast<std::size_t>(n), 1);

    auto linkage = [&](i32 n1, i32 n2, i32 c, const Mx& mx) {
        const double t1 = mx.defined(c, n1), t2 = mx.defined(c, n2);
        if (in.linkage == "link_average") {
            const double c1 = static_cast<double>(tcount[static_cast<std::size_t>(n1)]);
            const double c2 = static_cast<double>(tcount[static_cast<std::size_t>(n2)]);
            return (c1 * t1 + c2 * t2) / (c1 + c2);
        }
        if (in.linkage == "link_average_unweighted") return (t1 + t2) / 2;
        if (in.linkage == "link_minimum") return t2 < t1 ? t2 : t1;
        if (in.linkage == "link_maximum") return t2 < t1 ? t1 : t2;
        throw std::invalid_argument("linkage function not supported: " + in.linkage);
    };

    std::map<std::pair<std::string, std::string>, double> tie_cache;
    double min_value = NaN;
    i32 join_param = -1;
    for (std::size_t it = 0; it < nm; ++it) {
        Mx& sim = mxs[it];
        if (sim.element_count() == 0) continue;
        i32 join_number = join_param != 0 ? join_param : -1;  // `|| -1`
        double msv = NaN, prev = NaN;
        bool have_prev = false;
        while (sim.element_count() > 0) {
            msv = sim.extreme(in.objective_max);
            if (have_prev && msv != prev) tie_cache.clear();
            // get_most_similar_pair
            auto pairs = sim.pairs_with(msv);
            std::sort(pairs.begin(), pairs.end(), [&](const auto& x, const auto& y) {
                if (x.first != y.first) return by_name(x.first, y.first);
                return by_name(x.second, y.second);
            });
            i32 node1 = -1, node2 = -1;
            std::vector<std::pair<i32, i32>> extras;
            if (msv == 0 && in.min_poss == 0 && in.can_lump_zeroes) {
                node1 = pairs.front().first;
                node2 = pairs.front().second;
                std::set<i32> k1;
                std::size_t k2 = 0;
                for (const auto& p : pairs) {
                    k1.insert(p.first);
                    if (p.first == node1) ++k2;
                }
                if (k1.size() > 1 || k2 > 1) extras = pairs;
            } else if (!in.have_tie_breaker) {
                std::vector<i32> keys1;
                for (const auto& p : pairs)
                    if (keys1.empty() || keys1.back() != p.first) keys1.push_back(p.first);
                rng.shuffle(keys1);
                node1 = keys1[0];
                std::vector<i32> keys2;
                for (const auto& p : pairs)
                    if (p.first == node1) keys2.push_back(p.second);
                rng.shuffle(keys2);
                node2 = keys2[0];
            } else if (pairs.size() == 1) {
                node1 = pairs[0].first;
                node2 = pairs[0].second;
            } else if (in.breaker_keys.empty()) {
                node1 = pairs.back().first;  // the largest by name
                node2 = pairs.back().second;
            } else {
                // random scores per pair (kept while the value stays the same)
                std::size_t lead = 0;
                std::vector<double> scores(pairs.size());
                for (std::size_t k = 0; k < pairs.size(); ++k) {
                    const auto& nm1 = names[static_cast<std::size_t>(pairs[k].first)];
                    const auto& nm2 = names[static_cast<std::size_t>(pairs[k].second)];
                    auto c = tie_cache.find({nm1, nm2});
                    if (c == tie_cache.end()) c = tie_cache.find({nm2, nm1});
                    if (c == tie_cache.end()) c = tie_cache.emplace(std::make_pair(nm1, nm2), rng.rand()).first;
                    scores[k] = c->second;
                    if (k == 0) continue;
                    int cmp = scores[lead] < scores[k] ? -1 : (scores[lead] > scores[k] ? 1 : 0);
                    if (cmp == 0) continue;  // ties keep the lead
                    if (in.breaker_minmax[0].rfind("max", 0) == 0) cmp = -cmp;
                    if (cmp > 0) lead = k;
                }
                node1 = pairs[lead].first;
                node2 = pairs[lead].second;
            }
            std::set<i32> done;
            std::size_t extra_pos = 0;
            while (node1 >= 0) {
                ++join_number;
                const std::string new_name = std::to_string(join_number) + "___";
                const i32 nid = static_cast<i32>(names.size());
                names.push_back(new_name);
                tcount.push_back(tcount[static_cast<std::size_t>(node1)] + tcount[static_cast<std::size_t>(node2)]);
                out.joins.push_back({join_number, new_name, names[static_cast<std::size_t>(node1)],
                                     names[static_cast<std::size_t>(node2)], msv, static_cast<i32>(it)});
                // run_linkage
                Mx& mwe = have_shadow ? shadow : mxs[0];
                std::vector<i32> check;
                for (i32 c : mwe.elements()) {
                    if (c == node1 || c == node2) continue;
                    if (!(mwe.exists(c, node1) && mwe.exists(c, node2))) continue;
                    check.push_back(c);
                }
                std::sort(check.begin(), check.end(), by_name);
                std::vector<double> recalc;
                if (in.linkage == "link_recalculate") {
                    std::vector<std::string> cn;
                    for (i32 c : check) cn.push_back(names[static_cast<std::size_t>(c)]);
                    recalc = in.recalculate(names[static_cast<std::size_t>(node1)], names[static_cast<std::size_t>(node2)],
                                            new_name, cn);
                }
                for (std::size_t ci = 0; ci < check.size(); ++ci) {
                    const i32 c = check[ci];
                    const double v = in.linkage == "link_recalculate" ? recalc[ci] : linkage(node1, node2, c, mwe);
                    if (std::isnan(v)) continue;  // an undefined value is not added
                    if (have_shadow) shadow.add(nid, c, v);
                    for (std::size_t mi = it; mi < nm; ++mi) {
                        if (!(mxs[mi].exists(node1, c) && mxs[mi].exists(node2, c))) continue;
                        mxs[mi].add(nid, c, v);
                        break;
                    }
                }
                auto del_links = [&](Mx& mx) {
                    for (i32 c : mx.elements()) {
                        mx.del(c, node1);
                        mx.del(c, node2);
                    }
                };
                if (have_shadow) del_links(shadow);
                for (auto& mx : mxs) del_links(mx);
                done.insert(node1);
                done.insert(node2);
                node1 = node2 = -1;
                while (extra_pos < extras.size()) {
                    const auto p = extras[extra_pos++];
                    if (done.count(p.first) || done.count(p.second)) continue;
                    node1 = p.first;
                    node2 = p.second;
                    break;
                }
            }
            prev = msv;
            have_prev = true;
            if (!std::isnan(in.max_poss) && in.max_poss == msv) break;
        }
        tie_cache.clear();
        join_param = join_number;
        min_value = msv;
    }
    out.join_number = join_param;
    out.min_value = min_value;
    return out;
}

}  // namespace bio
