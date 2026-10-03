// SPDX-License-Identifier: GPL-3.0-or-later
//
// r.futures.simulation (GRASS GIS addons, FUTURES 3) step for step: see
// futures.hpp. Corrected here (the reference is the module patched the same
// way): after growing a patch the module updates the development pressure
// around the first n grown cells, n being the cells grown inside the
// seed's subregion, so a patch crossing into another subregion leaves its
// last cells without their pressure; here every grown cell adds its
// pressure.

#include "sim/futures.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "lsm/parallel.hpp"
#include "sim/rng.hpp"

namespace sim {

namespace {

constexpr int kMaxCandidateIter = 100;
constexpr int kMaxSeedIter = 20;
constexpr int kMaxTriesIter = 1000;

struct Developable {
    std::int64_t id;
    float probability;
    float cumulative;
    bool tried;
};

struct Candidate {
    double selection;
    double ranking;
    std::int64_t id;
};

inline double distance(std::int64_t r1, std::int64_t c1, std::int64_t r2, std::int64_t c2) {
    return std::sqrt(static_cast<double>(static_cast<int>((r1 - r2) * (r1 - r2) + (c1 - c2) * (c1 - c2))));
}

class Futures {
public:
    Futures(const FuturesSettings& s, const int* developed, const int* region, const int* pot_region,
            const float* devpressure, const float* aggregated, const float* weight, const int* zone)
        : s_(s), ncell_(s.rows * s.cols), dev_(developed, developed + s.rows * s.cols), region_(region),
          pot_(pot_region), pressure_(devpressure, devpressure + s.rows * s.cols), agg_(aggregated),
          weight_(weight), zone_(zone), prob_(static_cast<std::size_t>(s.rows * s.cols), 0.0f),
          rng_(s.seed), cells_(s.nregions), demand_(s.demand) {
        const int n = s.neighborhood, side = 2 * n + 1;
        matrix_.assign(static_cast<std::size_t>(side * side), 0.0f);
        for (int i = 0; i < side; ++i) {
            for (int j = 0; j < side; ++j) {
                const double d = distance(i, j, n, n);
                double v;
                if (d > n) v = 0;
                else if (s.approach == 0) v = 1;
                else if (s.approach == 1) v = static_cast<double>(s.scaling) / std::pow(d, static_cast<double>(s.gamma));
                else v = static_cast<double>(s.scaling) * std::exp(-2 * d / static_cast<double>(s.gamma));
                matrix_[i * side + j] = static_cast<float>(v);
            }
        }
    }

    FuturesResult run() {
        FuturesResult out;
        if (s_.climate) init_climate();
        std::vector<int> overflow(s_.nregions, 0);
        bool overgrow = true;
        added_.assign(static_cast<std::size_t>(std::max(s_.max_patch_size, 1)), 0);
        for (int step = 0; step < s_.nsteps; ++step) {
            recompute();
            if (step == s_.nsteps - 1) overgrow = false;
            for (int r = 0; r < s_.nregions; ++r) {
                int warn = 0;
                const auto done = compute_step(step, r, overgrow, overflow, warn);
                out.converted.push_back(done);
                out.warnings.push_back(warn);
            }
            for (int r = 0; r < s_.nregions; ++r) out.overflow.push_back(overflow[r]);
            if (s_.climate) {
                update_floods(step);
                for (int h = 0; h < s_.climate->nhuc; ++h) climate_step(step, h, out);
                out.adaptation.push_back(adaptation_);
                out.moved.push_back(moved_);
            }
            out.series.push_back(dev_);
        }
        out.developed = dev_;
        out.demand = demand_;
        return out;
    }

private:
    float probability_of(std::int64_t c, int reg) const {
        const int pot = pot_ ? pot_[c] : reg;
        float p = static_cast<float>(s_.intercept[pot]);
        p = static_cast<float>(static_cast<double>(p) + s_.devpressure_coef[pot] * static_cast<double>(pressure_[c]));
        p = p + agg_[c];
        p = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(p))));
        if (!s_.incentive.empty()) {
            const int size = static_cast<int>(s_.incentive.size());
            const int idx = static_cast<int>(p * static_cast<float>(size - 1));
            if (idx >= size || idx < 0)
                throw std::runtime_error("lookup position (" + std::to_string(idx) + ") out of range");
            p = s_.incentive[idx];
        }
        if (s_.use_weight) {
            const float w = weight_[c];
            if (w < 0) p = static_cast<float>(static_cast<double>(p) * (1 - std::fabs(static_cast<double>(w))));
            else if (w > 0) p = p + w - p * w;
        }
        if (s_.use_zone) {
            float effect = zone_effect(zone_[c], reg);
            if (effect != 0) {
                if (s_.user_effects) {
                    const float st = s_.stringency[reg];
                    if (st != 1) effect *= st;
                    if (effect > 1) effect = 1;
                    else if (effect < -1) effect = -1;
                }
                if (effect < 0) p *= 1 + effect;
                else if (effect > 0) p = (p + effect) - (p * effect);
            }
        }
        return p;
    }

    float zone_effect(int id, int reg) const {
        for (std::size_t k = 0; k < s_.zone_ids.size(); ++k) {
            if (s_.zone_ids[k] == id && (s_.zone_regions[k] < 0 || s_.zone_regions[k] == reg))
                return s_.zone_effects[k];
        }
        throw std::runtime_error("no zoning effect for zone id " + std::to_string(id) + " in region index " +
                                 std::to_string(reg));
    }

    void recompute() {
        for (auto& v : cells_) v.clear();
        for (std::int64_t c = 0; c < ncell_; ++c) {
            if (dev_[c] != -1) continue;  // null or developed
            const int reg = region_[c];
            const float p = probability_of(c, reg);
            prob_[c] = p;
            cells_[reg].push_back({c, p, 0.0f, false});
        }
        for (auto& v : cells_) {
            if (v.empty()) continue;
            v[0].cumulative = v[0].probability;
            for (std::size_t i = 1; i < v.size(); ++i) v[i].cumulative = v[i - 1].cumulative + v[i].probability;
            const float sum = v.back().cumulative;
            for (auto& x : v) x.cumulative /= sum;
        }
    }

    std::size_t find_probable_seed(int reg) {
        const auto& v = cells_[reg];
        const double p = rng_.next_double();
        int first = 0, last = static_cast<int>(v.size()) - 1;
        int middle = (first + last) / 2;
        if (p <= v[first].cumulative) return 0;
        if (p >= v[last].cumulative) return static_cast<std::size_t>(last);
        while (first <= last) {
            if (v[middle].cumulative < p) {
                first = middle + 1;
            } else if (v[middle - 1].cumulative < p && v[middle].cumulative >= p) {
                return static_cast<std::size_t>(middle);
            } else {
                last = middle - 1;
            }
            middle = (first + last) / 2;
        }
        return 0;
    }

    std::size_t get_seed(int reg) {
        if (!s_.seed_probability)
            return static_cast<std::size_t>(static_cast<int>(rng_.next_double() *
                                                             static_cast<double>(cells_[reg].size())));
        return find_probable_seed(reg);
    }

    int patch_size(int reg) {
        const auto& sizes = s_.patch_sizes[s_.single_column ? 0 : reg];
        return sizes[static_cast<int>(rng_.next_double() * static_cast<double>(sizes.size()))];
    }

    float alpha() {
        float a = static_cast<float>(static_cast<double>(s_.compactness_mean) -
                                     static_cast<double>(s_.compactness_range) * 0.5);
        a = static_cast<float>(static_cast<double>(a) + rng_.next_double() * static_cast<double>(s_.compactness_range));
        return a;
    }

    void add_neighbour(std::int64_t row, std::int64_t col, std::int64_t sr, std::int64_t sc) {
        if (row < 0 || row >= s_.rows || col < 0 || col >= s_.cols) return;
        const std::int64_t id = row * s_.cols + col;
        if (dev_[id] != -1) return;  // null, or developed
        for (const auto& c : cand_)
            if (c.id == id) return;
        const float p = prob_[id];
        const float a = alpha();
        cand_.push_back({static_cast<double>(p), static_cast<double>(p) / std::pow(distance(sr, sc, row, col),
                                                                                    static_cast<double>(a)),
                         id});
    }

    void add_neighbours(std::int64_t row, std::int64_t col, std::int64_t sr, std::int64_t sc) {
        add_neighbour(row - 1, col, sr, sc);
        add_neighbour(row + 1, col, sr, sc);
        add_neighbour(row, col - 1, sr, sc);
        add_neighbour(row, col + 1, sr, sc);
        if (s_.num_neighbors == 8) {
            add_neighbour(row - 1, col - 1, sr, sc);
            add_neighbour(row - 1, col + 1, sr, sc);
            add_neighbour(row + 1, col - 1, sr, sc);
            add_neighbour(row + 1, col + 1, sr, sc);
        }
    }

    // grows a patch; returns (cells inside the seed's subregion, cells in all)
    std::pair<int, int> grow_patch(std::int64_t seed, int size, int step, int reg, std::vector<int>& overflow) {
        const std::int64_t sr = seed / s_.cols, sc = seed % s_.cols;
        const int value = step + 1;
        int found = 1, inside = 1;
        bool skip = false;
        dev_[seed] = value;
        if (static_cast<int>(added_.size()) < std::max(size, 1)) added_.resize(static_cast<std::size_t>(size));
        added_[0] = static_cast<int>(seed);
        cand_.clear();
        add_neighbours(sr, sc, sr, sc);
        int iter = 0;
        while (!cand_.empty() && found < size && !skip) {
            std::size_t i = 0;
            while (true) {
                const double r = rng_.next_double();
                if (r < cand_[i].selection) {
                    const std::int64_t id = cand_[i].id;
                    added_[found] = static_cast<int>(id);
                    dev_[id] = value;
                    cand_.erase(cand_.begin() + static_cast<std::ptrdiff_t>(i));
                    add_neighbours(id / s_.cols, id % s_.cols, sr, sc);
                    std::stable_sort(cand_.begin(), cand_.end(),
                                     [](const Candidate& a, const Candidate& b) { return a.ranking > b.ranking; });
                    const int test = region_[id];
                    if (test != reg) ++overflow[test];
                    else ++inside;
                    ++found;
                    iter = 0;
                    break;
                }
                ++i;
                if (i == cand_.size()) {
                    i = 0;
                    ++iter;
                    if (iter > kMaxCandidateIter) {  // the module's strategy: skip
                        skip = true;
                        break;
                    }
                }
            }
        }
        return {inside, found};
    }

    void update_pressure(std::int64_t id) {
        const int n = s_.neighborhood, side = 2 * n + 1;
        const std::int64_t row = id / s_.cols, col = id % s_.cols;
        for (std::int64_t i = row - n; i <= row + n; ++i) {
            for (std::int64_t j = col - n; j <= col + n; ++j) {
                if (i < 0 || j < 0 || i >= s_.rows || j >= s_.cols) continue;
                const float v = matrix_[(n - (row - i)) * side + (n - (col - j))];
                if (v > 0) {
                    float& d = pressure_[i * s_.cols + j];
                    if (std::isnan(d)) continue;
                    d += v;
                }
            }
        }
    }

    std::int64_t compute_step(int step, int reg, bool overgrow, std::vector<int>& overflow, int& warn) {
        auto& cells = cells_[reg];
        int n_to_convert = static_cast<int>(std::lround(demand_[static_cast<std::size_t>(reg) * s_.nsteps + step]));
        int n_done = 0;
        bool force = false, allow_tried = false;
        int unsuccessful = 0;
        int extra = overflow[reg];
        if (extra > 0) {
            if (n_to_convert - extra > 0) {
                n_to_convert -= extra;
                extra = 0;
            } else {
                extra -= n_to_convert;
                n_to_convert = 0;
            }
        }
        if (static_cast<std::size_t>(std::max(n_to_convert, 0)) > cells.size()) {
            warn |= 1;
            n_to_convert = static_cast<int>(cells.size());
            force = true;
        }
        while (n_done < n_to_convert) {
            if (unsuccessful > kMaxTriesIter) {
                warn |= 2;
                break;
            }
            if (!allow_tried && unsuccessful > kMaxSeedIter * n_to_convert) allow_tried = true;
            const std::size_t idx = get_seed(reg);
            if (!allow_tried && cells[idx].tried) {
                ++unsuccessful;
                continue;
            }
            cells[idx].tried = true;
            const std::int64_t seed = cells[idx].id;
            if (dev_[seed] != -1) {
                ++unsuccessful;
                continue;
            }
            const float prob = prob_[seed];
            if (force || rng_.next_double() < static_cast<double>(prob)) {
                int size = patch_size(reg);
                if (!overgrow && size + n_done > n_to_convert) size = n_to_convert - n_done;
                const auto [inside, total] = grow_patch(seed, size, step, reg, overflow);
                n_done += inside;
                for (int k = 0; k < total; ++k) update_pressure(added_[k]);
            }
        }
        extra += n_done - n_to_convert;
        overflow[reg] = extra;
        return n_done;
    }

    // ------------------------------------------------------------------ climate

    void init_climate() {
        const auto& c = *s_.climate;
        adaptation_ = c.adaptation.empty() ? std::vector<int>(static_cast<std::size_t>(ncell_), 0) : c.adaptation;
        max_flood_.assign(static_cast<std::size_t>(c.nhuc), 0.0f);
        bbox_.assign(static_cast<std::size_t>(c.nhuc), {-1, -1, -1, -1});
        for (std::int64_t k = 0; k < ncell_; ++k) {  // create_bboxes over the simulated cells
            if (dev_[k] == kFuturesNull) continue;
            const int h = c.huc[k];
            if (h < 0) continue;
            const int row = static_cast<int>(k / s_.cols), col = static_cast<int>(k % s_.cols);
            auto& b = bbox_[h];
            if (b[0] < 0) {
                b = {row, row, col, col};  // n, s, e, w
            } else {
                b[2] = std::max(b[2], col);
                b[3] = std::min(b[3], col);
                b[0] = std::min(b[0], row);
                b[1] = std::max(b[1], row);
            }
        }
        if (c.has_matrix) moved_.assign(c.from_ids.size() * c.to_ids.size(), 0.0f);
    }

    int listed(int step) const {
        const auto& v = s_.climate->flood_steps;
        for (std::size_t i = 0; i < v.size(); ++i)
            if (v[i] == step) return static_cast<int>(i);
        return -1;
    }

    void update_floods(int step) {
        const auto& c = *s_.climate;
        const int k = listed(step);
        if (k < 0) return;
        if (c.depth_mode) {
            current_depths_ = &c.depths[k];
            float max_rp = 0;
            for (float rp : c.probabilities)
                if (rp > max_rp) max_rp = rp;
            std::fill(max_flood_.begin(), max_flood_.end(), max_rp);
        } else {
            current_prob_ = &c.flood_probability[k];
            std::fill(max_flood_.begin(), max_flood_.end(), 0.0f);
            for (std::int64_t i = 0; i < ncell_; ++i) {
                const float f = (*current_prob_)[i];
                if (std::isnan(f)) continue;
                const int h = c.huc[i];
                if (h < 0) continue;  // only catchments of simulated cells (corrected)
                if (f > max_flood_[h]) max_flood_[h] = f;
            }
        }
    }

    bool is_adapted(float p, std::int64_t k) const {
        const int a = adaptation_[k];
        if (a == kFuturesNull || a == 0) return false;
        return static_cast<double>(p) >= (1. / a);
    }

    void adapt(float p, std::int64_t k) {
        static const int rp[] = {2, 5, 10, 20, 50, 100};
        unsigned i;
        for (i = 0; i < 6; ++i) {
            if (1 / p < static_cast<float>(rp[i])) {
                ++i;
                break;
            }
        }
        adaptation_[k] = rp[i - 1];
    }

    float depth_to_damage(float depth, int r) const {
        const auto& c = *s_.climate;
        const int n = static_cast<int>(c.levels.size());
        for (int i = 0; i < n; ++i)
            if (static_cast<double>(depth) < c.levels[i]) return static_cast<float>(c.damage[r][i] / 100);
        return static_cast<float>(c.damage[r][n - 1] / 100);
    }

    void gauss_xy(double sd, double& x, double& y) {
        double r = 0., v1 = 0, v2 = 0;
        while (r >= 1. || r == 0.) {
            v1 = rng_.next_double() * 2. - 1.;
            v2 = rng_.next_double() * 2. - 1.;
            r = v1 * v1 + v2 * v2;
        }
        const double fac = std::sqrt(std::log(r) * -2. / r);
        y = 0 + sd * v1 * fac;
        x = 0 + sd * v2 * fac;
    }

    int response(float damage, float ac) {  // 0 retreat, 1 adapt, 2 stay
        const auto& c = *s_.climate;
        double x, y;
        if (c.stddev == 0) {
            x = y = 0;
        } else {
            gauss_xy(static_cast<double>(c.stddev), x, y);
        }
        if (static_cast<double>(ac) + x > 0) {
            const double v = static_cast<double>(c.resilience_a) * (static_cast<double>(ac) + x) +
                             static_cast<double>(c.resilience_b);
            return static_cast<double>(damage) + y > v ? 0 : 1;
        }
        const double v = static_cast<double>(c.vulnerability_a) * (static_cast<double>(ac) + x) +
                         static_cast<double>(c.vulnerability_b);
        return static_cast<double>(damage) + y > v ? 0 : 2;
    }

    void redistribute(int region_index, int step) {
        const auto& c = *s_.climate;
        if (!c.has_matrix) return;
        const int rid = c.region_ids[region_index];
        int from = -1;
        for (std::size_t i = 0; i < c.from_ids.size(); ++i)
            if (c.from_ids[i] == rid) from = static_cast<int>(i);
        if (from < 0) return;  // not in the matrix: the module warns and moves nothing
        const int dim_to = static_cast<int>(c.to_ids.size());
        const double p = rng_.next_double();
        float cumulative = 0;
        int to = dim_to - 1;
        for (int i = 0; i < dim_to; ++i) {
            cumulative += c.matrix[from][i];
            if (p <= cumulative) {
                to = i;
                break;
            }
        }
        const int to_id = c.to_ids[to];
        int to_index = -1;
        for (std::size_t i = 0; i < c.region_ids.size(); ++i)
            if (c.region_ids[i] == to_id) to_index = static_cast<int>(i);
        float& moved = moved_[static_cast<std::size_t>(from) * dim_to + to];
        if (to_index < 0) {  // outside the simulation: the same density
            moved += 1;
            return;
        }
        const std::size_t ns = static_cast<std::size_t>(s_.nsteps);
        const float dfrom = c.population[region_index * ns + step] / demand_[region_index * ns + step];
        const float dto = c.population[to_index * ns + step] / demand_[to_index * ns + step];
        float to_px;
        if (to_index == region_index) to_px = 1;
        else if (dfrom <= 0 || dto <= 0 || !std::isfinite(dfrom) || !std::isfinite(dto)) to_px = 1;
        else to_px = 1 * dfrom / dto;
        if (step + 1 < s_.nsteps) {
            demand_[to_index * ns + step + 1] += to_px;
            moved += to_px;
        }
    }

    float max_hand(const std::array<int, 4>& b, float p) {
        const auto& c = *s_.climate;
        hand_.clear();
        for (int row = b[0]; row <= b[1]; ++row) {
            for (int col = b[3]; col <= b[2]; ++col) {  // the module walks only column w (corrected)
                const std::int64_t k = static_cast<std::int64_t>(row) * s_.cols + col;
                const float f = current_prob_ ? (*current_prob_)[k] : std::nanf("");
                if (f >= p && !std::isnan(c.hand[k])) hand_.push_back(c.hand[k]);
            }
        }
        if (hand_.empty()) return 0;
        const int perc = static_cast<int>(static_cast<double>(hand_.size()) * 1e-2 *
                                          static_cast<int>(c.percentile) - 0.5);
        std::sort(hand_.begin(), hand_.end());
        return hand_[perc];
    }

    float depth_at(std::int64_t k, float p) const {
        const auto& c = *s_.climate;
        const auto& d = *current_depths_;
        const int n = static_cast<int>(c.probabilities.size());
        if (std::isnan(d[0][k])) return 0;
        for (int i = n - 1; i >= 0; --i) {
            if (p > c.probabilities[i]) return std::isnan(d[i][k]) ? 0.0f : d[i][k];
        }
        return d[0][k];
    }

    void climate_step(int step, int h, FuturesResult& out) {
        const auto& c = *s_.climate;
        if (!(max_flood_[h] > 0)) return;
        const double draw = rng_.next_double();
        if (!(draw <= max_flood_[h])) return;
        const float p = static_cast<float>(draw);
        const auto& b = bbox_[h];
        if (b[0] < 0) return;
        float level = 0;
        if (!c.depth_mode) {
            level = max_hand(b, p);
            if (level == 0) return;
        }
        out.flood_steps.push_back(step);
        out.flood_hucs.push_back(c.huc_ids[h]);
        out.flood_probabilities.push_back(p);
        for (int row = b[0]; row <= b[1]; ++row) {
            for (int col = b[3]; col <= b[2]; ++col) {
                const std::int64_t k = static_cast<std::int64_t>(row) * s_.cols + col;
                if (dev_[k] == kFuturesNull || c.huc[k] != h) continue;
                float depth;
                if (c.depth_mode) {
                    depth = depth_at(k, p);
                } else {
                    depth = level - c.hand[k];
                    depth = depth > 0 ? depth : 0;
                }
                float damage = 0;
                if (depth > 0 && !is_adapted(p, k)) damage = depth_to_damage(depth, c.ddf_region[k]);
                if (damage > 0 && dev_[k] >= 0) {
                    const int resp = response(damage, c.adaptive_capacity[k]);
                    if (resp == 0) {
                        dev_[k] = -step - 2;
                        redistribute(region_[k], step);
                    } else if (resp == 1) {
                        adapt(p, k);
                    } else {
                        adaptation_[k] = 0;
                    }
                }
            }
        }
    }

    const FuturesSettings& s_;
    std::int64_t ncell_;
    std::vector<int> dev_;
    const int* region_;
    const int* pot_;
    std::vector<float> pressure_;
    const float* agg_;
    const float* weight_;
    const int* zone_;
    std::vector<float> prob_;
    std::vector<float> matrix_;
    Drand48 rng_;
    std::vector<std::vector<Developable>> cells_;
    std::vector<Candidate> cand_;
    std::vector<int> added_;
    std::vector<float> demand_;
    // climate
    std::vector<int> adaptation_;
    std::vector<float> max_flood_;
    std::vector<std::array<int, 4>> bbox_;
    std::vector<float> moved_;
    std::vector<float> hand_;
    const std::vector<std::vector<float>>* current_depths_ = nullptr;
    const std::vector<float>* current_prob_ = nullptr;
};

}  // namespace

FuturesResult futures_run(const FuturesSettings& s, const int* developed, const int* region, const int* pot_region,
                          const float* devpressure, const float* aggregated, const float* weight, const int* zone) {
    if (s.nregions < 1) throw std::invalid_argument("at least one subregion");
    if (static_cast<int>(s.intercept.size()) < s.npot || static_cast<int>(s.devpressure_coef.size()) < s.npot)
        throw std::invalid_argument("one intercept and pressure coefficient per potential subregion");
    if (s.patch_sizes.empty()) throw std::invalid_argument("patch sizes needed");
    for (const auto& v : s.patch_sizes)
        if (v.empty()) throw std::invalid_argument("every subregion needs at least one patch size");
    if (static_cast<std::int64_t>(s.demand.size()) != static_cast<std::int64_t>(s.nregions) * s.nsteps)
        throw std::invalid_argument("demand must be (subregions, steps)");
    Futures f(s, developed, region, pot_region, devpressure, aggregated, weight, zone);
    return f.run();
}

std::vector<float> futures_mfilter(const double* input, std::int64_t rows, std::int64_t cols, const double* matrix,
                                   int size) {
    const int side = 2 * size + 1;
    std::vector<float> out(static_cast<std::size_t>(rows * cols));
    lsm::parallel_for(rows * cols * side * side >= (1 << 18), rows, [&](std::int64_t r) {
        for (std::int64_t c = 0; c < cols; ++c) {
            const std::int64_t k = r * cols + c;
            if (r < size || r >= rows - size || c < size || c >= cols - size) {
                out[k] = static_cast<float>(input[k]);
                continue;
            }
            double v = 0;
            bool null = false;
            for (int a = 0; a < side && !null; ++a) {
                for (int b = 0; b < side; ++b) {
                    const double x = input[(r - size + a) * cols + (c - size + b)];
                    if (std::isnan(x)) {
                        null = true;
                        break;
                    }
                    v += x * matrix[a * side + b];
                }
            }
            out[k] = null ? std::nanf("") : static_cast<float>(v / 1.0);
        }
    });
    return out;
}

}  // namespace sim
