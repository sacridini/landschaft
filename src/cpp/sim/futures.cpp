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
          rng_(s.seed), cells_(s.nregions) {
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
            out.series.push_back(dev_);
        }
        out.developed = dev_;
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
        int n_to_convert = static_cast<int>(std::lround(s_.demand[static_cast<std::size_t>(reg) * s_.nsteps + step]));
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
