// SPDX-License-Identifier: GPL-3.0-or-later AND MIT
//
// Port of Marxan 4.0.6 (MIT licence, Copyright (c) 2020 Hugh Possingham, Ian
// Ball, Matthew Watts): score_change.cpp, computation.cpp, algorithms.cpp,
// thermal_annealing.cpp, iterative_improvement.cpp, heuristics.cpp and the
// run loop of marxan.cpp, for the features without clumping, separation or
// probability targets (see marxan.hpp).

#include "opt/marxan.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace opt::marxan {

namespace {

constexpr double kDelta = 1e-14;  // Marxan's delta

// std::mt19937 with libstdc++'s distributions (GCC 13), whatever the compiler's library.
struct Rng {
    std::mt19937 g;
    explicit Rng(std::uint32_t seed) : g(seed) {}

    std::uint32_t raw() { return static_cast<std::uint32_t>(g()); }

    // uniform_int_distribution's downscaling of a 32-bit engine: Lemire's nearly divisionless method,
    // a value in [0, range) for 0 < range <= 2^32 - 1
    std::uint32_t below(std::uint32_t range) {
        std::uint64_t product = std::uint64_t(raw()) * std::uint64_t(range);
        std::uint32_t low = static_cast<std::uint32_t>(product);
        if (low < range) {
            const std::uint32_t threshold = static_cast<std::uint32_t>(-range) % range;
            while (low < threshold) {
                product = std::uint64_t(raw()) * std::uint64_t(range);
                low = static_cast<std::uint32_t>(product);
            }
        }
        return static_cast<std::uint32_t>(product >> 32);
    }

    // uniform_int_distribution<T>(a, b)
    std::int64_t uniform_int(std::int64_t a, std::int64_t b) {
        const std::uint64_t urange = static_cast<std::uint64_t>(b) - static_cast<std::uint64_t>(a);
        if (urange < 0xFFFFFFFFull) return a + static_cast<std::int64_t>(below(static_cast<std::uint32_t>(urange + 1)));
        if (urange == 0xFFFFFFFFull) return a + static_cast<std::int64_t>(raw());
        throw std::logic_error("ranges wider than 32 bits are not used by Marxan");
    }

    // uniform_real_distribution<double>(0, 1): generate_canonical<double, 53> on a 32-bit engine
    double uniform01() {
        double sum = 0.0, tmp = 1.0;
        for (int k = 0; k < 2; ++k) {
            sum += static_cast<double>(raw()) * tmp;
            tmp *= 4294967296.0;
        }
        double ret = sum / tmp;
        if (ret >= 1.0) ret = std::nextafter(1.0, 0.0);
        return ret * (1.0 - 0.0) + 0.0;
    }

    // std::shuffle (libstdc++): two swap positions per draw when the range allows it
    void shuffle(std::vector<int>& v) {
        const std::uint64_t n = v.size();
        if (n == 0) return;
        const std::uint64_t urngrange = 0xFFFFFFFFull;
        if (urngrange / n >= n) {
            std::uint64_t i = 1;
            if (n % 2 == 0) {
                std::swap(v[i], v[below(2)]);
                ++i;
            }
            while (i != n) {
                const std::uint64_t swap_range = i + 1;
                const std::uint64_t x = below(static_cast<std::uint32_t>(swap_range * (swap_range + 1)));
                const std::uint64_t p1 = x / (swap_range + 1), p2 = x % (swap_range + 1);
                std::swap(v[i], v[p1]);
                ++i;
                std::swap(v[i], v[p2]);
                ++i;
            }
            return;
        }
        for (std::uint64_t i = 1; i != n; ++i) std::swap(v[i], v[below(static_cast<std::uint32_t>(i + 1))]);
    }
};

struct Species {
    double target, prop, spf, penalty = 0.0, amount = 0.0;
    int targetocc, occurrence = 0;
};

struct Model {
    const Data& d;
    double cm;
    int connectivity_in;

    double amount_at(int ipu, int isp) const {  // returnAmountSpecAtPu
        const auto b = d.spindex.begin() + d.offset[ipu], e = d.spindex.begin() + d.offset[ipu + 1];
        const auto it = std::lower_bound(b, e, isp);
        if (it != e && *it == isp) return d.amount[static_cast<std::size_t>(it - d.spindex.begin())];
        return 0.0;
    }

    double connection_cost1(int ipu) const {
        double fcost = d.fixedcost[ipu];
        for (std::int64_t k = d.nb_start[ipu]; k < d.nb_start[ipu + 1]; ++k) fcost += d.nb_cost[k];
        return fcost * cm;
    }

    double unit_value(int ipu) const {  // computePlanningUnitValue
        double v = d.cost[ipu];
        v += connection_cost1(ipu);
        return v;
    }

    // ConnectionCost2, symmetric connectivity
    double connection_cost2(int ipu, const std::vector<int>& R, int imode, int imode2, double cmm) const {
        double fcost = d.fixedcost[ipu] * imode;
        for (std::int64_t k = d.nb_start[ipu]; k < d.nb_start[ipu + 1]; ++k) {
            const int nbr = d.nb[k];
            const double c = d.nb_cost[k];
            const bool in = R[nbr] == 1 || R[nbr] == 2;
            if (connectivity_in == 1) {
                if (in) fcost += imode * c;
                else fcost += imode * imode2 * c * -1;
            } else {
                if (in) fcost += imode * imode2 * c * -1;
                else fcost += imode * c;
            }
        }
        return fcost * cmm;
    }
};

void initialise_reserve(double prop, const Data& d, std::vector<int>& R, Rng& rng) {
    for (int i = 0; i < d.npu; ++i)
        if (d.status[i]) R[i] = d.status[i];
        else R[i] = rng.uniform01() < prop ? 1 : 0;
}

void species_amounts(const Data& d, std::vector<Species>& spec, const std::vector<int>& R) {
    for (auto& s : spec) {
        s.amount = 0;
        s.occurrence = 0;
    }
    for (int ipu = 0; ipu < d.npu; ++ipu)
        if (R[ipu] == 1 || R[ipu] == 2)
            for (std::int64_t ism = d.offset[ipu]; ism < d.offset[ipu + 1]; ++ism) {
                const int isp = d.spindex[ism];
                spec[isp].amount += d.amount[ism];
                spec[isp].occurrence++;
            }
}

void reserve_value(const Model& m, const std::vector<int>& R, const std::vector<Species>& spec, Score& reserve) {
    const Data& d = m.d;
    reserve.pus = 0;
    reserve.cost = 0;
    reserve.penalty = 0;
    reserve.connection = 0;
    reserve.shortfall = 0;
    for (const Species& s : spec) {
        double fractionAmount = 0;
        if (s.target > s.amount) {
            if (s.target != 0) fractionAmount = (s.target - s.amount) / s.target;
            reserve.shortfall += s.target - s.amount;
        }
        if (s.targetocc > 0) {
            if (s.targetocc > s.occurrence) {
                fractionAmount += (double)(s.targetocc - s.occurrence) / (double)s.targetocc;
                reserve.shortfall += s.targetocc - s.occurrence;
            }
            if (s.target && s.targetocc) fractionAmount /= 2;
        }
        reserve.penalty += fractionAmount * s.penalty * s.spf;
    }
    for (int j = 0; j < d.npu; ++j)
        if (R[j] == 1 || R[j] == 2) {
            reserve.cost += d.cost[j];
            reserve.pus += 1;
            reserve.connection += m.connection_cost2(j, R, 1, 0, m.cm);
        }
    reserve.total = reserve.cost + reserve.connection + reserve.penalty + 0.0 + 0.0;
}

double change_penalty(const Model& m, int ipu, const std::vector<Species>& spec, int imode, double& rShortfall) {
    const Data& d = m.d;
    rShortfall = 0;
    double penalty = 0;
    for (std::int64_t ism = d.offset[ipu]; ism < d.offset[ipu + 1]; ++ism) {
        const int isp = d.spindex[ism];
        const Species& s = spec[isp];
        if (d.amount[ism] == 0.0) continue;
        double fractionAmount = 0, newamount = 0, rOldShortfall = 0, rNewShortfall = 0;
        if (s.target > s.amount && s.target != 0) {
            fractionAmount = (s.target - s.amount) / s.target;
            rOldShortfall = s.target - s.amount;
        }
        const double rNewAmountHeld = s.amount + (d.amount[ism] * imode);
        if (s.target > rNewAmountHeld) rNewShortfall = s.target - rNewAmountHeld;
        rShortfall += rNewShortfall - rOldShortfall;
        if (s.targetocc > 0) {
            if (s.targetocc > s.occurrence)
                fractionAmount += ((double)s.targetocc - (double)s.occurrence) / (double)s.targetocc;
            if (s.target && s.targetocc) fractionAmount /= 2;
        }
        if (s.target) newamount = std::max(0.0, s.target - s.amount - d.amount[ism] * imode) / s.target;
        if (s.targetocc) {
            const double tamount = (double)(s.targetocc - s.occurrence - imode) / (double)s.targetocc;
            newamount += tamount < 0 ? 0 : tamount;
        }
        if (s.target && s.targetocc) newamount /= 2;
        penalty += s.penalty * s.spf * (newamount - fractionAmount);
    }
    return penalty;
}

double threshold_penalty(double tpf1, double tpf2, double timeprop) {
    if (tpf2 < 0) return tpf1;
    return tpf1 * std::exp(tpf2 * timeprop);
}

void change_score(const Model& m, int ipu, const std::vector<int>& R, const std::vector<Species>& spec, int imode,
                  Score& change, const Score& reserve, double costthresh, double tpf1, double tpf2, double timeprop) {
    double threshpen = 0;
    change.cost = m.d.cost[ipu] * imode;
    change.connection = m.connection_cost2(ipu, R, imode, 1, m.cm);
    change.penalty = change_penalty(m, ipu, spec, imode, change.shortfall);
    if (costthresh) {
        if (reserve.cost + reserve.connection <= costthresh) {
            if (change.cost + change.connection + reserve.cost + reserve.connection <= costthresh) threshpen = 0;
            else
                threshpen = (change.cost + change.connection + reserve.cost + reserve.connection - costthresh) *
                            threshold_penalty(tpf1, tpf2, timeprop);
        } else {
            if (change.cost + change.connection + reserve.cost + reserve.connection <= costthresh)
                threshpen = (reserve.cost + reserve.connection - costthresh) * threshold_penalty(tpf1, tpf2, timeprop);
            else
                threshpen = (change.cost + change.connection) * threshold_penalty(tpf1, tpf2, timeprop);
        }
    }
    change.threshpen = threshpen;
    change.total = change.cost + change.connection + change.penalty + change.threshpen + 0.0 + 0.0;
}

void do_change(const Model& m, int ipu, std::vector<int>& R, Score& reserve, const Score& change,
               std::vector<Species>& spec, int imode) {
    const Data& d = m.d;
    R[ipu] = imode == 1 ? 1 : 0;
    reserve.pus += imode;
    reserve.cost += change.cost;
    reserve.connection += change.connection;
    reserve.penalty += change.penalty;
    reserve.shortfall += change.shortfall;
    for (std::int64_t ism = d.offset[ipu]; ism < d.offset[ipu + 1]; ++ism) {
        Species& s = spec[d.spindex[ism]];
        const double rAmount = d.amount[ism];
        s.occurrence += (rAmount > 0) * imode;
        s.amount += rAmount * imode;
        if (s.amount < 0.0001)
            if (s.amount > -0.0001) s.amount = 0;
    }
    reserve.total = reserve.cost + reserve.connection + reserve.penalty + 0.0 + 0.0;
}

// computePenalties: every feature's penalty, the cost of a greedy cover of its target
int compute_penalties(const Model& m, std::vector<Species>& spec, Rng& rng) {
    const Data& d = m.d;
    std::vector<int> put(static_cast<std::size_t>(d.npu));
    initialise_reserve(0, d, put, rng);
    int badspecies = 0;
    for (int i = 0; i < d.nsp; ++i) {
        Species& s = spec[i];
        double ftarget = 0, penalty = 0;
        int itargetocc = 0;
        for (int j = 0; j < d.npu; ++j)
            if (put[j] == 2) {
                ftarget += m.amount_at(j, i);
                itargetocc++;
                penalty += m.unit_value(j);
            }
        s.penalty = penalty;
        if (ftarget >= s.target && itargetocc >= s.targetocc) continue;  // met by the locked-in units
        for (int j = 0; j < d.npu; ++j)
            if (put[j] < 2) put[j] = 0;
        double fbest;
        do {
            fbest = 0;
            int imaxtarget = 0, ibest = 0;
            double fbestrat = 0, rAmountBest = 0;
            for (int j = 0; j < d.npu; ++j) {
                if (put[j] != 0) continue;
                const double rAmount = m.amount_at(j, i);
                if (!(rAmount > 0)) continue;
                double fcost = m.unit_value(j);
                if (fcost == 0) fcost = kDelta;
                if (rAmount >= s.target - ftarget && (imaxtarget == 0 || (imaxtarget == 1 && fcost < fbest))) {
                    imaxtarget = 1;
                    ibest = j;
                    fbest = fcost;
                    rAmountBest = rAmount;
                } else if (fbestrat < rAmount / fcost) {
                    fbest = fcost;
                    fbestrat = rAmount / fbest;
                    ibest = j;
                    rAmountBest = rAmount;
                }
            }
            if (fbest > 0) {
                put[ibest] = 1;
                ftarget += rAmountBest;
                itargetocc++;
                s.penalty += fbest;
            }
        } while ((ftarget < s.target || itargetocc < s.targetocc) && fbest > 0);
        if (fbest == 0) {  // the target cannot be met with every available unit
            if (ftarget == 0) ftarget = kDelta;
            double ftemp = 0;
            if (ftarget < s.target) ftemp = s.target / ftarget;
            if (itargetocc < s.targetocc && itargetocc) ftemp += (float)s.targetocc / (float)itargetocc;
            s.penalty = s.penalty * ftemp;
            badspecies++;
        }
    }
    return badspecies;
}

// the units of every feature (the transpose of the units' features)
struct ByFeature {
    std::vector<std::int64_t> start;
    std::vector<int> unit;
    std::vector<double> amount;
};

ByFeature by_feature(const Data& d) {
    ByFeature f;
    f.start.assign(static_cast<std::size_t>(d.nsp) + 1, 0);
    for (int isp : d.spindex) f.start[static_cast<std::size_t>(isp) + 1]++;
    for (int i = 0; i < d.nsp; ++i) f.start[i + 1] += f.start[i];
    f.unit.resize(d.spindex.size());
    f.amount.resize(d.spindex.size());
    std::vector<std::int64_t> at(f.start.begin(), f.start.end() - 1);
    for (int ipu = 0; ipu < d.npu; ++ipu)
        for (std::int64_t ism = d.offset[ipu]; ism < d.offset[ipu + 1]; ++ism) {
            const std::int64_t k = at[d.spindex[ism]]++;
            f.unit[k] = ipu;
            f.amount[k] = d.amount[ism];
        }
    return f;
}

// landschaft's penalties (Options::fast_penalties): the locked-in units' amounts first, then the available
// units in order of amount per unit value (Marxan's computePlanningUnitValue) until the target is met; the
// penalty is the value of the units added. A target that cannot be met scales it as Marxan does.
int fast_penalties(const Model& m, const ByFeature& f, std::vector<Species>& spec, int threads) {
    const Data& d = m.d;
    std::vector<double> value(static_cast<std::size_t>(d.npu));
    for (int j = 0; j < d.npu; ++j) {
        value[j] = m.unit_value(j);
        if (!(value[j] > 0)) value[j] = kDelta;
    }
    int bad = 0;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads > 0 ? threads : 1) reduction(+ : bad)
    for (int i = 0; i < d.nsp; ++i) {
        Species& s = spec[i];
        double held = 0;
        std::vector<std::int64_t> cand;
        for (std::int64_t k = f.start[i]; k < f.start[i + 1]; ++k) {
            const int st = d.status[f.unit[k]];
            if (st == 2) held += f.amount[k];
            else if (st != 3 && f.amount[k] > 0) cand.push_back(k);
        }
        s.penalty = 0;
        if (held >= s.target) continue;
        std::sort(cand.begin(), cand.end(), [&](std::int64_t a, std::int64_t b) {
            const double ra = f.amount[a] / value[f.unit[a]], rb = f.amount[b] / value[f.unit[b]];
            return ra > rb || (ra == rb && a < b);
        });
        for (const std::int64_t k : cand) {
            if (held >= s.target) break;
            held += f.amount[k];
            s.penalty += value[f.unit[k]];
        }
        if (held < s.target) {
            s.penalty *= s.target / (held == 0 ? kDelta : held);
            bad++;
        }
    }
    return bad;
}

double target_tolerance(const Species& s) { return 1e-10 * std::max(1.0, std::abs(s.target)); }

// Options::repair: every target met (if it can be), then no unit that the targets do not need and the score
// is better without
void repair(const Model& m, const ByFeature& f, std::vector<int>& R, std::vector<Species>& spec, Score& reserve) {
    const Data& d = m.d;
    species_amounts(d, spec, R);  // afresh: the annealing's running sums drift
    reserve_value(m, R, spec, reserve);
    Score change;
    std::vector<int> stamp(static_cast<std::size_t>(d.npu), -1);
    std::vector<int> cand;
    for (int round = 0;; ++round) {
        cand.clear();
        for (int i = 0; i < d.nsp; ++i) {
            const Species& s = spec[i];
            if (!(s.target - s.amount > target_tolerance(s))) continue;
            for (std::int64_t k = f.start[i]; k < f.start[i + 1]; ++k) {
                const int j = f.unit[k];
                if (R[j] == 0 && f.amount[k] > 0 && stamp[j] != round) {
                    stamp[j] = round;
                    cand.push_back(j);
                }
            }
        }
        if (cand.empty()) break;  // met, or nothing left to add
        int best = -1;
        bool best_free = false;
        double best_key = 0;
        for (const int j : cand) {
            double gain = 0;
            for (std::int64_t ism = d.offset[j]; ism < d.offset[j + 1]; ++ism) {
                const Species& s = spec[d.spindex[ism]];
                const double shortfall = s.target - s.amount;
                if (shortfall > target_tolerance(s) && d.amount[ism] > 0)
                    gain += std::min(d.amount[ism], shortfall) / s.target;
            }
            if (!(gain > 0)) continue;
            const double delta = d.cost[j] + m.connection_cost2(j, R, 1, 1, m.cm);
            const bool free = delta <= 0;
            const double key = free ? gain : gain / delta;
            if (best < 0 || (free && !best_free) || (free == best_free && key > best_key)) {
                best = j;
                best_free = free;
                best_key = key;
            }
        }
        if (best < 0) break;
        change_score(m, best, R, spec, 1, change, reserve, 0, 0, 0, 0);
        do_change(m, best, R, reserve, change, spec, 1);
    }
    // drop units in order of what dropping them saves, while the targets stay met
    for (bool dropped = true; dropped;) {
        dropped = false;
        std::vector<std::pair<double, int>> order;
        for (int j = 0; j < d.npu; ++j)
            if (R[j] == 1) {
                const double delta = -d.cost[j] + m.connection_cost2(j, R, -1, 1, m.cm);
                if (delta < 0) order.emplace_back(delta, j);
            }
        std::sort(order.begin(), order.end());
        for (const auto& [saving, j] : order) {
            bool keeps = true;
            for (std::int64_t ism = d.offset[j]; ism < d.offset[j + 1] && keeps; ++ism) {
                const Species& s = spec[d.spindex[ism]];
                if (d.amount[ism] > 0 && s.amount - d.amount[ism] < s.target - target_tolerance(s)) keeps = false;
            }
            if (!keeps) continue;
            if (-d.cost[j] + m.connection_cost2(j, R, -1, 1, m.cm) >= 0) continue;  // its neighbours changed
            change_score(m, j, R, spec, -1, change, reserve, 0, 0, 0, 0);
            do_change(m, j, R, reserve, change, spec, -1);
            dropped = true;
        }
    }
    species_amounts(d, spec, R);
    reserve_value(m, R, spec, reserve);
}

struct Anneal {
    long titns;
    long long iterations;
    long tlen;
    double tinit, tcool, temp = 0, tempold = 0, sigma = 0, sum = 0, sum2 = 0;
    int type;
};

void connolly(const Model& m, std::vector<int>& R, std::vector<Species>& spec, Anneal& a, double prop, Rng& rng) {
    const Data& d = m.d;
    double deltamin = 0, deltamax = 0;
    const double localdelta = 1E-10;
    Score change, reserve;
    initialise_reserve(prop, d, R, rng);
    species_amounts(d, spec, R);  // the correction: the amounts of this reserve
    reserve_value(m, R, spec, reserve);
    for (long long i = 1; i <= a.iterations / 100; ++i) {
        const int ipu = static_cast<int>(rng.uniform_int(0, d.npu - 1));
        const int imode = R[ipu] == 1 ? -1 : 1;
        change_score(m, ipu, R, spec, imode, change, reserve, 0, 0, 0, 0);
        do_change(m, ipu, R, reserve, change, spec, imode);
        if (change.total > deltamax) deltamax = change.total;
        if (change.total > localdelta && (deltamin < localdelta || change.total < deltamin)) deltamin = change.total;
    }
    a.tinit = deltamax;
    deltamin *= 0.1;
    a.tcool = std::exp(std::log(deltamin / a.tinit) / (double)a.titns);
}

void adaptive(const Model& m, std::vector<int>& R, std::vector<Species>& spec, Anneal& a, double prop, Rng& rng) {
    const long isamples = 1000;
    double sum = 0, sum2 = 0;
    Score cost;
    for (long i = 0; i < isamples; ++i) {
        initialise_reserve(prop, m.d, R, rng);
        species_amounts(m.d, spec, R);  // the correction: the amounts of this reserve
        reserve_value(m, R, spec, cost);
        sum += cost.total;
        sum2 += cost.total * cost.total;
    }
    const double sigma = std::sqrt(sum2 - std::pow(sum / isamples, 2)) / (isamples - 1);
    a.tinit = 10 * sigma;
    a.sigma = sigma;
    a.temp = a.tinit;
    a.tempold = a.temp;
    a.sum = 0;
    a.sum2 = 0;
}

void reduce_temperature(Anneal& a) {
    const double omega = 0.7, lambda = 0.7;
    const double sigmanew = (a.sum2 - std::pow((a.sum / a.tlen), 2)) / (a.tlen - 1);
    const double sigmamod = (1 - omega) * sigmanew + omega * a.sigma * (a.temp / a.tempold);
    a.tempold = a.temp;
    a.temp = std::exp(-lambda * a.temp / sigmamod);
    a.sigma = sigmamod;
    a.sum = 0;
    a.sum2 = 0;
}

using Clock = std::chrono::steady_clock;

struct Deadline {
    bool timed = false;
    Clock::time_point at{};
    std::atomic<bool>* hit = nullptr;
    bool passed() const {
        if (!timed) return false;
        if (Clock::now() < at) return false;
        hit->store(true);
        return true;
    }
};

// run_seconds > 0 (landschaft's, Options::time_schedule): the schedule's temperature decreases spread over
// the run's seconds instead of its iterations, as many iterations as fit
void thermal_annealing(const Model& m, std::vector<int>& R, std::vector<Species>& spec, Score& reserve, Anneal& a,
                       double costthresh, const Options& o, Rng& rng, const Deadline& deadline, double run_seconds) {
    const Data& d = m.d;
    Score change;
    const double rThreshold = costthresh;
    costthresh = rThreshold * o.start_dec_mult;
    double rTemperature = 1;
    const bool timed = run_seconds > 0;
    const Clock::time_point t0 = Clock::now();
    long step = 0, target_step = 0;
    long long since = 0;
    double frac = 0;
    for (long long itime = 1; timed || itime <= a.iterations; ++itime) {
        if ((itime & 1023) == 0) {
            if (deadline.passed()) break;
            if (timed) {
                frac = std::chrono::duration<double>(Clock::now() - t0).count() / run_seconds;
                if (frac >= 1) break;
                target_step = static_cast<long>(frac * static_cast<double>(a.titns));
            }
        }
        int ipu = static_cast<int>(rng.uniform_int(0, d.npu - 1));
        while (R[ipu] > 1) ipu = static_cast<int>(rng.uniform_int(0, d.npu - 1));
        const int itemp = R[ipu] == 1 ? -1 : 1;
        change_score(m, ipu, R, spec, itemp, change, reserve, costthresh, o.tpf1, o.tpf2,
                     timed ? frac : (double)itime / (double)a.iterations);
        ++since;
        if (timed ? step < target_step : itime % a.tlen == 0) {
            if (timed) {
                ++step;
                a.tlen = static_cast<long>(std::max<long long>(2, since));
                since = 0;
            }
            rTemperature = rTemperature * a.tcool;
            if (rTemperature > o.start_dec_thresh) costthresh = rThreshold * o.start_dec_mult;
            else if (rTemperature < o.end_dec_thresh) costthresh = rThreshold * o.end_dec_mult;
            else {
                const double mult = (rTemperature - o.end_dec_thresh) / (o.start_dec_thresh - o.end_dec_thresh);
                costthresh = (o.end_dec_mult + (mult * (o.start_dec_mult - o.end_dec_mult))) * rThreshold;
            }
            if (a.type == 3) reduce_temperature(a);
            else a.temp = a.temp * a.tcool;
        }
        bool good;
        if (change.total <= 0) good = true;
        else good = std::exp(-change.total / a.temp) > rng.uniform01();
        if (good) do_change(m, ipu, R, reserve, change, spec, itemp);
        if (a.type == 3) {
            a.sum += reserve.total;
            a.sum2 += reserve.total * reserve.total;
        }
    }
}

void iterative_improvement(const Model& m, std::vector<int>& R, std::vector<Species>& spec, Score& reserve,
                           double costthresh, const Options& o, Rng& rng) {
    const Data& d = m.d;
    std::vector<int> order;
    for (int i = 0; i < d.npu; ++i)
        if (R[i] < 2 && d.status[i] < 2) order.push_back(i);
    if (order.empty()) return;
    rng.shuffle(order);
    Score change;
    for (const int ichoice : order) {
        if (R[ichoice] < 2 && d.status[ichoice] < 2) {
            const int imode = R[ichoice] == 1 ? -1 : 1;
            change_score(m, ichoice, R, spec, imode, change, reserve, costthresh, o.tpf1, o.tpf2, 1);
            if (change.total < 0) do_change(m, ichoice, R, reserve, change, spec, imode);
        }
    }
}

double greedy_score(const Model& m, int ipu, const std::vector<int>& R, const std::vector<Species>& spec) {
    double famount = 0.0;
    for (int i = 0; i < m.d.nsp; ++i) {
        const Species& s = spec[i];
        const double fold = (s.target - s.amount);
        if (fold > 0) {
            const double newamount = std::max(0.0, s.target - s.amount - m.amount_at(ipu, i) * 1);
            famount += (newamount - fold) * s.spf;
        }
    }
    const double currcost = m.d.cost[ipu] + m.connection_cost2(ipu, R, 1, 1, m.cm);
    if (currcost <= 0) return -1.0 / kDelta;
    return famount / currcost;
}

void heuristics(const Model& m, std::vector<int>& R, std::vector<Species>& spec, Score& reserve, double costthresh,
                const Options& o, Rng& rng) {
    const Data& d = m.d;
    Score change;
    double bestscore;
    do {
        int bestpu = 0;
        bestscore = 0;
        for (int i = 0; i < d.npu; ++i) {
            if (R[i]) continue;
            double currscore = 1;
            if (o.heurtype == 0) currscore = greedy_score(m, i, R, spec);
            if (o.heurtype == 1) {
                change_score(m, i, R, spec, 1, change, reserve, costthresh, o.tpf1, o.tpf2, 1);
                currscore = change.total;
            }
            currscore *= rng.uniform01() * 0.001 + 1.0;
            if (!costthresh || d.cost[i] + reserve.cost <= costthresh)
                if (currscore < bestscore) {
                    bestpu = i;
                    bestscore = currscore;
                }
        }
        if (bestscore) {
            change_score(m, bestpu, R, spec, 1, change, reserve, costthresh, o.tpf1, o.tpf2, 1);
            do_change(m, bestpu, R, reserve, change, spec, 1);
        }
    } while (bestscore);
    reserve.total = reserve.cost + reserve.connection + reserve.penalty + 0.0 + 0.0;
}

}  // namespace

Result solve(const Data& d, const Options& o) {
    if (d.npu <= 0) throw std::invalid_argument("no planning units");
    if (o.runmode != 0 && o.runmode != 1 && o.runmode != 2 && o.runmode != 6)
        throw std::invalid_argument("run modes 0, 1, 2 and 6 are supported (the others need user penalties)");
    if (o.heurtype != 0 && o.heurtype != 1) throw std::invalid_argument("heuristics 0 and 1 are supported");
    const bool heuristic_on = o.runmode == 0 || o.runmode == 2;
    const bool itimp_on = o.runmode == 1 || o.runmode == 2;
    const Model m{d, o.blm, o.connectivity_in};

    std::vector<Species> spec(static_cast<std::size_t>(d.nsp));
    for (int i = 0; i < d.nsp; ++i) {
        Species& s = spec[i];
        s.target = std::max(0.0, d.target[i]);
        s.prop = d.prop[i];
        s.spf = d.spf[i] < 0 ? 1 : d.spf[i];
        s.targetocc = std::max(0, d.targetocc[i]);
    }
    for (int i = 0; i < d.nsp; ++i)  // computeSpecProp
        if (spec[i].prop > 0) {
            double total = 0.0;
            for (int ipu = 0; ipu < d.npu; ++ipu) total += m.amount_at(ipu, i);
            spec[i].target = total * spec[i].prop;
        }

    Rng global(o.seed);
    Result res;
    ByFeature f;
    if (o.fast_penalties || o.repair) f = by_feature(d);
    res.unmet = o.fast_penalties ? fast_penalties(m, f, spec, o.threads) : compute_penalties(m, spec, global);
    for (const Species& s : spec) {
        res.penalty.push_back(s.penalty);
        res.target.push_back(s.target);
    }

    Anneal a0{};
    a0.iterations = o.iterations;
    a0.tinit = o.tinit;
    a0.tcool = o.tcool;
    a0.titns = o.titns;
    a0.type = 1;
    if (a0.iterations < 1) a0.type = 0;
    if (a0.tinit < 0) a0.type = (int)(-a0.tinit) + 1;
    if (a0.type > 3) throw std::invalid_argument("annealing types 1 (fixed), 2 (Connolly) and 3 (adaptive) are supported");
    if (a0.type != 0) {
        if (!a0.titns) throw std::invalid_argument("the number of temperature decreases must not be zero");
        a0.tlen = static_cast<long>(a0.iterations / a0.titns);
        if (a0.tlen < 1) throw std::invalid_argument("fewer iterations than temperature decreases");
    }

    std::atomic<bool> hit{false};
    Deadline deadline;
    if (o.time_limit >= 0) {
        deadline.timed = true;
        deadline.at = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(o.time_limit));
        deadline.hit = &hit;
    }
    double run_seconds = 0;  // Options::time_schedule: the runs in waves of `threads`, each its share
    if (o.time_schedule && deadline.timed) {
        const long threads = std::max(1, o.threads);
        const long waves = (o.repeats + threads - 1) / threads;
        const double left = std::chrono::duration<double>(deadline.at - Clock::now()).count();
        run_seconds = std::max(1e-3, 0.85 * left / static_cast<double>(std::max(1L, waves)));
    }
    const bool any_free = std::any_of(d.status.begin(), d.status.end(), [](int st) { return st < 2; });
    std::vector<std::uint32_t> seeds(static_cast<std::size_t>(o.repeats));
    for (auto& s : seeds) s = global.raw();
    res.runs.resize(static_cast<std::size_t>(o.repeats));
#pragma omp parallel for schedule(dynamic, 1) num_threads(o.threads > 0 ? o.threads : 1)
    for (long r = 0; r < o.repeats; ++r) {
        Rng rng(seeds[static_cast<std::size_t>(r)]);
        Anneal a = a0;
        std::vector<int> R(static_cast<std::size_t>(d.npu));
        std::vector<Species> sp = spec;
        if (!o.spf_scale.empty())
            for (Species& s : sp) s.spf *= o.spf_scale[static_cast<std::size_t>(r) % o.spf_scale.size()];
        Score reserve;
        const bool annealing = a.type != 0 && any_free;  // Marxan's loop never ends with every unit locked
        if (annealing) {
            if (a.type == 2) connolly(m, R, sp, a, o.prop, rng);
            if (a.type == 3) adaptive(m, R, sp, a, o.prop, rng);
            a.temp = a.tinit;
        }
        initialise_reserve(o.prop, d, R, rng);
        species_amounts(d, sp, R);
        reserve_value(m, R, sp, reserve);
        if (annealing) thermal_annealing(m, R, sp, reserve, a, o.costthresh, o, rng, deadline, run_seconds);
        if (heuristic_on) heuristics(m, R, sp, reserve, o.costthresh, o, rng);
        if (itimp_on) {
            iterative_improvement(m, R, sp, reserve, o.costthresh, o, rng);
            if (o.itimptype == 3) iterative_improvement(m, R, sp, reserve, o.costthresh, o, rng);
        }
        if (o.repair) repair(m, f, R, sp, reserve);
        Run& run = res.runs[static_cast<std::size_t>(r)];
        run.R = R;
        run.incremental = reserve;
        reserve_value(m, R, sp, run.score);
        // the summary's figures (computeSummary): misslevel, MPM, shortfall, the boundary length
        double shortfall = 0, mpm = 1;
        int missing = 0;
        for (const Species& s : sp) {
            if (s.target > 0 && s.amount < s.target) {
                shortfall += s.target - s.amount;
                const double p = s.amount / s.target;
                if (p < mpm) mpm = p;
            }
            if (s.targetocc > 0 && s.occurrence < s.targetocc) {
                shortfall += s.targetocc - s.occurrence;
                const double p = (double)s.occurrence / s.targetocc;
                if (p < mpm) mpm = p;
            }
            if (s.target) {
                if (s.amount / s.target < o.misslevel) {
                    missing++;
                    continue;
                }
            }
            if (s.targetocc)
                if ((double)s.occurrence / (double)s.targetocc < o.misslevel) missing++;
        }
        run.shortfall = shortfall;
        run.min_prop_met = mpm;
        run.missing = missing;
        double conn = 0.0;
        for (int i = 0; i < d.npu; ++i)
            if (R[i] == 1 || R[i] == 2) conn += m.connection_cost2(i, R, 1, 0, 1);
        run.connectivity = conn;
    }
    res.timed_out = hit.load();
    res.best = 0;
    for (std::size_t r = 1; r < res.runs.size(); ++r)
        if (res.runs[r].score.total < res.runs[static_cast<std::size_t>(res.best)].score.total) res.best = static_cast<int>(r);
    return res;
}

}  // namespace opt::marxan
