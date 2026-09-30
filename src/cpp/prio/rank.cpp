// SPDX-License-Identifier: GPL-3.0-or-later
//
// Zonation 5's priority ranking (priorityrank.cpp, balancingmethod.h): the cells
// are sorted by their marginal loss; each iteration computes, along the current
// order, what remains of every feature when each cell is removed and the loss of
// the cell at that point, and sorts again (stable, from the cell order), until
// the mean ranking error is below the threshold.
//
// Zonation walks all cells in order with a vector of all layers. Here every
// layer is walked on its own along its occurrences in the current order (the
// remaining level of a layer only changes at its occurrences, and subtracting or
// adding zero does not change a double), in parallel over layers and segments of
// layers; the losses are then summed per cell in layer order. Same bits.

#include "prio/prio.hpp"
#include "prio/vdt.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

// Eigen defines pround<Packet2d> (round half away from zero) only with SSE4.1,
// which Zonation's AVX build has; generic_pow needs it.
#if defined(EIGEN_VECTORIZE_SSE2) && !defined(EIGEN_VECTORIZE_SSE4_1)
namespace Eigen::internal {
template <>
EIGEN_STRONG_INLINE Packet2d pround<Packet2d>(const Packet2d& a) {
    alignas(16) double v[2];
    _mm_store_pd(v, a);
    return _mm_set_pd(std::round(v[1]), std::round(v[0]));
}
}  // namespace Eigen::internal
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace prio {

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Total time of each stage over the iterations, printed to stderr when
// LANDSCHAFT_PROFILE is set.
class Profile {
  public:
    Profile() : on_(std::getenv("LANDSCHAFT_PROFILE") != nullptr), last_(Clock::now()) {}
    void operator()(int stage) {
        if (!on_) return;
        const auto now = Clock::now();
        ms_[stage] += std::chrono::duration<double, std::milli>(now - last_).count();
        last_ = now;
    }
    void mark() { last_ = Clock::now(); }
    ~Profile() {
        if (!on_) return;
        static const char* names[] = {"initial", "scatter", "levels", "losses", "change", "error", "sort", "relayout", "curves"};
        for (int i = 0; i < 9; ++i) std::fprintf(stderr, "[prio] %-10s %10.1f ms\n", names[i], ms_[i]);
    }

  private:
    using Clock = std::chrono::steady_clock;
    bool on_;
    Clock::time_point last_;
    double ms_[9] = {};
};

struct Sum {
    double pos = 0.0, neg = 0.0;
    double value() const { return pos - neg; }
};

// x^y for the two parts of a sum, as Eigen 3.4 computes Array2d::pow (generic_pow on Packet2d).
Sum packet_pow(Sum s, double y) {
    using Eigen::internal::Packet2d;
    alignas(16) double in[2] = {s.pos, s.neg}, yy[2] = {y, y}, out[2];
    const Packet2d r = Eigen::internal::generic_pow(Eigen::internal::pload<Packet2d>(in),
                                                   Eigen::internal::pload<Packet2d>(yy));
    Eigen::internal::pstore(out, r);
    return {out[0], out[1]};
}

// ---------------------------------------------------------------- stable sort

// Order-preserving map of a double to an unsigned integer (-0 as +0).
inline u64 sort_key(double v) {
    if (v == 0.0) v = 0.0;
    u64 b = vdt::double_to_bits(v);
    return (b & 0x8000000000000000ULL) ? ~b : (b | 0x8000000000000000ULL);
}

// A cell to sort: Zonation's stable sort of the cells (from their order) by
// (class, value) is the sort by (class, value, cell), whatever the input order.
struct Item {
    u64 key;
    u32 cls;
    u32 cell;
    u32 from;  // its position in the input (the previous order)
};

inline bool item_less(const Item& a, const Item& b) {
    if (a.cls != b.cls) return a.cls < b.cls;
    if (a.key != b.key) return a.key < b.key;
    return a.cell < b.cell;
}

struct SortBuffers {
    std::vector<Item> a, b;
    std::vector<std::uint16_t> bucket;
    std::vector<i64> counts;
};

// Parallel sample sort of buf.a into `out` (the cells): splitters from a sample,
// the items distributed to their buckets, each bucket sorted. Given the items in
// the previous order, the buckets are nearly sorted already.
void sort_items(SortBuffers& buf, std::vector<u32>& out, std::vector<u32>& from) {
    auto& a = buf.a;
    const i64 n = static_cast<i64>(a.size());
    out.resize(static_cast<size_t>(n));
    from.resize(static_cast<size_t>(n));
    const int T = std::max(1, max_threads());
    if (n < 65536 || T == 1) {
        std::sort(a.begin(), a.end(), item_less);
        for (i64 i = 0; i < n; ++i) {
            out[i] = a[i].cell;
            from[i] = a[i].from;
        }
        return;
    }
    const int NB = std::min(4096, T * 8);  // buckets
    const int per = 32;
    std::vector<Item> sample(static_cast<size_t>(NB) * per);
    const i64 S = static_cast<i64>(sample.size());
    for (i64 k = 0; k < S; ++k) sample[k] = a[static_cast<size_t>((2 * k + 1) * n / (2 * S))];
    std::sort(sample.begin(), sample.end(), item_less);
    std::vector<Item> split(static_cast<size_t>(NB) - 1);
    for (int b = 1; b < NB; ++b) split[b - 1] = sample[static_cast<size_t>(b) * per];
    auto& bucket = buf.bucket;
    bucket.resize(static_cast<size_t>(n));
    const int C = T * 2;  // input chunks
    auto& counts = buf.counts;
    counts.assign(static_cast<size_t>(C) * NB, 0);
#pragma omp parallel for schedule(static, 1)
    for (int c = 0; c < C; ++c) {
        i64* cnt = &counts[static_cast<size_t>(c) * NB];
        for (i64 i = n * c / C; i < n * (c + 1) / C; ++i) {
            const int b = static_cast<int>(std::upper_bound(split.begin(), split.end(), a[i], item_less) - split.begin());
            bucket[i] = static_cast<std::uint16_t>(b);
            ++cnt[b];
        }
    }
    std::vector<i64> bstart(static_cast<size_t>(NB) + 1, 0);
    {
        i64 run = 0;
        for (int b = 0; b < NB; ++b) {
            bstart[b] = run;
            for (int c = 0; c < C; ++c) {
                i64& v = counts[static_cast<size_t>(c) * NB + b];
                const i64 k = v;
                v = run;
                run += k;
            }
        }
        bstart[NB] = run;
    }
    auto& d = buf.b;
    d.resize(static_cast<size_t>(n));
#pragma omp parallel for schedule(static, 1)
    for (int c = 0; c < C; ++c) {
        i64* cnt = &counts[static_cast<size_t>(c) * NB];
        for (i64 i = n * c / C; i < n * (c + 1) / C; ++i) d[cnt[bucket[i]]++] = a[i];
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < NB; ++b) {
        std::sort(d.begin() + bstart[b], d.begin() + bstart[b + 1], item_less);
        for (i64 i = bstart[b]; i < bstart[b + 1]; ++i) {
            out[i] = d[i].cell;
            from[i] = d[i].from;
        }
    }
}

// Exclusive prefix sums of v into out (out[n] = total), in parallel blocks.
void prefix_sums(const std::vector<i64>& v, std::vector<i64>& out) {
    const i64 n = static_cast<i64>(v.size());
    out.resize(static_cast<size_t>(n + 1));
    const int T = std::max(1, std::min(max_threads(), static_cast<int>(n / 65536) + 1));
    std::vector<i64> part(static_cast<size_t>(T) + 1, 0);
#pragma omp parallel for num_threads(T) schedule(static, 1)
    for (int t = 0; t < T; ++t) {
        i64 s = 0;
        for (i64 i = n * t / T; i < n * (t + 1) / T; ++i) s += v[i];
        part[t + 1] = s;
    }
    for (int t = 0; t < T; ++t) part[t + 1] += part[t];
#pragma omp parallel for num_threads(T) schedule(static, 1)
    for (int t = 0; t < T; ++t) {
        i64 s = part[t];
        for (i64 i = n * t / T; i < n * (t + 1) / T; ++i) {
            out[i] = s;
            s += v[i];
        }
    }
    out[n] = part[T];
}

// ---------------------------------------------------------------- the ranking

class Ranker {
  public:
    Ranker(const Data& d, const RankSettings& s) : data_(d), set_(s) {
        N_ = d.cells();
        L_ = d.rules.layers;
        if (N_ <= 0) throw std::invalid_argument("no cells with data");
        if (N_ >= static_cast<i64>(std::numeric_limits<u32>::max()))
            throw std::invalid_argument("more than 2^32 - 1 cells with data");
        if (static_cast<int>(s.weights.size()) != L_ || static_cast<int>(s.abf_z.size()) != L_)
            throw std::invalid_argument("one weight and ABF exponent per layer");
        for (const auto* v : {&s.cost, &s.solution})
            if (!v->empty() && static_cast<i64>(v->size()) != N_) throw std::invalid_argument("one value per cell");
        if (!s.hmask.empty() && static_cast<i64>(s.hmask.size()) != N_)
            throw std::invalid_argument("one mask level per cell");
        if (s.rule == Rule::LOAD && s.solution.empty()) throw std::invalid_argument("LOAD needs a solution");
        if (s.max_iterations < 1) throw std::invalid_argument("max_iterations must be >= 1");
        w_.resize(L_);
        wabs_.resize(L_);
        z_.resize(L_);
        for (int l = 0; l < L_; ++l) {
            w_[l] = static_cast<double>(s.weights[l]);
            wabs_[l] = std::fabs(w_[l]);
            z_[l] = std::isnan(s.abf_z[l]) ? s.param : static_cast<double>(s.abf_z[l]);
        }
        p_ = s.param;
        // cost
        have_cost_ = !s.cost.empty();
        cost_prio_ = have_cost_ && s.cost_in_ranking;
        if (have_cost_) {
            total_cost_ = 0.0;
            for (float c : s.cost) total_cost_ += static_cast<double>(c);
            cost_mean_r_ = total_cost_ <= 0 ? 0.0 : static_cast<double>(N_) / total_cost_;
        }
        if (!s.hmask.empty() || cost_prio_) {
            cls_.resize(static_cast<size_t>(N_));
            for (i64 i = 0; i < N_; ++i) {
                const u32 h = s.hmask.empty() ? 0u : s.hmask[i];
                const u32 zc = cost_prio_ && !(s.cost[i] > 0.0f) ? 1u : 0u;
                cls_[i] = (h << 1) | zc;
            }
        }
        // losses of layers without occurrence in a cell (CAZP/CAZPW: 0^p is not 0)
        absent_.assign(L_, 0.0);
        dense_ = false;
        if (s.rule == Rule::CAZP || s.rule == Rule::CAZPW) {
            const double c0 = vdt::fast_pow(0.0, p_);
            for (int l = 0; l < L_; ++l) absent_[l] = s.rule == Rule::CAZPW ? c0 * wabs_[l] : c0;
            dense_ = c0 != 0.0;
        }
        build_occurrences();
    }

    RankResult run();
    void flexibility(const RankResult& ranking, double max_error, bool up, bool down, std::vector<float>& pos,
                     std::vector<float>& neg) const;

  private:
    const Data& data_;
    const RankSettings& set_;
    i64 N_ = 0;
    int L_ = 0;
    std::vector<double> w_, wabs_, z_, absent_;
    double p_ = 0.0;
    bool dense_ = false;
    std::vector<i64> nnz_, start_;
    bool have_cost_ = false, cost_prio_ = false;
    double total_cost_ = 0.0, cost_mean_r_ = 0.0;
    std::vector<u32> cls_;

    // the occurrences (entries with a value > 0) of the cells in the current order
    std::vector<i64> lptr_, nptr_;
    std::vector<u32> lent_, nent_;  // entries: layer << 16 | value
    std::vector<u32> lcls_, ncls_;      // class (hierarchic mask, zero cost) of the cells in that order
    std::vector<float> lcost_, ncost_;  // cost of the cells in that order (cost in the ranking)
    int chunks_ = 0;                    // chunks of positions
    std::vector<i64> chunk_off_;  // (chunks, layers): first occurrence of the layer in the chunk

    // work arrays (reused: fresh allocations of this size cost page faults every iteration)
    std::vector<std::uint16_t> occ_q_;  // the occurrences of every layer in the current order
    std::vector<double> snap_;          // (chunks, layers): level of the layer at the chunk's start
    std::vector<Sum> ord_sums_;         // sums (as compared) in the current order
    std::vector<i64> len_;
    std::vector<u32> to_;
    std::vector<double> terms_;
    SortBuffers sortbuf_;
    Profile prof_;

    double f_of(int l, std::uint16_t q) const { return static_cast<double>(q) * data_.layer_scale[l]; }

    // Z5's orderingModifier: the sum as compared, for a cell of cost c
    Sum modified(const Sum& s, float c) const {
        if (!cost_prio_ || !(c > 0.0f)) return s;
        const double cc = static_cast<double>(c) * cost_mean_r_;
        return {s.pos / cc, s.neg * cc};
    }
    float cost_at(i64 p) const { return cost_prio_ ? lcost_[p] : 0.0f; }
    u32 cls_at(i64 p) const { return lcls_.empty() ? 0u : lcls_[p]; }
    i64 chunk_lo(int c) const { return N_ * c / chunks_; }

    void build_occurrences();
    void relayout(const std::vector<u32>& from);
    void initial_sums(std::vector<Sum>& sums) const;
    void sort_cells_by(const std::vector<Sum>& sums, std::vector<u32>& out, std::vector<u32>& from);
    void sort_positions(const std::vector<u32>& order, const std::vector<Sum>* by_cell, std::vector<u32>& out,
                        std::vector<u32>& from);
    void scatter();
    void iteration_sums(const std::vector<u32>& order, std::vector<Sum>* sums);
    void ranking_error(double& err, i64& n_err);
    void curves(const std::vector<u32>& order, RankResult& res);
};

// The occurrences in the cell order (the first layout).
void Ranker::build_occurrences() {
    const auto& ptr = data_.cell_ptr;
    const auto& val = data_.entry_value;
    std::vector<i64> len(static_cast<size_t>(N_));
#pragma omp parallel for schedule(static)
    for (i64 c = 0; c < N_; ++c) {
        i64 k = 0;
        for (i64 e = ptr[c]; e < ptr[c + 1]; ++e) k += val[e] != 0;
        len[c] = k;
    }
    prefix_sums(len, lptr_);
    const i64 total = lptr_[N_];
    lent_.resize(static_cast<size_t>(total));
#pragma omp parallel for schedule(static)
    for (i64 c = 0; c < N_; ++c) {
        i64 o = lptr_[c];
        for (i64 e = ptr[c]; e < ptr[c + 1]; ++e)
            if (val[e]) lent_[o++] = (static_cast<u32>(data_.entry_layer[e]) << 16) | val[e];
    }
    lcls_ = cls_;
    if (cost_prio_) lcost_ = set_.cost;
    nnz_.assign(L_, 0);
    for (i64 o = 0; o < total; ++o) ++nnz_[lent_[o] >> 16];
    start_.assign(L_ + 1, 0);
    for (int l = 0; l < L_; ++l) start_[l + 1] = start_[l] + nnz_[l];
    chunks_ = static_cast<int>(std::min<i64>(N_, std::max(1, max_threads()) * 4));
}

// Move the occurrences, classes and costs to the new order (from: the previous
// position of each), and count the occurrences of every layer in every chunk of
// positions. The order changes little between iterations: the reads are nearly
// in sequence.
void Ranker::relayout(const std::vector<u32>& from) {
    auto& len = len_;
    len.resize(static_cast<size_t>(N_));
#pragma omp parallel for schedule(static)
    for (i64 p = 0; p < N_; ++p) {
        const u32 q = from[p];
        len[p] = lptr_[q + 1] - lptr_[q];
    }
    prefix_sums(len, nptr_);
    nent_.resize(lent_.size());
    chunk_off_.assign(static_cast<size_t>(chunks_) * L_, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < chunks_; ++c) {
        for (i64 p = chunk_lo(c); p < chunk_lo(c + 1); ++p) {
            const u32 q = from[p];
            const i64 a = lptr_[q], k = lptr_[q + 1] - a;
            if (k) std::memcpy(&nent_[nptr_[p]], &lent_[a], static_cast<size_t>(k) * sizeof(u32));
        }
        i64* cnt = &chunk_off_[static_cast<size_t>(c) * L_];
        for (i64 o = nptr_[chunk_lo(c)]; o < nptr_[chunk_lo(c + 1)]; ++o) ++cnt[nent_[o] >> 16];
    }
    for (int l = 0; l < L_; ++l) {
        i64 run = start_[l];
        for (int c = 0; c < chunks_; ++c) {
            i64& v = chunk_off_[static_cast<size_t>(c) * L_ + l];
            const i64 k = v;
            v = run;
            run += k;
        }
    }
    lptr_.swap(nptr_);
    lent_.swap(nent_);
    if (!lcls_.empty()) {
        ncls_.resize(lcls_.size());
#pragma omp parallel for schedule(static)
        for (i64 p = 0; p < N_; ++p) ncls_[p] = lcls_[from[p]];
        lcls_.swap(ncls_);
    }
    if (cost_prio_) {
        ncost_.resize(lcost_.size());
#pragma omp parallel for schedule(static)
        for (i64 p = 0; p < N_; ++p) ncost_[p] = lcost_[from[p]];
        lcost_.swap(ncost_);
    }
}

void Ranker::initial_sums(std::vector<Sum>& sums) const {
    sums.assign(static_cast<size_t>(N_), Sum{});
    if (set_.rule == Rule::LOAD) {
        for (i64 i = 0; i < N_; ++i) sums[i].pos = static_cast<double>(set_.solution[i]);
        return;
    }
    if (set_.rule == Rule::RAND) {
        std::mt19937 gen(static_cast<std::mt19937::result_type>(set_.seed));
        std::uniform_real_distribution<double> dist;
        for (i64 i = 0; i < N_; ++i) sums[i].pos = dist(gen);
        return;
    }
    // the first layout is the cell order
#pragma omp parallel for schedule(static)
    for (i64 i = 0; i < N_; ++i) {
        Sum s;
        for (i64 o = lptr_[i]; o < lptr_[i + 1]; ++o) {
            const int l = static_cast<int>(lent_[o] >> 16);
            const double v = f_of(l, static_cast<std::uint16_t>(lent_[o])) * wabs_[l];
            (w_[l] >= 0 ? s.pos : s.neg) += v;
        }
        sums[i] = s;
    }
}

// Sort the cells, in the cell order (the layout then), by their sums.
void Ranker::sort_cells_by(const std::vector<Sum>& sums, std::vector<u32>& out, std::vector<u32>& from) {
    auto& a = sortbuf_.a;
    a.resize(static_cast<size_t>(N_));
#pragma omp parallel for schedule(static)
    for (i64 i = 0; i < N_; ++i)
        a[i] = {sort_key(modified(sums[i], cost_at(i)).value()), cls_at(i), static_cast<u32>(i), static_cast<u32>(i)};
    sort_items(sortbuf_, out, from);
}

// Sort the cells, given in the current order, by their sums: `by_cell` (sums per
// cell), or the sums of the last iteration (ord_sums_, in the current order).
void Ranker::sort_positions(const std::vector<u32>& order, const std::vector<Sum>* by_cell, std::vector<u32>& out,
                            std::vector<u32>& from) {
    auto& a = sortbuf_.a;
    a.resize(static_cast<size_t>(N_));
#pragma omp parallel for schedule(static)
    for (i64 p = 0; p < N_; ++p) {
        const u32 cell = order[p];
        const double v = by_cell ? modified((*by_cell)[cell], cost_at(p)).value() : ord_sums_[p].value();
        a[p] = {sort_key(v), cls_at(p), cell, static_cast<u32>(p)};
    }
    sort_items(sortbuf_, out, from);
}

// The occurrences of every layer in the current order (occ_q_).
void Ranker::scatter() {
    occ_q_.resize(static_cast<size_t>(start_[L_]));
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < chunks_; ++c) {
        std::vector<i64> cur(chunk_off_.begin() + static_cast<i64>(c) * L_,
                             chunk_off_.begin() + static_cast<i64>(c + 1) * L_);
        for (i64 o = lptr_[chunk_lo(c)]; o < lptr_[chunk_lo(c + 1)]; ++o) {
            const u32 e = lent_[o];
            occ_q_[cur[e >> 16]++] = static_cast<std::uint16_t>(e);
        }
    }
}

void Ranker::iteration_sums(const std::vector<u32>& order, std::vector<Sum>* sums) {
    prof_.mark();
    scatter();
    prof_(1);
    // the level of every layer at the start of every chunk: Zonation's sequential subtractions
    snap_.resize(static_cast<size_t>(chunks_) * L_);
#pragma omp parallel for schedule(dynamic, 1)
    for (int l = 0; l < L_; ++l) {
        double R = data_.initial_levels[l];
        const double sc = data_.layer_scale[l];
        for (int c = 0; c < chunks_; ++c) {
            snap_[static_cast<size_t>(c) * L_ + l] = R;
            const i64 end = c + 1 < chunks_ ? chunk_off_[static_cast<size_t>(c + 1) * L_ + l] : start_[l + 1];
            for (i64 o = chunk_off_[static_cast<size_t>(c) * L_ + l]; o < end; ++o)
                R = R - static_cast<double>(occ_q_[o]) * sc;
        }
    }
    prof_(2);
    // walk every chunk of positions: the losses of each cell, from the levels
    // before and after it, summed in layer order
    const Rule rule = set_.rule;
    const bool maxrule = rule == Rule::CAZMAX;
    const double root = (rule == Rule::CAZP || rule == Rule::CAZPW) && p_ > 0 ? 1.0 / p_ : 0.0;
    const double* scale = data_.layer_scale.data();
    const double* init = data_.initial_levels.data();
    if (sums) sums->resize(static_cast<size_t>(N_));
    ord_sums_.resize(static_cast<size_t>(N_));
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < chunks_; ++c) {
        std::vector<double> R(snap_.begin() + static_cast<i64>(c) * L_, snap_.begin() + static_cast<i64>(c + 1) * L_);
        std::vector<double> bval;       // ABF: b(R) of the current level
        std::vector<std::uint8_t> bset;
        if (rule == Rule::ABF) {
            bval.assign(L_, 0.0);
            bset.assign(L_, 0);
        }
        auto b_of = [&](int l, double r) {
            const double x = w_[l] >= 0 ? r : init[l] - r;
            return vdt::fast_pow(x > 0.0 ? x : 0.0, z_[l]);
        };
        auto loss = [&](int l, std::uint16_t q) {
            const double Rb = R[l], Ra = Rb - static_cast<double>(q) * scale[l];
            R[l] = Ra;
            if (rule == Rule::ABF) {
                if (!bset[l]) {
                    bval[l] = b_of(l, Rb);
                    bset[l] = 1;
                }
                const double b1 = b_of(l, Ra);
                const double v = (bval[l] - b1) * w_[l];
                bval[l] = b1;
                return v;
            }
            const double ob = Rb - Ra;
            double od = w_[l] >= 0 ? Rb : init[l] - Ra;
            od = od > 0 ? ob / od : 0.0;
            if (rule != Rule::CAZPW) od *= wabs_[l];
            if (rule == Rule::CAZ2)
                od = od * od;
            else if (rule == Rule::CAZP || rule == Rule::CAZPW)
                od = vdt::fast_pow(od, p_);
            if (rule == Rule::CAZPW) od *= wabs_[l];
            return od;
        };
        auto add = [&](Sum& s, int l, double v) {
            double& acc = w_[l] >= 0 ? s.pos : s.neg;
            if (maxrule)
                acc = std::max(acc, v);
            else
                acc += v;
        };
        for (i64 p = chunk_lo(c); p < chunk_lo(c + 1); ++p) {
            Sum s;
            if (dense_) {
                int l = 0;
                for (i64 o = lptr_[p]; o < lptr_[p + 1]; ++o) {
                    const int le = static_cast<int>(lent_[o] >> 16);
                    for (; l < le; ++l) add(s, l, absent_[l]);
                    add(s, le, loss(le, static_cast<std::uint16_t>(lent_[o])));
                    l = le + 1;
                }
                for (; l < L_; ++l) add(s, l, absent_[l]);
            } else {
                for (i64 o = lptr_[p]; o < lptr_[p + 1]; ++o) {
                    const int le = static_cast<int>(lent_[o] >> 16);
                    add(s, le, loss(le, static_cast<std::uint16_t>(lent_[o])));
                }
            }
            if (rule == Rule::CAZ2) {
                s.pos = std::sqrt(s.pos);
                s.neg = std::sqrt(s.neg);
            } else if (root > 0) {
                s = packet_pow(s, root);
            }
            if (sums) (*sums)[order[p]] = s;
            ord_sums_[p] = modified(s, cost_at(p));
        }
    }
    prof_(3);
}

// Zonation's mean ranking error of the current order: the terms in parallel,
// their sum in sequence (adding the zeros of the pairs in order changes nothing).
void Ranker::ranking_error(double& err, i64& n_err) {
    terms_.resize(static_cast<size_t>(N_));
    i64 tests = 0, errors = 0;
#pragma omp parallel for schedule(static) reduction(+ : tests, errors)
    for (i64 i = 1; i < N_; ++i) {
        double t = 0.0;
        if (lcls_.empty() || lcls_[i - 1] == lcls_[i]) {
            const Sum& s1 = ord_sums_[i - 1];
            const Sum& s2 = ord_sums_[i];
            ++tests;
            const double div = s1.pos + s1.neg;
            if (s1.value() > s2.value() && div > 0) {
                ++errors;
                t = (s1.value() - s2.value()) / div;
            }
        }
        terms_[i] = t;
    }
    double e = 0.0;
    for (i64 i = 1; i < N_; ++i) e += terms_[i];
    n_err = errors;
    err = errors == 0 ? 0.0 : e / static_cast<double>(tests);
}

// Zonation's curves (FeatureOccurrenceSampler) of the order: the remaining level
// of every layer and the remaining cost before the cells at positions 0, k, 2k, ...
// and after the last one; and the area under the mean curve of the positive layers.
void Ranker::curves(const std::vector<u32>& order, RankResult& res) {
    const i64 N = N_;
    const i64 req = static_cast<i64>(set_.curves_interval * static_cast<float>(N));
    i64 si;
    if (req <= 0)
        si = std::max<i64>(1, N / 10000);
    else if (N / req > 100000)
        si = std::max<i64>(1, N / 100000);
    else
        si = req;
    res.curve_interval = si;
    std::vector<i64> at;
    for (i64 k = 0; k < N; k += si) at.push_back(k);
    at.push_back(N);
    const i64 S = static_cast<i64>(at.size());
    res.curve_rank.resize(static_cast<size_t>(S));
    for (i64 s = 0; s < S; ++s)
        res.curve_rank[s] = static_cast<float>(static_cast<double>(at[s]) / static_cast<double>(N));
    // the occurrences of each layer in the order, and their positions
    scatter();
    std::vector<i64> occ_pos(static_cast<size_t>(start_[L_]));
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < chunks_; ++c) {
        std::vector<i64> cur(chunk_off_.begin() + static_cast<i64>(c) * L_,
                             chunk_off_.begin() + static_cast<i64>(c + 1) * L_);
        for (i64 p = chunk_lo(c); p < chunk_lo(c + 1); ++p)
            for (i64 o = lptr_[p]; o < lptr_[p + 1]; ++o) occ_pos[cur[lent_[o] >> 16]++] = p;
    }
    res.curve_levels.assign(static_cast<size_t>(S) * L_, 0.0f);
    std::vector<double> area(L_, 0.0);
#pragma omp parallel for schedule(dynamic, 1)
    for (int l = 0; l < L_; ++l) {
        double R = data_.initial_levels[l];
        i64 o = start_[l];
        const i64 end = start_[l + 1];
        i64 prev = 0;  // positions [prev, p) have the level R
        double a = 0.0;
        for (i64 s = 0; s < S; ++s) {
            const i64 p = at[s];
            while (o < end && occ_pos[o] < p) {
                const i64 q = occ_pos[o];
                a += R * static_cast<double>(q + 1 - prev);
                prev = q + 1;
                R = R - f_of(l, occ_q_[o]);
                ++o;
            }
            const float v = static_cast<float>(R);
            res.curve_levels[static_cast<size_t>(s) * L_ + l] = v > 0.0f ? v : 0.0f;
        }
        a += R * static_cast<double>(N - prev);
        area[l] = a;
    }
    res.curve_cost.assign(static_cast<size_t>(S), 0.0);
    if (have_cost_) {
        double rc = total_cost_;
        i64 s = 0;
        for (i64 p = 0; p <= N; ++p) {
            while (s < S && at[s] == p) res.curve_cost[s++] = rc;
            if (p < N) rc -= static_cast<double>(set_.cost[order[p]]);
        }
    }
    double auc = 0.0;
    int npos = 0;
    for (int l = 0; l < L_; ++l)
        if (w_[l] > 0) {
            auc += area[l];
            ++npos;
        }
    res.auc = npos ? auc / npos / static_cast<double>(N) : 0.0;
    res.total_cost = total_cost_;
}

RankResult Ranker::run() {
    RankResult res;
    const double ndr = 1.0 / static_cast<double>(N_);
    std::vector<Sum> sums, weighted;
    std::vector<u32> order, next, from;
    prof_.mark();
    initial_sums(sums);
    sort_cells_by(sums, order, from);
    relayout(from);
    prof_(0);
    const bool iterative = set_.rule != Rule::RAND && set_.rule != Rule::LOAD;
    if (iterative) {
        const bool smooth = set_.smoothing > 0;
        if (smooth) weighted = sums;
        double prev = 0.0;
        const double sf = static_cast<double>(set_.smoothing);
        const double sf1 = static_cast<double>(1.0f - set_.smoothing);
        std::vector<double> moves(static_cast<size_t>(N_));
        to_.resize(static_cast<size_t>(N_));
        for (int it = 0; it < set_.max_iterations; ++it) {
            iteration_sums(order, smooth ? &sums : nullptr);
            double err;
            i64 n_err;
            ranking_error(err, n_err);
            prof_(5);
            if (smooth) {
#pragma omp parallel for schedule(static)
                for (i64 i = 0; i < N_; ++i) {
                    weighted[i].pos = weighted[i].pos * sf + sums[i].pos * sf1;
                    weighted[i].neg = weighted[i].neg * sf + sums[i].neg * sf1;
                }
            }
            sort_positions(order, smooth ? &weighted : nullptr, next, from);
            prof_(6);
            // Zonation's mean rank change: |old position - new position| in the old order
#pragma omp parallel for schedule(static)
            for (i64 j = 0; j < N_; ++j) to_[from[j]] = static_cast<u32>(j);
#pragma omp parallel for schedule(static)
            for (i64 i = 0; i < N_; ++i) moves[i] = std::fabs(static_cast<double>(i - static_cast<i64>(to_[i])) * ndr);
            double change = 0.0;
            for (i64 i = 0; i < N_; ++i) change += moves[i];
            change /= static_cast<double>(N_);
            prof_(4);
            order.swap(next);
            relayout(from);
            prof_(7);
            res.iterations = it + 1;
            res.ranking_error = err;
            res.ranking_errors = n_err;
            res.rank_change = change;
            res.history_error.push_back(err);
            res.history_errors.push_back(n_err);
            res.history_change.push_back(change);
            if (err <= prev && err <= static_cast<double>(set_.ranking_error) * 0.01) break;
            if (set_.stop_on_plateau && it >= 5 && err >= prev) break;
            prev = err;
        }
    }
    res.rank.resize(static_cast<size_t>(N_));
#pragma omp parallel for schedule(static)
    for (i64 i = 0; i < N_; ++i) res.rank[order[i]] = static_cast<float>(static_cast<double>(i) * ndr);
    prof_.mark();
    curves(order, res);
    prof_(8);
    res.order = std::move(order);
    return res;
}

}  // namespace

void Ranker::flexibility(const RankResult& rr, double max_error, bool up, bool down, std::vector<float>& fpos,
                         std::vector<float>& fneg) const {
    const i64 N = N_;
    const double ndr = 1.0 / static_cast<double>(N);
    const auto& order = rr.order;
    const i64 S = static_cast<i64>(rr.curve_rank.size());
    const double scalar = static_cast<double>(N) / static_cast<double>(rr.curve_interval);
    const Rule rule = set_.rule;
    const bool maxrule = rule == Rule::CAZMAX;
    const double root = (rule == Rule::CAZP || rule == Rule::CAZPW) && p_ > 0 ? 1.0 / p_ : 0.0;
    const double* scale = data_.layer_scale.data();
    const double* init = data_.initial_levels.data();
    const auto& ptr = data_.cell_ptr;
    const auto& lay = data_.entry_layer;
    const auto& val = data_.entry_value;
    const bool calc_pos = up, calc_neg = down;
    if (calc_pos) fpos.assign(static_cast<size_t>(N), 0.0f);
    if (calc_neg) fneg.assign(static_cast<size_t>(N), 0.0f);
    // getInterpolatedOccurrences: the curve sample columns and the weight of the next one
    auto interp = [&](float rank, i64& i1, i64& i2, float& t) {
        i1 = std::min<i64>(static_cast<i64>(static_cast<double>(rank) * scalar), S - 1);
        i2 = std::min<i64>(S - 1, i1 + 1);
        const float r = rr.curve_rank[i2] - rr.curve_rank[i1];
        t = r > 0.0f ? (rank - rr.curve_rank[i1]) / r : 0.0f;
    };
    // the loss of a cell at the remaining levels (Zonation's BalancingMethod::sum(remaining, features))
    auto cell_sum = [&](u32 cell, i64 i1, i64 i2, float t) {
        auto rem = [&](int l) {
            const float v1 = rr.curve_levels[static_cast<size_t>(i1) * L_ + l];
            const float v2 = rr.curve_levels[static_cast<size_t>(i2) * L_ + l];
            return static_cast<double>((v2 - v1) * t + v1);
        };
        auto loss = [&](int l, std::uint16_t q) {
            const double Rb = rem(l), Ra = Rb - static_cast<double>(q) * scale[l];
            if (rule == Rule::ABF) {
                auto b_of = [&](double r) {
                    const double x = w_[l] >= 0 ? r : init[l] - r;
                    return vdt::fast_pow(x > 0.0 ? x : 0.0, z_[l]);
                };
                return (b_of(Rb) - b_of(Ra)) * w_[l];
            }
            const double ob = Rb - Ra;
            double od = w_[l] >= 0 ? Rb : init[l] - Ra;
            od = od > 0 ? ob / od : 0.0;
            if (rule != Rule::CAZPW) od *= wabs_[l];
            if (rule == Rule::CAZ2)
                od = od * od;
            else if (rule == Rule::CAZP || rule == Rule::CAZPW)
                od = vdt::fast_pow(od, p_);
            if (rule == Rule::CAZPW) od *= wabs_[l];
            return od;
        };
        Sum s;
        auto add = [&](int l, double v) {
            double& acc = w_[l] >= 0 ? s.pos : s.neg;
            if (maxrule)
                acc = std::max(acc, v);
            else
                acc += v;
        };
        int l = 0;
        for (i64 e = ptr[cell]; e < ptr[cell + 1]; ++e) {
            if (!val[e]) continue;
            const int le = lay[e];
            if (dense_)
                for (; l < le; ++l) add(l, absent_[l]);
            add(le, loss(le, val[e]));
            l = le + 1;
        }
        if (dense_)
            for (; l < L_; ++l) add(l, absent_[l]);
        if (rule == Rule::CAZ2) {
            s.pos = std::sqrt(s.pos);
            s.neg = std::sqrt(s.neg);
        } else if (root > 0) {
            s = packet_pow(s, root);
        }
        return s;
    };
#pragma omp parallel for schedule(dynamic, 256)
    for (i64 index = 0; index < N; ++index) {
        const u32 cell = order[index];
        auto rank_error = [&](i64 offset) {
            const i64 ti = index + offset;
            if (ti < 0 || ti >= N) return max_error;
            const u32 other = order[ti];
            i64 i1, i2;
            float t;
            interp(static_cast<float>(static_cast<double>(ti) * ndr), i1, i2, t);
            Sum test = cell_sum(cell, i1, i2, t);
            Sum real = cell_sum(other, i1, i2, t);
            if (!cls_.empty() && cls_[cell] != cls_[other]) return max_error;
            if (cost_prio_) {
                test = modified(test, set_.cost[cell]);
                real = modified(real, set_.cost[other]);
            }
            const double diff = std::fabs(test.value() - real.value());
            const double div = test.pos + test.neg;
            if (diff == 0) return 0.0;
            if (div <= 0) return max_error;
            return diff / div;
        };
        auto test_rank = [&](i64 dir, i64 i) {
            const double e1 = rank_error(dir * i), e2 = rank_error(dir * (i + 1));
            return (e1 + e2) / 2 >= max_error;
        };
        // upperBound(first, last, 9 iterations, comp)
        auto upper = [&](i64 first, i64 last, i64 dir) {
            i64 count = last - first;
            int iter = 9;
            while (count > 0 && iter-- > 0) {
                const i64 step = count / 2;
                const i64 i = first + step;
                if (!test_rank(dir, i)) {
                    first = i + 1;
                    count -= step + 1;
                } else {
                    count = step;
                }
            }
            return first;
        };
        if (calc_pos) fpos[cell] = static_cast<float>(static_cast<double>(upper(1, N - index, 1)) * ndr);
        if (calc_neg) fneg[cell] = static_cast<float>(static_cast<double>(upper(1, index, -1)) * ndr);
    }
}

RankResult rank(const Data& data, const RankSettings& settings) {
    Ranker r(data, settings);
    return r.run();
}

}  // namespace prio

namespace prio {

void flexibility(const Data& data, const RankSettings& settings, const RankResult& ranking, double max_error,
                 bool up, bool down, std::vector<float>& pos, std::vector<float>& neg) {
    Ranker r(data, settings);
    r.flexibility(ranking, max_error, up, down, pos, neg);
}

}  // namespace prio
