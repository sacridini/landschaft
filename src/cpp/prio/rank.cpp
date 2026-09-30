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
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
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
        static const char* names[] = {"initial", "scatter", "levels", "losses", "change",
                                      "error",   "sort",    "relayout", "curves", "build"};
        for (int i = 0; i < 10; ++i) std::fprintf(stderr, "[prio] %-10s %10.1f ms\n", names[i], ms_[i]);
    }

  private:
    using Clock = std::chrono::steady_clock;
    bool on_;
    Clock::time_point last_;
    double ms_[10] = {};
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
    static bool less(const Item& a, const Item& b) {
        if (a.cls != b.cls) return a.cls < b.cls;
        if (a.key != b.key) return a.key < b.key;
        return a.cell < b.cell;
    }
};

// The same without classes (no hierarchic mask or cost classes): 16 bytes.
struct PlainItem {
    u64 key;
    u32 cell;
    u32 from;
    static bool less(const PlainItem& a, const PlainItem& b) {
        if (a.key != b.key) return a.key < b.key;
        return a.cell < b.cell;
    }
};

template <class It>
struct SortBuffers {
    Buffer<It> a, b;
    std::vector<std::uint16_t> bucket;
    std::vector<i64> counts;
};

template <class It>
void insertion_sort(It* a, i64 n) {
    for (i64 i = 1; i < n; ++i) {
        const It v = a[i];
        i64 j = i;
        for (; j > 0 && It::less(v, a[j - 1]); --j) a[j] = a[j - 1];
        a[j] = v;
    }
}

inline int clz64(u64 m) {
#if defined(_MSC_VER)
    unsigned long i;
    _BitScanReverse64(&i, m);
    return 63 - static_cast<int>(i);
#else
    return __builtin_clzll(m);
#endif
}

// Sort a bucket of the sample sort (in cache): with classes, std::sort; else the
// items are distributed by the key bits below the prefix common to the bucket,
// and each small group is sorted by (key, cell).
inline void sort_bucket(Item* a, i64 n) { std::sort(a, a + n, Item::less); }

inline void sort_bucket(PlainItem* a, i64 n) {
    if (n <= 32) {
        insertion_sort(a, n);
        return;
    }
    u64 lo = a[0].key, hi = a[0].key;
    for (i64 i = 1; i < n; ++i) {
        lo = std::min(lo, a[i].key);
        hi = std::max(hi, a[i].key);
    }
    if (lo == hi) {
        std::sort(a, a + n, PlainItem::less);
        return;
    }
    const int top = 64 - clz64(lo ^ hi);  // bits below the common prefix
    int bits = 1;
    while (bits < 12 && (i64{1} << bits) * 4 < n) ++bits;
    bits = std::min(bits, top);
    const int shift = top - bits;
    const u64 mask = (u64{1} << bits) - 1;
    thread_local std::vector<PlainItem> tmp;
    thread_local std::vector<i64> cnt;
    tmp.resize(static_cast<size_t>(n));
    cnt.assign((size_t{1} << bits) + 1, 0);
    for (i64 i = 0; i < n; ++i) ++cnt[((a[i].key >> shift) & mask) + 1];
    for (size_t g = 1; g < cnt.size(); ++g) cnt[g] += cnt[g - 1];
    for (i64 i = 0; i < n; ++i) tmp[cnt[(a[i].key >> shift) & mask]++] = a[i];
    // cnt[g] is now the end of group g
    i64 b = 0;
    for (size_t g = 0; g + 1 < cnt.size(); ++g) {
        const i64 e = cnt[g];
        if (e - b <= 32)
            insertion_sort(tmp.data() + b, e - b);
        else
            std::sort(tmp.data() + b, tmp.data() + e, PlainItem::less);
        b = e;
    }
    std::memcpy(a, tmp.data(), static_cast<size_t>(n) * sizeof(PlainItem));
}

// Parallel sample sort of buf.a into `out` (the cells): splitters from a sample,
// the items distributed to their buckets, each bucket sorted. Given the items in
// the previous order, the buckets are nearly sorted already.
template <class It>
void sort_items(SortBuffers<It>& buf, std::vector<u32>& out, std::vector<u32>& from) {
    const auto item_less = It::less;
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
    std::vector<It> sample(static_cast<size_t>(NB) * per);
    const i64 S = static_cast<i64>(sample.size());
    for (i64 k = 0; k < S; ++k) sample[k] = a[static_cast<size_t>((2 * k + 1) * n / (2 * S))];
    std::sort(sample.begin(), sample.end(), item_less);
    std::vector<It> split(static_cast<size_t>(NB) - 1);
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
        sort_bucket(d.data() + bstart[b], bstart[b + 1] - bstart[b]);
        for (i64 i = bstart[b]; i < bstart[b + 1]; ++i) {
            out[i] = d[i].cell;
            from[i] = d[i].from;
        }
    }
}

// Exclusive prefix sums of v into out (out[n] = total), in parallel blocks.
template <class Out>
void prefix_sums(const std::vector<i64>& v, Out& out) {
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

// The same in place: v[0..n) the values, v[n] set to the total.
template <class V>
void prefix_sums_inplace(V& v) {
    const i64 n = static_cast<i64>(v.size()) - 1;
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
            const i64 k = v[i];
            v[i] = s;
            s += k;
        }
    }
    v[n] = part[T];
}

// ---------------------------------------------------------------- compact rows

// The occurrences of a cell (its layers with a value > 0, in layer order) are a
// row of bytes, whichever of two formats is smaller for the data: delta rows
// (per occurrence the gap from the previous layer, one byte or 255 and two more,
// then the value) for many sparse layers, bitmap rows (a bit per layer, then the
// values) for few or dense ones. Values take one byte with 8-bit data. The rows
// are moved every iteration; their size is what that costs.

inline int ctz64(u64 m) {
#if defined(_MSC_VER)
    unsigned long i;
    _BitScanForward64(&i, m);
    return static_cast<int>(i);
#else
    return __builtin_ctzll(m);
#endif
}

constexpr i64 kPad = 8;  // spare bytes after the rows

template <int VB>
inline unsigned read_value(const std::uint8_t* p) {
    if constexpr (VB == 1)
        return p[0];
    else
        return static_cast<unsigned>(p[0]) | (static_cast<unsigned>(p[1]) << 8);
}

template <int VB>
struct DeltaRows {
    static constexpr bool bitmap = false;
    static constexpr int vb = VB;
    static i64 occurrence_bytes(int gap) { return (gap >= 255 ? 3 : 1) + VB; }
    // f(layer, value) for the occurrences of the row [p, end)
    template <class F>
    void each(const std::uint8_t* p, const std::uint8_t* end, F&& f) const {
        int l = -1;
        while (p < end) {
            int g = *p++;
            if (g == 255) {
                g += p[0] | (p[1] << 8);
                p += 2;
            }
            l += g + 1;
            f(l, read_value<VB>(p));
            p += VB;
        }
    }
    std::uint8_t* put(std::uint8_t* p, int& prev, int l, unsigned q) const {
        const int g = l - prev - 1;
        prev = l;
        if (g >= 255) {
            *p++ = 255;
            *p++ = static_cast<std::uint8_t>((g - 255) & 0xff);
            *p++ = static_cast<std::uint8_t>((g - 255) >> 8);
        } else {
            *p++ = static_cast<std::uint8_t>(g);
        }
        *p++ = static_cast<std::uint8_t>(q & 0xff);
        if constexpr (VB == 2) *p++ = static_cast<std::uint8_t>(q >> 8);
        return p;
    }
};

template <int VB>
struct BitmapRows {
    static constexpr bool bitmap = true;
    static constexpr int vb = VB;
    int mask_bytes = 0;
    template <class F>
    void each(const std::uint8_t* p, const std::uint8_t*, F&& f) const {
        const std::uint8_t* v = p + mask_bytes;
        for (int k = 0; k < mask_bytes; k += 8) {
            // the row buffers end in kPad spare bytes: read 8 and drop what is past the mask
            u64 m;
            std::memcpy(&m, p + k, 8);
            if (mask_bytes - k < 8) m &= (u64{1} << (8 * (mask_bytes - k))) - 1;
            while (m) {
                const int b = ctz64(m);
                m &= m - 1;
                f(k * 8 + b, read_value<VB>(v));
                v += VB;
            }
        }
    }
};

// ---------------------------------------------------------------- the ranking

class Ranker {
  public:
    // rows: build the rows for the ranking (not needed for the flexibility map)
    Ranker(const Data& d, const RankSettings& s, bool rows = true) : data_(d), set_(s) {
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
        if (!s.storage.empty()) {
            for (auto* b : {&lptr_, &nptr_}) b->set_storage(s.storage);
            for (auto* b : {&lrow_, &nrow_, &occ8_}) b->set_storage(s.storage);
            occ16_.set_storage(s.storage);
            ord_sums_.set_storage(s.storage);
            work_.set_storage(s.storage);
            sortbuf_.a.set_storage(s.storage);
            sortbuf_.b.set_storage(s.storage);
            plainbuf_.a.set_storage(s.storage);
            plainbuf_.b.set_storage(s.storage);
        }
        if (rows) {
            prof_.mark();
            build_occurrences();
            prof_(9);
        }
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

    // the rows of the cells in the current order (byte offsets)
    Buffer<i64> lptr_, nptr_;
    Buffer<std::uint8_t> lrow_, nrow_;
    bool bitmap_ = false;
    int vb_ = 1, mask_bytes_ = 0;
    std::vector<u32> lcls_, ncls_;      // class (hierarchic mask, zero cost) of the cells in that order
    std::vector<float> lcost_, ncost_;  // cost of the cells in that order (cost in the ranking)
    int chunks_ = 0;                    // chunks of positions
    std::vector<i64> chunk_off_;  // (chunks, layers): first occurrence of the layer in the chunk

    // work arrays (reused: fresh allocations of this size cost page faults every iteration)
    // the values of the occurrences of every layer in the current order (8 or 16 bits)
    Buffer<std::uint8_t> occ8_;
    Buffer<std::uint16_t> occ16_;
    unsigned occ_value(i64 o) const { return vb_ == 1 ? occ8_[o] : occ16_[o]; }
    std::vector<double> snap_;          // (chunks, layers): level of the layer at the chunk's start
    Buffer<Sum> ord_sums_;              // sums (as compared) in the current order
    Buffer<double> work_;  // terms of the ranking error, then rank moves
    SortBuffers<Item> sortbuf_;
    SortBuffers<PlainItem> plainbuf_;
    Profile prof_;

    double f_of(int l, std::uint16_t q) const { return static_cast<double>(q) * data_.layer_scale[l]; }

    // Z5's orderingModifier: the sum as compared, for a cell of cost c
    Sum modified(const Sum& s, float c) const {
        if (!cost_prio_ || !(c > 0.0f)) return s;
        const double cc = static_cast<double>(c) * cost_mean_r_;
        return {s.pos / cc, s.neg * cc};
    }
    float cost_at(i64 p) const { return cost_prio_ ? lcost_[p] : 0.0f; }
    i64 chunk_lo(int c) const { return N_ * c / chunks_; }

    // fn(rows) with the row format of the data
    template <class Fn>
    void with_rows(Fn&& fn) const {
        if (bitmap_) {
            if (vb_ == 1)
                fn(BitmapRows<1>{mask_bytes_});
            else
                fn(BitmapRows<2>{mask_bytes_});
        } else {
            if (vb_ == 1)
                fn(DeltaRows<1>{});
            else
                fn(DeltaRows<2>{});
        }
    }
    const std::uint8_t* row(i64 p) const { return lrow_.data() + lptr_[p]; }
    const std::uint8_t* row_end(i64 p) const { return lrow_.data() + lptr_[p + 1]; }

    void build_occurrences();
    template <class R>
    void encode(const R& rows);
    void relayout(const std::vector<u32>& from);
    template <class R>
    void relayout_rows(const R& rows, const std::vector<u32>& from);
    void initial_sums(std::vector<Sum>& sums) const;
    template <class R>
    void initial_sums_rows(const R& rows, std::vector<Sum>& sums) const;
    template <class Make>
    void sort_by(Make&& item, std::vector<u32>& out, std::vector<u32>& from);
    void sort_cells_by(const std::vector<Sum>& sums, std::vector<u32>& out, std::vector<u32>& from);
    void sort_positions(const std::vector<u32>& order, const std::vector<Sum>* by_cell, std::vector<u32>& out,
                        std::vector<u32>& from);
    void scatter();
    template <class R>
    void scatter_rows(const R& rows);
    void iteration_sums(const std::vector<u32>& order, std::vector<Sum>* sums);
    template <class R>
    void losses_rows(const R& rows, const std::vector<u32>& order, std::vector<Sum>* sums);
    template <Rule RULE, class Rows>
    void losses_rule(const Rows& rows, const std::vector<u32>& order, std::vector<Sum>* sums);
    template <class R>
    void positions_rows(const R& rows, Buffer<u32>& occ_pos) const;
    void ranking_error(double& err, i64& n_err);
    void curves(const std::vector<u32>& order, RankResult& res);
};

// The rows in the cell order (the first layout), in the smaller format.
void Ranker::build_occurrences() {
    vb_ = data_.value_bytes();
    mask_bytes_ = (L_ + 7) / 8;
    // occurrences (values > 0) per layer, and the size of the delta rows
    nnz_.assign(L_, 0);
    i64 delta = 0;
    {
        const int T = std::max(1, std::min(max_threads(), static_cast<int>(N_ / 16384) + 1));
        std::vector<i64> part(static_cast<size_t>(T) * L_, 0);
#pragma omp parallel for num_threads(T) schedule(static, 1) reduction(+ : delta)
        for (int t = 0; t < T; ++t) {
            i64* cnt = &part[static_cast<size_t>(t) * L_];
            for (i64 c = N_ * t / T; c < N_ * (t + 1) / T; ++c) {
                int prev = -1;
                data_.each<false>(c, [&](int l, unsigned q) {
                    if (!q) return;
                    ++cnt[l];
                    delta += (l - prev - 1 >= 255 ? 3 : 1) + vb_;
                    prev = l;
                });
            }
        }
        for (int t = 0; t < T; ++t)
            for (int l = 0; l < L_; ++l) nnz_[l] += part[static_cast<size_t>(t) * L_ + l];
    }
    start_.assign(L_ + 1, 0);
    for (int l = 0; l < L_; ++l) start_[l + 1] = start_[l] + nnz_[l];
    const i64 bitmap = N_ * mask_bytes_ + start_[L_] * vb_;
    bitmap_ = set_.row_format == 2 || (set_.row_format == 0 && bitmap < delta);
    with_rows([&](const auto& rows) { encode(rows); });
    lcls_ = cls_;
    if (cost_prio_) lcost_ = set_.cost;
    chunks_ = static_cast<int>(std::min<i64>(N_, std::max(1, max_threads()) * 4));
}

template <class R>
void Ranker::encode(const R& rows) {
    std::vector<i64> len(static_cast<size_t>(N_));
#pragma omp parallel for schedule(static)
    for (i64 c = 0; c < N_; ++c) {
        i64 k = R::bitmap ? mask_bytes_ : 0;
        int prev = -1;
        data_.each<false>(c, [&](int l, unsigned q) {
            if (!q) return;
            if constexpr (R::bitmap)
                k += R::vb;
            else
                k += R::occurrence_bytes(l - prev - 1);
            prev = l;
        });
        len[c] = k;
    }
    prefix_sums(len, lptr_);
    lrow_.assign(static_cast<size_t>(lptr_[N_] + kPad), 0);
#pragma omp parallel for schedule(static)
    for (i64 c = 0; c < N_; ++c) {
        std::uint8_t* p = lrow_.data() + lptr_[c];
        if constexpr (R::bitmap) {
            std::uint8_t* v = p + mask_bytes_;
            std::memset(p, 0, static_cast<size_t>(mask_bytes_));
            data_.each<false>(c, [&](int l, unsigned q) {
                if (!q) return;
                p[l >> 3] = static_cast<std::uint8_t>(p[l >> 3] | (1u << (l & 7)));
                *v++ = static_cast<std::uint8_t>(q & 0xff);
                if constexpr (R::vb == 2) *v++ = static_cast<std::uint8_t>(q >> 8);
            });
        } else {
            int prev = -1;
            data_.each<false>(c, [&](int l, unsigned q) {
                if (q) p = rows.put(p, prev, l, q);
            });
        }
    }
}

// Move the rows, classes and costs to the new order (from: the previous position
// of each), and count the occurrences of every layer in every chunk of
// positions. The order changes little between iterations: the reads are nearly
// in sequence.
void Ranker::relayout(const std::vector<u32>& from) {
    with_rows([&](const auto& rows) { relayout_rows(rows, from); });
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

// spread[x]: byte b is bit b of x
struct Spread {
    u64 v[256];
    Spread() {
        for (int x = 0; x < 256; ++x) {
            v[x] = 0;
            for (int b = 0; b < 8; ++b)
                if (x >> b & 1) v[x] |= u64{1} << (8 * b);
        }
    }
};

template <class R>
void Ranker::relayout_rows(const R& rows, const std::vector<u32>& from) {
    static const Spread spread_table;
    const u64* spread = spread_table.v;
    nptr_.resize(static_cast<size_t>(N_) + 1);
#pragma omp parallel for schedule(static)
    for (i64 p = 0; p < N_; ++p) {
        const u32 q = from[p];
        nptr_[p] = lptr_[q + 1] - lptr_[q];
    }
    prefix_sums_inplace(nptr_);
    if (nrow_.size() != lrow_.size()) nrow_.assign(lrow_.size(), 0);
    chunk_off_.assign(static_cast<size_t>(chunks_) * L_, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < chunks_; ++c) {
        i64* cnt = &chunk_off_[static_cast<size_t>(c) * L_];
        // bitmap rows: the counts of layers 8j .. 8j+7 in the bytes of acc[j], added
        // to cnt every 255 rows
        std::vector<u64> acc(R::bitmap ? static_cast<size_t>(mask_bytes_) : 0, 0);
        int in_acc = 0;
        auto flush = [&]() {
            for (int j = 0; j < static_cast<int>(acc.size()); ++j) {
                for (int b = 0; b < 8 && 8 * j + b < L_; ++b)
                    cnt[8 * j + b] += static_cast<i64>((acc[j] >> (8 * b)) & 0xff);
                acc[j] = 0;
            }
            in_acc = 0;
        };
        for (i64 p = chunk_lo(c); p < chunk_lo(c + 1); ++p) {
            const u32 q = from[p];
            const i64 a = lptr_[q], k = lptr_[q + 1] - a;
            std::uint8_t* d = nrow_.data() + nptr_[p];
            if (k) std::memcpy(d, lrow_.data() + a, static_cast<size_t>(k));
            if constexpr (R::bitmap) {
                for (int j = 0; j < mask_bytes_; ++j) acc[j] += spread[d[j]];
                if (++in_acc == 255) flush();
            } else {
                rows.each(d, d + k, [cnt](int l, unsigned) { ++cnt[l]; });
            }
        }
        if constexpr (R::bitmap) flush();
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
    lrow_.swap(nrow_);
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
    with_rows([&](const auto& rows) { initial_sums_rows(rows, sums); });
}

template <class R>
void Ranker::initial_sums_rows(const R& rows, std::vector<Sum>& sums) const {
    // the first layout is the cell order
#pragma omp parallel for schedule(static)
    for (i64 i = 0; i < N_; ++i) {
        Sum s;
        rows.each(row(i), row_end(i), [&](int l, unsigned q) {
            const double v = f_of(l, static_cast<std::uint16_t>(q)) * wabs_[l];
            (w_[l] >= 0 ? s.pos : s.neg) += v;
        });
        sums[i] = s;
    }
}

// Sort the cells of the current layout by (class, key, cell): item(p, key, cell)
// gives the key and cell at position p.
template <class Make>
void Ranker::sort_by(Make&& item, std::vector<u32>& out, std::vector<u32>& from) {
    if (lcls_.empty()) {
        auto& a = plainbuf_.a;
        a.resize(static_cast<size_t>(N_));
#pragma omp parallel for schedule(static)
        for (i64 p = 0; p < N_; ++p) {
            u64 key;
            u32 cell;
            item(p, key, cell);
            a[p] = {key, cell, static_cast<u32>(p)};
        }
        sort_items(plainbuf_, out, from);
    } else {
        auto& a = sortbuf_.a;
        a.resize(static_cast<size_t>(N_));
#pragma omp parallel for schedule(static)
        for (i64 p = 0; p < N_; ++p) {
            u64 key;
            u32 cell;
            item(p, key, cell);
            a[p] = {key, lcls_[p], cell, static_cast<u32>(p)};
        }
        sort_items(sortbuf_, out, from);
    }
}

// Sort the cells, in the cell order (the layout then), by their sums.
void Ranker::sort_cells_by(const std::vector<Sum>& sums, std::vector<u32>& out, std::vector<u32>& from) {
    sort_by(
        [&](i64 i, u64& key, u32& cell) {
            key = sort_key(modified(sums[i], cost_at(i)).value());
            cell = static_cast<u32>(i);
        },
        out, from);
}

// Sort the cells, given in the current order, by their sums: `by_cell` (sums per
// cell), or the sums of the last iteration (ord_sums_, in the current order).
void Ranker::sort_positions(const std::vector<u32>& order, const std::vector<Sum>* by_cell, std::vector<u32>& out,
                            std::vector<u32>& from) {
    sort_by(
        [&](i64 p, u64& key, u32& cell) {
            cell = order[p];
            key = sort_key(by_cell ? modified((*by_cell)[cell], cost_at(p)).value() : ord_sums_[p].value());
        },
        out, from);
}

// The occurrences of every layer in the current order (occ8_ or occ16_).
void Ranker::scatter() {
    with_rows([&](const auto& rows) { scatter_rows(rows); });
}

template <class R>
void Ranker::scatter_rows(const R& rows) {
    auto& occ = [this]() -> auto& {
        if constexpr (R::vb == 1)
            return occ8_;
        else
            return occ16_;
    }();
    using V = typename std::remove_reference_t<decltype(occ)>::value_type;
    occ.resize(static_cast<size_t>(start_[L_]));
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < chunks_; ++c) {
        std::vector<i64> cur(chunk_off_.begin() + static_cast<i64>(c) * L_,
                             chunk_off_.begin() + static_cast<i64>(c + 1) * L_);
        for (i64 p = chunk_lo(c); p < chunk_lo(c + 1); ++p)
            rows.each(row(p), row_end(p),
                      [&](int l, unsigned q) { occ[cur[l]++] = static_cast<V>(q); });
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
            const i64 o0 = chunk_off_[static_cast<size_t>(c) * L_ + l];
            if (vb_ == 1)
                for (i64 o = o0; o < end; ++o) R = R - static_cast<double>(occ8_[o]) * sc;
            else
                for (i64 o = o0; o < end; ++o) R = R - static_cast<double>(occ16_[o]) * sc;
        }
    }
    prof_(2);
    with_rows([&](const auto& rows) { losses_rows(rows, order, sums); });
    prof_(3);
}

// the losses of each cell along the current order (after the levels)
template <class R>
void Ranker::losses_rows(const R& rows, const std::vector<u32>& order, std::vector<Sum>* sums) {
    switch (set_.rule) {
        case Rule::ABF: losses_rule<Rule::ABF>(rows, order, sums); break;
        case Rule::CAZ1: losses_rule<Rule::CAZ1>(rows, order, sums); break;
        case Rule::CAZ2: losses_rule<Rule::CAZ2>(rows, order, sums); break;
        case Rule::CAZMAX: losses_rule<Rule::CAZMAX>(rows, order, sums); break;
        case Rule::CAZP: losses_rule<Rule::CAZP>(rows, order, sums); break;
        case Rule::CAZPW: losses_rule<Rule::CAZPW>(rows, order, sums); break;
        default: break;  // RAND and LOAD do not iterate
    }
}

// the same for one rule (known when compiled)
template <Rule RULE, class Rows>
void Ranker::losses_rule(const Rows& rows, const std::vector<u32>& order, std::vector<Sum>* sums) {
    // walk every chunk of positions: the losses of each cell, from the levels
    // before and after it, summed in layer order
    constexpr Rule rule = RULE;
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
                rows.each(row(p), row_end(p), [&](int le, unsigned q) {
                    for (; l < le; ++l) add(s, l, absent_[l]);
                    add(s, le, loss(le, static_cast<std::uint16_t>(q)));
                    l = le + 1;
                });
                for (; l < L_; ++l) add(s, l, absent_[l]);
            } else {
                rows.each(row(p), row_end(p),
                          [&](int le, unsigned q) { add(s, le, loss(le, static_cast<std::uint16_t>(q))); });
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
}

// Zonation's mean ranking error of the current order: the terms in parallel,
// their sum in sequence (adding the zeros of the pairs in order changes nothing).
void Ranker::ranking_error(double& err, i64& n_err) {
    // the nonzero terms of each segment, in order, at its start (adding the zeros
    // of the pairs in order would not change the sum)
    auto& terms = work_;
    terms.resize(static_cast<size_t>(N_));
    const int T = std::max(1, std::min(max_threads() * 4, static_cast<int>(N_ / 16384) + 1));
    std::vector<i64> kept(static_cast<size_t>(T), 0);
    i64 tests = 0, errors = 0;
#pragma omp parallel for schedule(static, 1) reduction(+ : tests, errors)
    for (int t = 0; t < T; ++t) {
        const i64 lo = std::max<i64>(1, N_ * t / T), hi = N_ * (t + 1) / T;
        i64 k = lo;
        for (i64 i = lo; i < hi; ++i) {
            if (!lcls_.empty() && lcls_[i - 1] != lcls_[i]) continue;
            const Sum& s1 = ord_sums_[i - 1];
            const Sum& s2 = ord_sums_[i];
            ++tests;
            const double div = s1.pos + s1.neg;
            if (s1.value() > s2.value() && div > 0) {
                ++errors;
                terms[k++] = (s1.value() - s2.value()) / div;
            }
        }
        kept[t] = k - lo;
    }
    double e = 0.0;
    for (int t = 0; t < T; ++t) {
        const i64 lo = std::max<i64>(1, N_ * t / T);
        for (i64 i = lo; i < lo + kept[t]; ++i) e += terms[i];
    }
    n_err = errors;
    err = errors == 0 ? 0.0 : e / static_cast<double>(tests);
}

// the position of every occurrence of every layer (in the order of occ8_/occ16_)
template <class R>
void Ranker::positions_rows(const R& rows, Buffer<u32>& occ_pos) const {
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < chunks_; ++c) {
        std::vector<i64> cur(chunk_off_.begin() + static_cast<i64>(c) * L_,
                             chunk_off_.begin() + static_cast<i64>(c + 1) * L_);
        for (i64 p = chunk_lo(c); p < chunk_lo(c + 1); ++p)
            rows.each(row(p), row_end(p), [&](int l, unsigned) { occ_pos[cur[l]++] = static_cast<u32>(p); });
    }
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
    // the work arrays of the iterations are no longer needed
    for (auto* b : {&sortbuf_.a, &sortbuf_.b}) b->release();
    for (auto* b : {&plainbuf_.a, &plainbuf_.b}) b->release();
    nrow_.release();
    nptr_.release();
    ord_sums_.release();
    work_.release();
    Buffer<u32> occ_pos;
    occ_pos.set_storage(set_.storage);
    occ_pos.resize(static_cast<size_t>(start_[L_]));
    with_rows([&](const auto& rows) { positions_rows(rows, occ_pos); });
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
            while (o < end && static_cast<i64>(occ_pos[o]) < p) {
                const i64 q = static_cast<i64>(occ_pos[o]);
                a += R * static_cast<double>(q + 1 - prev);
                prev = q + 1;
                R = R - f_of(l, static_cast<std::uint16_t>(occ_value(o)));
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
    const bool smooth = iterative && set_.smoothing > 0;
    if (smooth)
        weighted = sums;
    else
        std::vector<Sum>().swap(sums);
    if (iterative) {
        double prev = 0.0;
        const double sf = static_cast<double>(set_.smoothing);
        const double sf1 = static_cast<double>(1.0f - set_.smoothing);
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
            auto& moves = work_;
#pragma omp parallel for schedule(static)
            for (i64 j = 0; j < N_; ++j)
                moves[from[j]] = std::fabs(static_cast<double>(static_cast<i64>(from[j]) - j) * ndr);
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
        data_.each<false>(cell, [&](int le, unsigned q) {
            if (!q) return;
            if (dense_)
                for (; l < le; ++l) add(l, absent_[l]);
            add(le, loss(le, static_cast<std::uint16_t>(q)));
            l = le + 1;
        });
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
    Ranker r(data, settings, false);
    r.flexibility(ranking, max_error, up, down, pos, neg);
}

}  // namespace prio
