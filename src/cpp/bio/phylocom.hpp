// SPDX-License-Identifier: GPL-3.0-or-later
//
// PhyloCom indices (Biodiverse::Indices::PhyloCom): mean pairwise and
// nearest taxon distances (MPD, MNTD), their expected values under random
// draws of tips (NRI, NTI and net VPD), and the Mersenne Twister Biodiverse
// samples with (Math::Random::MT::Auto, 64-bit build).
//
// Biodiverse caches path lengths across the whole analysis. On ultrametric
// trees a length is twice the path from the first label of the first pair
// queried below each last common ancestor, and the cache fills the pairs of
// adjacent children (by terminal count) with it, so values depend on the order
// groups are processed in: PhyloCom keeps that state, and runs sequentially
// when it matters.

#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <vector>

#include "bio/phylo.hpp"

namespace bio {

class MT64 {
public:
    explicit MT64(const std::vector<std::uint64_t>& seed);
    std::uint64_t irand();
    // Math::Random::MT::Auto::rand: [0, 1)
    double rand() { return static_cast<double>(irand() >> 12) * 2.220446049250313080847263336181640625e-16; }
    // Math::Random::MT::Auto::binomial (prob, trials)
    std::int64_t binomial(double prob, std::int64_t trials);
    // state vector, position and count left (for saving and resuming)
    std::vector<std::uint64_t> get_state() const;
    void set_state(const std::vector<std::uint64_t>& s);
    // Math::Random::MT::Auto::shuffle (Fisher-Yates from the last element)
    template <class T>
    void shuffle(std::vector<T>& v) {
        for (std::int64_t ii = static_cast<std::int64_t>(v.size()) - 1; ii > 0; --ii) {
            const std::uint64_t jj = irand() % static_cast<std::uint64_t>(ii + 1);
            std::swap(v[static_cast<std::size_t>(ii)], v[static_cast<std::size_t>(jj)]);
        }
    }

private:
    static constexpr int N = 312, M = 156;
    std::uint64_t state_[N];
    int next_ = 0;
    std::int64_t left_ = 1;
    std::uint64_t algo();
    // binomial's cached terms
    std::int64_t b_trials_ = 0;
    double b_term_ = 0, b_prob_ = 0, b_plog_ = 0, b_pclog_ = 0;
};

struct MpdStats {
    static constexpr double U = std::numeric_limits<double>::quiet_NaN();
    double mean = U, variance = U, rmsd = U, n = U, min = U, max = U;
};

struct MpdResult {
    MpdStats s[2];  // 0: nearest taxon (PNTD), 1: pairwise (PMPD)
};

struct NriNti {
    static constexpr double U = std::numeric_limits<double>::quiet_NaN();
    double nri_mean = U, nri_sd = U, nti_mean = U, nti_sd = U, n = U;
    double vpd_mean = U, vpd_sd = U, vpd_n = U;
};

class PhyloCom {
public:
    PhyloCom(const TreeView& t, bool vpd, std::uint64_t seed);

    bool ultrametric = false;
    bool multifurcating = false;

    // path length between two nodes, the first being Biodiverse's label1
    double path(i32 a, i32 b);

    // _calc_phylo_mpd_mntd: nodes (name order) with non-zero weights, the
    // number of labels on the tree, and the mode (0 all, 1 means, 2 means and
    // variances)
    MpdResult mpd_mntd(const std::vector<i32>& nodes, const std::vector<double>& wts, bool use_wts, int mode,
                       std::size_t n_on_tree);

    // _calc_nri_nti_expected_values for k labels on the tree (cache as Biodiverse)
    const NriNti* expected_step(std::size_t k);
    const NriNti* cached(std::size_t k) const;

    // Bifurcating ultrametric trees: a cached length never changes once set, by
    // the first label (in name order) below the ancestor of the first set that
    // spans it. touch() sets them for a label set as its MPD would; after
    // freeze() lengths are read only, from any thread.
    void touch(const std::vector<i32>& nodes);
    void freeze() { frozen_ = true; }
    bool parallel_sampling = false;

private:
    const TreeView& t_;
    bool vpd_;
    bool frozen_ = false;
    std::vector<i32> stamp_, visit_child_;
    std::vector<double> visit_cum_;
    i32 stamp_id_ = 0;
    MT64 prng_;
    i32 s_ = 0;                        // terminals
    std::vector<i32> tcount_;          // terminals below each node
    std::vector<i32> terminals_;       // name order
    std::vector<i32> sib_rank_;        // position among siblings sorted by terminal count
    std::vector<double> um_fill_;      // ultrametric path cache per last common ancestor
    std::map<std::size_t, NriNti> cache_;
    std::vector<double> lgamma_{0.0, 0.0};
    double nri_mean_ = 0;
    bool have_tce_ = false;
    double sum_tcuu_ = 0, sum_tce_ = 0;
    bool have_nn_ = false;
    double mean_nn_ = 0;
    std::map<i32, double> nti_mean_cache_;

    NriNti compute_expected(std::size_t k);
    double nri_expected_mean();
    double nri_expected_sd(i32 r);
    double nti_expected_mean(i32 r);
    double nti_expected_sd(i32 r);
    double mean_nearest_neighbour_distance();
    const std::vector<double>& lgamma(i32 n);
};

}  // namespace bio
