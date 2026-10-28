// SPDX-License-Identifier: GPL-3.0-or-later
//
// Statistics of maps, as Terraflux's stats.* computes them: the monotonic trend of
// every cell along a series (the Mann-Kendall test and Sen's slope, pymannkendall's
// original_test, the same bits), and the spatial autocorrelation of a map over the
// contiguity of its cells (global and local Moran's I, Getis and Ord's Gi*, as
// PySAL's esda to rounding), with their permutation tests.
#pragma once

#include <cstdint>
#include <vector>

namespace stats {

using i64 = std::int64_t;
using u64 = std::uint64_t;

// The Mann-Kendall test and Sen's slope of a series (NaN missing).
struct MannKendall {
    double s, var_s, z, p, tau, slope, intercept, h, trend;
};
MannKendall mann_kendall(const double* series, int t, double alpha);

// Every cell's trend along ``t`` layers of ``n`` cells (t x n): out is 6 x n
// (slope, intercept, p, z, tau, trend), NaN where a cell has fewer than
// ``min_values`` values.
void trend(const double* layers, int t, i64 n, double alpha, int min_values, double* out, int threads);

// The cells with a value of an ny x nx map and the contiguity of each among them
// (rook: 4 neighbours, queen: 8), by their index among the cells with a value:
// the neighbours of k are nb[start[k] .. start[k + 1]).
struct Lattice {
    std::vector<i64> cells;
    std::vector<i64> start;
    std::vector<i64> nb;
    i64 size() const { return static_cast<i64>(cells.size()); }
    i64 count(i64 k) const { return start[k + 1] - start[k]; }
};
Lattice lattice(const double* data, i64 ny, i64 nx, bool queen);

// The global Moran's I (row-standardised weights) and its inference.
struct GlobalMoran {
    double i, ei, vi_norm, z_norm, p_norm, vi_rand, z_rand, p_rand;
};
GlobalMoran global_moran(const double* y, const Lattice& l);
// The pseudo p of ``perms`` random permutations of the values (numpy's
// default_rng(seed).shuffle, again and again), NaN without permutations.
double moran_p_sim(const double* y, const Lattice& l, double observed, int perms, u64 seed, int threads);

// The local Moran's I, its quadrant (1 high-high, 2 low-high, 3 low-low, 4
// high-low), its expectation, variance and z under randomisation, and the pseudo
// p of conditional permutations (a generator per cell; NaN without neighbours or
// permutations). Each output has l.size() values.
void local_moran(const double* y, const Lattice& l, int perms, u64 seed, double* i, double* quadrant, double* ei,
                 double* vi, double* z, double* p, int threads);

// Getis and Ord's Gi* (the cell among its neighbours, row-standardised), its z,
// its one-sided p and the hot (1 to 3) and cold (-1 to -3) spots at 90, 95, 99 %.
void getis_ord(const double* y, const Lattice& l, double* g, double* z, double* p, double* spots, int threads);

}  // namespace stats
