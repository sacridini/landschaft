// SPDX-License-Identifier: GPL-3.0-or-later
//
// The Patcher of a continuous state (Dinamica EGO's PatcherContinuousState,
// clean-room from its documentation and black-box runs of Dinamica EGO 8.13):
// new patches that take the stock of their cells (biomass, carbon, timber)
// until a quota is met.
//
//  - Candidates: the cells with stock > 0 and probability > 0, ranked from
//    the highest probability; the first ones kept while their summed stock
//    is at most prune_factor x quota (measured).
//  - Seeds: drawn from that list without replacement in proportion to P
//    (measured: the mean probability of the chosen cells, 32 x 3 draws of 20
//    among 100 cells of P 0.01-1, is 0.653 against 0.657 for P, 0.737 for
//    P^2, 0.506 uniform; low probabilities never leave the quota unmet, so
//    there is no acceptance test). The list is not ranked again when it runs
//    out (the quota is left).
//  - Each cell of a patch gives all its stock; the patch and the allocation
//    stop as soon as the quota is met (the last cell may exceed it: the
//    remaining quota is then negative, as Dinamica's).
//  - Patch size: lognormal with the given mean and variance (cells); growth
//    as the Patcher's (sim/patches.hpp): a neighbour (window) with stock and
//    probability, drawn in proportion to P x isometry^k, k the patch cells
//    in its window (Dinamica's functor crashes with patches of more than one
//    cell, so this part could not be measured).
//  - The cells taken have stock and probability 0 afterwards.
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

struct ContinuousResult {
    std::vector<double> taken;           // stock taken from each cell (0 elsewhere)
    std::vector<std::int32_t> patch;     // patch number of each cell taken (1, 2, ...; 0 elsewhere)
    double remaining = 0;                // quota left (negative when exceeded)
    std::int64_t patches = 0, cells = 0;
};

// stock: per cell, in the quota's units (NaN or <= 0: none); probability per
// cell; patch sizes in cells; window of odd sides.
ContinuousResult continuous_patches(std::int64_t ny, std::int64_t nx, const double* stock, const double* probability,
                                    double quota, double mean, double variance, double isometry,
                                    double prune_factor, int win_rows, int win_cols, std::uint64_t seed);

}  // namespace sim
