// SPDX-License-Identifier: GPL-3.0-or-later
//
// CARS, the cellular automaton of PLUS based on multi-type random patch
// seeds (Liang et al. 2021, Computers, Environment and Urban Systems 85,
// 101569), written from the paper (PLUS is distributed without source or
// licence). Each iteration, for every cell i and class k:
//
//   OP(i,k) = P(i,k) x Omega(i,k) x D(k)                     (Eq. 2)
//   Omega(i,k) = con(k) / (n x n - 1) x w(k)                 (Eq. 3)
//
// con(k) the cells of class k in the n x n window at the end of the last
// iteration (the centre left out), w(k) the class's neighbourhood weight.
// Where Omega(i,k) = 0, a random seed: OP = P x (r x mu) x D when r < P,
// r uniform in [0, 1), mu(k) the class's patch generation threshold (Eq. 5). A roulette
// wheel over OP picks a class c; when c is not the cell's class k, the cell
// changes if P(i,c) > tau and the transition matrix allows k -> c, with
// tau = delta^l x r, r normal of mean 1 in [0, 2] (standard deviation 1/3
// here: the paper gives none) and l raised by one each iteration that
// brought the total gap down by less than `step` cells (Eqs. 6-7). A class
// gains no cell once it has its demand and (our choice: the paper says
// nothing) loses none while it has no more than its demand. The adaptive coefficient D(k)
// starts at 1 and follows the gaps G = demand - cells of the last two
// iterations (Eq. 4): unchanged while |G| does not grow, times G(t-1) /
// G(t-2) when a shortage grows, times G(t-2) / G(t-1) when an excess grows.
// The iterations stop when every class has its demand.
//
// The randomness of a cell in an iteration comes from a hash of (seed,
// iteration, cell), so the result does not depend on the number of threads;
// the changes of an iteration are applied in a random order (the caps).
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

struct CarsSettings {
    int window = 3;             // n
    double decay = 0.9;         // delta
    double step = 500;          // cells of gap reduction below which tau decays
    std::int64_t max_iter = 5000;
    std::uint64_t seed = 1;
};

struct CarsResult {
    std::vector<int> landscape;               // class index per cell (-1 kept)
    std::int64_t iterations = 0;
    bool converged = false;
    std::int64_t decays = 0;                  // l at the end
    std::vector<std::int64_t> trace;          // (iteration, region x class) cells, iteration 0 = the start
    std::vector<std::int64_t> trace_decays;   // l after each iteration
    std::vector<double> trace_d;              // (iteration, region x class) D after each iteration
};

// land: class index of each cell (0 .. k-1), -1 for cells outside the model
// (no data, classes left out); fixed: cells that keep their class (may be
// null); region: region index of each cell (0 .. nreg - 1; -1 outside, the
// cell keeps its class; null: one region); prob: (k, cells) growth
// probabilities; weights, seeds (mu): k; allowed: (k, k) row the current
// class; demand: (nreg, k) cells of each class in each region. Demand, gaps
// and D are per region and class; trace and trace_d are (iteration,
// nreg x k).
CarsResult cars_run(std::int64_t ny, std::int64_t nx, int k, const int* land, const std::uint8_t* fixed,
                    const int* region, int nreg, const double* prob, const double* weights, const double* seeds, const std::uint8_t* allowed,
                    const std::int64_t* demand, const CarsSettings& s);

}  // namespace sim
