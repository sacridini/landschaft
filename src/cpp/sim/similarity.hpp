// SPDX-License-Identifier: GPL-3.0-or-later
//
// Fuzzy similarity of two categorical maps (Hagen 2003) as Dinamica EGO's
// CalcReciprocalSimilarityMap computes it (measured: the code is closed).
// At a cell with a class c in map A, the membership of c in map B is the
// largest f(d) over the cells of B with class c in the window x window
// square around it (0 if none), f(d) = 2^(-d/2) with d the Euclidean
// distance in cells (exponential decay) or 1 (constant), stored as float;
// where B also has a class there, the similarity is the smaller of that and
// the membership of B's class in A. Cells without a class in A have none.
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

struct SimilarityResult {
    std::vector<float> first, second;  // per cell; NaN where the map has no class
    double first_mean = 0, second_mean = 0;
    std::int64_t first_cells = 0, second_cells = 0;
};

// a, b: classes (-1: no class); window odd.
SimilarityResult reciprocal_similarity(std::int64_t ny, std::int64_t nx, const int* a, const int* b, int window,
                                       bool exponential);

}  // namespace sim
