// SPDX-License-Identifier: GPL-3.0-or-later
//
// The decision rules of CLUE-S (Verburg et al. 2002) as lulcc 1.0.4 applies
// them at every step of a simulation: the conversions a cell may make (the
// conversion matrix with its time and demand codes), the conversions forced
// on cells whose class is no longer allowed, and the neighbourhood
// thresholds. Matrices of cells x classes are row-major (cell i, class j at
// i * ncode + j); 1 marks an allowed conversion and NaN one that is not.
#pragma once

#include <cstdint>
#include <vector>

namespace sim {

// lulcc's allow(): x the class of each cell (a value of `classes`), hist the
// steps it has had it, cd the change of each class's demand over the step,
// rules the ncode x ncode conversion matrix (rows the current class): 1
// allowed, 0 not; -1 only while the demand of the current class decreases;
// -2 only while the demand of the next class increases; 100 + t only while
// the cell has had its class for less than t steps; 1000 + t only once it
// has had it for t steps or more. Throws when a cell's class is unknown.
std::vector<double> allow_matrix(const double* x, const std::int64_t* hist, std::int64_t ncell, const double* classes,
                                 int ncode, const double* cd, const std::int64_t* rules);

// lulcc's autoconvert: the class each cell is forced to take this step (NaN:
// none). A cell whose own class is no longer allowed (NaN suitability) takes
// the class of highest suitability (the first in case of ties; its own class
// when no suitability is above -1); a restricted cell (mask 0) keeps its
// class. prob: ncell x ncode.
std::vector<double> auto_convert(const double* x, const double* prob, std::int64_t ncell, const double* classes,
                                 int ncode, const std::int64_t* mask);

// A neighbourhood rule: a class may go only where the focal mean of its
// presence (weights wy x wx, odd sides) reaches the threshold.
struct NeighbourhoodRule {
    double cls = 0.0;
    double threshold = 0.0;
    int wy = 3, wx = 3;
    std::vector<double> weights;  // wy x wx, row-major
};

// lulcc's neighbourhood rules: ny * nx cells x ncode, 1 where each class may
// go and NaN where its rule forbids it (the focal mean of its presence over
// the cells with a value is below the threshold, or there is none). Rules of
// classes not in `classes` are skipped; a later rule of the same class
// replaces an earlier one.
std::vector<double> neighbourhood_allow(const double* landscape, std::int64_t ny, std::int64_t nx,
                                        const double* classes, int ncode,
                                        const std::vector<NeighbourhoodRule>& rules);

}  // namespace sim
