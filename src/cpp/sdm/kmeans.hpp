// SPDX-License-Identifier: GPL-3.0-or-later
//
// R's stats::kmeans() (R 4.4.2): Hartigan-Wong (AS 136, kmns.f), Lloyd / Forgy
// and MacQueen (kmeans.c), with R's random starts (unique rows, sample.int)
// and its choice of the best start: the same clusters, centres and sums of
// squares, bit for bit, and the same use of the random stream as R for the
// same set.seed(). blockCV::cv_cluster() calls
// kmeans(x, k, iter.max = 500, nstart = 25).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nlm/rng.hpp"

namespace sdm {

using i64 = std::int64_t;

enum class KmeansAlgorithm : int { HartiganWong = 1, Lloyd = 2, MacQueen = 3 };  // "Forgy" is Lloyd

struct KmeansResult {
    std::vector<int> cluster;      // m, 1-based
    std::vector<double> centers;   // k x p, column-major (R's matrix)
    double totss = 0;              // sum(scale(x, scale = FALSE)^2)
    std::vector<double> withinss;  // k
    double tot_withinss = 0;       // sum(withinss), R's 80-bit sum
    double betweenss = 0;          // totss - tot_withinss
    std::vector<int> size;         // k
    int iter = 0;
    int ifault = -1;  // -1 for R's NULL (Lloyd and MacQueen without a fault)
    std::vector<std::string> warnings;  // R's warnings, in order
};

// duplicated.matrix(x) of R 4.4.2 for a column-major m x p matrix: true for a
// row equal to an earlier one. Rows are compared as asplit(x, 1) lists, i.e.
// element by element with == (0 equal to -0; NA equal to NA, NaN to NaN).
std::vector<bool> r_duplicated_rows(const double* x, i64 m, i64 p);

// unique.matrix(x): the rows kept (0-based, first occurrences in order).
std::vector<i64> r_unique_rows(const double* x, i64 m, i64 p);

// kmeans(x, centers = k, iter.max, nstart, algorithm) on a column-major m x p
// matrix, drawing the initial centres from `rng` exactly as R does:
// nstart == 1 takes x[sample.int(m, k), ] (unique(x)[sample.int(mm, k), ] if
// those rows repeat), nstart >= 2 takes unique(x)[sample.int(mm, k), ] for
// every start (see r_unique_rows). R's errors are thrown as
// std::invalid_argument (bad arguments) or std::runtime_error ("empty
// cluster: try a better set of initial centers").
KmeansResult kmeans(const double* x, i64 m, i64 p, int k, int iter_max, int nstart, KmeansAlgorithm algorithm,
                    nlm::RRng& rng);

// kmeans(x, centers = <k x p matrix>, iter.max, algorithm): given initial
// centres (column-major), one start.
KmeansResult kmeans(const double* x, i64 m, i64 p, const double* centers, int k, int iter_max,
                    KmeansAlgorithm algorithm);

}  // namespace sdm
