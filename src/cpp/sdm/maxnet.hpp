// SPDX-License-Identifier: GPL-3.0-or-later
//
// MaxNet: R's maxnet 0.1.4 (Phillips et al. 2017), Maxent fitted as an
// infinitely weighted logistic regression through glmnet's lasso path
// (sdm/lognet.hpp). The features (linear, quadratic, hinge, threshold,
// product, categorical), the default regularisation, the lambda sequence,
// the choice of the last lambda, the normalising constant (alpha), the
// entropy and the four prediction types, with R's bits on x86-64.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sdm {
namespace maxnet {

using i64 = std::int64_t;

enum class Kind : std::int32_t { Linear = 0, Quadratic = 1, Hinge = 2, Threshold = 3, Categorical = 4, Product = 5 };

// One column of maxnet's model matrix.
struct Feature {
    Kind kind = Kind::Linear;
    std::int32_t var = 0;   // continuous variable (factor for Categorical)
    std::int32_t var2 = 0;  // Product: the second variable; Categorical: the level
    double k1 = 0, k2 = 0;  // knots: hinge (x - k1) / (k2 - k1); threshold x >= k1
    // The knots as predict.maxnet() reads them back from the column names,
    // which R writes with as.character() (15 significant digits).
    double pk1 = 0, pk2 = 0;
    std::string s1, s2;  // those names of the knots
};

// Predictors, column-major.
struct Table {
    i64 n = 0;
    std::int32_t n_cont = 0, n_cat = 0;
    const double* cont = nullptr;          // n x n_cont
    const std::int32_t* cat = nullptr;     // n x n_cat: level 0.., -1 missing, -2 a level unseen in training
    std::vector<std::int32_t> n_levels;    // levels of each factor
};

struct Settings {
    std::string classes = "default";  // "default" or letters among l, q, p, h, t
    double regmult = 1.0;
    bool add_samples_to_background = true;
    i64 maxit = 100000;
};

struct Model {
    std::string classes;  // the classes used ("default" resolved)
    std::vector<Feature> features;  // every column of the model matrix
    std::vector<double> penalty;    // penalty.factor
    std::vector<double> lambda;     // the 200 lambdas
    std::vector<double> feature_min, feature_max;  // over the training model matrix
    std::vector<double> var_min, var_max;          // continuous variables
    std::vector<double> sample_means;              // continuous variables: mean over the presences
    std::vector<std::int32_t> majority;            // factors: most frequent level among the presences
    std::vector<std::int32_t> beta_index;          // columns with a nonzero coefficient
    std::vector<double> betas;
    double alpha = 0, entropy = 0;
    i64 n_passes = 0, n_added = 0, n_presences = 0, n_rows = 0;
};

Model fit(const Table& data, const double* p, const Settings& settings);

enum class Output : std::int32_t { Link = 0, Exponential = 1, Cloglog = 2, Logistic = 3 };

// What predict.maxnet() needs: the features with nonzero coefficients.
struct Predictor {
    std::vector<Feature> features;
    std::vector<double> betas, feature_min, feature_max;  // per feature
    std::vector<double> var_min, var_max;                 // per continuous variable
    double alpha = 0, entropy = 0;
};

Predictor predictor(const Model& m);

// predict.maxnet(type, clamp) for every row of `data` (NaN where a variable
// the model uses is missing; R drops those rows).
void predict(const Predictor& m, const Table& data, bool clamp, Output type, double* out);

// R's as.character() of a double (15 significant digits, fixed or
// scientific notation, whichever is narrower).
std::string r_as_character(double x);

}  // namespace maxnet
}  // namespace sdm
