// SPDX-License-Identifier: GPL-3.0-or-later
//
// Multi-action planning (prioriactions 0.5.0, Salgado-Rojas et al. 2020; GPL >= 2): the model and the
// solution's figures, from the inputs as prioriactions's inputData() leaves them (internal ids from 0).
//
// - pairs(): every (feature, unit) pair in prioriactions's order (features, then units ascending) and,
//   for each, the threats in the unit that harm the feature (in the threats' order) with the
//   coefficients of their actions in the pair's benefit row (the response curve's delta1-4);
// - benefit(): rcpp_stats_benefit, the most benefit every feature can get, or a solution's;
// - compile(): problem()'s model (getModelInfo()'s matrices) bit for bit, the activation rows
//   optionally one per action (x_ik <= w_i: the same integer solutions, a tighter relaxation) and the
//   response curves (curve 2, 3) optionally as an explicit piecewise-linear formulation;
// - equivalent(): the model the solvers get for curve 1: activation one per action, the benefit
//   columns b substituted (b_is is its row's right side at an optimum when the coefficients are >= 0
//   and sum to at most 1) and the locks as bounds;
// - costs(), connectivity(): rcpp_stats_costs_* and rcpp_stats_connectivity_* of a solution.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "opt/milp.hpp"

namespace opt::actions {

struct Input {
    int n_units = 0, n_features = 0, n_threats = 0;
    std::vector<double> monitoring_cost, unit_status;           // n_units (status 0, 2, 3)
    std::vector<double> target_recovery, target_conservation;   // n_features
    std::vector<i64> df_unit, df_feature;                       // dist_features rows
    std::vector<double> df_amount;
    std::vector<double> blm_actions;                            // n_threats
    std::vector<i64> act_unit, act_threat;                      // dist_threats rows: the actions
    std::vector<double> act_amount, act_cost, act_status;
    std::vector<i64> sens_feature, sens_threat;                 // sensitivity rows
    std::vector<double> sens_delta;                             // 4 x rows: delta1, ..., delta4
    bool has_boundary = false;
    std::vector<i64> bnd_id1, bnd_id2;                          // boundary rows (internal ids)
    std::vector<double> bnd_value;

    i64 n_actions() const { return static_cast<i64>(act_unit.size()); }
    void check() const;  // sizes and ids; throws std::invalid_argument
};

struct Pairs {
    std::vector<i64> feature, unit;   // per pair
    std::vector<double> amount, sum_alpha;
    std::vector<std::uint8_t> threatened;
    std::vector<i64> start;           // pairs + 1: the pair's (harmful threat) entries
    std::vector<i64> action;          // per entry: the action
    std::vector<double> coef;         // its coefficient in the pair's benefit row
    std::vector<double> contribution; // its benefit per unit amount (-coef)
    std::vector<double> action_status;
    i64 size() const { return static_cast<i64>(feature.size()); }
};

Pairs pairs(const Input& in);

// The boundary matrix of prioriactions (create_boundary_matrix_extended), by rows, columns ascending.
struct Boundary {
    std::vector<i64> start, index;
    std::vector<double> value;
    bool empty() const { return start.empty(); }
    double at(i64 i, i64 j) const;
};
Boundary boundary_matrix(const Input& in);

struct Benefit {
    // most benefit (no solution): dist, dist_threatened, conservation, recovery (each n_features)
    std::vector<double> dist, dist_threatened, conservation, recovery;
};
// solution: units then actions (empty: the most benefit)
Benefit benefit(const Input& in, const Pairs& p, const std::vector<double>& solution);

struct Settings {
    bool maximize = false;  // maximizeBenefits
    double blm = 0.0;
    int curve = 1;
    int segments = 1;
    double step = 0.0;  // the pieces' length (round(1 / segments, 1)); 0: 1 / segments
    double budget = 0.0;
    // directional dependencies between units (upstream -> downstream; an extension of prioriactions,
    // after prioripy): monitoring dep_to without dep_from costs dep_weight (max(0, w_to - w_from) in the
    // objective), or is forbidden (dep_hard: w_to <= w_from)
    std::vector<i64> dep_from, dep_to;
    std::vector<double> dep_weight;
    std::vector<std::uint8_t> dep_hard;
};

enum Sense : std::int8_t { kLe = 0, kGe = 1, kEq = 2 };

struct Compiled {
    Model model;
    std::vector<std::int8_t> sense;
    std::vector<double> rhs;
    i64 n_pairs = 0;
    i64 first_pow = -1;  // curve != 1: the first column of b^curve
    // equivalent() only: the closure engine's view (opt/closure.hpp) of the model, a minimisation of
    // every unit's and action's own term plus w z_i (1 - z_j) for the boundary pairs, with the targets
    // or the budget from side_row0 on; every column after the units and actions is the product of two
    std::vector<double> node_cost;
    std::vector<i64> pen_i, pen_j;
    std::vector<double> pen_w;
    std::vector<i64> prod_a, prod_b;
    std::vector<std::uint8_t> prod_neg;  // the column is x_a (1 - x_b) instead of x_a x_b
    std::vector<i64> req_from, req_to;   // unit requirements (hard dependencies) besides action -> unit
    i64 side_row0 = 0;
};

Compiled compile(const Input& in, const Pairs& p, const Settings& s, bool disaggregate, bool curves);

// The equivalent model for solving (curve 1 only; empty model when the elimination does not apply:
// a negative coefficient or a pair's coefficients summing to more than 1). Columns: units, actions,
// then the boundary's y and p columns.
Compiled equivalent(const Input& in, const Pairs& p, const Settings& s);

// Solves the problem (curve 1): the equivalent model on the closure engine (Lagrangian bound, repaired
// selections), then, while the gap is wider than asked, on HiGHS's MIP from the engine's solution, the
// units and actions the bound proves fixed at their value (with a time limit only when at most 1e5
// columns are left free). x: the units, then the actions. Without the equivalent model (a coefficient
// < 0 or summing to more than 1 in a pair) or with engine false, HiGHS on the equivalent or the full
// model.
SolveResult solve(const Input& in, const Pairs& p, const Settings& s, const SolveOptions& o, bool engine = true);

// Every pair's benefit (b) for a solution of units and actions: min(1, sum of its actions'
// contributions) or the unit's value.
std::vector<double> local_benefit(const Pairs& p, i64 n_units, const std::vector<double>& x);

// (monitoring, every threat's actions' cost)
std::vector<double> costs(const Input& in, const std::vector<double>& units, const std::vector<double>& actions);

// (units, every threat's actions' boundary): prioriactions's getConnectivityPenalty() before rounding
std::vector<double> connectivity(const Input& in, const std::vector<double>& units,
                                 const std::vector<double>& actions);

}  // namespace opt::actions
