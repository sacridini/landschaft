# SPDX-License-Identifier: GPL-3.0-or-later
"""Management zones against prioritizr 8.0.6 (tests/optimize_cases.py, group ``zones``).

- the compiled model is prioritizr's ``compile()``, bit for bit (the easy form with contributions and
  targets summed over the zones is prioritizr's manual targets with a list of zones);
- the optimum (gap 0) is prioritizr's on Gurobi; the solution maps back to the zones.
"""

from __future__ import annotations

import numpy as np
import optimize_cases
import pandas as pd
import pytest
from optimize_reference import case_ids, model, result

import landschaft as ls

ZONE_CASES = [c for c in case_ids("zones") if optimize_cases.CASES[c].py is not None]
# contiguity: prioritizr's formulation is a restriction, ours exact (not the same model; the optimum <= theirs)
EXACT = [c for c in ZONE_CASES if "contiguity" not in c]
# boundary penalties and contiguity take minutes on HiGHS's MIP (even to a 1% gap): a time limit
SLOW = ("zones_boundary", "zones_boundary_matrix", "zones_contiguity")
SLOW_SECONDS = 15


def _assert_reference_optimum(ours: float, ref: float, sense: str) -> None:
    """The reference's optimum (Gurobi, gap 0); in the max objectives the cost only breaks ties and Gurobi
    stops within its tolerances: ours may be better by up to 1e-5, never worse (as test_prioritize_milp)."""
    if ours == pytest.approx(ref, rel=1e-9, abs=1e-9):
        return
    better = ours > ref if sense == "max" else ours < ref
    assert better and abs(ours - ref) <= 1e-5 * max(1.0, abs(ref)), (ours, ref)


def _assert_same_model(ours, ref) -> None:
    keep = ref.A_x != 0
    order = np.lexsort((ref.A_j[keep], ref.A_i[keep]))
    a = ours.A.tocoo()
    np.testing.assert_array_equal(a.row, ref.A_i[keep][order], err_msg="A_i")
    np.testing.assert_array_equal(a.col, ref.A_j[keep][order], err_msg="A_j")
    np.testing.assert_array_equal(a.data, ref.A_x[keep][order], err_msg="A_x")
    assert ours.A.shape == (len(ref.rhs), len(ref.obj))
    for key in ("obj", "lb", "ub", "rhs"):
        np.testing.assert_array_equal(getattr(ours, key), getattr(ref, key), err_msg=key)
    assert list(ours.sense) == ref.sense
    assert list(ours.vtype) == ref.vtype
    assert ours.modelsense == ref.modelsense
    assert list(ours.row_ids) == ref.row_ids and list(ours.col_ids) == ref.col_ids


@pytest.mark.parametrize("name", EXACT)
def test_compile_is_prioritizr(name):
    _assert_same_model(optimize_cases.CASES[name].py().compile(), model("zones", name))


@pytest.mark.parametrize("name", ZONE_CASES)
def test_solve_reaches_prioritizr_optimum(name):
    p = optimize_cases.CASES[name].py()
    slow = name in SLOW
    sol = p.add_highs_solver(gap=0, time_limit=SLOW_SECONDS if slow else None).solve()
    ref = result("zones", name)
    kind = p.objective[0]
    r = ref["objective"][0]
    if slow:  # within the time limit: the bound below prioritizr's optimum; ours no better than it (but with
        # contiguity, exact here, a restriction there)
        assert sol.status in ("optimal", "time_limit")
        assert sol.bound <= r * (1 + 1e-9) + 1e-9
        if "contiguity" not in name:
            assert sol.objective >= r * (1 - 1e-9) - 1e-9
    else:
        assert sol.status == "optimal"
        _assert_reference_optimum(sol.objective, r, "max" if kind.startswith("max") else "min")
    if kind == "min_set" and not slow and not any(sp_[0] == "penalty" for sp_ in p.spatial):
        assert sol.cost == pytest.approx(r, rel=1e-9)
    if kind == "min_set":
        assert sol.coverage["met"].all()
    else:  # within the budget, or every zone's
        budget = np.atleast_1d(p.objective[1])
        spent = np.array(list(sol.zone_costs.values())) if budget.size > 1 else np.array([sol.cost])
        assert (spent <= budget * (1 + 1e-9)).all()
    v = sol.values
    assert v.shape == (p.units.n, len(p.zones))
    assert (v.sum(axis=1) <= 1 + 1e-9).all()
    closed = ~np.isfinite(p.units.cost.reshape(len(p.zones), -1).T)
    assert (v[closed] == 0).all()
    if p.mandatory:
        assert np.allclose(v.sum(axis=1), 1.0)


def test_gurobi_reaches_the_same_optimum():
    gp = pytest.importorskip("gurobipy")
    p = optimize_cases.CASES["zones_minset_sum"].py()
    try:
        sol = p.add_gurobi_solver(gap=0, n_threads=1).solve()
    except gp.GurobiError as e:  # no licence
        pytest.skip(str(e))
    assert sol.objective == pytest.approx(result("zones", "zones_minset_sum")["objective"][0], rel=1e-9)


def test_solution_maps_back_to_the_zones():
    p = optimize_cases.CASES["zones_minset_sum"].py()
    sol = p.solve()
    assert sol.zone_names == optimize_cases.ZONES
    sel = sol.selection
    assert sel.dims == ("zone", "y", "x") and list(sel["zone"].values) == list(optimize_cases.ZONES)
    cost = optimize_cases._zone_costs()
    units = np.isfinite(cost).any(axis=0)
    # the zone map: the zone's number (1..3), 0 for none, NaN outside the planning units
    z = sol.zones.values
    assert np.isnan(z[~units]).all() and np.isfinite(z[units]).all()
    v = sol.values
    code = np.where(v.max(axis=1) > 0, v.argmax(axis=1) + 1, 0)
    assert np.array_equal(z.ravel()[optimize_cases._zone_units()], code)
    assert sol.zones.attrs["zones"] == {1: "protect", 2: "use", 3: "restore"}
    # a cell without a cost in a zone is NaN in that zone's layer
    assert np.isnan(sel.values[np.isnan(cost)]).all()
    assert set(sol.coverage["zone"]) == {"protect + use + restore"}


def test_table_solution_columns():
    p = optimize_cases.CASES["zones_table"].py()
    sol = p.solve()
    t = sol.selection
    for z in optimize_cases.ZONES:
        assert f"solution_{z}" in t.columns
    assert "zone" in t.columns
    picked = t["zone"].notna()
    assert picked.sum() == int(np.round(sol.values.sum()))


def test_easy_and_general_forms_are_the_same_model():
    """The contributions build every zone's features as prioritizr's zones() of the products would."""
    easy = optimize_cases._zone_easy().add_min_set_objective().add_relative_targets(0.2, zone="use")
    general = optimize_cases._zone_general().add_min_set_objective().add_relative_targets(0.2, zone="use")
    a, b = easy.compile(), general.compile()
    assert (a.A != b.A).nnz == 0 and np.array_equal(a.rhs, b.rhs) and np.array_equal(a.obj, b.obj)


def test_targets_of_zones_add_up():
    p = optimize_cases._zone_easy().add_min_set_objective().add_relative_targets(0.3)
    q = p.add_relative_targets(0.1, zone="protect").add_absolute_targets([0.5] * 5, zone=["use", "restore"])
    m = q.compile()
    assert m.row_ids.count("spp_target") == 15
    # the same zones again: replaced, not added
    r = q.add_relative_targets(0.2, zone="protect")
    assert r.compile().row_ids.count("spp_target") == 15
    t = r.solve().coverage
    assert list(t["zone"][:5]) == ["protect + use + restore"] * 5
    assert list(t["zone"][5:10]) == ["protect"] * 5 and list(t["zone"][10:]) == ["use + restore"] * 5
    # a zone's relative target is a share of the feature's total in that zone
    tot = p.units.totals[:, 0]
    assert np.allclose(t["absolute_target"][5:10], 0.2 * tot)


def test_relative_targets_summed_over_zones_are_shares_of_the_base_features():
    p = optimize_cases._zone_easy().add_min_set_objective().add_relative_targets(0.3)
    f = optimize_cases._features().reshape(5, -1)
    totals = np.array([np.cumsum(np.nan_to_num(v))[-1] for v in f])
    assert np.array_equal(p.compile().rhs[:5], 0.3 * totals)


def test_contribution_per_feature():
    base = optimize_cases._zone_easy()
    w = np.array([1.0, 0.5, 0.25, 0.75, 0.0])
    p = ls.prioritize.problem(optimize_cases.zone_cost(), optimize_cases.features(), contribution={"use": w})
    rij_use = p.units.zone_rij[1]
    expect = base.units.zone_rij[0].multiply(w[:, None]).tocsr()
    expect.eliminate_zeros()
    assert (rij_use != expect).nnz == 0


def test_closed_zones_stay_closed():
    cells = optimize_cases._zone_units()
    use = optimize_cases._zone_costs()[1].ravel()[cells]
    j = int(np.flatnonzero(np.isnan(use))[0])
    p = optimize_cases._zone_easy().add_min_set_objective().add_relative_targets(0.3).add_proportion_decisions()
    pu = cells[j] + 1.0
    m = p.add_manual_bounded_constraints(pd.DataFrame({"pu": [pu], "zone": ["use"], "lower": [0.0],
                                                       "upper": [1.0]})).compile()
    assert m.ub[p.units.n + j] == 0.0  # prioritizr would open it (with cost 0)
    with pytest.raises(ValueError, match="no cost"):
        p.add_manual_bounded_constraints(pd.DataFrame({"pu": [pu], "zone": ["use"], "lower": [0.5],
                                                       "upper": [1.0]})).compile()
    # a raster lock into a zone without cost is ignored there (prioritizr's rasters)
    lock = np.zeros((3, optimize_cases.NY, optimize_cases.NX))
    lock[1].ravel()[cells[j]] = 1.0
    m = p.add_locked_in_constraints(optimize_cases.grid(lock, names=optimize_cases.ZONES)).compile()
    assert m.lb[p.units.n + j] == 0.0 and m.ub[p.units.n + j] == 0.0


def test_errors():
    p = optimize_cases._zone_easy().add_min_set_objective()
    with pytest.raises(ValueError, match="unknown zone"):
        p.add_relative_targets(0.3, zone="forest")
    with pytest.raises(ValueError, match="features x zones"):
        p.add_relative_targets(np.ones((5, 2)))
    with pytest.raises(ValueError, match="mandatory allocation needs"):
        ls.prioritize.problem(optimize_cases.pu_cost(), optimize_cases.features()).add_mandatory_allocation_constraints()
    with pytest.raises(ValueError, match="more than one zone"):
        lock = {"protect": np.array([0, 1]), "use": np.array([1])}
        p.add_relative_targets(0.3).add_locked_in_constraints(lock).compile()
    with pytest.raises(ValueError, match="not supported with zones"):
        p.add_relative_targets(0.3).add_feature_contiguity_constraints().compile()
    with pytest.raises(ValueError, match="zones are given but the problem has none"):
        ls.prioritize.problem(optimize_cases.pu_cost(), optimize_cases.features()).add_boundary_penalties(1, zones="same")
    with pytest.raises(ValueError, match="symmetric"):
        p.add_relative_targets(0.3).add_boundary_penalties(1.0, zones=optimize_cases._ZM_A).compile()
    with pytest.raises(ValueError, match="contribution: unknown zones"):
        ls.prioritize.problem(optimize_cases.zone_cost(), optimize_cases.features(), contribution={"x": 1})
    with pytest.raises(TypeError, match="mapping zone"):
        p.add_relative_targets(0.3).add_locked_in_constraints(optimize_cases.locked_in()).compile()


ONE_ZONE = {
    "min_set": lambda p: p.add_min_set_objective().add_relative_targets(0.3),
    "min_shortfall": lambda p: p.add_min_shortfall_objective(2500).add_relative_targets(0.4),
    "max_utility": lambda p: p.add_max_utility_objective(2500),
    "max_cover": lambda p: p.add_max_cover_objective(1500),
    "max_features": lambda p: p.add_max_features_objective(2000).add_relative_targets(0.3),
    "max_phylo_div": lambda p: p.add_max_phylo_div_objective(2000, optimize_cases.TREE_SHUFFLED)
    .add_relative_targets(0.3),
    "min_extinction_risk": lambda p: p.add_min_extinction_risk_objective(2500),
}


@pytest.mark.parametrize("objective", list(ONE_ZONE))
def test_one_zone_is_the_problem_without_zones(objective):
    """A problem with a single zone is the problem without zones plus a row per unit that changes nothing."""
    cost, feats = optimize_cases.pu_cost(), optimize_cases.features()
    plain = ONE_ZONE[objective](ls.prioritize.problem(cost, feats)).add_highs_solver(gap=0)
    zoned = ONE_ZONE[objective](ls.prioritize.problem({"only": cost}, feats)).add_highs_solver(gap=0)
    a, b = plain.compile(), zoned.compile()
    assert b.nrow == a.nrow + plain.units.n and b.ncol == a.ncol
    assert np.array_equal(a.obj, b.obj) and np.array_equal(a.lb, b.lb) and np.array_equal(a.ub, b.ub)
    # the same optimum, up to the small cost term that only breaks ties in the max objectives (the engines'
    # and HiGHS's MIP stop within their tolerances there)
    assert plain.solve().objective == pytest.approx(zoned.solve().objective, rel=1e-5, abs=1e-12)


def test_subregional_targets_with_zones():
    regions = np.where(np.arange(optimize_cases.NX)[None, :] < 7, 1.0, 2.0) * np.ones((optimize_cases.NY, 1))
    t = pd.DataFrame({"region": [1, 2, 2], "feature": ["f1", "f4", "f5"], "target": [0.3, 0.2, 0.1],
                      "zone": ["protect", ["protect", "use", "restore"], "use"]})
    base = optimize_cases._zone_easy().add_min_set_objective().add_relative_targets(0.1)
    p = base.add_subregional_targets(optimize_cases.grid(regions), t)
    m = p.compile()
    rows = [i for i, r in enumerate(m.row_ids) if r == "spp_target"][5:]
    u = p.units
    n = u.n
    code = regions.ravel()[u.index]
    zf = optimize_cases._zone_features().reshape(3, 5, -1)
    f = optimize_cases._features().reshape(5, -1)

    def tot(v, r):
        s = 0.0
        for x in v[regions.ravel() == r]:
            if np.isfinite(x):
                s += x
        return s

    # one zone: a share of the feature's total in that zone and region; every zone: of the base features'
    expect = [0.3 * tot(zf[0, 0], 1), 0.2 * tot(f[3], 2), 0.1 * tot(zf[1, 4], 2)]
    np.testing.assert_array_equal(m.rhs[rows], expect)
    a = m.A.tocsr()
    # the rows count the units of their region only, in their zones
    cols = a[rows[0]].indices
    assert (cols < n).all() and (code[cols] == 1).all()
    cols = a[rows[1]].indices
    assert (code[cols % n] == 2).all() and set(cols // n) == {0, 1, 2}
    sol = p.solve()
    cov = sol.coverage
    assert list(cov["region"][-3:]) == [1, 2, 2] and cov["met"].all()


def test_budgets_per_zone():
    p = optimize_cases._zone_easy().add_max_utility_objective({"protect": 900, "use": 400, "restore": 300})
    sol = p.solve()
    spent = sol.zone_costs
    assert spent["protect"] <= 900 and spent["use"] <= 400 and spent["restore"] <= 300
    with pytest.raises(ValueError, match="a budget for every zone"):
        optimize_cases._zone_easy().add_max_utility_objective({"protect": 900})
    with pytest.raises(ValueError, match="one per zone"):
        optimize_cases._zone_easy().add_max_utility_objective([1, 2])
    with pytest.raises(ValueError, match="problems have one zone"):
        ls.prioritize.problem(optimize_cases.pu_cost(), optimize_cases.features()).add_max_utility_objective([1, 2])


def test_extinction_risk_with_zones_counts_every_zone():
    p = optimize_cases._zone_easy().add_min_extinction_risk_objective(2500).add_highs_solver(gap=0)
    sol = p.solve()
    assert sol.cost <= 2500 * (1 + 1e-9)
    cov = sol.coverage
    assert list(cov["zone"]) == ["protect + use + restore"] * 5
    # the amounts are summed over the zones by their contributions; the totals are the base features'
    np.testing.assert_allclose(cov["total_amount"], p.units.base_totals)


def _random_zone_problem(n_side: int, seed: int, mandatory: bool = False, decisions: str = "binary"):
    rng = np.random.default_rng(seed)
    shape = (n_side, n_side)
    cost = {z: optimize_cases.grid(rng.uniform(lo, hi, shape)) for z, lo, hi in
            (("protect", 5, 50), ("use", 2, 30), ("restore", 4, 40))}
    feats = optimize_cases.grid(np.where(rng.random((6, *shape)) < 0.3, rng.random((6, *shape)), 0.0))
    p = ls.prioritize.problem(cost, feats, contribution={"use": 0.5, "restore": 0.7})
    p = p.add_min_set_objective().add_relative_targets(0.3).add_relative_targets(0.05, zone="protect")
    if mandatory:
        p = p.add_mandatory_allocation_constraints()
    return p.add_proportion_decisions() if decisions == "proportion" else p.add_binary_decisions()


@pytest.mark.parametrize(("mandatory", "decisions"), [(False, "binary"), (True, "binary"), (False, "proportion")])
def test_zones_engine_reaches_the_mip_optimum(mandatory, decisions):
    """The zones engine (Lagrangian by unit, sifting, repair) proves the optimum HiGHS's MIP finds."""
    from landschaft.prioritize._milp_solve import _highs, _is_zone_covering

    p = _random_zone_problem(12, 3, mandatory, decisions).add_highs_solver(gap=0)
    m = p.compile()
    assert _is_zone_covering(m, p.units.n)
    sol = p.solve()
    ref = _highs(m, 0.0, np.inf, 1, "choose", False, None)
    assert sol.status == "optimal"
    assert sol.objective == pytest.approx(ref["objective"], rel=1e-9)
    assert sol.coverage["met"].all()
    if mandatory:
        assert np.allclose(sol.values.sum(axis=1), 1.0)


def test_zones_engine_on_a_wide_problem_keeps_the_time_limit():
    import time

    p = _random_zone_problem(150, 5).add_highs_solver(gap=0.001, time_limit=20)
    t0 = time.perf_counter()
    sol = p.solve()
    assert time.perf_counter() - t0 < 60
    assert sol.coverage["met"].all() and sol.gap < 0.05


def _random_budget_problem(n_side: int, seed: int, objective: str, per_zone: bool, decisions: str = "binary"):
    rng = np.random.default_rng(seed)
    shape = (n_side, n_side)
    cost = {z: optimize_cases.grid(rng.uniform(lo, hi, shape)) for z, lo, hi in
            (("protect", 5, 50), ("use", 2, 30), ("restore", 4, 40))}
    feats = optimize_cases.grid(np.where(rng.random((6, *shape)) < 0.3, rng.random((6, *shape)), 0.0))
    p = ls.prioritize.problem(cost, feats, contribution={"use": 0.5, "restore": 0.7})
    total = n_side * n_side * 25.0
    budget = [0.08 * total, 0.05 * total, 0.04 * total] if per_zone else 0.15 * total
    if objective == "max_utility":
        p = p.add_max_utility_objective(budget)
    elif objective == "min_shortfall":
        p = p.add_min_shortfall_objective(budget).add_relative_targets(0.3)
    else:
        p = p.add_max_features_objective(budget).add_relative_targets(0.2)
    return p.add_proportion_decisions() if decisions == "proportion" else p.add_binary_decisions()


@pytest.mark.parametrize(("objective", "per_zone", "decisions"), [
    ("max_utility", False, "binary"), ("max_utility", True, "binary"), ("max_utility", True, "proportion"),
    ("min_shortfall", True, "binary"), ("max_features", False, "binary")])
def test_zones_wide_engine_reaches_the_mip_optimum(objective, per_zone, decisions):
    from landschaft.prioritize._milp_solve import _highs

    p = _random_budget_problem(10, 7, objective, per_zone, decisions).add_highs_solver(gap=0)
    m = p.compile()
    sol = p.solve()
    ref = _highs(m, 0.0, np.inf, 1, "choose", False, None)
    assert sol.status == "optimal"
    # the max objectives' cost term only breaks ties: within its tolerance (as test_prioritize_milp)
    assert sol.objective == pytest.approx(ref["objective"], rel=1e-6, abs=1e-9)
    budget = np.atleast_1d(p.objective[1])
    spent = np.array(list(sol.zone_costs.values())) if budget.size > 1 else np.array([sol.cost])
    assert (spent <= budget * (1 + 1e-9)).all()


def test_zones_wide_engine_on_a_wide_problem():
    import time

    p = _random_budget_problem(150, 9, "max_utility", True).add_highs_solver(gap=0.001, time_limit=30)
    t0 = time.perf_counter()
    sol = p.solve()
    assert time.perf_counter() - t0 < 90
    assert sol.status == "optimal" and sol.gap <= 0.001


@pytest.mark.parametrize("kind", ["cuts", "shuffle"])
def test_portfolios_with_zones(kind):
    p = _random_zone_problem(8, 11).add_highs_solver(gap=0)
    p = p.add_cuts_portfolio(3) if kind == "cuts" else p.add_shuffle_portfolio(3, seed=2)
    port = p.solve()
    assert len(port) >= 1
    keys = {tuple(s.values.ravel()) for s in port}
    if kind == "cuts":
        assert len(keys) == len(port) == 3  # distinct allocations, the first the optimum
        objs = [s.objective for s in port]
        assert objs == sorted(objs)
    assert port.frequency.dims[0] == "zone" and port.frequency.shape[0] == 3
    assert port.selection.attrs["zones"] == {1: "protect", 2: "use", 3: "restore"}
    for s in port:
        assert s.coverage["met"].all() and (s.values.sum(axis=1) <= 1 + 1e-9).all()


def test_zones_cut_engine_on_boundary_penalties():
    """Minimum sets with zones and boundary penalties go to the zones cut engine: a solution that meets every
    target and keeps one zone per unit, within a few percent of the bound (the linear relaxation's)."""
    from landschaft.prioritize._zones import zone_cut_model

    p = _random_zone_problem(40, 13).add_boundary_penalties(0.05).add_highs_solver(gap=0.01, time_limit=30)
    m = p.compile()
    assert zone_cut_model(m, p.units.n, 3) is not None
    sol = p.solve()
    assert sol.coverage["met"].all() and (sol.values.sum(axis=1) <= 1 + 1e-9).all()
    assert sol.bound <= sol.objective and sol.gap < 0.05
    # the reported objective is the compiled model's at the solution (the pair columns: x_a x_b)
    x = np.concatenate([sol.values.T.ravel(), np.zeros(m.ncol - p.units.cost.size)])
    a = m.A.tocsc()
    for c in np.flatnonzero(np.asarray(m.col_ids) == "b"):
        rows = a[:, c].indices
        units = [m.A[r].indices[m.A[r].indices != c][0] for r in rows[:2]]
        x[c] = x[units[0]] * x[units[1]]
    assert float(m.obj @ x) == pytest.approx(sol.objective, rel=1e-9)


# ----------------------------------------------------------------------------- evaluation with zones

def _eval_problem(decisions="binary"):
    p = optimize_cases._zone_general().add_min_set_objective().add_relative_targets(np.full((5, 3), 0.2))
    return p.add_proportion_decisions() if decisions == "proportion" else p.add_binary_decisions()


def _same_table(ours: pd.DataFrame, ref: dict, prefix: str, columns) -> None:
    for c in columns:
        r = ref[f"{prefix}/{c}"]
        o = ours[c].tolist() if isinstance(r, list) else ours[c].to_numpy(dtype=float)
        if isinstance(r, list):
            assert [str(v) for v in o] == r, (prefix, c)
        else:
            np.testing.assert_array_equal(o, r, err_msg=f"{prefix}/{c}")


def test_eval_summaries_with_zones():
    P = ls.prioritize
    ref = result("zones", "zones_eval_summaries")
    p, sol = _eval_problem(), optimize_cases.zone_sol()
    _same_table(P.eval_n_summary(p, sol), ref, "n", ["summary", "n"])
    _same_table(P.eval_cost_summary(p, sol), ref, "cost", ["summary", "cost"])
    _same_table(P.eval_boundary_summary(p, sol), ref, "boundary", ["summary", "boundary"])
    _same_table(P.eval_feature_representation_summary(p, sol), ref, "representation",
                ["summary", "feature", "total_amount", "absolute_held", "relative_held"])
    _same_table(P.eval_target_coverage_summary(p, sol, include_sense=True, include_zone=False), ref, "coverage",
                ["feature", "sense", "met", "total_amount", "absolute_target", "absolute_held", "absolute_shortfall",
                 "relative_target", "relative_held", "relative_shortfall"])


def test_eval_proportion_with_zones():
    P = ls.prioritize
    ref = result("zones", "zones_eval_proportion")
    p, sol = _eval_problem("proportion"), optimize_cases.zone_solp()
    _same_table(P.eval_n_summary(p, sol), ref, "n", ["summary", "n"])
    _same_table(P.eval_cost_summary(p, sol), ref, "cost", ["summary", "cost"])
    _same_table(P.eval_boundary_summary(p, sol, edge_factor=[0.5, 1, 0.2]), ref, "boundary", ["summary", "boundary"])
    _same_table(P.eval_feature_representation_summary(p, sol), ref, "representation",
                ["summary", "feature", "total_amount", "absolute_held", "relative_held"])


def test_eval_spatial_with_zones():
    P = ls.prioritize
    ref = result("zones", "zones_eval_spatial")
    p, sol = _eval_problem(), optimize_cases.zone_sol()
    cm, am = optimize_cases._cells(optimize_cases.cell_conn()), optimize_cases._cells(optimize_cases.cell_asym())
    bm = optimize_cases._cells(optimize_cases.zone_bnd())
    ze = np.array([1, 0.5, 0.2, 0.5, 1, 0, 0.2, 0, 0.8]).reshape(3, 3)
    _same_table(P.eval_boundary_summary(p, sol, edge_factor=[0.5, 1, 0.2], zones=optimize_cases._ZM_B, data=bm),
                ref, "boundary", ["summary", "boundary"])
    _same_table(P.eval_connectivity_summary(p, sol, cm, zones=ze), ref, "connectivity", ["summary", "connectivity"])
    a = P.eval_asym_connectivity_summary(p, sol, am, zones=optimize_cases._ZM_A)
    _same_table(a, ref, "asym", ["summary", "asym_connectivity"])


def test_eval_coverage_of_zone_lists():
    ref = result("zones", "zones_eval_coverage_lists")
    p = optimize_cases._zone_general().add_min_set_objective().add_manual_targets(optimize_cases._zone_manual_table())
    cov = ls.prioritize.eval_target_coverage_summary(p.add_binary_decisions(), optimize_cases.zone_sol(),
                                                     include_sense=True, include_zone=False)
    _same_table(cov, ref, "coverage", ["feature", "sense", "met", "total_amount", "absolute_target", "absolute_held",
                                       "absolute_shortfall", "relative_target", "relative_held", "relative_shortfall"])


def test_eval_of_a_solution_object_and_importance():
    P = ls.prioritize
    p = _random_zone_problem(8, 21).add_highs_solver(gap=0)
    sol = p.solve()
    n = P.eval_n_summary(p, sol)
    assert list(n["summary"]) == ["overall", "protect", "use", "restore"]
    assert n["n"][0] == pytest.approx(sol.values.sum())
    cost = P.eval_cost_summary(p, sol)
    assert cost["cost"][0] == pytest.approx(sol.cost) and list(cost["cost"][1:]) == pytest.approx(
        list(sol.zone_costs.values()))
    rc = P.eval_replacement_importance(p, sol, n_threads=2)
    assert rc.dims[0] == "zone"
    v = rc.values.reshape(3, -1)[:, p.units.index].T
    assert ((v > 0) == (sol.values > 0.5)).all()  # every selected unit-zone column, and no other
    ranks, steps = P.eval_rank_importance(p, sol, n=3)
    assert list(steps.columns[:3]) == ["budget_protect", "budget_use", "budget_restore"]
    rv = ranks.values.reshape(3, -1)[:, p.units.index].T
    assert ((rv > 0) <= (sol.values > 0.5)).all()
    with pytest.raises(ValueError, match="single zone"):
        P.eval_ferrier_importance(p, sol)
    risk = P.eval_extinction_risk(p, sol)
    assert len(risk) == 7  # the features and their sum


def test_landscape_metrics_and_mspa_with_zones():
    P = ls.prioritize
    sol = optimize_cases.CASES["zones_minset_sum"].py().solve()
    t = P.eval_landscape_metrics(sol, metric=["np", "ca"])
    assert set(t["zone"]) <= set(optimize_cases.ZONES) and (t["class"] >= 1).all()
    ca = t[t["metric"] == "ca"].set_index("zone")["value"]
    cell_ha = (optimize_cases.RES ** 2) / 1e4
    for k, z in enumerate(optimize_cases.ZONES):
        if z in ca.index:
            assert ca[z] == pytest.approx(sol.values[:, k].sum() * cell_ha)
    one = P.eval_landscape_metrics(sol, metric=["np"], zone="protect")
    assert len(one) == 1
    mspa, _ = P.eval_mspa(sol, zone="use")
    assert mspa.shape == sol.zones.shape
    with pytest.raises(ValueError, match="give the zone"):
        P.eval_mspa(sol)


def test_report_and_plot_with_zones():
    import os

    os.environ.setdefault("LANDSCHAFT_PLOT_SHOW", "0")
    p = optimize_cases.CASES["zones_minset_sum"].py()
    sol = p.solve()
    r = ls.prioritize.report(p, sol)
    assert list(r.tables["summary"]["summary"]) == ["overall", *optimize_cases.ZONES]
    assert "zone" in r.tables["targets"].columns
    assert sol.zones.attrs["landschaft_product"] == "ZONES"
    pl = ls.plot(sol, show=False)
    assert pl.panels[0].kind == "map"
    with pytest.raises(ValueError, match="single zone"):
        ls.prioritize.report(p, sol, zonation=True)
