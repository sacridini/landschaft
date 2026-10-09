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

ZONE_CASES = case_ids("zones")


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


@pytest.mark.parametrize("name", ZONE_CASES)
def test_compile_is_prioritizr(name):
    _assert_same_model(optimize_cases.CASES[name].py().compile(), model("zones", name))


@pytest.mark.parametrize("name", ZONE_CASES)
def test_solve_reaches_prioritizr_optimum(name):
    p = optimize_cases.CASES[name].py()
    sol = p.add_highs_solver(gap=0).solve()
    ref = result("zones", name)
    assert sol.status == "optimal"
    assert sol.objective == pytest.approx(ref["objective"][0], rel=1e-9)
    assert sol.cost == pytest.approx(ref["objective"][0], rel=1e-9)
    assert sol.coverage["met"].all()
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
        p.add_relative_targets(0.3).add_boundary_penalties(1.0).compile()
    with pytest.raises(ValueError, match="not supported with zones"):
        optimize_cases._zone_easy().add_max_utility_objective(1000).compile()
    with pytest.raises(ValueError, match="contribution: unknown zones"):
        ls.prioritize.problem(optimize_cases.zone_cost(), optimize_cases.features(), contribution={"x": 1})
    with pytest.raises(TypeError, match="mapping zone"):
        p.add_relative_targets(0.3).add_locked_in_constraints(optimize_cases.locked_in()).compile()
