# SPDX-License-Identifier: GPL-3.0-or-later
"""Allocation (ls.simulate phase 3): demand, rules, CLUE-S and the ordered procedure against lulcc."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls
from landschaft.simulate._demand import _approx_extrap, _round_sum
from landschaft.simulate._rules import allow_matrix, auto_convert


@pytest.mark.parametrize("case", simulate_reference.case_ids("allocation"))
def test_same_as_lulcc(case):
    ref = simulate_reference.result("allocation", case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = simulate_cases.run_python(case)
    assert set(ours) == set(ref), sorted(set(ours) ^ set(ref))
    for key, value in ours.items():
        simulate_reference.assert_same(value, ref[key], f"{case}/{key}")


# ----------------------------------------------------------------------------- demand

def test_approx_extrap_inside_and_outside():
    x = np.array([0.0, 6.0, 14.0])
    y = np.array([100.0, 160.0, 120.0])
    out = _approx_extrap(x, y, np.array([-2.0, 0.0, 3.0, 6.0, 10.0, 14.0, 20.0]))
    np.testing.assert_allclose(out, [80, 100, 130, 160, 140, 120, 90])


def test_round_sum_keeps_the_total():
    x = np.array([[1.4, 1.4, 1.2], [10.0, 0.5, 0.5]])
    out = _round_sum(x, 7)
    np.testing.assert_array_equal(out.sum(axis=1), [7, 7])
    np.testing.assert_array_equal(out[0], [2, 3, 2])  # ties: the later of equal fractions rounds up (R's sort)


def test_area_demand_table_and_array():
    d = ls.simulate.AreaDemand(np.array([[5, 5], [4, 6], [3, 7]]), classes=[1, 2])
    assert d.steps == 2 and d.classes == [1, 2]
    np.testing.assert_array_equal(d.at(2, [2, 1]), [7, 3])
    np.testing.assert_array_equal(d.at(1, [1, 2, 9]), [4, 6, 0])
    assert repr(d) == "AreaDemand(2 steps, 2 classes)"
    with pytest.raises(ValueError, match="class of each column"):
        ls.simulate.AreaDemand(np.ones((3, 2)))
    with pytest.raises(ValueError, match="at least"):
        ls.simulate.AreaDemand(pd.DataFrame({1: [5]}))


def test_area_demand_from_maps_extrapolates():
    lu = simulate_cases.pie()
    d = ls.simulate.AreaDemand.from_maps(lu, times=[0, 6, 14], out_times=range(21))
    assert d.steps == 20
    tab = d.table.to_numpy()
    n = int(np.sum(~np.isnan(lu[0].values)))
    np.testing.assert_array_equal(tab.sum(axis=1), n)
    assert (tab >= 0).all()


# ----------------------------------------------------------------------------- rules

def test_allow_matrix_codes():
    classes = [1, 2, 3]
    rules = np.array([[1, 1008, 105], [0, 1, -2], [-1, 1, 1]])
    x = np.array([1.0, 1.0, 2.0, 3.0])
    hist = np.array([3, 9, 0, 0])
    cd = np.array([-5.0, 0.0, 5.0])
    a = allow_matrix(x, hist, classes, cd, rules)
    nan = np.nan
    expected = [[1, nan, 1],      # forest for 3 steps: not yet to built (1008), still to other (105)
                [1, 1, nan],      # forest for 9 steps: to built, no longer to other
                [nan, 1, 1],      # built -> other only while other's demand grows (it does)
                [nan, 1, 1]]      # other -> forest only while other's demand shrinks (it grows)
    np.testing.assert_array_equal(a, expected)


def test_auto_convert():
    classes = [1, 2, 3]
    nan = np.nan
    prob = np.array([[nan, 0.2, 0.7], [0.5, 0.1, 0.1], [nan, nan, nan], [0.3, 0.3, 0.1]])
    x = np.array([1.0, 1.0, 2.0, 1.0])
    out = auto_convert(x, prob, classes, mask=np.array([1, 0, 1, 1]))
    # own class not allowed -> the most suitable one; restricted -> stays; nothing allowed -> stays; free: none
    np.testing.assert_array_equal(out, [3, 1, 2, nan])


def test_neighbourhood_threshold():
    land = np.array([[1, 1, 2], [1, 2, 2], [2, 2, 2]], dtype=float)
    nb = ls.simulate.Neighbourhood({1: 0.5})
    a = nb.allow(land, [1, 2])
    frac = np.array([3 / 4, 3 / 6, 1 / 4, 3 / 6, 3 / 9, 0, 1 / 4, 0, 0])
    np.testing.assert_array_equal(a[:, 0], np.where(frac >= 0.5, 1, np.nan))
    np.testing.assert_array_equal(a[:, 1], 1)


def test_rules_matrix_from_frame_and_checks():
    df = pd.DataFrame([[1, 0], [1, 1]], index=["2", "1"], columns=["2", "1"])
    r = ls.simulate.Rules(allowed=df, elasticity={2: 0.5})
    np.testing.assert_array_equal(r.matrix([1, 2]), [[1, 1], [0, 1]])
    np.testing.assert_array_equal(r.elasticities([1, 2]), [0, 0.5])
    with pytest.raises(ValueError, match="square"):
        ls.simulate.Rules(allowed=np.ones((2, 3))).matrix([1, 2])
    with pytest.raises(ValueError, match="one elasticity"):
        ls.simulate.Rules(elasticity=[0.1]).elasticities([1, 2])


# ----------------------------------------------------------------------------- model

def _toy():
    rng = np.random.default_rng(3)
    land = np.where(rng.random((20, 30)) < 0.6, 1.0, 2.0)
    land[0, :3] = np.nan
    pot = ls.simulate.RasterPotential({1: rng.random((20, 30)), 2: rng.random((20, 30))})
    n1 = int(np.nansum(land == 1))
    n = int(np.sum(~np.isnan(land)))
    demand = ls.simulate.AreaDemand(np.array([[n1, n - n1], [n1 - 40, n - n1 + 40], [n1 - 80, n - n1 + 80]]),
                                    classes=[1, 2])
    return land, pot, demand


def test_model_meets_demand_and_reproduces():
    land, pot, demand = _toy()
    model = ls.simulate.Model(pot, demand, ls.simulate.CLUE(scale=0.0005, max_diff=3, mean_diff=2),
                              ls.simulate.Rules(elasticity=[0.3, 0.1]))
    sim = model.run(land, seed=4)
    assert sim.maps.shape == (3, 20, 30)
    assert np.isnan(sim.maps.values[:, 0, :3]).all()
    np.testing.assert_array_equal(sim.maps.values[0], land)
    for step in (1, 2):
        got = [np.sum(sim.maps.values[step] == c) for c in (1, 2)]
        assert np.abs(got - demand.at(step, [1, 2])).max() <= 3
    assert sim.steps["converged"].all() and repr(sim) == "Simulation(2 steps, seed=4)"
    again = model.run(land, seed=4)
    np.testing.assert_array_equal(again.maps.values, sim.maps.values)
    drawn = model.run(land)
    assert isinstance(drawn.seed, int)
    assert "CLUE" in repr(model)


def test_restricted_cells_and_history():
    land, pot, demand = _toy()
    restricted = np.zeros(land.shape, dtype=bool)
    restricted[5:15, 5:20] = True
    model = ls.simulate.Model(pot, demand, ls.simulate.Ordered([1, 2], stochastic=False),
                              ls.simulate.Rules(restrictions=restricted))
    sim = model.run(land, seed=1)
    final = sim.maps.values[-1]
    np.testing.assert_array_equal(final[restricted], land[restricted])
    changed = sim.maps.values[1] != sim.maps.values[2]
    assert (sim.history[changed & ~np.isnan(land)] == 0).all()
    assert (sim.history[~np.isnan(land)] <= 3).all()


def test_ordered_unmet_demand_warns():
    land, pot, demand = _toy()
    allowed = np.array([[1, 0], [1, 1]])  # nothing may become class 2
    model = ls.simulate.Model(pot, demand, ls.simulate.Ordered([1, 2]), ls.simulate.Rules(allowed=allowed))
    with pytest.warns(UserWarning, match="not met"):
        sim = model.run(land, seed=2)
    assert not sim.steps["converged"].any()
    assert set(np.unique(sim.maps.values[-1][~np.isnan(land)])) <= {1.0, 2.0}  # no -1 left


def test_clue_max_iter_warns():
    land, pot, demand = _toy()
    model = ls.simulate.Model(pot, demand, ls.simulate.CLUE(scale=1e-9, max_iter=5, max_diff=0, mean_diff=0))
    with pytest.warns(UserWarning, match="did not meet the demand"):
        sim = model.run(land, seed=2)
    assert (sim.steps["iterations"] == 6).all()


def test_model_errors():
    land, pot, demand = _toy()
    small = ls.simulate.AreaDemand(np.array([[10], [10]]), classes=[1])
    with pytest.raises(ValueError, match="without demand"):
        ls.simulate.Model(pot, small, ls.simulate.CLUE()).run(land, seed=1)
    one = ls.simulate.RasterPotential({1: np.ones(land.shape)})
    with pytest.raises(ValueError, match="no suitability"):
        ls.simulate.Model(one, demand, ls.simulate.CLUE()).run(land, seed=1)
    with pytest.raises(ValueError, match="each once"):
        ls.simulate.Model(pot, demand, ls.simulate.Ordered([1])).run(land, seed=1)
