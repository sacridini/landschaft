# SPDX-License-Identifier: GPL-3.0-or-later
"""CLUMondo (ls.simulate phase 4): land systems, goods and services, against CLUMondo itself."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls

S = ls.simulate
CASES = simulate_reference.case_ids("clumondo")


@pytest.mark.parametrize("case", [c for c in CASES if not c.startswith("clumondo2017")])
def test_same_as_clumondo(case):
    ref = simulate_reference.result("clumondo", case)
    ours = simulate_cases.run_python(case)
    assert set(ours) == set(ref), sorted(set(ours) ^ set(ref))
    for key, value in ours.items():
        simulate_reference.assert_same(value, ref[key], f"{case}/{key}")


@pytest.mark.parametrize("case", [c for c in CASES if c.startswith("clumondo2017")])
def test_close_to_clumondo_2017(case):
    # the executable of May 2017 has no published source and its own floating point (2 cells of 2000 differ
    # in year 1 of the plain case): the extensions written from its manual agree on >= 97% of the cells
    ref = simulate_reference.result("clumondo", case)
    ours = simulate_cases.run_python(case)
    assert set(ref) <= set(ours)
    for key in ref:
        assert np.mean(ours[key] == ref[key]) >= 0.97, key


# ----------------------------------------------------------------------------- pieces

def test_service_demand_and_supply():
    d = S.ServiceDemand({"crops": [10, 11, 12], "meat": [5, 5, 6]})
    np.testing.assert_array_equal(d.at(2, ["meat", "crops"]), [6, 12])
    assert repr(d) == "ServiceDemand(2 steps, 2 services)"
    with pytest.raises(ValueError, match="positive"):
        S.ServiceDemand({"crops": [10, 0]})
    with pytest.raises(ValueError, match="no demand"):
        d.at(1, ["wood"])
    sup = S.Supply(pd.DataFrame({"crops": [0.0, 1.0, 3.0], "meat": [0.5, 0.5, 0.0]}, index=[10, 20, 30]))
    land = np.array([[10, 20], [30, 30]], dtype=float)
    np.testing.assert_allclose(sup.of(land), [7.0, 1.0])
    dyn = S.Supply({1: sup.at(1), 4: sup.at(1) * 2})
    assert dyn.at(3) is dyn.tables[1] and dyn.at(9) is dyn.tables[4]
    with pytest.raises(ValueError, match="step 1"):
        S.Supply({2: sup.at(1)})


def test_default_order_ranks_the_supply():
    sup = S.Supply(pd.DataFrame({"a": [0.0, 2.0, 1.0, 2.0], "b": [1.0, 0.0, 0.0, 3.0]}))
    order = S.CLUMondo(sup).lusconv()
    np.testing.assert_array_equal(order.to_numpy(), [[0, 1], [2, 0], [1, 0], [2, 2]])


def test_settings_are_checked():
    sup = S.Supply({0: {"a": 1.0}})
    with pytest.raises(ValueError, match="speed"):
        S.CLUMondo(sup, speed=2)
    with pytest.raises(ValueError, match="percentage"):
        S.CLUMondo(sup, deviation="absolute", max_deviation={"a": 1})
    with pytest.raises(ValueError, match="kinds"):
        S.CLUMondo(sup, kinds={"a": "at least"})


# ----------------------------------------------------------------------------- model

SERVICES = ["s0", "s1", "s2"]
SUPPLY = [f"supply_{v}" for v in SERVICES]


def _relabel(a, values):
    return np.where(np.isnan(a), np.nan, np.asarray(values, dtype=float)[np.nan_to_num(a).astype(int)])


def _toy(values=None, **kw):
    """The 'services' reference case (5 systems, 3 services, restricted block, no data around) with its
    suitability given as maps (the double-precision regression)."""
    model, land, drivers = simulate_cases.clumondo_model(simulate_cases._clumondo_services(), kw)
    p = model.potential.predict(drivers)
    keys = list(range(5)) if values is None else list(values)
    model.potential = S.RasterPotential({k: p.values[c] for c, k in enumerate(keys)})
    if values is not None:
        land = _relabel(land, values)
        alloc = model.allocation
        t = alloc.supply.tables[1].copy()
        t.index = keys
        alloc.supply = S.Supply(t)
        o = alloc.order.copy()
        o.index = keys
        alloc.order = o
    return model, land


def test_meets_the_demand_and_keeps_its_state():
    model, land = _toy()
    sim = model.run(land, seed=3)
    assert sim.maps.shape == (9, 40, 50) and sim.steps["converged"].all()
    for step in sim.steps.index:
        dev = 100 * np.abs(sim.steps.loc[step, SUPPLY].to_numpy(dtype=float) / model.demand.at(step, SERVICES) - 1)
        assert dev.max() <= 0.8 and dev.mean() <= 0.3 + 1e-6
        np.testing.assert_allclose(model.allocation.supply.of(sim.maps[step]).to_numpy(),
                                   sim.steps.loc[step, SUPPLY].to_numpy(dtype=float))
    assert np.isnan(sim.maps.values[:, :2, :]).all()
    changed = sim.maps.values[2] != sim.maps.values[1]
    assert (sim.ages.values[2][changed & ~np.isnan(land)] == 1).all()
    assert sim.changed.dtype == bool and sim.changed.sum() > 0
    out = sim.outputs(pd.DataFrame({"carbon": [100.0, 40.0, 10.0, 60.0, 30.0]}))
    np.testing.assert_array_equal(out.sel(output="carbon").values[1][sim.maps.values[1] == 0], 100.0)
    init = np.full(land.shape, 7.0)
    kept = sim.outputs(pd.DataFrame({"carbon": [100.0, 40.0, 10.0, 60.0, 30.0]}), initial={"carbon": init})
    same = (sim.maps.values == sim.maps.values[0]).all(axis=0) & ~np.isnan(land)
    np.testing.assert_array_equal(kept.sel(output="carbon").values[-1][same], 7.0)


def test_system_values_are_labels():
    a, land = _toy()
    b, land_b = _toy(values=(10, 20, 30, 40, 50))
    sa, sb = a.run(land, seed=2), b.run(land_b, seed=2)
    np.testing.assert_array_equal(_relabel(sa.maps.values, (10, 20, 30, 40, 50)), sb.maps.values)


def test_minimum_and_maximum_demands():
    model, land = _toy(kinds={"s1": "minimum", "s2": "maximum"})
    sim = model.run(land, seed=1)
    for step in sim.steps.index:
        d = model.demand.at(step, SERVICES)
        got = sim.steps.loc[step, SUPPLY].to_numpy(dtype=float)
        assert abs(got[0] / d[0] - 1) <= 0.008
        assert got[1] >= d[1] * (1 - 0.008) and got[2] <= d[2] * (1 + 0.008)


def test_deviation_per_service():
    model, land = _toy(max_deviation={"s0": 0.5, "s1": 2.0, "s2": 5.0})
    sim = model.run(land, seed=1)
    for step in sim.steps.index:
        dev = 100 * np.abs(sim.steps.loc[step, SUPPLY].to_numpy(dtype=float) / model.demand.at(step, SERVICES) - 1)
        assert (dev <= [0.5, 2.0, 5.0]).all()


def _window_counts(mask):
    pad = np.pad(mask, 1)
    return sum(pad[1 + a: pad.shape[0] - 1 + a, 1 + b: pad.shape[1] - 1 + b] for a in (-1, 0, 1) for b in (-1, 0, 1))


def test_patches_respect_rules_and_size():
    model, land = _toy(patches={2: (1, 0.0, 4)})
    restricted = np.asarray(model.rules.restrictions)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = model.run(land, seed=4)
    assert sim.maps.shape[0] > 1
    for step in range(1, sim.maps.shape[0]):
        np.testing.assert_array_equal(sim.maps.values[step][restricted], land[restricted])
    new2 = (sim.maps.values[1] == 2) & (land != 2)
    assert new2.sum() > 0
    assert (_window_counts(sim.maps.values[1] == 2)[new2] >= 2).all()  # no lone new cells: patches


def test_no_solution_stops_with_a_warning():
    model, land = _toy(max_deviation=0.0, mean_deviation=0.0, max_loops=50)
    with pytest.warns(UserWarning, match="no solution in step 1") as rec:
        sim = model.run(land, seed=1)
    assert rec[0].filename == __file__
    assert sim.maps.shape[0] == 1 and not sim.steps["converged"].any()


def test_errors():
    model, land = _toy()
    with pytest.raises(ValueError, match="without supply"):
        model.run(np.where(land == 1, 7, land), seed=1)
    model.rules.allowed = np.where(np.eye(5) > 0, 1, 5)
    with pytest.raises(ValueError, match="2-99 refer to maps"):
        model.run(land, seed=1)
    model.rules.allowed = None
    model.demand = S.AreaDemand(np.ones((2, 5)), classes=[0, 1, 2, 3, 4])
    with pytest.raises(TypeError, match="ServiceDemand"):
        model.run(land, seed=1)


def test_random_ages_as_clumondo():
    # CLUMondo's agemode 1: rand() % max + 1 per cell from the runtime's generator with seed 1
    ages = ls._core.sim_clumondo_random_age(5, 8, 1)
    crt = ls._core.sim_crt_rand(1, 5)
    np.testing.assert_array_equal(ages, crt % 8 + 1)
    model, land = _toy()
    model.rules.history = 6
    sim = model.run(land, seed=9)
    assert sim.ages.values[0][~np.isnan(land)].max() <= 6
