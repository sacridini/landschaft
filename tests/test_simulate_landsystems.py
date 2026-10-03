# SPDX-License-Identifier: GPL-3.0-or-later
"""Land systems as overlapping binary layers, roles, caps and succession (ls.simulate phase 4)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls

S = ls.simulate


def _world(seed=21):
    """Forest, extensive and intensive pasture, soy on 40 x 50 cells, overlapping; their suitabilities."""
    rng = np.random.default_rng(seed)
    ny, nx = 40, 50
    yy, xx = np.mgrid[0:ny, 0:nx]
    g, h = xx / (nx - 1), yy / (ny - 1)

    def noise():
        return 0.15 * rng.random((ny, nx))
    forest = (g + noise() < 0.45).astype(float)
    ext = ((g + noise() > 0.35) & (g + noise() < 0.8)).astype(float)
    inten = ((g + noise() > 0.75) & (h + noise() > 0.4)).astype(float)
    soy = ((g + noise() > 0.75) & (h + noise() <= 0.4)).astype(float)
    forest[(forest + ext + inten + soy) == 0] = 1
    layers = {"forest": forest, "extensive": ext, "intensive": inten, "soy": soy}
    suit = {"forest": 1 - g + noise(), "extensive": 0.6 + noise() - 0.2 * h, "intensive": g * h + noise(),
            "soy": g * (1 - h) + noise()}
    return layers, suit


ROLES = S.Roles(natural=["forest"], families={"cattle": ["extensive", "intensive"], "crops": ["soy"]})


def _model(st, suit, rules=None, years=8, **kw):
    sup = st.supply({"extensive": {"cattle": 0.6}, "intensive": {"cattle": 2.5}, "soy": {"soy": 3.4}})
    start = sup.of(st.map)
    t = np.arange(years + 1)
    demand = S.ServiceDemand(pd.DataFrame({"cattle": start["cattle"] * 1.03 ** t, "soy": start["soy"] * 1.02 ** t}))
    rules = rules or S.Rules()
    if rules.elasticity is None:
        rules.elasticity = [0.5] * len(st.systems)
    if rules.roles is None:
        rules.roles = ROLES
    return S.Model(st.suitability(suit), demand, S.CLUMondo(sup, max_deviation=1.0, mean_deviation=0.5, **kw), rules)


# ----------------------------------------------------------------------------- parity

def test_categorical_stack_is_clumondo():
    # a categorical map as a stack without overlap (classes 0, 1, 2 -> bit masks 1, 2, 4) gives CLUMondo's maps
    spec = simulate_cases._clumondo_basic()
    model, land, drivers = simulate_cases.clumondo_model(spec)
    st = S.LandSystemStack.from_categorical(land)
    assert st.systems == [1, 2, 4]
    codes = {0: 1, 1: 2, 2: 4}
    alloc = model.allocation
    t = alloc.supply.tables[1].copy()
    t.index = [codes[k] for k in t.index]
    alloc.supply = S.Supply(t)
    o = alloc.order.copy()
    o.index = [codes[k] for k in o.index]
    alloc.order = o
    model.potential = S.LogisticRegression.from_coefficients(
        {codes[k]: v for k, v in model.potential.coefficients.iterrows()})
    sim = model.run(st, drivers, seed=1)
    ref = simulate_reference.result("clumondo", "clumondo_basic")
    back = {1: 0, 2: 1, 4: 2}
    for y in range(1, sim.maps.shape[0]):
        m = sim.maps.values[y]
        ours = np.where(np.isnan(m), -9999, np.vectorize(back.get, otypes=[float])(np.nan_to_num(m, nan=1)))
        simulate_reference.assert_same(ours, ref[f"cov_all.{y}.asc"], f"cov_all.{y}")


def test_mosaics_are_systems():
    # the stack runs exactly as the categorical map of its combinations
    layers, suit = _world()
    st = S.LandSystemStack(layers)
    a = _model(st, suit).run(st, seed=1)
    b = _model(st, suit).run(st.map, seed=1)
    np.testing.assert_array_equal(a.maps.values, b.maps.values)
    assert a.steps["converged"].all()
    lay = a.layers
    assert set(lay) == set(st.names)
    np.testing.assert_array_equal(lay["forest"].values[0], np.where(np.isnan(a.maps.values[0]), np.nan,
                                                                     layers["forest"]))
    cover = a.cover()
    np.testing.assert_array_equal(cover["soy"].to_numpy(), np.nansum(lay["soy"].values, axis=(1, 2)))
    np.testing.assert_allclose(a.shortfall.to_numpy(),
                               a.demand_table.iloc[1:].to_numpy() - a.supply.to_numpy())
    tr = a.trajectory()
    changed = [(a.maps.values[k] != a.maps.values[k - 1]).sum() for k in range(1, a.maps.shape[0])]
    np.testing.assert_array_equal(tr.sum(axis=1).to_numpy(), changed)


# ----------------------------------------------------------------------------- stack

def test_overlap_policies():
    f = np.array([[1, 1, 0, 0]])
    p = np.array([[0, 1, 1, 1]])
    u = np.array([[0, 0, 0, 1]])
    st = S.LandSystemStack({"forest": f, "pasture": p, "urban": u}, overlap={("pasture", "urban"): "urban"})
    np.testing.assert_array_equal(st.codes, [[1, 3, 2, 4]])
    assert st.overlaps().loc[4, "cells_before_policies"] == 0
    pr = S.LandSystemStack({"forest": f, "pasture": p}, default_overlap="priority", priority=["pasture", "forest"])
    np.testing.assert_array_equal(pr.codes, [[1, 2, 2, 2]])
    with pytest.raises(ValueError, match="overlap"):
        S.LandSystemStack({"forest": f, "pasture": p}, default_overlap="error")
    rare = S.LandSystemStack({"forest": f, "pasture": p}, min_cells=2)
    np.testing.assert_array_equal(rare.codes, [[1, 1, 2, 2]])  # the mosaic loses its weakest layer
    new = S.LandSystemStack({"forest": f, "pasture": p, "urban": u}, overlap={("pasture", "urban"): "urban"},
                            allow_new_mosaics=[("forest", "urban")])
    assert 5 in new.systems and new.label(5) == "forest+urban"
    nodata = S.LandSystemStack({"a": np.array([[1.0, np.nan]]), "b": np.array([[0.0, np.nan]])})
    assert np.isnan(nodata.map.values[0, 1])
    with pytest.raises(ValueError, match="unknown"):
        st.code_of("water")


def test_supply_modes():
    layers, _ = _world()
    st = S.LandSystemStack(layers)
    per = {"extensive": {"cattle": 0.6}, "intensive": {"cattle": 2.5}, "soy": {"soy": 3.4}}
    split = st.supply(per).tables[1]
    full = st.supply(per, mosaic="full").tables[1]
    mixed = st.code_of(("extensive", "intensive"))
    assert split.loc[mixed, "cattle"] == pytest.approx(1.55) and full.loc[mixed, "cattle"] == pytest.approx(3.1)
    given = st.supply(per, mosaic={("extensive", "intensive"): {"cattle": 2.0}}).tables[1]
    assert given.loc[mixed, "cattle"] == 2.0 and given.loc[mixed, "soy"] == 0.0


def test_roles():
    r = S.Roles(natural=["forest"], families={"cattle": ["extensive", "intensive"]})
    assert r.classify({"forest"}, {"forest"}) == "stable"
    assert r.classify({"extensive"}, {"forest"}) == "regeneration"
    assert r.classify({"intensive"}, {"extensive"}) == "deintensification"
    assert r.classify({"extensive"}, {"extensive", "soy"}) == "intensification"
    assert r.classify({"intensive"}, {"extensive", "soy"}) == "other"
    tab = r.table(["forest", "extensive"])
    assert tab.set_index(["from", "to"]).loc[("forest", "extensive"), "role"] == "expansion"
    r.set(("extensive", "forest"), "other")
    assert r.matrix(["forest", "extensive"])[1, 0] == "other"
    with pytest.raises(ValueError, match="unknown role"):
        r.set(("a", "b"), "growth")


# ----------------------------------------------------------------------------- rules by role

def test_cap_limits_expansion():
    layers, suit = _world()
    st = S.LandSystemStack(layers)
    free = _model(st, suit).run(st, seed=1).trajectory()
    assert free["expansion"].max() > 3
    capped = _model(st, suit, S.Rules(caps=[S.Cap("expansion", cells=3)])).run(st, seed=1)
    assert capped.steps["converged"].all()
    assert capped.trajectory().get("expansion", pd.Series([0])).max() <= 3
    with pytest.raises(ValueError, match="either"):
        S.Cap("expansion")


def test_conversion_order_and_forbidden_role():
    layers, suit = _world()
    st = S.LandSystemStack(layers)
    base = _model(st, suit).run(st, seed=1).trajectory()["expansion"].sum()
    late = _model(st, suit, S.Rules(conversion_order=["intensification", "expansion"],
                                    transition_resistance={"expansion": 0.3})).run(st, seed=1)
    assert late.trajectory().get("expansion", pd.Series([0])).sum() < base
    none = _model(st, suit, S.Rules(allowed={"expansion": 0})).run(st, seed=1)
    assert "expansion" not in none.trajectory() or none.trajectory()["expansion"].sum() == 0
    with pytest.raises(ValueError, match="need Rules"):
        m = _model(st, suit)
        S.Model(m.potential, m.demand, m.allocation, S.Rules(allowed={"expansion": 0})).run(st, seed=1)


def test_succession():
    layers, suit = _world()
    st = S.LandSystemStack(layers)
    ext, forest = st.code_of("extensive"), st.code_of("forest")
    sim = _model(st, suit, S.Rules(succession=[S.Succession(3, from_=("extensive",), to=("forest",))],
                                   history=2)).run(st, seed=1)
    # cells of extensive pasture that stayed 3 years become forest
    ages = sim.ages.values
    for k in range(1, sim.maps.shape[0]):
        prev, now = sim.maps.values[k - 1], sim.maps.values[k]
        old = (prev == ext) & (ages[k - 1] >= 3)
        assert not ((now == ext) & old).any()
    assert ((sim.maps.values[:-1] == ext) & (sim.maps.values[1:] == forest)).sum() > 0
    with pytest.raises(ValueError, match="from_ and to"):
        S.Succession(3)
    by_role = _model(st, suit, S.Rules(succession=[S.Succession(3, role="regeneration")], history=2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = by_role.run(st, seed=1)
    assert ((sim.maps.values[:-1] == ext) & (sim.maps.values[1:] == forest)).sum() > 0
    two = S.Roles(natural=["forest", "soy"])  # extensive -> forest or -> soy: both regeneration
    with pytest.raises(ValueError, match="several targets"):
        _model(st, suit, S.Rules(succession=[S.Succession(3, role="regeneration")], roles=two)).run(st, seed=1)
