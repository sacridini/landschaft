# SPDX-License-Identifier: GPL-3.0-or-later
"""MOLUSCE's weights of evidence and simulator (ls.simulate phase 7b), bit for bit with the plugin."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls

S = ls.simulate


def _same(ref, ours, keys):
    for k in keys:
        np.testing.assert_array_equal(np.asarray(ours[k], float), np.asarray(ref[k], float), err_msg=k)


@pytest.mark.external_reference
def test_same_as_patched_molusce():
    ref = simulate_reference.result("molusce", "molusce_fixed")
    ours = simulate_cases.run_python("molusce_fixed")
    assert set(ours) == set(ref)
    _same(ref, ours, sorted(ref))  # weights, prediction, confidence and three iterations


@pytest.mark.external_reference
def test_published_molusce_differs_only_by_its_confidence():
    ref = simulate_reference.result("molusce", "molusce_original")
    ours = simulate_cases.run_python("molusce_original")
    _same(ref, ours, [k for k in ref if k.startswith("w_")] + ["prediction"])
    differ = ref["confidence"] != ours["confidence"]
    assert differ.sum() > 100  # its "second" weight is the largest seen before the largest
    ok = ~np.isnan(ref["confidence"])
    assert (ours["confidence"][ok] <= ref["confidence"][ok]).all()  # ours: the real gap, never larger


def _world():
    w = simulate_cases.molusce_world()
    return w.sel(band="t0"), w.sel(band="t1"), {k: w.sel(band=k) for k in ("dist", "soil", "slope")}


def test_weights_table_and_potentials():
    t0, t1, drivers = _world()
    woe = S.MolusceWoE(bins=simulate_cases._MOL_BINS).fit(t0, t1, drivers)
    assert woe.classes == [1, 2, 3] and woe.drivers == ["dist", "soil", "slope"]
    tab = woe.weights_table()
    assert set(tab.columns) == {"from", "to", "driver", "category", "weight"}
    assert set(tab.loc[tab.driver == "dist", "category"]) <= {1.0, 2.0, 3.0, 4.0}
    pot = woe.potentials()
    assert (1, 2) in pot and (1, 1) in pot
    v = pot[(1, 2)]
    assert np.nanmin(v) >= 0 and np.nanmax(v) <= 100
    p = woe.predict(drivers)
    assert "probability_1_to_2" in [str(b) for b in p.band.values]
    # the weights applied to the same drivers give the fitted maps where the maps had values
    tot, fit = woe.total(drivers), woe.total()
    ok = ~np.isnan(fit[woe.encode(1, 2)])
    np.testing.assert_array_equal(tot[woe.encode(1, 2)][ok], fit[woe.encode(1, 2)][ok])


def test_ca_with_a_class_potential():
    land = np.ones((30, 30))
    land[10:20, 10:20] = 2
    p1 = np.full(land.shape, 0.5)
    p2 = np.zeros(land.shape)
    p2[:, 20:] = 0.9  # class 2 wins on the east side
    pot = S.RasterPotential({1: p1, 2: p2})
    sim = S.Model(pot, {(1, 2): 25}, S.MolusceCA(iterations=2)).run(land, seed=1)
    m = sim.maps.values
    for k in (1, 2):
        new = (m[k] == 2) & (m[k - 1] == 1)
        assert new.sum() == 25 and (np.argwhere(new)[:, 1] >= 20).all()
    assert sim.steps.loc[1, "changed_1_2"] == 25


def test_ca_counts_from_a_cross_table_are_capped():
    t0, t1, drivers = _world()
    woe = S.MolusceWoE(bins=simulate_cases._MOL_BINS).fit(t0, t1, drivers)
    cross = pd.DataFrame([[0, 10**6, 0], [0, 0, 0], [0, 0, 0]], index=[1, 2, 3], columns=[1, 2, 3])
    sim = S.Model(woe, cross, S.MolusceCA()).run(t1, seed=3)
    pred, _ = woe.prediction(t1)
    can = (t1.values == 1) & (pred == 2)
    assert sim.steps.loc[1, "changed_1_2"] == can.sum()  # all the cells predicted to take it, no more


def test_dynamic_drivers_reach_the_weights():
    t0, t1, drivers = _world()
    woe = S.MolusceWoE(bins=simulate_cases._MOL_BINS).fit(t0, t1, drivers)
    cross = {(1, 2): 30}
    sim = S.Model(woe, cross, S.MolusceCA(iterations=3), dynamic={"dist": S.distance_to(2)}).run(
        t1, drivers, seed=2)
    assert sim.maps.shape[0] == 4 and (sim.steps["changed_1_2"] == 30).all()


def test_errors():
    t0, t1, drivers = _world()
    with pytest.raises(ValueError, match="sorted"):
        S.MolusceWoE(bins={"dist": [3, 1]})
    with pytest.raises(ValueError, match="fractional"):
        S.MolusceWoE().fit(t0, t1, drivers)
    other = t1.copy(data=np.where(t1.values == 3, 4, t1.values))
    with pytest.raises(ValueError, match="same classes"):
        S.MolusceWoE(bins=simulate_cases._MOL_BINS).fit(t0, other, drivers)
    with pytest.raises(RuntimeError, match="fit"):
        S.MolusceWoE().prediction(t1)
    with pytest.raises(TypeError, match="cross table"):
        S.Model(S.MolusceWoE(), [1, 2], S.MolusceCA()).run(t1)
