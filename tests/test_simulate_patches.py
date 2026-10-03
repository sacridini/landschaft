# SPDX-License-Identifier: GPL-3.0-or-later
"""Expander and Patcher (ls.simulate phase 5c): Dinamica's transition functions, written from the publications.

Dinamica EGO's code is closed and its functors crash for patches of more than one cell (8.13), so the
comparison is statistical, on one-cell patches: the Expander's acceptance by the number of target-class
neighbours and the Patcher's seed draw.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import simulate_cases
import simulate_reference
from scipy import ndimage

import landschaft as ls

S = ls.simulate


def _nj(t0):
    j = (t0 == 2).astype(int)
    return ndimage.convolve(j, np.ones((3, 3), int), mode="constant") - j


def _stack(d, prefix="out"):
    return np.array([d[k] for k in sorted(d, key=lambda k: int(k[len(prefix):-4]))])


def test_expander_like_dinamica():
    t0 = simulate_cases.patches_scattered().values
    ref = _stack(simulate_reference.result("patches", "patches_expander"))
    ours = _stack(simulate_cases.run_python("patches_expander"))
    nj, c1 = _nj(t0), t0 == 1
    for k in range(1, 9):
        cells = c1 & (nj == k)
        a = float(((ours == 2) & cells).sum() / (cells.sum() * len(ours)))
        b = float(((ref == 2) & cells).sum() / (cells.sum() * len(ref)))
        assert abs(a - b) < 0.05 + 0.5 / np.sqrt(cells.sum()), (k, a, b)  # acceptance per n_j (3000 of ~5000)
    assert ((ours == 2) & c1 & (nj == 0)).sum() == 0 and ((ref == 2) & c1 & (nj == 0)).sum() == 0


def test_patcher_like_dinamica():
    blobs = simulate_cases.patches_blobs()
    t0, prob = blobs["t0"].values, blobs["prob"].values
    ref = _stack(simulate_reference.result("patches", "patches_patcher"))
    ours = _stack(simulate_cases.run_python("patches_patcher"))
    nj, c1 = _nj(t0), t0 == 1

    def stats(maps):
        new = (maps == 2) & c1
        return new.sum(axis=(1, 2)), np.array([(nj[m] > 0).sum() for m in new]), \
            np.array([prob[m].mean() for m in new])
    n_r, adj_r, p_r = stats(ref)
    n_o, adj_o, p_o = stats(ours)
    assert (n_r == 507).all() and (n_o == 507).all()
    assert abs(adj_o.mean() - adj_r.mean()) < 3 * adj_r.std() / np.sqrt(len(adj_r)) + 2
    assert abs(p_o.mean() - p_r.mean()) < 0.01  # seeds drawn in proportion to P among the pruned cells


# ----------------------------------------------------------------------------- properties

def _landscape(n=80):
    yy, xx = np.mgrid[0:n, 0:n]
    land = np.ones((n, n))
    land[(yy - 40) ** 2 + (xx - 40) ** 2 < 49] = 2
    land[:, :5] = 3
    rng = np.random.default_rng(2)
    prob = np.clip(0.3 + 0.4 * rng.random((n, n)), 0, 1)
    return simulate_cases.grid(land), simulate_cases.grid(prob)


def _model(**kw):
    land, prob = _landscape()
    alloc = S.ExpanderPatcher(steps=3, **kw)
    return S.Model({(1, 2): prob}, {(1, 2): 0.05}, alloc), land


def test_changes_are_made():
    model, land = _model(expansion=0.6, expander=(20, 100, 1.5), patcher=(20, 100, 1.5))
    sim = model.run(land, seed=3)
    m = sim.maps.values
    assert m.shape == (4, 80, 80)
    for k in range(1, 4):
        req = sim.steps.loc[k, "requested_1_2"]
        assert req == int(np.floor(0.05 * np.sum(m[k - 1] == 1) + 0.5))
        assert np.sum((m[k] == 2) & (m[k - 1] == 1)) == req
        assert sim.steps.loc[k, "expanded_1_2"] + sim.steps.loc[k, "patched_1_2"] == req
    assert (m[:, :, :5] == 3).all()  # classes without transitions stay
    np.testing.assert_array_equal(model.run(land, seed=3).maps.values, m)


def test_isometry_compacts_patches():
    edges = []
    for iso in (0.0, 1.0, 2.0):
        model, land = _model(expansion=0.0, patcher=(30, 0, iso))
        new = (model.run(land, seed=1).maps.values[1] == 2) & (land.values == 1)
        e = np.sum(new[:, 1:] != new[:, :-1]) + np.sum(new[1:] != new[:-1])
        edges.append(e / new.sum())
    assert edges[0] > edges[1] > edges[2]
    model, land = _model(expansion=0.0, patcher=(30, 0, 0.0))
    sim = model.run(land, seed=1)
    assert sim.steps.loc[1, "patches_1_2"] == sim.steps.loc[1, "requested_1_2"]  # isometry 0: one cell per patch


def test_expander_stays_next_to_patches():
    model, land = _model(expansion=1.0, expander=(1, 0, 1.0))
    sim = model.run(land, seed=5)
    first = (sim.maps.values[1] == 2) & (land.values == 1)
    expanded = sim.steps.loc[1, "expanded_1_2"]
    assert expanded > 0
    near = _nj(land.values)[first] > 0
    assert near.sum() >= expanded  # the Expander's cells touch the existing patch


def test_patch_size_units():
    land, prob = _landscape()
    alloc = S.ExpanderPatcher(steps=1, expansion=0.0, patcher=(0.9, 0.0, 2.0), size_unit="hectares")
    assert alloc._cells(0.9, ls.simulate._maps.read_map(land)) == pytest.approx(10.0)  # 30 m cells: 0.09 ha
    sim = S.Model({(1, 2): prob}, {(1, 2): 0.05}, alloc).run(land, seed=2)
    assert sim.steps.loc[1, "patches_1_2"] < sim.steps.loc[1, "requested_1_2"] / 3


def test_with_markov_and_weights_of_evidence():
    d = simulate_cases.woe_synth()
    t0, t1 = d.sel(band="t0"), d.sel(band="t1")
    drivers = {"dist": d.sel(band="dist")}
    woe = S.WeightsOfEvidence({"dist": 25}, transitions=[(1, 2), (1, 3)]).fit(t0, t1, drivers)
    markov = S.Markov.from_maps(t0, t1, steps=2)
    rates = markov.multi.copy()
    rates.loc[3, 2] = 0.0  # transitions the potential does not give
    rates.loc[2, 3] = rates.loc[2, 1] = rates.loc[3, 1] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = S.Model(woe, rates, S.ExpanderPatcher(steps=2, expander=(4, 4, 1.2), patcher=(4, 4, 1.2))).run(
            t0, drivers, seed=1)
    assert sim.maps.shape[0] == 3 and {"requested_1_2", "patched_1_3"} <= set(sim.steps.columns)


def test_errors():
    land, prob = _landscape()
    with pytest.raises(ValueError, match="size_unit"):
        S.ExpanderPatcher(size_unit="acres")
    with pytest.raises(ValueError, match="window"):
        S.ExpanderPatcher(window=(2, 3))
    with pytest.raises(ValueError, match="steps"):
        S.Model({(1, 2): prob}, {(1, 2): 0.1}, S.ExpanderPatcher()).run(land, seed=1)
    with pytest.raises(ValueError, match="no probability"):
        S.Model({(1, 3): prob}, {(1, 2): 0.1}, S.ExpanderPatcher(steps=1)).run(land, seed=1)
    with pytest.raises(ValueError, match="no expansion share"):
        S.Model({(1, 2): prob}, {(1, 2): 0.1}, S.ExpanderPatcher(steps=1, expansion={(2, 1): 0.5})).run(land, seed=1)
    with pytest.raises(TypeError, match="Markov"):
        S.Model({(1, 2): prob}, [0.1], S.ExpanderPatcher(steps=1)).run(land, seed=1)
    df = pd.DataFrame([[0, 0.1], [0, 0]], index=[1, 2], columns=[1, 2])
    sim = S.Model({(1, 2): prob}, df, S.ExpanderPatcher(steps=1)).run(land, seed=1)
    assert sim.steps.loc[1, "requested_1_2"] > 0
