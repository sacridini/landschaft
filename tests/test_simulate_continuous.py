# SPDX-License-Identifier: GPL-3.0-or-later
"""The Patcher of a continuous state (Dinamica EGO's PatcherContinuousState)."""

from __future__ import annotations

import numpy as np
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls

S = ls.simulate
CASE = ("continuous_patcher", "continuous_patcher_dinamica")


def _one_cell(**kw):
    return S.ContinuousPatcher(mean=0.09, variance=0.0, isometry=1.5, **kw)  # one 30 m cell per patch


def test_deterministic_outcomes_as_dinamica():
    # quotas and prune factors whose result does not depend on the draw: the cells kept, the quota left
    ref = simulate_reference.result(*CASE)
    world = simulate_cases.build_input("pcs_world")
    stock, prob = world["stock"], world["prob"]
    for k, (quota, prune) in enumerate(simulate_cases._PCS_DETERMINISTIC):
        a = _one_cell(prune_factor=prune).allocate(stock, prob, quota, seed=k + 1)
        assert a.remaining == ref["remaining.csv/Value"][k], (quota, prune)
        dinamica = ref[f"det_{k}.tif"]
        assert a.n_cells == int(np.sum(dinamica != np.asarray(stock)))
        if k < 3:  # the whole ranked list is taken: the same cells
            np.testing.assert_array_equal(np.asarray(a.stock), dinamica)
        assert float(np.asarray(a.taken).sum()) == pytest.approx(quota - a.remaining)


def test_seed_draw_as_dinamica():
    # 20 cells drawn among 100 of probability 0.01-1: the mean probability of the chosen cells
    ref = simulate_reference.result(*CASE)
    world = simulate_cases.build_input("pcs_world")
    stock, prob = world["draw_stock"], world["draw_prob"]
    p, s = np.asarray(prob), np.asarray(stock)
    dinamica = np.array([p[ref[f"draw_{k}.tif"] != s].mean() for k in range(simulate_cases._PCS_DRAWS)])
    cp = _one_cell(prune_factor=5)
    ours = np.array([p[np.asarray(cp.allocate(stock, prob, 180, seed=k).taken) > 0].mean() for k in range(400)])
    se = np.hypot(dinamica.std(ddof=1) / np.sqrt(dinamica.size), ours.std(ddof=1) / np.sqrt(ours.size))
    assert abs(ours.mean() - dinamica.mean()) < 4 * se
    # and not uniform (0.506) nor in proportion to P^2 (0.737)
    assert 0.62 < ours.mean() < 0.69 and 0.62 < dinamica.mean() < 0.69
    assert ref[f"draw_0.tif"].size == s.size and all(
        int(np.sum(ref[f"draw_{k}.tif"] != s)) == 20 for k in range(simulate_cases._PCS_DRAWS))


def test_patches_units_and_outputs():
    rng = np.random.default_rng(2)
    stock = simulate_cases.grid(np.where(rng.random((40, 50)) < 0.9, 150.0, 0.0))  # per hectare, 0.09 ha cells
    prob = simulate_cases.grid(rng.random((40, 50)))
    cp = S.ContinuousPatcher(mean=0.9, variance=0.2, isometry=1.8)  # about 10 cells per patch
    a = cp.allocate(stock, prob, 400.0, seed=3)
    taken, patch = np.asarray(a.taken), np.asarray(a.patches)
    hit = taken > 0
    assert a.n_cells == hit.sum() and a.n_patches == patch.max() and np.array_equal(hit, patch > 0)
    np.testing.assert_allclose(taken[hit], 150.0 * 0.09)
    assert -150 * 0.09 < a.remaining <= 0 and taken.sum() == pytest.approx(400 - a.remaining)
    assert (np.asarray(a.stock)[hit] == 0).all() and (np.asarray(a.probability)[hit] == 0).all()
    np.testing.assert_array_equal(np.asarray(a.stock)[~hit], np.asarray(stock)[~hit])
    assert a.stock.rio.crs == stock.rio.crs
    # patches are connected (8 neighbours) and larger than one cell on average
    from scipy import ndimage

    for k in range(1, a.n_patches + 1):
        assert ndimage.label(patch == k, structure=np.ones((3, 3)))[1] == 1
    assert a.n_cells / a.n_patches > 3
    # the same seed, the same allocation; cells without stock are never taken
    b = cp.allocate(stock, prob, 400.0, seed=3)
    np.testing.assert_array_equal(np.asarray(b.patches), patch)
    assert not hit[np.asarray(stock) == 0].any()


def test_isometry_compacts_patches():
    rng = np.random.default_rng(4)
    stock, prob = np.ones((60, 60)), rng.random((60, 60))

    def edges(iso):
        cp = S.ContinuousPatcher(mean=25, variance=0, isometry=iso, unit="cells")
        e = 0
        for seed in range(5):
            patch = np.asarray(cp.allocate(stock, prob, 200, seed=seed).patches) > 0
            e += np.sum(patch[:, 1:] != patch[:, :-1]) + np.sum(patch[1:] != patch[:-1])
        return e

    assert edges(2.0) < edges(1.0) < edges(0.2)


def test_run_steps_and_regrowth():
    rng = np.random.default_rng(5)
    stock = simulate_cases.grid(np.full((30, 30), 100.0))
    prob = simulate_cases.grid(rng.random((30, 30)))
    cp = S.ContinuousPatcher(mean=0.27, variance=0.0, isometry=1.0)
    sim = cp.run(stock, prob, [90, 90, 90], regrowth=lambda s, step: s + 1.0, seed=7)
    assert sim.stock.shape == (4, 30, 30) and sim.taken.shape == (3, 30, 30)
    assert list(sim.steps.index) == [1, 2, 3]
    np.testing.assert_allclose(sim.steps["taken"], np.asarray(sim.taken).sum(axis=(1, 2)))
    # a cell taken stays out of later steps (its probability is depleted) and regrows from 0
    t1 = np.asarray(sim.taken[0]) > 0
    assert not (np.asarray(sim.taken[1:]) > 0)[:, t1].any()
    np.testing.assert_allclose(np.asarray(sim.stock[1])[t1], 1.0)
    np.testing.assert_allclose(np.asarray(sim.stock[3])[~(np.asarray(sim.taken) > 0).any(0)], 103.0)


def test_errors_and_edges():
    with pytest.raises(ValueError, match="unit"):
        S.ContinuousPatcher(unit="m2")
    with pytest.raises(ValueError, match="isometry"):
        S.ContinuousPatcher(isometry=3)
    with pytest.raises(ValueError, match="window"):
        S.ContinuousPatcher(window=(2, 3))
    with pytest.raises(ValueError, match="georeferenced"):
        S.ContinuousPatcher().allocate(np.ones((3, 3)), np.ones((3, 3)), 1)
    with pytest.raises(ValueError, match="shape"):
        S.ContinuousPatcher(unit="cells").allocate(np.ones((3, 3)), np.ones((2, 3)), 1)
    a = S.ContinuousPatcher(unit="cells").allocate(np.ones((3, 3)), np.ones((3, 3)), 0, seed=1)
    assert a.n_cells == 0 and a.remaining == 0
    b = S.ContinuousPatcher(unit="cells").allocate(np.full((3, 3), np.nan), np.ones((3, 3)), 5, seed=1)
    assert b.n_cells == 0 and b.remaining == 5
