# SPDX-License-Identifier: GPL-3.0-or-later
"""Pattern-oriented calibration (ls.simulate phase 9): known settings recovered from their own simulations."""

from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls

S = ls.simulate

_RNG = np.random.default_rng(0)
_D1, _D2 = _RNG.random((30, 30)), _RNG.random((30, 30))
_LAND = np.ones((30, 30))
_DEMAND = S.AreaDemand(np.array([[900, 0], [800, 100]]), classes=[1, 2])


def _build(w):
    pot = S.RasterPotential({1: np.full((30, 30), 0.5), 2: w * _D1 + (1 - w) * _D2})
    return S.Model(pot, _DEMAND, S.Ordered(order=[2, 1], stochastic=False))


def _observed(w=0.3):
    return _build(w).run(_LAND, seed=1).maps[-1]


def test_grid_recovers_the_setting():
    cal = S.calibrate(_build, _LAND, _observed(), params={"w": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                      targets={"figure_of_merit": 1}, method="grid")
    assert cal.best == {"w": 0.3} and cal.loss == 0.0
    assert list(cal.history["w"]) == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # in the order of the grid, run in parallel
    assert (cal.history["loss"].drop(index=3) > 0).all()
    assert set(cal.history.columns) == {"w", "loss", "loss_figure_of_merit"}
    v = cal.validate(factors=[1, 2], windows=[1, 3])
    assert v.summary()["figure_of_merit"] == 1.0
    assert "Calibration(w=0.3" in repr(cal)


def test_nelder_mead_finds_a_perfect_fit():
    cal = S.calibrate(_build, _LAND, _observed(0.7), params={"w": (0.0, 1.0)}, targets={"figure_of_merit": 1},
                      method="nelder-mead", max_evals=40)
    assert cal.loss == 0.0 and abs(cal.best["w"] - 0.7) < 0.1
    assert len(cal.history) <= 40


def test_methods_are_reproducible():
    obs = _observed(0.4)
    for method in ("random", "differential-evolution"):
        a = S.calibrate(_build, _LAND, obs, params={"w": (0.0, 1.0)}, targets={"figure_of_merit": 1},
                        method=method, max_evals=20, seed=3)
        b = S.calibrate(_build, _LAND, obs, params={"w": (0.0, 1.0)}, targets={"figure_of_merit": 1},
                        method=method, max_evals=20, seed=3, n_workers=1)
        np.testing.assert_array_equal(a.history["w"], b.history["w"])
        assert a.loss < 0.5 and len(a.history) <= 25


def _cars_world():
    land = np.ones((40, 40))
    land[10:20, 10:20] = 2
    pot = S.RasterPotential({1: np.full((40, 40), 0.2), 2: np.random.default_rng(1).random((40, 40))})
    start = np.array([np.sum(land == c) for c in (1, 2)], float)
    demand = S.AreaDemand(np.array([start, start + np.array([-150, 150])]), classes=[1, 2])
    return S.Model(pot, demand, S.CARS(seeds={2: 0.1}, step_size=10)), land


def test_paths_into_the_model_and_replicates():
    model, land = _cars_world()
    observed = model.run(land, seed=5).maps[-1]
    cal = S.calibrate(model, land, observed, params={"allocation.seeds[2]": [0.0, 0.1, 0.9],
                                                     "allocation.window": [3, 5]},
                      targets={"similarity_3": 1, "quantity": 1,
                               "edges": (lambda o, s: float(np.abs(np.diff(np.asarray(o), axis=0)).sum() -
                                                          np.abs(np.diff(np.asarray(s), axis=0)).sum()) ** 2 / 1e6,
                                         0.5)},
                      method="grid", replicates=2, seed=5)
    assert model.allocation.seeds == {2: 0.1} and model.allocation.window == 3  # the model given is unchanged
    assert set(cal.history.columns) >= {"allocation.seeds[2]", "allocation.window", "loss_similarity_3",
                                        "loss_quantity", "loss_edges"}
    assert len(cal.history) == 6 and cal.model.allocation.seeds[2] == cal.best["allocation.seeds[2]"]
    assert (cal.history["loss_quantity"] == 0).all()  # CARS meets the demand


def test_lsm_target_and_integers():
    model, land = _cars_world()
    observed = model.run(land, seed=5).maps[-1]
    cal = S.calibrate(model, land, observed, params={"allocation.window": (3, 7)},
                      targets={"lsm": (["np", "pland"], 1.0)}, method="grid", grid_points=3, seed=5)
    assert list(cal.history["allocation.window"]) == [3, 5, 7] and isinstance(cal.best["allocation.window"], int)
    assert "loss_lsm" in cal.history.columns and cal.history["loss_lsm"].min() >= 0


def test_errors():
    obs = _observed()
    with pytest.raises(ValueError, match="method"):
        S.calibrate(_build, _LAND, obs, params={"w": (0, 1)}, method="annealing")
    with pytest.raises(ValueError, match="params"):
        S.calibrate(_build, _LAND, obs, params={})
    with pytest.raises(ValueError, match="grid"):
        S.calibrate(_build, _LAND, obs, params={"w": [0.1, 0.2]}, method="random")
    with pytest.raises(ValueError, match="upper bound"):
        S.calibrate(_build, _LAND, obs, params={"w": (1, 0)})
    with pytest.raises(ValueError, match="unknown target"):
        S.calibrate(_build, _LAND, obs, params={"w": [0.1]}, targets={"beauty": 1}, method="grid")
    model, land = _cars_world()
    with pytest.raises(ValueError, match="no setting"):
        S.calibrate(model, land, obs, params={"allocation.colour": [1]}, method="grid")
    with pytest.raises(TypeError, match="Model or a function"):
        S.calibrate("model", land, obs, params={"w": [1]}, method="grid")


def test_calibration_plot():
    cal = S.calibrate(_build, _LAND, _observed(), params={"w": [0.2, 0.3]}, targets={"figure_of_merit": 1},
                      method="grid")
    assert [p.title for p in ls.plot(cal).panels] == ["loss", "w", None]
