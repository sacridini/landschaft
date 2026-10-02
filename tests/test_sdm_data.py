# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.dataset and SDMData."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import landschaft as ls


@pytest.fixture(scope="module")
def inputs():
    env = ls.load_raster(ls.sdm.example_data("bioclim"))
    occ = ls.load_table(ls.sdm.example_data("bradypus"))
    bg = ls.sdm.background(env, 500, occ=occ, seed=1)
    return env, occ, bg


def test_values_rows_and_weights(inputs):
    env, occ, bg = inputs
    d = ls.sdm.dataset(occ, bg, env)
    assert d.n_presences == 116 and d.n_background == 500
    assert d.predictors == [str(b) for b in env.band.values]
    # the values are the cells' values
    cells = bg["cell"].to_numpy()
    flat = env.values.reshape(env.shape[0], -1)
    np.testing.assert_array_equal(d.X.iloc[116:].to_numpy(), flat[:, cells].T)
    w = d.sample_weight
    assert w[:116].sum() == pytest.approx(w[116:].sum())
    assert ls.sdm.dataset(occ, bg, env, weights="equal").sample_weight is None


def test_missing_values_are_dropped_and_reported(inputs):
    env, occ, bg = inputs
    sea = pd.DataFrame({"lon": [-30.0, -100.0], "lat": [-20.0, -40.0]})  # in the sea: no climate
    d = ls.sdm.dataset(pd.concat([occ[["lon", "lat"]], sea]), bg, env)
    assert d.n_presences == 116
    assert d.report.loc[0, "removed"] == 2


def test_presence_absence_table(inputs):
    env, occ, _ = inputs
    pa = occ[["lon", "lat"]].copy()
    pa["presence"] = [1, 0] * 58
    d = ls.sdm.dataset(pa, None, env)
    assert d.n_presences == 58 and d.n_background == 58


def test_cv_and_folds(inputs):
    env, occ, bg = inputs
    d = ls.sdm.dataset(occ, bg, env)
    with pytest.raises(ValueError, match="no folds"):
        _ = d.cv
    folds = np.where(d.y == 1, np.arange(len(d.y)) % 3 + 1, 0)
    d2 = d.with_folds(folds)
    cv = d2.cv
    assert len(cv) == 3
    for k, (train, test) in enumerate(cv, start=1):
        assert set(test) == set(np.flatnonzero(folds == k))
        assert set(np.flatnonzero(d.y == 0)) <= set(train)


def test_save(inputs, tmp_path):
    env, occ, bg = inputs
    folder = ls.sdm.dataset(occ, bg, env).save(tmp_path / "data")
    names = sorted(p.name for p in folder.iterdir())
    assert "data.csv" in names and "report.csv" in names
