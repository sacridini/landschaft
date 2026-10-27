# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.prepare / fit / predict / response_curves / importance: the pipeline and its pieces agree."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import landschaft as ls

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


@pytest.fixture(scope="module")
def inputs():
    return ls.load_table(ls.sdm.example_data("bradypus")), ls.load_raster(ls.sdm.example_data("bioclim"))


def glm():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def test_prepare_equals_the_steps_by_hand(inputs):
    occ, env = inputs
    data = ls.sdm.prepare(occ, env, clean=False, thin=50_000, area=800_000, predictors={"vif": 10},
                          background=1500, folds=4, seed=3)
    rec = ls.sdm.thin(ls.sdm.one_per_cell(occ, env), 50_000, seed=3)
    m = ls.sdm.accessible_area(rec, distance=800_000)
    env2 = ls.sdm.select_predictors(env, area=m, vif=10, seed=3)
    bg = ls.sdm.background(env2, 1500, occ=rec, area=m, seed=3)
    manual = ls.sdm.spatial_blocks(ls.sdm.dataset(rec, bg, env2), k=4, seed=3)
    pd.testing.assert_frame_equal(data.X, manual.X)
    np.testing.assert_array_equal(data.y, manual.y)
    np.testing.assert_array_equal(data.folds, manual.folds)
    assert data.report["step"].tolist() == ["input", "one per cell", "thin", "area", "predictors", "background",
                                            "predictor values", "folds"]
    assert data.settings["seed"] == 3 and data.area is not None


def test_step_forms_and_redo(inputs):
    occ, env = inputs
    base = ls.sdm.prepare(occ, env, clean=False, background=800, folds=False)
    assert base.folds is None
    kept_all = base.redo(predictors=False)
    assert len(kept_all.predictors) == 9 and len(base.predictors) < 9
    np.testing.assert_array_equal(kept_all.y, base.y)
    by_dict = ls.sdm.prepare(occ, env, clean=False, background={"n": 800}, folds=False)
    pd.testing.assert_frame_equal(by_dict.X, base.X)
    custom = ls.sdm.prepare(occ, env, clean=False, background=800, folds=False,
                            thin=lambda records: records.iloc[::2])
    assert custom.n_presences < base.n_presences
    quad = ls.sdm.prepare(occ, env, clean=False, background=800, folds={"method": "quadrants"})
    assert set(np.unique(quad.folds)) <= {0, 1, 2, 3, 4}
    with pytest.raises(ValueError, match="folds method"):
        ls.sdm.prepare(occ, env, clean=False, background=800, folds={"method": "nope"})


def test_fit_predict_ensemble(inputs):
    occ, env = inputs
    data = ls.sdm.prepare(occ, env, clean=False, background=1000, folds=3)
    from sklearn.ensemble import RandomForestClassifier

    res = ls.sdm.fit(data, {"glm": glm(), "rf": RandomForestClassifier(50, min_samples_leaf=5, random_state=1)})
    assert len(res.metrics) == 6 and set(res.metrics["model"]) == {"glm", "rf"}
    assert ((res.metrics["auc_test"] > 0.5) & (res.metrics["auc_test"] <= 1)).all()
    suit = ls.sdm.predict(res, env)
    assert [str(b) for b in suit.band.values] == ["ensemble", "sd", "glm", "rf"]
    w = res.weights("auc_test")
    expect = (w["glm"] * suit.sel(band="glm") + w["rf"] * suit.sel(band="rf")) / (w["glm"] + w["rf"])
    np.testing.assert_allclose(suit.sel(band="ensemble"), expect, rtol=1e-12)
    # in blocks of rows = all at once; NaN outside the predictors' data
    again = ls.sdm.predict(res, env, block_rows=13)
    np.testing.assert_array_equal(suit.values, again.values)
    assert np.isnan(suit.values[:, np.isnan(env.values[0])]).all()
    one = ls.sdm.predict(res, env, models="glm")
    assert [str(b) for b in one.band.values] == ["glm"]


def test_clamp_and_missing_predictors(inputs):
    occ, env = inputs
    data = ls.sdm.prepare(occ, env, clean=False, background=800, folds=False, predictors=False)
    res = ls.sdm.fit(data, {"glm": glm()})
    hot = env + 1000.0  # far beyond the data
    clamped = ls.sdm.predict(res, hot)
    assert np.isfinite(clamped.values).any()
    free = ls.sdm.predict(res, hot, clamp=False)
    assert not np.array_equal(clamped.values, free.values, equal_nan=True)
    with pytest.raises(ValueError, match="lacks the predictors"):
        ls.sdm.predict(res, env.isel(band=[0, 1]))


def test_curves_importance_and_save(inputs, tmp_path):
    occ, env = inputs
    data = ls.sdm.prepare(occ, env, clean=False, background=800, folds=False)
    res = ls.sdm.fit(data, {"glm": glm()})
    curves = ls.sdm.response_curves(res, n=15)
    assert len(curves) == 15 * len(res.predictors)
    imp = ls.sdm.importance(res, n_repeats=3)
    assert len(imp) == len(res.predictors) and imp["importance"].iloc[0] >= imp["importance"].iloc[-1]
    folder = res.save(tmp_path / "res")
    assert (folder / "ranges.csv").exists()


def test_blocks_of_the_autocorrelation_range(inputs):
    occ, env = inputs
    data = ls.sdm.prepare(occ, env, clean=False, background=800, folds={"size": "auto", "k": 4})
    r = ls.sdm.autocorrelation_range(env.sel(band=data.predictors), seed=1)["range"]
    manual = ls.sdm.spatial_blocks(ls.sdm.prepare(occ, env, clean=False, background=800, folds=False),
                                   size=r, k=4, seed=1)
    np.testing.assert_array_equal(data.folds, manual.folds)


def test_pipelines_get_the_weights(inputs):
    occ, env = inputs
    data = ls.sdm.prepare(occ, env, clean=False, background=800, folds=False)
    res = ls.sdm.fit(data, {"glm": glm()})
    direct = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
        data.X, data.y, logisticregression__sample_weight=data.sample_weight)
    np.testing.assert_allclose(res.models["glm"].predict_proba(data.X), direct.predict_proba(data.X))


def test_out_of_fold_predictions(inputs, tmp_path):
    occ, env = inputs
    data = ls.sdm.prepare(occ, env, clean=False, background=800, folds=4, seed=2)
    res = ls.sdm.fit(data, {"glm": glm()})
    oof = res.predictions
    assert list(oof.columns) == ["glm"] and len(oof) == len(data.y)
    # each row: the model fitted without its fold
    for k, (train, test) in enumerate(data.cv, start=1):
        m = glm().fit(data.X.iloc[train], data.y[train], logisticregression__sample_weight=data.sample_weight[train])
        np.testing.assert_allclose(oof["glm"].to_numpy()[test], m.predict_proba(data.X.iloc[test])[:, 1],
                                   rtol=1e-12)
    # rows in every training set (fold 0) have none
    assert np.array_equal(np.isnan(oof["glm"].to_numpy()), data.folds == 0)
    # the test AUC of each fold, from them
    for k, (_, test) in enumerate(data.cv, start=1):
        p = oof["glm"].to_numpy()[test]
        yk = data.y[test]
        if (yk == 0).any():
            auc = ls.sdm.evaluate(p[yk == 1], p[yk == 0]).stats["auc"]
            assert auc == pytest.approx(res.metrics.query("fold == @k")["auc_test"].iloc[0], rel=1e-12)
    assert (res.save(tmp_path / "res") / "predictions.csv").exists()
    assert ls.sdm.fit(data, {"glm": glm()}, cv=False).predictions is None


def test_out_of_fold_random_folds_and_stacking(inputs):
    # background in the folds too (custom folds, one per row), then a model stacked on the predictions
    occ, env = inputs
    data = ls.sdm.prepare(occ, env, clean=False, background=800, folds=False, seed=4)
    rng = np.random.default_rng(1)
    data = data.with_folds(rng.integers(1, 4, size=len(data.y)))
    res = ls.sdm.fit(data, {"glm": glm()})
    oof = res.predictions["glm"].to_numpy()
    assert np.isfinite(oof).all()
    stacked = LogisticRegression().fit(oof[:, None], data.y)
    assert stacked.coef_[0, 0] > 0
