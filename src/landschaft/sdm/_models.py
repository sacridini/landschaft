# SPDX-License-Identifier: GPL-3.0-or-later
"""Fitting, cross-validating and projecting species distribution models."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .._save import Saveable
from ._data import SDMData
from ._evaluate import boyce, evaluate, omission_rate

__all__ = ["SDMResult", "fit", "importance", "predict", "response_curves"]


def _clone(model):
    try:
        from sklearn.base import clone

        return clone(model)
    except Exception:  # not a scikit-learn estimator (or scikit-learn missing)
        return copy.deepcopy(model)


def _accepts_weight(model) -> bool:
    try:
        return "sample_weight" in inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        return False


def _weight_argument(model) -> str | None:
    """How the model's fit takes sample weights: ``sample_weight``, ``<last step>__sample_weight`` for a
    scikit-learn Pipeline (whose own fit only routes them), or None."""
    steps = getattr(model, "steps", None)
    if steps:
        name, last = steps[-1]
        return f"{name}__sample_weight" if last != "passthrough" and _accepts_weight(last) else None
    return "sample_weight" if _accepts_weight(model) else None


def _fit_one(model, X: pd.DataFrame, y: np.ndarray, w):
    m = _clone(model)
    arg = _weight_argument(m) if w is not None else None
    if arg is not None:
        m.fit(X, y, **{arg: w})
    else:
        m.fit(X, y)
    return m


def _suitability(model, X: pd.DataFrame) -> np.ndarray:
    """The model's suitability (probability of presence) for each row."""
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        return p[:, 1] if p.ndim == 2 else p
    if hasattr(model, "decision_function"):
        d = np.asarray(model.decision_function(X), dtype=float)
        return 1 / (1 + np.exp(-d))
    return np.asarray(model.predict(X), dtype=float)


def _default_models() -> dict:
    try:
        from ._maxnet import MaxNet
    except ImportError as err:  # pragma: no cover - until MaxNet is built
        raise ValueError("give models: a dict of scikit-learn estimators (name -> estimator)") from err
    return {"maxnet": MaxNet()}


@dataclass
class SDMResult(Saveable):
    """Fitted models, their cross-validated scores and what is needed to project them.

    Attributes
    ----------
    models : dict
        Each model fitted to all the data (name -> estimator).
    metrics : pandas.DataFrame
        One row per model and fold: ``auc_train``, ``auc_test``,
        ``auc_diff`` (overfitting), ``tss`` (maximum, on the test fold),
        ``boyce`` (test presences against the background), ``or_mtp`` and
        ``or_10p`` (omission rates).
    summary : pandas.DataFrame
        Their mean and standard deviation per model.
    thresholds : dict
        Per model, the maximum-TSS threshold of its predictions on all data.
    predictors : list of str
        The predictors, in the order the models take them.
    ranges : pandas.DataFrame
        Minimum and maximum of each predictor in the data (for clamping).
    """

    models: dict
    metrics: pd.DataFrame
    summary: pd.DataFrame
    thresholds: dict
    predictors: list
    ranges: pd.DataFrame
    data: SDMData | None = field(default=None, repr=False)

    def weights(self, metric: str = "auc_test") -> dict:
        """Ensemble weights: each model's mean test score above chance (AUC - 0.5, TSS, Boyce), 0 if below."""
        if metric not in self.summary.columns.get_level_values(0):
            raise ValueError(f"unknown metric {metric!r}")
        base = 0.5 if metric.startswith("auc") else 0.0
        w = {m: max(float(self.summary.loc[m, (metric, "mean")]) - base, 0.0) for m in self.models}
        if not np.isfinite(list(w.values())).all() or sum(w.values()) == 0:
            return {m: 1.0 for m in self.models}
        return w

    def _save_items(self) -> dict:
        return {"metrics": self.metrics, "summary": self.summary.reset_index(), "thresholds": self.thresholds,
                "ranges": self.ranges.reset_index()}


def _fold_metrics(pred: np.ndarray, data: SDMData, train: np.ndarray, test: np.ndarray) -> dict:
    y = data.y
    tr_p, te_p = pred[train][y[train] == 1], pred[test][y[test] == 1]
    tr_b = pred[train][y[train] == 0]
    te_b = pred[test][y[test] == 0]
    bg_all = pred[y == 0]
    ref_bg = te_b if te_b.size else bg_all  # blocks hold background; random / jackknife folds do not
    out = {"auc_train": np.nan, "auc_test": np.nan, "auc_diff": np.nan, "tss": np.nan, "boyce": np.nan,
           "or_mtp": np.nan, "or_10p": np.nan, "n_test": int(te_p.size)}
    if tr_p.size and tr_b.size:
        out["auc_train"] = evaluate(tr_p, tr_b).stats["auc"]
    if te_p.size and ref_bg.size:
        e = evaluate(te_p, ref_bg)
        out["auc_test"] = e.stats["auc"]
        out["tss"] = e.stats["max_tss"]
        out["boyce"] = boyce(te_p, np.concatenate([ref_bg, te_p]))["cor"] if te_p.size > 1 else np.nan
        out["auc_diff"] = abs(out["auc_train"] - out["auc_test"])
        out["or_mtp"] = omission_rate(tr_p, te_p, "mtp")
        out["or_10p"] = omission_rate(tr_p, te_p, "p10")
    return out


def fit(data: SDMData, models=None, *, cv: bool = True) -> SDMResult:
    """Fit models to SDMData, cross-validate them on its folds, and fit them to all the data.

    Parameters
    ----------
    data
        From :func:`prepare` or :func:`dataset` (with folds for the
        cross-validated scores).
    models
        A dict of estimators (name -> estimator) with scikit-learn's API
        (``fit``; ``predict_proba`` or ``decision_function``), a list, or one
        estimator; default ``{"maxnet": ls.sdm.MaxNet()}``. Estimators whose
        ``fit`` takes ``sample_weight`` get ``data.sample_weight`` (in a
        scikit-learn Pipeline, its last step does).
    cv
        Cross-validate on ``data.folds`` (when there are folds).

    Returns
    -------
    SDMResult

    Examples
    --------
    >>> import landschaft as ls
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> data = ls.sdm.prepare(occ, env, clean=False, background=2000)
    >>> glm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    >>> res = ls.sdm.fit(data, {"glm": glm})
    >>> round(float(res.summary.loc["glm", ("auc_test", "mean")]), 2)
    0.87
    """
    if not isinstance(data, SDMData):
        raise TypeError("data must be SDMData (ls.sdm.prepare / ls.sdm.dataset)")
    if models is None:
        models = _default_models()
    elif isinstance(models, (list, tuple)):
        models = {type(m).__name__ if not hasattr(m, "steps") else type(m.steps[-1][1]).__name__: m for m in models}
    elif not isinstance(models, dict):
        models = {type(models).__name__: models}
    X, y, w = data.X, data.y, data.sample_weight
    rows = []
    if cv and data.folds is not None:
        for name, model in models.items():
            for k, (train, test) in enumerate(data.cv, start=1):
                m = _fit_one(model, X.iloc[train], y[train], None if w is None else w[train])
                pred = np.full(len(y), np.nan)
                pred[train] = _suitability(m, X.iloc[train])
                pred[test] = _suitability(m, X.iloc[test])
                rows.append({"model": name, "fold": k, **_fold_metrics(pred, data, train, test)})
    metrics = pd.DataFrame(rows)
    fitted = {name: _fit_one(model, X, y, w) for name, model in models.items()}
    thresholds = {}
    for name, m in fitted.items():
        pred = _suitability(m, X)
        thresholds[name] = evaluate(pred[y == 1], pred[y == 0]).thresholds["max_spec_sens"]
    if len(metrics):
        cols = ["auc_train", "auc_test", "auc_diff", "tss", "boyce", "or_mtp", "or_10p"]
        summary = metrics.groupby("model")[cols].agg(["mean", "std"])
    else:
        summary = pd.DataFrame(index=pd.Index(list(models), name="model"))
    ranges = pd.DataFrame({"min": X.min(), "max": X.max()})
    return SDMResult(fitted, metrics, summary, thresholds, list(data.predictors), ranges, data)


def predict(result: SDMResult, env, *, models=None, ensemble: str | None = "weighted", metric: str = "auc_test",
            clamp: bool = True, sd: bool = True, block_rows: int = 256):
    """Project fitted models onto rasters of predictors (present or future conditions).

    Predicted in blocks of rows, so rasters larger than memory (lazy,
    dask-backed: ``ls.load_raster(..., chunks="auto")``) work; the result is
    the same as predicting every cell at once.

    Parameters
    ----------
    result
        From :func:`fit`.
    env
        Predictors with the bands the models were fitted with.
    models
        Names of the models to project (default all).
    ensemble : {"weighted", "mean", "median", None}
        Combine the models: weighted by their cross-validated ``metric``
        above chance (see :meth:`SDMResult.weights`), the plain mean or the
        median; None: only each model.
    metric
        Score weighting the ensemble (``auc_test``, ``tss``, ``boyce``).
    clamp
        Clip each predictor to its range in the data first (projections do
        not extrapolate the response beyond what was observed).
    sd
        Add the standard deviation across models: where they disagree.
    block_rows
        Rows of cells predicted at a time.

    Returns
    -------
    xarray.DataArray
        Bands: ``ensemble`` and ``sd`` (with several models), then one per
        model; NaN where some predictor is missing.

    Examples
    --------
    >>> import landschaft as ls
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> data = ls.sdm.prepare(occ, env, clean=False, background=2000)
    >>> res = ls.sdm.fit(data, {"glm": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))})
    >>> suit = ls.sdm.predict(res, env)
    >>> [str(b) for b in suit.band.values], suit.shape
    (['glm'], (1, 192, 186))
    """
    names = list(result.models) if models is None else [models] if isinstance(models, str) else list(models)
    if env.ndim == 2:
        env = env.expand_dims(band=[env.name or "layer"])
    bands = [str(b) for b in env.band.values]
    missing = [p for p in result.predictors if p not in bands]
    if missing:
        raise ValueError(f"env lacks the predictors {missing}")
    env = env.isel(band=[bands.index(p) for p in result.predictors])
    ny, nx = env.shape[-2:]
    out = np.full((len(names), ny, nx), np.nan)
    nodata = env.rio.nodata
    lo = result.ranges["min"].reindex(result.predictors).to_numpy(float)
    hi = result.ranges["max"].reindex(result.predictors).to_numpy(float)
    for r0 in range(0, ny, block_rows):
        r1 = min(ny, r0 + block_rows)
        block = np.asarray(env.isel(y=slice(r0, r1)).values, dtype=float).reshape(len(result.predictors), -1).T
        if nodata is not None and not np.isnan(nodata):
            block = np.where(block == nodata, np.nan, block)
        ok = ~np.isnan(block).any(axis=1)
        if not ok.any():
            continue
        Xb = pd.DataFrame(block[ok], columns=result.predictors)
        for k, name in enumerate(names):
            m = result.models[name]
            Xk = Xb
            if clamp and not hasattr(m, "clamp"):
                Xk = pd.DataFrame(np.clip(Xb.to_numpy(), lo, hi), columns=result.predictors)
            v = np.full(block.shape[0], np.nan)
            v[ok] = _suitability(m, Xk)
            out[k, r0:r1] = v.reshape(r1 - r0, nx)
    layers, labels = [], []
    if len(names) > 1 and ensemble is not None:
        if ensemble == "weighted":
            wd = result.weights(metric) if len(result.summary.columns) else {n: 1.0 for n in names}
            w = np.array([wd[n] for n in names])
            ens = np.tensordot(w / w.sum(), out, axes=1)
        elif ensemble == "mean":
            ens = out.mean(axis=0)
        elif ensemble == "median":
            ens = np.median(out, axis=0)
        else:
            raise ValueError("ensemble must be 'weighted', 'mean', 'median' or None")
        layers.append(ens)
        labels.append("ensemble")
        if sd:
            layers.append(out.std(axis=0, ddof=1))
            labels.append("sd")
    layers.extend(out)
    labels.extend(names)
    da = xr.DataArray(np.stack(layers), dims=("band", "y", "x"),
                      coords={"band": labels, "y": env["y"], "x": env["x"]}, name="suitability")
    if env.rio.crs is not None:
        da = da.rio.write_crs(env.rio.crs)
    return da.rio.write_transform(env.rio.transform()).rio.write_nodata(np.nan)


def response_curves(result: SDMResult, *, predictors=None, n: int = 100, fixed: str = "mean") -> pd.DataFrame:
    """How each model's suitability changes along one predictor, the others held at their mean (or median).

    Returns
    -------
    pandas.DataFrame
        Long table: ``model``, ``predictor``, ``value``, ``suitability``.

    Examples
    --------
    >>> import landschaft as ls
    >>> from sklearn.linear_model import LogisticRegression
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> data = ls.sdm.prepare(occ, env, clean=False, background=2000, folds=False)
    >>> res = ls.sdm.fit(data, {"glm": LogisticRegression(max_iter=5000)})
    >>> curves = ls.sdm.response_curves(res, n=20)
    >>> curves.groupby("predictor").size().tolist()
    [20, 20, 20, 20, 20]
    """
    if result.data is None:
        raise ValueError("the result has no data (fit it with ls.sdm.fit)")
    X = result.data.X[result.predictors]
    base = X.mean() if fixed == "mean" else X.median()
    preds = result.predictors if predictors is None else ([predictors] if isinstance(predictors, str) else list(predictors))
    rows = []
    for p in preds:
        grid = np.linspace(X[p].min(), X[p].max(), n)
        Xg = pd.DataFrame(np.tile(base.to_numpy(), (n, 1)), columns=result.predictors)
        Xg[p] = grid
        for name, m in result.models.items():
            rows.append(pd.DataFrame({"model": name, "predictor": p, "value": grid, "suitability": _suitability(m, Xg)}))
    return pd.concat(rows, ignore_index=True)


def importance(result: SDMResult, *, n_repeats: int = 5, seed: int = 1) -> pd.DataFrame:
    """Permutation importance: how much AUC drops when a predictor's values are shuffled.

    On the data the models were fitted to; ``n_repeats`` shuffles per
    predictor (the mean and standard deviation of the drop).

    Returns
    -------
    pandas.DataFrame
        ``model``, ``predictor``, ``importance`` (mean drop in AUC), ``sd``.

    Examples
    --------
    >>> import landschaft as ls
    >>> from sklearn.linear_model import LogisticRegression
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> data = ls.sdm.prepare(occ, env, clean=False, background=2000, folds=False)
    >>> res = ls.sdm.fit(data, {"glm": LogisticRegression(max_iter=5000)})
    >>> imp = ls.sdm.importance(res)
    >>> imp.columns.tolist()
    ['model', 'predictor', 'importance', 'sd']
    """
    if result.data is None:
        raise ValueError("the result has no data (fit it with ls.sdm.fit)")
    X = result.data.X[result.predictors].reset_index(drop=True)
    y = result.data.y
    rng = np.random.default_rng(seed)
    rows = []
    for name, m in result.models.items():
        p0 = _suitability(m, X)
        auc0 = evaluate(p0[y == 1], p0[y == 0]).stats["auc"]
        for col in result.predictors:
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                p = _suitability(m, Xp)
                drops.append(auc0 - evaluate(p[y == 1], p[y == 0]).stats["auc"])
            rows.append({"model": name, "predictor": col, "importance": float(np.mean(drops)),
                         "sd": float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0})
    return pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)
