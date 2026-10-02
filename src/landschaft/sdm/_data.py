# SPDX-License-Identifier: GPL-3.0-or-later
"""The data a species distribution model is fitted to: SDMData."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .._save import Saveable
from . import _points

__all__ = ["SDMData", "dataset"]


@dataclass
class SDMData(Saveable):
    """Presences and background (or absences) with their predictor values, ready for a model.

    Every scikit-learn estimator takes it as it is::

        model.fit(data.X, data.y, sample_weight=data.sample_weight)
        GridSearchCV(model, grid, cv=data.cv).fit(data.X, data.y)

    Attributes
    ----------
    X : pandas.DataFrame
        One row per record, one column per predictor.
    y : numpy.ndarray
        1 for a presence, 0 for a background point or absence.
    coords : geopandas.GeoDataFrame
        The points (CRS of the predictors), with ``y``.
    sample_weight : numpy.ndarray or None
        Weight of each row (see :func:`dataset`).
    folds : numpy.ndarray or None
        Cross-validation fold of each row (1..k; 0 = in every training set).
    predictors : list of str
        The predictor names (columns of ``X``).
    categorical : list of str
        The categorical predictors.
    area : geopandas.GeoDataFrame or None
        The accessible area the background was drawn from.
    report : pandas.DataFrame
        One row per preparation step: records before and after, and why.
    settings : dict
        Every choice made (thresholds, distances, seeds): enough to repeat
        the preparation or to write the methods of a paper.
    """

    X: pd.DataFrame
    y: np.ndarray
    coords: object
    sample_weight: np.ndarray | None = None
    folds: np.ndarray | None = None
    predictors: list = field(default_factory=list)
    categorical: list = field(default_factory=list)
    area: object = None
    report: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings: dict = field(default_factory=dict)
    _prepare: dict | None = field(default=None, repr=False)

    @property
    def n_presences(self) -> int:
        return int((self.y == 1).sum())

    @property
    def n_background(self) -> int:
        return int((self.y == 0).sum())

    @property
    def cv(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """The folds as scikit-learn takes them: a list of (train rows, test rows).

        Rows with fold 0 (e.g. background points in a leave-one-out design)
        are always in training.
        """
        if self.folds is None:
            raise ValueError("no folds: make them with ls.sdm.spatial_blocks() / random_folds() (folds=...)")
        f = np.asarray(self.folds)
        ks = [k for k in np.unique(f) if k > 0]
        return [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in ks]

    def with_folds(self, folds) -> SDMData:
        """A copy with these folds (an array, one fold per row)."""
        f = np.asarray(folds)
        if f.shape != (len(self.y),):
            raise ValueError(f"folds must have one value per row ({len(self.y)})")
        out = self._copy()
        out.folds = f.astype(np.int64)
        return out

    def redo(self, **changes) -> SDMData:
        """Prepare again from the same inputs, with some steps changed.

        Only for data made by :func:`prepare`. ``data.redo(thin=40_000)``
        repeats every step with the same settings (and seeds) except the
        thinning distance; any argument of :func:`prepare` can be changed.
        """
        if self._prepare is None:
            raise ValueError("redo() needs data made by ls.sdm.prepare()")
        from ._prepare import prepare

        args = {**self._prepare["kwargs"], **changes}
        return prepare(self._prepare["occ"], self._prepare["env"], **args)

    def _copy(self) -> SDMData:
        return SDMData(self.X.copy(), self.y.copy(), self.coords.copy(),
                       None if self.sample_weight is None else self.sample_weight.copy(),
                       None if self.folds is None else self.folds.copy(), list(self.predictors),
                       list(self.categorical), self.area, self.report.copy(), dict(self.settings), self._prepare)

    def table(self) -> pd.DataFrame:
        """Everything as one table: coordinates, ``y``, weight, fold and the predictors."""
        x, y = self.coords.geometry.x.to_numpy(), self.coords.geometry.y.to_numpy()
        out = pd.DataFrame({"x": x, "y_coord": y, "y": self.y})
        if self.sample_weight is not None:
            out["weight"] = self.sample_weight
        if self.folds is not None:
            out["fold"] = self.folds
        return pd.concat([out, self.X.reset_index(drop=True)], axis=1)

    def _save_items(self) -> dict:
        items = {"data": self.table(), "report": self.report, "settings": self.settings}
        if self.area is not None:
            items["area"] = self.area
        return items

    def __repr__(self) -> str:
        folds = "none" if self.folds is None else str(len([k for k in np.unique(self.folds) if k > 0]))
        return (f"SDMData({self.n_presences} presences, {self.n_background} background, "
                f"{len(self.predictors)} predictors: {', '.join(self.predictors)}; folds: {folds})")


def _values_at(env: xr.DataArray, gdf):
    from .._zonal import extract

    vals = extract(env, gdf)
    names = [str(b) for b in env["band"].values] if "band" in env.dims else [str(env.name or "value")]
    cols = [c for c in vals.columns if str(c) in names]
    return pd.DataFrame(vals[cols].to_numpy(dtype=float), columns=names)


def dataset(occ, background, env, *, weights: str | None = "balanced", folds=None, categorical=None,
            x: str | None = None, y: str | None = None, crs=None) -> SDMData:
    """Put presences, background points and predictor values together.

    Parameters
    ----------
    occ
        Presence records (GeoDataFrame of points, or a table with coordinate
        columns). A table with a column ``y`` / ``presence`` of 1 and 0 is
        taken as presences and absences, and ``background`` may be None.
    background
        Background points or absences (e.g. :func:`background`'s), or None.
    env
        Predictors (raster stack): their values at the points become ``X``.
    weights : {"balanced", "equal", None}
        ``"balanced"`` (default): presences weigh 1 and the background
        ``n_presences / n_background`` each, so that both sides weigh the
        same (as recommended for presence-background data, Barbet-Massin et
        al. 2012); ``"equal"`` / None: all 1.
    folds
        Cross-validation folds, one per row (or set them later with
        :meth:`SDMData.with_folds`).
    categorical
        Names of categorical predictors (codes, not quantities).
    x, y, crs
        Coordinate columns and CRS of tables.

    Returns
    -------
    SDMData
        Rows without a value in some predictor are dropped (counted in
        ``report``).

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> bg = ls.sdm.background(env, 2000, occ=occ, seed=1)
    >>> data = ls.sdm.dataset(occ, bg, env)
    >>> data
    SDMData(116 presences, 2000 background, 9 predictors: bio1, bio5, bio6, bio7, bio8, bio9, bio12, bio16, bio17; folds: none)
    """
    import geopandas as gpd

    if env.ndim == 2:
        env = env.expand_dims(band=[env.name or "layer"])
    crs_env = env.rio.crs

    def points(obj):
        xs, ys, c = _points.coordinates(obj, x=x, y=y, crs=crs)
        xs, ys = _points.to_crs(xs, ys, c, crs_env)
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=crs_env)

    pres = points(occ)
    yvals = np.ones(len(pres))
    label = next((c for c in ("y", "presence", "pa") if isinstance(occ, pd.DataFrame) and c in occ.columns), None)
    if label is not None:
        yvals = pd.to_numeric(occ[label], errors="coerce").to_numpy(dtype=float)
    parts = [pres]
    ys = [yvals]
    if background is not None:
        parts.append(points(background))
        ys.append(np.zeros(len(parts[-1])))
    allpts = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=crs_env)
    yall = np.concatenate(ys)
    X = _values_at(env, allpts)
    ok = X.notna().all(axis=1).to_numpy() & np.isfinite(yall) & allpts.geometry.notna().to_numpy()
    report = pd.DataFrame([{"step": "predictor values", "presences": int((yall[ok] == 1).sum()),
                            "background": int((yall[ok] == 0).sum()), "removed": int((~ok).sum()),
                            "reason": "no value in some predictor"}])
    X = X[ok].reset_index(drop=True)
    yall = yall[ok].astype(np.int8)
    coords = allpts[ok].reset_index(drop=True)
    coords["y"] = yall
    w = None
    if weights == "balanced":
        npres, nbg = int((yall == 1).sum()), int((yall == 0).sum())
        w = np.where(yall == 1, 1.0, npres / nbg if nbg else 1.0)
    elif weights not in (None, "equal"):
        raise ValueError("weights must be 'balanced', 'equal' or None")
    cats = [str(c) for c in (categorical or [])]
    unknown = set(cats) - set(X.columns)
    if unknown:
        raise ValueError(f"unknown categorical predictors: {sorted(unknown)}")
    f = None
    if folds is not None:
        f = np.asarray(folds)[ok] if len(folds) == len(ok) else np.asarray(folds)
    return SDMData(X, yall, coords, w, f, list(X.columns), cats, None, report, {"weights": weights})
