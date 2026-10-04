# SPDX-License-Identifier: GPL-3.0-or-later
"""Transition potential from any scikit-learn classifier, or from given rasters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .._threads import threaded
from ._change import _class_value, _classes
from ._maps import read_map
from ._woe import _drivers

__all__ = ["Estimator", "RasterPotential"]


def _stack_like(layers, names, dim, template):
    data = np.stack(layers)
    if template is not None:
        da = xr.DataArray(data, dims=(dim, *template.dims),
                          coords={dim: names, **{k: template.coords[k] for k in template.dims}}, name="probability")
        try:
            from ..io import _with_georef

            da = _with_georef(da, transform=template.rio.transform())
            if template.rio.crs is not None:
                da = _with_georef(da, crs=template.rio.crs)
        except Exception:
            pass
        return da
    return xr.DataArray(data, dims=(dim, "y", "x"), coords={dim: names}, name="probability")


class Estimator:
    """Potential of each class from a scikit-learn classifier, one model per class.

    Parameters
    ----------
    estimator
        A scikit-learn classifier (or any object with ``fit`` and
        ``predict_proba``); a clone is fitted for each class.
    target : {"presence", "expansion"}
        What the models learn: ``"presence"``, where each class is in the map
        (as lulcc's models); ``"expansion"``, where it appeared between two
        maps among the cells that did not have it (PLUS's land expansion
        analysis, LEAS), which needs the later map in :meth:`fit`.
    classes
        The classes to model (default: all).
    sample
        Cells to train each model on: ``None`` (all), an ``int`` (at most this
        many, drawn at random) or a fraction in (0, 1).
    balance
        Draw as many absences as presences (after ``sample``).
    seed
        Seed of the draws (numpy's generator).

    Examples
    --------
    >>> import landschaft as ls
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> lu = ls.load_raster(ls.simulate.example_data("pie_landuse"))
    >>> ef = ls.load_raster(ls.simulate.example_data("pie_drivers"))
    >>> est = ls.simulate.Estimator(RandomForestClassifier(n_estimators=20, random_state=0),
    ...                             target="expansion", classes=[2], sample=5000, seed=1)
    >>> p = est.fit(lu[0], ef, t1=lu[2]).predict(ef)
    >>> p.dims
    ('class', 'y', 'x')
    """

    def __init__(self, estimator, *, target: str = "presence", classes=None, sample=None, balance: bool = False,
                 seed=None):
        if target not in ("presence", "expansion"):
            raise ValueError("target must be 'presence' or 'expansion'")
        self.estimator = estimator
        self.target = target
        self.classes = None if classes is None else [_class_value(c) for c in classes]
        self.sample = sample
        self.balance = bool(balance)
        self.seed = seed
        self.models: dict = {}
        self.drivers: list[str] = []
        self._train: dict = {}

    def _draw(self, idx: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
        if self.sample is not None:
            n = int(self.sample) if self.sample >= 1 else round(self.sample * idx.size)
            if n < idx.size:
                pick = rng.choice(idx.size, size=n, replace=False)
                pick.sort()
                idx, y = idx[pick], y[pick]
        if self.balance:
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            m = min(pos.size, neg.size)
            keep = np.sort(np.concatenate([rng.choice(pos, m, replace=False), rng.choice(neg, m, replace=False)]))
            idx = idx[keep]
        return idx

    @threaded
    def fit(self, landscape, drivers, *, t1=None, cells=None, nodata=None) -> Estimator:
        """Fit a model per class.

        Parameters
        ----------
        landscape
            The map at the start.
        drivers
            Explanatory variables (Dataset, ``dict`` or DataArray with a band
            per variable).
        t1
            The later map (``target="expansion"``).
        cells
            Optional boolean map of the cells to train on.
        nodata
            Value marking missing cells in the maps.
        """
        from sklearn.base import clone

        a = read_map(landscape, nodata)
        maps = _drivers(drivers)
        self.drivers = sorted(maps)
        x_all = np.column_stack([maps[k].as_float().ravel() for k in self.drivers])
        av = a.as_float().ravel()
        valid = ~np.isnan(av) & ~np.isnan(x_all).any(axis=1)
        if cells is not None:
            valid &= np.asarray(cells, dtype=bool).ravel()
        bv = None
        if self.target == "expansion":
            if t1 is None:
                raise ValueError("target='expansion' needs the later map t1")
            bv = read_map(t1, nodata).as_float().ravel()
            valid &= ~np.isnan(bv)
        classes = self.classes or [_class_value(c) for c in _classes([a] if bv is None else [a, read_map(t1, nodata)])]
        rng = np.random.default_rng(self.seed)
        self.models = {}
        self._train = {}
        for c in classes:
            if self.target == "presence":
                idx = np.flatnonzero(valid)
                y = (av[idx] == c).astype(int)
            else:
                idx = np.flatnonzero(valid & (av != c))
                y = (bv[idx] == c).astype(int)
            idx = self._draw(idx, y, rng)
            y = (av[idx] == c).astype(int) if self.target == "presence" else (bv[idx] == c).astype(int)
            if y.min() == y.max():
                raise ValueError(f"class {c}: the training cells are all {'presences' if y[0] else 'absences'}")
            model = clone(self.estimator)
            model.fit(x_all[idx], y)
            self.models[c] = model
            self._train[c] = (idx, y)
        self._x = x_all
        self._template = a.template
        return self

    @threaded
    def predict(self, drivers, *, classes=None, landscape=None, nodata=None) -> xr.DataArray:
        """Probability of each class (``predict_proba`` of the class's model), NaN where a driver is missing."""
        if not self.models:
            raise RuntimeError("fit the estimator first")
        maps = _drivers(drivers)
        missing = set(self.drivers) - set(maps)
        if missing:
            raise ValueError(f"drivers lack {sorted(missing)}")
        x_all = np.column_stack([maps[k].as_float().ravel() for k in self.drivers])
        ok = ~np.isnan(x_all).any(axis=1)
        if landscape is not None:
            ok &= ~np.isnan(read_map(landscape, nodata).as_float().ravel())
        shape = next(iter(maps.values())).shape
        cls = list(self.models) if classes is None else [_class_value(c) for c in classes]
        layers = []
        for c in cls:
            out = np.full(ok.size, np.nan)
            proba = self.models[c].predict_proba(x_all[ok])
            col = list(self.models[c].classes_).index(1)
            out[ok] = proba[:, col]
            layers.append(out.reshape(shape))
        return _stack_like(layers, cls, "class", next(iter(maps.values())).template)

    probability = predict
    potential = predict

    def importance(self, method: str = "impurity", *, n_repeats: int = 5, seed=None) -> pd.DataFrame:
        """Contribution of each driver to each class's model (PLUS's driver contributions).

        Parameters
        ----------
        method : {"impurity", "permutation"}
            ``"impurity"``: the estimator's ``feature_importances_`` (a random
            forest's mean decrease in impurity); ``"permutation"``: the drop in
            accuracy on the training cells when a driver's values are shuffled
            (scikit-learn's ``permutation_importance``, any estimator).
        n_repeats, seed
            Shuffles per driver and their seed (``"permutation"``).

        Returns
        -------
        pandas.DataFrame
            A row per driver, a column per class.
        """
        if not self.models:
            raise RuntimeError("fit the estimator first")
        if method not in ("impurity", "permutation"):
            raise ValueError("method must be 'impurity' or 'permutation'")
        cols = {}
        for c, model in self.models.items():
            if method == "impurity":
                if not hasattr(model, "feature_importances_"):
                    raise ValueError(f"{type(model).__name__} has no feature_importances_: use "
                                     "method='permutation'")
                cols[c] = np.asarray(model.feature_importances_, dtype=np.float64)
            else:
                from sklearn.inspection import permutation_importance

                idx, y = self._train[c]
                r = permutation_importance(model, self._x[idx], y, n_repeats=n_repeats, random_state=seed)
                cols[c] = r.importances_mean
        return pd.DataFrame(cols, index=pd.Index(self.drivers, name="driver"))


class RasterPotential:
    """A potential given as maps: one layer per class (or transition), as computed elsewhere.

    Parameters
    ----------
    maps
        A DataArray with a first dimension naming the classes (``class``,
        ``band`` or ``transition`` values), or a ``dict`` ``{class: map}``.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> pot = ls.simulate.RasterPotential({1: np.full((3, 4), 0.2), 2: np.full((3, 4), 0.7)})
    >>> pot.predict().sel({"class": 2}).values.max()
    np.float64(0.7)
    """

    def __init__(self, maps):
        if isinstance(maps, dict):
            names = [_class_value(k) if not isinstance(k, (tuple, str)) else k for k in maps]
            arrays = [read_map(v) for v in maps.values()]
            self._layers = [m.as_float() for m in arrays]
            self._names = names
            self._template = arrays[0].template
        elif isinstance(maps, xr.DataArray) and maps.ndim == 3:
            dim = maps.dims[0]
            self._names = [v.item() if hasattr(v, "item") else v for v in maps[dim].values] if dim in maps.coords \
                else list(range(maps.shape[0]))
            self._layers = [read_map(maps[k]).as_float() for k in range(maps.shape[0])]
            self._template = read_map(maps[0]).template
        else:
            raise TypeError("maps must be a dict of maps or a 3-D DataArray")

    def fit(self, *args, **kwargs) -> RasterPotential:
        """Nothing to fit: the potential is given."""
        return self

    def predict(self, *args, classes=None, **kwargs) -> xr.DataArray:
        """The given maps (``classes`` selects some)."""
        names = self._names if classes is None else list(classes)
        layers = [self._layers[self._names.index(c)] for c in names]
        return _stack_like(layers, names, "class", self._template)

    probability = predict
    potential = predict
