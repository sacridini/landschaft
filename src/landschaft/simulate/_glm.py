# SPDX-License-Identifier: GPL-3.0-or-later
"""Logistic regression of land use on explanatory variables, as lulcc fits it (R's glm).

``glmModels`` of lulcc fits, for each land use class, R's
``glm(class ~ drivers, family = binomial)`` to the presence of the class in
the cells; the fitted probabilities are the suitability maps that CLUE-S
and the ordered model allocate on. The fit is R's ``glm.fit`` reproduced
operation for operation (src/cpp/sim/rglm.cpp): the same coefficients,
deviances and predictions, bit for bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._threads import threaded
from ._change import _class_value, _classes
from ._maps import read_map
from ._woe import _drivers

__all__ = ["LogisticRegression"]


class LogisticRegression:
    """Logistic regression of the presence of each land use class on the drivers (lulcc's ``glmModels``).

    Parameters
    ----------
    formulas
        The drivers of each class, ``{class: [driver, ...]}`` (default: every
        class of the map on every driver), as the right-hand sides of lulcc's
        formulas ``class ~ d1 + d2``; the intercept is always included.
    maxit, epsilon
        glm.fit's iteration limit and convergence tolerance (R's defaults).

    Examples
    --------
    >>> import landschaft as ls
    >>> lu = ls.load_raster(ls.simulate.example_data("pie_landuse"))
    >>> ef = ls.load_raster(ls.simulate.example_data("pie_drivers"))
    >>> lr = ls.simulate.LogisticRegression({2: ["elevation", "slope", "dist_built_1985"]}).fit(lu[0], ef)
    >>> lr.coefficients.columns.tolist()
    ['(Intercept)', 'elevation', 'slope', 'dist_built_1985']
    """

    def __init__(self, formulas: dict | None = None, *, maxit: int = 25, epsilon: float = 1e-8):
        self.formulas = None if formulas is None else {_class_value(k): list(v) for k, v in formulas.items()}
        self.maxit = int(maxit)
        self.epsilon = float(epsilon)
        self._fits: dict = {}

    @threaded
    def fit(self, landscape, drivers, *, cells=None, nodata=None) -> LogisticRegression:
        """Fit a model per class: presence (1) or absence (0) of the class in the cells.

        Parameters
        ----------
        landscape
            The categorical map (lulcc fits on the map of the first date).
        drivers
            The explanatory variables: a Dataset, a ``dict`` of maps or a
            DataArray with a band per variable.
        cells
            The cells to fit on: a boolean map or flat indices (row by row);
            default every cell. Cells missing in the landscape or in a
            driver of the model are left out (glm's ``na.omit``).
        nodata
            Value marking missing cells in the landscape.

        Returns
        -------
        LogisticRegression
            ``self``, fitted.
        """
        land = read_map(landscape, nodata)
        maps = _drivers(drivers)
        lv = land.as_float().ravel()
        use = np.ones(lv.size, dtype=bool)
        if cells is not None:
            c = np.asarray(cells)
            if c.dtype == bool:
                use = c.ravel().copy()
            else:
                use = np.zeros(lv.size, dtype=bool)
                use[c.ravel().astype(np.int64)] = True
        use &= ~np.isnan(lv)
        formulas = self.formulas
        if formulas is None:
            formulas = {_class_value(c): sorted(maps) for c in _classes([land])}
        unknown = {d for v in formulas.values() for d in v} - set(maps)
        if unknown:
            raise ValueError(f"no driver named {sorted(unknown)}")
        values = {k: m.as_float().ravel() for k, m in maps.items()}
        self._fits = {}
        for cls, names in formulas.items():
            ok = use.copy()
            for nm in names:
                ok &= ~np.isnan(values[nm])
            idx = np.flatnonzero(ok)
            x = np.empty((len(names) + 1, idx.size))
            x[0] = 1.0
            for j, nm in enumerate(names, start=1):
                x[j] = values[nm][idx]
            y = (lv[idx] == cls).astype(np.float64)
            fit = _core.sim_glm_logit(x, y, self.maxit, self.epsilon)
            fit["drivers"] = list(names)
            fit["n"] = int(idx.size)
            self._fits[cls] = fit
        self.drivers = sorted({d for v in formulas.values() for d in v})
        return self

    @classmethod
    def from_coefficients(cls, coefficients) -> LogisticRegression:
        """A model with given coefficients (CLUMondo's and CLUE-S's ``alloc1.reg``).

        Parameters
        ----------
        coefficients
            ``{class: {"(Intercept)": b0, driver: b, ...}}`` or a DataFrame
            with a row per class and a column per term (NaN: not in the
            model), as :attr:`coefficients` gives it.

        Examples
        --------
        >>> import landschaft as ls
        >>> lr = ls.simulate.LogisticRegression.from_coefficients({1: {"(Intercept)": -1.5, "slope": 0.2}})
        >>> lr.coefficients.loc[1, "slope"]
        np.float64(0.2)
        """
        if isinstance(coefficients, pd.DataFrame):
            coefficients = {k: {t: v for t, v in row.items() if not pd.isna(v)}
                            for k, row in coefficients.iterrows()}
        out = cls()
        for k, terms in coefficients.items():
            terms = dict(terms)
            b0 = float(terms.pop("(Intercept)", 0.0))
            names = list(terms)
            p = len(names) + 1
            out._fits[_class_value(k)] = {
                "coefficients": np.array([b0, *(float(terms[n]) for n in names)]),
                "pivot": np.arange(1, p + 1, dtype=np.int32), "rank": p, "drivers": names, "n": 0,
                "iterations": 0, "converged": True, "deviance": np.nan, "null_deviance": np.nan}
        out.drivers = sorted({d for f in out._fits.values() for d in f["drivers"]})
        out.formulas = {k: list(f["drivers"]) for k, f in out._fits.items()}
        return out

    def _terms(self, cls) -> tuple[float, list, list]:
        """Intercept, drivers and coefficients of a class's model (aliased drivers left out)."""
        f = self._fits[_class_value(cls)]
        b = f["coefficients"]
        names = [n for n, v in zip(f["drivers"], b[1:]) if not np.isnan(v)]
        coefs = [v for v in b[1:] if not np.isnan(v)]
        return float(b[0]), names, coefs

    def _check(self):
        if not self._fits:
            raise RuntimeError("fit the logistic regression first")

    @property
    def classes(self) -> list:
        self._check()
        return list(self._fits)

    @property
    def coefficients(self) -> pd.DataFrame:
        """The coefficients of each class's model (NaN for aliased drivers, R's NA)."""
        self._check()
        cols = ["(Intercept)", *dict.fromkeys(d for f in self._fits.values() for d in f["drivers"])]
        rows = []
        for f in self._fits.values():
            row = dict(zip(["(Intercept)", *f["drivers"]], f["coefficients"]))
            rows.append([row.get(c, np.nan) for c in cols])
        return pd.DataFrame(rows, index=pd.Index(list(self._fits), name="class"), columns=cols)

    @property
    def summary(self) -> pd.DataFrame:
        """Per class: cells used, iterations, convergence, deviance, null deviance."""
        self._check()
        return pd.DataFrame([{"class": k, "cells": f["n"], "iterations": f["iterations"], "converged": f["converged"],
                              "deviance": f["deviance"], "null_deviance": f["null_deviance"]}
                             for k, f in self._fits.items()]).set_index("class")

    @threaded
    def predict(self, drivers, *, classes=None, landscape=None, nodata=None) -> xr.DataArray:
        """Probability of each class in every cell (predict(type = "response") of each model).

        Parameters
        ----------
        drivers
            The explanatory variables (they may have changed since the fit).
        classes
            The classes to predict (default: all fitted).
        landscape
            Optional map whose missing cells stay NaN.

        Returns
        -------
        xarray.DataArray
            ``(class, y, x)``, NaN where a driver of the model is missing.
        """
        self._check()
        maps = _drivers(drivers)
        missing = set(self.drivers) - set(maps)
        if missing:
            raise ValueError(f"drivers lack {sorted(missing)}")
        first = next(iter(maps.values()))
        shape = first.shape
        values = {k: maps[k].as_float().ravel() for k in self.drivers}
        mask = np.ones(int(np.prod(shape)), dtype=bool)
        if landscape is not None:
            mask &= ~np.isnan(read_map(landscape, nodata).as_float().ravel())
        cls = list(self._fits) if classes is None else [_class_value(c) for c in classes]
        layers = []
        for c in cls:
            f = self._fits[c]
            ok = mask.copy()
            for nm in f["drivers"]:
                ok &= ~np.isnan(values[nm])
            idx = np.flatnonzero(ok)
            x = np.empty((len(f["drivers"]) + 1, idx.size))
            x[0] = 1.0
            for j, nm in enumerate(f["drivers"], start=1):
                x[j] = values[nm][idx]
            out = np.full(mask.size, np.nan)
            out[idx] = _core.sim_glm_predict(x, f["coefficients"], f["pivot"], f["rank"])
            layers.append(out.reshape(shape))
        t = first.template
        data = np.stack(layers)
        if t is not None:
            da = xr.DataArray(data, dims=("class", *t.dims), coords={"class": cls, **{k: t.coords[k] for k in t.dims}},
                              name="probability")
            try:
                from ..io import _with_georef

                da = _with_georef(da, transform=t.rio.transform())
                if t.rio.crs is not None:
                    da = _with_georef(da, crs=t.rio.crs)
            except Exception:
                pass
            return da
        return xr.DataArray(data, dims=("class", "y", "x"), coords={"class": cls}, name="probability")

    probability = predict
    potential = predict

    def __repr__(self) -> str:
        return f"LogisticRegression({len(self._fits)} classes fitted)" if self._fits else "LogisticRegression()"
