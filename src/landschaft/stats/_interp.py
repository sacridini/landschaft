# SPDX-License-Identifier: GPL-3.0-or-later
"""Interpolation of points onto a map, as R's gstat: variograms, inverse distance weighting, ordinary kriging."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ._common import read_values, wrap

__all__ = ["Kriging", "Variogram", "VariogramModel", "idw", "kriging", "variogram"]

_SHAPES = {"spherical": 0, "sph": 0, "exponential": 1, "exp": 1, "gaussian": 2, "gau": 2}
_NAMES = ("spherical", "exponential", "gaussian")


def _shape(s: str) -> int:
    k = _SHAPES.get(str(s).lower())
    if k is None:
        raise ValueError(f"model: spherical, exponential or gaussian, not {s!r}")
    return k


@dataclass
class VariogramModel(Saveable):
    """A variogram model: a nugget and a structure of a shape, its partial sill and range (gstat's ``vgm``).

    Attributes
    ----------
    shape : str
        ``"spherical"``, ``"exponential"`` (its practical range 3 times
        ``range``) or ``"gaussian"`` (√3 times), gstat's Sph, Exp and Gau.
    psill : float
        The structure's partial sill.
    range : float
        Its range.
    nugget : float
        The nugget (default 0).

    Examples
    --------
    >>> import landschaft as ls
    >>> m = ls.stats.VariogramModel("spherical", psill=1.2, range=30, nugget=0.2)
    >>> m.gamma([0, 15, 30, 60]).round(4).tolist()
    [0.0, 1.025, 1.4, 1.4]
    """

    shape: str
    psill: float
    range: float
    nugget: float = 0.0

    def __post_init__(self):
        self.shape = _NAMES[_shape(self.shape)]

    def gamma(self, h) -> np.ndarray:
        """The semivariance at distances ``h``."""
        h = np.asarray(h, dtype=np.float64)
        return _core.stats_vario_gamma(h, _shape(self.shape), float(self.nugget), float(self.psill),
                                       float(self.range))

    def covariance(self, h) -> np.ndarray:
        """The covariance at distances ``h``: the sills less the semivariance."""
        return self.nugget + self.psill - self.gamma(h)


@dataclass
class Variogram(Saveable):
    """The sample variogram of points and the model fitted to it (:func:`variogram`).

    Attributes
    ----------
    table : pandas.DataFrame
        A row per distance class with pairs: ``np`` (the pairs), ``dist``
        (their mean distance), ``gamma`` (the semivariance) and ``model`` (the
        model's at ``dist``, NaN without a model).
    model : VariogramModel or None
        The model fitted, for :func:`kriging`.
    sserr : float
        The fit's weighted sum of squared errors.
    converged : bool
        Whether the fit converged in 200 iterations.
    """

    table: pd.DataFrame
    model: VariogramModel | None
    sserr: float
    converged: bool

    def __repr__(self) -> str:
        m = self.model
        fit = "no model" if m is None else (f"{m.shape}: nugget {m.nugget:.4g}, psill {m.psill:.4g}, "
                                            f"range {m.range:.4g}")
        return f"Variogram({len(self.table)} classes, {fit})"


def _points(points, field, crs=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, values) of the points with a value, in ``crs`` when the points and ``crs`` have one."""
    if hasattr(points, "geometry"):
        g = points.geometry
        if getattr(points, "crs", None) is not None and crs is not None:
            g = g.to_crs(crs)
        kinds = set(g.geom_type.dropna())
        if kinds - {"Point"}:
            raise ValueError(f"points: points, not {', '.join(sorted(kinds - {'Point'}))}")
        if field is None:
            numeric = [c for c in points.columns if c != points.geometry.name
                       and pd.api.types.is_numeric_dtype(points[c]) and not pd.api.types.is_bool_dtype(points[c])]
            if len(numeric) != 1:
                raise ValueError(f"field: the column of the values, one of {numeric}")
            field = numeric[0]
        if field not in points.columns:
            raise ValueError(f"field: no column {field!r}")
        x, y = g.x.to_numpy(np.float64), g.y.to_numpy(np.float64)
        z = points[field].to_numpy(np.float64)
    else:
        a = np.asarray(points, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 3:
            raise TypeError("points: a GeoDataFrame of points, or an (n, 3) array of x, y and the value")
        x, y, z = a[:, 0], a[:, 1], a[:, 2]
    ok = ~(np.isnan(z) | np.isnan(x) | np.isnan(y))
    if ok.sum() < 2:
        raise ValueError("points: two with a value at least")
    return (np.ascontiguousarray(x[ok]), np.ascontiguousarray(y[ok]), np.ascontiguousarray(z[ok]))


def variogram(points, field: str | None = None, *, cutoff: float | None = None, width: float | None = None,
              model: str | None = "spherical", psill: float | None = None, range: float | None = None,
              nugget: float | None = None, fit_nugget: bool = True) -> Variogram:
    """The sample variogram of points and a model fitted to it (R's gstat ``variogram`` and ``fit.variogram``).

    Half the mean squared difference of the pairs of points at a distance
    (lower, upper] of each class up to ``cutoff`` (Matheron's estimator), and a
    nugget and a spherical, exponential or Gaussian structure fitted by
    weighted least squares (gstat's default: the pairs over the squared
    distance as weights; a negative sill fixed at zero and fitted again), the
    same numbers: the sample to the bit, the fit to the bit when it converges.

    Parameters
    ----------
    points
        A GeoDataFrame of points, or an ``(n, 3)`` array of x, y and the value.
        Points without a value are left out.
    field
        The column of the values (default: the only numeric column).
    cutoff
        The largest distance (default: a third of the diagonal of the points'
        bounding box, as gstat).
    width
        The classes' width (default: ``cutoff / 15``, as gstat).
    model
        The structure fitted: ``"spherical"`` (default), ``"exponential"``,
        ``"gaussian"`` or None (only the sample).
    psill, range, nugget
        The values the fit starts from (default: the values' variance, a third
        of ``cutoff``, a tenth of the variance).
    fit_nugget
        Fit the nugget (default), else keep it.

    Returns
    -------
    Variogram

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> g = np.random.default_rng(1)
    >>> x, y = g.random(200) * 100, g.random(200) * 100
    >>> pts = np.c_[x, y, np.sin(x / 15) + np.cos(y / 15) + g.normal(0, 0.1, 200)]
    >>> v = ls.stats.variogram(pts, model="gaussian")
    >>> v.model.shape, round(v.model.range), v.converged
    ('gaussian', 40, True)
    """
    x, y, z = _points(points, field)
    if cutoff is None:
        cutoff = _core.stats_default_cutoff(x, y)
    if width is None:
        width = cutoff / 15.0
    if not (cutoff > 0 and width > 0):
        raise ValueError("cutoff and width: above 0")
    n_pairs, dist, gamma = _core.stats_variogram(x, y, z, float(cutoff), float(width))
    fitted, sserr, converged = None, np.nan, True
    if model is not None:
        shape = _shape(model)
        var = float(np.sum((z - z.sum() / len(z)) ** 2) / (len(z) - 1))
        start = (var / 10.0 if nugget is None else float(nugget), var if psill is None else float(psill),
                 cutoff / 3.0 if range is None else float(range))
        nug, ps, rg, sserr, singular, converged = _core.stats_fit_variogram(
            n_pairs, dist, gamma, shape, *start, bool(fit_nugget), True, True)
        if singular:
            raise ValueError("the variogram's fit is singular: try other starting values, or another model")
        if not converged:
            warnings.warn("the variogram's fit did not converge in 200 iterations", RuntimeWarning, stacklevel=2)
        fitted = VariogramModel(_NAMES[shape], psill=ps, range=rg, nugget=nug)
    table = pd.DataFrame({"np": n_pairs.astype(np.int64), "dist": dist, "gamma": gamma,
                          "model": fitted.gamma(dist) if fitted is not None else np.full(len(dist), np.nan)})
    return Variogram(table, fitted, float(sserr), bool(converged))


def _places(like, nodata):
    """(the cells with a value: flat indices, x and y of their centres; values; template) of the map to fill."""
    values, tpl = read_values(like, nodata, "like")
    if tpl is None or "x" not in tpl.coords or "y" not in tpl.coords:
        raise ValueError("like: a map with x and y coordinates (a DataArray from ls.load_raster)")
    cx, cy = np.meshgrid(np.asarray(tpl["x"], np.float64), np.asarray(tpl["y"], np.float64))
    cells = np.flatnonzero(~np.isnan(values.ravel()))
    return cells, np.ascontiguousarray(cx.ravel()[cells]), np.ascontiguousarray(cy.ravel()[cells]), values, tpl


def _crs(tpl):
    try:
        return tpl.rio.crs
    except Exception:  # noqa: BLE001  (no rioxarray georeference)
        return None


def _nmax(nmax) -> int:
    if nmax is None:
        return 0
    if int(nmax) < 1:
        raise ValueError("nmax: 1 or more")
    return int(nmax)


def idw(points, like, field: str | None = None, *, power: float = 2.0, nmax: int | None = None, nodata=None,
        n_threads: int | None = None) -> xr.DataArray:
    """Inverse distance weighting of points onto a map (R's gstat ``idw``, the same numbers).

    Every cell of ``like`` with a value gets the mean of the points' values
    weighted by one over the distance from its centre to the power, with every
    point or the ``nmax`` nearest; a point's value at its own place. The points
    are reprojected to the map's CRS.

    Parameters
    ----------
    points
        A GeoDataFrame of points, or an ``(n, 3)`` array of x, y and the value.
    like
        The map whose cells to fill (those with a value; NaN and ``nodata``
        stay NaN).
    field
        The column of the values (default: the only numeric column).
    power
        The distance's power (default 2).
    nmax
        The nearest points used (default all).
    nodata
        Value marking the cells of ``like`` not to fill.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        The values interpolated, on ``like``'s grid.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> like = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"), coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]})
    >>> ls.stats.idw(np.array([[0.5, 2.5, 1.0], [3.5, 0.5, 3.0]]), like).values.round(2)
    array([[1.  , 1.22, 1.89, 2.38],
           [1.18, 1.57, 2.43, 2.82],
           [1.62, 2.11, 2.78, 3.  ]])
    """
    if power < 0:
        raise ValueError("power: 0 or more")
    cells, px, py, values, tpl = _places(like, nodata)
    x, y, z = _points(points, field, _crs(tpl))
    out = np.full(values.size, np.nan)
    out[cells] = _core.stats_idw(x, y, z, px, py, float(power), _nmax(nmax), resolve(n_threads))
    return wrap(out.reshape(values.shape), tpl, "idw")


@dataclass
class Kriging(Saveable):
    """Ordinary kriging of points onto a map (:func:`kriging`).

    Attributes
    ----------
    prediction : xarray.DataArray
        The values predicted.
    variance : xarray.DataArray
        The prediction's variance (0 at a point's own place).
    """

    prediction: xr.DataArray
    variance: xr.DataArray

    def __repr__(self) -> str:
        return f"Kriging({int(np.isfinite(np.asarray(self.prediction)).sum())} cells)"


def kriging(points, like, model, field: str | None = None, *, nmax: int | None = None, nodata=None,
            n_threads: int | None = None) -> Kriging:
    """Ordinary kriging of points onto a map (R's gstat ``krige``, to rounding: 1e-10).

    Every cell of ``like`` with a value gets the best linear unbiased
    prediction from the points' values under a variogram model, with every
    point (the system factored once) or the ``nmax`` nearest each cell, and
    the prediction's variance. The points are reprojected to the map's CRS.

    Parameters
    ----------
    points
        A GeoDataFrame of points, or an ``(n, 3)`` array of x, y and the value.
    like
        The map whose cells to fill (those with a value).
    model
        The variogram model: a :class:`VariogramModel`, the :class:`Variogram`
        fitted by :func:`variogram`, or a dict ``{"shape", "psill", "range",
        "nugget"}``.
    field
        The column of the values (default: the only numeric column).
    nmax
        The nearest points used (default all).
    nodata
        Value marking the cells of ``like`` not to fill.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    Kriging

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> like = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"), coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]})
    >>> pts = np.array([[0.5, 2.5, 1.0], [3.5, 0.5, 3.0], [2.0, 2.0, 2.5]])
    >>> k = ls.stats.kriging(pts, like, ls.stats.VariogramModel("exponential", psill=1, range=2))
    >>> float(k.prediction[0, 0]), float(k.variance[0, 0])
    (1.0, 0.0)
    """
    if isinstance(model, Variogram):
        model = model.model
        if model is None:
            raise ValueError("model: the variogram has no model fitted")
    elif isinstance(model, dict):
        model = VariogramModel(**model)
    if not isinstance(model, VariogramModel):
        raise TypeError("model: a VariogramModel, a Variogram or a dict {shape, psill, range, nugget}")
    if not (model.range > 0 and model.psill >= 0 and model.nugget >= 0 and model.psill + model.nugget > 0):
        raise ValueError("model: a range above 0 and sills of 0 or more, not all 0")
    cells, px, py, values, tpl = _places(like, nodata)
    x, y, z = _points(points, field, _crs(tpl))
    pred, var = _core.stats_krige(x, y, z, _shape(model.shape), float(model.nugget), float(model.psill),
                                  float(model.range), px, py, _nmax(nmax), resolve(n_threads))
    p, v = np.full(values.size, np.nan), np.full(values.size, np.nan)
    p[cells], v[cells] = pred, np.maximum(var, 0.0)
    return Kriging(wrap(p.reshape(values.shape), tpl, "prediction"), wrap(v.reshape(values.shape), tpl, "variance"))
