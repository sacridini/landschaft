# SPDX-License-Identifier: GPL-3.0-or-later
"""Choosing predictors: collinearity filters of the R package usdm (Naimi et al. 2014)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from ._thin import _seed

__all__ = ["select_predictors", "vif"]

_ALL_CELLS_BELOW = 7000  # usdm: rasters (and tables) with fewer cells are used whole
_SAMPLE_SIZE = 5000


# ----------------------------------------------------------------------------------------- sampling

def _is_lonlat(env: xr.DataArray) -> bool:
    """terra's is.lonlat(perhaps = TRUE): a geographic CRS, or none and an extent within the globe."""
    crs = env.rio.crs
    if crs is not None:
        return bool(crs.is_geographic)
    from ..io import _transform

    tr = _transform(env)
    ny, nx = env.shape[-2:]
    xmin, ymax = tr.c, tr.f
    xmax, ymin = xmin + nx * tr.a, ymax + ny * tr.e
    return xmin >= -180.1 and xmax <= 180.1 and ymin >= -90.1 and ymax <= 90.1


def _row_weights(env: xr.DataArray) -> np.ndarray:
    """|cos(pi * y / 180)| of each row's centre (terra's yFromRow, R's x87 cos)."""
    from ..io import _transform

    tr = _transform(env)
    ny = env.shape[-2]
    ymax = tr.f
    ymin = ymax + ny * tr.e
    yres = (ymax - ymin) / ny
    y = ymax - (np.arange(ny) + 0.5) * yres
    return np.abs(_core.sdm_r_cos(np.ascontiguousarray((np.pi * y) / 180.0)))


def _sample_cells(env: xr.DataArray, complete: np.ndarray, size: int, rng) -> np.ndarray:
    """The cells usdm takes from a raster: terra's spatSample(x, size, na.rm = TRUE) (values).

    ``complete`` marks the cells with a value in every band (row-major). With
    terra's quirk corrected: a raster of 7000-10 000 cells is sampled at
    random too (terra used its first complete cells).
    """
    ny, nx = env.shape[-2:]
    ncell = ny * nx
    lonlat = _is_lonlat(env)
    esize = 2 * size * 5  # ssize = 2 * size candidates, times exp = 5
    if ncell <= 50 * esize:  # terra's in-memory sampler (the rasters usdm meets in practice)
        nona = np.flatnonzero(complete)
        if nona.size == 0:
            return nona
        take = min(2 * size, nona.size)  # only the first draws are used: they do not depend on the later ones
        prob = None
        if lonlat:
            prob = _row_weights(env)[nona // nx].tolist()
        i = rng.sample_int(nona.size, take, False, prob)
        return nona[i - 1][:size]
    # large rasters: rejection sampling, rounds of candidates until `size` complete cells
    found: list[int] = []
    for rnd in range(10):
        if lonlat:
            w = _row_weights(env).tolist()
            rows = rng.sample_int(ny, 2 * esize, True, w)
            cols = rng.sample_int(nx, 2 * esize, True, None)
            cells = pd.unique((rows - 1) * nx + (cols - 1))
        else:
            cells = rng.sample_int(ncell, min(ncell, esize), False, None) - 1
        cells = cells[complete[cells]][: 2 * size]
        found = list(cells) if rnd == 0 else list(pd.unique(np.concatenate([found, cells])))
        if len(found) >= size:
            break
    return np.asarray(found[:size], dtype=np.int64)


def _values(env, size: int | None, seed, area) -> tuple[pd.DataFrame, dict]:
    """The table usdm works on: complete rows of a table / the cells of a raster (sampled)."""
    info: dict = {}
    if isinstance(env, pd.DataFrame):
        table = env.dropna()
        names = [str(c) for c in table.columns]
        values = table.to_numpy(dtype=float)
        if size is not None and len(values) >= _ALL_CELLS_BELOW and len(values) > size:
            s = _seed(seed)
            rows = _core.RRandom(s).sample_int(len(values), int(size), False, None) - 1
            values = values[rows]
            info["seed"] = s
        return pd.DataFrame(values, columns=names), info
    if not isinstance(env, xr.DataArray) or env.ndim != 3:
        raise TypeError("predictors must be a raster stack (band, y, x) or a table (one column per predictor)")
    if area is not None:
        from ..io import clip

        env = clip(env, area, all_touched=True)  # as terra::mask(crop(x, v), v): touched cells
    names = [str(b) for b in env["band"].values] if "band" in env.coords else [f"band{i + 1}" for i in range(len(env))]
    data = np.asarray(env.values, dtype=float).reshape(env.shape[0], -1)
    nodata = env.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        data = np.where(data == nodata, np.nan, data)
    complete = ~np.isnan(data).any(axis=0)
    if size is None or data.shape[1] < _ALL_CELLS_BELOW:
        cells = np.flatnonzero(complete)
    else:
        s = _seed(seed)
        cells = _sample_cells(env, complete, int(size), _core.RRandom(s))
        info["seed"] = s
    info["cells"] = int(cells.size)
    return pd.DataFrame(data[:, cells].T, columns=names), info


# -------------------------------------------------------------------------------------------- VIF

def _vif_of(values: np.ndarray, which) -> np.ndarray:
    if values.shape[1] < 2:
        return np.ones(len(which))  # one variable: an intercept-only model, R^2 = 0
    return _core.sdm_vif(np.ascontiguousarray(values), [int(w) for w in which])


def vif(env, *, area=None, size: int | None = _SAMPLE_SIZE, seed: int | None = None) -> pd.DataFrame:
    """Variance inflation factor of each predictor: how well the others predict it.

    VIF = 1 / (1 - R²) of each variable regressed on all the others: 1 for an
    independent variable, above 5-10 for one the others nearly determine
    (its coefficient in a linear model becomes unstable). The same values as
    the R package usdm's ``vif()`` (R's ``lm`` and ``summary.lm``, to the last
    bit), on the same cells: a raster's cells with a value in every band
    (5000 of them, drawn as terra's ``spatSample`` with ``set.seed(seed)``,
    when it has 7000 or more; weighted by the cosine of the latitude for
    longitude / latitude grids), or the complete rows of a table.

    Parameters
    ----------
    env
        Raster stack (band, y, x) or a table with one column per predictor.
    area
        Polygons the raster is cut to first (the accessible area).
    size
        Cells sampled from rasters of 7000 cells or more (None: all).
    seed
        Seed of R's generator for the sample (default random).

    Returns
    -------
    pandas.DataFrame
        ``variable`` and ``vif``.

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> ls.sdm.vif(env, seed=1).round(1).head(3)    # bio7 = bio5 - bio6: exactly collinear
      variable      vif
    0     bio1      inf
    1     bio5   9952.0
    2     bio6  40802.8
    """
    table, _ = _values(env, size, seed, area)
    values = table.to_numpy(dtype=float)
    return pd.DataFrame({"variable": list(table.columns), "vif": _vif_of(values, range(values.shape[1]))})


def _cor(values: np.ndarray, method: str) -> np.ndarray:
    return _core.lsm_r_cor(np.ascontiguousarray(values), method)


def _vifstep(table: pd.DataFrame, th: float, keep: list[str]) -> list[str]:
    names = list(table.columns)
    values = table.to_numpy(dtype=float)
    excluded = []
    while len(names) >= 2:
        v = _vif_of(values, range(len(names)))
        cand = [i for i, nm in enumerate(names) if nm not in keep]
        if not cand:
            break
        vc = v[cand]
        if np.all(np.isnan(vc)):
            break
        j = cand[int(np.nanargmax(vc))]  # the first maximum
        if not v[j] >= th:
            break
        excluded.append(names.pop(j))
        values = np.delete(values, j, axis=1)
    return excluded


def _vifcor(table: pd.DataFrame, th: float, keep: list[str], method: str) -> list[str]:
    names = list(table.columns)
    values = table.to_numpy(dtype=float)
    excluded = []
    while len(names) >= 2:
        c = np.abs(_cor(values, method))
        n = len(names)
        # the pairs of the strict upper triangle in column-major order (r < c), largest first, stable
        pairs = [(r, cc) for cc in range(n) for r in range(cc)]
        vals = np.array([c[r, cc] for r, cc in pairs])
        order = sorted(range(len(pairs)), key=lambda k: (np.isnan(vals[k]), -vals[k] if not np.isnan(vals[k]) else 0))
        pick = None
        for k in order:
            r, cc = pairs[k]
            if names[r] not in keep or names[cc] not in keep:
                pick = (r, cc)
                break
        if pick is None:
            break
        r, cc = pick
        if not c[r, cc] >= th:
            break
        later, earlier = cc, r
        if names[later] in keep:
            ex = earlier
        elif names[earlier] in keep:
            ex = later
        else:
            v = _vif_of(values, [later, earlier])
            # which.max over (later, earlier): the first maximum, NaN ignored
            ex = earlier if (np.isnan(v[0]) and not np.isnan(v[1])) or v[1] > v[0] else later
        excluded.append(names.pop(ex))
        values = np.delete(values, ex, axis=1)
    return excluded


def select_predictors(env, *, vif: float | None = 10.0, cor: float | None = None, keep=None, area=None,
                      method: str = "pearson", size: int | None = _SAMPLE_SIZE, seed: int | None = None):
    """Drop predictors that the others already explain (collinearity).

    Climate layers are strongly correlated (the 19 bioclimatic variables
    carry far fewer than 19 independent signals); collinear predictors make
    a model's coefficients unstable and its projections to new conditions
    unreliable (Dormann et al. 2013, *Ecography* 36: 27-46). Two filters of
    the R package usdm, with the same results:

    - ``cor`` (usdm's ``vifcor``): while the most correlated pair has
      ``|r| >= cor``, drop the one of the two with the larger VIF;
    - ``vif`` (usdm's ``vifstep``): while the largest VIF is ``>= vif``, drop
      that variable.

    With both, the correlation filter runs first. Variables in ``keep`` are
    never dropped. Where usdm errs, the correct form: the loops stop when
    one variable (or only kept ones) remains instead of failing, names with
    spaces work, rasters of 7000-10 000 cells are sampled at random (terra
    took their first cells) and tables are sampled from 7000 rows on in both
    filters (usdm's ``vifstep`` already from 6000).

    Parameters
    ----------
    env
        Raster stack (band, y, x) or a table with one column per predictor.
    vif
        VIF threshold (usdm's default 10; 5 is stricter); None: no VIF filter.
    cor
        Correlation threshold (e.g. 0.7); None: no correlation filter.
    keep
        Names of predictors to keep whatever their collinearity.
    area
        Polygons the raster is cut to first: collinearity in the accessible
        area, not in the whole raster.
    method : {"pearson", "spearman", "kendall"}
        Correlation coefficient of the ``cor`` filter.
    size
        Cells sampled from rasters of 7000 cells or more (usdm's 5000; None:
        all cells).
    seed
        Seed of R's generator for the sample (default random).

    Returns
    -------
    xarray.DataArray or pandas.DataFrame
        The predictors kept (the same type as ``env``), with
        ``attrs["selection"]``: the variables excluded in order, the VIF of
        the kept ones and the seed.

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> kept = ls.sdm.select_predictors(env, vif=10, seed=1)
    >>> [str(b) for b in kept.band.values]
    ['bio5', 'bio7', 'bio8', 'bio16', 'bio17']
    >>> kept.attrs["selection"]["excluded"]
    ['bio1', 'bio6', 'bio9', 'bio12']
    """
    if vif is None and cor is None:
        raise ValueError("give a vif and / or a cor threshold")
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError("method must be 'pearson', 'spearman' or 'kendall'")
    keep = [str(k) for k in ([keep] if isinstance(keep, str) else (keep or []))]
    table, info = _values(env, size, seed, area)
    if table.shape[1] < 2:
        raise ValueError("need at least two predictors")
    unknown = set(keep) - set(table.columns)
    if unknown:
        raise ValueError(f"keep names unknown predictors: {sorted(unknown)}")
    excluded: list[str] = []
    if cor is not None:
        excluded += _vifcor(table, float(cor), keep, method)
    if vif is not None:
        excluded += _vifstep(table.drop(columns=excluded), float(vif), keep)
    kept_names = [c for c in table.columns if c not in excluded]
    final = table[kept_names].to_numpy(dtype=float)
    selection = {"excluded": excluded, "kept": kept_names,
                 "vif": dict(zip(kept_names, _vif_of(final, range(len(kept_names))).tolist())),
                 "vif_threshold": vif, "cor_threshold": cor, "method": method, **info}
    if isinstance(env, pd.DataFrame):
        out = env[[c for c in env.columns if str(c) in kept_names]]
    else:
        names = [str(b) for b in env["band"].values]
        out = env.isel(band=[names.index(k) for k in kept_names])
    out = out.copy()
    out.attrs["selection"] = selection
    return out
