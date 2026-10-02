# SPDX-License-Identifier: GPL-3.0-or-later
"""Where projections extrapolate: MESS, MOP and ExDet."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core

__all__ = ["exdet", "mess", "mop"]


# ------------------------------------------------------------------------------------------ inputs

def _reference(reference, names=None) -> pd.DataFrame:
    """The reference conditions as a table (rows with a missing value dropped)."""
    from ._data import SDMData

    if isinstance(reference, SDMData):
        table = reference.X
    elif isinstance(reference, xr.DataArray):
        table = _raster_table(reference)[0]
    elif isinstance(reference, pd.DataFrame):
        table = reference
    else:
        raise TypeError("reference must be SDMData, a DataFrame or a raster stack")
    if names is not None:
        missing = [n for n in names if n not in table.columns]
        if missing:
            raise ValueError(f"the reference has no predictor {missing}")
        table = table[names]
    return table.dropna()


def _raster_table(env: xr.DataArray):
    """A raster stack as a (cells x bands) table, NoData as NaN."""
    if env.ndim == 2:
        env = env.expand_dims(band=[env.name or "layer"])
    data = np.asarray(env.values, dtype=float).reshape(env.shape[0], -1)
    nodata = env.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        data = np.where(data == nodata, np.nan, data)
    names = [str(b) for b in env["band"].values]
    return pd.DataFrame(data.T, columns=names), env


def _target(env):
    """The conditions to test: a raster stack or a table; returns (table, template or None)."""
    if isinstance(env, xr.DataArray):
        table, env = _raster_table(env)
        return table, env
    if isinstance(env, pd.DataFrame):
        return env, None
    raise TypeError("env must be a raster stack or a DataFrame")


def _output(values: dict, template):
    """Bands on the grid of ``template`` (or a DataFrame for a table)."""
    if template is None:
        return pd.DataFrame(values)
    ny, nx = template.shape[-2:]
    data = np.stack([np.asarray(v, dtype=float).reshape(ny, nx) for v in values.values()])
    out = xr.DataArray(data, dims=("band", "y", "x"),
                       coords={"band": list(values), "y": template["y"], "x": template["x"]})
    if template.rio.crs is not None:
        out = out.rio.write_crs(template.rio.crs)
    return out.rio.write_transform(template.rio.transform()).rio.write_nodata(np.nan)


# ------------------------------------------------------------------------------------------- MESS

def _messi(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """predicts' .messi: the similarity of the values p to the reference sample v of one variable."""
    v = np.sort(v[~np.isnan(v)])
    n = v.size
    k = np.searchsorted(v, p, side="right").astype(float)  # findInterval(p, v): number of v <= p
    f = (100 * k) / n
    f[np.isnan(p)] = np.nan
    minv, maxv = v[0], v[-1]
    res = 2 * f
    ff = np.where(np.isnan(f), -99.0, f)
    i = (ff > 50) & (ff < 100)
    res[i] = 200 - res[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        i = ff == 0
        res[i] = (100 * (p[i] - minv)) / (maxv - minv)
        i = ff == 100
        res[i] = (100 * (maxv - p[i])) / (maxv - minv)
    return res


def mess(reference, env, *, full: bool = False):
    """Multivariate environmental similarity surface: how far conditions lie outside the reference.

    For each variable, the similarity of a cell's value to the distribution
    of the reference values (positive inside the reference range, 100 at its
    median; negative outside it, in percent of the range); the MESS is the
    smallest over the variables, so negative values mark cells where at
    least one variable is outside anything the model saw (Elith et al. 2010,
    *Methods in Ecology and Evolution* 1: 330-342). ``predicts::mess``, the
    same values.

    Parameters
    ----------
    reference
        The conditions the model was fitted to: :class:`SDMData` (its
        predictor values), a table, or a raster stack (its cells).
    env
        The conditions to project to: raster stack (e.g. future climate) or
        a table, with the reference's variables.
    full
        Also return the similarity of every variable (bands ``<name>_mess``).

    Returns
    -------
    xarray.DataArray or pandas.DataFrame
        Band ``mess`` (and the per-variable bands), NaN where every variable
        is missing.

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> data = ls.sdm.dataset(occ, ls.sdm.background(env, 2000, occ=occ, seed=1), env)
    >>> m = ls.sdm.mess(data, env + 30)      # 3 degrees warmer (and 30 mm wetter)
    >>> bool((m.sel(band="mess") < 0).sum() > 0)
    True
    """
    table, template = _target(env)
    ref = _reference(reference, list(table.columns))
    if len(ref) < 2:
        raise ValueError("the reference needs at least two complete rows")
    parts = {}
    for name in table.columns:
        parts[f"{name}_mess"] = _messi(table[name].to_numpy(dtype=float), ref[name].to_numpy(dtype=float))
    stack = np.column_stack(list(parts.values()))
    with np.errstate(all="ignore"):
        allnan = np.isnan(stack).all(axis=1)
        m = np.where(allnan, np.nan, np.nanmin(np.where(np.isnan(stack), np.inf, stack), axis=1))
    m[~np.isfinite(m)] = np.nan
    values = {**parts, "mess": m} if full else {"mess": m}
    return _output(values, template)


# -------------------------------------------------------------------------------------------- MOP

def _r_colmeans(x: np.ndarray) -> np.ndarray:
    """R's colMeans: one 80-bit pass per column, divided by n."""
    return _core.sdm_r_colmeans(np.ascontiguousarray(x, dtype=float))


def mop(reference, env, *, type: str = "basic", distance: str | None = "euclidean", where: str = "in_range",
        percentage: float = 1.0, scale: bool = False, center: bool = False, rescale: bool = False):
    """Mobility-oriented parity: where new conditions are outside or far from the reference ones.

    The variables outside the reference range (``simple``: how many;
    ``basic``: 1 where any) and, for each cell, the mean distance to its
    nearest reference conditions (the closest ``percentage`` percent of
    them): large distances mean dissimilar conditions even within the ranges
    (Owens et al. 2013, *Ecological Modelling* 263: 10-18). The R package
    mop's ``mop()``, the same values; the Mahalanobis distance uses the
    covariance of the reference conditions (mop takes that of the cells
    tested, so a cell's distance depended on which other cells were passed).

    Parameters
    ----------
    reference
        Conditions the model was fitted to (:class:`SDMData`, table or
        raster stack).
    env
        Conditions to test (raster stack or table).
    type : {"basic", "simple", "detailed"}
        ``"simple"`` adds the number of variables outside the range;
        ``"detailed"`` also which variables are below / above it
        (``low_combined`` / ``high_combined``: sums of 2^j for variable j,
        1-based).
    distance : {"euclidean", "mahalanobis", None}
        Distance to the reference (None: none).
    where : {"in_range", "out_range", "all"}
        Cells whose distance is computed.
    percentage
        Share of the nearest reference conditions averaged, in percent.
    scale, center
        Standardise the variables (over reference and tested conditions
        together, as mop) before the Euclidean distance.
    rescale
        Rescale the distances to 0-1.

    Returns
    -------
    xarray.DataArray or pandas.DataFrame
        Bands ``mop`` (distance), ``basic`` (and ``simple``, ``low_combined``,
        ``high_combined`` with the other types).

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> data = ls.sdm.dataset(occ, ls.sdm.background(env, 2000, occ=occ, seed=1), env)
    >>> out = ls.sdm.mop(data, env + 30)
    >>> [str(b) for b in out.band.values]
    ['mop', 'basic']
    """
    if type not in ("basic", "simple", "detailed"):
        raise ValueError("type must be 'basic', 'simple' or 'detailed'")
    if distance not in ("euclidean", "mahalanobis", None):
        raise ValueError("distance must be 'euclidean', 'mahalanobis' or None")
    if where not in ("in_range", "out_range", "all"):
        raise ValueError("where must be 'in_range', 'out_range' or 'all'")
    if not 0 < percentage <= 100:
        raise ValueError("percentage must be in (0, 100]")
    table, template = _target(env)
    names = list(table.columns)
    m = _reference(reference, names).to_numpy(dtype=float)
    gall = table.to_numpy(dtype=float)
    ok = ~np.isnan(gall).any(axis=1)  # a cell missing one variable is missing in all (mop's fix_NA)
    g = gall[ok]
    if distance == "euclidean" and (scale or center):
        both = np.vstack([m, g])
        if center:
            both = both - _r_colmeans(both)
        if scale:
            sc = np.array([np.sqrt(_core.nlm_r_sum(np.ascontiguousarray(c * c)) / max(1, c.size - 1))
                           for c in both.T])
            both = both / sc
        m, g = both[: len(m)], both[len(m):]
    lo, hi = m.min(axis=0), m.max(axis=0)
    below, above = g < lo, g > hi
    simple = (below | above).sum(axis=1).astype(float)
    basic = np.where(simple > 0, 1.0, np.nan)
    simple = np.where(simple > 0, simple, np.nan)
    n = gall.shape[0]

    def full(v):
        out = np.full(n, np.nan)
        out[ok] = v
        return out

    values = {}
    if distance is not None:
        rows = {"in_range": np.isnan(basic), "out_range": ~np.isnan(basic), "all": np.ones(len(g), bool)}[where]
        per_out = round(len(m) * (percentage / 100))  # half to even, as R; 0 = the nearest one
        d = np.full(len(g), np.nan)
        if rows.any():
            if distance == "euclidean":
                d[rows] = _core.sdm_knn_mean_distance(np.ascontiguousarray(m), np.ascontiguousarray(g[rows]),
                                                      max(per_out, 1), False)
            else:
                cov = np.cov(m, rowvar=False)
                w = np.linalg.cholesky(_inverse(cov)).T  # S^-1 = W' W
                d[rows] = _core.sdm_knn_mean_distance(np.ascontiguousarray(m @ w.T),
                                                      np.ascontiguousarray(g[rows] @ w.T), max(per_out, 1), True)
            if rescale:
                v = d[rows]
                d[rows] = (v - v.min()) / (v.max() - v.min())
        values["mop"] = full(d)
    values["basic"] = full(basic)
    if type in ("simple", "detailed"):
        values["simple"] = full(simple)
    if type == "detailed":
        codes = 2.0 ** np.arange(1, len(names) + 1)
        low = (below * codes).sum(axis=1)
        high = (above * codes).sum(axis=1)
        values["low_combined"] = full(np.where(low > 0, low, np.nan))
        values["high_combined"] = full(np.where(high > 0, high, np.nan))
    return _output(values, template)


# ------------------------------------------------------------------------------------------ ExDet

def _inverse(cov: np.ndarray) -> np.ndarray:
    """The inverse of a covariance matrix; an error when it is singular (collinear predictors), as R's solve()."""
    cov = np.atleast_2d(cov)
    if 1.0 / np.linalg.cond(cov, 1) < np.finfo(float).eps:
        raise ValueError("the covariance of the predictors is singular: some are collinear "
                         "(drop them first, e.g. with ls.sdm.select_predictors)")
    return np.linalg.inv(cov)


def _mahalanobis2(x: np.ndarray, center: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = x - center
    return np.einsum("ij,jk,ik->i", d, _inverse(cov), d)


def exdet(reference, env):
    """Extrapolation detection: univariate (NT1) and combinatorial (NT2) novelty.

    Mesgaran et al. (2014, *Diversity and Distributions* 20: 1147-1159):
    ``NT1`` < 0 where some variable is outside its reference range (more
    negative, farther out); inside every range, ``NT2`` is the Mahalanobis
    distance to the reference centre relative to the largest one among the
    reference points, > 1 for a combination of values the reference never
    had. ``mic`` names the variable most responsible (1-based; 0 for analogue
    conditions). The ``ExDet`` of the R package dsmextra, except that its
    combinatorial ``mic`` pointed at the wrong variable (it numbered the
    variables backwards).

    Returns
    -------
    xarray.DataArray or pandas.DataFrame
        Bands ``exdet`` and ``mic``.

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> data = ls.sdm.dataset(occ, ls.sdm.background(env, 2000, occ=occ, seed=1), env)
    >>> out = ls.sdm.exdet(data, env.sel(band=["bio1", "bio12", "bio16"]))
    >>> [str(b) for b in out.band.values]
    ['exdet', 'mic']
    """
    table, template = _target(env)
    names = list(table.columns)
    ref = _reference(reference, names).to_numpy(dtype=float)
    tg_all = table.to_numpy(dtype=float)
    ok = ~np.isnan(tg_all).any(axis=1)
    tg = tg_all[ok]
    a, b = ref.min(axis=0), ref.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        nt1_parts = np.minimum(np.minimum(tg - a, b - tg), 0.0) / (b - a)
    nt1 = np.array([_core.nlm_r_sum(np.ascontiguousarray(r)) for r in nt1_parts])
    mic_uni = np.where(nt1 != 0, np.argmin(nt1_parts, axis=1) + 1.0, np.nan)
    u = nt1 == 0
    centre = np.array([_core.sdm_r_mean(np.ascontiguousarray(c)) for c in ref.T])
    cov = np.cov(ref, rowvar=False)
    d2_ref = _mahalanobis2(ref, centre, cov)
    d2max = d2_ref[np.isfinite(d2_ref)].max()
    ex = nt1.copy()
    mic_comb = np.full(len(tg), np.nan)
    if u.any():
        d2 = _mahalanobis2(tg[u], centre, cov)
        ex[u] = d2 / d2max
        p = len(names)
        if p > 1 and u.sum() >= 2:
            pct = np.empty((int(u.sum()), p))
            for k in range(p):  # drop variable k
                keep = [j for j in range(p) if j != k]
                d2k = _mahalanobis2(tg[u][:, keep], centre[keep], np.cov(ref[:, keep], rowvar=False))
                pct[:, k] = 100 * (d2 - d2k) / d2k
            comb = np.where(ex[u] > 1, np.argmax(pct, axis=1) + 1.0, np.nan)
            mic_comb[u] = comb
    mic = np.nan_to_num(mic_uni) + np.nan_to_num(mic_comb)
    out_ex = np.full(len(tg_all), np.nan)
    out_ex[ok] = ex
    out_mic = np.full(len(tg_all), np.nan)
    out_mic[ok] = mic
    return _output({"exdet": out_ex, "mic": out_mic}, template)
