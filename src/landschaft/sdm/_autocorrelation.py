# SPDX-License-Identifier: GPL-3.0-or-later
"""The range of spatial autocorrelation of predictors: how large spatial blocks should be."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from ._thin import _seed

__all__ = ["autocorrelation_range"]

_BOUNDS = np.array([2, 4, 6, 9, 12, 15, 25, 35, 50, 65, 80, 100], dtype=float)
_KAPPAS = [0.05, *np.round(np.arange(0.2, 2.0001, 0.1), 10).tolist(), 5.0, 10.0]


# --------------------------------------------------------------------------------------- distances

def _gcdist(lon1, lat1, lon2, lat2):
    """gstat's great-circle distance (km): Andoyer-Lambert on WGS 84 (sp's spDists)."""
    rad = math.pi / 180
    a, f = 6378.137, 1 / 298.257223563
    F = (lat1 * rad + lat2 * rad) / 2
    G = (lat1 * rad - lat2 * rad) / 2
    L = (lon1 * rad - lon2 * rad) / 2
    sg2, cg2 = np.sin(G) ** 2, np.cos(G) ** 2
    sf2, cf2 = np.sin(F) ** 2, np.cos(F) ** 2
    sl2, cl2 = np.sin(L) ** 2, np.cos(L) ** 2
    S = sg2 * cl2 + cf2 * sl2
    C = cg2 * cl2 + sf2 * sl2
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.arctan(np.sqrt(S / C))
        R = np.sqrt(S * C) / w
        D = 2 * w * a
        H1 = (3 * R - 1) / (2 * C)
        H2 = (3 * R + 1) / (2 * S)
        d = D * (1 + f * H1 * sf2 * cg2 - f * H2 * cf2 * sg2)
    same = (lon1 == lon2) & (lat1 == lat2)
    return np.where(same, 0.0, d)


def _sample_variogram(x, y, z, bounds, lonlat):
    """gstat's sample variogram: per distance class (<= each bound), np, mean distance, semivariance (C++)."""
    np_, dist, gamma = _core.sdm_sample_variogram(np.ascontiguousarray(x, dtype=float),
                                                  np.ascontiguousarray(y, dtype=float),
                                                  np.ascontiguousarray(z, dtype=float),
                                                  [float(b) for b in bounds], bool(lonlat))
    keep = np_ > 0
    return np_[keep], dist[keep], gamma[keep]


# ------------------------------------------------------------------------------------------- models

def _unit(model, h, r, kappa):
    h = np.asarray(h, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if model == "Sph":
            hr = h / r
            out = np.where(h >= r, 1.0, hr * (1.5 - 0.5 * hr * hr))
        elif model == "Exp":
            out = 1.0 - np.exp(-h / r)
        elif model == "Gau":
            hr = h / r
            out = 1.0 - np.exp(-(hr * hr))
        else:  # Ste: Matern, M. Stein's parameterisation
            from scipy.special import gamma, kv

            hh = h / r
            arg = 2.0 * math.sqrt(kappa) * hh
            bes = kv(kappa, arg)
            mult = 2.0 ** (1.0 - kappa) / gamma(kappa) * arg ** kappa
            out = 1.0 - bes * mult
            out = np.where(~np.isfinite(bes), 0.0, np.where(bes == 0, 1.0, np.where(~np.isfinite(mult), 1.0, out)))
    return np.where(h == 0, 0.0, out)


def _d_range(model, h, r, kappa, sill):
    """d semivariance / d range (gstat's analytic derivatives; numerical for Ste)."""
    if model == "Sph":
        hr2 = h / (r * r)
        return sill * np.where(h > r, 0.0, 1.5 * hr2 * (-1.0 + h * hr2))
    if model == "Exp":
        hr = -h / r
        return sill * (hr / r) * np.exp(hr)
    if model == "Gau":
        hr = h / r
        return sill * (-hr / r) * np.exp(-(hr * hr))
    rr = max(1e-20, r)
    lo = _unit(model, h, rr * 1.001, kappa)
    hi = _unit(model, h, rr * 0.999, kappa)
    return sill * (lo - hi) / (2.0 * rr * 0.001)


def _fit(model, dist, gamma, nh, nugget, psill, rng_, kappa):
    """gstat's fit.variogram (fit.method 7: weights N_j / h_j^2): Gauss-Newton with bounded steps; a
    negative nugget or partial sill is fixed at zero and the fit repeated from the initial range (C++)."""
    out = _core.sdm_fit_variogram(model, float(kappa), np.ascontiguousarray(dist, dtype=float),
                                  np.ascontiguousarray(gamma, dtype=float), np.ascontiguousarray(nh, dtype=float),
                                  float(nugget), float(psill), float(rng_))
    if out is None:
        return None
    nug, ps, r, sse = out
    return np.array([nug, ps, r]), sse


def _autofit(x, y, z, lonlat):
    """automap's autofitVariogram (Sph, Exp, Gau and Ste over its kappas): the best fit."""
    if lonlat:
        diagonal = float(_gcdist(x.min(), y.min(), x.max(), y.max()))
    else:
        diagonal = float(np.hypot(x.max() - x.min(), y.max() - y.min()))
    bounds = _BOUNDS * diagonal * 0.35 / 100
    nh, dist, gamma = _sample_variogram(x, y, z, bounds, lonlat)
    while (nh < 5).any() and bounds.size > 1:  # merge small bins: drop the first boundary
        bounds = bounds[1:]
        nh, dist, gamma = _sample_variogram(x, y, z, bounds, lonlat)
    nugget0 = float(gamma.min())
    range0 = 0.1 * diagonal
    sill0 = float(np.mean([gamma.max(), np.median(gamma)]))
    best = None
    for model in ("Sph", "Exp", "Gau", "Ste"):
        for kappa in (_KAPPAS if model == "Ste" else [0.0]):
            out = _fit(model, dist, gamma, nh, nugget0, sill0 - nugget0, range0, kappa)
            if out is None:
                continue
            (nug, ps, r), sse = out
            if ps < 0 or r < 0 or not np.isfinite(sse):
                continue
            if best is None or sse < best["sserr"]:
                best = {"model": model, "kappa": kappa, "nugget": nug, "sill": ps, "range": r, "sserr": sse}
    if best is None:
        raise ValueError("no variogram model could be fitted")
    best["variogram"] = pd.DataFrame({"np": nh, "dist": dist, "gamma": gamma})
    return best


def _sample_points(layer: xr.DataArray, n: int, rng):
    """terra's spatSample(layer, n, "random", as.points = TRUE, na.rm = TRUE) for one band."""
    from ._background import _cell_centres, _could_be_lonlat
    from ._predictors import _row_weights

    v = np.asarray(layer.values, dtype=float).ravel()
    nodata = layer.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        v = np.where(v == nodata, np.nan, v)
    nona = np.flatnonzero(~np.isnan(v))
    ncell = v.size
    if n >= ncell:
        cells = nona[:n]
    else:
        take = min(5 * n, nona.size)  # the whole draw advances R's generator
        prob = None
        if _could_be_lonlat(layer):
            nx = layer.shape[-1]
            prob = _row_weights(layer.expand_dims(band=[0]) if layer.ndim == 2 else layer)[nona // nx].tolist()
        i = rng.sample_int(nona.size, take, False, prob)
        cells = nona[i - 1][:n]
    x, y = _cell_centres(layer, cells)
    return x, y, v[cells]


def autocorrelation_range(env, *, n: int = 5000, seed: int | None = None, deg_to_metre: float = 111_325.0):
    """The range of spatial autocorrelation of each predictor (and their median), in metres.

    A variogram is fitted to a random sample of each band's cells (the
    models tried by automap: spherical, exponential, Gaussian and Matern
    with Stein's parameterisation over a range of smoothness, the one with
    the smallest weighted squared error kept); its range parameter is how
    far values stay alike. Spatial blocks for cross-validation should be
    about as wide as the median range (Roberts et al. 2017; Valavi et al.
    2019). As ``blockCV::cv_spatial_autocor``: the same sample (R's
    generator, ``set.seed(seed)``) and variogram classes, and gstat's
    Gauss-Newton fit step by step: the ranges agree to about 1e-6 (the
    Matern model's derivative is numerical). Climate layers on a continent
    often vary so smoothly that the fitted range exceeds the study area:
    use the result as a guide, not blindly.

    Parameters
    ----------
    env
        Predictors (raster stack).
    n
        Cells sampled per band.
    seed
        Seed of R's generator (default random).
    deg_to_metre
        Metres per degree, for the block size on longitude / latitude grids.

    Returns
    -------
    dict
        ``range`` (median, metres), ``block_size`` (the range in the units of
        the grid: degrees on longitude / latitude) and ``layers`` (a table:
        layer, range, sill, nugget, model, kappa).

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> r = ls.sdm.autocorrelation_range(env.sel(band=["bio1", "bio12"]), n=1000, seed=1)
    >>> round(r["range"] / 1000)      # km: temperature varies smoothly across the continent
    226192
    """
    from ._background import _could_be_lonlat

    if env.ndim == 2:
        env = env.expand_dims(band=[env.name or "layer"])
    s = _seed(seed)
    rng = _core.RRandom(s)
    lonlat = _could_be_lonlat(env)
    rows = []
    for b in env.band.values:
        layer = env.sel(band=b)
        nn = n
        if layer.size < 5 * n:
            nn = min(n, int(np.isfinite(np.asarray(layer.values, dtype=float)).sum()))
        x, y, z = _sample_points(layer, nn, rng)
        fit = _autofit(x, y, z, lonlat)
        rng_m = fit["range"] * 1000 if lonlat else fit["range"]
        rows.append({"layer": str(b), "range": rng_m, "sill": fit["sill"], "nugget": fit["nugget"],
                     "model": fit["model"], "kappa": fit["kappa"]})
    layers = pd.DataFrame(rows)
    med = float(np.median(layers["range"]))
    size = med / deg_to_metre if lonlat else med
    return {"range": med, "block_size": size, "layers": layers, "seed": s}
