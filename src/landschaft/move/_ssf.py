# SPDX-License-Identifier: GPL-3.0-or-later
"""Step selection: distributions of steps, random steps and covariates at steps (amt's fit_distr, random_steps & co.)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import warnings

import numpy as np
import pandas as pd

from .. import _core
from ._steps import BURST
from ._tracks import ATTRS

_DEFAULTS = {"exp": {"rate": 1.0}, "gamma": {"shape": 1.0, "scale": 1.0}, "hnorm": {"sd": 1.0},
             "lnorm": {"meanlog": 0.0, "sdlog": 1.0}, "unif": {"min": -math.pi, "max": math.pi},
             "vonmises": {"kappa": 1.0, "mu": 0.0}}
_TA = ("vonmises", "unif")


@dataclass
class Distribution:
    """A distribution of step lengths or turn angles (amt's ``amt_distr``).

    ``name``: ``"gamma"``, ``"exp"``, ``"lnorm"``, ``"hnorm"``, ``"unif"`` or
    ``"vonmises"``; ``params``: its parameters by name; ``vcov``: the
    variance-covariance matrix of the fitted parameters, when known.
    """
    name: str
    params: dict
    vcov: pd.DataFrame | None = field(default=None, repr=False)

    def _draw_params(self) -> list[float]:
        p = self.params
        if self.name == "gamma":
            scale = p["scale"] if "scale" in p else 1 / p["rate"]
            return [float(p["shape"]), float(scale)]
        if self.name == "vonmises":
            return [float(p["kappa"]), float(p.get("mu", 0.0))]
        return [float(p[k]) for k in _DEFAULTS[self.name]]


def make_distribution(name: str, **params) -> Distribution:
    """A distribution of step lengths or turn angles (amt's ``make_distribution``, ``make_*_distr``).

    Parameters
    ----------
    name
        ``"gamma"`` (``shape``, and ``scale`` or ``rate``), ``"exp"`` (``rate``),
        ``"lnorm"`` (``meanlog``, ``sdlog``), ``"hnorm"`` (``sd``) for step lengths;
        ``"vonmises"`` (``kappa``, ``mu`` 0) or ``"unif"`` (``min``, ``max``) for turn
        angles.
    **params
        The parameters; those not given take amt's defaults (gamma 1, 1; exp 1;
        lnorm 0, 1; hnorm 1; unif -pi, pi; von Mises 1, 0).

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.move.make_distribution("gamma", shape=2, scale=150)
    Distribution(name='gamma', params={'shape': 2, 'scale': 150})
    >>> ls.move.make_distribution("vonmises", kappa=0.5)
    Distribution(name='vonmises', params={'kappa': 0.5, 'mu': 0.0})
    """
    if name not in _DEFAULTS:
        raise ValueError(f"{name!r} is not implemented: one of {', '.join(_DEFAULTS)}")
    if name == "gamma" and "rate" in params:
        if "scale" in params:
            raise ValueError("gamma: scale or rate, not both")
        known = {"shape", "rate"}
        p = {"shape": params.get("shape", 1.0), "rate": params["rate"]}
    else:
        known = set(_DEFAULTS[name])
        p = {k: params.get(k, v) for k, v in _DEFAULTS[name].items()}
    unknown = set(params) - known
    if unknown:
        raise ValueError(f"parameters of {name}: {', '.join(sorted(known))}; not {', '.join(sorted(unknown))}")
    for k in ("rate", "sd", "sdlog", "kappa"):
        if k in p and not p[k] >= 0:
            raise ValueError(f"{k}: 0 or more")
    return Distribution(name, p)


def fit_distr(x, name: str, *, na_rm: bool = True) -> Distribution:
    """A distribution fitted to step lengths or turn angles (amt's ``fit_distr``, the same numbers).

    * ``"gamma"``: maximum likelihood by R's Nelder-Mead from the closed-form
      estimates (``fitdistrplus::fitdist``); steps of length 0 take the
      smallest other length. ``vcov``: of ``scale`` and ``shape``.
    * ``"exp"``: the maximum likelihood rate, ``1 / mean(x)``. amt fits it by
      BFGS from that value, which fails for rates below 0.001 (mean steps over
      1000 m) and otherwise only adds the optimizer's noise.
    * ``"lnorm"``: maximum likelihood by Nelder-Mead (``fitdistrplus``).
    * ``"hnorm"``: ``sqrt(mean(x^2))``.
    * ``"unif"``: the smallest and the largest value.
    * ``"vonmises"``: ``kappa`` by maximum likelihood (``circular::mle.vonmises``),
      ``mu`` 0. ``vcov``: the variance of ``kappa`` (amt stores its standard
      error there).

    Examples
    --------
    >>> import landschaft as ls
    >>> sl = ls.move.random_numbers(ls.move.make_distribution("gamma", shape=2, scale=100), 500, seed=1)
    >>> d = ls.move.fit_distr(sl, "gamma")
    >>> round(d.params["shape"], 3), round(d.params["scale"], 2)
    (1.896, 102.56)
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    if na_rm:
        v = v[~np.isnan(v)]
    v = np.ascontiguousarray(v)
    if name == "gamma":
        f = _core.move_fit_gamma(v)
        vcov = None
        if len(f["vcov"]) == 4:
            vcov = pd.DataFrame(np.asarray(f["vcov"]).reshape(2, 2), index=["scale", "shape"],
                                columns=["scale", "shape"])
        return Distribution("gamma", {"shape": f["shape"], "scale": f["scale"]}, vcov)
    if name == "exp":
        return Distribution("exp", {"rate": _core.move_fit_exp(v)})
    if name == "lnorm":
        meanlog, sdlog = _core.move_fit_lnorm(v)
        return Distribution("lnorm", {"meanlog": meanlog, "sdlog": sdlog})
    if name == "hnorm":
        if not len(v):
            raise ValueError("no values to fit")
        return Distribution("hnorm", {"sd": math.sqrt(1 / len(v) * _core.move_r_sum(v * v))})
    if name == "unif":
        if not len(v):
            raise ValueError("no values to fit")
        return Distribution("unif", {"min": float(v.min()), "max": float(v.max())})
    if name == "vonmises":
        f = _core.move_fit_vonmises(v)
        return Distribution("vonmises", {"kappa": f["kappa"], "mu": 0.0},
                            pd.DataFrame([[f["se_kappa"] ** 2]], index=["kappa"], columns=["kappa"]))
    raise ValueError(f"{name!r} is not supported: one of {', '.join(_DEFAULTS)}")


def _seed(seed):
    return int(np.random.default_rng().integers(1, 2**31 - 1)) if seed is None else int(seed)


def random_numbers(distr: Distribution, n: int = 100, *, seed: int | None = None) -> np.ndarray:
    """Draws of a distribution of steps (amt's ``random_numbers``).

    With a seed, R's generator after ``set.seed(seed)``: amt's numbers. Turn
    angles of a von Mises distribution are wrapped to (-pi, pi].

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.move.random_numbers(ls.move.make_distribution("exp", rate=0.01), 4, seed=1).round(2)
    array([ 75.52, 118.16,  14.57,  13.98])
    """
    return _core.move_random_numbers(_seed(seed), distr.name, distr._draw_params(), int(n))


def _fit_draws(sl: np.ndarray) -> list[tuple[int, int]]:
    """The sample.int calls of fitting a gamma distribution to ``sl`` with fitdistrplus (keepdata = FALSE).

    fitdist keeps ``min(100, n) - 2`` values at random besides the smallest and
    the largest: ``sample((1:n)[-c(imin, imax)], size)``, which draws from R's
    generator; with one value left, ``sample(k, size)`` draws from ``1:k``.
    """
    v = sl[~np.isnan(sl)]
    if (v == 0).any() and (v > 0).any():
        v = np.where(v == 0, v[v > 0].min(), v)
    n = len(v)
    keep = min(100, n) - 2
    if keep <= 0:
        return []
    imin, imax = int(np.argmin(v)), int(np.argmax(v))
    rest = [i + 1 for i in range(n) if i not in (imin, imax)]
    return [(rest[0] if len(rest) == 1 else len(rest), keep)]


def random_steps(steps, n_control: int = 10, *, sl_distr: Distribution | None = None,
                 ta_distr: Distribution | None = None, n_draws: int = 100_000, seed: int | None = None):
    """Each observed step with ``n_control`` random steps from its start (amt's ``random_steps``).

    The data of a step selection function: for each step but the first of a
    track (or of a burst), the observed step (``case_`` True) and
    ``n_control`` steps from the same start (``case_`` False) whose lengths and
    turn angles are drawn from ``sl_distr`` and ``ta_distr``, turned from the
    direction of the previous step; ``step_id_`` names the stratum. As amt,
    ``n_draws`` numbers are drawn from each distribution once per track and
    each random step takes one of them at random.

    Bursts (:func:`resample`) of fewer than 3 steps are left out. With a seed,
    R's generator after ``set.seed(seed)``, the tracks in turn: amt's random
    steps (``set.seed(seed)``, then ``random_steps`` of each track's steps).

    Parameters
    ----------
    steps
        The :func:`steps` of tracks, in a projected CRS.
    n_control
        Random steps per observed step.
    sl_distr, ta_distr
        The distributions of the step lengths and turn angles (default: a gamma
        and a von Mises distribution fitted to each track's steps).
    n_draws
        Numbers drawn from each distribution per track.
    seed
        The seed of R's generator.

    Returns
    -------
    pandas.DataFrame
        The columns of ``steps`` but ``direction_p``, with ``case_`` and
        ``step_id_``; the distributions of each track in
        ``attrs["landschaft.move"]["sl_distr"]`` and ``["ta_distr"]``.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(30), tracks=1, seed=1)
    >>> rs = ls.move.random_steps(ls.move.steps(tr), n_control=5, seed=2)
    >>> rs.shape, int(rs["case_"].sum())
    ((168, 12), 28)
    >>> rs[["sl_", "ta_", "case_", "step_id_"]].head(3).round(3)
         sl_    ta_  case_  step_id_
    0  1.371 -1.117   True         3
    1  1.096 -1.138  False         3
    2  0.933  1.012  False         3
    """
    info = steps.attrs.get(ATTRS, {})
    k = info.get("track")
    if k is None or k not in steps.columns or "x1_" not in steps.columns:
        raise TypeError("steps: the steps of ls.move.steps")
    if int(n_control) < 1 or int(n_draws) < 1:
        raise ValueError("n_control and n_draws: 1 or more")
    if info.get("lonlat"):
        raise ValueError("random steps are drawn in a projected CRS: project the tracks first "
                         "(tracks.to_crs(ls.move.aeqd_crs(tracks)))")
    bursts = BURST in steps.columns
    codes, _ = pd.factorize(steps[k], sort=False)
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True]) if len(codes) else np.array([0])
    tracks, calls, distrs, short = [], [], {}, 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        one = steps.iloc[a:b]
        tid = one[k].iloc[0]
        if bursts:
            sizes = one.groupby(BURST, sort=True)[BURST].size()
            kept = sizes.index[sizes.to_numpy() >= 3]
            short += len(sizes) - len(kept)
            if not len(kept):
                raise ValueError(f"track {tid!r}: no burst with 3 steps or more")
            parts = [one[one[BURST].to_numpy() == v] for v in kept]
            # start_ids: 1 + the steps of the bursts before
            starts = np.r_[1, 1 + np.cumsum([len(p) for p in parts])[:-1]]
        else:
            if len(one) < 2:
                raise ValueError(f"track {tid!r}: random steps need 2 steps or more")
            parts, starts = [one], np.array([1])
        sl = sl_distr if sl_distr is not None else fit_distr(one["sl_"], "gamma")
        ta = ta_distr if ta_distr is not None else fit_distr(one["ta_"], "vonmises")
        distrs[tid] = (sl, ta)
        tracks.append((parts, starts))
        burn = _fit_draws(one["sl_"].to_numpy(np.float64)) if sl_distr is None else []
        calls.append((sl.name, sl._draw_params(), ta.name, ta._draw_params(), sum(len(p) - 1 for p in parts), burn))
    if short:
        warnings.warn(f"{short} bursts of fewer than 3 steps are left out", UserWarning, stacklevel=2)
    draws = _core.move_random_step_draws(_seed(seed), calls, int(n_draws), int(n_control))
    blocks = []
    nc = int(n_control)
    for (parts, starts), (slr, tar) in zip(tracks, draws):
        used = 0
        for part, start in zip(parts, starts):
            m = len(part)
            if m < 2:
                continue
            obs = part.iloc[1:].drop(columns="direction_p").copy()
            obs["case_"] = True
            obs["step_id_"] = np.arange(2, m + 1) + int(start)
            ctl = obs.loc[obs.index.repeat(nc)].copy()
            x1 = part["x1_"].to_numpy(np.float64)
            y1 = part["y1_"].to_numpy(np.float64)
            angle = part["direction_p"].to_numpy(np.float64)

            def rep(v):
                return np.ascontiguousarray(np.repeat(v, nc))

            s = np.ascontiguousarray(slr[used:used + m - 1].ravel())
            t = np.ascontiguousarray(tar[used:used + m - 1].ravel())
            used += m - 1
            x2, y2, ta_ = _core.move_amt_controls(rep(x1[1:]), rep(y1[1:]), rep(x1[:-1]), rep(y1[:-1]),
                                                  rep(angle[:-1]), s, t)
            ctl["x2_"], ctl["y2_"], ctl["sl_"], ctl["ta_"] = x2, y2, s, ta_
            ctl["case_"] = False
            # observed first, then its random steps (dplyr::arrange is stable)
            order = np.argsort(np.r_[np.arange(m - 1) * (nc + 1), np.repeat(np.arange(m - 1) * (nc + 1), nc)
                                     + np.tile(np.arange(1, nc + 1), m - 1)], kind="stable")
            blocks.append(pd.concat([obs, ctl], ignore_index=True).iloc[order])
    out = pd.concat(blocks, ignore_index=True) if blocks else steps.iloc[:0].drop(columns="direction_p")
    out.attrs[ATTRS] = dict(info, sl_distr={t: d[0] for t, d in distrs.items()},
                            ta_distr={t: d[1] for t, d in distrs.items()}, n_control=nc)
    return out


def _layers(covariates) -> list[tuple[str, object, np.ndarray]]:
    """(name, raster, values (y, x)) of each band of a raster or of a dict / list of rasters."""
    from .._zonal import _raster, _values

    if isinstance(covariates, dict):
        items = list(covariates.items())
    elif isinstance(covariates, (list, tuple)):
        items = [(None, r) for r in covariates]
    else:
        items = [(None, covariates)]
    out = []
    for name, r in items:
        r, has_nodata, nodata, _, labels, _ = _raster(r, "covariates")
        v = _values(r).astype(np.float64)
        if v.ndim == 2:
            v = v[None]
        if has_nodata:
            v = np.where(v == nodata, np.nan, v)
        for b, label in enumerate(labels):
            out.append((name if name is not None and len(labels) == 1 else
                        (f"{name}_{label}" if name is not None else label), r, v[b]))
    return out


def extract_covariates(x, covariates, *, where: str = "end"):
    """The values of rasters at locations, or at the ends or starts of steps (amt's ``extract_covariates``).

    The value of the cell holding each point (terra's ``extract``: a point on
    the edge between two cells takes the cell to its right / below), NaN
    outside the raster or on no data. The points are taken to the raster's CRS.

    Parameters
    ----------
    x
        Tracks (a GeoDataFrame of points), their :func:`steps` or :func:`random_steps`.
    covariates
        A ``(y, x)`` or ``(band, y, x)`` DataArray, or a dict (name: raster) or
        list of them.
    where
        For steps: at their ``"end"`` (default), ``"start"`` or ``"both"``
        (columns suffixed ``_start`` and ``_end``).

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        ``x`` with one column per band (named by the band, the raster or the
        dict's key).

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> import landschaft as ls
    >>> forest = xr.DataArray(np.arange(16.0).reshape(4, 4), dims=("y", "x"),
    ...                       coords={"y": [3.5, 2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]}, name="forest")
    >>> tr = ls.move.simulate_brownian(range(4), tracks=1, start=(2, 2), seed=1)
    >>> ls.move.extract_covariates(tr, forest)["forest"].tolist()
    [10.0, 1.0, 1.0, 0.0]
    """
    from ..sdm._points import grid_cells, to_crs

    layers = _layers(covariates)
    if "x1_" in getattr(x, "columns", ()):
        crs = x.attrs.get(ATTRS, {}).get("crs")
        if where not in ("end", "start", "both"):
            raise ValueError('where: "end", "start" or "both"')
        spots = [("2", "_end" if where == "both" else "")] if where in ("end", "both") else []
        if where in ("start", "both"):
            spots.insert(0, ("1", "_start" if where == "both" else ""))
        points = [(x[f"x{e}_"].to_numpy(np.float64), x[f"y{e}_"].to_numpy(np.float64), suffix)
                  for e, suffix in spots]
    else:
        geom = getattr(x, "geometry", None)
        if geom is None:
            raise TypeError("x: tracks (a GeoDataFrame of points) or their steps")
        crs = x.crs
        points = [(geom.x.to_numpy(np.float64), geom.y.to_numpy(np.float64), "")]
    out = x.copy()
    out.attrs = dict(x.attrs)
    for xs, ys, suffix in points:
        for name, r, v in layers:
            px, py = to_crs(xs, ys, crs, r.rio.crs)
            cells = grid_cells(px, py, r)
            vals = np.full(len(cells), np.nan)
            ok = cells >= 0
            vals[ok] = v.ravel()[cells[ok]]
            out[name + suffix] = vals
    return out
