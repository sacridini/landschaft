# SPDX-License-Identifier: GPL-3.0-or-later
"""A land change projection in one call: fit on two maps, check against the later one, project."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from ._change import _class_value
from ._maps import read_map

__all__ = ["Projection", "run"]

_PRESETS = ("dinamica", "plus", "molusce")


def _year_of(x, default):
    """A map's year: a scalar ``year``/``time``/``band`` coordinate that is a number, else the default."""
    if isinstance(x, xr.DataArray):
        for name in ("year", "time", "band"):
            if name in x.coords and x.coords[name].ndim == 0:
                v = x.coords[name].item()
                try:
                    return float(v) if not float(v).is_integer() else int(v)
                except (TypeError, ValueError):
                    continue
    return default


def _with_steps(model, n: int):
    """The model with its allocation set to simulate ``n`` steps."""
    m = copy.copy(model)
    m.allocation = copy.copy(model.allocation)
    if hasattr(m.allocation, "iterations"):
        m.allocation.iterations = n
    elif hasattr(m.allocation, "steps"):
        m.allocation.steps = n
    else:
        raise ValueError(f"{type(model.allocation).__name__} takes its steps from the demand: give a demand "
                         "covering the projection")
    return m


def agreement(observed, simulated, start, *, nodata=None) -> dict:
    """How a simulated map matches the observed one, cell by cell and in quantity.

    Returns ``overall`` (share of cells of the same class), ``figure_of_merit``
    (hits / (hits + misses + false alarms + wrong hits), Pontius et al.
    2008), the ``hits``, ``misses``, ``false_alarms`` and ``wrong_hits``
    (change simulated where it happened, to another class), and the
    ``quantity`` difference of each class (simulated - observed cells).
    """
    o, s, a = (read_map(m, nodata).as_float() for m in (observed, simulated, start))
    ok = ~(np.isnan(o) | np.isnan(s) | np.isnan(a))
    o, s, a = o[ok], s[ok], a[ok]
    obs_ch, sim_ch = o != a, s != a
    hits = int((obs_ch & sim_ch & (o == s)).sum())
    wrong = int((obs_ch & sim_ch & (o != s)).sum())
    misses = int((obs_ch & ~sim_ch).sum())
    false = int((~obs_ch & sim_ch).sum())
    denom = hits + misses + false + wrong
    classes = sorted({_class_value(c) for c in np.unique(np.concatenate([o, s]))})
    return {"overall": float((o == s).mean()) if o.size else np.nan,
            "figure_of_merit": hits / denom if denom else np.nan,
            "hits": hits, "misses": misses, "false_alarms": false, "wrong_hits": wrong,
            "quantity": {c: int((s == c).sum() - (o == c).sum()) for c in classes}}


@dataclass
class Projection:
    """The result of :func:`run`.

    Attributes
    ----------
    maps : xarray.DataArray
        ``(year, y, x)``: the later observed map and the projected years.
    calibration : Simulation or None
        The simulation of the observed period, from the earlier map.
    validation : dict or None
        :func:`agreement` of the calibration's last map with the later map.
    model : Model
        The fitted model.
    report : pandas.DataFrame
        What each stage did.
    settings : dict
        The choices (preset, years, seed, stage settings), enough to repeat
        the projection.
    """

    maps: xr.DataArray
    calibration: object
    validation: dict | None
    model: object
    report: pd.DataFrame
    settings: dict
    simulation: object = None
    _call: dict = field(default=None, repr=False)

    def redo(self, **changes) -> Projection:
        """Run again from the same maps with some settings changed (``proj.redo(seed=2, until=2060)``)."""
        args = {**self._call["kwargs"], **changes}
        return run(*self._call["args"], **args)

    def to_dataset(self) -> xr.Dataset:
        """The projected maps (and the calibration's, on its own year axis) as one Dataset."""
        ds = xr.Dataset({"landuse": self.maps})
        if self.calibration is not None:
            ds["calibration"] = self.calibration.maps.rename({"step": "calibration_year"}).assign_coords(
                calibration_year=self.settings["calibration_years"])
        return ds

    def __repr__(self) -> str:
        y = self.maps["year"].values
        fom = None if self.validation is None else self.validation["figure_of_merit"]
        return (f"Projection({self.settings['preset']}, years {y[0]}-{y[-1]}"
                + ("" if fom is None else f", figure of merit {fom:.3f}") + ")")


def run(t0, t1, drivers=None, *, until=None, years=None, preset="dinamica", seed=None, validate=True,
        dynamic=None, rules=None, nodata=None, **options) -> Projection:
    """Fit a land change model on two maps, check it on the observed period and project it.

    1. The preset is fitted on ``t0 -> t1`` with one step per year;
    2. ``validate``: the model simulates the observed period from ``t0``
       and its last map is compared with ``t1`` (:func:`agreement`);
    3. ``until``: the model projects from ``t1``, a step per year.

    Parameters
    ----------
    t0, t1
        The earlier and later maps.
    drivers
        The explanatory variables.
    until
        Last year of the projection (default: none, only the calibration).
    years
        ``(year of t0, year of t1)``; default: the maps' ``year``/``time``/
        ``band`` coordinate, else ``(0, 1)``.
    preset
        ``"dinamica"`` (Markov + weights of evidence + Expander/Patcher with
        the observed patch sizes), ``"plus"`` (LEAS + CARS) or
        ``"molusce"`` (its WoE and simulator, a step per observed period),
        or a function ``f(t0, t1, drivers, steps=..., **options)`` returning
        a :class:`Model`.
    seed
        Seed of every random draw (default: drawn, kept in ``settings``).
    validate
        Simulate the observed period first and compare with ``t1``.
    dynamic, rules
        Dynamic variables and rules given to the model.
    options
        The preset's settings (e.g. ``woe=30``, ``patches=0.6``,
        ``isometry=1.3`` for ``"dinamica"``; ``estimator=``, ``seeds=``
        for ``"plus"``; ``bins=`` for ``"molusce"``).

    Returns
    -------
    Projection

    Examples
    --------
    >>> import landschaft as ls
    >>> lu = ls.load_raster(ls.simulate.example_data("sibuyan_landuse"))
    >>> drivers = ls.load_raster(ls.simulate.example_data("sibuyan_drivers"))[[0, 10]]
    >>> proj = ls.simulate.run(lu[0], lu[1], drivers, until=2015, seed=1, woe=250)   # maps of 1997 and 2011
    >>> proj.maps["year"].values.tolist()
    [2011, 2012, 2013, 2014, 2015]
    >>> sorted(proj.report["stage"])
    ['fit', 'projection', 'validation']
    """
    from . import presets

    call = {"args": (t0, t1, drivers),
            "kwargs": dict(until=until, years=years, preset=preset, seed=seed, validate=validate, dynamic=dynamic,
                           rules=rules, nodata=nodata, **options)}
    if years is None:
        years = (_year_of(t0, 0), _year_of(t1, 1))
    y0, y1 = years
    if not y1 > y0:
        raise ValueError("the later map must come after the earlier one (years=(t0, t1))")
    n = round(y1 - y0)
    if until is not None and until <= y1:
        raise ValueError("until must come after the later map")
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    if callable(preset):
        fit, name = preset, getattr(preset, "__name__", "custom")
    elif preset in _PRESETS:
        fit, name = getattr(presets, preset), preset
    else:
        raise ValueError(f"preset must be one of {_PRESETS} or a function")
    rows = []
    kw = dict(options)
    if name == "molusce":
        steps_per_period = 1
        model = fit(t0, t1, drivers, dynamic=dynamic, nodata=nodata, **kw)
    else:
        steps_per_period = n
        if name == "plus":
            kw.setdefault("seed", seed)
        model = fit(t0, t1, drivers, steps=n, rules=rules, dynamic=dynamic, nodata=nodata, **kw)
    rows.append({"stage": "fit", "detail": f"{name} on {y0}-{y1}, {steps_per_period} step(s) in the period"})
    calibration, validation = None, None
    cal_years = [y0, y1] if name == "molusce" else [y0 + j for j in range(n + 1)]
    if validate:
        calibration = _with_steps(model, steps_per_period).run(t0, drivers, seed=seed, nodata=nodata)
        last = calibration.maps.isel(step=-1)
        validation = agreement(t1, last, t0, nodata=nodata)
        rows.append({"stage": "validation", "detail": f"figure of merit {validation['figure_of_merit']:.4f}, "
                                                      f"agreement {validation['overall']:.4f}"})
    later = read_map(t1, nodata)
    simulation = None
    if until is not None:
        if name == "molusce":
            k = int(np.ceil((until - y1) / (y1 - y0)))
            step_years = [y1 + (j + 1) * (y1 - y0) for j in range(k)]
        else:
            k = round(until - y1)
            step_years = [y1 + j + 1 for j in range(k)]
        simulation = _with_steps(model, k).run(t1, drivers, seed=seed + 1, nodata=nodata)
        maps = simulation.maps.rename({"step": "year"}).assign_coords(year=[y1, *step_years])
        rows.append({"stage": "projection", "detail": f"{k} step(s) from {y1} to {step_years[-1]}"})
    else:
        from ._model import _as_maps

        maps = _as_maps(later.as_float()[None], later.template, "landuse").rename({"step": "year"}).assign_coords(
            year=[y1])
    settings = {"preset": name, "years": [y0, y1], "calibration_years": cal_years, "until": until, "seed": seed, "validate": bool(validate),
                "dynamic": sorted(dynamic) if dynamic else None,
                "options": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else type(v).__name__)
                            for k, v in options.items()}}
    return Projection(maps, calibration, validation, model, pd.DataFrame(rows), settings, simulation, call)
