# SPDX-License-Identifier: GPL-3.0-or-later
"""A species' dispersal through the habitat suitability of successive periods, MigClim 1.6's cellular automaton."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from ..simulate._maps import read_map

NO_DATA = -9999
STAT_COLUMNS = ["period", "step", "step_id", "unlimited", "no_dispersal", "occupied", "absent", "colonized",
                "decolonized", "ldd"]


@dataclass
class Migration(Saveable):
    """A dispersal simulation (:func:`migrate`).

    Attributes
    ----------
    map : xarray.DataArray
        MigClim's final state: 1 the initial cells, a step's id (period × 100
        + step) where the species colonised, 30000 suitable but not reached,
        below 0 lost, 0 absent, NaN no data.
    occupied : xarray.DataArray
        1 where the species is at the end, 0 elsewhere.
    steps : xarray.DataArray
        The state after every dispersal step (dimension ``step``, its ids).
    table : pandas.DataFrame
        MigClim's statistics per step, the first row the start: ``period``,
        ``step``, ``step_id``, ``unlimited`` (the suitable cells),
        ``no_dispersal`` (the initial cells still suitable), ``occupied``,
        ``absent``, ``colonized``, ``decolonized``, ``ldd`` (colonised by
        long-distance dispersal).
    """

    map: xr.DataArray
    occupied: xr.DataArray
    steps: xr.DataArray
    table: pd.DataFrame


def _ints(m) -> np.ndarray:
    v = m.as_float()
    return np.where(np.isnan(v), NO_DATA, np.round(v)).astype(np.int32)


def _probabilities(v, name: str) -> list[float]:
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    if v.size == 0 or np.any(~((v >= 0) & (v <= 1))):
        raise ValueError(f"{name}: probabilities from 0 to 1")
    # MigClim reads them as C floats
    return [float(p) for p in v.astype(np.float32)]


def _floats(v: np.ndarray) -> np.ndarray:
    return np.where(v == NO_DATA, np.nan, v.astype(np.float64))


def _like(template, values: np.ndarray, name: str) -> xr.DataArray | np.ndarray:
    if template is None:
        return values
    out = xr.DataArray(values, dims=template.dims, coords=template.coords, name=name)
    if template.rio.crs is not None:
        out = out.rio.write_crs(template.rio.crs)
    return out


def migrate(initial, suitability, *, kernel, barriers=None, steps: int = 1, ini_maturity: int = 1,
            full_maturity: int = 1, propagules=None, threshold: int = 0, ldd_frequency: float = 0.0,
            ldd_min: int = 10, ldd_max: int = 15, barrier_type: str = "strong", seed: int = 1,
            nodata=None) -> Migration:
    """A species' dispersal through the habitat suitability of successive periods, as MigClim simulates it.

    MigClim 1.6's cellular automaton (Engler and Guisan 2009; Engler et al.
    2012), the same run for a seed: in every dispersal step of every period,
    a suitable empty cell is colonised from an occupied cell within the
    kernel's distance with the kernel's probability at that distance (times
    the source's propagule production while it matures, times the sink's
    suitability), unless barriers lie between; long-distance dispersal sends
    propagules to a random cell at a random distance; the cells that turn
    unsuitable stay, temporarily resilient, until the period ends. Its random
    numbers are Microsoft's C runtime ``rand()``, the one MigClim's Windows
    builds draw from.

    Parameters
    ----------
    initial
        The initial distribution: 1 present, 0 absent (NaN or ``nodata``: no
        data).
    suitability
        The habitat suitability of each period, 0 to 1000: a 3-D DataArray (a
        band per period), a list of maps, or one map.
    kernel
        The probability of colonisation at a distance of 1, 2... cells (its
        length the dispersal distance), e.g. ``[1, 0.4, 0.1]``.
    barriers
        The barriers (1), or None.
    steps
        Dispersal steps per period.
    ini_maturity, full_maturity
        The age (in steps) a colonised cell starts and fully produces
        propagules.
    propagules
        The propagule production of the ages from ``ini_maturity`` to
        ``full_maturity`` (default 1 each).
    threshold
        Suitability below it 0, at or above it 1000 (0: none).
    ldd_frequency
        Long-distance dispersal's frequency (0: none).
    ldd_min, ldd_max
        Its distances, in cells.
    barrier_type
        ``"strong"`` (blocks when two of the five lines between the cells cross
        a barrier) or ``"weak"`` (all five).
    seed
        The seed of ``rand()``.
    nodata
        The maps' missing value (besides NaN).

    Returns
    -------
    Migration

    Examples
    --------
    >>> import numpy as np, landschaft as ls
    >>> ini = np.zeros((20, 30)); ini[8:12, 2:5] = 1
    >>> hs = np.full((2, 20, 30), 800.0)
    >>> r = ls.sdm.migrate(ini, hs, kernel=[1.0, 0.5], steps=3, seed=1)
    >>> r.table[["period", "step", "occupied"]].iloc[-1].tolist()
    [2, 3, 297]
    """
    ini = read_map(initial, nodata)
    if isinstance(suitability, xr.DataArray) and suitability.ndim == 3:
        layers = [read_map(suitability[k], nodata) for k in range(suitability.shape[0])]
    elif isinstance(suitability, (list, tuple)):
        layers = [read_map(m, nodata) for m in suitability]
    else:
        a = suitability if isinstance(suitability, xr.DataArray) else np.asarray(suitability)
        layers = [read_map(a[k], nodata) for k in range(a.shape[0])] if a.ndim == 3 else [read_map(a, nodata)]
    if not layers or any(m.shape != ini.shape for m in layers):
        raise ValueError("suitability: a map per period, of the initial distribution's shape")
    bar = None
    if barriers is not None:
        b = read_map(barriers, nodata)
        if b.shape != ini.shape:
            raise ValueError("barriers: the initial distribution's shape")
        bar = _ints(b)
    if ini_maturity < 1 or full_maturity < ini_maturity:
        raise ValueError("ini_maturity: 1 or more; full_maturity: ini_maturity or more")
    ages = max(full_maturity - ini_maturity, 1)
    prop = _probabilities(np.ones(ages) if propagules is None else propagules, "propagules")
    if len(prop) < ages:
        raise ValueError(f"propagules: {ages} values (the ages from ini_maturity to full_maturity)")
    if barrier_type not in ("strong", "weak"):
        raise ValueError(f"barrier_type: 'strong' or 'weak', not {barrier_type!r}")
    ldd = float(np.float32(ldd_frequency))
    if not 0 <= threshold <= 1000 or not 0 <= ldd <= 1 or ldd_min < 0 or steps < 1:
        raise ValueError("threshold: 0 to 1000; ldd_frequency: 0 to 1; ldd_min: 0 or more; steps: 1 or more")
    if ldd > 0 and ldd_max <= ldd_min:
        raise ValueError("ldd_max: above ldd_min")
    fin, st, ids, stats = _core.sdm_migclim(
        _ints(ini), np.stack([_ints(m) for m in layers]), bar, int(steps), _probabilities(kernel, "kernel"),
        int(ini_maturity), int(full_maturity), prop, int(threshold), ldd, int(ldd_min), int(ldd_max),
        barrier_type == "weak", int(seed) & 0xFFFFFFFF)
    tpl = ini.template
    occ = np.where(fin == NO_DATA, np.nan, ((fin > 0) & (fin < 30000)).astype(np.float64))
    if tpl is None:
        layers_out = _floats(st)
    else:
        layers_out = xr.DataArray(_floats(st), dims=("step", *tpl.dims), coords={"step": ids, **tpl.coords},
                                  name="state")
        if tpl.rio.crs is not None:
            layers_out = layers_out.rio.write_crs(tpl.rio.crs)
    return Migration(_like(tpl, _floats(fin), "migclim"), _like(tpl, occ, "occupied"), layers_out,
                     pd.DataFrame(stats, columns=STAT_COLUMNS))
