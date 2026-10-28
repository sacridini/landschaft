# SPDX-License-Identifier: GPL-3.0-or-later
"""A metapopulation in a network of habitat patches: Hanski's incidence function model as MetaLandSim simulates it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import _core
from .._save import Saveable

__all__ = ["Metapopulation", "metapopulation"]

_KERNELS = ("exponential", "power")
_CONNECTIVITY = ("plain", "area")
_COLONISATION = ("logistic", "exponential", "power")
_EXTINCTION = ("area", "exponential", "rescue")


@dataclass
class Metapopulation(Saveable):
    """A metapopulation's simulation (:func:`metapopulation`).

    Attributes
    ----------
    patches : pandas.DataFrame
        A row per patch: ``patch`` (1, 2...), ``x``, ``y``, ``area``,
        ``occupied`` (at the end, 0 or 1) and ``incidence`` (the share of the
        steps it was occupied).
    table : pandas.DataFrame
        A row per step, 0 the start: ``step``, ``occupied`` (patches),
        ``occupied_area``, ``colonised``, ``extinct``.
    occupancy : numpy.ndarray
        The occupancy after every step, ``(steps, patches)`` booleans.
    """

    patches: pd.DataFrame
    table: pd.DataFrame
    occupancy: np.ndarray


def _patches(patches, area: str, occupied: str):
    try:
        import geopandas as gpd
    except ImportError:  # pragma: no cover
        gpd = None
    if gpd is not None and isinstance(patches, gpd.GeoDataFrame):
        geom = patches.geometry
        if geom.geom_type.isin(["Point"]).all():
            x, y = geom.x.to_numpy(float), geom.y.to_numpy(float)
        else:
            c = geom.centroid
            x, y = c.x.to_numpy(float), c.y.to_numpy(float)
        a = patches[area].to_numpy(float) if area in patches else geom.area.to_numpy(float)
        table = patches
    else:
        table = pd.DataFrame(patches)
        for col in ("x", "y", area):
            if col not in table:
                raise ValueError(f"patches: no column {col!r}")
        x, y, a = (table[c].to_numpy(float) for c in ("x", "y", area))
    if occupied not in table:
        raise ValueError(f"patches: no column {occupied!r}")
    occ = table[occupied].to_numpy(float)
    occ = (occ != 0) & ~np.isnan(occ)
    if len(x) < 2 or np.any(~(a > 0)):
        raise ValueError("patches: two at least, every one with an area above 0")
    return x, y, a, occ


def metapopulation(patches, *, alpha: float, x: float, y: float, e: float, steps: int = 100, b: float = 1.0,
                   kernel: str = "exponential", beta: float = 1.0, connectivity: str = "plain",
                   c1: float | None = None, colonisation: str = "logistic", c2: float | None = None,
                   z: float | None = None, extinction: str = "area", r: float | None = None, area: str = "area",
                   occupied: str = "occupied", seed: int = 1) -> Metapopulation:
    """A metapopulation in a network of habitat patches over time, Hanski's incidence function model.

    As R's MetaLandSim (``spom``) iterates it, the same run for a seed: every
    step, each patch's connectivity to the occupied patches is a kernel of the
    distance times their areas^b; an empty patch is colonised with a
    probability that grows with it, an occupied one lost with a probability
    that falls with its area (and, with the rescue effect, with its
    colonisation); the draws are R's ``runif`` after ``set.seed(seed)``.

    Parameters
    ----------
    patches
        The patches: a table with columns ``x``, ``y``, the area and the
        occupancy, or a GeoDataFrame (points, or polygons: their centroids,
        and their areas when it has no area column).
    alpha
        The kernel's α: 1 / the mean dispersal distance.
    x
        Extinction's scaling with area.
    y
        Colonisation's y.
    e
        Extinction's e: the extinction probability of a patch of area 1.
    steps
        The steps.
    b
        The sources' areas' exponent in the connectivity.
    kernel
        ``"exponential"`` (exp(-α d)) or ``"power"`` (1 / (1 + α d^beta)).
    beta
        The power kernel's exponent.
    connectivity
        ``"plain"`` or ``"area"``: times the patch's own area^c1.
    c1
        The exponent of ``"area"`` connectivity.
    colonisation
        ``"logistic"`` (S² / (S² + y²)), ``"exponential"`` (1 - exp(-y S)) or
        ``"power"`` (S^z / (S^z + 1/c2)).
    c2, z
        Power colonisation's parameters.
    extinction
        ``"area"`` (min(1, e / A^x)), ``"exponential"`` (1 - exp(-e / A^x)) or
        ``"rescue"`` (min(1, e / A^x (1 - C)^r)).
    r
        The rescue effect's exponent.
    area, occupied
        The names of the area and occupancy columns.
    seed
        R's ``set.seed``.

    Returns
    -------
    Metapopulation

    Examples
    --------
    >>> import numpy as np, pandas as pd, landschaft as ls
    >>> g = np.random.default_rng(1)
    >>> p = pd.DataFrame({"x": g.uniform(0, 50, 30), "y": g.uniform(0, 50, 30), "area": g.uniform(0.5, 4, 30),
    ...                   "occupied": g.random(30) < 0.5})
    >>> m = ls.conn.metapopulation(p, alpha=0.1, x=0.5, y=5, e=0.3, steps=20, seed=2)
    >>> m.table["occupied"].tolist()[:5]
    [18, 22, 20, 24, 25]
    """
    xs, ys, a, occ = _patches(patches, area, occupied)
    for name, value, allowed in (("kernel", kernel, _KERNELS), ("connectivity", connectivity, _CONNECTIVITY),
                                 ("colonisation", colonisation, _COLONISATION),
                                 ("extinction", extinction, _EXTINCTION)):
        if value not in allowed:
            raise ValueError(f"{name}: one of {', '.join(allowed)}, not {value!r}")

    def need(name, value):
        if value is None:
            raise ValueError(f"{name}: needed by the model chosen")
        return float(value)

    c1 = need("c1", c1) if connectivity == "area" else 0.0
    c2, z = (need("c2", c2), need("z", z)) if colonisation == "power" else (1.0, 1.0)
    r = need("r", r) if extinction == "rescue" else 1.0
    if steps < 0:
        raise ValueError("steps: 0 or more")
    out = _core.sdm_metapopulation(
        xs, ys, a, occ.astype(np.uint8), int(steps), _KERNELS.index(kernel), _CONNECTIVITY.index(connectivity),
        _COLONISATION.index(colonisation), _EXTINCTION.index(extinction), float(alpha), float(x), float(y),
        float(e), float(b), float(beta), c1, c2, z, r, int(seed)).astype(bool)
    states = np.vstack([occ[None, :], out])
    before, after = states[:-1], states[1:]
    table = pd.DataFrame({
        "step": np.arange(steps + 1),
        "occupied": states.sum(axis=1),
        "occupied_area": [float(np.sum(a[s])) for s in states],
        "colonised": np.r_[0, np.sum(after & ~before, axis=1)],
        "extinct": np.r_[0, np.sum(~after & before, axis=1)],
    })
    final = states[-1]
    incidence = out.mean(axis=0) if steps > 0 else final.astype(float)
    table_p = pd.DataFrame({"patch": np.arange(1, len(xs) + 1), "x": xs, "y": ys, "area": a,
                            "occupied": final.astype(int), "incidence": incidence})
    return Metapopulation(table_p, table, out)
