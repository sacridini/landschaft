# SPDX-License-Identifier: GPL-3.0-or-later
"""Restoration Planner setup and isochrone maps (GuidosToolbox *Restoration Planner*).

- :func:`distance_to_resistance`: *Setup Tools -> Distance -> Resistance*, a
  resistance map growing with the distance to the foreground objects.
- :func:`isochrone_map`: *Assessment -> Isochrone Map A*, the accumulated cost
  (geodesic time) of reaching every pixel from object A across a resistance map.
- :func:`isochrone_map_ab`: *Isochrone Map AB*, the cost of the cheapest route
  from A to B through every pixel, with the least-cost path.

The costs are those of GTB's ``ggeo`` program (miallib ``sqtgsym``): a step
between neighbouring pixels p and q (8-connectivity, diagonals included) costs
``(r[p] + r[q]) / 2``; foreground pixels (resistance 2) cost 1 and missing
pixels of the marker map block the way. The maps use GTB's display codes.
"""

from __future__ import annotations

from ._threads import threaded

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import xarray as xr

from . import _core
from .io import _load_map
from ._save import Saveable

__all__ = [
    "ISOCHRONE_CLASSES",
    "IsochroneResult",
    "distance_to_resistance",
    "isochrone_map",
    "isochrone_map_ab",
]

ISOCHRONE_CLASSES: dict[int, str] = {
    **{v: "Cost, scaled to 0-100" for v in (0, 100)},
    101: "Marker objects (A, B)",
    102: "Missing (marker value 3)",
    103: "Least-cost path (map AB)",
    104: "Unreachable",
}

_EEW = 2  # GTB's frame around the maps given to ggeo
_LONG_MIN = -(2**31)


def _load(data):
    if isinstance(data, (str, Path)):
        data = _load_map(data)
    template = data if isinstance(data, xr.DataArray) else None
    arr = np.asarray(data.values if template is not None else data)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D map, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"maps must be uint8 (byte), got {arr.dtype}")
    return np.ascontiguousarray(arr), template


def _wrap(values: np.ndarray, template, name: str, attrs: dict):
    if template is None:
        return values
    da = template.copy(data=values)
    da.name = name
    da.attrs = dict(template.attrs) | attrs
    return da


def _idl_round_long(v: np.ndarray) -> np.ndarray:
    """IDL ``round(ulong / 2.0)``: single precision, half away from zero, LONG (overflow -> -2^31)."""
    f = v.astype(np.float32) / np.float32(2.0)
    r = np.floor(np.abs(f.astype(np.float64)) + 0.5) * np.sign(f)
    out = r.astype(np.int64)
    out[(r >= 2**31) | (r < _LONG_MIN)] = _LONG_MIN
    return out


def _idl_round_scalar(v: float) -> int:
    f = float(np.float32(v) / np.float32(2.0))
    r = int(math.copysign(math.floor(abs(f) + 0.5), f))
    return r if _LONG_MIN <= r < 2**31 else _LONG_MIN


def _bytscl(a: np.ndarray, lo: int, hi: int, top: int = 100) -> np.ndarray:
    """IDL BYTSCL of an integer array: ((top + 1) * (a - min) - 1) / (max - min), clipped."""
    if hi <= lo:
        return np.where(a >= hi, top, 0).astype(np.uint8)
    x = np.clip(a.astype(np.int64), lo, hi)
    num = (top + 1) * (x - lo) - 1
    q = np.where(num >= 0, num // (hi - lo), -((-num) // (hi - lo)))  # C integer division
    return np.clip(q, 0, top).astype(np.uint8)


def _ggeo_mask(resist: np.ndarray) -> np.ndarray:
    """GTB's mask for ggeo: foreground (2) costs 1.

    GTB does ``q = where(resist eq 2b) & resist[q] = 1b``; without foreground,
    ``where`` returns the scalar -1, which IDL (8+) takes as the last element
    of GTB's row-flipped image: the top-right pixel then costs 1 as well.
    """
    mask = resist.copy()
    if (resist == 2).any():
        mask[resist == 2] = 1
    else:
        mask[0, -1] = 1
    return mask


def _geodesic_time(mask: np.ndarray, ref: np.ndarray) -> np.ndarray:
    t = _core.compute_geodesic_time(np.pad(mask, _EEW), np.pad(ref, _EEW, constant_values=3))
    return t[_EEW:-_EEW, _EEW:-_EEW]


def _check_marker(marker: np.ndarray, resist: np.ndarray, values: tuple[int, ...]) -> None:
    if marker.shape != resist.shape:
        raise ValueError(f"marker and resistance maps differ in size: {marker.shape} and {resist.shape}")
    present = set(np.unique(marker).tolist())
    for v in (0, *values):
        if v not in present:
            names = {0: "0: background", 1: "1: object A", 2: "2: object B"}
            raise ValueError("the marker map must contain " + ", ".join(names[k] for k in (0, *values)))
    for v in values:
        if ndi.label(marker == v, structure=np.ones((3, 3)))[1] >= 2:
            raise ValueError(f"the marker map must have only one object with value {v}")


@dataclass(frozen=True)
class IsochroneResult(Saveable):
    """Result of :func:`isochrone_map` / :func:`isochrone_map_ab`.

    Attributes
    ----------
    map
        ``uint8`` isochrone map in GTB's codes: 0-100 cost scaled between
        ``min_cost`` and ``max_cost``, 101 markers, 102 missing, 103 least-cost
        path (AB), 104 unreachable.
    cost
        Accumulated cost as GTB rounds it (``int64``; unreachable/missing: -2^31).
        For map AB, the cost of the cheapest A-B route through each pixel.
    lcp
        Least-cost path mask (map AB), else ``None``: the pixels of minimal
        A-B cost, thinned to a 1-pixel wide path as GTB's ``ggeo`` does.
    """

    map: xr.DataArray | np.ndarray
    cost: xr.DataArray | np.ndarray
    lcp: xr.DataArray | np.ndarray | None
    min_cost: int
    max_cost: int


def isochrone_map(resistance, marker) -> IsochroneResult:
    """Accumulated cost from object A (GTB Restoration Planner -> Isochrone Map A).

    Parameters
    ----------
    resistance
        ``uint8`` resistance map: 2 = foreground (cost 1), 3-100 = background
        resistance (GTB's Restoration Planner coding; 0 costs nothing, as in GTB).
    marker
        ``uint8`` map of the same size: 0 background, 1 object A (a single
        8-connected object), 3 missing (not crossed); 2 is ignored.

    Returns
    -------
    IsochroneResult

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> import numpy as np
    >>> resistance = ls.distance_to_resistance(forest)
    >>> objects = ls.accounting(forest)                     # the two largest forest objects
    >>> ids = objects.ids.values
    >>> marker = np.zeros(forest.shape, dtype="uint8")
    >>> marker[ids == objects.largest[0][0]] = 1            # object A
    >>> marker[ids == objects.largest[1][0]] = 2            # object B
    >>> marker[forest.values == 0] = 3                      # missing: not crossed
    >>> only_a = np.where(marker == 2, 0, marker).astype("uint8")
    >>> rings = ls.isochrone_map(resistance, only_a)
    >>> rings.min_cost, rings.max_cost
    (2, 1166)
    """
    resist, template = _load(resistance)
    mark, _ = _load(marker)
    _check_marker(mark, resist, (1,))
    mask = _ggeo_mask(resist)
    ref = np.where(mark == 2, 0, mark).astype(np.uint8)
    t = _geodesic_time(mask, ref)

    # GTB: cost = round(cost / 2.0); maxcost, possibly below the missing pixels' value
    cost = _idl_round_long(t)
    maxcost = int(cost.max())
    missing = mark == 3
    if missing.any():
        maxcost = int(((cost < maxcost) * cost).max())
    unreachable = cost < -2
    out = _bytscl(cost, 0, maxcost)
    out[unreachable] = 104
    out[mark == 1] = 101
    out[missing] = 102
    positive = t[t > 0]
    mincost = _idl_round_scalar(float(positive.min())) if positive.size else 0
    attrs = {"landschaft_product": "ISOCHRONE_A", "min_cost": mincost, "max_cost": maxcost}
    return IsochroneResult(
        map=_wrap(out, template, "isochrone_a", attrs),
        cost=_wrap(cost, template, "isochrone_a_cost", attrs | {"landschaft_product": "ISOCHRONE_A_COST"}),
        lcp=None,
        min_cost=mincost,
        max_cost=maxcost,
    )


@threaded
def isochrone_map_ab(resistance, marker) -> IsochroneResult:
    """Cheapest route between objects A and B (GTB Restoration Planner -> Isochrone Map AB).

    ``marker`` holds 1 = object A, 2 = object B (each a single 8-connected
    object, not touching each other), 3 = missing, 0 = background; see
    :func:`isochrone_map` for ``resistance``.

    Parameters
    ----------
    resistance
        ``uint8`` resistance map, as in :func:`isochrone_map`.
    marker
        ``uint8`` map: 1 object A, 2 object B, 3 missing, 0 background.
    n_threads
        Threads (0 or ``None``: all cores but one).

    Returns
    -------
    IsochroneResult
        With the least-cost path in ``lcp``.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> import numpy as np
    >>> resistance = ls.distance_to_resistance(forest)
    >>> objects = ls.accounting(forest)                     # the two largest forest objects
    >>> ids = objects.ids.values
    >>> marker = np.zeros(forest.shape, dtype="uint8")
    >>> marker[ids == objects.largest[0][0]] = 1            # object A
    >>> marker[ids == objects.largest[1][0]] = 2            # object B
    >>> marker[forest.values == 0] = 3                      # missing: not crossed
    >>> route = ls.isochrone_map_ab(resistance, marker)
    >>> route.min_cost, int(np.asarray(route.lcp).sum())
    (210, 215)
    """
    resist, template = _load(resistance)
    mark, _ = _load(marker)
    _check_marker(mark, resist, (1, 2))
    if ndi.label((mark == 1) | (mark == 2), structure=np.ones((3, 3)))[1] == 1:
        raise ValueError("marker objects A and B must not touch each other")
    if (mark == 3).any():
        comp, _ = ndi.label(mark != 3, structure=np.ones((3, 3)))
        if comp[mark == 1][0] != comp[mark == 2][0]:
            raise ValueError("marker objects A and B can not be connected (separated by missing pixels)")

    mask = _ggeo_mask(resist)
    t_a = _geodesic_time(mask, np.where(mark == 2, 0, mark).astype(np.uint8))
    ref_b = mark.copy()  # GTB's marker2: B becomes the reference, A background
    ref_b[mark == 1] = 0
    ref_b[mark == 2] = 1
    t_b = _geodesic_time(mask, ref_b)
    total = t_a + t_b  # uint32 arithmetic, as in ggeo
    # ggeo's least-cost path: the pixels of minimal A-B cost, thinned to a
    # 1-pixel wide path (miallib binOIthin_FIFO, type 0) anchored on A and B
    on_min = np.pad((total == total.min()).astype(np.uint8), _EEW)
    anchors = np.pad(((mark == 1) | (mark == 2)).astype(np.uint8), _EEW)
    lcp = _core.compute_thin_anchored(on_min, 0, anchors)[_EEW:-_EEW, _EEW:-_EEW] > 0

    mincost = _idl_round_scalar(float(total.min()))
    maxcost_u = int(total.max())
    missing = mark == 3
    if missing.any():
        maxcost = _idl_round_scalar(float(((total < maxcost_u) * total.astype(np.int64)).max()))
    else:
        maxcost = _idl_round_scalar(float(maxcost_u))
    cost = _idl_round_long(total)
    unreachable = cost < -2
    out = _bytscl(cost, mincost, maxcost)
    out[lcp] = 103
    out[(mark == 1) | (mark == 2)] = 101
    out[missing] = 102
    out[unreachable] = 104
    attrs = {"landschaft_product": "ISOCHRONE_AB", "min_cost": mincost, "max_cost": maxcost}
    return IsochroneResult(
        map=_wrap(out, template, "isochrone_ab", attrs),
        cost=_wrap(cost, template, "isochrone_ab_cost", attrs | {"landschaft_product": "ISOCHRONE_AB_COST"}),
        lcp=_wrap(lcp, template, "isochrone_ab_lcp", attrs | {"landschaft_product": "ISOCHRONE_AB_LCP"}),
        min_cost=mincost,
        max_cost=maxcost,
    )


@threaded
def distance_to_resistance(data):
    """Resistance growing with the distance to the foreground (GTB Setup Tools -> Distance -> Resistance).

    ``data`` is coded 0 missing, 1 background, 2 foreground. Background pixels
    get ``round(distance + 2)`` capped at 100 (so 3 next to an object),
    foreground 2 and missing 0. The distance is IDL's MORPH_DISTANCE as GTB
    uses it (quasi-Euclidean; see :func:`landschaft.influence_zones` for the
    remaining differences to IDL).

    Parameters
    ----------
    data
        Map coded 0 missing, 1 background, 2 foreground.
    n_threads
        Threads (0 or ``None``: all cores but one).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        ``uint8`` resistance map (see :func:`resistance_colormap`).

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> resistance = ls.distance_to_resistance(forest)
    >>> int(resistance.min()), int(resistance.max())
    (0, 38)
    """
    arr, template = _load(data)
    if not (arr == 2).any():
        raise ValueError("the map has no foreground objects (value 2)")
    flipped = arr[::-1]  # GTB works on the image as displayed (rows flipped)
    ext = np.pad((flipped == 2).astype(np.uint8), _EEW)
    d = _core.compute_morph_distance(ext, True, "quasi-euclidean")[_EEW:-_EEW, _EEW:-_EEW][::-1]
    v = np.minimum(d + np.float32(2.0), np.float32(100.0)).astype(np.float64)
    out = np.maximum(np.floor(v + 0.5), 0).astype(np.uint8)
    out[arr == 2] = 2
    out[arr == 0] = 0
    return _wrap(out, template, "resistance", {"landschaft_product": "RESISTANCE", "setup": "distance"})
