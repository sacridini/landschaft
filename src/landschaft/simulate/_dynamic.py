# SPDX-License-Identifier: GPL-3.0-or-later
"""Dynamic variables: drivers computed again from the simulated map at every step.

Distances to a class, the share of a class around each cell, or any
function of the map ("moving frontier" variables): a :class:`Model` given
``dynamic={name: piece}`` recomputes them from the map at the start of each
step and predicts the potential again with them in place of the static
driver of the same name. Each piece is also a plain function of a map, to
fit the potential with the same values on the starting map.
"""

from __future__ import annotations

import math

import numpy as np

from .. import _core
from ._change import _class_value
from ._maps import read_map

__all__ = ["distance_to", "share_of"]


def _classes_of(classes) -> list:
    if np.isscalar(classes):
        classes = [classes]
    return [_class_value(c) for c in classes]


def _resolution(template) -> tuple[float, float] | None:
    if template is None:
        return None
    try:
        rx, ry = (abs(float(v)) for v in template.rio.resolution())
    except Exception:  # noqa: BLE001  (no georeference: cells)
        return None
    return rx, ry


def _like(values: np.ndarray, template, name: str):
    if template is None:
        return values
    return template.copy(data=values).rename(name)


class distance_to:  # used as a function: S.distance_to(3)
    """Distance from each cell to the nearest cell of some classes.

    Parameters
    ----------
    classes
        A class or a list of classes.
    cost
        ``None``: the exact Euclidean distance (Felzenszwalb & Huttenlocher's
        transform), in map units (cells without a georeference); or a
        ``ls.cost`` :class:`~landschaft.cost.Transition` of conductances:
        the accumulated cost from the nearest such cell (gdistance's
        ``accCost``), a distance along roads, rivers or any network.
    unit : {"map", "cells"}
        Euclidean distance in the map's units (square cells) or in cells.

    Calling it with a map gives the distance map (NaN where the map has no
    value). With no cell of the classes, every cell gets the length of the
    map's diagonal (Euclidean) or ``inf`` (cost).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> land = np.array([[1, 1, 1, 3], [1, 1, 1, 1]])
    >>> ls.simulate.distance_to(3)(land)
    array([[3.        , 2.        , 1.        , 0.        ],
           [3.16227766, 2.23606798, 1.41421356, 1.        ]])
    """

    def __init__(self, classes, *, cost=None, unit: str = "map"):
        if unit not in ("map", "cells"):
            raise ValueError("unit must be 'map' or 'cells'")
        self.classes = _classes_of(classes)
        self.cost = cost
        self.unit = unit

    def __call__(self, landscape, nodata=None):
        land = read_map(landscape, nodata)
        v = land.as_float()
        origin = np.isin(v, self.classes)
        if self.cost is not None:
            from ..cost import acc_cost

            if tuple(self.cost.shape) != v.shape:
                raise ValueError("the cost transition differs in shape from the map")
            if origin.any():
                out = np.asarray(acc_cost(self.cost, origin.astype(np.uint8)), dtype=np.float64).copy()
            else:
                out = np.full(v.shape, np.inf)
        else:
            res = 1.0
            if self.unit == "map":
                r = _resolution(land.template)
                if r is not None:
                    if not math.isclose(r[0], r[1], rel_tol=1e-9):
                        raise ValueError("Euclidean distances in map units need square cells (unit='cells')")
                    res = r[0]
            if origin.any():
                out = np.asarray(_core.nlm_distance(np.ascontiguousarray(origin.astype(np.uint8)), "exact", res),
                                 dtype=np.float64)
            else:
                out = np.full(v.shape, math.hypot(*v.shape) * res)
        out[np.isnan(v)] = np.nan
        return _like(out, land.template, f"distance_to_{'_'.join(f'{c:g}' for c in self.classes)}")

    def __repr__(self) -> str:
        return f"distance_to({self.classes!r}{', cost=...' if self.cost is not None else ''})"


class share_of:
    """Share of the cells around each cell that hold some classes.

    Parameters
    ----------
    classes
        A class or a list of classes.
    window
        Side of a square window of ones (odd), or the window's weights.

    The mean of ``weights x presence`` over the window, the cells without a
    value left out (the centre included), as :class:`Neighbourhood`.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> land = np.array([[1, 2, 2], [1, 1, 2], [1, 1, 1]])
    >>> float(ls.simulate.share_of(2)(land)[1, 1])
    0.3333333333333333
    """

    def __init__(self, classes, window=3):
        self.classes = _classes_of(classes)
        w = np.ones((int(window), int(window))) if np.isscalar(window) else np.asarray(window, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] % 2 == 0 or w.shape[1] % 2 == 0:
            raise ValueError("the window needs two odd sides")
        self.window = np.ascontiguousarray(w)

    def __call__(self, landscape, nodata=None):
        land = read_map(landscape, nodata)
        v = land.as_float()
        presence = np.where(np.isnan(v), np.nan, np.isin(v, self.classes).astype(np.float64))
        out = np.asarray(_core.sim_focal_mean(np.ascontiguousarray(presence), self.window), dtype=np.float64)
        out = out.reshape(v.shape)
        out[np.isnan(v)] = np.nan
        return _like(out, land.template, f"share_of_{'_'.join(f'{c:g}' for c in self.classes)}")

    def __repr__(self) -> str:
        return f"share_of({self.classes!r}, window={self.window.shape[0]})"


def with_dynamic(dynamic, drivers, landscape):
    """The drivers with the dynamic variables computed from ``landscape`` (they replace drivers of the same name)."""
    if not dynamic:
        return drivers
    from ._woe import _drivers

    maps: dict = dict(_drivers(drivers)) if drivers is not None else {}
    for name, piece in dynamic.items():
        if not callable(piece):
            raise TypeError(f"the dynamic variable {name!r} is not a function of the map")
        maps[str(name)] = read_map(piece(landscape))
    return maps


def current_map(values: np.ndarray, template):
    """The simulated map as a DataArray on the template's grid (or the array itself)."""
    if template is None:
        return values
    return template.copy(data=values)
