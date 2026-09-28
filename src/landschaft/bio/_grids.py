# SPDX-License-Identifier: GPL-3.0-or-later
"""Hexagonal and H3 grids for BaseData groups."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

_SQRT3 = math.sqrt(3.0)


def hex_centres(x, y, size: float, orientation: str = "pointy", origin: Sequence[float] = (0.0, 0.0)):
    """Centres of the hexagons (circumradius ``size``) containing each point.

    Parameters
    ----------
    x, y : array-like
        Point coordinates (projected units).
    size : float
        Distance from a hexagon's centre to its vertices.
    orientation : {"pointy", "flat"}
        Pointy-topped hexagons (rows) or flat-topped (columns).
    origin : (float, float)
        Centre of the hexagon at axial coordinates (0, 0).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The centre coordinates.

    Examples
    --------
    >>> import landschaft as ls
    >>> cx, cy = ls.bio.hex_centres([0.1, 1.6], [0.0, 0.1], 1.0)
    >>> [round(float(v), 4) for v in cx], [round(float(v), 4) for v in cy]
    ([0.0, 1.7321], [0.0, 0.0])
    """
    if orientation not in ("pointy", "flat"):
        raise ValueError("orientation must be 'pointy' or 'flat'")
    px = (np.asarray(x, dtype=np.float64) - origin[0]) / size
    py = (np.asarray(y, dtype=np.float64) - origin[1]) / size
    if orientation == "pointy":
        q = _SQRT3 / 3 * px - py / 3
        r = 2 / 3 * py
    else:
        q = 2 / 3 * px
        r = -px / 3 + _SQRT3 / 3 * py
    # cube rounding
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    if orientation == "pointy":
        cx = size * _SQRT3 * (rq + rr / 2)
        cy = size * 1.5 * rr
    else:
        cx = size * 1.5 * rq
        cy = size * _SQRT3 * (rr + rq / 2)
    return np.round(cx + origin[0], 10), np.round(cy + origin[1], 10)


def hex_polygon(cx: float, cy: float, size: float, orientation: str = "pointy"):
    """The hexagon around a centre (shapely Polygon)."""
    from shapely.geometry import Polygon

    start = 30.0 if orientation == "pointy" else 0.0
    pts = [(cx + size * math.cos(math.radians(start + 60 * k)), cy + size * math.sin(math.radians(start + 60 * k)))
           for k in range(6)]
    return Polygon(pts)


def _require_h3():
    try:
        import h3
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("H3 grids need the 'h3' package (pip install h3)") from e
    return h3


def h3_cells(lat, lon, resolution: int) -> list[str]:
    """H3 cell of each point (``h3`` package).

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.bio.h3_cells([-33.86], [151.21], 5)
    ['85be0e37fffffff']
    """
    h3 = _require_h3()
    return [h3.latlng_to_cell(float(a), float(b), int(resolution)) for a, b in zip(lat, lon)]


def _grouped(data: pd.DataFrame, gx, gy, label_column, count_column, cell_ids=None):
    from ._basedata import BaseData

    df = pd.DataFrame({"_gx": gx, "_gy": gy})
    labels = [label_column] if isinstance(label_column, Hashable) and not isinstance(label_column, list) else list(
        label_column)
    for c in labels:
        df[c] = data[c].to_numpy()
    count = None
    if count_column is not None:
        df["_count"] = data[count_column].to_numpy()
        count = "_count"
    bd = BaseData.from_dataframe(df, ["_gx", "_gy"], labels if len(labels) > 1 else labels[0], count,
                                 cell_sizes=[0, 0])
    return bd
