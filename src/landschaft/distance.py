# SPDX-License-Identifier: GPL-3.0-or-later
"""Euclidean Distance Transform (EDT) & Edge Distance Analysis.

Reimplements GuidosToolbox (GTB) Euclidean Distance tool:
- Computes distance from each foreground pixel to the nearest edge/boundary.
- Computes distance from each background pixel to the nearest forest edge.
- Signed distance map (positive inside foreground, negative inside background).
- Byte classified distance map (10..129 for background, 130 boundary, 131..250 foreground).
- Average and maximum distance metrics.

References
----------
Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object analysis.
    European Journal of Remote Sensing, 50(1), 352-361.
Meijster, A., Roerdink, J. B., & Hesselink, W. H. (2000). A general algorithm for
    computing distance transforms in linear time. Mathematical Morphology and its
    applications to image and signal processing, 331-340.
"""

from __future__ import annotations

from ._threads import threaded

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from . import _core
from .io import DIST_COLORMAP
from .validation import validate_mspa_input

__all__ = [
    "euclidean_distance",
    "EuclideanDistanceResult",
]


def _get_resolution(data: Any, pixel_size: float | tuple[float, float] | None = None) -> float | None:
    """Return pixel dimension in linear units if available."""
    if pixel_size is not None:
        if isinstance(pixel_size, (int, float)):
            return float(pixel_size)
        return float(pixel_size[0])
    if isinstance(data, xr.DataArray) and hasattr(data, "rio"):
        try:
            res = data.rio.resolution()
            if res is not None and abs(res[0]) >= 0.1:
                return float(abs(res[0]))
        except Exception:
            pass
    return None


def _wrap_as_dataarray(
    arr: np.ndarray,
    reference: Any,
    name: str,
    attrs: dict[str, Any] | None = None,
) -> xr.DataArray | np.ndarray:
    if isinstance(reference, xr.DataArray):
        da = reference.copy(data=arr)
        da.name = name
        if attrs:
            da.attrs.update(attrs)
        return da
    return arr


@dataclass
class EuclideanDistanceResult:
    """Result of :func:`euclidean_distance`.

    Attributes
    ----------
    signed_distance
        Distance of every pixel in pixels: positive in the foreground, negative
        in the background, NaN on missing pixels.
    byte_map
        GTB's ``uint8`` display map of the distances (see :func:`dist_colormap`).
    mean_fg_distance, max_fg_distance
        Mean and maximum distance of the foreground pixels, in pixels.
    mean_bg_distance, max_bg_distance
        The same for the background (absolute values).
    fg_pixel_count, bg_pixel_count
        Number of foreground and background pixels.
    stats
        The same statistics as a table, in pixels and in map units.
    pixel_size
        Pixel size used for the map units (``None``: unknown).

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> print(ls.euclidean_distance(forest, pixel_size=25).summary())  # doctest: +ELLIPSIS
    ================================================================
    GuidosToolbox (GTB) - Euclidean Distance Transform
    ...
    """

    signed_distance: xr.DataArray | np.ndarray
    byte_map: xr.DataArray | np.ndarray
    mean_fg_distance: float
    max_fg_distance: float
    mean_bg_distance: float
    max_bg_distance: float
    fg_pixel_count: int
    bg_pixel_count: int
    stats: pd.DataFrame
    pixel_size: float | None = None

    def summary(self) -> str:
        """Text summary of the distance statistics.

        Returns
        -------
        str
        """
        buf = io.StringIO()
        buf.write("=" * 64 + "\n")
        buf.write("GuidosToolbox (GTB) - Euclidean Distance Transform\n")
        buf.write("=" * 64 + "\n")
        buf.write(self.stats.to_string(index=False))
        buf.write("\n" + "=" * 64 + "\n")
        return buf.getvalue()


@threaded
def euclidean_distance(
    data: xr.DataArray | np.ndarray,
    *,
    pixel_size: float | tuple[float, float] | None = None,
    threads: int = 0,
) -> EuclideanDistanceResult:
    """Euclidean distance to the habitat edge, inside and outside the habitat (GTB's ``batch_eucldist``).

    Every foreground pixel gets its straight-line distance to the nearest
    background pixel (positive), every background pixel its distance to the
    nearest foreground (negative). The border pixels are replicated so that the
    map edge does not act as an edge.

    Parameters
    ----------
    data
        Map coded 0 missing, 1 background, 2 foreground (DataArray, array or
        path).
    pixel_size
        Pixel size (e.g. in metres) for the statistics in map units; read from a
        georeferenced DataArray when omitted.
    threads, n_threads
        Threads (0 or ``None``: all cores but one).

    Returns
    -------
    EuclideanDistanceResult

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> distance = ls.euclidean_distance(forest, pixel_size=25)
    >>> round(distance.mean_fg_distance, 2), round(distance.max_bg_distance, 2)
    (6.01, 33.54)
    >>> distance.stats[["class", "pixel_count", "mean_distance_units"]]
                     class  pixel_count  mean_distance_units
    0  Foreground (Forest)       428490               150.37
    1  Background (Matrix)       571240               125.76
    """
    validate_mspa_input(data)
    arr = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)
    raw = _core.compute_euclidean_distance(arr, threads)

    signed_dist = raw["signed_distance"]
    byte_map = raw["byte_map"]
    st = raw["stats"]

    px_res = _get_resolution(data, pixel_size)

    stats_rows = [
        {
            "class": "Foreground (Forest)",
            "pixel_count": st["fg_pixel_count"],
            "mean_distance_px": round(st["mean_fg_distance"], 2),
            "max_distance_px": round(st["max_fg_distance"], 2),
        },
        {
            "class": "Background (Matrix)",
            "pixel_count": st["bg_pixel_count"],
            "mean_distance_px": round(st["mean_bg_distance"], 2),
            "max_distance_px": round(st["max_bg_distance"], 2),
        },
    ]

    if px_res is not None:
        stats_rows[0]["mean_distance_units"] = round(st["mean_fg_distance"] * px_res, 2)
        stats_rows[0]["max_distance_units"] = round(st["max_fg_distance"] * px_res, 2)
        stats_rows[1]["mean_distance_units"] = round(st["mean_bg_distance"] * px_res, 2)
        stats_rows[1]["max_distance_units"] = round(st["max_bg_distance"] * px_res, 2)

    stats_df = pd.DataFrame(stats_rows)

    dist_attrs = {
        "landschaft_product": "EUCLIDEAN_DISTANCE_FLOAT",
        "mean_fg_dist": st["mean_fg_distance"],
        "max_fg_dist": st["max_fg_distance"],
        "mean_bg_dist": st["mean_bg_distance"],
        "max_bg_dist": st["max_bg_distance"],
    }
    byte_attrs = {
        "landschaft_product": "EUCLIDEAN_DISTANCE",
        "colormap": "dist",
        "mean_fg_dist": st["mean_fg_distance"],
        "max_fg_dist": st["max_fg_distance"],
    }

    return EuclideanDistanceResult(
        signed_distance=_wrap_as_dataarray(signed_dist, data, "euclidean_distance_float", dist_attrs),
        byte_map=_wrap_as_dataarray(byte_map, data, "euclidean_distance", byte_attrs),
        mean_fg_distance=float(st["mean_fg_distance"]),
        max_fg_distance=float(st["max_fg_distance"]),
        mean_bg_distance=float(st["mean_bg_distance"]),
        max_bg_distance=float(st["max_bg_distance"]),
        fg_pixel_count=int(st["fg_pixel_count"]),
        bg_pixel_count=int(st["bg_pixel_count"]),
        stats=stats_df,
        pixel_size=px_res,
    )
