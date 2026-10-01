# SPDX-License-Identifier: GPL-3.0-or-later
"""Morphological and Landscape Shape Analysis (Parcellation & Contortion).

This module reimplements GuidosToolbox (GTB) morphology tools:
- :func:`parcellation`: Landscape division and mesh size metrics (APS, AWAPS/effective
  mesh size, Division Index, and Logarithmic Parcellation Index).
- :func:`contortion`: Boundary tortuosity, perimeter inflection points, and shape
  compactness/elongation analysis for habitat components.

References
----------
Jaeger, J. A. (2000). Landscape division, splitting index, and effective mesh size:
    new measures of landscape fragmentation. Landscape ecology, 15(2), 115-130.
Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object analysis.
    European Journal of Remote Sensing, 50(1), 352-361. doi:10.1080/22797254.2017.1330650
"""

from __future__ import annotations

from ._threads import threaded

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from . import _core
from .io import _load_map
from ._save import Saveable

__all__ = [
    "parcellation",
    "contortion",
    "ParcellationResult",
]


@dataclass(frozen=True)
class ParcellationResult(Saveable):
    """Results from landscape parcellation analysis.

    Attributes
    ----------
    stats : pandas.DataFrame
        Class-level parcellation metrics (Count, Area, APS, AWAPS, DIVISION, PARC).
    global_stats : dict[str, Any]
        Landscape-level metrics across all valid classes.
    parcel_labels : xarray.DataArray or numpy.ndarray, optional
        Labeled parcels raster (if return_labels=True was requested).

    Examples
    --------
    >>> import landschaft as ls
    >>> result = ls.parcellation(ls.load_raster(ls.example_data("input")))
    >>> round(result.global_stats["division"], 3)
    0.689
    """

    stats: pd.DataFrame
    global_stats: dict[str, Any]
    parcel_labels: xr.DataArray | np.ndarray | None = None

    def summary(self) -> str:
        """Return a formatted string summary of parcellation metrics."""
        g = self.global_stats
        lines = [
            "Landscape Parcellation Summary",
            "==============================",
            f"Total Valid Area [pixels]: {g['data_area']:,}",
            f"Total Parcels (Patches)  : {g['total_parcels']:,}",
            f"Average Patch Size (APS) : {g['aps']:.2f} pixels",
            f"Effective Mesh Size(AWAPS): {g['awaps']:.2f} pixels",
            f"Landscape Division Index : {g['division']:.4f}",
            f"Parcellation Index (PARC): {g['parc']:.2f}%",
            "",
            "Class Breakdown:",
            "----------------",
            self.stats.to_string(index=False),
        ]
        return "\n".join(lines)


@threaded
def parcellation(
    data: xr.DataArray | np.ndarray | str | Path,
    connectivity: int = 8,
    return_labels: bool = False,
) -> ParcellationResult:
    """How finely the landscape is divided into parcels.

    Every connected region of one value (e.g. habitat and background, or the
    classes of a land-cover map) is a parcel. Per class and for the whole map:
    the number of parcels, the average parcel size (APS), the area-weighted
    average parcel size (AWAPS, the size of the parcel a random pixel falls in),
    the division index (probability that two random pixels are in different
    parcels) and the degree of parcellation (PARC).

    Parameters
    ----------
    data
        Class raster (array, DataArray or path; integer classes, 0 = NoData).
    connectivity
        Pixels of one parcel touch by a side (4) or also by a corner (8).
    return_labels
        Also return a map of the parcels.
    n_threads
        Threads (``None`` / 0: all cores but one).

    Returns
    -------
    ParcellationResult
        ``stats`` per class, ``global_stats`` for the map and, with
        ``return_labels``, ``parcel_labels``.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> result = ls.parcellation(forest)
    >>> list(result.stats.columns)
    ['Class', 'Count', 'Area[pixels]', 'APS', 'AWAPS', 'AWAPS/data', 'DIVISION', 'PARC[%]']
    >>> result.global_stats["total_parcels"]
    4020
    >>> labels = ls.parcellation(forest, return_labels=True).parcel_labels
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    is_xr = isinstance(data, xr.DataArray)
    xr_template = data if is_xr else None

    if isinstance(data, (str, Path)):
        data = _load_map(data)
        is_xr = True
        xr_template = data

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Input data must be a 2-D raster")

    u8_arr = np.ascontiguousarray(arr, dtype=np.uint8)
    class_stats_list, global_stats_dict, labels_arr = _core.compute_parcellation(
        u8_arr, connectivity, return_labels
    )

    rows = []
    for cs in class_stats_list:
        rows.append(
            {
                "Class": int(cs["class_id"]),
                "Count": int(cs["count"]),
                "Area[pixels]": int(cs["area"]),
                "APS": round(float(cs["aps"]), 2),
                "AWAPS": round(float(cs["awaps_rel"]), 2),
                "AWAPS/data": round(float(cs["awaps_data"]), 2),
                "DIVISION": round(float(cs["division"]), 4),
                "PARC[%]": round(float(cs["parc"]), 2),
            }
        )

    df = pd.DataFrame(rows)

    out_labels = None
    if return_labels and labels_arr is not None:
        if is_xr and xr_template is not None:
            out_labels = xr.DataArray(
                labels_arr,
                coords=xr_template.coords,
                dims=xr_template.dims,
                attrs={"long_name": "Parcellation Labels", "connectivity": connectivity},
            )
        else:
            out_labels = labels_arr

    return ParcellationResult(
        stats=df,
        global_stats=global_stats_dict,
        parcel_labels=out_labels,
    )


def contortion(
    data: xr.DataArray | np.ndarray | str | Path,
    min_area: int = 1,
    return_display: bool = False,
) -> xr.DataArray | np.ndarray:
    """How convoluted the boundary of every habitat object is.

    For each habitat object, a measure of how much its boundary bends relative to
    its size: compact objects score low, elongated or dendritic ones high.

    Parameters
    ----------
    data
        Habitat map (array, DataArray or path) coded 0 missing, 1 background,
        2 habitat.
    min_area
        Objects smaller than this (pixels), or without an interior (no 3 x 3
        core), are left out and coded 2.
    return_display
        Rescale the values to 3-255 for display (the byte scale of the colour
        table :func:`contortion_colormap`).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        0 missing, 1 background, 2 left-out objects, 3 and above the contortion
        of each object.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> values = ls.contortion(forest, min_area=100)
    >>> int(values.max())
    42
    """
    is_xr = isinstance(data, xr.DataArray)
    xr_template = data if is_xr else None

    if isinstance(data, (str, Path)):
        data = _load_map(data, validate="mspa")
        is_xr = True
        xr_template = data

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Input data must be a 2-D raster")

    u8_arr = np.ascontiguousarray(arr, dtype=np.uint8)
    cont_arr = _core.compute_contortion(u8_arr, int(min_area))

    if return_display:
        # Scale contortion values >= 3 into byte [3, 255]
        mask_cont = cont_arr >= 3
        out_display = np.copy(cont_arr).astype(np.uint8)
        if np.any(mask_cont):
            c_vals = cont_arr[mask_cont]
            c_min = np.min(c_vals)
            c_max = np.max(c_vals)
            if c_max > c_min:
                scaled = 3 + np.round(((c_vals - c_min) / (c_max - c_min)) * 252.0).astype(np.uint8)
            else:
                scaled = np.full_like(c_vals, 3, dtype=np.uint8)
            out_display[mask_cont] = scaled
        res_arr = out_display
    else:
        res_arr = cont_arr

    if is_xr and xr_template is not None:
        return xr.DataArray(
            res_arr,
            coords=xr_template.coords,
            dims=xr_template.dims,
            attrs={"long_name": "Contortion", "min_area": min_area},
        )
    return res_arr
