# SPDX-License-Identifier: GPL-3.0-or-later
"""Contagion (Pff) and Clustering (FAC) Analysis.

Reimplements GuidosToolbox (GTB) Moving Window Adjacency tools:
- :func:`contagion_pff`: Moving-window Contagion metric (Pff / Spatcon rule 77).
- :func:`clustering_fac`: Moving-window Foreground Area Clustering metric (FAC / Spatcon rule 76).

Both tools count edge transitions (horizontal and vertical pixel adjacencies) inside
a square moving window:
- **Pff (Contagion)**: Proportion of foreground pixel edges that are self-joins
  (foreground-foreground), measuring the degree of aggregation of the foreground class.
- **FAC (Clustering)**: Proportion of all non-missing pixel edges that are foreground
  self-joins, measuring overall foreground edge clustering within the landscape context.

References
----------
Riitters, K. H., et al. (1995). A factor analysis of landscape pattern and structure metrics.
    Landscape Ecology, 10(1), 23-39.
Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object analysis.
    European Journal of Remote Sensing, 50(1), 352-361.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from . import _core
from .validation import validate_mspa_input

__all__ = [
    "contagion_pff",
    "clustering_fac",
]


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


def contagion_pff(
    data: xr.DataArray | np.ndarray,
    window_size: int = 7,
    *,
    as_byte: bool = True,
    apply_gtb_mask: bool = True,
    threads: int = 0,
) -> xr.DataArray | np.ndarray:
    """Compute moving-window Contagion (Pff) matching GuidosToolbox batch_pff.

    Contagion (Pff) measures the conditional probability that a pixel adjacent
    to a foreground pixel is also foreground:

        Pff = E_ff / E_f*

    where E_ff is the count of foreground-foreground joins and E_f* is the count
    of joins involving foreground and any non-missing pixel.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        2D binary raster coded 0=missing, 1=background, 2=foreground.
    window_size : int, default 7
        Odd integer >= 3 (e.g. 7, 13, 27).
    as_byte : bool, default True
        If True, returns uint8 with GTB byte scaling:
        - 0..100: Contagion percentage
        - 101: Background mask
        - 102: Missing mask
        If False, returns float32 with values in [0.0, 1.0] and NaN for background/missing.
    apply_gtb_mask : bool, default True
        If True, applies background (101) and missing (102) masks.
    threads : int, default 0
        Number of OpenMP threads to use (0 = auto).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Contagion raster (uint8 or float32).
    """
    validate_mspa_input(data)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    arr = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)

    if as_byte:
        res = _core.compute_contagion_pff(arr, window_size, apply_gtb_mask, threads)
    else:
        res = _core.compute_contagion_pff_float(arr, window_size, apply_gtb_mask, threads)

    attrs = {
        "guidospy_product": "CONTAGION",
        "metric": "Pff",
        "window_size": window_size,
        "colormap": "entropy",
        "TIFFTAG_IMAGEDESCRIPTION": "GTB_CON, https://forest.jrc.ec.europa.eu/activities/lpa/gtb/",
    }
    return _wrap_as_dataarray(res, data, "contagion_pff", attrs)


def clustering_fac(
    data: xr.DataArray | np.ndarray,
    window_size: int = 7,
    *,
    as_byte: bool = True,
    apply_gtb_mask: bool = True,
    threads: int = 0,
) -> xr.DataArray | np.ndarray:
    """Compute moving-window Foreground Area Clustering (FAC) matching GuidosToolbox batch_fac.

    Foreground Area Clustering (FAC) measures the proportion of all non-missing
    pixel edges in the moving window that are foreground self-joins:

        FAC = E_ff / E_total

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        2D binary raster coded 0=missing, 1=background, 2=foreground.
    window_size : int, default 7
        Odd integer >= 3 (e.g. 7, 13, 27).
    as_byte : bool, default True
        If True, returns uint8 with GTB byte scaling:
        - 0..100: Clustering percentage
        - 101: Background mask
        - 102: Missing mask
        If False, returns float32 with values in [0.0, 1.0] and NaN for background/missing.
    apply_gtb_mask : bool, default True
        If True, applies background (101) and missing (102) masks.
    threads : int, default 0
        Number of OpenMP threads to use (0 = auto).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Clustering raster (uint8 or float32).
    """
    validate_mspa_input(data)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    arr = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)

    if as_byte:
        res = _core.compute_clustering_fac(arr, window_size, apply_gtb_mask, threads)
    else:
        res = _core.compute_clustering_fac_float(arr, window_size, apply_gtb_mask, threads)

    attrs = {
        "guidospy_product": "CLUSTERING",
        "metric": "FAC",
        "window_size": window_size,
        "colormap": "entropy",
        "TIFFTAG_IMAGEDESCRIPTION": "GTB_FAC, https://forest.jrc.ec.europa.eu/activities/lpa/gtb/",
    }
    return _wrap_as_dataarray(res, data, "clustering_fac", attrs)
