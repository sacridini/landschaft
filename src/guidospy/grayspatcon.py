# SPDX-License-Identifier: GPL-3.0-or-later
"""Continuous Grayscale Moving-Window Spatial Convolution (GraySpatCon).

This module implements moving-window spatial metrics over continuous/grayscale rasters:
- Mean / Average
- Median
- Majority / Mode
- Minimum
- Maximum
- Range
- Standard Deviation
- Shannon Diversity
- Shannon Evenness
- Diagonal Contagion

The core computation is implemented in C++ using an O(1) 101-bin sliding histogram
and parallelized across threads using OpenMP.

References
----------
Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object analysis.
    European Journal of Remote Sensing, 50(1), 352-361. doi:10.1080/22797254.2017.1330650
Shannon, C. E. (1948). A mathematical theory of communication.
    The Bell System Technical Journal, 27(3), 379-423.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from . import _core
from .io import load_raster

__all__ = [
    "gray_spatial_convolution",
]

_SUPPORTED_METRICS = {
    "mean": "mean",
    "average": "mean",
    "median": "median",
    "majority": "majority",
    "mode": "majority",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "range": "range",
    "stddev": "stddev",
    "std": "stddev",
    "standard_deviation": "stddev",
    "shannon_diversity": "shannon_diversity",
    "shannon_evenness": "shannon_evenness",
    "diagonal_contagion": "diagonal_contagion",
}


def gray_spatial_convolution(
    data: xr.DataArray | np.ndarray | str | Path,
    metric: str = "mean",
    window_size: int = 7,
    mask_missing: bool = True,
    n_threads: int = 0,
) -> xr.DataArray | np.ndarray:
    """Compute moving-window spatial convolution metric over a grayscale raster.

    Parameters
    ----------
    data : xarray.DataArray, numpy.ndarray, str, or Path
        Continuous grayscale raster (byte values 0..100, 255 = missing/nodata).
    metric : str, default "mean"
        Spatial metric to compute within each moving window.
        Supported metrics:
        - "mean" (or "average"): window mean
        - "median": window median
        - "majority" (or "mode"): most frequent value in window
        - "min": window minimum
        - "max": window maximum
        - "range": max - min
        - "stddev" (or "std"): standard deviation
        - "shannon_diversity": Shannon entropy index H' = -sum(p * ln(p))
        - "shannon_evenness": Pielou's evenness J' = H' / ln(S)
        - "diagonal_contagion": contagion index over the gray spectrum
    window_size : int, default 7
        Side length of the square sliding window in pixels (must be odd, >= 3).
    mask_missing : bool, default True
        If True, pixels that are missing (value 255) remain 255 (or NaN) in the output.
    n_threads : int, default 0
        Number of OpenMP worker threads (0 = all available).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Floating-point (float32) raster of the computed spatial convolution metric.
    """
    key = metric.lower().strip()
    if key not in _SUPPORTED_METRICS:
        valid_keys = sorted(set(_SUPPORTED_METRICS.keys()))
        raise ValueError(f"Unknown metric '{metric}'. Supported metrics: {valid_keys}")
    canonical_metric = _SUPPORTED_METRICS[key]

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")

    is_xr = False
    xr_template = None

    if isinstance(data, (str, Path)):
        data = load_raster(data)
        is_xr = True
        xr_template = data
    elif isinstance(data, xr.DataArray):
        is_xr = True
        xr_template = data

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Input raster must be 2-dimensional")

    u8_arr = np.ascontiguousarray(arr, dtype=np.uint8)
    out_arr = _core.compute_gray_convolution(
        u8_arr, canonical_metric, window_size, mask_missing, n_threads
    )

    if is_xr and xr_template is not None:
        attrs = {
            **xr_template.attrs,
            "guidospy_product": "GRAY_SPATIAL_CONVOLUTION",
            "spatial_metric": canonical_metric,
            "window_size": window_size,
            "mask_missing": mask_missing,
        }
        return xr.DataArray(
            out_arr,
            coords=xr_template.coords,
            dims=xr_template.dims,
            attrs=attrs,
            name=f"gray_conv_{canonical_metric}",
        )

    return out_arr
