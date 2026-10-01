# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial Entropy & Contagion Analysis.

Reimplements GuidosToolbox (GTB) Spatial Entropy tools:
- :func:`spatial_entropy`: Normalized joint directional entropy index of landscape patterns.
- :func:`spatial_entropy_map`: Moving-window local spatial entropy raster.

References
----------
Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object analysis.
    European Journal of Remote Sensing, 50(1), 352-361.
Riitters, K. H., et al. (1995). A factor analysis of landscape pattern and structure
    metrics. Landscape Ecology, 10(1), 23-39.
"""

from __future__ import annotations

from ._threads import threaded

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from . import _core
from .io import ENTROPY_COLORMAP
from .validation import validate_mspa_input
from ._save import Saveable

__all__ = [
    "spatial_entropy",
    "spatial_entropy_map",
    "SpatialEntropyResult",
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


@dataclass
class SpatialEntropyResult(Saveable):
    """Result of global spatial entropy analysis (:func:`spatial_entropy`)."""

    entropy_index: float
    histogram: np.ndarray

    def summary(self) -> str:
        """Formatted text summary of entropy."""
        buf = io.StringIO()
        buf.write("=" * 64 + "\n")
        buf.write("GuidosToolbox (GTB) - Spatial Joint Entropy\n")
        buf.write("=" * 64 + "\n")
        buf.write(f"Normalized Spatial Entropy Index: {self.entropy_index:.4f}%\n")
        buf.write("=" * 64 + "\n")
        return buf.getvalue()


@threaded
def spatial_entropy(
    data: xr.DataArray | np.ndarray,
    *,
    threads: int = 0,
) -> SpatialEntropyResult:
    """Compute global normalized spatial joint directional entropy index.

    Evaluates the frequency of transitions across 8 directions among foreground,
    background, and missing pixels, normalized to [0, 100]% matching GuidosToolbox
    entropy_mspainp.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        2D raster coded 0=missing, 1=background, 2=foreground.
    threads : int, default 0
        Threads of the compiled code (also as ``n_threads``); 0: all cores but one.

    Returns
    -------
    SpatialEntropyResult
        Result containing normalized entropy index and directional transition histogram.

    Examples
    --------
    >>> import landschaft as ls
    >>> result = ls.spatial_entropy(ls.load_raster(ls.example_data("input")))
    >>> round(result.entropy_index, 1)
    32.5
    """
    validate_mspa_input(data)
    arr = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)
    raw = _core.compute_spatial_entropy(arr, threads)

    return SpatialEntropyResult(
        entropy_index=float(raw["entropy_index"]),
        histogram=np.array(raw["histogram"], dtype=np.uint64),
    )


@threaded
def spatial_entropy_map(
    data: xr.DataArray | np.ndarray,
    window_size: int = 50,
    *,
    threads: int = 0,
) -> xr.DataArray | np.ndarray:
    """Compute moving-window spatial entropy map matching GuidosToolbox batch_ent.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        2D raster coded 0=missing, 1=background, 2=foreground.
    window_size : int, default 50
        Size of moving convolution window.
    threads : int, default 0
        Threads of the compiled code (also as ``n_threads``); 0: all cores but one.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Local spatial entropy raster in [0, 100]%.

    Examples
    --------
    >>> import landschaft as ls
    >>> local = ls.spatial_entropy_map(ls.load_raster(ls.example_data("input")), window_size=51)
    >>> local.shape
    (1000, 1000)
    """
    validate_mspa_input(data)
    arr = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)
    if window_size < 3:
        raise ValueError("window_size must be >= 3")

    raw_map = _core.compute_spatial_entropy_map(arr, window_size, threads)
    attrs = {
        "landschaft_product": "SPATIAL_ENTROPY",
        "window_size": window_size,
        "colormap": "entropy",
    }
    return _wrap_as_dataarray(raw_map, data, "spatial_entropy", attrs)
