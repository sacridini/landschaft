# SPDX-License-Identifier: GPL-3.0-or-later
"""Morphological Spatial Pattern Analysis (MSPA).

Soille, P. & Vogt, P. (2009). Morphological segmentation of binary patterns.
Pattern Recognition Letters 30(4), 456-459. doi:10.1016/j.patrec.2008.10.015
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import _core
from .validation import InvalidInputError, validate_mspa_input

__all__ = ["mspa", "mspa_stats", "MSPA_CLASSES"]

#: Byte value -> class name of an MSPA map. With ``internal=True`` pixels of
#: internal features (inside core-openings) carry an offset of 100.
MSPA_CLASSES: dict[int, str] = {
    0: "Background",
    1: "Branch",
    3: "Edge",
    5: "Perforation",
    9: "Islet",
    17: "Core",
    33: "Bridge",
    35: "Bridge in Edge",
    37: "Bridge in Perforation",
    65: "Loop",
    67: "Loop in Edge",
    69: "Loop in Perforation",
    100: "Core-Opening",
    129: "Missing",
    220: "Border-Opening",
}
for _v in [v for v in MSPA_CLASSES if v not in (0, 100, 129, 220)]:
    MSPA_CLASSES[_v + 100] = MSPA_CLASSES[_v] + " (internal)"
MSPA_CLASSES = dict(sorted(MSPA_CLASSES.items()))

# Class groups used by GuidosToolbox's MSPA statistics.
_FOREGROUND_GROUPS = {
    "Core": (17, 117),
    "Islet": (9, 109),
    "Perforation": (5, 105),
    "Edge": (3, 103),
    "Loop": (65, 67, 69, 165, 167, 169),
    "Bridge": (33, 35, 37, 133, 135, 137),
    "Branch": (1, 101),
}
_OTHER_GROUPS = {
    "Background": (0,),
    "Core-Opening": (100,),
    "Border-Opening": (220,),
    "Missing": (129,),
}


def _check_params(connectivity, edge_width) -> None:
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
    if int(edge_width) != edge_width or not 1 <= edge_width <= 255:
        raise ValueError(f"edge_width must be an integer in [1, 255], got {edge_width!r}")


def _mspa_2d(arr: np.ndarray, connectivity: int, edge_width: int, internal: bool, n_threads: int) -> np.ndarray:
    try:
        return _core.mspa(np.ascontiguousarray(arr), connectivity, int(edge_width), internal, n_threads)
    except ValueError as err:  # invalid pixel values detected by the C++ core
        raise InvalidInputError(str(err)) from None


def _mspa_nd(arr: np.ndarray, **kw) -> np.ndarray:
    """Apply MSPA to every 2-D slice over the last two axes."""
    if arr.ndim == 2:
        return _mspa_2d(arr, **kw)
    out = np.empty(arr.shape, dtype=np.uint8)
    for idx in np.ndindex(*arr.shape[:-2]):
        out[idx] = _mspa_2d(arr[idx], **kw)
    return out


def _spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    if "y" in da.dims and "x" in da.dims:
        return "y", "x"
    return da.dims[-2], da.dims[-1]


def mspa(
    data,
    *,
    connectivity: int = 8,
    edge_width: int = 1,
    transition: bool = True,
    internal: bool = True,
    n_threads: int | None = None,
    validate: bool = True,
):
    """Morphological Spatial Pattern Analysis, identical to GuidosToolbox's MSPA.

    Parameters
    ----------
    data
        Map coded ``0`` = missing, ``1`` = background, ``2`` = foreground
        (``uint8``). Accepts a 2-D numpy array, a dask array or an
        ``xarray.DataArray``. Arrays with more dimensions (e.g.
        ``(time, y, x)``) are processed slice by slice over the last two
        (or ``y``/``x``) dimensions; with dask the slices run in parallel.
    connectivity
        Foreground connectivity, 8 (default) or 4 (GTB "FGconn").
    edge_width
        Effective edge width in pixels, 1-255 (GTB "EdgeWidth").
    transition
        Whether loops/bridges crossing edges or perforations are shown as
        transitions (GTB "Transition"). As in GuidosToolbox this only changes
        the colour table written by :func:`write_raster`, not the values.
    internal
        Distinguish internal features with a +100 offset and label
        core-openings (100) and border-openings (220) (GTB "Intext").
    n_threads
        Threads used inside one map; ``1`` runs sequentially. ``None`` uses
        all cores, except for dask-backed input where it defaults to 1
        because dask already runs the slices in parallel.
    validate
        Check the input like GuidosToolbox before processing.

    Returns
    -------
    Same container type as ``data`` with the ``uint8`` MSPA classes; see
    :data:`MSPA_CLASSES`. DataArrays keep coordinates and georeferencing and
    record the parameters in ``attrs``.
    """
    _check_params(connectivity, edge_width)
    lazy = hasattr(getattr(data, "data", None), "dask") or hasattr(data, "dask")
    if n_threads is None:
        n_threads = 1 if lazy else 0  # 0 = all cores
    kw = dict(connectivity=int(connectivity), edge_width=int(edge_width), internal=bool(internal),
              n_threads=int(n_threads))

    if isinstance(data, xr.DataArray):
        if validate:
            validate_mspa_input(data, spatial_only=False)
        ydim, xdim = _spatial_dims(data)
        src = data
        if src.chunks is not None:
            src = src.chunk({ydim: -1, xdim: -1})
        result = xr.apply_ufunc(
            _mspa_nd,
            src,
            input_core_dims=[[ydim, xdim]],
            output_core_dims=[[ydim, xdim]],
            kwargs=kw,
            dask="parallelized",
            output_dtypes=[np.uint8],
            keep_attrs=False,
        )
        result = result.transpose(*data.dims)
        result.name = "mspa"
        result.attrs = {
            "long_name": "MSPA classes",
            "guidospy_product": "MSPA",
            "mspa_connectivity": kw["connectivity"],
            "mspa_edge_width": kw["edge_width"],
            "mspa_transition": int(bool(transition)),
            "mspa_internal": int(kw["internal"]),
        }
        if "spatial_ref" in data.coords:
            result = result.assign_coords(spatial_ref=data.coords["spatial_ref"])
        result.encoding = {}
        return result

    if hasattr(data, "dask"):  # plain dask array
        import dask.array as dsa

        if validate:
            validate_mspa_input(data, spatial_only=False)
        src = data.rechunk({data.ndim - 2: -1, data.ndim - 1: -1})
        return dsa.map_blocks(_mspa_nd, src, dtype=np.uint8, **kw)

    arr = np.asarray(data)
    if validate:
        validate_mspa_input(arr, spatial_only=False)
    return _mspa_nd(arr, **kw)


def mspa_stats(result) -> pd.DataFrame:
    """Pixel statistics of an MSPA map, grouped as in GuidosToolbox.

    Returns a DataFrame indexed by class group with the number of pixels,
    the percentage of the data area (all non-missing pixels) and, for
    foreground classes, the percentage of the foreground area.
    """
    values = np.asarray(result.values if isinstance(result, xr.DataArray) else result)
    counts = np.bincount(values.ravel(), minlength=256)
    data_area = int(counts.sum() - counts[129])
    groups = {**_FOREGROUND_GROUPS, **_OTHER_GROUPS}
    fg_area = int(sum(counts[list(v)].sum() for v in _FOREGROUND_GROUPS.values()))
    rows = []
    for name, vals in groups.items():
        n = int(counts[list(vals)].sum())
        rows.append(
            {
                "class": name,
                "pixels": n,
                "pct_data": 100.0 * n / data_area if data_area and name != "Missing" else np.nan,
                "pct_foreground": 100.0 * n / fg_area if fg_area and name in _FOREGROUND_GROUPS else np.nan,
            }
        )
    df = pd.DataFrame(rows).set_index("class")
    df.attrs.update(data_area=data_area, foreground_area=fg_area,
                    internal_foreground_area=fg_area + int(counts[100] + counts[220]))
    return df
