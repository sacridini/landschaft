# SPDX-License-Identifier: GPL-3.0-or-later
"""Forest Area Density (FAD) & Multiscale Fragmentation Analysis.

Reimplements GuidosToolbox (GTB) FAD and FAD-Multiscale:
- :func:`fad`: Single-scale local density and fragmentation classification.
- :func:`fad_multiscale`: Multi-scale observation analysis across 5 scales
  (7x7, 13x13, 27x27, 81x81, 243x243) and composite average map.

References
----------
Vogt, P., et al. (2007). Mapping landscape corridors. Ecological Indicators, 7(2), 481-488.
Riitters, K., et al. (2002). Fragmentation of continental United States forests.
    Ecosystems, 5(8), 815-822.
"""

from __future__ import annotations

from ._threads import threaded

import io
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import _core
from .io import FAD_COLORMAP
from .validation import validate_fad_input, validate_mspa_input

__all__ = [
    "fad",
    "fad_multiscale",
    "FADResult",
    "FADMultiscaleResult",
    "FAD_CLASSES",
]

FAD_CLASSES: dict[int, str] = {
    1: "Rare (<10%)",
    2: "Patchy (10-40%)",
    3: "Transitional (40-60%)",
    4: "Dominant (60-90%)",
    5: "Interior (90-100%)",
    6: "Intact (100%)",
    101: "Background",
    102: "Missing",
    105: "Special background (input 3)",
    106: "Special background, non-fragmenting (input 4)",
}


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
class FADResult:
    """Result of single-scale FAD analysis (:func:`fad`).

    Examples
    --------
    >>> import landschaft as ls
    >>> result = ls.fad(ls.load_raster(ls.example_data("input")), window_size=13)
    >>> result.window_size, result.density.shape
    (13, (1000, 1000))
    """

    density: xr.DataArray | np.ndarray
    raw_density: xr.DataArray | np.ndarray
    classes: xr.DataArray | np.ndarray
    window_size: int
    stats: pd.DataFrame

    def summary(self) -> str:
        """Formatted text summary of fragmentation statistics."""
        return self.stats.to_string(index=False)


@dataclass
class FADMultiscaleResult:
    """Result of multiscale FAD analysis (:func:`fad_multiscale`).

    Examples
    --------
    >>> import landschaft as ls
    >>> result = ls.fad_multiscale(ls.load_raster(ls.example_data("input")), window_sizes=(7, 27))
    >>> result.window_sizes
    [7, 27]
    """

    multiscale_map: xr.DataArray | np.ndarray
    classes: xr.DataArray | np.ndarray
    single_scales: dict[int, xr.DataArray | np.ndarray]
    raw_single_scales: dict[int, xr.DataArray | np.ndarray]
    window_sizes: list[int]
    stats: pd.DataFrame

    def summary(self) -> str:
        """Formatted text summary of multiscale fragmentation statistics."""
        buf = io.StringIO()
        buf.write("=" * 72 + "\n")
        buf.write("GuidosToolbox (GTB) - Multiscale Forest Area Density (FAD-MS)\n")
        buf.write("=" * 72 + "\n")
        buf.write(self.stats.to_string(index=False))
        buf.write("\n" + "=" * 72 + "\n")
        return buf.getvalue()


@threaded
def fad(
    data: xr.DataArray | np.ndarray,
    window_size: int = 7,
    *,
    threads: int = 0,
) -> FADResult:
    """Compute Forest Area Density (FAD) for a single window size.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        2D raster coded 0=missing, 1=background, 2=foreground.
    window_size : int, default 7
        Moving window size (must be an odd integer >= 3).
    threads : int, default 0
        Threads of the compiled code (also as ``n_threads``); 0: all cores but one.

    Returns
    -------
    FADResult
        Result object containing density map, 6-class map, and statistics table.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> result = ls.fad(forest, window_size=27)
    >>> float(result.stats["average_fad"].iloc[0])
    73.54
    >>> _ = ls.write_raster(result.density, "fad27.tif")   # with the colours of the six classes
    """
    validate_mspa_input(data)
    arr = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    res = _core.compute_fad(arr, window_size, threads)
    c_density, c_raw, c_classes = res["density"], res["raw"], res["classes"]

    fg_total = res["foreground"]
    if fg_total > 0:
        rare, patchy, transitional, dominant, interior, intact = (n / fg_total * 100.0 for n in res["class_counts"])
        avg_fad = res["density_sum"] / fg_total
    else:
        intact = interior = dominant = transitional = patchy = rare = avg_fad = 0.0

    stats_df = pd.DataFrame(
        [
            {
                "scale": window_size,
                "intact_%": round(intact, 2),
                "interior_%": round(interior, 2),
                "dominant_%": round(dominant, 2),
                "transitional_%": round(transitional, 2),
                "patchy_%": round(patchy, 2),
                "rare_%": round(rare, 2),
                "average_fad": round(avg_fad, 2),
            }
        ]
    )

    attrs = {
        "landschaft_product": "FAD",
        "window_size": window_size,
        "average_fad": avg_fad,
        "colormap": "fad",
    }

    return FADResult(
        density=_wrap_as_dataarray(c_density, data, f"fad_{window_size}", attrs),
        raw_density=_wrap_as_dataarray(c_raw, data, f"fad_raw_{window_size}"),
        classes=_wrap_as_dataarray(c_classes, data, f"fad_classes_{window_size}"),
        window_size=window_size,
        stats=stats_df,
    )


@threaded
def fad_multiscale(
    data: xr.DataArray | np.ndarray,
    window_sizes: Sequence[int] = (7, 13, 27, 81, 243),
    *,
    threads: int = 0,
) -> FADMultiscaleResult:
    """Compute Multiscale Forest Area Density (FAD-MS) matching GuidosToolbox.

    Evaluates local forest density across 5 observation scales (or user-defined
    window sizes) using high-performance C++ box integrals, classifies pixels
    into 6 fragmentation categories, and generates the multi-scale composite map.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        2D raster coded 0=missing, 1=background, 2=foreground, and optionally
        3=special background and 4=non-fragmenting special background (treated
        as missing when measuring density), as in GTB. The maps code them 105
        and 106 (and background 101, missing 102).
    window_sizes : sequence of int, default (7, 13, 27, 81, 243)
        List of odd window sizes.
    threads : int, default 0
        Threads of the compiled code (also as ``n_threads``); 0: all cores but one.

    Returns
    -------
    FADMultiscaleResult
        Result containing composite map, single scale maps, and statistics table.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> result = ls.fad_multiscale(forest)
    >>> list(result.stats["window_size"])
    [7, 13, 27, 81, 243, 'Average']
    >>> sorted(result.single_scales)
    [7, 13, 27, 81, 243]
    """
    validate_fad_input(data)
    arr = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)
    ws_list = [int(w) for w in window_sizes]
    for w in ws_list:
        if w < 3 or w % 2 == 0:
            raise ValueError(f"window size {w} must be an odd integer >= 3")

    raw_res = _core.compute_fad_multiscale(arr, ws_list, threads)

    single_scales: dict[int, xr.DataArray | np.ndarray] = {}
    raw_single_scales: dict[int, xr.DataArray | np.ndarray] = {}

    for w, sm, rm in zip(ws_list, raw_res["single_scale_maps"], raw_res["raw_densities"]):
        attrs = {"landschaft_product": "FAD", "window_size": w, "colormap": "fad"}
        single_scales[w] = _wrap_as_dataarray(sm, data, f"fad_{w}", attrs)
        raw_single_scales[w] = _wrap_as_dataarray(rm, data, f"fad_raw_{w}")

    ms_map = raw_res["multiscale_map"]
    ms_attrs = {"landschaft_product": "FAD_MULTISCALE", "colormap": "fad"}
    wrapped_ms = _wrap_as_dataarray(ms_map, data, "fad_multiscale", ms_attrs)
    classes = _wrap_as_dataarray(raw_res["classes"], data, "fad_classes")

    # GTB's AVCON: average FAD scaled to the reporting unit (non-missing area)
    avcon_factor = int((arr == 2).sum()) / int((arr != 0).sum())
    rows = []
    for d in raw_res["stats"]:
        scale_label = f"Scale {d['scale']}" if d["scale"] > 0 else "MultiScale"
        rows.append(
            {
                "observation_scale": scale_label,
                "window_size": d["scale"] if d["scale"] > 0 else "Average",
                "intact_%": round(d["intact_pct"], 2),
                "interior_%": round(d["interior_pct"], 2),
                "dominant_%": round(d["dominant_pct"], 2),
                "transitional_%": round(d["transitional_pct"], 2),
                "patchy_%": round(d["patchy_pct"], 2),
                "rare_%": round(d["rare_pct"], 2),
                "average_fad": round(d["average_fad"], 2),
                "avcon": round(d["average_fad"] * avcon_factor, 2),
            }
        )
    stats_df = pd.DataFrame(rows)

    return FADMultiscaleResult(
        multiscale_map=wrapped_ms,
        classes=classes,
        single_scales=single_scales,
        raw_single_scales=raw_single_scales,
        window_sizes=ws_list,
        stats=stats_df,
    )
