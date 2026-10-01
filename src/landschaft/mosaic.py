# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape Mosaic (LM) Tri-Polar Analysis.

Reimplements GuidosToolbox (GTB) Landscape Mosaic moving-window tool (Spatcon rule 7):
- :func:`landscape_mosaic`: Moving-window tri-polar land-cover mosaic classification
  (19 classes or 103 classes).
- :func:`landscape_mosaic_stats`: 103-class frequency distribution and barycentric statistics.

Landscape Mosaic models the local composition of three land-cover components
within a moving observation window:
- Component 1: Agriculture / Non-forest / Matrix 1
- Component 2: Forest / Natural / Matrix 2
- Component 3: Developed / Antropic / Matrix 3
(and 0 = Missing / Unclassified).

The tri-polar percentages (p_agr, p_for, p_dev where p_agr + p_for + p_dev = 100%)
are mapped into discrete cells of a barycentric triangle (103 classes), which are
subsequently recoded into 19 ecologically meaningful landscape mosaic classes.

References
----------
Riitters, K., Wickham, J., O'Neill, R., Jones, B., & Smith, E. (2000). Global-scale patterns
    of forest fragmentation. Conservation Ecology, 4(2), 3.
Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object analysis.
    European Journal of Remote Sensing, 50(1), 352-361.
"""

from __future__ import annotations

from ._threads import threaded

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from . import _core

__all__ = [
    "landscape_mosaic",
    "landscape_mosaic_stats",
    "LM19_CLASSES",
]

LM19_CLASSES: dict[int, str] = {
    0: "Missing",
    1: "Agriculture (>60%)",
    2: "Developed (>60%)",
    3: "Forest (>60%)",
    4: "Agriculture-Developed",
    5: "Agriculture-Forest",
    6: "Developed-Agriculture",
    7: "Developed-Agriculture-Forest",
    8: "Forest-Developed-Agriculture",
    9: "Forest-Agriculture",
    10: "Forest-Developed",
    11: "Forest-Agriculture-Developed",
    12: "Forest-Developed-Agriculture",
    13: "Mosaic Agriculture-Developed",
    14: "Mosaic Agriculture-Forest",
    15: "Mosaic Developed-Forest",
    16: "Mosaic Agriculture-Developed-Forest",
    17: "Pure Forest (100%)",
    18: "Pure Agriculture (100%)",
    19: "Pure Developed (100%)",
}

RECODE_103_TO_19: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 4, 4, 4, 4, 4, 46, 47, 48, 49, 50, 5, 5, 5, 5, 5, 56, 57, 58, 59,
    60, 6, 6, 6, 6, 6, 66, 67, 68, 69, 70, 7, 7, 7, 7, 7, 76, 77, 78, 79,
    80, 8, 8, 8, 8, 8, 86, 87, 88, 89, 90, 9, 9, 9, 9, 9, 96, 97, 98, 99,
    100, 10, 10, 10, 10, 105, 106, 107, 108, 109, 110, 11, 11, 11, 11, 115, 116, 117, 118, 119,
    120, 12, 12, 12, 12, 125, 126, 127, 128, 129, 130, 13, 13, 13, 13, 13, 136, 137, 138, 139,
    140, 14, 14, 14, 14, 14, 146, 147, 148, 149, 150, 15, 15, 15, 15, 15, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 17, 3, 3, 173, 174, 175, 176, 177, 178, 179,
    18, 1, 1, 183, 184, 185, 186, 187, 188, 189, 19, 2, 2, 193, 194, 195, 196, 197, 198, 199,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
)


def _validate_lm_input(data: Any) -> None:
    arr = data.values if isinstance(data, xr.DataArray) else np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"Input data must be a 2D array, got {arr.ndim}D")
    u = np.unique(arr)
    invalid = set(u) - {0, 1, 2, 3}
    if invalid:
        raise ValueError(
            f"Landscape Mosaic input must contain only classes 0 (missing), 1 (agriculture), "
            f"2 (forest), and 3 (developed). Found invalid values: {sorted(invalid)}"
        )


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


@threaded
def landscape_mosaic(
    data: xr.DataArray | np.ndarray,
    window_size: int = 27,
    output_classes: int = 19,
    *,
    threads: int = 0,
) -> xr.DataArray | np.ndarray:
    """Compute Landscape Mosaic (LM) moving-window metric matching GuidosToolbox batch_lm.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        2D raster coded:
        - 0: Missing
        - 1: Agriculture (or Class 1)
        - 2: Forest / Natural (or Class 2)
        - 3: Developed / Antropic (or Class 3)
    window_size : int, default 27
        Size of the square moving convolution window. Must be an odd integer >= 3.
        Common GTB observation scales: 3, 5, 7, 9, 11, 13, 27, 81, 243.
    output_classes : int, default 19
        Number of output classes:
        - 19: Standard GuidosToolbox 19-class landscape mosaic.
        - 103: Detailed 103-class barycentric triangle cells.
    threads : int, default 0
        Threads of the compiled code (also as ``n_threads``); 0: all cores but one.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Landscape mosaic raster (uint8).

    Examples
    --------
    Land cover in three classes (1 agriculture, 2 natural, 3 developed) from
    the CORINE example:

    >>> import landschaft as ls
    >>> clc = ls.load_raster(ls.example_data("clc"))
    >>> three = ls.reclassify(clc, [(1, 12, 3), (12, 23, 1), (23, 35, 2)], others=0)   # developed, agriculture, natural
    >>> mosaic = ls.landscape_mosaic(three, window_size=27)
    >>> ls.LM19_CLASSES[17]
    'Pure Forest (100%)'
    """
    _validate_lm_input(data)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    if output_classes not in (19, 103):
        raise ValueError("output_classes must be either 19 or 103")

    arr = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)
    res = _core.compute_landscape_mosaic(arr, window_size, output_classes, threads)

    attrs = {
        "landschaft_product": "LANDSCAPE_MOSAIC",
        "output_classes": output_classes,
        "window_size": window_size,
        "colormap": "lm" if output_classes == 19 else None,
        "TIFFTAG_IMAGEDESCRIPTION": "GTB_LM, https://forest.jrc.ec.europa.eu/activities/lpa/gtb/",
    }
    return _wrap_as_dataarray(res, data, f"landscape_mosaic_{output_classes}", attrs)


@threaded
def landscape_mosaic_stats(
    data: xr.DataArray | np.ndarray,
    window_size: int = 27,
    *,
    is_lm103: bool = False,
    threads: int = 0,
) -> pd.DataFrame:
    """Compute Landscape Mosaic 103-class frequency distribution matching GTB heatmap.csv.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        Either:
        - 2D landcover raster with values in {0, 1, 2, 3} (if is_lm103=False).
        - Precomputed 103-class landscape mosaic raster (if is_lm103=True).
    window_size : int, default 27
        Window size used when computing 103-class LM from raw landcover.
    is_lm103 : bool, default False
        Set to True if ``data`` is already a 103-class LM raster.
    threads : int, default 0
        Threads of the compiled code (also as ``n_threads``); 0: all cores but one.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``class_code``, ``pixel_count``, ``percentage``,
        ``recode_class_19`` and ``recode_name_19`` (the 19-class code and name).

    Examples
    --------
    >>> import landschaft as ls
    >>> clc = ls.load_raster(ls.example_data("clc"))
    >>> three = ls.reclassify(clc, [(1, 12, 3), (12, 23, 1), (23, 35, 2)], others=0)   # developed, agriculture, natural
    >>> table = ls.landscape_mosaic_stats(three, window_size=27)
    >>> list(table.columns)
    ['class_code', 'pixel_count', 'percentage', 'recode_class_19', 'recode_name_19']
    """
    if is_lm103:
        im103 = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)
    else:
        _validate_lm_input(data)
        arr = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)
        im103 = _core.compute_landscape_mosaic(arr, window_size, 103, threads)

    raw_stats = _core.compute_landscape_mosaic_stats(im103)
    codes = raw_stats["class_codes"]
    counts = raw_stats["pixel_counts"]
    pcts = raw_stats["percentages"]

    recode_19 = [RECODE_103_TO_19[c] for c in codes]
    class_names = [LM19_CLASSES.get(r, f"Class {r}") for r in recode_19]

    df = pd.DataFrame({
        "class_code": codes,
        "pixel_count": counts,
        "percentage": pcts,
        "recode_class_19": recode_19,
        "recode_name_19": class_names,
    })
    return df
