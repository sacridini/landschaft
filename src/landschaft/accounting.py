# SPDX-License-Identifier: GPL-3.0-or-later
"""Accounting: foreground objects grouped into size classes.

Reimplements GuidosToolbox's *Image Analysis → Objects → Accounting* (and its
batch version): every foreground object is labelled, measured and assigned
to one of up to six size classes. The maps and numbers are identical to the
GTB outputs ``<name>_acc.tif``, ``<name>_acc_ids.tif``,
``<name>_acc_pixels.tif`` and ``<name>_acc.txt/.csv``.

Reference
---------
Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object analysis.
    European Journal of Remote Sensing, 50(1), 352-361. doi:10.1080/22797254.2017.1330650
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import _core
from .io import _load_map
from .validation import validate_fad_input
from ._save import Saveable

__all__ = ["ACCOUNTING_CLASSES", "AccountingResult", "accounting"]

DEFAULT_THRESHOLDS = (50, 250, 1000, 5000, 20000)  # GTB batch Accounting defaults

# Codes of the ``classes`` map (GTB's colour table) and their meaning.
ACCOUNTING_CLASSES: dict[int, str] = {
    0: "Background",
    103: "Size class 1 (black)",
    33: "Size class 2 (red)",
    65: "Size class 3 (yellow)",
    1: "Size class 4 (orange)",
    9: "Size class 5 (brown)",
    17: "Size class 6 (green)",
    80: "Three largest objects (pink)",
    129: "Missing",
    105: "Special background 1 (input 3)",
    176: "Special background 2 (input 4)",
}
_CLASS_CODES = (103, 33, 65, 1, 9, 17)
_COLOR_NAMES = ("black", "red", "yellow", "orange", "brown", "green")


@dataclass(frozen=True)
class AccountingResult(Saveable):
    """Result of :func:`accounting`.

    Attributes
    ----------
    classes
        ``uint8`` map in GTB's colour codes (see :data:`ACCOUNTING_CLASSES`).
    ids
        ``int32`` object ids ``1..n_objects`` (numbered like GTB: raster order
        starting from the bottom row); 0 background, -1 missing, -3/-4 special
        background.
    area
        ``int32`` map of each object's area in pixels, same codes as ``ids``.
    stats
        One row per size class: range, colour, number of objects, area and
        their percentages of all objects / of the total foreground area.
    object_area
        Area of object ``k`` at index ``k - 1``.
    largest
        Ids of the largest, second and third largest objects (objects of equal
        area share a rank).
    highlighted
        Whether the three largest objects are painted pink in ``classes``.

    Examples
    --------
    >>> import landschaft as ls
    >>> result = ls.accounting(ls.load_raster(ls.example_data("input")))
    >>> result.largest[0]
    (1,)
    >>> int(result.object_area[0])
    214811
    """

    classes: xr.DataArray | np.ndarray
    ids: xr.DataArray | np.ndarray
    area: xr.DataArray | np.ndarray
    stats: pd.DataFrame
    object_area: np.ndarray
    largest: tuple[tuple[int, ...], ...]
    thresholds: tuple[int, ...]
    connectivity: int
    pixel_resolution: float
    highlighted: bool

    @property
    def n_objects(self) -> int:
        return int(self.object_area.size)

    @property
    def foreground_area(self) -> int:
        return int(self.object_area.sum())

    @property
    def median_area(self) -> int:
        """Median object area as GTB reports it (the upper middle value for an even count)."""
        return int(np.sort(self.object_area)[self.n_objects // 2])

    @property
    def mean_area(self) -> float:
        return float(self.object_area.mean())

    @property
    def std_area(self) -> float:
        """Sample standard deviation of the object areas (NaN for a single object)."""
        return float(self.object_area.std(ddof=1)) if self.n_objects > 1 else float("nan")

    @property
    def hectare_factor(self) -> float:
        """Hectares per pixel for ``pixel_resolution`` metres."""
        return self.pixel_resolution**2 / 10000.0

    def summary(self) -> str:
        """Report in the layout of GTB's ``acc.txt``."""
        rule = "-" * 93 + " "
        header = "          # Objects      Area[pixels]     % of all objects  % of total FGarea"
        ha = self.hectare_factor
        lines = [
            "Accounting size classes result using: landschaft",
            f"Base settings: {self.connectivity}-connectivity, pixel resolution: {self.pixel_resolution:g} [m]",
            f"Conversion factor: pixel_to_hectare: {ha:g}, pixel_to_acres: {ha * 2.47105:g}",
            rule,
        ]
        for i, row in self.stats.iterrows():
            last = i == len(self.stats) - 1
            if last and row["objects"] == 0:
                continue  # GTB omits an empty last class
            lines += [
                f" Size class {row['size_class']}: [{row['range']}] pixels; color: {row['color']}",
                header,
                "".join(f"{v:>18}" for v in (row["objects"], row["area"],
                                              f"{row['pct_objects']:.6g}", f"{row['pct_area']:.6g}")),
                rule,
            ]
        lines += [
            rule,
            "Sum of all classes:",
            header,
            "".join(f"{v:>18}" for v in (self.n_objects, self.foreground_area, "100", "100")),
            " ",
            f"{'Median Patch Size: ':>20}{self.median_area:>16}",
            f"{'Average Patch Size: ':>20}{self.mean_area:>16.6g}",
            f"{'Standard Deviation: ':>20}{self.std_area:>16.6g}",
            " ",
        ]
        if self.highlighted:
            lines += ["Three largest object IDs and area[pixels]; color: pink",
                      "These 3 objects overlay objects listed above"]
            for rank, ids in enumerate(self.largest, 1):
                if ids:
                    lines.append(f"{rank}) {','.join(map(str, ids)):>18}{self.object_area[ids[0] - 1]:>18}")
        else:
            lines.append(f"Largest object:     {int(self.object_area.max())}")
        return "\n".join(lines)


def _class_table(object_area: np.ndarray, thresholds: tuple[int, ...]) -> pd.DataFrame:
    n = object_area.size
    total = int(object_area.sum())
    cls = np.searchsorted(np.asarray(thresholds, dtype=np.int64), object_area, side="left")
    nclass = len(thresholds) + 1
    counts = np.bincount(cls, minlength=nclass)
    areas = np.zeros(nclass, dtype=np.int64)
    np.add.at(areas, cls, object_area)
    ranges = [f"1 - {thresholds[0]}"]
    ranges += [f"{thresholds[k - 1] + 1} - {thresholds[k]}" for k in range(1, len(thresholds))]
    ranges.append(f"{thresholds[-1] + 1} -> ")
    return pd.DataFrame({
        "size_class": np.arange(1, nclass + 1),
        "range": ranges,
        "color": list(_COLOR_NAMES[:nclass]),
        "code": list(_CLASS_CODES[:nclass]),
        "objects": counts.astype(np.int64),
        "area": areas,
        "pct_objects": counts * 100.0 / n,
        "pct_area": areas * 100.0 / total,
    })


def accounting(
    data: xr.DataArray | np.ndarray | str | Path,
    thresholds: Sequence[int] | str = DEFAULT_THRESHOLDS,
    *,
    connectivity: int = 8,
    highlight_largest: bool = False,
    pixel_resolution: float = 25.0,
    n_threads: int = 0,
) -> AccountingResult:
    """Group the foreground objects into size classes (GTB Accounting).

    Parameters
    ----------
    data
        2-D ``uint8`` map (array, DataArray or raster path) coded 0 = missing,
        1 = background, 2 = foreground, 3/4 = special background.
    thresholds
        Upper bounds [pixels] of the size classes, 1 to 5 non-decreasing
        values >= 1 (repeats are dropped, as in GTB). Class 1 holds objects of
        up to ``thresholds[0]`` pixels, class k those in
        ``(thresholds[k-2], thresholds[k-1]]`` and the last class the larger
        ones. The default is GTB's batch setting; ``"auto"`` uses the classes
        GTB's interactive Accounting proposes (0.1, 1, 10, 50 and 90% of the
        range of the 8-connected object areas).
    connectivity
        Foreground connectivity, 8 (default) or 4.
    highlight_largest
        Paint the three largest objects pink (code 80), GTB's "big3pink".
        Only applied when there are at least three objects, as in GTB.
    pixel_resolution
        Pixel size in metres, used for the hectare/acre factors of the report.
    n_threads
        OpenMP threads (0: all cores but one).

    Returns
    -------
    AccountingResult

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> result = ls.accounting(forest, thresholds=(50, 250, 1000, 5000, 20000))
    >>> result.n_objects
    2850
    >>> result.stats[["size_class", "range", "objects"]].head(2)
       size_class     range  objects
    0           1    1 - 50     2668
    1           2  51 - 250      129
    >>> _ = ls.write_raster(result.classes, "size_classes.tif")

    Classes proposed from the range of object sizes:

    >>> auto = ls.accounting(forest, thresholds="auto")
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if isinstance(data, (str, Path)):
        data = _load_map(data)
    validate_fad_input(data)

    if isinstance(thresholds, str):
        if thresholds != "auto":
            raise ValueError(f"thresholds must be a sequence of integers or 'auto', got {thresholds!r}")
        cut: list[int] = []
    else:
        values = [int(t) for t in thresholds]
        if not 1 <= len(values) <= 5:
            raise ValueError(f"1 to 5 size thresholds are required, got {len(values)}")
        if any(t < 1 for t in values) or any(b < a for a, b in zip(values, values[1:])):
            raise ValueError(f"size thresholds must be >= 1 and non-decreasing, got {values}")
        cut = [t for i, t in enumerate(values) if i == 0 or t != values[i - 1]]

    arr = np.ascontiguousarray(data.values if isinstance(data, xr.DataArray) else data, dtype=np.uint8)
    out = _core.compute_accounting(arr, connectivity, cut, highlight_largest, n_threads)
    used = tuple(int(t) for t in out["thresholds"])
    object_area = np.asarray(out["object_area"][1:], dtype=np.int64)
    largest = tuple(tuple(int(i) for i in ids) for ids in out["largest"])

    attrs = {
        "landschaft_product": "ACCOUNTING",
        "connectivity": connectivity,
        "thresholds": ",".join(map(str, used)),
        "n_objects": int(object_area.size),
    }

    def wrap(values: np.ndarray, name: str, product: str | None = None):
        if not isinstance(data, xr.DataArray):
            return values
        da = data.copy(data=values)
        da.name = name
        da.attrs = dict(data.attrs) | attrs
        if product:
            da.attrs["landschaft_product"] = product
        return da

    return AccountingResult(
        classes=wrap(out["classes"], "accounting"),
        ids=wrap(out["ids"], "accounting_ids", "ACCOUNTING_IDS"),
        area=wrap(out["area"], "accounting_pixels", "ACCOUNTING_PIXELS"),
        stats=_class_table(object_area, used),
        object_area=object_area,
        largest=largest,
        thresholds=used,
        connectivity=connectivity,
        pixel_resolution=float(pixel_resolution),
        highlighted=bool(highlight_largest and object_area.size >= 3),
    )
