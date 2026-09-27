# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape Mosaic heatmap and its change (GuidosToolbox LM heatmap / Change -> LM Heatmap).

GTB summarises a 103-class Landscape Mosaic by its *heatmap*: the share [%] of
every class, arranged in the tri-polar triangle (agriculture / forest /
developed). :func:`landscape_mosaic_heatmap` reproduces GTB's heatmap vector
(single precision, as GTB stores it in ``*_heatmap.sav``) and its
``heatmap.csv``; :func:`landscape_mosaic_change` the difference B - A and
``delta_heatmap.csv``, character for character.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .io import load_raster
from .mosaic import RECODE_103_TO_19, landscape_mosaic

__all__ = [
    "LM_TRIANGLE",
    "LMHeatmap",
    "LMHeatmapChange",
    "landscape_mosaic_change",
    "landscape_mosaic_heatmap",
]

# The 103 LM classes in GTB's triangle layout: (first column, class codes) per
# row, top (forest) to bottom; the last row holds the agriculture (190) and
# developed (180) corners at columns 0 and 20.
LM_TRIANGLE: tuple[tuple[int, tuple[int, ...]], ...] = (
    (10, (170,)),
    (10, (171,)),
    (9, (91, 172, 81)),
    (8, (93, 92, 121, 82, 83)),
    (7, (95, 94, 124, 122, 123, 84, 85)),
    (6, (152, 151, 217, 216, 215, 214, 213, 145, 144)),
    (5, (154, 153, 219, 218, 233, 232, 231, 212, 211, 143, 142)),
    (4, (65, 155, 221, 220, 235, 234, 236, 230, 229, 210, 209, 141, 55)),
    (3, (63, 64, 113, 222, 223, 224, 225, 226, 227, 228, 207, 208, 104, 54, 53)),
    (2, (61, 62, 111, 112, 114, 200, 201, 202, 203, 204, 205, 206, 103, 102, 101, 52, 51)),
    (1, (191, 192, 71, 72, 73, 74, 75, 131, 132, 133, 134, 135, 45, 44, 43, 42, 41, 182, 181)),
)
_CORNERS = ((0, 190), (20, 180))
LM103_CODES: tuple[int, ...] = tuple(sorted({c for _, row in LM_TRIANGLE for c in row} | {c for _, c in _CORNERS}))


def _triangle(values: np.ndarray) -> np.ndarray:
    """GTB's ``hmap``: the heatmap values on a 12 x 21 grid, NaN outside the triangle."""
    grid = np.full((12, 21), np.nan, dtype=np.float32)
    for y, (x0, codes) in enumerate(LM_TRIANGLE):
        grid[y, x0:x0 + len(codes)] = values[list(codes)]
    for x, code in _CORNERS:
        grid[11, x] = values[code]
    return grid


def _idl_cell(value: np.float32) -> str:
    """GTB's CSV cell: ``strmid(string(round(v*1000)/1000.0), 0, dot+4)`` in IDL single precision."""
    scaled = float(np.float32(value) * np.float32(1000))
    rounded = int(math.copysign(math.floor(abs(scaled) + 0.5), scaled))  # IDL ROUND: half away from zero
    text = "%#13.6g" % float(np.float32(rounded) / np.float32(1000.0))  # IDL string(float): G13.6
    dot = text.find(".")
    return text[: dot + 4]


def _csv_text(title: str, values: np.ndarray) -> str:
    """The rows of GTB's heatmap.csv / delta_heatmap.csv."""
    cells = [_idl_cell(v) for v in values]
    lines = [title + ", " * 20 + ","]
    for y, (_, codes) in enumerate(LM_TRIANGLE):
        pad = 10 if y < 2 else 11 - y
        lines.append(", " * pad + "," + ",".join(cells[c] for c in codes) + ", " * (pad - 1) + ",")
    lines.append("," + cells[190] + ", " * 19 + "," + cells[180])
    return "\n".join(lines) + "\n"


def _class_table(values: np.ndarray, name: str) -> pd.DataFrame:
    codes = list(LM103_CODES)
    return pd.DataFrame(
        {name: values[codes].astype(np.float64), "recode_class_19": [RECODE_103_TO_19[c] for c in codes]},
        index=pd.Index(codes, name="class_code"),
    )


@dataclass(frozen=True)
class LMHeatmap:
    """Landscape Mosaic heatmap of one map (GTB ``*_heatmap.sav`` / ``.csv``).

    Attributes
    ----------
    values
        GTB's heatmap vector ``hn``: 256 single-precision entries, entry ``c`` =
        share [%] of LM class ``c`` among the non-missing pixels.
    window
        Window size label used in the reports (e.g. ``"27"`` or ``"multiscale"``).
    """

    values: np.ndarray
    window: str

    @property
    def table(self) -> pd.DataFrame:
        """Share [%] per LM class code, with its 19-class recode."""
        return _class_table(self.values, "percent")

    @property
    def triangle(self) -> np.ndarray:
        """The heatmap as GTB's 12 x 21 triangle grid (NaN outside the triangle)."""
        return _triangle(self.values)

    def csv_text(self) -> str:
        """Content of GTB's ``heatmap.csv``."""
        if self.window == "multiscale":
            title = "Landscape Mosaic (multiscale)"
        else:
            title = f"Landscape Mosaic using Window size ({self.window}x{self.window})"
        return _csv_text(title, self.values)

    def to_csv(self, path: str | Path) -> Path:
        """Write GTB's ``heatmap.csv``."""
        path = Path(path)
        path.write_text(self.csv_text())
        return path


@dataclass(frozen=True)
class LMHeatmapChange:
    """Heatmap difference B - A (GTB ``delta_heatmap.csv``); ``values`` as in :class:`LMHeatmap`."""

    values: np.ndarray
    a: LMHeatmap
    b: LMHeatmap

    @property
    def table(self) -> pd.DataFrame:
        """Share [%] per LM class code at A and B and its change."""
        df = _class_table(self.values, "change")
        df.insert(0, "percent_a", self.a.values[list(LM103_CODES)].astype(np.float64))
        df.insert(1, "percent_b", self.b.values[list(LM103_CODES)].astype(np.float64))
        return df

    @property
    def triangle(self) -> np.ndarray:
        """The change as GTB's 12 x 21 triangle grid (NaN outside the triangle)."""
        return _triangle(self.values)

    def csv_text(self) -> str:
        """Content of GTB's ``delta_heatmap.csv``."""
        title = f"Delta Landscape Mosaic: B(Window size: {self.b.window}) - A(Window size: {self.a.window})"
        return _csv_text(title, self.values)

    def to_csv(self, path: str | Path) -> Path:
        """Write GTB's ``delta_heatmap.csv``."""
        path = Path(path)
        path.write_text(self.csv_text())
        return path


def landscape_mosaic_heatmap(
    data,
    window_size: int = 27,
    *,
    is_lm103: bool = False,
    window: str | None = None,
    threads: int = 0,
) -> LMHeatmap:
    """Heatmap of a Landscape Mosaic, as GTB computes it.

        Parameters
        ----------
        data
            Land-cover map coded 0 = missing, 1 = agriculture, 2 = forest,
            3 = developed (array, DataArray or raster path), or with
            ``is_lm103=True`` a 103-class LM map (missing = 0).
        window_size
            LM window size for a land-cover map.
        is_lm103
            ``data`` already is a 103-class LM map.
        window
            Window label for the CSV title (default: ``window_size``; GTB writes
            ``"multiscale"`` for multiscale LM).
        threads
            OpenMP threads for the LM (0: all cores but one).

        Returns
        -------
        LMHeatmap
            The share of the map (%) in each of the 103 cells of the composition
            triangle (``values``); ``to_csv(path)`` writes the table.

        Examples
        --------
        >>> import xarray as xr
        >>> import landschaft as ls
        >>> clc = ls.load_raster(ls.example_data("clc"), validate=None)
        >>> three = xr.where(clc <= 11, 3, xr.where(clc <= 22, 1, xr.where(clc <= 34, 2, 0))).astype("uint8")
        >>> heat = ls.landscape_mosaic_heatmap(three, window_size=27)
        >>> _ = heat.to_csv("heatmap.csv")
    """
    if isinstance(data, (str, Path)):
        data = load_raster(data, validate=None)
    if is_lm103:
        im103 = np.asarray(data.values if isinstance(data, xr.DataArray) else data)
    else:
        im103 = np.asarray(landscape_mosaic(data, window_size, output_classes=103, threads=threads))
    if im103.dtype != np.uint8:
        raise ValueError(f"a 103-class LM map must be uint8, got {im103.dtype}")
    hist = np.bincount(im103.ravel(), minlength=256)
    # IDL: tot = total(hist[1:236]) & hn = hist/tot*100.0, all in single precision
    tot = np.float32(hist[1:237].sum())
    if tot == 0:
        raise ValueError("the LM map has no classified (non-missing) pixels")
    values = (hist.astype(np.float32) / tot) * np.float32(100.0)
    label = window if window is not None else str(int(window_size))
    return LMHeatmap(values=values, window=label)


def landscape_mosaic_change(a, b, window_size: int = 27, *, is_lm103: bool = False,
                            threads: int = 0) -> LMHeatmapChange:
    """Change of the Landscape Mosaic heatmap from A to B (GTB Change -> Landscape Mosaic Heatmap).

    ``a`` and ``b`` are :class:`LMHeatmap` objects or maps accepted by
    :func:`landscape_mosaic_heatmap` (with ``window_size`` / ``is_lm103``).
    """
    ha = a if isinstance(a, LMHeatmap) else landscape_mosaic_heatmap(a, window_size, is_lm103=is_lm103,
                                                                      threads=threads)
    hb = b if isinstance(b, LMHeatmap) else landscape_mosaic_heatmap(b, window_size, is_lm103=is_lm103,
                                                                      threads=threads)
    return LMHeatmapChange(values=hb.values - ha.values, a=ha, b=hb)
