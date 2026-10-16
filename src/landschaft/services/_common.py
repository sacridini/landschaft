# SPDX-License-Identifier: GPL-3.0-or-later
"""Land use maps and parameter tables of the ecosystem service models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core
from .._threads import resolve
from ..cost._transition import _crs_lonlat
from ..surface._grid import Surface, read_surface


class LandUse:
    """A land use map: its codes, which cells have one, and its grid."""

    def __init__(self, x, nodata=None, what: str = "the land use map"):
        s = read_surface(x, nodata=nodata) if not isinstance(x, Surface) else x
        self.surface = s
        v = s.values
        self.valid = ~np.isnan(v)
        codes = np.where(self.valid, v, 0)
        if np.any(codes != np.round(codes)):
            raise ValueError(f"{what} must hold integer class codes")
        self.codes = codes.astype(np.int64)
        self.what = what

    @property
    def shape(self):
        return self.surface.shape

    def pixel(self) -> tuple[float, float]:
        """Cell width and height in map units (metres: a projected map, as InVEST requires)."""
        if self.surface.crs is not None and _crs_lonlat(self.surface.crs):
            raise ValueError(f"{self.what} is in longitude / latitude: give it projected, in metres (as InVEST)")
        dx, dy = self.surface.res
        return abs(float(dx)), abs(float(dy))

    def classes(self, codes, n_threads=None) -> np.ndarray:
        """The row of ``codes`` (sorted) of every cell; -1 where the map has no value."""
        cls, first = _core.serv_classify(self.codes, self.valid.astype(np.uint8), np.asarray(codes, np.int64),
                                         resolve(n_threads))
        if first >= 0:
            missing = sorted(set(np.unique(self.codes[self.valid & (cls == -2)]).tolist()))
            raise ValueError(f"{self.what} has class(es) {missing} that the table lacks")
        return cls

    def raster(self, values, name: str):
        return self.surface.raster(values, name)


def same_grid(a: LandUse, b: LandUse) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{b.what} has {b.shape} cells, {a.what} {a.shape}: give them on the same grid")


def lookup_table(table, columns, what: str, index: str = "lucode") -> tuple[np.ndarray, pd.DataFrame]:
    """A table by land use code: ``(codes sorted, the columns as float64)``.

    ``table``: a DataFrame with a ``lucode`` column (or indexed by the codes), or a
    ``dict`` ``{code: {column: value}}``. Column names are not case-sensitive.
    """
    if isinstance(table, dict):
        df = pd.DataFrame.from_dict(table, orient="index")
    elif isinstance(table, pd.DataFrame):
        df = table.copy()
    else:
        raise TypeError(f"{what} is a DataFrame (or a dict by class code), not {type(table).__name__}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if index in df.columns:
        df = df.set_index(index)
    df.index = pd.Index([int(v) for v in df.index], name=index)
    if df.index.has_duplicates:
        raise ValueError(f"{what} lists a class more than once")
    missing = [c for c in columns if c.lower() not in df.columns]
    if missing:
        raise ValueError(f"{what} lacks the column(s) {missing}")
    df = df.sort_index()
    out = df[[c.lower() for c in columns]].astype(np.float64)
    if out.isna().any().any():
        raise ValueError(f"{what} has empty values")
    return df.index.to_numpy(np.int64), out
