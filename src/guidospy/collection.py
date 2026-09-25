# SPDX-License-Identifier: GPL-3.0-or-later
"""Collections of rasters: many files handled as one lazy object."""

from __future__ import annotations

import glob
import os
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr

from .validation import InvalidInputError

__all__ = ["RasterCollection", "RasterInfo", "expand_sources"]

RASTER_SUFFIXES = (".tif", ".tiff")
_GLOB_CHARS = frozenset("*?[")


def is_single_source(source) -> bool:
    """True for one file path/URL, False for lists, directories and glob patterns."""
    if not isinstance(source, (str, os.PathLike)):
        return False
    text = os.fspath(source)
    return not Path(text).is_dir() and not (_GLOB_CHARS & set(text))


def expand_sources(source, *, recursive: bool = False) -> list[Path]:
    """Turn a path, directory, glob pattern or iterable of those into file paths.

    Directories contribute their ``.tif``/``.tiff`` files (sorted; with
    ``recursive`` also those of sub-directories). Duplicates are dropped and
    the order is kept.
    """
    items = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    paths: list[Path] = []
    for item in items:
        text = os.fspath(item)
        p = Path(text)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            paths += sorted(q for q in p.glob(pattern) if q.is_file() and q.suffix.lower() in RASTER_SUFFIXES)
        elif _GLOB_CHARS & set(text):
            paths += sorted(Path(m) for m in glob.glob(text, recursive=recursive) if Path(m).is_file())
        else:
            paths.append(p)
    return list(dict.fromkeys(paths))


@dataclass(frozen=True)
class RasterInfo:
    """Header of one raster of a collection (pixels are not read)."""

    path: Path
    height: int
    width: int
    count: int
    dtype: str
    crs: Any
    transform: Any
    nodata: float | None

    @property
    def pixels(self) -> int:
        return self.height * self.width

    @classmethod
    def read(cls, path: Path) -> RasterInfo:
        with rasterio.open(path) as src:
            return cls(Path(path), src.height, src.width, src.count, src.dtypes[0], src.crs, src.transform, src.nodata)

    def problem(self, *, band: int | None, validate: str | None) -> str | None:
        """Why this raster cannot be used, judging from its header only."""
        if band is None and self.count != 1:
            return f"{self.count} bands, a single band is required (select one with band=N)"
        if band is not None and not 1 <= band <= self.count:
            return f"band {band} not found (file has {self.count} bands)"
        if validate == "mspa" and self.dtype != "uint8":
            return f"map is of type {self.dtype}, not uint8 (BYTE)"
        return None


class RasterCollection(Sequence):
    """Many rasters handled as one object; pixels are read only when processed.

    Create it with :func:`guidospy.load_raster` given a list of paths, a
    directory or a glob pattern. Pass it to :func:`guidospy.mspa` to process
    every raster in parallel (one thread per raster), or turn it into a lazy
    ``(file, y, x)`` stack with :meth:`to_stack` when all rasters share a grid.
    """

    def __init__(
        self,
        infos: Iterable[RasterInfo],
        *,
        band: int | None = None,
        validate: str | None = "mspa",
        skipped: dict[Path, str] | None = None,
    ):
        self._infos = list(infos)
        self.band = band
        self.validate = validate
        #: Files left out by ``errors="skip"`` with the reason.
        self.skipped: dict[Path, str] = dict(skipped or {})

    @classmethod
    def from_sources(
        cls,
        source,
        *,
        band: int | None = None,
        validate: str | None = "mspa",
        errors: str = "raise",
        recursive: bool = False,
    ) -> RasterCollection:
        """Collect rasters and check their headers (band count, data type)."""
        if errors not in ("raise", "skip"):
            raise ValueError(f"errors must be 'raise' or 'skip', got {errors!r}")
        paths = expand_sources(source, recursive=recursive)
        if not paths:
            raise InvalidInputError(f"no raster found in {source!r}")

        def header(path: Path) -> RasterInfo | str:
            try:
                info = RasterInfo.read(path)
            except Exception as err:  # noqa: BLE001 - reported per file
                return f"cannot open: {err}"
            return info.problem(band=band, validate=validate) or info

        with ThreadPoolExecutor(max_workers=min(32, len(paths))) as pool:
            headers = list(pool.map(header, paths))
        good = [h for h in headers if isinstance(h, RasterInfo)]
        bad = {p: h for p, h in zip(paths, headers) if isinstance(h, str)}
        if bad and errors == "raise":
            listing = "\n".join(f"  {p}: {msg}" for p, msg in list(bad.items())[:10])
            more = f"\n  ... and {len(bad) - 10} more" if len(bad) > 10 else ""
            raise InvalidInputError(
                f"{len(bad)} of {len(paths)} rasters cannot be used (pass errors='skip' to leave them out):\n"
                f"{listing}{more}"
            )
        if bad:
            warnings.warn(f"skipping {len(bad)} of {len(paths)} rasters; see .skipped", stacklevel=3)
        if not good:
            raise InvalidInputError(f"none of the {len(paths)} rasters can be used")
        return cls(good, band=band, validate=validate, skipped=bad)

    # --- sequence protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self._infos)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return RasterCollection(self._infos[index], band=self.band, validate=self.validate)
        return self._infos[index]

    def __repr__(self) -> str:
        grid = "shared grid" if self.same_grid else "different grids"
        return (
            f"<RasterCollection: {len(self)} rasters, {grid}, "
            f"{self.total_pixels / 1e6:,.1f} Mpx{f', {len(self.skipped)} skipped' if self.skipped else ''}>"
        )

    # --- properties ------------------------------------------------------------
    @property
    def paths(self) -> list[Path]:
        return [i.path for i in self._infos]

    @property
    def total_pixels(self) -> int:
        return sum(i.pixels for i in self._infos)

    @property
    def same_grid(self) -> bool:
        """All rasters have the same shape, CRS and transform."""
        first = self._infos[0]
        return all(
            (i.height, i.width, i.crs, i.transform) == (first.height, first.width, first.crs, first.transform)
            for i in self._infos[1:]
        )

    # --- data access -------------------------------------------------------------
    def load(self, index: int) -> xr.DataArray:
        """Read one raster into memory (with the collection's checks)."""
        from .io import load_raster

        return load_raster(self._infos[index].path, band=self.band, validate=self.validate)

    def to_stack(self) -> xr.DataArray:
        """Lazy ``(file, y, x)`` DataArray, one dask chunk per raster.

        Requires a shared grid (see :attr:`same_grid`). Pixel values are not
        checked here; :func:`guidospy.mspa` checks them when it runs.
        """
        if not self.same_grid:
            raise InvalidInputError(
                "rasters do not share the same grid (shape, CRS and transform); "
                "process them as a collection instead of a stack"
            )
        names = [p.stem for p in self.paths]
        if len(set(names)) != len(names):
            names = [str(p) for p in self.paths]
        layers = []
        for info in self._infos:
            da = rioxarray.open_rasterio(info.path, chunks={"band": 1, "y": -1, "x": -1}, masked=False)
            layers.append(da.sel(band=self.band or 1, drop=True))
        stack = xr.concat(
            layers, dim=pd.Index(names, name="file"), coords="minimal", compat="override", join="override"
        )
        stack.name = "stack"
        stack.attrs = {k: v for k, v in layers[0].attrs.items() if k not in ("_FillValue", "scale_factor", "add_offset")}
        stack = stack.assign_coords(path=("file", np.array([str(p) for p in self.paths])))
        return stack
