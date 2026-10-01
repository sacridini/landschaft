# SPDX-License-Identifier: GPL-3.0-or-later
"""``save``: any data or result written in the formats that suit it."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["Saveable", "save"]

#: 2-D arrays of a result with fewer cells than this are tables (matrices), not maps.
_MIN_MAP_CELLS = 1000


class Saveable:
    """Mixin of the result classes: ``result.save(folder)``."""

    def save(self, path, **kwargs) -> Path:
        """Save the result into a folder: every map as a GeoTIFF, every table as
        a CSV and the numbers in ``summary.json`` (see :func:`landschaft.save`).

        Examples
        --------
        >>> import landschaft as ls
        >>> forest = ls.load_raster(ls.example_data("input"))
        >>> folder = ls.fos(forest).save("fos_result")
        >>> sorted(p.name for p in folder.iterdir())
        ['histogram.csv', 'map.tif', 'stats.csv', 'summary.json']
        """
        return save(self, path, **kwargs)


def _is_scalar(v) -> bool:
    return isinstance(v, (bool, int, float, str, np.generic)) or v is None


def _plain(v):
    """A JSON value."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (Path, os.PathLike)):
        return os.fspath(v)
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()]
    return v


def _is_geo(v) -> bool:
    return hasattr(v, "geometry") and hasattr(v, "to_file") or type(v).__name__ == "GeoSeries"


def _is_array(v) -> bool:
    return isinstance(v, (np.ndarray, xr.DataArray)) or hasattr(v, "dask")


def _is_map(v, ref_shape) -> bool:
    """A raster: a (y, x) DataArray, or a 2-/3-D array the size of the result's maps."""
    if isinstance(v, xr.DataArray) and "y" in v.dims and "x" in v.dims and v.ndim in (2, 3):
        return True
    if v.ndim not in (2, 3) or not (np.issubdtype(v.dtype, np.number) or v.dtype == bool):
        return False
    shape = tuple(v.shape[-2:])
    return shape == ref_shape and shape[0] * shape[1] >= _MIN_MAP_CELLS


def _items(obj) -> dict:
    """The named parts of a result."""
    hook = getattr(obj, "_save_items", None)
    if callable(hook):
        return dict(hook())
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if not f.name.startswith("_")}
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def _reference(items: dict):
    """The georeferenced map and the map shape of a result (the largest 2-D array)."""
    like, best, size = None, None, 0
    stack = list(items.values())
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        elif _is_array(v) and getattr(v, "ndim", 0) >= 2:
            n = int(np.prod(v.shape[-2:]))
            if n > size:
                best, size = tuple(v.shape[-2:]), n
            if like is None and isinstance(v, xr.DataArray) and "x" in v.dims and v.rio.crs is not None:
                like = v
    return like, best


def _save_folder(obj, folder: Path, like, table_format: str) -> Path:
    from ._table import write_table
    from .io import write_raster

    items = _items(obj)
    found_like, ref_shape = _reference(items)
    like = like if like is not None else found_like
    folder.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    def put(name: str, v, depth: int = 0) -> None:
        if v is None:
            return
        if _is_scalar(v) or isinstance(v, (Path, os.PathLike)):
            summary[name] = _plain(v)
        elif _is_geo(v):
            save(v, folder / f"{name}.gpkg")
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            write_table(v, folder / f"{name}.{table_format}")
        elif _is_array(v):
            if getattr(v, "ndim", 0) == 0:
                summary[name] = _plain(np.asarray(v).item())
            elif _is_map(v, ref_shape):
                georef = isinstance(v, xr.DataArray) and "x" in v.dims and v.rio.crs is not None
                same = like is not None and tuple(v.shape[-2:]) == tuple(like.shape[-2:])
                write_raster(v, folder / f"{name}.tif", like=None if georef or not same else like)
            elif v.ndim == 1 and v.size <= 100 and not isinstance(v, xr.DataArray):
                summary[name] = _plain(np.asarray(v))
            elif v.ndim <= 2:
                a = np.asarray(v)
                frame = pd.DataFrame(a) if a.ndim == 2 else pd.DataFrame({name: a})
                write_table(frame, folder / f"{name}.{table_format}")
            else:
                np.save(folder / f"{name}.npy", np.asarray(v))
        elif isinstance(v, dict):
            if all(_is_scalar(x) for x in v.values()):
                summary[name] = _plain(v)
            else:
                for k, x in v.items():
                    put(f"{name}_{k}", x, depth + 1)
        elif isinstance(v, (list, tuple)):
            if all(_is_scalar(x) for x in v):
                summary[name] = _plain(list(v))
            else:
                for i, x in enumerate(v):
                    put(f"{name}_{i}", x, depth + 1)
        elif depth < 3 and (hasattr(v, "to_newick") or hasattr(v, "_save_items") or dataclasses.is_dataclass(v)):
            save(v, folder / name, table_format=table_format)
        else:
            summary[name] = repr(v)

    for name, value in items.items():
        put(name, value)
    if summary:
        summary = {"result": type(obj).__name__, **summary}
        (folder / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return folder


def _with_suffix(path: Path, suffix: str) -> Path:
    return path if path.suffix else path.with_suffix(suffix)


def save(obj, path, *, like=None, table_format: str = "csv", **kwargs) -> Path:
    """Save data or a result, each part in the format that suits it.

    One call for everything landschaft produces:

    - a raster (DataArray, numpy or dask array of 2 or 3 dimensions) becomes a
      GeoTIFF, as :func:`write_raster` writes it;
    - a table (DataFrame, Series) a CSV (or the format of the extension, see
      :func:`write_table`), a GeoDataFrame a GeoPackage (:func:`write_vector`);
    - a tree a Newick file;
    - a result (of :func:`fos`, :func:`restoration_status`,
      :func:`landschaft.prioritize.zonation`... or a ``dict`` of results) a
      folder: one GeoTIFF per map, one table per table, and the numbers and
      settings in ``summary.json``. Maps that are numpy arrays get the
      georeferencing of a georeferenced map of the same result, or of ``like``.
      Results that have a format of their own (Zonation's output files,
      Biodiverse's) are written in it.

    Without an extension, ``path`` gets the usual one (``.tif``, ``.csv``,
    ``.gpkg``) or is a folder for a result. Missing folders are created.

    Parameters
    ----------
    obj
        What to save.
    path
        File, or folder for a result.
    like
        A raster (DataArray or path) that georeferences numpy maps.
    table_format
        Extension of the tables of a result: ``"csv"`` (default), ``"parquet"``,
        ``"xlsx"``...
    **kwargs
        For a single raster, table or vector: passed to its writer
        (``nodata=``, ``compress=``...).

    Returns
    -------
    pathlib.Path
        The file or folder written.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> ls.save(ls.mspa(forest), "results/mspa").name           # a raster: GeoTIFF
    'mspa.tif'
    >>> status = ls.restoration_status(forest)
    >>> folder = ls.save(status, "results/restoration")         # a result: a folder
    >>> sorted(p.name for p in folder.iterdir())
    ['summary.json']
    >>> import json
    >>> json.loads((folder / "summary.json").read_text())["nr_comp"]
    2850
    """
    from ._table import write_table
    from ._vector import write_vector
    from .io import write_raster

    path = Path(path)
    if isinstance(obj, (Saveable, dict)):
        return _save_folder(obj, path, like, table_format)
    if _is_geo(obj):
        return write_vector(obj, _with_suffix(path, ".gpkg"), **kwargs)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return write_table(obj, _with_suffix(path, f".{table_format}"), **kwargs)
    if _is_array(obj):
        if getattr(obj, "ndim", 0) not in (2, 3):
            raise ValueError(f"a raster has 2 or 3 dimensions, got shape {obj.shape}; save tables as DataFrames")
        return write_raster(obj, _with_suffix(path, ".tif"), like=like, **kwargs)
    if hasattr(obj, "to_newick"):
        path = _with_suffix(path, ".nwk")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(obj.to_newick() + "\n", encoding="utf-8")
        return path
    for method in ("save", "write"):  # results with a format of their own
        if callable(getattr(obj, method, None)):
            path.parent.mkdir(parents=True, exist_ok=True)
            out = getattr(obj, method)(path)
            return Path(out) if out is not None else path
    if hasattr(obj, "_save_items") or dataclasses.is_dataclass(obj) or hasattr(obj, "__dict__"):
        return _save_folder(obj, path, like, table_format)
    raise TypeError(f"cannot save a {type(obj).__name__}")
