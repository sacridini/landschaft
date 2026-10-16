# SPDX-License-Identifier: GPL-3.0-or-later
"""Values to and from files, for the arguments and results of tasks: never pickle.

``write(value, folder)`` stores a value as ``value.json`` plus array files and
``read(folder)`` builds it again: numbers, strings, lists, tuples, dicts, numpy arrays
(not of objects), xarray DataArrays and Datasets (values, coordinates, attributes:
georeferencing included), pandas DataFrames and Series, GeoDataFrames (GeoParquet), and
dataclasses of landschaft built from these. A dataclass is rebuilt only if its class is
one of landschaft's: nothing else is ever imported or called.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path

import numpy as np


class Unpackable(TypeError):
    """A value that cannot travel between computers."""


class Param:
    """A placeholder for the i-th drawn parameter (see ``ls.parallel.param``)."""

    def __init__(self, index: int):
        self.index = int(index)

    def __repr__(self) -> str:
        return f"param({self.index})"


def _plain(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class _Writer:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def _name(self, ext: str) -> str:
        self.n += 1
        return f"v{self.n}{ext}"

    def array(self, a: np.ndarray) -> str:
        a = np.asarray(a)
        if a.dtype.kind == "O":
            raise Unpackable("arrays of Python objects cannot travel")
        name = self._name(".npy")
        np.save(self.folder / name, a, allow_pickle=False)
        return name

    def encode(self, v, depth: int = 0):
        import pandas as pd
        import xarray as xr

        if depth > 20:
            raise Unpackable("too deeply nested")
        if v is None or isinstance(v, (bool, int, float, str)):
            if isinstance(v, float) and not np.isfinite(v):
                return {"$float": repr(v)}
            return v
        if isinstance(v, np.generic):
            if v.dtype.kind in "fc" and not np.isfinite(v):
                return {"$float": repr(float(v))}
            return {"$scalar": self.array(np.asarray(v))}
        if isinstance(v, Param):
            return {"$param": v.index}
        if isinstance(v, np.ndarray):
            return {"$array": self.array(v)}
        if isinstance(v, xr.Dataset):
            return {"$dataset": {str(k): self.encode(v[k], depth + 1) for k in v.data_vars},
                    "attrs": self.encode(dict(v.attrs), depth + 1)}
        if isinstance(v, xr.DataArray):
            coords = {}
            for k, c in v.coords.items():
                coords[str(k)] = {"dims": list(c.dims), "values": self.array(np.asarray(c.values)),
                                  "attrs": self.encode(dict(c.attrs), depth + 1)}
            nodata = v.rio.encoded_nodata if hasattr(v, "rio") else None
            return {"$dataarray": self.array(np.asarray(v.values)), "dims": list(v.dims), "coords": coords,
                    "attrs": self.encode(dict(v.attrs), depth + 1),
                    "name": None if v.name is None else str(v.name),
                    "nodata": None if nodata is None else float(nodata)}
        if _is_geo(v):
            name = self._name(".parquet")
            v.to_parquet(self.folder / name)
            return {"$geoframe": name}
        if isinstance(v, pd.DataFrame):
            if isinstance(v.columns, pd.MultiIndex) or not all(isinstance(c, str) for c in v.columns):
                cols = list(v.columns)
                frame = v.copy()
                frame.columns = [f"c{i}" for i in range(len(cols))]
                name = self._name(".parquet")
                frame.to_parquet(self.folder / name)
                return {"$frame": name, "columns": self.encode([_column_key(c) for c in cols], depth + 1),
                        "multi": isinstance(v.columns, pd.MultiIndex),
                        "names": [None if n is None else str(n) for n in v.columns.names]}
            name = self._name(".parquet")
            v.to_parquet(self.folder / name)
            return {"$frame": name}
        if isinstance(v, pd.Series):
            return {"$series": self.encode(v.to_frame(name="__value__"), depth + 1),
                    "name": self.encode(None if v.name is None else _column_key(v.name), depth + 1)}
        if isinstance(v, tuple):
            return {"$tuple": [self.encode(x, depth + 1) for x in v]}
        if isinstance(v, list):
            return [self.encode(x, depth + 1) for x in v]
        if isinstance(v, dict):
            if all(isinstance(k, str) for k in v):
                return {"$dict": {k: self.encode(x, depth + 1) for k, x in v.items()}}
            return {"$items": [[self.encode(k, depth + 1), self.encode(x, depth + 1)] for k, x in v.items()]}
        if isinstance(v, Path):
            return {"$path": str(v)}
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            cls = type(v)
            if not cls.__module__.startswith("landschaft."):
                raise Unpackable(f"{cls.__name__} is not one of landschaft's results")
            fields = {f.name: self.encode(getattr(v, f.name), depth + 1) for f in dataclasses.fields(v)}
            return {"$dataclass": f"{cls.__module__}:{cls.__qualname__}", "fields": fields}
        raise Unpackable(f"a {type(v).__name__} cannot travel between computers")


def _column_key(c):
    return list(c) if isinstance(c, tuple) else c


def _is_geo(v) -> bool:
    return type(v).__name__ == "GeoDataFrame" and hasattr(v, "geometry")


def write(value, folder) -> Path:
    """Store ``value`` in ``folder`` (``value.json`` and its arrays)."""
    w = _Writer(Path(folder))
    spec = w.encode(value)
    (w.folder / "value.json").write_text(json.dumps(spec, allow_nan=False))
    return w.folder


class _Reader:
    def __init__(self, folder: Path, params=None):
        self.folder = Path(folder)
        self.params = params

    def _file(self, name) -> Path:
        from ._tasks import check_name

        check_name(name)
        if "/" in name:
            raise ValueError(f"not a value file: {name!r}")
        return self.folder / name

    def array(self, name) -> np.ndarray:
        return np.load(self._file(name), allow_pickle=False)

    def decode(self, v):
        import pandas as pd
        import xarray as xr

        if not isinstance(v, (dict, list)):
            return v
        if isinstance(v, list):
            return [self.decode(x) for x in v]
        if "$float" in v:
            return float(v["$float"])
        if "$scalar" in v:
            return self.array(v["$scalar"])[()]
        if "$param" in v:
            if self.params is None:
                raise ValueError("a parameter placeholder outside a Monte Carlo draw")
            return float(self.params[int(v["$param"])])
        if "$array" in v:
            return self.array(v["$array"])
        if "$dataset" in v:
            ds = xr.Dataset({k: self.decode(x) for k, x in v["$dataset"].items()})
            ds.attrs = self.decode(v["attrs"])
            return ds
        if "$dataarray" in v:
            coords = {k: xr.Variable(c["dims"], self.array(c["values"]), attrs=self.decode(c["attrs"]))
                      for k, c in v["coords"].items()}
            da = xr.DataArray(self.array(v["$dataarray"]), dims=v["dims"], coords=coords,
                              attrs=self.decode(v["attrs"]), name=v.get("name"))
            if v.get("nodata") is not None:
                da = da.rio.write_nodata(v["nodata"], encoded=True)
            return da
        if "$geoframe" in v:
            import geopandas as gpd

            return gpd.read_parquet(self._file(v["$geoframe"]))
        if "$frame" in v:
            frame = pd.read_parquet(self._file(v["$frame"]))
            if "columns" in v:
                cols = [tuple(c) if isinstance(c, list) else c for c in self.decode(v["columns"])]
                frame.columns = (pd.MultiIndex.from_tuples(cols, names=v.get("names")) if v.get("multi")
                                 else pd.Index(cols, name=(v.get("names") or [None])[0]))
            return frame
        if "$series" in v:
            s = self.decode(v["$series"])["__value__"]
            name = self.decode(v.get("name"))
            s.name = tuple(name) if isinstance(name, list) else name
            return s
        if "$tuple" in v:
            return tuple(self.decode(x) for x in v["$tuple"])
        if "$dict" in v:
            return {k: self.decode(x) for k, x in v["$dict"].items()}
        if "$items" in v:
            return {_hashable(self.decode(k)): self.decode(x) for k, x in v["$items"]}
        if "$path" in v:
            return Path(v["$path"])
        if "$dataclass" in v:
            module, _, qualname = str(v["$dataclass"]).partition(":")
            if not module.startswith("landschaft."):
                raise ValueError(f"refusing to build {v['$dataclass']}")
            cls = importlib.import_module(module)
            for part in qualname.split("."):
                cls = getattr(cls, part)
            if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
                raise ValueError(f"{v['$dataclass']} is not a dataclass")
            values = {k: self.decode(x) for k, x in v["fields"].items()}
            init = {f.name for f in dataclasses.fields(cls) if f.init}
            obj = cls(**{k: x for k, x in values.items() if k in init})
            for k, x in values.items():
                if k not in init:
                    object.__setattr__(obj, k, x)
            return obj
        raise ValueError(f"unknown value {sorted(v)[:3]}")


def _hashable(k):
    return tuple(_hashable(x) for x in k) if isinstance(k, list) else k


def read(folder, params=None):
    """The value stored by :func:`write` in ``folder``; ``params``: the draw that
    replaces the parameter placeholders."""
    folder = Path(folder)
    spec = json.loads((folder / "value.json").read_text())
    return _Reader(folder, params).decode(spec)


def decode(spec, folder, params=None):
    """A value from a spec already read (its files in ``folder``)."""
    return _Reader(Path(folder), params).decode(spec)


def files(folder) -> list:
    """The files of a stored value."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file())
