# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape metrics of categorical rasters (FRAGSTATS, as in landscapemetrics).

The raster is analysed once by the C++ core (patch labelling, per-patch
statistics, class adjacencies) and kept in a :class:`Landscape`; every metric
is then computed from those tables. Values match the R package
landscapemetrics (``lsm_p_*``, ``lsm_c_*``, ``lsm_l_*``).
"""

from __future__ import annotations

import functools
import os
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .catalog import LEVELS, select_metrics

__all__ = ["Landscape", "landscape_metrics"]

_INT32 = np.iinfo(np.int32)


def _read(source, band: int | None) -> xr.DataArray:
    from ..io import load_raster

    return load_raster(source, band=band, validate=None)


def _prepare(data, res, nodata, band) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float], Any]:
    """Class values (int32), NA mask, (x, y) cell size and a template for rasters."""
    if isinstance(data, (str, os.PathLike)):
        data = _read(data, band)
    template = None
    mask = None
    if isinstance(data, xr.DataArray):
        da = data
        if "band" in da.dims:
            if da.sizes["band"] != 1 and band is None:
                raise ValueError("the raster has several bands; select one with band=")
            da = da.sel(band=band) if band is not None else da.isel(band=0)
        if da.ndim != 2:
            raise ValueError(f"expected a 2-D raster, got dimensions {da.dims}")
        template = da
        if res is None:
            try:
                rx, ry = da.rio.resolution()
                res = (abs(float(rx)), abs(float(ry)))
            except Exception:  # no georeferencing: unit cells
                res = None
            try:
                crs = da.rio.crs
            except Exception:
                crs = None
            if crs is not None and crs.is_geographic:
                warnings.warn(
                    "the raster has a geographic CRS: areas and distances are in squared degrees and degrees "
                    "(as in landscapemetrics); project it for metric units",
                    stacklevel=3,
                )
        if nodata is None:
            nodata = da.rio.nodata if hasattr(da, "rio") else None
        values = np.asarray(da.values)
    elif np.ma.isMaskedArray(data):
        mask = np.ma.getmaskarray(data).copy()
        values = np.ma.getdata(data)
    else:
        values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D raster, got shape {values.shape}")

    if mask is None:
        mask = np.zeros(values.shape, dtype=bool)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        mask |= values == nodata
    if np.issubdtype(values.dtype, np.floating):
        mask |= np.isnan(values)
        valid = values[~mask]
        if valid.size and not np.array_equal(valid, np.trunc(valid)):
            warnings.warn("non-integer values are truncated to integers (as terra::as.int)", stacklevel=3)
        values = np.trunc(np.where(mask, 0, values))
    elif not (np.issubdtype(values.dtype, np.integer) or values.dtype == bool):
        raise TypeError(f"expected a numeric raster, got dtype {values.dtype}")
    valid = values[~mask]
    if valid.size and (valid.min() < _INT32.min + 1 or valid.max() > _INT32.max):
        raise ValueError("class values must fit in a 32-bit integer")
    values = np.ascontiguousarray(values, dtype=np.int32)
    if res is None:
        res = (1.0, 1.0)
    elif np.isscalar(res):
        res = (float(res), float(res))
    res = (float(res[0]), float(res[1]))
    if res[0] <= 0 or res[1] <= 0:
        raise ValueError(f"cell size must be positive, got {res}")
    return values, (mask.astype(np.uint8) if mask.any() else None), res, template


# landscapemetrics' metric options and their defaults.
OPTION_DEFAULTS = {
    "directions": 8,
    "count_boundary": False,
    "consider_boundary": False,
    "edge_depth": 1,
    "cell_center": False,
    "classes_max": None,
    "neighbourhood": 4,
    "ordered": True,
    "base": "log2",
}
_LEVEL_CODE = {"patch": 0, "class": 1, "landscape": 2}


def metric_options(**options) -> dict[str, Any]:
    """The metric options (defaults filled in, types checked)."""
    unknown = set(options) - set(OPTION_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown option(s): {sorted(unknown)}; expected {sorted(OPTION_DEFAULTS)}")
    out = {**OPTION_DEFAULTS, **options}
    out["directions"] = int(out["directions"])
    out["count_boundary"] = bool(out["count_boundary"])
    out["consider_boundary"] = bool(out["consider_boundary"])
    out["edge_depth"] = int(out["edge_depth"])
    out["cell_center"] = bool(out["cell_center"])
    out["neighbourhood"] = int(out["neighbourhood"])
    out["ordered"] = bool(out["ordered"])
    return out


def chosen_metrics(level=None, metric=None, name=None, type=None, what=None) -> pd.DataFrame:
    """Selected catalogue rows in calculate_lsm's order (level, metric)."""
    return select_metrics(level, metric, name, type, what).sort_values(["level", "metric"]).reset_index(drop=True)


def metric_specs(chosen: pd.DataFrame) -> tuple[list[int], list[str]]:
    return [_LEVEL_CODE[lv] for lv in chosen["level"]], list(chosen["metric"])


def long_frame(table: dict, chosen: pd.DataFrame) -> pd.DataFrame:
    """calculate_lsm's table from the C++ long table of the metrics `chosen`."""
    level_names = sorted(set(chosen["level"]))
    metric_order = list(dict.fromkeys(chosen["metric"]))
    spec_level = np.array([level_names.index(lv) for lv in chosen["level"]], dtype=np.int8)
    spec_metric = np.array([metric_order.index(m) for m in chosen["metric"]], dtype=np.int16)
    spec = table["metric"]
    return pd.DataFrame(
        {
            "level": pd.Categorical.from_codes(spec_level[spec], level_names),
            "class": pd.arrays.IntegerArray(table["class"].astype(np.int64), table["class_na"]),
            "id": pd.arrays.IntegerArray(table["id"].astype(np.int64), table["id_na"]),
            "metric": pd.Categorical.from_codes(spec_metric[spec], metric_order),
            "value": table["value"],
        }
    )


class Landscape:
    """Categorical raster prepared for landscape metrics.

    Patches and adjacencies are computed once, on first use, and shared by
    every metric. Metric names are the landscapemetrics / FRAGSTATS
    abbreviations (see :func:`landschaft.list_metrics`).

    Parameters
    ----------
    data
        Categorical map: a file path, an ``xarray.DataArray`` (cell size, NA
        value and CRS are taken from it), a numpy array or a masked array.
        Floats are truncated to integers; NaN is NA.
    res
        Cell size ``(x, y)`` (or one number) in map units. Default: from the
        raster, else 1.
    nodata
        Value treated as NA (default: the raster's nodata value).
    directions
        Patch connectivity: 8 (queen's case, default) or 4 (rook's case).
    count_boundary
        Count the landscape boundary (and NA cells) as edge in ``te``/``ed``.
    consider_boundary, edge_depth
        Core-area options (cells at the boundary may be core; distance in
        cells from the patch edge).
    cell_center
        ``gyrate``: use the patch cell nearest to the centroid.
    classes_max
        Potential number of classes, for ``rpr``.
    neighbourhood, ordered, base
        Complexity metrics (``ent``, ``condent``, ``joinent``, ``mutinf``,
        ``relmutinf``): 4 or 8 neighbours, ordered pairs, and the logarithm
        base ("log2", "log" or "log10").
    band
        Band to read from multi-band files.
    block_rows
        Out-of-core computation for rasters larger than memory: ``data`` must
        be a file, read by blocks of this many rows (rounded up to a multiple
        of 64; ``"auto"``: about 4 million cells per block). Only the patch
        tables stay in memory. Results are identical to the in-memory ones;
        ``labels`` and ``to_raster`` are not available.

    Examples
    --------
    >>> lm = Landscape("landcover.tif")
    >>> lm.class_metrics(["ca", "pland", "np", "area_mn"])
    >>> lm.landscape_metrics(["shdi", "contag", "ed"])
    >>> lm.metrics(type="aggregation metric")          # long format
    """

    def __init__(
        self,
        data,
        *,
        res=None,
        nodata=None,
        directions: int = 8,
        count_boundary: bool = False,
        consider_boundary: bool = False,
        edge_depth: int = 1,
        cell_center: bool = False,
        classes_max: float | None = None,
        neighbourhood: int = 4,
        ordered: bool = True,
        base: str = "log2",
        band: int | None = None,
        block_rows: int | str | None = None,
    ):
        self.options = metric_options(
            directions=directions,
            count_boundary=count_boundary,
            consider_boundary=consider_boundary,
            edge_depth=edge_depth,
            cell_center=cell_center,
            classes_max=classes_max,
            neighbourhood=neighbourhood,
            ordered=ordered,
            base=base,
        )
        self._tiled = None
        if block_rows is not None:
            from .tiled import TiledReader

            self._tiled = TiledReader(
                data, band=band, nodata=nodata, res=res, block_rows=block_rows, options=self.options
            )
            self._template = None
            self.resolution = self._tiled.res
            self.shape = self._tiled.shape
            self._core = self._tiled.engine.landscape()
            return
        values, mask, res, template = _prepare(data, res, nodata, band)
        self._template = template
        self.resolution = res
        self.shape = values.shape
        self._core = _core.LsmLandscape(values, mask, res[0], res[1], self.options)

    def __repr__(self) -> str:
        return (
            f"Landscape(shape={self.shape}, resolution={self.resolution}, classes={self.classes.tolist()}, "
            f"directions={self.options['directions']})"
        )

    # -- data ---------------------------------------------------------------

    @property
    def classes(self) -> np.ndarray:
        """Class values present, in increasing order."""
        return self._core.class_values

    @property
    def n_patches(self) -> int:
        return self._core.n_patches()

    @functools.cached_property
    def patch_classes(self) -> np.ndarray:
        """Class value of every patch (patch ``id`` = position + 1)."""
        return self.classes[self._core.patch_class()] if self.classes.size else np.empty(0, np.int32)

    @property
    def out_of_core(self) -> bool:
        """Read by blocks (``block_rows``)."""
        return self._tiled is not None

    def labels(self) -> xr.DataArray | np.ndarray:
        """Patch ids (1..n, the ``id`` of the patch tables); 0 where NA."""
        self._no_raster("labels")
        return self._wrap(self._core.labels(), "patches")

    def _no_raster(self, what: str) -> None:
        if self._tiled is not None:
            raise NotImplementedError(f"{what} is not available for an out-of-core landscape (block_rows)")

    def _ensure(self, metrics) -> None:
        """Out of core: read the file again for the metrics that need it."""
        if self._tiled is not None:
            self._tiled.ensure(metrics)

    def to_raster(self, metric: str) -> xr.DataArray | np.ndarray:
        """Patch-level metric painted on its patches (NaN where NA).

        The equivalent of landscapemetrics' ``spatialize_lsm``.
        """
        self._no_raster("to_raster")
        values = np.concatenate([[np.nan], self._patch_values(metric)])
        return self._wrap(values[self._core.labels()], metric)

    def _wrap(self, arr: np.ndarray, name: str):
        if self._template is None:
            return arr
        da = self._template.copy(data=arr)
        da.name = name
        da.attrs = {}
        da.encoding = {}
        if np.issubdtype(arr.dtype, np.floating):
            da = da.rio.write_nodata(np.nan, encoded=False)
        else:
            da = da.rio.write_nodata(0, encoded=False)
        return da

    # -- metrics ------------------------------------------------------------

    def _patch_values(self, metric: str) -> np.ndarray:
        _check_level([metric], "patch")
        self._ensure([metric])
        return self._core.patch_metric(metric)

    def patch_metrics(self, metrics: Iterable[str] | str | None = None) -> pd.DataFrame:
        """Patch-level metrics: one row per patch (``class``, ``id``, metrics...)."""
        metrics = _names(metrics, "patch")
        self._ensure(metrics)
        df = pd.DataFrame({"class": self.patch_classes, "id": np.arange(1, self.n_patches + 1)})
        for m in metrics:
            df[m] = self._core.patch_metric(m)
        return df

    def class_metrics(self, metrics: Iterable[str] | str | None = None) -> pd.DataFrame:
        """Class-level metrics: one row per class (``class``, metrics...)."""
        metrics = _names(metrics, "class")
        self._ensure(metrics)
        df = pd.DataFrame({"class": self.classes})
        for m in metrics:
            df[m] = self._core.class_metric(m)
        return df

    def landscape_metrics(self, metrics: Iterable[str] | str | None = None) -> pd.DataFrame:
        """Landscape-level metrics: a single row."""
        metrics = _names(metrics, "landscape")
        self._ensure(metrics)
        return pd.DataFrame({m: [self._core.landscape_metric(m)] for m in metrics})

    def metrics(self, level=None, metric=None, name=None, type=None, what=None) -> pd.DataFrame:
        """Selected metrics in long format, like landscapemetrics' ``calculate_lsm``.

        Selection as in :func:`landschaft.list_metrics` (all metrics by
        default).

        Returns
        -------
        pandas.DataFrame
            Columns ``level``, ``class``, ``id``, ``metric``, ``value``
            (``class``/``id`` are missing where they do not apply). An all-NA
            map gives one missing value per metric, as in landscapemetrics.
        """
        chosen = chosen_metrics(level, metric, name, type, what)
        self._ensure(chosen["metric"])
        return long_frame(self._core.long_table(*metric_specs(chosen)), chosen)


def _names(metrics, level: str) -> list[str]:
    if metrics is None:
        return list(select_metrics(level=level)["metric"])
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = list(metrics)
    _check_level(metrics, level)
    return metrics


def _check_level(metrics: Sequence[str], level: str) -> None:
    available = set(select_metrics(level=level)["metric"])
    unknown = [m for m in metrics if m not in available]
    if unknown:
        other = [lv for lv in LEVELS if lv != level and set(unknown) & set(select_metrics(level=lv)["metric"])]
        hint = f" (available at the {' and '.join(other)} level)" if other else ""
        raise ValueError(f"not {level}-level metric(s): {unknown}{hint}; see list_metrics(level={level!r})")


def landscape_metrics(
    data,
    level=None,
    metric=None,
    name=None,
    type=None,
    what=None,
    **options,
) -> pd.DataFrame:
    """Landscape metrics of one or several rasters, in long format.

    The one-call equivalent of landscapemetrics' ``calculate_lsm``.

    Parameters
    ----------
    data
        A raster (see :class:`Landscape`), or a list of rasters /
        :class:`~landschaft.RasterCollection` (a ``layer`` column numbers them
        from 1).
    level, metric, name, type, what
        Metric selection, as in :func:`landschaft.list_metrics`; all metrics
        by default.
    **options
        :class:`Landscape` options (``directions``, ``count_boundary``,
        ``edge_depth``, ``res``, ``nodata``, ``block_rows`` for files larger
        than memory, ...).

    Returns
    -------
    pandas.DataFrame
        ``level``, ``class``, ``id``, ``metric``, ``value`` (plus ``layer``
        for several rasters). Several rasters are computed in parallel.
    """
    layers = as_layers(data)
    if layers is None:
        return Landscape(data, **options).metrics(level, metric, name, type, what)
    if options.get("block_rows") is not None:
        frames = [Landscape(layer, **options).metrics(level, metric, name, type, what) for layer in layers]
        for i, df in enumerate(frames, start=1):
            df.insert(0, "layer", i)
        return pd.concat(frames, ignore_index=True)
    res, nodata, band = options.pop("res", None), options.pop("nodata", None), options.pop("band", None)
    opts = metric_options(**options)
    chosen = chosen_metrics(level, metric, name, type, what)
    rasters = []
    for layer in layers:
        values, mask, r, _ = _prepare(layer, res, nodata, band)
        rasters.append((values, mask, r[0], r[1]))
    table = _core.lsm_batch(rasters, opts, *metric_specs(chosen))
    df = long_frame(table, chosen)
    df.insert(0, "layer", table["item"].astype(np.int64) + 1)
    return df


def as_layers(data) -> list | None:
    """The rasters of a list / RasterCollection; None for a single raster."""
    from ..collection import RasterCollection

    if isinstance(data, RasterCollection):
        return [data.load(i) for i in range(len(data))]
    if isinstance(data, (list, tuple)):
        return list(data)
    return None
