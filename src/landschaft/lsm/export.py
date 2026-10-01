# SPDX-License-Identifier: GPL-3.0-or-later
"""Writing landscape-metric results: tables (CSV, Parquet, GeoPackage...) and
zonal metrics as a GeoDataFrame (one row per zone, with its geometry)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .._table import write_table

__all__ = ["write_table", "zonal_metrics"]

_TABLE_SUFFIXES = {".csv": "csv", ".gz": "csv", ".parquet": "parquet", ".pq": "parquet", ".gpkg": "vector",
                   ".geojson": "vector", ".shp": "vector", ".fgb": "vector"}


class TableSink:
    """A table written chunk by chunk (CSV, ``.csv.gz`` or Parquet), for
    tables too large to build in memory at once.

    Every chunk must have the same columns and types (categoricals with the
    same categories); a Parquet file gets one row group per chunk.
    """

    def __init__(self, path: str | os.PathLike, **kwargs):
        self.path = Path(path)
        self.kind = _TABLE_SUFFIXES.get(self.path.suffix.lower())
        if self.kind not in ("csv", "parquet"):
            raise ValueError(f"chunked tables are written as .csv, .csv.gz or .parquet, got {self.path.suffix!r}")
        self.kwargs = kwargs
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._writer = None
        self._schema = None

    def write(self, frame: pd.DataFrame) -> None:
        if self.kind == "csv":
            if self._handle is None:
                if self.path.suffix.lower() == ".gz":
                    import gzip

                    self._handle = gzip.open(self.path, "wt", newline="", encoding="utf-8")
                else:
                    self._handle = open(self.path, "w", newline="", encoding="utf-8")
                frame.to_csv(self._handle, index=False, **self.kwargs)
            else:
                frame.to_csv(self._handle, index=False, header=False, **self.kwargs)
            return
        pa, pq = _pyarrow()
        if self._writer is None:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.path, self._schema, **self.kwargs)
        else:
            table = pa.Table.from_pandas(frame, schema=self._schema, preserve_index=False)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> TableSink:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("writing Parquet needs pyarrow (pip install pyarrow)") from err
    return pa, pq


def _zones(zones):
    """(GeoDataFrame or None, list of polygon geometries, CRS)."""
    try:
        import geopandas as gpd
    except ImportError as err:  # pragma: no cover
        raise ImportError("zonal_metrics needs geopandas (pip install geopandas)") from err
    if isinstance(zones, gpd.GeoDataFrame):
        return zones, list(zones.geometry), zones.crs
    if isinstance(zones, gpd.GeoSeries):
        return gpd.GeoDataFrame(geometry=zones), list(zones), zones.crs
    from shapely.geometry import shape

    from .sampling import _geometries

    geoms = [shape(g) for g in _geometries(zones)]
    return gpd.GeoDataFrame(geometry=geoms), geoms, None


def zonal_metrics(
    data,
    zones,
    level="landscape",
    metric=None,
    name=None,
    type=None,
    what=None,
    *,
    extras: bool = False,
    separator: str = "_",
    **options,
):
    """Landscape and class metrics of every zone, as a GeoDataFrame.

    The zones are cut out of the raster as in :func:`sample_metrics` (same
    values), and the results come back in wide form next to the zones'
    geometry and attributes: one row per zone, one column per landscape-level
    metric and one per class-level metric and class (``<metric>_<class>``),
    plus ``percentage_inside``. Write it with :func:`write_table` or
    ``.to_file("zones.gpkg")``.

    Parameters
    ----------
    data
        One raster (see :class:`~landschaft.Landscape`).
    zones
        Polygons: a GeoDataFrame / GeoSeries (attributes and CRS are kept),
        shapely polygons or GeoJSON-like geometries, in the raster's
        coordinates.
    level, metric, name, type, what, extras
        Metric selection (class and landscape level), as in
        :func:`landschaft.list_metrics`; landscape level by default.
    separator
        Between metric and class in the class-level column names.
    **options
        :func:`sample_metrics` / :class:`~landschaft.Landscape` options.

    Returns
    -------
    geopandas.GeoDataFrame

    Examples
    --------
    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"))
    >>> zones = ls.grid_zones(landcover, 10000)          # 10 km squares
    >>> table = ls.zonal_metrics(landcover, zones, level="landscape", metric=["shdi"])
    >>> len(table)
    25
    >>> _ = ls.write_table(table, "zones.gpkg")
    """
    import geopandas as gpd

    from .sampling import sample_metrics

    frame, geoms, crs = _zones(zones)
    if any(g.geom_type not in ("Polygon", "MultiPolygon") for g in geoms):
        raise ValueError("zones must be polygons")
    options.setdefault("verbose", False)
    ids = np.arange(len(geoms))
    result = sample_metrics(data, [g.__geo_interface__ for g in geoms], plot_id=ids, level=level, metric=metric,
                            name=name, type=type, what=what, extras=extras, **options)
    if "layer" in result.columns:
        raise ValueError("zonal_metrics takes one raster")
    if (result["level"] == "patch").any():
        raise ValueError("zonal_metrics gives class and landscape metrics (patch metrics: sample_metrics)")
    out = pd.DataFrame(index=ids)
    land = result[result["level"] == "landscape"]
    if not land.empty:
        land = land.assign(metric=land["metric"].astype(str))
        out = out.join(land.groupby(["plot_id", "metric"], sort=False)["value"].first().unstack("metric"))
    cls = result[(result["level"] == "class") & result["class"].notna()]
    if not cls.empty:
        cls = cls.assign(column=cls["metric"].astype(str) + separator + cls["class"].astype(str))
        wide = cls.groupby(["plot_id", "column"], sort=False)["value"].first().unstack("column")
        out = out.join(wide)
    pct = result.groupby("plot_id")["percentage_inside"].first()
    out["percentage_inside"] = pct.reindex(ids).to_numpy()
    out.columns = [str(c) for c in out.columns]
    attrs = frame.drop(columns=frame.geometry.name).reset_index(drop=True)
    clash = set(attrs.columns) & set(out.columns)
    if clash:
        raise ValueError(f"zone attributes clash with metric columns: {sorted(clash)}")
    combined = pd.concat([attrs, out.reset_index(drop=True)], axis=1)
    return gpd.GeoDataFrame(combined, geometry=list(frame.geometry), crs=crs)
