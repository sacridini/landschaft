# SPDX-License-Identifier: GPL-3.0-or-later
"""Writing landscape-metric results: tables (CSV, Parquet, GeoPackage...) and
zonal metrics as a GeoDataFrame (one row per zone, with its geometry)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["write_table", "zonal_metrics"]

_TABLE_SUFFIXES = {".csv": "csv", ".gz": "csv", ".parquet": "parquet", ".pq": "parquet", ".gpkg": "vector",
                   ".geojson": "vector", ".shp": "vector", ".fgb": "vector"}


def write_table(table: pd.DataFrame, path: str | os.PathLike, **kwargs) -> Path:
    """Write a metric table, the format given by the file extension.

    ``.csv`` (``.csv.gz`` compressed), ``.parquet`` (needs pyarrow; keeps the
    categorical and nullable integer columns, and is compact and fast for
    patch tables of millions of rows), or, for a GeoDataFrame (e.g. from
    :func:`zonal_metrics`), a vector format: ``.gpkg``, ``.geojson``,
    ``.shp``, ``.fgb``. Extra keyword arguments go to the pandas / geopandas
    writer.
    """
    path = Path(path)
    kind = _TABLE_SUFFIXES.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"unknown table format {path.suffix!r}; use .csv, .csv.gz, .parquet, .gpkg, .geojson...")
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "csv":
        table.to_csv(path, index=False, **kwargs)
    elif kind == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError as err:  # pragma: no cover - depends on the environment
            raise ImportError("writing Parquet needs pyarrow (pip install pyarrow)") from err
        table.to_parquet(path, index=False, **kwargs)
    else:
        if not hasattr(table, "to_file"):
            raise TypeError(f"{path.suffix} needs a GeoDataFrame (see zonal_metrics)")
        table.to_file(path, **kwargs)
    return path


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
