# SPDX-License-Identifier: GPL-3.0-or-later
"""Vector input/output (GeoPackage, Shapefile, GeoJSON, GeoParquet, coordinate tables) and
rasterization on the grid of a raster."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

__all__ = ["load_vector", "rasterize", "write_vector"]

_TABLE_SUFFIXES = {".csv", ".txt", ".tsv", ".gz", ".xlsx", ".xls", ".json"}
_X_NAMES = ("x", "lon", "long", "lng", "longitude", "decimallongitude")
_Y_NAMES = ("y", "lat", "latitude", "decimallatitude")
_LONLAT = {"lon", "long", "lng", "longitude", "decimallongitude", "lat", "latitude", "decimallatitude"}
_VECTOR_DRIVERS = {".gpkg": "GPKG", ".shp": "ESRI Shapefile", ".geojson": "GeoJSON", ".json": "GeoJSON",
                   ".fgb": "FlatGeobuf", ".kml": "KML", ".gml": "GML", ".csv": "CSV"}


def _gpd():
    try:
        import geopandas as gpd
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("vector data need geopandas (pip install geopandas)") from err
    return gpd


def _crs_of(like):
    """The CRS of a raster or vector (object or path)."""
    if like is None:
        return None
    if isinstance(like, xr.DataArray):
        return like.rio.crs
    if hasattr(like, "crs"):
        return like.crs
    import rasterio

    try:
        with rasterio.open(like) as src:
            return src.crs
    except rasterio.errors.RasterioIOError:
        import pyogrio

        return pyogrio.read_info(os.fspath(like))["crs"]


def _coordinate_column(columns, given, names, axis):
    if given is not None:
        if given not in columns:
            raise ValueError(f"no column {given!r}; columns: {', '.join(map(str, columns))}")
        return given
    lower = {str(c).lower(): c for c in columns}
    for n in names:
        if n in lower:
            return lower[n]
    raise ValueError(f"no {axis} coordinate column found (looked for {', '.join(names)}); pass {axis}=")


def load_vector(source, *, layer=None, like=None, x: str | None = None, y: str | None = None, crs=None,
                columns=None, bbox=None):
    """Read a vector layer (polygons, lines, points) as a GeoDataFrame.

    Any format GDAL reads (GeoPackage, Shapefile, GeoJSON, FlatGeobuf, KML...),
    GeoParquet, or a table of point coordinates (CSV, Excel, Parquet): the
    coordinate columns are found by name (``x``/``y``, ``lon``/``lat``,
    ``longitude``/``latitude``, Darwin Core's ``decimalLongitude``...).

    Parameters
    ----------
    source
        File path (or a GDAL URL).
    layer
        Layer of a multi-layer file (GeoPackage...), by name or number.
    like
        A raster or vector (object or path) whose CRS the result is reprojected
        to, so that it overlays the rasters it will be used with.
    x, y
        Coordinate columns of a table (default: found by name).
    crs
        CRS of the coordinates of a table (default: longitude/latitude, EPSG:4326,
        for columns named like longitude and latitude, else none).
    columns
        Attribute columns to read (default: all).
    bbox
        ``(xmin, ymin, xmax, ymax)``: only the features intersecting it, in the
        CRS of the file.

    Returns
    -------
    geopandas.GeoDataFrame

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> sites = pd.DataFrame({"name": ["a", "b"], "longitude": [10.1, 10.4], "latitude": [52.3, 52.6]})
    >>> _ = ls.write_table(sites, "sites.csv")
    >>> points = ls.load_vector("sites.csv")                  # columns found by name, EPSG:4326
    >>> points.crs.to_epsg(), len(points)
    (4326, 2)
    >>> landcover = ls.load_raster(ls.example_data("clc"))
    >>> ls.load_vector("sites.csv", like=landcover).crs.to_epsg()    # on the raster's CRS
    3035
    """
    gpd = _gpd()
    path = Path(os.fspath(source))
    suffix = path.suffix.lower()
    if suffix in _TABLE_SUFFIXES or x is not None or y is not None:
        from ._table import load_table

        table = load_table(source, columns=None) if suffix in _TABLE_SUFFIXES else None
        if table is None:
            import pandas as pd

            table = pd.read_parquet(source)
        xc = _coordinate_column(table.columns, x, _X_NAMES, "x")
        yc = _coordinate_column(table.columns, y, _Y_NAMES, "y")
        if crs is None and str(xc).lower() in _LONLAT and str(yc).lower() in _LONLAT:
            crs = "EPSG:4326"
        if columns is not None:
            table = table[[*columns, *(c for c in (xc, yc) if c not in columns)]]
        gdf = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table[xc], table[yc]), crs=crs)
    elif suffix in (".parquet", ".geoparquet", ".pq"):
        gdf = gpd.read_parquet(source, columns=columns, bbox=bbox)
    else:
        options: dict[str, Any] = {"engine": "pyogrio"} if importlib.util.find_spec("pyogrio") else {}
        if options and importlib.util.find_spec("pyarrow"):
            options["use_arrow"] = True
        gdf = gpd.read_file(source, layer=layer, columns=columns, bbox=bbox, **options)
    if like is not None:
        target = _crs_of(like)
        if target is not None and gdf.crs is not None and gdf.crs != target:
            gdf = gdf.to_crs(target)
    return gdf


def write_vector(data, path, **kwargs) -> Path:
    """Write a vector layer, the format given by the file extension.

    ``.gpkg`` (GeoPackage), ``.shp``, ``.geojson``, ``.fgb`` (FlatGeobuf),
    ``.kml``, ``.parquet`` (GeoParquet), ``.csv`` (geometry as WKT). Missing
    folders are created.

    Parameters
    ----------
    data
        A GeoDataFrame or GeoSeries (e.g. from :func:`load_vector` or
        :func:`landschaft.zonal_metrics`).
    path
        Output file.
    **kwargs
        Passed to geopandas (``layer=`` for a GeoPackage...).

    Returns
    -------
    pathlib.Path

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import box
    >>> import landschaft as ls
    >>> reserves = gpd.GeoDataFrame({"name": ["north"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:3035")
    >>> ls.write_vector(reserves, "out/reserves.gpkg").name
    'reserves.gpkg'
    >>> ls.load_vector("out/reserves.gpkg")["name"].tolist()
    ['north']
    """
    gpd = _gpd()
    if isinstance(data, gpd.GeoSeries):
        data = data.to_frame()
    if not isinstance(data, gpd.GeoDataFrame):
        raise TypeError("write_vector needs a GeoDataFrame or GeoSeries; use write_table for tables")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".geoparquet", ".pq"):
        data.to_parquet(path, **kwargs)
    elif suffix == ".csv":
        import pandas as pd

        geom = data.geometry.name
        out = pd.DataFrame(data.drop(columns=geom))
        out[geom] = data.geometry.to_wkt()
        out.to_csv(path, index=False, **kwargs)
    else:
        driver = kwargs.pop("driver", None) or _VECTOR_DRIVERS.get(suffix)
        if driver is None:
            raise ValueError(f"cannot tell the vector format of {path.name!r}; pass driver=")
        if importlib.util.find_spec("pyogrio"):
            kwargs.setdefault("engine", "pyogrio")
        data.to_file(path, driver=driver, **kwargs)
    return path


def _as_geodataframe(vector):
    gpd = _gpd()
    if isinstance(vector, gpd.GeoDataFrame):
        return vector
    if isinstance(vector, gpd.GeoSeries):
        return gpd.GeoDataFrame(geometry=vector)
    if isinstance(vector, (str, os.PathLike)):
        return load_vector(vector)
    geoms = [vector] if hasattr(vector, "geom_type") else list(vector)
    return gpd.GeoDataFrame(geometry=geoms)


def _rasterize_dtype(values: np.ndarray, fill) -> np.dtype:
    """The smallest type rasterio burns that holds the values and the fill."""
    fill_float = isinstance(fill, float) and not float(fill).is_integer()
    if np.issubdtype(values.dtype, np.floating) or fill_float or (isinstance(fill, float) and np.isnan(fill)):
        if np.issubdtype(values.dtype, np.floating) and values.dtype.itemsize > 4:
            return np.dtype(np.float64)
        return np.dtype(np.float32)
    allv = np.concatenate([values.astype(np.int64), [int(fill)]]) if values.size else np.array([int(fill)])
    lo, hi = int(allv.min()), int(allv.max())
    for dt in (np.uint8, np.uint16, np.int16, np.uint32, np.int32):
        info = np.iinfo(dt)
        if info.min <= lo and hi <= info.max:
            return np.dtype(dt)
    return np.dtype(np.float64)


def _burn(arr, geoms, values, tr, all_touched: bool, add: bool, gdal) -> None:
    """Burns geometries into arr: polygons by the C++ port of GDAL's rasterizer (the same
    cells, in parallel), points by their cell, lines by GDAL. Polygons first, then lines,
    then points; within each, the later features win (or every value adds up)."""
    import shapely

    from . import _core

    kinds = shapely.get_type_id(geoms)  # 0 point, 1 line, 2 ring, 3 polygon, 4 multipoint, 5 multiline, 6 multipolygon
    polys = np.isin(kinds, (3, 6))
    if polys.any() and not (tr.b or tr.d):
        from ._zonal import _polygon_arrays

        xy, ring_off, geom_rings = _polygon_arrays(geoms[polys])
        runs = _core.io_polygon_runs(xy, ring_off, geom_rings, (tr.c, tr.a, tr.b, tr.f, tr.d, tr.e), arr.shape[1],
                                     arr.shape[0], all_touched)
        _core.io_burn_runs(arr, runs, values[polys], add)
        done = polys
    else:
        done = np.zeros(len(geoms), bool)
    lines = ~done & ~np.isin(kinds, (0, 4))
    if lines.any():
        gdal(geoms[lines], values[lines])
    points = np.isin(kinds, (0, 4))
    if points.any():
        parts, index = shapely.get_parts(geoms[points], return_index=True)
        xy = shapely.get_coordinates(parts)
        vals = values[points][index]
        col = np.floor((xy[:, 0] - tr.c) / tr.a).astype(np.int64)
        row = np.floor((xy[:, 1] - tr.f) / tr.e).astype(np.int64)
        ok = (col >= 0) & (col < arr.shape[1]) & (row >= 0) & (row < arr.shape[0])
        flat = row[ok] * arr.shape[1] + col[ok]
        if add:
            np.add.at(arr.reshape(-1), flat, vals[ok].astype(arr.dtype))
        else:  # the last point of every cell wins
            last = len(flat) - 1 - np.unique(flat[::-1], return_index=True)[1]
            arr.reshape(-1)[flat[last]] = vals[ok][last].astype(arr.dtype)


def rasterize(vector, like, *, field: str | None = None, value=1, fill=0, all_touched: bool = False,
              merge: str = "replace", dtype=None, name: str | None = None) -> xr.DataArray:
    """Burn vector features into a raster on the grid of another raster.

    Protected areas, municipalities, roads or sample sites become a raster
    that lines up cell by cell with ``like``: a mask (1 inside, 0 outside), the
    values of an attribute, or the number of overlapping features. The
    features are reprojected to the CRS of ``like`` first.

    Parameters
    ----------
    vector
        GeoDataFrame, GeoSeries, a list of shapely geometries or a vector path.
    like
        The grid: a DataArray or a raster path.
    field
        Attribute whose values are burnt (default: ``value`` for every feature).
    value
        The value burnt when ``field`` is not given (default 1).
    fill
        The value of the cells no feature covers (default 0; ``np.nan`` for a
        float raster with NoData outside).
    all_touched
        Burn every cell a feature touches (default: the cells whose centre is
        inside, which does not overestimate small polygons).
    merge
        Where features overlap: ``"replace"`` (the last one wins) or ``"add"``
        (the values are summed, e.g. to count overlapping features).
    dtype
        Output type (default: the smallest that holds the values and ``fill``).
    name
        Name of the result (default: ``field``, else ``"rasterized"``).

    Returns
    -------
    xarray.DataArray
        On the grid of ``like``, with its coordinates and CRS.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import box
    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"))
    >>> x0, y0 = float(landcover.x[0]), float(landcover.y[-1])
    >>> reserves = gpd.GeoDataFrame({"id": [1, 2]}, crs=landcover.rio.crs,
    ...                             geometry=[box(x0, y0, x0 + 5000, y0 + 5000), box(x0 + 20000, y0, x0 + 30000, y0 + 8000)])
    >>> mask = ls.rasterize(reserves, like=landcover)          # 1 inside the reserves, 0 outside
    >>> str(mask.dtype), int(mask.sum()) * 25 * 25 / 1e6       # km2
    ('uint8', 105.375)
    >>> ls.rasterize(reserves, like=landcover, field="id").max().item()
    2
    """
    import shapely
    from rasterio.enums import MergeAlg
    from rasterio.features import rasterize as burn

    from .io import _grid_template, _transform

    if merge not in ("replace", "add"):
        raise ValueError("merge must be 'replace' or 'add'")
    gdf = _as_geodataframe(vector)
    template = _grid_template(like)
    target = template.rio.crs
    if gdf.crs is not None and target is not None and gdf.crs != target:
        gdf = to_crs(gdf, target)
    gdf = gdf[~shapely.is_missing(gdf.geometry.values) & ~shapely.is_empty(gdf.geometry.values)]
    if field is not None:
        if field not in gdf.columns:
            raise ValueError(f"no column {field!r}; columns: {', '.join(map(str, gdf.columns))}")
        gdf = gdf[gdf[field].notna()]
        values = gdf[field].to_numpy()
        if values.dtype == bool:
            values = values.astype(np.uint8)
    else:
        values = np.full(len(gdf), value)
    out_dtype = np.dtype(dtype) if dtype is not None else _rasterize_dtype(np.asarray(values), fill)
    shape = (template.sizes["y"], template.sizes["x"])
    tr = _transform(template)
    arr = np.full(shape, fill, dtype=out_dtype)
    if len(gdf):
        _burn(arr, gdf.geometry.values, np.asarray(values, dtype=np.float64), tr, all_touched, merge == "add",
              lambda geoms, vals: burn(zip(geoms, vals.astype(out_dtype)), out_shape=shape, transform=tr, fill=fill,
                                       all_touched=all_touched, out=arr,
                                       merge_alg=MergeAlg.add if merge == "add" else MergeAlg.replace,
                                       dtype=out_dtype))
    da = xr.DataArray(arr, coords={"y": template["y"].values, "x": template["x"].values}, dims=("y", "x"),
                      name=name or field or "rasterized")
    da = da.rio.write_crs(target).rio.write_transform(_transform(template))
    if isinstance(fill, float) and np.isnan(fill):
        da = da.rio.write_nodata(np.nan, encoded=False)
    return da


def transform_xy(x: np.ndarray, y: np.ndarray, src, dst="EPSG:3857"):
    """Coordinates reprojected by pyproj in blocks over threads (it releases the GIL; one
    Transformer per block, as they are not thread-safe)."""
    from concurrent.futures import ThreadPoolExecutor

    from pyproj import Transformer

    from ._threads import default_threads

    n = len(x)
    blocks = max(1, min(default_threads(), n // 100_000))
    if blocks == 1:
        return Transformer.from_crs(src, dst, always_xy=True).transform(x, y)
    edges = np.linspace(0, n, blocks + 1).astype(np.int64)
    ox, oy = np.empty(n), np.empty(n)

    def run(k):
        a, b = edges[k], edges[k + 1]
        ox[a:b], oy[a:b] = Transformer.from_crs(src, dst, always_xy=True).transform(x[a:b], y[a:b])

    with ThreadPoolExecutor(blocks) as pool:
        list(pool.map(run, range(blocks)))
    return ox, oy


def to_crs(gdf, crs):
    """gdf.to_crs(crs), its coordinates reprojected in parallel (the same coordinates)."""
    import shapely

    if gdf.crs is None or gdf.crs == crs or len(gdf) < 20_000:
        return gdf.to_crs(crs) if gdf.crs is not None and gdf.crs != crs else gdf
    geoms = gdf.geometry.values
    xy = shapely.get_coordinates(geoms)
    x, y = transform_xy(xy[:, 0], xy[:, 1], gdf.crs, crs)
    out = shapely.set_coordinates(np.array(geoms, dtype=object).copy(), np.column_stack([x, y]))
    import geopandas as gpd

    return gpd.GeoDataFrame(gdf.drop(columns=gdf.geometry.name), geometry=out, crs=crs)
