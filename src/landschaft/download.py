# SPDX-License-Identifier: GPL-3.0-or-later
"""Data downloaded on demand onto a grid: the climate of WorldClim 2.1 and CHELSA 2.1 (now and under the CMIP6
scenarios), the Copernicus DEM, SoilGrids, MapBiomas, Hydrography90m and the scenes of a STAC catalogue, or any
raster a URL names; and the environment of Hydrography90m's sub-catchments from Environment90m.

The rasters are read by GDAL straight from the servers, only the part the grid covers (their files are
cloud-optimised, zipped or virtual), and warped onto the grid of a map (``like``) or onto a grid of longitudes and
latitudes (``extent`` and ``cellsize``): a file per layer kept in the cache (``LANDSCHAFT_CACHE``), downloaded once.
Classes are resampled by the most common cell, values bilinearly. The values are the servers' (CHELSA's scaled
integers: × 0.1, and - 273.15 for temperatures; SoilGrids' in its mapped units).

Typical use::

    import landschaft as ls

    forest = ls.load_raster("forest.tif")
    clim = ls.download.worldclim(forest, layers=[1, 12])          # bio01 and bio12 on the forest's grid
    dem = ls.download.copernicus_dem(forest)
    future = ls.download.chelsa(forest, gcm="MPI-ESM1-2-HR", ssp=370, period="2041-2070")
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import _download
from ._save import Saveable
from ._threads import resolve

__all__ = ["H90M_VARIABLES", "Environment90m", "chelsa", "copernicus_dem", "environment90m", "hydrography90m",
           "mapbiomas", "raster", "soilgrids", "stac", "worldclim"]


# -- the sources' paths -----------------------------------------------------------------------------------------------

@dataclass
class _Layer:
    """A layer to fetch: its name (the file's), the GDAL paths of its sources (tiles warped one after another), the
    band and whether it holds classes."""

    name: str
    paths: list[str]
    band: int = 1
    classes: bool = False
    key: str = ""


def _curl(url: str) -> str:
    return f"/vsicurl/{url}"


def _numbers(v, all_: range) -> list[int]:
    if v is None:
        return list(all_)
    return [int(x) for x in np.atleast_1d(v)]


def _worldclim(variable: str, resolution: str, layers: list[int]) -> list[_Layer]:
    base = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base"
    if variable == "elev":
        f = f"wc2.1_{resolution}_elev"
        return [_Layer("elev", [f"/vsizip/{_curl(f'{base}/{f}.zip')}/{f}.tif"])]
    z = _curl(f"{base}/wc2.1_{resolution}_{variable}.zip")
    out = []
    for i in layers:
        file = f"wc2.1_{resolution}_bio_{i}" if variable == "bio" else f"wc2.1_{resolution}_{variable}_{i:02d}"
        out.append(_Layer(f"{variable}{i:02d}", [f"/vsizip/{z}/{file}.tif"]))
    return out


def _worldclim_future(variable, resolution, gcm, ssp, period, layers) -> list[_Layer]:
    f = (f"https://geodata.ucdavis.edu/cmip6/{resolution}/{gcm}/{ssp}/"
         f"wc2.1_{resolution}_{variable}_{gcm}_{ssp}_{period}.tif")
    return [_Layer(f"{variable}{i:02d}", [_curl(f)], band=i) for i in layers]


def _chelsa(layers, future=None) -> list[_Layer]:
    base = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies"
    out = []
    for i in layers:
        if future is None:
            url = f"{base}/1981-2010/bio/CHELSA_bio{i}_1981-2010_V.2.1.tif"
        else:
            gcm, ssp, period = future
            url = f"{base}/{period}/{gcm.upper()}/{ssp}/bio/CHELSA_bio{i}_{period}_{gcm.lower()}_{ssp}_V.2.1.tif"
        out.append(_Layer(f"bio{i:02d}", [_curl(url)]))
    return out


def _copernicus_tiles(resolution: int, lon, lat) -> list[str]:
    code, bucket = (30, "copernicus-dem-90m") if resolution == 90 else (10, "copernicus-dem-30m")
    out = []
    for la in range(int(np.floor(lat[0])), int(np.ceil(lat[1]))):
        for lo in range(int(np.floor(lon[0])), int(np.ceil(lon[1]))):
            ns = f"S{-la:02d}" if la < 0 else f"N{la:02d}"
            ew = f"W{-lo:03d}" if lo < 0 else f"E{lo:03d}"
            name = f"Copernicus_DSM_COG_{code}_{ns}_00_{ew}_00_DEM"
            out.append(_curl(f"https://{bucket}.s3.amazonaws.com/{name}/{name}.tif"))
    return out


def _soilgrids(prop: str, depths: list[str], stat: str) -> list[_Layer]:
    return [_Layer(f"{prop}_{d}_{stat}", [_curl(f"https://files.isric.org/soilgrids/latest/data/{prop}/"
                                                f"{prop}_{d}_{stat}.vrt")]) for d in depths]


def _mapbiomas(collection: int, years: list[int]) -> list[_Layer]:
    base = "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil"
    out = []
    for y in years:
        if collection == 9:
            url = f"{base}/collection_9/lclu/coverage/brasil_coverage_{y}.tif"
        elif collection == 10:
            url = f"{base}/collection_10/lulc/coverage/brazil_coverage_{y}.tif"
        elif collection == 11:
            url = f"{base}/collection11/lulc/coverage/brazil_coverage/brazil_coverage-col11_{y}.tif"
        else:
            raise ValueError(f"collection: 9, 10 or 11, not {collection}")
        out.append(_Layer(str(y), [_curl(url)], classes=True))
    return out


#: Hydrography90m's files on the IGB Berlin's server, by their path.
_H90M = "https://public.igb-berlin.de/index.php/s/agciopgzXjWswF4/download?path="

#: Hydrography90m's variables in tiles (hydrographr's ``download_tiles``), by the folder that holds them.
H90M_VARIABLES = {
    "r.watershed": ("sub_catchment", "segment", "basin", "outlet", "direction", "accumulation", "depression"),
    "r.stream.order": ("order_strahler", "order_shreve", "order_horton", "order_hack", "order_topo"),
    "r.stream.slope": ("slope_grad_dw_cel", "slope_elv_dw_cel", "slope_curv_max_dw_cel", "slope_curv_min_dw_cel"),
    "r.stream.distance": ("stream_dist_up_near", "stream_dist_up_farth", "stream_dist_dw_near",
                          "stream_dist_proximity", "outlet_dist_dw_basin", "outlet_dist_dw_scatch",
                          "stream_diff_up_near", "stream_diff_up_farth", "stream_diff_dw_near",
                          "outlet_diff_dw_basin", "outlet_diff_dw_scatch"),
    "r.stream.channel": ("channel_grad_dw_seg", "channel_grad_up_seg", "channel_grad_up_cel", "channel_curv_cel",
                         "channel_elv_dw_seg", "channel_elv_up_seg", "channel_elv_up_cel", "channel_elv_dw_cel",
                         "channel_dist_dw_seg", "channel_dist_up_seg", "channel_dist_up_cel"),
    "flow.index": ("cti", "spi", "sti"),
}
_H90M_CLASSES = ("sub_catchment", "segment", "basin", "outlet", "direction", "depression")


def _h90m_tiles(lon, lat) -> list[str]:
    """The 20° tiles of Hydrography90m and Environment90m over a box of longitudes and latitudes: hXXvYY, h00v00 the
    one from 180° W and 85° N, their numbers stepping by 2 every 20°."""
    def span(a, b, n):
        return (int(np.clip(np.floor(a / 20.0), 0, n - 1)), int(np.clip(int(np.ceil(b / 20.0)) - 1, 0, n - 1)))

    c0, c1 = span(lon[0] + 180.0, lon[1] + 180.0, 18)
    r0, r1 = span(85.0 - lat[1], 85.0 - lat[0], 7)
    return [f"h{2 * c:02d}v{2 * r:02d}" for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _h90m(variable: str, tiles: list[str]) -> tuple[list[str], bool]:
    folder = next((f for f, vs in H90M_VARIABLES.items() if variable in vs), None)
    if folder is None:
        names = ", ".join(v for vs in H90M_VARIABLES.values() for v in vs)
        raise ValueError(f"hydrography90m: variable {variable!r} is not one of {names}")
    classes = folder == "r.stream.order" or variable in _H90M_CLASSES
    paths = [_curl(_H90M + f"/{folder}/{variable}_tiles20d/{variable}_{t}.tif".replace("/", "%2F")) for t in tiles]
    return paths, classes


# -- the grid ---------------------------------------------------------------------------------------------------------

@dataclass
class _Grid:
    crs: object
    transform: object
    width: int
    height: int
    like: xr.DataArray | None = None
    key: str = field(default="", init=False)

    def __post_init__(self):
        from rasterio.crs import CRS

        self.crs = CRS.from_user_input(self.crs)
        self.key = f"{tuple(self.transform)[:6]}|{self.width}|{self.height}|{self.crs.to_wkt()}"


def _grid_of(like, extent, cellsize) -> _Grid:
    from affine import Affine

    if like is not None:
        from .io import _transform

        if not isinstance(like, xr.DataArray):
            raise TypeError("like: a raster (a DataArray with coordinates, as ls.load_raster gives)")
        if like.rio.crs is None:
            raise ValueError("like has no coordinate reference system (rio.write_crs gives it one)")
        return _Grid(like.rio.crs, _transform(like), like.sizes["x"], like.sizes["y"], like)
    if extent is None or cellsize is None:
        raise ValueError("give a map whose grid to fill (like), or extent and cellsize")
    west, south, east, north = (float(v) for v in extent)
    res = float(cellsize)
    if not res > 0 or east <= west or north <= south:
        raise ValueError("extent: (west, south, east, north), degrees; cellsize: degrees, above 0")
    w, h = max(round((east - west) / res), 1), max(round((north - south) / res), 1)
    return _Grid("EPSG:4326", Affine(res, 0.0, west, 0.0, -res, north), w, h)


def _lonlat_box(g: _Grid) -> tuple[list[float], list[float]]:
    """The longitudes and latitudes a grid spans (its edges sampled, in WGS 84)."""
    from rasterio.warp import transform

    f = np.linspace(0.0, 1.0, 21)
    cols = np.concatenate([f * g.width, f * g.width, np.zeros(21), np.full(21, g.width)])
    rows = np.concatenate([np.zeros(21), np.full(21, g.height), f * g.height, f * g.height])
    t = g.transform
    xs, ys = t.c + cols * t.a + rows * t.b, t.f + cols * t.d + rows * t.e
    if not g.crs.is_geographic or g.crs.to_epsg() != 4326:
        xs, ys = transform(g.crs, "EPSG:4326", list(xs), list(ys))
    return [float(np.min(xs)), float(np.max(xs))], [float(np.min(ys)), float(np.max(ys))]


# -- reading and warping ----------------------------------------------------------------------------------------------

_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "2"}

# what curl says when the server cannot be reached at all
_UNREACHABLE = ("could not resolve host", "couldn't resolve host", "couldn't connect", "failed to connect",
                "connection timed out", "timeout was reached", "network is unreachable")


def _exists(path: str) -> bool:
    import rasterio
    from rasterio.errors import RasterioIOError

    try:
        with rasterio.Env(**_ENV), rasterio.open(path):
            return True
    except RasterioIOError:
        return False


def _method(resampling: str, classes: bool):
    from rasterio.enums import Resampling

    if resampling == "auto":
        resampling = "mode" if classes else "bilinear"
    name = {"near": "nearest", "med": "med", "cubicspline": "cubic_spline"}.get(resampling, resampling)
    try:
        return Resampling[name]
    except KeyError:
        raise ValueError(f"resampling: auto, near, bilinear, cubic, average, mode... not {resampling!r}") from None


def _warp(paths: list[str], band: int, g: _Grid, method, dtype, n_threads: int) -> np.ndarray:
    """The sources warped onto the grid one after another, each where it has values (gdalwarp's several inputs)."""
    import rasterio
    from rasterio.warp import reproject

    dst = np.full((g.height, g.width), np.nan, dtype=dtype)
    with rasterio.Env(**_ENV):
        for p in paths:
            with rasterio.open(p) as src:
                if src.crs is None:
                    raise ValueError(f"{p}: the raster has no coordinate reference system")
                reproject(source=rasterio.band(src, band), destination=dst, src_nodata=src.nodata,
                          dst_transform=g.transform, dst_crs=g.crs, dst_nodata=np.nan, resampling=method,
                          init_dest_nodata=False, num_threads=n_threads)
    return dst


def _write(path: Path, a: np.ndarray, g: _Grid) -> None:
    import rasterio

    tmp = path.with_name(f".{path.stem}-{os.getpid()}.tif")
    profile = {"driver": "GTiff", "width": g.width, "height": g.height, "count": 1, "dtype": a.dtype.name,
               "crs": g.crs, "transform": g.transform, "nodata": np.nan, "compress": "deflate"}
    if g.width >= 256 and g.height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(a, 1)
    os.replace(tmp, path)


def _cache_folder(layers: list[_Layer], g: _Grid, resampling: str) -> Path:
    h = hashlib.sha1()
    for layer in layers:
        h.update(f"{layer.paths}|{layer.band}|{layer.classes}|{layer.key}".encode())
    h.update(f"{g.key}|{resampling}".encode())
    d = _download.cache_dir() / "data" / h.hexdigest()[:16]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _as_raster(arrays: list[np.ndarray], names: list[str], g: _Grid, name: str) -> xr.DataArray:
    t = g.transform
    if g.like is not None:
        coords = {"y": g.like["y"].values, "x": g.like["x"].values}
    else:
        coords = {"y": t.f + (np.arange(g.height) + 0.5) * t.e, "x": t.c + (np.arange(g.width) + 0.5) * t.a}
    if len(arrays) == 1:
        da = xr.DataArray(arrays[0], dims=("y", "x"), coords=coords, name=names[0])
    else:
        dtype = np.result_type(*arrays)
        da = xr.DataArray(np.stack([a.astype(dtype) for a in arrays]), dims=("band", "y", "x"),
                          coords={"band": names, **coords}, name=name)
    da = da.rio.write_crs(g.crs).rio.write_transform(t)
    return da.rio.write_nodata(np.nan, encoded=False)


def _fetch(layers: list[_Layer], g: _Grid, resampling: str, refresh: bool, n_threads, name: str) -> xr.DataArray:
    import rasterio

    folder = _cache_folder(layers, g, resampling)
    n = resolve(n_threads)
    arrays = []
    for layer in layers:
        out = folder / f"{layer.name}.tif"
        if out.is_file() and not refresh:
            with rasterio.open(out) as src:
                arrays.append(src.read(1))
            continue
        dtype = np.float64 if layer.classes else np.float32
        try:
            a = _warp(layer.paths, layer.band, g, _method(resampling, layer.classes), dtype, n)
        except Exception as err:
            if _download.connection_failed(err) or any(m in str(err).lower() for m in _UNREACHABLE):
                raise OSError(_download.offline_message(f"the server of {layer.name!r}") + f" ({err})") from err
            raise
        _write(out, a, g)
        arrays.append(a)
    return _as_raster(arrays, [layer.name for layer in layers], g, name)


# -- the sources ------------------------------------------------------------------------------------------------------

def _ssp(ssp) -> str:
    return "ssp" + str(ssp).removeprefix("ssp")


def worldclim(like=None, *, variable: str = "bio", layers=None, resolution: str = "10m", gcm: str | None = None,
              ssp=245, period: str = "2041-2060", extent=None, cellsize=None, resampling: str = "auto",
              refresh: bool = False, n_threads: int | None = None) -> xr.DataArray:
    """WorldClim 2.1's climate on a grid: the historical climate (1970-2000) or a CMIP6 future.

    Parameters
    ----------
    like
        The map whose grid to fill (else ``extent`` and ``cellsize``).
    variable
        Historical: ``bio`` (the 19 bioclimatic variables), ``tavg``,
        ``tmin``, ``tmax``, ``prec``, ``srad``, ``wind``, ``vapr`` (12 months)
        or ``elev``. A future (``gcm`` given): ``bioc``, ``tmin``, ``tmax``,
        ``prec``.
    layers
        The variables' numbers (bio 1-19, months 1-12; default all).
    resolution
        ``10m``, ``5m``, ``2.5m`` or ``30s``.
    gcm, ssp, period
        A future: the climate model (e.g. ``"MPI-ESM1-2-HR"``), the scenario
        (126, 245, 370, 585) and the years (2021-2040, 2041-2060, 2061-2080,
        2081-2100).
    extent, cellsize
        Without ``like``: (west, south, east, north) and the cells' side, in
        degrees.
    resampling
        ``auto`` (bilinear), or gdalwarp's: near, cubic, average...
    refresh
        Download again although the cache has it.
    n_threads
        GDAL's warping threads.

    Returns
    -------
    xarray.DataArray
        A layer (``bio01``, ``tavg03``...), or a band per layer.

    Examples
    --------
    >>> import landschaft as ls
    >>> clim = ls.download.worldclim(  # doctest: +SKIP
    ...     extent=(-47.2, -23.0, -46.8, -22.7), cellsize=0.05, layers=[1, 12])
    >>> list(clim.band.values)  # doctest: +SKIP
    ['bio01', 'bio12']
    """
    g = _grid_of(like, extent, cellsize)
    if gcm is None:
        if variable == "bioc":
            variable = "bio"
        ls_ = _worldclim(variable, resolution, _numbers(layers, range(1, 20) if variable == "bio" else range(1, 13)))
    else:
        if variable == "bio":
            variable = "bioc"
        all_ = range(1, 20) if variable == "bioc" else range(1, 13)
        ls_ = _worldclim_future(variable, resolution, gcm, _ssp(ssp), period, _numbers(layers, all_))
    return _fetch(ls_, g, resampling, refresh, n_threads, "worldclim")


def chelsa(like=None, *, layers=None, gcm: str | None = None, ssp=370, period: str = "2041-2070", extent=None,
           cellsize=None, resampling: str = "auto", refresh: bool = False,
           n_threads: int | None = None) -> xr.DataArray:
    """CHELSA 2.1's bioclimatic variables on a grid: 1981-2010, or a CMIP6 future.

    The values are CHELSA's scaled integers: × 0.1 (and - 273.15 for the
    temperatures, in tenths of a kelvin) to physical units.

    Parameters
    ----------
    like
        The map whose grid to fill (else ``extent`` and ``cellsize``).
    layers
        The bioclimatic variables' numbers, 1-19 (default all).
    gcm, ssp, period
        A future: the climate model (``GFDL-ESM4``, ``IPSL-CM6A-LR``,
        ``MPI-ESM1-2-HR``, ``MRI-ESM2-0``, ``UKESM1-0-LL``), the scenario (126,
        370, 585) and the years (2011-2040, 2041-2070, 2071-2100).
    extent, cellsize, resampling, refresh, n_threads
        As :func:`worldclim`.

    Returns
    -------
    xarray.DataArray
        A layer (``bio01``...), or a band per layer.
    """
    g = _grid_of(like, extent, cellsize)
    future = None if gcm is None else (gcm, _ssp(ssp), period)
    return _fetch(_chelsa(_numbers(layers, range(1, 20)), future), g, resampling, refresh, n_threads, "chelsa")


def copernicus_dem(like=None, *, resolution: int = 90, extent=None, cellsize=None, resampling: str = "auto",
                   refresh: bool = False, n_threads: int | None = None) -> xr.DataArray:
    """The Copernicus DEM (GLO-30 or GLO-90) on a grid, from its tiles of a degree over the grid.

    Parameters
    ----------
    like
        The map whose grid to fill (else ``extent`` and ``cellsize``).
    resolution
        30 or 90 (m).
    extent, cellsize, resampling, refresh, n_threads
        As :func:`worldclim`.

    Returns
    -------
    xarray.DataArray
        The elevations (m); NaN at sea.
    """
    if resolution not in (30, 90):
        raise ValueError(f"resolution: 30 or 90, not {resolution}")
    g = _grid_of(like, extent, cellsize)
    lon, lat = _lonlat_box(g)
    tiles = [t for t in _copernicus_tiles(resolution, lon, lat) if _exists(t)]
    if not tiles:
        raise ValueError("the Copernicus DEM has no tile over the grid (all sea?)")
    return _fetch([_Layer("elevation", tiles)], g, resampling, refresh, n_threads, "elevation")


def soilgrids(like=None, *, property: str = "soc", depth="0-5cm", stat: str = "mean", extent=None,
              cellsize=None, resampling: str = "auto", refresh: bool = False,
              n_threads: int | None = None) -> xr.DataArray:
    """SoilGrids 2.0's properties on a grid, at depths.

    Parameters
    ----------
    like
        The map whose grid to fill (else ``extent`` and ``cellsize``).
    property
        ``soc`` (organic carbon), ``phh2o``, ``clay``, ``sand``, ``silt``,
        ``bdod``, ``cec``, ``cfvo``, ``nitrogen``, ``ocd``...
    depth
        ``0-5cm``, ``5-15cm``, ``15-30cm``, ``30-60cm``, ``60-100cm``,
        ``100-200cm``, or a list.
    stat
        ``mean``, ``Q0.05``, ``Q0.5``, ``Q0.95``, ``uncertainty``.
    extent, cellsize, resampling, refresh, n_threads
        As :func:`worldclim`.

    Returns
    -------
    xarray.DataArray
        A layer (``soc_0-5cm_mean``), or a band per depth, in SoilGrids'
        mapped units.
    """
    g = _grid_of(like, extent, cellsize)
    depths = [depth] if isinstance(depth, str) else list(depth)
    return _fetch(_soilgrids(property, depths, stat), g, resampling, refresh, n_threads, "soilgrids")


def mapbiomas(like=None, *, years=2023, collection: int = 10, extent=None, cellsize=None, resampling: str = "auto",
              refresh: bool = False, n_threads: int | None = None) -> xr.DataArray:
    """MapBiomas Brazil's land use and cover of years on a grid (collections 9, 10 and 11).

    Parameters
    ----------
    like
        The map whose grid to fill (else ``extent`` and ``cellsize``).
    years
        A year or a list.
    collection
        9, 10 or 11.
    extent, cellsize, resampling, refresh, n_threads
        As :func:`worldclim` (``auto``: the most common class).

    Returns
    -------
    xarray.DataArray
        The classes, a layer (named by its year), or a band per year.
    """
    g = _grid_of(like, extent, cellsize)
    return _fetch(_mapbiomas(collection, _numbers(years, range(2023, 2024))), g, resampling, refresh, n_threads,
                  "mapbiomas")


def hydrography90m(like=None, *, variable: str = "sub_catchment", extent=None, cellsize=None,
                   resampling: str = "auto", refresh: bool = False, n_threads: int | None = None) -> xr.DataArray:
    """Hydrography90m (Amatulli et al. 2022) on a grid: the river network of the world at 90 m.

    Read from its 20° tiles over the grid, as hydrographr's
    ``download_tiles``: the sub-catchments, segments, basins, flow
    directions and accumulation, stream orders, slopes and distances.

    Parameters
    ----------
    like
        The map whose grid to fill (else ``extent`` and ``cellsize``).
    variable
        Hydrographr's names: ``sub_catchment``, ``segment``, ``basin``,
        ``outlet``, ``direction``, ``accumulation``, ``order_strahler``,
        ``slope_grad_dw_cel``, ``stream_dist_up_near``, ``cti``... (see
        :data:`H90M_VARIABLES`).
    extent, cellsize, resampling, refresh, n_threads
        As :func:`worldclim` (``auto``: the most common value for ids,
        directions and orders).

    Returns
    -------
    xarray.DataArray
        The variable.
    """
    g = _grid_of(like, extent, cellsize)
    lon, lat = _lonlat_box(g)
    paths, classes = _h90m(variable, _h90m_tiles(lon, lat))
    tiles = [p for p in paths if _exists(p)]
    if not tiles:
        raise ValueError("Hydrography90m has no tile over the grid (all sea?)")
    return _fetch([_Layer(variable, tiles, classes=classes)], g, resampling, refresh, n_threads, variable)


def raster(url, like=None, *, extent=None, cellsize=None, classes: bool = False, resampling: str = "auto",
           refresh: bool = False, n_threads: int | None = None) -> xr.DataArray:
    """Any raster GDAL reads (a URL, a file), or several, warped onto a grid; only the part the grid covers is read.

    Parameters
    ----------
    url
        A URL (``https://...``: read through ``/vsicurl/``), a GDAL path or a
        file, or a list (a layer each).
    like
        The map whose grid to fill (else ``extent`` and ``cellsize``).
    classes
        The values are classes (``auto`` resampling: the most common).
    extent, cellsize, resampling, refresh, n_threads
        As :func:`worldclim`.

    Returns
    -------
    xarray.DataArray
        A layer, or a band per URL (``layer01``, ``layer02``...).

    Examples
    --------
    >>> import landschaft as ls
    >>> r = ls.download.raster(ls.example_data("input"), ls.load_raster(ls.example_data("input")),
    ...                        resampling="near")
    >>> r.shape
    (1000, 1000)
    """
    urls = [url] if isinstance(url, (str, os.PathLike)) else list(url)
    g = _grid_of(like, extent, cellsize)
    layers = []
    for k, u in enumerate(urls):
        u = os.fspath(u)
        path, key = (_curl(u) if u.startswith(("http://", "https://")) else u), ""
        if not u.startswith(("http://", "https://", "/vsi")) and Path(u).is_file():
            # a file: its cache follows its changes
            path = str(Path(u).resolve())
            st = Path(u).stat()
            key = f"{st.st_size}|{st.st_mtime_ns}"
        layers.append(_Layer("layer" if len(urls) == 1 else f"layer{k + 1:02d}", [path], classes=classes, key=key))
    return _fetch(layers, g, resampling, refresh, n_threads, "layer")


# -- STAC -------------------------------------------------------------------------------------------------------------

def _open_json(req: urllib.request.Request) -> dict:
    import urllib.error

    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as err:
        why = err.read()[:500].decode("utf-8", "replace")
        raise OSError(f"{req.full_url}: HTTP {err.code}: {why}") from err


def _post(url: str, body: dict) -> dict:
    return _open_json(urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                             headers={"Content-Type": "application/json", "User-Agent": "landschaft"}))


def _get(url: str) -> dict:
    return _open_json(urllib.request.Request(url, headers={"User-Agent": "landschaft"}))


def _rfc3339(datetime: str) -> str:
    """STAC's dates (RFC 3339) from dates without a time: a day, or an interval of days (``..`` open)."""
    def one(d: str, end: bool) -> str:
        d = d.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            return d + ("T23:59:59Z" if end else "T00:00:00Z")
        return d

    if "/" in datetime:
        a, b = datetime.split("/", 1)
        return f"{one(a, False)}/{one(b, True)}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", datetime.strip()):
        return f"{one(datetime, False)}/{one(datetime, True)}"
    return datetime


def _search(url: str, body: dict, limit: int) -> list[dict]:
    items: list[dict] = []
    page = _post(url.rstrip("/") + "/search", body)
    while True:
        items.extend(page.get("features", []))
        nxt = next((ln for ln in page.get("links", []) if ln.get("rel") == "next"), None)
        if len(items) >= limit or nxt is None:
            break
        if nxt.get("method", "GET").upper() == "POST":
            page = _post(nxt["href"], {**body, **nxt.get("body", {})} if nxt.get("merge") else nxt.get("body", body))
        else:
            page = _get(nxt["href"])
    return items[:limit]


def stac(url: str, collection: str, like=None, *, assets, datetime: str | None = None, query: dict | None = None,
         limit: int = 50, composite: str = "median", extent=None, cellsize=None, sign=None, classes: bool = False,
         resampling: str = "auto", refresh: bool = False, n_threads: int | None = None) -> xr.DataArray:
    """The scenes of a STAC catalogue over a grid, each asset warped onto it and the scenes combined.

    The catalogue's search (``POST /search``) gives the items of the
    collection over the grid's box of longitudes and latitudes (and the
    dates, and a query on their properties: ``{"eo:cloud_cover": {"lt":
    20}}``); every asset asked for of every item is read (only the part the
    grid covers) and warped onto the grid, and the items are combined cell
    by cell.

    Parameters
    ----------
    url
        The catalogue's API, e.g.
        ``"https://earth-search.aws.element84.com/v1"``.
    collection
        e.g. ``"sentinel-2-l2a"``.
    like
        The map whose grid to fill (else ``extent`` and ``cellsize``).
    assets
        The assets' names (``"red"``, ``"nir"``...), one or a list: a layer
        each.
    datetime
        The dates: a day (``"2024-06-01"``) or an interval
        (``"2024-06-01/2024-08-31"``, ``..`` open), or STAC's own RFC 3339
        times.
    query
        A query on the items' properties (the API's query extension).
    limit
        The largest number of items.
    composite
        ``median``, ``mean``, ``min``, ``max`` (the cells with values of
        every item) or ``first`` (each cell from the first item, in the
        catalogue's order, that has a value).
    sign
        A function that makes an asset's URL readable (a catalogue whose
        files need a token, as Planetary Computer's ``planetary_computer.sign``).
    classes
        The assets are classes (``auto`` resampling: the most common).
    extent, cellsize, resampling, refresh, n_threads
        As :func:`worldclim`.

    Returns
    -------
    xarray.DataArray
        A layer per asset; ``attrs["items"]``: the ids of the items combined.
    """
    if composite not in ("median", "mean", "min", "max", "first"):
        raise ValueError(f"composite: median, mean, min, max or first, not {composite!r}")
    assets = [assets] if isinstance(assets, str) else list(assets)
    g = _grid_of(like, extent, cellsize)
    lon, lat = _lonlat_box(g)
    body: dict = {"collections": [collection], "bbox": [lon[0], lat[0], lon[1], lat[1]], "limit": min(int(limit), 100)}
    if datetime:
        body["datetime"] = _rfc3339(datetime)
    if query:
        body["query"] = query
    try:
        items = _search(url, body, int(limit))
    except Exception as err:
        if _download.connection_failed(err):
            raise OSError(_download.offline_message(f"the STAC catalogue {url}")) from err
        raise
    if not items:
        raise ValueError(f"the catalogue has no item of {collection!r} over the grid")
    n = resolve(n_threads)
    ids = [it.get("id", "") for it in items]
    h = hashlib.sha1(f"{url}|{collection}|{ids}|{assets}|{composite}|{classes}|{g.key}|{resampling}".encode())
    folder = _download.cache_dir() / "data" / h.hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    arrays = []
    import rasterio

    for asset in assets:
        out = folder / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', asset)}.tif"
        if out.is_file() and not refresh:
            with rasterio.open(out) as src:
                arrays.append(src.read(1))
            continue
        method = _method(resampling, classes)
        dtype = np.float64 if classes else np.float32
        scenes = []
        for it in items:
            a = it.get("assets", {}).get(asset)
            if a is None:
                continue
            href = a["href"]
            if sign is not None:
                href = sign(href)
            path = _curl(href) if href.startswith(("http://", "https://")) else href
            scenes.append(_warp([path], 1, g, method, dtype, n))
        if not scenes:
            raise ValueError(f"no item has the asset {asset!r}")
        stack = np.stack(scenes)
        if composite == "first":
            a = stack[0].copy()
            for s in stack[1:]:
                a = np.where(np.isnan(a), s, a)
        else:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                a = {"median": np.nanmedian, "mean": np.nanmean, "min": np.nanmin, "max": np.nanmax}[composite](
                    stack, axis=0).astype(dtype)
        _write(out, a, g)
        arrays.append(a)
    da = _as_raster(arrays, assets, g, collection)
    da.attrs["items"] = ids
    return da


# -- Environment90m ---------------------------------------------------------------------------------------------------

#: Environment90m's files on the IGB Berlin's server, by their path.
_ENV90M = "https://public.igb-berlin.de/index.php/s/zw56kEd25NsQqcQ/download?path="
_ENV90M_LISTS = ("observedclimate", "projectedclimate", "landcover", "soil", "meritdem", "cgiar", "hydro")
_ENV90M_STATS = ("min", "max", "range", "mean", "sd")


def _env90m_url(path: str) -> str:
    return _ENV90M + f"/{path}".replace("/", "%2F")


def _variable_and_tile(path: str):
    """A file's variable and tile, from its name (``<variable>_<tile>.zip``)."""
    name = path.rsplit("/", 1)[-1]
    if not name.endswith(".zip"):
        return None
    var, _, tile = name[:-4].rpartition("_")
    return (var, tile) if var and re.fullmatch(r"h\d\dv\d\d", tile) else None


def _env90m_file(path: str, refresh: bool = False) -> Path:
    return _download.fetch(f"env90m/{path}", _env90m_url(path), refresh=refresh)


def _env90m_catalogue() -> dict[tuple[str, str], str]:
    out = {}
    for name in _ENV90M_LISTS:
        f = _env90m_file(f"_file_lists/env90m_{name}_paths_file_sizes.txt")
        for line in f.read_text(encoding="utf-8").splitlines()[1:]:
            path = line.split(";", 1)[0]
            key = _variable_and_tile(path)
            if key is not None:
                out[key] = path
    return out


def _read_table(text, wanted: set[int], stats: list[str]) -> pd.DataFrame:
    """The rows of a table (``subcID`` and statistics, by spaces; ``na`` missing) of the sub-catchments wanted."""
    parts = []
    reader = pd.read_csv(text, sep=r"\s+", na_values=["na"], chunksize=500_000, dtype={"subcID": np.float64})
    for chunk in reader:
        missing = [s for s in stats if s not in chunk.columns]
        if missing:
            raise ValueError(f"no column {missing[0]!r} (it has {', '.join(chunk.columns)})")
        ids = chunk["subcID"].to_numpy()
        keep = np.isin(ids, list(wanted))
        if keep.any():
            parts.append(chunk.loc[keep, ["subcID", *stats]])
    if not parts:
        return pd.DataFrame(columns=["subcID", *stats])
    out = pd.concat(parts)
    out["subcID"] = out["subcID"].astype(np.int64)
    return out.drop_duplicates("subcID")


@dataclass
class Environment90m(Saveable):
    """The environment of Hydrography90m's sub-catchments (:func:`environment90m`).

    Attributes
    ----------
    table : pandas.DataFrame
        A row per sub-catchment of the map: ``subcID``, then
        ``<variable>_<statistic>``.
    map : xarray.DataArray
        The column ``to_map`` on the cells of each sub-catchment.
    """

    table: pd.DataFrame
    map: xr.DataArray


def environment90m(sub_catchments, variables, *, stats="mean", to_map: str | None = None,
                   refresh: bool = False) -> Environment90m:
    """The environment of every sub-catchment of a map of Hydrography90m's sub-catchments, from Environment90m.

    Environment90m (Garcia Marquez et al. 2025), as hydrographr's
    ``download_env90m_tables``: the climate (CHELSA's bio01 to bio19,
    observed 1981-2010 and projected), the land cover (ESA CCI's share of
    each class by year), the soil (SoilGrids), the elevation (MERIT DEM), the
    aridity and evapotranspiration (CGIAR) and the hydrography's variables,
    each summarised over the sub-catchment, in the units of their sources
    (CHELSA's temperatures in tenths of a kelvin: × 0.1 - 273.15 for degrees).
    The tables of the tiles over the map are downloaded once into the cache
    (some are hundreds of MB), the rows of the map's sub-catchments kept.

    Parameters
    ----------
    sub_catchments
        The sub-catchment of each cell (Hydrography90m's ids: :func:`hydrography90m`).
    variables
        The variables, a name or a list, as hydrographr names them:
        ``bio01_1981-2010_observed``, ``bio12_2071-2100_gfdl-esm4_ssp370_v2_1``,
        ``c10_2020`` (land cover class 10 in 2020), ``clyppt`` (clay),
        ``elev``, ``garid`` (aridity), ``slope_grad_dw_cel``...
    stats
        ``min``, ``max``, ``range``, ``mean``, ``sd``, or a list.
    to_map
        The column of the map (default: the first after ``subcID``).
    refresh
        Download the tables again although the cache has them.

    Returns
    -------
    Environment90m
    """
    variables = [variables] if isinstance(variables, str) else list(variables)
    if not variables:
        raise ValueError("name the variables")
    stats = [s.strip() for s in stats.split(",")] if isinstance(stats, str) else list(stats)
    bad = [s for s in stats if s not in _ENV90M_STATS]
    if bad:
        raise ValueError(f"unknown statistic {bad[0]!r} (min, max, range, mean, sd)")
    from .io import _transform

    if not isinstance(sub_catchments, xr.DataArray) or sub_catchments.rio.crs is None:
        raise ValueError("sub_catchments: a raster with a coordinate reference system (hydrography90m's)")
    v = np.asarray(sub_catchments.values, dtype=np.float64)
    g = _Grid(sub_catchments.rio.crs, _transform(sub_catchments), sub_catchments.sizes["x"],
              sub_catchments.sizes["y"], sub_catchments)
    wanted = {int(x) for x in np.unique(v[np.isfinite(v) & (v > 0)])}
    tiles = _h90m_tiles(*_lonlat_box(g))
    try:
        files = _env90m_catalogue()
    except OSError as err:
        raise OSError(f"Environment90m's lists of files: {err}") from err
    table = pd.DataFrame({"subcID": sorted(wanted)}, dtype=np.int64)
    for var in variables:
        paths = [files[(var, t)] for t in tiles if (var, t) in files]
        if not paths:
            if any(k[0] == var for k in files):
                raise ValueError(f"{var}: Environment90m has no table over the map (all sea?)")
            raise ValueError(f"{var}: not one of Environment90m's variables (as hydrographr names them: "
                             "bio01_1981-2010_observed, c10_2020, clyppt, elev...)")
        found = []
        for path in paths:
            z = _env90m_file(path, refresh)
            inner = path.rsplit("/", 1)[-1].replace(".zip", ".txt")
            with zipfile.ZipFile(z) as zf, zf.open(inner) as f:
                found.append(_read_table(io.TextIOWrapper(f, encoding="utf-8"), wanted, stats))
        got = pd.concat(found).drop_duplicates("subcID").rename(columns={s: f"{var}_{s}" for s in stats})
        table = table.merge(got, on="subcID", how="left")
    to_map = to_map or table.columns[1]
    if to_map not in table:
        raise ValueError(f"to_map: no column {to_map!r}")
    lookup = pd.Series(table[to_map].to_numpy(np.float64), index=table["subcID"].to_numpy())
    ids = np.where(np.isfinite(v) & (v > 0), v, -1).astype(np.int64)
    painted = lookup.reindex(ids.ravel()).to_numpy().reshape(v.shape)
    m = sub_catchments.copy(data=painted.astype(np.float64)).rename(to_map)
    return Environment90m(table, m)
