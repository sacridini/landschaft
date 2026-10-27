# SPDX-License-Identifier: GPL-3.0-or-later
"""Coastal vulnerability (InVEST's ``coastal_vulnerability`` model)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve

__all__ = ["CoastalVulnerability", "coastal_vulnerability"]

#: The 16 sectors of the fetch rays and of the Wave Watch III fields, in compass degrees.
SECTORS = [int(i * 360 / 16) for i in range(16)]
_WWIII = ("REI_PCT", "REI_V", "WavP_", "WavPPCT", "V10PCT_")
_MAX_SURGE = 1.5e6  # metres around the shore points to look for the shelf contour
_MAX_SLR = 500000.0  # metres to the sea-level rise points


@dataclass
class CoastalVulnerability(Saveable):
    """Exposure of the coast to erosion and flooding (:func:`coastal_vulnerability`).

    InVEST's outputs, under its names when saved. Tables have a row per shore
    point (``shore_id``, from 0); ranks go from 1 (least exposed) to 5.

    Attributes
    ----------
    exposure : geopandas.GeoDataFrame
        The shore points (``coastal_exposure``): ``R_hab`` (habitats), ``R_wind``,
        ``R_wave``, ``R_surge``, ``R_relief`` and, with their inputs, ``R_geomorph``,
        ``R_slr`` and ``population`` (people per km²); ``exposure`` (the geometric
        mean of the ``R_`` ranks), ``exposure_no_habitats`` (with ``R_hab`` = 5) and
        ``habitat_role`` (their difference). NaN where a variable has no value.
    intermediate : geopandas.GeoDataFrame
        The values before ranking (``intermediate_exposure``): ``wind`` (relative
        exposure index), ``wave`` (wave power, kW/m), ``surge`` (distance to the
        shelf contour, m), ``relief`` (mean elevation) and ``slr``.
    habitat_protection : pandas.DataFrame
        The rank each habitat gives each point (5: none within its protection
        distance) and ``R_hab``.
    fetch : geopandas.GeoDataFrame
        The points with the Wave Watch III fields interpolated to them, ``REI`` and,
        by sector, the fetch ``fdist_<deg>`` (m) and mean depth ``fdepth_<deg>``
        (``fetch_points``).
    fetch_rays : geopandas.GeoDataFrame
        The rays cast from the points that start at sea, with ``fetch_dist`` and
        ``direction``.
    wave_energies : geopandas.GeoDataFrame
        ``E_ocean``, ``E_local``, ``Eo_El_diff``, ``max_E_type`` and the range of the
        local wave heights and periods.
    missing_geomorphology : geopandas.GeoDataFrame or None
        With geomorphology: the points with no line within half the model
        resolution (they take the fill value).
    bathymetry, relief : xarray.DataArray
        The bathymetry clipped to the area and its negative values
        (``negative_bathymetry``), and the DEM with its negative values set to 0
        (``positive_dem``), in the CRS of the area of interest.
    population : xarray.DataArray or None
        The population raster clipped to the points.
    habitat_rasters : dict
        The habitat rasters clipped to the points, by habitat.
    """

    exposure: object
    intermediate: object
    habitat_protection: pd.DataFrame
    fetch: object
    fetch_rays: object
    wave_energies: object
    missing_geomorphology: object = None
    bathymetry: xr.DataArray | None = None
    relief: xr.DataArray | None = None
    population: xr.DataArray | None = None
    habitat_rasters: dict = field(default_factory=dict)

    def _save_items(self):
        items = {"coastal_exposure": self.exposure, "intermediate_exposure": self.intermediate,
                 "habitat_protection": self.habitat_protection, "fetch_points": self.fetch,
                 "fetch_rays": self.fetch_rays, "wave_energies": self.wave_energies}
        for k, v in (("shore_points_missing_geomorphology", self.missing_geomorphology),
                     ("negative_bathymetry", self.bathymetry), ("positive_dem", self.relief),
                     ("clipped_projected_pop", self.population)):
            if v is not None:
                items[k] = v
        for k, v in self.habitat_rasters.items():
            items[f"clipped_projected_{k}"] = v
        return items

    def __repr__(self) -> str:
        e = self.exposure["exposure"].to_numpy(np.float64)
        return (f"CoastalVulnerability({len(e)} shore points, exposure {np.nanmin(e):.3g} to {np.nanmax(e):.3g}, "
                f"mean {np.nanmean(e):.3g})")


# --------------------------------------------------------------------------- inputs

def _gpd():
    import geopandas as gpd

    return gpd


def _vector(x, what: str):
    gpd = _gpd()
    if isinstance(x, (str, os.PathLike)):
        from .._vector import load_vector

        x = load_vector(x)
    if isinstance(x, gpd.GeoSeries):
        x = gpd.GeoDataFrame(geometry=x)
    if not isinstance(x, gpd.GeoDataFrame):
        raise TypeError(f"{what} is a GeoDataFrame (or a vector file), not {type(x).__name__}")
    return x


def _raster(x, what: str) -> xr.DataArray:
    if isinstance(x, (str, os.PathLike)):
        from ..io import load_raster

        x = load_raster(x, band=1)
    if not isinstance(x, xr.DataArray):
        raise TypeError(f"{what} is a raster (DataArray or file), not {type(x).__name__}")
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError(f"{what} must have one band")
        x = x.isel({x.dims[0]: 0})
    if x.rio.crs is None:
        raise ValueError(f"{what} has no CRS")
    return x


def _pyproj(crs):
    from pyproj import CRS

    return None if crs is None else CRS.from_user_input(crs)


def _same(a, b) -> bool:
    return a is None or b is None or _pyproj(a) == _pyproj(b)


def _transform_xy(x, y, src, dst):
    """Coordinates from ``src`` to ``dst`` by GDAL's coordinate transformation (as OGR's Transform)."""
    if _same(src, dst):
        return np.asarray(x, np.float64), np.asarray(y, np.float64)
    from rasterio.crs import CRS
    from rasterio.warp import transform

    xs, ys = transform(CRS.from_user_input(_pyproj(src).to_wkt()), CRS.from_user_input(_pyproj(dst).to_wkt()),
                       np.asarray(x, np.float64).tolist(), np.asarray(y, np.float64).tolist())
    return np.asarray(xs, np.float64), np.asarray(ys, np.float64)


def _transform_geoms(geoms, src, dst):
    import shapely

    if _same(src, dst):
        return geoms

    def fn(xy):
        x, y = _transform_xy(xy[:, 0], xy[:, 1], src, dst)
        return np.c_[x, y]

    return shapely.transform(geoms, fn)


def _transform_bbox(bb, src, dst, edge_samples: int = 11):
    """pygeoprocessing.transform_bounding_box: the edges sampled, transformed, their extremes."""
    p0, p1 = np.array((bb[0], bb[3])), np.array((bb[0], bb[1]))
    p2, p3 = np.array((bb[2], bb[1])), np.array((bb[2], bb[3]))
    out = []
    for pa, pb, k, f in ((p0, p1, 0, min), (p1, p2, 1, min), (p2, p3, 0, max), (p3, p0, 1, max)):
        pts = np.array([pa * v + pb * (1 - v) for v in np.linspace(0, 1, edge_samples)])
        x, y = _transform_xy(pts[:, 0], pts[:, 1], src, dst)
        out.append(f((x if k == 0 else y).tolist()))
    minx, maxx = sorted([out[0], out[2]])
    miny, maxy = sorted([out[1], out[3]])
    box = [minx, miny, maxx, maxy]
    if not np.all(np.isfinite(box)):
        raise ValueError(f"could not transform the bounding box {bb} between the CRS of the inputs")
    return box


def _geometries(gdf):
    """InVEST's _ogr_to_geometry_list: the features' geometries, invalid ones buffered by 0."""
    import shapely

    g = np.asarray(gdf.geometry.values, dtype=object)
    g = g[~shapely.is_missing(g)]
    bad = ~shapely.is_valid(g)
    if bad.any():
        g = g.copy()
        g[bad] = shapely.buffer(g[bad], 0)
    return g


def _parts(geom):
    from shapely.geometry.base import BaseMultipartGeometry

    return list(geom.geoms) if isinstance(geom, BaseMultipartGeometry) else [geom]


def _column(gdf, name: str, what: str):
    lower = {str(c).lower(): c for c in gdf.columns if c != gdf.geometry.name}
    if name.lower() not in lower:
        raise ValueError(f"{what} lacks the field {name!r}")
    return gdf[lower[name.lower()]]


def _xy(gdf):
    import shapely

    pts = np.asarray(gdf.geometry.values, dtype=object)
    if not np.all(shapely.get_type_id(pts) == 0):
        raise ValueError("expected points")
    return shapely.get_coordinates(pts)


# --------------------------------------------------------------------------- rasters

def _nodata_of(da: xr.DataArray):
    """The raster's NoData value (its encoded one for float rasters read as NaN), or None."""
    for nd in (da.rio.encoded_nodata, da.rio.nodata):
        if nd is not None:
            return float(nd)
    return None


def _grid(da: xr.DataArray):
    t = da.rio.transform()
    if t.b or t.d:
        raise ValueError("rotated rasters are not supported")
    return (t.c, t.a, t.b, t.f, t.d, t.e)


def _warp_grid(bb, px):
    """pygeoprocessing.warp_raster's target_bb fitted to the pixel size, as gdal.Warp then makes it."""
    nx = int(abs(float(bb[2] - bb[0]) / px[0]))
    ny = int(abs(float(bb[3] - bb[1]) / px[1]))
    if not np.isclose(abs(nx * px[0]) - (bb[2] - bb[0]), 0.0):
        nx += 1
    if not np.isclose(abs(ny * px[1]) - (bb[3] - bb[1]), 0.0):
        ny += 1
    nx, ny = max(nx, 1), max(ny, 1)
    maxx = bb[0] + abs(px[0] * nx)
    maxy = bb[1] + abs(px[1] * ny)
    xres, yres = abs(px[0]), abs(px[1])
    gx = int((maxx - bb[0] + xres / 2.0) / xres)  # gdalwarp -te -tr
    gy = int((abs(maxy - bb[1]) + yres / 2.0) / yres)
    return (bb[0], xres, 0.0, maxy, 0.0, -yres), (gy, gx)


def _template(gt, shape, crs, dtype):
    from affine import Affine

    from ..io import _with_georef

    ny, nx = shape
    x = gt[0] + (np.arange(nx) + 0.5) * gt[1]
    y = gt[3] + (np.arange(ny) + 0.5) * gt[5]
    da = xr.DataArray(np.broadcast_to(np.zeros((), dtype), (ny, nx)), dims=("y", "x"), coords={"y": y, "x": x})
    return _with_georef(da, crs=crs, transform=Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3]))


def _linear_units(crs):
    from rasterio.crs import CRS

    c = CRS.from_user_input(_pyproj(crs).to_wkt())
    if not c.is_projected:
        return None
    return float(c.linear_units_factor[1])


def _warp(da, gt, shape, crs, threads):
    """gdal.Warp, bilinear, onto the grid ``gt``/``shape`` in ``crs``.

    Within one CRS ls.reproject (exact coordinates) gives gdal.Warp's bits. Across CRSs
    gdal.Warp moves the source coordinates by its approximate transformation (1/8 cell),
    which InVEST keeps: that warp is GDAL's own, through rasterio, in one chunk as InVEST's.
    """
    from .._reproject import reproject

    template = _template(gt, shape, crs, da.dtype)
    if _same(da.rio.crs, crs):
        return reproject(da, like=template, resampling="bilinear", tolerance=0, n_threads=threads)
    import rasterio.warp
    from affine import Affine
    from rasterio.enums import Resampling

    nd = _nodata_of(da)
    src = np.asarray(da.values)
    floating = np.issubdtype(src.dtype, np.floating)
    if nd is not None and floating:
        src = np.where(np.isnan(src), nd, src).astype(src.dtype)
    dst = np.zeros(shape, src.dtype)
    rasterio.warp.reproject(src, dst, src_transform=da.rio.transform(), src_crs=da.rio.crs,
                            dst_transform=Affine.from_gdal(*gt), dst_crs=_pyproj(crs).to_wkt(),
                            resampling=Resampling.bilinear, src_nodata=nd, dst_nodata=nd,
                            init_dest_nodata=nd is not None, num_threads=max(int(threads), 1))
    out = template.copy(data=dst)
    if nd is None:
        return out
    if floating:
        out = out.where(dst != nd).astype(src.dtype)
        out.rio.write_nodata(nd, encoded=True, inplace=True)
        return out
    return out.rio.write_nodata(nd)


def _zero_outside(da):
    """gdal.Warp of a raster without NoData leaves 0 where there is no source."""
    if np.issubdtype(da.dtype, np.floating):
        return da.fillna(0)
    nd = da.rio.nodata
    return da.where(da != nd, 0).astype(da.dtype) if nd is not None else da


@dataclass
class _Clipped:
    values: np.ndarray  # NaN for NoData (float rasters), else the codes with ``nodata``
    gt: tuple
    nodata: float | None  # the raster's NoData value (None: none)
    da: xr.DataArray

    def kernel_args(self):
        """(array for the C++ kernels, has NoData, NoData, exact NoData test)."""
        v = self.values
        if np.issubdtype(v.dtype, np.floating):
            nd = self.nodata
            ok = nd is not None and not math.isnan(nd)
            return v, ok, (nd if ok else 0.0), False
        nd = self.nodata
        return v.astype(np.float64), nd is not None, (nd if nd is not None else 0.0), True


def _clip_project(da, box, crs, res, threads, what) -> _Clipped:
    """InVEST's clip_and_project_raster: the raster clipped (bilinear, its own CRS and pixel size)
    to the box (given in ``crs``), then warped (bilinear) to ``crs``."""
    src_crs = da.rio.crs
    gt0 = _grid(da)
    px = (gt0[1], gt0[5])
    nodata = _nodata_of(da)
    gt1, shape1 = _warp_grid(_transform_bbox(box, crs, src_crs), px)
    w1 = _warp(da, gt1, shape1, src_crs, threads)
    if nodata is None:
        w1 = _zero_outside(w1)
    ny, nx = shape1
    xb = [gt1[0], gt1[0] + nx * gt1[1] + ny * gt1[2]]
    yb = [gt1[3], gt1[3] + nx * gt1[4] + ny * gt1[5]]
    bb1 = [np.min(xb), np.min(yb), np.max(xb), np.max(yb)]
    units = _linear_units(src_crs)
    tpx = tuple(np.multiply(px, units)) if units is not None else (res, -res)
    gt2, shape2 = _warp_grid(_transform_bbox(bb1, src_crs, crs), tpx)
    if _same(src_crs, crs) and tuple(gt2) == tuple(gt1) and tuple(shape2) == tuple(shape1):
        out = w1  # the same grid: gdal.Warp copies the values
    else:
        out = _warp(w1, gt2, shape2, crs, threads)
        if nodata is None:
            out = _zero_outside(out)
        gt1 = gt2
    values = np.ascontiguousarray(np.asarray(out.values))
    if values.size == 0:
        raise ValueError(f"{what} does not cover the area")
    return _Clipped(values, tuple(float(v) for v in gt1), nodata, out)


def _map(values, like: _Clipped, crs, nodata, name):
    from affine import Affine

    from ..io import _with_georef

    gt = like.gt
    ny, nx = values.shape
    x = gt[0] + (np.arange(nx) + 0.5) * gt[1]
    y = gt[3] + (np.arange(ny) + 0.5) * gt[5]
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}, name=name)
    da = _with_georef(da, crs=crs, transform=Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3]))
    if np.issubdtype(values.dtype, np.floating):
        return _with_georef(da, nodata=np.nan)
    return _with_georef(da, nodata=nodata) if nodata is not None else da


def _kernel(radius, gt):
    """pygeoprocessing's dichotomous kernel of InVEST's _aggregate_raster_values_in_radius."""
    d = int(abs(radius / gt[1]))
    apothem = math.floor(d)
    yy, xx = np.mgrid[-apothem:apothem + 1, -apothem:apothem + 1]
    return (np.hypot(yy, xx) <= d).astype(np.uint8), d


def _aggregate(points, clipped: _Clipped, radius, threads):
    kernel, d = _kernel(radius, clipped.gt)
    arr, has, nd, exact = clipped.kernel_args()
    s, n, nz = _core.serv_coastal_aggregate(points, arr, np.asarray(clipped.gt, np.float64), has, nd, exact, kernel,
                                            d, threads)
    return s, n, nz


def _means(points, clipped: _Clipped, radius, threads):
    """numpy.mean of the valid cells within the radius (numpy scalars of InVEST's type), NaN for none."""
    s, n, _ = _aggregate(points, clipped, radius, threads)
    f32 = clipped.values.dtype == np.float32
    out = []
    for si, ni in zip(s.tolist(), n.tolist()):
        if ni <= 0:
            out.append(np.nan)
        elif f32:
            out.append(np.float32(si) / np.float32(ni))
        else:
            out.append(np.float64(si) / np.float64(ni))
    return out


# --------------------------------------------------------------------------- the steps

def _shore(aoi_geoms, union, res):
    """InVEST's prepare_landmass_line_index_and_interpolate_shore_points: the exterior edges of
    the landmass (all, for the rays) and the points every ``res`` along those in the AOI."""
    import shapely
    import shapely.ops
    from shapely.geometry import LineString, MultiLineString

    aoi = shapely.ops.unary_union(list(aoi_geoms))
    shapely.prepare(aoi)
    segments, points = [], []
    for poly in _parts(union):
        if poly.geom_type != "Polygon":
            continue
        c = shapely.get_coordinates(poly.exterior)
        if len(c) == 0:
            continue
        start = c[0]
        rest = c[1:]
        rest = rest[~((rest[:, 0] == start[0]) & (rest[:, 1] == start[1]))]
        seq = np.vstack([start[None, :], rest])
        seg = np.c_[seq, np.vstack([seq[1:], start[None, :]])]
        # only edges of length 0 are left out (InVEST: numpy.isclose(..., rtol=1e-7), see the docs)
        seg = seg[~((seg[:, 0] == seg[:, 2]) & (seg[:, 1] == seg[:, 3]))]
        segments.append(seg)
        lines = shapely.linestrings(seg.reshape(-1, 2, 2))
        inside = shapely.intersects(aoi, lines)
        pieces = []
        for g in shapely.intersection(aoi, lines[inside]):
            if isinstance(g, (LineString, MultiLineString)):
                pieces.extend(_parts(g))
        merged = shapely.ops.linemerge(pieces)
        for line in _parts(merged):
            n = math.ceil(line.length / res)
            if n == 0:
                continue
            fr = np.array([float(i) / n for i in range(n)])
            points.append(shapely.get_coordinates(shapely.line_interpolate_point(line, fr, normalized=True)))
    seg = np.vstack(segments) if segments else np.zeros((0, 4))
    pts = np.vstack(points) if points else np.zeros((0, 2))
    return np.ascontiguousarray(seg), np.ascontiguousarray(pts)


def _deltas():
    out = []
    for k in range(16):
        compass = float(k) / 16 * 360
        cart = -(compass - 90)
        out.append((math.cos(cart * math.pi / 180), math.sin(cart * math.pi / 180)))
    return np.asarray(out, np.float64)


def _nearest(tree_xy, xy, k):
    from scipy.spatial import cKDTree

    k = min(k, len(tree_xy))
    _, idx = cKDTree(tree_xy).query(xy, k=k)
    idx = np.asarray(idx).reshape(len(xy), k)
    diff = tree_xy[idx] - xy[:, None, :]
    dist = np.sqrt(diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1])
    return idx, dist


def _weighted(values, idx, dist, close, max_dist):
    """numpy.average(values[close], weights=max_dist - d[close], axis=0), sums in order of nearness."""
    n, k = idx.shape
    num = np.zeros((n, values.shape[1]))
    den = np.zeros(n)
    started = np.zeros(n, bool)
    for j in range(k):
        m = close[:, j]
        w = max_dist - dist[:, j]
        t = values[idx[:, j]] * w[:, None]
        first, more = m & ~started, m & started
        num[first] = t[first]
        den[first] = w[first]
        num[more] = num[more] + t[more]
        den[more] = den[more] + w[more]
        started |= m
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den[:, None], started


def _wwiii(w, xy, crs, threads):
    """InVEST's interpolate_wwiii_to_shore: the fields of the 3 nearest points, weighted by distance."""
    wcrs = w.crs
    names = [c for c in w.columns if c != w.geometry.name and c not in ("I", "J")]
    lower = {str(c).lower(): c for c in names}
    need = [f"{p}{d}" for p in _WWIII for d in SECTORS]
    missing = [c for c in need if c.lower() not in lower]
    if missing:
        raise ValueError(f"the Wave Watch III points lack the field(s) {missing[:5]}{'...' if len(missing) > 5 else ''}")
    max_dist = 210000.0 if (wcrs is not None and _pyproj(wcrs).is_projected) else 3.0
    sx, sy = _transform_xy(xy[:, 0], xy[:, 1], crs, wcrs)
    wxy = _xy(w)
    vals = np.column_stack([pd.to_numeric(w[c], errors="coerce").to_numpy(np.float64) for c in names])
    idx, dist = _nearest(wxy, np.c_[sx, sy], 3)
    close = dist < max_dist
    if not close.any(axis=1).all():
        raise ValueError("no Wave Watch III point was found near the area of interest: is it far outside the "
                         "coverage of the Wave Watch III points?")
    avg, _ = _weighted(vals, idx, dist, close, max_dist)
    return {c: avg[:, i] for i, c in enumerate(names)}, lower


def _wave_hp(un, fn, dn):
    """compute_wave_height and compute_wave_period of InVEST, element by element as its scalars."""
    def pw(x, e):  # Python's float ** float (C pow), not numpy's fast paths
        return np.power(x, np.full(np.shape(x), e))

    un = np.where(un < 1, 1.0, un)
    g = 9.81
    dn = -dn
    un2 = pw(un, 2.0)
    ds = g * dn / un2
    fs = g * fn / un2
    a = np.tanh(0.343 * pw(ds, 1.14))
    b = np.tanh(4.41e-4 * pw(fs, 0.79) / a)
    h = (0.24 * un2 / g) * pw(a * b, 0.572)
    a = np.tanh(0.1 * pw(ds, 2.01))
    b = np.tanh(2.77e-7 * pw(fs, 1.45) / a)
    t = 7.69 * un / g * pw(a * b, 0.187)
    return h, t


def _bin(values, invert=False):
    """InVEST's _bin_values_to_percentiles (values: a list as InVEST builds it)."""
    base = values
    if invert:
        base = np.multiply(base, -1)
    pct = [20, 40, 60, 80, 100]
    rank = np.searchsorted(np.nanpercentile(base, pct), base).astype(float) + 1
    rank[rank > len(pct)] = np.nan
    return rank


def _r_hab(ranks):
    """InVEST's _calc_Rhab of each row (habitat ranks), with Python's arithmetic."""
    out = np.full(len(ranks), np.nan)
    if ranks.shape[1] == 0:
        return np.full(len(ranks), 5.0)
    ok = ~np.isnan(ranks).any(axis=1)
    cache = {}
    for i in np.flatnonzero(ok):
        row = tuple(float(v) for v in ranks[i])
        if row not in cache:
            sum_sq = 0
            min_rank = 5
            for r in row:
                min_rank = min(min_rank, r)
                sum_sq += (5 - r) ** 2
            if sum_sq > 0:
                v = max(1, 4.8 - 0.5 * ((1.5 * (5 - min_rank)) ** 2 + sum_sq - (5 - min_rank) ** 2) ** 0.5)
            else:
                v = 5
            cache[row] = float(v)
        out[i] = cache[row]
    return out


def _habitat_table(habitats):
    """(id, data, rank, distance) of each habitat."""
    rows = []
    if isinstance(habitats, dict):
        for hid, spec in habitats.items():
            if isinstance(spec, dict):
                low = {str(k).lower(): v for k, v in spec.items()}
                data = low.get("data", low.get("path"))
                rank = low.get("rank")
                dist = low.get("distance", low.get("protection distance (m)"))
            else:
                data, rank, dist = spec
            rows.append((str(hid), data, rank, dist))
    elif isinstance(habitats, pd.DataFrame):
        df = habitats.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        dcol = next((c for c in ("protection distance (m)", "distance") if c in df.columns), None)
        src = next((c for c in ("data", "path") if c in df.columns), None)
        if "id" not in df.columns or "rank" not in df.columns or dcol is None or src is None:
            raise ValueError("the habitat table needs the columns id, path (or data), rank and "
                             "'protection distance (m)' (or distance)")
        for _, r in df.iterrows():
            rows.append((str(r["id"]), r[src], r["rank"], r[dcol]))
    else:
        raise TypeError("habitats is a DataFrame (id, path, rank, protection distance (m)) or a dict "
                        "{id: (data, rank, distance)}")
    out = []
    for hid, data, rank, dist in rows:
        if rank is None or dist is None or data is None:
            raise ValueError(f"habitat {hid!r}: give its data, rank and protection distance")
        rank, dist = int(rank), float(dist)
        if not 1 <= rank <= 5:
            raise ValueError(f"habitat {hid!r}: the rank goes from 1 to 5, got {rank}")
        if dist < 0:
            raise ValueError(f"habitat {hid!r}: the protection distance must be >= 0")
        out.append((hid, data, rank, dist))
    return out


def _is_vector_input(x) -> bool:
    gpd = _gpd()
    if isinstance(x, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return True
    if isinstance(x, xr.DataArray):
        return False
    if isinstance(x, (str, os.PathLike)):
        ext = os.path.splitext(os.fspath(x))[1].lower()
        return ext not in (".tif", ".tiff", ".vrt", ".img", ".asc", ".nc", ".jp2", ".bil", ".hgt", ".grd")
    raise TypeError(f"a habitat is a GeoDataFrame, a raster or a file, not {type(x).__name__}")


def _vector_habitat(gdf, rank, dist, xy, bb, crs):
    """InVEST's search_for_vector_habitat: rank where a polygon lies within dist of the point."""
    import shapely
    from shapely.geometry import box

    hcrs = gdf.crs
    clip = box(*_transform_bbox(bb, crs, hcrs))
    geoms = np.asarray(gdf.geometry.values, dtype=object)
    geoms = geoms[~shapely.is_missing(geoms)]
    bad = ~shapely.is_valid(geoms)
    if bad.any():
        geoms = geoms.copy()
        geoms[bad] = shapely.buffer(geoms[bad], 0)
    clipped = shapely.intersection(geoms, clip)
    clipped = clipped[~shapely.is_empty(clipped)]
    clipped = _transform_geoms(clipped, hcrs, crs)
    parts = [p for g in clipped for p in _parts(g)]
    out = np.full(len(xy), 5.0)
    if not parts:
        return out
    parts = np.asarray(parts, dtype=object)
    tree = shapely.STRtree(parts)
    pts = shapely.points(xy)
    pi, gi = tree.query(shapely.buffer(pts, dist))
    near = shapely.distance(pts[pi], parts[gi]) <= dist
    out[np.unique(pi[near])] = rank
    return out


# --------------------------------------------------------------------------- the model

def coastal_vulnerability(aoi, landmass, wwiii, bathymetry, shelf_contour, dem, habitats, *,
                          model_resolution: float, max_fetch_distance: float, dem_averaging_radius: float,
                          geomorphology=None, geomorphology_fill_value: float | None = None, population=None,
                          population_radius: float | None = None, slr=None, slr_field: str | None = None,
                          n_threads: int | None = None) -> CoastalVulnerability:
    """Exposure of the coast to erosion and flooding (InVEST's coastal vulnerability model).

    Shore points are spaced ``model_resolution`` apart along the coast of the
    landmass inside the area of interest; each gets a rank from 1 (sheltered) to 5
    (exposed) for every hazard: wind and waves (16 fetch rays cast from the point
    up to ``max_fetch_distance`` or the land they hit; the relative exposure index
    ``REI = sum F_k pct_k V_k`` and the wave power, the larger of the ocean waves
    (along rays that reach the maximum fetch) and the waves the local wind raises
    (from the fetch, the mean depth along the ray and the wind speed), from the Wave
    Watch III points), surge potential (distance to the continental shelf contour),
    relief (mean elevation within ``dem_averaging_radius``), habitats (each one
    within its protection distance lowers ``R_hab`` from 5), and optionally
    geomorphology (the mean rank of the lines within half the model resolution)
    and sea-level rise. Continuous variables become ranks by their quintiles over
    the points (relief reversed). The exposure index is the geometric mean of the
    ranks; without habitats, with ``R_hab = 5``; the difference is the role of
    the habitats. Optionally, the population density around each point.

    The outputs are InVEST's (natcap.invest 3.20), bit for bit, also for inputs
    in other CRS (``tests/test_services_coastal.py``), with two
    corrections: InVEST drops the edges of the landmass whose ends are
    ``numpy.isclose(..., rtol=1e-7)``, meant for repeated vertices, but relative to
    the coordinates (at UTM northings of 8e6, edges up to 0.8 m long), which
    breaks the coastline there: here only edges of length 0 are left out; and for
    a fetch ray with no valid depth InVEST widens a window around its end, moving
    it instead of cutting it where it meets the top or left edge of the raster:
    here the window is the square around the end, cut at the edges. Where a ray
    meets several edges of the land, its end is the nearest intersection with
    the whole ray (InVEST shortens the ray edge by edge, in the order of its
    R-tree, so its end may differ in the last bits, ~1e-8 m; REI and the local
    waves carry that, ~1e-11 relative). Inputs in other CRS are reprojected as
    InVEST does: vectors by GDAL's coordinate transformation, rasters by GDAL's
    warp (bilinear, its approximate transformation of 1/8 cell).

    Parameters
    ----------
    aoi
        Polygons of the area of interest, in a projected CRS in metres (the CRS of
        the results).
    landmass
        Polygons of the land (any CRS); their holes (lakes) are not coast.
    wwiii
        Wave Watch III points with, for each sector ``<deg>`` of 0, 22, 45, ...,
        337: ``REI_PCT<deg>``, ``REI_V<deg>``, ``WavP_<deg>``, ``WavPPCT<deg>`` and
        ``V10PCT_<deg>``.
    bathymetry
        Depth (negative at sea), covering the area out to the maximum fetch.
    shelf_contour
        Lines of the edge of the continental shelf.
    dem
        Elevation, over the coast.
    habitats
        A DataFrame with ``id``, ``path`` (or ``data``: a GeoDataFrame of polygons
        or a raster, any value but 0 being habitat), ``rank`` (1 to 5) and
        ``protection distance (m)``, as InVEST's habitat table; or a dict
        ``{id: (data, rank, distance)}``.
    model_resolution
        Spacing of the shore points (m); also the cell size of rasters in
        longitude / latitude once projected.
    max_fetch_distance
        Length of the fetch rays (m).
    dem_averaging_radius
        Radius (m) of the mean elevation around the points.
    geomorphology, geomorphology_fill_value
        Lines with a ``RANK`` field (1 to 5) and the rank of points with no line
        within half the model resolution.
    population, population_radius
        People per cell and the radius (m) of their density around the points.
    slr, slr_field
        Sea-level rise points and their field; each shore point takes the mean of
        its two nearest (within 500 km) weighted by distance.
    n_threads
        Threads (default: all cores but one); the results do not depend on it.

    Returns
    -------
    CoastalVulnerability

    Examples
    --------
    >>> import landschaft as ls
    >>> cv = ls.services.coastal_vulnerability(                         # doctest: +SKIP
    ...     ls.load_vector("aoi.gpkg"), ls.load_vector("land.gpkg"), ls.load_vector("wwiii.gpkg"),
    ...     ls.load_raster("bathymetry.tif"), ls.load_vector("shelf.gpkg"), ls.load_raster("dem.tif"),
    ...     ls.load_table("habitats.csv"), model_resolution=250, max_fetch_distance=12000,
    ...     dem_averaging_radius=5000)
    >>> cv.exposure[["shore_id", "exposure", "habitat_role"]].head()  # doctest: +SKIP
    """
    import shapely
    import shapely.ops
    from shapely.geometry import box

    gpd = _gpd()
    threads = resolve(n_threads)
    res = float(model_resolution)
    fetch = float(max_fetch_distance)
    dem_radius = float(dem_averaging_radius)
    if not res > 0 or not fetch > 0 or not dem_radius >= 0:
        raise ValueError("model_resolution and max_fetch_distance must be > 0, dem_averaging_radius >= 0")
    if (geomorphology is None) != (geomorphology_fill_value is None):
        raise ValueError("give geomorphology and geomorphology_fill_value together")
    if (population is None) != (population_radius is None):
        raise ValueError("give population and population_radius together")
    if (slr is None) != (slr_field is None):
        raise ValueError("give slr and slr_field together")

    aoi = _vector(aoi, "the area of interest")
    crs = aoi.crs
    if crs is None or not _pyproj(crs).is_projected:
        raise ValueError("the area of interest must be in a projected CRS, in metres")
    units = _linear_units(crs)
    if units is not None and units != 1.0:
        raise ValueError("the area of interest must be in a projected CRS in metres")
    aoi_bb = [float(v) for v in shapely.total_bounds(np.asarray(aoi.geometry.values, dtype=object))]
    bb = list(aoi_bb)
    pad = fetch + res
    bb[0] -= pad
    bb[1] -= pad
    bb[2] += pad
    bb[3] += pad

    # the landmass clipped to the area plus the fetch, in the AOI's CRS
    land = _vector(landmass, "the landmass")
    lgeoms = _geometries(land)
    lbox = box(*_transform_bbox(bb, crs, land.crs))
    hit = shapely.intersects(lbox, lgeoms)
    clipped = shapely.intersection(lbox, lgeoms[hit])
    clipped = _transform_geoms(clipped, land.crs, crs)
    bad = ~shapely.is_valid(clipped)
    if bad.any():
        clipped = clipped.copy()
        clipped[bad] = shapely.buffer(clipped[bad], 0)
    union = shapely.ops.unary_union(list(clipped))

    segments, xy = _shore(_geometries(aoi), union, res)
    n = len(xy)
    if n == 0:
        raise ValueError("no shoreline was found within the area of interest: the landmass must have some "
                         "edges within it")
    ids = np.arange(n, dtype=np.int64)
    points = gpd.GeoDataFrame({"shore_id": ids}, geometry=shapely.points(xy), crs=crs)

    # fetch rays and the bathymetry along them
    deltas = _deltas()
    origins = xy[:, None, :] + deltas[None, :, :] * 1
    shapely.prepare(union)
    on_land = shapely.intersects(union, shapely.points(origins.reshape(-1, 2))).reshape(n, 16)
    fdist, ends = _core.serv_coastal_rays(xy, on_land.astype(np.uint8), deltas, fetch, segments, threads)
    bathy = _clip_project(_raster(bathymetry, "the bathymetry"), bb, crs, res, threads, "the bathymetry")
    bv = bathy.values
    if np.issubdtype(bv.dtype, np.floating):
        neg = np.where(bv < 0, bv, np.nan).astype(bv.dtype)
        arr, has, nd = neg, bathy.nodata is not None and not math.isnan(bathy.nodata), bathy.nodata or 0.0
    else:
        nd0 = bathy.nodata
        neg = np.where(bv < 0, bv, nd0 if nd0 is not None else np.iinfo(bv.dtype).min).astype(bv.dtype)
        arr, has, nd = neg.astype(np.float64), True, float(nd0 if nd0 is not None else np.iinfo(bv.dtype).min)
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float64)
    neg_da = _map(neg, bathy, crs, bathy.nodata, "negative_bathymetry")
    water = ~on_land.reshape(-1)
    starts = origins.reshape(-1, 2)[water]
    rend = ends.reshape(-1, 2)[water]
    depth, status = _core.serv_coastal_ray_depth(np.ascontiguousarray(starts), np.ascontiguousarray(rend),
                                                 np.ascontiguousarray(arr), np.asarray(bathy.gt, np.float64),
                                                 bool(has), float(nd if has else 0.0), threads)
    if status >= 0:
        raise ValueError("failed to read the bathymetry along a fetch ray: does the bathymetry cover the area "
                         "of the fetch rays?")
    if status <= -2:
        raise ValueError("searched the entire bathymetry raster for valid values and found none")
    fdepth = np.full(n * 16, np.nan)
    fdepth[water] = depth
    fdepth = fdepth.reshape(n, 16)

    # Wave Watch III at the shore, wind and waves
    w = _vector(wwiii, "the Wave Watch III points")
    wvals, lower = _wwiii(w, xy, crs, threads)

    def wfield(prefix, d):
        return wvals[lower[f"{prefix}{d}".lower()]]

    rei = np.zeros(n)
    for k, d in enumerate(SECTORS):
        rei = rei + fdist[:, k] * wfield("REI_PCT", d) * wfield("REI_V", d)
    e_ocean = np.zeros(n)
    e_local = np.zeros(n)
    hmin = np.full(n, np.inf)
    hmax = np.full(n, -np.inf)
    tmin = np.full(n, np.inf)
    tmax = np.full(n, -np.inf)
    any_ray = np.zeros(n, bool)
    for k, d in enumerate(SECTORS):
        fd = fdist[:, k]
        ocean = np.isclose(fd, fetch)
        e_ocean[ocean] = e_ocean[ocean] + wfield("WavP_", d)[ocean] * wfield("WavPPCT", d)[ocean]
        m = fd > 0
        if not m.any():
            continue
        dep = fdepth[m, k]
        dep = np.where(np.isnan(dep), -1.0, dep)
        h, t = _wave_hp(wfield("V10PCT_", d)[m], fd[m], dep)
        power = 0.5 * np.power(h, np.full(h.shape, 2.0)) * t
        e_local[m] = e_local[m] + power * wfield("REI_PCT", d)[m]
        hmin[m] = np.minimum(hmin[m], h)
        hmax[m] = np.maximum(hmax[m], h)
        tmin[m] = np.minimum(tmin[m], t)
        tmax[m] = np.maximum(tmax[m], t)
        any_ray |= m
    for a in (hmin, hmax, tmin, tmax):
        a[~any_ray] = np.nan
    wave = [max(o, lo) for o, lo in zip(e_ocean.tolist(), e_local.tolist())]
    diff = e_ocean - e_local
    etype = np.where(diff > 0, "ocean", np.where(diff < 0, "local", None)).astype(object)

    # surge: distance to the shelf contour
    shelf = _vector(shelf_contour, "the shelf contour")
    sbb = [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]
    big = [sbb[0] - _MAX_SURGE, sbb[1] - _MAX_SURGE, sbb[2] + _MAX_SURGE, sbb[3] + _MAX_SURGE]
    sclip = box(*_transform_bbox(big, crs, shelf.crs))
    sg = np.asarray(shelf.geometry.values, dtype=object)
    sg = sg[~shapely.is_missing(sg)]
    sg = shapely.intersection(sg, sclip)
    sg = _transform_geoms(sg[~shapely.is_empty(sg)], shelf.crs, crs)
    parts = [p for g in sg for p in _parts(g)]
    if not parts:
        raise ValueError("no part of the shelf contour is within 1500 km of the area of interest")
    shelf_union = shapely.ops.linemerge(parts)
    pts = shapely.points(xy)
    surge = shapely.distance(pts, shelf_union).tolist()

    # relief
    dbb = [sbb[0] - dem_radius, sbb[1] - dem_radius, sbb[2] + dem_radius, sbb[3] + dem_radius]
    demc = _clip_project(_raster(dem, "the DEM"), dbb, crs, res, threads, "the DEM")
    dv = demc.values
    if np.issubdtype(dv.dtype, np.floating):
        pos = np.where(np.isnan(dv), np.nan, np.where(dv > 0, dv, 0)).astype(dv.dtype)
    else:
        valid = dv != demc.nodata if demc.nodata is not None else np.ones(dv.shape, bool)
        pos = np.where(dv > 0, dv, np.where(valid, 0, dv)).astype(dv.dtype)
    posc = _Clipped(np.ascontiguousarray(pos), demc.gt, demc.nodata, demc.da)
    relief = _means(xy, posc, dem_radius, threads)

    # habitats
    hab_ids, hab_ranks, hab_maps = [], [], {}
    for hid, data, rank, dist in _habitat_table(habitats):
        hbb = [sbb[0] - dist, sbb[1] - dist, sbb[2] + dist, sbb[3] + dist]
        if _is_vector_input(data):
            r = _vector_habitat(_vector(data, f"habitat {hid!r}"), rank, dist, xy, hbb, crs)
        else:
            hc = _clip_project(_raster(data, f"habitat {hid!r}"), hbb, crs, res, threads, f"habitat {hid!r}")
            _, cnt, nz = _aggregate(xy, hc, dist, threads)
            r = np.where(cnt < 0, np.nan, np.where(nz.astype(bool), float(rank), 5.0))
            hab_maps[hid] = _map(hc.values, hc, crs, hc.nodata, hid)
        hab_ids.append(hid)
        hab_ranks.append(r)
    ranks = np.column_stack(hab_ranks) if hab_ranks else np.zeros((n, 0))
    r_hab = _r_hab(ranks)
    habitat_protection = pd.DataFrame({"shore_id": ids, **{h: ranks[:, i] for i, h in enumerate(hab_ids)},
                                       "R_hab": r_hab})

    # geomorphology
    geo_vals = missing = None
    if geomorphology is not None:
        g = _vector(geomorphology, "the geomorphology")
        rk = pd.to_numeric(_column(g, "RANK", "the geomorphology"), errors="coerce").to_numpy(np.float64)
        gg = np.asarray(g.geometry.values, dtype=object)
        keep = ~shapely.is_missing(gg)
        gg, rk = _transform_geoms(gg[keep], g.crs, crs), rk[keep]
        ok = shapely.is_valid(gg)
        gg, rk = gg[ok], rk[ok]
        fill = float(geomorphology_fill_value)
        geo_vals = np.full(n, fill)
        found = np.zeros(n, bool)
        if len(gg):
            tree = shapely.STRtree(gg)
            bufs = shapely.buffer(pts, res / 2, quad_segs=30)
            pi, gi = tree.query(bufs, predicate="intersects")
            order = np.argsort(pi, kind="stable")
            pi, gi = pi[order], gi[order]
            for i in np.unique(pi):
                rs = set(rk[gi[pi == i]].tolist())
                geo_vals[i] = sum(rs) / float(len(rs))
                found[i] = True
        missing = points[~found].reset_index(drop=True)

    # sea-level rise
    slr_vals = None
    if slr is not None:
        s = _vector(slr, "the sea-level rise points")
        sv = pd.to_numeric(_column(s, slr_field, "the sea-level rise points"), errors="coerce").to_numpy(np.float64)
        sxy = _xy(s)
        tx, ty = _transform_xy(sxy[:, 0], sxy[:, 1], s.crs, crs)
        idx, dist = _nearest(np.c_[tx, ty], xy, 2)
        close = dist < _MAX_SLR
        avg, got = _weighted(sv[:, None], idx, dist, close, _MAX_SLR)
        slr_vals = np.where(got, avg[:, 0], np.nan).tolist()

    # population
    pop_vals = pop_map = None
    if population is not None:
        pr = float(population_radius)
        pbb = [sbb[0] - pr, sbb[1] - pr, sbb[2] + pr, sbb[3] + pr]
        popc = _clip_project(_raster(population, "the population"), pbb, crs, res, threads, "the population")
        area = abs(popc.gt[1] * popc.gt[5]) / 1e6
        pop_vals = [v / area for v in _means(xy, popc, pr, threads)]
        pop_map = _map(popc.values, popc, crs, popc.nodata, "population")

    # ranks and exposure
    final = {"R_hab": r_hab}
    inter = {}
    for name, vals, to_bin in (("R_wind", rei.tolist(), True), ("R_wave", wave, True), ("R_surge", surge, True),
                               ("R_relief", relief, True), ("R_geomorph", geo_vals, False),
                               ("R_slr", slr_vals, True), ("population", pop_vals, False)):
        if vals is None:
            continue
        if to_bin:
            final[name] = _bin(vals, invert=name == "R_relief")
            inter[name[2:]] = np.array([float(v) for v in vals])
        else:
            final[name] = np.array([float(v) for v in vals])
    rnames = [k for k in final if k.startswith("R_")]
    r = np.column_stack([final[k] for k in rnames])
    others = np.column_stack([final[k] for k in rnames if k != "R_hab"] + [np.full(n, 5.0)])

    def gmean(a):
        p = a[:, 0].copy()
        for j in range(1, a.shape[1]):
            p = p * a[:, j]
        return np.power(p, np.full(len(p), 1 / a.shape[1]))

    exposure = gmean(r)
    no_hab = gmean(others)
    out = gpd.GeoDataFrame({"shore_id": ids, **final, "exposure": exposure, "habitat_role": np.abs(exposure - no_hab),
                            "exposure_no_habitats": no_hab}, geometry=shapely.points(xy), crs=crs)
    intermediate = gpd.GeoDataFrame({"shore_id": ids, **inter}, geometry=shapely.points(xy), crs=crs)
    fetch_cols = {"shore_id": ids}
    for c in w.columns:
        if c != w.geometry.name and c not in ("I", "J") and any(str(c).lower() == f"{p}{d}".lower()
                                                              for p in _WWIII for d in SECTORS):
            fetch_cols[c] = wvals[c]
    fetch_cols["REI"] = rei
    for k, d in enumerate(SECTORS):
        fetch_cols[f"fdist_{d}"] = fdist[:, k]
        fetch_cols[f"fdepth_{d}"] = fdepth[:, k]
    fetch_points = gpd.GeoDataFrame(fetch_cols, geometry=shapely.points(xy), crs=crs)
    sel = water.reshape(n, 16)
    pi, ki = np.nonzero(sel)
    rays = gpd.GeoDataFrame({"fetch_dist": fdist[pi, ki], "direction": np.asarray(SECTORS, np.float64)[ki]},
                            geometry=shapely.linestrings(np.stack([origins[pi, ki], ends[pi, ki]], axis=1)), crs=crs)
    energies = gpd.GeoDataFrame({"shore_id": ids, "E_ocean": e_ocean, "E_local": e_local, "Eo_El_diff": diff,
                                 "max_E_type": etype, "maxH_local": hmax, "minH_local": hmin, "maxT_local": tmax,
                                 "minT_local": tmin}, geometry=shapely.points(xy), crs=crs)
    relief_da = _map(pos, demc, crs, demc.nodata, "positive_dem")
    return CoastalVulnerability(out, intermediate, habitat_protection, fetch_points, rays, energies, missing, neg_da,
                                relief_da, pop_map, hab_maps)
