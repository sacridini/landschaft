# SPDX-License-Identifier: GPL-3.0-or-later
"""Terrain indices of a DEM (the R package spatialEco's tpi, tri, vrm, srr,
dissection, curvature, spherical.sd, hsp, trasp, hli, on terra's terrain;
corrected where spatialEco errs)."""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

from .. import _core
from .._threads import threaded
from ..cost._transition import _crs_lonlat
from ._grid import Surface, read_surface

__all__ = ["terrain", "tpi", "tri", "vrm", "srr", "dissection", "curvature", "spherical_sd", "hsp", "trasp",
           "hli"]

MEAN, SUM, MIN, MAX, SD = 0, 1, 2, 3, 4
CURVATURES = ("planform", "profile", "total", "mcnab", "bolstad")


def _read(x, extent, crs, band, nodata) -> Surface:
    return read_surface(x, extent=extent, crs=crs, band=band, nodata=nodata)


def _e(s: Surface) -> tuple:
    return tuple(map(float, s.extent))


def _size(s, name: str) -> tuple[int, int]:
    """A window size: one odd number of cells, or (rows, columns)."""
    v = (s, s) if np.isscalar(s) else tuple(s)
    if len(v) != 2:
        raise ValueError(f"{name} must be a number of cells or (rows, columns)")
    r, c = (int(k) for k in v)
    if r < 1 or c < 1 or r % 2 == 0 or c % 2 == 0 or (r, c) != tuple(v):
        raise ValueError(f"{name} must be odd numbers of cells, got {s!r}")
    return r, c


def _rect(r: int, c: int) -> tuple[int, int, list[int]]:
    return r, c, [1] * (r * c)


def _circle(s: Surface, d: float) -> tuple[int, int, list[int]]:
    """The cells within distance d (map units) of the centre (terra's focalMat
    "circle" with exact distances)."""
    xr_, yr_ = s.res
    nx, ny = 1 + 2 * math.floor(d / xr_), 1 + 2 * math.floor(d / yr_)
    cx, cy = nx // 2, ny // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    inside = np.sqrt(((xx - cx) * xr_) ** 2 + ((yy - cy) * yr_) ** 2) <= d
    return ny, nx, inside.astype(np.int8).ravel().tolist()


def _focal(s: Surface, window, stat: int) -> np.ndarray:
    wr, wc, mask = window
    return _core.surf_focal(s.values, _e(s), wr, wc, mask, stat)


def _slope_aspect(s: Surface) -> tuple[np.ndarray, np.ndarray]:
    ll = _crs_lonlat(s.crs)
    return (_core.surf_terrain(s.values, _e(s), ll, 8, False, False),
            _core.surf_terrain(s.values, _e(s), ll, 8, False, True))


@threaded
def terrain(x, v: str = "slope", unit: str = "degrees", neighbors: int = 8, *, extent=None, crs=None, band=None,
            nodata=None) -> xr.DataArray:
    """Slope or aspect of a DEM, as terra's ``terrain``.

    Horn's method (``neighbors=8``: the 3 x 3 neighbourhood weighted 1-2-1) or
    the 4 direct neighbours; on longitude/latitude rasters with the cell sizes
    in metres (geodesics on WGS84) and elevation in metres. The edge cells are
    missing.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The DEM (a single band; NaN or ``nodata`` = missing).
    v : {"slope", "aspect"}, default "slope"
        Slope (0 flat) or aspect (clockwise from north, where the slope faces).
    unit : {"degrees", "radians"}, default "degrees"
    neighbors : {8, 4}, default 8
    extent : tuple of float, optional
        ``(xmin, xmax, ymin, ymax)`` of an array (default: unit cells).
    crs : optional
        CRS of an array.
    band : int, optional
        Band of a multi-band file or DataArray (1-based).
    nodata : float, optional
        Value that marks missing cells.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import landschaft as ls
    >>> dem = ls.surface.example_data("orelevation")
    >>> slope = ls.surface.terrain(dem, "slope")
    >>> round(float(slope.mean()), 2)
    18.45
    """
    if v not in ("slope", "aspect"):
        raise ValueError(f"v must be 'slope' or 'aspect', got {v!r}")
    if unit not in ("degrees", "radians"):
        raise ValueError(f"unit must be 'degrees' or 'radians', got {unit!r}")
    if neighbors not in (4, 8):
        raise ValueError(f"neighbors must be 4 or 8, got {neighbors!r}")
    s = _read(x, extent, crs, band, nodata)
    out = _core.surf_terrain(s.values, _e(s), _crs_lonlat(s.crs), neighbors, unit == "degrees", v == "aspect")
    return s.raster(out, v)


@threaded
def tpi(x, scale=3, win: str = "rectangle", normalize: bool = False, *, extent=None, crs=None, band=None,
        nodata=None) -> xr.DataArray:
    """Topographic position index: the elevation minus the mean of the window.

    Positive on ridges and hilltops, negative in valleys, near 0 on flats and
    even slopes. The window includes the cell itself; cells whose window
    reaches a missing cell or the edge are missing.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The DEM (see :func:`terrain`).
    scale : int or float, default 3
        ``"rectangle"``: the window's side in cells (odd); ``"circle"``: its
        radius in map units (at least two cells).
    win : {"rectangle", "circle"}, default "rectangle"
    normalize : bool, default False
        Divided by the window's standard deviation.
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    xarray.DataArray

    Notes
    -----
    The window means (and standard deviations) come from compensated prefix
    sums along the rows, one step per window row: within about 1e-13 of the
    values' range of summing each window.

    spatialEco's ``zero.correct`` option computes the same mean (and, with
    ``normalize``, a standard deviation of 0 by a misplaced parenthesis): it
    is not offered.

    Examples
    --------
    >>> import landschaft as ls
    >>> dem = ls.surface.example_data("orelevation")
    >>> t = ls.surface.tpi(dem, 5)
    >>> round(float(t.min()), 1), round(float(t.max()), 1)
    (-31.2, 34.4)
    """
    s = _read(x, extent, crs, band, nodata)
    window = _window(s, scale, win)
    wr, wc, mask = window
    return s.raster(_core.surf_tpi(s.values, _e(s), wr, wc, mask, bool(normalize)), "tpi")


def _window(s: Surface, scale, win: str):
    if win == "circle":
        if scale < s.res[0] * 2:
            raise ValueError("the scale of a circular window must be at least two cells")
        return _circle(s, float(scale))
    if win != "rectangle":
        raise ValueError(f"win must be 'rectangle' or 'circle', got {win!r}")
    return _rect(*_size(scale, "scale"))


@threaded
def tri(x, s=3, *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Terrain ruggedness index (Riley et al. 1999).

    sqrt(sum((z - z_i)^2)) over the other cells of the window that have a
    value (so the edges get one too).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The DEM (see :func:`terrain`).
    s : int or (int, int), default 3
        The window in cells (odd).
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import landschaft as ls
    >>> t = ls.surface.tri(ls.surface.example_data("orelevation"))
    >>> round(float(t.mean()), 2)
    23.15
    """
    su = _read(x, extent, crs, band, nodata)
    r, c = _size(s, "s")
    return su.raster(_core.surf_tri(su.values, _e(su), r, c), "tri")


@threaded
def vrm(x, s=3, *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Vector ruggedness measure (Sappington et al. 2007).

    1 minus the length of the mean of the unit normal vectors of the window's
    cells: 0 for a plane however steep, towards 1 for rugged terrain. Normals
    from :func:`terrain`'s slope and aspect.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The DEM (see :func:`terrain`).
    s : int or (int, int), default 3
        The window in cells (odd).
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import landschaft as ls
    >>> v = ls.surface.vrm(ls.surface.example_data("orelevation"))
    >>> round(float(v.mean()), 4)
    0.006
    """
    su = _read(x, extent, crs, band, nodata)
    r, c = _size(s, "s")
    slp, asp = _slope_aspect(su)
    sin_s = np.sin(slp)
    parts = [np.sin(asp) * sin_s, np.cos(asp) * sin_s, np.cos(slp)]
    sums = [_core.surf_focal(p, _e(su), r, c, [1] * (r * c), SUM) for p in parts]
    length = np.sqrt(sums[0] ** 2 + sums[1] ** 2 + sums[2] ** 2)
    return su.raster(1 - length / round(r * c), "vrm")


@threaded
def srr(x, s=5, *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Surface relief ratio (Pike & Wilson 1971): (mean - min) / (max - min)
    of the window.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The DEM (see :func:`terrain`).
    s : int or (int, int), default 5
        The window in cells (odd).
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import landschaft as ls
    >>> r = ls.surface.srr(ls.surface.example_data("orelevation"))
    >>> round(float(r.mean()), 3)
    0.501
    """
    su = _read(x, extent, crs, band, nodata)
    w = _rect(*_size(s, "s"))
    mean, lo, hi = (_focal(su, w, k) for k in (MEAN, MIN, MAX))
    with np.errstate(invalid="ignore", divide="ignore"):
        return su.raster((mean - lo) / (hi - lo), "srr")


@threaded
def dissection(x, s=5, *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Martonne's modified dissection: (z - min) / (max - min) of the window.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The DEM (see :func:`terrain`).
    s : int or (int, int), default 5
        The window in cells (odd).
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import landschaft as ls
    >>> d = ls.surface.dissection(ls.surface.example_data("orelevation"))
    >>> round(float(d.mean()), 3)
    0.502
    """
    su = _read(x, extent, crs, band, nodata)
    w = _rect(*_size(s, "s"))
    lo, hi = _focal(su, w, MIN), _focal(su, w, MAX)
    with np.errstate(invalid="ignore", divide="ignore"):
        return su.raster((su.values - lo) / (hi - lo), "dissection")


@threaded
def curvature(x, type: str = "planform", *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Surface curvature of a DEM.

    ``"planform"``, ``"profile"`` and ``"total"`` (their sum) from Zevenbergen
    & Thorne's (1987) partial derivatives of the 3 x 3 neighbourhood, p, q
    (first) and r, s, t (second): planform -(q^2 r - 2pqs + p^2 t) /
    ((p^2 + q^2) sqrt(1 + p^2 + q^2)), profile -(p^2 r + 2pqs + q^2 t) /
    ((p^2 + q^2) (1 + p^2 + q^2)^(3/2)); ``"mcnab"``: the mean of the
    differences between the cell and its 8 neighbours, divided by its largest
    value; ``"bolstad"``: 10000 (z - mean of the 3 x 3) / 1000 / (1.5 cell
    width).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The DEM, on a projected raster (see :func:`terrain`).
    type : {"planform", "profile", "total", "mcnab", "bolstad"}
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    xarray.DataArray
        Missing on the edges (and flat cells for the Zevenbergen & Thorne
        curvatures).

    Notes
    -----
    spatialEco halves the second derivatives (divides by 2 L^2), pads the
    edges with 0, rounds to 6 decimals, uses the cell width for both
    directions, divides McNab's sum by 9 and subtracts 1 (operator
    precedence), and fails for ``"bolstad"``: corrected here.

    Examples
    --------
    >>> import landschaft as ls
    >>> c = ls.surface.curvature(ls.surface.example_data("normforest"), "total")
    >>> c.shape
    (371, 371)
    """
    if type not in CURVATURES:
        raise ValueError(f"type must be one of {', '.join(CURVATURES)}, got {type!r}")
    s = _read(x, extent, crs, band, nodata)
    return s.raster(_core.surf_curvature(s.values, _e(s), CURVATURES.index(type)), type)


@threaded
def spherical_sd(x, d=3, variance: bool = False, *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Spherical standard deviation of the slope and aspect (terrain
    heterogeneity).

    The unit normal vectors of the cells (from :func:`terrain`) averaged over
    the window: sqrt(1 - R^2), R the length of the mean vector (``variance``:
    1 - R^2).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The DEM (see :func:`terrain`).
    d : int or (int, int), default 3
        The window in cells (odd).
    variance : bool, default False
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import landschaft as ls
    >>> v = ls.surface.spherical_sd(ls.surface.example_data("orelevation"))
    >>> round(float(v.mean()), 4)
    0.0978
    """
    su = _read(x, extent, crs, band, nodata)
    w = _rect(*_size(d, "d"))
    slp, asp = _slope_aspect(su)
    sin_s = np.sin(slp)
    parts = [np.cos(asp) * sin_s, np.sin(asp) * sin_s, np.cos(slp)]
    mx, my, mz = (_core.surf_focal(p, _e(su), w[0], w[1], w[2], MEAN) for p in parts)
    v = 1 - (mx ** 2 + my ** 2 + mz ** 2)
    with np.errstate(invalid="ignore"):
        return su.raster(v if variance else np.sqrt(v), "spherical_sd")


@threaded
def hsp(x, min_scale=3, max_scale=27, inc=4, win: str = "rectangle", normalize: bool = False, *, extent=None,
        crs=None, band=None, nodata=None) -> xr.DataArray:
    """Hierarchical slope position (Murphy et al. 2010).

    The sum over the scales min_scale, min_scale + inc, ... max_scale of 100
    times the z-score of :func:`tpi` at that scale ((tpi - mean) / sd over the
    raster).

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The DEM (see :func:`terrain`).
    min_scale, max_scale, inc : default 3, 27, 4
        The scales: odd window sides in cells, or radii in map units for
        circles.
    win : {"rectangle", "circle"}, default "rectangle"
    normalize : bool, default False
        Rescaled to 0-1.
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    xarray.DataArray

    Notes
    -----
    spatialEco computes tpi - mean / sd (operator precedence), not the
    z-score: corrected here.

    Examples
    --------
    >>> import landschaft as ls
    >>> h = ls.surface.hsp(ls.surface.example_data("orelevation"), 3, 11, 4, normalize=True)
    >>> round(float(h.min()), 1), round(float(h.max()), 1)
    (0.0, 1.0)
    """
    s = _read(x, extent, crs, band, nodata)
    scales = []
    v = min_scale
    while v <= max_scale + 1e-9 * abs(max_scale):
        scales.append(v)
        v = v + inc
    if not scales:
        raise ValueError("no scale between min_scale and max_scale")
    windows = [_window(s, k, win) for k in scales]
    return s.raster(_core.surf_hsp(s.values, _e(s), [w[0] for w in windows], [w[1] for w in windows],
                                   [w[2] for w in windows], bool(normalize)), "hsp")


@threaded
def trasp(x, *, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Transformed aspect (Roberts & Cooper 1989): (1 - cos(aspect - 30 deg)) / 2.

    0 on north-northeast slopes (cool, wet), 1 on south-southwest ones (hot,
    dry). spatialEco uses 3.142 for pi.

    Parameters
    ----------
    x : str, path, xarray.DataArray or array_like
        The DEM (see :func:`terrain`).
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import landschaft as ls
    >>> t = ls.surface.trasp(ls.surface.example_data("orelevation"))
    >>> round(float(t.min()), 3), round(float(t.max()), 3)
    (0.0, 1.0)
    """
    s = _read(x, extent, crs, band, nodata)
    asp = _core.surf_terrain(s.values, _e(s), _crs_lonlat(s.crs), 8, True, True)
    return s.raster((1 - np.cos((math.pi / 180) * (asp - 30))) / 2, "trasp")


@threaded
def hli(aspect, slope, latitude=None, direct: bool = False, scaled: bool = False, units: str = "degrees",
        hemisphere: str = "northern", force_hemisphere: bool = True, equation: int = 1, *, extent=None, crs=None,
        band=None, nodata=None) -> xr.DataArray:
    """Heat load index (McCune & Keon 2002).

    Their equations 1-3 of potential direct incident radiation from slope,
    folded aspect and latitude: the aspect folded about the northeast-southwest
    line (hottest on southwest slopes; southern hemisphere: about northwest-
    southeast) or, with ``direct``, about north-south. The equations give the
    logarithm of the radiation (MJ cm^-2 yr^-1); ``scaled`` gives the
    radiation.

    Parameters
    ----------
    aspect, slope : str, path, xarray.DataArray or array_like
        Aspect and slope rasters (see :func:`terrain`).
    latitude : float, optional
        Latitude (degrees). Default: the centre of the raster (which needs a
        CRS).
    direct : bool, default False
        Fold about north-south (direct radiation) instead of NE-SW.
    scaled : bool, default False
        exp() of the equation.
    units : {"degrees", "radians"}, default "degrees"
        Units of aspect and slope.
    hemisphere : {"northern", "southern"}, default "northern"
    force_hemisphere : bool, default True
        Use ``hemisphere``; otherwise the sign of the latitude decides.
    equation : {1, 2, 3}, default 1
        McCune & Keon's equation (3: latitudes 30-60 degrees).
    extent, crs, band, nodata
        As in :func:`terrain` (for arrays).

    Returns
    -------
    xarray.DataArray

    Notes
    -----
    spatialEco's southern hemisphere folds the aspect in degrees without the
    absolute values (then takes its cosine as radians), uses the signed
    latitude with the mirrored aspect, and with ``units="radians"`` converts
    the aspect again: corrected here (the southern hemisphere uses the
    absolute latitude).

    Examples
    --------
    >>> import landschaft as ls
    >>> dem = ls.surface.example_data("orelevation")
    >>> h = ls.surface.hli(ls.surface.terrain(dem, "aspect"), ls.surface.terrain(dem, "slope"))
    >>> round(float(h.mean()), 3)
    -0.288
    """
    if units not in ("degrees", "radians"):
        raise ValueError(f"units must be 'degrees' or 'radians', got {units!r}")
    if equation not in (1, 2, 3):
        raise ValueError("equation must be 1, 2 or 3")
    if hemisphere not in ("northern", "southern"):
        raise ValueError(f"hemisphere must be 'northern' or 'southern', got {hemisphere!r}")
    a = _read(aspect, extent, crs, band, nodata)
    sl = _read(slope, extent, crs, band, nodata)
    if a.shape != sl.shape:
        raise ValueError("aspect and slope must have the same shape")
    if latitude is None:
        latitude = _centre_latitude(a)
    if not force_hemisphere:
        hemisphere = "southern" if latitude <= 0 else "northern"
    rad = math.pi / 180
    asp = a.values * rad if units == "degrees" else a.values
    S = sl.values * rad if units == "degrees" else sl.values
    if hemisphere == "northern":
        A = math.pi - np.abs(asp - math.pi) if direct else np.abs(math.pi - np.abs(asp - math.pi * 5 / 4))
        L = latitude * rad
    else:
        A = np.abs(asp - math.pi) if direct else np.abs(math.pi - np.abs(asp - math.pi * 7 / 4))
        L = abs(latitude) * rad
    cL, sL, cS, sS, cA, sA = math.cos(L), math.sin(L), np.cos(S), np.sin(S), np.cos(A), np.sin(A)
    if equation == 1:
        res = -1.467 + 1.582 * cL * cS - 1.5 * cA * sS * sL - 0.262 * sL * sS + 0.607 * sA * sS
    elif equation == 2:
        res = -1.236 + 1.35 * cL * cS - 1.376 * cA * sS * sL - 0.331 * sL * sS + 0.375 * sA * sS
    else:
        res = 0.339 + 0.808 * cL * cS - 0.196 * sL * sS - 0.482 * cA * sS
    if scaled:
        res = np.exp(res)
    return a.raster(res, "hli")


def _centre_latitude(s: Surface) -> float:
    xmin, xmax, ymin, ymax = s.extent
    if s.crs is None:
        raise ValueError("hli needs the latitude, or a raster with a CRS")
    if _crs_lonlat(s.crs):
        return (ymin + ymax) / 2
    from rasterio.warp import transform

    _, lat = transform(s.crs, "EPSG:4326", [(xmin + xmax) / 2], [(ymin + ymax) / 2])
    return float(lat[0])
