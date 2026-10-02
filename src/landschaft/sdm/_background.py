# SPDX-License-Identifier: GPL-3.0-or-later
"""Background points (pseudo-absences): where the species could be and was not looked for."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from . import _points
from ._thin import _seed

__all__ = ["background"]


def _grid(raster: xr.DataArray):
    """terra's / raster's extent and resolution of a raster: (xmin, xmax, ymin, ymax, xres, yres, ny, nx)."""
    from ..io import _transform

    tr = _transform(raster)
    ny, nx = raster.shape[-2:]
    xmin, ymax = tr.c, tr.f
    xmax, ymin = xmin + nx * tr.a, ymax + ny * tr.e
    return xmin, xmax, ymin, ymax, (xmax - xmin) / nx, (ymax - ymin) / ny, ny, nx


def _could_be_lonlat(raster: xr.DataArray) -> bool:
    """raster's couldBeLonLat: a geographic CRS, or none and an extent like longitude / latitude."""
    crs = raster.rio.crs
    if crs is not None:
        return bool(crs.is_geographic)
    xmin, xmax, ymin, ymax, *_ = _grid(raster)
    return xmin > -365 and xmax < 365 and ymin > -90.1 and ymax < 90.1


def _cell_centres(raster: xr.DataArray, cells: np.ndarray):
    """terra's xyFromCell (0-based cells)."""
    xmin, xmax, ymin, ymax, xres, yres, ny, nx = _grid(raster)
    row, col = cells // nx, cells % nx
    return xmin + (col + 0.5) * xres, ymax - (row + 0.5) * yres


def _complete(env: xr.DataArray) -> np.ndarray:
    """Cells (row-major) with a value in every band."""
    data = np.asarray(env.values, dtype=float)
    data = data.reshape(-1, data.shape[-2] * data.shape[-1]) if data.ndim == 3 else data.reshape(1, -1)
    nodata = env.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        data = np.where(data == nodata, np.nan, data)
    return ~np.isnan(data).any(axis=0)


def _within_area(env: xr.DataArray, area) -> np.ndarray:
    from .._vector import rasterize

    # terra::mask(x, polygons) burns every cell the polygons touch
    mask = rasterize(area, env.isel(band=0) if "band" in env.dims else env, value=1, fill=0, all_touched=True)
    return np.asarray(mask.values).reshape(-1) == 1


def _distance_band(env, occ_cells: np.ndarray, distance) -> np.ndarray:
    """Cells whose distance to the nearest occurrence cell is in (min, max] (biomod2's "disk")."""
    lo, hi = (0.0, distance) if np.isscalar(distance) else distance
    lo = 0.0 if lo is None else float(lo)
    hi = np.inf if hi is None else float(hi)
    xmin, xmax, ymin, ymax, xres, yres, ny, nx = _grid(env)
    target = np.zeros(ny * nx, dtype=np.uint8)
    target[occ_cells] = 1
    # terra::distance: haversine on longitude / latitude grids, GDAL's proximity (float32) on projected ones
    if _could_be_lonlat(env):
        d = _core.nlm_distance(target.reshape(ny, nx), "haversine", 1.0, xmin, ymax, xres, yres)
    else:
        d = _core.nlm_distance(target.reshape(ny, nx), "gdal", xres)
    d = np.asarray(d, dtype=float).reshape(-1)
    return (d > lo) & (d <= hi)


def _r_quantile7(x: np.ndarray, probs) -> np.ndarray:
    x = np.sort(x[~np.isnan(x)])
    n = x.size
    out = []
    for p in probs:
        index = 1 + max(n - 1, 0) * p
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        q = x[lo - 1]
        if index > lo and x[hi - 1] != q:
            h = index - lo
            q = (1 - h) * q + h * x[hi - 1]
        out.append(q)
    return np.array(out)


def _outside_envelope(env, occ_cells: np.ndarray, quantile: float) -> np.ndarray:
    """Cells outside the environmental envelope of the occurrences (biomod2's "sre")."""
    if not 0 <= quantile < 0.5:
        raise ValueError("envelope must be in [0, 0.5)")
    data = np.asarray(env.values, dtype=float).reshape(env.shape[0], -1)
    cells = np.unique(occ_cells)
    inside = np.ones(data.shape[1], dtype=bool)
    for band in data:
        lo, hi = _r_quantile7(band[cells], (quantile, 1 - quantile))
        with np.errstate(invalid="ignore"):
            inside &= (band >= lo) & (band <= hi)
    return ~inside


def background(env, n: int = 10_000, *, occ=None, area=None, bias=None, distance=None, envelope=None,
               exclude_occurrences: bool = True, tryf: float = 3, lonlat_correction: bool = True,
               seed: int | None = None, x: str | None = None, y: str | None = None, crs=None):
    """Background points: a sample of the conditions available to the species.

    Presence-only models (MaxEnt, presence-background regressions, machine
    learning) contrast the presences with a sample of the cells the species
    could have been found in. The sample is drawn as ``dismo::randomPoints``
    draws it, with the same cells for the same seed: cells with a value in
    every predictor, without replacement, on longitude / latitude grids
    weighted by the cells' east-west width (so that each square kilometre
    counts the same), leaving out the cells of the occurrences. Where to draw
    from is the choice that matters most (Barve et al. 2011; VanDerWal et al.
    2009): restrict it to the accessible area, to a distance band around the
    records, or weight it by sampling effort.

    Parameters
    ----------
    env
        Predictors (raster stack or one band): the grid and the cells with
        data.
    n
        Number of points (fewer, with a warning, when the area has fewer
        cells).
    occ
        Occurrence records: their cells are left out (``exclude_occurrences``)
        and they anchor ``distance`` and ``envelope``.
    area
        Polygons to draw from (the accessible area, see
        :func:`accessible_area`).
    bias
        A raster of sampling weights on the grid of ``env`` (e.g. the density
        of records of the whole taxonomic group: "target-group background",
        Phillips et al. 2009, which cancels the sampling bias of the
        presences). ``dismo::randomPoints(prob = TRUE)``.
    distance
        Metres: only cells within this distance of an occurrence, or a pair
        ``(min, max)`` for cells farther than ``min`` and at most ``max``
        (biomod2's "disk" pseudo-absences; longitude / latitude grids in
        metres on the sphere, projected ones in their units).
    envelope
        Leave out the cells inside the occurrences' environmental envelope
        (every predictor between its ``envelope`` and ``1 - envelope``
        quantiles at the occurrences): biomod2's "SRE" pseudo-absences, e.g.
        0.025.
    exclude_occurrences
        Leave out the cells holding occurrences (dismo's ``excludep``).
    tryf
        Candidates drawn per point before excluding the occurrence cells
        (dismo's ``tryf``).
    lonlat_correction
        Weight longitude / latitude cells by their area (dismo's
        ``lonlatCorrection``).
    seed
        Seed of R's generator (default random; recorded in ``attrs``).
    x, y, crs
        Coordinate columns and CRS of a table of occurrences.

    Returns
    -------
    geopandas.GeoDataFrame
        Points at the centres of the cells drawn (CRS of ``env``), with their
        ``cell`` (row-major, 0-based).

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> m = ls.sdm.accessible_area(occ, distance=500_000)
    >>> bg = ls.sdm.background(env, 1000, occ=occ, area=m, seed=1)
    >>> len(bg), bg.crs.to_epsg()
    (1000, 4326)
    """
    import geopandas as gpd

    if not isinstance(env, xr.DataArray):
        raise TypeError("env must be a raster (DataArray)")
    if env.ndim == 2:
        env = env.expand_dims(band=[env.name or "layer"])
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    xmin, xmax, ymin, ymax, xres, yres, ny, nx = _grid(env)
    ncell = ny * nx
    mask = _complete(env)
    occ_cells = np.array([], dtype=np.int64)
    if occ is not None:
        ox, oy, ocrs = _points.coordinates(occ, x=x, y=y, crs=crs)
        ox, oy = _points.to_crs(ox, oy, ocrs, env.rio.crs)
        occ_cells = _points.grid_cells(ox, oy, env)
        occ_cells = occ_cells[occ_cells >= 0]
    if area is not None:
        mask &= _within_area(env, area)
    if distance is not None:
        if occ_cells.size == 0:
            raise ValueError("distance needs occurrences (occ=)")
        mask &= _distance_band(env, occ_cells, distance)
    if envelope is not None:
        if occ_cells.size == 0:
            raise ValueError("envelope needs occurrences (occ=)")
        mask &= _outside_envelope(env, occ_cells, float(envelope))
    if bias is not None:
        b = np.asarray(bias.values if isinstance(bias, xr.DataArray) else bias, dtype=float).reshape(-1)
        if b.size != ncell:
            raise ValueError("bias must be a raster on the grid of env")
        if np.nanmin(b[mask]) < 0 if mask.any() else False:
            raise ValueError("bias weights must not be negative")
        mask &= ~np.isnan(b)

    s = _seed(seed)
    rng = _core.RRandom(s)
    tryf = max(round(tryf), 1)  # R's round: half to even (Python's too)
    if n > ncell:
        warnings.warn(f"n is larger than the number of cells: {ncell} points", stacklevel=2)
        n = ncell
    nn = max(n * tryf, 10)
    cells = np.flatnonzero(mask)
    if cells.size == 0:
        raise ValueError("no cell with data to draw from")
    lonlat = lonlat_correction and _could_be_lonlat(env)
    dx = None
    if lonlat:  # the east-west width of each row's cells (raster's pointDistance), then per cell
        yc = ymax - (np.arange(ny) + 0.5) * yres
        width = _core.cost_geodesic(np.zeros(ny), yc, np.full(ny, xres), yc)
        dx = width[cells // nx]
    if bias is not None:
        prob = b[cells]
        if lonlat:
            prob = prob * (dx / dx.max())
        nn = min(nn, int(np.count_nonzero(prob > 0)))  # dismo forgot this cap (an error in R)
        pick = rng.sample_int(cells.size, nn, False, prob.tolist())
    else:
        nn = min(cells.size, nn)
        pick = rng.sample_int(cells.size, nn, False, dx.tolist() if lonlat else None)
    cells = cells[pick - 1]
    if exclude_occurrences and occ_cells.size:
        cells = cells[~np.isin(cells, occ_cells)]
    if cells.size > n:
        cells = cells[rng.sample_int(cells.size, n, False, None) - 1]
    elif cells.size < n:
        if cells.size < 0.1 * n:
            raise ValueError(f"only {cells.size} cells available for {n} points: widen the area")
        warnings.warn(f"only {cells.size} background points could be drawn ({n} requested)", stacklevel=2)
    px, py = _cell_centres(env, cells)
    out = gpd.GeoDataFrame({"cell": cells}, geometry=gpd.points_from_xy(px, py), crs=env.rio.crs)
    out.attrs["background"] = {"seed": s, "n": int(n), "candidates": int(mask.sum())}
    return out
