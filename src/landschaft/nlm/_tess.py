# SPDX-License-Identifier: GPL-3.0-or-later
"""Voronoi mosaics: NLMR's ``nlm_mosaictess`` and ``nlm_mosaicgibbs``."""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from .. import _core, _threads
from ._common import check_resolution, check_size, grid, resolve_seed

__all__ = ["mosaic_gibbs", "mosaic_tess"]


def _geos_cell_order(x: np.ndarray, y: np.ndarray, tolerance: float) -> np.ndarray | None:
    """The germ of each Voronoi cell in the order GEOS returns the cells
    (``sf::st_voronoi(sf::st_union(points))``), or None without shapely."""
    try:
        import shapely
    except ImportError:
        return None
    pts = shapely.points(x, y)
    # st_union() of points: sorted coordinates
    order = np.lexsort((y, x))
    cells = shapely.get_parts(shapely.voronoi_polygons(shapely.multipoints(np.column_stack([x[order], y[order]])),
                                                       tolerance=tolerance))
    tree = shapely.STRtree(pts)
    cell_idx, germ_idx = tree.query(cells, predicate="contains")
    germ_of_cell = np.full(len(cells), -1, dtype=np.int64)
    germ_of_cell[cell_idx] = germ_idx
    if np.any(germ_of_cell < 0) or len(cells) != len(x):
        raise RuntimeError("GEOS's Voronoi cells do not match the germs one to one")
    return germ_of_cell


def _values_by_cell_order(x, y, values, tolerance, nlmr_compat, model):
    """NLMR gives the k-th value to the k-th Voronoi polygon in GEOS's order."""
    if not nlmr_compat:
        return values
    germ_of_cell = _geos_cell_order(x, y, tolerance)
    if germ_of_cell is None:
        warnings.warn(f"{model}: shapely (GEOS) is needed for NLMR's assignment of values to the Voronoi cells; "
                      "each germ gets its own value instead (install shapely, or pass nlmr_compat=False)",
                      stacklevel=3)
        return values
    out = np.empty_like(values)
    out[germ_of_cell] = values[: len(germ_of_cell)]
    return out


def _finish(values: np.ndarray, nlmr_compat: bool, rescale: bool) -> np.ndarray:
    if nlmr_compat:
        # terra::rasterize() writes float32 values
        values = values.astype(np.float32).astype(np.float64)
    return _core.nlm_rescale(values) if rescale else values


def mosaic_tess(ncol: int, nrow: int, germs: int, resolution: float = 1.0, seed: int | None = None,
                rescale: bool = True, nlmr_compat: bool = True, *, n_threads: int | None = None) -> xr.DataArray:
    """Polygonal (Voronoi) mosaic neutral landscape model (NLMR's ``nlm_mosaictess``, Gaucherel 2008).

    ``germs`` uniform random points split the landscape into Voronoi tiles, each
    with a uniform random value.

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    germs : int
        Number of germs (tiles).
    resolution : float, default 1
        Cell size.
    seed : int, optional
        Seed of R's generator (NLMR's ``user_seed``); ``None`` draws one, stored in
        ``attrs["seed"]``.
    rescale : bool, default True
        Rescale the values to 0-1.
    nlmr_compat : bool, default True
        Reproduce NLMR: the k-th value goes to the k-th tile in the order GEOS
        returns the Voronoi cells (found with shapely; without it a warning and
        the other assignment), and values are rounded to float32 as
        ``terra::rasterize`` stores them. With False the k-th germ gets the k-th
        value, in double precision.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 values.

    Examples
    --------
    >>> import landschaft as ls
    >>> t = ls.nlm.mosaic_tess(80, 60, germs=40, seed=1)
    >>> t.shape, int(len(set(t.values.ravel())))
    ((60, 80), 40)
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    germs = int(germs)
    if germs < 1:
        raise ValueError("germs must be a positive integer")
    s = resolve_seed(seed)
    with _threads.limit(n_threads):
        x, y, v = _core.nlm_tess_germs(s, germs, float(ncol), float(nrow))
        v = _values_by_cell_order(x, y, v, 0.0, nlmr_compat, "mosaic_tess")
        values = _finish(_core.nlm_nearest_germ_raster(x, y, v, nrow, ncol), nlmr_compat, rescale)
    return grid(values, resolution, "mosaic_tess",
                {"seed": s, "model": "mosaic_tess", "germs": germs, "nlmr_compat": int(bool(nlmr_compat))})


def mosaic_gibbs(ncol: int, nrow: int, germs: int, R: float, patch_classes: int, resolution: float = 1.0,
                 seed: int | None = None, rescale: bool = True, nlmr_compat: bool = True,
                 *, n_threads: int | None = None) -> xr.DataArray:
    """Gibbs-process Voronoi mosaic (NLMR's ``nlm_mosaicgibbs``, Gauchel et al. 2008).

    ``germs`` points from a simple sequential inhibition process (spatstat's
    ``rSSI``: no two points closer than ``R``) split the landscape into Voronoi
    tiles; the classes 1..``patch_classes``, repeated to ``germs`` values and
    shuffled, are given to the tiles.

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    germs : int
        Number of germs.
    R : float
        Minimum distance between germs, in cells.
    patch_classes : int
        Number of classes.
    resolution : float, default 1
        Cell size.
    seed : int, optional
        Seed of R's generator (NLMR's ``user_seed``); ``None`` draws one.
    rescale : bool, default True
        Rescale the classes to 0-1.
    nlmr_compat : bool, default True
        Reproduce NLMR, which gives the k-th germ's class to the k-th Voronoi
        polygon in GEOS's order (not to the germ's own tile; found with shapely,
        without it a warning and the other assignment). With False each germ's
        tile takes its own class.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 values (classes before rescaling).

    Examples
    --------
    >>> import landschaft as ls
    >>> g = ls.nlm.mosaic_gibbs(80, 60, germs=30, R=6, patch_classes=3, seed=2, rescale=False)
    >>> sorted(int(v) for v in set(g.values.ravel()))
    [1, 2, 3]
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    germs, patch_classes, R = int(germs), int(patch_classes), float(R)
    if germs < 1 or patch_classes < 1:
        raise ValueError("germs and patch_classes must be positive integers")
    if R < 0:
        raise ValueError("R must be non-negative")
    s = resolve_seed(seed)
    with _threads.limit(n_threads):
        x, y, classes = _core.nlm_gibbs_germs(s, germs, R, patch_classes, float(ncol), float(nrow))
        if len(x) < germs:
            warnings.warn(f"mosaic_gibbs: only {len(x)} of {germs} germs fit at distance {R} "
                          "(rSSI gave up after 1000 attempts)", stacklevel=2)
        v = _values_by_cell_order(x, y, classes[: len(x)], 0.1, nlmr_compat, "mosaic_gibbs")
        values = _finish(_core.nlm_nearest_germ_raster(x, y, v, nrow, ncol), nlmr_compat, rescale)
    return grid(values, resolution, "mosaic_gibbs",
                {"seed": s, "model": "mosaic_gibbs", "germs": germs, "R": R, "nlmr_compat": int(bool(nlmr_compat))})
