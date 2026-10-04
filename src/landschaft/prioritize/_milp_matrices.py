# SPDX-License-Identifier: GPL-3.0-or-later
"""Matrices of planning units for penalties and constraints: prioritizr's
``boundary_matrix()``, ``adjacency_matrix()``, ``connectivity_matrix()`` and
``proximity_matrix()`` (the same values), and bridges from landschaft's
distance modules (``ls.cost``, ``ls.circuit``, ``ls.conn``) to connectivity.

Every matrix covers every unit: the cells of a raster (those without data have
no entries) or the rows of a GeoDataFrame, in their order, as the ``data`` of
:meth:`Problem.add_boundary_penalties` and the others expect.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import xarray as xr

__all__ = ["adjacency_matrix", "boundary_matrix", "connectivity_matrix", "decay_connectivity", "proximity_matrix",
           "unit_points"]


def _raster_cells(x: xr.DataArray) -> tuple[np.ndarray, int, int]:
    """The cells with data in some band (terra's cells(allNA(x), 0)) and the grid's shape."""
    v = np.asarray(x.values, dtype=np.float64)
    if v.ndim == 2:
        v = v[None]
    ny, nx = v.shape[-2:]
    ok = np.isfinite(v).any(axis=0).ravel()
    return np.flatnonzero(ok), ny, nx


def _pairs_4(include: np.ndarray, ny: int, nx: int):
    """Pairs of included cells above/below and left/right of each other (both orders)."""
    mask = np.zeros(ny * nx, dtype=bool)
    mask[include] = True
    idx = np.arange(ny * nx).reshape(ny, nx)
    m = mask.reshape(ny, nx)
    out = []
    for a, b, ma, mb in ((idx[:-1, :], idx[1:, :], m[:-1, :], m[1:, :]), (idx[:, :-1], idx[:, 1:], m[:, :-1], m[:, 1:])):
        ok = (ma & mb).ravel()
        out.append((a.ravel()[ok], b.ravel()[ok]))
    return out  # [(up, down), (left, right)]


def _res(x: xr.DataArray) -> tuple[float, float]:
    """terra's xres and yres: the extent over the number of cells."""
    from ..io import _transform

    t = _transform(x)
    ny, nx = x.shape[-2:]
    return ((t.c + nx * t.a) - t.c) / nx, (t.f - (t.f + ny * t.e)) / ny


def _symmetric(i, j, v, n) -> sp.csc_matrix:
    m = sp.csc_matrix((np.concatenate([v, v]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n))
    m.sum_duplicates()
    m.sort_indices()
    return m


def boundary_matrix(x) -> sp.csc_matrix:
    """The length of the boundary every two units share, and every unit's perimeter on the diagonal
    (prioritizr's ``boundary_matrix()``): raster cells with data share ``xres`` with the cells above
    and below and ``yres`` with those left and right, perimeter ``2 (xres + yres)``; polygons share
    the length of their common paths, perimeters their own, rounded to 8 decimals.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> bm = P.boundary_matrix(cost)
    >>> bm.shape
    (2400, 2400)
    >>> float(bm[100, 100])
    4000.0
    """
    if isinstance(x, xr.DataArray):
        include, ny, nx = _raster_cells(x)
        xres, yres = _res(x)
        (u, d), (le, ri) = _pairs_4(include, ny, nx)
        n = ny * nx
        m = _symmetric(np.concatenate([u, le]), np.concatenate([d, ri]),
                       np.concatenate([np.full(u.size, xres), np.full(le.size, yres)]), n)
        m = m + sp.csc_matrix((np.full(include.size, (xres + yres) * 2), (include, include)), shape=(n, n))
        m = m.tocsc()
        m.sort_indices()
        return m
    import shapely

    geoms = np.asarray(x.geometry.values, dtype=object)
    n = geoms.size
    left, right = shapely.STRtree(geoms).query(geoms, predicate="intersects")
    keep = left < right
    left, right = left[keep], right[keep]
    bounds = shapely.boundary(geoms)
    length = np.round(shapely.length(shapely.intersection(bounds[left], bounds[right])), 8)
    m = _symmetric(left, right, length, n)
    m = (m + sp.csc_matrix((np.round(shapely.length(geoms), 8), (np.arange(n), np.arange(n))), shape=(n, n))).tocsc()
    m.eliminate_zeros()
    m.sort_indices()
    return m


def adjacency_matrix(x, directions: int = 4) -> sp.csc_matrix:
    """1 for every two adjacent units (prioritizr's ``adjacency_matrix()``): raster cells with data
    that share an edge (``directions`` 4) or also a corner (8); polygons that intersect, touching at
    a corner included.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> adj = P.adjacency_matrix(cost)
    >>> int(adj.sum(axis=1).max())
    4
    """
    if isinstance(x, xr.DataArray):
        if directions not in (4, 8):
            raise ValueError("directions must be 4 or 8")
        include, ny, nx = _raster_cells(x)
        pairs = _pairs_4(include, ny, nx)
        i = [p[0] for p in pairs]
        j = [p[1] for p in pairs]
        if directions == 8:
            mask = np.zeros(ny * nx, dtype=bool)
            mask[include] = True
            idx = np.arange(ny * nx).reshape(ny, nx)
            m = mask.reshape(ny, nx)
            for a, b, ma, mb in ((idx[:-1, :-1], idx[1:, 1:], m[:-1, :-1], m[1:, 1:]),
                                 (idx[:-1, 1:], idx[1:, :-1], m[:-1, 1:], m[1:, :-1])):
                ok = (ma & mb).ravel()
                i.append(a.ravel()[ok])
                j.append(b.ravel()[ok])
        i, j = np.concatenate(i), np.concatenate(j)
        out = _symmetric(i, j, np.ones(i.size), ny * nx)
    else:
        import shapely

        geoms = np.asarray(x.geometry.values, dtype=object)
        left, right = shapely.STRtree(geoms).query(geoms, predicate="intersects")
        keep = left < right
        out = _symmetric(left[keep], right[keep], np.ones(int(keep.sum())), geoms.size)
    out.data[:] = 1.0
    return out


def connectivity_matrix(x, y) -> sp.csc_matrix:
    """Connectivity between adjacent units (prioritizr's ``connectivity_matrix()``): the length of
    their shared boundary times the mean of ``y`` in the two (``y``: a raster on the units' grid; with
    polygons a raster summed over each, or a column name).

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> cm = P.connectivity_matrix(cost, features[0])
    >>> cm.shape
    (2400, 2400)
    """
    bm = boundary_matrix(x)
    n = bm.shape[0]
    if isinstance(x, xr.DataArray):
        v = np.asarray(y.squeeze().values if isinstance(y, xr.DataArray) else y, dtype=np.float64).ravel()
        if v.size != n:
            raise ValueError("y must be a raster on the units' grid")
    elif isinstance(y, str):
        v = x[y].to_numpy(dtype=np.float64)
    else:
        v = exact_means(y, x.geometry.values)
    up = sp.triu(bm, k=1).tocoo()
    val = up.data * ((v[up.row] + v[up.col]) * 0.5)
    keep = val > 0
    out = _symmetric(up.row[keep], up.col[keep], val[keep], n)
    return out


def exact_means(raster: xr.DataArray, geometries) -> np.ndarray:
    """exactextractr's exact_extract(fun = "mean") of a one-band raster, as prioritizr's fast_extract
    gives it: the coverage-weighted sum over the cells with a value divided by their coverage, both
    rounded to float and divided in float (exactextract's RasterStats::mean), NaN where no cell with a
    value is covered; values below 1e-10 in magnitude become 0."""
    from ._milp import exact_sums

    r = raster.squeeze()
    v = np.asarray(r.values, dtype=np.float64)
    stack = xr.DataArray(np.stack([v, np.isfinite(v).astype(np.float64)]), dims=("band", *r.dims),
                         coords={**{d: r[d] for d in r.dims if d in r.coords}, "band": ["value", "count"]})
    from ..io import _transform

    stack = stack.rio.write_transform(_transform(r))
    if r.rio.crs is not None:
        stack = stack.rio.write_crs(r.rio.crs)
    sums = exact_sums(stack, geometries)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (sums[:, 0].astype(np.float32) / sums[:, 1].astype(np.float32)).astype(np.float64)
    out[np.abs(out) < 1e-10] = 0.0
    return out


def proximity_matrix(x, distance: float) -> sp.csc_matrix:
    """1 for every two units within ``distance`` of each other (prioritizr's ``proximity_matrix()``):
    raster cell centres, or polygons by their nearest points.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> pm = P.proximity_matrix(cost, 1500)
    >>> pm.shape
    (2400, 2400)
    >>> int(pm.getrow(500).nnz)
    8
    """
    import shapely

    if isinstance(x, xr.DataArray):
        include, ny, nx = _raster_cells(x)
        pts = unit_points(x)[include]
        tree = shapely.STRtree(shapely.points(pts))
        left, right = tree.query(shapely.points(pts), predicate="dwithin", distance=distance)
        keep = left < right
        return _symmetric(include[left[keep]], include[right[keep]], np.ones(int(keep.sum())), ny * nx)
    geoms = np.asarray(x.geometry.values, dtype=object)
    left, right = shapely.STRtree(geoms).query(geoms, predicate="dwithin", distance=distance)
    keep = left < right
    return _symmetric(left[keep], right[keep], np.ones(int(keep.sum())), geoms.size)


def unit_points(x) -> np.ndarray:
    """A point per unit, (n, 2) coordinates: raster cell centres (every cell), or a point inside every
    polygon (shapely's point on surface), to measure distances between units with
    :func:`landschaft.cost.cost_distance` or :func:`landschaft.cost.commute_distance`.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> pts = P.unit_points(cost)
    >>> pts.shape
    (2400, 2)
    """
    if isinstance(x, xr.DataArray):
        from ..io import _transform

        t = _transform(x)
        ny, nx = x.shape[-2:]
        rows, cols = np.divmod(np.arange(ny * nx), nx)
        return np.column_stack([t.c + (cols + 0.5) * t.a + (rows + 0.5) * t.b,
                                t.f + (cols + 0.5) * t.d + (rows + 0.5) * t.e])
    import shapely

    return shapely.get_coordinates(shapely.point_on_surface(np.asarray(x.geometry.values, dtype=object)))


def decay_connectivity(distances, median: float, min_probability: float = 0.001) -> sp.csc_matrix:
    """Connectivity from distances between units as a dispersal probability (Conefor's negative
    exponential): ``exp(-ln(2) d / median)``, 0.5 at the ``median`` dispersal distance, dropped below
    ``min_probability`` (sparse), 0 on the diagonal and for unreachable pairs (``inf``, NaN).

    ``distances``: units x units (every unit), e.g. least-cost distances from
    :func:`landschaft.cost.cost_distance` between :func:`unit_points`, commute distances, effective
    resistances (:mod:`landschaft.circuit`) or Euclidean ones.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> from scipy.spatial.distance import cdist
    >>> pts = P.unit_points(cost)
    >>> dc = P.decay_connectivity(cdist(pts, pts), median=3000)
    >>> dc.shape
    (2400, 2400)
    >>> round(float(dc[0, 1]), 3)
    0.794
    """
    d = np.asarray(distances.toarray() if sp.issparse(distances) else distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distances must be a square matrix of units")
    if not median > 0:
        raise ValueError("median must be > 0")
    with np.errstate(invalid="ignore", over="ignore"):
        p = np.exp(-math.log(2.0) * d / median)
    p[~np.isfinite(d)] = 0.0
    p[p < min_probability] = 0.0
    np.fill_diagonal(p, 0.0)
    out = sp.csc_matrix(p)
    out.eliminate_zeros()
    out.sort_indices()
    return out
