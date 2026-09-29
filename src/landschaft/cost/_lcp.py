# SPDX-License-Identifier: GPL-3.0-or-later
"""Least-cost surfaces, distances and paths (gdistance's accCost, costDistance, shortestPath)."""

from __future__ import annotations

import warnings

import numpy as np

from .. import _core
from ._points import as_cells
from ._transition import Transition

__all__ = ["acc_cost", "cost_distance", "shortest_path"]

PATH_OUTPUTS = ("transition", "transitions", "cells", "lines")


def _args(x: Transition) -> tuple:
    if not isinstance(x, Transition):
        raise TypeError("x must be a Transition (see ls.cost.transition)")
    return x.values, x.grid.core, x.directions, x.symmetric


def acc_cost(x: Transition, origins):
    """Accumulated cost surface from the nearest origin: gdistance's ``accCost()``.

    The cost of a step between neighbouring cells is the reciprocal of the
    transition's value (a conductance); every cell gets the cost of the cheapest
    route from any origin.

    Parameters
    ----------
    x
        A :class:`Transition` of conductances (usually after
        :func:`geo_correction`).
    origins
        Coordinates (an ``(x, y)`` pair or an ``(n, 2)`` array), shapely
        points or a GeoDataFrame; or a raster (array or DataArray of the
        transition's shape) whose non-zero cells are the origins.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        The accumulated cost; ``inf`` where no origin can be reached. A
        DataArray when the transition came from one.

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> r = np.array([[2, 2, 1, 1, 5, 5, 5],
    ...               [2, 2, 8, 8, 5, 2, 1],
    ...               [7, 1, 1, 8, 2, 2, 2],
    ...               [8, 7, 8, 8, 8, 8, 5],
    ...               [8, 8, 1, 1, 5, 3, 9],
    ...               [8, 1, 1, 2, 5, 3, 9]], dtype=float)   # GRASS r.cost example
    >>> tr = ls.cost.geo_correction(1 / ls.cost.transition(r, "mean", 8, extent=(0, 7, 0, 6)))
    >>> cost = ls.cost.acc_cost(tr, (5.5, 1.5))
    >>> cost.round(1)[0]                                # the top row
    array([21.8, 21.1, 20.7, 19.7, 16.9, 14.9, 13.7])
    """
    values, grid, d, s = _args(x)
    cells = as_cells(origins, x.grid, "origins")
    cells = cells[cells >= 0]
    out = _core.cost_acc_cost(values, grid, d, s, cells.tolist())
    return x._wrap(out.reshape(x.shape), "accumulated_cost")


def cost_distance(x: Transition, from_coords, to_coords=None) -> np.ndarray:
    """Least-cost distances between points: gdistance's ``costDistance()``.

    Parameters
    ----------
    x
        A :class:`Transition` of conductances.
    from_coords, to_coords
        Points (coordinates, shapely points or a GeoDataFrame). Without
        ``to_coords`` the distances among ``from_coords``.

    Returns
    -------
    numpy.ndarray
        ``(n_from, n_to)`` costs (``inf``: unreachable; NaN for points outside
        the grid). Among one set of points with a symmetric transition the
        matrix is symmetric, each pair summed from its later point as
        gdistance's ``dist`` result.

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> r = np.ones((10, 10))
    >>> tr = ls.cost.geo_correction(ls.cost.transition(r, "mean", 8, extent=(0, 10, 0, 10)))
    >>> ls.cost.cost_distance(tr, [(0.5, 0.5), (9.5, 9.5), (0.5, 9.5)]).round(3)
    array([[ 0.   , 12.728,  9.   ],
           [12.728,  0.   ,  9.   ],
           [ 9.   ,  9.   ,  0.   ]])
    """
    values, grid, d, s = _args(x)
    fc = as_cells(from_coords, x.grid, "points")
    tc = fc if to_coords is None else as_cells(to_coords, x.grid, "points")
    uf = np.unique(fc[fc >= 0])
    ut = uf if to_coords is None else np.unique(tc[tc >= 0])
    m = _core.cost_cost_matrix(values, grid, d, s, uf.tolist(), ut.tolist())
    out = np.full((fc.size, tc.size), np.nan)
    fi, ti = np.flatnonzero(fc >= 0), np.flatnonzero(tc >= 0)
    out[np.ix_(fi, ti)] = m[np.ix_(np.searchsorted(uf, fc[fi]), np.searchsorted(ut, tc[ti]))]
    if to_coords is None and x.is_symmetric():
        low = np.tril(out, -1)  # as.dist: the lower triangle, row i summed from point i
        out = low + low.T
        np.fill_diagonal(out, np.where(fc >= 0, 0.0, np.nan))
    return out


def shortest_path(x: Transition, origin, goal, output: str = "transition"):
    """Least-cost path from an origin to one or more goals: gdistance's ``shortestPath()``.

    Among equally cheap paths the one gdistance (igraph's Dijkstra) returns.

    Parameters
    ----------
    x
        A :class:`Transition` of conductances.
    origin
        One point (the first one is taken if several are given).
    goal
        One or more points.
    output : {"transition", "transitions", "cells", "lines"}
        ``"transition"``: a :class:`Transition` with ``1 / n_goals`` on the
        connections of each path (added up where paths share connections);
        ``"transitions"``: one such transition per goal; ``"cells"``: the cells
        of each path (row-major indices, origin first); ``"lines"``: a
        GeoDataFrame (or, without geopandas, a list) of line strings through
        the cell centres.

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> r = np.array([[1, 1, 1], [9, 9, 1], [1, 1, 1]], dtype=float)
    >>> tr = ls.cost.geo_correction(1 / ls.cost.transition(r, "mean", 4, extent=(0, 3, 0, 3)))
    >>> ls.cost.shortest_path(tr, (0.5, 0.5), (0.5, 2.5), output="cells")[0]   # around the wall
    array([6, 7, 8, 5, 2, 1, 0])
    """
    if output not in PATH_OUTPUTS:
        raise ValueError(f"output must be one of {', '.join(PATH_OUTPUTS)}")
    values, grid, d, s = _args(x)
    oc = as_cells(origin, x.grid, "origins")
    if oc.size > 1:
        warnings.warn("several origins given: only the first one is used", RuntimeWarning, stacklevel=2)
    if oc.size == 0 or oc[0] < 0:
        raise ValueError("the origin is outside the grid")
    gc = as_cells(goal, x.grid, "goals")
    valid = gc >= 0
    paths = [np.zeros(0, dtype=np.int64)] * gc.size
    found = _core.cost_shortest_paths(values, grid, d, s, int(oc[0]), gc[valid].tolist())
    for k, p in zip(np.flatnonzero(valid), found):
        paths[k] = np.asarray(p, dtype=np.int64)
    if output == "cells":
        return paths
    if output == "lines":
        return _lines(x, paths)
    n = len(paths)
    if output == "transitions":
        return [_path_transition(x, [p], 1.0 / n) for p in paths]
    return _path_transition(x, paths, 1.0 / n)


def _path_transition(x: Transition, paths, weight: float) -> Transition:
    """gdistance's path transition: ``weight`` added on each step, both ways (a general matrix)."""
    offsets = [tuple(o) for o in _core.cost_offsets(x.directions, False)]
    index = {o: k for k, o in enumerate(offsets)}
    ny, nx = x.shape
    out = np.zeros((len(offsets), x.n_cells))

    def offset(a: int, b: int) -> tuple[int, int]:
        dr, dc = b // nx - a // nx, b % nx - a % nx
        if x.grid.wrap and abs(dc) > 2:  # across the 180th meridian
            dc = dc - nx if dc > 0 else dc + nx
        return dr, dc

    for p in paths:
        for a, b in zip(p[:-1].tolist(), p[1:].tolist()):
            for u, v in ((a, b), (b, a)):
                k = index[offset(u, v)]
                out[k, u] = weight + out[k, u]
    return Transition(out, x.grid, x.directions, False, x.matrix_values, x._template)


def _lines(x: Transition, paths):
    import shapely

    ny, nx = x.shape
    xmin, xmax, ymin, ymax = x.extent
    xres, yres = (xmax - xmin) / nx, (ymax - ymin) / ny
    lines = []
    for p in paths:
        cx = (p % nx + 0.5) * xres + xmin
        cy = ymax - (p // nx + 0.5) * yres
        lines.append(shapely.linestrings(np.column_stack([cx, cy])) if p.size >= 2 else
                     shapely.from_wkt("LINESTRING EMPTY"))
    try:
        import geopandas as gpd
    except ImportError:
        return lines
    return gpd.GeoDataFrame({"id": np.arange(1, len(lines) + 1)}, geometry=lines, crs=x.crs)
