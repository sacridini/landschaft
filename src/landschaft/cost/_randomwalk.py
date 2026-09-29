# SPDX-License-Identifier: GPL-3.0-or-later
"""Random walks and randomised shortest paths (gdistance's commuteDistance, passage, rSPDistance)."""

from __future__ import annotations

import numpy as np

from .. import _core
from ._points import as_cells
from ._transition import Transition

__all__ = ["commute_distance", "passage", "rsp_distance"]


def _symmetric(x: Transition, what: str) -> Transition:
    if not isinstance(x, Transition):
        raise TypeError("x must be a Transition (see ls.cost.transition)")
    if not x.is_symmetric():
        raise ValueError(f"{what} needs a symmetric transition matrix")
    return x


def commute_distance(x: Transition, coords) -> np.ndarray:
    """Expected commute distances of a random walk between points: gdistance's ``commuteDistance()``.

    A random walk that moves from a cell to a neighbour with probability
    proportional to the conductance, starting at one point, takes on average
    this long (in steps weighted by the transition) to reach the other and
    come back. It equals the effective resistance between
    the points (the circuit view: :mod:`landschaft.circuit`) times the sum of
    all conductances (the graph's volume).

    Parameters
    ----------
    x
        A symmetric :class:`Transition` of conductances; for longitude/latitude
        grids gdistance recommends ``geo_correction(tr, "r")``.
    coords
        Points (coordinates, shapely points or a GeoDataFrame).

    Returns
    -------
    numpy.ndarray
        ``(n, n)`` symmetric; ``inf`` between points in separate parts of the
        graph (gdistance fails there; the volume is then that of each part),
        NaN for points outside the grid or on cells without connections.

    Notes
    -----
    Solved with the circuit module's multifrontal Cholesky factorisation, one
    per connected part; gdistance uses CHOLMOD, and the two agree to about
    1e-12 (relative).

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> tr = ls.cost.transition(np.ones((1, 3)), "mean", 4)   # a chain of 3 cells, conductance 1
    >>> ls.cost.commute_distance(tr, [(0.5, 0.5), (2.5, 0.5)]).round(9)   # resistance 2, volume 4
    array([[0., 8.],
           [8., 0.]])
    """
    x = _symmetric(x, "commute_distance")
    cells = as_cells(coords, x.grid, "points")
    r, volume, comp_volume, comp = _core.cost_commute(x.values, x.grid.core, x.directions, x.symmetric,
                                                      cells.tolist())
    if comp_volume.size == 1:
        return r * volume
    vol = np.where(comp >= 0, comp_volume[np.clip(comp, 0, None)], np.nan)
    return r * vol[:, None]


def _check_theta(theta: float) -> float:
    theta = float(theta)
    if not 0 <= theta <= 20:
        raise ValueError("theta value out of range (between 0 and 20)")
    return theta


def passage(x: Transition, origin, goal, theta: float | None = None, *, total_net: str = "net",
            output: str = "raster", gdistance_compat: bool = True):
    """Passages of a random walk from origins to goals: gdistance's ``passage()``.

    Without ``theta``: the net number of times a random walk from the origin
    (one of them at random) to the goal passes through each cell, which is
    the current through the cell when a unit current flows from the origins
    to the goals (as in Circuitscape).

    With ``theta``: randomised shortest paths (Saerens et al. 2009), walks
    that prefer cheap routes more and more as ``theta`` grows: 0 is the random
    walk, large values approach the least-cost path. The walk's steps follow
    ``exp(-theta * cost) * P``, P the random walk's probabilities, cost the
    reciprocal of the transition's values; with several goals each of them
    absorbs part of the walk (gdistance's ``1 - 1/n`` rows).

    Parameters
    ----------
    x
        A :class:`Transition` of conductances (symmetric without ``theta``).
    origin, goal
        Points, or rasters whose non-zero cells are the origins / goals.
    theta
        None, or 0 to 20 (gdistance's range).
    total_net : {"net", "total"}
        Net passages (through minus back) or total visits; without ``theta``
        only ``"net"``.
    output : {"raster", "transition"}
        Passages per cell, or per connection (a transition; ``"net"``: the
        positive part of the net flow). Without ``theta`` only ``"raster"``.
    gdistance_compat
        Without ``theta``: True (default) keeps gdistance's potential of the
        circuit's grounded cell (the last cell with connections takes the
        potential of the one before it), which changes the passages around
        that cell; False gives the exact currents.

    Returns
    -------
    xarray.DataArray, numpy.ndarray or Transition
        Without ``theta``: 1 at the origins and goals, 0 off the graph. With
        ``theta``: NaN off the graph; net passages at the origins and goals
        doubled, as gdistance.

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> tr = ls.cost.transition(np.ones((3, 3)), "mean", 4)
    >>> ls.cost.passage(tr, (0.5, 1.5), (2.5, 1.5), gdistance_compat=False).round(3)
    array([[0.25, 0.25, 0.25],
           [1.  , 0.5 , 1.  ],
           [0.25, 0.25, 0.25]])
    >>> ls.cost.passage(tr, (0.5, 1.5), (2.5, 1.5), theta=2).round(3)   # the straight route takes more
    array([[0.009, 0.009, 0.009],
           [1.   , 0.982, 1.   ],
           [0.009, 0.009, 0.009]])
    """
    if not isinstance(x, Transition):
        raise TypeError("x must be a Transition (see ls.cost.transition)")
    if total_net not in ("net", "total"):
        raise ValueError("totalNet should be either total or net")
    if output not in ("raster", "transition"):
        raise ValueError("output should be either 'raster' or 'transition'")
    oc = as_cells(origin, x.grid, "origins")
    gc = as_cells(goal, x.grid, "goals")
    oc, gc = oc[oc >= 0], gc[gc >= 0]
    if theta is None:
        x = _symmetric(x, "passage without theta")
        if total_net != "net" or output != "raster":
            raise ValueError("without theta only total_net='net' and output='raster' are available "
                             "(gdistance: 'no method available -- try a low value of theta instead')")
        out = _core.cost_rw_passages(x.values, x.grid.core, x.directions, x.symmetric, oc.tolist(), gc.tolist(),
                                     bool(gdistance_compat))
        return x._wrap(out.reshape(x.shape), "passages")
    theta = _check_theta(theta)
    edges = output == "transition"
    out = _core.cost_rsp_passages(x.values, x.grid.core, x.directions, x.symmetric, oc.tolist(), gc.tolist(),
                                  theta, total_net == "net", edges)
    if edges:
        n_dirs = len(_core.cost_offsets(x.directions, False))
        return Transition(out.reshape(n_dirs, x.n_cells), x.grid, x.directions, False, x.matrix_values,
                          x._template)
    return x._wrap(out.reshape(x.shape), "passages")


def rsp_distance(x: Transition, from_coords, to_coords, theta: float, *, total_net: str = "net",
                 method: int = 1) -> np.ndarray:
    """Randomised-shortest-path distances between points: gdistance's ``rSPDistance()``.

    The expected cost (the sum of the reciprocals of the transition's values
    along the way) of the randomised shortest-path walk from each origin to
    each goal (see :func:`passage`): with ``theta`` near 0 close to a
    random-walk distance, with large ``theta`` close to the least-cost
    distance.

    Parameters
    ----------
    x
        A :class:`Transition` of conductances.
    from_coords, to_coords
        Points (coordinates, shapely points or a GeoDataFrame).
    theta
        0 to 20 (gdistance's range).
    total_net : {"net", "total"}
        Cost of the net passages (default) or of all of them.
    method : {1, 2}
        1: steps weighted by ``exp(-theta * cost) * P``; 2: by ``P ** theta``
        (a walk that only follows the random walk's probabilities).

    Returns
    -------
    numpy.ndarray
        ``(n_from, n_to)``; NaN for points off the grid.

    Notes
    -----
    gdistance factorises ``I - W`` once per pair of points; here once per
    goal (the goals in parallel), with a sparse Cholesky factorisation for
    symmetric transitions (``I - W`` scaled row by row is symmetric positive
    definite) and sparse LU otherwise. The results agree to about 1e-12.

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> tr = ls.cost.transition(np.ones((5, 5)), "mean", 4)
    >>> pts = [(0.5, 2.5), (4.5, 2.5)]
    >>> ls.cost.rsp_distance(tr, pts, pts, theta=10).round(3)    # 4 steps of cost 1 at best
    array([[0., 4.],
           [4., 0.]])
    """
    if not isinstance(x, Transition):
        raise TypeError("x must be a Transition (see ls.cost.transition)")
    theta = _check_theta(theta)
    if method not in (1, 2):
        raise ValueError("method should be either 1 or 2")
    if total_net not in ("net", "total"):
        raise ValueError("totalNet should be either total or net")
    fc = as_cells(from_coords, x.grid, "points")
    tc = as_cells(to_coords, x.grid, "points")
    return _core.cost_rsp_distances(x.values, x.grid.core, x.directions, x.symmetric, fc.tolist(), tc.tolist(),
                                    theta, total_net == "net", int(method))
