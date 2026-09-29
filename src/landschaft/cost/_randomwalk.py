# SPDX-License-Identifier: GPL-3.0-or-later
"""Random walks and randomised shortest paths (gdistance's commuteDistance, passage, rSPDistance)."""

from __future__ import annotations

import numpy as np

from .. import _core
from ._points import as_cells
from ._transition import Transition

__all__ = ["commute_distance", "passage"]


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


def passage(x: Transition, origin, goal, theta: float | None = None, *, total_net: str = "net",
            output: str = "raster", gdistance_compat: bool = True):
    """Passages of a random walk from origins to goals: gdistance's ``passage()``.

    Without ``theta``: the net number of times a random walk from the origin
    (one of them at random) to the goal passes through each cell, which is
    the current through the cell when a unit current flows from the origins
    to the goals (as in Circuitscape) — gdistance's ``passage(x, origin,
    goal)``, net passages as a raster.

    Parameters
    ----------
    x
        A symmetric :class:`Transition` of conductances.
    origin, goal
        Points, or rasters whose non-zero cells are the origins / goals.
    theta
        Not yet available here (randomised shortest paths; see the roadmap).
    total_net, output
        ``"net"`` and ``"raster"`` (the only choices without ``theta``, as in
        gdistance).
    gdistance_compat
        True (default): as gdistance, the circuit's grounded cell (the last
        cell with connections) takes the potential of the one before it, which
        changes the passages around that cell; False: the exact currents.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Net passages per cell (1 at the origins and goals, 0 off the graph).

    Examples
    --------
    >>> import landschaft as ls
    >>> import numpy as np
    >>> tr = ls.cost.transition(np.ones((3, 3)), "mean", 4)
    >>> ls.cost.passage(tr, (0.5, 1.5), (2.5, 1.5), gdistance_compat=False).round(3)
    array([[0.25, 0.25, 0.25],
           [1.  , 0.5 , 1.  ],
           [0.25, 0.25, 0.25]])
    """
    x = _symmetric(x, "passage")
    if theta is not None:
        raise NotImplementedError("passage with theta (randomised shortest paths) is not available yet")
    if total_net != "net" or output != "raster":
        raise ValueError("without theta only total_net='net' and output='raster' are available "
                         "(gdistance: 'no method available -- try a low value of theta instead')")
    oc = as_cells(origin, x.grid, "origins")
    gc = as_cells(goal, x.grid, "goals")
    oc, gc = oc[oc >= 0], gc[gc >= 0]
    out = _core.cost_rw_passages(x.values, x.grid.core, x.directions, x.symmetric, oc.tolist(), gc.tolist(),
                                 bool(gdistance_compat))
    return x._wrap(out.reshape(x.shape), "passages")
