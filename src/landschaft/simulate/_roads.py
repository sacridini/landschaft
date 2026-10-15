# SPDX-License-Identifier: GPL-3.0-or-later
"""Endogenous road building: Dinamica EGO's road constructor.

New roads grow from the existing network towards attractive places, as in
the road constructor of DINAMICA's Amazon models (Soares-Filho et al. 2004,
2006): destinations are drawn in quadrants of an attraction map, among the
cells at a given range of distances from the roads, and each is joined to
the network down the cheapest way over a friction surface. Clean-room from
Dinamica EGO's documentation and black-box runs of its ``CalcCostMap``,
``CalcDestinationMap`` and ``CalcPathwayMap`` (8.13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from .. import _core
from ._maps import read_map

__all__ = ["RoadBuilder", "destinations", "friction_cost", "pathways"]

# Dinamica's direction codes and their (row, column) steps
DIRECTIONS = {1: (-1, -1), 2: (-1, 0), 3: (-1, 1), 4: (0, 1), 5: (1, 1), 6: (1, 0), 7: (1, -1), 8: (0, -1)}


def _res(m) -> tuple[float, float] | None:
    t = m.template
    if t is None:
        return None
    try:
        rx, ry = (abs(float(v)) for v in t.rio.resolution())
    except Exception:
        return None
    return rx, ry


def _wrap(m, values, name):
    t = m.template
    if t is None:
        return values
    return t.copy(data=values).rename(name)


def _present(m) -> np.ndarray:
    """Cells with a value other than 0 (NaN and nodata are absent)."""
    v = m.as_float()
    return np.ascontiguousarray((~np.isnan(v) & (v != 0)).astype(np.uint8))


def _seed(seed) -> int:
    if seed is None:
        return int(np.random.default_rng().integers(1, 2**31 - 1))
    return int(seed)


def friction_cost(sources, friction, *, diagonal: bool = True, relative: bool = False, directions: bool = False,
                  nodata=None):
    """Cost of reaching every cell from the nearest source over a friction surface.

    A step into a cell costs its friction, times sqrt(2) on a diagonal
    (``diagonal``), 8 neighbours; every cell gets the cost of its cheapest
    path from a source (Dinamica EGO's ``CalcCostMap``; Dijkstra's
    algorithm, the same sums as Dinamica's push-broom run to convergence).

    Parameters
    ----------
    sources : xarray.DataArray or array_like
        Cells with a value other than 0 are sources (cost 0).
    friction : xarray.DataArray or array_like
        The effort to cross each cell (> 0; NaN: impassable).
    diagonal : bool, default True
        Diagonal steps cost sqrt(2) times more.
    relative : bool, default False
        Frictions per map unit (metres): multiplied by the cell size
        (needs square cells).
    directions : bool, default False
        Also give the direction of each cell's predecessor on its cheapest
        path (Dinamica's codes 1 NW, 2 N, 3 NE, 4 E, 5 SE, 6 S, 7 SW, 8 W;
        0 at the sources).
    nodata
        Missing value of ``sources``.

    Returns
    -------
    xarray.DataArray or numpy.ndarray, or a tuple (cost, directions)
        NaN where the friction is missing or no source is reached
        (directions: 255).

    Notes
    -----
    Ties between predecessors go to the first neighbour in row order (NW, N,
    NE, W, E, SW, S, SE); Dinamica's follow the order of its sweeps.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> roads = np.zeros((3, 4)); roads[0, 0] = 1
    >>> ls.simulate.friction_cost(roads, np.ones((3, 4)), diagonal=False)
    array([[0., 1., 2., 3.],
           [1., 1., 2., 3.],
           [2., 2., 2., 3.]])
    """
    s = read_map(sources, nodata)
    f = read_map(friction)
    if f.shape != s.shape:
        raise ValueError(f"the friction has shape {f.shape}, the sources {s.shape}")
    fv = np.ascontiguousarray(f.as_float())
    if (fv <= 0).any():
        raise ValueError("frictions must be positive (NaN: impassable)")
    if relative:
        r = _res(s) or _res(f)
        if r is None:
            raise ValueError("relative frictions need a georeferenced map")
        if not math.isclose(r[0], r[1], rel_tol=1e-9):
            raise ValueError("relative frictions need square cells")
        fv = fv * r[0]
    cost, dirs = _core.sim_friction_cost(_present(s), fv, math.sqrt(2.0) if diagonal else 1.0, bool(directions))
    cost = np.asarray(cost)
    out = _wrap(s, cost, "cost")
    if not directions:
        return out
    d = np.asarray(dirs).copy()
    d[np.isnan(cost)] = 255
    return out, _wrap(s, d, "direction")


def destinations(attraction, quadrants=(1, 1), *, per_quadrant: int = 1, min_distance: float = 0.0,
                 max_attempts: int = 15, seed: int | None = None, nodata=None):
    """Cells drawn by attraction in quadrants of a map.

    The map is cut into ``quadrants = (horizontal, vertical)`` parts (the
    columns into ``horizontal``, the rows into ``vertical``); in each, up to
    ``per_quadrant`` cells are drawn in proportion to their attraction
    (cells of attraction <= 0 or missing never); a draw closer than
    ``min_distance`` (map units) to an earlier destination is drawn again,
    up to ``max_attempts`` times. Dinamica EGO's ``CalcDestinationMap``.

    Parameters
    ----------
    attraction : xarray.DataArray or array_like
    quadrants : (int, int)
        Horizontal (column) and vertical (row) divisions.
    per_quadrant : int
    min_distance : float
        In map units (cells without a georeference).
    max_attempts : int
    seed : int, optional

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        The attraction of the destinations, NaN elsewhere.

    Notes
    -----
    Dinamica EGO 8.13 gives each quadrant ``n // q`` rows or columns (the
    last ``n % q`` never get a destination) and draws an integer of the
    summed attraction with rounding (the first cell of a quadrant gets half
    its share, and a draw past the end takes the quadrant's last cell even
    with no attraction, about 1 / (2 x summed attraction) of the time).
    Here the quadrants cover the map (``floor(k n / q)`` boundaries) and
    the draw is in exact proportion to the attraction.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> a = np.zeros((4, 6)); a[1, 1] = 5; a[2, 4] = 1
    >>> d = ls.simulate.destinations(a, (2, 1), seed=1)
    >>> np.argwhere(~np.isnan(d)).tolist()
    [[1, 1], [2, 4]]
    """
    a = read_map(attraction, nodata)
    h, v = (int(q) for q in quadrants)
    if h < 1 or v < 1:
        raise ValueError("quadrants must be at least (1, 1)")
    if per_quadrant < 0 or max_attempts < 1:
        raise ValueError("per_quadrant must be >= 0 and max_attempts >= 1")
    dx, dy = _res(a) or (1.0, 1.0)
    values = np.ascontiguousarray(a.as_float())
    idx = np.asarray(_core.sim_destinations(values, h, v, int(per_quadrant), float(min_distance),
                                            int(max_attempts), dx, dy, _seed(seed) & 0xFFFFFFFFFFFFFFFF))
    out = np.full(values.shape, np.nan)
    out.flat[idx] = values.flat[idx]
    return _wrap(a, out, "destination")


def pathways(network, sources, cost=None, *, directions=None, lottery: bool = False, merge: bool = True,
             value: float = 1, seed: int | None = None, nodata=None):
    """Paths from source cells to a network down a cost surface.

    From each source (in row order) a path goes to the cheapest of its 8
    neighbours until a cell of cost <= 0 (the network, e.g. the sources of
    :func:`friction_cost`); or, given ``directions``, follows them until a
    cell of the network. Dinamica EGO's ``CalcPathwayMap``.

    Parameters
    ----------
    network : xarray.DataArray or array_like
        The network (cells with a value other than 0 or missing).
    sources : xarray.DataArray or array_like
        Where paths start (cells with a value other than 0), e.g.
        :func:`destinations`.
    cost : xarray.DataArray or array_like, optional
        The cost of reaching the network (``<= 0`` on it), e.g.
        :func:`friction_cost` from the network.
    directions : xarray.DataArray or array_like, optional
        Instead of ``cost``: the direction codes of :func:`friction_cost`.
    lottery : bool, default False
        Ties between cheapest neighbours drawn at random (else the first in
        row order: NW, N, NE, W, E, SW, S, SE).
    merge : bool, default True
        The network with the new paths; else only the new paths.
    value : float, default 1
        The value of the new path cells.
    seed : int, optional
        For the lottery.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        NaN off the network and the paths.

    Notes
    -----
    A source of missing cost is skipped; a local minimum of the costs (no
    cheaper neighbour before the network) is an error, as in Dinamica.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> net = np.zeros((3, 5)); net[:, 0] = 1
    >>> src = np.zeros((3, 5)); src[1, 4] = 1
    >>> cost = ls.simulate.friction_cost(net, np.ones((3, 5)))
    >>> np.nan_to_num(ls.simulate.pathways(net, src, cost, value=2))
    array([[1., 2., 2., 2., 0.],
           [1., 0., 0., 0., 2.],
           [1., 0., 0., 0., 0.]])
    """
    n = read_map(network, nodata)
    s = read_map(sources)
    if s.shape != n.shape:
        raise ValueError(f"the sources have shape {s.shape}, the network {n.shape}")
    if (cost is None) == (directions is None):
        raise ValueError("give the cost or the directions")
    c = d = None
    if cost is not None:
        c = np.ascontiguousarray(read_map(cost).as_float())
        if c.shape != n.shape:
            raise ValueError("the cost map differs in shape from the network")
    else:
        dv = read_map(directions).as_float()
        if dv.shape != n.shape:
            raise ValueError("the direction map differs in shape from the network")
        d = np.ascontiguousarray(np.where(np.isnan(dv), 255, dv).astype(np.uint8))
    new = np.asarray(_core.sim_pathways(_present(n), _present(s), c, d, bool(lottery), _seed(seed) & 0xFFFFFFFFFFFFFFFF))
    base = n.as_float()
    out = np.where(new > 0, float(value), np.where(_present(n) > 0, base, np.nan) if merge else np.nan)
    return _wrap(n, out, "network")


def _distance(present: np.ndarray, res: float) -> np.ndarray:
    if not present.any():
        return np.full(present.shape, np.inf)
    return np.asarray(_core.nlm_distance(np.ascontiguousarray(present.astype(np.uint8)), "exact", res),
                      dtype=np.float64)


@dataclass
class RoadBuilder:
    """New roads each step, from the network towards attractive places.

    Dinamica's road constructor (``ClassicRoadConstructor``): each step,
    the attraction is kept where the distance to the roads is within
    ``distance`` (and ``mask`` allows); destinations are drawn in quadrants
    (:func:`destinations`); each is joined to the network down the cost of
    reaching it over the friction (:func:`friction_cost`,
    :func:`pathways`, ties at random with ``lottery``), and the new roads
    are merged into the network.

    Parameters
    ----------
    friction : xarray.DataArray or array_like
        The effort to build across each cell (> 0; NaN: never).
    attraction : xarray.DataArray, array_like or callable
        How attractive each cell is as a road's end, or a function of the
        simulated map giving it (e.g. a potential of deforestation).
    quadrants : (int, int), default (2, 10)
        Horizontal and vertical divisions of the map (Dinamica's default).
    distance : (float, float), default (1500, 3000)
        Range of distances from the roads (map units) where new roads may
        end.
    spacing : float, default 1000
        Minimum distance between destinations of a step.
    per_quadrant, max_attempts : int
        As in :func:`destinations`.
    lottery : bool, default True
    diagonal, relative : bool
        As in :func:`friction_cost`.
    mask : xarray.DataArray, array_like or callable, optional
        Where roads may end (true / non-zero), or a function of the
        simulated map giving it.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> rng = np.random.default_rng(0)
    >>> net = np.zeros((40, 40)); net[0, :] = 1
    >>> rb = ls.simulate.RoadBuilder(np.ones((40, 40)), rng.random((40, 40)), quadrants=(2, 2),
    ...                              distance=(5, 20), spacing=5)
    >>> new = rb.step(net, seed=1)
    >>> int(np.nansum(new) > np.nansum(net))
    1
    """

    friction: object
    attraction: object
    quadrants: tuple = (2, 10)
    distance: tuple = (1500.0, 3000.0)
    spacing: float = 1000.0
    per_quadrant: int = 1
    max_attempts: int = 15
    lottery: bool = True
    diagonal: bool = True
    relative: bool = False
    mask: object = None
    _last: object = field(default=None, init=False, repr=False)

    def _map(self, x, landscape):
        if callable(x):
            if landscape is None:
                raise ValueError("a function of the map needs the landscape")
            x = x(landscape)
        return read_map(x).as_float()

    def step(self, network, *, landscape=None, seed: int | None = None):
        """One step of building: the network with the new roads.

        ``landscape`` feeds ``attraction`` and ``mask`` when they are
        functions of the map.
        """
        n = read_map(network)
        present = _present(n) > 0
        res = _res(n)
        if res is not None and not math.isclose(res[0], res[1], rel_tol=1e-9):
            raise ValueError("road building needs square cells")
        cell = res[0] if res is not None else 1.0
        seed = _seed(seed)
        dist = _distance(present, cell)
        att = self._map(self.attraction, landscape)
        if att.shape != dist.shape:
            raise ValueError("the attraction differs in shape from the network")
        lo, hi = self.distance
        valid = (dist >= lo) & (dist <= hi) & ~present
        if self.mask is not None:
            m = self._map(self.mask, landscape)
            valid &= ~np.isnan(m) & (m != 0)
        att = np.where(valid, att, np.nan)
        template = n.template
        wrap = (lambda v: v) if template is None else (lambda v: template.copy(data=v))
        dest = destinations(wrap(att), self.quadrants, per_quadrant=self.per_quadrant, min_distance=self.spacing,
                            max_attempts=self.max_attempts, seed=seed)
        cost = friction_cost(wrap(present.astype(np.float64)), self.friction, diagonal=self.diagonal,
                             relative=self.relative)
        return pathways(network, dest, cost, lottery=self.lottery, merge=True, seed=seed + 1)

    def run(self, network, steps: int, *, landscape=None, seed: int | None = None) -> xr.DataArray:
        """``steps`` steps of building: the network at the start and after each step."""
        seed = _seed(seed)
        first = read_map(network)
        nets = [np.where(_present(first) > 0, first.as_float(), np.nan)]
        cur = network
        for k in range(1, int(steps) + 1):
            cur = self.step(cur, landscape=landscape, seed=(seed + k * 0x9E3779B9) & 0x7FFFFFFFFFFFFFFF)
            nets.append(read_map(cur).as_float())
        t = read_map(network).template
        if t is None:
            return xr.DataArray(np.stack(nets), dims=("step", "y", "x"), coords={"step": list(range(len(nets)))},
                                name="network")
        return xr.concat([t.copy(data=v) for v in nets], dim="step").assign_coords(
            step=list(range(len(nets)))).rename("network")

    def dynamic(self, network, *, seed: int | None = None) -> RoadDistance:
        """The distance to the roads as a dynamic variable of a :class:`Model` that builds roads each step."""
        return RoadDistance(self, network, _seed(seed))


class RoadDistance:
    """Distance to a growing road network: a dynamic variable (see :meth:`RoadBuilder.dynamic`).

    Called with a map it gives the distance (map units) to the roads; called
    with a different map than the last time (the next step of a
    simulation), it first builds a step of roads, with the map feeding the
    builder's attraction and mask. ``networks`` keeps the network of each
    step; :meth:`reset` goes back to the first.
    """

    def __init__(self, builder: RoadBuilder, network, seed: int):
        self.builder = builder
        self.seed = int(seed)
        self._first = network
        self.reset()

    def reset(self) -> None:
        self.networks = [self._first]
        self._last = None

    def __call__(self, landscape, nodata=None):
        land = read_map(landscape, nodata)
        v = land.as_float()
        if self._last is not None and not np.array_equal(self._last, v, equal_nan=True):
            k = len(self.networks)
            self.networks.append(self.builder.step(self.networks[-1], landscape=landscape,
                                                   seed=(self.seed + k * 0x9E3779B9) & 0x7FFFFFFFFFFFFFFF))
        self._last = v.copy()
        n = read_map(self.networks[-1])
        res = _res(n) or _res(land)
        out = _distance(_present(n) > 0, res[0] if res is not None else 1.0)
        out[np.isnan(v)] = np.nan
        t = land.template
        return out if t is None else t.copy(data=out).rename("distance_to_roads")

    def __repr__(self) -> str:
        return f"RoadDistance({len(self.networks) - 1} steps built)"
