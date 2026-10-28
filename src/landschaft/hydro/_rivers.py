# SPDX-License-Identifier: GPL-3.0-or-later
"""Rivers as a network, as hydrographr does on the Hydrography90m and riverconn measures it: points moved onto the
streams, the land upstream of each link or point summarised, the catchment of each point, distances along the
network, the stretches up and down the river from each point, the centrality of the links, and the dendritic
connectivity index of a network cut by barriers."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import threaded
from ..surface._grid import read_surface
from ._basins import _cells_lonlat, _down, _points
from ._flow import _cells, _d8, _read

__all__ = ["DCI", "Catchments", "NetworkReach", "Upstream", "along_network", "centrality", "dci", "network_distance",
           "snap_points", "upstream", "upstream_catchment"]

_STATS = ("count", "sum", "mean", "min", "max", "std")


# -- the network ------------------------------------------------------------------------------------------------------

class _Net:
    """A DEM's D8 network: the directions, the cell each cell drains to and the step to it."""

    def __init__(self, dem, flats, extent, crs, band, nodata):
        self.s = _read(dem, extent, crs, band, nodata)
        self.d = _d8(self.s, flats)
        self.down = _down(self.d)
        self.dx, self.dy = _cells(self.s)
        ny, nx = self.s.shape
        self.nx = nx
        k = np.arange(ny * nx)
        to = self.down
        ok = to >= 0
        diag = math.hypot(self.dx, self.dy)
        vertical = (to // nx) != (k // nx)
        horizontal = (to % nx) != (k % nx)
        self.step = np.where(vertical & horizontal, diag, np.where(vertical, self.dy, self.dx))
        # the smaller side out of the grid (as stream_links' lengths)
        self.step = np.where(ok, self.step, min(self.dx, self.dy))
        self.valid = (self.d != -2).ravel()

    def on_grid(self, x, what: str) -> np.ndarray:
        v = read_surface(x).values
        if v.shape != self.s.shape:
            raise ValueError(f"the {what} differ in shape from the DEM: ls.match_grid makes them one grid")
        return np.asarray(v, dtype=np.float64).ravel()

    def label(self, seed: np.ndarray) -> np.ndarray:
        """Every cell labelled with the first seed at or below it (0: none)."""
        return _core.hydro_watersheds(self.d, seed.reshape(self.s.shape).astype(np.int32)).ravel()


class _Grid:
    """The cells of a surface in its coordinates."""

    def __init__(self, s):
        self.ny, self.nx = s.shape
        self.e = s.extent
        self.rx = (self.e[1] - self.e[0]) / self.nx
        self.ry = (self.e[3] - self.e[2]) / self.ny

    def centre(self, k):
        k = np.asarray(k)
        return self.e[0] + (k % self.nx + 0.5) * self.rx, self.e[3] - (k // self.nx + 0.5) * self.ry

    def at(self, xy: np.ndarray) -> np.ndarray:
        """The cell each point falls in (-1 off the grid)."""
        c = np.floor((xy[:, 0] - self.e[0]) / self.rx)
        r = np.floor((self.e[3] - xy[:, 1]) / self.ry)
        ok = (c >= 0) & (r >= 0) & (c < self.nx) & (r < self.ny) & np.isfinite(c) & np.isfinite(r)
        return np.where(ok, np.where(ok, r, 0) * self.nx + np.where(ok, c, 0), -1).astype(np.int64)

    def snap(self, xy: np.ndarray, on: np.ndarray, distance: float) -> np.ndarray:
        """The cell each point moves to on the cells ``on``: the nearest within the distance, of equally near ones the
        first in raster order (-1: none)."""
        on = on.reshape(self.ny, self.nx)
        reach = distance * distance * (1.0 + 1e-12)
        out = np.full(len(xy), -1, dtype=np.int64)
        for i, (x, y) in enumerate(xy):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            span = []
            for lo, size, n in ((self.e[3] - y, self.ry, self.ny), (x - self.e[0], self.rx, self.nx)):
                a, b = (lo - distance) / size, (lo + distance) / size
                if b < 0 or a >= n:
                    break
                span.append((max(math.floor(a), 0), min(math.floor(b), n - 1)))
            if len(span) < 2:
                continue
            (r0, r1), (c0, c1) = span
            rr, cc = np.nonzero(on[r0:r1 + 1, c0:c1 + 1])
            if rr.size == 0:
                continue
            k = (rr + r0) * self.nx + (cc + c0)
            cx, cy = self.centre(k)
            d2 = (cx - x) * (cx - x) + (cy - y) * (cy - y)
            j = int(np.argmin(d2))
            if d2[j] <= reach:
                out[i] = k[j]
        return out


def _snap_distance(distance, s) -> float:
    if distance is not None:
        d = float(distance)
        if not d >= 0:
            raise ValueError("the snapping distance must be 0 or more")
        return d
    g = _Grid(s)
    return 300.0 if s.crs is not None and not _cells_lonlat(s) else 3.0 * max(g.rx, g.ry)


def _stream_cells(values: np.ndarray) -> np.ndarray:
    return ~np.isnan(values) & (values != 0)


def _point_cells(net: _Net, points, streams, snap) -> tuple[np.ndarray, np.ndarray]:
    """The points' coordinates and cells: moved onto the streams when given, else the cell each falls in."""
    xy = _points(points, net.s)
    g = _Grid(net.s)
    if streams is None:
        cells = g.at(xy)
        why = "off the grid"
    else:
        distance = _snap_distance(snap, net.s)
        cells = g.snap(xy, _stream_cells(net.on_grid(streams, "streams")), distance)
        why = f"farther than {distance} from the streams"
    lost = np.nonzero(cells < 0)[0]
    if lost.size:
        warnings.warn(f"point(s) {', '.join(str(i + 1) for i in lost)} {why}: left without values", stacklevel=3)
    return xy, cells


def _attributes(points) -> pd.DataFrame:
    if hasattr(points, "geometry"):
        return pd.DataFrame(points.drop(columns=points.geometry.name)).reset_index(drop=True)
    return pd.DataFrame(index=pd.RangeIndex(len(np.atleast_2d(points))))


def _geo(attrs: pd.DataFrame, geometry, s):
    import geopandas as gpd

    return gpd.GeoDataFrame(attrs, geometry=list(geometry), crs=s.crs)


# -- snapping ---------------------------------------------------------------------------------------------------------

def snap_points(points, streams, *, distance: float | None = None, accumulation=None,
                min_accumulation: float | None = None):
    """Points moved onto the stream network, as hydrographr's ``snap_to_network``.

    Each point (a gauge, a dam, a record of a fish) moves to the centre of
    the nearest stream cell within ``distance`` (of equally near ones the
    first in raster order), or of the nearest whose flow accumulation is at
    least ``min_accumulation`` (on the river, not a tributary beside it). A
    point farther than the distance stays where it was, ``snapped`` 0.

    Parameters
    ----------
    points
        A GeoDataFrame of points, or (x, y) pairs in the streams' coordinates.
    streams
        The streams: non-zero on their cells (:func:`streams`, the links of
        :func:`stream_links`).
    distance
        The largest move, map units (default 300 m; 3 cells without a
        projection).
    accumulation, min_accumulation
        A flow accumulation, and the smallest a stream cell must have.

    Returns
    -------
    geopandas.GeoDataFrame
        The points moved, their attributes and ``row``, ``col``, ``stream``
        (the streams' value there), ``moved`` (the distance) and ``snapped``
        (1 or 0), in the streams' coordinate system.

    Examples
    --------
    >>> import numpy as np, landschaft as ls
    >>> dem = np.array([[5, 4, 5], [5, 3, 5], [5, 2, 5], [5, 1, 5]], float)
    >>> st = ls.hydro.streams(ls.hydro.flow_accumulation(dem), 2)
    >>> p = ls.hydro.snap_points([(0.9, 1.5), (9.0, 9.0)], st, distance=1.0)
    >>> p[["row", "col", "moved", "snapped"]].to_numpy().tolist()
    [[2.0, 1.0, 0.6, 1.0], [nan, nan, nan, 0.0]]
    """
    s = read_surface(streams)
    st = np.asarray(s.values, dtype=np.float64).ravel()
    on = _stream_cells(st)
    if min_accumulation is not None:
        if accumulation is None:
            raise ValueError("min_accumulation needs the accumulation")
        a = read_surface(accumulation).values
        if a.shape != s.shape:
            raise ValueError("the accumulation differs in shape from the streams: ls.match_grid makes them one grid")
        on &= np.asarray(a, dtype=np.float64).ravel() >= float(min_accumulation)
    xy = _points(points, s)
    distance = _snap_distance(distance, s)
    g = _Grid(s)
    to = g.snap(xy, on, distance)
    lost = np.nonzero(to < 0)[0]
    if lost.size:
        warnings.warn(f"point(s) {', '.join(str(i + 1) for i in lost)} farther than {distance} from the streams: "
                      "not moved", stacklevel=2)
    ok = to >= 0
    cx, cy = g.centre(np.maximum(to, 0))
    x = np.where(ok, cx, xy[:, 0])
    y = np.where(ok, cy, xy[:, 1])
    nan = np.full(len(to), np.nan)
    attrs = _attributes(points)
    for name, col in (("row", np.where(ok, to // g.nx, nan)), ("col", np.where(ok, to % g.nx, nan)),
                      ("stream", np.where(ok, st[np.maximum(to, 0)], nan)),
                      ("moved", np.where(ok, np.hypot(cx - xy[:, 0], cy - xy[:, 1]), nan)),
                      ("snapped", ok.astype(np.float64))):
        attrs[name] = col
    import geopandas as gpd

    return _geo(attrs, gpd.points_from_xy(x, y), s)


# -- the land upstream ------------------------------------------------------------------------------------------------

class _Groups:
    """Places on the network as groups of seed cells (a link's last cells, a point's cell), numbered 1..; every cell
    labelled with the first group at or below it, and the group each group drains to."""

    def __init__(self, net: _Net, seed: np.ndarray, groups: int):
        self.seed, self.groups = seed, groups
        self.label = net.label(seed)
        self.below = np.zeros(groups + 1, dtype=np.int64)
        down = net.down
        for k in np.nonzero(seed > 0)[0]:
            g = seed[k]
            if self.below[g] == 0 and down[k] >= 0:
                b = self.label[down[k]]
                if b != g:
                    self.below[g] = b

    def upward_first(self) -> list[int]:
        """The groups in an order where each comes before the one it drains to."""
        ups = np.bincount(self.below[1:], minlength=self.groups + 1)
        order = [g for g in range(1, self.groups + 1) if ups[g] == 0]
        i = 0
        while i < len(order):
            b = self.below[order[i]]
            if b > 0:
                ups[b] -= 1
                if ups[b] == 0:
                    order.append(int(b))
            i += 1
        return order

    def above(self) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in range(self.groups + 1)]
        for g in range(1, self.groups + 1):
            out[self.below[g]].append(g)
        return out


def _layers(values, net: _Net) -> tuple[list[str], list[np.ndarray]]:
    if values is None:
        return [], []
    if isinstance(values, dict):
        return [str(k) for k in values], [net.on_grid(v, "values") for v in values.values()]
    if isinstance(values, (list, tuple)):
        return [f"layer{i + 1}" for i in range(len(values))], [net.on_grid(v, "values") for v in values]
    if isinstance(values, xr.DataArray) and values.ndim == 3:
        dim = values.dims[0]
        names = [str(v) for v in values[dim].values] if dim in values.coords else [
            f"layer{i + 1}" for i in range(values.shape[0])]
        return names, [net.on_grid(values[i], "values") for i in range(values.shape[0])]
    a = values if isinstance(values, xr.DataArray) else np.asarray(values)
    if a.ndim == 3:
        return [f"layer{i + 1}" for i in range(a.shape[0])], [net.on_grid(a[i], "values") for i in range(a.shape[0])]
    return ["values"], [net.on_grid(a, "values")]


def _summaries(grp: _Groups, valid: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """The statistics of a layer over each group's own cells, then joined from the sources down (Chan et al.'s
    pairwise update of the mean and the squared deviations)."""
    n = grp.groups + 1
    k = np.nonzero((grp.label > 0) & valid & ~np.isnan(v))[0]
    g, x = grp.label[k], v[k]
    cnt = np.bincount(g, minlength=n).astype(np.float64)
    tot = np.bincount(g, weights=x, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, tot / cnt, 0.0)
    m2 = np.bincount(g, weights=(x - mean[g]) ** 2, minlength=n)
    lo, hi = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(lo, g, x)
    np.maximum.at(hi, g, x)
    for a in grp.upward_first():
        b = grp.below[a]
        if b == 0 or cnt[a] == 0:
            continue
        nb = cnt[b] + cnt[a]
        d = mean[a] - mean[b]
        mean[b] += d * cnt[a] / nb
        m2[b] += m2[a] + d * d * cnt[b] * cnt[a] / nb
        cnt[b] = nb
        tot[b] += tot[a]
        lo[b] = min(lo[b], lo[a])
        hi[b] = max(hi[b], hi[a])
    empty = cnt == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"count": cnt, "sum": tot, "mean": np.where(empty, np.nan, mean), "min": np.where(empty, np.nan, lo),
                "max": np.where(empty, np.nan, hi), "std": np.where(empty, np.nan, np.sqrt(m2 / cnt))}


def _upland_counts(grp: _Groups, cells: np.ndarray) -> np.ndarray:
    """A count per group of its own cells (``cells`` a mask), joined from the sources down."""
    c = np.bincount(grp.label[cells], minlength=grp.groups + 1).astype(np.float64)
    for a in grp.upward_first():
        if grp.below[a] > 0:
            c[grp.below[a]] += c[a]
    return c


def _name(c: float) -> str:
    return str(int(c)) if float(c).is_integer() else str(c)


def _link_groups(net: _Net, links: np.ndarray):
    ids = np.unique(links[~np.isnan(links) & (links != 0)])
    if np.any(ids < 0) or np.any(ids != np.round(ids)):
        raise ValueError("the links are positive whole numbers (stream_links' map)")
    group = np.zeros(len(links), dtype=np.int64)
    on = ~np.isnan(links) & (links != 0)
    group[on] = np.searchsorted(ids, links[on]) + 1
    down = net.down
    nxt = np.where(down >= 0, links[np.maximum(down, 0)], np.nan)
    last = on & ((down < 0) | (nxt != links))
    seed = np.where(last, group, 0)
    rows = [(float(l), i + 1, -1) for i, l in enumerate(ids)]
    return seed, len(ids), rows


def _point_groups(cells: np.ndarray, n: int):
    seed = np.zeros(n, dtype=np.int64)
    groups = 0
    rows = []
    for i, k in enumerate(cells):
        g = None
        if k >= 0:
            if seed[k] == 0:
                groups += 1
                seed[k] = groups
            g = int(seed[k])
        rows.append((float(i + 1), g, int(k)))
    return seed, groups, rows


def _row_below(rows, grp: _Groups, g) -> float:
    if g is None:
        return np.nan
    b = grp.below[g]
    if b == 0:
        return 0.0
    return next((r[0] for r in rows if r[1] == b), 0.0)


@dataclass
class Upstream(Saveable):
    """The land upstream of links or points (:func:`upstream`).

    Attributes
    ----------
    table : pandas.DataFrame
        A row per link (``link``, ``downstream``) or point (``point``,
        ``downstream``, ``row``, ``col``), then ``cells``, ``area`` (map units
        squared; m² for longitude / latitude), the statistics of the values
        (``<layer>_<statistic>`` of several layers) and the shares of the
        classes (``share_<class>``).
    map : xarray.DataArray
        The column ``to_map`` painted on each link's cells, or on each point's
        cell.
    """

    table: pd.DataFrame
    map: xr.DataArray


@threaded
def upstream(dem, *, links=None, points=None, streams=None, snap: float | None = None, values=None, classes=None,
             stats=("mean",), to_map: str | None = None, flats: bool = True, extent=None, crs=None, band=None,
             nodata=None) -> Upstream:
    """The land upstream of every link of a stream network, or of every point on it, summarised.

    As hydrographr's ``get_upstream_variable``: the statistics of maps over
    all the cells that drain through a link's last cell or a point's cell
    along the D8 directions (the mean slope, the rain summed), and the share
    of each class of a map (the forest, the cities upstream); its cells and
    area. A map's cells without a value are left out of its statistics.

    Parameters
    ----------
    dem
        The elevations, best with depressions filled (:func:`fill`).
    links
        The links of :func:`stream_links`: a row per link.
    points
        Or points (gauges, dams, sites; a GeoDataFrame or (x, y) pairs): a row
        per point.
    streams
        With points: the streams they are moved onto (non-zero on their
        cells); without them, the cell each falls in.
    snap
        The largest move of a point onto the streams, map units (300 m; 3
        cells without a projection).
    values
        A map, a stack of maps (a band per layer), a list or a dict of maps,
        whose statistics are given.
    classes
        A map of classes whose shares are given.
    stats
        Of the values: ``count``, ``sum``, ``mean``, ``min``, ``max``, ``std``
        (a list, or names joined by commas).
    to_map
        The column of the map (default: the first after the area).
    flats, extent, crs, band, nodata
        As :func:`stream_links`.

    Returns
    -------
    Upstream

    Examples
    --------
    >>> import numpy as np, landschaft as ls
    >>> dem = np.array([[5, 4, 5], [5, 3, 5], [5, 2, 5], [5, 1, 5]], float)
    >>> st = ls.hydro.streams(ls.hydro.flow_accumulation(dem), 2)
    >>> links, _ = ls.hydro.stream_links(dem, st)
    >>> rain = np.arange(12.0).reshape(4, 3)
    >>> ls.hydro.upstream(dem, links=links, values=rain, stats="sum, mean").table.to_numpy().tolist()
    [[1.0, 0.0, 12.0, 12.0, 66.0, 5.5]]
    """
    net = _Net(dem, flats, extent, crs, band, nodata)
    n = net.s.shape[0] * net.s.shape[1]
    if (links is None) == (points is None):
        raise ValueError("give the links or the points (one of them)")
    if isinstance(stats, str):
        stats = [x.strip() for x in stats.split(",") if x.strip()]
    stats = list(stats)
    bad = [x for x in stats if x not in _STATS]
    if bad:
        raise ValueError(f"unknown statistic {bad[0]!r} (count, sum, mean, min, max, std)")
    by_link = links is not None
    if by_link:
        lv = net.on_grid(links, "links")
        seed, groups, rows = _link_groups(net, lv)
    else:
        _, cells = _point_cells(net, points, streams, snap)
        seed, groups, rows = _point_groups(cells, n)
    grp = _Groups(net, seed, groups)
    names, layers = _layers(values, net)
    columns = ["link", "downstream"] if by_link else ["point", "downstream", "row", "col"]
    columns += ["cells", "area"]
    first = len(columns)
    data: dict[str, np.ndarray] = {}
    for name, v in zip(names, layers, strict=True):
        summ = _summaries(grp, net.valid, v)
        for st in stats:
            col = f"{name}_{st}" if len(names) > 1 else st
            columns.append(col)
            data[col] = summ[st]
    cells_up = _upland_counts(grp, (grp.label > 0) & net.valid)
    if classes is not None:
        cv = net.on_grid(classes, "classes")
        ids = np.unique(cv[~np.isnan(cv)])
        mask = (grp.label > 0) & net.valid
        classified = _upland_counts(grp, mask & ~np.isnan(cv))
        for c in ids:
            col = f"share_{_name(c)}"
            columns.append(col)
            with np.errstate(invalid="ignore", divide="ignore"):
                data[col] = np.where(classified > 0, _upland_counts(grp, mask & (cv == c)) / classified, np.nan)
    area = net.dx * net.dy
    out = []
    for rid, g, cell in rows:
        row = [rid, _row_below(rows, grp, g)]
        if not by_link:
            row += [cell // net.nx if cell >= 0 else np.nan, cell % net.nx if cell >= 0 else np.nan]
        if g is None:
            row += [np.nan] * (len(columns) - len(row))
        else:
            row += [cells_up[g], cells_up[g] * area] + [data[c][g] for c in columns[first:]]
        out.append(row)
    table = pd.DataFrame(out, columns=columns, dtype=np.float64)
    to_map = to_map or (columns[first] if len(columns) > first else "cells")
    if to_map not in table:
        raise ValueError(f"to_map: no column {to_map!r} (of {', '.join(columns)})")
    of_group = np.full(groups + 1, np.nan)
    for (_, g, _), value in zip(rows, table[to_map].to_numpy(), strict=True):
        if g is not None and np.isnan(of_group[g]):
            of_group[g] = value
    if by_link:
        on = ~np.isnan(lv) & (lv != 0)
        painted = np.where(on, of_group[grp.label], np.nan)
    else:
        painted = np.where(seed > 0, of_group[seed], np.nan)
    return Upstream(table, net.s.raster(painted.reshape(net.s.shape), to_map))


# -- catchments -------------------------------------------------------------------------------------------------------

@dataclass
class Catchments(Saveable):
    """The catchments of points (:func:`upstream_catchment`).

    Attributes
    ----------
    polygons : geopandas.GeoDataFrame
        A polygon per point: its attributes and ``point`` (1..),
        ``downstream`` (the point below, 0 none), ``cells``, ``area``.
    table : pandas.DataFrame
        ``point``, ``downstream``, ``row``, ``col``, ``cells``, ``area``.
    map : xarray.DataArray
        The first point each cell drains to (1..; 0 none): the catchments
        between one point and the next.
    """

    polygons: object
    table: pd.DataFrame
    map: xr.DataArray


def _polygons(mask: np.ndarray, transform):
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    parts = [shape(geom) for geom, _ in shapes(mask.astype(np.uint8), mask=mask, transform=transform)]
    return unary_union(parts) if parts else None


@threaded
def upstream_catchment(dem, points, *, streams=None, snap: float | None = None, flats: bool = True, extent=None,
                       crs=None, band=None, nodata=None) -> Catchments:
    """The catchment of every point (a gauge, a dam, a site), as hydrographr's ``get_upstream_catchment``.

    All the land that drains to the point along the D8 directions, as a
    polygon per point (the catchments of points one above another overlap),
    and a map of each cell's first point downstream (the catchments nested,
    between one point and the next).

    Parameters
    ----------
    dem
        The elevations, best with depressions filled (:func:`fill`).
    points
        The points: a GeoDataFrame or (x, y) pairs.
    streams, snap
        The streams the points are moved onto, and the largest move (as
        :func:`upstream`); without streams, the cell each falls in.
    flats, extent, crs, band, nodata
        As :func:`stream_links`.

    Returns
    -------
    Catchments

    Examples
    --------
    >>> import numpy as np, landschaft as ls
    >>> dem = np.array([[5, 4, 5], [5, 3, 5], [5, 2, 5], [5, 1, 5]], float)
    >>> c = ls.hydro.upstream_catchment(dem, [(1.5, 0.5), (1.5, 2.5)])
    >>> c.table[["point", "downstream", "cells"]].to_numpy().tolist()
    [[1.0, 0.0, 12.0], [2.0, 1.0, 4.0]]
    >>> float(c.polygons.area[1])
    4.0
    """
    from affine import Affine

    net = _Net(dem, flats, extent, crs, band, nodata)
    ny, nx = net.s.shape
    _, cells = _point_cells(net, points, streams, snap)
    seed, groups, rows = _point_groups(cells, ny * nx)
    grp = _Groups(net, seed, groups)
    cells_up = _upland_counts(grp, (grp.label > 0) & net.valid)
    above = grp.above()
    g = _Grid(net.s)
    label2 = grp.label.reshape(ny, nx)
    valid2 = net.valid.reshape(ny, nx)
    geoms, table = [], []
    for rid, gr, cell in rows:
        below = _row_below(rows, grp, gr)
        r, c = (cell // nx, cell % nx) if cell >= 0 else (np.nan, np.nan)
        if gr is None:
            geoms.append(None)
            table.append([rid, below, r, c, np.nan, np.nan])
            continue
        members = [gr]
        i = 0
        while i < len(members):
            members.extend(above[members[i]])
            i += 1
        mask = np.isin(label2, members) & valid2
        rr, cc = np.nonzero(mask)
        geom = None
        if rr.size:
            r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
            t = Affine(g.rx, 0.0, g.e[0] + c0 * g.rx, 0.0, -g.ry, g.e[3] - r0 * g.ry)
            geom = _polygons(mask[r0:r1 + 1, c0:c1 + 1], t)
        geoms.append(geom)
        table.append([rid, below, r, c, cells_up[gr], cells_up[gr] * net.dx * net.dy])
    table = pd.DataFrame(table, columns=["point", "downstream", "row", "col", "cells", "area"], dtype=np.float64)
    attrs = _attributes(points)
    for name in ("point", "downstream", "cells", "area"):
        attrs[name] = table[name].to_numpy()
    first = np.zeros(groups + 1)
    for rid, gr, _ in reversed(rows):
        if gr is not None:
            first[gr] = rid
    m = np.where(net.valid, first[grp.label], np.nan).reshape(ny, nx)
    return Catchments(_geo(attrs, geoms, net.s), table, net.s.raster(m, "catchment"))


# -- distances along the network --------------------------------------------------------------------------------------

@threaded
def network_distance(dem, points, *, streams=None, snap: float | None = None, flats: bool = True, extent=None,
                     crs=None, band=None, nodata=None) -> pd.DataFrame:
    """The distance along the river network between every two points, as hydrographr's ``get_distance``.

    Down the D8 directions from each point to where their waters meet (fish
    swimming between two gauges, two populations), and the straight
    distance; whether one drains into the other. Points in different basins
    are not connected.

    Parameters
    ----------
    dem
        The elevations, best with depressions filled (:func:`fill`).
    points
        The points: a GeoDataFrame or (x, y) pairs.
    streams, snap
        The streams the points are moved onto, and the largest move (as
        :func:`upstream`).
    flats, extent, crs, band, nodata
        As :func:`stream_links`.

    Returns
    -------
    pandas.DataFrame
        A row per two points: ``from``, ``to`` (1..), ``network`` (map units;
        m for longitude / latitude; NaN when not connected), ``euclidean``,
        ``connected`` and ``flow`` (1 when one drains into the other).

    Examples
    --------
    >>> import numpy as np, landschaft as ls
    >>> dem = np.array([[5, 4, 5], [5, 3, 5], [5, 2, 5], [5, 1, 5]], float)
    >>> ls.hydro.network_distance(dem, [(1.5, 3.5), (1.5, 0.5)]).to_numpy().tolist()
    [[1.0, 2.0, 3.0, 3.0, 1.0, 1.0]]
    """
    net = _Net(dem, flats, extent, crs, band, nodata)
    _, cells = _point_cells(net, points, streams, snap)
    network, connected, flow = _core.hydro_river_distances(net.down, net.step, cells)
    p = len(cells)
    i, j = np.triu_indices(p, 1)
    g = _Grid(net.s)
    cx, cy = g.centre(np.maximum(cells, 0))
    ex, ey = net.dx / g.rx, net.dy / g.ry
    ok = (cells[i] >= 0) & (cells[j] >= 0)
    euclid = np.where(ok, np.hypot((cx[i] - cx[j]) * ex, (cy[i] - cy[j]) * ey), np.nan)
    return pd.DataFrame({"from": i + 1, "to": j + 1, "network": network, "euclidean": euclid,
                         "connected": connected.astype(np.int64), "flow": flow.astype(np.int64)})


@dataclass
class NetworkReach(Saveable):
    """The stretches of river up and down from points (:func:`along_network`).

    Attributes
    ----------
    map : xarray.DataArray
        The point (1..) nearest along the network to each cell within reach,
        0 elsewhere.
    distance : xarray.DataArray
        The distance along the network to that point (map units; m for
        longitude / latitude), NaN elsewhere.
    """

    map: xr.DataArray
    distance: xr.DataArray


@threaded
def along_network(dem, points, streams, *, upstream: float = 0.0, downstream: float = 0.0, snap: float | None = None,
                  flats: bool = True, extent=None, crs=None, band=None, nodata=None) -> NetworkReach:
    """The stretches of river up and down from every point within a distance along the network.

    As hydrographr's ``get_buffer_along_the_network``: downstream along the
    D8 directions, upstream along every stream that flows into it (a buffer
    that follows the river, not a circle: the reach of a fish, of a spill
    below a mine). Each cell is given the point nearest to it along the
    network.

    Parameters
    ----------
    dem
        The elevations, best with depressions filled (:func:`fill`).
    points
        The points (a GeoDataFrame or (x, y) pairs), moved onto the streams.
    streams
        The streams: non-zero on their cells.
    upstream, downstream
        How far up and down the river, map units (m for longitude / latitude).
    snap
        The largest move of a point onto the streams (as :func:`upstream`).
    flats, extent, crs, band, nodata
        As :func:`stream_links`.

    Returns
    -------
    NetworkReach

    Examples
    --------
    >>> import numpy as np, landschaft as ls
    >>> dem = np.array([[5, 4, 5], [5, 3, 5], [5, 2, 5], [5, 1, 5]], float)
    >>> st = ls.hydro.streams(ls.hydro.flow_accumulation(dem), 2)
    >>> r = ls.hydro.along_network(dem, [(1.5, 1.5)], st, upstream=1, downstream=1)
    >>> r.distance.values[:, 1].tolist()
    [nan, 1.0, 0.0, 1.0]
    """
    up, dn = float(upstream), float(downstream)
    if not (up >= 0 and dn >= 0):
        raise ValueError("upstream and downstream are distances of 0 or more")
    net = _Net(dem, flats, extent, crs, band, nodata)
    on = _stream_cells(net.on_grid(streams, "streams"))
    _, cells = _point_cells(net, points, streams, snap)
    label, dist = _core.hydro_along_network(net.down, net.step, on.astype(np.uint8), cells, up, dn)
    label = np.where(net.valid, label.astype(np.float64), np.nan)
    dist = np.where(np.isfinite(dist) & net.valid, dist, np.nan)
    shape = net.s.shape
    return NetworkReach(net.s.raster(label.reshape(shape), "point"), net.s.raster(dist.reshape(shape), "distance"))


# -- centrality -------------------------------------------------------------------------------------------------------

def centrality(links: pd.DataFrame) -> pd.DataFrame:
    """The place of every link in its river network, as hydrographr's ``get_centrality`` on the graph of the links.

    The links flowing into each, its degree, its Shreve magnitude (the
    sources above it), the length of river above it, and its betweenness
    (the pairs of links whose path goes through it) and closeness (one over
    the sum of the steps to every other link of its network), igraph's on
    the undirected graph.

    Parameters
    ----------
    links
        The links' table of :func:`stream_links` (``link``, as its index or a
        column, ``downstream`` and ``length``).

    Returns
    -------
    pandas.DataFrame
        ``link``, ``downstream``, ``upstream``, ``degree``, ``magnitude``,
        ``upstream_length``, ``betweenness``, ``closeness``.

    Examples
    --------
    >>> import pandas as pd, landschaft as ls
    >>> t = pd.DataFrame({"link": [1, 2, 3], "downstream": [3, 3, 0], "length": [10.0, 20.0, 5.0]})
    >>> ls.hydro.centrality(t)[["upstream", "magnitude", "upstream_length", "betweenness"]].to_numpy().tolist()
    [[0.0, 1.0, 10.0, 0.0], [0.0, 1.0, 20.0, 0.0], [2.0, 2.0, 35.0, 1.0]]
    """
    t = links.reset_index() if "link" not in links.columns and links.index.name == "link" else links
    for c in ("link", "downstream", "length"):
        if c not in t:
            raise ValueError(f"the links' table has no column {c!r} (stream_links')")
    ids = t["link"].to_numpy(np.float64)
    below = t["downstream"].to_numpy(np.float64)
    length = t["length"].to_numpy(np.float64)
    n = len(ids)
    at = {v: i for i, v in enumerate(ids)}
    parent = [at.get(b) for b in below]
    children: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p is not None:
            children[p].append(i)
    order = [i for i in range(n) if not children[i]]
    left = [len(c) for c in children]
    i = 0
    while i < len(order):
        p = parent[order[i]]
        if p is not None:
            left[p] -= 1
            if left[p] == 0:
                order.append(p)
        i += 1
    if len(order) != n:
        raise ValueError("the links go round in a circle")
    size, magnitude, above, steps = np.ones(n), np.zeros(n), length.copy(), np.zeros(n)
    for v in order:
        if not children[v]:
            magnitude[v] = 1.0
        p = parent[v]
        if p is not None:
            size[p] += size[v]
            magnitude[p] += magnitude[v]
            above[p] += above[v]
            steps[p] += steps[v] + size[v]
    total, ssum = np.zeros(n), np.zeros(n)
    for v in reversed(order):
        p = parent[v]
        if p is None:
            total[v], ssum[v] = size[v], steps[v]
        else:
            total[v] = total[p]
            ssum[v] = ssum[p] - size[v] + (total[v] - size[v])
    rows = []
    for v in range(n):
        branches = [size[c] for c in children[v]]
        if parent[v] is not None:
            branches.append(total[v] - size[v])
        s = sum(branches)
        rows.append([ids[v], below[v] if parent[v] is not None else 0.0, float(len(children[v])),
                     float(len(branches)), magnitude[v], above[v], (s * s - sum(b * b for b in branches)) / 2.0,
                     1.0 / ssum[v] if ssum[v] > 0 else np.nan])
    return pd.DataFrame(rows, columns=["link", "downstream", "upstream", "degree", "magnitude", "upstream_length",
                                       "betweenness", "closeness"])


# -- the dendritic connectivity index ---------------------------------------------------------------------------------

class _Segments:
    """A river network as a forest of segments (the stretches between barriers): each one's downstream segment (None
    at a mouth), its weight, and the passability of the barrier between it and the one below, going up and down."""

    def __init__(self, below, weight, up, down):
        self.below = list(below)
        self.weight = np.asarray(weight, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64).copy()
        self.down = np.asarray(down, dtype=np.float64).copy()

    def order(self) -> list[int]:
        n = len(self.below)
        left = [0] * n
        for b in self.below:
            if b is not None:
                left[b] += 1
        order = [i for i in range(n) if left[i] == 0]
        i = 0
        while i < len(order):
            b = self.below[order[i]]
            if b is not None:
                left[b] -= 1
                if left[b] == 0:
                    order.append(b)
            i += 1
        return order

    def reach(self, symmetric: bool, from_: bool = False) -> np.ndarray:
        """riverconn's reach index of every segment, unscaled: the weights of the segments of its network, each times
        the passability of the barriers on the way from it to them (``from_``: from them to it)."""
        if symmetric:
            both = self.up * self.down
            going_down, going_up = both, both
        else:
            going_down, going_up = self.down, self.up
        if from_:
            going_down, going_up = going_up, going_down
        order = self.order()
        above = self.weight.copy()
        for v in order:
            b = self.below[v]
            if b is not None:
                above[b] += going_up[v] * above[v]
        out = np.zeros(len(self.below))
        for v in reversed(order):
            b = self.below[v]
            out[v] = above[v] if b is None else above[v] + going_down[v] * (out[b] - going_up[v] * above[v])
        return out

    def mouths(self) -> list[int]:
        mouth = list(range(len(self.below)))
        for v in reversed(self.order()):
            b = self.below[v]
            if b is not None:
                mouth[v] = mouth[b]
        return mouth

    def dci(self, symmetric: bool) -> dict[int, tuple[float, float]]:
        """The DCI of each network (by its mouth), 0 to 100: the potamodromous (between every two segments) and the
        diadromous one (the reach index of the mouth)."""
        reach = self.reach(symmetric)
        sums: dict[int, list[float]] = {}
        for i, m in enumerate(self.mouths()):
            e = sums.setdefault(m, [0.0, 0.0])
            e[0] += self.weight[i] * reach[i]
            e[1] += self.weight[i]
        out = {}
        for m, (num, total) in sums.items():
            p, d = (num / (total * total), reach[m] / total) if total > 0 else (np.nan, np.nan)
            out[m] = (100.0 * p, 100.0 * d)
        return out


@dataclass
class DCI(Saveable):
    """The dendritic connectivity index of a river network cut by barriers (:func:`dci`).

    Attributes
    ----------
    table : pandas.DataFrame
        A row per network with barriers: ``network`` (its mouth segment),
        ``segments``, ``barriers``, ``weight``, ``dci_p`` (potamodromous, 0 to
        100) and ``dci_d`` (diadromous).
    barriers : geopandas.GeoDataFrame
        The barriers moved onto the streams, their attributes and
        ``segment``, ``network``, ``pass_u``, ``pass_d``, ``dci_p_removed``,
        ``d_index`` (the percent the DCI of its network grows when it is
        removed) and ``rank`` (1 the largest).
    segments : xarray.DataArray
        The segment of each stream cell (1..), 0 elsewhere.
    reach : xarray.DataArray
        On each stream cell, riverconn's reach index of its segment, 0 to 100:
        the share of its network a fish there can reach.
    """

    table: pd.DataFrame
    barriers: object
    segments: xr.DataArray
    reach: xr.DataArray


def _passabilities(points, n: int, default: float) -> tuple[np.ndarray, np.ndarray]:
    def column(name):
        if hasattr(points, "columns") and name in points.columns:
            return points[name].to_numpy(np.float64)
        return np.full(n, np.nan)

    both = column("passability")
    up = column("pass_u")
    down = column("pass_d")
    up = np.where(np.isnan(up), np.where(np.isnan(both), default, both), up)
    down = np.where(np.isnan(down), np.where(np.isnan(both), default, both), down)
    bad = np.nonzero(~((up >= 0) & (up <= 1) & (down >= 0) & (down <= 1)))[0]
    if bad.size:
        raise ValueError(f"barrier {bad[0] + 1}: a passability is between 0 and 1")
    return up, down


@threaded
def dci(dem, streams, barriers, *, weights=None, passability: float = 0.0, direction: str = "symmetric",
        snap: float | None = None, flats: bool = True, extent=None, crs=None, band=None, nodata=None) -> DCI:
    """The dendritic connectivity index of a river network cut by barriers (Cote et al. 2009), as R's riverconn.

    The network is split into segments at each dam, weir or culvert, each
    segment weighted by its length (or its habitat); the potamodromous DCI
    (the chance that a fish can swim between two points of the river, 0 to
    100) and the diadromous one (between the mouth and a point), from the
    passability of each barrier up and down. For every barrier, the DCI if it
    were removed and its gain (riverconn's ``d_index``, percent): which dams
    to remove first.

    Parameters
    ----------
    dem
        The elevations, best with depressions filled (:func:`fill`).
    streams
        The river network: non-zero on its cells (:func:`streams`, the links).
    barriers
        The barriers as points (a GeoDataFrame or (x, y) pairs), moved onto
        the streams; their passability from the columns ``pass_u`` and
        ``pass_d``, or ``passability`` (0 to 1).
    weights
        The habitat of each cell: a segment weighs the length of its cells
        times this (default 1).
    passability
        Of the barriers without those columns (0: impassable).
    direction
        ``"symmetric"`` (the product of the passabilities up and down at every
        barrier, Cote's) or ``"asymmetric"`` (up through a barrier going
        upstream, down going downstream).
    snap
        The largest move of a barrier onto the streams (as :func:`upstream`).
    flats, extent, crs, band, nodata
        As :func:`stream_links`.

    Returns
    -------
    DCI

    Examples
    --------
    >>> import numpy as np, landschaft as ls
    >>> dem = np.array([[5, 4, 5], [5, 3, 5], [5, 2, 5], [5, 1, 5]], float)
    >>> st = ls.hydro.streams(ls.hydro.flow_accumulation(dem), 2)

    a weir (half passable each way) on the middle of the three stream cells: segments of 2 and 1 cells, so
    (2² + 1² + 2 × 2 × 1 × 0.5 × 0.5) / 3²:

    >>> r = ls.hydro.dci(dem, st, [(1.5, 1.5)], passability=0.5)
    >>> r.table[["segments", "barriers", "weight", "dci_p"]].round(4).to_numpy().tolist()
    [[2.0, 1.0, 3.0, 66.6667]]
    """
    if direction not in ("symmetric", "asymmetric"):
        raise ValueError(f"direction must be 'symmetric' or 'asymmetric', not {direction!r}")
    symmetric = direction == "symmetric"
    net = _Net(dem, flats, extent, crs, band, nodata)
    n = net.s.shape[0] * net.s.shape[1]
    on = _stream_cells(net.on_grid(streams, "streams")) & net.valid
    w = None if weights is None else np.nan_to_num(net.on_grid(weights, "weights"), nan=0.0)
    xy = _points(barriers, net.s)
    pass_u, pass_d = _passabilities(barriers, len(xy), float(passability))
    distance = _snap_distance(snap, net.s)
    g = _Grid(net.s)
    at = g.snap(xy, on, distance)
    lost = np.nonzero(at < 0)[0]
    if lost.size:
        warnings.warn(f"barrier(s) {', '.join(str(i + 1) for i in lost)} farther than {distance} from the streams: "
                      "left out", stacklevel=2)
    down = net.down
    # the segments: one above each barrier's cell (barriers on one cell are one), one at each mouth of the network
    seed = np.zeros(n, dtype=np.int64)
    barrier_cell: list[int | None] = []
    for k in at[at >= 0]:
        if seed[k] == 0:
            barrier_cell.append(int(k))
            seed[k] = len(barrier_cell)
    nxt_on = np.where(down >= 0, on[np.maximum(down, 0)], False)
    for k in np.nonzero(on & (seed == 0) & ((down < 0) | ~nxt_on))[0]:
        barrier_cell.append(None)
        seed[k] = len(barrier_cell)
    label = net.label(seed)
    m = len(barrier_cell)
    seg = _Segments([None] * m, np.zeros(m), np.ones(m), np.ones(m))
    k = np.nonzero(on & (label > 0))[0]
    seg.weight = np.bincount(label[k] - 1, weights=net.step[k] * (1.0 if w is None else w[k]), minlength=m)
    gone = []
    for s, cell in enumerate(barrier_cell):
        if cell is None:
            continue
        b = down[cell]
        here = np.nonzero(at == cell)[0]
        if b >= 0 and on[b] and label[b] > 0:
            seg.below[s] = int(label[b]) - 1
            seg.up[s] = np.prod(pass_u[here])
            seg.down[s] = np.prod(pass_d[here])
        else:
            gone.extend(str(j + 1) for j in here)
    if gone:
        warnings.warn(f"barrier(s) {', '.join(gone)} at the mouth of the network: nothing below them", stacklevel=2)
    mouth = seg.mouths()
    base = seg.dci(symmetric)
    reach = seg.reach(symmetric)
    total: dict[int, float] = {}
    for s, mo in enumerate(mouth):
        total[mo] = total.get(mo, 0.0) + seg.weight[s]
    removed = np.full(m, np.nan)
    for s in range(m):
        if seg.below[s] is None:
            continue
        kept = seg.up[s], seg.down[s]
        seg.up[s] = seg.down[s] = 1.0
        removed[s] = seg.dci(symmetric)[mouth[s]][0]
        seg.up[s], seg.down[s] = kept
    networks = sorted({mouth[s] for s in range(m) if seg.below[s] is not None})
    table = pd.DataFrame(
        [[mo + 1, sum(1 for x in mouth if x == mo), sum(1 for s in range(m) if mouth[s] == mo and seg.below[s] is not None),
          total[mo], *base[mo]] for mo in networks],
        columns=["network", "segments", "barriers", "weight", "dci_p", "dci_d"], dtype=np.float64)
    cols = {c: np.full(len(xy), np.nan) for c in ("segment", "network", "pass_u", "pass_d", "dci_p_removed",
                                                  "d_index", "rank")}
    gains = []
    for i, cell in enumerate(at):
        if cell < 0:
            continue
        s = int(label[cell]) - 1
        mo = mouth[s]
        for c, x in zip(("segment", "network", "pass_u", "pass_d", "dci_p_removed"),
                        (s + 1, mo + 1, pass_u[i], pass_d[i], removed[s]), strict=True):
            cols[c][i] = x
        if not np.isnan(removed[s]):
            gain = (removed[s] - base[mo][0]) / base[mo][0] * 100.0
            cols["d_index"][i] = gain
            gains.append((gain, i))
    gains.sort(key=lambda t: (-t[0], t[1]))
    for r, (_, i) in enumerate(gains):
        cols["rank"][i] = r + 1
    attrs = _attributes(barriers)
    for c, v in cols.items():
        attrs[c] = v
    import geopandas as gpd

    cx, cy = g.centre(np.maximum(at, 0))
    moved = gpd.points_from_xy(np.where(at >= 0, cx, xy[:, 0]), np.where(at >= 0, cy, xy[:, 1]))
    segments = np.where(net.valid, np.where(on, label, 0).astype(np.float64), np.nan)
    tot = np.array([total[mo] for mo in mouth]) if m else np.zeros(0)
    seg_reach = 100.0 * reach / tot if m else np.zeros(0)
    reached = np.where(on & (label > 0), seg_reach[np.maximum(label - 1, 0)] if m else np.nan, np.nan)
    shape = net.s.shape
    return DCI(table, _geo(attrs, moved, net.s), net.s.raster(segments.reshape(shape), "segments"),
               net.s.raster(reached.reshape(shape), "reach"))
