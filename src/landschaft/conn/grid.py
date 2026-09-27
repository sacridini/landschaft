# SPDX-License-Identifier: GPL-3.0-or-later
"""Connectivity on a grid of cells (Makurhini's make_grid, MK_Connect_grid, MK_dECA_grid)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import HabitatGraph, r_sum
from .nodes import AREA_UNITS, gdal_areas
from .protconn import INDICATORS, _explode, protconn, QUAD_SEGS


def _r_seq(start: float, stop: float, by: float) -> np.ndarray:
    """R's seq(from, to, by) for doubles: from + (0:n) * by, capped at `to`."""
    n = int((stop - start) / by + 1e-10)
    return np.minimum(start + np.arange(n + 1) * by, stop)


def _hexagons(xmin, ymin, xmax, ymax, cellsize):
    """sf's st_make_grid(square = FALSE): pointy-topped hexagons of width `cellsize`."""
    import shapely

    dx = cellsize / np.sqrt(3)
    dy = np.sqrt(3) * dx / 2
    # make_hex_grid(flat_topped = FALSE) works on the swapped axes.
    xlim, ylim = (ymin, ymax), (xmin, xmax)
    x0 = _r_seq(xlim[0] - dx, xlim[1] + 2 * dx, dx) + 0.0
    y0 = _r_seq(ylim[0] - 2 * dy, ylim[1] + 2 * dy, dy) + 0.0
    nx, ny = len(x0), len(y0)
    y = np.repeat(y0, nx)
    x = np.resize(np.concatenate([x0, x0 + dx / 2]), len(y))
    odd = np.arange(1, 2 * nx, 2)
    even = np.arange(2, 2 * nx + 1, 2)
    xi = np.resize(np.concatenate([odd, even]), len(y))
    yi = np.repeat(np.arange(1, ny + 1), nx)
    keep = np.isin(xi, np.arange(3, xi.max() - 2 + 1, 3)) & (yi > 1) & (yi < yi.max())
    centers = np.column_stack([xi[keep], yi[keep]])
    pattern = np.array([[-2, 0], [-1, -1], [1, -1], [2, 0], [1, 1], [-1, 1], [-2, 0]])
    polys = []
    for c in centers:
        m = c + pattern
        idx = ((m[:, 0] - 1) // 2) + (m[:, 1] - 1) * nx
        polys.append(shapely.Polygon(np.column_stack([y[idx], x[idx]])))  # back to (x, y)
    return polys


def _squares(xmin, ymin, xmax, ymax, width):
    """raster(extent(x) + width, resolution = width) as polygons, row by row from the top."""
    import shapely

    xmin, xmax = xmin - width / 2, xmax + width / 2
    ymin, ymax = ymin - width / 2, ymax + width / 2
    nc = max(1, int(round((xmax - xmin) / width)))
    nr = max(1, int(round((ymax - ymin) / width)))
    xmax = xmin + nc * width
    ymin = ymax - nr * width
    xres = (xmax - xmin) / nc
    yres = (ymax - ymin) / nr
    polys = []
    for r in range(nr):
        for c in range(nc):
            cx = (c + 0.5) * xres + xmin
            cy = ymax - (r + 0.5) * yres
            x0, x1, y0, y1 = cx - xres / 2, cx + xres / 2, cy - yres / 2, cy + yres / 2
            # rasterToPolygons' ring: from the top left corner, clockwise.
            polys.append(shapely.Polygon([(x0, y1), (x1, y1), (x1, y0), (x0, y0), (x0, y1)]))
    return polys


def make_grid(region, cell_area: float, *, hexagonal: bool = True, clip: bool = False,
              grid_boundary: bool = False):
    """Grid of cells of `cell_area` over a region, as Makurhini's ``make_grid``.

    Hexagons come from sf's ``st_make_grid(square = FALSE)`` (the same
    coordinates), squares from a raster over the region's extent grown by
    one cell; the cells that touch the region are kept. ``clip`` cuts them to
    the region, ``grid_boundary`` keeps only whole cells inside it.

    Parameters
    ----------
    region
        Polygons of the region (GeoDataFrame, projected).
    cell_area
        Area of each cell, in square map units.
    hexagonal
        Hexagons (default) or squares.
    clip
        Cut the cells to the region.
    grid_boundary
        Keep only the cells entirely inside the region.

    Returns
    -------
    geopandas.GeoDataFrame
        ``OBJECTID`` and the cell polygons.

    Examples
    --------
    Hexagons of 25 km² over a region of 11 x 11 km:

    >>> import geopandas as gpd
    >>> import shapely
    >>> import landschaft as ls
    >>> region = gpd.GeoDataFrame(geometry=[shapely.box(-1000, -1000, 10000, 10000)], crs="EPSG:3035")
    >>> grid = ls.conn.make_grid(region, cell_area=25e6)
    >>> len(grid)
    12
    """
    import geopandas as gpd
    import shapely

    if clip and grid_boundary:
        raise ValueError("choose clip or grid_boundary, not both")
    geoms = np.asarray(region.geometry.values if hasattr(region, "geometry") else region)
    union = shapely.union_all(geoms)
    xmin, ymin, xmax, ymax = shapely.total_bounds(geoms)
    if hexagonal:
        cells = _hexagons(xmin, ymin, xmax, ymax, np.sqrt(2 * cell_area / np.sqrt(3)))
    else:
        cells = _squares(xmin, ymin, xmax, ymax, np.sqrt(cell_area))
    cells = np.array(cells, dtype=object)
    cells = cells[shapely.intersects(cells, union)]
    if clip:
        cells = shapely.intersection(cells, union)
        cells = cells[~shapely.is_empty(cells)]
    elif grid_boundary:
        full = round(float(gdal_areas(cells[:1])[0]))
        inside = shapely.intersection(cells, shapely.buffer(union, 0))
        cells = inside[np.round(gdal_areas(inside)) >= full]
    crs = getattr(region, "crs", None)
    return gpd.GeoDataFrame({"OBJECTID": np.arange(1, len(cells) + 1)}, geometry=list(cells), crs=crs)


def _pc_cell(nodes_in_cell, la: float, distance: float, probability: float, factor: float,
             single: bool = False) -> dict:
    """get_pc_grid: PC and ECA of the dissolved patches of a cell."""
    import shapely

    if not nodes_in_cell:
        return {"Protected_surface": 0.0, "LA": la, "ECA": np.nan, "ECA_Normalized": np.nan, "PC": np.nan}
    areas = gdal_areas(np.array(nodes_in_cell, dtype=object)) * factor
    if len(areas) == 1 or single:
        a = min(r_sum(areas), la)  # Protconn_nodes: one area, capped at the cell's
        full = a >= la
        return {"Protected_surface": a, "LA": la, "ECA": la if full else a,
                "ECA_Normalized": 100.0 if full else (a * 100) / la, "PC": 1.0 if full else a / la**2}
    geoms = np.array(nodes_in_cell, dtype=object)
    n = len(geoms)
    i, j = np.triu_indices(n, 1)
    g = HabitatGraph(areas, pd.DataFrame({"a": i + 1, "b": j + 1, "d": shapely.distance(geoms[i], geoms[j])}))
    num = float(g.overall("PC", distance=distance, probability=probability, pij_min=0.0).iloc[0]["PCnum"])
    eca = float(np.sqrt(num))
    return {"Protected_surface": r_sum(areas), "LA": la, "ECA": eca, "ECA_Normalized": (eca * 100) / la,
            "PC": num / la**2}


def connectivity_grid(protected_areas, region, *, distance: float, grid=None, cell_area: float | None = None,
                      hexagonal: bool = True, protconn_index: bool = True, probability: float = 0.5,
                      transboundary: float | None = None, area_unit: str = "m2", pij_min: float | None = None):
    """ProtConn (or PC and ECA) in every cell of a grid, as Makurhini's ``MK_Connect_grid``.

    Parameters
    ----------
    protected_areas
        Polygons of the protected areas (or habitat patches).
    region
        The study region: the grid covers it, and only the areas within
        ``transboundary`` of it are used.
    distance, probability
        Dispersal distance and its probability.
    grid, cell_area, hexagonal
        A grid of polygons, or the area of the cells of a new one
        (:func:`make_grid`).
    protconn_index
        ``True``: ProtConn and its fractions of each cell (the cell as the
        region, the areas in the other cells transboundary with
        ``transboundary``). ``False``: PC, ECA and normalised ECA of the
        patches dissolved in each cell.
    pij_min
        ProtConn: links below this probability are cut. Makurhini's
        ``MK_Connect_grid`` turns its default (NULL) into infinity, which cuts
        every link (each node alone): ``pij_min=np.inf`` gives its values.
    transboundary
        ProtConn: distance of the transboundary zone of each cell.
    area_unit
        Unit of the areas.

    Returns
    -------
    geopandas.GeoDataFrame
        The grid with ``OBJECTID`` and, per cell, the columns of :func:`protconn`
        (or ``ECA``, ``PC`` and ``Normalized_ECA`` with ``protconn_index=False``).

    Examples
    --------
    >>> import geopandas as gpd
    >>> import shapely
    >>> import landschaft as ls
    >>> reserves = gpd.GeoDataFrame({"name": ["a", "b", "c", "d"]}, crs="EPSG:3035", geometry=[
    ...     shapely.box(0, 0, 2000, 2000), shapely.box(3000, 0, 4500, 1500),
    ...     shapely.box(0, 5000, 1000, 6000), shapely.box(9000, 9000, 11000, 11000)])
    >>> region = gpd.GeoDataFrame(geometry=[shapely.box(-1000, -1000, 10000, 10000)], crs="EPSG:3035")
    >>> cells = ls.conn.connectivity_grid(reserves, region, distance=2000, cell_area=25e6)
    >>> cells.columns.tolist()[:6]
    ['OBJECTID', 'ECA', 'PC', 'LA', 'Protected_surface', 'Prot']
    >>> len(cells)
    12
    """
    import geopandas as gpd
    import shapely

    if grid is None:
        if cell_area is None:
            raise ValueError("give a grid or a cell_area")
        grid = make_grid(region, cell_area, hexagonal=hexagonal)
    factor = AREA_UNITS[area_unit]
    union = shapely.union_all(np.asarray(region.geometry.values if hasattr(region, "geometry") else region))
    pas = np.asarray(protected_areas.geometry.values)
    zone = union if not transboundary else shapely.buffer(union, transboundary, quad_segs=QUAD_SEGS)
    pre = protected_areas[shapely.intersects(pas, zone)]
    rows = []
    for cell in grid.geometry.values:
        la = float(gdal_areas(np.array([cell], dtype=object))[0]) * factor
        if protconn_index:
            table = protconn(pre, gpd.GeoDataFrame(geometry=[cell], crs=grid.crs), distance=[distance],
                             probability=probability, transboundary=transboundary, area_unit=area_unit,
                             preselect=False, pij_min=pij_min)
            row = table.iloc[0].drop(["distance", "ProtUnconn_Design", "ProtConn_Bound"]).to_dict()
            if row["Protected_surface"] == 0:
                row.update(ECA=np.nan, PC=np.nan, ProtConn=np.nan, ProtUnconn=np.nan)
        else:
            y1 = pre.geometry.values[shapely.intersects(pre.geometry.values, cell)]
            clipped = shapely.intersection(y1, cell)
            dissolved = _explode(shapely.buffer(shapely.union_all(clipped), 0)) if len(clipped) else []
            # One protected area in the cell is one node, whatever its parts (Makurhini).
            row = _pc_cell(dissolved, la, distance, probability, factor, single=len(y1) == 1)
        rows.append(row)
    out = pd.DataFrame(rows)
    return gpd.GeoDataFrame(pd.concat([grid.drop(columns="geometry").reset_index(drop=True), out], axis=1),
                            geometry=grid.geometry.values, crs=grid.crs)




def eca_change_grid(times, region, *, distance: float, metric: str = "IIC", grid=None,
                    cell_area: float | None = None, hexagonal: bool = True, probability: float | None = None,
                    pij_min: float | None = None, names=None, area_unit: str = "m2"):
    """ECA change between times in every cell of a grid, as Makurhini's ``MK_dECA_grid``.

    Parameters
    ----------
    times
        Habitat polygons (GeoDataFrames), one per time.
    region, grid, cell_area, hexagonal
        As in :func:`connectivity_grid`.
    distance, metric, probability, pij_min
        As in :func:`eca_change` (centroid distances between the pieces of
        the patches in each cell).
    names
        Names of the times (default ``T1, T2, ...``); pairs are named
        ``"T1.T2"``.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per cell and pair of consecutive times: ``time``, ``ECA_i``,
        ``ECA_j``, ``ECAnorm_i``, ``ECAnorm_j`` (in % of the cell), ``dA``,
        ``dECA``, ``rECA``, ``comparison``, ``change``. Makurhini's rules for
        cells without habitat at one time are kept: habitat that disappears
        gives dA = dECA = -100 (and ECA_i the area of its pieces when there are
        two or fewer), habitat that appears gives ECA_j = the cell's area and
        dA = dECA = 100, and none at both times gives NA.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import shapely
    >>> import landschaft as ls
    >>> reserves = gpd.GeoDataFrame({"name": ["a", "b", "c", "d"]}, crs="EPSG:3035", geometry=[
    ...     shapely.box(0, 0, 2000, 2000), shapely.box(3000, 0, 4500, 1500),
    ...     shapely.box(0, 5000, 1000, 6000), shapely.box(9000, 9000, 11000, 11000)])
    >>> region = gpd.GeoDataFrame(geometry=[shapely.box(-1000, -1000, 10000, 10000)], crs="EPSG:3035")
    >>> later = reserves.copy()
    >>> later.loc[0, "geometry"] = shapely.box(0, 0, 2800, 2000)       # area "a" grows
    >>> cells = ls.conn.eca_change_grid([reserves, later], region, distance=2000, cell_area=25e6)
    >>> cells.columns.tolist()[:4]
    ['OBJECTID', 'time', 'ECA_i', 'ECA_j']
    """
    import geopandas as gpd
    import shapely

    from .change import _comparison, eca_change

    times = list(times)
    if len(times) < 2:
        raise ValueError("give two or more times")
    names = [f"T{k + 1}" for k in range(len(times))] if names is None else list(names)
    if grid is None:
        if cell_area is None:
            raise ValueError("give a grid or a cell_area")
        grid = make_grid(region, cell_area, hexagonal=hexagonal)
    factor = AREA_UNITS[area_unit]
    metric = str(metric).upper()
    probability_used = 0.5 if (metric == "PC" and probability is None) else probability
    rows = []
    for k, cell in enumerate(grid.geometry.values):
        la = float(gdal_areas(np.array([cell], dtype=object))[0]) * factor
        pieces = []
        for t in times:
            g = t.geometry.values
            cut = shapely.intersection(g[shapely.intersects(g, cell)], cell)
            parts = [p for c in cut for p in _explode(c)]
            pieces.append(gpd.GeoDataFrame(geometry=parts, crs=t.crs) if parts else None)
        for i in range(len(times) - 1):
            a, b = pieces[i], pieces[i + 1]
            row = {"cell": k, "time": f"{names[i]}.{names[i + 1]}"}
            if a is not None and b is not None:
                graphs = [HabitatGraph.from_vector(x, area_unit=area_unit) for x in (a, b)]
                table = eca_change(graphs, metric, distance=distance, landscape_area=la,
                                   probability=probability_used, pij_min=pij_min)
                row.update(ECA_i=table["ECA"].iloc[0], ECA_j=table["ECA"].iloc[1], ECAnorm_i=table["ECA_LA"].iloc[0],
                           ECAnorm_j=table["ECA_LA"].iloc[1], dA=table["dA"].iloc[1], dECA=table["dECA"].iloc[1],
                           rECA=table["rECA"].iloc[1], comparison=table["comparison"].iloc[1],
                           change=table["change"].iloc[1])
            elif a is not None:
                if len(a) > 2:
                    g = HabitatGraph.from_vector(a, area_unit=area_unit)
                    g.landscape_area = la
                    eca_i = float(g.overall([metric], distance=distance, probability=probability_used,
                                            pij_min=pij_min).iloc[0][f"EC({metric})"])
                else:
                    eca_i = r_sum(gdal_areas(a.geometry.values) * factor)
                comparison, change = _comparison(-100.0, -100.0)
                row.update(ECA_i=eca_i, ECA_j=0.0, ECAnorm_i=(eca_i * 100) / la, ECAnorm_j=0.0, dA=-100.0,
                           dECA=-100.0, rECA=1.0, comparison=comparison, change=change)
            elif b is not None:
                comparison, change = _comparison(100.0, 100.0)
                row.update(ECA_i=0.0, ECA_j=la, ECAnorm_i=0.0, ECAnorm_j=100.0, dA=100.0, dECA=100.0, rECA=1.0,
                           comparison=comparison, change=change)
            else:
                row.update(ECA_i=np.nan, ECA_j=np.nan, ECAnorm_i=np.nan, ECAnorm_j=np.nan, dA=np.nan, dECA=np.nan,
                           rECA=np.nan, comparison="NA", change="NA")
            rows.append(row)
    # One block of cells per pair of times, as MK_dECA_grid's list.
    table = pd.DataFrame(rows).sort_values(["time", "cell"], kind="stable",
                                            key=lambda c: c.map({f"{names[i]}.{names[i + 1]}": i
                                                                 for i in range(len(times) - 1)})
                                            if c.name == "time" else c).reset_index(drop=True)
    base = grid.drop(columns="geometry").reset_index(drop=True)
    out = pd.concat([base.iloc[table["cell"]].reset_index(drop=True), table.drop(columns="cell")], axis=1)
    return gpd.GeoDataFrame(out, geometry=grid.geometry.values[table["cell"].to_numpy()], crs=grid.crs)


__all__ = ["connectivity_grid", "eca_change_grid", "make_grid"]
