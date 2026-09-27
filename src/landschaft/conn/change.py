# SPDX-License-Identifier: GPL-3.0-or-later
"""Change of the equivalent connected area over time (Makurhini's MK_dECA)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .graph import HabitatGraph, _as_list, r_sum


def _comparison(deca: float, da: float) -> tuple[str, str]:
    """Makurhini's dECAfun and dECAfun2."""
    if deca < da < 0:
        return "dECA < dA < 0", "+ Connectivity loss"
    if da < deca < 0:
        return "dA < dECA < 0", "+ Habitat loss"
    if deca == da and da < 0:
        return "dECA = dA < 0", "Equal loss"
    if deca == da and da == 0:
        return "dECA = dA = 0", "Habitat or connectivity maintained"
    return "dECA or dA gain", "Habitat or connectivity gain"


def eca_change(graphs: Sequence[HabitatGraph], metric: str = "PC", *, distance, landscape_area: float,
               probability: float | None = None, pij_min: float | None = None, names=None) -> pd.DataFrame:
    """ECA of habitat graphs over time and its change, as Makurhini's ``MK_dECA``.

    Parameters
    ----------
    graphs
        One :class:`HabitatGraph` per time (or scenario), in order.
    metric
        ``"PC"`` or ``"IIC"``: the ECA is EC(PC) or EC(IIC).
    distance, probability, pij_min
        As in :meth:`HabitatGraph.overall` (several distances give one block
        of rows per distance).
    landscape_area
        LA, the maximum landscape attribute: the first time is compared with
        it, and the ECA cannot exceed it.
    names
        Names of the times (default ``1, 2, ...``).

    Returns
    -------
    pandas.DataFrame
        ``time``, ``LA``, ``area`` (habitat area: the areas of the nodes of
        graphs made with ``from_raster`` / ``from_vector``, else the sum of
        the attributes), ``distance``, ``ECA``, ``ECA_LA`` and ``ECA_area``
        (ECA in % of LA and of the habitat area), ``dA`` and ``dECA`` (change
        in % from the previous time, the first from LA), ``rECA`` (dECA / dA)
        and Makurhini's two readings of them, ``comparison`` and ``change``.

    Examples
    --------
    Area "a" grows between two times:

    >>> import geopandas as gpd
    >>> import shapely
    >>> import landschaft as ls
    >>> reserves = gpd.GeoDataFrame({"name": ["a", "b", "c", "d"]}, crs="EPSG:3035", geometry=[
    ...     shapely.box(0, 0, 2000, 2000), shapely.box(3000, 0, 4500, 1500),
    ...     shapely.box(0, 5000, 1000, 6000), shapely.box(9000, 9000, 11000, 11000)])
    >>> region = gpd.GeoDataFrame(geometry=[shapely.box(-1000, -1000, 10000, 10000)], crs="EPSG:3035")
    >>> later = reserves.copy()
    >>> later.loc[0, "geometry"] = shapely.box(0, 0, 2800, 2000)
    >>> graphs = [ls.HabitatGraph.from_vector(g, id="name", area_unit="ha", distance="edge",
    ...                                       landscape_area=12100) for g in (reserves, later)]
    >>> table = ls.conn.eca_change(graphs, "PC", distance=2000, probability=0.5,
    ...                            landscape_area=12100)
    >>> table[["time", "dA", "dECA", "change"]].round(1)
       time    dA  dECA                        change
    0     1 -90.7 -93.8           + Connectivity loss
    1     2  14.2  22.5  Habitat or connectivity gain
    """
    metric = str(metric).upper()
    if metric not in ("PC", "IIC"):
        raise ValueError("metric must be 'PC' or 'IIC'")
    graphs = list(graphs)
    if not graphs:
        raise ValueError("give at least one graph")
    names = list(range(1, len(graphs) + 1)) if names is None else list(names)
    if len(names) != len(graphs):
        raise ValueError("one name per graph")
    la = float(landscape_area)
    areas = np.array([g.habitat_area if g.habitat_area is not None else r_sum(g.attributes) for g in graphs])
    blocks = []
    for d in _as_list(distance):
        eca = []
        for g in graphs:
            # MK_dECA reads EC(metric) of MK_dPCIIC with LA (capped at LA).
            capped = HabitatGraph.__new__(HabitatGraph)
            capped.__dict__.update(g.__dict__)
            capped.landscape_area = la
            capped.restoration = None
            eca.append(float(capped.overall([metric], distance=d, probability=probability,
                                            pij_min=pij_min).iloc[0][f"EC({metric})"]))
        eca = np.array(eca)
        area_pct = areas * 100 / la
        da = np.empty(len(graphs))
        deca = np.empty(len(graphs))
        da[0] = (areas[0] - la) / la * 100
        deca[0] = (eca[0] - la) / la * 100
        da[1:] = (areas[1:] - areas[:-1]) / areas[:-1] * 100
        deca[1:] = (eca[1:] - eca[:-1]) / eca[:-1] * 100
        with np.errstate(divide="ignore", invalid="ignore"):
            reca = deca / da
        readings = [_comparison(x, y) for x, y in zip(deca, da)]
        blocks.append(pd.DataFrame({
            "time": names, "LA": la, "area": areas, "distance": d, "ECA": eca,
            "ECA_LA": eca * area_pct / areas, "ECA_area": eca * 100 / areas, "dA": da, "dECA": deca,
            "rECA": reca, "comparison": [c for c, _ in readings], "change": [c for _, c in readings]}))
    return pd.concat(blocks, ignore_index=True)
