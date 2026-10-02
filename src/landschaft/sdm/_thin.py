# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial thinning of occurrence records."""

from __future__ import annotations

import numpy as np

from .. import _core
from . import _points

__all__ = ["thin"]


def _seed(seed) -> int:
    """An R seed (int32): the one given, or a random one."""
    if seed is None:
        return int(np.random.default_rng().integers(1, 2**31 - 1))
    return int(seed)


def thin(occ, distance: float, *, reps: int = 10, seed: int | None = None, x: str | None = None,
         y: str | None = None, crs=None, return_all: bool = False):
    """Thin occurrence records so that no two kept records are closer than ``distance``.

    Records cluster where people sample (near roads, towns, universities),
    and a model fitted to them learns that bias along with the species'
    preferences. Thinning removes records until the remaining ones are at
    least ``distance`` apart while keeping as many as possible: the
    randomised heuristic of spThin (Aiello-Lammens et al. 2015, *Ecography*
    38: 541-545), which repeatedly drops a record with the most neighbours
    (a random one among ties) and keeps the best of ``reps`` repetitions.
    The same records as ``spThin::thin`` for the same seed; distances on a
    sphere of radius 6378.388 km, as spThin's ``fields::rdist.earth``.

    Parameters
    ----------
    occ
        Occurrence records: GeoDataFrame of points, or a table with
        coordinate columns.
    distance
        Minimum distance between kept records, in metres.
    reps
        Repetitions of the heuristic; the one keeping the most records is
        returned (the first among equals). More repetitions keep more records,
        rarely many more beyond 10-100.
    seed
        Seed of R's generator (``set.seed``); default random (recorded in
        ``attrs["thin"]``).
    x, y, crs
        Coordinate columns and CRS of a table (default: found by name;
        longitude / latitude in EPSG:4326).
    return_all
        Return the records kept by every repetition (a list, largest first)
        instead of the best one.

    Returns
    -------
    GeoDataFrame or DataFrame
        The records kept, with their original index; ``attrs["thin"]`` holds
        the seed and the number of records each repetition kept. Records
        without coordinates are always kept (they have no neighbours): drop
        them first with :func:`clean_occurrences`.

    Examples
    --------
    >>> import landschaft as ls
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> kept = ls.sdm.thin(occ, 50_000, reps=20, seed=1)
    >>> len(occ), len(kept)
    (116, 85)
    """
    if not distance >= 0:
        raise ValueError("distance must be >= 0 (metres)")
    if reps < 1:
        raise ValueError("reps must be >= 1")
    lon, lat = _points.lonlat(occ, x=x, y=y, crs=crs)
    s = _seed(seed)
    keep, kept = _core.sdm_thin_spthin(np.ascontiguousarray(lon), np.ascontiguousarray(lat), distance / 1000.0,
                                       int(reps), s)
    order = np.argsort(-kept, kind="stable")  # spThin: order(nrow, decreasing = TRUE), stable
    info = {"seed": s, "distance": float(distance), "reps": int(reps), "kept": kept[order].tolist()}
    if return_all:
        out = [_points.subset(occ, keep[r].astype(bool)) for r in order]
        for o in out:
            o.attrs["thin"] = info
        return out
    out = _points.subset(occ, keep[order[0]].astype(bool))
    out.attrs["thin"] = info
    return out
