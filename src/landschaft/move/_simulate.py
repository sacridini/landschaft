# SPDX-License-Identifier: GPL-3.0-or-later
"""Tracks of Brownian motion (move2's mt_sim_brownian_motion)."""

from __future__ import annotations

import numpy as np

from .. import _core


def simulate_brownian(t=None, sigma=1.0, tracks=2, start=(0.0, 0.0), *, seed: int | None = None):
    """Tracks of Brownian motion (move2's ``mt_sim_brownian_motion``).

    Each step moves by normal draws of standard deviation
    ``sqrt(sigma**2 * dt)`` along x and along y, from ``start``.

    Parameters
    ----------
    t
        The times (numbers; default 1 to 10): one sequence for all tracks, or
        a list of one sequence per track.
    sigma
        The diffusion: one number or one per step, for all tracks; or a list
        of one such list per track.
    tracks
        The number of tracks (ids 1, 2...) or their ids.
    start
        The first location (x, y), or a list of one per track.
    seed
        With a seed, R's generator after ``set.seed(seed)``: the same tracks as
        move2 (the tracks one after the other, x before y).

    Returns
    -------
    geopandas.GeoDataFrame
        Columns ``time``, ``track`` and point geometries without a CRS.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(tracks=2, seed=1)
    >>> tr.shape, list(tr.columns)
    ((20, 3), ['time', 'track', 'geometry'])
    """
    import geopandas as gpd

    ids = list(range(1, int(tracks) + 1)) if np.isscalar(tracks) else list(tracks)
    if len(set(map(str, ids))) != len(ids):
        raise ValueError("tracks: duplicated ids")
    k = len(ids)

    def per_track(v, default):
        """A list of one value per track (a list of lists / arrays), or the same value for all."""
        if v is None:
            v = default
        if isinstance(v, (list, tuple)) and len(v) and not np.isscalar(v[0]):
            if len(v) != k:
                raise ValueError("one value per track: as many as the tracks")
            return list(v)
        return [v] * k

    times = per_track(t, list(range(1, 11)))
    sigmas = per_track(sigma, 1.0)
    starts = per_track(start, (0.0, 0.0))
    tv = [np.asarray(v, dtype=np.float64) for v in times]
    sv = [np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in sigmas]
    for a, s in zip(tv, sv):
        if np.any(np.diff(a) < 0):
            raise ValueError("t: the times must not decrease")
        if np.any(s < 0):
            raise ValueError("sigma: values of 0 or more")
        if len(s) not in (1, max(len(a) - 1, 0)):
            raise ValueError("sigma: one value or one per step")
    st = np.asarray(starts, dtype=np.float64).reshape(k, 2)
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    xy = _core.move_brownian(tv, sv, st, int(seed))
    time = np.concatenate(tv)
    track = np.repeat(np.asarray(ids, dtype=object if not all(isinstance(i, (int, np.integer)) for i in ids)
                                 else np.int64), [len(a) for a in tv])
    x = np.concatenate([p[0] for p in xy])
    y = np.concatenate([p[1] for p in xy])
    out = gpd.GeoDataFrame({"time": time, "track": track}, geometry=gpd.points_from_xy(x, y))
    out.attrs["landschaft.move"] = {"time": "time", "track": "track"}
    return out
