# SPDX-License-Identifier: GPL-3.0-or-later
"""Distances, speeds, directions and time lags between consecutive locations (move2's mt_distance & co.)."""

from __future__ import annotations

import numpy as np

from .. import _core
from .._threads import resolve
from ._tracks import duration_factor, lags, length_factor, speed_factor, steps


def distance(tracks, units: str | None = None, *, time: str | None = None, track: str | None = None,
             n_threads: int | None = None) -> np.ndarray:
    """Distance from each location to the next location of its track.

    As move2's ``mt_distance``, the same numbers: on the sphere of S2 (radius
    6 371 010 m, as ``sf::st_distance`` with s2) for longitude / latitude, the
    straight line in the CRS's units otherwise.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points, one row per location, ordered by track and time
        (:func:`landschaft.load_tracks`, :func:`as_tracks`).
    units
        ``"m"`` (the default for longitude / latitude) or ``"km"``; ``None`` keeps
        metres or the CRS's units.
    time, track
        The columns of the time and of the track, if not recorded
        (:func:`set_columns`) or with the usual names (``timestamp``,
        ``individual-local-identifier``, ``track``...).
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    numpy.ndarray
        One value per location; NaN at the last location of each track.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)  # two tracks of six locations
    >>> ls.move.distance(tr).round(3)       # NaN at the last location of each track
    array([1.032, 0.521, 1.115, 1.696, 0.449,   nan, 1.512, 0.39 , 1.13 ,
           2.362, 1.272,   nan])
    """
    s = steps(tracks, time, track)
    d = _core.move_distances(s.x, s.y, s.code, s.lonlat, resolve(n_threads))
    f = length_factor(units, s.crs, s.lonlat)
    return d if f == 1.0 else d / f


def time_lags(tracks, units: str | None = None, *, time: str | None = None,
              track: str | None = None) -> np.ndarray:
    """Time from each location to the next of its track (move2's ``mt_time_lags``), in seconds.

    ``units``: ``"s"`` (default), ``"min"``, ``"h"`` or ``"d"``. Between datetimes the
    lags are exact (differences of nanoseconds). NaN at the last location of each track.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)  # two tracks of six locations
    >>> ls.move.time_lags(tr)
    array([ 1.,  1.,  1.,  1.,  1., nan,  1.,  1.,  1.,  1.,  1., nan])
    """
    s = steps(tracks, time, track)
    out = lags(tracks[s.time_col], s.code)
    f = duration_factor(units)
    return out if f == 1.0 else out / f


def speed(tracks, units: str | None = None, *, time: str | None = None, track: str | None = None,
          n_threads: int | None = None) -> np.ndarray:
    """Speed of each step: :func:`distance` over :func:`time_lags` (move2's ``mt_speed``).

    In metres (or the CRS's units) per second, or ``units``: ``"m/s"``, ``"km/h"``,
    ``"m/h"``, ``"km/d"``. Locations of a track must not share their time
    (:func:`filter_unique` drops them). NaN at the last location of each track.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)  # two tracks of six locations
    >>> ls.move.speed(tr).round(3)
    array([1.032, 0.521, 1.115, 1.696, 0.449,   nan, 1.512, 0.39 , 1.13 ,
           2.362, 1.272,   nan])
    """
    s = steps(tracks, time, track, need_time=True, strictly=True)
    d = _core.move_distances(s.x, s.y, s.code, s.lonlat, resolve(n_threads))
    v = d / lags(tracks[s.time_col], s.code)
    length, duration = speed_factor(units, s.crs, s.lonlat)
    return v if (length, duration) == (1.0, 1.0) else v / length * duration


def azimuth(tracks, units: str = "rad", *, time: str | None = None, track: str | None = None,
            n_threads: int | None = None) -> np.ndarray:
    """Direction of each step, clockwise from north (move2's ``mt_azimuth``).

    The geodesic azimuth on the CRS's ellipsoid for longitude / latitude (as
    ``lwgeom::st_geod_azimuth``, the same numbers), in (-pi, pi]; the direction on
    the grid (``atan2(dx, dy)``) in a projected CRS or without one (move2 refuses
    projected CRSs; on a conformal projection this is the direction from grid
    north). NaN at the last location of each track and where a location repeats
    the previous one. ``units``: ``"rad"`` or ``"deg"``.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)  # two tracks of six locations
    >>> ls.move.azimuth(tr, "deg").round(1)
    array([-142.6,   20.6,  -48.5,   70.2,  132.8,    nan,   91.7,   92.4,
            -33.4,  -69.7,   62.2,    nan])
    """
    s = steps(tracks, time, track)
    az = _core.move_azimuths(s.x, s.y, s.code, s.lonlat, s.a, s.inv_flattening, resolve(n_threads))
    return _angle(az, units)


def turn_angle(tracks, units: str = "rad", *, time: str | None = None, track: str | None = None,
               n_threads: int | None = None) -> np.ndarray:
    """Change of direction at each location, in [-pi, pi) (move2's ``mt_turnangle``).

    The difference between the azimuth of the step leaving a location and that
    of the step arriving at it, positive to the right; NaN at the first and last
    location of each track. ``units``: ``"rad"`` or ``"deg"``.

    Examples
    --------
    >>> import landschaft as ls
    >>> tr = ls.move.simulate_brownian(range(6), tracks=2, seed=1)  # two tracks of six locations
    >>> ls.move.turn_angle(tr, "deg").round(1)
    array([   nan,  163.3,  -69.2,  118.7,   62.7,    nan,    nan,    0.7,
           -125.7,  -36.3,  131.8,    nan])
    """
    s = steps(tracks, time, track)
    threads = resolve(n_threads)
    az = _core.move_azimuths(s.x, s.y, s.code, s.lonlat, s.a, s.inv_flattening, threads)
    return _angle(_core.move_turn_angles(az, threads), units)


def _angle(values: np.ndarray, units: str) -> np.ndarray:
    if units == "rad":
        return values
    if units == "deg":
        return values * (180.0 / np.pi)
    raise ValueError('units: "rad" or "deg"')
