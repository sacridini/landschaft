# SPDX-License-Identifier: GPL-3.0-or-later
"""Movement ecology: animal tracks, their steps and their relation to the landscape.

Tracks are GeoDataFrames of points, one row per location, ordered by track and
time (:func:`landschaft.load_tracks` reads Movebank's files, :func:`as_tracks`
any table of locations). The first part is R's move2 (Kranstauber et al.
2024), the same numbers: the distance, speed, direction and turn of each step
(``sf`` / ``s2`` and ``lwgeom`` on the sphere and the ellipsoid), duplicates and
one location per interval, interpolation in time, tracks as lines, a local
projection, Brownian tracks. Computed in C++ with OpenMP.

Typical use::

    import landschaft as ls

    tracks = ls.load_tracks("study.csv")                    # Movebank's export
    tracks = ls.move.filter_unique(tracks)                  # one record per time and track
    tracks["speed"] = ls.move.speed(tracks)                 # m/s, NaN at the last location
    tracks["turn"] = ls.move.turn_angle(tracks, "deg")
    hourly = ls.move.filter_per_interval(tracks, "hour")
    lines = ls.move.track_lines(tracks)                     # one line per animal
    ls.write_vector(lines, "tracks.gpkg")
"""

from ._filters import (filter_movebank_visible, filter_per_interval, filter_unique, movebank_visible, per_interval,
                       unique)
from ._geometry import aeqd_crs, interpolate, segments, track_data, track_lines
from ._measures import azimuth, distance, speed, time_lags, turn_angle
from ._read import as_tracks, load_tracks
from ._simulate import simulate_brownian
from ._ssf import Distribution, extract_covariates, fit_distr, make_distribution, random_numbers, random_steps
from ._steps import (cum_dist, filter_min_n_burst, msd, nsd, resample, sinuosity, steps, straightness, tac,
                     time_of_day, tot_dist)
from ._tracks import set_columns

__all__ = ["aeqd_crs", "as_tracks", "azimuth", "cum_dist", "distance", "Distribution", "extract_covariates",
           "filter_min_n_burst", "filter_movebank_visible", "filter_per_interval", "filter_unique", "fit_distr",
           "interpolate", "load_tracks", "make_distribution", "movebank_visible", "msd", "nsd", "per_interval",
           "random_numbers", "random_steps", "resample", "segments", "set_columns", "simulate_brownian", "sinuosity",
           "speed", "steps", "straightness", "tac", "time_lags", "time_of_day", "tot_dist", "track_data",
           "track_lines", "turn_angle", "unique"]
