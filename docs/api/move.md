# Animal movement

Animal tracks and their steps: the tracks of tagged animals as GeoDataFrames of
points, one row per location, ordered by track and time. Reads Movebank's
files; the distance, duration, speed, direction and turn of every step;
duplicates, one location per interval and Movebank's outliers; locations
interpolated in time; tracks as lines and segments; a local projection centred
on the tracks; Brownian tracks. The same numbers as R's move2 0.5.0
(Kranstauber et al. 2024): on the sphere of S2 and on the ellipsoid of lwgeom
(GeographicLib) for longitudes and latitudes, the plane of the CRS otherwise.
C++ with OpenMP; the same with any number of threads. The `ls.move` module.

<!-- no-run -->
```python
import landschaft as ls

tracks = ls.load_tracks("study.csv")                    # Movebank's export (.csv, .csv.gz, .zip)
tracks = ls.move.filter_movebank_visible(tracks)        # without the outliers Movebank hides
tracks = ls.move.filter_unique(tracks)                  # one record per time and track
tracks = tracks[~tracks.geometry.is_empty]

tracks["step"] = ls.move.distance(tracks)               # metres to the next location
tracks["speed"] = ls.move.speed(tracks, "km/h")
tracks["turn"] = ls.move.turn_angle(tracks, "deg")

hourly = ls.move.filter_per_interval(tracks, "hour")    # the first location of each hour
regular = ls.move.interpolate(tracks, "30 mins", omit=True)
local = tracks.to_crs(ls.move.aeqd_crs(tracks))         # metres around the tracks
ls.write_vector(ls.move.track_lines(tracks), "tracks.gpkg")
```

Every function finds the time and the track in the usual columns (`timestamp`,
`individual-local-identifier`, `time`, `track`...); others are named once with
`ls.move.set_columns` (or `ls.move.as_tracks`), or in each call with `time=` and
`track=`.

## Tracks

::: landschaft.move.load_tracks

::: landschaft.move.as_tracks

::: landschaft.move.set_columns

::: landschaft.move.simulate_brownian

## Steps

::: landschaft.move.distance

::: landschaft.move.time_lags

::: landschaft.move.speed

::: landschaft.move.azimuth

::: landschaft.move.turn_angle

## Filters

::: landschaft.move.unique

::: landschaft.move.filter_unique

::: landschaft.move.per_interval

::: landschaft.move.filter_per_interval

::: landschaft.move.movebank_visible

::: landschaft.move.filter_movebank_visible

## Interpolation

::: landschaft.move.interpolate

## Lines and projections

::: landschaft.move.track_lines

::: landschaft.move.segments

::: landschaft.move.track_data

::: landschaft.move.aeqd_crs
