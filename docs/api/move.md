# Animal movement

Animal tracks and their steps: the tracks of tagged animals as GeoDataFrames of
points, one row per location, ordered by track and time. Reads Movebank's
files; the distance, duration, speed, direction and turn of every step;
duplicates, one location per interval and Movebank's outliers; locations
interpolated in time; tracks as lines and segments; a local projection centred
on the tracks; Brownian tracks. The same numbers as R's move2 0.5.0
(Kranstauber et al. 2024): on the sphere of S2 and on the ellipsoid of lwgeom
(GeographicLib) for longitudes and latitudes, the plane of the CRS otherwise.
Then amt 0.3.1's (Signer et al. 2019) steps for step selection functions:
tracks resampled at a regular rate in bursts, the steps with their lengths and
turn angles, summaries of tracks (straightness, sinuosity, net squared
displacement...), day and night, distributions of step lengths and turn angles,
random steps and the covariates at their ends, with amt's numbers (and R's
random numbers given a seed). C++ with OpenMP; the same with any number of
threads. The `ls.move` module.

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

The data of a step selection function, as amt makes them:

<!-- no-run -->
```python
utm = tracks.to_crs(ls.move.aeqd_crs(tracks))           # random steps need a projected CRS
bursts = ls.move.resample(utm, "2h", "15min")           # a location every 2 hours, in bursts
steps = ls.move.steps(bursts)                           # sl_, ta_ per burst
ssf = ls.move.random_steps(steps, n_control=10, seed=1) # observed + 10 random steps per stratum
ssf = ls.move.extract_covariates(ssf, {"forest": forest, "elev": dem})
ssf["tod"] = ls.move.time_of_day(ssf, where="start")
ssf["log_sl"] = np.log(ssf["sl_"])
ssf["cos_ta"] = np.cos(ssf["ta_"])
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

## Steps of amt

::: landschaft.move.resample

::: landschaft.move.filter_min_n_burst

::: landschaft.move.steps

::: landschaft.move.time_of_day

## Summaries of tracks

::: landschaft.move.nsd

::: landschaft.move.cum_dist

::: landschaft.move.tot_dist

::: landschaft.move.straightness

::: landschaft.move.msd

::: landschaft.move.sinuosity

::: landschaft.move.tac

## Step selection

::: landschaft.move.make_distribution

::: landschaft.move.Distribution

::: landschaft.move.fit_distr

::: landschaft.move.random_numbers

::: landschaft.move.random_steps

::: landschaft.move.extract_covariates
