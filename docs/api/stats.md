# Statistics of maps

Trends along a series of maps and the spatial autocorrelation of a map: the
Mann-Kendall test and Sen's slope of every cell (pymannkendall's
`original_test`, the same bits), the global and the local Moran's I and Getis
and Ord's Gi* over the rook or queen contiguity of the cells (PySAL's esda, to
rounding), with pseudo p values of random permutations; and the interpolation of
points onto a map as R's gstat: sample variograms and their fit, inverse
distance weighting and ordinary kriging. C++ with OpenMP; the same with any
number of threads. The `ls.stats` module.

<!-- no-run -->
```python
import landschaft as ls

ndvi = ls.load_raster("ndvi_2000_2024.tif")       # a band per year
t = ls.stats.trend(ndvi)
t.slope, t.p, t.trend                              # Sen's slope, the test's p, 1 up / -1 down / 0 none

ls.stats.moran(t.slope)                            # global Moran's I and its p values
ls.stats.lisa(t.slope).clusters                    # high-high, low-low... cells
ls.stats.getis_ord(ls.load_raster("density.tif")).spots   # hot and cold spots

pts = ls.load_vector("soil_samples.gpkg")          # points with a value
v = ls.stats.variogram(pts, "carbon", model="exponential")
k = ls.stats.kriging(pts, ndvi[0], v, "carbon")    # onto the cells of a map
k.prediction, k.variance
ls.stats.idw(pts, ndvi[0], "carbon", nmax=12)
```

## Trends

::: landschaft.stats.trend

::: landschaft.stats.Trend

::: landschaft.stats.mann_kendall

## Spatial autocorrelation

::: landschaft.stats.moran

::: landschaft.stats.Moran

::: landschaft.stats.lisa

::: landschaft.stats.LISA

::: landschaft.stats.getis_ord

::: landschaft.stats.GetisOrd

## Interpolation

::: landschaft.stats.variogram

::: landschaft.stats.Variogram

::: landschaft.stats.VariogramModel

::: landschaft.stats.idw

::: landschaft.stats.kriging

::: landschaft.stats.Kriging
