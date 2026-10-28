# SPDX-License-Identifier: GPL-3.0-or-later
"""Statistics of maps: trends along series of maps, spatial autocorrelation and interpolation of points.

Computed in C++ with OpenMP, cell by cell, on maps larger than PySAL's and
pymannkendall's loops handle comfortably, with their numbers: the Mann-Kendall
test and Sen's slope of every cell (pymannkendall's ``original_test``, the same
bits), the global and local Moran's I and Getis and Ord's Gi* over the
contiguity of the cells (PySAL's esda, to rounding), and the interpolation of
points onto a map as R's gstat: sample variograms and their fit, inverse
distance weighting, ordinary kriging. The same as Terraflux's ``stats.*``
operators.

Typical use::

    import landschaft as ls

    ndvi = ls.load_raster("ndvi_2000_2024.tif")       # a band per year
    t = ls.stats.trend(ndvi)
    t.slope, t.p, t.trend                              # Sen's slope, the test's p, up / down / none

    ls.stats.moran(t.slope)                            # global Moran's I of the slopes
    hot = ls.stats.getis_ord(ls.load_raster("density.tif")).spots

    pts = ls.load_vector("soil_samples.gpkg")          # points with a value
    v = ls.stats.variogram(pts, "carbon", model="exponential")
    k = ls.stats.kriging(pts, dem, v, "carbon")       # onto the cells of a map
    k.prediction, k.variance
"""

from ._autocorr import LISA, GetisOrd, Moran, getis_ord, lisa, moran
from ._interp import Kriging, Variogram, VariogramModel, idw, kriging, variogram
from ._trend import Trend, mann_kendall, trend

__all__ = ["LISA", "GetisOrd", "Kriging", "Moran", "Trend", "Variogram", "VariogramModel", "getis_ord", "idw",
           "kriging", "lisa", "mann_kendall", "moran", "trend", "variogram"]
