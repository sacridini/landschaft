# SPDX-License-Identifier: GPL-3.0-or-later
"""Statistics of maps: trends along series of maps and spatial autocorrelation.

Computed in C++ with OpenMP, cell by cell, on maps larger than PySAL's and
pymannkendall's loops handle comfortably, with their numbers: the Mann-Kendall
test and Sen's slope of every cell (pymannkendall's ``original_test``, the same
bits), the global and local Moran's I and Getis and Ord's Gi* over the
contiguity of the cells (PySAL's esda, to rounding). The same as Terraflux's
``stats.*`` operators.

Typical use::

    import landschaft as ls

    ndvi = ls.load_raster("ndvi_2000_2024.tif")       # a band per year
    t = ls.stats.trend(ndvi)
    t.slope, t.p, t.trend                              # Sen's slope, the test's p, up / down / none

    ls.stats.moran(t.slope)                            # global Moran's I of the slopes
    hot = ls.stats.getis_ord(ls.load_raster("density.tif")).spots
"""

from ._autocorr import LISA, GetisOrd, Moran, getis_ord, lisa, moran
from ._trend import Trend, mann_kendall, trend

__all__ = ["LISA", "GetisOrd", "Moran", "Trend", "getis_ord", "lisa", "mann_kendall", "moran", "trend"]
