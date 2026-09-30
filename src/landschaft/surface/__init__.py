# SPDX-License-Identifier: GPL-3.0-or-later
"""Surface metrics: the texture of continuous rasters (ISO 25178).

Gradient surface metrics describe a continuous surface (NDVI, elevation,
canopy height...) without classifying it into patches: its amplitude, slopes,
peaks and valleys, the shape of its height distribution, its spatial
autocorrelation and orientation. A port of the R package geodiv (Smith et al.
2021): the same values, bit for bit, computed in C++ with OpenMP.

Typical use::

    import landschaft as ls

    ndvi = ls.surface.example_data("orforest")
    z = ls.surface.remove_plane(ndvi)             # residuals from the trend
    ls.surface.metrics(z)                          # every metric, one table
    ls.surface.sa(z), ls.surface.sku(z)
"""

from ._amplitude import sa, sku, sph, sq, ssk, smean, sv
from ._bearing import bearing_area, height_ba, sbi, sci, sdc, sk, spk, svi, svk
from ._examples import example_data
from ._features import find_peaks, find_valleys, s10z, sds, ssc
from ._hybrid import flat_area, sdq, sdq6, sdr, surface_area
from ._metrics import list_metrics, metrics
from ._plane import best_fit_plane, fit_plane, remove_plane
from ._spatial import aacf, scl, sfd, srw, std, stxr

__all__ = [
    "aacf",
    "bearing_area",
    "best_fit_plane",
    "example_data",
    "find_peaks",
    "find_valleys",
    "fit_plane",
    "flat_area",
    "height_ba",
    "list_metrics",
    "metrics",
    "remove_plane",
    "s10z",
    "sa",
    "sbi",
    "sci",
    "scl",
    "sdc",
    "sdq",
    "sdq6",
    "sdr",
    "sds",
    "sfd",
    "sk",
    "sku",
    "smean",
    "sph",
    "spk",
    "sq",
    "srw",
    "ssc",
    "ssk",
    "std",
    "stxr",
    "surface_area",
    "sv",
    "svi",
    "svk",
]
