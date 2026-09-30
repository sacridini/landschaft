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
from ._examples import example_data
from ._metrics import list_metrics, metrics
from ._plane import best_fit_plane, fit_plane, remove_plane

__all__ = [
    "best_fit_plane",
    "example_data",
    "fit_plane",
    "list_metrics",
    "metrics",
    "remove_plane",
    "sa",
    "sku",
    "smean",
    "sph",
    "sq",
    "ssk",
    "sv",
]
