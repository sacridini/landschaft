# SPDX-License-Identifier: GPL-3.0-or-later
"""Surface metrics: the texture of continuous rasters (ISO 25178).

Gradient surface metrics describe a continuous surface (NDVI, elevation,
canopy height...) without classifying it into patches: its amplitude, slopes,
peaks and valleys, the shape of its height distribution, its spatial
autocorrelation and orientation; and terrain indices of DEMs (TPI, TRI, VRM,
curvature, heat load...). Ports of the R packages geodiv (Smith et al. 2021)
and spatialEco (Evans 2021), computed in C++ with OpenMP: their values, bit for
bit where they are right, and the corrected ones where they err.

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
from ._terrain import curvature, dissection, hli, hsp, spherical_sd, srr, terrain, tpi, trasp, tri, vrm
from ._window import window_metrics

__all__ = [
    "aacf",
    "bearing_area",
    "best_fit_plane",
    "curvature",
    "dissection",
    "example_data",
    "find_peaks",
    "find_valleys",
    "fit_plane",
    "flat_area",
    "height_ba",
    "hli",
    "hsp",
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
    "spherical_sd",
    "spk",
    "sq",
    "srr",
    "srw",
    "ssc",
    "ssk",
    "std",
    "stxr",
    "surface_area",
    "sv",
    "svi",
    "svk",
    "terrain",
    "tpi",
    "trasp",
    "tri",
    "vrm",
    "window_metrics",
]
