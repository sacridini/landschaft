# SPDX-License-Identifier: GPL-3.0-or-later
"""Neutral landscape models: simulated landscapes that serve as null models.

A port of the R package NLMR (Sciaini et al. 2018): with the same seed each
model gives the same landscape as NLMR, bit for bit (R's random number
generator and the C++ library NLMR is built with are reproduced), computed in
C++ with OpenMP.

Typical use::

    import landschaft as ls

    r = ls.nlm.random(100, 100, seed=42)           # xarray.DataArray, 0-1
    classes = ls.nlm.classify(r, weighting=[0.6, 0.4])
    ls.landscape_metrics(classes, ["pland", "np"], level="class")
"""

from ._basic import distance_gradient, edge_gradient, percolation, planar_gradient, random, rescale
from ._classify import classify
from ._cluster import curds, random_cluster
from ._mosaic import mosaic_field
from ._spectral import fbm, gaussian_field
from ._stdmodels import mpd, neigh, random_rectangular_cluster
from ._tess import mosaic_gibbs, mosaic_tess

__all__ = [
    "classify",
    "curds",
    "distance_gradient",
    "edge_gradient",
    "fbm",
    "gaussian_field",
    "mosaic_field",
    "mosaic_gibbs",
    "mosaic_tess",
    "mpd",
    "neigh",
    "percolation",
    "planar_gradient",
    "random",
    "random_cluster",
    "random_rectangular_cluster",
    "rescale",
]
