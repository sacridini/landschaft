# SPDX-License-Identifier: GPL-3.0-or-later
"""Distances and routes on grids: least-cost, random-walk and randomised shortest paths.

A port of the R package gdistance (van Etten 2017): a raster becomes a graph
of neighbouring cells whose connections carry a conductance (a *transition*),
over which cost surfaces, least-cost distances and paths, commute distances,
passages of random walks and randomised shortest paths are computed, with
the same results as gdistance, in C++ with OpenMP.

Typical use::

    import landschaft as ls

    dem = ls.load_raster(ls.cost.example_data("maungawhau"), validate=None)
    slope = ls.cost.geo_correction(ls.cost.transition(dem, "diff", 8, symmetric=False))
    speed = slope.map(lambda s: 6 * np.exp(-3.5 * np.abs(s + 0.05)), pairs="all")
    conductance = ls.cost.geo_correction(speed)      # Tobler's hiking function
"""

from ._examples import example_data
from ._transition import Transition, geo_correction, normalize, sum_reciprocal, transition

__all__ = ["Transition", "example_data", "geo_correction", "normalize", "sum_reciprocal", "transition"]
