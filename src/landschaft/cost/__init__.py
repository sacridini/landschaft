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
"""

from ._examples import example_data

__all__ = ["example_data"]
