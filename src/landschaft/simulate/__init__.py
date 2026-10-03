# SPDX-License-Identifier: GPL-3.0-or-later
"""Land use and land cover change: analysis, transition potential, allocation, validation.

``ls.simulate`` brings together what the land change models of the field
do, as interchangeable pieces of one model: how much changes (Markov chains,
area demands as in CLUE-S, demands for goods and services as in CLUMondo),
where change is likely (weights of evidence, logistic regression, any
scikit-learn estimator), the rules of change (allowed conversions,
elasticities, succession, adaptive inertia) and the shape of the new patches
(CLUE-S allocation, Dinamica EGO's Expander and Patcher, PLUS's multi-type
random patch seeds, FUTURES' patch growing). The reference implementations
(lulcc, CLUMondo, r.futures, MOLUSCE, OpenLand, diffeR) are reproduced bit
for bit where their code is open; Dinamica EGO and PLUS are implemented from
their publications.

Typical use::

    import landschaft as ls

    lu = ls.load_raster(ls.simulate.example_data("sibuyan_landuse"))
"""

from ._examples import example_data

__all__ = ["example_data"]
