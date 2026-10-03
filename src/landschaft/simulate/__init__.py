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

    lu = ls.load_raster(ls.simulate.example_data("pie_landuse"))   # 1985, 1991, 1999
    ch = ls.simulate.changes(lu)                                   # transitions, areas, intensity
    ch.intensity(gain=2, loss=1)                                   # Aldwaik & Pontius
    ls.simulate.difference_components(ch.matrix())                 # quantity, exchange, shift
    ls.simulate.Markov.from_maps(lu[0], lu[2], steps=14).multi     # rates per year
"""

from ._allocation import CLUE, Ordered
from ._change import Changes, IntensityAnalysis, changes, crosstab
from ._clumondo import CLUMondo, NeighbourhoodEffect, ServiceDemand, Supply
from ._demand import AreaDemand
from ._difference import (
    composite_matrix,
    difference_components,
    difference_resolutions,
    mad,
    memberships,
    overall_difference,
    sample_to_population,
)
from ._estimator import Estimator, RasterPotential
from ._examples import example_data
from ._futures import (
    DevelopmentDemand,
    DevelopmentPotential,
    PatchGrowing,
    development_pressure,
)
from ._glm import LogisticRegression
from ._landsystems import Cap, LandSystemStack, Roles, Succession
from ._markov import Markov
from ._mce import MultiCriteria
from ._model import Model, Simulation
from ._rules import Neighbourhood, Rules
from ._woe import WeightsOfEvidence, weights_of_evidence

__all__ = [
    "CLUE",
    "AreaDemand",
    "CLUMondo",
    "Cap",
    "Changes",
    "DevelopmentDemand",
    "DevelopmentPotential",
    "Estimator",
    "IntensityAnalysis",
    "LandSystemStack",
    "LogisticRegression",
    "Markov",
    "Model",
    "MultiCriteria",
    "Neighbourhood",
    "NeighbourhoodEffect",
    "Ordered",
    "PatchGrowing",
    "RasterPotential",
    "Roles",
    "Rules",
    "ServiceDemand",
    "Simulation",
    "Succession",
    "Supply",
    "WeightsOfEvidence",
    "changes",
    "composite_matrix",
    "crosstab",
    "development_pressure",
    "difference_components",
    "difference_resolutions",
    "example_data",
    "mad",
    "memberships",
    "overall_difference",
    "sample_to_population",
    "weights_of_evidence",
]
