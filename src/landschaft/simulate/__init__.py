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

from . import presets
from ._allocation import CLUE, Ordered
from ._calibrate import Calibration, calibrate
from ._change import Changes, IntensityAnalysis, changes, crosstab
from ._clumondo import CLUMondo, NeighbourhoodEffect, ServiceDemand, Supply
from ._continuous import ContinuousPatcher, StockAllocation, StockSimulation
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
from ._dynamic import distance_to, gravity_of, share_of
from ._ensemble import Ensemble, scenarios
from ._estimator import Estimator, RasterPotential
from ._examples import example_data
from ._futures import (
    DevelopmentDemand,
    DevelopmentPotential,
    Flooding,
    PatchGrowing,
    calibrate_patches,
    development_pressure,
    patch_library,
)
from ._glm import LogisticRegression
from ._graph import terraflux
from ._landsystems import Cap, LandSystemStack, Roles, Succession
from ._markov import Markov
from ._mce import MultiCriteria
from ._model import Model, Simulation
from ._molusce import MolusceCA, MolusceWoE
from ._patches import ExpanderPatcher
from ._plus import CARS
from ._regional import Regional
from ._roads import RoadBuilder, RoadDistance, destinations, friction_cost, pathways
from ._rules import Neighbourhood, Rules
from ._run import Projection, agreement, run
from ._similarity import (
    FuzzySimilarity,
    differences,
    fuzzy_similarity,
    similarity_of_differences,
)
from ._validate import Validation, futures_validation_grid, validate
from ._validation import (
    ThreeMapComparison,
    compare_maps,
    error_budget,
    futures_validation,
    three_map_comparison,
)
from ._woe import WeightsOfEvidence, weights_of_evidence

__all__ = [
    "CARS",
    "CLUE",
    "AreaDemand",
    "CLUMondo",
    "Calibration",
    "Cap",
    "Changes",
    "ContinuousPatcher",
    "DevelopmentDemand",
    "DevelopmentPotential",
    "Ensemble",
    "Estimator",
    "ExpanderPatcher",
    "Flooding",
    "FuzzySimilarity",
    "IntensityAnalysis",
    "LandSystemStack",
    "LogisticRegression",
    "Markov",
    "Model",
    "MolusceCA",
    "MolusceWoE",
    "MultiCriteria",
    "Neighbourhood",
    "NeighbourhoodEffect",
    "Ordered",
    "PatchGrowing",
    "Projection",
    "RasterPotential",
    "Regional",
    "RoadBuilder",
    "RoadDistance",
    "Roles",
    "Rules",
    "ServiceDemand",
    "Simulation",
    "StockAllocation",
    "StockSimulation",
    "Succession",
    "Supply",
    "ThreeMapComparison",
    "Validation",
    "WeightsOfEvidence",
    "agreement",
    "calibrate",
    "calibrate_patches",
    "changes",
    "compare_maps",
    "composite_matrix",
    "crosstab",
    "destinations",
    "development_pressure",
    "difference_components",
    "difference_resolutions",
    "differences",
    "distance_to",
    "error_budget",
    "example_data",
    "futures_validation",
    "futures_validation_grid",
    "friction_cost",
    "fuzzy_similarity",
    "gravity_of",
    "mad",
    "memberships",
    "overall_difference",
    "pathways",
    "patch_library",
    "presets",
    "run",
    "sample_to_population",
    "scenarios",
    "share_of",
    "similarity_of_differences",
    "terraflux",
    "three_map_comparison",
    "validate",
    "weights_of_evidence",
]
