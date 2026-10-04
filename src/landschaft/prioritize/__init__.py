# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial conservation prioritization: ranking the cells of a landscape, or
selecting the best set of planning units by optimization.

A port of Zonation 5 (Lehtinen, Moilanen et al., University of Helsinki; GPL-3):
the cells are ranked from the least to the most important for a set of
features (species, habitats, ecosystem services), weighted, with costs,
hierarchical masks and the core-area (CAZ) or additive-benefit (ABF) marginal
loss rules; with performance curves of how much of every feature the best
fractions of the landscape keep. Same results as the z5 program, bit for bit,
computed in C++ with OpenMP.

Typical use::

    import landschaft as ls

    res = ls.prioritize.zonation(["sp1.tif", "sp2.tif", "sp3.tif"], weights=[2, 1, 1])
    res.rank                      # 0 = lowest priority, ~1 = highest
    res.feature_curves            # fraction of every feature kept
    res.write("zonation_output")  # Zonation's files

    # or an existing Zonation 5 project, with z5's switches
    res = ls.prioritize.run_project("settings.z5", "-w -X", mode="CAZ2")

The optimization paradigm of prioritizr (Hanson et al.; GPL-3): a problem of
planning units, features, an objective, targets and constraints, compiled to
the same mixed-integer model as prioritizr and solved on HiGHS (built in) or
Gurobi::

    p = (ls.prioritize.problem(cost, features)
           .add_min_set_objective()
           .add_relative_targets(0.3)
           .add_binary_decisions())
    sol = p.solve()
    sol.selection                 # 1 = selected
    sol.coverage                  # targets met per feature
"""

from ._actions import ActionsData, ActionsProblem, ActionsSolution, actions_data
from ._actions_raster import actions_raster_data
from ._connectivity import Interaction, Kernel, convolve
from ._impacts import CumulativeScenario, ImpactScenario, ImpactSource, Pressure
from ._marxan import MarxanResult, marxan
from ._milp import OptimizationProblem, Problem, h3_units, problem
from ._milp_matrices import (adjacency_matrix, boundary_matrix, connectivity_matrix, decay_connectivity,
                             proximity_matrix, unit_points)
from ._milp_portfolio import Portfolio
from ._milp_solve import Solution
from ._postprocess import AreaStatistics
from ._project import read_project, run_project
from ._zonation import RULES, ZonationResult, zonation

__all__ = ["ActionsData", "ActionsProblem", "ActionsSolution", "actions_data", "actions_raster_data", "adjacency_matrix", "boundary_matrix", "connectivity_matrix", "decay_connectivity", "proximity_matrix",
           "unit_points", "Portfolio", "MarxanResult", "marxan", "AreaStatistics", "CumulativeScenario", "ImpactScenario", "ImpactSource", "Interaction", "Kernel",
           "OptimizationProblem", "Pressure", "Problem", "RULES", "Solution", "ZonationResult", "convolve",
           "h3_units", "problem", "read_project", "run_project", "zonation"]
