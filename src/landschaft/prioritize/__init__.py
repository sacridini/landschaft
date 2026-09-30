# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial conservation prioritization: ranking the cells of a landscape.

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
"""

from ._connectivity import Interaction, Kernel, convolve
from ._impacts import CumulativeScenario, ImpactScenario, ImpactSource, Pressure
from ._postprocess import AreaStatistics
from ._project import read_project, run_project
from ._zonation import RULES, ZonationResult, zonation

__all__ = ["AreaStatistics", "CumulativeScenario", "ImpactScenario", "ImpactSource", "Interaction", "Kernel",
           "Pressure", "RULES", "ZonationResult", "convolve", "read_project", "run_project", "zonation"]
