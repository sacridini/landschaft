# SPDX-License-Identifier: GPL-3.0-or-later
"""SWAT+ (``ls.hydro.swat``): the watershed model of USDA-ARS and Texas A&M, ported.

The core of SWAT+ 62.0.1 (github.com/swat-model/swatplus, LGPL-2.1) in C++: the
results of SWAT+ itself, bit for bit, for the part of the model this port covers
(see IDEAS.md, "SWAT+"): weather (measured series and the weather generator), HRUs
(snow, ET, curve number runoff, MUSLE erosion, percolation, lateral flow, nutrients,
plant growth, scheduled management and decision-table irrigation), routing units,
1-D aquifers and SWAT-DEG channels, and the tables SWAT+ prints.

Typical use::

    import landschaft as ls

    project = ls.hydro.swat.load_txtinout("my_model/TxtInOut")
    result = ls.hydro.swat.run(project, tables={"hru_ls": "year", "channel_sd": "day"})
    result.table("hru_ls", "year")["sedyld"]        # t/ha of each HRU and year
    result.table("channel_sd", "day")["flo_out"]    # m3/s leaving each channel
"""

from ._project import SwatProject, load_txtinout
from ._run import STEPS, TABLES, WEATHER_COLUMNS, SwatResult, run

__all__ = ["STEPS", "SwatProject", "SwatResult", "TABLES", "WEATHER_COLUMNS", "load_txtinout", "run"]
