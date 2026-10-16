# SPDX-License-Identifier: GPL-3.0-or-later
"""SWAT+ (``ls.hydro.swat``): the watershed model of USDA-ARS and Texas A&M, ported.

The core of SWAT+ 62.0.1 (github.com/swat-model/swatplus, LGPL-2.1) in C++: the
results of SWAT+ itself, bit for bit, for the part of the model this port covers
(see IDEAS.md, "SWAT+"). Work in progress: so far the daily weather (measured
series and the weather generator).

Typical use::

    import landschaft as ls

    project = ls.hydro.swat.load_txtinout("my_model/TxtInOut")
    result = ls.hydro.swat.run(project)
    result.weather.xs("s43952n71725w", level="station")
"""

from ._project import SwatProject, load_txtinout
from ._run import WEATHER_COLUMNS, SwatResult, run

__all__ = ["SwatProject", "SwatResult", "WEATHER_COLUMNS", "load_txtinout", "run"]
