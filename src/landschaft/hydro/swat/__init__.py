# SPDX-License-Identifier: GPL-3.0-or-later
"""SWAT+ (``ls.hydro.swat``): the watershed model of USDA-ARS and Texas A&M, ported.

The core of SWAT+ 62.0.1 (github.com/swat-model/swatplus, LGPL-2.1) in C++: the
results of SWAT+ itself, bit for bit, for the part of the model this port covers
(see IDEAS.md, "SWAT+"): weather (measured series and the weather generator), HRUs
(snow, ET, curve number runoff, MUSLE erosion, percolation, lateral flow, nutrients,
plant growth, scheduled management and decision-table irrigation), routing units,
1-D aquifers and SWAT-DEG channels, and the tables SWAT+ prints.

A project comes from a TxtInOut folder, or from the landscape: :func:`watershed`
divides a DEM with land use and soil maps into subbasins, channels and HRUs as QSWAT+
does, :func:`from_tables` writes the project as the SWAT+ Editor does (the same files),
:func:`from_rasters` does both; :func:`hru_map` and :func:`channel_map` put results
back on the map.

Typical use::

    import landschaft as ls

    project = ls.hydro.swat.load_txtinout("my_model/TxtInOut")
    result = ls.hydro.swat.run(project, tables={"hru_ls": "year", "channel_sd": "day"})
    result.table("hru_ls", "year")["sedyld"]        # t/ha of each HRU and year
    result.table("channel_sd", "day")["flo_out"]    # m3/s leaving each channel

    wx = ls.hydro.swat.climate(stations, pcp=pcp, tmax=tmax, tmin=tmin, monthly={"pcp_hhr": 15.0, ...})
    project = ls.hydro.swat.from_rasters(dem, landuse, soils, landuse_lookup=lu_codes, soil_lookup=soil_codes,
                                         soil_table=usersoil, climate=wx, stream_threshold=2.0)
    soil_loss = ls.hydro.swat.hru_map(project, "sedyld")      # t/ha per year
    discharge = ls.hydro.swat.channel_map(project)            # mean m3/s on the streams
"""

from ._editor import from_tables
from ._landscape import (CHANNEL_COLUMNS, HRU_COLUMNS, channel_map, from_rasters, hru_map, load_qswat_project,
                         reach_discharge)
from ._project import SwatProject, load_txtinout
from ._qswat import SwatWatershed, watershed
from ._run import STEPS, TABLES, WEATHER_COLUMNS, SwatResult, run
from ._weather import WGN_COLUMNS, SwatClimate, climate, wgn_from_series

__all__ = ["CHANNEL_COLUMNS", "HRU_COLUMNS", "STEPS", "SwatClimate", "SwatProject", "SwatResult", "SwatWatershed",
           "TABLES", "WEATHER_COLUMNS", "WGN_COLUMNS", "channel_map", "climate", "from_rasters", "from_tables",
           "hru_map", "load_qswat_project", "load_txtinout", "reach_discharge", "run", "watershed", "wgn_from_series"]
