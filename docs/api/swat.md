# Watershed model (SWAT+)

Daily water, sediment and nutrient balances of a watershed: the core of the
SWAT+ model (USDA-ARS and Texas A&M AgriLife, version 62.0.1), ported to C++.
For the part of SWAT+ it covers, the port gives SWAT+'s own results, value for
value: weather (measured series and the weather generator), the land phase of
each hydrologic response unit (snow, interception, soil temperature, potential
and actual evapotranspiration, curve number runoff, MUSLE erosion, percolation,
lateral flow, residue and nutrient cycling with the static carbon model, plant
growth, scheduled operations and decision tables), routing units, 1-D aquifers
and SWAT-DEG channels with variable storage or Muskingum routing, and the
tables SWAT+ prints (by day, month, year and average annual). They live in the
`ls.hydro.swat` module.

A model is a `SwatProject`: the files of a SWAT+ TxtInOut folder, read with
`load_txtinout`, or made from the landscape. `watershed` divides a DEM with
land use and soil maps into subbasins, channels and HRUs as QSWAT+ does,
`from_tables` writes the project files as the SWAT+ Editor does (the same
files), `from_rasters` does both. `run` simulates the project; `hru_map` and
`channel_map` put the results back on the map. A project made here also runs
in SWAT+ itself (`SwatProject.write`).

The erosion of MUSLE uses SWAT+ 62's cover factor C by default: the cover of
the residue times the cover of the growing biomass, in which a tall canopy
hardly protects the soil, so a forest's C (0.5-0.9 under little litter) can
exceed a pasture's. `run(..., c_factor="plant")` takes the earlier SWAT+
method instead, the plants' minimum C (forest 0.001); only sediment and what
travels with it change.

What the port does not simulate is refused with a clear error when a project
uses it (reservoirs, wetlands, pesticides, salts, constituents, Green-Ampt
infiltration, the CENTURY carbon model, water allocation...).

<!-- no-run -->
```python
import landschaft as ls

project = ls.hydro.swat.load_txtinout("my_model/TxtInOut")
result = ls.hydro.swat.run(project, tables={"hru_ls": "aa", "channel_sd": "day"})
result.table("hru_ls", "aa")["sedyld"]          # t/ha per year of each HRU
result.table("channel_sd", "day")["flo_out"]    # m3/s leaving each channel

wx = ls.hydro.swat.climate(stations, pcp=pcp, tmax=tmax, tmin=tmin,
                           monthly={"pcp_hhr": 15.0, "slr_ave": 14.0, "dew_ave": 5.0, "wnd_ave": 3.0})
project = ls.hydro.swat.from_rasters(dem, landuse, soils, landuse_lookup=lu_codes, soil_lookup=soil_codes,
                                     soil_table=usersoil, climate=wx, stream_threshold=2.0)
erosion = ls.hydro.swat.hru_map(project, "sedyld")    # t/ha per year, on the DEM's grid
```

## Projects

::: landschaft.hydro.swat.load_txtinout

::: landschaft.hydro.swat.SwatProject

## Simulation

::: landschaft.hydro.swat.run

::: landschaft.hydro.swat.SwatResult

## From the landscape

::: landschaft.hydro.swat.watershed

::: landschaft.hydro.swat.SwatWatershed

::: landschaft.hydro.swat.from_tables

::: landschaft.hydro.swat.from_rasters

::: landschaft.hydro.swat.load_qswat_project

::: landschaft.hydro.swat.example_landscape

## Weather

::: landschaft.hydro.swat.climate

::: landschaft.hydro.swat.SwatClimate

::: landschaft.hydro.swat.wgn_from_series

## Results on the map

::: landschaft.hydro.swat.hru_map

::: landschaft.hydro.swat.channel_map

::: landschaft.hydro.swat.reach_discharge
