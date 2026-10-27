# SPDX-License-Identifier: GPL-3.0-or-later
"""Ecosystem services: the models of InVEST, reimplemented.

Ports of the Natural Capital Project's InVEST models (natcap.invest, Apache-2.0),
computed in C++ with OpenMP from landschaft's data objects: their maps, bit for
bit where the arithmetic permits, from land use maps and parameter tables.

Typical use::

    import landschaft as ls

    lulc = ls.load_raster("lulc_2020.tif")
    pools = ls.load_table("carbon_pools.csv")
    c = ls.services.carbon(lulc, pools, future=ls.load_raster("lulc_2050.tif"))
    c.totals                                   # tons of carbon, and the change

    hq = ls.services.habitat_quality(lulc, threats, sensitivity, threat_maps, half_saturation=0.05)
    hq.quality["current"]

    wy = ls.services.water_yield(lulc, biophysical, rain, et0, soil_depth, pawc, seasonality=5,
                                 watersheds=ls.load_vector("watersheds.gpkg"))
    wy.watersheds                              # mean yield and volume per watershed

    nd = ls.services.ndr(dem, lulc, runoff_proxy, nutrients_table, watersheds, threshold=1000,
                         subsurface_critical_length=200, subsurface_efficiency=0.8)
    nd.watersheds                              # nitrogen and phosphorus exports per watershed

    bees = ls.services.pollination(lulc, guilds, biophysical, farms=ls.load_vector("farms.gpkg"))
    bees.abundance["apis"]["spring"], bees.farms   # abundance by species and season, farm yields

    swy = ls.services.seasonal_water_yield(dem, lulc, swy_biophysical, watersheds, threshold=1000,
                                           precipitation=monthly_rain, et0=monthly_et0, soil_group=soils,
                                           rain_events=[12, 10, 9, 6, 4, 2, 1, 1, 3, 6, 9, 11])
    swy.baseflow                               # the recharge that reaches the streams (mm / yr)

    cv = ls.services.coastal_vulnerability(aoi, landmass, wwiii, bathymetry, shelf, dem, habitats,
                                           model_resolution=250, max_fetch_distance=12000,
                                           dem_averaging_radius=1000)
    cv.exposure                                # exposure index of the shore points, with and without habitats
"""

from ._carbon import Carbon, carbon, valuation_constant
from ._coastal import CoastalVulnerability, coastal_vulnerability
from ._habitat import HabitatQuality, habitat_quality
from ._ndr import NDR, ndr
from ._pollination import Pollination, pollination
from ._sdr import SDR, sdr
from ._swy import SeasonalWaterYield, seasonal_water_yield
from ._water import WaterYield, water_yield

__all__ = ["Carbon", "CoastalVulnerability", "HabitatQuality", "NDR", "Pollination", "SDR", "SeasonalWaterYield",
           "WaterYield", "carbon", "coastal_vulnerability", "habitat_quality", "ndr", "pollination", "sdr",
           "seasonal_water_yield", "valuation_constant", "water_yield"]
