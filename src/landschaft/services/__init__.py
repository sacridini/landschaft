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
"""

from ._carbon import Carbon, carbon, valuation_constant
from ._habitat import HabitatQuality, habitat_quality

__all__ = ["Carbon", "HabitatQuality", "carbon", "habitat_quality", "valuation_constant"]
