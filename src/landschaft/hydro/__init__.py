# SPDX-License-Identifier: GPL-3.0-or-later
"""Terrain hydrology (``ls.hydro``): depression filling, flow directions, flow
accumulation, stream networks and the height above the nearest drainage (HAND).

Typical use::

    import landschaft as ls

    dem = ls.load_raster("srtm.tif")
    filled = ls.hydro.fill(dem)
    acc = ls.hydro.flow_accumulation(filled, method="fd8", units="sca")
    rivers = ls.hydro.streams(ls.hydro.flow_accumulation(filled), 1000)
    height = ls.hydro.hand(filled, rivers)
    wetness = ls.hydro.twi(filled)
"""

from ._flow import (D8_CODES, breach, distance_to_channel, fill, flow_accumulation, flow_direction, hand,
                    streams)
from ._indices import spi, sti, twi
from ..services._sdr import connectivity_index
from . import swat

__all__ = ["D8_CODES", "breach", "connectivity_index", "distance_to_channel", "fill", "flow_accumulation",
           "flow_direction", "hand", "spi", "sti", "streams", "swat", "twi"]
