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
    links, reaches = ls.hydro.stream_links(filled, rivers)
    basins = ls.hydro.watersheds(filled, links)
    wetness = ls.hydro.twi(filled)
"""

from ._flow import (D8_CODES, breach, distance_to_channel, fill, flow_accumulation, flow_direction, hand,
                    streams)
from ._basins import snap_outlets, stream_links, watersheds
from ._indices import spi, sti, twi
from ..services._sdr import connectivity_index
from . import swat

__all__ = ["D8_CODES", "breach", "connectivity_index", "distance_to_channel", "fill", "flow_accumulation",
           "flow_direction", "hand", "snap_outlets", "spi", "sti", "stream_links", "streams", "swat", "twi",
           "watersheds"]
