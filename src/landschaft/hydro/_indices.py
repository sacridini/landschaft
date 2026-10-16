# SPDX-License-Identifier: GPL-3.0-or-later
"""Topographic wetness, stream power and sediment transport indices.

Each combines the specific contributing area of a cell (from a multiple-flow
accumulation by default: D8 draws artificial parallel lines on hillslopes) with its
slope, as WhiteboxTools' ``WetnessIndex``, ``StreamPowerIndex`` and
``SedimentTransportIndex`` do. The slope is the gradient of Horn (1981) as a tangent,
computed without trigonometry, or a slope raster in degrees.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .._threads import threaded
from ..surface._grid import Surface, read_surface
from ._flow import _cells, _read, flow_accumulation

__all__ = ["spi", "sti", "twi"]


def _tan_slope(s: Surface) -> np.ndarray:
    """tan(slope) by Horn (1981); a missing or outside neighbour takes the cell's own value."""
    z = s.values
    dx, dy = _cells(s)
    p = np.pad(z, 1, mode="constant", constant_values=np.nan)
    centre = p[1:-1, 1:-1]

    def at(dr, dc):
        v = p[1 + dr:p.shape[0] - 1 + dr, 1 + dc:p.shape[1] - 1 + dc]
        return np.where(np.isnan(v), centre, v)

    z1, z2, z3 = at(-1, -1), at(-1, 0), at(-1, 1)
    z4, z6 = at(0, -1), at(0, 1)
    z7, z8, z9 = at(1, -1), at(1, 0), at(1, 1)
    gx = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * dx)
    gy = ((z7 + 2.0 * z8 + z9) - (z1 + 2.0 * z2 + z3)) / (8.0 * dy)
    return np.sqrt(gx * gx + gy * gy)


def _parts(dem, sca, slope, method, flats, extent, crs, band, nodata):
    s = _read(dem, extent, crs, band, nodata)
    if sca is None:
        a = flow_accumulation(s, method=method, units="sca", flats=flats).values
    else:
        a = read_surface(sca).values
        if a.shape != s.shape:
            raise ValueError("the specific contributing area differs in shape from the DEM")
    if slope is None:
        t = _tan_slope(s)
    else:
        deg = read_surface(slope).values
        if deg.shape != s.shape:
            raise ValueError("the slope differs in shape from the DEM")
        t = np.tan(np.radians(deg))
    return s, a, t


@threaded
def twi(dem, *, method: str = "fd8", min_slope: float = 0.0, sca=None, slope=None, flats: bool = True, extent=None,
        crs=None, band=None, nodata=None) -> xr.DataArray:
    """Topographic wetness index: ln(a / tan(slope)) (Beven & Kirkby 1979).

    High where much area drains through a gentle slope (valley floors, hollows):
    a proxy of soil moisture and saturation used in species distribution and land
    change models.

    Parameters
    ----------
    dem
        The elevations, with depressions filled (:func:`fill`).
    method
        The flow accumulation of the specific contributing area ``a`` (see
        :func:`flow_accumulation`): ``"fd8"`` (default), ``"quinn"``, ``"dinf"``, ``"d8"``.
    min_slope
        A floor for tan(slope), so flat cells get a value (e.g. 0.001); with 0 they
        are NaN, as in WhiteboxTools.
    sca
        A specific contributing area raster to use instead.
    slope
        A slope raster in degrees to use instead of the gradient of the DEM.
    flats
        Drain flat areas (see :func:`flow_direction`).
    extent, crs, band, nodata
        For arrays without georeferencing, or to pick a band and a missing value.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.add.outer(np.arange(5.0, 0, -1), np.zeros(4))   # a plane falling to the south
    >>> w = ls.hydro.twi(dem)
    >>> bool(w.values[4, 1] > w.values[0, 1])                     # wetter downslope
    True
    """
    s, a, t = _parts(dem, sca, slope, method, flats, extent, crs, band, nodata)
    if min_slope > 0:
        t = np.maximum(t, min_slope)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(t > 0, np.log(a / t), np.nan)
    return s.raster(out, "twi", {"method": method})


@threaded
def spi(dem, *, method: str = "fd8", sca=None, slope=None, flats: bool = True, extent=None, crs=None, band=None,
        nodata=None) -> xr.DataArray:
    """Stream power index: a · tan(slope) (Moore, Grayson & Ladson 1991), the erosive
    power of overland flow.

    Parameters
    ----------
    dem, method, sca, slope, flats, extent, crs, band, nodata
        As in :func:`twi`.

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.add.outer(np.arange(5.0, 0, -1), np.zeros(4))
    >>> bool(ls.hydro.spi(dem).values[3, 1] > ls.hydro.spi(dem).values[1, 1])
    True
    """
    s, a, t = _parts(dem, sca, slope, method, flats, extent, crs, band, nodata)
    return s.raster(a * t, "spi", {"method": method})


@threaded
def sti(dem, *, method: str = "fd8", sca_exponent: float = 0.4, slope_exponent: float = 1.3, sca=None, slope=None,
        flats: bool = True, extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Sediment transport index, the LS factor of the USLE from the contributing area
    (Moore & Burch 1986): (m + 1) (a / 22.13)^m (sin(slope) / 0.0896)^n.

    Parameters
    ----------
    dem, method, sca, slope, flats, extent, crs, band, nodata
        As in :func:`twi`.
    sca_exponent, slope_exponent
        m (0.4) and n (1.3).

    Returns
    -------
    xarray.DataArray

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dem = np.add.outer(np.arange(5.0, 0, -1), np.zeros(4))
    >>> float(ls.hydro.sti(dem).values[0, 0]) > 0
    True
    """
    s, a, t = _parts(dem, sca, slope, method, flats, extent, crs, band, nodata)
    if slope is None:
        sin = t / np.sqrt(1.0 + t * t)
    else:
        sin = np.sin(np.radians(read_surface(slope).values))
    m, n = float(sca_exponent), float(slope_exponent)
    out = (m + 1.0) * np.power(a / 22.13, m) * np.power(sin / 0.0896, n)
    return s.raster(out, "sti", {"method": method})
