# SPDX-License-Identifier: GPL-3.0-or-later
"""Gravitational interaction potential (Dinamica EGO's CalcInteractionPotentialMap)."""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

from .. import _core, _threads
from .._threads import threaded
from ._grid import read_surface

__all__ = ["gravity"]

# method="auto": direct sums with up to this many sources (or source-cell pairs), the FFT beyond (measured:
# ~1e-10 s per pair against ~3e-7 s per cell for the FFT, on 4 threads)
_DIRECT_SOURCES = 2000
_DIRECT_PAIRS = 1e8


def _fft_potential(mass: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The same sum as a linear convolution of the masses with the mirrored weight table."""
    from scipy import fft

    ny, nx = mass.shape
    # k[di + ny - 1, dj + nx - 1] = weight of the offset (|di|, |dj|)
    k = weights[np.ix_(np.abs(np.arange(1 - ny, ny)), np.abs(np.arange(1 - nx, nx)))]
    shape = (fft.next_fast_len(3 * ny - 2, real=True), fft.next_fast_len(3 * nx - 2, real=True))
    workers = _threads.current()
    f = fft.rfft2(mass, s=shape, workers=workers) * fft.rfft2(k, s=shape, workers=workers)
    full = fft.irfft2(f, s=shape, workers=workers)
    return np.ascontiguousarray(full[ny - 1:2 * ny - 1, nx - 1:2 * nx - 1])


@threaded
def gravity(sources, exponent: float = 2.0, *, mask=None, unit: str = "cells", method: str = "auto",
            extent=None, crs=None, band=None, nodata=None) -> xr.DataArray:
    """Gravitational interaction potential of source cells.

    Every cell gets the sum over the source cells of their mass over the
    distance to them raised to ``exponent``: P_i = sum_k m_k / d_ik^beta, the
    gravity model of spatial interaction (a "pressure" of towns, markets or
    roads to use as a driver of land change). The sources are the cells of
    ``sources`` with a value (other than 0); the value is the mass. A source's
    own cell is at a distance of one cell. This is Dinamica EGO's
    ``CalcInteractionPotentialMap``.

    Parameters
    ----------
    sources : str, path, xarray.DataArray or array_like
        The masses (NaN or ``nodata``: not a source).
    exponent : float, default 2
        The distance decay exponent beta (0: no decay).
    mask : str, path, xarray.DataArray or array_like, optional
        A raster on the same grid; the potential is missing where it is
        missing. The sources under its missing cells still count.
    unit : {"cells", "map"}, default "cells"
        Distances in cells (Dinamica) or in map units (cell width along
        rows, height along columns; a source's own cell at the square root of
        the cell area).
    method : {"auto", "direct", "fft"}, default "auto"
        ``"direct"``: the sum over every source-cell pair, in a fixed order
        (the same bits with any thread count); ``"fft"``: the same sum as a
        convolution by FFT (rounding errors of about 1e-15 of the largest
        term, much faster with many sources); ``"auto"``: direct up to 2000
        sources (or 1e8 source-cell pairs).
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    xarray.DataArray
        The potential (float64), 0 where no source reaches (no sources).

    Notes
    -----
    Dinamica computes each term and their sum in float32 (relative errors
    around 1e-7, growing with the number of sources) and scales the
    potential to the range of an integer cell type when asked for one; here
    the sum is in float64 and never scaled. Its distances are always in
    cells.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> towns = np.full((3, 4), np.nan)
    >>> towns[0, 0], towns[2, 3] = 100, 50
    >>> p = ls.surface.gravity(towns)
    >>> round(float(p[0, 0]), 4), round(float(p[1, 1]), 4)
    (103.8462, 60.0)
    """
    if unit not in ("cells", "map"):
        raise ValueError(f"unit must be 'cells' or 'map', got {unit!r}")
    if method not in ("auto", "direct", "fft"):
        raise ValueError(f"method must be 'auto', 'direct' or 'fft', got {method!r}")
    beta = float(exponent)
    if not math.isfinite(beta):
        raise ValueError("exponent must be finite")
    s = read_surface(sources, extent=extent, crs=crs, band=band, nodata=nodata)
    m = None
    if mask is not None:
        m = read_surface(mask, extent=extent, crs=crs).values
        if m.shape != s.shape:
            raise ValueError(f"the mask has shape {m.shape}, the sources {s.shape}")
        m = np.ascontiguousarray(m, dtype=np.float64)
    dx, dy = (1.0, 1.0) if unit == "cells" else tuple(float(abs(v)) for v in s.res)
    self_d = math.sqrt(dx * dy)
    mass = np.ascontiguousarray(s.values, dtype=np.float64)
    n_sources = int(np.count_nonzero(~np.isnan(mass) & (mass != 0)))
    if method == "auto":
        method = "direct" if n_sources <= _DIRECT_SOURCES or n_sources * float(mass.size) <= _DIRECT_PAIRS else "fft"
    if method == "direct" or n_sources == 0:
        out = np.asarray(_core.surf_gravity(mass, m, beta, dx, dy, self_d))
    else:
        ny, nx = mass.shape
        w = np.asarray(_core.surf_gravity_weights(ny, nx, beta, dx, dy, self_d))
        out = _fft_potential(np.where(np.isnan(mass), 0.0, mass), w)
        if m is not None:
            out[np.isnan(m)] = np.nan
    return s.raster(out, "gravity", {"exponent": beta, "unit": unit})
