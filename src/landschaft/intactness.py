# SPDX-License-Identifier: GPL-3.0-or-later
"""Ecological intactness from the human footprint (Beyer et al. 2020).

A cell's habitat quality is q = exp(-gamma HFI) for its human footprint index HFI, and its
intactness Q_i = sum_j (q_i q_j)^z w_ij / sum_j w_ij: the quality of the cell and of its
surroundings within a radius, weighted by w = exp(beta d) at a distance d (km). The computation and
its parameters follow plangea's ``multi_calc_it`` (International Institute for Sustainability;
GPL >= 3), which uses it as a benefit of restoration: restoring a share x of a cell lowers its
footprint by a reduction per unit, and :func:`intactness_gradient` gives what that adds to the
total intactness.

References
----------
Beyer, H. L., Venter, O., Grantham, H. S. & Watson, J. E. M. (2020). Substantial losses in
    ecoregion intactness highlight urgency of globally coordinated action. *Conservation
    Letters* 13, e12692. doi:10.1111/conl.12692
"""

from __future__ import annotations

import math

import numpy as np
import xarray as xr

from . import _core, _threads
from ._layers import like

__all__ = ["intactness", "intactness_gradient", "INTACTNESS_GAMMA"]

#: Beyer et al.'s gamma: a footprint of 4 gives a quality of 0.2.
INTACTNESS_GAMMA = -math.log(0.2) / 4


def _grid(hfi) -> tuple[xr.DataArray, np.ndarray]:
    from .io import _load_map

    if isinstance(hfi, np.ndarray):
        da = xr.DataArray(hfi, dims=("y", "x"))
    else:
        da = hfi if isinstance(hfi, xr.DataArray) else _load_map(hfi)
    if da.ndim == 3 and da.shape[0] == 1:
        da = da.isel({da.dims[0]: 0})
    if da.ndim != 2:
        raise ValueError("the human footprint must be a single 2-D raster")
    return da, np.ascontiguousarray(np.asarray(da.values, dtype=np.float64))


def _like_grid(data, shape, what: str) -> np.ndarray | None:
    if data is None:
        return None
    if isinstance(data, (int, float, np.integer, np.floating)):
        return np.full(shape, float(data))
    v = np.asarray(getattr(data, "values", data), dtype=np.float64).squeeze()
    if v.shape != shape:
        raise ValueError(f"{what}: grid {v.shape}, the footprint's {shape}")
    return np.ascontiguousarray(v)


def _cell_km(da: xr.DataArray, cell_size) -> tuple[float, float]:
    if cell_size is not None:
        if np.ndim(cell_size) == 0:
            return float(cell_size), float(cell_size)
        dx, dy = cell_size
        return float(dx), float(dy)
    try:
        crs = da.rio.crs
        rx, ry = da.rio.resolution()
    except Exception as e:  # noqa: BLE001
        raise ValueError("give cell_size (km): the footprint has no georeferenced grid") from e
    if crs is None:
        raise ValueError("give cell_size (km): the footprint has no coordinate system")
    from pyproj import CRS

    c = CRS.from_user_input(crs)
    if c.is_geographic:
        raise ValueError("the footprint is in degrees: project it, or give cell_size (km)")
    unit = c.axis_info[0].unit_conversion_factor if c.axis_info else 1.0
    # plangea: xres(HFI) / 1e3 (metres to km)
    return abs(float(rx)) * unit / 1e3, abs(float(ry)) * unit / 1e3


def _check(radius, z) -> int:
    r = int(radius)
    if r != radius or r < 0:
        raise ValueError(f"radius must be a whole number of cells >= 0, got {radius}")
    if not z > 0:
        raise ValueError(f"z must be above 0, got {z}")
    return r


def intactness(
    hfi,
    radius: int = 3,
    *,
    gamma: float = INTACTNESS_GAMMA,
    beta: float = -0.2,
    z: float = 0.5,
    cell_size=None,
    reduction=None,
    restored=None,
    n_threads: int | None = None,
) -> xr.DataArray:
    """Ecological intactness of every cell from the human footprint (Beyer et al. 2020).

    The habitat quality of a cell is q = exp(-gamma HFI), from 1 where
    people leave no footprint down towards 0; its intactness

    .. math:: Q_i = \\frac{\\sum_j (q_i q_j)^z w_{ij}}{\\sum_j w_{ij}}

    over the cells j within ``radius`` cells of it (itself included), with
    weights w = exp(beta d) that fall with the distance d in km: a cell is
    intact when it and its surroundings are of high quality.

    Parameters
    ----------
    hfi
        The human footprint index (a raster, NaN outside the area).
    radius
        The radius of the neighbourhood, in cells (plangea's ``n_pixels``):
        the cells whose centres are within ``radius`` cell widths.
    gamma, beta, z
        Beyer et al.'s parameters: gamma = -ln(0.2) / 4 (a footprint of 4
        gives a quality of 0.2), beta = -0.2 per km, z = 0.5.
    cell_size
        The width (and height) of a cell in km, or ``(width, height)``; by
        default from a projected grid (its unit converted to km).
    reduction, restored
        Restoration: the footprint a cell loses per unit of its area restored
        (a raster or a number) and the share of each cell restored. The
        footprint becomes HFI - reduction x restored, at least 0.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        Q on the footprint's grid, 0 to 1 (NaN where the footprint is).

    Notes
    -----
    The values are plangea's ``multi_calc_it`` (with ``aggregate = TRUE``),
    bit for bit, except at the edges: plangea leaves out the cells closer
    than the radius to an edge (and one more row and column at the end) and
    gives them the mean of the others; here their window is the part inside
    the raster, as it is around cells without a footprint.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> hfi = np.zeros((5, 5))
    >>> hfi[:, 3:] = 20.0                      # a city on the right
    >>> q = ls.intactness(hfi, radius=1, cell_size=1.0)
    >>> q.values[2].round(3)
    array([1.   , 1.   , 0.812, 0.004, 0.   ])
    """
    r = _check(radius, z)
    da, h = _grid(hfi)
    a = _like_grid(reduction, h.shape, "reduction")
    x = _like_grid(restored, h.shape, "restored")
    threads = _threads.resolve(n_threads)
    if a is not None and x is not None:
        h = _core.rest_footprint(h, a, x, threads)
    dx, dy = _cell_km(da, cell_size)
    q = _core.rest_intactness(h, float(gamma), float(beta), float(z), r, dx, dy, threads)
    return like(da, q, "intactness")


def intactness_gradient(
    hfi,
    reduction,
    radius: int = 3,
    *,
    gamma: float = INTACTNESS_GAMMA,
    beta: float = -0.2,
    z: float = 0.5,
    cell_size=None,
    restored=None,
    n_threads: int | None = None,
) -> xr.DataArray:
    """What restoring each cell adds to the total ecological intactness.

    Restoring a share x of a cell lowers its human footprint by
    ``reduction`` x; this is the derivative of the sum of
    :func:`intactness` over all cells with respect to x: the cell's own
    intactness rises, and so does that of every cell whose neighbourhood
    holds it,

    .. math:: \\frac{\\partial \\sum_i Q_i}{\\partial x_k} = z\\,\\gamma\\,a_k
              \\Big(Q_k + q_k^z \\sum_{i} \\frac{q_i^z w_{ik}}{W_i}\\Big)

    (a the reduction, W_i the sum of the weights of i's window), 0 where
    the footprint is already 0. It is the linear benefit of restoration
    for intactness in a restoration plan.

    Parameters
    ----------
    hfi, radius, gamma, beta, z, cell_size, n_threads
        As :func:`intactness`.
    reduction
        The footprint a cell loses per unit of its area restored (a raster
        or a number): in plangea, the anthropic classes' shares weighted by
        their footprint (``hfi_weights``).
    restored
        Shares already restored (the derivative at that point).

    Returns
    -------
    xarray.DataArray
        The derivative on the footprint's grid.

    Notes
    -----
    plangea's derivative changes only the cell's own factor q_i (by a finite
    difference), leaving out the cell inside its own window and its effect
    on its neighbours' intactness; this is the complete one (the derivative
    of the total intactness).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> hfi = np.full((5, 5), 4.0)
    >>> g = ls.intactness_gradient(hfi, reduction=2.0, radius=1, cell_size=1.0)
    >>> round(float(g.values[2, 2]), 4)
    0.1609
    """
    r = _check(radius, z)
    da, h = _grid(hfi)
    a = _like_grid(reduction, h.shape, "reduction")
    if a is None:
        raise ValueError("the reduction of the footprint per unit restored is needed")
    threads = _threads.resolve(n_threads)
    x = _like_grid(restored, h.shape, "restored")
    if x is not None:
        h = _core.rest_footprint(h, a, x, threads)
    dx, dy = _cell_km(da, cell_size)
    g = _core.rest_intactness_gradient(h, a, float(gamma), float(beta), float(z), r, dx, dy, threads)
    return like(da, g, "intactness_gradient")
