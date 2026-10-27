# SPDX-License-Identifier: GPL-3.0-or-later
"""Sampling-bias surfaces: a kernel density of the records of a whole taxonomic group."""

from __future__ import annotations

import numpy as np
import xarray as xr

from . import _points
from ._background import _complete, _grid

__all__ = ["bias_surface"]

# dnorm(u) underflows to 0 (in R and in double precision) beyond |u| ~ 38.6: the cells beyond 40
# bandwidths of a point add exactly nothing, so they are skipped without changing the sum.
_REACH = 40.0
_CHUNK = 512


def _bw_nrd(v: np.ndarray) -> float:
    """R's ``stats::bw.nrd``: 1.06 min(sd, IQR / 1.34) n^(-1/5)."""
    n = v.size
    if n < 2:
        return float("nan")
    q1, q3 = np.quantile(v, [0.25, 0.75])  # type 7, R's default
    h = (q3 - q1) / 1.34
    return float(1.06 * min(float(np.std(v, ddof=1)), h) * n ** (-0.2))


def _bandwidths(bandwidth, isotropic: bool, limits, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    if isinstance(bandwidth, str):
        if bandwidth != "nrd":
            raise ValueError(f"unknown bandwidth rule {bandwidth!r}; 'nrd' or a number (CRS units)")
        bx, by = _bw_nrd(xs), _bw_nrd(ys)
    elif np.ndim(bandwidth) == 0:
        bx = by = float(bandwidth)
    else:
        bx, by = (float(b) for b in bandwidth)
    if isotropic:
        bx = by = max(bx, by)
    if limits is not None:
        lo, hi = limits
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        bx, by = min(max(bx, lo), hi), min(max(by, lo), hi)
    if not (np.isfinite(bx) and np.isfinite(by) and bx > 0 and by > 0):
        raise ValueError(f"bandwidths must be positive (got {bx}, {by}): with fewer than two distinct "
                         "coordinates, give a number or limits=")
    return bx, by


def _density(gx: np.ndarray, gy: np.ndarray, xs: np.ndarray, ys: np.ndarray, w: np.ndarray, bx: float,
             by: float) -> np.ndarray:
    """sum_k w_k phi((gx - x_k) / bx) phi((gy - y_k) / by) / (sum w bx by), on the grid (len(gy), len(gx)).

    The product kernel of ``MASS::kde2d`` as a matrix product, in blocks of points sorted by y, each
    over only the rows and columns its points reach.
    """
    z = np.zeros((gy.size, gx.size))
    order = np.argsort(ys, kind="stable")
    xs, ys, w = xs[order], ys[order], w[order]
    # the grid's coordinates in increasing order, for searchsorted
    ix = np.argsort(gx, kind="stable")
    iy = np.argsort(gy, kind="stable")
    sx, sy = gx[ix], gy[iy]
    c = 1.0 / np.sqrt(2.0 * np.pi)
    for s in range(0, xs.size, _CHUNK):
        px, py, pw = xs[s:s + _CHUNK], ys[s:s + _CHUNK], w[s:s + _CHUNK]
        c0, c1 = np.searchsorted(sx, [px.min() - _REACH * bx, px.max() + _REACH * bx], side="left")
        r0, r1 = np.searchsorted(sy, [py.min() - _REACH * by, py.max() + _REACH * by], side="left")
        c1 = min(c1 + 1, sx.size)
        r1 = min(r1 + 1, sy.size)
        if c0 >= c1 or r0 >= r1:
            continue
        cols, rows = ix[c0:c1], iy[r0:r1]
        ux = (gx[cols][:, None] - px[None, :]) / bx
        uy = (gy[rows][:, None] - py[None, :]) / by
        ax = c * np.exp(-0.5 * ux * ux)
        ay = c * np.exp(-0.5 * uy * uy)
        z[np.ix_(rows, cols)] += (ay * pw[None, :]) @ ax.T
    return z / (w.sum() * bx * by)


def bias_surface(points, env, *, bandwidth="nrd", isotropic: bool = False, limits=None, weights=None,
                 rescale=None, mask: bool = True, x: str | None = None, y: str | None = None, crs=None):
    """Sampling effort as a surface: a Gaussian kernel density of the records of a group.

    Records pile up where people look for species, near roads, towns and
    research stations, and a presence-background model fitted to them
    learns that bias along with the species' preferences. Drawing the
    background with the same bias cancels it: the "target-group" background
    of Phillips et al. (2009), whose effort is estimated from the records of
    every species sampled the same way (the same taxonomic group, the same
    collectors and decades). The density of those records, smoothed by a
    Gaussian kernel, is the bias grid of Fourcade et al. (2014) and
    Kramer-Schadt et al. (2013), the ``bias`` of :func:`background` (and
    Maxent's "bias file").

    The density is ``MASS::kde2d``'s, evaluated at the centres of the cells:
    a product of two normal kernels whose standard deviations are
    ``bandwidth`` in x and in y. Note that ``kde2d(h = H)`` divides ``H`` by
    4: it is ``bandwidth = H / 4`` here.

    Parameters
    ----------
    points
        The records of the target group (GeoDataFrame of points, or a table
        with coordinate columns); reprojected to the CRS of ``env``.
    env
        The grid (a raster or a stack of predictors).
    bandwidth
        Standard deviation of the kernel, in the units of the CRS of ``env``
        (metres on projected grids, degrees on longitude / latitude ones): a
        number, a pair ``(x, y)`` or ``"nrd"``, the normal reference rule of
        R's ``bw.nrd`` in each direction (``kde2d``'s default bandwidth).
    isotropic
        The same bandwidth in both directions, the larger of the two.
    limits
        ``(min, max)``: bandwidths below / above them are raised / lowered to
        them (either may be None). A floor keeps the surface of a clustered
        group from collapsing onto its records; a ceiling keeps a widespread
        one from flattening.
    weights
        Non-negative weight of each record (e.g. the number of specimens or
        visits); default 1.
    rescale
        ``(lo, hi)``: map the densities linearly to ``[lo, hi]`` over the cells
        kept, e.g. ``(0.05, 0.95)`` so that every cell keeps some chance of
        being drawn. Default: the densities themselves (records per unit of
        area of the CRS).
    mask
        NaN on the cells without a value in every band of ``env`` (as
        :func:`background` reads them); False: every cell of the grid.
    x, y, crs
        Coordinate columns and CRS of a table of records.

    Returns
    -------
    xarray.DataArray
        ``(y, x)`` on the grid of ``env``, named ``bias``; ``attrs`` hold the
        bandwidths used (``bandwidth``) and the number of records (``n``).

    Notes
    -----
    Records without coordinates are left out; records outside the grid
    count (their kernels reach into it), as in ``kde2d``. The sum runs over
    the cells within 40 bandwidths of each record, beyond which the normal
    density is 0 in double precision: the result is ``kde2d``'s up to
    rounding.

    References
    ----------
    Phillips, S. J. et al. (2009). Sample selection bias and presence-only
    distribution models: implications for background and pseudo-absence
    data. *Ecological Applications* 19: 181-197.

    Fourcade, Y. et al. (2014). Mapping species distributions with MAXENT
    using a geographically biased sample of presence data: a performance
    assessment of methods for correcting sampling bias. *PLoS ONE* 9:
    e97122.

    Examples
    --------
    >>> import landschaft as ls
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> group = ls.load_table(ls.sdm.example_data("bradypus"))   # stands for the records of a whole group
    >>> bias = ls.sdm.bias_surface(group, env, bandwidth=2.0, rescale=(0.05, 0.95))
    >>> float(bias.min()), float(bias.max())
    (0.05, 0.95)
    >>> bg = ls.sdm.background(env, 1000, bias=bias, seed=1)
    """
    if not isinstance(env, xr.DataArray):
        raise TypeError("env must be a raster (DataArray)")
    if env.ndim == 2:
        env = env.expand_dims(band=[env.name or "layer"])
    xs, ys, src = _points.coordinates(points, x=x, y=y, crs=crs)
    xs, ys = _points.to_crs(xs, ys, src, env.rio.crs)
    w = np.ones(xs.size) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if w.size != xs.size:
        raise ValueError(f"weights must have one value per record ({xs.size})")
    if not (np.isfinite(w).all() and (w >= 0).all()):
        raise ValueError("weights must be finite and non-negative")
    ok = np.isfinite(xs) & np.isfinite(ys) & (w > 0)
    xs, ys, w = xs[ok], ys[ok], w[ok]
    if xs.size == 0:
        raise ValueError("no record with coordinates")
    bx, by = _bandwidths(bandwidth, isotropic, limits, xs, ys)

    xmin, xmax, ymin, ymax, xres, yres, ny, nx = _grid(env)
    gx = xmin + (np.arange(nx) + 0.5) * xres
    gy = ymax - (np.arange(ny) + 0.5) * yres
    z = _density(gx, gy, xs, ys, w, bx, by)
    if mask:
        z[~_complete(env).reshape(ny, nx)] = np.nan
    if rescale is not None:
        lo, hi = (float(v) for v in rescale)
        if not lo < hi:
            raise ValueError("rescale must be (lo, hi) with lo < hi")
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        if not zmax > zmin:
            raise ValueError("the density is the same in every cell (records far from the grid, or a bandwidth "
                             "too large): nothing to rescale")
        z = np.clip(lo + (z - zmin) / (zmax - zmin) * (hi - lo), lo, hi)  # lo and hi exactly at the extremes
    template = env.isel(band=0, drop=True)
    out = xr.DataArray(z, dims=template.dims, coords=template.coords, name="bias")
    out = out.rio.write_crs(env.rio.crs) if env.rio.crs is not None else out
    out = out.rio.write_nodata(np.nan)
    out.attrs.update({"bandwidth": (bx, by), "n": int(xs.size)})
    return out
