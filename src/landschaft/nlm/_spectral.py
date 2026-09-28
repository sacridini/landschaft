# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectral neutral landscape models: Gaussian random fields and fractional
Brownian motion (NLMR 1.4's reimplementations without RandomFields)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .. import _core, _threads
from ._common import check_resolution, check_size, grid, resolve_seed

__all__ = ["fbm", "gaussian_field"]


def gaussian_field(ncol: int, nrow: int, resolution: float = 1.0, autocorr_range: float = 10, mag_var: float = 5,
                   nug: float = 0.2, mean: float = 0.5, seed: int | None = None, rescale: bool = True,
                   *, n_threads: int | None = None) -> xr.DataArray:
    """Gaussian random field neutral landscape model (NLMR's ``nlm_gaussianfield``).

    A stationary Gaussian field with an exponential covariance
    ``mag_var * exp(-d / autocorr_range)``, simulated exactly by circulant
    embedding (FFT on a grid twice as large in each direction), plus a nugget
    (white noise of variance ``nug``) and a mean.

    NLMR 1.4 disabled ``nlm_gaussianfield`` when RandomFields left CRAN and ships
    this method as its internal ``simulate_gaussian_field``; the results here are
    that function's, rescaled as NLMR's previous ``nlm_gaussianfield`` did.

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    resolution : float, default 1
        Cell size.
    autocorr_range : float, default 10
        Range of the spatial autocorrelation, in cells.
    mag_var : float, default 5
        Variance of the autocorrelated part.
    nug : float, default 0.2
        Variance of the nugget (uncorrelated noise); 0 for none.
    mean : float, default 0.5
        Mean of the field.
    seed : int, optional
        Seed of R's generator: the same seed gives NLMR's field. ``None`` draws
        one, stored in ``attrs["seed"]``.
    rescale : bool, default True
        Rescale the values to 0-1.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 values.

    Examples
    --------
    >>> import landschaft as ls
    >>> g = ls.nlm.gaussian_field(80, 60, autocorr_range=15, seed=1)
    >>> g.shape, float(g.min()), float(g.max())
    ((60, 80), 0.0, 1.0)
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    autocorr_range, mag_var, nug = float(autocorr_range), float(mag_var), float(nug)
    if not autocorr_range > 0:
        raise ValueError("autocorr_range must be positive")
    if mag_var < 0 or nug < 0:
        raise ValueError("mag_var and nug must be non-negative")
    s = resolve_seed(seed)
    with _threads.limit(n_threads):
        values = _core.nlm_gaussian_field(s, nrow, ncol, autocorr_range, mag_var, nug, float(mean))
        if rescale:
            values = _core.nlm_rescale(values)
    return grid(values, resolution, "gaussian_field",
                {"seed": s, "model": "gaussian_field", "autocorr_range": autocorr_range, "mag_var": mag_var,
                 "nug": nug})


def fbm(ncol: int, nrow: int, resolution: float = 1.0, fract_dim: float = 1.0, seed: int | None = None,
        rescale: bool = True, *, n_threads: int | None = None) -> xr.DataArray:
    """Fractional Brownian motion neutral landscape model (NLMR's ``nlm_fbm``).

    A fractal surface from a power-law spectrum ``|k|^-(fract_dim + 1)`` with
    random phases (inverse FFT), centred and scaled to unit standard deviation
    of the steps between neighbouring cells. ``fract_dim`` is the Hurst-like
    exponent alpha of NLMR (0 < alpha <= 2): small values give rough surfaces,
    large ones smooth ones; alpha = 2 is a random plane.

    NLMR 1.4 disabled ``nlm_fbm`` when RandomFields left CRAN and ships this
    method as its internal ``simulate_fbm``; the results here are that
    function's, rescaled as NLMR's previous ``nlm_fbm`` did.

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    resolution : float, default 1
        Cell size.
    fract_dim : float, default 1
        The exponent alpha, in (0, 2].
    seed : int, optional
        Seed of R's generator: the same seed gives NLMR's surface. ``None`` draws
        one, stored in ``attrs["seed"]``.
    rescale : bool, default True
        Rescale the values to 0-1.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray
        float64 values.

    Examples
    --------
    >>> import landschaft as ls
    >>> f = ls.nlm.fbm(64, 64, fract_dim=1.5, seed=2)
    >>> f.shape, float(f.max())
    ((64, 64), 1.0)
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    fract_dim = float(fract_dim)
    if not 0 < fract_dim <= 2:
        raise ValueError(f"fract_dim must be in (0, 2], got {fract_dim!r}")
    s = resolve_seed(seed)
    with _threads.limit(n_threads):
        values = _core.nlm_fbm(s, nrow, ncol, fract_dim)
        if rescale:
            values = _core.nlm_rescale(values)
    return grid(values, resolution, "fbm", {"seed": s, "model": "fbm", "fract_dim": fract_dim})
