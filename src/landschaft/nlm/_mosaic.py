# SPDX-License-Identifier: GPL-3.0-or-later
"""Mosaic neutral landscape models: Poisson line mosaics (NLMR's
``nlm_mosaicfield``, spatstat's ``rMosaicField``)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .. import _core, _threads
from ._common import check_resolution, check_size, grid, resolve_seed

__all__ = ["mosaic_field"]


def _rescale_layers(layers: list[np.ndarray], nlmr_compat: bool) -> list[np.ndarray]:
    if not layers:
        return layers
    if not nlmr_compat:
        return [_core.nlm_rescale(v) for v in layers]
    # util_rescale() of a multi-layer raster uses the first layer's minimum
    # and maximum for every layer
    first = layers[0]
    lo, hi = np.nanmin(first), np.nanmax(first)
    probe = _core.nlm_rescale(np.array([lo, hi]))
    if probe[1] == 0:  # all.equal(min, max): every cell 0
        return [np.zeros_like(v) for v in layers]
    return [(v - lo) / (hi - lo) for v in layers]


def mosaic_field(ncol: int, nrow: int, resolution: float = 1.0, n: int | None = 20, mosaic_mean: float = 0.5,
                 mosaic_sd: float = 0.5, seed: int | None = None, collect: bool = False, infinit: bool = False,
                 rescale: bool = True, nlmr_compat: bool = True, *, n_threads: int | None = None):
    """Mosaic random field neutral landscape model (NLMR's ``nlm_mosaicfield``, Schlather 1999).

    The sum of ``n`` mosaic fields: each splits the landscape by a Poisson line
    tessellation (intensity 4 on the unit square, spatstat's ``rpoislinetess``)
    and gives every tile a normal value (``rMosaicField``). As ``n`` grows the sum
    tends to a Gaussian random field; ``infinit=True`` also returns that limit, a
    Gaussian field with exponential covariance (the log intensity of spatstat's
    ``rLGCP("exp", 4, var = 1, scale = 0.2)``).

    Parameters
    ----------
    ncol, nrow : int
        Columns and rows.
    resolution : float, default 1
        Cell size.
    n : int or None, default 20
        Number of mosaic fields summed; ``None`` for only the ``infinit`` field.
        With ``nlmr_compat=True``, ``n = 1`` sums three fields as NLMR does
        (``for (i in 2:n)`` runs over ``c(2, 1)``).
    mosaic_mean, mosaic_sd : float, default 0.5
        Mean and standard deviation of the tile values.
    seed : int, optional
        Seed of R's generator (NLMR's ``user_seed``): the same seed gives NLMR's
        landscape. ``None`` draws one, stored in ``attrs["seed"]``.
    collect : bool, default False
        Also return every field, the i-th divided by sqrt(i), as NLMR's ``steps``.
    infinit : bool, default False
        Also return the limiting Gaussian field.
    rescale : bool, default True
        Rescale to 0-1 (the ``steps`` together, with the first step's range as
        NLMR does, unless ``nlmr_compat=False``).
    nlmr_compat : bool, default True
        Reproduce NLMR's quirks described above.
    n_threads : int, optional
        Threads (default: all cores but one).

    Returns
    -------
    xarray.DataArray or dict
        The summed field; with ``collect`` or ``infinit`` a dict with the
        entries ``"mosaic_field"``, ``"steps"`` (dims ``step, y, x``) and
        ``"infinit"`` that were asked for (a single one is returned as is).

    Examples
    --------
    >>> import landschaft as ls
    >>> m = ls.nlm.mosaic_field(60, 40, n=10, seed=1)
    >>> m.shape, float(m.min()), float(m.max())
    ((40, 60), 0.0, 1.0)
    >>> out = ls.nlm.mosaic_field(30, 20, n=4, collect=True, seed=2)
    >>> sorted(out), out["steps"].shape
    (['mosaic_field', 'steps'], (4, 20, 30))
    """
    ncol, nrow = check_size(ncol, nrow)
    resolution = check_resolution(resolution)
    if n is None:
        if not infinit:
            raise ValueError("n = None needs infinit=True")
        nn = 0
    else:
        nn = int(n)
        if nn < 1:
            raise ValueError("n must be a positive integer or None")
    s = resolve_seed(seed)
    with _threads.limit(n_threads):
        raw = _core.nlm_mosaic_field(s, nrow, ncol, nn, bool(nlmr_compat), float(mosaic_mean), float(mosaic_sd),
                                     bool(collect) and nn > 0, bool(infinit))
        out = {}
        attrs = {"seed": s, "model": "mosaic_field", "nlmr_compat": int(bool(nlmr_compat))}
        if "sum" in raw:
            v = _core.nlm_rescale(raw["sum"]) if rescale else raw["sum"]
            out["mosaic_field"] = grid(v, resolution, "mosaic_field", attrs)
        if "steps" in raw:
            layers = list(raw["steps"])
            if rescale:
                layers = _rescale_layers(layers, nlmr_compat)
            base = grid(layers[0], resolution, "steps", attrs)
            steps = xr.concat([base.copy(data=layer) for layer in layers], dim="step")
            out["steps"] = steps.assign_coords(step=np.arange(1, len(layers) + 1))
        if "infinit" in raw:
            v = _core.nlm_rescale(raw["infinit"]) if rescale else raw["infinit"]
            out["infinit"] = grid(v, resolution, "mosaic_field_infinit", attrs)
    if len(out) == 1:
        return next(iter(out.values()))
    return out
