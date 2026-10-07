# SPDX-License-Identifier: GPL-3.0-or-later
"""The original vegetation of converted land, and the vegetation restoration brings back.

After plangea's ``harmonize_oa`` (International Institute for Sustainability; GPL >= 3), the
preparation of the restoration model of Strassburg et al. (2020): the anthropic share of every
pixel is given back to the natural classes in the proportions they had there in the past; where
the past has no natural cover, or the pixel is (nearly) all converted, in the proportions of its
ecoregion; where that is missing too, in the classes' mean proportions.

References
----------
Strassburg, B. B. N. et al. (2020). Global priority areas for ecosystem restoration. *Nature*
    586, 724-729. doi:10.1038/s41586-020-2784-9
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from . import _core, _threads
from ._layers import like, named_layers
from ._save import Saveable

__all__ = ["OriginalVegetation", "original_vegetation", "VEGETATION_SOURCES"]

#: The codes of :attr:`OriginalVegetation.source`.
VEGETATION_SOURCES = {0: "past cover", 1: "ecoregion", 2: "failsafe"}


@dataclass(frozen=True)
class OriginalVegetation(Saveable):
    """Result of :func:`original_vegetation`.

    Attributes
    ----------
    original
        The share of every natural class in each pixel's original cover
        (``(band, y, x)``, one band per natural class): its current share
        plus the part of the anthropic share that was that class. NaN outside
        the land (no natural or anthropic cover).
    restore
        The shares of the natural classes that restored land in each pixel
        becomes (they sum to 1).
    source
        Where each pixel's proportions come from: 0 its past cover, 1 its
        ecoregion, 2 the failsafe (see :data:`VEGETATION_SOURCES`).
    anthropic
        The anthropic share of each pixel: the land restoration can work on.
    failsafe
        The failsafe proportions of the natural classes.
    """

    original: xr.DataArray = field(repr=False)
    restore: xr.DataArray = field(repr=False)
    source: xr.DataArray = field(repr=False)
    anthropic: xr.DataArray = field(repr=False)
    failsafe: dict = field(default_factory=dict)

    @property
    def classes(self) -> list[str]:
        """The natural classes, in the order of the bands."""
        return [str(c) for c in self.original["band"].values]


def _by_name(names: list[str], values: np.ndarray, wanted: list[str], what: str) -> np.ndarray:
    pos = {n: i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in pos]
    if missing:
        raise ValueError(f"{what}: no layer for the class(es) {missing} (layers: {names})")
    return values[[pos[w] for w in wanted]]


def original_vegetation(
    land_cover,
    natural: Iterable[str],
    anthropic: Iterable[str],
    *,
    past=None,
    ecoregions=None,
    failsafe: Iterable[str] | None = None,
    threshold: float = 0.95,
    restore_to: str = "original",
    n_threads: int | None = None,
) -> OriginalVegetation:
    """The original vegetation of converted land, and what restoration brings back.

    Every pixel holds shares of land-cover classes: natural ones (forest,
    grassland, wetland...), anthropic ones (crops, pasture...) and others
    that are neither (water, rock). Restoring a pixel's anthropic share
    brings back natural vegetation; this function says which, as plangea
    prepares it for its restoration model (Strassburg et al. 2020):

    1. the anthropic share is split among the natural classes in the
       proportions they had in the pixel in the ``past``;
    2. where the past has no natural cover, or more than ``threshold`` of
       the pixel is anthropic today, in the proportions of the pixel's
       ``ecoregions`` (equal proportions without them);
    3. where those are all zero too, in the mean proportions of the
       ``failsafe`` classes over the pixels with anthropic land (over all
       the land with ``restore_to="current"``).

    The original cover of a class is then its current share plus its part
    of the anthropic share.

    Parameters
    ----------
    land_cover
        The current share (0 to 1) of every class in each pixel: a
        ``(band, y, x)`` raster whose band names are the classes, a mapping
        class -> raster, a list of named rasters or files. NaN counts as 0.
    natural, anthropic
        The natural and the anthropic classes. The other layers are ignored.
    past
        The past share of every natural class (same forms; NaN counts as 0).
        By default the current shares.
    ecoregions
        For every natural class, its proportion in each pixel's ecoregion
        (same forms). By default equal proportions.
    failsafe
        The classes the failsafe fills (default: every natural class).
    threshold
        Above this anthropic share, the ecoregion's proportions are used
        even where there is a past cover.
    restore_to
        ``"original"``: restored land becomes the original cover's natural
        classes in their proportions (restoration alone); ``"current"``: the
        current natural classes of the pixel where it has some, else the
        original ones (plangea's restoration with conversion).
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    OriginalVegetation
        The original cover, the restored proportions, the source of each
        pixel's proportions and the anthropic share.

    Notes
    -----
    The arithmetic is plangea's, bit for bit (``harmonize_oa``), with two
    corrections: without ecoregion maps, plangea gives every natural class
    the pixel's total natural share divided by the number of classes instead
    of equal proportions (its own comment says "flat among natural types"),
    which leaves most of a converted pixel to no class; and pixels the
    failsafe fills get 0 for the classes it does not fill, where plangea can
    stop with NA.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> cover = xr.DataArray(np.array([[[0.6, 0.0]], [[0.2, 0.1]], [[0.2, 0.9]]]),
    ...                      dims=("band", "y", "x"), coords={"band": ["forest", "savanna", "crops"]})
    >>> past = xr.DataArray(np.array([[[0.9, 0.0]], [[0.1, 0.0]]]),
    ...                     dims=("band", "y", "x"), coords={"band": ["forest", "savanna"]})
    >>> veg = ls.original_vegetation(cover, ["forest", "savanna"], ["crops"], past=past)
    >>> veg.original.sel(band="forest").values.round(3)
    array([[0.78, 0.45]])
    >>> veg.source.values
    array([[0., 1.]])
    """
    if restore_to not in ("original", "current"):
        raise ValueError(f"restore_to must be 'original' or 'current', got {restore_to!r}")
    natural, anthropic = [str(c) for c in natural], [str(c) for c in anthropic]
    if not natural or not anthropic:
        raise ValueError("at least one natural and one anthropic class are needed")
    if set(natural) & set(anthropic):
        raise ValueError(f"classes both natural and anthropic: {sorted(set(natural) & set(anthropic))}")
    names, values, template = named_layers(land_cover, "land_cover")
    values = np.nan_to_num(values, nan=0.0)
    shape = values.shape[1:]
    nat = _by_name(names, values, natural, "land_cover")
    ant = _by_name(names, values, anthropic, "land_cover")
    # the land (plangea's terrestrial pixels): natural or anthropic cover, the classes in their order
    order = [n for n in names if n in natural or n in anthropic]
    land = np.zeros(shape)
    for i, n in enumerate(order):
        land = values[names.index(n)] if i == 0 else land + values[names.index(n)]
    cells = np.flatnonzero(land.ravel() > 0)

    def flat(stack: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(stack.reshape(stack.shape[0], -1)[:, cells])

    def optional(data, what):
        if data is None:
            return None
        n2, v2, _ = named_layers(data, what)
        if v2.shape[1:] != shape:
            raise ValueError(f"{what}: grid {v2.shape[1:]}, land_cover {shape}")
        return flat(np.nan_to_num(_by_name(n2, v2, natural, what), nan=0.0))

    fill_classes = natural if failsafe is None else [str(c) for c in failsafe]
    bad = [c for c in fill_classes if c not in natural]
    if bad:
        raise ValueError(f"failsafe: {bad} are not natural classes")
    fill = np.array([c in fill_classes for c in natural], dtype=np.uint8)
    nat_f, ant_f = flat(nat), flat(ant)
    a_total = ant_f[0].copy()
    for row in ant_f[1:]:
        a_total = a_total + row
    # plangea's master_index: the pixels with anthropic land (restoration), all the land (restoration with
    # conversion): the failsafe means are over them
    domain = None if restore_to == "current" else (a_total > 0).astype(np.uint8)
    orig, restore, source, shares = _core.rest_original_vegetation(
        nat_f, ant_f, optional(past, "past"), optional(ecoregions, "ecoregions"), domain, fill, float(threshold),
        restore_to == "current", _threads.resolve(n_threads))

    def grid(v: np.ndarray) -> np.ndarray:
        out = np.full((v.shape[0], shape[0] * shape[1]) if v.ndim == 2 else shape[0] * shape[1], np.nan)
        if v.ndim == 2:
            out[:, cells] = v
            return out.reshape((v.shape[0],) + shape)
        out[cells] = v
        return out.reshape(shape)

    return OriginalVegetation(
        original=like(template, grid(orig), "original_vegetation", natural),
        restore=like(template, grid(restore), "restored_vegetation", natural),
        source=like(template, grid(source.astype(np.float64)), "source"),
        anthropic=like(template, grid(a_total), "anthropic_share"),
        failsafe={c: float(v) for c, v, f in zip(natural, shares, fill) if f},
    )
