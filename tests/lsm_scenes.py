# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic categorical landscapes for the landscape-metric tests."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def clustered(shape, classes, *, seed: int, sigma: float = 2.0, na_fraction: float = 0.0) -> np.ma.MaskedArray:
    """Smoothed noise cut into ``classes`` (values used as given), with NA blobs."""
    rng = np.random.default_rng(seed)
    field = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    edges = np.quantile(field, np.linspace(0, 1, len(classes) + 1)[1:-1])
    out = np.asarray(classes)[np.digitize(field, edges)]
    mask = np.zeros(shape, dtype=bool)
    if na_fraction > 0:
        holes = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
        mask = holes > np.quantile(holes, 1 - na_fraction)
    return np.ma.MaskedArray(out.astype(np.int32), mask=mask)


def salt_and_pepper(shape, classes, *, seed: int, na_fraction: float = 0.0) -> np.ma.MaskedArray:
    """Independent random cells: many one-cell patches, diagonal contacts."""
    rng = np.random.default_rng(seed)
    out = rng.choice(np.asarray(classes), size=shape).astype(np.int32)
    mask = rng.random(shape) < na_fraction
    return np.ma.MaskedArray(out, mask=mask)


def small_cases() -> dict[str, np.ma.MaskedArray]:
    """Degenerate shapes and class layouts."""
    checker = (np.indices((8, 8)).sum(axis=0) % 2 + 1).astype(np.int32)
    stripes = np.repeat(np.array([[1, 1, 2, 2, 3, 3, 1]], dtype=np.int32), 5, axis=0)
    ring = np.full((9, 9), 4, dtype=np.int32)
    ring[2:7, 2:7] = 2
    ring[4, 4] = 4
    na_ring = np.ma.MaskedArray(ring.copy(), mask=np.zeros_like(ring, dtype=bool))
    na_ring[0, :] = np.ma.masked
    na_ring[3:6, 3:6] = np.ma.masked
    return {
        "single_class": np.ma.MaskedArray(np.full((6, 5), 3, dtype=np.int32)),
        "two_class": np.ma.MaskedArray(np.where(np.arange(35).reshape(5, 7) % 3 == 0, 1, 2).astype(np.int32)),
        "checker": np.ma.MaskedArray(checker),
        "stripes": np.ma.MaskedArray(stripes),
        "row": np.ma.MaskedArray(np.array([[1, 1, 2, 3, 3, 3, 2, 1, 1, 1, 2, 2, 3, 1, 1]], dtype=np.int32)),
        "column": np.ma.MaskedArray(np.array([[5], [5], [6], [6], [6], [5], [7], [7], [5], [5], [6], [7]], dtype=np.int32)),
        "na_ring": na_ring,
    }
