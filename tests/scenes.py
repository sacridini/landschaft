# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic synthetic MSPA inputs (0 = missing, 1 = background, 2 = foreground)."""

from __future__ import annotations

import numpy as np


def _smooth_noise(rng: np.random.Generator, shape, scale: int) -> np.ndarray:
    """Cheap smooth random field: upsampled coarse noise, box-blurred."""
    ny, nx = shape
    coarse = rng.random((ny // scale + 2, nx // scale + 2))
    field = np.kron(coarse, np.ones((scale, scale)))[:ny, :nx]
    k = max(1, scale // 2)
    for axis in (0, 1):
        field = sum(np.roll(field, s, axis=axis) for s in range(-k, k + 1)) / (2 * k + 1)
    return field


def blobs(seed: int, shape=(240, 320), scale=12, cover=0.5, missing=0.08) -> np.ndarray:
    """Patchy foreground with holes, plus some missing-data patches."""
    rng = np.random.default_rng(seed)
    field = _smooth_noise(rng, shape, scale)
    out = np.where(field > np.quantile(field, 1 - cover), 2, 1).astype(np.uint8)
    if missing:
        miss = _smooth_noise(rng, shape, scale * 2)
        out[miss > np.quantile(miss, 1 - missing)] = 0
    return out


def salt(seed: int, shape=(150, 170), p=0.55, missing=0.02) -> np.ndarray:
    """Pixel noise: many islets, branches, bridges and loops."""
    rng = np.random.default_rng(seed)
    out = np.where(rng.random(shape) < p, 2, 1).astype(np.uint8)
    out[rng.random(shape) < missing] = 0
    return out


def network(seed: int, shape=(200, 260), n_patches=14, n_links=24) -> np.ndarray:
    """Square patches connected by 1-3 pixel wide corridors (bridges/loops)."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    out = np.ones(shape, dtype=np.uint8)
    centers = []
    for _ in range(n_patches):
        h, w = rng.integers(8, 30, size=2)
        y, x = rng.integers(0, ny - h), rng.integers(0, nx - w)
        out[y : y + h, x : x + w] = 2
        if h > 14 and w > 14 and rng.random() < 0.5:  # perforation
            out[y + h // 2 - 2 : y + h // 2 + 2, x + w // 2 - 2 : x + w // 2 + 2] = 1
        centers.append((y + h // 2, x + w // 2))
    for _ in range(n_links):
        (y0, x0), (y1, x1) = (centers[i] for i in rng.choice(len(centers), 2, replace=False))
        width = int(rng.integers(1, 4))
        n = max(abs(y1 - y0), abs(x1 - x0)) + 1
        ys = np.linspace(y0, y1, n).round().astype(int)
        xs = np.linspace(x0, x1, n).round().astype(int)
        for d in range(width):
            out[np.clip(ys + d, 0, ny - 1), xs] = 2
    return out


def border_touching(seed: int, shape=(120, 90)) -> np.ndarray:
    """Foreground cut by the image border and by missing data."""
    rng = np.random.default_rng(seed)
    out = blobs(seed, shape, scale=8, cover=0.6, missing=0.0)
    out[:, :3] = 2
    out[-2:, :] = 2
    out[: shape[0] // 3, -1] = 0
    out[rng.integers(0, shape[0], 40), rng.integers(0, shape[1], 40)] = 0
    return out


def tiny(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = tuple(int(v) for v in rng.integers(2, 12, size=2))
    out = np.where(rng.random(shape) < 0.6, 2, 1).astype(np.uint8)
    out.flat[0] = 1
    out.flat[-1] = 2
    return out


SCENES = {
    "blobs": blobs,
    "salt": salt,
    "network": network,
    "border": border_touching,
    "tiny": tiny,
}


def make(kind: str, seed: int) -> np.ndarray:
    return SCENES[kind](seed)
