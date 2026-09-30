# SPDX-License-Identifier: GPL-3.0-or-later
"""Zonation 5's connectivity transforms (connectivity.cpp, convolver.cpp): smoothing of the
feature layers with a dispersal kernel, interactions between features and connectivity
matrices, by FFT convolution in blocks as Zonation computes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import _core

MIN_FILTER = 128  # Zonation's Z5_FILTER_MIN_SIZE
SHAPES = ("uniform", "triangular", "negexp", "gaussian")


@dataclass(frozen=True)
class Kernel:
    """A dispersal kernel of Zonation 5's connectivity transforms.

    Parameters
    ----------
    shape
        ``"uniform"`` (1 within ``distance``), ``"triangular"`` (1 - d / distance
        within ``distance``), ``"negexp"`` (exp(-2 d / distance): ``distance`` is the
        mean dispersal distance) or ``"gaussian"`` (normal density of standard
        deviation ``distance``).
    distance
        In cells.
    cut
        ``negexp`` and ``gaussian``: the kernel is 0 beyond ``distance * cut``.

    Examples
    --------
    >>> import landschaft as ls
    >>> k = ls.prioritize.Kernel("negexp", 10, cut=3)
    >>> k.image().shape       # the kernel image Zonation convolves with
    (128, 128)
    """

    shape: str
    distance: float
    cut: float = 1.0

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f"unknown kernel {self.shape!r}; one of {', '.join(SHAPES)}")
        if not self.distance > 0:
            raise ValueError("the kernel distance must be > 0")

    def size(self, max_distance: int = 2**31 - 1) -> int:
        """The side of the kernel image (getKernel's getFilterSize)."""
        d = float(np.float32(self.distance))
        if self.shape in ("negexp", "gaussian"):
            d = float(np.float32(np.float32(self.distance) * np.float32(self.cut)))
        d = min(int(d), int(max_distance))  # the float distance converted to uint32
        return max(MIN_FILTER, int(2 * 2 ** math.ceil(math.log2(d + 1))))

    def image(self, max_distance: int = 2**31 - 1) -> np.ndarray:
        """The kernel image (float64, not normalized)."""
        n = self.size(max_distance)
        d = float(np.float32(self.distance))
        c = float(np.float32(np.float32(self.distance) * np.float32(self.cut)))
        off = np.arange(n, dtype=np.float64) - n * 0.5
        dist = np.sqrt(off[:, None] * off[:, None] + off[None, :] * off[None, :])
        if self.shape == "uniform":
            return np.where(dist <= d, 1.0, 0.0)
        if self.shape == "triangular":
            return np.where(dist <= d, 1.0 - dist / d, 0.0)
        if self.shape == "negexp":
            return np.where(dist <= c, np.exp(-2.0 * dist / d), 0.0)
        s2 = d * d
        return np.where(dist <= c, np.exp(-(dist * dist / (2.0 * s2))) / (2.0 * math.pi * s2), 0.0)


@dataclass(frozen=True)
class Interaction:
    """An interaction connectivity transform (Zonation's ``-i``/``-I``): a feature
    weighted by the connectivity of a resource (``positive``) or reduced by it.

    Parameters
    ----------
    kernel
        Dispersal kernel from the resource.
    resource
        Raster of the resource (path, DataArray or array).
    positive
        ``True``: the feature is multiplied by the smoothed resource; ``False``:
        ``beta`` times the feature's share of the product is removed.
    beta
        Strength of the negative interaction.
    normalize
        Negative interactions: keep the feature's level (``False``: its level
        becomes what remains).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> rng = np.random.default_rng(0)
    >>> species = rng.random((2, 30, 40))
    >>> food = rng.random((30, 40))          # a resource of species 0
    >>> ia = ls.prioritize.Interaction(ls.prioritize.Kernel("negexp", 3, 3), food)
    >>> res = ls.prioritize.zonation(species, interaction=[ia, None])
    >>> res.iterations > 0
    True
    """

    kernel: Kernel
    resource: object
    positive: bool = True
    beta: float = 1.0
    normalize: bool = True


def _filter_ft(kernel: Kernel, max_distance: int, shape2) -> np.ndarray:
    from scipy import fft

    img = kernel.image(max_distance)
    total = _core.prio_eigen_sum_double(np.ascontiguousarray(img.ravel()))
    buf = np.zeros(shape2, dtype=np.float64)
    n = img.shape[0]
    buf[:n, :n] = img / total
    return fft.rfft2(buf)


def convolve(image: np.ndarray, kernel: Kernel, workers: int = 1) -> np.ndarray:
    """Convolve a raster with a kernel normalized to sum 1, as Zonation 5 does.

    FFT convolution in blocks of the kernel's size, the blocks' results added as
    float32 in Zonation's order: the smoothing of its connectivity transforms.

    Parameters
    ----------
    image
        2-D array (rows, columns); cells outside the data should be 0.
    kernel
        A :class:`Kernel`.
    workers
        FFT threads.

    Returns
    -------
    numpy.ndarray
        float32, the image's shape.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> img = np.zeros((50, 60))
    >>> img[25, 30] = 1.0
    >>> out = ls.prioritize.convolve(img, ls.prioritize.Kernel("uniform", 2))
    >>> round(float(out.sum()), 5), int((out > 1e-6).sum())   # mass kept, spread over the disc
    (1.0, 13)
    """
    from scipy import fft

    H, W = image.shape
    n = kernel.size(max(H, W))
    s = (2 * n, 2 * n)
    F = _filter_ft(kernel, max(H, W), s)
    result = np.zeros((H, W), dtype=np.float32)
    blocks = [(by, bx) for by in range(0, H, n) for bx in range(0, W, n)]
    batch = max(1, (64 * 2**20) // (s[0] * s[1] * 16))
    for i in range(0, len(blocks), batch):
        part = []
        for by, bx in blocks[i:i + batch]:
            blk = image[by:by + n, bx:bx + n]
            if blk.any():
                part.append((by, bx, blk))
        if not part:
            continue
        buf = np.zeros((len(part),) + s, dtype=np.float64)
        for k, (_, _, blk) in enumerate(part):
            buf[k, :blk.shape[0], :blk.shape[1]] = blk
        out = fft.irfft2(fft.rfft2(buf, axes=(1, 2), workers=workers) * F[None], s=s, axes=(1, 2),
                         workers=workers)
        for k, (by, bx, _) in enumerate(part):
            y1, x1 = by - n // 2, bx - n // 2
            y2, x2 = y1 + s[0], x1 + s[1]
            oy, ox = max(0, -y1), max(0, -x1)
            y1, x1 = max(y1, 0), max(x1, 0)
            y2, x2 = min(y2, H), min(x2, W)
            if y2 > y1 and x2 > x1:
                result[y1:y2, x1:x2] += out[k, oy:oy + y2 - y1, ox:ox + x2 - x1].astype(np.float32)
    return result


def _dense(values: np.ndarray, cells: np.ndarray, H: int, W: int, dtype=np.float64) -> np.ndarray:
    out = np.zeros(H * W, dtype=dtype)
    out[cells] = values
    return out.reshape(H, W)


def apply(data, kept, cells, grid, *, connectivity=None, self_multiply=None, interaction=None,
          interaction2=None, matrix_connectivity=None, matrices=None, open_aux=None, gather=None,
          workers: int = 1) -> None:
    """Zonation's connectivity transforms of the quantized layers, in its order:
    single feature, interaction 1, matrix, interaction 2."""
    H, W = grid.shape
    L = len(kept)
    storage_r = np.float32(1.0 / (255.0 if data.bits == 8 else 65535.0))

    def spec(values, i):
        return None if values is None else values[i]

    def q_of(k):
        return np.asarray(data.layer(k, False, -1.0))

    def replace(k, packed, level):
        data.replace_layer(k, np.ascontiguousarray(packed, dtype=np.float32), float(level))

    # single-feature connectivity
    for k, i in enumerate(kept):
        kern = spec(connectivity, i)
        if kern is None:
            continue
        q = q_of(k)
        present = q >= 0
        img = _dense(np.where(present, q, 0.0), cells, H, W)
        res = convolve(img, kern, workers).ravel()[cells]
        if spec(self_multiply, i):
            res = (res * (np.where(present, q, 0.0).astype(np.float32) * storage_r)).astype(np.float32)
        res[~present] = 0.0
        replace(k, res, np.asarray(data.initial_levels)[k])

    def interact(specs):
        for k, i in enumerate(kept):
            ia = spec(specs, i)
            if ia is None:
                continue
            resource = gather(open_aux(ia.resource), cells).astype(np.float32)
            img = _dense(resource, cells, H, W)
            dsb = convolve(img, ia.kernel, workers).ravel()[cells]
            q = q_of(k)
            present = q >= 0
            dsb[~present] = 0.0
            scale = np.asarray(data.layer_scale)[k]
            a = np.where(present, (q.astype(np.float32).astype(np.float64) * scale).astype(np.float32),
                         np.float32(0)).astype(np.float32)
            level = np.asarray(data.initial_levels)[k]
            if ia.positive:
                a = (a * dsb).astype(np.float32)
                to = level
            else:
                p = (a * dsb).astype(np.float32)
                total = _core.prio_eigen_sum_double(p.astype(np.float64))
                if total > 0:
                    p = (p * np.float32(1 / total)).astype(np.float32)
                a = (a - np.float32(ia.beta) * p).astype(np.float32)
                a = np.maximum(a, np.float32(0))
                a_sum = _core.prio_eigen_sum_double(a.astype(np.float64))
                to = level if ia.normalize else a_sum
            replace(k, a, to)

    interact(interaction)

    # matrix connectivity: groups of layers sharing a matrix
    if matrix_connectivity is not None:
        groups: dict = {}
        for k, i in enumerate(kept):
            m = spec(matrix_connectivity, i)
            if m is not None:
                kern, key = m
                groups.setdefault(key, []).append((k, kern))
        for key, members in groups.items():
            mat = np.asarray(matrices[key], dtype=np.float32)
            n = len(members)
            if mat.shape != (n, n):
                raise ValueError(f"matrix {key!r}: {mat.shape} for {n} features")
            conv = []
            for k, kern in members:
                q = q_of(k)
                scale = np.asarray(data.layer_scale)[k]
                v = np.where(q >= 0, (q.astype(np.float32).astype(np.float64) * scale).astype(np.float32), 0)
                conv.append(convolve(_dense(v.astype(np.float32), cells, H, W, np.float32).astype(np.float64),
                                     kern, workers).ravel()[cells])
            for t, (k, _) in enumerate(members):
                res = np.zeros(cells.size, dtype=np.float32)
                for s_, c in enumerate(conv):
                    res = (res + mat[s_, t] * c).astype(np.float32)
                q = q_of(k)
                present = q >= 0
                v = np.where(present, q, 0.0).astype(np.float32).astype(np.float64) * (1.0 / (
                    255.0 if data.bits == 8 else 65535.0))
                res = (res.astype(np.float64) * v).astype(np.float32)
                res[~present] = 0.0
                replace(k, res, np.asarray(data.initial_levels)[k])

    interact(interaction2)
