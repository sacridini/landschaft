# SPDX-License-Identifier: GPL-3.0-or-later
"""Demands per region: any demand of the allocations, one per region of a map."""

from __future__ import annotations

import numpy as np

from ._change import _class_value
from ._maps import read_map, read_series

__all__ = ["Regional"]


class Regional:
    """A demand per region (municipalities, biomes, basins...).

    Parameters
    ----------
    regions
        Map of region ids (cells without a value, or with an id not in
        ``demands``, belong to no region and keep their class).
    demands
        ``{region id: demand}``: :class:`AreaDemand`, :class:`Markov` or a
        rate matrix / ``{(from, to): rate}`` as the allocation takes them,
        all with the same number of steps.

    The allocations meet each region's demand on its cells: :class:`CLUE`
    and :class:`Ordered` allocate region by region (in the order of the
    ids, with the one generator), :class:`CARS` with a demand and an
    adaptive coefficient per region and class, :class:`ExpanderPatcher`
    with each region's rates and its probabilities set to 0 outside the
    region (Dinamica's regionalization).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> regions = np.repeat([[1, 1, 2, 2]], 2, axis=0)
    >>> t0 = np.array([[1, 1, 1, 2], [1, 2, 1, 1]])
    >>> t1 = np.array([[1, 2, 1, 2], [2, 2, 2, 2]])
    >>> d = ls.simulate.Regional.from_maps(regions, [t0, t1])
    >>> d.at(1, [1, 2], region=1), d.at(1, [1, 2], region=2)
    (array([1., 3.]), array([1., 3.]))
    """

    def __init__(self, regions, demands: dict):
        if not isinstance(demands, dict) or not demands:
            raise ValueError("give a demand per region: {region id: demand}")
        m = read_map(regions)
        v = m.as_float()
        if (np.abs(np.nan_to_num(v) - np.round(np.nan_to_num(v))) > 0).any():
            raise ValueError("the region ids must be integers")
        self.map = v
        self.template = m.template
        self.demands = {int(k): d for k, d in sorted(demands.items())}
        steps = {self._steps(d) for d in self.demands.values()} - {None}
        if len(steps) > 1:
            raise ValueError(f"the regions' demands have different numbers of steps: {sorted(steps)}")
        self._n = steps.pop() if steps else None

    @staticmethod
    def _steps(d):
        return getattr(d, "steps", None)

    @property
    def ids(self) -> list[int]:
        return list(self.demands)

    @property
    def steps(self) -> int | None:
        return self._n

    @property
    def classes(self) -> list:
        out: list = []
        for d in self.demands.values():
            cls = getattr(d, "classes", None)
            if cls is None:  # rate tables
                cls = list(getattr(d, "index", [])) + list(getattr(d, "columns", []))
                if isinstance(d, dict):
                    cls = [c for k in d for c in k]
            for c in cls:
                c = _class_value(c)
                if c not in out:
                    out.append(c)
        return sorted(out)

    def mask(self, region) -> np.ndarray:
        """Boolean map of a region's cells."""
        return self.map == int(region)

    def index(self, shape) -> np.ndarray:
        """Region index (0 .. n - 1, in the order of the ids) of each cell, -1 outside the regions."""
        if self.map.shape != tuple(shape):
            raise ValueError("the regions map differs in size from the landscape")
        out = np.full(self.map.shape, -1, dtype=np.int32)
        for k, r in enumerate(self.ids):
            out[self.map == r] = k
        return out

    def at(self, step: int, classes: list, region=None) -> np.ndarray:
        """A region's area demand at a step (``region=None``: all regions added up)."""
        if region is not None:
            return self.demands[int(region)].at(step, classes)
        return np.sum([d.at(step, classes) for d in self.demands.values()], axis=0)

    @classmethod
    def from_maps(cls, regions, maps, times=None, out_times=None, *, nodata=None) -> Regional:
        """Each region's areas in observed maps interpolated linearly (:meth:`AreaDemand.from_maps`)."""
        from ._demand import AreaDemand

        ms, ts = read_series(maps, times, nodata)
        r = read_map(regions).as_float()
        if r.shape != ms[0].shape:
            raise ValueError("the regions map differs in size from the maps")
        ids = sorted({int(v) for v in np.unique(r[~np.isnan(r)])})
        demands = {}
        for rid in ids:
            inside = r == rid
            series = [np.where(inside, m.as_float(), np.nan) for m in ms]
            demands[rid] = AreaDemand.from_maps(series, ts, out_times)
        return cls(regions, demands)

    @classmethod
    def markov(cls, regions, t0, t1, *, steps: int = 1, nodata=None) -> Regional:
        """A Markov chain per region from two maps (:meth:`Markov.from_maps`)."""
        from ._markov import Markov

        a, b = read_map(t0, nodata).as_float(), read_map(t1, nodata).as_float()
        r = read_map(regions).as_float()
        if not r.shape == a.shape == b.shape:
            raise ValueError("the regions map differs in size from the maps")
        demands = {}
        for rid in sorted({int(v) for v in np.unique(r[~np.isnan(r)])}):
            inside = r == rid
            demands[rid] = Markov.from_maps(np.where(inside, a, np.nan), np.where(inside, b, np.nan), steps=steps)
        return cls(regions, demands)

    def __repr__(self) -> str:
        return f"Regional({len(self.demands)} regions, steps={self.steps})"
