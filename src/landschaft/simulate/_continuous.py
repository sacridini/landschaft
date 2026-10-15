# SPDX-License-Identifier: GPL-3.0-or-later
"""The Patcher of a continuous state: patches that take a stock up to a quota.

Dinamica EGO's ``PatcherContinuousState``, clean-room from its documentation
and black-box runs of Dinamica EGO 8.13: instead of changing the class of a
number of cells, new patches (logging, clearing, harvest) take the stock of
their cells (biomass, carbon, timber) until a quota of stock is met.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from ._maps import read_map

__all__ = ["ContinuousPatcher", "StockAllocation", "StockSimulation"]

_UNITS = ("hectares", "cells", "area")


@dataclass
class StockAllocation:
    """What :meth:`ContinuousPatcher.allocate` took.

    Attributes
    ----------
    stock : xarray.DataArray or numpy.ndarray
        The stock left (0 in the cells taken), in the stock's units.
    probability : xarray.DataArray or numpy.ndarray
        The probabilities with the cells taken at 0 (Dinamica's "depleted"
        probabilities, for the next step).
    taken : xarray.DataArray or numpy.ndarray
        The stock taken from each cell in the quota's units (the stock times
        the cell's area; 0 elsewhere).
    patches : xarray.DataArray or numpy.ndarray
        The number of the patch of each cell taken (1, 2, ...; 0 elsewhere).
    remaining : float
        The quota not met: negative when the last cell exceeded it, positive
        when the ranked cells ran out.
    n_patches, n_cells : int
        Patches and cells taken.
    seed : int
    """

    stock: object
    probability: object
    taken: object
    patches: object
    remaining: float
    n_patches: int
    n_cells: int
    seed: int

    def __repr__(self) -> str:
        return (f"StockAllocation({self.n_patches} patches, {self.n_cells} cells, "
                f"remaining quota {self.remaining:g})")


@dataclass
class StockSimulation:
    """What :meth:`ContinuousPatcher.run` did, step by step.

    Attributes
    ----------
    stock : xarray.DataArray
        The stock at the start and after each step (dimension ``step``).
    taken : xarray.DataArray
        The stock taken in each step (quota units per cell).
    steps : pandas.DataFrame
        Per step: ``quota``, ``taken``, ``remaining``, ``patches``, ``cells``.
    seed : int
    """

    stock: xr.DataArray
    taken: xr.DataArray
    steps: pd.DataFrame
    seed: int

    def __repr__(self) -> str:
        return f"StockSimulation({len(self.steps)} steps, seed={self.seed})"


@dataclass
class ContinuousPatcher:
    """New patches that take a stock (biomass, carbon, timber) up to a quota.

    Dinamica EGO's ``PatcherContinuousState``: the Patcher of a continuous
    state. Each patch seed is drawn among the cells with stock and
    probability, the patch grows to a lognormal size, every cell of it gives
    all its stock, and patches are made until the stock taken reaches the
    quota.

    - Candidates: the cells with stock > 0 and probability > 0 ranked from
      the highest probability; the first ones kept while their summed stock
      is at most ``prune_factor x quota``.
    - Seeds: drawn from those without replacement in proportion to the
      probability (measured in Dinamica EGO 8.13). The list is not ranked
      again when it runs out: the quota is then left unmet.
    - Patches: lognormal size of the given mean and variance; each new cell
      drawn among the cells with stock around the patch in proportion to
      ``P x isometry^k``, ``k`` the patch cells in its window (as
      :class:`ExpanderPatcher`); a patch stops as soon as the quota is met,
      so the last cell may exceed it.

    Parameters
    ----------
    mean, variance
        Mean and variance of the patch sizes, in ``unit``.
    isometry
        0 leaves single cells, 1 follows the probabilities, 2 makes compact
        patches (0-2).
    unit : {"hectares", "cells", "area"}, default "hectares"
        Units of the patch sizes and of the stock's area: Dinamica's stock
        per hectare and sizes in hectares (from the map's cell area in square
        metres), or per cell and in cells, or per map area unit and in map
        area units.
    prune_factor
        How much more stock than the quota the ranked cells hold (Dinamica's
        10).
    window
        The neighbour window of patch growth (rows, columns), odd.

    Notes
    -----
    Dinamica's functor crashes for patches of more than one cell (as its
    Patcher does), so the candidates, the pruning, the seed draw and the
    quota were measured with one-cell patches; the patch growth is the
    published Patcher's. Its random numbers depend on more than its seed
    (the same run in another folder differs), so the comparison is
    statistical.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> rng = np.random.default_rng(1)
    >>> biomass = np.full((20, 20), 9.0)           # per cell
    >>> p = rng.random((20, 20))
    >>> cp = ls.simulate.ContinuousPatcher(mean=4, variance=1, isometry=1.5, unit="cells")
    >>> a = cp.allocate(biomass, p, quota=100, seed=1)
    >>> a.n_cells, a.remaining
    (12, -8.0)
    """

    mean: float = 1.0
    variance: float = 0.0
    isometry: float = 1.0
    unit: str = "hectares"
    prune_factor: float = 10.0
    window: tuple = (3, 3)

    def __post_init__(self):
        if self.unit not in _UNITS:
            raise ValueError(f"unit must be one of {_UNITS}")
        if not self.mean > 0:
            raise ValueError("the mean patch size must be positive")
        if not self.variance >= 0:
            raise ValueError("the variance must be >= 0")
        if not 0 <= self.isometry <= 2:
            raise ValueError("the isometry must be between 0 and 2")
        if not self.prune_factor > 0:
            raise ValueError("prune_factor must be positive")
        if len(self.window) != 2 or any(int(w) < 1 or int(w) % 2 == 0 for w in self.window):
            raise ValueError("the window needs two odd sides")

    def _area(self, m) -> float:
        """The area of a cell in ``unit``."""
        if self.unit == "cells":
            return 1.0
        t = m.template
        try:
            rx, ry = (abs(float(v)) for v in t.rio.resolution())
        except Exception as e:
            raise ValueError(f"unit={self.unit!r} needs a georeferenced map (or unit='cells')") from e
        area = rx * ry
        return area / 1e4 if self.unit == "hectares" else area

    def _read(self, stock, probability, nodata):
        s = read_map(stock, nodata)
        p = read_map(probability)
        if s.shape != p.shape:
            raise ValueError(f"the probability map has shape {p.shape}, the stock {s.shape}")
        return s, np.ascontiguousarray(s.as_float()), np.ascontiguousarray(p.as_float())

    def _step(self, values, prob, area, quota, seed):
        r = _core.sim_continuous_patches(values * area, prob, float(quota), self.mean / area,
                                         self.variance / (area * area), float(self.isometry),
                                         float(self.prune_factor), int(self.window[0]), int(self.window[1]),
                                         int(seed) & 0xFFFFFFFFFFFFFFFF)
        return r

    def allocate(self, stock, probability, quota: float, *, seed: int | None = None, nodata=None) -> StockAllocation:
        """Take ``quota`` units of stock as new patches.

        Parameters
        ----------
        stock : xarray.DataArray or array_like
            The stock of each cell (per hectare, per cell or per area unit:
            see ``unit``; NaN, ``nodata`` or <= 0: none).
        probability : xarray.DataArray or array_like
            The probability of each cell (NaN or <= 0: never taken).
        quota : float
            The stock to take, in the stock's units times the area unit.
        seed : int, optional
            Seed of the generator (xoshiro256**); default: drawn at random.

        Returns
        -------
        StockAllocation
        """
        m, values, prob = self._read(stock, probability, nodata)
        area = self._area(m)
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        r = self._step(values, prob, area, quota, seed)
        taken = np.asarray(r["taken"])
        hit = taken > 0
        left = np.where(hit, 0.0, values)
        p_out = np.where(hit, 0.0, prob)
        t = m.template
        wrap = (lambda v, name: v) if t is None else (lambda v, name: t.copy(data=v).rename(name))
        return StockAllocation(wrap(left, "stock"), wrap(p_out, "probability"), wrap(taken, "taken"),
                               wrap(np.asarray(r["patch"]), "patch"), float(r["remaining"]), int(r["patches"]),
                               int(r["cells"]), int(seed))

    def run(self, stock, probability, quotas, *, regrowth=None, seed: int | None = None,
            nodata=None) -> StockSimulation:
        """Take a quota per step, the probabilities depleted from step to step.

        Parameters
        ----------
        stock, probability
            As in :meth:`allocate`.
        quotas : sequence of float
            The quota of each step.
        regrowth : callable, optional
            ``regrowth(stock, step) -> stock`` applied to the stock left after
            each step (numpy arrays, in the stock's units), e.g. growth of
            logged forest.
        seed : int, optional

        Returns
        -------
        StockSimulation
        """
        m, values, prob = self._read(stock, probability, nodata)
        area = self._area(m)
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        stocks, takens, rows = [values.copy()], [], []
        for k, q in enumerate(quotas, start=1):
            r = self._step(values, prob, area, q, (int(seed) + k * 0x9E3779B9) & 0xFFFFFFFFFFFFFFFF)
            taken = np.asarray(r["taken"])
            hit = taken > 0
            values = np.where(hit, 0.0, values)
            prob = np.where(hit, 0.0, prob)
            if regrowth is not None:
                values = np.asarray(regrowth(values.copy(), k), dtype=np.float64)
                if values.shape != prob.shape:
                    raise ValueError("regrowth changed the shape of the stock")
            stocks.append(values.copy())
            takens.append(taken)
            rows.append({"step": k, "quota": float(q), "taken": float(taken.sum()), "remaining": float(r["remaining"]),
                         "patches": int(r["patches"]), "cells": int(r["cells"])})
        steps = list(range(len(stocks)))
        t = m.template

        def stack(arrs, name, first):
            a = np.stack(arrs)
            if t is None:
                return xr.DataArray(a, dims=("step", "y", "x"), coords={"step": steps[first:]}, name=name)
            out = xr.concat([t.copy(data=v) for v in arrs], dim="step").assign_coords(step=steps[first:])
            return out.rename(name)

        return StockSimulation(stack(stocks, "stock", 0), stack(takens, "taken", 1) if takens else None,
                               pd.DataFrame(rows).set_index("step") if rows else pd.DataFrame(), int(seed))

    def __repr__(self) -> str:
        return (f"ContinuousPatcher(mean={self.mean:g}, variance={self.variance:g}, isometry={self.isometry:g}, "
                f"unit={self.unit!r})")
