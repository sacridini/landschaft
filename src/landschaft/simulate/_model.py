# SPDX-License-Identifier: GPL-3.0-or-later
"""A land change model made of interchangeable pieces, and its simulations.

``Model(potential, demand, allocation, rules)``: the suitability of each
class (any potential piece), how many cells each class must have at each
step, how cells are allocated to meet it, and the rules that constrain it.
``run`` simulates step by step from a starting map, as lulcc's
``allocate``: at each step the suitabilities get the elasticities of the
cells' own classes, the neighbourhood and decision rules remove the
conversions not allowed, cells whose class is no longer allowed are
converted, and the allocation meets the step's demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._threads import threaded
from ._change import _class_value, _classes
from ._maps import read_map
from ._rules import Rules, allow_matrix, auto_convert

__all__ = ["Model", "Simulation"]


@dataclass
class Simulation:
    """The maps of a simulation and what happened at each step.

    Attributes
    ----------
    maps : xarray.DataArray
        ``(step, y, x)``: the starting map and one map per step.
    steps : pandas.DataFrame
        Per step: passes of the allocation, whether it met the demand, the
        largest and mean difference (cells), the cells of each class.
    seed : int
        The seed of R's generator the simulation used.
    """

    maps: xr.DataArray
    steps: pd.DataFrame
    seed: int
    history: np.ndarray | None = field(default=None, repr=False)
    ages: xr.DataArray | None = field(default=None, repr=False)
    changed: np.ndarray | None = field(default=None, repr=False)
    supply_table: object = field(default=None, repr=False)
    demand_table: object = field(default=None, repr=False)
    stack: object = field(default=None, repr=False)
    roles: object = field(default=None, repr=False)

    @property
    def layers(self) -> dict:
        """With a :class:`LandSystemStack`: ``{layer: (step, y, x)}`` binary maps, as the input."""
        if self.stack is None:
            raise ValueError("layers need a simulation of a LandSystemStack")
        return self.stack.layers(self.maps)

    @property
    def supply(self) -> pd.DataFrame:
        """The goods and services supplied at each step (CLUMondo)."""
        cols = [c for c in self.steps.columns if c.startswith("supply_")]
        if not cols:
            raise ValueError("no supply: not a CLUMondo simulation")
        return self.steps[cols].rename(columns=lambda c: c[len("supply_"):]).astype(float)

    @property
    def shortfall(self) -> pd.DataFrame:
        """Demand minus supply at each step (positive: short)."""
        sup = self.supply
        dem = self.demand_table.iloc[1: len(sup) + 1][list(sup.columns)].to_numpy(dtype=float)
        return pd.DataFrame(dem - sup.to_numpy(), index=sup.index, columns=sup.columns)

    def cover(self) -> pd.DataFrame:
        """Cells of each layer (a stack) or class at each step."""
        m = self.maps.values
        steps = pd.Index(range(m.shape[0]), name="step")
        if self.stack is not None:
            return pd.DataFrame({n: np.nansum(np.asarray(v), axis=(1, 2)).astype(int)
                                 for n, v in self.layers.items()}, index=steps)
        classes = sorted(c for c in np.unique(m[~np.isnan(m)]))
        return pd.DataFrame({_class_value(c): (m == c).sum(axis=(1, 2)) for c in classes}, index=steps)

    def trajectory(self, roles=None) -> pd.DataFrame:
        """Cells changing at each step, by the role of their transition (:class:`Roles`)."""
        roles = roles or self.roles
        if roles is None:
            raise ValueError("give the Roles")
        layers_of = self.stack.names_of if self.stack is not None else (lambda v: (str(_class_value(v)),))
        m = self.maps.values.reshape(self.maps.shape[0], -1)
        rows = []
        for k in range(1, m.shape[0]):
            a, b = m[k - 1], m[k]
            ch = (a != b) & ~np.isnan(a) & ~np.isnan(b)
            row = {"step": k}
            if ch.any():
                pairs, counts = np.unique(np.stack([a[ch], b[ch]]), axis=1, return_counts=True)
                for (f, t), c in zip(pairs.T, counts):
                    role = roles._lookup(_class_value(f), _class_value(t), layers_of)
                    row[role] = row.get(role, 0) + int(c)
            rows.append(row)
        return pd.DataFrame(rows).set_index("step").fillna(0).astype(int)

    def outputs(self, table, *, initial=None) -> xr.DataArray:
        """Per step, the value in ``table`` of each cell's class (CLUMondo's output maps).

        Parameters
        ----------
        table
            A DataFrame with a row per class and a column per output (carbon,
            jobs...), or a Series for one output.
        initial
            Optional maps (``{output: map}``) of the real values at the start:
            cells that never changed class keep them (CLUMondo's
            ``initoutputmap``).

        Returns
        -------
        xarray.DataArray
            ``(output, step, y, x)``.
        """
        if isinstance(table, pd.Series):
            table = table.to_frame()
        t = table.copy()
        t.index = [_class_value(v) for v in t.index]
        m = self.maps.values
        flat = m.reshape(m.shape[0], -1)
        layers = []
        for col in t.columns:
            lut = dict(zip(t.index, t[col].to_numpy(dtype=np.float64)))
            v = np.full(flat.shape, np.nan)
            for cls, val in lut.items():
                v[flat == cls] = val
            if initial is not None and col in initial:
                init = read_map(initial[col]).as_float().ravel()
                changed = np.zeros(flat.shape[1], dtype=bool)
                for k in range(1, flat.shape[0]):
                    changed |= (flat[k] != flat[k - 1]) & ~np.isnan(flat[k])
                    v[k, ~changed] = init[~changed]
                v[0] = np.where(np.isnan(flat[0]), np.nan, init)
            layers.append(v.reshape(m.shape))
        da = xr.DataArray(np.stack(layers), dims=("output", *self.maps.dims),
                          coords={"output": list(t.columns), **dict(self.maps.coords)}, name="output")
        return da

    def __repr__(self) -> str:
        return f"Simulation({self.maps.shape[0] - 1} steps, seed={self.seed})"


def _as_maps(stack: np.ndarray, t, name: str) -> xr.DataArray:
    """``(step, y, x)`` maps on the template's grid."""
    if t is None:
        return xr.DataArray(stack, dims=("step", "y", "x"), coords={"step": np.arange(stack.shape[0])}, name=name)
    da = xr.DataArray(stack, dims=("step", *t.dims),
                      coords={"step": np.arange(stack.shape[0]), **{k: t.coords[k] for k in t.dims}}, name=name)
    try:
        from ..io import _with_georef

        da = _with_georef(da, transform=t.rio.transform())
        if t.rio.crs is not None:
            da = _with_georef(da, crs=t.rio.crs)
    except Exception:
        pass
    return da


class Model:
    """A land change model: potential, demand, allocation and rules.

    Parameters
    ----------
    potential
        The suitability of each class: a fitted :class:`LogisticRegression`,
        :class:`Estimator`, :class:`RasterPotential`... (anything with
        ``predict(drivers, landscape=...)`` giving a layer per class).
    demand
        The cells of each class at each step (:class:`AreaDemand`).
    allocation
        :class:`CLUE` or :class:`Ordered`.
    rules
        :class:`Rules` (allowed conversions, elasticities, restrictions,
        neighbourhood, history).

    Examples
    --------
    >>> import landschaft as ls
    >>> lu = ls.load_raster(ls.simulate.example_data("pie_landuse"))
    >>> ef = ls.load_raster(ls.simulate.example_data("pie_drivers"))
    >>> forms = {1: ["elevation", "slope"], 2: ["elevation", "slope", "dist_built_1985"], 3: ["elevation", "slope"]}
    >>> model = ls.simulate.Model(
    ...     potential=ls.simulate.LogisticRegression(forms).fit(lu[0], ef),
    ...     demand=ls.simulate.AreaDemand.from_maps(lu, times=[0, 6, 14]),
    ...     allocation=ls.simulate.CLUE(jitter=0.0002, scale=1e-6, max_diff=50, mean_diff=50),
    ...     rules=ls.simulate.Rules(elasticity=[0.2, 0.2, 0.2]))
    >>> sim = model.run(lu[0], ef, seed=1)
    >>> sim.maps.shape[0]
    15
    """

    def __init__(self, potential, demand, allocation, rules: Rules | None = None):
        self.potential = potential
        self.demand = demand
        self.allocation = allocation
        self.rules = rules or Rules()

    @threaded
    def run(self, landscape, drivers=None, *, seed: int | None = None, nodata=None) -> Simulation:
        """Simulate every step of the demand from ``landscape``.

        Parameters
        ----------
        landscape
            The map at the start (step 0 of the demand).
        drivers
            The explanatory variables the potential needs.
        seed
            Seed of R's generator (``set.seed``): the same seed gives the
            same simulation, and the same as lulcc's with ``set.seed(seed)``.
            Default: drawn at random (kept in the result).
        nodata
            Value marking missing cells in the landscape.

        Returns
        -------
        Simulation
        """
        if hasattr(self.allocation, "simulate"):  # allocations with their own yearly loop (CLUMondo)
            return self.allocation.simulate(self, landscape, drivers, seed=seed, nodata=nodata)
        land = read_map(landscape, nodata)
        values = land.as_float().ravel()
        cells = np.flatnonzero(~np.isnan(values))
        classes = list(self.demand.classes)
        found = {_class_value(c) for c in _classes([land])}
        if not found <= set(classes):
            raise ValueError(f"the landscape has classes without demand: {sorted(found - set(classes))}")
        prob_da = self.potential.predict(drivers, landscape=landscape) if drivers is not None \
            else self.potential.predict(landscape=landscape)
        pclasses = [_class_value(c) for c in prob_da[prob_da.dims[0]].values]
        missing = set(classes) - set(pclasses)
        if missing:
            raise ValueError(f"the potential gives no suitability for classes {sorted(missing)}")
        flat = prob_da.values.reshape(prob_da.shape[0], -1)
        prob = np.stack([flat[pclasses.index(c)][cells] for c in classes], axis=1)
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        rng = _core.RRandom(int(seed))
        x0 = values[cells].copy()
        hist = self.rules.history_values(cells, values.size)
        mask = self.rules.mask_values(cells, values.size)
        rules = self.rules.matrix(classes)
        elas = self.rules.elasticities(classes)
        nb = self.rules.neighbourhood
        maps = [values.reshape(land.shape)]
        rows = []
        current = values.reshape(land.shape).copy()
        for step in range(1, self.demand.steps + 1):
            d = self.demand.at(step, classes)
            tprob = prob.copy()
            if getattr(self.allocation, "uses_elasticity", False):
                for j, c in enumerate(classes):
                    own = x0 == c
                    tprob[own, j] = tprob[own, j] + elas[j]
            if nb is not None:
                tprob = tprob * nb.allow(current, classes)[cells]
            if rules is not None:
                cd = d - self.demand.at(step - 1, classes)
                tprob = tprob * allow_matrix(x0, hist, classes, cd, rules)
            auto = auto_convert(x0, tprob, classes, mask)
            fixed = ~np.isnan(auto)
            x0 = np.where(fixed, auto, x0)
            tprob[fixed] = np.nan
            x1, info = self.allocation.allocate(tprob, x0, d, classes, rng)
            x1 = np.asarray(x1, dtype=np.float64)
            hist = np.where(x0 == x1, hist + 1, 0)
            out = np.full(values.size, np.nan)
            out[cells] = x1
            current = out.reshape(land.shape)
            maps.append(current)
            rows.append({"step": step, **info, **{f"cells_{c}": int(np.sum(x1 == c)) for c in classes}})
            x0 = x1
        da = _as_maps(np.stack(maps), land.template, "landuse")
        hist_map = np.full(values.size, np.nan)
        hist_map[cells] = hist
        return Simulation(da, pd.DataFrame(rows).set_index("step"), int(seed), hist_map.reshape(land.shape))

    def __repr__(self) -> str:
        return (f"Model(potential={type(self.potential).__name__}, demand={self.demand!r}, "
                f"allocation={type(self.allocation).__name__})")
