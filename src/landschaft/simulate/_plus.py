# SPDX-License-Identifier: GPL-3.0-or-later
"""CARS: the cellular automaton of PLUS based on multi-type random patch seeds.

PLUS (Liang et al. 2021, Computers, Environment and Urban Systems 85,
101569) pairs a land expansion analysis (LEAS: a model of where each class
appeared between two maps, :class:`Estimator` with
``target="expansion"``) with CARS, a cellular automaton whose classes
compete for cells through their growth probability, their share of the
neighbourhood and an adaptive coefficient that follows the gap to the
demand; classes without neighbours in a cell may still start a patch there
(random seeds), and a threshold that decreases while the demand is not met
decides which proposed changes happen. PLUS is distributed without source
code or licence: this is written from the paper alone.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import _core
from ._change import _class_value
from ._demand import _round_sum
from ._maps import read_map

__all__ = ["CARS"]


def _per_class(value, classes, what, default):
    """One value per class: a number for all, or ``{class: value}`` (``default`` for the others)."""
    if isinstance(value, dict):
        v = {_class_value(k): float(x) for k, x in value.items()}
        unknown = set(v) - set(classes)
        if unknown:
            raise ValueError(f"{what} for classes the model does not have: {sorted(unknown)}")
        return np.array([v.get(c, default) for c in classes], dtype=np.float64)
    return np.full(len(classes), float(value))


@dataclass
class CARS:
    """PLUS's CA based on multi-type random patch seeds (Liang et al. 2021).

    Each step, CARS iterates from the current map until every class has the
    cells the demand asks for. In an iteration, each free cell ``i`` gives
    each class ``k`` the overall probability

    - ``OP = P(i,k) x Omega(i,k) x D(k)``, ``Omega = w(k) x`` the cells of
      ``k`` in the ``window x window`` neighbourhood (the centre left out)
      over ``window^2 - 1``;
    - where ``Omega = 0``, a random seed: ``OP = P x (r x mu(k)) x D`` when a
      uniform ``r < P``, else 0;

    a roulette wheel over ``OP`` picks a class ``c``, and the cell turns to
    ``c`` when ``P(i,c) > tau`` and the conversion is allowed;
    ``tau = decay^l x r``, ``r`` normal of mean 1 in [0, 2], ``l`` raised by
    one after each iteration that cut the total gap to the demand by less
    than ``step_size`` cells. ``D(k)`` starts at 1 and grows while the
    class's shortage grows (shrinks while its excess grows).

    The growth probabilities ``P`` come from the model's potential (one
    layer per class, as an :class:`Estimator` with ``target="expansion"``
    gives: PLUS's LEAS); the demand is an :class:`AreaDemand` or a
    :class:`Markov` chain projected from the starting map (PLUS's default);
    the rules give the conversions allowed (``allowed``, 0/1, rows the
    current class) and the restricted cells (``restrictions``).

    Parameters
    ----------
    weights
        Neighbourhood weight ``w(k)`` of each class, 0 to 1 (a number for
        all or ``{class: weight}``; default 1).
    seeds
        Patch generation threshold ``mu(k)`` of each class, 0 to 1 (the
        paper's 0.1): how strongly random seeds compete with the growth of
        existing patches.
    decay
        Decay factor ``delta`` of the threshold, in (0, 1] (the paper's 0.9).
    step_size
        Gap reduction (cells per iteration) below which the threshold
        decays (PLUS's "Step"; the paper's 500).
    window
        Side of the neighbourhood, odd (3 = the Moore neighbourhood).
    max_iter
        Iterations per step before giving up (the map is kept as it is).
    steps
        Steps to simulate (default: the demand's).

    Notes
    -----
    From the paper alone (PLUS has no source code): the standard deviation
    of the threshold's normal ``r`` is not given (here 1/3, values outside
    [0, 2] drawn again); the changes proposed in an iteration are made in a
    random order, a class taking no cell once it has its demand and losing
    none while it has no more than its demand (the paper does not say how a
    class is held at its demand). The random numbers of a cell come from a
    hash of (seed, step, iteration, cell): the same seed gives the same
    simulation with any number of threads.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> S = ls.simulate
    >>> land = np.ones((40, 40)); land[15:25, 15:25] = 2
    >>> p = S.RasterPotential({1: np.full((40, 40), 0.1), 2: np.full((40, 40), 0.8)})
    >>> demand = S.AreaDemand(np.array([[1500, 100], [1400, 200]]), classes=[1, 2])
    >>> sim = S.Model(p, demand, S.CARS()).run(land, seed=1)
    >>> int((sim.maps[-1] == 2).sum())
    200
    """

    weights: object = 1.0
    seeds: object = 0.1
    decay: float = 0.9
    step_size: float = 500.0
    window: int = 3
    max_iter: int = 5000
    steps: int | None = None

    def __post_init__(self):
        if not 0 < self.decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        if self.window < 3 or self.window % 2 == 0 or self.window > 255:
            raise ValueError("window must be odd, from 3 to 255")
        if self.step_size < 0:
            raise ValueError("step_size must not be negative")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")

    # ------------------------------------------------------------------ pieces of the model

    @staticmethod
    def _check_rules(rules):
        if rules is None:
            return
        unused = [name for name in ("elasticity", "history", "neighbourhood", "allow_maps", "roles",
                                    "transition_resistance", "conversion_order", "caps", "succession")
                  if getattr(rules, name, None) is not None]
        if unused:
            raise ValueError(f"CARS uses only the rules 'allowed' and 'restrictions', not {unused}")

    def _demand(self, demand, classes, counts, ncell):
        """(steps + 1, classes) cells, row 0 the start."""
        from ._demand import AreaDemand
        from ._markov import Markov

        if isinstance(demand, Markov):
            mc = [_class_value(c) for c in demand.classes]
            extra = set(classes) - set(mc)
            if any(counts[classes.index(c)] for c in extra):
                raise ValueError(f"the Markov chain has no rows for classes {sorted(extra)}")
            n = self.steps if self.steps is not None else demand.steps
            start = {c: counts[classes.index(c)] for c in mc}
            proj = demand.project(pd.Series(start), int(n))
            table = np.zeros((proj.shape[0], len(classes)))
            for c in mc:
                table[:, classes.index(c)] = proj[c].to_numpy(dtype=np.float64)
            table[1:] = _round_sum(table[1:], ncell)
            table[0] = counts
            return table
        if isinstance(demand, AreaDemand):
            n = self.steps if self.steps is not None else demand.steps
            if n > demand.steps:
                raise ValueError(f"the demand has {demand.steps} steps, not {n}")
            table = np.array([demand.at(s, classes) for s in range(int(n) + 1)])
            if (np.abs(table - np.round(table)) > 0).any() or (table < 0).any():
                raise ValueError("the demand must be whole, non-negative numbers of cells")
            sums = table[1:].sum(axis=1)
            bad = np.flatnonzero(sums != ncell)
            if bad.size:
                raise ValueError(f"the demand of step {bad[0] + 1} adds up to {sums[bad[0]]:g} cells, "
                                 f"the map has {ncell}")
            return table
        raise TypeError("CARS takes an AreaDemand or a Markov chain as demand")

    def _probability(self, potential, landscape, drivers, classes, shape):
        p = potential.predict(drivers, landscape=landscape) if drivers is not None \
            else potential.predict(landscape=landscape)
        names = [_class_value(c) for c in p[p.dims[0]].values]
        out = np.zeros((len(classes), shape[0] * shape[1]))
        missing = []
        for j, c in enumerate(classes):
            if c in names:
                out[j] = np.nan_to_num(np.asarray(p.values[names.index(c)], dtype=np.float64).ravel())
            else:
                missing.append(c)
        if ((out < 0) | (out > 1)).any():
            raise ValueError("the growth probabilities must be in [0, 1]")
        return out, missing

    # ------------------------------------------------------------------ simulation

    def simulate(self, model, landscape, drivers=None, *, seed=None, nodata=None):
        from ._demand import AreaDemand
        from ._dynamic import current_map, with_dynamic
        from ._markov import Markov
        from ._model import Simulation, _as_maps
        from ._regional import Regional

        regional = isinstance(model.demand, Regional)
        parts = list(model.demand.demands.values()) if regional else [model.demand]
        if not all(isinstance(d, (AreaDemand, Markov)) for d in parts):
            raise TypeError("CARS takes an AreaDemand or a Markov chain as demand (or a Regional of them)")
        self._check_rules(model.rules)
        land = read_map(landscape, nodata)
        shape = land.shape
        values = land.as_float()
        nan = np.isnan(values)
        if (np.abs(np.nan_to_num(values) - np.round(np.nan_to_num(values))) > 0).any():
            raise ValueError("the landscape's classes must be integers")
        found = sorted({_class_value(c) for c in np.unique(values[~nan])})
        demand_classes = [_class_value(c) for c in model.demand.classes]
        unknown = set(found) - set(demand_classes)
        if unknown:
            raise ValueError(f"the landscape has classes without demand: {sorted(unknown)}")
        classes = demand_classes
        index = np.full(values.size, -1, dtype=np.int32)
        flat = values.ravel()
        for j, c in enumerate(classes):
            index[flat == c] = j
        region = model.demand.index(shape).ravel() if regional else None
        tables = []
        for k, d in enumerate(parts):
            inside = index >= 0 if region is None else (index >= 0) & (region == k)
            counts = np.bincount(index[inside], minlength=len(classes)).astype(np.float64)
            try:
                tables.append(self._demand(d, classes, counts, int(inside.sum())))
            except ValueError as e:
                if not regional:
                    raise
                raise ValueError(f"region {model.demand.ids[k]}: {e}") from None
        if len({t.shape[0] for t in tables}) > 1:
            raise ValueError("the regions' demands have different numbers of steps")
        table = np.stack(tables, axis=1)  # (steps + 1, regions, classes)

        def probability(drv, lmap):
            p, missing = self._probability(model.potential, lmap, drv, classes, shape)
            grows = [c for c in missing
                     if (table[1:, :, classes.index(c)] > table[:-1, :, classes.index(c)]).any()]
            if grows:
                raise ValueError(f"the potential gives no growth probability for classes {grows}, "
                                 "whose demand grows")
            return p

        prob = None if model.dynamic else probability(drivers, landscape)
        allowed = np.ones((len(classes), len(classes)), dtype=np.uint8)
        fixed = None
        if model.rules is not None:
            m = model.rules.matrix(classes)
            if m is not None:
                if not np.isin(m, (0, 1)).all():
                    raise ValueError("CARS takes a 0/1 conversion matrix")
                allowed = m.astype(np.uint8)
            if model.rules.restrictions is not None:
                r = read_map(model.rules.restrictions).as_float()
                if r.shape != shape:
                    raise ValueError("the restrictions map differs in size from the landscape")
                fixed = (np.nan_to_num(r, nan=0.0) != 0).astype(np.uint8)
        weights = _per_class(self.weights, classes, "weights", 1.0)
        mu = _per_class(self.seeds, classes, "seeds", 0.1)
        if ((weights < 0) | (weights > 1)).any() or ((mu < 0) | (mu > 1)).any():
            raise ValueError("weights and seeds must be in [0, 1]")
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        current = index.reshape(shape)
        rmap = None if region is None else region.reshape(shape)
        codes = np.asarray(classes, dtype=np.float64)
        maps = [values.copy()]
        rows = []
        out = values.copy()
        for step in range(1, table.shape[0]):
            if model.dynamic:
                lmap = current_map(out, land.template)
                prob = probability(with_dynamic(model.dynamic, drivers, lmap), lmap)
            r = _core.sim_cars(current, fixed, rmap, prob, weights, mu, allowed, table[step].astype(np.int64),
                               int(self.window), float(self.decay), float(self.step_size), int(self.max_iter),
                               (int(seed) + step * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
            current = np.asarray(r["landscape"], dtype=np.int32)
            out = np.where(current >= 0, codes[np.maximum(current, 0)], np.nan)
            maps.append(out)
            last = np.asarray(r["trace"])[-1]
            gap = int(np.abs(table[step].ravel() - last).sum())
            cells = np.bincount(current[current >= 0], minlength=len(classes))
            rows.append({"step": step, "iterations": int(r["iterations"]), "converged": bool(r["converged"]),
                         "decays": int(r["decays"]), "gap": gap,
                         **{f"cells_{c:g}": int(v) for c, v in zip(classes, cells)}})
            if not r["converged"]:
                warnings.warn(f"step {step}: CARS stopped after {self.max_iter} iterations {gap} cells away "
                              "from the demand (too few cells can take the classes short of it)", stacklevel=4)
        return Simulation(_as_maps(np.stack(maps), land.template, "landuse"), pd.DataFrame(rows).set_index("step"),
                          int(seed))

    def __repr__(self) -> str:
        return (f"CARS(weights={self.weights!r}, seeds={self.seeds!r}, decay={self.decay}, "
                f"step_size={self.step_size}, window={self.window})")
