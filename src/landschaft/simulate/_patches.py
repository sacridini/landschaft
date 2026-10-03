# SPDX-License-Identifier: GPL-3.0-or-later
"""Expander and Patcher: transitions allocated as expansions of patches and as new patches.

The transition engine of DINAMICA (Soares-Filho, Cerqueira & Pennachin
2002) and Dinamica EGO, written from its publications (its code is closed)
and from measurements of Dinamica EGO 8's functors: each step, the cells a
transition must change (its rate times the cells of its origin class) are
split between the Expander, which grows patches next to existing patches of
the target class, and the Patcher, which seeds new ones; patch sizes are
lognormal and the isometry shapes them.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import _core
from ._change import _class_value
from ._maps import read_map

__all__ = ["ExpanderPatcher"]


def _per_transition(value, transitions, what):
    """A value per transition: one for all, or {(from, to): value}."""
    if isinstance(value, dict):
        v = {(_class_value(f), _class_value(t)): x for (f, t), x in value.items()}
        missing = [tr for tr in transitions if tr not in v]
        if missing:
            raise ValueError(f"no {what} for transitions {missing}")
        return [v[tr] for tr in transitions]
    return [value] * len(transitions)


@dataclass
class ExpanderPatcher:
    """Dinamica's Expander and Patcher allocation of transitions (``AllocateTransitions``).

    Each step, the changes of a transition are ``round(rate x cells of the
    origin class)`` (the demand's rates, e.g. a :class:`Markov` chain); a
    share ``expansion`` goes to the Expander, the rest (and what the
    Expander could not place) to the Patcher.

    - Expander: a cell's probability becomes ``P`` where at least 5 cells of
      its 3 x 3 window hold the target class, else ``P n_j / 5`` (Dinamica
      EGO 8, measured; Soares-Filho et al. 2002 give ``n_j / 4`` from 4).
      The cells are ranked, the first ``prune_factor x changes`` kept, drawn
      in proportion to that probability and each tried once: a seed with
      that probability.
    - Patcher: seeds drawn in proportion to the probability among the first
      ``prune_factor x changes`` cells, until its changes are made.
    - Each seed grows a patch of lognormal size (mean, variance) cell by
      cell, each new cell drawn among the origin-class cells around the
      patch in proportion to ``P x isometry^k``, ``k`` the patch cells in its
      window: isometry 0 leaves single cells, 1 follows the probabilities,
      2 makes compact patches.

    Parameters
    ----------
    expansion
        Share of the changes made by the Expander (one value or
        ``{(from, to): share}``).
    expander, patcher
        ``(mean, variance, isometry)`` of the patches (one tuple or per
        transition), the sizes in ``size_unit``.
    size_unit
        ``"cells"``, ``"hectares"`` (Dinamica's: from the map's cell area in
        square metres) or ``"area"`` (the map's area units).
    prune_factor
        How many cells per change are ranked for the draw (Dinamica's 10).
    window
        The neighbour window (rows, columns), odd.
    steps
        Steps to simulate (default: the demand's ``steps``, e.g. a Markov
        chain's).

    Notes
    -----
    The functors of Dinamica EGO 8.13 crash for patches of more than one
    cell, so only their seed selection could be measured (Expander:
    acceptance ``n_j / 5`` per cell and the draw by probability, from
    12 x 5000 trials; Patcher: the draw by probability among the pruned
    cells); the patch growth follows the published descriptions.
    """

    expansion: object = 0.5
    expander: object = (1.0, 0.0, 1.0)
    patcher: object = (1.0, 0.0, 1.0)
    size_unit: str = "cells"
    prune_factor: float = 10.0
    window: tuple = (3, 3)
    steps: int | None = None

    def __post_init__(self):
        if self.size_unit not in ("cells", "hectares", "area"):
            raise ValueError("size_unit must be 'cells', 'hectares' or 'area'")
        if not self.prune_factor >= 1:
            raise ValueError("prune_factor must be at least 1")
        if len(self.window) != 2 or any(w < 1 or w % 2 == 0 for w in self.window):
            raise ValueError("the window needs two odd sides")

    def _rates(self, demand) -> pd.DataFrame:
        from ._markov import Markov

        if isinstance(demand, Markov):
            return demand.multi
        if isinstance(demand, pd.DataFrame):
            t = demand.copy()
            t.index = [_class_value(v) for v in t.index]
            t.columns = [_class_value(v) for v in t.columns]
            return t
        if isinstance(demand, dict):
            classes = sorted({_class_value(c) for k in demand for c in k})
            t = pd.DataFrame(0.0, index=classes, columns=classes)
            for (f, tt), v in demand.items():
                t.loc[_class_value(f), _class_value(tt)] = float(v)
            return t
        raise TypeError("the demand must be a Markov chain, a rate matrix or {(from, to): rate}")

    def _probability(self, potential, landscape, drivers, transitions, shape):
        """The probability of each transition at each cell (NaN -> 0)."""
        if isinstance(potential, dict):
            maps = {(_class_value(f), _class_value(t)): read_map(m).as_float() for (f, t), m in potential.items()}
            out = []
            for tr in transitions:
                if tr not in maps:
                    raise ValueError(f"no probability for transition {tr}")
                out.append(np.nan_to_num(maps[tr]))
            return out
        p = potential.probability(landscape, drivers) if drivers is not None else potential.probability(landscape)
        names = [str(v) for v in p[p.dims[0]].values]
        out = []
        for f, t in transitions:
            key = f"probability_{f:g}_to_{t:g}"
            if key not in names:
                raise ValueError(f"the potential gives no probability for {f:g} -> {t:g}")
            out.append(np.nan_to_num(p.values[names.index(key)]).reshape(shape))
        return out

    def _cells(self, value, land) -> float:
        if self.size_unit == "cells":
            return float(value)
        t = land.template
        rx = ry = 1.0
        if t is not None:
            try:
                rx, ry = (abs(float(v)) for v in t.rio.resolution())
            except Exception:
                rx = ry = 1.0
        area = rx * ry
        if self.size_unit == "hectares":
            area /= 1e4
        return float(value) / area

    def simulate(self, model, landscape, drivers=None, *, seed=None, nodata=None):
        from ._model import Simulation, _as_maps

        land = read_map(landscape, nodata)
        shape = land.shape
        values = land.as_float()
        rates = self._rates(model.demand)
        transitions = [(f, t) for f in rates.index for t in rates.columns
                       if f != t and rates.loc[f, t] > 0]
        if not transitions:
            raise ValueError("the demand has no transitions")
        steps = self.steps if self.steps is not None else getattr(model.demand, "steps", None)
        if steps is None:
            raise ValueError("give the steps (ExpanderPatcher(steps=...))")
        share = _per_transition(self.expansion, transitions, "expansion share")
        e_par = _per_transition(self.expander, transitions, "expander parameters")
        p_par = _per_transition(self.patcher, transitions, "patcher parameters")
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        nan = np.isnan(values)
        current = np.where(nan, np.iinfo(np.int32).min, np.nan_to_num(values)).astype(np.int64)
        if (np.abs(np.nan_to_num(values) - current.astype(float))[~nan] > 0).any():
            raise ValueError("the landscape's classes must be integers")
        current = current.astype(np.int32)
        maps = [values.copy()]
        rows = []
        for step in range(1, int(steps) + 1):
            cur_map = np.where(nan, np.nan, current.astype(np.float64))
            probs = self._probability(model.potential, _like(land, cur_map), drivers, transitions, shape)
            specs = []
            for k, (f, t) in enumerate(transitions):
                n = math.floor(rates.loc[f, t] * float(np.sum(current == f)) + 0.5)
                ne = math.floor(n * float(share[k]) + 0.5)
                me, ve, ie = e_par[k]
                mp, vp, ip = p_par[k]
                specs.append({"from": int(f), "to": int(t), "probability": probs[k].ravel().astype(np.float64),
                              "expand": ne, "patch": n - ne,
                              "expander": (self._cells(me, land), self._cells(ve, land) * self._cells(1, land), ie),
                              "patcher": (self._cells(mp, land), self._cells(vp, land) * self._cells(1, land), ip)})
            r = _core.sim_patches(current, specs, float(self.prune_factor), int(self.window[0]),
                                  int(self.window[1]), (int(seed) + step * 0x9E3779B9) & 0xFFFFFFFFFFFFFFFF)
            current = np.asarray(r["landscape"], dtype=np.int32)
            maps.append(np.where(nan, np.nan, current.astype(np.float64)))
            row = {"step": step}
            for k, (f, t) in enumerate(transitions):
                req = specs[k]["expand"] + specs[k]["patch"]
                done = int(r["expanded"][k] + r["patched"][k])
                row.update({f"requested_{f:g}_{t:g}": req, f"expanded_{f:g}_{t:g}": int(r["expanded"][k]),
                            f"patched_{f:g}_{t:g}": int(r["patched"][k]), f"patches_{f:g}_{t:g}": int(r["patches"][k])})
                if done < req:
                    warnings.warn(f"step {step}: {f:g} -> {t:g} made {done} of {req} changes "
                                  "(too few cells with probability)", stacklevel=4)
            rows.append(row)
        return Simulation(_as_maps(np.stack(maps), land.template, "landuse"), pd.DataFrame(rows).set_index("step"),
                          int(seed))

    def __repr__(self) -> str:
        return f"ExpanderPatcher(expansion={self.expansion!r}, prune_factor={self.prune_factor})"


def _like(land, values):
    t = land.template
    if t is None:
        return values
    return t.copy(data=values)
