# SPDX-License-Identifier: GPL-3.0-or-later
"""MOLUSCE's weights of evidence and cellular automaton simulator.

MOLUSCE (the QGIS plugin, GPL >= 2) fits a weights-of-evidence model per
transition, persistence included, on factors cut into user bins
(``WoeManager``); at each iteration its simulator (``Simulator``) predicts
for every cell the transition of largest total weight and its confidence,
and makes the transitions of the observed cross table where the
confidence is highest. Ported line by line: the same maps as MOLUSCE with
``numpy.random.seed(seed)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import _core
from ._change import _class_value
from ._maps import read_map

__all__ = ["MolusceCA", "MolusceWoE"]

_EPSILON = 4 * np.finfo(float).eps


def _sigmoid(x):
    x = np.maximum(-100, np.minimum(x, 100))
    return 1 / (1 + np.exp(-x))


def _binary_woe(fm: np.ndarray, sm: np.ndarray, unit_cell: float) -> tuple[float, float]:
    """W+ and W- of a binary pattern (MOLUSCE's ``_binary_woe``, on the unmasked cells)."""
    a = 1.0 * len(fm) / unit_cell
    b = 1.0 * int(fm.sum()) / unit_cell
    n = 1.0 * int(sm.sum())
    nb = 1.0 * int((fm & sm).sum())
    if a == 0 or b == 0 or n == 0:
        raise ValueError("a factor category, the sites or the map is empty")
    if nb > n or n >= a:
        raise ValueError("the unit cell is too big for the data")
    p_sf = nb / n + _EPSILON
    p_nsf = (b - nb) / (a - n) + _EPSILON
    p_snf = (n - nb) / n + _EPSILON
    p_nsnf = (a - b - n + nb) / (a - n) + _EPSILON
    return math.log(p_sf / p_nsf), math.log(p_snf / p_nsnf)


def _woe(values: np.ndarray, sites: np.ndarray, unit_cell: float):
    """MOLUSCE's multiclass ``woe``: categories and their weights w+ + sum(w-) - w- (halved for two)."""
    cats = np.unique(values)
    if cats.size < 2:
        raise ValueError("a factor needs at least two categories where it is defined")
    w = [_binary_woe(values == c, sites, unit_cell) for c in cats]
    total_minus = sum(x[1] for x in w)
    wmap = [x[0] + total_minus - x[1] for x in w]
    if cats.size == 2:
        wmap = [v / 2 for v in wmap]
    return cats, np.array(wmap)


class MolusceWoE:
    """MOLUSCE's weights of evidence (``WoeManager``): one model per transition, persistence included.

    Each factor is cut into categories (``bins``: ``(-inf, b0) -> 1``,
    ``[b0, b1) -> 2``... ``[bn, inf) -> n + 2``; or its integer values
    themselves); for each transition code observed between the two maps
    (the cells that went from ``i`` to ``j``, ``i == j`` included), every
    category gets ``W+ + sum(W-) - W-`` over the factor's categories (halved
    with two categories), and a cell's total weight is the sum over the
    factors. A cell of class ``i`` is predicted to take the transition
    ``i -> j`` of largest total weight; its confidence is ``100 x
    (sigmoid(largest) - sigmoid(second largest))``, truncated.

    Parameters
    ----------
    bins
        ``{driver: [bounds]}`` (sorted); drivers not named are used as
        integer categories.
    unit_cell
        MOLUSCE's unit cell (areas in unit cells).

    Notes
    -----
    MOLUSCE keeps the largest weight seen *before* the largest as the
    second, which depends on the order of the classes: here the second
    largest weight (tested against MOLUSCE with ``tools/molusce_fixed.patch``).
    MOLUSCE also stores the factors as 8-bit integers, wrapping integer
    categories outside 0-255 and truncating fractions; here integer
    categories keep their values and fractional values need bins.
    """

    def __init__(self, bins: dict | None = None, unit_cell: float = 1):
        self.bins = {str(k): list(v) for k, v in (bins or {}).items()}
        for k, b in self.bins.items():
            if b != sorted(b) or len(set(b)) != len(b):
                raise ValueError(f"the bins of {k!r} must be sorted and distinct")
        self.unit_cell = float(unit_cell)
        self.classes: list = []
        self.codes: list[int] = []
        self.drivers: list[str] = []
        self.weights: dict = {}
        self._maps: dict = {}

    def _categories(self, name: str, v: np.ndarray) -> np.ndarray:
        """A factor's categories (NaN stays NaN)."""
        if name in self.bins:
            v = np.asarray(v, dtype=np.float64)
            return _core.sim_molusce_categories(v.ravel(), self.bins[name]).reshape(v.shape)
        ok = ~np.isnan(v)
        if (np.abs(v[ok] - np.round(v[ok])) > 0).any():
            raise ValueError(f"the driver {name!r} has fractional values: give its bins")
        return v

    def encode(self, f, t) -> int:
        m = len(self.classes)
        return self.classes.index(_class_value(f)) * m + self.classes.index(_class_value(t))

    def decode(self, code: int) -> tuple:
        m = len(self.classes)
        return self.classes[code // m], self.classes[code % m]

    def fit(self, t0, t1, drivers, *, nodata=None) -> MolusceWoE:
        """Fit the weights on the change between two maps.

        Parameters
        ----------
        t0, t1
            The maps before and after (the same classes).
        drivers
            The factors (a Dataset, ``dict`` or DataArray with a band per
            variable), in the order MOLUSCE gets them.
        """
        from ._woe import _drivers

        a, b = read_map(t0, nodata).as_float(), read_map(t1, nodata).as_float()
        if a.shape != b.shape:
            raise ValueError("the maps differ in shape")
        mask = np.isnan(a) | np.isnan(b)
        ca = sorted({_class_value(c) for c in np.unique(a[~np.isnan(a)])})
        cb = sorted({_class_value(c) for c in np.unique(b[~np.isnan(b)])})
        if set(ca) != set(cb):
            raise ValueError("the two maps must have the same classes (as MOLUSCE)")
        self.classes = ca
        m = len(ca)
        lut = {c: k for k, c in enumerate(ca)}
        change = np.full(a.shape, -1, dtype=np.int64)
        ok = ~mask
        change[ok] = [lut[_class_value(x)] * m + lut[_class_value(y)] for x, y in zip(a[ok], b[ok])]
        maps = _drivers(drivers)
        self.drivers = list(maps)
        factors = {k: self._categories(k, maps[k].as_float()) for k in self.drivers}
        self.codes = sorted(int(c) for c in np.unique(change[ok]))
        self.weights, self._maps = {}, {}
        for code in self.codes:
            sites_mask = mask.copy()
            total = np.zeros(a.shape)
            total_mask = np.zeros(a.shape, dtype=bool)
            self.weights[code] = {}
            for k in self.drivers:
                f = factors[k]
                sites_mask = sites_mask | np.isnan(f)  # MOLUSCE's masks_identity: the masks pile up
                keep = ~sites_mask
                cats, w = _woe(f[keep], change[keep] == code, self.unit_cell)
                self.weights[code][k] = (cats, w)
                layer = np.zeros(a.shape)
                for c, x in zip(cats, w):
                    layer[f == c] = x
                total = total + layer
                total_mask |= sites_mask
            self._maps[code] = np.where(total_mask, np.nan, total)
        return self

    def weights_table(self) -> pd.DataFrame:
        """The weight of each category of each driver, per transition."""
        rows = []
        for code, per in self.weights.items():
            f, t = self.decode(code)
            for k, (cats, w) in per.items():
                rows += [{"from": f, "to": t, "driver": k, "category": float(c), "weight": float(x)}
                         for c, x in zip(cats, w)]
        return pd.DataFrame(rows)

    def total(self, drivers=None) -> dict:
        """Total weight map of each transition code (NaN where a factor or the maps had none).

        Without ``drivers``, the maps of the fit (MOLUSCE's); with them, the
        weights applied to these drivers (dynamic variables).
        """
        if drivers is None:
            return self._maps
        from ._woe import _drivers

        maps = _drivers(drivers)
        missing = set(self.drivers) - set(maps)
        if missing:
            raise ValueError(f"drivers lack {sorted(missing)}")
        factors = [self._categories(k, maps[k].as_float()) for k in self.drivers]
        shape = factors[0].shape
        weights = [[(np.asarray(self.weights[c][k][0], dtype=np.float64).tolist(),
                     np.asarray(self.weights[c][k][1], dtype=np.float64).tolist()) for k in self.drivers]
                   for c in self.weights]
        totals = _core.sim_molusce_total(np.stack([f.ravel() for f in factors]), weights)
        return {c: totals[j].reshape(shape) for j, c in enumerate(self.weights)}

    def potentials(self) -> dict:
        """MOLUSCE's transition potentials: ``sigmoid(total weight) x 100`` as 8-bit integers, per (from, to)."""
        return {self.decode(c): np.where(np.isnan(v), np.nan, (_sigmoid(np.nan_to_num(v)) * 100).astype(np.uint8))
                for c, v in self._maps.items()}

    def prediction(self, state, drivers=None, *, nodata=None) -> tuple[np.ndarray, np.ndarray]:
        """The transition each cell is predicted to take and its confidence (0-100).

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The predicted class (NaN where the cell has none) and the
            confidence (NaN outside the cells with a class of the model).
        """
        if not self.codes:
            raise RuntimeError("fit the model first")
        s = read_map(state, nodata).as_float()
        total = self.total(drivers)
        codes = list(total)
        pred, conf = _core.sim_molusce_prediction_woe(
            s.ravel(), [float(c) for c in self.classes], [int(c) for c in codes],
            np.stack([np.asarray(total[c], dtype=np.float64).ravel() for c in codes]))
        return pred.reshape(s.shape), conf.reshape(s.shape)

    def predict(self, drivers=None, *, landscape=None, nodata=None):
        """Total weight per transition as ``probability_i_to_j`` layers (``sigmoid``), for the other allocations."""
        import xarray as xr

        total = self.total(drivers)
        names, layers = [], []
        for code, v in total.items():
            f, t = self.decode(code)
            if f == t:
                continue
            names.append(f"probability_{f:g}_to_{t:g}")
            layers.append(np.where(np.isnan(v), np.nan, _sigmoid(np.nan_to_num(v))))
        return xr.DataArray(np.stack(layers), dims=("band", "y", "x"), coords={"band": names})

    def probability(self, landscape=None, drivers=None):
        return self.predict(drivers, landscape=landscape)

    def __repr__(self) -> str:
        return f"MolusceWoE({len(self.codes)} transitions, drivers={self.drivers})"


def _counts(demand) -> dict:
    """{(from, to): cells} from a cross table (rows from, columns to) or a dict."""
    if isinstance(demand, pd.DataFrame):
        return {(_class_value(f), _class_value(t)): int(demand.loc[f, t])
                for f in demand.index for t in demand.columns}
    if isinstance(demand, dict):
        return {(_class_value(f), _class_value(t)): int(v) for (f, t), v in demand.items()}
    raise TypeError("MolusceCA takes the cells of each transition: a cross table (rows from, columns to) "
                    "or {(from, to): cells}")


@dataclass
class MolusceCA:
    """MOLUSCE's simulator: each iteration makes the observed transitions where the model is most confident.

    At each iteration the potential predicts every cell's transition and
    its confidence (:class:`MolusceWoE`, or any potential with a layer per
    class: the class of largest potential, confidence ``100 x`` the gap to
    the second); then, for each pair of classes in order, the ``n`` cells
    (the demand: the cross table of the observed change, the same every
    iteration) predicted to take that transition with the highest
    confidence plus ``U(0, 1) / 1000`` change.

    Parameters
    ----------
    iterations
        Iterations (steps) to simulate.

    Notes
    -----
    The random numbers are numpy's legacy generator (``RandomState(seed)``),
    as MOLUSCE draws them after ``numpy.random.seed(seed)``. Among cells of
    equal confidence and draw (which the 1/1000 rarely leaves), the later
    cells change first (MOLUSCE leaves the order to numpy's quicksort).
    """

    iterations: int = 1

    def _prediction(self, potential, state, drivers, classes):
        if hasattr(potential, "prediction"):
            return potential.prediction(state, drivers)
        p = potential.predict(drivers, landscape=state) if drivers is not None else potential.predict(landscape=state)
        names = [_class_value(c) for c in p[p.dims[0]].values]
        s = read_map(state).as_float()
        # a class without a layer: NaN (-inf, as a layer without values)
        layers = np.stack([np.asarray(p.values[names.index(c)], dtype=np.float64).ravel() if c in names
                           else np.full(s.size, np.nan) for c in classes])
        pred, conf = _core.sim_molusce_prediction_classes(s.ravel(), layers, [float(c) for c in classes])
        return pred.reshape(s.shape), conf.reshape(s.shape)

    def simulate(self, model, landscape, drivers=None, *, seed=None, nodata=None):
        from ._dynamic import current_map, with_dynamic
        from ._model import Simulation, _as_maps

        counts = _counts(model.demand)
        land = read_map(landscape, nodata)
        state = land.as_float()
        classes = list(getattr(model.potential, "classes", None) or
                       sorted({c for pair in counts for c in pair} | {_class_value(c) for c in np.unique(
                           state[~np.isnan(state)])}))
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        rng = _core.NumpyRandomState(int(seed) % 2**32)
        pairs = [(float(f), float(t), int(n)) for (f, t), n in counts.items()]
        maps = [state.copy()]
        rows = []
        for it in range(1, int(self.iterations) + 1):
            lmap = current_map(state, land.template)
            if model.dynamic:
                drv = with_dynamic(model.dynamic, drivers, lmap)
            else:  # MOLUSCE's WoE keeps the maps of its fit (the factors it got at the start)
                drv = None if isinstance(model.potential, MolusceWoE) else drivers
            pred, conf = self._prediction(model.potential, lmap, drv, classes)
            new, changed = _core.sim_molusce_ca_step(state.ravel(), np.asarray(pred, dtype=np.float64).ravel(),
                                                     np.asarray(conf, dtype=np.float64).ravel(), pairs, rng)
            row = {"step": it, **{f"changed_{f:g}_{t:g}": int(n) for f, t, n in changed}}
            state = new.reshape(state.shape)
            maps.append(state.copy())
            rows.append(row)
        return Simulation(_as_maps(np.stack(maps), land.template, "landuse"),
                          pd.DataFrame(rows).set_index("step").fillna(0).astype(int), int(seed))

    def __repr__(self) -> str:
        return f"MolusceCA(iterations={self.iterations})"
