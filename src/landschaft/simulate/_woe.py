# SPDX-License-Identifier: GPL-3.0-or-later
"""Weights of evidence for land change (Bonham-Carter 1994), as Dinamica EGO calibrates them.

Implemented from the method's publications (Agterberg & Bonham-Carter 1990;
Soares-Filho et al. 2002, 2009) and checked against Dinamica EGO's outputs:
the ranges of a continuous variable come from nested buffers of
``increment`` (a cumulative count of cells ``A`` plotted against
``A * exp(W+)``) cut by a line-generalising algorithm (a new range when the
direction from the last break deviates from that of the previous range by
more than ``tolerance`` degrees, or the range exceeds ``max_delta`` cells;
none below ``min_delta`` cells); a categorical variable has a range per
category. The weight of a range is W+ = ln(P(B|D) / P(B|~D)), its contrast
W+ - W-, significant when larger than 1.96 standard deviations; the
probabilities of the transitions from a class at a cell are
exp(S_j) / (1 + sum_k exp(S_k)), S the sums of the weights of its ranges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._threads import threaded
from ._change import _class_value, _classes
from ._maps import Map, read_map

__all__ = ["WeightsOfEvidence", "weights_of_evidence"]


@dataclass
class _Variable:
    name: str
    categorical: bool = False
    increment: float | str | None = "auto"
    min_delta: float = 1.0
    max_delta: float = 100000.0
    tolerance: float = 5.0
    breaks: list | None = None  # given range limits


def _spec(name: str, value, defaults: dict) -> _Variable:
    if isinstance(value, _Variable):
        return value
    if value == "categorical":
        return _Variable(name, categorical=True)
    if isinstance(value, (list, tuple, np.ndarray)):
        return _Variable(name, breaks=sorted(float(v) for v in value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _Variable(name, **{**defaults, "increment": float(value)})
    if isinstance(value, dict):
        opts = {**defaults, **value}
        if opts.pop("categorical", False):
            return _Variable(name, categorical=True)
        unknown = set(opts) - {"increment", "min_delta", "max_delta", "tolerance", "breaks"}
        if unknown:
            raise ValueError(f"unknown options for {name!r}: {sorted(unknown)}")
        return _Variable(name, **opts)
    if value is None:
        return _Variable(name, **defaults)
    raise TypeError(f"cannot read the range settings of {name!r}: {value!r}")


def _drivers(drivers, nodata=None) -> dict[str, Map]:
    """Explanatory maps by name: a Dataset, a dict of maps, or a DataArray with a band per variable."""
    if isinstance(drivers, xr.Dataset):
        return {str(k): read_map(v, nodata) for k, v in drivers.data_vars.items()}
    if isinstance(drivers, dict):
        return {str(k): read_map(v, nodata) for k, v in drivers.items()}
    if isinstance(drivers, xr.DataArray):
        if drivers.ndim == 2:
            return {str(drivers.name or "variable"): read_map(drivers, nodata)}
        dim = drivers.dims[0]
        names = [str(v) for v in drivers[dim].values] if dim in drivers.coords else \
            [f"variable_{k + 1}" for k in range(drivers.shape[0])]
        return {n: read_map(drivers[k], nodata) for k, n in enumerate(names)}
    raise TypeError("drivers must be a Dataset, a dict of maps or a DataArray with a band per variable")


def _weights(nb, b, n, a, fix):
    """W+, W-, contrast and significance of a range: nb transitions in b possible cells, of n in a."""
    return _core.sim_woe_weights(float(nb), float(b), float(n), float(a), bool(fix))


def _generalise(possible: np.ndarray, executed: np.ndarray, v: _Variable, fix: bool) -> list[int]:
    """Range limits (bucket indices) of a continuous variable: the line generalisation of (A, A exp(W+))."""
    return _core.sim_woe_generalise(np.asarray(possible, dtype=np.int64), np.asarray(executed, dtype=np.int64),
                                    float(v.min_delta), float(v.max_delta), float(v.tolerance), bool(fix))


@dataclass
class _Layout:
    """How a variable's cells are bucketed: increments from ``start`` or categories."""

    categorical: bool
    classes: np.ndarray
    start: float = 0.0
    increment: float = 1.0
    n: int = 0
    buckets: np.ndarray | None = None

    def limit(self, k: int) -> float:
        if self.categorical:
            return float(self.classes[k]) if k < self.classes.size else float(self.classes[-1]) + 1.0
        return self.start + k * self.increment


def _layout(m: Map, v: _Variable, cells: np.ndarray) -> _Layout:
    """Buckets of a variable for one initial class: its extent (or categories) over ``cells``, the cells of
    that class with a value in both maps (as Dinamica EGO takes them)."""
    f = m.as_float()
    inside = f[cells & np.isfinite(f)]
    if v.categorical:
        classes = np.unique(inside)
        b = _core.sim_woe_buckets(m.values, m.nodata, True, classes, 0.0, 1.0, int(classes.size))
        return _Layout(True, classes, n=int(classes.size), buckets=b)
    if v.breaks is not None:
        edges = np.asarray(v.breaks, dtype=np.float64)
        return _Layout(False, edges, buckets=None, n=edges.size - 1)
    lo, hi = (float(inside.min()), float(inside.max())) if inside.size else (0.0, 0.0)
    inc = v.increment
    if inc in (None, "auto"):
        inc = (hi - lo) / 100 if hi > lo else 1.0
    inc = float(inc)
    if not inc > 0:
        raise ValueError(f"the increment of {v.name!r} must be > 0")
    start = min(0.0, math.floor(lo / inc) * inc)
    top = (math.floor(hi / inc) + 1) * inc
    n = max(1, round((top - start) / inc))
    b = _core.sim_woe_buckets(m.values, m.nodata, False, np.empty(0), start, inc, n)
    return _Layout(False, np.empty(0), start, inc, n, b)


def _explicit_buckets(m: Map, edges: np.ndarray) -> np.ndarray:
    f = m.as_float().ravel()
    k = np.searchsorted(edges, f, side="right") - 1
    k[(f < edges[0]) | (f >= edges[-1]) | np.isnan(f)] = -1
    return k.astype(np.int32)


class WeightsOfEvidence:
    """Weights of evidence of land use transitions, calibrated on two maps (Dinamica EGO's method).

    Parameters
    ----------
    ranges
        How each explanatory variable is cut into ranges, by name:
        ``"categorical"`` (a range per category); a number (the increment of
        a continuous variable, the other settings by default); a ``dict`` with
        ``increment``, ``min_delta``, ``max_delta``, ``tolerance``; or a list
        of range limits. Variables not named are continuous with the
        defaults below; ``categorical`` names more categorical ones.
    transitions
        The transitions ``(from, to)`` to calibrate (default: every one
        observed between the maps).
    increment
        Default buffer increment of continuous variables, in the variable's
        units (``"auto"``: a hundredth of its range).
    min_delta, max_delta
        Smallest and largest range, in cells (Dinamica's defaults: 1 and
        100 000).
    tolerance
        Tolerance angle, degrees (default 5): a smaller angle gives more,
        narrower ranges.
    categorical
        Names of categorical variables.

        ``ranges`` may also be keyed by transition, ``{(1, 2): {"dist": 30,
        "soil": "categorical"}, (1, 3): {...}}``: each transition with its own
        variables and settings (Dinamica's skeleton).
    fix_abnormal
        Give ranges with no transition, or only transitions, finite weights
        (``-ln(B N + 1)`` and ``ln(B (A - N) + 1)``, B the range's cells,
        N the transitions, A the possible cells) instead of 0.
    possible : {"any", "stay"}
        Cells where a transition was possible but did not happen: every
        other cell of the initial class (``"any"``, default; Dinamica's
        "potential transition criteria") or only those that kept their class.

    Examples
    --------
    >>> import landschaft as ls
    >>> lu = ls.load_raster(ls.simulate.example_data("sibuyan_landuse"))
    >>> drivers = ls.load_raster(ls.simulate.example_data("sibuyan_drivers"))
    >>> woe = ls.simulate.WeightsOfEvidence({"dist_roads": 250, "dist_sea": 250}, transitions=[(1, 3)])
    >>> woe = woe.fit(lu[0], lu[1], drivers[[0, 10]])
    >>> woe.table.columns[:5].tolist()
    ['Transition_From', 'Transition_To', 'Variable', 'Range_Lower_Limit', 'Range_Upper_Limit']
    """

    def __init__(self, ranges: dict | None = None, *, transitions=None, increment="auto", min_delta: float = 1.0,
                 max_delta: float = 100000.0, tolerance: float = 5.0, categorical=(), fix_abnormal: bool = False,
                 possible: str = "any"):
        if possible not in ("any", "stay"):
            raise ValueError("possible must be 'any' or 'stay'")
        self._defaults = {"increment": increment, "min_delta": float(min_delta), "max_delta": float(max_delta),
                          "tolerance": float(tolerance)}
        settings = dict(ranges or {})
        per_transition = {tuple(k): dict(v) for k, v in settings.items() if isinstance(k, tuple)}
        common = {k: v for k, v in settings.items() if not isinstance(k, tuple)}
        for name in categorical:
            common[name] = "categorical"
            for v in per_transition.values():
                if name in v:
                    v[name] = "categorical"
        self._common = common
        self._per_transition = per_transition
        if transitions is None and per_transition:
            transitions = list(per_transition)
        self.transitions = None if transitions is None else [tuple(t) for t in transitions]
        self.fix_abnormal = bool(fix_abnormal)
        self.possible = possible
        self.variables: list[str] = []
        self._plan: dict = {}        # (from, to) -> {var: _Variable}
        self._layouts: dict = {}     # (from, to, var) -> _Layout
        self._ranges: dict = {}      # (from, to, var) -> list of bucket limits
        self._rows: list = []
        self._weights: dict = {}     # (from, to, var) -> weight per range

    # ------------------------------------------------------------------ fitting

    @threaded
    def fit(self, t0, t1, drivers, *, nodata=None) -> WeightsOfEvidence:
        """Calibrate the ranges and weights from the map at the start (``t0``) and at the end (``t1``).

        Parameters
        ----------
        t0, t1
            Categorical maps of the two dates.
        drivers
            The explanatory variables on the same grid: an
            ``xarray.Dataset``, a ``dict`` of maps, or a DataArray with a
            band per variable (the band names name them).
        nodata
            Value marking missing cells in the land use maps.
        n_threads
            Threads (default: all cores but one).

        Returns
        -------
        WeightsOfEvidence
            ``self``, fitted.
        """
        a, b = read_map(t0, nodata), read_map(t1, nodata)
        maps = _drivers(drivers)
        if any(m.shape != a.shape for m in [b, *maps.values()]):
            raise ValueError("the maps and the drivers differ in shape")
        named = set(self._common) | {k for v in self._per_transition.values() for k in v}
        unknown = named - set(maps)
        if unknown:
            raise ValueError(f"no driver named {sorted(unknown)}")
        self.variables = sorted(maps)
        self._maps = maps
        trans = self.transitions
        if trans is None:
            cls = _classes([a, b])
            ct = _core.sim_crosstab(a.values, b.values, cls, cls, a.nodata, b.nodata)
            trans = [(_class_value(cls[i]), _class_value(cls[j])) for i, j in zip(*np.nonzero(ct)) if i != j]
        self.transitions = [(_class_value(f), _class_value(t)) for f, t in trans]
        self._plan = {}
        for f, t in self.transitions:
            own = self._per_transition.get((f, t))
            if own is None:
                own = {k: self._common.get(k) for k in self.variables}
            self._plan[(f, t)] = {k: _spec(k, v, self._defaults) for k, v in sorted(own.items())}
        final = _classes([b])
        valid = ~a.missing() & ~b.missing()
        layout_cache, counts = {}, {}
        self._layouts, self._ranges, self._weights, rows = {}, {}, {}, []
        for f, t in self.transitions:
            j = int(np.searchsorted(final, t))
            stay = int(np.searchsorted(final, f))
            has_t = j < final.size and final[j] == t
            has_stay = stay < final.size and final[stay] == f
            for name, spec in self._plan[(f, t)].items():
                sig = (f, name, spec.categorical, spec.increment,
                       None if spec.breaks is None else tuple(spec.breaks))
                if sig not in layout_cache:
                    lay = _layout(maps[name], spec, valid & (a.values == f))
                    if spec.breaks is not None:
                        lay.buckets = _explicit_buckets(maps[name], lay.classes)
                    layout_cache[sig] = lay
                lay = layout_cache[sig]
                self._layouts[(f, t, name)] = lay
                key = sig
                if key not in counts:
                    counts[key] = _core.sim_woe_counts(a.values, b.values, lay.buckets, lay.n, float(f), final,
                                                       a.nodata, b.nodata)
                c = counts[key]
                executed = c[:, j] if has_t else np.zeros(lay.n, np.int64)
                if self.possible == "any":
                    possible = c.sum(axis=1)
                else:
                    possible = executed + (c[:, stay] if has_stay else 0)
                if lay.categorical or spec.breaks is not None:
                    limits = list(range(1, lay.n + 1))
                else:
                    limits = _generalise(possible, executed, spec, self.fix_abnormal)
                self._ranges[(f, t, name)] = limits
                cb, cn = np.concatenate([[0], np.cumsum(possible)]), np.concatenate([[0], np.cumsum(executed)])
                big_a, big_n = float(cb[-1]), float(cn[-1])
                ws = []
                lo = 0
                for hi in limits:
                    nb_, b_ = cn[hi] - cn[lo], cb[hi] - cb[lo]
                    wp, contrast, sig_ = _weights(nb_, b_, big_n, big_a, self.fix_abnormal)
                    ws.append(wp)
                    lower = lay.limit(lo) if spec.breaks is None else float(lay.classes[lo])
                    upper = lay.limit(hi) if spec.breaks is None else float(lay.classes[hi])
                    rows.append((f, t, name, lower, upper, int(b_), int(nb_), wp, contrast, int(sig_)))
                    lo = hi
                self._weights[(f, t, name)] = ws
        self._rows = rows
        return self

    def _check(self):
        if not self._rows:
            raise RuntimeError("fit the weights of evidence first")

    @property
    def table(self) -> pd.DataFrame:
        """Ranges and weights, as Dinamica EGO's report: transition, variable, range limits, possible and
        executed transitions, weight (W+), contrast and its significance (1 = 95 %)."""
        self._check()
        return pd.DataFrame(self._rows, columns=["Transition_From", "Transition_To", "Variable", "Range_Lower_Limit",
                                                 "Range_Upper_Limit", "Possible_Transitions", "Executed_Transitions",
                                                 "Weight_Coefficient", "Contrast", "Significant"])

    @property
    def ranges(self) -> pd.DataFrame:
        """The range limits of each variable and transition."""
        return self.table[["Transition_From", "Transition_To", "Variable", "Range_Lower_Limit", "Range_Upper_Limit"]]

    # ------------------------------------------------------------------ range of each cell

    def _range_of_bucket(self, f, t, name) -> np.ndarray:
        limits = self._ranges[(f, t, name)]
        lay = self._layouts[(f, t, name)]
        out = np.full(lay.n, -1, np.int32)
        lo = 0
        for r, hi in enumerate(limits):
            out[lo:hi] = r
            lo = hi
        return out

    def _cell_buckets(self, f, t, name, m: Map | None):
        lay, spec = self._layouts[(f, t, name)], self._plan[(f, t)][name]
        if m is None:
            return lay.buckets
        if spec.breaks is not None:
            return _explicit_buckets(m, lay.classes)
        if lay.categorical:
            return _core.sim_woe_buckets(m.values, m.nodata, True, lay.classes, 0.0, 1.0, lay.n)
        k = _core.sim_woe_buckets(m.values, m.nodata, False, np.empty(0), lay.start, lay.increment, lay.n)
        out = k < 0
        if out.any():  # values beyond the calibrated extent take the first or the last range
            v = m.as_float().ravel()
            k = k.copy()
            k[out & (v < lay.start)] = 0
            k[out & (v >= lay.start + lay.n * lay.increment)] = lay.n - 1
        return k

    def _cell_ranges(self, f, t, name) -> np.ndarray:
        """The range of each cell for the correlation: values beyond the extent calibrated for the
        transition's initial class form categories of their own (above, below), as in Dinamica EGO."""
        lay, spec, m = self._layouts[(f, t, name)], self._plan[(f, t)][name], self._maps[name]
        rb = self._range_of_bucket(f, t, name)
        if lay.categorical:
            k = _core.sim_woe_buckets(m.values, m.nodata, True, lay.classes, 0.0, 1.0, lay.n)
            v = m.as_float().ravel()
            r = np.where(k >= 0, rb[np.maximum(k, 0)], -1)
            r[(k < 0) & np.isfinite(v)] = rb.max() + 1  # categories absent from the class's cells
            return r
        if spec.breaks is not None:
            k = _explicit_buckets(m, lay.classes)
            lo, hi = lay.classes[0], lay.classes[-1]
        else:
            k = _core.sim_woe_buckets(m.values, m.nodata, False, np.empty(0), lay.start, lay.increment, lay.n)
            lo, hi = lay.start, lay.start + lay.n * lay.increment
        v = m.as_float().ravel()
        r = np.where(k >= 0, rb[np.maximum(k, 0)], -1)
        r[(k < 0) & (v >= hi)] = rb.max() + 1
        r[(k < 0) & (v < lo)] = rb.max() + 2
        return r

    # ------------------------------------------------------------------ correlation

    @threaded
    def correlation(self) -> pd.DataFrame:
        """Association between the ranges of each pair of variables (Dinamica EGO's correlation report).

        For each transition and pair of variables, the cross-tabulation of
        their ranges over every cell where both have a value gives
        chi-square, Cramér's V, the contingency coefficient, the joint
        entropy (natural logarithms) and the joint information uncertainty
        ``2 (H(A) + H(B) - H(A, B)) / (H(A) + H(B))`` (Bonham-Carter 1994).
        Values of Cramér's V or the joint uncertainty above ~0.5 suggest
        dropping or merging one of the two variables.

        Returns
        -------
        pandas.DataFrame
        """
        self._check()
        rows = []
        for f, t in self.transitions:
            names = list(self._plan[(f, t)])
            for i, va in enumerate(names):
                for vb in names[i + 1:]:
                    ra, rb = self._cell_ranges(f, t, va), self._cell_ranges(f, t, vb)
                    ok = (ra >= 0) & (rb >= 0)
                    ra = np.where(ok, ra, -1).astype(np.int32)
                    rb = np.where(ok, rb, -1).astype(np.int32)
                    ca = np.unique(ra[ok]).astype(np.float64)
                    cb = np.unique(rb[ok]).astype(np.float64)
                    o = _core.sim_crosstab(ra, rb, ca, cb, -1.0, -1.0).astype(np.float64)
                    # the table has a row per range, and two more (below, above) when a variable has values
                    # beyond its calibrated extent: Cramer's V divides by min(rows, columns) - 1 of that table
                    dims = [len(self._ranges[(f, t, x)]) + (2 if (r >= len(self._ranges[(f, t, x)])).any() else 0)
                            for x, r in ((va, ra), (vb, rb))]
                    rows.append((f, t, va, vb, *_association(o, min(dims) - 1)))
        return pd.DataFrame(rows, columns=["Transition_From", "Transition_To", "First_Variable", "Second_Variable",
                                           "Chi_2", "Cramer", "Contingency", "Joint_Entropy", "Joint_Uncertainty"])

    # ------------------------------------------------------------------ probability

    @threaded
    def probability(self, landscape, drivers, *, transitions=None, dtype="float64", nodata=None):
        """Transition probabilities from the weights (Dinamica EGO's CalcWOfEProbabilityMap).

        With S the sum of the weights of the cell's range in every variable of
        a transition, the probability of the transition i -> j at a cell of
        class i is ``exp(S_ij) / (1 + sum_k exp(S_ik))`` over the transitions
        from i asked for together (``1 / (1 + exp(-S))`` for a single one); a
        variable missing at a cell adds nothing to its sum (as in Dinamica).

        Parameters
        ----------
        landscape
            The land use map the probabilities are for.
        drivers
            The explanatory variables (as in :meth:`fit`; they may have
            changed since).
        transitions
            The transitions (default: all calibrated).
        dtype : {"float64", "float32", "uint8", "uint16"}
            float: probabilities 0-1, NaN elsewhere; integer:
            ``trunc(p * max)``, 0 elsewhere (Dinamica's default ``uint8``).
        nodata
            Value marking missing cells in the landscape.
        n_threads
            Threads (default: all cores but one).

        Returns
        -------
        xarray.DataArray
            ``(transition, y, x)``, the transitions named
            ``"probability_<from>_to_<to>"``.
        """
        self._check()
        land = read_map(landscape, nodata)
        maps = _drivers(drivers)
        trans = self.transitions if transitions is None else [tuple(x) for x in transitions]
        layers, names = [], []
        for f, t in trans:
            if (f, t) not in self._plan:
                raise ValueError(f"the transition {f}->{t} was not calibrated")
            missing = set(self._plan[(f, t)]) - set(maps)
            if missing:
                raise ValueError(f"drivers lack {sorted(missing)}")
            bks, wts = [], []
            for name in self._plan[(f, t)]:
                rb = self._range_of_bucket(f, t, name)
                bks.append(self._cell_buckets(f, t, name, maps[name]))
                wts.append(np.asarray(self._weights[(f, t, name)], dtype=np.float64)[np.maximum(rb, 0)])
            layers.append(_core.sim_woe_probability(land.values, land.nodata, float(f), bks, wts).reshape(land.shape))
            names.append(f"probability_{f}_to_{t}")
        sums = np.stack(layers)
        out = np.empty_like(sums)
        froms = [f for f, _ in trans]
        for f in dict.fromkeys(froms):
            idx = [k for k, g in enumerate(froms) if g == f]
            s = sums[idx].reshape(len(idx), -1)
            out[idx] = _core.sim_woe_normalise(s).reshape((len(idx), *land.shape))
        if dtype in ("uint8", "uint16"):
            top = np.iinfo(dtype).max
            out = np.where(np.isnan(out), 0, np.trunc(out * top)).astype(dtype)
        elif dtype == "float32":
            out = out.astype(np.float32)
        elif dtype != "float64":
            raise ValueError("dtype must be float64, float32, uint8 or uint16")
        t = land.template
        if t is not None:
            da = xr.DataArray(out, dims=("transition", *t.dims),
                              coords={"transition": names, **{k: t.coords[k] for k in t.dims}}, name="probability")
            try:
                from ..io import _with_georef

                da = _with_georef(da, transform=t.rio.transform())
                if t.rio.crs is not None:
                    da = _with_georef(da, crs=t.rio.crs)
            except Exception:
                pass
            return da
        return xr.DataArray(out, dims=("transition", "y", "x"), coords={"transition": names}, name="probability")

    potential = probability

    def __repr__(self) -> str:
        state = f"{len(self.transitions or [])} transitions, {len(self.variables)} variables" if self._rows \
            else "not fitted"
        return f"WeightsOfEvidence({state})"


def _association(o: np.ndarray, k: int | None = None) -> tuple:
    total = o.sum()
    if total == 0 or min(o.shape) < 2:
        return (math.nan,) * 5
    e = o.sum(axis=1, keepdims=True) * o.sum(axis=0, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((o - e) ** 2 / e))
    if k is None:
        k = min(o.shape) - 1
    cramer = math.sqrt(chi2 / (total * k))
    contingency = math.sqrt(chi2 / (chi2 + total))
    p = o / total

    def entropy(q):
        q = q[q > 0]
        return float(-np.sum(q * np.log(q)))

    hab, ha, hb = entropy(p.ravel()), entropy(p.sum(axis=1)), entropy(p.sum(axis=0))
    jiu = 2 * (ha + hb - hab) / (ha + hb) if ha + hb > 0 else math.nan
    return chi2, cramer, contingency, hab, jiu


def weights_of_evidence(t0, t1, drivers, ranges: dict | None = None, **kwargs) -> WeightsOfEvidence:
    """Calibrate weights of evidence in one call: ``WeightsOfEvidence(ranges, **kwargs).fit(t0, t1, drivers)``.

    See :class:`WeightsOfEvidence` for the settings.
    """
    n_threads = kwargs.pop("n_threads", None)
    nodata = kwargs.pop("nodata", None)
    return WeightsOfEvidence(ranges, **kwargs).fit(t0, t1, drivers, nodata=nodata, n_threads=n_threads)
