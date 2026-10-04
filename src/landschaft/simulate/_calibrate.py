# SPDX-License-Identifier: GPL-3.0-or-later
"""Pattern-oriented calibration of a land change model.

The model's free settings are searched for the values whose simulation of
an observed period best reproduces the observed map: its change cell by
cell (figure of merit), its change in the neighbourhood (fuzzy
similarity), its quantities, its landscape pattern (landscape metrics) or
any measure the user gives. Each evaluation averages a few replicates run
with the same seeds, so that two settings are compared on the same draws.
"""

from __future__ import annotations

import copy
import itertools
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._threads import resolve, threaded
from ._maps import read_map

__all__ = ["Calibration", "calibrate"]

_METHODS = ("nelder-mead", "differential-evolution", "grid", "random")
_PATH = re.compile(r"^([A-Za-z_]\w*)((?:\.[A-Za-z_]\w*|\[[^\]]+\])*)$")


def _parse(path: str) -> list:
    """``"allocation.seeds[2]"`` -> ``["allocation", ".seeds", "[2]"]``."""
    m = _PATH.match(path)
    if not m:
        raise ValueError(f"cannot read the parameter path {path!r}")
    return [m.group(1), *re.findall(r"\.[A-Za-z_]\w*|\[[^\]]+\]", m.group(2))]


def _key(text: str):
    import ast

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _set(model, path: str, value):
    """A copy of the model with the setting at ``path`` changed (the pieces along the way copied)."""
    parts = _parse(path)
    root = copy.copy(model)
    obj = root
    for k, part in enumerate(parts):
        last = k == len(parts) - 1
        if part.startswith("["):
            key = _key(part[1:-1])
            if isinstance(obj, (list, tuple)):
                key = int(key)
            if last:
                obj[key] = value
                return root
            child = copy.copy(obj[key])
            obj[key] = child
        else:
            name = part.lstrip(".")
            if not hasattr(obj, name):
                raise ValueError(f"{type(obj).__name__} has no setting {name!r} ({path})")
            if last:
                setattr(obj, name, value)
                return root
            child = getattr(obj, name)
            if isinstance(child, dict):
                child = dict(child)
            elif isinstance(child, list):
                child = list(child)
            elif isinstance(child, tuple):
                child = list(child)
                setattr(obj, name, child)
                obj = child
                continue
            else:
                child = copy.copy(child)
            setattr(obj, name, child)
        obj = child
    return root


@dataclass
class Calibration:
    """The result of :func:`calibrate`.

    Attributes
    ----------
    best : dict
        The settings of lowest loss.
    loss : float
        Its loss.
    history : pandas.DataFrame
        Every evaluation: the settings, the loss and each target's part.
    model : Model
        The model with the best settings.
    simulation : Simulation
        Its first replicate.
    """

    best: dict
    loss: float
    history: pd.DataFrame
    model: object
    simulation: object
    settings: dict = field(default_factory=dict)
    _maps: tuple = field(default=None, repr=False)

    def validate(self, **settings):
        """Every comparison of the best simulation with the observed map (:func:`validate`)."""
        from ._validate import validate

        start, observed, nodata = self._maps
        return validate(start, observed, self.simulation.maps.isel(step=-1), nodata=nodata, **settings)

    def __repr__(self) -> str:
        best = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.best.items())
        return f"Calibration({best}; loss {self.loss:.4f} after {len(self.history)} evaluations)"


def _losses(targets, start, observed, simulated, nodata) -> dict:
    """Each target's loss (lower is better)."""
    from ._run import agreement
    from ._similarity import similarity_of_differences
    from ._validation import compare_maps, futures_validation

    out = {}
    fut = None
    for name, spec in targets.items():
        fn = spec[0] if isinstance(spec, tuple) else None
        if fn is not None:
            out[name] = float(fn(observed, simulated))
        elif name == "figure_of_merit":
            fom = agreement(observed, simulated, start, nodata=nodata)["figure_of_merit"]
            out[name] = 1.0 - (0.0 if np.isnan(fom) else fom)
        elif name.startswith("similarity"):
            w = int(name.split("_")[1]) if "_" in name else 5
            t = similarity_of_differences(start, observed, simulated, windows=[w], nodata=nodata)
            v = float(t["minimum"].iloc[0])
            out[name] = 1.0 - (0.0 if np.isnan(v) else v)
        elif name == "kappa":
            out[name] = 1.0 - compare_maps(observed, simulated, nodata=nodata)["kappa"]
        elif name in ("quantity", "allocation"):
            if fut is None:
                o, s = read_map(observed, nodata).as_float(), read_map(simulated, nodata).as_float()
                fut = futures_validation(o, s)
            out[name] = float(fut[f"total_{name}"])
        elif name == "lsm":
            raise ValueError("give the lsm target as {'lsm': ([metrics], weight)} or {'lsm': {'metrics': [...]}}")
        else:
            raise ValueError(f"unknown target {name!r}")
    return out


def _lsm_loss(metrics, level):
    from ..lsm import landscape_metrics

    def loss(observed, simulated):
        tables = []
        for m in (observed, simulated):
            v = read_map(m).as_float()
            ints = np.where(np.isnan(v), -9999, v).astype(np.int32)
            t = landscape_metrics(ints, level=level, metric=list(metrics), nodata=-9999)
            tables.append(t.set_index(["level", "class", "id", "metric"])["value"])
        o, s = tables[0].align(tables[1])
        o, s = o.to_numpy(dtype=float), s.to_numpy(dtype=float)
        ok = ~(np.isnan(o) & np.isnan(s))
        o, s = np.nan_to_num(o[ok]), np.nan_to_num(s[ok])
        return float(np.mean(np.abs(s - o) / np.maximum(np.abs(o), 1e-12))) if o.size else 0.0

    return loss


def _targets(targets) -> dict:
    """``{name: (function or None, weight)}``."""
    out = {}
    for name, spec in (targets or {"figure_of_merit": 1.0, "similarity": 1.0}).items():
        if name == "lsm":
            if isinstance(spec, dict):
                fn, w = _lsm_loss(spec["metrics"], spec.get("level", "class")), float(spec.get("weight", 1.0))
            else:
                metrics, w = spec
                fn, w = _lsm_loss(metrics, "class"), float(w)
            out[name] = (fn, w)
        elif callable(spec):
            out[name] = (spec, 1.0)
        elif isinstance(spec, tuple) and callable(spec[0]):
            out[name] = (spec[0], float(spec[1]))
        else:
            out[name] = (None, float(spec))
    return out


@threaded
def calibrate(model, start, observed, drivers=None, *, params: dict, targets: dict | None = None,
              method: str = "nelder-mead", replicates: int = 1, seed: int | None = None, max_evals: int = 50,
              grid_points: int = 5, x0: dict | None = None, n_workers: int | None = None, nodata=None) -> Calibration:
    """Search the settings of a model that best reproduce an observed map.

    Parameters
    ----------
    model
        A :class:`Model`, its settings named by ``params`` as paths into its
        pieces (``"allocation.step_size"``, ``"allocation.seeds[2]"``,
        ``"rules.elasticity[1]"``); or a function ``build(**params)``
        returning a model.
    start, observed
        The map the simulation starts from and the observed map at its end.
    drivers
        The drivers of the model.
    params
        ``{name: (low, high)}`` (integers if both bounds are) or ``{name:
        [values]}`` (searched by ``method="grid"``).
    targets
        ``{name: weight}``: ``"figure_of_merit"`` (1 - FoM), ``"similarity"``
        or ``"similarity_<window>"`` (1 - fuzzy similarity of the
        differences, window 5 by default), ``"kappa"``, ``"quantity"``,
        ``"allocation"`` (disagreement); ``"lsm": ([metrics], weight)`` or
        ``{"metrics": [...], "level": "landscape", "weight": w}`` (mean
        relative difference of the landscape metrics); or ``{name: f}`` /
        ``{name: (f, weight)}`` with ``f(observed, simulated)`` a loss.
        Default: figure of merit and similarity, equally weighted.
    method : {"nelder-mead", "differential-evolution", "grid", "random"}
        The search (scipy's optimizers; ``"grid"``: every combination, each
        range cut into ``grid_points`` values; ``"random"``: ``max_evals``
        Latin hypercube draws).
    replicates
        Simulations per evaluation (their mean loss), seeds ``seed + r`` in
        every evaluation.
    max_evals
        Evaluations at most (not for ``"grid"``).
    x0
        Starting settings for Nelder-Mead (default: the middle of the ranges).

    Returns
    -------
    Calibration

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> S = ls.simulate
    >>> rng = np.random.default_rng(0)
    >>> d1, d2 = rng.random((30, 30)), rng.random((30, 30))
    >>> land = np.ones((30, 30))
    >>> demand = S.AreaDemand(np.array([[900, 0], [800, 100]]), classes=[1, 2])
    >>> def build(w):
    ...     pot = S.RasterPotential({1: np.full((30, 30), 0.5), 2: w * d1 + (1 - w) * d2})
    ...     return S.Model(pot, demand, S.Ordered(order=[2, 1], stochastic=False))
    >>> observed = build(0.3).run(land, seed=1).maps[-1]
    >>> cal = S.calibrate(build, land, observed, params={"w": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
    ...                   targets={"figure_of_merit": 1}, method="grid")
    >>> cal.best
    {'w': 0.3}
    """
    from scipy import optimize

    from ._model import Model

    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    if not params:
        raise ValueError("give the settings to calibrate (params)")
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    build = model if callable(model) and not isinstance(model, Model) else None
    if build is None and not isinstance(model, Model):
        raise TypeError("calibrate() takes a Model or a function building one")
    names = list(params)
    kinds = {}
    for k, v in params.items():
        if isinstance(v, tuple) and len(v) == 2:
            lo, hi = v
            if not hi > lo:
                raise ValueError(f"{k}: the upper bound must exceed the lower one")
            kinds[k] = "int" if isinstance(lo, (int, np.integer)) and isinstance(hi, (int, np.integer)) else "float"
        elif isinstance(v, (list, np.ndarray)) and len(v):
            kinds[k] = "values"
        else:
            raise ValueError(f"{k}: give (low, high) or a list of values")
    if method != "grid" and any(kinds[k] == "values" for k in names):
        raise ValueError("lists of values are searched with method='grid'")
    tg = _targets(targets)
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    seeds = [int(seed) + r for r in range(int(replicates))]
    cache: dict = {}
    rows: list = []
    sims: dict = {}

    def model_for(values):
        if build is not None:
            return build(**values)
        m = model
        for k, v in values.items():
            m = _set(m, k, v)
        return m

    def clean(values):
        out = {}
        for k in names:
            v = values[k]
            if kinds[k] == "int":
                v = round(float(v))
                v = int(min(max(v, params[k][0]), params[k][1]))
            elif kinds[k] == "float":
                v = float(min(max(float(v), params[k][0]), params[k][1]))
            out[k] = v
        return out

    def evaluate(values, inner=None, record=True):
        values = clean(values)
        key = tuple(values[k] for k in names)
        if key in cache:
            return cache[key][0]
        m = model_for(values)
        parts_all = []
        first = None
        for s in seeds:
            sim = m.run(start, drivers, seed=s, nodata=nodata, n_threads=inner)
            if first is None:
                first = sim
            parts_all.append(_losses(tg, start, observed, sim.maps.isel(step=-1), nodata))
        parts = {k: float(np.mean([p[k] for p in parts_all])) for k in tg}
        loss = float(sum(tg[k][1] * parts[k] for k in tg))
        row = {**values, "loss": loss, **{f"loss_{k}": v for k, v in parts.items()}}
        cache[key] = (loss, row)
        sims[key] = (m, first)
        if record:
            rows.append(row)
        return loss

    def in_order(points):  # the parallel evaluations, recorded in the order of the points
        _parallel(points, lambda p, inner: evaluate(p, inner, record=False), n_workers)
        seen = set()
        for p in points:
            key = tuple(clean(p)[k] for k in names)
            if key not in seen:
                seen.add(key)
                rows.append(cache[key][1])

    cont = [k for k in names if kinds[k] != "values"]
    lo = np.array([params[k][0] for k in cont], dtype=float)
    hi = np.array([params[k][1] for k in cont], dtype=float)

    def from_unit(u):
        return dict(zip(cont, lo + np.clip(np.asarray(u, dtype=float), 0, 1) * (hi - lo)))

    if method == "grid":
        axes = []
        for k in names:
            if kinds[k] == "values":
                axes.append(list(params[k]))
            elif kinds[k] == "int":
                axes.append(sorted({round(float(x)) for x in np.linspace(params[k][0], params[k][1], grid_points)}))
            else:
                axes.append(list(np.linspace(params[k][0], params[k][1], grid_points)))
        in_order([dict(zip(names, p)) for p in itertools.product(*axes)])
    elif method == "random":
        sampler = __import__("scipy.stats", fromlist=["qmc"]).qmc.LatinHypercube(d=len(cont), seed=seed)
        in_order([from_unit(u) for u in sampler.random(int(max_evals))])
    elif method == "differential-evolution":
        pop = 5
        maxiter = max(1, int(max_evals) // (pop * len(cont)) - 1)
        optimize.differential_evolution(lambda u: evaluate(from_unit(u)), [(0, 1)] * len(cont), seed=seed,
                                        popsize=pop, maxiter=maxiter, polish=False, tol=0, updating="immediate")
    else:
        start_x = np.full(len(cont), 0.5)
        if x0:
            start_x = np.array([(float(x0[k]) - params[k][0]) / (params[k][1] - params[k][0]) for k in cont])
        optimize.minimize(lambda u: evaluate(from_unit(u)), start_x, method="Nelder-Mead",
                          options={"maxfev": int(max_evals), "initial_simplex": _simplex(start_x),
                                   "xatol": 1e-3, "fatol": 1e-6})
    history = pd.DataFrame(rows)
    history.index.name = "evaluation"
    i = int(np.argmin([r["loss"] for r in rows]))  # the first of equal losses
    best = {k: rows[i][k] for k in names}
    m, sim = sims[tuple(best[k] for k in names)]
    return Calibration(best, float(rows[i]["loss"]), history, m, sim,
                       {"method": method, "replicates": int(replicates), "seed": int(seed),
                        "targets": {k: v[1] for k, v in tg.items()}, "params": {k: str(v) for k, v in params.items()}},
                       (start, observed, nodata))


def _simplex(x):
    """A starting simplex spanning a quarter of each range (scipy's default is too small on the unit cube)."""
    n = len(x)
    pts = [np.asarray(x, dtype=float)]
    for k in range(n):
        p = np.array(x, dtype=float)
        p[k] = p[k] + 0.25 if p[k] + 0.25 <= 1 else p[k] - 0.25
        pts.append(p)
    return np.array(pts)


def _parallel(points, evaluate, n_workers):  # evaluate(point, inner threads)
    total = resolve(None)
    workers = min(len(points), total) if n_workers is None else max(1, int(n_workers))
    inner = max(1, total // workers)
    if workers == 1:
        for p in points:
            evaluate(p, inner)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda p: evaluate(p, inner), points))
