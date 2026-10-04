# SPDX-License-Identifier: GPL-3.0-or-later
"""Land systems driven by the demand for goods and services: CLUMondo (Van Asselen & Verburg 2013).

A land system supplies several goods and services per cell (``Supply``,
CLUMondo's ``lusmatrix``); the demand (``ServiceDemand``) is given in their
units. Each year an iteration variable per demand moves the cells towards
the systems that supply more of a demand short of its target (in the
conversion order, ``lusconv``) until every demand is met within the allowed
deviations. The conversion resistance of the cells' own systems, the
conversion rules (allowed, by map, minimum and maximum age, autonomous
succession), restricted areas, neighbourhood effects and location
preferences shape where. The same maps as CLUMondo bit for bit
(github.com/VUEG/CLUMondo, its Win32 build): single precision where it uses
floats, its random factor from the Microsoft runtime's ``rand()``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import _core
from ._change import _class_value, _classes
from ._maps import read_map

__all__ = ["CLUMondo", "NeighbourhoodEffect", "ServiceDemand", "Supply"]


def _frame(table, index=None, columns=None, what="table") -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        out = table.copy()
    elif isinstance(table, dict) and all(isinstance(v, dict) for v in table.values()):
        out = pd.DataFrame.from_dict(table, orient="index")
    else:
        a = np.asarray(table, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"the {what} must be 2-D")
        out = pd.DataFrame(a, index=index, columns=columns)
    return out


class ServiceDemand:
    """The demand for goods and services at each time step (CLUMondo's ``demand.in``).

    Parameters
    ----------
    table
        A DataFrame with a row per time step (the first is the start) and a
        column per good or service, in their own units (tons, heads,
        hectares...); or an array with ``services``.
    services
        The names of an array's columns.

    Examples
    --------
    >>> import landschaft as ls
    >>> d = ls.simulate.ServiceDemand({"crops": [100, 104, 108], "cattle": [50, 50, 55]})
    >>> d.steps, d.services
    (2, ['crops', 'cattle'])
    """

    def __init__(self, table, services=None):
        if isinstance(table, dict) and not all(isinstance(v, dict) for v in table.values()):
            table = pd.DataFrame(table)
        self.table = _frame(table, columns=services, what="demand")
        if len(self.table) < 2:
            raise ValueError("a demand needs at least the start and one step")
        if not (self.table.to_numpy(dtype=np.float64)[1:] > 0).all():
            raise ValueError("the demands must be positive (CLUMondo's iteration divides by them)")

    @property
    def services(self) -> list:
        return list(self.table.columns)

    @property
    def classes(self) -> list:
        return self.services

    @property
    def steps(self) -> int:
        return len(self.table) - 1

    def at(self, step: int, services: list) -> np.ndarray:
        row = self.table.iloc[step]
        missing = set(services) - set(row.index)
        if missing:
            raise ValueError(f"no demand for {sorted(missing)}")
        return np.array([float(row[c]) for c in services])

    def __repr__(self) -> str:
        return f"ServiceDemand({self.steps} steps, {len(self.services)} services)"


class Supply:
    """What one cell of each land system supplies of each good or service (CLUMondo's ``lusmatrix``).

    Parameters
    ----------
    table
        A DataFrame with a row per land system (its value in the map) and a
        column per good or service; or ``{step: table}`` for a supply that
        changes over time (CLUMondo's dynamic ``lusmatrix``: each table holds
        from its step until the next one given; the first must be step 1).

    Examples
    --------
    >>> import landschaft as ls
    >>> s = ls.simulate.Supply({0: {"crops": 0.0}, 1: {"crops": 1.0}, 2: {"crops": 3.0}})
    >>> s.systems, s.services
    ([0, 1, 2], ['crops'])
    """

    def __init__(self, table):
        if isinstance(table, dict) and table and all(isinstance(v, pd.DataFrame) for v in table.values()):
            self.tables = {int(k): self._clean(v) for k, v in sorted(table.items())}
            if 1 not in self.tables:
                raise ValueError("a supply by step must give step 1")
        else:
            self.tables = {1: self._clean(_frame(table, what="supply"))}
        first = self.tables[1]
        for t in self.tables.values():
            if list(t.index) != list(first.index) or list(t.columns) != list(first.columns):
                raise ValueError("every step's supply must have the same systems and services")

    @staticmethod
    def _clean(t: pd.DataFrame) -> pd.DataFrame:
        t = t.copy()
        t.index = [_class_value(v) for v in t.index]
        return t.astype(np.float64)

    @property
    def systems(self) -> list:
        return list(self.tables[1].index)

    @property
    def services(self) -> list:
        return list(self.tables[1].columns)

    def at(self, step: int) -> pd.DataFrame:
        """The table in force at a step (1 = the first simulated year)."""
        return self.tables[max(k for k in self.tables if k <= max(step, 1))]

    def of(self, landscape, *, step: int = 1, nodata=None) -> pd.Series:
        """The goods and services a map of land systems supplies."""
        v = read_map(landscape, nodata).as_float().ravel()
        t = self.at(step)
        counts = np.array([np.sum(v == s) for s in t.index], dtype=np.float64)
        return pd.Series(counts @ t.to_numpy(), index=t.columns, name="supply")

    def __repr__(self) -> str:
        return f"Supply({len(self.systems)} systems, {len(self.services)} services)"


@dataclass
class NeighbourhoodEffect:
    """CLUMondo's neighbourhood function (``neighmat.txt`` and ``alloc2.reg``).

    For each system ``i`` it applies to, the suitability of cells whose
    current system is in ``from_`` gets ``weight x (constant + sum_k b_k x
    share_k)``, where ``share_k`` is the weighted share of system ``k`` in
    the kernel around the cell (over the cells with data), recomputed every
    year from the current map.

    Parameters
    ----------
    systems
        ``{system: {"weight": w, "kernel": radius or square array, "from":
        [current systems], "constant": c, "coefficients": {k: b}}}``. The
        kernel is a radius (a square of ones) or the weights (odd side);
        default radius 1. ``from`` defaults to every system. As in CLUMondo,
        the share of a system is only measured for systems listed here with a
        positive weight (0 for the others).
    """

    systems: dict

    def _parts(self, systems: list):
        index = {s: k for k, s in enumerate(systems)}
        out = {}
        for key, spec in self.systems.items():
            i = index.get(_class_value(key))
            if i is None:
                raise ValueError(f"neighbourhood for an unknown land system {key!r}")
            kernel = spec.get("kernel", 1)
            if np.isscalar(kernel):
                r = int(kernel)
                kernel = np.ones((2 * r + 1, 2 * r + 1), dtype=np.float32)
            kernel = np.ascontiguousarray(kernel, dtype=np.float32)
            frm = spec.get("from")
            frm = np.ones(len(systems), dtype=bool) if frm is None else np.isin(
                np.arange(len(systems)), [index[_class_value(f)] for f in frm])
            coefs = [(index[_class_value(k)], np.float32(b)) for k, b in spec.get("coefficients", {}).items()]
            out[i] = {"weight": np.float32(spec.get("weight", 1.0)), "kernel": kernel, "from": frm,
                      "constant": float(spec.get("constant", 0.0)), "coefficients": coefs}
        return out


@dataclass
class CLUMondo:
    """CLUMondo's demand-driven allocation of land systems.

    Parameters
    ----------
    supply
        :class:`Supply` (or its table): what each land system supplies per
        cell; its systems are the model's.
    order
        CLUMondo's ``lusconv``: per service, the rank of each system (higher
        supplies more; -1: the system does not take part in that service's
        changes). Default: the rank of the systems' supply of the service.
    weights
        Weight of each service in the iteration variables (``{service: w}``
        or a list; default 1).
    deviation
        ``"percent"``: the deviations are percentages of the demand;
        ``"absolute"``: in the demand's units.
    mean_deviation, max_deviation
        Largest mean deviation over the services and largest deviation of
        one service accepted. ``max_deviation`` may also be ``{service:
        percent}``: each service within its own (the 2017 executable's
        iteration mode 2; ``mean_deviation`` is then not used).
    kinds
        ``{service: "exact" | "minimum" | "maximum"}`` (default exact): a
        minimum is met by any supply above it, a maximum by any below (the
        2017 executable's demand types -1 and 101).
    speed
        Starting step of the iteration (0.001-0.9; CLUMondo's default 0.05).
    max_loops
        Iterations per year before giving up (the simulation stops there, as
        CLUMondo).
    neighbourhood
        :class:`NeighbourhoodEffect`.
    preferences
        Location-specific additions to the suitability: ``{system: (map,
        weight)}`` (CLUMondo's ``locspec`` files; a map per step may be
        given as ``{step: map}``).
    patches
        ``{system: (radius, threshold, min_cells)}``: the 2017 executable's
        patches. Each cell newly allocated to the system in an iteration
        brings along the cells of the square window of that radius that may
        convert to it and whose suitability reaches the threshold (whatever
        their resistance and the demand); a patch smaller than
        ``min_cells`` is dropped, seed included. Written from the manual
        (its source is not published).

    Notes
    -----
    The conversion resistance of each system is the model's
    ``Rules(elasticity=...)``; the conversion matrix uses CLUMondo's codes
    (see :class:`Rules`); ``Rules(history=n)`` draws the starting ages as
    CLUMondo does (``rand() % n + 1``, the runtime's generator with the
    run's seed; CLUMondo's own runs use seed 1).
    """

    supply: object
    order: object = None
    weights: object = None
    deviation: str = "percent"
    mean_deviation: float = 0.5
    max_deviation: float = 1.0
    speed: float = 0.05
    max_loops: int = 20000
    kinds: dict | None = None
    neighbourhood: NeighbourhoodEffect | None = None
    preferences: dict | None = None
    patches: dict | None = None
    uses_elasticity = True
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not isinstance(self.supply, Supply):
            self.supply = Supply(self.supply)
        if self.deviation not in ("percent", "absolute"):
            raise ValueError("deviation must be 'percent' or 'absolute'")
        if not 0.001 <= self.speed <= 0.9:
            raise ValueError("speed must be between 0.001 and 0.9")
        if isinstance(self.max_deviation, dict) and self.deviation != "percent":
            raise ValueError("a deviation per service is a percentage (deviation='percent')")
        bad = set((self.kinds or {}).values()) - {"exact", "minimum", "maximum"}
        if bad:
            raise ValueError(f"unknown demand kinds {sorted(bad)}: use 'exact', 'minimum' or 'maximum'")

    # ------------------------------------------------------------------ tables

    def lusconv(self) -> pd.DataFrame:
        """The conversion order (systems x services)."""
        systems, services = self.supply.systems, self.supply.services
        if self.order is None:
            t = self.supply.at(1)
            return t.rank(method="dense").astype(int) - 1
        o = _frame(self.order, index=systems, columns=services, what="order")
        o.index = [_class_value(v) for v in o.index]
        return o.loc[systems, services].astype(int)

    def _weights(self) -> np.ndarray:
        services = self.supply.services
        w = self.weights
        if w is None:
            return np.ones(len(services), dtype=np.float32)
        if isinstance(w, dict):
            return np.array([w.get(s, 1.0) for s in services], dtype=np.float32)
        w = np.asarray(w, dtype=np.float32)
        if w.size != len(services):
            raise ValueError("one weight per service")
        return w

    # ------------------------------------------------------------------ simulation

    def simulate(self, model, landscape, drivers=None, *, seed=None, nodata=None):
        from ._dynamic import current_map, with_dynamic
        from ._glm import LogisticRegression
        from ._landsystems import LandSystemStack
        from ._model import Simulation, _as_maps

        stack = landscape if isinstance(landscape, LandSystemStack) else None
        if stack is not None:
            landscape = stack.map
        land = read_map(landscape, nodata)
        shape = land.shape
        values = land.as_float().ravel()
        ncell = values.size
        systems = self.supply.systems
        services = self.supply.services
        index = {s: k for k, s in enumerate(systems)}
        found = {_class_value(c) for c in _classes([land])}
        if not found <= set(systems):
            raise ValueError(f"the landscape has land systems without supply: {sorted(found - set(systems))}")
        demand = model.demand
        if not isinstance(demand, ServiceDemand):
            raise TypeError("CLUMondo needs a ServiceDemand")
        rules = model.rules
        nan = np.isnan(values)
        region = np.zeros(ncell, dtype=np.int32)
        if rules.restrictions is not None:
            r = read_map(rules.restrictions).as_float().ravel()
            if r.size != ncell:
                raise ValueError("the restrictions map differs in size from the landscape")
            region[np.nan_to_num(r, nan=0.0) != 0] = -9998
        region[nan] = -9999
        oldco = np.full(ncell, -9999, dtype=np.int32)
        oldco[~nan] = [index[v] for v in values[~nan]]
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        age = self._ages(rules, ncell, seed)
        allowed, allow_maps = self._allowed(rules, systems, ncell)
        resistance = rules.elasticities(systems).astype(np.float32)
        lusconv = np.ascontiguousarray(self.lusconv().to_numpy(), dtype=np.int32)
        weight = self._weights()
        dem_elas = np.zeros(len(services), dtype=np.float32)
        nb = self.neighbourhood._parts(systems) if self.neighbourhood is not None else {}
        lr = model.potential if isinstance(model.potential, LogisticRegression) else None
        extra = self._extensions(systems, services, shape)
        extra.update(self._role_rules(rules, systems, allowed, stack, land))
        max_dev = np.float32(extra.pop("max_dev"))
        maps = [values.reshape(shape)]
        ages = [age.copy()]
        rows = []
        changed = np.zeros(ncell, dtype=bool)
        base, last = None, None
        for step in range(1, demand.steps + 1):
            d_step = _at_step(drivers, step)
            if model.dynamic:  # dynamic variables: recomputed from the map of every step
                lmap = current_map(maps[-1], land.template)
                base = self._base_suitability(model.potential, lr, with_dynamic(model.dynamic, d_step, lmap), lmap,
                                              systems, region, shape)
            elif base is None or d_step is not last:  # drivers given per step: recomputed when they change
                base = self._base_suitability(model.potential, lr, d_step, landscape, systems, region, shape)
                last = d_step
            suit = base.copy()
            if nb:
                self._add_neighbourhood(suit, nb, oldco, region, shape)
            if self.preferences:
                self._add_preferences(suit, systems, region, step)
            lus = np.ascontiguousarray(self.supply.at(step).to_numpy(), dtype=np.float32)
            r = _core.sim_clumondo_year(oldco, region, age, suit, allow_maps, allowed, lusconv, lus, weight,
                                        resistance, demand.at(step, services), dem_elas,
                                        0 if self.deviation == "percent" else 1, np.float32(self.mean_deviation),
                                        max_dev, np.float32(self.speed), int(self.max_loops), **extra)
            row = {"step": step, "loops": r["loops"], "converged": r["converged"],
                   "max_deviation": float(r["max_diff"]), "mean_deviation": float(r["mean_diff"]),
                   **{f"supply_{s}": v for s, v in zip(services, r["supply"])},
                   **{f"cells_{s}": int(c) for s, c in zip(systems, r["cells"])}}
            rows.append(row)
            if not r["converged"]:
                warnings.warn(f"no solution in step {step}: the demand was not met within the deviations in "
                              f"{self.max_loops} iterations; the simulation stops there (as CLUMondo)",
                              stacklevel=4)
                break
            dem_elas = r["dem_elas"].astype(np.float32)
            newco = r["newco"].astype(np.int32)
            changed |= (newco != oldco) & (region > -9999)
            out = np.full(ncell, np.nan)
            out[~nan] = np.asarray(systems, dtype=np.float64)[newco[~nan]]
            maps.append(out.reshape(shape))
            age = r["age"].astype(np.int32)
            ages.append(age.copy())
            oldco = newco
        da = _as_maps(np.stack(maps), land.template, "landuse")
        age_maps = np.stack([np.where(nan, np.nan, a).reshape(shape) for a in ages])
        sim = Simulation(da, pd.DataFrame(rows).set_index("step"), int(seed), age_maps[-1])
        sim.ages = _as_maps(age_maps, land.template, "age")
        sim.changed = np.where(nan, False, changed).reshape(shape)
        sim.supply_table = self.supply
        sim.demand_table = demand.table
        sim.stack = stack
        sim.roles = rules.roles
        return sim

    def _role_rules(self, rules, systems: list, allowed: np.ndarray, stack, land) -> dict:
        """Rules given by role (or pair) compiled into the conversion matrix, resistances and caps."""
        from ._maps import cell_area_of

        n = len(systems)
        layers_of = stack.names_of if stack is not None else (lambda v: (str(v),))
        rm = rules.roles.matrix(systems, layers_of) if rules.roles is not None else None

        def by_role(role):
            if rm is None:
                raise ValueError(f"rules by role ({role!r}) need Rules(roles=...)")
            return rm == role

        def index(v):
            if stack is not None and isinstance(v, (tuple, list, str)):
                v = stack.code_of(v)
            else:
                v = _class_value(v)
            if v not in systems:
                raise ValueError(f"unknown land system {v!r}")
            return systems.index(v)

        def mask(key):
            if isinstance(key, str):
                return by_role(key)
            m = np.zeros((n, n), dtype=bool)
            m[index(key[0]), index(key[1])] = True
            return m

        if isinstance(rules.allowed, dict):
            for key, code in rules.allowed.items():
                allowed[mask(key)] = int(code)
        out = {}
        pr = np.zeros((n, n), dtype=np.float32)
        for k, role in enumerate(rules.conversion_order or []):
            pr[by_role(role)] = np.float32(0.1 * k)
        for key, v in (rules.transition_resistance or {}).items():
            pr[mask(key)] = np.float32(v)
        np.fill_diagonal(pr, 0)
        if pr.any():
            out["pair_resistance"] = np.ascontiguousarray(pr)
        if rules.caps:
            area = cell_area_of(land, None)
            pairs = []
            for cap in rules.caps:
                m = np.zeros((n, n), dtype=bool)
                if cap.pairs:
                    for f, t in cap.pairs:
                        m[index(f), index(t)] = True
                else:
                    m = by_role(cap.role).copy()
                np.fill_diagonal(m, False)
                pairs.append(m)
            out["cap_pairs"] = np.ascontiguousarray(np.stack(pairs).astype(np.uint8))
            out["cap_cells"] = np.array([c.limit(area) for c in rules.caps], dtype=np.int64)
        for sc in rules.succession or []:
            if sc.role is None:
                allowed[index(sc.from_), index(sc.to)] = 1000 + int(sc.after_years)
                continue
            roles = by_role(sc.role)
            for i in range(n):
                targets = [j for j in range(n) if roles[i, j] and j != i]
                if not targets:
                    continue
                size = [len(layers_of(systems[j])) for j in targets]
                best = [j for j, k in zip(targets, size) if k == min(size)]
                if len(best) > 1:
                    raise ValueError(f"succession by role: {systems[i]!r} has several targets "
                                     f"{[systems[j] for j in best]}: give from_ and to")
                allowed[i, best[0]] = 1000 + int(sc.after_years)
        bad = [systems[i] for i in range(n) if (allowed[i] > 1000).sum() > 1]
        if bad:
            raise ValueError(f"a land system can change autonomously into one other only: {bad}")
        return out

    def _extensions(self, systems: list, services: list, shape) -> dict:
        out = {"max_dev": 0.0 if isinstance(self.max_deviation, dict) else float(self.max_deviation)}
        if isinstance(self.max_deviation, dict):
            missing = set(services) - set(self.max_deviation)
            if missing:
                raise ValueError(f"no deviation for services {sorted(missing)}")
            out["dev_limit"] = np.array([self.max_deviation[v] for v in services], dtype=np.float32)
        if self.kinds:
            codes = {"exact": 0, "minimum": 1, "maximum": 2}
            out["kind"] = np.array([codes[self.kinds.get(v, "exact")] for v in services], dtype=np.int32)
        if self.patches:
            n = len(systems)
            rad, thr, mn = np.zeros(n, np.int32), np.zeros(n, np.float32), np.zeros(n, np.int32)
            for key, (r, t, m) in self.patches.items():
                i = systems.index(_class_value(key))
                rad[i], thr[i], mn[i] = int(r), np.float32(t), int(m)
            out.update(nx=int(shape[1]), patch_radius=rad, patch_threshold=thr, patch_min=mn)
        return out

    def _ages(self, rules, ncell: int, seed: int) -> np.ndarray:
        h = rules.history
        if h is None:
            return np.ones(ncell, dtype=np.int32)
        if np.isscalar(h):
            return np.asarray(_core.sim_clumondo_random_age(ncell, int(h), int(seed) & 0xFFFFFFFF), dtype=np.int32)
        a = read_map(h).as_float().ravel()
        if a.size != ncell:
            raise ValueError("the history map differs in size from the landscape")
        return np.nan_to_num(a, nan=1.0).astype(np.int32)

    def _allowed(self, rules, systems: list, ncell: int):
        n = len(systems)
        m = rules.matrix(systems)
        m = np.ones((n, n), dtype=np.int32) if m is None else m.astype(np.int32)
        maps = []
        for (f, t), mp in (rules.allow_maps or {}).items():
            i, j = systems.index(_class_value(f)), systems.index(_class_value(t))
            v = read_map(mp).as_float().ravel()
            if v.size != ncell:
                raise ValueError("a conversion map differs in size from the landscape")
            if len(maps) >= 98:
                raise ValueError("at most 98 conversion maps")
            m[i, j] = 2 + len(maps)
            maps.append(np.nan_to_num(v, nan=0.0).astype(np.float32))
        bad = (m > 1) & (m < 100) & ~np.isin(m, 2 + np.arange(len(maps)))
        if bad.any():
            raise ValueError("conversion codes 2-99 refer to maps: give them in Rules(allow_maps=...)")
        allow_maps = np.ascontiguousarray(np.stack(maps) if maps else np.zeros((0, ncell), dtype=np.float32))
        return np.ascontiguousarray(m), allow_maps

    def _base_suitability(self, potential, lr, drivers, landscape, systems, region, shape) -> np.ndarray:
        from ._glm import _drivers

        ncell = region.size
        suit = np.empty((len(systems), ncell), dtype=np.float32)
        if lr is not None:
            maps = _drivers(drivers)
            for i, s in enumerate(systems):
                if s not in lr._fits:
                    raise ValueError(f"the potential gives no suitability for land system {s!r}")
                b0, names, coefs = lr._terms(s)
                x = np.stack([maps[n].as_float().ravel() for n in names]).astype(np.float32) if names else \
                    np.zeros((0, ncell), dtype=np.float32)
                suit[i] = _core.sim_clumondo_logit(region, np.ascontiguousarray(x), b0,
                                                   np.asarray(coefs, dtype=np.float32))
            return suit
        p = potential.predict(drivers, landscape=landscape) if drivers is not None \
            else potential.predict(landscape=landscape)
        pcls = [_class_value(c) for c in p[p.dims[0]].values]
        flat = p.values.reshape(p.shape[0], -1)
        for i, s in enumerate(systems):
            if s not in pcls:
                raise ValueError(f"the potential gives no suitability for land system {s!r}")
            suit[i] = flat[pcls.index(s)].astype(np.float32)
        suit[:, region < 0] = -9999
        return suit

    @staticmethod
    def _add_neighbourhood(suit, nb, oldco, region, shape):
        o2, r2 = oldco.reshape(shape), region.reshape(shape)
        shares = {u: _core.sim_clumondo_fraction(o2, r2, u, p["kernel"]).ravel()
                  for u, p in nb.items() if p["weight"] > 0}
        zero = np.zeros(oldco.size, dtype=np.float32)
        inside = region >= 0
        for i, p in nb.items():
            if not p["weight"] > 0:
                continue
            t = np.full(oldco.size, np.float32(p["constant"]), dtype=np.float32)
            for k, b in p["coefficients"]:
                t = t + b * shares.get(k, zero)
            cells = inside & p["from"][np.clip(oldco, 0, None)]
            suit[i, cells] = suit[i, cells] + p["weight"] * t[cells]

    def _add_preferences(self, suit, systems, region, step):
        inside = region >= 0
        for key, (mp, w) in self.preferences.items():
            i = systems.index(_class_value(key))
            m = _at_step(mp, step) if isinstance(mp, dict) else mp
            v = read_map(m).as_float().ravel().astype(np.float32)
            suit[i, inside] = suit[i, inside] + v[inside] * np.float32(w)

    def __repr__(self) -> str:
        return (f"CLUMondo({len(self.supply.systems)} systems, {len(self.supply.services)} services, "
                f"deviation={self.deviation!r})")


def _at_step(x, step: int):
    """A value given per step (``{step: value}``, the last one at or before ``step``) or fixed."""
    if isinstance(x, dict) and x and all(isinstance(k, (int, np.integer)) for k in x):
        keys = [k for k in x if k <= step]
        return x[max(keys)] if keys else x[min(x)]
    return x

