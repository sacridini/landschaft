# SPDX-License-Identifier: GPL-3.0-or-later
"""Restoration planning: how much of every converted cell to restore, for the most benefit
(the restoration problem of plangea, Strassburg et al. 2020, International Institute for
Sustainability; GPL >= 3).

The decision is the share x_i of each cell restored, between 0 and the share that can be (its
anthropic land, times a limit). The benefits:

- the reduction of species' extinction risk (or ecosystems' risk of collapse): a species with habitat
  A of a potential P has the risk 1 - (A / P)^z (the species-area relationship, z = 0.25), and
  restoring gives it habitat, A = A_now + sum_i g_si x_i;
- linear benefits (carbon, per km² restored);
- ecological intactness, by its derivative (:func:`landschaft.intactness_gradient`).

The risk reduction is concave in x, so the problem (the most weighted benefit within area targets and a
budget) is solved to a proven tolerance by simplicial decomposition: plans that are vertices of the
set of plans, each the solution of a linear program over the cells with only the problem's rows,
mixed by a small smooth master problem; the bound is the Frank-Wolfe one. The cheapest plan for a
benefit is a bisection on the budget. plangea's procedure (a ratio of benefits to costs per cell, the
target in steps with the benefits recomputed between them: a heuristic whose result depends on the
number of steps) is ``method="steps"``, reproduced for comparison.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

from .._save import Saveable
from ._milp import OptimizationProblem

__all__ = ["RestorationProblem", "RestorationSolution", "RiskReduction", "ecosystem_areas",
           "eval_risk_reduction", "restoration_problem"]

PLANGEA_Z = 0.25  # plangea's exponent of the species-area relationship
PLANGEA_DELTA = 1e-6  # plangea's finite difference of the risk (in habitat area), method="steps"
EXRISK_EPS = 1e-8  # plangea's multi_exrisk_aggregate: an updated habitat this close to 0 is 0


# ----------------------------------------------------------------------------- inputs

def _grid_values(data, what: str) -> tuple[xr.DataArray | None, np.ndarray]:
    if isinstance(data, (str, bytes)) or hasattr(data, "__fspath__"):
        import landschaft as ls

        data = ls.load_raster(data)
    if isinstance(data, xr.DataArray):
        v = np.asarray(data.values, dtype=np.float64)
        if v.ndim == 3 and v.shape[0] == 1:
            v = v[0]
            data = data.isel({data.dims[0]: 0}, drop=True)
        if v.ndim != 2:
            raise ValueError(f"{what}: a single-band raster")
        return data, v
    v = np.asarray(data, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"{what}: a 2-D raster or array")
    return None, v


def _per_species(values, names: np.ndarray, default: float, what: str) -> np.ndarray:
    """A number for every species, one per species or a mapping species -> value (default for the others)."""
    if isinstance(values, Mapping):
        unknown = set(map(str, values)) - set(names)
        if unknown:
            raise ValueError(f"{what}: no species {sorted(unknown)}")
        out = np.array([float(values.get(s, default)) for s in names])
    else:
        v = np.asarray(values, dtype=np.float64)
        if v.ndim > 1 or (v.ndim == 1 and v.size != names.size):
            raise ValueError(f"{what}: a number, one per species ({names.size}) or a mapping species -> value")
        out = np.broadcast_to(v, names.shape).astype(np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{what}: NaN or infinite")
    return out


def _on_units(p: RestorationProblem, values, what: str) -> np.ndarray:
    """A number, a raster or an array on the problem's grid, or one value per unit: the units' values."""
    if np.isscalar(values):
        return np.full(p.index.size, float(values))
    if isinstance(values, xr.DataArray) or (isinstance(values, np.ndarray) and values.ndim == 2) or isinstance(
            values, (str, bytes)) or hasattr(values, "__fspath__"):
        _, v = _grid_values(values, what)
        if v.shape != p.shape:
            raise ValueError(f"{what}: the grid {v.shape} is not the problem's {p.shape}")
        return v.ravel()[p.index]
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size != p.index.size:
        raise ValueError(f"{what}: one value per planning unit ({p.index.size}) or a raster")
    return v


@dataclass(frozen=True)
class _Risk:
    """A risk benefit: species (or ecosystems) with habitat now, potential habitat and the gain per unit
    of every unit restored (already discounted)."""

    name: str
    weight: float
    species: tuple
    current: np.ndarray
    potential: np.ndarray
    gain: sp.csr_matrix  # species x units: habitat per unit share restored, times (1 - discount)
    hab: sp.csr_matrix  # species x units: the same per unit of the cell's area (plangea's hab_values)
    z: np.ndarray  # one exponent per species
    discount: np.ndarray


@dataclass(frozen=True)
class _Linear:
    name: str
    weight: float
    coef: np.ndarray  # benefit per unit share restored
    layer: np.ndarray  # plangea's per-cell layer (method="steps")
    intactness: tuple | None = None  # (hfi grid, reduction grid, radius, gamma, beta, z, cell_size)


@dataclass(frozen=True)
class _Targets:
    rows: sp.csr_matrix  # targets x units: area per unit share restored
    rhs: np.ndarray
    sense: tuple
    labels: tuple


@dataclass(frozen=True)
class RestorationProblem:
    """A restoration problem (see :func:`restoration_problem`), built up by its ``add_*`` methods; every
    method returns a new problem."""

    template: xr.DataArray | None = field(repr=False)
    shape: tuple
    index: np.ndarray = field(repr=False)  # the units' cells (row-major)
    upper: np.ndarray = field(repr=False)  # the share of each unit that can be restored
    area: np.ndarray = field(repr=False)  # each unit's area (km²)
    benefits: tuple = ()
    targets: tuple = ()
    cost: np.ndarray | None = field(default=None, repr=False)  # per unit share restored
    cost_layer: np.ndarray | None = field(default=None, repr=False)  # per km² (plangea's layer)
    budget: float | None = None
    objective: tuple = ("max_benefit",)
    solver: tuple = ("highs", {})

    @property
    def n(self) -> int:
        """The number of planning units (cells that can be restored)."""
        return int(self.index.size)

    # --- benefits
    def add_extinction_risk_benefit(self, areas, z=PLANGEA_Z, weight: float = 1.0,
                                    restoration_discount=0.0, name: str | None = None) -> RestorationProblem:
        """The reduction of the species' extinction risk (or of ecosystems' risk of collapse).

        A species with habitat A of a potential P has the risk 1 - (A / P)^z (0 at A >= P); restoring a
        share x_i of a cell gives it ``g_i x_i`` more habitat, g the gain of
        :func:`landschaft.sdm.area_of_habitat` (or of :func:`ecosystem_areas`). The benefit is the sum
        over the species of the risk now minus the risk after.

        Parameters
        ----------
        areas
            A :class:`~landschaft.sdm.HabitatAreas` with its gain (``area_of_habitat`` given the
            original vegetation), on the problem's grid; the species that are not ``valid`` are left
            out.
        z
            The exponent of the species-area relationship, in (0, 1): a number for every species
            (default 0.25, plangea's), one per species (in the order of ``areas.table``) or a mapping
            species -> z (species left out of it get 0.25).
        weight
            The benefit's weight in the objective.
        restoration_discount
            The share of the habitat value that restored land does NOT deliver, in [0, 1] (0: restored
            land is habitat in full, as Strassburg et al. 2020 assume): the gain becomes
            g (1 - discount). A number, one per species (in the order of ``areas.table``) or a mapping
            species -> discount. It is not a financial discount rate.
        name
            The benefit's name (default ``"extinction_risk"``, numbered when repeated).
        """
        gain = getattr(areas, "gain", None)
        table = getattr(areas, "table", None)
        if gain is None or table is None:
            raise ValueError("areas: a HabitatAreas with its gain (area_of_habitat given the original vegetation)")
        if gain.shape[1] != self.shape[0] * self.shape[1]:
            raise ValueError("areas: not on the problem's grid")
        names = table["species"].astype(str).to_numpy()
        zs = _per_species(z, names, PLANGEA_Z, "z")
        if (zs <= 0).any() or (zs >= 1).any():
            raise ValueError("z must be in (0, 1)")
        d = _per_species(restoration_discount, names, 0.0, "restoration_discount")
        if not np.isfinite(d).all() or (d < 0).any() or (d > 1).any():
            raise ValueError("restoration_discount must be in [0, 1]")
        keep = np.flatnonzero(table["valid"].to_numpy(bool) & (table["potential"].to_numpy(np.float64) > 0))
        g = sp.csr_matrix(gain)[keep][:, self.index].tocsr()
        g.sort_indices()
        area = self.area
        hab = g.copy()
        hab.data = hab.data / area[hab.indices]
        gd = g.multiply(sp.csr_matrix((1.0 - d[keep])[:, None])).tocsr()
        gd.sort_indices()
        risk = _Risk(name=self._name(name, "extinction_risk"), weight=float(weight), species=tuple(names[keep]),
                     current=table["current"].to_numpy(np.float64)[keep].copy(),
                     potential=table["potential"].to_numpy(np.float64)[keep].copy(), gain=gd, hab=hab,
                     z=zs[keep].copy(), discount=d[keep])
        return replace(self, benefits=self.benefits + (risk,))

    def add_linear_benefit(self, values, weight: float = 1.0, name: str | None = None,
                           per: str = "area") -> RestorationProblem:
        """A benefit proportional to the area restored, such as carbon.

        ``values``: a raster on the problem's grid, a number or one value per unit; per km² restored
        (``per="area"``) or per cell restored in full (``per="cell"``)."""
        v = _on_units(self, values, "values")
        if not np.isfinite(v).all():
            raise ValueError("values: NaN or infinite on a planning unit")
        if per == "area":
            coef, layer = v * self.area, v
        elif per == "cell":
            coef, layer = v.copy(), v / self.area
        else:
            raise ValueError("per: 'area' or 'cell'")
        return replace(self, benefits=self.benefits + (_Linear(self._name(name, "linear"), float(weight), coef,
                                                               layer),))

    def add_intactness_benefit(self, hfi, reduction, radius: int = 3, weight: float = 1.0, *,
                               gamma: float | None = None, beta: float = -0.2, z: float = 0.5, cell_size=None,
                               name: str | None = None) -> RestorationProblem:
        """Ecological intactness (Beyer et al. 2020): restoring a share x of a cell lowers its human
        footprint by ``reduction`` x; the benefit is the total intactness gained, linearised at no
        restoration (:func:`landschaft.intactness_gradient`; with ``method="steps"`` it is recomputed at
        every step, as plangea does). ``hfi`` on the problem's grid; the other parameters as
        :func:`landschaft.intactness`."""
        from ..intactness import INTACTNESS_GAMMA, intactness_gradient

        gamma = INTACTNESS_GAMMA if gamma is None else float(gamma)
        g = intactness_gradient(hfi, reduction, radius, gamma=gamma, beta=beta, z=z, cell_size=cell_size)
        gv = np.asarray(g.values, dtype=np.float64)
        if gv.shape != self.shape:
            raise ValueError(f"hfi: the grid {gv.shape} is not the problem's {self.shape}")
        coef = np.nan_to_num(gv.ravel()[self.index])
        lin = _Linear(self._name(name, "intactness"), float(weight), coef, coef.copy(),
                      intactness=(hfi, reduction, int(radius), gamma, float(beta), float(z), cell_size))
        return replace(self, benefits=self.benefits + (lin,))

    def _name(self, name, default):
        if name is not None:
            if any(b.name == name for b in self.benefits):
                raise ValueError(f"a benefit is already named {name!r}")
            return str(name)
        used = {b.name for b in self.benefits}
        out, k = default, 2
        while out in used:
            out, k = f"{default}_{k}", k + 1
        return out

    # --- constraints
    def add_area_target(self, target, sense: str = "<=", regions=None) -> RestorationProblem:
        """The area restored (km²): in the whole landscape, or in every region.

        Parameters
        ----------
        target
            A number (km²), or with ``regions`` a mapping region -> target, or a table with columns
            ``region``, ``target`` and optionally ``sense``.
        sense
            ``"<="`` (at most: the most valuable cells for that area), ``">="`` or ``"="``.
        regions
            A raster of region codes on the problem's grid (NaN: no region) or one code per unit.
        """
        if sense not in ("<=", ">=", "="):
            raise ValueError("sense: '<=', '>=' or '='")
        if regions is None:
            if not np.isscalar(target):
                raise ValueError("a target per region needs regions")
            rows = sp.csr_matrix(self.area[None, :])
            t = _Targets(rows, np.array([float(target)]), (sense,), ("landscape",))
        else:
            codes = _on_units(self, regions, "regions")
            if isinstance(target, Mapping):
                tab = pd.DataFrame({"region": list(target.keys()), "target": list(target.values())})
            else:
                tab = pd.DataFrame(target)
                if "region" not in tab or "target" not in tab:
                    raise ValueError("target: a mapping region -> target or a table with region and target")
            senses = tab["sense"].astype(str).tolist() if "sense" in tab else [sense] * len(tab)
            if any(s not in ("<=", ">=", "=") for s in senses):
                raise ValueError("sense: '<=', '>=' or '='")
            r, c, v = [], [], []
            for k, code in enumerate(tab["region"].to_numpy(np.float64)):
                cells = np.flatnonzero(codes == code)
                r += [k] * cells.size
                c += cells.tolist()
                v += self.area[cells].tolist()
            rows = sp.csr_matrix((v, (r, c)), shape=(len(tab), self.n))
            t = _Targets(rows, tab["target"].to_numpy(np.float64), tuple(senses),
                         tuple(str(x) for x in tab["region"].tolist()))
        if not np.isfinite(t.rhs).all():
            raise ValueError("targets must be finite")
        return replace(self, targets=self.targets + (t,))

    def add_budget(self, budget: float, cost) -> RestorationProblem:
        """Spend at most ``budget`` restoring, ``cost`` per km² restored (a raster, a number or one value
        per unit; the opportunity cost of the land, say)."""
        c = self._cost(cost)
        if not math.isfinite(float(budget)):
            raise ValueError("budget must be finite")
        return replace(self, cost=c * self.area, cost_layer=c, budget=float(budget))

    def add_max_benefit_objective(self) -> RestorationProblem:
        """The most weighted benefit within the targets and the budget (the default)."""
        return replace(self, objective=("max_benefit",))

    def add_min_cost_objective(self, benefit: float, cost=None) -> RestorationProblem:
        """The cheapest plan whose weighted benefit reaches ``benefit``; ``cost`` per km² restored (or the
        budget's, without it)."""
        if cost is None and self.cost_layer is None:
            raise ValueError("the minimum cost needs a cost")
        out = self if cost is None else replace(self, cost=self._cost(cost) * self.area, cost_layer=self._cost(cost))
        return replace(out, objective=("min_cost", float(benefit)))

    def _cost(self, cost) -> np.ndarray:
        c = _on_units(self, cost, "cost")
        if not np.isfinite(c).all():
            raise ValueError("cost: NaN or infinite on a planning unit")
        return c

    # --- solvers
    def add_highs_solver(self, tolerance: float = 1e-7, time_limit: float | None = None,
                         n_threads: int | None = None, verbose: bool = False) -> RestorationProblem:
        """Solve the linear programs on HiGHS (built in). ``tolerance``: the relative gap between the
        bound and the plan's benefit (or cost) at which the exact method stops."""
        return replace(self, solver=("highs", {"tolerance": float(tolerance), "time_limit": time_limit,
                                               "n_threads": n_threads, "verbose": bool(verbose)}))

    def add_gurobi_solver(self, tolerance: float = 1e-7, time_limit: float | None = None,
                          n_threads: int | None = None, verbose: bool = False) -> RestorationProblem:
        """Solve the linear programs on Gurobi (needs ``gurobipy`` and a licence)."""
        return replace(self, solver=("gurobi", {"tolerance": float(tolerance), "time_limit": time_limit,
                                                "n_threads": n_threads, "verbose": bool(verbose)}))

    # --- solving
    def solve(self, method: str = "exact", steps: int = 1) -> RestorationSolution:
        """Solve the problem.

        ``method="exact"``: the optimum within the solver's ``tolerance`` (relative, between the plan's
        benefit and a bound on the best one). Simplicial decomposition: every vertex of the set of plans
        is the solution of a linear program over the units with the gradient of the benefits as its
        objective and only the targets and the budget as rows; the plan is the best mix of the vertices
        found (a small concave problem, by SLSQP); the bound is F(x) + grad F(x) . (v - x), v the
        next vertex (valid because the benefits are concave). The minimum cost: a bisection on the
        budget, each step the most benefit for it; the bound is the largest budget proven too small.

        ``method="steps"``: plangea's procedure, for comparison: every cell scored by its weighted
        benefits (each layer scaled by 1 / 10^trunc(log10(max))) per cost, the area targets met in
        ``steps`` equal parts, the extinction-risk and intactness layers recomputed after every part.
        It is a heuristic: its benefit is at most the exact one, and depends on ``steps``. Area
        targets only (no budget, no minimum cost)."""
        t0 = time.perf_counter()
        if method == "exact":
            solver = _solve_sd if self.objective[0] == "max_benefit" else _solve_min_cost
            x, bound, status, rounds = solver(self)
        elif method == "steps":
            x = _solve_steps(self, int(steps))
            bound, status, rounds = math.nan, "heuristic", int(steps)
        else:
            raise ValueError("method: 'exact' or 'steps'")
        return self._solution(x, bound=bound, status=status, method=method, rounds=rounds,
                              seconds=time.perf_counter() - t0)

    def evaluate(self, restored) -> RestorationSolution:
        """The benefits, cost and area of a plan: the share restored per unit (a raster on the grid or
        one value per unit)."""
        x = _on_units(self, restored, "restored")
        return self._solution(np.nan_to_num(x), bound=math.nan, status="given", method="given", rounds=0,
                              seconds=0.0)

    def uniform(self, area: float | None = None) -> RestorationSolution:
        """plangea's uniform scenario: every unit restored by the same share of what it can be, for a
        total ``area`` (default: the first landscape area target)."""
        if area is None:
            land = [t for t in self.targets if t.labels == ("landscape",)]
            if not land:
                raise ValueError("give the area (or add a landscape area target)")
            area = float(land[0].rhs[0])
        total = float(np.sum(self.upper * self.area))
        x = self.upper * (float(area) / total if total > 0 else 0.0)
        return self._solution(x, bound=math.nan, status="uniform", method="uniform", rounds=0, seconds=0.0)

    def _solution(self, x, *, bound, status, method, rounds, seconds) -> RestorationSolution:
        x = np.asarray(x, dtype=np.float64)
        rows = []
        total = 0.0
        for b in self.benefits:
            v = _benefit_value(b, x)
            rows.append({"benefit": b.name, "kind": "extinction_risk" if isinstance(b, _Risk) else "linear",
                         "value": v, "weight": b.weight, "weighted": b.weight * v})
            total += b.weight * v
        table = pd.DataFrame(rows, columns=["benefit", "kind", "value", "weight", "weighted"])
        cost = float(self.cost @ x) if self.cost is not None else math.nan
        objective = cost if self.objective[0] == "min_cost" and method != "steps" else total
        grid = np.full(self.shape[0] * self.shape[1], np.nan)
        grid[self.index] = x
        grid = grid.reshape(self.shape)
        restored = self.template.copy(data=grid) if self.template is not None else xr.DataArray(grid, dims=("y", "x"))
        restored.name = "restored"
        return RestorationSolution(restored=restored, values=x, area=float(self.area @ x), cost=cost,
                                   benefits=table, benefit=total, objective=objective, bound=bound, status=status,
                                   method=method, rounds=rounds, seconds=seconds)


@dataclass
class RestorationSolution(Saveable):
    """A restoration plan.

    Attributes
    ----------
    restored
        The share of every cell restored (NaN outside the planning units).
    values
        The same, one value per unit.
    area
        The area restored (km²).
    cost
        Its cost (NaN without a cost).
    benefits
        One row per benefit: its value, weight and weighted value.
    benefit
        The weighted benefit.
    objective, bound
        The objective (the weighted benefit, or the cost with the minimum-cost objective) and the bound
        on the optimum the exact method proved (NaN for ``method="steps"``).
    status, method, rounds, seconds
        ``"optimal"`` (within the tolerance), ``"time_limit"``, ``"heuristic"`` (steps) ...; the method,
        the iterations (vertices or bisection steps; steps for ``method="steps"``), the time.
    """

    restored: xr.DataArray = field(repr=False)
    values: np.ndarray = field(repr=False)
    area: float = 0.0
    cost: float = math.nan
    benefits: pd.DataFrame = field(default=None, repr=False)
    benefit: float = 0.0
    objective: float = math.nan
    bound: float = math.nan
    status: str = ""
    method: str = ""
    rounds: int = 0
    seconds: float = 0.0

    def _save_items(self) -> dict:
        return {"restored": self.restored, "benefits": self.benefits, "area": self.area, "cost": self.cost,
                "benefit": self.benefit, "objective": self.objective, "bound": self.bound, "status": self.status,
                "method": self.method}


def restoration_problem(restorable, *, limit=1.0, cell_area=None) -> RestorationProblem:
    """A restoration problem: how much of every converted cell to restore.

    The planning units are the cells with land that can be restored; the decision is the share x_i of
    each restored, from 0 to ``restorable`` x ``limit``. Then add benefits
    (:meth:`~RestorationProblem.add_extinction_risk_benefit`,
    :meth:`~RestorationProblem.add_linear_benefit`, :meth:`~RestorationProblem.add_intactness_benefit`),
    targets (:meth:`~RestorationProblem.add_area_target`), a budget, and solve.

    Parameters
    ----------
    restorable
        The share of every cell that can be restored: a raster, or a
        :class:`~landschaft.OriginalVegetation` (its anthropic share).
    limit
        plangea's bound limit: a number or a raster multiplying ``restorable``.
    cell_area
        The cells' area in km²: a number, a raster, or None (from the grid, see
        :func:`landschaft.sdm.cell_area_km2`).

    Returns
    -------
    RestorationProblem

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> rng = np.random.default_rng(1)
    >>> anthropic = rng.random((20, 30)) * 0.6
    >>> carbon = rng.random((20, 30)) * 100
    >>> p = (ls.prioritize.restoration_problem(anthropic, cell_area=1.0)
    ...        .add_linear_benefit(carbon, name="carbon")
    ...        .add_area_target(50.0))
    >>> sol = p.solve()
    >>> round(sol.area, 6), sol.status
    (50.0, 'optimal')
    """
    from .._vegetation import OriginalVegetation

    if isinstance(restorable, OriginalVegetation):
        restorable = restorable.anthropic
    template, r = _grid_values(restorable, "restorable")
    shape = r.shape
    lim = np.full(shape, float(limit)) if np.isscalar(limit) else _grid_values(limit, "limit")[1]
    if lim.shape != shape:
        raise ValueError("limit: not on the grid of restorable")
    up = np.nan_to_num(r, nan=0.0) * np.nan_to_num(lim, nan=0.0)
    index = np.flatnonzero(up.ravel() > 0)
    if index.size == 0:
        raise ValueError("no cell can be restored")
    if cell_area is None:
        if template is None:
            raise ValueError("give cell_area (km²) with a plain array")
        from ..sdm import cell_area_km2

        area = cell_area_km2(template)
    elif np.isscalar(cell_area):
        area = np.full(shape, float(cell_area))
    else:
        area = _grid_values(cell_area, "cell_area")[1]
        if area.shape != shape:
            raise ValueError("cell_area: not on the grid of restorable")
    a = area.ravel()[index]
    if not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError("cell_area: positive on every cell that can be restored")
    if template is not None:
        template = template.copy(data=np.full(shape, np.nan))
    return RestorationProblem(template=template, shape=shape, index=index.astype(np.int64),
                              upper=up.ravel()[index].copy(), area=a.copy())


def ecosystem_areas(land_cover, natural, anthropic, ecoregions, *, cell_area=None):
    """The natural area of every ecosystem (ecoregion) now and before conversion, and what restoring a
    cell gives it: the input of :meth:`RestorationProblem.add_extinction_risk_benefit` for the risk of
    ecosystem collapse (plangea's ``ec``).

    Parameters
    ----------
    land_cover
        The share of every class in each cell (``(band, y, x)``, band names the classes).
    natural, anthropic
        The natural and the anthropic classes.
    ecoregions
        A raster of ecoregion codes (NaN: none).
    cell_area
        As :func:`restoration_problem`.

    Returns
    -------
    landschaft.sdm.HabitatAreas
        One row per ecoregion (``species`` its code): ``current`` the natural area, ``potential`` the
        natural plus the anthropic area, and the gain: the cell's area per unit restored, in the cells
        with anthropic land.

    Notes
    -----
    plangea sums the natural area of an ecoregion only over the cells of its problem (those with
    anthropic land), which leaves out the ecoregion's untouched land and overstates its loss; here
    every cell of the ecoregion counts.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> cover = xr.DataArray(np.array([[[0.8, 0.2, 0.0, 0.5]], [[0.2, 0.8, 1.0, 0.5]]]),
    ...                      dims=("band", "y", "x"), coords={"band": ["forest", "crops"]})
    >>> eco = np.array([[1.0, 1.0, 2.0, 2.0]])
    >>> ea = ls.prioritize.ecosystem_areas(cover, ["forest"], ["crops"], eco, cell_area=1.0)
    >>> ea.table[["species", "current", "potential"]]
      species  current  potential
    0       1      1.0        2.0
    1       2      0.5        2.0
    """
    from .._layers import named_layers
    from ..sdm import cell_area_km2
    from ..sdm._habitat import HabitatAreas

    names, cover, template = named_layers(land_cover, "land_cover")
    pos = {n: i for i, n in enumerate(names)}
    for c in list(natural) + list(anthropic):
        if c not in pos:
            raise ValueError(f"land_cover has no layer {c!r}")
    shape = cover.shape[1:]
    nat = np.zeros(shape)
    for c in natural:
        nat = nat + np.nan_to_num(cover[pos[c]])
    ant = np.zeros(shape)
    for c in anthropic:
        ant = ant + np.nan_to_num(cover[pos[c]])
    _, eco = _grid_values(ecoregions, "ecoregions")
    if eco.shape != shape:
        raise ValueError("ecoregions: not on the land cover's grid")
    if cell_area is None:
        area = cell_area_km2(template)
    elif np.isscalar(cell_area):
        area = np.full(shape, float(cell_area))
    else:
        area = _grid_values(cell_area, "cell_area")[1]
    e, a, nv, av = eco.ravel(), area.ravel(), nat.ravel(), ant.ravel()
    codes = np.unique(e[np.isfinite(e)])
    cur, pot, rows, cols, vals = [], [], [], [], []
    for k, code in enumerate(codes):
        cells = np.flatnonzero(e == code)
        cur.append(float(np.sum(a[cells] * nv[cells])))
        pot.append(float(np.sum(a[cells] * (nv[cells] + av[cells]))))
        g = cells[av[cells] > 0]
        rows += [k] * g.size
        cols += g.tolist()
        vals += a[g].tolist()
    gain = sp.csr_matrix((vals, (rows, cols)), shape=(codes.size, e.size))
    label = [str(int(c)) if float(c).is_integer() else str(c) for c in codes]
    table = pd.DataFrame({"species": label, "current": cur, "potential": pot,
                          "range_cells": [int(np.sum(e == c)) for c in codes], "classes": ", ".join(natural),
                          "valid": (np.array(pot) > 0)})
    return HabitatAreas(table=table, gain=gain, _template=template)


# ----------------------------------------------------------------------------- the benefits

def _h(r: np.ndarray, z: float) -> np.ndarray:
    """1 - risk: min(r, 1)^z."""
    return np.minimum(np.maximum(r, 0.0), 1.0) ** z


def _ratio(b: _Risk, x: np.ndarray) -> np.ndarray:
    return (b.current + b.gain @ x) / b.potential


def _benefit_value(b, x: np.ndarray) -> float:
    if isinstance(b, _Linear):
        return float(b.coef @ x)
    return float(np.sum(_h(_ratio(b, x), b.z) - _h(b.current / b.potential, b.z)))


def _true_objective(p: RestorationProblem, x: np.ndarray) -> float:
    return sum(b.weight * _benefit_value(b, x) for b in p.benefits)


# ----------------------------------------------------------------------------- exact

def _solve_lp(p: RestorationProblem, m: OptimizationProblem, time_limit: float, n_units: int | None = None) -> dict:
    from .. import _threads
    from ._milp_solve import _gurobi, _highs

    solver, opts = p.solver
    threads = _threads.resolve(opts.get("n_threads"))
    verbose = bool(opts.get("verbose", False))
    if solver == "gurobi":
        return _gurobi(m, 0.0, time_limit, threads, verbose, None)
    return _highs(m, 0.0, time_limit, threads, "choose", verbose, None, n_units=n_units)


# ----------------------------------------------------------------------------- exact: simplicial decomposition

def _feasible_set(p: RestorationProblem) -> tuple[sp.csr_matrix, np.ndarray, tuple]:
    """The rows of the plans (area targets, budget) on the units."""
    rows, rhs, sense = [], [], []
    for t in p.targets:
        rows.append(t.rows)
        rhs.append(t.rhs)
        sense += list(t.sense)
    if p.budget is not None:
        rows.append(sp.csr_matrix(p.cost[None, :]))
        rhs.append(np.array([p.budget]))
        sense.append("<=")
    if not rows:
        return sp.csr_matrix((0, p.n)), np.zeros(0), ()
    return sp.vstack(rows).tocsr(), np.concatenate(rhs), tuple(sense)


def _solve_sd(p: RestorationProblem) -> tuple[np.ndarray, float, str, int]:
    """The most weighted benefit by simplicial decomposition: the plan is a mix of vertices of the set of
    plans, each from a linear program over the units (the benefits' gradient at the current plan, the
    targets and the budget as rows), the mix by a small smooth master problem (SLSQP on the weights); the
    bound is the Frank-Wolfe one, F(x) + grad F(x) . (v - x) for the best vertex v, valid because the
    benefit is concave. Its linear programs have only the problem's few rows, so it scales with the
    units, where one linear program with tangent planes of every species' curve (Kelley's cutting planes,
    tried first) did not: 9 x 10^4 cells and 500 species, 0.57 s against more than 15 minutes."""
    opts = p.solver[1]
    tol = float(opts.get("tolerance", 1e-7))
    tl = opts.get("time_limit")
    deadline = time.perf_counter() + (math.inf if tl is None else float(tl))
    n = p.n
    risks = [b for b in p.benefits if isinstance(b, _Risk)]
    lin = np.zeros(n)
    for b in p.benefits:
        if isinstance(b, _Linear):
            lin += b.weight * b.coef
    rows, rhs, sense = _feasible_set(p)
    floors = [np.maximum(1e-9 * _ratio(b, p.upper), 1e-15) for b in risks]

    def left() -> float:
        return deadline - time.perf_counter()

    def gradient(x: np.ndarray) -> np.ndarray:
        g = lin.copy()
        for b, fl in zip(risks, floors):
            r = _ratio(b, x)
            d = np.where(r < 1.0, b.z * np.maximum(r, fl) ** (b.z - 1.0), 0.0) * (b.weight / b.potential)
            g += b.gain.T @ d
        return g

    def vertex(c: np.ndarray) -> tuple[np.ndarray, bool]:
        m = OptimizationProblem(A=rows, obj=c, lb=np.zeros(n), ub=p.upper, rhs=rhs, sense=sense, vtype=("C",) * n,
                                modelsense="max", row_ids=("row",) * rows.shape[0], col_ids=("x",) * n)
        r = _solve_lp(p, m, max(left(), 1e-3), n)
        if r["status"] == "infeasible":
            raise RuntimeError("the restoration problem is infeasible (targets or budget)")
        if not r["has_solution"]:
            raise RuntimeError(f"no plan ({p.solver[0]}: {r['status']})")
        # only an optimal vertex gives a valid bound
        return np.clip(np.asarray(r["x"], dtype=np.float64)[:n], 0.0, p.upper), r["status"] == "optimal"

    # the vertices (sparse columns), their gains per risk benefit (species x vertices) and linear benefit
    verts: list[sp.csr_matrix] = []
    gv = [np.zeros((b.current.size, 0)) for b in risks]
    lv = np.zeros(0)

    def add_vertex(v: np.ndarray):
        nonlocal lv
        verts.append(sp.csr_matrix(v[None, :]))
        for i, b in enumerate(risks):
            gv[i] = np.hstack([gv[i], (b.gain @ v)[:, None]])
        lv = np.append(lv, lin @ v)

    def master(lam0: np.ndarray) -> np.ndarray:
        """The best mix of the vertices (weights on the simplex), by SLSQP with the exact gradient from
        ``lam0``: a smooth concave problem of a few variables. Its accuracy only steers the search: the
        bound is Frank-Wolfe's on the plan it gives."""
        from scipy.optimize import minimize

        k = len(verts)
        if k == 1:
            return np.ones(1)

        def f(lam):
            v = float(lv @ lam)
            for i, b in enumerate(risks):
                v += b.weight * float(np.sum(_h((b.current + gv[i] @ lam) / b.potential, b.z)))
            return -v

        def grad(lam):
            g = lv.copy()
            for i, b in enumerate(risks):
                r = (b.current + gv[i] @ lam) / b.potential
                d = np.where(r < 1.0, b.z * np.maximum(r, floors[i]) ** (b.z - 1.0), 0.0) * (b.weight / b.potential)
                g += gv[i].T @ d
            return -g

        res = minimize(f, lam0, jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                       constraints=[{"type": "eq", "fun": lambda lam: np.sum(lam) - 1.0, "jac": lambda lam: np.ones(k)}],
                       options={"maxiter": 1000, "ftol": 1e-15})
        lam = np.clip(res.x, 0.0, None)
        lam = lam / lam.sum() if lam.sum() > 0 else lam0
        return lam if f(lam) <= f(lam0) else lam0

    add_vertex(vertex(gradient(np.zeros(n)))[0])
    bound, best_x, best = math.inf, None, -math.inf
    status, it = "time_limit", 0
    lam = np.ones(1)
    for it in range(1, 5001):
        lam = master(lam)
        x = np.zeros(n)
        for w, v in zip(lam, verts):
            if w > 0:
                x += w * v.toarray().ravel()
        x = np.clip(x, 0.0, p.upper)
        f = _true_objective(p, x)
        if f > best:
            best_x, best = x, f
        g = gradient(x)
        v, exact = vertex(g)
        if exact:
            bound = min(bound, max(f + float(g @ (v - x)), f))
        if bound - best <= tol * max(1.0, abs(bound)):
            status = "optimal"
            break
        if left() <= 0:
            break
        # drop the vertices out of the mix (the master grows with them)
        keep = np.flatnonzero(lam > 1e-12)
        if keep.size < lam.size:
            verts[:] = [verts[j] for j in keep]
            gv[:] = [m_[:, keep] for m_ in gv]
            lv = lv[keep]
            lam = lam[keep] / lam[keep].sum()
        add_vertex(v)
        lam = np.append(lam, 0.0)
    return best_x, bound, status, it


def _solve_min_cost(p: RestorationProblem) -> tuple[np.ndarray, float, str, int]:
    """The cheapest plan for a weighted benefit: the most benefit for a budget is concave and increasing
    in the budget, so a bisection on the budget, each step the most benefit for it
    (:func:`_solve_sd`), finds the least budget that reaches the benefit; a budget whose bound falls
    short proves the cost above it (the bound returned)."""
    target = p.objective[1]
    tol = float(p.solver[1].get("tolerance", 1e-7))
    tl = p.solver[1].get("time_limit")
    deadline = time.perf_counter() + (math.inf if tl is None else float(tl))
    inner = p.solver

    def most(budget: float | None):
        q = replace(p, objective=("max_benefit",), budget=budget, solver=(inner[0], {
            **inner[1], "time_limit": max(deadline - time.perf_counter(), 1e-3) if math.isfinite(deadline) else None}))
        try:
            return _solve_sd(q)
        except RuntimeError as e:
            if "infeasible" in str(e):
                return None
            raise

    top = most(p.budget)  # the most benefit at all (within the problem's own budget)
    if top is None or top[1] < target:
        raise RuntimeError("the restoration problem is infeasible: the benefit asked for cannot be reached")
    best_x, hi, lo, proven = top[0], float(p.cost @ top[0]), 0.0, 0.0
    if _true_objective(p, best_x) < target:  # the best plan found falls short within the tolerance
        best_x = None
    status, it = "time_limit", 1
    while best_x is not None and time.perf_counter() < deadline and it < 200:
        if hi - lo <= tol * max(1.0, abs(hi)):
            status = "optimal"
            break
        it += 1
        mid = 0.5 * (lo + hi)
        r = most(mid)
        if r is not None and _true_objective(p, r[0]) >= target:
            best_x, hi = r[0], min(mid, float(p.cost @ r[0]))
        else:
            lo = mid
            if r is None or r[1] < target:  # proven: no plan within mid reaches the benefit
                proven = mid
    if best_x is None:
        raise RuntimeError(f"no plan reaches the benefit within the tolerance ({status})")
    return best_x, proven, status, it


# ----------------------------------------------------------------------------- plangea's steps

def _plangea_slopes(b: _Risk, extra: np.ndarray) -> np.ndarray:
    """plangea's .multi_extinction_slope: |risk((A + delta) / P) - risk(A / P)| / delta, A the habitat now
    plus what was restored."""
    a = b.current + extra
    r1 = np.maximum(1.0 - (a / b.potential) ** b.z, 0.0)
    r2 = np.maximum(1.0 - ((a + PLANGEA_DELTA) / b.potential) ** b.z, 0.0)
    return np.abs(r2 - r1) / PLANGEA_DELTA


def _steps_layer(p: RestorationProblem, b, cum: np.ndarray) -> np.ndarray:
    if isinstance(b, _Risk):
        slopes = _plangea_slopes(b, b.gain @ cum)
        # plangea's bd: the slopes times the habitat share a unit of the cell restored gives
        hab = b.hab.multiply(sp.csr_matrix((slopes * (1.0 - b.discount))[:, None])).tocsc()
        return np.asarray(hab.sum(axis=0)).ravel()
    if b.intactness is not None and cum.any():
        from ..intactness import intactness_gradient

        hfi, reduction, radius, gamma, beta, z, cell_size = b.intactness
        full = np.zeros(p.shape[0] * p.shape[1])
        full[p.index] = cum
        g = intactness_gradient(hfi, reduction, radius, gamma=gamma, beta=beta, z=z, cell_size=cell_size,
                                restored=full.reshape(p.shape))
        return np.nan_to_num(np.asarray(g.values, dtype=np.float64).ravel()[p.index])
    return b.layer


def _scale(layer: np.ndarray) -> float:
    """plangea's automatic scaling factor: 1 / 10^trunc(log10(max |layer|))."""
    mx = float(np.max(np.abs(layer))) if layer.size else 0.0
    if not mx > 0 or not math.isfinite(mx):
        return 1.0
    return 1.0 / 10.0 ** math.trunc(math.log10(mx))


def _solve_steps(p: RestorationProblem, steps: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if p.budget is not None or p.objective[0] != "max_benefit":
        raise ValueError("method='steps' takes area targets only (plangea's procedure): no budget, no minimum cost")
    if not p.targets:
        raise ValueError("method='steps' needs an area target")
    n = p.n
    cum = np.zeros(n)
    ub = p.upper.copy()
    layers = [_steps_layer(p, b, cum) for b in p.benefits]
    sf = [_scale(v) for v in layers]
    csf = _scale(p.cost_layer) if p.cost_layer is not None else 1.0
    denom = p.cost_layer * csf if p.cost_layer is not None else 1.0
    rows = sp.vstack([t.rows for t in p.targets]).tocsr()
    rhs = np.concatenate([t.rhs for t in p.targets]) / steps
    sense = tuple(s for t in p.targets for s in t.sense)
    for k in range(steps):
        b_sum = 0.0
        for b, v, s in zip(p.benefits, layers, sf):
            b_sum = b_sum + b.weight * s * v
        obj = (b_sum / denom) if p.benefits else np.ones(n) / denom
        obj = np.broadcast_to(np.asarray(obj, dtype=np.float64), (n,)).copy()
        m = OptimizationProblem(A=rows, obj=obj, lb=np.zeros(n), ub=ub.copy(), rhs=rhs, sense=sense,
                                vtype=("C",) * n, modelsense="max", row_ids=("area",) * rows.shape[0],
                                col_ids=("x",) * n)
        r = _solve_lp(p, m, math.inf)
        if not r["has_solution"]:
            raise RuntimeError(f"step {k + 1}: no solution ({r['status']})")
        x = np.clip(np.asarray(r["x"], dtype=np.float64)[:n], 0.0, None)
        ub = np.where(x > 0, ub - x, ub)
        cum = cum + x
        if k + 1 < steps:
            layers = [_steps_layer(p, b, cum) if (isinstance(b, _Risk) or b.intactness is not None) else v
                      for b, v in zip(p.benefits, layers)]
    return np.minimum(cum, p.upper)


# ----------------------------------------------------------------------------- evaluation

@dataclass
class RiskReduction(Saveable):
    """Result of :func:`eval_risk_reduction`: per species, its habitat and risk before and after; the
    total reduction and its standard deviation from the uncertainty of z."""

    table: pd.DataFrame = field(repr=False)
    total: float = 0.0
    sd: float = math.nan
    method: str = ""

    def _save_items(self) -> dict:
        return {"table": self.table, "total": self.total, "sd": self.sd, "method": self.method}


def eval_risk_reduction(problem: RestorationProblem, solution, z_sd: float | None = None, *,
                        benefit: str | None = None, method: str = "correlated",
                        step: float = 1e-6) -> RiskReduction:
    """The reduction of the extinction risk a plan brings (plangea's ``multi_exrisk_aggregate``), with
    the uncertainty of the species-area exponent z.

    Parameters
    ----------
    problem
        The restoration problem.
    solution
        A :class:`RestorationSolution`, or the share restored per unit (or a raster).
    z_sd
        The standard deviation of z (plangea: 0.1): a number for every species, one per species (in
        the order of the benefit's species) or a mapping species -> deviation (0 for the others); None:
        no uncertainty.
    benefit
        The extinction-risk benefit to evaluate (default: the first).
    method
        How the species' sensitivities to z add up. ``"correlated"`` (the default): the species' z err
        together (one z for all of them, or per-species values from one source), so the total's
        deviation is |sum_s d reduction_s / dz_s sd_s| (the delta method on the total).
        ``"independent"`` sums them in quadrature and ``"linear"`` sums their absolute values, plangea's
        two forms (they treat the species' z as separate parameters).
    step
        The forward difference of z (plangea's 1e-6).

    Returns
    -------
    RiskReduction

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> cover = xr.DataArray(np.array([[[0.6, 0.0, 0.3]], [[0.2, 0.1, 0.0]], [[0.2, 0.9, 0.7]]]),
    ...                      dims=("band", "y", "x"), coords={"band": ["forest", "savanna", "crops"]})
    >>> ranges = xr.DataArray(np.ones((2, 1, 3)), dims=("band", "y", "x"), coords={"band": ["toucan", "rhea"]})
    >>> veg = ls.original_vegetation(cover, ["forest", "savanna"], ["crops"])
    >>> aoh = ls.sdm.area_of_habitat(ranges, cover, ["forest", "savanna"], original=veg, cell_area=1.0,
    ...                              suitability={"toucan": ["forest"], "rhea": ["savanna"]})
    >>> p = (ls.prioritize.restoration_problem(veg, cell_area=1.0)
    ...        .add_extinction_risk_benefit(aoh).add_area_target(0.8))
    >>> sol = p.solve()
    >>> rr = ls.prioritize.eval_risk_reduction(p, sol, z_sd=0.1)
    >>> round(rr.total, 4), round(rr.sd, 4)
    (0.269, 0.0871)
    """
    risks = [b for b in problem.benefits if isinstance(b, _Risk)]
    if benefit is not None:
        risks = [b for b in risks if b.name == benefit]
    if not risks:
        raise ValueError("the problem has no such extinction-risk benefit")
    b = risks[0]
    x = solution.values if isinstance(solution, RestorationSolution) else _on_units(problem, solution, "solution")
    gained = b.gain @ np.nan_to_num(np.asarray(x, dtype=np.float64))
    after = b.current + gained
    after = np.where(np.abs(after) < EXRISK_EPS, 0.0, after)

    def reduction(z: np.ndarray) -> np.ndarray:
        before = np.maximum(1.0 - (b.current / b.potential) ** z, 0.0)
        aft = np.maximum(1.0 - (after / b.potential) ** z, 0.0)
        return np.nan_to_num(before - aft)

    red = reduction(b.z)
    table = pd.DataFrame({"species": list(b.species), "current": b.current, "potential": b.potential,
                          "gained": gained, "risk_before": np.maximum(1.0 - (b.current / b.potential) ** b.z, 0.0),
                          "risk_after": np.maximum(1.0 - (after / b.potential) ** b.z, 0.0), "reduction": red})
    total = float(np.sum(red))
    sd = math.nan
    if z_sd is not None:
        sz = _per_species(z_sd, np.asarray(b.species), 0.0, "z_sd")
        d = (reduction(b.z + step) - red) / step
        if method == "correlated":
            sd = abs(float(np.sum(d * sz)))
        elif method == "independent":
            sd = math.sqrt(float(np.sum((d * sz) ** 2)))
        elif method == "linear":
            sd = float(np.sum(np.abs(d * sz)))
        else:
            raise ValueError("method: 'correlated', 'independent' or 'linear'")
    return RiskReduction(table=table, total=total, sd=sd, method=method)
