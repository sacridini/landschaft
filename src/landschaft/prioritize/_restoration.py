# SPDX-License-Identifier: GPL-3.0-or-later
"""Restoration planning: how much of every converted cell to restore, for the most benefit
(the restoration problem of plangea, Strassburg et al. 2020, International Institute for
Sustainability; GPL >= 3), and its conservation and conversion forms.

The decision is the share x_i of each cell restored, between 0 and the share that can be (its
anthropic land, times a limit). A plan can also conserve natural land that would otherwise be lost,
or convert natural land (to meet a demand for it): every action is a block of shares, each from 0 to
what it can be. The benefits:

- the reduction of species' extinction risk (or ecosystems' risk of collapse): a species with habitat
  A of a potential P has the risk 1 - (A / P)^z (the species-area relationship, z = 0.25), and
  restoring gives it habitat, A = A_0 + sum_i g_si x_i (conserving keeps habitat that would be lost,
  converting takes it away: g < 0);
- linear benefits (carbon, per km² restored);
- ecological intactness, by its derivative (:func:`landschaft.intactness_gradient`).

The risk reduction is concave in x (A is affine in x), so the problem (the most weighted benefit
within area targets and a budget) is solved to a proven tolerance by simplicial decomposition: plans
that are vertices of the set of plans, each the solution of a linear program over the cells with only
the problem's rows, mixed by a small smooth master problem; the bound is the Frank-Wolfe one. The
cheapest plan for a benefit is a bisection on the budget. plangea's procedure (a ratio of benefits
to costs per cell, the target in steps with the benefits recomputed between them: a heuristic whose
result depends on the number of steps) is ``method="steps"``, reproduced for comparison.
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

__all__ = ["RestorationProblem", "RestorationSolution", "RestorationTradeOff", "RiskReduction", "ecosystem_areas", "ecosystem_targets",
           "eval_risk_reduction", "restoration_problem"]

PLANGEA_Z = 0.25  # plangea's exponent of the species-area relationship
PLANGEA_DELTA = 1e-6  # plangea's finite difference of the risk (in habitat area), method="steps"
EXRISK_EPS = 1e-8  # plangea's multi_exrisk_aggregate: an updated habitat this close to 0 is 0
ACTIONS = ("restore", "conserve", "convert")  # the blocks of a plan's shares, in this order
_CODE = {a: k for k, a in enumerate(ACTIONS)}
_SHARE = {"restore": "restorable", "conserve": "conservable", "convert": "convertible"}


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


def _is_grid(values) -> bool:
    return isinstance(values, xr.DataArray) or (isinstance(values, np.ndarray) and values.ndim == 2) or isinstance(
        values, (str, bytes)) or hasattr(values, "__fspath__")


def _on_units(p: RestorationProblem, values, what: str) -> np.ndarray:
    """A number, a raster or an array on the problem's grid, or one value per unit: the units' values."""
    if np.isscalar(values):
        return np.full(p.index.size, float(values))
    if _is_grid(values):
        _, v = _grid_values(values, what)
        if v.shape != p.shape:
            raise ValueError(f"{what}: the grid {v.shape} is not the problem's {p.shape}")
        return v.ravel()[p.index]
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size != p.index.size:
        raise ValueError(f"{what}: one value per planning unit ({p.index.size}) or a raster")
    return v


def _per_column(p: RestorationProblem, values, what: str) -> np.ndarray:
    """As :func:`_on_units`, or a mapping action -> values (0 for the actions left out)."""
    if not isinstance(values, Mapping):
        return _on_units(p, values, what)
    unknown = set(values) - set(p.actions)
    if unknown:
        raise ValueError(f"{what}: no action {sorted(unknown)} in the problem ({', '.join(p.actions)})")
    out = np.zeros(p.n)
    for a, v in values.items():
        m = p.action == _CODE[a]
        out[m] = _on_units(p, v, f"{what}[{a!r}]")[m]
    return out


@dataclass(frozen=True)
class _Risk:
    """A risk benefit: species (or ecosystems) with habitat now, without the plan (``base``: less than
    now when conservation's baseline loses natural land), potential habitat and the habitat every unit
    of a share adds (restoring, already discounted; conserving) or takes away (converting)."""

    name: str
    weight: float
    species: tuple
    current: np.ndarray
    potential: np.ndarray
    gain: sp.csr_matrix  # species x columns: habitat per unit share, signed
    hab: sp.csr_matrix  # species x columns: the same per unit of the cell's area (plangea's hab_values)
    z: np.ndarray  # one exponent per species
    discount: np.ndarray
    base: np.ndarray  # the habitat without the plan


@dataclass(frozen=True)
class _Footprint:
    """What a plan does to the human footprint, for the intactness benefit."""

    hfi: object  # the footprint as given (its grid gives the cells' size)
    h0: np.ndarray  # the footprint without the plan (flat; conservation's baseline loss added)
    change: np.ndarray  # per column: the footprint's change per unit share (negative lowers it)
    radius: int
    gamma: float
    beta: float
    z: float
    cell_size: object


@dataclass(frozen=True)
class _Linear:
    name: str
    weight: float
    coef: np.ndarray  # benefit per unit share
    layer: np.ndarray  # plangea's per-cell layer (method="steps")
    intactness: _Footprint | None = None


@dataclass(frozen=True)
class _Targets:
    rows: sp.csr_matrix  # targets x columns: area (or another value) per unit share
    rhs: np.ndarray
    sense: tuple
    labels: tuple


@dataclass(frozen=True)
class RestorationProblem:
    """A restoration problem (see :func:`restoration_problem`), built up by its ``add_*`` methods; every
    method returns a new problem."""

    template: xr.DataArray | None = field(repr=False)
    shape: tuple
    index: np.ndarray = field(repr=False)  # the columns' cells (row-major), block after block
    upper: np.ndarray = field(repr=False)  # the share of each column's cell that can be acted on
    area: np.ndarray = field(repr=False)  # each column's cell area (km²)
    benefits: tuple = ()
    targets: tuple = ()
    cost: np.ndarray | None = field(default=None, repr=False)  # per unit share
    cost_layer: np.ndarray | None = field(default=None, repr=False)  # per km² (plangea's layer)
    budget: float | None = None
    objective: tuple = ("max_benefit",)
    solver: tuple = ("highs", {})
    action: np.ndarray | None = field(default=None, repr=False)  # per column: 0 restore, 1 conserve, 2 convert
    natural: np.ndarray | None = field(default=None, repr=False)  # conservation: the natural share lost without it
    threat: np.ndarray | None = field(default=None, repr=False)  # conservation: the part of it lost
    lower: np.ndarray | None = field(default=None, repr=False)  # per column: the least share (None: 0)

    def __post_init__(self):
        if self.action is None:
            object.__setattr__(self, "action", np.zeros(self.index.size, dtype=np.int8))
        if self.natural is None:
            object.__setattr__(self, "natural", np.zeros(self.index.size))
        if self.threat is None:
            object.__setattr__(self, "threat", np.ones(self.index.size))

    @property
    def n(self) -> int:
        """The number of decision columns: the planning units of every action."""
        return int(self.index.size)

    @property
    def actions(self) -> tuple:
        """The actions of the plan, of ``"restore"``, ``"conserve"`` and ``"convert"``."""
        present = set(np.unique(self.action).tolist())
        return tuple(a for k, a in enumerate(ACTIONS) if k in present)

    def _lb(self) -> np.ndarray:
        return np.zeros(self.n) if self.lower is None else self.lower

    # --- benefits
    def add_extinction_risk_benefit(self, areas, z=PLANGEA_Z, weight: float = 1.0,
                                    restoration_discount=0.0, name: str | None = None) -> RestorationProblem:
        """The reduction of the species' extinction risk (or of ecosystems' risk of collapse).

        A species with habitat A of a potential P has the risk 1 - (A / P)^z (0 at A >= P); restoring a
        share x_i of a cell gives it ``g_i x_i`` more habitat, g the gain of
        :func:`landschaft.sdm.area_of_habitat` (or of :func:`ecosystem_areas`). Conserving a share of a
        cell's natural land keeps the habitat that would be lost (its ``loss`` times the threat);
        converting it takes that habitat away. The benefit is the sum over the species of the risk
        without the plan minus the risk with it (negative when conversion raises the risk).

        Parameters
        ----------
        areas
            A :class:`~landschaft.sdm.HabitatAreas` with its gain (``area_of_habitat`` given the
            original vegetation) for restoration and its loss (``loss=True``) for conservation and
            conversion, on the problem's grid; the species that are not ``valid`` are left out.
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
            species -> discount. It is not a financial discount rate, and only restoration has it.
        name
            The benefit's name (default ``"extinction_risk"``, numbered when repeated).
        """
        gain = getattr(areas, "gain", None)
        table = getattr(areas, "table", None)
        loss = getattr(areas, "loss", None)
        acts = self.actions
        if table is None or ("restore" in acts and gain is None):
            raise ValueError("areas: a HabitatAreas with its gain (area_of_habitat given the original vegetation)")
        if ("conserve" in acts or "convert" in acts) and loss is None:
            raise ValueError("areas: no loss for conservation or conversion (area_of_habitat(..., loss=True))")
        ncell = self.shape[0] * self.shape[1]
        for m in (gain, loss):
            if m is not None and m.shape[1] != ncell:
                raise ValueError("areas: not on the problem's grid")
        names = table["species"].astype(str).to_numpy()
        zs = _per_species(z, names, PLANGEA_Z, "z")
        if (zs <= 0).any() or (zs >= 1).any():
            raise ValueError("z must be in (0, 1)")
        d = _per_species(restoration_discount, names, 0.0, "restoration_discount")
        if not np.isfinite(d).all() or (d < 0).any() or (d > 1).any():
            raise ValueError("restoration_discount must be in [0, 1]")
        keep = np.flatnonzero(table["valid"].to_numpy(bool) & (table["potential"].to_numpy(np.float64) > 0))
        current = table["current"].to_numpy(np.float64)[keep].copy()
        base = current.copy()
        blocks, habs = [], []
        for code, act in enumerate(ACTIONS):
            cols = np.flatnonzero(self.action == code)
            if cols.size == 0:
                continue
            src = gain if act == "restore" else loss
            g = sp.csr_matrix(src)[keep][:, self.index[cols]].tocsr()
            g.sort_indices()
            hab = g.copy()
            hab.data = hab.data / self.area[cols][hab.indices]
            if act == "restore":
                g = g.multiply(sp.csr_matrix((1.0 - d[keep])[:, None])).tocsr()
                hab = hab.multiply(sp.csr_matrix((1.0 - d[keep])[:, None])).tocsr()
            elif act == "conserve":
                # without the plan the natural land of the units is lost (by the threat): conserving keeps it
                base = base - g @ (self.threat[cols] * self.natural[cols])
                w = sp.diags(self.threat[cols])
                g, hab = (g @ w).tocsr(), (hab @ w).tocsr()
            else:
                g, hab = -g, -hab
            blocks.append(g)
            habs.append(hab)
        gd = sp.hstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]
        hb = sp.hstack(habs).tocsr() if len(habs) > 1 else habs[0]
        gd.sort_indices()
        hb.sort_indices()
        risk = _Risk(name=self._name(name, "extinction_risk"), weight=float(weight), species=tuple(names[keep]),
                     current=current, potential=table["potential"].to_numpy(np.float64)[keep].copy(), gain=gd,
                     hab=hb, z=zs[keep].copy(), discount=d[keep], base=np.maximum(base, 0.0))
        return replace(self, benefits=self.benefits + (risk,))

    def add_linear_benefit(self, values, weight: float = 1.0, name: str | None = None,
                           per: str = "area") -> RestorationProblem:
        """A benefit proportional to the area acted on, such as carbon.

        ``values``: a raster on the problem's grid, a number or one value per unit, or a mapping action
        -> values (0 for the actions left out); per km² (``per="area"``) or per cell in full
        (``per="cell"``) of the action. A loss is negative (the carbon stock of converted land, say);
        conservation's values count the land the plan saves from loss (they are multiplied by the
        threat)."""
        v = _per_column(self, values, "values")
        if not np.isfinite(v).all():
            raise ValueError("values: NaN or infinite on a planning unit")
        v = np.where(self.action == _CODE["conserve"], v * self.threat, v)
        if per == "area":
            coef, layer = v * self.area, v
        elif per == "cell":
            coef, layer = v.copy(), v / self.area
        else:
            raise ValueError("per: 'area' or 'cell'")
        return replace(self, benefits=self.benefits + (_Linear(self._name(name, "linear"), float(weight), coef,
                                                               layer),))

    def add_intactness_benefit(self, hfi, reduction=None, radius: int = 3, weight: float = 1.0, *, increase=None,
                               gamma: float | None = None, beta: float = -0.2, z: float = 0.5, cell_size=None,
                               name: str | None = None) -> RestorationProblem:
        """Ecological intactness (Beyer et al. 2020): restoring a share x of a cell lowers its human
        footprint by ``reduction`` x, converting it raises the footprint by ``increase`` x (and natural
        land lost for want of conservation by ``increase`` times the threat, which conserving spares);
        the benefit is the total intactness gained, linearised at the plan of nothing
        (:func:`landschaft.intactness_gradient`; with ``method="steps"`` it is recomputed at every
        step, as plangea does). ``hfi``, ``reduction`` and ``increase`` on the problem's grid (or
        numbers); the other parameters as :func:`landschaft.intactness`."""
        from ..intactness import INTACTNESS_GAMMA

        gamma = INTACTNESS_GAMMA if gamma is None else float(gamma)
        _, h = _grid_values(hfi, "hfi")
        if h.shape != self.shape:
            raise ValueError(f"hfi: the grid {h.shape} is not the problem's {self.shape}")
        acts = self.actions
        change = np.zeros(self.n)
        h0 = h.ravel().copy()
        if "restore" in acts:
            if reduction is None:
                raise ValueError("restoration needs the reduction of the footprint per unit restored")
            m = self.action == _CODE["restore"]
            change[m] = -np.nan_to_num(_on_units(self, reduction, "reduction"))[m]
        if "conserve" in acts or "convert" in acts:
            if increase is None:
                raise ValueError("conservation and conversion need the increase of the footprint per unit converted")
            inc = np.nan_to_num(_on_units(self, increase, "increase"))
            m = self.action == _CODE["convert"]
            change[m] = inc[m]
            m = self.action == _CODE["conserve"]
            if m.any():
                change[m] = -inc[m] * self.threat[m]
                lost = np.bincount(self.index[m], weights=inc[m] * self.threat[m] * self.natural[m],
                                   minlength=h0.size)
                h0 = h0 + lost
        fp = _Footprint(hfi=hfi, h0=h0, change=change, radius=int(radius), gamma=gamma, beta=float(beta), z=float(z),
                        cell_size=cell_size)
        coef = _intactness_coef(self, fp, np.zeros(self.n))
        lin = _Linear(self._name(name, "intactness"), float(weight), coef, coef.copy(), intactness=fp)
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
    def _area_coef(self, action) -> np.ndarray:
        """The area a unit share of every column counts for: of one action, of all of them (``"all"``),
        or the change of natural land (``"net"``: restored minus converted)."""
        acts = self.actions
        if action is None:
            action = "net" if ("restore" in acts and "convert" in acts) else "all"
        if action == "all":
            return self.area.copy()
        if action == "net":
            sign = np.select([self.action == _CODE["restore"], self.action == _CODE["convert"]], [1.0, -1.0], 0.0)
            return self.area * sign
        if action not in ACTIONS:
            raise ValueError("action: 'restore', 'conserve', 'convert', 'net' or 'all'")
        if action not in acts:
            raise ValueError(f"action: the problem has no {action!r} ({', '.join(acts)})")
        return np.where(self.action == _CODE[action], self.area, 0.0)

    def add_area_target(self, target, sense: str = "<=", regions=None, action: str | None = None,
                        relative: bool = False) -> RestorationProblem:
        """The area acted on (km²): in the whole landscape, or in every region.

        Parameters
        ----------
        target
            A number (km²), or with ``regions`` a mapping region -> target, or a table with columns
            ``region``, ``target`` and optionally ``sense``.
        sense
            ``"<="`` (at most: the most valuable cells for that area), ``">="`` or ``"="``.
        regions
            A raster of region codes on the problem's grid (NaN: no region) or one code per unit.
        action
            The area of ``"restore"``, ``"conserve"`` or ``"convert"``, of ``"all"`` of them, or the
            ``"net"`` change of natural land (restored minus converted). Default: ``"net"`` when the
            plan restores and converts, ``"all"`` otherwise.
        relative
            The targets are shares (0-1) of the area that can be acted on (in the landscape or in every
            region): plangea's flat targets.
        """
        return self._add_rows(self._area_coef(action), target, sense, regions, relative)

    def add_linear_target(self, values, target, sense: str = "<=", regions=None, per: str = "area",
                          relative: bool = False) -> RestorationProblem:
        """A target on a value other than the area (production, opportunity cost, carbon ...): the sum of
        ``values`` over the plan, in the landscape or in every region.

        ``values``: a raster, a number, one value per unit or a mapping action -> values; per km²
        (``per="area"``) or per cell in full (``per="cell"``) of the action. ``target``, ``sense``,
        ``regions`` and ``relative`` as :meth:`add_area_target` (plangea's targets of other types)."""
        v = _per_column(self, values, "values")
        if not np.isfinite(v).all():
            raise ValueError("values: NaN or infinite on a planning unit")
        if per == "area":
            coef = v * self.area
        elif per == "cell":
            coef = v.copy()
        else:
            raise ValueError("per: 'area' or 'cell'")
        return self._add_rows(coef, target, sense, regions, relative)

    def add_projected_change(self, change, regions, sense: str = "=", restrict: bool = True,
                             compensate: bool | None = None) -> RestorationProblem:
        """Projected changes of natural land per region (plangea's econometric targets): positive, land
        spared for restoration; negative, a demand for land to convert.

        Parameters
        ----------
        change
            A mapping region -> change (km²) or a table with columns ``region``, ``change`` and
            optionally ``sense``.
        regions
            A raster of region codes on the problem's grid or one code per unit.
        sense
            Of every region's row (``"="``: the change as projected).
        restrict
            A plan that restores and converts acts in every region in the projected direction only:
            restoring where land is spared, converting where it is demanded, neither where the change
            is 0 (what plangea's ``harmonize_sr_r_or_c_bvals`` meant to do). Its rows are the net change.
        compensate
            A plan that only restores cannot convert: with ``compensate`` (the default, plangea's
            ``harmonize_sr_sparable``) the land demanded elsewhere is taken from the restoration, every
            region with spared land restoring its share of the net spared land; without it, the regions
            restore what they spare. A plan that only converts converts what the regions demand.
        """
        acts = self.actions
        if "conserve" in acts:
            raise ValueError("projected changes of natural land are for plans that restore or convert")
        codes = _on_units(self, regions, "regions")
        if isinstance(change, Mapping):
            tab = pd.DataFrame({"region": list(change.keys()), "change": list(change.values())})
        else:
            tab = pd.DataFrame(change)
            if "region" not in tab or "change" not in tab:
                raise ValueError("change: a mapping region -> change or a table with region and change")
        ch = tab["change"].to_numpy(np.float64)
        if not np.isfinite(ch).all():
            raise ValueError("change: NaN or infinite")
        restore, convert = "restore" in acts, "convert" in acts
        upper = self.upper
        if restore and convert:
            if compensate:
                raise ValueError("compensate is for plans that only restore")
            target = ch
            if restrict:
                upper = upper.copy()
                r_col, d_col = self.action == _CODE["restore"], self.action == _CODE["convert"]
                for code, c in zip(tab["region"].to_numpy(np.float64), ch):
                    inside = codes == code
                    upper[inside & (d_col if c > 0 else (r_col if c < 0 else True))] = 0.0
            coef = self._area_coef("net")
        elif restore:
            spared = ch[ch > 0].sum()
            if compensate is None or compensate:
                net = max(float(ch.sum()), 0.0)
                target = np.where(ch > 0, ch * (net / spared if spared > 0 else 0.0), 0.0)
            else:
                target = np.where(ch > 0, ch, 0.0)
            coef = self._area_coef("restore")
        else:
            if compensate:
                raise ValueError("compensate is for plans that only restore")
            target = np.where(ch < 0, -ch, 0.0)
            coef = self._area_coef("convert")
        rows = tab.assign(target=target)[["region", "target"] + (["sense"] if "sense" in tab else [])]
        return replace(self, upper=upper)._add_rows(coef, rows, sense, regions, False)

    def _add_rows(self, coef: np.ndarray, target, sense: str, regions, relative: bool) -> RestorationProblem:
        """Target rows of the coefficients ``coef`` per column: in the landscape or per region."""
        if sense not in ("<=", ">=", "="):
            raise ValueError("sense: '<=', '>=' or '='")
        if regions is None:
            if not np.isscalar(target):
                raise ValueError("a target per region needs regions")
            rows = sp.csr_matrix(coef[None, :])
            rhs = np.array([float(target)])
            t = _Targets(rows, rhs, (sense,), ("landscape",))
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
                cells = np.flatnonzero((codes == code) & (coef != 0))
                r += [k] * cells.size
                c += cells.tolist()
                v += coef[cells].tolist()
            rows = sp.csr_matrix((v, (r, c)), shape=(len(tab), self.n))
            t = _Targets(rows, tab["target"].to_numpy(np.float64), tuple(senses),
                         tuple(str(x) for x in tab["region"].tolist()))
        if relative:
            # shares of what can be acted on in each row
            avail = abs(t.rows) @ self.upper
            t = replace(t, rhs=t.rhs * avail)
        if not np.isfinite(t.rhs).all():
            raise ValueError("targets must be finite")
        return replace(self, targets=self.targets + (t,))

    def add_budget(self, budget: float, cost) -> RestorationProblem:
        """Spend at most ``budget``, ``cost`` per km² acted on (a raster, a number, one value per unit
        or a mapping action -> cost; the opportunity cost of the land, say)."""
        c = self._cost(cost)
        if not math.isfinite(float(budget)):
            raise ValueError("budget must be finite")
        return replace(self, cost=c * self.area, cost_layer=c, budget=float(budget))

    def add_max_benefit_objective(self) -> RestorationProblem:
        """The most weighted benefit within the targets and the budget (the default)."""
        return replace(self, objective=("max_benefit",))

    def add_min_cost_objective(self, benefit: float, cost=None) -> RestorationProblem:
        """The cheapest plan whose weighted benefit reaches ``benefit``; ``cost`` per km² acted on (or the
        budget's, without it)."""
        if cost is None and self.cost_layer is None:
            raise ValueError("the minimum cost needs a cost")
        out = self if cost is None else replace(self, cost=self._cost(cost) * self.area, cost_layer=self._cost(cost))
        return replace(out, objective=("min_cost", float(benefit)))

    def _cost(self, cost) -> np.ndarray:
        c = _per_column(self, cost, "cost")
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
            x = _solve_steps(self, int(steps))[-1]
            bound, status, rounds = math.nan, "heuristic", int(steps)
        else:
            raise ValueError("method: 'exact' or 'steps'")
        return self._solution(x, bound=bound, status=status, method=method, rounds=rounds,
                              seconds=time.perf_counter() - t0)

    def evaluate(self, plan=None, *, restored=None, conserved=None, converted=None) -> RestorationSolution:
        """The benefits, cost and area of a plan: the shares of every column (``plan``: a raster on the
        grid or one value per unit when the problem has one action, or one value per column), or the
        shares ``restored``, ``conserved`` and ``converted`` (rasters or one value per unit; 0 for those
        left out)."""
        given = {"restore": restored, "conserve": conserved, "convert": converted}
        if plan is not None:
            if any(v is not None for v in given.values()):
                raise ValueError("give the plan or the shares of each action, not both")
            if len(self.actions) > 1 and _is_grid(plan):
                raise ValueError("a plan of several actions: give restored=, conserved= and converted=")
            x = _on_units(self, plan, "plan")
        else:
            x = np.zeros(self.n)
            for a, v in given.items():
                if v is None:
                    continue
                if a not in self.actions:
                    raise ValueError(f"the problem has no {a!r} ({', '.join(self.actions)})")
                m = self.action == _CODE[a]
                x[m] = _on_units(self, v, f"{a}")[m]
        return self._solution(np.nan_to_num(x), bound=math.nan, status="given", method="given", rounds=0,
                              seconds=0.0)

    def uniform(self, area: float | None = None) -> RestorationSolution:
        """plangea's uniform scenario: every unit acted on by the same share of what it can be, for a
        total ``area`` (default: the first landscape area target). For a plan of one action."""
        if len(self.actions) > 1:
            raise ValueError("the uniform plan is for a problem of one action")
        if area is None:
            land = [t for t in self.targets if t.labels == ("landscape",)]
            if not land:
                raise ValueError("give the area (or add a landscape area target)")
            area = float(land[0].rhs[0])
        total = float(np.sum(self.upper * self.area))
        x = self.upper * (float(area) / total if total > 0 else 0.0)
        return self._solution(x, bound=math.nan, status="uniform", method="uniform", rounds=0, seconds=0.0)

    def tradeoff(self, levels=(0.25, 0.5, 0.75, 1.0), *, nested: bool = True, targets: str = "landscape",
                 method: str = "exact", steps: int = 1, threshold: float = 0.0) -> RestorationTradeOff:
        """The plans for a growing target: the trade-off curve, and the priority of every cell.

        Parameters
        ----------
        levels
            Shares of the targets (in (0, 1], increasing): plangea's gradient steps.
        nested
            Every level's plan contains the previous one (a priority order: the exact best plan given
            the earlier levels, with their shares as the least shares), or (False) every level is the
            exact optimum on its own (the frontier: plans need not contain each other).
        targets
            The target rows that grow: the landscape's (``"landscape"``, plangea's default; the regions'
            targets hold at every level) or ``"all"``.
        method
            ``"exact"`` or ``"steps"`` (plangea's procedure, nested by construction: the levels follow
            one another, ``steps`` parts each, the layers recomputed from the whole plan so far).
        threshold
            A cell counts in a level when its share is above it (the gradient).

        Returns
        -------
        RestorationTradeOff
            The table (one row per level: the target, area, cost, benefits), the plans and the gradient
            (plangea's ``post_gradient_output``): for restoration and conservation, the number of levels
            whose plan acts on the cell (the most for the first chosen); for conversion, the levels plus
            one minus that number (the least for the first converted).
        """
        lv = np.asarray(levels, dtype=np.float64).ravel()
        if lv.size == 0 or (lv <= 0).any() or (lv > 1).any() or (np.diff(lv) <= 0).any():
            raise ValueError("levels: increasing shares in (0, 1]")
        if targets not in ("landscape", "all"):
            raise ValueError("targets: 'landscape' or 'all'")
        flags = [np.full(t.rhs.size, targets == "all" or t.labels == ("landscape",)) for t in self.targets]
        scaled = np.concatenate(flags) if flags else np.zeros(0, dtype=bool)
        if not scaled.any():
            raise ValueError("the trade-off needs a target to grow (add a landscape area target)")
        t0 = time.perf_counter()
        sols = []
        if method == "steps":
            plans = _solve_steps(self, int(steps), lv, scaled)
            for x in plans:
                sols.append(self._solution(x, bound=math.nan, status="heuristic", method="steps", rounds=int(steps),
                                           seconds=time.perf_counter() - t0))
        elif method == "exact":
            prev = None
            for level in lv:
                ts = tuple(replace(t, rhs=t.rhs * level) if f.all() else t for t, f in zip(self.targets, flags))
                q = replace(self, targets=ts)
                if nested and prev is not None:
                    q = replace(q, lower=np.maximum(prev, self._lb()))
                s = q.solve()
                sols.append(s)
                prev = s.values
        else:
            raise ValueError("method: 'exact' or 'steps'")
        land = [t for t in self.targets if t.labels == ("landscape",)]
        rows = []
        for level, s in zip(lv, sols):
            row = {"level": float(level), "target": float(land[0].rhs[0] * level) if land else math.nan,
                   "area": s.area, "cost": s.cost, "benefit": s.benefit, "status": s.status}
            for a, v in s.areas.items():
                row[f"area_{a}"] = v
            for b in s.benefits.itertuples():
                row[b.benefit] = b.value
            rows.append(row)
        table = pd.DataFrame(rows)
        gradients = {}
        for code, act in enumerate(ACTIONS):
            m = self.action == code
            if not m.any():
                continue
            count = np.sum([s.values[m] > threshold for s in sols], axis=0).astype(np.float64)
            if act == "convert":
                count = lv.size + 1 - count
            grid = np.full(self.shape[0] * self.shape[1], np.nan)
            grid[self.index[m]] = count
            grid = grid.reshape(self.shape)
            da = self.template.copy(data=grid) if self.template is not None else xr.DataArray(grid, dims=("y", "x"))
            da.name = f"gradient_{act}"
            gradients[act] = da
        return RestorationTradeOff(table=table, solutions=sols, gradients=gradients)

    def compare_weights(self, weights, *, method: str = "exact", steps: int = 1) -> RestorationTradeOff:
        """Solve with every set of the benefits' weights and rank the sets by plangea's metric
        (``post_choose_weights``): every benefit's value (and the cost, inverted) rescaled to 0-1 over
        the sets, the mean over them divided by their standard deviation (see :func:`mean_sd`); the
        best set does well on every benefit at once.

        ``weights``: a list of mappings benefit name -> weight (the others keep theirs). Returns the
        table (``w.<benefit>``, the values, ``cost``, ``mean``, ``sd``, ``score``, ``rank``) and the
        plans; ``best()`` is the first in rank."""
        from ._milp_calibrate import mean_sd

        sets = [dict(w) for w in weights]
        if not sets:
            raise ValueError("no weights")
        names = [b.name for b in self.benefits]
        for w in sets:
            unknown = set(w) - set(names)
            if unknown:
                raise ValueError(f"weights: no benefit {sorted(unknown)}")
        sols, rows = [], []
        for w in sets:
            q = replace(self, benefits=tuple(replace(b, weight=float(w.get(b.name, b.weight))) for b in self.benefits))
            s = q.solve(method=method, steps=steps)
            sols.append(s)
            row = {f"w.{b.name}": float(w.get(b.name, b.weight)) for b in self.benefits}
            row.update(dict(zip(s.benefits["benefit"], s.benefits["value"])))
            row["cost"] = s.cost
            rows.append(row)
        table = pd.DataFrame(rows)
        crit = list(names) + (["cost"] if self.cost is not None else [])
        score = mean_sd(table[crit].to_numpy(np.float64), ["+"] * len(names) + ["-"] * (len(crit) - len(names)))
        table = pd.concat([table, score[["mean", "sd", "score", "rank"]]], axis=1)
        return RestorationTradeOff(table=table, solutions=sols, gradients={})

    def _solution(self, x, *, bound, status, method, rounds, seconds) -> RestorationSolution:
        x = np.asarray(x, dtype=np.float64)
        rows = []
        total = 0.0
        for b in self.benefits:
            v = _benefit_value(self, b, x)
            rows.append({"benefit": b.name, "kind": "extinction_risk" if isinstance(b, _Risk) else "linear",
                         "value": v, "weight": b.weight, "weighted": b.weight * v})
            total += b.weight * v
        table = pd.DataFrame(rows, columns=["benefit", "kind", "value", "weight", "weighted"])
        cost = float(self.cost @ x) if self.cost is not None else math.nan
        objective = cost if self.objective[0] == "min_cost" and method != "steps" else total
        maps, areas = {}, {}
        for code, act in enumerate(ACTIONS):
            m = self.action == code
            if not m.any():
                maps[act] = None
                continue
            grid = np.full(self.shape[0] * self.shape[1], np.nan)
            grid[self.index[m]] = x[m]
            grid = grid.reshape(self.shape)
            da = self.template.copy(data=grid) if self.template is not None else xr.DataArray(grid, dims=("y", "x"))
            da.name = {"restore": "restored", "conserve": "conserved", "convert": "converted"}[act]
            maps[act] = da
            areas[act] = float(self.area[m] @ x[m])
        return RestorationSolution(restored=maps["restore"], values=x, area=float(self.area @ x), cost=cost,
                                   benefits=table, benefit=total, objective=objective, bound=bound, status=status,
                                   method=method, rounds=rounds, seconds=seconds, conserved=maps["conserve"],
                                   converted=maps["convert"], areas=areas)


@dataclass
class RestorationSolution(Saveable):
    """A restoration plan.

    Attributes
    ----------
    restored, conserved, converted
        The share of every cell restored, conserved and converted (NaN outside the planning units;
        None for an action the problem does not have).
    values
        The shares, one value per column (the units of every action, block after block: restore,
        conserve, convert).
    area, areas
        The area acted on (km²) and the area of every action.
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
        ``"optimal"`` (within the tolerance), ``"time_limit"``, ``"iteration_limit"``, ``"heuristic"``
        (steps) ...; the method, the iterations (vertices or bisection steps; steps for
        ``method="steps"``), the time.
    """

    restored: xr.DataArray | None = field(repr=False)
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
    conserved: xr.DataArray | None = field(default=None, repr=False)
    converted: xr.DataArray | None = field(default=None, repr=False)
    areas: dict = field(default_factory=dict)

    def _save_items(self) -> dict:
        out = {"benefits": self.benefits, "area": self.area, "cost": self.cost, "benefit": self.benefit,
               "objective": self.objective, "bound": self.bound, "status": self.status, "method": self.method}
        for key in ("restored", "conserved", "converted"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        return out


@dataclass
class RestorationTradeOff(Saveable):
    """Plans over a trade-off (:meth:`RestorationProblem.tradeoff`) or over sets of weights
    (:meth:`RestorationProblem.compare_weights`).

    Attributes
    ----------
    table
        One row per plan.
    solutions
        The :class:`RestorationSolution` of every row.
    gradients
        The trade-off's gradient per action (an empty mapping for weights).
    """

    table: pd.DataFrame = field(repr=False)
    solutions: list = field(repr=False)
    gradients: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return f"RestorationTradeOff({len(self.solutions)} plans)"

    @property
    def gradient(self) -> xr.DataArray:
        """The gradient of a plan of one action (see :meth:`RestorationProblem.tradeoff`)."""
        if len(self.gradients) != 1:
            raise ValueError("the gradient of a plan of several actions: see gradients")
        return next(iter(self.gradients.values()))

    def best(self) -> RestorationSolution:
        """The plan ranked first (compare_weights), or the knee of the curve of benefit against area."""
        from ._milp_calibrate import knee_point

        if "rank" in self.table:
            return self.solutions[int(np.argmin(self.table["rank"].to_numpy()))]
        return self.solutions[knee_point(self.table["area"].to_numpy(), -self.table["benefit"].to_numpy())]

    def _save_items(self) -> dict:
        out = {"table": self.table}
        out.update({f"gradient_{a}": g for a, g in self.gradients.items()})
        return out


def restoration_problem(restorable=None, *, conservable=None, convertible=None, limit=1.0, threat=1.0,
                        cell_area=None) -> RestorationProblem:
    """A restoration problem: how much of every converted cell to restore (and of natural land to
    conserve or convert).

    The planning units are the cells with land an action can take; the decision is the share x_i of
    each, from 0 to the share that can be times ``limit``. Then add benefits
    (:meth:`~RestorationProblem.add_extinction_risk_benefit`,
    :meth:`~RestorationProblem.add_linear_benefit`, :meth:`~RestorationProblem.add_intactness_benefit`),
    targets (:meth:`~RestorationProblem.add_area_target`), a budget, and solve.

    Parameters
    ----------
    restorable
        The share of every cell that can be restored: a raster, or a
        :class:`~landschaft.OriginalVegetation` (its anthropic share).
    conservable
        The share of every cell's natural land that conservation can keep (plangea's ``C``): without
        the plan, ``threat`` of it is lost (plangea: all of it). The benefits count what conserving
        saves.
    convertible
        The share of every cell's natural land that can be converted (plangea's ``D``, a demand for
        land met where it costs the least benefit). With ``restorable`` too, the plan restores some
        cells and converts others (plangea's ``R + D``). Not with ``conservable``.
    limit
        plangea's bound limit: a number or a raster multiplying the shares, or a mapping action ->
        limit.
    threat
        The share of the conservable land lost without conservation, in [0, 1]: a number or a raster
        (default 1, plangea's).
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
    given = [(a, d) for a, d in zip(ACTIONS, (restorable, conservable, convertible)) if d is not None]
    if not given:
        raise ValueError("give the land that can be restored (restorable), conserved or converted")
    if conservable is not None and convertible is not None:
        raise ValueError("conservable and convertible: the same natural land is either conserved or converted")
    template, shape = None, None
    shares = {}
    for act, data in given:
        t, v = _grid_values(data, _SHARE[act])
        if shape is None:
            shape = v.shape
        elif v.shape != shape:
            raise ValueError(f"{_SHARE[act]}: the grid {v.shape} is not {shape}")
        template = template if template is not None else t
        shares[act] = np.nan_to_num(v, nan=0.0)

    def grid_of(value, what):
        if np.isscalar(value):
            return np.full(shape, float(value))
        g = _grid_values(value, what)[1]
        if g.shape != shape:
            raise ValueError(f"{what}: not on the grid of {_SHARE[given[0][0]]}")
        return g

    if isinstance(limit, Mapping):
        unknown = set(limit) - {a for a, _ in given}
        if unknown:
            raise ValueError(f"limit: no action {sorted(unknown)} in the problem")
    th = grid_of(threat, "threat") if conservable is not None else None
    if th is not None and (np.nanmin(th) < 0 or np.nanmax(th) > 1):
        raise ValueError("threat must be in [0, 1]")
    idx, up, act_codes, nat, thr = [], [], [], [], []
    for act, _ in given:
        lim_v = limit.get(act, 1.0) if isinstance(limit, Mapping) else limit
        lim = grid_of(lim_v, "limit")
        u = (shares[act] * np.nan_to_num(lim, nan=0.0)).ravel()
        cells = np.flatnonzero(u > 0)
        idx.append(cells)
        up.append(u[cells])
        act_codes.append(np.full(cells.size, _CODE[act], dtype=np.int8))
        nat.append(shares[act].ravel()[cells] if act == "conserve" else np.zeros(cells.size))
        thr.append(np.nan_to_num(th.ravel()[cells], nan=0.0) if act == "conserve" else np.ones(cells.size))
    index = np.concatenate(idx)
    if index.size == 0:
        raise ValueError("no cell can be " + " or ".join({"restore": "restored", "conserve": "conserved",
                                                          "convert": "converted"}[a] for a, _ in given))
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
            raise ValueError(f"cell_area: not on the grid of {_SHARE[given[0][0]]}")
    a = area.ravel()[index]
    if not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError("cell_area: positive on every cell of the planning units")
    if template is not None:
        template = template.copy(data=np.full(shape, np.nan))
    return RestorationProblem(template=template, shape=shape, index=index.astype(np.int64),
                              upper=np.concatenate(up).copy(), area=a.copy(), action=np.concatenate(act_codes),
                              natural=np.concatenate(nat), threat=np.concatenate(thr))


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
        natural plus the anthropic area, the gain (the cell's area per unit restored, in the cells
        with anthropic land) and the loss (the cell's area per unit of natural land lost, in the cells
        with natural land).

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
    cur, pot, rows, cols, vals, lrows, lcols, lvals = [], [], [], [], [], [], [], []
    for k, code in enumerate(codes):
        cells = np.flatnonzero(e == code)
        cur.append(float(np.sum(a[cells] * nv[cells])))
        pot.append(float(np.sum(a[cells] * (nv[cells] + av[cells]))))
        g = cells[av[cells] > 0]
        rows += [k] * g.size
        cols += g.tolist()
        vals += a[g].tolist()
        g = cells[nv[cells] > 0]
        lrows += [k] * g.size
        lcols += g.tolist()
        lvals += a[g].tolist()
    gain = sp.csr_matrix((vals, (rows, cols)), shape=(codes.size, e.size))
    loss = sp.csr_matrix((lvals, (lrows, lcols)), shape=(codes.size, e.size))
    label = [str(int(c)) if float(c).is_integer() else str(c) for c in codes]
    table = pd.DataFrame({"species": label, "current": cur, "potential": pot,
                          "range_cells": [int(np.sum(e == c)) for c in codes], "classes": ", ".join(natural),
                          "valid": (np.array(pot) > 0)})
    return HabitatAreas(table=table, gain=gain, _template=template, loss=loss)


def ecosystem_targets(areas, target: float, *, minimum: float = 0.0, z: float = PLANGEA_Z,
                      half_life: float = 0.0, sense: str = "<=", method: str = "exact",
                      steps: int = 10) -> pd.DataFrame:
    """Share a restoration target among ecosystems for the least risk of collapse (plangea's
    ecosystemic sub-region targets, ``harmonize_sr_ecosys``).

    Every ecosystem e with natural area N_e of a potential T_e (:func:`ecosystem_areas`) gets a_e of
    the target, between its least share and its converted area T_e - N_e, for the most total of
    ((N_e + a_e) / T_e)^z, each ecosystem's marginal value damped by exp(``half_life`` (N_e + a_e) /
    T_e) (plangea's "yukawa" factor; 0: none).

    Parameters
    ----------
    areas
        A :class:`~landschaft.sdm.HabitatAreas` of ecosystems (:func:`ecosystem_areas`).
    target
        The area to share (km²).
    minimum
        Every ecosystem's least share, as a part (0-1) of an equal split of the target (plangea's
        ``subregion_ecosystems_minimum_par``), up to its converted area.
    z
        The exponent of the species-area relationship.
    half_life
        The damping exponent (below 1 - z, so the marginal value falls with the area).
    sense
        The sense of the targets returned (plangea's ``subregion_ecosystemic_targets_sense``).
    method
        ``"exact"``: the optimum (the marginal values equal among the ecosystems not at a bound, by
        bisection). ``"steps"``: plangea's procedure (the target in ``steps`` parts, each a linear
        program on the marginal values, recomputed between them).

    Returns
    -------
    pandas.DataFrame
        ``region`` (the ecosystem's code) and ``target``, ``sense``: a table for
        :meth:`RestorationProblem.add_area_target` with the ecosystem raster as ``regions``.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> cover = xr.DataArray(np.array([[[0.8, 0.2, 0.1, 0.5]], [[0.2, 0.8, 0.9, 0.5]]]),
    ...                      dims=("band", "y", "x"), coords={"band": ["forest", "crops"]})
    >>> eco = np.array([[1.0, 1.0, 2.0, 2.0]])
    >>> ea = ls.prioritize.ecosystem_areas(cover, ["forest"], ["crops"], eco, cell_area=1.0)
    >>> ls.prioritize.ecosystem_targets(ea, 1.0).round(4)
       region  target sense
    0     1.0     0.3    <=
    1     2.0     0.7    <=
    """
    tab = areas.table
    keep = tab["valid"].to_numpy(bool) & (tab["potential"].to_numpy(np.float64) > 0)
    n_e = tab["current"].to_numpy(np.float64)[keep]
    t_e = tab["potential"].to_numpy(np.float64)[keep]
    codes = pd.to_numeric(tab["species"].astype(str)[keep], errors="coerce").to_numpy(np.float64)
    if not 0 <= minimum <= 1:
        raise ValueError("minimum must be in [0, 1]")
    if not 0 < z < 1:
        raise ValueError("z must be in (0, 1)")
    if half_life >= 1 - z:
        raise ValueError("half_life must be below 1 - z (the marginal value must fall with the area)")
    if sense not in ("<=", ">=", "="):
        raise ValueError("sense: '<=', '>=' or '='")
    target = float(target)
    up = np.maximum(t_e - n_e, 0.0)
    k = n_e.size
    if method == "steps":
        a = _ecosystem_steps(n_e, t_e, up, target, minimum, z, half_life, int(steps))
    elif method == "exact":
        least = np.minimum(target / k * minimum, up)

        def marginal(a):
            r = (n_e + a) / t_e
            with np.errstate(divide="ignore"):
                return z * r ** (z - 1.0) * np.exp(half_life * r) / t_e

        def share(lam):
            """Every ecosystem's area where its marginal value falls to lam (within its bounds)."""
            lo, hi = least.copy(), up.copy()
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                above = marginal(mid) > lam
                lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
            return np.where(marginal(up) >= lam, up, np.where(marginal(least) <= lam, least, 0.5 * (lo + hi)))

        if up.sum() <= target:
            a = up
        else:
            lam_lo, lam_hi = 0.0, float(np.max(np.where(np.isfinite(marginal(least)), marginal(least), 0.0))) + 1.0
            if not np.isfinite(marginal(least)).all():
                lam_hi = float(np.max(marginal(np.minimum(least + 1e-12 * np.maximum(t_e, 1.0), up)))) * 2 + 1.0
            for _ in range(200):
                lam = 0.5 * (lam_lo + lam_hi)
                if share(lam).sum() > target:
                    lam_lo = lam
                else:
                    lam_hi = lam
            a = share(lam_hi)
            # the last bit of the target to the ecosystems at the marginal value
            free = (a > least) & (a < up)
            rest = target - a.sum()
            if rest > 0 and free.any():
                room = up[free] - a[free]
                a[free] += room * min(1.0, rest / room.sum())
    else:
        raise ValueError("method: 'exact' or 'steps'")
    return pd.DataFrame({"region": codes, "target": a, "sense": sense})


def _ecosystem_steps(n_e, t_e, up, target, minimum, z, half_life, steps) -> np.ndarray:
    """plangea's harmonize_sr_ecosys: the target in steps, each a linear program on the ecosystems'
    marginal values (plangea's finite difference, times the damping), every ecosystem at least its
    share of the step; the marginal values recomputed between the steps."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    k = n_e.size

    def values(nat):
        r1 = np.maximum(1.0 - (nat / t_e) ** z, 0.0)
        r2 = np.maximum(1.0 - ((nat + PLANGEA_DELTA) / t_e) ** z, 0.0)
        return np.abs(r2 - r1) / PLANGEA_DELTA * np.exp(half_life * nat / t_e)

    obj = values(n_e)
    ub = up.copy()
    step = target / steps
    least = np.minimum(np.full(k, step / k * minimum), ub)
    total = np.zeros(k)
    for _ in range(steps):
        # the linear program: x >= least, sum x <= step, x <= ub; the rest of the step by value
        x = least.copy()
        left = step - x.sum()
        for e in np.argsort(-obj, kind="stable"):
            if left <= 0:
                break
            add = min(ub[e] - x[e], left)
            if add > 0:
                x[e] += add
                left -= add
        ub = np.maximum(ub - x, 0.0)
        least = np.minimum(least, ub)
        total = total + x
        obj = values(n_e + total)
    return total


# ----------------------------------------------------------------------------- the benefits

def _h(r: np.ndarray, z: float) -> np.ndarray:
    """1 - risk: min(r, 1)^z."""
    return np.minimum(np.maximum(r, 0.0), 1.0) ** z


def _ratio(b: _Risk, x: np.ndarray) -> np.ndarray:
    return (b.base + b.gain @ x) / b.potential


def _benefit_value(p: RestorationProblem, b, x: np.ndarray) -> float:
    if isinstance(b, _Linear):
        return float(b.coef @ x)
    return float(np.sum(_h(_ratio(b, x), b.z) - _h(b.base / b.potential, b.z)))


def _true_objective(p: RestorationProblem, x: np.ndarray) -> float:
    return sum(b.weight * _benefit_value(p, b, x) for b in p.benefits)


def _footprint_at(p: RestorationProblem, fp: _Footprint, x: np.ndarray) -> np.ndarray:
    """The human footprint (flat) under the plan x: without it, plus the columns' changes, at least 0."""
    h = fp.h0.copy()
    if x.any():
        d = np.bincount(p.index, weights=fp.change * x, minlength=h.size)
        ok = ~np.isnan(h)
        v = h[ok] + d[ok]
        h[ok] = np.where(v < 0, 0.0, v)
    return h


def _intactness_coef(p: RestorationProblem, fp: _Footprint, x: np.ndarray) -> np.ndarray:
    """The derivative of the total intactness in every column's share, at the plan x."""
    from ..intactness import intactness_gradient

    h = _footprint_at(p, fp, x).reshape(p.shape)
    grid = fp.hfi.copy(data=h) if isinstance(fp.hfi, xr.DataArray) and fp.hfi.ndim == 2 else h
    out = np.zeros(p.n)
    for code in np.unique(p.action):
        m = p.action == code
        a = np.zeros(h.size)
        a[p.index[m]] = -fp.change[m]  # what a unit share lowers the footprint by
        g = intactness_gradient(grid, a.reshape(p.shape), fp.radius, gamma=fp.gamma, beta=fp.beta, z=fp.z,
                                cell_size=fp.cell_size)
        out[m] = np.nan_to_num(np.asarray(g.values, dtype=np.float64).ravel()[p.index[m]])
    return out


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


def _positive(g: sp.csr_matrix) -> sp.csr_matrix:
    return g if g.nnz == 0 or g.data.min() >= 0 else g.maximum(0.0).tocsr()


def _has_negative(g: sp.csr_matrix) -> np.ndarray:
    if g.nnz == 0 or g.data.min() >= 0:
        return np.zeros(g.shape[0], dtype=bool)
    return np.asarray((g < 0).sum(axis=1)).ravel() > 0


def _solve_sd(p: RestorationProblem) -> tuple[np.ndarray, float, str, int]:
    """The most weighted benefit by simplicial decomposition: the plan is a mix of vertices of the set of
    plans, each from a linear program over the units (the benefits' gradient at the current plan, the
    targets and the budget as rows), the mix by a small smooth master problem (SLSQP on the weights); the
    bound is the Frank-Wolfe one, F(x) + grad F(x) . (v - x) for the best vertex v, valid because the
    benefit is concave. Its linear programs have only the problem's few rows, so it scales with the
    units, where one linear program with tangent planes of every species' curve (Kelley's cutting planes,
    tried first) did not: 9 x 10^4 cells and 500 species, 0.57 s against more than 15 minutes.

    A species that reaches its potential habitat (r = 1) puts a kink in the benefit, where the
    Frank-Wolfe bound does not close. Such a species is lifted: its habitat gained becomes a variable y
    of its own, y <= g x and y <= P - A (one row and one column in the linear programs), on which the
    benefit is smooth up to r = 1. Species are lifted as the vertices overshoot their potential."""
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
    lb_x = p._lb()
    floors = [np.maximum(1e-9 * (b.base + _positive(b.gain) @ p.upper) / b.potential, 1e-15) for b in risks]
    caps = [np.maximum(b.potential - b.base, 0.0) for b in risks]
    # the least habitat gained: below 0 for species that conversion takes habitat from
    floor_y = [np.where(_has_negative(b.gain), -b.base, 0.0) for b in risks]
    lifted = [np.zeros(b.base.size, dtype=bool) for b in risks]

    def left() -> float:
        return deadline - time.perf_counter()

    def slopes(i: int, y: np.ndarray) -> np.ndarray:
        """The derivative of benefit i in the habitat gained y of every species: 0 past the potential,
        except for a lifted species (y <= P - A), whose left derivative it keeps there."""
        b = risks[i]
        r = (b.base + y) / b.potential
        d = b.z * np.maximum(np.minimum(r, 1.0), floors[i]) ** (b.z - 1.0) * (b.weight / b.potential)
        return np.where((r < 1.0) | lifted[i], d, 0.0)

    def vertex(cx: np.ndarray, dy: list) -> tuple[np.ndarray, bool]:
        """The best vertex for the objective cx on the units and dy on the lifted species."""
        cols = [(i, np.flatnonzero(lifted[i])) for i in range(len(risks))]
        m_y = sum(s.size for _, s in cols)
        if m_y == 0:
            A, obj, lb, ub, b_rhs, b_sense = rows, cx, lb_x, p.upper, rhs, sense
        else:
            # y_s - g_s x <= 0 for every lifted species, 0 <= y_s <= P_s - A_s
            g_rows = sp.vstack([risks[i].gain[s] for i, s in cols if s.size]).tocsr()
            A = sp.vstack([sp.hstack([rows, sp.csr_matrix((rows.shape[0], m_y))]),
                           sp.hstack([-g_rows, sp.identity(m_y, format="csr")])]).tocsr()
            obj = np.concatenate([cx] + [dy[i][s] for i, s in cols if s.size])
            lb = np.concatenate([lb_x] + [floor_y[i][s] for i, s in cols if s.size])
            ub = np.concatenate([p.upper] + [caps[i][s] for i, s in cols if s.size])
            b_rhs, b_sense = np.concatenate([rhs, np.zeros(m_y)]), tuple(sense) + ("<=",) * m_y
        m = OptimizationProblem(A=A, obj=obj, lb=lb, ub=ub, rhs=b_rhs, sense=b_sense, vtype=("C",) * (n + m_y),
                                modelsense="max", row_ids=("row",) * A.shape[0], col_ids=("x",) * (n + m_y))
        r = _solve_lp(p, m, max(left(), 1e-3), n)
        if r["status"] == "infeasible":
            raise RuntimeError("the restoration problem is infeasible (targets or budget)")
        if not r["has_solution"]:
            raise RuntimeError(f"no plan ({p.solver[0]}: {r['status']})")
        # only an optimal vertex gives a valid bound
        return np.clip(np.asarray(r["x"], dtype=np.float64)[:n], lb_x, p.upper), r["status"] == "optimal"

    # the vertices (sparse columns), their gains per risk benefit (species x vertices) and linear benefit;
    # a lifted species counts min(g v, P - A) of a vertex (its best y there)
    verts: list[sp.csr_matrix] = []
    gv = [np.zeros((b.base.size, 0)) for b in risks]
    lv = np.zeros(0)

    def effective(i: int) -> np.ndarray:
        return np.where(lifted[i][:, None], np.minimum(gv[i], caps[i][:, None]), gv[i])

    def add_vertex(v: np.ndarray) -> np.ndarray:
        nonlocal lv
        verts.append(sp.csr_matrix(v[None, :]))
        for i, b in enumerate(risks):
            gv[i] = np.hstack([gv[i], (b.gain @ v)[:, None]])
            lifted[i] |= (caps[i] > 0) & (gv[i][:, -1] >= caps[i] * (1.0 - 1e-12))
        lv = np.append(lv, lin @ v)

    def master(lam0: np.ndarray, ev: list) -> np.ndarray:
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
                v += b.weight * float(np.sum(_h((b.base + ev[i] @ lam) / b.potential, b.z)))
            return -v

        def grad(lam):
            g = lv.copy()
            for i in range(len(risks)):
                g += ev[i].T @ slopes(i, ev[i] @ lam)
            return -g

        res = minimize(f, lam0, jac=grad, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                       constraints=[{"type": "eq", "fun": lambda lam: np.sum(lam) - 1.0, "jac": lambda lam: np.ones(k)}],
                       options={"maxiter": 1000, "ftol": 1e-15})
        lam = np.clip(res.x, 0.0, None)
        lam = lam / lam.sum() if lam.sum() > 0 else lam0
        return lam if f(lam) <= f(lam0) else lam0

    add_vertex(vertex(lin + sum(b.gain.T @ slopes(i, np.zeros(b.base.size)) for i, b in enumerate(risks)),
                      [slopes(i, np.zeros(b.base.size)) for i, b in enumerate(risks)])[0])
    bound, best_x, best = math.inf, None, -math.inf
    status, it = "iteration_limit", 0
    lam = np.ones(1)
    for it in range(1, 5001):
        ev = [effective(i) for i in range(len(risks))]
        lam = master(lam, ev)
        x = np.zeros(n)
        for w, v in zip(lam, verts):
            if w > 0:
                x += w * v.toarray().ravel()
        x = np.clip(x, lb_x, p.upper)
        f = _true_objective(p, x)
        if f > best:
            best_x, best = x, f
        # the lifted objective at the mix (x, y): at most the plan's benefit f
        ys = [e @ lam for e in ev]
        f_lift = float(lv @ lam) + sum(
            b.weight * float(np.sum(_h((b.base + y) / b.potential, b.z) - _h(b.base / b.potential, b.z)))
            for b, y in zip(risks, ys))
        dy = [slopes(i, ys[i]) for i in range(len(risks))]
        cx = lin.copy()
        for i, b in enumerate(risks):
            free = np.where(lifted[i], 0.0, dy[i])
            if free.any():
                cx += b.gain.T @ free
        lift_val = [np.where(lifted[i], dy[i], 0.0) for i in range(len(risks))]
        v, exact = vertex(cx, lift_val)
        if exact:
            # the vertex's y: min(g v, P - A) on the lifted species (the linear program's best y there)
            gain_v = sum(float(lift_val[i] @ (np.minimum(b.gain @ v, caps[i]) - ys[i])) for i, b in enumerate(risks))
            bound = min(bound, max(f_lift + float(cx @ (v - x)) + gain_v, f))
        if bound - best <= tol * max(1.0, abs(bound)):
            status = "optimal"
            break
        if left() <= 0:
            status = "time_limit"
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
    if status != "optimal" and time.perf_counter() < deadline:
        status = "iteration_limit"
    if best_x is None:
        raise RuntimeError(f"no plan reaches the benefit within the tolerance ({status})")
    return best_x, proven, status, it


# ----------------------------------------------------------------------------- plangea's steps

def _plangea_slopes(b: _Risk, extra: np.ndarray) -> np.ndarray:
    """plangea's .multi_extinction_slope: |risk((A + delta) / P) - risk(A / P)| / delta, A the habitat
    without the plan plus what the plan so far changed."""
    a = b.base + extra
    r1 = np.maximum(1.0 - (a / b.potential) ** b.z, 0.0)
    r2 = np.maximum(1.0 - ((a + PLANGEA_DELTA) / b.potential) ** b.z, 0.0)
    return np.abs(r2 - r1) / PLANGEA_DELTA


def _steps_layer(p: RestorationProblem, b, cum: np.ndarray) -> np.ndarray:
    if isinstance(b, _Risk):
        slopes = _plangea_slopes(b, b.gain @ cum)
        # plangea's bd: the slopes times the habitat share a unit of the cell gives (or takes, signed)
        hab = b.hab.multiply(sp.csr_matrix(slopes[:, None])).tocsc()
        return np.asarray(hab.sum(axis=0)).ravel()
    if b.intactness is not None and cum.any():
        return _intactness_coef(p, b.intactness, cum)
    return b.layer


def _scale(layer: np.ndarray) -> float:
    """plangea's automatic scaling factor: 1 / 10^trunc(log10(max |layer|))."""
    mx = float(np.max(np.abs(layer))) if layer.size else 0.0
    if not mx > 0 or not math.isfinite(mx):
        return 1.0
    return 1.0 / 10.0 ** math.trunc(math.log10(mx))


def _solve_steps(p: RestorationProblem, steps: int, levels=(1.0,), scaled=None) -> list:
    """plangea's procedure: the targets met in ``steps`` parts, each a linear program on the weighted
    layers per cost, the layers recomputed from the plan so far between the parts (with more than one
    part). Over the ``levels`` of a trade-off curve (shares of the ``scaled`` target rows, plangea's
    gradient steps) the parts follow one another and the plan of every level is kept; the other rows
    hold for the plan so far: every part asks for its share of what is left of the row (plangea's
    ``scen_check_sr_targets``; its rows of "<=" are a cap per level there). Returns the plans."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if p.budget is not None or p.objective[0] != "max_benefit":
        raise ValueError("method='steps' takes area targets only (plangea's procedure): no budget, no minimum cost")
    if p.lower is not None and p.lower.any():
        raise ValueError("method='steps' takes no least shares")
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
    full = np.concatenate([t.rhs for t in p.targets])
    sense = tuple(s for t in p.targets for s in t.sense)
    scaled = np.ones(full.size, dtype=bool) if scaled is None else np.asarray(scaled, dtype=bool)
    out, prev = [], 0.0
    for li, level in enumerate(levels):
        inc, prev = float(level) - prev, float(level)
        for k in range(steps):
            done = rows @ cum
            rhs = np.where(scaled, full * inc / steps, np.maximum(np.minimum(full / steps, full - done), 0.0))
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
            if steps > 1 and not (li + 1 == len(levels) and k + 1 == steps):
                layers = [_steps_layer(p, b, cum) if (isinstance(b, _Risk) or b.intactness is not None) else v
                          for b, v in zip(p.benefits, layers)]
        out.append(np.minimum(cum, p.upper))
    return out


# ----------------------------------------------------------------------------- evaluation

@dataclass
class RiskReduction(Saveable):
    """Result of :func:`eval_risk_reduction`: per species, its habitat now (``current``), without the
    plan (``baseline``: less than now when conservation's baseline loses land) and gained by it, its
    risk before and after; the total reduction (negative when conversion raises the risk) and its
    standard deviation from the uncertainty of z."""

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
    after = b.base + gained
    after = np.where(np.abs(after) < EXRISK_EPS, 0.0, after)

    def reduction(z: np.ndarray) -> np.ndarray:
        before = np.maximum(1.0 - (b.base / b.potential) ** z, 0.0)
        aft = np.maximum(1.0 - (after / b.potential) ** z, 0.0)
        return np.nan_to_num(before - aft)

    red = reduction(b.z)
    table = pd.DataFrame({"species": list(b.species), "current": b.current, "baseline": b.base,
                          "potential": b.potential, "gained": gained,
                          "risk_before": np.maximum(1.0 - (b.base / b.potential) ** b.z, 0.0),
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
