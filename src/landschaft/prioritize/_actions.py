# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-action planning (prioriactions 0.5.0, Salgado-Rojas et al. 2020; GPL >= 2):
which planning units to monitor and which actions against which threats to
take in them so that the features recover (or are kept) to their targets at
the least cost (``minimizeCosts``), or recover the most within a budget
(``maximizeBenefits``).

The input checks and defaults are ``inputData()``'s, the model is
``problem()``'s (its constraint matrix, objective, bounds, senses and types
bit for bit, in prioriactions's order: :meth:`ActionsProblem.compile`), the
solution's figures ``solve()``'s, ``getCost()``'s and ``getSolutionBenefit()``'s.

Response curves other than linear (``curve`` 2 or 3: a feature's benefit is its
amount x b^curve) are, in prioriactions, Gurobi's general power constraints
approximated by Gurobi with pieces of length ``round(1 / segments, 1)`` through
points on the curve; here the same piecewise-linear function as an explicit
formulation (binary segments), on any solver.

Corrections (implemented only in their correct form, see
:func:`actions_data`): with no ``status`` in ``dist_threats``, prioriactions
resets the planning units' status instead of setting the actions' to 0; the
maximum conservation benefit counts the units instead of the feature's
amount in them; the lock-in budget check of ``maximizeBenefits`` indexes the
units by their ids instead of their positions.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._save import Saveable
from ._milp import OptimizationProblem
from ._milp_spatial import r_round

__all__ = ["ActionsData", "ActionsProblem", "ActionsSolution", "actions_data"]

MODELS = {"minimizeCosts": "minimizeCosts", "min_costs": "minimizeCosts", "maximizeBenefits": "maximizeBenefits",
          "max_benefits": "maximizeBenefits"}


def _warn(msg: str) -> None:
    warnings.warn(msg, UserWarning, stacklevel=3)


def _require(t: pd.DataFrame, what: str, cols) -> None:
    missing = [c for c in cols if c not in t.columns]
    if missing:
        raise ValueError(f"{what} needs the columns {missing}")
    for c in cols:
        if t[c].isna().any():
            raise ValueError(f"{what}: {c} has missing values")


def _numeric(t: pd.DataFrame, what: str, cols) -> None:
    for c in cols:
        if not pd.api.types.is_numeric_dtype(t[c]):
            raise ValueError(f"{what}: {c} must be numeric")


@dataclass(frozen=True)
class ActionsData:
    """Checked inputs of a multi-action problem (prioriactions's ``inputData()``): see :func:`actions_data`.

    The tables keep their columns plus ``internal_id`` (pu, features, threats: positions from 0) and
    ``internal_pu``, ``internal_feature``, ``internal_threat``, ``internal_id1``, ``internal_id2``."""

    pu: pd.DataFrame
    features: pd.DataFrame
    dist_features: pd.DataFrame
    threats: pd.DataFrame
    dist_threats: pd.DataFrame
    sensitivity: pd.DataFrame
    boundary: pd.DataFrame | None = None
    grid: object = field(default=None, repr=False, compare=False)  # (template raster, flat cell of every unit)

    @property
    def n_units(self) -> int:
        return len(self.pu)

    @property
    def n_actions(self) -> int:
        return len(self.dist_threats)

    def problem(self, model: str = "minimizeCosts", budget: float = 0.0, blm: float = 0.0, curve: int = 1,
                segments: int = 3) -> ActionsProblem:
        """The optimization problem (prioriactions's ``problem()``).

        Parameters
        ----------
        model
            ``"minimizeCosts"`` (``"min_costs"``): meet the features' recovery and conservation targets at
            the least cost of monitoring and actions (plus ``blm`` x the boundary); ``"maximizeBenefits"``
            (``"max_benefits"``): the most benefit within ``budget``.
        budget
            The most the monitoring and the actions may cost (``maximizeBenefits``).
        blm
            The weight of the selected units' boundary (needs boundary data); every threat's
            ``blm_actions`` weighs the boundary of its actions.
        curve
            The features' response to the actions: 1 linear, 2 quadratic, 3 cubic (benefit = b^curve,
            needs threat amounts of 0 or 1).
        segments
            Pieces of the curve's linear approximation (1 to 5): pieces of length ``round(1 / segments,
            1)``.
        """
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r} (minimizeCosts or maximizeBenefits)")
        model = MODELS[model]
        for name, v in (("blm", blm), ("curve", curve), ("segments", segments), ("budget", budget)):
            if not np.isscalar(v) or not math.isfinite(float(v)):
                raise ValueError(f"{name} must be a finite number")
        budget = float(r_round(np.array([float(budget)]), 3)[0])
        if abs(blm) <= 1e-10 and self.boundary is not None:
            _warn("The blm argument was set to 0, so the boundary data has no effect")
        if abs(blm) > 1e-50 and self.boundary is None:
            _warn("No boundary data supplied so the blm argument has no effect")
        if (self.threats["blm_actions"] <= 1e-10).all() and self.boundary is not None:
            _warn("Some blm_actions argument were set to 0, so the boundary data has no effect for these cases")
        if (self.threats["blm_actions"] > 1e-10).any() and self.boundary is None:
            _warn("No boundary data supplied so the blm_actions arguments has no effect")
        if curve not in (1, 2, 3):
            raise ValueError("invalid curve type (1, 2 or 3)")
        if curve == 1:
            segments = 1
        if curve != 1 and (self.dist_threats["amount"] < 1).any():
            curve = 1
            _warn("Curve set to 1 because there are non-binary values in threat amount for a recovery target")
        if segments not in (1, 2, 3, 4, 5):
            raise ValueError("invalid number of segments for linearization (1 to 5)")
        data = self
        if model == "minimizeCosts":
            data = _presolve_targets(self)
        else:
            budget = _presolve_budget(self, budget)
        return ActionsProblem(data=data, model=model, budget=budget, blm=float(blm), curve=int(curve),
                              segments=int(segments))

    def potential_benefit(self) -> pd.DataFrame:
        """Every feature's distribution (units, threatened units) and the most benefit the actions can give
        it: conservation (amount in the units without its threats), recovery and total (prioriactions's
        ``getPotentialBenefit()``, the conservation benefit as the amount)."""
        return _benefit(self, None)


def actions_data(pu: pd.DataFrame, features: pd.DataFrame, dist_features: pd.DataFrame, threats: pd.DataFrame,
                 dist_threats: pd.DataFrame, sensitivity: pd.DataFrame | None = None,
                 boundary: pd.DataFrame | None = None) -> ActionsData:
    """The inputs of a multi-action problem, checked and completed as prioriactions's ``inputData()``.

    Parameters
    ----------
    pu
        Planning units: ``id``, ``monitoring_cost`` and optionally ``status`` (0 free, 2 locked in, 3
        locked out).
    features
        ``id``, ``target_recovery``, optionally ``target_conservation`` (default 0) and ``name``.
    dist_features
        The features' amounts in the units: ``pu``, ``feature``, ``amount``.
    threats
        ``id``, optionally ``name`` and ``blm_actions`` (the weight of the boundary of the actions
        against it; default 0).
    dist_threats
        The threats in the units and the actions against them: ``pu``, ``threat``, ``amount`` (the
        threat's intensity), ``action_cost`` and optionally ``status`` (0, 2 the action locked in, 3
        locked out).
    sensitivity
        Which features each threat harms: ``feature``, ``threat`` and optionally the response's
        ``delta1`` to ``delta4`` (default: every feature to every threat, deltas 0, the threat's highest
        amount, 0 and 1).
    boundary
        ``id1``, ``id2``, ``boundary`` (shared lengths).

    Notes
    -----
    Rows with amount 0 are dropped; features without units, threats without units and threats that harm
    no feature are dropped with a warning; numbers are rounded to 3 decimals (R's ``round``). With no
    ``status`` in ``dist_threats`` the actions are free (prioriactions resets the units' status there
    instead), and the actions in locked-out units are locked out.
    """
    pu = pd.DataFrame(pu).copy()
    features = pd.DataFrame(features).copy()
    dist_features = pd.DataFrame(dist_features).copy()
    threats = pd.DataFrame(threats).copy()
    dist_threats = pd.DataFrame(dist_threats).copy()

    _require(pu, "pu", ("id", "monitoring_cost"))
    _numeric(pu, "pu", ("id", "monitoring_cost"))
    if not len(pu) or pu["id"].duplicated().any():
        raise ValueError("pu needs unique ids")
    if "status" in pu.columns:
        _numeric(pu, "pu", ("status",))
        if pu["status"].isna().any() or not pu["status"].isin([0, 2, 3]).all():
            raise ValueError("pu status must be 0, 2 or 3")
    else:
        pu["status"] = 0.0
    pu = pu.sort_values("id", kind="stable").reset_index(drop=True)

    _require(features, "features", ("id", "target_recovery"))
    _numeric(features, "features", ("id", "target_recovery"))
    if not len(features) or features["id"].duplicated().any():
        raise ValueError("features needs unique ids")
    if "name" in features.columns:
        names = features["name"].astype(str)
        if features["name"].isna().any() or names.duplicated().any():
            raise ValueError("feature names must be unique and not missing")
    else:
        features["name"] = [f"feature.{k}" for k in range(1, len(features) + 1)]
    if "target_conservation" in features.columns:
        _numeric(features, "features", ("target_conservation",))
        if features["target_conservation"].isna().any():
            raise ValueError("features: target_conservation has missing values")
    else:
        features["target_conservation"] = 0.0

    _require(dist_features, "dist_features", ("pu", "feature", "amount"))
    _numeric(dist_features, "dist_features", ("pu", "feature", "amount"))
    if not len(dist_features) or (dist_features["amount"] < 0).any():
        raise ValueError("dist_features needs rows with amounts >= 0")
    if not dist_features["pu"].isin(pu["id"]).all() or not dist_features["feature"].isin(features["id"]).all():
        raise ValueError("dist_features refers to planning units or features that do not exist")
    dist_features = dist_features[dist_features["amount"] != 0]
    if dist_features.duplicated(["pu", "feature"]).any():
        raise ValueError("There are duplicate values of the pu and feature pair in dist_features")

    _require(threats, "threats", ("id",))
    _numeric(threats, "threats", ("id",))
    if not len(threats) or threats["id"].duplicated().any():
        raise ValueError("threats needs unique ids")
    if "name" in threats.columns:
        names = threats["name"].astype(str)
        if threats["name"].isna().any() or names.duplicated().any():
            raise ValueError("threat names must be unique and not missing")
    else:
        threats["name"] = [f"threat.{k}" for k in range(1, len(threats) + 1)]
    if "blm_actions" in threats.columns:
        _numeric(threats, "threats", ("blm_actions",))
        if (threats["blm_actions"] < 0).any():
            raise ValueError("blm_actions must be >= 0")
    else:
        threats["blm_actions"] = 0.0

    _require(dist_threats, "dist_threats", ("pu", "threat", "action_cost", "amount"))
    _numeric(dist_threats, "dist_threats", ("pu", "threat", "action_cost", "amount"))
    if not len(dist_threats) or (dist_threats["amount"] < 0).any():
        raise ValueError("dist_threats needs rows with amounts >= 0")
    if not dist_threats["pu"].isin(pu["id"]).all():
        raise ValueError("dist_threats refers to planning units that do not exist")
    if not dist_threats["threat"].isin(threats["id"]).all():
        raise ValueError("dist_threats refers to threats that do not exist")
    if "status" in dist_threats.columns:
        _numeric(dist_threats, "dist_threats", ("status",))
        if dist_threats["status"].isna().any() or not dist_threats["status"].isin([0, 2, 3]).all():
            raise ValueError("dist_threats status must be 0, 2 or 3")
    else:
        dist_threats["status"] = 0.0
    out_ids = pu.loc[pu["status"] == 3, "id"]
    for uid in out_ids:  # the actions in locked-out units are locked out
        rows = dist_threats["pu"] == uid
        if (dist_threats.loc[rows, "status"] == 2).any():
            _warn(f"The pu {uid:g} was set as locked out so it is not possible to set actions on it (lock in). "
                  "Therefore, all actions in that unit will be set as locked out")
        dist_threats.loc[rows, "status"] = 3.0
    dist_threats = dist_threats[dist_threats["amount"] != 0]
    if dist_threats.duplicated(["pu", "threat"]).any():
        raise ValueError("There are duplicate values of the pu and threat pair in dist_threats")

    if sensitivity is not None:
        sensitivity = pd.DataFrame(sensitivity).copy()
        _require(sensitivity, "sensitivity", ("feature", "threat"))
        _numeric(sensitivity, "sensitivity", ("feature", "threat"))
        if not len(sensitivity):
            raise ValueError("sensitivity has no rows")
    else:  # expand.grid(feature, threat): features vary fastest
        f, t = np.meshgrid(features["id"].to_numpy(), threats["id"].to_numpy())
        sensitivity = pd.DataFrame({"feature": f.ravel(), "threat": t.ravel()})
    sensitivity["delta1"] = sensitivity["delta1"].fillna(0.0) if "delta1" in sensitivity else 0.0
    if "delta2" not in sensitivity:
        sensitivity["delta2"] = np.nan
    # delta2 by default: the threat's highest amount
    top = dist_threats.groupby("threat")["amount"].max()
    d2 = sensitivity["delta2"].to_numpy(dtype=np.float64, copy=True)
    fill = np.isnan(d2)
    d2[fill] = sensitivity["threat"].map(top).to_numpy(dtype=np.float64)[fill]
    sensitivity["delta2"] = d2
    sensitivity["delta3"] = sensitivity["delta3"].fillna(0.0) if "delta3" in sensitivity else 0.0
    sensitivity["delta4"] = sensitivity["delta4"].fillna(1.0) if "delta4" in sensitivity else 1.0
    ok = sensitivity["delta2"].isna() | (sensitivity["delta2"] > sensitivity["delta1"])
    if not ok.all():
        raise ValueError("Every value of delta1 parameter must be less than every value of delta2 parameter")
    if not (sensitivity["delta4"] > sensitivity["delta3"]).all():
        raise ValueError("Every value of delta3 parameter must be less than every value of delta4 parameter")

    if boundary is not None:
        boundary = pd.DataFrame(boundary).copy()
        _require(boundary, "boundary", ("id1", "id2", "boundary"))
        _numeric(boundary, "boundary", ("id1", "id2", "boundary"))
        boundary["boundary"] = r_round(boundary["boundary"].to_numpy(dtype=np.float64), 3)

    if (pu["monitoring_cost"] < 0).any():
        raise ValueError("argument to pu has negative monitoring cost data")
    empty_pu = np.setdiff1d(pu["id"].unique(), dist_features["pu"].unique())
    if empty_pu.size:
        _warn("The following pu's do not contain features: " + " ".join(f"{v:g}" for v in empty_pu))
    missing_f = np.setdiff1d(features["id"].unique(), dist_features["feature"].unique())
    if missing_f.size:
        _warn("The following features are not represented (it'll not be considered in the model): "
              + " ".join(f"{v:g}" for v in missing_f))
        features = features[~features["id"].isin(missing_f)]
        sensitivity = sensitivity[~sensitivity["feature"].isin(missing_f)]
    not_threatened = np.setdiff1d(features["id"].unique(), sensitivity["feature"].unique())
    if not_threatened.size:
        _warn("The following features are not threatened: " + " ".join(f"{v:g}" for v in not_threatened))
    missing_t = np.setdiff1d(threats["id"].unique(), dist_threats["threat"].unique())
    if missing_t.size:
        _warn("The following threats are not represented (it'll not be considered in the model): "
              + " ".join(f"{v:g}" for v in missing_t))
        threats = threats[~threats["id"].isin(missing_t)]
        sensitivity = sensitivity[~sensitivity["threat"].isin(missing_t)]
    harmless = np.setdiff1d(threats["id"].unique(), sensitivity["threat"].unique())
    if harmless.size:
        _warn("The following threats are not dangerous to any features (it'll not be considered in the model): "
              + " ".join(f"{v:g}" for v in harmless))
        threats = threats[~threats["id"].isin(harmless)]
        dist_threats = dist_threats[~dist_threats["threat"].isin(harmless)]
    if boundary is not None:
        keep = boundary["id1"].isin(pu["id"]) & boundary["id2"].isin(pu["id"])
        if not keep.all():
            _warn("The boundary data contain pu which ids does not exist in pu data (it'll not be considered in "
                  "the model)")
            boundary = boundary[keep]

    pu["monitoring_cost"] = r_round(pu["monitoring_cost"].to_numpy(dtype=np.float64), 3)
    features["target_recovery"] = r_round(features["target_recovery"].to_numpy(dtype=np.float64), 3)
    features["target_conservation"] = r_round(features["target_conservation"].to_numpy(dtype=np.float64), 3)
    dist_features["amount"] = r_round(dist_features["amount"].to_numpy(dtype=np.float64), 3)
    threats["blm_actions"] = r_round(threats["blm_actions"].to_numpy(dtype=np.float64), 3)
    dist_threats["amount"] = r_round(dist_threats["amount"].to_numpy(dtype=np.float64), 3)
    dist_threats["action_cost"] = r_round(dist_threats["action_cost"].to_numpy(dtype=np.float64), 3)
    for d in ("delta1", "delta2", "delta3", "delta4"):
        sensitivity[d] = r_round(sensitivity[d].to_numpy(dtype=np.float64), 3)

    pu = pu.reset_index(drop=True)
    features = features.reset_index(drop=True)
    threats = threats.reset_index(drop=True)
    dist_features = dist_features.reset_index(drop=True)
    dist_threats = dist_threats.reset_index(drop=True)
    sensitivity = sensitivity.reset_index(drop=True)
    pu["internal_id"] = np.arange(len(pu))
    features["internal_id"] = np.arange(len(features))
    threats["internal_id"] = np.arange(len(threats))
    pu_ix = pd.Series(pu["internal_id"].to_numpy(), index=pu["id"].to_numpy())
    f_ix = pd.Series(features["internal_id"].to_numpy(), index=features["id"].to_numpy())
    t_ix = pd.Series(threats["internal_id"].to_numpy(), index=threats["id"].to_numpy())
    dist_features["internal_pu"] = pu_ix.reindex(dist_features["pu"].to_numpy()).to_numpy()
    dist_features["internal_feature"] = f_ix.reindex(dist_features["feature"].to_numpy()).to_numpy()
    dist_threats["internal_pu"] = pu_ix.reindex(dist_threats["pu"].to_numpy()).to_numpy()
    dist_threats["internal_threat"] = t_ix.reindex(dist_threats["threat"].to_numpy()).to_numpy()
    sensitivity = sensitivity[sensitivity["feature"].isin(f_ix.index) & sensitivity["threat"].isin(t_ix.index)]
    sensitivity = sensitivity.reset_index(drop=True)
    sensitivity["internal_feature"] = f_ix.reindex(sensitivity["feature"].to_numpy()).to_numpy()
    sensitivity["internal_threat"] = t_ix.reindex(sensitivity["threat"].to_numpy()).to_numpy()
    if boundary is not None:
        boundary = boundary.reset_index(drop=True)
        boundary["internal_id1"] = pu_ix.reindex(boundary["id1"].to_numpy()).to_numpy()
        boundary["internal_id2"] = pu_ix.reindex(boundary["id2"].to_numpy()).to_numpy()
    return ActionsData(pu=pu, features=features, dist_features=dist_features, threats=threats,
                       dist_threats=dist_threats, sensitivity=sensitivity, boundary=boundary)


# ----------------------------------------------------------------------------- the model (C++: opt/actions.cpp)

def _arrays(d: ActionsData) -> dict:
    """The inputs as the C++ side takes them (internal ids)."""
    i64 = np.int64
    sens = d.sensitivity
    out = {
        "n_units": len(d.pu), "n_features": len(d.features), "n_threats": len(d.threats),
        "monitoring_cost": d.pu["monitoring_cost"].to_numpy(np.float64),
        "unit_status": d.pu["status"].to_numpy(np.float64),
        "target_recovery": d.features["target_recovery"].to_numpy(np.float64),
        "target_conservation": d.features["target_conservation"].to_numpy(np.float64),
        "df_unit": d.dist_features["internal_pu"].to_numpy(i64),
        "df_feature": d.dist_features["internal_feature"].to_numpy(i64),
        "df_amount": d.dist_features["amount"].to_numpy(np.float64),
        "blm_actions": d.threats["blm_actions"].to_numpy(np.float64),
        "act_unit": d.dist_threats["internal_pu"].to_numpy(i64),
        "act_threat": d.dist_threats["internal_threat"].to_numpy(i64),
        "act_amount": d.dist_threats["amount"].to_numpy(np.float64),
        "act_cost": d.dist_threats["action_cost"].to_numpy(np.float64),
        "act_status": d.dist_threats["status"].to_numpy(np.float64),
        "sens_feature": sens["internal_feature"].to_numpy(i64),
        "sens_threat": sens["internal_threat"].to_numpy(i64),
        "sens_delta": np.concatenate([sens[f"delta{k}"].to_numpy(np.float64) for k in range(1, 5)]),
        "has_boundary": d.boundary is not None,
    }
    if d.boundary is not None:
        out["bnd_id1"] = d.boundary["internal_id1"].to_numpy(i64)
        out["bnd_id2"] = d.boundary["internal_id2"].to_numpy(i64)
        out["bnd_value"] = d.boundary["boundary"].to_numpy(np.float64)
    return out


@dataclass(frozen=True)
class _Pairs:
    """Every (feature, unit) pair in prioriactions's order (features, then units ascending) and, for each,
    its threats that harm the feature (in the order of the threats) with the response's coefficients."""

    feature: np.ndarray  # per pair
    unit: np.ndarray
    amount: np.ndarray
    threatened: np.ndarray  # bool: a threat in the unit harms the feature (the recovery target's pair)
    sum_alpha: np.ndarray
    # per (pair, harmful threat), grouped by pair in the threats' order
    start: np.ndarray  # pairs + 1
    action: np.ndarray  # the action's position in dist_threats
    coef: np.ndarray  # the coefficient of the action in the pair's benefit row (as prioriactions computes it)
    contribution: np.ndarray  # (variable x alpha) / sum_alpha: the benefit per unit amount of taking the action
    action_status: np.ndarray


def _pairs(d: ActionsData) -> _Pairs:
    from .. import _core

    r = dict(_core.opt_actions_pairs(_arrays(d)))
    r["threatened"] = r["threatened"].astype(bool)
    return _Pairs(**r)


def _benefit(d: ActionsData, solution: np.ndarray | None) -> pd.DataFrame:
    """rcpp_stats_benefit: the most benefit (solution None) or a solution's (units then actions)."""
    from .. import _core

    b = _core.opt_actions_benefit(_arrays(d), None if solution is None else np.asarray(solution, np.float64))
    ids = d.features["id"].to_numpy()
    cons, rec = b["conservation"], b["recovery"]
    if solution is not None:
        return pd.DataFrame({"feature": ids, "benefit.conservation": r_round(cons, 4),
                             "benefit.recovery": r_round(rec, 4), "benefit.total": r_round(rec + cons, 4)})
    return pd.DataFrame({"feature": ids, "dist": b["dist"], "dist_threatened": b["dist_threatened"],
                         "maximum.conservation.benefit": r_round(cons, 4),
                         "maximum.recovery.benefit": r_round(rec, 4),
                         "maximum.benefit": r_round(rec + cons, 4)})


def _presolve_targets(d: ActionsData) -> ActionsData:
    a = _benefit(d, None)
    f = d.features.copy()
    for col, top, what in (("target_recovery", "maximum.recovery.benefit", "recovery"),
                           ("target_conservation", "maximum.conservation.benefit", "conservation")):
        over = f[col].to_numpy() > a[top].to_numpy()
        if over.any():
            _warn(f"Infeasible model. There is not enough representativeness to achieve the targets of {what} "
                  "required of following features: " + " ".join(f"{v:g}" for v in a["feature"][over])
                  + f". The targets of {what} for these features will be set to the maximum {what} benefit values")
            v = f[col].to_numpy(dtype=np.float64, copy=True)
            v[over] = a[top].to_numpy()[over] - 10 ** (-1)
            f[col] = v
    return replace(d, features=f)


def _costs(d: ActionsData, units: np.ndarray, actions: np.ndarray) -> tuple[float, np.ndarray]:
    """rcpp_stats_costs_units / _actions: the monitoring cost and every threat's actions' cost."""
    from .. import _core

    c = _core.opt_actions_costs(_arrays(d), np.asarray(units, np.float64), np.asarray(actions, np.float64))
    return float(c[0]), c[1:]


def _presolve_budget(d: ActionsData, budget: float) -> float:
    units = (d.pu["status"].to_numpy() == 2).astype(np.float64)
    st = d.dist_threats["status"].to_numpy()
    actions = (st == 2).astype(np.float64)
    units[d.dist_threats["internal_pu"].to_numpy(dtype=np.int64)[st == 2]] = 1.0
    cu, ca = _costs(d, units, actions)
    need = float(np.sum(ca)) + cu
    if budget < need:
        _warn("Infeasible model. There is not enough budget to achieve the actions required (lock-in). The budget "
              f"will be set as the sum of the minimum costs to achieve the required actions: {need:g}")
        return need + 10 ** (-1)
    return budget


_SENSES = np.array(["<=", ">=", "="], dtype=object)


def _settings(p: ActionsProblem) -> dict:
    return {"maximize": p.model == "maximizeBenefits", "blm": float(p.blm), "curve": int(p.curve),
            "segments": int(p.segments), "step": float(r_round(np.array([1 / p.segments]), 1)[0]),
            "budget": float(p.budget)}


def _optimization_problem(c: dict, maximize: bool) -> OptimizationProblem:
    nrow, ncol = c["rhs"].size, c["obj"].size
    a = sp.csr_matrix((c["value"], c["index"], c["start"]), shape=(nrow, ncol))
    return OptimizationProblem(
        A=a, obj=c["obj"], lb=c["lb"], ub=c["ub"], rhs=c["rhs"], sense=tuple(_SENSES[c["sense"]]),
        vtype=tuple(np.where(c["integer"] != 0, "B", "C").astype(object)), modelsense="max" if maximize else "min",
        row_ids=("",) * nrow, col_ids=("",) * ncol)


def compile_actions(p: ActionsProblem, disaggregate: bool = False, curves: bool = False) -> OptimizationProblem:
    """prioriactions's model; with disaggregate, the activation rows one per action (x_ik <= w_i: the same
    integer solutions, a much tighter linear relaxation) instead of one per unit; with curves, the response
    curve (curve 2, 3) as the piecewise-linear function Gurobi uses in prioriactions (FuncPieces 1,
    FuncPieceLength round(1 / segments, 1), points on the curve), in an incremental formulation with a
    binary per piece."""
    from .. import _core

    s = _settings(p)
    c = _core.opt_actions_compile(_arrays(p.data), s, disaggregate, curves)
    return _optimization_problem(c, s["maximize"])


# ----------------------------------------------------------------------------- problem and solution

@dataclass(frozen=True)
class ActionsProblem:
    """A multi-action problem (prioriactions's ``problem()``): see :meth:`ActionsData.problem`."""

    data: ActionsData = field(repr=False)
    model: str = "minimizeCosts"
    budget: float = 0.0
    blm: float = 0.0
    curve: int = 1
    segments: int = 3
    solver: tuple = ("highs", {"gap": 0.0})

    def compile(self) -> OptimizationProblem:
        """prioriactions's model (``getModelInfo()``'s matrices), bit for bit; with ``curve`` 2 or 3 the
        power constraints are not in it (Gurobi's general constraints in prioriactions)."""
        return compile_actions(self)

    def add_highs_solver(self, gap: float = 0.0, time_limit: float | None = None, n_threads: int | None = None,
                         verbose: bool = False, engine: bool = True) -> ActionsProblem:
        """Solve on HiGHS to a relative ``gap`` (prioriactions's ``gap_limit``, default 0), behind
        landschaft's closure engine (``engine=False``: HiGHS's MIP alone; see :meth:`solve`)."""
        return replace(self, solver=("highs", {"gap": gap, "time_limit": time_limit, "n_threads": n_threads,
                                               "verbose": verbose, "engine": engine}))

    def add_gurobi_solver(self, gap: float = 0.0, time_limit: float | None = None, n_threads: int | None = None,
                          verbose: bool = False) -> ActionsProblem:
        """Solve on Gurobi (needs ``gurobipy`` and a licence)."""
        return replace(self, solver=("gurobi", {"gap": gap, "time_limit": time_limit, "n_threads": n_threads,
                                                "verbose": verbose}))

    def solve(self, start=None) -> ActionsSolution:
        """Solve the problem.

        On HiGHS (the default), the equivalent model (activation one row per action, the benefit
        columns substituted) first goes to landschaft's closure engine (a Lagrangian bound as strong as
        the linear relaxation, by a closed form or minimum cuts, and repaired selections), then, while
        the gap is wider than asked, to HiGHS's MIP from its solution with the units and actions the
        bound proves fixed. ``start``: a solution to start from (an :class:`ActionsSolution` or the
        units then the actions, 0/1)."""
        from .. import _core, _threads
        from ._milp_solve import solve_model

        d = self.data
        n, na = len(d.pu), len(d.dist_threats)
        arrays = _arrays(d)
        s = _settings(self)
        solver, opts = self.solver
        init = None
        if start is not None:
            init = (np.concatenate([start.units, start.actions]) if isinstance(start, ActionsSolution)
                    else np.asarray(start, dtype=np.float64).ravel())
            if init.size != n + na:
                raise ValueError(f"start needs {n + na} values (the units, then the actions)")
        if solver == "highs":
            tl = opts.get("time_limit")
            r = dict(_core.opt_actions_solve(
                arrays, s, float(opts.get("gap", 0.0)), math.inf if tl is None else float(tl),
                _threads.resolve(opts.get("n_threads")), bool(opts.get("verbose", False)),
                bool(opts.get("engine", True)), init))
            r["solver"] = "highs"
        else:
            c = _core.opt_actions_equivalent(arrays, s) if self.curve == 1 else None
            if c is not None:  # benefit columns substituted, activation per action, locks as bounds
                r = solve_model(_optimization_problem(c, s["maximize"]), solver, opts)
                if r["has_solution"]:
                    wx = np.round(np.asarray(r["x"], dtype=np.float64)[:n + na])
                    r["x"] = np.concatenate([wx, _core.opt_actions_local_benefit(arrays, wx)])
            else:
                r = solve_model(compile_actions(self, disaggregate=True, curves=True), solver, opts)
        if not r["has_solution"]:
            raise RuntimeError(f"no solution ({r['solver']}: {r['status']})")
        return self._solution(np.asarray(r["x"], dtype=np.float64), r)

    def _solution(self, x: np.ndarray, r: dict | None = None) -> ActionsSolution:
        """The solution of a full vector of the model's columns (units, actions, benefits, ...)."""
        r = r or {}
        d = self.data
        n, na = len(d.pu), len(d.dist_threats)
        pairs = _pairs(d)
        local = x[n + na:n + na + pairs.feature.size]
        # prioriactions's local table: the dist_features rows by unit (stable), the benefits by (feature, unit)
        table = pd.DataFrame({"pu": d.pu["id"].to_numpy()[pairs.unit],
                              "feature": d.features["id"].to_numpy()[pairs.feature],
                              "benefit.conservation": np.where(pairs.threatened, 0.0, local),
                              "benefit.recovery": np.where(pairs.threatened, local, 0.0)})
        table["benefit.total"] = table["benefit.recovery"] + table["benefit.conservation"]
        table = table.sort_values("pu", kind="stable").reset_index(drop=True)
        return ActionsSolution(
            data=d, units=np.round(x[:n]), actions=np.round(x[n:n + na]), local_benefit=table,
            objective=float(r.get("objective", math.nan)), bound=float(r.get("bound", math.nan)),
            gap=float(r.get("gap", math.nan)), status=r.get("status", ""), solver=r.get("solver", ""),
            seconds=float(r.get("seconds", 0.0)), blm=self.blm, x=x)


@dataclass
class ActionsSolution(Saveable):
    """A solution of an :class:`ActionsProblem`.

    Attributes
    ----------
    units, actions
        The monitored units (0/1, in the order of ``data.pu``) and the actions taken (0/1, in the order of
        ``data.dist_threats``): prioriactions's ``sol_monitoring`` and ``sol_actions``.
    local_benefit
        Every (unit, feature)'s benefit, by conservation or recovery (``getSolutionBenefit(type =
        "local")``).
    objective, bound, gap, status, solver, seconds
        As :class:`~landschaft.prioritize.Solution`'s.
    """

    data: ActionsData = field(repr=False)
    units: np.ndarray = field(repr=False)
    actions: np.ndarray = field(repr=False)
    local_benefit: pd.DataFrame = field(repr=False)
    objective: float = math.nan
    bound: float = math.nan
    gap: float = math.nan
    status: str = ""
    solver: str = ""
    seconds: float = 0.0
    blm: float = 0.0
    x: np.ndarray = field(default=None, repr=False)

    def maps(self):
        """The monitored cells and every threat's actions as rasters (band ``monitoring``, then the threats'
        names; NaN outside the planning units), for problems from :func:`actions_raster_data`."""
        from ._actions_raster import solution_maps

        return solution_maps(self.data, self.units, self.actions)

    def monitoring(self) -> pd.DataFrame:
        """``id`` and ``solution`` of every planning unit."""
        return pd.DataFrame({"id": self.data.pu["id"].to_numpy(), "solution": self.units})

    def action_table(self) -> pd.DataFrame:
        """``pu``, ``threat`` and ``solution`` of every action."""
        dt = self.data.dist_threats
        return pd.DataFrame({"pu": dt["pu"].to_numpy(), "threat": dt["threat"].to_numpy(), "solution": self.actions})

    def cost(self) -> pd.DataFrame:
        """The monitoring cost and every threat's actions' cost (``getCost()``)."""
        cu, ca = _costs(self.data, self.units, self.actions)
        out = {"monitoring": [float(r_round(np.array([cu]), 3)[0])]}
        for tid, v in zip(self.data.threats["id"].to_numpy(), r_round(ca, 3)):
            out[f"threat_{tid:g}"] = [float(v)]
        return pd.DataFrame(out)

    def connectivity_penalty(self) -> pd.DataFrame | None:
        """The boundary of the monitored units and of every threat's actions (``getConnectivityPenalty()``;
        None without boundary data)."""
        from .. import _core

        d = self.data
        if d.boundary is None or not len(d.boundary):
            return None
        c = _core.opt_actions_connectivity(_arrays(d), np.asarray(self.units, np.float64),
                                           np.asarray(self.actions, np.float64))
        out = {"units": [float(r_round(np.array([c[0]]), 3)[0])]}
        for tid, v in zip(d.threats["id"].to_numpy(), r_round(c[1:], 3)):
            out[f"threat_{tid:g}"] = [float(v)]
        return pd.DataFrame(out)

    def action_summary(self, format: str = "wide") -> pd.DataFrame:
        """Every unit's actions (by threat id), whether it conserves a feature without actions
        (``conservation``) and whether it is monitored only for connectivity (``connectivity``):
        prioriactions's ``getActions()``. ``format="long"``: one row per (``pu``, ``action``)."""
        if format not in ("wide", "long", "large"):
            raise ValueError("format must be 'wide' or 'long'")
        d = self.data
        ids = d.pu["id"].to_numpy()
        lb = self.local_benefit
        cons_units = lb.loc[lb["benefit.conservation"] > 0, "pu"].unique()
        conservation = np.isin(ids, cons_units).astype(np.float64)
        dt = d.dist_threats
        acted = np.isin(ids, dt["pu"].to_numpy()[self.actions > 0]).astype(np.float64)
        both = ((conservation + acted) >= 1).astype(np.float64)
        connectivity = self.units - both
        long = pd.concat([
            pd.DataFrame({"pu": dt["pu"].to_numpy(), "action": [f"{t:g}" for t in dt["threat"].to_numpy()],
                          "solution": self.actions}),
            pd.DataFrame({"pu": ids, "action": "conservation", "solution": conservation}),
            pd.DataFrame({"pu": ids, "action": "connectivity", "solution": connectivity})], ignore_index=True)
        if format != "wide":
            return long
        keys = sorted(long["action"].unique())
        wide = long.pivot(index="pu", columns="action", values="solution").reindex(columns=keys).fillna(0.0)
        wide = wide.sort_index().reset_index()
        wide.columns.name = None
        cols = list(wide.columns)
        wide = wide[cols[:-2] + [cols[-1], cols[-2]]]
        num = wide.columns[1:]
        wide[num] = r_round(wide[num].to_numpy(dtype=np.float64), 1)
        return wide

    def benefit(self) -> pd.DataFrame:
        """Every feature's benefit, by conservation and recovery (``getSolutionBenefit()``; the
        conservation benefit as the amount in the monitored units)."""
        return _benefit(self.data, np.concatenate([self.units, self.actions]))

    def _save_items(self) -> dict:
        maps = {"maps": self.maps()} if self.data.grid is not None else {}
        return {**maps, "monitoring": self.monitoring(), "actions": self.action_table(), "local_benefit": self.local_benefit,
                "cost": self.cost(), "benefit": self.benefit(), "objective": self.objective, "status": self.status}
