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


# ----------------------------------------------------------------------------- the response to the actions

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
    nf, n = len(d.features), len(d.pu)
    df = d.dist_features
    order = np.lexsort((df["internal_pu"].to_numpy(), df["internal_feature"].to_numpy()))
    feature = df["internal_feature"].to_numpy(dtype=np.int64)[order]
    unit = df["internal_pu"].to_numpy(dtype=np.int64)[order]
    amount = df["amount"].to_numpy(dtype=np.float64)[order]
    dt = d.dist_threats
    nt = len(d.threats)
    # the threats of every unit, in the threats' order
    t_unit = dt["internal_pu"].to_numpy(dtype=np.int64)
    t_threat = dt["internal_threat"].to_numpy(dtype=np.int64)
    by_unit = sp.csr_matrix((np.arange(1, len(dt) + 1, dtype=np.float64), (t_unit, t_threat)), shape=(n, nt))
    by_unit.sort_indices()
    sens = d.sensitivity
    sf = sens["internal_feature"].to_numpy(dtype=np.int64)
    st = sens["internal_threat"].to_numpy(dtype=np.int64)
    params = np.zeros((4, nf, nt))
    harms = np.zeros((nf, nt), dtype=bool)
    for k, name in enumerate(("delta1", "delta2", "delta3", "delta4")):
        params[k, sf, st] = sens[name].to_numpy(dtype=np.float64)  # later rows win, as arma's assignment
    harms[sf, st] = True
    intensity = dt["amount"].to_numpy(dtype=np.float64)
    status = dt["status"].to_numpy(dtype=np.float64)
    starts = [0]
    actions, coefs, contribs, astatus = [], [], [], []
    threatened = np.zeros(feature.size, dtype=bool)
    sum_alpha = np.zeros(feature.size)
    indptr, indices, data = by_unit.indptr, by_unit.indices, by_unit.data
    for k in range(feature.size):
        s, i = feature[k], unit[k]
        lo, hi = indptr[i], indptr[i + 1]
        ts = indices[lo:hi]
        acts = data[lo:hi].astype(np.int64) - 1
        alphas, rows = [], []
        total = 0.0
        for t, a in zip(ts, acts):
            if not harms[s, t]:
                continue
            threatened[k] = True
            x = intensity[a]
            pa, pb, pc, pd_ = params[0, s, t], params[1, s, t], params[2, s, t], params[3, s, t]
            if x <= pa:
                var, const = 0.0, pd_
            elif x >= pb:
                var, const = pd_ - pc, pc
            else:
                var = ((pa - x) * (pc - pd_)) / (pb - pa)
                const = (pc * (x - pa) - pd_ * (x - pb)) / (pb - pa)
            alpha = 1 - const
            total = total + alpha
            alphas.append(alpha)
            rows.append((a, var))
        sum_alpha[k] = total
        if total != 0.0:
            for (a, var), alpha in zip(rows, alphas):
                actions.append(a)
                coefs.append(-1 * (var * alpha) / total)
                contribs.append((var * alpha) / total)
                astatus.append(status[a])
        starts.append(len(actions))
    return _Pairs(feature=feature, unit=unit, amount=amount, threatened=threatened, sum_alpha=sum_alpha,
                  start=np.asarray(starts, dtype=np.int64), action=np.asarray(actions, dtype=np.int64),
                  coef=np.asarray(coefs, dtype=np.float64), contribution=np.asarray(contribs, dtype=np.float64),
                  action_status=np.asarray(astatus, dtype=np.float64))


def _benefit(d: ActionsData, solution: np.ndarray | None) -> pd.DataFrame:
    """rcpp_stats_benefit: the most benefit (solution None) or a solution's (units then actions)."""
    p = _pairs(d)
    nf = len(d.features)
    unit_status = d.pu["status"].to_numpy()
    dist = np.bincount(p.feature, minlength=nf).astype(np.float64)
    dist_threatened = np.zeros(nf)
    max_cons, max_rec = np.zeros(nf), np.zeros(nf)
    sol_cons, sol_rec = np.zeros(nf), np.zeros(nf)
    n = len(d.pu)
    for k in range(p.feature.size):
        s, i = p.feature[k], p.unit[k]
        if unit_status[i] == 3:
            continue
        if p.threatened[k]:
            dist_threatened[s] += 1
        if p.sum_alpha[k] != 0.0:
            for j in range(p.start[k], p.start[k + 1]):
                if p.action_status[j] == 3:
                    continue
                v = p.contribution[j] * p.amount[k]
                max_rec[s] = max_rec[s] + v
                if solution is not None:
                    sol_rec[s] = sol_rec[s] + solution[n + p.action[j]] * v
        else:
            max_cons[s] = max_cons[s] + p.amount[k]
            if solution is not None:
                sol_cons[s] = sol_cons[s] + solution[i] * p.amount[k]
    ids = d.features["id"].to_numpy()
    if solution is not None:
        return pd.DataFrame({"feature": ids, "benefit.conservation": r_round(sol_cons, 4),
                             "benefit.recovery": r_round(sol_rec, 4), "benefit.total": r_round(sol_rec + sol_cons, 4)})
    return pd.DataFrame({"feature": ids, "dist": dist, "dist_threatened": dist_threatened,
                         "maximum.conservation.benefit": r_round(max_cons, 4),
                         "maximum.recovery.benefit": r_round(max_rec, 4),
                         "maximum.benefit": r_round(max_rec + max_cons, 4)})


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
    c = d.pu["monitoring_cost"].to_numpy(dtype=np.float64)
    cu = 0.0
    for i in range(c.size):
        cu = cu + c[i] * units[i]
    dt = d.dist_threats
    per_action = dt["action_cost"].to_numpy(dtype=np.float64) * actions
    nt = len(d.threats)
    by_threat = np.zeros(nt)
    t = dt["internal_threat"].to_numpy(dtype=np.int64)
    u = dt["internal_pu"].to_numpy(dtype=np.int64)
    for a in np.lexsort((u, t)):  # every threat's column, units ascending
        if per_action[a] != 0:
            by_threat[t[a]] = by_threat[t[a]] + per_action[a]
    return cu, by_threat


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


# ----------------------------------------------------------------------------- the model

class _Builder:
    def __init__(self) -> None:
        self.i: list = []
        self.j: list = []
        self.x: list = []
        self.rhs: list = []
        self.sense: list = []
        self.obj: list = []
        self.lb: list = []
        self.ub: list = []
        self.vtype: list = []

    def entries(self, i, j, x) -> None:
        self.i.append(np.asarray(i, dtype=np.int64).ravel())
        self.j.append(np.asarray(j, dtype=np.int64).ravel())
        self.x.append(np.asarray(x, dtype=np.float64).ravel())

    def rows(self, rhs, sense) -> None:
        rhs = np.asarray(rhs, dtype=np.float64).ravel()
        self.rhs.append(rhs)
        self.sense.append(np.full(rhs.size, sense, dtype=object) if isinstance(sense, str) else
                          np.asarray(sense, dtype=object))

    def columns(self, obj, vtype: str, lb=0.0, ub=1.0) -> None:
        obj = np.asarray(obj, dtype=np.float64).ravel()
        self.obj.append(obj)
        self.lb.append(np.full(obj.size, lb))
        self.ub.append(np.full(obj.size, ub))
        self.vtype.append(np.full(obj.size, vtype, dtype=object))

    @property
    def nrow(self) -> int:
        return int(sum(r.size for r in self.rhs))

    @property
    def ncol(self) -> int:
        return int(sum(o.size for o in self.obj))


def _boundary_matrix(d: ActionsData) -> sp.csc_matrix | None:
    """create_boundary_matrix_extended: both directions, unless the pair is already there (then only id1,
    id2 changes)."""
    if d.boundary is None or not len(d.boundary):
        return None
    n = len(d.pu)
    vals: dict = {}
    for a, b, v in zip(d.boundary["internal_id1"].to_numpy(dtype=np.int64),
                       d.boundary["internal_id2"].to_numpy(dtype=np.int64),
                       d.boundary["boundary"].to_numpy(dtype=np.float64)):
        if vals.get((a, b), 0.0) == 0.0:
            vals[(a, b)] = v
            vals[(b, a)] = v
        else:
            vals[(a, b)] = v
    keys = [(k, v) for k, v in vals.items() if v != 0.0]  # arma keeps no zeros
    if not keys:
        return sp.csc_matrix((n, n))
    r = np.array([k[0] for k, _ in keys])
    c = np.array([k[1] for k, _ in keys])
    v = np.array([v for _, v in keys])
    m = sp.csc_matrix((v, (r, c)), shape=(n, n))
    m.sort_indices()
    return m


@dataclass(frozen=True)
class ActionsModel:
    """A compiled multi-action model and what the solution needs to read it."""

    model: OptimizationProblem
    n_units: int
    n_actions: int
    n_pairs: int
    # curve != 1: y = x^a for the benefit columns xvar -> yvar (Gurobi's genconpow in prioriactions)
    xvar: np.ndarray
    yvar: np.ndarray
    curve: int
    segments: int


def compile_actions(p: ActionsProblem, disaggregate: bool = False) -> ActionsModel:
    """The model; with disaggregate, the activation rows one per action (x_ik <= w_i: the same integer
    solutions, a much tighter linear relaxation) instead of prioriactions's one per unit."""
    d = p.data
    n, na = len(d.pu), len(d.dist_threats)
    nf, nt = len(d.features), len(d.threats)
    pairs = _pairs(d)
    npairs = pairs.feature.size
    b = _Builder()
    # --- benefit rows: b_is - sum coef x_ik <= 0, or b_is - w_i <= 0 (no harmful threat with a response)
    row0 = 0
    for k in range(npairs):
        r = row0 + k
        col = n + na + k
        ii, jj, xx = [r], [col], [1.0]
        if pairs.sum_alpha[k] != 0.0:
            lo, hi = pairs.start[k], pairs.start[k + 1]
            ii += [r] * (hi - lo)
            jj += list(n + pairs.action[lo:hi])
            xx += list(pairs.coef[lo:hi])
        else:
            ii.append(r)
            jj.append(pairs.unit[k])
            xx.append(-1.0)
        b.entries(ii, jj, xx)
    b.rows(np.zeros(npairs), "<=")
    # --- activation rows: sum_k x_ik - |K_i| w_i <= 0, one per unit
    dt = d.dist_threats
    t_unit = dt["internal_pu"].to_numpy(dtype=np.int64)
    t_threat = dt["internal_threat"].to_numpy(dtype=np.int64)
    row0 = b.nrow
    order = np.lexsort((t_threat, t_unit))
    counts = np.bincount(t_unit, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)])
    if disaggregate:
        b.entries(np.repeat(row0 + np.arange(na), 2), np.column_stack([n + np.arange(na), t_unit]).ravel(),
                  np.tile([1.0, -1.0], na))
        b.rows(np.zeros(na), "<=")
    else:
        for i in range(n):
            acts = order[starts[i]:starts[i + 1]]
            b.entries([row0 + i] * (acts.size + 1), list(n + acts) + [i], [1.0] * acts.size + [-1.0 * acts.size])
        b.rows(np.zeros(n), "<=")
    # --- locks: units, then actions
    us = d.pu["status"].to_numpy()
    locked = np.flatnonzero(us != 0)
    row0 = b.nrow
    b.entries(row0 + np.arange(locked.size), locked, np.ones(locked.size))
    b.rows(np.where(us[locked] == 2, 1.0, 0.0), "=")
    st = dt["status"].to_numpy()
    locked = np.flatnonzero(st != 0)
    row0 = b.nrow
    b.entries(row0 + np.arange(locked.size), n + locked, np.ones(locked.size))
    b.rows(np.where(st[locked] == 2, 1.0, 0.0), "=")
    # --- objective
    maximize = p.model == "maximizeBenefits"
    bm = _boundary_matrix(d)
    blm = p.blm
    conn_units = np.zeros(n)
    if bm is not None and blm != 0:
        z = bm.T.tocsc()  # z.begin(): by column, rows ascending
        z.sort_indices()
        for c in range(n):
            for k in range(z.indptr[c], z.indptr[c + 1]):
                if z.indices[k] != c:
                    conn_units[c] = conn_units[c] + z.data[k]
    unit_cost = d.pu["monitoring_cost"].to_numpy(dtype=np.float64)
    b.columns(-blm * conn_units if maximize else blm * conn_units + unit_cost, "B")
    blm_actions = d.threats["blm_actions"].to_numpy(dtype=np.float64)
    action_cost = dt["action_cost"].to_numpy(dtype=np.float64)
    conn_actions = np.zeros(na)
    # every threat's units ascending (arma's column of dist_threats_extended)
    by_threat = [np.sort(t_unit[t_threat == t]) for t in range(nt)]
    act_of = {(int(u), int(t)): a for a, (u, t) in enumerate(zip(t_unit, t_threat))}
    pair_terms = []  # (action a, action of the other unit, value)
    for a in range(na):
        t = t_threat[a]
        if bm is not None and blm_actions[t] != 0:
            u1 = t_unit[a]
            for u2 in by_threat[t]:
                v = bm[u1, u2]
                if u1 != u2 and v != 0:
                    conn_actions[a] = conn_actions[a] + v
                    pair_terms.append((a, act_of[(int(u2), int(t))], v))
    bt = blm_actions[t_threat]
    b.columns(-bt * conn_actions if maximize else bt * conn_actions + action_cost, "B")
    curve = p.curve
    if maximize and curve == 1:
        b.columns(pairs.amount, "C")
    else:
        b.columns(np.zeros(npairs), "C")
    first_pow = b.ncol
    if curve != 1:
        if maximize:
            row = b.nrow  # prioriactions puts explicit zeros here (the next row's index)
            b.entries(np.full(npairs, row), first_pow + np.arange(npairs), np.zeros(npairs))
            b.columns(pairs.amount, "C")
        else:
            b.columns(np.zeros(npairs), "C")
    # y[i1, i2]: every directed pair of the boundary matrix (z = M', by column, rows ascending)
    if bm is not None and blm != 0:
        z = bm.T.tocsc()
        z.sort_indices()
        col = b.ncol
        row = b.nrow
        objs, ii, jj, xx, rhs, sense = [], [], [], [], [], []
        for c in range(n):
            for k in range(z.indptr[c], z.indptr[c + 1]):
                r = z.indices[k]
                if r == c:
                    continue
                v = z.data[k]
                objs.append(blm * (1 * v) if maximize else blm * (-1 * v))
                ii += [row, row, row + 1, row + 1, row + 2, row + 2, row + 2]
                jj += [col, r, col, c, col, r, c]
                xx += [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0]
                rhs += [0.0, 0.0, -1.0]
                sense += ["<=", "<=", ">="]
                row += 3
                col += 1
        if objs:
            b.columns(objs, "B")
            b.entries(ii, jj, xx)
            b.rows(rhs, sense)
    # p[i1, i2, k]: the boundary between two units' actions against the same threat
    if pair_terms:
        col = b.ncol
        row = b.nrow
        objs, ii, jj, xx, rhs, sense = [], [], [], [], [], []
        for a1, a2, v in pair_terms:
            objs.append(blm_actions[t_threat[a1]] * (v if maximize else -1 * v))
            ii += [row, row, row + 1, row + 1, row + 2, row + 2, row + 2]
            jj += [col, n + a1, col, n + a2, col, n + a1, n + a2]
            xx += [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0]
            rhs += [0.0, 0.0, -1.0]
            sense += ["<=", "<=", ">="]
            row += 3
            col += 1
        b.columns(objs, "B")
        b.entries(ii, jj, xx)
        b.rows(rhs, sense)
    # --- targets (two rows per feature: conservation, recovery) or the budget
    if not maximize:
        row0 = b.nrow
        tcol = n + na + (0 if curve == 1 else npairs)
        tr = d.features["target_recovery"].to_numpy(dtype=np.float64)
        tc = d.features["target_conservation"].to_numpy(dtype=np.float64)
        rhs = []
        for s in range(nf):
            ks = np.flatnonzero(pairs.feature == s)
            rec = pairs.threatened[ks]
            r = row0 + 2 * s
            b.entries(np.where(rec, r + 1, r), tcol + ks, pairs.amount[ks])
            if not (~rec).any():
                b.entries([r], [1], [0.0])
            rhs.append(tc[s] if (~rec).any() else 0.0)
            if not rec.any():
                b.entries([r + 1], [1], [0.0])
            rhs.append(tr[s] if rec.any() else 0.0)
        b.rows(rhs, ">=")
    else:
        row = b.nrow
        b.entries(np.full(n + na, row), np.arange(n + na), np.concatenate([unit_cost, action_cost]))
        b.rows([p.budget], "<=")
    nrow, ncol = b.nrow, b.ncol
    i = np.concatenate(b.i) if b.i else np.zeros(0, np.int64)
    j = np.concatenate(b.j) if b.j else np.zeros(0, np.int64)
    x = np.concatenate(b.x) if b.x else np.zeros(0)
    a = sp.csr_matrix((x, (i, j)), shape=(nrow, ncol))  # duplicates summed (Matrix::sparseMatrix)
    a.sort_indices()
    m = OptimizationProblem(
        A=a, obj=np.concatenate(b.obj), lb=np.concatenate(b.lb), ub=np.concatenate(b.ub),
        rhs=np.concatenate(b.rhs), sense=tuple(np.concatenate(b.sense)), vtype=tuple(np.concatenate(b.vtype)),
        modelsense="max" if maximize else "min", row_ids=("",) * nrow, col_ids=("",) * ncol)
    xvar = n + na + np.arange(npairs) if curve != 1 else np.zeros(0, np.int64)
    yvar = first_pow + np.arange(npairs) if curve != 1 else np.zeros(0, np.int64)
    return ActionsModel(model=m, n_units=n, n_actions=na, n_pairs=npairs, xvar=xvar, yvar=yvar, curve=curve,
                        segments=p.segments)


def _eliminate_benefits(am: ActionsModel, pairs: _Pairs):
    """The model without the benefit columns b (linear curve) when every b equals its row's right side at
    an optimum: b_is = sum_k coef x_ik (recovery; the coefficients >= 0 and summing to at most 1, so the
    bound b <= 1 never binds) or b_is = w_i (conservation). Substituted into the targets (or the
    objective), and the benefit rows dropped: the same optimum, (units + actions) columns fewer. Returns
    (model, expand) with expand(x) the full solution of the original columns, or None."""
    m = am.model
    n, na, npairs = am.n_units, am.n_actions, am.n_pairs
    if pairs.contribution.size and (pairs.contribution < 0).any():
        return None
    sums = np.zeros(npairs)
    k_of = np.repeat(np.arange(npairs), np.diff(pairs.start))
    np.add.at(sums, k_of, pairs.contribution)
    if (sums > 1.0 + 1e-12).any():
        return None
    # b = E [w; x]
    rec = pairs.sum_alpha != 0.0
    rows = np.concatenate([k_of, np.flatnonzero(~rec)])
    cols = np.concatenate([n + pairs.action, pairs.unit[~rec]])
    vals = np.concatenate([pairs.contribution, np.ones(int((~rec).sum()))])
    E = sp.csr_matrix((vals, (rows, cols)), shape=(npairs, n + na))
    b0 = n + na
    keep_cols = np.concatenate([np.arange(b0), np.arange(b0 + npairs, m.ncol)])
    A = m.A.tocsc()
    Ab = A[:, b0:b0 + npairs]
    Ao = A[:, keep_cols]
    # the benefit rows are the first npairs rows; the others get Ab E on the unit and action columns
    other = np.arange(npairs, m.nrow)
    sub = (Ab[other] @ E).tocsc()
    sub.resize((other.size, keep_cols.size))
    A2 = (Ao[other] + sub).tocsr()
    A2.sort_indices()
    obj2 = m.obj[keep_cols].copy()
    obj2[:n + na] += E.T @ m.obj[b0:b0 + npairs]
    lb2, ub2 = m.lb[keep_cols], m.ub[keep_cols]
    sense = tuple(np.asarray(m.sense, dtype=object)[other])
    out = OptimizationProblem(A=A2, obj=obj2, lb=lb2, ub=ub2, rhs=m.rhs[other], sense=sense,
                              vtype=tuple(np.asarray(m.vtype, dtype=object)[keep_cols]), modelsense=m.modelsense,
                              row_ids=tuple(np.asarray(m.row_ids, dtype=object)[other]),
                              col_ids=tuple(np.asarray(m.col_ids, dtype=object)[keep_cols]))

    def expand(x: np.ndarray) -> np.ndarray:
        full = np.zeros(m.ncol)
        full[keep_cols] = x
        wx = np.round(x[:n + na])
        full[:n + na] = wx
        full[b0:b0 + npairs] = np.minimum(1.0, E @ wx)
        return full

    return out, expand


def _with_curves(am: ActionsModel) -> OptimizationProblem:
    """The model with y = x^curve as the piecewise-linear function Gurobi uses in prioriactions (FuncPieces
    1, FuncPieceLength round(1 / segments, 1), points on the curve), written as an incremental formulation:
    x = sum d_k, 0 <= d_k <= L_k, d_k >= L_k z_k, d_{k+1} <= L_{k+1} z_k (z binary), y = sum slope_k d_k."""
    m = am.model
    if am.curve == 1:
        return m
    step = float(r_round(np.array([1 / am.segments]), 1)[0])
    pts = [0.0]
    while pts[-1] + step < 1.0 - 1e-12:
        pts.append(pts[-1] + step)
    pts.append(1.0)
    pts = np.asarray(pts)
    lengths = np.diff(pts)
    slopes = np.diff(pts ** am.curve) / lengths
    K = lengths.size
    npairs = am.xvar.size
    ncol0 = m.ncol
    # new columns: d (npairs x K, continuous), z (npairs x (K - 1), binary)
    dcol = ncol0 + np.arange(npairs * K).reshape(npairs, K)
    zcol = ncol0 + npairs * K + np.arange(npairs * (K - 1)).reshape(npairs, K - 1)
    rows_i, rows_j, rows_x, rhs, sense = [], [], [], [], []
    r = m.nrow
    for q in range(npairs):
        # x - sum d = 0
        rows_i += [r] * (K + 1)
        rows_j += [am.xvar[q], *dcol[q]]
        rows_x += [1.0] + [-1.0] * K
        rhs.append(0.0)
        sense.append("=")
        r += 1
        # y - sum slope d = 0
        rows_i += [r] * (K + 1)
        rows_j += [am.yvar[q], *dcol[q]]
        rows_x += [1.0] + list(-slopes)
        rhs.append(0.0)
        sense.append("=")
        r += 1
        for k in range(K - 1):
            # d_k - L_k z_k >= 0 ; d_{k+1} - L_{k+1} z_k <= 0
            rows_i += [r, r]
            rows_j += [dcol[q, k], zcol[q, k]]
            rows_x += [1.0, -lengths[k]]
            rhs.append(0.0)
            sense.append(">=")
            r += 1
            rows_i += [r, r]
            rows_j += [dcol[q, k + 1], zcol[q, k]]
            rows_x += [1.0, -lengths[k + 1]]
            rhs.append(0.0)
            sense.append("<=")
            r += 1
    nnew = npairs * K + npairs * (K - 1)
    a = m.A.tocoo()
    i = np.concatenate([a.row, rows_i])
    j = np.concatenate([a.col, rows_j])
    x = np.concatenate([a.data, rows_x])
    A = sp.csr_matrix((x, (i, j)), shape=(r, ncol0 + nnew))
    ub_d = np.tile(lengths, npairs)
    return OptimizationProblem(
        A=A, obj=np.concatenate([m.obj, np.zeros(nnew)]),
        lb=np.concatenate([m.lb, np.zeros(nnew)]), ub=np.concatenate([m.ub, ub_d, np.ones(npairs * (K - 1))]),
        rhs=np.concatenate([m.rhs, rhs]), sense=m.sense + tuple(sense),
        vtype=m.vtype + ("C",) * (npairs * K) + ("B",) * (npairs * (K - 1)), modelsense=m.modelsense,
        row_ids=m.row_ids + ("curve",) * (r - m.nrow), col_ids=m.col_ids + ("curve",) * nnew)


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
        power constraints are not in it (see :meth:`compile_curves`)."""
        return compile_actions(self).model

    def add_highs_solver(self, gap: float = 0.0, time_limit: float | None = None, n_threads: int | None = None,
                         verbose: bool = False) -> ActionsProblem:
        """Solve on HiGHS to a relative ``gap`` (prioriactions's ``gap_limit``, default 0)."""
        return replace(self, solver=("highs", {"gap": gap, "time_limit": time_limit, "n_threads": n_threads,
                                               "verbose": verbose}))

    def add_gurobi_solver(self, gap: float = 0.0, time_limit: float | None = None, n_threads: int | None = None,
                          verbose: bool = False) -> ActionsProblem:
        """Solve on Gurobi (needs ``gurobipy`` and a licence)."""
        return replace(self, solver=("gurobi", {"gap": gap, "time_limit": time_limit, "n_threads": n_threads,
                                                "verbose": verbose}))

    def solve(self) -> ActionsSolution:
        from ._milp_solve import solve_model

        am = compile_actions(self, disaggregate=True)
        reduced = _eliminate_benefits(am, _pairs(self.data)) if am.curve == 1 else None
        if reduced is not None:
            m, expand = reduced
            r = solve_model(m, self.solver[0], self.solver[1])
            if r["has_solution"]:
                r["x"] = expand(np.asarray(r["x"], dtype=np.float64))
        else:
            r = solve_model(_with_curves(am), self.solver[0], self.solver[1])
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
        bm = _boundary_matrix(self.data)
        if bm is None:
            return None
        d = self.data
        n = len(d.pu)
        sol = self.units
        z = bm.T.tocsc()
        z.sort_indices()
        conn = np.zeros(n)
        for c in range(n):
            for k in range(z.indptr[c], z.indptr[c + 1]):
                if z.indices[k] != c:
                    conn[c] = conn[c] + z.data[k]
        acc = conn * sol
        for c in range(n):
            for k in range(z.indptr[c], z.indptr[c + 1]):
                r = z.indices[k]
                if r != c and sol[r] > 0.99 and sol[c] > 0.99:
                    acc[r] = acc[r] + (-1 * z.data[k])
        units = 0.0
        for v in acc:
            units = units + v
        dt = d.dist_threats
        t_unit = dt["internal_pu"].to_numpy(dtype=np.int64)
        t_threat = dt["internal_threat"].to_numpy(dtype=np.int64)
        nt = len(d.threats)
        act_of = {(int(u), int(t)): a for a, (u, t) in enumerate(zip(t_unit, t_threat))}
        by_threat = [np.sort(t_unit[t_threat == t]) for t in range(nt)]
        sa = self.actions
        per_action = np.zeros(len(dt))
        for a in range(len(dt)):
            t, u1 = t_threat[a], t_unit[a]
            total = 0.0
            acc_a = 0.0
            for u2 in by_threat[t]:
                v = bm[u1, u2]
                if u1 != u2 and v != 0:
                    total = total + v
                    if sa[a] > 0.99 and sa[act_of[(int(u2), int(t))]] > 0.99:
                        acc_a = acc_a + (-1 * v)
            per_action[a] = acc_a + total * sa[a]
        threats = np.zeros(nt)
        for a in np.lexsort((t_unit, t_threat)):
            if per_action[a] != 0:
                threats[t_threat[a]] = threats[t_threat[a]] + per_action[a]
        out = {"units": [float(r_round(np.array([units]), 3)[0])]}
        for tid, v in zip(d.threats["id"].to_numpy(), r_round(threats, 3)):
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
        return {"monitoring": self.monitoring(), "actions": self.action_table(), "local_benefit": self.local_benefit,
                "cost": self.cost(), "benefit": self.benefit(), "objective": self.objective, "status": self.status}
