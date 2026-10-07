# SPDX-License-Identifier: GPL-3.0-or-later
"""ProxIRR, a proxy of irreplaceability (Baisero, Schuster & Plumptre 2022), as the CRAN package
proxirr 0.5 (MIT) computes it: alpha per feature and unit, beta and gamma per unit, bit for bit.

* alpha = 0 if the target is 0 (or, with ``triage``, above the feature's global amount); if the
  target reaches the global amount, 1 where the unit holds the feature, else 0; otherwise
  min(1, local / (global - target)): the share of the feature's spare amount the unit holds.
* beta = 1 - prod(1 - alpha): the chance the unit is needed for at least one feature (R's prod()).
* gamma = the number of features for which the unit is essential (alpha = 1) + beta of the others,
  which ranks units where beta saturates at 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._save import Saveable

__all__ = ["ProxIRR", "eval_proxirr", "eval_proxirr_marxan"]


@dataclass
class ProxIRR(Saveable):
    """ProxIRR of every planning unit (:func:`eval_proxirr`, :func:`eval_proxirr_marxan`).

    Attributes
    ----------
    beta, gamma
        Per planning unit, like a solution's selection: a raster (NaN outside the units) or the units'
        table with a ``beta`` / ``gamma`` column (from Marxan tables: the ``pu`` table with the column,
        as proxirr's ``marxan_betas`` and ``marxan_gammas``).
    alpha
        Per (target, unit): a sparse matrix, targets x planning units (from Marxan tables: the
        ``puvspr`` table with ``global``, ``target`` and ``alpha`` columns, as ``marxan_alphas``).
    targets
        Every target's feature, global amount and amount (from Marxan tables: ``spec`` with ``global``
        and ``target``).
    """

    beta: object = field(repr=False)
    gamma: object = field(repr=False)
    alpha: object = field(repr=False)
    targets: pd.DataFrame = field(repr=False)

    def __repr__(self) -> str:
        b = _column(self.beta, "beta")
        g = _column(self.gamma, "gamma")
        return (f"ProxIRR({b.size} units; beta max {np.nanmax(b) if b.size else np.nan:.4g}, "
                f"gamma max {np.nanmax(g) if g.size else np.nan:.4g})")

    def _save_items(self) -> dict:
        alpha = self.alpha
        if sp.issparse(alpha):
            a = alpha.tocoo()
            alpha = pd.DataFrame({"target": a.row, "unit": a.col, "alpha": a.data})
        return {"beta": self.beta, "gamma": self.gamma, "alpha": alpha, "targets": self.targets}


def _column(x, name: str) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x[name].to_numpy(dtype=np.float64)
    v = np.asarray(getattr(x, "values", x), dtype=np.float64)
    return v[np.isfinite(v)]


def _beta_gamma(alpha: sp.csr_matrix, threads: int) -> tuple[np.ndarray, np.ndarray]:
    """Beta and gamma of every unit (column) from the alphas in target order (R's vector order)."""
    from .. import _core

    c = alpha.tocsc()
    c.sort_indices()
    return _core.opt_proxirr_beta_gamma(c.indptr.astype(np.int64), c.data, threads)


def eval_proxirr(p, triage: bool = False, n_threads: int | None = None) -> ProxIRR:
    """ProxIRR (Baisero et al. 2022) of every planning unit for the problem's targets, as the CRAN
    package proxirr computes it.

    For every ``>=`` target of the problem, ``local`` is the amount in the unit, ``global`` the total
    its relative targets are shares of (the whole raster or table, the region for sub-regional
    targets) and ``target`` its amount; alpha, beta and gamma follow (see the module's notes).

    Parameters
    ----------
    p
        A :class:`Problem` with targets (relative, absolute, manual, log-linear or sub-regional); its
        objective, constraints and penalties play no part.
    triage
        proxirr's triage: a feature whose target exceeds what is left counts for nothing (alpha 0)
        instead of making every unit holding it essential.
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    ProxIRR

    Notes
    -----
    A feature absent from the landscape (global amount 0) gives alpha 0 (proxirr stops with an error).
    Targets of sense ``<=`` or ``=`` are left out.

    Examples
    --------
    >>> import landschaft as ls
    >>> P = ls.prioritize
    >>> cost = ls.load_raster(P.example_data("cost"))
    >>> features = ls.load_raster(P.example_data("features"))
    >>> p = P.problem(cost, features).add_relative_targets(0.3)
    >>> irr = P.eval_proxirr(p)
    >>> float(irr.beta.max()) <= 1.0, float(irr.gamma.max()) >= float(irr.beta.max())
    (True, True)
    """
    from .. import _core, _threads
    from ._milp_compile import NO_TARGETS, compile_problem
    from ._milp_eval import _per_unit

    if p.targets is None and p.subregional is None:
        raise ValueError("ProxIRR needs targets (add_relative_targets, add_absolute_targets, ...)")
    if p.objective is not None and p.objective[0] in NO_TARGETS:
        raise ValueError(f"the {p.objective[0]} objective has no targets")
    q = p._with(objective=("min_set",), spatial=(), portfolio=None, bounds=(), locked_in=None, locked_out=None,
                weights=None)
    _, t = compile_problem(q)
    u = p.units
    threads = _threads.resolve(n_threads)
    keep = np.flatnonzero(np.asarray(t.sense) == ">=")
    total = np.asarray(t.total if t.total is not None else u.totals[t.feature], dtype=np.float64)
    mat = sp.csr_matrix(t.matrix)[keep]
    mat.sort_indices()
    if mat.nnz and float(mat.data.min()) < 0:
        raise ValueError("ProxIRR needs amounts >= 0")
    row = np.repeat(np.arange(keep.size), np.diff(mat.indptr))
    a = _core.opt_proxirr_alpha(mat.data.astype(np.float64), total[keep][row], t.value[keep][row], bool(triage),
                                threads)
    alpha = sp.csr_matrix((np.asarray(a), mat.indices, mat.indptr), shape=(keep.size, u.n))
    beta, gamma = _beta_gamma(alpha, threads)
    targets = pd.DataFrame({"feature": [u.feature_names[f] for f in t.feature[keep]],
                            "global": total[keep], "target": t.value[keep]})
    if t.region is not None and (t.region >= 0).any():
        targets.insert(1, "region", [t.region_label[k] for k in keep])
    return ProxIRR(_per_unit(p, np.asarray(beta), ["beta"]), _per_unit(p, np.asarray(gamma), ["gamma"]), alpha,
                   targets)


def eval_proxirr_marxan(pu: pd.DataFrame, spec: pd.DataFrame, puvspr: pd.DataFrame, triage: bool = False,
                        n_threads: int | None = None) -> ProxIRR:
    """ProxIRR of a Marxan problem's planning units (proxirr's ``marxan_alphas``, ``marxan_betas`` and
    ``marxan_gammas``), from the tables :func:`marxan` takes.

    Parameters
    ----------
    pu, spec, puvspr
        Marxan's ``pu.dat`` (``id``), ``spec.dat`` (``id`` and ``prop`` (a share of the feature's total,
        used when the column is there) or ``target``) and ``puvspr.dat`` (``species``, ``pu``,
        ``amount``).
    triage, n_threads
        As :func:`eval_proxirr`.

    Returns
    -------
    ProxIRR
        ``alpha``: ``puvspr`` with ``global``, ``target`` and ``alpha``; ``beta`` / ``gamma``: ``pu``
        with the column; ``targets``: ``spec`` with ``global`` and ``target``.

    Notes
    -----
    proxirr 0.5 looks a feature's global amount and target up by its *position* in ``spec`` rather than
    by its id, so with ids other than 1, 2, ... in order every feature gets another's; and a feature
    of ``puvspr`` missing from ``spec`` stops ``marxan_betas`` with an error. Here features are looked
    up by id, and those without a target (NaN alphas) are left out of beta and gamma.

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> pu = pd.DataFrame({"id": [1, 2, 3], "cost": [1.0, 1.0, 1.0]})
    >>> spec = pd.DataFrame({"id": [10, 20], "target": [4.0, 1.0]})
    >>> puvspr = pd.DataFrame({"species": [10, 10, 20, 20], "pu": [1, 2, 2, 3], "amount": [3.0, 2.0, 1.0, 1.0]})
    >>> irr = ls.prioritize.eval_proxirr_marxan(pu, spec, puvspr)
    >>> irr.alpha["alpha"].tolist()
    [1.0, 1.0, 1.0, 1.0]
    >>> irr.gamma["gamma"].tolist()
    [1.0, 2.0, 1.0]
    """
    from .. import _core, _threads

    pu, spec, puvspr = pd.DataFrame(pu), pd.DataFrame(spec), pd.DataFrame(puvspr)
    for tab, cols, what in ((pu, ("id",), "pu"), (spec, ("id",), "spec"), (puvspr, ("species", "pu", "amount"),
                                                                         "puvspr")):
        missing = [c for c in cols if c not in tab.columns]
        if missing:
            raise ValueError(f"{what} needs the columns {missing}")
    if "prop" not in spec.columns and "target" not in spec.columns:
        raise ValueError("spec needs a 'prop' or a 'target' column")
    if not spec["id"].is_unique:
        raise ValueError("feature ids must be unique")
    threads = _threads.resolve(n_threads)
    species = puvspr["species"].to_numpy()
    amount = puvspr["amount"].to_numpy(dtype=np.float64)
    # every feature's global amount: R's sum() of its rows, in the table's order
    glob = []
    for sid in spec["id"].to_numpy():
        glob.append(float(_core.nlm_r_sum(np.ascontiguousarray(amount[species == sid]))))
    glob = np.asarray(glob, dtype=np.float64)
    target = glob * spec["prop"].to_numpy(dtype=np.float64) if "prop" in spec.columns \
        else spec["target"].to_numpy(dtype=np.float64)
    pos = pd.Series(np.arange(len(spec)), index=spec["id"].to_numpy())
    k = pos.reindex(species).to_numpy(dtype=np.float64)  # NaN: a feature not in spec
    ok = np.isfinite(k)
    g = np.full(len(puvspr), np.nan)
    tt = np.full(len(puvspr), np.nan)
    g[ok] = glob[k[ok].astype(np.int64)]
    tt[ok] = target[k[ok].astype(np.int64)]
    a = np.asarray(_core.opt_proxirr_alpha(amount, g, tt, bool(triage), threads))
    alphas = puvspr.copy()
    alphas["global"] = g
    alphas["target"] = tt
    alphas["alpha"] = a
    # every unit's alphas in the table's order (proxirr: alphas[alphas$pu == x, "alpha"])
    upos = pd.Series(np.arange(len(pu)), index=pu["id"].to_numpy())
    if not upos.index.is_unique:
        raise ValueError("planning-unit ids must be unique")
    unit = upos.reindex(puvspr["pu"].to_numpy()).to_numpy(dtype=np.float64)
    inside = np.isfinite(unit)
    ui = unit[inside].astype(np.int64)
    order = np.argsort(ui, kind="stable")
    start = np.zeros(len(pu) + 1, dtype=np.int64)
    np.add.at(start, ui + 1, 1)
    start = np.cumsum(start)
    beta, gamma = _core.opt_proxirr_beta_gamma(start, a[inside][order], threads)
    b = pu.copy()
    b["beta"] = np.asarray(beta)
    gm = pu.copy()
    gm["gamma"] = np.asarray(gamma)
    targets = spec.copy()
    targets["global"] = glob
    targets["target"] = target
    return ProxIRR(b, gm, alphas, targets)
