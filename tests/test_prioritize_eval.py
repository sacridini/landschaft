# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluating solutions against prioritizr 8.0.6's eval_* (tests/optimize_cases.py, group "eval"): the
summaries and the Ferrier and rarity scores bit for bit, the replacement and rank importance from
re-solves (equal optima)."""

from __future__ import annotations

import warnings

import numpy as np
import optimize_cases as oc
import pandas as pd
import pytest
import scipy.sparse as sp
from optimize_reference import result

import landschaft as ls

P = ls.prioritize


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _raster_problem():
    return oc._raster().add_relative_targets(0.3).add_binary_decisions()


def _table_problem():
    t = pd.DataFrame({"feature": ["f1", "f2", "f4", "f4", "f5"],
                      "type": ["relative", "absolute", "relative", "absolute", "relative"],
                      "sense": [">=", "<=", "=", "<=", ">="], "target": [0.3, 2.0, 0.2, 120, 0.25]})
    return oc._table().add_manual_targets(t).add_binary_decisions()


def _matrix(name: str, n: int) -> sp.csr_matrix:
    t = getattr(oc, name)()
    return sp.csr_matrix((t["x"], (t["i"].astype(int) - 1, t["j"].astype(int) - 1)), shape=(n, n))


def _same(df: pd.DataFrame, r: dict, key: str, cols=None):
    for col in cols or [c for c in df.columns if c not in ("summary", "feature")]:
        want = np.asarray(r[f"{key}/{col}"], dtype=np.float64)
        np.testing.assert_array_equal(np.asarray(df[col], dtype=np.float64), want, err_msg=f"{key}/{col}")


@pytest.mark.external_reference
def test_summaries_raster():
    r = result("eval", "eval_raster")
    p, s = _raster_problem(), oc.sol_raster()
    _same(P.eval_n_summary(p, s), r, "n")
    _same(P.eval_cost_summary(p, s), r, "cost")
    _same(P.eval_boundary_summary(p, s), r, "boundary")
    _same(P.eval_boundary_summary(p, s, edge_factor=0.3), r, "boundary_edge")
    rep = P.eval_feature_representation_summary(p, s)
    _same(rep, r, "representation")
    assert list(rep["feature"]) == r["representation/feature"]
    cov = P.eval_target_coverage_summary(p, s)
    _same(cov, r, "coverage")
    assert list(cov["feature"]) == r["coverage/feature"]


@pytest.mark.external_reference
def test_summaries_proportion():
    r = result("eval", "eval_raster_proportion")
    p = oc._raster().add_absolute_targets([1.2, 0.9, 1, 120, 4]).add_proportion_decisions()
    s = oc.solp_raster()
    _same(P.eval_n_summary(p, s), r, "n")
    _same(P.eval_cost_summary(p, s), r, "cost")
    _same(P.eval_boundary_summary(p, s), r, "boundary")
    _same(P.eval_feature_representation_summary(p, s), r, "representation")
    _same(P.eval_target_coverage_summary(p, s), r, "coverage")


@pytest.mark.external_reference
def test_summaries_table():
    r = result("eval", "eval_table")
    p, s = _table_problem(), oc.sol_table()
    n = len(oc.pu_table())
    _same(P.eval_n_summary(p, s), r, "n")
    _same(P.eval_cost_summary(p, s), r, "cost")
    _same(P.eval_boundary_summary(p, s, data=_matrix("bnd_table", n)), r, "boundary")
    _same(P.eval_connectivity_summary(p, s, data=_matrix("conn_table", n)), r, "connectivity")
    _same(P.eval_asym_connectivity_summary(p, s, data=_matrix("asym_table", n)), r, "asym")
    _same(P.eval_feature_representation_summary(p, s), r, "representation")
    cov = P.eval_target_coverage_summary(p, s)
    _same(cov, r, "coverage")
    assert list(cov["feature"]) == r["coverage/feature"]


@pytest.mark.external_reference
def test_ferrier_and_rarity():
    r = result("eval", "eval_raster")
    p, s = _raster_problem(), oc.sol_raster()
    fe = P.eval_ferrier_importance(p, s)
    assert list(fe["band"].values) == ["f1", "f2", "f3", "f4", "f5", "total"]
    for name in fe["band"].values:
        np.testing.assert_array_equal(fe.sel(band=name).values, r[f"ferrier/{name}"], err_msg=name)
    np.testing.assert_array_equal(P.eval_rare_richness_importance(p, s).values, r["rare/rwr"])
    np.testing.assert_array_equal(P.eval_rare_richness_importance(p, s, rescale=False).values, r["rare_raw/rwr"])


def test_pnorm_is_r():
    from landschaft import _core

    x = np.array([-40, -37.6, -10, -5.7, -5.6, -1, -0.6744, -1e-17, 0, 0.3, 0.6745, 2.5, 5.65, 6, 8.3, 40, np.inf,
                  -np.inf])
    v = _core.opt_eval_pnorm(x)
    from scipy.stats import norm

    np.testing.assert_allclose(v, norm.cdf(x), rtol=1e-14, atol=1e-300)


def _solution_raster(values):
    out = np.full(oc.NY * oc.NX, np.nan)
    out[oc._units()] = values
    return out.reshape(oc.NY, oc.NX)


def test_replacement_importance():
    r = result("eval", "eval_replacement")
    p = _raster_problem().add_highs_solver(gap=0)
    s = np.asarray(r["solution"], dtype=np.float64)
    rc = P.eval_replacement_importance(p, s, rescale=False)
    want = np.asarray(r["rc"], dtype=np.float64).reshape(oc.NY, oc.NX)
    np.testing.assert_allclose(rc.values, want, rtol=1e-9, atol=1e-9)
    scaled = P.eval_replacement_importance(p, s)
    np.testing.assert_allclose(scaled.values, np.asarray(r["rc_scaled"]).reshape(oc.NY, oc.NX), rtol=1e-9, atol=1e-9)


def test_replacement_importance_max():
    r = result("eval", "eval_replacement_max")
    p = (oc._tp().add_max_utility_objective(1500).add_boundary_penalties(0.001, data=_matrix("bnd_table", len(oc.pu_table())))
         .add_binary_decisions().add_highs_solver(gap=0))
    s = np.asarray(r["solution"], dtype=np.float64)
    rc = P.eval_replacement_importance(p, s, rescale=False)
    np.testing.assert_allclose(rc["rc"].to_numpy(), np.asarray(r["rc"], dtype=np.float64), rtol=1e-7, atol=1e-7)


def test_rank_importance():
    r = result("eval", "eval_rank")
    p = _raster_problem().add_highs_solver(gap=0)
    s = np.asarray(r["solution"], dtype=np.float64)
    rank, info = P.eval_rank_importance(p, s, n=4, rescale=False)
    np.testing.assert_array_equal(info["budget"].to_numpy(), np.asarray(r["budgets"], dtype=np.float64))
    np.testing.assert_allclose(info["objective"].to_numpy(), np.asarray(r["objective"]), rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(np.nan_to_num(rank.values, nan=-1),
                                  np.nan_to_num(np.asarray(r["rank"]).reshape(oc.NY, oc.NX), nan=-1))


def test_landscape_of_the_solution():
    p = _raster_problem().add_highs_solver(gap=0)
    sol = p.solve()
    t = P.eval_landscape_metrics(sol, level="class", metric=["np", "ca"])
    assert set(t["metric"]) == {"np", "ca"} and (t["class"] == 1).all()
    ref = ls.lsm.landscape_metrics(P._milp_eval._selection_map(sol), level="class", metric=["ca"], nodata=np.nan)
    assert t.loc[t["metric"] == "ca", "value"].item() == ref.loc[ref["class"] == 1, "value"].item()
    m, stats = P.eval_mspa(sol)
    assert stats["pixels"].sum() == oc.NY * oc.NX
    assert stats.loc[[c for c in stats.index if c not in ("Background", "Missing", "Core-Opening", "Border-Opening")], "pixels"].sum() == sol.values.sum()


def test_inputs_checked():
    p = _raster_problem()
    with pytest.raises(ValueError, match="one value per planning unit"):
        P.eval_n_summary(p, np.ones(3))
    with pytest.raises(ValueError, match="selects no planning unit"):
        P.eval_ferrier_importance(p, np.zeros(p.units.n))
    with pytest.raises(ValueError, match="no targets"):
        P.eval_target_coverage_summary(oc._rp().add_max_utility_objective(2500), oc.sol_raster())
    with pytest.raises(ValueError, match="either n or budgets"):
        P.eval_rank_importance(p, oc.sol_raster())
    with pytest.raises(ValueError, match="table planning units need"):
        P.eval_boundary_summary(_table_problem(), oc.sol_table())
    # a Solution, a raster, a table and plain values give the same summary
    sol = p.add_highs_solver(gap=0.05).solve()
    a = P.eval_cost_summary(p, sol)["cost"].item()
    assert a == P.eval_cost_summary(p, sol.selection)["cost"].item() == P.eval_cost_summary(p, sol.values)["cost"].item()
