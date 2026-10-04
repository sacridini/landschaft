# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration of the optimization with the rest of ls.prioritize and landschaft: Zonation's feature
transforms, the species-area extinction risk (objective and evaluation, against enumeration), the
comparison with Zonation at the same cost, SDM features, reports and plots."""

from __future__ import annotations

import itertools
import os
import warnings

import numpy as np
import optimize_cases as oc
import pandas as pd
import pytest

import landschaft as ls

P = ls.prioritize


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _raster():
    return oc._raster().add_relative_targets(0.3).add_binary_decisions().add_highs_solver(gap=0)


def test_zonation_features():
    f = oc.features()
    raw = np.nan_to_num(f.values)
    z = P.zonation_features(f)
    assert list(z["band"].values) == list(f["band"].values) and z.shape == f.shape
    top = raw.max(axis=(1, 2))[:, None, None]
    assert (np.abs(np.nan_to_num(z.values) - raw) <= 2e-5 * top).all()  # 16-bit quantization
    cond = oc.grid(np.clip(oc.smooth_field(77) * 1.5, 0, 1))
    zc = P.zonation_features(f, condition=[cond] * 5)
    assert (np.abs(np.nan_to_num(zc.values) - raw * np.nan_to_num(cond.values)) <= 4e-5 * top).all()
    p = P.problem(oc.pu_cost(), zc).add_min_set_objective().add_relative_targets(0.3).add_binary_decisions()
    assert p.solve().status == "optimal"
    with pytest.raises(ValueError, match="bits"):
        P.zonation_features(f, bits=12)


def _sar_table():
    rng = np.random.default_rng(4)
    n = 11
    t = pd.DataFrame({"id": np.arange(1.0, n + 1), "cost": rng.integers(1, 6, n).astype(float)})
    for k in range(3):
        t[f"f{k + 1}"] = np.where(rng.random(n) < 0.6, rng.integers(1, 5, n), 0).astype(float)
    return t


def test_extinction_risk_objective_is_its_definition():
    """The optimum equals enumeration of every selection within the budget, the risk being the
    piecewise-linear 1 - r^z through the breakpoints (plus the cost tie-break 0.01 x cost / total cost)."""
    from landschaft.prioritize._milp_compile import sar_lines

    t = _sar_table()
    budget, z, segs = 12.0, np.array([0.25, 0.4, 0.15]), 8
    p = (P.problem(t, ["f1", "f2", "f3"], cost_column="cost").add_min_extinction_risk_objective(budget, z, segs)
         .add_binary_decisions().add_highs_solver(gap=0))
    s = p.solve()
    amounts = t[["f1", "f2", "f3"]].to_numpy().T
    total = amounts.sum(axis=1)
    cost = t["cost"].to_numpy()
    best = np.inf
    for bits in itertools.product((0, 1), repeat=len(t)):
        x = np.asarray(bits, dtype=float)
        if cost @ x > budget + 1e-9:
            continue
        r = (amounts @ x) / total
        risk = 0.0
        for f in range(3):
            c, sl = sar_lines(z[f], segs)
            risk += max(0.0, max(1 - (c + sl * r[f])))
        best = min(best, risk + 0.01 * (cost @ x) / cost.sum())
    assert s.objective == pytest.approx(best, abs=1e-7)
    ev = P.eval_extinction_risk(p, s, z)
    exact = ev["extinction_risk"].to_numpy()[:3]
    np.testing.assert_allclose(exact, 1 - ev["relative_held"].to_numpy()[:3] ** z)
    pwl = s.objective - 0.01 * s.cost / cost.sum()
    assert exact.sum() <= pwl + 1e-9 and pwl - exact.sum() < 0.15  # the chords lie below r^z
    assert ev["extinction_risk"].iloc[-1] == pytest.approx(exact.sum())
    with pytest.raises(ValueError, match="z must"):
        P.problem(t, ["f1"], cost_column="cost").add_min_extinction_risk_objective(5, z=0)


def test_compare_zonation():
    p = _raster()
    sol = p.solve()
    c = P.compare_zonation(p, sol)
    assert list(c.table.columns[:3]) == ["feature", "milp", "zonation"]
    s = c.summary.set_index("method")
    assert s.loc["zonation", "cost"] <= s.loc["milp", "cost"] + 1e-6
    assert s.loc["milp", "cost"] == pytest.approx(sol.cost)
    assert c.table["milp_met"].all()  # the optimization meets every target
    np.testing.assert_array_equal(np.nan_to_num(c.selection.sel(band="milp").values, nan=-1),
                                  np.nan_to_num(sol.selection.values, nan=-1))
    z = P.zonation(oc.features(), cost=oc.pu_cost())
    c2 = P.compare_zonation(p, sol, zonation=z)
    pd.testing.assert_frame_equal(c.table, c2.table)
    with pytest.raises(ValueError, match="raster planning units"):
        P.compare_zonation(oc._table().add_relative_targets(0.3).add_binary_decisions())


def test_sdm_features():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    occ = ls.load_table(ls.sdm.example_data("bradypus"))
    env = ls.load_raster(ls.sdm.example_data("bioclim"))
    data = ls.sdm.prepare(occ, env, clean=False, background=500, seed=1)
    res = ls.sdm.fit(data, {"glm": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))})
    f = P.sdm_features({"bradypus": res}, env)
    assert list(f["band"].values) == ["bradypus"] and f.shape[1:] == env.shape[1:]
    b = P.sdm_features([res], env, threshold="max_tss")
    v = b.values[np.isfinite(b.values)]
    assert set(np.unique(v)) <= {0.0, 1.0} and list(b["band"].values) == ["species_1"]
    assert (b.values[np.isfinite(b.values)] == (f.values[np.isfinite(f.values)] >= res.thresholds["glm"])).all()


def test_report_and_plots(tmp_path):
    os.environ.setdefault("LANDSCHAFT_PLOT_SHOW", "0")
    p = _raster()
    r = P.report(p, zonation=True, extinction_z=0.25)
    assert set(r.tables) == {"summary", "representation", "targets", "extinction risk"}
    assert "irreplaceability" in r.maps and r.comparison is not None
    page = ls.plot(r, show=False).save(tmp_path / "report.html")
    html = page.read_text(encoding="utf-8")
    assert "<html" in html.lower() and len(html) > 10000
    ls.save(r, tmp_path / "report")
    assert (tmp_path / "report" / "selection.tif").exists()
    for obj in (p.solve(), p.add_cuts_portfolio(2).solve(), P.calibrate_penalty(p, values=[0.01, 1])):
        assert ls.plot(obj, show=False).panels
    # table units: a report without boundary and maps
    tr = P.report(oc._table().add_relative_targets(0.3).add_binary_decisions(), importance=False)
    assert "boundary" not in tr.tables["summary"] and ls.plot(tr, show=False).panels


def test_actions_plots():
    os.environ.setdefault("LANDSCHAFT_PLOT_SHOW", "0")
    cost = oc.pu_cost()
    thr = oc.grid(np.stack([oc.smooth_field(301) > 0.5, oc.smooth_field(302) > 0.6]).astype(float),
                  names=["fire", "cattle"])
    d = P.actions_raster_data(cost, oc.features(), thr, 2.0, target_recovery=0.2)
    s = d.problem("minimizeCosts").add_highs_solver(gap=0.01).solve()
    assert any(pn.kind == "map" for pn in ls.plot(s, show=False).panels)
    port = d.problem("minimizeCosts").add_cuts_portfolio(2).add_highs_solver(gap=0.01).solve()
    assert ls.plot(port, show=False).panels
