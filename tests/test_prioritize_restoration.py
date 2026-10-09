# SPDX-License-Identifier: GPL-3.0-or-later
"""The restoration problem (ls.prioritize.restoration_problem): plangea's procedure (method="steps")
against plangea's own loop, the risk of ecosystem collapse against multi_calc_ec, the exact solution
against an independent optimiser, and the evaluation of the risk reduction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plangea_cases as pc
import pytest
import scipy.sparse as sp
from test_plangea_reference import REF, _vegetation

import landschaft as ls
from landschaft.prioritize import _restoration as R
from landschaft.sdm._habitat import HabitatAreas

P = ls.prioritize


@pytest.fixture(scope="module")
def ref():
    with np.load(REF / "reference.npz") as z:
        arrays = {k: z[k] for k in z.files}
    texts = json.loads((REF / "reference.json").read_text(encoding="utf-8"))["texts"]
    return arrays, texts


@pytest.fixture(scope="module")
def case():
    """bd_absent: the vegetation, the areas of habitat and the restoration problem with its risk benefit."""
    cover, veg = _vegetation(pc.OA_CASES["oa_past_er"])
    names, r = pc.ranges()
    aoh = ls.sdm.area_of_habitat(pc.grid(r, names=names), cover, pc.NATURAL, suitability=pc.habitat_table(),
                                 original=veg)
    p = P.restoration_problem(veg).add_extinction_risk_benefit(aoh)
    return cover, veg, aoh, p


# ----------------------------------------------------------------------------- parity with plangea

def test_units_and_bounds_are_plangea(ref, case):
    arrays, _ = ref
    *_, p = case
    np.testing.assert_array_equal(p.index, arrays["bd_absent/master_index"].astype(np.int64) - 1)
    np.testing.assert_array_equal(p.upper, arrays["scen_1/ub"])


def test_biodiversity_layer_is_plangea(ref, case):
    """plangea's bd: the finite-difference slopes of the species' risk times the habitat share a unit of
    a cell restored gives; to the last bits but one (the gain is stored per km² and divided back)."""
    arrays, _ = ref
    *_, p = case
    bd = R._steps_layer(p, p.benefits[0], np.zeros(p.n))
    np.testing.assert_allclose(bd, arrays["bd_absent/bd"], rtol=1e-15, atol=0)


@pytest.mark.parametrize("name", list(pc.SCEN_CASES))
def test_steps_is_plangea(ref, case, name):
    """scen's loop: the target in steps, bd refreshed between them, each step's LP (Gurobi there, HiGHS
    here; their optimum is unique)."""
    arrays, _ = ref
    *_, p = case
    spec = pc.SCEN_CASES[name]
    q = p.add_area_target(float(arrays[f"{name}/target"][0]))
    sol = q.solve(method="steps", steps=spec["steps"])
    np.testing.assert_allclose(sol.values, arrays[f"{name}/x"], rtol=0, atol=1e-12)
    assert sol.status == "heuristic" and sol.method == "steps"
    # the refreshed layers at plangea's restored shares: the finite difference over delta = 1e-6 of an
    # area of ~20 turns last-bit differences of the habitat sums into ~1e-8 of the slope
    for k in range(1, spec["steps"]):
        bd = R._steps_layer(p, p.benefits[0], arrays[f"{name}/cum_{k}"])
        np.testing.assert_allclose(bd, arrays[f"{name}/bd_{k}"], rtol=1e-7)


def test_ecosystem_risk_is_plangea_corrected(ref, case):
    """multi_calc_ec's layer with the ecoregions' natural area over all their land (the corrected form);
    plangea sums it over its problem's cells only, which overstates the loss."""
    arrays, _ = ref
    cover, veg, _, _ = case
    ea = P.ecosystem_areas(cover, pc.NATURAL, pc.ANTHROPIC, pc.grid(pc.ecoregion_codes()), cell_area=1.0)
    p = P.restoration_problem(veg, cell_area=1.0).add_extinction_risk_benefit(ea)
    master = arrays["ec_land/master_index"].astype(np.int64) - 1
    x = pc.ec_x(master.size)
    for name, rtol in (("ec_land", 1e-7), ("ec_master", None)):
        pos = np.searchsorted(arrays[f"{name}/domain"].astype(np.int64) - 1, master)
        for key, cum in (("ec0", np.zeros(p.n)), ("ecx", x)):
            ours = R._steps_layer(p, p.benefits[0], cum)
            theirs = arrays[f"{name}/{key}"][pos]
            if rtol is None:
                assert np.max(np.abs(ours - theirs) / theirs) > 0.1
            else:
                np.testing.assert_allclose(ours, theirs, rtol=rtol)


def test_risk_reduction_is_plangea_corrected(ref, case):
    """multi_exrisk_aggregate (its square root after the loop): the total and, from z's deviation 0.1,
    plangea's two forms; the default treats z as one parameter of every species."""
    arrays, _ = ref
    *_, p = case
    x = arrays["bd_absent/x_vals"]
    for method, key in (("independent", "quad"), ("linear", "lin")):
        val, unc = arrays[f"bd_absent/exrisk_{key}"]
        rr = P.eval_risk_reduction(p, x, z_sd=0.1, method=method)
        assert rr.total == pytest.approx(val, rel=1e-12)
        assert rr.sd == pytest.approx(unc, rel=1e-6)
    rr = P.eval_risk_reduction(p, p.evaluate(x), z_sd=0.1)
    d = (P.eval_risk_reduction(p, x).table["reduction"].to_numpy())
    assert rr.total == pytest.approx(d.sum(), rel=1e-15)
    assert rr.sd <= arrays["bd_absent/exrisk_lin"][1] * (1 + 1e-9)
    assert set(rr.table.columns) >= {"species", "risk_before", "risk_after", "reduction"}


def test_uniform_is_plangea(case):
    *_, p = case
    target = 10.0
    u = p.add_area_target(target).uniform()
    np.testing.assert_allclose(u.values, p.upper * (target / np.sum(p.upper * p.area)), rtol=1e-15)
    assert u.area == pytest.approx(target, rel=1e-12)


# ----------------------------------------------------------------------------- the exact solution

def _scipy_optimum(p, target):
    """An independent solution of the concave problem (one landscape area row) by scipy's SLSQP."""
    from scipy.optimize import minimize

    b = p.benefits[0]
    g = b.gain.toarray()

    def f(x):
        r = (b.current + g @ x) / b.potential
        return -np.sum(np.minimum(r, 1.0) ** b.z)

    def grad(x):
        r = (b.current + g @ x) / b.potential
        d = np.where(r < 1.0, b.z * np.maximum(r, 1e-300) ** (b.z - 1.0), 0.0) / b.potential
        return -(g.T @ d)

    best = None
    for start in (p.upper * target / np.sum(p.upper * p.area), np.zeros(p.n)):
        res = minimize(f, start, jac=grad, bounds=list(zip(np.zeros(p.n), p.upper)), method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda x: target - p.area @ x, "jac": lambda x: -p.area}],
                       options={"maxiter": 2000, "ftol": 1e-14})
        if best is None or res.fun < best.fun:
            best = res
    return -best.fun - np.sum(np.minimum(b.current / b.potential, 1.0) ** b.z)


@pytest.mark.parametrize("share", [0.1, 0.3, 0.6])
def test_exact_is_the_optimum(case, share):
    *_, p = case
    target = share * float(p.upper @ p.area)
    q = p.add_area_target(target)
    sol = q.solve()
    assert sol.status == "optimal" and sol.method == "exact"
    assert sol.bound >= sol.benefit - 1e-9 and (sol.bound - sol.benefit) <= 1e-7 * max(1.0, sol.bound)
    assert sol.area <= target * (1 + 1e-9)
    assert sol.benefit == pytest.approx(_scipy_optimum(q, target), rel=1e-6)
    # plangea's procedure is a heuristic: never better, closer with more steps
    for steps in (1, 3, 10):
        assert q.solve(method="steps", steps=steps).benefit <= sol.benefit + 1e-12
    assert q.uniform().benefit < sol.benefit


def test_z_per_species(ref, case):
    """z: one number for all species (default plangea's 0.25), one per species or a mapping; the exact
    optimum with species of different z is still the optimum, and the uncertainty takes a z_sd per
    species."""
    arrays, _ = ref
    _, veg, aoh, p = case
    names = list(aoh.table["species"])
    assert np.all(p.benefits[0].z == 0.25)
    same = P.restoration_problem(veg).add_extinction_risk_benefit(aoh, z=[0.25] * len(names))
    x = arrays["bd_absent/x_vals"]
    assert same.evaluate(x).benefit == p.evaluate(x).benefit
    zmap = {"sp_a": 0.15, "sp_b": 0.4}
    q = P.restoration_problem(veg).add_extinction_risk_benefit(aoh, z=zmap)
    arr = P.restoration_problem(veg).add_extinction_risk_benefit(
        aoh, z=[zmap.get(s, 0.25) for s in names])
    b = q.benefits[0]
    assert dict(zip(b.species, b.z)) == {s: zmap.get(s, 0.25) for s in b.species}
    np.testing.assert_array_equal(arr.benefits[0].z, b.z)
    r0, r1 = b.current / b.potential, (b.current + b.gain @ x) / b.potential
    manual = np.sum(np.minimum(r1, 1) ** b.z - np.minimum(r0, 1) ** b.z)
    assert q.evaluate(x).benefit == pytest.approx(manual, rel=1e-14)
    target = 0.3 * float(q.upper @ q.area)
    qq = q.add_area_target(target)
    sol = qq.solve()
    assert sol.status == "optimal" and sol.benefit == pytest.approx(_scipy_optimum(qq, target), rel=1e-6)
    assert sol.benefit >= qq.solve(method="steps", steps=3).benefit - 1e-12
    one = P.eval_risk_reduction(q, x, z_sd=0.1)
    per = P.eval_risk_reduction(q, x, z_sd=dict.fromkeys(b.species, 0.1))
    assert one.sd == pytest.approx(per.sd, rel=1e-15)
    only_a = P.eval_risk_reduction(q, x, z_sd={"sp_a": 0.1}, method="linear")
    assert 0 < only_a.sd < one.sd
    with pytest.raises(ValueError, match="z must be"):
        p.add_extinction_risk_benefit(aoh, z={"sp_a": 1.2}, name="bad")
    with pytest.raises(ValueError, match="no species"):
        p.add_extinction_risk_benefit(aoh, z={"nobody": 0.3}, name="bad")
    with pytest.raises(ValueError, match="one per species"):
        p.add_extinction_risk_benefit(aoh, z=[0.2, 0.3], name="bad")


def test_min_cost_is_the_inverse_of_max_benefit(case):
    *_, p = case
    rng = np.random.default_rng(3)
    cost = rng.random(p.n) * 5 + 1
    budget = 0.25 * float(np.sum(cost * p.area * p.upper))
    best = p.add_budget(budget, cost).solve()
    assert best.cost <= budget * (1 + 1e-9) and best.status == "optimal"
    cheap = p.add_min_cost_objective(best.benefit * (1 - 1e-9), cost).solve()
    assert cheap.status == "optimal"
    assert cheap.benefit >= best.benefit * (1 - 1e-6)
    assert cheap.cost == pytest.approx(budget, rel=1e-5)
    assert cheap.objective == pytest.approx(cheap.cost)


def test_regions_and_linear_benefits(case):
    *_, p = case
    rng = np.random.default_rng(4)
    carbon = pc.grid(rng.random(pc.SHAPE) * 100)
    regions = pc.grid(np.where(np.arange(pc.SHAPE[1])[None, :] < 5, 1.0, 2.0) * np.ones(pc.SHAPE))
    q = (p.add_linear_benefit(carbon, weight=0.001, name="carbon")
         .add_area_target({1: 3.0, 2: 4.0}, regions=regions))
    sol = q.solve()
    reg = regions.values.ravel()[q.index]
    assert sol.status == "optimal"
    assert np.sum(sol.values[reg == 1] * q.area[reg == 1]) <= 3.0 * (1 + 1e-9)
    assert np.sum(sol.values[reg == 2] * q.area[reg == 2]) <= 4.0 * (1 + 1e-9)
    t = sol.benefits.set_index("benefit")
    assert t.loc["carbon", "value"] == pytest.approx(float(np.sum(carbon.values.ravel()[q.index] * q.area * sol.values)))
    assert sol.benefit == pytest.approx(float(t["weighted"].sum()))
    # a minimum per region
    low = p.add_area_target(pd.DataFrame({"region": [1, 2], "target": [2.0, 20.0], "sense": [">=", "<="]}),
                            regions=regions).solve()
    assert np.sum(low.values[reg == 1] * q.area[reg == 1]) >= 2.0 * (1 - 1e-9)


def test_linear_benefit_is_the_greedy_knapsack():
    rng = np.random.default_rng(5)
    up = rng.random((12, 15))
    value = rng.random((12, 15))
    p = P.restoration_problem(up, cell_area=2.0).add_linear_benefit(value, per="area").add_area_target(40.0)
    sol = p.solve()
    order = np.argsort(-value.ravel()[p.index])
    x, left = np.zeros(p.n), 40.0
    for i in order:
        x[i] = min(p.upper[i], left / 2.0)
        left -= 2.0 * x[i]
    assert sol.benefit == pytest.approx(float(np.sum(value.ravel()[p.index] * 2.0 * x)), rel=1e-9)


def test_restoration_discount(case):
    *_, aoh, _ = case
    veg = case[1]
    full = P.restoration_problem(veg).add_extinction_risk_benefit(aoh, restoration_discount=1.0).add_area_target(5.0)
    assert full.solve(method="steps").benefit == 0.0
    half = P.restoration_problem(veg).add_extinction_risk_benefit(aoh, restoration_discount=0.5)
    g = aoh.gain.copy()
    g.data = g.data * 0.5
    manual = P.restoration_problem(veg).add_extinction_risk_benefit(HabitatAreas(aoh.table, g, aoh._template))
    x = np.random.default_rng(6).random(half.n) * half.upper
    assert half.evaluate(x).benefit == pytest.approx(manual.evaluate(x).benefit, rel=1e-14)
    per = {s: (0.5 if s == "sp_a" else 0.0) for s in aoh.table["species"]}
    one = P.restoration_problem(veg).add_extinction_risk_benefit(aoh, restoration_discount=per)
    assert one.benefits[0].discount[list(one.benefits[0].species).index("sp_a")] == 0.5


def test_intactness_benefit_is_its_gradient():
    rng = np.random.default_rng(7)
    up = rng.random((8, 9)) * 0.5
    hfi = rng.random((8, 9)) * 20
    red = rng.random((8, 9)) * 4
    p = P.restoration_problem(up, cell_area=1.0).add_intactness_benefit(hfi, red, radius=2, cell_size=1.0)
    g = ls.intactness_gradient(hfi, red, 2, cell_size=1.0).values.ravel()[p.index]
    np.testing.assert_array_equal(p.benefits[0].coef, g)
    q = p.add_area_target(5.0)
    assert q.solve().benefit >= q.solve(method="steps", steps=3).benefit - 1e-9


def test_errors(case):
    *_, aoh, p = case
    with pytest.raises(ValueError, match="no cell"):
        P.restoration_problem(np.zeros((3, 3)), cell_area=1.0)
    with pytest.raises(ValueError, match="restoration_discount"):
        p.add_extinction_risk_benefit(aoh, restoration_discount=1.5)
    with pytest.raises(ValueError, match="sense"):
        p.add_area_target(1.0, sense="<")
    with pytest.raises(ValueError, match="steps"):
        p.add_area_target(1.0).add_budget(1.0, 1.0).solve(method="steps")
    with pytest.raises(ValueError, match="area target"):
        p.solve(method="steps")
    with pytest.raises(ValueError, match="cost"):
        p.add_min_cost_objective(0.1)
    with pytest.raises(RuntimeError, match="infeasible"):
        p.add_area_target(1e9, sense=">=").solve()
    with pytest.raises(ValueError, match="named"):
        p.add_linear_benefit(1.0, name="extinction_risk")
    with pytest.raises(ValueError, match="HabitatAreas"):
        p.add_extinction_risk_benefit(object())


def _synthetic(side=100, n_species=100, seed=1, radius=(0.1, 0.3)):
    """Circular ranges with random habitat shares on a side x side grid; anthropic land everywhere."""
    rng = np.random.default_rng(seed)
    anth = rng.random((side, side)) * 0.7
    yy, xx = np.mgrid[0:side, 0:side] / side
    rows, cols, vals, cur, pot = [], [], [], [], []
    for s in range(n_species):
        cx, cy, rad = rng.random(), rng.random(), radius[0] + radius[1] * rng.random()
        cells = np.flatnonzero((((xx - cx) ** 2 + (yy - cy) ** 2) < rad ** 2).ravel())
        share = rng.random(cells.size)
        rows += [s] * cells.size
        cols += cells.tolist()
        vals += share.tolist()
        a = anth.ravel()[cells]
        cur.append(float(np.sum((1 - a) * share)))
        pot.append(cur[-1] + float(np.sum(a * share)))
    gain = sp.csr_matrix((vals, (rows, cols)), shape=(n_species, side * side))
    table = pd.DataFrame({"species": [f"s{k}" for k in range(n_species)], "current": cur, "potential": pot,
                          "range_cells": 0, "classes": "", "valid": True})
    return anth, HabitatAreas(table=table, gain=gain, _template=None)


def test_exact_scales_with_the_units():
    """10^4 cells and 100 species (1.7 x 10^5 gains): proven within the tolerance in about a second,
    better than plangea's procedure; the cheapest plan for nearly that benefit costs less."""
    import time

    anth, aoh = _synthetic()
    p = P.restoration_problem(anth, cell_area=1.0).add_extinction_risk_benefit(aoh).add_area_target(0.1 * anth.sum())
    t = time.perf_counter()
    sol = p.solve()
    assert time.perf_counter() - t < 20
    assert sol.status == "optimal" and sol.bound - sol.benefit <= 1e-7 * sol.bound
    assert sol.benefit >= p.solve(method="steps", steps=5).benefit
    anth, aoh = _synthetic(40, 30)
    p = P.restoration_problem(anth, cell_area=1.0).add_extinction_risk_benefit(aoh).add_area_target(0.1 * anth.sum())
    sol = p.solve()
    cheap = p.add_min_cost_objective(0.999 * sol.benefit, cost=1.0).solve()
    assert cheap.status == "optimal" and cheap.benefit >= 0.999 * sol.benefit * (1 - 1e-9)
    assert cheap.bound <= cheap.cost < sol.area


def test_gurobi_solver(case):
    pytest.importorskip("gurobipy")
    *_, p = case
    q = p.add_area_target(10.0)
    a = q.solve()
    try:
        b = q.add_gurobi_solver().solve()
    except Exception as e:  # noqa: BLE001 - no licence
        pytest.skip(f"Gurobi unavailable: {e}")
    assert b.benefit == pytest.approx(a.benefit, rel=1e-7)


def test_exact_with_species_at_their_potential():
    """Species restored to their whole potential habitat put a kink in the benefit (r = 1), where the
    Frank-Wolfe bound alone does not close: lifted, they are proven optimal in a few vertices."""
    anth, aoh = _synthetic(14, 20, seed=3, radius=(0.08, 0.2))
    p = P.restoration_problem(anth, cell_area=1.0).add_extinction_risk_benefit(aoh)
    q = p.add_area_target(0.3 * anth.sum())
    sol = q.solve()
    rr = P.eval_risk_reduction(q, sol).table
    assert ((rr["current"] + rr["gained"]) >= rr["potential"] * (1 - 1e-9)).sum() >= 2
    assert sol.status == "optimal" and sol.rounds < 200
    assert sol.bound >= sol.benefit - 1e-12 and sol.bound - sol.benefit <= 1e-7 * sol.bound
    assert sol.benefit == pytest.approx(_scipy_optimum(q, 0.3 * anth.sum()), rel=1e-6)
    assert sol.benefit >= q.solve(method="steps", steps=5).benefit
