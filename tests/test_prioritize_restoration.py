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
    assert cheap.rounds < 25  # the bisection's 25 solves
    half = p.add_min_cost_objective(0.5 * sol.benefit, cost=1.0).solve()
    assert half.status == "optimal" and half.rounds <= 15 and half.benefit >= 0.5 * sol.benefit * (1 - 1e-9)


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


# ----------------------------------------------------------------------------- conservation and conversion

@pytest.fixture(scope="module")
def natural_case(case):
    """bd_absent's habitats with their loss, and the natural share of every cell."""
    cover, veg, _, _ = case
    names, r = pc.ranges()
    aoh = ls.sdm.area_of_habitat(pc.grid(r, names=names), cover, pc.NATURAL, suitability=pc.habitat_table(),
                                 original=veg, loss=True)
    nat = sum(np.nan_to_num(cover.values[k]) for k in range(len(pc.NATURAL)))
    return aoh, pc.grid(nat)


@pytest.mark.parametrize("name", list(pc.SCEN_CD_CASES))
def test_conversion_and_conservation_steps_are_plangea(ref, natural_case, name):
    """scen's loop for D (x in [-natural, 0]) and C (corrected: its baseline loses the natural land not
    conserved, as an area): the pixels, the first layer, the plan and the refreshed layers."""
    arrays, _ = ref
    aoh, nat = natural_case
    spec = pc.SCEN_CD_CASES[name]
    conv = spec["type"] == "D"
    if "cols" in spec:
        nat = nat.where(np.arange(pc.SHAPE[1])[None, :] < spec["cols"], 0.0)
    p = P.restoration_problem(convertible=nat) if conv else P.restoration_problem(conservable=nat)
    p = p.add_extinction_risk_benefit(aoh)
    assert p.actions == (("convert",) if conv else ("conserve",))
    np.testing.assert_array_equal(p.index, arrays[f"{name}/master_n"].astype(np.int64) - 1)
    sign = -1.0 if conv else 1.0  # plangea's x of conversion is negative, its layer positive
    np.testing.assert_allclose(sign * R._steps_layer(p, p.benefits[0], np.zeros(p.n)), arrays[f"{name}/bd_0"],
                               rtol=1e-7)
    q = p.add_area_target(float(arrays[f"{name}/target"][0]), sense=">=" if conv else "<=")
    sol = q.solve(method="steps", steps=spec["steps"])
    np.testing.assert_allclose(sign * sol.values, arrays[f"{name}/x"], rtol=0, atol=1e-12)
    for k in range(1, spec["steps"]):
        bd = R._steps_layer(p, p.benefits[0], sign * arrays[f"{name}/cum_{k}"])
        np.testing.assert_allclose(sign * bd, arrays[f"{name}/bd_{k}"], rtol=1e-7)


def _scipy_general(q):
    """An independent optimum of the concave problem with any area rows, by scipy's SLSQP."""
    from scipy.optimize import minimize

    rows, rhs, sense = R._feasible_set(q)
    dense = rows.toarray()
    cons = []
    for a, r, s in zip(dense, rhs, sense):
        if s == "<=":
            cons.append({"type": "ineq", "fun": lambda x, a=a, r=r: r - a @ x, "jac": lambda x, a=a: -a})
        elif s == ">=":
            cons.append({"type": "ineq", "fun": lambda x, a=a, r=r: a @ x - r, "jac": lambda x, a=a: a})
        else:
            cons.append({"type": "eq", "fun": lambda x, a=a, r=r: a @ x - r, "jac": lambda x, a=a: a})
    risks = [b for b in q.benefits if isinstance(b, R._Risk)]
    lin = sum((b.weight * b.coef for b in q.benefits if isinstance(b, R._Linear)), np.zeros(q.n))

    def f(x):
        v = lin @ x
        for b in risks:
            v += b.weight * np.sum(np.minimum(np.maximum(b.base + b.gain @ x, 0) / b.potential, 1.0) ** b.z)
        return -v

    def grad(x):
        g = lin.copy()
        for b in risks:
            r = (b.base + b.gain @ x) / b.potential
            d = np.where(r < 1.0, b.z * np.maximum(r, 1e-12) ** (b.z - 1.0), 0.0) / b.potential
            g += b.weight * (b.gain.T @ d)
        return -g

    best = None
    for start in (q.upper * 0.5, np.zeros(q.n), q.upper):
        res = minimize(f, start, jac=grad, bounds=list(zip(np.zeros(q.n), q.upper)), method="SLSQP",
                       constraints=cons, options={"maxiter": 3000, "ftol": 1e-15})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    base = sum(b.weight * np.sum(np.minimum(b.base / b.potential, 1.0) ** b.z) for b in risks)
    return -best.fun - base


@pytest.mark.parametrize("kind", ["convert", "conserve"])
def test_exact_conversion_and_conservation(natural_case, kind):
    """The exact plans of D and C: the optimum (against SLSQP), never worse than plangea's steps or the
    uniform plan; converting raises the risk (a negative benefit), conserving lowers it from the
    baseline."""
    aoh, nat = natural_case
    p = P.restoration_problem(**{("convertible" if kind == "convert" else "conservable"): nat})
    p = p.add_extinction_risk_benefit(aoh)
    total = float(p.upper @ p.area)
    q = p.add_area_target(0.3 * total, sense=">=" if kind == "convert" else "<=")
    sol = q.solve()
    assert sol.status == "optimal" and sol.bound >= sol.benefit - 1e-12
    assert sol.benefit == pytest.approx(_scipy_general(q), rel=1e-6, abs=1e-9)
    assert sol.benefit >= q.solve(method="steps", steps=3).benefit - 1e-12
    assert sol.benefit >= q.uniform().benefit - 1e-12
    assert sol.areas == {kind: pytest.approx(sol.area)}
    m = sol.converted if kind == "convert" else sol.conserved
    assert sol.restored is None and np.nansum(m.values) == pytest.approx(sol.values.sum())
    rr = P.eval_risk_reduction(q, sol)
    assert rr.total == pytest.approx(sol.benefit, rel=1e-12)
    if kind == "convert":
        assert sol.benefit < 0 and (rr.table["baseline"] == rr.table["current"]).all()
    else:
        assert sol.benefit > 0 and (rr.table["baseline"] <= rr.table["current"] + 1e-12).all()
        # the baseline loses every species' habitat in the natural land of the units
        b = q.benefits[0]
        np.testing.assert_allclose(b.current - b.base, b.gain @ q.natural, rtol=1e-12)


def test_restoration_with_conversion(case, natural_case):
    """R + D: separate columns for the anthropic land restored and the natural land converted; a demand
    for land met with no net loss of natural land; restoring gains (original vegetation), converting
    loses (current natural cover)."""
    _, veg, _, _ = case
    aoh, nat = natural_case
    p = P.restoration_problem(veg, convertible=nat).add_extinction_risk_benefit(aoh)
    assert p.actions == ("restore", "convert")
    nr = int(np.sum(p.action == 0))
    g = p.benefits[0].gain.toarray()
    assert (g[:, :nr] >= 0).all() and (g[:, nr:] <= 0).all()
    q = p.add_area_target(5.0, sense=">=", action="convert").add_area_target(0.0, sense=">=")  # net >= 0
    sol = q.solve()
    assert sol.status == "optimal"
    assert sol.areas["convert"] >= 5.0 * (1 - 1e-9)
    assert sol.areas["restore"] >= sol.areas["convert"] * (1 - 1e-9)
    assert sol.benefit == pytest.approx(_scipy_general(q), rel=1e-6, abs=1e-9)
    ev = q.evaluate(restored=sol.restored, converted=sol.converted)
    assert ev.benefit == pytest.approx(sol.benefit, rel=1e-12)
    np.testing.assert_array_equal(q.evaluate(sol.values).values, sol.values)
    with pytest.raises(ValueError, match="several actions"):
        q.evaluate(sol.restored)
    with pytest.raises(ValueError, match="one action"):
        q.uniform(5.0)


def test_actions_values_costs_and_errors(case, natural_case):
    _, veg, aoh_r, _ = case
    aoh, nat = natural_case
    rng = np.random.default_rng(8)
    carbon = pc.grid(rng.random(pc.SHAPE) * 50)
    p = P.restoration_problem(veg, convertible=nat, limit={"convert": 0.5})
    d, r = p.action == 2, p.action == 0
    np.testing.assert_allclose(p.upper[d], 0.5 * nat.values.ravel()[p.index[d]])
    q = p.add_linear_benefit({"restore": carbon, "convert": -2.0}, name="carbon")
    lin = q.benefits[0]
    np.testing.assert_array_equal(lin.coef[d], -2.0 * p.area[d])
    np.testing.assert_array_equal(lin.coef[r], carbon.values.ravel()[p.index[r]] * p.area[r])
    c = q.add_budget(10.0, {"restore": 3.0})
    assert (c.cost[d] == 0).all() and (c.cost[r] == 3.0 * p.area[r]).all()
    # conservation's values count what conserving saves from the threat
    t = 0.4
    cons = P.restoration_problem(conservable=nat, threat=t).add_linear_benefit(1.0)
    np.testing.assert_allclose(cons.benefits[0].coef, t * cons.area)
    b = P.restoration_problem(conservable=nat, threat=t).add_extinction_risk_benefit(aoh).benefits[0]
    full = P.restoration_problem(conservable=nat).add_extinction_risk_benefit(aoh).benefits[0]
    np.testing.assert_allclose(b.gain.toarray(), t * full.gain.toarray(), rtol=1e-15)
    np.testing.assert_allclose(b.current - b.base, t * (full.current - full.base), rtol=1e-12)
    with pytest.raises(ValueError, match="either conserved or converted"):
        P.restoration_problem(conservable=nat, convertible=nat)
    with pytest.raises(ValueError, match="loss"):
        P.restoration_problem(convertible=nat).add_extinction_risk_benefit(aoh_r)
    with pytest.raises(ValueError, match="no action"):
        P.restoration_problem(veg).add_linear_benefit({"convert": 1.0})
    with pytest.raises(ValueError, match="no 'convert'"):
        P.restoration_problem(veg).add_area_target(1.0, action="convert")
    with pytest.raises(ValueError, match="threat"):
        P.restoration_problem(conservable=nat, threat=1.5)
    with pytest.raises(ValueError, match="increase"):
        P.restoration_problem(convertible=nat).add_intactness_benefit(np.ones(pc.SHAPE), cell_size=1.0)
    with pytest.raises(ValueError, match="give the land"):
        P.restoration_problem()


@pytest.mark.parametrize("kind", ["convert", "conserve"])
def test_intactness_of_conversion_and_conservation(kind):
    """The linear intactness benefit of converting (the footprint rises by increase x) and conserving
    (the baseline loses the natural land by increase x threat; conserving spares it) is the derivative
    of the total intactness."""
    rng = np.random.default_rng(9)
    shape = (7, 8)
    share = rng.random(shape) * 0.8
    hfi = 2.0 + rng.random(shape) * 10
    inc = 1.0 + rng.random(shape) * 3
    threat = 0.6
    kw = {"convertible": share} if kind == "convert" else {"conservable": share, "threat": threat}
    p = P.restoration_problem(cell_area=1.0, **kw).add_intactness_benefit(hfi, increase=inc, radius=2, cell_size=1.0)
    coef = p.benefits[0].coef

    def total(x):
        h = hfi.ravel().copy()
        if kind == "convert":
            h[p.index] += inc.ravel()[p.index] * x
        else:
            h[p.index] += inc.ravel()[p.index] * threat * (share.ravel()[p.index] - x)
        return float(ls.intactness(h.reshape(shape), 2, cell_size=1.0).values.sum())

    for k in (0, 5, p.n - 1):
        e = np.zeros(p.n)
        e[k] = 1e-5
        assert coef[k] == pytest.approx((total(e) - total(-e)) / 2e-5, rel=1e-5)
    assert (coef < 0).all() if kind == "convert" else (coef > 0).all()


def test_ecosystem_areas_loss(case):
    cover, _, _, _ = case
    ea = P.ecosystem_areas(cover, pc.NATURAL, pc.ANTHROPIC, pc.grid(pc.ecoregion_codes()), cell_area=1.0)
    nat = sum(np.nan_to_num(cover.values[k]) for k in range(3)).ravel()
    codes = pc.ecoregion_codes().ravel()
    for k, code in enumerate(np.unique(codes)):
        row = ea.loss[k].toarray().ravel()
        np.testing.assert_array_equal(row, np.where((codes == code) & (nat > 0), 1.0, 0.0))
    p = P.restoration_problem(convertible=pc.grid(nat.reshape(pc.SHAPE)), cell_area=1.0)
    b = p.add_extinction_risk_benefit(ea).benefits[0]
    x = np.full(p.n, 0.1)
    expected = [np.sum(0.1 * ((codes == c) & (nat > 0))) for c in np.unique(codes)]
    np.testing.assert_allclose(b.gain @ x, -np.array(expected))


# ----------------------------------------------------------------------------- targets per region

def _regions():
    return pc.grid(np.where(np.arange(pc.SHAPE[1])[None, :] < 5, 1.0, 2.0) * np.ones(pc.SHAPE))


@pytest.mark.parametrize("name", list(pc.ESYS_CASES))
def test_ecosystem_targets_steps_are_plangea(ref, case, name):
    """harmonize_sr_ecosys (on Gurobi): the target shared among the ecoregions in steps; the exact
    shares give the most total of ((N + a) / T)^z (the KKT conditions: one marginal value among the
    ecosystems off their bounds)."""
    arrays, _ = ref
    cover, *_ = case
    spec = pc.ESYS_CASES[name]
    ea = P.ecosystem_areas(cover, pc.NATURAL, pc.ANTHROPIC, pc.grid(pc.ecoregion_codes()))
    steps = P.ecosystem_targets(ea, spec["target"], minimum=spec["minimum"], half_life=spec["hl"], method="steps",
                                steps=spec["steps"])
    np.testing.assert_array_equal(steps["region"], arrays[f"{name}/codes"])
    np.testing.assert_allclose(steps["target"], arrays[f"{name}/targets"], rtol=1e-12)
    exact = P.ecosystem_targets(ea, spec["target"], minimum=spec["minimum"], half_life=spec["hl"])
    assert exact["target"].sum() == pytest.approx(spec["target"], rel=1e-12)
    t = ea.table
    n, pot = t["current"].to_numpy(), t["potential"].to_numpy()
    least = np.minimum(spec["target"] / 3 * spec["minimum"], pot - n)
    a = exact["target"].to_numpy()
    assert (a >= least - 1e-12).all() and (a <= pot - n + 1e-12).all()
    r = (n + a) / pot
    marginal = 0.25 * r ** -0.75 * np.exp(spec["hl"] * r) / pot
    free = (a > least + 1e-9) & (a < pot - n - 1e-9)
    assert np.ptp(marginal[free]) <= 1e-7 * marginal[free].max()
    assert (marginal[a <= least + 1e-9] <= marginal[free].min() * (1 + 1e-7)).all()
    if spec["hl"] == 0:
        def total(x):
            return np.sum(((n + x) / pot) ** 0.25)

        assert total(a) >= total(steps["target"].to_numpy()) - 1e-12
    # the table is a target per region of the ecoregion raster
    veg = case[1]
    q = P.restoration_problem(veg).add_area_target(exact, regions=pc.grid(pc.ecoregion_codes()))
    assert q.targets[0].rhs.tolist() == a.tolist() and q.targets[0].sense == ("<=",) * 3


@pytest.mark.parametrize("name", list(pc.ESYS_CASES))
def test_projected_change_compensates_like_plangea(ref, case, name):
    """A plan that only restores takes the land demanded elsewhere from the restoration
    (harmonize_sr_sparable); a net demand leaves nothing to restore (plangea: negative targets)."""
    arrays, _ = ref
    veg = case[1]
    change = dict(zip([1.0, 2.0, 3.0], pc.ESYS_CASES[name]["change"]))
    q = P.restoration_problem(veg).add_projected_change(change, pc.grid(pc.ecoregion_codes()), sense="<=")
    np.testing.assert_allclose(q.targets[0].rhs, np.maximum(arrays[f"{name}/sparable"], 0.0), rtol=1e-15)
    plain = P.restoration_problem(veg).add_projected_change(change, pc.grid(pc.ecoregion_codes()), compensate=False)
    np.testing.assert_array_equal(plain.targets[0].rhs, np.maximum(list(change.values()), 0.0))


def test_projected_change_with_restoration_and_conversion(case, natural_case):
    """R + D: every region restores where land is spared and converts where it is demanded, its net
    change as projected; a plan that only converts meets the demands."""
    veg = case[1]
    aoh, nat = natural_case
    regions = _regions()
    change = {1.0: 4.0, 2.0: -3.0}
    p = P.restoration_problem(veg, convertible=nat).add_extinction_risk_benefit(aoh)
    q = p.add_projected_change(change, regions)
    reg = regions.values.ravel()[q.index]
    r_col, d_col = q.action == 0, q.action == 2
    assert (q.upper[(reg == 1) & d_col] == 0).all() and (q.upper[(reg == 2) & r_col] == 0).all()
    assert (q.upper[(reg == 1) & r_col] > 0).any() and (q.upper[(reg == 2) & d_col] > 0).any()
    sol = q.solve()
    assert sol.status == "optimal"
    net = np.where(r_col, 1.0, -1.0) * q.area * sol.values
    assert net[reg == 1].sum() == pytest.approx(4.0, rel=1e-9)
    assert net[reg == 2].sum() == pytest.approx(-3.0, rel=1e-9)
    free = p.add_projected_change(change, regions, restrict=False)
    assert np.array_equal(free.upper, p.upper)
    d = P.restoration_problem(convertible=nat).add_extinction_risk_benefit(aoh).add_projected_change(change, regions)
    assert d.targets[0].rhs.tolist() == [0.0, 3.0]
    sol = d.solve()
    reg = regions.values.ravel()[d.index]
    assert np.sum(d.area[reg == 2] * sol.values[reg == 2]) == pytest.approx(3.0, rel=1e-9)
    with pytest.raises(ValueError, match="only restore"):
        p.add_projected_change(change, regions, compensate=True)
    with pytest.raises(ValueError, match="restore or convert"):
        P.restoration_problem(conservable=nat).add_projected_change(change, regions)


def test_relative_and_linear_targets(case):
    """Targets as shares of what can be acted on (plangea's flat targets), and on other values
    (production, opportunity cost) per region."""
    veg = case[1]
    regions = _regions()
    p = P.restoration_problem(veg)
    reg = regions.values.ravel()[p.index]
    q = p.add_area_target({1.0: 0.25, 2.0: 0.5}, regions=regions, relative=True)
    avail = [float(np.sum((p.upper * p.area)[reg == c])) for c in (1.0, 2.0)]
    np.testing.assert_allclose(q.targets[0].rhs, [0.25 * avail[0], 0.5 * avail[1]], rtol=1e-15)
    land = p.add_area_target(0.3, relative=True)
    assert land.targets[0].rhs[0] == pytest.approx(0.3 * float(p.upper @ p.area), rel=1e-15)
    rng = np.random.default_rng(10)
    prod = pc.grid(rng.random(pc.SHAPE) * 5)
    carbon = pc.grid(rng.random(pc.SHAPE))
    t = (p.add_linear_benefit(carbon).add_linear_target(prod, 6.0, sense="<=")
         .add_linear_target(prod, {1.0: 1.0}, sense=">=", regions=regions))
    sol = t.solve()
    pv = prod.values.ravel()[p.index] * p.area * sol.values
    assert pv.sum() <= 6.0 * (1 + 1e-9) and pv[reg == 1].sum() >= 1.0 * (1 - 1e-9)
    assert sol.status == "optimal"


# ----------------------------------------------------------------------------- trade-off curves and weights

@pytest.mark.parametrize("name", list(pc.TRADEOFF_CASES))
def test_tradeoff_steps_is_plangea(ref, case, name):
    """scen's gradient steps (the landscape target in levels, each in refresh steps, the habitat refreshed
    from the whole plan so far): the plan of every level; the gradient counts the levels that act on a
    cell (plangea's post_gradient_output)."""
    arrays, _ = ref
    *_, p = case
    spec = pc.TRADEOFF_CASES[name]
    q = p.add_area_target(float(arrays[f"{name}/target"][0]))
    levels = [(k + 1) / spec["levels"] for k in range(spec["levels"])]
    curve = q.tradeoff(levels, method="steps", steps=spec["steps"])
    # units with the same gains and area tie in every linear program: their sums are compared
    g = q.benefits[0].gain.tocsc()
    keys = [g[:, j].toarray().tobytes() + q.area[j].tobytes() for j in range(q.n)]
    groups = {}
    for j, key in enumerate(keys):
        groups.setdefault(key, []).append(j)
    for k, s in enumerate(curve.solutions):
        theirs = arrays[f"{name}/x_{k + 1}"]
        for cols in groups.values():
            np.testing.assert_allclose(s.values[cols].sum(), theirs[cols].sum(), rtol=0, atol=1e-12)
            if len(cols) == 1:
                np.testing.assert_allclose(s.values[cols], theirs[cols], rtol=0, atol=1e-12)
    count = np.sum([s.values > 0 for s in curve.solutions], axis=0)
    np.testing.assert_array_equal(curve.gradient.values.ravel()[q.index], count)
    assert curve.table["level"].tolist() == levels
    np.testing.assert_allclose(curve.table["area"], np.array(levels) * curve.table["target"].iloc[-1], rtol=1e-9)
    if spec["steps"] == 1:  # a single solve in steps is the last level
        np.testing.assert_allclose(q.solve(method="steps").values, curve.solutions[-1].values, atol=1e-12)


def test_tradeoff_exact_nested_and_frontier(case):
    """Nested: every plan contains the previous (a priority order) and is the best given it; the
    frontier's plans are each the optimum, at least as good as the nested ones, the benefit concave in
    the target; the gradient of conversion counts down."""
    *_, p = case
    q = p.add_area_target(0.5 * float(p.upper @ p.area))
    levels = [0.2, 0.4, 0.7, 1.0]
    nested = q.tradeoff(levels)
    front = q.tradeoff(levels, nested=False)
    for a, b in zip(nested.solutions[:-1], nested.solutions[1:]):
        assert (b.values >= a.values - 1e-9).all()
    for k, s in enumerate(front.solutions):
        own = replace_target(q, levels[k]).solve()
        assert s.benefit == pytest.approx(own.benefit, rel=1e-9)
        assert s.benefit >= nested.solutions[k].benefit - 1e-9
        assert s.status == "optimal" and nested.solutions[k].status == "optimal"
    assert nested.solutions[0].benefit == pytest.approx(front.solutions[0].benefit, rel=1e-9)
    b = front.table["benefit"].to_numpy()
    t = front.table["target"].to_numpy()
    slopes = np.diff(np.concatenate([[0.0], b])) / np.diff(np.concatenate([[0.0], t]))
    assert (np.diff(slopes) <= 1e-9).all()
    g = nested.gradient.values.ravel()[q.index]
    first = nested.solutions[0].values > 0
    assert (g[first] == len(levels)).all() and set(np.unique(g)) <= set(range(len(levels) + 1))
    assert any(nested.best() is s for s in nested.solutions)
    with pytest.raises(ValueError, match="increasing"):
        q.tradeoff([0.5, 0.2])
    with pytest.raises(ValueError, match="to grow"):
        P.restoration_problem(case[1]).add_area_target({1.0: 2.0}, regions=_regions()).tradeoff()


def replace_target(q, level):
    from dataclasses import replace

    return replace(q, targets=tuple(replace(t, rhs=t.rhs * level) for t in q.targets))


def test_tradeoff_conversion_gradient(natural_case):
    aoh, nat = natural_case
    q = P.restoration_problem(convertible=nat).add_extinction_risk_benefit(aoh)
    q = q.add_area_target(0.4 * float(q.upper @ q.area), sense=">=")
    curve = q.tradeoff([0.5, 1.0])
    g = curve.gradients["convert"].values.ravel()[q.index]
    first = curve.solutions[0].values > 0
    assert (g[first] == 1).all() and (g[~(curve.solutions[1].values > 0)] == 3).all()


def test_compare_weights_ranks_by_mean_over_sd(case):
    """plangea's post_choose_weights: the values rescaled over the sets, the cost inverted, mean / sd."""
    *_, p = case
    rng = np.random.default_rng(11)
    carbon = pc.grid(rng.random(pc.SHAPE) * 100)
    cost = rng.random(p.n) + 0.5
    q = p.add_linear_benefit(carbon, name="carbon").add_area_target(8.0).add_budget(1e9, cost)
    sets = [{"extinction_risk": 1.0, "carbon": 0.0}, {"extinction_risk": 1.0, "carbon": 1e-3},
            {"extinction_risk": 0.0, "carbon": 1.0}]
    cmp_ = q.compare_weights(sets)
    t = cmp_.table
    assert list(t.columns[:2]) == ["w.extinction_risk", "w.carbon"]
    d = t[["extinction_risk", "carbon", "cost"]].to_numpy()
    v = (d - d.min(axis=0)) / np.ptp(d, axis=0)
    v[:, 2] = 1 - v[:, 2]
    np.testing.assert_allclose(t["score"], v.mean(axis=1) / v.std(axis=1, ddof=1), rtol=1e-12)
    assert cmp_.best() is cmp_.solutions[int(np.argmin(t["rank"]))]
    assert t.loc[0, "extinction_risk"] >= t.loc[2, "extinction_risk"] and t.loc[2, "carbon"] >= t.loc[0, "carbon"]
    with pytest.raises(ValueError, match="no benefit"):
        q.compare_weights([{"water": 1.0}])


# ----------------------------------------------------------------------------- exact intactness

def _intactness_problem(shape=(8, 9), seed=12, low=3.0, high=20.0, red=(1.0, 6.0), kind="restorable"):
    rng = np.random.default_rng(seed)
    share = rng.random(shape) * 0.8
    hfi = low + rng.random(shape) * (high - low)
    a = red[0] + rng.random(shape) * (red[1] - red[0])
    p = P.restoration_problem(**{kind: share}, cell_area=1.0)
    kw = {"reduction": a} if kind == "restorable" else {"increase": a}
    return p, hfi, a, kw


def test_intactness_minorant_and_secant_bound():
    """The tangent minorant never overestimates the total intactness and touches it at its point; the
    secant majorant (C++) never underestimates it over the plans' box."""
    from landschaft import _core

    p, hfi, _, kw = _intactness_problem(low=0.0, red=(2.0, 30.0))  # footprints that can reach 0
    q = p.add_intactness_benefit(hfi, radius=2, cell_size=1.0, form="exact", **kw)
    b = q.benefits[0]
    fp = b.intactness
    cells, amat, lo, hi = R._reduction_terms(q, fp)
    h0 = fp.h0
    tlo, thi = np.zeros(h0.size), np.zeros(h0.size)
    tlo[cells], thi[cells] = np.minimum(lo, h0[cells]), np.minimum(hi, h0[cells])
    u0, sigma = _core.rest_intactness_secant(h0.reshape(q.shape), tlo.reshape(q.shape), thi.reshape(q.shape),
                                             fp.gamma, fp.beta, fp.z, fp.radius, 1.0, 1.0, 2)
    sigma = np.asarray(sigma).ravel()
    rng = np.random.default_rng(13)
    x0 = rng.random(q.n) * q.upper
    t0 = np.minimum(amat @ x0, h0[cells])
    g = -ls.intactness_gradient(h0.reshape(q.shape) - np.bincount(cells, t0, h0.size).reshape(q.shape),
                                -np.ones(q.shape), 2, cell_size=1.0).values.ravel()
    T0 = R._intactness_total(q, fp, x0)
    for _ in range(20):
        x = rng.random(q.n) * q.upper
        t = np.minimum(amat @ x, h0[cells])
        T = R._intactness_total(q, fp, x)
        assert T >= T0 + float(g[cells] @ (t - t0)) - 1e-9 * T
        assert T <= u0 + float(sigma[cells] @ t) + 1e-9 * T
    assert R._intactness_total(q, fp, x0) == pytest.approx(T0)
    assert (sigma >= 0).all()


def test_exact_intactness_against_the_vertices():
    """A small problem whose only benefit is the exact intactness: its optimum is at a vertex of the set
    of plans (a convex objective), all enumerated; the plan found is at most it, the bound at least it,
    and at least the plan of the linearised benefit."""
    from itertools import product

    p, hfi, _, kw = _intactness_problem(shape=(3, 4), seed=14, low=1.0, high=8.0, red=(0.5, 3.0))
    target = 0.4 * float(p.upper @ p.area)
    q = p.add_intactness_benefit(hfi, radius=1, cell_size=1.0, form="exact", **kw).add_area_target(target)
    sol = q.solve()
    assert sol.status in ("optimal", "local") and sol.method == "exact"
    assert sol.area <= target * (1 + 1e-9)
    t0 = q.benefits[0].t0
    best = -np.inf
    n = q.n
    for k in range(n):  # one share between its bounds, the others at theirs
        others = [j for j in range(n) if j != k]
        for bits in product((0.0, 1.0), repeat=n - 1):
            x = np.zeros(n)
            x[others] = np.array(bits) * q.upper[others]
            room = target - float(q.area[others] @ x[others])
            if room < -1e-12:
                continue
            x[k] = min(q.upper[k], room / q.area[k])
            best = max(best, R._intactness_total(q, q.benefits[0].intactness, x) - t0)
    assert sol.benefit <= best + 1e-9
    assert sol.bound >= best - 1e-9
    assert sol.benefit >= 0.97 * best
    lin = p.add_intactness_benefit(hfi, radius=1, cell_size=1.0, **kw).add_area_target(target).solve()
    assert sol.benefit >= q.evaluate(lin.values).benefit - 1e-9
    assert q.evaluate(sol.values).benefit == pytest.approx(sol.benefit, rel=1e-12)
    assert sol.benefits["kind"].tolist() == ["intactness"]


def test_exact_intactness_with_other_benefits_and_conversion(case):
    """With the extinction risk too (the minorant's steps solve both exactly), and for conversion (the
    footprint rises: the benefit is negative and the least loss is sought)."""
    *_, p = case
    rng = np.random.default_rng(15)
    hfi = pc.grid(5.0 + rng.random(pc.SHAPE) * 10)
    red = pc.grid(1.0 + rng.random(pc.SHAPE) * 4)
    q = p.add_intactness_benefit(hfi, red, radius=2, weight=0.01, form="exact").add_area_target(6.0)
    sol = q.solve()
    assert sol.status in ("optimal", "local") and sol.bound >= sol.benefit - 1e-12
    lin = p.add_intactness_benefit(hfi, red, radius=2, weight=0.01).add_area_target(6.0).solve()
    assert sol.benefit >= q.evaluate(lin.values).benefit - 1e-9
    d, hfi_d, _, kw = _intactness_problem(kind="convertible", seed=16)
    qd = d.add_intactness_benefit(hfi_d, radius=2, cell_size=1.0, form="exact", **kw)
    qd = qd.add_area_target(0.3 * float(d.upper @ d.area), sense=">=")
    sd = qd.solve()
    assert sd.benefit < 0 and sd.bound >= sd.benefit - 1e-12
    unif = qd.uniform()
    assert sd.benefit >= unif.benefit - 1e-9
    with pytest.raises(ValueError, match="form"):
        p.add_intactness_benefit(hfi, red, form="cubic")
    with pytest.raises(ValueError, match="weight"):
        p.add_intactness_benefit(hfi, red, weight=-1.0, form="exact")


# ----------------------------------------------------------------------------- the solver's pieces at scale

@pytest.mark.parametrize("sense", ["<=", ">=", "="])
def test_knapsack_vertex_is_the_linear_program(sense):
    """The vertex of plans with one row (C++ weighted selection) has the linear program's value, with
    least shares, free units (a = 0), ties and units worth nothing or less."""
    from landschaft import _core
    from landschaft.prioritize._milp import OptimizationProblem

    rng = np.random.default_rng(16)
    n = 3000
    c = rng.normal(size=n)
    c[::7] = c[3]  # ties
    a = rng.random(n) * 2
    a[::50] = 0.0
    lb = np.where(rng.random(n) < 0.1, rng.random(n) * 0.2, 0.0)
    ub = lb + rng.random(n)
    cap = 0.3 * float(a @ ub)
    ok, x = _core.rest_knapsack(c, a, lb, ub, cap, sense)
    assert ok and (x >= lb - 1e-15).all() and (x <= ub + 1e-15).all()
    row = float(a @ x)
    assert {"<=": row <= cap * (1 + 1e-12), ">=": row >= cap * (1 - 1e-12), "=": abs(row - cap) <= 1e-9 * cap}[sense]
    m = OptimizationProblem(A=sp.csr_matrix(a[None, :]), obj=c, lb=lb, ub=ub, rhs=np.array([cap]), sense=(sense,),
                            vtype=("C",) * n, modelsense="max", row_ids=("r",), col_ids=("x",) * n)
    r = R._solve_lp(P.restoration_problem(np.ones((1, 1)), cell_area=1.0), m, 60.0)
    assert float(c @ x) == pytest.approx(r["objective"], rel=1e-10, abs=1e-10)
    ok, _ = _core.rest_knapsack(c, a, lb, ub, 10 * float(a @ ub), ">=")
    assert not ok
    ok, _ = _core.rest_knapsack(c, a, lb, ub, -1.0, "<=")
    assert not ok


def test_sparse_products_and_column_selection():
    """G x, G' y and the selection of columns (C++, in parallel) against scipy, with int32 and int64
    indices and empty rows."""
    from landschaft import _core

    rng = np.random.default_rng(17)
    g = sp.random(300, 5000, density=0.05, format="csr", random_state=1)
    g = sp.vstack([g, sp.csr_matrix((2, 5000)), g[:3]]).tocsr()
    x = rng.random(5000)
    y = rng.random(g.shape[0])
    for m in (g, sp.csr_matrix((g.data, g.indices.astype(np.int64), g.indptr.astype(np.int64)), shape=g.shape)):
        np.testing.assert_allclose(_core.rest_csr_matvec(m.indptr, m.indices, m.data, m.shape[0], x, 4), m @ x,
                                   rtol=1e-12)
        np.testing.assert_allclose(_core.rest_csr_rmatvec(m.indptr, m.indices, m.data, m.shape[0], m.shape[1], y, 4),
                                   m.T @ y, rtol=1e-12)
        rows = np.array([0, 5, 300, 301, 302, 40])
        cells = np.sort(rng.choice(5000, 700, replace=False))
        sel = R._select(m, rows, cells)
        assert (sel != m[rows][:, cells]).nnz == 0 and sel.has_sorted_indices
        assert sel.indices.dtype == np.int32
    big = sp.random(200, 20000, density=0.1, format="csr", random_state=2)  # above the parallel threshold
    xx = rng.random(20000)
    np.testing.assert_allclose(R._mv(big, xx), big @ xx, rtol=1e-12)
    yy = rng.random(200)
    np.testing.assert_allclose(R._rmv(big, yy), big.T @ yy, rtol=1e-12)
