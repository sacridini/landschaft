# SPDX-License-Identifier: GPL-3.0-or-later
"""Parity with plangea (tools/make_reference_plangea.py): original vegetation, area of habitat,
ecological intactness and the propagation of uncertainties."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import landschaft as ls
import plangea_cases as pc

REF = Path(__file__).parent / "data" / "reference" / "plangea"


@pytest.fixture(scope="module")
def ref():
    with np.load(REF / "reference.npz") as z:
        arrays = {k: z[k] for k in z.files}
    texts = json.loads((REF / "reference.json").read_text(encoding="utf-8"))["texts"]
    return arrays, texts


def _vegetation(spec: dict):
    cover = pc.grid(pc.land_cover(), names=pc.CLASSES)
    past = pc.grid(pc.past(), names=pc.NATURAL) if spec["past"] else None
    er = pc.grid(pc.ecoregions(), names=pc.NATURAL) if spec["er"] else None
    restore_to = "current" if "D" in spec["problem"] else "original"
    return cover, ls.original_vegetation(cover, pc.NATURAL, pc.ANTHROPIC, past=past, ecoregions=er,
                                         failsafe=spec["failsafe"], restore_to=restore_to)


@pytest.mark.parametrize("name", list(pc.OA_CASES))
def test_original_vegetation_is_plangea(ref, name):
    arrays, _ = ref
    _, veg = _vegetation(pc.OA_CASES[name])
    cells = arrays[f"{name}/master_index"].astype(np.int64) - 1
    for k, c in enumerate(pc.NATURAL):
        np.testing.assert_array_equal(veg.original.values[k].ravel()[cells], arrays[f"{name}/oa/{c}"], err_msg=c)
        np.testing.assert_array_equal(veg.restore.values[k].ravel()[cells], arrays[f"{name}/restore/{c}"], err_msg=c)


def test_original_vegetation_sources():
    _, veg = _vegetation(pc.OA_CASES["oa_past_er"])
    s = veg.source.values
    assert np.isnan(s[0, 0])          # water only: not land
    assert s[1, 1] == 1 and s[2, 2] == 1  # nearly / all converted: the ecoregion
    assert s[8, 10] == 2              # no past, no ecoregion: the failsafe
    assert set(veg.failsafe) == {"forest", "savanna"}
    assert sum(veg.failsafe.values()) == pytest.approx(1.0)
    restore = veg.restore.values
    land = ~np.isnan(s)
    np.testing.assert_allclose(restore.sum(axis=0)[land], 1.0, atol=1e-15)


@pytest.mark.parametrize("name", list(pc.BD_CASES))
def test_area_of_habitat_is_plangea(ref, name):
    arrays, texts = ref
    spec = pc.BD_CASES[name]
    cover, veg = _vegetation(pc.OA_CASES[spec["oa"]])
    names, r = pc.ranges()
    kw = {}
    if spec["elevation"]:
        kw = {"elevation": pc.grid(pc.elevation()), "elevation_limits": pc.limits_table()}
    aoh = ls.sdm.area_of_habitat(pc.grid(r, names=names), cover, pc.NATURAL, suitability=pc.habitat_table(),
                                 original=veg, include_unlisted=spec["include_absent"], **kw)
    valid = texts[f"{name}/species"]
    t = aoh.table.set_index("species")
    assert sorted(t.index[t["valid"]]) == sorted(valid)
    np.testing.assert_array_equal(t.loc[valid, "current"].to_numpy(), arrays[f"{name}/hab_now"])
    np.testing.assert_array_equal(t.loc[valid, "potential"].to_numpy(), arrays[f"{name}/hab_pot"])
    rows = {s: i for i, s in enumerate(aoh.table["species"])}
    for s in valid:
        expected = np.zeros(aoh.gain.shape[1])
        expected[arrays[f"{name}/gain_index/{s}"].astype(np.int64) - 1] = arrays[f"{name}/gain/{s}"]
        np.testing.assert_array_equal(aoh.gain[rows[s]].toarray().ravel(), expected, err_msg=s)


def test_area_of_habitat_flags():
    cover, veg = _vegetation(pc.OA_CASES["oa_past_er"])
    names, r = pc.ranges()
    aoh = ls.sdm.area_of_habitat(pc.grid(r, names=names), cover, pc.NATURAL, suitability=pc.habitat_table(),
                                 original=veg)
    t = aoh.table.set_index("species")
    assert not t.loc["sp_c", "valid"]   # only crops: no natural class
    assert not t.loc["sp_f", "valid"]   # only water: no habitat
    assert t.loc["sp_e", "classes"] == ", ".join(pc.NATURAL)  # unlisted: every natural class
    m = aoh.gain_map("sp_a")
    assert m.shape == pc.SHAPE and float(m.sum()) == pytest.approx(aoh.gain[0].sum())


def _footprint(spec):
    h = pc.footprint(spec["shape"], spec["seed"], spec.get("low", 0.0), nan=not spec.get("gradient"))
    if spec.get("gradient"):
        h[3, 4] = np.nan
    return h


@pytest.mark.parametrize("name", list(pc.IT_CASES))
@pytest.mark.external_reference
def test_intactness_is_plangea(ref, name):
    arrays, _ = ref
    spec = pc.IT_CASES[name]
    h = _footprint(spec)
    hfi = pc.grid(h, res=spec["res"])
    kw = {}
    if spec.get("restore"):
        conv = pc.conversion(spec["shape"], spec["seed"] + 100)
        x = np.zeros(spec["shape"]) if spec.get("gradient") else pc.restored(spec["shape"], spec["seed"] + 200)
        kw = {"reduction": pc.reduction(conv), "restored": x}
    q = ls.intactness(hfi, spec["radius"], gamma=spec["gamma"], beta=spec["beta"], z=spec["z"], **kw).values
    cells = arrays[f"{name}/master_index"].astype(np.int64) - 1
    np.testing.assert_array_equal(q.ravel()[cells], arrays[f"{name}/fixed"])
    # plangea itself, away from the edges (it skips radius + 1 rows and columns at the end)
    ny, nx = h.shape
    r = spec["radius"]
    inner = np.zeros(h.shape, dtype=bool)
    inner[r:ny - r - 1, r:nx - r - 1] = True
    orig = np.full(h.size, np.nan)
    orig[cells] = arrays[f"{name}/original"]
    sel = inner.ravel() & ~np.isnan(h.ravel())
    np.testing.assert_array_equal(q.ravel()[sel], orig[sel])


def test_intactness_gradient_is_the_derivative_of_plangea_total(ref):
    arrays, _ = ref
    spec = pc.IT_CASES["it_gradient"]
    h = _footprint(spec)
    conv = pc.conversion(spec["shape"], spec["seed"] + 100)
    g = ls.intactness_gradient(pc.grid(h), pc.reduction(conv), spec["radius"], gamma=spec["gamma"],
                               beta=spec["beta"], z=spec["z"]).values
    cells = arrays["it_gradient/master_index"].astype(np.int64) - 1
    np.testing.assert_allclose(g.ravel()[cells], arrays["it_gradient/gradient"], rtol=1e-6)


def test_uncertainty_is_plangea(ref):
    arrays, _ = ref
    u = pc.uncertainty_inputs()
    w = u["px_area"] * u["x_vals"]
    assert ls.uncertainty.sum_sd(u["v"], w, method="quadratic") == pytest.approx(arrays["unc/prop_quad"][0], rel=1e-14)
    assert ls.uncertainty.sum_sd(u["v"], w, method="linear") == pytest.approx(arrays["unc/prop_lin"][0], rel=1e-14)
    q = ls.uncertainty.product_sd(u["var_x"], u["sd_x"], u["var_v"], u["sd_v"], total=True, method="quadratic")
    lin = ls.uncertainty.product_sd(u["var_x"], u["sd_x"], u["var_v"], u["sd_v"], total=True, method="linear")
    assert q == pytest.approx(arrays["unc/var_sd_quad"][0], rel=1e-14)
    assert lin == pytest.approx(arrays["unc/var_sd_lin"][0], rel=1e-14)
    # plangea's own (unsquared cross term) differs
    assert q != pytest.approx(arrays["unc/var_sd_quad_original"][0], rel=1e-6)


def test_extinction_risk_uncertainty_is_plangea_corrected(ref):
    """The total reduction of the extinction risk and its uncertainty from z (sd 0.1), as the corrected
    multi_exrisk_aggregate: the delta method on the species' reductions, combined over species."""
    arrays, texts = ref
    cover, veg = _vegetation(pc.OA_CASES["oa_past_er"])
    names, r = pc.ranges()
    aoh = ls.sdm.area_of_habitat(pc.grid(r, names=names), cover, pc.NATURAL, suitability=pc.habitat_table(),
                                 original=veg)
    t = aoh.table.set_index("species")
    valid = texts["bd_absent/species"]
    rows = [list(aoh.table["species"]).index(s) for s in valid]
    cells = np.flatnonzero(veg.anthropic.values.ravel() > 0)
    x = np.zeros(aoh.gain.shape[1])
    x[cells] = arrays["bd_absent/x_vals"]
    gained = aoh.gain[rows] @ x
    now, pot = t.loc[valid, "current"].to_numpy(), t.loc[valid, "potential"].to_numpy()

    def reduction(z):
        z = np.atleast_1d(z)[0]
        risk = lambda a: np.maximum(1.0 - (a / pot) ** z, 0.0)  # noqa: E731
        return risk(now) - risk(now + gained)

    for method, key in (("quadratic", "quad"), ("linear", "lin")):
        val, unc = arrays[f"bd_absent/exrisk_{key}"]
        d = ls.uncertainty.delta_method(reduction, [0.25], [0.1], scheme="forward", step=[1e-6])
        assert float(np.sum(d.value)) == pytest.approx(val, rel=1e-12)
        assert ls.uncertainty.sum_sd(d.sd, method=method) == pytest.approx(unc, rel=1e-6)
