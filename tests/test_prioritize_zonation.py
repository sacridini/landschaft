# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.prioritize.zonation against Zonation 5 v2.4 (tools/make_reference_prioritize.py) and its API."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import landschaft as ls
import prioritize_cases
from prioritize_reference import case_ids, load, parse_table, run_case, table_text

TABLES = {
    "feature_curves": "feature_curves.csv",
    "summary_curves": "summary_curves.csv",
    "density_curves": "feature_density_curves.csv",
    "group_curves": "group_curves.csv",
}


@pytest.fixture(scope="module")
def rasters(tmp_path_factory):
    return tmp_path_factory.mktemp("prio_rasters")


def _run(name, rasters, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_case(name, rasters, **kw)


@pytest.mark.reference
@pytest.mark.parametrize("name", case_ids("core"))
def test_rank_matches_zonation(name, rasters):
    """The rank map, bit for bit, and the number of iterations."""
    arrays, meta = load("core")
    res = _run(name, rasters)
    ref = arrays[f"{name}/rank"]
    ours = np.asarray(res.rank)
    same = (ours == ref) | (np.isnan(ours) & np.isnan(ref))
    assert same.all(), f"{int((~same).sum())} cells differ"
    stats = meta[f"{name}/stats"]
    assert res.iterations == int(stats.get("Number of iterations", 0))
    if res.iterations:
        conv = float(stats["Final convergence level"].rstrip("%"))
        assert res.ranking_error == pytest.approx(conv, rel=1e-5, abs=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("name", [n for n in case_ids("core") if prioritize_cases.CASES[n].load])
def test_tables_match_zonation(name, rasters, tmp_path):
    """The curve tables of the returned ranking, digit for digit (Zonation's LOAD run of its rank map)."""
    arrays, meta = load("core")
    res = _run(name, rasters)
    out = res.write(tmp_path)
    for key, fname in TABLES.items():
        k = f"{name}/load/{key}"
        if k not in arrays:
            assert not (out / fname).exists() or key != "group_curves"
            continue
        ref = [ln.strip() for ln in table_text(arrays, k).splitlines()]
        ours = [ln.strip() for ln in (out / fname).read_text(encoding="utf-8").splitlines()]
        assert len(ours) == len(ref), key
        bad = [i for i, (a, b) in enumerate(zip(ours, ref)) if a != b]
        assert not bad, f"{key}: {len(bad)} lines differ, first {ours[bad[0]]!r} vs {ref[bad[0]]!r}"
    # features_info: Zonation prints the sums with 6 digits
    from landschaft.prioritize._tables import _g

    info = arrays[f"{name}/features_info"]
    cols = meta[f"{name}/features_info"]
    for col in ("number", "count", "missing"):
        assert res.features[col].tolist() == info[:, cols.index(col)].astype(int).tolist(), col
    for col in ("dsum", "quant_error"):
        assert [_g(v) for v in res.features[col]] == [_g(v) for v in info[:, cols.index(col)]], col


@pytest.mark.reference
def test_auc_close_to_zonation(rasters):
    """Zonation's AUC is of the order before the last sort; ours of the ranking returned: close."""
    arrays, meta = load("core")
    for name in ("caz2", "abf_025", "big_caz2"):
        res = _run(name, rasters)
        assert res.auc == pytest.approx(float(meta[f"{name}/stats"]["Area under curve"]), rel=2e-4)


# ---------------------------------------------------------------- inputs and API


def _stack(name):
    case = prioritize_cases.CASES[name]
    return np.stack([prioritize_cases.RASTERS[r].values for r in case.layers]).astype(np.float32)


def test_array_inputs_equal_files(rasters):
    """NumPy, xarray and dask stacks give the file result (one block of rows)."""
    import xarray as xr

    ref = _run("caz2_weights", rasters)
    w = prioritize_cases.W8
    stack = _stack("caz2_weights")
    for x in (stack, list(stack), xr.DataArray(stack, dims=("feature", "y", "x"))):
        res = ls.prioritize.zonation(x, weights=w)
        np.testing.assert_array_equal(np.asarray(res.rank), np.asarray(ref.rank))
    da = pytest.importorskip("dask.array")
    res = ls.prioritize.zonation(da.from_array(stack, chunks=(1, 7, 50)), weights=w)
    np.testing.assert_array_equal(np.asarray(res.rank), np.asarray(ref.rank))


def test_threads_do_not_change_result(rasters):
    a = _run("big_abf", rasters, n_threads=1)
    b = _run("big_abf", rasters, n_threads=4)
    np.testing.assert_array_equal(np.asarray(a.rank), np.asarray(b.rank))
    assert a.iterations == b.iterations
    np.testing.assert_array_equal(a.feature_curves.to_numpy(), b.feature_curves.to_numpy())


def test_rank_is_a_permutation_and_curves_start_full():
    rng = np.random.default_rng(3)
    x = (rng.random((4, 30, 40)) ** 2).astype(np.float32)
    x[:, :3, :5] = np.nan
    res = ls.prioritize.zonation(x, rule="ABF")
    r = np.asarray(res.rank)
    ok = np.isfinite(r)
    n = int(ok.sum())
    assert n == 30 * 40 - 15
    np.testing.assert_array_equal(np.sort(r[ok]), (np.arange(n) / n).astype(np.float32))
    fc = res.feature_curves
    assert (fc.iloc[0, 1:] == 1.0).all() and (fc.iloc[-1, 1:].abs() < 1e-12).all()  # rounding residue
    assert fc["rank"].iloc[-1] == 1.0
    # the order visits the cells by increasing rank
    assert np.all(np.diff(r.ravel()[res.order]) > 0)


def test_rand_reproducible_with_seed():
    x = np.random.default_rng(1).random((2, 10, 10))
    a = ls.prioritize.zonation(x, rule="RAND", seed=5)
    b = ls.prioritize.zonation(x, rule="RAND", seed=5)
    np.testing.assert_array_equal(np.asarray(a.rank), np.asarray(b.rank))
    assert a.iterations == 0


def test_load_evaluates_given_order():
    x = np.random.default_rng(2).random((3, 12, 15))
    sol = np.random.default_rng(4).random((12, 15))
    res = ls.prioritize.zonation(x, rule="LOAD", solution=sol)
    r = np.asarray(res.rank)
    assert np.array_equal(np.argsort(r, axis=None, kind="stable"), np.argsort(sol, axis=None, kind="stable"))


def test_empty_layer_left_out():
    x = np.random.default_rng(2).random((3, 8, 9))
    x[1] = 0.0
    with pytest.warns(UserWarning, match="left out"):
        res = ls.prioritize.zonation(x)
    assert res.features["number"].tolist() == [0, 2]


def test_rule_parameter_forms():
    x = np.random.default_rng(6).random((3, 10, 11))
    a = ls.prioritize.zonation(x, rule="ABF 0.4")
    b = ls.prioritize.zonation(x, rule="ABF", rule_param=0.4)
    np.testing.assert_array_equal(np.asarray(a.rank), np.asarray(b.rank))
    with pytest.raises(ValueError, match="unknown"):
        ls.prioritize.zonation(x, rule="XYZ")
    with pytest.raises(ValueError, match="twice"):
        ls.prioritize.zonation(x, rule="ABF 0.3", rule_param=0.2)
    with pytest.raises(ValueError, match="solution"):
        ls.prioritize.zonation(x, rule="LOAD")
    with pytest.raises(ValueError, match="bits"):
        ls.prioritize.zonation(x, bits=12)


def test_grid_mismatch_rejected():
    x = np.random.default_rng(6).random((2, 10, 11))
    with pytest.raises(ValueError, match="cost"):
        ls.prioritize.zonation(x, cost=np.ones((10, 12)))


def test_dataarray_output_keeps_grid():
    import xarray as xr
    from affine import Affine

    x = np.random.default_rng(7).random((2, 6, 8)).astype(np.float32)
    da = xr.DataArray(x, dims=("band", "y", "x"),
                      coords={"band": [1, 2], "y": 1000 - np.arange(6) * 10 - 5, "x": np.arange(8) * 10 + 5})
    da = da.rio.write_transform(Affine(10, 0, 0, 0, -10, 1000)).rio.write_crs("EPSG:32633")
    res = ls.prioritize.zonation(da)
    assert res.rank.rio.crs.to_epsg() == 32633
    np.testing.assert_array_equal(res.rank["x"].values, da["x"].values)


def test_cazp_small_exponent_runs():
    """With p < 1, 0^p is not zero in Zonation (vdt's fast_log(0) = -709.09): absent features count."""
    rng = np.random.default_rng(0)
    x = rng.random((2, 10, 10))
    # CAZP with p < 1: 0^p is not zero in Zonation (fast_log(0) = -709.09)
    res = ls.prioritize.zonation(x, rule="CAZP 0.5")
    assert res.iterations > 0
