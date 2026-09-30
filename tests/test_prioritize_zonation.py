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


# ---------------------------------------------------------------- transforms (phase 2)


@pytest.mark.reference
@pytest.mark.parametrize("name", case_ids("transform"))
def test_transforms_match_zonation(name, rasters, tmp_path):
    """Conditions, retention, pre-processing maps and transformed layers, bit for bit."""
    arrays, meta = load("transform")
    res = _run(name, rasters)
    ref = arrays[f"{name}/rank"]
    ours = np.asarray(res.rank)
    assert ((ours == ref) | (np.isnan(ours) & np.isnan(ref))).all()
    assert res.iterations == int(meta[f"{name}/stats"]["Number of iterations"])
    out = res.write(tmp_path)
    for key, fname in TABLES.items():
        k = f"{name}/load/{key}"
        if k in arrays:
            want = [ln.strip() for ln in table_text(arrays, k).splitlines()]
            got = [ln.strip() for ln in (out / fname).read_text(encoding="utf-8").splitlines()]
            assert got == want, key
    pre = {k.split("/pre/")[1]: v for k, v in arrays.items() if k.startswith(f"{name}/pre/")}
    assert set(res.preprocessing) == set(pre)
    for k, v in pre.items():
        a = np.asarray(res.preprocessing[k])
        assert ((a == v) | (np.isnan(a) & np.isnan(v))).all(), k
    number = list(res.features["number"])
    tr = {int(k.split("/tr/")[1]): v for k, v in arrays.items() if k.startswith(f"{name}/tr/")}
    assert len(res.transformed) == len(tr)
    for i, v in tr.items():
        a = np.asarray(res.transformed[res.features["name"][number.index(i)]])
        assert ((a == v) | (np.isnan(a) & np.isnan(v))).all(), i


@pytest.mark.reference
@pytest.mark.parametrize("name", list(prioritize_cases.CASES))
def test_run_project_matches_zonation(name, tmp_path):
    """The Zonation project of every case (settings, feature list, link files), read and run."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_prioritize import write_project

    case = prioritize_cases.CASES[name]
    arrays, _ = load(case.group)
    write_project(case, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ls.prioritize.run_project(tmp_path / "settings.z5", case.flags, mode=case.mode, bits=case.bits)
    ref = arrays[f"{name}/rank"]
    ours = np.asarray(res.rank)
    assert ((ours == ref) | (np.isnan(ours) & np.isnan(ref))).all()


def test_retention_forms_and_errors():
    rng = np.random.default_rng(9)
    x = rng.random((3, 12, 14)).astype(np.float32)
    frac = np.full((12, 14), 0.25, np.float32)
    a = ls.prioritize.zonation(x, retention=[0.25, None, None])
    b = ls.prioritize.zonation(x, retention=[frac, None, None])
    np.testing.assert_array_equal(np.asarray(a.rank), np.asarray(b.rank))
    # the retained part stays: the curve ends near the retained fraction
    assert 0.2 < a.feature_curves.iloc[-1, 1] < 0.3
    with pytest.raises(ValueError, match="negative"):
        ls.prioritize.zonation(x, retention=[("absolute", -frac), None, None])
    with pytest.raises(ValueError, match="0, 1"):
        ls.prioritize.zonation(x, retention=[frac * 5, None, None])
    with pytest.raises(ValueError, match="retention"):
        ls.prioritize.zonation(x, retention=[0.1])


def test_condition_checks_and_renormalize():
    rng = np.random.default_rng(10)
    x = rng.random((2, 10, 10)).astype(np.float32)
    cond = np.full((10, 10), 0.5, np.float32)
    r = ls.prioritize.zonation(x, condition=[cond, None])
    assert r.feature_curves.iloc[0, 1] == pytest.approx(0.5, abs=0.01)  # initial level: the conditioned fraction
    rn = ls.prioritize.zonation(x, condition=[cond, None], renormalize=True)
    assert rn.feature_curves.iloc[0, 1] == 1.0
    with pytest.raises(ValueError, match="0, 1"):
        ls.prioritize.zonation(x, condition=[cond * 3, None])


def test_mixed_sign_group_warns():
    x = np.random.default_rng(11).random((3, 8, 8))
    with pytest.warns(UserWarning, match="output group 1"):
        ls.prioritize.zonation(x, weights=[1, -1, 1], groups=[1, 1, 2])


# ---------------------------------------------------------------- connectivity (phase 3)


@pytest.mark.reference
@pytest.mark.parametrize("name", case_ids("connectivity"))
def test_connectivity_matches_zonation(name, rasters, tmp_path):
    """Single-feature, interaction and matrix connectivity: the smoothed layers and the ranking."""
    arrays, meta = load("connectivity")
    res = _run(name, rasters)
    ref = arrays[f"{name}/rank"]
    ours = np.asarray(res.rank)
    assert ((ours == ref) | (np.isnan(ours) & np.isnan(ref))).all()
    assert res.iterations == int(meta[f"{name}/stats"]["Number of iterations"])
    number = list(res.features["number"])
    tr = {int(k.split("/tr/")[1]): v for k, v in arrays.items() if k.startswith(f"{name}/tr/")}
    assert tr
    for i, v in tr.items():
        a = np.asarray(res.transformed[res.features["name"][number.index(i)]])
        assert ((a == v) | (np.isnan(a) & np.isnan(v))).all(), i
    out = res.write(tmp_path)
    for key, fname in TABLES.items():
        k = f"{name}/load/{key}"
        if k in arrays:
            want = [ln.strip() for ln in table_text(arrays, k).splitlines()]
            got = [ln.strip() for ln in (out / fname).read_text(encoding="utf-8").splitlines()]
            assert got == want, key


def test_kernel_sizes_and_shapes():
    K = ls.prioritize.Kernel
    assert K("negexp", 4, 3).size() == 128  # at least Z5_FILTER_MIN_SIZE
    assert K("negexp", 40, 3).size() == 256  # 2 * 2^ceil(log2(121))
    assert K("uniform", 100).size() == 256
    assert K("uniform", 100).size(max_distance=50) == 128
    img = K("uniform", 3).image()
    c = img.shape[0] // 2
    assert img[c, c] == 1 and img[c, c + 3] == 1 and img[c, c + 4] == 0
    tri = K("triangular", 4).image()
    assert tri[c, c + 2] == pytest.approx(0.5)
    g = K("gaussian", 2, 3).image()
    assert g[c, c + 7] == 0 and g[c, c + 6] > 0
    with pytest.raises(ValueError, match="kernel"):
        K("box", 3)
    with pytest.raises(ValueError, match="> 0"):
        K("negexp", 0)


def test_convolve_matches_direct_convolution():
    from scipy.signal import convolve2d

    rng = np.random.default_rng(12)
    img = rng.random((150, 170))
    k = ls.prioritize.Kernel("negexp", 3, 3)
    ours = ls.prioritize.convolve(img, k)
    ker = k.image()
    ker = ker / ker.sum()
    full = convolve2d(img, ker, mode="full")
    n = ker.shape[0]
    want = full[n // 2:n // 2 + 150, n // 2:n // 2 + 170]
    np.testing.assert_allclose(ours, want, rtol=1e-5, atol=1e-7)


def test_self_multiply_and_interaction_forms():
    rng = np.random.default_rng(13)
    x = (rng.random((3, 40, 50)) ** 2).astype(np.float32)
    res_r = rng.random((40, 50)).astype(np.float32)
    k = ls.prioritize.Kernel("gaussian", 2, 3)
    a = ls.prioritize.zonation(x, connectivity=[k, None, k], self_multiply=True, transformed=True)
    b = ls.prioritize.zonation(x, connectivity=[k, None, k], self_multiply=False, transformed=True)
    assert not np.array_equal(np.asarray(a.transformed["feature_0"]), np.asarray(b.transformed["feature_0"]))
    np.testing.assert_array_equal(np.asarray(a.transformed["feature_1"]), np.asarray(b.transformed["feature_1"]))
    ia = ls.prioritize.Interaction(k, res_r, positive=False, beta=0.5, normalize=False)
    c = ls.prioritize.zonation(x, interaction=[None, ia, None])
    assert c.feature_curves.iloc[0, 2] < 1.0  # the level is what remains
    with pytest.raises(ValueError, match="matrix"):
        ls.prioritize.zonation(x, matrix_connectivity=[(k, "M"), (k, "M"), None], matrices={"M": np.eye(3)})


# ---------------------------------------------------------------- post-processing (phase 4)


def _compare_file(ours, ref_arrays, key, name):
    import rasterio

    if name.endswith(".tif"):
        a = rasterio.open(ours).read(1)
        b = ref_arrays[key]
        return ((a == b) | (np.isnan(a) & np.isnan(b))).all()
    want = table_text(ref_arrays, key).splitlines()
    got = ours.read_text(encoding="utf-8").splitlines()
    if name.startswith("highest"):  # the file paths differ
        want = [" ".join(ln.split()[:3]) for ln in want]
        got = [" ".join(ln.split()[:3]) for ln in got]
    if name.startswith("solution"):
        want, got = want[1:], got[1:]
    return [ln.strip() for ln in got] == [ln.strip() for ln in want]


@pytest.mark.reference
@pytest.mark.parametrize("name", case_ids("post"))
def test_postprocessing_matches_zonation(name, rasters, tmp_path):
    """Area statistics, solution comparison and histograms digit for digit (Zonation's LOAD
    run: the curves of the returned ranking); flexibility maps bit for bit where Zonation's
    curves are those of its ranking, and else against a direct port of its algorithm."""
    arrays, _ = load("post")
    case = prioritize_cases.CASES[name]
    res = _run(name, rasters)
    pp = rasters / f"pp_{name}.txt"
    pp.write_text(case.files["pp.txt"], encoding="utf-8")
    files = {f.name: f for f in res.postprocess(pp, tmp_path)}
    ref = sorted(k.split("/post/")[1] for k in arrays if k.startswith(f"{name}/post/"))
    assert sorted(files) == ref
    for fname in ref:
        if fname.startswith("flexibility"):
            continue
        assert _compare_file(files[fname], arrays, f"{name}/load/post/{fname}", fname), fname
    flex = [f for f in ref if f.startswith("flexibility")]
    if name == "post_presence":  # its last sort did not change the order: Zonation's curves are ours
        for fname in flex:
            assert _compare_file(files[fname], arrays, f"{name}/post/{fname}", fname), fname
    else:
        import rasterio

        for fname in flex:  # Zonation's curves come from the order before its last sort: close
            a = rasterio.open(files[fname]).read(1)
            b = arrays[f"{name}/post/{fname}"]
            ok = np.isfinite(b)
            assert (a[ok] == b[ok]).mean() > 0.9, fname
        import prioritize_flex_reference as fr

        cells = res._ctx["cells"]
        f = res.flexibility_map(2.0)
        up = np.asarray(f["up"]).ravel()[cells]
        down = np.asarray(f["down"]).ravel()[cells]
        U, D = fr.flexibility(res, 2.0, range(0, cells.size, 11))
        assert all(U[c] == up[c] and D[c] == down[c] for c in U)


@pytest.mark.reference
def test_hierarchic_mask_ranks(rasters):
    _, meta = load("core")
    res = _run("caz2_hmask", rasters)
    stats = meta["caz2_hmask/stats"]
    from landschaft.prioritize._tables import _g

    for level, rank in res.hierarchic_mask_ranks.items():
        assert _g(rank) == stats[f"Hierarchic mask level {level} minimum rank"]


def test_run_project_postprocessing(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_prioritize import write_project

    case = prioritize_cases.CASES["post_caz2"]
    write_project(case, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls.prioritize.run_project(tmp_path / "settings.z5", case.flags, mode=case.mode, output=tmp_path / "out")
    names = {p.name for p in (tmp_path / "out" / "post_processing").iterdir()}
    assert "area_stats.pos.ext.csv" in names and "flexibilitymap.flex.tif" in names


def test_postprocessing_errors():
    res = ls.prioritize.zonation(np.random.default_rng(14).random((2, 10, 12)))
    with pytest.raises(ValueError, match="min_rank"):
        res.area_statistics_rank_range(0.9, 0.1)
    with pytest.raises(ValueError, match="0 and 1"):
        res.area_statistics_rank_range(-0.1, 0.5)
    with pytest.raises(ValueError, match="min_area_size"):
        res.area_statistics_separate_areas(0.5, 1.0, 0)
    with pytest.raises(ValueError, match="bars"):
        res.histogram(0.5, 1.0, 0)
    with pytest.raises(ValueError, match="max error"):
        res.flexibility_map(0.0)


# ---------------------------------------------------------------- impact assessment (phase 5)


@pytest.mark.reference
@pytest.mark.parametrize("name", case_ids("impact"))
def test_impacts_match_zonation(name, tmp_path):
    """Impact scenarios run from Zonation projects: every output file and the ranking."""
    import sys
    from pathlib import Path

    import rasterio

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_prioritize import write_project

    arrays, meta = load("impact")
    case = prioritize_cases.CASES[name]
    write_project(case, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ls.prioritize.run_project(tmp_path / "settings.z5", case.flags, mode=case.mode, output=tmp_path / "out")
    ref = arrays[f"{name}/rank"]
    ours = np.asarray(res.rank)
    assert ((ours == ref) | (np.isnan(ours) & np.isnan(ref))).all()
    assert res.iterations == int(meta[f"{name}/stats"]["Number of iterations"])
    want = sorted(k.split("/impacts/", 1)[1] for k in arrays if k.startswith(f"{name}/impacts/"))
    root = tmp_path / "out" / "impacts"
    got = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert got == want
    for k in want:
        if k.endswith(".tif"):
            a = rasterio.open(root / k).read(1)
            assert np.array_equal(a, arrays[f"{name}/impacts/{k}"], equal_nan=True), k
        else:
            w = [ln.strip() for ln in table_text(arrays, f"{name}/impacts/{k}").splitlines()]
            g = [ln.strip() for ln in (root / k).read_text(encoding="utf-8").splitlines()]
            assert g == w, k


def test_impact_api():
    rng = np.random.default_rng(15)
    x = (rng.random((2, 30, 40)) ** 2).astype(np.float32)
    src = np.zeros((30, 40), np.uint16)
    src[15, :] = 1
    P = ls.prioritize
    sc = P.ImpactScenario(src, [P.ImpactSource(1, 1, 100, ((3, 50),))], name="road")
    res = P.zonation(x, impact_groups=[1, -1], impact=sc)
    z = res.impacts["zones"]
    whole = z[(z["area"] == 0) & (z["bin"] == -1)]
    assert set(whole["zone"]) == {0, 1}
    foot = whole[whole["zone"] == 0].iloc[0]
    # the road's cells with a value above 0 (some quantize to 0), all lost
    assert 30 < foot["cells"] <= 40 and foot["loss"] == pytest.approx(foot["distribution"], rel=1e-6)
    assert res.feature_curves.iloc[0, 1] < 1.0  # the impacted feature starts below 1
    assert res.feature_curves.iloc[0, 2] == 1.0
    assert "ig1_zones" in res.impacts["rasters"]
    cum = P.CumulativeScenario(direct="sum", direct_rasters=[np.full((30, 40), 0.5, np.float32)],
                               direct_matrix=np.array([[1.0]]))
    res2 = P.zonation(x, impact_groups=[1, -1], cumulative_impact=cum)
    assert res2.feature_curves.iloc[0, 1] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("bits", [8, 16])
def test_row_formats_and_storage_agree(monkeypatch, bits, tmp_path):
    """Delta rows (gaps over 255 layers), bitmap rows (several mask words) and
    memory-mapped storage give the same ranking."""
    rng = np.random.default_rng(21)
    x = rng.random((300, 30, 40)).astype(np.float32)
    x[x < 0.97] = 0
    x[:, 0, :3] = 0
    x[299, 0, 0] = 0.5  # a row with a gap of 299 layers
    x[[0, 280], 0, 1] = 0.7
    got = {}
    for fmt in ("delta", "bitmap", ""):
        monkeypatch.setenv("LANDSCHAFT_PRIO_ROWS", fmt)
        res = ls.prioritize.zonation(x, bits=bits, rule="CAZ2", storage=tmp_path / "tmp" if not fmt else None)
        got[fmt] = (np.asarray(res.rank), res.iterations, res.feature_curves.to_numpy())
        del res
    for fmt in ("bitmap", ""):
        assert np.array_equal(got[fmt][0], got["delta"][0], equal_nan=True), fmt
        assert got[fmt][1] == got["delta"][1]
        assert np.array_equal(got[fmt][2], got["delta"][2])
    import gc

    gc.collect()
    assert not any((tmp_path / "tmp").iterdir())  # the temporary files are gone


@pytest.mark.reference
@pytest.mark.parametrize("name", ["caz2_16bit", "caz2_hmask_cost", "caz2_smoothing", "abf_cost_negweights"])
def test_storage_matches_zonation(name, rasters, tmp_path):
    """Out-of-core storage: the same bits as Zonation."""
    arrays, _ = load("core")
    res = _run(name, rasters, storage=tmp_path)
    assert np.array_equal(np.asarray(res.rank), arrays[f"{name}/rank"], equal_nan=True)
    assert np.isfinite(np.asarray(res.flexibility_map()["up"])).any()


@pytest.mark.parametrize("bits", [8, 16])
def test_data_runs_roundtrip(bits, tmp_path):
    """The runs of NoData, zeros and values (long runs, LEB128 lengths) keep every
    layer's values, and rewriting a layer splits the runs of zeros."""
    from landschaft import _core

    rng = np.random.default_rng(3)
    L, h, w = 300, 6, 7
    top = 255 if bits == 8 else 65535
    shape = (L, h, w)
    v = np.where(rng.random(shape) < 0.05, rng.integers(1, top + 1, size=shape), 0).astype(np.float32)
    v[rng.random(v.shape) < 0.3] = np.nan  # NoData
    v[:200, 0, 0] = np.nan  # a long run of NoData, then zeros
    v[200:, 0, 0] = 0
    v[:, 0, 1] = 0  # all zeros
    v[5, 0, 1] = 7
    for storage in ("", str(tmp_path)):
        d = _core.PrioData([np.nan] * L, [np.nan] * L, [0.0] * L, [float(top)] * L, [1] * L, bits, storage)
        d.add_block(v, None, None)
        d.finish()
        cells = np.asarray(d.cell_index)
        flat = v.reshape(L, -1)[:, cells]
        want = np.where(np.isnan(flat), -1, flat)
        got = np.stack([np.asarray(d.layer(i, scaled=False, missing=-1)) for i in range(L)])
        assert np.array_equal(got, want)
        assert d.entries == int((~np.isnan(flat)).sum())
        # values into a long run of zeros (the middle, both ends) and into NoData (ignored)
        new = np.arange(1, cells.size + 1, dtype=np.float32)
        for layer in (250, 200, 299, 0):
            d.set_layer_values(layer, new)
            want[layer] = np.where(want[layer] >= 0, new, -1)
        got = np.stack([np.asarray(d.layer(i, scaled=False, missing=-1)) for i in range(L)])
        assert np.array_equal(got, want)


def test_block_sums_inside_chunks():
    """Zonation's per-block sums, flushed inside a chunk of rows, equal adding and
    flushing block by block (bit for bit)."""
    from landschaft import _core

    rng = np.random.default_rng(8)
    L, h, w, z = 5, 23, 17, 4
    v = (rng.random((L, h, w)) ** 3).astype(np.float32)
    cond = rng.random((1, h, w)).astype(np.float32)

    def build():
        d = _core.PrioData([np.nan] * L, [np.nan] * L, [0.0] * L, [float(v[i].max()) for i in range(L)], [1] * L, 8)
        d.set_conditions([0, -1, 0, -1, -1], [0] * L)
        return d

    a = build()
    for y in range(0, h, z):
        a.add_block(v[:, y:y + z], None, cond[:, y:y + z])
        a.flush()
    a.finish()
    b = build()
    for y0, y1 in ((0, 9), (9, 10), (10, h)):  # chunks across the blocks
        b.add_block(v[:, y0:y1], None, cond[:, y0:y1], block_rows=z)
    b.finish()
    for k in ("max_norm_sums", "layer_scale", "initial_levels", "quantization_errors"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
