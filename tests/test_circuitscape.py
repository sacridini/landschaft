# SPDX-License-Identifier: GPL-3.0-or-later
"""Circuitscape's raster modes (ls.circuit.circuitscape_*)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import landschaft as ls
from landschaft.circuit import circuitscape as cs

REF = Path(__file__).parent / "data" / "reference" / "circuit"
CS = REF / "circuitscape"


def _inis(scenario):
    return sorted(p.relative_to(CS).as_posix() for p in (CS / "input" / "raster" / scenario).glob("*/*.ini"))


@pytest.fixture
def in_cs(monkeypatch):
    # Circuitscape's test INI files hold paths relative to its test directory.
    monkeypatch.chdir(CS)


def test_file_types(tmp_path):
    root = CS / "input" / "raster"
    assert cs.file_type(root / "advanced" / "1" / "cellmap5x5.asc") == "aagrid"
    assert cs.file_type(root / "pairwise" / "1" / "polygons.tif") == "geotiff"
    f = tmp_path / "pts.txt"
    f.write_text("1 105 145\n2 115 125\n")
    assert cs.file_type(f) == "list"
    f.write_text("mode include\n1 2\n")
    assert cs.file_type(f) == "pairs"


def test_ascii_grids_are_read_in_double(tmp_path):
    f = tmp_path / "a.asc"
    f.write_text("ncols 2\nnrows 2\nxllcenter 0.5\nyllcenter 0.5\ncellsize 1\nNODATA_value -1\n"
                 "0.1 0.2\n-1 0.30000000000000004\n")
    a, meta = cs.read_raster(f)
    np.testing.assert_array_equal(a, [[0.1, 0.2], [-9999.0, 0.30000000000000004]])
    assert (meta.xllcorner, meta.yllcorner, meta.cellsize) == (0.0, 0.0, 1.0)


def test_advanced_policies():
    g = np.ones((1, 3))
    src = np.array([[1.0, 0.0, 2.0]])
    gnd = np.array([[1.0, np.nan, 1.0]])  # resistances: 1 S to ground at both ends (0 would be direct)
    kw = dict(habitat_is_resistance=False)
    keep = cs.circuitscape_advanced(g, src, gnd, **kw)
    # Both ends are sources and grounds: 3 A leave through two grounds of 1 S.
    np.testing.assert_allclose(keep.voltage, [[1.5 - 0.25, 1.5, 1.5 + 0.25]], rtol=1e-13)
    rmvsrc = cs.circuitscape_advanced(g, src, gnd, remove_src_or_gnd="rmvsrc", **kw)
    assert not rmvsrc.maps["solved"] and np.all(rmvsrc.voltage == 0)
    # rmvgnd drops the grounds from the sums but keeps them on the diagonal.
    rmvgnd = cs.circuitscape_advanced(g, src, gnd, remove_src_or_gnd="rmvgnd", **kw)
    assert not rmvgnd.maps["solved"]
    with pytest.raises(ValueError, match="remove_src_or_gnd"):
        cs.circuitscape_advanced(g, src, gnd, remove_src_or_gnd="none", **kw)


def test_polygons_merge_cells():
    g = np.ones((3, 5))
    g[1, 1:4] = np.nan  # a gap the region bridges
    poly = np.zeros((3, 5))
    poly[0, 1] = poly[2, 3] = 7  # far apart, one node
    src = np.zeros((3, 5))
    src[0, 0] = 1.0
    gnd = np.full((3, 5), np.nan)
    gnd[2, 4] = 0.0  # zero resistance: a direct ground
    r = cs.circuitscape_advanced(g, src, gnd, polygons=poly)
    assert r.nodemap[0, 1] == r.nodemap[2, 3]
    assert r.voltage[0, 1] == r.voltage[2, 3] and r.current[0, 1] == r.current[2, 3]
    assert r.voltage[2, 4] == 0 and r.current[2, 4] == pytest.approx(1.0, rel=1e-12)


def test_advanced_arrays_match_the_ini(in_cs):
    ini = "input/raster/advanced/1/mgVerify1.ini"
    a = cs.circuitscape_ini(ini, write_outputs=False)
    root = Path("input/raster/advanced/1")
    b = cs.circuitscape_advanced(cs.read_raster(root / "cellmap5x5.asc")[0], cs.read_raster(root / "sources5x5.asc")[0],
                                 cs.read_raster(root / "grounds5x5.asc")[0],
                                 polygons=cs.read_raster(root / "polygons5x5.asc")[0], four_neighbors=True,
                                 avg_resistances=True)
    np.testing.assert_array_equal(a.voltage, b.voltage)
    np.testing.assert_array_equal(a.current, b.current)


def test_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(CS)
    ini = tmp_path / "run.ini"
    text = (CS / "input/raster/advanced/1/mgVerify1.ini").read_text()
    ini.write_text(text.replace("output/mgVerify1.out", (tmp_path / "out" / "mg1.out").as_posix()))
    r = cs.circuitscape_ini(ini)
    v, _ = cs.read_raster(tmp_path / "out" / "mg1_voltmap.asc")
    c, _ = cs.read_raster(tmp_path / "out" / "mg1_curmap.asc")
    np.testing.assert_array_equal(v, r.voltage)
    np.testing.assert_array_equal(c, r.current)
    assert "scenario = advanced" in (tmp_path / "out" / "mg1.out").read_text()


def test_ini_errors(tmp_path):
    f = tmp_path / "a.ini"
    f.write_text("scenario = advanced\n")
    with pytest.raises(ValueError, match="habitat_file is not set"):
        cs.circuitscape_ini(f, write_outputs=False)
    f.write_text("scenario = pairwis\nhabitat_file = x\n")
    with pytest.raises(ValueError, match="scenario"):
        cs.circuitscape_ini(f, write_outputs=False)


@pytest.mark.reference
@pytest.mark.parametrize("ini", _inis("advanced"))
def test_advanced_matches_circuitscape(ini, in_cs):
    z = np.load(REF / "circuitscape.npz")
    name = Path(ini).stem
    r = cs.circuitscape_ini(ini, write_outputs=False)
    checked = 0
    for key, ours in (("voltmap.asc", r.voltage), ("curmap.asc", r.current), ("result.txt", r.voltage)):
        k = f"{name}/{key}"
        if k not in z.files:
            continue
        ref = z[k]
        np.testing.assert_array_equal(ours == 0, ref == 0)
        np.testing.assert_allclose(ours, ref, rtol=1e-9, atol=1e-11 * np.abs(ref).max())
        checked += 1
    assert checked >= 2


def test_pairwise_shortcut_matches_full_solves(in_cs):
    ini = "input/raster/pairwise/1/sgVerify1.ini"
    cfg = cs.parse_config(ini)
    cells, meta = cs.read_raster(cfg["habitat_file"])
    points = cs.read_point_map(cfg["point_file"], meta)
    poly = cs.read_raster(cfg["polygon_file"])[0]
    kw = dict(polygons=poly, four_neighbors=cfg["connect_four_neighbors_only"] == "True",
              avg_resistances=cfg["connect_using_avg_resistances"] == "True")
    short = cs.circuitscape_pairwise(cells, points, **kw)
    full = cs.circuitscape_pairwise(cells, points, cum_map=True, n_threads=1, **kw)
    np.testing.assert_allclose(full.resistances, short.resistances, rtol=1e-11, atol=1e-13)
    again = cs.circuitscape_pairwise(cells, points, cum_map=True, **kw)
    np.testing.assert_array_equal(again.resistances, full.resistances)
    np.testing.assert_array_equal(again.cum_current, full.cum_current)


def test_pairwise_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(CS)
    ini = tmp_path / "run.ini"
    text = (CS / "input/raster/pairwise/1/sgVerify1.ini").read_text()
    ini.write_text(text.replace("output/sgVerify1.out", (tmp_path / "pw.out").as_posix()))
    r = cs.circuitscape_ini(ini)
    written = np.loadtxt(tmp_path / "pw_resistances.out")
    np.testing.assert_array_equal(written, r.resistances)
    three = np.loadtxt(tmp_path / "pw_resistances_3columns.out")
    n = len(r.resistances) - 1
    assert three.shape == (n * (n - 1) // 2, 3)
    np.testing.assert_array_equal(cs.read_raster(tmp_path / "pw_cum_curmap.asc")[0], r.cum_current)
    assert (tmp_path / "pw_curmap_1_2.asc").exists()


@pytest.mark.reference
@pytest.mark.parametrize("ini", _inis("pairwise"))
def test_pairwise_matches_circuitscape(ini, in_cs):
    z = np.load(REF / "circuitscape.npz")
    name = Path(ini).stem
    r = cs.circuitscape_ini(ini, write_outputs=False)
    ref = z[f"{name}/result.txt"]
    # sgVerify17: resistances up to 3e3, where CHOLMOD's shifted singular solve is 1e-9 off.
    np.testing.assert_allclose(r.resistances, ref, rtol=1e-8, atol=1e-12)
    for key, ours in (("cum_curmap.asc", r.cum_current), ("max_curmap.asc", r.max_current)):
        if f"{name}/{key}" in z.files:
            want = z[f"{name}/{key}"]
            np.testing.assert_allclose(ours, want, rtol=1e-9, atol=1e-11 * np.abs(want).max())
    theirs = {k.split("/")[1][:-4] for k in z.files if k.startswith(name + "/")
              and k.split("/")[1].startswith(("curmap_", "voltmap_"))}
    assert set(r.maps) == theirs
    for kind in theirs:
        want = z[f"{name}/{kind}.asc"]
        np.testing.assert_allclose(r.maps[kind], want, rtol=1e-9, atol=1e-11 * max(np.abs(want).max(), 1e-300))


def test_one_to_all_api_matches_the_ini(in_cs):
    ini = "input/raster/one_to_all/1/oneToAllVerify1.ini"
    cfg = cs.parse_config(ini)
    a = cs.circuitscape_ini(ini, write_outputs=False)
    cells, meta = cs.read_raster(cfg["habitat_file"])
    b = cs.circuitscape_one_to_all(cells, cs.read_point_map(cfg["point_file"], meta),
                                   polygons=cs.read_raster(cfg["polygon_file"])[0],
                                   four_neighbors=cfg["connect_four_neighbors_only"] == "True",
                                   avg_resistances=cfg["connect_using_avg_resistances"] == "True",
                                   current_maps=True, n_threads=1)
    np.testing.assert_array_equal(a.resistances, b.resistances)
    np.testing.assert_array_equal(a.cum_current, b.cum_current)
    ours = {f"curmap_{n}" for kind, n in b.maps if kind == "current"}
    assert ours == {k for k in a.maps if k.startswith("curmap_")}
    for kind, n in b.maps:
        np.testing.assert_array_equal(b.maps[(kind, n)], a.maps[f"curmap_{n}"])


def test_one_to_all_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(CS)
    ini = tmp_path / "run.ini"
    src = CS / "input/raster/one_to_all/2/oneToAllVerify2.ini"
    text = src.read_text()
    out = cs.parse_config(src)["output_file"]
    ini.write_text(text.replace(out, (tmp_path / "ota.out").as_posix()))
    r = cs.circuitscape_ini(ini)
    np.testing.assert_array_equal(cs.read_raster(tmp_path / "ota_cum_curmap.asc")[0], r.cum_current)
    assert (tmp_path / "ota_curmap_1.asc").exists() and (tmp_path / "ota_voltmap_1.asc").exists()


@pytest.mark.reference
@pytest.mark.parametrize("ini", _inis("one_to_all") + _inis("all_to_one"))
def test_one_to_all_matches_circuitscape(ini, in_cs):
    z = np.load(REF / "circuitscape.npz")
    name = Path(ini).stem
    r = cs.circuitscape_ini(ini, write_outputs=False)
    np.testing.assert_allclose(r.resistances, z[f"{name}/result.txt"], rtol=1e-11, atol=1e-13)
    for key, ours in (("cum_curmap.asc", r.cum_current), ("max_curmap.asc", r.max_current)):
        if f"{name}/{key}" in z.files:
            want = z[f"{name}/{key}"]
            np.testing.assert_allclose(ours, want, rtol=1e-11, atol=1e-12 * np.abs(want).max())
    theirs = {k.split("/")[1][:-4] for k in z.files if k.startswith(name + "/")
              and k.split("/")[1].startswith(("curmap_", "voltmap_"))}
    assert set(r.maps) == theirs
    for kind in theirs:
        want = z[f"{name}/{kind}.asc"]
        np.testing.assert_allclose(r.maps[kind], want, rtol=1e-11, atol=1e-12 * max(np.abs(want).max(), 1e-300))


def test_large_factorisation_is_exact_and_deterministic():
    # 400 x 400 cells: an L above the size factorised by parts in parallel
    import scipy.sparse as sp

    rng = np.random.default_rng(12)
    g = rng.uniform(0.2, 5.0, (400, 400))
    src = np.full(g.shape, np.nan)
    src[3, 5] = 1.0
    src[200, 300] = 2.0
    gnd = np.full(g.shape, np.nan)
    gnd[399, 399] = 0.0
    kw = dict(habitat_is_resistance=False, four_neighbors=True)
    one = cs.circuitscape_advanced(g, src, gnd, n_threads=1, **kw)
    many = cs.circuitscape_advanced(g, src, gnd, n_threads=8, **kw)
    np.testing.assert_array_equal(one.voltage, many.voltage)
    np.testing.assert_array_equal(one.current, many.current)
    # the voltages solve the grounded Laplacian (4 neighbours, average conductances)
    ny, nx = g.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, vals = [], [], []
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        w = (g.ravel()[a.ravel()] + g.ravel()[b.ravel()]) / 2
        rows += [a.ravel(), b.ravel()]
        cols += [b.ravel(), a.ravel()]
        vals += [w, w]
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nx,) * 2).tocsr()
    L = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A
    v = one.voltage.ravel()
    b = np.nan_to_num(src).ravel()
    res = (L @ v - b)[:-1]
    assert np.abs(res).max() < 1e-9
