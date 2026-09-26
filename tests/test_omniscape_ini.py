# SPDX-License-Identifier: GPL-3.0-or-later
"""Omniscape INI files, conditional connectivity, reclassification and single precision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import landschaft as ls

pytest.importorskip("rasterio")

REF = Path(__file__).parent / "data" / "reference" / "circuit"
INI = REF / "omniscape_ini"
OUTPUTS = ("cum_currmap", "flow_potential", "normalized_cum_currmap", "classified_resistance")


def _map(seed=1, shape=(30, 40)):
    rng = np.random.default_rng(seed)
    r = np.exp(rng.normal(0.5, 1.0, shape))
    r[rng.random(shape) < 0.05] = np.nan
    return r


def test_parse_ini(tmp_path):
    f = tmp_path / "a.ini"
    f.write_text("[Input files]\nresistance_file = r.tif \nradius=5\nweird = a = b\nno equals here\n[x]\n")
    assert ls.circuit.parse_ini(f) == {"resistance_file": "r.tif", "radius": "5", "weird": "a = b"}


def test_reclassify_uses_the_original_values():
    r = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = [[1, 2], [2, 3], [4, np.nan]]
    np.testing.assert_array_equal(ls.circuit.reclassify(r, t), [[2.0, 3.0], [3.0, np.nan]])
    np.testing.assert_array_equal(ls.circuit.reclassify(r, {1: 2, 2: 3, 4: None}), [[2.0, 3.0], [3.0, np.nan]])
    table = ls.circuit.read_reclass_table(REF / "omniscape_input" / "reclass.txt")
    assert table.shape == (5, 2) and np.isnan(table[4, 1])


def test_reclassified_sources_and_resistance():
    r = np.random.default_rng(2).integers(1, 5, (25, 30)).astype(float)
    table = {1: 10.0, 2: 0.5, 4: None}
    a = ls.omniscape(r, 4, reclass_table=table)
    b = ls.omniscape(ls.circuit.reclassify(r, table), 4)
    np.testing.assert_array_equal(a.cum_currmap.values, b.cum_currmap.values)
    assert np.isnan(a.cum_currmap.values[r == 4]).all()


def test_conditions():
    r = _map(3)
    temp = np.random.default_rng(3).normal(15, 3, r.shape)
    plain = ls.omniscape(r, 5, block_size=3)
    # A condition every source meets changes nothing.
    wide = ls.omniscape(r, 5, block_size=3, condition1=temp, condition1_lower=-1e9, condition1_upper=1e9)
    np.testing.assert_array_equal(wide.cum_currmap.values, plain.cum_currmap.values)
    assert wide.attrs["conditional"] == 1
    # One no source meets: no current at all.
    none = ls.omniscape(r, 5, block_size=3, condition1=temp, condition1_lower=1e8, condition1_upper=1e9)
    assert np.nansum(none.cum_currmap.values) == 0
    # "equal" on a constant layer, and futures equal to the present.
    const = np.full(r.shape, 3.0)
    eq = ls.omniscape(r, 5, block_size=3, condition1=const, comparison1="equal", condition2=temp,
                      condition2_future=temp, compare_to_future="2", condition2_lower=-1e9, condition2_upper=1e9)
    np.testing.assert_array_equal(eq.cum_currmap.values, plain.cum_currmap.values)
    # A tight window keeps some sources and drops others.
    tight = ls.omniscape(r, 5, block_size=3, condition1=temp, condition1_lower=-1, condition1_upper=1)
    assert 0 < np.nansum(tight.cum_currmap.values) < np.nansum(plain.cum_currmap.values)
    with pytest.raises(ValueError, match="comparison1"):
        ls.omniscape(r, 5, condition1=temp, comparison1="near")
    with pytest.raises(ValueError, match="condition1_future"):
        ls.omniscape(r, 5, condition1=temp, compare_to_future="1")


def test_single_precision():
    r = _map(4)
    a = ls.omniscape(r, 4, block_size=3, flow_potential=True, precision="single")
    b = ls.omniscape(r.astype(np.float32).astype(float), 4, block_size=3, flow_potential=True)
    assert a.cum_currmap.dtype == np.float32 and a.attrs["precision"] == "single"
    np.testing.assert_array_equal(a.cum_currmap.values, b.cum_currmap.values.astype(np.float32))


def test_write_outputs(tmp_path, monkeypatch):
    import rasterio

    monkeypatch.chdir(tmp_path)
    ini = tmp_path / "run.ini"
    ini.write_text((INI / "config_32bit.ini").read_text().replace("../omniscape_input/", str(REF / "omniscape_input") + "/"))
    with pytest.warns(UserWarning, match="block_size is even"):
        ds = ls.circuit.omniscape_ini(ini)
    out = tmp_path / "test1_32bit"
    assert ds.attrs["project_dir"] == "test1_32bit"
    assert sorted(p.name for p in out.iterdir()) == ["config.ini", "cum_currmap.tif", "flow_potential.tif",
                                                    "normalized_cum_currmap.tif"]
    with rasterio.open(out / "cum_currmap.tif") as f:
        v = f.read(1)
        assert f.nodata == -9999 and v.dtype == np.float32
    np.testing.assert_array_equal(np.where(v == -9999, np.nan, v), ds.cum_currmap.values)
    with pytest.warns(UserWarning) as record:
        ds2 = ls.circuit.omniscape_ini(ini, n_threads=1)
    assert any("already exists" in str(w.message) for w in record)
    assert ds2.attrs["project_dir"] == "test1_32bit_1"
    assert "resistance_file = " in (out / "config.ini").read_text()


def test_ini_errors(tmp_path):
    f = tmp_path / "a.ini"
    f.write_text("radius = 3\nproject_name = x\n")
    with pytest.raises(ValueError, match="resistance_file"):
        ls.circuit.omniscape_ini(f, write_outputs=False)
    # Omniscape's test configuration 7 has a resistance of 0.
    with pytest.raises(ValueError, match="> 0"):
        ls.circuit.omniscape_ini(INI / "config7.ini", write_outputs=False)


def _cases():
    z = np.load(REF / "omniscape_ini.npz")
    return sorted({k.split("/")[0] for k in z.files})


@pytest.mark.reference
@pytest.mark.parametrize("name", _cases())
def test_matches_omniscape_ini(name):
    z = np.load(REF / "omniscape_ini.npz")
    with pytest.warns(UserWarning) if name in ("config", "config3", "config5", "config_32bit") else _nothing():
        ds = ls.circuit.omniscape_ini(INI / f"{name}.ini", write_outputs=False)
    single = name == "config_32bit"
    checked = 0
    for out in OUTPUTS:
        key = f"{name}/cholmod/{out}"
        if key not in z.files:
            continue
        ours = ds[out].values.astype(float)
        chol, cg = z[key], z[f"{name}/cgamg/{out}"]
        np.testing.assert_array_equal(np.isnan(ours), np.isnan(chol))
        np.testing.assert_array_equal(ours == 0, chol == 0)
        top = np.nanmax(np.abs(chol))
        if single:
            # Omniscape computes in float32; landschaft in double and rounds the result.
            np.testing.assert_allclose(ours, chol, rtol=1e-3, atol=1e-3 * top)
            np.testing.assert_allclose(ours, cg, rtol=0, atol=1e-2 * top)
        else:
            np.testing.assert_allclose(ours, chol, rtol=1e-9, atol=1e-13 * top)
            np.testing.assert_allclose(ours, cg, rtol=0, atol=1e-3 * top)
        checked += 1
    assert checked


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
