# SPDX-License-Identifier: GPL-3.0-or-later
"""MSPA Tiling: unit tests and agreement with GuidosToolbox (its IDL code in GDL + GTB's gdal tools and mspa)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr

import landschaft as ls
import scenes
from gtb_reference import find_gdl, find_gtb_fwtools, find_gtb_mspa, run_gtb_mspa_tile

REFERENCE = Path(__file__).parent / "data" / "reference" / "idl_reference.json"


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def test_tile_layout_gtb_windows():
    # GTB on MS-Windows: 6500-pixel tiles every 4500 pixels, seams in the middle of the 2000-pixel overlaps
    assert ls.tile_layout(20000) == [(0, 6500, 0, 5500), (4500, 11000, 5500, 10000), (9000, 15500, 10000, 14500),
                                     (13500, 20000, 14500, 20000)]
    assert ls.tile_layout(6500) == [(0, 6500, 0, 6500)]
    assert ls.tile_layout(6501) == [(0, 6500, 0, 5500), (4500, 6501, 5500, 6501)]


@pytest.mark.parametrize("size,tile,overlap", [(1000, 300, 60), (241, 60, 10), (50, 60, 16), (997, 101, 0), (500, 90, 11)])
def test_tile_layout_partitions_the_axis(size, tile, overlap):
    tiles = ls.tile_layout(size, tile, overlap)
    assert tiles[0][2] == 0 and tiles[-1][3] == size
    for (r0, r1, k0, k1), nxt in zip(tiles, tiles[1:] + [None]):
        assert r0 <= k0 < k1 <= r1 and r1 - r0 <= tile
        if nxt is not None:
            assert nxt[2] == k1  # kept parts are contiguous


def test_invalid_parameters():
    with pytest.raises(ValueError, match="tile_size"):
        ls.tile_layout(100, 50, 50)
    with pytest.raises(ValueError, match="edge_width"):
        ls.mspa_tiled(scenes.make("blobs", 1), edge_width=0)
    bad = scenes.make("blobs", 1).copy()
    bad[0, 0] = 3
    with pytest.raises(ls.InvalidInputError):
        ls.mspa_tiled(bad, tile_size=60, overlap=10)


def test_one_tile_is_mspa():
    d = scenes.make("blobs", 2)
    np.testing.assert_array_equal(ls.mspa_tiled(d), np.asarray(ls.mspa(d)))
    np.testing.assert_array_equal(ls.mspa_tiled(d, tile_size=400, overlap=100), np.asarray(ls.mspa(d)))


def test_tile_shortcuts():
    d = np.ones((60, 90), np.uint8)
    d[:30, :30] = 0  # tile of missing only
    d[:30, 30:60] = 2  # tile of foreground only
    d[30:, :30] = np.where(np.arange(30) % 3 == 0, 0, 2)  # foreground and missing
    d[30:, 30:60] = np.where(np.arange(30) % 4 == 0, 0, 1)  # background and missing
    d[30:, 60:] = 1
    d[40:50, 70:80] = 2
    m = np.asarray(ls.mspa_tiled(d, tile_size=30, overlap=0))
    assert (m[:30, :30] == 129).all() and (m[:30, 30:60] == 17).all()
    np.testing.assert_array_equal(m[30:, :30], np.where(d[30:, :30] == 2, 17, 129))
    np.testing.assert_array_equal(m[30:, 30:60], np.where(d[30:, 30:60] == 0, 129, 0))
    np.testing.assert_array_equal(m[30:, 60:], np.asarray(ls.mspa(d[30:, 60:])))


def test_small_objects_match_global_mspa():
    # objects far smaller than the overlap: tiling does not change MSPA
    d = scenes.make("salt", 5)
    np.testing.assert_array_equal(ls.mspa_tiled(d, tile_size=70, overlap=30), np.asarray(ls.mspa(d)))


def test_file_mode_workers_and_naming(tmp_path):
    da = ls.load_raster(ls.example_data("input"))
    src = ls.write_raster(da, tmp_path / "forest.tif")
    mem = ls.mspa_tiled(da, tile_size=300, overlap=60)
    assert isinstance(mem, xr.DataArray) and mem.attrs["mspa_tile_size"] == 300
    out = ls.mspa_tiled(src, tile_size=300, overlap=60, n_workers=3)
    assert out == tmp_path / "forest_8_1_1_1.tif"
    with rasterio.open(out) as f:
        np.testing.assert_array_equal(f.read(1), mem.values)
        assert f.transform == da.rio.transform() and f.crs == da.rio.crs
        assert f.colormap(1)[17][:3] == ls.mspa_colormap()[17][:3]
        assert "MSPA Tiling" in f.tags()["TIFFTAG_IMAGEDESCRIPTION"]
    with pytest.raises(FileExistsError):
        ls.mspa_tiled(src, tile_size=300, overlap=60)
    out2 = ls.mspa_tiled(src, tmp_path / "x.tif", connectivity=4, internal=False, tile_size=250, overlap=40)
    with rasterio.open(out2) as f:
        np.testing.assert_array_equal(f.read(1), ls.mspa_tiled(da.values, connectivity=4, internal=False,
                                                              tile_size=250, overlap=40))
    ls.mspa_tiled(da, tmp_path / "mem.tif", tile_size=300, overlap=60)
    assert ls.load_raster(tmp_path / "mem.tif").shape == da.shape


# --- agreement with GTB ------------------------------------------------------------


def _reference() -> list[dict]:
    return json.loads(REFERENCE.read_text()).get("mspa_tile", []) if REFERENCE.exists() else []


@pytest.fixture(scope="module")
def reference_inputs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_idl import mspa_tile_inputs

    return mspa_tile_inputs()


@pytest.mark.reference
@pytest.mark.parametrize("case", _reference(),
                         ids=lambda c: f"{c['input']}-t{c['tile_size']}-o{c['overlap']}-{len(c['params'])}")
def test_mspa_tiled_matches_stored_gtb_outputs(reference_inputs, case):
    data = reference_inputs[case["input"]]
    assert digest(data) == case["input_sha256"]
    m = ls.mspa_tiled(data, tile_size=case["tile_size"], overlap=case["overlap"], **case["params"])
    assert digest(m) == case["sha256"], "tiled MSPA differs from GTB"


@pytest.mark.live_reference
@pytest.mark.parametrize("seed", range(3))
def test_mspa_tiled_live_against_gtb(seed):
    gdl, mspa_exe, fwtools = find_gdl(), find_gtb_mspa(), find_gtb_fwtools()
    if gdl is None or mspa_exe is None or fwtools is None:
        pytest.skip("GDL, GTB's mspa or GTB's gdal tools not found")
    rng = np.random.default_rng(1400 + seed)
    kind = ["blobs", "network", "border"][seed]
    data = scenes.make(kind, int(rng.integers(1, 10_000)))
    tile = int(rng.integers(40, 130))
    overlap = int(rng.integers(0, tile // 2))
    params = {"connectivity": int(rng.choice([4, 8])), "edge_width": int(rng.integers(1, 4)),
              "internal": bool(rng.integers(0, 2)), "transition": bool(rng.integers(0, 2))}
    ref = run_gtb_mspa_tile(gdl, mspa_exe, fwtools, data, tile_size=tile, overlap=overlap, **params)
    np.testing.assert_array_equal(ls.mspa_tiled(data, tile_size=tile, overlap=overlap, **params), ref)
