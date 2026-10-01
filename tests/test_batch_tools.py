# SPDX-License-Identifier: GPL-3.0-or-later
"""batch(): any landschaft tool on many rasters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

import landschaft as ls
import scenes


@pytest.fixture
def rasters(tmp_path) -> list[Path]:
    paths = []
    for i, kind in enumerate(("blobs", "network", "salt", "border")):
        data = scenes.make(kind, 300 + i)
        path = tmp_path / "in" / f"{kind}.tif"
        path.parent.mkdir(exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
                           dtype="uint8", crs="EPSG:3035", transform=from_origin(4_000_000, 3_000_000, 25, 25)) as dst:
            dst.write(data, 1)
        paths.append(path)
    return paths


def read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.crs, src.transform


@pytest.mark.parametrize("tool, params", [
    ("accounting", {"highlight_largest": True}),
    ("spa", {"variant": 6}),
    ("fad", {"window_size": 13}),
    ("euclidean_distance", {}),
    ("proximity", {"min_area": 20}),
])
def test_outputs_match_direct_calls(rasters, tmp_path, tool, params):
    out = tmp_path / "out"
    results = ls.batch(tool, rasters, out, n_workers=3, **params)
    assert [r.input for r in results] == rasters
    assert all(r.ok for r in results), [r.error for r in results]
    for src, res in zip(rasters, results):
        meta = json.loads((out / f"{src.stem}_{tool}.json").read_text())
        assert meta["tool"] == tool and meta["params"] == params
        assert set(map(Path, meta["outputs"])) | {out / f"{src.stem}_{tool}.json"} == set(res.outputs)
        direct = getattr(ls, tool)(ls.load_raster(src), **params)
        tifs = [p for p in res.outputs if p.suffix == ".tif"]
        assert tifs
        for tif in tifs:
            field = tif.stem[len(f"{src.stem}_{tool}_"):]
            expected = getattr(direct, field)
            got, crs, transform = read(tif)
            np.testing.assert_array_equal(got, np.asarray(expected).astype(got.dtype))
            assert crs.to_epsg() == 3035 and transform == from_origin(4_000_000, 3_000_000, 25, 25)


def test_reports_tables_and_values(rasters, tmp_path):
    out = tmp_path / "out"
    res = ls.batch("accounting", rasters[:1], out)[0]
    name = rasters[0].stem
    txt = (out / f"{name}_accounting.txt").read_text()
    assert "Accounting size classes result" in txt
    stats = pd.read_csv(out / f"{name}_accounting_stats.csv")
    assert list(stats["objects"]) == list(ls.accounting(ls.load_raster(rasters[0])).stats["objects"])
    values = json.loads((out / f"{name}_accounting.json").read_text())["values"]
    assert values["connectivity"] == 8 and values["thresholds"] == [50, 250, 1000, 5000, 20000]
    rss = ls.batch("restoration_status", rasters[:1], out)[0]
    assert rss.ok and json.loads((out / f"{name}_restoration_status.json").read_text())["values"]


def test_overwrite_and_skip(rasters, tmp_path):
    out = tmp_path / "out"
    first = ls.batch("spa", rasters, out, variant=2)
    again = ls.batch("spa", rasters, out, variant=2)
    assert all(r.seconds == 0.0 for r in again)
    assert [r.outputs for r in again] == [r.outputs for r in first]
    forced = ls.batch("spa", rasters[:1], out, variant=2, overwrite=True)
    assert forced[0].seconds > 0


def test_errors(rasters, tmp_path):
    bad = tmp_path / "in" / "bad.tif"
    with rasterio.open(bad, "w", driver="GTiff", width=5, height=5, count=1, dtype="uint8") as dst:
        dst.write(np.full((5, 5), 7, np.uint8), 1)
    results = ls.batch("spa", [rasters[0], bad], tmp_path / "out")
    assert results[0].ok and not results[1].ok and "InvalidInputError" in results[1].error
    with pytest.raises(ls.InvalidInputError):
        ls.batch("spa", [bad], tmp_path / "out2", raise_errors=True)
    with pytest.raises(ValueError, match="unknown tool"):
        ls.batch("nope", rasters, tmp_path / "out3")


def test_custom_function_and_threads(rasters, tmp_path):
    seen = []

    def my_tool(data, factor=1, threads=0):
        seen.append(threads)
        return (np.asarray(data) * factor).astype(np.uint8)

    results = ls.batch(my_tool, rasters[:2], tmp_path / "out", threads_per_worker=2, factor=2)
    assert all(r.ok for r in results) and seen == [2, 2]
    got, _, _ = read(results[0].outputs[0])
    np.testing.assert_array_equal(got, scenes.make("blobs", 300) * 2)


def test_mspa_uses_gtb_naming(rasters, tmp_path):
    results = ls.batch("mspa", rasters[:1], tmp_path / "out", edge_width=2)
    assert results[0].output.name == "blobs_8_2_1_1.tif"


def test_directory_and_glob_inputs(rasters, tmp_path):
    by_dir = ls.batch("spa", tmp_path / "in", tmp_path / "a", variant=3)
    by_glob = ls.batch("spa", str(tmp_path / "in" / "*.tif"), tmp_path / "b", variant=3)
    assert len(by_dir) == len(by_glob) == 4 and all(r.ok for r in by_dir + by_glob)
