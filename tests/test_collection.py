# SPDX-License-Identifier: GPL-3.0-or-later
"""load_raster on many files, RasterCollection and parallel MSPA over it."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import landschaft as ls
import scenes
from landschaft import InvalidInputError, RasterCollection
from landschaft.batch import parse_memory, run_parallel

TRANSFORM = from_origin(0.0, 1000.0, 10.0, 10.0)


def write(path, data, transform=TRANSFORM):
    data = np.asarray(data)
    bands = data if data.ndim == 3 else data[None]
    with rasterio.open(path, "w", driver="GTiff", width=bands.shape[2], height=bands.shape[1],
                       count=bands.shape[0], dtype=bands.dtype, crs="EPSG:3035", transform=transform) as dst:
        dst.write(bands)
    return path


@pytest.fixture
def tiles(tmp_path):
    folder = tmp_path / "tiles"
    folder.mkdir()
    return [write(folder / f"t{i}.tif", scenes.blobs(i, shape=(60, 70))) for i in range(4)]


def test_directory_gives_lazy_collection(tiles):
    coll = ls.load_raster(tiles[0].parent)
    assert isinstance(coll, RasterCollection) and len(coll) == 4
    assert coll.paths == tiles and coll.same_grid
    assert coll[0].pixels == 60 * 70 and coll.total_pixels == 4 * 60 * 70
    assert "4 rasters" in repr(coll)
    assert isinstance(coll[1:3], RasterCollection) and len(coll[1:3]) == 2


def test_glob_list_and_mixed_sources(tiles):
    folder = tiles[0].parent
    assert ls.load_raster(str(folder / "t[12].tif")).paths == tiles[1:3]
    assert ls.load_raster([tiles[3], tiles[0]]).paths == [tiles[3], tiles[0]]
    assert ls.load_raster([folder, tiles[0]]).paths == tiles  # duplicates dropped


def test_single_path_still_returns_dataarray(tiles):
    assert isinstance(ls.load_raster(tiles[0]), xr.DataArray)


def test_empty_directory(tmp_path):
    with pytest.raises(InvalidInputError, match="no raster"):
        ls.load_raster(tmp_path)


def test_bad_headers_raise_or_are_skipped(tiles):
    folder = tiles[0].parent
    write(folder / "multiband.tif", np.stack([scenes.blobs(9, shape=(60, 70))] * 2))
    write(folder / "float.tif", scenes.blobs(9, shape=(60, 70)).astype("float32"))
    with pytest.raises(InvalidInputError, match="2 of 6 rasters"):
        ls.load_raster(folder, validate="mspa")
    with pytest.warns(UserWarning, match="skipping 2"):
        coll = ls.load_raster(folder, errors="skip", validate="mspa")
    assert len(coll) == 4 and {p.name for p in coll.skipped} == {"multiband.tif", "float.tif"}
    with pytest.raises(InvalidInputError, match="1 of 6 rasters"):  # without checks only the stack is out
        ls.load_raster(folder)


def test_mspa_on_collection_in_memory(tiles):
    results = ls.mspa(ls.load_raster(tiles[0].parent), edge_width=2, n_workers=3)
    assert len(results) == 4
    for path, res in zip(tiles, results):
        assert isinstance(res, xr.DataArray) and res.name == path.stem
        np.testing.assert_array_equal(res.values, ls.mspa(ls.load_raster(path), edge_width=2).values)


def test_mspa_on_collection_to_disk(tiles, tmp_path):
    results = ls.mspa(ls.load_raster(tiles[0].parent), output_dir=tmp_path / "out", max_memory="1MB")
    assert all(r.ok for r in results)
    assert [r.output.name for r in results] == [f"t{i}_8_1_1_1.tif" for i in range(4)]
    with rasterio.open(results[3].output) as src:
        np.testing.assert_array_equal(src.read(1), ls.mspa(scenes.blobs(3, shape=(60, 70))))


def test_mspa_batch_accepts_directory(tiles, tmp_path):
    results = ls.mspa_batch(tiles[0].parent, tmp_path / "out")
    assert len(results) == 4 and all(r.ok for r in results)


def test_invalid_values_are_reported_when_processed(tiles, tmp_path):
    write(tiles[0].parent / "bad.tif", np.full((10, 10), 7, np.uint8))  # header is fine
    coll = ls.load_raster(tiles[0].parent)
    results = ls.mspa(coll, output_dir=tmp_path / "out")
    bad = [r for r in results if not r.ok]
    assert len(bad) == 1 and "must be 0 (missing), 1 (background) or 2" in bad[0].error


def test_stack_of_shared_grid(tiles):
    stack = ls.load_raster(tiles[0].parent, stack=True)
    assert stack.dims == ("file", "y", "x") and stack.chunks is not None
    assert list(stack["file"].values) == ["t0", "t1", "t2", "t3"]
    out = ls.mspa(stack).compute()
    np.testing.assert_array_equal(out.sel(file="t2").values, ls.mspa(scenes.blobs(2, shape=(60, 70))))


def test_stack_requires_shared_grid(tiles):
    write(tiles[0].parent / "shifted.tif", scenes.blobs(5, shape=(60, 70)), transform=from_origin(5, 1000, 10, 10))
    coll = ls.load_raster(tiles[0].parent)
    assert not coll.same_grid
    with pytest.raises(InvalidInputError, match="same grid"):
        ls.load_raster(tiles[0].parent, stack=True)


def test_collection_options_rejected_for_arrays():
    with pytest.raises(TypeError):
        ls.mspa(scenes.blobs(1), output_dir="x")


@pytest.mark.parametrize("text, expected", [("16GB", 16 * 10**9), ("500 MiB", 500 * 2**20), ("2kb", 2000), (123, 123)])
def test_parse_memory(text, expected):
    assert parse_memory(text) == expected


def test_memory_budget_limits_concurrency():
    active, peak, lock = 0, 0, threading.Lock()

    def work(_):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1

    # each item "needs" 32 bytes; a 64-byte budget lets only two run together
    run_parallel(list(range(8)), work, pixels=lambda _: 1, n_workers=8, threads_per_worker=1, max_memory=64)
    assert peak == 2
