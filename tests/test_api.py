# SPDX-License-Identifier: GPL-3.0-or-later
"""High-level API: xarray, dask, batch processing and statistics."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import xarray as xr

import landschaft as ls
import scenes


@pytest.fixture
def stack() -> np.ndarray:
    return np.stack([scenes.blobs(s, shape=(64, 80)) for s in range(3)])


def test_mspa_on_dataarray_keeps_coords_and_records_parameters():
    data = scenes.blobs(1, shape=(50, 60))
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": np.arange(50) * -10.0, "x": np.arange(60) * 10.0})
    da = da.rio.write_crs("EPSG:3035")
    out = ls.mspa(da, edge_width=3)
    assert isinstance(out, xr.DataArray) and out.dims == ("y", "x")
    np.testing.assert_array_equal(out["x"], da["x"])
    np.testing.assert_array_equal(out.values, ls.mspa(data, edge_width=3))
    assert out.attrs["landschaft_product"] == "MSPA" and out.attrs["mspa_edge_width"] == 3
    assert out.rio.crs.to_string() == "EPSG:3035"


def test_mspa_on_dask_time_stack_is_lazy_and_matches(stack):
    pytest.importorskip("dask")
    da = xr.DataArray(stack, dims=("time", "y", "x")).chunk({"time": 1, "y": 32, "x": 40})
    out = ls.mspa(da, connectivity=4)
    assert out.chunks is not None  # still lazy
    computed = out.compute()
    for t in range(stack.shape[0]):
        np.testing.assert_array_equal(computed[t].values, ls.mspa(stack[t], connectivity=4))


def test_mspa_on_numpy_stack(stack):
    out = ls.mspa(stack)
    assert out.shape == stack.shape
    np.testing.assert_array_equal(out[1], ls.mspa(stack[1]))


def test_mspa_on_plain_dask_array(stack):
    dsa = pytest.importorskip("dask.array")
    out = ls.mspa(dsa.from_array(stack, chunks=(1, 32, 32)))
    np.testing.assert_array_equal(out.compute(), ls.mspa(stack))


def test_mspa_stats_accounts_for_every_pixel():
    data = scenes.network(3)
    result = ls.mspa(data)
    stats = ls.mspa_stats(result)
    assert stats["pixels"].sum() == data.size
    fg = stats.loc[["Core", "Islet", "Perforation", "Edge", "Loop", "Bridge", "Branch"]]
    assert fg["pixels"].sum() == stats.attrs["foreground_area"] == int((data == 2).sum())
    assert np.isclose(fg["pct_foreground"].sum(), 100.0)
    assert np.isclose(stats["pct_data"].drop("Missing").sum(), 100.0)


def test_mspa_batch_processes_files_in_parallel(tmp_path):
    inputs = []
    for i in range(5):
        path = tmp_path / f"tile_{i}.tif"
        data = scenes.blobs(i, shape=(60, 70))
        with rasterio.open(path, "w", driver="GTiff", width=70, height=60, count=1, dtype="uint8") as dst:
            dst.write(data, 1)
        inputs.append(path)
    bad = tmp_path / "bad.tif"
    with rasterio.open(bad, "w", driver="GTiff", width=5, height=5, count=1, dtype="uint8") as dst:
        dst.write(np.full((5, 5), 7, np.uint8), 1)
    inputs.append(bad)

    results = ls.mspa_batch(inputs, tmp_path / "out", edge_width=2, n_workers=3)
    assert [r.input for r in results] == inputs
    assert all(r.ok for r in results[:5]) and not results[5].ok
    assert "larger than 2" in results[5].error
    assert results[0].output.name == "tile_0_8_2_1_1.tif"  # GTB naming
    with rasterio.open(results[2].output) as src:
        np.testing.assert_array_equal(src.read(1), ls.mspa(scenes.blobs(2, shape=(60, 70)), edge_width=2))

    again = ls.mspa_batch(inputs[:2], tmp_path / "out", edge_width=2)  # existing outputs are skipped
    assert all(r.ok and r.seconds == 0.0 for r in again)
