# SPDX-License-Identifier: GPL-3.0-or-later
"""Out-of-core landscape metrics (block_rows): identical to the in-memory ones."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

import landschaft as ls
import lsm_scenes
from conftest import DATA
from landschaft import _core
from landschaft.lsm.landscape import metric_options

REF_DIR = DATA / "reference" / "lsm"
MANIFEST = json.loads((REF_DIR / "manifest.json").read_text())


def write(path, data, dtype="int32", nodata=None, res=(1.0, 1.0)):
    values = np.ma.getdata(data)
    mask = np.ma.getmaskarray(data)
    if nodata is not None:
        values = np.where(mask, nodata, values)
    elif np.issubdtype(np.dtype(dtype), np.floating):
        values = np.where(mask, np.nan, values)
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0], count=1, dtype=dtype,
        nodata=nodata, transform=from_origin(0, values.shape[0] * res[1], res[0], res[1]),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as dst:
        dst.write(values.astype(dtype), 1)
    return path


def same_as_memory(path, block_rows, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        memory = ls.Landscape(path, **options).metrics()
        tiled = ls.Landscape(path, block_rows=block_rows, **options).metrics()
    pd.testing.assert_frame_equal(tiled, memory, check_exact=True)


@pytest.mark.parametrize("fname", sorted(MANIFEST["files"]))
def test_reference_maps(fname):
    """Every stored landscapemetrics case, read in blocks of 64 rows."""
    info = MANIFEST["files"][fname]
    same_as_memory(DATA / "lsm" / info["input"], 64, **info["options"])


@pytest.mark.parametrize(
    "block_rows, options",
    [
        (64, {}),
        (128, {"directions": 4}),
        (192, {"edge_depth": 3, "consider_boundary": True, "count_boundary": True}),
        (64, {"edge_depth": 70, "directions": 4}),  # context deeper than a block
        (64, {"cell_center": True, "neighbourhood": 8, "ordered": False, "base": "log10"}),
        ("auto", {"classes_max": 9}),
    ],
)
def test_clustered_map(tmp_path, block_rows, options):
    data = lsm_scenes.clustered((330, 170), [1, 2, 3, 4, 5], seed=3, sigma=2.5, na_fraction=0.05)
    path = write(tmp_path / "map.tif", data, nodata=-9, res=(30.0, 20.0))
    same_as_memory(path, block_rows, **options)


@pytest.mark.parametrize("directions", [4, 8])
def test_patches_across_many_blocks(tmp_path, directions):
    # stripes and a comb: patches spanning every block, joined through seams
    data = np.ones((400, 150), dtype=np.int32)
    data[::5, :] = 2
    data[:, ::9] = 3
    data[200:260, 40:90] = 4
    rng = np.random.default_rng(1)
    data[rng.random(data.shape) < 0.02] = 5
    path = write(tmp_path / "comb.tif", np.ma.MaskedArray(data))
    same_as_memory(path, 64, directions=directions, edge_depth=2)


@pytest.mark.parametrize(
    "dtype, nodata, classes",
    [
        ("uint8", 0, [1, 2, 3]),
        ("int16", -32768, [-300, -2, 5, 700]),
        ("int32", None, [-100000, 7, 2**30]),
        ("float32", None, [1, 2, 3]),
    ],
)
def test_file_types(tmp_path, dtype, nodata, classes):
    data = lsm_scenes.salt_and_pepper((150, 90), classes, seed=5, na_fraction=0.1)
    path = write(tmp_path / f"{dtype}.tif", data, dtype=dtype, nodata=nodata)
    same_as_memory(path, 64)


def test_single_block_and_exact_multiple(tmp_path):
    small = write(tmp_path / "small.tif", lsm_scenes.clustered((40, 50), [1, 2], seed=1))
    same_as_memory(small, 64)
    exact = write(tmp_path / "exact.tif", lsm_scenes.clustered((128, 50), [1, 2, 3], seed=2))
    same_as_memory(exact, 64)


def test_float_truncation_warns(tmp_path):
    data = np.array([[1.7, 2.2], [2.0, 1.0]] * 40)
    path = tmp_path / "float.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=80, count=1, dtype="float64",
                       transform=from_origin(0, 80, 1, 1)) as dst:
        dst.write(data, 1)
    with pytest.warns(UserWarning, match="truncated"):
        lm = ls.Landscape(path, block_rows=64)
    assert lm.classes.tolist() == [1, 2]


def test_detail_pass_only_when_needed(tmp_path):
    path = write(tmp_path / "map.tif", lsm_scenes.clustered((200, 80), [1, 2, 3], seed=4))
    lm = ls.Landscape(path, block_rows=64)
    assert lm.out_of_core
    lm.class_metrics(["ca", "np", "ed", "ai"])
    assert lm._tiled.done == set()
    lm.patch_metrics(["enn", "area"])
    assert lm._tiled.done == {"enn"}
    lm.landscape_metrics(["core_mn", "gyrate_mn"])
    assert lm._tiled.done == {"enn", "core", "gyrate"}


def test_raster_outputs_are_not_available(tmp_path):
    path = write(tmp_path / "map.tif", lsm_scenes.clustered((70, 30), [1, 2], seed=1))
    lm = ls.Landscape(path, block_rows=64)
    with pytest.raises(NotImplementedError):
        lm.labels()
    with pytest.raises(NotImplementedError):
        lm.to_raster("area")


def test_landscape_metrics_function(tmp_path):
    a = write(tmp_path / "a.tif", lsm_scenes.clustered((100, 60), [1, 2], seed=1))
    b = write(tmp_path / "b.tif", lsm_scenes.clustered((90, 70), [1, 2, 3], seed=2))
    one = ls.landscape_metrics(a, level="landscape", block_rows=64)
    pd.testing.assert_frame_equal(one, ls.landscape_metrics(a, level="landscape"))
    both = ls.landscape_metrics([a, b], level="class", block_rows=64)
    pd.testing.assert_frame_equal(both, ls.landscape_metrics([a, b], level="class"))


def test_errors(tmp_path):
    with pytest.raises(TypeError, match="file"):
        ls.Landscape(np.ones((5, 5), dtype=int), block_rows=64)
    path = write(tmp_path / "map.tif", lsm_scenes.clustered((70, 30), [1, 2], seed=1))
    with pytest.raises(ValueError, match="positive"):
        ls.Landscape(path, block_rows=0)


class TestEngine:
    """The C++ engine checks how it is fed."""

    def engine(self, ny=130, nx=10):
        return _core.LsmTiled(nx, ny, 1.0, 1.0, metric_options(), np.array([1], dtype=np.int32))

    def block(self, rows, nx=10):
        return np.ones((rows, nx), dtype=np.int32), np.zeros((rows, nx), dtype=np.uint8)

    def test_blocks_in_order(self):
        e = self.engine()
        with pytest.raises(ValueError, match="order"):
            e.add_block(*self.block(65), 64, 64, 1, 0)

    def test_blocks_multiple_of_64(self):
        e = self.engine()
        with pytest.raises(ValueError, match="multiple of 64"):
            e.add_block(*self.block(51), 0, 50, 0, 1)

    def test_context_rows(self):
        e = self.engine()
        with pytest.raises(ValueError, match="context"):
            e.add_block(*self.block(64), 0, 64, 0, 0)

    def test_unknown_class(self):
        e = self.engine()
        values, mask = self.block(65)
        values[3, 3] = 2
        with pytest.raises(ValueError, match="class"):
            e.add_block(values, mask, 0, 64, 0, 1)

    def test_finish_needs_the_whole_raster(self):
        e = self.engine()
        e.add_block(*self.block(65), 0, 64, 0, 1)
        with pytest.raises(RuntimeError):
            e.finish()
