# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests of the landscape-metric engine and API (values checked by hand)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin
from scipy import ndimage

import landschaft as ls
import lsm_scenes

# 1 1 2
# 1 2 2
# 3 3 2
SMALL = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 2]])


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Two label images describe the same patches (0 = background in both)."""
    if not np.array_equal(a == 0, b == 0):
        return False
    pairs = np.unique(np.stack([a[a > 0], b[b > 0]]), axis=1)
    return len(np.unique(pairs[0])) == pairs.shape[1] == len(np.unique(pairs[1]))


def scipy_labels(values: np.ndarray, mask: np.ndarray, directions: int) -> np.ndarray:
    structure = np.ones((3, 3)) if directions == 8 else None
    out = np.zeros(values.shape, dtype=np.int64)
    offset = 0
    for c in np.unique(values[~mask]):
        lab, n = ndimage.label((values == c) & ~mask, structure=structure)
        out[lab > 0] = lab[lab > 0] + offset
        offset += n
    return out


class TestHandComputed:
    def test_patch_metrics(self):
        p = ls.Landscape(SMALL).patch_metrics(["area", "perim", "shape", "para", "contig", "gyrate"])
        assert p["class"].tolist() == [1, 2, 3]
        assert p["id"].tolist() == [1, 2, 3]
        np.testing.assert_allclose(p["area"], [3e-4, 4e-4, 2e-4])
        np.testing.assert_allclose(p["perim"], [8, 10, 6])
        np.testing.assert_allclose(p["shape"], [0.25 * 8 / np.sqrt(3), 0.25 * 10 / 2, 0.25 * 6 / np.sqrt(2)])
        np.testing.assert_allclose(p["para"], [8 / 3, 10 / 4, 6 / 2])
        # class 1: 3 cells, 2 orthogonal pairs (weight 2) and 1 diagonal pair, each counted from both cells
        np.testing.assert_allclose(p["contig"][0], ((2 * 1 + 2 * 2 * 2 + 3) / 3 - 1) / 12)
        # class 3: two cells one apart -> 0.5 from the centroid each
        assert p["gyrate"][2] == pytest.approx(0.5)

    def test_class_metrics(self):
        c = ls.Landscape(SMALL, res=10).class_metrics(["ca", "pland", "np", "te", "lpi", "ai"])
        np.testing.assert_allclose(c["ca"], [0.03, 0.04, 0.02])
        np.testing.assert_allclose(c["pland"], [300 / 9, 400 / 9, 200 / 9])
        np.testing.assert_allclose(c["np"], [1, 1, 1])
        # shared sides: 1|2 = 3, 1|3 = 1, 2|3 = 2
        np.testing.assert_allclose(c["te"], [40, 50, 30])
        # like adjacencies over the maximum for the class size (3 cells -> 2, 4 cells -> 4)
        np.testing.assert_allclose(c["ai"], [100, 75, 100])

    def test_landscape_metrics(self):
        lm = ls.Landscape(SMALL).landscape_metrics(["ta", "np", "te", "pr", "shdi", "sidi"])
        p = np.array([3, 4, 2]) / 9
        assert lm["ta"][0] == pytest.approx(9e-4)
        assert lm["np"][0] == 3
        assert lm["te"][0] == 6
        assert lm["pr"][0] == 3
        assert lm["shdi"][0] == pytest.approx(-(p * np.log(p)).sum())
        assert lm["sidi"][0] == pytest.approx(1 - (p**2).sum())

    def test_count_boundary(self):
        te = ls.Landscape(SMALL, count_boundary=True).landscape_metrics("te")["te"][0]
        assert te == 6 + 12  # plus the 12 cell sides on the map edge

    def test_non_square_cells(self):
        # a single horizontal pair of cells: 2 left/right sides, 4 top/bottom sides
        lm = ls.Landscape(np.array([[1, 1]]), res=(30, 10))
        assert lm.patch_metrics("perim")["perim"][0] == pytest.approx(2 * 30 + 4 * 10)
        assert lm.patch_metrics("area")["area"][0] == pytest.approx(2 * 300 / 10000)

    def test_directions(self):
        diagonal = np.array([[1, 2], [2, 1]])
        assert ls.Landscape(diagonal, directions=8).n_patches == 2
        assert ls.Landscape(diagonal, directions=4).n_patches == 4

    def test_single_cell_patch_frac_is_one(self):
        f = ls.Landscape(np.array([[1, 2, 1]])).patch_metrics("frac")["frac"]
        np.testing.assert_allclose(f, 1.0)


class TestLabels:
    @pytest.mark.parametrize("directions", [4, 8])
    @pytest.mark.parametrize("shape", [(37, 53), (300, 280)])  # small: serial, large: parallel strips
    def test_same_patches_as_scipy(self, directions, shape):
        data = lsm_scenes.salt_and_pepper(shape, [1, 2, 3], seed=4, na_fraction=0.1)
        lm = ls.Landscape(data, directions=directions)
        mask = np.ma.getmaskarray(data)
        labels = lm.labels()
        assert same_partition(labels, scipy_labels(data.data, mask, directions))
        assert labels.max() == lm.n_patches

    def test_long_patches_cross_strips(self):
        # spirals and stripes spanning many parallel strips must stay one patch
        data = np.ones((600, 400), dtype=np.int32)
        data[::4, :] = 2
        data[:, ::7] = 3
        lm = ls.Landscape(data)
        assert same_partition(lm.labels(), scipy_labels(data, np.zeros_like(data, bool), 8))

    def test_patch_order(self):
        # landscapemetrics: by class, then by first cell in column-major order
        data = np.array([[2, 1, 2], [1, 2, 1], [1, 1, 2]])
        lm = ls.Landscape(data, directions=4)
        labels = lm.labels()
        assert lm.patch_classes.tolist() == [1, 1, 1, 2, 2, 2, 2]
        assert labels[1, 0] == 1  # class 1, column 0
        assert labels[0, 1] == 2  # class 1, column 1
        assert labels[1, 2] == 3
        assert labels[0, 0] == 4
        assert labels[1, 1] == 5
        assert labels[0, 2] == 6
        assert labels[2, 2] == 7

    def test_patch_table_matches_labels(self):
        data = lsm_scenes.clustered((60, 70), [1, 2, 3, 4], seed=8, na_fraction=0.1)
        lm = ls.Landscape(data)
        labels = lm.labels()
        cells = np.bincount(labels.ravel(), minlength=lm.n_patches + 1)[1:]
        np.testing.assert_allclose(lm.patch_metrics("area")["area"], cells / 10000)
        for pid, cls in zip(range(1, lm.n_patches + 1), lm.patch_classes):
            assert np.all(data.data[labels == pid] == cls)


class TestInput:
    def test_masked_array_and_nodata(self):
        data = np.array([[1, 1, -9], [2, 2, 1]])
        a = ls.Landscape(np.ma.masked_equal(data, -9)).class_metrics("ca")
        b = ls.Landscape(data, nodata=-9).class_metrics("ca")
        pd.testing.assert_frame_equal(a, b)
        assert a["class"].tolist() == [1, 2]

    def test_float_nan_is_na(self):
        data = np.array([[1.0, np.nan], [2.0, 2.0]])
        lm = ls.Landscape(data)
        assert lm.classes.tolist() == [1, 2]
        assert lm.landscape_metrics("ta")["ta"][0] == pytest.approx(3e-4)

    def test_float_truncation_warns(self):
        with pytest.warns(UserWarning, match="truncated"):
            lm = ls.Landscape(np.array([[1.7, 2.2]]))
        assert lm.classes.tolist() == [1, 2]

    def test_dataarray_resolution_and_nodata(self):
        da = xr.DataArray(np.array([[1, 1], [0, 2]], dtype=np.int16), dims=("y", "x"))
        da = da.rio.write_transform(from_origin(0, 60, 30, 30)).rio.write_nodata(0)
        lm = ls.Landscape(da)
        assert lm.resolution == (30.0, 30.0)
        assert lm.classes.tolist() == [1, 2]
        labels = lm.labels()
        assert isinstance(labels, xr.DataArray)
        assert labels.rio.transform() == da.rio.transform()
        raster = lm.to_raster("area")
        assert np.isnan(raster.values[1, 0])
        assert raster.values[0, 0] == pytest.approx(0.18)

    def test_geographic_crs_warns(self):
        da = xr.DataArray(np.array([[1, 2]], dtype=np.int32), dims=("y", "x"))
        da = da.rio.write_transform(from_origin(10, 50, 0.01, 0.01)).rio.write_crs("EPSG:4326")
        with pytest.warns(UserWarning, match="geographic"):
            ls.Landscape(da)

    def test_file(self, tmp_path):
        import lsm_reference

        path = tmp_path / "map.tif"
        data = np.ma.MaskedArray(np.array([[1, 2], [2, 2]]), mask=[[False, True], [False, False]])
        lsm_reference.write_landscape(path, data, res=(5, 5))
        lm = ls.Landscape(path)
        assert lm.resolution == (5.0, 5.0)
        assert lm.class_metrics("np")["np"].tolist() == [1, 1]

    def test_all_na(self):
        lm = ls.Landscape(np.ma.masked_all((3, 3), dtype=np.int32))
        assert lm.class_metrics().empty
        assert np.isnan(lm.landscape_metrics("ta")["ta"][0])

    @pytest.mark.parametrize(
        "kwargs",
        [{"directions": 6}, {"neighbourhood": 5}, {"base": "ln"}, {"res": (0, 1)}],
    )
    def test_bad_options(self, kwargs):
        with pytest.raises(ValueError):
            ls.Landscape(SMALL, **kwargs)

    def test_three_dimensional_input(self):
        with pytest.raises(ValueError, match="2-D"):
            ls.Landscape(np.zeros((2, 3, 3), dtype=int))


class TestSelection:
    def test_list_metrics_filters(self):
        assert set(ls.list_metrics(level="patch")["level"]) == {"patch"}
        area = ls.list_metrics(metric="area", level="class")
        assert area["metric"].tolist() == ["area_cv", "area_mn", "area_sd"]
        no_patch = ls.list_metrics(level="-patch")
        assert "patch" not in set(no_patch["level"])
        what = ls.list_metrics(what=["lsm_c_ca", "landscape"])
        assert "lsm_c_ca" in set(what["function_name"])
        assert set(what["level"]) == {"class", "landscape"}
        diversity = ls.list_metrics(type="diversity metric")
        assert set(diversity["level"]) == {"landscape"}

    def test_list_metrics_errors(self):
        with pytest.raises(ValueError, match="mix"):
            ls.list_metrics(level=["patch", "-class"])
        with pytest.raises(ValueError):
            ls.list_metrics(metric="nonexistent")
        with pytest.raises(ValueError, match="unknown"):
            ls.list_metrics(what="lsm_x_foo")

    def test_wrong_level(self):
        lm = ls.Landscape(SMALL)
        with pytest.raises(ValueError, match="class level"):
            lm.patch_metrics("ca")
        with pytest.raises(ValueError, match="not landscape-level"):
            lm.landscape_metrics("pland")

    def test_long_format(self):
        lm = ls.Landscape(SMALL)
        df = lm.metrics(metric=["np", "area"])
        assert list(df.columns) == ["level", "class", "id", "metric", "value"]
        patch = df[df["level"] == "patch"]
        assert patch["id"].tolist() == [1, 2, 3]
        assert df[df["level"] == "landscape"]["class"].isna().all()
        assert df[df["level"] == "class"]["id"].isna().all()
        wide = lm.class_metrics(["area_mn", "np"])
        long = df[(df["level"] == "class") & (df["metric"] == "np")]["value"].to_numpy()
        np.testing.assert_array_equal(wide["np"].to_numpy(), long)

    def test_landscape_metrics_function(self):
        one = ls.landscape_metrics(SMALL, level="landscape")
        assert "layer" not in one.columns
        many = ls.landscape_metrics([SMALL, SMALL.T], what="lsm_l_np", directions=4)
        assert many["layer"].tolist() == [1, 2]
        assert many["value"].tolist() == [3, 3]

    def test_all_metrics_computable(self):
        data = lsm_scenes.clustered((30, 40), [1, 2, 3, 4], seed=2, na_fraction=0.1)
        df = ls.Landscape(data, classes_max=10).metrics()
        assert set(df["metric"]) == set(ls.list_metrics()["metric"])
        assert df.loc[df["metric"] != "pafrac", "value"].notna().mean() > 0.95


class TestCaching:
    def test_metrics_do_not_change_on_repeat(self):
        lm = ls.Landscape(lsm_scenes.clustered((40, 40), [1, 2, 3], seed=5))
        first = lm.metrics()
        second = lm.metrics()
        pd.testing.assert_frame_equal(first, second)

    def test_threads_do_not_change_results(self, monkeypatch):
        data = lsm_scenes.salt_and_pepper((400, 300), [1, 2, 3, 4], seed=9, na_fraction=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = ls.Landscape(data).metrics(level=["class", "landscape"])
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api="openmp"):
            b = ls.Landscape(data).metrics(level=["class", "landscape"])
        pd.testing.assert_frame_equal(a, b)
