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


def brute_core_cells(values, mask, depth, consider_boundary):
    """Core cells: taxicab distance to another class / NA (/ outside) above depth."""
    depth = max(depth, 1)
    core = np.zeros(values.shape, dtype=bool)
    for c in np.unique(values[~mask]):
        inside = (values == c) & ~mask
        if consider_boundary:
            dist = ndimage.distance_transform_cdt(inside, metric="taxicab")
            if inside.all():
                dist = np.full(values.shape, np.iinfo(np.int32).max)
        else:
            dist = ndimage.distance_transform_cdt(np.pad(inside, 1), metric="taxicab")[1:-1, 1:-1]
        core |= inside & (dist > depth)
    return core


class TestCore:
    def test_block(self):
        block = np.ones((5, 5), dtype=int)
        lm = ls.Landscape(block)
        p = lm.patch_metrics(["core", "ncore", "cai"])
        assert p["core"][0] == pytest.approx(9e-4)  # inner 3 x 3
        assert p["ncore"][0] == 1
        assert p["cai"][0] == pytest.approx(36)
        assert ls.Landscape(block, edge_depth=2).patch_metrics("core")["core"][0] == pytest.approx(1e-4)
        # without other classes and with the map edge ignored everything is core
        assert ls.Landscape(block, consider_boundary=True).patch_metrics("core")["core"][0] == pytest.approx(25e-4)

    def test_edge_depth_zero_is_one(self):
        data = lsm_scenes.clustered((30, 30), [1, 2], seed=3, sigma=3).data
        a = ls.Landscape(data, edge_depth=0).patch_metrics("core")
        b = ls.Landscape(data, edge_depth=1).patch_metrics("core")
        pd.testing.assert_frame_equal(a, b)

    def test_dumbbell_has_two_core_areas(self):
        data = np.full((7, 15), 2, dtype=int)
        data[1:6, 1:6] = 1
        data[1:6, 9:14] = 1
        data[3, 6:9] = 1  # one-cell corridor
        lm = ls.Landscape(data)
        p = lm.patch_metrics(["ncore", "core"])
        one = p[p["class"] == 1]
        assert one["ncore"].tolist() == [2]
        # 3 x 3 inner cells per block, plus the block cell touching the corridor
        assert one["core"].tolist() == pytest.approx([20e-4])
        c = lm.class_metrics(["ndca", "tca"])
        assert c.loc[c["class"] == 1, "ndca"].item() == 2

    @pytest.mark.parametrize("depth", [1, 2, 3])
    @pytest.mark.parametrize("consider_boundary", [False, True])
    @pytest.mark.parametrize("shape", [(40, 50), (330, 260)])
    def test_core_cells_brute_force(self, depth, consider_boundary, shape):
        data = lsm_scenes.clustered(shape, [1, 2, 3], seed=11, sigma=3, na_fraction=0.05)
        mask = np.ma.getmaskarray(data)
        lm = ls.Landscape(data, edge_depth=depth, consider_boundary=consider_boundary)
        core = brute_core_cells(data.data, mask, depth, consider_boundary)
        labels = lm.labels()
        expected = np.bincount(labels[core], minlength=lm.n_patches + 1)[1:]
        np.testing.assert_allclose(lm.patch_metrics("core")["core"], expected / 10000)
        # disjunct core areas: components of the core cells, per patch
        structure = np.ones((3, 3))
        comp, n = ndimage.label(core, structure=structure)
        owner = ndimage.labeled_comprehension(labels, comp, np.arange(1, n + 1), np.min, int, 0)
        ncore = np.bincount(owner, minlength=lm.n_patches + 1)[1:]
        np.testing.assert_array_equal(lm.patch_metrics("ncore")["ncore"], ncore)


def brute_enn(labels, patch_classes, res):
    """Per patch: exact distance to the nearest cell of another patch of its class."""
    out = np.full(len(patch_classes), np.nan)
    for cls in np.unique(patch_classes):
        ids = np.flatnonzero(patch_classes == cls) + 1
        if ids.size < 2:
            continue
        of_class = np.isin(labels, ids)
        for pid in ids:
            others = of_class & (labels != pid)
            dist = ndimage.distance_transform_edt(~others, sampling=(res[1], res[0]))
            out[pid - 1] = dist[labels == pid].min()
    return out


def min_circle_radius(points):
    """Smallest enclosing circle by brute force: every circle through two or
    three convex hull points, keeping the smallest that holds all points."""
    from itertools import combinations

    from scipy.spatial import ConvexHull

    pts = np.unique(points, axis=0)
    if len(pts) > 3:
        pts = pts[ConvexHull(pts).vertices]
    pairs = np.array(list(combinations(range(len(pts)), 2)))
    centres = [(pts[pairs[:, 0]] + pts[pairs[:, 1]]) / 2]
    if len(pts) >= 3:
        t = np.array(list(combinations(range(len(pts)), 3)))
        a, b, c = pts[t[:, 0]], pts[t[:, 1]], pts[t[:, 2]]
        d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
        ok = d != 0
        a, b, c, d = a[ok], b[ok], c[ok], d[ok]
        sa, sb, sc = (a**2).sum(1), (b**2).sum(1), (c**2).sum(1)
        ux = (sa * (b[:, 1] - c[:, 1]) + sb * (c[:, 1] - a[:, 1]) + sc * (a[:, 1] - b[:, 1])) / d
        uy = (sa * (c[:, 0] - b[:, 0]) + sb * (a[:, 0] - c[:, 0]) + sc * (b[:, 0] - a[:, 0])) / d
        centres.append(np.stack([ux, uy], axis=1))
    best = np.inf
    for chunk in np.array_split(np.concatenate(centres), max(1, len(pts) ** 3 // 200000)):
        r = np.sqrt(((chunk[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).max(1)
        best = min(best, r.min())
    return best


def extreme_corners(mask, res):
    """Corners of the patch cells in its first/last rows and columns (landscapemetrics)."""
    ys, xs = np.nonzero(mask)
    keep = (xs == xs.min()) | (xs == xs.max()) | (ys == ys.min()) | (ys == ys.max())
    xs, ys = xs[keep], ys[keep]
    corners = [(xs + dx) * res[0] for dx in (0, 1) for _ in (0, 1)], [(ys + dy) * res[1] for _ in (0, 1) for dy in (0, 1)]
    return np.stack([np.concatenate(corners[0]), np.concatenate(corners[1])], axis=1)


class TestNearestNeighbour:
    def test_hand_computed(self):
        # class 1: two cells three columns apart; class 2: one patch
        enn = ls.Landscape(np.array([[1, 2, 2, 1]]), res=10).patch_metrics("enn")
        assert enn["class"].tolist() == [1, 1, 2]
        np.testing.assert_allclose(enn["enn"], [30, 30, np.nan])

    def test_diagonal_patches_with_four_directions(self):
        diagonal = np.array([[1, 2], [2, 1]])
        enn = ls.Landscape(diagonal, directions=4, res=(3, 4)).patch_metrics("enn")["enn"]
        np.testing.assert_allclose(enn, 5.0)

    @pytest.mark.parametrize("directions", [4, 8])
    @pytest.mark.parametrize(
        "shape, res", [((40, 50), (1.0, 1.0)), ((300, 280), (30.0, 30.0)), ((60, 45), (30.0, 20.0))]
    )
    def test_brute_force(self, directions, shape, res):
        # large map (parallel paths): smoother, so the brute force stays quick
        sigma = 4.0 if shape[0] > 100 else 1.2
        data = lsm_scenes.clustered(shape, [1, 2, 3], seed=21, sigma=sigma, na_fraction=0.05)
        lm = ls.Landscape(data, res=res, directions=directions)
        expected = brute_enn(lm.labels(), lm.patch_classes, res)
        np.testing.assert_allclose(lm.patch_metrics("enn")["enn"], expected, rtol=1e-12)

    def test_sparse_distant_patches(self):
        data = lsm_scenes.sparse((200, 240), [2, 3], seed=5, n_blobs=40, background=1)
        lm = ls.Landscape(data, res=2.5)
        expected = brute_enn(lm.labels(), lm.patch_classes, (2.5, 2.5))
        np.testing.assert_allclose(lm.patch_metrics("enn")["enn"], expected, rtol=1e-12)

    def test_single_patch_classes_long_format(self):
        # every class has a single patch: one class-level row without class,
        # as in landscapemetrics
        df = ls.Landscape(np.array([[1, 1], [2, 2]])).metrics(metric="enn")
        cls = df[df["level"] == "class"]
        assert len(cls) == 3
        assert cls["class"].isna().all()
        assert cls["value"].isna().all()
        wide = ls.Landscape(np.array([[1, 1], [2, 2]])).class_metrics("enn_mn")
        assert wide["class"].tolist() == [1, 2]


class TestCircle:
    def test_hand_computed(self):
        # one cell: circle through its corners; a 1 x 3 bar: diameter sqrt(10)
        c = ls.Landscape(np.array([[1, 2, 2, 2]])).patch_metrics("circle")["circle"]
        np.testing.assert_allclose(c, [1 - 1 / (np.pi * 0.5), 1 - 3 / (np.pi * 2.5)])

    @pytest.mark.parametrize(
        "shape, res",
        [((30, 36), (1.0, 1.0)), ((30, 36), (0.3, 0.3)), ((30, 36), (30.0, 20.0)), ((300, 240), (10.0, 10.0))],
    )
    def test_brute_force(self, shape, res):
        # the large map uses the parallel path, with patches across strip seams
        sigma = 5.0 if shape[0] > 100 else 1.5
        data = lsm_scenes.clustered(shape, [1, 2, 3], seed=9, sigma=sigma, na_fraction=0.05)
        lm = ls.Landscape(data, res=res)
        labels = lm.labels()
        area = lm.patch_metrics("area")["area"].to_numpy() * 10000
        expected = [
            1 - area[i] / (np.pi * min_circle_radius(extreme_corners(labels == i + 1, res)) ** 2)
            for i in range(lm.n_patches)
        ]
        np.testing.assert_allclose(lm.patch_metrics("circle")["circle"], expected, rtol=1e-10, atol=1e-12)

    def test_repeatable(self):
        data = lsm_scenes.clustered((300, 280), [1, 2, 3], seed=3, sigma=4)
        a = ls.Landscape(data).patch_metrics("circle")
        b = ls.Landscape(data).patch_metrics("circle")
        pd.testing.assert_frame_equal(a, b)
