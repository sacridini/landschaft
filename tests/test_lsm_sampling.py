# SPDX-License-Identifier: GPL-3.0-or-later
"""Sample plots, moving windows and several rasters (landscapemetrics'
sample_lsm, window_lsm and calculate_lsm on layers).

Stored references come from ``tools/make_reference_lsm.py sample window``;
the live tests run landscapemetrics now on random maps (Rscript required).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

import landschaft as ls
import lsm_reference
import lsm_scenes
from conftest import DATA
from test_lsm_reference import ATOL, NEIGHBOURHOOD_QUIRK, NOT_BITWISE, RTOL

SAMPLE_DIR = DATA / "reference" / "lsm_sample"
WINDOW_DIR = DATA / "reference" / "lsm_window"
SAMPLE_MANIFEST = json.loads((SAMPLE_DIR / "manifest.json").read_text())
WINDOW_MANIFEST = json.loads((WINDOW_DIR / "manifest.json").read_text())
SAMPLE_ARGS = {"shape", "size", "sizes", "all_classes", "level"}


def assert_window_equal(ours: np.ndarray, ref: np.ndarray, label: str, metric: str) -> None:
    np.testing.assert_allclose(ours, ref, rtol=RTOL, atol=ATOL, err_msg=label)
    if not metric.startswith(NOT_BITWISE):
        same = (ours == ref) | (np.isnan(ours) & np.isnan(ref))
        assert same.all(), f"{label}: not the same bits at {np.argwhere(~same)[:5].tolist()}"


def compare_sample(ours: pd.DataFrame, ref: pd.DataFrame, label: str) -> None:
    keys = ["level", "class", "id", "metric", "plot_id"] + (["size"] if "size" in ours.columns else [])
    ours = ours.assign(level=ours["level"].astype(str), metric=ours["metric"].astype(str))
    merged = ours.merge(ref, on=keys, how="outer", suffixes=("", "_ref"), indicator=True)
    missing = merged[merged["_merge"] != "both"]
    assert missing.empty, f"{label}: rows only on one side:\n{missing.head(10)}"
    # geodesic plot areas (plots with a CRS) agree to 1e-9: PROJ / GeographicLib versions
    for col, rtol in (("value", RTOL), ("percentage_inside", 1e-9)):
        a = merged[col].to_numpy(float)
        b = merged[col + "_ref"].to_numpy(float)
        ok = np.isclose(a, b, rtol=rtol, atol=ATOL, equal_nan=True)
        assert ok.all(), f"{label}: {col} differs:\n{merged.loc[~ok, keys + [col, col + '_ref']].head(10)}"
        if col == "value":
            bitwise = ~merged["metric"].str.startswith(NOT_BITWISE).to_numpy()
            bad = bitwise & ~((a == b) | (np.isnan(a) & np.isnan(b)))
            assert not bad.any(), f"{label}: not the same bits:\n{merged.loc[bad, keys + [col, col + '_ref']].head(10)}"
    # sample_lsm's row order (plot, level, metric, class, id)
    assert ours["plot_id"].tolist() == ref["plot_id"].tolist()
    assert ours["level"].tolist() == ref["level"].tolist()


def sample_input(info: dict):
    """The plots of a stored case as sample_metrics takes them, and its options."""
    plots = pd.read_csv(SAMPLE_DIR / info["plots"], float_precision="round_trip")
    options = dict(info["options"])
    kwargs = {k: options.pop(k) for k in list(options) if k in SAMPLE_ARGS}
    if "plot_id" in plots:
        kwargs["plot_id"] = plots["plot_id"].to_numpy()
    if "wkt" in plots:
        shapely = pytest.importorskip("shapely")
        y = [shapely.wkt.loads(w) for w in plots["wkt"]]
    elif options.pop("vector", False):
        gpd = pytest.importorskip("geopandas")
        import rasterio

        with rasterio.open(DATA / "lsm" / info["input"]) as src:
            crs = src.crs
        y = gpd.GeoDataFrame(geometry=gpd.points_from_xy(plots["x"], plots["y"]), crs=crs)
    else:
        y = plots[["x", "y"]].to_numpy()
    return y, kwargs, options


@pytest.mark.reference
@pytest.mark.parametrize("fname", sorted(SAMPLE_MANIFEST["files"]))
def test_sample_matches_landscapemetrics(fname):
    info = SAMPLE_MANIFEST["files"][fname]
    y, kwargs, options = sample_input(info)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "sizes" in kwargs:  # scale_sample
            sizes = [float(v) for v in str(kwargs.pop("sizes")).split(",")]
            ours = ls.scale_sample(DATA / "lsm" / info["input"], y, size=sizes, **kwargs, **options)
        else:
            ours = ls.sample_metrics(DATA / "lsm" / info["input"], y, **kwargs, **options)
    ref = lsm_reference.read_reference(SAMPLE_DIR / fname)
    ref["percentage_inside"] = ref["percentage_inside"].astype(float)
    if "size" in ref.columns:
        ref["size"] = ref["size"].astype(float)
        assert ours["size"].tolist() == ref["size"].tolist()  # scale_sample's row order
    if not ref["metric"].str.startswith("circle").any():  # non-square cells
        ours = ours[~ours["metric"].astype(str).str.startswith("circle")].reset_index(drop=True)
    compare_sample(ours, ref, fname)


@pytest.mark.reference
@pytest.mark.parametrize("fname", sorted(WINDOW_MANIFEST["files"]))
def test_window_matches_landscapemetrics(fname):
    info = WINDOW_MANIFEST["files"][fname]
    ref = np.load(WINDOW_DIR / fname)
    metrics = info["metrics"]
    ours = ls.window_metrics(WINDOW_DIR / info["input"], ref["window"], what=["lsm_l_" + m for m in metrics],
                             **info["options"])
    for m in metrics:
        assert_window_equal(np.asarray(ours[m]), ref[m], f"{fname}: {m}", m)


# -- live comparison ---------------------------------------------------------


@pytest.fixture(scope="module")
def rscript():
    exe = lsm_reference.find_rscript()
    if exe is None:
        pytest.skip("Rscript not found (set LANDSCHAFT_RSCRIPT)")
    return exe


@pytest.mark.live_reference
@pytest.mark.parametrize(
    "seed, shape, size, res, options",
    [
        (1, "square", 7.0, (2.0, 2.0), {}),
        (2, "circle", 9.5, (1.0, 1.0), {"all_classes": True, "directions": 4}),
        (3, "circle", 33.0, (3.0, 3.0), {"edge_depth": 2, "consider_boundary": True, "count_boundary": True}),
        (4, "square", 12.5, (2.5, 2.5), {"level": "landscape", "neighbourhood": 8, "base": "log"}),
    ],
)
def test_sample_live(rscript, tmp_path, seed, shape, size, res, options):
    data = lsm_scenes.clustered((50, 60), [1, 2, 3, 4], seed=seed, sigma=1.5, na_fraction=0.05)
    tif = tmp_path / "in.tif"
    lsm_reference.write_landscape(tif, data, res)
    rng = np.random.default_rng(seed)
    # plots anywhere on the map, some across its edge
    plots = pd.DataFrame({"x": rng.uniform(0, 60 * res[0], 8), "y": rng.uniform(0, 50 * res[1], 8)})
    plots["plot_id"] = rng.permutation(8) + 1
    ref = lsm_reference.run_sample(rscript, tif, plots, shape=shape, size=size, **options)
    kwargs = {k: options[k] for k in options if k in SAMPLE_ARGS}
    metric_opts = {k: v for k, v in options.items() if k not in SAMPLE_ARGS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = ls.sample_metrics(data, plots[["x", "y"]].to_numpy(), size=size, shape=shape,
                                 plot_id=plots["plot_id"], res=res, **kwargs, **metric_opts)
    if options.get("neighbourhood") == 8:  # calculate_lsm's quirk, see test_lsm_reference
        ref = ref[~ref["metric"].isin(NEIGHBOURHOOD_QUIRK)].reset_index(drop=True)
        ours = ours[~ours["metric"].astype(str).isin(NEIGHBOURHOOD_QUIRK)].reset_index(drop=True)
    compare_sample(ours, ref, f"seed {seed}")


@pytest.mark.live_reference
@pytest.mark.parametrize(
    "seed, window, res, options",
    [
        (5, 3, (1.0, 1.0), {}),
        (6, "circle", (10.0, 10.0), {"directions": 4, "neighbourhood": 8, "ordered": False}),
    ],
)
def test_window_live(rscript, tmp_path, seed, window, res, options):
    data = lsm_scenes.clustered((9, 11), [1, 2, 3], seed=seed, sigma=1.0, na_fraction=0.08)
    tif = tmp_path / "in.tif"
    lsm_reference.write_landscape(tif, data, res)
    if window == "circle":
        window = np.hypot(*np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))) <= 2.2
    else:
        window = np.ones((window, window), bool)
    what = ["lsm_l_ta", "lsm_l_np", "lsm_l_shdi", "lsm_l_contag", "lsm_l_te", "lsm_l_enn_mn", "lsm_l_core_mn",
            "lsm_l_condent", "lsm_l_lsi", "lsm_l_circle_mn"]
    ref = lsm_reference.run_window(rscript, tif, window, what, **options)
    ours = ls.window_metrics(data, window, what=what, res=res, **options)
    for m in ref:
        assert_window_equal(np.asarray(ours[m]), ref[m], m, m)


# -- sample plots --------------------------------------------------------------


def georeferenced(values, res=10.0, left=1000.0, top=5000.0) -> xr.DataArray:
    da = xr.DataArray(np.asarray(values, dtype=np.int32), dims=("y", "x"))
    return da.rio.write_transform(from_origin(left, top, res, res)).rio.write_nodata(-1)


class TestSample:
    def test_square_plot_is_the_window_below_it(self):
        data = lsm_scenes.clustered((40, 50), [1, 2, 3], seed=1, sigma=2).data
        da = georeferenced(data)
        # a plot on cell edges: rows 10..21, columns 5..24
        x, y = 1000 + 150, 5000 - 160
        df = ls.sample_metrics(da, np.array([[x, y]]), size=np.array(60.0), shape="square", verbose=False)
        # square of half side 60 -> 12 x 12 cells
        sub = data[10:22, 9:21]
        expected = ls.Landscape(sub, res=10).metrics()
        pd.testing.assert_series_equal(df["value"], expected["value"], check_names=False)
        assert (df["percentage_inside"] == 100).all()

    def test_circle_mask_keeps_touched_cells(self):
        da = georeferenced(np.ones((20, 20)), res=1.0, left=0.0, top=20.0)
        df = ls.sample_metrics(da, np.array([[10.0, 10.0]]), size=3.0, shape="circle", what="lsm_l_ta")
        # the 100-vertex circle of radius 3 touches all 6 x 6 cells around it
        # (it passes 2.83 cells from the centre at the corner cells)
        assert df["value"].item() == pytest.approx(36 / 10000)
        assert df["percentage_inside"].item() == pytest.approx(36 / (np.pi * 9) * 100, rel=1e-3)
        small = ls.sample_metrics(da, np.array([[10.5, 10.5]]), size=0.4, shape="circle", what="lsm_l_ta")
        assert small["value"].item() == pytest.approx(1 / 10000)

    def test_plot_outside_the_raster(self):
        da = georeferenced(np.ones((5, 5)))
        with pytest.warns(UserWarning, match="percentage_inside"):
            df = ls.sample_metrics(da, np.array([[0.0, 0.0]]), size=5.0, what=["lsm_l_ta", "lsm_c_ca"])
        assert len(df) == 2
        assert df["value"].isna().all()
        assert df["class"].isna().all()
        assert (df["percentage_inside"] == 0).all()

    def test_plot_ids_order_the_table(self):
        da = georeferenced(lsm_scenes.clustered((30, 30), [1, 2], seed=2).data)
        pts = np.array([[1050.0, 4950.0], [1200.0, 4800.0], [1100.0, 4900.0]])
        df = ls.sample_metrics(da, pts, size=40.0, plot_id=["c", "a", "b"], what="lsm_l_np", verbose=False)
        assert df["plot_id"].tolist() == ["a", "b", "c"]
        with pytest.warns(UserWarning, match="plot_id"):
            df = ls.sample_metrics(da, pts, size=40.0, plot_id=[1, 2], what="lsm_l_np", verbose=False)
        assert df["plot_id"].tolist() == [1, 2, 3]

    def test_geometry_inputs(self):
        shapely = pytest.importorskip("shapely")
        da = georeferenced(lsm_scenes.clustered((30, 30), [1, 2, 3], seed=3).data)
        square = shapely.geometry.box(1040, 4840, 1160, 4960)
        a = ls.sample_metrics(da, [square], what="lsm_l_shdi", verbose=False)
        b = ls.sample_metrics(da, [square.__geo_interface__], what="lsm_l_shdi", verbose=False)
        c = ls.sample_metrics(da, np.array([[1100.0, 4900.0]]), size=60.0, what="lsm_l_shdi", verbose=False)
        d = ls.sample_metrics(da, [shapely.geometry.Point(1100, 4900)], size=60.0, what="lsm_l_shdi", verbose=False)
        for other in (b, c, d):
            assert other["value"].item() == a["value"].item()
        with pytest.raises(ValueError, match="size"):
            ls.sample_metrics(da, np.array([[1100.0, 4900.0]]), what="lsm_l_shdi")

    def test_all_classes(self):
        values = np.ones((20, 20), dtype=int)
        values[:, 10:] = 2
        values[0, 0] = -1  # NA somewhere in the raster
        da = georeferenced(values)
        df = ls.sample_metrics(da, np.array([[1030.0, 4900.0]]), size=20.0, what="lsm_c_ca", all_classes=True)
        # the plot holds class 1 only; class 2 and NA get NA rows
        assert df["class"].tolist()[:2] == [1, 2]
        assert df["class"].isna().tolist() == [False, False, True]
        assert np.isnan(df["value"].iloc[1]) and np.isnan(df["value"].iloc[2])

    def test_several_rasters(self):
        a = georeferenced(lsm_scenes.clustered((30, 30), [1, 2], seed=4).data)
        b = georeferenced(lsm_scenes.clustered((30, 30), [1, 2, 3], seed=5).data)
        pts = np.array([[1100.0, 4900.0], [1200.0, 4850.0]])
        both = ls.sample_metrics([a, b], pts, size=40.0, what="lsm_l_np", verbose=False)
        assert both["layer"].tolist() == [1, 1, 2, 2]
        one = ls.sample_metrics(b, pts, size=40.0, what="lsm_l_np", verbose=False)
        np.testing.assert_array_equal(both["value"].to_numpy()[2:], one["value"].to_numpy())


# -- moving windows ------------------------------------------------------------


def brute_window(data, window, res, what, **options):
    """Every window as a Landscape of its own (shdi and shei with 8
    directions: window_lsm computes them from 8-connected patch areas),
    transposed with the cell sides swapped as window_metrics reads it (the
    patch order of window_lsm)."""
    values = np.ma.getdata(data)
    mask = np.ma.getmaskarray(data)
    hy, hx = window.shape[0] // 2, window.shape[1] // 2
    padded = np.pad(values, ((hy, hy), (hx, hx)))
    padded_mask = np.pad(mask, ((hy, hy), (hx, hx)), constant_values=True)
    out = {m: np.full(values.shape, np.nan) for m in what}
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            sub = padded[y : y + window.shape[0], x : x + window.shape[1]]
            sub_mask = padded_mask[y : y + window.shape[0], x : x + window.shape[1]] | ~window
            sub_t = np.ma.MaskedArray(sub.T, mask=sub_mask.T)
            res_t = (res[1], res[0])
            lm = ls.Landscape(sub_t, res=res_t, **options)
            lm8 = ls.Landscape(sub_t, res=res_t, **{**options, "directions": 8})
            for m in what:
                out[m][y, x] = (lm8 if m in ("shdi", "shei") else lm).landscape_metrics(m)[m].item()
    return out


class TestWindow:
    @pytest.mark.parametrize(
        "window, res, options",
        [
            (np.ones((3, 3), bool), (1.0, 1.0), {}),
            (np.ones((3, 5), bool), (30.0, 20.0), {"directions": 4}),  # non-square window and cells
            (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool), (5.0, 5.0), {"edge_depth": 0}),
        ],
    )
    def test_brute_force(self, window, res, options):
        data = lsm_scenes.clustered((13, 16), [1, 2, 3], seed=7, sigma=1.0, na_fraction=0.1)
        what = ["ta", "np", "shdi", "contag", "ed", "enn_mn", "core_mn", "gyrate_mn", "iji"]
        ours = ls.window_metrics(data, window, what=["lsm_l_" + m for m in what], res=res, **options)
        expected = brute_window(data, window, res, what, **options)
        for m in what:
            np.testing.assert_array_equal(ours[m], expected[m], err_msg=m)

    def test_single_cell_window(self):
        data = np.ma.MaskedArray(np.array([[1, 2], [3, 4]]), mask=[[False, True], [False, False]])
        out = ls.window_metrics(data, 1, what=["lsm_l_ta", "lsm_l_np"], res=10)
        np.testing.assert_array_equal(out["ta"], [[0.01, np.nan], [0.01, 0.01]])
        np.testing.assert_array_equal(out["np"], [[1, np.nan], [1, 1]])

    def test_default_is_every_landscape_metric(self):
        out = ls.window_metrics(lsm_scenes.clustered((6, 7), [1, 2], seed=1).data, 3)
        assert list(out) == list(ls.list_metrics(level="landscape")["metric"])

    def test_dataarray_output(self):
        da = georeferenced(lsm_scenes.clustered((12, 14), [1, 2, 3], seed=8).data)
        out = ls.window_metrics(da, 5, what="lsm_l_shdi")
        assert isinstance(out["shdi"], xr.DataArray)
        assert out["shdi"].rio.transform() == da.rio.transform()
        assert out["shdi"].shape == da.shape

    def test_several_rasters(self):
        a = lsm_scenes.clustered((8, 9), [1, 2], seed=1).data
        out = ls.window_metrics([a, a.T], 3, what="lsm_l_np")
        assert len(out) == 2
        np.testing.assert_array_equal(out[1]["np"], out[0]["np"].T)

    def test_errors(self):
        data = np.ones((5, 5), dtype=int)
        with pytest.raises(ValueError, match="odd"):
            ls.window_metrics(data, 4)
        with pytest.raises(ValueError, match="odd"):
            ls.window_metrics(data, np.ones((3, 4)))
        with pytest.raises(ValueError, match="landscape-level"):
            ls.window_metrics(data, 3, what="lsm_c_ca")

    def test_threads_do_not_change_results(self):
        from threadpoolctl import threadpool_limits

        data = lsm_scenes.clustered((60, 70), [1, 2, 3, 4], seed=9, na_fraction=0.05)
        a = ls.window_metrics(data, 7, what=["lsm_l_shdi", "lsm_l_enn_mn", "lsm_l_core_mn"])
        with threadpool_limits(limits=1, user_api="openmp"):
            b = ls.window_metrics(data, 7, what=["lsm_l_shdi", "lsm_l_enn_mn", "lsm_l_core_mn"])
        for m in a:
            np.testing.assert_array_equal(a[m], b[m])


# -- several rasters and the long table ------------------------------------------


class TestLayers:
    def test_layers_match_single_landscapes(self):
        small = lsm_scenes.clustered((30, 40), [1, 2, 3], seed=1, na_fraction=0.05)
        large = lsm_scenes.clustered((300, 250), [1, 2, 3, 4], seed=2)  # computed on its own, in parallel
        both = ls.landscape_metrics([small, large, small])
        for layer, data in enumerate([small, large, small], start=1):
            part = both[both["layer"] == layer].drop(columns="layer").reset_index(drop=True)
            pd.testing.assert_frame_equal(part, ls.Landscape(data).metrics())

    def test_all_na_landscape_gives_one_row_per_metric(self):
        df = ls.Landscape(np.ma.masked_all((3, 4), dtype=np.int32)).metrics()
        assert len(df) == len(ls.list_metrics())
        assert df["value"].isna().all()
        assert df["class"].isna().all() and df["id"].isna().all()
