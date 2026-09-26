# SPDX-License-Identifier: GPL-3.0-or-later
"""FRAGSTATS extras of the landscape metrics (_am, _md, _ra, perim_*), the
metric-name aliases and n_threads."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
import lsm_scenes
from conftest import DATA

FAMILIES = ["area", "cai", "circle", "contig", "core", "dcore", "enn", "frac", "gyrate", "para", "perim", "shape"]

MAPS = {
    "clustered": (lsm_scenes.clustered((60, 70), [1, 2, 3, 5], seed=1, na_fraction=0.05), (30.0, 30.0)),
    "salt_pepper": (lsm_scenes.salt_and_pepper((35, 40), [1, 2, 3], seed=2, na_fraction=0.05), (10.0, 10.0)),
    "sparse": (lsm_scenes.sparse((50, 60), [2, 3], seed=3, n_blobs=15, background=1), (2.0, 3.0)),
}


def numpy_stat(stat: str, x: np.ndarray, w: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    if stat == "am":
        return float(np.sum(x * w) / np.sum(w))
    if stat == "md":
        return float(np.median(x))
    if stat == "ra":
        return float(np.max(x) - np.min(x))
    if stat == "mn":
        return float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    return sd if stat == "sd" else sd / np.mean(x) * 100


class TestSummaryStatistics:
    @pytest.mark.parametrize("case", sorted(MAPS))
    @pytest.mark.parametrize("stat", ["am", "md", "ra"])
    def test_class_and_landscape_match_numpy(self, case, stat):
        data, res = MAPS[case]
        lm = ls.Landscape(data, res=res)
        patches = lm.patch_metrics(["area", "ncore"] + [f for f in FAMILIES if f != "dcore"])
        patches["dcore"] = patches["ncore"]  # landscapemetrics' dcore: core areas per patch
        names = [f"{f}_{stat}" for f in FAMILIES]
        cls = lm.class_metrics(names)
        land = lm.landscape_metrics(names)
        for f in FAMILIES:
            for c, row in cls.iterrows():
                sub = patches[patches["class"] == row["class"]]
                expected = numpy_stat(stat, sub[f].to_numpy(), sub["area"].to_numpy())
                np.testing.assert_allclose(row[f"{f}_{stat}"], expected, rtol=1e-13, err_msg=f"{case} {f} {c}")
            expected = numpy_stat(stat, patches[f].to_numpy(), patches["area"].to_numpy())
            np.testing.assert_allclose(land[f"{f}_{stat}"][0], expected, rtol=1e-13, err_msg=f"{case} {f}")

    def test_perim_statistics(self):
        data, res = MAPS["clustered"]
        lm = ls.Landscape(data, res=res)
        perim = lm.patch_metrics("perim")
        cls = lm.class_metrics(["perim_mn", "perim_sd", "perim_cv"])
        for _, row in cls.iterrows():
            x = perim.loc[perim["class"] == row["class"], "perim"].to_numpy()
            np.testing.assert_allclose(row["perim_mn"], x.mean(), rtol=1e-14)
            np.testing.assert_allclose(row["perim_sd"], x.std(ddof=1), rtol=1e-13)
            np.testing.assert_allclose(row["perim_cv"], x.std(ddof=1) / x.mean() * 100, rtol=1e-13)

    def test_na_propagates(self):
        # class 2 has a single patch: its enn is NA, and so are its statistics
        data = np.array([[1, 1, 2, 2], [1, 3, 2, 2], [3, 3, 1, 3]])
        lm = ls.Landscape(data)
        cls = lm.class_metrics(["enn_am", "enn_md", "enn_ra"]).set_index("class")
        assert cls.loc[2].isna().all()
        assert np.isfinite(cls.loc[1]).all()
        assert lm.landscape_metrics(["enn_am", "enn_md", "enn_ra"]).isna().all(axis=None)

    def test_hand_computed(self):
        # 1 1 2 / 1 2 2 / 3 3 2: patches of 3, 4 and 2 cells
        lm = ls.Landscape(np.array([[1, 1, 2], [1, 2, 2], [3, 3, 2]]), res=100)
        land = lm.landscape_metrics(["area_am", "area_md", "area_ra"])
        a = np.array([3, 4, 2.0])
        assert land["area_am"][0] == pytest.approx((a * a).sum() / a.sum())
        assert land["area_md"][0] == pytest.approx(3)
        assert land["area_ra"][0] == pytest.approx(2)
        # median of an even number of patches: mean of the two middle ones
        lm2 = ls.Landscape(np.array([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]]))
        assert lm2.landscape_metrics("area_md")["area_md"][0] == pytest.approx((2 + 3) / 2 / 10000)

    def test_gyrate_am_is_correlation_length(self):
        lm = ls.Landscape(MAPS["clustered"][0], res=30)
        p = lm.patch_metrics(["gyrate", "area"])
        expected = (p["gyrate"] * p["area"]).sum() / p["area"].sum()
        assert lm.landscape_metrics("gyrate_am")["gyrate_am"][0] == pytest.approx(expected, rel=1e-13)


@pytest.fixture(scope="module")
def pylandstats():
    return pytest.importorskip("pylandstats")


class TestAgainstPylandstats:
    """pylandstats' distribution statistics of the patch metrics whose patch
    values it computes as landscapemetrics does (area, perimeter, fractal
    dimension; ENN and core area when they agree)."""

    PAIRS = {
        "area": "area",
        "perim": "perimeter",
        "frac": "fractal_dimension",
        "enn": "euclidean_nearest_neighbor",
        "core": "core_area",
    }

    @pytest.mark.parametrize("case", sorted(MAPS))
    def test_distribution_statistics(self, case, pylandstats):
        data, res = MAPS[case]
        res = (res[0], res[0])  # non-square cells: perimeter conventions differ
        values = np.where(np.ma.getmaskarray(data), 0, np.ma.getdata(data))  # pylandstats: 0 = nodata
        lm = ls.Landscape(values, res=res, nodata=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pl = pylandstats.Landscape(values, res=res, nodata=0, neighborhood_rule="8")
            pp = pl.compute_patch_metrics_df()
        ours = lm.patch_metrics(list(self.PAIRS))
        compared = 0
        for fam, pl_name in self.PAIRS.items():
            a = np.sort(ours[fam].to_numpy())
            b = np.sort(pp[pl_name].to_numpy())
            if not np.allclose(a, b, rtol=1e-12, equal_nan=True):
                assert fam in ("enn", "core"), f"{case}: patch {fam} differs from pylandstats"
                continue  # different definitions there (edge-to-edge distance, boundary)
            compared += 1
            for stat in ("mn", "am", "md", "ra"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    theirs_cls = pl.compute_class_metrics_df(metrics=[f"{pl_name}_{stat}"])
                    theirs_land = pl.compute_landscape_metrics_df(metrics=[f"{pl_name}_{stat}"])
                cls = lm.class_metrics(f"{fam}_{stat}").set_index("class")[f"{fam}_{stat}"]
                np.testing.assert_allclose(
                    cls.to_numpy(), theirs_cls.loc[cls.index, f"{pl_name}_{stat}"].to_numpy(dtype=float), rtol=1e-12,
                    equal_nan=True, err_msg=f"{case} class {fam}_{stat}",
                )
                ours_land = lm.landscape_metrics(f"{fam}_{stat}")[f"{fam}_{stat}"][0]
                theirs = float(theirs_land[f"{pl_name}_{stat}"].iloc[0])
                if np.isnan(ours_land):
                    continue  # pylandstats (pandas) skips NA; it propagates here, as in landscapemetrics
                np.testing.assert_allclose(ours_land, theirs, rtol=1e-12, err_msg=f"{case} landscape {fam}_{stat}")
        assert compared >= 3


class TestSelection:
    def test_extras_not_selected_by_default(self):
        default = ls.list_metrics()
        assert len(default) == 133 and not default["extra"].any()
        full = ls.list_metrics(extras=True)
        extra = full[full["extra"]]
        stats = extra[extra["metric"].str.contains("_")]
        assert set(stats["metric"].str.split("_").str[1]) == {"am", "md", "ra", "mn", "sd", "cv"}
        assert set(stats["level"]) == {"class", "landscape"}
        assert set(extra.loc[~extra["metric"].str.contains("_"), "metric"]) == {
            "prox", "simi", "econ", "connect", "cwed", "teci",
        }
        assert ls.list_metrics(metric="area", level="class", extras=True)["metric"].tolist() == [
            "area_am", "area_cv", "area_md", "area_mn", "area_ra", "area_sd",
        ]

    def test_extras_in_what(self):
        out = ls.list_metrics(what=["lsm_c_area_am", "lsm_l_perim_mn", "lsm_l_ta"])
        assert out["function_name"].tolist() == ["lsm_c_area_am", "lsm_l_perim_mn", "lsm_l_ta"]

    def test_long_table_with_extras(self):
        lm = ls.Landscape(MAPS["clustered"][0])
        df = lm.metrics(level="landscape", metric="shape", extras=True)
        assert df["metric"].tolist() == ["shape_am", "shape_cv", "shape_md", "shape_mn", "shape_ra", "shape_sd"]
        wide = lm.landscape_metrics(list(df["metric"]))
        np.testing.assert_array_equal(df["value"].to_numpy(), wide.iloc[0].to_numpy())
        assert len(lm.metrics()) < len(lm.metrics(extras=True))

    def test_batch_sample_window_accept_extras(self):
        data = MAPS["salt_pepper"][0]
        one = ls.landscape_metrics(data, what=["lsm_l_area_am", "lsm_c_frac_md"])
        two = ls.landscape_metrics([data, data], what=["lsm_l_area_am", "lsm_c_frac_md"])
        np.testing.assert_array_equal(two.loc[two["layer"] == 2, "value"].to_numpy(), one["value"].to_numpy())
        maps = ls.window_metrics(data, 5, what=["lsm_l_area_ra"])
        lm = ls.Landscape(np.ma.MaskedArray(data.data[8:13, 8:13], data.mask[8:13, 8:13]))
        assert maps["area_ra"][10, 10] == lm.landscape_metrics("area_ra")["area_ra"][0]


class TestAliases:
    def test_fragstats_abbreviations(self):
        lm = ls.Landscape(MAPS["clustered"][0], res=30)
        cls = lm.class_metrics(["CA", "PLAND", "AREA_MN", "ENN_AM"])
        ref = lm.class_metrics(["ca", "pland", "area_mn", "enn_am"])
        assert cls.columns.tolist() == ["class", "CA", "PLAND", "AREA_MN", "ENN_AM"]
        np.testing.assert_array_equal(cls.iloc[:, 1:].to_numpy(), ref.iloc[:, 1:].to_numpy())
        land = lm.landscape_metrics(["TA", "SHDI", "CONTAG"])
        np.testing.assert_array_equal(land.to_numpy(), lm.landscape_metrics(["ta", "shdi", "contag"]).to_numpy())

    def test_full_names(self):
        lm = ls.Landscape(MAPS["clustered"][0], res=30)
        pairs = {
            "patch": {"perimeter": "perim", "shape_index": "shape", "euclidean_nearest_neighbor": "enn"},
            "class": {"total_area": "ca", "number_of_patches": "np", "shape_index_am": "shape_am",
                      "disjunct_core_area_mn": "dcore_mn", "perimeter_area_ratio_md": "para_md"},
            "landscape": {"total_area": "ta", "shannon_diversity_index": "shdi", "contagion": "contag",
                          "effective_mesh_size": "mesh", "fractal_dimension_ra": "frac_ra"},
        }
        getters = {"patch": lm.patch_metrics, "class": lm.class_metrics, "landscape": lm.landscape_metrics}
        for level, names in pairs.items():
            a = getters[level](list(names))
            b = getters[level](list(names.values()))
            np.testing.assert_array_equal(a[list(names)].to_numpy(), b[list(names.values())].to_numpy())

    def test_family_alias_in_filters(self):
        assert set(ls.list_metrics(metric="shape_index")["metric"]) == {"shape", "shape_mn", "shape_sd", "shape_cv"}

    def test_to_raster_alias(self):
        lm = ls.Landscape(MAPS["clustered"][0])
        np.testing.assert_array_equal(lm.to_raster("AREA"), lm.to_raster("area"))

    def test_unknown_names(self):
        lm = ls.Landscape(MAPS["clustered"][0])
        with pytest.raises(ValueError, match="landscape level"):
            lm.class_metrics("shannon_diversity_index")
        with pytest.raises(ValueError, match="not class-level"):
            lm.class_metrics("no_such_metric")


class TestThreads:
    def test_same_values_with_any_number_of_threads(self):
        data = lsm_scenes.clustered((300, 280), [1, 2, 3, 4], seed=9, na_fraction=0.05)
        full = ls.Landscape(data).metrics()
        one = ls.Landscape(data, n_threads=1).metrics()
        pd.testing.assert_frame_equal(full, one)
        two = ls.landscape_metrics([data, data], n_threads=2, level="landscape")
        np.testing.assert_array_equal(two.loc[two["layer"] == 1, "value"], full.loc[full["level"] == "landscape", "value"])

    def test_invalid(self):
        with pytest.raises(ValueError, match="n_threads"):
            ls.Landscape(np.ones((3, 3)), n_threads=-1)


def test_out_of_core_extras(tmp_path):
    import lsm_reference

    data, res = MAPS["sparse"]
    tif = tmp_path / "m.tif"
    lsm_reference.write_landscape(tif, data, res)
    names = ["area_am", "gyrate_am", "enn_md", "core_ra", "perim_cv"]
    tiled = ls.Landscape(tif, block_rows=64, n_threads=2).landscape_metrics(names)
    mem = ls.Landscape(tif).landscape_metrics(names)
    pd.testing.assert_frame_equal(tiled, mem)


def test_reference_maps_extras_are_finite():
    # every extra is computed on the landscapemetrics reference maps
    lm = ls.Landscape(DATA / "lsm" / "augusta.tif")
    df = lm.metrics(extras=True)
    extra = set(ls.list_metrics(extras=True).query("extra")["metric"])
    sub = df[df["metric"].isin(extra)]
    assert set(sub["metric"]) == extra
    # NA only where landscapemetrics' statistics are NA too (enn of single patches, sd of one value)
    # (prox, simi, connect and the contrast metrics need their parameters: NaN here)
    needs = ("enn", "prox", "simi", "connect", "econ", "cwed", "teci")
    plain = ~sub["metric"].str.startswith(needs) & ~sub["metric"].str.endswith(("_sd", "_cv"))
    assert sub.loc[plain, "value"].notna().all()
