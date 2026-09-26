# SPDX-License-Identifier: GPL-3.0-or-later
"""Writing landscape-metric results: tables, zonal GeoDataFrames, metric
rasters and batch processing."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import rasterio

import landschaft as ls
import lsm_reference
import lsm_scenes

DATA = lsm_scenes.clustered((40, 50), [1, 2, 3], seed=4, na_fraction=0.05)


@pytest.fixture
def tif(tmp_path):
    path = tmp_path / "map.tif"
    lsm_reference.write_landscape(path, DATA, (10.0, 10.0), crs="EPSG:3857")
    return path


class TestWriteTable:
    def test_csv_and_parquet_round_trip(self, tmp_path):
        pytest.importorskip("pyarrow")
        df = ls.Landscape(DATA).metrics()
        ls.write_table(df, tmp_path / "m.parquet")
        back = pd.read_parquet(tmp_path / "m.parquet")
        pd.testing.assert_frame_equal(back, df)
        ls.write_table(df, tmp_path / "m.csv.gz")
        csv = pd.read_csv(tmp_path / "m.csv.gz", float_precision="round_trip")
        np.testing.assert_array_equal(csv["value"].to_numpy(), df["value"].to_numpy())

    def test_errors(self, tmp_path):
        df = ls.Landscape(DATA).metrics(level="landscape")
        with pytest.raises(ValueError, match="format"):
            ls.write_table(df, tmp_path / "m.xyz")
        with pytest.raises(TypeError, match="GeoDataFrame"):
            ls.write_table(df, tmp_path / "m.gpkg")


class TestZonalMetrics:
    def zones(self):
        gpd = pytest.importorskip("geopandas")
        from shapely.geometry import box

        return gpd.GeoDataFrame({"name": ["a", "b", "c"]},
                                geometry=[box(0, 0, 200, 150), box(150, 100, 480, 390), box(300, 0, 500, 120)],
                                crs="EPSG:3857")

    def test_matches_sample_metrics(self, tif, tmp_path):
        zones = self.zones()
        z = ls.zonal_metrics(tif, zones, level=["class", "landscape"], metric=["ca", "np", "shdi", "ta"])
        assert list(z["name"]) == ["a", "b", "c"] and z.crs == zones.crs
        s = ls.sample_metrics(tif, [g.__geo_interface__ for g in zones.geometry], level=["class", "landscape"],
                              metric=["ca", "np", "shdi", "ta"], verbose=False)
        for _, row in s.iterrows():
            col = row["metric"] if row["level"] == "landscape" else f"{row['metric']}_{row['class']}"
            ours = z.loc[row["plot_id"] - 1, col]
            assert (np.isnan(ours) and np.isnan(row["value"])) or ours == row["value"]
        np.testing.assert_array_equal(z["percentage_inside"], s.groupby("plot_id")["percentage_inside"].first())
        path = ls.write_table(z, tmp_path / "zones.gpkg")
        import geopandas as gpd

        back = gpd.read_file(path)
        assert list(back.columns) == list(z.columns)

    def test_errors(self, tif):
        zones = self.zones()
        with pytest.raises(ValueError, match="patch"):
            ls.zonal_metrics(tif, zones, level="patch")
        with pytest.raises(ValueError, match="polygons"):
            ls.zonal_metrics(tif, [{"type": "Point", "coordinates": (5, 5)}])


class TestToRaster:
    def test_class_metric_raster(self):
        lm = ls.Landscape(DATA)
        pland = lm.class_metrics("pland").set_index("class")["pland"]
        r = lm.to_raster("pland")
        values = np.ma.getdata(DATA)
        mask = np.ma.getmaskarray(DATA)
        np.testing.assert_array_equal(r[~mask], pland.loc[values[~mask]].to_numpy())
        assert np.isnan(r[mask]).all()
        np.testing.assert_array_equal(lm.to_raster("np", level="class")[~mask] > 0, True)

    def test_write(self, tif, tmp_path):
        lm = ls.Landscape(tif)
        path = lm.to_raster("area", path=tmp_path / "area.tif")
        with rasterio.open(path) as src, rasterio.open(tif) as ref:
            assert src.transform == ref.transform and src.crs == ref.crs
            np.testing.assert_array_equal(src.read(1), np.asarray(lm.to_raster("area")))
        paths = lm.to_raster(["area", "ca"], path=tmp_path / "out")
        assert sorted(p.name for p in paths.values()) == ["area.tif", "ca.tif"]
        # numpy input: the extent of the Landscape
        arr = ls.Landscape(np.ma.getdata(DATA), res=5).to_raster("area", path=tmp_path / "np.tif")
        with rasterio.open(arr) as src:
            assert src.transform.a == 5 and src.transform.f == 40 * 5


class TestScaleSample:
    def test_sizes_and_order(self, tif):
        pts = np.array([[250.0, 200.0], [100.0, 300.0]])
        out = ls.scale_sample(tif, pts, size=[50, 100], level="landscape", metric=["ta", "np"])
        assert out["size"].tolist() == [50, 100, 50, 100, 50, 100, 50, 100]
        assert out["plot_id"].tolist() == [1] * 4 + [2] * 4
        one = ls.sample_metrics(tif, pts, size=100, level="landscape", metric=["ta", "np"], verbose=False)
        np.testing.assert_array_equal(out.loc[out["size"] == 100, "value"], one["value"])
        with pytest.raises(ValueError, match="points"):
            ls.scale_sample(tif, [{"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 0)]]}], size=5)


class TestBatch:
    def test_landscape_metrics(self, tif, tmp_path):
        out = tmp_path / "out"
        res = ls.batch("landscape_metrics", [tif], out, level="landscape", metric=["ta", "shdi"])
        assert res[0].ok, res[0].error
        table = pd.read_csv(out / "map_landscape_metrics.csv", float_precision="round_trip")
        expected = ls.landscape_metrics(tif, level="landscape", metric=["ta", "shdi"])
        np.testing.assert_array_equal(table["value"], expected["value"])
        meta = json.loads((out / "map_landscape_metrics.json").read_text())
        assert meta["tool"] == "landscape_metrics"

    def test_parquet_and_block_rows(self, tif, tmp_path):
        pytest.importorskip("pyarrow")
        out = tmp_path / "pq"
        res = ls.batch("landscape_metrics", [tif], out, table_format="parquet", block_rows=64, level="class")
        assert res[0].ok, res[0].error
        table = pd.read_parquet(out / "map_landscape_metrics.parquet")
        expected = ls.landscape_metrics(tif, level="class")
        np.testing.assert_array_equal(table["value"].to_numpy(), expected["value"].to_numpy())

    def test_window_metrics(self, tif, tmp_path):
        res = ls.batch("window_metrics", [tif], tmp_path / "w", window=3, what=["lsm_l_shdi"])
        assert res[0].ok, res[0].error
        assert (tmp_path / "w" / "map_window_metrics_shdi.tif").exists()
