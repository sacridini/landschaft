# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.download without the network (Terraflux's tests of data.download and data.environment90m): the sources'
paths, rasters warped onto a grid, tiles mosaicked, the cache, a STAC catalogue and Environment90m's tables served by
fakes from local files. The downloads themselves with LANDSCHAFT_ONLINE_TESTS=1 (they read the servers)."""

from __future__ import annotations

import os
import zipfile

import numpy as np
import pytest
import rasterio
import xarray as xr
from affine import Affine

import landschaft as ls
from landschaft import download as D

ONLINE = pytest.mark.skipif(os.environ.get("LANDSCHAFT_ONLINE_TESTS") != "1",
                            reason="downloads from the servers (LANDSCHAFT_ONLINE_TESTS=1)")


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LANDSCHAFT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LANDSCHAFT_NUM_THREADS", "2")
    return tmp_path / "cache"


def write(path, values, transform, crs="EPSG:4326", nodata=None):
    values = np.asarray(values, dtype=np.float32)
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(values, 1)
    return str(path)


def test_the_sources_paths():
    w = D._worldclim("bio", "10m", [1, 12])
    assert w[1].paths == [("/vsizip//vsicurl/https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip/"
                           "wc2.1_10m_bio_12.tif")]
    assert w[1].name == "bio12"
    assert D._worldclim("tavg", "2.5m", [3])[0].paths[0].endswith("wc2.1_2.5m_tavg.zip/wc2.1_2.5m_tavg_03.tif")
    f = D._worldclim_future("bioc", "10m", "ACCESS-CM2", "ssp245", "2041-2060", [5])[0]
    assert f.paths == [("/vsicurl/https://geodata.ucdavis.edu/cmip6/10m/ACCESS-CM2/ssp245/"
                        "wc2.1_10m_bioc_ACCESS-CM2_ssp245_2041-2060.tif")] and f.band == 5
    assert D._chelsa([1])[0].paths[0].endswith("/1981-2010/bio/CHELSA_bio1_1981-2010_V.2.1.tif")
    assert D._chelsa([1], ("GFDL-ESM4", "ssp126", "2041-2070"))[0].paths[0].endswith(
        "/2041-2070/GFDL-ESM4/ssp126/bio/CHELSA_bio1_2041-2070_gfdl-esm4_ssp126_V.2.1.tif")
    # a box across the equator and the meridian of Greenwich: four tiles
    t = D._copernicus_tiles(90, [-0.5, 0.5], [-0.5, 0.5])
    assert len(t) == 4
    assert t[0].endswith("Copernicus_DSM_COG_30_S01_00_W001_00_DEM/Copernicus_DSM_COG_30_S01_00_W001_00_DEM.tif")
    assert "copernicus-dem-90m" in t[3] and "_N00_00_E000_00_" in t[3]
    assert D._soilgrids("soc", ["0-5cm"], "mean")[0].paths[0].endswith("/data/soc/soc_0-5cm_mean.vrt")
    m = D._mapbiomas(11, [2025])[0]
    assert m.classes and m.paths[0].endswith("collection11/lulc/coverage/brazil_coverage/brazil_coverage-col11_2025.tif")
    with pytest.raises(ValueError, match="collection"):
        D._mapbiomas(8, [2000])
    # Hydrography90m's tiles of 20 degrees: h10v04 from 80 to 60 W and 45 to 25 N
    assert D._h90m_tiles([-79.0, -61.0], [26.0, 44.0]) == ["h10v04"]
    assert D._h90m_tiles([-47.2, -46.8], [-23.0, -22.7]) == ["h12v10"]
    assert D._h90m_tiles([-61.0, -59.0], [44.0, 46.0]) == ["h10v02", "h12v02", "h10v04", "h12v04"]
    assert len(D._h90m_tiles([-180.0, 180.0], [-60.0, 85.0])) == 18 * 7
    paths, classes = D._h90m("sub_catchment", ["h12v10"])
    assert classes and paths[0] == ("/vsicurl/https://public.igb-berlin.de/index.php/s/agciopgzXjWswF4/download?path="
                                    "%2Fr.watershed%2Fsub_catchment_tiles20d%2Fsub_catchment_h12v10.tif")
    paths, classes = D._h90m("slope_grad_dw_cel", ["h00v00"])
    assert not classes and "%2Fr.stream.slope%2Fslope_grad_dw_cel_tiles20d%2F" in paths[0]
    assert D._h90m("order_strahler", [])[1]
    with pytest.raises(ValueError, match="rivers"):
        D._h90m("rivers", [])
    assert D._variable_and_tile("esa_cci_landcover_v2_1_1/c100/c100_1992_h00v02.zip") == ("c100_1992", "h00v02")
    assert D._variable_and_tile("chelsa_bioclim_v2_1/2041-2070_projected/bio01/"
                                "bio01_2041-2070_ipsl-cm6a-lr_ssp126_v2_1_h34v12.zip") == (
        "bio01_2041-2070_ipsl-cm6a-lr_ssp126_v2_1", "h34v12")
    assert D._variable_and_tile("README.txt") is None


def test_a_raster_warped_onto_a_grid(tmp_path, cache):
    # 20 x 10 cells of 1 degree, the value the column, onto cells of 2 degrees: the mean of two columns
    path = write(tmp_path / "columns.tif", np.tile(np.arange(20.0), (10, 1)), Affine(1, 0, -10, 0, -1, 5))
    r = ls.download.raster(path, extent=(-10, -5, 10, 5), cellsize=2, resampling="average")
    assert r.shape == (5, 10) and r.name == "layer"
    assert r.values[0].tolist() == [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 18.5]
    assert r.rio.crs.to_epsg() == 4326 and float(r.x[0]) == -9.0 and float(r.y[0]) == 4.0
    # onto a map's grid: its coordinates
    like = xr.DataArray(np.zeros((2, 4)), dims=("y", "x"), coords={"y": [2.5, -2.5], "x": [-7.5, -2.5, 2.5, 7.5]})
    like = like.rio.write_crs("EPSG:4326")
    r = ls.download.raster(path, like, resampling="near")
    assert r.shape == (2, 4) and (r.x == like.x).all() and (r.y == like.y).all()
    # in the cache: read again from it, not warped
    assert len(list(cache.glob("data/*/*.tif"))) == 2
    calls = []
    real = D._warp
    D_warp = lambda *a, **k: calls.append(1) or real(*a, **k)
    try:
        D._warp = D_warp
        again = ls.download.raster(path, like, resampling="near")
        np.testing.assert_array_equal(again.values, r.values)
        assert not calls
        # a file changed: warped again
        write(path, np.tile(np.arange(20.0) * 2, (10, 1)), Affine(1, 0, -10, 0, -1, 5))
        changed = ls.download.raster(path, like, resampling="near")
        assert calls and changed.values[0, 0] == 2 * r.values[0, 0]
    finally:
        D._warp = real
    # several: a band each
    two = ls.download.raster([path, path], like, resampling="near")
    assert two.dims == ("band", "y", "x") and list(two.band.values) == ["layer01", "layer02"]
    with pytest.raises(ValueError, match="extent and cellsize"):
        ls.download.raster(path)
    with pytest.raises(ValueError, match="resampling"):
        ls.download.raster(path, like, resampling="nearest-ish")


def test_tiles_are_mosaicked(tmp_path, monkeypatch):
    # two tiles of a degree side by side, and the sea (no tile) beside them
    a = write(tmp_path / "a.tif", np.full((10, 10), 100.0), Affine(0.1, 0, 0, 0, -0.1, 1), nodata=-9999)
    b = write(tmp_path / "b.tif", np.full((10, 10), 200.0), Affine(0.1, 0, 1, 0, -0.1, 1), nodata=-9999)
    seen = {}

    def tiles(res, lon, lat):
        seen["box"] = (lon, lat)
        return [a, b, str(tmp_path / "missing.tif")]

    monkeypatch.setattr(D, "_copernicus_tiles", tiles)
    dem = ls.download.copernicus_dem(extent=(0.5, 0.5, 2.5, 1.0), cellsize=0.25, resampling="near")
    assert seen["box"][0] == [0.5, 2.5] and seen["box"][1] == [0.5, 1.0]
    np.testing.assert_array_equal(dem.values[0], [100, 100, 200, 200, 200, 200, np.nan, np.nan])
    monkeypatch.setattr(D, "_copernicus_tiles", lambda *a: [str(tmp_path / "missing.tif")])
    with pytest.raises(ValueError, match="no tile"):
        ls.download.copernicus_dem(extent=(0.5, 0.5, 2.5, 1.0), cellsize=0.25)


def test_a_stac_catalogue(tmp_path, monkeypatch):
    t = Affine(0.5, 0, 0, 0, -0.5, 2)
    scenes = [write(tmp_path / f"s{k}.tif", np.full((4, 4), float(k)), t) for k in (1, 2, 3)]
    clouded = np.full((4, 4), 4.0)
    clouded[:2] = np.nan
    scenes.append(write(tmp_path / "s4.tif", clouded, t, nodata=np.nan))
    items = [{"id": f"i{k}", "assets": {"red": {"href": s}}} for k, s in enumerate(scenes)]
    bodies = []

    def post(url, body):
        bodies.append((url, body))
        return {"features": items[:2], "links": [{"rel": "next", "href": "page2"}]}

    monkeypatch.setattr(D, "_post", post)
    monkeypatch.setattr(D, "_get", lambda url: {"features": items[2:], "links": []})
    kw = {"extent": (0, 0, 2, 2), "cellsize": 0.5, "resampling": "near"}
    r = ls.download.stac("https://stac.example/v1", "s2", assets="red", datetime="2024-01-01/2024-12-31",
                         query={"eo:cloud_cover": {"lt": 20}}, **kw)
    url, body = bodies[0]
    assert url == "https://stac.example/v1/search"
    assert body["collections"] == ["s2"] and body["bbox"] == [0.0, 0.0, 2.0, 2.0]
    assert body["datetime"] == "2024-01-01T00:00:00Z/2024-12-31T23:59:59Z"
    assert body["query"] == {"eo:cloud_cover": {"lt": 20}}
    assert D._rfc3339("2024-05-02") == "2024-05-02T00:00:00Z/2024-05-02T23:59:59Z"
    assert D._rfc3339("../2024-05-02") == "../2024-05-02T23:59:59Z"
    assert D._rfc3339("2024-05-02T10:00:00Z") == "2024-05-02T10:00:00Z"
    assert r.attrs["items"] == ["i0", "i1", "i2", "i3"]
    # the median of 1, 2, 3 (and 4 where not clouded)
    assert r.values[0, 0] == 2.0 and r.values[3, 0] == 2.5
    first = ls.download.stac("https://stac.example/v1", "s2", assets=["red"], composite="first", **kw)
    assert (first.values == 1).all()
    signed = []
    ls.download.stac("https://stac.example/v1", "s2", assets="red", composite="max", limit=3,
                     sign=lambda h: signed.append(h) or h, **kw)
    assert len(signed) == 3
    with pytest.raises(ValueError, match="asset"):
        ls.download.stac("https://stac.example/v1", "s2", assets="nir", **kw)


def test_environment90m_tables(tmp_path, monkeypatch):
    # a map of three sub-catchments near Campinas (tile h12v10); a table per tile and variable, zipped
    ids = np.array([[476322246, 476322247], [476322248, 0]], float)
    sub = xr.DataArray(ids, dims=("y", "x"), coords={"y": [-22.8, -22.9], "x": [-47.0, -46.9]})
    sub = sub.rio.write_crs("EPSG:4326")
    lists = tmp_path / "lists.txt"
    lists.write_text("file;size\nmerit/elev/elev_h12v10.zip;10\nmerit/elev/elev_h00v00.zip;10\nREADME.txt;1\n",
                     encoding="utf-8")
    empty = tmp_path / "empty.txt"
    empty.write_text("file;size\n", encoding="utf-8")
    z = tmp_path / "elev_h12v10.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("elev_h12v10.txt", "subcID min max range mean sd\n476322246 600 700 100 650.5 20\n"
                                       "476322247 610 720 110 655 21\n476322248 na na na na na\n9 1 1 0 1 0\n")

    def fetch(path, refresh=False):
        if path.startswith("_file_lists/"):
            return lists if "meritdem" in path else empty
        assert path == "merit/elev/elev_h12v10.zip"
        return z

    monkeypatch.setattr(D, "_env90m_file", fetch)
    out = ls.download.environment90m(sub, "elev", stats=["mean", "sd"])
    t = out.table
    assert list(t.columns) == ["subcID", "elev_mean", "elev_sd"]
    assert t["subcID"].tolist() == [476322246, 476322247, 476322248]
    assert t["elev_mean"].tolist()[:2] == [650.5, 655.0] and np.isnan(t["elev_mean"][2])
    m = out.map.values
    assert m[0, 0] == 650.5 and m[0, 1] == 655.0 and np.isnan(m[1, 0]) and np.isnan(m[1, 1])
    assert out.map.name == "elev_mean"
    with pytest.raises(ValueError, match="not one of"):
        ls.download.environment90m(sub, "elevation")
    with pytest.raises(ValueError, match="statistic"):
        ls.download.environment90m(sub, "elev", stats="median")


CAMPINAS = {"extent": (-47.2, -23.0, -46.8, -22.7), "cellsize": 0.01}


@ONLINE
def test_downloads_worldclim_and_chelsa():
    w = ls.download.worldclim(layers=[1], **CAMPINAS)
    assert ((w.values > 10) & (w.values < 30)).all()
    c = ls.download.chelsa(layers=[1, 12], **CAMPINAS)
    assert list(c.band.values) == ["bio01", "bio12"] and np.isfinite(c.values).all()


@ONLINE
def test_downloads_copernicus_dem_soilgrids_mapbiomas():
    d = ls.download.copernicus_dem(**CAMPINAS)
    assert ((d.values > 400) & (d.values < 1300)).all()
    s = ls.download.soilgrids(property="clay", depth=["0-5cm", "5-15cm"], **CAMPINAS)
    assert s.shape[0] == 2 and (s.values[0] > 0).any()
    m = ls.download.mapbiomas(years=[2000, 2023], **CAMPINAS).values
    ok = np.isfinite(m)
    assert ((m[ok] >= 0) & (m[ok] <= 75) & (m[ok] == np.round(m[ok]))).all()


@ONLINE
def test_downloads_hydrography90m_and_environment90m():
    sub = ls.download.hydrography90m(**CAMPINAS)
    v = sub.values[np.isfinite(sub.values)]
    assert (v > 0).all() and (v == np.round(v)).all() and len(np.unique(v)) > 20
    out = ls.download.environment90m(sub, ["bio01_1981-2010_observed", "elev"], stats=["mean", "sd"])
    elev = out.table["elev_mean"].to_numpy()
    assert ((elev[np.isfinite(elev)] > 400) & (elev[np.isfinite(elev)] < 1300)).all()
