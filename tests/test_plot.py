# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.plot: the plot every kind of data calls for, its colours and legends, showing and saving."""

from __future__ import annotations

import base64
import io
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

import landschaft as ls
from landschaft._plotting import _env, _raster

pytest.importorskip("folium")
pytest.importorskip("plotly")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point, box

T = from_origin(500000, 4500000, 30, 30)


def grid(values, name="v", nodata=None):
    values = np.asarray(values)
    h, w = values.shape[-2:]
    coords = {"y": 4500000 - 30 * (np.arange(h) + .5), "x": 500000 + 30 * (np.arange(w) + .5)}
    dims = ("y", "x")
    if values.ndim == 3:
        coords["band"] = [f"b{i}" for i in range(values.shape[0])]
        dims = ("band", "y", "x")
    da = xr.DataArray(values, coords=coords, dims=dims, name=name).rio.write_crs(32633).rio.write_transform(T)
    return da.rio.write_nodata(nodata) if nodata is not None else da


def page(p) -> str:
    html = p.to_html()
    assert "LSIMAGE" not in html and "LSPAYLOAD" not in html  # every image and data set in place
    return html


def images(html: str) -> list:
    """The PNG images of a page (not the icons of its style), decoded."""
    from PIL import Image

    out = []
    for b64 in re.findall(r"(?<!url\()data:image/png;base64,([A-Za-z0-9+/=]+)", html):
        out.append(Image.open(io.BytesIO(base64.b64decode(b64))))
    return out


@pytest.fixture
def forest():
    return ls.load_raster(ls.example_data("input"))[:300, :300]


# --- rasters ------------------------------------------------------------------------------------------


def test_products_take_their_colours_and_class_names(forest):
    p = ls.plot(ls.mspa(forest))
    assert repr(p) == "Plot(map)"
    html = page(p)
    for name in ("Core", "Edge", "Background"):
        assert f">{name}<" in html
    assert "Bridge in Edge" not in html  # classes of one colour under one name, as GuidosToolbox
    img = images(html)[0]
    assert img.mode == "P"  # 8-bit with a palette
    palette = np.array(img.getpalette()[:768]).reshape(-1, 3)
    assert [0, 200, 0] in palette.tolist()  # MSPA's core green
    img = _raster.render(ls.mspa(forest), mercator=False)
    labels = [label for _, label in img.legend.classes]
    assert labels[:4] == ["Core", "Islet", "Perforation", "Edge"] and "Missing" not in labels  # GTB's order
    assert (img.pixels[np.asarray(ls.mspa(forest)) == 129] == 0).all()  # missing: the imagery shows through


def test_integers_are_classes_and_floats_a_ramp():
    classes = grid(np.arange(60 * 80).reshape(60, 80) % 5)
    leg = _raster.render(classes).legend
    assert [label for _, label in leg.classes] == ["0", "1", "2", "3", "4"]
    ramp = _raster.render(grid(np.linspace(0, 1, 60 * 80).reshape(60, 80).astype("float32"))).legend
    assert ramp.ramp and ramp.classes == [] and 0 <= ramp.vmin < ramp.vmax <= 1
    many = _raster.render(grid(np.arange(60 * 80).reshape(60, 80))).legend  # 4800 values: a ramp
    masked = np.where(np.arange(60 * 80).reshape(60, 80) % 4 == 0, np.nan, np.arange(60 * 80).reshape(60, 80) % 3)
    read_masked = _raster.render(grid(masked.astype("float32"))).legend  # classes read with masked=True
    assert [label for _, label in read_masked.classes] == ["0", "1", "2"]
    assert _raster.render(grid(masked.astype("float32") + 0.5)).legend.ramp  # not whole numbers: a ramp
    assert many.ramp


def test_colours_can_be_chosen():
    v = grid(np.arange(60 * 80).reshape(60, 80) % 3)
    leg = _raster.render(v, cmap={0: "red", 1: "#00ff00", 2: (0, 0, 255)}, labels={0: "urban", 2: "water"}).legend
    assert leg.classes == [("#ff0000", "urban"), ("#00ff00", "1"), ("#0000ff", "water")]
    assert _raster.render(v, categorical=False).legend.ramp  # integers on a ramp
    hidden = _raster.render(v, cmap={0: "#00000000", 1: "red", 2: "blue"}, mercator=False)
    assert [label for _, label in hidden.legend.classes] == ["1", "2"] and hidden.palette[1, 3] == 0
    f = grid(np.linspace(-1, 1, 60 * 80).reshape(60, 80).astype("float32"))
    centred = _raster.render(f).legend
    assert centred.vmin == -centred.vmax  # around zero: a diverging ramp, centred
    fixed = _raster.render(f, cmap="magma", vmin=0, vmax=0.5).legend
    assert (fixed.vmin, fixed.vmax) == (0, 0.5)
    assert _raster.render(f, cmap="cividis").legend.ramp  # a matplotlib palette
    cur = grid(np.exp(np.linspace(0, 8, 60 * 80)).reshape(60, 80).astype("float32"), name="cum_currmap")
    assert _raster.render(cur).legend.log  # currents: a log scale


def test_nodata_is_transparent():
    v = np.arange(60 * 80, dtype="float32").reshape(60, 80)
    v[:10] = np.nan
    img = _raster.render(grid(v), mercator=False)
    assert (img.pixels[:10] == 0).all() and img.palette[0, 3] == 0 and (img.pixels[10:] > 0).all()
    c = grid(np.where(np.arange(60 * 80).reshape(60, 80) % 2, 7, 255).astype("uint8"), nodata=255)
    ci = _raster.render(c, mercator=False)
    assert set(np.unique(ci.pixels)) == {0, 1} and [label for _, label in ci.legend.classes] == ["7"]
    on_map = _raster.render(grid(v)).pixels  # reprojected to Web Mercator: NoData still transparent
    assert (on_map == 0).mean() >= 10 / 60 - 0.02


def test_large_and_lazy_rasters_are_reduced(tmp_path):
    pytest.importorskip("dask")
    v = np.random.default_rng(0).random((3001, 2503)).astype("float32")
    v[::11, ::7] = -9
    da = grid(v, nodata=-9)
    path = ls.write_raster(da, tmp_path / "a.tif")
    a, ta = _raster.reduce(ls.load_raster(path, chunks="auto", masked=False), 1000, False)
    b, tb = _raster.reduce(da, 1000, False)
    assert a.shape == b.shape == (750, 625) and ta == tb  # blocks of 4 x 4 (ceil(3001 / 1000))
    np.testing.assert_allclose(a, b, equal_nan=True)
    ref = np.where(v == -9, np.nan, v)[:3000, :2500].reshape(750, 4, 625, 4)
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(b, np.nanmean(ref, axis=(1, 3)), rtol=1e-6)
    img = _raster.render(da, max_size=512)
    assert max(img.pixels.shape) <= 600


def test_stacks_and_true_colour():
    stack = grid(np.random.default_rng(1).random((41, 40, 50)).astype("float32"))
    p = ls.plot(stack)
    assert repr(p) == "Plot(map)" and len(images(page(p))) == 41  # every band, a map alone filling the window
    html = p.to_html()
    assert "leaflet-control-layers-list { max-height" in html  # the long list of layers scrolls
    assert "resize:vertical" in p._repr_html_()  # a notebook's map is made taller by its corner
    p = ls.plot(grid(np.zeros((130, 4, 5), "float32")))
    assert "the first 128 of 130 bands" in page(p)
    rgb = grid(np.random.default_rng(1).random((3, 40, 50)).astype("float32"))
    rgb = rgb.assign_coords(band=["red", "green", "blue"])
    img = images(page(ls.plot(rgb)))[0]
    assert img.mode == "RGBA"


def test_rasters_without_georeferencing_are_charts():
    assert repr(ls.plot(np.random.default_rng(0).random((20, 30)))) == "Plot(chart)"
    assert repr(ls.plot(xr.DataArray(np.arange(10.0)))) == "Plot(chart)"


# --- vectors ------------------------------------------------------------------------------------------


def test_vectors_as_geojson_or_image():
    polys = gpd.GeoDataFrame({"name": ["a", "b"], "area": [1.0, 2.0]}, crs=32633,
                             geometry=[box(500000, 4498000, 501000, 4499000), box(501500, 4498500, 502000, 4499500)])
    html = page(ls.plot(polys, column="name"))
    assert '{"type":"FeatureCollection"' in html and '"__c"' in html and ">a<" in html
    rng = np.random.default_rng(0)
    pts = gpd.GeoDataFrame({"v": rng.random(30000)}, crs=32633,
                           geometry=gpd.points_from_xy(rng.uniform(5e5, 6e5, 30000), rng.uniform(4.4e6, 4.5e6, 30000)))
    html = page(ls.plot(pts, column="v"))
    assert '{"type":"FeatureCollection"' not in html and images(html)  # too many: an image


def test_lists_of_anything():
    table = pd.Series([3, 1, 2], index=["a", "b", "c"], name="n")
    p = ls.plot([table, np.arange(5.0)])  # not maps: each part as it calls for
    assert p.panels and all(x.kind == "chart" for x in p.panels)


def test_rasters_and_vectors_share_a_map(forest):
    poly = gpd.GeoDataFrame(geometry=[Point(float(forest.x[150]), float(forest.y[150])).buffer(0.01)],
                            crs=forest.rio.crs)
    p = ls.plot([forest, poly], title="Forest and site")
    html = page(p)
    assert repr(p) == "Plot('Forest and site': map)" and "Forest and site" in html and images(html)


# --- tables and results ----------------------------------------------------------------------------------


def test_tables_are_charts():
    m = ls.landscape_metrics(ls.load_raster(ls.example_data("clc"))[:200, :200], level="class", metric=["ca", "pland"])
    p = ls.plot(m)
    assert repr(p) == "Plot(chart)" and len(p.charts[0].data) == 2  # one panel per metric
    s = pd.Series([3, 1, 2], index=["a", "b", "c"], name="n")
    assert ls.plot(s).charts[0].data[0].type == "bar"
    curve = pd.DataFrame({"x": np.arange(50.0), "a": np.arange(50.0) ** 2, "b": np.arange(50.0)})
    assert ls.plot(curve).charts[0].data[0].type == "scattergl"


def test_results_show_their_maps_tables_and_numbers(forest):
    p = ls.plot(ls.fos(forest))
    assert [x.kind for x in p.panels][:1] == ["map"] and "html" in [x.kind for x in p.panels]
    res = ls.prioritize.zonation(grid(np.random.default_rng(2).random((2, 40, 50)).astype("float32")))
    z = ls.plot(res)
    assert [x.kind for x in z.panels] == ["map", "chart", "html"]
    assert "Performance curves" in page(z) and "AUC" in page(z)
    ds = xr.Dataset({"cum_currmap": grid(np.random.default_rng(3).random((40, 50)).astype("float32"))},
                    attrs={"radius": 10})
    assert [x.kind for x in ls.plot(ds).panels] == ["map", "html"]


# --- showing and saving -------------------------------------------------------------------------------


def test_save_html_and_static_images(tmp_path, forest):
    p = ls.plot(ls.mspa(forest))
    out = p.save(tmp_path / "sub" / "mspa.html")
    text = out.read_text(encoding="utf-8")
    assert "LSIMAGE" not in text and "leaflet" in text.lower()
    pytest.importorskip("matplotlib")
    png = p.save(tmp_path / "mspa.png")  # a static rendering
    from PIL import Image

    assert Image.open(png).size[0] > 100
    s = ls.plot(forest, static=True, basemap=None)
    assert repr(s) == "Plot(figure)" and s.figure is not None
    folium_map = p.maps[0]
    folium_map.save(str(tmp_path / "raw.html"))  # folium's own save puts the image in too
    assert "LSIMAGE" not in (tmp_path / "raw.html").read_text(encoding="utf-8")


def test_windows_only_outside_notebooks(monkeypatch, forest):
    opened = []
    monkeypatch.setattr(_env, "open_window", lambda html, title="": opened.append(title))
    monkeypatch.delenv("LANDSCHAFT_PLOT_SHOW", raising=False)
    monkeypatch.setattr(_env, "in_notebook", lambda: False)
    ls.plot(forest, title="here")
    assert opened == ["here"]
    monkeypatch.setattr(_env, "in_notebook", lambda: True)
    ls.plot(forest)  # the Plot shows itself in a notebook
    assert opened == ["here"]
    monkeypatch.setenv("LANDSCHAFT_PLOT_SHOW", "0")
    monkeypatch.setattr(_env, "in_notebook", lambda: False)
    ls.plot(forest)
    ls.plot(forest, show=False)
    assert opened == ["here"]


# --- offline pages, NoData, bands, vector ramps -------------------------------------------------------


def test_pages_open_offline(forest):
    html = page(ls.plot(forest))
    assert "L.map(" in html and "Leaflet 1.9.3" in html  # Leaflet in the page
    assert not re.findall(r"<(?:script|link)[^>]+(?:src|href)=[\"']https?://", html)  # nothing from a CDN
    assert "jquery" not in html.lower() and "bootstrap" not in html.lower()
    s = pd.Series([3, 1, 2], index=["a", "b", "c"], name="n")
    charts = page(ls.plot([s, s * 2]))
    assert not re.findall(r"<script[^>]+src=[\"']https?://", charts)  # plotly's script in the page...
    assert charts.count("plotly.js v") == 1  # ...once
    assert _env.plotly_url() is True or _env.plotly_url().startswith("file:")  # a window loads it from disk


def test_true_colour_leaves_nodata_out():
    v = np.random.default_rng(4).integers(100, 2000, (3, 40, 50)).astype("uint16")
    v[:, :10] = 0
    rgb = grid(v, nodata=0).assign_coords(band=["red", "green", "blue"])
    img = _raster.render_rgb(rgb)
    alpha = img.rgba[..., 3]
    assert (alpha == 0).mean() >= 10 / 40 - 0.05 and alpha.max() == 255  # NoData transparent, not black
    lazy = pytest.importorskip("dask.array")
    f = np.where(np.arange(3000 * 40).reshape(3000, 40) % 3 == 0, -9, 5).astype("float32")
    da = grid(f, nodata=-9).copy(data=lazy.from_array(f, chunks=500))
    out, _ = _raster.reduce(da, 1000, False)
    assert np.nanmax(out) == 5  # NoData out of the block means of a lazy raster


def test_bands_by_label_then_position():
    from landschaft._plotting._core import _band_index

    stack = grid(np.zeros((3, 4, 5), "float32")).assign_coords(band=[1, 2, 3])
    assert _band_index(stack, "band", [1]) == [0]  # the band numbered 1 (a file's first band)
    named = grid(np.zeros((3, 4, 5), "float32"))
    assert _band_index(named, "band", ["b2", 0]) == [2, 0]
    with pytest.raises(ValueError, match="the bands are"):
        _band_index(named, "band", ["nir"])
    p = ls.plot(stack, bands=[3])
    assert len(images(page(p))) == 1


def test_vector_ramps_take_their_range():
    pts = gpd.GeoDataFrame({"v": np.geomspace(1, 1e4, 50), "s": ["<b>&"] * 50}, crs=32633,
                           geometry=gpd.points_from_xy(np.linspace(5e5, 5.1e5, 50), np.full(50, 4.45e6)))
    from landschaft._plotting import _vectors

    _, _, lg = _vectors.layer(pts, column="v", vmin=10, vmax=100)
    assert (lg.vmin, lg.vmax, lg.under, lg.over) == (10, 100, True, True)
    _, _, lg = _vectors.layer(pts, column="v", log=True)
    assert lg.log and lg.vmin > 1 and lg.vmax < 1e4
    html = page(ls.plot(pts, column="v", vmin=10, vmax=100))
    assert "≤ 10" in html and "≥ 100" in html and "esc(f.properties[k])" in html
