# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.plot: the plot every kind of data calls for, its colours and legends, showing and saving."""

from __future__ import annotations

import base64
import io
import re
from html import unescape

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
    start = html.index("class='ls-legends'")
    legend = html[start:html.index("<script", start)]
    assert "Bridge in Edge" not in legend  # classes of one colour under one name, as GuidosToolbox
    assert '"Bridge in Edge"' in html  # but the cursor reads each class by its name
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


def test_vectors_as_geojson_webgl_or_image(monkeypatch):
    polys = gpd.GeoDataFrame({"name": ["a", "b"], "area": [1.0, 2.0]}, crs=32633,
                             geometry=[box(500000, 4498000, 501000, 4499000), box(501500, 4498500, 502000, 4499500)])
    html = page(ls.plot(polys, column="name"))
    assert '{"type":"FeatureCollection"' in html and '"__c"' in html and ">a<" in html
    rng = np.random.default_rng(0)
    pts = gpd.GeoDataFrame({"v": rng.random(30000)}, crs=32633,
                           geometry=gpd.points_from_xy(rng.uniform(5e5, 6e5, 30000), rng.uniform(4.4e6, 4.5e6, 30000)))
    html = page(ls.plot(pts, column="v"))
    assert '{"type":"FeatureCollection"' not in html and "new L.LsGL({" in html  # many: WebGL
    monkeypatch.setattr(_gl_module(), "MAX_FEATURES", 20_000)
    html = page(ls.plot(pts, column="v"))
    assert "new L.LsGL({" not in html and images(html)  # too many: an image


def _gl_arrays(layer):
    """The arrays of a WebGL layer decoded as lsgl.js decodes them."""
    import zlib

    blob = base64.b64decode(layer._ls_payload)
    out = {}
    def inflate(pieces):
        return b"".join(zlib.decompress(blob[off:off + size]) for off, size in pieces)

    for k, (pieces, n, delta) in layer.spec["arrays"].items():
        raw = inflate(pieces)
        v = np.frombuffer(raw, np.uint8).reshape(4, n).T.copy().view("<u4").ravel()
        if delta:
            z = v.astype(np.int64)
            v = np.cumsum((z >> 1) ^ -(z & 1)).astype(np.int32)
        out[k] = v
    p = layer.spec["props"]
    import json

    out["props"] = json.loads(inflate(p)) if p else None
    return out


@pytest.mark.parametrize("crs", [32633, 4326])
def test_many_vectors_are_drawn_with_webgl(monkeypatch, crs):
    from landschaft._plotting import _vectors

    monkeypatch.setattr(_vectors, "MAX_FEATURES", 2)
    donut = box(500000, 4490000, 510000, 4500000).difference(box(503000, 4493000, 507000, 4497000))
    from shapely.geometry import LineString, MultiPolygon

    two = MultiPolygon([box(512000, 4490000, 514000, 4492000), box(515000, 4495000, 518000, 4498000)])
    g = gpd.GeoDataFrame({"k": ["a", "b", "c", "d"], "n": [1.5, None, 3.0, 4.0]}, crs=32633, geometry=[
        donut, two, LineString([(500000, 4502000), (518000, 4504000)]), Point(519000, 4491000)]).to_crs(crs)
    kind, layer, legend = _vectors.layer(g, column="k")
    assert kind == "webgl" and [c for _, c in legend.classes] == ["a", "b", "c", "d"]
    a = _gl_arrays(layer)
    q, (x0, y0) = layer.spec["q"], layer.spec["origin"]
    n = len(a["pxy"]) // 2
    x, y = a["pxy"][:n] * q + x0, a["pxy"][n:] * q + y0
    # the triangles of each polygon cover its area (in Web Mercator), holes left out
    merc = g.to_crs(3857).geometry.values
    tri, start, v0 = a["pidx"], 0, 0
    count_of = a["ptri"].astype(np.int64)
    feature, areas = np.cumsum(a["pfeat"]), {}
    verts = np.cumsum(np.r_[0, a["pring"]])
    rings = np.cumsum(np.r_[0, a["prings"]])
    for p_, count in enumerate(count_of):
        t = tri[start:start + count].reshape(-1, 3).astype(np.int64) + int(v0)
        ax_, ay_ = x[t], y[t]
        area = 0.5 * np.abs((ax_[:, 1] - ax_[:, 0]) * (ay_[:, 2] - ay_[:, 0]) - (ax_[:, 2] - ax_[:, 0]) * (ay_[:, 1] - ay_[:, 0]))
        areas[feature[p_]] = areas.get(feature[p_], 0.0) + area.sum()
        start += count
        v0 = verts[rings[p_ + 1]]
    assert sorted(areas) == [0, 1]
    for f, area in areas.items():
        assert area == pytest.approx(merc[f].area, rel=1e-3)
    assert list(np.cumsum(a["lfeat"])) == [2] and list(np.cumsum(a["tfeat"])) == [3]
    assert list(a["fcol"]) == [0, 1, 2, 3] and a["props"] == {"k": ["a", "b", "c", "d"], "n": [1.5, None, 3.0, 4.0]}
    html = page(ls.plot(g, column="k"))
    assert html.count("L.LsGL = L.Renderer.extend") == 1 and "new L.LsGL({" in html and "LSPAYLOAD" not in html
    assert '{"type":"FeatureCollection"' not in html
    monkeypatch.setattr(_gl_module(), "MAX_FEATURES", 3)  # too many even for WebGL: an image
    assert _vectors.layer(g, column="k")[0] == "image"



def test_many_points_with_webgl_read_their_values_back_from_the_colours(monkeypatch):
    from landschaft._plotting import _vectors

    monkeypatch.setattr(_vectors, "MAX_FEATURES", 2)
    monkeypatch.setattr(_vectors, "MAX_VALUES", 4)  # too many values to hold: none, the colour read
    g = gpd.GeoDataFrame({"v": [1.0, 2.0, 3.0], "s": ["x", "y", None]}, crs=32633,
                         geometry=gpd.points_from_xy([500000, 501000, 502500], [4490000, 4491000, 4490500]))
    kind, layer, _ = _vectors.layer(g, column="v", categorical=False, vmin=1, vmax=3)
    assert kind == "webgl"
    a = _gl_arrays(layer)
    q, (x0, y0) = layer.spec["q"], layer.spec["origin"]
    x, y = a["txy"][:3] * q + x0, a["txy"][3:] * q + y0
    from landschaft._vector import transform_xy

    mx, my = transform_xy(np.array([500000.0, 501000, 502500]), np.array([4490000.0, 4491000, 4490500]), "EPSG:32633",
                          "EPSG:3857")
    assert np.abs(x - mx).max() <= q and np.abs(y - my).max() <= q
    assert list(np.cumsum(a["tfeat"])) == [0, 1, 2] and list(a["fcol"]) == [0, 127, 254]
    assert a["props"] is None and layer.spec["fields"] == []
    assert layer.spec["lookup"] == {"vmin": 1.0, "vmax": 3.0, "log": False, "under": False, "over": False}
    monkeypatch.setattr(_vectors, "MAX_VALUES", 100)
    _, layer, _ = _vectors.layer(g, column="s")
    assert _gl_arrays(layer)["props"] == {"v": [1.0, 2.0, 3.0], "s": ["x", "y", None]}
    assert layer.spec["lookup"] == {"labels": ["x", "y"]}  # None: no value, the missing colour



# --- detail on demand (live) --------------------------------------------------------------------------------


def _live_png(answer):
    from PIL import Image as PILImage

    assert answer["ok"] and answer["url"].startswith("data:image/png;base64,")
    return PILImage.open(io.BytesIO(base64.b64decode(answer["url"].split(",", 1)[1])))


def test_live_raster_detail_has_the_colours_of_the_overview():
    from landschaft._plotting import _live

    yy, xx = np.mgrid[0:3000, 0:3000]
    values = (np.sin(xx / 3.0) + np.cos(yy / 3.0)).astype("float32")
    p = ls.plot(grid(values), basemap=None, max_size=512)
    ids = p._live_maps()
    assert len(ids) == 1 and "LS_LIVE.add(" in page(p)
    assert _live.handle({"op": "hello", "map": ids[0]}) == {"ok": True}
    assert _live.handle({"op": "hello", "map": "gone"}) == {"ok": False}
    m = p.maps[0]
    key = next(iter(_live._MAPS[ids[0]]))
    ext = [float(v) for v in m.__dict__["_ls_live_layers"][0]["ext"]]
    cx, cy = (ext[0] + ext[2]) / 2, (ext[1] + ext[3]) / 2
    box = [cx - 300, cy - 300, cx + 300, cy + 300]  # about 60 x 60 cells of 30 m (in Web Mercator)
    a = _live.handle({"op": "detail", "map": ids[0], "key": key, "bbox": box, "w": 800, "h": 800})
    img = _live_png(a)
    assert max(img.size) >= 700  # at the screen's resolution, finer than the cells
    from landschaft._plotting import _raster

    assert a["lookup"]["vmin"] == pytest.approx(_raster.render(grid(values), max_size=512).lookup["vmin"])
    assert a["ext"][0] <= box[0] + 60 and a["ext"][2] >= box[2] - 60
    bad = _live.handle({"op": "detail", "map": ids[0], "key": "nope", "bbox": box, "w": 10, "h": 10})
    assert bad["ok"] is False and "error" in bad



def test_a_window_of_a_lazy_raster_is_read_reduced_from_its_file(tmp_path):
    values = np.random.default_rng(9).random((900, 1100)).astype("float32")
    path = ls.write_raster(grid(values), tmp_path / "r.tif")
    lazy, mem = ls.load_raster(path, chunks="auto"), grid(values)
    for categorical in (False, True):
        a, ta = _raster.reduce(lazy.isel(x=slice(101, 1001), y=slice(53, 803)), 200, categorical)
        b, tb = _raster.reduce(mem.isel(x=slice(101, 1001), y=slice(53, 803)), 200, categorical)
        assert np.array_equal(a, b) and ta == tb

def test_live_class_detail_keeps_the_class_colours():
    from landschaft._plotting import _raster

    v = (np.arange(600)[None, :] // 7 + np.arange(600)[:, None] // 7) % 5
    v[300, 300] = 9  # a class of one cell, left out of the overview
    da = grid(v.astype("uint8"))
    overview = _raster.render(da, max_size=100)
    assert 9 not in overview.style["table"]
    part = _raster.render(da.isel(x=slice(280, 320), y=slice(280, 320)), max_size=400, style=overview.style)
    same = {lab: c for c, lab in overview.legend.classes}
    for c, lab in part.legend.classes:
        if lab in same:
            assert same[lab] == c  # the classes of the overview in their colours
    assert "9" in [lab for _, lab in part.legend.classes]  # the new one with a colour of its own
    assert part.lookup["labels"][0] is None


def test_live_vectors_answer_with_the_features_in_view(monkeypatch):
    from landschaft._plotting import _gl, _vectors

    monkeypatch.setattr(_vectors, "MAX_FEATURES", 10)
    rng = np.random.default_rng(4)
    g = gpd.GeoDataFrame({"v": rng.random(200)}, crs=32633,
                         geometry=[Point(x, y).buffer(200, quad_segs=16) for x, y in
                                   zip(rng.uniform(5e5, 6e5, 200), rng.uniform(4.4e6, 4.5e6, 200))])
    kind, layer, _ = _vectors.layer(g, column="v")
    assert kind == "webgl" and layer._live is not None
    box = [float(v) for v in layer._ls_merc]
    a = layer._live(box, 1000, 1000)
    assert a["spec"]["arrays"]["pxy"][1] > 0 and a["bbox"] == box
    assert layer._live([0.0, 0.0, 1.0, 1.0], 100, 100) is None  # nothing there
    monkeypatch.setattr(_gl, "MAX_FEATURES", 5)  # too many for WebGL: an image, and its detail too
    kind, img, _ = _vectors.layer(g, column="v")
    assert kind == "image"
    pic = _live_png({"ok": True, **img._live(list(img.mercator_extent), 600, 600)})
    assert max(pic.size) >= 550


def test_live_bridges(monkeypatch):
    pytest.importorskip("anywidget")
    from landschaft._plotting import _live

    p = ls.plot(grid(np.random.default_rng(2).random((3000, 3000)).astype("float32")), basemap=None, max_size=256)
    shown = []
    monkeypatch.setattr("IPython.display.display", lambda x: shown.append(x))
    p._ipython_display_()
    assert type(shown[0]).__name__ == "HTML" and type(shown[1]).__name__ == "LiveBridge"
    assert shown[1].maps == p._live_maps()
    replies = []
    monkeypatch.setattr(shown[1], "send", lambda content: replies.append(content))
    shown[1]._answer(shown[1], {"req": "a:1", "msg": {"op": "hello", "map": p._live_maps()[0]}}, [])
    assert replies == [{"req": "a:1", "map": p._live_maps()[0], "data": {"ok": True}}]
    assert _live.WindowApi().ls_live({"op": "hello", "map": p._live_maps()[0]}) == {"ok": True}
    monkeypatch.setenv("LANDSCHAFT_PLOT_LIVE", "0")
    q = ls.plot(grid(np.random.default_rng(2).random((3000, 3000)).astype("float32")), basemap=None, max_size=256)
    assert not q._live_maps() and "LS_LIVE" not in page(q)


def _gl_module():
    from landschaft._plotting import _gl

    return _gl


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
    monkeypatch.setattr(_env, "open_window", lambda html, title="", **_: opened.append(title))
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


def test_maps_hidden_at_load_fit_when_shown(forest):
    # a notebook's output can load hidden, at no size: the view is fitted again once it has one
    html = page(ls.plot(forest))
    assert "new ResizeObserver" in html
    assert html.index("map.fitBounds = function") < html.rindex(".fitBounds(")  # in place before the fit
    # the tiles' fade-in waits on animation frames a notebook's frame may not get (VS Code)
    assert '"fadeAnimation": false' in html


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


# --- reading values on the map, legend cards ----------------------------------------------------------


def test_a_pixel_colour_gives_back_its_value(forest):
    from landschaft._plotting._raster import distinct_palette

    pal = np.array([[0, 0, 0, 0], [9, 9, 9, 255], [9, 9, 9, 255], [9, 9, 255, 255], [9, 9, 255, 255]], np.uint8)
    out = distinct_palette(pal)
    assert len({tuple(c) for c in out[1:, :3].tolist()}) == 4 and (out[:2] == pal[:2]).all()
    assert np.abs(out.astype(int) - pal.astype(int)).max() <= 1  # nudged by a unit, invisible
    img = _raster.render(ls.mspa(forest), mercator=False)
    rgb = img.rgba  # what the browser reads: colours back to their indices, one to one
    keys = {tuple(c): i for i, c in enumerate(img.palette[:, :3].tolist()) if i}
    drawn = img.pixels > 0
    back = np.array([keys[tuple(c)] for c in rgb[drawn][:, :3].tolist()])
    assert (back == img.pixels[drawn]).all() and (img.palette[1:, 3] == 255).all()
    labels = img.lookup["labels"]
    m = np.asarray(ls.mspa(forest))
    from landschaft.mspa import MSPA_CLASSES

    i, j = np.argwhere(drawn)[100]
    assert labels[img.pixels[i, j]] == MSPA_CLASSES[int(m[i, j])]
    dem = grid(np.linspace(0, 1000, 60 * 80).reshape(60, 80).astype("float32"), name="dem")
    ramp = _raster.render(dem, vmin=0, vmax=1000, mercator=False)
    u = ramp.lookup
    k = ramp.pixels[30, 40]
    assert abs(u["vmin"] + (k - 1) / 254 * (u["vmax"] - u["vmin"]) - float(dem[30, 40])) <= 1000 / 254
    assert ramp.opacity == 0.85  # applied by the map to the layer, the palette opaque


def test_maps_read_values_and_have_a_card_per_layer(forest, monkeypatch):
    monkeypatch.setattr(_gl_module(), "MAX_FEATURES", 0)  # many features as an image
    stack = grid(np.random.default_rng(5).random((3, 40, 50)).astype("float32"), name="stack")
    poly = gpd.GeoDataFrame(geometry=[box(500300, 4498900, 500900, 4499500)], crs=32633)
    html = page(ls.plot([stack, poly]))
    assert html.count("class='ls-legend'") == 2  # the stack's bands share a card; the polygons have theirs
    assert html.count("title='Opacity'") == 2 and html.count("class='bands'") == 1  # opacity; the band shown
    assert "getImageData" in html and '"n": "stack: b0"' in html and '"vmin"' in html  # values under the cursor
    pts = gpd.GeoDataFrame({"kind": np.resize(["oak", "pine"], 30000)}, crs=32633,
                           geometry=gpd.points_from_xy(np.linspace(5e5, 6e5, 30000), np.linspace(4.4e6, 4.5e6, 30000)))
    html = page(ls.plot(pts, column="kind"))
    assert '"labels": [null, "oak", "pine", "no value"]' in html  # features drawn as an image read too


# --- class names of the products, comparing two dates, change results ----------------------------------


def test_density_and_change_maps_name_their_classes(forest):
    fad = ls.fad(forest)
    labels = [label for _, label in _raster.render(fad.density, mercator=False).legend.classes]
    assert labels[:3] == ["Rare (0-9%)", "Patchy (10-39%)", "Transitional (40-59%)"] and "Background" in labels
    cls = _raster.render(fad.classes, mercator=False).legend.classes
    assert cls[0] == ("#d73228", "Rare (<10%)") and cls[5][0] == "#007800"  # the density map's colours
    after = forest.where(forest.x < forest.x[150], 1)
    change = ls.simple_change(forest, after)
    names = [label for _, label in _raster.render(change.change_map, mercator=False).legend.classes]
    assert names == ["Stable foreground", "Loss", "Stable background"]
    fos = ls.fos_change(forest, after)
    legend = _raster.render(fos.change_map, mercator=False).legend.classes
    assert len(legend) <= 30 and any(lab == "no change" for _, lab in legend)


def test_compare_two_dates_with_a_swipe(forest):
    after = forest.where(forest.x < forest.x[150], 1)
    html = page(ls.plot([forest, after], compare=True))
    assert len(re.findall(r'createPane\(\s*"ls(?:left|right)"', html)) == 2
    assert '"pane": "lsleft"' in html and '"pane": "lsright"' in html
    assert "ls-swipe-knob" in html and '"input (1)", "input (2)"' in html  # one side each, named
    assert html.count("class='ls-legend'") == 1 and "input (1) · input (2)" in html  # the same colours: one card
    pts = gpd.GeoDataFrame(geometry=[Point(float(forest.x[10]), float(forest.y[10]))], crs=forest.rio.crs)
    assert 'pane: "lsright"' in page(ls.plot([forest, pts], compare=True))
    with pytest.raises(ValueError, match="two rasters"):
        ls.plot([forest], compare=True)


def test_change_results_show_their_flows(forest):
    after = forest.where(forest.x < forest.x[150], 1)
    p = ls.plot(ls.simple_change(forest, after))
    assert [x.kind for x in p.panels] == ["map", "chart", "html"] and p.charts[0].data[0].type == "sankey"
    flows = p.charts[0].data[0]
    assert sum(flows.link.value) == ls.simple_change(forest, after).total_valid
    p = ls.plot(ls.fos_change(forest, after))
    assert p.charts[0].data[0].type == "sankey" and "Rare" in p.charts[0].data[0].node.label[0]
    p = ls.plot(ls.morphological_change(forest, after))
    assert [x.kind for x in p.panels] == ["map", "chart", "html"]
    p = ls.plot(ls.fad_multiscale_change(forest, after))
    assert p.charts[0].data[0].type == "sankey" and len(p.charts[0].data[0].node.label) == 14


# --- static figures ----------------------------------------------------------------------------------------


def _maps_of(fig):
    return [ax for ax in fig.axes if ax.get_visible() and ax.images]


def test_static_maps_have_a_scale_bar_a_north_arrow_and_coordinates(forest):
    pytest.importorskip("matplotlib")
    from matplotlib.offsetbox import AnchoredOffsetbox

    utm = grid(np.arange(60 * 80).reshape(60, 80) % 3)
    fig = ls.plot(utm, static=True, basemap=None).figure
    ax = _maps_of(fig)[0]
    bars = [a for a in ax.artists if isinstance(a, AnchoredOffsetbox)]
    assert len(bars) == 1 and any(t.get_text() == "N" for t in ax.texts)
    assert not ax.get_xticks().size  # a projected CRS: no degrees to show
    from landschaft._plotting._static import _metres_per_unit

    assert _metres_per_unit(ax, utm.rio.crs) == 1.0
    assert forest.rio.crs.is_geographic
    fig = ls.plot(forest, static=True, basemap=None).figure
    fig.canvas.draw()
    labels = [t.get_text() for t in _maps_of(fig)[0].get_xticklabels()]
    assert labels and all(lab.endswith("°E") for lab in labels if lab) and len(set(labels)) == len(labels)
    fig = ls.plot(forest, static=True, basemap=None, scalebar=False, north_arrow=False).figure
    assert not _maps_of(fig)[0].artists and not _maps_of(fig)[0].texts


def test_static_small_multiples(forest):
    pytest.importorskip("matplotlib")
    stack = grid(np.random.default_rng(6).random((4, 40, 50)).astype("float32"), name="ndvi")
    fig = ls.plot(stack, static=True, basemap=None).figure
    maps = _maps_of(fig)
    assert len(maps) == 4 and len(fig.axes) == 5  # the bands, and one colour bar
    assert fig.axes[-1].get_ylabel() == "ndvi"
    assert len(_maps_of(ls.plot(stack, static=True, basemap=None, bands=["b1", "b3"]).figure)) == 2
    after = forest.where(forest.x < forest.x[150], 1)
    fig = ls.plot([ls.mspa(forest), ls.mspa(after)], compare=True, static=True, basemap=None).figure
    maps = _maps_of(fig)
    assert [ax.get_title() for ax in maps] == ["mspa (1)", "mspa (2)"]
    assert maps[0].get_legend() is None and maps[1].get_legend().get_title().get_text() == "mspa"  # once
    fig = ls.plot(ls.fad(forest), static=True, basemap=None).figure
    assert [ax.get_title() for ax in _maps_of(fig)] == ["density", "raw_density", "classes"]


def test_static_vectors_take_the_map_colours():
    pytest.importorskip("matplotlib")
    polys = gpd.GeoDataFrame({"kind": ["oak", "pine"]}, crs=32633,
                             geometry=[box(500000, 4498000, 501000, 4499000), box(501500, 4498500, 502000, 4499500)])
    fig = ls.plot(polys, column="kind", cmap={"oak": "#ff0000", "pine": "#0000ff"}, static=True, basemap=None).figure
    ax = fig.axes[0]
    colours = ax.collections[0].get_facecolors()[:, :3].round(2).tolist()
    assert colours == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["oak", "pine"]



def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3]


def _at(fig, ax, x, y):
    """The colour drawn at a point of the data (the layout drawn first)."""
    a = _pixels(fig)
    px, py = ax.transData.transform((x, y))
    return a[a.shape[0] - round(float(py)), round(float(px))].tolist()


def test_static_vectors_holes_parts_lines_and_points():
    pytest.importorskip("matplotlib")
    import shapely
    from shapely.geometry import LineString, MultiPolygon

    donut = box(500000, 4490000, 510000, 4500000).difference(box(503000, 4493000, 507000, 4497000))
    turned = shapely.Polygon(donut.exterior.coords[::-1], [donut.interiors[0].coords])  # both rings one way
    parts = MultiPolygon([box(512000, 4490000, 514000, 4492000), box(515000, 4495000, 518000, 4498000)])
    g = gpd.GeoDataFrame({"k": ["a", "b", "c", "d", "e"]}, crs=32633, geometry=[
        donut, parts, LineString([(500000, 4502000), (518000, 4504000)]), Point(519000, 4491000),
        shapely.affinity.translate(turned, 0, -12000)])
    cmap = {"a": "#ff0000", "b": "#0000ff", "c": "#00ff00", "d": "#ff00ff", "e": "#00ffff"}
    fig = ls.plot(g, column="k", cmap=cmap, opacity=1, static=True, basemap=None, scalebar=False,
                  north_arrow=False).figure
    ax = fig.axes[0]
    assert _at(fig, ax, 501500, 4495000) == [255, 0, 0]  # the ring
    assert _at(fig, ax, 505000, 4495000) == [255, 255, 255]  # its hole
    assert _at(fig, ax, 505000, 4483000) == [255, 255, 255]  # the hole of a polygon turned the other way
    assert _at(fig, ax, 501500, 4483000) == [0, 255, 255]
    assert _at(fig, ax, 513000, 4491000) == [0, 0, 255] == _at(fig, ax, 516500, 4496500)  # both parts
    assert {type(c).__name__ for c in ax.collections} == {"PathCollection", "LineCollection"}


def test_static_vectors_very_many_are_an_image(monkeypatch):
    pytest.importorskip("matplotlib")
    from landschaft._plotting import _static

    rng = np.random.default_rng(3)
    polys = gpd.GeoDataFrame({"v": rng.random(300)}, crs=32633,
                             geometry=[box(x, y, x + 500, y + 500) for x, y in
                                       zip(rng.uniform(5e5, 6e5, 300), rng.uniform(4.4e6, 4.5e6, 300))])
    fig = ls.plot(polys, column="v", static=True, basemap=None).figure
    assert not fig.axes[0].images and fig.axes[0].collections
    monkeypatch.setattr(_static, "MAX_FEATURES", 100)
    fig = ls.plot(polys, column="v", static=True, basemap=None).figure
    ax = fig.axes[0]
    assert len(ax.images) == 1 and not ax.collections
    x0, _, _, y1 = ax.images[0].get_extent()
    assert x0 == pytest.approx(polys.total_bounds[0]) and y1 == pytest.approx(polys.total_bounds[3])


def test_vectors_simplified_keep_their_small_features():
    import shapely

    from landschaft._plotting._vectors import simplify

    geoms = np.array([shapely.Point(0, 0).buffer(100, quad_segs=32), shapely.Point(5000, 0).buffer(1)])
    out = simplify(geoms, 10.0)
    assert shapely.get_num_coordinates(out[0]) < shapely.get_num_coordinates(geoms[0])
    assert not shapely.is_empty(out).any()  # the small circle is still there


# --- stacks band by band, results of their own -------------------------------------------------------------


def test_a_stack_is_one_layer_with_a_band_slider():
    stack = grid(np.random.default_rng(7).random((5, 40, 50)).astype("float32"), name="ndvi")
    html = page(ls.plot(stack))
    assert html.count("class='bands'") == 1 and "max='4'" in html and ">b0</span>" in html
    assert "LC.addOverlay(grp, g.t)" in html and '"b": ["b0", "b1", "b2", "b3", "b4"]' in html



def test_the_bands_of_a_stack_share_a_budget_of_pixels(monkeypatch):
    from landschaft._plotting import _core

    stack = grid(np.random.default_rng(8).random((3, 1200, 1200)).astype("float32"))
    side = lambda: max(_core._map_layers([("s", stack)], {"max_size": 2048})[0][0].items[0].pixels.shape)
    assert side() >= 1200
    monkeypatch.setattr(_core, "STACK_PIXELS", 3 * 600 * 600)
    assert side() <= 610
    monkeypatch.setattr(_core, "STACK_PIXELS", 3 * 100 * 100)
    assert 395 <= side() <= 410  # never coarser than 512 cells: blocks of 3 x 3 (1200 / 512, up)

def test_species_distribution_results():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    occ = ls.load_table(ls.sdm.example_data("bradypus"))
    env = ls.load_raster(ls.sdm.example_data("bioclim"))
    data = ls.sdm.prepare(occ, env, clean=False, background=500)
    res = ls.sdm.fit(data, {"glm": LogisticRegression(max_iter=5000)})
    p = ls.plot(res)
    assert [x.kind for x in p.panels] == ["chart", "chart", "html"]
    assert len(p.charts[0].data) == len(res.predictors)  # a response curve per predictor
    assert p.charts[1].data[0].error_y.array is not None  # scores with their spread
    p = ls.plot(res, env=env)
    assert p.panels[0].kind == "map"
    html = unescape(page(p))  # the map in a frame of the page
    assert '"n": "suitability: glm"' in html and "<b>presences</b>" in html


def test_clustergram_least_cost_path_and_graph_links(forest):
    signatures = ls.SpatialSignatureAnalysis(ls.example_data("clc"), grid=5000, landscape_metrics=["shdi", "ed"])
    p = ls.plot(signatures.clustergram(range(1, 5)))
    assert p.charts[0].data[-1].mode == "markers" and p.charts[0].layout.xaxis.title.text.startswith("number")
    resistance = ls.distance_to_resistance(forest)
    objects = ls.accounting(forest)
    ids = objects.ids.values
    marker = np.zeros(forest.shape, dtype="uint8")
    marker[ids == objects.largest[0][0]] = 1
    marker[ids == objects.largest[1][0]] = 2
    path = ls.least_cost_path(resistance, marker == 1, marker == 2)
    html = unescape(page(ls.plot(path)))
    assert "least-cost path" in html and '{"type":"FeatureCollection"' in html  # the path drawn as polygons
    patches = gpd.GeoDataFrame({"name": ["a", "b", "c"]}, crs="EPSG:3035", geometry=[
        box(0, 0, 500, 500), box(900, 0, 1400, 400), box(3000, 0, 3300, 300)])
    graph = ls.HabitatGraph.from_vector(patches, id="name", area_unit="ha", distance="edge")
    html = unescape(page(ls.plot(graph, distance=5000)))
    assert html.count("class='ls-legend'") == 2  # nodes and links
    assert re.search(r"<b>links</b>.*?class='ramp'", html)  # the links on a ramp of their distance
