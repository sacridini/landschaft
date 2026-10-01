# SPDX-License-Identifier: GPL-3.0-or-later
"""Vector layers for maps: GeoJSON with rounded coordinates, or, for very many features, an image."""

from __future__ import annotations

import math

import numpy as np

from . import _palettes
from ._raster import Image, Legend

#: Above this many features (or vertices), a layer is drawn as an image instead of GeoJSON.
MAX_FEATURES = 20_000
MAX_VERTICES = 1_000_000
#: Columns shown in the tooltip.
MAX_TOOLTIP = 6
_MISSING = "#808080"


def _colours(gdf, column, cmap, categorical):
    """A palette index per feature (-1: no value), the palette (hex) and the legend; vectorised."""
    if column is None:
        return None, None, None
    import pandas as pd

    values = gdf[column]
    numeric = _palettes.is_number(values.dtype) and categorical is not True
    if numeric and (categorical is False or values.nunique(dropna=True) > 12):
        v = values.to_numpy(np.float64)
        ok = np.isfinite(v)
        lo, hi = (np.percentile(v[ok], (2, 98)) if ok.any() else (0.0, 1.0))
        hi = hi if hi > lo else lo + 1.0
        table = _palettes.lut(cmap if isinstance(cmap, (str, list, tuple)) else "viridis", 255)
        codes = np.clip((np.nan_to_num(v, nan=lo) - lo) / (hi - lo) * 254 + 0.5, 0, 254).astype(np.int64)
        codes[~ok] = -1
        palette = [_palettes.rgb_to_hex(c) for c in table]
        stops = palette[:: 255 // 8] + [palette[-1]]
        return codes, palette, Legend(str(column), ramp=stops, vmin=float(lo), vmax=float(hi))
    codes, cats = pd.factorize(values, sort=False)
    cats = list(cats)
    if isinstance(cmap, dict):
        palette = [cmap.get(c, _MISSING) for c in cats]
    elif isinstance(cmap, (str, list, tuple)):
        palette = [_palettes.rgb_to_hex(c) for c in _palettes.lut(cmap, max(len(cats), 2))[: len(cats)]]
    else:
        palette = _palettes.distinct(len(cats))
    return codes.astype(np.int64), palette, Legend(str(column), classes=[(c, str(k)) for k, c in zip(cats, palette)])


def _geojson(geoms, props, precision: int = 6) -> str:
    """A GeoJSON FeatureCollection (text) of geometries in longitude/latitude, coordinates rounded,
    written by GEOS (several times faster than a GeoJSON driver)."""
    import shapely

    rounded = shapely.transform(geoms, lambda c: np.round(c, precision))
    geometries = shapely.to_geojson(rounded)
    if props is not None and len(props.columns):
        records = props.to_json(orient="records", lines=True, double_precision=6).splitlines()
    else:
        records = ["{}"] * len(geometries)
    return ('{"type":"FeatureCollection","features":['
            + ",".join(f'{{"type":"Feature","geometry":{g},"properties":{r}}}' for g, r in zip(geometries, records))
            + "]}")


_LAYER_CLASS = None


def _layer_class():
    """A Leaflet GeoJSON layer whose data enter the page after folium's templates (see _maps.render)
    and whose style and tooltips are computed in the browser from the features' properties."""
    global _LAYER_CLASS
    if _LAYER_CLASS is None:
        from branca.element import Template
        from folium.map import Layer

        class GeoJsonLayer(Layer):
            _template = Template("""
{% macro script(this, kwargs) %}
    var {{ this.get_name() }} = L.geoJson(null, {
        style: function(f) {
            var c = (f.properties && f.properties.__c) || {{ this.color|tojson }};
            return {color: {{ this.edge|tojson }} || c, weight: 1.5, fillColor: c,
                    fillOpacity: {{ this.fill_opacity }}, opacity: 0.9};
        },
        pointToLayer: function(f, latlng) {
            var c = (f.properties && f.properties.__c) || {{ this.color|tojson }};
            return L.circleMarker(latlng, {radius: 4, color: "#222222", weight: 1, fillColor: c,
                                           fillOpacity: 0.9});
        },
        onEachFeature: function(f, layer) {
            var keys = {{ this.fields|tojson }};
            if (keys.length) {
                layer.bindTooltip("<table>" + keys.map(function(k) {
                    return "<tr><th style='text-align:left;padding-right:8px'>" + k + "</th><td>" +
                           f.properties[k] + "</td></tr>"; }).join("") + "</table>", {sticky: false});
            }
        }
    });
    {{ this.get_name() }}.addData({{ this.token }});
    {{ this.get_name() }}.addTo({{ this._parent.get_name() }});
{% endmacro %}
""")

            def __init__(self, token, name, color, edge, fill_opacity, fields, show=True):
                super().__init__(name=name, overlay=True, control=True, show=show)
                self._name = "GeoJsonLayer"
                self.token, self.color, self.edge = token, color, edge
                self.fill_opacity, self.fields = fill_opacity, list(fields)

        _LAYER_CLASS = GeoJsonLayer
    return _LAYER_CLASS


def layer(gdf, *, column=None, cmap=None, categorical=None, color: str = "#ffd60a", opacity: float = 0.7,
          name: str | None = None, max_size: int = 2048, tooltip: bool = True):
    """A vector layer for a folium map: ("geojson", folium layer, legend) or ("image", Image, legend)."""
    import shapely

    name = name or (str(column) if column else "features")
    geoms = gdf.geometry.values
    keep_rows = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
    if not keep_rows.all():
        gdf = gdf[keep_rows]
    codes, palette, legend = _colours(gdf, column, cmap, categorical)
    vertices = int(shapely.get_num_coordinates(gdf.geometry.values).sum())
    if len(gdf) > MAX_FEATURES or vertices > MAX_VERTICES:
        return "image", _as_image(gdf, codes, palette, color, opacity, name, max_size, legend), legend
    if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    # simplify to about half a pixel of the whole extent; coordinates to about 0.1 m
    x0, y0, x1, y1 = gdf.total_bounds
    tol = max(x1 - x0, y1 - y0) / (max_size * 2.0)
    kinds = set(gdf.geometry.geom_type)
    geoms = gdf.geometry.values
    if tol > 0 and kinds - {"Point", "MultiPoint"}:
        geoms = shapely.simplify(geoms, tol, preserve_topology=True)
    keep = [c for c in gdf.columns if c != gdf.geometry.name][:MAX_TOOLTIP] if tooltip else []
    props = gdf[keep].copy()
    for c in keep:  # JSON-friendly values
        if not (_palettes.is_number(props[c].dtype) or props[c].dtype == bool):
            props[c] = props[c].astype(str)
    if codes is not None:
        hexes = np.array(palette + [_MISSING], dtype=object)
        props["__c"] = hexes[codes]  # -1: the missing colour, last
    text = _geojson(geoms, props)
    layer_cls = _layer_class()
    gj = layer_cls(None, name, color, None if codes is not None else color, opacity * 0.6, keep)
    gj._ls_payload = text  # put in place by _maps (after folium's templates)
    gj._ls_bounds = ((float(y0), float(x0)), (float(y1), float(x1)))
    return "geojson", gj, legend


def _as_image(gdf, codes, palette, color, opacity, name, max_size, legend) -> Image:
    """Very many features burnt into an image at the screen's resolution (Web Mercator, reprojected once)."""
    import geopandas as gpd
    import shapely
    import xarray as xr
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    from .._vector import rasterize
    from ..io import _with_georef

    if set(np.unique(shapely.get_type_id(gdf.geometry.values))) == {0}:  # points: their coordinates only
        return _points_image(gdf, codes, palette, color, opacity, name, max_size, legend)
    merc = to_web_mercator(gdf)
    x0, y0, x1, y1 = merc.total_bounds
    size = max(x1 - x0, y1 - y0) / max_size or 1.0
    w, h = max(1, math.ceil((x1 - x0) / size)), max(1, math.ceil((y1 - y0) / size))
    t = from_origin(x0, y1, size, size)
    x = x0 + size * (np.arange(w) + 0.5)
    y = y1 - size * (np.arange(h) + 0.5)
    like = xr.DataArray(np.zeros((h, w), np.uint8), coords={"y": y, "x": x}, dims=("y", "x"))
    like = _with_georef(like, crs="EPSG:3857", transform=t)
    if codes is None:
        colours, values = [color], np.ones(len(merc), np.int64)
    else:
        colours = palette + [_MISSING]
        values = np.where(codes < 0, len(palette), codes) + 1  # 1-based: 0 is no feature
    frame = gpd.GeoDataFrame({"__k": values}, geometry=merc.geometry.values, crs=merc.crs)
    burnt = np.asarray(rasterize(frame, like, field="__k", all_touched=True, dtype="uint16"))
    pal = np.zeros((len(colours) + 1, 4), np.uint8)
    for i, c in enumerate(colours, 1):
        pal[i] = (*_palettes.hex_to_rgb(c), int(255 * opacity))
    if len(pal) <= 256:  # palette indices
        pixels, palette_out = burnt.astype(np.uint8), pal
    else:
        pixels, palette_out = pal[burnt], None
    w_, s_, e_, n_ = transform_bounds("EPSG:3857", "EPSG:4326", x0, y1 - size * h, x0 + size * w, y1)
    return Image(pixels, ((s_, w_), (n_, e_)), legend, name, None, "EPSG:3857",
                 (x0, y1 - size * h, x0 + size * w, y1), palette_out)


def _points_image(gdf, codes, palette, color, opacity, name, max_size, legend) -> Image:
    """Very many points as an image: their coordinates reprojected (no geometries made), the cell of
    every point found directly, the last point of a cell drawn."""
    import shapely
    from rasterio.warp import transform_bounds

    xy = shapely.get_coordinates(gdf.geometry.values)
    x, y = xy[:, 0], xy[:, 1]
    if gdf.crs is not None:
        x, y = transform_xy(x, y, gdf.crs)
    x0, x1, y0, y1 = float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))
    size = max(x1 - x0, y1 - y0) / max_size or 1.0
    w, h = max(1, math.ceil((x1 - x0) / size)), max(1, math.ceil((y1 - y0) / size))
    col = np.minimum(((x - x0) / size).astype(np.int64), w - 1)
    row = np.minimum(((y1 - y) / size).astype(np.int64), h - 1)
    if codes is None:
        colours, values = [color], np.ones(len(x), np.int64)
    else:
        colours = palette + [_MISSING]
        values = np.where(codes < 0, len(palette), codes) + 1
    flat = row * w + col
    pixels = np.zeros(h * w, np.uint16)
    pixels[flat] = values  # several points in a cell: one of them
    pal = np.zeros((len(colours) + 1, 4), np.uint8)
    for i, c in enumerate(colours, 1):
        pal[i] = (*_palettes.hex_to_rgb(c), int(255 * opacity))
    pixels = pixels.reshape(h, w)
    if len(pal) <= 256:
        pixels, palette_out = pixels.astype(np.uint8), pal
    else:
        pixels, palette_out = pal[pixels], None
    w_, s_, e_, n_ = transform_bounds("EPSG:3857", "EPSG:4326", x0, y1 - size * h, x0 + size * w, y1)
    return Image(pixels, ((s_, w_), (n_, e_)), legend, name, None, "EPSG:3857",
                 (x0, y1 - size * h, x0 + size * w, y1), palette_out)



def transform_xy(x, y, src, dst="EPSG:3857"):
    from .._vector import transform_xy as _t

    return _t(x, y, src, dst)


def to_web_mercator(gdf):
    from .._vector import to_crs

    return to_crs(gdf, "EPSG:3857")
