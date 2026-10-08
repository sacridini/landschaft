# SPDX-License-Identifier: GPL-3.0-or-later
"""Large vector layers drawn with WebGL (_js/lsgl.js): the features in Web Mercator, simplified,
quantized, polygons cut into triangles in C++, packed into one deflated payload of the page."""

from __future__ import annotations

import base64
import functools
import json
import zlib
from pathlib import Path

import numpy as np

from . import _palettes

#: Up to this many features (and vertices once simplified) a layer is drawn with WebGL; above, an image.
MAX_FEATURES = 2_000_000
MAX_VERTICES = 8_000_000
#: The features are simplified to this fraction of a pixel of max_size (finer than GeoJSON's half
#: pixel: the layer is drawn again at every zoom) and their coordinates rounded to half that.
DETAIL = 8


@functools.lru_cache(1)
def script() -> str:
    return "<script>" + (Path(__file__).parent / "_js" / "lsgl.js").read_text(encoding="utf-8") + "</script>"


def _planes(values: np.ndarray) -> bytes:
    """uint32 values as four planes of bytes (low bytes first): small numbers deflate well."""
    v = np.ascontiguousarray(values, dtype="<u4")
    return v.view(np.uint8).reshape(-1, 4).T.tobytes()


def _zigzag_deltas(values: np.ndarray) -> np.ndarray:
    d = np.diff(values.astype(np.int64), prepend=0).astype(np.int32)
    return ((d.astype(np.int64) << 1) ^ (d.astype(np.int64) >> 31)).astype(np.uint32)


#: Bytes deflated as one piece (the pieces of a large array are deflated in parallel).
CHUNK = 1 << 21


class _Packer:
    """Arrays deflated in pieces over threads (zlib lets go of the GIL), in one payload."""

    def __init__(self):
        self.parts, self.arrays = [], {}

    def add(self, name, values, delta=False):
        values = np.asarray(values)
        self.arrays[name] = [len(self.parts), int(values.size), bool(delta)]
        self.parts.append(_planes(_zigzag_deltas(values) if delta else values))

    def raw(self, data: bytes) -> int:
        self.parts.append(data)
        return len(self.parts) - 1

    def payload(self):
        """The payload (base64) and, for each part, the [offset, length] of its deflated pieces."""
        from concurrent.futures import ThreadPoolExecutor

        from .._threads import default_threads

        pieces = [(i, bytes(memoryview(part)[k:k + CHUNK])) for i, part in enumerate(self.parts)
                  for k in range(0, max(1, len(part)), CHUNK)]
        threads = min(default_threads(), len(pieces))
        if threads > 1:
            with ThreadPoolExecutor(threads) as pool:
                packed = list(pool.map(lambda x: zlib.compress(x[1], 6), pieces))
        else:
            packed = [zlib.compress(x[1], 6) for x in pieces]
        where, at = [[] for _ in self.parts], 0
        for (i, _), b in zip(pieces, packed):
            where[i].append([at, len(b)])
            at += len(b)
        return base64.b64encode(b"".join(packed)).decode("ascii"), where


def _quantized(coords, x0, y0, q):
    """(x..., y...) as int32 cells of q metres from the origin."""
    xy = np.empty(2 * len(coords), np.int32)
    xy[: len(coords)] = np.rint((coords[:, 0] - x0) / q)
    xy[len(coords):] = np.rint((coords[:, 1] - y0) / q)
    return xy


def _feature_steps(owner):
    """The feature of each part as steps from the previous one (0 or more: the parts are in order)."""
    return np.diff(np.asarray(owner, np.int64), prepend=0)


def prepare(gdf, codes, palette, color, max_size):
    """The arrays of a layer, or None when it is too large for WebGL (more than MAX_FEATURES, or
    MAX_VERTICES once simplified)."""
    import shapely

    from .. import _core
    from ._vectors import simplify_parts

    if len(gdf) > MAX_FEATURES:
        return None
    colours = [color] if codes is None else palette + ["#808080"]
    fcol = np.zeros(len(gdf), np.uint32) if codes is None else np.where(codes < 0, len(palette), codes)
    palette_rgb = [list(_palettes.hex_to_rgb(c)) for c in colours]
    pk = _Packer()
    if set(np.unique(shapely.get_type_id(gdf.geometry.values)).tolist()) == {0}:  # points: no geometries made
        from ._vectors import transform_xy

        xy = shapely.get_coordinates(gdf.geometry.values)
        x, y = transform_xy(xy[:, 0], xy[:, 1], gdf.crs) if gdf.crs is not None else (xy[:, 0], xy[:, 1])
        x0, x1, y0, y1 = float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))
        q = max(x1 - x0, y1 - y0) / (max_size * DETAIL * 2) or 1.0
        pk.add("txy", _quantized(np.column_stack([x, y]), x0, y0, q), delta=True)
        pk.add("tfeat", np.r_[0, np.ones(len(x) - 1, np.uint32)])
        pk.add("fcol", fcol)
        return pk, palette_rgb, (x0, y0, q), (x0, y0, x1, y1)
    # simplified in the layer's own CRS (to the same share of its extent), and only the coordinates
    # left reprojected: no geometries made twice
    geoms = np.asarray(gdf.geometry.values, dtype=object)
    bx0, by0, bx1, by1 = (float(v) for v in shapely.total_bounds(geoms))
    tol = max(bx1 - bx0, by1 - by0) / (max_size * DETAIL)
    parts, owner = shapely.get_parts(geoms, return_index=True)
    kind = shapely.get_type_id(parts)
    ring = kind == 2  # a LinearRing: a line
    if ring.any():
        parts = parts.copy()
        parts[ring] = shapely.linestrings([shapely.get_coordinates(r) for r in parts[ring]])
        kind = shapely.get_type_id(parts)
    polys, lines, points = kind == 3, kind == 1, kind == 0
    pieces = {}
    owners = {}
    if polys.any():
        sp, owners["p"] = simplify_parts(parts[polys], owner[polys], tol, (3,))
        _, coords, offsets = shapely.to_ragged_array(sp)
        pieces["p"] = (coords, offsets)
    if lines.any():
        sl, owners["l"] = simplify_parts(parts[lines], owner[lines], tol, (1,))
        _, coords, offsets = shapely.to_ragged_array(sl)
        pieces["l"] = (coords, offsets)
    if points.any():
        pieces["t"] = (shapely.get_coordinates(parts[points]), None)
    if sum(len(c) for c, _ in pieces.values()) > MAX_VERTICES:
        return None
    if gdf.crs is not None:
        from ._vectors import transform_xy

        for k, (c, off) in pieces.items():
            x, y = transform_xy(c[:, 0], c[:, 1], gdf.crs)
            pieces[k] = (np.column_stack([x, y]), off)
    every = np.concatenate([c[:, :2] for c, _ in pieces.values()])
    x0, y0 = (float(v) for v in every.min(axis=0))
    x1, y1 = (float(v) for v in every.max(axis=0))
    q = max(x1 - x0, y1 - y0) / (max_size * DETAIL * 2) or 1.0
    if "p" in pieces:
        coords, (ring_off, poly_off) = pieces["p"]
        xy = _quantized(coords, x0, y0, q)
        n = len(coords)
        flat = np.empty((n, 2), np.float64)
        flat[:, 0], flat[:, 1] = xy[:n], xy[n:]
        idx, counts = _core.io_triangulate(flat, np.asarray(ring_off, np.int64), np.asarray(poly_off, np.int64))
        pk.add("pxy", xy, delta=True)
        pk.add("pring", np.diff(ring_off))
        pk.add("prings", np.diff(poly_off))
        pk.add("pfeat", _feature_steps(owners["p"]))
        pk.add("ptri", counts)
        pk.add("pidx", idx, delta=True)
    if "l" in pieces:
        coords, (line_off,) = pieces["l"]
        pk.add("lxy", _quantized(coords, x0, y0, q), delta=True)
        pk.add("llen", np.diff(line_off))
        pk.add("lfeat", _feature_steps(owners["l"]))
    if "t" in pieces:
        pk.add("txy", _quantized(pieces["t"][0], x0, y0, q), delta=True)
        pk.add("tfeat", _feature_steps(owner[points]))
    pk.add("fcol", fcol)
    return pk, palette_rgb, (x0, y0, q), (x0, y0, x1, y1)


def _props(gdf, fields) -> bytes:
    """The tooltip's columns, column by column, as JSON (NaN, None: null), written by pandas (C)."""
    parts = []
    for c in fields:
        s = gdf[c]
        if not (_palettes.is_number(s.dtype) or s.dtype == bool):
            text = s.astype(str).astype(object)
            text[s.isna().to_numpy()] = None
            s = text
        parts.append(json.dumps(str(c)) + ":" + s.reset_index(drop=True).to_json(orient="values", double_precision=6))
    return ("{" + ",".join(parts) + "}").encode("utf-8")


_LAYER_CLASS = None


def _layer_class():
    """A folium layer whose script makes an L.LsGL with its data (put in after folium's templates)."""
    global _LAYER_CLASS
    if _LAYER_CLASS is None:
        from branca.element import Template
        from folium.map import Layer

        class GlLayer(Layer):
            _template = Template("""
{% macro script(this, kwargs) %}
    var {{ this.get_name() }} = new L.LsGL({{ this.spec|tojson }}, "{{ this.token }}",
        {{ ({"pane": this.pane} if this.pane else {})|tojson }});
    {% if this.show %}{{ this.get_name() }}.addTo({{ this._parent.get_name() }});{% endif %}
{% endmacro %}
""")

            def __init__(self, spec, name, show=True):
                super().__init__(name=name, overlay=True, control=True, show=show)
                self._name = "LsGL"
                self.spec, self.token, self.pane = spec, None, None

        _LAYER_CLASS = GlLayer
    return _LAYER_CLASS


def layer(gdf, codes, palette, color, opacity, name, max_size, fields, column, legend=None):
    """A WebGL layer for a folium map, or None when the layer is too large for one. Without the
    colouring column among ``fields``, the browser reads a feature's value back from its colour
    (``legend``: its class, or its place on the ramp)."""
    from ._raster import ramp_lookup

    prepared = prepare(gdf, codes, palette, color, max_size)
    if prepared is None:
        return None
    pk, colours, (x0, y0, q), (bx0, by0, bx1, by1) = prepared
    props = pk.raw(_props(gdf, fields)) if fields else None
    fields = [str(c) for c in fields]
    payload, where = pk.payload()
    arrays = {k: [where[i], n, delta] for k, (i, n, delta) in pk.arrays.items()}
    spec = {"name": name, "origin": [x0, y0], "q": q, "palette": colours, "arrays": arrays,
            "props": where[props] if props is not None else None, "fields": fields,
            "column": str(column) if column is not None and str(column) in fields else None,
            "fill": round(opacity * 0.6, 4), "line": 0.9}
    if legend is not None:
        spec["lookup"] = ramp_lookup(legend) if legend.ramp else {"labels": [lab for _, lab in legend.classes]}
    gl = _layer_class()(spec, name)
    gl._ls_payload = payload
    from rasterio.warp import transform_bounds

    w, s, e, n = transform_bounds("EPSG:3857", "EPSG:4326", bx0, by0, bx1, by1)
    gl._ls_bounds = ((s, w), (n, e))
    gl._ls_merc = (bx0, by0, bx1, by1)
    gl._ls_res = 2 * q  # the simplification's tolerance: finer than that, detail is asked for (live)
    return gl
