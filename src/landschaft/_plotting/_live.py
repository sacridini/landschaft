# SPDX-License-Identifier: GPL-3.0-or-later
"""Detail on demand: while Python runs (a notebook's kernel, the window of a script), a map asks
for the part of a layer in view at the screen's resolution as it is zoomed in, through a bridge
that needs no server or port: a widget's messages in a notebook (anywidget), pywebview's
JavaScript API in a window. A page saved or opened elsewhere keeps the overview it holds."""

from __future__ import annotations

import base64
import functools
import math
import os
import threading
import uuid
from collections import OrderedDict

import numpy as np

#: Maps whose layers answer (the most recent; their data stay in memory as long as they are here).
MAX_MAPS = 32
#: Pixels of a detail image on its long side, at most.
MAX_DETAIL = 4096

_MAPS: OrderedDict[str, dict] = OrderedDict()
_LOCK = threading.Lock()


def enabled() -> bool:
    """Detail on demand, unless LANDSCHAFT_PLOT_LIVE=0."""
    return os.environ.get("LANDSCHAFT_PLOT_LIVE", "1") not in ("0", "false", "no")


def new_map() -> str:
    return uuid.uuid4().hex


def register(map_id: str, key: str, handler) -> None:
    """``handler(bbox, width, height)`` answers for layer ``key`` of map ``map_id``: ``bbox`` the
    view in Web Mercator (xmin, ymin, xmax, ymax), ``width`` x ``height`` its pixels on screen."""
    with _LOCK:
        layers = _MAPS.setdefault(map_id, {})
        layers[key] = handler
        _MAPS.move_to_end(map_id)
        while len(_MAPS) > MAX_MAPS:
            _MAPS.popitem(last=False)


def handle(msg: dict) -> dict:
    """The answer to a message of a map: ``{"op": "hello"}`` (whether its layers answer),
    ``{"op": "detail", "key", "bbox", "w", "h"}``, or ``{"op": "details", "keys": [...], ...}``
    (several layers in the same view, the bands of a series: answered over threads, as
    ``{"ok", "answers": [...]}``). Errors are answered, not raised."""
    try:
        map_id = str(msg.get("map"))
        with _LOCK:
            layers = _MAPS.get(map_id)
        if msg.get("op") == "hello":
            return {"ok": layers is not None}
        if layers is None:
            return {"ok": False, "error": "this map is no longer held (draw it again)"}
        bbox = [float(v) for v in msg["bbox"]]
        w = int(min(MAX_DETAIL, max(1, msg["w"])))
        h = int(min(MAX_DETAIL, max(1, msg["h"])))
        if msg.get("op") == "details":
            from concurrent.futures import ThreadPoolExecutor

            from .._threads import default_threads

            keys = [str(k) for k in msg.get("keys", [])]
            with ThreadPoolExecutor(max(1, min(default_threads(), len(keys)))) as pool:
                return {"ok": True, "answers": list(pool.map(lambda k: _one(layers, k, bbox, w, h), keys))}
        return _one(layers, msg.get("key"), bbox, w, h)
    except Exception as err:  # noqa: BLE001 - reported to the page
        return {"ok": False, "error": f"{type(err).__name__}: {err}"}


def _one(layers: dict, key, bbox, w, h) -> dict:
    if key not in layers:
        return {"ok": False, "error": "this map is no longer held (draw it again)"}
    try:
        out = layers[key](bbox, w, h)
    except Exception as err:  # noqa: BLE001 - reported to the page
        return {"ok": False, "error": f"{type(err).__name__}: {err}"}
    return {"ok": out is not None, **(out or {})}


# --- the answers of layers ----------------------------------------------------------------------------


def _png_url(img) -> str:
    return "data:image/png;base64," + base64.b64encode(img.png()).decode("ascii")


def _image_answer(img) -> dict:
    out = {"url": _png_url(img), "ext": [float(v) for v in img.mercator_extent]}
    if img.palette is not None and img.lookup is not None:
        out["palette"] = img.palette[:, :3].tolist()
        out["lookup"] = {k: (round(v, 12) if isinstance(v, float) else v) for k, v in img.lookup.items()}
    return out


@functools.lru_cache(64)
def _bounds_in(crs: str, bbox: tuple) -> tuple:
    """A Web Mercator box in a CRS (the same view asked for every band of a series: once), through
    the transformation made once for the CRS (rasterio's transform_bounds sets PROJ up every time)."""
    from ._raster import _to_source

    return tuple(_to_source(crs).transform_bounds(*bbox, densify_pts=21))


def _source_window(da, bbox):
    """The cells of a raster under a Web Mercator box (and one more around), or None outside it."""
    crs = da.rio.crs
    x0, y0, x1, y1 = _bounds_in(crs.to_wkt(), tuple(bbox)) if crs is not None else bbox  # not to_string(): slow
    xs, ys = np.asarray(da.x.values), np.asarray(da.y.values)

    def span(c, lo, hi):
        asc = c[0] <= c[-1]
        a = np.searchsorted(c, lo, "left") if asc else len(c) - np.searchsorted(c[::-1], hi, "right")
        b = np.searchsorted(c, hi, "right") if asc else len(c) - np.searchsorted(c[::-1], lo, "left")
        return max(0, int(a) - 1), min(len(c), int(b) + 1)

    ix, iy = span(xs, x0, x1), span(ys, y0, y1)
    if ix[1] - ix[0] < 1 or iy[1] - iy[0] < 1:
        return None
    return da.isel(x=slice(*ix), y=slice(*iy))


def raster_handler(da, style: dict):
    """The part of a raster in view, coloured as its overview (``style``: _raster.render's choices)."""
    from ._raster import render

    def answer(bbox, w, h):
        sub = _source_window(da, bbox)
        if sub is None:
            return None
        # the cells in view at the screen's resolution (the long side of the view, in cells)
        img = render(sub, max_size=max(w, h, 1), style=style, title=style.get("title"), view=(bbox, w, h))
        return _image_answer(img)

    return answer


def vector_image_handler(gdf, codes, palette, color, legend, name, opacity):
    """The features in view burnt at the screen's resolution (a layer too large for WebGL)."""
    import shapely

    from . import _vectors

    held = {}

    def answer(bbox, w, h):
        if not held:  # made at the first question: drawing the map does not wait for them
            held["merc"] = _vectors.to_web_mercator(gdf)
            held["tree"] = shapely.STRtree(np.asarray(held["merc"].geometry.values, dtype=object))
        merc, tree = held["merc"], held["tree"]
        idx = np.sort(tree.query(shapely.box(*bbox)))
        if not len(idx):
            return None
        part = merc.iloc[idx]
        size = max((bbox[2] - bbox[0]) / w, (bbox[3] - bbox[1]) / h)
        sub_codes = None if codes is None else np.asarray(codes)[idx]
        clipped = part.geometry.values
        if shapely.get_type_id(clipped).max() > 0:  # lines and polygons: cut to the view (a little more)
            pad = 2 * size
            clipped = shapely.clip_by_rect(clipped, bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        frame = part.set_geometry(clipped)
        keep = ~shapely.is_empty(frame.geometry.values)
        frame, sub_codes = frame[keep], (None if sub_codes is None else sub_codes[keep])
        if not len(frame):
            return None
        long_side = max(frame.total_bounds[2] - frame.total_bounds[0], frame.total_bounds[3] - frame.total_bounds[1])
        cells = max(1, math.ceil(long_side / size))
        if set(np.unique(shapely.get_type_id(frame.geometry.values)).tolist()) == {0}:
            xy = shapely.get_coordinates(frame.geometry.values)
            burnt, colours, extent = _vectors.burn_points(xy[:, 0], xy[:, 1], sub_codes, palette, color, cells)
        else:
            burnt, colours, extent = _vectors.burn(frame, sub_codes, palette, color, cells)
        return _image_answer(_vectors._image(burnt, colours, legend, name, opacity, extent))

    return answer


def gl_handler(gdf, codes, palette, color, opacity, name, fields, column, legend):
    """The features in view, simplified to the screen's resolution, as a WebGL layer of their own."""
    import shapely

    from . import _gl

    crs = gdf.crs
    held = {}

    def answer(bbox, w, h):
        from rasterio.warp import transform_bounds

        if "tree" not in held:  # made at the first question
            held["tree"] = shapely.STRtree(np.asarray(gdf.geometry.values, dtype=object))
        tree = held["tree"]
        b = transform_bounds("EPSG:3857", crs, *bbox) if crs is not None else bbox
        idx = np.sort(tree.query(shapely.box(*b)))
        if not len(idx) or len(idx) > _gl.MAX_FEATURES:
            return None
        part = gdf.iloc[idx]
        sub_codes = None if codes is None else np.asarray(codes)[idx]
        # an eighth of a pixel of this view: max_size such that extent / (max_size * DETAIL) is it
        view = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        merc = part.geometry.to_crs(3857) if crs is not None else part.geometry
        x0, y0, x1, y1 = merc.total_bounds
        max_size = max(256, math.ceil(max(x1 - x0, y1 - y0) / view * max(w, h)))
        layer = _gl.layer(part, sub_codes, palette, color, opacity, name, max_size, fields, column, legend)
        if layer is None:
            return None
        return {"spec": layer.spec, "data": layer._ls_payload, "bbox": bbox}

    return answer


# --- the bridges ------------------------------------------------------------------------------------------


class WindowApi:
    """pywebview's JavaScript API of a window: ``window.pywebview.api.ls_live(msg)``."""

    def ls_live(self, msg):
        return handle(msg)


_BRIDGE = None

_ESM = """
export default {
  render({ model, el }) {
    el.style.cssText = "font: 11px sans-serif; color: #8a8985; margin: 0; height: 0; overflow: hidden";
    const sources = new Map();
    function onMessage(e) {
      const d = e.data;
      if (!d || d.lsLive === undefined || !model.get("maps").includes(d.lsLive.map)) { return; }
      sources.set(d.lsLive.req, e.source);
      model.send({ req: d.lsLive.req, msg: d.lsLive });
    }
    model.on("msg:custom", (m) => {
      const src = sources.get(m.req);
      sources.delete(m.req);
      if (src) { src.postMessage({ lsLiveReply: { req: m.req, map: m.map, data: m.data } }, "*"); }
    });
    window.addEventListener("message", onMessage);
    // the maps already on the page: tell them the kernel answers
    for (const f of document.querySelectorAll("iframe")) {
      try { f.contentWindow.postMessage({ lsLiveHere: model.get("maps") }, "*"); } catch (err) { /* another origin */ }
    }
    return () => window.removeEventListener("message", onMessage);
  }
};
"""


def bridge(map_ids: list[str]):
    """A widget (anywidget) that carries the messages of these maps to the kernel and back; None
    without anywidget."""
    global _BRIDGE
    try:
        import anywidget
        import traitlets
    except ImportError:
        return None
    if _BRIDGE is None:

        class LiveBridge(anywidget.AnyWidget):
            _esm = _ESM
            maps = traitlets.List(traitlets.Unicode()).tag(sync=True)

            def __init__(self, maps):
                super().__init__(maps=list(maps))
                self.on_msg(self._answer)

            def _answer(self, _widget, content, _buffers):
                msg = content.get("msg") or {}
                self.send({"req": content.get("req"), "map": msg.get("map"), "data": handle(msg)})

        _BRIDGE = LiveBridge
    return _BRIDGE(map_ids)
