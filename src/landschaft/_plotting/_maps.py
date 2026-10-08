# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactive maps (folium / Leaflet): base maps, image and vector layers, legends."""

from __future__ import annotations

import base64
import functools
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

from ._raster import Image, Legend

#: Leaflet (BSD-2-Clause, see _js/LICENSE-leaflet.txt), put in every page: no CDN, a saved map opens offline.
_JS = Path(__file__).parent / "_js"

BASEMAPS = {
    "satellite": ("Esri.WorldImagery", "Satellite"),
    "osm": ("OpenStreetMap.Mapnik", "OpenStreetMap"),
    "topo": ("Esri.WorldTopoMap", "Topographic"),
    "light": ("Esri.WorldGrayCanvas", "Light grey"),
    "relief": ("Esri.WorldShadedRelief", "Shaded relief"),
}

_CSS = """
<style>
.leaflet-image-layer { image-rendering: pixelated; image-rendering: crisp-edges; }
/* a long list of layers (a stack of many bands) scrolls within the map */
.leaflet-control-layers-expanded .leaflet-control-layers-list { max-height: calc(100vh - 70px); overflow-y: auto; }
.ls-legends { position: absolute; bottom: 24px; right: 10px; z-index: 1000; display: flex; flex-direction: column;
  gap: 6px; max-height: 70%; overflow-y: auto; font: 12px/1.35 Inter, "Segoe UI", Helvetica, Arial, sans-serif; }
.ls-legend { background: rgba(255,255,255,0.92); color: #1d1d1b; border-radius: 6px; padding: 6px 9px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.3); min-width: 140px; }
.ls-legend b { display: block; margin-bottom: 4px; font-weight: 600; }
.ls-legend .row { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.ls-legend .sw { width: 14px; height: 12px; border: 1px solid rgba(0,0,0,0.25); flex: none; }
.ls-legend .ramp { height: 12px; border: 1px solid rgba(0,0,0,0.25); }
.ls-legend .ends { display: flex; justify-content: space-between; color: #4a4a48; }
.ls-legend .head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.ls-legend .head b { margin: 0; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 220px; }
.ls-legend input[type=range] { width: 56px; height: 12px; margin: 0; accent-color: #555; cursor: pointer; opacity: 0.55; }
.ls-legend:hover input[type=range] { opacity: 1; }
.ls-legend .bands { display: flex; align-items: center; gap: 6px; margin-top: 6px; }
.ls-legend .bands input { flex: 1; min-width: 70px; height: 12px; margin: 0; accent-color: #555; cursor: pointer; }
.ls-legend .bands span { min-width: 3em; max-width: 120px; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; font-variant-numeric: tabular-nums; }
.ls-legend .bands button { border: 1px solid #bbb; background: #fff; border-radius: 4px; width: 22px; height: 20px;
  padding: 0; font-size: 10px; line-height: 18px; cursor: pointer; color: #333; }
.ls-full a { font: bold 18px/30px sans-serif; text-align: center; cursor: pointer; }
.ls-coords { background: rgba(255,255,255,0.88); padding: 2px 7px; margin: 0 0 4px 10px !important; border-radius: 4px;
  font: 11px/1.45 Inter, "Segoe UI", Helvetica, Arial, sans-serif; color: #1d1d1b; font-variant-numeric: tabular-nums;
  box-shadow: 0 1px 3px rgba(0,0,0,0.25); max-width: 320px; }
.ls-coords .ll { color: #6b6a66; }
.ls-coords b { font-weight: 600; margin-right: 4px; }
.ls-coords:empty { display: none; }
/* the base map's credits on one line (all of them on hover): they hid the scale and the legends */
.leaflet-control-attribution { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 55vw; }
.leaflet-control-attribution:hover { white-space: normal; }
.ls-swipe { position: absolute; top: 0; bottom: 0; width: 0; z-index: 800; pointer-events: none; }
.ls-swipe::before { content: ""; position: absolute; top: 0; bottom: 0; left: -1.5px; width: 3px; background: #fff;
  box-shadow: 0 0 4px rgba(0,0,0,0.5); }
.ls-swipe-knob { position: absolute; top: 50%; left: -19px; width: 38px; height: 26px; margin-top: -13px;
  border-radius: 13px; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.45); cursor: ew-resize;
  pointer-events: auto; font: 10px/26px sans-serif; text-align: center; color: #444; letter-spacing: -1px;
  touch-action: none; user-select: none; }
.ls-swipe-tag { position: absolute; top: 52px; padding: 2px 8px; border-radius: 4px; white-space: nowrap;
  background: rgba(255,255,255,0.9); box-shadow: 0 1px 3px rgba(0,0,0,0.3); font: 600 12px Inter, "Segoe UI",
  Helvetica, Arial, sans-serif; color: #1d1d1b; }
.ls-swipe-tag.left { right: 10px; }
.ls-swipe-tag.left::before { content: "\\25C0  "; color: #888; }
.ls-swipe-tag.right { left: 10px; }
.ls-swipe-tag.right::after { content: "  \\25B6"; color: #888; }
.ls-pop th { text-align: left; padding: 1px 10px 1px 0; font-weight: 600; }
.ls-pop td { font-variant-numeric: tabular-nums; }
</style>
"""


@functools.lru_cache(1)
def _leaflet() -> str:
    """Leaflet's style and script, the layer control's icons as data URLs."""
    css = (_JS / "leaflet.css").read_text(encoding="utf-8")
    for png in ("layers-2x.png", "layers.png"):
        url = "data:image/png;base64," + base64.b64encode((_JS / png).read_bytes()).decode("ascii")
        css = css.replace(f"url(images/{png})", f"url({url})")
    return f"<style>{css}</style>\n<script>{(_JS / 'leaflet.js').read_text(encoding='utf-8')}</script>"


@dataclass
class Group:
    """Layers that share a legend card (a raster, the bands of a stack, a vector layer): the card
    shows while one of them is on the map, and its slider sets their opacity."""

    title: str
    legend: Legend | None
    opacity: float
    colour: str | None = None                   # a vector layer drawn in one colour
    items: list = field(default_factory=list)   # Images and vector layers
    layers: list = field(default_factory=list)  # their layers on the map
    bands: list | None = None                   # a stack: the band names, one band shown at a time


_CONTROLS = None

# The value under the cursor: each image's palette is opaque with one colour per index, so the
# colour of a pixel, read from the image drawn on a canvas, gives back its index and the index
# its class or (on a ramp of 255 colours) its value, to 1/254 of the ramp.
_SCRIPT = """
{% macro script(this, kwargs) %}
(function(map, L_, P, G, LC) {
    var Full = L.Control.extend({options: {position: "topleft"}, onAdd: function() {
        var bar = L.DomUtil.create("div", "leaflet-bar leaflet-control ls-full");
        var a = L.DomUtil.create("a", "", bar);
        a.innerHTML = "&#x26F6;"; a.title = "Full screen"; a.href = "#"; a.setAttribute("role", "button");
        L.DomEvent.disableClickPropagation(bar);
        L.DomEvent.on(a, "click", function(e) {
            L.DomEvent.preventDefault(e);
            if (document.fullscreenElement) { document.exitFullscreen(); }
            else if (document.documentElement.requestFullscreen) { document.documentElement.requestFullscreen(); }
        });
        return bar;
    }});
    new Full().addTo(map);
    document.addEventListener("fullscreenchange", function() { map.invalidateSize(); });

    // A frame loaded while hidden (a notebook's output not yet on screen) has no size: fitted
    // then, the map stays on the whole world at zoom 0. The view is fitted again once it has one.
    var fit = map.fitBounds, unfitted = null, box = map.getContainer();
    map.fitBounds = function(b, o) {
        unfitted = (box.clientWidth && box.clientHeight) ? null : [b, o];
        return fit.call(map, b, o);
    };
    if (window.ResizeObserver) {
        new ResizeObserver(function() {
            if (unfitted && box.clientWidth && box.clientHeight) {
                var f = unfitted; unfitted = null;
                map.invalidateSize(false);
                fit.call(map, f[0], L.extend({}, f[1], {animate: false}));
            }
        }).observe(box);
    }

    var esc = function(v) { return String(v).replace(/[&<>"]/g, function(c) {
        return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); };
    var fmt = function(x) {
        var a = Math.abs(x);
        return (a !== 0 && (a >= 1e6 || a < 1e-3)) ? x.toExponential(3) : String(+x.toPrecision(4));
    };
    var keys = P.map(function(pal) {
        var k = {};
        pal.forEach(function(c, i) { if (i > 0) { k[(c[0] << 16) | (c[1] << 8) | c[2]] = i; } });
        return k;
    });
    function index(l, d) {
        var i = keys[l.p][(d[0] << 16) | (d[1] << 8) | d[2]];
        if (i !== undefined) { return i; }
        var best = 1, dist = Infinity;  // a colour blended by the browser: the nearest
        P[l.p].forEach(function(c, j) {
            if (j === 0) { return; }
            var e = (c[0] - d[0]) * (c[0] - d[0]) + (c[1] - d[1]) * (c[1] - d[1]) + (c[2] - d[2]) * (c[2] - d[2]);
            if (e < dist) { dist = e; best = j; }
        });
        return best;
    }
    function value(l, i) {
        var u = l.u;
        if (u.labels) { return u.labels[i]; }
        if (i <= 1 && u.under) { return "\\u2264 " + fmt(u.vmin); }
        if (i >= 255 && u.over) { return "\\u2265 " + fmt(u.vmax); }
        var x = (i - 1) / 254;
        var v = u.log ? Math.pow(10, Math.log10(u.vmin) + x * (Math.log10(u.vmax) - Math.log10(u.vmin)))
                      : u.vmin + x * (u.vmax - u.vmin);
        return "\\u2248 " + fmt(v);
    }
    function read(l, latlng) {  // the value of a layer at a point; null: no data; undefined: outside
        var lay = window[l.v], e = l.ext;
        if (!map.hasLayer(lay)) { return undefined; }
        var p = L.CRS.EPSG3857.project(latlng);
        if (p.x < e[0] || p.x >= e[2] || p.y <= e[1] || p.y > e[3]) { return undefined; }
        if (!l.ctx) {
            var img = lay.getElement();
            if (!img || !img.complete || !img.naturalWidth) { return undefined; }
            var c = document.createElement("canvas");
            c.width = img.naturalWidth; c.height = img.naturalHeight;
            l.ctx = c.getContext("2d", {willReadFrequently: true});
            l.ctx.drawImage(img, 0, 0);
        }
        var w = l.ctx.canvas.width, h = l.ctx.canvas.height;
        var col = Math.min(w - 1, Math.floor((p.x - e[0]) / (e[2] - e[0]) * w));
        var row = Math.min(h - 1, Math.floor((e[3] - p.y) / (e[3] - e[1]) * h));
        var d = l.ctx.getImageData(col, row, 1, 1).data;
        return d[3] === 0 ? null : value(l, index(l, d));
    }
    function rows(latlng, full) {
        var out = [];
        L_.forEach(function(l) {
            var v = read(l, latlng);
            if (v !== undefined) { out.push([l.n, v === null ? "no data" : v]); }
        });
        // large vector layers (WebGL): the feature under the cursor, its columns on a click
        (window.LS_GL || []).forEach(function(l) {
            if (l._map === map) { out.push.apply(out, l.rowsAt(latlng, full)); }
        });
        return out;
    }

    var Readout = L.Control.extend({options: {position: "bottomleft"}, onAdd: function() {
        var div = L.DomUtil.create("div", "leaflet-control ls-coords");
        var last = null, queued = false;
        map.on("mousemove", function(e) {  // a pixel read per visible layer, once a frame
            last = e.latlng;
            if (queued) { return; }
            queued = true;
            requestAnimationFrame(function() {
                queued = false;
                if (!last) { return; }
                var ll = last;
                var h = "<span class='ll'>" + ll.lat.toFixed(5) + ", " + ll.lng.toFixed(5) + "</span>";
                rows(ll).forEach(function(r) { h += "<div><b>" + esc(r[0]) + "</b> " + esc(r[1]) + "</div>"; });
                div.innerHTML = h;
            });
        });
        map.on("mouseout", function() { last = null; div.innerHTML = ""; });
        return div;
    }});
    new Readout().addTo(map);
    if (L_.length || (window.LS_GL || []).length) {
        map.on("click", function(e) {
            var r = rows(e.latlng, true);
            if (!r.length) { return; }
            var h = "<table class='ls-pop'>" + r.map(function(x) {
                return "<tr><th>" + esc(x[0]) + "</th><td>" + esc(x[1]) + "</td></tr>"; }).join("") +
                "<tr><th>lat, lon</th><td>" + e.latlng.lat.toFixed(6) + ", " + e.latlng.lng.toFixed(6) +
                "</td></tr></table>";
            L.popup({maxWidth: 360}).setLatLng(e.latlng).setContent(h).openOn(map);
        });
    }

    G.forEach(function(g, i) {
        var card = document.querySelector(".ls-legend[data-g='" + i + "']");
        if (!card) { return; }
        var lays = g.n.map(function(n) { return window[n]; });
        if (g.b && lays.length > 1) {  // a stack: one entry in the layer control, the band chosen on the card
            var grp = L.layerGroup(), cur = 0, timer = null;
            lays.forEach(function(x) { if (map.hasLayer(x)) { map.removeLayer(x); } LC.removeLayer(x); });
            grp.addLayer(lays[0]).addTo(map);
            LC.addOverlay(grp, g.t);
            var range = card.querySelector(".bands input"), lab = card.querySelector(".bands span");
            var play = card.querySelector(".bands button");
            var show = function(k) {
                grp.removeLayer(lays[cur]); cur = k; grp.addLayer(lays[cur]);
                range.value = k; lab.textContent = g.b[k];
            };
            range.addEventListener("input", function() { show(+range.value); });
            play.addEventListener("click", function() {
                if (timer) { clearInterval(timer); timer = null; play.innerHTML = "&#9654;"; return; }
                play.innerHTML = "&#10073;&#10073;";
                timer = setInterval(function() { show((cur + 1) % lays.length); }, 900);
            });
        }
        var sync = function() {
            card.style.display = lays.some(function(x) { return map.hasLayer(x); }) ? "" : "none";
        };
        map.on("overlayadd overlayremove", sync);
        sync();
        var r = card.querySelector(".head input");
        if (!r) { return; }
        r.addEventListener("input", function() {
            var o = r.value / 100;
            lays.forEach(function(x) {
                if (x.setOpacity) { x.setOpacity(o); }
                else if (x.eachLayer) {
                    x.eachLayer(function(f) {
                        if (f.setStyle) {
                            f.setStyle(f instanceof L.CircleMarker ? {fillOpacity: o, opacity: o}
                                                                   : {fillOpacity: o * 0.6, opacity: Math.min(1, o + 0.2)});
                        }
                    });
                }
            });
        });
    });
})({{ this._parent.get_name() }}, {{ this.layers|tojson }}, {{ this.palettes|tojson }}, {{ this.groups|tojson }},
  {{ this.control }});
{% endmacro %}
"""


def _controls(layers: list, palettes: list, groups: list, control: str):
    """Full screen (top left); the coordinates and the values under the cursor (bottom left), and
    in a popup on a click; the legend cards following their layers, with their opacity."""
    global _CONTROLS
    if _CONTROLS is None:
        from branca.element import MacroElement, Template

        class Controls(MacroElement):
            _template = Template(_SCRIPT)

            def __init__(self, layers, palettes, groups, control):
                super().__init__()
                self._name = "LsControls"
                self.layers, self.palettes, self.groups, self.control = layers, palettes, groups, control

        _CONTROLS = Controls
    return _CONTROLS(layers, palettes, groups, control)


_SWIPE = """
{% macro script(this, kwargs) %}
(function(map, names) {
    var panes = [map.getPane("lsleft"), map.getPane("lsright")], c = map.getContainer();
    var bar = L.DomUtil.create("div", "ls-swipe", c);
    var knob = L.DomUtil.create("div", "ls-swipe-knob", bar);
    knob.innerHTML = "&#x25C0;&#x25B6;"; knob.title = "Drag to compare";
    var tags = names.map(function(n, i) {
        var t = L.DomUtil.create("div", "ls-swipe-tag " + (i ? "right" : "left"), bar);
        t.textContent = n;
        return t;
    });
    var x = 0.5, drag = false;
    function clip() {
        var size = map.getSize(), nw = map.containerPointToLayerPoint([0, 0]);
        var se = map.containerPointToLayerPoint(size), cx = nw.x + size.x * x;
        panes[0].style.clip = "rect(" + [nw.y, cx, se.y, nw.x].join("px,") + "px)";
        panes[1].style.clip = "rect(" + [nw.y, se.x, se.y, cx].join("px,") + "px)";
        bar.style.left = Math.round(size.x * x) + "px";
    }
    function move(e) {
        var r = c.getBoundingClientRect();
        x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
        clip();
    }
    L.DomEvent.disableClickPropagation(bar);
    L.DomEvent.on(knob, "pointerdown", function(e) { L.DomEvent.stop(e); drag = true; map.dragging.disable(); });
    document.addEventListener("pointermove", function(e) { if (drag) { move(e); } });
    document.addEventListener("pointerup", function() { if (drag) { drag = false; map.dragging.enable(); } });
    map.on("move zoom resize viewreset", clip);
    clip();
})({{ this._parent.get_name() }}, {{ this.names|tojson }});
{% endmacro %}
"""


def add_swipe(m, names: list[str]):
    """Two panes, ``lsleft`` and ``lsright``, for the layers of the two sides, created before them."""
    from branca.element import MacroElement, Template
    from folium.map import CustomPane

    CustomPane("lsleft", z_index=401, pointer_events=True).add_to(m)
    CustomPane("lsright", z_index=402, pointer_events=True).add_to(m)

    class Swipe(MacroElement):
        _template = Template(_SWIPE)

        def __init__(self):
            super().__init__()
            self._name = "LsSwipe"
            self.names = [str(n) for n in names]

    return Swipe()


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def ramp_ends(lg: Legend) -> tuple[str, str]:
    """The labels of a ramp's ends; ``≤`` / ``≥`` where values beyond them share their colour."""
    return ("≤ " if lg.under else "") + _fmt(lg.vmin), ("≥ " if lg.over else "") + _fmt(lg.vmax)


def cards(groups: list[Group]) -> list[Group]:
    """The legend cards: groups of the same colours (two dates of a product, say) share one, its
    slider setting the opacity of them all."""
    out, seen = [], {}
    for g in groups:
        lg = g.legend
        key = None if lg is None or g.bands else (tuple(lg.classes), tuple(lg.ramp), lg.vmin, lg.vmax, lg.log,
                                                   lg.under, lg.over)
        if key is not None and key in seen:
            card = seen[key]
            card.title = f"{card.title} · {g.title}"
            card.layers = card.layers + g.layers
            continue
        card = Group(g.title, lg, g.opacity, g.colour, list(g.items), list(g.layers), g.bands)
        out.append(card)
        if key is not None:
            seen[key] = card
    return out


def legend_html(groups: list[Group]) -> str:
    """A card per group of layers: its title, its opacity, its colours."""
    parts = []
    for i, g in enumerate(groups):
        lg = g.legend
        name = html.escape(g.title)
        body = [(f"<div class='head'><b>{name}</b><input type='range' min='0' max='100' "
                 f"value='{round(100 * g.opacity)}' title='Opacity' aria-label='Opacity of {name}'></div>")]
        if lg is not None and lg.classes:
            for colour, label in lg.classes[:40]:
                body.append(f"<div class='row'><span class='sw' style='background:{colour}'></span>"
                            f"{html.escape(str(label))}</div>")
            if len(lg.classes) > 40:
                body.append(f"<div class='row'>... {len(lg.classes) - 40} more</div>")
        elif lg is not None and lg.ramp:
            grad = ", ".join(lg.ramp)
            low, high = ramp_ends(lg)
            body.append(f"<div class='ramp' style='background: linear-gradient(to right, {grad})'></div>"
                        f"<div class='ends'><span>{low}</span><span>{'log ' if lg.log else ''}</span>"
                        f"<span>{high}</span></div>")
        elif g.colour:
            body.append(f"<div class='row'><span class='sw' style='background:{g.colour}'></span>features</div>")
        if g.bands and len(g.layers) > 1:  # the band shown, and a button to step through them
            body.append(f"<div class='bands'><button type='button' title='Play'>&#9654;</button>"
                        f"<input type='range' min='0' max='{len(g.layers) - 1}' value='0' step='1' "
                        f"aria-label='Band of {name}'><span>{html.escape(str(g.bands[0]))}</span></div>")
        parts.append(f"<div class='ls-legend' data-g='{i}'>" + "".join(body) + "</div>")
    return "<div class='ls-legends'>" + "".join(parts) + "</div>" if parts else ""


def new_map(basemap: str | None = "satellite"):
    try:
        import folium
        import xyzservices.providers as xyz
        from branca.element import Element
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("interactive maps need folium (pip install folium)") from err
    # no fade-in of the tiles: it runs on animation frames, which a notebook's output frame may not
    # get (VS Code), and the tiles stayed transparent, the map without its base map
    m = folium.Map(tiles=None, prefer_canvas=True, control_scale=True, zoom_snap=0.25, fade_animation=False,
                   width="100%", height="100%")
    m.default_js, m.default_css = [], []  # Leaflet only, in the page (folium's jQuery, Bootstrap... are not used)
    _install(m)
    token = f"LSASSET{id(m)}"
    m.__dict__["_ls_images"][token] = _leaflet()
    m.get_root().header.add_child(Element(token), name="ls_leaflet")
    order = [basemap] + [b for b in BASEMAPS if b != basemap] if basemap in BASEMAPS else []
    for i, key in enumerate(order):
        provider, label = BASEMAPS[key]
        p = xyz.query_name(provider)
        folium.TileLayer(tiles=p.build_url(), attr=p.html_attribution, name=label, max_zoom=p.get("max_zoom", 19),
                         overlay=False, control=True, show=i == 0).add_to(m)
    return m


def add_image(m, img: Image, show: bool = True, pane: str | None = None):
    """An image layer. The map holds a short token where the image goes, replaced by the PNG
    only in the page rendered (render): folium re-parses its scripts as Jinja templates, which
    takes about a second per 20 MB of image."""
    import folium

    if "_ls_images" not in m.__dict__:
        _install(m)
    images = m.__dict__["_ls_images"]
    token = f"data:image/png;base64,LSIMAGE{len(images)}x{id(m)}"
    images[token] = img.data_url()
    layer = folium.raster_layers.ImageOverlay(image=token, bounds=[list(img.bounds_latlon[0]),
                                                                   list(img.bounds_latlon[1])],
                                              name=img.name, opacity=img.opacity, interactive=False,
                                              cross_origin=False, zindex=2, show=show,
                                              **({"pane": pane} if pane else {}))
    layer.add_to(m)
    return layer


def add_vector(m, layer):
    """A GeoJSON or WebGL layer of _vectors, its data put in by render."""
    add_payload(m, layer)
    if type(layer).__name__ == "GlLayer" and "ls_gl" not in m.get_root().header._children:
        from branca.element import Element

        from . import _gl

        token = f"LSASSETGL{id(m)}"
        m.__dict__["_ls_images"][token] = _gl.script()
        m.get_root().header.add_child(Element(token), name="ls_gl")
    layer.add_to(m)
    return layer


def add_payload(m, layer):
    if "_ls_images" not in m.__dict__:
        _install(m)
    images = m.__dict__["_ls_images"]
    token = f"LSPAYLOAD{len(images)}x{id(m)}"
    images[token] = layer._ls_payload
    layer.token = token


def _install(m):
    """folium's own save and display put the images and data in too."""
    from ._core import _maps_iframe

    m.save = lambda path, **_: Path(path).write_text(render(m), encoding="utf-8")
    m._repr_html_ = lambda **_: _maps_iframe(render(m), "560px")
    m.__dict__["_ls_images"] = {}


_TOKENS = re.compile(r"data:image/png;base64,LSIMAGE\d+x\d+|LSPAYLOAD\d+x\d+|LSASSET(?:GL)?\d+")


def render(m) -> str:
    """The page of a map, its images and data in place: in one pass over the page as folium
    renders it (a replace per image went over the page as it grew, a second for 47 images)."""
    page = m.get_root().render()
    payloads = m.__dict__.get("_ls_images", {})
    return _TOKENS.sub(lambda t: payloads.get(t.group(), t.group()), page) if payloads else page


def _readable(groups: list[Group]) -> tuple[list, list]:
    """The image layers whose values the cursor reads, and their palettes (each once)."""
    layers, palettes, seen = [], [], {}
    for g in groups:
        for img, layer in zip(g.items, g.layers):
            if not isinstance(img, Image) or img.lookup is None or img.palette is None:
                continue
            pal = img.palette[:, :3].tolist()
            key = repr(pal)
            if key not in seen:
                seen[key] = len(palettes)
                palettes.append(pal)
            lookup = {k: (v if not isinstance(v, float) else round(v, 12)) for k, v in img.lookup.items()}
            layers.append({"v": layer.get_name(), "ext": [float(x) for x in img.mercator_extent], "p": seen[key],
                           "n": img.name, "u": lookup})
    return layers, palettes


def finish(m, bounds: list, groups: list[Group], title: str | None = None):
    """Layer control, legend cards, the readout of values, title, and the view on everything drawn."""
    import folium
    from branca.element import Element

    control = folium.LayerControl(collapsed=True, position="topright").add_to(m)
    layers, palettes = _readable(groups)
    merged = cards(groups)
    _controls(layers, palettes, [{"n": [x.get_name() for x in g.layers], "b": g.bands, "t": g.title} for g in merged],
              control.get_name()).add_to(m)
    root = m.get_root()
    root.header.add_child(Element(_CSS))
    extra = legend_html(merged)
    if title:
        extra += ("<div style='position:absolute;top:10px;left:50%;transform:translateX(-50%);z-index:1000;"
                  "background:rgba(255,255,255,0.92);padding:4px 12px;border-radius:6px;font:600 14px Inter,"
                  f"Segoe UI,Helvetica,Arial,sans-serif;box-shadow:0 1px 4px rgba(0,0,0,0.3)'>{html.escape(title)}"
                  "</div>")
    if extra:
        m.get_root().html.add_child(Element(extra))
    if bounds:
        s = min(b[0][0] for b in bounds)
        w = min(b[0][1] for b in bounds)
        n = max(b[1][0] for b in bounds)
        e = max(b[1][1] for b in bounds)
        m.fit_bounds([[s, w], [n, e]])
    return m
