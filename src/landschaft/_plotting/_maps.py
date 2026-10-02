# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactive maps (folium / Leaflet): base maps, image and vector layers, legends."""

from __future__ import annotations

import base64
import functools
import html
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
.ls-full a { font: bold 18px/30px sans-serif; text-align: center; cursor: pointer; }
.ls-coords { background: rgba(255,255,255,0.8); padding: 1px 6px; margin: 0 0 4px 10px !important; border-radius: 4px;
  font: 11px/1.5 Inter, "Segoe UI", Helvetica, Arial, sans-serif; color: #1d1d1b; font-variant-numeric: tabular-nums; }
.ls-coords:empty { display: none; }
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


_CONTROLS = None


def _controls():
    """Full screen (top left) and the coordinates under the cursor (bottom left)."""
    global _CONTROLS
    if _CONTROLS is None:
        from branca.element import MacroElement, Template

        class Controls(MacroElement):
            _template = Template("""
{% macro script(this, kwargs) %}
(function(map) {
    var Full = L.Control.extend({options: {position: "topleft"}, onAdd: function() {
        var bar = L.DomUtil.create("div", "leaflet-bar leaflet-control ls-full");
        var a = L.DomUtil.create("a", "", bar);
        a.innerHTML = "&#x26F6;"; a.title = "Full screen"; a.href = "#"; a.setAttribute("role", "button");
        L.DomEvent.disableClickPropagation(bar);
        L.DomEvent.on(a, "click", function(e) {
            L.DomEvent.preventDefault(e);
            var c = map.getContainer();
            if (document.fullscreenElement) { document.exitFullscreen(); }
            else if (c.requestFullscreen) { c.requestFullscreen(); }
        });
        return bar;
    }});
    new Full().addTo(map);
    document.addEventListener("fullscreenchange", function() { map.invalidateSize(); });
    var Coords = L.Control.extend({options: {position: "bottomleft"}, onAdd: function() {
        var div = L.DomUtil.create("div", "leaflet-control ls-coords");
        map.on("mousemove", function(e) {
            div.textContent = e.latlng.lat.toFixed(5) + ", " + e.latlng.lng.toFixed(5);
        });
        map.on("mouseout", function() { div.textContent = ""; });
        return div;
    }});
    new Coords().addTo(map);
})({{ this._parent.get_name() }});
{% endmacro %}
""")

            def __init__(self):
                super().__init__()
                self._name = "LsControls"

        _CONTROLS = Controls
    return _CONTROLS


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def ramp_ends(lg: Legend) -> tuple[str, str]:
    """The labels of a ramp's ends; ``≤`` / ``≥`` where values beyond them share their colour."""
    return ("≤ " if lg.under else "") + _fmt(lg.vmin), ("≥ " if lg.over else "") + _fmt(lg.vmax)


def legend_html(legends: list[Legend]) -> str:
    parts = []
    for lg in legends:
        if lg is None:
            continue
        body = [f"<b>{html.escape(lg.title)}</b>"]
        if lg.classes:
            for colour, label in lg.classes[:40]:
                body.append(f"<div class='row'><span class='sw' style='background:{colour}'></span>"
                            f"{html.escape(str(label))}</div>")
            if len(lg.classes) > 40:
                body.append(f"<div class='row'>... {len(lg.classes) - 40} more</div>")
        elif lg.ramp:
            grad = ", ".join(lg.ramp)
            low, high = ramp_ends(lg)
            body.append(f"<div class='ramp' style='background: linear-gradient(to right, {grad})'></div>"
                        f"<div class='ends'><span>{low}</span><span>{'log ' if lg.log else ''}</span>"
                        f"<span>{high}</span></div>")
        parts.append("<div class='ls-legend'>" + "".join(body) + "</div>")
    return "<div class='ls-legends'>" + "".join(parts) + "</div>" if parts else ""


def new_map(basemap: str | None = "satellite"):
    try:
        import folium
        import xyzservices.providers as xyz
        from branca.element import Element
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("interactive maps need folium (pip install folium)") from err
    m = folium.Map(tiles=None, prefer_canvas=True, control_scale=True, zoom_snap=0.25, width="100%", height="100%")
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
    _controls()().add_to(m)
    return m


def add_image(m, img: Image, show: bool = True):
    """An image layer. The map holds a short token where the image goes, replaced by the PNG
    only in the page rendered (render): folium re-parses its scripts as Jinja templates, which
    takes about a second per 20 MB of image."""
    import folium

    if "_ls_images" not in m.__dict__:
        _install(m)
    images = m.__dict__["_ls_images"]
    token = f"data:image/png;base64,LSIMAGE{len(images)}x{id(m)}"
    images[token] = img.data_url()
    folium.raster_layers.ImageOverlay(image=token, bounds=[list(img.bounds_latlon[0]), list(img.bounds_latlon[1])],
                                      name=img.name, opacity=1.0, interactive=False, cross_origin=False,
                                      zindex=2, show=show).add_to(m)


def add_vector(m, layer):
    """A GeoJSON layer of _vectors, its data put in by render."""
    add_payload(m, layer)
    layer.add_to(m)


def add_payload(m, layer):
    if "_ls_images" not in m.__dict__:
        _install(m)
    images = m.__dict__["_ls_images"]
    token = f"LSPAYLOAD{len(images)}x{id(m)}"
    images[token] = layer._ls_payload
    layer.token = token


def _install(m):
    """folium's own save and display put the images and data in too."""
    from pathlib import Path

    from ._core import _maps_iframe

    m.save = lambda path, **_: Path(path).write_text(render(m), encoding="utf-8")
    m._repr_html_ = lambda **_: _maps_iframe(render(m), "560px")
    m.__dict__["_ls_images"] = {}


def render(m) -> str:
    """The page of a map, its images in place."""
    page = m.get_root().render()
    for token, url in m.__dict__.get("_ls_images", {}).items():
        page = page.replace(token, url)
    return page


def finish(m, bounds: list, legends: list[Legend], title: str | None = None):
    """Layer control, legends, title, and the view on everything drawn."""
    import folium
    from branca.element import Element

    folium.LayerControl(collapsed=True, position="topright").add_to(m)
    root = m.get_root()
    root.header.add_child(Element(_CSS))
    html_legends = legend_html(legends)
    extra = html_legends
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
