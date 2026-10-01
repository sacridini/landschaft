# SPDX-License-Identifier: GPL-3.0-or-later
"""Interactive maps (folium / Leaflet): base maps, image and vector layers, legends."""

from __future__ import annotations

import html

from ._raster import Image, Legend

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
.ls-legends { position: absolute; bottom: 24px; right: 10px; z-index: 1000; display: flex; flex-direction: column;
  gap: 6px; max-height: 70%; overflow-y: auto; font: 12px/1.35 Inter, "Segoe UI", Helvetica, Arial, sans-serif; }
.ls-legend { background: rgba(255,255,255,0.92); color: #1d1d1b; border-radius: 6px; padding: 6px 9px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.3); min-width: 140px; }
.ls-legend b { display: block; margin-bottom: 4px; font-weight: 600; }
.ls-legend .row { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.ls-legend .sw { width: 14px; height: 12px; border: 1px solid rgba(0,0,0,0.25); flex: none; }
.ls-legend .ramp { height: 12px; border: 1px solid rgba(0,0,0,0.25); }
.ls-legend .ends { display: flex; justify-content: space-between; color: #4a4a48; }
</style>
"""


def _fmt(v: float) -> str:
    return f"{v:.4g}"


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
            low, high = _fmt(lg.vmin), _fmt(lg.vmax)
            body.append(f"<div class='ramp' style='background: linear-gradient(to right, {grad})'></div>"
                        f"<div class='ends'><span>{low}</span><span>{'log ' if lg.log else ''}</span>"
                        f"<span>{high}</span></div>")
        parts.append("<div class='ls-legend'>" + "".join(body) + "</div>")
    return "<div class='ls-legends'>" + "".join(parts) + "</div>" if parts else ""


def new_map(basemap: str | None = "satellite"):
    try:
        import folium
        import xyzservices.providers as xyz
        from folium import plugins
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("interactive maps need folium (pip install landschaft[plot])") from err
    m = folium.Map(tiles=None, prefer_canvas=True, control_scale=True, zoom_snap=0.25, width="100%", height="100%")
    order = [basemap] + [b for b in BASEMAPS if b != basemap] if basemap in BASEMAPS else []
    for i, key in enumerate(order):
        provider, label = BASEMAPS[key]
        p = xyz.query_name(provider)
        folium.TileLayer(tiles=p.build_url(), attr=p.html_attribution, name=label, max_zoom=p.get("max_zoom", 19),
                         overlay=False, control=True, show=i == 0).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.MousePosition(position="bottomleft", separator=" | ", num_digits=5, prefix="lat | lon:").add_to(m)
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
    import html as _h
    from pathlib import Path

    m.save = lambda path, **_: Path(path).write_text(render(m), encoding="utf-8")
    m._repr_html_ = lambda **_: ("<iframe style='width:100%;height:560px;border:0' "
                                 f"srcdoc=\"{_h.escape(render(m), quote=True)}\"></iframe>")
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
