# SPDX-License-Identifier: GPL-3.0-or-later
"""``ls.plot``: the plot a piece of data calls for, and the Plot that shows and saves it."""

from __future__ import annotations

import dataclasses
import html as _html
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import _env

#: Rasters and vectors of a result drawn on its map.
MAX_LAYERS = 16
#: Bands of a stack drawn as layers (the others are left out, with a note).
MAX_BANDS = 128

_RGB_NAMES = ({"r", "g", "b"}, {"red", "green", "blue"}, {"b4", "b3", "b2"})


@dataclasses.dataclass
class _Panel:
    kind: str          # "map" (folium), "chart" (plotly), "html", "figure" (matplotlib)
    obj: object
    title: str | None = None


class Plot:
    """A plot made by :func:`plot`: interactive maps and charts (or a static figure).

    It shows itself inline in a notebook (Jupyter, JupyterLab, VS Code, Colab);
    elsewhere :func:`plot` opens it in a window of its own. :meth:`save` writes
    a self-contained HTML page (or, for a static plot, PNG, PDF, SVG).
    """

    def __init__(self, panels: list[_Panel], title: str | None = None, static_recipe=None):
        self.panels = panels
        self.title = title
        self._static_recipe = static_recipe

    # -- the parts --------------------------------------------------------------------------------
    @property
    def maps(self) -> list:
        """The folium maps (to customise them further)."""
        return [p.obj for p in self.panels if p.kind == "map"]

    @property
    def charts(self) -> list:
        """The plotly figures."""
        return [p.obj for p in self.panels if p.kind == "chart"]

    @property
    def figure(self):
        """The matplotlib figure of a static plot."""
        return next((p.obj for p in self.panels if p.kind == "figure"), None)

    # -- HTML -------------------------------------------------------------------------------------
    def _panel_html(self, p: _Panel, height: str, plotlyjs="cdn") -> str:
        if p.kind == "map":
            from ._maps import render

            return _maps_iframe(render(p.obj), height)
        if p.kind == "chart":
            return p.obj.to_html(full_html=False, include_plotlyjs=plotlyjs, config={"responsive": True})
        if p.kind == "figure":
            import base64
            import io

            buf = io.BytesIO()
            p.obj.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            return f"<img style='max-width:100%' src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"
        return str(p.obj)

    def _repr_html_(self) -> str:
        maps = sum(p.kind == "map" for p in self.panels)
        height = "560px" if maps == 1 and len(self.panels) == 1 else "480px"
        parts = [self._panel_html(p, height) for p in self.panels]
        return "<div class='ls-plot'>" + "".join(f"<div style='margin-bottom:12px'>{x}</div>" for x in parts) + "</div>"

    def to_html(self) -> str:
        """A self-contained HTML page of the plot: it opens without a network connection (but
        for the satellite imagery and other base maps)."""
        return self._page(True)

    def _page(self, plotlyjs) -> str:
        """The page, plotly's script (charts) put in once: ``True`` inline, ``"cdn"``, or a URL."""
        alone = len(self.panels) == 1 and self.panels[0].kind == "map"
        if alone:
            from ._maps import render

            return render(self.panels[0].obj)
        # a map fills the window, whatever its size; the other panels follow below
        parts = []
        for p in self.panels:
            parts.append(self._panel_html(p, "max(480px, calc(100vh - 64px))", plotlyjs))
            if p.kind == "chart":
                plotlyjs = False  # once a page
        body = "".join(f"<section>{x}</section>" for x in parts)
        title = _html.escape(self.title or "landschaft")
        return ("<!doctype html><html><head><meta charset='utf-8'><title>" + title + "</title>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<style>body{margin:0;padding:16px;background:#fafaf8;color:#1d1d1b;font:14px/1.4 Inter,"
                "'Segoe UI',Helvetica,Arial,sans-serif} section{background:#fff;border-radius:8px;padding:8px;"
                "margin:0 auto 16px;max-width:1400px;box-shadow:0 1px 3px rgba(0,0,0,.12)} table{border-collapse:"
                "collapse} td,th{padding:3px 10px;border-bottom:1px solid #e4e3df;text-align:left}</style></head>"
                f"<body>{body}</body></html>")

    # -- showing and saving -------------------------------------------------------------------------
    def show(self) -> None:
        """Show the plot: inline in a notebook, else in a window of its own."""
        if _env.in_notebook():
            from IPython.display import HTML, display

            display(HTML(self._repr_html_()))
            return
        if self.figure is not None and len(self.panels) == 1:
            import matplotlib.pyplot as plt

            plt.show()
            return
        _env.open_window(self._page(_env.plotly_url()), self.title or "landschaft")

    def save(self, path, *, dpi: int = 200) -> Path:
        """Write the plot: ``.html`` (interactive, self-contained), or ``.png``, ``.pdf``,
        ``.svg``, ``.jpg`` (a static rendering). Missing folders are created."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in (".html", ".htm"):
            path.write_text(self.to_html(), encoding="utf-8")
            return path
        fig = self.figure
        if fig is not None:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            return path
        if self._static_recipe is None:
            raise ValueError("this plot has no static rendering; save it as .html")
        import matplotlib.pyplot as plt

        fig = self._static_recipe()
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        finally:
            plt.close(fig)  # drawn only to be saved
        return path

    def __repr__(self) -> str:
        kinds = ", ".join(p.kind for p in self.panels)
        return f"Plot({self.title!r}: {kinds})" if self.title else f"Plot({kinds})"


def _maps_iframe(page: str, height: str) -> str:
    """A map's page in a frame as wide as the cell or page, ``height`` high, its corner
    dragged to make it taller or shorter (Leaflet follows the frame's size)."""
    return (f"<div class='ls-map' style='width:100%;height:{height};resize:vertical;overflow:hidden;"
            "border-radius:6px'><iframe style='width:100%;height:100%;border:0;display:block' "
            f"srcdoc=\"{_html.escape(page, quote=True)}\"></iframe></div>")


# --- what an object is ---------------------------------------------------------------------------------


def _is_geo_raster(da) -> bool:
    return isinstance(da, xr.DataArray) and "x" in da.dims and "y" in da.dims and da.rio.crs is not None


def _is_vector(obj) -> bool:
    return type(obj).__name__ in ("GeoDataFrame", "GeoSeries") or hasattr(obj, "geom_type")


def _band_dim(da):
    return next(d for d in da.dims if d not in ("y", "x"))


def _is_rgb(da) -> bool:
    if da.ndim != 3 or da.sizes[_band_dim(da)] != 3:
        return False
    names = {str(v).lower() for v in da[_band_dim(da)].values.tolist()} if _band_dim(da) in da.coords else set()
    return any(names == s for s in _RGB_NAMES)


def _band_index(da, dim, sel) -> list[int]:
    """Positions of the bands chosen: by their labels (the band coordinate: names, or the band
    numbers of a file, from 1), else by position (from 0)."""
    n = da.sizes[dim]
    if sel is None:
        return list(range(n))
    if isinstance(sel, (str, int, np.integer)):
        sel = [sel]
    labels = da[dim].values.tolist() if dim in da.coords else []
    out = []
    for b in sel:
        b = b.item() if isinstance(b, np.generic) else b
        if b in labels:
            out.append(labels.index(b))
        elif isinstance(b, int) and 0 <= b < n:
            out.append(b)
        else:
            raise ValueError(f"no band {b!r}: the bands are {labels or list(range(n))}")
    return out


# --- the layers of a map -----------------------------------------------------------------------------------


def _map_layers(items, opts):
    """(images, vector layers, legends, notes) of spatial items."""
    from . import _raster, _vectors

    images, vectors, legends, notes = [], [], [], []
    base = opts
    for item in items:
        name, obj = item[0], item[1]
        opts = {**base, **item[2]} if len(item) > 2 else base  # a layer's own options
        if _is_vector(obj):
            gdf = obj
            if hasattr(obj, "geom_type") and not hasattr(obj, "geometry"):
                import geopandas as gpd

                gdf = gpd.GeoDataFrame(geometry=[obj], crs=opts.get("crs"))
            elif type(obj).__name__ == "GeoSeries":
                import geopandas as gpd

                gdf = gpd.GeoDataFrame(geometry=obj)
            kind, layer, legend = _vectors.layer(gdf, column=opts.get("column"), cmap=opts.get("cmap"),
                                                 categorical=opts.get("categorical"), name=name,
                                                 vmin=opts.get("vmin"), vmax=opts.get("vmax"), log=opts.get("log"),
                                                 max_size=opts["max_size"],
                                                 color=opts.get("color") or "#ffd60a",
                                                 opacity=opts.get("opacity") or 0.7)
            (images if kind == "image" else vectors).append(layer)
            legends.append(legend)
            continue
        da = obj
        if da.ndim == 3 and _is_rgb(da) and opts.get("rgb", True):
            images.append(_raster.render_rgb(da, max_size=opts["max_size"], title=name))
            continue
        if da.ndim == 3:
            dim = _band_dim(da)
            n = da.sizes[dim]
            idx = _band_index(da, dim, opts.get("bands"))
            if len(idx) > MAX_BANDS:
                notes.append(f"{name}: the first {MAX_BANDS} of {len(idx)} bands are drawn (bands= to choose)")
                idx = idx[:MAX_BANDS]
            # the more bands, the coarser each (the page holds them all)
            size = opts["max_size"] if len(idx) <= 4 else max(512, opts["max_size"] // (2 if len(idx) <= 64 else 4))
            # one scale for every band of a stack, unless given
            vmin, vmax = opts.get("vmin"), opts.get("vmax")
            if vmin is None and vmax is None and np.issubdtype(da.dtype, np.floating):
                sub = da.isel({dim: idx})
                step = max(1, int(np.ceil(max(sub.sizes["y"], sub.sizes["x"]) / 512)))
                s = np.asarray(sub.data[..., ::step, ::step])
                s = s[np.isfinite(s)]
                if s.size:
                    vmin, vmax = (float(v) for v in np.percentile(s, (2, 98)))
            labels = [str(v) for v in da[dim].values.tolist()] if dim in da.coords else [str(i + 1) for i in range(n)]
            for j, k in enumerate(idx):
                img = _raster.render(da.isel({dim: k}), max_size=size, cmap=opts.get("cmap"), vmin=vmin, vmax=vmax,
                                     log=opts.get("log"), opacity=opts.get("opacity") or 0.85,
                                     categorical=opts.get("categorical"), labels=opts.get("labels"),
                                     title=f"{name}: {labels[k]}")
                img.show = j == 0
                images.append(img)
            legends.append(images[-1].legend)
            continue
        img = _raster.render(da, max_size=opts["max_size"], cmap=opts.get("cmap"), vmin=opts.get("vmin"),
                             vmax=opts.get("vmax"), log=opts.get("log"), opacity=opts.get("opacity") or 0.85,
                             categorical=opts.get("categorical"), labels=opts.get("labels"), title=name)
        images.append(img)
        legends.append(img.legend)
    return images, vectors, legends, notes


def _map(items, opts, title):
    from . import _maps

    images, vectors, legends, notes = _map_layers(items, opts)
    m = _maps.new_map(opts.get("basemap", "satellite"))
    bounds = []
    for img in images:
        _maps.add_image(m, img, show=getattr(img, "show", True))
        bounds.append(img.bounds_latlon)
    for layer in vectors:
        _maps.add_vector(m, layer)
        b = getattr(layer, "_ls_bounds", None) or layer.get_bounds()
        if b and b[0][0] is not None:
            bounds.append(((b[0][0], b[0][1]), (b[1][0], b[1][1])))
    # a stack shows one legend (its first layer's); every other layer its own
    _maps.finish(m, bounds, [lg for lg in legends if lg is not None], title)
    panels = [_Panel("map", m, title)]
    if notes:
        panels.append(_Panel("html", "<p style='color:#6b6a66'>" + "<br>".join(_html.escape(n) for n in notes)
                             + "</p>"))
    return panels


# --- everything else ------------------------------------------------------------------------------------


def _plain_array_chart(a: np.ndarray, title):
    from . import _charts

    a = np.asarray(a)
    if a.ndim == 1:
        return _charts.lines(np.arange(a.size), {title or "value": a}, title)
    if a.ndim == 2:
        import plotly.express as px

        fig = px.imshow(a, color_continuous_scale="viridis", origin="upper", title=title, aspect="equal")
        fig.update_layout(template="plotly_white", height=560)
        return fig
    raise ValueError(f"cannot plot an array of shape {a.shape}")


def _summary_html(values: dict, title: str | None) -> str:
    rows = "".join(f"<tr><th>{_html.escape(str(k))}</th><td>{_html.escape(_short(v))}</td></tr>" for k, v in
                   values.items())
    head = f"<b>{_html.escape(title)}</b>" if title else ""
    return f"<div style='font:13px Inter,Segoe UI,Helvetica,Arial,sans-serif'>{head}<table>{rows}</table></div>"


def _short(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)) and len(v) > 8:
        return f"[{', '.join(map(str, v[:8]))}, ... ({len(v)})]"
    return str(v)


def _result_panels(obj, opts, title):
    """A result of a tool (or a dict): its maps on one map, its tables as charts, its numbers in a table."""
    from .._save import _is_scalar, _items, _reference

    hook = getattr(obj, "_plot_panels", None)
    if callable(hook):
        return hook(opts, title)
    from . import _registry

    custom = _registry.panels(obj, opts, title)
    if custom is not None:
        return custom
    items = _items(obj)
    like, _ref_shape = _reference(items)
    spatial, charts, scalars = [], [], {}

    def visit(name, v, depth=0):
        if v is None:
            return
        if _is_scalar(v):
            scalars[name] = v.item() if isinstance(v, np.generic) else v
        elif (_is_vector(v) and type(v).__name__ != "Tree") or (_is_geo_raster(v) and v.ndim in (2, 3)):
            spatial.append((name, v))
        elif isinstance(v, np.ndarray) and v.ndim == 2 and like is not None and v.shape == tuple(like.shape[-2:]) \
                and v.size >= 1000:
            spatial.append((name, like.copy(data=v).rename(name)))
        elif isinstance(v, (pd.DataFrame, pd.Series)) and len(v):
            charts.append((name, v))
        elif isinstance(v, dict) and depth < 2:
            if all(_is_scalar(x) for x in v.values()):
                scalars[name] = {str(k): x for k, x in v.items()}
            else:
                for k, x in v.items():
                    visit(f"{name} {k}", x, depth + 1)
        elif isinstance(v, (list, tuple)) and all(_is_scalar(x) for x in v):
            scalars[name] = list(v)

    for k, v in items.items():
        visit(k, v)
    panels = []
    if spatial:
        panels += _map(spatial[:MAX_LAYERS], {**opts, "rgb": False}, title)
    from . import _charts

    for name, table in charts[:6]:
        try:
            panels.append(_Panel("chart", _charts.table_chart(table, name), name))
        except ValueError:
            continue
    if scalars:
        panels.append(_Panel("html", _summary_html(scalars, title if not panels else None)))
    if not panels:
        raise TypeError(f"nothing to plot in a {type(obj).__name__}")
    return panels


def _panels(obj, opts, title):
    from . import _charts

    if isinstance(obj, (list, tuple)) and obj and all(_is_geo_raster(o) or _is_vector(o) for o in obj):
        items = [(_name(o, i), o) for i, o in enumerate(obj)]
        return _map(items, opts, title)
    if _is_geo_raster(obj):
        if obj.ndim not in (2, 3):
            raise ValueError(f"a raster has 2 or 3 dimensions, got {obj.dims}")
        return _map([(_name(obj, 0), obj)], opts, title)
    if _is_vector(obj):
        return _map([(_name(obj, 0), obj)], opts, title)
    if isinstance(obj, xr.DataArray):  # not georeferenced
        return [_Panel("chart", _plain_array_chart(obj.values, title or obj.name), title)]
    if isinstance(obj, np.ndarray):
        return [_Panel("chart", _plain_array_chart(obj, title), title)]
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return [_Panel("chart", _charts.table_chart(obj, title), title)]
    return _result_panels(obj, opts, title)


def _name(obj, i) -> str:
    n = getattr(obj, "name", None)
    if isinstance(n, str) and n:
        return n
    return "layer" if i == 0 else f"layer {i + 1}"


def plot(obj, *, column=None, cmap=None, labels=None, categorical=None, vmin=None, vmax=None, log=None,
         opacity=None, bands=None, title=None, basemap="satellite", static=False, max_size: int = 2048,
         show=None, **kwargs) -> Plot:
    """Plot anything: a raster, a vector layer, a table or the result of a tool, as it calls for.

    The form follows the data. A raster (any, or a landschaft product) goes on an
    interactive map over satellite imagery: a class map (integers, a product such
    as MSPA or FAD) in its classes' colours with their names, continuous values
    on a colour ramp that suits them (elevation, NDVI, priority ranks, currents on
    a log scale, values around zero on a diverging ramp), a stack band by band
    (or as true colour), a vector layer as polygons, lines or points, coloured by
    a column. A table becomes the chart it calls for; the result of a tool, its
    maps, charts and numbers together. Several rasters and vectors in a list share
    one map.

    In a notebook the plot shows inline; elsewhere it opens in a window of its own
    (pywebview, else the web browser). Rasters are reduced to the screen's
    resolution before anything else (a lazy raster is read reduced), so a map of a
    whole country draws in about the time of a small one.

    Parameters
    ----------
    obj
        A DataArray (2-D, or a ``(band, y, x)`` stack), GeoDataFrame, GeoSeries,
        shapely geometry, DataFrame, Series, numpy array, a result of a
        landschaft tool, a dict of these, or a list of rasters and vectors.
    column
        Vectors: the column that colours the features.
    cmap
        The colours: a palette name (``"viridis"``, ``"magma"``, ``"terrain"``,
        ``"rdylgn"``, ``"rdbu"``, ``"blues"``, ``"greens"``, ``"ylorrd"``,
        ``"rank"``, ``"current"``, or any matplotlib colormap; ``"_r"`` reverses
        it), a list of colours, or a dict ``{value: colour}`` for classes.
    labels
        ``{value: name}``: the names of the classes in the legend.
    categorical
        ``True`` draws the values as classes, ``False`` on a ramp (default:
        detected: products, booleans and integers with at most 64 values are
        classes).
    vmin, vmax, log
        The ramp's range (default: the 2nd and 98th percentiles) and a log scale.
    opacity
        Opacity of the layers.
    bands
        A stack: the bands to draw, by their labels (names, or the band numbers
        of a file read by :func:`load_raster`, from 1), else by position (from 0).
    title
        Title of the map or chart.
    basemap
        ``"satellite"`` (default, Esri World Imagery), ``"osm"``, ``"topo"``,
        ``"light"``, ``"relief"`` or ``None``; the others stay selectable.
    static
        A static matplotlib figure instead (for publications and offline use; the
        satellite imagery needs contextily).
    max_size
        Pixels on the long side of every raster drawn (default 2048).
    show
        Show the plot now (default: outside notebooks, unless the environment
        variable ``LANDSCHAFT_PLOT_SHOW`` is ``0``); in a notebook, the returned
        Plot shows itself.

    Returns
    -------
    Plot
        Shows itself in a notebook; ``.save("map.html")`` writes it.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> p = ls.plot(ls.mspa(forest))                      # GuidosToolbox's colours and class names
    >>> p
    Plot(map)
    >>> dem = ls.load_raster(ls.example_data("orelevation"))
    >>> p = ls.plot(dem, cmap="terrain", title="Elevation")
    >>> p.save("elevation.html").name
    'elevation.html'
    """
    opts = dict(column=column, cmap=cmap, labels=labels, categorical=categorical, vmin=vmin, vmax=vmax, log=log,
                opacity=opacity, bands=bands, basemap=basemap, max_size=int(max_size), **kwargs)

    def static_figure():
        from . import _static

        return _static.figure(obj, opts, title)

    if static:
        p = Plot([_Panel("figure", static_figure(), title)], title)
    else:  # saved as an image, the same data drawn by matplotlib
        p = Plot(_panels(obj, opts, title), title, static_recipe=static_figure)
    if show if show is not None else _env.auto_show():
        p.show()
    return p
