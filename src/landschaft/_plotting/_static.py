# SPDX-License-Identifier: GPL-3.0-or-later
"""Static figures (matplotlib) with the same colours and legends, satellite imagery through contextily."""

from __future__ import annotations

import dataclasses
import itertools
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from . import _palettes
from ._raster import Legend


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("static plots need matplotlib (pip install matplotlib)") from err
    return plt


def _contextily():
    try:
        import contextily
    except ImportError:
        return None
    return contextily


def _legend(ax, legend: Legend | None) -> None:
    """The legend of a layer beside its axes (or beside several: a list of axes sharing it)."""
    if legend is None:
        return
    plt = _plt()
    axes = ax if isinstance(ax, list) else [ax]
    if legend.classes:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=c, edgecolor="#444", label=lab) for c, lab in legend.classes[:30]]
        axes[-1].legend(handles=handles, title=legend.title, loc="upper left", bbox_to_anchor=(1.01, 1),
                        frameon=False, fontsize=8, title_fontsize=9)
    elif legend.ramp:
        from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

        cmap = LinearSegmentedColormap.from_list("ls", legend.ramp)
        norm = LogNorm(legend.vmin, legend.vmax) if legend.log and legend.vmin > 0 else Normalize(legend.vmin,
                                                                                                 legend.vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        extend = {(True, True): "both", (True, False): "min", (False, True): "max"}.get((legend.under, legend.over),
                                                                                       "neither")
        cb = axes[0].figure.colorbar(sm, ax=ax, shrink=0.75, pad=0.02, extend=extend)
        cb.set_label(legend.title)


def _draw_spatial(ax, items, opts, legends: bool = True):
    """Rasters and vectors on one axes; in Web Mercator over imagery when contextily is there.
    The legends (one per layer, or none: ``legends=False``) and the kind of coordinates drawn."""
    from . import _raster, _vectors
    from ._core import _band_dim, _is_rgb, _is_vector

    ctx = _contextily() if opts.get("basemap") else None
    merc = ctx is not None
    crs = None
    out = []
    for name, obj in items:
        if _is_vector(obj):
            import geopandas as gpd

            gdf = obj if hasattr(obj, "geometry") else gpd.GeoDataFrame(geometry=[obj], crs=opts.get("crs"))
            if type(gdf).__name__ == "GeoSeries":
                gdf = gpd.GeoDataFrame(geometry=gdf)
            if merc and gdf.crs is not None:
                gdf = gdf.to_crs(3857)
            crs = crs or (3857 if merc else gdf.crs)
            # the colours and legend of the interactive map
            codes, palette, legend = _vectors._colours(gdf, opts.get("column"), opts.get("cmap"),
                                                       opts.get("categorical"), opts.get("vmin"), opts.get("vmax"),
                                                       opts.get("log"))
            if codes is None:
                legend = None
            _draw_vector(ax, gdf, codes, palette, opts.get("color") or "#ffd60a", opts.get("opacity") or 0.7,
                         opts["max_size"])
            out.append(legend)
            continue
        da = obj
        if da.ndim == 3 and _is_rgb(da):
            img = _raster.render_rgb(da, max_size=opts["max_size"], title=name)
            ext = img.mercator_extent
            ax.imshow(img.rgba, extent=(ext[0], ext[2], ext[1], ext[3]), interpolation="nearest", alpha=img.opacity,
                      zorder=2)
            crs = crs or 3857
            continue
        if da.ndim == 3:
            da = da.isel({_band_dim(da): 0})
        img = _raster.render(da, max_size=opts["max_size"], cmap=opts.get("cmap"), vmin=opts.get("vmin"),
                             vmax=opts.get("vmax"), log=opts.get("log"), opacity=opts.get("opacity") or 0.9,
                             categorical=opts.get("categorical"), labels=opts.get("labels"), title=name,
                             mercator=merc)
        ext = img.mercator_extent if merc else img.extent
        ax.imshow(img.rgba, extent=(ext[0], ext[2], ext[1], ext[3]), interpolation="nearest", zorder=2,
                  alpha=img.opacity)
        crs = crs or (3857 if merc else da.rio.crs)
        out.append(img.legend)
    if merc:
        try:
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Esri.WorldImagery, attribution_size=5,
                            zorder=1)
        except Exception as err:  # noqa: BLE001 - offline: no imagery
            warnings.warn(f"no satellite imagery ({err})", stacklevel=4)
    ax.set_aspect("equal")
    if legends:
        for lg in out:
            _legend(ax, lg)
    return crs, [lg for lg in out if lg is not None]


#: Above this many features (or vertices), a static map burns a vector layer into an image.
MAX_FEATURES = 100_000
MAX_VERTICES = 2_000_000


def _draw_vector(ax, gdf, codes, palette, colour, opacity, max_size) -> None:
    """A vector layer drawn as one collection per kind of geometry, made from the coordinates of all
    features at once (geopandas makes a patch per feature, 8 times slower), simplified to half a
    pixel of ``max_size``; very many features burnt into an image as on interactive maps."""
    import shapely
    from matplotlib.colors import to_rgba_array

    from . import _vectors

    geoms = np.asarray(gdf.geometry.values, dtype=object)
    keep = ~shapely.is_missing(geoms) & ~shapely.is_empty(geoms)
    geoms = geoms[keep]
    codes = None if codes is None else np.asarray(codes)[keep]
    if not len(geoms):
        return
    colours = [colour] if codes is None else palette + [_vectors._MISSING]
    rgba = to_rgba_array(colours)
    which = np.zeros(len(geoms), np.int64) if codes is None else np.where(codes < 0, len(palette), codes)
    if len(geoms) > MAX_FEATURES or int(shapely.get_num_coordinates(geoms).sum()) > MAX_VERTICES:
        burnt, cols, (x0, y0, x1, y1) = (
            _vectors.burn_points(*shapely.get_coordinates(geoms).T, codes, palette, colour, max_size)
            if set(np.unique(shapely.get_type_id(geoms)).tolist()) == {0} else
            _vectors.burn(_frame(geoms, gdf.crs), codes, palette, colour, max_size))
        lut = np.zeros((len(cols) + 1, 4), np.float32)
        lut[1:] = to_rgba_array(cols)
        lut[1:, 3] = opacity
        ax.imshow(lut[burnt], extent=(x0, x1, y0, y1), interpolation="nearest", zorder=3)
        return
    x0, y0, x1, y1 = shapely.total_bounds(geoms)
    tol = max(x1 - x0, y1 - y0) / (max_size * 2.0)
    parts, owner = shapely.get_parts(geoms, return_index=True)
    kind = shapely.get_type_id(parts)
    polygons, lines, points = kind == 3, np.isin(kind, (1, 2)), kind == 0
    if polygons.any():
        polys = parts[polygons]
        if tol > 0:
            polys = _vectors.simplify(polys, tol)
        paths, feature = _polygon_paths(polys, owner[polygons])
        if paths:
            from matplotlib.collections import PathCollection

            ax.add_collection(PathCollection(paths, facecolors=rgba[which[feature]], edgecolors="#222222",
                                             linewidths=0.5, alpha=opacity, zorder=3))
    if lines.any():
        from matplotlib.collections import LineCollection

        ls_ = parts[lines]
        if tol > 0:
            ls_ = _vectors.simplify(ls_, tol)
        xy, line = shapely.get_coordinates(ls_, return_index=True)
        cut = np.flatnonzero(np.diff(line)) + 1
        segs = np.split(xy, cut)
        feature = owner[lines][line[np.r_[0, cut]]] if len(xy) else []
        if segs and len(xy):
            ax.add_collection(LineCollection(segs, colors=rgba[which[feature]], linewidths=1.0, alpha=opacity,
                                             zorder=3))
    if points.any():
        xy = shapely.get_coordinates(parts[points])
        ax.scatter(xy[:, 0], xy[:, 1], s=14, c=rgba[which[owner[points]]], edgecolors="#222222", linewidths=0.5,
                   alpha=opacity, zorder=3)
    ax.update_datalim([(x0, y0), (x1, y1)])
    ax.autoscale_view()


def _frame(geoms, crs):
    import geopandas as gpd

    return gpd.GeoDataFrame(geometry=gpd.GeoSeries(geoms, crs=crs))


def _polygon_paths(polys, owner):
    """A compound path per feature (its polygons and their holes) and the feature of each path. Matplotlib
    fills by the non-zero rule: holes are turned against their exterior."""
    import shapely
    from matplotlib.path import Path as MplPath

    rings, poly_of_ring = shapely.get_rings(polys, return_index=True)
    xy, ring_of = shapely.get_coordinates(rings, return_index=True)
    if not len(xy):
        return [], np.array([], np.int64)
    n_rings = len(rings)
    start = np.searchsorted(ring_of, np.arange(n_rings))
    stop = np.r_[start[1:], len(xy)]
    exterior = np.r_[True, poly_of_ring[1:] != poly_of_ring[:-1]]
    flip = shapely.is_ccw(rings) != exterior  # exteriors anticlockwise, holes clockwise
    if flip.any():
        r = ring_of
        i = np.arange(len(xy))
        xy = xy[np.where(flip[r], start[r] + stop[r] - 1 - i, i)]
    codes = np.full(len(xy), MplPath.LINETO, np.uint8)
    codes[start] = MplPath.MOVETO
    codes[stop - 1] = MplPath.CLOSEPOLY
    feature_of_vertex = owner[poly_of_ring[ring_of]]
    cut = np.r_[0, np.flatnonzero(np.diff(feature_of_vertex)) + 1, len(xy)]
    paths = [MplPath(xy[a:b], codes[a:b]) for a, b in itertools.pairwise(cut)]
    return paths, feature_of_vertex[cut[:-1]]


# --- the furniture of a map: coordinates, scale bar, north arrow -------------------------------------------

_R = 6378137.0  # Web Mercator's sphere


def _crs(crs):
    if crs is None:
        return None
    from pyproj import CRS

    try:
        return CRS.from_user_input(crs)
    except Exception:  # noqa: BLE001 - an unknown CRS: no furniture
        return None


def _coordinates(ax, crs) -> None:
    """Longitudes and latitudes on the axes of a Web Mercator or geographic map; nothing else."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    c = _crs(crs)
    if c is None or not (c.is_geographic or c.to_epsg() == 3857):
        ax.set_xticks([])
        ax.set_yticks([])
        return
    if c.is_geographic:
        lon, lat = (lambda x: x), (lambda y: y)
    else:
        lon = lambda x: np.degrees(x / _R)
        lat = lambda y: np.degrees(np.arctan(np.sinh(y / _R)))

    def decimals(lo, hi, f):  # enough to tell four ticks apart
        span = abs(float(f(hi)) - float(f(lo)))
        return int(np.clip(np.ceil(-np.log10(span / 4)) if span > 0 else 2, 0, 5))

    nx, ny = decimals(*ax.get_xlim(), lon), decimals(*ax.get_ylim(), lat)

    def deg(v, hemis, n):
        return f"{abs(v):.{n}f}°{hemis[0] if v >= 0 else hemis[1]}"

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: deg(float(lon(x)), "EW", nx)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: deg(float(lat(y)), "NS", ny)))
    ax.tick_params(labelsize=7, length=3, color="#555", labelcolor="#333")
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_verticalalignment("center")


def _metres_per_unit(ax, crs) -> float | None:
    """Ground metres per unit of the axes at their centre (Web Mercator and degrees shrink with latitude)."""
    c = _crs(crs)
    if c is None:
        return None
    y0, y1 = ax.get_ylim()
    yc = (y0 + y1) / 2
    if c.is_geographic:
        return 111_320.0 * np.cos(np.radians(yc))
    if c.to_epsg() == 3857:
        return float(np.cos(np.arctan(np.sinh(yc / _R))))
    try:
        return float(c.axis_info[0].unit_conversion_factor)
    except (IndexError, AttributeError):
        return None


def _scalebar(ax, crs) -> None:
    """A bar of a round length, about a fifth of the map's width, at its bottom right (the base map's
    credits are at the left)."""
    from matplotlib.offsetbox import (
        AnchoredOffsetbox,
        AuxTransformBox,
        HPacker,
        TextArea,
        VPacker,
    )
    from matplotlib.patches import Rectangle

    m = _metres_per_unit(ax, crs)
    if not m:
        return
    x0, x1 = ax.get_xlim()
    target = abs(x1 - x0) * m / 5
    if target <= 0 or not np.isfinite(target):
        return
    k = 10 ** np.floor(np.log10(target))
    metres = max(f * k for f in (1, 2, 5) if f * k <= target)
    label = f"{metres / 1000:g} km" if metres >= 1000 else f"{metres:g} m"
    length = metres / m
    height = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
    bar = AuxTransformBox(ax.transData)
    bar.add_artist(Rectangle((0, 0), length / 2, height, facecolor="#1d1d1b", edgecolor="#1d1d1b", linewidth=0.6))
    bar.add_artist(Rectangle((length / 2, 0), length / 2, height, facecolor="white", edgecolor="#1d1d1b",
                             linewidth=0.6))
    box = VPacker(children=[bar, HPacker(children=[TextArea(label, textprops={"fontsize": 7})], align="center")],
                  align="center", pad=0, sep=2)
    anchored = AnchoredOffsetbox(loc="lower right", child=box, pad=0.4, borderpad=0.6, frameon=True)
    anchored.patch.set(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2")
    anchored.set_zorder(5)
    ax.add_artist(anchored)


def _north_arrow(ax) -> None:
    import matplotlib.patheffects as pe

    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    ax.annotate("", xy=(0.95, 0.91), xytext=(0.95, 0.80), xycoords="axes fraction", zorder=6,
                arrowprops={"arrowstyle": "-|>,head_width=0.35,head_length=0.7", "color": "#1d1d1b", "linewidth": 1.4,
                            "shrinkA": 0, "shrinkB": 0, "path_effects": halo})
    ax.text(0.95, 0.955, "N", transform=ax.transAxes, ha="center", va="center", fontsize=9, fontweight="bold",
            color="#1d1d1b", path_effects=halo, zorder=6)


def _furnish(ax, crs, opts, first: bool, alone: bool) -> None:
    if alone:
        _coordinates(ax, crs)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if first and opts.get("scalebar", True):
        _scalebar(ax, crs)
    if first and opts.get("north_arrow", True):
        _north_arrow(ax)


def _draw_table(ax, table, title) -> None:
    if isinstance(table, pd.Series):
        if _palettes.is_number(table.index.dtype) and table.index.is_monotonic_increasing and len(table) > 12:
            table.plot(ax=ax)
        else:
            table.plot.bar(ax=ax)
    elif {"metric", "value"} <= set(table.columns):
        piv = table.pivot_table(index="class" if "class" in table else "metric", columns="metric", values="value")
        piv.plot.bar(ax=ax, subplots=False)
    else:
        num = table.select_dtypes("number")
        if len(num.columns) and num.iloc[:, 0].is_monotonic_increasing and len(num.columns) > 1:
            num.plot(x=num.columns[0], ax=ax, legend=len(num.columns) <= 13)
        else:
            num.plot(ax=ax, legend=len(num.columns) <= 13)
    ax.set_title(title or "")


#: Maps of a figure (a stack's bands, a result's rasters) and tables drawn beside them.
MAX_PANELS = 12
MAX_TABLES = 3


def _panels(obj, opts, title):
    """[(title, [(name, layer)...]), ...] of the maps of a figure, whether the maps share one scale
    (a stack's bands: one legend), and the tables."""
    from ._core import _band_dim, _band_index, _is_geo_raster, _is_rgb, _is_vector

    def name(o, i):
        n = getattr(o, "name", None)
        return n if isinstance(n, str) and n else ("value" if i == 0 else f"layer {i + 1}")

    if isinstance(obj, (list, tuple)) and obj and all(_is_geo_raster(o) or _is_vector(o) for o in obj):
        items = [(name(o, i), o) for i, o in enumerate(obj)]
        if opts.get("compare"):  # side by side
            same = items[0][0] == items[-1][0]  # the panels told apart, the legend keeps the name
            return [(f"{n} ({i + 1})" if same else n, [(n, o)]) for i, (n, o) in enumerate(items)], False, []
        return [(title or items[0][0], items)], False, []
    if _is_geo_raster(obj) and obj.ndim == 3 and not (_is_rgb(obj) and opts.get("rgb", True)):
        dim = _band_dim(obj)
        idx = _band_index(obj, dim, opts.get("bands"))[:MAX_PANELS]
        labels = obj[dim].values.tolist() if dim in obj.coords else list(range(1, obj.sizes[dim] + 1))
        return [(str(labels[k]), [(str(labels[k]), obj.isel({dim: k}))]) for k in idx], True, []
    if _is_geo_raster(obj) or _is_vector(obj):
        return [(title or name(obj, 0), [(name(obj, 0), obj)])], False, []
    from .._save import _items

    maps, tables = [], []
    for k, v in _items(obj).items():
        if (_is_geo_raster(v) and v.ndim in (2, 3)) or (_is_vector(v) and type(v).__name__ != "Tree"):
            maps.append((k, [(k, v)]))
        elif isinstance(v, (pd.DataFrame, pd.Series)) and len(v) and _drawable(v):
            tables.append((k, v))
    return maps[:MAX_PANELS], False, tables[:MAX_TABLES]


def _drawable(table) -> bool:
    """A table with numbers to draw."""
    if isinstance(table, pd.Series):
        return _palettes.is_number(table.dtype)
    return {"metric", "value"} <= set(table.columns) or not table.select_dtypes("number").empty


def _shared_scale(panels, opts) -> dict:
    """One ramp for the bands of a stack (the 2nd and 98th percentiles of them all), unless given."""
    if opts.get("vmin") is not None or opts.get("vmax") is not None:
        return opts
    arrays = [layer for _, items in panels for _, layer in items]
    if not arrays or not np.issubdtype(arrays[0].dtype, np.floating):
        return opts
    s = np.concatenate([np.asarray(a.data[::max(1, a.shape[0] // 256), ::max(1, a.shape[1] // 256)]).ravel()
                        for a in arrays])
    s = s[np.isfinite(s)]
    if not s.size:
        return opts
    lo, hi = np.percentile(s, (2, 98))
    return {**opts, "vmin": float(lo), "vmax": float(hi)}


def figure(obj, opts: dict, title: str | None):
    """A matplotlib figure of anything ls.plot draws: maps (small multiples for a stack's bands,
    a result's rasters or two dates compared) with a scale bar and a north arrow, and tables."""
    plt = _plt()

    from ._core import _is_vector

    if isinstance(obj, (pd.DataFrame, pd.Series)) and not _is_vector(obj):
        fig, ax = plt.subplots(figsize=opts.get("figsize") or (8, 5), layout="constrained")
        _draw_table(ax, obj, title)
        return fig
    if isinstance(obj, (np.ndarray, xr.DataArray)) and not (isinstance(obj, xr.DataArray) and "x" in obj.dims
                                                             and obj.rio.crs is not None):
        fig, ax = plt.subplots(figsize=opts.get("figsize") or (8, 6))
        im = ax.imshow(np.asarray(obj), cmap="viridis", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.75)
        ax.set_title(title or "")
        return fig
    maps, shared, tables = _panels(obj, opts, title)
    n = len(maps) + len(tables)
    if n == 0:
        raise TypeError(f"nothing to plot in a {type(obj).__name__}")
    if shared:
        opts = _shared_scale(maps, opts)
    ncols = int(opts.get("ncols") or min(n, 3 if n != 4 else 2))
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=opts.get("figsize") or (5.6 * ncols, 4.9 * nrows), squeeze=False,
                             layout="constrained")
    axes = list(axes.ravel())
    legends = {}  # the same colours drawn once, beside the last map that has them
    for i, (name, items) in enumerate(maps):
        ax = axes[i]
        crs, drawn = _draw_spatial(ax, items, opts, legends=False)
        for lg in drawn:
            key = (tuple(lg.classes), tuple(lg.ramp), lg.vmin, lg.vmax, lg.log) if not shared else "stack"
            legends.setdefault(key, [lg, []])[1].append(ax)
        ax.set_title(name if len(maps) > 1 or not title else title, fontsize=10)
        _furnish(ax, crs, opts, first=i == 0, alone=len(maps) == 1)
    for lg, where in legends.values():
        if shared:  # one legend for the bands, named after the stack
            lg = dataclasses.replace(lg, title=str(getattr(obj, "name", None) or ""))
        _legend(where if len(where) > 1 else where[0], lg)
    for ax, (name, table) in zip(axes[len(maps):], tables):
        _draw_table(ax, table, name)
    for ax in axes[n:]:
        ax.set_visible(False)
    if title and len(maps) > 1:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig
