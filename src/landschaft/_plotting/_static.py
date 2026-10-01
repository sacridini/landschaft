# SPDX-License-Identifier: GPL-3.0-or-later
"""Static figures (matplotlib) with the same colours and legends, satellite imagery through contextily."""

from __future__ import annotations

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
        raise ImportError("static plots need matplotlib (pip install landschaft[plot])") from err
    return plt


def _contextily():
    try:
        import contextily
    except ImportError:
        return None
    return contextily


def _legend(ax, legend: Legend | None) -> None:
    if legend is None:
        return
    plt = _plt()
    if legend.classes:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=c, edgecolor="#444", label=lab) for c, lab in legend.classes[:30]]
        ax.legend(handles=handles, title=legend.title, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False,
                  fontsize=8, title_fontsize=9)
    elif legend.ramp:
        from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

        cmap = LinearSegmentedColormap.from_list("ls", legend.ramp)
        norm = LogNorm(legend.vmin, legend.vmax) if legend.log and legend.vmin > 0 else Normalize(legend.vmin,
                                                                                                 legend.vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = ax.figure.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(legend.title)


def _draw_spatial(ax, items, opts) -> None:
    """Rasters and vectors on one axes; in Web Mercator over imagery when contextily is there."""
    from . import _raster
    from ._core import _band_dim, _is_rgb, _is_vector

    ctx = _contextily() if opts.get("basemap") else None
    merc = ctx is not None
    crs = None
    for name, obj in items:
        if _is_vector(obj):
            import geopandas as gpd

            gdf = obj if hasattr(obj, "geometry") else gpd.GeoDataFrame(geometry=[obj])
            if type(gdf).__name__ == "GeoSeries":
                gdf = gpd.GeoDataFrame(geometry=gdf)
            if merc and gdf.crs is not None:
                gdf = gdf.to_crs(3857)
            crs = crs or gdf.crs
            column = opts.get("column")
            kw = {"alpha": opts.get("opacity") or 0.7, "edgecolor": "#222", "linewidth": 0.5}
            if column:
                gdf.plot(ax=ax, column=column, cmap=opts.get("cmap") if isinstance(opts.get("cmap"), str) else None,
                         legend=True, **kw)
            else:
                gdf.plot(ax=ax, color=opts.get("color") or "#ffd60a", **kw)
            continue
        da = obj
        if da.ndim == 3 and _is_rgb(da):
            img = _raster.render_rgb(da, max_size=opts["max_size"], title=name)
            ext = img.mercator_extent
            ax.imshow(img.rgba, extent=(ext[0], ext[2], ext[1], ext[3]), interpolation="nearest")
            crs = crs or 3857
            continue
        if da.ndim == 3:
            da = da.isel({_band_dim(da): 0})
        img = _raster.render(da, max_size=opts["max_size"], cmap=opts.get("cmap"), vmin=opts.get("vmin"),
                             vmax=opts.get("vmax"), log=opts.get("log"), opacity=opts.get("opacity") or 0.9,
                             categorical=opts.get("categorical"), labels=opts.get("labels"), title=name,
                             mercator=merc)
        ext = img.mercator_extent if merc else img.extent
        ax.imshow(img.rgba, extent=(ext[0], ext[2], ext[1], ext[3]), interpolation="nearest", zorder=2)
        crs = crs or (3857 if merc else da.rio.crs)
        _legend(ax, img.legend)
    if merc:
        try:
            ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.Esri.WorldImagery, attribution_size=6,
                            zorder=1)
        except Exception as err:  # noqa: BLE001 - offline: no imagery
            warnings.warn(f"no satellite imagery ({err})", stacklevel=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


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


def figure(obj, opts: dict, title: str | None):
    """A matplotlib figure of anything ls.plot draws."""
    plt = _plt()
    from ._core import _is_geo_raster, _is_vector

    if isinstance(obj, (list, tuple)) and obj and all(_is_geo_raster(o) or _is_vector(o) for o in obj):
        spatial, tables = [(getattr(o, "name", None) or f"layer {i + 1}", o) for i, o in enumerate(obj)], []
    elif _is_geo_raster(obj) or _is_vector(obj):
        spatial, tables = [(getattr(obj, "name", None) or "value", obj)], []
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        spatial, tables = [], [(title, obj)]
    elif isinstance(obj, (np.ndarray, xr.DataArray)):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(np.asarray(obj), cmap="viridis", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.75)
        ax.set_title(title or "")
        return fig
    else:
        from .._save import _items

        spatial, tables = [], []
        for k, v in _items(obj).items():
            if (_is_geo_raster(v) and v.ndim in (2, 3)) or (_is_vector(v) and type(v).__name__ != "Tree"):
                spatial.append((k, v))
            elif isinstance(v, (pd.DataFrame, pd.Series)) and len(v):
                tables.append((k, v))
    n = (1 if spatial else 0) + min(len(tables), 3)
    if n == 0:
        raise TypeError(f"nothing to plot in a {type(obj).__name__}")
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6), squeeze=False)
    axes = list(axes[0])
    if spatial:
        ax = axes.pop(0)
        _draw_spatial(ax, spatial[:1] if len(spatial) > 1 and not isinstance(obj, (list, tuple)) else spatial, opts)
        ax.set_title(title or spatial[0][0])
    for ax, (name, table) in zip(axes, tables):
        _draw_table(ax, table, name)
    fig.tight_layout()
    return fig
