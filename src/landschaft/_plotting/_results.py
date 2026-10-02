# SPDX-License-Identifier: GPL-3.0-or-later
"""The plots of landschaft's results that call for more than the generic one (maps, charts, numbers)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _palettes
from ._registry import register


def _summary(values: dict, title=None):
    from ._core import _Panel, _summary_html

    return _Panel("html", _summary_html(values, title))


@register("ZonationResult")
def _zonation(res, opts, title):
    """The rank map (Zonation's colours, 0-1) and the performance curves of the features."""
    from . import _charts
    from ._core import _is_geo_raster, _map, _Panel, _plain_array_chart

    panels = []
    rank = res.rank
    if _is_geo_raster(rank):
        panels += _map([("priority rank", rank)], {**opts, "cmap": opts.get("cmap") or "rank",
                                                   "vmin": 0.0 if opts.get("vmin") is None else opts["vmin"],
                                                   "vmax": 1.0 if opts.get("vmax") is None else opts["vmax"],
                                                   "categorical": False}, title or "Priority rank")
    else:
        panels.append(_Panel("chart", _plain_array_chart(np.asarray(rank), title or "Priority rank")))
    fc = res.feature_curves
    kept = 1.0 - fc["rank"].to_numpy()
    values = fc.drop(columns="rank")
    series = {c: values[c].to_numpy() for c in values.columns}
    highlight = {"mean": values.mean(axis=1).to_numpy(), "worst feature": values.min(axis=1).to_numpy()}
    fig = _charts.lines(kept, series, "Performance curves", "fraction of the landscape kept (best cells first)",
                        "fraction of each feature kept", highlight=highlight, percent=True)
    panels.append(_Panel("chart", fig, "Performance curves"))
    panels.append(_summary({"features": len(res.features), "cells": int(np.isfinite(np.asarray(rank)).sum()),
                            "iterations": res.iterations, "ranking error (%)": res.ranking_error,
                            "AUC": res.auc, **{f"top of level {k}": v for k, v in res.hierarchic_mask_ranks.items()}}))
    return panels


@register("HabitatGraph")
def _graph(g, opts, title):
    """The nodes coloured by their attribute; with ``distance=``, the links up to that distance."""
    from ._core import _map

    attr = np.asarray(g.attributes, dtype=np.float64)
    name = opts.get("column") or "attribute"
    vector = getattr(g, "_vector", None)
    items = []
    if vector is not None:
        nodes = g.to_geodataframe(attr, name=name)
        items.append(("nodes", nodes))
        distance = opts.get("distance")
        if distance is not None and getattr(g, "_pi", None) is not None:
            import geopandas as gpd
            import shapely

            pi, pj, val = np.asarray(g._pi), np.asarray(g._pj), np.asarray(g._value, dtype=np.float64)
            keep = (val <= float(distance)) & (pi < pj)
            cent = shapely.centroid(nodes.geometry.values)
            xy = shapely.get_coordinates(cent)
            lines = shapely.linestrings(np.stack([xy[pi[keep]], xy[pj[keep]]], axis=1))
            # the links coloured by their distance (or cost): the shortest, the strongest, bright
            items.append(("links", gpd.GeoDataFrame({"distance": val[keep]}, geometry=lines, crs=nodes.crs),
                          {"column": "distance", "cmap": "magma_r", "categorical": False, "vmin": None,
                           "vmax": None, "log": None}))
        return _map(items, {**opts, "column": name if opts.get("column") is None else opts["column"]}, title)
    raster = g.to_raster(attr, name=name)
    return _map([(name, raster)], {**opts, "categorical": False}, title)


@register("SpatialResult")
def _biodiverse(res, opts, title):
    """An index of a Biodiverse spatial analysis on its group cells (``column=``, else the first)."""
    from . import _charts
    from ._core import _map, _Panel

    numeric = [c for c in res.table.columns if _palettes.is_number(res.table[c].dtype)]
    column = opts.get("column") or (numeric[0] if numeric else res.table.columns[0])
    gdf = res.to_geodataframe([column])
    if gdf.crs is not None:
        return _map([(column, gdf)], {**opts, "column": column}, title or column)
    c = gdf.geometry.centroid
    fig = _charts.scatter(c.x, c.y, title or column, "x", "y", text=gdf[column].astype(str))
    fig.data[0].marker.update(color=gdf[column], colorscale="Viridis", showscale=True, symbol="square")
    fig.update_yaxes(scaleanchor="x")
    return [_Panel("chart", fig, column)]


@register("PCAResult")
def _pca(res, opts, title):
    """The landscapes on the first two principal components, and the loadings of the first."""
    from . import _charts
    from ._core import _Panel

    s = res.scores
    if s.shape[1] < 2:
        return [_Panel("chart", _charts.table_chart(s, title), title)]
    a, b = s.columns[:2]
    ratio = getattr(res, "explained_variance_ratio", None)
    lab = (lambda i, c: f"{c} ({ratio.iloc[i]:.0%})") if isinstance(ratio, pd.Series) else (lambda i, c: str(c))
    fig = _charts.scatter(s[a], s[b], title or "Principal components", lab(0, a), lab(1, b),
                          text=[str(i) for i in s.index])
    load = res.loadings[a].sort_values(key=np.abs, ascending=False)
    return [_Panel("chart", fig, "scores"),
            _Panel("chart", _charts.bars(load.index, {str(a): load.to_numpy()}, f"Loadings of {a}"), "loadings")]


# --- change between two dates --------------------------------------------------------------------------


def _change_map(res, opts, title, default):
    from ._core import _is_geo_raster, _map, _Panel, _plain_array_chart

    m = res.change_map
    if _is_geo_raster(m):
        return _map([(default, m)], opts, title or default[:1].upper() + default[1:])
    return [_Panel("chart", _plain_array_chart(np.asarray(m), title or default))]


def _foreground_flows(stable_bg, gain, loss, stable_fg) -> pd.DataFrame:
    names = ["Background", "Foreground"]
    return pd.DataFrame([[stable_bg, gain], [loss, stable_fg]], index=names, columns=names)


_FG_COLOURS = {"Background": "#c8c8c8", "Foreground": "#00a000"}
_FG_FLOWS = {("Background", "Foreground"): "#00c000", ("Foreground", "Background"): "#e03030"}  # gain, loss


@register("SimpleChangeResult")
def _simple_change(res, opts, title):
    """The change map, the flows between background and foreground, and the counts."""
    from . import _charts
    from ._core import _Panel

    panels = _change_map(res, opts, title, "change")
    flows = _foreground_flows(res.stable_bg, res.gain, res.loss, res.stable_fg)
    panels.append(_Panel("chart", _charts.sankey(flows, "Background and foreground, A → B", _FG_COLOURS,
                                                       flows=_FG_FLOWS), "flows"))
    panels.append(_summary({"stable background": res.stable_bg, "gain": res.gain, "loss": res.loss,
                            "stable foreground": res.stable_fg, "net change": res.gain - res.loss,
                            "missing": res.missing}))
    return panels


@register("MorphologicalChangeResult")
def _morphological_change(res, opts, title):
    """The change map and its counts (structural gain and loss, spurious change)."""
    from . import _charts
    from ._core import _Panel

    panels = _change_map(res, opts, title, "morphological change")
    counts = pd.Series({"stable foreground": res.stable_fg, "gain": res.mcd_gain, "loss": res.mcd_loss,
                        "spurious change": res.spurious}, name="pixels")
    fig = _charts.bars(counts.index, {"pixels": counts.to_numpy()}, "Pixels by kind of change")
    fig.data[0].marker.color = ["#00a000", "#3b8bff", "#e03030", "#aaaaaa"]
    panels.append(_Panel("chart", fig, "counts"))
    panels.append(_summary({"net change": res.net_change, "stable background": res.stable_bg,
                            "elasticity": res.elasticity, "missing": res.missing}))
    return panels


def _density_colours(names) -> dict:
    """The FAD / FOS colours of class names (``Rare (0-9%)``...), background grey."""
    from ._palettes import FAD_CLASS_COLOURS, rgb_to_hex

    out = {}
    for n in names:
        key = str(n).split(" (")[0].split("-", 1)[-1].lower()
        for code, word in ((1, "rare"), (2, "patchy"), (3, "transitional"), (4, "dominant"), (5, "interior"),
                           (6, "intact"), (101, "background")):
            if key.startswith(word):
                out[str(n)] = rgb_to_hex(FAD_CLASS_COLOURS[code])
    return out


@register("FOSChangeResult")
def _fos_change(res, opts, title):
    """The change map, the flows between the fragmentation classes, and the counts."""
    from . import _charts
    from ._core import _Panel

    panels = _change_map(res, opts, title, "FOS change")
    m = res.matrix_6x6
    panels.append(_Panel("chart", _charts.sankey(m, "Fragmentation classes of the common foreground, A → B",
                                                 _density_colours(m.index)), "flows"))
    panels.append(_summary({"foreground gain": res.fg_gain, "foreground loss": res.fg_loss,
                            "same class": res.diag_same_class, "changed class": res.diff_class,
                            "less fragmented (%)": res.frag_decrease_pct,
                            "more fragmented (%)": res.frag_increase_pct}))
    return panels


@register("FADMultiscaleChangeResult")
def _fad_change(res, opts, title):
    """The flows between the FAD classes at one scale (``scale=``, default the multiscale map) and the
    averages of every scale."""
    from . import _charts
    from ._core import _Panel

    scale = opts.get("scale", "mscale")
    if scale not in res.scales:
        raise ValueError(f"no scale {scale!r}: the scales are {list(res.scales)}")
    sc = res.scales[scale]
    m = sc.matrix.copy()
    m.index = [str(i).split("-", 1)[-1] for i in m.index]
    m.columns = [str(c).split("-", 1)[-1] for c in m.columns]
    label = "multiscale" if scale == "mscale" else f"{scale} x {scale}"
    panels = [_Panel("chart", _charts.sankey(m, title or f"FAD classes, A → B ({label})",
                                             _density_colours(m.index)), "flows")]
    try:
        panels.append(_Panel("chart", _charts.table_chart(res.averages, "Averages by scale"), "averages"))
    except ValueError:
        pass
    panels.append(_summary({"scale": label, "foreground A": sc.fg_area_a, "foreground B": sc.fg_area_b,
                            "net change": sc.net_change, "same class": sc.same_class, "changed class": sc.changed,
                            "less fragmented (%)": sc.frag_decrease_pct, "more fragmented (%)": sc.frag_increase_pct}))
    return panels


# --- species distributions -----------------------------------------------------------------------------


@register("SDMResult")
def _sdm(res, opts, title):
    """With ``env=`` (the predictor rasters), the suitability map (one band per model, the ensemble
    and its spread) with the presences; the response curves; the cross-validated scores."""
    from plotly.subplots import make_subplots

    from .. import sdm
    from . import _charts
    from ._core import _map, _Panel

    panels = []
    env = opts.get("env")
    if env is not None:
        suit = sdm.predict(res, env).rename("suitability")
        items = [("suitability", suit)]
        if res.data is not None and getattr(res.data, "coords", None) is not None:
            presences = res.data.coords[np.asarray(res.data.y) == 1]
            items.append(("presences", presences, {"column": None, "color": "#ffffff", "cmap": None}))
        panels += _map(items, {**opts, "cmap": opts.get("cmap") or "viridis", "categorical": False,
                               "vmin": 0.0 if opts.get("vmin") is None else opts["vmin"],
                               "vmax": 1.0 if opts.get("vmax") is None else opts["vmax"]},
                       title or "Habitat suitability")
    if res.data is not None:
        curves = sdm.response_curves(res, n=60)
        preds = list(dict.fromkeys(curves["predictor"]))
        cols = min(3, len(preds))
        rows = -(-len(preds) // cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=preds, shared_yaxes=True,
                            vertical_spacing=0.12 if rows > 1 else 0.1)
        models = list(dict.fromkeys(curves["model"]))
        colours = dict(zip(models, _palettes.distinct(len(models))))
        for i, p in enumerate(preds):
            for m in models:
                sub = curves[(curves["predictor"] == p) & (curves["model"] == m)]
                fig.add_scatter(x=sub["value"], y=sub["suitability"], mode="lines", name=str(m),
                                line={"color": colours[m], "width": 2}, legendgroup=str(m), showlegend=i == 0,
                                row=i // cols + 1, col=i % cols + 1)
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(**{**_charts._LAYOUT, "height": max(360, 260 * rows)},
                          title="Response curves (the other predictors at their mean)")
        panels.append(_Panel("chart", fig, "response curves"))
    s = res.summary
    metrics = [m for m in ("auc_test", "tss", "boyce") if (m, "mean") in s.columns]
    if metrics:
        go = _charts._go()
        fig = _charts.figure("Cross-validated scores (mean and standard deviation over the folds)",
                             barmode="group", yaxis_title="score")
        for m, c in zip(metrics, _palettes.distinct(len(metrics))):
            fig.add_trace(go.Bar(x=[str(i) for i in s.index], y=s[(m, "mean")], name=m, marker_color=c,
                                 error_y={"type": "data", "array": s[(m, "std")].fillna(0).to_numpy()}))
        panels.append(_Panel("chart", fig, "scores"))
    numbers = {f"threshold ({m}, max TSS)": float(t) for m, t in res.thresholds.items()}
    if res.data is not None:
        y = np.asarray(res.data.y)
        numbers |= {"presences": int((y == 1).sum()), "background": int((y == 0).sum())}
    panels.append(_summary(numbers | {"predictors": list(res.predictors)}))
    return panels


# --- landscapes compared -------------------------------------------------------------------------------


@register("ClustergramResult")
def _clustergram(res, opts, title):
    """Schonlau's clustergram: for each k a dot per cluster at its mean, sized by its rows, and the
    rows moving between the clusters of k and k + 1 as lines as wide as their number."""
    from . import _charts
    from ._core import _Panel

    go = _charts._go()
    means = res.means
    n = max(len(means), 1)
    ks = list(means.columns)
    fig = _charts.figure(title or "Clustergram", xaxis_title="number of clusters (k)",
                         yaxis_title="PCA-weighted mean of the clusters" if res.pca_weighted else "mean of the clusters",
                         showlegend=False)
    for i, k in enumerate(ks[:-1]):
        if ks[i + 1] != k + 1:
            continue
        for (a, b), m in means.groupby([k, k + 1]).size().items():
            fig.add_trace(go.Scatter(x=[k, k + 1], y=[a, b], mode="lines", hoverinfo="text",
                                     text=f"{m} rows", line={"color": "rgba(40,40,40,0.55)",
                                                              "width": max(0.5, 30 * m / n)}))
    for k in ks:
        counts = means[k].value_counts()
        fig.add_trace(go.Scatter(x=[k] * len(counts), y=counts.index, mode="markers",
                                 marker={"size": 6 + 30 * np.sqrt(counts.to_numpy() / n), "color": "#d62728",
                                         "line": {"color": "white", "width": 1.5}},
                                 text=[f"k = {k}: {c} rows" for c in counts.to_numpy()], hoverinfo="text"))
    fig.update_xaxes(dtick=1)
    panels = [_Panel("chart", fig, "clustergram")]
    if res.dropped:
        panels.append(_summary({"rows left out (missing values)": len(res.dropped)}))
    return panels


# --- corridors -------------------------------------------------------------------------------------------


def _mask_outline(mask):
    """The cells of a mask as polygons (a path stays visible however far the map zooms out)."""
    import geopandas as gpd
    import shapely
    from rasterio.features import shapes

    from ..io import _transform

    m = np.asarray(mask) > 0
    if not m.any():
        return None
    geoms = [shapely.geometry.shape(g) for g, v in shapes(m.astype("uint8"), mask=m, transform=_transform(mask))
             if v == 1]
    return gpd.GeoDataFrame(geometry=geoms, crs=mask.rio.crs)


@register("LeastCostPathResult")
def _least_cost_path(res, opts, title):
    """The path over the combined cost surface, and its numbers."""
    from ._core import _is_geo_raster, _map, _Panel, _plain_array_chart

    cost, path = res.combined_cost, res.lcp_mask
    if _is_geo_raster(cost) and _is_geo_raster(path):
        items = [("combined cost", cost)]
        outline = _mask_outline(path)
        if outline is not None:
            items.append(("least-cost path", outline, {"column": None, "color": "#ff2d2d", "opacity": 0.95}))
        panels = _map(items, {**opts, "categorical": False}, title or "Least-cost path")
    else:
        panels = [_Panel("chart", _plain_array_chart(np.asarray(cost), title or "combined cost"))]
    panels.append(_summary({"path length (cells)": res.path_length, "cells to restore": res.rest_pixels,
                            "cost": res.expense, "mean distance of the restored cells": res.avg_distance}))
    return panels
