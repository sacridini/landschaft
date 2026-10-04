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


# --- land change simulations ---------------------------------------------------------------------------


def _class_flows(a, b) -> pd.DataFrame:
    """Cells of each class in map a (rows) going to each class in map b (columns)."""
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    ok = ~(np.isnan(a) | np.isnan(b))
    t = pd.crosstab(a[ok], b[ok])
    t.index = [f"{v:g}" for v in t.index]
    t.columns = [f"{v:g}" for v in t.columns]
    return t


def _land_maps(maps, dim, opts, title):
    """A class map per step or year (a slider over them), or the last one as a chart without a georeference."""
    from ._core import _is_geo_raster, _map, _Panel, _plain_array_chart

    if _is_geo_raster(maps):
        return _map([("land use", maps)], {**opts, "categorical": True if opts.get("categorical") is None
                                           else opts["categorical"]}, title)
    last = maps.isel({dim: -1})
    return [_Panel("chart", _plain_array_chart(np.asarray(last), title or f"land use, {dim} {last[dim].item()}"))]


def _cover_lines(maps, dim, title):
    from . import _charts
    from ._core import _Panel

    v = maps.values.reshape(maps.shape[0], -1)
    classes = sorted(np.unique(v[~np.isnan(v)]))
    series = {f"{c:g}": (v == c).sum(axis=1) for c in classes}
    return _Panel("chart", _charts.lines(maps[dim].values, series, title, xlabel=dim, ylabel="cells"), "cover")


@register("Simulation")
def _simulation(sim, opts, title):
    """The maps (a slider over the steps), the cells of each class per step, the flows first -> last."""
    from . import _charts
    from ._core import _Panel

    maps = sim.maps
    panels = _land_maps(maps, "step", opts, title or "Simulated land use")
    panels.append(_cover_lines(maps, "step", "Cells of each class"))
    panels.append(_Panel("chart", _charts.sankey(_class_flows(maps.values[0], maps.values[-1]),
                                                 "Classes from the start to the last step", unit="cells",
                                                 left="start", right=f"step {maps.shape[0] - 1}"), "flows"))
    return panels


@register("Projection")
def _projection(proj, opts, title):
    """The projected maps (a slider over the years), cover per year, flows from the later map, the check."""
    from . import _charts
    from ._core import _Panel

    maps = proj.maps
    y = maps["year"].values
    panels = _land_maps(maps, "year", opts, title or f"Projected land use, {y[0]}-{y[-1]}")
    panels.append(_cover_lines(maps, "year", "Cells of each class"))
    if maps.shape[0] > 1:
        panels.append(_Panel("chart", _charts.sankey(_class_flows(maps.values[0], maps.values[-1]),
                                                     f"Classes, {y[0]} -> {y[-1]}", unit="cells", left=str(y[0]),
                                                     right=str(y[-1])), "flows"))
    if proj.validation is not None:
        v = proj.validation
        panels.append(_summary({"preset": proj.settings["preset"], "seed": proj.settings["seed"],
                                "figure of merit": round(v["figure_of_merit"], 4),
                                "agreement": round(v["overall"], 4), "hits": v["hits"], "misses": v["misses"],
                                "false alarms": v["false_alarms"], "wrong hits": v["wrong_hits"]},
                               "Check on the observed period"))
    return panels


@register("Ensemble")
def _ensemble(ens, opts, title):
    """The probability of change by the last year (a slider over the scenarios) and the mean cover."""
    from . import _charts
    from ._core import _is_geo_raster, _map, _Panel, _plain_array_chart

    panels = []
    p = ens.change_probability().isel(year=-1)
    if _is_geo_raster(p):
        panels += _map([("probability of change", p)], {**opts, "categorical": False, "cmap": opts.get("cmap")
                                                        or "magma", "vmin": 0.0, "vmax": 1.0},
                       title or f"Probability of change by {ens.years[-1]}")
    else:
        panels.append(_Panel("chart", _plain_array_chart(np.asarray(p.isel(scenario=0)),
                                                         title or "probability of change")))
    cover = ens.cover().groupby(["scenario", "year"]).mean(numeric_only=True).drop(columns="replicate")
    series = {f"{s}: {c:g}": cover.loc[s][c].to_numpy() for s in ens.scenarios for c in ens.classes}
    panels.append(_Panel("chart", _charts.lines(ens.years, series, "Cells of each class (mean of the replicates)",
                                                xlabel="year", ylabel="cells"), "cover"))
    return panels


@register("Validation")
def _validation(v, opts, title):
    """The budget of agreement per resolution, the figure of merit and the fuzzy similarity per window."""
    from . import _charts
    from ._core import _Panel

    b = v.three_map.agreement_budget()
    o = b[b.level == "overall"]
    parts = ["hits", "misses", "wrong_hits", "false_alarms"]
    fig = _charts.bars([f"{f} cells" for f in o["factor"]], {p.replace("_", " "): o[p].to_numpy() for p in parts},
                       title or "Agreement and disagreement of the change (share of the area)",
                       xlabel="resolution", ylabel="share", stacked=True)
    panels = [_Panel("chart", fig, "budget")]
    fom = v.three_map.figure_of_merit()["overall"]
    panels.append(_Panel("chart", _charts.lines(fom.index.to_numpy(), {"figure of merit": fom.to_numpy()},
                                                "Figure of merit by resolution", xlabel="block side (cells)",
                                                ylabel="figure of merit"), "figure of merit"))
    s = v.similarity
    panels.append(_Panel("chart", _charts.lines(s["window"].to_numpy(), {
        "simulated changes": s["simulated"].to_numpy(), "observed changes": s["observed"].to_numpy(),
        "minimum": s["minimum"].to_numpy()}, "Fuzzy similarity of the differences", xlabel="window (cells)",
        ylabel="similarity"), "similarity"))
    panels.append(_summary({k: (round(float(x), 4) if x is not None else None) for k, x in v.summary().items()},
                           "Summary"))
    return panels


@register("Calibration")
def _calibration(cal, opts, title):
    """The loss of every evaluation and the loss against each setting (the best marked)."""
    from . import _charts
    from ._core import _Panel

    h = cal.history
    best = int(np.argmin(h["loss"].to_numpy()))
    panels = [_Panel("chart", _charts.lines(np.arange(len(h)), {"loss": h["loss"].to_numpy(),
                                                              "best so far": np.minimum.accumulate(h["loss"])},
                                            title or "Loss of each evaluation", xlabel="evaluation",
                                            ylabel="loss"), "loss")]
    go = _charts._go()
    for name in cal.best:
        x = h[name]
        if not np.issubdtype(np.asarray(x).dtype, np.number):
            x = x.astype(str)
        fig = _charts.figure(f"Loss against {name}", xaxis_title=name, yaxis_title="loss")
        fig.add_trace(go.Scatter(x=x, y=h["loss"], mode="markers", name="evaluations",
                                 marker={"color": "#7f7f7f"}))
        fig.add_trace(go.Scatter(x=[x.iloc[best]], y=[h["loss"].iloc[best]], mode="markers", name="best",
                                 marker={"color": "#e03030", "size": 12}))
        panels.append(_Panel("chart", fig, name))
    panels.append(_summary({**{k: v for k, v in cal.best.items()}, "loss": round(cal.loss, 6),
                            "evaluations": len(h), "method": cal.settings.get("method")}, "Best settings"))
    return panels


# ----------------------------------------------------------------------------- ls.prioritize (optimization)

_SELECTION = {"cmap": {0: "#d9d9d9", 1: "#1a9850"}, "labels": {0: "not selected", 1: "selected"},
              "categorical": True}


def _selection_panels(sel, opts, title):
    """A selection (raster, polygons or a table) on a map; a table without geometry gives nothing."""
    from ._core import _is_geo_raster, _is_vector, _map

    if _is_geo_raster(sel):
        v = np.asarray(sel.values, dtype=np.float64)
        binary = np.isin(v[np.isfinite(v)], (0.0, 1.0)).all()
        o = {**opts, **(_SELECTION if binary else {"cmap": opts.get("cmap") or "greens", "categorical": False,
                                                   "vmin": 0.0, "vmax": 1.0})}
        return _map([(title, sel)], o, title)
    if _is_vector(sel):
        return _map([(title, sel, {"column": "solution"})], opts, title)
    return []


def _coverage_chart(coverage: pd.DataFrame, title="Targets"):
    from . import _charts
    from ._core import _Panel

    series = {"held": coverage["relative_held"].to_numpy()}
    if "relative_target" in coverage:
        series["target"] = coverage["relative_target"].to_numpy()
    return _Panel("chart", _charts.bars(list(coverage["feature"]), series, title, ylabel="share of the feature"),
                  title)


@register("Solution")
def _prio_solution(sol, opts, title):
    """The selection, the targets' coverage and the solution's figures."""
    panels = _selection_panels(sol.selection, opts, title or "Selected planning units")
    panels.append(_coverage_chart(sol.coverage, "Representation of the features"))
    met = sol.coverage.get("met")
    extra = {"targets met": f"{int(met.sum())} of {len(met)}"} if met is not None else {}
    panels.append(_summary({"objective": sol.objective, "cost": sol.cost,
                            "units selected": float(np.nansum(sol.values)), **extra, "status": sol.status,
                            "gap": sol.gap, "solver": sol.solver, "seconds": sol.seconds}))
    return panels


@register("Portfolio")
def _prio_portfolio(port, opts, title):
    """How often every unit is selected, and the solutions' objectives."""
    from . import _charts
    from ._core import _is_geo_raster, _map, _Panel

    panels = []
    freq = port.frequency
    if _is_geo_raster(freq):
        panels += _map([("selection frequency", freq)], {**opts, "cmap": opts.get("cmap") or "greens",
                                                       "categorical": False, "vmin": 0.0, "vmax": 1.0},
                       title or "Selection frequency")
    s = port.summary
    panels.append(_Panel("chart", _charts.bars(list(s["solution"]), {"objective": s["objective"].to_numpy()},
                                               "Objective of each solution"), "objectives"))
    return panels


@register("TradeOffs")
def _prio_tradeoffs(t, opts, title):
    """Cost against the criterion traded off, the candidates TOPSIS and the knee choose marked."""
    from . import _charts
    from ._core import _Panel

    tab = t.table
    crit = t.criterion
    go = _charts._go()
    fig = _charts.figure(title or f"Cost against {crit}", xaxis_title=crit, yaxis_title="cost")
    order = np.argsort(tab[crit].to_numpy(), kind="stable")
    fig.add_trace(go.Scatter(x=tab[crit].to_numpy()[order], y=tab["cost"].to_numpy()[order], mode="lines+markers",
                             name="candidates", text=[f"value {v:g}" for v in tab["value"].to_numpy()[order]],
                             marker={"color": "#7f7f7f"}))
    for col, colour in (("topsis", "#e03030"), ("knee", "#377eb8")):
        pick = tab[tab[col]]
        fig.add_trace(go.Scatter(x=pick[crit], y=pick["cost"], mode="markers", name=col,
                                 marker={"color": colour, "size": 13, "symbol": "circle-open",
                                         "line": {"width": 3}}))
    panels = [_Panel("chart", fig, "trade-off")]
    best = tab[tab["topsis"]].iloc[0]
    panels.append(_summary({"candidates": len(tab), "TOPSIS: value": best["value"], "TOPSIS: cost": best["cost"],
                            f"TOPSIS: {crit}": best[crit]}, "Choice"))
    return panels


@register("ZonationComparison")
def _prio_zonation_comparison(c, opts, title):
    """The two selections side by side, every feature's share held by each, and their figures."""
    from . import _charts
    from ._core import _map, _Panel

    panels = _map([("milp", c.selection.sel(band="milp")), ("zonation", c.selection.sel(band="zonation"))],
                  {**opts, **_SELECTION}, title or "Optimization and Zonation at the same cost")
    t = c.table
    panels.append(_Panel("chart", _charts.bars(list(t["feature"]), {"optimization": t["milp"].to_numpy(),
                                                                     "zonation": t["zonation"].to_numpy()},
                                               "Share of every feature held", ylabel="share"), "features"))
    panels.append(_Panel("chart", _charts.table_chart(c.summary.round(4), "Figures"), "figures"))
    return panels


@register("ActionsSolution")
def _prio_actions(sol, opts, title):
    """The monitored units and the actions (maps, for problems from rasters), and the benefits."""
    from . import _charts
    from ._core import _map, _Panel

    panels = []
    if sol.data.grid is not None:
        m = sol.maps()
        panels += _map([(str(b), m.sel(band=b)) for b in m["band"].values], {**opts, **_SELECTION},
                       title or "Monitoring and actions")
    b = sol.benefit()
    panels.append(_Panel("chart", _charts.bars([str(f) for f in b["feature"]],
                                               {"conservation": b["benefit.conservation"].to_numpy(),
                                                "recovery": b["benefit.recovery"].to_numpy()},
                                               "Benefit of every feature", stacked=True), "benefits"))
    cost = sol.cost()
    panels.append(_summary({"objective": sol.objective, **{k: float(cost[k].iloc[0]) for k in cost.columns},
                            "status": sol.status, "gap": sol.gap, "seconds": sol.seconds}))
    return panels


@register("ActionsPortfolio")
def _prio_actions_portfolio(port, opts, title):
    """How often every unit is monitored and every action taken, and the solutions' objectives."""
    from . import _charts
    from ._core import _map, _Panel

    panels = []
    if port.solutions[0].data.grid is not None:
        m = port.maps()
        panels += _map([(str(b), m.sel(band=b)) for b in m["band"].values],
                       {**opts, "cmap": opts.get("cmap") or "greens", "categorical": False, "vmin": 0.0,
                        "vmax": 1.0}, title or "How often monitored or acted on")
    s = port.summary
    panels.append(_Panel("chart", _charts.bars(list(s["solution"]), {"objective": s["objective"].to_numpy()},
                                               "Objective of each solution"), "objectives"))
    return panels


@register("PrioritizationReport")
def _prio_report(rep, opts, title):
    """A solution with its figures, the importance of the selected units and the comparison with Zonation."""
    from . import _charts
    from ._core import _is_geo_raster, _map, _Panel

    panels = _prio_solution(rep.solution, opts, title or rep.title)
    for name, frame in rep.tables.items():
        panels.append(_Panel("chart", _charts.table_chart(frame.round(6), name), name))
    for name, da in rep.maps.items():
        if _is_geo_raster(da):
            panels += _map([(name, da)], {**opts, "cmap": opts.get("cmap") or "viridis", "categorical": False},
                           name)
    if rep.comparison is not None:
        panels += _prio_zonation_comparison(rep.comparison, opts, None)
    return panels
