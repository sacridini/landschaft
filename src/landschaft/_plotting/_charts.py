# SPDX-License-Identifier: GPL-3.0-or-later
"""Charts of tables (plotly): the form follows the table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _palettes

_LAYOUT = {"template": "plotly_white", "margin": {"l": 60, "r": 20, "t": 50, "b": 50}, "height": 420,
               "font": {"family": "Inter, Segoe UI, Helvetica, Arial, sans-serif", "size": 12}}


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("charts need plotly (pip install landschaft[plot])") from err
    return go


def figure(title: str | None = None, **layout):
    go = _go()
    fig = go.Figure()
    fig.update_layout(**_LAYOUT, title=title, **layout)
    return fig


def lines(x, series: dict, title=None, xlabel=None, ylabel=None, highlight: dict | None = None, percent=False):
    """Lines of several series against x; ``highlight`` series drawn thicker, on top."""
    go = _go()
    fig = figure(title, xaxis_title=xlabel, yaxis_title=ylabel)
    many = len(series) > 12
    colours = _palettes.distinct(len(series))
    for (name, y), c in zip(series.items(), colours):
        fig.add_trace(go.Scattergl(x=x, y=y, name=str(name), mode="lines",
                                   line={"width": 1, "color": "#b8b8b8" if many else c},
                                   showlegend=not many, hovertemplate=f"{name}<br>%{{x}}: %{{y:.4g}}<extra></extra>"))
    for i, (name, y) in enumerate((highlight or {}).items()):
        fig.add_trace(go.Scattergl(x=x, y=y, name=str(name), mode="lines",
                                   line={"width": 3, "color": _palettes.CATEGORICAL[i % 10]}))
    if percent:
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
    return fig


def bars(categories, series: dict, title=None, xlabel=None, ylabel=None, horizontal=None, stacked=False):
    go = _go()
    cats = [str(c) for c in categories]
    horizontal = len(cats) > 12 if horizontal is None else horizontal
    fig = figure(title, barmode="stack" if stacked else "group", xaxis_title=ylabel if horizontal else xlabel,
                 yaxis_title=xlabel if horizontal else ylabel)
    colours = _palettes.distinct(len(series))
    for (name, values), c in zip(series.items(), colours):
        v = list(values)
        if horizontal:
            fig.add_trace(go.Bar(y=cats, x=v, name=str(name), orientation="h", marker_color=c))
        else:
            fig.add_trace(go.Bar(x=cats, y=v, name=str(name), marker_color=c))
    if horizontal:
        fig.update_layout(height=max(420, 22 * len(cats) + 100))
        fig.update_yaxes(autorange="reversed")
    return fig


def scatter(x, y, title=None, xlabel=None, ylabel=None, text=None):
    go = _go()
    fig = figure(title, xaxis_title=xlabel, yaxis_title=ylabel)
    fig.add_trace(go.Scattergl(x=x, y=y, mode="markers", text=text, marker={"size": 6, "opacity": 0.7,
                                                                              "color": _palettes.CATEGORICAL[0]}))
    return fig


def histogram(values, title=None, xlabel=None, bins=50):
    go = _go()
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    fig = figure(title, xaxis_title=xlabel, yaxis_title="count", bargap=0.02)
    fig.add_trace(go.Histogram(x=v, nbinsx=bins, marker_color=_palettes.CATEGORICAL[0]))
    return fig


def metrics(table: pd.DataFrame, title=None):
    """A landscapemetrics-style long table (level, class, metric, value): one panel per metric."""
    from plotly.subplots import make_subplots

    go = _go()
    names = list(dict.fromkeys(table["metric"]))
    has_class = "class" in table and table["class"].notna().any()
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names)
    for i, m in enumerate(names):
        sub = table[table["metric"] == m]
        r, c = i // cols + 1, i % cols + 1
        if has_class and sub["class"].notna().any():
            x = sub["class"].astype("Int64").astype(str)
        else:
            x = sub.get("layer", pd.Series(range(len(sub)))).astype(str)
        fig.add_trace(go.Bar(x=x, y=sub["value"], name=m, marker_color=_palettes.CATEGORICAL[i % 20],
                             showlegend=False), row=r, col=c)
    fig.update_layout(**{**_LAYOUT, "height": max(360, 300 * rows)}, title=title)
    return fig


def table_chart(df, title=None):
    """The chart a table calls for."""
    if isinstance(df, pd.Series):
        name = df.name or "value"
        if _palettes.is_number(df.dtype):
            if df.index.is_monotonic_increasing and _palettes.is_number(df.index.dtype) and len(df) > 12:
                return lines(df.index.to_numpy(), {name: df.to_numpy()}, title, df.index.name, name)
            return bars(df.index, {name: df.to_numpy()}, title, df.index.name, name)
        return bars(df.value_counts().index, {"count": df.value_counts().to_numpy()}, title, name, "count")
    if {"metric", "value"} <= set(df.columns):
        return metrics(df, title)
    numeric = [c for c in df.columns if _palettes.is_number(df[c].dtype)]
    labels = [c for c in df.columns if c not in numeric]
    if not numeric:
        raise ValueError("no numeric column to plot")
    if labels:
        return bars(df[labels[0]], {c: df[c] for c in numeric[:8]}, title, labels[0])
    if len(numeric) == 2 and len(df) > 20 and not df[numeric[0]].is_monotonic_increasing:
        return scatter(df[numeric[0]], df[numeric[1]], title, numeric[0], numeric[1])
    x = df.index.to_numpy()
    if df[numeric[0]].is_monotonic_increasing and len(numeric) > 1:
        x, numeric = df[numeric[0]].to_numpy(), numeric[1:]
        return lines(x, {c: df[c].to_numpy() for c in numeric}, title, df.columns[0])
    if len(df) > 30:
        return lines(x, {c: df[c].to_numpy() for c in numeric}, title)
    return bars(df.index, {c: df[c] for c in numeric[:8]}, title)
