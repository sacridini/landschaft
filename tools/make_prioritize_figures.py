# SPDX-License-Identifier: GPL-3.0-or-later
"""Figures of the Learn chapters on conservation planning (docs/learn/prioritization*), computed with
ls.prioritize on Serra Verde (ls.prioritize.example_data). The numbers quoted in the text are written to
build/prioritize_figures.json.

    python tools/make_prioritize_figures.py            # every figure
    python tools/make_prioritize_figures.py hotspots   # the figures whose name contains a word
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    LightSource,
    LinearSegmentedColormap,
    ListedColormap,
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

import landschaft as ls

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "figures"
NUMBERS = ROOT / "build" / "prioritize_figures.json"
P = ls.prioritize

# ----------------------------------------------------------------------------- style
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
OCEAN, LAND = "#dfe8ef", "#ecebe6"
SELECTED, PROTECTED, URBAN, TARGET = "#1f7a4d", "#4a3aa7", "#52514e", "#e34948"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#c9a227"]
CMAP_BIO = LinearSegmentedColormap.from_list("bio", ["#eef6ef", "#a9d5b3", "#4fa06a", "#1f7a4d", "#0b4d2c"])
CMAP_COST = LinearSegmentedColormap.from_list("cost", ["#fdf0e6", "#f6c29f", "#e88550", "#c2551f", "#7a2a0a"])
CMAP_VALUE = LinearSegmentedColormap.from_list("value", ["#e7f0fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CMAP_THREAT = LinearSegmentedColormap.from_list("threat", ["#fdecec", "#f4a3a2", "#e34948", "#a8201f", "#5e0f0f"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
})

FIGURES: dict = {}
NUM: dict = {}


def figure(fn):
    FIGURES[fn.__name__] = fn
    return fn


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  {name}.png", flush=True)


def style_chart(ax, grid_axis="y"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- data
def load(name):
    return ls.load_raster(P.example_data(name))


COST = load("cost")
FEATURES = load("features")
NAMES = [str(b) for b in FEATURES.band.values]
LABELS = [n.replace("_", " ").capitalize() for n in NAMES]
PROTECTED_R = load("protected")
URBAN_R = load("urban")
ELEV = load("elevation")
LINES = load("lines")
LAND_MASK = np.isfinite(COST.values)
_CACHE: dict = {}


def base_problem(target=0.3):
    return P.problem(COST, FEATURES).add_min_set_objective().add_relative_targets(target).add_binary_decisions()


def solve(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def min_set(target=0.3, gap=0.0):
    return solve(("minset", target, gap), lambda: base_problem(target).add_highs_solver(gap=gap).solve())


def held(sol, p=None):
    p = p or base_problem()
    return P.eval_feature_representation_summary(p, sol)["relative_held"].to_numpy()


# ----------------------------------------------------------------------------- map helpers
def base_map(ax, title=None, relief=True):
    shade = np.where(LAND_MASK, LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(ELEV.values) * 25), np.nan)
    ax.imshow(np.where(LAND_MASK, np.nan, 1.0), cmap=ListedColormap([OCEAN]), interpolation="nearest")
    if relief:
        ax.imshow(shade, cmap="Greys_r", vmin=-0.3, vmax=1.6, interpolation="bilinear", alpha=0.35)
    else:
        ax.imshow(np.where(LAND_MASK, 1.0, np.nan), cmap=ListedColormap([LAND]), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)


def overlay(ax, mask, color, alpha=1.0):
    m = np.where(np.asarray(mask) > 0.5, 1.0, np.nan)
    ax.imshow(m, cmap=ListedColormap([color]), alpha=alpha, interpolation="nearest", vmin=0, vmax=1)


def lines(ax, rivers=True, roads=False):
    if rivers:
        overlay(ax, np.nan_to_num(LINES.sel(band="rivers").values), "#3b82c4", 0.8)
    if roads:
        overlay(ax, np.nan_to_num(LINES.sel(band="roads").values), "#8a8983", 0.6)


def solution_map(ax, sol, title=None, color=SELECTED, protected=False, values=None):
    base_map(ax, title)
    v = np.asarray(sol.selection.values) if values is None else values
    overlay(ax, np.nan_to_num(v), color)
    overlay(ax, np.nan_to_num(URBAN_R.values), URBAN, 0.9)
    if protected:
        overlay(ax, np.nan_to_num(PROTECTED_R.values) * (np.nan_to_num(v) > 0.5), PROTECTED)


def continuous(ax, arr, cmap, title=None, label=None, vmin=None, vmax=None, zero_empty=True, cbar=True):
    base_map(ax, title, relief=False)
    a = np.asarray(arr, dtype=float)
    if zero_empty:
        a = np.where(a > 0, a, np.nan)
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.outline.set_visible(False)
        if label:
            cb.set_label(label)
    return im


def representation_bars(ax, values: dict, target=None, title=None, colors=None):
    k = len(values)
    w = 0.8 / k
    x = np.arange(len(NAMES))
    colors = colors or SERIES
    for i, (lab, v) in enumerate(values.items()):
        ax.bar(x + (i - (k - 1) / 2) * w, np.asarray(v) * 100, w, label=lab, color=colors[i % len(colors)])
    if target is not None:
        ax.axhline(target * 100, color=TARGET, lw=1.4, ls="--", label=f"target ({target:.0%})")
    ax.set_xticks(x, LABELS, rotation=35, ha="right")
    ax.set_ylabel("share of the feature held (%)")
    style_chart(ax)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=k + (target is not None))
    if title:
        ax.set_title(title)


def legend_below(fig, handles, ncol=None, y=-0.02):
    fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles), bbox_to_anchor=(0.5, y), fontsize=8.5)


# ----------------------------------------------------------------------------- figures
@figure
def prio_sv_landscape():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9))
    base_map(a, "Serra Verde")
    lines(a, roads=True)
    overlay(a, np.nan_to_num(URBAN_R.values), URBAN)
    overlay(a, np.nan_to_num(PROTECTED_R.values), PROTECTED, 0.85)
    continuous(b, COST.values, CMAP_COST, "Cost of every planning unit", "thousand USD per km²", zero_empty=False)
    legend_below(fig, [Patch(color=OCEAN, label="ocean (no data)"), Patch(color=URBAN, label="cities"),
                       Patch(color=PROTECTED, label="protected areas"), Patch(color="#3b82c4", label="rivers"),
                       Patch(color="#8a8983", label="roads")])
    NUM["units"] = int(LAND_MASK.sum())
    NUM["cost_range"] = [float(np.nanmin(COST)), float(np.nanmax(COST))]
    save(fig, "prio_sv_landscape")


@figure
def prio_sv_features():
    fig, axes = plt.subplots(2, 4, figsize=(12, 4.6))
    for ax, name, lab in zip(axes.flat, NAMES, LABELS):
        v = FEATURES.sel(band=name).values
        continuous(ax, v, CMAP_BIO, lab, vmin=0, vmax=1, cbar=False)
        NUM.setdefault("feature_cells", {})[name] = int(np.nansum(v > 0))
    sm = plt.cm.ScalarMappable(cmap=CMAP_BIO, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("habitat suitability")
    cb.outline.set_visible(False)
    save(fig, "prio_sv_features")


@figure
def prio_sv_hotspots():
    sol = min_set(0.3)
    budget = sol.cost
    rich = np.nansum(np.nan_to_num(FEATURES.values), axis=0)
    c = COST.values
    order = np.argsort(-np.where(LAND_MASK, rich, -np.inf), axis=None)
    pick = np.zeros(c.size)
    spent = 0.0
    for i in order:
        ci = c.flat[i]
        if not np.isfinite(ci):
            continue
        if spent + ci > budget:
            break
        pick[i] = 1
        spent += ci
    pick = pick.reshape(c.shape)
    p = base_problem()
    hot = P.eval_feature_representation_summary(p, pick.ravel()[p.units.index])["relative_held"].to_numpy()
    opt = held(sol)
    fig = plt.figure(figsize=(12.5, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3])
    a, b, d = (fig.add_subplot(gs[0, k]) for k in range(3))
    solution_map(a, sol, "A. The richest cells", values=pick, color="#c2551f")
    solution_map(b, sol, "B. The optimized plan")
    representation_bars(d, {"richest cells": hot, "optimized": opt}, 0.3, "Same budget, different outcomes",
                        colors=["#c2551f", SELECTED])
    NUM["hotspots"] = {"budget": budget, "hot_held": hot.round(3).tolist(), "opt_held": opt.round(3).tolist(),
                       "hot_met": int((hot >= 0.3 - 1e-9).sum()), "units": float(sol.values.sum())}
    save(fig, "prio_sv_hotspots")


@figure
def prio_targets():
    targets = [0.1, 0.3, 0.5]
    curve_t = np.round(np.arange(0.05, 0.65, 0.05), 2)
    costs = [min_set(t, 0.005).cost for t in curve_t]
    fig = plt.figure(figsize=(12.5, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15])
    for k, t in enumerate(targets):
        s = min_set(t)
        solution_map(fig.add_subplot(gs[0, k]), s, f"{t:.0%} of every feature\n{s.cost / 1000:.1f} million USD")
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(curve_t * 100, np.array(costs) / 1000, color=SELECTED, lw=2, marker="o", ms=4)
    ax.set_xlabel("target (% of every feature)")
    ax.set_ylabel("cost of the plan (million USD)")
    ax.set_title("The price of the targets")
    style_chart(ax, "both")
    NUM["targets"] = {f"{t:.2f}": min_set(t).cost for t in targets}
    NUM["target_curve"] = dict(zip([f"{t:.2f}" for t in curve_t], costs))
    save(fig, "prio_targets")


@figure
def prio_objectives():
    budget = 0.5 * min_set(0.3).cost
    p = P.problem(COST, FEATURES).add_relative_targets(0.3).add_binary_decisions()
    sols = {
        "maximum utility": P.problem(COST, FEATURES).add_max_utility_objective(budget).add_binary_decisions()
        .add_highs_solver(gap=0.005).solve(),
        "minimum shortfall": p.add_min_shortfall_objective(budget).add_highs_solver(gap=0.005).solve(),
        "maximum features": p.add_max_features_objective(budget).add_highs_solver(gap=0.005).solve(),
    }
    fig = plt.figure(figsize=(12.5, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
    for k, (name, s) in enumerate(sols.items()):
        solution_map(fig.add_subplot(gs[0, k]), s, name.capitalize(), color=SERIES[k])
    ax = fig.add_subplot(gs[1, :])
    representation_bars(ax, {k: held(s) for k, s in sols.items()}, 0.3,
                        f"Representation with the same budget ({budget / 1000:.1f} million USD: half the 30% plan)")
    NUM["objectives"] = {"budget": budget, **{k: {"held": held(s).round(3).tolist(),
                                                  "met": int((held(s) >= 0.3 - 1e-6).sum())} for k, s in sols.items()}}
    save(fig, "prio_objectives")


def boundary(sol, p):
    return float(P.eval_boundary_summary(p, sol)["boundary"].iloc[0])


@figure
def prio_boundary():
    pens = [0.0, 0.002, 0.01, 0.05]
    p = base_problem()
    sols = []
    for v in pens:
        q = p if v == 0 else p.add_boundary_penalties(v)
        sols.append(solve(("blm", v), lambda q=q: q.add_highs_solver(gap=0.01).solve()))
    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.2])
    rows = []
    for k, (v, s) in enumerate(zip(pens, sols)):
        solution_map(fig.add_subplot(gs[0, k]), s, f"penalty {v:g}")
        rows.append((s.cost, boundary(s, p)))
    ax = fig.add_subplot(gs[0, 4])
    c, b = np.array(rows).T
    ax.plot(b / 1000, c / 1000, color=SELECTED, marker="o")
    for v, x, y in zip(pens, b / 1000, c / 1000):
        ax.annotate(f"{v:g}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("boundary of the plan (km)")
    ax.set_ylabel("cost (million USD)")
    ax.set_title("Compactness costs a little")
    style_chart(ax, "both")
    NUM["boundary"] = {f"{v:g}": {"cost": float(x), "boundary_km": float(y) / 1000} for v, x, y in zip(pens, c, b)}
    save(fig, "prio_boundary")


@figure
def prio_spatial_tools():
    c, f = P.coarsen(COST, FEATURES, 3)
    base = P.problem(c, f).add_min_set_objective().add_relative_targets(0.2).add_binary_decisions()
    plain = base.add_highs_solver(gap=0.01).solve()
    neigh = base.add_neighbors_constraints(2).add_highs_solver(gap=0.01, time_limit=60).solve()
    contig = base.add_contiguity_constraints().add_highs_solver(gap=0.05, time_limit=120).solve()
    term = ls.load_raster(P.example_data("terminals"))
    tc = np.zeros(c.shape)
    tv = np.nan_to_num(term.values)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            tc[i, j] = tv[3 * i:3 * i + 3, 3 * j:3 * j + 3].max() if np.isfinite(c.values[i, j]) else np.nan
    corr = base.add_corridor_constraints(c.copy(data=tc), resistance=c).add_highs_solver(gap=0.05, time_limit=120).solve()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, s, t in zip(axes, (plain, neigh, contig, corr),
                        ("No spatial rule", "At least 2 selected neighbours", "One contiguous network",
                         "A corridor between the parks")):
        ax.imshow(np.where(np.isfinite(c.values), 0.0, np.nan), cmap=ListedColormap([LAND]))
        ax.imshow(np.where(np.nan_to_num(s.selection.values) > 0.5, 1.0, np.nan), cmap=ListedColormap([SELECTED]))
        if t.startswith("A corridor"):
            ax.imshow(np.where(tc > 0.5, 1.0, np.nan), cmap=ListedColormap([PROTECTED]))
        ax.set_title(t, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
    NUM["spatial_tools"] = {k: {"cost": s.cost, "gap": s.gap, "status": s.status}
                            for k, s in zip(("plain", "neighbors", "contiguity", "corridor"), (plain, neigh, contig, corr))}
    save(fig, "prio_spatial_tools")


@figure
def prio_calibration():
    p = base_problem().add_highs_solver(gap=0.02, time_limit=60)
    blended = P.calibrate_penalty(p, "boundary", values=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
    hier = P.calibrate_budget(p.add_highs_solver(gap=0.05, time_limit=180), "boundary",
                              inflation=[0.005, 0.01, 0.02, 0.04, 0.07, 0.1], n_threads=3)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for ax, cal, lab, fmt in ((a, blended, "Blended: a weight on the boundary", "{:g}"),
                              (b, hier, "Hierarchical: least boundary under a cost cap", "{:,.0f}")):
        t = cal.table.sort_values("boundary")
        ax.plot(t["boundary"] / 1000, t["cost"] / 1000, color="#8a8983", marker="o", ms=4, zorder=1)
        for _, r in t.iterrows():
            ax.annotate(fmt.format(r["value"]), (r["boundary"] / 1000, r["cost"] / 1000), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=INK2)
        for col, colour, label in (("topsis", TARGET, "TOPSIS"), ("knee", SERIES[0], "knee")):
            r = t[t[col]]
            ax.scatter(r["boundary"] / 1000, r["cost"] / 1000, s=150, facecolors="none", edgecolors=colour, lw=2,
                       label=label, zorder=3)
        ax.set_xlabel("boundary (km)")
        ax.set_ylabel("cost (million USD)")
        ax.set_title(lab)
        style_chart(ax, "both")
        ax.legend(fontsize=8)
    NUM["calibration"] = {"blended": blended.table[["value", "cost", "boundary", "topsis", "knee"]].to_dict("list"),
                          "hierarchical": hier.table[["value", "cost", "boundary", "topsis", "knee"]].to_dict("list")}
    save(fig, "prio_calibration")


@figure
def prio_pareto():
    p = base_problem()
    top = min_set(0.3).cost
    front = P.pareto_front(p.add_highs_solver(gap=0.005), budgets=np.linspace(top / 8, top * 1.01, 8))
    t = front.table
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(t["value"] / 1000, t["shortfall"], color=SELECTED, marker="o")
    for _, r in t.iterrows():
        ax.annotate(f"{int(r['met'])}/8 met", (r["value"] / 1000, r["shortfall"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7.5, color=INK2)
    ax.set_xlabel("budget (million USD)")
    ax.set_ylabel("total shortfall (sum of shares of targets missing)")
    ax.set_title("What each budget buys")
    style_chart(ax, "both")
    NUM["pareto"] = t[["value", "cost", "shortfall", "met"]].to_dict("list")
    save(fig, "prio_pareto")


@figure
def prio_importance():
    p = base_problem().add_highs_solver(gap=0.001)
    sol = min_set(0.3, 0.001)
    rc = P.eval_replacement_importance(p, sol)
    fe = P.eval_ferrier_importance(p, sol).sel(band="total")
    port = p.add_gap_portfolio(20, pool_gap=0.02).solve()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    continuous(axes[0], rc.values, CMAP_VALUE, "Replacement cost (rescaled)", vmin=0, vmax=1)
    continuous(axes[1], fe.values, CMAP_VALUE, "Irreplaceability (Ferrier)")
    continuous(axes[2], port.frequency.values, CMAP_BIO, f"Selected in {len(port)} near-optimal plans", "share of plans",
               vmin=0, vmax=1)
    f = port.frequency.values
    NUM["importance"] = {"portfolio": len(port), "always": int(np.nansum(f >= 1 - 1e-9)),
                         "sometimes": int(np.nansum((f > 0) & (f < 1 - 1e-9))), "plan_units": float(sol.values.sum()),
                         "costs": port.summary["cost"].round(1).tolist()}
    save(fig, "prio_importance")


@figure
def prio_actions():
    threats = load("threats")
    d = P.actions_raster_data(COST, FEATURES, threats, action_cost=[30, 10], target_recovery=0.3,
                              target_conservation=0.1)
    sol = d.problem("minimizeCosts").add_highs_solver(gap=0.01).solve()
    m = sol.maps()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    continuous(axes[0], threats.sel(band="deforestation").values, CMAP_THREAT, "Deforestation threat", vmin=0, vmax=1)
    continuous(axes[1], threats.sel(band="fire").values, CMAP_THREAT, "Fire threat", vmin=0, vmax=1)
    ax = axes[2]
    base_map(ax, "The plan: where to monitor and act")
    mon = np.nan_to_num(m.sel(band="monitoring").values)
    df = np.nan_to_num(m.sel(band="deforestation").values)
    fi = np.nan_to_num(m.sel(band="fire").values)
    overlay(ax, mon, "#a9d5b3")
    overlay(ax, df, SERIES[1])
    overlay(ax, fi, TARGET)
    overlay(ax, (df > 0.5) & (fi > 0.5), SERIES[3])
    ax.legend(handles=[Patch(color="#a9d5b3", label="monitored (conservation)"),
                       Patch(color=SERIES[1], label="act on deforestation"), Patch(color=TARGET, label="act on fire"),
                       Patch(color=SERIES[3], label="both actions")], loc="upper left",
              bbox_to_anchor=(1.0, 1.0), fontsize=8)
    cost = sol.cost()
    NUM["actions"] = {"objective": sol.objective, "status": sol.status, "units": int(sol.units.sum()),
                      "actions": int(sol.actions.sum()), "cost": {k: float(cost[k].iloc[0]) for k in cost.columns},
                      "n_units": d.n_units, "n_actions": d.n_actions}
    save(fig, "prio_actions")


@figure
def prio_response_curves():
    x = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    ax = axes[0]
    for (d1, d2, d3, d4), col, lab in (((0, 1, 0, 1), SERIES[0], "linear (defaults)"),
                                       ((0.2, 0.7, 0, 1), SERIES[1], "tolerant up to 0.2, lost above 0.7"),
                                       ((0, 0.5, 0.2, 1), SERIES[2], "never below 0.2")):
        y = np.where(x <= d1, d4, np.where(x >= d2, d3, d4 + (d3 - d4) * (x - d1) / (d2 - d1)))
        ax.plot(x, y, color=col, lw=2, label=lab)
    ax.set_xlabel("threat intensity")
    ax.set_ylabel("probability the feature persists")
    ax.set_title("Response of a feature to a threat")
    ax.legend(fontsize=8)
    style_chart(ax, "both")
    ax = axes[1]
    for k, col in zip((1, 2, 3), SERIES):
        ax.plot(x, x ** k, color=col, lw=2, label=f"curve = {k}")
    ax.set_xlabel("share of the threats removed in a unit")
    ax.set_ylabel("benefit (share of recovery)")
    ax.set_title("How removing threats becomes benefit")
    ax.legend(fontsize=8)
    style_chart(ax, "both")
    save(fig, "prio_response_curves")


@figure
def prio_zonation_vs_milp():
    p = base_problem()
    sol = min_set(0.3)
    comp = P.compare_zonation(p, sol)
    rank = comp.zonation.rank
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9])
    ax = fig.add_subplot(gs[0, 0])
    base_map(ax, "Zonation's ranking", relief=False)
    im = ax.imshow(rank.values, cmap="Spectral_r", vmin=0, vmax=1, interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    solution_map(fig.add_subplot(gs[0, 1]), sol, "Zonation's best cells, same cost", color=SERIES[1],
                 values=comp.selection.sel(band="zonation").values)
    solution_map(fig.add_subplot(gs[0, 2]), sol, "The minimum set (optimization)")
    t = comp.table
    representation_bars(fig.add_subplot(gs[1, :]), {"Zonation": t["zonation"], "optimization": t["milp"]}, 0.3,
                        "Share of every feature held at the same cost", colors=[SERIES[1], SELECTED])
    NUM["zonation_vs_milp"] = {"table": t.round(3).to_dict("list"), "summary": comp.summary.round(1).to_dict("list")}
    save(fig, "prio_zonation_vs_milp")


@figure
def prio_condition():
    cond = load("condition")
    z = P.zonation_features(FEATURES, condition=[cond] * 8)
    s0 = min_set(0.3)
    s1 = P.problem(COST, z).add_min_set_objective().add_relative_targets(0.3).add_binary_decisions() \
        .add_highs_solver(gap=0).solve()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    continuous(axes[0], cond.values, CMAP_BIO, "Habitat condition", vmin=0, vmax=1, zero_empty=False)
    continuous(axes[1], FEATURES.sel(band="lowland_tapir").values, CMAP_BIO, "Tapir habitat", vmin=0, vmax=1, cbar=False)
    continuous(axes[2], z.sel(band="lowland_tapir").values, CMAP_BIO, "... in good condition", vmin=0, vmax=1,
               cbar=False)
    a, b = np.nan_to_num(s0.selection.values) > 0.5, np.nan_to_num(s1.selection.values) > 0.5
    base_map(axes[3], "Plans without and with condition")
    overlay(axes[3], a & b, SELECTED)
    overlay(axes[3], a & ~b, "#c2a24a")
    overlay(axes[3], ~a & b, SERIES[0])
    axes[3].legend(handles=[Patch(color=SELECTED, label="both"), Patch(color="#c2a24a", label="habitat only"),
                            Patch(color=SERIES[0], label="with condition only")], loc="upper left",
                   bbox_to_anchor=(1.0, 1.0), fontsize=8)
    NUM["condition"] = {"cost_plain": s0.cost, "cost_condition": s1.cost, "shared": int((a & b).sum()),
                        "only_plain": int((a & ~b).sum()), "only_condition": int((~a & b).sum())}
    save(fig, "prio_condition")


@figure
def prio_extinction():
    r = np.linspace(0, 1, 300)
    from landschaft.prioritize._milp_compile import sar_lines

    c, s = sar_lines(0.25, 6)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    ax = axes[0]
    ax.plot(r, 1 - r ** 0.25, color=TARGET, lw=2.2, label="risk = 1 - r^0.25")
    ax.plot(r, 1 - np.minimum(1, np.min(c[:, None] + s[:, None] * r[None, :], axis=0)), color=INK2, lw=1.2, ls="--",
            label="its piecewise-linear form (6 pieces)")
    ax.set_xlabel("share of the habitat kept (r)")
    ax.set_ylabel("extinction risk")
    ax.set_title("The species-area relationship")
    ax.legend(fontsize=8)
    style_chart(ax, "both")
    budget = 0.5 * min_set(0.3).cost
    mu = P.problem(COST, FEATURES).add_max_utility_objective(budget).add_binary_decisions().add_highs_solver(gap=0.005).solve()
    er = P.problem(COST, FEATURES).add_min_extinction_risk_objective(budget).add_binary_decisions() \
        .add_highs_solver(gap=0.005).solve()
    p = base_problem()
    representation_bars(axes[1], {"maximum utility": held(mu, p), "least extinction risk": held(er, p)}, None,
                        f"Same budget ({budget / 1000:.1f} million USD)", colors=[SERIES[0], TARGET])
    risk_mu = P.eval_extinction_risk(p, mu)["extinction_risk"].iloc[-1]
    risk_er = P.eval_extinction_risk(p, er)["extinction_risk"].iloc[-1]
    NUM["extinction"] = {"budget": budget, "risk_max_utility": float(risk_mu), "risk_min_risk": float(risk_er)}
    save(fig, "prio_extinction")


@figure
def prio_heuristic():
    p = base_problem()
    opt = min_set(0.3)
    ann = p.add_anneal_solver(num_reps=20, num_iters=2_000_000, seed=3).solve()
    res = P.marxan(*_marxan_tables(p), num_reps=20, num_iters=2_000_000, seed=3)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.hist(res.summary["Cost"] / 1000, bins=12, color="#c9a227", alpha=0.85,
            label="20 runs of Marxan (2 million iterations each)")
    ax.axvline(opt.cost / 1000, color=SELECTED, lw=2.2, label="the proven optimum")
    ax.axvline(ann.cost / 1000, color=SERIES[0], lw=1.6, ls="--", label="annealing with repair (best run)")
    ax.set_xlabel("cost of the plan (million USD)")
    ax.set_ylabel("runs")
    ax.set_title("Heuristic runs and the optimum")
    ax.legend(fontsize=8)
    style_chart(ax)
    NUM["heuristic"] = {"optimum": opt.cost, "anneal_best": ann.cost, "marxan_min": float(res.summary["Cost"].min()),
                        "marxan_max": float(res.summary["Cost"].max()),
                        "marxan_shortfall_runs": int((res.summary["Shortfall"] > 1e-9).sum())}
    save(fig, "prio_heuristic")


def _marxan_tables(p):
    import pandas as pd

    u = p.units
    rij = u.rij.tocoo()
    pu = pd.DataFrame({"id": np.arange(1, u.n + 1), "cost": u.cost, "status": 0})
    spec = pd.DataFrame({"id": np.arange(1, len(NAMES) + 1), "prop": 0.3, "spf": 10.0})
    puvspr = pd.DataFrame({"species": rij.row + 1, "pu": rij.col + 1, "amount": rij.data})
    return pu, spec, puvspr


@figure
def prio_coarsen():
    c, f = P.coarsen(COST, FEATURES, 3)
    coarse = P.problem(c, f).add_min_set_objective().add_relative_targets(0.3).add_binary_decisions() \
        .add_highs_solver(gap=0).solve()
    fine = P.refine(coarse, COST)
    sol = min_set(0.3)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    solution_map(axes[0], sol, f"1 km cells: cost {sol.cost:,.0f}")
    solution_map(axes[1], sol, f"3 km blocks, refined: cost {coarse.cost:,.0f}", values=fine.values, color=SERIES[0])
    NUM["coarsen"] = {"fine": sol.cost, "coarse": coarse.cost, "coarse_units": coarse.values.size}
    save(fig, "prio_coarsen")


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color="white", weight="bold", wrap=True)


def _arrow(ax, a, b, color=INK2, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=1.4,
                                 connectionstyle=f"arc3,rad={rad}"))


@figure
def prio_workflow():
    steps = ["1. Scope and\nstakeholders", "2. Goals and\ntargets", "3. Data", "4. Gap\nanalysis",
             "5. Scenarios", "6. Evaluation and\ndiscussion", "7. Implementation\nand monitoring"]
    colors = ["#52514e", "#2a78d6", "#2a78d6", "#1f7a4d", "#1f7a4d", "#eb6834", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    w, h, gap = 1.6, 0.9, 0.22
    xs = [0.1 + k * (w + gap) for k in range(len(steps))]
    for x, s, c in zip(xs, steps, colors):
        _box(ax, x, 1.2, w, h, s, c)
    for k in range(len(steps) - 1):
        _arrow(ax, (xs[k] + w, 1.65), (xs[k + 1], 1.65))
    _arrow(ax, (xs[5] + w / 2, 1.2), (xs[1] + w / 2, 1.2), color=TARGET, rad=-0.25)
    ax.text((xs[1] + xs[5]) / 2 + w / 2, 0.25, "revise the goals, the data or the rules, and solve again",
            ha="center", color=TARGET, fontsize=9)
    save(fig, "prio_workflow")


@figure
def prio_objective_tree():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    q = "#52514e"
    _box(ax, 4.3, 3.7, 3.4, 0.7, "Are the targets non-negotiable?", q)
    _box(ax, 0.3, 2.3, 3.2, 0.7, "Only a budget, no targets", q)
    _box(ax, 7.8, 2.3, 3.6, 0.7, "Is the budget fixed too?", q)
    _box(ax, 0.3, 0.9, 3.2, 0.8, "Maximum utility\n(add_max_utility_objective)", SELECTED)
    _box(ax, 3.9, 0.9, 2.5, 0.8, "Minimum set\n(add_min_set_objective)", SELECTED)
    _box(ax, 6.7, 0.9, 2.3, 0.8, "Partial progress\ncounts? Minimum\nshortfall", SELECTED)
    _box(ax, 9.3, 0.9, 2.4, 0.8, "All or nothing?\nMaximum features", SELECTED)
    _arrow(ax, (5.2, 3.7), (2.0, 3.0))
    _arrow(ax, (6.8, 3.7), (9.6, 3.0))
    _arrow(ax, (1.9, 2.3), (1.9, 1.7))
    _arrow(ax, (8.6, 2.3), (5.2, 1.7))
    _arrow(ax, (9.6, 2.3), (7.8, 1.7))
    _arrow(ax, (10.4, 2.3), (10.5, 1.7))
    for xy, t in (((3.2, 3.45), "no"), ((8.6, 3.45), "yes"), ((6.4, 2.05), "no"), ((8.4, 1.95), "yes"),
                  ((10.6, 2.0), "yes")):
        ax.text(*xy, t, fontsize=8.5, color=INK2, style="italic")
    save(fig, "prio_objective_tree")


@figure
def prio_gap():
    p = base_problem()
    lo = p.add_proportion_decisions().add_highs_solver(gap=0).solve()
    exact = min_set(0.3)
    loose = min_set(0.3, 0.1)
    rows = [("Linear relaxation: fractions of cells allowed.\nA bound: no real plan can be cheaper", lo.objective,
             SERIES[0]),
            ("The optimum, proven with gap = 0", exact.cost, SELECTED),
            (f"The plan returned with gap = 0.1 (10 %):\nwithin {loose.gap:.2%} of the bound", loose.cost,
             SERIES[1])]
    fig, ax = plt.subplots(figsize=(9, 2.9))
    for k, (lab, v, c) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.hlines(y, lo.objective, v, color=c, lw=2, alpha=0.5)
        ax.plot(v, y, "o", color=c, ms=10)
        ax.annotate(f"{v:,.0f}", (v, y), xytext=(10, -4), textcoords="offset points", fontsize=9, color=c,
                    weight="bold")
        ax.text(1.02, y, lab, transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK2)
    ax.axvline(lo.objective, color=SERIES[0], lw=1, ls=":")
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(lo.objective - 15, loose.cost + 30)
    ax.set_xlabel("cost of the plan (thousand USD)")
    ax.set_title("Bounds and gaps: how far a plan can be from the best one")
    style_chart(ax, "x")
    NUM["gap"] = {"bound": float(lo.objective), "optimum": float(exact.cost), "loose": float(loose.cost),
                  "loose_bound": float(loose.bound), "loose_gap": loose.gap}
    save(fig, "prio_gap")


def main():
    words = sys.argv[1:]
    for name, fn in FIGURES.items():
        if not words or any(w in name for w in words):
            fn()
    NUMBERS.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(NUMBERS.read_text()) if NUMBERS.exists() else {}
    old.update(NUM)
    NUMBERS.write_text(json.dumps(old, indent=1, default=float))


if __name__ == "__main__":
    main()
