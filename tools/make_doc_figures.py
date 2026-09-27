# SPDX-License-Identifier: GPL-3.0-or-later
"""Figures of the documentation website, drawn from the bundled example data.

Usage::

    python tools/make_doc_figures.py [name ...]

Writes PNGs to ``docs/assets/figures`` (all figures, or the ones named). Every
figure is computed with landschaft itself, so the pictures show what the
tutorials' code returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import landschaft as ls  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
FIGURES = {}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
})


def figure(fn):
    FIGURES[fn.__name__] = fn
    return fn


def palette_map(ax, values, palette: dict, labels: dict | None = None, title: str = "", by_label: bool = False):
    """A categorical map drawn with a {value: (r, g, b, a)} palette."""
    values = np.asarray(values)
    keys = sorted(palette)
    lut = {k: i for i, k in enumerate(keys)}
    idx = np.vectorize(lambda v: lut.get(int(v), 0))(values)
    cmap = ListedColormap([np.array(palette[k]) / 255 for k in keys])
    ax.imshow(idx, cmap=cmap, norm=BoundaryNorm(np.arange(len(keys) + 1) - 0.5, len(keys)), interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()
    if labels:
        present = set(np.unique(values).astype(int))
        shown = [k for k in keys if k in labels and k in present]
        if by_label:
            shown.sort(key=lambda k: labels[k])
        handles = [Patch(color=np.array(palette[k]) / 255, label=labels[k]) for k in shown]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{name}.png")


BINARY = {0: (255, 255, 255, 255), 1: (238, 232, 213, 255), 2: (21, 128, 61, 255)}
BINARY_LABELS = {0: "0 missing", 1: "1 background", 2: "2 habitat"}


@figure
def quickstart_input():
    forest = ls.load_raster(ls.example_data("input"))
    fig, ax = plt.subplots(figsize=(5.2, 4))
    palette_map(ax, forest.values, BINARY, BINARY_LABELS, "Example map: habitat (green) and background")
    save(fig, "quickstart_input")


@figure
def quickstart_mspa():
    forest = ls.load_raster(ls.example_data("input"))
    morph = ls.mspa(forest, edge_width=2)
    names = {v: n for v, n in ls.MSPA_CLASSES.items()} if isinstance(ls.MSPA_CLASSES, dict) else {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    palette_map(axes[0], forest.values[300:600, 300:600], BINARY, None, "Input (detail)")
    palette_map(axes[1], morph.values[300:600, 300:600], ls.mspa_colormap(), names, "MSPA classes (detail)")
    save(fig, "quickstart_mspa")


MSPA_NAMES = ls.MSPA_CLASSES
DETAIL = (slice(300, 600), slice(300, 600))


@figure
def patterns_edge_width():
    forest = ls.load_raster(ls.example_data("input"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, w in zip(axes, (1, 5)):
        morph = ls.mspa(forest, edge_width=w)
        palette_map(ax, morph.values[DETAIL], ls.mspa_colormap(), MSPA_NAMES if w == 5 else None,
                    f"edge_width={w}")
    save(fig, "patterns_edge_width")


@figure
def patterns_spa():
    forest = ls.load_raster(ls.example_data("input"))
    result = ls.spa(forest, variant=5)
    names = {k: v.split(" (")[0] for k, v in ls.SPA_CLASSES.items()}
    fig, ax = plt.subplots(figsize=(5.5, 4))
    palette_map(ax, result.map.values[DETAIL], ls.mspa_colormap(), names, "SPA5 (detail)")
    save(fig, "patterns_spa")


@figure
def patterns_accounting():
    forest = ls.load_raster(ls.example_data("input"))
    result = ls.accounting(forest, thresholds=(50, 250, 1000, 5000, 20000))
    names = {k: v for k, v in ls.ACCOUNTING_CLASSES.items() if k not in (105, 176, 80)}
    fig, ax = plt.subplots(figsize=(6, 4.6))
    palette_map(ax, result.classes.values, ls.accounting_colormap(), names, "Objects by size class", by_label=True)
    save(fig, "patterns_accounting")


FAD_LEGEND = [(5, "Rare (< 10%)"), (25, "Patchy (10-40%)"), (50, "Transitional (40-60%)"),
              (75, "Dominant (60-90%)"), (95, "Interior (90-99%)"), (100, "Intact (100%)"), (101, "Background")]


def fad_legend(ax):
    cm = ls.fad_colormap()
    handles = [Patch(color=np.array(cm[v]) / 255, label=label) for v, label in FAD_LEGEND]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)


def landcover3():
    """The CORINE example as 1 agriculture, 2 natural, 3 developed (0: water, wetlands)."""
    import xarray as xr

    clc = ls.load_raster(ls.example_data("clc"), validate=None)
    return xr.where(clc <= 11, 3, xr.where(clc <= 22, 1, xr.where(clc <= 34, 2, 0))).astype("uint8")


@figure
def frag_fad():
    forest = ls.load_raster(ls.example_data("input"))
    result = ls.fad(forest, window_size=27)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    palette_map(ax, result.density.values, ls.fad_colormap(), None, "Forest area density, 27 x 27 window")
    fad_legend(ax)
    save(fig, "frag_fad")


@figure
def frag_scales():
    forest = ls.load_raster(ls.example_data("input"))
    result = ls.fad_multiscale(forest, window_sizes=(7, 27, 243))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, w in zip(axes, (7, 27, 243)):
        palette_map(ax, result.single_scales[w].values, ls.fad_colormap(), None, f"window {w} x {w}")
    fad_legend(axes[-1])
    save(fig, "frag_scales")


@figure
def frag_mosaic():
    lc = landcover3()
    lm = ls.landscape_mosaic(lc, window_size=27)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    palette_map(axes[0], lc.values, {0: (60, 120, 220, 255), 1: (240, 210, 110, 255), 2: (21, 128, 61, 255),
                                     3: (200, 40, 40, 255)},
                {0: "water / wetland", 1: "agriculture", 2: "natural", 3: "developed"}, "Land cover (3 classes)")
    palette_map(axes[1], lm.values, ls.lm_colormap(), ls.LM19_CLASSES, "Landscape mosaic, 27 x 27")
    fig.subplots_adjust(wspace=0.55)
    save(fig, "frag_mosaic")


def clc_crop():
    return ls.load_raster(ls.example_data("clc"), validate=None)[:600, :600]


@figure
def lsm_patches():
    lc = ls.Landscape(clc_crop())
    fig = lc.show_lsm("shape")
    fig.set_size_inches(6, 5)
    save(fig, "lsm_shape")


@figure
def lsm_window():
    landcover = ls.load_raster(ls.example_data("clc"), validate=None)[:800, :800]
    maps = ls.window_metrics(landcover, window=41, what=["lsm_l_shdi"])
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    im = ax.imshow(maps["shdi"].values, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Shannon diversity (41 x 41 window)")
    ax.set_title("Moving-window diversity")
    ax.set_axis_off()
    save(fig, "lsm_window")


@figure
def lsm_zonal():
    landcover = ls.load_raster(ls.example_data("clc"), validate=None)
    zones = ls.grid_zones(landcover, 5000)
    table = ls.zonal_metrics(landcover, zones, level="landscape", metric=["shdi"])
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    table.plot(column="shdi", cmap="viridis", legend=True, ax=ax, edgecolor="white", linewidth=0.3,
               legend_kwds={"label": "Shannon diversity per 5 km cell"})
    ax.set_title("Metrics per zone")
    ax.set_axis_off()
    save(fig, "lsm_zonal")


@figure
def net_importance():
    landcover = ls.load_raster(ls.example_data("clc"), validate=None)
    g = ls.HabitatGraph.from_raster(landcover, habitat=[23, 24, 25], area_unit="ha", distance="edge",
                                    max_distance=5000, landscape_area=landcover.size * 25 * 25 / 1e4)
    comp = g.components(distance=1000).rename("component").reset_index()
    imp = g.node_importance("PC", distance=1000, probability=0.5)
    cmap_map = g.to_raster(comp, "component").values
    dpc = g.to_raster(imp, "dPC").values
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    rng = np.random.default_rng(3)
    colours = plt.cm.tab20(rng.permutation(20))
    comp_rgb = np.ones(cmap_map.shape + (4,)) * np.array([0.93, 0.93, 0.93, 1])
    ok = ~np.isnan(cmap_map)
    comp_rgb[ok] = colours[cmap_map[ok].astype(int) % 20]
    axes[0].imshow(comp_rgb, interpolation="nearest")
    axes[0].set_title("Connected components at 1 km")
    axes[0].set_axis_off()
    bg = np.where(np.isnan(dpc), np.nan, dpc)
    axes[1].imshow(np.where(np.isnan(bg), 1, np.nan), cmap="Greys", vmin=0, vmax=12, interpolation="nearest")
    im = axes[1].imshow(bg, cmap="YlGn", interpolation="nearest")
    fig.colorbar(im, ax=axes[1], label="dPC: loss of connectivity without the patch (%)")
    axes[1].set_title("Importance of every forest patch")
    axes[1].set_axis_off()
    save(fig, "net_importance")


def resistance_map():
    landcover = clc_crop()
    table = {**{c: 100.0 for c in range(1, 12)}, **{c: 20.0 for c in range(12, 23)},
             **{c: 1.0 for c in range(23, 35)}, **{c: 200.0 for c in range(35, 45)}}
    return landcover.copy(data=np.vectorize(table.get)(landcover.values).astype(float))


@figure
def circ_omniscape():
    from matplotlib.colors import LogNorm

    resistance = resistance_map()
    flow = ls.omniscape(resistance, radius=30, block_size=5, flow_potential=True, normalized_current=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    im = axes[0].imshow(resistance.values, cmap="magma_r", norm=LogNorm(1, 200), interpolation="nearest")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="resistance")
    axes[0].set_title("Resistance")
    im = axes[1].imshow(flow["cum_currmap"].values, cmap="inferno", norm=LogNorm(), interpolation="nearest")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="cumulative current")
    axes[1].set_title("Omniscape current")
    im = axes[2].imshow(flow["normalized_cum_currmap"].values, cmap="RdBu_r", vmin=0, vmax=2, interpolation="nearest")
    fig.colorbar(im, ax=axes[2], fraction=0.046, label="current / flow potential")
    axes[2].set_title("Normalised current")
    for ax in axes:
        ax.set_axis_off()
    save(fig, "circ_omniscape")


@figure
def circ_pairwise():
    from matplotlib.colors import LogNorm
    from landschaft import circuit

    resistance = resistance_map()
    points = np.zeros(resistance.shape)
    cells = [(50, 60), (120, 500), (300, 300), (520, 100), (550, 540)]
    for k, (i, j) in enumerate(cells, 1):
        points[i, j] = k
    result = circuit.circuitscape_pairwise(resistance, points, cum_map=True)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(result.cum_current, cmap="inferno", norm=LogNorm(1e-3, None), interpolation="nearest")
    fig.colorbar(im, ax=ax, label="cumulative current, 10 pairs")
    for k, (i, j) in enumerate(cells, 1):
        ax.plot(j, i, "o", color="#a3e635", markersize=7, markeredgecolor="black")
        ax.annotate(str(k), (j, i), xytext=(5, 5), textcoords="offset points", color="white", fontweight="bold")
    ax.set_title("Current between five focal points")
    ax.set_axis_off()
    save(fig, "circ_pairwise")


def two_objects():
    forest = ls.load_raster(ls.example_data("input"))
    acc = ls.accounting(forest)
    ids = acc.ids.values
    marker = np.zeros(forest.shape, dtype="uint8")
    marker[ids == acc.largest[0][0]] = 1
    marker[ids == acc.largest[1][0]] = 2
    marker[forest.values == 0] = 3
    return forest, marker


@figure
def dist_distance():
    forest = ls.load_raster(ls.example_data("input"))
    d = ls.euclidean_distance(forest, pixel_size=25)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    v = np.asarray(d.signed_distance, dtype=float)
    lim = np.nanpercentile(np.abs(v), 99)
    im = ax.imshow(v, cmap="PuOr", vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="distance to the forest edge (m): + inside, - outside")
    ax.set_title("Signed distance to the forest edge")
    ax.set_axis_off()
    save(fig, "dist_distance")


@figure
def dist_isochrone():
    forest, marker = two_objects()
    resistance = ls.distance_to_resistance(forest)
    iso = ls.isochrone_map_ab(resistance, marker)
    cost = np.asarray(iso.cost, dtype=float)
    cost = np.where(cost > 0, cost, np.nan)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(cost, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="accumulated cost A to B")
    obj = np.ma.masked_where(~np.isin(marker, (1, 2)), marker)
    ax.imshow(obj, cmap=ListedColormap(["#15803d", "#a3e635"]), interpolation="nearest")
    rows, cols = np.nonzero(np.asarray(iso.lcp))
    ax.plot(cols, rows, "s", color="red", markersize=1.2)
    ax.set_title("Cheapest route between two forest blocks")
    ax.set_axis_off()
    save(fig, "dist_isochrone")


def two_dates():
    before = ls.load_raster(ls.example_data("input"))
    after = before.copy()
    v = after.values
    v[600:700, 100:260] = np.where(v[600:700, 100:260] == 2, 1, v[600:700, 100:260])
    v[820:900, 400:560] = np.where(v[820:900, 400:560] == 2, 1, v[820:900, 400:560])
    v[250:330, 420:520] = np.where(v[250:330, 420:520] == 1, 2, v[250:330, 420:520])
    return before, after


@figure
def change_maps():
    before, after = two_dates()
    simple = ls.simple_change(before, after)
    mcd = ls.morphological_change(before, after)
    names = {11: "stable background", 12: "gain", 21: "loss", 22: "stable habitat", 176: "spurious (boundary)",
             0: "missing", 129: "missing"}
    cm = ls.change_colormap()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    palette_map(axes[0], np.asarray(simple.change_map)[200:950, 50:650], cm, None, "Simple change")
    palette_map(axes[1], np.asarray(mcd.change_map)[200:950, 50:650], cm, names, "Morphological change")
    save(fig, "change_maps")


# Timings of docs/benchmarks.md: (task, reference tool, original [s], landschaft [s]).
BENCHMARKS = [
    ("MSPA, CORINE 2000×2000", "GuidosToolbox", 1.30, 0.078),
    ("All 133 landscape metrics", "landscapemetrics", 15.6, 0.03),
    ("sample_lsm, 6 plots", "landscapemetrics", 3.7, 0.02),
    ("PC and importances, 404 patches", "Makurhini", 10.6, 0.017),
    ("IIC and importances, 404 patches", "Makurhini", 1.4, 0.005),
    ("Least-cost distances, 398 patches", "Makurhini", 10.2, 0.5),
    ("Omniscape 800×800, radius 10", "Omniscape", 29.4, 0.64),
    ("Omniscape 1000×1000, radius 100", "Omniscape", 275, 12.4),
    ("Pairwise with current map, 1000×1000", "Circuitscape", 76.5, 9.5),
    ("Network pairwise, 1 M nodes", "Circuitscape", 27.7, 5.0),
]


@figure
def bench_speedup():
    rows = sorted(BENCHMARKS, key=lambda r: r[2] / r[3])
    speedup = [orig / ours for _, _, orig, ours in rows]
    labels = [f"{task}  ·  {tool}" for task, tool, _, _ in rows]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.barh(range(len(rows)), speedup, color="#15803d", height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), labels)
    for bar, s in zip(bars, speedup):
        ax.text(bar.get_width() * 1.08, bar.get_y() + bar.get_height() / 2, f"{s:.1f}×" if s < 10 else f"{s:,.0f}×", va="center", fontsize=8.5,
                fontweight="bold", color="#0f172a")
    ax.set_xlim(1, max(speedup) * 3)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.set_xlabel("times faster than the original (log scale)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", which="major", color="#e2e8f0")
    ax.set_axisbelow(True)
    save(fig, "bench_speedup")


def main() -> None:
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
