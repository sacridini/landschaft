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


# ---- Learn (the course) ---------------------------------------------------


def coarsen_habitat(habitat, k):
    """Majority aggregation of k x k blocks of a 0/1/2 habitat map (as in the Scale chapter)."""
    n_rows, n_cols = habitat.shape[0] // k, habitat.shape[1] // k
    blocks = habitat[: n_rows * k, : n_cols * k].reshape(n_rows, k, n_cols, k)
    return np.where((blocks == 2).mean(axis=(1, 3)) >= 0.5, 2, 1).astype("uint8")


@figure
def learn_grain():
    forest = ls.load_raster(ls.example_data("input")).values
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    for ax, k in zip(axes, (1, 4, 16)):
        coarse = coarsen_habitat(forest, k)
        n = ls.Landscape(coarse, res=25 * k).class_metrics(["np"]).set_index("class").loc[2, "np"]
        palette_map(ax, coarse, BINARY, None, f"{25 * k} m cells: {n:,.0f} forest patches")
    save(fig, "learn_grain")


@figure
def learn_percolation():
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(11, 3.4))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.7], wspace=0.35)
    for i, p in enumerate((0.3, 0.5, 0.7)):
        ax = fig.add_subplot(grid[0, i])
        random_map = np.where(rng.random((80, 80)) < p, 2, 1).astype("uint8")
        palette_map(ax, random_map, BINARY, None, f"{p:.0%} habitat")
    ax = fig.add_subplot(grid[0, 3])
    amounts = np.linspace(0.05, 0.95, 37)
    colours = {4: "#0f172a", 8: "#15803d"}
    for directions in (4, 8):
        lpi = []
        for p in amounts:
            random_map = np.where(rng.random((300, 300)) < p, 2, 1).astype("uint8")
            table = ls.Landscape(random_map, directions=directions).class_metrics(["lpi"]).set_index("class")
            lpi.append(table.loc[2, "lpi"])
        ax.plot(100 * amounts, lpi, color=colours[directions], lw=2, marker="o", ms=3.5,
                label=f"{directions} neighbours")
        p_c = {4: 59.3, 8: 40.7}[directions]
        ax.axvline(p_c, color=colours[directions], lw=1, ls=":")
        ax.text(p_c - 1.5, 20 if directions == 8 else 8, f"$p_c$ ≈ {p_c:.0f}%", color=colours[directions], fontsize=8, ha="right")
    ax.set_xlabel("habitat in the map (%)")
    ax.set_ylabel("largest patch (% of the map)")
    ax.set_title("Largest patch index of random maps")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e2e8f0")
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.02, right=0.98)
    ax.set_position(ax.get_position().translated(0.04, 0))
    save(fig, "learn_percolation")


def mspa_toy():
    """The drawn landscape of the Learn chapter on the shape of habitat."""
    toy = np.ones((24, 40), dtype="uint8")
    toy[3:15, 2:14] = 2
    toy[5:19, 24:37] = 2
    toy[8:10, 14:24] = 2
    toy[9:12, 29:32] = 1
    toy[7:17, 38] = 2
    toy[7, 37] = toy[16, 37] = 2
    toy[14:22, 9] = 2
    toy[19:22, 4:6] = 2
    toy[1:3, 20:34] = 2
    return toy


@figure
def learn_mspa_toy():
    toy = mspa_toy()
    morph = ls.mspa(toy, edge_width=1, internal=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    palette_map(axes[0], toy, BINARY, None, "Habitat map")
    for text, (row, col) in {"A": (8.5, 7.5), "B": (15.5, 30)}.items():
        axes[0].text(col, row, text, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    names = {code: name for code, name in ls.MSPA_CLASSES.items() if code in (0, 1, 3, 5, 9, 17, 33, 65)}
    palette_map(axes[1], morph, ls.mspa_colormap(), names, "MSPA classes, edge_width=1")
    save(fig, "learn_mspa_toy")


@figure
def learn_cost_distance():
    from matplotlib.colors import LogNorm
    from scipy import ndimage

    landcover = ls.load_raster(ls.example_data("clc"), validate=None)[:800, :800]
    table = {}
    table.update({c: 50.0 for c in range(1, 12)})
    table.update({c: 10.0 for c in range(12, 23)})
    table.update({c: 1.0 for c in range(23, 35)})
    table.update({c: 100.0 for c in range(35, 45)})
    resistance = np.vectorize(table.get)(landcover.values).astype(float)
    forest = np.isin(landcover.values, [23, 24, 25])
    euclidean_km = ndimage.distance_transform_edt(~forest) * 25 / 1000
    cost_km = np.asarray(ls.cost_distance(resistance, forest)) * 25 / 1000
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    panels = [(euclidean_km, "Straight-line distance to forest (km)", None),
              (cost_km, "Cost distance to forest (km-equivalents, log scale)", LogNorm(0.1, cost_km.max()))]
    for ax, (values, title, norm) in zip(axes, panels):
        shown = np.ma.masked_where(forest, values)
        image = ax.imshow(shown, cmap="Blues", norm=norm, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~forest, forest), cmap=ListedColormap(["#15803d"]), interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    save(fig, "learn_cost_distance")


@figure
def learn_circuit_gaps():
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    from landschaft import circuit

    resistance = np.ones((60, 100))
    resistance[:, 48:52] = np.nan
    resistance[8:11, 48:52] = 1.0
    resistance[34:52, 48:52] = 1.0
    points = np.zeros(resistance.shape)
    points[25:35, 5] = 1
    points[25:35, 94] = 2
    current = np.asarray(circuit.circuitscape_pairwise(resistance, points, cum_map=True).cum_current)
    path = np.asarray(ls.least_cost_path(np.where(np.isnan(resistance), 0, resistance),
                                         points == 1, points == 2).lcp_mask)
    omni = ls.omniscape(resistance, radius=15, block_size=1, flow_potential=True, normalized_current=True)
    normalised = np.asarray(omni["normalized_cum_currmap"])

    wall = np.ma.masked_where(~np.isnan(resistance), np.ones(resistance.shape))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.imshow(np.where(np.isnan(resistance), np.nan, 1.0), cmap=ListedColormap(["#eee8d5"]), interpolation="nearest")
    ax.imshow(np.ma.masked_where(~path.astype(bool), path), cmap=ListedColormap(["#15803d"]), interpolation="nearest")
    ax.imshow(np.ma.masked_where(points == 0, points), cmap=ListedColormap(["#0f172a"]), interpolation="nearest")
    ax.set_title("Least-cost path")
    shown = np.where(current > 0, current, np.nan)
    image = axes[1].imshow(shown, cmap="Blues", norm=LogNorm(1e-4, np.nanmax(current)), interpolation="nearest")
    axes[1].set_title("Pairwise current (log scale)")
    fig.colorbar(image, ax=axes[1], fraction=0.035, pad=0.02)
    image = axes[2].imshow(normalised, cmap="RdBu_r", norm=TwoSlopeNorm(1.0, 0.0, 4.0), interpolation="nearest")
    axes[2].set_title("Normalised current (Omniscape)")
    fig.colorbar(image, ax=axes[2], fraction=0.035, pad=0.02)
    for ax in axes:
        ax.imshow(wall, cmap=ListedColormap(["#0f172a"]), interpolation="nearest")
        ax.set_axis_off()
    save(fig, "learn_circuit_gaps")


@figure
def learn_processes():
    def initial():
        m = np.ones((100, 100), dtype="uint8")
        m[20:80, 20:80] = 2
        m[5:12, 85:92] = 2
        m[88:94, 6:12] = 2
        return m

    scenarios = {"initial": initial()}
    m = initial(); m[35:40, 35:40] = 1; m[55:62, 60:66] = 1
    scenarios["perforation"] = m
    m = initial(); m[20:80, 49:51] = 1
    scenarios["dissection"] = m
    m = initial(); m[20:80, 39:41] = 1; m[20:80, 59:61] = 1; m[39:41, 20:80] = 1; m[59:61, 20:80] = 1
    scenarios["fragmentation"] = m
    m = initial(); m[20:80, 20:80] = 1; m[24:76, 24:76] = 2
    scenarios["shrinkage"] = m
    m = initial(); m[5:12, 85:92] = 1; m[88:94, 6:12] = 1
    scenarios["attrition"] = m
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.5))
    for ax, (name, m) in zip(axes, scenarios.items()):
        palette_map(ax, m, BINARY, None, name)
    save(fig, "learn_processes")


def nlm_models():
    return {
        "random": ls.nlm.random(200, 200, seed=1),
        "planar gradient": ls.nlm.planar_gradient(200, 200, direction=60),
        "edge gradient": ls.nlm.edge_gradient(200, 200, direction=60),
        "distance gradient": ls.nlm.distance_gradient(200, 200, origin=(90, 110, 90, 110), nlmr_compat=False),
        "midpoint displacement": ls.nlm.mpd(200, 200, roughness=0.6, seed=2, nlmr_compat=False),
        "fractional Brownian motion": ls.nlm.fbm(200, 200, fract_dim=1.4, seed=3),
        "Gaussian field": ls.nlm.gaussian_field(200, 200, autocorr_range=20, seed=4),
        "random clusters": ls.nlm.random_cluster(200, 200, p=0.55, ai=[0.3, 0.3, 0.4], seed=5),
        "neighbourhood clusters": ls.nlm.neigh(200, 200, p_neigh=0.95, p_empty=0.002, categories=4, seed=6),
        "rectangular clusters": ls.nlm.random_rectangular_cluster(200, 200, minl=5, maxl=30, seed=7),
        "Voronoi mosaic": ls.nlm.mosaic_tess(200, 200, germs=60, seed=8),
        "Poisson line mosaic": ls.nlm.mosaic_field(200, 200, n=20, seed=9),
    }


@figure
def nlm_gallery():
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.6))
    for ax, (name, m) in zip(axes.ravel(), nlm_models().items()):
        ax.imshow(m.values, cmap="viridis", interpolation="nearest")
        ax.set_title(name)
        ax.set_axis_off()
    save(fig, "nlm_gallery")


@figure
def nlm_binarize():
    surface = ls.nlm.fbm(200, 200, fract_dim=1.6, seed=3)
    maps = ls.nlm.binarize(surface, [0.2, 0.4, 0.6])
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    axes[0].imshow(surface.values, cmap="viridis", interpolation="nearest")
    axes[0].set_title("fractal surface")
    axes[0].set_axis_off()
    for ax, b in zip(axes[1:], (0.2, 0.4, 0.6)):
        palette_map(ax, maps.sel(p=b).values, BINARY, None, f"{int(b * 100)} % habitat")
    save(fig, "nlm_binarize")


@figure
def nlm_null():
    forest = ls.load_raster(ls.example_data("input")).values[:250, :250]
    habitat = np.where(forest == 2, 2, 1)
    share = float((habitat == 2).mean())

    def aggregation(m):
        table = ls.landscape_metrics(np.asarray(m), level="class", metric="ai")
        return float(table.loc[table["class"] == 2, "value"].iloc[0])

    observed = aggregation(habitat)
    random_ai = [aggregation(ls.nlm.binarize(ls.nlm.random(250, 250, seed=s), share)) for s in range(1, 100)]
    fractal_ai = [aggregation(ls.nlm.binarize(ls.nlm.fbm(250, 250, fract_dim=1.5, seed=s), share))
                  for s in range(1, 100)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bins = np.linspace(min(random_ai + fractal_ai + [observed]) - 1, max(random_ai + fractal_ai + [observed]) + 1, 60)
    ax.hist(random_ai, bins=bins, color="#94a3b8", label="99 random landscapes")
    ax.hist(fractal_ai, bins=bins, color="#0d9488", alpha=0.8, label="99 fractal landscapes (fbm, 1.5)")
    ax.axvline(observed, color="#b91c1c", lw=2, label="example forest")
    ax.set_xlabel("aggregation index of the habitat (ai, %)")
    ax.set_ylabel("landscapes")
    ax.legend(frameon=False)
    save(fig, "nlm_null")


_BIO: dict = {}


def bio_run():
    """The tutorial's analysis of Biodiverse's example data."""
    if not _BIO:
        bd = ls.bio.BaseData.from_table(ls.bio.example_data("sites"), group_columns=["x", "y"],
                                        label_columns=["genus", "species"], cell_sizes=[100000, 100000])
        tree = ls.bio.read_tree(ls.bio.example_data("tree"))
        res = ls.bio.spatial_analysis(bd, "sp_circle (radius => 200000)",
                                      ["richness", "endemism_whole", "endemism_whole_lists", "pd", "pe", "phylo_rpe2"],
                                      tree=tree)
        rnd = ls.bio.randomise(res, "rand_structured", iterations=999, seed=2024)
        _BIO.update(bd=bd, tree=tree, res=res, rnd=rnd)
    return _BIO


def bio_map(ax, grid, title: str, **kw):
    x, y = grid["x"].values, grid["y"].values
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    extent = [(x.min() - dx / 2) / 1000, (x.max() + dx / 2) / 1000, (y.min() - dy / 2) / 1000,
              (y.max() + dy / 2) / 1000]
    im = ax.imshow(grid.values, extent=extent, interpolation="nearest", **kw)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    return im


@figure
def bio_indices():
    run = bio_run()
    res = run["res"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (index, title) in zip(axes, [("RICHNESS_ALL", "species richness"), ("ENDW_WE", "weighted endemism"),
                                         ("PD", "phylogenetic diversity"), ("PE_WE", "phylogenetic endemism")]):
        im = bio_map(ax, res.to_raster(index), title, cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.75)
    save(fig, "bio_indices")


@figure
def bio_canape():
    run = bio_run()
    can = run["rnd"].canape()
    grid = run["bd"].to_raster(can["CANAPE_CODE"])
    palette = {0: (203, 213, 225, 255), 1: (220, 38, 38, 255), 2: (37, 99, 235, 255), 3: (168, 85, 247, 255),
               4: (88, 28, 135, 255)}
    labels = {0: "not significant", 1: "neo-endemism", 2: "palaeo-endemism", 3: "mixed endemism",
              4: "super-endemism"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    keys = sorted(palette)
    cmap = ListedColormap([np.array(palette[k]) / 255 for k in keys])
    bio_map(axes[0], grid, "CANAPE (999 randomisations)", cmap=cmap,
            norm=BoundaryNorm(np.arange(len(keys) + 1) - 0.5, len(keys)))
    present = set(can["CANAPE_CODE"].dropna().astype(int))
    axes[0].legend(handles=[Patch(color=np.array(palette[k]) / 255, label=labels[k]) for k in keys if k in present],
                   loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    pr = run["rnd"].p_rank()["PE_WE"]
    im = bio_map(axes[1], run["bd"].to_raster(pr), "p-rank of PE", cmap="RdBu_r", vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[1], shrink=0.75)
    fig.tight_layout()
    save(fig, "bio_canape")


@figure
def bio_clusters():
    run = bio_run()
    cl = ls.bio.cluster(run["bd"], "SORENSON", "link_average", prng_seed=1)
    groups = cl.cut(5)
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = ListedColormap(["#0d9488", "#f59e0b", "#6366f1", "#e11d48", "#65a30d"])
    bio_map(ax, run["bd"].to_raster(groups), "five Sorenson clusters", cmap=cmap, vmin=0.5, vmax=5.5)
    save(fig, "bio_clusters")


def main() -> None:
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
