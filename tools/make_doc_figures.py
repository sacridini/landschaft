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

    clc = ls.load_raster(ls.example_data("clc"))
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
    return ls.load_raster(ls.example_data("clc"))[:600, :600]


@figure
def lsm_patches():
    lc = ls.Landscape(clc_crop())
    fig = lc.show_lsm("shape")
    fig.set_size_inches(6, 5)
    save(fig, "lsm_shape")


@figure
def lsm_window():
    landcover = ls.load_raster(ls.example_data("clc"))[:800, :800]
    maps = ls.window_metrics(landcover, window=41, what=["lsm_l_shdi"])
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    im = ax.imshow(maps["shdi"].values, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Shannon diversity (41 x 41 window)")
    ax.set_title("Moving-window diversity")
    ax.set_axis_off()
    save(fig, "lsm_window")


@figure
def lsm_zonal():
    landcover = ls.load_raster(ls.example_data("clc"))
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
    landcover = ls.load_raster(ls.example_data("clc"))
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
    ("Transition + correction, 300×300", "gdistance", 2.0, 0.0049),
    ("Randomised shortest paths, 300×300", "gdistance", 5.98, 0.078),
    ("Route overlap, 10 locations, 300×300", "gdistance", 5.97, 0.157),
    ("Peaks and valleys (sds, ssc, s10z)", "geodiv", 1.84, 0.0017),
    ("Slopes and area (sdq, sdq6, sdr)", "geodiv", 2.16, 0.0048),
    ("Texture image of Sk, 30×30", "geodiv", 134, 0.0219),
    ("Curvature, 371×371", "spatialEco", 1.58, 0.0016),
    ("Zonation CAZ2, 2000×2000, 30 features", "Zonation 5", 48.0, 7.29),
    ("Zonation CAZ2, 1000×1000, 200 features", "Zonation 5", 33.6, 3.16),
    ("Thin 2 000 records, 10 repetitions", "spThin", 0.91, 0.0014),
    ("Autocorrelation range, 9 layers", "blockCV", 17.5, 0.73),
    ("MaxNet, 10 000 background points", "maxnet", 3.25, 0.30),
]


@figure
def bench_speedup():
    rows = sorted(BENCHMARKS, key=lambda r: r[2] / r[3])
    speedup = [orig / ours for _, _, orig, ours in rows]
    labels = [f"{task}  ·  {tool}" for task, tool, _, _ in rows]
    fig, ax = plt.subplots(figsize=(9, 5.4))
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

    landcover = ls.load_raster(ls.example_data("clc"))[:800, :800]
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


_COST: dict = {}


def cost_hiking_run():
    """The tutorial's hiking example on the Maungawhau volcano."""
    if not _COST:
        dem = ls.load_raster(ls.cost.example_data("maungawhau"))
        slope = ls.cost.geo_correction(ls.cost.transition(dem, "diff", 8, symmetric=False))
        speed = slope.map(lambda s: 6 * np.exp(-3.5 * np.abs(s + 0.05)) / 3.6, pairs="all")
        conductance = ls.cost.geo_correction(speed)
        A, B = (2667670, 6479000), (2667800, 6479400)
        _COST.update(dem=dem, conductance=conductance, A=A, B=B)
    return _COST


def _cost_extent(da):
    t = da.rio.transform()
    ny, nx = da.shape
    return (t.c, t.c + nx * t.a, t.f + ny * t.e, t.f)


@figure
def cost_hiking():
    run = cost_hiking_run()
    dem, cond, A, B = run["dem"], run["conductance"], run["A"], run["B"]
    minutes = np.asarray(ls.cost.acc_cost(cond, A)) / 60
    there = ls.cost.shortest_path(cond, A, B, output="lines").geometry.iloc[0]
    back = ls.cost.shortest_path(cond, B, A, output="lines").geometry.iloc[0]
    ext = _cost_extent(dem)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))
    im = axes[0].imshow(dem.values, cmap="terrain", extent=ext)
    axes[0].contour(dem.values, levels=np.arange(100, 200, 10), colors="k", linewidths=0.4, extent=ext,
                    origin="upper")
    fig.colorbar(im, ax=axes[0], shrink=0.7, label="elevation (m)")
    axes[0].set_title("Maungawhau")
    im = axes[1].imshow(minutes, cmap="magma_r", extent=ext)
    fig.colorbar(im, ax=axes[1], shrink=0.7, label="minutes from A")
    axes[1].set_title("walking time and routes")
    for ax in axes:
        ax.plot(*there.xy, color="#0ea5e9", lw=2, label="A to B")
        ax.plot(*back.xy, color="#f43f5e", lw=2, ls="--", label="B to A")
        for name, (x, y) in (("A", A), ("B", B)):
            ax.plot(x, y, "o", color="white", mec="black", ms=7)
            ax.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points", fontweight="bold")
        ax.set_axis_off()
    axes[1].legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    save(fig, "cost_hiking")


@figure
def cost_theta():
    run = cost_hiking_run()
    dem, cond, A, B = run["dem"], run["conductance"], run["A"], run["B"]
    ext = _cost_extent(dem)
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for ax, theta in zip(axes, (0, 0.001, 0.01, 1)):
        p = np.asarray(ls.cost.passage(cond, A, B, theta=theta))
        ax.imshow(np.log10(np.clip(p, 1e-4, None)), cmap="viridis", vmin=-4, vmax=0, extent=ext)
        ax.contour(dem.values, levels=np.arange(100, 200, 20), colors="w", linewidths=0.3, extent=ext,
                   origin="upper")
        ax.set_title("random walk (theta = 0)" if theta == 0 else f"theta = {theta:g}")
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(-4, 0))
    cb = fig.colorbar(sm, ax=axes, shrink=0.7, ticks=[-4, -3, -2, -1, 0])
    cb.ax.set_yticklabels(["0.0001", "0.001", "0.01", "0.1", "1"])
    cb.set_label("net passages")
    save(fig, "cost_theta")


@figure
def cost_europe():
    import pandas as pd

    europe = ls.load_raster(ls.cost.example_data("europe"))
    pops = pd.read_csv(ls.cost.example_data("pop_coord"))
    genetic = pd.read_csv(ls.cost.example_data("gen_dist"), index_col="population")
    walk = ls.cost.geo_correction(ls.cost.transition(europe, "mean", 16), "r", scale=True)
    xy = pops[["x", "y"]].to_numpy()
    origin = (37.0, 39.0)
    routes = ls.cost.path_inc(walk, origin, xy)
    coast = ls.cost.geo_correction(ls.cost.transition(europe, "mean", 16), "c")
    least = ls.cost.cost_distance(coast, xy)
    pairs = np.triu_indices(len(pops), 1)
    g = genetic.loc[pops["population"], pops["population"]].to_numpy()[pairs]
    passages = np.asarray(ls.cost.passage(walk, origin, xy[0]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ext = _cost_extent(europe)
    land = np.where(np.isnan(europe.values.astype(float)) | (europe.values == 0), np.nan, 1.0)
    axes[0].imshow(land, cmap=ListedColormap(["#e5e7eb"]), extent=ext)
    axes[0].imshow(np.log10(np.clip(np.where(np.isnan(land), np.nan, passages), 1e-3, None)), cmap="viridis",
                   alpha=0.85, vmin=-3, vmax=0, extent=ext)
    axes[0].plot(*xy.T, "o", color="#f97316", mec="black", ms=4)
    axes[0].plot(*origin, "*", color="#dc2626", mec="black", ms=12)
    axes[0].set_title(f"random walk from Anatolia to {pops['population'][0]}")
    axes[0].set_axis_off()
    for ax, x, name in ((axes[1], routes["overlap"][pairs], "route overlap"),
                        (axes[2], least[pairs] / 1000, "least-cost distance over land (km)")):
        r = np.corrcoef(x, g)[0, 1]
        ax.scatter(x, g, s=10, color="#0d9488", alpha=0.7)
        ax.set_xlabel(name)
        ax.set_ylabel("genetic distance")
        ax.set_title(f"r = {r:.2f}")
    fig.tight_layout()
    save(fig, "cost_europe")


def _surface(name: str):
    return ls.load_raster(ls.surface.example_data(name))


@figure
def surf_texture():
    ndvi = _surface("orforest")
    z = ls.surface.remove_plane(ndvi)
    tex = ls.surface.window_metrics(z, ["sa", "sku"], window=9)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    panels = [(ndvi.values, "NDVI", "YlGn", None), (z.values, "residuals from the trend", "RdBu", "sym"),
              (tex["sa"].values, "Sa, 9 x 9 windows", "magma", None),
              (tex["sku"].values, "Sku, 9 x 9 windows", "viridis", None)]
    for ax, (v, title, cmap, sym) in zip(axes, panels):
        if sym:
            m = np.nanpercentile(np.abs(v), 99)
            im = ax.imshow(v, cmap=cmap, vmin=-m, vmax=m)
        else:
            lo, hi = np.nanpercentile(v, [1, 99])
            im = ax.imshow(v, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    save(fig, "surf_texture")


@figure
def surf_bearing():
    z = ls.surface.remove_plane(_surface("orforest"))
    curve = ls.surface.bearing_area(z)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    v = z.values[~np.isnan(z.values)]
    axes[0].hist(v, bins=80, color="#15803d")
    axes[0].set_title("height distribution")
    axes[0].set_xlabel("NDVI residual")
    axes[0].spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(curve["bearing"] * 100, curve["height"], color="#0f172a", lw=1.5)
    h = ls.surface.height_ba(z, [0, 0.05, 0.8, 1])
    sk, spk, svk = (ls.surface.metrics(z, ["sk", "spk", "svk"])[k] for k in ("sk", "spk", "svk"))
    ax.axvspan(80, 100, color="#bfdbfe", alpha=0.6, label="valley zone (Svi)")
    ax.axhline(h[1], color="#f97316", ls="--", lw=1, label="height at 5 % (Sbi)")
    ax.set_xlabel("bearing: share of the surface at or above (%)")
    ax.set_ylabel("height")
    ax.set_title(f"bearing area curve  (Sk {sk:.3f}, Spk {spk:.3f}, Svk {svk:.3f})")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "surf_bearing")


@figure
def surf_terrain():
    dem = _surface("orelevation")
    slope = ls.surface.terrain(dem, "slope")
    tpi = ls.surface.tpi(dem, 15)
    hli = ls.surface.hli(ls.surface.terrain(dem, "aspect"), slope, scaled=True)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    panels = [(dem.values, "elevation (m)", "terrain", None), (slope.values, "slope (degrees)", "YlOrBr", None),
              (tpi.values, "TPI, 15 x 15 (m)", "RdBu_r", "sym"), (hli.values, "heat load index", "inferno", None)]
    for ax, (v, title, cmap, sym) in zip(axes, panels):
        if sym:
            m = np.nanpercentile(np.abs(v), 99)
            im = ax.imshow(v, cmap=cmap, vmin=-m, vmax=m)
        else:
            lo, hi = np.nanpercentile(v, [1, 99])
            im = ax.imshow(v, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    save(fig, "surf_terrain")


@figure
def learn_threshold():
    ndvi = _surface("orforest").values
    fig = plt.figure(figsize=(11, 3.4))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.3)
    for i, t in enumerate((0.60, 0.70, 0.75)):
        ax = fig.add_subplot(grid[0, i])
        palette_map(ax, np.where(ndvi >= t, 2, 1).astype("uint8"), BINARY, None, f"NDVI ≥ {t:.2f}")
    ax = fig.add_subplot(grid[0, 3])
    thresholds = np.round(np.arange(0.50, 0.82, 0.01), 2)
    n = [int(ls.Landscape(np.where(ndvi >= t, 2, 1).astype("uint8")).class_metrics(["np"]).set_index("class").loc[2, "np"])
         if (ndvi >= t).any() else 0 for t in thresholds]
    ax.plot(thresholds, n, color="#15803d", lw=2, marker="o", ms=3)
    ax.set_xlabel("NDVI threshold for habitat")
    ax.set_ylabel("habitat patches")
    ax.set_title("One map, many patch structures")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e2e8f0")
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.02, right=0.98)
    ax.set_position(ax.get_position().translated(0.05, 0))
    save(fig, "learn_threshold")


@figure
def learn_aacf():
    field = ls.nlm.gaussian_field(120, 120, autocorr_range=15, seed=1)
    x = np.arange(120)
    ridges = np.sin(2 * np.pi * x / 12)[None, :] + 0.3 * field.values
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for col, (name, s) in enumerate((("isotropic field", field.values), ("ridges", ridges))):
        axes[2 * col].imshow(s, cmap="viridis")
        axes[2 * col].set_title(name)
        a = ls.surface.aacf(s).values
        im = axes[2 * col + 1].imshow(a, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[2 * col + 1].set_title(f"its autocorrelation")
    for ax in axes:
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=0.7, label="autocorrelation")
    save(fig, "learn_aacf")


# ---- Spatial prioritization (tutorial and course) ---------------------------

_PRIO = {}
PRIO_NAMES = ["broadleaf_bird", "conifer_bird", "mire_plant", "wetland_bird",
              "grassland_butterfly", "farmland_bird", "edge_bat", "forest_mammal"]
# Zonation's usual rank classes: the best 2 %, 5 %, 10 %, 25 %, 50 % and the rest
RANK_BOUNDS = [0, 0.5, 0.75, 0.9, 0.95, 0.98, 1.0]
RANK_COLORS = ["#e2e8f0", "#93c5fd", "#2563eb", "#facc15", "#f97316", "#b91c1c"]
RANK_LABELS = ["rest", "best 50 %", "best 25 %", "best 10 %", "best 5 %", "best 2 %"]


def prio_share(*codes):
    import xarray as xr  # noqa: F401

    if "clc" not in _PRIO:
        _PRIO["clc"] = ls.load_raster(ls.example_data("clc"))
    clc = _PRIO["clc"]
    return clc.isin(codes).coarsen(x=5, y=5).mean().astype("float32")


def prio_species():
    """The tutorial's eight hypothetical species (docs/tutorials/prioritization.md)."""
    import xarray as xr

    if "species" not in _PRIO:
        share = prio_share
        broadleaf, conifer, mixed = share(23), share(24), share(25)
        forest = broadleaf + conifer + mixed
        _PRIO["species"] = xr.concat([
            broadleaf + 0.5 * mixed, conifer + 0.5 * mixed, share(35, 36), share(35, 36, 40, 41), share(18, 26),
            share(20, 21) + 0.3 * share(12), 4 * forest * (1 - forest), forest ** 2,
        ], dim="species").assign_coords(species=PRIO_NAMES)
    return _PRIO["species"]


def prio_run(key, **kw):
    if key not in _PRIO:
        _PRIO[key] = ls.prioritize.zonation(prio_species(), names=PRIO_NAMES, **kw)
    return _PRIO[key]


def rank_map(ax, rank, title=""):
    cmap = ListedColormap(RANK_COLORS)
    ax.imshow(np.asarray(rank), cmap=cmap, norm=BoundaryNorm(RANK_BOUNDS, len(RANK_COLORS)), interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()


def rank_legend(ax):
    handles = [Patch(color=c, label=lab) for c, lab in zip(RANK_COLORS[::-1], RANK_LABELS[::-1])]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)


def curves_plot(ax, res, marks=(0.1, 0.3)):
    fc = res.feature_curves
    x = fc["rank"].to_numpy()
    colors = plt.get_cmap("tab10").colors
    for i, name in enumerate(fc.columns[1:]):
        ax.plot(x, fc[name].to_numpy(), color=colors[i % 10], lw=1.6, label=name.replace("_", " "))
    for m in marks:
        ax.axvline(1 - m, color="#475569", lw=0.8, ls="--")
        ax.text(1 - m, 1.02, f"best {m:.0%}", ha="center", va="bottom", fontsize=8, color="#475569")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("share of the landscape removed (rank)")
    ax.set_ylabel("share of the feature remaining")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e2e8f0")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)


def best_map(ax, rank, best, title, color="#15803d"):
    v = (np.asarray(rank) >= 1 - best).astype(float)
    v[~np.isfinite(np.asarray(rank))] = np.nan
    ax.imshow(v, cmap=ListedColormap(["#f1f5f9", color]), vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()


@figure
def prio_features():
    sp = prio_species()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for ax, name in zip(axes.flat, PRIO_NAMES):
        im = ax.imshow(sp.sel(species=name).values, cmap="YlGn", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name.replace("_", " "))
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=0.6, label="habitat value (share of the cell)")
    save(fig, "prio_features")


@figure
def prio_rank():
    res = prio_run("caz2")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    rank_map(a, res.rank, "Rank (CAZ2)")
    rank_legend(a)
    curves_plot(b, res)
    b.set_title("Performance curves")
    fig.tight_layout()
    save(fig, "prio_rank")


@figure
def prio_hotspots():
    sp = prio_species()
    res = prio_run("caz2")
    richness = (sp / sp.sum(dim=("y", "x"))).sum("species")
    hot = ls.prioritize.zonation(sp, names=PRIO_NAMES, rule="LOAD", solution=richness)

    def kept(r, best):
        fc = r.feature_curves
        return fc.iloc[(fc["rank"] - (1 - best)).abs().argmin(), 1:]

    fig = plt.figure(figsize=(14, 4.2))
    g = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.55)
    best_map(fig.add_subplot(g[0, 0]), hot.rank, 0.3, "Best 30 % by richness", "#b45309")
    best_map(fig.add_subplot(g[0, 1]), res.rank, 0.3, "Best 30 % by Zonation (CAZ2)")
    ax = fig.add_subplot(g[0, 2])
    y = np.arange(len(PRIO_NAMES))
    ax.barh(y + 0.2, kept(hot, 0.3).to_numpy(), height=0.38, color="#b45309", label="richness hotspots")
    ax.barh(y - 0.2, kept(res, 0.3).to_numpy(), height=0.38, color="#15803d", label="Zonation")
    ax.set_yticks(y, [n.replace("_", " ") for n in PRIO_NAMES])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the species in the best 30 %")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "prio_hotspots")


@figure
def prio_cost_mask():
    sp = prio_species()
    towns = prio_share(*range(1, 12))
    near = ls.prioritize.convolve(towns.values, ls.prioritize.Kernel("gaussian", 8, cut=3))
    cost = towns.copy(data=(1 + 20 * near).astype("float32"))
    cheap = ls.prioritize.zonation(sp, names=PRIO_NAMES, cost=cost)
    reserves = (prio_share(36) > 0.5).astype("uint8")
    prot = ls.prioritize.zonation(sp, names=PRIO_NAMES, hierarchic_mask=reserves)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    im = axes[0].imshow(cost.values, cmap="magma_r", interpolation="nearest")
    axes[0].set_title("Cost (higher near towns)")
    axes[0].set_axis_off()
    fig.colorbar(im, ax=axes[0], shrink=0.75)
    rank_map(axes[1], cheap.rank, "Rank with cost")
    rank_map(axes[2], prot.rank, "Rank around a reserve (outlined)")
    axes[2].contour(reserves.values, levels=[0.5], colors="black", linewidths=0.8)
    rank_legend(axes[2])
    save(fig, "prio_cost_mask")


@figure
def prio_connectivity():
    res = prio_run("caz2")
    kernels = [ls.prioritize.Kernel("negexp", d) for d in (4, 4, 1, 8, 2, 4, 8, 16)]
    con = prio_run("conn", connectivity=kernels)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    best_map(axes[0], res.rank, 0.1, "Best 10 %, features as mapped")
    best_map(axes[1], con.rank, 0.1, "Best 10 %, with dispersal kernels")
    fig.tight_layout()
    save(fig, "prio_connectivity")


@figure
def learn_complementarity():
    sites = np.array([[1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, picks, title, color in ((axes[0], (0, 1), "Two richest sites: 3 species", "#b45309"),
                                    (axes[1], (0, 2), "Two complementary sites: 4 species", "#15803d")):
        ax.imshow(np.zeros_like(sites), cmap=ListedColormap(["white"]), vmin=0, vmax=1)
        for c in picks:
            ax.add_patch(plt.Rectangle((c - 0.5, -0.5), 1, 4, color=color, alpha=0.18, lw=0))
        for r in range(4):
            for c in range(4):
                if sites[r, c]:
                    covered = c in picks
                    ax.plot(c, r, "o", ms=16, color=color if covered else "#94a3b8")
        ax.set_xticks(range(4), ["A", "B", "C", "D"])
        ax.set_yticks(range(4), [f"species {i}" for i in range(1, 5)])
        ax.set_xticks(np.arange(-0.5, 4), minor=True)
        ax.set_yticks(np.arange(-0.5, 4), minor=True)
        ax.grid(which="minor", color="#cbd5e1")
        ax.tick_params(which="minor", length=0)
        ax.set_title(title)
        ax.set_xlabel("site")
    fig.tight_layout()
    save(fig, "learn_complementarity")


@figure
def learn_prio_curves():
    import xarray as xr

    share = prio_share
    names = ["forest_bird", "mire_plant", "grassland_butterfly", "farmland_bird"]
    sp = xr.concat([share(23, 24, 25), share(35, 36), share(18, 26), share(20, 21) + 0.3 * share(12)], dim="species")
    res = ls.prioritize.zonation(sp, names=names, connectivity=[ls.prioritize.Kernel("negexp", 4)] * 4)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    rank_map(a, res.rank, "Rank of four species")
    rank_legend(a)
    curves_plot(b, res, marks=(0.17, 0.3))
    b.set_title("Performance curves")
    fig.tight_layout()
    save(fig, "learn_prio_curves")


# ----------------------------------------------------------------------- species distributions

_SDM = {}
SDM_FOLD_COLORS = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#9333ea"]
SDM_TEMPS = ["bio1", "bio5", "bio6", "bio8", "bio9"]


def sdm_inputs():
    """The Bradypus records and the bioclimatic predictors."""
    if "env" not in _SDM:
        _SDM["occ"] = ls.load_table(ls.sdm.example_data("bradypus"))
        _SDM["env"] = ls.load_raster(ls.sdm.example_data("bioclim"))
    return _SDM["occ"], _SDM["env"]


def sdm_warmer(env):
    """The predictors 2 degrees warmer (temperatures in tenths of a degree)."""
    warmer = env.copy()
    warmer.loc[{"band": SDM_TEMPS}] = env.sel(band=SDM_TEMPS) + 20
    return warmer


def sdm_extent(da):
    x, y = da["x"].values, da["y"].values
    dx, dy = abs(x[1] - x[0]) / 2, abs(y[1] - y[0]) / 2
    return (x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy)


def sdm_base(ax, env, title="", window=(-95, -32, -36, 22)):
    """The land of the predictor grid in light grey, cut to South and Central America."""
    land = env.isel(band=0).notnull().values.astype(float)
    land[land == 0] = np.nan
    ax.imshow(land, extent=sdm_extent(env), cmap=ListedColormap(["#e2e8f0"]), vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(window[2], window[3])
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cbd5e1")


def sdm_raster(ax, da, env, title="", **kw):
    sdm_base(ax, env, title)
    return ax.imshow(np.asarray(da.values, dtype=float), extent=sdm_extent(da), interpolation="nearest", **kw)


def sdm_glm():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    return make_pipeline(StandardScaler(), PolynomialFeatures(2),
                         LogisticRegression(max_iter=5000))


def sdm_tutorial():
    """The tutorial's prepared data, models and projections (docs/tutorials/species-distributions.md)."""
    if "res" not in _SDM:
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            RandomForestClassifier,
        )

        occ, env = sdm_inputs()
        data = ls.sdm.prepare(occ, env, background=5000)
        careful = data.redo(thin=50_000, area=500_000, predictors=5, background=2000,
                            folds={"size": 500_000, "k": 5})
        models = {
            "glm": sdm_glm(),
            "rf": RandomForestClassifier(n_estimators=500, max_samples=0.1, min_samples_leaf=5, random_state=1),
            "brt": HistGradientBoostingClassifier(learning_rate=0.02, max_iter=200, max_depth=2,
                                                  min_samples_leaf=20, random_state=1),
        }
        res = ls.sdm.fit(careful, models)
        suit = ls.sdm.predict(res, env)
        at = ls.extract(suit.sel(band="ensemble"), careful.coords)["suitability"].to_numpy()
        t = ls.sdm.threshold(at[careful.y == 1], at[careful.y == 0])
        _SDM.update(data=data, careful=careful, res=res, suit=suit, threshold=t)
    return _SDM


@figure
def learn_sdm_bam():
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    circles = [((0.0, 0.35), "#f59e0b", "A  abiotic conditions\n(climate, soils)", (-1.55, 1.45)),
               ((0.95, 0.35), "#16a34a", "B  biotic conditions\n(resources, competitors)", (1.75, 1.45)),
               ((0.47, -0.45), "#2563eb", "M  accessible area\n(dispersal, history)", (0.47, -1.75))]
    for (cx, cy), color, label, (tx, ty) in circles:
        ax.add_patch(Circle((cx, cy), 0.95, facecolor=color, alpha=0.16, edgecolor=color, lw=2))
        ax.text(tx, ty, label, ha="center", va="center", fontsize=9, color=color, fontweight="bold")
    ax.text(0.47, 0.12, "occupied\narea", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.47, 0.95, "invadable:\nsuitable, not\nreached", ha="center", va="center", fontsize=8, color="#475569")
    ax.text(-0.42, -0.3, "reached,\nexcluded by\nbiotic factors", ha="center", va="center", fontsize=7.5,
            color="#475569")
    ax.set_xlim(-2.5, 3.0)
    ax.set_ylim(-2.1, 1.95)
    ax.set_aspect("equal")
    ax.set_axis_off()
    save(fig, "learn_sdm_bam")


def sdm_folds_map(ax, data, env, title, random=False):
    sdm_base(ax, env, title)
    xy = np.column_stack([data.coords.geometry.x, data.coords.geometry.y])
    f, y = np.asarray(data.folds), data.y
    if random:
        ax.plot(xy[y == 0, 0], xy[y == 0, 1], ".", ms=2, color="#94a3b8", label="background (always training)")
    else:
        for k in range(1, 6):
            sel = (y == 0) & (f == k)
            ax.plot(xy[sel, 0], xy[sel, 1], ".", ms=2, color=SDM_FOLD_COLORS[k - 1], alpha=0.35)
    for k in range(1, 6):
        sel = (y == 1) & (f == k)
        ax.plot(xy[sel, 0], xy[sel, 1], "o", ms=5, mfc=SDM_FOLD_COLORS[k - 1], mec="black", mew=0.5,
                label=f"fold {k}")
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.9, markerscale=1)


@figure
def learn_sdm_folds():
    occ, env = sdm_inputs()
    data = ls.sdm.prepare(occ, env, background=2000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    sdm_folds_map(axes[0], ls.sdm.random_folds(data, 5, seed=1), env, "Random folds", random=True)
    sdm_folds_map(axes[1], data, env, "Spatial blocks (presences and background)")
    fig.tight_layout()
    save(fig, "learn_sdm_folds")


@figure
def learn_sdm_mess():
    from matplotlib.colors import TwoSlopeNorm

    occ, env = sdm_inputs()
    data = ls.sdm.prepare(occ, env, background=2000)
    full = (-125.5, -31.5, -56.5, 40.5)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    for ax, clim, title in ((axes[0], env, "MESS, present climate"), (axes[1], sdm_warmer(env), "MESS, 2 °C warmer")):
        m = ls.sdm.mess(data, clim.sel(band=data.predictors)).sel(band="mess")
        sdm_base(ax, env, f"{title}: {int((m < 0).sum())} novel cells", window=full)
        im = ax.imshow(np.asarray(m.values, dtype=float), extent=sdm_extent(m), interpolation="nearest",
                       cmap="RdBu", norm=TwoSlopeNorm(0, -15, 100))
        ys, xs = np.where(m.values < 0)
        ax.plot(m.x.values[xs], m.y.values[ys], "s", ms=3.2, mfc="none", mec="#b91c1c", mew=0.8)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, label="similarity to the reference (negative, outlined: novel)")
    cb.set_ticks([-15, 0, 25, 50, 75, 100])
    save(fig, "learn_sdm_mess")


@figure
def sdm_clean():
    _, env = sdm_inputs()
    records = ls.load_table(ls.sdm.example_data("acaule"))
    records["species"] = "Solanum acaule"
    flags = ls.sdm.clean_occurrences(records, action="flag", countries={"column": "ISO2"})
    flags = flags[~flags["flag_invalid"]]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    sdm_base(ax, env, "Solanum acaule: records and flags", window=(-85, -35, -45, 5))
    ok = flags[~flags["flagged"]]
    ax.plot(ok["lon"], ok["lat"], "o", ms=3, color="#15803d", alpha=0.5, label=f"kept ({len(ok)})")
    styles = {"countries": ("#2563eb", "s"), "outliers": ("#9333ea", "D"), "capitals": ("#f59e0b", "*"),
              "seas": ("#0891b2", "v"), "zeros": ("#b91c1c", "X"), "equal": ("#b91c1c", "P")}
    for test, (color, marker) in styles.items():
        sel = flags[flags[f"flag_{test}"]]
        if len(sel):
            ax.plot(sel["lon"], sel["lat"], marker, ms=7, mfc=color, mec="black", mew=0.5, ls="none",
                    label=f"{test} ({len(sel)})")
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.9, title="flag (records)",
              title_fontsize=8)
    ax.text(-84, 3.5, "zeros at 0°/0° and the record in Antarctica\nlie outside the map", fontsize=7.5,
            color="#475569", va="top")
    save(fig, "sdm_clean")


@figure
def sdm_prepare():
    occ, env = sdm_inputs()
    occ1 = ls.sdm.one_per_cell(ls.sdm.clean_occurrences(occ), env)
    occ2 = ls.sdm.thin(occ1, 50_000, seed=1)
    area = ls.sdm.accessible_area(occ2, distance=500_000)
    env2 = ls.sdm.select_predictors(env, vif=5, area=area, seed=1)
    bg = ls.sdm.background(env2, 2000, occ=occ2, area=area, seed=1)
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    sdm_base(ax, env, "Accessible area, background and records")
    area.boundary.plot(ax=ax, color="#2563eb", lw=1.2)
    ax.plot(bg.geometry.x, bg.geometry.y, ".", ms=2, color="#64748b", label="background (2000)")
    gone = occ1.loc[~occ1.index.isin(occ2.index)]
    ax.plot(gone["lon"], gone["lat"], "o", ms=5, mfc="white", mec="#b91c1c", label=f"thinned out ({len(gone)})")
    ax.plot(occ2["lon"], occ2["lat"], "o", ms=5, mfc="#15803d", mec="black", mew=0.5,
            label=f"presences ({len(occ2)})")
    ax.plot([], [], color="#2563eb", label="500 km around the records")
    ax.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)
    save(fig, "sdm_prepare")


@figure
def sdm_folds():
    _, env = sdm_inputs()
    careful = sdm_tutorial()["careful"]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    sdm_folds_map(ax, careful, env, "Five folds of 500 km hexagons")
    save(fig, "sdm_folds")


@figure
def sdm_ensemble():
    _, env = sdm_inputs()
    s = sdm_tutorial()
    suit, careful = s["suit"], s["careful"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    im = sdm_raster(axes[0], suit.sel(band="ensemble"), env, "Ensemble suitability", cmap="viridis", vmin=0, vmax=1)
    pres = careful.coords[careful.y == 1]
    axes[0].plot(pres.geometry.x, pres.geometry.y, "o", ms=3, mfc="white", mec="black", mew=0.4)
    fig.colorbar(im, ax=axes[0], shrink=0.7)
    im = sdm_raster(axes[1], suit.sel(band="sd"), env, "Disagreement between models (sd)", cmap="magma_r",
                    vmin=0, vmax=0.4)
    fig.colorbar(im, ax=axes[1], shrink=0.7)
    fig.tight_layout()
    save(fig, "sdm_ensemble")


@figure
def sdm_response():
    res = sdm_tutorial()["res"]
    curves = ls.sdm.response_curves(res)
    labels = {"bio5": "bio5: max temperature of the warmest month (°C × 10)",
              "bio7": "bio7: annual range of temperature (°C × 10)",
              "bio16": "bio16: rain of the wettest quarter (mm)",
              "bio17": "bio17: rain of the driest quarter (mm)"}
    colors = {"glm": "#2563eb", "rf": "#16a34a", "brt": "#f97316"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for ax, p in zip(axes, res.predictors):
        for model, g in curves[curves["predictor"] == p].groupby("model"):
            ax.plot(g["value"], g["suitability"], color=colors[model], lw=1.8, label=model)
        ax.set_xlabel(labels[p], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e2e8f0")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("suitability")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "sdm_response")


@figure
def sdm_threshold():
    _, env = sdm_inputs()
    s = sdm_tutorial()
    ens = s["suit"].sel(band="ensemble")
    v = (ens >= s["threshold"]).astype(float).where(ens.notnull())
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    sdm_raster(ax, v, env, f"Suitable at the maximum-TSS threshold ({s['threshold']:.2f})",
               cmap=ListedColormap(["#f1f5f9", "#15803d"]), vmin=0, vmax=1)
    ax.legend(handles=[Patch(color="#15803d", label="suitable"), Patch(color="#f1f5f9", label="unsuitable")],
              loc="lower left", fontsize=8, frameon=True, framealpha=0.9)
    save(fig, "sdm_threshold")


@figure
def sdm_future():
    _, env = sdm_inputs()
    s = sdm_tutorial()
    res, careful, t = s["res"], s["careful"], s["threshold"]
    warmer = sdm_warmer(env)
    ens = s["suit"].sel(band="ensemble")
    today = (ens >= t).values
    later = (ls.sdm.predict(res, warmer).sel(band="ensemble") >= t).values
    change = np.where(ens.notnull().values, 1.0, np.nan)
    change[today & later] = 2
    change[today & ~later] = 3
    change[~today & later] = 4
    mess = ls.sdm.mess(careful, warmer.sel(band=careful.predictors)).sel(band="mess")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    sdm_base(axes[0], env, "Suitable today and 2 °C warmer")
    axes[0].imshow(change, extent=sdm_extent(ens), interpolation="nearest",
                   cmap=ListedColormap(["#f1f5f9", "#15803d", "#dc2626", "#2563eb"]), vmin=0.5, vmax=4.5)
    axes[0].legend(handles=[Patch(color="#15803d", label="suitable in both"), Patch(color="#dc2626", label="lost"),
                            Patch(color="#2563eb", label="gained")], loc="lower left", fontsize=8, frameon=True,
                   framealpha=0.9)
    from matplotlib.colors import TwoSlopeNorm

    im = sdm_raster(axes[1], mess, env, "MESS of the warmer climate", cmap="RdBu", norm=TwoSlopeNorm(0, -15, 100))
    if careful.area is not None:
        careful.area.boundary.plot(ax=axes[1], color="black", lw=0.8)
    cb = fig.colorbar(im, ax=axes[1], shrink=0.7, label="similarity (negative: novel); outline: accessible area")
    cb.set_ticks([-15, 0, 25, 50, 75, 100])
    fig.tight_layout()
    save(fig, "sdm_future")


def main() -> None:
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
