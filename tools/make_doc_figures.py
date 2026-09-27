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


def palette_map(ax, values, palette: dict, labels: dict | None = None, title: str = ""):
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
        handles = [Patch(color=np.array(palette[k]) / 255, label=labels[k]) for k in keys if k in labels and k in present]
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


def main() -> None:
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
