# SPDX-License-Identifier: GPL-3.0-or-later
"""Colour palettes (without matplotlib) and the styles of landschaft's products."""

from __future__ import annotations

import numpy as np

# Anchor colours of the continuous palettes, interpolated to 256 (the matplotlib /
# ColorBrewer palettes of the same names, sampled).
_ANCHORS: dict[str, list[str]] = {
    "viridis": ["#440154", "#482878", "#3e4a89", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b",
                "#fde725"],
    "magma": ["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55964", "#fb8761", "#fec287", "#fcfdbf"],
    "inferno": ["#000004", "#1f0c48", "#550f6d", "#88226a", "#ba3655", "#e35933", "#f98e09", "#f9cb35", "#fcffa4"],
    "rdylgn": ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#ffffbf", "#d9ef8b", "#a6d96a", "#66bd63",
               "#1a9850", "#006837"],
    "rdbu": ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3",
             "#2166ac", "#053061"],
    "blues": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
    "greens": ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b"],
    "ylorrd": ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"],
    "terrain": ["#333399", "#0d81e1", "#05c4a2", "#5cd96e", "#bbe983", "#e3dc88", "#a99064", "#8a6d5d", "#c6b6b2",
                "#ffffff"],
    # Zonation's rank maps: the lowest priority dark, the top fraction red
    "rank": ["#000000", "#2b1a5b", "#2f4b9a", "#2a85b4", "#4fb9a4", "#a6d96a", "#fee08b", "#fdae61", "#f46d43",
             "#d73027", "#a50026"],
    # current densities (Circuitscape, Omniscape): dark to bright, on a log scale
    "current": ["#000000", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06", "#f7d13d",
                "#fcffa4"],
}

#: Distinct colours for classes without a palette of their own (tab20, then golden-angle hues).
CATEGORICAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
               "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
               "#dbdb8d", "#9edae5"]


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(v) for v in rgb[:3]))


def is_number(dtype) -> bool:
    """Numbers, not booleans, of a pandas or NumPy dtype (pandas' own string and nullable
    dtypes included, which ``np.issubdtype`` does not take)."""
    from pandas.api.types import is_bool_dtype, is_numeric_dtype

    return is_numeric_dtype(dtype) and not is_bool_dtype(dtype)


def lut(name, n: int = 256) -> np.ndarray:
    """(n, 3) uint8 colours of a palette: a built-in name (``"viridis"``...), a
    matplotlib colormap name (``"cividis"``, ``"RdBu_r"``...), or a list of colours."""
    reverse = isinstance(name, str) and name.endswith("_r")
    key = name[:-2] if reverse else name
    if isinstance(key, str) and key.lower() in _ANCHORS:
        anchors = np.array([hex_to_rgb(c) for c in _ANCHORS[key.lower()]], dtype=np.float64)
    elif isinstance(name, (list, tuple)):
        anchors = np.array([hex_to_rgb(c) if isinstance(c, str) else c[:3] for c in name], dtype=np.float64)
        reverse = False
    else:
        import matplotlib

        cmap = matplotlib.colormaps[name]
        return (np.asarray(cmap(np.linspace(0, 1, n)))[:, :3] * 255 + 0.5).astype(np.uint8)
    if reverse:
        anchors = anchors[::-1]
    t = np.linspace(0, 1, len(anchors))
    s = np.linspace(0, 1, n)
    out = np.stack([np.interp(s, t, anchors[:, k]) for k in range(3)], axis=1)
    return (out + 0.5).astype(np.uint8)


def distinct(n: int) -> list[str]:
    """n distinct colours."""
    if n <= len(CATEGORICAL):
        return CATEGORICAL[:n]
    import colorsys

    extra = []
    for i in range(n - len(CATEGORICAL)):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.9)
        extra.append(rgb_to_hex((r * 255, g * 255, b * 255)))
    return CATEGORICAL + extra


#: The order of a product's classes in its legend (GuidosToolbox's); others: by value.
LEGEND_ORDER = {
    "MSPA": ["Core", "Islet", "Perforation", "Edge", "Loop", "Bridge", "Branch", "Background", "Core-Opening",
             "Border-Opening"],
}


def product_style(attrs: dict) -> tuple[dict | None, dict | None]:
    """The colour table {value: (r, g, b, a)} and the class names {value: name} of a
    landschaft product (from its ``landschaft_product`` attribute), or (None, None)."""
    product = attrs.get("landschaft_product")
    if not product:
        return None, None
    from .. import io

    cmap = io._resolve_colormap("auto", type("A", (), {"attrs": attrs})())
    return cmap, class_names(product, attrs)


def class_names(product: str, attrs: dict) -> dict | None:
    """The names of the classes of a product."""
    if product == "MSPA":
        from ..mspa import MSPA_CLASSES

        return MSPA_CLASSES
    if product in ("FAD", "FAD_MULTISCALE", "FOS"):
        from ..fad import FAD_CLASSES

        return FAD_CLASSES
    if product == "LANDSCAPE_MOSAIC":
        from ..mosaic import LM19_CLASSES

        return LM19_CLASSES
    if product in ("SPA2", "SPA3", "SPA5", "SPA6", "SPA"):
        from ..spa import SPA_CLASSES

        return SPA_CLASSES
    if product in ("INFLUENCE_ZONES", "PROXIMITY"):
        from ..influence import INFLUENCE_CLASSES

        return INFLUENCE_CLASSES
    if product.startswith("ISOCHRONE"):
        from ..isochrone import ISOCHRONE_CLASSES

        return ISOCHRONE_CLASSES
    if product == "ACCOUNTING":
        from ..accounting import ACCOUNTING_CLASSES

        return ACCOUNTING_CLASSES
    return None
