# SPDX-License-Identifier: GPL-3.0-or-later
"""Maps of patches, core areas and patch metrics, and correlation heatmaps,
like landscapemetrics' ``show_patches``, ``show_cores``, ``show_lsm`` and
``show_correlation`` (matplotlib needed)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = ["show_correlation", "show_cores", "show_lsm", "show_patches"]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("the plots need matplotlib (pip install matplotlib)") from err
    return plt


def _panels(n: int, ncols: int | None, nrows: int | None, size: float = 4.0):
    plt = _plt()
    if ncols is None and nrows is None:
        ncols = math.ceil(math.sqrt(n))
    if ncols is None:
        ncols = math.ceil(n / nrows)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]


def _as_landscape(landscape, options: dict):
    from .landscape import Landscape

    return landscape if isinstance(landscape, Landscape) else Landscape(landscape, **options)


def _which_classes(landscape, classes):
    """"global": one panel; "all": one per class; else the given classes."""
    if isinstance(classes, str):
        if classes == "global":
            return [None]
        if classes == "all":
            return [int(c) for c in landscape.classes]
        raise ValueError("classes must be 'global', 'all' or class values")
    wanted = [int(c) for c in np.atleast_1d(classes)]
    missing = set(wanted) - set(landscape.classes.tolist())
    if missing:
        raise ValueError(f"class(es) not present in the landscape: {sorted(missing)}")
    return wanted


def _extent(landscape):
    xmin, xmax, ymin, ymax = landscape.extent
    return (xmin, xmax, ymin, ymax)


def _cell_class(landscape) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(landscape.labels())
    pclass = np.concatenate([[np.iinfo(np.int32).min], landscape.patch_classes])
    return labels, pclass[labels]


def _label_patches(ax, landscape, ids, texts):
    c = landscape.centroids()
    c = c.set_index("id")
    for i, t in zip(ids, texts):
        ax.annotate(t, (c.loc[i, "x"], c.loc[i, "y"]), ha="center", va="center", fontsize=7)


def show_patches(landscape, classes="global", *, labels: bool = False, cmap="tab20", ncols=None, nrows=None,
                 **options):
    """Map of the patches, each in its own colour (landscapemetrics'
    ``show_patches``).

    Parameters
    ----------
    landscape
        A :class:`~landschaft.Landscape`, or a raster (then ``**options`` are
        the :class:`~landschaft.Landscape` options, e.g. ``directions``).
    classes
        "global" (all patches on one map), "all" (one map per class) or class
        value(s).
    labels
        Write the patch ids at the patch centroids.
    cmap, ncols, nrows
        Colour map (cycled over patch ids) and panel layout.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"))[:400, :400]
    >>> fig = ls.show_patches(landcover)
    >>> fig = ls.show_patches(landcover, classes=[12, 23], labels=True)
    """
    plt = _plt()
    landscape = _as_landscape(landscape, options)
    wanted = _which_classes(landscape, classes)
    lab, cell_class = _cell_class(landscape)
    fig, axes = _panels(len(wanted), ncols, nrows)
    colours = plt.get_cmap(cmap)
    n = getattr(colours, "N", 256)
    for ax, c in zip(axes, wanted):
        keep = lab > 0 if c is None else (lab > 0) & (cell_class == c)
        img = np.where(keep, (lab - 1) % n, np.nan)
        ax.imshow(img, cmap=colours, vmin=0, vmax=n - 1, extent=_extent(landscape), interpolation="nearest")
        ax.set_title("all classes" if c is None else f"class {c}")
        if labels:
            ids = np.unique(lab[keep])
            _label_patches(ax, landscape, ids, [str(i) for i in ids])
    return fig


def show_cores(landscape, classes="all", *, edge_depth: int = 1, consider_boundary: bool = True,
               labels: bool = False, ncols=None, nrows=None, **options):
    """Core and edge cells of the patches (landscapemetrics' ``show_cores``):
        core cells dark, edge cells light, other classes blank.

        ``edge_depth`` and ``consider_boundary`` as for the core-area metrics
        (landscapemetrics' plot defaults: ``consider_boundary=True``);
        ``labels`` writes the core area (ha) of each patch.

        Parameters
        ----------
        landscape
            A :class:`Landscape`, or a raster (``**options`` are then the
            :class:`Landscape` options).
        classes
            "all" (one panel per class, default), "global" or class value(s).
        edge_depth, consider_boundary
            As for the core-area metrics.
        labels
            Write the core area (ha) of each patch.
        ncols, nrows
            Panel layout.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> import landschaft as ls
        >>> landcover = ls.load_raster(ls.example_data("clc"))[:400, :400]
        >>> fig = ls.show_cores(landcover, classes=[23], edge_depth=2)
    """
    plt = _plt()
    landscape = _as_landscape(landscape, options)
    wanted = _which_classes(landscape, classes)
    b = np.asarray(landscape.boundaries(consider_boundary=consider_boundary, edge_depth=edge_depth))
    lab, cell_class = _cell_class(landscape)
    fig, axes = _panels(len(wanted), ncols, nrows)
    cmap = plt.get_cmap("Greens")
    core = _core_areas(landscape, edge_depth, consider_boundary) if labels else None
    for ax, c in zip(axes, wanted):
        keep = lab > 0 if c is None else (lab > 0) & (cell_class == c)
        img = np.where(keep, np.where(b == 1, 0.35, 0.9), np.nan)
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, extent=_extent(landscape), interpolation="nearest")
        ax.set_title("all classes" if c is None else f"class {c}")
        if labels:
            ids = np.unique(lab[keep])
            _label_patches(ax, landscape, ids, [f"{core[i - 1]:.3g}" for i in ids])
    return fig


def _core_areas(landscape, edge_depth: int, consider_boundary: bool) -> np.ndarray:
    b = np.asarray(landscape.boundaries(consider_boundary=consider_boundary, edge_depth=edge_depth))
    lab = np.asarray(landscape.labels())
    cells = np.bincount(lab[b == 0], minlength=landscape.n_patches + 1)[1:]
    return cells * landscape.resolution[0] * landscape.resolution[1] / 10000


def show_lsm(landscape, metric: str, classes="global", *, labels: bool = False, cmap="viridis", ncols=None,
             nrows=None, **options):
    """A patch-level metric painted on the patches (landscapemetrics'
        ``show_lsm``), with a colour bar; ``labels`` writes the values.

        Parameters
        ----------
        landscape
            A :class:`Landscape`, or a raster (``**options`` are then the
            :class:`Landscape` options).
        metric
            A patch-level metric.
        classes
            "global" (one map, default), "all" or class value(s).
        labels
            Write the values on the patches.
        cmap, ncols, nrows
            Colour map and panel layout.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> import landschaft as ls
        >>> landcover = ls.load_raster(ls.example_data("clc"))[:400, :400]
        >>> fig = ls.show_lsm(landcover, "shape")
    """
    _plt()
    landscape = _as_landscape(landscape, options)
    wanted = _which_classes(landscape, classes)
    values = np.asarray(landscape.to_raster(metric, level="patch"), dtype=float)
    lab, cell_class = _cell_class(landscape)
    fig, axes = _panels(len(wanted), ncols, nrows)
    finite = values[np.isfinite(values)]
    vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
    patch_values = landscape.patch_metrics(metric).iloc[:, 2].to_numpy()
    for ax, c in zip(axes, wanted):
        keep = lab > 0 if c is None else (lab > 0) & (cell_class == c)
        im = ax.imshow(np.where(keep, values, np.nan), cmap=cmap, vmin=vmin, vmax=vmax, extent=_extent(landscape),
                       interpolation="nearest")
        ax.set_title(f"{metric}" if c is None else f"{metric}, class {c}")
        if labels:
            ids = np.unique(lab[keep])
            _label_patches(ax, landscape, ids, [f"{patch_values[i - 1]:.3g}" for i in ids])
        fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def show_correlation(data, method: str = "pearson", *, diag: bool = True, labels: bool = False, cmap="RdBu_r"):
    """Heatmap of the correlations between metrics (landscapemetrics'
    ``show_correlation``), one panel per level.

    Parameters
    ----------
    data
        A long table of metrics (correlated with
        :func:`landschaft.calculate_correlation` and ``method``) or its
        result (a dict of tables, or one table).
    labels
        Write the correlation values in the cells.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import landschaft as ls
    >>> table = ls.landscape_metrics(ls.example_data("clc"), what=["lsm_p_area", "lsm_p_perim", "lsm_p_shape"])
    >>> fig = ls.show_correlation(table, method="spearman", labels=True)
    """
    from .utils import calculate_correlation

    if isinstance(data, dict):
        tables = data
    elif isinstance(data, pd.DataFrame) and {"metric_1", "metric_2", "value"} <= set(data.columns):
        tables = {"": data}
    else:
        tables = calculate_correlation(data, method=method, diag=diag)
    fig, axes = _panels(len(tables), None, 1, size=5.0)
    for ax, (level, table) in zip(axes, tables.items()):
        names = list(dict.fromkeys(list(table["metric_2"]) + list(table["metric_1"])))
        names = sorted(names)
        pos = {m: i for i, m in enumerate(names)}
        grid = np.full((len(names), len(names)), np.nan)
        for m1, m2, v in zip(table["metric_1"], table["metric_2"], table["value"]):
            grid[pos[m1], pos[m2]] = v
        im = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=90)
        ax.set_yticks(range(len(names)), names)
        ax.set_title(level)
        if labels:
            for (i, j), v in np.ndenumerate(grid):
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    return fig
