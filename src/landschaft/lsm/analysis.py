# SPDX-License-Identifier: GPL-3.0-or-later
"""Analyses of many landscapes, as in pylandstats: spatial signatures
(principal components and clusters of landscapes or zones described by their
metrics) and spatio-temporal series of metrics (whole maps, zones, buffers
around points or a grid of zones)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["PCAResult", "SpatialSignatureAnalysis", "SpatioTemporalAnalysis", "grid_zones"]

# Class metrics that are 0 (not NA) for a class absent from a landscape.
_ZERO_WHEN_ABSENT = {"ca", "pland", "np", "pd", "lpi", "te", "ed", "tca", "cpland", "ndca", "dcad"}


# -- zones ---------------------------------------------------------------------------------


def grid_zones(data, size, *, res=None, nodata=None, band=None) -> list[dict]:
    """Square zones of side ``size`` (map units) tiling a raster, from its
    top-left corner (partial squares at the right and bottom edges), as
    GeoJSON-like polygons in row-major order."""
    from .sampling import _grid

    values, _, _, transform = _grid(data, res, nodata, band)
    ny, nx = values.shape
    xmin, ymax = transform.c, transform.f
    xmax, ymin = xmin + nx * transform.a, ymax + ny * transform.e
    size = float(size)
    if not size > 0:
        raise ValueError("size must be positive")
    zones = []
    y1 = ymax
    while y1 > ymin + 1e-9 * size:
        y0 = max(ymin, y1 - size)
        x0 = xmin
        while x0 < xmax - 1e-9 * size:
            x1 = min(xmax, x0 + size)
            ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            zones.append({"type": "Polygon", "coordinates": [ring]})
            x0 = x1
        y1 = y0
    return zones


# -- spatio-temporal ------------------------------------------------------------------------


class SpatioTemporalAnalysis:
    """Landscape metrics along a time series of rasters (pylandstats'
    ``SpatioTemporalAnalysis`` and its zonal, buffer and grid variants).

    Parameters
    ----------
    rasters
        The rasters (paths, DataArrays, arrays) or a
        :class:`~landschaft.RasterCollection`, one per date.
    dates
        Labels of the rasters (default 1..n).
    zones
        Polygons to analyse separately (as in :func:`sample_metrics`).
    points, size, shape
        Buffers of ``size`` around points instead of zones.
    grid
        Side of square zones tiling the rasters instead of zones.
    **options
        Metric options (``directions``, ``edge_depth``, ...).
    """

    def __init__(self, rasters, dates=None, *, zones=None, points=None, size=None, shape: str = "square",
                 grid=None, **options):
        from .landscape import as_layers

        layers = as_layers(rasters)
        self.rasters = [rasters] if layers is None else layers
        self.dates = list(range(1, len(self.rasters) + 1)) if dates is None else list(dates)
        if len(self.dates) != len(self.rasters):
            raise ValueError("one date per raster")
        if sum(v is not None for v in (zones, points, grid)) > 1:
            raise ValueError("give zones, points or grid, not several")
        self.options = options
        self._zonal_kwargs = None
        if zones is not None:
            self._zonal_kwargs = {"y": zones}
        elif points is not None:
            if size is None:
                raise ValueError("points need a buffer size")
            self._zonal_kwargs = {"y": points, "size": size, "shape": shape}
        elif grid is not None:
            self._zonal_kwargs = {"y": grid_zones(self.rasters[0], grid, res=options.get("res"))}
        self.zonal = self._zonal_kwargs is not None

    def metrics(self, level=None, metric=None, name=None, type=None, what=None, *, extras: bool = False) -> pd.DataFrame:
        """Long table of the selected metrics with a ``date`` column (and
        ``zone`` and ``percentage_inside`` for zonal analyses)."""
        from .landscape import landscape_metrics
        from .sampling import sample_metrics

        if self.zonal:
            df = sample_metrics(self.rasters, level=level, metric=metric, name=name, type=type, what=what,
                                extras=extras, verbose=False, **self._zonal_kwargs, **self.options)
            if "layer" not in df.columns:
                df.insert(0, "layer", 1)
            df = df.rename(columns={"plot_id": "zone"})
        else:
            df = landscape_metrics(self.rasters, level, metric, name, type, what, extras=extras, **self.options)
            if "layer" not in df.columns:
                df.insert(0, "layer", 1)
        df.insert(0, "date", np.asarray(self.dates, dtype=object)[df.pop("layer").to_numpy() - 1])
        return df

    def landscape_metrics(self, metrics=None) -> pd.DataFrame:
        """Landscape-level metrics: one row per date (and zone), one column
        per metric."""
        df = self.metrics(level="landscape") if metrics is None else self.metrics(
            what=["lsm_l_" + m for m in _as_list(metrics)])
        index = ["zone", "date"] if self.zonal else ["date"]
        wide = df.assign(metric=df["metric"].astype(str)).pivot_table(
            index=index, columns="metric", values="value", aggfunc="first", dropna=False, sort=False)
        wide.columns.name = None
        return wide

    def class_metrics(self, metrics=None, classes=None, *, fillna: bool = True) -> pd.DataFrame:
        """Class-level metrics: one row per class and date (and zone). With
        ``fillna``, a class missing at a date has 0 for area / edge / count
        metrics (``ca``, ``pland``, ``np``, ``te``...) instead of NaN."""
        df = self.metrics(level="class") if metrics is None else self.metrics(
            what=["lsm_c_" + m for m in _as_list(metrics)])
        df = df[df["class"].notna()]
        index = ["class", "zone", "date"] if self.zonal else ["class", "date"]
        wide = df.assign(metric=df["metric"].astype(str)).pivot_table(
            index=index, columns="metric", values="value", aggfunc="first", dropna=False, sort=True)
        wide.columns.name = None
        keys = [list(dict.fromkeys(sorted(df["class"].astype(int))))]
        if self.zonal:
            keys.append(list(dict.fromkeys(sorted(df["zone"]))))
        keys.append(self.dates)
        wide = wide.reindex(pd.MultiIndex.from_product(keys, names=index))
        if classes is not None:
            wide = wide.loc[[int(c) for c in np.atleast_1d(classes)]]
        if fillna:
            for col in wide.columns:
                if col in _ZERO_WHEN_ABSENT:
                    wide[col] = wide[col].fillna(0.0)
        return wide

    def plot_metric(self, metric: str, classes=None, *, ax=None, **plot_kwargs):
        """Line plot of a metric over the dates: the landscape level, or one
        line per class with ``classes`` ("all" or class values); zonal
        analyses: one line per zone. Returns the matplotlib Figure."""
        from .plotting import _plt

        plt = _plt()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            fig = ax.figure
        x = np.arange(len(self.dates))
        if classes is None:
            table = self.landscape_metrics([metric])[metric]
            if self.zonal:
                for zone, series in table.groupby(level="zone"):
                    ax.plot(x, series.to_numpy(), marker="o", label=f"zone {zone}", **plot_kwargs)
            else:
                ax.plot(x, table.to_numpy(), marker="o", **plot_kwargs)
        else:
            table = self.class_metrics([metric])[metric]
            wanted = sorted(set(table.index.get_level_values("class"))) if classes == "all" else np.atleast_1d(classes)
            for c in wanted:
                series = table.xs(int(c), level="class")
                if self.zonal:
                    for zone, s in series.groupby(level="zone"):
                        ax.plot(x, s.to_numpy(), marker="o", label=f"class {c}, zone {zone}", **plot_kwargs)
                else:
                    ax.plot(x, series.to_numpy(), marker="o", label=f"class {c}", **plot_kwargs)
        ax.set_xticks(x, [str(d) for d in self.dates])
        ax.set_ylabel(metric)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        return fig


def _as_list(value) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


# -- spatial signatures ------------------------------------------------------------------------


@dataclass
class PCAResult:
    """Principal components of the spatial signatures."""

    scores: pd.DataFrame  #: one row per landscape / zone, one column per component
    loadings: pd.DataFrame  #: one row per metric, one column per component
    explained_variance_ratio: np.ndarray
    dropped: list  #: rows left out (NaN values)


class SpatialSignatureAnalysis:
    """Landscapes (or zones) described by their metrics: the "spatial
    signature" matrix, its principal components and clusters (pylandstats'
    ``SpatialSignatureAnalysis``).

    Parameters
    ----------
    rasters
        Rasters (or a :class:`~landschaft.RasterCollection`) to compare, or
        one raster with ``zones`` / ``points`` + ``size`` / ``grid``.
    landscape_metrics, class_metrics
        Metric names; class metrics become one column per class
        (``<metric>_<class>``), restricted to ``classes`` if given.
    fillna
        0 instead of NaN for area / edge / count class metrics of absent
        classes.
    zones, points, size, shape, grid
        Zonal signatures of one raster (see :class:`SpatioTemporalAnalysis`).
    **options
        Metric options.
    """

    def __init__(self, rasters=None, *, landscape_metrics=None, class_metrics=None, classes=None, fillna: bool = True,
                 zones=None, points=None, size=None, shape: str = "square", grid=None, table: pd.DataFrame = None,
                 **options):
        if table is not None:
            self.table = table.astype(float)
            return
        if landscape_metrics is None and class_metrics is None:
            raise ValueError("give landscape_metrics and/or class_metrics")
        sta = SpatioTemporalAnalysis(rasters, zones=zones, points=points, size=size, shape=shape, grid=grid,
                                     **options)
        parts = []
        by = "zone" if sta.zonal else "date"
        if landscape_metrics is not None:
            land = sta.landscape_metrics(landscape_metrics)
            if sta.zonal:
                land = land.droplevel("date")
            parts.append(land)
        if class_metrics is not None:
            cls = sta.class_metrics(class_metrics, classes, fillna=fillna)
            if sta.zonal:
                cls = cls.droplevel("date")
            wide = cls.unstack("class")
            wide.columns = [f"{m}_{c}" for m, c in wide.columns]
            parts.append(wide)
        self.table = pd.concat(parts, axis=1).astype(float)
        self.table.index.name = by if sta.zonal else "landscape"

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> SpatialSignatureAnalysis:
        """An analysis of a ready signature matrix (rows = landscapes, columns = metrics)."""
        return cls(table=table)

    def _standardised(self) -> tuple[pd.DataFrame, list]:
        x = self.table
        bad = x.index[x.isna().any(axis=1)].tolist()
        if bad:
            warnings.warn(f"{len(bad)} rows with NaN values are left out", RuntimeWarning, stacklevel=3)
            x = x.dropna()
        sd = x.std(ddof=0)
        scale = x.abs().max()
        constant = sd[(sd <= 1e-12 * scale) | (sd == 0)].index.tolist()
        if constant:
            x = x.drop(columns=constant)
            sd = sd.drop(constant)
        return (x - x.mean()) / sd, bad

    def pca(self, n_components: int | None = None) -> PCAResult:
        """Principal components of the standardised metrics (z-scores; constant
        metrics, up to rounding, are left out). Signs as scikit-learn's PCA (the largest loading
        of each component is positive)."""
        z, dropped = self._standardised()
        u, s, vt = np.linalg.svd(z.to_numpy(), full_matrices=False)
        k = len(s) if n_components is None else min(int(n_components), len(s))
        # sign: the largest absolute loading of each component positive
        signs = np.sign(vt[np.arange(len(s)), np.argmax(np.abs(vt), axis=1)])
        signs[signs == 0] = 1
        u, vt = u * signs, vt * signs[:, None]
        names = [f"PC{i + 1}" for i in range(k)]
        scores = pd.DataFrame(u[:, :k] * s[:k], index=z.index, columns=names)
        loadings = pd.DataFrame(vt[:k].T, index=z.columns, columns=names)
        return PCAResult(scores, loadings, (s**2 / np.sum(s**2))[:k], dropped)

    def cluster(self, n_clusters: int, *, n_components: int | None = None, n_init: int = 10,
                seed: int = 0) -> pd.Series:
        """k-means clusters of the landscapes on the standardised metrics (or
        on the first ``n_components`` principal components), k-means++
        starts, the best of ``n_init``. Returns the cluster (0..k-1) of every
        row."""
        if n_components is not None:
            x = self.pca(n_components).scores
        else:
            x, _ = self._standardised()
        labels = _kmeans(x.to_numpy(), int(n_clusters), n_init, seed)
        return pd.Series(labels, index=x.index, name="cluster")

    def plot_pca(self, clusters: pd.Series | None = None, components=(1, 2), *, ax=None, labels: bool = False):
        """Scatter plot of two principal components (coloured by cluster),
        with the metric loadings as arrows. Returns the Figure."""
        from .plotting import _plt

        plt = _plt()
        res = self.pca()
        a, b = (f"PC{components[0]}", f"PC{components[1]}")
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure
        colours = None if clusters is None else clusters.reindex(res.scores.index).to_numpy()
        ax.scatter(res.scores[a], res.scores[b], c=colours, cmap="tab10")
        if labels:
            for name, row in res.scores.iterrows():
                ax.annotate(str(name), (row[a], row[b]), fontsize=7)
        scale = np.abs(res.scores[[a, b]].to_numpy()).max()
        for metric, row in res.loadings.iterrows():
            ax.arrow(0, 0, row[a] * scale, row[b] * scale, color="grey", alpha=0.5, head_width=0.02 * scale)
            ax.annotate(metric, (row[a] * scale, row[b] * scale), fontsize=7, color="grey")
        i, j = components[0] - 1, components[1] - 1
        ax.set_xlabel(f"{a} ({res.explained_variance_ratio[i]:.0%})")
        ax.set_ylabel(f"{b} ({res.explained_variance_ratio[j]:.0%})")
        return fig

    def plot_cluster_metrics(self, clusters: pd.Series):
        """Box plots of every metric per cluster. Returns the Figure."""
        from .plotting import _panels

        cols = list(self.table.columns)
        fig, axes = _panels(len(cols), None, None, size=3.0)
        groups = clusters.reindex(self.table.index)
        for ax, col in zip(axes, cols):
            data = [self.table.loc[groups == k, col].dropna().to_numpy() for k in sorted(groups.dropna().unique())]
            ax.boxplot(data)
            ax.set_xticks(range(1, len(data) + 1), [str(int(k)) for k in sorted(groups.dropna().unique())])
            ax.set_title(col, fontsize=9)
        return fig


def _kmeans(x: np.ndarray, k: int, n_init: int, seed: int) -> np.ndarray:
    n = len(x)
    if not 1 <= k <= n:
        raise ValueError(f"n_clusters must be between 1 and the number of rows ({n})")
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(max(1, n_init)):
        # k-means++ initialisation
        centres = [x[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.array(centres)[None]) ** 2).sum(axis=2), axis=1)
            total = d2.sum()
            centres.append(x[rng.choice(n, p=d2 / total)] if total > 0 else x[rng.integers(n)])
        centres = np.array(centres)
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(300):
            d2 = ((x[:, None, :] - centres[None]) ** 2).sum(axis=2)
            new = np.argmin(d2, axis=1)
            moved = np.array([x[new == j].mean(axis=0) if np.any(new == j) else centres[j] for j in range(k)])
            if np.array_equal(new, labels) and np.allclose(moved, centres):
                break
            labels, centres = new, moved
        inertia = ((x - centres[labels]) ** 2).sum()
        if inertia < best_inertia - 1e-12:
            best_labels, best_inertia = labels, inertia
    # clusters numbered by first appearance
    order = {}
    for lab in best_labels:
        order.setdefault(lab, len(order))
    return np.array([order[lab] for lab in best_labels])
