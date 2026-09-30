# SPDX-License-Identifier: GPL-3.0-or-later
"""Zonation 5's spatial prioritization: ``zonation`` and its result."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import _core, _threads
from . import _inputs, _tables

RULES = ("CAZ1", "CAZ2", "CAZMAX", "CAZP", "CAZPW", "ABF", "RAND", "LOAD")
_DEFAULT_PARAM = {"ABF": 0.25, "CAZP": 1.0, "CAZPW": 1.0}


@dataclass
class ZonationResult:
    """The result of :func:`zonation`.

    Attributes
    ----------
    rank : xarray.DataArray
        Priority rank of every cell with data, from 0 (first removed, lowest
        priority) to (n - 1) / n (last removed, highest priority); NaN elsewhere.
    feature_curves : pandas.DataFrame
        Fraction remaining of every feature (columns) as the lowest-ranked cells
        are removed (``rank`` = fraction of the cells removed); Zonation's
        ``feature_curves.csv``.
    summary_curves, density_curves : pandas.DataFrame
        Mean, weighted mean, maximum and minimum of the curves (by the sign of
        the weights), remaining area and cost; and the loss rates of the means.
    group_curves : pandas.DataFrame or None
        Mean, weighted mean and minimum per output group.
    features : pandas.DataFrame
        Per feature: input number, name, weight, the sum of its values, cells with
        values above zero, cells without data, quantization error (%).
    iterations : int
        Sorting iterations run (0 for ``RAND`` and ``LOAD``).
    ranking_error : float
        Mean ranking error of the last iteration (%).
    auc : float
        Area under the mean curve of the positively weighted features.
    history : pandas.DataFrame
        Ranking error (%), number of cells out of order and mean rank change (%)
        per iteration.
    order : numpy.ndarray
        Indices (row-major) of the cells in removal order.
    """

    rank: object
    feature_curves: pd.DataFrame
    summary_curves: pd.DataFrame
    density_curves: pd.DataFrame
    group_curves: pd.DataFrame | None
    features: pd.DataFrame
    iterations: int
    ranking_error: float
    auc: float
    history: pd.DataFrame
    order: np.ndarray
    info: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        n = int(np.isfinite(np.asarray(self.rank)).sum())
        return (f"ZonationResult(rule={self.info.get('rule')}, features={len(self.features)}, cells={n}, "
                f"iterations={self.iterations}, ranking_error={self.ranking_error:.4g}%, auc={self.auc:.4f})")

    def write(self, directory) -> Path:
        """Write the rank map and the tables as Zonation 5 does.

        Parameters
        ----------
        directory
            Output directory (created if needed): ``rankmap.tif``,
            ``feature_curves.csv``, ``summary_curves.csv``,
            ``feature_density_curves.csv``, ``group_curves.csv`` (with output
            groups) and ``features_info.csv``, in Zonation's formats.

        Returns
        -------
        pathlib.Path
            The directory.

        Examples
        --------
        >>> import tempfile
        >>> import numpy as np
        >>> import landschaft as ls
        >>> res = ls.prioritize.zonation(np.random.default_rng(0).random((2, 10, 12)))
        >>> out = res.write(tempfile.mkdtemp())
        >>> sorted(p.name for p in out.iterdir())
        ['feature_curves.csv', 'feature_density_curves.csv', 'features_info.csv', 'rankmap.tif', 'summary_curves.csv']
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            return self._write(d)

    def _write(self, d: Path) -> Path:
        headers = self.info["curve_headers"]
        (d / "feature_curves.csv").write_text(_tables.format_feature_curves(self.feature_curves, headers),
                                              encoding="utf-8")
        (d / "summary_curves.csv").write_text(_tables.format_summary_curves(self.summary_curves), encoding="utf-8")
        (d / "feature_density_curves.csv").write_text(_tables.format_density_curves(self.density_curves),
                                                      encoding="utf-8")
        if self.group_curves is not None:
            (d / "group_curves.csv").write_text(_tables.format_group_curves(self.group_curves), encoding="utf-8")
        f = self.features
        lines = ['"number" "dsum" "count" "missing" "quant_error" "name"']
        for r in f.itertuples():
            lines.append(f'{r.number} {_tables._g(r.dsum)} {r.count} {r.missing} {_tables._g(r.quant_error)} '
                         f'"{r.name}"')
        (d / "features_info.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rank = self.rank
        try:
            from rasterio.errors import NotGeoreferencedWarning

            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            out = rank.fillna(-np.inf) if hasattr(rank, "fillna") else np.where(np.isnan(rank), -np.inf, rank)
            if hasattr(out, "rio"):
                out = out.rio.write_nodata(-np.inf, encoded=False)
                out.rio.to_raster(d / "rankmap.tif")
            else:
                import rasterio

                with rasterio.open(d / "rankmap.tif", "w", driver="GTiff", height=out.shape[0], width=out.shape[1],
                                   count=1, dtype="float32", nodata=-np.inf) as dst:
                    dst.write(np.asarray(out, dtype=np.float32), 1)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"could not write rankmap.tif: {e}")
        return d


def _per_layer(value, n: int, default, what: str, dtype=np.float32) -> np.ndarray:
    if value is None:
        return np.full(n, default, dtype=dtype)
    if np.isscalar(value):
        return np.full(n, value, dtype=dtype)
    v = list(value)
    if len(v) != n:
        raise ValueError(f"{what}: {len(v)} values for {n} features")
    return np.array([default if x is None else x for x in v], dtype=dtype)


def _parse_rule(rule: str, rule_param):
    parts = str(rule).split()
    name = parts[0].upper() if parts else "CAZ2"
    if name not in RULES:
        raise ValueError(f"unknown marginal loss rule {rule!r}; one of {', '.join(RULES)}")
    if len(parts) > 1:
        if rule_param is not None:
            raise ValueError("the rule parameter is given twice")
        rule_param = float(parts[1])
    if rule_param is None:
        rule_param = _DEFAULT_PARAM.get(name, 0.0)
    param = float(np.float32(rule_param))  # Zonation keeps it as a float
    if name in ("CAZP", "CAZPW") and param < 0:
        raise ValueError("the CAZP/CAZPW exponent must be >= 0")
    return name, param


def zonation(
    features,
    *,
    weights=None,
    rule: str = "CAZ2",
    rule_param: float | None = None,
    abf_z=None,
    names=None,
    weight_groups=None,
    group_weights: dict | None = None,
    groups=None,
    thresholds=None,
    zero_mode: str = "all",
    keep_zeros=None,
    extract=None,
    cost=None,
    cost_in_ranking: bool = True,
    hierarchic_mask=None,
    analysis_area=None,
    area: int = 0,
    solution=None,
    bits: int = 8,
    max_iterations: int = 500,
    ranking_error: float = 0.01,
    smoothing_factor: float = 0.0,
    stop_on_plateau: bool = False,
    curves_interval: float = 0.0001,
    seed: int | None = None,
    nodata=None,
    n_threads: int | None = None,
) -> ZonationResult:
    """Rank the cells of a landscape for conservation, as Zonation 5 does.

    Every feature (species distribution, habitat, ecosystem service...) is
    normalized to sum 1 and quantized to ``bits`` bits; the cells are then
    ordered from the least to the most important by their marginal loss: the
    loss to the features, weighted, of removing a cell at the point where it is
    removed, with the other cells removed before it. Zonation iterates stable
    sorts of the cells by that loss until the order is consistent (the mean
    ranking error below ``ranking_error`` %). The result is the same, bit for
    bit, as the z5 program (v2.4) on one thread.

    Parameters
    ----------
    features
        Raster files (every band is a feature), 2-D arrays or DataArrays, or a
        list of them; or a 3-D array/DataArray (features, rows, columns), NumPy
        or dask. All on the same grid.
    weights
        One weight per feature (Zonation's ``-w``); negative weights mark
        features to avoid. ``None``: all 1.
    rule
        Marginal loss rule: ``"CAZ2"`` (default), ``"CAZ1"``, ``"CAZMAX"`` (core-area
        Zonation: the loss is driven by the features that lose the largest
        fraction), ``"CAZP"``/``"CAZPW"`` (p-norms), ``"ABF"`` (additive benefit
        function, benefit ``R^z`` of what remains), ``"RAND"`` (random order) or
        ``"LOAD"`` (the order of ``solution``, to evaluate a given ranking). A
        parameter may follow the name (``"ABF 0.25"``).
    rule_param
        ABF exponent z (0.25) or CAZP/CAZPW exponent p (1).
    abf_z
        ABF exponent per feature (``None`` entries use ``rule_param``).
    names
        Feature names for the tables.
    weight_groups, group_weights
        Weight group of each feature (-1: none) and the total weight of each group
        (``{group: weight}``): the weights of a group are rescaled to sum (in
        absolute value) to its weight (Zonation's ``-W``).
    groups
        Output group of each feature (-1: none), summarized in ``group_curves``.
    thresholds
        Per feature, values below become 0.
    zero_mode
        ``"all"`` (zeros are data), ``"none"`` (zeros are NoData) or ``"dynamic"``
        (per feature, from ``keep_zeros``).
    extract
        Per feature, a value to extract from a categorical raster (cells with it
        keep it, the others become NoData; ``None``: no extraction).
    cost
        Cost of every cell. With ``cost_in_ranking`` the losses are divided by the
        relative cost (``-X``); otherwise the cost is only followed in the curves
        (``-x``). Cells of cost 0 are removed last.
    hierarchic_mask
        Integer levels 0-255 per cell: lower levels are removed first.
    analysis_area, area
        Raster of analysis areas: cells outside (0, or other than ``area`` when
        ``area`` > 0) are left out.
    solution
        The ranking to evaluate with ``rule="LOAD"``.
    bits
        Quantization of the feature values: 8 (``z5``) or 16 (``z5_16bit``).
    max_iterations, ranking_error, smoothing_factor, stop_on_plateau
        The iteration: at most ``max_iterations`` sorts, until the mean ranking
        error is at most ``ranking_error`` % and no longer decreasing; sort by a
        running mix of the losses (``smoothing_factor`` of the previous ones);
        stop when the error stops decreasing after 5 iterations.
    curves_interval
        Fraction of the cells between two rows of the curves (0.0001: 10 000 rows).
    seed
        Seed of ``RAND``.
    nodata
        NoData value of inputs without one (NaN is always NoData).
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    ZonationResult

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> rng = np.random.default_rng(0)
    >>> species = rng.random((3, 20, 30)) ** 3          # three features
    >>> res = ls.prioritize.zonation(species, weights=[2, 1, 1])
    >>> res.rank.shape
    (20, 30)
    >>> float(res.rank.max()) < 1.0
    True
    >>> round(float(res.feature_curves.iloc[0, 1]), 4)  # nothing removed yet
    1.0
    """
    name, param = _parse_rule(rule, rule_param)
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")
    if zero_mode not in ("all", "none", "dynamic"):
        raise ValueError("zero_mode must be 'all', 'none' or 'dynamic'")
    if name == "LOAD" and solution is None:
        raise ValueError("rule='LOAD' needs a solution")
    with _threads.limit(n_threads):
        src = _inputs.open_features(features, nodata)
        try:
            return _run(src, name, param, locals())
        finally:
            src.close()


def _run(src, name, param, a) -> ZonationResult:
    import xarray as xr

    L0 = len(src.layers)
    H, W = src.grid.shape
    names = list(a["names"]) if a["names"] is not None else None
    if names is not None and len(names) != L0:
        raise ValueError(f"names: {len(names)} for {L0} features")
    weights_on = a["weights"] is not None or a["weight_groups"] is not None
    weights = _per_layer(a["weights"], L0, 1.0, "weights")
    nodata = np.array([ly.nodata for ly in src.layers], dtype=np.float32)
    extract = _per_layer(a["extract"], L0, np.nan, "extract")
    thresholds = _per_layer(a["thresholds"], L0, 0.0, "thresholds")
    abf_z = _per_layer(a["abf_z"], L0, np.nan, "abf_z")
    if a["zero_mode"] == "all":
        keep = np.ones(L0, np.uint8)
    elif a["zero_mode"] == "none":
        keep = np.zeros(L0, np.uint8)
    else:
        keep = _per_layer(a["keep_zeros"], L0, 0, "keep_zeros", np.uint8)
    wgroups = _per_layer(a["weight_groups"], L0, -1, "weight_groups", np.int64)
    ogroups = None if a["groups"] is None else _per_layer(a["groups"], L0, -1, "groups", np.int64)

    area_layer = _inputs.open_aux(a["analysis_area"], src.grid, "analysis area", src)
    which = int(a["area"])

    # statistics: maxima, empty layers
    stats = _core.PrioStats(nodata.tolist(), extract.tolist(), thresholds.tolist())
    for y0, y1, _ in _inputs.blocks(H, W, L0, src.block_rows):
        stats.add_block(_inputs.read_rows(src.layers, y0, y1, W), _inputs.area_rows(area_layer, y0, y1, which))
    maxv = np.asarray(stats.max_values, dtype=np.float32)
    kept = [i for i in range(L0) if maxv[i] > 0]
    for i in range(L0):
        if not maxv[i] > 0:
            why = ("has no data after masking and thresholding" if stats.orig_max_values[i] > 0
                   else "value range is zero")
            warnings.warn(f"feature {i} ({src.layers[i].name}) {why}, left out", stacklevel=3)
    if not kept:
        raise ValueError("nothing remains after removing empty layers")
    if int(stats.negative.sum()):
        warnings.warn("negative values found, taken as zero", stacklevel=3)

    # weights after removing empty layers (weight groups over the remaining ones)
    w = weights[kept].astype(np.float32)
    if a["weight_groups"] is not None and a["group_weights"]:
        gw = {int(k): np.float32(v) for k, v in a["group_weights"].items()}
        wg = wgroups[kept]
        sums = {}
        for g, x in zip(wg, w):
            if g >= 0:
                sums[int(g)] = np.float32(sums.get(int(g), np.float32(0)) + np.float32(abs(x)))
        for i, g in enumerate(wg):
            g = int(g)
            if g in gw and sums.get(g, 0) > 0:
                w[i] = np.float32(w[i] * np.float32(gw[g] / sums[g]))
    L = len(kept)

    # the quantized data
    data = _core.PrioData(nodata[kept].tolist(), extract[kept].tolist(), thresholds[kept].tolist(),
                          maxv[kept].tolist(), keep[kept].tolist(), int(a["bits"]))
    kl = [src.layers[i] for i in kept]
    for y0, y1, end in _inputs.blocks(H, W, L, src.block_rows):
        data.add_block(_inputs.read_rows(kl, y0, y1, W), _inputs.area_rows(area_layer, y0, y1, which))
        if end:
            data.flush()
    data.finish()
    if data.cells == 0:
        raise ValueError("no cells with data")
    cells = np.asarray(data.cell_index)

    cost = _inputs.gather(_inputs.open_aux(a["cost"], src.grid, "cost", src), cells, src.grid, lo=0.0)
    hmask = _inputs.gather(_inputs.open_aux(a["hierarchic_mask"], src.grid, "hierarchic mask", src), cells,
                           src.grid, lo=0, hi=255, round_int=True)
    solution = _inputs.gather(_inputs.open_aux(a["solution"], src.grid, "solution", src), cells, src.grid)
    seed = a["seed"]
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    r = _core.prio_rank(
        data, name, param, w.tolist(), abf_z[kept].tolist(),
        None if cost is None else cost.astype(np.float32), bool(a["cost_in_ranking"]),
        None if hmask is None else hmask.astype(np.uint8),
        None if solution is None else solution.astype(np.float32),
        int(a["max_iterations"]), float(a["ranking_error"]), bool(a["stop_on_plateau"]),
        float(a["smoothing_factor"]), float(a["curves_interval"]), int(seed))

    # rank map on the grid
    flat = np.full(H * W, np.nan, dtype=np.float32)
    flat[cells] = r["rank"]
    rank = _raster(flat.reshape(H, W), src.grid)

    lnames = [names[i] if names else src.layers[i].name for i in kept]
    headers = [f'"{names[i]}"' if names else str(i) for i in kept]
    classes = _tables.weight_classes(w, weights_on)
    levels = np.asarray(r["curve_levels"], dtype=np.float32)
    crank = np.asarray(r["curve_rank"], dtype=np.float32)
    fc = _tables.feature_curves(crank, levels, lnames)
    sc = _tables.summary_curves(crank, levels, r["curve_cost"], w, classes, data.cells)
    dc = _tables.density_curves(crank, levels, w, classes)
    gc = _tables.group_curves(crank, levels, w, None if ogroups is None else ogroups[kept])
    count = stats.pixels - np.asarray(stats.nodata)[kept] - np.asarray(stats.zeros)[kept]
    features = pd.DataFrame({
        "number": kept,
        "name": lnames,
        "weight": w,
        "dsum": np.asarray(data.orig_abs_sums),
        "count": count.astype(np.int64),
        "missing": np.asarray(stats.nodata)[kept].astype(np.int64),
        "quant_error": np.asarray(data.quantization_errors),
    })
    history = pd.DataFrame({"ranking_error": np.asarray(r["history_error"]) * 100.0,
                            "errors": np.asarray(r["history_errors"]),
                            "rank_change": np.asarray(r["history_change"]) * 100.0})
    info = {"rule": name if name not in _DEFAULT_PARAM else f"{name} {param:g}", "bits": int(a["bits"]),
            "cells": int(data.cells), "curve_headers": headers, "total_cost": r["total_cost"],
            "ranking_errors": int(r["ranking_errors"]), "rank_change": float(r["rank_change"]) * 100.0,
            "quantization_error": float(np.mean(np.asarray(data.quantization_errors)))}
    return ZonationResult(rank, fc, sc, dc, gc, features, int(r["iterations"]), float(r["ranking_error"]) * 100.0,
                          float(r["auc"]), history, cells[np.asarray(r["order"])], info)


def _raster(values: np.ndarray, grid) -> object:
    import xarray as xr

    t = grid.template
    if t is not None and tuple(t.shape) == values.shape:
        da = t.copy(deep=False, data=values)
        da.name = "rank"
        da.attrs = {}
        da.encoding = {}
        return da
    ny, nx = values.shape
    if grid.transform is not None:
        tr = grid.transform
        x = tr.c + (np.arange(nx) + 0.5) * tr.a
        y = tr.f + (np.arange(ny) + 0.5) * tr.e
        da = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}, name="rank")
        try:
            da = da.rio.write_transform(tr)
            if grid.crs is not None:
                da = da.rio.write_crs(grid.crs)
        except Exception:
            pass
        return da
    return xr.DataArray(values, dims=("y", "x"), name="rank")
