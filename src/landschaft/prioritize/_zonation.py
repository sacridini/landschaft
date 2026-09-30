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
    preprocessing : dict
        With ``preprocessing=True``: Zonation's pre-processing maps
        (``richness.pos``, ``rsr.pos``, ... ``.weighted``; see :func:`zonation`).
    transformed : dict
        With ``transformed``: the feature layers as ranked (after quantization,
        condition and retention), normalized to sum 1, by feature name.
    hierarchic_mask_ranks : dict
        With a hierarchic mask: the lowest rank of each of its levels.
    impacts : dict
        With impact scenarios: ``zones`` (per area, group, source, zone, feature and
        bin: the features' distribution in the zone, its loss and cells),
        ``cumulative`` (per area, group, phase and feature) and ``rasters`` (the
        impact of each group).
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
    preprocessing: dict = field(default_factory=dict, repr=False)
    transformed: dict = field(default_factory=dict, repr=False)
    hierarchic_mask_ranks: dict = field(default_factory=dict, repr=False)
    impacts: dict = field(default_factory=dict, repr=False)
    _ctx: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------ post-processing

    def _packed_rank(self) -> np.ndarray:
        return self._ctx["rank_packed"]

    def area_statistics(self, areas):
        """Statistics of areas of the landscape: their size, mean rank, the share
        of every feature they hold and how rich they are relative to their size
        (Zonation's ``area_statistics_external``).

        Parameters
        ----------
        areas
            Raster of area numbers (0 or NoData: none) on the features' grid.

        Returns
        -------
        AreaStatistics

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> x = np.random.default_rng(0).random((3, 20, 30))
        >>> res = ls.prioritize.zonation(x)
        >>> zones = np.repeat([[1, 2, 3]], 20 * 10, axis=0).reshape(20, 30)
        >>> st = res.area_statistics(zones)
        >>> st.areas[["area", "size"]].values.tolist()
        [[1, 200], [2, 200], [3, 200]]
        """
        from . import _inputs, _postprocess

        c = self._ctx
        src = _inputs.Sources([], c["grid"])
        try:
            layer = _inputs.open_aux(areas, c["grid"], "areas", src)
            mask = _inputs.gather(layer, c["cells"], c["grid"], round_int=True).astype(np.int32)
        finally:
            src.close()
        return _postprocess.area_statistics(c, self._packed_rank(), mask)

    def area_statistics_rank_range(self, min_rank: float, max_rank: float = 1.0):
        """:meth:`area_statistics` of the cells ranked in [min_rank, max_rank]
        (Zonation's ``area_statistics_rank_range``); the mask is in the result.

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> res = ls.prioritize.zonation(np.random.default_rng(0).random((3, 20, 30)))
        >>> top = res.area_statistics_rank_range(0.9)      # about the best 10 %
        >>> int(top.areas["size"].iloc[0])                  # 540/600 in float32 is just below 0.9
        59
        """
        from . import _postprocess

        mask = _postprocess.rank_range_mask(self._packed_rank(), min_rank, max_rank)
        st = _postprocess.area_statistics(self._ctx, self._packed_rank(), mask)
        st.mask = self._unpacked(mask.astype(np.uint8), 0)
        return st

    def area_statistics_separate_areas(self, min_rank: float, max_rank: float = 1.0, min_area_size: int = 1):
        """:meth:`area_statistics` of the separate (8-connected) areas of cells
        ranked in [min_rank, max_rank] of at least ``min_area_size`` cells
        (Zonation's ``area_statistics_separate_areas``).

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> res = ls.prioritize.zonation(np.random.default_rng(0).random((3, 20, 30)))
        >>> st = res.area_statistics_separate_areas(0.8, 1.0, min_area_size=3)
        >>> bool((st.areas["size"] >= 3).all())
        True
        """
        from . import _postprocess

        c = self._ctx
        labels = _postprocess.separate_areas(np.asarray(self.rank, dtype=np.float32), min_rank, max_rank,
                                             int(min_area_size))
        mask = labels.ravel()[c["cells"]]
        st = _postprocess.area_statistics(c, self._packed_rank(), mask)
        st.mask = labels
        return st

    def solution_comparison(self, other, min_rank: float, max_rank: float, min_value: float, max_value: float):
        """Compare the cells ranked in [min_rank, max_rank] with the cells of another
        solution with values in [min_value, max_value] (Zonation's
        ``solution_comparison``).

        Returns
        -------
        dict
            ``both``, ``only_this``, ``only_other``, ``neither`` (cell counts) and
            ``map`` (1 both, 2 only this, 3 only the other, 4 neither, 0 no data).

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> res = ls.prioritize.zonation(np.random.default_rng(0).random((3, 20, 30)))
        >>> cmp = res.solution_comparison(res.rank, 0.5, 1.0, 0.5, 1.0)   # itself
        >>> cmp["both"], cmp["only_this"], cmp["only_other"]
        (300, 0, 0)
        """
        from . import _inputs

        c = self._ctx
        src = _inputs.Sources([], c["grid"])
        try:
            ext = _inputs.gather(_inputs.open_aux(other, c["grid"], "solution", src), c["cells"], c["grid"])
        finally:
            src.close()
        ext = ext.astype(np.float32)
        r = self._packed_rank()
        in_p = (r >= np.float32(min_rank)) & (r <= np.float32(max_rank))
        in_e = (ext >= np.float32(min_value)) & (ext <= np.float32(max_value))
        code = np.full(r.size, 4, np.uint8)
        code[in_p & in_e] = 1
        code[in_p & ~in_e] = 2
        code[~in_p & in_e] = 3
        return {"both": int((in_p & in_e).sum()), "only_this": int((in_p & ~in_e).sum()),
                "only_other": int((~in_p & in_e).sum()), "neither": int((~in_p & ~in_e).sum()),
                "map": self._unpacked(code, 0)}

    def histogram(self, min_rank: float, max_rank: float = 1.0, bins: int = 10) -> dict:
        """How many features keep which fraction in the cells ranked in [min_rank,
        max_rank]: per weight class and output group, the count of features per bin
        of coverage (Zonation's ``histogram``).

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> res = ls.prioritize.zonation(np.random.default_rng(0).random((3, 20, 30)))
        >>> h = res.histogram(0.9, bins=10)["pos"]
        >>> int(h["count"].sum())
        3
        """
        from . import _postprocess

        c = self._ctx
        levels = self.feature_curves.iloc[:, 1:].to_numpy(np.float32)
        crank = self.feature_curves["rank"].to_numpy(np.float32)
        return _postprocess.histogram(c, crank, levels, float(min_rank), float(max_rank), int(bins))

    def flexibility_map(self, max_error: float = 2.0, *, up: bool = True, down: bool = True) -> dict:
        """How far each cell could move in the ranking before its marginal loss, at the
        other position, differs from that of the cell there by ``max_error`` %
        (Zonation's ``flexibility_map``): ``up``, ``down`` and their sum ``both``,
        as fractions of the cells.

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> res = ls.prioritize.zonation(np.random.default_rng(0).random((3, 20, 30)))
        >>> f = res.flexibility_map(5.0)
        >>> sorted(f)
        ['both', 'down', 'up']
        """
        from . import _postprocess

        c = self._ctx
        levels = self.feature_curves.iloc[:, 1:].to_numpy(np.float32)
        crank = self.feature_curves["rank"].to_numpy(np.float32)
        pos, neg = _postprocess.flexibility(c, c["order_packed"], crank, levels, max_error, up, down)
        out = {}
        if up:
            out["up"] = self._unpacked(pos, np.nan)
        if down:
            out["down"] = self._unpacked(neg, np.nan)
        if up and down:
            out["both"] = self._unpacked((pos + neg).astype(np.float32), np.nan)
        return out

    def postprocess(self, calls, directory) -> list:
        """Run Zonation's post-processing calls and write its files into ``directory``.

        Parameters
        ----------
        calls
            A post-processing file (its paths are relative to it), its text, or a list
            of ``(name, [parameters])``: ``area_statistics_external(file, id)``,
            ``area_statistics_rank_range(min_rank, max_rank, id)``,
            ``area_statistics_separate_areas(min_rank, max_rank, min_area_size, id)``,
            ``solution_comparison(min_rank, max_rank, file, min_value, max_value, id)``,
            ``histogram(min_rank, max_rank, bins, id)``,
            ``flexibility_map(max_error, both, up, down, id)``.
        directory
            Output directory.

        Returns
        -------
        list of pathlib.Path
            The files written.

        Examples
        --------
        >>> import tempfile
        >>> import numpy as np
        >>> import landschaft as ls
        >>> res = ls.prioritize.zonation(np.random.default_rng(0).random((3, 20, 30)))
        >>> files = res.postprocess('histogram(0.9, 1.0, 10, "top")', tempfile.mkdtemp())
        >>> [f.name for f in files]
        ['histogram.pos.top.txt']
        """
        from . import _postprocess

        base = Path.cwd()
        if isinstance(calls, os.PathLike) or (isinstance(calls, str) and "(" not in calls):
            base = Path(calls).resolve().parent
            calls = Path(calls).read_text(encoding="utf-8")
        if isinstance(calls, str):
            calls = _postprocess.parse_calls(calls)
        return _postprocess.run_calls(self, calls, Path(directory), base)

    def _unpacked(self, values, fill):
        from ._zonation import _raster

        c = self._ctx
        H, W = c["grid"].shape
        out = np.full(H * W, fill, dtype=np.asarray(values).dtype)
        out[c["cells"]] = values
        return _raster(out.reshape(H, W), c["grid"])

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
        imp = self._ctx.get("impacts")
        if imp is not None:
            from . import _impacts
            from ._postprocess import _write_tif

            root = d / "impacts"
            for key, state, writer in (("source", imp["source"], _impacts.write_source),
                                       ("cumulative", imp["cumulative"], _impacts.write_cumulative)):
                if state is None:
                    continue
                writer(imp, self._ctx, root)
                sc = state["scenario"]
                for name, r in self.impacts.get("rasters", {}).items():
                    if (key == "source" and name.endswith("_zones")) or (key == "cumulative" and not name.endswith(
                            "_zones")):
                        _write_tif(root / sc.name / f"{name}.tif", r, "float32")
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
    condition=None,
    renormalize=False,
    connectivity=None,
    self_multiply=False,
    interaction=None,
    interaction2=None,
    matrix_connectivity=None,
    matrices: dict | None = None,
    retention=None,
    preprocessing: bool = False,
    transformed=False,
    impact_groups=None,
    impact_bins=None,
    impact=None,
    cumulative_impact=None,
    nodata=None,
    storage=None,
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
    condition
        Per feature, a raster of the condition of its habitat, 0-1 (``None``: none;
        Zonation's ``-c``): the quantized values are multiplied by it (truncated),
        and the feature's initial level becomes the conditioned fraction.
    renormalize
        Per feature (or for all), rescale a conditioned feature to level 1.
    connectivity
        Per feature, a :class:`Kernel` (``None``: none; Zonation's ``-d``): the
        feature is replaced by its convolution with the kernel (its connectivity),
        kept where it had data, and quantized again.
    self_multiply
        Per feature (or for all): multiply the smoothed feature by the feature.
    interaction, interaction2
        Per feature, an :class:`Interaction` (``-i``, ``-I``): the feature weighted
        by (or reduced by) the connectivity of a resource. The second set is applied
        after the matrix transforms.
    matrix_connectivity, matrices
        Per feature, ``(kernel, key)`` (``-m``): the features sharing a key are
        smoothed and mixed by the matrix ``matrices[key]`` (row: source feature,
        column: target feature, in the features' order), then multiplied by the
        target feature.
    retention
        Per feature (Zonation's ``-r``): ``None``; a number, the fraction of the
        feature retained everywhere; a raster of fractions retained per cell; or
        ``("absolute", raster)``, the amounts retained per cell. What is retained
        is taken out of the feature (it stays whatever the ranking), so its curve
        ends at the retained fraction.
    preprocessing
        Also compute Zonation's pre-processing maps (``-b``): per cell, the sum of
        the features' original values (``richness``) and of their normalized values
        (range-size rarity, ``rsr``), for the positive, negative and zero weights and
        each output group, and weighted with weights.
    transformed
        Per feature (or ``True`` for all), return the feature as ranked (``-f``).
    impact_groups
        Per feature, its impact group (-1: none), for the impact scenarios.
    impact_bins
        Per feature, the value classes its impacts are reported in: ``None`` (none),
        ``0`` (ten equal classes up to its maximum) or a list of class limits in the
        feature's units.
    impact, cumulative_impact
        An :class:`ImpactScenario` (``-s``: land-use change around impact sources)
        and/or a :class:`CumulativeScenario` (``-e``: direct habitat loss and
        cumulative pressures). The features of the impact groups lose what the
        scenarios take before the ranking; the impacts are in ``impacts``.
    nodata
        NoData value of inputs without one (NaN is always NoData).
    storage
        A directory for temporary files: the large arrays (the quantized
        features, the rows of the ranking and its work arrays) are then
        memory-mapped files that the operating system pages to disk when they
        do not fit in memory, for maps larger than the RAM (out-of-core). The
        files are deleted when no longer needed. ``None``: in memory.
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


def _storage(path) -> str:
    """The directory for the temporary files ("": in memory)."""
    if path is None:
        return ""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


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
    # conditions: one plane per distinct raster
    cond_index = np.full(L0, -1, np.int32)
    cond_layers = []
    if a["condition"] is not None:
        conds = list(a["condition"])
        if len(conds) != L0:
            raise ValueError(f"condition: {len(conds)} for {L0} features")
        seen = {}
        for i, c in enumerate(conds):
            if c is None:
                continue
            key = os.fspath(c) if _inputs._is_path(c) else id(c)
            if key not in seen:
                seen[key] = len(cond_layers)
                cond_layers.append(_inputs.open_aux(c, src.grid, "condition", src))
            cond_index[i] = seen[key]
    renorm = _per_layer(a["renormalize"], L0, False, "renormalize", np.uint8)

    # statistics: maxima, empty layers
    stats = _core.PrioStats(nodata.tolist(), extract.tolist(), thresholds.tolist())
    for y0, y1 in _inputs.chunks(H, W, L0):
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
    for what, gs in (("weight", wgroups[kept] if a["weight_groups"] is not None else None),
                     ("output", ogroups[kept] if ogroups is not None else None)):
        if gs is None:
            continue
        for g in sorted({int(x) for x in gs if x >= 0}):
            gw = w[gs == g]
            if (gw > 0).any() and (gw < 0).any():
                warnings.warn(f"{what} group {g} contains both positive and negative weights", stacklevel=3)

    # the quantized data
    storage = _storage(a["storage"])
    data = _core.PrioData(nodata[kept].tolist(), extract[kept].tolist(), thresholds[kept].tolist(),
                          maxv[kept].tolist(), keep[kept].tolist(), int(a["bits"]), storage)
    kl = [src.layers[i] for i in kept]
    if cond_layers:
        data.set_conditions(cond_index[kept].tolist(), renorm[kept].tolist())
    # Zonation adds up the layers block of rows by block (the files' blocks; arrays: one block)
    zonation_rows = src.block_rows or H
    for y0, y1 in _inputs.chunks(H, W, L + len(cond_layers)):
        conds = None
        if cond_layers:
            conds = _inputs.read_rows(cond_layers, y0, y1, W)
            for k, cl in enumerate(cond_layers):
                c = conds[k]
                c[np.isnan(c) | (c == cl.nodata)] = 0.0
            if (conds < 0).any() or (conds > 1).any():
                raise ValueError("condition rasters must have values in [0, 1]")
        data.add_block(_inputs.read_rows(kl, y0, y1, W), _inputs.area_rows(area_layer, y0, y1, which), conds,
                       zonation_rows)
    data.finish()
    if data.cells == 0:
        raise ValueError("no cells with data")
    cells = np.asarray(data.cell_index)

    cost = _inputs.gather(_inputs.open_aux(a["cost"], src.grid, "cost", src), cells, src.grid, lo=0.0)
    hmask = _inputs.gather(_inputs.open_aux(a["hierarchic_mask"], src.grid, "hierarchic mask", src), cells,
                           src.grid, lo=0, hi=255, round_int=True)
    solution = _inputs.gather(_inputs.open_aux(a["solution"], src.grid, "solution", src), cells, src.grid)
    if any(a[k] is not None for k in ("connectivity", "interaction", "interaction2", "matrix_connectivity")):
        from ._connectivity import apply as _connect

        def per(key, default=None):
            v = a[key]
            if v is None:
                return None
            if key == "self_multiply" and isinstance(v, bool):
                return [v] * L0
            v = list(v)
            if len(v) != L0:
                raise ValueError(f"{key}: {len(v)} for {L0} features")
            return v

        _connect(data, kept, cells, src.grid, connectivity=per("connectivity"), self_multiply=per("self_multiply"),
                 interaction=per("interaction"), interaction2=per("interaction2"),
                 matrix_connectivity=per("matrix_connectivity"), matrices=a["matrices"] or {},
                 open_aux=lambda r: _inputs.open_aux(r, src.grid, "interaction resource", src),
                 gather=lambda layer, c: _inputs.gather(layer, c, src.grid),
                 workers=_threads.current())
    _retention(data, a["retention"], kept, L0, cells, src)
    imp_out = None
    if a["impact"] is not None or a["cumulative_impact"] is not None:
        from . import _impacts

        groups = _per_layer(a["impact_groups"], L0, -1, "impact_groups", np.int64)
        ib = a["impact_bins"]
        if ib is not None:
            ib = list(ib)
            if len(ib) != L0:
                raise ValueError(f"impact_bins: {len(ib)} for {L0} features")

        def read_full(r):
            layer = _inputs.open_aux(r, src.grid, "impact raster", src)
            v = layer.read(0, H).astype(np.float32)
            v[np.isnan(v) | (v == layer.nodata)] = 0
            return v

        def read_sources(r):
            layer = _inputs.open_aux(r, src.grid, "impact source raster", src)
            return layer.read(0, H).astype(np.float64)

        def read_packed(r):
            return _inputs.gather(_inputs.open_aux(r, src.grid, "subareas raster", src), cells, src.grid)

        imp = a["impact"]
        if imp is not None:
            import dataclasses

            imp = dataclasses.replace(imp, sources=read_sources(imp.sources))
        imp_out = _impacts.run(data, kept, cells, src.grid, groups, ib, imp, a["cumulative_impact"],
                               read_full, read_packed)
    seed = a["seed"]
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    r = _core.prio_rank(
        data, name, param, w.tolist(), abf_z[kept].tolist(),
        None if cost is None else cost.astype(np.float32), bool(a["cost_in_ranking"]),
        None if hmask is None else hmask.astype(np.uint8),
        None if solution is None else solution.astype(np.float32),
        int(a["max_iterations"]), float(a["ranking_error"]), bool(a["stop_on_plateau"]),
        float(a["smoothing_factor"]), float(a["curves_interval"]), int(seed),
        # the storage of the rows (the smaller by default; forced in the tests of both)
        {"delta": 1, "bitmap": 2}.get(os.environ.get("LANDSCHAFT_PRIO_ROWS", ""), 0), storage)

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
    res = ZonationResult(rank, fc, sc, dc, gc, features, int(r["iterations"]), float(r["ranking_error"]) * 100.0,
                         float(r["auc"]), history, cells[np.asarray(r["order"])], info)
    rank_packed = np.asarray(r["rank"], dtype=np.float32)
    if hmask is not None:
        hm = hmask.astype(np.uint8)
        res.hierarchic_mask_ranks = {int(lv): float(rank_packed[hm == lv].min()) for lv in np.unique(hm)}
    if imp_out is not None:
        from . import _impacts

        res.impacts = {**_impacts.tables(imp_out, {"numbers": kept}),
                       "rasters": {k: _raster(v, src.grid) for k, v in imp_out["rasters"].items()}}
    res._ctx = {
        "data": data, "grid": src.grid, "cells": cells, "rank_packed": rank_packed,
        "order_packed": np.asarray(r["order"], dtype=np.uint32), "weights": w, "classes": classes,
        "groups": None if ogroups is None else ogroups[kept], "groups_on": ogroups is not None, "names": lnames,
        "headers": headers, "numbers": kept,
        "quoted_names": [f'"{names[i]}"' if names else '""' for i in kept],
        "files": [str(src.layers[i].source) if _inputs._is_path(src.layers[i].source) else "" for i in kept],
        "cost": None if cost is None else cost.astype(np.float32), "cost_in_ranking": bool(a["cost_in_ranking"]),
        "hmask": None if hmask is None else hmask.astype(np.uint8), "rule": name, "param": param,
        "abf_z": abf_z[kept], "curve_interval": int(r["curve_interval"]),
        "curve_scalar": data.cells / int(r["curve_interval"]),
        "impacts": imp_out, "orig_abs_sums": np.asarray(data.orig_abs_sums),
        "plain_names": [names[i] if names else "" for i in kept],
        "unique_names": _unique_names(src, kept, a, names),
    }
    if a["preprocessing"]:
        res.preprocessing = _preprocessing(data, w, classes, weights_on, ogroups[kept] if ogroups is not None
                                           else None, cells, src.grid)
    tr = a["transformed"]
    if tr is not None and tr is not False:
        flags = [True] * L0 if tr is True else list(tr)
        if len(flags) != L0:
            raise ValueError(f"transformed: {len(flags)} for {L0} features")
        scale = np.asarray(data.layer_scale)
        for k, i in enumerate(kept):
            if flags[i]:
                q = np.asarray(data.layer(k, False, -1.0))
                v = np.where(q >= 0, (q.astype(np.float32).astype(np.float64) * scale[k]).astype(np.float32), np.nan)
                res.transformed[lnames[k]] = _raster(_unpack(v.astype(np.float32), cells, H, W), src.grid)
    return res


def _unique_names(src, kept, a, names) -> list[str]:
    """Zonation's unique layer names: the file name with ".layer<N>" and the transforms applied."""
    out = []
    for i in kept:
        ly = src.layers[i]
        base = Path(os.fspath(ly.source)).stem if _inputs._is_path(ly.source) else (names[i] if names else ly.name)
        ext = ""

        def on(key):
            v = a[key]
            return v is not None and v is not False and (v is True or (isinstance(v, (list, tuple)) and
                                                                        v[i] is not None and v[i] is not False))

        if a["thresholds"] is not None and float(np.asarray(a["thresholds"], dtype=float)[i]) > 0:
            ext += "t"
        if on("condition"):
            ext += "c"
        if on("retention"):
            ext += "r"
        if on("connectivity"):
            ext += "d"
        if on("interaction"):
            ext += "i"
        if on("interaction2"):
            ext += "I"
        if on("matrix_connectivity"):
            ext += "m"
        g = -1 if a["impact_groups"] is None else int(a["impact_groups"][i])
        if a["impact"] is not None and g >= 0 and any(s.group == g for s in a["impact"].impacts):
            ext += "s"
        if a["cumulative_impact"] is not None and g >= 0:
            ext += "e"
        out.append(f"{base}.layer{i}" + (f".{ext}" if ext else ""))
    return out


def _unpack(values: np.ndarray, cells: np.ndarray, H: int, W: int) -> np.ndarray:
    out = np.full(H * W, np.nan, dtype=np.float32)
    out[cells] = values
    return out.reshape(H, W)


def _retention(data, retention, kept, L0, cells, src) -> None:
    """Zonation's retention transforms (-r), after the other transforms."""
    if retention is None:
        return
    ret = list(retention)
    if len(ret) != L0:
        raise ValueError(f"retention: {len(ret)} for {L0} features")
    for k, i in enumerate(kept):
        r = ret[i]
        if r is None:
            continue
        if isinstance(r, (int, float, np.floating, np.integer)) and not isinstance(r, bool):
            data.retain(k, 2, None, float(np.float32(r)))
            continue
        kind = 1
        if isinstance(r, (tuple, list)):
            what, r = r
            if what not in ("absolute", "fraction"):
                raise ValueError("retention: ('absolute' | 'fraction', raster)")
            kind = 0 if what == "absolute" else 1
        v = _inputs.gather(_inputs.open_aux(r, src.grid, "retention", src), cells, src.grid).astype(np.float32)
        if (v < 0).any():
            raise ValueError("retention rasters must not have negative values")
        if kind == 1 and (v > 1).any():
            raise ValueError("fractional retention rasters must have values in [0, 1]")
        clipped = data.retain(k, kind, v, 0.0)
        if clipped:
            warnings.warn(f"feature {i}: {clipped} cells where the retention is higher than the value, clipped",
                          stacklevel=4)


def _preprocessing(data, w, classes, weights_on, groups, cells, grid) -> dict:
    """Zonation's pre-processing maps (main.cpp outputPreprocessedMaps)."""
    H, W = grid.shape
    L = len(w)
    sets = [("pos", classes["positive"]), ("neg", classes["negative"]), ("zero", classes["zero"])]
    if groups is not None:
        for g in sorted({int(x) for x in groups if x >= 0}):
            sets.append((f"group{g}", [i for i, x in enumerate(groups) if x == g]))
    out = {}

    def one(idx, weights, unscaled):
        mult = np.zeros(L, np.float32)
        mult[idx] = 1
        v = np.asarray(data.preprocessing(mult, weights, unscaled), dtype=np.float32)
        return _raster(_unpack(v, cells, H, W), grid)

    for name, idx in sets:
        if idx:
            out[f"richness.{name}"] = one(idx, None, True)
            out[f"rsr.{name}"] = one(idx, None, False)
    if weights_on:
        wf = np.asarray(w, np.float32)
        for name, idx in sets:
            if idx and name != "zero":
                out[f"richness.{name}.weighted"] = one(idx, wf, True)
                out[f"rsr.{name}.weighted"] = one(idx, wf, False)
    return out


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
