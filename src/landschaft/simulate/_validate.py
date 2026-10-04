# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation of a simulation in one call: every comparison of the observed and the simulated map."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .._threads import threaded
from ._maps import read_map

__all__ = ["Validation", "futures_validation_grid", "validate"]

_FUTURES_SCALARS = ["total_quantity", "total_allocation", "kappa", "kappasimulation", "misses", "hits",
                    "false_alarms", "null_successes", "figure_of_merit", "producer", "user", "initially_developed"]


@threaded
def futures_validation_grid(reference, simulated, original=None, *, size: int | None = None, regions=None,
                            nodata=None) -> pd.DataFrame:
    """FUTURES' validation per tile or per region (``r.futures.gridvalidation``).

    Parameters
    ----------
    reference, simulated, original
        As :func:`futures_validation`.
    size
        Tiles of ``size x size`` cells from the top-left corner.
    regions
        Or a map of region ids (one row per region).

    Returns
    -------
    pandas.DataFrame
        A row per tile (``row``, ``col`` and the centre ``x``, ``y`` in map
        units when the maps are georeferenced) or region (``region``), the
        metrics of :func:`futures_validation` (``None`` where they cannot be
        computed), and ``cells``.
    """
    from ._validation import futures_validation

    if (size is None) == (regions is None):
        raise ValueError("give the tile size or the regions")
    maps = [read_map(m, nodata) for m in (reference, simulated) + ((original,) if original is not None else ())]
    vals = [m.as_float() for m in maps]
    shape = vals[0].shape
    if any(v.shape != shape for v in vals):
        raise ValueError("the maps differ in shape")
    template = maps[0].template
    rows = []

    def one(sel, info):
        sub = [np.where(sel, v, np.nan) for v in vals]
        r = futures_validation(sub[0], sub[1], sub[2] if original is not None else None)
        row = dict(info)
        row["cells"] = int(np.sum(sel & ~np.any([np.isnan(v) for v in vals], axis=0)))
        for k in _FUTURES_SCALARS:
            if k in r:
                row[k] = r[k]
        for i, c in enumerate(r["classes"]):
            row[f"quantity_class_{c}"] = float(r["quantity"][i])
            row[f"allocation_class_{c}"] = float(r["allocation"][i])
        rows.append(row)

    if size is not None:
        if size < 1:
            raise ValueError("size must be at least 1")
        xs = ys = None
        if template is not None:
            xs, ys = template[template.dims[1]].values, template[template.dims[0]].values
        for r0 in range(0, shape[0], size):
            for c0 in range(0, shape[1], size):
                sel = np.zeros(shape, dtype=bool)
                sel[r0:r0 + size, c0:c0 + size] = True
                info = {"row": r0 // size, "col": c0 // size}
                if xs is not None:
                    info["x"] = float(np.mean(xs[c0:c0 + size]))
                    info["y"] = float(np.mean(ys[r0:r0 + size]))
                one(sel, info)
    else:
        reg = read_map(regions).as_float()
        if reg.shape != shape:
            raise ValueError("the regions map differs in size from the maps")
        for rid in np.unique(reg[~np.isnan(reg)]):
            one(reg == rid, {"region": int(rid) if float(rid).is_integer() else float(rid)})
    return pd.DataFrame(rows)


@dataclass
class Validation:
    """Every comparison of a simulated map with the observed one (:func:`validate`).

    Attributes
    ----------
    three_map : ThreeMapComparison
        Pontius et al. (2011) at several resolutions (``agreement_budget()``,
        ``figure_of_merit()``).
    similarity : pandas.DataFrame
        Fuzzy similarity of the differences per window (Dinamica EGO).
    kappa : dict
        MOLUSCE's kappa, kappa location and histogram, correctness...
    error_budget : pandas.DataFrame
        MOLUSCE's error budget at coarser resolutions.
    futures : dict
        Quantity and allocation disagreement, kappa simulation (FUTURES).
    metrics : pandas.DataFrame or None
        Landscape metrics of the observed and the simulated map side by side.
    """

    three_map: object
    similarity: pd.DataFrame
    kappa: dict
    error_budget: pd.DataFrame
    futures: dict
    metrics: pd.DataFrame | None = None
    settings: dict = field(default_factory=dict)

    def summary(self) -> pd.Series:
        """The headline numbers: figure of merit, kappas, disagreement, similarity at each window."""
        fom = self.three_map.figure_of_merit()["overall"]
        out = {"figure_of_merit": float(fom.iloc[0]), "kappa": self.kappa["kappa"],
               "kappa_location": self.kappa["kappa_location"], "kappa_histogram": self.kappa["kappa_histogram"],
               "kappa_simulation": self.futures.get("kappasimulation"),
               "quantity_disagreement": self.futures["total_quantity"],
               "allocation_disagreement": self.futures["total_allocation"]}
        for _, row in self.similarity.iterrows():
            out[f"similarity_{int(row['window'])}"] = row["minimum"]
        return pd.Series(out, name="value")

    def __repr__(self) -> str:
        s = self.summary()
        return (f"Validation(figure of merit {s['figure_of_merit']:.4f}, kappa {s['kappa']:.4f}, "
                f"similarity {s.iloc[-1]:.4f} at {int(self.similarity['window'].iloc[-1])} cells)")


@threaded
def validate(t0, observed, simulated, *, factors=(1, 2, 4, 8, 16), windows=(1, 3, 5, 7, 9, 11),
             decay: str = "exponential", budget_iterations: int = 4, metrics=None, level: str = "class",
             nodata=None) -> Validation:
    """Compare a simulated map with the observed one in every way the land change tools do.

    Parameters
    ----------
    t0
        The map at the start of the simulated period.
    observed, simulated
        The observed and the simulated end.
    factors
        Resolutions of the three-map comparison (block sides, cells).
    windows, decay
        Windows of the fuzzy similarity of the differences.
    budget_iterations
        Resolutions of MOLUSCE's error budget (each twice the last).
    metrics
        Landscape metrics (``ls.landscape_metrics`` names, at ``level``) to
        put side by side.

    Returns
    -------
    Validation

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> rng = np.random.default_rng(0)
    >>> t0 = rng.integers(1, 4, (40, 40)).astype(float)
    >>> obs = t0.copy(); obs[5:15, 5:15] = 3
    >>> sim = t0.copy(); sim[7:17, 6:16] = 3
    >>> v = ls.simulate.validate(t0, obs, sim, factors=[1, 2, 4], windows=[1, 3, 5])
    >>> bool(v.summary()["similarity_5"] > v.summary()["similarity_1"])
    True
    """
    from ._similarity import similarity_of_differences
    from ._validation import (
        compare_maps,
        error_budget,
        futures_validation,
        three_map_comparison,
    )

    tmc = three_map_comparison(t0, observed, simulated, factors=factors, nodata=nodata)
    sim = similarity_of_differences(t0, observed, simulated, windows=windows, decay=decay, nodata=nodata)
    kappa = compare_maps(observed, simulated, nodata=nodata)
    eb = error_budget(observed, simulated, iterations=budget_iterations, nodata=nodata)
    a, o, s = (read_map(m, nodata).as_float() for m in (t0, observed, simulated))
    fut = futures_validation(o, s, a)
    table = None
    if metrics:
        from ..lsm import landscape_metrics

        parts = []
        for name, m in (("observed", o), ("simulated", s)):
            ints = np.where(np.isnan(m), -9999, m).astype(np.int32)
            t = landscape_metrics(ints, level=level, metric=list(metrics), nodata=-9999)
            parts.append(t.set_index(["level", "class", "id", "metric"])["value"].rename(name))
        table = pd.concat(parts, axis=1).reset_index()
        table["difference"] = table["simulated"] - table["observed"]
    return Validation(tmc, sim, kappa, eb, fut, table,
                      {"factors": list(factors), "windows": list(windows), "decay": decay,
                       "budget_iterations": budget_iterations, "metrics": list(metrics) if metrics else None})
