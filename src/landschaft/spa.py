# SPDX-License-Identifier: GPL-3.0-or-later
"""Simplified Pattern Analysis (GuidosToolbox *Pattern -> Morphological -> SPA2/3/5/6*).

Simplified alternatives to MSPA for the 8-connected foreground, with a fixed
edge width of one pixel:

- SPA2: contiguous foreground and small linear features (SLF: foreground
  removed by a 3x3 opening);
- SPA3: core, core-opening and margin;
- SPA5: core, edge, perforation, margin (and core-opening);
- SPA6: SPA5 plus islets (foreground objects without core).

The maps and statistics are identical to GTB's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import _core
from .io import load_raster
from .validation import validate_mspa_input

__all__ = ["SPA_CLASSES", "SPAResult", "spa"]

SPA_CLASSES: dict[int, str] = {
    17: "Core (SPA2: Contiguous)",
    1: "Margin (SPA2: Small Linear Features)",
    3: "Edge",
    5: "Perforation",
    9: "Islet",
    100: "Core-Opening",
    0: "Other background",
    129: "Missing",
}

_ROWS = {
    2: [("Contiguous", 17), ("SLF", 1)],
    3: [("Core", 17), ("Margin", 1)],
    5: [("Core", 17), ("Edge", 3), ("Perforation", 5), ("Margin", 1)],
    6: [("Core", 17), ("Islet", 9), ("Edge", 3), ("Perforation", 5), ("Margin", 1)],
}


@dataclass(frozen=True)
class SPAResult:
    """Result of :func:`spa`.

    Attributes
    ----------
    map
        ``uint8`` map in GTB's codes (see :data:`SPA_CLASSES`).
    stats
        Foreground classes with area [pixels] and their share of the data
        area and of the foreground (GTB's SPA report).
    counts
        Number of 8-connected objects per class (core-openings: 4-connected)
        and of foreground objects, as GTB reports them.
    porosity
        ``100 - contiguous / (contiguous + core-openings) * 100``.

    Examples
    --------
    >>> import landschaft as ls
    >>> result = ls.spa(ls.load_raster(ls.example_data("input")), variant=6)
    >>> result.map.shape
    (1000, 1000)
    """

    map: xr.DataArray | np.ndarray
    variant: int
    stats: pd.DataFrame
    counts: dict[str, int]
    areas: dict[str, int]
    porosity: float

    def summary(self) -> str:
        """Report in the layout of GTB's ``spaN.txt`` (areas and proportions)."""
        a, name = self.areas, f"SPA{self.variant}"
        line, rule = "=" * 80, "-" * 80
        out = [f"{name}: 8-connected Foreground, summary analysis", "", line,
               "           Category              Area [pixels]: ", line]
        for i, row in self.stats.iterrows():
            label = ("  " if i == 0 else "+ ") + row["class"] + ": "
            out.append(f"{label:>20}{row['area']:>22}")
        if self.variant in (5, 6):
            out.append(f"(Contiguous = Core + Edge + Perforation: {a['contiguous']})")
        out += [rule, f"{'= Foreground Total: ':>20}{a['foreground']:>22}", ""]
        if self.variant != 2:
            out += [f"{'Core-Opening: ':>20}{a['core_opening']:>22}",
                    f"{'+ Other BG: ':>20}{a['other_background']:>22}", rule,
                    f"{'= Background Total: ':>20}{a['background']:>22}", ""]
        out += [f"{'Data Area: ':>20}{a['data']:>22}", f"{'+ Missing: ':>20}{a['missing']:>22}", rule,
                f"{'= Image Area Total: ':>20}{a['image']:>22}", "", line,
                "           Category    Proportion [%]: ", line]
        for _, row in self.stats.iterrows():
            out.append(f"{row['class'] + '/Data: ':>20}{row['pct_data']:11.4f}")
        out.append(rule)
        for _, row in self.stats.iterrows():
            out.append(f"{row['class'] + '/FG: ':>20}{row['pct_fg']:11.4f}")
        out += ["", "Objects: " + ", ".join(f"{k} {v}" for k, v in self.counts.items())]
        if self.variant != 2:
            out.append(f"Porosity [%]: {self.porosity:.4f}")
        return "\n".join(out)


def spa(data, variant: int = 5, *, n_threads: int = 0) -> SPAResult:
    """Simplified pattern analysis: the habitat in 2, 3, 5 or 6 classes.

    A coarser, faster summary than :func:`mspa`:

    - ``2``: contiguous habitat (core, edge, perforation) and small linear
      features (margin);
    - ``3``: core, edge (with perforation) and margin;
    - ``5``: core, edge, perforation and margin;
    - ``6``: core, edge, perforation, margin and islet.

    Parameters
    ----------
    data
        2-D ``uint8`` habitat map (array, DataArray or raster path) coded
        0 missing, 1 background, 2 habitat.
    variant
        2, 3, 5 (default) or 6.
    n_threads
        Threads (``None`` / 0: all cores but one).

    Returns
    -------
    SPAResult
        The class map, class areas, object counts and porosity.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> result = ls.spa(forest, variant=5)
    >>> list(result.stats["class"])
    ['Core', 'Edge', 'Perforation', 'Margin']
    >>> round(result.porosity, 2)
    1.42
    >>> print(result.summary())  # doctest: +SKIP
    """
    if variant not in (2, 3, 5, 6):
        raise ValueError(f"variant must be 2, 3, 5 or 6, got {variant}")
    if isinstance(data, (str, Path)):
        data = load_raster(data, validate=None)
    validate_mspa_input(data)
    template = data if isinstance(data, xr.DataArray) else None
    arr = np.ascontiguousarray(data.values if template is not None else data, dtype=np.uint8)
    out = _core.compute_spa(arr, variant, n_threads)
    m = out["map"]

    h = np.bincount(m.ravel(), minlength=256)
    image = int(arr.size)
    missing = int((arr == 0).sum())
    data_area = image - missing
    holes, other_bg = int(h[100]), int(h[0])
    background = other_bg + holes
    foreground = data_area - background
    contiguous = int(h[17] + h[3] + h[5])
    internal = contiguous + holes
    porosity = 100.0 - contiguous / internal * 100.0 if internal else float("nan")

    rows = [(label, code, int(h[code])) for label, code in _ROWS[variant]]
    stats = pd.DataFrame(
        {
            "class": [r[0] for r in rows],
            "code": [r[1] for r in rows],
            "area": [r[2] for r in rows],
            "pct_data": [r[2] * 100.0 / data_area for r in rows],
            "pct_fg": [r[2] * 100.0 / foreground if foreground else float("nan") for r in rows],
        }
    )
    names = {"nr_fg": "foreground", "nr_cores": "core", "nr_margin": "margin", "nr_holes": "core_opening",
             "nr_edge": "edge", "nr_perf": "perforation", "nr_islet": "islet", "nr_contiguous": "contiguous"}
    counts = {names[k]: int(out[k]) for k in names if int(out[k]) >= 0}
    if variant == 2:
        counts = {"foreground": counts["foreground"], "contiguous": counts["core"], "slf": counts["margin"]}
    areas = {"image": image, "missing": missing, "data": data_area, "foreground": foreground,
             "background": background, "core_opening": holes, "other_background": other_bg,
             "contiguous": contiguous}

    result_map = m
    if template is not None:
        result_map = template.copy(data=m)
        result_map.name = f"spa{variant}"
        result_map.attrs = dict(template.attrs) | {"landschaft_product": f"SPA{variant}", "spa_variant": variant}
    return SPAResult(map=result_map, variant=variant, stats=stats, counts=counts, areas=areas, porosity=porosity)
