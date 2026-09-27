# SPDX-License-Identifier: GPL-3.0-or-later
"""FAD Multiscale change (GuidosToolbox *Change -> FAD Multiscale*).

Compares two FAD-MS analyses (times A and B): at every observation scale and
for the multiscale map, the pixels are cross-tabulated between the
fragmentation classes of A and B (background, rare, patchy, transitional,
dominant, interior, intact), as in GTB's ``FAD_change_<scale>.txt/.csv``
tables, together with the change of the average FAD (FAD_AV, AVCON).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import _core
from .fad import FADMultiscaleResult, fad_multiscale
from .io import load_raster

__all__ = ["FADChangeScale", "FADMultiscaleChangeResult", "fad_multiscale_change"]

FAD_CHANGE_CLASSES = ("Background", "Rare", "Patchy", "Transitional", "Dominant", "Interior", "Intact")


@dataclass(frozen=True)
class FADChangeScale:
    """FAD change A -> B at one observation scale (one GTB ``FAD_change_<scale>`` table).

    Attributes
    ----------
    matrix
        7x7 pixel counts; rows are the class at time A (``A0-Background`` to
        ``A6-Intact``), columns the class at time B. Pixels missing at either
        time are left out.
    percent
        The 6x6 foreground part of ``matrix`` with its diagonal set to 0, in %
        of the ``changed`` pixels (rows A1..A6, columns B1..B6).
    same_class, changed
        Foreground pixels (at both times) on / off the diagonal.
    frag_decrease_pct, frag_increase_pct
        % of the changed pixels above / below the diagonal, i.e. moving to a
        less / more fragmented class (NaN when nothing changed).
    """

    scale: int | str
    matrix: pd.DataFrame
    percent: pd.DataFrame
    fg_area_a: int
    fg_area_b: int
    forcom: int
    same_class: int
    changed: int
    frag_decrease_pct: float
    frag_increase_pct: float

    @property
    def net_change(self) -> int:
        """Net foreground cover change A -> B [pixels]."""
        return self.fg_area_b - self.fg_area_a


@dataclass(frozen=True)
class FADMultiscaleChangeResult:
    """Result of :func:`fad_multiscale_change`.

    ``scales`` maps every window size and ``"mscale"`` (the multiscale map) to
    a :class:`FADChangeScale`. ``averages`` holds GTB's FAD_AV (mean FAD of the
    foreground) and AVCON (FAD_AV x foreground / non-missing area) per scale at
    times A and B, and their differences.
    """

    scales: dict[int | str, FADChangeScale]
    averages: pd.DataFrame

    def summary(self, scale: int | str = "mscale") -> str:
        """Report of one scale in the layout of GTB's ``FAD_change_<scale>.txt``."""
        sc = self.scales[scale]
        windows = [w for w in self.scales if w != "mscale"]
        label = ("average over " + "-".join(f"{w}x{w}" for w in windows)) if scale == "mscale" else f"{scale}x{scale}"
        names = list(FAD_CHANGE_CLASSES)
        lines = [
            f"FAD: Fragmentation class change from A -> B at observation scale: {label} pixels",
            "=" * 142,
            "# pixels A->B  :" + "".join(f"{f'B{i}-{c}':>18}" for i, c in enumerate(names)),
        ]
        for i, c in enumerate(names):
            lines.append(f"{f'A{i}-{c}':<15}:" + "".join(f"{v:>18}" for v in sc.matrix.iloc[i]))
        lines += [
            "  ",
            f"Foreground cover at time A [pixels]: {sc.fg_area_a}",
            f"Foreground cover at time B [pixels]: {sc.fg_area_b}",
            f"Net foreground cover change (A->B) [pixels]: {sc.net_change}",
            "=" * 89,
            "Observation scale:  " + "".join(f"{str(w):>12}" for w in self.averages.index),
            "Diff: FAD_AV(A->B): " + "".join(f"{v:>12.4f}" for v in self.averages["delta_fad_av"]),
            "Diff: AVCON(A->B):  " + "".join(f"{v:>12.4f}" for v in self.averages["delta_avcon"]),
            "  ",
            f"Change matrix constrained to the {sc.changed} pixels in different fragmentation classes [%]: ",
            "Fragmentation decrease (=connectivity increase) - above the matrix diagonal [%]: "
            f"{sc.frag_decrease_pct:.6g}",
            "Fragmentation increase (=connectivity decrease) - below the matrix diagonal [%]: "
            f"{sc.frag_increase_pct:.6g}",
            "A->B [%]     :  " + "".join(f"{f'B{i}-{c}':>16}" for i, c in enumerate(names) if i),
        ]
        for i, c in enumerate(names[1:], 1):
            lines.append(f"{f'A{i}-{c}':<15}:" + "".join(f"{v:>16.6g}" for v in sc.percent.iloc[i - 1]))
        return "\n".join(lines)


def _fad_ms_maps(source, window_sizes: list[int], n_threads: int) -> dict[int | str, np.ndarray]:
    """The FAD-MS maps of one time: every single scale and "mscale"."""
    if isinstance(source, (str, Path)) and str(source).endswith("_fad_mscale.tif"):
        # GTB FAD-MS output: <name>_fad_mscale.tif next to <name>_fad_<window>.tif
        prefix = str(source)[: -len("mscale.tif")]
        maps: dict[int | str, np.ndarray] = {
            w: load_raster(f"{prefix}{w}.tif", validate=None).values for w in window_sizes
        }
        maps["mscale"] = load_raster(source, validate=None).values
        return maps
    if isinstance(source, FADMultiscaleResult):
        if [int(w) for w in source.window_sizes] != window_sizes:
            raise ValueError(f"FAD-MS result has window sizes {source.window_sizes}, expected {window_sizes}")
    else:
        if isinstance(source, (str, Path)):
            source = load_raster(source, validate=None)
        source = fad_multiscale(source, window_sizes, threads=n_threads)
    maps = {int(w): np.asarray(source.single_scales[w]) for w in source.window_sizes}
    maps["mscale"] = np.asarray(source.multiscale_map)
    return maps


def fad_multiscale_change(
    a,
    b,
    *,
    window_sizes: Sequence[int] = (7, 13, 27, 81, 243),
    n_threads: int = 0,
) -> FADMultiscaleChangeResult:
    """Fragmentation class change between two FAD-MS analyses (GTB Change -> FAD Multiscale).

    Parameters
    ----------
    a, b
        Times A and B. Each is a :class:`~landschaft.FADMultiscaleResult`, an
        FAD input map (array, DataArray or raster path coded 0/1/2 and
        optionally 3/4, analysed with :func:`~landschaft.fad_multiscale`), or
        the path of a GTB ``*_fad_mscale.tif`` (its single-scale maps are read
        from the same directory).
    window_sizes
        FAD-MS observation scales (GTB: 7, 13, 27, 81, 243).
    n_threads
        OpenMP threads (0: all cores but one).

    Returns
    -------
    FADMultiscaleChangeResult
        One transition table per observation scale and for the multiscale map.
    """
    window_sizes = [int(w) for w in window_sizes]
    maps_a = _fad_ms_maps(a, window_sizes, n_threads)
    maps_b = _fad_ms_maps(b, window_sizes, n_threads)
    shape_a, shape_b = maps_a["mscale"].shape, maps_b["mscale"].shape
    if shape_a != shape_b:
        raise ValueError(f"FAD maps A and B have different dimensions: {shape_a} and {shape_b}")
    if bool((maps_a["mscale"] == 106).any()) != bool((maps_b["mscale"] == 106).any()):
        raise ValueError(
            "One of the input images has non-fragmenting background pixels (input 4 / FAD 106). A FAD change "
            "analysis is only meaningful if either both or none of the two images have them."
        )

    keys: list[int | str] = [*window_sizes, "mscale"]
    labels_a = [f"A{i}-{c}" for i, c in enumerate(FAD_CHANGE_CLASSES)]
    labels_b = [f"B{i}-{c}" for i, c in enumerate(FAD_CHANGE_CLASSES)]
    scales: dict[int | str, FADChangeScale] = {}
    fad_av: dict[str, list[float]] = {"a": [], "b": []}
    for key in keys:
        counts = _core.compute_fad_change_counts(
            np.ascontiguousarray(maps_a[key], dtype=np.uint8),
            np.ascontiguousarray(maps_b[key], dtype=np.uint8),
            n_threads,
        )
        matrix = counts["matrix"]
        fad_av["a"].append(counts["fg_sum_a"] / counts["fg_area_a"])
        fad_av["b"].append(counts["fg_sum_b"] / counts["fg_area_b"])
        if key == "mscale":  # GTB's AVCON factor: foreground / non-missing area
            factor_a = counts["fg_area_a"] / counts["ru_area_a"]
            factor_b = counts["fg_area_b"] / counts["ru_area_b"]
        fg = matrix[1:, 1:].astype(np.float64)
        same = int(np.trace(matrix[1:, 1:]))
        np.fill_diagonal(fg, 0.0)
        changed = int(fg.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            perc = fg / changed * 100
        above = float(np.triu(perc).sum())  # B class above the A class: less fragmented
        scales[key] = FADChangeScale(
            scale=key,
            matrix=pd.DataFrame(matrix, index=labels_a, columns=labels_b),
            percent=pd.DataFrame(perc, index=labels_a[1:], columns=labels_b[1:]),
            fg_area_a=int(counts["fg_area_a"]),
            fg_area_b=int(counts["fg_area_b"]),
            forcom=int(counts["forcom"]),
            same_class=same,
            changed=changed,
            frag_decrease_pct=above,
            frag_increase_pct=float(perc.sum()) - above,
        )

    av_a, av_b = fad_av["a"], fad_av["b"]
    con_a = [v * factor_a for v in av_a]
    con_b = [v * factor_b for v in av_b]
    table = pd.DataFrame(
        {"fad_av_a": av_a, "fad_av_b": av_b, "delta_fad_av": np.subtract(av_b, av_a),
         "avcon_a": con_a, "avcon_b": con_b, "delta_avcon": np.subtract(con_b, con_a)},
        index=pd.Index(keys, name="scale"),
    )
    return FADMultiscaleChangeResult(scales=scales, averages=table)
