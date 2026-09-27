# SPDX-License-Identifier: GPL-3.0-or-later
"""Change Analysis and Morphological Change Detection (MCD).

This module implements GuidosToolbox (GTB) change analysis tools:
- :func:`simple_change`: 4-class binary difference (Stable BG, Gain, Loss, Stable FG).
- :func:`morphological_change`: Morphological Change Detection (MCD) with geodesic
  reconstruction of core changes, closing, boundary chatter segregation (spurious),
  and landscape elasticity calculation.
- :func:`fos_change`: Fragmentation (FOS) change analysis, delta FOS continuous map,
  6x6 transition matrix, and fragmentation shift percentages.

References
----------
Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object analysis.
    European Journal of Remote Sensing, 50(1), 352-361. doi:10.1080/22797254.2017.1330650
Riitters, K. et al. (2000). Global-scale patterns of forest fragmentation.
    Conservation Ecology, 4(2).
"""

from __future__ import annotations

from ._threads import threaded

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from . import _core
from .io import load_raster

__all__ = [
    "simple_change",
    "morphological_change",
    "fos_change",
    "SimpleChangeResult",
    "MorphologicalChangeResult",
    "FOSChangeResult",
]

_FOS_CLASSES = [
    "Rare (0-9%)",
    "Patchy (10-39%)",
    "Transitional (40-59%)",
    "Dominant (60-89%)",
    "Interior (90-99%)",
    "Intact (100%)",
]


@dataclass(frozen=True)
class SimpleChangeResult:
    """Result of :func:`simple_change`.

    Attributes
    ----------
    change_map : xarray.DataArray or numpy.ndarray
        Byte raster with pixel values:
        11 = Stable Background, 12 = Gain, 21 = Loss, 22 = Stable Foreground, 255 = Missing.
    stable_bg : int
        Pixels of stable background.
    gain : int
        Pixels of foreground gain.
    loss : int
        Pixels of foreground loss.
    stable_fg : int
        Pixels of stable foreground.
    missing : int
        Pixels missing at one or both times.
    total_valid : int
        Total valid pixels (stable_bg + gain + loss + stable_fg).

    Examples
    --------
    >>> import landschaft as ls
    >>> before = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> after = before.copy()
    >>> cut = after.values[600:700, 100:260]
    >>> cut[cut == 2] = 1                                   # a clear-cut
    >>> plantation = after.values[250:330, 420:520]
    >>> plantation[plantation == 1] = 2                     # a new plantation
    >>> print(ls.simple_change(before, after).summary())  # doctest: +ELLIPSIS
    Simple Change Analysis Summary (A -> B)
    ...
    Net Change [pixels]           : +1,329
    ...
    """

    change_map: xr.DataArray | np.ndarray
    stable_bg: int
    gain: int
    loss: int
    stable_fg: int
    missing: int
    total_valid: int

    @property
    def net_change(self) -> int:
        """Net foreground change in pixels (gain - loss).
        """
        return self.gain - self.loss

    def summary(self) -> str:
        """Text summary of the change.

        Returns
        -------
        str
        """
        fg_a = self.stable_fg + self.loss
        fg_b = self.stable_fg + self.gain
        lines = [
            "Simple Change Analysis Summary (A -> B)",
            "=======================================",
            f"Foreground Area at T1 [pixels]: {fg_a:,}",
            f"Foreground Area at T2 [pixels]: {fg_b:,}",
            f"Net Change [pixels]           : {self.net_change:+,}",
            "",
            "Change Breakdown:",
            "-----------------",
            f"Stable Background (11)        : {self.stable_bg:,}",
            f"Foreground Gain   (12)        : {self.gain:,}",
            f"Foreground Loss   (21)        : {self.loss:,}",
            f"Stable Foreground (22)        : {self.stable_fg:,}",
            f"Missing / Nodata  (255)       : {self.missing:,}",
            f"Total Valid Pixels            : {self.total_valid:,}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class MorphologicalChangeResult:
    """Result of :func:`morphological_change`.

    Attributes
    ----------
    change_map : xarray.DataArray or numpy.ndarray
        Byte raster with pixel values:
        11 = Stable BG, 12 = MCD Gain, 21 = MCD Loss, 22 = Stable FG,
        176 = Spurious boundary chatter, 255 = Missing.
    fg_area_t1 : int
        Total foreground area at time 1.
    fg_area_t2 : int
        Total foreground area at time 2.
    net_change : int
        Net foreground change (T2 - T1).
    stable_bg : int
        Stable background pixels.
    stable_fg : int
        Stable foreground pixels.
    mcd_gain : int
        Structural foreground gain pixels.
    mcd_loss : int
        Structural foreground loss pixels.
    spurious : int
        Boundary chatter / spurious change pixels.
    missing : int
        Missing pixels at one or both times.
    elasticity : float
        Landscape elasticity index (ratio of core interior change to total change).
    """

    change_map: xr.DataArray | np.ndarray
    fg_area_t1: int
    fg_area_t2: int
    net_change: int
    stable_bg: int
    stable_fg: int
    mcd_gain: int
    mcd_loss: int
    spurious: int
    missing: int
    elasticity: float

    def summary(self) -> str:
        """Text summary of the change.

        Returns
        -------
        str
        """
        spurious_gain_b = self.fg_area_t2 - self.stable_fg - self.mcd_gain
        lines = [
            "Morphological Change Detection (MCD) Summary",
            "============================================",
            f"Foreground Area at T1 [pixels]: {self.fg_area_t1:,}",
            f"Foreground Area at T2 [pixels]: {self.fg_area_t2:,}",
            f"Net Change [pixels]           : {self.net_change:+,}",
            f"Landscape Elasticity          : {self.elasticity:.4f}" if np.isfinite(self.elasticity) else "Landscape Elasticity          : NA",
            "",
            "MCD Class Breakdown:",
            "--------------------",
            f"MCD-Loss     (21)             : {self.mcd_loss:,}",
            f"MCD-Stable   (22)             : {self.stable_fg:,}",
            f"MCD-Gain     (12)             : {self.mcd_gain:,}",
            f"MCD-Spurious (176)            : {self.spurious:,}",
            f"Stable BG    (11)             : {self.stable_bg:,}",
            f"Missing      (255)            : {self.missing:,}",
            "",
            "Boundary Verification:",
            f"Spurious Gain at T2 [pixels]  : {spurious_gain_b:,}",
            f"FG Cover T2 = Stable + Gain + Spurious Gain: {self.stable_fg + self.mcd_gain + spurious_gain_b:,}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class FOSChangeResult:
    """Result of :func:`fos_change`.

    Attributes
    ----------
    change_map : xarray.DataArray or numpy.ndarray
        Byte raster with pixel values:
        0..200 = Delta FOS shifted by +100 (100 = No change),
        250 = Foreground Gain, 251 = Foreground Loss, 252 = Stable BG,
        253 = Water, 254 = Missing.
    fg_gain : int
        Pixels transitioning from BG to FG.
    fg_loss : int
        Pixels transitioning from FG to BG.
    stable_bg : int
        Pixels remaining background at both times.
    forcom : int
        Common foreground pixels evaluated for fragmentation shift.
    diag_same_class : int
        Pixels maintaining the exact same FOS density class.
    diff_class : int
        Pixels shifting between FOS density classes.
    frag_decrease_pct : float
        Percentage of shifting pixels with decreased fragmentation (increased connectivity).
    frag_increase_pct : float
        Percentage of shifting pixels with increased fragmentation (decreased connectivity).
    matrix_6x6 : pandas.DataFrame
        6x6 transition matrix across FOS classes (T1 rows -> T2 columns).
    """

    change_map: xr.DataArray | np.ndarray
    fg_gain: int
    fg_loss: int
    stable_bg: int
    forcom: int
    diag_same_class: int
    diff_class: int
    frag_decrease_pct: float
    frag_increase_pct: float
    matrix_6x6: pd.DataFrame

    def summary(self) -> str:
        """Text summary of the change.

        Returns
        -------
        str
        """
        lines = [
            "Fragmentation (FOS) Change Summary",
            "==================================",
            f"Common Foreground [pixels]    : {self.forcom:,}",
            f"Foreground Gain (250)         : {self.fg_gain:,}",
            f"Foreground Loss (251)         : {self.fg_loss:,}",
            f"Stable Background (252)       : {self.stable_bg:,}",
            "",
            "Fragmentation Shifts on Common Foreground:",
            "------------------------------------------",
            f"Unchanged Density Class       : {self.diag_same_class:,}",
            f"Shifted Density Class         : {self.diff_class:,}",
            f"Fragmentation Decrease        : {self.frag_decrease_pct:.2f}% (above diagonal)",
            f"Fragmentation Increase        : {self.frag_increase_pct:.2f}% (below diagonal)",
            "",
            "FOS 6x6 Transition Matrix (Rows: T1, Columns: T2):",
            "--------------------------------------------------",
            self.matrix_6x6.to_string(),
        ]
        return "\n".join(lines)


def _prepare_inputs(
    t1: xr.DataArray | np.ndarray | str | Path,
    t2: xr.DataArray | np.ndarray | str | Path,
) -> tuple[np.ndarray, np.ndarray, xr.DataArray | None]:
    is_xr = False
    xr_template = None

    if isinstance(t1, (str, Path)):
        t1 = load_raster(t1)
    if isinstance(t2, (str, Path)):
        t2 = load_raster(t2)

    if isinstance(t1, xr.DataArray):
        is_xr = True
        xr_template = t1
    elif isinstance(t2, xr.DataArray):
        is_xr = True
        xr_template = t2

    arr1 = np.asarray(t1)
    arr2 = np.asarray(t2)

    if arr1.ndim != 2 or arr2.ndim != 2:
        raise ValueError("Both input rasters must be 2-dimensional")
    if arr1.shape != arr2.shape:
        raise ValueError(f"Input rasters must have identical dimensions, got {arr1.shape} and {arr2.shape}")

    u8_1 = np.ascontiguousarray(arr1, dtype=np.uint8)
    u8_2 = np.ascontiguousarray(arr2, dtype=np.uint8)
    return u8_1, u8_2, xr_template if is_xr else None


def simple_change(
    t1: xr.DataArray | np.ndarray | str | Path,
    t2: xr.DataArray | np.ndarray | str | Path,
) -> SimpleChangeResult:
    """Pixel-by-pixel change of a habitat map between two dates.

    Every pixel gets one of four transitions: stable background (11), gain (12),
    loss (21) or stable foreground (22); 255 where either date is missing.

    Parameters
    ----------
    t1, t2
        Maps of the two dates, coded 0 missing, 1 background, 2 foreground
        (DataArray, array or path), on the same grid.

    Returns
    -------
    SimpleChangeResult

    See Also
    --------
    morphological_change : separates structural change from edge noise.

    Examples
    --------
    >>> import landschaft as ls
    >>> before = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> after = before.copy()
    >>> cut = after.values[600:700, 100:260]
    >>> cut[cut == 2] = 1                                   # a clear-cut
    >>> plantation = after.values[250:330, 420:520]
    >>> plantation[plantation == 1] = 2                     # a new plantation
    >>> change = ls.simple_change(before, after)
    >>> change.gain, change.loss, change.net_change
    (6839, 5510, 1329)
    """
    u8_1, u8_2, xr_tpl = _prepare_inputs(t1, t2)
    res = _core.compute_simple_change(u8_1, u8_2)

    out_map = res["change_map"]
    if xr_tpl is not None:
        out_map = xr.DataArray(
            out_map,
            coords=xr_tpl.coords,
            dims=xr_tpl.dims,
            attrs={
                **xr_tpl.attrs,
                "landschaft_product": "SIMPLE_CHANGE",
                "stable_bg": res["stable_bg"],
                "gain": res["gain"],
                "loss": res["loss"],
                "stable_fg": res["stable_fg"],
                "missing": res["missing"],
            },
            name="simple_change",
        )

    return SimpleChangeResult(
        change_map=out_map,
        stable_bg=int(res["stable_bg"]),
        gain=int(res["gain"]),
        loss=int(res["loss"]),
        stable_fg=int(res["stable_fg"]),
        missing=int(res["missing"]),
        total_valid=int(res["total_valid"]),
    )


@threaded
def morphological_change(
    t1: xr.DataArray | np.ndarray | str | Path,
    t2: xr.DataArray | np.ndarray | str | Path,
) -> MorphologicalChangeResult:
    """Change of a habitat map that separates real structural change from edge noise (GTB's Morphological Change Detection).

    Small shifts along patch edges (e.g. from slightly different
    classifications of the two dates) are "spurious" (176); gains and losses
    that change the shape of the habitat are kept (12, 21). The method finds the
    cores of each date (pixels more than 2 pixels from the edge), reconstructs
    them by geodesic dilation and closes the result with a 3 x 3 window.

    The landscape elasticity compares the change of the core interior with the
    total change.

    Parameters
    ----------
    t1, t2
        Maps of the two dates, coded 0 missing, 1 background, 2 foreground
        (DataArray, array or path).
    n_threads
        Threads (0 or ``None``: all cores but one).

    Returns
    -------
    MorphologicalChangeResult

    Examples
    --------
    >>> import landschaft as ls
    >>> before = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> after = before.copy()
    >>> cut = after.values[600:700, 100:260]
    >>> cut[cut == 2] = 1                                   # a clear-cut
    >>> plantation = after.values[250:330, 420:520]
    >>> plantation[plantation == 1] = 2                     # a new plantation
    >>> change = ls.morphological_change(before, after)
    >>> change.mcd_gain, change.mcd_loss, change.spurious
    (7348, 5504, 395)
    """
    u8_1, u8_2, xr_tpl = _prepare_inputs(t1, t2)
    res = _core.compute_morphological_change(u8_1, u8_2)

    out_map = res["change_map"]
    if xr_tpl is not None:
        out_map = xr.DataArray(
            out_map,
            coords=xr_tpl.coords,
            dims=xr_tpl.dims,
            attrs={
                **xr_tpl.attrs,
                "landschaft_product": "MORPHOLOGICAL_CHANGE",
                "fg_area_t1": res["fg_area_t1"],
                "fg_area_t2": res["fg_area_t2"],
                "net_change": res["net_change"],
                "mcd_gain": res["mcd_gain"],
                "mcd_loss": res["mcd_loss"],
                "spurious": res["spurious"],
                "elasticity": res["elasticity"],
            },
            name="mcd_change",
        )

    return MorphologicalChangeResult(
        change_map=out_map,
        fg_area_t1=int(res["fg_area_t1"]),
        fg_area_t2=int(res["fg_area_t2"]),
        net_change=int(res["net_change"]),
        stable_bg=int(res["stable_bg"]),
        stable_fg=int(res["stable_fg"]),
        mcd_gain=int(res["mcd_gain"]),
        mcd_loss=int(res["mcd_loss"]),
        spurious=int(res["spurious"]),
        missing=int(res["missing"]),
        elasticity=float(res["elasticity"]),
    )


@threaded
def fos_change(
    t1: xr.DataArray | np.ndarray | str | Path,
    t2: xr.DataArray | np.ndarray | str | Path,
) -> FOSChangeResult:
    """Change of fragmentation between two dates (GTB's FOS change).

    Compares two :func:`fos` maps: for the foreground at both dates, the shift
    of its fragmentation class (a 6 x 6 transition matrix from Rare to Intact),
    and whether it became more or less fragmented; gains and losses of
    foreground are marked separately.

    Parameters
    ----------
    t1, t2
        FOS maps of the two dates (``fos(...).map``: 0-100 foreground density,
        101 background, 102 missing, 105 water).
    n_threads
        Threads (0 or ``None``: all cores but one).

    Returns
    -------
    FOSChangeResult

    Examples
    --------
    >>> import landschaft as ls
    >>> before = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> after = before.copy()
    >>> cut = after.values[600:700, 100:260]
    >>> cut[cut == 2] = 1                                   # a clear-cut
    >>> plantation = after.values[250:330, 420:520]
    >>> plantation[plantation == 1] = 2                     # a new plantation
    >>> shift = ls.fos_change(ls.fos(before, window_size=27).map, ls.fos(after, window_size=27).map)
    >>> round(shift.frag_increase_pct, 2), round(shift.frag_decrease_pct, 2)
    (54.71, 45.29)
    >>> shift.matrix_6x6.shape
    (6, 6)
    """
    u8_1, u8_2, xr_tpl = _prepare_inputs(t1, t2)
    res = _core.compute_fos_change(u8_1, u8_2)

    out_map = res["change_map"]
    if xr_tpl is not None:
        out_map = xr.DataArray(
            out_map,
            coords=xr_tpl.coords,
            dims=xr_tpl.dims,
            attrs={
                **xr_tpl.attrs,
                "landschaft_product": "FOS_CHANGE",
                "fg_gain": res["fg_gain"],
                "fg_loss": res["fg_loss"],
                "stable_bg": res["stable_bg"],
                "forcom": res["forcom"],
                "frag_decrease_pct": res["frag_decrease_pct"],
                "frag_increase_pct": res["frag_increase_pct"],
            },
            name="fos_change",
        )

    mat_raw = res["matrix_6x6"]
    df_matrix = pd.DataFrame(mat_raw, index=_FOS_CLASSES, columns=_FOS_CLASSES)

    return FOSChangeResult(
        change_map=out_map,
        fg_gain=int(res["fg_gain"]),
        fg_loss=int(res["fg_loss"]),
        stable_bg=int(res["stable_bg"]),
        forcom=int(res["forcom"]),
        diag_same_class=int(res["diag_same_class"]),
        diff_class=int(res["diff_class"]),
        frag_decrease_pct=float(res["frag_decrease_pct"]),
        frag_increase_pct=float(res["frag_increase_pct"]),
        matrix_6x6=df_matrix,
    )
