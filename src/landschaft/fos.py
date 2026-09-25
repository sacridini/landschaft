# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragmentation at a Fixed Observation Scale (GuidosToolbox *Fragmentation -> FOS*).

For every foreground pixel, the connectivity of the foreground in a moving
window of the chosen size, by one of three methods:

- **FAD**, foreground area density (share of foreground in the window);
- **FED**, foreground edge density (GraySpatCon "clustering": mean of the
  pixel pairs of the window);
- **FAC**, foreground area clustering (share of foreground-foreground pairs;
  binary maps only).

The input is either a binary map (0 missing, 1 background, 2 foreground,
3/4 special background) or a grayscale map (0-100 % cover, 255 missing,
103/104 special background) whose foreground is the cover at or above a
threshold. Results are reported in 5 or 6 fragmentation classes, as by GTB.
The maps and statistics are those of GTB's batch FOS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import xarray as xr

from . import _core
from .io import load_raster
from .validation import InvalidInputError, validate_fad_input

__all__ = ["FOSResult", "fos"]

FOS_CLASSES_6 = ("Rare", "Patchy", "Transitional", "Dominant", "Interior", "Intact")
FOS_CLASSES_5 = FOS_CLASSES_6[:5]


@dataclass(frozen=True)
class FOSResult:
    """Result of :func:`fos`.

    Attributes
    ----------
    map
        ``uint8`` FOS map: 0-100 connectivity [%] of the foreground pixels,
        101 background, 102 missing, 105 special background, 106
        non-fragmenting special background.
    stats
        Share [%] of the foreground in each fragmentation class.
    histogram
        Share [%] of the foreground pixels with each value 0..100.
    """

    map: xr.DataArray | np.ndarray
    stats: pd.DataFrame
    histogram: np.ndarray
    method: str
    classes: int
    window_size: int
    grayscale: bool
    gray_threshold: int | None
    connectivity: int
    pixel_resolution: float
    fg_area: int
    reporting_area: int
    fg_area_pct: float
    n_patches: int
    average_patch_size: float
    eca: float
    coherence: float
    coherence_ru: float
    average: float
    avcon: float

    @property
    def observation_scale_ha(self) -> float:
        """Window area in hectares."""
        return (self.pixel_resolution * self.window_size) ** 2 / 10000.0

    def summary(self) -> str:
        """Report in the layout of GTB's ``fos.txt``."""
        m = self.method
        ranges = {"Rare": "[0 - 9]", "Patchy": "[10 - 39]", "Transitional": "[40 - 59]", "Dominant": "[60 - 89]",
                  "Interior": "[90 - 99]" if self.classes == 6 else "[90 - 100]", "Intact": ": 100"}
        inp = f"Grayscale (FG threshold: {self.gray_threshold})" if self.grayscale else "Binary"
        lines = [
            "Fragmentation analysis using Fixed Observation Scale (FOS)",
            "(Fragmentation is complementary to Connectivity: Fragmentation = 100% - Connectivity)",
            "Method options: FAD - FG Area Density; FED - FG Edge Density; FAC - FG Area Clustering; ",
            "=" * 80,
            "FOS parameter settings:", inp,
            f"Foreground connectivity: {self.connectivity}-connectivity",
            f"FOS-type selected: {m}_{self.classes}class",
            f"Pixel resolution [m]: {self.pixel_resolution:g}",
            f"Window size [pixels]: {self.window_size}",
            f"Observation scale [(window size * pixel resolution)^2]: {self.observation_scale_ha:g} hectares <-> "
            f"{self.observation_scale_ha * 2.47105:g} acres",
            "=" * 80,
            "Proportion [%] of foreground area in foreground cover class:",
        ]
        for _, row in self.stats.iterrows():
            rng = ranges[row["class"]]
            label = f"{row['class']} ({m}-pixel value{' within' if rng.startswith('[') else ''}: {rng.lstrip(': ')}): "
            lines.append(f"{label:>55}{row['percent']:11.4f}")
        lines += [
            "=" * 80, "A) Image summary:", "=" * 80,
            f"Reporting unit area [pixels]: {self.reporting_area}",
            f"Foreground area [pixels]: {self.fg_area}",
            f"Foreground area [%]: {self.fg_area_pct:.6g}",
            f"Number of foreground patches: {self.n_patches}",
            f"Average foreground patch size [pixels]: {self.average_patch_size:.6g}",
            "=" * 80, "B) Reporting levels", "=" * 80,
            f"- Average {m} at WS{self.window_size} [%]: {self.average:.6g}",
            f"- ECA (Equivalent Connected Area) [pixels]: {self.eca:.6g}",
            f"- COH (Coherence = ECA/ECA_max*100) [%]: {self.coherence:.6g}",
            f"- AVCON (average connectivity) at WS{self.window_size} [%]: {self.avcon:.6g}",
            f"- COH_ru (ECA/Reporting unit area*100) [%]: {self.coherence_ru:.6g}",
        ]
        return "\n".join(lines)


def _class_shares(im: np.ndarray, fg_area: int, classes: int) -> list[float]:
    """GTB's proportions [%] of the foreground per class (rare .. intact)."""
    def share(mask) -> float:
        return float(np.count_nonzero(mask)) / fg_area * 100.0

    rare = share(im < 10)
    patchy = share((im >= 10) & (im < 40))
    transitional = share((im >= 40) & (im < 60))
    dominant = share((im >= 60) & (im < 90))
    if classes == 6:
        return [rare, patchy, transitional, dominant, share((im >= 90) & (im < 100)), share(im == 100)]
    return [rare, patchy, transitional, dominant, share((im >= 90) & (im <= 100))]


def fos(
    data,
    window_size: int = 27,
    method: str = "FAD",
    classes: int = 6,
    *,
    connectivity: int = 8,
    grayscale: bool | None = None,
    gray_threshold: int = 30,
    pixel_resolution: float = 25.0,
    n_threads: int = 0,
) -> FOSResult:
    """Fragmentation at a Fixed Observation Scale (GTB FOS).

    Parameters
    ----------
    data
        Binary map (0 missing, 1 background, 2 foreground, 3/4 special
        background) or grayscale map (0-100, 255 missing, 103/104 special
        background); array, DataArray or raster path.
    window_size
        Side of the square moving window [pixels], odd (GTB offers 3-243).
    method
        ``"FAD"``, ``"FED"`` or ``"FAC"`` (binary maps only).
    classes
        5 or 6 reporting classes (6 separates intact = 100 from interior).
    connectivity
        Foreground connectivity for the patch statistics (8 or 4).
    grayscale
        Treat the map as grayscale; default: when its maximum exceeds 4 (GTB).
    gray_threshold
        Grayscale maps: pixels with a value of at least this are foreground.
    pixel_resolution
        Pixel size [m], for the observation scale in hectares.
    n_threads
        OpenMP threads (0 = all).
    """
    method = method.upper()
    if method not in ("FAD", "FED", "FAC"):
        raise ValueError(f"method must be 'FAD', 'FED' or 'FAC', got {method!r}")
    if classes not in (5, 6):
        raise ValueError(f"classes must be 5 or 6, got {classes}")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")
    if isinstance(data, (str, Path)):
        data = load_raster(data, validate=None)
    template = data if isinstance(data, xr.DataArray) else None
    arr = np.ascontiguousarray(data.values if template is not None else data)
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise InvalidInputError(f"FOS needs a 2-D uint8 map, got {arr.dtype} with shape {arr.shape}")
    if grayscale is None:
        grayscale = int(arr.max()) > 4
    if window_size >= min(arr.shape):
        raise ValueError(f"window_size {window_size} must be smaller than the map ({arr.shape})")

    if grayscale:
        if method == "FAC":
            raise ValueError("FAC is available for binary maps only (as in GTB)")
        missing, special, special_nf = arr == 255, arr == 103, arr == 104
        background = arr < gray_threshold
        foreground = (arr >= gray_threshold) & (arr < 101)
        if not background.any():
            raise InvalidInputError(f"the map has no background (values below the threshold {gray_threshold})")
        if not foreground.any():
            raise InvalidInputError(f"the map has no foreground (values {gray_threshold}-100)")
        bad = ~(missing | special | special_nf | (arr <= 100))
        if bad.any():
            raise InvalidInputError("grayscale FOS maps must hold 0-100, 103, 104 and 255 only")
    else:
        validate_fad_input(arr)
        missing, special, special_nf = arr == 0, arr == 3, arr == 4
        background, foreground = arr == 1, arr == 2
    reporting_area = int(arr.size - np.count_nonzero(missing))
    fg_area = int(np.count_nonzero(foreground))

    # patch statistics of the (binary) foreground
    structure = np.ones((3, 3), bool) if connectivity == 8 else None
    labels, n_patches = ndi.label(foreground, structure=structure)
    areas = np.bincount(labels.ravel())[1:].astype(np.float64)
    eca = math.sqrt(float((areas**2).sum()))
    eca_max = float(areas.sum())
    # GTB: 100.0/n_ruarea*farea in single precision (farea is a float total for grayscale maps)
    pct = np.float32(100.0) / np.float32(reporting_area)
    if grayscale:
        fg_area_pct = float(pct * np.float32(fg_area))
        aps = float(np.float32(fg_area) / np.float32(n_patches)) if n_patches else float("nan")
    else:
        fg_area_pct = float(pct) * fg_area
        aps = fg_area / n_patches if n_patches else float("nan")

    # the connectivity map
    work = arr.copy()
    if grayscale:
        work[special] = 0
        work[special_nf] = 255
        metric = 1 if method == "FAD" else 52
        im = _core.compute_grayspatcon_fos(work, metric, window_size, n_threads)
    else:
        work[special_nf] = 0  # non-fragmenting background: missing for the density
        if method == "FAD":
            im = _core.compute_fad_density(work, window_size, False, n_threads)
        elif method == "FAC":
            im = _core.compute_clustering_fac(work, window_size, False, n_threads)
        else:  # FED on 0 background, 100 foreground, 255 missing
            gsc = np.zeros_like(arr)
            gsc[foreground] = 100
            gsc[missing | special_nf] = 255
            im = _core.compute_grayspatcon_fos(gsc, 52, window_size, n_threads)
    im = np.array(im, dtype=np.uint8, copy=True)

    fg_values = im[foreground]
    average = float(fg_values.mean(dtype=np.float64))
    avcon = average * fg_area / reporting_area
    im[special] = 105
    im[special_nf] = 106
    im[missing] = 102
    im[background] = 101

    hist = np.bincount(fg_values, minlength=256)[:101]
    histogram = hist.astype(np.float32) / np.float32(fg_area) * np.float32(100.0)
    names = FOS_CLASSES_6 if classes == 6 else FOS_CLASSES_5
    stats = pd.DataFrame({"class": list(names), "percent": _class_shares(im, fg_area, classes)})

    if template is not None:
        out = template.copy(data=im)
        out.name = f"fos_{method.lower()}_{window_size}"
        out.attrs = dict(template.attrs) | {
            "landschaft_product": "FOS", "fos_method": method, "fos_classes": classes, "window_size": window_size,
            "avcon": avcon,
        }
    else:
        out = im
    return FOSResult(
        map=out, stats=stats, histogram=histogram, method=method, classes=classes, window_size=window_size,
        grayscale=bool(grayscale), gray_threshold=int(gray_threshold) if grayscale else None,
        connectivity=connectivity, pixel_resolution=float(pixel_resolution), fg_area=fg_area,
        reporting_area=reporting_area, fg_area_pct=fg_area_pct, n_patches=int(n_patches), average_patch_size=aps,
        eca=eca, coherence=eca / eca_max * 100.0 if eca_max else float("nan"),
        coherence_ru=eca / reporting_area * 100.0, average=average, avcon=avcon,
    )
