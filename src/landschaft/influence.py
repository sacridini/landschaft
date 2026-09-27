# SPDX-License-Identifier: GPL-3.0-or-later
"""Influence Zones and Proximity (GuidosToolbox *Distance -> Influence Zones / Proximity*).

The foreground objects of at least ``min_area`` pixels (8-connected, holes
filled, optionally reduced to their core) divide the map into zones of
influence: the watershed of the distance to the objects. GTB draws the
objects (coloured by id), the watershed lines between zones and, for
Influence Zones, a background buffer around the objects. Proximity measures
the gaps between neighbouring objects along the watershed lines and their
connected area gain (CAG).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import _core
from .io import load_raster
from .validation import InvalidInputError, validate_mspa_input

__all__ = [
    "INFLUENCE_CLASSES",
    "InfluenceZonesResult",
    "ProximityResult",
    "influence_zones",
    "proximity",
]

# IDL MORPH_DISTANCE(..., NEIGHBOR_SAMPLING=3) is a quasi-Euclidean chamfer (see influence.cpp).
_METRIC = "quasi-euclidean"

INFLUENCE_CLASSES: dict[int, str] = {
    0: "Zone (background) / hole in an object",
    4: "Watershed line (zone boundary)",
    12: "Object (Proximity)",
    59: "Object (Influence Zones colours: 59, 5, 9, 17, 33, 65, 18, 56, 55, 57, 58, 101 by id mod 12)",
    80: "Buffer boundary",
    160: "Buffer",
    129: "Missing",
    165: "Watershed line with proximity <= prox_max",
    176: "Foreground not part of an object (too small, or its core buffer)",
    193: "Watershed line at the minimum proximity",
}


def _prepare(data, min_area: int, core_buffer: int) -> tuple[np.ndarray, xr.DataArray | None, str]:
    name = ""
    if isinstance(data, (str, Path)):
        name = str(data)
        data = load_raster(data, validate=None)
    template = data if isinstance(data, xr.DataArray) else None
    try:
        validate_mspa_input(data)
    except InvalidInputError as err:
        raise InvalidInputError(str(err).replace("MSPA-compliant", "Influence Zones/Proximity input")) from None
    if min_area < 1:
        raise ValueError(f"min_area must be >= 1, got {min_area}")
    if core_buffer < 0:
        raise ValueError(f"core_buffer must be >= 0, got {core_buffer}")
    arr = np.ascontiguousarray(data.values if template is not None else data, dtype=np.uint8)
    if not name and template is not None:
        name = str(template.attrs.get("source", template.name or ""))
    return arr, template, name


def _wrap(values: np.ndarray, template, name: str, attrs: dict):
    if template is None:
        return values
    da = template.copy(data=values)
    da.name = name
    da.attrs = dict(template.attrs) | attrs
    return da


@dataclass(frozen=True)
class InfluenceZonesResult:
    """Result of :func:`influence_zones`.

    Attributes
    ----------
    classes
        ``uint8`` map in GTB's viewport codes (see :data:`INFLUENCE_CLASSES`).
    ids
        ``int32`` object id on the object pixels (holes included); 0 zones,
        -1 holes, -2 foreground that is not part of an object, -3 watershed
        lines, -4 missing, -5 buffer boundary, -6 buffer.
    zones
        ``int32`` zone (watershed basin) of every pixel, 0 on the lines.
    object_area
        Area of object ``k`` (without its holes) at index ``k - 1``.
    """

    classes: xr.DataArray | np.ndarray
    ids: xr.DataArray | np.ndarray
    zones: xr.DataArray | np.ndarray
    object_area: np.ndarray
    min_area: int
    core_buffer: int
    buffer: int

    @property
    def n_objects(self) -> int:
        return int(self.object_area.size)


@dataclass(frozen=True)
class ProximityResult:
    """Result of :func:`proximity`.

    Attributes
    ----------
    classes, ids, zones, object_area
        As in :class:`InfluenceZonesResult` (objects are all coded 12; lines at
        or below ``prox_max`` 165, at the minimum proximity 193).
    distance
        ``int32`` proximity map: ``round(2 * distance to the nearest object)``,
        on lines between more than two zones scaled by their number.
    cag
        ``int64`` connected area gain on the watershed lines:
        ``round(sqrt(2 * sum_{i<j} a_i * a_j))`` over the adjacent objects.
    lines
        One row per watershed pixel between zones (GTB's proximity CSV):
        1-based column/row, number of adjacent zones, connector length,
        CAG and the adjacent zone ids with their object areas.
    prox_min, prox_min_count
        Minimum proximity on the watershed lines and the number of pixels with it.
    """

    classes: xr.DataArray | np.ndarray
    ids: xr.DataArray | np.ndarray
    zones: xr.DataArray | np.ndarray
    distance: xr.DataArray | np.ndarray
    cag: xr.DataArray | np.ndarray
    object_area: np.ndarray
    lines: pd.DataFrame
    prox_min: int | None
    prox_min_count: int
    min_area: int
    core_buffer: int
    prox_max: int
    source: str = field(default="", repr=False)

    @property
    def n_objects(self) -> int:
        return int(self.object_area.size)

    def csv_text(self, source: str | None = None) -> str:
        """GTB's ``<name>_proximity_<min_area>_<core_buffer>_<prox_max>.csv``."""
        targets = " Core-obj." if self.core_buffer > 0 else " FG-obj."
        prox_min = "undef." if self.prox_min is None else str(self.prox_min)
        title = f"Min.{targets} size: {self.min_area} - {self.n_objects} objects - prox_min: {prox_min}"
        if self.core_buffer > 0:
            title += f" - core-buffer: {self.core_buffer} pixels"
        rows = [f"{self.source if source is None else source} - {title}", "WS-X,WS-Y, #nbr, clength, CAG, IDs(area),"]
        for r in self.lines.itertuples(index=False):
            idarea = "  ".join(f"{i}-{a}" for i, a in zip(r.ids, r.areas))
            rows.append(f"{r.x},{r.y},{r.nbr},{r.clength},{r.cag},{idarea}")
        return "\n".join(rows) + "\n"

    def to_csv(self, path: str | Path, source: str | None = None) -> Path:
        """Write GTB's proximity CSV."""
        path = Path(path)
        path.write_text(self.csv_text(source))
        return path


def influence_zones(
    data,
    min_area: int = 5000,
    *,
    core_buffer: int = 0,
    buffer: int = 0,
    n_threads: int = 0,
) -> InfluenceZonesResult:
    """Zones of influence of the foreground objects (GTB Distance -> Influence Zones).

    Parameters
    ----------
    data
        2-D ``uint8`` map (array, DataArray or raster path) coded 0 = missing,
        1 = background, 2 = foreground.
    min_area
        Minimum object area [pixels] (GTB default 5000). Smaller objects are
        not objects; they are shown as code 176.
    core_buffer
        Reduce the objects to their core: drop the foreground within
        ``core_buffer`` pixels of the object edge before measuring areas (GTB
        "CoreZone"; 0 = off).
    buffer
        Width [pixels] of a background buffer drawn around the objects (0 = none).
    n_threads
        OpenMP threads (0: all cores but one).
    """
    arr, template, _ = _prepare(data, min_area, core_buffer)
    if buffer < 0:
        raise ValueError(f"buffer must be >= 0, got {buffer}")
    out = _core.compute_influence(arr, int(min_area), int(core_buffer), int(buffer), False, _METRIC, n_threads)
    attrs = {"landschaft_product": "INFLUENCE_ZONES", "min_area": int(min_area), "core_buffer": int(core_buffer),
             "buffer": int(buffer), "n_objects": int(out["n_objects"])}
    return InfluenceZonesResult(
        classes=_wrap(out["classes"], template, "influence_zones", attrs),
        ids=_wrap(out["ids"], template, "influence_ids", attrs | {"landschaft_product": "INFLUENCE_IDS"}),
        zones=_wrap(out["zones"], template, "influence_zone_labels", attrs | {"landschaft_product": "INFLUENCE_LABELS"}),
        object_area=np.asarray(out["object_area"][1:], dtype=np.int64),
        min_area=int(min_area),
        core_buffer=int(core_buffer),
        buffer=int(buffer),
    )


def proximity(
    data,
    min_area: int = 5000,
    *,
    core_buffer: int = 0,
    prox_max: int = 0,
    n_threads: int = 0,
) -> ProximityResult:
    """Proximity between neighbouring objects (GTB Distance -> Proximity).

    Parameters are those of :func:`influence_zones`, with ``prox_max``: the
    watershed lines with a proximity up to ``prox_max`` (or the minimum
    proximity, if larger) are marked 165 (GTB default 0).
    """
    arr, template, name = _prepare(data, min_area, core_buffer)
    if prox_max < 0:
        raise ValueError(f"prox_max must be >= 0, got {prox_max}")
    out = _core.compute_influence(arr, int(min_area), int(core_buffer), int(prox_max), True, _METRIC, n_threads)
    attrs = {"landschaft_product": "PROXIMITY", "min_area": int(min_area), "core_buffer": int(core_buffer),
             "prox_max": int(prox_max), "n_objects": int(out["n_objects"])}
    has_lines = "cag" in out
    cag = out["cag"] if has_lines else np.zeros(arr.shape, dtype=np.int64)
    lines = pd.DataFrame(
        [(x, y, nbr, cl, c, list(ids), list(areas)) for x, y, nbr, cl, c, ids, areas in out.get("lines", [])],
        columns=["x", "y", "nbr", "clength", "cag", "ids", "areas"],
    )
    return ProximityResult(
        classes=_wrap(out["classes"], template, "proximity", attrs),
        ids=_wrap(out["ids"], template, "proximity_ids", attrs | {"landschaft_product": "PROXIMITY_IDS"}),
        zones=_wrap(out["zones"], template, "proximity_zone_labels", attrs | {"landschaft_product": "PROXIMITY_LABELS"}),
        distance=_wrap(out["distance"], template, "proximity_distance", attrs | {"landschaft_product": "PROXIMITY_DISTANCE"}),
        cag=_wrap(cag, template, "proximity_cag", attrs | {"landschaft_product": "PROXIMITY_CAG"}),
        object_area=np.asarray(out["object_area"][1:], dtype=np.int64),
        lines=lines,
        prox_min=int(out["prox_min"]) if has_lines else None,
        prox_min_count=int(out["prox_min_count"]) if has_lines else 0,
        min_area=int(min_area),
        core_buffer=int(core_buffer),
        prox_max=int(prox_max),
        source=name,
    )
