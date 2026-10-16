# SPDX-License-Identifier: GPL-3.0-or-later
"""Geomorphons: landforms from the lines of sight of every cell (Jasiewicz & Stepinski 2013).

A port of GRASS GIS's ``r.geomorphon`` (8.4) in C++: its landforms, ternary
patterns and measures, bit for bit, with four of its errors corrected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import threaded
from ..cost._transition import _crs_lonlat
from ._grid import read_surface

__all__ = ["GEOMORPHON_FORMS", "Geomorphons", "geomorphons"]

#: The 10 landforms of the geomorphons, by their code in ``Geomorphons.forms``.
GEOMORPHON_FORMS = {1: "flat", 2: "peak", 3: "ridge", 4: "shoulder", 5: "spur", 6: "slope", 7: "hollow",
                    8: "footslope", 9: "valley", 10: "pit"}

_OUTPUTS = {"forms": 1, "ternary": 2, "positive": 4, "negative": 8, "intensity": 16, "exposition": 32,
            "range": 64, "variance": 128, "elongation": 256, "azimuth": 512, "extend": 1024, "width": 2048}
_MEASURES = ("intensity", "exposition", "range", "variance", "elongation", "azimuth", "extend", "width")
_COMPARISONS = {"anglev1": 0, "anglev2": 1, "anglev2_distance": 2}


@dataclass
class Geomorphons(Saveable):
    """The geomorphons of a DEM (:func:`geomorphons`).

    Attributes
    ----------
    forms : xarray.DataArray
        The landform of each cell, 1 to 10 (:data:`GEOMORPHON_FORMS`: flat,
        peak, ridge, shoulder, spur, slope, hollow, footslope, valley, pit);
        NaN where the DEM has no value and on its outer ``skip + 1`` cells.
    ternary : xarray.DataArray
        The ternary pattern: the 8 signs (higher, level, lower) of the
        directions as a base-3 number, the smallest of its rotations and
        reflections (498 distinct codes, 0 to 6560).
    positive, negative : xarray.DataArray
        The directions whose zenith angle exceeds the flatness threshold
        (whose nadir angle is below minus the threshold), as 8 bits, the
        smallest of their rotations (0 to 255).
    intensity, exposition, range, variance, elongation, azimuth, extend, width : xarray.DataArray or None
        With ``measures=True``: the mean height of the cell above the points
        that decide its directions (``intensity``), minus the height of the
        farthest of them in elevation (``exposition``), their range and
        variance of elevation; the shape of the octagon they draw on the
        plane: its elongation, the azimuth of its long axis (degrees), its
        area relative to the octagon of the search radius (``extend``) and
        its width (map units).
    """

    forms: xr.DataArray
    ternary: xr.DataArray
    positive: xr.DataArray
    negative: xr.DataArray
    intensity: xr.DataArray | None = None
    exposition: xr.DataArray | None = None
    range: xr.DataArray | None = None
    variance: xr.DataArray | None = None
    elongation: xr.DataArray | None = None
    azimuth: xr.DataArray | None = None
    extend: xr.DataArray | None = None
    width: xr.DataArray | None = None

    def share(self) -> pd.DataFrame:
        """Cells and share of each landform: ``form`` (code), ``name``, ``cells``, ``share``."""
        v = self.forms.values
        v = v[~np.isnan(v)].astype(np.int64)
        counts = np.bincount(v, minlength=11)[1:11]
        total = max(int(counts.sum()), 1)
        return pd.DataFrame({"form": list(GEOMORPHON_FORMS), "name": list(GEOMORPHON_FORMS.values()),
                             "cells": counts.astype(np.int64), "share": counts / total}).set_index("form")

    def __repr__(self) -> str:
        top = self.share().sort_values("cells", ascending=False).head(3)
        parts = ", ".join(f"{r.name} {r.share:.0%}" for r in top.itertuples())
        return f"Geomorphons({self.forms.shape[0]} x {self.forms.shape[1]}; {parts})"


@threaded
def geomorphons(dem, *, search: float = 3, skip: float = 0, flat: float = 1.0, dist: float = 0.0,
                units: str = "cells", comparison: str = "anglev1", extended: bool = False, measures: bool = False,
                extent=None, crs=None, band=None, nodata=None, _grass_errors: bool = False) -> Geomorphons:
    """Landforms of a DEM by geomorphons (Jasiewicz & Stepinski 2013).

    From each cell, along the 8 directions, the terrain is looked at from
    ``skip + 1`` cells away up to the search radius: the highest angle above
    the horizon (zenith) and the lowest (nadir). A direction is higher (+)
    when its zenith angle wins, lower (-) when the nadir angle does, level
    when neither exceeds the flatness threshold ``flat``. The numbers of +
    and - name the landform (flat, peak, ridge, shoulder, spur, slope,
    hollow, footslope, valley, pit), and the 8 signs, up to rotation and
    reflection, one of 498 patterns. Unlike slope and curvature, which see a
    3 x 3 window, the forms follow the terrain at the scale of the search
    radius: a better covariate of the site's shape for land change,
    species distributions and soils.

    The same forms, patterns and measures as GRASS GIS's ``r.geomorphon``
    (8.4), bit for bit, but for four corrections (see Notes).

    Parameters
    ----------
    dem : str, path, xarray.DataArray or array_like
        The DEM (projected; a single band; NaN or ``nodata`` = missing). It is
        read as float32, as GRASS reads it.
    search : float, default 3
        The outer search radius, in cells (or map units with
        ``units="map"``): ``r.geomorphon``'s ``search``.
    skip : float, default 0
        The inner radius not looked at (``skip``), less than ``search``.
    flat : float, default 1
        The flatness threshold, in degrees.
    dist : float, default 0
        The flatness distance (``dist``): beyond it the threshold falls as
        the angle of the height ``tan(flat) x dist`` (0: none; it must lie
        between ``skip`` and ``search``).
    units : {"cells", "map"}, default "cells"
        The unit of ``search``, ``skip`` and ``dist`` (``"map"``: map units,
        ``r.geomorphon -m``).
    comparison : {"anglev1", "anglev2", "anglev2_distance"}, default "anglev1"
        How the zenith and nadir angles decide a direction
        (``r.geomorphon``'s ``comparison``).
    extended : bool, default False
        The extended correction (``-e``): with a search radius above 10
        cells, a ridge, peak, shoulder or footslope is looked at again within
        half the radius.
    measures : bool, default False
        Also the measures of the geomorphon (intensity, exposition, range,
        variance, elongation, azimuth, extend, width).
    extent, crs, band, nodata
        As in :func:`terrain`.

    Returns
    -------
    Geomorphons

    Notes
    -----
    Four things ``r.geomorphon`` 8.4 does not do as intended, here in the
    correct form: its lines of sight stop one row short of the DEM's last
    row for the cells of its bottom ``search`` rows (a buffer of rows that
    holds the last one out of reach); its binary ``positive`` and
    ``negative`` patterns add the direction numbers instead of setting their
    bits; its extended correction keeps the inner radius in the shorter,
    second search, though it sets it to 0; and the shape of a geomorphon
    whose first direction (NE) has no point is all zeros. Elsewhere the values
    are GRASS's, compared on synthetic and real DEMs.

    Longitude / latitude DEMs are refused: GRASS measures their distances on
    the ellipsoid; reproject the DEM first.

    References
    ----------
    Jasiewicz, J. & Stepinski, T. F. (2013). Geomorphons, a pattern
    recognition approach to classification and mapping of landforms.
    *Geomorphology* 182, 147-156.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> yy, xx = np.mgrid[0:60, 0:60] * 10.0
    >>> hill = 300 * np.exp(-((yy - 300) ** 2 + (xx - 300) ** 2) / 2e4)   # a round hill
    >>> g = ls.surface.geomorphons(hill, search=10, extent=(0, 600, 0, 600))
    >>> ls.surface.GEOMORPHON_FORMS[int(g.forms[30, 30])]                 # its summit
    'peak'
    >>> g                                                                  # the hill's flanks: slopes and spurs
    Geomorphons(60 x 60; slope 61%, spur 27%, footslope 8%)
    >>> g.share().loc[[2, 8], ["name", "cells"]]                          # doctest: +NORMALIZE_WHITESPACE
               name  cells
    form
    2          peak     21
    8     footslope    274
    """
    if units not in ("cells", "map"):
        raise ValueError(f"units must be 'cells' or 'map', got {units!r}")
    if comparison not in _COMPARISONS:
        raise ValueError(f"comparison must be one of {list(_COMPARISONS)}, got {comparison!r}")
    if not flat > 0:
        raise ValueError("the flatness threshold must be greater than 0")
    s = read_surface(dem, extent=extent, crs=crs, band=band, nodata=nodata)
    if _crs_lonlat(s.crs):
        raise ValueError("geomorphons need a projected DEM (distances in map units): reproject it first")
    xmin, xmax, ymin, ymax = (float(v) for v in s.extent)
    ny, nx = s.shape
    ns_res, ew_res = (ymax - ymin) / ny, (xmax - xmin) / nx
    max_res = max(ns_res, ew_res)
    meters = units == "map"
    search_cells = int(search / max_res) if meters else int(search)
    if search_cells < 1:
        raise ValueError("the search radius must cover at least 1 cell")
    skip_cells = int(skip / max_res) if meters else int(skip)
    if skip_cells >= search_cells:
        raise ValueError("the skip radius must be at least 1 cell less than the search radius")
    search_distance = float(search) if meters else ns_res * search_cells
    skip_distance = float(skip) if meters else ns_res * skip_cells
    pi = 4 * math.atan(1.0)
    flat_threshold = flat / (180 / pi)
    flat_distance = float(dist) if meters else ns_res * float(dist)
    flat_height = math.tan(flat_threshold) * flat_distance
    if (0 < flat_distance <= skip_distance) or flat_distance >= search_distance:
        flat_distance = 0.0
    names = ["forms", "ternary", "positive", "negative"] + (list(_MEASURES) if measures else [])
    mask = sum(_OUTPUTS[k] for k in names) | (4096 if _grass_errors else 0)  # tests: r.geomorphon's errors
    z = np.ascontiguousarray(s.values, dtype=np.float32)
    r = _core.surf_geomorphons(z, ymax, xmin, ns_res, ew_res, search_cells, skip_cells, search_distance,
                               flat_threshold, flat_distance, flat_height, max_res, bool(extended),
                               _COMPARISONS[comparison], mask)
    out = {}
    for k in names:
        v = np.asarray(r[k])
        if k == "forms":
            v = np.where(v == 0, np.nan, v.astype(np.float64))
        elif k in ("ternary", "positive", "negative"):
            v = np.where(v < 0, np.nan, v.astype(np.float64))
        else:
            v = v.astype(np.float64)
        out[k] = s.raster(v, k)
    return Geomorphons(**out)
