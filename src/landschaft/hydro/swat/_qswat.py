# SPDX-License-Identifier: GPL-3.0-or-later
"""From the landscape to the model: what QSWAT+ does before the SWAT+ Editor.

Delineation (filled DEM, D8 directions, streams by an area threshold, stream links,
one subbasin, channel and landscape unit per link), HRUs (land use x soil x slope band
in each landscape unit, then QSWAT+'s HRU selection) and the routing, written as
QSWAT+ 4 writes its project database: the ``gis_*`` tables the SWAT+ Editor imports,
with the soils of ``soils_sol``/``soils_sol_layer``. The methods follow QSWAT+3 (C.
George, GPL-2.0-or-later; ``hrus.py``, ``QSWATTopology.py``, ``dataInC.pyx``); the
terrain analysis is landschaft's own (``ls.hydro``) instead of TauDEM's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from ... import _core
from ..._threads import threaded
from ...surface._grid import read_surface
from .._basins import _down, _link_table, _steps, snap_outlets
from .._flow import _cells, _d8, _read, fill

__all__ = ["SwatWatershed", "watershed"]

# QSWAT+ parameters.py
WIDTH_MULTIPLIER, WIDTH_EXPONENT = 1.29, 0.6
DEPTH_MULTIPLIER, DEPTH_EXPONENT = 0.13, 0.4
_METHODS = ("dominant_hru", "dominant", "filter", "area")

_USERSOIL_LAYER = ("SOL_Z", "SOL_BD", "SOL_AWC", "SOL_K", "SOL_CBN", "CLAY", "SILT", "SAND", "ROCK", "SOL_ALB",
                   "USLE_K", "SOL_EC")
_LAYER_COLUMNS = ("dp", "bd", "awc", "soil_k", "carbon", "clay", "silt", "sand", "rock", "alb", "usle_k", "ec",
                  "caco3", "ph")


@dataclass
class SwatWatershed:
    """A watershed divided as QSWAT+ divides it, ready for the SWAT+ Editor's import.

    Made by :func:`watershed`; :func:`from_tables` turns it into a SWAT+ project.

    Attributes
    ----------
    tables : dict
        QSWAT+'s project tables, as DataFrames with its columns: ``gis_subbasins``,
        ``gis_channels``, ``gis_lsus``, ``gis_hrus``, ``gis_aquifers``,
        ``gis_deep_aquifers``, ``gis_points``, ``gis_routing``, ``soils_sol`` and
        ``soils_sol_layer``.
    subbasins : xarray.DataArray
        The subbasin (= channel, SWAT+ numbering) of each cell, 0 outside the
        watershed, NaN where the DEM is missing.
    hrus : xarray.DataArray
        The HRU of each cell (``gis_hrus`` id), NaN for cells whose HRU was removed by
        the HRU selection (their area went to the HRUs kept) and outside.
    channels : xarray.DataArray
        The channel of each stream cell, 0 elsewhere.
    """

    tables: dict
    subbasins: xr.DataArray
    hrus: xr.DataArray
    channels: xr.DataArray
    notes: list = field(default_factory=list)

    def __repr__(self) -> str:
        t = self.tables
        return (f"SwatWatershed({len(t['gis_subbasins'])} subbasins, {len(t['gis_hrus'])} HRUs, "
                f"{t['gis_subbasins']['area'].sum():.1f} ha)")


def _lookup(table, what: str) -> dict:
    """{raster value: name} from a dict or a two-column table (value, name)."""
    if isinstance(table, dict):
        items = list(table.items())
    else:
        df = pd.DataFrame(table)
        if df.shape[1] < 2:
            raise ValueError(f"the {what} lookup needs two columns: raster value and name")
        items = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
    out = {}
    for k, v in items:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out[int(k)] = str(v).strip()
    if not out:
        raise ValueError(f"the {what} lookup is empty")
    return out


def _soil_tables(soil_table, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """soils_sol and soils_sol_layer of the soils used, from a usersoil table.

    ``soil_table``: ArcSWAT/QSWAT+'s ``usersoil`` layout (``SNAM``, ``NLAYERS``,
    ``HYDGRP``, ``SOL_ZMX``, ``ANION_EXCL``, ``SOL_CRK``, ``TEXTURE``, then
    ``SOL_Z1``... ``SOL_EC1``... per layer, optionally ``SOL_CAL1``... and
    ``SOL_PH1``...), or SWAT+'s own: one row per layer with ``name``, ``hyd_grp``,
    ``dp_tot``, ``anion_excl``, ``perc_crk``, ``texture`` and the layer columns ``dp``,
    ``bd``, ``awc``, ``soil_k``, ``carbon``, ``clay``, ``silt``, ``sand``, ``rock``,
    ``alb``, ``usle_k``, ``ec``, ``caco3``, ``ph``.
    """
    df = pd.DataFrame(soil_table)
    upper = {c.upper(): c for c in df.columns}
    soils, layers = [], []
    if "SNAM" in upper:
        col = lambda name: upper.get(name)  # noqa: E731
        by_name = {str(r[col("SNAM")]).strip(): r for _, r in df.iterrows()}
        for sid, name in enumerate(names, 1):
            if name not in by_name:
                raise ValueError(f"soil {name!r} is not in the soil table")
            r = by_name[name]
            soils.append({"id": sid, "name": name, "hyd_grp": str(r[col("HYDGRP")]).strip(),
                          "dp_tot": float(r[col("SOL_ZMX")]), "anion_excl": float(r[col("ANION_EXCL")]),
                          "perc_crk": float(r[col("SOL_CRK")]),
                          "texture": None if col("TEXTURE") is None else str(r[col("TEXTURE")]).strip(),
                          "description": None})
            for k in range(1, int(r[col("NLAYERS")]) + 1):
                lay = {"soil_id": sid, "layer_num": k}
                for src, dst in zip(_USERSOIL_LAYER, _LAYER_COLUMNS):
                    lay[dst] = float(r[col(f"{src}{k}")])
                for src, dst in (("SOL_CAL", "caco3"), ("SOL_PH", "ph")):
                    c = col(f"{src}{k}")
                    lay[dst] = None if c is None or pd.isna(r[c]) else float(r[c])
                layers.append(lay)
    elif "NAME" in upper:
        name_col = upper["NAME"]
        for sid, name in enumerate(names, 1):
            rows = df[df[name_col].astype(str).str.strip() == name]
            if rows.empty:
                raise ValueError(f"soil {name!r} is not in the soil table")
            r = rows.iloc[0]
            g = lambda c: r[upper[c.upper()]]  # noqa: E731
            soils.append({"id": sid, "name": name, "hyd_grp": str(g("hyd_grp")).strip(), "dp_tot": float(g("dp_tot")),
                          "anion_excl": float(g("anion_excl")), "perc_crk": float(g("perc_crk")),
                          "texture": str(g("texture")).strip() if "TEXTURE" in upper else None,
                          "description": None})
            for k, (_, lr) in enumerate(rows.iterrows(), 1):
                lay = {"soil_id": sid, "layer_num": k}
                for c in _LAYER_COLUMNS:
                    v = lr[upper[c.upper()]] if c.upper() in upper else None
                    lay[c] = None if v is None or pd.isna(v) else float(v)
                layers.append(lay)
    else:
        raise ValueError("the soil table is neither a usersoil table (SNAM, NLAYERS, ...) nor SWAT+'s (name, dp, ...)")
    soils_df = pd.DataFrame(soils)
    layers_df = pd.DataFrame(layers)
    layers_df.insert(0, "id", np.arange(1, len(layers_df) + 1))
    return soils_df, layers_df


def _slope_band(percent: np.ndarray, limits) -> np.ndarray:
    """QSWAT+'s slopeIndex: the first band whose upper limit is above the slope."""
    out = np.full(percent.shape, len(limits), dtype=np.int64)
    for i in range(len(limits) - 1, -1, -1):
        out[percent < limits[i]] = i
    return out


def _slope_range(i: int, limits) -> str:
    """QSWAT+'s slopeRange label, e.g. ``0-9999``."""
    lo = 0 if i == 0 else limits[i - 1]
    hi = 9999 if i == len(limits) else limits[i]
    fmt = lambda v: str(int(v)) if float(v).is_integer() else str(v)  # noqa: E731
    return f"{fmt(lo)}-{fmt(hi)}"


def getSlsubbsn(mean_slope: float) -> int:  # noqa: N802 (QSWAT+'s name)
    """QSWATUtils.getSlsubbsn: the average slope length (m) from the mean slope (m/m)."""
    if mean_slope < 0.01:
        return 120
    if mean_slope < 0.02:
        return 100
    if mean_slope < 0.03:
        return 90
    if mean_slope < 0.05:
        return 60
    return 30


class _Cell:
    """CellData of QSWAT+: totals of an HRU (or of a whole LSU)."""

    __slots__ = ("count", "area", "elev", "slope", "x", "y")

    def __init__(self, count, area, elev, slope, x, y):
        self.count, self.area, self.elev, self.slope, self.x, self.y = count, area, elev, slope, x, y

    def multiply(self, f: float) -> None:
        self.count = int(self.count * f + 0.5)
        self.area *= f
        self.slope *= f
        self.elev *= f
        self.x *= f
        self.y *= f


class _Lsu:
    """LSUData of QSWAT+ (no landscapes, no water bodies)."""

    def __init__(self):
        self.count = 0
        self.area = 0.0
        self.elev = self.slope = self.x = self.y = 0.0
        self.css_area = 0.0                   # cropSoilSlopeArea
        self.hru: dict[int, _Cell] = {}       # hruMap
        self.numbers: dict = {}               # cropSoilSlopeNumbers: crop -> soil -> slope -> hru
        self.orig_crop: dict = {}
        self.orig_soil: dict = {}
        self.orig_slope: dict = {}
        self.far_distance = -1.0
        self.far_elevation = 0.0

    def remove(self, hru, crop, soil, slope) -> None:
        del self.hru[hru]
        del self.numbers[crop][soil][slope]
        if not self.numbers[crop][soil]:
            del self.numbers[crop][soil]
            if not self.numbers[crop]:
                del self.numbers[crop]

    def redistribute(self, f: float) -> None:
        for c in self.hru.values():
            c.multiply(f)

    def crop_area(self, crop) -> float:
        return sum(self.hru[h].area for sl in self.numbers[crop].values() for h in sl.values())

    def crop_soil_areas(self, crop) -> dict:
        return {s: sum(self.hru[h].area for h in sl.values()) for s, sl in self.numbers[crop].items()}

    def crop_soil_area(self, crop, soil) -> float:
        return sum(self.hru[h].area for h in self.numbers[crop][soil].values())

    def slope_areas(self, crop, soil) -> dict:
        return {sl: self.hru[h].area for sl, h in self.numbers[crop][soil].items()}

    def dominant_hru(self):
        best, out = 0.0, (-1, -1, -1)
        for crop, ss in self.numbers.items():
            for soil, sl in ss.items():
                for slope, h in sl.items():
                    if self.hru[h].area > best:
                        best, out = self.hru[h].area, (crop, soil, slope)
        return out


def _dominant_key(m: dict):
    best, key = 0.0, -1
    for k, v in m.items():
        if v > best:
            best, key = v, k
    return key


def _select(lsu: _Lsu, method: str, thresholds, min_area: float, area_is_ha: bool, exempt: set) -> None:
    """QSWAT+'s HRU selection in one landscape unit (hrus.py)."""
    if method in ("dominant_hru", "dominant"):
        if method == "dominant_hru":
            crop, soil, slope = lsu.dominant_hru()
        else:
            crop = _dominant_key(lsu.orig_crop)
            soil = _dominant_key(lsu.orig_soil)
            slope = _dominant_key(lsu.orig_slope)
        if crop < 0:
            raise ValueError("a landscape unit has no land use data")
        lsu.hru = {1: _Cell(lsu.count, lsu.area, lsu.elev, lsu.slope, lsu.x, lsu.y)}
        lsu.numbers = {crop: {soil: {slope: 1}}}
        return
    # redistributeNodataAndWater: cells without land use, soil or slope data
    lost = lsu.area - lsu.css_area
    if lsu.area > lost > 0:
        lsu.redistribute(lsu.area / (lsu.area - lost))
    if method == "area":
        threshold = min_area * 10000 if area_is_ha else lsu.css_area * min_area / 100
        count = len(lsu.hru)
        hrus_area = lsu.area
        while True:
            m = (0, 0, 0, 0)
            m_area = threshold
            for crop, ss in lsu.numbers.items():
                if crop in exempt:
                    continue
                for soil, sl in ss.items():
                    for slope, h in sl.items():
                        if lsu.hru[h].area < m_area:
                            m_area, m = lsu.hru[h].area, (h, crop, soil, slope)
            if m_area >= threshold:
                break
            redistribute = 0.0
            done = False
            if count > 1:
                lsu.remove(*m)
                count -= 1
                redistribute = m_area
            else:
                done = True
            if redistribute > 0:
                if hrus_area - redistribute == 0:
                    raise ValueError("a landscape unit has no HRUs left")
                lsu.redistribute(hrus_area / (hrus_area - redistribute))
            if done:
                break
        return
    # filter by land use, soil and slope percentages (removeSmallHRUsByThresholdPercent)
    min_crop, min_soil, min_slope = thresholds
    crop_areas = lsu.orig_crop
    hrus_area = lsu.area
    min_crop_area = lsu.css_area * min_crop / 100
    if not any(c in exempt for c in lsu.numbers):
        min_crop_area = min(min_crop_area, max(crop_areas.values(), default=0.0))
    redistribute = 0.0
    for crop, area in crop_areas.items():
        if crop not in exempt and area < min_crop_area:
            redistribute += area
            for soil, sl in list(lsu.numbers[crop].items()):
                for slope, h in list(sl.items()):
                    lsu.remove(h, crop, soil, slope)
    if redistribute > 0:
        if hrus_area - redistribute == 0:
            raise ValueError("a landscape unit has no land use data")
        lsu.redistribute(hrus_area / (hrus_area - redistribute))
    for crop, ss in lsu.numbers.items():
        crop_area = lsu.crop_area(crop)
        soil_areas = lsu.crop_soil_areas(crop)
        min_area_ = min(crop_area * min_soil / 100, max(soil_areas.values()))
        removed = 0.0
        for soil, sl in list(ss.items()):
            if soil_areas[soil] < min_area_:
                removed += soil_areas[soil]
                for slope, h in list(sl.items()):
                    lsu.remove(h, crop, soil, slope)
        if removed > 0:
            if crop_area - removed == 0:
                raise ValueError("a land use has no soil data")
            f = crop_area / (crop_area - removed)
            for sl in ss.values():
                for h in sl.values():
                    lsu.hru[h].multiply(f)
    for crop, ss in lsu.numbers.items():
        for soil, sl in ss.items():
            soil_area = sum(lsu.hru[h].area for h in sl.values())
            slope_areas = lsu.slope_areas(crop, soil)
            min_area_ = min(soil_area * min_slope / 100, max(slope_areas.values()))
            removed = 0.0
            for slope, h in list(sl.items()):
                if slope_areas[slope] < min_area_:
                    removed += slope_areas[slope]
                    lsu.remove(h, crop, soil, slope)
            if removed > 0:
                if soil_area - removed == 0:
                    raise ValueError("a land use and soil have no slope data")
                f = soil_area / (soil_area - removed)
                for h in sl.values():
                    lsu.hru[h].multiply(f)


def _transformer(crs):
    from pyproj import CRS, Transformer

    return Transformer.from_crs(CRS.from_user_input(crs), CRS.from_epsg(4326), always_xy=True)


@threaded
def watershed(dem, landuse, soils, *, landuse_lookup, soil_lookup, soil_table, stream_threshold: float | None = None,
              outlets=None, snap: float | None = None, slope_bands=(), hru_method: str = "dominant_hru",
              thresholds=(0.0, 0.0, 0.0), min_area: float = 0.0, area_in_ha: bool = True, exempt=()) -> SwatWatershed:
    """Divide a watershed into subbasins, channels and HRUs, as QSWAT+ does.

    The DEM is filled (:func:`ls.hydro.fill`), drains along D8 directions, and every
    cell whose drainage area reaches ``stream_threshold`` is a stream. Each stream link
    (:func:`ls.hydro.stream_links`, split at the ``outlets``) is a channel with its
    subbasin (:func:`ls.hydro.watersheds`) and one landscape unit (QSWAT+ without
    landscapes); each subbasin has a shallow aquifer, each watershed outlet a deep one.
    HRUs are the combinations of land use, soil and slope band in a landscape unit,
    selected by ``hru_method``. Channel width and depth come from the drainage area
    (QSWAT+'s ``1.29 A^0.6`` and ``0.13 A^0.4``, A in km2).

    Parameters
    ----------
    dem
        The elevations (m), in a projected CRS (or longitude / latitude).
    landuse, soils
        Land use and soil maps (integer codes), resampled onto the DEM's cells
        (nearest neighbour) when their grids differ.
    landuse_lookup
        Land use code -> SWAT+ plant or urban name (``plants.plt`` / ``urban.urb``,
        e.g. ``{1: "FRSD", 2: "PAST"}``), as a dict or a two-column table. Codes with
        the same name make one land use; missing cells take QSWAT+'s default land use
        (code 0 if listed, else the first listed).
    soil_lookup
        Soil code -> soil name (``SNAM`` of the soil table), likewise.
    soil_table
        The soils: a ``usersoil`` table of ArcSWAT / QSWAT+ or one row per layer in
        SWAT+'s ``soils.sol`` columns (see the module notes); only the soils used are
        kept.
    stream_threshold
        The drainage area of a stream, in km2 (default: 1 % of the DEM's area, as
        QSWAT+ proposes).
    outlets
        Watershed outlets (points; :func:`ls.hydro.snap_outlets` moves them onto the
        streams within ``snap`` map units). Without them every stream that leaves the
        map (or ends in a pit) is an outlet.
    slope_bands
        Slope limits in percent between the bands, e.g. ``(5, 15)`` (default: one band,
        ``0-9999``).
    hru_method : {"dominant_hru", "dominant", "filter", "area"}
        QSWAT+'s choices: one HRU per landscape unit with the largest combination
        (``"dominant_hru"``, QSWAT+'s default) or with the dominant land use, soil and
        slope (``"dominant"``); all combinations above ``thresholds`` (land use, soil,
        slope, in percent; ``"filter"``); or above ``min_area`` (ha, or percent with
        ``area_in_ha=False``; ``"area"``). Removed HRUs give their area to the others.
    exempt
        Land use names never removed by the filters.

    Returns
    -------
    SwatWatershed
    """
    if hru_method not in _METHODS:
        raise ValueError(f"hru_method must be one of {_METHODS}")
    s = _read(dem, None, None, None, None)
    if s.crs is None:
        raise ValueError("the DEM needs a CRS (the model needs latitudes)")
    ny, nx = s.shape
    dx, dy = _cells(s)
    cell_area = dx * dy
    z = s.values
    filled = read_surface(fill(s.raster(z, "dem"))).values
    fs = type(s)(np.ascontiguousarray(filled), s.extent, s.crs, s.template)
    d = _d8(fs, True)
    acc = _core.hydro_d8_accumulation(d, None)
    valid = d != -2
    if stream_threshold is None:
        thr = 0.01 * valid.sum()
    else:
        thr = float(stream_threshold) * 1e6 / cell_area
    stream = ((np.nan_to_num(acc) >= thr) & valid).astype(np.uint8)
    if not stream.any():
        raise ValueError("no streams: lower stream_threshold")
    breaks = None
    if outlets is not None:
        rc = snap_outlets(outlets, fs.raster(stream.astype(np.float64), "streams"), distance=snap,
                          accumulation=fs.raster(acc, "acc"))
        breaks = np.zeros(s.shape, dtype=np.uint8)
        breaks[rc[:, 0], rc[:, 1]] = 1
    link, start, end, down = _core.hydro_stream_links(d, stream, breaks)
    links = _link_table(fs, d, link, start, end, down, breaks)
    n_links = len(links)
    # the links kept (upstream of an outlet, the outlet links included), numbered 1..n
    keep = np.ones(n_links + 1, bool)
    if outlets is not None:
        reach = np.zeros(n_links + 1, np.int8)  # 0 unknown, 1 reaches an outlet, -1 does not
        for k0 in range(1, n_links + 1):
            path, j = [], k0
            while j > 0 and reach[j] == 0 and not links.at[j, "outlet"]:
                path.append(j)
                j = int(links.at[j, "downstream"])
            v = 1 if j > 0 and (reach[j] == 1 or links.at[j, "outlet"]) else -1
            if j > 0 and links.at[j, "outlet"]:
                reach[j] = 1
            for p_ in path:
                reach[p_] = v
        keep = reach == 1
    keep = keep[1:]
    new_id = np.zeros(n_links + 1, dtype=np.int64)
    new_id[1:][keep] = np.arange(1, keep.sum() + 1)
    links = links[keep].copy()
    links.index = pd.RangeIndex(1, len(links) + 1, name="channel")
    links["downstream"] = new_id[links["downstream"].to_numpy()]
    seeds = new_id[link].astype(np.int32)
    basin = _core.hydro_watersheds(d, seeds)
    n = len(links)
    if n == 0:
        raise ValueError("no stream drains to the outlets")

    # rasters on the DEM grid
    lu = _on_grid(landuse, s)
    so = _on_grid(soils, s)
    lu_names = _lookup(landuse_lookup, "land use")
    soil_names = _lookup(soil_lookup, "soil")
    crops = list(dict.fromkeys(lu_names.values()))
    crop_of = {k: crops.index(v) for k, v in lu_names.items()}
    default_crop = crop_of[0] if 0 in crop_of else crop_of[next(iter(lu_names))]
    soil_list = list(dict.fromkeys(soil_names.values()))
    soil_of = {k: soil_list.index(v) for k, v in soil_names.items()}
    default_soil = soil_of[0] if 0 in soil_of else soil_of[next(iter(soil_names))]

    inside = basin > 0
    flat = np.flatnonzero(inside)
    lsu = basin.ravel()[flat].astype(np.int64)
    crop = _codes(lu.ravel()[flat], crop_of, default_crop, "land use")
    soil = _codes(so.ravel()[flat], soil_of, default_soil, "soil")
    nxt = _down(d)
    zf = filled.ravel()
    step = _steps(dx, dy)[np.clip(d.ravel()[flat], 0, 7)]
    has = nxt[flat] >= 0
    slope = np.where(has, (zf[flat] - zf[np.maximum(nxt[flat], 0)]) / step, 0.0)
    slope = np.maximum(slope, 0.0)
    limits = [float(v) for v in slope_bands]
    band = _slope_band(slope * 100, limits)
    elev = z.ravel()[flat]
    xmin, xmax, ymin, ymax = s.extent
    rx, ry = (xmax - xmin) / nx, (ymax - ymin) / ny
    rows, cols = flat // nx, flat % nx
    xs = xmin + (cols + 0.5) * rx
    ys = ymax - (rows + 0.5) * ry
    # flow distances to the channels and to the subbasin outlets (TauDEM's distances)
    on_channel = (seeds > 0).astype(np.uint8)
    dist_ch = _core.hydro_flow_distance(d, on_channel, dx, dy).ravel()[flat]
    outlet_cells = np.zeros(ny * nx, dtype=np.uint8)
    end_cells = links["end_row"].to_numpy() * nx + links["end_col"].to_numpy()
    outlet_cells[end_cells] = 1
    dist_st = _core.hydro_flow_distance(d, outlet_cells.reshape(ny, nx), dx, dy).ravel()[flat]

    # LSU and HRU data (BasinData.addCell over the cells in raster order)
    lsus = {c: _Lsu() for c in range(1, n + 1)}
    m = n + 1
    sums = lambda w: np.bincount(lsu, weights=w, minlength=m)  # noqa: E731
    l_count = np.bincount(lsu, minlength=m)
    l_x, l_y, l_slope, l_elev = sums(xs), sums(ys), sums(slope), sums(elev)
    sub_far = np.zeros(m)
    np.maximum.at(sub_far, lsu, dist_st)
    sub_min = np.full(m, np.inf)
    np.minimum.at(sub_min, lsu, elev)
    sub_max = np.full(m, -np.inf)
    np.maximum.at(sub_max, lsu, elev)
    # the farthest cell from the channel (the first in raster order of the largest distances)
    far_order = np.lexsort((np.arange(lsu.size), -dist_ch, lsu))
    first = far_order[np.r_[True, lsu[far_order][1:] != lsu[far_order][:-1]]]
    for c in range(1, m):
        L = lsus[c]
        L.count = int(l_count[c])
        L.area = L.count * cell_area
        L.css_area = L.area
        L.x, L.y, L.slope, L.elev = l_x[c], l_y[c], l_slope[c], l_elev[c]
    for i in first:
        lsus[int(lsu[i])].far_distance = float(dist_ch[i])
        lsus[int(lsu[i])].far_elevation = float(elev[i])
    nc, ns, nb = len(crops), len(soil_list), len(limits) + 1
    for what, key in (("orig_crop", lsu * nc + crop), ("orig_soil", lsu * ns + soil), ("orig_slope", lsu * nb + band)):
        u, idx, cnt = np.unique(key, return_index=True, return_counts=True)
        for kk in np.argsort(idx, kind="stable"):
            c, v = divmod(int(u[kk]), {"orig_crop": nc, "orig_soil": ns, "orig_slope": nb}[what])
            getattr(lsus[c], what)[v] = cnt[kk] * cell_area
    full = ((lsu * nc + crop) * ns + soil) * nb + band
    u, idx, inv = np.unique(full, return_index=True, return_inverse=True)
    g = lambda w: np.bincount(inv, weights=w, minlength=u.size)  # noqa: E731
    g_count = np.bincount(inv, minlength=u.size)
    g_x, g_y, g_slope, g_elev = g(xs), g(ys), g(slope), g(elev)
    for kk in np.argsort(idx, kind="stable"):
        rest, bd = divmod(int(u[kk]), nb)
        rest, sl = divmod(rest, ns)
        c, cr = divmod(rest, nc)
        L = lsus[c]
        h = len(L.hru) + 1
        L.numbers.setdefault(cr, {}).setdefault(sl, {})[bd] = h
        L.hru[h] = _Cell(int(g_count[kk]), g_count[kk] * cell_area, g_elev[kk], g_slope[kk], g_x[kk], g_y[kk])
    exempt_ids = {crops.index(e) for e in exempt if e in crops}
    for L in lsus.values():
        _select(L, hru_method, thresholds, min_area, area_in_ha, exempt_ids)

    to_ll = _transformer(s.crs)

    def lonlat(x, y):
        lon, lat = to_ll.transform(x, y)
        return float(lat), float(lon)

    # gis tables (DBUtils / hrus.py), subbasins in SWAT+ order
    subs, chans, lsu_rows, hru_rows, points, routing = [], [], [], [], [], []
    hru_raster = np.full(ny * nx, np.nan)
    hru_id = 0
    lsu_hru_map = {}
    for c in range(1, n + 1):
        L = lsus[c]
        area_ha = L.area / 1e4
        mean_slope = 0.0 if L.count == 0 else L.slope / L.count
        lat, lon = lonlat(L.x / L.count, L.y / L.count)
        mean_elev = L.elev / L.count
        subs.append({"id": c, "area": area_ha, "slo1": mean_slope * 100, "len1": float(sub_far[c]),
                     "sll": getSlsubbsn(mean_slope), "lat": lat, "lon": lon, "elev": mean_elev,
                     "elevmin": float(sub_min[c]), "elevmax": float(sub_max[c]), "waterid": 0})
        r = links.loc[c]
        drain_ha = float(acc.ravel()[int(r.end_row) * nx + int(r.end_col)]) * cell_area / 1e4
        drain_km = drain_ha / 100
        length = float(r.length)
        drop = float(r.z_start) - float(r.z_end)
        slope_pct = 0.0 if length <= 0 else drop / length * 100
        x0 = xmin + (r.start_col + 0.5) * rx
        y0 = ymax - (r.start_row + 0.5) * ry
        er, ec = int(r.end_row), int(r.end_col)
        j = nxt[er * nx + ec]
        if j >= 0:
            er, ec = j // nx, j % nx
        x1 = xmin + (ec + 0.5) * rx
        y1 = ymax - (er + 0.5) * ry
        mlat, mlon = lonlat((x0 + x1) / 2, (y0 + y1) / 2)
        chans.append({"id": c, "subbasin": c, "areac": drain_ha, "strahler": int(r.order), "len2": length,
                      "slo2": slope_pct, "wid2": WIDTH_MULTIPLIER * drain_km**WIDTH_EXPONENT,
                      "dep2": DEPTH_MULTIPLIER * drain_km**DEPTH_EXPONENT, "elevmin": float(r.z_end),
                      "elevmax": float(r.z_start), "midlat": mlat, "midlon": mlon})
        # the landscape unit
        lsu_id = c * 10
        source_elev = float(r.z_start)
        trib_distance = L.far_distance
        trib_drop = abs(L.far_elevation - source_elev)
        area_km = area_ha / 100
        lsu_rows.append({"id": lsu_id, "category": 0, "channel": c, "area": area_ha, "slope": mean_slope * 100,
                         "len1": trib_distance,
                         "csl": 0.0 if trib_distance < 1 else trib_drop / trib_distance * 100,
                         "wid1": WIDTH_MULTIPLIER * area_km**WIDTH_EXPONENT,
                         "dep1": DEPTH_MULTIPLIER * area_km**DEPTH_EXPONENT, "lat": lat, "lon": lon,
                         "elev": mean_elev})
        # HRUs (printLSUHRUs), in the order QSWAT+ keeps them
        for cr, ss in L.numbers.items():
            arland = L.crop_area(cr) / 1e4
            for sl, slopes in ss.items():
                arso = L.crop_soil_area(cr, sl) / 1e4
                for bd, h in slopes.items():
                    hru_id += 1
                    cd = L.hru[h]
                    mult = cell_area / cd.area
                    hlat, hlon = lonlat(cd.x * mult, cd.y * mult)
                    luse = crops[cr]
                    if luse.upper() == "WATR":
                        luse = "WETW"
                    hru_rows.append({"id": hru_id, "lsu": lsu_id, "arsub": area_ha, "arlsu": area_ha,
                                     "landuse": None if luse.upper() == "BARR" else luse, "arland": arland,
                                     "soil": soil_list[sl], "arso": arso, "slp": _slope_range(bd, limits),
                                     "arslp": cd.area / 1e4, "slope": cd.slope * mult * 100, "lat": hlat,
                                     "lon": hlon, "elev": cd.elev * mult})
                    lsu_hru_map[(c, cr, sl, bd)] = hru_id
                    routing.append((hru_id, "HRU", "tot", c, "CH", 100.0))
    # the HRU of each cell (removed combinations: NaN)
    by_group = np.array([lsu_hru_map.get(_split(int(v), nc, ns, nb), np.nan) for v in u])
    hru_raster[flat] = by_group[inv]

    # outlets of the subbasins, the watershed outlets and the routing
    downstream = links["downstream"].to_numpy()
    for c in range(1, n + 1):
        r = links.loc[c]
        er, ec = int(r.end_row), int(r.end_col)
        x = xmin + (ec + 0.5) * rx
        y = ymax - (er + 0.5) * ry
        plat, plon = lonlat(x, y)
        points.append({"id": c, "subbasin": c, "ptype": "O", "xpr": x, "ypr": y, "lat": plat, "lon": plon,
                       "elev": float(z[er, ec])})
    for c in range(1, n + 1):
        ds = int(downstream[c - 1])
        routing.append((c, "CH", "tot", c, "PT", 100.0))
        routing.append((c, "PT", "tot", ds, "CH", 100.0) if ds > 0 else (c, "PT", "tot", 0, "X", 100.0))
    for c in range(1, n + 1):
        routing.append((c, "SUB", "tot", c, "PT", 100.0))
    for c in range(1, n + 1):
        routing.append((c * 10, "LSU", "tot", c, "CH", 100.0))
        routing.append((c * 10, "LSU", "rhg", c * 10, "AQU", 100.0))
    # aquifers: one shallow per subbasin, one deep per watershed outlet
    def final(c):
        while downstream[c - 1] > 0:
            c = int(downstream[c - 1])
        return c

    aquifers, deep = [], {}
    for c in range(1, n + 1):
        L = lsus[c]
        out = final(c)
        cx, cy = L.x / L.count, L.y / L.count
        row = min(max(int((ymax - cy) / ry), 0), ny - 1)
        col = min(max(int((cx - xmin) / rx), 0), nx - 1)
        e = z[row, col]
        e = 0.0 if math.isnan(e) else float(e)
        lat, lon = lonlat(cx, cy)
        aquifers.append({"id": c * 10, "category": 0, "subbasin": c, "deep_aquifer": out, "area": L.area / 1e4,
                         "lat": lat, "lon": lon, "elev": e})
        a, em, xm, ym = deep.get(out, (0.0, 0.0, 0.0, 0.0))
        deep[out] = (a + L.area, em + e * L.area, xm + cx * L.area, ym + cy * L.area)
        routing.append((c * 10, "AQU", "tot", c, "PT", 100.0))
        routing.append((c * 10, "AQU", "rhg", out, "DAQ", 100.0))
    deep_rows = []
    for out, (a, em, xm, ym) in deep.items():
        lat, lon = lonlat(xm / a, ym / a)
        deep_rows.append({"id": out, "subbasin": out, "area": a / 1e4, "lat": lat, "lon": lon, "elev": em / a})
        routing.append((out, "DAQ", "tot", out, "PT", 100.0))

    soils_sol, soils_layer = _soil_tables(soil_table, sorted({soil_list[s_] for s_ in set(soil.tolist())},
                                                              key=soil_list.index))
    tables = {
        "gis_subbasins": pd.DataFrame(subs),
        "gis_channels": pd.DataFrame(chans),
        "gis_lsus": pd.DataFrame(lsu_rows),
        "gis_hrus": pd.DataFrame(hru_rows),
        "gis_aquifers": pd.DataFrame(aquifers),
        "gis_deep_aquifers": pd.DataFrame(deep_rows),
        "gis_points": pd.DataFrame(points),
        "gis_routing": pd.DataFrame(routing, columns=["sourceid", "sourcecat", "hyd_typ", "sinkid", "sinkcat",
                                                      "percent"]),
        "soils_sol": soils_sol,
        "soils_sol_layer": soils_layer,
    }
    sub_r = basin.astype(np.float64)
    sub_r[~valid] = np.nan
    ch_r = np.where(valid, seeds, np.nan).astype(np.float64)
    return SwatWatershed(tables=tables, subbasins=s.raster(sub_r, "subbasins"),
                         hrus=s.raster(hru_raster.reshape(ny, nx), "hrus"), channels=s.raster(ch_r, "channels"))


def _split(key: int, nc: int, ns: int, nb: int) -> tuple[int, int, int, int]:
    rest, bd = divmod(key, nb)
    rest, sl = divmod(rest, ns)
    c, cr = divmod(rest, nc)
    return c, cr, sl, bd


def _on_grid(raster, s) -> np.ndarray:
    """A raster's values on the DEM's cells (nearest neighbour when the grids differ)."""
    if isinstance(raster, np.ndarray) and raster.shape == s.shape:
        return raster.astype(np.float64)
    r = read_surface(raster)
    if r.shape == s.shape and np.allclose(r.extent, s.extent):
        return r.values
    from ... import io

    da = raster if isinstance(raster, xr.DataArray) else io.load_raster(raster)
    like = s.template if s.template is not None else s.raster(s.values, "dem")
    return np.asarray(io.match_grid(da, like, resampling="nearest").values, dtype=np.float64).reshape(s.shape)


def _codes(values: np.ndarray, index: dict, default: int, what: str) -> np.ndarray:
    out = np.full(values.shape, default, dtype=np.int64)
    ok = ~np.isnan(values)
    v = values[ok].astype(np.int64)
    unknown = sorted(set(np.unique(v).tolist()) - set(index))
    if unknown:
        raise ValueError(f"{what} codes {unknown[:10]} are not in the {what} lookup")
    lut_keys = np.array(sorted(index), dtype=np.int64)
    lut_vals = np.array([index[k] for k in lut_keys], dtype=np.int64)
    out[ok] = lut_vals[np.searchsorted(lut_keys, v)]
    return out
