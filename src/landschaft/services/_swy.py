# SPDX-License-Identifier: GPL-3.0-or-later
"""Seasonal water yield (InVEST's ``seasonal_water_yield`` model).

Each raster of InVEST's chain is computed in the type InVEST writes it in, with its
expression: the DEM filled, routed and traced into streams by pygeoprocessing's
algorithms (ported in ``ls.hydro``), the monthly quickflow and the curve numbers in numpy
as ``pygeoprocessing.raster_map`` applies them to the valid cells, and the local recharge
and the baseflow, which follow the flow, in C++ (``src/cpp/serv/swy.cpp``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ..surface._grid import read_surface
from ._common import LandUse, align_inputs, lookup_table
from ._sdr import BLOCK, _dtype_of, _routing

__all__ = ["SeasonalWaterYield", "seasonal_water_yield"]

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
SOIL_COLUMNS = ["cn_a", "cn_b", "cn_c", "cn_d"]


@dataclass
class SeasonalWaterYield(Saveable):
    """Quickflow, recharge and baseflow of a landscape (:func:`seasonal_water_yield`).

    Annual maps in millimetres per year (float32), NaN where a cell has no value.

    Attributes
    ----------
    quickflow : xarray.DataArray
        QF, the rain that runs off during and soon after the storms (the sum of
        the months).
    local_recharge : xarray.DataArray
        L, the rain that infiltrates minus what the cell evaporates
        (``P - QF - AET``); negative where the cell uses water from upslope. The
        recharge given, with ``recharge``.
    l_avail : xarray.DataArray
        The part of L available downslope: ``min(gamma L, L)``.
    l_sum_avail : xarray.DataArray
        The recharge available from the cells upslope (None with ``recharge``).
    l_sum : xarray.DataArray
        The recharge of the cell and of all the cells upslope.
    baseflow : xarray.DataArray
        B, the cell's contribution to the slow flow that reaches the streams.
    b_sum : xarray.DataArray
        The baseflow through the cell from the cells upslope.
    vri : xarray.DataArray
        The cell's share of the recharge of the whole area (L over its sum).
    aet : xarray.DataArray
        Actual evapotranspiration (None with ``recharge``).
    precipitation : xarray.DataArray
        Annual precipitation (None with ``recharge``).
    stream : xarray.DataArray
        The streams (1), 0 elsewhere (uint8, 255 outside the DEM).
    watersheds : geopandas.GeoDataFrame
        The polygons with ``qb`` (the mean recharge, mm), ``vri_sum`` and
        ``geom_id`` (the row).
    monthly : pandas.DataFrame or None
        For each polygon (``geom_id``) and month, the quickflow, baseflow (the
        annual over 12) and precipitation in cubic metres.
    intermediate : dict
        ``quickflow_monthly`` and ``kc`` and ``n_events`` (twelve months), ``cn``,
        ``si``, ``flow_dir``, ``flow_accumulation`` and ``pit_filled_dem``, in
        InVEST's types.
    """

    quickflow: xr.DataArray | None
    local_recharge: xr.DataArray
    l_avail: xr.DataArray
    l_sum_avail: xr.DataArray | None
    l_sum: xr.DataArray
    baseflow: xr.DataArray
    b_sum: xr.DataArray
    vri: xr.DataArray
    aet: xr.DataArray | None
    precipitation: xr.DataArray | None
    stream: xr.DataArray
    watersheds: object
    monthly: pd.DataFrame | None = None
    intermediate: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        b = float(np.nanmean(self.baseflow.values))
        lr = float(np.nanmean(self.local_recharge.values))
        if self.quickflow is None:
            return f"SeasonalWaterYield(mean recharge {lr:.1f} mm, baseflow {b:.1f} mm)"
        q = float(np.nanmean(self.quickflow.values))
        return f"SeasonalWaterYield(mean quickflow {q:.1f} mm, recharge {lr:.1f} mm, baseflow {b:.1f} mm)"


def _number(v, what) -> float:
    """A parameter as InVEST reads it: ``float(Fraction(v))`` (``"1/12"`` too)."""
    try:
        return float(Fraction(str(v).strip()))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"{what} must be a number, not {v!r}") from None


def _monthly(x, what) -> list:
    """Twelve rasters: a sequence of them, or a raster of twelve layers."""
    if isinstance(x, xr.DataArray) and x.ndim == 3:
        if x.shape[0] != 12:
            raise ValueError(f"{what} must have 12 layers (months), not {x.shape[0]}")
        return [x.isel({x.dims[0]: m}) for m in range(12)]
    if isinstance(x, np.ndarray) and x.ndim == 3:
        if x.shape[0] != 12:
            raise ValueError(f"{what} must have 12 layers (months), not {x.shape[0]}")
        return list(x)
    if isinstance(x, dict):
        if sorted(x) != list(range(1, 13)):
            raise ValueError(f"{what} by month must have the months 1 to 12")
        return [x[m] for m in range(1, 13)]
    xs = list(x)
    if len(xs) != 12:
        raise ValueError(f"{what} must be 12 rasters (months), not {len(xs)}")
    return xs


def _by_month(x, column, what) -> np.ndarray:
    """Twelve numbers: a sequence, or a table (``month`` and ``column``) as InVEST's."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        df = x.to_frame() if isinstance(x, pd.Series) else x.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "month" in df.columns:
            df = df.set_index("month")
        if column not in df.columns:
            raise ValueError(f"{what} lacks the column {column!r}")
        s = df[column]
        s.index = [int(i) for i in s.index]
        if sorted(s.index) != list(range(1, 13)):
            raise ValueError(f"{what} must have the months 1 to 12")
        return s.sort_index().to_numpy(np.float64)
    v = np.asarray(x, np.float64).ravel()
    if v.size != 12:
        raise ValueError(f"{what} must hold 12 months, not {v.size}")
    return v


def _grid(x, shape, what):
    """A raster's values (NaN missing) and its type."""
    s = read_surface(x)
    if s.shape != shape:
        raise ValueError(f"{what} has {s.shape} cells, the land use map {shape}")
    dt = _dtype_of(x)
    if not np.issubdtype(dt, np.number):
        dt = np.dtype(np.float64)
    return s.values, dt


def _as(values, dtype):
    """The valid cells of a raster in its type (0 where missing: InVEST masks them out)."""
    ok = ~np.isnan(values)
    return np.where(ok, values, 0).astype(dtype), ok


def _curve_numbers(lu_codes, lu_ok, lu_dtype, soil, soil_ok, soil_dtype, codes, cn_table):
    """InVEST's _calculate_curve_number_raster: the table's CN (float32) of each class and soil group.

    The raster is in the type of the land use and soil maps together, as InVEST writes it,
    except float64 where both are integers (InVEST then truncates fractional CNs; float64
    gives its S for integer CNs, which it divides in float64).
    """
    dt = np.result_type(lu_dtype, soil_dtype)
    if np.issubdtype(dt, np.integer):
        dt = np.dtype(np.float64)
    ok = lu_ok & soil_ok
    groups = soil[ok]
    bad = sorted(set(np.unique(groups).tolist()) - {1, 2, 3, 4})
    if bad:
        raise ValueError(f"the soil group map must hold only the groups 1, 2, 3 and 4 (found {bad})")
    idx = np.searchsorted(codes, lu_codes[ok])
    missing = (idx >= codes.size) | (codes[np.minimum(idx, codes.size - 1)] != lu_codes[ok])
    if np.any(missing):
        raise ValueError(f"the land use map has class(es) {sorted(set(lu_codes[ok][missing].tolist()))} "
                         "that the biophysical table lacks")
    cn32 = cn_table.astype(np.float32)  # (classes, 4)
    cn = np.full(lu_codes.shape, np.nan, dt)
    cn[ok] = cn32[idx, groups.astype(np.int64) - 1]
    return cn


def _quickflow(p, p_ok, si, n_m, stream, threads=1):
    """InVEST's qf_op (eq. 1): the quickflow of a month (float32), NaN as InVEST's nodata."""
    import scipy.special

    valid_p, valid_n = p_ok, ~np.isnan(n_m)
    nm = np.where(valid_n, n_m, 0).astype(np.float32)
    precip_ok = valid_p & valid_n & (p > 0) & (nm > 0)
    on_stream = stream == 1
    valid = valid_p & valid_n & (stream != 255) & ~np.isnan(si)
    s = np.where(np.isnan(si), -1, si).astype(np.float32)
    qf = np.full(p.shape, np.nan, np.float32)
    qf[~precip_ok & valid_p & valid_n] = 0
    qf[precip_ok & on_stream] = p[precip_ok & on_stream]
    c3 = valid & precip_ok & ~on_stream
    a = np.full(p.shape, np.nan, np.float32)
    a[c3] = p[c3] / (nm[c3] * 25.4)
    c3a = c3 & (s == 0)
    qf[c3a] = p[c3a]
    with np.errstate(divide="ignore", invalid="ignore"):
        c3b = c3 & (s / a > 100)
    qf[c3b] = 0
    c = c3 & ~(c3a | c3b)
    s_c, a_c, n_c = s[c], a[c], nm[c]
    out = np.empty(s_c.shape, np.float32)

    def part(lo, hi):  # cell by cell, so the chunks (numpy releases the GIL) give the same bits
        s_, a_ = s_c[lo:hi], a_c[lo:hi]
        out[lo:hi] = (25.4 * n_c[lo:hi] * (((a_ - s_) * np.exp(-0.2 * s_ / a_))
                                           + (s_ ** 2 / a_ * np.exp(0.8 * s_ / a_) * scipy.special.exp1(s_ / a_))))

    _chunks(part, s_c.size, threads)
    qf[c] = out
    qf[valid & (qf < 0)] = 0
    return qf


def _chunks(fn, n, threads, size=1 << 16):
    """``fn(lo, hi)`` over [0, n) in chunks, on ``threads`` threads."""
    bounds = [(lo, min(n, lo + size)) for lo in range(0, n, size)]
    if threads <= 1 or len(bounds) <= 1:
        for lo, hi in bounds:
            fn(lo, hi)
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(lambda b: fn(*b), bounds))


def _vri(l_values, l_dtype):
    """InVEST's _calculate_vri: L over its sum, the sum taken by pygeoprocessing's blocks (float32 out)."""
    a = np.where(np.isnan(l_values), 0, l_values).astype(l_dtype)
    ok = ~np.isnan(l_values)
    qb_sum = 0
    ny, nx = a.shape
    # iterblocks over 256 x 256 tiles: widened, then heightened, up to 2**16 cells
    cols, rows, largest = BLOCK, BLOCK, 2 ** 16
    if largest // (cols * rows) > 0:
        cols = min(cols * (largest // (cols * rows)), nx)
    if largest // (cols * rows) > 0:
        rows = min(rows * (largest // (cols * rows)), ny)
    for y in range(0, ny, rows):
        for x in range(0, nx, cols):
            blk, v = a[y:y + rows, x:x + cols], ok[y:y + rows, x:x + cols]
            blk = blk[v]
            qb_sum += np.sum(blk[~np.isinf(blk)])
    out = np.full(a.shape, np.nan, np.float32)
    if qb_sum > 0:
        out[ok] = a[ok] / qb_sum
    return out, qb_sum


def _f32(a):
    return np.asarray(a, np.float64).astype(np.float32)


def seasonal_water_yield(dem, lulc, biophysical, watersheds, *, threshold: float, precipitation=None, et0=None,
                         soil_group=None, rain_events=None, climate_zones=None, climate_zone_table=None,
                         alpha=1 / 12, beta: float = 1.0, gamma: float = 1.0, recharge=None, method: str = "mfd",
                         nodata=None, n_threads: int | None = None) -> SeasonalWaterYield:
    """Quickflow, local recharge and baseflow by month and year (InVEST's seasonal water yield model).

    The quickflow of each month is the SCS curve number runoff of its storms
    (``QF = 25.4 n ((a - S) e^(-0.2 S/a) + S^2/a e^(0.8 S/a) E1(S/a))``, ``a`` the
    rain of a storm in inches, ``S = 1000 / CN - 10``, all the rain on the
    streams). The local recharge is the rain that is not quickflow nor evaporated,
    ``L = P - QF - AET``, with the monthly ``AET = min(Kc ET0, P - QF + alpha_m
    beta L_sum_avail)``: the cell can use water from upslope, the available
    recharge ``L_avail = min(gamma L, L)`` of the cells above, summed by their
    shares of the flow. The baseflow is the part of the recharge that reaches a
    stream: ``B_sum`` along the flow from the outlets up and ``B = max(B_sum L /
    L_sum, 0)``.

    The rasters are InVEST's (natcap.invest 3.20) bit for bit, in its float32 and
    float64 types: the DEM filled, routed (pygeoprocessing's multiple flow
    directions or D8) and traced into streams as pygeoprocessing does, the
    quickflow, curve numbers, recharge and baseflow; the polygon sums agree to
    float32 precision (InVEST adds the cells in float32). Five bugs of InVEST
    are corrected (and the tests compare against InVEST with them corrected):
    cells without recharge add none to the baseflow (InVEST reads their nodata
    as numbers, which makes the baseflow infinite there and upslope),
    ``L_sum_avail`` sums the upslope terms by their shares of the flow (InVEST
    takes their mean weighted by the raw flow weights, so a cell fed by three
    cells gets the mean of their recharge instead of the sum), the flow off the
    raster ends the baseflow path as the flow into a hole of the DEM does
    (InVEST gives such an outlet ``B_sum = 0``), a cell without elevation that a
    D8 direction points into gets no recharge, and fractional curve numbers are
    kept with integer land use and soil maps (InVEST truncates them).

    Parameters
    ----------
    dem
        Elevations (projected, in metres).
    lulc
        Land use map (integer codes).
    biophysical
        A DataFrame by class (``lucode``): ``cn_a`` to ``cn_d`` (the curve
        number of each hydrologic soil group, 0 to 100) and ``kc_1`` to ``kc_12``
        (the monthly crop coefficients).
    watersheds
        Polygons (GeoDataFrame) over which the recharge is summarised.
    threshold
        Flow accumulation (cells) from which a cell is a stream.
    precipitation, et0
        Monthly precipitation and reference evapotranspiration (mm): twelve
        rasters (a list, a dict by month 1 to 12, or a raster of 12 layers).
    soil_group
        Hydrologic soil groups (1 to 4 for A to D).
    rain_events
        The number of rain events of each month (12 numbers, or a table with
        ``month`` and ``events``); or ``climate_zones`` (a raster of zone ids)
        and ``climate_zone_table`` (by ``cz_id``, the events in the columns
        ``jan`` to ``dec``).
    alpha
        The share of the upslope available recharge usable in each month: one
        number (1/12) or twelve (a table with ``month`` and ``alpha``).
    beta
        The share of the upslope subsidy available for evapotranspiration.
    gamma
        The share of the local recharge available to the cells downslope.
    recharge
        A local recharge map (mm) to route instead of computing it: the climate,
        soil and quickflow inputs are not needed then.
    method
        ``"mfd"`` (multiple flow directions, the default) or ``"d8"``.
    nodata
        Value marking missing cells in the land use map.

    Returns
    -------
    SeasonalWaterYield

    Notes
    -----
    The rasters may be on different grids (cells, extent, CRS): as InVEST does,
    they are put on the DEM's cells first, by the nearest cell, over the
    intersection of their bounds and the watersheds' (its corner moved onto the
    DEM's cells), so the maps cover the watersheds' box only.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> from affine import Affine
    >>> from shapely.geometry import box
    >>> import landschaft as ls
    >>> y, x = np.mgrid[:40, :50]
    >>> z = (100 + 0.6 * x + 6 * np.sin(y / 6.0)).astype("float32")
    >>> t = Affine(30, 0, 500000, 0, -30, 8000000)
    >>> dem = ls.load_raster(ls.write_raster(z, "swy_dem.tif", crs="EPSG:31983", transform=t))
    >>> lulc = dem.copy(data=np.where(x < 25, 1, 2).astype("int16"))       # forest, pasture
    >>> soil = dem.copy(data=np.full(z.shape, 2, "uint8"))                  # group B
    >>> bio = pd.DataFrame({"lucode": [1, 2], "cn_a": [30, 49], "cn_b": [55, 69], "cn_c": [70, 79],
    ...                     "cn_d": [77, 84], **{f"kc_{m}": [1.0, 0.8] for m in range(1, 13)}})
    >>> rain = [dem.copy(data=np.full(z.shape, v, "float32")) for v in [250, 220, 180, 90, 40, 20,
    ...                                                                   10, 15, 50, 120, 180, 230]]
    >>> et0 = [dem.copy(data=np.full(z.shape, 110, "float32"))] * 12
    >>> ws = gpd.GeoDataFrame({"ws_id": [1]}, geometry=[box(*dem.rio.bounds())], crs=dem.rio.crs)
    >>> s = ls.services.seasonal_water_yield(dem, lulc, bio, ws, threshold=60, precipitation=rain, et0=et0,
    ...                                      soil_group=soil, rain_events=[12, 11, 9, 6, 3, 2, 1, 1, 3, 7, 10, 12])
    >>> land, qf = s.stream.values == 0, s.quickflow.values               # mm / yr (all the rain on streams)
    >>> bool(qf[land & (x > 30)].mean() > 3 * qf[land & (x < 20)].mean())  # pasture runs off more than forest
    True
    >>> list(s.watersheds.columns)
    ['ws_id', 'geometry', 'qb', 'vri_sum', 'geom_id']
    """
    threads = resolve(n_threads)
    method = str(method).lower()
    if method not in ("mfd", "d8"):
        raise ValueError("method is 'mfd' or 'd8'")
    if not threshold > 0:
        raise ValueError("threshold must be positive")
    import geopandas as gpd

    if not isinstance(watersheds, gpd.GeoDataFrame):
        raise TypeError("the watersheds are a GeoDataFrame")
    beta_f, gamma_f = _number(beta, "beta"), _number(gamma, "gamma")
    # InVEST's alignment: the DEM's cells over the rasters and the area of interest, the nearest cell
    rasters = {"lulc": (lulc, "near"), "dem": (dem, "near")}
    if recharge is None:
        if precipitation is not None:
            precipitation = _monthly(precipitation, "the precipitation")
            rasters.update({f"p{m}": (x, "near") for m, x in enumerate(precipitation)})
        if et0 is not None:
            et0 = _monthly(et0, "the ET0")
            rasters.update({f"e{m}": (x, "near") for m, x in enumerate(et0)})
        rasters["soil_group"] = (soil_group, "near")
        if rain_events is None:
            rasters["climate_zones"] = (climate_zones, "near")
    else:
        rasters["recharge"] = (recharge, "near")
    g = align_inputs(rasters, reference="dem", vectors=[watersheds], n_threads=threads)
    lulc, dem = g["lulc"], g["dem"]
    if recharge is None:
        if precipitation is not None:
            precipitation = [g[f"p{m}"] for m in range(12)]
        if et0 is not None:
            et0 = [g[f"e{m}"] for m in range(12)]
        soil_group, climate_zones = g.get("soil_group"), g.get("climate_zones", climate_zones)
    else:
        recharge = g["recharge"]
    lu = LandUse(lulc, nodata, "the land use map")
    shape = lu.shape
    dem_v, dem_dt = _grid(dem, shape, "the DEM")
    try:
        tr = lu.surface.template.rio.transform()
        pixel_area = abs(float(tr.a) * float(tr.e))
    except Exception:  # noqa: BLE001
        dx, dy = lu.pixel()
        pixel_area = dx * dy
    lu.pixel()  # projected maps only, as InVEST

    filled = _core.hydro_fill(dem_v.astype(np.float64), False).astype(dem_dt).astype(np.float64)
    d, acc, accumulate, _distance, _missing = _routing(filled, method)
    if method == "mfd":
        stream = _core.hydro_mfd_streams(acc, d, float(threshold), 1.0, BLOCK)
    else:
        stream = np.full(shape, 255, np.uint8)
        ok = ~np.isnan(acc)
        stream[ok] = (acc[ok] > threshold).astype(np.uint8)
    ras = lu.raster
    months = xr.DataArray(np.arange(1, 13), dims="month", name="month")

    def stack(arrays, name):
        return xr.concat([ras(a, name) for a in arrays], dim=months)

    inter = {"pit_filled_dem": filled.astype(dem_dt), "flow_dir": d, "flow_accumulation": acc}
    if recharge is None:
        if precipitation is None or et0 is None or soil_group is None:
            raise ValueError("give the precipitation, ET0 and soil groups (or the recharge map)")
        if (rain_events is None) == (climate_zones is None):
            raise ValueError("give the rain events or the climate zones (and their table)")
        if np.ndim(alpha) == 0 and not isinstance(alpha, (pd.DataFrame, pd.Series)):
            alpha_m = np.full(12, _number(alpha, "alpha"))
        else:
            alpha_m = _by_month(alpha, "alpha", "the monthly alpha")
        kc_cols = [f"kc_{m}" for m in range(1, 13)]
        codes, bio = lookup_table(biophysical, SOIL_COLUMNS + kc_cols, "the biophysical table")
        p_list = [_grid(x, shape, f"the precipitation of month {m + 1}")
                  for m, x in enumerate(_monthly(precipitation, "the precipitation"))]
        e_list = [_grid(x, shape, f"the ET0 of month {m + 1}") for m, x in enumerate(_monthly(et0, "the ET0"))]
        soil_v, soil_dt = _grid(soil_group, shape, "the soil group map")
        soil_ok = ~np.isnan(soil_v)
        soil_i = np.where(soil_ok, soil_v, 0).astype(np.int64)
        # rain events per month: a constant (also off the DEM, as InVEST's new raster) or by zone
        if rain_events is not None:
            ev = _by_month(rain_events, "events", "the rain events")
            n_events = [np.full(shape, np.float32(e), np.float32) for e in ev]
        else:
            if climate_zone_table is None:
                raise ValueError("give the climate zone table with the climate zones")
            cz_v, _ = _grid(climate_zones, shape, "the climate zone map")
            tab = climate_zone_table.copy()
            tab.columns = [str(c).strip().lower() for c in tab.columns]
            if "cz_id" in tab.columns:
                tab = tab.set_index("cz_id")
            missing_cols = [m for m in MONTHS if m not in tab.columns]
            if missing_cols:
                raise ValueError(f"the climate zone table lacks the column(s) {missing_cols}")
            zid = np.array([int(i) for i in tab.index], np.int64)
            order = np.argsort(zid)
            zid = zid[order]
            cz_ok = ~np.isnan(cz_v)
            cz_i = np.where(cz_ok, cz_v, 0).astype(np.int64)
            idx = np.searchsorted(zid, cz_i[cz_ok])
            bad = (idx >= zid.size) | (zid[np.minimum(idx, zid.size - 1)] != cz_i[cz_ok])
            if np.any(bad):
                raise ValueError(f"the climate zone map has zone(s) {sorted(set(cz_i[cz_ok][bad].tolist()))} "
                                 "that the table lacks")
            n_events = []
            for m in MONTHS:
                vals = tab[m].to_numpy(np.float64)[order].astype(np.float32)
                a = np.full(shape, np.nan, np.float32)
                a[cz_ok] = vals[idx]
                n_events.append(a)
        cn = _curve_numbers(lu.codes, lu.valid, _dtype_of(lulc), soil_i, soil_ok, soil_dt, codes,
                            bio[SOIL_COLUMNS].to_numpy())
        cn_ok = ~np.isnan(cn)
        si64 = np.full(shape, np.nan)
        si_ok = cn_ok & (np.where(cn_ok, cn, 0) > 0)
        si64[si_ok] = (1000 / cn[si_ok] - 10) * (stream[si_ok] != 1)
        si = si64.astype(np.float32)
        qf_months = []
        p_typed = []
        for m in range(12):
            pv, pdt = p_list[m]
            p, p_ok = _as(pv, pdt)
            p_typed.append((p, p_ok))
            qf_months.append(_quickflow(p, p_ok, si, n_events[m], stream, threads))
        qf_all_ok = np.logical_and.reduce([~np.isnan(q) for q in qf_months])
        qf = np.full(shape, np.nan, np.float32)
        if qf_all_ok.any():
            total = qf_months[0][qf_all_ok].copy()
            for q in qf_months[1:]:
                total += q[qf_all_ok]
            qf[qf_all_ok] = total
        cls = lu.classes(codes, threads)
        kc_months = []
        for col in kc_cols:
            v = bio[col].to_numpy().astype(np.float32)
            kc_months.append(np.where(cls >= 0, v[np.maximum(cls, 0)], np.float32(np.nan)).astype(np.float32))

        def months_array(lst):
            wide = any(np.result_type(dt, np.float32) != np.float32 for _, dt in lst)
            t = np.float64 if wide else np.float32
            return np.ascontiguousarray(np.stack([v for v, _ in lst]).astype(t))

        l64, la64, lsa64, aet64, p64 = _core.serv_swy_local_recharge(
            d, months_array(p_list), months_array(e_list), np.ascontiguousarray(np.stack(qf_months)),
            np.ascontiguousarray(np.stack(kc_months)), alpha_m.astype(np.float32), beta_f, gamma_f, BLOCK)
        l32, la32 = _f32(l64), _f32(la64)
        l_dt = np.dtype(np.float32)
        inter.update({"quickflow_monthly": stack(qf_months, "quickflow_monthly"), "cn": cn, "si": si,
                      "kc": stack(kc_months, "kc"), "n_events": stack(n_events, "n_events")})
    else:
        rv, l_dt = _grid(recharge, shape, "the recharge map")
        if not np.issubdtype(l_dt, np.floating):
            l_dt = np.result_type(l_dt, np.float32)
        lt, l_ok = _as(rv, l_dt)
        la = np.full(shape, np.nan, l_dt)
        la[l_ok] = np.min(np.stack((gamma_f * lt[l_ok], lt[l_ok])), axis=0)
        l32 = np.where(l_ok, lt, np.nan).astype(l_dt)
        la32 = la
    vri, _qb = _vri(l32.astype(np.float64), l_dt)
    lsum = accumulate(np.asarray(l32, np.float64))
    # missing recharge (a map with holes) counts nothing, as in the accumulation
    l_route = np.where(np.isnan(l32), 0, l32).astype(np.float64)
    la_route = np.where(np.isnan(la32), 0, la32).astype(np.float64)
    b64, bsum64 = _core.serv_swy_baseflow(d, l_route, la_route, lsum, stream, BLOCK)
    b32, bsum32 = _f32(b64), _f32(bsum64)
    sheds = _aggregate(watersheds, np.asarray(l32, np.float64), vri, ras, threads)
    monthly = None
    if recharge is None:
        monthly = _monthly_table(watersheds, b32, qf_months, [np.where(ok, p, np.nan) for p, ok in p_typed],
                                 pixel_area, ras, threads)
    return SeasonalWaterYield(
        ras(qf, "quickflow") if recharge is None else None, ras(l32, "local_recharge"), ras(la32, "l_avail"),
        ras(_f32(lsa64), "l_sum_avail") if recharge is None else None, ras(lsum, "l_sum"), ras(b32, "baseflow"),
        ras(bsum32, "b_sum"), ras(vri, "vri"), ras(_f32(aet64), "aet") if recharge is None else None,
        ras(_f32(p64), "precipitation") if recharge is None else None, ras(stream, "stream"), sheds, monthly,
        {k: (v if isinstance(v, xr.DataArray) else ras(v, k)) for k, v in inter.items()})


def _zonal(watersheds, maps: dict, ras, stats, threads):
    from .._zonal import zonal_stats

    names = list(maps)
    st = xr.concat([ras(np.asarray(maps[k], np.float64), k) for k in names], dim="band").assign_coords(band=names)
    return zonal_stats(st, watersheds, stats, n_threads=threads)


def _aggregate(watersheds, l_values, vri, ras, threads):
    """InVEST's _aggregate_recharge: the mean recharge ``qb`` and ``vri_sum`` per polygon, ``geom_id`` its row."""
    stats = _zonal(watersheds, {"l": l_values, "vri": vri}, ras, ["sum", "count"], threads)
    out = watersheds.copy()
    s, n = stats["l_sum"].to_numpy(np.float64), stats["l_count"].to_numpy(np.float64)
    out["qb"] = np.where(n > 0, s / np.where(n > 0, n, 1), 0.0)
    out["vri_sum"] = np.nan_to_num(stats["vri_sum"].to_numpy(np.float64))
    out["geom_id"] = np.arange(len(out), dtype=np.int64)
    return out


def _monthly_table(watersheds, b, qf_months, p_months, pixel_area, ras, threads):
    """InVEST's _generate_monthly_qf_b_p_csv: cubic metres per polygon and month."""
    maps = {"b": b, **{f"qf{m}": q for m, q in enumerate(qf_months)}, **{f"p{m}": p for m, p in enumerate(p_months)}}
    stats = _zonal(watersheds, maps, ras, ["sum"], threads)
    k = 0.001 * pixel_area
    rows = []
    for fid in range(len(watersheds)):
        bflow = np.nan_to_num(stats["b_sum"].iloc[fid]) * k / 12
        for m in range(12):
            rows.append({"geom_id": fid, "month": m + 1, "quickflow": np.nan_to_num(stats[f"qf{m}_sum"].iloc[fid]) * k,
                         "baseflow": bflow, "precipitation": np.nan_to_num(stats[f"p{m}_sum"].iloc[fid]) * k})
    return pd.DataFrame(rows)
