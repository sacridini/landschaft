# SPDX-License-Identifier: GPL-3.0-or-later
"""Annual water yield and hydropower (InVEST's ``annual_water_yield`` model)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ..surface._grid import read_surface
from ._common import LandUse, lookup_table

__all__ = ["WaterYield", "water_yield"]

#: kWh produced by a cubic metre falling one metre (InVEST's constant: 9.81 kN x 1 m / 3.6e6 J per kWh).
KWH_PER_M3_M = 0.00272


@dataclass
class WaterYield(Saveable):
    """Annual water yield of a landscape (:func:`water_yield`).

    Maps are in millimetres per year, NaN where an input has no value (or
    there is no rain, as InVEST leaves those cells out of the balance);
    float32, or float64 where InVEST computes in float64 (float64 inputs).

    Attributes
    ----------
    wyield : xarray.DataArray
        Water yield: precipitation minus actual evapotranspiration.
    aet : xarray.DataArray
        Actual evapotranspiration.
    pet : xarray.DataArray
        Potential evapotranspiration: reference evapotranspiration times Kc.
    fractp : xarray.DataArray
        Actual evapotranspiration as a fraction of precipitation (float32).
    demand : xarray.DataArray or None
        Consumptive water use per cell (m3 per year), with a demand table.
    watersheds, subwatersheds : geopandas.GeoDataFrame or None
        The polygons with ``precip_mn``, ``PET_mn``, ``AET_mn``, ``wyield_mn``
        (means over the cells with values, mm), ``wyield_vol`` (the mean yield
        over the polygon's area, m3), with demand ``consum_vol`` (m3),
        ``consum_mn`` (m3 per cell), ``rsupply_vl`` (realized supply, yield minus
        consumption, m3) and ``rsupply_mn`` (m3 per cell), and with valuation
        (watersheds) ``hp_energy`` (kWh per year) and ``hp_val`` (net present
        value).
    """

    wyield: xr.DataArray
    aet: xr.DataArray
    pet: xr.DataArray
    fractp: xr.DataArray
    demand: xr.DataArray | None = None
    watersheds: object = None
    subwatersheds: object = None

    def __repr__(self) -> str:
        return f"WaterYield(mean yield {float(np.nanmean(self.wyield.values)):.1f} mm)"


def _wide(x) -> bool:
    """Whether numpy computes ``float32 op x`` in float64 (InVEST's arithmetic)."""
    dtype = getattr(x, "dtype", None)
    if dtype is None:
        dtype = np.asarray(x).dtype
    return np.result_type(np.float32, dtype) == np.float64


def _values(x, shape, what, dtype):
    s = read_surface(x)
    if s.shape != shape:
        raise ValueError(f"{what} has {s.shape} cells, the land use map {shape}")
    return np.ascontiguousarray(s.values.astype(dtype))


def _sheds(sheds, id_name, maps: dict, like, cell_area, valuation, n_threads):
    """InVEST's per-watershed table: means of the maps, volumes, realized supply, hydropower."""
    import geopandas as gpd

    from .._zonal import zonal_stats

    if not isinstance(sheds, gpd.GeoDataFrame):
        raise TypeError(f"the {'sub' if id_name == 'subws_id' else ''}watersheds are a GeoDataFrame")
    cols = {str(c).lower(): c for c in sheds.columns}
    if id_name not in cols:
        raise ValueError(f"the {'sub' if id_name == 'subws_id' else ''}watersheds need a {id_name!r} column")
    names = list(maps)
    stack = xr.concat([like.copy(data=np.asarray(maps[k], np.float64)) for k in names], dim="band")
    stack = stack.assign_coords(band=names)
    stats = zonal_stats(stack, sheds, ["sum", "count"], n_threads=n_threads)
    out = sheds.copy()
    for k in names:
        s, n = stats[f"{k}_sum"].to_numpy(np.float64), stats[f"{k}_count"].to_numpy(np.float64)
        mean = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
        if k == "demand":
            out["consum_vol"] = np.where(n > 0, s, np.nan)
            out["consum_mn"] = mean
        else:
            out[k] = mean
            if k == "wyield_mn":
                out["wyield_vol"] = mean * out.geometry.area.to_numpy() / 1000
    if "demand" in maps:
        out["rsupply_vl"] = out["wyield_vol"] - out["consum_vol"]
        # per cell in m3, as consum_mn: the yield's mm over a cell, minus the demand of a cell
        out["rsupply_mn"] = out["wyield_mn"] * cell_area / 1000 - out["consum_mn"]
    if valuation is not None:
        val = valuation.copy()
        val.columns = [str(c).strip().lower() for c in val.columns]
        if "ws_id" in val.columns:
            val = val.set_index("ws_id")
        need = ["efficiency", "fraction", "height", "kw_price", "cost", "time_span", "discount"]
        missing = [c for c in need if c not in val.columns]
        if missing:
            raise ValueError(f"the valuation table lacks the column(s) {missing}")
        ids = [int(v) for v in out[cols[id_name]]]
        absent = sorted(set(ids) - {int(v) for v in val.index})
        if absent:
            raise ValueError(f"the valuation table lacks the watershed(s) {absent}")
        val.index = [int(v) for v in val.index]
        energy, npv = [], []
        for ws, rs in zip(ids, out["rsupply_vl"]):
            r = val.loc[ws]
            if not np.isfinite(rs):
                energy.append(np.nan)
                npv.append(np.nan)
                continue
            e = float(r["efficiency"]) * float(r["fraction"]) * float(r["height"]) * float(rs) * KWH_PER_M3_M
            ratio = 1 / (1 + float(r["discount"]) / 100)
            years = float(r["time_span"])
            dsum = (1 - math.pow(ratio, years)) / (1 - ratio) if ratio != 1 else years
            energy.append(e)
            npv.append((float(r["kw_price"]) * e - float(r["cost"])) * dsum)
        out["hp_energy"] = energy
        out["hp_val"] = npv
    return out


def water_yield(lulc, biophysical, precipitation, eto, depth, pawc, *, seasonality: float, watersheds=None,
                subwatersheds=None, demand=None, valuation=None, nodata=None,
                n_threads: int | None = None) -> WaterYield:
    """Annual water yield, consumption and hydropower value (InVEST's annual water yield model).

    The water balance of each cell on the Budyko curve in Fu and Zhang's form:
    actual evapotranspiration over precipitation
    ``AET/P = 1 + PET/P - (1 + (PET/P)^w)^(1/w)``, at most PET/P, with
    ``PET = Kc ET0`` and ``w = Z AWC / P + 1.25`` (at most 5),
    ``AWC = min(root depth, soil depth) x PAWC``; classes that are not vegetation
    (water, wetlands, towns) evaporate ``min(Kc ET0, P)``. The yield is
    ``P - AET``. With watersheds, the means over each polygon and the yield
    volume; with a demand table, the consumption and the realized supply; with
    a valuation table, the hydropower energy and its net present value.

    The maps are InVEST's (natcap.invest 3.20) bit for bit, in the same float32
    (or float64) arithmetic; the watershed means agree to float32 precision
    (InVEST adds the cells in float32).

    Parameters
    ----------
    lulc
        Land use map (integer codes; projected, in metres).
    biophysical
        A DataFrame by class (``lucode``): ``lulc_veg`` (1 vegetation, 0 not),
        ``root_depth`` (mm) and ``kc`` (crop coefficient).
    precipitation, eto
        Annual precipitation and reference evapotranspiration (mm), on the grid of
        ``lulc``. Cells with no rain are left out, as InVEST does.
    depth
        Depth to the root restricting layer (mm).
    pawc
        Plant available water content (fraction, 0 to 1).
    seasonality
        The Z parameter (about 1 to 30): the seasonal distribution of rain.
    watersheds, subwatersheds
        Polygons (GeoDataFrame) with a ``ws_id`` (``subws_id``) column, in the
        maps' CRS.
    demand
        A DataFrame by class (``lucode``, ``demand``): consumptive use, m3 per
        year and cell.
    valuation
        A DataFrame by watershed (``ws_id``): ``efficiency``, ``fraction``,
        ``height`` (m), ``kw_price``, ``cost`` (per year), ``time_span`` (years)
        and ``discount`` (percent per year). Needs the demand table.
    nodata
        Value marking missing cells in the land use map.

    Returns
    -------
    WaterYield

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from affine import Affine
    >>> import landschaft as ls
    >>> codes = np.array([[1, 1, 2], [1, 2, 3]])                 # forest, pasture, water
    >>> lulc = ls.load_raster(ls.write_raster(codes, "wy_lulc.tif", crs="EPSG:31983",
    ...                                       transform=Affine(30, 0, 500000, 0, -30, 8000000)))
    >>> bio = pd.DataFrame({"lucode": [1, 2, 3], "lulc_veg": [1, 1, 0], "root_depth": [5000, 1500, 1],
    ...                     "kc": [1.0, 0.65, 1.05]})
    >>> rain, et0 = lulc.copy(data=np.full((2, 3), 1500.0)), lulc.copy(data=np.full((2, 3), 1200.0))
    >>> soil, pawc = lulc.copy(data=np.full((2, 3), 2000.0)), lulc.copy(data=np.full((2, 3), 0.2))
    >>> wy = ls.services.water_yield(lulc, bio, rain, et0, soil, pawc, seasonality=5.0)
    >>> wy.wyield.values.round(1)                                # mm per year
    array([[582.6, 582.6, 864.3],
           [582.6, 864.3, 240. ]])
    """
    threads = resolve(n_threads)
    lu = LandUse(lulc, nodata, "the land use map")
    shape = lu.shape
    codes, bio = lookup_table(biophysical, ["lulc_veg", "root_depth", "kc"], "the biophysical table")
    veg = bio["lulc_veg"].to_numpy()
    if not np.all(np.isin(veg, (0, 1))):
        raise ValueError("lulc_veg is 1 (vegetation) or 0")
    kc = np.ascontiguousarray(bio["kc"].to_numpy().astype(np.float32))
    root = np.ascontiguousarray(np.where(veg == 1, bio["root_depth"].to_numpy(), 1.0).astype(np.float32))
    veg32 = np.ascontiguousarray(veg.astype(np.float32))
    if not seasonality > 0:
        raise ValueError("seasonality (Z) must be positive")
    cls = lu.classes(codes, threads)
    wide = any(_wide(x) for x in (eto, precipitation, depth, pawc))
    t = np.float64 if wide else np.float32
    e, p, s, w = (_values(x, shape, what, t) for x, what in (
        (eto, "the reference evapotranspiration"), (precipitation, "the precipitation"), (depth, "the soil depth"),
        (pawc, "the plant available water content")))
    fractp = _core.serv_water_fraction(cls, kc, root, veg32, e, p, s, w, float(seasonality), wide, threads)
    pw = _wide(precipitation)
    wyield, aet = _core.serv_water_yield(fractp, p if pw == wide else _values(precipitation, shape, "", np.float64
                                                                                if pw else np.float32), pw, threads)
    ew = _wide(eto)
    pet = _core.serv_potential_et(cls, kc, e if ew == wide else _values(eto, shape, "", np.float64 if ew
                                                                        else np.float32), ew, threads)
    out = WaterYield(lu.raster(wyield, "wyield"), lu.raster(aet, "aet"), lu.raster(pet, "pet"),
                     lu.raster(fractp, "fractp"))
    maps = {"precip_mn": p, "PET_mn": pet, "AET_mn": aet, "wyield_mn": wyield}
    if demand is not None:
        dcodes, dt = lookup_table(demand, ["demand"], "the demand table")
        dcls = lu.classes(dcodes, threads)
        dv = dt["demand"].to_numpy().astype(np.float32)
        dmap = np.where(dcls >= 0, dv[np.maximum(dcls, 0)], np.float32(np.nan)).astype(np.float32)
        out.demand = lu.raster(dmap, "demand")
        maps["demand"] = dmap
    if valuation is not None and demand is None:
        raise ValueError("the valuation values the realized supply: give the demand table too")
    if watersheds is None and (subwatersheds is not None or valuation is not None):
        raise ValueError("give the watersheds")
    if watersheds is not None:
        like = lu.raster(np.zeros(shape), "x")
        dx, dy = lu.pixel()
        precip_map = read_surface(precipitation).values  # the precipitation's own cells, rain or not
        maps["precip_mn"] = precip_map
        out.watersheds = _sheds(watersheds, "ws_id", maps, like, dx * dy, valuation, threads)
        if subwatersheds is not None:
            out.subwatersheds = _sheds(subwatersheds, "subws_id", maps, like, dx * dy, None, threads)
    return out
