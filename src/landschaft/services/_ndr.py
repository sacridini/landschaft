# SPDX-License-Identifier: GPL-3.0-or-later
"""Nutrient delivery ratio (InVEST's ``ndr`` model).

Each raster of InVEST's chain is computed in the type InVEST writes it in (float32 or
float64), with its expression: the elevations filled and routed by pygeoprocessing's
algorithms (ported in ``ls.hydro``), the percent slope and the effective retention in
C++, and the cell by cell algebra in numpy, as ``pygeoprocessing.raster_map`` applies it
to the valid cells. The routing and the index of connectivity are those of the sediment
delivery ratio (``ls.services.sdr``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ..surface._grid import read_surface
from ._common import LandUse, lookup_table
from ._sdr import BLOCK, _inside, _report, _rmap, _routing, _thresholds

__all__ = ["NDR", "ndr"]

LOAD_TYPES = ("application-rate", "measured-runoff")
# the watershed totals, in the order InVEST writes them
FIELDS = {"n": ["n_surface_load", "n_surface_export", "n_subsurface_export", "n_total_export", "n_subsurface_load"],
          "p": ["p_surface_load", "p_surface_export"]}


@dataclass
class NDR(Saveable):
    """Nutrient export and retention of a landscape (:func:`ndr`).

    Maps in kilograms per hectare and year, NaN where a cell has no value; the
    surface exports and the delivery ratios have none on the streams, as in
    InVEST. The maps of a nutrient not modelled are ``None``.

    Attributes
    ----------
    n_surface_export, p_surface_export : xarray.DataArray
        The surface load that reaches the streams: load x NDR (float32).
    n_subsurface_export : xarray.DataArray
        The dissolved nitrogen that reaches the streams through the soil (float64).
    n_total_export : xarray.DataArray
        Surface plus subsurface export of nitrogen (float64).
    ndr_n, ndr_p : xarray.DataArray
        The surface nutrient delivery ratio, ``(1 - eff') / (1 + exp((IC_0 - IC) / k))``.
    effective_retention_n, effective_retention_p : xarray.DataArray
        The share of the load retained on its way down to a stream.
    ic : xarray.DataArray
        Index of connectivity, log10(D_up / D_dn).
    stream : xarray.DataArray
        The streams (1; 0 elsewhere).
    drains_to_stream : xarray.DataArray
        1 where the flow reaches a stream, 0 where it does not.
    watersheds : geopandas.GeoDataFrame
        The watersheds with the loads and exports in kilograms per year
        (``n_surface_load``, ``n_surface_export``, ``n_subsurface_export``,
        ``n_total_export``, ``n_subsurface_load``, ``p_surface_load``,
        ``p_surface_export``; those of the nutrients modelled).
    ic_0 : float
        The IC_0 of the delivery ratio, half way between the lowest and the
        highest IC.
    runoff_proxy_mean : float
        The mean the runoff proxy was divided by.
    intermediate : dict
        The other rasters of the chain under InVEST's names (``load_n``,
        ``modified_load_n``, ``surface_load_n``, ``sub_load_n``, ``sub_ndr_n``,
        ``eff_n``, ``crit_len_n``, ``runoff_proxy_index``, ``slope``,
        ``thresholded_slope``, ``flow_accumulation``, ``s_accumulation``,
        ``s_bar``, ``d_up``, ``s_factor_inverse``, ``d_dn``,
        ``dist_to_channel``, ``filled_dem``, ``flow_direction``...), in its types.
    """

    n_surface_export: xr.DataArray | None
    n_subsurface_export: xr.DataArray | None
    n_total_export: xr.DataArray | None
    p_surface_export: xr.DataArray | None
    ndr_n: xr.DataArray | None
    ndr_p: xr.DataArray | None
    effective_retention_n: xr.DataArray | None
    effective_retention_p: xr.DataArray | None
    ic: xr.DataArray
    stream: xr.DataArray
    drains_to_stream: xr.DataArray
    watersheds: object
    ic_0: float
    runoff_proxy_mean: float
    intermediate: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        parts = []
        for col, label in (("n_total_export", "N export"), ("p_surface_export", "P export")):
            if col in self.watersheds:
                parts.append(f"{label} {self.watersheds[col].sum():.6g} kg/yr")
        return f"NDR({', '.join(parts)})"


def _iterblocks(ny: int, nx: int, block: int = BLOCK, largest: int = 2 ** 16):
    """pygeoprocessing.iterblocks over a raster tiled ``block`` x ``block``: its windows."""
    cols, rows = block, block
    area = cols * rows
    if largest // area > 0:
        cols = min(cols * (largest // area), nx)
        area = cols * rows
    if largest // area > 0:
        rows = min(rows * (largest // area), ny)
    for y0 in range(0, ny, rows):
        for x0 in range(0, nx, cols):
            yield slice(y0, min(ny, y0 + rows)), slice(x0, min(nx, x0 + cols))


def _mean(values: np.ndarray) -> np.float32:
    """InVEST's _normalize_raster mean: float32 block sums of the valid cells, in block order."""
    total, count = 0, 0
    for ys, xs in _iterblocks(*values.shape):
        block = values[ys, xs]
        data = block[~np.isnan(block)]
        total, count = total + np.sum(data), count + data.size
    mean = total
    if count > 0:
        mean /= count
    return mean


def _table(biophysical, nutrients):
    """The biophysical table: the codes, the numeric columns and the load types of each nutrient."""
    if isinstance(biophysical, dict):
        df = pd.DataFrame.from_dict(biophysical, orient="index")
    elif isinstance(biophysical, pd.DataFrame):
        df = biophysical.copy()
    else:
        raise TypeError(f"the biophysical table is a DataFrame (or a dict by class code), not "
                        f"{type(biophysical).__name__}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    numeric = []
    for x in nutrients:
        numeric += [f"load_{x}", f"eff_{x}", f"crit_len_{x}"]
    if "n" in nutrients:
        numeric.append("proportion_subsurface_n")
    missing = [f"load_type_{x}" for x in nutrients if f"load_type_{x}" not in df.columns]
    if missing:
        raise ValueError(f"the biophysical table lacks the column(s) {missing}")
    codes, values = lookup_table(df[[c for c in df.columns if not c.startswith("load_type_")]], numeric,
                                 "the biophysical table")
    by_code = df.set_index("lucode") if "lucode" in df.columns else df
    by_code.index = pd.Index([int(v) for v in by_code.index])
    types = {}
    for x in nutrients:
        t = by_code[f"load_type_{x}"].astype(str).str.strip().str.lower().reindex(codes).to_numpy()
        bad = sorted(set(t) - set(LOAD_TYPES))
        if bad:
            raise ValueError(f"load_type_{x} is 'application-rate' or 'measured-runoff', not {bad}")
        types[x] = t
    return codes, values, types


def ndr(dem, lulc, runoff_proxy, biophysical, watersheds, *, threshold: float, k: float = 2.0,
        nitrogen: bool = True, phosphorus: bool = True, subsurface_critical_length: float | None = None,
        subsurface_efficiency: float | None = None, runoff_proxy_mean: float | None = None, method: str = "mfd",
        nodata=None, n_threads: int | None = None) -> NDR:
    """Nutrient export and retention (InVEST's nutrient delivery ratio model).

    The nitrogen and phosphorus loads of each land use (scaled by the runoff
    proxy over its mean) split into a surface part and, for nitrogen, a dissolved
    subsurface part. The surface load reaches a stream in the share
    ``NDR = (1 - eff') / (1 + exp((IC_0 - IC) / k))``: ``eff'`` the retention of
    the cells it crosses on its way down (each land use's maximum efficiency
    reached over its critical length), ``IC = log10(D_up / D_dn)`` the index of
    connectivity of the slope (Borselli et al. 2008) and ``IC_0`` half way between
    its lowest and highest values. The subsurface load reaches it in the share
    ``1 - eff_sub (1 - exp(-5 l / l_sub))``, ``l`` the distance down to the stream.

    The rasters are InVEST's (natcap.invest 3.20) bit for bit, in its float32 and
    float64 types: the DEM filled, routed (pygeoprocessing's multiple flow
    directions or D8) and traced into streams as pygeoprocessing does, its percent
    slope and the effective retention; the watershed totals agree to float32
    precision. One correction: InVEST measures the distance to the stream of the
    subsurface delivery in cells against a critical length in metres; here it is in
    metres (cells x cell size).

    Parameters
    ----------
    dem
        Elevations (projected, in metres).
    lulc
        Land use map (integer codes), on the DEM's grid.
    runoff_proxy
        The capacity of the runoff to carry the nutrients (a quickflow index or the
        annual precipitation, any units), on the DEM's grid.
    biophysical
        A DataFrame by class (``lucode``): for nitrogen ``load_n`` (kg / ha / yr),
        ``load_type_n`` (``"application-rate"``: the load is multiplied by
        1 - ``eff_n``; ``"measured-runoff"``: as given), ``eff_n`` (the maximum
        retention efficiency, 0 to 1), ``crit_len_n`` (m) and
        ``proportion_subsurface_n``; for phosphorus the same columns ending in
        ``_p`` (no subsurface part).
    watersheds
        Polygons (GeoDataFrame): cells outside them are left out (as InVEST masks
        the inputs to them), and the totals are given per polygon.
    threshold
        Flow accumulation (cells) from which a cell is a stream.
    k
        Borselli's calibration parameter of the NDR.
    nitrogen, phosphorus
        The nutrients modelled.
    subsurface_critical_length, subsurface_efficiency
        The distance (m) over which the soil reaches its maximum retention of the
        dissolved nitrogen, and that maximum (0 to 1); required with ``nitrogen``.
    runoff_proxy_mean
        The mean of the runoff proxy; computed from the map if not given.
    method
        ``"mfd"`` (multiple flow directions, the default) or ``"d8"``.
    nodata
        Value marking missing cells in the land use map.

    Returns
    -------
    NDR

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import geopandas as gpd
    >>> from affine import Affine
    >>> from shapely.geometry import box
    >>> import landschaft as ls
    >>> y, x = np.mgrid[:40, :50]
    >>> z = (100 + 0.5 * x + 8 * np.sin(y / 5.0)).astype("float32")      # a slope with ridges
    >>> t = Affine(30, 0, 500000, 0, -30, 8000000)
    >>> dem = ls.load_raster(ls.write_raster(z, "ndr_dem.tif", crs="EPSG:31983", transform=t))
    >>> lulc = dem.copy(data=np.where(x < 25, 1, 2).astype("int16"))    # forest, crops
    >>> rain = dem.copy(data=np.full(z.shape, 1200, "float32"))
    >>> bio = pd.DataFrame({"lucode": [1, 2], "load_n": [2.0, 40.0], "load_type_n": "measured-runoff",
    ...                     "eff_n": [0.8, 0.2], "crit_len_n": [150.0, 30.0], "proportion_subsurface_n": [0.1, 0.4],
    ...                     "load_p": [0.1, 3.0], "load_type_p": "measured-runoff", "eff_p": [0.8, 0.3],
    ...                     "crit_len_p": [150.0, 30.0]})
    >>> ws = gpd.GeoDataFrame({"ws_id": [1]}, geometry=[box(*dem.rio.bounds())], crs=dem.rio.crs)
    >>> r = ls.services.ndr(dem, lulc, rain, bio, ws, threshold=50, subsurface_critical_length=200,
    ...                     subsurface_efficiency=0.8)
    >>> crops, forest = np.nanmean(r.n_total_export.values[:, 30:]), np.nanmean(r.n_total_export.values[:, :20])
    >>> bool(crops > 10 * forest)                     # crops export far more nitrogen (kg / ha / yr)
    True
    >>> list(r.watersheds.columns)[2:4]
    ['n_surface_load', 'n_surface_export']
    """
    threads = resolve(n_threads)
    method = str(method).lower()
    if method not in ("mfd", "d8"):
        raise ValueError("method is 'mfd' or 'd8'")
    if not threshold > 0:
        raise ValueError("threshold must be positive")
    nutrients = [x for x, on in (("n", nitrogen), ("p", phosphorus)) if on]
    if not nutrients:
        raise ValueError("model nitrogen, phosphorus or both")
    if nitrogen:
        if subsurface_critical_length is None or subsurface_efficiency is None:
            raise ValueError("nitrogen needs the subsurface_critical_length and the subsurface_efficiency")
        if not subsurface_critical_length > 0:
            raise ValueError("the subsurface critical length must be positive")
        if not 0 <= subsurface_efficiency <= 1:
            raise ValueError("the subsurface efficiency is between 0 and 1")
    if runoff_proxy_mean is not None and not runoff_proxy_mean > 0:
        raise ValueError("the runoff proxy mean must be positive")
    lu = LandUse(lulc, nodata, "the land use map")
    shape = lu.shape
    codes, bio, types = _table(biophysical, nutrients)
    values = {}
    for name, x in (("dem", dem), ("runoff proxy", runoff_proxy)):
        s = read_surface(x)
        if s.shape != shape:
            raise ValueError(f"the {name} raster has {s.shape} cells, the land use map {shape}")
        values[name] = s.values
    # InVEST masks each input to the watersheds (cell centres) on its own
    like = lu.raster(np.zeros(shape, np.float32), "mask")
    inside = _inside(watersheds, like)
    dem32 = np.where(inside & ~np.isnan(values["dem"]), values["dem"], np.nan).astype(np.float32)
    proxy = np.where(inside & ~np.isnan(values["runoff proxy"]), values["runoff proxy"], np.nan).astype(np.float32)
    proxy[proxy == -1] = np.nan  # the masked proxy's nodata
    lulc_ok = inside & lu.valid
    dx, dy = lu.pixel()
    try:
        tr = lu.surface.template.rio.transform()
        xcell, ycell = float(tr.a), float(tr.e)
    except Exception:  # noqa: BLE001
        xcell, ycell = dx, -dy
    cell_area = abs(xcell * ycell)

    # routing and the index of connectivity
    filled = _core.hydro_fill(dem32.astype(np.float64), False).astype(np.float32).astype(np.float64)
    slope = _core.serv_pgp_slope(filled, xcell, ycell, threads)
    slope_thr = _thresholds(slope)
    d, acc, accumulate, distance, dir_missing = _routing(filled, method)
    if method == "mfd":  # InVEST's NDR keeps pygeoprocessing's default trace proportion (its SDR passes 0.7)
        stream = _core.hydro_mfd_streams(acc, d, float(threshold), 1.0, BLOCK)
    else:
        stream = np.full(shape, 255, np.uint8)
        ok = ~np.isnan(acc)
        stream[ok] = (acc[ok] > threshold).astype(np.uint8)
    s_acc = accumulate(slope_thr.astype(np.float64))
    s_bar = _rmap(np.divide, [s_acc, acc], np.float32)
    d_up = _rmap(lambda s, f: s * np.sqrt(f * cell_area), [s_bar, acc], np.float32)
    s_inv = _rmap(lambda b: np.where(b == 0, 0, 1 / b), [slope_thr], np.float32)
    d_dn = distance(stream, s_inv.astype(np.float64))
    dist = distance(stream, None)
    drains = np.full(shape, 255, np.uint8)
    drains[~dir_missing] = (~np.isnan(dist[~dir_missing])).astype(np.uint8)
    ok = ~np.isnan(d_up) & ~np.isnan(d_dn) & (d_up != 0) & (d_dn != 0)
    ic = np.full(shape, np.nan, np.float32)
    ic[ok] = np.log10(d_up[ok] / d_dn[ok])
    if not ok.any():
        raise ValueError("no cell has an index of connectivity: no cell drains to a stream")
    # InVEST reads the IC's extremes from the statistics pygeoprocessing stores in the GeoTIFF: as text, "%.14g"
    ic_0 = (float(f"{float(np.min(ic[ok])):.14g}") + float(f"{float(np.max(ic[ok])):.14g}")) / 2

    # the runoff proxy index
    mean = _mean(proxy) if runoff_proxy_mean is None else float(runoff_proxy_mean)
    rpi = _rmap(lambda a: a if mean == 0 else a / mean, [proxy], np.float32)

    index, first = _core.serv_classify(lu.codes, lulc_ok.astype(np.uint8), codes, threads)
    if first >= 0:
        missing = sorted(set(np.unique(lu.codes[lulc_ok & (index == -2)]).tolist()))
        raise ValueError(f"the land use map has class(es) {missing} that the biophysical table lacks")
    has = index >= 0
    idx = np.maximum(index, 0)
    stream_ok = stream != 255
    not_stream = (1 - stream).astype(np.uint8)
    k = float(k)
    maps, eff_crit = {}, {}
    for x in nutrients:
        load = bio[f"load_{x}"].to_numpy(np.float64)
        eff = bio[f"eff_{x}"].to_numpy(np.float64)
        load = np.where(types[x] == "application-rate", load * (1 - eff), load)
        load_map = np.full(shape, np.nan, np.float32)
        load_map[has] = load[idx[has]]
        modified = _rmap(lambda a, b: np.prod(np.stack([a, b]), axis=0), [load_map, rpi], np.float32)
        if x == "n":
            sub = bio["proportion_subsurface_n"].to_numpy(np.float64)
            ok = ~np.isnan(modified)
            surface = np.full(shape, np.nan, np.float32)
            surface[ok] = modified[ok] * (1 - sub[idx[ok]])
            sub_load = np.full(shape, np.nan, np.float32)
            sub_load[ok] = modified[ok] * sub[idx[ok]]
            maps["sub_load_n"] = sub_load
        else:
            surface = modified.copy()
        # the efficiency and critical length of each land use, 0 on the streams
        valid = has & stream_ok
        eff_map, crit_map = np.full(shape, np.nan, np.float32), np.full(shape, np.nan, np.float32)
        for out, col in ((eff_map, f"eff_{x}"), (crit_map, f"crit_len_{x}")):
            v = bio[col].to_numpy(np.float64)
            out[valid] = v[idx[valid]] * not_stream[valid]
        maps.update({f"load_{x}": load_map, f"modified_load_{x}": modified, f"surface_load_{x}": surface,
                     f"eff_{x}": eff_map, f"crit_len_{x}": crit_map})
        eff_crit[x] = (eff_map, crit_map)

    def retain(x):
        e, c = eff_crit[x]
        return _core.serv_ndr_retention(d, stream, e, c, abs(xcell), BLOCK)

    with ThreadPoolExecutor(max_workers=max(1, min(threads, len(nutrients)))) as pool:
        retained = dict(zip(nutrients, pool.map(retain, nutrients)))
    for x in nutrients:
        ret = retained[x]
        ratio = _rmap(lambda e, i: (1 - e) / (1 + np.exp((ic_0 - i) / k)), [ret, ic], np.float32)
        maps[f"effective_retention_{x}"] = ret
        maps[f"ndr_{x}"] = ratio
        maps[f"{x}_surface_export"] = _rmap(np.multiply, [maps[f"surface_load_{x}"], ratio], np.float32)
    if "n" in nutrients:
        eff_sub, crit_sub, cell = float(subsurface_efficiency), float(subsurface_critical_length), abs(xcell)
        sub_ndr = _rmap(lambda dc: 1 - eff_sub * (1 - np.exp(-5 * (dc * cell) / crit_sub)), [dist], np.float64)
        maps["sub_ndr_n"] = sub_ndr
        maps["n_subsurface_export"] = _rmap(np.multiply, [maps["sub_load_n"], sub_ndr], np.float64)
        maps["n_total_export"] = _rmap(lambda a, b: np.sum([a, b], axis=0),
                                       [maps["n_surface_export"], maps["n_subsurface_export"]], np.float64)

    totals = {"n_surface_load": "surface_load_n", "n_surface_export": "n_surface_export",
              "n_subsurface_export": "n_subsurface_export", "n_total_export": "n_total_export",
              "n_subsurface_load": "sub_load_n", "p_surface_load": "surface_load_p",
              "p_surface_export": "p_surface_export"}
    fields = [f for x in nutrients for f in FIELDS[x]]
    sheds = _report(watersheds, {f: maps[totals[f]] for f in fields}, like, cell_area / 10000, threads)
    ras = lu.raster
    masked_lulc = np.where(lulc_ok, lu.codes, np.iinfo(np.int32).min).astype(np.int32)
    inter = {"masked_dem": dem32, "masked_lulc": masked_lulc, "masked_runoff_proxy": proxy,
             "filled_dem": filled.astype(np.float32), "slope": slope, "thresholded_slope": slope_thr,
             "flow_direction": d, "flow_accumulation": acc, "s_accumulation": s_acc, "runoff_proxy_index": rpi,
             "s_bar": s_bar, "d_up": d_up, "s_factor_inverse": s_inv, "d_dn": d_dn, "dist_to_channel": dist,
             **{k_: v for k_, v in maps.items() if not k_.endswith("_export") and not k_.startswith("ndr_")
                and not k_.startswith("effective_retention_")}}

    def get(key):
        return ras(maps[key], key) if key in maps else None

    return NDR(get("n_surface_export"), get("n_subsurface_export"), get("n_total_export"), get("p_surface_export"),
               get("ndr_n"), get("ndr_p"), get("effective_retention_n"), get("effective_retention_p"),
               ras(ic, "ic"), ras(stream, "stream"), ras(drains, "drains_to_stream"), sheds, ic_0, float(mean),
               {k_: ras(v, k_) for k_, v in inter.items()})
