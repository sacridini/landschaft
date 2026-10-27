# SPDX-License-Identifier: GPL-3.0-or-later
"""Sediment delivery ratio (InVEST's ``sdr`` model) and the index of connectivity.

Each raster of InVEST's chain is computed in the type InVEST writes it in (float32 or
float64), with its expression: the elevations filled and routed by pygeoprocessing's
algorithms (ported in ``ls.hydro``), its percent slope and the sediment deposition in
C++, and the cell by cell algebra in numpy, as ``pygeoprocessing.raster_map`` applies it
to the valid cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ..surface._grid import read_surface
from ._common import LandUse, align_inputs, lookup_table

__all__ = ["SDR", "sdr"]

BLOCK = 256  # pygeoprocessing's tiles: the order its routing and InVEST's deposition visit the cells


def _rmap(op, arrays, dtype, valid=None):
    """pygeoprocessing.raster_map: ``op`` on the cells where every input has a value."""
    ok = np.ones(arrays[0].shape, bool) if valid is None else valid.copy()
    for a in arrays:
        if a.dtype.kind == "f":
            ok &= ~np.isnan(a)
    out = np.full(arrays[0].shape, np.nan, dtype=dtype)
    out[ok] = op(*[a[ok] for a in arrays])
    return out


def _dtype_of(x):
    d = getattr(x, "dtype", None)
    return np.asarray(x).dtype if d is None else np.dtype(d)


def _ls_factor(percent_slope, flow_accumulation, cell_size, cell_area, l_max):
    """InVEST's ls_factor_function, as it is written (float32 slope, float64 accumulation)."""
    contributing_area = np.sqrt((flow_accumulation - 1) * cell_area)
    slope_in_radians = np.arctan(percent_slope / 100)
    aspect_length = (np.fabs(np.sin(slope_in_radians)) + np.fabs(np.cos(slope_in_radians)))
    slope_factor = np.where(percent_slope < 9, 10.8 * np.sin(slope_in_radians) + 0.03,
                            16.8 * np.sin(slope_in_radians) - 0.5)
    beta = ((np.sin(slope_in_radians) / 0.0896) / (3 * np.sin(slope_in_radians) ** 0.8 + 0.56))
    slope_table = np.array([1., 3.5, 5., 9.])
    m_table = np.array([0.2, 0.3, 0.4, 0.5])
    big_slope_mask = percent_slope > slope_table[-1]
    m_indexes = np.digitize(percent_slope[~big_slope_mask], slope_table, right=True)
    m_exp = np.empty(big_slope_mask.shape, dtype=np.float32)
    m_exp[big_slope_mask] = (beta[big_slope_mask] / (1 + beta[big_slope_mask]))
    m_exp[~big_slope_mask] = m_table[m_indexes]
    l_factor = (((contributing_area + cell_area) ** (m_exp + 1) - contributing_area ** (m_exp + 1))
                / ((cell_size ** (m_exp + 2)) * (aspect_length ** m_exp) * (22.13 ** m_exp)))
    l_factor[l_factor > l_max] = l_max
    return l_factor * slope_factor


def _routing(filled64, method):
    """Directions, accumulation and their functions, pygeoprocessing's MFD or D8."""
    if method == "mfd":
        d = _core.hydro_mfd_directions(filled64, BLOCK)
        acc = _core.hydro_mfd_packed_accumulation(d, None)

        def accumulate(w):
            return _core.hydro_mfd_packed_accumulation(d, w)

        def distance(channel, w):
            return _core.hydro_mfd_distance_to_channel(d, channel, w)

        missing = d == 0
    else:
        d = _core.hydro_pgp_d8(filled64, BLOCK)
        acc = _core.hydro_pgp_d8_accumulation(d, None)

        def accumulate(w):
            return _core.hydro_pgp_d8_accumulation(d, w)

        def distance(channel, w):
            return _core.hydro_pgp_d8_distance_to_channel(d, channel, w)

        missing = d == 128
    return d, acc, accumulate, distance, missing


def _thresholds(slope):
    """InVEST's threshold_slope_op (m/m, from 0.005 to 1), float32."""
    def op(s):
        m = s / 100
        m[m < 0.005] = 0.005
        m[m > 1] = 1
        return m

    return _rmap(op, [slope], np.float32)


def _connectivity(w_thr, s_thr, acc, accumulate, distance, channel, cell_area):
    """W bar, S bar, D_up, the inverse WS weight, D_dn and the IC (float32), as InVEST."""
    w_acc = accumulate(w_thr.astype(np.float64))
    s_acc = accumulate(s_thr.astype(np.float64))
    w_bar = _rmap(lambda a, f: a / f, [w_acc, acc], np.float64)
    s_bar = _rmap(lambda a, f: a / f, [s_acc, acc], np.float64)
    d_up = _rmap(lambda w, s, f: w * s * np.sqrt(f * cell_area), [w_bar, s_bar, acc], np.float64)
    ws_inv = _rmap(lambda w, s: 1 / (w * s), [w_thr, s_thr], np.float32)
    d_dn = distance(channel, ws_inv.astype(np.float64))
    ok = ~np.isnan(d_up) & ~np.isnan(d_dn) & (d_dn != 0) & (d_up != 0)
    ic = np.full(acc.shape, np.nan, np.float32)
    ic[ok] = np.log10(d_up[ok] / d_dn[ok])
    return {"w_bar": w_bar, "s_bar": s_bar, "d_up": d_up, "ws_inverse": ws_inv, "d_dn": d_dn, "ic": ic}


@dataclass
class SDR(Saveable):
    """Sediment export and retention of a landscape (:func:`sdr`).

    Maps in tons per hectare and year (float32), NaN where a cell has no value;
    ``usle``, ``rkls`` and ``avoided_erosion`` have none on the streams, as in
    InVEST.

    Attributes
    ----------
    usle : xarray.DataArray
        Soil loss, RKLS x C x P.
    sed_export : xarray.DataArray
        The soil loss that reaches the streams: USLE x SDR.
    sed_deposition : xarray.DataArray
        Sediment from upslope that the cell traps.
    avoided_export : xarray.DataArray
        Total retention: avoided erosion x SDR plus the deposition.
    avoided_erosion : xarray.DataArray
        Erosion the vegetation and practices avoid: RKLS - USLE.
    rkls : xarray.DataArray
        Soil loss of bare soil, R x K x LS.
    sdr : xarray.DataArray
        Sediment delivery ratio (1 on the streams).
    ic : xarray.DataArray
        Index of connectivity, log10(D_up / D_dn).
    stream : xarray.DataArray
        The streams (1), with the drainage given (0 elsewhere).
    drains_to_stream : xarray.DataArray
        1 where the flow reaches a stream, 0 where it does not (no D_dn).
    watersheds : geopandas.GeoDataFrame
        The watersheds with ``usle_tot``, ``sed_export``, ``sed_dep``,
        ``avoid_exp`` and ``avoid_eros`` (tons per year).
    intermediate : dict
        The other rasters of the chain (``slope``, ``flow_accumulation``,
        ``ls``, ``w``, ``cp``, ``w_bar``, ``s_bar``, ``d_up``, ``d_dn``,
        ``e_prime``, ``flux``...), in InVEST's types.
    """

    usle: xr.DataArray
    sed_export: xr.DataArray
    sed_deposition: xr.DataArray
    avoided_export: xr.DataArray
    avoided_erosion: xr.DataArray
    rkls: xr.DataArray
    sdr: xr.DataArray
    ic: xr.DataArray
    stream: xr.DataArray
    drains_to_stream: xr.DataArray
    watersheds: object
    intermediate: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        t = self.watersheds[["usle_tot", "sed_export", "sed_dep"]].sum()
        return (f"SDR(soil loss {t['usle_tot']:.6g} t/yr, export {t['sed_export']:.6g} t/yr, "
                f"deposition {t['sed_dep']:.6g} t/yr)")


def _inside(watersheds, like):
    """The cells whose centre lies in a watershed (InVEST masks the inputs to them)."""
    import geopandas as gpd

    from .._vector import rasterize

    if not isinstance(watersheds, gpd.GeoDataFrame):
        raise TypeError("the watersheds are a GeoDataFrame")
    mask = rasterize(watersheds.assign(_inside=1), like, field="_inside", fill=0)
    return np.asarray(mask.values) == 1


def sdr(dem, erosivity, erodibility, lulc, biophysical, watersheds, *, threshold: float, k: float = 2.0,
        ic_0: float = 0.5, sdr_max: float = 0.8, l_max: float = 122.0, method: str = "mfd", drainage=None,
        nodata=None, n_threads: int | None = None) -> SDR:
    """Sediment export, deposition and retention (InVEST's sediment delivery ratio model).

    The soil loss of each cell by the revised universal soil loss equation,
    ``USLE = R K LS C P`` (the LS factor of Desmet & Govers 1996 from the flow
    accumulation), and the share of it that reaches a stream, the sediment
    delivery ratio ``SDR = SDR_max / (1 + exp((IC_0 - IC) / k))`` of the index of
    connectivity ``IC = log10(D_up / D_dn)`` (Borselli et al. 2008):
    ``D_up = W S sqrt(A)`` upslope (W the C factor, S the slope, averaged over
    the contributing area A) and ``D_dn = sum d / (W S)`` down to the stream.
    The sediment that does not reach a stream is deposited downslope; the
    retention is what the vegetation and practices avoid, plus that deposition.

    The rasters are InVEST's (natcap.invest 3.20) bit for bit, in its float32 and
    float64 types: the DEM filled, routed (pygeoprocessing's multiple flow
    directions or D8) and traced into streams as pygeoprocessing does, its percent
    slope, and the deposition; the watershed totals agree to float32 precision.

    Parameters
    ----------
    dem
        Elevations (projected, in metres).
    erosivity, erodibility
        Rainfall erosivity R (MJ mm / (ha h yr)) and soil erodibility K
        (t ha h / (ha MJ mm)).
    lulc
        Land use map (integer codes).
    biophysical
        A DataFrame by class (``lucode``): ``usle_c`` and ``usle_p``.
    watersheds
        Polygons (GeoDataFrame): cells outside them are left out (as InVEST masks
        the inputs to them), and the totals are given per polygon.
    threshold
        Flow accumulation (cells) from which a cell is a stream.
    k, ic_0, sdr_max
        Borselli's calibration parameters of the SDR.
    l_max
        The largest slope length factor L.
    method
        ``"mfd"`` (multiple flow directions, the default) or ``"d8"``.
    drainage
        Cells that drain to a stream otherwise (1), such as roads or ditches.
    nodata
        Value marking missing cells in the land use map.

    Returns
    -------
    SDR

    Notes
    -----
    The rasters may be on different grids (cells, extent, CRS): as InVEST does,
    they are put on the DEM's grid first, square cells of its smaller side in its
    CRS, over the intersection of their bounds and the watersheds' (its corner
    moved onto the DEM's cells); the DEM, erosivity and erodibility bilinear,
    the land use map by its most frequent class, the drainage by the nearest
    cell. The maps cover the watersheds' box only.

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
    >>> dem = ls.load_raster(ls.write_raster(z, "sdr_dem.tif", crs="EPSG:31983", transform=t))
    >>> lulc = dem.copy(data=np.where(x < 25, 1, 2).astype("int16"))    # forest, crops
    >>> r, k = dem.copy(data=np.full(z.shape, 4000, "float32")), dem.copy(data=np.full(z.shape, 0.03, "float32"))
    >>> bio = pd.DataFrame({"lucode": [1, 2], "usle_c": [0.003, 0.25], "usle_p": [1.0, 0.8]})
    >>> ws = gpd.GeoDataFrame({"ws_id": [1]}, geometry=[box(*dem.rio.bounds())], crs=dem.rio.crs)
    >>> s = ls.services.sdr(dem, r, k, lulc, bio, ws, threshold=50)
    >>> crops, forest = np.nanmean(s.usle.values[:, 30:]), np.nanmean(s.usle.values[:, :20])
    >>> bool(crops > 10 * forest)                     # crops lose far more soil (t / ha / yr)
    True
    >>> list(s.watersheds.columns)
    ['ws_id', 'geometry', 'usle_tot', 'sed_export', 'sed_dep', 'avoid_exp', 'avoid_eros']
    """
    threads = resolve(n_threads)
    method = str(method).lower()
    if method not in ("mfd", "d8"):
        raise ValueError("method is 'mfd' or 'd8'")
    if not threshold > 0:
        raise ValueError("threshold must be positive")
    # InVEST's alignment: square cells of the DEM's smaller side, in its CRS, over the rasters and the
    # watersheds; the continuous rasters bilinear, the land use map by its most frequent class
    g = align_inputs({"dem": (dem, "bilinear"), "lulc": (lulc, "mode"), "erosivity": (erosivity, "bilinear"),
                      "erodibility": (erodibility, "bilinear"), "drainage": (drainage, "near")}, reference="dem",
                     pixel="square", vectors=[watersheds], transform_bounds=True, n_threads=threads)
    dem, lulc, erosivity, erodibility, drainage = (g["dem"], g["lulc"], g["erosivity"], g["erodibility"],
                                                   g.get("drainage"))
    lu = LandUse(lulc, nodata, "the land use map")
    shape = lu.shape
    codes, bio = lookup_table(biophysical, ["usle_c", "usle_p"], "the biophysical table")
    grids = {"dem": dem, "erosivity": erosivity, "erodibility": erodibility}
    if drainage is not None:
        grids["drainage"] = drainage
    values = {}
    for name, x in grids.items():
        s = read_surface(x)
        if s.shape != shape:
            raise ValueError(f"the {name} raster has {s.shape} cells, the land use map {shape}")
        values[name] = s.values
    # InVEST masks every input to the cells valid in all of them and inside the watersheds;
    # the masked rasters take the type of all the inputs together
    dt = np.result_type(*[_dtype_of(x) for x in grids.values()], _dtype_of(lulc))
    like = lu.raster(np.zeros(shape, np.float32), "mask")
    valid = _inside(watersheds, like) & lu.valid
    for v in values.values():
        valid &= ~np.isnan(v)
    masked = {k_: np.where(valid, v, np.nan).astype(dt) for k_, v in values.items()}
    dx, dy = lu.pixel()
    try:
        tr = lu.surface.template.rio.transform()
        xcell, ycell = float(tr.a), float(tr.e)
    except Exception:  # noqa: BLE001
        xcell, ycell = dx, -dy
    cell_size, cell_area = abs(xcell), abs(xcell) ** 2

    filled = _core.hydro_fill(masked["dem"].astype(np.float64), False).astype(dt).astype(np.float64)
    slope = _core.serv_pgp_slope(filled, xcell, ycell, threads)
    slope_thr = _thresholds(slope)
    d, acc, accumulate, distance, dir_missing = _routing(filled, method)
    if method == "mfd":
        stream = _core.hydro_mfd_streams(acc, d, float(threshold), 0.7, BLOCK)
    else:
        stream = np.full(shape, 255, np.uint8)
        ok = ~np.isnan(acc)
        stream[ok] = (acc[ok] > threshold).astype(np.uint8)
    ls_ = _rmap(lambda s, f: _ls_factor(s, f, cell_size, cell_area, float(l_max)), [slope, acc], np.float64)
    channel = stream
    if drainage is not None:
        dr = masked["drainage"]
        channel = np.full(shape, 255, np.uint8)
        ok = (stream != 255) & ~np.isnan(dr)
        channel[ok] = np.where(dr[ok] == 1, 1, stream[ok])
    index, first = _core.serv_classify(lu.codes, valid.astype(np.uint8), codes, threads)
    if first >= 0:
        missing = sorted(set(np.unique(lu.codes[valid & (index == -2)]).tolist()))
        raise ValueError(f"the land use map has class(es) {missing} that the biophysical table lacks")
    c32 = bio["usle_c"].to_numpy().astype(np.float32)
    cp32 = (bio["usle_c"] * bio["usle_p"]).to_numpy().astype(np.float32)
    w = np.where(index >= 0, c32[np.maximum(index, 0)], np.float32(np.nan)).astype(np.float32)
    cp = np.where(index >= 0, cp32[np.maximum(index, 0)], np.float32(np.nan)).astype(np.float32)
    w_thr = _rmap(lambda a: np.where(a < 0.001, 0.001, a), [w], np.float32)
    # RKLS on the cells that are not streams, in float64, stored as float32
    rk_ok = valid & ~np.isnan(ls_) & (channel != 255) & (channel == 0)
    rkls = np.full(shape, np.nan, np.float32)
    rkls[rk_ok] = ls_[rk_ok] * masked["erosivity"][rk_ok] * masked["erodibility"][rk_ok]
    usle = _rmap(lambda r, c: r * c, [rkls, cp], np.float32)
    con = _connectivity(w_thr, slope_thr, acc, accumulate, distance, channel, cell_area)
    ic = con["ic"]
    sdr_ok = ~np.isnan(ic) & (channel != 1)
    sdr_f = np.full(shape, np.nan, np.float32)
    sdr_f[sdr_ok] = sdr_max / (1 + np.exp((ic_0 - ic[sdr_ok]) / k))
    sdr_f[channel == 1] = 1
    sed_export = _rmap(np.multiply, [usle, sdr_f], np.float32)
    e_ok = ~np.isnan(usle) & ~np.isnan(sdr_f)
    e_prime = np.full(shape, np.nan, np.float32)
    e_prime[e_ok] = usle[e_ok] * (1 - sdr_f[e_ok])
    e_prime[channel == 1] = 0
    dep, flux = _core.serv_sediment_deposition(d, e_prime, sdr_f, BLOCK)
    avoided_erosion = _rmap(np.subtract, [rkls, usle], np.float32)
    avoided_export = _rmap(lambda a, s, t: a * s + t, [avoided_erosion, sdr_f, dep], np.float32)
    drains = np.full(shape, 255, np.uint8)
    drains[~dir_missing] = (~np.isnan(con["d_dn"][~dir_missing])).astype(np.uint8)

    ras = lu.raster
    sheds = _report(watersheds, {"usle_tot": usle, "sed_export": sed_export, "sed_dep": dep,
                                 "avoid_exp": avoided_export, "avoid_eros": avoided_erosion}, like,
                    abs(xcell * ycell) / 10000, threads)
    inter = {"slope": slope, "slope_threshold": slope_thr, "flow_accumulation": acc, "ls": ls_, "w": w,
             "w_threshold": w_thr, "cp": cp, "e_prime": e_prime, "flux": flux, "pit_filled_dem": filled.astype(dt),
             **{k_: v for k_, v in con.items() if k_ != "ic"}}
    return SDR(ras(usle, "usle"), ras(sed_export, "sed_export"), ras(dep, "sed_deposition"),
               ras(avoided_export, "avoided_export"), ras(avoided_erosion, "avoided_erosion"), ras(rkls, "rkls"),
               ras(sdr_f, "sdr"), ras(ic, "ic"), ras(channel, "stream"), ras(drains, "drains_to_stream"), sheds,
               {k_: ras(v, k_) for k_, v in inter.items()})


def _report(watersheds, maps: dict, like, ha_per_cell, threads):
    """The totals of the maps per watershed, in tons per year (InVEST's _generate_report)."""
    from .._zonal import zonal_stats

    names = list(maps)
    stack = xr.concat([like.copy(data=np.asarray(maps[k], np.float64)) for k in names], dim="band")
    stack = stack.assign_coords(band=names)
    stats = zonal_stats(stack, watersheds, ["sum"], n_threads=threads)
    out = watersheds.copy()
    for k in names:
        out[k] = stats[f"{k}_sum"].to_numpy(np.float64) * ha_per_cell
    return out


def connectivity_index(dem, weight, *, streams=None, threshold: float | None = None, method: str = "mfd",
                       n_threads: int | None = None) -> xr.DataArray:
    """The index of connectivity of Borselli et al. (2008), as InVEST's SDR model computes it.

    ``IC = log10(D_up / D_dn)``: ``D_up = W S sqrt(A)``, the weight W and the slope
    S (m/m) averaged over the contributing area A of each cell, and
    ``D_dn = sum d_i / (W_i S_i)`` along the flow down to a stream. Higher values:
    cells better connected to the streams. The DEM is filled and routed by
    pygeoprocessing's algorithms (multiple flow directions or D8), the slope is
    pygeoprocessing's percent slope; slopes are kept between 0.005 and 1 and the
    weights from 0.001 (Cavalli et al. 2013), as in InVEST.

    Parameters
    ----------
    dem
        Elevations (projected, in metres).
    weight
        The impedance W, usually the USLE C factor of the land cover (0 to 1),
        on the DEM's grid.
    streams
        The stream cells (1), on the DEM's grid; or ``threshold``.
    threshold
        Flow accumulation (cells) from which a cell is a stream (traced as InVEST).
    method
        ``"mfd"`` (default) or ``"d8"``.

    Returns
    -------
    xarray.DataArray
        The index (float32), NaN on the streams and where the flow reaches none.

    Examples
    --------
    >>> import numpy as np
    >>> from affine import Affine
    >>> import landschaft as ls
    >>> y, x = np.mgrid[:30, :40]
    >>> z = (50 + 0.8 * x + 3 * np.cos(y / 4.0)).astype("float32")
    >>> dem = ls.load_raster(ls.write_raster(z, "ic_dem.tif", crs="EPSG:31983",
    ...                                      transform=Affine(30, 0, 500000, 0, -30, 8000000)))
    >>> c = dem.copy(data=np.where(x > 20, 0.2, 0.01))                   # crops uphill of forest
    >>> ic = ls.hydro.connectivity_index(dem, c, threshold=30)
    >>> bool(np.nanmean(ic.values[:, 25:]) > np.nanmean(ic.values[:, 5:15]))
    True
    """
    threads = resolve(n_threads)
    method = str(method).lower()
    s = read_surface(dem)
    z = s.values
    wv = read_surface(weight).values
    if wv.shape != z.shape:
        raise ValueError("the weight differs in shape from the DEM")
    try:
        tr = s.template.rio.transform()
        xcell, ycell = float(tr.a), float(tr.e)
    except Exception:  # noqa: BLE001
        xcell, ycell = s.res[0], -s.res[1]
    filled = _core.hydro_fill(z, False)
    slope = _core.serv_pgp_slope(filled, xcell, ycell, threads)
    d, acc, accumulate, distance, _ = _routing(filled, method)
    if streams is not None:
        sv = read_surface(streams).values
        channel = np.where(np.isnan(sv), 255, (sv == 1).astype(np.uint8)).astype(np.uint8)
    elif threshold is not None:
        if method == "mfd":
            channel = _core.hydro_mfd_streams(acc, d, float(threshold), 0.7, BLOCK)
        else:
            channel = np.full(z.shape, 255, np.uint8)
            ok = ~np.isnan(acc)
            channel[ok] = (acc[ok] > threshold).astype(np.uint8)
    else:
        raise ValueError("give the streams or a threshold")
    w32 = np.where(np.isnan(wv), np.nan, wv).astype(np.float32)
    w_thr = _rmap(lambda a: np.where(a < 0.001, 0.001, a), [w32], np.float32)
    con = _connectivity(w_thr, _thresholds(slope), acc, accumulate, distance, channel, abs(xcell) ** 2)
    return s.raster(con["ic"], "ic")
