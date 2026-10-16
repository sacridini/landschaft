# SPDX-License-Identifier: GPL-3.0-or-later
"""From QSWAT+'s tables to a SWAT+ project: what the SWAT+ Editor 4.0.2 writes.

A port of the editor's GIS import (``actions/import_gis.py``), of its observed weather
import (``actions/import_weather.py``: stations at the coordinates of the weather files,
each object matched to the nearest station) and of its file writers (``fileio/*.py``),
for the projects QSWAT+ makes without landscapes, lakes, reservoirs or point sources.
The SWAT+ Editor (github.com/swat-model/swatplus-editor) and its datasets database are
public domain; the files the editor writes the same for every project (parameter
databases, decision tables, defaults) come from them (``data/swatplus_editor.zip``,
``tools/swat/make_editor_data.py``).
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import re
import zipfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from ._project import SwatProject
from ._weather import SwatClimate, _time_sim, wgn_records

__all__ = ["from_tables"]

_DATA = Path(__file__).parent / "data" / "swatplus_editor.zip"
EDITOR_VERSION = "4.0.2"


@functools.lru_cache(maxsize=1)
def _editor_data() -> tuple[dict, dict]:
    with zipfile.ZipFile(_DATA) as z:
        static = {n[len("static/"):]: z.read(n).decode() for n in z.namelist() if n.startswith("static/")}
        data = json.loads(z.read("datasets.json"))
    return static, data


# ------------------------------------------------------------------ helpers/utils.py
def _remove_space(s):
    return s if s is None or s == "" else s.strip().replace(" ", "_")


def _is_number(v) -> bool:
    if isinstance(v, bool):
        return False
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def spad(val, pad=16, right=True, null="null", after=2) -> str:
    text = null if val is None or val == "" else _remove_space(str(val))
    return (str(text).rjust(pad) if right else str(text).ljust(pad)) + " " * after


def npad(val, dec=5, pad=12, right=True, null="0", nzmin=False, nzmin_val=0.00001) -> str:
    text = val
    if _is_number(val):
        if nzmin and val < nzmin_val:
            val = nzmin_val
        text = "{:.{prec}f}".format(float(val), prec=dec)
    return spad(text, pad, right, null)


def ipad(val, pad=8, right=True) -> str:
    return npad(val, 0, pad, right, "0")


def epad(val, dec=4, pad=12, right=True, nzmin=False) -> str:
    text = val
    if _is_number(val):
        if nzmin and val == 0:
            val = 0.00000001
        text = "{:.{prec}E}".format(Decimal(val), prec=dec)
    return spad(text, pad, right, "0")


def yn(val, pad=12, right=True) -> str:
    return spad("y" if val else "n", pad, right)


def _name(prefix: str, i: int, maxid: int) -> str:
    """import_gis.get_name."""
    return f"{prefix}{str(i).zfill(len(str(maxid)))}"


def _slope_len(slope: float) -> float:
    """import_gis.get_slope_len (slope in percent)."""
    if slope < 1:
        return 121.0
    if slope < 3:
        return 90.0
    if slope < 5:
        return 60.0
    if slope < 8:
        return 30.0
    return 10.0


def _hydrology(hyd_grp: str, slope: float) -> dict:
    """database/project/hydrology.Hydrology_hyd.get_perco_cn3_swf_latq_co."""
    leach, runoff = "low", "low"
    if hyd_grp == "A":
        leach = "high"
        runoff = "low" if slope < 6 else "mod" if slope <= 12 else "high"
    elif hyd_grp == "B":
        leach = "high" if slope < 6 else "mod"
        runoff = "low" if slope < 4 else "mod" if slope <= 6 else "high"
    elif hyd_grp == "C":
        leach = "mod" if slope < 12 else "low"
        runoff = "low" if slope < 2 else "mod" if slope <= 6 else "high"
    elif hyd_grp == "D":
        runoff = "low" if slope < 2 else "mod" if slope <= 4 else "high"
    return {"perco": 0.9 if leach == "high" else 0.5 if leach == "mod" else 0.05,
            "cn3_swf": 0 if runoff == "high" else 0.3 if runoff == "mod" else 0.95,
            "latq_co": 0.9 if runoff == "high" else 0.2 if runoff == "mod" else 0.01}


def _sta_name(lat: float, lon: float, prefix="s", mult=1000) -> str:
    """import_weather.weather_sta_name."""
    return "{p}{lat}{a}{lon}{b}".format(p=prefix, lat=abs(round(lat * mult)), a="n" if lat >= 0 else "s",
                                        lon=abs(round(lon * mult)), b="e" if lon >= 0 else "w")


def _nearest(lat, lon, cands) -> int:
    """import_weather.update_closest_lat_lon: the first of the nearest (squared degrees)."""
    best, bi = None, -1
    for i, (la, lo) in enumerate(cands):
        d = (lat - la) * (lat - la) + (lon - lo) * (lon - lo)
        if best is None or d < best:
            best, bi = d, i
    return bi


def _ele_ids(positions: list[int]) -> str:
    """BaseFileModel.write_ele_ids2: element numbers with ``-`` for runs."""
    out, last, last_app, wrote = [], 0, 0, False
    for p in positions:
        if last == 0:
            out.append(ipad(p))
            wrote, last_app = True, p
        elif p > last + 1:
            if last_app != last:
                out.append(ipad(-last))
            out.append(ipad(p))
            last_app, wrote = p, True
        else:
            wrote = False
        last = p
    if not wrote and positions:
        out.append(ipad(-last))
    return ipad(len(out)) + "".join(out)


# ------------------------------------------------------------------ the project
def _table(tables, name: str) -> pd.DataFrame:
    t = tables[name]
    return t if isinstance(t, pd.DataFrame) else pd.DataFrame(t)


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict(orient="records"):
        out.append({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()})
    return out


def from_tables(watershed, climate: SwatClimate, *, start=None, end=None, codes: dict | None = None,
                name: str = "landschaft") -> SwatProject:
    """A SWAT+ project from QSWAT+'s tables and the weather, as the SWAT+ Editor makes it.

    Parameters
    ----------
    watershed : SwatWatershed or dict
        The subbasins, channels, landscape units, HRUs, aquifers, routing and soils:
        a :class:`SwatWatershed` (:func:`watershed`) or QSWAT+'s project tables as
        DataFrames (``gis_subbasins``, ``gis_channels``, ``gis_lsus``, ``gis_hrus``,
        ``gis_aquifers``, ``gis_deep_aquifers``, ``gis_routing``, ``soils_sol``,
        ``soils_sol_layer``; :func:`load_qswat_project` reads them from a QSWAT+
        project database).
    climate : SwatClimate
        The weather (:func:`climate`).
    start, end
        The simulated period (default: the days all weather files cover, as the
        editor sets it).
    codes : dict, optional
        Columns of ``codes.bsn`` to change (e.g. ``{"pet": 2}`` for Hargreaves,
        ``{"rte_cha": 1}`` for Muskingum).
    name
        The project name (``object.cnt``).

    Returns
    -------
    SwatProject



    Examples
    --------
    >>> import landschaft as ls
    >>> ex = ls.hydro.swat.example_landscape()
    >>> w = ls.hydro.swat.watershed(ex["dem"], ex["landuse"], ex["soils"], landuse_lookup=ex["landuse_lookup"],
    ...                             soil_lookup=ex["soil_lookup"], soil_table=ex["soil_table"], stream_threshold=1.0)
    >>> wx = ls.hydro.swat.climate(ex["stations"], pcp=ex["pcp"], tmax=ex["tmax"], tmin=ex["tmin"], monthly=ex["monthly"])
    >>> project = ls.hydro.swat.from_tables(w, wx)
    >>> project
    SwatProject(62 files)
    >>> project.text("time.sim").splitlines()[2].split()
    ['0', '2001', '0', '2004', '0']
    >>> print(project.text("hru-data.hru").splitlines()[2])
           1  hru01                    topohru01             hyd01       UPLAND_LOAM          frsd_lum        soilplant1              null           snow001              null  
    """
    tables = getattr(watershed, "tables", watershed)
    static, data = _editor_data()
    files: dict[str, str] = {}
    meta = lambda f: f"{f}: written by landschaft for SWAT+ 62 (SWAT+ Editor {EDITOR_VERSION} formats) on " \
                     f"{dt.datetime.now():%Y-%m-%d %H:%M}\n"  # noqa: E731
    for fname, text in static.items():
        files[fname] = meta(fname) + text
    if codes:
        files["codes.bsn"] = _set_codes(files["codes.bsn"], codes)

    lsus = _records(_table(tables, "gis_lsus").sort_values("id"))
    chans = _records(_table(tables, "gis_channels").sort_values("id"))
    hrus = _records(_table(tables, "gis_hrus"))
    aqus = _records(_table(tables, "gis_aquifers").sort_values("id"))
    deeps = _records(_table(tables, "gis_deep_aquifers").sort_values("id"))
    subs = _table(tables, "gis_subbasins")
    routing = _table(tables, "gis_routing")
    soils = _table(tables, "soils_sol")
    layers = _table(tables, "soils_sol_layer")

    # insert_routing_units
    cnt = max(r["id"] for r in lsus)
    gis_to_rtu = {}
    topo_sub, fields, rtus = [], [], []
    for i, r in enumerate(lsus, 1):
        gis_to_rtu[r["id"]] = i
        sl = _slope_len(r["slope"] / 100)
        topo_sub.append({"name": _name("toportu", r["id"], cnt), "slp": r["slope"] / 100, "slp_len": sl,
                         "lat_len": sl, "dist_cha": 121.0, "depos": 0.0})
        fields.append({"name": _name("fld", r["id"], cnt), "len": 500.0, "wd": 100.0, "ang": 30.0})
        rtus.append({"name": _name("rtu", r["id"], cnt), "topo": topo_sub[-1]["name"], "field": fields[-1]["name"],
                     "gis_id": r["id"], "elev": r["elev"], "lat": r["lat"], "lon": r["lon"], "area": r["area"]})
    # insert_channels_lte
    cnt = max(r["id"] for r in chans)
    gis_to_cha = {}
    chas = []
    for i, r in enumerate(chans, 1):
        gis_to_cha[r["id"]] = i
        cname = _name("cha", r["id"], cnt)
        chas.append({"name": cname, "hyd": "hyd" + cname, "order": str(r["strahler"]), "wd": r["wid2"],
                     "dp": r["dep2"], "slp": r["slo2"] / 100, "len": r["len2"] / 1000, "gis_id": r["id"],
                     "lat": r["midlat"], "lon": r["midlon"], "area": r["areac"]})
    # insert_hrus (and insert_landuse)
    lum_rows, pcoms, mgts = _landuse(hrus, data)
    soil_by_name = {r["name"]: r for r in _records(soils)}
    bsn_area = float(subs["area"].sum())
    cnt = max(r["id"] for r in hrus)
    gis_to_hru = {}
    hyds, topo_hru, hru_data, hru_cons, ru_ele, ls_ele = [], [], [], [], [], []
    for i, r in enumerate(hrus, 1):
        gis_to_hru[r["id"]] = i
        hname = _name("hru", r["id"], cnt)
        soil = soil_by_name.get(r["soil"])
        if soil is None:
            raise ValueError(f'soil "{r["soil"]}" of the HRUs is not in soils_sol')
        h = _hydrology(soil["hyd_grp"], r["slope"] / 100)
        hyds.append({"name": _name("hyd", r["id"], cnt), "lat_ttime": 0.0, "lat_sed": 0.0, "can_max": 1.0,
                     "esco": 0.95, "epco": 0.5, "orgn_enrich": 0.0, "orgp_enrich": 0.0, "cn3_swf": h["cn3_swf"],
                     "bio_mix": 0.2, "perco": h["perco"], "lat_orgn": 0.0, "lat_orgp": 0.0, "pet_co": 1.0,
                     "latq_co": h["latq_co"]})
        sl = _slope_len(r["slope"])
        topo_hru.append({"name": _name("topohru", r["id"], cnt), "slp": r["slope"] / 100, "slp_len": sl,
                         "lat_len": sl, "dist_cha": 121.0, "depos": 0.0})
        lu = None if r["landuse"] is None else str(r["landuse"]).lower()
        if lu in ("wehb", "wetf", "wetl", "wetn", "wewo", "watr", "playa", "wetw", "wetm"):
            raise ValueError(f"land use {r['landuse']!r} is a wetland in SWAT+, which the port does not simulate; "
                             "give the water cells another land use")
        hru_data.append({"name": hname, "topo": topo_hru[-1]["name"], "hydro": hyds[-1]["name"],
                         "soil": soil["name"], "lu_mgt": None if lu is None else lum_rows[lu]["name"],
                         "soil_plant_init": "soilplant1", "surf_stor": None, "snow": "snow001", "field": None})
        hru_cons.append({"name": hname, "gis_id": r["id"], "elev": r["elev"], "lat": r["lat"], "lon": r["lon"],
                         "area": r["arslp"]})
        ru_ele.append({"name": hname, "rtu": gis_to_rtu[r["lsu"]], "obj_id": i, "frac": r["arslp"] / r["arlsu"]})
        ls_ele.append({"name": hname, "obj_typ_no": i, "bsn_frac": r["arslp"] / bsn_area,
                       "sub_frac": r["arslp"] / r["arlsu"], "rtu": gis_to_rtu[r["lsu"]]})
    # insert_aquifers
    cnt = max(r["id"] for r in aqus)
    gis_to_aqu, gis_to_deep = {}, {}
    aquifers = []
    for i, r in enumerate(aqus, 1):
        gis_to_aqu[r["id"]] = i
        aquifers.append({"name": _name("aqu", r["id"], cnt), "deep": False, "gis_id": r["id"], "lat": r["lat"],
                         "lon": r["lon"], "area": r["area"], "elev": r["elev"]})
    for r in deeps:
        i = len(aquifers) + 1
        gis_to_deep[r["id"]] = i
        aquifers.append({"name": _name("aqu_deep", r["id"], cnt), "deep": True, "gis_id": r["id"], "lat": r["lat"],
                         "lon": r["lon"], "area": r["area"], "elev": r["elev"]})
    # insert_connections
    ids = {"CH": gis_to_cha, "AQU": gis_to_aqu, "DAQ": gis_to_deep, "LSU": gis_to_rtu}
    typ = {"CH": "sdc", "AQU": "aqu", "DAQ": "aqu", "LSU": "ru"}
    rtu_out = _connections(routing, ["LSU"], gis_to_rtu, ids, typ)
    cha_out = _connections(routing, ["CH"], gis_to_cha, ids, typ)
    aqu_out = _connections(routing, ["AQU"], gis_to_aqu, ids, typ)

    # the weather: files, stations at their coordinates, nearest files, generator and objects
    wfiles, records = climate.files()
    files.update(wfiles)
    coords = sorted({(la, lo) for _, _, la, lo in records})
    stations = []
    for la, lo in coords:
        st = {"name": _sta_name(la, lo), "lat": la, "lon": lo}
        for kind in ("hmd", "pcp", "slr", "tmp", "wnd", "pet"):
            cands = [(f, a, b) for f, k, a, b in records if k == kind]
            if cands:
                st[kind] = cands[_nearest(la, lo, [(a, b) for _, a, b in cands])][0]
            else:
                st[kind] = None
        stations.append(st)
    wgns = wgn_records(climate)
    for st in stations:
        st["wgn"] = wgns[_nearest(st["lat"], st["lon"], [(w["lat"], w["lon"]) for w in wgns])]["name"]
    sta_ll = [(s["lat"], s["lon"]) for s in stations]
    wst = lambda la, lo: stations[_nearest(la, lo, sta_ll)]["name"]  # noqa: E731

    # time.sim
    if start is None or end is None:
        s0, e0 = _weather_period(climate)
        start = s0 if start is None else start
        end = e0 if end is None else end
    t = _time_sim(start, end)

    # ---------------------------------------------------------------- the files
    files["time.sim"] = meta("time.sim") + "".join(ipad(h) for h in ("day_start", "yrc_start", "day_end", "yrc_end",
                                                                     "step")) + "\n" \
        + "".join(ipad(v) for v in t) + ipad(0) + "\n"
    files["topography.hyd"] = meta("topography.hyd") + spad("name", right=False) + "".join(
        npad(h) for h in ("slp", "slp_len", "lat_len", "dist_cha", "depos")) + "\n" + "".join(
        spad(r["name"], right=False) + "".join(npad(r[k]) for k in ("slp", "slp_len", "lat_len", "dist_cha", "depos"))
        + "\n" for r in topo_hru + topo_sub)
    hcols = ("lat_ttime", "lat_sed", "can_max", "esco", "epco", "orgn_enrich", "orgp_enrich", "cn3_swf", "bio_mix",
             "perco", "lat_orgn", "lat_orgp", "pet_co", "latq_co")
    files["hydrology.hyd"] = meta("hydrology.hyd") + spad("name", right=False) + "".join(npad(h) for h in hcols) \
        + "\n" + "".join(spad(r["name"], right=False) + "".join(npad(r[k]) for k in hcols) + "\n" for r in hyds)
    files["field.fld"] = meta("field.fld") + spad("name", right=False) + "".join(
        npad(h) for h in ("len", "wd", "ang")) + "\n" + "".join(
        spad(r["name"], right=False) + "".join(npad(r[k]) for k in ("len", "wd", "ang")) + "\n" for r in fields)
    files["hru-data.hru"] = meta("hru-data.hru") + ipad("id") + spad("name", right=False) + "".join(
        spad(h) for h in ("topo", "hydro", "soil", "lu_mgt", "soil_plant_init", "surf_stor", "snow", "field")) + "\n" \
        + "".join(ipad(i) + spad(r["name"], right=False) + "".join(
            spad(r[k]) for k in ("topo", "hydro", "soil", "lu_mgt", "soil_plant_init", "surf_stor", "snow", "field"))
            + "\n" for i, r in enumerate(hru_data, 1))
    files["hru.con"] = _con(meta("hru.con"), "hru", hru_cons, [[] for _ in hru_cons], wst)
    files["rout_unit.con"] = _con(meta("rout_unit.con"), "rtu", rtus, [rtu_out.get(i, []) for i in
                                                                       range(1, len(rtus) + 1)], wst)
    files["chandeg.con"] = _con(meta("chandeg.con"), "lcha", [dict(c, elev=None) for c in chas],
                                [cha_out.get(i, []) for i in range(1, len(chas) + 1)], wst)
    files["aquifer.con"] = _con(meta("aquifer.con"), "aqu", aquifers,
                                [aqu_out.get(i, []) for i in range(1, len(aquifers) + 1)], wst)
    files["rout_unit.rtu"] = meta("rout_unit.rtu") + ipad("id") + "".join(
        spad(h) for h in ("name", "define", "dlr", "topo", "field")) + "\n" + "".join(
        ipad(i) + spad(r["name"]) + spad(r["name"]) + spad(None) + spad(r["topo"]) + spad(r["field"]) + "\n"
        for i, r in enumerate(rtus, 1))
    files["rout_unit.ele"] = meta("rout_unit.ele") + ipad("id") + spad("name", right=False) + spad("obj_typ", 12) \
        + ipad("obj_id") + npad("frac") + spad("dlr") + "\n" + "".join(
        ipad(i) + spad(r["name"], right=False) + spad("hru", 12) + ipad(r["obj_id"]) + epad(r["frac"])
        + spad(None, null="0") + "\n" for i, r in enumerate(ru_ele, 1))
    files["rout_unit.def"] = meta("rout_unit.def") + ipad("id") + spad("name") + ipad("elem_tot") \
        + ipad("elements") + "\n" + "".join(
        ipad(i) + spad(r["name"]) + _ele_ids([e["obj_id"] for e in ru_ele if e["rtu"] == i]) + "\n"
        for i, r in enumerate(rtus, 1))
    ls_defs = [(i, r) for i, r in enumerate(rtus, 1) if any(e["rtu"] == i for e in ru_ele)]
    files["ls_unit.def"] = meta("ls_unit.def") + f"{len(ls_defs)}\n" + ipad("id") + spad("name") + npad("area") \
        + ipad("elem_tot") + ipad("elements") + "\n" + "".join(
        ipad(k) + spad(r["name"]) + npad(r["area"]) + _ele_ids([e["obj_typ_no"] for e in ls_ele if e["rtu"] == i])
        + "\n" for k, (i, r) in enumerate(ls_defs, 1))
    files["ls_unit.ele"] = meta("ls_unit.ele") + ipad("id") + spad("name", right=False) + spad("obj_typ", 12) \
        + ipad("obj_typ_no") + npad("bsn_frac") + npad("sub_frac") + npad("reg_frac") + "\n" + "".join(
        ipad(i) + spad(r["name"], right=False) + spad("hru", 12) + ipad(r["obj_typ_no"], 10)
        + epad(r["bsn_frac"], nzmin=True) + npad(r["sub_frac"]) + npad(0) + "\n" for i, r in enumerate(ls_ele, 1))
    tot_area = sum(r["area"] for r in rtus)
    files["aqu_catunit.ele"] = meta("aqu_catunit.ele") + ipad("id") + spad("name", right=False) \
        + spad("obj_typ", 12) + ipad("obj_typ_no", 10) + npad("bsn_frac") + npad("sub_frac") + npad("reg_frac") \
        + "\n" + "".join(ipad(i) + spad(r["name"], right=False) + spad("aqu", 12) + ipad(i, 10)
                          + npad(r["area"] / tot_area, nzmin=True) + npad(0) + npad(0) + "\n"
                          for i, r in enumerate(aquifers, 1))
    acols = ("gw_flo", "dep_bot", "dep_wt", "no3_n", "sol_p", "carbon", "flo_dist", "bf_max", "alpha_bf", "revap",
             "rchg_dp", "spec_yld", "hl_no3n", "flo_min", "revap_min")
    files["aquifer.aqu"] = meta("aquifer.aqu") + ipad("id") + spad("name", right=False) + spad("init") + "".join(
        npad(h) for h in acols) + "\n" + "".join(
        ipad(i) + spad(r["name"], right=False) + spad("initaqu1") + "".join(npad(v) for v in _aquifer(r["deep"]))
        + "\n" for i, r in enumerate(aquifers, 1))
    files["channel-lte.cha"] = meta("channel-lte.cha") + ipad("id") + spad("name", right=False) + "".join(
        spad(h) for h in ("cha_ini", "cha_hyd", "cha_sed", "cha_nut")) + "\n" + "".join(
        ipad(i) + spad(r["name"], right=False) + spad("initcha1") + spad(r["hyd"]) + spad(None) + spad("nutcha1")
        + "\n" for i, r in enumerate(chas, 1))
    files["hyd-sed-lte.cha"] = _hyd_sed(meta("hyd-sed-lte.cha"), chas)
    files["landuse.lum"] = _landuse_text(meta("landuse.lum"), list(lum_rows.values()))
    files["plant.ini"] = _plant_ini(meta("plant.ini"), pcoms)
    if mgts:
        files["management.sch"] = _management(meta("management.sch"), mgts)
    files["soils.sol"] = _soils(meta("soils.sol"), soils, layers)
    files["weather-sta.cli"] = meta("weather-sta.cli") + spad("name", right=False) + spad("wgn") + "".join(
        spad(h, 25) for h in ("pcp", "tmp", "slr", "hmd", "wnd", "pet")) + spad("atmo_dep") + "\n" + "".join(
        spad(s["name"], right=False) + spad(s["wgn"]) + "".join(spad(s[k], 25, null="sim") for k in
                                                                 ("pcp", "tmp", "slr", "hmd", "wnd"))
        + spad(s["pet"], 25, null="null") + spad(None) + "\n" for s in sorted(stations, key=lambda s: s["name"]))
    files["weather-wgn.cli"] = _wgn_text(meta("weather-wgn.cli"), wgns)
    n_obj = len(hru_cons) + len(rtus) + len(aquifers) + len(chas)
    counts = {"obj": n_obj, "hru": len(hru_cons), "lhru": 0, "rtu": len(rtus), "gwfl": 0, "aqu": len(aquifers),
              "cha": 0, "res": 0, "rec": 0, "exco": 0, "dlr": 0, "can": 0, "pmp": 0, "out": 0, "lcha": len(chas),
              "aqu2d": 0, "hrd": 0, "wro": 0}
    ls_area = sum(r["area"] for _, r in ls_defs)
    files["object.cnt"] = meta("object.cnt") + spad("name", right=False) + spad("ls_area", 12) \
        + spad("tot_area", 12) + "".join(ipad(k) for k in counts) + "\n" + spad(re.sub(r"\W+", "", name), right=False) \
        + npad(ls_area) + npad(tot_area) + "".join(ipad(v) for v in counts.values()) + "\n"
    files["file.cio"] = _file_cio(meta("file.cio"), set(files))
    return SwatProject(files)


def _weather_period(c: SwatClimate):
    """import_weather.add_weather_files: the latest first day and the earliest last day."""
    wf, rec = c.files()
    starts, ends = [], []
    for fname, _, _, _ in rec:
        rows = [ln.split() for ln in wf[fname].split("\n")[3:] if ln.strip()]
        y, d = int(rows[0][0]), int(rows[0][1])
        starts.append(dt.datetime(y, 1, 1) + dt.timedelta(days=d - 1))
        y, d = int(rows[-1][0]), int(rows[-1][1])
        ends.append(dt.datetime(y, 1, 1) + dt.timedelta(days=d - 1))
    return max(starts), min(ends)


def _set_codes(text: str, codes: dict) -> str:
    lines = text.split("\n")
    head, vals = lines[1].split(), lines[2].split()
    for k, v in codes.items():
        if k not in head:
            raise ValueError(f"codes.bsn has no column {k!r}")
        vals[head.index(k)] = str(v)
    lines[2] = "".join(spad(v, 16 if i < 2 or head[i] == "atmo_dep" else 8) for i, v in enumerate(vals))
    return "\n".join(lines)


def _aquifer(deep: bool) -> list[float]:
    """database/project/aquifer.Aquifer_aqu.get_default_shallow / get_default_deep."""
    if deep:
        return [0.0, 100.0, 20.0, 0.0, 0.0, 0.5, 50.0, 1.0, 0.01, 0.0, 0.0, 0.03, 30.0, 0.0, 0.0]
    return [0.05, 10.0, 3.0, 0.0, 0.0, 0.5, 50.0, 1.0, 0.05, 0.02, 0.05, 0.05, 30.0, 3.0, 5.0]


def _connections(routing: pd.DataFrame, cats: list[str], src_ids: dict, ids: dict, typ: dict) -> dict:
    """import_gis.get_connections (no point sources, reservoirs or lte)."""
    rows = routing[routing["sourcecat"].isin(cats) & (routing["percent"] > 0) & (routing["sinkcat"] != "X")]
    rows = rows.iloc[np.argsort(rows["sourceid"].to_numpy(), kind="stable")]
    pt_src = {int(r.sourceid): r for r in routing[routing["sourcecat"] == "PT"].itertuples()}
    ch_to_pt = {int(r.sinkid): r for r in routing[(routing["sourcecat"] == "CH") & (routing["sinkcat"] == "PT")]
                .itertuples()}
    out: dict = {}
    for r in rows.itertuples():
        if r.sinkcat == "PT" and "AQU" in cats:
            src = ch_to_pt.get(int(r.sinkid))
            if src is None:
                continue
            out.setdefault(src_ids[int(r.sourceid)], []).append(
                (typ[src.sourcecat], ids[src.sourcecat][int(src.sourceid)], src.hyd_typ, src.percent / 100))
            continue
        con = r
        while con is not None and con.sinkcat == "PT":
            con = pt_src.get(int(con.sinkid))
        if con is not None and con.sinkcat in typ:
            out.setdefault(src_ids[int(r.sourceid)], []).append(
                (typ[con.sinkcat], ids[con.sinkcat][int(con.sinkid)], con.hyd_typ, con.percent / 100))
    return out


def _con(meta: str, elem: str, cons: list, outs: list, wst) -> str:
    """fileio/connect.write_con_table."""
    has_out = any(outs)
    head = ipad("id") + spad("name", right=False) + ipad("gis_id") + npad("area") + npad("lat") + npad("lon") \
        + npad("elev") + ipad(elem) + spad("wst") + ipad("cst") + ipad("ovfl") + ipad("rule") + ipad("out_tot")
    if has_out:
        head += spad("obj_typ", 12) + ipad("obj_id") + spad("hyd_typ", 12) + npad("frac")
    lines = [meta + head + "\n"]
    for i, (c, o) in enumerate(zip(cons, outs), 1):
        s = ipad(i) + spad(c["name"], right=False) + ipad(c["gis_id"]) + npad(c["area"], nzmin=True) \
            + npad(c["lat"]) + npad(c["lon"]) + npad(c.get("elev")) + ipad(i) + spad(wst(c["lat"], c["lon"])) \
            + ipad(0) + ipad(0) + ipad(0) + ipad(len(o))
        for t, oid, hyd, frac in o:
            s += spad(t, 12) + ipad(oid) + spad(hyd, 12) + npad(frac)
        lines.append(s + "\n")
    return "".join(lines)


def _hyd_sed(meta: str, chas: list) -> str:
    """fileio/channel.Hyd_sed_lte_cha (import_gis.insert_channels_lte values)."""
    cols = ["order", "wd", "dp", "slp", "len", "mann", "k", "erod_fact", "cov_fact", "sinu", "eq_slp", "d50", "clay",
            "carbon", "dry_bd", "side_slp", "bankfull_flo", "fps", "fpn", "n_conc", "p_conc", "p_bio"]
    fixed = {"mann": 0.05, "k": 1.0, "erod_fact": 0.01, "cov_fact": 0.005, "sinu": 1.05, "eq_slp": 0.001, "d50": 12.0,
             "clay": 50.0, "carbon": 0.04, "dry_bd": 1.0, "side_slp": 0.5, "bankfull_flo": 0.5, "fps": 0.00001,
             "fpn": 0.1, "n_conc": 0.0, "p_conc": 0.0, "p_bio": 0.0}
    nz = {"wd": 0.001, "dp": 0.001, "slp": 0.00001, "len": 0.00001}
    head = spad("name", right=False) + spad("order") + "".join(npad(c) for c in cols[1:]) + "description"
    lines = [meta + head + "\n"]
    for c in chas:
        s = spad(c["hyd"], right=False) + spad(c["order"])
        for k in cols[1:]:
            v = c[k] if k in nz else fixed[k]
            s += npad(v, nzmin=k in nz, nzmin_val=nz.get(k, 0.00001))
        lines.append(s + "\n")
    return "".join(lines)


def _landuse(hrus: list, data: dict):
    """import_gis.insert_landuse: a land use and plant community per land use of the HRUs."""
    plants, urban = data["plants"], data["urban"]
    lus = list(dict.fromkeys(str(r["landuse"]).lower() for r in hrus if r["landuse"] not in (None, "NULL")))
    lums, pcoms, mgts = {}, [], []
    defaults = {"cn2": data["cntable"]["5"], "cons_prac": data["cons_prac"]["1"], "ov_mann": data["ovn_table"]["2"]}
    for lu in lus:
        plant = plants.get(lu)
        lum_name, comm_name = f"{lu}_lum", f"{lu}_comm"
        if plant is None:
            u = urban.get(lu)
            if u is None:
                raise ValueError(f"{lu} does not exist in plants_plt or urban_urb, but is used as land use in your "
                                 "GIS HRUs.")
            lums[lu] = {"name": lum_name, "cal_group": None, "plnt_com": None, "mgt": None,
                        "cn2": data["cntable"]["49"], "cons_prac": defaults["cons_prac"], "urban": u,
                        "urb_ro": "buildup_washoff", "ov_mann": data["ovn_table"]["18"]}
            continue
        ds_pi = data["plant_ini"].get(comm_name)
        if ds_pi is not None:
            pcoms.append({"name": ds_pi["name"], "rot_yr_ini": ds_pi["rot_yr_ini"],
                          "plants": [dict(p, plnt_name=plant["name"]) for p in ds_pi["plants"]]})
            mgt = None
            if plant["typ"].startswith("warm_annual") or plant["typ"].startswith("cold_annual"):
                table = "pl_hv_summer1" if plant["typ"].startswith("warm_annual") else "pl_hv_winter1"
                mgt = f"{plant['name']}_rot"
                if not any(m["name"] == mgt for m in mgts):
                    mgts.append({"name": mgt, "d_table": table, "plant1": plant["name"]})
            ds_m = data["landuse_lum"].get(lum_name)
            if ds_m is not None:
                lums[lu] = {"name": ds_m["name"], "cal_group": ds_m["cal_group"], "plnt_com": ds_pi["name"],
                            "mgt": mgt, "cn2": ds_m["cn2"], "cons_prac": ds_m["cons_prac"], "urban": None,
                            "urb_ro": None, "ov_mann": ds_m["ov_mann"]}
            else:
                lums[lu] = dict(name=lum_name, cal_group=None, plnt_com=ds_pi["name"], mgt=mgt, urban=None,
                                urb_ro=None, **defaults)
        else:
            pcoms.append({"name": comm_name, "rot_yr_ini": 1,
                          "plants": [{"plnt_name": plant["name"], "lc_status": 0, "lai_init": 0.0, "bm_init": 0.0,
                                      "phu_init": 0.0, "plnt_pop": 0.0, "yrs_init": 0.0, "rsd_init": 10000.0}]})
            lums[lu] = dict(name=lum_name, cal_group=None, plnt_com=comm_name, mgt=None, urban=None, urb_ro=None,
                            **defaults)
    return lums, pcoms, mgts


def _landuse_text(meta: str, rows: list) -> str:
    cols = ("cal_group", "plnt_com", "mgt", "cn2", "cons_prac", "urban", "urb_ro", "ov_mann", "tile", "sep", "vfs",
            "grww", "bmp")
    head = spad("name", 20, right=False) + "".join(spad(c, 41 if c == "mgt" else 16) for c in cols)
    out = [meta + head + "\n"]
    for r in rows:
        out.append(spad(r["name"], 20, right=False) + "".join(
            spad(r.get(c), 41 if c == "mgt" else 16) for c in cols) + "\n")
    return "".join(out)


def _plant_ini(meta: str, pcoms: list) -> str:
    head = spad("pcom_name", right=False) + ipad("plt_cnt") + ipad("rot_yr_ini") + spad("plt_name") \
        + spad("lc_status", 12) + "".join(npad(c) for c in ("lai_init", "bm_init", "phu_init", "plnt_pop",
                                                              "yrs_init", "rsd_init"))
    out = [meta + head + "\n"]
    for p in pcoms:
        out.append(spad(p["name"], right=False) + ipad(len(p["plants"])) + ipad(p["rot_yr_ini"]) + "\n")
        for it in p["plants"]:
            out.append(spad("", null="") + spad("", 8, null="") + spad(it["plnt_name"]) + yn(it["lc_status"])
                       + "".join(npad(it[c]) for c in ("lai_init", "bm_init", "phu_init", "plnt_pop", "yrs_init",
                                                       "rsd_init")) + "\n")
    return "".join(out)


def _management(meta: str, mgts: list) -> str:
    head = spad("name", 41, right=False) + spad("numb_ops", 8) + spad("numb_auto", 9) + spad("op_typ") \
        + ipad("mon") + ipad("day") + npad("hu_sch") + spad("op_data1") + spad("op_data2") + npad("op_data3")
    out = [meta + head + "\n"]
    for m in mgts:
        out.append(spad(m["name"], 25, right=False) + ipad(0) + ipad(1, 9) + "\n")
        out.append(spad(" ", 46) + spad(m["d_table"]) + spad(m["plant1"], 5) + "\n")
    return "".join(out)


def _soils(meta: str, soils: pd.DataFrame, layers: pd.DataFrame) -> str:
    lcols = ("dp", "bd", "awc", "soil_k", "carbon", "clay", "silt", "sand", "rock", "alb", "usle_k", "ec", "caco3",
             "ph")
    head = spad("name", 25, right=False) + spad("nly", 8) + spad("hyd_grp") + npad("dp_tot") + npad("anion_excl") \
        + npad("perc_crk") + spad("texture", 25, right=False) + "".join(npad(c) for c in lcols)
    out = [meta + head + "\n"]
    lay = _records(layers)
    for s in _records(soils.sort_values("id")):
        ls = sorted((x for x in lay if x["soil_id"] == s["id"]), key=lambda x: x["layer_num"])
        out.append(spad(s["name"], 25, right=False) + ipad(len(ls)) + spad(s["hyd_grp"]) + npad(s["dp_tot"])
                   + npad(s["anion_excl"]) + npad(s["perc_crk"]) + spad(s.get("texture"), 25, right=False) + "\n")
        for x in ls:
            out.append(spad(" ", 122) + "".join(npad(x[c], null="0.0") for c in lcols) + "\n")
    return "".join(out)


def _wgn_text(meta: str, wgns: list) -> str:
    from ._weather import WGN_COLUMNS

    out = [meta]
    for w in sorted(wgns, key=lambda w: w["name"]):
        out.append(spad(w["name"], 25, right=False) + npad(w["lat"]) + npad(w["lon"]) + npad(w["elev"])
                   + ipad(w["rain_yrs"]) + "\n")
        out.append("".join(npad(c) for c in WGN_COLUMNS) + "\n")
        for m in sorted(w["months"], key=lambda m: m["month"]):
            out.append("".join(npad(m[c]) for c in WGN_COLUMNS) + "\n")
    return "".join(out)


_CIO = [
    ("simulation", ["time.sim", "print.prt", None, "object.cnt", None]),
    ("basin", ["codes.bsn", "parameters.bsn", None]),
    ("climate", ["weather-sta.cli", "weather-wgn.cli", "pet.cli", "pcp.cli", "tmp.cli", "slr.cli", "hmd.cli",
                 "wnd.cli", None]),
    ("connect", ["hru.con", None, "rout_unit.con", None, "aquifer.con", None, None, None, None, None, None, None,
                 "chandeg.con"]),
    ("channel", ["initial.cha", None, None, None, "nutrients.cha", "channel-lte.cha", "hyd-sed-lte.cha", None]),
    ("reservoir", [None] * 8),
    ("routing_unit", ["rout_unit.def", "rout_unit.ele", "rout_unit.rtu", None]),
    ("hru", ["hru-data.hru", None]),
    ("exco", [None] * 6),
    ("recall", [None]),
    ("dr", [None] * 6),
    ("aquifer", ["initial.aqu", "aquifer.aqu"]),
    ("herd", [None] * 3),
    ("water_rights", [None] * 3),
    ("link", [None] * 2),
    ("hydrology", ["hydrology.hyd", "topography.hyd", "field.fld"]),
    ("structural", ["tiledrain.str", "septic.str", "filterstrip.str", "grassedww.str", "bmpuser.str"]),
    ("hru_parm_db", ["plants.plt", "fertilizer.frt", "tillage.til", "pesticide.pes", None, None, None, "urban.urb",
                     "septic.sep", "snow.sno"]),
    ("ops", ["harv.ops", "graze.ops", "irr.ops", "chem_app.ops", "fire.ops", "sweep.ops"]),
    ("lum", ["landuse.lum", "management.sch", "cntable.lum", "cons_practice.lum", "ovn_table.lum"]),
    ("chg", ["cal_parms.cal"] + [None] * 8),
    ("init", ["plant.ini", "soil_plant.ini", "om_water.ini"] + [None] * 8),
    ("soils", ["soils.sol", "nutrients.sol", None]),
    ("decision_table", ["lum.dtl", "res_rel.dtl", None, None]),
    ("regions", ["ls_unit.ele", "ls_unit.def", None, None, None, None, None, None, "aqu_catunit.ele"] + [None] * 8),
    ("pcp_path", [None]), ("tmp_path", [None]), ("slr_path", [None]), ("hmd_path", [None]), ("wnd_path", [None]),
    ("out_path", [None]),
]


def _file_cio(meta: str, present: set) -> str:
    """fileio/config.File_cio: the files of each class, null where the project has none."""
    out = [meta]
    for cls, names in _CIO:
        out.append(spad(cls, right=False) + "".join(
            spad(n if n is not None and n in present else None, right=False) for n in names) + "\n")
    return "".join(out)
