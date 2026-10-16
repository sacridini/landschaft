# SPDX-License-Identifier: GPL-3.0-or-later
"""Build ``src/landschaft/hydro/swat/data/swatplus_editor.zip``: what ls.hydro.swat.from_tables
takes from the SWAT+ Editor 4.0.2 and its datasets database (public domain).

* ``static/<file>``: the files the editor writes the same for every project (the
  parameter databases ``plants.plt``, ``fertilizer.frt``, ``tillage.til``, ...,
  ``lum.dtl``, ``codes.bsn``, ``parameters.bsn``, ``print.prt``, the default
  initialisations), as the editor writes them (first line, its date stamp, dropped);
* ``datasets.json``: the datasets tables the GIS import reads (plants and their types,
  urban land uses, plant communities, land use defaults, the curve number, practice
  and Manning tables by id).

Usage::

    LANDSCHAFT_SWAT_EDITOR=<swatplus-editor> python tools/swat/make_editor_data.py
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import zipfile
from pathlib import Path


import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from editor_txtinout import EDITOR, run_editor  # noqa: E402

STATIC = ["bmpuser.str", "cal_parms.cal", "chem_app.ops", "cntable.lum", "codes.bsn", "cons_practice.lum",
          "fertilizer.frt", "filterstrip.str", "fire.ops", "grassedww.str", "graze.ops", "harv.ops", "initial.aqu",
          "initial.cha", "irr.ops", "lum.dtl", "nutrients.cha", "nutrients.sol", "om_water.ini", "ovn_table.lum",
          "parameters.bsn", "pesticide.pes", "plants.plt", "print.prt", "res_rel.dtl", "septic.sep", "septic.str",
          "snow.sno", "soil_plant.ini", "sweep.ops", "tiledrain.str", "tillage.til", "urban.urb"]


def _tiny_project():
    """A one-subbasin, one-HRU QSWAT+ project (the static files do not depend on it)."""
    t = {
        "gis_subbasins": [{"id": 1, "area": 100.0, "slo1": 2.0, "len1": 500.0, "sll": 90, "lat": 40.0, "lon": -90.0,
                           "elev": 200.0, "elevmin": 190.0, "elevmax": 210.0, "waterid": 0}],
        "gis_channels": [{"id": 1, "subbasin": 1, "areac": 100.0, "strahler": 1, "len2": 1000.0, "slo2": 1.0,
                          "wid2": 1.0, "dep2": 0.1, "elevmin": 190.0, "elevmax": 200.0, "midlat": 40.0,
                          "midlon": -90.0}],
        "gis_lsus": [{"id": 10, "category": 0, "channel": 1, "area": 100.0, "slope": 2.0, "len1": 500.0, "csl": 1.0,
                      "wid1": 1.0, "dep1": 0.1, "lat": 40.0, "lon": -90.0, "elev": 200.0}],
        "gis_hrus": [{"id": 1, "lsu": 10, "arsub": 100.0, "arlsu": 100.0, "landuse": "FRSD", "arland": 100.0,
                      "soil": "S1", "arso": 100.0, "slp": "0-9999", "arslp": 100.0, "slope": 2.0, "lat": 40.0,
                      "lon": -90.0, "elev": 200.0}],
        "gis_aquifers": [{"id": 10, "category": 0, "subbasin": 1, "deep_aquifer": 1, "area": 100.0, "lat": 40.0,
                          "lon": -90.0, "elev": 200.0}],
        "gis_deep_aquifers": [{"id": 1, "subbasin": 1, "area": 100.0, "lat": 40.0, "lon": -90.0, "elev": 200.0}],
        "gis_points": [{"id": 1, "subbasin": 1, "ptype": "O", "xpr": 0.0, "ypr": 0.0, "lat": 40.0, "lon": -90.0,
                        "elev": 190.0}],
        "gis_routing": [(1, "HRU", "tot", 1, "CH", 100.0), (1, "CH", "tot", 1, "PT", 100.0),
                        (1, "PT", "tot", 0, "X", 100.0), (1, "SUB", "tot", 1, "PT", 100.0),
                        (10, "LSU", "tot", 1, "CH", 100.0), (10, "LSU", "rhg", 10, "AQU", 100.0),
                        (10, "AQU", "tot", 1, "PT", 100.0), (10, "AQU", "rhg", 1, "DAQ", 100.0),
                        (1, "DAQ", "tot", 1, "PT", 100.0)],
        "soils_sol": [{"id": 1, "name": "S1", "hyd_grp": "B", "dp_tot": 1000.0, "anion_excl": 0.5, "perc_crk": 0.5,
                       "texture": "L", "description": None}],
        "soils_sol_layer": [{"id": 1, "soil_id": 1, "layer_num": 1, "dp": 1000.0, "bd": 1.4, "awc": 0.15,
                             "soil_k": 20.0, "carbon": 1.0, "clay": 20.0, "silt": 40.0, "sand": 40.0, "rock": 0.0,
                             "alb": 0.1, "usle_k": 0.3, "ec": 0.0, "caco3": 0.0, "ph": 6.5}],
    }
    tables = {k: (pd.DataFrame(v, columns=["sourceid", "sourcecat", "hyd_typ", "sinkid", "sinkcat", "percent"])
                  if k == "gis_routing" else pd.DataFrame(v)) for k, v in t.items()}
    days = pd.date_range("2000-01-01", "2000-12-31")
    pcp = "\n".join(["pcp.cli: x", "filename", "s.pcp"]) + "\n"
    rows = "".join(f"{d.year}{str(d.dayofyear).rjust(5)}    1.00000\n" for d in days)
    spcp = "s.pcp: x\nnbyr tstep lat lon elev\n   1         0    40.000   -90.000   200.000\n" + rows
    wgn = [{"name": "w1", "lat": 40.0, "lon": -90.0, "elev": 200.0, "rain_yrs": 10,
            "months": [{"month": m, "tmp_max_ave": 10.0, "tmp_min_ave": 0.0, "tmp_max_sd": 3.0, "tmp_min_sd": 3.0,
                        "pcp_ave": 50.0, "pcp_sd": 5.0, "pcp_skew": 2.0, "wet_dry": 0.3, "wet_wet": 0.5,
                        "pcp_days": 8.0, "pcp_hhr": 10.0, "slr_ave": 15.0, "dew_ave": 2.0, "wnd_ave": 3.0}
                       for m in range(1, 13)]}]
    return tables, {"pcp.cli": pcp.encode(), "s.pcp": spcp.encode()}, wgn


def datasets() -> dict:
    db = sqlite3.connect(EDITOR / "release" / "build" / "swatplus_datasets.sqlite")
    q = lambda sql: db.execute(sql).fetchall()  # noqa: E731
    out = {"version": q("select value from version")[0][0]}
    out["plants"] = {n.lower(): {"name": n, "typ": t}
                     for n, t in q("select name, plnt_typ from plants_plt order by id")}
    out["urban"] = {n.lower(): n for (n,) in q("select name from urban_urb order by id")}
    comms = {}
    for pid, name, rot in q("select id, name, rot_yr_ini from plant_ini order by id"):
        items = [dict(zip(("plnt_name", "lc_status", "lai_init", "bm_init", "phu_init", "plnt_pop", "yrs_init",
                           "rsd_init"), r)) for r in
                 q(f"select p.name, i.lc_status, i.lai_init, i.bm_init, i.phu_init, i.plnt_pop, i.yrs_init, i.rsd_init "
                   f"from plant_ini_item i left join plants_plt p on p.id = i.plnt_name_id "
                   f"where i.plant_ini_id = {pid} "
                   f"order by i.id")]
        comms[name.lower()] = {"name": name, "rot_yr_ini": rot, "plants": items}
    out["plant_ini"] = comms
    out["landuse_lum"] = {
        n.lower(): {"name": n, "cal_group": cg, "cn2": cn, "cons_prac": cp, "ov_mann": ov}
        for n, cg, cn, cp, ov in q("select l.name, l.cal_group, cn.name, cp.name, o.name from landuse_lum l "
                                   "left join cntable_lum cn on cn.id = l.cn2_id "
                                   "left join cons_prac_lum cp on cp.id = l.cons_prac_id "
                                   "left join ovn_table_lum o on o.id = l.ov_mann_id order by l.id")}
    out["cntable"] = {str(i): n for i, n in q("select id, name from cntable_lum")}
    out["cons_prac"] = {str(i): n for i, n in q("select id, name from cons_prac_lum")}
    out["ovn_table"] = {str(i): n for i, n in q("select id, name from ovn_table_lum")}
    out["d_tables"] = [n for (n,) in q("select name from d_table_dtl where file_name = 'lum.dtl' order by id")]
    return out


def main() -> None:
    tables, weather, wgn = _tiny_project()
    files = run_editor(tables, weather, wgn, (0, 2000, 0, 2000))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in STATIC:
            text = files[name].decode().replace("\r\n", "\n")
            z.writestr(f"static/{name}", text.split("\n", 1)[1])
        z.writestr("datasets.json", json.dumps(datasets(), indent=0, sort_keys=True))
    out = ROOT / "src" / "landschaft" / "hydro" / "swat" / "data" / "swatplus_editor.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf.getvalue())
    print(out, len(buf.getvalue()))



if __name__ == "__main__":
    main()
