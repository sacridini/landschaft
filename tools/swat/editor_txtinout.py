# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the SWAT+ Editor itself on QSWAT+ tables: the reference of ls.hydro.swat.from_tables.

The SWAT+ Editor 4.0.2 (github.com/swat-model/swatplus-editor, public domain) turns
QSWAT+'s project database (``gis_*`` tables, soils) into a SWAT+ 62 TxtInOut folder:
``setup_project`` (copies the datasets database and imports the GIS tables),
``import_weather`` (observed files in SWAT+ format, then the stations matched to the
objects) and ``write_files``. This script builds such a project database from the
tables of :class:`ls.hydro.swat.SwatWatershed` and the weather files of a project,
runs those steps headless and returns the files the editor writes.

Needs the editor's sources (``src/api``) and its datasets database
(``release/build/swatplus_datasets.sqlite``), and ``peewee`` in the Python that runs
it::

    pip install peewee
    # the editor's sources: github.com/swat-model/swatplus-editor, 4.0.2 (commit d8551ab)
    LANDSCHAFT_SWAT_EDITOR=<swatplus-editor> python tools/make_reference_swat.py landscape
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# github.com/swat-model/swatplus-editor at d8551ab (4.0.2), next to the SWAT+ sources
EDITOR = Path(os.environ.get("LANDSCHAFT_SWAT_EDITOR", Path(os.environ.get(
    "LANDSCHAFT_SWAT_DATA", Path.home() / "Documents" / "dev" / "swat_sources")) / "swatplus-editor"))

_DRIVER = r'''
import json, os, sys
api, project_db, datasets_db, payload, out_dir = sys.argv[1:6]
sys.path.insert(0, api)
os.chdir(api)
from peewee import fn
from database.project.setup import SetupProjectDatabase
from database.project import base, config, gis, soils, climate, simulation
from database.project.config import Project_config
data = json.load(open(payload))
SetupProjectDatabase.init(project_db)
SetupProjectDatabase.create_tables()
base.db.create_tables([gis.Gis_aquifers, gis.Gis_deep_aquifers])
Project_config.create(project_name="landschaft", gis_type="qgis", gis_version="4.0.0", delineation_done=True,
                      hrus_done=True)
tables = {"gis_subbasins": gis.Gis_subbasins, "gis_channels": gis.Gis_channels, "gis_lsus": gis.Gis_lsus,
          "gis_hrus": gis.Gis_hrus, "gis_aquifers": gis.Gis_aquifers, "gis_deep_aquifers": gis.Gis_deep_aquifers,
          "gis_points": gis.Gis_points}
for name, model in tables.items():
    cols = set(model._meta.fields)
    rows = [{k: v for k, v in r.items() if k in cols} for r in data[name]]
    if rows:
        model.insert_many(rows).execute()
# gis_routing has no key of its own: raw rows in QSWAT+'s order
cur = base.db.execute_sql("drop table gis_routing")
base.db.execute_sql("create table gis_routing (sourceid INTEGER, sourcecat TEXT, hyd_typ TEXT, sinkid INTEGER, "
                    "sinkcat TEXT, percent REAL)")
for r in data["gis_routing"]:
    base.db.execute_sql("insert into gis_routing values (?,?,?,?,?,?)",
                        (r["sourceid"], r["sourcecat"], r["hyd_typ"], r["sinkid"], r["sinkcat"], r["percent"]))
for r in data["soils_sol"]:
    soils.Soils_sol.insert(id=r["id"], name=r["name"], hyd_grp=r["hyd_grp"], dp_tot=r["dp_tot"],
                           anion_excl=r["anion_excl"], perc_crk=r["perc_crk"], texture=r["texture"],
                           description=r["description"]).execute()
for r in data["soils_sol_layer"]:
    soils.Soils_sol_layer.insert(id=r["id"], soil=r["soil_id"], layer_num=r["layer_num"],
                                 **{k: r[k] for k in ("dp", "bd", "awc", "soil_k", "carbon", "clay", "silt", "sand",
                                                      "rock", "alb", "usle_k", "ec", "caco3", "ph")}).execute()
SetupProjectDatabase.close()

from actions.setup_project import SetupProject
api_ = SetupProject(project_db, "4.0.2", "landschaft", datasets_db)
del api_
SetupProjectDatabase.init(project_db)
# weather generator stations (as an imported WGN table), then the observed files
for w in data["wgn"]:
    wid = climate.Weather_wgn_cli.insert(name=w["name"], lat=w["lat"], lon=w["lon"], elev=w["elev"],
                                         rain_yrs=w["rain_yrs"]).execute()
    for m in w["months"]:
        climate.Weather_wgn_cli_mon.insert(weather_wgn_cli=wid, **m).execute()
Project_config.update(weather_data_dir=data["weather_dir"], weather_data_format="plus",
                      input_files_dir=out_dir).execute()
SetupProjectDatabase.close()
from actions.import_weather import WeatherImport
w = WeatherImport(project_db, True, True)
w.import_data()
del w
SetupProjectDatabase.init(project_db)
t = data["time"]
simulation.Time_sim.update(day_start=t[0], yrc_start=t[1], day_end=t[2], yrc_end=t[3]).execute()
SetupProjectDatabase.close()
from actions.write_files import WriteFiles
WriteFiles(project_db, "62.0").write()
'''


def _plain(x):
    """A JSON value with the exact float (json writes repr, which round-trips)."""
    if x is None:
        return None
    if isinstance(x, (bool, str)):
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    if f != f:
        return None
    return int(x) if float(x).is_integer() and not isinstance(x, float) and "float" not in type(x).__name__ else f


def run_editor(tables: dict, weather_files: dict, wgn: list, time: tuple, *, python: str | None = None) -> dict:
    """The TxtInOut files (name -> bytes) the SWAT+ Editor writes for these tables.

    ``tables``: QSWAT+'s tables (DataFrames); ``weather_files``: the observed weather
    files in SWAT+ format (``pcp.cli``, ``*.pcp``, ...); ``wgn``: weather generator
    stations (``name``, ``lat``, ``lon``, ``elev``, ``rain_yrs``, ``months``: 12 dicts of
    ``weather_wgn_cli_mon`` columns); ``time``: ``(day_start, yrc_start, day_end, yrc_end)``.
    """
    api = EDITOR / "src" / "api"
    datasets = EDITOR / "release" / "build" / "swatplus_datasets.sqlite"
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        weather = wd / "weather"
        weather.mkdir()
        for name, b in weather_files.items():
            (weather / name).write_bytes(b)
        out = wd / "TxtInOut"
        payload = {k: [{c: _plain(x) for c, x in r.items()} for r in v.to_dict(orient="records")]
                   for k, v in tables.items()}
        payload["wgn"] = wgn
        payload["weather_dir"] = str(weather)
        payload["time"] = list(time)
        (wd / "payload.json").write_text(json.dumps(payload))
        (wd / "driver.py").write_text(_DRIVER)
        res = subprocess.run([python or sys.executable, str(wd / "driver.py"), str(api), str(wd / "project.sqlite"),
                              str(datasets), str(wd / "payload.json"), str(out)], capture_output=True, text=True)
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"the SWAT+ Editor failed ({res.returncode}):\n{res.stdout[-3000:]}\n"
                               f"{res.stderr[-3000:]}")
        return {p.name: p.read_bytes() for p in sorted(out.iterdir()) if p.is_file()}
