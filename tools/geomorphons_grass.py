# SPDX-License-Identifier: GPL-3.0-or-later
"""Run inside GRASS (``grass --tmp-project EPSG:32723 --exec python tools/geomorphons_grass.py jobs.json``):
r.geomorphon on each job's GeoTIFF, every output written as a GeoTIFF next to it.
Called by ``tools/make_reference_geomorphons.py``."""

import json
import sys

import grass.script as gs

OUTPUTS = {"forms": "Int32", "ternary": "Int32", "positive": "Int32", "negative": "Int32",
           "intensity": "Float32", "exposition": "Float32", "range": "Float32", "variance": "Float32",
           "elongation": "Float32", "azimuth": "Float32", "extend": "Float32", "width": "Float32"}

jobs = json.load(open(sys.argv[1]))
for job in jobs:
    gs.run_command("r.in.gdal", input=job["input"], output="dem", flags="o", overwrite=True, quiet=True)
    gs.run_command("g.region", raster="dem", quiet=True)
    names = {k: f"gm_{k}" for k in OUTPUTS}
    flags = ("m" if job.get("meters") else "") + ("e" if job.get("extended") else "")
    gs.run_command("r.geomorphon", elevation="dem", search=job["search"], skip=job["skip"], flat=job["flat"],
                   dist=job["dist"], comparison=job["comparison"], flags=flags, overwrite=True, quiet=True,
                   **names)
    for k, t in OUTPUTS.items():
        nodata = -9999 if t == "Int32" else None
        kw = {"nodata": nodata} if nodata is not None else {}
        gs.run_command("r.out.gdal", input=names[k], output=f"{job['output']}_{k}.tif", type=t, format="GTiff",
                       flags="fc", overwrite=True, quiet=True, **kw)
print("done", len(jobs))
