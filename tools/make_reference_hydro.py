# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of WhiteboxTools for ls.hydro.

Usage::

    python tools/make_reference_hydro.py

Builds the test DEMs (stored with the outputs, so the tests never redraw them
with the platform's maths), runs WhiteboxTools 2.4 on each (``FillDepressions``,
``D8Pointer``, ``D8FlowAccumulation``, ``DInfPointer``, ``DInfFlowAccumulation``,
``FD8FlowAccumulation``, ``ExtractStreams``, ``ElevationAboveStream``, ``Slope``,
``WetnessIndex``, ``StreamPowerIndex``, ``SedimentTransportIndex``) and stores
inputs and outputs in ``tests/data/reference/hydro/whitebox.npz`` (``<case>/<key>``),
with the versions in ``whitebox.json``.

``FillDepressions`` does not let water leave the map at its edge (nor at missing
cells): a depression that touches the edge is raised with the edge cells in it. The
correct filling (``filled_edge``) is FillDepressions run on the DEM framed by a ring of
very low cells, cropped back; the downstream tools all read WhiteboxTools' own
``filled`` DEM, so each is checked on the same input. Needs the ``whitebox`` Python package's
binary (``LANDSCHAFT_WBT`` overrides its path). The test-suite reads the stored files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "reference" / "hydro"
NODATA = -9999.0


def wbt_exe() -> Path:
    env = os.environ.get("LANDSCHAFT_WBT")
    if env:
        return Path(env)
    import whitebox

    return Path(whitebox.__file__).parent / "WBT" / ("whitebox_tools.exe" if sys.platform == "win32" else "whitebox_tools")


def run(exe: Path, wd: Path, tool: str, *args: str) -> None:
    cmd = [str(exe), f"-r={tool}", f"--wd={wd}", "--max_procs=4", *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or "Error" in res.stdout:
        raise RuntimeError(f"{tool}: {res.stdout[-2000:]}{res.stderr[-2000:]}")


def write(path: Path, values: np.ndarray, dx: float, dy: float) -> None:
    import rasterio
    from rasterio.transform import from_origin

    v = np.where(np.isnan(values), NODATA, values)
    with rasterio.open(path, "w", driver="GTiff", height=v.shape[0], width=v.shape[1], count=1, dtype="float64",
                       crs="EPSG:32610", transform=from_origin(500000.0, 4800000.0, dx, dy), nodata=NODATA) as dst:
        dst.write(v, 1)


def read(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float64)
        nd = src.nodata
    if nd is not None:
        a[a == nd] = np.nan
    return a


def cases() -> dict:
    import landschaft as ls

    oregon = ls.load_raster(ls.surface.example_data("orelevation")).values.squeeze().astype(np.float64)[100:280, 60:280]
    holes = oregon.copy()
    holes[60:90, 40:55] = np.nan            # a lake of missing cells
    holes[150:160, 170:220] = np.nan        # touching the edge
    rng = np.random.default_rng(20261016)
    walk = np.cumsum(np.cumsum(rng.uniform(-1.0, 1.0, (120, 160)), axis=0), axis=1)   # rough terrain with pits
    walk += rng.uniform(0.0, 3.0, walk.shape)
    terraces = np.round(oregon / 10.0) * 10.0                                          # many flats and ties
    return {"oregon": (oregon, 30.0, 30.0), "holes": (holes, 30.0, 30.0), "walk": (walk, 30.0, 25.0),
            "terraces": (terraces, 30.0, 30.0)}


def main() -> None:
    exe = wbt_exe()
    version = subprocess.run([str(exe), "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    out: dict = {}
    with tempfile.TemporaryDirectory(prefix="ls-hydro-ref-") as tmp:
        wd = Path(tmp)
        for name, (dem, dx, dy) in cases().items():
            write(wd / "dem.tif", dem, dx, dy)
            run(exe, wd, "FillDepressions", "--dem=dem.tif", "-o=filled.tif")
            if not np.isnan(dem).any():      # the filling with the edge as outlet
                framed = np.pad(dem, 1, constant_values=-1e9)
                write(wd / "framed.tif", framed, dx, dy)
                run(exe, wd, "FillDepressions", "--dem=framed.tif", "-o=framed_filled.tif")
                out[f"{name}/filled_edge"] = read(wd / "framed_filled.tif")[1:-1, 1:-1]
            run(exe, wd, "D8Pointer", "--dem=dem.tif", "-o=d8_raw.tif", "--esri_pntr")
            run(exe, wd, "D8Pointer", "--dem=filled.tif", "-o=d8.tif", "--esri_pntr")
            run(exe, wd, "D8FlowAccumulation", "--input=filled.tif", "-o=d8acc.tif", "--out_type=cells")
            run(exe, wd, "D8FlowAccumulation", "--input=filled.tif", "-o=d8sca.tif",
                "--out_type=specific contributing area")
            run(exe, wd, "DInfPointer", "--dem=filled.tif", "-o=dinf.tif")
            run(exe, wd, "DInfFlowAccumulation", "--input=filled.tif", "-o=dinfacc.tif", "--out_type=cells")
            run(exe, wd, "FD8FlowAccumulation", "--dem=filled.tif", "-o=fd8acc.tif", "--out_type=cells")
            run(exe, wd, "ExtractStreams", "--flow_accum=d8acc.tif", "-o=streams.tif", "--threshold=50")
            run(exe, wd, "ElevationAboveStream", "--dem=filled.tif", "--streams=streams.tif", "-o=hand.tif")
            run(exe, wd, "Slope", "--dem=filled.tif", "-o=slope.tif", "--units=degrees")
            run(exe, wd, "FD8FlowAccumulation", "--dem=filled.tif", "-o=fd8sca.tif",
                "--out_type=specific contributing area")
            run(exe, wd, "WetnessIndex", "--sca=fd8sca.tif", "--slope=slope.tif", "-o=twi.tif")
            run(exe, wd, "StreamPowerIndex", "--sca=fd8sca.tif", "--slope=slope.tif", "-o=spi.tif")
            run(exe, wd, "SedimentTransportIndex", "--sca=fd8sca.tif", "--slope=slope.tif", "-o=sti.tif")
            out[f"{name}/dem"] = dem
            out[f"{name}/cell"] = np.array([dx, dy])
            for key in ("filled", "d8_raw", "d8", "d8acc", "d8sca", "dinf", "dinfacc", "fd8acc", "streams", "hand",
                        "slope", "fd8sca", "twi", "spi", "sti"):
                v = read(wd / f"{key}.tif")
                if key in ("d8_raw", "d8", "streams"):     # small integers, -1 where missing
                    v = np.where(np.isnan(v), -1, v).astype(np.int16)
                out[f"{name}/{key}"] = v
            print(name, "done")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "whitebox.npz", **out)
    (OUT / "whitebox.json").write_text(json.dumps({"whitebox": version, "cases": sorted({k.split("/")[0] for k in out})},
                                                  indent=1))
    print("written", OUT / "whitebox.npz")


if __name__ == "__main__":
    main()
