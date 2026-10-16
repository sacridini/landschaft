# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of GRASS GIS's r.geomorphon (8.4) for ls.surface.geomorphons.

Usage::

    python tools/make_reference_geomorphons.py [--grass PATH] [--wbt PATH]

Writes ``tests/data/reference/surface/geomorphons.npz`` (``<case>/dem``: the input,
float32; ``<case>/<output>``: r.geomorphon's 12 outputs, NaN / -9999 where null) and
``geomorphons.json`` (the cases' parameters, the cell sizes, the GRASS version), and,
with WhiteboxTools, ``<case>/wbt_forms`` of the cases it can run (a second, looser check).

GRASS runs headless: ``grass84.bat --tmp-project EPSG:32723 --exec python
tools/geomorphons_grass.py`` (``--grass``, default the GRASS of QGIS 3.34). Each DEM is
given to GRASS with one more row of nulls at the bottom: r.geomorphon's row buffer
never shows the last row of the map to the cells of its bottom rows, so the padding
row is the one left out and the lines of sight reach the real last row, as
ls.surface.geomorphons does (its correct form).

The test-suite reads the stored files, so it runs anywhere without GRASS.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "reference" / "surface"
GRASS = r"C:\Program Files\QGIS 3.34.14\bin\grass84.bat"
WBT = Path.home() / "miniforge3" / "Lib" / "site-packages" / "whitebox" / "WBT" / "whitebox_tools.exe"
X0, Y0 = 300000.0, 7400000.0


def _dems() -> dict[str, np.ndarray]:
    """The inputs (stored with the references: never drawn again)."""
    import landschaft as ls

    srtm = ls.load_raster(ls.surface.example_data("orelevation")).values.astype(np.float32)
    srtm = srtm[0] if srtm.ndim == 3 else srtm
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:70, 0:90].astype(np.float64)
    hills = (200 * np.exp(-((yy - 30) ** 2 + (xx - 35) ** 2) / 300) + 120 * np.exp(-((yy - 50) ** 2 + (xx - 70) ** 2)
             / 120) - 80 * np.exp(-((yy - 15) ** 2 + (xx - 75) ** 2) / 90) + 0.4 * xx + rng.normal(0, 0.5, yy.shape))
    holes = hills.copy()
    holes[20:24, 40:47] = np.nan
    holes[60, 10] = np.nan
    holes[0:3, 0:5] = np.nan
    return {
        "srtm": srtm[100:220, 60:210].copy(),
        "srtm_int": np.round(srtm[200:290, 150:280]).astype(np.float32),
        "hills": hills.astype(np.float32),
        "holes": holes.astype(np.float32),
        "terraces": (np.floor(hills / 25) * 25).astype(np.float32),
    }


# name -> (dem, cell (ew, ns), r.geomorphon parameters)
CASES = {
    "default": ("srtm", (30.0, 30.0), dict(search=3, skip=0, flat=1.0, dist=0.0, comparison="anglev1")),
    "search10_skip2_dist": ("srtm", (30.0, 30.0), dict(search=10, skip=2, flat=1.5, dist=5.0, comparison="anglev1")),
    "extended_anglev2": ("srtm", (30.0, 30.0), dict(search=16, skip=0, flat=1.0, dist=0.0, comparison="anglev2",
                                                    extended=True)),
    "extended_skip": ("srtm", (30.0, 30.0), dict(search=16, skip=3, flat=1.0, dist=0.0, comparison="anglev1",
                                                 extended=True)),
    "ties_anglev2_distance": ("srtm_int", (30.0, 30.0), dict(search=8, skip=1, flat=0.8, dist=0.0,
                                                             comparison="anglev2_distance")),
    "ties_anglev1": ("srtm_int", (30.0, 30.0), dict(search=8, skip=0, flat=0.8, dist=3.0, comparison="anglev1")),
    "holes": ("holes", (25.0, 25.0), dict(search=6, skip=1, flat=1.0, dist=0.0, comparison="anglev1")),
    "map_units": ("hills", (20.0, 30.0), dict(search=250, skip=40, flat=2.0, dist=150.0, comparison="anglev2",
                                              meters=True)),
    "terraces_extended": ("terraces", (10.0, 10.0), dict(search=12, skip=0, flat=1.0, dist=0.0,
                                                         comparison="anglev1", extended=True)),
}
OUTPUTS = ("forms", "ternary", "positive", "negative", "intensity", "exposition", "range", "variance",
           "elongation", "azimuth", "extend", "width")


def _write(path: Path, z: np.ndarray, cell, pad: bool = True) -> None:
    ew, ns = cell
    if pad:  # GRASS: a row of nulls at the bottom (see the module docstring)
        z, nodata = np.vstack([z, np.full((1, z.shape[1]), np.nan, np.float32)]), np.nan
    else:    # WhiteboxTools: a nodata value
        z, nodata = np.where(np.isnan(z), -32768, z).astype(np.float32), -32768
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1,
                       dtype="float32", transform=from_origin(X0, Y0, ew, ns), nodata=nodata,
                       crs="EPSG:32723") as dst:
        dst.write(z, 1)


def _read(path: Path, ny: int) -> np.ndarray:
    with rasterio.open(path) as src:
        v = src.read(1).astype(np.float64)
        nd = src.nodata
    if nd is not None and not np.isnan(nd):
        v[v == nd] = np.nan
    return v[:ny]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grass", default=GRASS)
    ap.add_argument("--wbt", default=str(WBT))
    args = ap.parse_args()
    dems = _dems()
    store: dict[str, np.ndarray] = {}
    meta = {"grass": None, "cases": {}}
    with tempfile.TemporaryDirectory(prefix="gm-ref-") as tmp:
        tmp = Path(tmp)
        jobs = []
        for name, (dem, cell, params) in CASES.items():
            z = dems[dem]
            _write(tmp / f"{name}.tif", z, cell)
            jobs.append({"input": str(tmp / f"{name}.tif"), "output": str(tmp / name),
                         **{k: params.get(k, False) for k in ("search", "skip", "flat", "dist", "comparison",
                                                              "meters", "extended")}})
            store[f"{name}/dem"] = z
            meta["cases"][name] = {"dem": dem, "cell": list(cell), "origin": [X0, Y0], **params}
        (tmp / "jobs.json").write_text(json.dumps(jobs))
        run = subprocess.run([args.grass, "--tmp-project", "EPSG:32723", "--exec", "python",
                              str(ROOT / "tools" / "geomorphons_grass.py"), str(tmp / "jobs.json")],
                             capture_output=True, text=True)
        if "done" not in run.stdout:
            raise SystemExit(f"GRASS failed:\n{run.stdout}\n{run.stderr}")
        ver = subprocess.run([args.grass, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
        meta["grass"] = ver[0] if ver else "GRASS 8.4"
        for name, (dem, _, _) in CASES.items():
            ny = dems[dem].shape[0]
            for k in OUTPUTS:
                store[f"{name}/{k}"] = _read(tmp / f"{name}_{k}.tif", ny).astype(np.float32)
        wbt = Path(args.wbt)
        if wbt.is_file():
            for name, (dem, cell, params) in CASES.items():
                if params.get("meters") or params["comparison"] != "anglev1" or params.get("extended"):
                    continue
                out = tmp / f"{name}_wbt.tif"
                _write(tmp / f"{name}_wbt_in.tif", dems[dem], cell, pad=False)
                subprocess.run([str(wbt), "-r=Geomorphons", f"--wd={tmp}", f"--dem={name}_wbt_in.tif",
                                f"-o={out.name}",
                                f"--search={params['search']}", f"--threshold={params['flat']}",
                                f"--fdist={int(params['dist'])}", f"--skip={params['skip']}", "--forms"],
                               capture_output=True, text=True)
                if out.is_file():
                    store[f"{name}/wbt_forms"] = _read(out, dems[dem].shape[0]).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "geomorphons.npz", **store)
    (OUT / "geomorphons.json").write_text(json.dumps(meta, indent=1))
    print("wrote", OUT / "geomorphons.npz", len(store), "arrays")


if __name__ == "__main__":
    main()
