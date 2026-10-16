# SPDX-License-Identifier: GPL-3.0-or-later
"""Speed of ls.surface.geomorphons against GRASS GIS's r.geomorphon (8.4, single-threaded).

Usage::

    python tools/benchmark_geomorphons.py [--size 2000] [--search 10] [--cpus 0-3] [--threads 4]

A DEM of ``size`` x ``size`` cells (the SRTM example tiled, plus a gentle trend) is
written as a GeoTIFF; GRASS times r.geomorphon alone (forms output, no import or export),
inside ``grass84.bat --exec``; landschaft times geomorphons() on the same cells with
``--threads`` threads, pinned to ``--cpus``. Both outputs are compared.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

GRASS = r"C:\Program Files\QGIS 3.34.14\bin\grass84.bat"
INNER = '''
import sys, time
import grass.script as gs
gs.run_command("r.in.gdal", input=sys.argv[1], output="dem", flags="o", quiet=True)
gs.run_command("g.region", raster="dem", quiet=True)
t = time.perf_counter()
gs.run_command("r.geomorphon", elevation="dem", forms="f", search=int(sys.argv[2]), quiet=True)
print("seconds", time.perf_counter() - t)
gs.run_command("r.out.gdal", input="f", output=sys.argv[3], type="Int32", nodata=-9999, flags="fc", quiet=True)
'''


def main() -> None:
    import landschaft as ls

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--search", type=int, default=10)
    ap.add_argument("--cpus", default="0-3")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.cpus.split("-"))
    try:
        import psutil

        psutil.Process().cpu_affinity(list(range(lo, hi + 1)))
    except Exception:  # noqa: BLE001
        pass
    srtm = ls.load_raster(ls.surface.example_data("orelevation")).values.astype(np.float32)
    srtm = srtm[0] if srtm.ndim == 3 else srtm
    reps = -(-a.size // srtm.shape[0])
    z = np.tile(srtm, (reps, reps))[:a.size, :a.size]
    z = (z + np.arange(a.size, dtype=np.float32)[None, :] * 0.01).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pad = np.vstack([z, np.full((1, a.size), np.nan, np.float32)])  # see make_reference_geomorphons.py
        with rasterio.open(tmp / "dem.tif", "w", driver="GTiff", height=pad.shape[0], width=a.size, count=1,
                           dtype="float32", transform=from_origin(0, a.size * 30.0 + 30, 30, 30), nodata=np.nan,
                           crs="EPSG:32723") as f:
            f.write(pad, 1)
        (tmp / "inner.py").write_text(INNER)
        r = subprocess.run([GRASS, "--tmp-project", "EPSG:32723", "--exec", "python", str(tmp / "inner.py"),
                            str(tmp / "dem.tif"), str(a.search), str(tmp / "f.tif")], capture_output=True, text=True)
        grass_s = float(r.stdout.split("seconds")[-1].split()[0])
        with rasterio.open(tmp / "f.tif") as s:
            ref = s.read(1).astype(float)[:a.size]
        ref[ref == -9999] = np.nan
    ext = (0, a.size * 30.0, 30.0, a.size * 30.0 + 30)
    ls.surface.geomorphons(z[:200, :200], search=a.search)   # warm up
    best = np.inf
    for _ in range(3):
        t = time.perf_counter()
        g = ls.surface.geomorphons(z, search=a.search, extent=ext, n_threads=a.threads)
        best = min(best, time.perf_counter() - t)
    m = ~np.isnan(g.forms.values) & ~np.isnan(ref)
    same = np.mean(g.forms.values[m] == ref[m])
    print(f"{a.size} x {a.size}, search {a.search}: GRASS r.geomorphon {grass_s:.2f} s; landschaft {best:.2f} s "
          f"({a.threads} threads, cpus {a.cpus}); {grass_s / best:.1f}x; forms identical on {same:.4%} of cells")


if __name__ == "__main__":
    main()
