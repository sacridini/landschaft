# SPDX-License-Identifier: GPL-3.0-or-later
"""Time ls.zonal_stats and ls.extract against rasterstats and exactextract.

    python tools/benchmark_zonal.py [--reps N] [--threads N] [--cpus 0-11]

A 5000 x 5000 float32 raster (25 million cells) with 2000 small or 20 large
random polygons, and a million points. Every configuration runs in turn in
every round (alternating), and the fastest of ``--reps`` rounds is reported:
Landschaft on one thread and on ``--threads``, rasterstats and exactextract
(single-threaded; exactextract weights the cells by the share of them a polygon
covers, the others take the cells whose centre is inside). ``--cpus`` pins the
process to those processors (e.g. the performance cores of a hybrid CPU).
Needs ``pip install rasterstats exactextract``.
"""

from __future__ import annotations

import argparse
import tempfile
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import Polygon

import landschaft as ls

SIDE = 5000
T = from_origin(0, 150000, 30, 30)
STATS = ["mean", "min", "max", "std", "median"]


def polygons(rng, n, rmin, rmax):
    out = []
    while len(out) < n:
        cx, cy = rng.uniform(0, 150000, 2)
        a = np.sort(rng.uniform(0, 2 * np.pi, 16))
        r = rng.uniform(rmin, rmax, 16)
        g = Polygon(zip(cx + r * np.cos(a), cy + r * np.sin(a))).buffer(0)
        if not g.is_empty:
            out.append(g)
    return gpd.GeoDataFrame(geometry=out, crs="EPSG:32633")


def rounds(configs: dict, reps: int) -> dict:
    """The fastest of reps alternating runs of every configuration."""
    best = {k: float("inf") for k in configs}
    for _ in range(reps):
        for name, f in configs.items():
            t0 = time.perf_counter()
            f()
            best[name] = min(best[name], time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=ls.default_threads())
    ap.add_argument("--cpus", default=None, help="processors to pin to, e.g. 0-11")
    args = ap.parse_args()
    if args.cpus:
        import psutil

        lo, hi = (int(v) for v in args.cpus.split("-"))
        psutil.Process().cpu_affinity(list(range(lo, hi + 1)))
    from exactextract import exact_extract
    from rasterstats import point_query, zonal_stats

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    v = rng.random((SIDE, SIDE)).astype(np.float32)
    da = xr.DataArray(v, coords={"y": 150000 - 30 * (np.arange(SIDE) + .5), "x": 30 * (np.arange(SIDE) + .5)},
                      dims=("y", "x"), name="v").rio.write_crs("EPSG:32633").rio.write_transform(T)
    work = Path(tempfile.mkdtemp(prefix="landschaft_zonal_"))
    path = ls.write_raster(da, work / "values.tif", compress=None)
    n = args.threads
    print(f"| Case | rasterstats | exactextract | Landschaft, 1 thread | Landschaft, {n} threads |")
    print("|---|---|---|---|---|")
    with rasterio.open(path) as src:
        for label, zones in (("2000 polygons, 16 million cells", polygons(rng, 2000, 200, 3000)),
                             ("20 polygons, 22 million cells", polygons(rng, 20, 8000, 30000))):
            ref = ls.zonal_stats(da, zones, STATS, n_threads=1)
            check = np.array([d["mean"] for d in zonal_stats(zones.geometry, v, affine=T, stats="mean", nodata=-9999)])
            assert np.nanmax(np.abs(check - ref["mean"].to_numpy())) < 1e-6, "rasterstats disagrees"
            best = rounds({
                "rasterstats": lambda: zonal_stats(zones.geometry, v, affine=T, stats=" ".join(STATS), nodata=-9999),
                "exactextract": lambda: exact_extract(src, zones, ["mean", "min", "max", "stdev", "median"],
                                                      output="pandas"),
                "one": lambda: ls.zonal_stats(da, zones, STATS, n_threads=1),
                "many": lambda: ls.zonal_stats(da, zones, STATS, n_threads=n),
            }, args.reps)
            print(f"| zonal_stats, {label} | {best['rasterstats']:.2f} s | {best['exactextract']:.2f} s | "
                  f"{best['one']:.2f} s | {best['many']:.2f} s |")
    m, sample = 1_000_000, 20_000
    x = rng.uniform(0, 150000, m)
    y = rng.uniform(0, 150000, m)
    pts = gpd.GeoSeries(gpd.points_from_xy(x[:sample], y[:sample]), crs="EPSG:32633")
    assert np.array_equal(np.array(point_query(pts, v, affine=T, interpolate="nearest"), float),
                          ls.extract(da, (x[:sample], y[:sample]))["v"].to_numpy())
    best = rounds({
        "rasterstats": lambda: point_query(pts, v, affine=T, interpolate="nearest"),
        "one": lambda: ls.extract(da, (x, y), n_threads=1),
        "many": lambda: ls.extract(da, (x, y), n_threads=n),
    }, args.reps)
    print(f"| extract, 1 million points | {best['rasterstats'] * m / sample:.0f} s (from {sample:,}) | - | "
          f"{best['one']:.3f} s | {best['many']:.3f} s |")


if __name__ == "__main__":
    main()
