# SPDX-License-Identifier: GPL-3.0-or-later
"""Time ls.plot on rasters, stacks, vectors and tables of several sizes.

    python tools/benchmark_plot.py [raster] [vector] [table] [--cpus 0-11] [--reps 3]

Every case builds the plot and renders its HTML page (what a notebook or a window shows);
the fastest of ``--reps`` runs is reported with the page's size. A first plot is drawn
before timing (importing folium and compiling its templates, about a second once per
session). ``--cpus`` pins the process to those processors.
"""
import argparse
import os
import tempfile
import time
import warnings

os.environ["LANDSCHAFT_PLOT_SHOW"] = "0"
warnings.filterwarnings("ignore")
import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from rasterio.transform import from_origin

import landschaft as ls

rng = np.random.default_rng(0)
work = tempfile.mkdtemp()


def grid(values, res=30.0):
    h, w = values.shape[-2:]
    t = from_origin(500000, 4600000, res, res)
    coords = {"y": 4600000 - res * (np.arange(h) + .5), "x": 500000 + res * (np.arange(w) + .5)}
    dims = ("y", "x")
    if values.ndim == 3:
        coords["band"] = [f"sp{i}" for i in range(values.shape[0])]
        dims = ("band", "y", "x")
    return xr.DataArray(values, coords=coords, dims=dims, name="v").rio.write_crs(32633).rio.write_transform(t)


def smooth(h, w, scale=200):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.sin(xx / scale) * np.cos(yy / (scale * 1.3)) + 0.1 * rng.standard_normal((h, w)).astype(np.float32)


REPS = 3


def run(label, obj, **kw):
    best = 1e9
    for _ in range(REPS):
        t0 = time.perf_counter()
        p = ls.plot(obj, **kw)
        html = p.to_html()
        best = min(best, time.perf_counter() - t0)
    print(f"{label:42s} {best:7.2f} s   html {len(html) / 1e6:6.1f} MB", flush=True)


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("cases", nargs="*", default=["all"])
ap.add_argument("--cpus", default=None)
ap.add_argument("--reps", type=int, default=3)
args = ap.parse_args()
cases, REPS = args.cases, args.reps
if args.cpus:
    import psutil

    lo, hi = (int(v) for v in args.cpus.split("-"))
    psutil.Process().cpu_affinity(list(range(lo, hi + 1)))
ls.plot(grid(np.zeros((10, 10), np.float32))).to_html()  # imports and templates
big = smooth(10000, 10000)
if "all" in cases or "raster" in cases:
    run("float raster 10000x10000 (memory)", grid(big))
    cls = (np.floor((big + 1.2) * 4) % 12).astype(np.uint8)
    run("class raster 10000x10000 (memory)", grid(cls))
    path = ls.write_raster(grid(big), os.path.join(work, "big.tif"))
    run("float raster 10000x10000 (lazy file)", ls.load_raster(path, chunks="auto"))
    run("float raster 2000x2000", grid(big[:2000, :2000]))
    stack = np.stack([smooth(1000, 1000, 50 + 10 * i) for i in range(47)])
    run("stack 47 x 1000x1000", grid(stack))
if "all" in cases or "vector" in cases:
    for n in (10_000, 50_000, 200_000):
        cx, cy = rng.uniform(500000, 800000, n), rng.uniform(4300000, 4600000, n)
        polys = gpd.GeoDataFrame({"v": rng.random(n), "cat": rng.integers(0, 8, n).astype(str)}, crs=32633,
                                 geometry=shapely.buffer(shapely.points(cx, cy), rng.uniform(50, 400, n), quad_segs=4))
        run(f"{n} polygons, numeric column", polys, column="v")
    for n in (200_000, 2_000_000):
        pts = gpd.GeoDataFrame({"v": rng.random(n)}, crs=32633,
                               geometry=gpd.points_from_xy(rng.uniform(500000, 800000, n),
                                                           rng.uniform(4300000, 4600000, n)))
        run(f"{n} points, numeric column", pts, column="v")
if "all" in cases or "table" in cases:
    lc = ls.load_raster(ls.example_data("clc"))
    run("landscape metrics table", ls.landscape_metrics(lc[:500, :500], level="class", metric=["ca", "pland"]))
