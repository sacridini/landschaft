# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden landscape-metric outputs with the R package landscapemetrics.

Usage::

    python tools/make_reference_lsm.py [lsm] [sample] [window] [rmath] [utils]

Needs Rscript with landscapemetrics and terra (see ``tests/lsm_reference.py``
for ``LANDSCHAFT_RSCRIPT`` / ``LANDSCHAFT_R_LIBS``). ``lsm`` (default: all
three parts) writes the input rasters to ``tests/data/lsm/*.tif``, every value
of ``calculate_lsm`` for each option set to
``tests/data/reference/lsm/<case>__<options>.csv.gz`` and a ``manifest.json``;
``sample`` and ``window`` store ``sample_lsm`` and ``window_lsm`` results in
``tests/data/reference/lsm_sample`` and ``lsm_window``; ``rmath`` stores R's
``sum``, ``mean``, ``sd``, ``median``, ``log`` and ``sqrt`` of random vectors and
``lm()`` slopes
(``tests/data/reference/lsm_rmath``), the arithmetic the metrics reproduce;
``utils`` stores the results of landscapemetrics' utility functions
(``get_centroids``, ``get_nearestneighbour``, ``get_circumscribingcircle``,
``get_adjacencies``, ``get_boundaries``, ``extract_lsm``, ``check_landscape``,
``calculate_correlation``; ``tools/lsm_utils_reference.R``) in
``tests/data/reference/lsm_utils``. The test-suite
compares landschaft against these files, so it runs anywhere without R.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import lsm_reference  # noqa: E402
import lsm_scenes  # noqa: E402

INPUTS = ROOT / "tests" / "data" / "lsm"
OUT = ROOT / "tests" / "data" / "reference" / "lsm"
OUT_SAMPLE = ROOT / "tests" / "data" / "reference" / "lsm_sample"
OUT_WINDOW = ROOT / "tests" / "data" / "reference" / "lsm_window"
OUT_RMATH = ROOT / "tests" / "data" / "reference" / "lsm_rmath"
OUT_UTILS = ROOT / "tests" / "data" / "reference" / "lsm_utils"
R_UTILS_SCRIPT = ROOT / "tools" / "lsm_utils_reference.R"

# (input, options): the stored maps, with the utilities' options
UTILS_CASES = {
    "augusta": ("augusta.tif", {}),
    "augusta_d4": ("augusta.tif", {"directions": 4, "consider_boundary": True, "edge_depth": 2}),
    "lm_landscape": ("lm_landscape.tif", {"edge_depth": 3}),
    "synthetic_na": ("synthetic_na.tif", {}),
    "sparse": ("sparse.tif", {"directions": 4}),
    "salt_pepper": ("salt_pepper.tif", {"consider_boundary": True}),
    "na_ring": ("na_ring.tif", {"edge_depth": 2}),
    "checker": ("checker.tif", {"directions": 4}),
}
# an asymmetric neighbourhood (landscapemetrics reads it transposed)
UTILS_NEIGHBOURHOOD = np.array([[np.nan, 1, np.nan], [1, 0, 1], [np.nan, 1, 1]])

# R's arithmetic on vectors like the metrics' (sd uses a scalar sqrt, which
# R's byte-code computes inline; sqrt of a vector goes through the x87 libm).
RMATH_SCRIPT = r"""
lines <- readLines(commandArgs(trailingOnly = TRUE)[[1]])
f <- function(s) {
    x <- as.numeric(strsplit(s, ",")[[1]])
    sprintf("%.17g,%.17g,%.17g,%.17g,%.17g,%.17g", sum(x), mean(x), sd(x), median(x), log(x)[1], sqrt(x)[1])
}
writeLines(vapply(lines, f, ""), commandArgs(trailingOnly = TRUE)[[2]])
"""


def rmath_vectors() -> list[np.ndarray]:
    rng = np.random.default_rng(2026)
    out = []
    for i in range(600):
        n = int(rng.integers(2, 80))
        kind = i % 6
        if kind == 0:
            x = rng.integers(0, 5, n).astype(float)  # counts (ncore)
        elif kind == 1:
            x = rng.integers(1, 400, n) * (0.09 / 10000)  # areas in ha, 0.3 m cells
        elif kind == 2:
            x = rng.uniform(0.001, 3, n)
        elif kind == 3:
            x = np.exp(rng.normal(0, 4, n))
        elif kind == 4:
            x = rng.integers(1, 3000, n) * 0.25 + rng.uniform(0, 1e-9, n)
        else:
            x = 1 + rng.uniform(-0.3, 0.3, n)  # log near 1 (fyl2xp1)
        out.append(x)
    return out


# The slope of lm(y ~ x), as pafrac takes it (NA when R drops x as collinear).
LM_SCRIPT = r"""
lines <- readLines(commandArgs(trailingOnly = TRUE)[[1]])
f <- function(s) {
    v <- strsplit(s, ";")[[1]]
    x <- as.numeric(strsplit(v[1], ",")[[1]])
    y <- as.numeric(strsplit(v[2], ",")[[1]])
    sprintf("%.17g", stats::lm(y ~ x)$coefficients[[2]])
}
writeLines(vapply(lines, f, ""), commandArgs(trailingOnly = TRUE)[[2]])
"""


def lm_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    """(x, y) pairs like pafrac's log(perimeter), log(area), with lengths that
    exercise every remainder of BLAS' unrolled loops, and collinear ones."""
    rng = np.random.default_rng(2027)
    out = []
    for i in range(400):
        n = int(rng.integers(2, 60)) if i % 4 else int(rng.integers(60, 3000))
        kind = i % 5
        if kind == 0:  # patches: cells and their perimeter in cell sides
            cells = rng.geometric(0.3, n)
            sides = 2 * cells + 2 + 2 * rng.integers(0, cells + 1)
            res = float(rng.choice([1.0, 0.3, 30.0, 2.5]))
            x, y = np.log(sides * res), np.log(cells * res * res / 10000 * 10000)
        elif kind == 1:  # mostly single cells: nearly collinear
            cells = np.where(rng.uniform(size=n) < 0.9, 1, rng.integers(2, 5, n))
            x, y = np.log(2.0 * cells + 2), np.log(cells * 1e-4)
        elif kind == 2:
            x = rng.normal(0, 3, n)
            y = 1.7 * x + rng.normal(0, 0.1, n)
        elif kind == 3:  # constant x: NA
            x = np.full(n, float(rng.uniform(1, 9)))
            y = rng.normal(0, 1, n)
        else:  # x constant up to the last bits: collinear by R's tolerance
            x = float(rng.uniform(1, 9)) * (1 + rng.integers(-4, 5, n) * 2.0**-52)
            y = rng.normal(0, 1, n)
        out.append((np.asarray(x, float), np.asarray(y, float)))
    return out


def utils_points(tif: Path) -> pd.DataFrame:
    """Points in and around the raster, on cell edges and corners too."""
    import rasterio

    with rasterio.open(tif) as src:
        b = src.bounds
        rx, ry = src.res
    rng = np.random.default_rng(7)
    xs = list(rng.uniform(b.left, b.right, 12)) + [b.left, b.right, b.left + 3 * rx, b.left - rx, b.right + rx]
    ys = list(rng.uniform(b.bottom, b.top, 12)) + [b.top, b.bottom, b.top - 2 * ry, b.bottom, b.top]
    return pd.DataFrame({"x": xs, "y": ys})


def make_utils(rscript) -> None:
    import gzip
    import shutil

    OUT_UTILS.mkdir(parents=True, exist_ok=True)
    stack = [INPUTS / "synthetic_na.tif", INPUTS / "salt_pepper.tif", INPUTS / "sparse.tif"]
    files = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nb = tmp / "neighbourhood.csv"
        nb.write_text("\n".join(",".join("NA" if np.isnan(v) else str(int(v)) for v in row)
                                for row in UTILS_NEIGHBOURHOOD) + "\n")
        for name, (tif, options) in UTILS_CASES.items():
            points = utils_points(INPUTS / tif)
            pts = tmp / f"{name}_points.csv"
            points.to_csv(pts, index=False, float_format="%.17g")
            out = tmp / name
            opts = {**options, "points": pts, "neighbourhood": nb}
            if name == "augusta":
                opts["stack"] = ",".join(str(s) for s in stack)
            lsm_reference._run_r(rscript, R_UTILS_SCRIPT, [INPUTS / tif, out], opts)
            target = OUT_UTILS / name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            points.to_csv(target / "points.csv", index=False, float_format="%.17g")
            for csv in sorted(out.glob("*.csv")):
                with open(csv, "rb") as src, gzip.GzipFile(target / (csv.name + ".gz"), "wb", mtime=0) as dst:
                    shutil.copyfileobj(src, dst)
            files[name] = {"input": tif, "options": options, "outputs": sorted(p.name for p in out.glob("*.csv"))}
            print(f"{name}: {len(files[name]['outputs'])} tables")
    write_manifest(OUT_UTILS, rscript, files, {"stack": [p.name for p in stack],
                                              "neighbourhood": UTILS_NEIGHBOURHOOD.tolist()})


def make_rmath(rscript) -> None:
    OUT_RMATH.mkdir(parents=True, exist_ok=True)
    vectors = rmath_vectors()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "x.txt"
        dst = Path(tmp) / "out.txt"
        src.write_text("\n".join(",".join(f"{v:.17g}" for v in x) for x in vectors) + "\n")
        script = Path(tmp) / "rmath.R"
        script.write_text(RMATH_SCRIPT)
        subprocess.run([str(rscript), str(script), str(src), str(dst)], check=True, env=lsm_reference.r_environment())
        results = dst.read_text().split()
    df = pd.DataFrame({"x": [",".join(f"{v:.17g}" for v in x) for x in vectors]})
    df[["sum", "mean", "sd", "median", "log", "sqrt"]] = [r.split(",") for r in results]
    df.to_csv(OUT_RMATH / "rmath.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    print(f"rmath.csv.gz: {len(vectors)} vectors")
    pairs = lm_pairs()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "xy.txt"
        dst = Path(tmp) / "out.txt"
        src.write_text("\n".join(";".join(",".join(f"{v:.17g}" for v in a) for a in xy) for xy in pairs) + "\n")
        script = Path(tmp) / "lm.R"
        script.write_text(LM_SCRIPT)
        subprocess.run([str(rscript), str(script), str(src), str(dst)], check=True, env=lsm_reference.r_environment())
        slopes = dst.read_text().split()
    df = pd.DataFrame({"x": [",".join(f"{v:.17g}" for v in x) for x, _ in pairs],
                       "y": [",".join(f"{v:.17g}" for v in y) for _, y in pairs], "slope": slopes})
    df.to_csv(OUT_RMATH / "lm.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    print(f"lm.csv.gz: {len(pairs)} regressions")
    write_manifest(OUT_RMATH, rscript, {"rmath.csv.gz": {"vectors": len(vectors)}, "lm.csv.gz": {"pairs": len(pairs)}})

OPTION_SETS = {
    "default": {},
    "d4": {"directions": 4},
    "boundary": {"count_boundary": True, "consider_boundary": True, "edge_depth": 2},
    "complexity": {"neighbourhood": 8, "ordered": False, "base": "log", "classes_max": 20, "cell_center": True},
    "core": {"edge_depth": 3, "directions": 4},
}

# landscapemetrics' own example data, cropped so the reference stays small.
EXPORT_LM_DATA = r"""
suppressPackageStartupMessages({library(terra); library(landscapemetrics)})
out <- commandArgs(trailingOnly = TRUE)[[1]]
save <- function(r, name) {
    if (!inherits(r, "SpatRaster")) r <- terra::rast(r)
    levels(r) <- NULL
    terra::writeRaster(r, file.path(out, paste0(name, ".tif")), datatype = "INT4S",
                       NAflag = -2147483648, overwrite = TRUE)
}
save(landscapemetrics::landscape, "lm_landscape")
aug <- terra::rast(landscapemetrics::augusta_nlcd)
save(aug[101:220, 201:360, drop = FALSE], "augusta")
pod <- terra::rast(landscapemetrics::podlasie_ccilc)
save(pod[101:200, 151:270, drop = FALSE], "podlasie")
"""


def synthetic_cases() -> dict[str, tuple]:
    cases = {
        "synthetic_na": (
            lsm_scenes.clustered((45, 60), [-5, 0, 7, 1000], seed=11, sigma=2.5, na_fraction=0.12),
            (30.0, 20.0),
        ),
        "salt_pepper": (lsm_scenes.salt_and_pepper((25, 30), [1, 2, 3], seed=7, na_fraction=0.05), (10.0, 10.0)),
        # few, distant patches (enn over long distances) and a non-integer cell size
        "sparse": (lsm_scenes.sparse((70, 90), [2, 3, 4], seed=12, n_blobs=25, background=1), (0.3, 0.3)),
        "sparse_na": (lsm_scenes.sparse((60, 50), [5, 6], seed=13, n_blobs=18, na_fraction=0.02), (2.5, 2.5)),
    }
    for name, data in lsm_scenes.small_cases().items():
        cases[name] = (data, (1.0, 1.0))
    return cases


def sample_cases() -> dict[str, tuple]:
    """name: (input raster, plots table, sample_lsm options)."""
    import rasterio

    with rasterio.open(INPUTS / "augusta.tif") as src:
        b = src.bounds
    rng = np.random.default_rng(3)
    points = pd.DataFrame({"x": rng.uniform(b.left - 100, b.right + 100, 6), "y": rng.uniform(b.bottom - 100, b.top + 100, 6)})
    points.loc[0] = [(b.left + b.right) / 2, (b.bottom + b.top) / 2]
    polygons = pd.DataFrame(
        {
            "wkt": [
                "POLYGON ((100 100, 700 120, 650 600, 150 500, 100 100))",
                "POLYGON ((900 200, 1500 200, 1500 800, 900 800, 900 200), "
                "(1100 400, 1300 400, 1300 600, 1100 600, 1100 400))",
                "MULTIPOLYGON (((10 10, 200 10, 200 200, 10 200, 10 10)), "
                "((1600 700, 1790 700, 1790 890, 1600 890, 1600 700)))",
                "POLYGON ((1000 850, 1700 880, 1750 950, 1000 950, 1000 850))",
            ],
            "plot_id": [40, 10, 30, 20],
        }
    )
    return {
        "augusta_square": ("augusta.tif", points, {"shape": "square", "size": 450}),
        "augusta_circle": ("augusta.tif", points, {"shape": "circle", "size": 600}),
        "augusta_circle_all": ("augusta.tif", points, {"shape": "circle", "size": 1000, "all_classes": True, "directions": 4}),
        "augusta_vector": ("augusta.tif", points.iloc[:3], {"shape": "circle", "size": 500, "vector": True, "level": "landscape"}),
        "synthetic_polygons": ("synthetic_na.tif", polygons, {}),
        "synthetic_polygons_all": ("synthetic_na.tif", polygons, {"all_classes": True}),
        "synthetic_polygons_boundary": (
            "synthetic_na.tif", polygons,
            {"consider_boundary": True, "edge_depth": 2, "count_boundary": True, "directions": 4},
        ),
        # scale_sample: several buffer sizes
        "augusta_scale_square": ("augusta.tif", points, {"shape": "square", "sizes": "150,450,900"}),
        "augusta_scale_circle": ("augusta.tif", points.iloc[:4], {"shape": "circle", "sizes": "300,700",
                                                                   "directions": 4}),
    }


# window_lsm is slow: small rasters and a selection of landscape metrics
# (gyrate_* fail in window_lsm, which does not pass cell_center).
WINDOW_METRICS = [
    "ta", "np", "pd", "lpi", "te", "ed", "shdi", "sidi", "msiei", "contag", "pr", "lsi", "ai", "area_mn", "enn_mn",
    "core_mn", "mesh", "iji", "ent", "condent", "cohesion", "pafrac", "circle_mn", "pladj", "division", "split",
    "ndca", "shape_mn", "rpr",
]


def window_cases() -> dict[str, tuple]:
    """name: (data, cell size, window, window_lsm options)."""
    circle = np.hypot(*np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))) <= 2.2
    return {
        "square3": (lsm_scenes.clustered((14, 17), [1, 2, 3], seed=1, sigma=1.2, na_fraction=0.08), (10.0, 10.0),
                    np.ones((3, 3), bool), {"classes_max": 5}),
        "circle5_d4": (lsm_scenes.clustered((12, 15), [2, 5, 7, 9], seed=2, sigma=1.0), (1.0, 1.0), circle,
                       {"directions": 4}),
        "square5_boundary": (lsm_scenes.salt_and_pepper((11, 13), [1, 2], seed=3, na_fraction=0.1), (5.0, 5.0),
                             np.ones((5, 5), bool), {"count_boundary": True, "edge_depth": 2, "consider_boundary": True}),
    }


def make_sample(rscript, force: bool = False) -> None:
    """sample_lsm / scale_sample references; existing ones are kept unless
    `force` (landscapemetrics' random circles would change their last bits)."""
    OUT_SAMPLE.mkdir(parents=True, exist_ok=True)
    manifest = OUT_SAMPLE / "manifest.json"
    files = json.loads(manifest.read_text())["files"] if manifest.exists() and not force else {}
    for name, (tif, plots, options) in sample_cases().items():
        if f"{name}.csv.gz" in files and (OUT_SAMPLE / f"{name}.csv.gz").exists():
            continue
        df = lsm_reference.run_sample(rscript, INPUTS / tif, plots, **options)
        target = OUT_SAMPLE / f"{name}.csv.gz"
        df.to_csv(target, index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
        plots.to_csv(OUT_SAMPLE / f"{name}__plots.csv", index=False, float_format="%.17g")
        files[target.name] = {"input": tif, "plots": f"{name}__plots.csv", "options": options, "rows": len(df)}
        print(f"{target.name}: {len(df)} values")
    write_manifest(OUT_SAMPLE, rscript, files)


def make_window(rscript) -> None:
    OUT_WINDOW.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, (data, res, window, options) in window_cases().items():
        tif = OUT_WINDOW / f"{name}.tif"
        lsm_reference.write_landscape(tif, data, res)
        maps = lsm_reference.run_window(rscript, tif, window, ["lsm_l_" + m for m in WINDOW_METRICS], **options)
        target = OUT_WINDOW / f"{name}.npz"
        np.savez_compressed(target, window=window, **maps)
        files[target.name] = {"input": tif.name, "options": options, "metrics": sorted(maps)}
        print(f"{target.name}: {len(maps)} metrics")
    write_manifest(OUT_WINDOW, rscript, files)


def write_manifest(folder: Path, rscript, files: dict, extra: dict | None = None) -> None:
    version = subprocess.run(
        [str(rscript), "-e", "cat(as.character(packageVersion('landscapemetrics')))"],
        capture_output=True, text=True, env=lsm_reference.r_environment(), check=True,
    ).stdout.strip()
    manifest = {"generated": dt.date.today().isoformat(), "landscapemetrics": version, **(extra or {}), "files": files}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parts = {a for a in sys.argv[1:] if not a.startswith("--")} or {"lsm", "sample", "window", "rmath", "utils"}
    unknown = parts - {"lsm", "sample", "window", "rmath", "utils"}
    if unknown:
        sys.exit(f"unknown part(s): {sorted(unknown)}")
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    if "lsm" in parts:
        make_lsm(rscript)
    if "sample" in parts:
        make_sample(rscript, force="--force" in sys.argv)
    if "rmath" in parts:
        make_rmath(rscript)
    if "utils" in parts:
        make_utils(rscript)
    if "window" in parts:
        make_window(rscript)


def make_lsm(rscript) -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "export.R"
        script.write_text(EXPORT_LM_DATA)
        subprocess.run([str(rscript), str(script), str(INPUTS)], check=True, env=lsm_reference.r_environment())
    for name, (data, res) in synthetic_cases().items():
        lsm_reference.write_landscape(INPUTS / f"{name}.tif", data, res)

    files = {}
    for tif in sorted(INPUTS.glob("*.tif")):
        for opt_name, options in OPTION_SETS.items():
            df = lsm_reference.run_tif(rscript, tif, **options)
            target = OUT / f"{tif.stem}__{opt_name}.csv.gz"
            df.to_csv(target, index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
            files[target.name] = {"input": tif.name, "options": options, "rows": len(df)}
            print(f"{target.name}: {len(df)} values")

    write_manifest(OUT, rscript, files, {"option_sets": OPTION_SETS})


if __name__ == "__main__":
    main()
