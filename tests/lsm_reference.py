# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the R package landscapemetrics on numpy arrays.

``LANDSCHAFT_RSCRIPT`` points to ``Rscript`` (found on PATH or under
``C:/Program Files/R`` otherwise) and ``LANDSCHAFT_R_LIBS`` to an R library
holding the landscapemetrics version to compare against.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT = ROOT / "tools" / "lsm_reference.R"
R_SAMPLE_SCRIPT = ROOT / "tools" / "lsm_sample_reference.R"
R_WINDOW_SCRIPT = ROOT / "tools" / "lsm_window_reference.R"
NODATA = np.iinfo(np.int32).min
DEFAULT_R_LIBS = Path.home() / "AppData" / "Local" / "R" / "landschaft-lsm-lib"


def find_rscript() -> Path | None:
    exe = os.environ.get("LANDSCHAFT_RSCRIPT")
    if exe:
        return Path(exe) if Path(exe).is_file() else None
    found = shutil.which("Rscript")
    if found:
        return Path(found)
    candidates = sorted(glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"))
    return Path(candidates[-1]) if candidates else None


def r_environment() -> dict[str, str]:
    env = dict(os.environ)
    if "LANDSCHAFT_R_LIBS" not in env and DEFAULT_R_LIBS.is_dir():
        env["LANDSCHAFT_R_LIBS"] = str(DEFAULT_R_LIBS)
    return env


def write_landscape(path: Path, data: np.ndarray, res=(1.0, 1.0), crs=None) -> None:
    """GeoTIFF (int32, NA = int32 min) that R reads back unchanged."""
    values = np.asarray(data)
    mask = None
    if np.ma.isMaskedArray(data):
        mask = np.ma.getmaskarray(data)
        values = np.ma.getdata(data)
    out = values.astype(np.int32)
    if mask is not None:
        out = np.where(mask, NODATA, out)
    transform = Affine(res[0], 0.0, 0.0, 0.0, -res[1], res[1] * out.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path, "w", driver="GTiff", width=out.shape[1], height=out.shape[0], count=1,
            dtype="int32", nodata=NODATA, transform=transform, crs=crs,
        ) as dst:
            dst.write(out, 1)


def run_tif(rscript: os.PathLike, tif: os.PathLike, **options) -> pd.DataFrame:
    """calculate_lsm on a raster file; returns level, class, id, metric, value."""
    with tempfile.TemporaryDirectory(prefix="lsm_ref_") as tmp:
        out = Path(tmp) / "out.csv"
        _run_r(rscript, R_SCRIPT, [tif, out], options)
        return read_reference(out)


def _run_r(rscript: os.PathLike, script: Path, files: list, options: dict) -> None:
    cmd = [str(rscript), str(script)] + [str(f) for f in files]
    for key, value in options.items():
        if isinstance(value, bool):
            value = "TRUE" if value else "FALSE"
        elif value is None:
            value = "NULL"
        elif isinstance(value, (list, tuple)):
            value = ",".join(value)
        cmd.append(f"{key}={value}")
    proc = subprocess.run(cmd, capture_output=True, env=r_environment(), timeout=3600)
    if proc.returncode != 0 or not Path(files[-1]).exists():
        raise RuntimeError(f"Rscript failed ({proc.returncode}): {proc.stderr.decode(errors='replace')}")


def run_sample(rscript: os.PathLike, tif: os.PathLike, plots: pd.DataFrame, **options) -> pd.DataFrame:
    """sample_lsm on a raster file; `plots` has x, y (points) or wkt, and
    optionally plot_id. Returns level, class, id, metric, value, plot_id,
    percentage_inside."""
    with tempfile.TemporaryDirectory(prefix="lsm_sample_") as tmp:
        plots_csv = Path(tmp) / "plots.csv"
        plots.to_csv(plots_csv, index=False, float_format="%.17g")
        out = Path(tmp) / "out.csv"
        _run_r(rscript, R_SAMPLE_SCRIPT, [tif, plots_csv, out], options)
        df = read_reference(out)
        df["percentage_inside"] = df["percentage_inside"].astype(float)
        return df


def run_window(rscript: os.PathLike, tif: os.PathLike, window: np.ndarray, what: list[str], **options) -> dict:
    """window_lsm on a raster file; returns {metric: values (rows x columns)}."""
    with rasterio.open(tif) as src:
        shape = (src.height, src.width)
    with tempfile.TemporaryDirectory(prefix="lsm_window_") as tmp:
        window_csv = Path(tmp) / "window.csv"
        w = np.where(np.asarray(window, dtype=bool), "1", "NA")
        window_csv.write_text("\n".join(",".join(row) for row in w) + "\n")
        out = Path(tmp) / "out.csv"
        _run_r(rscript, R_WINDOW_SCRIPT, [tif, window_csv, out], {"what": list(what), **options})
        df = pd.read_csv(out, na_values=["NA", ""], keep_default_na=False, float_precision="round_trip")
    return {m: g.sort_values("cell")["value"].to_numpy(float).reshape(shape) for m, g in df.groupby("metric")}


def run_array(rscript: os.PathLike, data: np.ndarray, res=(1.0, 1.0), **options) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="lsm_in_") as tmp:
        tif = Path(tmp) / "in.tif"
        write_landscape(tif, data, res)
        return run_tif(rscript, tif, **options)


def read_reference(path: os.PathLike) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["NA", ""], keep_default_na=False, float_precision="round_trip")
    df["class"] = df["class"].astype("Int64")
    df["id"] = df["id"].astype("Int64")
    df["value"] = df["value"].astype(float)
    return df
