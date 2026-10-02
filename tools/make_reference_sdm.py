# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of the R packages behind ls.sdm.

Usage::

    python tools/make_reference_sdm.py [group ...] [--case NAME ...]

Runs the cases of ``tests/sdm_cases.py`` (all groups by default) through
``tools/sdm_reference.R`` and stores, per group, the numeric results in
``tests/data/reference/sdm/<group>.npz`` (``<case>/<key>`` arrays) and their
kinds, shapes and the text results in ``<group>.json`` (with the package
versions). ``--case`` regenerates only the named cases of the group (the
others are kept).

Needs Rscript (``LANDSCHAFT_RSCRIPT`` as in ``tests/lsm_reference.py``) and a
library with CoordinateCleaner, spThin, usdm, blockCV, ENMeval, ecospat,
predicts, dismo, maxnet, glmnet, biomod2 and mop (``LANDSCHAFT_SDM_R_LIBS``,
default ``%LOCALAPPDATA%/R/landschaft-sdm-lib``, put before the user library,
which provides terra, sf and jsonlite). The test-suite reads the stored
files, so it runs anywhere without R.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import lsm_reference  # noqa: E402
import sdm_cases  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference" / "sdm"
R_SCRIPT = ROOT / "tools" / "sdm_reference.R"
DEFAULT_LIBS = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "R" / "landschaft-sdm-lib"
# The user library (terra, sf, jsonlite), as R_LIBS replaces it.
USER_LIBS = sorted(DEFAULT_LIBS.parent.glob("win-library/*"))


def r_environment() -> dict[str, str]:
    env = lsm_reference.r_environment()
    libs = [env.get("LANDSCHAFT_SDM_R_LIBS", str(DEFAULT_LIBS)), *(str(p) for p in USER_LIBS[-1:])]
    env["R_LIBS"] = os.pathsep.join([*libs, *filter(None, [env.get("R_LIBS")])])
    return env


def write_input(name: str, obj, tmp: Path) -> dict:
    """Write one input for R; returns its description in the job."""
    import geopandas as gpd
    import xarray as xr

    if isinstance(obj, gpd.GeoDataFrame):
        path = tmp / f"vector_{name}.gpkg"
        obj.to_file(path, driver="GPKG")
        return {"kind": "vector", "file": path.as_posix()}
    if isinstance(obj, xr.DataArray):
        import landschaft as ls

        path = tmp / f"raster_{name}.tif"
        data = obj if "band" in obj.dims else obj.expand_dims(band=[name])
        ls.write_raster(data.astype("float64"), path, nodata=np.nan, compress="none")
        return {"kind": "raster", "file": path.as_posix()}
    if isinstance(obj, pd.DataFrame):
        columns = []
        for k, col in enumerate(obj.columns):
            s = obj[col]
            if pd.api.types.is_bool_dtype(s):
                path = tmp / f"table_{name}_{k}.bin"
                s.astype("float64").to_numpy("<f8").tofile(path)
                columns.append({"name": str(col), "type": "logical", "file": path.as_posix()})
            elif pd.api.types.is_numeric_dtype(s):
                path = tmp / f"table_{name}_{k}.bin"
                s.astype("float64").to_numpy("<f8").tofile(path)
                columns.append({"name": str(col), "type": "numeric", "file": path.as_posix()})
            else:
                values = [None if pd.isna(v) else str(v) for v in s]
                columns.append({"name": str(col), "type": "text", "values": values})
        return {"kind": "table", "nrow": len(obj), "columns": columns}
    raise TypeError(f"cannot hand {type(obj).__name__} to R ({name})")


def run_group(group: str, only: list[str] | None = None) -> None:
    cases = {k: c for k, c in sdm_cases.CASES.items() if c.group == group and (not only or k in only)}
    if not cases:
        return
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    names = sorted({n for c in cases.values() for n in c.inputs})
    with tempfile.TemporaryDirectory(prefix="sdm_ref_") as tmp:
        tmp = Path(tmp)
        inputs = {n: write_input(n, sdm_cases.build_input(n), tmp) for n in names}
        job = {"inputs": inputs, "cases": [{"name": k, "code": c.r} for k, c in cases.items()]}
        (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
        out = tmp / "out"
        out.mkdir()
        subprocess.run([str(rscript), str(R_SCRIPT), str(tmp / "job.json"), str(out)], check=True,
                       env=r_environment())
        result = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}
        for item in result["manifest"]:
            key = f"{item['case']}/{item['key']}"
            info = {k: v for k, v in item.items() if k not in ("case", "key", "file", "n")}
            if item["kind"] != "text":
                arrays[key] = np.fromfile(out / item["file"], dtype="<f8")
            meta[key] = info
    OUT.mkdir(parents=True, exist_ok=True)
    npz, js = OUT / f"{group}.npz", OUT / f"{group}.json"
    if only and npz.exists():  # keep the other cases
        with np.load(npz) as old:
            kept = {k: old[k] for k in old.files if k.split("/", 1)[0] not in cases}
        kept_meta = {k: v for k, v in json.loads(js.read_text(encoding="utf-8"))["arrays"].items()
                     if k.split("/", 1)[0] not in cases}
        arrays = {**kept, **arrays}
        meta = {**kept_meta, **meta}
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps({"versions": result["versions"], "r": result["r"], "arrays": meta}, indent=1,
                             sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(f"{group}: {len(cases)} cases, {len(meta)} results -> {npz.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", default=sdm_cases.GROUPS)
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()
    for group in args.groups:
        run_group(group, args.case)


if __name__ == "__main__":
    main()
