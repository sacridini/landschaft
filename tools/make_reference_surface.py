# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of the R packages geodiv and spatialEco for ls.surface.

Usage::

    python tools/make_reference_surface.py [group ...] [--case NAME ...]

Runs the cases of ``tests/surface_cases.py`` (all groups by default) through
``tools/surface_reference.R`` and stores, per group, the results in
``tests/data/reference/surface/<group>.npz`` (``<case>/<key>`` arrays) and
their kinds and shapes in ``<group>.json`` (with the package versions).
``--case`` regenerates only the named cases of the group (the others are kept).

Needs Rscript (``LANDSCHAFT_RSCRIPT`` as in ``tests/lsm_reference.py``) and a
library with geodiv 1.1.1 and spatialEco 2.0-5 (``LANDSCHAFT_SURF_R_LIBS``,
default ``%LOCALAPPDATA%/R/landschaft-surf-lib``, put before the default
libraries, which provide terra, sf and jsonlite). The test-suite reads the
stored files, so it runs anywhere without R.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import lsm_reference  # noqa: E402
import surface_cases  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference" / "surface"
R_SCRIPT = ROOT / "tools" / "surface_reference.R"
DEFAULT_LIBS = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "R" / "landschaft-surf-lib"
# The user library (terra, sf, jsonlite), as R_LIBS replaces it.
USER_LIBS = sorted(DEFAULT_LIBS.parent.glob("win-library/*"))


def r_environment() -> dict[str, str]:
    env = lsm_reference.r_environment()
    libs = [env.get("LANDSCHAFT_SURF_R_LIBS", str(DEFAULT_LIBS)), *(str(p) for p in USER_LIBS[-1:])]
    env["R_LIBS"] = os.pathsep.join([*libs, *filter(None, [env.get("R_LIBS")])])
    return env


def run_group(group: str, only: list[str] | None = None) -> None:
    cases = {k: c for k, c in surface_cases.CASES.items() if c.group == group and (not only or k in only)}
    if not cases:
        return
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    names = sorted({c.raster for c in cases.values() if c.raster} | set(surface_cases.EXTRA_RASTERS.get(group, ())))
    with tempfile.TemporaryDirectory(prefix="surface_ref_") as tmp:
        tmp = Path(tmp)
        rasters = {}
        for name in names:
            spec = surface_cases.RASTERS[name]
            v = np.asarray(spec.values, dtype="<f8")
            path = tmp / f"raster_{name}.bin"
            v.tofile(path)
            xmin, xmax, ymin, ymax = spec.extent
            rasters[name] = {"file": path.as_posix(), "nrows": v.shape[0], "ncols": v.shape[1],
                             "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "crs": spec.crs}
        job = {"rasters": rasters, "cases": [{"name": k, "raster": c.raster, "code": c.r} for k, c in cases.items()]}
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
            arrays[key] = np.fromfile(out / item["file"], dtype="<f8")
            meta[key] = {k: v for k, v in item.items() if k not in ("case", "key", "file", "n")}
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
                             sort_keys=True), encoding="utf-8")
    print(f"{group}: {len(cases)} cases, {len(arrays)} arrays -> {npz.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", default=surface_cases.GROUPS)
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()
    for group in args.groups:
        run_group(group, args.case)


if __name__ == "__main__":
    main()
