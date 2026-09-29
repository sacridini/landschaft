# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of the R package gdistance for ls.cost.

Usage::

    python tools/make_reference_cost.py [group ...] [--case NAME ...]

Runs the cases of ``tests/cost_cases.py`` (all groups by default) through
``tools/cost_reference.R`` and stores, per group, the results in
``tests/data/reference/cost/<group>.npz`` (``<case>/<key>`` arrays; a
transition as ``<key>/i``, ``/j``, ``/x``: its matrix in triplets, 0-based, both
triangles) and their kinds in ``<group>.json``. ``--case`` regenerates only the
named cases of the group (the others are kept).

Needs Rscript (``LANDSCHAFT_RSCRIPT`` as in ``tests/lsm_reference.py``) with
gdistance (and raster, igraph, jsonlite) in the default library. The
test-suite reads the stored files, so it runs anywhere without R.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import cost_cases  # noqa: E402
import lsm_reference  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference" / "cost"
R_SCRIPT = ROOT / "tools" / "cost_reference.R"


def run_group(group: str, only: list[str] | None = None) -> None:
    cases = {k: c for k, c in cost_cases.CASES.items() if c.group == group and (not only or k in only)}
    if not cases:
        return
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    names = sorted({c.raster for c in cases.values() if c.raster} | set(cost_cases.EXTRA_RASTERS.get(group, ())))
    with tempfile.TemporaryDirectory(prefix="cost_ref_") as tmp:
        tmp = Path(tmp)
        rasters = {}
        for name in names:
            spec = cost_cases.RASTERS[name]
            v = np.asarray(spec.values, dtype="<f8")
            layers = 1 if v.ndim == 2 else v.shape[0]
            path = tmp / f"raster_{name}.bin"
            v.tofile(path)
            xmin, xmax, ymin, ymax = spec.extent
            rasters[name] = {"file": path.as_posix(), "nrows": v.shape[-2], "ncols": v.shape[-1], "nlayers": layers,
                             "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "crs": spec.crs}
        job = {"rasters": rasters, "cases": [{"name": k, "raster": c.raster, "code": c.r} for k, c in cases.items()]}
        (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
        out = tmp / "out"
        out.mkdir()
        subprocess.run([str(rscript), str(R_SCRIPT), str(tmp / "job.json"), str(out)], check=True,
                       env=lsm_reference.r_environment())
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}
        for item in manifest:
            values = np.fromfile(out / item["file"], dtype="<f8")
            key = f"{item['case']}/{item['key']}"
            if item["kind"] == "transition" and key.rsplit("/", 1)[1] in ("i", "j"):
                values = values.astype(np.int32)
            arrays[key] = values
            info = {k: v for k, v in item.items() if k not in ("case", "key", "file", "n")}
            if item["kind"] == "transition":
                meta[key.rsplit("/", 1)[0]] = info
            else:
                meta[key] = info
    OUT.mkdir(parents=True, exist_ok=True)
    npz, js = OUT / f"{group}.npz", OUT / f"{group}.json"
    if only and npz.exists():  # keep the other cases
        with np.load(npz) as old:
            kept = {k: old[k] for k in old.files if k.split("/", 1)[0] not in cases}
        kept_meta = {k: v for k, v in json.loads(js.read_text(encoding="utf-8")).items()
                     if k.split("/", 1)[0] not in cases}
        arrays = {**kept, **arrays}
        meta = {**kept_meta, **meta}
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{group}: {len(cases)} cases, {len(arrays)} arrays -> {npz.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", default=cost_cases.GROUPS)
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()
    for group in args.groups:
        run_group(group, args.case)


if __name__ == "__main__":
    main()
