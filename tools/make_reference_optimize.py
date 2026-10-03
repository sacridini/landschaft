# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of prioritizr and prioriactions for the MILP prioritization.

Usage::

    python tools/make_reference_optimize.py [group ...] [--case NAME ...]

Runs the cases of ``tests/optimize_cases.py`` (all groups by default) in R
through ``tools/optimize_reference.R`` and stores, per group, the numeric
results in ``tests/data/reference/optimize/<group>.npz`` (``<case>/<key>``
arrays) and their kinds, shapes and the text results in ``<group>.json``
(with the package versions). ``--case`` regenerates only the named cases of
the group (the others are kept).

Needs Rscript (``LANDSCHAFT_RSCRIPT`` as in ``tests/lsm_reference.py``) with
prioritizr, prioriactions, the gurobi R package (a Gurobi licence),
exactextractr and sf; ``LANDSCHAFT_OPT_R_LIBS`` is put before the default
libraries. The test-suite reads the stored files, so it runs anywhere
without R or Gurobi.
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
sys.path.insert(0, str(ROOT / "tools"))

import lsm_reference
import make_reference_sdm
import optimize_cases

OUT = ROOT / "tests" / "data" / "reference" / "optimize"
R_SCRIPT = ROOT / "tools" / "optimize_reference.R"


def r_environment() -> dict[str, str]:
    env = dict(os.environ)
    if env.get("LANDSCHAFT_OPT_R_LIBS"):
        env["R_LIBS"] = os.pathsep.join(filter(None, [env["LANDSCHAFT_OPT_R_LIBS"], env.get("R_LIBS")]))
    return env


def run_r(cases: dict) -> tuple[dict, dict, dict]:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    names = sorted({n for c in cases.values() for n in c.inputs})
    with tempfile.TemporaryDirectory(prefix="opt_ref_") as tmp:
        tmp = Path(tmp)
        inputs = {n: make_reference_sdm.write_input(n, optimize_cases.build_input(n), tmp) for n in names}
        job = {"inputs": inputs, "cases": [{"name": k, "code": c.r} for k, c in cases.items()]}
        (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
        out = tmp / "out"
        out.mkdir()
        subprocess.run([str(rscript), str(R_SCRIPT), str(tmp / "job.json"), str(out)], check=True,
                       env=r_environment(), cwd=tmp)
        result = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}
        for item in result["manifest"]:
            key = f"{item['case']}/{item['key']}"
            info = {k: v for k, v in item.items() if k not in ("case", "key", "file", "n")}
            if item["kind"] != "text":
                arrays[key] = np.fromfile(out / item["file"], dtype="<f8")
            meta[key] = info
    return arrays, meta, {"r": result["versions"], "r_version": result["r"]}


def run_group(group: str, only: list[str] | None = None) -> None:
    cases = {k: c for k, c in optimize_cases.CASES.items() if c.group == group and (not only or k in only)}
    if not cases:
        return
    arrays, meta, versions = run_r(cases)
    OUT.mkdir(parents=True, exist_ok=True)
    npz, js = OUT / f"{group}.npz", OUT / f"{group}.json"
    if only and npz.exists():  # keep the other cases
        with np.load(npz) as old:
            kept = {k: old[k] for k in old.files if k.split("/", 1)[0] not in cases}
        old_info = json.loads(js.read_text(encoding="utf-8"))
        kept_meta = {k: v for k, v in old_info["arrays"].items() if k.split("/", 1)[0] not in cases}
        arrays = {**kept, **arrays}
        meta = {**kept_meta, **meta}
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps({"versions": versions, "arrays": meta}, indent=1, sort_keys=True, ensure_ascii=False),
                  encoding="utf-8")
    print(f"{group}: {len(cases)} cases, {len(meta)} results -> {npz.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", default=optimize_cases.GROUPS)
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()
    for group in args.groups:
        run_group(group, args.case)


if __name__ == "__main__":
    main()
