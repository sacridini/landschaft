# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of the reference land change tools behind ls.simulate.

Usage::

    python tools/make_reference_simulate.py [group ...] [--case NAME ...]

Runs the cases of ``tests/simulate_cases.py`` (all groups by default) on
their engines and stores, per group, the numeric results in
``tests/data/reference/simulate/<group>.npz`` (``<case>/<key>`` arrays) and
their kinds, shapes and the text results in ``<group>.json`` (with the tool
versions). ``--case`` regenerates only the named cases of the group (the
others are kept).

Engines: R (lulcc, OpenLand, diffeR through ``tools/simulate_reference.R``;
Rscript as in ``tests/lsm_reference.py``, the packages in
``LANDSCHAFT_SIM_R_LIBS``, default ``%LOCALAPPDATA%/R/landschaft-sim-lib``,
before the user library), CLUMondo, GRASS / r.futures in WSL, Dinamica EGO
and MOLUSCE (``tools/simulate_engines.py``). The test-suite reads the stored
files, so it runs anywhere without these tools.
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
import simulate_cases
import simulate_engines

OUT = ROOT / "tests" / "data" / "reference" / "simulate"
R_SCRIPT = ROOT / "tools" / "simulate_reference.R"
DEFAULT_LIBS = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "R" / "landschaft-sim-lib"
USER_LIBS = sorted(DEFAULT_LIBS.parent.glob("win-library/*"))


def r_environment() -> dict[str, str]:
    env = lsm_reference.r_environment()
    libs = [env.get("LANDSCHAFT_SIM_R_LIBS", str(DEFAULT_LIBS)), *(str(p) for p in USER_LIBS[-1:])]
    env["R_LIBS"] = os.pathsep.join([*libs, *filter(None, [env.get("R_LIBS")])])
    return env


def run_r(cases: dict) -> tuple[dict, dict, dict]:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    names = sorted({n for c in cases.values() for n in c.inputs})
    with tempfile.TemporaryDirectory(prefix="sim_ref_") as tmp:
        tmp = Path(tmp)
        inputs = {n: make_reference_sdm.write_input(n, simulate_cases.build_input(n), tmp) for n in names}
        job = {"inputs": inputs, "tools": (ROOT / "tools").as_posix(),
               "cases": [{"name": k, "code": c.r} for k, c in cases.items()]}
        (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
        out = tmp / "out"
        out.mkdir()
        subprocess.run([str(rscript), str(R_SCRIPT), str(tmp / "job.json"), str(out)], check=True,
                       env=r_environment(), cwd=tmp)  # plots R makes (Rplots.pdf) stay in tmp
        result = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}
        for item in result["manifest"]:
            key = f"{item['case']}/{item['key']}"
            info = {k: v for k, v in item.items() if k not in ("case", "key", "file", "n")}
            if item["kind"] != "text":
                arrays[key] = np.fromfile(out / item["file"], dtype="<f8")
            meta[key] = info
    return arrays, meta, {"r": {"versions": result["versions"], "r": result["r"]}}


def store(case: str, result: dict, arrays: dict, meta: dict) -> None:
    for name, value in result.items():
        key = f"{case}/{name}"
        if isinstance(value, list):
            meta[key] = {"kind": "text", "values": [str(v) for v in value]}
            continue
        v = np.asarray(value, dtype=np.float64)
        arrays[key] = v.ravel()
        meta[key] = {"kind": "matrix", "dim": list(v.shape)} if v.ndim > 1 else {"kind": "vector"}


def engine_versions(engine: str) -> dict:
    if engine == "clumondo":
        return {"clumondo": "github.com/VUEG/CLUMondo a5c0c1f (Win32, MSVC)"}
    if engine == "dinamica":
        return {"dinamica": simulate_engines.dinamica_version()}
    if engine == "molusce":
        return {"molusce": simulate_engines.molusce_version()}
    if engine in ("grass", "wsl_python"):
        proc = simulate_engines._wsl("grass --version 2>/dev/null | head -1", 120)
        return {"grass": proc.stdout.strip(), "addons": "r.futures (OSGeo/grass-addons, grass8 branch)"}
    return {}


def run_group(group: str, only: list[str] | None = None) -> None:
    cases = {k: c for k, c in simulate_cases.CASES.items() if c.group == group and (not only or k in only)}
    if not cases:
        return
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    versions: dict[str, dict] = {}
    r_cases = {k: c for k, c in cases.items() if c.engine == "r"}
    if r_cases:
        a, m, v = run_r(r_cases)
        arrays.update(a)
        meta.update(m)
        versions.update(v)
    for name, case in cases.items():
        if case.engine == "r":
            continue
        run = simulate_engines.ENGINES[case.engine]
        with tempfile.TemporaryDirectory(prefix=f"sim_{case.engine}_") as tmp:
            job = case.job(**{n: simulate_cases.build_input(n) for n in case.inputs})
            store(name, run(job, Path(tmp)), arrays, meta)
        versions.setdefault(case.engine, engine_versions(case.engine))
    OUT.mkdir(parents=True, exist_ok=True)
    npz, js = OUT / f"{group}.npz", OUT / f"{group}.json"
    if only and npz.exists():  # keep the other cases
        with np.load(npz) as old:
            kept = {k: old[k] for k in old.files if k.split("/", 1)[0] not in cases}
        old_info = json.loads(js.read_text(encoding="utf-8"))
        kept_meta = {k: v for k, v in old_info["arrays"].items() if k.split("/", 1)[0] not in cases}
        arrays = {**kept, **arrays}
        meta = {**kept_meta, **meta}
        versions = {**old_info.get("versions", {}), **versions}
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps({"versions": versions, "arrays": meta}, indent=1, sort_keys=True, ensure_ascii=False),
                  encoding="utf-8")
    print(f"{group}: {len(cases)} cases, {len(meta)} results -> {npz.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", default=simulate_cases.GROUPS)
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()
    for group in args.groups:
        run_group(group, args.case)


if __name__ == "__main__":
    main()
