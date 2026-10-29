# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of R's move2 and amt for ls.move.

Usage::

    python tools/make_reference_move.py [--case NAME ...]

Draws the track sets of ``tests/move_cases.py``, runs the cases through
``tools/move_reference.R`` (with ``tools/move2_fixed.R`` and
``tools/amt_fixed.R``, the packages' corrected functions) and stores the
inputs in ``tests/data/reference/move/inputs.npz`` and the results in
``tests/data/reference/move/<group>.npz`` (``move2.npz``, ``amt.npz``; keys
``<case>/<key>``).
``--case`` regenerates only the named cases (the others are kept).

Needs Rscript (``LANDSCHAFT_RSCRIPT`` as in ``tests/lsm_reference.py``) and the
library ``%LOCALAPPDATA%/R/landschaft-move-lib`` (``LANDSCHAFT_MOVE_R_LIBS``) with
move2 0.5.0, sf, s2, lwgeom, units, amt 0.3.1, fitdistrplus 1.2-6, circular
0.5-2, geosphere 1.6-8, suncalc 0.5.1 and terra. Runs R with ``TZ=UTC`` (suncalc
takes the noon of the machine's time zone). The test-suite reads the stored files,
so it runs anywhere without R.
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
import move_cases  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference" / "move"
R_SCRIPT = ROOT / "tools" / "move_reference.R"
DEFAULT_LIBS = Path(os.environ.get("LOCALAPPDATA", "")) / "R" / "landschaft-move-lib"


def r_env() -> dict[str, str]:
    env = lsm_reference.r_environment()
    libs = [env.get("LANDSCHAFT_MOVE_R_LIBS", str(DEFAULT_LIBS))]
    user = sorted(DEFAULT_LIBS.parent.glob("win-library/*"))
    libs += [str(p) for p in user[-1:]]
    env["R_LIBS"] = os.pathsep.join([*libs, *filter(None, [env.get("R_LIBS")])])
    env["TOOLS"] = str(ROOT / "tools")
    env["OMP_NUM_THREADS"] = "1"
    env["TZ"] = "UTC"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()
    cases = {k: c for k, c in move_cases.CASES.items() if not args.case or k in args.case}
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    sets = move_cases.track_sets()
    inputs: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="move_ref_") as tmp:
        tmp = Path(tmp)
        job_sets = {}
        for name, s in sets.items():
            d = tmp / name
            d.mkdir()
            extra = [k for k in s if k not in ("x", "y", "t", "track", "crs")]
            for key in ("x", "y", "t", *extra):
                np.asarray(s[key], "<f8").tofile(d / f"{key}.bin")
                inputs[f"{name}/{key}"] = np.asarray(s[key], np.float64)
            inputs[f"{name}/track"] = np.asarray(s["track"]).astype("U")
            inputs[f"{name}/crs"] = np.array(s["crs"] or "")
            job_sets[name] = {"dir": d.as_posix(), "n": len(s["x"]), "crs": s["crs"],
                              "track": [str(v) for v in s["track"]], "extra": extra}
        job = {"sets": job_sets, "cases": [{"name": k, "data": c.data, "code": c.r} for k, c in cases.items()]}
        (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
        out = tmp / "out"
        out.mkdir()
        subprocess.run([str(rscript), str(R_SCRIPT), str(tmp / "job.json"), str(out)], check=True, env=r_env())
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        arrays = {f"{m['case']}/{m['key']}": np.fromfile(out / m["file"], "<f8") for m in manifest}
    OUT.mkdir(parents=True, exist_ok=True)
    for group in sorted({c.group for c in cases.values()}):
        mine = {k: v for k, v in arrays.items() if cases[k.split("/", 1)[0]].group == group}
        npz = OUT / f"{group}.npz"
        if args.case and npz.exists():
            with np.load(npz) as old:
                mine = {**{k: old[k] for k in old.files if k.split("/", 1)[0] not in cases}, **mine}
        np.savez_compressed(npz, **mine)
        print(f"{group}: {len(mine)} arrays -> {npz.relative_to(ROOT)}")
    np.savez_compressed(OUT / "inputs.npz", **inputs)


if __name__ == "__main__":
    main()
