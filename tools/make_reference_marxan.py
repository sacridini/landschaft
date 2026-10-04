# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference runs of Marxan 4.0.6 for tests/test_prioritize_marxan.py.

    python tools/make_reference_marxan.py [--case NAME ...]

Writes every case of ``tests/marxan_cases.py`` as Marxan's input files, runs the
Marxan executable built with gcc in WSL (Ubuntu) and stores every run's
solution, the summary and the penalties in
``tests/data/reference/optimize/marxan.npz`` (``<case>/<key>`` arrays) and
``marxan.json``. Two builds, from ``C:/Users/eduar/Documents/dev/marxan``
(github.com/Marxan-source-code/marxan, MIT):

* ``~/marxan-src/bin/marxan``: as released (``make``);
* ``~/marxan-fixed/bin/marxan``: with the correction of ``src/cpp/opt/marxan.hpp``
  (``SpeciesAmounts`` after ``initialiseReserve`` in the Connolly and adaptive
  initialisations; ``readSpecies`` sets ``amount`` and ``occurrence`` to 0) and
  the summary and penalties written with 17 digits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import marxan_cases  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference" / "optimize"
EXE = {"original": "~/marxan-src/bin/marxan", "fixed": "~/marxan-fixed/bin/marxan"}
OPTION_NAMES = {"blm": "BLM", "prop": "PROP", "seed": "RANDSEED", "num_reps": "NUMREPS", "num_iters": "NUMITNS",
                "start_temp": "STARTTEMP", "cool_fac": "COOLFAC", "num_temp": "NUMTEMP", "cost_thresh": "COSTTHRESH",
                "thresh_pen1": "THRESHPEN1", "thresh_pen2": "THRESHPEN2", "run_mode": "RUNMODE",
                "misslevel": "MISSLEVEL", "heur_type": "HEURTYPE", "itimp_type": "ITIMPTYPE",
                "connectivity_in": "CONNECTIVITYIN"}
DEFAULTS = {"run_mode": 1, "heur_type": 1, "itimp_type": 0, "misslevel": 1.0, "prop": 0.0, "cool_fac": 0.0}


def _wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return "/mnt/" + s[0].lower() + s[2:]


def _write(t: pd.DataFrame, path: Path) -> None:
    t.to_csv(path, index=False, float_format="%.17g")


def run_case(name: str, case) -> dict:
    pu, spec, puvspr, bound = marxan_cases.tables(case.status, case.spec)
    opts = {**DEFAULTS, **case.options}
    with tempfile.TemporaryDirectory(prefix="marxan_ref_") as tmp:
        d = Path(tmp)
        (d / "input").mkdir()
        (d / "output").mkdir()
        _write(pu, d / "input" / "pu.dat")
        _write(spec, d / "input" / "spec.dat")
        _write(puvspr, d / "input" / "puvspr.dat")
        _write(bound, d / "input" / "bound.dat")
        lines = [f"{OPTION_NAMES[k]} {int(v) if isinstance(v, bool) else v}" for k, v in opts.items()]
        lines += ["INPUTDIR input", "OUTPUTDIR output", "PUNAME pu.dat", "SPECNAME spec.dat", "PUVSPRNAME puvspr.dat",
                  "BOUNDNAME bound.dat", "SCENNAME output", "SAVERUN 3", "SAVESUMMARY 3", "SAVEPENALTY 3",
                  "SAVEBEST 3", "VERBOSITY 1"]
        (d / "input.dat").write_text("\n".join(lines) + "\n")
        cmd = f"cd '{_wsl_path(d)}' && OMP_NUM_THREADS=2 {EXE[case.exe]} input.dat"
        subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], check=True, capture_output=True)
        out = d / "output"
        reps = int(opts["num_reps"])
        sols = []
        for r in range(1, reps + 1):
            s = pd.read_csv(out / f"output_r{r:05d}.csv", float_precision="round_trip")
            sols.append(s.iloc[:, 1].to_numpy(dtype=np.int64))
        summary = pd.read_csv(out / "output_sum.csv", float_precision="round_trip")
        penalty = pd.read_csv(out / "output_penalty.csv", float_precision="round_trip")
    res = {"solutions": np.stack(sols), "penalty_spid": penalty.iloc[:, 0].to_numpy(dtype=np.float64),
           "penalty": penalty.iloc[:, 1].to_numpy(dtype=np.float64)}
    for col in ("Score", "Cost", "Planning_Units", "Connectivity", "Penalty", "Shortfall", "Missing_Values", "MPM"):
        res[col] = summary[col].to_numpy(dtype=np.float64)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append")
    args = ap.parse_args()
    npz, js = OUT / "marxan.npz", OUT / "marxan.json"
    arrays = dict(np.load(npz)) if npz.exists() else {}
    meta = json.loads(js.read_text()) if js.exists() else {}
    for name, case in marxan_cases.CASES.items():
        if args.case and name not in args.case:
            continue
        res = run_case(name, case)
        for k, v in res.items():
            arrays[f"{name}/{k}"] = v
        meta[name] = {"exe": case.exe, "notes": case.notes, "options": case.options}
        print(name, "runs", res["solutions"].shape[0], "scores", res["Score"])
    meta["_marxan"] = "Marxan 4.0.6 (github.com/Marxan-source-code/marxan 806a48e), g++ 13.3, WSL Ubuntu 24.04"
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
