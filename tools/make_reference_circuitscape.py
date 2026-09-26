# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of Circuitscape 6.1's raster modes (Julia).

Usage::

    python tools/make_reference_circuitscape.py

Julia is found as in ``tools/make_reference_omniscape.py`` (``LANDSCHAFT_JULIA``,
``LANDSCHAFT_JULIA_PROJECT``; the environment needs Circuitscape, ArchGDAL and
NPZ). ``LANDSCHAFT_CIRCUITSCAPE_TEST`` is Circuitscape's ``test`` directory
(default: the installed package's, found through Julia).

Circuitscape's raster test inputs (``test/input/raster``: pairwise, advanced,
one-to-all and all-to-one, MIT, from Circuitscape.jl) are copied to
``tests/data/reference/circuit/circuitscape/input/raster``; every INI file there
runs through ``Circuitscape.compute`` with the ``cholmod`` solver
(``tools/circuitscape_reference.jl``) and every output (voltage and current maps,
cumulative and maximum current maps, resistances, the returned matrix) is stored
in ``tests/data/reference/circuit/circuitscape.npz`` as ``<ini name>/<output>``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "reference" / "circuit"
SCRIPT = ROOT / "tools" / "circuitscape_reference.jl"
SCENARIOS = ("advanced", "pairwise", "one_to_all", "all_to_one")


def julia_cmd() -> list[str]:
    cmd = [os.environ.get("LANDSCHAFT_JULIA", "julia")]
    project = os.environ.get("LANDSCHAFT_JULIA_PROJECT")
    if project:
        cmd.append(f"--project={project}")
    return cmd


def circuitscape_test_dir() -> Path:
    env = os.environ.get("LANDSCHAFT_CIRCUITSCAPE_TEST")
    if env:
        return Path(env)
    out = subprocess.run([*julia_cmd(), "-e", "using Circuitscape; print(pkgdir(Circuitscape))"], check=True,
                         capture_output=True, text=True).stdout.strip()
    return Path(out) / "test"


def _read_output(path: Path) -> np.ndarray:
    from landschaft.circuit.circuitscape import read_raster

    if path.suffix in (".asc", ".tif"):
        return read_raster(path)[0]
    return np.loadtxt(path, ndmin=2)


def main() -> None:
    src = circuitscape_test_dir() / "input" / "raster"
    dst = OUT / "circuitscape" / "input" / "raster"
    if dst.exists():
        # Julia's package files are read-only; so were the copies made from them before.
        shutil.rmtree(dst, onexc=lambda f, p, e: (os.chmod(p, 0o666), f(p)))
    shutil.copytree(src, dst, copy_function=shutil.copyfile)
    inis = sorted(p.relative_to(OUT / "circuitscape").as_posix() for s in SCENARIOS
                  for p in (dst / s).glob("*/*.ini"))
    tmp = Path(tempfile.mkdtemp(prefix="landschaft_circuitscape_"))
    try:
        subprocess.run([*julia_cmd(), str(SCRIPT), str(OUT / "circuitscape"), str(tmp), *inis], check=True)
        store: dict[str, np.ndarray] = {}
        for f in sorted(tmp.iterdir()):
            if f.suffix == ".out" and "_resistances" not in f.name:
                continue  # the configuration
            if f.suffix in (".prj", ".log"):
                continue
            name = f.stem
            for ini in inis:
                base = Path(ini).stem
                if name.startswith(base + "_") or name == base:
                    key = f"{base}/{name[len(base) + 1:]}{f.suffix}"
                    store[key] = _read_output(f)
                    break
        np.savez_compressed(OUT / "circuitscape.npz", **store)
        print(f"{len(store)} outputs of {len(inis)} runs")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
