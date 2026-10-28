# SPDX-License-Identifier: GPL-3.0-or-later
"""What MigClim (1.6.2, CRAN's archive) gives, for landschaft's tests (tests/test_dispersal.py), written to
tests/data/reference/dispersal/dispersal.npz:

    python tools/make_reference_dispersal.py

MigClim's C (migrate.c, src_cell.c, barriers.c, univ_disp.c, file_io.c) built as a program with gcc (Rtools on
Windows, its C runtime's rand), its seed taken from the environment instead of the clock (`srand(time(NULL))`, the
only change), run on a landscape of 40 x 50 cells: an initial distribution, three maps of habitat suitability (0 to
1000) that move east, cells without data, and a wall of barrier cells. Three simulations: weak barriers with
long-distance dispersal; strong barriers, a threshold of suitability and a ramp of propagule production; no
barriers and long-distance dispersal every step. Every dispersal step's state, the final state and the statistics.
"""

from __future__ import annotations

import glob
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "data" / "reference" / "dispersal"
URL = "https://cran.r-project.org/src/contrib/Archive/MigClim/MigClim_1.6.2.tar.gz"
SOURCES = ["migrate.c", "src_cell.c", "barriers.c", "univ_disp.c", "file_io.c"]

MAIN = r"""
#include "migclim.h"
int nrRows, nrCols, noData;
double xllCorner, yllCorner, cellSize;
int main(int argc, char **argv) {
  int n = 0;
  mcMigrate(&argv[1], &n);
  return n < 0;
}
"""

CASES = {
    "weak": {"seed": 7, "barrier": "weak", "rcThreshold": 0, "dispSteps": 5, "kernel": [1.0, 0.4, 0.16, 0.06], "iniMatAge": 1,
                 "fullMatAge": 1, "propaguleProd": [1.0], "lddFreq": 0.1, "lddMinDist": 6, "lddMaxDist": 15},
    "strong": {"seed": 11, "barrier": "strong", "rcThreshold": 250, "dispSteps": 4, "kernel": [0.9, 0.6, 0.3], "iniMatAge": 1,
                   "fullMatAge": 4, "propaguleProd": [0.3, 0.6, 0.9], "lddFreq": 0.0, "lddMinDist": 10, "lddMaxDist": 15},
    "open": {"seed": 3, "barrier": None, "rcThreshold": 0, "dispSteps": 3, "kernel": [0.9, 0.5], "iniMatAge": 2, "fullMatAge": 3,
                 "propaguleProd": [0.6], "lddFreq": 1.0, "lddMinDist": 3, "lddMaxDist": 10},
}


def gcc() -> str:
    found = shutil.which("gcc")
    if found:
        return found
    for p in sorted(glob.glob("C:/rtools4*/x86_64-w64-mingw32.static.posix/bin/gcc.exe"), reverse=True):
        return p
    raise SystemExit("gcc not found")


def build(tmp: Path) -> Path:
    data = urllib.request.urlopen(URL).read()
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        for name in SOURCES + ["migclim.h"]:
            (tmp / name).write_bytes(t.extractfile(f"MigClim/src/{name}").read())
    (tmp / "R.h").write_text("#include <stdio.h>\n#define Rprintf printf\n")
    m = (tmp / "migrate.c").read_text()
    clock = "srand ((unsigned int)time (NULL));"
    assert m.count(clock) == 1
    (tmp / "migrate.c").write_text(m.replace(clock, 'srand ((unsigned int)atoi (getenv ("MIGCLIM_SEED")));'))
    (tmp / "main.c").write_text(MAIN)
    exe = tmp / "migclim.exe"
    subprocess.run([gcc(), "-O2", "-I", str(tmp), "-o", str(exe), *[str(tmp / s) for s in SOURCES + ["main.c"]], "-lm"],
                   check=True)
    return exe


def write_asc(path: Path, a: np.ndarray) -> None:
    ny, nx = a.shape
    head = f"ncols {nx}\nnrows {ny}\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999\n"
    path.write_text(head + "\n".join(" ".join(str(int(v)) for v in row) for row in a) + "\n")


def read_asc(path: Path) -> np.ndarray:
    lines = path.read_text().split("\n")
    return np.array([[float(v) for v in l.split()] for l in lines[6:] if l.strip()])


def landscape() -> dict[str, np.ndarray]:
    ny, nx = 40, 50
    g = np.random.default_rng(5)
    rows, cols = np.mgrid[0:ny, 0:nx]
    out = {}
    for k, centre in enumerate([12, 24, 36], start=1):
        hs = 1000 - np.abs(cols - centre) * 45 - np.abs(rows - 20) * 12 + g.integers(-150, 150, (ny, nx))
        hs = np.clip(hs, 0, 1000)
        hs[0:3, 45:50] = -9999
        out[f"hs{k}"] = hs
    ini = np.zeros((ny, nx))
    ini[18:22, 3:7] = 1
    ini[0:3, 45:50] = -9999
    out["ini"] = ini
    bar = np.zeros((ny, nx))
    bar[5:31, 20] = 1
    bar[0:3, 45:50] = -9999
    out["barrier"] = bar
    return out


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    land = landscape()
    for k, v in land.items():
        arrays[f"land/{k}"] = v.astype(float)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        exe = build(t)
        for k, v in land.items():
            write_asc(t / f"{k}.asc", v)
        for name, c in CASES.items():
            lines = ["nrRows 40", "nrCols 50", "iniDist ini", "hsMap hs"]
            if c["barrier"]:
                lines += ["barrier barrier", f"barrierType {c['barrier']}"]
            lines += ["envChgSteps 3", f"dispSteps {c['dispSteps']}", f"dispDist {len(c['kernel'])}",
                      "dispKernel " + " ".join(str(x) for x in c["kernel"]), f"iniMatAge {c['iniMatAge']}",
                      f"fullMatAge {c['fullMatAge']}", "propaguleProd " + " ".join(str(x) for x in c["propaguleProd"]),
                      f"rcThreshold {c['rcThreshold']}", f"lddFreq {c['lddFreq']}", f"lddMinDist {c['lddMinDist']}",
                      f"lddMaxDist {c['lddMaxDist']}", "fullOutput true", f"simulName {name}"]
            (t / f"{name}_params.txt").write_text("\n".join(lines) + "\n")
            (t / name).mkdir()
            subprocess.run([str(exe), f"{name}_params.txt"], cwd=t, check=True, capture_output=True,
                           env={**os.environ, "MIGCLIM_SEED": str(c["seed"])})
            arrays[f"{name}/final"] = read_asc(t / name / f"{name}_raster.asc")
            stats = np.loadtxt(t / name / f"{name}_stats.txt", skiprows=1, ndmin=2)
            arrays[f"{name}/stats"] = stats
            steps = sorted(t.glob(f"{name}/{name}_step_*.asc"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
            arrays[f"{name}/steps"] = np.stack([read_asc(p) for p in steps])
            print(name, "steps", len(steps), "final occupied", int(stats[-1, 5]))
    REF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF / "dispersal.npz", **arrays)
    print(REF / "dispersal.npz", len(arrays))


if __name__ == "__main__":
    main()
