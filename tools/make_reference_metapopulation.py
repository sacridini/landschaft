# SPDX-License-Identifier: GPL-3.0-or-later
"""What R's MetaLandSim (2.0) gives, for landschaft's tests (tests/test_dispersal.py), written to
tests/data/reference/metapopulation/metapopulation.npz:

    python tools/make_reference_metapopulation.py

60 habitat patches (x, y, area) with an initial occupancy, and MetaLandSim's spom (Hanski's incidence function
model: the connectivity of every patch to the occupied ones, the colonisation of an empty patch, the extinction of an
occupied one with or without the rescue effect) iterated 40 steps from set.seed(seed), the occupancy after every
step, for four combinations of the kernel, the connectivity, the colonisation and the extinction. Needs R with
MetaLandSim (Rscript on the PATH or in C:/Program Files/R).
"""

from __future__ import annotations

import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "data" / "reference" / "metapopulation"

CASES = {
    # name: kern, conn, colnz, ext, alpha, x, y, e, beta1, b, c1, c2, z, R, seed
    "hanski": ("op1", "op1", "op1", "op1", 0.08, 0.5, 8.0, 0.4, 1, 1, 0, 0, 0, 0, 3),
    "rescue": ("op1", "op2", "op2", "op3", 0.05, 0.8, 0.05, 0.6, 1, 0.5, 0.3, 0, 0, 2, 5),
    "power": ("op2", "op1", "op3", "op2", 0.05, 1.0, 1.0, 0.4, 1.5, 1, 0, 0.05, 2, 0, 9),
    "dense": ("op1", "op1", "op1", "op1", 0.2, 0.3, 3.0, 0.3, 1, 1, 0, 0, 0, 0, 13),
}

R_SCRIPT = r"""
suppressMessages(library(MetaLandSim))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1])
cases <- read.csv(args[2], stringsAsFactors = FALSE)
steps <- as.integer(args[4])
out <- file(args[3], "w")
for (k in seq_len(nrow(cases))) {
  cs <- cases[k, ]
  nodes <- data.frame(x = d$x, y = d$y, areas = d$area, ID = seq_len(nrow(d)), species = d$occupied)
  sp <- list(mapsize = 100, minimum.distance = 1, mean.area = mean(d$area), SD.area = sd(d$area),
             number.patches = nrow(d), dispersal = 1, distance.to.neighbours = NULL, nodes.characteristics = nodes)
  class(sp) <- "metapopulation"
  param <- data.frame(p = c(cs$alpha, cs$x, cs$y, cs$e))
  set.seed(cs$seed)
  for (t in seq_len(steps)) {
    r <- spom(sp, kern = cs$kern, conn = cs$conn, colnz = cs$colnz, ext = cs$ext, param_df = param,
              beta1 = cs$beta1, b = cs$b, c1 = cs$c1, c2 = cs$c2, z = cs$z, R = cs$R)
    s2 <- as.numeric(r$nodes.characteristics$species2)
    cat(cs$name, t, paste(s2, collapse = " "), "\n", file = out)
    sp$nodes.characteristics$species <- s2
  }
}
close(out)
"""


def rscript() -> str:
    found = shutil.which("Rscript")
    if found:
        return found
    for p in sorted(glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"), reverse=True):
        return p
    raise SystemExit("Rscript not found")


def main() -> None:
    g = np.random.default_rng(17)
    n = 60
    x, y = g.random(n) * 100, g.random(n) * 100
    area = np.round(np.exp(g.normal(0.5, 0.8, n)), 4)
    occupied = (g.random(n) < 0.4).astype(float)
    arrays: dict[str, np.ndarray] = {"patches": np.c_[x, y, area, occupied]}
    steps = 40
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        np.savetxt(t / "d.csv", np.c_[x, y, area, occupied], delimiter=",", header="x,y,area,occupied", comments="",
                   fmt="%.17g")
        cols = "name,kern,conn,colnz,ext,alpha,x,y,e,beta1,b,c1,c2,z,R,seed"
        rows = [",".join([name, *(str(v) for v in c)]) for name, c in CASES.items()]
        (t / "cases.csv").write_text(cols + "\n" + "\n".join(rows) + "\n")
        (t / "s.R").write_text(R_SCRIPT)
        subprocess.run([rscript(), str(t / "s.R"), str(t / "d.csv"), str(t / "cases.csv"), str(t / "out.txt"),
                        str(steps)], check=True)
        runs: dict[str, list] = {}
        for line in (t / "out.txt").read_text().splitlines():
            if line.strip():
                name, _, *vals = line.split()
                runs.setdefault(name, []).append([float(v) for v in vals])
    for name, c in CASES.items():
        arrays[f"{name}/occupancy"] = np.array(runs[name])
        arrays[f"{name}/settings"] = np.array([float(v) for v in c[4:]])
        print(name, arrays[f"{name}/occupancy"].sum(axis=1)[[0, 9, 19, 39]])
    REF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF / "metapopulation.npz", **arrays)
    print(REF / "metapopulation.npz", len(arrays))


if __name__ == "__main__":
    main()
