# SPDX-License-Identifier: GPL-3.0-or-later
"""What numpy's sampling and R's mapaccuracy (0.1.2) give for a sample of a classified map, for landschaft's tests
(the same file as Terraflux's), written to tests/data/reference/accuracy/accuracy.json:

    python tools/make_reference_accuracy.py

A map of four classes (60 x 80 cells of 30 m, a few missing): a stratified random sample drawn as
`ls.accuracy.sample` draws it (numpy's default_rng(seed).choice of each class's cells, class after class), reference
classes made up for it, and a map of three strata of its own. R's `olofsson` (the strata are the map's classes)
and `stehman2014` (the map's classes as strata, and the strata of their own) on it; and `olofsson` on the
published example of Olofsson et al. (2014, table 8). Needs R with mapaccuracy (Rscript on the PATH or in
C:/Program Files/R).
"""

from __future__ import annotations

import glob
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "data" / "reference" / "accuracy"

R_SCRIPT = r"""
library(mapaccuracy)
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1])
nh <- read.csv(args[2])
ns <- read.csv(args[3])
Nh <- setNames(nh$cells, as.character(nh$class))
Ns <- setNames(ns$cells, as.character(ns$stratum))
out <- file(args[4], "w")
put <- function(key, x) {
  x <- unlist(x)
  cat(key, paste(ifelse(is.na(x), "NA", sprintf("%.17g", x)), collapse = " "), "\n", file = out)
}
emit <- function(name, e) {
  put(paste0(name, ".classes"), as.numeric(names(e$UA)))
  for (k in c("OA", "SEoa", "UA", "SEua", "PA", "SEpa", "area", "SEa")) put(paste0(name, ".", k), e[[k]])
  m <- e$matrix
  m <- m[rownames(m) != "sum", colnames(m) != "sum", drop = FALSE]
  put(paste0(name, ".matrix"), t(m))
}
emit("olofsson", olofsson(as.character(d$reference), as.character(d$map), Nh))
emit("stehman_map", stehman2014(as.character(d$map), d$reference, d$map, Nh))
emit("stehman_strata", stehman2014(as.character(d$stratum), d$reference, d$map, Ns))
p <- read.csv(args[5])
pn <- read.csv(args[6])
emit("paper", olofsson(as.character(p$reference), as.character(p$map), setNames(pn$cells, as.character(pn$class))))
close(out)
"""


def rscript() -> str:
    found = shutil.which("Rscript")
    if found:
        return found
    for p in sorted(glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"), reverse=True):
        return p
    raise SystemExit("Rscript not found")


def read_r(path: Path) -> dict:
    out: dict = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, *values = line.split()
        name, what = key.split(".", 1)
        out.setdefault(name, {})[what] = [None if v == "NA" else float(v) for v in values]
    return out


def main() -> None:
    ny, nx = 60, 80
    rng = np.random.default_rng(42)
    land = np.where(rng.random((ny, nx)) < 0.55, 1.0, 2.0)
    land[10:18, 20:35] = 3.0  # deforestation
    land[40:44, 50:70] = 4.0  # gain
    land[rng.random((ny, nx)) < 0.02] = np.nan
    flat = land.ravel()

    sizes = {1: 60, 2: 50, 3: 40, 4: 30}
    seed = 7
    g = np.random.default_rng(seed)
    cells, strata_of = [], []
    for c in sorted(sizes):
        idx = np.flatnonzero(flat == c)
        pick = np.sort(g.choice(idx, size=sizes[c], replace=False))
        cells += pick.tolist()
        strata_of += [c] * sizes[c]
    cells = np.array(cells)
    m = flat[cells]

    lab = np.random.default_rng(3)
    reference = m.copy()
    wrong = lab.random(m.size) < 0.18
    reference[wrong] = lab.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=int(wrong.sum()))

    # strata of their own: three bands of rows (an older map's regions)
    rows = np.repeat(np.arange(ny), nx)
    strata = np.where(rows < 20, 1.0, np.where(rows < 45, 2.0, 3.0))
    strata[np.isnan(flat)] = np.nan
    s = strata[cells]

    classes = np.unique(flat[~np.isnan(flat)])
    nh = [int((flat == c).sum()) for c in classes]
    sc = np.unique(strata[~np.isnan(strata)])
    ns = [int((strata == c).sum()) for c in sc]

    # Olofsson et al. (2014), table 8: map classes deforestation, forest gain, stable forest, stable non-forest
    paper_counts = np.array([[66, 0, 5, 4], [0, 55, 8, 12], [1, 0, 153, 11], [2, 1, 9, 313]])
    paper_nh = [200_000, 150_000, 3_200_000, 6_450_000]
    pm, pr = [], []
    for i in range(4):
        for j in range(4):
            pm += [i + 1] * int(paper_counts[i, j])
            pr += [j + 1] * int(paper_counts[i, j])

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "d.csv").write_text("map,reference,stratum\n" + "".join(
            f"{int(a)},{int(b)},{int(c)}\n" for a, b, c in zip(m, reference, s)))
        (t / "nh.csv").write_text("class,cells\n" + "".join(f"{int(c)},{n}\n" for c, n in zip(classes, nh)))
        (t / "ns.csv").write_text("stratum,cells\n" + "".join(f"{int(c)},{n}\n" for c, n in zip(sc, ns)))
        (t / "p.csv").write_text("map,reference\n" + "".join(f"{a},{b}\n" for a, b in zip(pm, pr)))
        (t / "pn.csv").write_text("class,cells\n" + "".join(f"{i + 1},{n}\n" for i, n in enumerate(paper_nh)))
        (t / "a.R").write_text(R_SCRIPT)
        subprocess.run([rscript(), str(t / "a.R"), str(t / "d.csv"), str(t / "nh.csv"), str(t / "ns.csv"),
                        str(t / "out.txt"), str(t / "p.csv"), str(t / "pn.csv")], check=True)
        r = read_r(t / "out.txt")

    def nan_none(a):
        return [None if np.isnan(v) else float(v) for v in a]

    out = {
        "shape": [ny, nx],
        "transform": [500000.0, 30.0, 0.0, 8000000.0, 0.0, -30.0],
        "map": nan_none(flat),
        "strata": nan_none(strata),
        "sizes": {str(k): v for k, v in sizes.items()},
        "seed": seed,
        "cells": cells.tolist(),
        "stratum_of": strata_of,
        "reference": reference.tolist(),
        "paper": {"map": pm, "reference": pr, "cells": paper_nh},
        "r": r,
    }
    REF.mkdir(parents=True, exist_ok=True)
    path = REF / "accuracy.json"
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(path, len(cells), "sample units")


if __name__ == "__main__":
    main()
