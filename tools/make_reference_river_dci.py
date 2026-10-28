# SPDX-License-Identifier: GPL-3.0-or-later
"""What R's riverconn (0.3) gives, for landschaft's tests (tests/test_rivers.py), written to
tests/data/reference/rivers/river_dci.json:

    python tools/make_reference_river_dci.py

River networks drawn at random as trees of segments (each with a length and the segment below it), the edge between
a segment and the one below either a barrier (its passabilities up and down) or a confluence (passable); and
riverconn's indices on them, with the passabilities only (c_ij, no dispersal): the full index (the DCI, symmetric and
asymmetric), the reach index of every segment (to and from the others, asymmetric; to them, symmetric) and the
d_index of every barrier removed alone (leave one out). Needs R with riverconn and igraph (Rscript on the PATH or in
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
REF = ROOT / "tests" / "data" / "reference" / "rivers"

# name: segments, share of the edges that are barriers, the lowest passability, seed
CASES = {"small": (12, 0.6, 0.1, 1), "branchy": (60, 0.4, 0.02, 7), "chain": (25, 0.8, 0.3, 11), "open": (30, 0.3, 0.5, 5)}

R_SCRIPT = r"""
suppressMessages({library(riverconn); library(igraph); library(dplyr)})
args <- commandArgs(trailingOnly = TRUE)
nodes <- read.csv(args[1], stringsAsFactors = FALSE)
edges <- read.csv(args[2], stringsAsFactors = FALSE)
nodes$name <- as.character(nodes$name)
edges$from <- as.character(edges$from)
edges$to <- as.character(edges$to)
edges$id_barrier <- as.character(edges$id_barrier)
g <- graph_from_data_frame(edges, vertices = nodes, directed = TRUE)
ix <- function(...) index_calculation(g, weight = "length", B_ij_flag = FALSE, ...)
out <- list(
  full_sym = ix(index_type = "full", dir_fragmentation_type = "symmetric")$index,
  full_asym = ix(index_type = "full", dir_fragmentation_type = "asymmetric")$index,
  reach_to_sym = ix(index_type = "reach", index_mode = "to", dir_fragmentation_type = "symmetric")$index,
  reach_to_asym = ix(index_type = "reach", index_mode = "to", dir_fragmentation_type = "asymmetric")$index,
  reach_from_asym = ix(index_type = "reach", index_mode = "from", dir_fragmentation_type = "asymmetric")$index
)
b <- edges[!is.na(edges$id_barrier), ]
meta <- data.frame(id_barrier = b$id_barrier, pass_u_updated = 1, pass_d_updated = 1, stringsAsFactors = FALSE)
d <- d_index_calculation(g, barriers_metadata = meta, B_ij_flag = FALSE, index_type = "full",
                         dir_fragmentation_type = "symmetric", parallel = FALSE)
out$d_barrier <- as.character(d$id_barrier)
out$d_index_value <- d$index
out$d_index <- d$d_index
writeLines(jsonlite::toJSON(out, digits = NA, auto_unbox = FALSE), args[3])
"""


def rscript() -> str:
    found = shutil.which("Rscript")
    if found:
        return found
    for p in sorted(glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"), reverse=True):
        return p
    raise SystemExit("Rscript not found")


def case(n: int, share: float, low: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    below = [-1] + [int(rng.integers(0, i)) for i in range(1, n)]
    length = [float(round(x, 3)) for x in rng.uniform(100, 5000, n)]
    up, down, barrier = [1.0] * n, [1.0] * n, [None] * n
    k = 0
    for i in range(1, n):
        if rng.random() < share:
            k += 1
            barrier[i] = f"b{k}"
            up[i] = float(round(rng.uniform(low, 1), 4))
            down[i] = float(round(rng.uniform(low, 1), 4))
    return {"below": below, "length": length, "up": up, "down": down, "barrier": barrier}


def main() -> None:
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "dci.R").write_text(R_SCRIPT, encoding="utf-8")
        for name, (n, share, low, seed) in CASES.items():
            c = case(n, share, low, seed)
            nodes = ["name,length"] + [f"{i + 1},{c['length'][i]}" for i in range(n)]
            edges = ["from,to,pass_u,pass_d,id_barrier"]
            for i in range(1, n):
                b = c["barrier"][i]
                pu, pd = (c["up"][i], c["down"][i]) if b else ("NA", "NA")
                edges.append(f"{i + 1},{c['below'][i] + 1},{pu},{pd},{b if b else 'NA'}")
            (tmp / "nodes.csv").write_text("\n".join(nodes) + "\n", encoding="utf-8")
            (tmp / "edges.csv").write_text("\n".join(edges) + "\n", encoding="utf-8")
            res = tmp / "out.json"
            subprocess.run(
                [rscript(), str(tmp / "dci.R"), str(tmp / "nodes.csv"), str(tmp / "edges.csv"), str(res)], check=True
            )
            c["riverconn"] = json.loads(res.read_text(encoding="utf-8"))
            out[name] = c
            print(name, n, "segments", sum(b is not None for b in c["barrier"]), "barriers", c["riverconn"]["full_sym"])
    REF.mkdir(parents=True, exist_ok=True)
    (REF / "river_dci.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(REF / "river_dci.json")


if __name__ == "__main__":
    main()
