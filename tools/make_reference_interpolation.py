# SPDX-License-Identifier: GPL-3.0-or-later
"""What R's gstat (2.1) gives for points of a field, for landschaft's tests (the same file as Terraflux's), written to
tests/data/reference/stats/interpolation.npz:

    python tools/make_reference_interpolation.py

120 points of a smooth field with noise over 100 x 80 units, and a grid of 25 x 20 cells of 4 units (its upper left
corner at 0, 80). gstat's sample variogram (its default cutoff and width, and a cutoff of 50 in 10 bins), its fit of
a spherical, an exponential and a Gaussian model (fit.variogram's default weights, N/h^2), inverse distance weighting
(power 2 with every point; power 2.5 with the 8 nearest) and ordinary kriging (a spherical model given, with every
point and with the 10 nearest; the exponential model fitted) at the cells' centres. Needs R with gstat and sp
(Rscript on the PATH or in C:/Program Files/R).
"""

from __future__ import annotations

import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "data" / "reference" / "stats"

R_SCRIPT = r"""
suppressMessages({library(sp); library(gstat)})
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1])
g <- read.csv(args[2])
coordinates(d) <- ~x + y
coordinates(g) <- ~x + y
out <- file(args[3], "w")
put <- function(key, x) {
  x <- unlist(x)
  cat(key, paste(ifelse(is.na(x), "NA", sprintf("%.17g", x)), collapse = " "), "\n", file = out)
}
v <- variogram(z ~ 1, d)
put("vario.np", v$np); put("vario.dist", v$dist); put("vario.gamma", v$gamma)
v50 <- variogram(z ~ 1, d, cutoff = 50, width = 5)
put("vario50.np", v50$np); put("vario50.dist", v50$dist); put("vario50.gamma", v50$gamma)
s2 <- var(d$z)
put("s2", s2)
for (m in c("Sph", "Exp", "Gau")) {
  f <- fit.variogram(v, vgm(s2, m, 20, s2 / 10))
  put(paste0("fit", m, ".params"), c(f$psill[1], f$psill[2], f$range[2]))
}
put("idw2.pred", idw(z ~ 1, d, g, idp = 2, debug.level = 0)$var1.pred)
put("idw25.pred", idw(z ~ 1, d, g, idp = 2.5, nmax = 8, debug.level = 0)$var1.pred)
m <- vgm(1.2, "Sph", 30, 0.2)
k <- krige(z ~ 1, d, g, model = m, debug.level = 0)
put("ok.pred", k$var1.pred); put("ok.var", k$var1.var)
k <- krige(z ~ 1, d, g, model = m, nmax = 10, debug.level = 0)
put("ok10.pred", k$var1.pred); put("ok10.var", k$var1.var)
f <- fit.variogram(v, vgm(s2, "Exp", 20, s2 / 10))
k <- krige(z ~ 1, d, g, model = f, debug.level = 0)
put("okexp.pred", k$var1.pred); put("okexp.var", k$var1.var)
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
    rng = np.random.default_rng(31)
    n = 120
    x, y = rng.random(n) * 100, rng.random(n) * 80
    z = 3 + np.sin(x / 6) * np.cos(y / 5) + np.sin((x + y) / 9) * 0.5 + rng.normal(size=n) * 0.3
    ny, nx, cell = 20, 25, 4.0
    gx = np.tile(cell / 2 + cell * np.arange(nx), ny)
    gy = np.repeat(80 - cell / 2 - cell * np.arange(ny), nx)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        np.savetxt(t / "d.csv", np.c_[x, y, z], delimiter=",", header="x,y,z", comments="", fmt="%.17g")
        np.savetxt(t / "g.csv", np.c_[gx, gy], delimiter=",", header="x,y", comments="", fmt="%.17g")
        (t / "s.R").write_text(R_SCRIPT)
        subprocess.run([rscript(), str(t / "s.R"), str(t / "d.csv"), str(t / "g.csv"), str(t / "out.txt")], check=True)
        arrays: dict[str, np.ndarray] = {"points": np.c_[x, y, z]}
        for line in (t / "out.txt").read_text().splitlines():
            if line.strip():
                key, *values = line.split()
                arrays[key.replace(".", "/")] = np.array([np.nan if v == "NA" else float(v) for v in values])
    REF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF / "interpolation.npz", **arrays)
    print(REF / "interpolation.npz", len(arrays))


if __name__ == "__main__":
    main()
