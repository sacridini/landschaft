# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden circuit outputs with Circuitscape 6.1 and Omniscape 0.7 (Julia).

Usage::

    python tools/make_reference_omniscape.py [advanced]

Julia runs from ``LANDSCHAFT_JULIA`` (default: ``julia``) with the project in
``LANDSCHAFT_JULIA_PROJECT`` (an environment with Circuitscape, Omniscape and
NPZ; keep it in its own depot with ``JULIA_DEPOT_PATH``). The Julia side is
``tools/omniscape_reference.jl``.

``advanced`` builds synthetic rasters (NoData islands, finite and direct
grounds, components without a source or a ground, 4 and 8 neighbours, an
Omniscape-like window, conductances over six orders of magnitude), stores them
in ``tests/data/reference/circuit/advanced_inputs.npz`` and Circuitscape's
``compute_omniscape_current`` with the ``cholmod`` and ``cg+amg`` solvers in
``advanced.npz``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "reference" / "circuit"
SCRIPT = ROOT / "tools" / "omniscape_reference.jl"


def julia(*args: str) -> None:
    exe = os.environ.get("LANDSCHAFT_JULIA", "julia")
    cmd = [exe]
    project = os.environ.get("LANDSCHAFT_JULIA_PROJECT")
    if project:
        cmd.append(f"--project={project}")
    subprocess.run([*cmd, str(SCRIPT), *args], check=True)


def advanced_cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260926)
    cases: dict[str, np.ndarray] = {}

    def add(name, g, s, gr, four=False):
        cases[f"{name}/conductance"] = g
        cases[f"{name}/source"] = s
        cases[f"{name}/ground"] = gr
        cases[f"{name}/four"] = np.array(int(four))

    # NoData lines and holes cut islands off; sources everywhere (also on the
    # ground cell, which drops it) and one direct ground.
    ny, nx = 40, 50
    g = 1 / rng.uniform(1, 100, (ny, nx))
    g[rng.random((ny, nx)) < 0.1] = np.nan
    g[12, :35] = np.nan
    g[:, 38] = np.nan
    g[30:, 20] = np.nan
    s = np.where(rng.random((ny, nx)) < 0.15, rng.uniform(0.1, 3, (ny, nx)), 0.0)
    gr = np.zeros((ny, nx))
    gr[25, 25] = np.inf
    s[25, 25] = 1.0
    g[25, 25] = 0.5
    add("islands", g, s, gr)
    add("islands4", g, s, gr, four=True)

    # Finite grounds (and one direct ground) in one component.
    ny, nx = 35, 45
    g = 1 / rng.uniform(1, 50, (ny, nx))
    s = np.where(rng.random((ny, nx)) < 0.1, rng.uniform(0.5, 1.5, (ny, nx)), 0.0)
    gr = np.zeros((ny, nx))
    for y, x in rng.integers(0, [ny, nx], (6, 2)):
        gr[y, x] = rng.uniform(0.5, 2)
    gr[5, 40] = np.inf
    add("finite", g, s, gr)

    # Four components split by NoData columns: ground and sources; sources
    # only; ground only; finite ground and sources.
    ny, nx = 20, 43
    g = rng.uniform(0.2, 5, (ny, nx))
    g[:, [10, 21, 32]] = np.nan
    s = np.zeros((ny, nx))
    gr = np.zeros((ny, nx))
    s[3, 2], s[15, 8], gr[10, 5] = 1.0, 2.0, np.inf
    s[4, 14], s[18, 19] = 1.0, 0.5
    gr[9, 27] = np.inf
    s[2, 35], s[17, 41], gr[10, 38] = 1.5, 0.5, 0.8
    add("components", g, s, gr)

    # An Omniscape moving window: disk of radius 20, sources proportional to
    # their strength outside the 3 x 3 target block, ground at the centre.
    r = 20
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = np.sqrt(yy ** 2 + xx ** 2) <= r
    g = np.where(disk, 1 / rng.uniform(1, 100, disk.shape), np.nan)
    g[rng.random(disk.shape) < 0.05] = np.nan
    s = np.where(disk & np.isfinite(g), rng.uniform(0, 1, disk.shape), 0.0)
    s[r - 1:r + 2, r - 1:r + 2] = 0
    s *= 9 / s.sum()
    gr = np.zeros(disk.shape)
    gr[r, r] = np.inf
    g[r, r] = 0.3
    add("window", g, s, gr)

    # Conductances over six orders of magnitude.
    ny, nx = 60, 80
    g = 10 ** rng.uniform(-3, 3, (ny, nx))
    s = np.where(rng.random((ny, nx)) < 0.05, 1.0, 0.0)
    gr = np.zeros((ny, nx))
    gr[30, 40] = np.inf
    add("wide", g, s, gr)
    return cases


def make_advanced() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = OUT / "advanced_inputs.npz"
    np.savez_compressed(inputs, **advanced_cases())
    julia("advanced", str(inputs), str(OUT / "advanced.npz"))


def main(argv: list[str]) -> None:
    tasks = argv or ["advanced"]
    for task in tasks:
        {"advanced": make_advanced}[task]()


if __name__ == "__main__":
    main(sys.argv[1:])
