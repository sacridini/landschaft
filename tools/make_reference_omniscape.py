# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden circuit outputs with Circuitscape 6.1 and Omniscape 0.7 (Julia).

Usage::

    python tools/make_reference_omniscape.py [advanced] [omniscape]

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

``omniscape`` runs ``run_omniscape`` (INI files, ``cholmod`` and ``cg+amg``) on
Omniscape's own test rasters with its test configurations 1-3 (the source
Omniscape clone is ``LANDSCHAFT_OMNISCAPE_SRC``, default ``../Omniscape.jl``)
and on synthetic maps (NoData blobs and lines, source files with NoData, block
sizes 1-7, buffers, 4 neighbours, conductance input, windows wider than the
map), and stores the inputs, the options and the output rasters in
``omniscape.npz``.
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


OMNISCAPE_TEST = Path(os.environ.get("LANDSCHAFT_OMNISCAPE_SRC", ROOT.parent / "Omniscape.jl")) / "test" / "input"
OUTPUTS = ("cum_currmap", "flow_potential", "normalized_cum_currmap")
SOLVERS = {"cholmod": "cholmod", "cgamg": "cg+amg"}


def _read(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
    return a


def _write(path: Path, a: np.ndarray) -> None:
    import rasterio
    from rasterio.transform import from_origin

    ny, nx = a.shape
    with rasterio.open(path, "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float64", nodata=-9999.0,
                       transform=from_origin(0, ny, 1, 1)) as ds:
        ds.write(np.where(np.isnan(a), -9999.0, a), 1)


def omniscape_cases() -> dict[str, dict]:
    """Omniscape's own test configurations (without the conditional ones) and synthetic maps."""
    rng = np.random.default_rng(20260927)
    cases: dict[str, dict] = {}
    res = _read(OMNISCAPE_TEST / "resistance.asc")
    src = _read(OMNISCAPE_TEST / "source.asc")
    res1 = _read(OMNISCAPE_TEST / "resistance1.asc")
    # test/input/config.ini (block_size 2 becomes 3), config2.ini and config3.ini.
    cases["os_test1"] = dict(resistance=res, source=src, options=dict(
        block_size=2, radius=5, buffer=2, source_threshold=0.2, calc_flow_potential=True,
        calc_normalized_current=True, correct_artifacts=True, mask_nodata=True))
    cases["os_test2"] = dict(resistance=res, source=src, options=dict(
        block_size=3, radius=5, buffer=2, source_threshold=0.2, r_cutoff=0.0, correct_artifacts=True))
    cases["os_test3"] = dict(resistance=res1, source=None, options=dict(
        block_size=3, radius=5, source_threshold=0.2, calc_flow_potential=True, source_from_resistance=True,
        r_cutoff=0.25, mask_nodata=True))
    # Synthetic: log-normal resistance with NoData blobs and a source file with NoData.
    ny, nx = 70, 90
    r = np.exp(rng.normal(1.0, 1.2, (ny, nx)))
    yy, xx = np.mgrid[:ny, :nx]
    for cy, cx, rad in [(20, 30, 6), (50, 70, 8), (60, 15, 5), (5, 85, 4)]:
        r[(yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2] = np.nan
    r[35, 10:60] = np.nan
    s = rng.uniform(0, 1, (ny, nx))
    s[rng.random((ny, nx)) < 0.3] = 0.0
    s[rng.random((ny, nx)) < 0.05] = np.nan
    s[20, 30] = 0.9  # a source on NoData resistance (a target without a node)
    cases["synth_block5"] = dict(resistance=r, source=s, options=dict(
        block_size=5, radius=10, buffer=3, calc_flow_potential=True, calc_normalized_current=True,
        correct_artifacts=True))
    cases["synth_block1_four"] = dict(resistance=1 / r, source=s, options=dict(
        block_size=1, radius=7, buffer=2, resistance_is_conductance=True, connect_four_neighbors_only=True,
        calc_flow_potential=True, mask_nodata=False))
    cases["synth_block7_nocorr"] = dict(resistance=r, source=None, options=dict(
        block_size=7, radius=8, buffer=0, correct_artifacts=False, source_from_resistance=True, r_cutoff=0.5,
        calc_normalized_current=True))
    cases["synth_block3_buffer"] = dict(resistance=r[:40, :60], source=s[:40, :60], options=dict(
        block_size=3, radius=8, buffer=5, source_threshold=0.3, calc_flow_potential=True,
        calc_normalized_current=True))
    # Windows wider than the map: cut on both sides (no correction there).
    cases["narrow"] = dict(resistance=r[:20, :], source=s[:20, :], options=dict(
        block_size=1, radius=10, buffer=2, calc_flow_potential=True))
    return cases


def _ini(options: dict, solver: str, source: bool) -> str:
    lines = ["[Input files]", "resistance_file = resistance.tif"]
    if source:
        lines.append("source_file = source.tif")
    lines += ["[Options]", f"project_name = out_{solver}", f"solver = {SOLVERS[solver]}", "write_as_tif = true",
              "parallelize = false", "write_raw_currmap = true"]
    for k, v in options.items():
        lines.append(f"{k} = {str(v).lower() if isinstance(v, bool) else v}")
    return "\n".join(lines) + "\n"


def make_omniscape() -> None:
    import json
    import shutil
    import tempfile

    OUT.mkdir(parents=True, exist_ok=True)
    cases = omniscape_cases()
    tmp = Path(tempfile.mkdtemp(prefix="landschaft_omniscape_"))
    try:
        for name, case in cases.items():
            d = tmp / name
            d.mkdir()
            _write(d / "resistance.tif", case["resistance"])
            if case["source"] is not None:
                _write(d / "source.tif", case["source"])
            for solver in SOLVERS:
                (d / f"config_{solver}.ini").write_text(_ini(case["options"], solver, case["source"] is not None))
        julia("omniscape", str(tmp))
        store: dict[str, np.ndarray] = {}
        for name, case in cases.items():
            store[f"{name}/resistance"] = case["resistance"]
            if case["source"] is not None:
                store[f"{name}/source"] = case["source"]
            store[f"{name}/options"] = np.array(json.dumps(case["options"]))
            for solver in SOLVERS:
                for out in OUTPUTS:
                    f = tmp / name / f"out_{solver}" / f"{out}.tif"
                    if f.exists():
                        store[f"{name}/{solver}/{out}"] = _read(f)
        np.savez_compressed(OUT / "omniscape.npz", **store)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str]) -> None:
    tasks = argv or ["advanced", "omniscape"]
    for task in tasks:
        {"advanced": make_advanced, "omniscape": make_omniscape}[task]()


if __name__ == "__main__":
    main(sys.argv[1:])
