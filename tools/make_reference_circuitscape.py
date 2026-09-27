# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of Circuitscape 6.1's raster modes (Julia).

Usage::

    python tools/make_reference_circuitscape.py [network]

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

``network`` does the same for the network mode alone: Circuitscape's network
inputs (``test/input/network``) and synthetic ones (``input/network_extra``,
written here: numbering from 0 and 1, several components, duplicate edges and
self loops, include / exclude pairs, finite and direct grounds, the conflict
policies, a 4000-node graph) go to ``circuitscape_network.npz``.
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


def _write(path: Path, rows) -> None:
    path.write_text("".join(" ".join(repr(v) if isinstance(v, float) else str(v) for v in r) + "\n" for r in rows))


def _random_graph(rng, n: int, extra: int, zero: bool) -> list:
    """A grid-like graph of n nodes (rows of 20) with `extra` random chords."""
    edges = []
    base = 0 if zero else 1
    for v in range(n):
        if (v + 1) % 20 and v + 1 < n:
            edges.append((v + base, v + 1 + base, float(rng.uniform(0.5, 5))))
        if v + 20 < n:
            edges.append((v + base, v + 20 + base, float(rng.uniform(0.5, 5))))
    for _ in range(extra):
        a, b = rng.integers(0, n, 2)
        edges.append((int(a) + base, int(b) + base, float(rng.uniform(0.5, 5))))
    return edges


INI = """[Options for advanced mode]
ground_file_is_resistances = {ground_res}
remove_src_or_gnd = {policy}
ground_file = {ground}
source_file = {source}

[Calculation options]
solver = cholmod

[Options for pairwise and one-to-all and all-to-one modes]
included_pairs_file = {pairs}
use_included_pairs = {use_pairs}
point_file = {points}

[Output options]
write_cur_maps = {cur}
write_volt_maps = {volt}
output_file = output/{name}.out

[Habitat raster or graph]
habitat_file = {graph}
habitat_map_is_resistances = {res}

[Circuitscape mode]
data_type = network
scenario = {scenario}
"""


def make_network_extra(root: Path) -> list[str]:
    """Synthetic network cases (INI paths relative to `root`)."""
    rng = np.random.default_rng(2026)
    d = root / "input" / "network_extra"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    rel = "input/network_extra/"
    inis = []

    def ini(name, **kw):
        opts = dict(ground_res="True", policy="keepall", ground="None", source="None", pairs="None",
                    use_pairs="False", points="None", cur="False", volt="False", res="True",
                    scenario="pairwise", name=name)
        opts.update(kw)
        (d / f"{name}.ini").write_text(INI.format(**opts))
        inis.append(rel + f"{name}.ini")

    # 1: numbered from 0, two components, an isolated node, duplicate edges, a
    # self loop; focal nodes repeated, isolated and out of the graph; all maps.
    edges = _random_graph(rng, 120, 15, zero=True)
    edges += [(e[1], e[0], e[2]) for e in edges[:6]]              # both orientations
    edges += [edges[10], (5, 5, 2.0)]                             # a repeated line, a self loop
    edges += [(130 + k, 131 + k, float(rng.uniform(1, 3))) for k in range(8)]  # a second component
    edges += [(145, 140, 1.5)]
    _write(d / "net1_graph.txt", edges)
    _write(d / "net1_points.txt", [[0], [17], [119], [132], [17], [139], [150], [60]])
    ini("net1", graph=rel + "net1_graph.txt", points=rel + "net1_points.txt", cur="True", volt="True")
    # 2: numbered from 1, conductances, include pairs (list), resistances only.
    edges = _random_graph(rng, 200, 30, zero=False)
    _write(d / "net2_graph.txt", edges)
    _write(d / "net2_points.txt", [[3], [50], [101], [150], [199], [77]])
    _write(d / "net2_pairs.txt", [["mode", "include"], [3, 50], [101, 199], [50, 150], [77, 3]])
    ini("net2", graph=rel + "net2_graph.txt", points=rel + "net2_points.txt", res="False",
        pairs=rel + "net2_pairs.txt", use_pairs="True")
    # 3: numbered from 0, exclude pairs as a min/max matrix (ids from 0), current maps only.
    edges = _random_graph(rng, 160, 20, zero=True)
    _write(d / "net3_graph.txt", edges)
    _write(d / "net3_points.txt", [[2], [40], [80], [120], [159]])
    _write(d / "net3_pairs.txt", [["min", 1], ["max", 5], [0, 2, 40, 80, 120, 159], [2, 0, 9, 1, 1, 1],
                                  [40, 9, 0, 1, 0, 1], [80, 1, 1, 0, 1, 1], [120, 1, 0, 1, 0, 1],
                                  [159, 1, 1, 1, 1, 0]])
    ini("net3", graph=rel + "net3_graph.txt", points=rel + "net3_points.txt", pairs=rel + "net3_pairs.txt",
        use_pairs="True", cur="True")
    # 4: advanced, numbered from 0: finite and direct grounds, a node both source and
    # ground (rmvgnd), a component without a ground; all maps.
    edges = _random_graph(rng, 140, 10, zero=True) + [(150, 151, 2.0), (151, 152, 1.0)]
    _write(d / "net4_graph.txt", edges)
    _write(d / "net4_sources.txt", [[0, 1.0], [70, 2.5], [139, 0.5], [150, 1.0]])
    _write(d / "net4_grounds.txt", [[139, 3.0], [20, 0.0], [100, 7.5], [60, 1.25]])
    ini("net4", graph=rel + "net4_graph.txt", source=rel + "net4_sources.txt", ground=rel + "net4_grounds.txt",
        policy="rmvgnd", scenario="advanced", cur="True", volt="True")
    # 5: advanced, numbered from 1, conductances to ground, rmvall.
    edges = _random_graph(rng, 100, 12, zero=False)
    _write(d / "net5_graph.txt", edges)
    _write(d / "net5_sources.txt", [[1, 1.0], [50, 2.0], [99, 1.0]])
    _write(d / "net5_grounds.txt", [[99, 0.5], [30, 2.0], [100, 1.0]])
    ini("net5", graph=rel + "net5_graph.txt", source=rel + "net5_sources.txt", ground=rel + "net5_grounds.txt",
        policy="rmvall", ground_res="False", scenario="advanced", cur="True", volt="True")
    # 6: a 4000-node graph (the nested dissection), pairwise with current maps.
    edges = _random_graph(rng, 4000, 400, zero=False)
    _write(d / "net6_graph.txt", edges)
    _write(d / "net6_points.txt", [[int(v)] for v in rng.choice(np.arange(1, 4001), 8, replace=False)])
    ini("net6", graph=rel + "net6_graph.txt", points=rel + "net6_points.txt", cur="True")
    return inis


def main() -> None:
    if "network" in sys.argv[1:]:
        main_network()
        return
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


def _run(inis: list[str], target: Path) -> None:
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
                    store[key] = _read_output(f) if f.stat().st_size else np.zeros((0, 3))
                    break
        np.savez_compressed(target, **store)
        print(f"{len(store)} outputs of {len(inis)} runs")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main_network() -> None:
    root = OUT / "circuitscape"
    src = circuitscape_test_dir() / "input" / "network"
    dst = root / "input" / "network"
    if dst.exists():
        shutil.rmtree(dst, onexc=lambda f, p, e: (os.chmod(p, 0o666), f(p)))
    shutil.copytree(src, dst, copy_function=shutil.copyfile)
    inis = sorted(p.relative_to(root).as_posix() for p in dst.glob("*.ini"))
    inis += make_network_extra(root)
    _run(inis, OUT / "circuitscape_network.npz")


if __name__ == "__main__":
    main()
