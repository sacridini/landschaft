# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden Biodiverse outputs for the ``landschaft.bio`` tests.

Usage::

    python tools/make_reference_bio.py [data] [basedata] [trees] [matrices] [rasters] [vectors] [neighbours]

``data`` writes the synthetic inputs of ``tests/bio_cases.py`` into
``tests/data/bio``; the other parts run Biodiverse (Perl) through
``tools/bio_harness.py`` (sorted-order overlay) and store JSON files in
``tests/data/reference/bio``. The test-suite reads the stored files, so it runs
anywhere without Perl. With ``--unsorted`` each part is also run with the
unpatched Biodiverse and the largest difference is printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import bio_cases  # noqa: E402
import bio_harness  # noqa: E402

OUT = bio_cases.REFERENCE
DATA = bio_cases.DATA


def _path(name: str) -> str:
    return str(DATA / name)


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=True), encoding="utf-8")
    print(f"wrote {OUT / (name + '.json')}")


def _max_diff(a, b) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max((_max_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max((_max_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return 0.0
        return abs(a - b) / max(abs(a), abs(b), 1e-300)
    return 0.0 if a == b else float("inf")


def run(job: dict, unsorted: bool) -> dict:
    res = bio_harness.run_job(job)
    if unsorted:
        raw = bio_harness.run_job(job, sorted_order=False, hash_seed=12345)
        print(f"  unpatched Biodiverse: max relative difference {_max_diff(res, raw):.3g}")
    return res


def make_basedata(unsorted: bool) -> None:
    out = {}
    for name, case in bio_cases.BASEDATA_CASES.items():
        spec = dict(case["biodiverse"])
        spec["file"] = _path(spec["file"])
        print(name)
        res = run({"basedata": spec, "tasks": [{"type": "basedata", "name": "bd"}]}, unsorted)
        out[name] = res["bd"]
    save("basedata", out)


def make_rasters(unsorted: bool) -> None:
    out = {}
    for name, case in bio_cases.RASTER_CASES.items():
        spec = {k: v for k, v in case.items() if k != "files"}
        spec["raster_files"] = [_path(f) for f in case["files"]]
        print(name)
        res = run({"basedata": spec, "tasks": [{"type": "basedata", "name": "bd"}]}, unsorted)
        out[name] = res["bd"]
    save("rasters", out)


def make_vectors(unsorted: bool) -> None:
    out = {}
    for name, case in bio_cases.VECTOR_CASES.items():
        spec = {
            "shapefiles": [_path(f) for f in case["files"]],
            "cell_sizes": case["cell_sizes"],
            "group_field_names": [":shape_x", ":shape_y"],
            "label_field_names": case["labels"],
        }
        if "cell_origins" in case:
            spec["cell_origins"] = case["cell_origins"]
        counts = list(case.get("counts", []))
        if case.get("measure"):
            counts.append(":shape_" + case["measure"])
        if counts:
            spec["sample_count_col_names"] = counts
        print(name)
        res = run({"basedata": spec, "tasks": [{"type": "basedata", "name": "bd"}]}, unsorted)
        out[name] = res["bd"]
    save("vectors", out)


def make_neighbours(unsorted: bool) -> None:
    out = {}
    case = bio_cases.BASEDATA_CASES["site_100k"]
    spec = dict(case["biodiverse"])
    spec["file"] = _path(spec["file"])
    shapes = _path("site_polygons.shp").replace("\\", "/")
    tasks = []
    for name, conds in bio_cases.NEIGHBOUR_CASES.items():
        conds = [c.replace("SHAPES", shapes) for c in conds]
        task = {"type": "neighbours", "name": name, "spatial_conditions": conds}
        if name in bio_cases.NEIGHBOUR_DEFQ:
            task["definition_query"] = bio_cases.NEIGHBOUR_DEFQ[name]
        tasks.append(task)
    print("neighbours")
    res = run({"basedata": spec, "tasks": tasks}, unsorted)
    for name in bio_cases.NEIGHBOUR_CASES:
        out[name] = res[name]
    save("neighbours", out)


def make_trees(unsorted: bool) -> None:
    out = {"files": {}, "newick": {}}
    for f in bio_cases.TREE_FILES:
        print(f)
        out["files"][f] = run({"tasks": [{"type": "trees", "name": "t", "file": _path(f)}]}, unsorted)["t"]
    for name, nwk in bio_cases.NEWICK_CASES.items():
        print(name)
        out["newick"][name] = run({"tasks": [{"type": "trees", "name": "t", "data": nwk}]}, unsorted)["t"]
    save("trees", out)


def make_matrices(unsorted: bool) -> None:
    out = {}
    for name, case in bio_cases.MATRIX_CASES.items():
        spec = {"file": _path(case["file"]), "params": case["params"], "sparse": case["sparse"], "args": case.get("args", {})}
        print(name)
        out[name] = run({"matrix": spec, "tasks": [{"type": "matrix", "name": "m"}]}, unsorted)["m"]
    save("matrices", out)


def main(argv: list[str]) -> None:
    unsorted = "--unsorted" in argv
    parts = [a for a in argv if not a.startswith("--")] or ["basedata", "trees", "matrices", "rasters", "vectors", "neighbours"]
    if "data" in parts:
        bio_cases.write_synthetic_data()
        bio_cases.write_rasters()
        bio_cases.write_points()
        bio_cases.write_site_polygons()
        print("wrote synthetic inputs")
    if not bio_harness.available():
        sys.exit("Biodiverse/Strawberry Perl not found (see tools/bio_harness.py)")
    if "basedata" in parts:
        make_basedata(unsorted)
    if "trees" in parts:
        make_trees(unsorted)
    if "matrices" in parts:
        make_matrices(unsorted)
    if "rasters" in parts:
        make_rasters(unsorted)
    if "vectors" in parts:
        make_vectors(unsorted)
    if "neighbours" in parts:
        make_neighbours(unsorted)


if __name__ == "__main__":
    main(sys.argv[1:])
