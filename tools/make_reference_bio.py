# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden Biodiverse outputs for the ``landschaft.bio`` tests.

Usage::

    python tools/make_reference_bio.py [data] [basedata] [trees] [matrices] [rasters] [vectors] [neighbours] [spatial] [phylo] [phylocom] [props] [rand] [cluster]

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


def _basedata_spec(case: dict) -> dict:
    spec = dict(bio_cases.BASEDATA_CASES[case["basedata"]]["biodiverse"])
    spec["file"] = _path(spec["file"])
    return spec


def _spatial_job(name: str, case: dict) -> dict:
    task = {"type": "spatial", "name": name, "spatial_conditions": case["conditions"],
            "calculations": bio_cases.spatial_calcs(case)}
    if case.get("definition_query"):
        task["definition_query"] = case["definition_query"]
    return {"basedata": _basedata_spec(case), "tasks": [task]}


def _phylo_job(name: str, case: dict) -> dict:
    task = {"type": "spatial", "name": name, "spatial_conditions": case["conditions"],
            "calculations": bio_cases.phylo_calcs(case)}
    return {"basedata": _basedata_spec(case), "tree": {"file": _path(case["tree"])}, "tasks": [task]}


def _props_job(name: str, case: dict) -> dict:
    job = {"basedata": _basedata_spec(case)}
    lb, gp = bio_cases.property_case_uses(case)
    for flag, key, props in ((lb, "labels_properties", bio_cases.LABEL_PROPS),
                             (gp, "groups_properties", bio_cases.GROUP_PROPS)):
        if flag:
            job[key] = {"file": _path(props["file"]),
                        "args": {"input_element_cols": props["element_columns"], **props["properties"]}}
    if "matrix" in case:
        mc = bio_cases.MATRIX_CASES[case["matrix"]]
        job["matrix"] = {"file": _path(mc["file"]), "params": mc["params"], "sparse": mc["sparse"],
                         "args": mc.get("args", {})}
    job["tasks"] = [{"type": "spatial", "name": name, "spatial_conditions": case["conditions"],
                     "calculations": case["calcs"]}]
    return job


def make_spatial(unsorted: bool) -> None:
    out = {}
    for name, case in bio_cases.SPATIAL_CASES.items():
        print(name)
        out[name] = run(_spatial_job(name, case), unsorted)[name]
    save("spatial", out)


def make_phylo(unsorted: bool) -> None:
    out = {}
    for name, case in bio_cases.PHYLO_CASES.items():
        print(name)
        out[name] = run(_phylo_job(name, case), unsorted)[name]
    save("phylo", out)


def make_props(unsorted: bool) -> None:
    out = {}
    for name, case in bio_cases.PROPERTY_CASES.items():
        print(name)
        out[name] = run(_props_job(name, case), unsorted)[name]
    save("props", out)


def make_export(unsorted: bool) -> None:
    import tempfile

    makers = {"spatial": (bio_cases.SPATIAL_CASES, _spatial_job), "phylo": (bio_cases.PHYLO_CASES, _phylo_job),
              "props": (bio_cases.PROPERTY_CASES, _props_job)}
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, case in bio_cases.EXPORT_CASES.items():
            cases, maker = makers[case["from"]]
            job = maker(name, cases[case["case"]])
            job["tasks"][0]["export"] = [
                {"list": lst, "file": str(Path(tmp) / f"{name}_{i}.csv"), "args": case.get("args", {})}
                for i, lst in enumerate(case["lists"])]
            print(name)
            out[name] = run(job, unsorted)[name]
    save("export", out)


def make_dependencies(unsorted: bool) -> None:
    """Biodiverse's calculation metadata used by landschaft (pre_calc order, neighbour sets)."""
    import json

    spec = dict(bio_cases.BASEDATA_CASES["site_100k"]["biodiverse"])
    spec["file"] = _path(spec["file"])
    meta = run({"basedata": spec, "tasks": [{"type": "metadata", "name": "metadata"}]}, unsorted)["metadata"]
    out = ROOT / "src" / "landschaft" / "bio" / "data" / "biodiverse_dependencies.json"
    out.write_text(json.dumps(meta, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def make_cluster(unsorted: bool) -> None:
    out = {}
    for name, case in bio_cases.CLUSTER_CASES.items():
        bdcase = bio_cases.BASEDATA_CASES[case["basedata"]]
        spec = dict(bdcase["biodiverse"])
        spec["file"] = _path(spec["file"])
        args = dict(case["args"])
        if case.get("calcs"):
            args["spatial_calculations"] = case["calcs"]
        task = {"type": "cluster", "name": name, "args": args, "region_grower": case.get("region_grower", False)}
        job = {"basedata": spec, "tasks": [task]}
        if "tree" in case:
            job["tree"] = {"file": _path(case["tree"])}
        print(name)
        out[name] = run(job, unsorted)[name]
    save("cluster", out)


def make_rand(unsorted: bool) -> None:
    out = {}
    for name, case in bio_cases.RAND_CASES.items():
        bdcase = bio_cases.BASEDATA_CASES[case["basedata"]]
        spec = dict(bdcase["biodiverse"])
        spec["file"] = _path(spec["file"])
        task = {"type": "spatial", "name": name, "spatial_conditions": case["conditions"],
                "calculations": case["calcs"],
                "randomise": [{"name": "rnd", "function": case["function"], "iterations": case["iterations"],
                               "seed": case["seed"], "args": case.get("args", {})}]}
        job = {"basedata": spec, "tasks": [task]}
        if "tree" in case:
            job["tree"] = {"file": _path(case["tree"])}
        if case.get("group_props"):
            props = bio_cases.GROUP_PROPS
            job["groups_properties"] = {"file": _path(props["file"]),
                                        "args": {"input_element_cols": props["element_columns"], **props["properties"]}}
        print(name)
        out[name] = run(job, unsorted)[name]
    save("rand", out)


def make_phylocom(unsorted: bool) -> None:
    out = {}
    for name, case in bio_cases.PHYLOCOM_CASES.items():
        bdcase = bio_cases.BASEDATA_CASES[case["basedata"]]
        spec = dict(bdcase["biodiverse"])
        spec["file"] = _path(spec["file"])
        task = {"type": "spatial", "name": name, "spatial_conditions": case["conditions"],
                "calculations": bio_cases.phylocom_calcs(case)}
        if "seed" in case:
            task["args"] = {"prng_seed": case["seed"]}
        print(name)
        out[name] = run({"basedata": spec, "tree": {"file": _path(case["tree"])}, "tasks": [task]}, unsorted)[name]
    save("phylocom", out)


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
    parts = [a for a in argv if not a.startswith("--")] or ["basedata", "trees", "matrices", "rasters", "vectors", "neighbours", "spatial", "phylo", "phylocom", "props", "rand", "cluster", "export"]
    if "data" in parts:
        bio_cases.write_synthetic_data()
        bio_cases.write_rasters()
        bio_cases.write_points()
        bio_cases.write_site_polygons()
        bio_cases.write_trees()
        bio_cases.write_label_props()
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
    if "spatial" in parts:
        make_spatial(unsorted)
    if "phylo" in parts:
        make_phylo(unsorted)
    if "phylocom" in parts:
        make_phylocom(unsorted)
    if "props" in parts:
        make_props(unsorted)
    if "rand" in parts:
        make_rand(unsorted)
    if "cluster" in parts:
        make_cluster(unsorted)
    if "export" in parts:
        make_export(unsorted)
    if "dependencies" in parts:
        make_dependencies(unsorted)


if __name__ == "__main__":
    main(sys.argv[1:])
