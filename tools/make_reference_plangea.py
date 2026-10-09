# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of plangea for the restoration planning inputs.

Usage::

    python tools/make_reference_plangea.py [--plangea DIR]

Builds the inputs of ``tests/plangea_cases.py``, writes them as GeoTIFFs and tables, runs plangea's
functions on them through ``tools/plangea_reference.R`` and stores the results in
``tests/data/reference/plangea/reference.npz`` (``<case>/<key>`` arrays), with the species names and
the md5 of every plangea R file in ``reference.json``.

plangea (International Institute for Sustainability, GPL >= 3) is not on CRAN: its R files are
sourced from ``--plangea`` (default ``LANDSCHAFT_PLANGEA`` or
``C:/Users/eduar/Documents/dev/plangea-pkg-dev``). Needs Rscript (``LANDSCHAFT_RSCRIPT`` as in
``tests/lsm_reference.py``) with raster, foreach, purrr, dplyr and jsonlite. The test-suite reads the
stored files, so it runs anywhere without R.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import lsm_reference
import plangea_cases as pc

OUT = ROOT / "tests" / "data" / "reference" / "plangea"
R_SCRIPT = ROOT / "tools" / "plangea_reference.R"
DEFAULT_PLANGEA = Path(os.environ.get("LANDSCHAFT_PLANGEA", "C:/Users/eduar/Documents/dev/plangea-pkg-dev"))


def write_tif(path: Path, values: np.ndarray, res: float = pc.RES) -> str:
    import landschaft as ls

    ls.write_raster(pc.grid(values, res=res), path)
    return path.as_posix()


def oa_job(tmp: Path, name: str, spec: dict) -> dict:
    d = tmp / name
    (d / "lu").mkdir(parents=True)
    (d / "past").mkdir()
    (d / "er").mkdir()
    cover = pc.land_cover()
    lu_files = [write_tif(d / "lu" / f"{c}.tif", cover[k]) for k, c in enumerate(pc.CLASSES)]
    past_files, er_files = [], []
    if spec["past"]:
        p = pc.past()
        past_files = [f"{c}_past.tif" for c in pc.NATURAL]
        for k, f in enumerate(past_files):
            write_tif(d / "past" / f, p[k])
    if spec["er"]:
        e = pc.ecoregions()
        er_files = [f"{c}_er.tif" for c in pc.NATURAL]
        for k, f in enumerate(er_files):
            write_tif(d / "er" / f, e[k])
    return {"lu_files": lu_files, "class_names": pc.CLASSES, "class_types": pc.TYPES, "natural": pc.NATURAL,
            "past_dir": (d / "past").as_posix(), "er_dir": (d / "er").as_posix(), "include_past": spec["past"],
            "include_er": spec["er"], "past_files": past_files, "er_files": er_files, "failsafe": spec["failsafe"],
            "problem_type": spec["problem"], "fixed": bool(spec.get("fixed"))}


def bd_job(tmp: Path, name: str, spec: dict) -> dict:
    d = tmp / name
    for group, species in pc.SPECIES.items():
        (d / "spp" / group).mkdir(parents=True)
    names, r = pc.ranges()
    groups = {s: g for g, ss in pc.SPECIES.items() for s in ss}
    for k, s in enumerate(names):
        write_tif(d / "spp" / groups[s] / f"{s}.tif", r[k])
    pc.habitat_table().to_csv(d / "spp" / "habitats.csv", index=False)
    (d / "var").mkdir()
    write_tif(d / "var" / "elevation.tif", pc.elevation())
    lim = pc.limits_table()
    lim.rename(columns={"species": "taxonid", "lower": "lowerelevation"})[["taxonid", "lowerelevation"]].to_csv(
        d / "var" / "lower.csv", index=False)
    lim.rename(columns={"species": "taxonid", "upper": "upperelevation"})[["taxonid", "upperelevation"]].to_csv(
        d / "var" / "upper.csv", index=False)
    job = {"spp_dir": (d / "spp").as_posix(), "var_dir": (d / "var").as_posix(),
           "subfolders": [f"{g}/" for g in pc.SPECIES], "table": "habitats.csv",
           "include_absent": spec["include_absent"], "elevation": spec["elevation"],
           "elevation_file": "elevation.tif", "lower_table": "lower.csv", "upper_table": "upper.csv",
           "exrisk": bool(spec.get("exrisk"))}
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plangea", type=Path, default=DEFAULT_PLANGEA)
    args = parser.parse_args()
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    with tempfile.TemporaryDirectory(prefix="plangea_ref_") as tmp:
        tmp = Path(tmp)
        cases = []
        oa_specs = {}
        for name, spec in pc.OA_CASES.items():
            oa_specs[name] = oa_job(tmp, name, spec)
            cases.append({"name": name, "kind": "oa", "oa": oa_specs[name]})
        for name, spec in pc.BD_CASES.items():
            base = oa_job(tmp, name + "_oa", pc.OA_CASES[spec["oa"]])
            bd = bd_job(tmp, name, spec)
            cases.append({"name": name, "kind": "bd", "oa": base, "bd": bd})
        for name, spec in pc.IT_CASES.items():
            shape = spec["shape"]
            h = pc.footprint(shape, spec["seed"], spec.get("low", 0.0), nan=not spec.get("gradient"))
            if spec.get("gradient"):
                h[3, 4] = np.nan
            case = {"name": name, "kind": "it", "hfi": write_tif(tmp / f"{name}_hfi.tif", h, spec["res"]),
                    "radius": spec["radius"], "gamma": spec["gamma"], "beta": spec["beta"], "z": spec["z"],
                    "hfi_weights": list(pc.HFI_WEIGHTS)}
            if spec.get("restore"):
                conv = pc.conversion(shape, spec["seed"] + 100)
                case["prop_convert"] = {"crops": write_tif(tmp / f"{name}_pc1.tif", conv[0], spec["res"]),
                                        "pasture": write_tif(tmp / f"{name}_pc2.tif", conv[1], spec["res"])}
                cells = np.flatnonzero(~np.isnan(h.ravel()))
                x = np.zeros(cells.size) if spec.get("gradient") else pc.restored(shape, spec["seed"] + 200).ravel()[cells]
                case["x_vals"] = x.tolist()
            if spec.get("gradient"):
                case["gradient"] = True
                case["step"] = 1e-5
            cases.append(case)
        cases.append({"name": "unc", "kind": "unc",
                      **{k: v.tolist() for k, v in pc.uncertainty_inputs().items()}})
        # bd_absent's restored shares: over its master pixels (anthropic land), known before R runs
        cover = pc.land_cover()
        anth = np.nan_to_num(cover[3]) + np.nan_to_num(cover[4])
        n_master = int((anth.ravel() > 0).sum())
        for c in cases:
            if c["kind"] == "bd" and c["bd"].get("exrisk"):
                c["bd"]["x_vals"] = pc.exrisk_x(n_master).tolist()
                c["bd"]["scen"] = [{"name": k, **v} for k, v in pc.SCEN_CASES.items()]
                c["bd"]["scen_cd"] = [{"name": k, **v} for k, v in pc.SCEN_CD_CASES.items()]
        eco_file = write_tif(tmp / "ecoregion_codes.tif", pc.ecoregion_codes())
        for name, spec in pc.EC_CASES.items():
            cases.append({"name": name, "kind": "ec", "oa": oa_job(tmp, name + "_oa", pc.OA_CASES["oa_past_er"]),
                          "domain": spec["domain"], "eco_file": eco_file, "x_vals": pc.ec_x(n_master).tolist()})
        for name, spec in pc.ESYS_CASES.items():
            cases.append({"name": name, "kind": "esys", "oa": oa_job(tmp, name + "_oa", pc.OA_CASES["oa_past_er"]),
                          "eco_file": eco_file, **spec})
        job = {"plangea": args.plangea.joinpath("R").as_posix(), "cases": cases}
        (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
        out = tmp / "out"
        out.mkdir()
        subprocess.run([str(rscript), str(R_SCRIPT), str(tmp / "job.json"), str(out)], check=True,
                       env=lsm_reference.r_environment(), cwd=tmp)
        result = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        arrays, texts = {}, {}
        for item in result["manifest"]:
            key = f"{item['case']}/{item['key']}"
            if item.get("kind") == "text":
                texts[key] = item["text"]
            else:
                arrays[key] = np.fromfile(out / item["file"], dtype="<f8")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "reference.npz", **arrays)
    (OUT / "reference.json").write_text(json.dumps({"r": result["r"], "plangea_md5": result["hashes"], "texts": texts},
                                                   indent=1, sort_keys=True), encoding="utf-8")
    print(f"{len(arrays)} arrays -> {(OUT / 'reference.npz').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
