# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of Zonation 5 for ls.prioritize.

Usage::

    python tools/make_reference_prioritize.py [group ...] [--case NAME ...] [--threads N]

Runs the cases of ``tests/prioritize_cases.py`` (all groups by default) with the
official Zonation 5 binaries (``z5.exe``, ``z5_16bit.exe``; directory in
``LANDSCHAFT_Z5_DIR``, default ``~/Documents/dev/zonation5/install``) on one
thread, and stores, per group, the results in
``tests/data/reference/prioritize/<group>.npz`` (``<case>/<key>`` arrays: the
rank map, the feature info as numbers and, lzma-compressed, the text of the
curve tables) and
``<group>.json`` (the column names, the statistics of ``stats.txt``, the
command lines and the Zonation version). ``--case`` regenerates only the named
cases (the others are kept).

For every case with ``load=True`` Zonation also evaluates its own final rank map
in LOAD mode (``load/*`` keys): the curves of the ranking it returns, as its
iterative runs report the curves of the order before the last sort.
The test-suite reads the stored files, so it runs anywhere without Zonation.
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import prioritize_cases  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference" / "prioritize"
Z5_DIR = Path(os.environ.get("LANDSCHAFT_Z5_DIR", Path.home() / "Documents" / "dev" / "zonation5" / "install"))
TABLES = {  # output file -> key
    "feature_curves.csv": "feature_curves",
    "summary_curves.csv": "summary_curves",
    "feature_density_curves.csv": "density_curves",
    "group_curves.csv": "group_curves",
}


def write_raster(path: Path, spec: prioritize_cases.RasterSpec) -> None:
    """A one-strip GeoTIFF (Zonation 5 sums the layers per block of rows)."""
    import rasterio
    from rasterio.transform import from_origin

    v = np.asarray(spec.values, dtype=np.float64)
    ny, nx = v.shape
    nodata = spec.nodata
    if np.dtype(spec.dtype).kind in "iu":
        data = np.where(np.isnan(v), nodata if nodata is not None else 0, v).astype(spec.dtype)
    else:
        data = v.astype(spec.dtype)
        if nodata is not None and not np.isnan(nodata):
            data[np.isnan(v)] = nodata
    profile = dict(driver="GTiff", height=ny, width=nx, count=1, dtype=spec.dtype, crs=f"EPSG:{prioritize_cases.EPSG}",
                   transform=from_origin(500_000.0, 4_000_000.0 + ny * spec.cellsize, spec.cellsize, spec.cellsize),
                   blockysize=ny, tiled=False)
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def read_table(path: Path) -> tuple[list[str], np.ndarray, list[list[str]]]:
    """A whitespace-separated table with a quoted header: names, numeric columns, text columns."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    header = shlex.split(lines[0], posix=True)
    rows = [shlex.split(ln.replace("\\", "/"), posix=True) for ln in lines[1:]]
    numeric, text = [], []
    for j in range(len(header)):
        col = [r[j] for r in rows]
        try:
            numeric.append([float(x) for x in col])
        except ValueError:
            text.append(col)
    names = [h for j, h in enumerate(header) if all(_isnum(r[j]) for r in rows)]
    values = np.array(numeric, dtype=np.float64).T if numeric else np.zeros((len(rows), 0))
    return names, values.reshape(len(rows), len(names)), text


def _isnum(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def read_stats(path: Path) -> dict[str, str]:
    stats = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            stats[k] = v
    return stats


def feature_list(case: prioritize_cases.Case, files: dict[str, Path]) -> str:
    cols = dict(case.columns)
    header = [*cols, "filename"]
    lines = [" ".join(header)]
    for i, name in enumerate(case.layers):
        fname = files[name].name
        if i in case.extract:
            fname = f"{fname}/{case.extract[i]:g}"
        vals = [("nan" if isinstance(v[i], float) and np.isnan(v[i]) else f"{v[i]}") for v in cols.values()]
        lines.append(" ".join([*vals, f'"{fname}"']))
    return "\n".join(lines) + "\n"


def run_z5(exe: Path, args: list[str], cwd: Path, threads: int) -> None:
    cmd = [str(exe), f"--maxthreads={threads}", *args]
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")


def collect(out: Path, prefix: str, arrays: dict, meta: dict) -> None:
    import rasterio

    with rasterio.open(out / "rankmap.tif") as src:
        rank = src.read(1)
    arrays[f"{prefix}rank"] = np.where(np.isfinite(rank), rank, np.nan).astype(np.float32)
    for fname, key in TABLES.items():
        if (out / fname).exists():
            names, _, _ = read_table(out / fname)
            meta[f"{prefix}{key}"] = names
            if prefix:  # the curves of the final ranking, as the CSV text (its rounding is compared)
                text = (out / fname).read_bytes().replace(b"\r\n", b"\n")
                arrays[f"{prefix}{key}"] = np.frombuffer(lzma.compress(text), dtype=np.uint8)
    for sub, key in (("pre_processing", "pre"), ("transformed_layers", "tr")):
        for tif in sorted((out / sub).glob("*.tif")) if (out / sub).exists() and not prefix else ():
            with rasterio.open(tif) as src:
                v = src.read(1)
            name = tif.stem if key == "pre" else tif.name.split(".layer", 1)[1].split(".", 1)[0]
            arrays[f"{prefix}{key}/{name}"] = np.where(np.isfinite(v), v, np.nan).astype(np.float32)
    imp = out / "impacts"
    for f in sorted(imp.rglob("*")) if imp.exists() and not prefix else ():
        if not f.is_file():
            continue
        key = "impacts/" + f.relative_to(imp).as_posix()
        if f.suffix == ".tif":
            with rasterio.open(f) as src:
                arrays[key] = src.read(1)
        else:
            text = f.read_bytes().replace(b"\r\n", b"\n")
            arrays[key] = np.frombuffer(lzma.compress(text), dtype=np.uint8)
    post = out / "post_processing"
    for f in sorted(post.iterdir()) if post.exists() else ():
        key = f"{prefix}post/{f.name}"
        if f.suffix == ".tif":
            with rasterio.open(f) as src:
                arrays[key] = src.read(1)
        elif f.suffix in (".txt", ".csv"):
            text = f.read_bytes().replace(b"\r\n", b"\n")
            arrays[key] = np.frombuffer(lzma.compress(text), dtype=np.uint8)
    names, values, text = read_table(out / "features_info.csv")
    arrays[f"{prefix}features_info"] = values
    meta[f"{prefix}features_info"] = names
    meta[f"{prefix}stats"] = read_stats(out / "stats.txt")
    issues = (out / "issues.log").read_text(encoding="utf-8").splitlines()
    meta[f"{prefix}warnings"] = [ln.split("] ", 2)[-1] for ln in issues if "[warning]" in ln or "[error]" in ln]


def settings_text(settings: dict, files: dict) -> str:
    """A settings file (``.z5``): raster names become their files, "file:NAME" the link file NAME."""
    lines = ["feature list file = features.txt"]
    for k, v in settings.items():
        if isinstance(v, str) and v in files:
            v = files[v].name
        elif isinstance(v, str) and v.startswith("file:"):
            v = v[5:]
        lines.append(f"{k} = {v}")
    return "\n".join(lines) + "\n"


def write_project(case: prioritize_cases.Case, tmp: Path) -> dict:
    """Write the rasters, link files, feature list and settings.z5 of a case into tmp."""
    rasters = set(case.layers) | set(case.extra) | {v for v in case.settings.values() if isinstance(v, str) and v in
                                                    prioritize_cases.RASTERS}
    files = {}
    for r in sorted(rasters):
        files[r] = tmp / f"{r}.tif"
        if not files[r].exists():
            write_raster(files[r], prioritize_cases.RASTERS[r])
    for fname, text in case.files.items():
        (tmp / fname).write_text(text, encoding="utf-8")
    (tmp / "features.txt").write_text(feature_list(case, files), encoding="utf-8")
    (tmp / "settings.z5").write_text(settings_text(case.settings, files), encoding="utf-8")
    return files


def run_case(name: str, case: prioritize_cases.Case, threads: int) -> tuple[dict, dict]:
    exe = Z5_DIR / ("z5_16bit.exe" if case.bits == 16 else "z5.exe")
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="z5_ref_") as tmp:
        tmp = Path(tmp)
        files = write_project(case, tmp)
        args = [f"--mode={case.mode}", *case.flags, "settings.z5", "out"]
        run_z5(exe, args, tmp, threads)
        meta["command"] = " ".join([exe.name, *args])
        out = tmp / "out"
        if "--area=" in " ".join(case.flags):
            out = out / f"subregion_{[f for f in case.flags if f.startswith('--area=')][0][7:]}"
        collect(out, "", arrays, meta)
        if case.load:
            load_settings = {k: v for k, v in case.settings.items() if k not in ("external solution",)}
            load_settings["external solution"] = str(out / "rankmap.tif")
            (tmp / "load.z5").write_text(settings_text(load_settings, files), encoding="utf-8")
            flags = tuple("-x" if f == "-X" else f for f in case.flags)
            args = ["--mode=LOAD", *flags, "load.z5", "load"]
            run_z5(exe, args, tmp, threads)
            meta["load/command"] = " ".join([exe.name, *args])
            lout = tmp / "load"
            if "--area=" in " ".join(case.flags):
                lout = lout / f"subregion_{[f for f in case.flags if f.startswith('--area=')][0][7:]}"
            collect(lout, "load/", arrays, meta)
    return arrays, meta


def z5_version() -> str:
    res = subprocess.run([str(Z5_DIR / "z5.exe"), "--version"], capture_output=True, text=True)
    return res.stdout.strip()


def run_group(group: str, only: list[str] | None, threads: int) -> None:
    cases = {k: c for k, c in prioritize_cases.CASES.items() if c.group == group and (not only or k in only)}
    if not cases:
        return
    if not (Z5_DIR / "z5.exe").exists():
        sys.exit(f"z5.exe not found in {Z5_DIR}: set LANDSCHAFT_Z5_DIR")
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, object] = {}
    for name, case in cases.items():
        a, m = run_case(name, case, threads)
        arrays.update({f"{name}/{k}": v for k, v in a.items()})
        meta.update({f"{name}/{k}": v for k, v in m.items()})
        print(f"  {name}: {m.get('stats', {}).get('Number of iterations', '-')} iterations")
    OUT.mkdir(parents=True, exist_ok=True)
    npz, js = OUT / f"{group}.npz", OUT / f"{group}.json"
    if only and npz.exists():  # keep the other cases
        with np.load(npz) as old:
            kept = {k: old[k] for k in old.files if k.split("/", 1)[0] not in cases}
        kept_meta = {k: v for k, v in json.loads(js.read_text(encoding="utf-8"))["cases"].items()
                     if k.split("/", 1)[0] not in cases}
        arrays = {**kept, **arrays}
        meta = {**kept_meta, **meta}
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps({"zonation": z5_version(), "threads": threads, "cases": meta}, indent=1,
                             sort_keys=True), encoding="utf-8")
    print(f"{group}: {len(cases)} cases, {len(arrays)} arrays -> {npz.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="*", default=prioritize_cases.GROUPS)
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    for group in args.groups:
        run_group(group, args.case, args.threads)


if __name__ == "__main__":
    main()
