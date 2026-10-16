# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the SWAT+ reference runs for ls.hydro.swat.

Usage::

    python tools/make_reference_swat.py [case ...]

Each case is a SWAT+ project (TxtInOut) taken from SWAT+'s own examples and edited
(period, options); its input files are stored in
``tests/data/reference/swat/<case>.zip`` and the reference's daily dump in
``tests/data/reference/swat/<case>.npz`` (``t<tag>_keys``: year, day, object;
``t<tag>_vals``: the values), with the versions in ``swat.json``. The test-suite
reads the stored files.

The reference is SWAT+ 62.0.1 (github.com/swat-model/swatplus, tag ``62.0.1``,
LGPL-2.1) with ``tools/swat/swat_reference.patch`` applied: a binary daily dump
(``ls_dump_module.f90``, active when ``SWAT_DUMP`` names a file), the name search
made exact (the reference's binary search misses names of unsorted lists) and three
REAL array indices declared INTEGER (flang rejects them; outside the ported subset).
Build it with conda-forge's flang 22.1.8 and the MSVC linker against the dynamic UCRT,
without optimisation, so it calls the same C library maths as landschaft's MSVC
build::

    git clone https://github.com/swat-model/swatplus && cd swatplus && git checkout 62.0.1
    git config core.autocrlf false        # then strip the CR line ends of src/*.f90
    git apply <landschaft>/tools/swat/swat_reference.patch
    conda create -n swat-ref22 -c conda-forge flang=22.1.8 flang_win-64=22.1.8 cmake ninja
    # in a VS x64 prompt, with the env's Library\\bin first on PATH:
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=flang
          -DTAG=62.0.1 -DCMAKE_TRY_COMPILE_CONFIGURATION=Release
          -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -DCMAKE_POLICY_DEFAULT_CMP0091=NEW
          "-DCMAKE_EXE_LINKER_FLAGS=-libpath:<env>/Library/lib/clang/22/lib/x86_64-pc-windows-msvc"
          "-DCMAKE_Fortran_FLAGS=-cpp"
    cmake --build build

``LANDSCHAFT_SWAT`` gives the executable, ``LANDSCHAFT_SWAT_DATA`` the folder with
``swatplus`` (for ``refdata/Ames_sub1``) and ``SWATPlus-datasets`` (for
``SWAT+_model_examples/example1``, Hubbard Brook watershed 2).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "reference" / "swat"
DATA = Path(os.environ.get("LANDSCHAFT_SWAT_DATA", Path.home() / "Documents" / "dev" / "swat_sources"))
EXE = Path(os.environ.get("LANDSCHAFT_SWAT", DATA / "swatplus" / "build-flang22md" / "swatplus-62.0.1--win_amd64-Rel.exe"))

# outputs of earlier runs found in the example folders
OUTPUT = re.compile(r"(\.txt|\.out|\.fin|\.csv|\.bin|\.log)$|^fort\.\d+$", re.I)


def read_inputs(src: Path) -> dict[str, bytes]:
    files = {}
    for p in sorted(src.iterdir()):
        if p.is_file() and not OUTPUT.search(p.name):
            b = p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            if b and not b.endswith(b"\n"):
                b += b"\n"
            files[p.name] = b
    return files


def set_period(files: dict[str, bytes], start: int, end: int) -> None:
    files["time.sim"] = (
        "time.sim: landschaft test period\n"
        "day_start  yrc_start   day_end   yrc_end      step\n"
        f"       0      {start}         0      {end}         0\n"
    ).encode()


def trim_climate(files: dict[str, bytes], start: int, end: int) -> None:
    """Keep the years start..end+1 of every daily climate series (the reference reads
    nothing else; the monthly means it takes of the rest are not used)."""
    for name in list(files):
        if not re.search(r"\.(pcp|tmp|tem|slr|hmd|wnd|pet)$", name, re.I):
            continue
        lines = files[name].decode().split("\n")
        if len(lines) < 4:
            continue
        head, rows = lines[:3], [r for r in lines[3:] if r.strip()]
        keep = [r for r in rows if start <= int(r.split()[0]) <= end + 1]
        if not keep:
            continue
        years = sorted({int(r.split()[0]) for r in keep})
        h = head[2].split()
        h[0] = str(len(years))
        head[2] = "  " + "  ".join(h)
        files[name] = ("\n".join(head + keep) + "\n").encode()


def hubbard_brook(start: int, end: int):
    def make() -> dict[str, bytes]:
        files = read_inputs(DATA / "SWATPlus-datasets" / "SWAT+_model_examples" / "example1")
        # written for rev 60.5.7: the basin line of file.cio needs the carbon file of 62
        cio = files["file.cio"].decode().split("\n")
        cio = [re.sub(r"^(basin\s+codes\.bsn\s+parameters\.bsn)\s*$", r"\1    null", ln) for ln in cio]
        files["file.cio"] = "\n".join(cio).encode()
        set_period(files, start, end)
        trim_climate(files, start, end)
        return files

    return make


def set_code(files: dict[str, bytes], column: str, value: str) -> None:
    """Set one column of codes.bsn (title, header, values)."""
    lines = files["codes.bsn"].decode().split("\n")
    head, vals = lines[1].split(), lines[2].split()
    vals[head.index(column)] = value
    lines[2] = " ".join(f"{v:>10}" for v in vals)
    files["codes.bsn"] = "\n".join(lines).encode()


def ames(start: int, end: int):
    def make() -> dict[str, bytes]:
        files = read_inputs(DATA / "swatplus" / "refdata" / "Ames_sub1")
        set_code(files, "carbon", "0")  # the static carbon of the port, not CENTURY
        set_period(files, start, end)
        trim_climate(files, start, end)
        return files

    return make


CASES = {
    "hubbard_brook": hubbard_brook(2000, 2002),
    "ames": ames(1975, 1979),
}


def run_reference(files: dict[str, bytes]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        for name, b in files.items():
            (wd / name).write_bytes(b)
        env = dict(os.environ, SWAT_DUMP=str(wd / "dump.bin"))
        res = subprocess.run([str(EXE)], cwd=wd, env=env, capture_output=True, text=True)
        if res.returncode != 0 or not (wd / "dump.bin").exists():
            raise RuntimeError(f"SWAT+ failed ({res.returncode}): {res.stdout[-2000:]}{res.stderr[-2000:]}")
        return read_dump((wd / "dump.bin").read_bytes())


def read_dump(b: bytes) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    recs: dict[int, list] = {}
    i = 0
    while i < len(b):
        h = np.frombuffer(b, np.int32, 5, i)
        n = int(h[4])
        recs.setdefault(int(h[0]), []).append((h[1:4].copy(), np.frombuffer(b, np.float32, n, i + 20)))
        i += 20 + 4 * n
    out = {}
    for tag, rs in recs.items():
        keys = np.array([k for k, _ in rs], np.int32)
        width = max(len(v) for _, v in rs)
        vals = np.zeros((len(rs), width), np.float32)
        for k, (_, v) in enumerate(rs):
            vals[k, : len(v)] = v
        out[tag] = (keys, vals)
    return out


def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "swat.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    patch = (ROOT / "tools" / "swat" / "swat_reference.patch").read_bytes()
    meta["reference"] = {
        "swatplus": "62.0.1",
        "compiler": "flang 22.1.8 (conda-forge), MSVC link, dynamic UCRT, -O0",
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
    }
    for name in names or list(CASES):
        files = CASES[name]()
        dump = run_reference(files)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for fname, b in sorted(files.items()):
                z.writestr(fname, b)
        (OUT / f"{name}.zip").write_bytes(buf.getvalue())
        arrays = {}
        for tag, (keys, vals) in dump.items():
            arrays[f"t{tag}_keys"] = keys
            arrays[f"t{tag}_vals"] = vals
        np.savez_compressed(OUT / f"{name}.npz", **arrays)
        meta.setdefault("cases", {})[name] = {tag: list(v[1].shape) for tag, v in sorted(dump.items())}
        print(name, {tag: v[1].shape for tag, v in sorted(dump.items())})
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
