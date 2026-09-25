# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden MSPA outputs with the original GuidosToolbox executable.

Usage::

    python tools/make_reference.py path/to/mspa_win64.exe [path/to/GTB/data/input.tif]

Writes ``tests/data/reference/*.npz`` (inputs and expected outputs) and a
``manifest.json`` describing how they were produced. The test-suite compares
landschaft against these files, so it runs anywhere without the executable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import scenes  # noqa: E402
from gtb_reference import run_original_mspa  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference"

# (connectivity, edge_width, internal)
GTB_INPUT_PARAMS = [(8, 1, 1), (4, 1, 1), (8, 2, 1), (4, 3, 0), (8, 5, 1), (8, 10, 0), (4, 20, 1)]
SYNTHETIC_PARAMS = [(8, 1, 1), (4, 2, 1), (8, 3, 0), (4, 7, 1)]
SYNTHETIC_CASES = [(kind, seed) for kind in scenes.SCENES for seed in (101, 202)]


def key(conn: int, eew: int, internal: int) -> str:
    return f"c{conn}_e{eew}_i{internal}"


def main(exe: str, gtb_input: str | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated": dt.date.today().isoformat(),
        "executable": Path(exe).name,
        "executable_sha256": hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
        "command": "mspa -graphfg C -eew E -internal I -disk -transition 1 -i in.tif -o out.tif -odir ./",
        "files": {},
    }

    if gtb_input:
        with rasterio.open(gtb_input) as src:
            data = src.read(1)
        arrays = {"input": data}
        for conn, eew, internal in GTB_INPUT_PARAMS:
            arrays[key(conn, eew, internal)] = run_original_mspa(
                exe, data, connectivity=conn, edge_width=eew, internal=bool(internal)
            )
            print("gtb_input", key(conn, eew, internal))
        np.savez_compressed(OUT / "gtb_input.npz", **arrays)
        manifest["files"]["gtb_input.npz"] = {"source": "GTB data/input.tif", "params": GTB_INPUT_PARAMS}

    arrays = {}
    for kind, seed in SYNTHETIC_CASES:
        data = scenes.make(kind, seed)
        arrays[f"{kind}_{seed}/input"] = data
        for conn, eew, internal in SYNTHETIC_PARAMS:
            arrays[f"{kind}_{seed}/{key(conn, eew, internal)}"] = run_original_mspa(
                exe, data, connectivity=conn, edge_width=eew, internal=bool(internal)
            )
        print("synthetic", kind, seed)
    np.savez_compressed(OUT / "synthetic.npz", **arrays)
    manifest["files"]["synthetic.npz"] = {"source": "tests/scenes.py", "params": SYNTHETIC_PARAMS}

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
