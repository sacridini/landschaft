# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of GTB tools that exist only as IDL code.

Usage::

    python tools/make_reference_idl.py [path/to/gdl.exe]

Runs the verbatim GTB code in ``tests/gtb_idl`` with GDL (or IDL) and writes
``tests/data/reference/idl_reference.json``: for every case, the SHA-256 of
each output map and the numbers of the GTB report. The test-suite compares
landschaft against this file, so it runs anywhere without GDL.
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
from gtb_reference import find_gdl, run_gtb_accounting, run_gtb_accounting_scaled_classes  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference" / "idl_reference.json"
GTB_INPUT = ROOT / "src" / "landschaft" / "data" / "input.tif"

# (connectivity, thresholds, highlight_largest)
ACCOUNTING_PARAMS = [
    (8, (50, 250, 1000, 5000, 20000), False),
    (4, (50, 250, 1000, 5000, 20000), True),
    (8, (1, 10, 100), True),
    (4, (5,), False),
    (8, (3, 3, 40, 40, 900), True),
]


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def accounting_inputs() -> dict[str, np.ndarray]:
    with rasterio.open(GTB_INPUT) as src:
        inputs = {"gtb_input": src.read(1)}
    for kind in scenes.SCENES:
        inputs[f"{kind}_101"] = scenes.make(kind, 101)
    special = scenes.make("blobs", 202).copy()
    rng = np.random.default_rng(7)
    bg = special == 1
    special[bg & (rng.random(special.shape) < 0.2)] = 3
    special[bg & (rng.random(special.shape) < 0.1)] = 4
    inputs["special_bg_202"] = special
    return inputs


def main(exe: Path) -> None:
    cases = []
    for name, data in accounting_inputs().items():
        for conn, thresholds, highlight in ACCOUNTING_PARAMS:
            ref = run_gtb_accounting(exe, data, thresholds, connectivity=conn, highlight_largest=highlight)
            cases.append({
                "input": name,
                "input_sha256": digest(data),
                "connectivity": conn,
                "thresholds": list(thresholds),
                "highlight_largest": highlight,
                "sha256": {k: digest(ref[k]) for k in ("classes", "ids", "area")},
                "stats": ref["stats"],
            })
            print("accounting", name, conn, thresholds, highlight)
        cases.append({
            "input": name,
            "input_sha256": digest(data),
            "scaled_classes": run_gtb_accounting_scaled_classes(exe, data),
        })
    OUT.write_text(json.dumps({
        "generated": dt.date.today().isoformat(),
        "interpreter": exe.name,
        "sources": ["tests/gtb_idl/gtb_accounting.pro", "tests/gtb_idl/gtb_accounting_scaled_classes.pro"],
        "accounting": cases,
    }, indent=1))


if __name__ == "__main__":
    gdl = Path(sys.argv[1]) if len(sys.argv) > 1 else find_gdl()
    if gdl is None:
        sys.exit("GDL not found: pass its path or set LANDSCHAFT_GDL_EXE")
    main(gdl)
