# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate the golden outputs of R's maxnet for ls.sdm.MaxNet.

Usage::

    python tools/make_reference_maxnet.py

Runs ``tools/maxnet_reference.R`` (maxnet 0.1.4 with glmnet 4.1-10) and
stores its cases (inputs, model, predictions) in
``tests/data/reference/sdm/maxnet.npz`` (``<case>/<key>`` arrays) and
``maxnet.json`` (kinds, shapes, text results and the package versions).
Needs Rscript and the library of ``tools/make_reference_sdm.py``. The
test-suite (``tests/test_sdm_maxnet.py``) reads the stored files, so it runs
anywhere without R.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import lsm_reference  # noqa: E402
from make_reference_sdm import r_environment  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference" / "sdm"
R_SCRIPT = ROOT / "tools" / "maxnet_reference.R"
DTYPES = {"f8": "<f8", "i4": "<i4", "i2": "<i2"}


def main() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    with tempfile.TemporaryDirectory(prefix="maxnet_ref_") as tmp:
        tmp = Path(tmp)
        subprocess.run([str(rscript), str(R_SCRIPT), str(tmp)], check=True, env=r_environment())
        result = json.loads((tmp / "manifest.json").read_text(encoding="utf-8"))
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}
        for item in result["manifest"]:
            key = f"{item['case']}/{item['key']}"
            if item["kind"] == "text":
                meta[key] = {"kind": "text", "values": item["values"]}
                continue
            a = np.fromfile(tmp / item["file"], dtype=DTYPES[item["kind"]])
            if "dim" in item:
                a = a.reshape(item["dim"])
            # the same data in several cases: stored once
            same = next((k for k, b in arrays.items() if b.dtype == a.dtype and b.shape == a.shape
                         and np.array_equal(b, a)), None)
            if same is not None:
                meta[key] = {"kind": item["kind"], "same_as": same}
                continue
            arrays[key] = a
            meta[key] = {"kind": item["kind"]}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "maxnet.npz", **arrays)
    (OUT / "maxnet.json").write_text(json.dumps({"versions": result["versions"], "r": result["r"], "arrays": meta},
                                                indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    cases = sorted({k.split("/", 1)[0] for k in meta})
    print(f"maxnet: {len(cases)} cases, {len(meta)} results -> {(OUT / 'maxnet.npz').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
