# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden landscape-metric outputs with the R package landscapemetrics.

Usage::

    python tools/make_reference_lsm.py

Needs Rscript with landscapemetrics and terra (see ``tests/lsm_reference.py``
for ``LANDSCHAFT_RSCRIPT`` / ``LANDSCHAFT_R_LIBS``). Writes the input rasters
to ``tests/data/lsm/*.tif``, every value of ``calculate_lsm`` for each option
set to ``tests/data/reference/lsm/<case>__<options>.csv.gz`` and a
``manifest.json``. The test-suite compares landschaft against these files, so
it runs anywhere without R.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import lsm_reference  # noqa: E402
import lsm_scenes  # noqa: E402

INPUTS = ROOT / "tests" / "data" / "lsm"
OUT = ROOT / "tests" / "data" / "reference" / "lsm"

OPTION_SETS = {
    "default": {},
    "d4": {"directions": 4},
    "boundary": {"count_boundary": True, "consider_boundary": True, "edge_depth": 2},
    "complexity": {"neighbourhood": 8, "ordered": False, "base": "log", "classes_max": 20, "cell_center": True},
}

# landscapemetrics' own example data, cropped so the reference stays small.
EXPORT_LM_DATA = r"""
suppressPackageStartupMessages({library(terra); library(landscapemetrics)})
out <- commandArgs(trailingOnly = TRUE)[[1]]
save <- function(r, name) {
    if (!inherits(r, "SpatRaster")) r <- terra::rast(r)
    levels(r) <- NULL
    terra::writeRaster(r, file.path(out, paste0(name, ".tif")), datatype = "INT4S",
                       NAflag = -2147483648, overwrite = TRUE)
}
save(landscapemetrics::landscape, "lm_landscape")
aug <- terra::rast(landscapemetrics::augusta_nlcd)
save(aug[101:220, 201:360, drop = FALSE], "augusta")
pod <- terra::rast(landscapemetrics::podlasie_ccilc)
save(pod[101:200, 151:270, drop = FALSE], "podlasie")
"""


def synthetic_cases() -> dict[str, tuple]:
    cases = {
        "synthetic_na": (
            lsm_scenes.clustered((45, 60), [-5, 0, 7, 1000], seed=11, sigma=2.5, na_fraction=0.12),
            (30.0, 20.0),
        ),
        "salt_pepper": (lsm_scenes.salt_and_pepper((25, 30), [1, 2, 3], seed=7, na_fraction=0.05), (10.0, 10.0)),
    }
    for name, data in lsm_scenes.small_cases().items():
        cases[name] = (data, (1.0, 1.0))
    return cases


def main() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set LANDSCHAFT_RSCRIPT")
    INPUTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "export.R"
        script.write_text(EXPORT_LM_DATA)
        subprocess.run([str(rscript), str(script), str(INPUTS)], check=True, env=lsm_reference.r_environment())
    for name, (data, res) in synthetic_cases().items():
        lsm_reference.write_landscape(INPUTS / f"{name}.tif", data, res)

    version = subprocess.run(
        [str(rscript), "-e", "cat(as.character(packageVersion('landscapemetrics')))"],
        capture_output=True, text=True, env=lsm_reference.r_environment(), check=True,
    ).stdout.strip()

    files = {}
    for tif in sorted(INPUTS.glob("*.tif")):
        for opt_name, options in OPTION_SETS.items():
            df = lsm_reference.run_tif(rscript, tif, **options)
            target = OUT / f"{tif.stem}__{opt_name}.csv.gz"
            df.to_csv(target, index=False, float_format="%.17g", compression={"method": "gzip", "mtime": 0})
            files[target.name] = {"input": tif.name, "options": options, "rows": len(df)}
            print(f"{target.name}: {len(df)} values")

    manifest = {
        "generated": dt.date.today().isoformat(),
        "landscapemetrics": version,
        "option_sets": OPTION_SETS,
        "files": files,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
