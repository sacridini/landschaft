# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading the Zonation 5 references (tools/make_reference_prioritize.py)."""

from __future__ import annotations

import json
import lzma
import shlex
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import prioritize_cases

REF = Path(__file__).parent / "data" / "reference" / "prioritize"


@lru_cache(maxsize=None)
def load(group: str) -> tuple[dict, dict]:
    npz, js = REF / f"{group}.npz", REF / f"{group}.json"
    if not npz.exists():
        pytest.skip(f"no Zonation 5 reference for {group} (run tools/make_reference_prioritize.py {group})")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(js.read_text(encoding="utf-8"))["cases"]


def case_ids(group: str) -> list[str]:
    return [k for k, c in prioritize_cases.CASES.items() if c.group == group]


def table_text(arrays: dict, key: str) -> str:
    """The text of a stored curve table (lzma-compressed CSV)."""
    return lzma.decompress(arrays[key].tobytes()).decode("utf-8")


def parse_table(text: str) -> tuple[list[str], list[list[str]]]:
    """Header names and rows (as strings) of a Zonation table."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return shlex.split(lines[0]), [ln.split() for ln in lines[1:]]


def run_case(name: str, tmp: Path, **overrides):
    """Run a case with ls.prioritize.zonation on the rasters written as the harness writes them."""
    import sys

    import landschaft as ls

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_reference_prioritize import write_raster

    case = prioritize_cases.CASES[name]
    paths = {}

    def path(r):
        if r not in paths:
            paths[r] = tmp / f"{r}.tif"
            if not paths[r].exists():
                write_raster(paths[r], prioritize_cases.RASTERS[r])
        return paths[r]

    def raster(v):
        if isinstance(v, str):
            return path(v)
        if isinstance(v, tuple):
            return (v[0], path(v[1]))
        return v

    kwargs = {}
    for k, v in case.params.items():
        if k in ("cost", "hierarchic_mask", "analysis_area", "solution"):
            v = raster(v)
        elif k in ("condition", "retention"):
            v = [raster(x) for x in v]
        elif k in ("interaction", "interaction2"):
            import dataclasses

            v = [None if x is None else dataclasses.replace(x, resource=path(x.resource)) for x in v]
        kwargs[k] = v
    kwargs.update(overrides)
    return ls.prioritize.zonation([path(r) for r in case.layers], **kwargs)
