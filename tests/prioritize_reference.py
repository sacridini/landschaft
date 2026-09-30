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
