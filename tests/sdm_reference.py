# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading the references of the R packages behind ls.sdm (tools/make_reference_sdm.py)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import sdm_cases

REF = Path(__file__).parent / "data" / "reference" / "sdm"


@lru_cache(maxsize=None)
def load(group: str) -> tuple[dict, dict]:
    npz, js = REF / f"{group}.npz", REF / f"{group}.json"
    if not npz.exists():
        pytest.skip(f"no R reference for {group} (run tools/make_reference_sdm.py {group})")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(js.read_text(encoding="utf-8"))


def case_ids(group: str) -> list[str]:
    return sdm_cases.case_ids(group)


def result(group: str, case: str) -> dict:
    """The R result of a case: key -> float64 array (matrices in their shape) or list of str."""
    arrays, info = load(group)
    prefix = f"{case}/"
    out = {}
    for key, meta in info["arrays"].items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if meta["kind"] == "text":
            out[name] = list(meta["values"])
        else:
            v = arrays[key]
            if meta["kind"] in ("matrix", "raster") and "dim" in meta:
                v = v.reshape(meta["dim"])
            out[name] = v
    if not out:
        pytest.skip(f"case {case} missing from the {group} reference")
    return out


def assert_same(ours, ref, key: str = "", rtol: float = 0.0, atol: float = 0.0) -> None:
    """Equal values (NaN = NA matches NaN); bitwise unless a tolerance is given."""
    if isinstance(ref, list):
        assert [None if o is None else str(o) for o in ours] == ref, key
        return
    a = np.asarray(ours, dtype=float).reshape(np.shape(ref))
    b = np.asarray(ref, dtype=float)
    if rtol == 0 and atol == 0:
        same = (a == b) | (np.isnan(a) & np.isnan(b))
    else:
        same = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    if not same.all():
        bad = np.flatnonzero(~same.ravel())
        raise AssertionError(f"{key}: {bad.size} of {same.size} differ; first at {bad[:5]}: "
                             f"ours {a.ravel()[bad[:5]]} vs R {b.ravel()[bad[:5]]}")
