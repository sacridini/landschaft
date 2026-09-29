# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading the gdistance references (tools/make_reference_cost.py) and comparing results."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import cost_cases
from conftest import BITWISE

REF = Path(__file__).parent / "data" / "reference" / "cost"


@lru_cache(maxsize=None)
def load(group: str) -> tuple[dict, dict]:
    npz, js = REF / f"{group}.npz", REF / f"{group}.json"
    if not npz.exists():
        pytest.skip(f"no gdistance reference for {group} (run tools/make_reference_cost.py {group})")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(js.read_text(encoding="utf-8"))


def case_ids(group: str) -> list[str]:
    return [k for k, c in cost_cases.CASES.items() if c.group == group]


def triplets(tr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i, j, x) of a transition's matrix, column by column (R's TsparseMatrix order)."""
    m = tr.to_sparse().tocoo()
    order = np.lexsort((m.row, m.col))
    return m.row[order].astype(np.int64), m.col[order].astype(np.int64), m.data[order]


def exact(case: str) -> bool:
    """Whether a case must give R's bits here: x87 maths (see conftest.BITWISE), and on
    longitude/latitude grids the UCRT's hypot of R's geodesics (Windows)."""
    c = cost_cases.CASES[case]
    lonlat = c.raster is not None and "longlat" in (cost_cases.RASTERS[c.raster].crs or "")
    return BITWISE and (sys.platform == "win32" or not lonlat)


def assert_values(ours, ref, tolerance: float, what: str, ulps: float = 0) -> None:
    ours = np.asarray(ours, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert ours.shape == ref.shape, f"{what}: shape {ours.shape} != {ref.shape}"
    if tolerance == 0:
        same = (ours == ref) | (np.isnan(ours) & np.isnan(ref))
        if ulps:
            with np.errstate(invalid="ignore"):
                same |= np.abs(ours - ref) <= ulps * np.spacing(np.abs(ref))
        if not same.all():
            k = np.flatnonzero(~same.ravel())
            raise AssertionError(f"{what}: {k.size} of {ours.size} values differ, e.g. at {k[:5]}: "
                                 f"{ours.ravel()[k[:5]]} vs {ref.ravel()[k[:5]]}")
    else:
        np.testing.assert_allclose(ours, ref, rtol=tolerance, atol=0, equal_nan=True, err_msg=what)


def compare(case: str, result: dict, group: str, tolerance: float = 0.0) -> int:
    """Compare every key of a case's result with the reference; the number of keys compared."""
    arrays, meta = load(group)
    ulps = 0 if exact(case) else 16
    keys = {k.split("/", 1)[1].rsplit("/", 1)[0] if meta.get(k.rsplit("/", 1)[0], {}).get("kind") == "transition"
            else k.split("/", 1)[1] for k in arrays if k.startswith(case + "/")}
    flat = _flatten(result)
    assert set(flat) == keys, f"{case}: keys {sorted(set(flat) ^ keys)} differ"
    for key, value in flat.items():
        full = f"{case}/{key}"
        info = meta[full]
        if info["kind"] == "transition":
            assert value.symmetric == info["symmetric"], f"{full}: symmetric"
            assert value.matrix_values == info["matrix_values"], f"{full}: matrix values"
            i, j, x = triplets(value)
            ri, rj, rx = arrays[full + "/i"], arrays[full + "/j"], arrays[full + "/x"]
            keep = rx != 0  # explicit zeros of R's old method
            ri, rj, rx = ri[keep], rj[keep], rx[keep]
            order = np.lexsort((ri, rj))
            ri, rj, rx = ri[order], rj[order], rx[order]
            assert i.size == ri.size, f"{full}: {i.size} connections vs {ri.size}"
            np.testing.assert_array_equal(i, ri, err_msg=f"{full}: rows")
            np.testing.assert_array_equal(j, rj, err_msg=f"{full}: columns")
            assert_values(x, rx, tolerance, full, ulps)
        else:
            ref = arrays[full]
            if "dim" in info:
                ref = ref.reshape(info["dim"])
            assert_values(value, ref, tolerance, full, ulps)
    return len(flat)


def _flatten(result: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in result.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        else:
            out[key] = v
    return out
