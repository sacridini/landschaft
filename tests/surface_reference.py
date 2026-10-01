# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading the geodiv / spatialEco references (tools/make_reference_surface.py) and comparing results."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import surface_cases
from conftest import BITWISE

REF = Path(__file__).parent / "data" / "reference" / "surface"


@lru_cache(maxsize=None)
def load(group: str) -> tuple[dict, dict]:
    npz, js = REF / f"{group}.npz", REF / f"{group}.json"
    if not npz.exists():
        pytest.skip(f"no geodiv reference for {group} (run tools/make_reference_surface.py {group})")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(js.read_text(encoding="utf-8"))["arrays"]


def case_ids(group: str) -> list[str]:
    return [k for k, c in surface_cases.CASES.items() if c.group == group]


def assert_values(ours, ref, tolerance, what: str, ulps: float = 0) -> None:
    ours = np.asarray(ours, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if ours.shape != ref.shape and ours.size == ref.size and ours.ndim and ref.ndim:  # rasters as vectors
        ours, ref = ours.ravel(), ref.ravel()
    assert ours.shape == ref.shape, f"{what}: shape {ours.shape} != {ref.shape}"
    if isinstance(tolerance, tuple) and tolerance and tolerance[0] == "ulps":  # ("ulps", n): n units in the last place
        with np.errstate(invalid="ignore"):
            ok = (ours == ref) | (np.isnan(ours) & np.isnan(ref)) | (np.abs(ours - ref) <= tolerance[1] * np.spacing(
                np.abs(ref)))
        assert ok.all(), f"{what}: {int((~ok).sum())} values differ by more than {tolerance[1]} ulp"
        return
    if isinstance(tolerance, tuple) and len(tolerance) == 3:  # angles: (rtol, atol, period), compared around
        rtol, atol, period = tolerance
        d = np.abs(ours - ref)
        d = np.minimum(d, period - d)
        ok = (np.isnan(ours) & np.isnan(ref)) | (d <= atol * period + rtol * np.abs(ref))
        assert ok.all(), f"{what}: {int((~ok).sum())} angles differ"
        return
    if isinstance(tolerance, tuple):  # (rtol, atol relative to the largest |value|)
        rtol, atol = tolerance
        scale = np.nanmax(np.abs(ref)) if np.isfinite(ref).any() else 1.0
        np.testing.assert_allclose(ours, ref, rtol=rtol, atol=atol * scale, equal_nan=True, err_msg=what)
        return
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
    ulps = 0 if BITWISE else 16
    keys = {k.split("/", 1)[1] for k in arrays if k.startswith(case + "/")}
    flat = _flatten(result)
    assert set(flat) == keys, f"{case}: keys {sorted(set(flat) ^ keys)} differ"
    for key, value in flat.items():
        full = f"{case}/{key}"
        ref = arrays[full]
        info = meta[full]
        if "dim" in info:
            ref = ref.reshape(info["dim"])
        elif np.ndim(value) == 0:
            ref = ref.reshape(())
        tol = tolerance.get(key, 0.0) if isinstance(tolerance, dict) else tolerance
        assert_values(value, ref, tol, full, ulps)
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
