# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sensitivity against SALib 1.6 and scipy.stats.qmc (tests/data/reference/sensitivity,
tools/make_reference_sensitivity.py, the same files as Terraflux's): Morris's trajectories and indices, Saltelli's
runs and Sobol's indices, the same numbers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import landschaft as ls

REF = Path(__file__).parent / "data" / "reference" / "sensitivity"


@pytest.fixture(scope="module")
def ref():
    return np.load(REF / "sensitivity.npz"), json.loads((REF / "sensitivity.json").read_text(encoding="utf-8"))


def _factors(problem):
    return [(n, lo, hi) for n, (lo, hi) in zip(problem["names"], problem["bounds"])]


@pytest.mark.external_reference
@pytest.mark.parametrize("levels", [4, 6])
def test_morris_is_salibs(ref, levels):
    arr, meta = ref
    d = ls.sensitivity.design("morris", _factors(meta["morris/problem"]), n=10, levels=levels, seed=3)
    np.testing.assert_array_equal(d.values, arr[f"morris/{levels}/x"])
    s = ls.sensitivity.analyze(d, arr[f"morris/{levels}/y"], resamples=50, seed=5)
    for k in ("mu", "mu_star", "sigma", "mu_star_conf"):
        np.testing.assert_array_equal(s.indices[k].to_numpy(), arr[f"morris/{levels}/{k}"], err_msg=k)
    assert s.indices.index.tolist() == ["a", "b", "c", "d", "e"]


@pytest.mark.external_reference
@pytest.mark.parametrize("second", [True, False])
def test_sobol_is_salibs(ref, second):
    arr, meta = ref
    tag = "second" if second else "first"
    d = ls.sensitivity.design("sobol", _factors(meta["sobol/problem"]), n=64, second_order=second, seed=11)
    np.testing.assert_array_equal(d.values, arr[f"sobol/{tag}/x"])
    s = ls.sensitivity.analyze(d, arr[f"sobol/{tag}/y"], resamples=40, seed=13)
    for ours, theirs in (("s1", "S1"), ("s1_conf", "S1_conf"), ("st", "ST"), ("st_conf", "ST_conf")):
        np.testing.assert_array_equal(s.indices[ours].to_numpy(), arr[f"sobol/{tag}/{theirs}"], err_msg=ours)
    if second:
        np.testing.assert_array_equal(s.s2.to_numpy(), arr["sobol/second/S2"])
        np.testing.assert_array_equal(s.s2_conf.to_numpy(), arr["sobol/second/S2_conf"])
    else:
        assert s.s2 is None


def test_evaluate_and_tables():
    d = ls.sensitivity.design("morris", {"a": (0, 1), "b": (-1, 1), "c": (5, 6)}, n=20, seed=1)
    y = ls.sensitivity.evaluate(lambda p: 3 * p[0] + p[1] ** 2, d)
    assert y.shape == (80,)
    s = ls.sensitivity.analyze(d, y)
    assert s.indices.loc["c", "mu_star"] == 0.0
    assert s.indices.loc["a", "mu"] == pytest.approx(3.0)
    # the design as a table with the method, and the results as a DataFrame
    s2 = ls.sensitivity.analyze(d.table, d.table.assign(out=y), method="morris")
    assert s2.indices.equals(s.indices)
    # maps summarised
    m = ls.sensitivity.evaluate(lambda p: np.full((3, 3), p[0]), d, summary=np.mean)
    assert np.allclose(m, d.values[:, 0])
    with pytest.raises(ValueError, match="summary"):
        ls.sensitivity.evaluate(lambda p: np.zeros(3), d)
    with pytest.raises(ValueError, match="low below high"):
        ls.sensitivity.design("sobol", {"a": (1, 0)})
    with pytest.raises(ValueError, match="Saltelli"):
        ls.sensitivity.analyze(ls.sensitivity.design("sobol", {"a": (0, 1), "b": (0, 1)}, n=8, second_order=False),
                               np.arange(32.0), second_order=True)
