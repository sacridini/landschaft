# SPDX-License-Identifier: GPL-3.0-or-later
"""The reference harness of ls.simulate: generators, inputs reaching every engine, example data."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import simulate_cases
import simulate_reference

import landschaft as ls
from landschaft import _core

SEEDS = (0, 1, 42, 123456789, -5, 2**40 + 17)


@pytest.mark.parametrize("seed", SEEDS)
def test_drand48_is_grass(seed):
    ref = simulate_reference.result("rng", "rng_grass")
    tag = str(seed).replace("-", "m")
    np.testing.assert_array_equal(_core.sim_drand48(seed, 1000), ref[f"drand48_{tag}"])
    np.testing.assert_array_equal(_core.sim_lrand48(seed, 1000), ref[f"lrand48_{tag}"])
    np.testing.assert_array_equal(_core.sim_lrand48(seed, 1000, signed=True), ref[f"mrand48_{tag}"])


@pytest.mark.skipif(sys.platform != "win32", reason="the Microsoft C runtime")
@pytest.mark.parametrize("seed", [1, 42, 101, 2**32 - 1])
def test_crt_rand_is_ucrt(seed):
    import ctypes

    crt = ctypes.cdll.ucrtbase
    crt.srand(ctypes.c_uint(seed))
    ref = [crt.rand() for _ in range(5000)]
    assert _core.sim_crt_rand(seed, 5000).tolist() == ref


def test_crt_rand_known_values():
    # srand(1) is the C runtime's default state
    assert _core.sim_crt_rand(1, 5).tolist() == [41, 18467, 6334, 26500, 19169]


@pytest.mark.parametrize("case", ["smoke_r_crosstab", "smoke_molusce_woe"])
def test_inputs_reach_the_reference_unchanged(case):
    ref = simulate_reference.result("smoke", case)
    ours = simulate_cases.run_python(case)
    assert set(ours) == set(ref)
    for key, value in ours.items():
        simulate_reference.assert_same(value, ref[key], key)


def test_clumondo_run_meets_the_demand():
    ref = simulate_reference.result("smoke", "smoke_clumondo")
    job = simulate_cases.clumondo_smoke_job()
    demand = [float(x) for x in job["files"]["demand.in1"].split()[1:]]
    np.testing.assert_array_equal(ref["cov_all.0.asc"], job["files"]["cov_all.0.asc"])
    for year in range(1, 4):
        cov = ref[f"cov_all.{year}.asc"]
        supply = (cov == 1).sum() * 1.0 + (cov == 2).sum() * 3.0
        assert abs(supply - demand[year]) / demand[year] < 0.01
        same = cov == ref[f"cov_all.{year - 1}.asc"]
        np.testing.assert_array_equal(ref[f"age.{year}.asc"][same], ref[f"age.{year - 1}.asc"][same] + 1)


def test_grass_development_pressure():
    ref = simulate_reference.result("smoke", "smoke_grass_devpressure")["pressure.tif"]
    inner = ref[4:-4, 4:-4]
    assert np.isfinite(inner).all() and (inner >= 0).all()
    assert ref[12, 15] > ref[2, 48]  # higher in the towns


def test_dinamica_transition_rates():
    ref = simulate_reference.result("smoke", "smoke_dinamica_matrix")
    a, b = (x.ravel() for x in simulate_cases.small_change())
    for i, j, rate in zip(ref["single.csv/From*"], ref["single.csv/To*"], ref["single.csv/Rate"]):
        expected = np.sum((a == i) & (b == j)) / np.sum(a == i)
        assert rate == pytest.approx(expected, rel=1e-14)


def test_example_data():
    lu = ls.load_raster(ls.simulate.example_data("sibuyan_landuse"))
    assert lu.shape == (2, 108, 128) and list(lu.band.values) == ["1997", "2011"]
    assert set(np.unique(lu.values[0])) == {1, 2, 3, 4, 5, 255}
    pie = ls.load_raster(ls.simulate.example_data("pie_landuse"))
    assert pie.shape == (3, 434, 497) and pie.rio.crs.to_epsg() == 26986
    drivers = ls.load_raster(ls.simulate.example_data("sibuyan_drivers"))
    assert drivers.shape[0] == 13
    demand = ls.load_table(ls.simulate.example_data("sibuyan_demand"))
    assert sorted(set(demand["scenario"])) == ["demand1", "demand2", "demand3", "demand4"]
    with pytest.raises(ValueError, match="unknown example"):
        ls.simulate.example_data("nope")
