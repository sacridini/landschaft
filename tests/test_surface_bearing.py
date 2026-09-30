# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.surface bearing area curve parameters against geodiv (and unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls
import surface_cases
import surface_reference


@pytest.mark.parametrize("case", surface_reference.case_ids("bearing"))
def test_bearing_matches_geodiv(case):
    c = surface_cases.CASES[case]
    assert surface_reference.compare(case, c.py(surface_cases.RASTERS), "bearing", c.tolerance) > 0


def test_curve_and_heights_by_definition():
    rng = np.random.default_rng(4)
    z = np.round(rng.normal(size=(30, 40)), 2)
    z[rng.random(z.shape) < 0.05] = np.nan
    v = z[~np.isnan(z)]
    curve = ls.surface.bearing_area(z)
    heights = np.unique(v)[::-1]
    np.testing.assert_array_equal(curve["height"], heights)
    share = np.array([(v >= h).sum() for h in heights]) / v.size
    np.testing.assert_array_equal(curve["bearing"], share)
    probs = [0, 0.05, 0.37, 0.5, 1]
    expected = np.quantile(v, 1 - np.array(probs))  # numpy's default is R's type 7
    np.testing.assert_allclose(ls.surface.height_ba(z, probs), expected, rtol=1e-13)
    assert ls.surface.height_ba(z, 0.5) == ls.surface.height_ba(z, [0.5])[0]
    assert ls.surface.sdc(z, 0.2, 0.7) == pytest.approx(
        ls.surface.height_ba(z, 0.2) - ls.surface.height_ba(z, 0.7), rel=1e-15)
    assert ls.surface.sbi(z) == pytest.approx(ls.surface.sq(z) / ls.surface.height_ba(z, 0.05), rel=1e-15)


def test_core_line_of_a_uniform_surface():
    # heights spread evenly: the bearing area curve is a straight line, all core
    z = np.linspace(0.0, 1.0, 10000).reshape(100, 100)
    assert ls.surface.sk(z) == pytest.approx(1.0, abs=1e-3)
    assert ls.surface.spk(z) == pytest.approx(0.0, abs=2e-3)
    assert ls.surface.svk(z) == pytest.approx(0.0, abs=2e-3)
    # heavy tails: peaks and valleys beyond the core
    tails = np.concatenate([np.linspace(-5, -1, 500), np.linspace(-1, 1, 9000), np.linspace(1, 5, 500)])
    t = tails.reshape(100, 100)
    assert ls.surface.spk(t) > 1 and ls.surface.svk(t) > 1
    # the core line spans the whole curve: its slope, 2 over 90 % of the cells, from 0 to 1
    assert ls.surface.sk(t) == pytest.approx(2 / 0.9, rel=0.01)


def test_bearing_errors_and_table():
    z = surface_cases.RASTERS["rand"].values
    with pytest.raises(ValueError, match="between 0 and 1"):
        ls.surface.height_ba(z, 1.5)
    with pytest.raises(ValueError, match="greater than low"):
        ls.surface.sdc(z, 0.8, 0.1)
    table = ls.surface.metrics(z, ["sbi", "svi", "sci", "sk", "spk", "svk"])
    for name in table.index:
        assert table[name] == getattr(ls.surface, name)(z)
