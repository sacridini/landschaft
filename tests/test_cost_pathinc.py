# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.cost path overlap (pathInc), barriers and areas against gdistance (and unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

import cost_cases
import cost_reference
import landschaft as ls


@pytest.mark.parametrize("case", cost_reference.case_ids("pathinc"))
def test_matches_gdistance(case):
    c = cost_cases.CASES[case]
    result = c.py(cost_cases.RASTERS)
    assert cost_reference.compare(case, result, "pathinc", c.tolerance) > 0


def _setup():
    rng = np.random.default_rng(10)
    tr = ls.cost.geo_correction(1 / ls.cost.transition(rng.uniform(1, 3, (12, 14)), "mean", 8))
    locs = np.array([(0.5, 0.5), (13.5, 0.5), (0.5, 11.5), (13.5, 11.5)])
    return tr, (7.5, 6.5), locs


@pytest.mark.parametrize("theta", [None, 0.5])
def test_from_to_equals_the_square_matrix(theta):
    tr, origin, locs = _setup()
    square = ls.cost.path_inc(tr, origin, locs, theta=theta)
    ft = ls.cost.path_inc(tr, origin, locs[:2], locs[2:], theta=theta)
    for name in ("overlap", "nonoverlap"):
        np.testing.assert_allclose(ft[name], square[name][:2, 2:], rtol=1e-12)


def test_callable_functions_match_the_builtins():
    tr, origin, locs = _setup()
    a = ls.cost.path_inc(tr, origin, locs)
    b = ls.cost.path_inc(tr, origin, locs, functions=[ls.cost.overlap, ls.cost.nonoverlap])
    for name in ("overlap", "nonoverlap"):
        np.testing.assert_allclose(b[name], a[name], rtol=1e-12)
    c = ls.cost.path_inc(tr, origin, locs, functions=[lambda x, y: np.abs(x - y)])
    assert set(c) == {"function1"} and c["function1"].shape == (4, 4)


def test_overlap_with_itself_and_zero_weights():
    tr, origin, locs = _setup()
    out = ls.cost.path_inc(tr, origin, [locs[0], locs[0], locs[1]])
    assert out["overlap"][0, 1] > out["overlap"][0, 2]  # the same route overlaps fully
    assert out["nonoverlap"][0, 1] < out["nonoverlap"][0, 2]
    zero = ls.cost.path_inc(tr, origin, locs, weight=tr * 1e300 * 1e300)  # conductance inf: weight 0
    assert np.all(zero["overlap"] == 0)


def test_grids_with_missing_cells_and_layers():
    r = np.ones((10, 10))
    r[4, 2:8] = np.nan
    tr = ls.cost.transition(r, "mean", 8)
    bars = ls.cost.barriers(np.tile(np.arange(10.0), (10, 1)), 8, breaks=[2.5, 6.5])
    out = ls.cost.path_inc(tr, (5.5, 1.5), [(0.5, 9.5), (9.5, 9.5), (5.5, 5.5)], weight=bars)  # y 5.5: row 4
    assert out["overlap"].shape == (2, 3, 3)
    assert np.isnan(out["overlap"][:, 2]).all()  # on a missing cell
    assert (out["overlap"][:, :2, :2] >= 0).all()


def test_transition_dispatches_barriers_and_areas():
    r = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert len(ls.cost.transition(r, "barriers", 4)) == 3
    assert len(ls.cost.transition(r, "barriers", 4, symmetric=False)) == 6
    assert len(ls.cost.transition(r, "barriers", 4, breaks=[1.5])) == 1
    assert len(ls.cost.transition(r, "areas", 4)) == 3
