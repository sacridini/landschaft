# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.cost random walks against gdistance (and unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

import cost_cases
import cost_reference
import landschaft as ls


@pytest.mark.parametrize("case", cost_reference.case_ids("rw"))
def test_matches_gdistance(case):
    c = cost_cases.CASES[case]
    result = c.py(cost_cases.RASTERS)
    assert cost_reference.compare(case, result, "rw", c.tolerance) > 0


def test_commute_is_volume_times_effective_resistance():
    # a ladder 2 x n with unit conductances: compare with a dense pseudo-inverse
    rng = np.random.default_rng(4)
    r = rng.uniform(1, 3, (4, 5))
    tr = 1 / ls.cost.transition(r, "mean", 8)
    L = -tr.to_sparse().toarray()
    np.fill_diagonal(L, -L.sum(axis=1))
    Lp = np.linalg.pinv(L)
    pts = [(0.5, 3.5), (4.5, 0.5), (2.5, 2.5)]
    cells = ls.cost._points.cells_of_xy(np.array(pts), tr.grid)
    R = np.diag(Lp)[cells][:, None] + np.diag(Lp)[cells][None, :] - 2 * Lp[np.ix_(cells, cells)]
    vol = tr.to_sparse().sum()
    np.testing.assert_allclose(ls.cost.commute_distance(tr, pts), R * vol, rtol=1e-11, atol=1e-9)


def test_commute_separate_parts_and_missing_cells():
    r = np.ones((3, 7))
    r[:, 3] = np.nan  # two parts
    tr = ls.cost.transition(r, "mean", 4)
    d = ls.cost.commute_distance(tr, [(0.5, 1.5), (2.5, 1.5), (5.5, 1.5), (3.5, 1.5)])
    assert np.isfinite(d[0, 1]) and np.isinf(d[0, 2]) and np.isinf(d[1, 2])
    assert np.isnan(d[3]).all() and np.isnan(d[:, 3]).all()
    # the volume of each part: the same as the part alone
    alone = ls.cost.commute_distance(ls.cost.transition(np.ones((3, 3)), "mean", 4), [(0.5, 1.5), (2.5, 1.5)])
    assert d[0, 1] == pytest.approx(alone[0, 1], rel=1e-12)


def test_commute_needs_a_symmetric_transition():
    tr = ls.cost.transition(np.arange(9.0).reshape(3, 3) + 1, "to", 4, symmetric=False)
    with pytest.raises(ValueError, match="symmetric"):
        ls.cost.commute_distance(tr, [(0.5, 0.5), (2.5, 2.5)])


def test_passages_conserve_the_unit_current():
    rng = np.random.default_rng(5)
    r = rng.uniform(1, 4, (15, 18))
    tr = 1 / ls.cost.transition(r, "mean", 8)
    p = ls.cost.passage(tr, (0.5, 7.5), (17.5, 7.5), gdistance_compat=False)
    # every column between the two ends carries the whole current (half counted twice)
    assert p.max() == 1.0 and (p >= 0).all()
    exact = ls.cost.passage(tr, (0.5, 7.5), (17.5, 7.5), gdistance_compat=False)
    quirk = ls.cost.passage(tr, (0.5, 7.5), (17.5, 7.5))
    far = np.ones_like(p, dtype=bool)
    far[-3:, -3:] = False  # the ground: the last cell and its neighbours
    np.testing.assert_allclose(quirk[far], exact[far], rtol=1e-12, atol=1e-15)


def test_passage_rejects_unsupported_options():
    tr = ls.cost.transition(np.ones((3, 3)), "mean", 4)
    with pytest.raises(ValueError):
        ls.cost.passage(tr, (0.5, 0.5), (2.5, 2.5), total_net="total")
