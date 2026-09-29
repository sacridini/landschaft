# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.cost randomised shortest paths against gdistance (and unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

import cost_cases
import cost_reference
import landschaft as ls


@pytest.mark.parametrize("case", cost_reference.case_ids("rsp"))
def test_matches_gdistance(case):
    c = cost_cases.CASES[case]
    result = c.py(cost_cases.RASTERS)
    assert cost_reference.compare(case, result, "rsp", c.tolerance) > 0


def _dense_rsp(tr, i, j, theta, method, net):
    """rSPDistance by the book, dense (Saerens et al. 2009)."""
    T = tr.to_sparse().toarray()
    C = np.where(T != 0, 1 / np.where(T != 0, T, 1), 0)
    P = T / T.sum(axis=1, keepdims=True)
    W = np.exp(-theta * C) * P if method == 1 else np.where(P > 0, P ** theta, 0)
    Wj = W.copy()
    Wj[j] = 0
    Z = np.linalg.inv(np.eye(len(T)) - Wj)
    N = Z[i][:, None] * Wj * Z[:, j][None, :] / Z[i, j]
    if net:
        N = np.maximum(N - N.T, 0)
    return (C * N).sum()


@pytest.mark.parametrize("method", [1, 2])
@pytest.mark.parametrize("net", [True, False])
@pytest.mark.parametrize("symmetric", [True, False])
def test_rsp_distance_matches_dense_formula(method, net, symmetric):
    rng = np.random.default_rng(7)
    r = rng.uniform(1, 4, (5, 6))
    if symmetric:
        tr = 1 / ls.cost.transition(r, "mean", 8)
    else:
        tr = ls.cost.transition(r, lambda x: 1 / (0.5 + x[1]), 8, symmetric=False)
    pts = np.array([(0.5, 4.5), (5.5, 0.5), (2.5, 2.5)])
    cells = ls.cost._points.cells_of_xy(pts, tr.grid)
    d = ls.cost.rsp_distance(tr, pts, pts, 0.8, method=method, total_net="net" if net else "total")
    for a in range(3):
        for b in range(3):
            if a != b:
                assert d[a, b] == pytest.approx(_dense_rsp(tr, cells[a], cells[b], 0.8, method, net), rel=1e-11)


def test_theta_zero_is_the_random_walk():
    rng = np.random.default_rng(8)
    tr = 1 / ls.cost.transition(rng.uniform(1, 3, (12, 14)), "mean", 8)
    o, g = (0.5, 6.5), (13.5, 3.5)
    rw = ls.cost.passage(tr, o, g, gdistance_compat=False)
    rsp = ls.cost.passage(tr, o, g, theta=0)
    np.testing.assert_allclose(rsp, rw, rtol=1e-10, atol=1e-13)


def test_high_theta_concentrates_on_the_least_cost_path():
    r = np.ones((9, 9))
    r[1:8, 4] = 5  # a costly band with a cheap gap at the top row and the bottom row
    r[0, 4] = 1
    tr = 1 / ls.cost.transition(r, "mean", 4)
    low = ls.cost.passage(tr, (0.5, 4.5), (8.5, 4.5), theta=0.01)
    high = ls.cost.passage(tr, (0.5, 4.5), (8.5, 4.5), theta=15)
    path = ls.cost.shortest_path(tr, (0.5, 4.5), (8.5, 4.5), output="cells")[0]
    on = np.zeros(81, bool)
    on[path] = True
    assert high.ravel()[on].min() > 0.99 and high.ravel()[~on].max() < 0.01
    assert low.ravel()[~on].max() > 0.05


def test_rsp_distance_between_least_cost_and_commute():
    rng = np.random.default_rng(9)
    tr = ls.cost.geo_correction(1 / ls.cost.transition(rng.uniform(1, 3, (10, 10)), "mean", 8))
    pts = [(0.5, 0.5), (9.5, 9.5)]
    lcd = ls.cost.cost_distance(tr, pts)[0, 1]
    d = [ls.cost.rsp_distance(tr, pts[:1], pts[1:], th)[0, 0] for th in (0.01, 0.5, 5, 20)]
    assert all(a >= b for a, b in zip(d, d[1:]))  # cheaper as theta grows
    assert d[-1] == pytest.approx(lcd, rel=1e-3)


def test_passage_transition_output_and_checks():
    tr = ls.cost.transition(np.ones((4, 4)), "mean", 8)
    t = ls.cost.passage(tr, (0.5, 0.5), (3.5, 3.5), theta=1, output="transition")
    assert isinstance(t, ls.cost.Transition) and not t.symmetric
    with pytest.raises(ValueError, match="theta"):
        ls.cost.passage(tr, (0.5, 0.5), (3.5, 3.5), theta=25)
    with pytest.raises(ValueError, match="method"):
        ls.cost.rsp_distance(tr, (0.5, 0.5), (3.5, 3.5), 1, method=3)
