# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.cost least-cost surfaces, distances and paths against gdistance (and unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

import cost_cases
import cost_reference
import landschaft as ls

pytestmark = pytest.mark.filterwarnings("ignore:transition function gives negative values")


@pytest.mark.parametrize("case", cost_reference.case_ids("lcp"))
@pytest.mark.external_reference
def test_matches_gdistance(case):
    c = cost_cases.CASES[case]
    result = c.py(cost_cases.RASTERS)
    assert cost_reference.compare(case, result, "lcp", c.tolerance) > 0


def _grid():
    r = np.ones((12, 15))
    r[4:8, 6] = np.nan  # a wall with gaps above and below
    return ls.cost.geo_correction(ls.cost.transition(r, "mean", 8, extent=(0, 15, 0, 12)))


def test_acc_cost_from_mask_equals_points():
    tr = _grid()
    mask = np.zeros((12, 15), dtype=bool)
    mask[1, 1] = mask[10, 13] = True
    a = ls.cost.acc_cost(tr, mask)
    b = ls.cost.acc_cost(tr, [(1.5, 10.5), (13.5, 1.5)])
    np.testing.assert_array_equal(a, b)
    assert a[1, 1] == 0 and np.isinf(a[5, 6])  # the wall cells are unreachable


def test_cost_distance_consistent_with_acc_cost():
    tr = _grid()
    pts = np.array([(1.5, 10.5), (13.5, 1.5), (7.5, 6.5)])
    d = ls.cost.cost_distance(tr, pts)
    acc = ls.cost.acc_cost(tr, pts[0])
    cells = ls.cost._points.cells_of_xy(pts, tr.grid)
    np.testing.assert_allclose(d[0], acc.ravel()[cells], rtol=1e-14)
    assert np.allclose(d, d.T)


def test_points_outside_are_nan_with_warning():
    tr = _grid()
    with pytest.warns(RuntimeWarning, match="outside"):
        d = ls.cost.cost_distance(tr, [(1.5, 10.5), (100, 100)])
    assert np.isnan(d[1]).all() and np.isnan(d[:, 1]).all() and d[0, 0] == 0


def test_shortest_path_outputs():
    tr = _grid()
    goals = [(13.5, 1.5), (13.5, 10.5)]
    cells = ls.cost.shortest_path(tr, (1.5, 5.5), goals, output="cells")
    each = ls.cost.shortest_path(tr, (1.5, 5.5), goals, output="transitions")
    both = ls.cost.shortest_path(tr, (1.5, 5.5), goals)
    total = sum(t.to_sparse() for t in each)
    assert abs(total - both.to_sparse()).max() < 1e-15
    for p, t in zip(cells, each):
        assert t.n_connections == 2 * (p.size - 1)
        assert p[0] == 6 * 15 + 1  # y = 5.5 is row 6 of 12
    lines = ls.cost.shortest_path(tr, (1.5, 5.5), goals, output="lines")
    geoms = lines.geometry if hasattr(lines, "geometry") else lines
    assert [len(g.coords) for g in geoms] == [p.size for p in cells]


def test_asymmetric_costs_differ_by_direction():
    r = np.tile(np.arange(10, dtype=float), (3, 1))  # rising to the east
    up = ls.cost.transition(r, lambda x: 1 / (1 + np.maximum(x[1] - x[0], 0)), 4, symmetric=False)
    d = ls.cost.cost_distance(up, [(0.5, 1.5), (9.5, 1.5)])
    assert d[0, 1] > d[1, 0] == 9.0


def test_rasters_keep_the_georeferencing_of_the_input():
    dem = ls.load_raster(ls.cost.example_data("maungawhau"))
    tr = ls.cost.geo_correction(1 / ls.cost.transition(dem, "mean", 8))
    cost = ls.cost.acc_cost(tr, (2667670, 6479000))
    assert cost.rio.crs == dem.rio.crs and cost.rio.transform() == dem.rio.transform()
