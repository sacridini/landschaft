# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.cost transitions against gdistance (and unit checks)."""

from __future__ import annotations

import numpy as np
import pytest

import cost_cases
import cost_reference
import landschaft as ls

# gdistance warns too: differences and ratios are negative for half the connections.
pytestmark = pytest.mark.filterwarnings("ignore:transition function gives negative values")


@pytest.mark.parametrize("case", cost_reference.case_ids("transition"))
@pytest.mark.external_reference
def test_matches_gdistance(case):
    c = cost_cases.CASES[case]
    result = c.py(cost_cases.RASTERS)
    assert cost_reference.compare(case, result, "transition", c.tolerance) > 0


def test_geodesic_matches_pyproj():
    pyproj = pytest.importorskip("pyproj")
    rng = np.random.default_rng(1)
    lon1, lon2 = rng.uniform(-180, 180, (2, 500))
    lat1, lat2 = rng.uniform(-90, 90, (2, 500))
    ours = ls._core.cost_geodesic(lon1, lat1, lon2, lat2)
    _, _, theirs = pyproj.Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12)


def test_mingw_cbrt():
    x = np.array([8.0, -27.0, 2.0, 1e-300, 3.5e200, 0.1])
    ours = np.array([ls._core.cost_mingw_cbrt(v) for v in x])
    np.testing.assert_allclose(ours, np.cbrt(x), rtol=2e-16)


def test_layouts_and_sparse():
    r = np.arange(12, dtype=float).reshape(3, 4) + 1
    for d in (4, 8, 16):
        sym = ls.cost.transition(r, "mean", d)
        asym = ls.cost.transition(r, "mean", d, symmetric=False)
        a, b = sym.to_sparse(), asym.to_sparse()
        assert (a != a.T).nnz == 0
        assert (a != b).nnz == 0
        assert sym.n_connections == asym.n_connections == a.nnz
        n = {4: 2 * (3 * 3 + 2 * 4), 8: 2 * (3 * 3 + 2 * 4 + 2 * 2 * 3)}.get(d)
        if n:
            assert a.nnz == n


def test_callable_and_builtins_agree():
    rng = np.random.default_rng(3)
    r = rng.uniform(1, 5, (6, 7))
    r[2, 3] = np.nan
    for name, f in [("min", lambda x: x.min(axis=0)), ("max", lambda x: x.max(axis=0)),
                    ("from", lambda x: x[0]), ("to", lambda x: x[1]), ("diff", lambda x: x[1] - x[0])]:
        for sym in (True, False):
            if name in ("from", "to", "diff") and sym:
                continue
            a = ls.cost.transition(r, name, 8, symmetric=sym).to_sparse()
            b = ls.cost.transition(r, f, 8, symmetric=sym).to_sparse()
            assert (a != b).nnz == 0, name


def test_arithmetic_rules():
    r = np.array([[1.0, 2.0], [4.0, 8.0]])
    t = ls.cost.transition(r, "mean", 4)
    with pytest.raises(TypeError):
        t + 1
    with pytest.raises(TypeError):
        t / t
    inv = 1 / t
    np.testing.assert_array_equal(inv.to_sparse().data, 1 / t.to_sparse().data)
    assert (t * 2).n_connections == t.n_connections
    both = t + ls.cost.transition(r, "max", 8)
    assert both.directions == 8 and both.symmetric
    mixed = t * ls.cost.transition(r, "to", 4, symmetric=False)
    assert not mixed.symmetric


def test_map_pairs_all_includes_missing_cells():
    r = np.array([[1.0, np.nan, 2.0]])
    t = ls.cost.transition(r, "mean", 4)
    assert t.n_connections == 0
    s = t.map(lambda v: v + 1, pairs="all")
    assert s.n_connections == 4  # both pairs, both triangles


def test_dataarray_input_and_raster_output():
    dem = ls.load_raster(ls.cost.example_data("maungawhau"))
    tr = ls.cost.transition(dem, "mean", 8)
    assert tr.shape == (87, 61) and not tr.lonlat
    out = tr.to_raster()
    assert out.shape == dem.shape and out.dims == dem.dims
    np.testing.assert_allclose(out.values, dem.values, rtol=0.2)


def test_global_grid_wraps():
    r = np.ones((4, 8))
    tr = ls.cost.transition(r, "mean", 4, extent=(-180, 180, -90, 90), crs="+proj=longlat +datum=WGS84")
    assert tr.grid.wrap and tr.lonlat
    assert tr.to_sparse()[0, 7] == 1.0  # across the 180th meridian
    projected = ls.cost.transition(r, "mean", 4, extent=(-180, 180, -90, 90), crs="EPSG:3857")
    assert not projected.grid.wrap
