# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.forest_code against the law's numbers worked by hand (Terraflux's tests): the exact distances (against brute
force), the margins of rivers by width and of lakes by area, springs, the consolidated areas' narrower margins by
fiscal modules, and the Reserva Legal's balance with art. 15's APP and art. 67's small properties."""

import numpy as np
import pytest
import xarray as xr

import landschaft as ls
from landschaft.forest_code import _nearest


def grid(values, cell=10.0):
    ny, nx = np.shape(values)
    return xr.DataArray(np.asarray(values, dtype=float), dims=("y", "x"),
                        coords={"y": (ny - 0.5 - np.arange(ny)) * cell, "x": (np.arange(nx) + 0.5) * cell})


def test_distances_are_exact():
    g = np.random.default_rng(7)
    f = g.random((23, 31)) < 1 / 17
    rr, cc = np.nonzero(f)
    for dx, dy in [(10.0, 10.0), (30.0, 12.5)]:
        d, near = _nearest(f, dx, dy)
        for i in range(f.size):
            r, c = divmod(i, f.shape[1])
            best = np.min(np.hypot((cc - c) * dx, (rr - r) * dy))
            assert abs(d.flat[i] - best) < 1e-9
            j = near.flat[i]
            assert abs(np.hypot((j % f.shape[1] - c) * dx, (j // f.shape[1] - r) * dy) - best) < 1e-9


def test_the_laws_widths():
    fc = ls.forest_code
    assert [fc.river_margin(w) for w in (5, 10, 49, 50, 51, 200, 600, 601)] == [30, 50, 50, 50, 100, 100, 200, 500]
    assert [fc.consolidated_river(m, 20) for m in (0.5, 1, 1.5, 3, 4, 9)] == [5, 5, 8, 15, 15, 20]
    assert [fc.consolidated_lake(m) for m in (1, 2, 4, 4.5)] == [5, 8, 15, 30]


def test_margins_of_rivers_lakes_and_springs():
    import geopandas as gpd

    nx, ny = 80, 40
    rivers = np.zeros((ny, nx))
    rivers[:, 0] = 1
    # a river 12 m wide in the top half, 5 m in the bottom
    width = np.where(np.arange(ny)[:, None] < 20, 12.0, 5.0) * np.ones((1, nx))
    a = ls.forest_code.app(rivers=grid(rivers), river_width=grid(width)).app.values
    # from the channel's edge (5 m from its centre): 50 m where it is 12 m wide, 30 m where 5 m
    assert np.sum(a[5] == 1) == 5 and np.sum(a[30] == 1) == 3
    assert a[0, 0] == 0, "the channel is not its margin"
    assert ls.forest_code.app(rivers=grid(rivers), river_width=12).app.values[30].tolist().count(1.0) == 5
    # consolidated, in a property of 2 fiscal modules (80 x 40 cells of 0.01 ha = 32 ha, modules of 16 ha): 8 m
    ones = grid(np.ones((ny, nx)))
    a = ls.forest_code.app(rivers=grid(rivers), properties=ones, consolidated=ones, fiscal_module=16).app.values
    assert np.sum(a[0] == 1) == 1
    with pytest.raises(ValueError, match="fiscal_module"):
        ls.forest_code.app(rivers=grid(rivers), consolidated=ones)
    # a lake of 9 cells (0.09 ha): 50 m; a spring: 50 m around it
    lake = np.zeros((ny, nx))
    lake[10:13, 40:43] = 1
    springs = gpd.GeoDataFrame(geometry=gpd.points_from_xy([705.0], [205.0]))
    out = ls.forest_code.app(lakes=grid(lake), springs=springs)
    a = out.app.values
    assert a[11, 47] == 2 and a[11, 48] == 0  # 50 m from the lake's last column 42: 45 m from its edge
    assert a[19, 70] == 3 and a[19, 75] == 3 and a[19, 76] == 0
    assert out.table.loc[1, "kind"] == 2 and out.table.loc[1, "name"] == "lake"
    assert out.table.loc[2, "cells"] == np.sum(a == 3)
    # urban: 30 m; a large lake (over 20 ha): 100 m
    urban = ls.forest_code.app(lakes=grid(lake), urban=ones).app.values
    assert urban[11, 45] == 2 and urban[11, 46] == 0
    big = np.zeros((60, 60))
    big[:50, :50] = 1  # 2500 cells of 0.01 ha: 25 ha
    b = ls.forest_code.app(lakes=grid(big)).app.values
    assert b[10, 59] == 2 and b[55, 10] == 2  # within 100 m of the edge


def test_the_other_kinds_and_their_order():
    s = grid([[50.0, 10.0], [10.0, 10.0]])
    e = grid([[100.0, 1900.0], [100.0, 100.0]])
    h = grid([[0.0, 0.0], [1.0, 0.0]])
    lc = grid([[5.0, 5.0], [5.0, np.nan]])
    a = ls.forest_code.app(slope=s, elevation=e, hilltops=h, landcover=lc, protected_classes=[5]).app.values
    assert a.tolist() == [[5.0, 7.0], [6.0, 0.0]]
    assert ls.forest_code.app(landcover=lc, protected_classes=[5]).app.values[1, 1] != 8


def test_the_reserva_legal_balance():
    # two properties of 100 cells of 1 ha (cells of 100 m): 1 with 30 ha of vegetation (10 of them in APP), 2 with 10
    nx, ny = 20, 10
    c = np.arange(nx)[None, :] * np.ones((ny, 1))
    r = np.arange(ny)[:, None] * np.ones((1, nx))
    props = grid(np.where(c < 10, 1.0, 2.0), 100)
    veg = grid((((c < 10) & (r < 3)) | ((c >= 10) & (r < 1))).astype(float), 100)
    app = grid((r == 0).astype(float), 100)
    t = ls.forest_code.reserve(props, veg, app=app, share=20).table
    assert tuple(t.loc[0, ["area_ha", "native_ha", "app_native_ha"]]) == (100, 30, 10)
    assert tuple(t.loc[0, ["required_ha", "surplus_ha", "deficit_ha"]]) == (20, 10, 0)
    assert tuple(t.loc[1, ["counted_ha", "deficit_ha"]]) == (10, 10)
    # without counting the APP (art. 15 not used): property 1 counts 20, property 2 none
    t2 = ls.forest_code.reserve(props, veg, app=app, share=20, count_app=False).table
    assert (t2.loc[0, "counted_ha"], t2.loc[1, "deficit_ha"]) == (20, 20)
    # art. 67: modules of 30 ha (property 2 has 3.3), the vegetation of 2008 the same as now: no deficit
    r3 = ls.forest_code.reserve(props, veg, app=app, vegetation_2008=veg, share=20, fiscal_module=30)
    assert (r3.table.loc[1, "required_ha"], r3.table.loc[1, "deficit_ha"]) == (10, 0)
    assert (r3.balance.values[0, 0], r3.balance.values[0, 15]) == (10, 0)
    # a map of the share: 80 % in property 1
    sm = grid(np.where(c < 10, 80.0, 20.0), 100)
    t4 = ls.forest_code.reserve(props, veg, share=sm).table
    assert (t4.loc[0, "share"], t4.loc[0, "deficit_ha"]) == (80, 50)
