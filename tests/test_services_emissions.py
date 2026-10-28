# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.services.emissions: the carbon of land change, against sums done by hand (Terraflux's carbon.emissions
tests: a forest kept, a forest cleared, a pasture left to regrow and a secondary forest of unknown age turned to
forest)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine

import landschaft as ls

# four cells of a hectare at 2000, 2010 and 2020: 1 forest (100 tC/ha), 2 pasture (10), 3 secondary forest
LAYERS = [[[1, 1], [2, 3]], [[1, 2], [3, 3]], [[1, 2], [3, 1]]]


def _maps():
    da = xr.DataArray(np.array(LAYERS, dtype=np.uint8), dims=("band", "y", "x"),
                      coords={"band": ["2000", "2010", "2020"], "y": [150.0, 50.0], "x": [50.0, 150.0]})
    return da.rio.write_transform(Affine(100, 0, 0, 0, -100, 200)).rio.write_crs("EPSG:31983")


def _row(t, year, a, b):
    r = t[(t["year_from"] == year) & (t["from"] == a) & (t["to"] == b)]
    assert len(r) == 1
    return r.iloc[0]


def test_emissions_and_removals_by_hand():
    regrowth = 80.0 * (1.0 - math.exp(-0.1 * 10.0))
    e = ls.services.emissions(_maps(), {1: 100, 2: 10, 3: 40}, regrowth={3: {"max": 80, "rate": 0.1}}, co2=False)
    p = e.periods
    assert p[["year_from", "year_to"]].values.tolist() == [[2000, 2010], [2010, 2020]]
    # 2000-2010: the forest cleared (90) and the pasture's 10 lost when it turned to young regrowth
    assert p["emissions"][0] == pytest.approx(100.0, rel=1e-12)
    assert p["removals"][0] == 0.0
    # 2010-2020: ten years of regrowth, and the secondary forest of unknown age (40) turned to forest (100)
    assert p["emissions"][1] == 0.0
    assert p["removals"][1] == pytest.approx(regrowth + 60.0, rel=1e-12)
    assert p["net"][1] == pytest.approx(-(regrowth + 60.0), rel=1e-12)
    t = e.transitions
    assert _row(t, 2000, 1, 2)["emissions"] == pytest.approx(90.0)
    assert _row(t, 2000, 2, 3)["emissions"] == pytest.approx(10.0)
    assert _row(t, 2010, 3, 3)["removals"] == pytest.approx(regrowth, rel=1e-12)
    assert _row(t, 2010, 3, 1)["removals"] == pytest.approx(60.0)
    assert _row(t, 2000, 1, 1)["area_ha"] == 1.0
    assert t[["from", "to"]].values.tolist()[:4] == [[1, 1], [1, 2], [2, 3], [3, 3]]  # sorted by pair
    assert e.stocks.sel(time=2010).values.ravel().tolist() == [100.0, 10.0, 0.0, 40.0]
    assert e.stocks.sel(time=2020).values[1, 0] == pytest.approx(regrowth, rel=1e-15)
    assert e.net.values[0, 1] == pytest.approx(90.0)
    assert e.net.values[1, 0] == pytest.approx(10.0 - regrowth, rel=1e-12)
    assert e.net.rio.transform() == Affine(100, 0, 0, 0, -100, 200)
    assert e.units == "t C"


def test_co2_and_linear_regrowth():
    e = ls.services.emissions(_maps(), {1: 100, 2: 10, 3: 40},
                              regrowth={3: {"max": 80, "rate": 3, "form": "linear"}})
    assert e.periods["emissions"][0] == pytest.approx(100.0 * 44.0 / 12.0, rel=1e-12)
    assert e.periods["removals"][1] == pytest.approx((30.0 + 60.0) * 44.0 / 12.0, rel=1e-12)
    assert e.units == "t CO2"


def test_missing_cells_and_regrowth_shape():
    a = np.array([[1, 3], [np.nan, 1]])
    b = np.array([[3, 3], [3, np.nan]])
    c = np.array([[3, 3], [3, 3]])
    e = ls.services.emissions([a, b, c], {1: 100}, regrowth={3: {"max": 80, "rate": 0.2, "shape": 2}},
                              times=[0, 5, 15], cell_area=10000, co2=False)
    s = e.stocks.values
    curve = lambda age: 80 * (1 - math.exp(-0.2 * age)) ** 2
    assert s[1, 0, 0] == 0.0 and s[2, 0, 0] == pytest.approx(curve(10))
    assert s[0, 0, 1] == 80.0  # regrowth at the first date without a stock entry: its max
    assert s[1, 1, 0] == 0.0  # after a missing date the class appeared at that date
    assert math.isnan(s[0, 1, 0]) and math.isnan(s[1, 1, 1])
    assert s[2, 1, 1] == 0.0  # appeared after a gap
    assert math.isnan(e.net.values[1, 1])  # never valid at two consecutive dates
    assert e.net.values[0, 1] == 0.0


def test_errors():
    with pytest.raises(ValueError, match="class 2 has no stock"):
        ls.services.emissions([np.ones((2, 2)), np.full((2, 2), 2)], {1: 10})
    with pytest.raises(ValueError, match="form"):
        ls.services.emissions([np.ones((2, 2)), np.ones((2, 2))], {1: 10},
                              regrowth={1: {"max": 1, "rate": 1, "form": "logistic"}})
    with pytest.raises(ValueError, match="two maps"):
        ls.services.emissions([np.ones((2, 2))], {1: 10})


def test_threads_do_not_change_the_sums():
    rng = np.random.default_rng(7)
    maps = [rng.integers(1, 5, size=(300, 400)) for _ in range(4)]
    stock = {1: 120.5, 2: 10.25, 3: 33.3, 4: 0.7}
    r = {3: {"max": 90, "rate": 0.07, "shape": 1.3}}
    one = ls.services.emissions(maps, stock, regrowth=r, times=[2000, 2005, 2012, 2020], cell_area=900,
                                n_threads=1)
    two = ls.services.emissions(maps, stock, regrowth=r, times=[2000, 2005, 2012, 2020], cell_area=900,
                                n_threads=2)
    assert one.periods.equals(two.periods) and one.transitions.equals(two.transitions)
    assert np.array_equal(one.net.values, two.net.values, equal_nan=True)
    # the totals are the transitions' sums
    by = one.transitions.groupby("year_from")[["emissions", "removals"]].sum()
    np.testing.assert_allclose(by.values, one.periods[["emissions", "removals"]].values, rtol=1e-12)
    # against a direct computation of the cells' stock differences
    st = one.stocks.values
    d = (st[:-1] - st[1:]) * 44 / 12 * 0.09
    np.testing.assert_allclose(one.periods["emissions"], np.where(d > 0, d, 0).sum(axis=(1, 2)), rtol=1e-12)
