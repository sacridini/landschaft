# SPDX-License-Identifier: GPL-3.0-or-later
"""Species dispersal (ls.sdm.migrate) against MigClim 1.6.2 and metapopulations (ls.conn.metapopulation) against R's
MetaLandSim 2.0 (tests/data/reference/dispersal, tools/make_reference_dispersal.py: MigClim's C built with a seed;
tests/data/reference/metapopulation, tools/make_reference_metapopulation.py: spom iterated from set.seed). MigClim:
three simulations of 3 periods on 40 x 50 cells (weak barriers and long-distance dispersal; strong barriers, a
threshold and propagule production that grows with age; no barriers and long-distance dispersal every step), every
step's state, the final state and the statistics, to the bit. MetaLandSim: 40 steps on 60 patches, four combinations
of the kernel, the connectivity, the colonisation and the extinction, the occupancy after every step."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import landschaft as ls

REF = Path(__file__).parent / "data" / "reference"

CASES = {
    "weak": {"seed": 7, "barrier_type": "weak", "threshold": 0, "steps": 5, "kernel": [1.0, 0.4, 0.16, 0.06],
             "ini_maturity": 1, "full_maturity": 1, "propagules": [1.0], "ldd_frequency": 0.1, "ldd_min": 6,
             "ldd_max": 15},
    "strong": {"seed": 11, "barrier_type": "strong", "threshold": 250, "steps": 4, "kernel": [0.9, 0.6, 0.3],
               "ini_maturity": 1, "full_maturity": 4, "propagules": [0.3, 0.6, 0.9], "ldd_frequency": 0.0,
               "ldd_min": 10, "ldd_max": 15},
    "open": {"seed": 3, "barrier_type": "strong", "threshold": 0, "steps": 3, "kernel": [0.9, 0.5], "ini_maturity": 2,
             "full_maturity": 3, "propagules": [0.6], "ldd_frequency": 1.0, "ldd_min": 3, "ldd_max": 10},
}


@pytest.fixture(scope="module")
def migclim():
    z = dict(np.load(REF / "dispersal" / "dispersal.npz"))
    ny, nx = z["land/ini"].shape
    coords = {"y": (ny - 0.5 - np.arange(ny)) * 100.0, "x": (np.arange(nx) + 0.5) * 100.0}

    def m(v):
        return xr.DataArray(np.where(v == -9999, np.nan, v), dims=("y", "x"), coords=coords)

    return z, m


@pytest.mark.parametrize("name", list(CASES))
def test_migclims_runs(migclim, name):
    z, m = migclim
    hs = xr.concat([m(z[f"land/hs{k}"]) for k in (1, 2, 3)], dim="period")
    bar = None if name == "open" else m(z["land/barrier"])
    r = ls.sdm.migrate(m(z["land/ini"]), hs, barriers=bar, **CASES[name])
    want = z[f"{name}/steps"]
    assert r.steps.shape == want.shape
    for k in range(want.shape[0]):
        got = np.where(np.isnan(r.steps.values[k]), -9999, r.steps.values[k])
        np.testing.assert_array_equal(got, want[k], err_msg=f"{name}: step {int(r.steps.step[k])}")
    np.testing.assert_array_equal(np.where(np.isnan(r.map.values), -9999, r.map.values), z[f"{name}/final"])
    np.testing.assert_array_equal(r.table.to_numpy(), z[f"{name}/stats"])
    fin = z[f"{name}/final"]
    np.testing.assert_array_equal(np.nan_to_num(r.occupied.values, nan=-1),
                                  np.where(fin == -9999, -1, (fin > 0) & (fin < 30000)))
    assert list(r.steps.step.values[:2]) == [101, 102]


def test_migrate_inputs(migclim):
    z, _ = migclim
    ini = z["land/ini"]
    hs = [z[f"land/hs{k}"] for k in (1, 2, 3)]
    r = ls.sdm.migrate(ini, hs, barriers=z["land/barrier"], nodata=-9999, **CASES["weak"])
    np.testing.assert_array_equal(np.nan_to_num(r.map, nan=-9999), z["weak/final"])
    one = ls.sdm.migrate(ini, hs[0], kernel=[1.0], nodata=-9999)
    assert len(one.table) == 2 and one.steps.shape == (1, 40, 50)
    with pytest.raises(ValueError, match="probabilities"):
        ls.sdm.migrate(ini, hs, kernel=[1.5])
    with pytest.raises(ValueError, match="propagules"):
        ls.sdm.migrate(ini, hs, kernel=[1.0], ini_maturity=1, full_maturity=4, propagules=[0.5])
    with pytest.raises(ValueError, match="ldd_max"):
        ls.sdm.migrate(ini, hs, kernel=[1.0], ldd_frequency=0.5, ldd_min=10, ldd_max=5)
    with pytest.raises(ValueError, match="shape"):
        ls.sdm.migrate(ini, hs[0][:10], kernel=[1.0])


def _model(name, s):
    kw = {"alpha": s[0], "x": s[1], "y": s[2], "e": s[3], "b": s[5], "seed": int(s[10])}
    if name == "rescue":
        kw.update(connectivity="area", c1=s[6], colonisation="exponential", extinction="rescue", r=s[9])
    elif name == "power":
        kw.update(kernel="power", beta=s[4], colonisation="power", c2=s[7], z=s[8], extinction="exponential")
    return kw


@pytest.mark.parametrize("name", ["hanski", "rescue", "power", "dense"])
def test_metalandsims_runs(name):
    z = np.load(REF / "metapopulation" / "metapopulation.npz")
    p = pd.DataFrame(z["patches"], columns=["x", "y", "area", "occupied"])
    want = z[f"{name}/occupancy"]
    m = ls.conn.metapopulation(p, steps=want.shape[0], **_model(name, z[f"{name}/settings"]))
    np.testing.assert_array_equal(m.occupancy, want != 0)
    np.testing.assert_array_equal(m.table["occupied"].to_numpy()[1:], want.sum(axis=1))
    np.testing.assert_array_equal(m.patches["occupied"], want[-1])
    np.testing.assert_allclose(m.patches["incidence"], want.mean(axis=0))
    assert m.table["occupied"].iloc[0] == p["occupied"].sum()
    d = np.diff(m.table["occupied"].to_numpy())
    np.testing.assert_array_equal(d, (m.table["colonised"] - m.table["extinct"]).to_numpy()[1:])


def test_metapopulation_inputs():
    import geopandas as gpd

    z = np.load(REF / "metapopulation" / "metapopulation.npz")
    p = z["patches"]
    g = gpd.GeoDataFrame({"area": p[:, 2], "occ": p[:, 3]}, geometry=gpd.points_from_xy(p[:, 0], p[:, 1]))
    m = ls.conn.metapopulation(g, steps=40, alpha=0.08, x=0.5, y=8.0, e=0.4, occupied="occ", seed=3)
    np.testing.assert_array_equal(m.occupancy, z["hanski/occupancy"] != 0)
    with pytest.raises(ValueError, match="c1"):
        ls.conn.metapopulation(g, alpha=0.1, x=1, y=1, e=1, connectivity="area", occupied="occ")
    with pytest.raises(ValueError, match="kernel"):
        ls.conn.metapopulation(g, alpha=0.1, x=1, y=1, e=1, kernel="gauss", occupied="occ")
    with pytest.raises(ValueError, match="no column"):
        ls.conn.metapopulation(pd.DataFrame({"x": [0, 1], "y": [0, 1]}), alpha=0.1, x=1, y=1, e=1)
