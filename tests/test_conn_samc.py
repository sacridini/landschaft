# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial absorbing Markov chains (Makurhini's MK_dPC_SAMC, the samc package)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
pytest.importorskip("rasterio")

REF = Path(__file__).parent / "data" / "reference" / "conn"
RESISTANCE = REF / "polygons_resistance.tif"
ABSORPTION = REF / "samc_absorption.tif"
POLY = gpd.read_file(REF / "polygons.gpkg")


def _cases():
    ref = pd.read_csv(REF / "makurhini_samc.csv.gz", keep_default_na=False, dtype=str)
    return [grp for _, grp in ref.groupby("case", sort=False)]


def _options(row) -> dict:
    return {"directions": int(row["dir"]), "settle": row["settle"],
            "vertices": None if row["nvertices"] == "" else int(row["nvertices"]),
            "vertex_fraction": None if row["pvertices"] == "" else float(row["pvertices"])}


@pytest.mark.parametrize("grp", _cases(), ids=lambda g: f"case{g.iloc[0]['case']}")
def test_samc_matches_makurhini(grp):
    opts = _options(grp.iloc[0])
    ids, P = ls.conn.settlement_probabilities(POLY, RESISTANCE, absorption=ABSORPTION, id="id", **opts)
    pij = grp[grp["level"] == "pij"]
    index = pd.Index(ids)
    pairs = np.array([s.split() for s in pij["id"]], dtype=int)
    ours = P[index.get_indexer(pairs[:, 0]), index.get_indexer(pairs[:, 1])]
    # samc solves with Eigen's SparseLU, landschaft with SuperLU.
    np.testing.assert_allclose(ours, pij["value"].astype(float), rtol=1e-12, atol=0)
    np.testing.assert_array_equal(P, P.T)
    g = ls.HabitatGraph.from_samc(POLY, RESISTANCE, absorption=ABSORPTION, id="id", area_unit="ha",
                                  landscape_area=2e6, **opts)
    imp = g.node_importance("PC", pij_min=0.001).set_index("id")  # MK_dPC_SAMC cuts at 0.001
    for name, sub in grp[grp["level"] == "node"].groupby("name", sort=False):
        a = imp.loc[sub["id"].astype(int), name].to_numpy(float)
        b = sub["value"].astype(float).to_numpy()
        if name == "dPCconnector":  # left unrounded (and at least 0) by MK_dPC_SAMC
            np.testing.assert_allclose(np.maximum(a, 0), b, rtol=0, atol=1e-10, err_msg=name)
        else:
            np.testing.assert_allclose(np.round(a, 7), b, rtol=0, atol=1e-12, err_msg=name)
    ov = g.overall("PC", pij_min=0.001).iloc[0]
    for _, row in grp[grp["level"] == "overall"].iterrows():
        assert ov[row["name"]] == pytest.approx(float(row["value"]), rel=1e-14)


def test_boundary_vertices_as_makurhini():
    square = shapely.Polygon([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (0, 0)])
    xy = shapely.get_coordinates(square)[:-1]
    np.testing.assert_array_equal(ls.conn.boundary_vertices(square, 10), xy)  # fewer than asked: all
    # round(seq(1, 8, length.out = 4)) = 1, 3 (2.667), 6 (5.333), 8
    np.testing.assert_array_equal(ls.conn.boundary_vertices(square, 4), xy[[0, 2, 5, 7]])
    # round(8 × 0.25) = 2: the first and the last
    np.testing.assert_array_equal(ls.conn.boundary_vertices(square, fraction=0.25), xy[[0, 7]])
    holed = shapely.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], [[(1, 1), (2, 1), (2, 2), (1, 1)]])
    # first and last points differ (the hole closes the list): every coordinate is kept
    assert len(ls.conn.boundary_vertices(holed, 100)) == len(shapely.get_coordinates(holed))


def test_samc_against_dense_markov_chain():
    rng = np.random.default_rng(5)
    res = rng.uniform(1, 20, (12, 15))
    res[4, 3:12] = np.nan
    absorb = np.where(rng.random(res.shape) < 0.3, 0.0, 0.05)
    polys = gpd.GeoDataFrame({"id": [1, 2, 3]}, geometry=[shapely.box(1.2, 1.2, 3.8, 3.8), shapely.box(9.2, 2.2, 12.8, 4.8),
                                                         shapely.box(5.2, 8.2, 7.8, 10.8)])
    ids, P = ls.conn.settlement_probabilities(polys, res, absorption=absorb, id="id", directions=8, vertices=4)
    # the chain with numpy: states row-major, samc's weights
    ok = np.isfinite(res)
    cells = np.flatnonzero(ok)
    m = cells.size
    state = np.full(res.size, -1)
    state[cells] = np.arange(m)
    Q = np.zeros((m, m))
    ny, nx = res.shape
    for s, a in enumerate(cells):
        r, c = divmod(a, nx)
        w = {}
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < ny and 0 <= c + dc < nx and ok[r + dr, c + dc]:
                    w[state[(r + dr) * nx + c + dc]] = (1 / ((res[r, c] + res[r + dr, c + dc]) / 2)
                                                        / (np.sqrt(2) if dr and dc else 1))
        total = sum(w.values())
        for t, v in w.items():
            Q[s, t] = (1 - absorb.flat[a]) * v / total
    F = np.linalg.inv(np.eye(m) - Q)
    M = F * absorb.ravel()[cells][None, :]
    own = []
    for g in polys.geometry:
        xy = ls.conn.boundary_vertices(g, 4)
        col, row = np.floor(xy[:, 0]).astype(int), (ny - np.ceil(xy[:, 1])).astype(int)
        own.append(np.unique(state[row * nx + col]))
    for i in range(3):
        for j in range(i + 1, 3):
            assert P[i, j] == pytest.approx(M[np.ix_(own[j], own[i])].sum(), rel=1e-10)


def test_default_absorption_and_errors():
    explicit = ls.conn.settlement_probabilities(POLY, RESISTANCE, absorption=ABSORPTION, id="id")[1]
    default = ls.conn.settlement_probabilities(POLY, RESISTANCE, dispersal_distance=5000, id="id")[1]
    np.testing.assert_allclose(default, explicit, rtol=1e-12)
    with pytest.raises(ValueError, match="dispersal distance"):
        ls.conn.settlement_probabilities(POLY, RESISTANCE)
    with pytest.raises(ValueError, match="settle"):
        ls.conn.settlement_probabilities(POLY, RESISTANCE, dispersal_distance=5000, settle="max")
    with pytest.raises(ValueError, match="at least one cell"):
        ls.conn.settlement_probabilities(POLY, np.ones((5, 5)), absorption=np.zeros((5, 5)))
    with pytest.raises(ValueError, match="between 0 and 1"):
        ls.conn.settlement_probabilities(POLY, np.ones((5, 5)), absorption=np.full((5, 5), 2.0))
