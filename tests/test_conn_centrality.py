# SPDX-License-Identifier: GPL-3.0-or-later
"""Centrality of habitat graphs: Makurhini's MK_RMCentrality and Conefor's BC, BC(IIC), BC(PC)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import shortest_path

import landschaft as ls
from test_conn_reference import SCENES

REF = Path(__file__).parent / "data" / "reference" / "conn"


def _num(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col, errors="coerce").to_numpy(float)


# ------------------------------------------------------------------ Makurhini
def _centrality_cases():
    ref = pd.read_csv(REF / "makurhini_centrality.csv.gz", keep_default_na=False, dtype=str)
    return [grp for _, grp in ref.groupby("case", sort=False)]


@pytest.mark.parametrize("grp", _centrality_cases(), ids=lambda g: f"case{g.iloc[0]['case']}")
def test_centrality_matches_makurhini(grp):
    gpd = pytest.importorskip("geopandas")
    row = grp.iloc[0]
    if row["kind"] == "vector":
        g = ls.HabitatGraph.from_vector(gpd.read_file(REF / "polygons.gpkg"), id="id", distance="centroid")
    else:
        g = ls.HabitatGraph.from_raster(REF / "habitat_nodes_raster.tif", distance="centroid")
    probability = None if row["probability"] == "" else float(row["probability"])
    ours = g.centrality(float(row["distance"]), probability=probability, binary=row["binary"] == "TRUE")
    ours = ours.set_index("id")
    for name, sub in grp.groupby("name", sort=False):
        if name == "memb.rw":
            continue  # needs python-igraph
        ids = sub["node"].astype(float).astype(int).to_numpy()
        a, b = ours.loc[ids, name].to_numpy(float), _num(sub["value"])
        assert np.array_equal(np.isnan(a), np.isnan(b)), name
        ok = ~np.isnan(b)
        if name in ("degree", "cluster"):
            np.testing.assert_array_equal(a[ok], b[ok], err_msg=name)
        elif name == "eigen":  # ARPACK (igraph) vs LAPACK
            np.testing.assert_allclose(a[ok], b[ok], rtol=1e-12, atol=1e-13, err_msg=name)
        else:
            np.testing.assert_allclose(a[ok], b[ok], rtol=1e-12, atol=1e-11, err_msg=name)


def test_walktrap_needs_igraph():
    ids, a, D = SCENES["points60"]
    frame = ls.HabitatGraph(a, D, ids=ids).centrality(8000)
    try:
        import igraph  # noqa: F401
    except ImportError:
        assert "memb_rw" not in frame
    else:
        assert frame["memb_rw"].min() == 1


# -------------------------------------------------------------- brute force
def _random_graph(n=30, seed=3):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10_000, (n, 2))
    D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    return rng.uniform(1, 5, n), D


def _brute_betweenness(W):
    """Pair-dependency betweenness over all shortest paths (lengths W, 0 = no edge), ties exact."""
    n = len(W)
    dist, _ = shortest_path(W, directed=False, return_predecessors=True)
    count = np.zeros((n, n))
    for s in range(n):
        order = np.argsort(dist[s])
        count[s, s] = 1
        for v in order[1:]:
            if not np.isfinite(dist[s, v]):
                continue
            u = np.flatnonzero((W[:, v] > 0) & np.isclose(dist[s] + W[:, v], dist[s, v], rtol=1e-12, atol=0))
            count[s, v] = count[s, u].sum()
    bc = np.zeros(n)
    for k in range(n):
        on = np.isclose(dist[:, [k]] + dist[[k], :], dist, rtol=1e-12, atol=0) & np.isfinite(dist)
        frac = np.where(on, count[:, [k]] * count[[k], :] / np.where(count > 0, count, 1), 0.0)
        frac[k, :] = frac[:, k] = 0
        np.fill_diagonal(frac, 0)
        bc[k] = frac.sum() / 2
    return bc, dist


@pytest.mark.parametrize("binary", [True, False])
def test_centrality_brute_force(binary):
    area, D = _random_graph()
    t = 2500.0
    g = ls.HabitatGraph(area, D)
    ours = g.centrality(t, binary=binary)
    if binary:
        W = ((D < t) & (D > 0)).astype(float)
    else:
        W = 1 / np.exp(-D / t)
        np.fill_diagonal(W, 0)
    bc, dist = _brute_betweenness(W)
    np.testing.assert_allclose(ours["BWC"], bc, rtol=1e-10, atol=1e-10)
    reach = np.isfinite(dist)
    sums = np.where(reach, dist, 0).sum(axis=1)
    reached = reach.sum(axis=1)
    with np.errstate(divide="ignore"):
        close = np.where(reached > 1, 1 / sums if binary else (reached - 1) / sums, np.nan)
    np.testing.assert_allclose(ours["close"], close, rtol=1e-12)
    np.testing.assert_allclose(ours["degree" if binary else "strength"], W.sum(axis=1), rtol=1e-13)
    vals, vecs = np.linalg.eigh(W)
    lead = np.abs(vecs[:, -1]) / np.abs(vecs[:, -1]).max()
    if binary:  # the leading eigenvector lives on one component
        np.testing.assert_allclose(np.abs(ours["eigen"]), lead, atol=1e-12)
    else:
        np.testing.assert_allclose(ours["eigen"], lead, rtol=1e-10)
    # one thread and many agree to the bit
    one = ls.HabitatGraph(area, D, n_threads=1).centrality(t, binary=binary)
    pd.testing.assert_frame_equal(one, ours)


def test_centrality_options():
    ids, a, D = SCENES["points60"]
    g = ls.HabitatGraph(a, D, ids=ids)
    both = g.centrality([4000, 8000])
    assert list(both["distance"].unique()) == [4000.0, 8000.0]
    assert "strength" in g.centrality(8000, probability=0.5)  # a probability makes it weighted
    conefor = ls.HabitatGraph(a, D, ids=ids, convention="conefor").centrality(float(D[0, 1]))
    makurhini = g.centrality(float(D[0, 1]))
    assert conefor["degree"].sum() == makurhini["degree"].sum() + 2  # the pair at the threshold
    with pytest.raises(ValueError, match="kind='distance'"):
        ls.HabitatGraph(a, np.exp(-D / 5000), ids=ids, kind="probability").centrality(0.1)


# --------------------------------------------------------------- Conefor BC
def _float_ties(D, d, p):
    """Nodes on a path whose length differs from the best by less than float precision."""
    n = len(D)
    P = np.exp((np.log(p) / d) * D)
    C = -np.log(P)
    np.fill_diagonal(C, 0)
    S = shortest_path(C, directed=False)
    tied = np.zeros(n, dtype=bool)
    for k in range(n):
        gap = (S[:, [k]] + S[[k], :] - S) / np.maximum(S, 1e-300)
        gap[k, :] = gap[:, k] = np.inf
        np.fill_diagonal(gap, np.inf)
        tied[k] = ((gap > 0) & (gap < 2.0**-23)).any()
    return tied


@pytest.mark.parametrize("scene", ["points60", "cost40", "clusters50"])
def test_betweenness_matches_conefor(scene):
    ref = pd.read_csv(REF / "conefor_betweenness.csv.gz")
    ref = ref[ref["scene"] == scene]
    ids, a, D = SCENES[scene]
    d, p = float(ref["distance"].iloc[0]), float(ref["probability"].iloc[0])
    g = ls.HabitatGraph(a, D, ids=ids, convention="conefor")
    ours = g.node_importance(["BC", "BCIIC", "BCPC"], distance=d, probability=p).set_index("id").loc[ref["Node"]]
    # Conefor 2.7.1 prints 6 significant digits.
    for col in ("BC", "dBC_IIC", "varBC_IIC"):
        np.testing.assert_allclose(ours[col], ref[col], rtol=1e-5, atol=1e-9, err_msg=col)
    # Conefor keeps probabilities in single precision: a path shorter by less
    # than float precision may win over the direct link (points60 and
    # clusters50 are Euclidean, so there BC(PC) is 0 in exact arithmetic).
    tied = _float_ties(D, d, p)[pd.Index(ids).get_indexer(ref["Node"])]
    for col in ("dBC_PC", "varBC_PC"):
        x, y = ours[col].to_numpy(), ref[col].to_numpy()
        differ = ~np.isclose(x, y, rtol=1e-5, atol=1e-9)
        assert not (differ & ~tied).any(), (col, ref["Node"][differ & ~tied].tolist())
    if scene == "cost40":
        assert (ref["varBC_PC"] > 0).sum() > 5 and not differ.any()
    else:
        assert (ours["varBC_PC"] == 0).all()


def test_betweenness_matches_centrality():
    """Conefor's BC is igraph's betweenness of the links, normalised by (n - 1)(n - 2) / 2."""
    ids, a, D = SCENES["clusters50"]
    g = ls.HabitatGraph(a, D, ids=ids, convention="conefor")
    n = len(a)
    bc = g.node_importance(["BC"], distance=5000)["BC"].to_numpy()
    bwc = g.centrality(5000)["BWC"].to_numpy()
    np.testing.assert_allclose(bc, bwc / ((n - 1) * (n - 2) / 2), rtol=1e-12, atol=1e-15)
