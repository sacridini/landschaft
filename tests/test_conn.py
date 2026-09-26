# SPDX-License-Identifier: GPL-3.0-or-later
"""Habitat graphs: every index against a brute-force recomputation per removed node."""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
from landschaft.conn.graph import MAKURHINI_UNCONNECTED


# ----------------------------------------------------------------------------
# Brute force: shortest paths from scratch for the graph and for each node
# removed (or with its attribute set to 0).


def _dijkstra(adj: dict, n: int, src: int, limit: float = np.inf) -> np.ndarray:
    dist = np.full(n, np.inf)
    dist[src] = 0.0
    heap = [(0.0, src)]
    done = np.zeros(n, bool)
    while heap:
        d, v = heapq.heappop(heap)
        if done[v] or d != dist[v]:
            continue
        done[v] = True
        if d > limit:
            continue
        for u, w in adj[v]:
            if not done[u] and d + w < dist[u]:
                dist[u] = d + w
                heapq.heappush(heap, (dist[u], u))
    return dist


def _graphs(D, t, P, pij_min, conefor, kind="distance"):
    n = len(D)
    off = ~np.eye(n, dtype=bool)
    valid = off & ~np.isnan(D)
    with np.errstate(invalid="ignore"):
        if kind == "distance":
            link = (D <= t) if conefor else (D < t)
            p = np.exp((np.log(P) / t) * D) if conefor else np.exp((D * np.log(P)) / t)
        elif kind == "probability":
            link = D >= t
            p = D.copy()
        else:
            link = D != 0
            p = None
    link &= valid
    if p is not None:
        p = np.where(valid, p, 0.0)
        with np.errstate(invalid="ignore"):
            keep = (p > pij_min) if conefor else ((p >= pij_min) & (p > 0))
        keep &= valid
    else:
        keep = None
    return link, p, keep


def _indices(a, link, p, keep, alive, conefor, limit):
    """Every index of the graph restricted to `alive` nodes (attributes a)."""
    n = len(a)
    idx = np.flatnonzero(alive)
    sub = {v: k for k, v in enumerate(idx)}
    m = len(idx)
    ladj = {k: [(sub[u], 1.0) for u in np.flatnonzero(link[v]) if alive[u]] for k, v in enumerate(idx)}
    hops = np.array([_dijkstra(ladj, m, s) for s in range(m)]) if m else np.zeros((0, 0))
    aa = np.outer(a[idx], a[idx])
    fin = np.isfinite(hops)
    out = {}
    out["NL"] = int(sum(len(v) for v in ladj.values()) // 2)
    # components
    comp = -np.ones(m, int)
    c = 0
    for s in range(m):
        if comp[s] < 0:
            comp[fin[s]] = c
            c += 1
    out["NC"] = c
    csum = np.array([a[idx][comp == k].sum() for k in range(c)])
    out["comp_sq"] = float((csum**2).sum())
    out["total"] = float(a[idx].sum())
    with np.errstate(divide="ignore"):
        inv = np.where(fin & (hops > 0), 1.0 / hops, 0.0)
    out["H"] = inv.sum() / 2
    out["iic_reach"] = float(np.where(fin, aa / (1 + np.where(fin, hops, 0)), 0).sum())
    out["unreach_aa"] = float(aa[~fin].sum())
    out["iic_row"] = np.where(fin, aa / (1 + np.where(fin, hops, 0)), 0).sum(1)
    out["unreach_row"] = np.where(~fin, aa, 0).sum(1)
    if p is not None:
        padj = {k: [(sub[u], -np.log(p[v, u])) for u in np.flatnonzero(keep[v]) if alive[u]] for k, v in enumerate(idx)}
        cost = np.array([_dijkstra(padj, m, s, limit) for s in range(m)])
        pstar = np.exp(-cost)
        out["pc_num"] = float((aa * pstar).sum())
        out["pc_row"] = (aa * pstar).sum(1)
        pp = p[np.ix_(idx, idx)]
        out["F"] = float(pp.sum())
        out["AWF"] = float((aa * pp).sum())
    out["idx"] = idx
    return out


def brute_force(a, D, t, P, pij_min, conefor, la, kind="distance"):
    n = len(a)
    link, p, keep = _graphs(D, t, P, pij_min, conefor, kind)
    limit = -np.log(pij_min) if conefor and pij_min > 0 else np.inf
    full = _indices(a, link, p, keep, np.ones(n, bool), conefor, limit)
    sent = 0.0 if conefor else 1.0 / (1.0 + MAKURHINI_UNCONNECTED)
    iic_num = full["iic_reach"] + sent * full["unreach_aa"]
    total = a.sum()
    rows = []
    for k in range(n):
        alive = np.ones(n, bool)
        alive[k] = False
        de = _indices(a, link, p, keep, alive, conefor, limit)
        a0 = a.copy()
        a0[k] = 0
        row = {"dA": 100 * a[k] / total, "dNC": 100 * (full["NC"] - de["NC"]) / full["NC"],
               "dNL": 100 * (full["NL"] - de["NL"]) / full["NL"] if full["NL"] else 0.0,
               "dH": 100 * (full["H"] - de["H"]) / full["H"] if full["H"] else 0.0}
        ccp = full["comp_sq"] / total**2
        row["dCCP"] = 100 * (ccp - de["comp_sq"] / de["total"] ** 2) / ccp
        row["dLCP"] = 100 * (full["comp_sq"] - de["comp_sq"]) / full["comp_sq"]
        # Makurhini: the graph without the node counts no unconnected pairs.
        row["dIIC"] = 100 * (iic_num - de["iic_reach"] - (sent * de["unreach_aa"] if conefor else 0)) / iic_num
        row["dIICintra"] = 100 * a[k] ** 2 / iic_num
        row["dIICflux"] = 100 * 2 * (full["iic_row"][k] + sent * full["unreach_row"][k] - a[k] ** 2) / iic_num
        row["dIICconnector"] = max(row["dIIC"] - row["dIICintra"] - row["dIICflux"], 0.0)
        if p is not None:
            num = full["pc_num"]
            zero = _indices(a0, link, p, keep, np.ones(n, bool), conefor, limit)
            row["dPC"] = 100 * (num - de["pc_num"]) / num
            row["dPCintra"] = 100 * a[k] ** 2 / num
            row["dPCconnector"] = 100 * (zero["pc_num"] - de["pc_num"]) / num
            row["dPCflux"] = row["dPC"] - row["dPCintra"] - row["dPCconnector"]
            row["varPC"] = (num - de["pc_num"]) / la**2 if la else np.nan
            row["dF"] = 100 * (full["F"] - de["F"]) / full["F"]
            row["dAWF"] = 100 * (full["AWF"] - de["AWF"]) / full["AWF"]
        rows.append(row)
    overall = {"NC": full["NC"], "NL": full["NL"], "H": full["H"], "CCP": full["comp_sq"] / total**2,
               "LCPnum": full["comp_sq"], "IICnum": iic_num}
    if p is not None:
        overall.update({"PCnum": full["pc_num"], "F": full["F"], "AWF": full["AWF"]})
    return overall, pd.DataFrame(rows)


def _scene(n, seed, metric=True):
    rng = np.random.default_rng(seed)
    a = rng.lognormal(0, 1, n)
    if metric:
        xy = rng.uniform(0, 100, (n, 2))
        D = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    else:
        D = rng.uniform(5, 60, (n, n))
        D = np.triu(D, 1)
        D = D + D.T
    return a, D


CASES = [
    # (n, seed, metric distances, t, P, pij_min, convention)
    (25, 1, True, 20.0, 0.5, 0.01, "makurhini"),
    (25, 2, True, 12.0, 0.5, 0.0, "makurhini"),
    (30, 3, False, 25.0, 0.3, 0.01, "makurhini"),
    (30, 4, False, 25.0, 0.3, 0.0, "conefor"),
    (25, 5, True, 15.0, 0.5, 0.05, "conefor"),
    (20, 6, False, 30.0, 0.5, 0.2, "conefor"),
    (25, 7, True, 8.0, 0.5, 0.01, "makurhini"),  # several components
]


@pytest.mark.parametrize("n,seed,metric,t,P,pij_min,convention", CASES)
def test_against_brute_force(n, seed, metric, t, P, pij_min, convention):
    a, D = _scene(n, seed, metric)
    la = 3 * a.sum()
    g = ls.HabitatGraph(a, D, landscape_area=la, convention=convention)
    ov = g.overall(distance=t, probability=P, pij_min=pij_min).iloc[0]
    imp = g.node_importance(distance=t, probability=P, pij_min=pij_min)
    b_ov, b_imp = brute_force(a, D, t, P, pij_min, convention == "conefor", la)
    for key, value in b_ov.items():
        assert ov[key] == pytest.approx(value, rel=1e-12, abs=1e-12), key
    for col in b_imp.columns:
        np.testing.assert_allclose(imp[col].to_numpy(), b_imp[col].to_numpy(), rtol=1e-9, atol=1e-10, err_msg=col)


def test_fractions_add_up():
    a, D = _scene(30, 11, False)
    g = ls.HabitatGraph(a, D, landscape_area=10 * a.sum())
    ov = g.overall(distance=25.0, probability=0.4).iloc[0]
    for m in ("IIC", "PC"):
        total = ov[f"{m}intra(%)"] + ov[f"{m}direct(%)"] + ov[f"{m}step(%)"]
        # Makurhini's IIC numerator also counts the unconnected pairs.
        assert total == pytest.approx(100, abs=1e-6)
    imp = g.node_importance(["PC"], distance=25.0, probability=0.4)
    parts = imp["dPCintra"] + imp["dPCflux"] + imp["dPCconnector"]
    np.testing.assert_allclose(parts, imp["dPC"], rtol=1e-12)


def test_four_nodes_by_hand():
    a = np.array([1.0, 2, 3, 4])
    D = np.abs(np.subtract.outer(np.arange(4), np.arange(4))).astype(float)
    g = ls.HabitatGraph(a, D, convention="conefor")
    ov = g.overall(distance=1.5, probability=0.5).iloc[0]
    assert ov["NL"] == 3 and ov["NC"] == 1
    assert ov["H"] == pytest.approx(1 + 1 + 1 + 0.5 + 0.5 + 1 / 3)
    # IICnum = sum a_i^2 + 2 * sum_{i<j} a_i a_j / (1 + nl_ij)
    iic = 30 + 2 * (2 / 2 + 6 / 2 + 12 / 2 + 3 / 3 + 8 / 3 + 4 / 4)
    assert ov["IICnum"] == pytest.approx(iic)
    imp = g.node_importance(["NC", "CCP"], distance=1.5).set_index("id")
    assert list(imp["dNC"]) == [0, -100, -100, 0]
    assert imp.loc[3, "dCCP"] == pytest.approx(100 * (1 - 25 / 49))


def test_pairs_equal_matrix_and_ids():
    a, D = _scene(20, 21, True)
    ids = np.arange(20) * 7 + 3
    dense = ls.HabitatGraph(a, D, ids=ids)
    i, j = np.triu_indices(20, 1)
    # Pairs in any order and orientation, with repeats.
    table = pd.DataFrame({"from": ids[j], "to": ids[i], "d": D[i, j]})
    table = pd.concat([table, table.iloc[:5]]).sample(frac=1.0, random_state=1)
    sparse = ls.HabitatGraph(a, table, ids=ids)
    pd.testing.assert_frame_equal(dense.node_importance(distance=30), sparse.node_importance(distance=30))
    assert list(dense.node_importance(distance=30)["id"]) == list(ids)


def test_missing_pairs_are_unconnected():
    a = np.ones(4)
    table = np.array([[1, 2, 1.0], [2, 3, 1.0]])
    g = ls.HabitatGraph(a, table, convention="conefor")
    ov = g.overall(["NC", "NL"], distance=5).iloc[0]
    assert ov["NC"] == 2 and ov["NL"] == 2
    assert list(g.components(5)) == [1, 1, 1, 2]


def test_threads_do_not_change_results():
    a, D = _scene(150, 31, False)
    one = ls.HabitatGraph(a, D, n_threads=1).node_importance(distance=20.0, probability=0.5)
    many = ls.HabitatGraph(a, D, n_threads=4).node_importance(distance=20.0, probability=0.5)
    pd.testing.assert_frame_equal(one, many, check_exact=True)


def test_several_distances():
    a, D = _scene(20, 41, True)
    g = ls.HabitatGraph(a, D)
    ov = g.overall(["PC", "NC"], distance=[10, 20, 40])
    assert list(ov["distance"]) == [10, 20, 40]
    assert ov["NC"].is_monotonic_decreasing
    imp = g.node_importance("PC", distance=[10, 20])
    assert len(imp) == 40 and list(imp.columns[:2]) == ["id", "distance"]
    single = g.node_importance("PC", distance=20)
    np.testing.assert_array_equal(imp["dPC"].to_numpy()[20:], single["dPC"].to_numpy())


def test_default_distance_is_the_median():
    a, D = _scene(15, 51, True)
    g = ls.HabitatGraph(a, D)
    assert g.overall("PC").iloc[0]["distance"] == pytest.approx(np.median(D))


def test_probability_and_link_kinds():
    a, D = _scene(20, 61, True)
    t, P = 25.0, 0.5
    prob = np.exp(D * np.log(P) / t)
    np.fill_diagonal(prob, 1.0)
    by_distance = ls.HabitatGraph(a, D, convention="conefor").node_importance(["PC", "F", "AWF"], distance=t,
                                                                               probability=P)
    by_prob = ls.HabitatGraph(a, prob, kind="probability", convention="conefor").node_importance(
        ["PC", "F", "AWF"], distance=0.5)
    for col in ("dPC", "dPCconnector", "dF", "dAWF"):
        np.testing.assert_allclose(by_prob[col], by_distance[col], rtol=1e-9, atol=1e-12)
    links = (D <= t).astype(float)
    by_link = ls.HabitatGraph(a, links, kind="link", convention="conefor").node_importance(["IIC", "H", "NC"])
    ref = ls.HabitatGraph(a, D, convention="conefor").node_importance(["IIC", "H", "NC"], distance=t)
    for col in ("dIIC", "dH", "dNC"):
        np.testing.assert_allclose(by_link[col], ref[col], rtol=1e-12)
    with pytest.raises(ValueError, match="needs distances"):
        ls.HabitatGraph(a, links, kind="link").overall("PC")


def test_landscape_area_caps_and_units():
    a, D = _scene(20, 71, True)
    ov = ls.HabitatGraph(a, D).overall(["PC", "IIC", "LCP"], distance=30).iloc[0]
    assert np.isnan(ov["PC"]) and np.isnan(ov["IIC"]) and np.isnan(ov["LCP"])
    la = a.sum() * 2
    ov = ls.HabitatGraph(a, D, landscape_area=la).overall(["PC"], distance=30).iloc[0]
    assert ov["PC"] == pytest.approx(ov["PCnum"] / la**2)
    assert ov["EC(PC)"] == pytest.approx(np.sqrt(ov["PCnum"]))
    ov = ls.HabitatGraph(a, D, convention="conefor").overall(["PC"], distance=30).iloc[0]
    assert ov["PC"] == ov["PCnum"]  # Conefor without a landscape area: LA = 1


def test_conefor_files_round_trip(tmp_path):
    a, D = _scene(12, 81, True)
    D[0, 5] = D[5, 0] = np.nan
    g = ls.HabitatGraph(a, D, ids=np.arange(12) + 10)
    g.to_conefor(tmp_path / "nodes.txt", tmp_path / "distances.txt")
    back = ls.HabitatGraph.from_conefor(tmp_path / "nodes.txt", tmp_path / "distances.txt")
    pd.testing.assert_frame_equal(g.node_importance(distance=40), back.node_importance(distance=40))
    np.savetxt(tmp_path / "matrix.txt", np.nan_to_num(D, nan=1e9))
    matrix = ls.HabitatGraph.from_conefor(tmp_path / "nodes.txt", tmp_path / "matrix.txt")
    assert matrix.n_nodes == 12


def test_invalid_input():
    with pytest.raises(ValueError, match="at least one"):
        ls.HabitatGraph([], np.zeros((0, 0)))
    with pytest.raises(ValueError, match="non-negative"):
        ls.HabitatGraph([1.0, -1.0], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="convention"):
        ls.HabitatGraph([1.0, 1.0], np.zeros((2, 2)), convention="x")
    with pytest.raises(ValueError, match="unknown metric"):
        ls.HabitatGraph([1.0, 1.0], np.ones((2, 2))).overall("XYZ", distance=1)
    with pytest.raises(ValueError, match="not nodes"):
        ls.HabitatGraph([1.0, 1.0], np.array([[1, 3, 2.0]]))
    with pytest.raises(ValueError, match="different values"):
        ls.HabitatGraph([1.0, 1.0], np.array([[1, 2, 2.0], [2, 1, 3.0]]))
    with pytest.raises(ValueError, match="give the distance"):
        ls.HabitatGraph([1.0, 1.0], np.ones((2, 2)), convention="conefor").overall()


def test_large_graph_runs_quickly():
    rng = np.random.default_rng(5)
    n = 2000
    xy = rng.uniform(0, 100_000, (n, 2))
    D = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1))
    g = ls.HabitatGraph(rng.lognormal(0, 1, n), D)
    imp = g.node_importance(["PC", "IIC"], distance=3_000)
    assert len(imp) == n and np.all(imp["dPC"] >= 0)


# ------------------------------------------- links and nodes added (restoration)


def _numerators(a, D, t, P, pij_min, conefor, alive=None, pc_cut=None):
    link, p, keep = _graphs(D, t, P, pij_min, conefor)
    if pc_cut is not None:  # an edge forced in (Makurhini's changed pairs are not cut)
        i, j = pc_cut
        keep[i, j] = keep[j, i] = True
    alive = np.ones(len(a), bool) if alive is None else alive
    return _indices(a, link, p, keep, alive, conefor, np.inf)


@pytest.mark.parametrize("metric", ["IIC", "PC"])
def test_link_change_brute_force(metric):
    a, D = _scene(18, 91, False)
    t, P = 25.0, 0.5
    g = ls.HabitatGraph(a, D)
    rng = np.random.default_rng(3)
    i, j = np.triu_indices(18, 1)
    pick = rng.choice(len(i), 25, replace=False)
    new = D.copy()
    new[i[pick], j[pick]] = D[i[pick], j[pick]] * rng.uniform(0.2, 1.8, 25)
    new[j[pick], i[pick]] = new[i[pick], j[pick]]
    out = g.link_change(new, metric, distance=t, probability=P).set_index(["from", "to"])
    full = _numerators(a, D, t, P, 0.01, False)
    sent = 1 / (1 + MAKURHINI_UNCONNECTED)
    num = full["iic_reach"] + sent * full["unreach_aa"] if metric == "IIC" else full["pc_num"]
    for k in pick:
        a_, b_ = i[k], j[k]
        D2 = D.copy()
        D2[a_, b_] = D2[b_, a_] = new[a_, b_]
        if metric == "IIC":
            if not new[a_, b_] < t:
                expected = 0.0
            else:
                link_d = D.copy()
                link_d[a_, b_] = link_d[b_, a_] = min(D[a_, b_], new[a_, b_])
                r = _numerators(a, link_d, t, P, 0.01, False)
                expected = 100 * (num - r["iic_reach"]) / num
        else:
            r = _numerators(a, D2, t, P, 0.01, False, pc_cut=(a_, b_))
            expected = 100 * (num - r["pc_num"]) / num
        assert out.loc[(a_ + 1, b_ + 1), f"d{metric}_change"] == pytest.approx(expected, rel=1e-10, abs=1e-11)


@pytest.mark.parametrize("convention", ["makurhini", "conefor"])
def test_nodes_to_add_brute_force(convention):
    a, D = _scene(22, 93, True)
    t, P, pmin = 22.0, 0.5, (0.01 if convention == "makurhini" else 0.0)
    conefor = convention == "conefor"
    restoration = np.ones(22, int)
    restoration[[3, 10, 17, 21]] = 0
    g = ls.HabitatGraph(a, D, restoration=restoration, convention=convention)
    imp = g.node_importance(distance=t, probability=P, pij_min=pmin).set_index("id")
    base = restoration == 1
    b = _numerators(a, D, t, P, pmin, conefor, alive=base)
    for k in np.flatnonzero(~base):
        alive = base.copy()
        alive[k] = True
        w = _numerators(a, D, t, P, pmin, conefor, alive=alive)
        if convention == "makurhini":
            assert imp.loc[k + 1, "dPCres"] == pytest.approx(100 * (w["pc_num"] - b["pc_num"]) / w["pc_num"], rel=1e-10)
            assert imp.loc[k + 1, "dIICres"] == pytest.approx(
                100 * (w["iic_reach"] - b["iic_reach"]) / w["iic_reach"], rel=1e-10)
        else:
            row = imp.loc[k + 1]
            assert row["dPCadd"] == pytest.approx(100 * (w["pc_num"] - b["pc_num"]) / b["pc_num"], rel=1e-10)
            assert row["dIICadd"] == pytest.approx(100 * (w["iic_reach"] - b["iic_reach"]) / b["iic_reach"], rel=1e-10)
            assert row["dHadd"] == pytest.approx(100 * (w["H"] - b["H"]) / b["H"], rel=1e-10)
            assert row["dNCadd"] == pytest.approx(100 * (w["NC"] - b["NC"]) / b["NC"])
            assert row["dNLadd"] == pytest.approx(100 * (w["NL"] - b["NL"]) / b["NL"])
            ccp_b, ccp_w = b["comp_sq"] / b["total"] ** 2, w["comp_sq"] / w["total"] ** 2
            assert row["dCCPadd"] == pytest.approx(100 * (ccp_w - ccp_b) / ccp_b, rel=1e-10, abs=1e-12)
            assert row["dLCPadd"] == pytest.approx(100 * (w["comp_sq"] - b["comp_sq"]) / b["comp_sq"], rel=1e-10)
            assert row["dFadd"] == pytest.approx(100 * (w["F"] - b["F"]) / b["F"], rel=1e-10)
            assert row["dAWFadd"] == pytest.approx(100 * (w["AWF"] - b["AWF"]) / b["AWF"], rel=1e-10)
            assert np.isnan(row["dPC"])  # removal importances belong to the existing nodes
    if convention == "conefor":
        assert np.isnan(imp.loc[1, "dPCadd"]) and np.isfinite(imp.loc[1, "dPC"])
    with pytest.raises(ValueError, match="restoration"):
        ls.HabitatGraph(a, D, restoration=np.full(22, 2))


def test_link_importance_threads_and_errors():
    a, D = _scene(150, 97, False)
    one = ls.HabitatGraph(a, D, n_threads=1).link_importance("PC", distance=20.0)
    many = ls.HabitatGraph(a, D, n_threads=4).link_importance("PC", distance=20.0)
    pd.testing.assert_frame_equal(one, many, check_exact=True)
    with pytest.raises(ValueError, match="IIC' or 'PC"):
        ls.HabitatGraph(a, D).link_importance("H", distance=20.0)
