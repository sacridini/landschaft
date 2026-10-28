# SPDX-License-Identifier: GPL-3.0-or-later
"""Rivers as a network (ls.hydro.snap_points, upstream, upstream_catchment, network_distance, along_network,
centrality, dci) against ls.hydro's own D8 kernels and brute force (Terraflux's tests): the land upstream of a link's
last cell is what the D8 accumulation sums there, a point's catchment is the watershed of that point alone, distances
are the paths walked down cell by cell, the centrality of the links is igraph's by breadth-first search; the DCI, the
reach of every segment and the gain of removing each barrier are R's riverconn 0.3 (tests/data/reference/rivers,
tools/make_reference_river_dci.py) to 1e-12. On the DEMs of tests/data/reference/hydro/whitebox.npz."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import pytest

import landschaft as ls
from landschaft.hydro._rivers import _Segments

H = ls.hydro
REF = Path(__file__).parent / "data" / "reference"


@pytest.fixture(scope="module")
def dems():
    return np.load(REF / "hydro" / "whitebox.npz")


def filled(dems, name):
    dem = dems[f"{name}/filled"]
    dx, dy = dems[f"{name}/cell"]
    return dem, {"extent": (0.0, dem.shape[1] * dx, 0.0, dem.shape[0] * dy)}


def close(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1.0) or (math.isnan(a) and math.isnan(b))


def values(n):
    i = np.arange(n)
    return np.where(i % 97 == 5, np.nan, ((i * 7919) % 1000) / 10.0)


def classes(n):
    i = np.arange(n)
    return np.where(i % 61 == 0, np.nan, ((i * 31 + i // 50) % 4 + 1).astype(float))


def centre(kw, shape, k):
    ny, nx = shape
    x0, x1, y0, y1 = kw["extent"]
    rx, ry = (x1 - x0) / nx, (y1 - y0) / ny
    return x0 + (k % nx + 0.5) * rx, y1 - (k // nx + 0.5) * ry


def network(dem, kw):
    acc = H.flow_accumulation(dem, **kw)
    st = H.streams(acc, 20)
    links, table = H.stream_links(dem, st, **kw)
    return acc, st, links, table


def steps(dem, kw):
    """The step from each cell to the one it drains to, from the D8 codes (ESRI's): a side or the diagonal."""
    d = H.flow_direction(dem, **kw).values.ravel()
    ny, nx = dem.shape
    x0, x1, y0, y1 = kw["extent"]
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    moves = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
    length = np.full(d.size, min(dx, dy))
    to = np.full(d.size, -1)
    z = dem.ravel()
    for k, code in enumerate(d):
        if np.isnan(code) or int(code) not in moves:
            continue
        dr, dc = moves[int(code)]
        r, c = k // nx + dr, k % nx + dc
        if 0 <= r < ny and 0 <= c < nx and not np.isnan(z[r * nx + c]):
            to[k] = r * nx + c
            length[k] = math.hypot(dx, dy) if dr and dc else (dy if dr else dx)
    return length, to


@pytest.mark.parametrize("name", ["oregon", "walk"])
def test_the_land_upstream_of_the_links_is_what_drains_through_them(dems, name):
    dem, kw = filled(dems, name)
    n = dem.size
    nx = dem.shape[1]
    acc, st, links, lt = network(dem, kw)
    v, c = values(n).reshape(dem.shape), classes(n).reshape(dem.shape)
    out = H.upstream(dem, links=links, values=v, classes=c, stats=["count", "sum", "mean", "min", "max", "std"],
                     to_map="sum", **kw)
    t = out.table
    assert list(t.columns) == ["link", "downstream", "cells", "area", "count", "sum", "mean", "min", "max", "std",
                               "share_1", "share_2", "share_3", "share_4"]
    np.testing.assert_array_equal(t["link"], lt.index.to_numpy())
    np.testing.assert_array_equal(t["downstream"], lt["downstream"])

    def weighted(w):
        return H.flow_accumulation(dem, weights=w.reshape(dem.shape), **kw).values.ravel()

    ones = acc.values.ravel()
    vf, cf = v.ravel(), c.ravel()
    total = weighted(np.nan_to_num(vf))
    count = weighted((~np.isnan(vf)).astype(float))
    classified = weighted((~np.isnan(cf)).astype(float))
    class_2 = weighted((cf == 2).astype(float))
    x0, x1, y0, y1 = kw["extent"]
    cell = (x1 - x0) / dem.shape[1] * (y1 - y0) / dem.shape[0]
    ends = (lt["end_row"] * nx + lt["end_col"]).to_numpy()
    for i, k in enumerate(ends):
        row = t.iloc[i]
        assert row["cells"] == ones[k]
        assert row["area"] == ones[k] * cell
        assert row["count"] == count[k]
        assert close(row["sum"], total[k])
        assert close(row["mean"], total[k] / count[k])
        assert close(row["share_2"], class_2[k] / classified[k])
        assert close(row[["share_1", "share_2", "share_3", "share_4"]].sum(), 1.0)
    # min, max and std of a few links against their cells: the watershed of the last cell alone
    for i in (0, len(t) // 2, len(t) - 1):
        seed = np.zeros(n)
        seed[ends[i]] = 1
        ws = H.watersheds(dem, seed.reshape(dem.shape), **kw).values.ravel()
        up = vf[(ws == 1) & ~np.isnan(vf)]
        row = t.iloc[i]
        assert row["min"] == up.min() and row["max"] == up.max()
        assert close(row["std"], float(np.sqrt(np.mean((up - up.mean()) ** 2))))
    # the map: each link's sum on its cells
    lm = links.values.ravel()
    m = out.map.values.ravel()
    on = lm > 0
    np.testing.assert_array_equal(m[on], t["sum"].to_numpy()[lm[on].astype(int) - 1])
    assert np.isnan(m[~on]).all()
    # points on the links' last cells: the same rows
    ps = [centre(kw, dem.shape, k) for k in ends]
    pt = H.upstream(dem, points=ps, streams=st, values=v, stats="sum, mean", **kw).table
    assert list(pt.columns) == ["point", "downstream", "row", "col", "cells", "area", "sum", "mean"]
    np.testing.assert_array_equal(pt["cells"], t["cells"])
    np.testing.assert_array_equal(pt["downstream"], t["downstream"])
    np.testing.assert_array_equal(pt["row"], lt["end_row"])


def test_a_catchment_is_all_that_drains_to_its_point(dems):
    import geopandas as gpd

    dem, kw = filled(dems, "oregon")
    n = dem.size
    acc = H.flow_accumulation(dem, **kw)
    st = H.streams(acc, 20)
    ones = acc.values.ravel()
    big = sorted((k for k in range(n) if not np.isnan(ones[k])), key=lambda k: -ones[k])
    outlet = big[0]

    def alone(k, label=1):
        seed = np.zeros(n)
        seed[k] = label
        return H.watersheds(dem, seed.reshape(dem.shape), **kw).values.ravel()

    ws = alone(outlet)
    up = next(k for k in big if ws[k] == 1 and ones[outlet] / 2 > ones[k] > 50)
    other = next(k for k in big if ws[k] != 1 and ones[k] > 50)
    xy = [centre(kw, dem.shape, k) for k in (outlet, up, other)] + [(-5000.0, -5000.0)]
    pts = gpd.GeoDataFrame({"name": ["a", "b", "c", "d"]}, geometry=gpd.points_from_xy(*zip(*xy, strict=True)))
    with pytest.warns(UserWarning, match="point"):
        out = H.upstream_catchment(dem, pts, streams=st, **kw)
    t = out.table
    assert t["downstream"].tolist()[:3] == [0.0, 1.0, 0.0], "the second drains to the first"
    assert np.isnan(t["cells"][3]), "off the network: no catchment"
    v = out.polygons
    assert len(v) == 4 and v.geometry[3] is None and "name" in v
    x0, x1, y0, y1 = kw["extent"]
    cell = (x1 - x0) / dem.shape[1] * (y1 - y0) / dem.shape[0]
    for i, k in enumerate((outlet, up, other)):
        cells = float(np.sum(alone(k) == 1))
        assert t["cells"][i] == cells == ones[k]
        assert close(v.geometry[i].area, cells * cell)
    seed = np.zeros(n)
    for i, k in enumerate((outlet, up, other)):
        seed[k] = i + 1
    nested = H.watersheds(dem, seed.reshape(dem.shape), **kw).values
    np.testing.assert_array_equal(out.map.values, nested)


def test_points_are_moved_onto_the_streams(dems):
    dem, kw = filled(dems, "oregon")
    n = dem.size
    nx = dem.shape[1]
    acc = H.flow_accumulation(dem, **kw)
    st = H.streams(acc, 20)
    s, a = st.values.ravel(), acc.values.ravel()
    on = np.nonzero(s == 1)[0]
    rx = (kw["extent"][1] - kw["extent"][0]) / nx

    def nearest(p, cells):
        best = (math.inf, 0)
        for k in cells:
            cx, cy = centre(kw, dem.shape, k)
            d = math.hypot(cx - p[0], cy - p[1])
            if d < best[0]:
                best = (d, k)
        return best

    cx, cy = centre(kw, dem.shape, on[len(on) // 3])
    ps = [(cx + 2 * rx, cy), (cx + 0.3 * rx, cy - 0.2 * rx), (-9000.0, 0.0)]
    with pytest.warns(UserWarning, match="not moved"):
        out = H.snap_points(ps, st, distance=2.5 * rx)
    for i, p in enumerate(ps[:2]):
        d, k = nearest(p, on)
        assert (out.geometry[i].x, out.geometry[i].y) == centre(kw, dem.shape, k)
        assert out["moved"][i] == d and out["row"][i] == k // nx and out["stream"][i] == 1
    assert (out.geometry[2].x, out.geometry[2].y) == (-9000.0, 0.0)
    assert out["snapped"].tolist() == [1, 1, 0]
    big = max(a[k] for k in on) / 2
    out = H.snap_points(ps[:1], st, accumulation=acc, distance=100000, min_accumulation=big)
    j = int(out["row"][0] * nx + out["col"][0])
    assert a[j] >= big and s[j] == 1
    large = [k for k in on if a[k] >= big]
    assert (out.geometry[0].x, out.geometry[0].y) == centre(kw, dem.shape, nearest(ps[0], large)[1])
    assert n == dem.size
    with pytest.raises(ValueError, match="min_accumulation"):
        H.snap_points(ps, st, min_accumulation=5)


def test_the_dci_is_riverconns():
    refs = json.loads((REF / "rivers" / "river_dci.json").read_text(encoding="utf-8"))

    def near(a, b, what):
        assert abs(a - b) <= 1e-12 * max(abs(b), 1e-3), (what, a, b)

    for name, c in refs.items():
        net = _Segments([b if b >= 0 else None for b in c["below"]], c["length"], c["up"], c["down"])
        r = c["riverconn"]
        total = net.weight.sum()
        for symmetric, key in ((True, "full_sym"), (False, "full_asym")):
            near(float(np.sum(net.weight * net.reach(symmetric)) / total**2), r[key][0], f"{name} {key}")
        for symmetric, from_, key in ((True, False, "reach_to_sym"), (False, False, "reach_to_asym"),
                                      (False, True, "reach_from_asym")):
            for i, x in enumerate(net.reach(symmetric, from_)):
                near(x / total, r[key][i], f"{name} {key} {i}")
        p, d = net.dci(True)[0]
        near(p / 100, r["full_sym"][0], f"{name} dci_p")
        near(d / 100, r["reach_to_sym"][0], f"{name} dci_d")
        for j, bid in enumerate(r["d_barrier"]):
            s = c["barrier"].index(bid)
            kept = net.up[s], net.down[s]
            net.up[s] = net.down[s] = 1.0
            without = net.dci(True)[0][0]
            net.up[s], net.down[s] = kept
            near(without / 100, r["d_index_value"][j], f"{name} without {bid}")
            near((without - p) / p * 100, r["d_index"][j], f"{name} d_index of {bid}")


def test_distances_and_reaches_follow_the_river(dems):
    dem, kw = filled(dems, "oregon")
    n = dem.size
    nx = dem.shape[1]
    _, st, _, lt = network(dem, kw)
    length, to = steps(dem, kw)

    def path(k):
        out = [(k, 0.0)]
        while to[k] >= 0:
            out.append((int(to[k]), out[-1][1] + length[k]))
            k = int(to[k])
        return out

    order, below = lt["order"].to_numpy(), lt["downstream"].to_numpy()
    cand = [i for i in range(len(lt)) if below[i] > 0]
    mid = max(cand, key=lambda i: (order[i], -i))
    sr, sc = lt["start_row"].to_numpy(), lt["start_col"].to_numpy()
    er, ec = lt["end_row"].to_numpy(), lt["end_col"].to_numpy()
    picks = [sr[0] * nx + sc[0], sr[3] * nx + sc[3], er[mid] * nx + ec[mid], sr[-1] * nx + sc[-1]]
    ps = [centre(kw, dem.shape, k) for k in picks]
    t = H.network_distance(dem, ps, streams=st, **kw)
    assert len(t) == 6
    for _, row in t.iterrows():
        a, b = picks[int(row["from"]) - 1], picks[int(row["to"]) - 1]
        pa, pb = path(a), path(b)
        da = dict(pa)
        meet = next((da[k] + db for k, db in pb if k in da), None)
        if meet is None:
            assert np.isnan(row["network"]) and row["connected"] == 0
        else:
            assert close(row["network"], meet) and row["connected"] == 1
            flows = any(k == b for k, _ in pa) or any(k == a for k, _ in pb)
            assert row["flow"] == int(flows)
        qa, qb = ps[int(row["from"]) - 1], ps[int(row["to"]) - 1]
        assert close(row["euclidean"], math.hypot(qa[0] - qb[0], qa[1] - qb[1]))
    # up and down the river from the third point (the last cell of a link in the middle)
    dx = (kw["extent"][1] - kw["extent"][0]) / nx
    up, down = 12 * dx, 7 * dx
    r = H.along_network(dem, ps[2:3], st, upstream=up, downstream=down, **kw)
    m, dist = r.map.values.ravel(), r.distance.values.ravel()
    s = st.values.ravel()
    k0 = picks[2]
    downstream = {k: d for k, d in path(k0) if d <= down}
    reached = 0
    for k in range(n):
        d = downstream.get(k)
        if d is None and s[k] == 1:
            p = path(k)
            i = next((i for i, (j, _) in enumerate(p) if j == k0), None)
            if i is not None and all(s[j] == 1 for j, _ in p[:i]) and p[i][1] <= up:
                d = p[i][1]
        if d is None:
            assert m[k] == 0 and np.isnan(dist[k]), k
        else:
            assert m[k] == 1 and close(dist[k], d), k
            reached += 1
    assert reached > 15


def test_centrality_counts_the_paths_through_each_link(dems):
    dem, kw = filled(dems, "oregon")
    _, _, _, lt = network(dem, kw)
    t = H.centrality(lt)
    n = len(t)
    ids = lt.index.to_numpy(float)
    below = lt["downstream"].to_numpy(float)
    np.testing.assert_array_equal(t["link"], ids)
    index = {v: i for i, v in enumerate(ids)}
    adj = [[] for _ in range(n)]
    for i in range(n):
        j = index.get(below[i])
        if j is not None:
            adj[i].append(j)
            adj[j].append(i)

    def bfs(s):
        d = [-1] * n
        d[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            for u in adj[v]:
                if d[u] < 0:
                    d[u] = d[v] + 1
                    q.append(u)
        return d

    alld = [bfs(v) for v in range(n)]
    length = lt["length"].to_numpy()
    for v in range(n):
        reach = [u for u in range(n) if alld[v][u] >= 0]
        total = sum(alld[v][u] for u in reach)
        through = sum(1 for a in reach for b in reach
                      if a < b and v not in (a, b) and alld[a][v] + alld[v][b] == alld[a][b])
        row = t.iloc[v]
        assert row["degree"] == len(adj[v])
        assert row["betweenness"] == through
        if total > 0:
            assert close(row["closeness"], 1 / total)
        ups = []
        for u in range(n):
            x = u
            while True:
                if x == v:
                    ups.append(u)
                    break
                j = index.get(below[x])
                if j is None:
                    break
                x = j
        assert close(row["upstream_length"], sum(length[u] for u in ups))
        assert row["magnitude"] == sum(1 for u in ups if ids[u] not in below)


def test_a_dam_splits_the_river_and_its_removal_is_ranked(dems):
    import geopandas as gpd

    dem, kw = filled(dems, "oregon")
    n = dem.size
    nx = dem.shape[1]
    acc, st, _, lt = network(dem, kw)
    a, s = acc.values.ravel(), st.values.ravel()
    outlet = max((k for k in range(n) if s[k] == 1), key=lambda k: a[k])
    ends = [int(r * nx + c) for r, c in zip(lt["end_row"], lt["end_col"], strict=True)]
    length, to = steps(dem, kw)

    def drains_to(k, t):
        while True:
            if k == t:
                return True
            if to[k] < 0:
                return False
            k = int(to[k])

    ends = sorted((k for k in ends if k != outlet and a[k] > 40 and drains_to(k, outlet)), key=lambda k: -a[k])
    dams = [ends[0], ends[len(ends) // 2]]
    xy = [centre(kw, dem.shape, k) for k in dams]
    barriers = gpd.GeoDataFrame({"passability": [0.0, 0.5]}, geometry=gpd.points_from_xy(*zip(*xy, strict=True)))
    out = H.dci(dem, st, barriers, **kw)
    seg = out.segments.values.ravel()
    weight: dict[int, float] = {}
    for k in range(n):
        if seg[k] > 0:
            weight[int(seg[k])] = weight.get(int(seg[k]), 0.0) + length[k]
    w1, w2 = weight[1], weight[2]
    s0 = int(seg[outlet])
    w0 = weight[s0]
    total = w0 + w1 + w2
    row = out.table[out.table["network"] == s0].iloc[0]
    assert row["barriers"] == 2 and close(row["weight"], total)

    def c(i, j):
        if i == j:
            return 1.0

        def path(x):
            if x == 0:
                return [0]
            if x == 1 and drains_to(dams[0], dams[1]):
                return [1, 2, 0]
            if x == 2 and drains_to(dams[1], dams[0]):
                return [2, 1, 0]
            return [x, 0]

        pi, pj = path(i), path(j)
        meet = next(x for x in pi if x in pj)
        steps_ = [x for x in pi[:pi.index(meet)]] + [x for x in pj[:pj.index(meet)]]
        return math.prod(0.0 if x == 1 else 0.25 for x in steps_)

    ws = [w0, w1, w2]
    want = sum(ws[i] * ws[j] * c(i, j) for i in range(3) for j in range(3)) / total**2 * 100
    assert close(row["dci_p"], want)
    gain, rank = out.barriers["d_index"].to_numpy(), out.barriers["rank"].to_numpy()
    assert (gain > 0).all() and (rank[0] < rank[1]) == (gain[0] > gain[1])
    reach0 = sum(c(0, j) * ws[j] for j in range(3)) / total * 100
    assert close(out.reach.values.ravel()[outlet], reach0) and close(row["dci_d"], reach0)
    with pytest.raises(ValueError, match="passability"):
        H.dci(dem, st, barriers.assign(passability=[2.0, 0.5]), **kw)
