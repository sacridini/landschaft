# SPDX-License-Identifier: GPL-3.0-or-later
"""Habitat-graph nodes and distances from rasters and polygons (Makurhini's nodesfile / distancefile)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

REF = Path(__file__).parent / "data" / "reference" / "conn"
SPATIAL = REF / "spatial"
RASTER = REF / "habitat_nodes_raster.tif"
POLYGONS = REF / "polygons.gpkg"


def _ref(name: str) -> pd.DataFrame:
    return pd.read_csv(SPATIAL / f"{name}.csv.gz", float_precision="round_trip")


def _merge(ours: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    m = ours.merge(ref, left_on=["from", "to"], right_on=["From", "To"], how="right", indicator=True)
    assert (m["_merge"] == "both").all()
    return m


# ------------------------------------------------------------ Makurhini: raster


@pytest.mark.parametrize("unit", ["m2", "ha"])
def test_raster_nodes(unit):
    ours = ls.conn.node_table(RASTER, area_unit=unit)
    ref = _ref(f"raster_nodes_{unit}")
    np.testing.assert_array_equal(ours["id"], ref["Id"])
    np.testing.assert_array_equal(ours["attribute"], ref["attribute"])  # same bits


def test_raster_nodes_weighted():
    ours = ls.conn.node_table(RASTER, attribute=np.arange(1, 399) / 398, weighted=True, area_unit="km2")
    np.testing.assert_array_equal(ours["attribute"], _ref("raster_nodes_weighted")["attribute"])


def test_raster_centroid_distances():
    m = _merge(ls.conn.distance_table(RASTER, distance="centroid"), _ref("raster_centroid"))
    np.testing.assert_array_equal(m["distance"], m["Distance"])  # R's 80-bit colMeans, same bits


def test_raster_edge_distances():
    m = _merge(ls.conn.distance_table(RASTER, distance="edge"), _ref("raster_edge"))
    # GEOS measures between terra's polygon vertices: a few values differ in the last bit.
    np.testing.assert_allclose(m["distance"], m["Distance"], rtol=3e-16, atol=0)
    assert (m["distance"] == m["Distance"]).mean() > 0.999


def test_raster_end_to_end():
    g = ls.HabitatGraph.from_raster(RASTER, area_unit="ha", distance="edge")
    imp = g.node_importance("PC", distance=5000, probability=0.5).set_index("id")
    ref = _ref("raster_edge_pc_nodes").set_index("id")
    for col in ("dPC", "dPCintra", "dPCflux"):
        np.testing.assert_array_equal(np.round(imp.loc[ref.index, col].to_numpy(), 7), ref[col].to_numpy())
    np.testing.assert_allclose(imp.loc[ref.index, "dPCconnector"], ref["dPCconnector"], rtol=1e-10, atol=1e-12)
    ov = g.overall("PC", distance=5000, probability=0.5).iloc[0]
    ref_ov = _ref("raster_edge_pc_overall").set_index("Index")["Value"]
    assert ov["PCnum"] == pytest.approx(ref_ov["PCnum"], rel=1e-13)


# ---------------------------------------------------------- Makurhini: polygons


def test_vector_nodes():
    ours = ls.conn.node_table(gpd.read_file(POLYGONS), id="id", area_unit="km2")
    ref = _ref("vector_nodes_km2")
    np.testing.assert_array_equal(ours["id"], ref["id"])
    np.testing.assert_array_equal(ours["attribute"], ref["Area"])  # GDAL's area sums, same bits


@pytest.mark.parametrize("kind,name,kw", [
    ("centroid", "vector_centroid", {}),
    ("edge", "vector_edge", {}),
    ("edge", "vector_edge_2km", {"max_distance": 2, "distance_unit": "km"}),
])
def test_vector_distances(kind, name, kw):
    ours = ls.conn.distance_table(gpd.read_file(POLYGONS), id="id", distance=kind, **kw)
    ref = _ref(name)
    assert len(ours) == len(ref)
    m = _merge(ours, ref)
    np.testing.assert_array_equal(m["distance"], m["Distance"])


def test_vector_end_to_end():
    g = ls.HabitatGraph.from_vector(gpd.read_file(POLYGONS), id="id", area_unit="km2")
    imp = g.node_importance("IIC", distance=3000).set_index("id")
    ref = _ref("vector_centroid_iic_nodes").set_index("id")
    for col in ("dIIC", "dIICintra", "dIICflux"):
        np.testing.assert_array_equal(np.round(imp.loc[ref.index, col].to_numpy(), 7), ref[col].to_numpy())
    np.testing.assert_allclose(imp.loc[ref.index, "dIICconnector"], ref["dIICconnector"], rtol=1e-10, atol=1e-12)
    ov = g.overall("IIC", distance=3000).iloc[0]
    assert ov["IICnum"] == pytest.approx(_ref("vector_centroid_iic_overall").set_index("Index")["Value"]["IICnum"],
                                         rel=1e-13)


# --------------------------------------------------------------- brute force


def _edge_brute(ids: np.ndarray, resx: float, resy: float) -> dict:
    out = {}
    cells = {v: np.argwhere(ids == v) for v in np.unique(ids[ids > 0])}
    keys = sorted(cells)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ca, cb = cells[keys[a]], cells[keys[b]]
            gr = np.maximum(np.abs(ca[:, None, 0] - cb[None, :, 0]) - 1, 0) * resy
            gc = np.maximum(np.abs(ca[:, None, 1] - cb[None, :, 1]) - 1, 0) * resx
            out[(keys[a], keys[b])] = np.sqrt(gc * gc + gr * gr).min()
    return out


@pytest.mark.parametrize("seed,res", [(1, (10.0, 10.0)), (2, (30.0, 12.5)), (3, (1.0, 4.0))])
def test_edge_distances_brute_force(seed, res):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 7, (25, 30)) * (rng.random((25, 30)) < 0.15)
    ids = ids.astype(np.int32)
    import xarray as xr

    da = xr.DataArray(ids, dims=("y", "x"), coords={"y": 1000 - (np.arange(25) + 0.5) * res[1],
                                                   "x": (np.arange(30) + 0.5) * res[0]})
    da = da.rio.write_nodata(0)
    table = ls.conn.distance_table(da, distance="edge", n_threads=2)
    brute = _edge_brute(ids, *res)
    got = {(f, t): d for f, t, d in table.itertuples(index=False)}
    assert got.keys() == brute.keys()
    for key, d in brute.items():
        assert got[key] == pytest.approx(d, rel=1e-15, abs=1e-12)
    limited = ls.conn.distance_table(da, distance="edge", max_distance=40.0)
    expected = {k: v for k, v in brute.items() if v <= 40.0}
    assert {(f, t) for f, t, _ in limited.itertuples(index=False)} == set(expected)


def test_edge_distances_match_geos_on_cell_polygons():
    rng = np.random.default_rng(9)
    ids = (rng.integers(1, 5, (12, 14)) * (rng.random((12, 14)) < 0.3)).astype(np.int32)
    table = ls.conn.distance_table(np.ma.masked_array(ids, mask=ids == 0), distance="edge")
    polys = {}
    for v in np.unique(ids[ids > 0]):
        cells = [shapely.box(c, 11 - r, c + 1, 12 - r) for r, c in np.argwhere(ids == v)]
        polys[v] = shapely.unary_union(cells)
    for f, t, d in table.itertuples(index=False):
        assert d == pytest.approx(shapely.distance(polys[f], polys[t]), abs=1e-12)


def test_point_distances_with_limit_equal_full_table():
    rng = np.random.default_rng(4)
    pts = gpd.GeoDataFrame({"id": np.arange(300)}, geometry=gpd.points_from_xy(*rng.uniform(0, 1e4, (2, 300))))
    full = ls.conn.distance_table(pts, id="id")
    limited = ls.conn.distance_table(pts, id="id", max_distance=800)
    expected = full[full["distance"] <= 800].reset_index(drop=True)
    pd.testing.assert_frame_equal(limited, expected)
    assert len(full) == 300 * 299 // 2


def test_habitat_classes_are_labelled():
    from scipy import ndimage

    rng = np.random.default_rng(5)
    classes = rng.integers(0, 4, (40, 50)).astype(np.int32)
    g = ls.HabitatGraph.from_raster(classes, habitat=[1, 2], distance="edge", directions=8)
    labels, n = ndimage.label(np.isin(classes, [1, 2]), structure=np.ones((3, 3)))
    assert g.n_nodes == n
    assert sorted(g.attributes) == sorted(np.bincount(labels.ravel())[1:].astype(float))


def test_to_raster_and_geodataframe():
    g = ls.HabitatGraph.from_raster(RASTER, distance="centroid")
    imp = g.node_importance("PC", distance=8000)
    r = g.to_raster(imp, "dPC")
    values = ls.load_raster(RASTER, validate=None).squeeze().values
    node = imp.set_index("id")["dPC"]
    inside = np.isfinite(r.values)
    np.testing.assert_array_equal(r.values[inside], node.loc[values[inside]].to_numpy())
    assert np.isnan(r.values[~inside]).all()
    v = gpd.read_file(POLYGONS)
    gv = ls.HabitatGraph.from_vector(v, id="id", distance="edge")
    out = gv.to_geodataframe(gv.node_importance("IIC", distance=500))
    assert list(out["id"]) == list(v["id"]) and out.geometry.equals(v.geometry)
    with pytest.raises(ValueError, match="from_raster"):
        gv.to_raster(np.zeros(gv.n_nodes))


def test_units_and_errors():
    t_m = ls.conn.distance_table(RASTER, distance="centroid")
    t_km = ls.conn.distance_table(RASTER, distance="centroid", distance_unit="km")
    np.testing.assert_array_equal(t_km["distance"], t_m["distance"] * 0.001)
    with pytest.raises(ValueError, match="unknown distance unit"):
        ls.conn.distance_table(RASTER, distance_unit="parsec")
    with pytest.raises(ValueError, match="distance must be"):
        ls.conn.distance_table(RASTER, distance="circuit")
    with pytest.raises(ValueError, match="one value per patch"):
        ls.conn.node_table(RASTER, attribute=[1.0, 2.0])


# ------------------------------------------------------- least-cost distances

RESISTANCE = REF / "resistance_matrix.tif"


@pytest.mark.parametrize("name,kw", [
    ("raster_lcd", {}),
    ("raster_lcd_4", {"cost_directions": 4}),
    ("raster_lcd_mean", {"cost_function": "mean"}),
])
def test_raster_least_cost(name, kw):
    ours = ls.conn.distance_table(RASTER, distance="least-cost", resistance=RESISTANCE, **kw)
    m = _merge(ours, _ref(name))
    np.testing.assert_array_equal(m["distance"], m["Distance"])  # gdistance's bits


def test_vector_least_cost_with_unreachable_pairs():
    v = gpd.read_file(POLYGONS)
    ours = ls.conn.distance_table(v, id="id", distance="least-cost", resistance=REF / "polygons_resistance.tif")
    ref = _ref("vector_lcd")
    assert len(ours) == len(ref)
    m = _merge(ours, ref)
    np.testing.assert_array_equal(m["distance"], m["Distance"])
    # Pairs cut off by the NA walls take their centroid distance (Makurhini).
    dropped = ls.conn.distance_table(v, id="id", distance="least-cost", resistance=REF / "polygons_resistance.tif",
                                     unreachable="drop")
    assert 0 < len(dropped) < len(ours)
    # A limit keeps pairs beyond it out, it does not make them unreachable.
    limit = float(np.median(ours["distance"]))
    limited = ls.conn.distance_table(v, id="id", distance="least-cost", resistance=REF / "polygons_resistance.tif",
                                     max_distance=limit)
    pd.testing.assert_frame_equal(limited, ours[ours["distance"] <= limit].reset_index(drop=True))


def test_least_cost_end_to_end():
    g = ls.HabitatGraph.from_raster(RASTER, area_unit="ha", distance="least-cost", resistance=RESISTANCE)
    imp = g.node_importance("PC", distance=250000, probability=0.5).set_index("id")
    ref = _ref("raster_lcd_pc_nodes").set_index("id")
    for col in ("dPC", "dPCintra", "dPCflux"):
        np.testing.assert_array_equal(np.round(imp.loc[ref.index, col].to_numpy(), 7), ref[col].to_numpy())


def _gdistance_graph(values, xres, yres, directions, function):
    """The graph gdistance builds (brute force, with numpy): edges and costs."""
    from scipy.sparse import coo_matrix

    ny, nx = values.shape
    xs = (np.arange(nx) + 0.5) * xres
    ys = ny * yres - (np.arange(ny) + 0.5) * yres
    offsets = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if directions == 8 else [])
    rows, cols, costs = [], [], []
    for r in range(ny):
        for c in range(nx):
            for dr, dc in offsets:
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < ny and 0 <= c2 < nx):
                    continue
                a, b = r * nx + c, r2 * nx + c2
                va, vb = values[r, c], values[r2, c2]
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue
                lo, hi = min(a, b), max(a, b)
                v_lo, v_hi = values.flat[lo], values.flat[hi]
                cond = 1 / v_hi if function == "makurhini" else 1 / ((v_lo + v_hi) / 2)
                d = np.sqrt((xs[c] - xs[c2]) ** 2 + (ys[r] - ys[r2]) ** 2)
                rows.append(a)
                cols.append(b)
                costs.append(1 / ((1 / d) * cond))
    return coo_matrix((costs, (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()


@pytest.mark.parametrize("directions,function", [(8, "makurhini"), (4, "makurhini"), (8, "mean")])
def test_least_cost_against_scipy(directions, function):
    import xarray as xr
    from scipy.sparse.csgraph import dijkstra

    rng = np.random.default_rng(directions + len(function))
    values = rng.uniform(1, 50, (30, 40))
    values[10, 5:35] = np.nan  # a wall with a gap
    values[20:, 20] = np.nan
    res = (25.0, 40.0)
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": 30 * res[1] - (np.arange(30) + 0.5) * res[1],
                                                      "x": (np.arange(40) + 0.5) * res[0]})
    cells = rng.choice(np.flatnonzero(np.isfinite(values)), 15, replace=False)
    r, c = np.divmod(cells, 40)
    x, y = (c + 0.5) * res[0], 30 * res[1] - (r + 0.5) * res[1]
    pts = gpd.GeoDataFrame({"id": np.arange(15)}, geometry=gpd.points_from_xy(x, y))
    table = ls.conn.distance_table(pts, id="id", distance="least-cost", resistance=da, cost_directions=directions,
                                   cost_function=function, unreachable="drop")
    graph = _gdistance_graph(values, *res, directions, function)
    full = dijkstra(graph, directed=False, indices=cells)[:, cells]
    got = {(f, t): d for f, t, d in table.itertuples(index=False)}
    for i in range(15):
        for j in range(i + 1, 15):
            if np.isfinite(full[i, j]):
                assert got[(i, j)] == pytest.approx(full[i, j], rel=1e-13)
            else:
                assert (i, j) not in got
    limited = ls.conn.distance_table(pts, id="id", distance="least-cost", resistance=da, cost_directions=directions,
                                     cost_function=function, unreachable="drop", max_distance=float(np.nanmedian(full)))
    assert {(f, t) for f, t, _ in limited.itertuples(index=False)} == {
        k for k, d in got.items() if d <= float(np.nanmedian(full))}


def test_least_cost_points_outside_or_in_na():
    values = np.ones((10, 10))
    values[0, 0] = np.nan
    pts = gpd.GeoDataFrame({"id": [1, 2, 3, 4]},
                           geometry=gpd.points_from_xy([0.5, 5.5, 50.0, 6.5], [9.5, 5.5, 5.0, 5.5]))
    t = ls.conn.distance_table(pts, id="id", distance="least-cost", resistance=values, unreachable="drop")
    pairs = {(f, to) for f, to, _ in t.itertuples(index=False)}
    assert pairs == {(2, 4)}  # 1 sits in an NA cell, 3 outside the raster
    t = ls.conn.distance_table(pts, id="id", distance="least-cost", resistance=values)
    assert len(t) == 6  # the others fall back to centroid distances
    with pytest.raises(ValueError, match="resistance raster"):
        ls.conn.distance_table(pts, id="id", distance="least-cost")
    with pytest.raises(ValueError, match="positive"):
        ls.conn.distance_table(pts, id="id", distance="least-cost", resistance=np.zeros((3, 3)))


# ---------------------------------------------------- commute-time distances

@pytest.mark.parametrize("name,nodes,kw", [
    ("raster_commute", RASTER, {"resistance": RESISTANCE}),
    ("vector_commute", "polygons", {"id": "id", "resistance": REF / "polygons_resistance.tif"}),
    ("vector_commute_4", "polygons", {"id": "id", "resistance": REF / "polygons_resistance.tif", "cost_directions": 4,
                                      "cost_function": "mean"}),
])
def test_commute_time_matches_gdistance(name, nodes, kw):
    if nodes == "polygons":
        nodes = gpd.read_file(POLYGONS)
    ours = ls.conn.distance_table(nodes, distance="commute-time", **kw)
    ref = _ref(name)
    m = _merge(ours, ref)
    # gdistance solves with CHOLMOD, landschaft with SuperLU.
    np.testing.assert_allclose(m["distance"], m["Distance"], rtol=1e-11)
    if name != "raster_commute":  # the whole table: nodes on NA cells are left out, as gdistance does
        assert len(ours) == len(ref)


@pytest.mark.parametrize("directions,function", [(8, "makurhini"), (4, "mean")])
def test_commute_time_against_pseudo_inverse(directions, function):
    import xarray as xr

    rng = np.random.default_rng(directions)
    values = rng.uniform(1, 50, (14, 18))
    values[:, 9] = np.nan  # two separate parts
    res = (25.0, 40.0)
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": 14 * res[1] - (np.arange(14) + 0.5) * res[1],
                                                      "x": (np.arange(18) + 0.5) * res[0]})
    cells = rng.choice(np.flatnonzero(np.isfinite(values)), 10, replace=False)
    cells[1] = cells[0]  # two nodes on one cell
    r, c = np.divmod(cells, 18)
    x, y = (c + 0.5) * res[0], 14 * res[1] - (r + 0.5) * res[1]
    pts = gpd.GeoDataFrame({"id": np.arange(10)}, geometry=gpd.points_from_xy(x, y))
    table = ls.conn.distance_table(pts, id="id", distance="commute-time", resistance=da, cost_directions=directions,
                                   cost_function=function, unreachable="drop")
    costs = _gdistance_graph(values, *res, directions, function)
    W = costs.copy()
    W.data = 1 / W.data
    W = (W + W.T).toarray()
    volume = W.sum()
    part = (np.arange(18) >= 9)[c]
    got = {(f, t): d for f, t, d in table.itertuples(index=False)}
    for side in (False, True):
        cols = np.flatnonzero(np.isfinite(values).ravel() & ((np.arange(values.size) % 18 >= 9) == side))
        L = np.diag(W[cols][:, cols].sum(axis=1)) - W[cols][:, cols]
        Lp = np.linalg.inv(L + 1 / cols.size) - 1 / cols.size  # the pseudo-inverse of a connected graph's Laplacian
        where = np.searchsorted(cols, cells)
        for i in range(10):
            for j in range(i + 1, 10):
                if part[i] != side or part[j] != side:
                    continue
                a, b = where[i], where[j]
                expected = (Lp[a, a] + Lp[b, b] - 2 * Lp[a, b]) * volume
                assert got[(i, j)] == pytest.approx(expected, rel=1e-9, abs=1e-6)
    assert all(part[i] == part[j] for i, j in got)  # separate parts: no pair (dropped)
    euclid = ls.conn.distance_table(pts, id="id", distance="commute-time", resistance=da, cost_directions=directions,
                                    cost_function=function)
    assert len(euclid) == 45  # the separate parts take the centroid distance


def test_commute_time_nodes_on_na_or_graph():
    values = np.ones((10, 10))
    values[0, 0] = np.nan
    pts = gpd.GeoDataFrame({"id": [1, 2, 3, 4]},
                           geometry=gpd.points_from_xy([0.5, 5.5, 50.0, 6.5], [9.5, 5.5, 5.0, 5.5]))
    t = ls.conn.distance_table(pts, id="id", distance="commute-time", resistance=values)
    # 1 sits in an NA cell, 3 outside the raster: left out, as gdistance does
    assert {(f, to) for f, to, _ in t.itertuples(index=False)} == {(2, 4)}
    g = ls.HabitatGraph.from_vector(pts, id="id", distance="commute-time", resistance=values)
    assert g.n_nodes == 4
    with pytest.raises(ValueError, match="commute-time distances need a resistance"):
        ls.conn.distance_table(pts, id="id", distance="commute-time")
