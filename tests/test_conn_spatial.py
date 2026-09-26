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
        ls.conn.distance_table(RASTER, distance="least-cost")
    with pytest.raises(ValueError, match="one value per patch"):
        ls.conn.node_table(RASTER, attribute=[1.0, 2.0])
