# SPDX-License-Identifier: GPL-3.0-or-later
"""Grids and raster ProtConn (Makurhini's make_grid, MK_Connect_grid, MK_dECA_grid, MK_ProtConn_raster)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

REF = Path(__file__).parent / "data" / "reference" / "conn"
GRID = REF / "grid"
PAS = gpd.read_file(REF / "protconn_areas.gpkg")
REGIONS = gpd.read_file(REF / "protconn_regions.gpkg")
REGION = REGIONS[REGIONS["id"] == 1]


def _ref(name: str) -> pd.DataFrame:
    return pd.read_csv(GRID / f"{name}.csv.gz", keep_default_na=False, dtype=str)


def _num(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col.replace({"NA": np.nan, "NaN": np.nan}), errors="coerce").to_numpy(float)


@pytest.mark.parametrize("kind,hexagonal", [("hex", True), ("square", False)])
def test_make_grid_same_coordinates(kind, hexagonal):
    ref = shapely.from_wkt(_ref(f"grid_{kind}")["wkt"].to_numpy())
    ours = ls.conn.make_grid(REGION, 2.5e7, hexagonal=hexagonal)
    assert len(ours) == len(ref)
    for a, b in zip(ours.geometry.values, ref):
        np.testing.assert_array_equal(shapely.get_coordinates(a), shapely.get_coordinates(b))


@pytest.mark.parametrize("name,protconn,pij_min", [
    ("connect_grid_protconn", True, np.inf),  # MK_Connect_grid's default cuts every link
    ("connect_grid_protconn_pij0", True, 0.0),
    ("connect_grid_pc", False, None),
])
def test_connectivity_grid(name, protconn, pij_min):
    ref = _ref(name).rename(columns={"Protected.surface": "Protected_surface", "ECA.Normalized": "ECA_Normalized"})
    ours = ls.conn.connectivity_grid(PAS, REGION, distance=3000, cell_area=2.5e7, protconn_index=protconn,
                                     probability=0.5, transboundary=2000, area_unit="ha", pij_min=pij_min)
    for col in ref.columns[1:]:
        a, b = ours[col].to_numpy(float), _num(ref[col])
        assert np.array_equal(np.isnan(a), np.isnan(b)), col
        both = ~np.isnan(b)
        # MK_Connect_grid rounds to 5 decimals (and leaves the single node's PC as it is).
        if col == "PC" and not protconn:
            np.testing.assert_allclose(a[both], b[both], rtol=1e-12, atol=5e-6, err_msg=col)
        else:  # numpy's and R's round() may differ in the last bit
            np.testing.assert_allclose(np.round(a[both], 5), b[both], rtol=1e-14, atol=1e-12, err_msg=col)


@pytest.mark.parametrize("case,ttype,tb,bound", [("nodes", "nodes", 4000.0, False), ("region", "region", 4000.0, False),
                                                 ("bound", "nodes", 6000.0, True)])
def test_protconn_raster(case, ttype, tb, bound):
    ref = _ref("protconn_raster")
    ref = ref[ref["case"] == case]
    ours = ls.conn.protconn_raster(REF / "protconn_areas.tif", REGION, distance=[2000.0, 6000.0], transboundary=tb,
                                   transboundary_type=ttype, bound=bound, area_unit="ha")
    for _, row in ref.iterrows():
        v = np.nan if row["value"] == "NA" else float(row["value"])
        mine = ours[ours["distance"] == float(row["distance"])].iloc[0][row["name"]]
        if np.isnan(v):
            assert np.isnan(mine), row["name"]
        else:
            assert round(mine, 4) == v, row["name"]  # MK_ProtConn_raster rounds to 4 decimals


def test_eca_change_grid():
    ref = _ref("deca_grid")
    poly = gpd.read_file(REF / "polygons.gpkg")
    steps = pd.read_csv(REF / "deca_steps.csv")
    times = [poly[poly["id"].isin(steps[(steps["kind"] == "vector") & (steps["time"] == t)]["id"])]
             .reset_index(drop=True) for t in (1, 2, 3)]
    region = gpd.GeoDataFrame(geometry=[shapely.box(*poly.total_bounds)], crs=poly.crs)
    ours = ls.conn.eca_change_grid(times, region, distance=3000, metric="PC", cell_area=6e7, probability=0.5,
                                   area_unit="km2")
    assert list(ours["OBJECTID"]) == list(ref["OBJECTID"].astype(int))
    assert list(ours["time"]) == list(ref["Time"])
    for rcol, col in [("ECA.i", "ECA_i"), ("ECA.j", "ECA_j"), ("ECAnorm.i", "ECAnorm_i"), ("ECAnorm.j", "ECAnorm_j"),
                      ("dA", "dA"), ("dECA", "dECA"), ("rECA", "rECA")]:
        a, b = ours[col].to_numpy(float), _num(ref[rcol])
        assert np.array_equal(np.isnan(a), np.isnan(b)), col
        np.testing.assert_allclose(a[~np.isnan(b)], b[~np.isnan(b)], rtol=1e-12, err_msg=col)
    assert list(ours["comparison"]) == list(ref["comparisons"])
    assert list(ours["change"]) == list(ref["Type.Change"])


def test_grid_options_and_errors():
    clipped = ls.conn.make_grid(REGION, 2.5e7, clip=True)
    union = shapely.union_all(REGION.geometry.values)
    assert all(shapely.covers(shapely.buffer(union, 1e-6), g) for g in clipped.geometry.values)
    whole = ls.conn.make_grid(REGION, 2.5e7, grid_boundary=True)
    assert 0 < len(whole) < len(ls.conn.make_grid(REGION, 2.5e7))
    with pytest.raises(ValueError, match="clip or grid_boundary"):
        ls.conn.make_grid(REGION, 2.5e7, clip=True, grid_boundary=True)
    with pytest.raises(ValueError, match="grid or a cell_area"):
        ls.conn.connectivity_grid(PAS, REGION, distance=3000)
    with pytest.raises(ValueError, match="two or more"):
        ls.conn.eca_change_grid([PAS], REGION, distance=3000, cell_area=2.5e7)
