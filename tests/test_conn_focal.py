# SPDX-License-Identifier: GPL-3.0-or-later
"""Focal connectivity (Makurhini's MK_Focal_nodes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

REF = Path(__file__).parent / "data" / "reference" / "conn"
DATA = {"polygons": REF / "polygons.gpkg", "pas": REF / "protconn_areas.gpkg"}
COLUMNS = {"Attrib_val": "attribute", "EC.IIC.": "EC(IIC)", "EC.PC.": "EC(PC)"}


def _cases():
    ref = pd.read_csv(REF / "makurhini_focal.csv.gz", keep_default_na=False, dtype=str)
    return [grp for _, grp in ref.groupby("case", sort=False)]


@pytest.mark.parametrize("grp", _cases(), ids=lambda g: f"case{g.iloc[0]['case']}")
def test_focal_nodes_match_makurhini(grp):
    row = grp.iloc[0]
    d = grp["distance"].astype(float)
    distances = list(dict.fromkeys(d))
    buffers = [float(grp.loc[d == x, "search_buffer"].iloc[0]) for x in distances]
    probability = None if row["probability"] == "" else float(row["probability"])
    ours = ls.conn.focal_nodes(gpd.read_file(DATA[row["data"]]), id="id", metric=row["metric"], distance=distances,
                               search_buffer=buffers, probability=probability, pij_min=float(row["pij_min"]))
    metric = row["metric"]
    for (dist, name), sub in grp.groupby([d, "name"], sort=False):
        col = COLUMNS.get(name, name)
        o = ours[ours["distance"] == dist].set_index("id").loc[sub["id"].astype(int)]
        a = o[col].to_numpy(float)
        b = pd.to_numeric(sub["value"], errors="coerce").to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), name
        ok = ~np.isnan(b)
        if col == "IComp":  # IIC (or PC) times the rounded dIIC
            a = o[metric].to_numpy(float) * np.round(o[f"d{metric}"].to_numpy(float), 7)
            np.testing.assert_allclose(a[ok], b[ok], rtol=1e-12, err_msg=name)
        elif col.startswith("d") and not col.endswith("connector"):  # MK_dPCIIC rounds them to 7 digits
            np.testing.assert_allclose(np.round(a[ok], 7), b[ok], rtol=0, atol=1e-12, err_msg=name)
        else:
            np.testing.assert_allclose(a[ok], b[ok], rtol=1e-12, atol=1e-12, err_msg=name)


def test_focal_node_is_its_neighbourhood_graph():
    poly = gpd.read_file(DATA["polygons"])
    ours = ls.conn.focal_nodes(poly, id="id", metric="PC", distance=3000, search_buffer=5000, area_unit="km2")
    geoms = poly.geometry.values
    for i in (0, 7, 30):
        near = np.flatnonzero(shapely.intersects(shapely.buffer(geoms[i], 5000, quad_segs=30), geoms))
        assert ours["n_nodes"].iloc[i] == near.size
        pts = shapely.get_coordinates(shapely.point_on_surface(geoms[near]))
        D = np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
        x0, y0 = shapely.bounds(geoms[near])[:, :2].min(axis=0)
        x1, y1 = shapely.bounds(geoms[near])[:, 2:].max(axis=0)
        area = shapely.area(geoms[near]) / 1e6
        g = ls.HabitatGraph(area, D, landscape_area=(x1 - x0) * (y1 - y0) / 1e6)
        pc = g.overall(["PC"], distance=3000)["PC"].iloc[0]
        dpc = g.node_importance(["PC"], distance=3000)["dPC"].iloc[int(np.flatnonzero(near == i)[0])]
        np.testing.assert_allclose(ours["PC"].iloc[i], pc, rtol=1e-9)
        np.testing.assert_allclose(ours["dPC"].iloc[i], dpc, rtol=1e-9)
        np.testing.assert_allclose(ours["IComp"].iloc[i], pc * dpc, rtol=1e-9)


def test_focal_options_and_errors():
    poly = gpd.read_file(DATA["polygons"])
    alone = ls.conn.focal_nodes(poly, id="id", distance=1000, search_buffer=1.0)
    few = alone["n_nodes"] <= 2
    assert few.any() and alone.loc[few, ["IIC", "dIIC", "IComp"]].isna().all().all()
    weights = np.linspace(1, 2, len(poly))
    by_column = ls.conn.focal_nodes(poly.assign(w=weights), id="id", attribute="w", distance=3000, search_buffer=5000)
    by_values = ls.conn.focal_nodes(poly, id="id", attribute=weights, distance=3000, search_buffer=5000)
    pd.testing.assert_frame_equal(by_column, by_values)
    np.testing.assert_array_equal(by_column["attribute"], weights)
    # a threshold cuts the far pairs: the same as linking up to it
    far = ls.conn.focal_nodes(poly, id="id", distance=3000, search_buffer=8000, threshold=2999.0)
    near = ls.conn.focal_nodes(poly, id="id", distance=3000, search_buffer=8000)
    np.testing.assert_allclose(far["dIIC"], near["dIIC"], rtol=1e-12)
    with pytest.raises(ValueError, match="'IIC' or 'PC'"):
        ls.conn.focal_nodes(poly, distance=3000, search_buffer=5000, metric="H")
    with pytest.raises(ValueError, match="one value per node"):
        ls.conn.focal_nodes(poly, distance=3000, search_buffer=5000, attribute=[1.0, 2.0])
