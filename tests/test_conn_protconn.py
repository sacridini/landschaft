# SPDX-License-Identifier: GPL-3.0-or-later
"""ProtConn (Makurhini's MK_ProtConn) against its inner steps and outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

REF = Path(__file__).parent / "data" / "reference" / "conn"
PAS = gpd.read_file(REF / "protconn_areas.gpkg")
REGIONS = gpd.read_file(REF / "protconn_regions.gpkg")
MK = pd.read_csv(REF / "makurhini_protconn.csv.gz", float_precision="round_trip", keep_default_na=False)
MK["extra"] = pd.to_numeric(MK["extra"], errors="coerce")
JOBS = [  # tools/make_reference_conn.PROTCONN_JOBS
    ("plain", 1, [1000.0, 5000.0], None, "nodes", False),
    ("trans_nodes", 1, [1000.0, 5000.0], 3000.0, "nodes", False),
    ("trans_region", 1, [2000.0], 3000.0, "region", False),
    ("bound", 1, [2000.0], 3000.0, "nodes", True),
    ("second", 2, [1500.0], None, "nodes", True),
]
RAW_NAMES = {"Protected.surface": "Protected_surface"}


def _value(v: str) -> float:
    return np.nan if v == "NA" else float(v)


def _mk_nodes(case: str):
    sub = MK[MK["case"] == case]
    nd = sub[sub["level"] == "node_attribute"]
    nodes = pd.DataFrame({"type": nd["name"].to_numpy(), "attribute": nd["value"].astype(float).to_numpy()})
    n = len(nodes)
    dd = sub[sub["level"] == "distance"]
    ij = np.array([list(map(int, s.split())) for s in dd["name"]]) - 1
    m = np.zeros((n, n))
    m[ij[:, 0], ij[:, 1]] = m[ij[:, 1], ij[:, 0]] = dd["value"].astype(float).to_numpy()
    la = float(sub[sub["level"] == "LA"]["value"].iloc[0])
    pieces = sub[sub["level"] == "piece"]["value"].astype(float).to_numpy()
    return nodes, m, la, pieces, sub[sub["level"] == "raw"]


@pytest.mark.parametrize("case,region,dists,tb,ttype,bound", JOBS, ids=[j[0] for j in JOBS])
def test_formulas_on_makurhini_nodes(case, region, dists, tb, ttype, bound):
    nodes, m, la, pieces, raw = _mk_nodes(case)
    for d in dists:
        ours = ls.conn.protconn_indicators(nodes, m, distance=d, la=la, pieces=pieces)
        for _, row in raw[raw["extra"] == d].iterrows():
            v = _value(row["value"])
            key = RAW_NAMES.get(row["name"], row["name"])
            if np.isnan(v):
                assert np.isnan(ours[key]), key
            else:
                assert ours[key] == v, key  # get_protconn_grid's bits


@pytest.mark.parametrize("case,region,dists,tb,ttype,bound", JOBS, ids=[j[0] for j in JOBS])
def test_end_to_end(case, region, dists, tb, ttype, bound):
    area = REGIONS[REGIONS["id"] == region]
    nodes, _, la, pieces, raw = _mk_nodes(case)
    prep = ls.conn.protconn_nodes(PAS, area, transboundary=tb, transboundary_type=ttype, bound=bound,
                                  area_unit="ha")
    assert sorted(prep.nodes["type"]) == sorted(nodes["type"])
    np.testing.assert_allclose(np.sort(prep.nodes["attribute"]), np.sort(nodes["attribute"]), rtol=1e-13)
    np.testing.assert_allclose(np.sort(prep.pieces), np.sort(pieces), rtol=1e-13)
    assert prep.la == pytest.approx(la, rel=1e-15)
    table, delta = ls.conn.protconn(PAS, area, distance=dists, transboundary=tb, transboundary_type=ttype,
                                    bound=bound, area_unit="ha", delta=True)
    sub = MK[MK["case"] == case]
    for d in dists:
        ours = table[table["distance"] == d].iloc[0]
        for _, row in raw[raw["extra"] == d].iterrows():
            v = _value(row["value"])
            if np.isfinite(v):
                # ProtUnconn[Design] subtracts two close ECAs: 1e-10.
                assert ours[RAW_NAMES.get(row["name"], row["name"])] == pytest.approx(v, rel=1e-9, abs=1e-12)
        for _, row in sub[(sub["level"] == "public") & (sub["extra"] == d)].iterrows():
            v = _value(row["value"])
            if np.isfinite(v):
                assert round(ours[row["name"]], 4) == v, row["name"]  # MK_ProtConn rounds to 4 decimals
        dp = sub[(sub["level"] == "dProt") & (sub["extra"] == d)]["value"].astype(float).to_numpy()
        dc = sub[(sub["level"] == "dProtConn") & (sub["extra"] == d)]["value"].astype(float)
        mine = delta[delta["distance"] == d]
        np.testing.assert_allclose(mine["dProt"], dp, rtol=1e-12, atol=1e-12)
        # Makurhini weighs the transboundary pairs 0 in the region's ECA and 1 without each
        # area; with no transboundary node both agree.
        if not (prep.nodes["type"] == "Transboundary").any():
            np.testing.assert_allclose(mine["dProtConn"], dc.to_numpy(), rtol=1e-11, atol=1e-12)


def test_regions_and_weights():
    rows = ls.conn.protconn_regions(PAS, REGIONS, id="id", distance=[2000.0], area_unit="ha")
    assert list(rows["region"]) == [1, 2]
    one = ls.conn.protconn(PAS, REGIONS[REGIONS["id"] == 2], distance=[2000.0], area_unit="ha")
    pd.testing.assert_series_equal(rows.iloc[1].drop("region"), one.iloc[0], check_names=False)
    # Weight 0: transboundary nodes are stepping stones only; the ECA is a graph's EC(PC).
    area = REGIONS[REGIONS["id"] == 1]
    prep = ls.conn.protconn_nodes(PAS, area, transboundary=3000.0, area_unit="ha")
    m = np.array([[shapely.distance(a, b) for b in prep.nodes.geometry] for a in prep.nodes.geometry])
    zero = ls.conn.protconn_indicators(prep.nodes, m, distance=2000.0, la=prep.la, transboundary_weight=0.0)
    i, j = np.triu_indices(len(m), 1)
    g = ls.HabitatGraph(prep.nodes["attribute"].to_numpy(), pd.DataFrame({"a": i + 1, "b": j + 1, "d": m[i, j]}))
    ec = g.overall("PC", distance=2000.0, pij_min=0.0).iloc[0]["EC(PC)"]
    assert zero["ECA"] == pytest.approx(ec, rel=1e-12)
    one_w = ls.conn.protconn_indicators(prep.nodes, m, distance=2000.0, la=prep.la)
    assert one_w["ECA"] > zero["ECA"]


def test_single_and_empty_regions():
    region = gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 1000, 1000)], crs=PAS.crs)
    pa = gpd.GeoDataFrame(geometry=[shapely.box(100, 100, 300, 400), shapely.box(5000, 5000, 5100, 5100)],
                          crs=PAS.crs)
    one = ls.conn.protconn(pa, region, distance=[500.0]).iloc[0]
    assert one["ProtConn"] == pytest.approx(100 * 200 * 300 / 1e6)
    assert one["PC"] == pytest.approx(200 * 300 / 1e12)  # MK's single-node branch: area / LA^2
    assert one["ProtConn_Prot"] == 100 and np.isnan(one["ProtConn_Trans"])
    none = ls.conn.protconn(pa.iloc[[1]], region, distance=[500.0]).iloc[0]
    assert none["ProtConn"] == 0 and none["Unprotected"] == 100
    with pytest.raises(ValueError, match="transboundary_type"):
        ls.conn.protconn(pa, region, distance=[500.0], transboundary=10.0, transboundary_type="x")
