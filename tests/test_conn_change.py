# SPDX-License-Identifier: GPL-3.0-or-later
"""ECA over time (Makurhini's MK_dECA)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls

gpd = pytest.importorskip("geopandas")

REF = Path(__file__).parent / "data" / "reference" / "conn"
COLUMNS = ["ECA", "ECA_LA", "ECA_area", "dA", "dECA", "rECA"]


def _graphs(kind: str) -> list:
    steps = pd.read_csv(REF / "deca_steps.csv")
    keep = [steps[(steps["kind"] == kind) & (steps["time"] == t)]["id"] for t in (1, 2, 3)]
    if kind == "vector":
        poly = gpd.read_file(REF / "polygons.gpkg")
        return [ls.HabitatGraph.from_vector(poly[poly["id"].isin(ids)].reset_index(drop=True), id="id",
                                            area_unit="km2") for ids in keep]
    r = ls.load_raster(REF / "habitat_nodes_raster.tif").squeeze()
    return [ls.HabitatGraph.from_raster(r.where(r.isin(ids)).rio.write_nodata(np.nan), area_unit="ha")
            for ids in keep]


@pytest.mark.parametrize("kind,metric,la,distance", [
    ("vector", "PC", 400.0, [2000, 6000]),
    ("vector", "IIC", 400.0, [2000, 6000]),
    ("raster", "PC", 3.5e6, [10000]),
])
def test_makurhini_deca(kind, metric, la, distance):
    ref = pd.read_csv(REF / "makurhini_deca.csv.gz", float_precision="round_trip")
    ref = ref[(ref["kind"] == kind) & (ref["metric"] == metric)].reset_index(drop=True)
    ours = ls.conn.eca_change(_graphs(kind), metric, distance=distance, landscape_area=la,
                              probability=0.5 if metric == "PC" else None)
    np.testing.assert_array_equal(ours["dA"], ref["dA"])  # R's sums of the areas, same bits
    for col in COLUMNS:
        np.testing.assert_allclose(ours[col], ref[col], rtol=1e-13, err_msg=col)
    # Makurhini rounds its habitat area to 3 decimals (the columns it rounds depend on the number of times).
    np.testing.assert_allclose(ours["area"], ref["area"], atol=5e-4, rtol=0)
    assert list(ours["comparison"]) == list(ref["comparison"])
    assert list(ours["change"]) == list(ref["change"])


def test_eca_change_by_hand():
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    g1 = ls.HabitatGraph([4.0, 4.0], D)
    g2 = ls.HabitatGraph([4.0], np.zeros((1, 1)))  # one node left: ECA = its attribute
    out = ls.conn.eca_change([g1, g2], "IIC", distance=5.0, landscape_area=20.0, names=["before", "after"])
    eca1 = np.sqrt(16 + 16 + 2 * 16 / 2)
    assert out["ECA"].tolist() == pytest.approx([eca1, 4.0])
    assert out.loc[0, "dA"] == pytest.approx((8 - 20) / 20 * 100)
    assert out.loc[1, "dA"] == pytest.approx(-50.0)
    assert out.loc[1, "dECA"] == pytest.approx((4 - eca1) / eca1 * 100)
    assert out.loc[1, "change"] == "+ Habitat loss"  # dA (-50 %) < dECA (-42 %) < 0
    assert list(out["time"]) == ["before", "after"]
    capped = ls.conn.eca_change([ls.HabitatGraph([30.0], np.zeros((1, 1)))], "PC", distance=5.0,
                                landscape_area=20.0)
    assert capped.loc[0, "ECA"] == 20.0  # MK_dPCIIC caps EC at LA
    with pytest.raises(ValueError, match="PC' or 'IIC"):
        ls.conn.eca_change([g1], "H", distance=5.0, landscape_area=20.0)
    with pytest.raises(ValueError, match="one name per graph"):
        ls.conn.eca_change([g1], distance=5.0, landscape_area=20.0, names=["a", "b"])
