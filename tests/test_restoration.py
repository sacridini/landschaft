# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Restoration Status (RSS) and Restoration Planner / Cost Distance."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import landschaft as ls
from landschaft.restoration import (
    RestorationCorridorResult,
    RestorationStatusResult,
    cost_distance,
    least_cost_path,
    restoration_corridor,
    restoration_status,
)


@pytest.fixture
def input_path() -> Path:
    return Path(__file__).resolve().parent.parent / "src" / "landschaft" / "data" / "input.tif"


class TestRestorationStatus:
    def test_canonical_input_values(self, input_path: Path):
        """Verify Restoration Status produces bit-for-bit exact GTB metrics on input.tif."""
        res = restoration_status(input_path)

        assert isinstance(res, RestorationStatusResult)
        assert res.data_area == 999730
        assert res.nr_comp == 2850
        assert res.eca_max == 428490
        assert res.larg_obj == 214811
        assert res.cnoa == 311712

        assert res.aps == pytest.approx(150.35, abs=0.01)
        assert res.eca == pytest.approx(221292.76, abs=0.01)
        assert res.doc == pytest.approx(51.64, abs=0.01)
        assert res.rest_pot == pytest.approx(48.36, abs=0.01)
        assert res.rac == pytest.approx(42.86, abs=0.01)

        # DataFrame and dict export
        df = res.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.iloc[0]["AREA"] == 999730
        assert df.iloc[0]["NR_OBJ"] == 2850
        assert df.iloc[0]["CNOA"] == 311712

        d = res.to_dict()
        assert d["data_area"] == 999730
        assert d["larg_obj"] == 214811

        s = res.summary()
        assert "Restoration Status (RSS) Summary" in s
        assert "999,730" in s

    def test_single_component(self):
        """Single component has DOC=100%, REST_POT=0%, and CNOA=0."""
        arr = np.ones((50, 50), dtype=np.uint8)
        arr[10:40, 10:40] = 2  # 30x30 = 900 foreground pixels

        res = restoration_status(arr)
        assert res.nr_comp == 1
        assert res.eca_max == 900
        assert res.larg_obj == 900
        assert res.eca == pytest.approx(900.0)
        assert res.doc == pytest.approx(100.0)
        assert res.rest_pot == pytest.approx(0.0)
        assert res.cnoa == 0

    def test_no_foreground(self):
        """Image with no foreground has nr_comp=0 and REST_POT=100%."""
        arr = np.ones((20, 20), dtype=np.uint8)
        arr[0, 0] = 0  # 1 missing pixel

        res = restoration_status(arr)
        assert res.data_area == 399
        assert res.nr_comp == 0
        assert res.eca_max == 0
        assert res.doc == 0.0
        assert res.rest_pot == 100.0
        assert res.cnoa == 0

    def test_two_equal_components(self):
        """Two 10-pixel components: manual formula verification."""
        arr = np.ones((20, 20), dtype=np.uint8)
        arr[2, 2:12] = 2  # component 1: 10 pixels
        arr[8, 2:12] = 2  # component 2: 10 pixels

        res = restoration_status(arr)
        assert res.nr_comp == 2
        assert res.eca_max == 20
        assert res.larg_obj == 10
        # ECA = sqrt(10^2 + 10^2) = sqrt(200) = 14.1421356...
        expected_eca = np.sqrt(200.0)
        assert res.eca == pytest.approx(expected_eca)
        assert res.doc == pytest.approx((expected_eca / 20.0) * 100.0)
        # CNOA = floor((2 * 20 * 200) / (400 - 200) + 1) = floor(40 + 1) = 41
        assert res.cnoa == 41


class TestCostDistanceAndLCP:
    def test_cost_distance_free_field(self):
        """Cost distance in uniform resistance field."""
        res = np.full((30, 30), 10, dtype=np.uint8)
        src = np.zeros((30, 30), dtype=np.uint8)
        src[15, 15] = 1

        cd = cost_distance(res, src)
        assert cd[15, 15] == pytest.approx(0.0)
        # Cardinal step distance: 10.0
        assert cd[15, 16] == pytest.approx(10.0)
        # Diagonal step distance: sqrt(2) * 10.0
        assert cd[16, 16] == pytest.approx(np.sqrt(2) * 10.0, rel=1e-4)

    def test_least_cost_path_around_barrier(self):
        """LCP navigates around an impassable resistance barrier."""
        res = np.full((50, 50), 5, dtype=np.uint8)
        res[20:30, 10:40] = 0  # impassable barrier

        ma = np.zeros((50, 50), dtype=np.uint8)
        ma[5, 25] = 1

        mb = np.zeros((50, 50), dtype=np.uint8)
        mb[45, 25] = 1

        lcp = least_cost_path(res, ma, mb)
        assert lcp.path_length > 0
        assert lcp.rest_pixels > 0
        assert lcp.expense > 0.0

        # Verify path does not cross barrier (barrier cells have resistance 0)
        mask = np.asarray(lcp.lcp_mask) > 0
        assert not np.any(res[mask] == 0)

    def test_restoration_corridor_evaluation(self):
        """Evaluating restoration corridor between two patches shows positive ECA gain."""
        hab = np.ones((50, 50), dtype=np.uint8)
        hab[10:15, 10:15] = 2  # Patch A: 25 pixels
        hab[35:40, 35:40] = 2  # Patch B: 25 pixels

        res_surf = np.full((50, 50), 2, dtype=np.uint8)

        ma = np.zeros((50, 50), dtype=np.uint8)
        ma[12, 12] = 1

        mb = np.zeros((50, 50), dtype=np.uint8)
        mb[37, 37] = 1

        corridor_res = restoration_corridor(hab, res_surf, ma, mb)
        assert isinstance(corridor_res, RestorationCorridorResult)
        assert corridor_res.delta_eca > 0.0
        assert corridor_res.delta_doc > 0.0
        assert corridor_res.efficiency > 0.0
        df = corridor_res.to_dataframe()
        assert df.iloc[0]["DELTA_ECA"] > 0
