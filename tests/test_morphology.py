# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for Parcellation and Contortion (Morphology)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import guidospy as gp
from guidospy.morphology import ParcellationResult, contortion, parcellation


@pytest.fixture
def input_path() -> Path:
    return Path(__file__).resolve().parent.parent / "src" / "guidospy" / "data" / "input.tif"


class TestParcellation:
    def test_canonical_input_conn8(self, input_path: Path):
        """Verify 8-connectivity parcellation on input.tif matches exact GTB numbers."""
        res = parcellation(input_path, connectivity=8)

        assert isinstance(res, ParcellationResult)
        g = res.global_stats
        assert g["total_parcels"] == 4020
        assert g["data_area"] == 999730
        assert g["aps"] == pytest.approx(248.69, abs=0.01)
        assert g["awaps"] == pytest.approx(310683.11, abs=0.05)
        assert g["division"] == pytest.approx(0.6892, abs=0.001)
        assert g["parc"] == pytest.approx(8.46, abs=0.01)

        df = res.stats
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

        # Class 1 (background)
        c1 = df[df["Class"] == 1].iloc[0]
        assert c1["Count"] == 1170
        assert c1["Area[pixels]"] == 571240
        assert c1["APS"] == pytest.approx(488.24, abs=0.01)
        assert c1["AWAPS"] == pytest.approx(458001.44, abs=0.05)
        assert c1["PARC[%]"] == pytest.approx(1.60, abs=0.01)

        # Class 2 (foreground)
        c2 = df[df["Class"] == 2].iloc[0]
        assert c2["Count"] == 2850
        assert c2["Area[pixels]"] == 428490
        assert c2["APS"] == pytest.approx(150.35, abs=0.01)
        assert c2["AWAPS"] == pytest.approx(114286.18, abs=0.05)
        assert c2["PARC[%]"] == pytest.approx(9.57, abs=0.01)

        s = res.summary()
        assert "Landscape Parcellation Summary" in s
        assert "4,020" in s

    def test_canonical_input_conn4(self, input_path: Path):
        """Verify 4-connectivity parcellation on input.tif matches exact GTB numbers."""
        res = parcellation(input_path, connectivity=4)

        g = res.global_stats
        assert g["total_parcels"] == 7173
        assert g["data_area"] == 999730
        assert g["aps"] == pytest.approx(139.37, abs=0.01)
        assert g["awaps"] == pytest.approx(254300.43, abs=0.05)
        assert g["division"] == pytest.approx(0.7456, abs=0.001)
        assert g["parc"] == pytest.approx(9.91, abs=0.01)

        df = res.stats
        c2 = df[df["Class"] == 2].iloc[0]
        assert c2["Count"] == 4854
        assert c2["Area[pixels]"] == 428490
        assert c2["APS"] == pytest.approx(88.28, abs=0.01)
        assert c2["PARC[%]"] == pytest.approx(13.49, abs=0.01)

    def test_parcellation_with_labels(self):
        """Verify return_labels=True produces labeled parcels array."""
        arr = np.array(
            [
                [0, 1, 1, 0],
                [1, 1, 0, 2],
                [0, 0, 2, 2],
            ],
            dtype=np.uint8,
        )
        res = parcellation(arr, connectivity=8, return_labels=True)
        labels = res.parcel_labels
        assert labels is not None
        assert labels.shape == arr.shape
        # Valid parcels should be labeled 1 and 2
        assert np.max(labels) == 2
        assert labels[0, 0] == 0  # nodata


class TestContortion:
    def test_contortion_square(self):
        """A simple square has contortion = 3 (contortion 2 + 1)."""
        arr = np.ones((30, 30), dtype=np.uint8)
        arr[10:20, 10:20] = 2  # 10x10 square with 8x8 core

        cont = contortion(arr, min_area=1)
        # Inside the square, contortion should be 3
        assert cont[15, 15] == 3
        # Background is 1
        assert cont[0, 0] == 1

    def test_contortion_neglected_no_core(self):
        """Object without 3x3 core is classified as neglected (value 2)."""
        arr = np.ones((20, 20), dtype=np.uint8)
        arr[5, 2:10] = 2  # 1-pixel thick line: no 3x3 core

        cont = contortion(arr, min_area=1)
        assert cont[5, 5] == 2  # neglected

    def test_contortion_canonical_input(self, input_path: Path):
        """Verify contortion on canonical input.tif."""
        cont = contortion(input_path, min_area=100)
        assert isinstance(cont, xr.DataArray)
        assert cont.shape == (1000, 1000)

        arr = np.asarray(cont)
        assert np.sum(arr == 0) == 270       # nodata
        assert np.sum(arr == 1) == 571240    # background
        assert np.sum(arr == 2) == 25318     # neglected

        # Display scaled format
        cont_disp = contortion(input_path, min_area=100, return_display=True)
        arr_disp = np.asarray(cont_disp)
        assert arr_disp.dtype == np.uint8
        assert np.max(arr_disp) <= 255
