# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls


@pytest.mark.parametrize("name", list(ls.EXAMPLE_DATA))
def test_example_files_exist(name):
    assert ls.example_data(name).is_file()


def test_input_example_matches_gtb_reference(reference_dir):
    data = ls.load_raster(ls.example_data())  # passes the MSPA checks
    with np.load(reference_dir / "gtb_input.npz") as ref:
        np.testing.assert_array_equal(data.values, ref["input"])
        np.testing.assert_array_equal(ls.mspa(data).values, ref["c8_e1_i1"])


def test_clc_example_reclassified_for_mspa():
    clc = ls.load_raster(ls.example_data("clc"))
    assert clc.shape == (2000, 2000) and clc.rio.crs.to_epsg() == 3035
    forest = ls.reclassify(clc, foreground=[23, 24, 25])
    ls.validate_mspa_input(forest)


def test_gscinput_example():
    gsc = ls.load_raster(ls.example_data("gscinput"))
    assert gsc.rio.nodata == 255 and int(gsc.where(gsc != 255).max()) <= 100


def test_unknown_example():
    with pytest.raises(ValueError, match="available"):
        ls.example_data("nope")
