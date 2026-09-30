# SPDX-License-Identifier: GPL-3.0-or-later
"""The stored Zonation 5 references are complete and consistent with the cases."""

from __future__ import annotations

import numpy as np
import pytest

import prioritize_cases
from prioritize_reference import case_ids, load, parse_table, table_text


@pytest.mark.parametrize("name", case_ids("core"))
def test_reference_case(name):
    arrays, meta = load("core")
    case = prioritize_cases.CASES[name]
    rank = arrays[f"{name}/rank"]
    shape = prioritize_cases.RASTERS[case.layers[0]].values.shape
    assert rank.shape == shape
    ok = np.isfinite(rank)
    n = int(ok.sum())
    assert n == int(meta[f"{name}/stats"]["Number of grid cells in map with data"]) or "--area=2" in case.flags
    # ranks are the positions in the removal order divided by the number of cells
    pos = np.sort(rank[ok].astype(np.float64))
    assert np.array_equal(pos, (np.arange(n) * (1.0 / n)).astype(np.float32).astype(np.float64))
    if case.load:
        names, rows = parse_table(table_text(arrays, f"{name}/load/feature_curves"))
        assert names[0] == "rank" and len(names) == len(case.layers) + 1
        assert float(rows[0][0]) == 0.0 and float(rows[-1][0]) == 1.0
