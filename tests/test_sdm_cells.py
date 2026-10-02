# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.one_per_cell against terra::cellFromXY + duplicated()."""

from __future__ import annotations

import pytest

import sdm_cases
import sdm_reference


@pytest.mark.parametrize("case", sdm_reference.case_ids("cells"))
def test_one_per_cell_matches_terra(case):
    ref = sdm_reference.result("cells", case)
    ours = sdm_cases.run_python(case)
    for key in ("cell", "keep"):
        sdm_reference.assert_same(ours[key], ref[key], key)
