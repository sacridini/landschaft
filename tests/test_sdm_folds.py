# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm folds against blockCV::cv_spatial and ENMeval's partitions."""

from __future__ import annotations

import pytest

import sdm_cases
import sdm_reference


@pytest.mark.parametrize("case", sdm_reference.case_ids("folds"))
def test_matches_r(case):
    ref = sdm_reference.result("folds", case)
    ours = sdm_cases.run_python(case)
    assert sorted(ours) == sorted(ref)
    for key in ref:
        sdm_reference.assert_same(ours[key], ref[key], key)
