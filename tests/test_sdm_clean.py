# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.clean_occurrences against CoordinateCleaner, test by test."""

from __future__ import annotations

import pytest

import sdm_cases
import sdm_reference


@pytest.mark.parametrize("case", sdm_reference.case_ids("clean"))
def test_flags_match_coordinatecleaner(case):
    ref = sdm_reference.result("clean", case)
    ours = sdm_cases.run_python(case)
    assert sorted(ours) == sorted(ref)
    for key in ref:
        sdm_reference.assert_same(ours[key], ref[key], key)
