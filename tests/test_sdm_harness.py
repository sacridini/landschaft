# SPDX-License-Identifier: GPL-3.0-or-later
"""The R reference harness of ls.sdm hands the inputs to R unchanged."""

from __future__ import annotations

import pytest

import sdm_cases
import sdm_reference


@pytest.mark.parametrize("case", sdm_reference.case_ids("smoke"))
def test_inputs_reach_r_unchanged(case):
    ref = sdm_reference.result("smoke", case)
    ours = sdm_cases.run_python(case)
    assert set(ours) == set(ref)
    for key, value in ours.items():
        sdm_reference.assert_same(value, ref[key], key)
