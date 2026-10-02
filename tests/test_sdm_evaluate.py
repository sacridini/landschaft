# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.evaluate / boyce / omission_rate against predicts, ecospat (corrected) and ENMeval."""

from __future__ import annotations

import pytest

import sdm_cases
import sdm_reference


@pytest.mark.parametrize("case", sdm_reference.case_ids("evaluate"))
def test_matches_r(case):
    ref = sdm_reference.result("evaluate", case)
    ours = sdm_cases.run_python(case)
    assert sorted(ours) == sorted(ref)
    for key in ref:
        # the p-value goes through R's pt() (TOMS 708); scipy's t distribution agrees to ~1e-12
        tol = dict(rtol=1e-9) if key == "pcor" else {}
        sdm_reference.assert_same(ours[key], ref[key], key, **tol)
