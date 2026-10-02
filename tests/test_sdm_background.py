# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.background against dismo::randomPoints (the same cells for the same seed)."""

from __future__ import annotations

import pytest

import sdm_cases
import sdm_reference


@pytest.mark.parametrize("case", sdm_reference.case_ids("background"))
def test_matches_dismo(case):
    ref = sdm_reference.result("background", case)
    ours = sdm_cases.run_python(case)
    sdm_reference.assert_same(ours["cell"], ref["cell"], "cell")
