# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.autocorrelation_range against blockCV::cv_spatial_autocor (automap / gstat)."""

from __future__ import annotations

import pytest

import sdm_cases
import sdm_reference


@pytest.mark.parametrize("case", sdm_reference.case_ids("autocorrelation"))
def test_matches_blockcv(case):
    ref = sdm_reference.result("autocorrelation", case)
    ours = sdm_cases.run_python(case)
    for key, value in ref.items():
        if key.endswith("/np"):
            sdm_reference.assert_same(ours[key], value, key)  # the same sample and distance classes
        elif key.endswith(("/dist", "/gamma")):
            sdm_reference.assert_same(ours[key], value, key, rtol=1e-12)
        else:
            # gstat's iterations stop at a relative change of 1e-5 in the fit: parameters to ~1e-6 (the derivative of the Matern model is numerical)
            sdm_reference.assert_same(ours[key], value, key, rtol=1e-6)
