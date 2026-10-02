# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.mess / mop / exdet against predicts::mess, mop::mop and dsmextra::ExDet."""

from __future__ import annotations

import pytest

import sdm_cases
import sdm_reference

# R inverts covariance matrices with LAPACK (Mahalanobis distances): those agree to a tolerance
TOLERANT = {"mop_mahalanobis": 1e-9, "exdet": 1e-9}


@pytest.mark.parametrize("case", sdm_reference.case_ids("extrapolation"))
def test_matches_r(case):
    ref = sdm_reference.result("extrapolation", case)
    ours = sdm_cases.run_python(case)
    assert sorted(ours) == sorted(ref)
    for key in ref:
        tol = dict(rtol=TOLERANT[case]) if case in TOLERANT and key in ("mop", "exdet") else {}
        sdm_reference.assert_same(ours[key], ref[key], key, **tol)
