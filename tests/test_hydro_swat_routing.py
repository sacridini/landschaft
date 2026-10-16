# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.hydro.swat routing (aquifers, routing units, SWAT-DEG channels, the command loop)
against SWAT+ 62.0.1, with the reference's HRU outputs put in place of hru_control:
every object's inflow and outflow hydrographs, the aquifers and the channels, day by
day, as the reference dumps them (tools/make_reference_swat.py)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

import landschaft as ls
from landschaft import _core
from conftest import assert_swat_close

REF = Path(__file__).parent / "data" / "reference" / "swat"
CASES = ["hubbard_brook", "hubbard_brook_musk"]


def _files(case: str) -> dict:
    with zipfile.ZipFile(REF / f"{case}.zip") as z:
        return {n: z.read(n) for n in z.namelist()}


@pytest.fixture(scope="module", params=CASES)
def routed(request):
    case = request.param
    ref = np.load(REF / f"{case}.npz")
    inject = {tag: (ref[f"t{tag}_keys"], ref[f"t{tag}_vals"]) for tag in (6, 10, 40)}
    ours = _core.swat_run(_files(case), True, inject)
    return case, ours, ref


def _compare(ours, ref, tag: int) -> None:
    keys, vals = ours["dump"][tag]
    rkeys, rvals = ref[f"t{tag}_keys"], ref[f"t{tag}_vals"]
    np.testing.assert_array_equal(keys, rkeys, err_msg=f"tag {tag}: (year, day, object)")
    w = rvals.shape[1]
    assert_swat_close(vals[:, :w], rvals[:, :w], err_msg=f"tag {tag}")


@pytest.mark.parametrize("tag", [41, 40, 20, 30])
def test_routing_matches_swatplus(routed, tag):
    # 41: inflows of each object (hin, hin_sur, hin_lat, hin_til, hin_aqu); 40: the outflow
    # hydrographs after each object; 20: aquifers; 30: channels
    _, ours, ref = routed
    _compare(ours, ref, tag)


def test_channel_carries_water(routed):
    _, ours, ref = routed
    keys, vals = ours["dump"][30]
    assert (vals[:, 2] > 0).mean() > 0.5  # chsd_d%flo (m3/s) on most days
