# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.hydro.swat HRUs (hru_control: snow, ET, curve number runoff, MUSLE, percolation,
lateral flow, nutrients, plant growth, management) and the whole daily simulation against SWAT+
62.0.1: the HRU balances of each day, and the routing fed by our own HRUs, as the
reference dumps them (tools/make_reference_swat.py)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from landschaft import _core
from conftest import assert_swat_close

REF = Path(__file__).parent / "data" / "reference" / "swat"
# Ames: 12 HRUs with scheduled operations (plant, harvest and kill, tillage, fertilizer,
# irrigation) and a decision table for automatic irrigation; no routing objects
CASES = ["hubbard_brook", "hubbard_brook_musk", "ames"]



def _files(case: str) -> dict:
    with zipfile.ZipFile(REF / f"{case}.zip") as z:
        return {n: z.read(n) for n in z.namelist()}


@pytest.fixture(scope="module", params=CASES)
def simulated(request):
    case = request.param
    return case, _core.swat_run(_files(case), True), np.load(REF / f"{case}.npz")


@pytest.mark.parametrize("tag", [10, 11, 12, 13, 6, 40, 41, 20, 30])
def test_simulation_matches_swatplus(simulated, tag):
    # 10-13: hwb_d, hnb_d, hls_d, hpw_d of each HRU after hru_control; 6: delivery
    # ratios; 40/41: outflow and inflow hydrographs of each object; 20: aquifers;
    # 30: channels
    case, ours, ref = simulated
    if f"t{tag}_keys" not in ref.files:
        pytest.skip(f"{case} has no objects of tag {tag}")
    keys, vals = ours["dump"][tag]
    rkeys, rvals = ref[f"t{tag}_keys"], ref[f"t{tag}_vals"]
    np.testing.assert_array_equal(keys, rkeys, err_msg=f"tag {tag}: (year, day, object)")
    vals = vals[:, : rvals.shape[1]]
    assert_swat_close(vals, rvals, err_msg=f"tag {tag}")


# what SWAT+ prints (print.prt), tag = its Fortran unit: HRU water, nutrient, losses and
# plant tables (2000-2043), the basin's (2050-2083), SWAT-DEG channels (2500-2503) and
# aquifers (2520-2523); the last digit is the step (day, month, year, average annual)
PRINTS = [u + k for u in (2000, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2500, 2520) for k in range(4)]


@pytest.mark.parametrize("tag", PRINTS)
@pytest.mark.external_reference
def test_printed_outputs_match_swatplus(simulated, tag):
    case, ours, ref = simulated
    if f"t{tag}_keys" not in ref.files:
        pytest.skip(f"{case} does not print table {tag}")
    keys, vals = ours["prints"][tag]
    np.testing.assert_array_equal(keys, ref[f"t{tag}_keys"], err_msg=f"print {tag}: (year, day, object)")
    assert_swat_close(vals, ref[f"t{tag}_vals"], err_msg=f"print {tag}")


def test_hrus_make_runoff_and_grow(simulated):
    _, ours, _ = simulated
    _, wb = ours["dump"][10]
    _, pw = ours["dump"][13]
    assert wb[:, 3].sum() > 0  # surq_gen
    assert wb[:, 7].mean() > 0.5  # et (mm/day)
    assert pw[:, 1].max() > 0  # bioms


def test_ames_management_happens():
    files = _files("ames")
    dump = _core.swat_run(files, True)["dump"]
    _, wb = dump[10]
    _, pw = dump[13]
    assert (wb[:, 22] > 0).sum() > 100  # irr: the decision table irrigates
    assert pw[:, 4].max() > 0  # yield: corn is harvested
