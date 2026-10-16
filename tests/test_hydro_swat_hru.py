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
from conftest import assert_ucrt_close

REF = Path(__file__).parent / "data" / "reference" / "swat"
# Ames: 12 HRUs with scheduled operations (plant, harvest and kill, tillage, fertilizer,
# irrigation) and a decision table for automatic irrigation; no routing objects
CASES = ["hubbard_brook", "hubbard_brook_musk", "ames"]

# Fields written only by the reference's hru_output (the printed outputs, not ported
# yet): hwb_d sw_init, sno_init and sno_final; hpw_d bm_max.
OUTPUT_ONLY = {10: [13, 17, 18], 13: [22]}


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
    vals = vals[:, : rvals.shape[1]].copy()
    rvals = rvals.copy()
    for c in OUTPUT_ONLY.get(tag, []):
        vals[:, c] = rvals[:, c] = 0.0
    assert_ucrt_close(vals, rvals, err_msg=f"tag {tag}")


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
