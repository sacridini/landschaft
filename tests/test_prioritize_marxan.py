# SPDX-License-Identifier: GPL-3.0-or-later
"""The Marxan port against Marxan 4.0.6 (tools/make_reference_marxan.py): the same
runs, unit by unit, the same scores (bit for bit where the reference prints 17
digits) and the same feature penalties."""

from __future__ import annotations

import json
from pathlib import Path

import marxan_cases
import numpy as np
import pandas as pd
import pytest

import landschaft as ls

REF = Path(__file__).parent / "data" / "reference" / "optimize"


def _ref(case: str) -> dict:
    if not (REF / "marxan.npz").exists():
        pytest.skip("no Marxan references")
    z = np.load(REF / "marxan.npz")
    return {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(case + "/")}


def _meta() -> dict:
    return json.loads((REF / "marxan.json").read_text())


def _run(case: str, **kw):
    c = marxan_cases.CASES[case]
    pu, spec, puvspr, bound = marxan_cases.tables(c.status, c.spec)
    return ls.prioritize.marxan(pu, spec, puvspr, bound, **{**c.options, **kw})


@pytest.mark.parametrize("case", list(marxan_cases.CASES))
def test_runs_are_marxan(case):
    ref = _ref(case)
    res = _run(case, n_threads=3)
    sol = res.solutions.filter(like="solution_").to_numpy().T
    np.testing.assert_array_equal(sol, ref["solutions"], err_msg="solutions")
    full = _meta()[case]["exe"] == "fixed"  # 17 digits; the released executable prints 6
    for col in ("Score", "Cost", "Penalty", "Connectivity", "Shortfall", "MPM"):
        if full:
            np.testing.assert_array_equal(res.summary[col].to_numpy(), ref[col], err_msg=col)
        else:
            np.testing.assert_allclose(res.summary[col].to_numpy(), ref[col], rtol=5e-6, atol=1e-9, err_msg=col)
    np.testing.assert_array_equal(res.summary["Planning_Units"].to_numpy(), ref["Planning_Units"])
    np.testing.assert_array_equal(res.summary["Missing_Values"].to_numpy(), ref["Missing_Values"])
    if full:
        np.testing.assert_array_equal(res.penalties["penalty"].to_numpy(), ref["penalty"])
    else:
        np.testing.assert_allclose(res.penalties["penalty"].to_numpy(), ref["penalty"], rtol=5e-6)


def test_same_runs_on_any_threads():
    a = _run("connolly", n_threads=1)
    b = _run("connolly", n_threads=4)
    pd.testing.assert_frame_equal(a.solutions, b.solutions)
    pd.testing.assert_frame_equal(a.summary, b.summary)
    assert a.best_run == b.best_run
    best = a.summary["Final_Score"].to_numpy()
    assert a.best_run == int(np.argmin(best)) + 1
    assert (a.best["solution"].to_numpy() == a.solutions[f"solution_{a.best_run}"].to_numpy()).all()


def test_marxan_inputs():
    pu, spec, puvspr, bound = marxan_cases.tables()
    with pytest.raises(ValueError, match="twice"):
        ls.prioritize.marxan(pu, spec, pd.concat([puvspr, puvspr.head(1)]), bound)
    with pytest.raises(ValueError, match="not in pu"):
        ls.prioritize.marxan(pu, spec, puvspr.assign(pu=puvspr["pu"] + 10_000), bound)
    with pytest.raises(ValueError, match="run modes"):
        ls.prioritize.marxan(pu, spec, puvspr, bound, run_mode=3)
    r = ls.prioritize.marxan(pu, spec, puvspr, bound, num_reps=2, num_iters=20_000, num_temp=100, seed=3)
    assert len(r.summary) == 2 and set(r.frequency["frequency"]) <= {0.0, 0.5, 1.0}
