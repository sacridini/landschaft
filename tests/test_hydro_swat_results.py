# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.hydro.swat results: SwatResult's tables are what SWAT+ prints (print.prt), as
DataFrames with SWAT+'s column names, and SwatProject.set_print chooses them."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
from conftest import assert_swat_close

S = ls.hydro.swat
REF = Path(__file__).parent / "data" / "reference" / "swat"


def _project(case: str) -> S.SwatProject:
    with zipfile.ZipFile(REF / f"{case}.zip") as z:
        return S.SwatProject({n: z.read(n) for n in z.namelist()})


@pytest.fixture(scope="module")
def hb():
    return S.run(_project("hubbard_brook")), np.load(REF / "hubbard_brook.npz")


@pytest.mark.parametrize("name", list(S.TABLES))
@pytest.mark.parametrize("step", S.STEPS)
def test_tables_are_the_printed_records(hb, name, step):
    result, ref = hb
    tag = S.TABLES[name][0] + S.STEPS.index(step)
    df = result.table(name, step)
    assert list(df.columns) == list(S.TABLES[name][2])
    rkeys = ref[f"t{tag}_keys"]
    assert len(df) == len(rkeys)
    dates = df.index.get_level_values("date")
    assert dates[0].year == rkeys[0, 0] and dates[0].dayofyear == rkeys[0, 1]
    if S.TABLES[name][1] != "basin":
        np.testing.assert_array_equal(df.index.get_level_values(S.TABLES[name][1]), rkeys[:, 2])
    assert_swat_close(df.to_numpy(), ref[f"t{tag}_vals"][:, : df.shape[1]], err_msg=f"{name} {step}")


def test_units_and_repr(hb):
    result, _ = hb
    assert result.units["hru_wb"]["precip"] == "mm"
    assert result.units["channel_sd"]["flo_out"] == "m^3/s"
    assert "hru_wb[day/month/year/aa]" in repr(result)


def test_yearly_water_balance_closes(hb):
    # the yearly HRU balance: precipitation = ET + water yield + percolation + change in
    # soil water and snow, within the lag storages SWAT+ keeps (small for 2002)
    result, _ = hb
    y = result.table("hru_wb", "year").xs(pd.Timestamp("2002-12-31"), level="date")
    rhs = y["et"] + y["surq_gen"] + y["latq"] + y["perc"] + (y["sw_final"] - y["sw_init"]) + (
        y["sno_final"] - y["sno_init"]) - y["surq_runon"]
    assert np.allclose(y["precip"], rhs, rtol=0.05)


def test_set_print_and_run_choose_tables():
    project = _project("hubbard_brook_musk")
    res = S.run(project, tables={"hru_wb": "year", "channel_sd": ["month"], "hru_nb": [], "hru_ls": [],
                                 "hru_pw": [], "aquifer": [], "basin_wb": [], "basin_nb": [], "basin_ls": [],
                                 "basin_pw": []}, nyskip=1)
    assert set(res.tables) == {"hru_wb", "channel_sd"}
    assert set(res.tables["hru_wb"]) == {"year"}
    assert res.table("hru_wb", "year").index.get_level_values("date").year.unique().tolist() == [2001]
    with pytest.raises(KeyError, match="did not print"):
        res.table("hru_wb", "day")
    # the project itself is left as it was
    assert "hru_wb" in S.run(project).tables
    p = project.copy()
    p.set_print({"aquifer": []})
    assert p.text("print.prt") != project.text("print.prt")
    p.set_print(["aquifer"])
    assert p.text("print.prt") == project.text("print.prt")
    with pytest.raises(ValueError, match="steps"):
        p.set_print({"aquifer": "week"})
    with pytest.raises(ValueError, match="no row"):
        p.set_print({"nonsense": "day"})


def test_results_save(tmp_path, hb):
    result, _ = hb
    folder = result.save(tmp_path / "swat")
    names = {p.name for p in folder.rglob("*.csv")}
    assert any("hru_wb" in n for n in names)
