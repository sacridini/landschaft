# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.hydro.swat against SWAT+ 62.0.1 (tools/make_reference_swat.py): the daily dump
of the reference, record by record, and the Fortran input it reads."""

from __future__ import annotations

import io
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import landschaft as ls
from landschaft import _core
from conftest import assert_ucrt_close

S = ls.hydro.swat
REF = Path(__file__).parent / "data" / "reference" / "swat"
# Ames needs the management operations and decision tables of the HRUs (next phase)
AMES = pytest.param("ames", marks=pytest.mark.skip(reason="SWAT+ HRU management not ported yet"))
CASES = ["hubbard_brook", AMES]


def _project(case: str) -> S.SwatProject:
    with zipfile.ZipFile(REF / f"{case}.zip") as z:
        return S.SwatProject({n: z.read(n) for n in z.namelist()})


@pytest.fixture(scope="module", params=CASES)
def case(request):
    name = request.param
    project = _project(name)
    ours = S._run._run_raw(project)
    ref = np.load(REF / f"{name}.npz")
    return name, project, ours, ref


def _compare(ours, ref, tag: int, width: int | None = None) -> None:
    keys, vals = ours["dump"][tag]
    rkeys, rvals = ref[f"t{tag}_keys"], ref[f"t{tag}_vals"]
    np.testing.assert_array_equal(keys, rkeys, err_msg=f"tag {tag}: (year, day, object)")
    w = width or rvals.shape[1]
    assert_ucrt_close(vals[:, :w], rvals[:, :w], err_msg=f"tag {tag}")


def test_weather_matches_swatplus(case):
    name, _, ours, ref = case
    _compare(ours, ref, 1)


def test_run_gives_weather_by_station(case):
    name, project, _, ref = case
    with pytest.warns() if name == "ames" else _nowarn():
        res = S.run(project)
    w = res.weather
    assert list(w.columns) == list(S.WEATHER_COLUMNS)
    rkeys = ref["t1_keys"]
    assert len(w) == len(rkeys)
    first = w.index[0][0]
    assert (first.year, first.dayofyear) == tuple(rkeys[0, :2])
    assert_ucrt_close(w["precip"].to_numpy(), ref["t1_vals"][:, 0])


class _nowarn:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("error")

    def __exit__(self, *a):
        self._w.__exit__(*a)


@pytest.mark.skip(reason="SWAT+ HRU management not ported yet")
def test_ames_warnings_are_the_diagnostics_of_swatplus():
    # Ames's weather-sta.cli names "ames.tem", its tmp.cli lists "ames.tmp": SWAT+ warns in
    # diagnostics.out and generates the temperatures; four land uses name schedules the
    # management.sch does not have
    raw = S._run._run_raw(_project("ames"))
    assert raw["warnings"] == [
        "agrr_lum AGRR_rot not found in management.sch",
        "soyb_lum SOYB_rot not found in management.sch",
        "oats_lum OATS_rot not found in management.sch",
        "wwht_lum WWHT_rot not found in management.sch",
        "ames.tem file not found (tgage)",
    ]


def test_project_round_trip(tmp_path):
    p = _project("hubbard_brook")
    folder = p.write(tmp_path / "TxtInOut")
    (folder / "hru_wb_aa.txt").write_text("an output of an earlier run\n")
    q = S.load_txtinout(folder)
    assert q.files == p.files
    assert "FILE.CIO" in q
    q.set_text("time.sim", q.text("time.sim").replace("2002", "2001"))
    assert "2001" in q.text("time.sim") and "2001" not in p.text("time.sim")
    with pytest.raises(ValueError, match="file.cio"):
        S.load_txtinout(tmp_path)


# ---- Fortran list-directed input ---------------------------------------------------


def test_list_read_records_and_separators():
    text = "title of file\nname  1  2.5, 3\n  4\n\n x  ,  , 7 / 8\n9*\n"
    r = _core.swat_list_read(text, ["s", "siff", "f", "sifi", "ff"])
    # each statement starts a record; the rest of its last record is dropped
    assert r[0] == (0, ["title"])
    assert r[1] == (0, ["name", 1, 2.5, 3.0])
    assert r[2] == (0, [4.0])
    # a blank record is skipped while looking for a value; two commas give a null (the
    # item keeps its value), a slash ends the statement
    assert r[3] == (0, ["x", -999, 7.0, -999])
    assert r[4] == (0, [-999.0, -999.0])


def test_list_read_slash_null_repeat_and_end_of_file():
    r = _core.swat_list_read("1, ,3 / 4\n2*5.5 3*\nlast\n", ["iiiii", "fffff", "ss"])
    assert r[0] == (0, [1, -999, 3, -999, -999])
    assert r[1] == (0, [5.5, 5.5, -999.0, -999.0, -999.0])
    # the file ends inside the statement: the items read keep their values
    assert r[2] == (-1, ["last", "<unset>"])


def test_list_read_bad_value_and_quotes():
    r = _core.swat_list_read("12 null 3\n'a b' \"c\"\"d\"\n", ["iii", "ss"])
    assert r[0] == (1, [12, -999, -999])
    assert r[1] == (0, ["a b", 'c"d'])


@pytest.mark.parametrize("text", ["0.1", "1.e-3", "2.5E+01", "-0.75", "1.d0", "3.4028235e38", "1e-45",
                                  "16777217", "0.30000001192092896", "1.17549435e-38", "7.006492e-46", "5."])
def test_text_to_float_is_correctly_rounded(text):
    # the nearest float of the exact decimal value (ties to even), never through a double
    v = Fraction(text.replace("d", "e"))
    got = _core.swat_parse_float(text)
    f = np.float32(got)
    err = abs(Fraction(float(f)) - v)
    for nb in (np.nextafter(f, np.float32(-np.inf)), np.nextafter(f, np.float32(np.inf))):
        if np.isfinite(nb):
            assert err <= abs(Fraction(float(nb)) - v)
