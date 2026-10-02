# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading R data files without R (landschaft._rdata), on files written by R 4.4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from landschaft import _rdata

DATA = Path(__file__).parent / "data" / "rdata"


def check_df(df: pd.DataFrame) -> None:
    assert df.columns.tolist() == ["i", "d", "l", "s", "f"]
    assert df["i"].tolist()[0] == 1 and pd.isna(df["i"][1]) and df["i"].tolist()[2] == 3
    assert df["d"][0] == 1.5 and np.isnan(df["d"][1]) and df["d"][2] == -0.1
    assert df["l"][0] is np.True_ or df["l"][0] == True  # noqa: E712
    assert pd.isna(df["l"][1]) and df["l"][2] == False  # noqa: E712
    assert df["s"].tolist() == ["á", None, "z"]
    assert df["f"].tolist() == ["b", "a", None]


@pytest.mark.parametrize("name", sorted(p.name for p in DATA.glob("objects_*.rda")))
def test_every_type_compression_and_version(name):
    objects = {k: _rdata.to_python(v) for k, v in _rdata.read_rda(DATA / name).items()}
    check_df(objects["df"])
    assert objects["seq_int"].tolist() == list(range(1, 11))  # ALTREP compact_intseq
    assert objects["seq_real"].tolist() == [1.5, 2.5, 3.5, 4.5, 5.5]
    np.testing.assert_array_equal(objects["m"], [[1, 3, 5], [2, 4, 6]])
    assert objects["nested"]["a"].tolist() == [1.0]
    assert objects["nested"]["b"]["c"].tolist() == ["x"] and objects["nested"]["b"]["d"].tolist() == [2, 3]
    pv = objects["pv"]
    assert pv.crs.to_epsg() == 4326 and pv["name"].tolist() == ["sq"]
    assert pv.geometry.iloc[0].area == 4.0


def test_rds():
    check_df(_rdata.to_python(_rdata.read_rds(DATA / "df.rds")))


def test_not_r_data(tmp_path):
    p = tmp_path / "x.rda"
    p.write_bytes(b"hello")
    with pytest.raises(ValueError, match="not an R data file"):
        _rdata.read_rda(p)
