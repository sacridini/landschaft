# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.bio readers for modern data: GBIF / Darwin Core, SDM stacks read in blocks."""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
from bio_cases import DATA

GBIF_COLUMNS = [
    "gbifID", "species", "scientificName", "occurrenceStatus", "individualCount", "decimalLatitude",
    "decimalLongitude", "coordinateUncertaintyInMeters", "basisOfRecord", "taxonKey", "issue",
]


def _gbif_rows():
    return [
        ["1", "Acacia a", "Acacia a Mill.", "PRESENT", "", "-10.2", "130.1", "50", "HUMAN_OBSERVATION", "11", ""],
        ["2", "Acacia a", "Acacia a Mill.", "PRESENT", "3", "-10.2", "130.1", "50", "HUMAN_OBSERVATION", "11", ""],
        ["3", "Acacia b", "Acacia b L.", "ABSENT", "", "-11.7", "131.3", "", "HUMAN_OBSERVATION", "12", ""],
        ["4", "Acacia b", "Acacia b L.", "PRESENT", "2", "-11.7", "131.3", "20000", "PRESERVED_SPECIMEN", "12", ""],
        ["5", "Acacia b", "Acacia b L.", "PRESENT", "", "-12.9", "129.4", "", "PRESERVED_SPECIMEN", "12",
         "ZERO_COORDINATE;COUNTRY_COORDINATE_MISMATCH"],
        ["6", "", "Acacia L.", "PRESENT", "", "-12.9", "129.4", "", "HUMAN_OBSERVATION", "10", ""],
        ["7", "Eucalyptus c", "Eucalyptus c F.Muell.", "PRESENT", "1", "", "130.5", "", "HUMAN_OBSERVATION", "13", ""],
        ["8", "Eucalyptus c", "Eucalyptus c F.Muell.", "PRESENT", "1", "-10.9", "130.5", "", "MACHINE_OBSERVATION", "13", ""],
    ]


def _write_simple_csv(path):
    text = "\t".join(GBIF_COLUMNS) + "\n" + "\n".join("\t".join(r) for r in _gbif_rows()) + "\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("0012345-260928.csv", text)


def _write_dwca(path):
    meta = """<?xml version="1.0" encoding="UTF-8"?>
<archive xmlns="http://rs.tdwg.org/dwc/text/" metadata="metadata.xml">
  <core encoding="UTF-8" fieldsTerminatedBy="\\t" linesTerminatedBy="\\n" fieldsEnclosedBy="" ignoreHeaderLines="1"
        rowType="http://rs.tdwg.org/dwc/terms/Occurrence">
    <files><location>occurrence.txt</location></files>
    <id index="0" />
""" + "".join(
        f'    <field index="{i}" term="http://rs.tdwg.org/dwc/terms/{c}"/>\n' for i, c in enumerate(GBIF_COLUMNS) if i
    ) + """  </core>
</archive>
"""
    header = "\t".join(GBIF_COLUMNS)
    text = header + "\n" + "\n".join("\t".join(r) for r in _gbif_rows()) + "\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("meta.xml", meta)
        z.writestr("occurrence.txt", text)


@pytest.mark.parametrize("writer", [_write_simple_csv, _write_dwca])
def test_read_occurrences_filters(tmp_path, writer):
    path = tmp_path / "download.zip"
    writer(path)
    occ = ls.bio.read_occurrences(path, chunksize=3)
    # absent (3), no species (6), no latitude (7) dropped; duplicate 2 of 1 dropped
    assert occ["label"].tolist() == ["Acacia a", "Acacia b", "Acacia b", "Eucalyptus c"]
    assert occ["count"].tolist() == [1.0, 1.0, 1.0, 1.0]
    occ = ls.bio.read_occurrences(path, count="individualCount", drop_duplicates=False,
                                  max_uncertainty=1000, exclude_issues=["ZERO_COORDINATE"],
                                  basis_of_record=["human_observation", "PRESERVED_SPECIMEN"])
    assert occ.index.tolist() == [0, 1]
    assert occ["count"].tolist() == [1.0, 3.0]
    occ = ls.bio.read_occurrences(path, label=["taxonKey"])
    assert occ["label"].tolist() == ["11", "12", "12", "10", "13"]


def test_from_occurrences_matches_table_import(tmp_path):
    path = tmp_path / "download.zip"
    _write_simple_csv(path)
    bd = ls.bio.BaseData.from_occurrences(path, [0.5, 0.5])
    occ = ls.bio.read_occurrences(path)
    ref = ls.bio.BaseData.from_dataframe(occ, ["decimalLongitude", "decimalLatitude"], "label", "count",
                                         cell_sizes=[0.5, 0.5])
    # the table import quotes names with spaces, as Biodiverse does; occurrences keep them plain
    assert bd.groups == ref.groups and bd.labels == [lb.strip("'") for lb in ref.labels]
    assert (bd.counts != ref.counts).nnz == 0
    assert str(bd.crs) == "EPSG:4326"
    assert ls.bio.BaseData.from_file(path, cell_sizes=[0.5, 0.5]).groups == bd.groups


def test_from_occurrences_reprojects():
    occ = pd.DataFrame({"species": ["a", "b"], "decimalLongitude": [0.0, 1.0], "decimalLatitude": [0.0, 0.0]})
    bd = ls.bio.BaseData.from_occurrences(occ, [100_000, 100_000], to_crs="EPSG:3857")
    # 1 degree of longitude at the equator is 111 319.49 m in Web Mercator
    assert bd.groups == ["150000:50000", "50000:50000"]


def test_raster_blocks_give_the_same_basedata():
    for f, lab in [("raster_multi.tif", True), ("raster_values.tif", False)]:
        whole = ls.bio.BaseData.from_raster(DATA / f, labels_as_bands=lab, cell_sizes=[20, 20])
        blocks = ls.bio.BaseData.from_raster(DATA / f, labels_as_bands=lab, cell_sizes=[20, 20], block_rows=5)
        assert whole.groups == blocks.groups and whole.labels == blocks.labels
        assert (whole.counts != blocks.counts).nnz == 0


def test_sdm_stack_from_lazy_xarray(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")
    rng = np.random.default_rng(3)
    z = rng.random((4, 30, 25))
    z[1, 3:8, 4:9] = np.nan
    ds = xr.Dataset(
        {f"sp{i}": (("y", "x"), z[i]) for i in range(4)},
        coords={"y": 1000.0 - 10 * np.arange(30) - 5, "x": 10 * np.arange(25) + 5.0},
    )
    path = tmp_path / "sdm.nc"
    ds.to_netcdf(path)
    eager = ls.bio.BaseData.from_xarray(ds, threshold=0.7, cell_sizes=[50, 50])
    lazy = ls.bio.BaseData.from_xarray(xr.open_dataset(path, chunks={"y": 7}), threshold=0.7, cell_sizes=[50, 50],
                                       block_rows=4)
    assert eager.labels == ["sp0", "sp1", "sp2", "sp3"]
    assert eager.groups == lazy.groups
    assert (eager.counts != lazy.counts).nnz == 0
    # presence counts per 5x5 block of pixels
    got = eager.to_dataframe().loc["25:975", "sp0"]
    assert got == float((z[0, 0:5, 0:5] >= 0.7).sum())
    weighted = ls.bio.BaseData.from_xarray(ds, cell_sizes=[50, 50])
    assert weighted.to_dataframe().loc["25:975", "sp0"] == pytest.approx(z[0, 0:5, 0:5].sum())
