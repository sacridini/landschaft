# SPDX-License-Identifier: GPL-3.0-or-later
"""Occurrence records from GBIF downloads and Darwin Core Archives."""

from __future__ import annotations

import csv
import io
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["read_occurrences"]

_DWC_NS = "http://rs.tdwg.org/dwc/text/"

# columns a BaseData needs, besides the label and count columns
_COORDS = ("decimalLongitude", "decimalLatitude")
_FILTER_COLUMNS = ("occurrenceStatus", "coordinateUncertaintyInMeters", "basisOfRecord", "issue", "hasGeospatialIssues")


def _term(uri: str) -> str:
    """Short Darwin Core term name of a term URI."""
    return re.split(r"[/#]", uri.rstrip("/"))[-1]


def _unescape(s: str | None, default: str) -> str:
    if s is None:
        return default
    return s.encode("utf-8").decode("unicode_escape")


class _Source:
    """A delimited occurrence table inside a file or a zip."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.zip: zipfile.ZipFile | None = None
        self.member: str | None = None
        self.sep = "\t"
        self.quote: str | None = None
        self.skip = 0
        self.columns: list[str] | None = None
        self.encoding = "utf-8"
        if zipfile.is_zipfile(self.path):
            self.zip = zipfile.ZipFile(self.path)
            names = self.zip.namelist()
            if "meta.xml" in names:
                self._read_meta(self.zip.read("meta.xml"))
            else:
                # a GBIF "simple CSV" download: one tab-separated table
                tables = [n for n in names if n.lower().endswith((".csv", ".txt", ".tsv"))]
                if not tables:
                    raise ValueError(f"no occurrence table in {self.path}")
                self.member = tables[0]
        elif self.path.is_dir() and (self.path / "meta.xml").exists():
            self._read_meta((self.path / "meta.xml").read_bytes())
        else:
            head = self.path.read_bytes()[:4096].decode("utf-8-sig", errors="replace")
            first = head.splitlines()[0] if head else ""
            self.sep = "\t" if "\t" in first else ","

    def _read_meta(self, xml: bytes) -> None:
        root = ET.fromstring(xml)
        core = root.find(f"{{{_DWC_NS}}}core")
        if core is None:
            core = root.find("core")
        if core is None:
            raise ValueError("meta.xml has no core table")
        location = core.find(f".//{{{_DWC_NS}}}location")
        if location is None:
            location = core.find(".//location")
        self.member = location.text.strip()
        self.sep = _unescape(core.get("fieldsTerminatedBy"), "\t")
        q = core.get("fieldsEnclosedBy")
        self.quote = _unescape(q, "") or None
        self.skip = int(core.get("ignoreHeaderLines", "0"))
        self.encoding = core.get("encoding", "utf-8") or "utf-8"
        fields = core.findall(f"{{{_DWC_NS}}}field") or core.findall("field")
        cols: dict[int, str] = {}
        for f in fields:
            if f.get("index") is not None:
                cols[int(f.get("index"))] = _term(f.get("term"))
        idx = core.find(f"{{{_DWC_NS}}}id")
        if idx is None:
            idx = core.find("id")
        if idx is not None and idx.get("index") is not None:
            cols.setdefault(int(idx.get("index")), "id")
        if cols:
            n = max(cols) + 1
            self.columns = [cols.get(i, f"_col{i}") for i in range(n)]

    def open(self) -> io.TextIOBase:
        if self.zip is not None:
            if self.member is None:
                raise ValueError("no table")
            raw = self.zip.open(self.member)
            if self.path.is_dir():
                raise AssertionError
            return io.TextIOWrapper(raw, encoding=self.encoding, errors="replace", newline="")
        if self.path.is_dir():
            return open(self.path / self.member, encoding=self.encoding, errors="replace", newline="")
        return open(self.path, encoding="utf-8-sig", errors="replace", newline="")

    def chunks(self, usecols: Sequence[str], chunksize: int) -> Iterator[pd.DataFrame]:
        kw: dict[str, Any] = {
            "sep": self.sep,
            "dtype": str,
            "keep_default_na": False,
            "na_values": [""],
            "chunksize": chunksize,
            "on_bad_lines": "warn",
        }
        if self.quote:
            kw["quotechar"] = self.quote
        else:
            kw["quoting"] = csv.QUOTE_NONE
        with self.open() as fh:
            if self.columns is not None:
                for _ in range(self.skip):
                    fh.readline()
                kw["header"] = None
                kw["names"] = self.columns
            available = self.columns
            if available is None:
                header = fh.readline().rstrip("\r\n").split(self.sep)
                available = [h.strip().strip('"') for h in header]
                kw["header"] = None
                kw["names"] = available
            use = [c for c in usecols if c in available]
            kw["usecols"] = use
            yield from pd.read_csv(fh, **kw)


def read_occurrences(
    source: str | os.PathLike | pd.DataFrame,
    *,
    label: str | Sequence[str] = "species",
    count: str | None = None,
    max_uncertainty: float | None = None,
    basis_of_record: Iterable[str] | None = None,
    exclude_issues: Iterable[str] | None = None,
    present_only: bool = True,
    drop_duplicates: bool = True,
    extra_columns: Sequence[str] = (),
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Occurrence records of a GBIF download or a Darwin Core Archive, cleaned.

    Parameters
    ----------
    source : path or DataFrame
        A GBIF "simple CSV" download (the zip or the tab-separated file), a
        Darwin Core Archive (zip or folder with ``meta.xml``; the core table is
        read) or a DataFrame with Darwin Core column names (e.g. from
        ``pygbif``).
    label : str or list of str
        Column(s) naming the taxon: ``"species"`` (default),
        ``"scientificName"``, ``"taxonKey"``, ``"speciesKey"``...
    count : str, optional
        Column with the number of individuals (``"individualCount"``); empty
        values count 1. Default: 1 per record.
    max_uncertainty : float, optional
        Drop records with ``coordinateUncertaintyInMeters`` above this (records
        without an uncertainty are kept).
    basis_of_record : list of str, optional
        Keep only these ``basisOfRecord`` values (e.g. ``["HUMAN_OBSERVATION",
        "PRESERVED_SPECIMEN"]``).
    exclude_issues : list of str, optional
        Drop records flagged with any of these GBIF issues (column ``issue``,
        e.g. ``"ZERO_COORDINATE"``, ``"COUNTRY_COORDINATE_MISMATCH"``).
    present_only : bool
        Drop records whose ``occurrenceStatus`` is ``ABSENT``.
    drop_duplicates : bool
        Keep one record per label and coordinate pair.
    extra_columns : list of str
        Other columns to keep.
    chunksize : int
        Rows read at a time (large downloads never sit whole in memory).

    Returns
    -------
    pandas.DataFrame
        ``label``, ``decimalLongitude``, ``decimalLatitude``, ``count`` (and
        the extra columns), in file order.

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> df = pd.DataFrame({"species": ["A a", "A a", "B b", None],
    ...                    "decimalLongitude": ["10.1", "10.1", "11.5", "12"],
    ...                    "decimalLatitude": ["-5.2", "-5.2", "-5.9", "-6"],
    ...                    "basisOfRecord": ["HUMAN_OBSERVATION"] * 4})
    >>> ls.bio.read_occurrences(df)
      label  decimalLongitude  decimalLatitude  count
    0   A a              10.1             -5.2    1.0
    2   B b              11.5             -5.9    1.0
    """
    labels = [label] if isinstance(label, str) else list(label)
    wanted = list(dict.fromkeys([*labels, *_COORDS, *_FILTER_COLUMNS, *([count] if count else []), *extra_columns]))
    issues = set(exclude_issues or [])
    bor = None if basis_of_record is None else {b.upper() for b in basis_of_record}

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in [*labels, *_COORDS] if c not in df.columns]
        if missing:
            raise KeyError(f"columns not found: {missing}")
        df = df.copy()
        for c in labels:
            df[c] = df[c].astype("string").str.strip()
        lab = df[labels[0]] if len(labels) == 1 else df[labels].astype("string").agg(":".join, axis=1)
        x = pd.to_numeric(df["decimalLongitude"], errors="coerce")
        y = pd.to_numeric(df["decimalLatitude"], errors="coerce")
        keep = lab.notna() & (lab != "") & x.notna() & y.notna()
        keep &= x.between(-180, 180) & y.between(-90, 90)
        if present_only and "occurrenceStatus" in df.columns:
            keep &= df["occurrenceStatus"].fillna("PRESENT").str.upper() != "ABSENT"
        if max_uncertainty is not None and "coordinateUncertaintyInMeters" in df.columns:
            u = pd.to_numeric(df["coordinateUncertaintyInMeters"], errors="coerce")
            keep &= u.isna() | (u <= max_uncertainty)
        if bor is not None and "basisOfRecord" in df.columns:
            keep &= df["basisOfRecord"].fillna("").str.upper().isin(bor)
        if issues and "issue" in df.columns:
            flags = df["issue"].fillna("").str.split(";")
            keep &= ~flags.apply(lambda f: bool(issues.intersection(f)))
        if count and count in df.columns:
            n = pd.to_numeric(df[count], errors="coerce").fillna(1.0)
        else:
            n = pd.Series(1.0, index=df.index)
        out = pd.DataFrame({"label": lab, "decimalLongitude": x, "decimalLatitude": y, "count": n.astype(np.float64)})
        for c in extra_columns:
            if c in df.columns:
                out[c] = df[c]
        return out[keep.to_numpy(dtype=bool)]

    if isinstance(source, pd.DataFrame):
        frames = [clean(source)]
    else:
        src = _Source(source)
        frames = []
        offset = 0
        for chunk in src.chunks(wanted, chunksize):
            chunk.index = pd.RangeIndex(offset, offset + len(chunk))
            offset += len(chunk)
            frames.append(clean(chunk))
    out = pd.concat(frames) if frames else pd.DataFrame(columns=["label", "decimalLongitude", "decimalLatitude", "count"])
    out["label"] = out["label"].astype(object)
    if drop_duplicates:
        out = out.drop_duplicates(subset=["label", "decimalLongitude", "decimalLatitude"], keep="first")
    return out
