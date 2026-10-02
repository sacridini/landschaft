# SPDX-License-Identifier: GPL-3.0-or-later
"""Table input/output: CSV (separator and decimal mark detected), Excel, Parquet, Feather, JSON."""

from __future__ import annotations

import gzip
import importlib.util
import os
import re
from pathlib import Path

import pandas as pd

__all__ = ["load_table", "write_table"]

_VECTOR_SUFFIXES = {".gpkg", ".shp", ".geojson", ".fgb", ".kml", ".gml", ".dbf"}


def _suffix(path: Path) -> str:
    s = path.suffix.lower()
    if s == ".gz" and len(path.suffixes) > 1:
        return path.suffixes[-2].lower()
    return s


def _sniff(path: Path, sep, decimal):
    """Separator and decimal mark of a delimited text file, from its first lines."""
    if sep is not None and decimal is not None:
        return sep, decimal
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace") as f:
        lines = [ln for ln in (f.readline() for _ in range(20)) if ln.strip()]
    header = lines[0] if lines else ""
    if sep is None:
        counts = {c: header.count(c) for c in (";", "\t", ",", "|")}
        sep = max(counts, key=counts.get) if max(counts.values()) else ","
    if decimal is None:
        body = "".join(lines[1:])
        decimal = "," if sep != "," and re.search(r"(?<![\w.])-?\d+,\d+(?![\w.])", body) else "."
    return sep, decimal


def load_table(source, *, sheet=None, sep: str | None = None, decimal: str | None = None, columns=None,
               encoding: str | None = None, **kwargs) -> pd.DataFrame:
    """Read a table, the format given by the file extension.

    ``.csv`` / ``.txt`` / ``.tsv`` (also ``.gz``; the separator ``,`` ``;``
    tab or ``|``, a decimal comma and a Windows / Latin-1 encoding are
    detected), ``.xlsx`` / ``.xls``
    (needs openpyxl), ``.parquet``, ``.feather``, ``.json``, and the
    attribute table of a vector file (``.gpkg``, ``.shp``, ``.dbf``...).

    Parameters
    ----------
    source
        File path.
    sheet
        Excel: the sheet, by name or number (default: the first).
    sep, decimal
        Delimited text: the column separator and the decimal mark (default:
        detected).
    columns
        Columns to read (default: all).
    encoding
        Delimited text: the character encoding (default: UTF-8, or Windows-1252
        / Latin-1 when the file is not valid UTF-8, as tables saved by Excel on
        Windows often are).
    **kwargs
        Passed to the pandas reader.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> from pathlib import Path
    >>> import landschaft as ls
    >>> _ = Path("species.csv").write_text("especie;peso\\nonca;2,5\\nanta;1\\n", encoding="utf-8")
    >>> ls.load_table("species.csv")                       # ';' and the decimal comma detected
      especie  peso
    0    onca   2.5
    1    anta   1.0
    """
    path = Path(os.fspath(source))
    suffix = _suffix(path)
    if suffix in (".csv", ".txt", ".tsv"):
        s, d = _sniff(path, sep, decimal)
        if encoding is not None:
            return pd.read_csv(path, sep=s, decimal=d, usecols=columns, encoding=encoding, **kwargs)
        for enc in ("utf-8", "cp1252", "latin-1"):  # latin-1 decodes any byte
            try:
                return pd.read_csv(path, sep=s, decimal=d, usecols=columns, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path, columns=columns, **kwargs)
    if suffix == ".feather":
        return pd.read_feather(path, columns=columns, **kwargs)
    if suffix in (".xlsx", ".xlsm", ".xls"):
        if not importlib.util.find_spec("openpyxl") and suffix != ".xls":
            raise ImportError("reading Excel files needs openpyxl (pip install openpyxl)")
        df = pd.read_excel(path, sheet_name=0 if sheet is None else sheet, **kwargs)
        return df[columns] if columns is not None else df
    if suffix == ".json":
        df = pd.read_json(path, **kwargs)
        return df[columns] if columns is not None else df
    if suffix in _VECTOR_SUFFIXES:
        import pyogrio

        return pyogrio.read_dataframe(path, read_geometry=False, columns=columns, **kwargs)
    raise ValueError(f"cannot tell the table format of {path.name!r}; use .csv, .xlsx, .parquet, .json...")


def write_table(table, path, **kwargs) -> Path:
    """Write a table, the format given by the file extension.

    ``.csv`` (``.csv.gz`` compressed), ``.tsv`` / ``.txt`` (tab-separated),
    ``.xlsx`` (needs openpyxl), ``.parquet`` (needs pyarrow; keeps the
    categorical and nullable integer columns, compact and fast for tables of
    millions of rows), ``.feather``, ``.json`` (one record per row), or, for a
    GeoDataFrame, a vector format (``.gpkg``, ``.geojson``, ``.shp``,
    ``.fgb``; see :func:`write_vector`). A named index (e.g. the metric names
    of a Series) is written as columns; missing folders are created.

    Parameters
    ----------
    table
        A DataFrame, Series or GeoDataFrame.
    path
        Output file; its extension selects the format.
    **kwargs
        Passed to the pandas / geopandas writer.

    Returns
    -------
    pathlib.Path
        The file written.

    Examples
    --------
    >>> import landschaft as ls
    >>> table = ls.landscape_metrics(ls.load_raster(ls.example_data("clc")), level="class", metric="ca")
    >>> ls.write_table(table, "class_area.csv").name
    'class_area.csv'
    >>> ls.load_table("class_area.csv").columns.tolist()[:4]
    ['level', 'class', 'id', 'metric']
    """
    path = Path(path)
    suffix = _suffix(path)
    if suffix in _VECTOR_SUFFIXES or (hasattr(table, "to_file") and suffix in (".geojson", ".fgb", ".kml")):
        from ._vector import write_vector

        return write_vector(table, path, **kwargs)
    if isinstance(table, pd.Series):
        table = table.to_frame()
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"write_table needs a DataFrame, got {type(table).__name__}")
    if isinstance(table.index, pd.MultiIndex) or any(n is not None for n in table.index.names):
        table = table.reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        table.to_csv(path, index=False, **kwargs)
    elif suffix in (".tsv", ".txt"):
        table.to_csv(path, index=False, sep="\t", **kwargs)
    elif suffix in (".parquet", ".pq"):
        if not importlib.util.find_spec("pyarrow"):
            raise ImportError("writing Parquet needs pyarrow (pip install pyarrow)")
        table.to_parquet(path, index=False, **kwargs)
    elif suffix == ".feather":
        table.to_feather(path, **kwargs)
    elif suffix in (".xlsx", ".xlsm"):
        if not importlib.util.find_spec("openpyxl"):
            raise ImportError("writing Excel files needs openpyxl (pip install openpyxl)")
        table.to_excel(path, index=False, **kwargs)
    elif suffix == ".json":
        table.to_json(path, orient="records", indent=1, **kwargs)
    else:
        raise ValueError(f"unknown table format {path.suffix!r}; use .csv, .csv.gz, .xlsx, .parquet, .gpkg...")
    return path
