# SPDX-License-Identifier: GPL-3.0-or-later
"""Element (label and group) properties, as Biodiverse::ElementProperties."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ._names import dequote_element, join_element, looks_like_number, perl_number

_RESERVED = {"INCLUDE", "EXCLUDE", "REMAP"}


def read_properties(
    source: str | os.PathLike,
    element_columns: Sequence[int],
    property_columns: Mapping[str, int],
    *,
    sep: str | None = None,
    quote: str | None = None,
) -> pd.DataFrame:
    """Read a Biodiverse element properties file (label or group properties).

    Property names are upper-cased and only values that look like numbers are
    kept, as Biodiverse does; empty fields are kept as empty strings (the
    property is set but undefined: listed, not used in statistics). A later
    row for the same element overrides the earlier one.

    Parameters
    ----------
    source : path
        Delimited text with a header line.
    element_columns : sequence of int
        Columns forming the element name (joined with ``:``).
    property_columns : mapping
        Property name to column.
    sep, quote : str, optional
        Separator and quote character (guessed by default).

    Returns
    -------
    pandas.DataFrame
        One row per element, one column per property; values are the strings
        read (``""`` for empty fields, ``None`` where not set); elements
        without any property are left out.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> import landschaft as ls
    >>> path = pathlib.Path(tempfile.mkdtemp()) / "props.csv"
    >>> _ = path.write_text("genus,species,mass\\nGenus,sp1,1.5\\nGenus,sp2,x\\nGenus,sp3,\\n")
    >>> props = ls.bio.read_properties(path, [0, 1], {"mass": 2})
    >>> props["MASS"].to_dict()
    {'Genus:sp1': '1.5', 'Genus:sp3': ''}
    """
    from ._basedata import _read_rows

    rows = _read_rows(source, sep, quote)[1:]
    names = {str(k).upper(): int(v) for k, v in property_columns.items()}
    records: dict[str, dict[str, str | None]] = {}
    for row in rows:
        if not row:
            continue
        cols = [row[c] if c < len(row) else None for c in element_columns]
        element = join_element(["" if v is None else v for v in cols])
        if len(cols) <= 1:
            element = dequote_element(element)
        rec = records.setdefault(element, {})
        for name, col in names.items():
            value = row[col] if col < len(row) else None
            if value is None:
                rec[name] = ""  # an undefined value
            elif looks_like_number(value):
                rec[name] = value
    records = {e: r for e, r in records.items() if r}
    df = pd.DataFrame.from_dict(records, orient="index", columns=list(names), dtype=object)
    df.index.name = "element"
    return df.where(pd.notna(df), None)


def normalise_properties(props: pd.DataFrame, elements: Sequence[str]) -> pd.DataFrame:
    """Properties of the given elements as Perl strings (numbers as Perl prints them)."""
    props = pd.DataFrame(props)
    props.columns = [str(c) for c in props.columns]
    props = props[[c for c in props.columns if c not in _RESERVED]]
    idx = [str(i) for i in props.index]
    props.index = idx

    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, str):
            return v if (v == "" or looks_like_number(v)) else None
        if isinstance(v, (bool, np.bool_)):
            return str(int(v))
        return perl_number(v)

    out = props.reindex(list(elements)).astype(object)
    out = out.apply(lambda col: col.map(conv))
    out.index.name = props.index.name
    return out


def property_tables(props: pd.DataFrame) -> list[tuple[str, list[str], np.ndarray, np.ndarray]]:
    """Per property: distinct values (Perl's string order), their numbers and each element's value index.

    The index is -1 where the property is not set and -2 where it is set but
    undefined (an empty string).
    """
    out = []
    for name in props.columns:
        col = props[name].tolist()
        if all(v is None for v in col):
            continue
        keys = sorted({v for v in col if v is not None and v != ""}, key=lambda s: s.encode("utf-8"))
        pos = {k: i for i, k in enumerate(keys)}
        idx = np.array([-1 if v is None else (-2 if v == "" else pos[v]) for v in col], dtype=np.int32)
        values = np.array([float(k) for k in keys], dtype=np.float64)
        out.append((name, keys, values, idx))
    return out
