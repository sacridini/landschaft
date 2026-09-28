# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial results in Biodiverse's delimited text export format.

Port of ``export_table_delimited_text`` (BaseStruct/Export.pm): one row per
group of the spatial output (every group of the BaseData, sorted), columns
``ELEMENT``, ``Axis_0``... and the list's keys in natural order
(Sort::Key::Natural). A list whose keys are the same in every group is
exported as it is (undefined values become ``no_data_value``); otherwise the
keys of all groups are used, missing keys become ``no_data_value`` and
undefined values stay empty. Array lists count the occurrences of each value.
"""

from __future__ import annotations

import os
import re

import numpy as np

from ._names import csv_field, perl_number, split_element

_PART = re.compile(r"\d+|[^\W\d_]+")


def natural_key(s: str) -> str:
    """Sort::Key::Natural ``mkkey_natural``: digit runs compare by length, then value."""
    parts = []
    for p in _PART.findall(s):
        if p[0].isdigit():
            p = p.lstrip("0")
            nines = len(p) // 9
            p = "9" * nines + str(len(p) - 9 * nines) + p
        parts.append(p)
    return "\0".join(parts)


def natsort(keys) -> list[str]:
    """``natsort`` of hash keys (ties keep the sorted-hash order of the reference runs)."""
    return sorted(sorted(keys), key=natural_key)


def _fmt(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    return perl_number(v)


def _char(c: str) -> str:
    return " " if "space" in c else "\t" if "tab" in c else c


def _element_lists(result, list_name: str):
    """({group: {key: value}}, is_array) for the groups that have the list."""
    from ._spatial import _KEY_GROUP_ARRAY, _KEY_NUMBER_ARRAY

    if list_name == "SPATIAL_RESULTS":
        t = result.table
        cols = [str(c) for c in t.columns]
        vals = t.to_numpy(dtype=np.float64)
        return {g: dict(zip(cols, row)) for g, row in zip(t.index, vals)}, False
    if list_name not in result.lists:
        raise KeyError(f"no list {list_name!r} in the results")
    kind = result.lists.raw(list_name)[0]
    df = result.lists[list_name]
    out: dict[str, dict] = {g: {} for g in result.table.index}
    if kind in (_KEY_NUMBER_ARRAY, _KEY_GROUP_ARRAY):
        for g, k in zip(df["group"], df["key"]):
            k = k if isinstance(k, str) else perl_number(k)
            d = out[g]
            d[k] = d.get(k, 0) + 1
        return out, True
    for g, k, v in zip(df["group"], df["key"], df["value"]):
        out[g][str(k)] = v
    return out, False


def biodiverse_table(result, list_name: str = "SPATIAL_RESULTS", no_data_value=None) -> list[list[str | None]]:
    """A result list as Biodiverse exports it: header row, then one row per group (None = empty)."""
    lists, is_array = _element_lists(result, list_name)
    elements = sorted(result.bd.groups)
    key_sets = [set(lists.get(g, ())) for g in elements]
    symmetric = not is_array and all(s == key_sets[0] for s in key_sets)
    columns = natsort(key_sets[0] if symmetric else set().union(*key_sets))
    names = {g: split_element(g) for g in elements}
    n_axes = max((len(a) for a in names.values()), default=0)
    no_data = _fmt(no_data_value)

    rows: list[list[str | None]] = [["ELEMENT", *(f"Axis_{i}" for i in range(n_axes)), *columns]]
    for g in elements:
        axes = names[g] + [""] * (n_axes - len(names[g]))
        vals = lists.get(g, {})
        row: list[str | None] = [g, *axes]
        for c in columns:
            if c not in vals:
                row.append(no_data)
            elif is_array:
                # Biodiverse increments the default (no data) value once per occurrence
                base = no_data_value if isinstance(no_data_value, (int, float)) else 0
                row.append(_fmt(base + vals[c]))
            else:
                f = _fmt(vals[c])
                row.append(no_data if f is None and symmetric else f)
        rows.append(row)
    return rows


def write_biodiverse_csv(result, path, list_name: str = "SPATIAL_RESULTS", sep: str = ",", quote: str = '"',
                         no_data_value=None) -> None:
    """Write :func:`biodiverse_table` as Text::CSV_XS does (native line endings)."""
    sep, quote = _char(sep), _char(quote)
    rows = biodiverse_table(result, list_name, no_data_value)
    with open(os.fspath(path), "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(sep.join("" if v is None else csv_field(v, sep, quote) for v in row) + "\n")
