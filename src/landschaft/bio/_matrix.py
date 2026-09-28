# SPDX-License-Identifier: GPL-3.0-or-later
"""Label dissimilarity matrices, read as Biodiverse reads them."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._basedata import _read_rows
from ._names import csv_field, dequote_element, looks_like_number

__all__ = ["Matrix", "read_matrix"]


class Matrix:
    """A symmetric matrix of values between elements (usually labels).

    Missing pairs are NaN. A pair holds the first value read for it (either
    order), like Biodiverse; the diagonal is ``self_similarity`` unless a value
    was given.

    Parameters
    ----------
    elements : sequence of str
        Element names (stored sorted).
    values : array (n, n)
        Values in the order of ``elements`` (NaN for missing pairs).
    self_similarity : float, optional
        Value of (a, a) pairs that were not given.
    """

    def __init__(self, elements: Sequence[str], values: np.ndarray, self_similarity: float | None = None) -> None:
        elements = list(elements)
        values = np.asarray(values, dtype=np.float64)
        n = len(elements)
        if values.shape != (n, n):
            raise ValueError("values must be (n, n)")
        order = sorted(range(n), key=elements.__getitem__)
        self.elements: list[str] = [elements[i] for i in order]
        self.values: np.ndarray = values[np.ix_(order, order)]
        self.self_similarity = self_similarity
        self._index = {e: i for i, e in enumerate(self.elements)}

    def __repr__(self) -> str:
        return f"Matrix({len(self.elements)} elements, {self.n_pairs} pairs)"

    def __contains__(self, element: object) -> bool:
        return element in self._index

    @property
    def n_pairs(self) -> int:
        """Number of defined pairs (each unordered pair once, diagonal included)."""
        v = self.values
        return int(np.count_nonzero(~np.isnan(v[np.triu_indices(len(self.elements))])))

    def get(self, a: str, b: str) -> float:
        """Value of the pair (NaN if missing; ``self_similarity`` for a missing (a, a))."""
        i, j = self._index.get(a), self._index.get(b)
        if i is None or j is None:
            return np.nan
        v = self.values[i, j]
        if np.isnan(v) and i == j and self.self_similarity is not None:
            return float(self.self_similarity)
        return float(v)

    def index(self, element: str) -> int:
        return self._index[element]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, index=self.elements, columns=self.elements)

    @classmethod
    def from_pairs(cls, pairs, self_similarity: float | None = None) -> Matrix:
        """From ``(element1, element2, value)`` records (first value per pair wins)."""
        pairs = list(pairs)
        names = sorted({str(p[0]) for p in pairs} | {str(p[1]) for p in pairs})
        idx = {e: i for i, e in enumerate(names)}
        v = np.full((len(names), len(names)), np.nan)
        for a, b, val in pairs:
            i, j = idx[str(a)], idx[str(b)]
            if np.isnan(v[i, j]):
                v[i, j] = v[j, i] = float(val)
        return cls(names, v, self_similarity)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, self_similarity: float | None = None) -> Matrix:
        """Square DataFrame (index = columns = elements); the upper-left value of a pair wins."""
        names = [str(c) for c in df.columns]
        v = df.to_numpy(dtype=np.float64)
        out = np.full(v.shape, np.nan)
        for i in range(len(names)):
            for j in range(len(names)):
                x = v[i, j]
                if np.isnan(x) or not np.isnan(out[i, j]):
                    continue
                out[i, j] = out[j, i] = x
        return cls(names, out, self_similarity)


def read_matrix(
    source: str | os.PathLike,
    element_columns: Sequence[int] = (1, 2),
    values_start_col: int | None = None,
    *,
    sparse: bool = False,
    row_columns: Sequence[int] | None = None,
    col_columns: Sequence[int] | None = None,
    value_column: int | None = None,
    sep: str | None = None,
    quote: str | None = None,
    self_similarity: float | None = None,
) -> Matrix:
    """Read a Biodiverse matrix file (full or lower/upper triangle, or sparse pairs).

    Parameters
    ----------
    source : path
        Delimited text.
    element_columns : sequence of int
        Columns forming the element name of each row (joined with ``:``);
        Biodiverse's default is columns 1 and 2 (genus, species).
    values_start_col : int, optional
        First value column (default: after the last element column). The
        value columns follow the order of the rows.
    sparse : bool
        Sparse format: one pair per line, names in ``row_columns`` and
        ``col_columns``, value in ``value_column``.

    Returns
    -------
    Matrix
    """
    rows = _read_rows(source, sep, quote)[1:]
    pairs: list[tuple[str, str, float]] = []
    if sparse:
        if row_columns is None or col_columns is None or value_column is None:
            raise ValueError("sparse matrices need row_columns, col_columns and value_column")
        for r in rows:
            if not r:
                continue
            a = dequote_element(":".join(csv_field(r[c] or "") for c in row_columns))
            b = dequote_element(":".join(csv_field(r[c] or "") for c in col_columns))
            val = r[value_column]
            # Biodiverse: row label from the col columns and vice versa; the pair is symmetric
            if val is None or not looks_like_number(val):
                continue
            pairs.append((b, a, float(val)))
    else:
        cols = list(element_columns)
        start = values_start_col if values_start_col is not None else cols[-1] + 1
        labels: list[str] = []
        data: list[list[Any]] = []
        for r in rows:
            if not r:
                continue
            lab = dequote_element(":".join(csv_field(r[c] or "") if c < len(r) else "" for c in cols))
            labels.append(lab)
            data.append(r[start:])
        for i, vals in enumerate(data):
            for j in range(len(labels)):
                val = vals[j] if j < len(vals) else None
                if val is None or val == "" or not looks_like_number(val):
                    continue
                pairs.append((labels[i], labels[j], float(val)))
    names = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    idx = {e: i for i, e in enumerate(names)}
    v = np.full((len(names), len(names)), np.nan)
    for a, b, val in pairs:
        i, j = idx[a], idx[b]
        if np.isnan(v[i, j]):
            v[i, j] = v[j, i] = val
    return Matrix(names, v, self_similarity)
