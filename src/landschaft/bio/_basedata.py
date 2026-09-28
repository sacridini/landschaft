# SPDX-License-Identifier: GPL-3.0-or-later
"""BaseData: the groups × labels table that Biodiverse analyses.

Groups are cells (or any unit with coordinates), labels are taxa, and each
(group, label) pair holds a sample count. Coordinates are snapped to the cell
grid exactly as Biodiverse does on import, and groups and labels are numbered
in the sorted order of their Biodiverse names (so sums over them follow the
order of the reference runs).
"""

from __future__ import annotations

import csv
import io
import os
import re
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._names import (
    csv_field,
    dequote_element,
    looks_like_number,
    perl_number,
    split_element,
)

__all__ = ["BaseData"]

_PRECISION = 1e10  # Biodiverse DEFAULT_PRECISION for the snapping


def _round_half_away(x: np.ndarray) -> np.ndarray:
    """POSIX ``round``: halves away from zero, exact."""
    t = np.trunc(x)
    frac = x - t
    return t + np.where(np.abs(frac) >= 0.5, np.sign(x), 0.0)


def snap_coordinates(values: np.ndarray, cell_size: float, origin: float = 0.0) -> np.ndarray:
    """Cell centres as Biodiverse's text import computes them (before naming).

    ``size * floor(round((x - origin) / size * 1e10) / 1e10) + size / 2 + origin``.
    """
    v = np.asarray(values, dtype=np.float64)
    r = _round_half_away(((v - origin) / cell_size) * _PRECISION) / _PRECISION
    return (cell_size * np.floor(r) + cell_size / 2) + origin


def _natural_key(s: str) -> list:
    return [(0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"(\d+)", s) if p != ""]


def _perl_true(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v not in ("", "0")
    return bool(v)


def _guess_separator(lines: list[str], quote: str) -> str:
    best, best_n = ",", 1
    for sep in (",", "\t", ";", " "):
        if not any(sep in ln for ln in lines):
            continue
        counts = []
        for row in csv.reader(lines, delimiter=sep, quotechar=quote):
            if row:
                counts.append(len(row))
        if counts and min(counts) > 1 and len(set(counts)) == 1 and counts[0] > best_n:
            best, best_n = sep, counts[0]
    if best_n == 1:
        # fall back: the separator giving most fields on the header
        for sep in (",", "\t", ";", " "):
            n = len(next(csv.reader([lines[0]], delimiter=sep, quotechar=quote)))
            if n > best_n:
                best, best_n = sep, n
    return best


def _guess_quote(text: str) -> str:
    counts = {}
    for q in ('"', "'"):
        k = text.count(q)
        if k and k % 2 == 0:
            counts.setdefault(k, q)
    return counts[max(counts)] if counts else '"'


def _read_rows(source: str | os.PathLike, sep: str | None, quote: str | None) -> list[list[str | None]]:
    raw = Path(source).read_bytes()
    text = raw.decode("utf-8-sig")
    head = text[:20000].splitlines()[:11] or [""]
    if quote is None or quote == "guess":
        quote = _guess_quote("\n".join(head))
    if sep is None or sep == "guess":
        sep = _guess_separator(head, quote)
    elif sep == "tab":
        sep = "\t"
    elif sep == "space":
        sep = " "
    rows = []
    for row in csv.reader(io.StringIO(text, newline=""), delimiter=sep, quotechar=quote):
        # Text::CSV_XS blank_is_undef: empty unquoted fields are undef
        rows.append([None if f == "" else f for f in row])
    return rows


class BaseData:
    """Groups × labels sample counts with the group coordinates.

    Build one with :meth:`from_table` (Biodiverse's text import),
    :meth:`from_dataframe`, :meth:`from_sparse`, :meth:`from_xarray`,
    :meth:`from_raster` or :meth:`from_points`.

    Attributes
    ----------
    groups : list of str
        Group names, sorted (Biodiverse's names: coordinates joined with ``:``).
    labels : list of str
        Label names, sorted.
    counts : scipy.sparse.csr_matrix
        Sample counts, ``(n_groups, n_labels)``.
    coords : numpy.ndarray
        Group coordinates ``(n_groups, n_axes)``; text axes hold their rank.
    cell_sizes, cell_origins : tuple of float
        Cell size per axis (0: coordinates kept as given, negative: text axis)
        and grid origin.
    crs : optional
        Coordinate reference system of the group coordinates (anything
        ``pyproj``/``rasterio`` accept), or None.
    """

    def __init__(
        self,
        counts: sp.spmatrix,
        groups: Sequence[str],
        labels: Sequence[str],
        coords: np.ndarray,
        cell_sizes: Sequence[float],
        cell_origins: Sequence[float] | None = None,
        crs: Any = None,
        group_axes: Sequence[Sequence[str]] | None = None,
    ) -> None:
        groups = list(groups)
        labels = list(labels)
        counts = sp.csr_matrix(counts, dtype=np.float64)
        coords = np.asarray(coords, dtype=np.float64).reshape(len(groups), -1)
        if counts.shape != (len(groups), len(labels)):
            raise ValueError("counts must have shape (n_groups, n_labels)")
        # keep Biodiverse's name order
        g_order = np.argsort(np.array(groups, dtype=object), kind="stable") if groups else np.zeros(0, int)
        l_order = np.argsort(np.array(labels, dtype=object), kind="stable") if labels else np.zeros(0, int)
        if len(set(groups)) != len(groups):
            raise ValueError("group names must be unique")
        if len(set(labels)) != len(labels):
            raise ValueError("label names must be unique")
        self.groups: list[str] = [groups[i] for i in g_order]
        self.labels: list[str] = [labels[i] for i in l_order]
        self.counts: sp.csr_matrix = counts[g_order][:, l_order].tocsr()
        self.counts.sort_indices()
        self.coords: np.ndarray = coords[g_order]
        self.cell_sizes = tuple(float(c) for c in cell_sizes)
        if cell_origins is None:
            cell_origins = [0.0] * len(self.cell_sizes)
        self.cell_origins = tuple(float(c) for c in cell_origins)
        if self.coords.shape[1] != len(self.cell_sizes):
            raise ValueError("one cell size per coordinate axis is needed")
        self.crs = crs
        self._group_axes = None if group_axes is None else [list(ax) for ax in np.asarray(group_axes, dtype=object)[g_order]]
        self._group_index = {g: i for i, g in enumerate(self.groups)}
        self._label_index = {lb: i for i, lb in enumerate(self.labels)}
        self.label_properties = pd.DataFrame(index=pd.Index(self.labels, name="label"))
        self.group_properties = pd.DataFrame(index=pd.Index(self.groups, name="group"))

    # -- sizes and lookups -----------------------------------------------------------
    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_labels(self) -> int:
        return len(self.labels)

    @property
    def n_axes(self) -> int:
        return len(self.cell_sizes)

    def group_index(self, name: str) -> int:
        """Row of group ``name``."""
        return self._group_index[name]

    def label_index(self, name: str) -> int:
        """Column of label ``name``."""
        return self._label_index[name]

    def __repr__(self) -> str:
        return (
            f"BaseData({self.n_groups} groups, {self.n_labels} labels, "
            f"cell_sizes={list(self.cell_sizes)}, {self.counts.nnz} records)"
        )

    # -- summaries -------------------------------------------------------------------
    @property
    def label_range(self) -> pd.Series:
        """Number of groups each label occurs in (Biodiverse's range)."""
        v = np.diff(self.counts.tocsc().indptr)
        return pd.Series(v, index=pd.Index(self.labels, name="label"), name="range")

    @property
    def label_sample_count(self) -> pd.Series:
        """Total sample count of each label (summed over groups in name order)."""
        # bincount adds sequentially in CSR order: groups ascending for every label
        out = np.bincount(self.counts.indices, weights=self.counts.data, minlength=self.n_labels)
        return pd.Series(out, index=pd.Index(self.labels, name="label"), name="sample_count")

    @property
    def group_richness(self) -> pd.Series:
        """Number of labels in each group."""
        v = np.diff(self.counts.indptr)
        return pd.Series(v, index=pd.Index(self.groups, name="group"), name="richness")

    @property
    def group_sample_count(self) -> pd.Series:
        """Total sample count of each group (summed over labels in name order)."""
        rows = np.repeat(np.arange(self.n_groups), np.diff(self.counts.indptr))
        out = np.bincount(rows, weights=self.counts.data, minlength=self.n_groups)
        return pd.Series(out, index=pd.Index(self.groups, name="group"), name="sample_count")

    @property
    def numeric_labels(self) -> bool:
        """True when every label looks like a number."""
        return all(looks_like_number(lb) for lb in self.labels)

    def labels_in_group(self, group: str) -> dict[str, float]:
        """Labels of ``group`` with their sample counts."""
        i = self._group_index[group]
        c = self.counts
        sl = slice(c.indptr[i], c.indptr[i + 1])
        return {self.labels[j]: float(v) for j, v in zip(c.indices[sl], c.data[sl])}

    def groups_with_label(self, label: str) -> dict[str, float]:
        """Groups where ``label`` occurs, with their sample counts."""
        j = self._label_index[label]
        col = self.counts[:, j].tocoo()
        return {self.groups[i]: float(v) for i, v in zip(col.row, col.data)}

    def group_axes(self, group: str) -> list[str]:
        """Axis values of a group as Biodiverse stores them (strings)."""
        if self._group_axes is not None:
            return list(self._group_axes[self._group_index[group]])
        return split_element(group) if self.n_axes > 1 else [group]

    # -- export ----------------------------------------------------------------------
    def to_table(self) -> pd.DataFrame:
        """Long table: one row per (group, label) with its count and coordinates."""
        coo = self.counts.tocoo()
        order = np.lexsort((coo.col, coo.row))
        rows, cols, vals = coo.row[order], coo.col[order], coo.data[order]
        df = pd.DataFrame(
            {
                "group": np.asarray(self.groups, dtype=object)[rows],
                "label": np.asarray(self.labels, dtype=object)[cols],
                "count": vals,
            }
        )
        for k in range(self.n_axes):
            df[f"axis_{k}"] = self.coords[rows, k]
        return df

    def to_dataframe(self) -> pd.DataFrame:
        """Wide groups × labels table of counts (0 where absent)."""
        return pd.DataFrame(self.counts.toarray(), index=pd.Index(self.groups, name="group"), columns=self.labels)

    def to_raster(self, values: Mapping[str, float] | pd.Series | np.ndarray, name: str | None = None, fill: float = np.nan):
        """Grid of per-group values (two numeric axes needed).

        Parameters
        ----------
        values : Series (indexed by group), mapping or array (group order)
            The values to place in each group's cell.
        name : str, optional
            Name of the returned DataArray.
        fill : float
            Value of cells without a group.

        Returns
        -------
        xarray.DataArray
            ``y`` decreasing, ``x`` increasing, cell centres as coordinates,
            ``crs`` in ``attrs`` when known.
        """
        import xarray as xr

        if self.n_axes != 2 or min(self.cell_sizes) <= 0:
            raise ValueError("to_raster needs two numeric axes with positive cell sizes")
        if isinstance(values, np.ndarray):
            vals = np.asarray(values, dtype=np.float64)
            if vals.shape != (self.n_groups,):
                raise ValueError("an array of values needs one value per group")
        else:
            s = pd.Series(values, dtype=np.float64)
            vals = s.reindex(self.groups).to_numpy()
        dx, dy = self.cell_sizes
        x, y = self.coords[:, 0], self.coords[:, 1]
        if self.n_groups == 0:
            raise ValueError("no groups")
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        nx = int(round((x1 - x0) / dx)) + 1
        ny = int(round((y1 - y0) / dy)) + 1
        col = np.rint((x - x0) / dx).astype(np.int64)
        row = np.rint((y1 - y) / dy).astype(np.int64)
        grid = np.full((ny, nx), fill, dtype=np.float64)
        grid[row, col] = vals
        da = xr.DataArray(
            grid,
            dims=("y", "x"),
            coords={"y": y1 - dy * np.arange(ny), "x": x0 + dx * np.arange(nx)},
            name=name,
        )
        if self.crs is not None:
            da.attrs["crs"] = str(self.crs)
            try:
                import rioxarray  # noqa: F401

                da = da.rio.write_crs(self.crs)
            except Exception:  # pragma: no cover - rioxarray is a dependency
                pass
        return da

    def to_geodataframe(self, values: pd.DataFrame | pd.Series | None = None):
        """Cells as polygons (needs geopandas), optionally with per-group values."""
        import geopandas as gpd
        from shapely.geometry import box

        if self.n_axes != 2:
            raise ValueError("to_geodataframe needs two axes")
        dx, dy = self.cell_sizes
        hx, hy = (dx / 2, dy / 2) if dx > 0 and dy > 0 else (0.0, 0.0)
        geoms = [box(x - hx, y - hy, x + hx, y + hy) if hx else None for x, y in self.coords]
        if not hx:
            from shapely.geometry import Point

            geoms = [Point(x, y) for x, y in self.coords]
        df = pd.DataFrame(index=pd.Index(self.groups, name="group"))
        if values is not None:
            df = df.join(pd.DataFrame(values).reindex(self.groups))
        return gpd.GeoDataFrame(df, geometry=geoms, crs=self.crs)

    # -- construction ------------------------------------------------------------------
    @classmethod
    def _from_records(
        cls,
        axis_values: list[np.ndarray],
        labels: np.ndarray,
        counts: np.ndarray,
        cell_sizes: Sequence[float],
        cell_origins: Sequence[float] | None,
        *,
        binarise_counts: bool = False,
        allow_empty_groups: bool = False,
        allow_empty_labels: bool = False,
        crs: Any = None,
        group_names: np.ndarray | None = None,
    ) -> BaseData:
        """Build from per-record group axis values, label names and counts.

        ``axis_values[k]`` holds floats for axes with cell size >= 0 (already
        numeric) or strings for text axes; ``labels`` is an object array of
        names (None for group-only records); counts are summed per
        (group, label) in record order, like Biodiverse's collation.
        """
        n_axes = len(cell_sizes)
        if cell_origins is None:
            cell_origins = [0.0] * n_axes
        n = len(labels)
        # group name parts per axis
        parts: list[np.ndarray] = []
        coord_cols: list[np.ndarray] = []
        text_axes: list[int] = []
        for k in range(n_axes):
            size, origin = float(cell_sizes[k]), float(cell_origins[k])
            vals = axis_values[k]
            if size > 0:
                snapped = snap_coordinates(np.asarray(vals, dtype=np.float64), size, origin)
                uniq, inv = np.unique(snapped, return_inverse=True)
                names_u = np.array([perl_number(u) for u in uniq], dtype=object)
                parts.append(names_u[inv])
                coord_cols.append(np.array([float(s) for s in names_u], dtype=np.float64)[inv])
            elif size == 0:
                # values kept as given: strings keep their text, numbers print as Perl does
                if vals.dtype == object:
                    strs = np.array([v if isinstance(v, str) else perl_number(v) for v in vals], dtype=object)
                else:
                    uniq, inv = np.unique(np.asarray(vals, dtype=np.float64), return_inverse=True)
                    strs = np.array([perl_number(u) for u in uniq], dtype=object)[inv]
                parts.append(strs)
                coord_cols.append(np.array([float(s) for s in strs], dtype=np.float64))
            else:
                strs = np.array(["" if v is None else str(v) for v in vals], dtype=object)
                parts.append(strs)
                coord_cols.append(np.zeros(n))
                text_axes.append(k)
        if group_names is None:
            if n_axes == 1:
                gnames = np.array([dequote_element(csv_field(p)) for p in parts[0]], dtype=object)
            else:
                quoted = [np.array([csv_field(p) for p in col], dtype=object) for col in parts]
                gnames = quoted[0].astype(object)
                for col in quoted[1:]:
                    gnames = gnames + ":" + col
        else:
            gnames = np.asarray(group_names, dtype=object)

        counts = np.asarray(counts, dtype=np.float64)
        g_uniq, g_inv = np.unique(gnames.astype(str), return_inverse=True) if n else (np.array([], dtype=str), np.zeros(0, int))
        has_label = np.array([lb is not None for lb in labels], dtype=bool)
        lab_str = np.array([lb if lb is not None else "" for lb in labels], dtype=object)
        l_uniq, l_inv = np.unique(lab_str.astype(str), return_inverse=True) if n else (np.array([], dtype=str), np.zeros(0, int))

        # collate: sum in record order
        key = g_inv.astype(np.int64) * max(len(l_uniq), 1) + l_inv
        sel = has_label
        k_uniq, k_inv = np.unique(key[sel], return_inverse=True)
        sums = np.zeros(len(k_uniq))
        np.add.at(sums, k_inv, counts[sel])
        if binarise_counts:
            sums = (sums != 0).astype(np.float64)
        rows = k_uniq // max(len(l_uniq), 1)
        cols = k_uniq % max(len(l_uniq), 1)
        keep = sums != 0

        group_used = np.zeros(len(g_uniq), bool)
        group_used[rows[keep]] = True
        label_used = np.zeros(len(l_uniq), bool)
        label_used[cols[keep]] = True
        if allow_empty_groups:
            group_used[g_inv] = True
        if allow_empty_labels:
            label_used[l_inv[has_label]] = True

        g_map = -np.ones(len(g_uniq), np.int64)
        g_map[group_used] = np.arange(group_used.sum())
        l_map = -np.ones(len(l_uniq), np.int64)
        l_map[label_used] = np.arange(label_used.sum())
        mat = sp.csr_matrix(
            (sums[keep], (g_map[rows[keep]], l_map[cols[keep]])),
            shape=(int(group_used.sum()), int(label_used.sum())),
        )
        # coordinates of each used group (first record)
        first = np.full(len(g_uniq), -1, np.int64)
        first[g_inv[::-1]] = np.arange(n)[::-1]
        idx = first[group_used]
        coords = np.column_stack([c[idx] for c in coord_cols]) if n_axes else np.zeros((len(idx), 0))
        axes = [[parts[k][i] for k in range(n_axes)] for i in idx]
        bd = cls(
            mat,
            [str(g) for g in g_uniq[group_used]],
            [str(lb) for lb in l_uniq[label_used]],
            coords,
            cell_sizes,
            cell_origins,
            crs=crs,
            group_axes=axes,
        )
        if text_axes:
            bd._set_text_axis_coords(text_axes)
        return bd

    def _set_text_axis_coords(self, axes: Iterable[int]) -> None:
        """Rank of each text value, reverse natural order (Biodiverse)."""
        for k in axes:
            vals = [ax[k] for ax in self._group_axes]
            uniq = sorted(set(vals), key=_natural_key, reverse=True)
            rank = {v: i for i, v in enumerate(uniq)}
            self.coords[:, k] = [rank[v] for v in vals]

    @classmethod
    def from_table(
        cls,
        source: str | os.PathLike | pd.DataFrame,
        group_columns: Sequence[int | str],
        label_columns: Sequence[int | str] | None = None,
        sample_count_columns: Sequence[int | str] | None = None,
        cell_sizes: Sequence[float] | None = None,
        cell_origins: Sequence[float] | None = None,
        *,
        sep: str | None = None,
        quote: str | None = None,
        matrix_form: bool = False,
        label_start_col: int | str | None = None,
        label_end_col: int | str | None = None,
        include_columns: Sequence[int | str] | None = None,
        exclude_columns: Sequence[int | str] | None = None,
        binarise_counts: bool = False,
        allow_empty_groups: bool = False,
        allow_empty_labels: bool = False,
        skip_lines_with_undef_groups: bool = True,
        crs: Any = None,
    ) -> BaseData:
        """Delimited text (or a DataFrame) read like Biodiverse's ``import_data``.

        Parameters
        ----------
        source : path or DataFrame
            CSV-like file (separator and quote guessed unless given) or a
            DataFrame (all values are read as text, like the file).
        group_columns, label_columns, sample_count_columns : sequence
            Column positions (0-based, as in Biodiverse) or names. Several
            label columns are joined with ``:``; sample counts are summed
            (1 per record when there are none).
        cell_sizes : sequence of float
            One per group column: > 0 snaps to cells of that size, 0 keeps the
            values, < 0 makes a text axis. Default 0 for every axis.
        cell_origins : sequence of float, optional
            Grid origin per axis (default 0).
        matrix_form : bool
            Site × species layout: the columns ``label_start_col`` to
            ``label_end_col`` are labels, their values the counts.
        include_columns, exclude_columns : sequence, optional
            Keep only records where any include column is true / drop records
            where any exclude column is true (Perl truth: not empty, not 0).
        binarise_counts : bool
            Counts become 1 (after summing per group and label).
        allow_empty_groups, allow_empty_labels : bool
            Keep groups/labels that only appear with zero counts.
        skip_lines_with_undef_groups : bool
            Skip records with an empty or ``NA`` coordinate.
        crs : optional
            CRS of the coordinates.

        Returns
        -------
        BaseData

        Examples
        --------
        >>> import pandas as pd
        >>> import landschaft as ls
        >>> df = pd.DataFrame({"sp": ["a", "b", "a"], "x": [10, 15, 150], "y": [5, 5, 5]})
        >>> bd = ls.bio.BaseData.from_table(df, group_columns=["x", "y"], label_columns=["sp"],
        ...                                 cell_sizes=[100, 100])
        >>> bd.groups
        ['150:50', '50:50']
        >>> bd.labels_in_group("50:50")
        {'a': 1.0, 'b': 1.0}
        """
        if isinstance(source, pd.DataFrame):
            header = [str(c) for c in source.columns]
            body = [[_cell_text(v) for v in row] for row in source.itertuples(index=False, name=None)]
        else:
            rows = _read_rows(source, sep, quote)
            if not rows:
                raise ValueError("empty table")
            header = [h if h is not None else "" for h in rows[0]]
            body = rows[1:]

        def col(c: int | str) -> int:
            if isinstance(c, (int, np.integer)):
                return int(c)
            try:
                return header.index(str(c))
            except ValueError:
                raise KeyError(f"column {c!r} not found") from None

        gcols = [col(c) for c in group_columns]
        n_axes = len(gcols)
        if cell_sizes is None:
            cell_sizes = [0.0] * n_axes
        if len(cell_sizes) != n_axes:
            raise ValueError(f"Number of group columns differs from cellsizes ({n_axes} != {len(cell_sizes)})")
        lcols = [col(c) for c in (label_columns or [])]
        scols = [col(c) for c in (sample_count_columns or [])]
        icols = [col(c) for c in (include_columns or [])]
        ecols = [col(c) for c in (exclude_columns or [])]

        mx_cols: list[tuple[str, int]] = []
        if matrix_form:
            start = col(label_start_col) if label_start_col is not None else 0
            end = col(label_end_col) if label_end_col is not None else len(header) - 1
            names: dict[str, int] = {}
            for i in range(start, end + 1):
                names[dequote_element(csv_field(header[i]))] = i
            mx_cols = list(names.items())

        axis_vals: list[list[Any]] = [[] for _ in range(n_axes)]
        labels: list[str | None] = []
        counts: list[float] = []

        def get(row: list, i: int):
            return row[i] if i < len(row) else None

        for row in body:
            if not row:
                continue
            if ecols and any(_perl_true(get(row, c)) for c in ecols):
                continue
            if icols and not any(_perl_true(get(row, c)) for c in icols):
                continue
            gvals = []
            skip = False
            for k, c in enumerate(gcols):
                v = get(row, c)
                size = float(cell_sizes[k])
                if size >= 0:
                    if skip_lines_with_undef_groups and (v is None or v == "NA" or v == ""):
                        skip = True
                        break
                    if v is None or not looks_like_number(v):
                        raise ValueError(f"Non-numeric group field in column {c} ({v})")
                    gvals.append(float(v) if size > 0 else v)
                else:
                    gvals.append("" if v is None else v)
            if skip:
                continue

            if matrix_form:
                elements = [(name, get(row, i)) for name, i in mx_cols]
            else:
                lab = dequote_element(":".join(csv_field(get(row, c) or "") for c in lcols)) if len(lcols) <= 1 else ":".join(
                    csv_field(get(row, c) or "") for c in lcols
                )
                if scols:
                    # an empty count reads as 0 (Biodiverse: `$fields->[$c] // 0`)
                    total = None
                    for c in scols:
                        v = get(row, c)
                        v = "0" if v is None else v
                        if not looks_like_number(v):
                            raise ValueError(f"Field {c} does not look like a number")
                        total = float(v) if total is None else total + float(v)
                    elements = [(lab, total)]
                else:
                    elements = [(lab, 1.0)]

            for name, cnt in elements:
                if cnt is None:
                    if matrix_form:
                        continue
                    cnt = 1.0
                if isinstance(cnt, str):
                    if cnt == "NA":
                        continue
                    if matrix_form and cnt == "":
                        continue
                    if not looks_like_number(cnt):
                        raise ValueError(f"count {cnt!r} does not look like a number")
                    cnt = float(cnt)
                if cnt == 0 and not allow_empty_groups:
                    continue
                for k in range(n_axes):
                    axis_vals[k].append(gvals[k])
                labels.append(name)
                counts.append(cnt)

        arrays = []
        for k in range(n_axes):
            size = float(cell_sizes[k])
            vals = axis_vals[k]
            if size > 0:
                arrays.append(np.array(vals, dtype=np.float64))
            else:
                arrays.append(np.array(vals, dtype=object))
        return cls._from_records(
            arrays,
            np.array(labels, dtype=object),
            np.array(counts, dtype=np.float64),
            cell_sizes,
            cell_origins,
            binarise_counts=binarise_counts,
            allow_empty_groups=allow_empty_groups,
            allow_empty_labels=allow_empty_labels,
            crs=crs,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        group_columns: Sequence[Hashable],
        label_column: Hashable | Sequence[Hashable] | None = None,
        count_column: Hashable | None = None,
        cell_sizes: Sequence[float] | None = None,
        cell_origins: Sequence[float] | None = None,
        *,
        label_columns: Sequence[Hashable] | None = None,
        binarise_counts: bool = False,
        crs: Any = None,
    ) -> BaseData:
        """Long table (one row per record) or wide table (labels as columns).

        Parameters
        ----------
        df : DataFrame
            Numeric coordinates in ``group_columns``.
        group_columns : sequence
            Coordinate columns (one axis each).
        label_column : column name or list of names, optional
            Long format: label of each record (several columns are joined with
            ``:``).
        count_column : column name, optional
            Long format: sample count of each record (default 1).
        label_columns : sequence, optional
            Wide format: these columns are labels and their values the counts
            (zeros and missing values are skipped).
        cell_sizes, cell_origins, binarise_counts, crs
            As in :meth:`from_table`.

        Examples
        --------
        >>> import pandas as pd
        >>> import landschaft as ls
        >>> wide = pd.DataFrame({"x": [0.5, 1.5], "y": [0.5, 0.5], "sp1": [2, 0], "sp2": [1, 3]})
        >>> bd = ls.bio.BaseData.from_dataframe(wide, ["x", "y"], label_columns=["sp1", "sp2"],
        ...                                     cell_sizes=[1, 1])
        >>> bd.to_dataframe()
                 sp1  sp2
        group
        0.5:0.5  2.0  1.0
        1.5:0.5  0.0  3.0
        """
        gcols = list(group_columns)
        n_axes = len(gcols)
        if cell_sizes is None:
            cell_sizes = [0.0] * n_axes
        if label_columns is not None:
            long = df.melt(id_vars=gcols, value_vars=list(label_columns), var_name="__label", value_name="__count")
            # melt goes column by column; keep record order (row, then column)
            long["__row"] = np.tile(np.arange(len(df)), len(list(label_columns)))
            long["__col"] = np.repeat(np.arange(len(list(label_columns))), len(df))
            long = long.sort_values(["__row", "__col"], kind="stable")
            long = long[long["__count"].notna() & (long["__count"] != 0)]
            labels = long["__label"].astype(str).to_numpy(dtype=object)
            counts = long["__count"].to_numpy(dtype=np.float64)
            frame = long
        else:
            if label_column is None:
                raise ValueError("label_column (long format) or label_columns (wide format) is needed")
            lc = [label_column] if not isinstance(label_column, (list, tuple)) else list(label_column)
            if len(lc) == 1:
                labels = np.array([dequote_element(csv_field(_cell_text(v) or "")) for v in df[lc[0]]], dtype=object)
            else:
                labels = np.array(
                    [":".join(csv_field(_cell_text(v) or "") for v in row) for row in df[lc].itertuples(index=False, name=None)],
                    dtype=object,
                )
            counts = np.ones(len(df)) if count_column is None else df[count_column].to_numpy(dtype=np.float64)
            keep = counts != 0
            labels, counts, frame = labels[keep], counts[keep], df[keep]
        arrays = []
        for k, c in enumerate(gcols):
            if float(cell_sizes[k]) < 0:
                arrays.append(np.array([_cell_text(v) for v in frame[c]], dtype=object))
            else:
                arrays.append(frame[c].to_numpy(dtype=np.float64))
        return cls._from_records(arrays, labels, counts, cell_sizes, cell_origins, binarise_counts=binarise_counts, crs=crs)

    @classmethod
    def from_sparse(
        cls,
        counts: sp.spmatrix | np.ndarray,
        coords: np.ndarray,
        labels: Sequence[str],
        cell_sizes: Sequence[float] | None = None,
        cell_origins: Sequence[float] | None = None,
        *,
        crs: Any = None,
    ) -> BaseData:
        """Community matrix (sites × species) with site coordinates.

        Sites falling in the same cell are merged (counts summed).

        Examples
        --------
        >>> import numpy as np
        >>> import landschaft as ls
        >>> m = np.array([[1, 0], [2, 5]])
        >>> bd = ls.bio.BaseData.from_sparse(m, [[0.5, 0.5], [0.7, 0.2]], ["a", "b"], cell_sizes=[1, 1])
        >>> bd.labels_in_group("0.5:0.5")
        {'a': 3.0, 'b': 5.0}
        """
        coo = sp.coo_matrix(counts)
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim == 1:
            coords = coords[:, None]
        n_axes = coords.shape[1]
        if cell_sizes is None:
            cell_sizes = [0.0] * n_axes
        order = np.lexsort((coo.col, coo.row))
        r, c, v = coo.row[order], coo.col[order], coo.data[order]
        labels_arr = np.asarray([str(lb) for lb in labels], dtype=object)
        return cls._from_records(
            [coords[r, k] for k in range(n_axes)], labels_arr[c], v.astype(np.float64), cell_sizes, cell_origins, crs=crs
        )

    @classmethod
    def from_xarray(
        cls,
        data,
        label_dim: str | None = None,
        cell_sizes: Sequence[float] | None = None,
        cell_origins: Sequence[float] | None = None,
        *,
        threshold: float | None = None,
        x: str = "x",
        y: str = "y",
    ) -> BaseData:
        """Stack of per-label grids (a DataArray with a label dimension, or a Dataset).

        Each slice (or data variable) is a label and its cell values the counts
        (Biodiverse's "labels as bands"); with ``threshold`` values >= it count
        1 (presence), smaller values 0. NaN and nodata are skipped. Cells are
        aggregated to ``cell_sizes`` (default: the grid resolution).

        Examples
        --------
        >>> import numpy as np, xarray as xr
        >>> import landschaft as ls
        >>> da = xr.DataArray(np.array([[[0.9, 0.1]], [[0.2, 0.8]]]), dims=("species", "y", "x"),
        ...                   coords={"species": ["a", "b"], "y": [0.5], "x": [0.5, 1.5]})
        >>> bd = ls.bio.BaseData.from_xarray(da, "species", threshold=0.5)
        >>> bd.to_dataframe()
                 a    b
        group
        0.5:0.5  1.0  0.0
        1.5:0.5  0.0  1.0
        """
        import xarray as xr

        if isinstance(data, xr.Dataset):
            data = data.to_dataarray(dim="__label")
            label_dim = "__label"
        if label_dim is None:
            others = [d for d in data.dims if d not in (x, y)]
            if len(others) != 1:
                raise ValueError("label_dim is needed")
            label_dim = others[0]
        data = data.transpose(label_dim, y, x)
        xs = np.asarray(data[x].values, dtype=np.float64)
        ys = np.asarray(data[y].values, dtype=np.float64)
        if cell_sizes is None:
            dx = abs(xs[1] - xs[0]) if len(xs) > 1 else 1.0
            dy = abs(ys[1] - ys[0]) if len(ys) > 1 else dx
            cell_sizes = [dx, dy]
        nodata = data.attrs.get("_FillValue", data.attrs.get("nodata"))
        axis_x, axis_y, labels, counts = [], [], [], []
        xx, yy = np.meshgrid(xs, ys)
        for k, lab in enumerate(data[label_dim].values):
            z = np.asarray(data.isel({label_dim: k}).values, dtype=np.float64)
            ok = np.isfinite(z)
            if nodata is not None:
                ok &= z != nodata
            if threshold is not None:
                z = np.where(z >= threshold, 1.0, 0.0)
            ok &= z != 0
            axis_x.append(xx[ok])
            axis_y.append(yy[ok])
            labels.append(np.full(int(ok.sum()), str(lab), dtype=object))
            counts.append(z[ok])
        crs = None
        try:
            crs = data.rio.crs
        except Exception:
            pass
        return cls._from_records(
            [np.concatenate(axis_x), np.concatenate(axis_y)],
            np.concatenate(labels) if labels else np.array([], dtype=object),
            np.concatenate(counts) if counts else np.array([]),
            cell_sizes,
            cell_origins,
            crs=crs,
        )

    @classmethod
    def from_raster(
        cls,
        files: str | os.PathLike | Sequence[str | os.PathLike],
        labels_as_bands: bool = True,
        cell_sizes: Sequence[float] | None = None,
        cell_origins: Sequence[float] | None = None,
        *,
        given_label: str | None = None,
        strip_file_extensions: bool = True,
        threshold: float | None = None,
    ) -> BaseData:
        """Raster files read like Biodiverse's ``import_data_raster``.

        Parameters
        ----------
        files : path or list of paths
            Any GDAL raster.
        labels_as_bands : bool
            True: each band is a label (its description, else the file name
            for single-band files, else ``band<i>``) and cell values are
            summed as counts per group. False: cell values are labels and
            each cell counts 1.
        cell_sizes, cell_origins : optional
            Group cells (default: the raster resolution, origin 0).
        given_label : str, optional
            Label for every band.
        strip_file_extensions : bool
            Drop the extension of file names used as labels.
        threshold : float, optional
            With ``labels_as_bands``: values >= threshold count 1, others 0.
        """
        import rasterio

        if isinstance(files, (str, os.PathLike)):
            files = [files]
        gx, gy, labs, cnts = [], [], [], []
        crs = None
        sizes = list(cell_sizes) if cell_sizes is not None else None
        origins = list(cell_origins) if cell_origins is not None else [0.0, 0.0]
        for f in files:
            with rasterio.open(f) as src:
                t = src.transform
                if crs is None and src.crs is not None:
                    crs = src.crs
                if sizes is None:
                    sizes = [abs(t.a), abs(t.e)]
                cs_e, cs_n = sizes
                ox, oy = origins
                cols = np.arange(src.width) + 0.5
                rows = np.arange(src.height) + 0.5
                for b in range(1, src.count + 1):
                    z = src.read(b).astype(np.float64)
                    nd = src.nodatavals[b - 1]
                    ok = np.isfinite(z)
                    if nd is not None and not np.isnan(nd):
                        ok &= z != nd
                    xg = t.c + cols[None, :] * t.a + rows[:, None] * t.b
                    yg = t.f + cols[None, :] * t.d + rows[:, None] * t.e
                    xg = np.broadcast_to(xg, z.shape)
                    yg = np.broadcast_to(yg, z.shape)
                    xs = np.floor((xg[ok] - ox) / cs_e) * cs_e + ox + 0.5 * cs_e
                    ys = np.floor((yg[ok] - oy) / cs_n) * cs_n + oy + 0.5 * cs_n
                    vals = z[ok]
                    if labels_as_bands:
                        if given_label is not None:
                            lab = given_label
                        else:
                            lab = src.descriptions[b - 1] or ""
                            if not lab.strip():
                                if src.count == 1:
                                    lab = Path(f).name
                                    if strip_file_extensions:
                                        lab = re.sub(r"\.\w+$", "", lab)
                                else:
                                    lab = f"band{b}"
                        lab = dequote_element(lab)
                        if threshold is not None:
                            vals = np.where(vals >= threshold, 1.0, 0.0)
                        nz = vals != 0
                        gx.append(xs[nz])
                        gy.append(ys[nz])
                        labs.append(np.full(int(nz.sum()), lab, dtype=object))
                        cnts.append(vals[nz])
                    else:
                        gx.append(xs)
                        gy.append(ys)
                        labs.append(np.array([perl_number(v) for v in vals], dtype=object))
                        cnts.append(np.ones(len(vals)))
        # groups are the snapped centres named as Perl prints them (no extra rounding)
        x = np.concatenate(gx) if gx else np.zeros(0)
        y = np.concatenate(gy) if gy else np.zeros(0)
        names = np.array([f"{perl_number(a)}:{perl_number(b)}" for a, b in zip(x, y)], dtype=object)
        bd = cls._from_records(
            [np.array([perl_number(a) for a in x], dtype=object), np.array([perl_number(b) for b in y], dtype=object)],
            np.concatenate(labs) if labs else np.array([], dtype=object),
            np.concatenate(cnts) if cnts else np.array([]),
            [0.0, 0.0],
            origins,
            crs=crs,
            group_names=names,
        )
        bd.cell_sizes = (float(sizes[0]), float(sizes[1])) if sizes else (0.0, 0.0)
        return bd

    @classmethod
    def from_vector(
        cls,
        source,
        label_columns: Hashable | Sequence[Hashable],
        cell_sizes: Sequence[float],
        cell_origins: Sequence[float] | None = None,
        *,
        count_columns: Sequence[Hashable] | None = None,
        measure: str | None = None,
        layer: str | None = None,
        to_crs: Any = None,
        binarise_counts: bool = False,
    ) -> BaseData:
        """Vector features (points, lines, polygons) read like Biodiverse's feature import.

        Points fall in the cell containing them. Lines and polygons are laid
        over a grid of cells spanning the layer: by default every cell they
        touch counts 1; ``measure="area"`` (polygons) or ``measure="length"``
        (lines) counts the area or length of the part inside each cell instead.

        Parameters
        ----------
        source : GeoDataFrame or path
            Read with geopandas (``.parquet`` as GeoParquet; GeoPackage,
            shapefile, GeoJSON...).
        label_columns : column name or list
            Label of each feature (several columns joined with ``:``); names
            match case-insensitively.
        cell_sizes : sequence of 2 floats
            Group cells in the units of the (possibly reprojected) CRS; 0 keeps
            point coordinates as they are.
        cell_origins : sequence of 2 floats, optional
            Grid origin (default 0).
        count_columns : list of column names, optional
            Sample count of each feature (summed); default 1 per point/cell,
            or the measured area/length.
        measure : {None, "area", "length"}
            See above.
        layer : str, optional
            Layer of a multi-layer file.
        to_crs : optional
            Reproject first.
        binarise_counts : bool
            Counts become 1.

        Examples
        --------
        >>> import geopandas as gpd
        >>> from shapely.geometry import box
        >>> import landschaft as ls
        >>> gdf = gpd.GeoDataFrame({"sp": ["a"]}, geometry=[box(0, 0, 15, 5)])
        >>> ls.bio.BaseData.from_vector(gdf, "sp", cell_sizes=[10, 10]).groups
        ['15:5', '5:5']
        >>> ls.bio.BaseData.from_vector(gdf, "sp", [10, 10], measure="area").labels_in_group("15:5")
        {'a': 25.0}
        """
        import geopandas as gpd
        import shapely

        if isinstance(source, gpd.GeoDataFrame):
            gdf = source
        elif str(source).lower().endswith((".parquet", ".geoparquet")):
            gdf = gpd.read_parquet(source)
        else:
            gdf = gpd.read_file(source, layer=layer)
        if to_crs is not None:
            gdf = gdf.to_crs(to_crs)
        if len(cell_sizes) != 2:
            raise ValueError("feature imports need two cell sizes (x and y)")
        sx, sy = (float(c) for c in cell_sizes)
        ox, oy = (0.0, 0.0) if cell_origins is None else (float(c) for c in cell_origins)
        lab_cols = [label_columns] if not isinstance(label_columns, (list, tuple)) else list(label_columns)
        lower = {str(c).lower(): c for c in gdf.columns}

        def colname(c):
            return c if c in gdf.columns else lower[str(c).lower()]

        lab_cols = [colname(c) for c in lab_cols]
        cnt_cols = [colname(c) for c in (count_columns or [])]
        labels_all = []
        for row in gdf[lab_cols].itertuples(index=False, name=None):
            name = ":".join(csv_field(_cell_text(v) or "") for v in row)
            labels_all.append(dequote_element(name) if len(lab_cols) <= 1 else name)
        counts_all = None
        if cnt_cols:
            vals = gdf[cnt_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            counts_all = vals[:, 0].copy()
            for j in range(1, vals.shape[1]):
                counts_all = counts_all + vals[:, j]

        geoms = gdf.geometry.to_numpy()
        gtypes = shapely.get_type_id(geoms)
        xs: list[float] = []
        ys: list[float] = []
        labs: list[str] = []
        cnts: list[float] = []
        xmin, ymin, xmax, ymax = gdf.total_bounds
        if sx > 0 and sy > 0:
            # the fishnet spans the layer extent snapped to multiples of the cell size
            gx0, gy0 = np.floor(xmin / sx) * sx, np.floor(ymin / sy) * sy
            gx1, gy1 = np.ceil(xmax / sx) * sx, np.ceil(ymax / sy) * sy
            cols = int(np.ceil((gx1 - gx0) / sx))
            rows = int(np.ceil((gy1 - gy0) / sy))
            # cell edges as Biodiverse accumulates them (repeated additions)
            west_all = np.empty(cols)
            acc = gx0
            for c in range(cols):
                west_all[c] = acc
                acc += sx
            east_all = np.empty(cols)
            acc = gx0 + sx
            for c in range(cols):
                east_all[c] = acc
                acc += sx
            north_all = np.empty(rows)
            south_all = np.empty(rows)
            acc_n, acc_s = gy1, gy1 - sy
            for r in range(rows):
                north_all[r] = acc_n
                south_all[r] = acc_s
                acc_n -= sy
                acc_s -= sy
        bt = 10e-14 * (sx + sy) / 2

        for k, geom in enumerate(geoms):
            if geom is None or geom.is_empty:
                continue
            base = 1.0 if counts_all is None else float(counts_all[k])
            if gtypes[k] in (0, 4):  # (multi)point
                for px, py in shapely.get_coordinates(geom):
                    xs.append(px)
                    ys.append(py)
                    labs.append(labels_all[k])
                    cnts.append(base)
                continue
            if not (sx > 0 and sy > 0):
                raise ValueError("lines and polygons need positive cell sizes")
            bx0, by0, bx1, by1 = geom.bounds
            c_lo = max(int(np.floor((bx0 - gx0) / sx)) - 1, 0)
            c_hi = min(int(np.ceil((bx1 - gx0) / sx)) + 1, cols)
            r_lo = max(int(np.floor((gy1 - by1) / sy)) - 1, 0)
            r_hi = min(int(np.ceil((gy1 - by0) / sy)) + 1, rows)
            ci, ri = np.meshgrid(np.arange(c_lo, c_hi), np.arange(r_lo, r_hi), indexing="ij")
            ci, ri = ci.ravel(), ri.ravel()
            if measure is None:
                w = west_all[ci] + bt
                e = east_all[ci] - bt
                n = north_all[ri] - bt
                so = south_all[ri] + bt
                boxes = shapely.box(w, so, e, n)
                hit = shapely.intersects(boxes, geom)
                for poly in boxes[hit]:
                    c = poly.centroid
                    xs.append(c.x)
                    ys.append(c.y)
                    labs.append(labels_all[k])
                    cnts.append(base)
            else:
                boxes = shapely.box(west_all[ci], south_all[ri], east_all[ci], north_all[ri])
                hit = shapely.intersects(boxes, geom)
                pieces = shapely.intersection(boxes[hit], geom)
                keep_dim = 2 if measure == "area" else 1
                for piece in pieces:
                    if piece.is_empty or shapely.get_dimensions(piece) < keep_dim:
                        continue
                    c = piece.centroid
                    size = piece.area if measure == "area" else piece.length
                    xs.append(c.x)
                    ys.append(c.y)
                    labs.append(labels_all[k])
                    cnts.append(size if counts_all is None else base)

        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        # the feature import snaps with a plain floor (no rounding step)
        if sx > 0:
            x = ox + np.floor((x - ox) / sx) * sx + sx / 2
        if sy > 0:
            y = oy + np.floor((y - oy) / sy) * sy + sy / 2
        names = np.array([f"{perl_number(a)}:{perl_number(b)}" for a, b in zip(x, y)], dtype=object)
        bd = cls._from_records(
            [np.array([perl_number(a) for a in x], dtype=object), np.array([perl_number(b) for b in y], dtype=object)],
            np.asarray(labs, dtype=object),
            np.asarray(cnts, dtype=np.float64),
            [0.0, 0.0],
            [ox, oy],
            crs=gdf.crs,
            group_names=names,
            binarise_counts=binarise_counts,
        )
        bd.cell_sizes = (sx, sy)
        return bd

    @classmethod
    def from_points(
        cls,
        source,
        label_columns: Hashable | Sequence[Hashable],
        cell_sizes: Sequence[float],
        cell_origins: Sequence[float] | None = None,
        *,
        count_column: Hashable | None = None,
        layer: str | None = None,
        to_crs: Any = None,
    ) -> BaseData:
        """Point records (GeoDataFrame, GeoParquet, GeoPackage, shapefile...).

        :meth:`from_vector` for points; ``count_column`` gives the sample count
        of each point (default 1).
        """
        return cls.from_vector(
            source, label_columns, cell_sizes, cell_origins,
            count_columns=None if count_column is None else [count_column], layer=layer, to_crs=to_crs,
        )

    @classmethod
    def from_file(cls, path: str | os.PathLike, **kwargs) -> BaseData:
        """Dispatch on the extension: text tables, rasters, NetCDF/Zarr, vector points."""
        p = str(path).lower()
        if p.endswith((".csv", ".txt", ".tsv")):
            return cls.from_table(path, **kwargs)
        if p.endswith((".tif", ".tiff", ".asc", ".img", ".vrt", ".jp2")):
            return cls.from_raster(path, **kwargs)
        if p.endswith((".nc", ".zarr")):
            import xarray as xr

            ds = xr.open_zarr(path) if p.endswith(".zarr") else xr.open_dataset(path)
            return cls.from_xarray(ds, **kwargs)
        return cls.from_vector(path, **kwargs)


def _cell_text(v: Any) -> str | None:
    """Value of a DataFrame cell as the text Biodiverse would read."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (float, np.floating)):
        return perl_number(float(v))
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    return str(v)
