# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial analysis: Biodiverse's indices over the neighbourhoods of groups."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .. import _core
from .._threads import resolve
from ._conditions import SpatialCondition, neighbours
from ._names import perl_number

__all__ = ["SpatialResult", "list_calculations", "spatial_analysis"]

_KEY_LABEL, _KEY_GROUP, _KEY_TEXT, _KEY_NUMBER, _KEY_GROUP_ARRAY, _KEY_NODE = range(6)
_DEPS = None


def _dependencies() -> dict:
    """Biodiverse's pre_calc dependencies of every calculation (dumped from Biodiverse)."""
    global _DEPS
    if _DEPS is None:
        import json
        from pathlib import Path

        _DEPS = json.loads((Path(__file__).parent / "data" / "biodiverse_dependencies.json").read_text())
    return _DEPS


def _precalc_order(calcs: list[str]) -> list[str]:
    """The order Biodiverse runs pre_calcs in (parse_dependencies_for_calc + aggregation)."""
    import re

    deps = _dependencies()
    agg: list[str] = []
    for calc in sorted(calcs):
        lst = [calc]
        for c in lst:  # grows while iterating, as in Perl
            lst.extend(deps.get(c, {}).get("pre_calc", []) or [])
        seen = set()
        rev = []
        for c in reversed(lst):
            if c not in seen:
                seen.add(c)
                rev.append(c)
        agg.extend(rev)
    seen = set()
    u = []
    for c in agg:
        if c not in seen:
            seen.add(c)
            u.append(c)
    if u:
        if not any(re.match(r"^calc_abc[23]?", c) for c in u) and "_calc_abc_any" in u:
            i = u.index("_calc_abc_any")
            if i > 0:
                u.insert(0, u.pop(i))
        i23 = next((i for i, c in enumerate(u) if re.match(r"^calc_abc[23]", c)), -1)
        if i23 > 0:
            u.insert(0, u.pop(i23))
        elif i23 < 0 and "calc_abc" in u and u.index("calc_abc") > 0:
            u.insert(0, u.pop(u.index("calc_abc")))
    return u


def _abc_any_variant(calcs: list[str]) -> int:
    """Label counts `_calc_abc_any` passes on: 1 presence, 2 groups, 3 samples."""
    order = _precalc_order(calcs)
    if "_calc_abc_any" not in order:
        return 1
    before = order[: order.index("_calc_abc_any")]
    if "calc_abc" in before:
        return 1
    if "calc_abc2" in before:
        return 2
    if "calc_abc3" in before:
        return 3
    return 1


def _registry() -> dict[str, dict]:
    out = {}
    for name, scalars, lists, nbr, proc, tree in _core.bio_calculations():
        out[name] = {"scalars": list(scalars), "lists": [n for n, _ in lists], "uses_nbr_lists": int(nbr),
                     "uses_processing_element": bool(proc), "uses_tree": bool(tree)}
    return out


def list_calculations(n_neighbour_sets: int | None = None) -> pd.DataFrame:
    """The calculations available to :func:`spatial_analysis`.

    Parameters
    ----------
    n_neighbour_sets : int, optional
        Only those valid with this many neighbour sets (some compare two sets).

    Returns
    -------
    pandas.DataFrame
        One row per calculation: its indices (scalar and list) and the number
        of neighbour sets it needs.

    Examples
    --------
    >>> import landschaft as ls
    >>> calcs = ls.bio.list_calculations()
    >>> calcs.loc["endemism_whole", "indices"]
    ['ENDW_CWE', 'ENDW_RICHNESS', 'ENDW_SINGLE', 'ENDW_WE']
    """
    rows = []
    for name, info in _registry().items():
        if n_neighbour_sets is not None and info["uses_nbr_lists"] > n_neighbour_sets:
            continue
        rows.append({
            "calculation": name.removeprefix("calc_"),
            "indices": info["scalars"],
            "list_indices": info["lists"],
            "neighbour_sets": info["uses_nbr_lists"],
        })
    return pd.DataFrame(rows).set_index("calculation")


def _basedata_view(bd) -> dict:
    counts = bd.counts
    variety = np.diff(counts.tocsc().indptr).astype(np.float64)
    sample = np.bincount(counts.indices, weights=counts.data, minlength=bd.n_labels).astype(np.float64)
    props = bd.label_properties
    label_range = variety.copy()
    if "RANGE" in props.columns:
        r = pd.to_numeric(props["RANGE"], errors="coerce").reindex(bd.labels).to_numpy(dtype=np.float64)
        label_range = np.where(np.isnan(r), variety, np.maximum(r, variety))
    abundance = sample.copy()
    if "ABUNDANCE" in props.columns:
        a = pd.to_numeric(props["ABUNDANCE"], errors="coerce").reindex(bd.labels).to_numpy(dtype=np.float64)
        abundance = np.where(np.isnan(a), sample, np.maximum(a, sample))
    levels, prefix, prefix_names = _label_hierarchy(bd)
    return {
        "label_sample_count": np.ascontiguousarray(sample),
        "abc_any_variant": 1,
        "n_label_levels": levels,
        "label_prefix": prefix,
        "prefix_names": prefix_names,
        "n_groups": bd.n_groups,
        "n_labels": bd.n_labels,
        "indptr": np.ascontiguousarray(counts.indptr, dtype=np.int64),
        "indices": np.ascontiguousarray(counts.indices, dtype=np.int32),
        "counts": np.ascontiguousarray(counts.data, dtype=np.float64),
        "label_range": np.ascontiguousarray(label_range),
        "label_abundance": np.ascontiguousarray(abundance),
    }


def _label_hierarchy(bd) -> tuple[int, np.ndarray, list[str]]:
    """Prefixes of the label axes (Biodiverse's labels-as-tree), numbered in name order."""
    from ._names import dequote_element, join_element, split_element

    if bd.n_labels == 0:
        return 1, np.zeros(0, np.int32), []
    levels = len(split_element(bd.labels[0]))
    comps = [split_element(lb) for lb in bd.labels]
    names_per = []
    for c in comps:
        row = []
        for i in range(levels):
            part = c[: i + 1] if i < len(c) else c
            row.append(dequote_element(join_element(part)))
        names_per.append(row)
    uniq = sorted({n for row in names_per for n in row})
    idx = {n: k for k, n in enumerate(uniq)}
    prefix = np.array([[idx[n] for n in row] for row in names_per], dtype=np.int32).ravel()
    return levels, prefix, uniq


def _row_bytes(csr, r) -> bytes:
    ip, ix = csr
    return ix[ip[r]:ip[r + 1]].tobytes()


def _unique_rows(set1, set2) -> tuple[np.ndarray, np.ndarray]:
    """Representative rows and, for every row, the index of its representative."""
    n = len(set1[0]) - 1
    seen: dict = {}
    rep = []
    inverse = np.empty(n, np.int64)
    for r in range(n):
        key = (_row_bytes(set1, r), _row_bytes(set2, r)) if set2 is not None else _row_bytes(set1, r)
        k = seen.get(key)
        if k is None:
            k = seen[key] = len(rep)
            rep.append(r)
        inverse[r] = k
    return np.asarray(rep, np.int64), inverse


def _take_rows(csr, rows: np.ndarray):
    ip, ix = csr
    lengths = np.diff(ip)[rows]
    new_ip = np.zeros(len(rows) + 1, np.int64)
    new_ip[1:] = np.cumsum(lengths)
    starts = ip[rows]
    idx = np.repeat(starts - new_ip[:-1], lengths) + np.arange(new_ip[-1])
    return new_ip, np.ascontiguousarray(ix[idx], dtype=np.int32)


def _expand_raw(raw: dict, inverse: np.ndarray) -> dict:
    out = dict(raw)
    out["scalars"] = np.asarray(raw["scalars"])[inverse]
    lists = []
    for name, kind, ip, keys, values in raw["lists"]:
        ip = np.asarray(ip)
        lengths = np.diff(ip)[inverse]
        new_ip = np.zeros(len(inverse) + 1, np.int64)
        new_ip[1:] = np.cumsum(lengths)
        idx = np.repeat(ip[inverse] - new_ip[:-1], lengths) + np.arange(new_ip[-1])
        lists.append((name, kind, new_ip, np.asarray(keys)[idx], np.asarray(values)[idx]))
    out["lists"] = lists
    return out


def _calc_name(c: str) -> str:
    return c if c.startswith("calc_") else "calc_" + c


class ListIndices(dict):
    """List indices of a :class:`SpatialResult`, built as DataFrames on first access.

    Each value is a long table with columns ``group``, ``key`` and ``value``.
    """

    def __init__(self, bd, groups: list[str], raw_lists, texts: list[str], node_names=None) -> None:
        super().__init__()
        self._bd = bd
        self._node_names = node_names or []
        self._groups = np.asarray(groups, dtype=object)
        self._texts = np.asarray(texts, dtype=object)
        self._raw = {name: (kind, np.asarray(ip), np.asarray(k), np.asarray(v)) for name, kind, ip, k, v in raw_lists}
        for name in self._raw:
            super().__setitem__(name, None)

    def __getitem__(self, name: str) -> pd.DataFrame:
        df = super().__getitem__(name)
        if df is None:
            df = self._build(name)
            super().__setitem__(name, df)
        return df

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]

    def get(self, name, default=None):
        return self[name] if name in self else default

    def raw(self, name: str) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """(key kind, indptr, keys, values) as the engine returns them."""
        return self._raw[name]

    def _build(self, name: str) -> pd.DataFrame:
        kind, indptr, keys, values = self._raw[name]
        rows = np.repeat(np.arange(len(self._groups)), np.diff(indptr))
        k = keys.astype(np.int64) if kind != _KEY_NUMBER else keys
        if kind == _KEY_LABEL:
            key_col = np.asarray(self._bd.labels, dtype=object)[k]
        elif kind in (_KEY_GROUP, _KEY_GROUP_ARRAY):
            key_col = np.asarray(self._bd.groups, dtype=object)[k]
        elif kind == _KEY_TEXT:
            key_col = self._texts[k]
        elif kind == _KEY_NODE:
            key_col = np.asarray(self._node_names, dtype=object)[k]
        else:
            uniq, inv = np.unique(keys, return_inverse=True)
            key_col = np.asarray([perl_number(float(u)) for u in uniq], dtype=object)[inv]
        return pd.DataFrame({"group": self._groups[rows], "key": key_col, "value": values.astype(np.float64)})


class SpatialResult:
    """Results of :func:`spatial_analysis`.

    Attributes
    ----------
    table : pandas.DataFrame
        Scalar indices, one row per processed group (Biodiverse's names:
        ``ENDW_WE``, ``SORENSON``...). NaN where Biodiverse gives no value.
    lists : dict of str to pandas.DataFrame
        List indices (``ENDW_WTLIST``, ``EL_LIST_SET1``...) as long tables with
        columns ``group``, ``key``, ``value``.
    neighbourhoods : Neighbourhoods
        The neighbour sets used.
    """

    def __init__(self, bd, table: pd.DataFrame, lists: dict[str, pd.DataFrame], nbrs, calculations: list[str]):
        self.bd = bd
        self.table = table
        self.lists = lists
        self.neighbourhoods = nbrs
        self.calculations = calculations

    def __repr__(self) -> str:
        return (f"SpatialResult({len(self.table)} groups, {self.table.shape[1]} indices, "
                f"{len(self.lists)} list indices)")

    def __getitem__(self, index: str) -> pd.Series:
        return self.table[index]

    def list(self, index: str, group: str) -> dict:
        """One group's entries of a list index (``{key: value}``)."""
        df = self.lists[index]
        sub = df[df["group"] == group]
        return dict(zip(sub["key"], sub["value"]))

    def to_raster(self, index: str, fill: float = np.nan):
        """A scalar index as a grid (see :meth:`BaseData.to_raster`)."""
        return self.bd.to_raster(self.table[index], name=index, fill=fill)

    def to_geodataframe(self, indices: Sequence[str] | None = None):
        """Group cells with their scalar indices (needs geopandas)."""
        cols = list(indices) if indices is not None else list(self.table.columns)
        return self.bd.to_geodataframe(self.table[cols])


def spatial_analysis(
    bd,
    conditions: str | SpatialCondition | Sequence[str | SpatialCondition],
    calculations: str | Sequence[str],
    definition_query: str | SpatialCondition | None = None,
    *,
    tree=None,
    prng_seed: int | None = None,
    n_threads: int | None = None,
    recycle: bool = True,
) -> SpatialResult:
    """Biodiverse's spatial analysis: indices over the neighbourhood of every group.

    Parameters
    ----------
    bd : BaseData
    conditions : condition or list of up to two conditions
        Neighbourhoods (see :func:`neighbours`); indices comparing two sets
        (``sorenson``, ``bray_curtis``...) need two.
    calculations : str or list of str
        Calculation names, with or without Biodiverse's ``calc_`` prefix (see
        :func:`list_calculations`). Calculations that need a second neighbour
        set are skipped when there is only one, as in Biodiverse.
    definition_query : condition, optional
        Only groups passing it are processed.
    tree : Tree, optional
        Phylogeny for the phylogenetic indices (``pd``, ``pe``...); its node
        names are matched to the labels.
    prng_seed : int, optional
        Seed of the Mersenne Twister Biodiverse draws random sets of tips
        with, for the expected NTI on trees that are not ultrametric and for
        the net VPD (Biodiverse's ``prng_seed``; the same seed gives
        Biodiverse's values). Random when not given.
    n_threads : int, optional
        Threads (default: all cores but one).
    recycle : bool
        Reuse neighbour lists as Biodiverse does (see :func:`neighbours`).

    Returns
    -------
    SpatialResult

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> df = pd.DataFrame({"sp": ["a", "b", "a", "c"], "x": [0.5, 0.5, 1.5, 2.5], "y": [0.5] * 4})
    >>> bd = ls.bio.BaseData.from_dataframe(df, ["x", "y"], "sp", cell_sizes=[1, 1])
    >>> res = ls.bio.spatial_analysis(bd, "sp_circle (radius => 1)", ["richness", "endemism_whole"])
    >>> res.table[["RICHNESS_ALL", "ENDW_WE"]]
             RICHNESS_ALL  ENDW_WE
    group
    0.5:0.5           2.0      2.0
    1.5:0.5           3.0      3.0
    2.5:0.5           2.0      1.5
    """
    if isinstance(calculations, str):
        calculations = [calculations]
    if isinstance(conditions, (str, SpatialCondition)):
        conditions = [conditions]
    reg = _registry()
    calcs = []
    for c in calculations:
        name = _calc_name(c)
        if name not in reg:
            raise ValueError(f"unknown calculation {c!r} (see ls.bio.list_calculations())")
        if name not in calcs:
            calcs.append(name)
    n_sets = len(conditions)
    valid = [c for c in calcs if reg[c]["uses_nbr_lists"] <= n_sets]
    if tree is None:
        needs = [c for c in valid if reg[c]["uses_tree"]]
        if needs:
            raise ValueError(f"these calculations need a tree: {', '.join(n.removeprefix('calc_') for n in needs)}")
    if not valid:
        raise ValueError("none of the calculations can run with this many neighbour sets")
    threads = resolve(n_threads)
    uses_proc = any(reg[c]["uses_processing_element"] for c in valid)
    nbrs = neighbours(bd, conditions, definition_query, n_threads=threads, recycle=recycle,
                      results_recyclable=None if not uses_proc else False)
    set1 = nbrs.sets[0]
    set2 = nbrs.sets[1] if n_sets > 1 else None
    procs = np.ascontiguousarray(nbrs.groups, dtype=np.int32)
    # PhyloCom: expected values drawn from one random stream, and (ultrametric
    # trees) path lengths cached as groups are processed
    order = _precalc_order(valid)
    pc_steps = [0 if c == "_calc_nri_nti_expected_values" else int(c[-1]) for c in order
                if c == "_calc_nri_nti_expected_values" or c.startswith("calc_phylo_mpd_mntd")]
    pc_vpd = any(c in ("calc_net_vpd", "calc_vpd_expected_values") for c in valid)
    order_sensitive = False
    if pc_steps and tree is not None:
        from ._phylo import tree_view as _tv

        ultrametric, multifurcating = _core.bio_ultrametric(_tv(tree, bd))
        order_sensitive = pc_vpd or (ultrametric and multifurcating)
    if prng_seed is None:
        import secrets

        prng_seed = secrets.randbits(32)
    # Groups with the same neighbour sets get the same results (Biodiverse copies
    # them too when nothing depends on the processing group): compute each once.
    # Not where results depend on the groups processed before, unless Biodiverse
    # copies them too.
    expand = None
    if not uses_proc and len(procs) > 1 and (not order_sensitive or nbrs.results_recyclable):
        rep, expand = _unique_rows(set1, set2)
        if len(rep) < len(procs):
            set1 = _take_rows(set1, rep)
            set2 = _take_rows(set2, rep) if set2 is not None else None
            procs = procs[rep]
        else:
            expand = None
    view = _basedata_view(bd)
    view["abc_any_variant"] = _abc_any_variant(valid)
    tview = None
    node_names = None
    if tree is not None and any(reg[c]["uses_tree"] for c in valid):
        from ._phylo import tree_view

        tview = tree_view(tree, bd)
        node_names = tview["names"]
        if pc_steps:
            tview["pc_steps"] = pc_steps
            tview["pc_vpd"] = pc_vpd
            tview["pc_seed"] = int(prng_seed) % (1 << 64)
    raw = _core.bio_run_indices(view, procs, set1, set2, valid, threads > 1, tview)
    if expand is not None:
        raw = _expand_raw(raw, expand)
    names = [bd.groups[g] for g in nbrs.groups]
    table = pd.DataFrame(np.asarray(raw["scalars"]), index=pd.Index(names, name="group"),
                         columns=list(raw["scalar_names"]))
    lists = ListIndices(bd, names, raw["lists"], list(raw["texts"]), node_names)
    return SpatialResult(bd, table, lists, nbrs, [c.removeprefix("calc_") for c in valid])
