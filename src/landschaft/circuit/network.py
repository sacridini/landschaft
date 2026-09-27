# SPDX-License-Identifier: GPL-3.0-or-later
"""Circuitscape 6.1's network (graph) mode on landschaft's solver: pairwise
and advanced, as ``data_type = network`` runs them."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._backend import core
from .circuitscape import (POLICIES, IncludedPairs, _flag, _open, _prefix, _resolve, _unset, _write_matrix,
                           read_included_pairs)

__all__ = ["NetworkResult", "circuitscape_network_advanced", "circuitscape_network_pairwise", "read_graph"]


# --- reading, as io.jl --------------------------------------------------------------------------


def _read_table(path) -> list[list[float]]:
    rows = []
    with _open(path) as f:
        for line in f:
            toks = line.replace(",", " ").split()
            if toks:
                rows.append([float(t) for t in toks])
    return rows


def _as_table(value, ncol: int, name: str) -> np.ndarray:
    if isinstance(value, (str, os.PathLike)):
        rows = _read_table(value)
        if not rows or any(len(r) != len(rows[0]) for r in rows):
            raise ValueError(f"{value}: expected a table with the same number of columns on every line")
        t = np.array(rows, dtype=np.float64)
    else:
        t = np.array(value, dtype=np.float64)
    t = t.reshape(-1, ncol) if t.size == 0 else t
    if t.ndim != 2 or t.shape[1] != ncol:
        raise ValueError(f"the {name} must have {ncol} columns")
    return t


def read_graph(edges) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Read a Circuitscape edge list, as its ``load_graph``.

    Parameters
    ----------
    edges
        Path of a text file or ``(k, 3)`` array: node, node, value.

    Returns
    -------
    tuple
        ``(from, to, value, shifted)``: node numbers (shifted to start at 1 when
        they start at 0; ``shifted`` says so) and edge values.

    Examples
    --------
    >>> import numpy as np
    >>> from landschaft import circuit
    >>> graph = np.array([[1, 2, 1.0],             # node, node, resistance
    ...                   [2, 3, 1.0],
    ...                   [1, 3, 2.0],
    ...                   [3, 4, 0.5]])
    >>> circuit.read_graph(graph)
    (array([1, 2, 1, 3]), array([2, 3, 3, 4]), array([1. , 1. , 2. , 0.5]), False)
    """
    t = _as_table(edges, 3, "graph")
    if len(t) == 0:
        raise ValueError("the graph has no edges")
    i, j = t[:, 0].astype(np.int64), t[:, 1].astype(np.int64)
    low = min(i.min(), j.min())
    if low > 1:
        raise ValueError(f"the graph numbers its nodes from {low}; Circuitscape counts from 1 (or 0)")
    if low < 0:
        raise ValueError("negative node numbers")
    zero = low == 0
    if zero:
        i, j = i + 1, j + 1
    return i, j, t[:, 2].copy(), bool(zero)


def _read_focal_nodes(points) -> np.ndarray:
    # read_focal_points: every number of the file, column by column; from 0 shifted to 1.
    if isinstance(points, (str, os.PathLike)):
        rows = _read_table(points)
        if rows and all(len(r) == len(rows[0]) for r in rows):
            fp = np.array(rows).ravel(order="F")
        else:
            fp = np.array([v for r in rows for v in r])
    else:
        fp = np.asarray(points, dtype=np.float64).ravel()
    fp = fp.astype(np.int64)
    if len(fp) and fp.min() == 0:
        fp = fp + 1
    return fp


def _read_strengths(value, starts_from_zero: bool) -> np.ndarray:
    # read_point_strengths: shifted when the ids start at 0 or the graph does.
    t = _as_table(value, 2, "node values")
    if len(t) and (t[:, 0].min() == 0 or starts_from_zero):
        t[:, 0] += 1
    return t


# --- the result -------------------------------------------------------------------------------


@dataclass
class NetworkResult:
    """Circuitscape's network outputs, with the node numbers it writes (from 1).

    Attributes
    ----------
    voltages
        Advanced: ``(n, 2)`` node, voltage.
    node_currents, branch_currents
        Advanced: ``(n, 2)`` node, current and ``(k, 3)`` node, node, current
        (branch currents above 1e-6, as Circuitscape writes them).
    resistances
        Pairwise: effective resistances, ids in the first row and column (-1:
        not connected).
    cum_node_currents, cum_branch_currents
        Pairwise with maps: node and branch currents summed over the pairs.
    maps
        Per-pair tables keyed ``("voltages" | "node_currents" |
        "branch_currents", id1, id2)``.
    n_components
        Number of connected components of the graph.

    Examples
    --------
    >>> import numpy as np
    >>> from landschaft import circuit
    >>> graph = np.array([[1, 2, 1.0],             # node, node, resistance
    ...                   [2, 3, 1.0],
    ...                   [1, 3, 2.0],
    ...                   [3, 4, 0.5]])
    >>> result = circuit.circuitscape_network_advanced(graph, [[1, 1.0]], [[4, 0.0]])
    >>> result.n_components
    1
    """

    voltages: np.ndarray | None = None
    node_currents: np.ndarray | None = None
    branch_currents: np.ndarray | None = None
    resistances: np.ndarray | None = None
    cum_node_currents: np.ndarray | None = None
    cum_branch_currents: np.ndarray | None = None
    maps: dict = field(default_factory=dict)
    n_components: int = 0


def _branch_table(r, c, v) -> np.ndarray:
    # write_currents: branches whose current is not within 1e-6 of 0 (1-based nodes).
    r, c, v = np.asarray(r), np.asarray(c), np.asarray(v)
    keep = ~(np.abs(v) <= 1e-6)
    return np.column_stack([r[keep] + 1.0, c[keep] + 1.0, v[keep]])


def _edges(i, j, w, is_resistance: bool):
    w = 1 / w if is_resistance else w
    m = int(max(i.max(), j.max()))
    return (np.ascontiguousarray(i - 1, dtype=np.int32), np.ascontiguousarray(j - 1, dtype=np.int32),
            np.ascontiguousarray(w, dtype=np.float64), m)


# --- advanced -----------------------------------------------------------------------------------


def circuitscape_network_advanced(graph, sources, grounds, *, graph_is_resistance: bool = True,
                                  ground_is_resistance: bool = True, remove_src_or_gnd: str = "keepall",
                                  n_threads: int = 0) -> NetworkResult:
    """Circuitscape's advanced mode on a network (``data_type = network``,
    ``scenario = advanced``).

    ``graph``: edge list (node, node, resistance or, with
    ``graph_is_resistance=False``, conductance), a path or a ``(k, 3)``
    array; nodes numbered from 1, or from 0 (then every number, the sources'
    and grounds' too, is shifted by 1 as Circuitscape.jl does). Duplicate
    edges add up; edges whose conductances add up to 0 are left out.
    ``sources`` and ``grounds``: ``(node, value)`` lists (paths or arrays),
    the current injected and the resistance (or conductance) to ground; a
    ground resistance of 0 is a direct ground. Every connected component with
    a source and a ground is solved exactly (``remove_src_or_gnd`` as in
    :func:`~landschaft.circuit.circuitscape_advanced`).

    Parameters
    ----------
    graph
        Edge list: path or ``(k, 3)`` array of node, node and resistance (or
        conductance).
    sources
        ``(node, current)`` rows: path or array.
    grounds
        ``(node, resistance or conductance to ground)`` rows; 0 resistance is a
        direct ground.
    graph_is_resistance, ground_is_resistance
        The values are resistances (default) or conductances.
    remove_src_or_gnd
        Nodes that are both: ``"keepall"``, ``"rmvsrc"``, ``"rmvgnd"`` or
        ``"rmvall"``.
    n_threads
        Threads (0 or ``None``: all cores but one).

    Returns
    -------
    NetworkResult
        ``voltages`` (every node; 0 where not solved), ``node_currents`` and
        ``branch_currents`` of the whole graph.

    Examples
    --------
    One amp into node 1, node 4 grounded:

    >>> import numpy as np
    >>> from landschaft import circuit
    >>> graph = np.array([[1, 2, 1.0],             # node, node, resistance
    ...                   [2, 3, 1.0],
    ...                   [1, 3, 2.0],
    ...                   [3, 4, 0.5]])
    >>> result = circuit.circuitscape_network_advanced(graph, [[1, 1.0]], [[4, 0.0]])
    >>> result.voltages
    array([[1. , 1.5],
           [2. , 1. ],
           [3. , 0.5],
           [4. , 0. ]])
    >>> result.branch_currents
    array([[1. , 2. , 0.5],
           [1. , 3. , 0.5],
           [2. , 3. , 0.5],
           [3. , 4. , 1. ]])
    """
    if remove_src_or_gnd not in POLICIES:
        raise ValueError(f"remove_src_or_gnd must be one of {POLICIES}")
    i, j, w, zero = read_graph(graph)
    a, b, w, m = _edges(i, j, w, graph_is_resistance)
    src_t = _read_strengths(sources, zero)
    gnd_t = _read_strengths(grounds, zero)
    if ground_is_resistance:
        with np.errstate(divide="ignore"):
            gnd_t[:, 1] = 1 / gnd_t[:, 1]
    src = np.zeros(m)
    gnd = np.zeros(m)
    for table, target, name in ((src_t, src, "source"), (gnd_t, gnd, "ground")):
        ids = table[:, 0].astype(np.int64)
        if len(ids) and (ids.min() < 1 or ids.max() > m):
            raise ValueError(f"a {name} node is not in the graph (nodes 1..{m})")
        target[ids - 1] = table[:, 1]  # as sources[ids] = values: the last one of a node wins
    r = core().network_advanced(a, b, w, m, src, gnd, remove_src_or_gnd, int(n_threads))
    nodes = np.arange(1, m + 1, dtype=np.float64)
    return NetworkResult(voltages=np.column_stack([nodes, r["voltage"]]),
                         node_currents=np.column_stack([nodes, r["current"]]),
                         branch_currents=_branch_table(*r["branch"]), n_components=int(r["n_components"]))


# --- pairwise -----------------------------------------------------------------------------------


def circuitscape_network_pairwise(graph, points, *, graph_is_resistance: bool = True, included_pairs=None,
                                  voltage_maps: bool = False, current_maps: bool = False,
                                  n_threads: int = 0) -> NetworkResult:
    """Circuitscape's pairwise mode on a network (``data_type = network``,
    ``scenario = pairwise``).

    Effective resistances between every pair of focal nodes (``points``: node
    numbers, a path or an array; numbered from 0 they are shifted by 1) on the
    graph of :func:`circuitscape_network_advanced`. Every component is
    factorised once and its pairs solved in batches (Circuitscape's resistance
    shortcut is for rasters only). ``included_pairs`` (:class:`IncludedPairs`
    or a path) restricts the pairs. With ``voltage_maps`` every pair's node
    voltages, and with any of the two options every pair's node and branch
    currents (as Circuitscape writes them) go to ``result.maps``, and the node
    and branch currents are summed over the pairs.

    Parameters
    ----------
    graph
        Edge list: path or ``(k, 3)`` array of node, node and resistance (or
        conductance).
    points
        Focal node numbers: path or array.
    graph_is_resistance
        The edge values are resistances (default) or conductances.
    included_pairs
        :class:`IncludedPairs` or the path of an include/exclude file.
    voltage_maps, current_maps
        Keep every pair's node voltages / node and branch currents in
        ``result.maps``.
    n_threads
        Threads (0 or ``None``: all cores but one).

    Returns
    -------
    NetworkResult
        ``resistances`` (ids in the first row and column), ``maps``,
        ``cum_node_currents`` and ``cum_branch_currents``.

    Examples
    --------
    >>> import numpy as np
    >>> from landschaft import circuit
    >>> graph = np.array([[1, 2, 1.0],             # node, node, resistance
    ...                   [2, 3, 1.0],
    ...                   [1, 3, 2.0],
    ...                   [3, 4, 0.5]])
    >>> result = circuit.circuitscape_network_pairwise(graph, [1, 3, 4])
    >>> result.resistances
    array([[0. , 1. , 3. , 4. ],
           [1. , 0. , 1. , 1.5],
           [3. , 1. , 0. , 0.5],
           [4. , 1.5, 0.5, 0. ]])
    """
    i, j, w, zero = read_graph(graph)
    a, b, w, m = _edges(i, j, w, graph_is_resistance)
    fp = _read_focal_nodes(points)
    if included_pairs is not None and not isinstance(included_pairs, IncludedPairs):
        included_pairs = read_included_pairs(included_pairs, starts_from_zero=zero)
    excluded: set = set()
    if included_pairs is not None:
        if included_pairs.mode == "include":
            fp = fp[np.isin(fp, included_pairs.ids)]
        excluded = included_pairs.excluded()
    n = len(fp)
    ex = np.zeros((n, n), dtype=np.uint8)
    if excluded:
        for p in range(n):
            for q in range(n):
                ex[p, q] = (int(fp[p]), int(fp[q])) in excluded
    maps: dict = {}

    def emit(ci, cj, nodes, volt, cur, branch):
        key = (int(fp[ci]), int(fp[cj]))
        ids = nodes + 1.0
        if volt is not None:
            maps[("voltages", *key)] = np.column_stack([ids, volt])
        maps[("node_currents", *key)] = np.column_stack([ids, cur])
        maps[("branch_currents", *key)] = _branch_table(*branch)

    opts = {"volt_maps": bool(voltage_maps), "cur_maps": bool(current_maps), "cum": bool(current_maps)}
    points_arr = np.ascontiguousarray(np.where((fp >= 1) & (fp <= m), fp - 1, -1), dtype=np.int32)
    r = core().network_pairwise(a, b, w, m, points_arr, ex, opts, emit, int(n_threads))
    res = np.zeros((n + 1, n + 1))
    res[0, 1:] = fp
    res[1:, 0] = fp
    res[1:, 1:] = r["resistances"]
    out = NetworkResult(resistances=res, maps=maps, n_components=int(r["n_components"]))
    if "cum" in r:
        out.cum_node_currents = np.column_stack([np.arange(1, m + 1, dtype=np.float64), r["cum"]])
        # Cumulative: one slot per edge of the input, the first line of a node pair
        # (either way round) takes its summed current, as initialize_cum_vectors does.
        cr, cc, cv = (np.asarray(x) for x in r["cum_branch"])
        total = {(int(p), int(q)): float(v) for p, q, v in zip(cr + 1, cc + 1, cv)}
        seen = set()
        vals = np.zeros(len(i))
        for k, (p, q) in enumerate(zip(i.tolist(), j.tolist())):
            key = (min(p, q), max(p, q))
            if key in seen:
                continue
            seen.add(key)
            vals[k] = total.get(key, 0.0)
        keep = ~(np.abs(vals) <= 1e-6)
        out.cum_branch_currents = np.column_stack([i[keep].astype(np.float64), j[keep].astype(np.float64),
                                                   vals[keep]])
    return out


# --- INI files ----------------------------------------------------------------------------------


def _write_table(path: str, a: np.ndarray, sep: str = "\t") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in np.atleast_2d(a):
            f.write(sep.join(repr(float(v)) for v in row) + "\n")


def network_ini(cfg: dict, base: Path, *, write_outputs: bool = True, n_threads: int = 0) -> NetworkResult:
    """``circuitscape_ini`` for ``data_type = network`` (``cfg``: the INI with
    Circuitscape's defaults)."""
    from .circuitscape import SCENARIOS

    def need(key):
        if _unset(cfg[key]):
            raise ValueError(f"{key} is not set")
        p = _resolve(cfg[key], base)
        if not p.is_file():
            raise FileNotFoundError(f'{key} = "{cfg[key]}" does not exist')
        return p

    scenario = SCENARIOS.get(cfg["scenario"])
    if scenario is None:
        raise ValueError(f'scenario = "{cfg["scenario"]}" is not one of {sorted(set(SCENARIOS.values()))}')
    graph = need("habitat_file")
    is_res = _flag(cfg, "habitat_map_is_resistances")
    volt, cur = _flag(cfg, "write_volt_maps"), _flag(cfg, "write_cur_maps")
    cum_only = _flag(cfg, "write_cum_cur_map_only")
    if scenario == "pairwise":
        pairs = None
        if _flag(cfg, "use_included_pairs"):
            pairs = read_included_pairs(need("included_pairs_file"), starts_from_zero=read_graph(graph)[3])
        # postprocess writes a pair's currents with any map option; write_cum_maps needs write_cur_maps
        any_maps = volt or cur or cum_only or _flag(cfg, "write_max_cur_maps")
        result = circuitscape_network_pairwise(graph, need("point_file"), graph_is_resistance=is_res,
                                               included_pairs=pairs, voltage_maps=volt, current_maps=any_maps,
                                               n_threads=n_threads)
        if not cur:
            result.cum_node_currents = result.cum_branch_currents = None
    else:
        # a network has no one-to-all / all-to-one: every other scenario runs as advanced
        policy = cfg["remove_src_or_gnd"]
        if policy == "not entered":
            policy = "keepall"
        result = circuitscape_network_advanced(graph, need("source_file"), need("ground_file"),
                                               graph_is_resistance=is_res,
                                               ground_is_resistance=_flag(cfg, "ground_file_is_resistances"),
                                               remove_src_or_gnd=policy, n_threads=n_threads)
    if not write_outputs:
        return result
    prefix, _ = _prefix(cfg)
    with open(cfg["output_file"], "w", encoding="utf-8") as f:
        for k, v in cfg.items():
            f.write(f"{k} = {v}\n")
    if scenario == "pairwise":
        r = result.resistances
        _write_matrix(prefix + "_resistances.out", r)
        fp = r[1:, 0]
        three = [(fp[p], fp[q], r[q + 1, p + 1]) for p in range(len(fp)) for q in range(p + 1, len(fp))]
        _write_matrix(prefix + "_resistances_3columns.out", np.array(three).reshape(-1, 3))
        for (kind, p, q), table in result.maps.items():
            _write_table(f"{prefix}_{kind}_{p}_{q}.txt", table)
        if result.cum_node_currents is not None:
            _write_table(f"{prefix}_node_currents_cum.txt", result.cum_node_currents)
            _write_table(f"{prefix}_branch_currents_cum.txt", result.cum_branch_currents)
    else:
        if volt:
            _write_table(f"{prefix}_voltages.txt", result.voltages)
        if cur or cum_only:
            _write_table(f"{prefix}_node_currents.txt", result.node_currents)
            _write_table(f"{prefix}_branch_currents.txt", result.branch_currents)
    return result
