# SPDX-License-Identifier: GPL-3.0-or-later
"""Agglomerative clustering of groups (Biodiverse::Cluster).

The matrix build and the joining loop run in C++ (``bio_cluster``); the tree
is finished here as Biodiverse's TreeNode does it: child lengths from the join
values, zero-length branches raised, several roots joined under a new one,
children ordered (ladderised) and terminals numbered.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .. import _core
from .._threads import resolve
from .._save import Saveable

_DIRECT = {"SORENSON", "JACCARD", "S2"}
_LINKAGES = ("link_average", "link_average_unweighted", "link_minimum", "link_maximum", "link_recalculate")


def _index_info(index: str, region_grower: bool = False) -> tuple[str, dict]:
    """The calculation giving a cluster (or region grower) index and the index's metadata."""
    from ._spatial import _dependencies, _registry

    if region_grower:
        for calc, meta in _registry().items():
            if index in meta["scalars"]:
                return calc, {}
        raise ValueError(f"{index!r} is not an index of the calculations")
    for calc, meta in _dependencies().items():
        info = meta.get("cluster_indices", {}).get(index)
        if info is not None:
            return calc, info
    valid = sorted(i for m in _dependencies().values() for i in m.get("cluster_indices", {}))
    raise ValueError(f"{index!r} is not a cluster index (valid: {', '.join(valid)})")


class _Node:
    __slots__ = ("name", "children", "parent", "length", "values")

    def __init__(self, name: str) -> None:
        self.name = name
        self.children: list[_Node] = []
        self.parent: _Node | None = None
        self.length: float = 0.0
        self.values: dict = {}

    def is_terminal(self) -> bool:
        return not self.children

    def length_below(self) -> float:
        best = 0.0
        for c in self.children:
            lb = c.length_below() or 0.0
            if lb > best:
                best = lb
        return (self.length or 0.0) + best

    def total_length(self) -> float:
        v = self.values.get("TOTAL_LENGTH")
        return v if v is not None else self.length_below()

    def add_children(self, children: list[_Node]) -> None:
        for child in children:
            if child in self.children:
                continue
            if child.parent is not None:
                child.parent.delete_child(child)
            self.children.append(child)
            child.parent = self

    def delete_child(self, child: _Node) -> None:
        for i, c in enumerate(self.children):
            if c is child:
                del self.children[i]
                return

    def descendants(self) -> list[_Node]:
        out = []
        stack = list(self.children)
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children)
        return out


def _raise_zero_length(node: _Node) -> tuple[int, list[str]]:
    """TreeNode::raise_zerolength_children."""
    raised, empty = 0, []
    if node.is_terminal():
        return raised, empty
    children = list(node.children)
    count = len(children)
    is_root = node.parent is None
    if not is_root:
        for child in children:
            if child.total_length() == node.total_length():
                node.parent.add_children([child])
                raised += 1
    if raised == count:
        node.parent.delete_child(node)
        empty.append(node.name)
    elif not is_root and raised == count - 1:
        child = node.children[0]
        parent = node.parent
        parent.add_children([child])
        child.length = (node.length or 0.0) + (child.length or 0.0)
        parent.delete_child(node)
        raised += 1
        empty.append(node.name)
    for child in children:
        r, e = _raise_zero_length(child)
        raised += r
        empty.extend(e)
    return raised, empty


def _number_terminals(node: _Node, count: int = 0) -> int:
    left = right = count
    for child in node.children:
        if child.is_terminal():
            right += 1
            child.values["TERMINAL_NODE_FIRST"] = right
            child.values["TERMINAL_NODE_LAST"] = right
        else:
            right = _number_terminals(child, right)
    node.values["TERMINAL_NODE_FIRST"] = left + 1
    node.values["TERMINAL_NODE_LAST"] = right
    return right


def _hierarchical_labels(bd, nodes: dict, order: list[str]):
    """Each node's labels, group counts and samples as Biodiverse's hierarchical mode merges them.

    A node takes its first child's labels; later children add theirs with a
    group count of 1 (Biodiverse sets rather than adds these) and their
    samples added.
    """
    c = bd.counts
    gidx = {g: i for i, g in enumerate(bd.groups)}
    cache: dict[str, tuple[dict, dict]] = {}

    def depth(nd):
        d = 0
        while nd.parent is not None:
            nd = nd.parent
            d += 1
        return d

    by_depth = sorted(order, key=lambda nm: (-depth(nodes[nm]), nm))
    for nm in by_depth:
        nd = nodes[nm]
        if nd.is_terminal():
            g = gidx[nm]
            labs = c.indices[c.indptr[g]:c.indptr[g + 1]].tolist()
            vals = c.data[c.indptr[g]:c.indptr[g + 1]].tolist()
            cache[nm] = ({lb: 1 for lb in labs}, dict(zip(labs, vals)))
            continue
        counts: dict = {}
        samples: dict = {}
        for child in nd.children:
            cc, cs = cache[child.name]
            if not counts:
                counts = dict(cc)
            else:
                for lb in cc:
                    counts[lb] = 1
            if not samples:
                samples = dict(cs)
            else:
                for lb, v in cs.items():
                    samples[lb] = samples.get(lb, 0.0) + v
        cache[nm] = (counts, samples)
    indptr = [0]
    labels, cnts, smps = [], [], []
    for nm in order:
        cc, cs = cache[nm]
        for lb in sorted(set(cc) | set(cs)):
            labels.append(lb)
            cnts.append(float(cc.get(lb, 0)))
            smps.append(float(cs.get(lb, 0.0)))
        indptr.append(len(labels))
    return (np.asarray(indptr, dtype=np.int64), np.asarray(labels, dtype=np.int32),
            np.asarray(cnts, dtype=np.float64), np.asarray(smps, dtype=np.float64))


class ClusterResult(Saveable):
    """Results of :func:`cluster`.

    Attributes
    ----------
    tree : Tree
        The dendrogram (terminals are the groups; internal nodes are named
        ``<join number>___``).
    nodes : pandas.DataFrame
        Per node: ``parent``, ``length`` and Biodiverse's node values
        (``TOTAL_LENGTH``, ``JOIN_NUMBER``, ``MATRIX_ITER_USED``,
        ``TERMINAL_NODE_FIRST``, ``TERMINAL_NODE_LAST``, ``JOIN_NOT_METRIC``).
    table : pandas.DataFrame
        Scalar indices of each node's groups (``calculations``), or empty.
    lists : dict
        List indices of each node's groups.
    """

    def __init__(self, tree, nodes: pd.DataFrame, table: pd.DataFrame, lists) -> None:
        self.tree = tree
        self.nodes = nodes
        self.table = table
        self.lists = lists

    def _save_items(self) -> dict:
        return {"tree": self.tree, "nodes": self.nodes, "table": self.table, "lists": {k: self.lists[k] for k in self.lists}}

    def __repr__(self) -> str:
        return f"ClusterResult({(self.nodes['children'].str.len() == 0).sum()} groups, {len(self.nodes)} nodes)"

    def cut(self, k: int) -> pd.Series:
        """Cut the dendrogram into ``k`` clusters (the highest joins are undone first).

        Parameters
        ----------
        k : int
            Number of clusters (at most the number of groups).

        Returns
        -------
        pandas.Series
            Cluster number (1 to ``k``, in the dendrogram's terminal order) of
            each group.

        Examples
        --------
        >>> import pandas as pd
        >>> import landschaft as ls
        >>> df = pd.DataFrame({"sp": ["a", "b", "a", "b", "c", "c", "d"],
        ...                    "x": [0.5, 0.5, 1.5, 1.5, 2.5, 3.5, 3.5], "y": [0.5] * 7})
        >>> bd = ls.bio.BaseData.from_dataframe(df, ["x", "y"], "sp", cell_sizes=[1, 1])
        >>> ls.bio.cluster(bd, prng_seed=1).cut(2).to_dict()
        {'0.5:0.5': 1, '1.5:0.5': 1, '2.5:0.5': 2, '3.5:0.5': 2}
        """
        nodes = self.nodes
        n_groups = int((nodes["children"].str.len() == 0).sum())
        if not 1 <= k <= n_groups:
            raise ValueError(f"k must be between 1 and {n_groups}")
        height = (nodes["TOTAL_LENGTH"] - nodes["length"].fillna(0)).to_dict()
        children = nodes["children"].to_dict()
        first = nodes["TERMINAL_NODE_FIRST"].to_dict()
        clusters = list(nodes.index[nodes["parent"].isna()])
        while len(clusters) < k:
            # the highest join still in place (ties: the first in terminal order)
            split = max((c for c in clusters if len(children[c])), key=lambda c: (height[c], -first[c]))
            clusters.remove(split)
            clusters.extend(children[split])
        out = {}
        for i, c in enumerate(sorted(clusters, key=lambda c: first[c]), start=1):
            stack = [c]
            while stack:
                nd = stack.pop()
                if len(children[nd]):
                    stack.extend(children[nd])
                else:
                    out[nd] = i
        return pd.Series(out, name="cluster").sort_index()


def region_grower(
    bd,
    index: str = "RICHNESS_ALL",
    conditions: str | Sequence[str] | None = None,
    definition_query: str | None = None,
    *,
    calculations: Sequence[str] | None = None,
    tree=None,
    matrix=None,
    prng_seed: int | None = None,
    tie_breaker: Sequence[str] | None = None,
    n_threads: int | None = None,
) -> ClusterResult:
    """Grow regions by joining the pairs with the highest value of an index (Biodiverse's RegionGrower).

    Like :func:`cluster`, but pairs are joined in decreasing order of an
    index computed for the pair's groups together (such as ``RICHNESS_ALL``
    or ``PE_WE``), recomputed for each merged region. With one condition
    that selects all groups, joining stops once a region reaches the value
    of all the groups together.

    Parameters
    ----------
    bd : BaseData
    index : str
        The index to maximise.
    conditions, definition_query, calculations, tree, matrix, prng_seed, tie_breaker, n_threads
        As for :func:`cluster`.

    Returns
    -------
    ClusterResult

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> df = pd.DataFrame({"sp": ["a", "b", "a", "c", "d"], "x": [0.5, 0.5, 1.5, 2.5, 3.5], "y": [0.5] * 5})
    >>> bd = ls.bio.BaseData.from_dataframe(df, ["x", "y"], "sp", cell_sizes=[1, 1])
    >>> res = ls.bio.region_grower(bd, prng_seed=1)
    >>> res.nodes.loc[res.tree.names[res.tree.root], "TOTAL_LENGTH"]
    np.float64(4.0)
    """
    return cluster(bd, index, "link_recalculate", conditions, definition_query, calculations=calculations, tree=tree,
                   matrix=matrix, prng_seed=prng_seed, tie_breaker=tie_breaker, n_threads=n_threads,
                   _region_grower=True)


def cluster(
    bd,
    index: str = "SORENSON",
    linkage: str = "link_average",
    conditions: str | Sequence[str] | None = None,
    definition_query: str | None = None,
    *,
    calculations: Sequence[str] | None = None,
    tree=None,
    matrix=None,
    prng_seed: int | None = None,
    tie_breaker: Sequence[str] | None = None,
    clear_singletons: bool = False,
    n_threads: int | None = None,
    _region_grower: bool = False,
) -> ClusterResult:
    """Cluster the groups by the dissimilarity of their labels (Biodiverse's cluster analysis).

    Pairs of groups (or clusters) with the lowest index value are joined
    until one tree remains; cluster-to-cluster values follow the linkage.
    With several spatial conditions, pairs within the first condition are
    joined first, then the second and so on.

    Parameters
    ----------
    bd : BaseData
    index : str
        Dissimilarity index (``SORENSON``, ``JACCARD``, ``S2``, ``BRAY_CURTIS``,
        ``PHYLO_SORENSON``...).
    linkage : str
        ``link_average`` (UPGMA), ``link_average_unweighted`` (WPGMA),
        ``link_minimum``, ``link_maximum`` or ``link_recalculate`` (the
        index recomputed for the merged groups).
    conditions : str or list of str, optional
        Spatial conditions limiting the pairs (default ``sp_select_all()``).
    definition_query : str, optional
        Only groups passing it are clustered.
    calculations : list of str, optional
        Indices computed for the groups of each node.
    tree, matrix : optional
        Phylogeny and label matrix for the index and the calculations.
    prng_seed : int, optional
        Seed of Biodiverse's Mersenne Twister, which breaks ties between equal
        values (random when not given).
    tie_breaker : list of str, optional
        Biodiverse's ``cluster_tie_breaker`` pairs, such as
        ``["random", "maximise"]``; by default ties are broken by random
        shuffles.
    clear_singletons : bool
        Leave out groups that joined nothing when joining several roots.
    n_threads : int, optional
        Threads for the index values and calculations.

    Returns
    -------
    ClusterResult

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> df = pd.DataFrame({"sp": ["a", "b", "a", "b", "c", "c", "d"],
    ...                    "x": [0.5, 0.5, 1.5, 1.5, 2.5, 3.5, 3.5], "y": [0.5] * 7})
    >>> bd = ls.bio.BaseData.from_dataframe(df, ["x", "y"], "sp", cell_sizes=[1, 1])
    >>> res = ls.bio.cluster(bd, prng_seed=1)
    >>> res.tree.to_newick(lengths=False)
    "(('0.5:0.5','1.5:0.5')0___,('2.5:0.5','3.5:0.5')1___)2___;"
    """
    from ._conditions import _as_condition, neighbours
    from ._spatial import _basedata_view, _calc_name, _registry

    if linkage not in _LINKAGES:
        raise ValueError(f"linkage must be one of {', '.join(_LINKAGES)}")
    calc, info = _index_info(index, _region_grower)
    if conditions is None:
        conditions = ["sp_select_all()"]
    elif isinstance(conditions, str):
        conditions = [conditions]
    conditions = [c if isinstance(c, str) else c for c in conditions]
    while conditions and not str(conditions[-1]).strip():
        conditions.pop()
    if not conditions:
        conditions = ["1"]
    threads = resolve(n_threads)
    nb = neighbours(bd, conditions, definition_query or None, n_threads=threads)
    procs = np.asarray(nb.groups, dtype=np.int64)
    n = len(procs)
    if not n:
        raise ValueError("no groups to cluster (check the conditions and definition query)")
    elem_of = np.full(bd.n_groups, -1, dtype=np.int64)
    elem_of[procs] = np.arange(n)
    nbr_lists = []
    for ip, ix in nb._stored:
        per = []
        for r in range(n):
            e = elem_of[ix[ip[r]:ip[r + 1]]]
            # neighbours processed before are excluded (exclude_processed_elements)
            per.append([int(x) for x in e if x > r])
        nbr_lists.append(per)
    types = [_as_condition(c).result_type for c in conditions]
    trim = len(conditions) == 1 and types[0] == "always_true"
    names = [bd.groups[g] for g in procs]

    labels: list[list[int]] = []
    pair_keys: list[int] = []
    pair_values: list[float] = []
    direct = index if index in _DIRECT and not _region_grower else ""
    if direct:
        c = bd.counts
        labels = [c.indices[c.indptr[g]:c.indptr[g + 1]].tolist() for g in procs]
    else:
        pairs = [(e1, e2) for per in nbr_lists for e1, lst in enumerate(per) for e2 in lst]
        pairs = sorted(set(pairs))
        if pairs:
            from ._spatial import _run_sets

            p1 = np.array([procs[a] for a, _ in pairs], dtype=np.int32)
            p2 = np.array([procs[b] for _, b in pairs], dtype=np.int32)
            ip = np.arange(len(pairs) + 1, dtype=np.int64)
            table, _ = _run_sets(bd, p1, (ip, p1), (ip, p2), [calc], tree=tree, matrix=matrix, threads=threads,
                                 view_extra={"pairwise_mode": True})
            vals = table[index].to_numpy(dtype=np.float64)
            for (a, b), v in zip(pairs, vals):
                if not np.isnan(v):
                    pair_keys.append(int(a) * n + int(b))
                    pair_values.append(float(v))

    bounds = info.get("bounds") or []
    min_poss = float(bounds[0]) if bounds and bounds[0] is not None and not isinstance(bounds[0], str) else 1e10
    keys, minmax = [], []
    if tie_breaker:
        tb = list(tie_breaker)
        for k in range(0, len(tb), 2):
            if tb[k] == "none":
                continue
            if tb[k] != "random":
                raise NotImplementedError("only the 'random' and 'none' tie breakers are supported so far")
            keys.append(tb[k])
            minmax.append(tb[k + 1] if k + 1 < len(tb) else "minimise")
    if prng_seed is None or prng_seed == 0:
        import secrets

        prng_seed = secrets.randbits(32)
    rng = _core.BioRandom(int(prng_seed) % (1 << 64))

    # link_recalculate: the index of the merged groups against each other node
    terminal_groups: dict[str, list[int]] = {nm: [int(g)] for nm, g in zip(names, procs)}

    def recalculate(n1, n2, new_name, checks):
        from ._spatial import _run_sets

        merged = sorted(set(terminal_groups[n1]) | set(terminal_groups[n2]))
        terminal_groups[new_name] = merged
        if not checks:
            return []
        rows2 = [terminal_groups[c] for c in checks]
        m = len(checks)
        ip1 = np.arange(m + 1, dtype=np.int64) * len(merged)
        ix1 = np.tile(np.asarray(merged, dtype=np.int32), m)
        ip2 = np.zeros(m + 1, dtype=np.int64)
        ip2[1:] = np.cumsum([len(r) for r in rows2])
        ix2 = np.array([g for r in rows2 for g in r], dtype=np.int32)
        tab, _ = _run_sets(bd, np.full(m, merged[0], dtype=np.int32), (ip1, ix1), (ip2, ix2), [calc], tree=tree,
                           matrix=matrix, threads=threads)
        return [float(v) for v in tab[index].to_numpy(dtype=np.float64)]

    max_poss = float("nan")
    if _region_grower and len(conditions) == 1 and types[0] == "always_true":
        from ._spatial import _run_sets

        allg = np.asarray(procs, dtype=np.int32)
        tab, _ = _run_sets(bd, allg[:1], (np.array([0, len(allg)], dtype=np.int64), allg), None, [calc], tree=tree,
                           matrix=matrix, threads=threads)
        max_poss = float(tab[index].iloc[0])
    terminals, joins, join_number, min_value = _core.bio_cluster(
        names, nbr_lists, trim, direct, labels, pair_keys, pair_values, linkage,
        bool(info.get("can_lump_zeroes")), min_poss, bool(tie_breaker), keys, minmax, _region_grower, max_poss, rng,
        recalculate if linkage == "link_recalculate" else None)

    # ---- the tree
    nodes: dict[str, _Node] = {}
    for t in terminals:
        nodes[t] = _Node(t)
    for jn, name, c1, c2, value, it in joins:
        node = _Node(name)
        nodes[name] = node
        node.add_children([nodes[c1], nodes[c2]])
        node.values["MATRIX_ITER_USED"] = it
        node.values["JOIN_NUMBER"] = jn
        for child in node.children:
            child.values["TOTAL_LENGTH"] = value
            if child.is_terminal():
                child.length = value
            else:
                child.length = value - child.children[0].total_length()
        for child in node.children:
            if child.is_terminal():
                child.values["MATRIX_ITER_USED"] = it
                child.values["JOIN_NUMBER"] = jn
    roots = [nd for nd in nodes.values() if nd.parent is None]
    for r in roots:
        if r.is_terminal():
            continue
        count = 1
        empty: list[str] = []
        while count > 0:
            count, e = _raise_zero_length(r)
            empty.extend(e)
        for name in empty:
            nodes.pop(name, None)
    roots = [nd for nd in nodes.values() if nd.parent is None]
    if len(roots) > 1:
        to_add, singletons = [], []
        target = None
        for r in roots:
            if clear_singletons and r.is_terminal():
                singletons.append(r.name)
                continue
            lb = r.length_below()
            target = lb if target is None else max(lb, target)
            to_add.append(r)
        k = 0
        while f"{k}___" in nodes:
            k += 1
        root = _Node(f"{k}___")
        root.values["JOIN_NOT_METRIC"] = 1
        for r in to_add:
            r.length = target - r.length_below()
            r.values["JOIN_NOT_METRIC"] = 1
        root.add_children(to_add)
        nodes[root.name] = root
        for s in singletons:
            nodes.pop(s, None)
    root = next(nd for nd in nodes.values() if nd.parent is None)
    root.length = 0.0
    root.values["TOTAL_LENGTH"] = min_value
    root.values["JOIN_NUMBER"] = join_number
    root.values.setdefault("MATRIX_ITER_USED", None)
    # ladderise: more descendants first, then by name
    for nd in [root] + root.descendants():
        nd.children.sort(key=lambda c: (-len(c.descendants()), c.name))
    _number_terminals(root)

    from ._tree import Tree

    order = sorted(nodes)
    idx = {nm: i for i, nm in enumerate(order)}
    parent = [idx[nodes[nm].parent.name] if nodes[nm].parent is not None else -1 for nm in order]
    lengths = [nodes[nm].length for nm in order]
    children = [[idx[c.name] for c in nodes[nm].children] for nm in order]
    tr = Tree(order, parent, lengths, children, name="cluster")
    cols = ["TOTAL_LENGTH", "JOIN_NUMBER", "MATRIX_ITER_USED", "TERMINAL_NODE_FIRST", "TERMINAL_NODE_LAST",
            "JOIN_NOT_METRIC"]
    frame = pd.DataFrame({
        "parent": [nodes[nm].parent.name if nodes[nm].parent is not None else None for nm in order],
        "length": lengths,
        "children": [[c.name for c in nodes[nm].children] for nm in order],
        **{c: [nodes[nm].values.get(c) for nm in order] for c in cols},
    }, index=pd.Index(order, name="node"))

    table = pd.DataFrame(index=frame.index)
    lists: dict = {}
    if calculations:
        from ._spatial import _run_sets

        reg = _registry()
        calcs = []
        for c in calculations:
            cn = _calc_name(c)
            if cn not in reg:
                raise ValueError(f"unknown calculation {c!r}")
            if reg[cn]["uses_nbr_lists"] <= 1 and cn not in calcs:
                calcs.append(cn)
        gidx = {g: i for i, g in enumerate(bd.groups)}
        rows = []
        for nm in order:
            nd = nodes[nm]
            terms = [nd] if nd.is_terminal() else [x for x in nd.descendants() if x.is_terminal()]
            rows.append(sorted(gidx[t.name] for t in terms))
        ip = np.zeros(len(rows) + 1, dtype=np.int64)
        ip[1:] = np.cumsum([len(r) for r in rows])
        ix = np.array([g for r in rows for g in r], dtype=np.int32)
        procs_rows = np.array([r[0] for r in rows], dtype=np.int32)
        row_labels = _hierarchical_labels(bd, nodes, order)
        table, lists = _run_sets(bd, procs_rows, (ip, ix), None, calcs, tree=tree, matrix=matrix, threads=threads,
                                 row_names=order, view_extra={"row_labels": row_labels})
    _ = _basedata_view  # (keeps the import used)
    return ClusterResult(tr, frame, table, lists)
