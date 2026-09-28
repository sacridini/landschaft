# SPDX-License-Identifier: GPL-3.0-or-later
"""Trees for the phylogenetic indices: arrays in name order and Biodiverse's trimming."""

from __future__ import annotations

import numpy as np

__all__ = ["tree_view", "trimmed_mask"]


def _is_internal_name(name: str) -> bool:
    return name.endswith("___")


def trimmed_mask(tree, keep_labels) -> np.ndarray:
    """Nodes kept when Biodiverse trims a tree to the basedata labels (Tree::trim).

    Terminals that are not labels go, with named internal nodes whose clade
    holds no label; then unnamed internal nodes (``..___``) left without a
    named descendant go too. Nothing is collapsed: kept nodes keep their
    parents and lengths.
    """
    n = tree.n_nodes
    names = tree.names
    children = tree.children
    # get_trimmed_tree: nothing to do when every terminal is a label
    if all(names[i] in keep_labels for i in range(n) if not children[i]):
        return np.ones(n, bool)
    keep = {nm for nm in names if nm in keep_labels}
    parent = tree.parent
    root = tree.root

    def descendants(i):
        out = []
        stack = list(children[i])
        while stack:
            c = stack.pop()
            out.append(c)
            stack.extend(children[c])
        return out

    trim = set()
    for i in range(n):
        nm = names[i]
        if nm in keep or _is_internal_name(nm) or i == root:
            continue
        if not children[i]:
            trim.add(i)
            continue
        desc = descendants(i)
        if any(names[d] in keep for d in desc):
            # keep this node and its named ancestors
            a = i
            while a >= 0:
                if not _is_internal_name(names[a]):
                    keep.add(names[a])
                a = int(parent[a])
        else:
            trim.add(i)
    alive = np.ones(n, bool)
    deleted = False
    for i in trim:
        if names[i] in keep or not alive[i]:
            continue
        alive[i] = False
        deleted = True
        for d in descendants(i):
            alive[d] = False
    # unnamed internal nodes without named descendants (only after deletions)
    changed = deleted
    while changed:
        changed = False
        for i in range(n):
            if not alive[i] or i == root or not _is_internal_name(names[i]):
                continue
            live_desc = [d for d in descendants(i) if alive[d]]
            if not any(not _is_internal_name(names[d]) for d in live_desc):
                alive[i] = False
                for d in live_desc:
                    alive[d] = False
                changed = True
    return alive


def tree_view(tree, bd) -> dict:
    """The tree as arrays for the index engine, nodes numbered in name order."""
    order = tree.node_order  # original index of each node, by name
    n = tree.n_nodes
    new_of = np.empty(n, np.int64)
    new_of[order] = np.arange(n)
    parent = np.array([new_of[tree.parent[i]] if tree.parent[i] >= 0 else -1 for i in order], dtype=np.int32)
    length = tree.length[order].astype(np.float64)
    child_ptr = np.zeros(n + 1, np.int64)
    child_idx = []
    for k, i in enumerate(order):
        kids = [int(new_of[c]) for c in tree.children[i]]
        child_idx.extend(kids)
        child_ptr[k + 1] = child_ptr[k] + len(kids)
    depth = tree.depth()[order].astype(np.int32)
    terminal = np.array([len(tree.children[i]) == 0 for i in order], dtype=np.uint8)
    labels = set(bd.labels)
    alive = trimmed_mask(tree, labels)[order]
    trimmed_children = np.zeros(n, np.int32)
    for k in range(n):
        if alive[k]:
            trimmed_children[k] = sum(1 for c in child_idx[child_ptr[k]:child_ptr[k + 1]] if alive[c])
    names_sorted = [tree.names[i] for i in order]
    index = {nm: k for k, nm in enumerate(names_sorted)}
    label_node = np.array([index.get(lb, -1) for lb in bd.labels], dtype=np.int32)
    node_label = np.full(n, -1, np.int32)
    for li, k in enumerate(label_node):
        if k >= 0:
            node_label[k] = li
    total = 0.0
    trimmed_total = 0.0
    first_t = True
    for k in range(n):
        total += float(length[k])
        if alive[k]:
            if first_t:
                trimmed_total = float(length[k])
                first_t = False
            else:
                trimmed_total += float(length[k])
    # equalised-branch trees (PhylogeneticRelative) and counts
    nonzero_trimmed = int(sum(1 for k in range(n) if alive[k] and length[k]))
    mean_trimmed = trimmed_total / (nonzero_trimmed or 1)
    eq_trimmed_total = 0.0
    first_e = True
    default_eq_len = 0.0
    for k in range(n):
        if not alive[k]:
            continue
        v = mean_trimmed if length[k] else 0.0
        if default_eq_len == 0.0 and v:
            default_eq_len = v
        if first_e:
            eq_trimmed_total = v
            first_e = False
        else:
            eq_trimmed_total += v
    eq_full_total = float(sum(1 for k in range(n) if length[k]))
    trimmed_terminals = int(sum(1 for k in range(n) if alive[k] and trimmed_children[k] == 0))
    return {
        "n_terminals": int(terminal.sum()),
        "n_trimmed_terminals": trimmed_terminals,
        "zero_length_count": int(sum(1 for k in range(n) if not length[k])),
        "eq_full_total": eq_full_total,
        "eq_trimmed_total": eq_trimmed_total,
        "eq_trimmed_length": mean_trimmed,
        "default_eq_len": default_eq_len,
        "n": n,
        "parent": parent,
        "length": np.ascontiguousarray(length),
        "child_ptr": child_ptr,
        "child_idx": np.asarray(child_idx, dtype=np.int32),
        "depth": depth,
        "terminal": terminal,
        "in_trimmed": alive.astype(np.uint8),
        "trimmed_child_count": trimmed_children,
        "total_length": tree.total_length,
        "trimmed_total_length": trimmed_total,
        "root": int(new_of[tree.root]),
        "label_node": label_node,
        "node_label": node_label,
        "names": names_sorted,
    }
