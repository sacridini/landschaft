# SPDX-License-Identifier: GPL-3.0-or-later
"""The compiler of :class:`~landschaft.prioritize.Problem`: the model of
prioritizr 8.0.6's ``compile()`` (Hanson et al.; GPL-3), bit for bit.

Layout (one zone, the compressed formulation): the columns are the planning
units, then the columns the objective adds; the rows are the targets (one per
target: the amounts of its feature in the units), then the rows the objective
adds. Every objective follows its ``rcpp_apply_*_objective``; the feature
weights ``rcpp_apply_feature_weights``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

# objectives whose model has no targets (prioritizr warns and ignores them)
NO_TARGETS = ("max_utility", "max_cover")
# objectives whose model ignores feature weights (prioritizr warns)
NO_WEIGHTS = ("min_set", "min_largest_shortfall")
PHYLO = ("max_phylo_div", "max_phylo_end")
BUDGETED = ("min_shortfall", "min_largest_shortfall", "max_utility", "max_features", "max_cover", "min_penalties",
            *PHYLO)


@dataclass(frozen=True)
class Targets:
    """The targets as the model's rows: feature (0-based), sense and amount of each."""

    feature: np.ndarray
    sense: tuple[str, ...]
    value: np.ndarray

    @property
    def n(self) -> int:
        return self.feature.size


def _sum(v: np.ndarray) -> float:
    """A sum in order, in double (Rcpp's sum(), Matrix's rowSums over a row)."""
    return float(np.cumsum(v, dtype=np.float64)[-1]) if v.size else 0.0


def tree_branches(tree, names: tuple[str, ...]) -> tuple[sp.csr_matrix, np.ndarray]:
    """A phylogeny's branches as ape orders them (``read.tree``'s cladewise order: depth first, children
    in the order given; the root's own length left out) and the branch matrix (prioritizr's
    ``branch_matrix()``): 1 where a feature's tip lies below a branch, one row per feature, in the
    problem's order. ``tree``: an ``ls.bio`` tree or a Newick string; its tips must be the features."""
    if isinstance(tree, str):
        from ..bio import read_tree

        tree = read_tree(tree, format="newick")
    children = tree.children
    tips: list[int] = []
    edges: list[tuple[int, int]] = []  # (parent node, child node)
    stack = [tree.root]
    while stack:
        node = stack.pop()
        if not children[node]:
            tips.append(node)
        for c in reversed(children[node]):
            stack.append(c)
    # edges in preorder: the edge to a node right before the edges below it
    order: list[int] = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed(children[node]))
    edge_of = {}
    for node in order:
        if node != tree.root:
            edge_of[node] = len(edges)
            edges.append((int(tree.parent[node]), node))
    labels = [tree.names[t] for t in tips]
    if len(labels) != len(names) or set(labels) != set(names):
        raise ValueError("the tree's tips must be the problem's features")
    tip_of = {tree.names[t]: t for t in tips}
    rows, cols = [], []
    for f, name in enumerate(names):  # the features' order (prioritizr reorders by the inverse permutation)
        node = tip_of[name]
        while node != tree.root:
            rows.append(f)
            cols.append(edge_of[node])
            node = int(tree.parent[node])
    bm = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(names), len(edges)))
    bm.sort_indices()
    lengths = np.array([tree.length[c] for _, c in edges], dtype=np.float64)
    return bm, lengths


def endemism_lengths(bm: sp.csr_matrix, lengths: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """add_max_phylo_end_objective's branch lengths: each divided by the total amount of the features
    below it (Matrix::colSums, in feature order), then all scaled up so that the shortest is 1 if it
    was shorter."""
    cc = bm.tocsc()
    cc.sort_indices()
    out = lengths.copy()
    for e in range(cc.shape[1]):
        f = cc.indices[cc.indptr[e]:cc.indptr[e + 1]]
        out[e] = out[e] * (1 / _sum(totals[f]))
    m = float(out.min())
    if m < 1:
        out = out * (1 / m)
    return out


def loglinear(x: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """prioritizr's loglinear_interpolation: y1 up to x1, y2 from x2, between them linear in
    log10(x) (R's approx(): y1 + (y2 - y1) * ((v - u1) / (u2 - u1)), u = log10)."""
    x = np.asarray(x, dtype=np.float64)
    if not x1 < x2:
        raise ValueError("the lower bound amount must be below the upper bound amount")
    out = np.full(x.size, np.nan)
    out[x <= x1] = y1
    out[x >= x2] = y2
    mid = np.isnan(out)
    u1, u2 = math.log10(x1), math.log10(x2)
    v = np.log10(x[mid])
    r = y1 + (y2 - y1) * ((v - u1) / (u2 - u1))
    # approx() gives the end values exactly at the ends
    r = np.where(v == u1, y1, np.where(v == u2, y2, r))
    out[mid] = r
    return out


def resolve_targets(spec, names: tuple[str, ...], totals: np.ndarray) -> Targets:
    """The rows of a target specification (``Problem.targets``)."""
    kind = spec[0]
    nf = len(names)
    if kind == "relative":
        return Targets(np.arange(nf), (">=",) * nf, totals * spec[1])
    if kind == "absolute":
        return Targets(np.arange(nf), (">=",) * nf, np.asarray(spec[1], dtype=np.float64).copy())
    if kind == "loglinear":
        lo_a, lo_t, hi_a, hi_t, cap_a, cap_t, abundances = spec[1]
        a = totals if abundances is None else abundances
        value = a * loglinear(a, lo_a, lo_t, hi_a, hi_t)
        if cap_a is not None:
            value[a >= cap_a] = cap_t
        return Targets(np.arange(nf), (">=",) * nf, value)
    if kind == "manual":
        t: pd.DataFrame = spec[1]
        index = {n: k for k, n in enumerate(names)}
        missing = sorted(set(map(str, t["feature"])) - set(index))
        if missing:
            raise ValueError(f"unknown features in the targets: {missing}")
        feature = np.array([index[str(f)] for f in t["feature"]], dtype=np.int64)
        target = t["target"].to_numpy(dtype=np.float64)
        kinds = t["type"].astype(str).to_numpy()
        if not np.isin(kinds, ("relative", "absolute")).all():
            raise ValueError("target types must be 'relative' or 'absolute'")
        sense = tuple(t["sense"].astype(str)) if "sense" in t.columns else (">=",) * len(t)
        if not set(sense) <= {">=", "<=", "="}:
            raise ValueError("target senses must be '>=', '<=' or '='")
        value = np.where(kinds == "relative", totals[feature] * target, target)
        return Targets(feature, sense, value)
    raise ValueError(f"unknown targets {kind!r}")


def compile_problem(p):
    """The model prioritizr's ``compile()`` gives for the problem ``p``."""
    from ._milp import OptimizationProblem

    u = p.units
    if p.objective is None:
        raise ValueError("the problem has no objective: add one (add_min_set_objective, ...)")
    kind = p.objective[0]
    nf = len(u.feature_names)
    n = u.n
    if kind in NO_TARGETS:
        if p.targets is not None:
            warnings.warn(f"targets are ignored by the {kind} objective", UserWarning, stacklevel=3)
        # the "fake" targets prioritizr's compile() makes: one per feature
        t = Targets(np.arange(nf), ("?",) * nf, np.zeros(nf))
    else:
        if p.targets is None:
            raise ValueError(f"the {kind} objective needs targets (add_relative_targets, add_absolute_targets, ...)")
        t = resolve_targets(p.targets, u.feature_names, u.totals)
    nt = t.n
    cost = u.cost

    # decisions: the planning units' columns
    lb = [np.zeros(n)]
    if p.decisions[0] == "binary":
        ub, vtype = [np.ones(n)], ["B"] * n
    elif p.decisions[0] == "proportion":
        ub, vtype = [np.ones(n)], ["C"] * n
    else:
        ub, vtype = [np.full(n, p.decisions[1])], ["C"] * n

    # the targets' rows (the features' amounts) and the objective's rows and columns
    rij = u.rij[t.feature] if not np.array_equal(t.feature, np.arange(nf)) else u.rij
    blocks = [sp.coo_matrix(rij)]  # (rows, cols) placed on a grid below
    extra_rows: list[tuple] = []  # (row, col, value) triplets of the objective
    obj = [np.zeros(n)]
    col_ids = ["pu"] * n
    row_ids: list[str] = []
    sense: list[str] = []
    rhs: list[np.ndarray] = []
    budget = None if kind not in BUDGETED else np.atleast_1d(np.asarray(p.objective[1], dtype=np.float64))
    nrow_extra = 0
    ncol_extra = 0
    weight_cols = None  # first column the feature weights replace

    def budget_row(row: int):
        extra_rows.append((np.full(n, row), np.arange(n), cost))

    if kind == "min_set":
        obj = [cost.copy()]
        sense, rhs, row_ids = list(t.sense), [t.value], ["spp_target"] * nt
    elif kind == "min_shortfall":
        sense = [*t.sense, "<="]
        rhs = [t.value, budget]
        obj.append(np.where(t.value > 1.0e-5, 1.0 / np.where(t.value > 1.0e-5, t.value, 1.0), 0.0))
        lb.append(np.zeros(nt))
        ub.append(np.full(nt, np.inf))
        vtype += ["C"] * nt
        extra_rows.append((np.arange(nt), n + np.arange(nt), np.ones(nt)))
        budget_row(nt)
        col_ids += ["spp_met"] * nt
        row_ids = ["spp_target"] * nt + ["budget"]
        nrow_extra, ncol_extra, weight_cols = 1, nt, n
    elif kind == "min_largest_shortfall":
        sense = [*t.sense, *([">="] * nt), "<="]
        rhs = [t.value, np.zeros(nt), budget]
        obj += [np.zeros(nt), np.ones(1)]
        lb.append(np.zeros(nt + 1))
        ub.append(np.full(nt + 1, np.inf))
        vtype += ["C"] * (nt + 1)
        extra_rows.append((np.arange(nt), n + np.arange(nt), np.ones(nt)))
        inv = np.where(t.value > 1.0e-5, -1.0 / np.where(t.value > 1.0e-5, t.value, 1.0), 0.0)
        extra_rows.append((nt + np.arange(nt), n + np.arange(nt), inv))
        extra_rows.append((nt + np.arange(nt), np.full(nt, n + nt), np.ones(nt)))
        budget_row(2 * nt)
        col_ids += ["spp_met"] * nt + ["max_shortfall"]
        row_ids = ["spp_target"] * nt + ["max_shortfall"] * nt + ["budget"]
        nrow_extra, ncol_extra = nt + 1, nt + 1
    elif kind in ("max_utility", "max_cover", "max_features"):
        scale = -0.01 / _sum(cost)
        obj = [cost * scale, np.ones(nt)]
        if kind == "max_utility":
            pos = u.rij.copy()
            pos.data = np.maximum(pos.data, 0.0)
            abund = np.array([_sum(pos.data[pos.indptr[i]:pos.indptr[i + 1]]) for i in range(nf)])
            negative = u.rij.nnz > 0 and float(u.rij.data.min()) < 0
            lb.append(np.full(nt, -np.inf if negative else 0.0))
            ub.append(np.maximum(abund, 0.0))
            vtype += ["C"] * nt
            extra_rows.append((np.arange(nt), n + np.arange(nt), -np.ones(nt)))
            sense, rhs = ["="] * nt + ["<="], [np.zeros(nt), budget]
            col_ids += ["amount"] * nt
            row_ids = ["spp_amount"] * nt + ["budget"]
        elif kind == "max_cover":
            lb.append(np.zeros(nt))
            ub.append(np.ones(nt))
            vtype += ["B"] * nt
            extra_rows.append((np.arange(nt), n + np.arange(nt), -np.ones(nt)))
            sense, rhs = [">="] * nt + ["<="], [np.zeros(nt), budget]
            col_ids += ["present"] * nt
            row_ids = ["spp_present"] * nt + ["budget"]
        else:
            lb.append(np.zeros(nt))
            ub.append(np.ones(nt))
            vtype += ["B"] * nt
            extra_rows.append((np.arange(nt), n + np.arange(nt), -1.0 * t.value))
            sense, rhs = [*t.sense, "<="], [np.zeros(nt), budget]
            col_ids += ["spp_met"] * nt
            row_ids = ["spp_target"] * nt + ["budget"]
        budget_row(nt)
        nrow_extra, ncol_extra, weight_cols = 1, nt, n
    elif kind in PHYLO:
        bm, lengths = tree_branches(p.objective[2], u.feature_names)
        if kind == "max_phylo_end":
            lengths = endemism_lengths(bm, lengths, u.totals)
        ne = lengths.size
        scale = (-0.01 * float(lengths.min())) / _sum(cost)
        obj = [cost * scale, np.zeros(nt), lengths]
        lb.append(np.zeros(nt + ne))
        ub.append(np.ones(nt + ne))
        vtype += ["B"] * (nt + ne)
        extra_rows.append((np.arange(nt), n + np.arange(nt), -1.0 * t.value))
        budget_row(nt)
        # a branch is met when a target of a feature below it is: sum of those targets' columns - branch >= 0
        below = sp.csr_matrix(bm[t.feature]).tocsc()  # targets x branches
        below.sort_indices()
        br_rows, br_cols = [], []
        for e in range(ne):
            ts = below.indices[below.indptr[e]:below.indptr[e + 1]]
            br_rows += [nt + 1 + e] * (ts.size + 1)
            br_cols += [*(n + ts), n + nt + e]
        br_vals = np.ones(len(br_rows))
        br_vals[np.cumsum([below.indptr[e + 1] - below.indptr[e] + 1 for e in range(ne)]) - 1] = -1.0
        extra_rows.append((np.array(br_rows), np.array(br_cols), br_vals))
        sense, rhs = [*t.sense, "<=", *([">="] * ne)], [np.zeros(nt), budget, np.zeros(ne)]
        col_ids += ["spp_met"] * nt + ["branch_met"] * ne
        row_ids = ["spp_target"] * nt + ["budget"] + ["branch_target"] * ne
        nrow_extra, ncol_extra, weight_cols = 1 + ne, nt + ne, n
    elif kind == "min_penalties":
        sense, rhs = [*t.sense, "<="], [t.value, budget]
        budget_row(nt)
        row_ids = ["spp_target"] * nt + ["budget"]
        nrow_extra = 1
    else:
        raise ValueError(f"unknown objective {kind!r}")
    if budget is not None and budget.size != 1:
        raise ValueError("one budget (a single zone)")

    nrow = nt + nrow_extra
    ncol = n + ncol_extra
    a = sp.coo_matrix(blocks[0])
    rows, cols, vals = [a.row.astype(np.int64)], [a.col.astype(np.int64)], [a.data]
    for r, c, v in extra_rows:
        rows.append(np.asarray(r, dtype=np.int64))
        cols.append(np.asarray(c, dtype=np.int64))
        vals.append(np.asarray(v, dtype=np.float64))
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    keep = v != 0.0  # prioritizr pushes zero costs and 0 = -1/target for targets <= 1e-5: no entry here
    A = sp.csr_matrix((v[keep], (r[keep], c[keep])), shape=(nrow, ncol))
    A.sort_indices()

    obj_v = np.concatenate(obj)
    if p.weights is not None:
        w = np.asarray(p.weights, dtype=np.float64)
        if kind in NO_WEIGHTS:
            warnings.warn(f"feature weights are ignored by the {kind} objective", UserWarning, stacklevel=3)
        else:
            if p.targets is not None and w.size != nt:
                raise ValueError(f"the weights need one value per target ({nt})")
            if kind == "min_shortfall":  # its default weights: 1 / target (R: 1 / 0 = Inf -> 0)
                with np.errstate(divide="ignore"):
                    d = 1.0 / t.value
                d[~np.isfinite(d)] = 0.0
                w = w * d
            obj_v[weight_cols:weight_cols + w.size] = w
    lb_v, ub_v = np.concatenate(lb), np.concatenate(ub)
    # locked units (constraints come last in prioritizr's compile)
    li, lo = p._mask(p.locked_in), p._mask(p.locked_out)
    if (li & lo).any():
        raise ValueError(f"{int((li & lo).sum())} planning units are both locked in and locked out")
    lb_v[:n][li] = 1.0
    ub_v[:n][lo] = 0.0
    return OptimizationProblem(
        A=A, obj=obj_v, lb=lb_v, ub=ub_v, rhs=np.concatenate([np.asarray(x, dtype=np.float64) for x in rhs]),
        sense=tuple(sense), vtype=tuple(vtype), modelsense="max" if kind.startswith("max") else "min",
        row_ids=tuple(row_ids), col_ids=tuple(col_ids)), t
