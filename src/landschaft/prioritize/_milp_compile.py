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
NO_TARGETS = ("max_utility", "max_cover", "min_extinction_risk")
# objectives whose model ignores feature weights (prioritizr warns)
NO_WEIGHTS = ("min_set", "min_largest_shortfall")
PHYLO = ("max_phylo_div", "max_phylo_end")
BUDGETED = ("min_shortfall", "min_largest_shortfall", "max_utility", "max_features", "max_cover", "min_penalties",
            "min_extinction_risk", *PHYLO)


def sar_lines(z: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """The chords of g(r) = r^z through r_j = (j / segments)^(1 / z), j = 0..segments (equal steps of g):
    intercepts c_k and slopes s_k. g is concave, so the interpolation is the least of the lines."""
    j = np.arange(segments + 1, dtype=np.float64)
    r = (j / segments) ** (1.0 / z)
    g = j / segments
    s = np.diff(g) / np.diff(r)
    return g[:-1] - s * r[:-1], s


@dataclass(frozen=True)
class Targets:
    """The targets as the model's rows: feature (0-based), sense and amount of each, the total that
    relative targets are shares of, and, for sub-regional targets, the region (code, -1: the whole
    problem) and its label."""

    feature: np.ndarray
    sense: tuple[str, ...]
    value: np.ndarray
    total: np.ndarray | None = None
    region: np.ndarray | None = None
    region_label: tuple | None = None
    matrix: object = None  # the targets' amounts in the planning units (targets x units), once compiled
    zone: tuple | None = None  # with zones: every target's zones (indices), whose amounts it adds up
    zone_label: tuple | None = None  # ... and their names ("protect + use")

    @property
    def n(self) -> int:
        return self.feature.size


def _concat(a: Targets | None, b: Targets) -> Targets:
    if a is None:
        return b
    n = a.n
    ra = a.region if a.region is not None else np.full(n, -1, dtype=np.int64)
    rb = b.region if b.region is not None else np.full(b.n, -1, dtype=np.int64)
    la = a.region_label if a.region_label is not None else (None,) * n
    lb = b.region_label if b.region_label is not None else (None,) * b.n
    zone = None if a.zone is None else a.zone + b.zone
    zlabel = None if a.zone_label is None else a.zone_label + b.zone_label
    return Targets(np.concatenate([a.feature, b.feature]), a.sense + b.sense, np.concatenate([a.value, b.value]),
                   np.concatenate([a.total, b.total]), np.concatenate([ra, rb]), la + lb, zone=zone, zone_label=zlabel)


def region_codes(u, regions) -> tuple[np.ndarray, list]:
    """The region of every unit, planning unit or not (raster: every cell; tables and polygons: every
    row), as codes into the sorted region ids (-1: none), and the ids."""
    import xarray as xr

    if u.kind == "raster":
        if isinstance(regions, xr.DataArray):
            v = np.asarray(regions.squeeze().values, dtype=np.float64)
        else:
            v = np.asarray(regions, dtype=np.float64)
        if v.shape != u.template.shape:
            raise ValueError(f"the regions ({v.shape}) must be on the planning units' grid ({u.template.shape})")
        v = v.ravel()
        ok = np.isfinite(v)
        ids = np.unique(v[ok])
        codes = np.full(v.size, -1, dtype=np.int64)
        codes[ok] = np.searchsorted(ids, v[ok])
        return codes, ids.tolist()
    col = u.template[regions] if isinstance(regions, str) else pd.Series(np.asarray(regions))
    if len(col) != len(u.template):
        raise ValueError("the regions need one value per row of the planning units' table")
    ok = col.notna().to_numpy()
    vals = col[ok].to_numpy()
    ids = sorted(set(vals.tolist()))
    index = {r: k for k, r in enumerate(ids)}
    codes = np.full(len(col), -1, dtype=np.int64)
    codes[ok] = [index[r] for r in vals]
    return codes, ids


def region_totals(u, codes: np.ndarray, nreg: int, features: np.ndarray) -> np.ndarray:
    """Every feature's total in every region (features x regions), over all units with or without a
    cost, added in unit order (np.add.at), as the problem's totals are."""
    out = np.zeros((len(u.feature_names), nreg))
    need = np.unique(features)
    if u.kind != "raster":
        a = u.all_rij
        for f in need:
            lo, hi = a.indptr[f], a.indptr[f + 1]
            c = codes[a.indices[lo:hi]]
            v = a.data[lo:hi]
            keep = (c >= 0) & np.isfinite(v)
            if u.r_sums:  # as R's colSums of the column cut by the region
                from ._milp import _r_sum

                for r in range(nreg):
                    out[f, r] = _r_sum(v[keep & (c == r)])
            else:
                np.add.at(out[f], c[keep], v[keep])
        return out
    from ._milp import BLOCK_BYTES, _layers

    layers = _layers(u.features)
    ny, nx = u.template.shape
    step = max(1, BLOCK_BYTES // (8 * nx))
    for f in need:
        da = layers[f][1]
        for y0 in range(0, ny, step):
            y1 = min(ny, y0 + step)
            v = np.asarray(da.data[y0:y1], dtype=np.float64)
            if hasattr(v, "compute"):
                v = np.asarray(v.compute(), dtype=np.float64)
            v = v.ravel()
            c = codes[y0 * nx:y1 * nx]
            keep = (c >= 0) & np.isfinite(v)
            np.add.at(out[f], c[keep], v[keep])
    return out


def resolve_subregional(u, spec, names: tuple[str, ...]) -> Targets:
    regions, t = spec
    codes, ids = region_codes(u, regions)
    numeric = all(isinstance(r, (int, float, np.integer, np.floating)) for r in ids)
    index = {(float(r) if numeric else str(r)): k for k, r in enumerate(ids)}
    findex = {n: k for k, n in enumerate(names)}
    missing = sorted(set(map(str, t["feature"])) - set(findex))
    if missing:
        raise ValueError(f"unknown features in the targets: {missing}")
    want = [float(r) if numeric else str(r) for r in t["region"]]
    unknown = sorted({str(r) for r in want if r not in index})
    if unknown:
        raise ValueError(f"unknown regions in the targets: {unknown}")
    region = np.array([index[r] for r in want], dtype=np.int64)
    feature = np.array([findex[str(f)] for f in t["feature"]], dtype=np.int64)
    target = t["target"].to_numpy(dtype=np.float64)
    kinds = t["type"].astype(str).to_numpy() if "type" in t.columns else np.full(len(t), "relative")
    if not np.isin(kinds, ("relative", "absolute")).all():
        raise ValueError("target types must be 'relative' or 'absolute'")
    sense = tuple(t["sense"].astype(str)) if "sense" in t.columns else (">=",) * len(t)
    if not set(sense) <= {">=", "<=", "="}:
        raise ValueError("target senses must be '>=', '<=' or '='")
    tot = region_totals(u, codes, len(ids), feature)[feature, region]
    value = np.where(kinds == "relative", tot * target, target)
    unit_codes = codes[u.index]
    return Targets(feature, sense, value, tot, region, tuple(ids[k] for k in region)), unit_codes


def _r_row_sum(v: np.ndarray) -> float:
    """R's rowSums() of a row of a dense matrix (an 80-bit accumulation)."""
    from ._milp import _r_sum

    return _r_sum(np.asarray(v, dtype=np.float64))


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
        return Targets(np.arange(nf), (">=",) * nf, totals * spec[1], totals.copy())
    if kind == "absolute":
        return Targets(np.arange(nf), (">=",) * nf, np.asarray(spec[1], dtype=np.float64).copy(), totals.copy())
    if kind == "loglinear":
        lo_a, lo_t, hi_a, hi_t, cap_a, cap_t, abundances = spec[1]
        a = totals if abundances is None else abundances
        value = a * loglinear(a, lo_a, lo_t, hi_a, hi_t)
        if cap_a is not None:
            value[a >= cap_a] = cap_t
        return Targets(np.arange(nf), (">=",) * nf, value, totals.copy())
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
        return Targets(feature, sense, value, totals[feature])
    raise ValueError(f"unknown targets {kind!r}")


def compile_problem(p):
    """The model prioritizr's ``compile()`` gives for the problem ``p``."""
    from ._milp import OptimizationProblem

    u = p.units
    if p.objective is None:
        raise ValueError("the problem has no objective: add one (add_min_set_objective, ...)")
    kind = p.objective[0]
    nf = len(u.feature_names)
    zoned = u.zoned
    if zoned:
        from ._zones import (
            check_zone_problem,
            fake_zone_targets,
            positive_abundances,
            resolve_zone_targets,
            sum_zone_targets,
            zone_target_matrix,
        )

        check_zone_problem(p)
    # the columns of the planning units: one per unit (and zone, zone after zone)
    n = u.cost.size
    if kind in NO_TARGETS:
        if p.targets is not None or p.zone_targets:
            warnings.warn(f"targets are ignored by the {kind} objective", UserWarning, stacklevel=3)
        # the "fake" targets prioritizr's compile() makes: one per feature (and zone); the extinction risk
        # (ours) of every feature over all its zones
        if not zoned:
            t = Targets(np.arange(nf), ("?",) * nf, np.zeros(nf), u.totals.copy())
        else:
            t = sum_zone_targets(u) if kind == "min_extinction_risk" else fake_zone_targets(u)
    elif zoned:
        if p.targets is None and not p.zone_targets and p.subregional is None:
            raise ValueError(f"the {kind} objective needs targets (add_relative_targets, add_absolute_targets, ...)")
        t = resolve_zone_targets(p) if p.targets is not None or p.zone_targets else None
    else:
        if p.targets is None and p.subregional is None:
            raise ValueError(f"the {kind} objective needs targets (add_relative_targets, add_absolute_targets, ...)")
        t = resolve_targets(p.targets, u.feature_names, u.totals) if p.targets is not None else None
    unit_codes = None
    if p.subregional is not None and kind not in NO_TARGETS:
        if zoned:
            from ._zones import resolve_zone_subregional

            sub, unit_codes = resolve_zone_subregional(p)
        else:
            sub, unit_codes = resolve_subregional(u, p.subregional, u.feature_names)
        t = _concat(t, sub)
    nt = t.n
    # a unit without a cost in a zone: cost 0 there and the column fixed at 0 (prioritizr's objectives)
    closed = ~np.isfinite(u.cost)
    cost = np.where(closed, 0.0, u.cost) if zoned else u.cost

    # decisions: the planning units' columns
    lb = [np.zeros(n)]
    if p.decisions[0] == "binary":
        ub, vtype = [np.ones(n)], ["B"] * n
    elif p.decisions[0] == "proportion":
        ub, vtype = [np.ones(n)], ["C"] * n
    else:
        ub, vtype = [np.full(n, p.decisions[1])], ["C"] * n

    # the targets' rows (the features' amounts) and the objective's rows and columns
    if zoned:
        rij = zone_target_matrix(u, t)
    else:
        rij = u.rij[t.feature] if not np.array_equal(t.feature, np.arange(nf)) else u.rij
    if t.region is not None and (t.region >= 0).any():
        # a sub-regional target: its feature's amounts in the units of its region only
        rij = sp.csr_matrix(rij, copy=True)
        row = np.repeat(np.arange(t.n), np.diff(rij.indptr))
        r = t.region[row]
        keep = (r < 0) | (unit_codes[rij.indices % u.n] == r)
        rij.data[~keep] = 0.0
        rij.eliminate_zeros()
    import dataclasses

    t = dataclasses.replace(t, matrix=sp.csr_matrix(rij))
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
    # one budget over every zone, or one per zone (a row each, zone after zone)
    nb = 0 if budget is None else budget.size
    bud_sense, bud_ids = ["<="] * nb, ["budget"] * nb

    def budget_row(row: int):
        if nb == 1:
            extra_rows.append((np.full(n, row), np.arange(n), cost))
        else:
            extra_rows.append((row + np.repeat(np.arange(nb), u.n), np.arange(n), cost))

    def positive_amounts() -> np.ndarray:
        """Every target's positive amount in the planning units (Matrix's rowSums, in unit order)."""
        if zoned:
            return positive_abundances(u, t)
        pos = u.rij.copy()
        pos.data = np.maximum(pos.data, 0.0)
        return np.array([_sum(pos.data[pos.indptr[i]:pos.indptr[i + 1]]) for i in range(nf)])

    if kind == "min_set":
        obj = [cost.copy()]
        sense, rhs, row_ids = list(t.sense), [t.value], ["spp_target"] * nt
    elif kind == "min_shortfall":
        sense = [*t.sense, *bud_sense]
        rhs = [t.value, budget]
        obj.append(np.where(t.value > 1.0e-5, 1.0 / np.where(t.value > 1.0e-5, t.value, 1.0), 0.0))
        lb.append(np.zeros(nt))
        ub.append(np.full(nt, np.inf))
        vtype += ["C"] * nt
        extra_rows.append((np.arange(nt), n + np.arange(nt), np.ones(nt)))
        budget_row(nt)
        col_ids += ["spp_met"] * nt
        row_ids = ["spp_target"] * nt + bud_ids
        nrow_extra, ncol_extra, weight_cols = nb, nt, n
    elif kind == "min_largest_shortfall":
        sense = [*t.sense, *([">="] * nt), *bud_sense]
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
        row_ids = ["spp_target"] * nt + ["max_shortfall"] * nt + bud_ids
        nrow_extra, ncol_extra = nt + nb, nt + 1
    elif kind in ("max_utility", "max_cover", "max_features"):
        scale = -0.01 / _sum(cost)
        obj = [cost * scale, np.ones(nt)]
        if kind == "max_utility":
            abund = positive_amounts()
            negative = u.rij.nnz > 0 and float(u.rij.data.min()) < 0
            lb.append(np.full(nt, -np.inf if negative else 0.0))
            ub.append(np.maximum(abund, 0.0))
            vtype += ["C"] * nt
            extra_rows.append((np.arange(nt), n + np.arange(nt), -np.ones(nt)))
            sense, rhs = ["="] * nt + bud_sense, [np.zeros(nt), budget]
            col_ids += ["amount"] * nt
            row_ids = ["spp_amount"] * nt + bud_ids
        elif kind == "max_cover":
            lb.append(np.zeros(nt))
            ub.append(np.ones(nt))
            vtype += ["B"] * nt
            extra_rows.append((np.arange(nt), n + np.arange(nt), -np.ones(nt)))
            sense, rhs = [">="] * nt + bud_sense, [np.zeros(nt), budget]
            col_ids += ["present"] * nt
            row_ids = ["spp_present"] * nt + bud_ids
        else:
            lb.append(np.zeros(nt))
            ub.append(np.ones(nt))
            vtype += ["B"] * nt
            extra_rows.append((np.arange(nt), n + np.arange(nt), -1.0 * t.value))
            sense, rhs = [*t.sense, *bud_sense], [np.zeros(nt), budget]
            col_ids += ["spp_met"] * nt
            row_ids = ["spp_target"] * nt + bud_ids
        budget_row(nt)
        nrow_extra, ncol_extra, weight_cols = nb, nt, n
    elif kind in PHYLO:
        bm, lengths = tree_branches(p.objective[2], u.feature_names)
        if kind == "max_phylo_end":  # with zones, over R's rowSums of the zones' totals
            totals = np.array([_r_row_sum(r) for r in u.totals]) if zoned else u.totals
            lengths = endemism_lengths(bm, lengths, totals)
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
            br_rows += [nt + nb + e] * (ts.size + 1)
            br_cols += [*(n + ts), n + nt + e]
        br_vals = np.ones(len(br_rows))
        br_vals[np.cumsum([below.indptr[e + 1] - below.indptr[e] + 1 for e in range(ne)]) - 1] = -1.0
        extra_rows.append((np.array(br_rows), np.array(br_cols), br_vals))
        sense, rhs = [*t.sense, *bud_sense, *([">="] * ne)], [np.zeros(nt), budget, np.zeros(ne)]
        col_ids += ["spp_met"] * nt + ["branch_met"] * ne
        row_ids = ["spp_target"] * nt + bud_ids + ["branch_target"] * ne
        nrow_extra, ncol_extra, weight_cols = nb + ne, nt + ne, n
    elif kind == "min_extinction_risk":
        # risk_f >= 1 - (c_k + s_k amount_f / total_f) for every chord k of the species-area curve
        # (amount_f / total_f)^z: the least risk is 1 - the piecewise-linear r^z (Pereira & Daily 2006);
        # the amounts as in max utility, then the risks
        z_all, segments = np.broadcast_to(np.asarray(p.objective[2], dtype=np.float64), (nf,)), int(p.objective[3])
        abund = positive_amounts()
        scale = 0.01 / _sum(cost)  # the cheaper of equal risks
        obj = [cost * scale, np.zeros(nt), np.ones(nt)]
        lb.append(np.zeros(2 * nt))
        ub.append(np.concatenate([abund, np.ones(nt)]))
        vtype += ["C"] * (2 * nt)
        extra_rows.append((np.arange(nt), n + np.arange(nt), -np.ones(nt)))
        sar_r, sar_c, sar_v, sar_rhs = [], [], [], []
        row = nt
        for f in range(nt):
            tot = float(t.total[f])
            c_k, s_k = sar_lines(float(z_all[f]), segments)
            for k in range(segments):
                if tot > 0:
                    sar_r += [row, row]
                    sar_c += [n + f, n + nt + f]
                    sar_v += [s_k[k] / tot, 1.0]
                    sar_rhs.append(1.0 - c_k[k])
                else:  # a feature with nothing: no risk to reduce
                    sar_r.append(row)
                    sar_c.append(n + nt + f)
                    sar_v.append(1.0)
                    sar_rhs.append(0.0)
                row += 1
        extra_rows.append((np.array(sar_r), np.array(sar_c), np.array(sar_v)))
        budget_row(nt + nt * segments)
        sense = ["="] * nt + [">="] * (nt * segments) + bud_sense
        rhs = [np.zeros(nt), np.array(sar_rhs), budget]
        col_ids += ["amount"] * nt + ["extinction_risk"] * nt
        row_ids = ["spp_amount"] * nt + ["extinction_risk"] * (nt * segments) + bud_ids
        nrow_extra, ncol_extra, weight_cols = nt * segments + nb, 2 * nt, n + nt
    elif kind == "min_penalties":
        sense, rhs = [*t.sense, *bud_sense], [t.value, budget]
        budget_row(nt)
        row_ids = ["spp_target"] * nt + bud_ids
        nrow_extra = nb
    else:
        raise ValueError(f"unknown objective {kind!r}")

    nrow = nt + nrow_extra
    ncol = n + ncol_extra
    if zoned:
        # every unit goes to one zone at most (exactly one with mandatory allocation): after the objective's
        # rows (rcpp_add_zones_constraints)
        nu, nz = u.n, u.nz
        extra_rows.append((nrow + np.repeat(np.arange(nu), nz), (np.arange(nz)[None, :] * nu + np.arange(nu)[:, None]).ravel(),
                           np.ones(nu * nz)))
        sense = [*sense, *(["="] * nu if p.mandatory else ["<="] * nu)]
        rhs = [*rhs, np.ones(nu)]
        row_ids = [*row_ids, *(["pu_zone"] * nu)]
        nrow += nu
        obj[0] = np.where(closed, 0.0, obj[0])
    a = sp.coo_matrix(blocks[0])
    rows, cols, vals = [a.row.astype(np.int64)], [a.col.astype(np.int64)], [a.data]
    for r, c, v in extra_rows:
        rows.append(np.asarray(r, dtype=np.int64))
        cols.append(np.asarray(c, dtype=np.int64))
        vals.append(np.asarray(v, dtype=np.float64))

    obj_v = np.concatenate(obj)
    if p.weights is not None:
        w = np.asarray(p.weights, dtype=np.float64)
        if kind in NO_WEIGHTS:
            warnings.warn(f"feature weights are ignored by the {kind} objective", UserWarning, stacklevel=3)
        else:
            has_targets = p.targets is not None or bool(p.zone_targets)
            if zoned and not (has_targets and kind not in NO_TARGETS):  # every feature in every zone
                if w.ndim == 2:
                    if w.shape != (nf, u.nz):
                        raise ValueError(f"the weights need one value per feature and zone ({nf} x {u.nz})")
                    w = w.T.ravel()  # zone after zone (prioritizr's c(matrix))
                elif kind != "min_extinction_risk" and w.size == nf:
                    w = np.tile(w, u.nz)  # the same weight in every zone
            w = w.ravel()
            if has_targets and kind not in NO_TARGETS and w.size != nt:
                raise ValueError(f"the weights need one value per target ({nt})")
            if w.size != nt:
                raise ValueError(f"the weights need {nt} values")
            if kind == "min_shortfall":  # its default weights: 1 / target (R: 1 / 0 = Inf -> 0)
                with np.errstate(divide="ignore"):
                    d = 1.0 / t.value
                d[~np.isfinite(d)] = 0.0
                w = w * d
            obj_v[weight_cols:weight_cols + w.size] = w
    lb_v, ub_v = np.concatenate(lb), np.concatenate(ub)
    modelsense = "max" if kind.startswith("max") else "min"
    if any(sp_[0] == "constraint" and sp_[1] == "feature_contiguity" for sp_ in p.spatial):
        # the expanded formulation (rcpp_add_rij_data without compression): a column per feature and
        # unit (y_fj <= x_j, feature-major) after the units, nf x n rows y_fj - x_j <= 0 first, and the
        # targets on the y columns; the objective's rows and columns move after them
        if u.rij.nnz and float(u.rij.data.min()) < 0:
            raise ValueError("the expanded formulation (feature contiguity) needs non-negative feature data")
        ny_ = nf * n
        # the targets' entries (rows 0..nt-1, unit columns) go to their feature's y columns
        cols = [n + t.feature[rows[0]] * n + cols[0]] + [np.where(c >= n, c + ny_, c) for c in cols[1:]]
        rows = [r + ny_ for r in rows]
        link_r = np.repeat(np.arange(ny_), 2)
        link_c = np.column_stack([np.tile(np.arange(n), nf), n + np.arange(ny_)]).ravel()
        link_v = np.tile([-1.0, 1.0], ny_)
        rows.insert(0, link_r)
        cols.insert(0, link_c)
        vals.insert(0, link_v)
        obj_v = np.concatenate([obj_v[:n], np.zeros(ny_), obj_v[n:]])
        lb_v = np.concatenate([lb_v[:n], np.tile(lb_v[:n], nf), lb_v[n:]])
        ub_v = np.concatenate([ub_v[:n], np.tile(ub_v[:n], nf), ub_v[n:]])
        vtype = vtype[:n] + vtype[:n] * nf + vtype[n:]
        col_ids = col_ids[:n] + ["pu_ijz"] * ny_ + col_ids[n:]
        row_ids = ["pu_ijz"] * ny_ + row_ids
        sense = ["<="] * ny_ + sense
        rhs = [np.zeros(ny_)] + rhs
        nrow += ny_
        ncol += ny_
    # penalties, then constraints, each in the order they were added (prioritizr's compile)
    from ._milp_spatial import Builder, apply_constraint, apply_penalty

    b = Builder(nrow, ncol, rows, cols, vals, obj_v, lb_v, ub_v, vtype,
                sense, [np.asarray(x, dtype=np.float64) for x in rhs], row_ids, col_ids, modelsense, n)
    b.n_features = nf
    if zoned:
        from ._zones_spatial import apply_zone_constraint as apply_constraint
        from ._zones_spatial import apply_zone_penalty as apply_penalty
    for spec in p.spatial:
        if spec[0] == "penalty":
            apply_penalty(b, u, spec[1:])
    for spec in p.spatial:
        if spec[0] == "constraint":
            apply_constraint(b, u, spec[1:])
    A = b.matrix()
    obj_v, lb_v, ub_v = b.obj, b.lb, b.ub
    vtype, sense, row_ids, col_ids = b.vtype, b.sense, b.row_ids, b.col_ids
    rhs = b.rhs
    if zoned:  # the locked units and manual bounds of every unit and zone, the zones without a cost closed
        lb_v[:n], ub_v[:n] = p._unit_bounds()
        return OptimizationProblem(
            A=A, obj=obj_v, lb=lb_v, ub=ub_v, rhs=np.concatenate([np.asarray(x, dtype=np.float64) for x in rhs]),
            sense=tuple(sense), vtype=tuple(vtype), modelsense=modelsense, row_ids=tuple(row_ids),
            col_ids=tuple(col_ids), connected=tuple(b.connected), corridors=tuple(b.corridors)), t
    # locked units (bounds only: their place among the constraints does not matter)
    li, lo = p._mask(p.locked_in), p._mask(p.locked_out)
    if (li & lo).any():
        raise ValueError(f"{int((li & lo).sum())} planning units are both locked in and locked out")
    lb_v[:n][li] = 1.0
    ub_v[:n][lo] = 0.0
    if p.bounds:  # manual bounded / locked constraints, in the order added (after the locked units)
        limit = 1.0 if p.decisions[0] != "semicontinuous" else float(p.decisions[1])
        p._manual_bounds(lb_v[:n], ub_v[:n], p.decisions[0] == "binary", limit)
    return OptimizationProblem(
        A=A, obj=obj_v, lb=lb_v, ub=ub_v, rhs=np.concatenate([np.asarray(x, dtype=np.float64) for x in rhs]),
        sense=tuple(sense), vtype=tuple(vtype), modelsense=modelsense,
        row_ids=tuple(row_ids), col_ids=tuple(col_ids), connected=tuple(b.connected), corridors=tuple(b.corridors)), t
