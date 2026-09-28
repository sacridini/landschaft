# SPDX-License-Identifier: GPL-3.0-or-later
"""Randomisations of a spatial analysis (Biodiverse::Randomise).

Each iteration randomises the basedata, runs the analysis again with the same
neighbourhoods and counts, per group and index, how often the observed value
exceeds (``C``), ties (``T``) and is compared with (``Q``) the randomised one,
with the sums needed for z-scores. The counts give the p-ranks, z-scores and
CANAPE categories, as in Biodiverse.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pandas as pd

from .. import _core
from ._spatial import _KEY_GROUP_ARRAY, _KEY_NUMBER_ARRAY, SpatialResult, _evaluate

_EPS = 1e-10  # Biodiverse's DEFAULT_PRECISION_SMALL
_COUNTERS = ("C", "Q", "T", "SUMX", "SUMXX")


def _with_counts(bd, counts):
    out = copy.copy(bd)
    counts = counts.tocsr()
    counts.sort_indices()
    out.counts = counts
    return out


def rand_nochange(bd, rng, **_):
    """The basedata unchanged."""
    return bd


def rand_csr_by_group(bd, rng, **_):
    """Complete spatial randomisation: whole assemblages move to random groups."""
    perm = rng.shuffle_indices(bd.n_groups)  # group i's labels go to group perm[i]
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    return _with_counts(bd, bd.counts[inverse])


def _has_text(cond) -> bool:
    """Whether a condition has more than whitespace and comments."""
    import re

    if cond is None:
        return False
    text = cond if isinstance(cond, str) else getattr(cond, "text", str(cond))
    lines = [re.sub(r"#.*$", "", ln).strip() for ln in text.splitlines()]
    return any(lines)


def _spatial_alloc(bd, state: dict, conditions, order: str, backtracking: str, reseed_prob: float, seed_condition):
    """The allocation state (neighbour lists, caches), built once per randomisation."""
    if "spatial_alloc" in state:
        return state["spatial_alloc"]
    from ._conditions import SpatialCondition, _as_condition, definition_query, neighbours
    from ._names import split_element

    alloc = None
    if _has_text(conditions if isinstance(conditions, (str, SpatialCondition)) else (conditions or [None])[0]):
        conds = [conditions] if isinstance(conditions, (str, SpatialCondition)) else list(conditions)
        nb = neighbours(bd, conds)
        stored = nb._stored
        rows = {int(g): r for r, g in enumerate(nb.groups)}
        lists = []
        for g in range(bd.n_groups):
            r = rows.get(g)
            per_set = []
            for ip, ix in stored:
                per_set.append([] if r is None else sorted(int(x) for x in ix[ip[r]:ip[r + 1]]))
            lists.append(per_set)
        types = [_as_condition(c).result_type for c in conds]
        same = all(t.startswith(("always_same", "always_true")) for t in types)
        coords = []
        for name in bd.groups:
            vals = []
            for v in split_element(name):
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(0.0)
            coords.append(vals)
        have_seeds = False
        seeds: list[int] = []
        if _has_text(seed_condition):
            sc = _as_condition(seed_condition)
            if sc.result_type != "always_false":
                have_seeds = True
                seeds = [int(i) for i in np.flatnonzero(definition_query(bd, sc))]
        alloc = _core.BioSpatialAlloc(order or "", backtracking or "", float(reseed_prob or 0), lists, same, coords,
                                      have_seeds, seeds)
    state["spatial_alloc"] = alloc
    return alloc


def rand_structured(bd, rng, richness_multiplier: float = 1, richness_addition: float = 0,
                    spatial_conditions_for_label_allocation=None, spatial_allocation_order: str = "",
                    label_allocation_backtracking: str = "", spatial_allocation_reseed_prob: float = 0,
                    spatial_conditions_for_seed_location=None, _state: dict | None = None, **_):
    """Labels to random groups until each reaches its richness target, then swaps.

    With ``spatial_conditions_for_label_allocation``, occurrences go to the
    neighbours of a seed group first (see :func:`rand_spatially_structured`).
    """
    import scipy.sparse as sp

    state = _state if _state is not None else {}
    alloc = _spatial_alloc(bd, state, spatial_conditions_for_label_allocation, spatial_allocation_order,
                           label_allocation_backtracking, spatial_allocation_reseed_prob,
                           spatial_conditions_for_seed_location)
    c = bd.counts
    indptr, indices, data = _core.bio_rand_structured(
        np.ascontiguousarray(c.indptr, dtype=np.int64), np.ascontiguousarray(c.indices, dtype=np.int32),
        np.ascontiguousarray(c.data, dtype=np.float64), bd.n_groups, bd.n_labels, rng,
        float(richness_multiplier), float(richness_addition or 0), alloc)
    return _with_counts(bd, sp.csr_matrix((data, indices, indptr), shape=c.shape))


_DEFAULT_ALLOCATION = "sp_select_all()"


def rand_spatially_structured(bd, rng, spatial_conditions_for_label_allocation=_DEFAULT_ALLOCATION,
                              spatial_allocation_order: str = "diffusion",
                              label_allocation_backtracking: str = "from_end", **args):
    """rand_structured allocating label occurrences around seed groups (diffusion by default)."""
    return rand_structured(bd, rng, spatial_conditions_for_label_allocation=spatial_conditions_for_label_allocation,
                           spatial_allocation_order=spatial_allocation_order,
                           label_allocation_backtracking=label_allocation_backtracking, **args)


def rand_diffusion(bd, rng, spatial_conditions_for_label_allocation=_DEFAULT_ALLOCATION, **args):
    """rand_spatially_structured with the diffusion allocation order."""
    args.pop("spatial_allocation_order", None)
    args.setdefault("label_allocation_backtracking", "")
    return rand_structured(bd, rng, spatial_conditions_for_label_allocation=spatial_conditions_for_label_allocation,
                           spatial_allocation_order="diffusion", **args)


def rand_random_walk(bd, rng, spatial_conditions_for_label_allocation=_DEFAULT_ALLOCATION,
                     label_allocation_backtracking: str = "from_end", **args):
    """rand_spatially_structured with the random walk allocation order."""
    args.pop("spatial_allocation_order", None)
    return rand_structured(bd, rng, spatial_conditions_for_label_allocation=spatial_conditions_for_label_allocation,
                           spatial_allocation_order="random_walk",
                           label_allocation_backtracking=label_allocation_backtracking, **args)


def _swaps(kind: str, bd, rng, swap_count=0, max_swap_attempts=0, stop_on_all_swapped=False, use_max_swap=False,
           use_hypergeometric=False, spatial_condition_for_swap_pairs=None, _state=None):
    import scipy.sparse as sp

    from ._spatial import _basedata_view

    if use_hypergeometric:
        raise NotImplementedError("use_hypergeometric is not supported yet")
    if spatial_condition_for_swap_pairs is not None:
        raise NotImplementedError("spatial_condition_for_swap_pairs is not supported yet")
    c = bd.counts
    ranges = np.ascontiguousarray(_basedata_view(bd)["label_range"], dtype=np.float64)
    indptr, indices, data = _core.bio_rand_swaps(
        np.ascontiguousarray(c.indptr, dtype=np.int64), np.ascontiguousarray(c.indices, dtype=np.int32),
        np.ascontiguousarray(c.data, dtype=np.float64), bd.n_groups, bd.n_labels, rng, ranges, kind,
        int(swap_count or 0), int(max_swap_attempts or 0), bool(stop_on_all_swapped), bool(use_max_swap))
    return _with_counts(bd, sp.csr_matrix((data, indices, indptr), shape=c.shape))


def rand_independent_swaps(bd, rng, **args):
    """Independent swaps (Gotelli 2000; Miklos & Podani 2004): keeps richness and ranges."""
    return _swaps("independent_swaps", bd, rng, **args)


def rand_independent_swaps_modified(bd, rng, **args):
    """Independent swaps drawing labels by range and target groups by richness (fewer misses)."""
    return _swaps("independent_swaps_modified", bd, rng, **args)


def rand_curveball(bd, rng, **args):
    """Curveball algorithm (Strona et al. 2014): keeps richness and ranges."""
    return _swaps("curveball", bd, rng, **args)


FUNCTIONS: dict[str, Callable] = {
    "rand_nochange": rand_nochange,
    "rand_csr_by_group": rand_csr_by_group,
    "rand_structured": rand_structured,
    "rand_spatially_structured": rand_spatially_structured,
    "rand_diffusion": rand_diffusion,
    "rand_random_walk": rand_random_walk,
    "rand_independent_swaps": rand_independent_swaps,
    "rand_independent_swaps_modified": rand_independent_swaps_modified,
    "rand_curveball": rand_curveball,
}


def _subset_bd(bd, groups: np.ndarray, labels: np.ndarray | None = None, group_props: bool = True):
    """A basedata of some groups (and labels), as Biodiverse builds subsets: no label properties."""
    from ._basedata import BaseData

    groups = np.asarray(groups, dtype=np.int64)
    rows = bd.counts[groups]
    if labels is None:
        labels = np.unique(rows.indices)
    labels = np.asarray(labels, dtype=np.int64)
    sub = rows[:, labels]
    out = BaseData(sub, [bd.groups[g] for g in groups], [bd.labels[lb] for lb in labels], bd.coords[groups],
                   bd.cell_sizes, bd.cell_origins, bd.crs)
    if group_props and len(bd.group_properties.columns):
        out.group_properties = bd.group_properties.iloc[groups].copy()
    return out, groups, labels


def _group_props(orig: pd.DataFrame, groups: list[str], rng, how: str | None,
                 current: pd.DataFrame | None = None) -> pd.DataFrame:
    """process_group_props: the group properties of a randomised basedata.

    ``no_change`` keeps them; ``by_set`` moves each group's properties
    together to a random group; ``by_item`` moves each property's values
    independently. ``current`` holds what the receiving basedata had.
    """
    how = (how or "no_change").removeprefix("process_group_props_")
    keys = sorted(c for c in orig.columns if orig[c].notna().any())
    if how == "no_change" or not keys:
        return orig.reindex(groups)
    n = len(groups)
    if how == "by_set":
        out = (current.reindex(groups) if current is not None
               else pd.DataFrame(None, index=groups, columns=orig.columns, dtype=object))
        perm = rng.shuffle_indices(n)
        rows = orig.reindex(groups)
        for i, element in enumerate(groups):
            vals = rows.iloc[i]
            if vals.isna().all():
                continue  # no properties
            out.iloc[int(perm[i])] = vals.to_numpy(dtype=object)
        return out
    if how == "by_item":
        out = pd.DataFrame(None, index=groups, columns=orig.columns, dtype=object)
        rows = orig.reindex(groups)
        for key in keys:
            perm = rng.shuffle_indices(n)
            col = rows[key].to_numpy(dtype=object)
            j = out.columns.get_loc(key)
            for i in range(n):
                if col[i] is not None and not (isinstance(col[i], float) and np.isnan(col[i])):
                    out.iat[int(perm[i]), j] = col[i]
        return out
    raise ValueError(f"unknown randomise_group_props_by {how!r} (no_change, by_set or by_item)")


def _place(n_groups: int, n_labels: int, parts) -> "sp.csr_matrix":
    """Merge randomised subsets (basedata, group indices, label indices) into full counts."""
    import scipy.sparse as sp

    rows, cols, vals = [], [], []
    for sbd, g_idx, l_idx in parts:
        c = sbd.counts.tocoo()
        rows.append(g_idx[c.row])
        cols.append(l_idx[c.col])
        vals.append(c.data)
    if not rows:
        return sp.csr_matrix((n_groups, n_labels))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals)
    m = sp.csr_matrix((v, (r, c)), shape=(n_groups, n_labels))
    m.sort_indices()
    return m


def _randomise_subsets(bd, rng, fn, state: dict, args: dict, conditions, def_query, gp_how=None):
    """get_rand_structured_subset: each subset of groups randomised on its own, then merged."""
    from ._conditions import neighbours

    key = "subsets"
    if key not in state:
        cond = conditions if isinstance(conditions, str) or conditions is None else (list(conditions) or [None])[0]
        if not _has_text(cond):
            cond = "sp_select_all()"
        dq = def_query if _has_text(def_query) else None
        nb = neighbours(bd, [cond], dq)
        ip, ix = nb._stored[0]
        stored = {int(g): [int(x) for x in ix[ip[r]:ip[r + 1]]] for r, g in enumerate(nb.groups)}
        passed = set(stored)
        failed = [g for g in range(bd.n_groups) if g not in passed] if dq is not None else []
        state[key] = (stored, failed, {})
    stored, failed, cache = state[key]
    parts = []
    done = np.zeros(bd.n_groups, bool)
    props = []  # (group indices, properties)
    if failed:
        if "failed" not in cache:
            fbd = _subset_bd(bd, np.array(failed))
            cache["failed"] = fbd
            cache["failed_props"] = None  # Biodiverse builds it without properties
        fbd, fg, fl = cache["failed"]
        parts.append((fbd, fg, fl))  # not randomised
        cache["failed_props"] = _group_props(bd.group_properties, fbd.groups, rng, gp_how, cache["failed_props"])
        props.append((fg, cache["failed_props"]))
        done[failed] = True
    failed_set = set(failed)
    for g in range(bd.n_groups):
        if done.all():
            break
        if g in failed_set:
            continue
        if g not in cache:
            nbrs = [x for x in stored.get(g, []) if not done[x]]
            if not nbrs:
                continue
            cache[g] = _subset_bd(bd, np.array(sorted(nbrs)))
        sbd, g_idx, l_idx = cache[g]
        rbd = fn(sbd, rng, _state={}, **args)
        parts.append((rbd, g_idx, l_idx))
        props.append((g_idx, _group_props(sbd.group_properties, sbd.groups, rng, gp_how)))
        done[g_idx] = True
    if not done.all():
        if "leftovers" not in cache:
            cache["leftovers"] = _subset_bd(bd, np.flatnonzero(~done))
        sbd, g_idx, l_idx = cache["leftovers"]
        rbd = fn(sbd, rng, _state={}, **args)
        parts.append((rbd, g_idx, l_idx))
        props.append((g_idx, _group_props(sbd.group_properties, sbd.groups, rng, gp_how)))
    out = _with_counts(bd, _place(bd.n_groups, bd.n_labels, parts))
    if len(bd.group_properties.columns):
        gp = pd.DataFrame(None, index=bd.groups, columns=bd.group_properties.columns, dtype=object)
        for g_idx, frame in props:
            gp.iloc[g_idx] = frame.to_numpy(dtype=object)
        gp.index.name = "group"
        out.group_properties = gp
    return out


class _ConstPart:
    """Counts of the labels kept in place (as a part to merge)."""

    def __init__(self, rows, cols, data, shape) -> None:
        import scipy.sparse as sp

        self.counts = sp.coo_matrix((data, (rows, cols)), shape=shape)


def _randomised_basedata(bd, rng, fn, state: dict, args: dict):
    """get_randomised_basedata: fixed labels, subsets, then the randomisation function."""
    args = dict(args)
    constant = args.pop("labels_not_to_randomise", None)
    gp_how = args.pop("randomise_group_props_by", None)
    args.pop("randomise_trees_by", None)
    subset_cond = args.pop("spatial_conditions_for_subset", None)
    def_query = args.pop("definition_query", None)
    use_subsets = _has_text(subset_cond if isinstance(subset_cond, (str, type(None))) else (subset_cond or [None])[0]) \
        or _has_text(def_query)

    def core(b, st):
        if use_subsets:
            return _randomise_subsets(b, rng, fn, st, args, subset_cond, def_query, gp_how)
        out = fn(b, rng, _state=st, **args)
        if len(b.group_properties.columns):
            out = copy.copy(out)
            gp = _group_props(b.group_properties, b.groups, rng, gp_how)
            gp.index.name = "group"
            out.group_properties = gp
        return out

    if not constant:
        return core(bd, state)
    if isinstance(constant, str):
        import re

        constant = [c for c in re.split(r"[\r\n]+", constant) if c]
    const = {bd._label_index[name] for name in constant if name in bd._label_index}
    lab_mask = np.zeros(bd.n_labels, bool)
    lab_mask[list(const)] = True
    c = bd.counts.tocsr()
    if "constant" not in state:
        # the other labels, in the groups that have any of them
        nz = c[:, ~lab_mask]
        g_idx = np.flatnonzero(np.diff(nz.indptr) > 0)
        l_idx = np.flatnonzero(~lab_mask & (np.diff(c.tocsc().indptr) > 0))
        nbd, g_idx, l_idx = _subset_bd(bd, g_idx, l_idx, group_props=False)
        cc = c.tocoo()
        keep = lab_mask[cc.col]
        const_part = _ConstPart(cc.row[keep], cc.col[keep], cc.data[keep], c.shape)
        state["constant"] = (nbd, g_idx, l_idx, const_part, {})
    nbd, g_idx, l_idx, const_part, st = state["constant"]
    rbd = core(nbd, st)
    all_g = np.arange(bd.n_groups)
    all_l = np.arange(bd.n_labels)
    out = _with_counts(bd, _place(bd.n_groups, bd.n_labels, [(rbd, g_idx, l_idx), (const_part, all_g, all_l)]))
    # Biodiverse's randomised basedata has no properties here
    out.label_properties = out.label_properties.iloc[:, :0]
    out.group_properties = out.group_properties.iloc[:, :0]
    return out


class _Counts:
    """Comparison counters of one list (or the scalar indices), aligned to the observed values."""

    def __init__(self, observed: np.ndarray) -> None:
        self.observed = observed
        shape = observed.shape
        self.c = {k: np.full(shape, np.nan) for k in _COUNTERS}

    def add(self, rand: np.ndarray) -> None:
        obs = self.observed
        ok = ~np.isnan(obs) & ~np.isnan(rand)
        diff = np.where(ok, obs - rand, 0.0)
        c = self.c
        for key, inc in (("C", (diff > _EPS).astype(float)), ("Q", np.ones_like(diff)), ("SUMX", rand),
                         ("SUMXX", rand * rand)):
            cur = c[key]
            c[key] = np.where(ok, np.where(np.isnan(cur), 0.0, cur) + np.where(ok, inc, 0.0), cur)
        tie = ok & (np.abs(diff) <= _EPS)
        c["T"] = np.where(tie, np.where(np.isnan(c["T"]), 0.0, c["T"]) + 1, c["T"])

    def p(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.c["C"] / self.c["Q"]

    def p_rank(self) -> np.ndarray:
        c, q, t = self.c["C"], self.c["Q"], self.c["T"]
        p = self.p()
        with np.errstate(invalid="ignore", divide="ignore"):
            low = (c + np.where(np.isnan(t), 0.0, t)) / q
        return np.where(np.isnan(c), np.nan, np.where(p > 0.5, p, low))

    def z_scores(self) -> np.ndarray:
        n, sx, sxx = self.c["Q"], self.c["SUMX"], self.c["SUMXX"]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (sxx - (sx * sx) / n) / n
            sd = _core.bio_perl_sqrt(np.ascontiguousarray(np.where(var > 0, var, 1.0).ravel())).reshape(var.shape)
            z = np.where(var > 0, (self.observed - sx / n) / sd, 0.0)
        return np.where(np.isnan(n) | (n == 0), np.nan, z)


class Randomisation:
    """Randomisations of a :class:`SpatialResult` (see :func:`randomise`).

    Attributes
    ----------
    result : SpatialResult
        The observed analysis.
    function : str
        Randomisation function.
    name : str
        Prefix of Biodiverse's result lists (``name>>SPATIAL_RESULTS``...).
    iterations : int
        Iterations run so far.
    """

    def __init__(self, result: SpatialResult, function: str, name: str, seed: int, args: dict) -> None:
        if function not in FUNCTIONS:
            raise ValueError(f"unknown randomisation {function!r} (available: {', '.join(sorted(FUNCTIONS))})")
        self.result = result
        self.function = function
        self.name = name
        self.seed = seed
        self.args = dict(args)
        self.iterations = 0
        self._rng = _core.BioRandom(int(seed) % (1 << 64))
        self._state: dict = {}  # caches a randomisation keeps across iterations
        self._scalars = _Counts(result.table.to_numpy(dtype=np.float64))
        self._lists: dict[str, tuple[pd.Index, _Counts]] = {}
        for lname in result.lists:
            kind = result.lists.raw(lname)[0]
            if kind in (_KEY_GROUP_ARRAY, _KEY_NUMBER_ARRAY):
                continue  # Biodiverse compares hash lists only
            df = result.lists[lname]
            idx = pd.MultiIndex.from_arrays([df["group"].to_numpy(), df["key"].to_numpy()])
            self._lists[lname] = (idx, _Counts(df["value"].to_numpy(dtype=np.float64)))

    def __repr__(self) -> str:
        return f"Randomisation({self.name!r}, {self.function}, {self.iterations} iterations)"

    def run(self, iterations: int = 1, n_threads: int | None = None) -> Randomisation:
        """Run more iterations (the random stream continues)."""
        from .._threads import resolve

        threads = resolve(n_threads)
        fn = FUNCTIONS[self.function]
        res = self.result
        for _ in range(int(iterations)):
            rand_bd = _randomised_basedata(res.bd, self._rng, fn, self._state, self.args)
            tree = self._shuffled_tree()
            table, lists = _evaluate(rand_bd, res._run, threads, tree=tree)
            self._scalars.add(table.reindex(columns=res.table.columns).to_numpy(dtype=np.float64))
            for lname, (idx, counts) in self._lists.items():
                df = lists[lname]
                r = pd.Series(df["value"].to_numpy(dtype=np.float64),
                              index=pd.MultiIndex.from_arrays([df["group"].to_numpy(), df["key"].to_numpy()]))
                counts.add(r.reindex(idx).to_numpy(dtype=np.float64))
            self.iterations += 1
        return self

    def _shuffled_tree(self):
        """The analysis tree for this iteration (``randomise_trees_by``)."""
        how = self.args.get("randomise_trees_by") or ""
        tree = self.result._run.tree
        if tree is None or not how or how.removeprefix("shuffle_") == "no_change":
            return tree
        if how.removeprefix("shuffle_") != "terminal_names":
            raise ValueError(f"unknown randomise_trees_by {how!r} (shuffle_no_change, shuffle_terminal_names)")
        from ._tree import Tree

        terminals = sorted(n for i, n in enumerate(tree.names) if not tree.children[i])
        perm = self._rng.shuffle_indices(len(terminals))
        new_of = {terminals[i]: terminals[int(perm[i])] for i in range(len(terminals))}
        names = [new_of.get(n, n) for n in tree.names]
        return Tree(names, tree.parent, tree.length, tree.children, tree.name, tree.bootstrap)

    # -- results ----------------------------------------------------------------

    def _frame(self, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=self.result.table.index, columns=self.result.table.columns)

    def _long(self, lname: str, values: dict[str, np.ndarray]) -> pd.DataFrame:
        idx, _ = self._lists[lname]
        out = pd.DataFrame({"group": idx.get_level_values(0), "key": idx.get_level_values(1)})
        for k, v in values.items():
            out[k] = v
        return out

    def comparison(self, list_name: str | None = None) -> pd.DataFrame:
        """The counters: ``C`` (observed greater), ``Q`` (comparisons), ``T`` (ties), ``P``
        (``C / Q``), ``SUMX`` and ``SUMXX`` (sums of the randomised values and their squares).

        For the scalar indices (``list_name`` None) the columns are
        ``C_<index>``..., one row per group; for a list index, a long table.
        """
        if list_name is None or list_name == "SPATIAL_RESULTS":
            c = self._scalars
            parts = {k: c.c[k] for k in _COUNTERS}
            parts["P"] = c.p()
            cols = {}
            for k in ("C", "P", "Q", "SUMX", "SUMXX", "T"):
                for j, index in enumerate(self.result.table.columns):
                    cols[f"{k}_{index}"] = parts[k][:, j]
            return pd.DataFrame(cols, index=self.result.table.index)
        _, c = self._lists[list_name]
        vals = {k: c.c[k] for k in _COUNTERS}
        vals["P"] = c.p()
        return self._long(list_name, vals)

    def p_rank(self, list_name: str | None = None) -> pd.DataFrame:
        """Rank of the observed values among the randomised ones: ``P`` when above 0.5,
        else the proportion of randomisations not below (Biodiverse's ``p_rank`` lists)."""
        if list_name is None or list_name == "SPATIAL_RESULTS":
            return self._frame(self._scalars.p_rank())
        _, c = self._lists[list_name]
        return self._long(list_name, {"value": c.p_rank()})

    def z_scores(self, list_name: str | None = None) -> pd.DataFrame:
        """Standardised difference of the observed values from the randomised ones."""
        if list_name is None or list_name == "SPATIAL_RESULTS":
            return self._frame(self._scalars.z_scores())
        _, c = self._lists[list_name]
        return self._long(list_name, {"value": c.z_scores()})

    def canape(self, central: bool = False, diff: bool = False) -> pd.DataFrame:
        """CANAPE categories (Mishler et al. 2014) from the p-ranks of PE and RPE.

        Needs ``pe`` and ``phylo_rpe2`` in the analysis (``pe_central`` and
        ``phylo_rpe_central`` for ``central``); ``diff`` uses the RPE
        difference instead of the null PE. Codes: 0 not significant, 1 neo,
        2 palaeo, 3 mixed, 4 super.
        """
        calcs = set(self.result._run.valid)
        need = ("calc_pe_central", "calc_phylo_rpe_central") if central else ("calc_pe", "calc_phylo_rpe2")
        if not all(c in calcs for c in need):
            raise ValueError(f"CANAPE needs {' and '.join(n.removeprefix('calc_') for n in need)}")
        pe_obs = "PEC_WE" if central else "PE_WE"
        pe_alt = ("PHYLO_RPE_DIFFC" if central else "PHYLO_RPE_DIFF2") if diff else (
            "PHYLO_RPE_NULLC" if central else "PHYLO_RPE_NULL2")
        rpe = "PHYLO_RPEC" if central else "PHYLO_RPE2"
        pr = self.p_rank()
        base = self.result.table
        obs_s = pr[pe_obs].fillna(0.5).to_numpy()
        alt_s = pr[pe_alt].fillna(0.5).to_numpy()
        rpe_s = pr[rpe].fillna(0.5).to_numpy()
        code = np.where((obs_s <= 0.95) & (alt_s <= 0.95), 0,
               np.where(rpe_s < 0.025, 1,
               np.where(rpe_s > 0.975, 2,
               np.where((obs_s > 0.99) & (alt_s > 0.99), 4, 3)))).astype(float)
        code[base[pe_obs].isna().to_numpy()] = np.nan
        out = pd.DataFrame({"CANAPE_CODE": code}, index=base.index)
        for k, name in ((1, "NEO"), (2, "PALAEO"), (3, "MIXED"), (4, "SUPER")):
            out[name] = np.where(np.isnan(code), np.nan, (code == k).astype(float))
        return out


def randomise(
    result: SpatialResult,
    function: str = "rand_csr_by_group",
    iterations: int = 99,
    *,
    seed: int | None = None,
    name: str | None = None,
    n_threads: int | None = None,
    **args,
) -> Randomisation:
    """Randomise the basedata of a spatial analysis and compare the results.

    Parameters
    ----------
    result : SpatialResult
        The observed analysis (its neighbourhoods are kept).
    function : str
        Randomisation: ``rand_structured`` (Biodiverse's default: labels go
        to random groups keeping each group's richness, optionally scaled
        by ``richness_multiplier`` and ``richness_addition``, and label
        ranges; with ``spatial_conditions_for_label_allocation`` the
        occurrences go around seed groups), its spatial forms
        ``rand_spatially_structured``, ``rand_diffusion`` and
        ``rand_random_walk`` (options ``spatial_conditions_for_label_allocation``,
        ``spatial_allocation_order``, ``label_allocation_backtracking``,
        ``spatial_allocation_reseed_prob``,
        ``spatial_conditions_for_seed_location``), the swap algorithms
        ``rand_independent_swaps``,
        ``rand_independent_swaps_modified`` and ``rand_curveball`` (both
        richness and ranges kept; options ``swap_count``,
        ``max_swap_attempts``, ``stop_on_all_swapped`` and, for curveball,
        ``use_max_swap``), ``rand_csr_by_group`` (whole assemblages to
        random groups) or ``rand_nochange``. All take ``spatial_conditions_for_subset``
        and ``definition_query`` (randomise within subsets of groups),
        ``labels_not_to_randomise``, ``randomise_group_props_by``
        (``no_change``, ``by_set``, ``by_item``) and ``randomise_trees_by``
        (``shuffle_terminal_names``).
    iterations : int
        Iterations to run now (see :meth:`Randomisation.run` for more).
    seed : int, optional
        Seed of Biodiverse's Mersenne Twister (the same seed gives
        Biodiverse's randomisations). Random when not given.
    name : str, optional
        Name of the randomisation (default: the function).
    n_threads : int, optional
        Threads for the analyses (default: all cores but one).
    **args
        Arguments of the randomisation function.

    Returns
    -------
    Randomisation

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> df = pd.DataFrame({"sp": ["a", "b", "a", "c", "c"], "x": [0.5, 0.5, 1.5, 2.5, 3.5], "y": [0.5] * 5})
    >>> bd = ls.bio.BaseData.from_dataframe(df, ["x", "y"], "sp", cell_sizes=[1, 1])
    >>> res = ls.bio.spatial_analysis(bd, "sp_self_only()", ["richness"])
    >>> rnd = ls.bio.randomise(res, "rand_csr_by_group", iterations=9, seed=1)
    >>> rnd.p_rank()["RICHNESS_ALL"].round(2).tolist()
    [0.56, 0.67, 0.89, 0.89]
    """
    if seed is None:
        import secrets

        seed = secrets.randbits(32)
    rnd = Randomisation(result, function, name or function, seed, args)
    if iterations:
        rnd.run(iterations, n_threads)
    return rnd
