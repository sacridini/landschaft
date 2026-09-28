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


FUNCTIONS: dict[str, Callable] = {
    "rand_nochange": rand_nochange,
    "rand_csr_by_group": rand_csr_by_group,
}


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
            rand_bd = fn(res.bd, self._rng, **self.args)
            table, lists = _evaluate(rand_bd, res._run, threads)
            self._scalars.add(table.reindex(columns=res.table.columns).to_numpy(dtype=np.float64))
            for lname, (idx, counts) in self._lists.items():
                df = lists[lname]
                r = pd.Series(df["value"].to_numpy(dtype=np.float64),
                              index=pd.MultiIndex.from_arrays([df["group"].to_numpy(), df["key"].to_numpy()]))
                counts.add(r.reindex(idx).to_numpy(dtype=np.float64))
            self.iterations += 1
        return self

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
        Randomisation: ``rand_csr_by_group`` (whole assemblages to random
        groups) or ``rand_nochange``.
    iterations : int
        Iterations to run now (see :meth:`Randomisation.run` for more).
    seed : int, optional
        Seed of Biodiverse's Mersenne Twister (the same seed gives
        Biodiverse's randomisations). Random when not given.
    name : str, optional
        Name of the randomisation (default: the function).
    n_threads : int, optional
        Threads for the analyses (default: all cores but one).

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
