# SPDX-License-Identifier: GPL-3.0-or-later
"""Global sensitivity analysis: Morris's elementary effects and Sobol's indices, as SALib computes them.

:func:`design` gives the runs (Morris's trajectories, or Saltelli's runs of a
scrambled Sobol' sequence), :func:`evaluate` runs a model at each of them (on
this computer, or on the helpers of :mod:`landschaft.parallel`), and
:func:`analyze` turns the outputs into the indices. The designs and the indices
are SALib 1.6's (MIT), the same numbers; the same as Terraflux's
``sensitivity.*`` operators.

Typical use::

    import landschaft as ls

    d = ls.sensitivity.design("sobol", {"rate": [0.01, 0.1], "threshold": [0.2, 0.8]}, n=256)
    y = ls.sensitivity.evaluate(lambda p: model(rate=p[0], threshold=p[1]), d)
    s = ls.sensitivity.analyze(d, y)
    s.indices                                  # S1, ST and their intervals by factor

    # a landschaft tool, its runs spread over the helpers of this computer's group
    f = ls.parallel.call("surface.gravity", sources, exponent=ls.parallel.param(0))
    y = ls.sensitivity.evaluate(f, d, on=ls.helpers(), summary=np.mean)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._save import Saveable

__all__ = ["Design", "Sensitivity", "analyze", "design", "evaluate"]

METHODS = ("morris", "sobol")


def _factors(factors) -> tuple[list[str], np.ndarray]:
    """Names and bounds (k x 2) from ``{name: (low, high)}``, ``[(name, low, high), ...]`` or a DataFrame."""
    bad = "factors: {name: (low, high)}, [(name, low, high), ...] or a DataFrame with name, low, high"
    if isinstance(factors, pd.DataFrame):
        cols = {c.lower(): c for c in factors.columns}
        if not {"name", "low", "high"} <= set(cols):
            raise ValueError(bad)
        items = list(zip(factors[cols["name"]], factors[cols["low"]], factors[cols["high"]]))
    elif isinstance(factors, dict):
        items = []
        for k, v in factors.items():
            if len(v) != 2:
                raise ValueError(bad)
            items.append((k, v[0], v[1]))
    else:
        try:
            items = [(n, lo, hi) for n, lo, hi in factors]
        except (TypeError, ValueError):
            raise ValueError(bad) from None
    names = [str(n) for n, _, _ in items]
    bounds = np.array([[float(lo), float(hi)] for _, lo, hi in items], dtype=np.float64).reshape(-1, 2)
    if not names or np.isnan(bounds).any() or (bounds[:, 1] <= bounds[:, 0]).any():
        raise ValueError("factors: one at least, each with low below high")
    if len(set(names)) != len(names):
        raise ValueError("factors: a name more than once")
    return names, bounds


def _scale(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """SALib's scale_samples: unit values onto the factors' ranges."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    np.add(np.multiply(x, hi - lo, out=x), lo, out=x)
    return x


def _delta(levels: int) -> float:
    return levels / (2.0 * (levels - 1))


def _morris_sample(bounds: np.ndarray, trajectories: int, levels: int, seed) -> np.ndarray:
    """SALib's morris.sample without groups nor optimised trajectories."""
    if levels % 2:
        warnings.warn("the number of levels should be even", stacklevel=3)
    rng = np.random.default_rng(seed)
    k = len(bounds)
    delta = _delta(levels)
    b = np.tril(np.ones([k + 1, k], dtype=int), -1)
    j = np.ones((k + 1, k))
    g = np.eye(k)
    out = []
    for _ in range(trajectories):
        p_star = np.eye(k, k)
        rng.shuffle(p_star)
        d_star = np.diag(rng.choice([-1, 1], k))
        x_star = np.zeros((1, k))
        x_star[0, :] = rng.choice(np.linspace(0, 1 - delta, int(levels / 2)), k)
        element_a = j[0, :] * x_star
        element_b = np.matmul(g, p_star).T
        element_c = np.matmul(2.0 * b, element_b)
        element_d = np.matmul(element_c - j, d_star)
        out.append(element_a + (delta / 2.0) * (element_d + j))
    x = np.array(out).reshape((trajectories * (k + 1), k))
    return _scale(x, bounds)


def _saltelli_sample(bounds: np.ndarray, n: int, second_order: bool, seed) -> np.ndarray:
    """SALib's sobol.sample: Saltelli's scheme on scipy's scrambled Sobol' sequence of twice the factors."""
    from scipy.stats import qmc

    d = len(bounds)
    if n & (n - 1) or n == 0:
        warnings.warn(f"the base points ({n}) should be a power of 2 for the balance of the Sobol' sequence",
                      stacklevel=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # scipy: the balance of n not a power of 2 (warned above)
        base = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random(n)
    rows = (2 * d + 2) if second_order else (d + 2)
    x = np.zeros([rows * n, d])
    index = 0
    for i in range(n):
        x[index] = base[i, :d]
        index += 1
        for k in range(d):
            x[index] = base[i, :d]
            x[index, k] = base[i, k + d]
            index += 1
        if second_order:
            for k in range(d):
                x[index] = base[i, d:]
                x[index, k] = base[i, k]
                index += 1
        x[index] = base[i, d:]
        index += 1
    return _scale(x, bounds)


@dataclass
class Design(Saveable):
    """The runs of a sensitivity analysis (:func:`design`).

    Attributes
    ----------
    table : pandas.DataFrame
        A row per run (``run`` 1, 2, ...), a column per factor.
    method : str
        ``"morris"`` or ``"sobol"``.
    factors : list of str
        The factors, in the columns' order.
    bounds : numpy.ndarray
        Their ranges (factors x 2).
    levels : int
        Morris's levels.
    second_order : bool
        Sobol's: the runs of the pairs' indices too.
    """

    table: pd.DataFrame
    method: str
    factors: list
    bounds: np.ndarray
    levels: int = 4
    second_order: bool = True

    @property
    def values(self) -> np.ndarray:
        """The runs as an array (runs x factors)."""
        return self.table[self.factors].to_numpy(np.float64)

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return f"Design({self.method}, {len(self.table)} runs of {len(self.factors)} factors)"


def design(method: str, factors, *, n: int | None = None, levels: int = 4, second_order: bool = True,
           seed: int | None = 1) -> Design:
    """The runs of a global sensitivity analysis (SALib's samplers, the same values).

    ``"morris"``: Morris's trajectories, a factor changed at a time on a grid
    of ``levels`` levels: ``n`` trajectories of ``factors + 1`` runs.
    ``"sobol"``: Saltelli's runs of Sobol's method on a scrambled Sobol'
    sequence (scipy's): ``n`` base points (a power of 2) of
    ``2 × factors + 2`` runs, or ``factors + 2`` without the second order. Run
    the model at every row (:func:`evaluate`, or a loop) and give the outputs
    to :func:`analyze`.

    Parameters
    ----------
    method
        ``"morris"`` or ``"sobol"``.
    factors
        The factors and their ranges: ``{name: (low, high)}``,
        ``[(name, low, high), ...]`` or a DataFrame with ``name``, ``low``,
        ``high``.
    n
        Trajectories (morris; default 10) or base points (sobol; default 64).
    levels
        Morris: the grid's levels (default 4, even).
    second_order
        Sobol: the runs of the pairs' indices too (default).
    seed
        The seed of numpy's ``default_rng`` (Morris) or of the scrambling
        (Sobol).

    Returns
    -------
    Design

    Examples
    --------
    >>> import landschaft as ls
    >>> d = ls.sensitivity.design("morris", {"a": (0, 1), "b": (10, 20)}, n=2, seed=3)
    >>> d.table
       run         a          b
    0    1  0.666667  16.666667
    1    2  0.666667  10.000000
    2    3  0.000000  10.000000
    3    4  0.000000  10.000000
    4    5  0.666667  10.000000
    5    6  0.666667  16.666667
    >>> len(ls.sensitivity.design("sobol", {"a": (0, 1), "b": (10, 20)}, n=8))     # 8 x (2 x 2 + 2)
    48
    """
    names, bounds = _factors(factors)
    if method == "morris":
        x = _morris_sample(bounds, 10 if n is None else int(n), int(levels), seed)
    elif method == "sobol":
        x = _saltelli_sample(bounds, 64 if n is None else int(n), bool(second_order), seed)
    else:
        raise ValueError(f"method: morris or sobol, not {method!r}")
    table = pd.DataFrame(x, columns=names)
    table.insert(0, "run", np.arange(1, len(x) + 1))
    return Design(table, method, names, bounds, int(levels), bool(second_order))


def evaluate(f, design: Design, *, on=None, chunk: int | None = None, summary=None) -> np.ndarray:
    """The outputs of a model at every run of a design.

    Parameters
    ----------
    f
        The model: ``f(x)`` with a run's factors as a 1-D array (in the
        design's order), returning a number (or a raster or array, see
        ``summary``). With ``on``, a landschaft tool with its arguments,
        :func:`landschaft.parallel.call`, the factors marked by
        :func:`landschaft.parallel.param` (its i-th factor).
    design
        The runs (:func:`design`).
    on
        ``ls.helpers()``: run here and on the helpers of this computer's group
        (:mod:`landschaft.parallel`), with the same results.
    chunk
        Runs per task with ``on`` (default: about a hundredth of them).
    summary
        A function of a run's output giving the number analysed (e.g.
        ``numpy.mean`` of a map); default: the output itself, which must be a
        number.

    Returns
    -------
    numpy.ndarray
        An output per run.

    Examples
    --------
    >>> import landschaft as ls
    >>> d = ls.sensitivity.design("sobol", {"a": (0, 1), "b": (0, 1)}, n=4, seed=1)
    >>> ls.sensitivity.evaluate(lambda p: p[0] + 2 * p[1], d).shape
    (24,)
    """
    x = design.values if isinstance(design, Design) else np.asarray(design, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("the design has no runs")

    def value(out):
        v = np.asarray(getattr(out, "values", out), dtype=np.float64)
        if summary is not None:
            v = np.asarray(summary(v), dtype=np.float64)
        return v.reshape(-1)

    first = value(f(x[0].copy()))
    if first.size != 1 and summary is None:
        raise ValueError("the model gives more than a number: give a summary (e.g. summary=numpy.mean)")
    if on is not None:
        from .uncertainty import _draws_on

        rest = _draws_on(on, f, x[1:], chunk)
        if summary is not None:
            rest = np.array([value(r) for r in rest]).reshape(len(rest), -1)
        out = np.concatenate([first.reshape(1, -1), rest.reshape(len(rest), -1)])
    else:
        out = np.array([first] + [value(f(row.copy())) for row in x[1:]])
    if out.shape[1] != 1:
        raise ValueError("the summary must give a number")
    return out[:, 0]


@dataclass
class Sensitivity(Saveable):
    """The indices of a sensitivity analysis (:func:`analyze`).

    Attributes
    ----------
    indices : pandas.DataFrame
        A row per factor: ``mu``, ``mu_star`` (the mean absolute elementary
        effect: the importance), ``sigma`` (interactions and non-linearity) and
        ``mu_star_conf`` (Morris); or ``s1`` (the share of the output's variance
        due to the factor alone), ``s1_conf``, ``st`` (with its interactions)
        and ``st_conf`` (Sobol).
    s2, s2_conf : pandas.DataFrame or None
        Sobol: the second order indices of the pairs (upper triangle; NaN
        elsewhere) and their intervals.
    method : str
        ``"morris"`` or ``"sobol"``.
    """

    indices: pd.DataFrame
    method: str
    s2: pd.DataFrame | None = None
    s2_conf: pd.DataFrame | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"Sensitivity({self.method}, {len(self.indices)} factors)"


def _ppf(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def _morris_analyze(x, y, levels, resamples, confidence, seed) -> dict:
    """SALib's morris.analyze without groups nor scaling."""
    rng = np.random.default_rng(seed)
    k = x.shape[1]
    if y.size % (k + 1):
        raise ValueError(f"{len(x)} runs and {y.size} outputs: trajectories of {k + 1} runs")
    t = y.size // (k + 1)
    size = y.size // t
    out = y.reshape(t, size)
    inp = x.reshape(t, size, k)
    dv = np.subtract(inp[:, 1:, :], inp[:, 0:-1, :])
    up, lo = dv > 0, dv < 0

    def reorganize(increase: bool):
        pad_up, pad_lo = ((1, 0), (0, 1)) if increase else ((0, 1), (1, 0))
        a = np.pad(up, ((0, 0), pad_up, (0, 0)), "constant")
        b = np.pad(lo, ((0, 0), pad_lo, (0, 0)), "constant")
        return np.einsum("ik,ikj->ij", out, a + b).T

    ee = np.subtract(reorganize(True), reorganize(False)) / _delta(levels)
    mu = np.average(ee, 1)
    mu_star = np.average(np.abs(ee), 1)
    sigma = np.std(ee, axis=1, ddof=1)
    z = _ppf(0.5 + confidence / 2)
    conf = []
    for j in range(k):
        e = ee[j, :]
        idx = rng.integers(len(e), size=(resamples, len(e)))
        conf.append(z * np.average(np.abs(e[idx]), axis=1).std(ddof=1))
    return {"mu": mu, "mu_star": mu_star, "sigma": sigma, "mu_star_conf": np.asarray(conf)}


def _first(a, ab, b):
    y = np.r_[a, b]
    if np.ptp(y) <= np.finfo(float).eps:
        return np.zeros(y.shape[1:], dtype=np.float64)
    v = np.var(y, axis=0)
    return np.divide(np.mean(b * (ab - a), axis=0), v, out=np.zeros_like(v, dtype=np.float64),
                     where=v > np.finfo(float).eps)


def _total(a, ab, b):
    y = np.r_[a, b]
    if np.ptp(y) <= np.finfo(float).eps:
        return np.zeros(y.shape[1:], dtype=np.float64)
    v = np.var(y, axis=0)
    return np.divide(0.5 * np.mean((a - ab) ** 2, axis=0), v, out=np.zeros_like(v, dtype=np.float64),
                     where=v > np.finfo(float).eps)


def _second(a, abj, abk, baj, b):
    y = np.r_[a, b]
    if np.ptp(y) <= np.finfo(float).eps:
        return np.zeros(y.shape[1:], dtype=np.float64)
    v = np.var(y, axis=0)
    vjk = np.divide(np.mean(baj * abk - a * b, axis=0), v, out=np.zeros_like(v, dtype=np.float64),
                    where=v > np.finfo(float).eps)
    return vjk - _first(a, abj, b) - _first(a, abk, b)


def _sobol_analyze(y, d, second_order, resamples, confidence, seed) -> dict:
    """SALib's sobol.analyze (serial)."""
    step = 2 * d + 2 if second_order else d + 2
    if y.size % step:
        raise ValueError(f"{y.size} outputs are not Saltelli's runs of {d} factors"
                         + (" with the second order" if second_order else " without the second order"))
    n = y.size // step
    if np.ptp(y) <= np.finfo(float).eps:
        warnings.warn("the outputs are constant: every index is 0", stacklevel=3)
    y = (y - y.mean()) / y.std()
    a = y[0:y.size:step]
    b = y[step - 1:y.size:step]
    ab = np.zeros((n, d))
    ba = np.zeros((n, d)) if second_order else None
    for j in range(d):
        ab[:, j] = y[j + 1:y.size:step]
        if second_order:
            ba[:, j] = y[j + 1 + d:y.size:step]
    r = np.random.default_rng(seed).integers(n, size=(n, resamples))
    z = _ppf(0.5 + confidence / 2)
    s = {"s1": np.zeros(d), "s1_conf": np.zeros(d), "st": np.zeros(d), "st_conf": np.zeros(d)}
    for j in range(d):
        s["s1"][j] = _first(a, ab[:, j], b)
        c = _first(a[r], ab[r, j], b[r])
        var_diff = np.ptp(np.r_[a[r], b[r]])
        s["s1_conf"][j] = z * c.std(ddof=1) if var_diff != 0.0 else 0.0
        s["st"][j] = _total(a, ab[:, j], b)
        c = _total(a[r], ab[r, j], b[r])
        s["st_conf"][j] = z * c.std(ddof=1) if var_diff != 0.0 else 0.0
    if second_order:
        s["s2"] = np.full((d, d), np.nan)
        s["s2_conf"] = np.full((d, d), np.nan)
        for j in range(d):
            for k in range(j + 1, d):
                s["s2"][j, k] = _second(a, ab[:, j], ab[:, k], ba[:, j], b)
                s["s2_conf"][j, k] = z * _second(a[r], ab[r, j], ab[r, k], ba[r, j], b[r]).std(ddof=1)
    return s


def analyze(design: Design, results, *, column=None, method: str | None = None, levels: int | None = None,
            second_order: bool | None = None, resamples: int = 100, confidence: float = 0.95,
            seed: int | None = 1) -> Sensitivity:
    """The indices of a global sensitivity analysis from its runs and their outputs (SALib's, the same numbers).

    Morris: μ, μ* (the mean absolute elementary effect: the importance) and σ
    (interactions and non-linearity), with μ*'s bootstrap interval. Sobol: the
    first order (S1: the share of the output's variance due to the factor
    alone), total (ST: with its interactions) and second order (S2: of pairs)
    indices, with their bootstrap intervals.

    Parameters
    ----------
    design
        The runs (:func:`design`), or a table of them (a column per factor;
        a ``run`` column is left out), with ``method`` and its settings.
    results
        The outputs of the runs in the same order: an array, a Series, or a
        DataFrame (its ``column``, default the last).
    column
        The column of a DataFrame of results.
    method, levels, second_order
        As the design's (taken from a :class:`Design`).
    resamples
        The bootstraps of the intervals (default 100).
    confidence
        The level of the intervals (default 0.95).
    seed
        The seed of the bootstraps (numpy's ``default_rng``).

    Returns
    -------
    Sensitivity

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> d = ls.sensitivity.design("sobol", {"x1": (-np.pi, np.pi), "x2": (-np.pi, np.pi), "x3": (-np.pi, np.pi)},
    ...                           n=1024, seed=11)
    >>> ishigami = lambda p: np.sin(p[0]) + 7 * np.sin(p[1]) ** 2 + 0.1 * p[2] ** 4 * np.sin(p[0])
    >>> s = ls.sensitivity.analyze(d, ls.sensitivity.evaluate(ishigami, d))
    >>> s.indices[["s1", "st"]].round(2)          # the exact values: 0.31 0.44 0 / 0.56 0.44 0.24
          s1    st
    x1  0.32  0.56
    x2  0.44  0.45
    x3  0.01  0.24
    """
    if isinstance(design, Design):
        x, names = design.values, list(design.factors)
        method = design.method if method is None else method
        levels = design.levels if levels is None else levels
        second_order = design.second_order if second_order is None else second_order
    else:
        t = pd.DataFrame(design)
        names = [str(c) for c in t.columns if str(c) != "run"]
        x = t[[c for c in t.columns if str(c) != "run"]].to_numpy(np.float64)
        levels = 4 if levels is None else levels
        second_order = True if second_order is None else second_order
    if isinstance(results, pd.DataFrame):
        y = results[results.columns[-1] if column is None else column]
    else:
        y = results
    y = np.asarray(getattr(y, "values", y), dtype=np.float64).reshape(-1)
    if np.isnan(y).any():
        raise ValueError("a run has no output")
    if y.size != len(x):
        raise ValueError(f"{len(x)} runs and {y.size} outputs")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if method == "morris":
        m = _morris_analyze(x, y, int(levels), int(resamples), confidence, seed)
        return Sensitivity(pd.DataFrame(m, index=pd.Index(names)), "morris")
    if method == "sobol":
        s = _sobol_analyze(y, len(names), bool(second_order), int(resamples), confidence, seed)
        idx = pd.Index(names)
        out = Sensitivity(pd.DataFrame({k: s[k] for k in ("s1", "s1_conf", "st", "st_conf")}, index=idx), "sobol")
        if second_order:
            out.s2 = pd.DataFrame(s["s2"], index=idx, columns=names)
            out.s2_conf = pd.DataFrame(s["s2_conf"], index=idx, columns=names)
        return out
    raise ValueError(f"method: morris or sobol, not {method!r}")
