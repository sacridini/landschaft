# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluating species distribution models: AUC, TSS, kappa, Boyce index, thresholds, omission."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pandas as pd

from .. import _core
from .._save import Saveable

__all__ = ["Evaluation", "boyce", "evaluate", "omission_rate", "threshold"]

_LOG10_2 = 0.301029995663981195213738894724


# ------------------------------------------------------------------------------------------ R helpers

def _r_round(x: float, digits: int) -> float:
    """R's round(x, digits) (R >= 4.0, fround.c): the nearer of the two candidates, ties to even."""
    if math.isnan(x) or not math.isfinite(x) or x == 0:
        return x
    if digits == 0:
        return float(np.rint(x))
    dig = math.floor(digits + 0.5)
    sgn = -1.0 if x < 0 else 1.0
    x = abs(x)
    if _LOG10_2 * (0.5 + (math.frexp(x)[1] - 1)) + dig > 15:
        return sgn * x
    p10 = float(10 ** dig)
    x10 = p10 * x
    i10 = math.floor(x10)
    xd = i10 / p10
    xu = math.ceil(x10) / p10
    du = xu - x
    dd = x - xd
    return sgn * (xu if (du < dd or (math.fmod(i10, 2) == 1 and du == dd)) else xd)


def _r_seq(start: float, end: float, by: float) -> np.ndarray:
    from ._folds import _r_seq as seq

    return seq(start, end, by)


def _r_quantile7(x: np.ndarray, probs) -> np.ndarray:
    x = np.sort(x)
    n = x.size
    out = np.empty(len(probs))
    for k, p in enumerate(probs):
        index = 1 + max(n - 1, 0) * p
        lo, hi = math.floor(index), math.ceil(index)
        q = x[lo - 1]
        if index > lo and x[hi - 1] != q:
            h = index - lo
            q = (1 - h) * q + h * x[hi - 1]
        out[k] = q
    return out


def _r_cor(x, y, method: str) -> float:
    """R's cor(x, y) of two vectors (Pearson, Spearman on average ranks, Kendall)."""
    from scipy.stats import rankdata

    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    if method == "spearman":
        x, y = rankdata(x).astype(float), rankdata(y).astype(float)
    elif method not in ("pearson", "kendall"):
        raise ValueError("method must be 'pearson', 'spearman' or 'kendall'")
    return _core.sdm_r_cor2(x, y, method == "kendall")


def _logical_mean(k: int, n: int) -> float:
    """R's mean() of a logical vector: k / n in 80 bits, then rounded to double."""
    from ._folds import _round_bits

    return float(_round_bits(Fraction(k, n))) if n else float("nan")


def _values(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    return v[~np.isnan(v)]


# ------------------------------------------------------------------------------------------- Boyce

def boyce(presences, available, *, nclass: int = 0, window: float | None = None, res: int = 100,
          method: str = "spearman", rm_duplicate: bool = True) -> dict:
    """Continuous Boyce index: does predicted suitability rise with the frequency of presences?

    For moving windows of suitability, the ratio of the share of presences
    to the share of available cells (background) predicted in the window;
    the index is the rank correlation of that ratio with suitability: 1 for
    a model whose higher predictions always hold relatively more presences,
    0 for a random one. The standard metric for presence-only data (Boyce et
    al. 2002; Hirzel et al. 2006). ``ecospat.boyce``, the same windows, ratios
    and correlation, except for its "+1 trick": ecospat shifts its last window
    by one unit of suitability, which removes that window for predictions on
    0-1 and keeps it on 0-1000, so the index depends on the scale of the
    predictions; here the last window is ``[max - width, max]`` on any scale.

    Parameters
    ----------
    presences
        Predictions at the presences (test records).
    available
        Predictions at the background points (or every cell of the area).
    nclass
        0: moving windows (default); k > 0: k classes of equal width.
    window
        Width of the moving window (default: a tenth of the range of
        ``available``).
    res
        Number of window steps (100).
    method : {"spearman", "pearson", "kendall"}
        Correlation.
    rm_duplicate
        Ignore successive equal ratios in the correlation.

    Returns
    -------
    dict
        ``cor`` (the index, rounded to 3 decimals as ecospat), ``f_ratio``
        (predicted / expected ratios), ``hs`` (window centres) and ``used``
        (the windows entering the correlation).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> rng = np.random.default_rng(1)
    >>> bg = rng.uniform(0, 1, 2000)
    >>> pres = rng.beta(4, 1.5, 200)            # presences where suitability is high
    >>> ls.sdm.boyce(pres, bg)["cor"]
    0.954
    """
    obs = _values(presences)
    fit = _values(available)
    if obs.size == 0 or fit.size == 0:
        raise ValueError("no prediction values")
    mini = float(min(fit.min(), obs.min()))
    maxi = float(max(fit.max(), obs.max()))
    if np.ndim(nclass) != 0:
        breaks = np.sort(np.asarray(nclass, dtype=float))
        vec = np.concatenate([[mini], breaks[(breaks >= mini) & (breaks <= maxi)]])
        upper = np.append(vec[1:], maxi)
    elif nclass == 0:
        w = (float(fit.max()) - float(fit.min())) / 10 if window is None else float(window)
        vec = _r_seq(mini, maxi - w, (maxi - mini - w) / res)
        upper = vec + w
    else:
        vec = _r_seq(mini, maxi, (maxi - mini) / nclass)
        upper = np.append(vec[1:], maxi)
    so, sf = np.sort(obs), np.sort(fit)

    def count(s, lo, hi):
        return np.searchsorted(s, hi, side="right") - np.searchsorted(s, lo, side="left")

    with np.errstate(divide="ignore", invalid="ignore"):
        pi = count(so, vec, upper) / obs.size
        ei = count(sf, vec, upper) / fit.size
        f = np.array([_r_round(v, 10) for v in pi / ei])
    keep = ~np.isnan(f)
    f, starts = f[keep], vec[keep]
    from ._folds import _round_bits

    # apply(interval, 1, sum) / 2: R's 80-bit sum of the two bounds, rounded to double, halved
    hs = np.array([float(_round_bits(Fraction(lo) + Fraction(hi))) / 2 for lo, hi in zip(vec[keep], upper[keep])])
    used = np.arange(f.size)
    if f.size < 2:
        b = float("nan")
    else:
        if rm_duplicate:  # the last of each run of equal successive ratios (a final ratio of 1 included)
            used = np.flatnonzero(np.append(f[1:] != f[:-1], True))
        b = _r_cor(f[used], starts[used], method)
    return {"cor": _r_round(b, 3), "f_ratio": f, "hs": hs, "used": used}


# --------------------------------------------------------------------------------------- evaluation

@dataclass
class Evaluation(Saveable):
    """How well predictions separate presences from background (or absences).

    Attributes
    ----------
    stats : dict
        ``auc``, ``cor`` (point-biserial), ``pcor`` (its p-value), ``np`` and
        ``na`` (numbers of presences and background), ``prevalence``,
        ``max_tss`` and ``boyce`` (continuous Boyce index).
    table : pandas.DataFrame
        One row per candidate threshold: the confusion matrix (``tp``,
        ``fp``, ``fn``, ``tn``), ``kappa``, ``ccr``, ``tpr`` (sensitivity),
        ``tnr`` (specificity), ``fpr``, ``fnr``, ``ppp``, ``npp``, ``mcr``,
        ``or`` and ``tss``.
    thresholds : dict
        ``max_kappa``, ``max_spec_sens`` (= max TSS), ``no_omission``,
        ``equal_prevalence``, ``equal_sens_spec``, ``p10`` (10th percentile
        of the presences) and ``mtp`` (minimum training presence).
    """

    stats: dict
    table: pd.DataFrame
    thresholds: dict
    boyce: dict = field(default_factory=dict, repr=False)

    def _save_items(self) -> dict:
        return {"stats": self.stats, "thresholds": self.thresholds, "table": self.table}


def _thresholds(p: np.ndarray, a: np.ndarray, tr=None) -> np.ndarray:
    if tr is not None:
        return np.sort(np.asarray(tr, dtype=float).ravel())
    probs = np.arange(1001) / 1000.0
    t1 = _r_quantile7(p, probs) if p.size > 1000 else p
    t2 = _r_quantile7(a, probs) if a.size > 1000 else a
    t = np.unique(np.array([_r_round(v, 8) for v in np.concatenate([t1, t2])]))
    return np.concatenate([t - 0.0001, [t[-1], t[-1] + 0.0001]])


def evaluate(presences, background, *, thresholds=None) -> Evaluation:
    """Evaluate predictions at presences against background points (or absences).

    The statistics of ``predicts::pa_evaluate`` / ``dismo::evaluate``, the
    same values: AUC (Mann-Whitney, ties counted one half), the
    point-biserial correlation and, for every candidate threshold (the
    prediction values minus 0.0001), the confusion matrix with kappa,
    sensitivity, specificity, TSS (sensitivity + specificity - 1, Allouche
    et al. 2006) and the rest; plus the continuous Boyce index (see
    :func:`boyce`) and the usual thresholds.

    Parameters
    ----------
    presences, background
        Predictions at the presences and at the background points (or
        absences); missing values are dropped.
    thresholds
        Candidate thresholds (default: from the predictions, as predicts).

    Returns
    -------
    Evaluation

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> p = np.array([0.9, 0.8, 0.8, 0.7, 0.6, 0.55, 0.3])
    >>> a = np.array([0.8, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.05])
    >>> e = ls.sdm.evaluate(p, a)
    >>> round(e.stats["auc"], 4), round(e.stats["max_tss"], 4)
    (0.8482, 0.7321)
    >>> e.thresholds["max_spec_sens"]
    0.5499
    """
    p, a = _values(presences), _values(background)
    if p.size == 0 or a.size == 0:
        raise ValueError("need predictions at presences and at background points")
    from scipy.stats import rankdata

    n_p, n_a = p.size, a.size
    ranks = rankdata(np.concatenate([p, a]))  # average ties, as R's rank()
    r = ranks[:n_p].sum() - (n_p * (n_p + 1.0) / 2)
    auc = r / (float(n_a) * float(n_p))
    cor = pcor = float("nan")
    if n_p + n_a > 2:
        cor = _r_cor(np.concatenate([p, a]), np.concatenate([np.ones(n_p), np.zeros(n_a)]), "pearson")
        if np.isfinite(cor):
            from scipy.stats import t as student

            df = n_p + n_a - 2
            tval = math.sqrt(df) * cor / math.sqrt(1 - cor * cor) if abs(cor) < 1 else math.copysign(np.inf, cor)
            pcor = 2 * min(student.cdf(tval, df), student.sf(tval, df))
    tr = _thresholds(p, a, thresholds)
    ps, as_ = np.sort(p), np.sort(a)
    tp = (n_p - np.searchsorted(ps, tr, side="left")).astype(float)
    fp = (n_a - np.searchsorted(as_, tr, side="left")).astype(float)
    fn = n_p - tp
    tn = n_a - fp
    N = float(n_p + n_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = tp / (tp + fn)
        tnr = tn / (fp + tn)
        pra = (tp + tn) / N
        pry = (((tp + fp) / N) * (tp + fn)) / N
        prn = (((fn + tn) / N) * (fp + tn)) / N
        pre = pry + prn
        kappa = (pra - pre) / (1 - pre)
        table = pd.DataFrame({
            "threshold": tr, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "kappa": kappa, "ccr": (tp + tn) / N,
            "tpr": tpr, "tnr": tnr, "fpr": fp / (fp + tn), "fnr": fn / (tp + fn), "ppp": tp / (tp + fp),
            "npp": tn / (fn + tn), "mcr": (fp + fn) / N, "or": (tp * tn) / (fn * fp), "tss": tpr + tnr - 1,
        })
    prevalence = (tp[0] + fn[0]) / N

    def first(values, best="max"):
        v = np.where(np.isnan(values), -np.inf if best == "max" else np.inf, values)
        return float(tr[int(np.argmax(v) if best == "max" else np.argmin(v))])

    no_omit = np.flatnonzero(fn == 0)
    thresholds_out = {
        "max_kappa": first(kappa),
        "max_spec_sens": first(tpr + tnr),
        "no_omission": float(tr[no_omit.max()]) if no_omit.size else float("nan"),
        "equal_prevalence": first(np.abs(tr - prevalence), "min"),
        "equal_sens_spec": first(np.abs(tpr - tnr), "min"),
        "mtp": float(p.min()),
        "p10": _p10(p),
    }
    b = boyce(p, np.concatenate([a, p]))
    stats = {"auc": float(auc), "cor": float(cor), "pcor": float(pcor), "np": int(n_p), "na": int(n_a),
             "prevalence": float(prevalence), "max_tss": float(np.nanmax(tpr + tnr - 1)), "boyce": b["cor"]}
    return Evaluation(stats, table, thresholds_out, b)


def _p10(train: np.ndarray) -> float:
    """ENMeval's 10th percentile training presence threshold (an order statistic)."""
    n = train.size
    k = math.floor(n * 0.9) if n < 10 else math.ceil(n * 0.9)
    if k < 1:
        return float("nan")
    return float(np.sort(train)[::-1][k - 1])


def threshold(presences, background, method: str = "max_spec_sens", *, sensitivity: float = 0.9) -> float:
    """A threshold that turns continuous predictions into presence / absence.

    Parameters
    ----------
    presences, background
        Predictions at the presences and the background (or absences).
    method
        ``"max_spec_sens"`` (maximum TSS, the recommended default; Liu et al.
        2013), ``"max_kappa"``, ``"no_omission"`` (the highest threshold that
        keeps every presence), ``"equal_prevalence"``, ``"equal_sens_spec"``,
        ``"sensitivity"`` (the highest threshold keeping at least
        ``sensitivity`` of the presences; dismo returns the first one below
        it), ``"p10"`` (10th percentile training presence) or ``"mtp"``
        (minimum training presence).
    sensitivity
        For ``method="sensitivity"``.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> p = np.array([0.9, 0.8, 0.8, 0.7, 0.6, 0.55, 0.3])
    >>> a = np.array([0.8, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.05])
    >>> ls.sdm.threshold(p, a, "sensitivity", sensitivity=0.8)
    0.5499
    """
    e = evaluate(presences, background)
    if method == "sensitivity":
        ok = np.flatnonzero(e.table["tpr"].to_numpy() >= sensitivity)
        return float(e.table["threshold"].to_numpy()[ok.max()]) if ok.size else float("nan")
    if method not in e.thresholds:
        raise ValueError(f"unknown method {method!r}; one of {', '.join([*e.thresholds, 'sensitivity'])}")
    return e.thresholds[method]


def omission_rate(train, test, threshold: str = "p10") -> float:
    """Share of test presences predicted below a threshold set on the training presences.

    ENMeval's ``or.mtp`` (``threshold="mtp"``: below the lowest training
    prediction) and ``or.10p`` (``"p10"``: below the 10th percentile of the
    training predictions). A well-fitted model omits about 0 and 0.1 of the
    test presences; much more means overfitting (Radosavljevic & Anderson
    2014).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> train = np.linspace(0.2, 1.0, 50)
    >>> test = np.array([0.1, 0.25, 0.5, 0.9])
    >>> ls.sdm.omission_rate(train, test, "mtp"), ls.sdm.omission_rate(train, test, "p10")
    (0.25, 0.5)
    """
    train, test = _values(train), _values(test)
    if threshold == "mtp":
        t = float(train.min())
    elif threshold == "p10":
        t = _p10(train)
    else:
        raise ValueError("threshold must be 'mtp' or 'p10'")
    if math.isnan(t):
        return float("nan")
    return _logical_mean(int((test < t).sum()), test.size)
