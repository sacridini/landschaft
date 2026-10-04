# SPDX-License-Identifier: GPL-3.0-or-later
"""The classic land change models, assembled from ls.simulate's pieces.

Each function returns a :class:`~landschaft.simulate.Model`. The first four
take the pieces the model needs (they name the allocation and its usual
settings); ``dinamica``, ``plus`` and ``molusce`` fit everything from two
maps and the drivers, as those programs do::

    import landschaft as ls

    S = ls.simulate
    model = S.presets.dinamica(lc2010, lc2020, drivers, steps=10)   # Markov + WoE + Expander/Patcher
    sim = model.run(lc2010, drivers, seed=1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._change import _class_value
from ._maps import read_map

__all__ = ["clue_s", "clumondo", "dinamica", "futures", "molusce", "observed_patches", "ordered", "plus"]


def clue_s(potential, demand, *, rules=None, dynamic=None, **clue):
    """CLUE-S (Verburg et al. 2002) as lulcc runs it: :class:`CLUE` allocation of an :class:`AreaDemand`.

    ``clue`` are :class:`CLUE`'s settings (``jitter``, ``scale``, ``max_iter``,
    ``max_diff``, ``mean_diff``); ``rules`` the :class:`Rules` (elasticities,
    conversion matrix, neighbourhood...).
    """
    from ._allocation import CLUE
    from ._model import Model

    return Model(potential, demand, CLUE(**clue), rules, dynamic)


def ordered(potential, demand, order, *, rules=None, dynamic=None, stochastic: bool = True):
    """The ordered procedure of Fuchs et al. (2013), lulcc's ``OrderedModel``."""
    from ._allocation import Ordered
    from ._model import Model

    return Model(potential, demand, Ordered(order, stochastic), rules, dynamic)


def clumondo(potential, demand, supply, *, rules=None, dynamic=None, **settings):
    """CLUMondo: land systems meeting a :class:`ServiceDemand` with their :class:`Supply`.

    ``settings`` are :class:`CLUMondo`'s (``order``, ``deviation``,
    ``speed``, ``neighbourhood``...); give the starting map or a
    :class:`LandSystemStack` to ``run``.
    """
    from ._clumondo import CLUMondo
    from ._model import Model

    return Model(potential, demand, CLUMondo(supply, **settings), rules, dynamic)


def futures(potential, demand, patch_sizes, *, rules=None, **settings):
    """FUTURES' patch growing (r.futures.simulation): :class:`PatchGrowing` with a :class:`DevelopmentDemand`."""
    from ._futures import PatchGrowing
    from ._model import Model

    return Model(potential, demand, PatchGrowing(patch_sizes, **settings), rules)


def observed_patches(t0, t1, transitions=None, *, nodata=None) -> pd.DataFrame:
    """The patches of each observed transition: expansions of existing patches and new ones.

    The cells that went from ``i`` to ``j`` are grouped in patches (8
    neighbours); a patch touching a cell of ``j`` in ``t0`` is an expansion
    (Dinamica's Expander), the others are new patches (the Patcher).

    Returns
    -------
    pandas.DataFrame
        Per transition: ``cells`` changed, ``expansion`` (share of the
        cells in expansions), the number, mean and variance (cells) of the
        expansion and new patches.
    """
    from scipy import ndimage

    a, b = read_map(t0, nodata).as_float(), read_map(t1, nodata).as_float()
    eight = np.ones((3, 3), dtype=bool)
    if transitions is None:
        ok = ~(np.isnan(a) | np.isnan(b)) & (a != b)
        transitions = sorted({(_class_value(x), _class_value(y)) for x, y in zip(a[ok], b[ok])})
    rows = []
    for f, t in transitions:
        changed = (a == f) & (b == t)
        lab, n = ndimage.label(changed, eight)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
        near = ndimage.binary_dilation(a == t, eight)
        touching = np.zeros(n + 1, dtype=bool)
        touching[np.unique(lab[near & changed])] = True
        exp, new = sizes[touching[1:]], sizes[~touching[1:]]

        def stats(s):
            return (float(s.mean()), float(s.var())) if s.size else (1.0, 0.0)

        (me, ve), (mn, vn) = stats(exp), stats(new)
        rows.append({"from": f, "to": t, "cells": int(changed.sum()),
                     "expansion": float(exp.sum() / sizes.sum()) if sizes.sum() else 0.0,
                     "expansion_patches": int(exp.size), "expansion_mean": me, "expansion_variance": ve,
                     "new_patches": int(new.size), "new_mean": mn, "new_variance": vn})
    return pd.DataFrame(rows)


def dinamica(t0, t1, drivers, *, steps: int = 1, woe=None, patches="observed", isometry: float = 1.5,
             prune_factor: float = 10.0, rules=None, dynamic=None, nodata=None):
    """Dinamica EGO's classic model: Markov rates, weights of evidence, Expander and Patcher.

    Fitted on two maps: the rates of each transition per step (the
    ``steps``-th root of the period's matrix), weights of evidence of every
    observed transition (the ranges found automatically, or ``woe``: the
    :class:`WeightsOfEvidence` settings, a number = the increment), and the
    share of the Expander, the mean and variance of the patch sizes of each
    transition measured on the observed change (:func:`observed_patches`;
    or ``patches``: the :class:`ExpanderPatcher` settings, a number = the
    Expander's share); ``isometry`` is not measurable and given (Dinamica's
    usual 1.5).
    """
    from ._markov import Markov
    from ._model import Model
    from ._patches import ExpanderPatcher
    from ._woe import WeightsOfEvidence

    markov = Markov.from_maps(t0, t1, steps=steps, nodata=nodata)
    rates = markov.multi
    transitions = [(f, t) for f in rates.index for t in rates.columns if f != t and rates.loc[f, t] > 0]
    w = {} if woe is None else ({"increment": woe} if np.isscalar(woe) else dict(woe))
    w.setdefault("transitions", transitions)
    potential = WeightsOfEvidence(w.pop("ranges", None), **w).fit(t0, t1, drivers)
    if isinstance(patches, dict):
        alloc = ExpanderPatcher(**{"prune_factor": prune_factor, **patches})
    else:
        obs = observed_patches(t0, t1, transitions, nodata=nodata).set_index(["from", "to"])
        share = {tr: (float(patches) if np.isscalar(patches) and patches != "observed" else obs.loc[tr, "expansion"])
                 for tr in transitions}
        alloc = ExpanderPatcher(
            expansion=share,
            expander={tr: (obs.loc[tr, "expansion_mean"], obs.loc[tr, "expansion_variance"], isometry)
                      for tr in transitions},
            patcher={tr: (obs.loc[tr, "new_mean"], obs.loc[tr, "new_variance"], isometry) for tr in transitions},
            prune_factor=prune_factor)
    return Model(potential, markov, alloc, rules, dynamic)


def plus(t0, t1, drivers, *, steps: int = 1, estimator=None, sample=0.05, demand=None, rules=None, dynamic=None,
         seed=None, nodata=None, **cars):
    """PLUS (Liang et al. 2021): LEAS (a random forest per class on the expansion areas) and CARS.

    ``estimator`` is any scikit-learn classifier (default: a random forest
    of 50 trees, the paper's), fitted for each class that expanded between
    the maps on a ``sample`` of the cells (the paper's 5 %, drawn from
    the expansion cells and the others separately); the demand is the Markov chain of the two maps projected
    (PLUS's default) unless ``demand`` is given; ``cars`` are :class:`CARS`'s
    settings.
    """
    from ._estimator import Estimator
    from ._markov import Markov
    from ._model import Model
    from ._plus import CARS

    if estimator is None:
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1)
    a, b = read_map(t0, nodata).as_float(), read_map(t1, nodata).as_float()
    gained = (a != b) & ~np.isnan(a) & ~np.isnan(b)
    grow = sorted({_class_value(c) for c in np.unique(b[gained])})  # LEAS: the classes that expanded
    if not grow:
        raise ValueError("no class expanded between the two maps")
    potential = Estimator(estimator, target="expansion", classes=grow, sample=sample, stratify=True, seed=seed).fit(
        t0, drivers, t1=t1, nodata=nodata)
    if demand is None:
        demand = Markov.from_maps(t0, t1, steps=steps, nodata=nodata)
    return Model(potential, demand, CARS(**cars), rules, dynamic)


def molusce(t0, t1, drivers, *, bins=None, iterations: int = 1, unit_cell: float = 1, dynamic=None, nodata=None):
    """MOLUSCE: its weights of evidence (:class:`MolusceWoE`) and simulator (:class:`MolusceCA`).

    Each iteration repeats the change observed between the maps (their
    cross table); start the run from the later map, as MOLUSCE does.
    """
    from ._model import Model
    from ._molusce import MolusceCA, MolusceWoE

    woe = MolusceWoE(bins, unit_cell).fit(t0, t1, drivers, nodata=nodata)
    a, b = read_map(t0, nodata).as_float().ravel(), read_map(t1, nodata).as_float().ravel()
    ok = ~(np.isnan(a) | np.isnan(b))
    cross = pd.crosstab(a[ok], b[ok])
    return Model(woe, cross, MolusceCA(iterations), None, dynamic)
