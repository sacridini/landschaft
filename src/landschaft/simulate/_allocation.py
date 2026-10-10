# SPDX-License-Identifier: GPL-3.0-or-later
"""Allocation of the demanded change: CLUE-S's iterative competition and the ordered procedure."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .. import _core
from ._change import _class_value

__all__ = ["CLUE", "Ordered"]


@dataclass
class CLUE:
    """CLUE-S allocation (Verburg et al. 2002), as lulcc's ``CluesModel``.

    Every cell takes the class of highest suitability plus an iteration
    variable per class; the variables move by ``scale`` x (demand -
    allocated cells) until the allocation meets the demand. Each pass
    jitters the suitabilities by U(-jitter, jitter).

    Parameters
    ----------
    jitter
        Half-width of the random perturbation of the suitabilities.
    scale
        Step of the iteration variables per cell of difference.
    max_iter
        Passes before giving up (the allocation of the last pass is kept).
    max_diff, mean_diff
        Largest and mean absolute difference between demand and allocation
        (cells) accepted. lulcc averages the signed differences, which add
        up to 0, so its mean criterion never acts; here it is the mean of
        the absolute differences, as CLUE-S defines it.
    """

    jitter: float = 0.0001
    scale: float = 0.0005
    max_iter: int = 1000
    max_diff: float = 5.0
    mean_diff: float = 5.0
    uses_elasticity = True

    def allocate(self, tprob: np.ndarray, x: np.ndarray, demand: np.ndarray, classes: list, rng) -> tuple[np.ndarray, dict]:
        r = _core.sim_clues(np.ascontiguousarray(tprob.T), np.ascontiguousarray(x, dtype=np.float64),
                            np.ascontiguousarray(demand, dtype=np.float64), np.asarray(classes, dtype=np.float64),
                            float(self.jitter), float(self.scale), float(self.max_iter), float(self.max_diff),
                            float(self.mean_diff), rng)
        info = {k: r[k] for k in ("iterations", "converged", "max_diff", "mean_diff")}
        if not info["converged"]:
            warnings.warn(f"the allocation did not meet the demand in {self.max_iter} iterations (largest "
                          f"difference {info['max_diff']:g} cells): the last one is kept", stacklevel=4)
        return r["allocation"], info


@dataclass
class Ordered:
    """The ordered allocation of Fuchs et al. (2013) with stochastic choice, as lulcc's ``OrderedModel``.

    The classes are served one at a time in ``order``: a class whose demand
    grows takes the cells of other classes in decreasing suitability, a
    class whose demand shrinks releases its cells of lowest suitability
    (to the classes served after it). With ``stochastic``, cells are drawn
    by comparing their suitability with uniform numbers (R's generator), in
    suitability order.

    Parameters
    ----------
    order
        The classes, in the order they are served (all of them).
    stochastic
        Draw the cells (default) or take the most suitable ones.
    """

    order: list
    stochastic: bool = True
    uses_elasticity = False

    def allocate(self, tprob: np.ndarray, x: np.ndarray, demand: np.ndarray, classes: list, rng) -> tuple[np.ndarray, dict]:
        order = [_class_value(c) for c in self.order]
        if sorted(order) != sorted(classes):
            raise ValueError("'order' must contain the model's classes, each once")
        r = _core.sim_ordered(np.ascontiguousarray(tprob, dtype=np.float64), np.ascontiguousarray(x, dtype=np.float64),
                              np.ascontiguousarray(demand, dtype=np.float64), np.asarray(classes, dtype=np.float64),
                              np.asarray(order, dtype=np.float64), bool(self.stochastic), rng)
        short = int(r["short"])
        if short:
            warnings.warn(f"the demand was not met by {short} cells: too few cells could take a class "
                          "(no suitability, or 1000 draws without enough)", stacklevel=4)
        return r["allocation"], {"iterations": int(r["iterations"]), "converged": bool(r["converged"])}
