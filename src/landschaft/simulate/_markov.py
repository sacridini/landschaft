# SPDX-License-Identifier: GPL-3.0-or-later
"""Markov chains of land change: transition rates over a period and per time step."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core
from .._threads import threaded
from ._change import _class_value, _classes, _count
from ._maps import read_map

__all__ = ["Markov"]


def _matrix_root(p: np.ndarray, n: int) -> np.ndarray:
    """The principal n-th root of a transition matrix.

    ``expm(logm(P) / n)``: the principal root, within a few units of 1e-14
    of the exact one (the eigendecomposition Dinamica EGO uses,
    ``V diag(w^(1/n)) V^-1``, gives the same root but loses up to 1e-11 on
    small rates, and Dinamica gives up when eigenvalues are complex or
    repeated). Singular matrices (a class that always turns into others)
    have no logarithm: their root comes from the eigendecomposition.
    Negative real eigenvalues have no real root: an error.
    """
    return _core.sim_markov_root(np.ascontiguousarray(p, dtype=np.float64), int(n))


class Markov:
    """A Markov chain of land change: the transition probabilities of a period and of each time step.

    ``single`` holds the probability that a cell of class i at the start of
    the period is of class j at its end (every cell counted once), ``multi``
    the probabilities per time step when the period spans ``steps`` steps:
    the principal ``steps``-th root of ``single`` (Dinamica EGO's
    DetermineTransitionMatrix, single-step and multi-step matrices).

    Create one with :meth:`from_maps` (or ``Changes.markov``), or from a
    matrix of probabilities.

    Parameters
    ----------
    matrix
        Transition probabilities of the period (rows sum to 1).
    classes
        The class of each row and column.
    steps
        Time steps in the period.
    """

    def __init__(self, matrix, classes, steps: int = 1):
        p = np.asarray(matrix, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] != p.shape[1]:
            raise ValueError("the transition matrix must be square")
        if len(classes) != p.shape[0]:
            raise ValueError("one class per row of the transition matrix")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.classes = [_class_value(c) for c in classes]
        self.steps = int(steps)
        self._single = p
        self._multi = None

    @classmethod
    def from_counts(cls, counts, classes, steps: int = 1) -> Markov:
        """From a matrix of cell counts (rows: classes at the start); classes without cells at the start are dropped."""
        p, kept = _core.sim_markov_from_counts(np.ascontiguousarray(counts, dtype=np.float64))
        return cls(p, np.asarray(classes)[kept], steps)

    @classmethod
    @threaded
    def from_maps(cls, t0, t1, *, steps: int = 1, nodata=None) -> Markov:
        """The chain of the change from map ``t0`` to map ``t1`` over ``steps`` time steps.

        Examples
        --------
        >>> import landschaft as ls
        >>> lu = ls.load_raster(ls.simulate.example_data("pie_landuse"))
        >>> mk = ls.simulate.Markov.from_maps(lu[0], lu[2], steps=14)
        >>> mk.single.shape
        (3, 3)
        """
        a, b = read_map(t0, nodata), read_map(t1, nodata)
        classes = _classes([a, b])
        return cls.from_counts(_count(a, b, classes), classes, steps)

    def _frame(self, m) -> pd.DataFrame:
        return pd.DataFrame(m, index=pd.Index(self.classes, name="from"), columns=pd.Index(self.classes, name="to"))

    @property
    def single(self) -> pd.DataFrame:
        """Transition probabilities over the whole period."""
        return self._frame(self._single)

    @property
    def multi(self) -> pd.DataFrame:
        """Transition probabilities per time step (the principal root of :attr:`single`)."""
        if self._multi is None:
            self._multi = _matrix_root(self._single, self.steps)
        return self._frame(self._multi)

    def rates(self, *, multi: bool = True) -> pd.DataFrame:
        """The transition rates as Dinamica EGO lists them: off-diagonal, positive, ``From``, ``To``, ``Rate``.

        Negative rates (roots that are not stochastic matrices) are left
        out, as Dinamica does; a cell then keeps its class with the
        remaining probability.
        """
        m = (self.multi if multi else self.single).to_numpy()
        rows = [(self.classes[i], self.classes[j], m[i, j]) for i in range(m.shape[0]) for j in range(m.shape[1])
                if i != j and m[i, j] > 0]
        return pd.DataFrame(rows, columns=["From", "To", "Rate"])

    def project(self, areas, steps: int | None = None, *, multi: bool = True) -> pd.DataFrame:
        """Areas of each class after each step: ``areas @ P`` repeated.

        Parameters
        ----------
        areas
            The area (or cells) of each class now: a sequence in the order of
            :attr:`classes`, a ``dict`` or a ``Series`` indexed by class.
        steps
            Steps to project (default: the chain's ``steps`` with
            ``multi``, else 1).
        multi
            Use the per-step matrix (default) or the period's.

        Returns
        -------
        pandas.DataFrame
            One row per step (0 = the given areas), one column per class.
        """
        if isinstance(areas, (dict, pd.Series)):
            a = np.array([float(dict(areas).get(c, 0.0)) for c in self.classes])
        else:
            a = np.asarray(areas, dtype=np.float64)
            if a.size != len(self.classes):
                raise ValueError(f"{a.size} areas for {len(self.classes)} classes")
        p = (self.multi if multi else self.single).to_numpy()
        n = (self.steps if multi else 1) if steps is None else int(steps)
        out = _core.sim_markov_project(a, np.ascontiguousarray(p, dtype=np.float64), n)
        return pd.DataFrame(out, index=pd.Index(range(n + 1), name="step"), columns=self.classes)

    def __repr__(self) -> str:
        return f"Markov({len(self.classes)} classes, steps={self.steps})"
