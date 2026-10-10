# SPDX-License-Identifier: GPL-3.0-or-later
"""The rules of land change: allowed conversions, elasticities, restrictions, neighbourhoods, history.

The decision rules of CLUE-S (Verburg et al. 2002) as lulcc applies them:
a conversion matrix (rows the current class, columns the next one) with
codes for time limits and demand directions, the elasticity added to the
suitability of a cell's own class, neighbourhood thresholds, restricted
areas where cells keep their class, and the automatic conversions of cells
whose class is no longer allowed where they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import _core
from ._change import _class_value
from ._maps import read_map

__all__ = ["Neighbourhood", "Rules"]


@dataclass
class Neighbourhood:
    """Neighbourhood thresholds (lulcc's ``NeighbRasterStack`` and ``nb.rules``).

    A class may be allocated to a cell only where the mean of ``weights x
    presence of the class`` over the window around the cell reaches its
    threshold (cells outside the map and missing cells are left out of the
    mean).

    Parameters
    ----------
    rules
        ``{class: threshold}`` (0-1 for a window of ones: the share of the
        window in the class).
    weights
        The window (odd sides), the same for every class, or ``{class:
        window}`` (default 3 x 3 ones).
    """

    rules: dict
    weights: object = None

    def window(self, cls) -> np.ndarray:
        w = self.weights
        if isinstance(w, dict):
            w = w[cls]
        if w is None:
            w = np.ones((3, 3))
        return np.ascontiguousarray(w, dtype=np.float64)

    def allow(self, landscape: np.ndarray, classes: list) -> np.ndarray:
        """1 where each class may go, NaN where not: (cells of the map, classes)."""
        rules = [(float(c), float(threshold), self.window(c))
                 for c, threshold in ((_class_value(k), v) for k, v in self.rules.items()) if c in classes]
        land = np.ascontiguousarray(landscape, dtype=np.float64)
        return _core.sim_neighbourhood_allow(land, np.asarray(classes, dtype=np.float64), rules)


@dataclass
class Rules:
    """Decision rules of the allocation (lulcc's ``rules``, ``elas``, ``mask``, ``hist``, ``neighb``).

    Parameters
    ----------
    allowed
        Conversion matrix, rows the current class, columns the next (a
        DataFrame named by class, or an array in the model's class order):
        ``1`` allowed, ``0`` not; ``-1`` the cell may leave its class only
        while the demand for that class decreases; ``-2`` the next class may
        be taken only while its demand increases; ``100 + t``: allowed only
        while the cell has had its class for less than ``t`` steps; ``1000 +
        t``: allowed only once it has had it for ``t`` steps or more.
    elasticity
        Added to the suitability of each cell's own class (CLUE-S's
        conversion elasticity, 0 to 1; one value for all, ``{class: value}``
        or a list in the class order). Used by the CLUE and CLUMondo
        allocations (not by the ordered one, as in lulcc).
    restrictions
        Map of restricted cells (``True`` or non-zero: the cell keeps its
        class), lulcc's mask with the opposite convention.
    history
        Map of the steps each cell has had its class (default 1 everywhere,
        as lulcc), updated at each step: +1 where the class stays, 0 where
        it changes.
    neighbourhood
        :class:`Neighbourhood` thresholds (lulcc's allocations).
    allow_maps
        CLUMondo: ``{(from, to): map}``, the conversion code of each cell for
        that pair (0 not allowed, 1 allowed, or an age code below).

    Notes
    -----
    With the :class:`CLUMondo` allocation the conversion matrix takes
    CLUMondo's codes instead: ``1`` allowed, ``0`` not; ``100 + t``: only
    once the cell has had its system for ``t`` years; ``-100 - t``: only
    until then (the system must leave after ``t`` years when the code is
    on the diagonal); ``1000 + t``: autonomous change (succession) after
    ``t`` years, whatever the demand. ``elasticity`` is CLUMondo's
    conversion resistance, ``history`` the starting ages (a map, or a
    number: random ages from 1 to it, as CLUMondo draws them).

    With CLUMondo, rules may also be given by the role of the transitions
    (``roles``, a :class:`Roles`):

    - ``allowed`` as ``{role: code}`` (``{"expansion": 0}``; other pairs 1);
    - ``transition_resistance``: ``{role or (from, to): value}`` subtracted
      from the score of making that transition;
    - ``conversion_order``: roles from the first tried to the last, as a
      resistance of 0.1 per position (``["intensification",
      "expansion"]``: expansion 0.1); ``transition_resistance`` overrides;
    - ``caps``: :class:`Cap` limits per year (``Cap("expansion",
      area=...)``);
    - ``succession``: :class:`Succession` (autonomous changes).
    """

    allowed: object = None
    elasticity: object = None
    restrictions: object = None
    history: object = None
    neighbourhood: Neighbourhood | None = None
    allow_maps: dict | None = None
    roles: object = None
    transition_resistance: dict | None = None
    conversion_order: list | None = None
    caps: list | None = None
    succession: list | None = None
    _matrix: np.ndarray | None = field(default=None, repr=False)

    def matrix(self, classes: list) -> np.ndarray | None:
        if self.allowed is None or (isinstance(self.allowed, dict) and not isinstance(self.allowed, pd.DataFrame)):
            return None
        if isinstance(self.allowed, pd.DataFrame):
            a = self.allowed.copy()
            a.index = [_class_value(c) for c in a.index]
            a.columns = [_class_value(c) for c in a.columns]
            return a.loc[classes, classes].to_numpy(dtype=np.int64)
        m = np.asarray(self.allowed, dtype=np.int64)
        if m.shape != (len(classes), len(classes)):
            raise ValueError("the conversion matrix must be square, one row and column per class")
        return m

    def elasticities(self, classes: list) -> np.ndarray:
        e = self.elasticity
        if e is None:
            return np.zeros(len(classes))
        if isinstance(e, dict):
            e = {_class_value(k): v for k, v in e.items()}
            return np.array([float(e.get(c, 0.0)) for c in classes])
        e = np.asarray(e, dtype=np.float64)
        if e.ndim == 0:
            return np.full(len(classes), float(e))
        if e.size != len(classes):
            raise ValueError("one elasticity per class")
        return e

    def mask_values(self, cells: np.ndarray, size: int) -> np.ndarray:
        """lulcc's mask on the cells: 0 restricted, 1 free."""
        if self.restrictions is None:
            return np.ones(cells.size, dtype=np.int64)
        r = read_map(self.restrictions).as_float().ravel()
        if r.size != size:
            raise ValueError("the restrictions map differs in size from the landscape")
        restricted = np.nan_to_num(r[cells], nan=0.0) != 0
        return np.where(restricted, 0, 1).astype(np.int64)

    def history_values(self, cells: np.ndarray, size: int) -> np.ndarray:
        if self.history is None:
            return np.ones(cells.size, dtype=np.int64)
        h = read_map(self.history).as_float().ravel()
        if h.size != size:
            raise ValueError("the history map differs in size from the landscape")
        return np.nan_to_num(h[cells], nan=1.0).astype(np.int64)


def allow_matrix(x: np.ndarray, hist: np.ndarray, classes: list, cd: np.ndarray, rules: np.ndarray) -> np.ndarray:
    """lulcc's allow(): 1 where a conversion is allowed, NaN where not; (cells, classes)."""
    return _core.sim_allow_matrix(np.asarray(x, dtype=np.float64), np.asarray(hist, dtype=np.int64),
                                  np.asarray(classes, dtype=np.float64), np.asarray(cd, dtype=np.float64),
                                  np.asarray(rules, dtype=np.int64))


def auto_convert(x: np.ndarray, prob: np.ndarray, classes: list, mask: np.ndarray) -> np.ndarray:
    """lulcc's autoconvert: the class a cell is forced to take this step (NaN: none).

    A cell whose own class is no longer allowed takes the class of highest
    suitability (the first in case of ties; its own class if none is
    allowed); a restricted cell keeps its class.
    """
    return _core.sim_auto_convert(np.asarray(x, dtype=np.float64), np.asarray(prob, dtype=np.float64),
                                  np.asarray(classes, dtype=np.float64), np.asarray(mask, dtype=np.int64))
