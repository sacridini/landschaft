# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-criteria evaluation with Saaty's pairwise comparisons (AHP), as MOLUSCE's MCE.

The weights of the factors are the principal eigenvector of the pairwise
comparison matrix, normalised to sum 1; its consistency ratio is
(lambda_max - n) / (n - 1) over Saaty's random index. The potential of a
transition is the weighted sum of the factors rescaled to 0-1 by their
minimum and maximum. MOLUSCE truncates every weighted term to an integer
percentage before adding them (losing up to one unit per factor); here the
sum is kept in floating point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .._threads import threaded
from ._change import _class_value
from ._estimator import _stack_like
from ._maps import read_map
from ._woe import _drivers

__all__ = ["MultiCriteria"]

# Saaty's random consistency index (as MOLUSCE tabulates it)
_RANDOM_INDEX = {2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48,
                 13: 1.56, 14: 1.57, 15: 1.59, 16: 1.60, 17: 1.61, 18: 1.62, 19: 1.63, 20: 1.63, 21: 1.64, 22: 1.65,
                 23: 1.65, 24: 1.66, 25: 1.66, 26: 1.67, 27: 1.67, 28: 1.67, 29: 1.68, 30: 1.68, 31: 1.68,
                 32: 1.69, 33: 1.69, 34: 1.69, 35: 1.69, 36: 1.70, 37: 1.70, 38: 1.70, 39: 1.70}


class MultiCriteria:
    """Potential from weighted factors, the weights from pairwise comparisons (AHP; MOLUSCE's MCE).

    Parameters
    ----------
    comparisons
        The pairwise comparison matrix: a square DataFrame indexed and named
        by the drivers (``a_ij`` = how much more important driver i is than
        j, 1 to 9, ``a_ji = 1 / a_ij``, ``a_ii = 1``), or an array with
        ``drivers``.
    drivers
        The driver names, in the order of an array's rows.
    transition
        ``(from, to)``: the potential is given only to the cells of the
        initial class (default: every cell).

    Examples
    --------
    >>> import pandas as pd
    >>> import landschaft as ls
    >>> a = pd.DataFrame([[1, 3, 5], [1 / 3, 1, 2], [1 / 5, 1 / 2, 1]],
    ...                  index=["roads", "slope", "soil"], columns=["roads", "slope", "soil"])
    >>> mce = ls.simulate.MultiCriteria(a)
    >>> round(float(mce.weights["roads"]), 3), mce.consistency < 0.1
    (0.648, True)
    """

    def __init__(self, comparisons, *, drivers=None, transition=None):
        if isinstance(comparisons, pd.DataFrame):
            names = [str(c) for c in comparisons.columns]
            if [str(i) for i in comparisons.index] != names:
                raise ValueError("the comparison matrix must name its rows and columns alike")
            a = comparisons.to_numpy(dtype=np.float64)
        else:
            a = np.asarray(comparisons, dtype=np.float64)
            if drivers is None:
                raise ValueError("give the driver names of the comparison matrix")
            names = [str(d) for d in drivers]
        n = len(names)
        if a.shape != (n, n):
            raise ValueError("the comparison matrix must be n x n, one row per driver")
        for i in range(n):
            if a[i, i] != 1:
                raise ValueError("the diagonal of the comparison matrix must be 1")
            for j in range(i + 1, n):
                if abs(a[i, j] * a[j, i] - 1) > 1e-6:
                    raise ValueError("the comparison matrix must be reciprocal: a[i, j] * a[j, i] = 1")
        self.drivers = names
        self.matrix = a
        self.transition = None if transition is None else tuple(_class_value(t) for t in transition)
        w, v = np.linalg.eig(a)
        top = np.max(w)
        k = list(w).index(top)
        weights = np.array([x.real for x in v[:, k]])
        self._weights = weights / sum(weights)
        lam = top.real
        if n > 2:
            ri = _RANDOM_INDEX.get(n)
            self._consistency = (lam - n) / (n - 1) / ri if ri else -1.0
        else:
            self._consistency = 0.0
        self.lambda_max = float(lam)

    @property
    def weights(self) -> pd.Series:
        """The weight of each driver (sum 1)."""
        return pd.Series(self._weights, index=self.drivers, name="weight")

    @property
    def consistency(self) -> float:
        """Saaty's consistency ratio (acceptable below 0.1; 0 for two drivers, -1 beyond the table)."""
        return float(self._consistency)

    def fit(self, *args, **kwargs) -> MultiCriteria:
        """Nothing to fit: the weights come from the comparisons."""
        return self

    @threaded
    def predict(self, drivers, *, landscape=None, nodata=None) -> xr.DataArray:
        """The potential: sum of weight x driver rescaled to 0-1 (by its minimum and maximum).

        Parameters
        ----------
        drivers
            The factors (Dataset, ``dict`` or DataArray with a band per factor).
        landscape
            With ``transition``: only the cells of its initial class get a
            potential (the others 0, missing cells NaN).

        Returns
        -------
        xarray.DataArray
            ``(transition, y, x)`` with one layer.
        """
        maps = _drivers(drivers)
        missing = set(self.drivers) - set(maps)
        if missing:
            raise ValueError(f"drivers lack {sorted(missing)}")
        first = maps[self.drivers[0]]
        total = np.zeros(first.shape)
        for w, name in zip(self._weights, self.drivers):
            v = maps[name].as_float()
            lo, hi = np.nanmin(v), np.nanmax(v)
            total = total + (v - lo) / (hi - lo) * w
        if landscape is not None:
            land = read_map(landscape, nodata).as_float()
            total = np.where(np.isnan(land), np.nan, total)
            if self.transition is not None:
                total = np.where(land == self.transition[0], total, np.where(np.isnan(total), np.nan, 0.0))
        name = "potential" if self.transition is None else f"probability_{self.transition[0]}_to_{self.transition[1]}"
        return _stack_like([total], [name], "transition", first.template)

    probability = predict
    potential = predict
