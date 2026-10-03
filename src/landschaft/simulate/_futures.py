# SPDX-License-Identifier: GPL-3.0-or-later
"""Urban growth by patches: FUTURES (Meentemeyer et al. 2013; Petrasova et al.), as its GRASS GIS modules run it.

The pieces of ``r.futures``: the development pressure of the developed cells
around each cell (``development_pressure``, r.futures.devpressure), the
cells each subregion develops at each step (``DevelopmentDemand``, by hand
or from population with r.futures.demand's regressions), the development
potential (``DevelopmentPotential``: intercept, pressure and predictor
coefficients per subregion, r.futures.potential's table) and the patch
growing allocation (``PatchGrowing``, r.futures.simulation): seeds drawn on
the development probability, patch sizes drawn from a library of observed
patches, patches grown cell by cell by probability and compactness, the
pressure updated around every new cell. The same maps as GRASS bit for bit
with the same ``random_seed``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from ._maps import read_map

__all__ = ["DevelopmentDemand", "DevelopmentPotential", "Flooding", "PatchGrowing", "calibrate_patches",
           "development_pressure", "patch_library"]

_APPROACHES = {"occurrence": 0, "gravity": 1, "kernel": 2}

# r.futures.simulation's predefined zoning effects (zone id -> effect)
ZONING_EFFECTS = {100: 0.0, 101: -0.124, 110: -0.440, 120: -0.656, 130: -0.78, 131: -0.79, 200: -0.157,
                  201: -0.026, 202: -0.127, 203: -0.817, 300: -0.105, 301: 0.115, 302: -1.0, 0: 0.0}


def _distance_matrix(size: int) -> np.ndarray:
    k = np.arange(2 * size + 1) - size
    d = np.sqrt((k[:, None] ** 2 + k[None, :] ** 2).astype(np.float64))
    return np.where(d <= size, d, 0.0)


def development_pressure(developed, size: int, *, method: str = "gravity", gamma: float | None = 1.5,
                         scaling_factor: float | None = 1.0, pad: bool = True, nodata=None) -> xr.DataArray:
    """The development pressure of each cell (r.futures.devpressure).

    The sum over the developed cells within ``size`` cells (circular window,
    the cell itself left out) of 1 (``"occurrence"``), ``scaling_factor /
    d^gamma`` (``"gravity"``) or ``scaling_factor * exp(-2 d / gamma)``
    (``"kernel"``), as GRASS's r.mfilter computes it.

    Parameters
    ----------
    developed
        1 developed, 0 not, NaN excluded.
    size
        Radius of the window, in cells.
    method, gamma, scaling_factor
        The decay of the pressure with distance.
    pad
        The module's ``-n`` flag (default): the map is padded with
        undeveloped cells so the cells near its edges get their pressure,
        and excluded cells count as undeveloped. Without it, as r.mfilter
        does, cells within ``size`` of the edges keep their input value and
        a window touching an excluded cell gives NaN.

    Returns
    -------
    xarray.DataArray
        float32, NaN where excluded.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> dev = np.zeros((7, 7))
    >>> dev[3, 3] = 1
    >>> p = ls.simulate.development_pressure(dev, 2, method="occurrence")
    >>> int(p.values.sum())
    12
    """
    if method not in _APPROACHES:
        raise ValueError("method must be 'occurrence', 'gravity' or 'kernel'")
    if method != "occurrence" and (gamma is None or scaling_factor is None):
        raise ValueError("methods gravity and kernel need gamma and scaling_factor")
    size = int(size)
    m = read_map(developed, nodata)
    v = m.as_float()
    d = _distance_matrix(size)
    if method == "occurrence":
        w = np.where(d > 0, 1.0, 0.0)
    elif method == "gravity":
        with np.errstate(divide="ignore"):
            den = np.power(d, float(gamma))
            w = float(scaling_factor) / den
        w[den == 0] = 0
    else:
        w = np.where(d > 0, float(scaling_factor) * np.exp(-2 * d / float(gamma)), 0.0)
    # the filter goes through text (str of each float64) and back: the same doubles
    w = np.array([[float(str(x)) for x in row] for row in w])
    if pad:
        nan = np.isnan(v)
        big = np.pad(np.where(nan, 0.0, v), size)
        out = _core.sim_futures_mfilter(np.ascontiguousarray(big), np.ascontiguousarray(w))
        out = out[size: size + v.shape[0], size: size + v.shape[1]].copy()
        out[nan] = np.nan
    else:
        out = _core.sim_futures_mfilter(np.ascontiguousarray(v), np.ascontiguousarray(w))
    t = m.template
    if t is not None:
        return t.copy(data=out).rename("development_pressure")
    return xr.DataArray(out, dims=("y", "x"), name="development_pressure")


# ----------------------------------------------------------------------------- demand

def _exp_approach(x, a, b, c):
    return (1 - np.exp(-a * (x - b))) + c


def _logarithmic2(x, a, b, c):
    return a + b * np.log(x - c)


def _logarithmic(x, a, b):
    return a + b * np.log(x)


def _magnitude(x):
    return int(math.log10(x))


class DevelopmentDemand:
    """Cells each subregion develops at each step (r.futures.simulation's ``demand`` file).

    Parameters
    ----------
    table
        A DataFrame with a row per simulated step (indexed by year) and a
        column per subregion (its value in the subregions map); or an array
        with ``subregions``. Values are rounded to whole cells (``lroundf``).
    subregions
        The subregion of each column of an array.
    population
        The population each subregion adds per step, as ``table`` (used by
        the redistribution of :class:`Flooding`).

    Examples
    --------
    >>> import landschaft as ls
    >>> d = ls.simulate.DevelopmentDemand({1: [10, 12], 2: [5, 0]})
    >>> d.steps, d.subregions
    (2, [1, 2])
    """

    def __init__(self, table, subregions=None, population=None):
        if isinstance(table, dict):
            table = pd.DataFrame(table)
        if not isinstance(table, pd.DataFrame):
            a = np.asarray(table, dtype=np.float64)
            if subregions is None or a.ndim != 2 or len(subregions) != a.shape[1]:
                raise ValueError("give the subregion of each column")
            table = pd.DataFrame(a, columns=list(subregions))
        self.table = table.copy()
        self.table.columns = [int(c) for c in self.table.columns]
        if (self.table.to_numpy(dtype=np.float64) < 0).any():
            raise ValueError("demands must not be negative")
        self.population: pd.DataFrame | None = None if population is None else pd.DataFrame(population).copy()
        if self.population is not None:
            self.population.columns = [int(c) for c in self.population.columns]
            if len(self.population) != len(self.table):
                raise ValueError("the population needs a row per step of the demand")
        self.curves: pd.DataFrame | None = None

    @property
    def steps(self) -> int:
        return len(self.table)

    @property
    def subregions(self) -> list:
        return list(self.table.columns)

    @classmethod
    def from_population(cls, developed, subregions, observed_population, projected_population,
                        simulation_times, *, methods=("linear", "logarithmic"), nodata=None) -> DevelopmentDemand:
        """The demand from the relation of developed cells to population (r.futures.demand).

        For each subregion, the developed cells of the observed maps are
        regressed on the observed population (``"linear"``,
        ``"logarithmic"``, ``"exponential"``, ``"logarithmic2"``,
        ``"exp_approach"``; the smallest RMSE wins; a decreasing fit falls
        back to the latest cells per person), the population is interpolated
        linearly at ``simulation_times`` and the demand is the yearly
        difference of the predicted cells, truncated to whole cells.

        Parameters
        ----------
        developed
            The observed maps of developed cells (1 developed), one per
            observed time (a list or a DataArray with a band per time).
        subregions
            The subregions map.
        observed_population, projected_population
            DataFrames with the time as first column (or index name) and a
            column per subregion.
        simulation_times
            The times of the simulation, the first being the start.
        methods
            The candidate regressions.

        Returns
        -------
        DevelopmentDemand
            With ``.population`` (the population added per step) and
            ``.curves`` (method, coefficients and RMSE per subregion).
        """
        obs = _population_frame(observed_population)
        proj = _population_frame(projected_population)
        if isinstance(developed, xr.DataArray) and developed.ndim == 3:
            maps = [developed[k] for k in range(developed.shape[0])]
        else:
            maps = list(developed)
        if len(maps) != len(obs):
            raise ValueError("one developed map per observed time")
        sub = read_map(subregions, nodata).as_float()
        methods = list(methods)
        unknown = set(methods) - {"linear", "logarithmic", "exponential", "logarithmic2", "exp_approach"}
        if unknown:
            raise ValueError(f"unknown methods {sorted(unknown)}")
        if len(maps) <= 2 and ("exp_approach" in methods or "logarithmic2" in methods):
            raise ValueError("methods exp_approach and logarithmic2 need at least 3 observed maps")
        times = np.asarray(obs.index, dtype=np.float64)
        ptimes = np.asarray(proj.index, dtype=np.float64)
        sim_times = [float(t) for t in simulation_times]
        zones = np.unique(sub[~np.isnan(sub)])
        developed_cells = {}
        for m in maps:
            v = read_map(m, nodata).as_float()
            for z in zones:
                cells = (sub == z) & ~np.isnan(v)
                developed_cells.setdefault(z, []).append(int(np.sum(v[cells])))
        ids = sorted(str(int(z)) for z in zones)  # r.univar's zones, sorted as text
        by_id = {str(int(z)): developed_cells[z] for z in zones}
        demand, pop_demand, curves = {}, {}, []
        for sid in ids:
            if int(sid) not in obs.columns:
                raise ValueError(f"no observed population for subregion {sid}")
            dev = np.asarray(by_id[sid], dtype=np.float64)
            reg_pop0 = obs[int(sid)].to_numpy(dtype=np.float64)
            pop_sim = np.interp(sim_times, np.append(times, ptimes),
                                np.append(reg_pop0, proj[int(sid)].to_numpy(dtype=np.float64)))
            rmse, predicted, coeff = {}, {}, {}
            for method in methods:
                reg_pop = reg_pop0
                if method in ("exp_approach", "logarithmic2"):
                    from scipy.optimize import curve_fit

                    magn = float(np.power(10, max(_magnitude(np.max(reg_pop)), _magnitude(np.max(dev)))))
                    x, y = reg_pop / magn, dev / magn
                    f = _exp_approach if method == "exp_approach" else _logarithmic2
                    if method == "exp_approach":
                        initial = (0.5, np.mean(x), np.mean(y))
                    else:
                        popt, _ = curve_fit(_logarithmic, x, y)
                        initial = (popt[0], popt[1], 0)
                    try:
                        with np.errstate(invalid="warn"), warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            popt, _ = curve_fit(f, x, y, p0=initial)
                        if np.isnan(popt).any():
                            raise RuntimeError
                        if method == "logarithmic2" and np.any(pop_sim / magn <= popt[-1]):
                            raise RuntimeError
                    except (FloatingPointError, RuntimeError):
                        warnings.warn(f"method {method!r} cannot converge for subregion {sid}", stacklevel=2)
                        rmse[method] = float(np.iinfo(np.int64).max)
                        coeff[method] = (np.nan, np.nan, np.nan)
                        predicted[method] = np.zeros(len(pop_sim))
                    else:
                        predicted[method] = f(pop_sim / magn, *popt) * magn
                        r = f(x, *popt) * magn - dev
                        coeff[method] = [each * magn for each in popt]
                        rmse[method] = np.sqrt(np.sum(r * r) / (len(reg_pop) - 3)) if len(reg_pop) > 3 else 0
                else:
                    if method == "logarithmic":
                        reg_pop = np.log(reg_pop)
                    y = np.log(dev) if method == "exponential" else dev
                    a = np.vstack((reg_pop, np.ones(len(reg_pop)))).T
                    keep = ~np.isinf(a).any(axis=1)
                    m, c = np.linalg.lstsq(a[keep], np.asarray(y)[keep], rcond=None)[0]
                    coeff[method] = (m, c)
                    if method == "logarithmic":
                        with np.errstate(invalid="ignore", divide="ignore"):
                            p = np.where(pop_sim > 1, np.log(pop_sim) * m + c, 0)
                        predicted[method] = np.where(p > 0, p, 0)
                        r = (reg_pop * m + c) - dev
                    elif method == "exponential":
                        predicted[method] = np.exp(m * pop_sim + c)
                        r = np.exp(m * reg_pop + c) - dev
                    else:
                        predicted[method] = pop_sim * m + c
                        r = (reg_pop * m + c) - dev
                    rmse[method] = np.sqrt(np.sum(r * r) / (len(reg_pop) - 2)) if len(reg_pop) > 2 else 0
                if (method in ("linear", "logarithmic", "exp_approach") and coeff[method][0] < 0) or \
                        (method == "logarithmic2" and coeff[method][1] < 0):
                    m = dev[-1] / reg_pop0[-1]
                    coeff["fallback"] = (m, 0)
                    rmse["fallback"] = -1
                    predicted["fallback"] = pop_sim * m + 0
            best = min(rmse, key=rmse.get)
            d = np.diff(predicted[best])
            if np.any(d < 0):
                warnings.warn(f"subregion {sid} has negative numbers of newly developed cells, set to 0",
                              stacklevel=2)
                d[d < 0] = 0
            if coeff[best][0] < 0 or np.isnan(coeff[best][0]):
                per = math.ceil((dev[-1] - dev[0]) / float(times[-1] - times[0]))
                d.fill(max(0, per))
            demand[sid] = d
            pop_demand[sid] = np.diff(pop_sim)
            curves.append({"subregion": int(sid), "method": best, "coefficients": list(coeff[best]),
                           "rmse": float(rmse[best])})
        order = [c for c in obs.columns if str(c) in demand]
        index = pd.Index([int(t) for t in sim_times[1:]], name="year")
        out = cls(pd.DataFrame({c: np.trunc(demand[str(c)]) for c in order}, index=index))
        out.population = pd.DataFrame({c: np.trunc(pop_demand[str(c)]) for c in order}, index=index)
        out.curves = pd.DataFrame(curves).set_index("subregion")
        return out

    def __repr__(self) -> str:
        return f"DevelopmentDemand({self.steps} steps, {len(self.subregions)} subregions)"


def _population_frame(t) -> pd.DataFrame:
    """Population by time: the index, or the first column when the index is a plain range."""
    if not isinstance(t, pd.DataFrame):
        raise TypeError("the population must be a DataFrame")
    t = t.copy()
    if isinstance(t.index, pd.RangeIndex):
        t = t.set_index(t.columns[0])
    t.columns = [int(c) for c in t.columns]
    return t.astype(np.float64)


# ----------------------------------------------------------------------------- potential

class DevelopmentPotential:
    """The development potential's coefficients per subregion (r.futures.potential's table).

    ``p = 1 / (1 + exp(-(intercept + b_pressure * pressure + sum(b_k * x_k))))``
    with the coefficients of the cell's (potential) subregion.

    Parameters
    ----------
    table
        A DataFrame indexed by subregion with the columns ``intercept``,
        ``devpressure`` (or FUTURES' ``Intercept``, ``development_pressure``)
        and one per predictor (named as the drivers); or one row (a dict)
        for every subregion.

    Examples
    --------
    >>> import landschaft as ls
    >>> pot = ls.simulate.DevelopmentPotential({"intercept": -2.0, "devpressure": 0.5, "slope": -0.1})
    >>> pot.predictors
    ['slope']
    """

    def __init__(self, table):
        if isinstance(table, dict) and not any(isinstance(v, (list, tuple, np.ndarray, dict)) for v in table.values()):
            table = pd.DataFrame([table], index=pd.Index([None], name="subregion"))
        t = pd.DataFrame(table).copy()
        rename = {}
        for c in t.columns:
            low = str(c).lower()
            if low == "intercept":
                rename[c] = "intercept"
            elif low in ("devpressure", "development_pressure", "dev_pressure"):
                rename[c] = "devpressure"
        t = t.rename(columns=rename)
        if "intercept" not in t.columns or "devpressure" not in t.columns:
            raise ValueError("the table needs the columns intercept and devpressure")
        self.table = t.astype(np.float64)
        self.attrs: dict = {}

    @property
    def predictors(self) -> list:
        return [c for c in self.table.columns if c not in ("intercept", "devpressure")]

    @classmethod
    def fit(cls, samples: pd.DataFrame, *, response: str = "developed", pressure: str = "devpressure",
            predictors=None, subregions: str | None = "subregion", random: str | None = None,
            dredge: bool = False, min_variables: int = 1, max_variables: int | None = None,
            fixed=None) -> DevelopmentPotential:
        """Fit the potential on sampled cells (r.futures.potential).

        A binomial mixed model of new development: ``response ~ predictors +
        (1 | subregion)`` (R's ``glmer``, Laplace), or with a random slope
        ``(random | subregion)``; with one subregion, R's ``glm`` (bit for
        bit). The coefficients per subregion are the fixed effects plus the
        subregion's random effects (lme4's ``coef``).

        Parameters
        ----------
        samples
            One row per sampled cell (rows with missing values are dropped).
        response
            The 0/1 column of new development.
        pressure
            The development pressure column (the ``devpressure`` coefficient).
        predictors
            The predictor columns (default: all other columns).
        subregions
            The subregion column (``None``: one subregion).
        random
            Column with a random slope per subregion (usually the pressure,
            which then has no fixed effect, as r.futures.potential).
        dredge
            Choose the predictors by AIC over every subset (MuMIn's dredge)
            with ``min_variables`` to ``max_variables`` terms and the
            ``fixed`` ones always in.

        Returns
        -------
        DevelopmentPotential
            With ``attrs``: ``aic``, ``deviance`` (-2 log likelihood), and
            ``dredge`` (the ranked models) when used.
        """
        from ._glmer import dredge as _dredge
        from ._glmer import glmer

        t = samples.copy()
        if predictors is None:
            predictors = [c for c in t.columns if c not in (response, subregions, random)]
        predictors = [c for c in predictors if c not in (subregions, random, response)]
        cols = list(dict.fromkeys([*predictors, response] + ([subregions] if subregions else [])
                                  + ([random] if random else [])))
        t = t[cols].dropna()
        y = t[response].to_numpy(dtype=np.float64)
        single = subregions is None or t[subregions].nunique() <= 1
        level_ids = [None] if subregions is None else sorted(t[subregions].unique().tolist())

        def fit_terms(terms):
            X = np.column_stack([np.ones(len(t))] + [t[c].to_numpy(dtype=np.float64) for c in terms])
            names = ["(Intercept)", *terms]
            if single:
                fit = _core.sim_glm_logit(np.ascontiguousarray(X.T), y, 25, 1e-8)
                return _GlmFit(fit, names, level_ids[0])
            return glmer(y, X, t[subregions].to_numpy(), names=names, random=random,
                         Zcol=None if random is None else t[random].to_numpy(dtype=np.float64))
        ranking = None
        if dredge:
            ranking = _dredge(fit_terms, list(predictors), fixed=list(fixed or ()), m_min=int(min_variables),
                              m_max=None if max_variables is None else int(max_variables))
            best_terms, best = ranking[0]
        else:
            best_terms, best = list(predictors), fit_terms(list(predictors))
        coefs = best.coef()
        rows = {}
        for lev, row in coefs.items():
            out = {"intercept": row.get("(Intercept)", 0.0), "devpressure": row.get(pressure, 0.0)}
            for c in predictors:
                if c != pressure:
                    out[c] = row.get(c, 0.0)
            rows[lev] = out
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "subregion"
        pot = cls(table)
        pot.attrs = {"aic": best.aic, "deviance": best.deviance, "terms": best_terms}
        if ranking is not None:
            aics = np.array([f.aic for _, f in ranking])
            pot.attrs["dredge"] = pd.DataFrame({"terms": ["+".join(tt) for tt, _ in ranking], "aic": aics,
                                                "delta": aics - aics[0], "loglik": [f.loglik for _, f in ranking],
                                                "df": [f.npar for _, f in ranking]})
        return pot

    def surface(self, drivers, pressure, subregions=None) -> xr.DataArray:
        """The development probability (r.futures.potsurface): ``1 / (1 + exp(-(intercept + b x ...)))`` in double.

        Parameters
        ----------
        drivers
            The predictors (named as the table's columns).
        pressure
            The development pressure map.
        subregions
            The subregions map (cells of subregions without coefficients
            are NaN); default: the only row.
        """
        maps = _driver_maps(drivers)
        missing = set(self.predictors) - set(maps)
        if missing:
            raise ValueError(f"drivers lack {sorted(missing)}")
        pm = read_map(pressure)
        p = pm.as_float()
        sub = None if subregions is None else read_map(subregions).as_float()
        out = np.full(p.shape, np.nan)
        rows = [(None, self.table.iloc[0])] if sub is None else [(r, self.table.loc[r]) for r in self.table.index]
        xs = {n: maps[n].as_float() for n in self.predictors}
        for rid, row in rows:
            cells = np.ones(p.shape, dtype=bool) if rid is None else (sub == rid)
            t = row["intercept"] + row["devpressure"] * p
            for n in self.predictors:
                t = t + row[n] * xs[n]
            out[cells] = (1.0 / (1.0 + np.exp(-t)))[cells]
        tpl = pm.template
        if tpl is not None:
            return tpl.copy(data=out).rename("potential")
        return xr.DataArray(out, dims=("y", "x"), name="potential")

    def row(self, subregion) -> pd.Series:
        if None in self.table.index:
            return self.table.loc[None]
        return self.table.loc[subregion]

    def __repr__(self) -> str:
        return f"DevelopmentPotential({len(self.table)} subregions, {len(self.predictors)} predictors)"


class _GlmFit:
    """glm.fit's result as the mixed models' interface (one level)."""

    def __init__(self, fit, names, level):
        self.b = np.asarray(fit["coefficients"], dtype=np.float64)
        self.names = names
        self.level = level
        self.deviance = float(fit["deviance"])
        self.rank = int(fit["rank"])

    @property
    def npar(self) -> int:
        return self.rank

    @property
    def aic(self) -> float:
        return self.deviance + 2 * self.rank

    @property
    def loglik(self) -> float:
        return -self.deviance / 2

    def coef(self) -> dict:
        return {self.level: {n: (0.0 if np.isnan(v) else float(v)) for n, v in zip(self.names, self.b)}}


# ----------------------------------------------------------------------------- allocation

def _first_appearance(v: np.ndarray) -> list:
    """Distinct values in the order they first appear row by row (NaN left out)."""
    flat = v.ravel()
    ok = ~np.isnan(flat)
    vals, first = np.unique(flat[ok], return_index=True)
    return [int(x) for x in vals[np.argsort(first)]]


@dataclass
class PatchGrowing:
    """FUTURES' patch growing allocation (r.futures.simulation).

    Parameters
    ----------
    patch_sizes
        The library of patch sizes (cells): a list for every subregion, or
        ``{subregion: list}`` (a DataFrame with a column per subregion, NaN
        padding, as r.futures.calib writes it).
    subregions
        The subregions map (default: one subregion, id 1).
    potential_subregions
        Subregions of the potential coefficients, when they differ.
    pressure
        The starting development pressure (default: computed from the
        landscape with ``development_pressure(..., pad=True)`` and the
        parameters below).
    neighborhood, approach, gamma, scaling_factor
        The development pressure added around every new cell (the module's
        ``n_dev_neighbourhood``, ``development_pressure_approach``,
        ``gamma``, ``scaling_factor``).
    discount_factor
        Multiplies the patch sizes of the library.
    compactness_mean, compactness_range
        The distance exponent of the candidate ranking, drawn uniformly in
        mean +- range / 2 for every candidate.
    num_neighbors
        4 or 8: the cells a patch grows to.
    seed_search
        ``"probability"`` (seeds drawn on the cumulative probability) or
        ``"random"``.
    incentive_power
        Infill (> 1) or sprawl (< 1): the probability raised to this power
        (on a lookup of 1001 values, as the module).
    weight
        Map of -1 to 1 changing the probability (``potential_weight``).
    zoning, zoning_effects
        Map of zone ids and their effects: ``None`` (the module's predefined
        zones) or a DataFrame indexed by subregion with a ``stringency``
        column (0-2) and one column per zone id.
    climate
        :class:`Flooding`: floods after each step's development, damage,
        retreat (abandonment), adaptation and migration between subregions.
    """

    patch_sizes: object
    subregions: object = None
    potential_subregions: object = None
    pressure: object = None
    neighborhood: int = 10
    approach: str = "gravity"
    gamma: float = 1.5
    scaling_factor: float = 1.0
    discount_factor: float = 1.0
    compactness_mean: float = 0.4
    compactness_range: float = 0.05
    num_neighbors: int = 4
    seed_search: str = "probability"
    incentive_power: float = 1.0
    weight: object = None
    zoning: object = None
    zoning_effects: object = None
    climate: object = None

    def __post_init__(self):
        if self.approach not in _APPROACHES:
            raise ValueError("approach must be 'occurrence', 'gravity' or 'kernel'")
        if self.num_neighbors not in (4, 8):
            raise ValueError("num_neighbors must be 4 or 8")
        if self.seed_search not in ("probability", "random"):
            raise ValueError("seed_search must be 'probability' or 'random'")
        if not 0 <= self.incentive_power <= 10:
            raise ValueError("incentive_power must be between 0 and 10")

    def _patches(self, regions: list) -> tuple[list, bool]:
        ps = self.patch_sizes
        if isinstance(ps, pd.DataFrame):
            ps = {int(c): ps[c].dropna().tolist() for c in ps.columns}
        if isinstance(ps, dict):
            missing = set(regions) - {int(k) for k in ps}
            if missing:
                raise ValueError(f"no patch sizes for subregions {sorted(missing)}")
            lists = [list(ps[k] if k in ps else ps[str(k)]) for k in regions]
            single = False
        else:
            lists = [list(ps)]
            single = True
        out = []
        for lst in lists:
            sizes = [int(int(x) * float(self.discount_factor)) for x in lst]
            sizes = [s for s in sizes if s > 0]
            out.append(sizes or [1])
        return out, single

    def simulate(self, model, landscape, drivers=None, *, seed=None, nodata=None):
        from ._model import Simulation, _as_maps

        land = read_map(landscape, nodata)
        shape = land.shape
        dev = land.as_float()
        rules = model.rules
        null = np.isnan(dev)
        if rules.restrictions is not None:
            r = read_map(rules.restrictions).as_float()
            null |= np.nan_to_num(r, nan=0.0) != 0
        sub = np.ones(shape) if self.subregions is None else read_map(self.subregions).as_float()
        regions = _first_appearance(sub)
        reg_index = np.full(shape, -1, dtype=np.int32)
        for k, rid in enumerate(regions):
            reg_index[sub == rid] = k
        null |= np.isnan(sub)
        pot = model.potential
        if not isinstance(pot, DevelopmentPotential):
            raise TypeError("PatchGrowing needs a DevelopmentPotential")
        if self.potential_subregions is not None:
            psub = read_map(self.potential_subregions).as_float()
            pregions = _first_appearance(psub)
            pot_index = np.full(shape, -1, dtype=np.int32)
            for k, rid in enumerate(pregions):
                pot_index[psub == rid] = k
            null |= np.isnan(psub)
        else:
            pregions, pot_index = regions, None
        if self.pressure is not None:
            pressure = read_map(self.pressure).as_float().astype(np.float32)
        else:
            pressure = development_pressure(np.where(np.isnan(dev), np.nan, dev), self.neighborhood,
                                            method=self.approach, gamma=self.gamma,
                                            scaling_factor=self.scaling_factor).values.astype(np.float32)
        null |= np.isnan(pressure)
        weight = None
        if self.weight is not None:
            w = read_map(self.weight).as_float()
            null |= np.isnan(w)
            weight = np.clip(np.nan_to_num(w, nan=0.0), -1, 1).astype(np.float32)
        zone = None
        if self.zoning is not None:
            z = read_map(self.zoning).as_float()
            zone = np.nan_to_num(z, nan=0.0).astype(np.int32)
        # coefficients per potential subregion
        rows = [pot.row(r) for r in pregions]
        intercept = np.array([r["intercept"] for r in rows], dtype=np.float64)
        dpcoef = np.array([r["devpressure"] for r in rows], dtype=np.float64)
        maps = _driver_maps(drivers)
        names = pot.predictors
        missing = set(names) - set(maps)
        if missing:
            raise ValueError(f"drivers lack {sorted(missing)}")
        xs = []
        for n in names:
            x = maps[n].as_float()
            if x.shape != shape:
                raise ValueError(f"driver {n!r} differs in size from the landscape")
            xs.append(x.astype(np.float32))
        demand = model.demand
        if not isinstance(demand, DevelopmentDemand):
            raise TypeError("PatchGrowing needs a DevelopmentDemand")
        climate = None
        if self.climate is not None:
            climate, null = self.climate._inputs(shape, null, regions, reg_index, pregions, pot_index, demand)
        for x in xs:
            null |= np.isnan(x)
        pidx = reg_index if pot_index is None else pot_index
        agg = np.zeros(shape, dtype=np.float32)
        ok = ~null
        for n, x in zip(names, xs):
            coef = np.array([r[n] for r in rows], dtype=np.float64)
            term = (coef[np.clip(pidx, 0, None)] * x.astype(np.float64)).astype(np.float32)
            agg[ok] = agg[ok] + term[ok]
        developed = np.where(null, np.iinfo(np.int32).min, np.nan_to_num(dev) - 1).astype(np.int32)
        absent = set(regions) - set(demand.subregions)
        if absent:
            raise ValueError(f"no demand for subregions {sorted(absent)}")
        table = np.stack([demand.table[r].to_numpy(dtype=np.float32) for r in regions])
        patches, single = self._patches(regions)
        incentive = np.zeros(0, dtype=np.float32)
        if self.incentive_power != 1:
            step = 1.0 / 1000
            incentive = np.array([math.pow(i * step, float(self.incentive_power)) for i in range(1001)],
                                 dtype=np.float32)
        zids, zregs, zeff, stringency, user = self._zoning(regions)
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        r = _core.sim_futures_run(
            np.ascontiguousarray(developed), np.ascontiguousarray(reg_index),
            None if pot_index is None else np.ascontiguousarray(pot_index), np.ascontiguousarray(pressure),
            np.ascontiguousarray(agg), weight, zone, len(regions), len(pregions), intercept, dpcoef,
            int(self.neighborhood), _APPROACHES[self.approach], np.float32(self.gamma),
            np.float32(self.scaling_factor), patches, single, np.float32(self.compactness_mean),
            np.float32(self.compactness_range), int(self.num_neighbors), self.seed_search == "probability",
            incentive, user, zids, zregs, zeff, stringency, np.ascontiguousarray(table), demand.steps, int(seed),
            climate)
        series = r["series"].astype(np.float64)
        series[series == np.iinfo(np.int32).min] = np.nan
        start = np.where(null, np.nan, developed.astype(np.float64))
        coded = np.concatenate([start[None], series])
        binary = np.where(np.isnan(coded), np.nan, (coded >= 0).astype(np.float64))  # abandoned: 0
        conv = np.asarray(r["converted"]).reshape(demand.steps, len(regions))
        over = np.asarray(r["overflow"]).reshape(demand.steps, len(regions))
        warn = np.asarray(r["warnings"]).reshape(demand.steps, len(regions))
        rows_out = []
        for k in range(demand.steps):
            row = {"step": k + 1, "year": demand.table.index[k]}
            for j, rid in enumerate(regions):
                row[f"developed_{rid}"] = int(conv[k, j])
                row[f"overflow_{rid}"] = int(over[k, j])
            rows_out.append(row)
        for k, j in zip(*np.nonzero(warn)):
            msgs = [m for bit, m in ((1, "not enough undeveloped cells, all developed"),
                                     (2, "too many failed attempts to find a seed")) if warn[k, j] & bit]
            warnings.warn(f"step {k + 1}, subregion {regions[j]}: {'; '.join(msgs)}", stacklevel=4)
        sim = Simulation(_as_maps(binary, land.template, "developed"), pd.DataFrame(rows_out).set_index("step"),
                         int(seed))
        sim.development = _as_maps(coded, land.template, "development")
        if climate is not None:
            start = climate.get("adaptation", np.zeros(int(np.prod(shape)), dtype=np.int32)).reshape(shape)
            ad = np.concatenate([start[None], r["adaptation"]]).astype(np.float64)
            ad[ad == np.iinfo(np.int32).min] = np.nan
            ad[:, null] = np.nan
            sim.adaptation = _as_maps(ad, land.template, "adaptation")
            sim.floods = pd.DataFrame({"step": np.asarray(r["flood_steps"]) + 1, "catchment": r["flood_hucs"],
                                       "probability": r["flood_probabilities"]})
            if "from_ids" in climate:
                mv = np.asarray(r["moved"]).reshape(demand.steps, climate["from_ids"].size, climate["to_ids"].size)
                sim.redistribution = xr.DataArray(mv, dims=("step", "from", "to"),
                                                  coords={"step": np.arange(1, demand.steps + 1),
                                                          "from": climate["from_ids"], "to": climate["to_ids"]},
                                                  name="cells_moved")
            sim.demand_table = pd.DataFrame(np.asarray(r["demand"]).reshape(len(regions), demand.steps).T,
                                            index=demand.table.index, columns=regions)
        return sim

    def _zoning(self, regions: list):
        empty_i, empty_f = np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
        if self.zoning is None:
            return empty_i, empty_i, empty_f, empty_f, False
        eff = self.zoning_effects
        if eff is None:
            ids = list(ZONING_EFFECTS)
            return (np.array(ids, dtype=np.int32), np.full(len(ids), -1, dtype=np.int32),
                    np.array([ZONING_EFFECTS[i] for i in ids], dtype=np.float32), empty_f, False)
        t = pd.DataFrame(eff).copy()
        t.index = [int(i) for i in t.index]
        if "stringency" not in t.columns:
            raise ValueError("the zoning effects need a stringency column")
        missing = set(regions) - set(t.index)
        if missing:
            raise ValueError(f"no zoning effects for subregions {sorted(missing)}")
        st = t["stringency"].astype(float)
        if ((st <= 0) | (st >= 2)).any():
            raise ValueError("zoning stringency must be between 0 and 2 (both excluded)")
        stringency = np.array([st[r] for r in regions], dtype=np.float32)
        user = not bool((stringency == 1).all())
        zone_cols = [c for c in t.columns if c != "stringency"]
        if not zone_cols:
            ids = list(ZONING_EFFECTS)
            return (np.array(ids, dtype=np.int32), np.full(len(ids), -1, dtype=np.int32),
                    np.array([ZONING_EFFECTS[i] for i in ids], dtype=np.float32), stringency, user)
        vals = t[zone_cols].astype(float)
        if ((vals < -1) | (vals > 1)).to_numpy().any():
            raise ValueError("zoning effects must be between -1 and 1")
        ids, regs, effs = [], [], []
        for k, rid in enumerate(regions):
            for c in zone_cols:
                ids.append(int(c))
                regs.append(k)
                effs.append(float(vals.loc[rid, c]))
        return (np.array(ids, dtype=np.int32), np.array(regs, dtype=np.int32), np.array(effs, dtype=np.float32),
                stringency, user)

    def __repr__(self) -> str:
        return f"PatchGrowing(compactness={self.compactness_mean}+-{self.compactness_range / 2}, seed_search=" \
               f"{self.seed_search!r})"


# ----------------------------------------------------------------------------- climate (flooding)

@dataclass
class Flooding:
    """FUTURES' flood submodel (Sanchez et al. 2023): floods, damage, response, abandonment, migration.

    After the development of each step, a flood may strike each catchment:
    with probability ``max`` (the largest flood probability of the
    catchment's cells, or of the depth maps), at a probability ``p`` drawn
    uniformly. Its depth at each cell comes from the depth maps of the
    probabilities (the map of the largest probability below ``p``) or, from
    flood probability maps, as the HAND percentile of the cells flooded at
    ``p`` minus the cell's HAND. The damage is the depth-damage curve's (%)
    unless the cell was adapted to such floods; a damaged developed cell
    then retreats (abandoned, its demand moved to a subregion drawn from the
    redistribution matrix, scaled by the population densities), adapts
    (protected up to the return period of the flood) or stays, from its
    adaptive capacity and the response lines, perturbed by a Gaussian.

    Parameters
    ----------
    catchments
        Map of catchments (HUCs): floods are drawn per catchment.
    adaptive_capacity
        Map from -1 (most vulnerable) to 1 (most resilient).
    damage
        Depth-damage curves: a DataFrame with the depth levels as columns
        and the damage (%) per damage region in rows (one row: the same
        curve everywhere).
    response
        ``(vulnerability_a, vulnerability_b, resilience_a, resilience_b)``:
        the lines ``damage = a * capacity + b`` above which a cell retreats
        (vulnerable cells otherwise stay, resilient ones adapt).
    response_stddev
        Standard deviation of the Gaussian added to capacity and damage.
    depths
        ``{step: {probability: map}}``: flood depths for floods of each
        annual probability, from that step on (steps count from 1).
    probabilities, hand, hand_percentile
        Instead of depths: ``{step: map}`` of flood probability, the HAND
        (height above nearest drainage) map and the percentile giving the
        flood level.
    damage_regions
        Which curve each cell uses: ``None`` (the only row), ``"subregions"``,
        ``"potential"`` (potential subregions) or a map of regions.
    adaptation
        Map of the return period each cell is already protected against.
    redistribution
        DataFrame with a row per subregion people leave and a column per
        destination: the probabilities (%). Without it, retreating cells
        are abandoned and their demand leaves the area.
    """

    catchments: object
    adaptive_capacity: object
    damage: object
    response: tuple
    response_stddev: float = 0.0
    depths: dict | None = None
    probabilities: dict | None = None
    hand: object = None
    hand_percentile: float | None = None
    damage_regions: object = None
    adaptation: object = None
    redistribution: object = None

    def __post_init__(self):
        if (self.depths is None) == (self.probabilities is None):
            raise ValueError("give either depths or probabilities")
        if self.probabilities is not None and (self.hand is None or self.hand_percentile is None):
            raise ValueError("flood probabilities need the HAND map and its percentile")
        if len(self.response) != 4:
            raise ValueError("response is (vulnerability_a, vulnerability_b, resilience_a, resilience_b)")

    def _inputs(self, shape, null_before, regions, reg_index, pregions, pot_index, demand) -> tuple[dict, np.ndarray]:
        """The climate dict for the core and the cells nulled by the climate inputs."""
        ncell = int(np.prod(shape))
        huc = read_map(self.catchments).as_float()
        ac = read_map(self.adaptive_capacity).as_float()
        null = null_before.copy()
        extra_null = np.isnan(ac)
        if self.probabilities is not None:
            hand = read_map(self.hand).as_float()
            extra_null |= np.isnan(hand)
        # catchment indices: first appearance among the cells still simulated before these checks
        cand = ~null & ~np.isnan(huc)
        ids = _first_appearance(np.where(cand, huc, np.nan))
        huc_index = np.full(shape, -1, dtype=np.int32)
        for k, rid in enumerate(ids):
            huc_index[cand & (huc == rid)] = k
        # damage regions
        dmg = pd.DataFrame(self.damage).copy()
        levels = np.array([float(c) for c in dmg.columns], dtype=np.float64)
        src = self.damage_regions
        if src is None:
            ddf = np.zeros(shape, dtype=np.int32)
            curves = [dmg.iloc[0].to_numpy(dtype=np.float64)]
        else:
            if isinstance(src, str) and src == "subregions":
                order, idx = regions, reg_index
            elif isinstance(src, str) and src == "potential":
                order, idx = pregions, (pot_index if pot_index is not None else reg_index)
            else:
                m = read_map(src).as_float()
                extra_null |= cand & np.isnan(m)
                order = _first_appearance(np.where(cand, m, np.nan))
                idx = np.full(shape, -1, dtype=np.int32)
                for k, rid in enumerate(order):
                    idx[cand & (m == rid)] = k
            dmg.index = [int(i) for i in dmg.index]
            missing = set(order) - set(dmg.index)
            if missing:
                raise ValueError(f"no depth-damage curve for regions {sorted(missing)}")
            curves = [dmg.loc[r].to_numpy(dtype=np.float64) for r in order]
            ddf = np.asarray(idx, dtype=np.int32)
        null |= cand & extra_null
        c = {"depth_mode": self.depths is not None, "huc": huc_index.ravel(), "nhuc": len(ids),
             "huc_ids": np.array(ids, dtype=np.int32), "adaptive_capacity": ac.astype(np.float32).ravel(),
             "levels": levels, "damage": np.ascontiguousarray(np.stack(curves)), "ddf_region": ddf.ravel(),
             "response": np.asarray(self.response, dtype=np.float32), "stddev": float(self.response_stddev),
             "region_ids": np.array(regions, dtype=np.int32)}
        if self.depths is not None:
            steps = sorted(int(k) for k in self.depths)
            probs = sorted({float(p) for v in self.depths.values() for p in v})
            c["probabilities"] = np.array(probs, dtype=np.float32)
            depth_maps = []
            for st in steps:
                given = {float(p): m for p, m in self.depths[st].items()}
                missing = set(probs) - set(given)
                if missing:
                    raise ValueError(f"step {st} lacks depth maps for probabilities {sorted(missing)}")
                depth_maps.append(np.ascontiguousarray(np.stack(
                    [read_map(given[p]).as_float().astype(np.float32).ravel() for p in probs])))
            c["depths"] = depth_maps
        else:
            steps = sorted(int(k) for k in self.probabilities)
            c["flood_probability"] = np.ascontiguousarray(np.stack(
                [read_map(self.probabilities[st]).as_float().astype(np.float32).ravel() for st in steps]))
            c["hand"] = read_map(self.hand).as_float().astype(np.float32).ravel()
            c["percentile"] = float(self.hand_percentile)
        c["flood_steps"] = np.array([st - 1 for st in steps], dtype=np.int32)
        if self.adaptation is not None:
            a = read_map(self.adaptation).as_float()
            c["adaptation"] = np.where(np.isnan(a), np.iinfo(np.int32).min, a).astype(np.int32).ravel()
        if self.redistribution is not None:
            m = pd.DataFrame(self.redistribution).copy()
            if demand.population is None:
                raise ValueError("redistribution needs the population demand (DevelopmentDemand(population=...))")
            c["from_ids"] = np.array([int(i) for i in m.index], dtype=np.int32)
            c["to_ids"] = np.array([int(i) for i in m.columns], dtype=np.int32)
            c["matrix"] = np.ascontiguousarray((m.to_numpy(dtype=np.float64) / 100.).astype(np.float32))
            pop = demand.population
            pop.columns = [int(x) for x in pop.columns]
            c["population"] = np.stack([pop[r].to_numpy(dtype=np.float32) for r in regions]).ravel()
        assert ncell == c["huc"].size
        return c, null


# ----------------------------------------------------------------------------- calibration (r.futures.calib)

def _patch_geometry(mask: np.ndarray, ew: float, ns: float) -> np.ndarray:
    """Area and perimeter of each 4-connected patch (r.clump + r.object.geometry -m), as (n, 2)."""
    from scipy import ndimage

    labels, n = ndimage.label(mask, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    if n == 0:
        return np.empty((0, 2))
    area = np.bincount(labels.ravel(), minlength=n + 1)[1:] * (ew * ns)
    pad = np.pad(labels, 1)
    inner = pad[1:-1, 1:-1]
    perim = np.zeros(n + 1)
    for shifted, length in ((pad[:-2, 1:-1], ew), (pad[2:, 1:-1], ew), (pad[1:-1, :-2], ns), (pad[1:-1, 2:], ns)):
        edge = (inner > 0) & (shifted != inner)
        perim += np.bincount(inner[edge], minlength=n + 1) * length
    return np.column_stack([area, perim[1:]])


def _resolution(m) -> tuple[float, float]:
    t = m.template
    if t is not None:
        try:
            rx, ry = t.rio.resolution()
            return abs(float(rx)), abs(float(ry))
        except Exception:
            pass
    return 1.0, 1.0


def _new_development(start, end, subregions):
    s = start
    sub_ok = ~np.isnan(subregions) & (np.nan_to_num(subregions) != 0)
    end_ok = ~np.isnan(end) & (np.nan_to_num(end) != 0)
    start_ok = np.isnan(s) | (np.nan_to_num(s) == 0)
    return sub_ok & end_ok & start_ok


def patch_library(start, end, *, subregions=None, per_subregion: bool = False, threshold: float = 0.0,
                  nodata=None):
    """The sizes of the patches developed between two maps (r.futures.calib's patch library).

    Patches are the 4-connected groups of cells developed in ``end`` and not
    in ``start`` (inside a subregion); their sizes, in cells, are what
    :class:`PatchGrowing` draws from.

    Parameters
    ----------
    start, end
        Developed (1) maps at two dates.
    subregions
        Map of subregions (cells outside, or in subregion 0, are ignored).
    per_subregion
        A library per subregion (``{subregion: sizes}``), each patch cut to
        its part in the subregion.
    threshold
        Smallest patch area kept (in the map's area units).

    Returns
    -------
    list or dict
        The sizes, sorted.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> a = np.zeros((5, 6))
    >>> b = a.copy()
    >>> b[0, :3] = 1
    >>> b[3:5, 4:6] = 1
    >>> ls.simulate.patch_library(a, b)
    [3, 4]
    """
    m0, m1 = read_map(start, nodata), read_map(end, nodata)
    ew, ns = _resolution(m1)
    sub = np.ones(m1.shape) if subregions is None else read_map(subregions).as_float()
    diff = _new_development(m0.as_float(), m1.as_float(), sub)
    cell = ((ns + ew) / 2.0) ** 2
    thr = float(threshold) - 1e-6

    def sizes(mask):
        g = _patch_geometry(mask, ew, ns)
        g = g[g[:, 0] > thr]
        return sorted(int(v) for v in np.round(g[:, 0] / cell).astype(int))
    if not per_subregion:
        return sizes(diff)
    from scipy import ndimage

    labels, _ = ndimage.label(diff, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    cats = sorted(int(c) for c in np.unique(sub[~np.isnan(sub)]))
    out = {}
    for c in cats:
        part = np.where(sub == c, labels, 0)
        if not (part > 0).any():
            out[c] = []
            continue
        # each patch of the whole map, cut to the subregion: one object per patch id (r.object.geometry)
        ids, counts = np.unique(part[part > 0], return_counts=True)
        g = np.column_stack([counts * (ew * ns), np.zeros(ids.size)])
        g = g[g[:, 0] > thr]
        out[c] = sorted(int(v) for v in np.round(g[:, 0] / cell).astype(int))
    if len(out) <= 1:
        return sizes(diff)
    return out


def _histograms(data, cell, bins_a, range_a, bins_c, range_c):
    area, perimeter = data.T
    compact = perimeter / (2 * np.sqrt(np.pi * area))
    ha, _ = np.histogram(area / cell, bins=bins_a, range=range_a, density=True)
    hc, _ = np.histogram(compact, bins=bins_c, range=range_c, density=True)
    return ha * 100, hc * 100


def _compare_histograms(h1, h2):
    mask = np.logical_not(np.logical_or(h1, h2))
    a = np.ma.masked_array(h1, mask=mask)
    b = np.ma.masked_array(h2, mask=mask)
    return 0.5 * np.sum(np.power(a - b, 2) / (a.astype(float) + b))


def calibrate_patches(model, start, end, drivers=None, *, compactness_means=(0.4,), compactness_ranges=(0.05,),
                      discount_factors=(1.0,), repeat: int = 10, seed: int = 1, threshold: float = 0.0,
                      per_subregion: bool = False) -> pd.DataFrame:
    """Calibrate FUTURES' patch compactness and discount factor (r.futures.calib).

    The patches observed between ``start`` and ``end`` give the patch
    library and the histograms of patch area and compactness
    (``perimeter / (2 sqrt(pi area))``); each combination of the parameters
    is simulated ``repeat`` times from ``start`` (the model's demand and
    potential) and its patches compared with the observed ones by the
    histograms' chi-square distance.

    Parameters
    ----------
    model
        A :class:`Model` with a :class:`PatchGrowing` allocation (its patch
        library is replaced by the observed one).
    start, end
        The observed maps of development.
    drivers
        The predictors of the potential.
    compactness_means, compactness_ranges, discount_factors
        The values to try (every combination).
    repeat
        Simulations per combination (seeds ``(seed + k) * 10000 + i``).
    threshold
        Smallest patch area kept.
    per_subregion
        A patch library per subregion.

    Returns
    -------
    pandas.DataFrame
        Per combination: discount factor, compactness mean and range, the
        area and compactness errors (normalized by their largest) and their
        mean, from the best; ``attrs["library"]`` holds the patch library.
    """
    import copy

    m0, m1 = read_map(start), read_map(end)
    ew, ns = _resolution(m1)
    alloc = model.allocation
    sub = np.ones(m1.shape) if alloc.subregions is None else read_map(alloc.subregions).as_float()
    cell = ((ns + ew) / 2.0) ** 2
    thr = float(threshold) - 1e-6
    data = _patch_geometry(_new_development(m0.as_float(), m1.as_float(), sub), ew, ns)
    data = data[data[:, 0] > thr]
    library = patch_library(start, end, subregions=alloc.subregions, per_subregion=per_subregion,
                            threshold=threshold)
    area = data[:, 0] / cell
    compact = data[:, 1] / (2 * np.sqrt(np.pi * data[:, 0]))
    bins_a, range_a = int(np.ptp(area) / 1.0), (np.min(area), np.max(area))
    bins_c, range_c = int(np.ptp(compact) / 0.1), (np.min(compact), np.max(compact))
    h_area, _ = np.histogram(area, bins=bins_a, range=range_a, density=True)
    h_comp, _ = np.histogram(compact, bins=bins_c, range=range_c, density=True)
    h_area, h_comp = h_area * 100, h_comp * 100
    rows = []
    s = int(seed)
    for mean in compactness_means:
        for rng in compactness_ranges:
            for disc in discount_factors:
                trial = copy.copy(model)
                trial.allocation = copy.copy(alloc)
                trial.allocation.patch_sizes = library
                trial.allocation.compactness_mean = float(mean)
                trial.allocation.compactness_range = float(rng)
                trial.allocation.discount_factor = float(disc)
                d_area = d_comp = 0.0
                for i in range(int(repeat)):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sim = trial.run(start, drivers, seed=s * 10000 + i)
                    final = sim.development.values[-1]
                    g = _patch_geometry(np.nan_to_num(final, nan=-1) > 0, ew, ns)
                    g = g[g[:, 0] > thr]
                    ha, hc = _histograms(g, cell, bins_a, range_a, bins_c, range_c)
                    d_area += _compare_histograms(h_area, ha)
                    d_comp += _compare_histograms(h_comp, hc)
                rows.append((float(disc), d_area / repeat, float(mean), float(rng), d_comp / repeat))
                s += 1
    res = np.array(rows, dtype=np.float64)
    disc, area_err, cm, cr, comp_err = res.T
    na, nc = area_err / np.max(area_err), comp_err / np.max(comp_err)
    out = np.column_stack((disc, cm, cr, na, nc, (na + nc) / 2))
    out = out[out[:, 5].argsort()]
    table = pd.DataFrame(out, columns=["discount_factor", "compactness_mean", "compactness_range", "area_error",
                                       "compactness_error", "combined_error"])
    table.attrs["library"] = library
    return table


def _driver_maps(drivers) -> dict:
    if drivers is None:
        return {}
    from ._woe import _drivers

    return _drivers(drivers)
