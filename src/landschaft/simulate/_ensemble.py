# SPDX-License-Identifier: GPL-3.0-or-later
"""Scenarios and replicates of a land change model, run in parallel."""

from __future__ import annotations

import copy
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .._threads import resolve
from ._change import _class_value
from ._maps import read_map

__all__ = ["Ensemble", "scenarios"]


@dataclass
class Ensemble:
    """The simulations of :func:`scenarios`: every scenario, replicate and year.

    Attributes
    ----------
    scenarios : list of str
    seeds : list of int
        The seed of each replicate (the same in every scenario).
    years : list
        The year (or step) of each kept map, the start first.
    classes : list
        The classes of the maps.
    steps : pandas.DataFrame
        The step tables of every run, with ``scenario`` and ``replicate``.
    """

    scenarios: list
    seeds: list
    years: list
    classes: list
    codes: np.ndarray = field(repr=False)  # (scenario, replicate, year, y, x) class index, -1 missing
    steps: pd.DataFrame = field(repr=False)
    template: object = field(default=None, repr=False)

    def _da(self, values, dims, coords, name):
        t = self.template
        if t is None:
            return xr.DataArray(values, dims=(*dims, "y", "x"), coords=coords, name=name)
        da = xr.DataArray(values, dims=(*dims, *t.dims), coords={**coords, **{k: t.coords[k] for k in t.dims}},
                          name=name)
        try:
            from ..io import _with_georef

            da = _with_georef(da, transform=t.rio.transform())
            if t.rio.crs is not None:
                da = _with_georef(da, crs=t.rio.crs)
        except Exception:  # noqa: BLE001, S110  (no georeference)
            pass
        return da

    def _scenario(self, scenario) -> int:
        if scenario is None:
            if len(self.scenarios) > 1:
                raise ValueError(f"name the scenario: {self.scenarios}")
            return 0
        return self.scenarios.index(scenario)

    def maps(self, scenario=None, replicate: int = 0) -> xr.DataArray:
        """The maps of one run, ``(year, y, x)``."""
        c = self.codes[self._scenario(scenario), replicate]
        lut = np.append(np.asarray(self.classes, dtype=np.float64), np.nan)
        return self._da(lut[c], ("year",), {"year": self.years}, "landuse")

    def probability(self, cls, *, year=None, scenario=None) -> xr.DataArray:
        """Share of the replicates where each cell has the class: ``(scenario, year, y, x)``.

        ``year`` and ``scenario`` select one (the dimension is dropped).
        """
        k = self.classes.index(_class_value(cls))
        valid = self.codes >= 0
        p = (self.codes == k).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
        p = np.where(valid.any(axis=1), p, np.nan)
        da = self._da(p, ("scenario", "year"), {"scenario": self.scenarios, "year": self.years},
                      f"probability_{k}")
        if scenario is not None:
            da = da.sel(scenario=scenario)
        if year is not None:
            da = da.sel(year=year)
        return da

    def change_probability(self, *, scenario=None) -> xr.DataArray:
        """Share of the replicates where each cell's class differs from the start: ``(scenario, year, y, x)``."""
        start = self.codes[:, :, :1]
        changed = (self.codes != start) & (self.codes >= 0)
        p = np.where((self.codes >= 0).any(axis=1), changed.mean(axis=1), np.nan)
        da = self._da(p, ("scenario", "year"), {"scenario": self.scenarios, "year": self.years},
                      "change_probability")
        return da if scenario is None else da.sel(scenario=scenario)

    def cover(self) -> pd.DataFrame:
        """Cells of each class: a row per scenario, replicate and year."""
        rows = []
        for s, name in enumerate(self.scenarios):
            for r in range(self.codes.shape[1]):
                for y, year in enumerate(self.years):
                    n = np.bincount(self.codes[s, r, y][self.codes[s, r, y] >= 0].ravel(),
                                    minlength=len(self.classes))
                    rows.append({"scenario": name, "replicate": r, "year": year,
                                 **{c: int(v) for c, v in zip(self.classes, n)}})
        return pd.DataFrame(rows)

    def metrics(self, metrics, *, years=None, level: str = "class", n_threads: int | None = None) -> pd.DataFrame:
        """Landscape metrics (or any function of a map) of every run and year, in long format.

        Parameters
        ----------
        metrics
            Names of ``ls.landscape_metrics`` metrics (``["pland", "np",
            "ai"]``, at ``level``), or a ``dict`` ``{name: function}`` of
            functions of a map (a DataArray) giving a number, a ``dict`` or a
            Series (e.g. an MSPA share or a :class:`~landschaft.HabitatGraph`
            index).
        years
            The years to measure (default: all kept).
        level
            The level of the named metrics (``"class"``, ``"landscape"``).

        Returns
        -------
        pandas.DataFrame
            ``scenario``, ``replicate``, ``year``, ``metric``, ``class``
            (NaN at landscape level and for functions), ``value``.
        """
        from ..lsm import landscape_metrics

        years = self.years if years is None else list(years)
        rows = []
        for s, name in enumerate(self.scenarios):
            for r in range(self.codes.shape[1]):
                maps = self.maps(name, r)
                for year in years:
                    m = maps.sel(year=year)
                    if isinstance(metrics, dict):
                        for key, fn in metrics.items():
                            v = fn(m)
                            items = v.items() if isinstance(v, (dict, pd.Series)) else [(None, v)]
                            for sub, x in items:
                                rows.append({"scenario": name, "replicate": r, "year": year,
                                             "metric": key if sub is None else f"{key}_{sub}", "class": np.nan,
                                             "value": float(x)})
                    else:
                        ints = m.fillna(-9999).astype(np.int32)
                        t = landscape_metrics(ints, level=level, metric=list(metrics), nodata=-9999,
                                              n_threads=n_threads)
                        for _, row in t.iterrows():
                            rows.append({"scenario": name, "replicate": r, "year": year, "metric": row["metric"],
                                         "class": row["class"], "value": row["value"]})
        return pd.DataFrame(rows)

    def to_dataset(self) -> xr.Dataset:
        """Every map as one Dataset: ``landuse (scenario, replicate, year, y, x)``."""
        lut = np.append(np.asarray(self.classes, dtype=np.float32), np.nan)
        da = self._da(lut[self.codes], ("scenario", "replicate", "year"),
                      {"scenario": self.scenarios, "replicate": np.arange(self.codes.shape[1]), "year": self.years},
                      "landuse")
        return xr.Dataset({"landuse": da}, attrs={"seeds": list(self.seeds)})

    def __repr__(self) -> str:
        return (f"Ensemble({len(self.scenarios)} scenario(s) x {len(self.seeds)} replicate(s), "
                f"years {self.years[0]}-{self.years[-1]})")


def _apply(model, change):
    """A scenario's model: the base model with some pieces replaced."""
    from ._model import Model

    if change is None:
        return model
    if isinstance(change, Model):
        return change
    if callable(change):
        return change(copy.copy(model))
    if isinstance(change, dict):
        m = copy.copy(model)
        for k, v in change.items():
            if k not in ("potential", "demand", "allocation", "rules", "dynamic"):
                raise ValueError(f"a scenario changes potential, demand, allocation, rules or dynamic, not {k!r}")
            setattr(m, k, dict(v) if k == "dynamic" and v else v)
        return m
    raise TypeError("a scenario is a Model, a dict of pieces or a function of the model")


def _saved(model, folder, name: str, start, drivers, nodata) -> list:
    """The model written as a Terraflux graph in ``folder``: the files of the task that runs it."""
    from pathlib import Path

    from .._dist import InputFile

    path = Path(folder) / f"{name}.json"
    try:
        model.save(path, start, drivers, seed=0, nodata=nodata)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"the model cannot be sent to helpers ({type(e).__name__}: {e}); run it without on=") from e
    data = path.with_name(path.stem + "_data")
    files = [InputFile.of(path)]
    if data.is_dir():
        files += [InputFile.of(p, p.relative_to(folder).as_posix()) for p in sorted(data.rglob("*")) if p.is_file()]
    return files


def _runs_on(on, models, names, runs, seeds, start, drivers, nodata) -> list:
    """The runs as tasks of ``on`` (ls.helpers()): their maps and step tables, in order."""
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from .._dist import Job, Task, run_job

    with tempfile.TemporaryDirectory(prefix="landschaft-scenarios-") as tmp:
        files = {s: _saved(models[name], tmp, f"scenario_{s}", start, drivers, nodata) for s, name in enumerate(names)}
        tasks = [Task(f"{names[s]} #{r}", "simulate.run", {"model": f"scenario_{s}.json", "seed": seeds[r]},
                      files[s], root=Path(tmp)) for s, r in runs]
        out = {}

        def collect(task, folder, stats):
            maps = np.load(folder / "maps.npy", allow_pickle=False)
            out[task.name] = SimpleNamespace(maps=SimpleNamespace(values=maps, shape=maps.shape),
                                             steps=pd.read_parquet(folder / "steps.parquet"))

        report = run_job(on, Job("scenarios", tasks, collect))
    if report.failed:
        name, error = next(iter(report.failed.items()))
        raise RuntimeError(f"{len(report.failed)} run(s) failed; {name}: {error}")
    return [out[t.name] for t in tasks]


def scenarios(model, start=None, drivers=None, *, scenarios=None, replicates: int = 10, seed=None, steps=None,
              years=None, keep=None, n_workers=None, n_threads=None, nodata=None, on=None) -> Ensemble:
    """Run a model under several scenarios, several times each, in parallel.

    Parameters
    ----------
    model
        A :class:`Model`, or a :class:`Projection` (its fitted model, from
        its later map with its drivers, over its years).
    start
        The starting map (default: the projection's later map).
    drivers
        The drivers (default: the projection's).
    scenarios
        ``{name: change}``: a :class:`Model`, a ``dict`` of the pieces to
        replace (``{"demand": ..., "rules": ...}``) or a function of the
        model; default one scenario, ``"baseline"``, the model as given.
    replicates
        Runs per scenario; replicate ``r`` uses the seed ``seed + r`` in
        every scenario (common random numbers: scenarios differ by their
        settings, not by their draws).
    steps
        Steps to simulate, set on the allocation (default: the model's own).
    years
        Year of the start (one step a year), or the years of all the maps.
    keep
        The years to keep (default all): with large maps, e.g. ``[2030,
        2050]``.
    n_workers
        Runs at the same time (default: as many as the threads allow).
    n_threads
        Threads in all (default: all cores but one), shared by the workers.
    on
        ``ls.helpers()``: run the simulations on this computer and on the helpers of
        its group (see :mod:`landschaft.parallel`), with the same maps. Each scenario's
        model travels as its Terraflux graph (:meth:`Model.save`), so it must be made
        of the pieces that :meth:`Model.save` writes.

    Returns
    -------
    Ensemble

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> S = ls.simulate
    >>> land = np.ones((30, 30)); land[10:20, 10:20] = 2
    >>> pot = S.RasterPotential({1: np.full((30, 30), 0.2), 2: np.random.default_rng(0).random((30, 30))})
    >>> demand = S.AreaDemand(np.array([[800, 100], [750, 150], [700, 200]]), classes=[1, 2])
    >>> fast = S.AreaDemand(np.array([[800, 100], [700, 200], [600, 300]]), classes=[1, 2])
    >>> ens = S.scenarios(S.Model(pot, demand, S.CARS(step_size=10)), land, replicates=4, seed=1,
    ...                   scenarios={"trend": {}, "fast": {"demand": fast}}, years=2020)
    >>> ens.probability(2, year=2022).sel(scenario="fast").values.sum().round()
    np.float64(300.0)
    """
    from ._model import Model
    from ._run import Projection, _with_steps

    if isinstance(model, Projection):
        proj = model
        model = proj.model
        if start is None:
            start = proj.maps.isel(year=0)
        if drivers is None:
            drivers = proj._call["args"][2]
        if steps is None:
            steps = proj.maps.shape[0] - 1
        if years is None:
            years = [*proj.maps["year"].values.tolist()]
    if not isinstance(model, Model):
        raise TypeError("scenarios() takes a Model or a Projection")
    if start is None:
        raise ValueError("give the starting map")
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    plan = {"baseline": None} if not scenarios else dict(scenarios)
    models = {k: _apply(model, v) for k, v in plan.items()}
    if steps is not None:
        models = {k: _with_steps(m, int(steps)) for k, m in models.items()}
    if seed is None:
        seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    seeds = [int(seed) + r for r in range(int(replicates))]
    total = resolve(n_threads)
    runs = [(s, r) for s in range(len(models)) for r in range(len(seeds))]
    workers = min(len(runs), total) if n_workers is None else max(1, int(n_workers))
    inner = max(1, total // workers)
    names = list(models)

    def one(job):
        s, r = job
        return models[names[s]].run(start, drivers, seed=seeds[r], nodata=nodata, n_threads=inner)

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        if on is not None:
            results = _runs_on(on, models, names, runs, seeds, start, drivers, nodata)
        elif workers == 1:
            results = [one(j) for j in runs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(one, runs))
    n_maps = {res.maps.shape[0] for res in results}
    if len(n_maps) > 1:
        raise ValueError(f"the scenarios simulate different numbers of steps: {sorted(n_maps)}")
    n = n_maps.pop()
    if years is None:
        all_years = list(range(n))
    elif np.isscalar(years):
        all_years = [years + k for k in range(n)]
    else:
        all_years = list(years)
        if len(all_years) != n:
            raise ValueError(f"{len(all_years)} years for {n} maps")
    keep_idx = list(range(n)) if keep is None else [all_years.index(y) for y in keep]
    values = [res.maps.values for res in results]
    classes = sorted({_class_value(c) for v in values for c in np.unique(v[~np.isnan(v)])})
    lut = {c: k for k, c in enumerate(classes)}
    first = values[0]
    codes = np.full((len(names), len(seeds), len(keep_idx), *first.shape[1:]), -1, dtype=np.int16)
    for (s, r), v in zip(runs, values):
        sub = v[keep_idx]
        for c, k in lut.items():
            codes[s, r][sub == c] = k
    tables = []
    for (s, r), res in zip(runs, results):
        t = res.steps.reset_index()
        t.insert(0, "replicate", r)
        t.insert(0, "scenario", names[s])
        tables.append(t)
    template = read_map(start, nodata).template
    return Ensemble(names, seeds, [all_years[i] for i in keep_idx], classes, codes,
                    pd.concat(tables, ignore_index=True), template)

