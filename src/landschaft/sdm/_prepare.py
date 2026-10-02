# SPDX-License-Identifier: GPL-3.0-or-later
"""The whole preparation of a species distribution model in one call."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _points

__all__ = ["prepare"]

_FOLD_METHODS = ("blocks", "quadrants", "checkerboard", "random", "jackknife", "clusters")


def _step(value, main: str | None):
    """A step's arguments: None (skip) for False / None, {} for True, {main: v} for a number, the dict
    itself, or the callable."""
    if value is None or value is False:
        return None
    if value is True:
        return {}
    if callable(value) and not isinstance(value, dict):
        return value
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and main is not None:
        return {main: value}
    return {"_object": value}


def _plain(v):
    """A JSON-able view of an argument for the settings."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if callable(v):
        return getattr(v, "__name__", "function")
    if hasattr(v, "geometry"):
        return f"<{len(v)} geometries>"
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return type(v).__name__


def prepare(occ, env, *, species: str | None = None, clean=True, one_per_cell: bool = True, thin=None,
            area=None, predictors=10.0, background=10_000, folds=5, weights: str | None = "balanced",
            seed: int = 1, x: str | None = None, y: str | None = None, crs=None):
    """Prepare occurrence records and predictors for a species distribution model, in one call.

    The steps, in order, each one a function of its own that can be called
    alone (and each argument here takes ``False`` to skip the step, a number
    for its main setting, a dict of that function's arguments, or a
    function of yours):

    1. ``clean`` – :func:`clean_occurrences` (drop records at centroids,
       capitals, institutions, in the sea, outliers...);
    2. ``one_per_cell`` – :func:`one_per_cell`;
    3. ``thin`` – :func:`thin` (a distance in metres; off by default);
    4. ``area`` – the accessible area: :func:`accessible_area` (a buffer
       distance in metres, its arguments, or your polygons; default the
       whole extent of ``env``);
    5. ``predictors`` – :func:`select_predictors` in the area (a VIF
       threshold, default 10);
    6. ``background`` – :func:`background` points in the area (a number,
       default 10 000);
    7. :func:`dataset` – predictor values at presences and background;
    8. ``folds`` – cross-validation folds: a number k of spatial blocks
       (:func:`spatial_blocks`, default 5) or a dict with ``method`` one of
       ``"blocks"``, ``"clusters"``, ``"quadrants"``, ``"checkerboard"``,
       ``"random"``, ``"jackknife"`` and the method's arguments; blocks take
       ``size="auto"`` for the predictors' range of autocorrelation
       (:func:`autocorrelation_range`).

    Every random step uses ``seed``, so the same call gives the same data.
    ``data.report`` counts what each step removed, ``data.settings`` holds
    every choice (for the methods of a paper) and ``data.redo(...)`` repeats
    the preparation with some settings changed.

    Parameters
    ----------
    occ
        Occurrence records: GeoDataFrame of points or a table with
        coordinate columns (e.g. :func:`download_gbif`'s or
        :func:`landschaft.bio.read_occurrences`').
    env
        Predictors: a raster stack (band, y, x), e.g. ``ls.load_raster(files,
        stack=True)``.
    species
        Keep only this species (the records' ``species`` column).
    clean, one_per_cell, thin, area, predictors, background, folds
        The steps (see above).
    weights
        Sample weights of :func:`dataset` (``"balanced"`` by default).
    seed
        Seed of every random step.
    x, y, crs
        Coordinate columns and CRS of a table of records.

    Returns
    -------
    SDMData

    Examples
    --------
    >>> import landschaft as ls
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> env = ls.load_raster(ls.sdm.example_data("bioclim"))
    >>> data = ls.sdm.prepare(occ, env, clean=False, background=2000, area=1_000_000)
    >>> data.report[["step", "presences", "removed"]]
                   step  presences  removed
    0             input        116        0
    1      one per cell         94       22
    2              area         94        0
    3        predictors         94        0
    4        background         94        0
    5  predictor values         94        0
    6             folds         94        0
    >>> data.predictors
    ['bio5', 'bio7', 'bio16', 'bio17']
    >>> again = data.redo(predictors=False)       # the same, keeping every predictor
    >>> len(again.predictors)
    9
    """
    from ._area import accessible_area
    from ._background import background as draw_background
    from ._cells import one_per_cell as per_cell
    from ._clean import clean_occurrences
    from ._data import dataset
    from ._folds import checkerboard_folds, jackknife_folds, quadrant_folds, random_folds, spatial_blocks
    from ._predictors import select_predictors
    from ._thin import thin as thin_records

    kwargs = dict(species=species, clean=clean, one_per_cell=one_per_cell, thin=thin, area=area,
                  predictors=predictors, background=background, folds=folds, weights=weights, seed=seed, x=x,
                  y=y, crs=crs)
    if env.ndim == 2:
        env = env.expand_dims(band=[env.name or "layer"])
    rows = []
    records = occ
    if x is not None or y is not None or crs is not None:  # normalise to points once
        import geopandas as gpd

        xs, ys, c = _points.coordinates(occ, x=x, y=y, crs=crs)
        records = gpd.GeoDataFrame(occ.drop(columns=[x, y], errors="ignore"), geometry=gpd.points_from_xy(xs, ys),
                                   crs=c)
    if species is not None:
        col = next((c for c in ("species", "scientificName", "label") if c in records.columns), None)
        if col is None:
            raise ValueError("species= needs a species column (species, scientificName or label)")
        records = records[records[col] == species]
        if len(records) == 0:
            raise ValueError(f"no record of {species!r}")

    def note(step, before, after, detail=""):
        rows.append({"step": step, "presences": after, "removed": before - after, "detail": detail})

    note("input", len(records), len(records))
    settings = {"seed": seed}

    s = _step(clean, None)
    if s is not None:
        before = len(records)
        if callable(s):
            records = s(records)
            detail = ""
        else:
            s.pop("_object", None)
            records = clean_occurrences(records, **s)
            detail = ", ".join(f"{k} {v}" for k, v in records.attrs.get("clean", {}).items() if v and k != "total")
        note("clean", before, len(records), detail)
        settings["clean"] = _plain(clean)

    if one_per_cell:
        before = len(records)
        records = per_cell(records, env)
        note("one per cell", before, len(records))

    s = _step(thin, "distance")
    if s is not None:
        before = len(records)
        if callable(s):
            records = s(records)
        else:
            s.setdefault("seed", seed)
            records = thin_records(records, **s)
        note("thin", before, len(records), f"{s.get('distance', '')} m" if isinstance(s, dict) else "")
        settings["thin"] = _plain(s)

    polygons = None
    s = _step(area, "distance")
    if s is not None:
        if callable(s):
            polygons = s(records)
        elif "_object" in s:
            polygons = s["_object"]
        else:
            polygons = accessible_area(records, **s)
        settings["area"] = _plain(s)
    note("area", len(records), len(records), "whole extent of the predictors" if polygons is None else "")

    s = _step(predictors, "vif")
    if s is not None:
        if callable(s):
            env_sel = s(env)
        else:
            s.setdefault("seed", seed)
            env_sel = select_predictors(env, area=polygons, **s)
        dropped = [str(b) for b in env.band.values if str(b) not in [str(v) for v in env_sel.band.values]]
        note("predictors", len(records), len(records), "dropped " + ", ".join(dropped) if dropped else "")
        settings["predictors"] = _plain(s)
        env_used = env_sel
    else:
        env_used = env

    s = _step(background, "n")
    if s is None:
        bg = None
    elif callable(s):
        bg = s(env_used, records)
    elif "_object" in s:
        bg = s["_object"]
    else:
        s.setdefault("seed", seed)
        bg = draw_background(env_used, occ=records, area=polygons, **s)
        settings["background"] = _plain(s)
    note("background", len(records), len(records), f"{0 if bg is None else len(bg)} points")

    data = dataset(records, bg, env_used, weights=weights)
    removed = int(data.report.loc[0, "removed"])
    note("predictor values", data.n_presences + removed if bg is None else len(records), data.n_presences,
         "no value in some predictor" if removed else "")
    rows[-1]["presences"] = data.n_presences
    rows[-1]["removed"] = len(records) - data.n_presences

    s = _step(folds, "k")
    if s is not None:
        if callable(s):
            data = s(data)
        else:
            method = s.pop("method", "blocks")
            if method not in _FOLD_METHODS:
                raise ValueError(f"folds method must be one of {', '.join(_FOLD_METHODS)}")
            if method == "blocks":
                s.setdefault("seed", seed)
                if s.get("size") == "auto":  # the median range of autocorrelation of the predictors
                    from ._autocorrelation import autocorrelation_range

                    s["size"] = autocorrelation_range(env_used, seed=seed)["range"]
                    try:
                        data = spatial_blocks(data, **s)
                    except ValueError:
                        import warnings

                        warnings.warn(f"blocks of the autocorrelation range ({s['size'] / 1000:.0f} km) are too large "
                                      "for the folds: using the default grid of blocks", stacklevel=2)
                        s.pop("size")
                        data = spatial_blocks(data, **s)
                else:
                    data = spatial_blocks(data, **s)
            elif method == "quadrants":
                data = quadrant_folds(data, **s)
            elif method == "checkerboard":
                data = checkerboard_folds(data, env_used, **s)
            elif method == "random":
                s.setdefault("seed", seed)
                data = random_folds(data, **s)
            elif method == "jackknife":
                data = jackknife_folds(data)
            else:
                from ._folds import spatial_clusters

                s.setdefault("seed", seed)
                data = spatial_clusters(data, **s)
            s["method"] = method
        settings["folds"] = _plain(s)
        k = len([f for f in np.unique(data.folds) if f > 0]) if data.folds is not None else 0
        note("folds", data.n_presences, data.n_presences, f"{k} folds")

    data.report = pd.DataFrame(rows, columns=["step", "presences", "removed", "detail"])
    data.settings = {**settings, **data.settings, "weights": weights, "n_presences": data.n_presences,
                     "n_background": data.n_background, "predictors_used": list(data.predictors)}
    data.area = polygons
    data._prepare = {"occ": occ, "env": env, "kwargs": kwargs}
    return data
