# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases shared by tools/make_reference_sdm.py and the ls.sdm tests.

Every input is built in Python (a DataFrame, a GeoDataFrame or a raster
DataArray) and handed to R unchanged: tables as raw float64 / JSON columns,
rasters as float64 GeoTIFFs (read with ``terra::rast``, band names kept),
vectors as GeoPackages (read with ``sf::st_read``). Every case runs R code
with the inputs it names as variables and gives a named list; its Python
counterpart (``py``, optional) takes the same inputs as keyword arguments and
gives a dict with the same keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np
import pandas as pd

GROUPS: list[str] = ["smoke", "cells", "thin"]


@dataclass
class Case:
    group: str
    r: str
    inputs: tuple[str, ...] = ()
    py: Callable | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- inputs

@lru_cache(maxsize=None)
def _example(name: str):
    import landschaft as ls

    path = ls.sdm.example_data(name)
    if path.suffix == ".tif":
        return ls.load_raster(path)
    return ls.load_table(path)


def bradypus() -> pd.DataFrame:
    return _example("bradypus").copy()


def acaule() -> pd.DataFrame:
    return _example("acaule").copy()


def bioclim():
    return _example("bioclim").astype("float64")


def edge_points() -> pd.DataFrame:
    """Points on the edges between cells of the bioclim grid, its outer edges and outside it, duplicated."""
    xs = [-125.0, -124.5, -100.0, -32.0, -32.25, -125.25, -31.9, -78.5, -78.5, -60.0]
    ys = [40.0, 39.5, -10.0, -56.0, -55.75, 0.0, 0.0, -8.0, -8.0, 40.25]
    xs, ys = xs + xs[::2], ys + ys[::2]
    return pd.DataFrame({"lon": xs, "lat": ys})


def _r_runif_points(seed: int, n: int, lo=(0.0, 0.0), hi=(1.0, 1.0)) -> pd.DataFrame:
    """lon = runif(n, ...), lat = runif(n, ...) after set.seed(seed), as in R."""
    from landschaft import _core

    r = _core.RRandom(seed)
    lon = r.runif(n, lo[0], hi[0])
    lat = r.runif(n, lo[1], hi[1])
    return pd.DataFrame({"lon": lon, "lat": lat})


def thin_ties() -> pd.DataFrame:
    """T1: 60 points in a 1-degree square (many tied neighbour counts)."""
    return _r_runif_points(7, 60)


def thin_edges() -> pd.DataFrame:
    """T2: T1 plus three exact duplicates and a record without longitude."""
    d = thin_ties()
    extra = pd.DataFrame({"lon": [d.lon[3], d.lon[3], d.lon[10], np.nan], "lat": [d.lat[3], d.lat[3], d.lat[10], 0.5]})
    return pd.concat([d, extra], ignore_index=True)


def thin_threshold() -> pd.DataFrame:
    """T3: points about 15 km apart along the equator and at 60 degrees north."""
    steps = np.array([0.1349, 0.13495, 0.135, 0.13505, 0.1351])
    lon = np.concatenate([np.cumsum(np.tile(steps, 4)), np.cumsum(np.tile(steps * 2, 4))])
    lat = np.concatenate([np.zeros(20), np.full(20, 60.0)])
    return pd.DataFrame({"lon": lon, "lat": lat})


def thin_global() -> pd.DataFrame:
    """300 points over the globe and a 1500 km threshold (rdist.earth's clamp at +-1)."""
    return _r_runif_points(11, 300, (-180.0, -90.0), (180.0, 90.0))


INPUTS: dict[str, Callable[[], object]] = {
    "thin_ties": thin_ties,
    "thin_edges": thin_edges,
    "thin_threshold": thin_threshold,
    "thin_global": thin_global,
    "edge_points": edge_points,
    "bradypus": bradypus,
    "acaule": acaule,
    "bioclim": bioclim,
}


def build_input(name: str):
    return INPUTS[name]()


# ----------------------------------------------------------------------------- cases

CASES: dict[str, Case] = {
    "smoke_inputs": Case(
        "smoke",
        r="list(n = nrow(bradypus), lon = bradypus$lon, species = bradypus$species,"
          " bio1 = values(bioclim[['bio1']], mat = FALSE), names = names(bioclim),"
          " extent = as.vector(ext(bioclim)))",
        inputs=("bradypus", "bioclim"),
        py=lambda bradypus, bioclim: {
            "n": [len(bradypus)], "lon": bradypus["lon"].to_numpy(), "species": bradypus["species"].tolist(),
            "bio1": bioclim.sel(band="bio1").values.ravel(), "names": list(bioclim.band.values),
            "extent": [float(bioclim.x.min()) - 0.25, float(bioclim.x.max()) + 0.25,
                       float(bioclim.y.min()) - 0.25, float(bioclim.y.max()) + 0.25],
        },
    ),
}


_ONE_PER_CELL_R = (
    "{{ cells <- cellFromXY(bioclim, as.matrix({p}[, c('lon', 'lat')]));"
    " keep <- !is.na(cells) & !duplicated(cells){drop};"
    " list(cell = cells - 1, keep = which(keep)) }}"
)
_DROP_NA_R = " & !is.na(cells) & {{ v <- extract(bioclim, ifelse(is.na(cells), 1, cells)); complete.cases(v) }}"


def _one_per_cell_py(points: str, drop: bool):
    def run(**kw):
        import landschaft as ls
        from landschaft.sdm import _points

        occ, env = kw[points], kw["bioclim"]
        xs, ys, _ = _points.coordinates(occ)
        cells = _points.grid_cells(xs, ys, env).astype(float)
        cells[cells < 0] = np.nan
        kept = ls.sdm.one_per_cell(occ, env, drop_missing=drop)
        return {"cell": cells, "keep": occ.index.get_indexer(kept.index) + 1}
    return run


for _p in ("bradypus", "edge_points"):
    for _drop in (False, True):
        CASES[f"one_per_cell_{_p}{'_drop' if _drop else ''}"] = Case(
            "cells", r=_ONE_PER_CELL_R.format(p=_p, drop=_DROP_NA_R if _drop else ""),
            inputs=(_p, "bioclim"), py=_one_per_cell_py(_p, _drop))


_THIN_R = (
    "{{ library(spThin); d <- {p}; set.seed({seed});"
    " out <- thin.algorithm(data.frame(d$lon, d$lat), {km!r}, {reps});"
    " res <- lapply(out, function(o) as.numeric(rownames(o))); names(res) <- paste0('rep', seq_along(res));"
    " res }}"
)


def _thin_py(points: str, distance: float, reps: int, seed: int):
    def run(**kw):
        import landschaft as ls

        occ = kw[points]
        out = ls.sdm.thin(occ, distance, reps=reps, seed=seed, return_all=True)
        return {f"rep{i + 1}": occ.index.get_indexer(o.index) + 1 for i, o in enumerate(out)}
    return run


for _name, _p, _m, _reps, _seed in [
    ("thin_ties", "thin_ties", 15_000.0, 20, 123),
    ("thin_edges", "thin_edges", 15_000.0, 5, 9),
    ("thin_edges_zero", "thin_edges", 0.0, 2, 1),
    ("thin_edges_all", "thin_edges", 1e8, 3, 2),
    ("thin_threshold", "thin_threshold", 15_000.0, 10, 4),
    ("thin_global", "thin_global", 1_500_000.0, 5, 5),
    ("thin_bradypus", "bradypus", 50_000.0, 20, 1),
]:
    CASES[_name] = Case("thin", r=_THIN_R.format(p=_p, seed=_seed, km=_m / 1000.0, reps=_reps), inputs=(_p,),
                        py=_thin_py(_p, _m, _reps, _seed))


def case_ids(group: str) -> list[str]:
    return [k for k, c in CASES.items() if c.group == group]


def run_python(name: str) -> dict:
    case = CASES[name]
    return case.py(**{k: build_input(k) for k in case.inputs})


__all__ = ["CASES", "GROUPS", "INPUTS", "Case", "build_input", "case_ids", "np", "run_python"]
