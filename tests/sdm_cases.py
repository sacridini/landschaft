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

GROUPS: list[str] = ["smoke", "cells"]


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


INPUTS: dict[str, Callable[[], object]] = {
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


def case_ids(group: str) -> list[str]:
    return [k for k, c in CASES.items() if c.group == group]


def run_python(name: str) -> dict:
    case = CASES[name]
    return case.py(**{k: build_input(k) for k in case.inputs})


__all__ = ["CASES", "GROUPS", "INPUTS", "Case", "build_input", "case_ids", "np", "run_python"]
