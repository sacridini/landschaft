# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases shared by tools/make_reference_simulate.py and the ls.simulate tests.

Every input is built in Python (a DataFrame or a raster DataArray). A case
runs on one reference engine:

- ``"r"``: R code (lulcc, OpenLand, diffeR) with the inputs it names as
  variables (rasters also as raster package stacks, ``<name>_raster``),
  giving a named list;
- ``"clumondo"``, ``"grass"``, ``"wsl_python"``, ``"dinamica"``,
  ``"molusce"``: ``job(**inputs)`` builds the engine's job (see
  ``tools/simulate_engines.py``).

Its Python counterpart (``py``, optional) takes the same inputs as keyword
arguments and gives a dict with the same keys.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

import numpy as np

GROUPS: list[str] = ["rng", "smoke"]


@dataclass
class Case:
    group: str
    engine: str
    inputs: tuple[str, ...] = ()
    r: str = ""
    job: Callable | None = None
    py: Callable | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- inputs

@cache
def _example(name: str):
    import landschaft as ls

    return ls.load_raster(ls.simulate.example_data(name))


def pie():
    """Plum Island land use in 1985, 1991 and 1999 (float64, NaN = no data)."""
    lu = _example("pie_landuse")
    return lu.where(lu != 255).astype("float64")


def sibuyan():
    lu = _example("sibuyan_landuse")
    return lu.where(lu != 255).astype("float64")


def grid(values: np.ndarray, res: float = 30.0, crs: str = "EPSG:32723"):
    """A raster DataArray of the values (top-left at 0, 0 + rows * res)."""
    import rioxarray  # noqa: F401  (the .rio accessor)
    import xarray as xr

    v = np.asarray(values)
    ny, nx = v.shape[-2:]
    da = xr.DataArray(v, dims=("y", "x"), coords={"y": (np.arange(ny)[::-1] + 0.5) * res,
                                                  "x": (np.arange(nx) + 0.5) * res})
    return da.rio.write_crs(crs)


def small_change():
    """Two maps of 3 classes on 20 x 30 cells, 20% of the cells changed (numpy's generator, seed 1)."""
    rng = np.random.default_rng(1)
    a = rng.integers(1, 4, (20, 30)).astype(np.uint8)
    b = a.copy()
    m = rng.random(a.shape) < 0.2
    b[m] = rng.integers(1, 4, int(m.sum()))
    return a, b


def small_t0():
    return grid(small_change()[0])


def small_t1():
    return grid(small_change()[1])


def developed():
    """A binary development map (1 developed, 0 not) of 40 x 50 cells with two towns."""
    yy, xx = np.mgrid[0:40, 0:50]
    d = ((yy - 12) ** 2 + (xx - 15) ** 2 < 30) | ((yy - 28) ** 2 + (xx - 38) ** 2 < 45)
    return grid(d.astype(np.uint8))


INPUTS: dict[str, Callable] = {
    "pie": pie,
    "sibuyan": sibuyan,
    "small_t0": small_t0,
    "small_t1": small_t1,
    "developed": developed,
}


def build_input(name: str):
    return INPUTS[name]()


# ----------------------------------------------------------------------------- python helpers

def crosstab(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cells of each class at time 1 (rows) moving to each class at time 2 (columns)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = ~(np.isnan(a) | np.isnan(b))
    classes = np.unique(np.concatenate([a[ok], b[ok]]))
    ia = np.searchsorted(classes, a[ok])
    ib = np.searchsorted(classes, b[ok])
    out = np.zeros((classes.size, classes.size))
    np.add.at(out, (ia, ib), 1)
    return out


# ----------------------------------------------------------------------------- CLUMondo smoke project

def clumondo_smoke_job() -> dict:
    """Three land systems (0 forest, 1 extensive, 2 intensive farming) and one demand (crops).

    Two drivers (a gradient and numpy noise), a 2% yearly growth of the crop
    demand over 3 years, every conversion allowed, random initial ages.
    """
    rng = np.random.default_rng(7)
    ny, nx = 15, 20
    _, xx = np.mgrid[0:ny, 0:nx]
    gradient = (xx / (nx - 1)).astype(np.float64)
    noise = rng.random((ny, nx))
    cov = np.where(gradient + 0.3 * noise > 0.85, 2, np.where(gradient + 0.3 * noise > 0.45, 1, 0))
    supply = {0: 0.0, 1: 1.0, 2: 3.0}
    s0 = sum(supply[k] * int((cov == k).sum()) for k in supply)
    demand = [s0 * 1.02 ** t for t in range(4)]
    main = "\n".join([
        "3", "1", "2", "2", "1",           # land systems, regions, factors per equation, drivers, demand types
        f"{ny}", f"{nx}", "1000", "0", "0",
        "0 1 2",                           # land system codes
        "0.5 0.3 0.5",                     # conversion resistance
        "1",                               # demand type: met exactly
        "0 0.5 1",                         # iteration mode (%), mean and maximum deviation
        "0 3",                             # start, end
        "0",                               # no changing drivers
        "3",                               # ASCII grids with header (.asc)
        "0",                               # one regression for all regions
        "1 5",                             # random initial age, maximum 5
        "0", "0", "0", "0",                # neighbourhood, location factor, dynamic lusmatrix, outputs
        "0.05",                            # iteration parameter
        "0",                               # no initial land system composition maps
    ]) + "\n"
    # per land system: code, constant, number of factors, then (coefficient, factor) pairs
    alloc = ("0\n1.5\n2\n-2.0 0\n0.5 1\n"
             "1\n-0.5\n2\n1.0 0\n-0.2 1\n"
             "2\n-2.5\n2\n3.0 0\n0.3 1\n")
    files = {
        "main.1": main,
        "alloc1.reg": alloc,
        "allow.txt": "1 1 1\n1 1 1\n1 1 1\n",
        "lusconv.txt": "0\n1\n2\n",
        "lusmatrix.txt": "0\n1\n3\n",
        "demand.in1": f"{len(demand)}\n" + "\n".join(repr(float(d)) for d in demand) + "\n",
        "region_park0.fil.asc": np.zeros((ny, nx), dtype=np.int32),
        "cov_all.0.asc": cov.astype(np.int32),
        "sc1gr0.fil.asc": gradient,
        "sc1gr1.fil.asc": noise,
    }
    return {"files": files, "args": ["demand.in1", "region_park0.fil.asc"], "grid": {"cellsize": 1000.0},
            "outputs": ["cov_all.*.asc", "age.*.asc"]}


# ----------------------------------------------------------------------------- MOLUSCE smoke

def molusce_woe_job(import_module) -> dict:
    """Weights of evidence of a 3-class factor for a binary site map (MOLUSCE's ``woe``)."""
    from numpy import ma

    woe = import_module("molusce.algorithms.models.woe.model")
    factor, sites = _woe_inputs()
    res = woe.woe(ma.array(factor), ma.array(sites))
    return {"weights": np.asarray(res["weights"], dtype=float), "map": np.asarray(res["map"], dtype=float)}


def _woe_inputs():
    rng = np.random.default_rng(3)
    factor = rng.integers(1, 4, (12, 15))
    sites = (rng.random((12, 15)) < np.where(factor == 3, 0.5, 0.15)).astype(int)
    return factor, sites


def woe_py() -> dict:
    """The binary weights of evidence written out (Bonham-Carter 1994), as MOLUSCE combines them."""
    import math

    factor, sites = _woe_inputs()
    eps = 4 * np.finfo(float).eps
    f, s = factor.ravel(), sites.ravel().astype(bool)
    A, N = float(f.size), float(s.sum())
    weights = []
    for cat in np.unique(f):
        b = f == cat
        B, Nb = float(b.sum()), float((b & s).sum())
        wp = math.log((Nb / N + eps) / ((B - Nb) / (A - N) + eps))
        wm = math.log(((N - Nb) / N + eps) / ((A - B - N + Nb) / (A - N) + eps))
        weights.append((wp, wm))
    total_minus = sum(w[1] for w in weights)
    w_map = [w[0] + total_minus - w[1] for w in weights]
    out = np.zeros(factor.shape)
    for k, cat in enumerate(np.unique(f)):
        out[factor == cat] = w_map[k]
    return {"weights": np.array(w_map), "map": out}


# ----------------------------------------------------------------------------- cases

_RNG_SCRIPT = r'''
import ctypes, sys, numpy as np
lib = ctypes.CDLL("/usr/lib/grass83/lib/libgrass_gis.so")
lib.G_srand48.argtypes = [ctypes.c_long]
lib.G_drand48.restype = ctypes.c_double
lib.G_lrand48.restype = ctypes.c_long
lib.G_mrand48.restype = ctypes.c_long
for seed in (0, 1, 42, 123456789, -5, 2**40 + 17):
    tag = str(seed).replace("-", "m")
    lib.G_srand48(seed)
    np.save(f"{sys.argv[1]}/drand48_{tag}.npy", np.array([lib.G_drand48() for _ in range(1000)]))
    lib.G_srand48(seed)
    np.save(f"{sys.argv[1]}/lrand48_{tag}.npy", np.array([lib.G_lrand48() for _ in range(1000)], dtype=np.int64))
    lib.G_srand48(seed)
    np.save(f"{sys.argv[1]}/mrand48_{tag}.npy", np.array([lib.G_mrand48() for _ in range(1000)], dtype=np.int64))
'''

_DINAMICA_SMOKE = """Script {{
  t0 := LoadCategoricalMap "small_t0.tif" .none .default 0;
  t1 := LoadCategoricalMap "small_t1.tif" .none .default 0;
  single multi := DetermineTransitionMatrix t0 t1 5;
  SaveTable single "single.csv";
  SaveTable multi "multi.csv";
}};
"""

CASES: dict[str, Case] = {
    "rng_grass": Case(
        "rng", "wsl_python", job=lambda: {"script": _RNG_SCRIPT},
        notes="G_drand48 / G_lrand48 / G_mrand48 of GRASS's C library after G_srand48(seed)."),
    "smoke_r_crosstab": Case(
        "smoke", "r", ("pie",),
        r="""
obs <- ObsLulcRasterStack(x = pie_raster[[c(1, 3)]], categories = c(1, 2, 3),
                          labels = c("Forest", "Built", "Other"), t = c(0, 14))
list(crosstab = as.matrix(crossTabulate(obs, times = c(0, 14))),
     differ = as.matrix(crosstabm(pie[[1]], pie[[3]])))
""",
        py=lambda pie: {"crosstab": crosstab(pie[0], pie[2]), "differ": crosstab(pie[0], pie[2])},
        notes="lulcc::crossTabulate and diffeR::crosstabm: the rasters reach R unchanged."),
    "smoke_clumondo": Case(
        "smoke", "clumondo", job=clumondo_smoke_job,
        notes="A 15 x 20 CLUMondo run of 3 years: the harness writes the project and reads the grids."),
    "smoke_grass_devpressure": Case(
        "smoke", "grass", ("developed",),
        job=lambda developed: {
            "rasters": {"developed": developed},
            "script": "r.futures.devpressure input=developed output=pressure method=gravity size=4 gamma=1.5\n"
                      "r.out.gdal --quiet -c input=pressure output=$OUT/pressure.tif type=Float64\n",
            "outputs": ["pressure.tif"]},
        notes="r.futures.devpressure (gravity): the harness runs GRASS with the r.futures addon in WSL."),
    "smoke_dinamica_matrix": Case(
        "smoke", "dinamica", ("small_t0", "small_t1"),
        job=lambda small_t0, small_t1: {"rasters": {"small_t0": small_t0, "small_t1": small_t1},
                                        "script": _DINAMICA_SMOKE, "outputs": ["single.csv", "multi.csv"]},
        notes="Dinamica EGO's DetermineTransitionMatrix through the console."),
    "smoke_molusce_woe": Case(
        "smoke", "molusce", job=lambda: molusce_woe_job, py=lambda: woe_py(),
        notes="MOLUSCE's weights of evidence imported without QGIS."),
}


def case_ids(group: str) -> list[str]:
    return [k for k, c in CASES.items() if c.group == group]


def run_python(case: str) -> dict:
    c = CASES[case]
    return c.py(**{n: build_input(n) for n in c.inputs})
