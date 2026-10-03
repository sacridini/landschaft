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


# ============================================================================= phase 1: change analysis

GROUPS += ["crosstab", "openland", "differ", "markov"]


def series():
    """Four maps (2000, 2005, 2008, 2015) of 40 x 50 cells: classes 2, 3, 5, 10, 11 and a class 12 that
    appears only in 2015, a missing corner, change growing from the west (numpy's generator, seed 11)."""
    import xarray as xr

    rng = np.random.default_rng(11)
    ny, nx = 40, 50
    base = rng.choice([2, 3, 5, 10, 11], size=(ny, nx), p=[0.3, 0.25, 0.2, 0.15, 0.1]).astype(np.float64)
    maps = [base]
    _, xx = np.mgrid[0:ny, 0:nx]
    for k, p in enumerate((0.08, 0.12, 0.2)):
        m = maps[-1].copy()
        change = rng.random((ny, nx)) < p * (1 - xx / nx)
        m[change] = rng.choice([2, 3, 11], size=int(change.sum()))
        if k == 2:
            m[(rng.random((ny, nx)) < 0.03)] = 12
        maps.append(m)
    stack = np.stack(maps)
    stack[:, :6, :4] = np.nan
    return xr.concat([grid(s) for s in stack], dim="band").assign_coords(band=["2000", "2005", "2008", "2015"])


def surfaces():
    """Two continuous maps of 30 x 37 cells and strata (0 to 3; 0 is outside), NaN holes (seed 21)."""
    import xarray as xr

    rng = np.random.default_rng(21)
    ny, nx = 30, 37
    yy, xx = np.mgrid[0:ny, 0:nx]
    g1 = np.sin(xx / 6) + yy / 20 + rng.normal(0, 0.3, (ny, nx))
    g2 = np.sin(xx / 6 + 0.4) + yy / 22 + rng.normal(0, 0.3, (ny, nx))
    g1[3:6, 10:14] = np.nan
    g2[20:22, 30:33] = np.nan
    strata = (1 + (xx // 13)).astype(np.float64)
    strata[:2, :] = 0
    return xr.concat([grid(g1), grid(g2), grid(strata)], dim="band").assign_coords(band=["g1", "g2", "strata"])


def _crafted(counts):
    """Two maps whose cross-tabulation is ``counts`` (classes 1..k, padded with a class 9 that stays)."""
    import xarray as xr

    a, b = [], []
    for i, row in enumerate(counts):
        for j, n in enumerate(row):
            a += [i + 1] * n
            b += [j + 1] * n
    w = 20
    h = -(-len(a) // w)
    pad = h * w - len(a)
    a = np.array(a + [9] * pad, np.float64).reshape(h, w)
    b = np.array(b + [9] * pad, np.float64).reshape(h, w)
    return xr.concat([grid(a), grid(b)], dim="band")


INPUTS.update({
    "series": series,
    "surfaces": surfaces,
    "chain": lambda: _crafted([[90, 10, 0], [0, 80, 20], [0, 0, 100]]),
    "absorbing": lambda: _crafted([[90, 10, 0], [0, 100, 0], [0, 0, 100]]),
    "idempotent": lambda: _crafted([[0, 50], [0, 50]]),
})


def _ls():
    import landschaft as ls

    return ls


def _names(v):
    return [str(int(x)) for x in v]


# ----------------------------------------------------------------------------- lulcc, diffeR crosstab

def _lulcc_py(pie):
    ch = _ls().simulate.changes(pie, times=[0, 6, 14])
    return {"ct06": ch.matrix(0, 6).to_numpy(float), "ct014": ch.matrix(0, 14).to_numpy(float),
            "total": ch.areas.to_numpy(float)}


def _crosstabm_py(series):
    ct = _ls().simulate.crosstab(series[0], series[3]).to_numpy(float)
    return {"ct": ct, "pc": ct / ct.sum() * 100}


# ----------------------------------------------------------------------------- OpenLand

_OPENLAND_R = """
s <- {name}_raster; names(s) <- paste0("map_", c({years}))
ct <- contingencyTable(s, pixelresolution = {pixel})
# OpenLand builds the legend from the classes of origin only (a class that appears only in the last map
# loses its transitions); the corrected legend has every class
cats <- sort(unique(c(ct$lulc_Multistep$From, ct$lulc_Multistep$To)))
ct$tb_legend <- dplyr::tibble(categoryValue = cats, categoryName = factor(as.character(cats)), color = "#000000")
ia <- intensityAnalysis(ct, category_n = "{gain}", category_m = "{loss}")
acc <- acc_changes(s)
{pixel_ia}
list(multi = ct$lulc_Multistep, one = ct$lulc_Onestep, total = as.data.frame(ct$totalArea),
     interval = ct$totalInterval,
     lvl1 = ia$interval_lvl@intervalData,
     gain = ia$category_lvlGain@categoryData, gain_st = ia$category_lvlGain@categoryStationarity,
     loss = ia$category_lvlLoss@categoryData, loss_st = ia$category_lvlLoss@categoryStationarity,
     tgain = ia$transition_lvlGain_n@transitionData, tgain_st = ia$transition_lvlGain_n@transitionStationarity,
     tloss = ia$transition_lvlLoss_m@transitionData, tloss_st = ia$transition_lvlLoss_m@transitionStationarity,
     acc_map = acc[[1]], acc_table = acc[[2]]{extra})
"""

_OPENLAND_PIXEL_R = """,
     px_lvl1 = ia2$interval_lvl@intervalData, px_gain = ia2$category_lvlGain@categoryData,
     px_tloss = ia2$transition_lvlLoss_m@transitionData"""


def _r_order(df, keys):
    """OpenLand's row order: the period factor reversed, then the class names as text."""
    d = df.copy()
    order = {p: -k for k, p in enumerate(sorted(d["Period"].unique()))}
    d["_p"] = d["Period"].map(order)
    sort = ["_p"]
    for k in keys:
        d[f"_{k}"] = d[k].map(lambda v: str(int(v)))
        sort.append(f"_{k}")
    return d.sort_values(sort, kind="stable").drop(columns=sort)


def _openland_py(maps, gain, loss, pixel, pixels=False):
    import pandas as pd

    ch = _ls().simulate.changes(maps, cell_area=float(pixel) ** 2)
    ia = ch.intensity(gain=gain, loss=loss)
    out = {}
    for name, t in (("multi", ch.table), ("one", ch.one_step)):
        out[f"{name}/Period"] = list(t["Period"])
        for col, key in (("From", "From"), ("To", "To"), ("km2", "km2"), ("cells", "QtPixel"),
                         ("Interval", "Interval"), ("yearFrom", "yearFrom"), ("yearTo", "yearTo")):
            out[f"{name}/{key}"] = t[col].to_numpy(float)
    out["total/area_km2"] = np.array([ch.total_area["km2"]])
    out["total/QtPixel"] = np.array([float(ch.total_area["cells"])])
    out["interval"] = np.array([float(ch.total_interval)])
    lvl1 = _r_order(ia.interval, [])
    out["lvl1/Period"] = list(lvl1["Period"])
    for c in ("PercentChange", "St", "U"):
        out[f"lvl1/{c}"] = lvl1[c].to_numpy()
    for key, df, side, cols in (("gain", ia.category_gain, ["To"], ["Interval", "GG_km2", "Gtj", "St"]),
                                ("loss", ia.category_loss, ["From"], ["Interval", "GL_km2", "Lti", "St"]),
                                ("tgain", ia.transition_gain, ["From", "To"], ["Interval", "T_i2n_km2", "Rtin", "Wtn"]),
                                ("tloss", ia.transition_loss, ["To", "From"], ["Interval", "T_m2j_km2", "Qtmj", "Vtm"])):
        d = _r_order(df, side[:1])
        out[f"{key}/Period"] = list(d["Period"])
        for s in side:
            out[f"{key}/{s}"] = _names(d[s])
        for c in cols:
            out[f"{key}/{c}"] = d[c].to_numpy(float)
    for key, df, side, count in (("gain_st", ia.category_gain_stationarity, "To", "Gain"),
                                 ("loss_st", ia.category_loss_stationarity, "From", "Loss"),
                                 ("tgain_st", ia.transition_gain_stationarity, "From", "Loss"),
                                 ("tloss_st", ia.transition_loss_stationarity, "To", "Gain")):
        parts = [g.assign(_k=g[side].map(lambda v: str(int(v)))).sort_values("_k", kind="stable")
                 for _, g in df.groupby("Stationarity", sort=False)]
        d = pd.concat(parts) if parts else df
        out[f"{key}/{side}"] = _names(d[side])
        out[f"{key}/{count}"] = d[count].to_numpy(float)
        out[f"{key}/N"] = d["N"].to_numpy(float)
        out[f"{key}/Stationarity"] = list(d["Stationarity"])
        out[f"{key}/Test"] = list(d["Test"])
    if pixels:
        ia2 = ch.intensity(gain=gain, loss=loss, area=False)
        lvl1 = _r_order(ia2.interval, [])
        out["px_lvl1/Period"] = list(lvl1["Period"])
        for c in ("PercentChange", "St", "U"):
            out[f"px_lvl1/{c}"] = lvl1[c].to_numpy()
        for key, df, side, cols in (("px_gain", ia2.category_gain, ["To"], ["Interval", "GG_pixel", "Gtj", "St"]),
                                    ("px_tloss", ia2.transition_loss, ["To", "From"],
                                     ["Interval", "T_m2j_pixel", "Qtmj", "Vtm"])):
            d = _r_order(df, side[:1])
            out[f"{key}/Period"] = list(d["Period"])
            for sd in side:
                out[f"{key}/{sd}"] = _names(d[sd])
            for c in cols:
                out[f"{key}/{c}"] = d[c].to_numpy(float)
    counts, table = ch.change_count()
    v = np.asarray(counts, dtype=np.float64)
    v[v < 0] = np.nan
    out["acc_map/layer"] = v
    out["acc_table/PxValue"] = table["PxValue"].to_numpy(float)
    out["acc_table/Qt"] = table["Qt"].to_numpy(float)
    out["acc_table/Percent"] = table["Percent"].to_numpy()
    return out


# ----------------------------------------------------------------------------- diffeR

_DIFFER_TABLE_R = """
a <- {name}[[{i}]]; b <- {name}[[{j}]]
ct <- crosstabm(a, b)
pc <- crosstabm(a, b, percent = TRUE)
o <- function(m) c(overallQtyD(m), overallExchangeD(m), overallShiftD(m), overallAllocD(m), overallDiff(m))
list(table = diffTablej(ct), table_change = diffTablej(ct, analysis = "change"), table_pc = diffTablej(pc, digits = 6),
     overall = o(ct), overall_pc = o(pc))
"""


def _differ_table_py(a, b):
    sim = _ls().simulate
    ct = sim.crosstab(a, b)
    pc = ct / ct.to_numpy().sum() * 100
    out = {}
    for key, kw, m in (("table", {"digits": 0}, ct), ("table_change", {"digits": 0, "analysis": "change"}, ct),
                       ("table_pc", {"digits": 6}, pc)):
        d = sim.difference_components(m, **kw)
        out[f"{key}/Category"] = [str(c) for c in d["Category"]]
        for c in d.columns[1:]:
            out[f"{key}/{c}"] = d[c].to_numpy(float)
    for key, m in (("overall", ct), ("overall_pc", pc)):
        o = sim.overall_difference(m)
        out[key] = np.array([o["quantity"], o["exchange"], o["shift"], o["allocation"], o["overall"]])
    return out


_DIFFER_MR_R = """
source(file.path(TOOLS, "differ_fixed.R"))
a <- {name}[[{i}]]; b <- {name}[[{j}]]
m <- differenceMR_fixed(a, b, fact = {fact})
p <- differenceMR_fixed(a, b, fact = {fact}, percent = FALSE)
list(multiple = m, proportion = p[, c("Quantity", "Exchange", "Shift", "Overall")])
"""


def _differ_mr_py(maps, i, j, fact):
    sim = _ls().simulate
    out = {}
    d = sim.difference_resolutions(maps[i], maps[j], factor=fact)
    for c in d.columns:
        out[f"multiple/{c}"] = d[c].to_numpy(float)
    d = sim.difference_resolutions(maps[i], maps[j], factor=fact, percent=False)
    for c in ("Quantity", "Exchange", "Shift", "Overall"):
        out[f"proportion/{c}"] = d[c].to_numpy(float)
    return out


def _composite_py(pie):
    sim = _ls().simulate
    out = {"composite": sim.composite_matrix(pie[0], pie[2], 4).to_numpy()}
    for c, layer in sim.memberships(pie[0], 3).items():
        out[f"members/{int(c)}"] = layer
    return out


def _mad_py(surfaces):
    sim = _ls().simulate
    g1, g2, s = surfaces[0], surfaces[1], surfaces[2]
    out = {}
    for key, kw in (("plain", {}), ("strata", {"strata": s})):
        d = sim.mad(g1, g2, **kw)
        for c in d.columns:
            out[f"{key}/{c}"] = d[c].to_numpy(float)
    d = sim.mad(g1, g2, multiple=False)
    for c in d.columns:
        out[f"original/{c}"] = d[c].to_numpy(float)
    return out


def _sample2pop_py(pie):
    import pandas as pd

    sim = _ls().simulate
    ct = sim.crosstab(pie[0], pie[2])
    pop = pd.DataFrame({"cat": [3, 1, 2], "n": [5000.0, 120000.0, 40000.0]})
    return {"fixed": sim.sample_to_population(ct.astype(float), pop).to_numpy()}


# ----------------------------------------------------------------------------- Dinamica EGO: Markov

_MARKOV_EGO = """Script {{
  t0 := LoadCategoricalMap "t0.tif" .none .default 0;
  t1 := LoadCategoricalMap "t1.tif" .none .default 0;
  single multi := DetermineTransitionMatrix t0 t1 %d;
  SaveTable single "single.csv";
  SaveTable multi "multi.csv";
}};
"""


def _categorical(da):
    """Dinamica's categorical maps are integers: int32, -1 = no data."""
    return da.fillna(-1).astype("int32").rio.write_nodata(-1)


def _markov_job(maps, i, j, steps):
    return {"rasters": {"t0": _categorical(maps[i]), "t1": _categorical(maps[j])}, "script": _MARKOV_EGO % steps,
            "outputs": ["single.csv", "multi.csv"]}


def _markov_py(maps, i, j, steps, multi=True):
    mk = _ls().simulate.Markov.from_maps(maps[i], maps[j], steps=steps)
    out = {}
    for key, per_step in (("single", False), ("multi", True))[:2 if multi else 1]:
        r = mk.rates(multi=per_step)
        out[f"{key}.csv/From*"] = r["From"].to_numpy(float)
        out[f"{key}.csv/To*"] = r["To"].to_numpy(float)
        out[f"{key}.csv/Rate"] = r["Rate"].to_numpy(float)
    return out


def _markov_case(name, i, j, steps, notes="", multi=True):
    job = _markov_job if multi else lambda *a: {**_markov_job(*a), "outputs": ["single.csv"]}
    return Case("markov", "dinamica", (name,), job=lambda **kw: job(kw[name], i, j, steps),
                py=lambda **kw: _markov_py(kw[name], i, j, steps, multi), notes=notes)


CASES.update({
    "lulcc_pie": Case(
        "crosstab", "r", ("pie",),
        r="""
obs <- ObsLulcRasterStack(x = pie_raster, categories = c(1, 2, 3), labels = c("Forest", "Built", "Other"),
                          t = c(0, 6, 14))
list(ct06 = as.matrix(crossTabulate(obs, times = c(0, 6))), ct014 = as.matrix(crossTabulate(obs, times = c(0, 14))),
     total = total(pie_raster, categories = c(1, 2, 3))$total)
""",
        py=_lulcc_py, notes="lulcc's crossTabulate and total."),
    "differ_crosstab_series": Case(
        "crosstab", "r", ("series",),
        r="list(ct = crosstabm(series[[1]], series[[4]]), pc = crosstabm(series[[1]], series[[4]], percent = TRUE))",
        py=_crosstabm_py, notes="diffeR's crosstabm with missing cells and a class found only in the last map."),
    "openland_pie": Case(
        "openland", "r", ("pie",),
        r=_OPENLAND_R.format(name="pie", years="1985, 1991, 1999", pixel=100, gain=2, loss=1,
                             pixel_ia='ia2 <- intensityAnalysis(ct, category_n = "2", category_m = "1", area_km2 = FALSE)',
                             extra=_OPENLAND_PIXEL_R),
        py=lambda pie: _openland_py(pie, 2, 1, 100, pixels=True), notes="Plum Island: gains of built land, losses of forest."),
    "openland_series": Case(
        "openland", "r", ("series",),
        r=_OPENLAND_R.format(name="series", years="2000, 2005, 2008, 2015", pixel=30, gain=11, loss=3,
                             pixel_ia="", extra=""),
        py=lambda series: _openland_py(series, 11, 3, 30),
        notes="Unequal intervals, missing cells, two-digit classes."),
    "differ_table_pie": Case(
        "differ", "r", ("pie",), r=_DIFFER_TABLE_R.format(name="pie", i=1, j=3),
        py=lambda pie: _differ_table_py(pie[0], pie[2]), notes="diffTablej and the overall components."),
    "differ_table_series": Case(
        "differ", "r", ("series",), r=_DIFFER_TABLE_R.format(name="series", i=1, j=4),
        py=lambda series: _differ_table_py(series[0], series[3])),
    "differ_mr_pie": Case(
        "differ", "r", ("pie",), r=_DIFFER_MR_R.format(name="pie", i=1, j=3, fact=2),
        py=lambda pie: _differ_mr_py(pie, 0, 2, 2),
        notes="differenceMR with composite corrected (diffeR gives NaN when a map has missing cells)."),
    "differ_mr_series": Case(
        "differ", "r", ("series",), r=_DIFFER_MR_R.format(name="series", i=2, j=3, fact=3),
        py=lambda series: _differ_mr_py(series, 1, 2, 3), notes="Factor 3, missing cells."),
    "differ_composite": Case(
        "differ", "r", ("pie",),
        r="""
source(file.path(TOOLS, "differ_fixed.R"))
m <- memberships(pie[[1]], fact = 3); names(m) <- sub(".*_", "", names(m))
list(composite = unname(composite_fixed(pie[[1]], pie[[3]], 4)), members = m)
""",
        py=_composite_py, notes="composite (corrected) and memberships."),
    "differ_mad": Case(
        "differ", "r", ("surfaces",),
        r="""
g1 <- surfaces[[1]]; g2 <- surfaces[[2]]; s <- surfaces[[3]]
list(plain = MAD(g1, g2, eval = "multiple"), strata = MAD(g1, g2, strata = s, eval = "multiple"),
     original = MAD(g1, g2, eval = "original"))
""",
        py=_mad_py, notes="MAD with and without strata."),
    "differ_sample2pop": Case(
        "differ", "r", ("pie",),
        r="""
source(file.path(TOOLS, "differ_fixed.R"))
ct <- crosstabm(pie[[1]], pie[[3]])
pop <- data.frame(cat = c(3, 1, 2), n = c(5000, 120000, 40000))
list(fixed = sample2pop_fixed(ct, pop))
""",
        py=_sample2pop_py, notes="sample2pop corrected: the population of the row's category."),
    "markov_pie": _markov_case("pie", 0, 2, 14),
    "markov_sibuyan": _markov_case("sibuyan", 0, 1, 14),
    "markov_series": _markov_case("series", 0, 3, 15, "Missing cells, a class only at the end: Dinamica gives that "
                                  "class a row of zeros (its cells vanish after a step) and its multi-step rates "
                                  "are wrong; here it stays (an identity row), so only the single step compares.",
                                  multi=False),
    "markov_chain": _markov_case("chain", 0, 1, 4, "A root with a negative rate (1 -> 3), left out."),
    "markov_absorbing": _markov_case("absorbing", 0, 1, 3),
    "markov_idempotent": _markov_case("idempotent", 0, 1, 2, "A singular matrix (eigenvalue 0)."),
})
