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

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

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


#: Inputs drawn with numpy's sin, cos or exp, whose last bits differ between platforms (and so the
#: results compared bit for bit): their values as drawn on x86-64 Windows, where the references were
#: made, are stored and used instead (``write_stored_inputs()`` writes them, on Windows).
STORED_INPUTS = Path(__file__).parent / "data" / "reference" / "simulate" / "inputs.npz"
STORED = ("surfaces", "glm_synth")


def build_input(name: str):
    v = INPUTS[name]()
    if name in STORED and STORED_INPUTS.exists():
        with np.load(STORED_INPUTS) as s:
            v = v.copy(data=s[name])
    return v


def write_stored_inputs() -> None:
    np.savez_compressed(STORED_INPUTS, **{n: np.asarray(INPUTS[n]().values) for n in STORED})


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


# ============================================================================= phase 2: weights of evidence

GROUPS += ["woe"]


def woe_synth():
    """Three classes on 60 x 70 cells, transitions 1 -> 2, 1 -> 3 and 3 -> 2 driven by a distance (minimum 5,
    decimals), an elevation, a slope with holes and a categorical soil (seed 31)."""
    import xarray as xr

    rng = np.random.default_rng(31)
    ny, nx = 60, 70
    yy, xx = np.mgrid[0:ny, 0:nx]
    dist = 5.0 + np.hypot(yy - 20, xx - 15) * 30.0 + rng.random((ny, nx)) * 3
    elev = 80.0 + yy * 7.5 + np.sin(xx / 9) * 40  # Dinamica mishandles negative values: kept positive
    slope = np.abs(np.cos(yy / 7) * 12 + rng.normal(0, 2, (ny, nx)))
    slope[40:46, 50:60] = np.nan
    soil = (1 + (xx // 18) + 2 * (yy > 35)).astype(np.float64)
    soil[soil == 4] = 9
    t0 = rng.choice([1, 2, 3], (ny, nx), p=[0.55, 0.15, 0.3]).astype(np.float64)
    t1 = t0.copy()
    p12 = 0.5 * np.exp(-dist / 500) + 0.02
    p13 = 0.15 * (elev > 300) + 0.03
    p32 = 0.25 * (soil == 2) + 0.05
    r = rng.random((ny, nx))
    t1[(t0 == 1) & (r < p12)] = 2
    t1[(t0 == 1) & (r >= p12) & (r < p12 + p13)] = 3
    t1[(t0 == 3) & (r < p32)] = 2
    t0[0:3, 0:5] = np.nan
    t1[5:7, 60:64] = np.nan
    names = ["t0", "t1", "dist", "elev", "slope", "soil"]
    return xr.concat([grid(v) for v in (t0, t1, dist, elev, slope, soil)], dim="band").assign_coords(band=names)


def woe_strips():
    """The designed strip maps of the range probes (cells and transitions per increment of 30)."""
    import xarray as xr

    def strip(cells, trans):
        rows, nk = max(cells), len(cells)
        t0 = np.full((rows, nk), 3.0)
        t1 = t0.copy()
        for k, (n, t) in enumerate(zip(cells, trans)):
            t0[:n, k] = 1
            t1[:n, k] = 1
            t1[:t, k] = 2
        return t0, t1, np.tile(np.arange(nk) * 30.0, (rows, 1))

    rng = np.random.default_rng(0)
    cells = [int(x) for x in rng.integers(10, 80, 40)]
    p = np.clip(0.3 * np.exp(-np.arange(40) / 12) + 0.02, 0, 1)
    trans = [int(rng.binomial(c, q)) for c, q in zip(cells, p)]
    out = []
    for cl, tr in (([50] * 40, [5] * 10 + [15] * 30), ([50] * 45, [5] * 15 + [20] * 15 + [8] * 15),
                   ([50] * 40, [15] * 10 + [5] * 30), (cells, trans)):
        t0, t1, v = strip(cl, tr)
        pad = 80 - t0.shape[0], 45 - t0.shape[1]
        t0, t1, v = (np.pad(a, ((0, pad[0]), (0, pad[1])), constant_values=np.nan) for a in (t0, t1, v))
        out.append(xr.concat([grid(t0), grid(t1), grid(v)], dim="band").assign_coords(band=["t0", "t1", "v"]))
    return xr.concat(out, dim="strip")


def sibuyan_woe():
    """Sibuyan 1997 -> 2011 with distance to roads, to the sea, elevation, slope and diorite (categorical)."""
    import xarray as xr

    import landschaft as ls

    lu = sibuyan()
    d = ls.load_raster(ls.simulate.example_data("sibuyan_drivers"))
    d = d.where(np.isfinite(d))
    keep = ["dist_roads", "dist_sea", "elevation", "slope", "diorite"]
    bands = [lu[0], lu[1], *[d.sel(band=k) for k in keep]]
    return xr.concat([b.drop_vars("band", errors="ignore") for b in bands], dim="band").assign_coords(
        band=["t0", "t1", *keep])


def pie_woe():
    """Plum Island 1985 -> 1999 (forest -> built) with elevation, slope and distance to built land."""
    import xarray as xr

    import landschaft as ls

    lu = pie()
    d = ls.load_raster(ls.simulate.example_data("pie_drivers"))
    d = d.where(np.isfinite(d))
    keep = ["elevation", "slope", "dist_built_1985"]
    bands = [lu[0], lu[2], *[d.sel(band=k) for k in keep]]
    return xr.concat([b.drop_vars("band", errors="ignore") for b in bands], dim="band").assign_coords(
        band=["t0", "t1", "elev", "slope", "dist"])


INPUTS.update({"woe_synth": woe_synth, "woe_strips": woe_strips, "sibuyan_woe": sibuyan_woe, "pie_woe": pie_woe})


def _woe_skeleton(plan) -> str:
    parts = []
    for (f, t), variables in plan.items():
        vs = []
        for name, spec in variables.items():
            if spec == "categorical":
                vs.append(f'"{name}/{name}"')
            else:
                inc, mx, mn, ang = spec
                vs.append(f'"{name}/{name}" {inc} {mx} {mn} {ang}')
        parts.append(f"{f}->{t} [ {', '.join(vs)} ]")
    return "[ " + ", ".join(parts) + " ]"


def _woe_job(data, plan, *, fix=False, criteria=True, prob=True, corr=True):
    names = sorted({n for v in plan.values() for n in v})
    cats = {n for v in plan.values() for n, s in v.items() if s == "categorical"}
    rasters = {k: data.sel(band=k).assign_attrs(categorical=True) for k in ("t0", "t1")}
    loads, maps = [], []
    for n in names:
        da = data.sel(band=n)
        if n in cats:
            da = da.assign_attrs(categorical=True)
            loads.append(f'  {n} := LoadCategoricalMap "{n}.tif" .none .default 0;')
        else:
            loads.append(f'  {n} := LoadMap "{n}.tif";')
        rasters[n] = da
        maps.append(f'    NameMap {n} "{n}";')
    nm = "\n".join(maps)
    opts = f"{'.yes' if fix else '.no'} {'.yes' if criteria else '.no'}"
    trans = "[ " + ", ".join(f"{f}->{t}" for f, t in plan) + " ]"
    body = [
        'Script {{',
        '  t0 := LoadCategoricalMap "t0.tif" .none .default 0;',
        '  t1 := LoadCategoricalMap "t1.tif" .none .default 0;',
        *loads,
        f"  ranges := DetermineWeightsOfEvidenceRanges t0 t1 {_woe_skeleton(plan)} {opts} {{{{", nm, "  }};",
        f"  weights report := DetermineWeightsOfEvidenceCoefficients t0 t1 ranges {opts} {{{{", nm, "  }};",
        '  SaveTable report "report.csv";',
    ]
    outputs = ["report.csv"]
    if corr and any(len(v) > 1 for v in plan.values()):
        body += [f"  corr := DetermineWeightsOfEvidenceCorrelation ranges {trans} {{{{", nm, "  }};",
                 '  SaveTable corr "corr.csv";']
        outputs.append("corr.csv")
    if prob:
        body += [f"  prob := CalcWOfEProbabilityMap t0 weights {trans} .float64 {{{{", nm, "  }};",
                 '  SaveMap prob "prob.tif";']
        outputs.append("prob.tif")
    body.append("}};")
    return {"rasters": rasters, "script": "\n".join(body) + "\n", "outputs": outputs}


def _woe_py(data, plan, *, fix=False, criteria=True, prob=True, corr=True):
    import landschaft as ls

    ranges = {}
    for (f, t), variables in plan.items():
        ranges[(f, t)] = {n: "categorical" if s == "categorical" else
                          {"increment": s[0], "max_delta": s[1], "min_delta": s[2], "tolerance": s[3]}
                          for n, s in variables.items()}
    names = sorted({n for v in plan.values() for n in v})
    drivers = {n: data.sel(band=n) for n in names}
    w = ls.simulate.WeightsOfEvidence(ranges, fix_abnormal=fix, possible="any" if criteria else "stay")
    w.fit(data.sel(band="t0"), data.sel(band="t1"), drivers)
    out = {}
    tab = w.table.sort_values(["Transition_From", "Transition_To", "Variable", "Range_Lower_Limit"], kind="stable")
    cols = {"Transition_From": "Transition_From*", "Transition_To": "Transition_To*", "Variable": "Variable*",
            "Range_Lower_Limit": "Range_Lower_Limit*", "Range_Upper_Limit": "Range_Upper_Limit*"}
    for c in tab.columns:
        key = f"report.csv/{cols.get(c, c)}"
        out[key] = [f"{v}/{v}" for v in tab[c]] if c == "Variable" else tab[c].to_numpy(float)
    if corr and any(len(v) > 1 for v in plan.values()):
        cr = w.correlation()
        ccols = {"Transition_From": "Transition_From*", "Transition_To": "Transition_To*",
                 "First_Variable": "First_Variable*", "Second_Variable": "Second_Variable*"}
        for c in cr.columns:
            key = f"corr.csv/{ccols.get(c, c)}"
            out[key] = [f"{v}/{v}" for v in cr[c]] if "Variable" in c else cr[c].to_numpy(float)
    if prob:
        out["prob.tif"] = np.squeeze(w.probability(data.sel(band="t0"), drivers).values)
    return out


_SYNTH_PLAN = {(1, 2): {"dist": (25, 100000, 1, 5), "elev": (10, 100000, 1, 5), "soil": "categorical"},
               (1, 3): {"elev": (12.5, 600, 30, 3), "slope": (0.5, 100000, 1, 2)},
               (3, 2): {"soil": "categorical", "dist": (40, 100000, 50, 10), "slope": (1, 100000, 1, 5)}}
_SIB_PLAN = {(1, 3): {"dist_roads": (250, 100000, 1, 5), "elevation": (20, 100000, 1, 5),
                      "slope": (1, 100000, 1, 5), "diorite": "categorical"},
             (3, 2): {"dist_roads": (250, 100000, 1, 5), "dist_sea": (250, 100000, 1, 5)}}
_PIE_PLAN = {(1, 2): {"dist": (100, 100000, 1, 5), "elev": (2, 100000, 1, 5), "slope": (0.5, 100000, 1, 5)}}


def _strip_plan(k):
    return {(1, 2): {"v": (30, 100000, 1, (5, 1, 0.5, 2)[k])}}


def _woe_case(name, plan, notes="", **kw):
    return Case("woe", "dinamica", (name,), job=lambda **d: _woe_job(d[name], plan, **kw),
                py=lambda **d: _woe_py(d[name], plan, **kw), notes=notes)


def _strip_case(k):
    return Case("woe", "dinamica", ("woe_strips",),
                job=lambda woe_strips: _woe_job(woe_strips[k], _strip_plan(k), prob=False),
                py=lambda woe_strips: _woe_py(woe_strips[k], _strip_plan(k), prob=False),
                notes="A designed curve of the range probes.")


CASES.update({
    "woe_synth": _woe_case("woe_synth", _SYNTH_PLAN, "Three transitions, per-transition settings, missing cells."),
    "woe_synth_fix": _woe_case("woe_synth", _SYNTH_PLAN, "Abnormal weights fixed.", fix=True),
    "woe_synth_stay": _woe_case("woe_synth", _SYNTH_PLAN, "Possible = kept the class or made the transition.",
                                criteria=False),
    "woe_sibuyan": _woe_case("sibuyan_woe", _SIB_PLAN, "Sibuyan 1997 -> 2011."),
    "woe_pie": _woe_case("pie_woe", _PIE_PLAN, "Plum Island 1985 -> 1999, forest to built."),
    **{f"woe_strip_{k}": _strip_case(k) for k in range(4)},
})


# ----------------------------------------------------------------------------- logistic regression (lulcc / glm)

GROUPS += ["glm"]


def pie_drivers():
    import landschaft as ls

    d = ls.load_raster(ls.simulate.example_data("pie_drivers")).astype("float64")
    return d.where(np.isfinite(d))


def glm_synth():
    """A map of 3 classes and 4 drivers on 50 x 60 cells, one driver a linear combination of two others
    (aliased in R's glm), holes in a driver (seed 41)."""
    import xarray as xr

    rng = np.random.default_rng(41)
    ny, nx = 50, 60
    yy, xx = np.mgrid[0:ny, 0:nx]
    a = xx / nx + rng.normal(0, 0.1, (ny, nx))
    b = yy / ny + rng.normal(0, 0.1, (ny, nx))
    c = 2 * a - 3 * b
    d = np.sin(xx / 5) * np.cos(yy / 7)
    d[10:14, 20:30] = np.nan
    eta = np.stack([0.5 + 2 * a - b, -1 + 3 * b + d, 0.2 * a])
    p = np.exp(eta) / np.exp(eta).sum(axis=0)
    u = rng.random((ny, nx))
    lu = np.where(u < p[0], 1.0, np.where(u < p[0] + p[1], 2.0, 3.0))
    lu[0:2, 0:6] = np.nan
    return xr.concat([grid(v) for v in (lu, a, b, c, d)], dim="band").assign_coords(band=["lu", "a", "b", "c", "d"])


INPUTS.update({"pie_drivers": pie_drivers, "glm_synth": glm_synth})


def _glm_py(lu, drivers, formulas):
    import landschaft as ls

    lr = ls.simulate.LogisticRegression(formulas).fit(lu, drivers)
    out = {}
    for k, cls in enumerate(formulas):
        f = lr._fits[cls]
        out[f"coef/{k + 1}"] = f["coefficients"]
        out[f"iter/{k + 1}"] = np.array([float(f["iterations"])])
        out[f"dev/{k + 1}"] = np.array([f["deviance"], f["null_deviance"]])
        p = lr.predict(drivers, classes=[cls], landscape=lu).values[0]
        out[f"pred/{k + 1}"] = p
    return out


_GLM_R = """
cells <- which(!is.na(values({lu})))
df <- as.data.frame(values(c({drivers})))
df$lu <- values({lu})[, 1]
forms <- list({forms})
out <- list()
for (k in seq_along(forms)) {{
  cls <- forms[[k]][[1]]; vars <- forms[[k]][[2]]
  d <- df; d$y <- as.numeric(d$lu == cls)
  m <- glm(reformulate(vars, "y"), family = binomial, data = d[cells, ])
  out[[paste0("coef/", k)]] <- unname(coef(m))
  out[[paste0("iter/", k)]] <- m$iter
  out[[paste0("dev/", k)]] <- c(m$deviance, m$null.deviance)
  pr <- rep(NA_real_, nrow(df)); ok <- complete.cases(df[, vars, drop = FALSE]) & !is.na(df$lu)
  pr[ok] <- predict(m, newdata = df[ok, ], type = "response")
  out[[paste0("pred/", k)]] <- matrix(pr, nrow({lu}), ncol({lu}), byrow = TRUE)
}}
out
"""


def _glm_r(lu, drivers, formulas):
    forms = ", ".join(f'list({c}, c({", ".join(repr(v).replace(chr(39), chr(34)) for v in vs)}))'
                      for c, vs in formulas.items())
    return _GLM_R.format(lu=lu, drivers=drivers, forms=forms)


_PIE_FORMS = {2: ["elevation", "slope", "dist_built_1985"], 1: ["elevation", "slope"], 3: ["elevation", "slope"]}
_SYN_FORMS = {1: ["a", "b", "c", "d"], 2: ["b", "d"], 3: ["a"]}

CASES.update({
    "glm_pie": Case(
        "glm", "r", ("pie", "pie_drivers"),
        r=_glm_r("pie[[1]]", "pie_drivers", _PIE_FORMS),
        py=lambda pie, pie_drivers: _glm_py(pie[0], pie_drivers, _PIE_FORMS),
        notes="lulcc's glmModels on Plum Island (the vignette's formulas), every cell."),
    "glm_synth": Case(
        "glm", "r", ("glm_synth",),
        r=_glm_r("glm_synth[[1]]", "glm_synth[[2:5]]", _SYN_FORMS),
        py=lambda glm_synth: _glm_py(glm_synth[0], glm_synth[1:5], _SYN_FORMS),
        notes="An aliased driver (c = 2a - 3b: NA coefficient), holes in a driver."),
})


# ----------------------------------------------------------------------------- MOLUSCE: multi-criteria evaluation

GROUPS += ["mce"]

_MCE_A = [[1, 3, 5, 7], [1 / 3, 1, 2, 4], [1 / 5, 1 / 2, 1, 3], [1 / 7, 1 / 4, 1 / 3, 1]]


def _mce_inputs():
    d = glm_synth()
    lu = d.sel(band="lu").values
    factors = [d.sel(band=k).values for k in ("a", "b", "c", "d")]
    return lu, factors


def molusce_mce_job(import_module) -> dict:
    from numpy import ma

    dp = import_module("molusce.algorithms.dataprovider")
    mce_mod = import_module("molusce.algorithms.models.mce.mce")
    aa = import_module("molusce.algorithms.models.area_analysis.manager")
    lu, factors = _mce_inputs()
    geo = {"xSize": lu.shape[1], "ySize": lu.shape[0], "proj": "", "transform": (0, 1, 0, 0, 0, -1),
           "units": "m"}

    def raster(a, dtype):
        r = dp.Raster()
        m = ma.masked_invalid(a.astype(np.float64)).astype(dtype)
        r.create([m], dict(geo))
        return r

    r = dp.Raster()
    r.create([ma.array(np.where(np.isnan(lu), 0, lu).astype(np.uint8), mask=np.isnan(lu))], dict(geo))
    state = r
    rasters = [raster(f, np.float64) for f in factors]
    analyst = aa.AreaAnalyst(state, second=None)
    m = mce_mod.MCE(rasters, _MCE_A, 1, 2, analyst)
    conf = m.getPrediction(state) and m.getConfidence().getBand(1)
    return {"weights": np.asarray(m.getWeights(), float), "consistency": np.array([m.getConsistency()]),
            "confidence": np.asarray(ma.filled(conf.astype(float), np.nan))}


def _mce_py() -> dict:
    import pandas as pd

    import landschaft as ls

    lu, factors = _mce_inputs()
    names = ["a", "b", "c", "d"]
    mce = ls.simulate.MultiCriteria(pd.DataFrame(_MCE_A, index=names, columns=names), transition=(1, 2))
    # MOLUSCE truncates each weighted term to an integer percentage and adds them as uint8: rebuilt here from
    # our weights and rescaled factors to show they are the same
    conf = np.zeros(lu.shape, dtype=np.uint8)
    mask = np.isnan(lu)
    for w, f in zip(mce.weights.to_numpy(), factors):
        norm = (f - np.nanmin(f)) / (np.nanmax(f) - np.nanmin(f))
        conf = conf + np.nan_to_num(norm * w * 100).astype(np.uint8)
        mask |= np.isnan(f)
    conf = conf * (lu == 1)
    return {"weights": mce.weights.to_numpy(), "consistency": np.array([mce.consistency]),
            "confidence": np.where(mask, np.nan, conf.astype(float))}


CASES.update({
    "mce_molusce": Case("mce", "molusce", job=lambda: molusce_mce_job, py=_mce_py,
                        notes="MOLUSCE's MCE: AHP weights, consistency ratio, the potential map."),
})


# ----------------------------------------------------------------------------- allocation (lulcc CLUE-S, ordered)

GROUPS += ["allocation"]


def pie_rules_maps():
    """Plum Island's land use history (0-12 steps, seed 5) and a mask (lulcc's: 0 = restricted) with a
    restricted block."""
    import xarray as xr

    lu = pie()[0]
    nan = np.isnan(lu.values)
    hist = np.random.default_rng(5).integers(0, 13, lu.shape).astype(np.float64)
    mask = np.ones(lu.shape)
    mask[100:200, 150:300] = 0
    hist[nan] = np.nan
    mask[nan] = np.nan
    return xr.concat([lu.copy(data=hist), lu.copy(data=mask)], dim="band").assign_coords(band=["hist", "mask"])


INPUTS.update({"pie_rules_maps": pie_rules_maps})

_ALLOC_R = """
source(file.path(TOOLS, "lulcc_fixed.R"))
obs <- ObsLulcRasterStack(x = pie_raster, categories = c(1, 2, 3), labels = c("Forest", "Built", "Other"),
                          t = c(0, 6, 14))
ef <- ExpVarRasterList(list(ef_001 = pie_drivers_raster[[1]], ef_002 = pie_drivers_raster[[2]],
                            ef_003 = pie_drivers_raster[[3]]))
cells <- which(!is.na(getValues(obs[[1]])))
train <- getPredictiveModelInputData(obs = obs, ef = ef, cells = cells)
forms <- list(Built ~ ef_001 + ef_002 + ef_003, Forest ~ ef_001 + ef_002, Other ~ ef_001 + ef_002)
models <- glmModels(formula = forms, family = binomial, data = train, obs = obs)
dmd <- approxExtrapDemand(obs = obs, tout = 0:14)
model <- {model}
set.seed({seed})
o <- allocate(model{extra})@output
res <- list(demand = dmd)
for (k in 1:nlayers(o)) res[[paste0("step", k - 1)]] <- matrix(getValues(o[[k]]), nrow(o), ncol(o), byrow = TRUE)
res
"""

_RULES = [[1, 1008, 105], [0, 1, 0], [-1, -2, 1]]
_NB = {1: 0.3, 2: 0.2, 3: 0.3}


def _alloc_r(model, seed, extra=""):
    return _ALLOC_R.replace("{model}", model).replace("{seed}", str(seed)).replace("{extra}", extra)


def _clues_r(elas, params, extra_args=""):
    p = ", ".join(f"{k} = {v!r}" for k, v in params.items())
    return (f"CluesModel(obs = obs, ef = ef, models = models, time = 0:14, demand = dmd, "
            f"elas = c({', '.join(map(str, elas))}), params = list({p}){extra_args})")


_R_RULES = ("rules = matrix(c(" + ", ".join(str(v) for row in _RULES for v in row) + "), 3, 3, byrow = TRUE), "
            "hist = pie_rules_maps_raster[[1]], mask = pie_rules_maps_raster[[2]]")
_R_NB = ("neighb = NeighbRasterStack(x = obs[[1]], weights = matrix(1, 3, 3), categories = c(1, 2, 3)), "
         f"nb.rules = c({', '.join(str(v) for v in _NB.values())})")


def _alloc_py(pie, pie_drivers, allocation, seed, rules=None, pie_rules_maps=None, **kw):
    ls = _ls()
    rules = rules or ls.simulate.Rules()
    if pie_rules_maps is not None:
        rules.allowed = np.array(_RULES)
        rules.history = pie_rules_maps[0]
        rules.restrictions = pie_rules_maps[1] == 0
    lr = ls.simulate.LogisticRegression(_PIE_FORMS).fit(pie[0], pie_drivers)
    demand = ls.simulate.AreaDemand.from_maps(pie, times=[0, 6, 14])
    sim = ls.simulate.Model(lr, demand, allocation, rules).run(pie[0], pie_drivers, seed=seed)
    out = {"demand": demand.table.to_numpy()}
    out.update({f"step{k}": sim.maps[k].values for k in range(sim.maps.shape[0])})
    return out


def _clue_case(seed, elas, params, *, rules=False, nb=False, notes=""):
    ls = _ls()
    names = {"jitter.f": "jitter", "scale.f": "scale", "max.iter": "max_iter", "max.diff": "max_diff",
             "ave.diff": "mean_diff"}
    extra = (", " + _R_RULES if rules else "") + (", " + _R_NB if nb else "")
    inputs = ("pie", "pie_drivers") + (("pie_rules_maps",) if rules else ())

    def py(**kw):
        r = ls.simulate.Rules(elasticity=list(elas),
                              neighbourhood=ls.simulate.Neighbourhood(dict(_NB)) if nb else None)
        return _alloc_py(allocation=ls.simulate.CLUE(**{names[k]: v for k, v in params.items()}), seed=seed,
                         rules=r, **kw)
    return Case("allocation", "r", inputs, r=_alloc_r(_clues_r(elas, params, extra), seed), py=py, notes=notes)


def _ordered_case(seed, order, stochastic, *, rules=False, nb=False, notes=""):
    ls = _ls()
    extra = (", " + _R_RULES if rules else "") + (", " + _R_NB if nb else "")
    inputs = ("pie", "pie_drivers") + (("pie_rules_maps",) if rules else ())
    model = (f"OrderedModel(obs = obs, ef = ef, models = models, time = 0:14, demand = dmd, "
             f"order = c({', '.join(map(str, order))}){extra})")

    def py(**kw):
        r = ls.simulate.Rules(neighbourhood=ls.simulate.Neighbourhood(dict(_NB)) if nb else None)
        return _alloc_py(allocation=ls.simulate.Ordered(order=list(order), stochastic=stochastic), seed=seed,
                         rules=r, **kw)
    return Case("allocation", "r", inputs, r=_alloc_r(model, seed, f", stochastic = {str(stochastic).upper()}"),
                py=py, notes=notes)


_VIGNETTE = {"jitter.f": 0.0002, "scale.f": 0.000001, "max.iter": 1000, "max.diff": 50, "ave.diff": 50}

CASES.update({
    "clues_pie": _clue_case(1, (0.2, 0.2, 0.2), _VIGNETTE,
                            notes="lulcc's vignette: CLUE-S on Plum Island, 14 steps."),
    "clues_pie_mean": _clue_case(7, (0.1, 0.5, 0.3), {**_VIGNETTE, "max.diff": 40, "ave.diff": 12},
                                 notes="The mean criterion (of the absolute differences) is what stops most "
                                       "steps."),
    "clues_pie_rules": _clue_case(11, (0.2, 0.6, 0.1), _VIGNETTE, rules=True,
                                  notes="Conversion rules with every code (0, 1, -1, -2, 100+, 1000+), a history "
                                        "map and a restricted block."),
    "clues_pie_neighb": _clue_case(13, (0.2, 0.2, 0.2), _VIGNETTE, nb=True,
                                   notes="Neighbourhood rules on the three classes (3 x 3 window), updated at "
                                         "every step."),
    "ordered_pie": _ordered_case(3, (2, 1, 3), True, notes="lulcc's vignette: the ordered allocation."),
    "ordered_pie_det": _ordered_case(3, (2, 1, 3), False, notes="Deterministic ordered allocation."),
    "ordered_pie_rules": _ordered_case(5, (2, 1, 3), True, rules=True, nb=True,
                                       notes="Ordered allocation with conversion rules, history, mask and "
                                             "neighbourhood rules."),
})


# ----------------------------------------------------------------------------- CLUMondo

GROUPS += ["clumondo"]


def _clumondo_world(ny=40, nx=50, nsys=3, seed=11, nodata_border=False):
    """Three drivers (float32: two gradients and noise) and a starting map of nsys land systems."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    drivers = [(xx / (nx - 1)).astype(np.float32), (yy / (ny - 1)).astype(np.float32),
               rng.random((ny, nx)).astype(np.float32)]
    score = np.stack([np.cos(k + 3 * drivers[0] + 2 * drivers[1] * (k % 2)) + 0.6 * rng.random((ny, nx))
                      for k in range(nsys)])
    cov = np.argmax(score, axis=0).astype(np.int32)
    region = np.zeros((ny, nx), dtype=np.int32)
    if nodata_border:
        region[:2, :] = -9999
        region[:, -3:] = -9999
    cov[region == -9999] = -9999
    return drivers, cov, region


def _f32(v) -> str:
    return repr(float(np.float32(v)))


def clumondo_job(spec: dict) -> dict:
    """CLUMondo's project files for a spec (see the cases below); the source version's main.1."""
    drivers, cov, region = spec["drivers"], spec["cov"], spec["region"]
    ny, nx = cov.shape
    nsys, ndem = len(spec["resistance"]), len(spec["weights"])
    allow_maps = spec.get("allow_maps", {})
    ndrv = max(len(drivers), 2)
    natt = ndrv + len(allow_maps)
    codes = {pair: ndrv + j for j, pair in enumerate(allow_maps)}
    allowed = np.array(spec.get("allowed", np.ones((nsys, nsys))), dtype=int)
    for (f, t), code in codes.items():
        allowed[f, t] = code
    years = len(spec["demand"]) - 1
    dyn = spec.get("dynamic_drivers", {})        # {driver: {year: array}}
    loc = spec.get("preferences")                # {system: (array, weight)}
    nb = spec.get("neighbourhood")               # {system: {...}}
    age = spec.get("age", 5)                     # int: random up to it (agemode 1); array: age.0
    lusmatrix = spec["lusmatrix"]                # array (nsys, ndem) or {year: array}
    maxf = max(len(c[1]) for c in spec["coefs"])
    main = [str(nsys), "1", str(max(maxf, 1)), str(natt), str(ndem), str(ny), str(nx), "1000", "0", "0",
            " ".join(str(k) for k in range(nsys)),
            " ".join(_f32(v) for v in spec["resistance"]),
            " ".join(_f32(v) for v in spec["weights"]),
            f"{spec.get('itermode', 0)} {_f32(spec['mean_dev'])} {_f32(spec['max_dev'])}",
            f"0 {years}",
            " ".join([str(len(dyn)), *map(str, dyn)]),
            "3", "0",
            f"1 {age}" if np.isscalar(age) else "0",
            "1" if nb else "0",
            " ".join(["1", *(_f32(loc[k][1]) if k in loc else "0" for k in range(nsys))]) if loc else "0",
            "1" if isinstance(lusmatrix, dict) else "0",
            "0", _f32(spec.get("speed", 0.05)), "0"]
    alloc = "".join(f"{k}\n{c[0]!r}\n{len(c[1])}\n" + "".join(f"{_f32(b)} {d}\n" for d, b in c[1])
                    for k, c in enumerate(spec["coefs"]))
    files = {
        "main.1": "\n".join(main) + "\n",
        "alloc1.reg": alloc,
        "allow.txt": "\n".join(" ".join(map(str, r)) for r in allowed) + "\n",
        "lusconv.txt": "\n".join(" ".join(map(str, r)) for r in np.asarray(spec["lusconv"], dtype=int)) + "\n",
        "demand.in1": f"{len(spec['demand'])}\n" + "\n".join(" ".join(repr(float(v)) for v in row)
                                                          for row in spec["demand"]) + "\n",
        "region.fil.asc": region,
        "cov_all.0.asc": cov,
    }

    def lus(a):
        return "\n".join(" ".join(_f32(v) for v in r) for r in np.asarray(a)) + "\n"
    if isinstance(lusmatrix, dict):
        for y in range(1, years + 1):
            files[f"lusmatrix.{y}"] = lus(lusmatrix[max(k for k in lusmatrix if k <= y)])
    else:
        files["lusmatrix.txt"] = lus(lusmatrix)
    for k in range(ndrv):
        files[f"sc1gr{k}.fil.asc"] = drivers[k].astype(np.float64) if k < len(drivers) else np.zeros((ny, nx))
    for d, per_year in dyn.items():
        for y in range(1, years + 1):
            keys = [k for k in per_year if k <= y]
            files[f"sc1gr{d}.{y}.asc"] = (per_year[max(keys)] if keys else drivers[d]).astype(np.float64)
    for mp, code in zip(allow_maps.values(), codes.values()):
        files[f"sc1gr{code}.fil.asc"] = np.asarray(mp, dtype=np.int32)
    if not np.isscalar(age):
        files["age.0.asc"] = np.asarray(age, dtype=np.int32)
    if loc:
        for k in range(nsys):
            files[f"locspec{k}.fil.asc"] = (loc[k][0].astype(np.float64) if k in loc else np.zeros((ny, nx)))
    if nb:
        weights = [nb[k]["weight"] if k in nb else 0.0 for k in range(nsys)]
        txt = [" ".join(_f32(w) for w in weights)]
        for k in range(nsys):
            if weights[k] > 0:
                frm = nb[k].get("from", list(range(nsys)))
                kern = np.asarray(nb[k]["kernel"], dtype=np.float32)
                txt += [str(len(frm)), " ".join(map(str, frm)), str(kern.shape[0] // 2),
                        *(" ".join(_f32(v) for v in r) for r in kern)]
        files["neighmat.txt"] = "\n".join(txt) + "\n"
        files["alloc2.reg"] = "".join(
            f"{k}\n{(nb[k]['constant'] if k in nb else 0.0)!r}\n{len(nb[k]['coefficients']) if k in nb else 0}\n"
            + ("".join(f"{_f32(b)} {s}\n" for s, b in nb[k]["coefficients"].items()) if k in nb else "")
            for k in range(nsys))
    outputs = [f"cov_all.{y}.asc" for y in range(1, years + 1)]
    if not spec.get("maps_only"):
        outputs += [f"age.{y}.asc" for y in range(1, years + 1)]
    return {"files": files, "args": ["demand.in1", "region.fil.asc"], "grid": {"cellsize": 1000.0},
            "outputs": outputs}


def clumondo_job_2017(spec: dict, options: dict) -> dict:
    """The job for the executable of May 2017: line 13 holds the demand types (no weights), line 14 the
    deviation per demand (iteration mode 2), line 26 the patch switch (patch_input.fil)."""
    job = clumondo_job({**spec, "maps_only": True})
    lines = job["files"]["main.1"].rstrip("\n").split("\n")
    services = [f"s{k}" for k in range(len(spec["weights"]))]
    kinds = options.get("kinds", {})
    lines[12] = " ".join({"exact": "1", "minimum": "-1", "maximum": "101"}[kinds.get(v, "exact")] for v in services)
    dev = options.get("max_deviation")
    if isinstance(dev, dict):
        lines[13] = " ".join(["2", *(_f32(dev[v]) for v in services)])
    patches = options.get("patches")
    lines.append("1" if patches else "0")
    if patches:
        n = len(spec["resistance"])
        rows = [[patches.get(k, (0, 0.0, 0))[j] for k in range(n)] for j in range(3)]
        job["files"]["patch_input.fil"] = "\n".join(" ".join(str(v) for v in r) for r in rows) + "\n"
    job["files"]["main.1"] = "\n".join(lines) + "\n"
    return job


def clumondo_model(spec: dict, options: dict | None = None):
    """The spec's model through ls.simulate (options: more CLUMondo arguments): model, landscape, drivers."""
    import pandas as pd

    S = _ls().simulate
    drivers, cov, region = spec["drivers"], spec["cov"], spec["region"]
    services = [f"s{k}" for k in range(len(spec["weights"]))]
    land = np.where(cov == -9999, np.nan, cov.astype(np.float64))
    maps = {f"x{k}": d for k, d in enumerate(drivers)}
    dyn = spec.get("dynamic_drivers", {})
    if dyn:
        drv = {}
        for y in sorted({y for v in dyn.values() for y in v} | {1}):
            m = dict(maps)
            for d, per_year in dyn.items():
                keys = [k for k in per_year if k <= y]
                if keys:
                    m[f"x{d}"] = per_year[max(keys)]
            drv[y] = m
    else:
        drv = maps
    lusmatrix = spec["lusmatrix"]

    def table(a):
        return pd.DataFrame(np.asarray(a, dtype=np.float32).astype(np.float64), columns=services)
    supply = S.Supply({y: table(a) for y, a in lusmatrix.items()} if isinstance(lusmatrix, dict) else table(lusmatrix))
    coefs = {k: {"(Intercept)": c[0], **{f"x{d}": float(np.float32(b)) for d, b in c[1]}}
             for k, c in enumerate(spec["coefs"])}
    nb = spec.get("neighbourhood")
    loc = spec.get("preferences")
    age = spec.get("age", 5)
    model = S.Model(
        potential=S.LogisticRegression.from_coefficients(coefs),
        demand=S.ServiceDemand(pd.DataFrame(np.asarray(spec["demand"], dtype=np.float64), columns=services)),
        allocation=S.CLUMondo(supply, **{
            "order": pd.DataFrame(spec["lusconv"], columns=services), "weights": list(spec["weights"]),
            "deviation": ["percent", "absolute"][spec.get("itermode", 0)], "mean_deviation": spec["mean_dev"],
            "max_deviation": spec["max_dev"], "speed": spec.get("speed", 0.05),
            "neighbourhood": S.NeighbourhoodEffect(nb) if nb else None,
            "preferences": dict(loc) if loc else None, **(options or {})}),
        rules=S.Rules(elasticity=list(spec["resistance"]), allowed=spec.get("allowed"),
                      restrictions=region == -9998, history=age if np.isscalar(age) else age.astype(np.float64),
                      allow_maps=spec.get("allow_maps")))
    return model, land, drv


def clumondo_py(spec: dict, options: dict | None = None) -> dict:
    """The same model through ls.simulate (options: more CLUMondo arguments)."""
    model, land, drv = clumondo_model(spec, options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a year without solution ends the run (as CLUMondo)
        sim = model.run(land, drv, seed=1)
    out = {}
    for y in range(1, sim.maps.shape[0]):
        out[f"cov_all.{y}.asc"] = np.nan_to_num(sim.maps.values[y], nan=-9999)
        if not options:  # the 2017 executable keeps CLUMondo's restricted-age defect: maps only
            out[f"age.{y}.asc"] = np.nan_to_num(sim.ages.values[y], nan=-9999)
    return out


def _clumondo_basic():
    drivers, cov, region = _clumondo_world()
    sup = np.array([[0.0], [1.0], [3.0]])
    s0 = sum(sup[k, 0] * np.sum(cov == k) for k in range(3))
    return {"drivers": drivers, "cov": cov, "region": region, "resistance": [0.5, 0.3, 0.5], "weights": [1.0],
            "lusmatrix": sup, "lusconv": [[0], [1], [2]], "demand": [[s0 * 1.02 ** t] for t in range(6)],
            "coefs": [(1.5, [(0, -2.0), (1, 0.5)]), (-0.5, [(0, 1.0), (2, -0.2)]), (-2.5, [(0, 3.0), (2, 0.3)])],
            "mean_dev": 0.5, "max_dev": 1.0}


def _clumondo_services(growth=(1.015, 1.01, 0.99), years=8):
    """Five systems, three services (one met by several systems, -1 in the order), weights, forbidden
    conversions, a restricted block, no data around."""
    drivers, cov, region = _clumondo_world(nsys=5, seed=12, nodata_border=True)
    region[10:20, 10:25] = np.where(region[10:20, 10:25] == -9999, -9999, -9998)
    sup = np.array([[0.0, 0.0, 1.0], [2.0, 0.5, 0.0], [5.0, 0.2, 0.0], [0.0, 3.0, 0.0], [1.0, 1.0, 0.5]])
    ncell = {k: np.sum(cov == k) for k in range(5)}
    s0 = [sum(sup[k, d] * ncell[k] for k in range(5)) for d in range(3)]
    demand = [[s0[0] * growth[0] ** t, s0[1] * growth[1] ** t, s0[2] * growth[2] ** t] for t in range(years + 1)]
    allowed = np.ones((5, 5), dtype=int)
    allowed[3, 0] = 0
    allowed[2, 0] = 0
    return {"drivers": drivers, "cov": cov, "region": region, "resistance": [0.8, 0.3, 0.6, 0.4, 0.2],
            "weights": [1.0, 0.7, 1.2], "lusmatrix": sup,
            "lusconv": [[0, 0, 2], [1, 1, -1], [2, 0, -1], [0, 2, -1], [1, 2, 1]], "demand": demand,
            "allowed": allowed,
            "coefs": [(0.5, [(1, -1.0)]), (-0.2, [(0, 1.5), (2, -0.4)]), (-1.5, [(0, 2.5)]),
                      (-0.3, [(1, 1.2), (2, 0.2)]), (-1.0, [(0, 0.5), (1, 0.5)])],
            "mean_dev": 0.3, "max_dev": 0.8, "speed": 0.03}


def _clumondo_ages():
    """Age rules: a minimum age (105), a maximum on the diagonal (-104), autonomous succession (1006, also in
    a restricted block), a conversion map; absolute deviations."""
    drivers, cov, region = _clumondo_world(nsys=4, seed=13)
    region[25:35, 5:20] = -9998
    sup = np.array([[0.0], [1.0], [2.5], [0.0]])
    s0 = sum(sup[k, 0] * np.sum(cov == k) for k in range(4))
    allowed = np.ones((4, 4), dtype=int)
    allowed[0, 2] = 105
    allowed[2, 2] = -104
    allowed[3, 0] = 1006
    gate = np.zeros(cov.shape, dtype=np.int32)
    gate[:, : cov.shape[1] // 2] = 1
    return {"drivers": drivers, "cov": cov, "region": region, "resistance": [0.4, 0.3, 0.2, 0.5], "weights": [1.0],
            "lusmatrix": sup, "lusconv": [[0], [1], [2], [0]], "demand": [[s0 + 6 * t] for t in range(11)],
            "allowed": allowed, "allow_maps": {(1, 0): gate}, "itermode": 1, "age": 8,
            "coefs": [(0.2, [(0, -1.0)]), (-0.4, [(0, 1.0), (2, 0.1)]), (-1.0, [(0, 2.0)]), (-0.5, [(1, 1.0)])],
            "mean_dev": 5.0, "max_dev": 10.0}


def _clumondo_neighbourhood():
    """Neighbourhood effects (two systems, kernels of radius 1 and 2) and location preferences."""
    spec = _clumondo_services()
    ny, nx = spec["cov"].shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    k2 = np.ones((5, 5), dtype=np.float32)
    k2[2, 2] = 0
    spec["neighbourhood"] = {
        2: {"weight": 0.6, "kernel": np.ones((3, 3), dtype=np.float32), "from": [1, 2, 4], "constant": -0.1,
            "coefficients": {2: 1.0, 3: -0.5}},
        3: {"weight": 0.4, "kernel": k2, "from": [0, 1, 3], "constant": 0.0, "coefficients": {3: 0.8}},
    }
    spec["preferences"] = {0: (((xx + yy) % 7 == 0).astype(np.float32), 0.5),
                           4: ((yy / ny).astype(np.float32), 0.3)}
    return spec


def _clumondo_dynamic():
    """Drivers and supply that change over time, starting ages from a map."""
    spec = _clumondo_basic()
    d0 = spec["drivers"][0]
    spec["dynamic_drivers"] = {0: {2: np.clip(d0 + 0.1, 0, 1).astype(np.float32),
                                   4: np.clip(d0 + 0.25, 0, 1).astype(np.float32)}}
    spec["lusmatrix"] = {1: np.array([[0.0], [1.0], [3.0]]), 3: np.array([[0.0], [1.2], [3.3]])}
    spec["age"] = (np.random.default_rng(3).integers(1, 12, spec["cov"].shape)).astype(np.int32)
    spec["allowed"] = np.array([[1, 1, 104], [1, 1, 1], [1, 1, 1]])
    return spec


_CLUMONDO_SPECS = {"basic": _clumondo_basic, "services": _clumondo_services, "ages": _clumondo_ages,
                   "neighbourhood": _clumondo_neighbourhood, "dynamic": _clumondo_dynamic,
                   "stops": lambda: _clumondo_services(growth=(1.03, 1.01, 0.99), years=6)}


def _services_2017():
    spec = _clumondo_services()
    spec["weights"] = [1.0, 1.0, 1.0]  # the 2017 executable has no weights (line 13 holds the demand types)
    return spec


def _clumondo_2017_case(spec, options, notes):
    return Case("clumondo", "clumondo_2017", job=lambda: clumondo_job_2017(spec(), options),
                py=lambda: clumondo_py(spec(), options), notes=notes, extra={"agreement": True})


def _clumondo_case(name, engine, notes):
    return Case("clumondo", engine, job=lambda: clumondo_job(_CLUMONDO_SPECS[name]()),
                py=lambda: clumondo_py(_CLUMONDO_SPECS[name]()), notes=notes)


CASES.update({
    "clumondo_basic": _clumondo_case("basic", "clumondo", "Three systems, one service, random ages; the "
                                     "original CLUMondo (no restricted cells: the correction does not act)."),
    "clumondo_services": _clumondo_case("services", "clumondo_fixed", "Five systems, three services, weights, "
                                        "-1 in the order, forbidden conversions, restricted block, no data."),
    "clumondo_ages": _clumondo_case("ages", "clumondo_fixed", "Minimum and maximum age, autonomous succession "
                                    "(also in a restricted block), a conversion map, absolute deviations."),
    "clumondo_neighbourhood": _clumondo_case("neighbourhood", "clumondo_fixed",
                                             "Neighbourhood effects and location preferences."),
    "clumondo_dynamic": _clumondo_case("dynamic", "clumondo", "Dynamic drivers and supply, ages from a map."),
    "clumondo2017_kinds": _clumondo_2017_case(
        _services_2017, {"kinds": {"s1": "minimum", "s2": "maximum"}},
        "Minimum and maximum demands against the executable of May 2017 (no source: cells in agreement)."),
    "clumondo2017_mode2": _clumondo_2017_case(
        _services_2017, {"max_deviation": {"s0": 0.5, "s1": 2.0, "s2": 5.0}},
        "A deviation per demand (iteration mode 2) against the 2017 executable (cells in agreement)."),
    "clumondo2017_patches": _clumondo_2017_case(
        _clumondo_basic, {"patches": {2: (1, 0.3, 4)}},
        "Patches (radius 1, threshold 0.3, 4 cells) against the 2017 executable (cells in agreement; it "
        "finds no solution after year 3)."),
    "clumondo_stops": _clumondo_case("stops", "clumondo_fixed", "No solution in year 6 (20000 iterations): "
                                     "the simulation stops after year 5."),
})


# ----------------------------------------------------------------------------- FUTURES

GROUPS += ["futures"]

_FUT_STEPS = 5


def futures_world():
    """60 x 80 cells (30 m): developed clusters, an excluded lake, three subregions (ids 30, 10, 20 in order of
    first appearance), two potential subregions, three predictors, a weight map and zones (seed 17)."""
    import xarray as xr

    rng = np.random.default_rng(17)
    ny, nx = 60, 80
    yy, xx = np.mgrid[0:ny, 0:nx]
    centres = [(15, 20), (40, 55), (10, 65), (48, 15)]
    dist = np.min([np.hypot(yy - a, xx - b) for a, b in centres], axis=0)
    developed = (dist + 3 * rng.random((ny, nx)) < 6).astype(np.float64)
    lake = np.hypot(yy - 30, xx - 35) < 6
    developed[lake] = np.nan
    sub = np.where(xx < 30, 30.0, np.where(yy < 30, 10.0, 20.0))
    sub[0, 40] = 20.0  # row order: 30 first, then 20 (this cell), then 10
    psub = np.where(yy < 35, 1.0, 2.0)
    slope = (np.abs(np.sin(xx / 9.0) * np.cos(yy / 7.0)) * 20 + rng.random((ny, nx))).round(3)
    road = np.minimum(np.abs(yy - 25), np.abs(xx - 45)) * 30.0
    forest = rng.random((ny, nx)).round(4)
    forest[5, 5] = np.nan  # a hole in a predictor: the cell is excluded
    weight = np.where((xx > 60) & (yy > 40), -0.6, np.where((xx < 15) & (yy < 15), 0.5, 0.0))
    zone = np.select([xx < 20, xx < 40, yy < 20], [130.0, 200.0, 101.0], 301.0)
    layers = {"developed": developed, "sub": sub, "psub": psub, "slope": slope, "road": road, "forest": forest,
              "weight": weight, "zone": zone}
    return xr.concat([grid(v) for v in layers.values()], dim="band").assign_coords(band=list(layers))


INPUTS.update({"futures_world": futures_world})

_FUT_POT = [  # subregion, intercept, development_pressure, slope, road, forest
    (30, -2.6, 0.08, -0.06, -0.0011, -0.8),
    (10, -2.2, 0.11, -0.05, -0.0009, -1.1),
    (20, -2.9, 0.06, -0.07, -0.0013, -0.5),
    (1, -2.5, 0.09, -0.06, -0.001, -0.9),
]
_FUT_POT_PSUB = [(1, -2.4, 0.09, -0.06, -0.001, -0.9), (2, -2.8, 0.07, -0.05, -0.0012, -0.6)]
_FUT_DEMAND = {30: [40, 35, 50, 30, 45], 10: [25, 30, 20, 35, 28], 20: [30, 22, 41, 18, 33],
               1: [95, 87, 111, 83, 106]}
_FUT_PATCHES = {30: [1, 3, 5, 8, 2, 12, 4, 1, 6], 10: [2, 2, 7, 15, 3, 1], 20: [4, 9, 1, 1, 5, 20, 2, 3]}


def _all_patches():
    return [v for sizes in _FUT_PATCHES.values() for v in sizes]


def _fut_table(rows, header):
    return "\n".join([",".join(header)] + [",".join(repr(v) for v in r) for r in rows]) + "\n"


def _fut_files(opts):
    pot = _FUT_POT_PSUB if opts.get("potential_subregions") else _FUT_POT
    regions = opts.get("regions", [30, 10, 20])
    files = {"potential.csv": _fut_table(pot, ["ID", "Intercept", "development_pressure", "slope", "road", "forest"])}
    years = list(range(2021, 2021 + _FUT_STEPS))
    files["demand.csv"] = _fut_table([(y, *(_FUT_DEMAND[r][k] for r in regions)) for k, y in enumerate(years)],
                                     ["year", *map(str, regions)])
    if opts.get("patch_columns"):
        n = max(len(v) for v in _FUT_PATCHES.values())
        rows = [[str(_FUT_PATCHES[r][k]) if k < len(_FUT_PATCHES[r]) else "" for r in regions] for k in range(n)]
        files["patches.csv"] = ",".join(map(str, regions)) + "\n" + "\n".join(",".join(r) for r in rows) + "\n"
    else:
        files["patches.csv"] = "\n".join(str(v) for v in _all_patches()) + "\n"
    if opts.get("zoning_effects") is not None:
        z = opts["zoning_effects"]
        files["zoning.csv"] = ",".join(["region", "stringency", *map(str, z["zones"])]) + "\n" + "\n".join(
            ",".join([str(r), repr(z["stringency"][r]), *(repr(e) for e in z["effects"][r])]) for r in regions) + "\n"
    return files


def futures_job(world, opts, module="r.futures.simulation"):
    bands = ["developed", "sub", "slope", "road", "forest"]
    if opts.get("potential_subregions"):
        bands.append("psub")
    if opts.get("weight"):
        bands.append("weight")
    if opts.get("zoning"):
        bands.append("zone")
    rasters = {b: world.sel(band=b).drop_vars("band") for b in bands}
    if opts.get("regions") == [1]:
        rasters["sub"] = rasters["sub"].copy(data=np.ones(rasters["sub"].shape))
    files = _fut_files(opts)
    heredocs = "".join(f"cat > {name} <<'EOF'\n{text}EOF\n" for name, text in files.items())
    pr = opts.get("pressure", {"method": "gravity", "size": 8, "gamma": 1.5, "scaling_factor": 1.0})
    dp = (f"r.futures.devpressure -n input=developed output=pressure method={pr['method']} size={pr['size']}"
          + (f" gamma={pr['gamma']} scaling_factor={pr['scaling_factor']}" if pr["method"] != "occurrence" else ""))
    args = ["developed=developed subregions=sub predictors=slope,road,forest development_pressure=pressure",
            f"n_dev_neighbourhood={pr['size']} development_pressure_approach={pr['method']}",
            f"gamma={pr.get('gamma', 1.5)} scaling_factor={pr.get('scaling_factor', 1.0)}",
            "devpot_params=potential.csv demand=demand.csv patch_sizes=patches.csv",
            f"num_neighbors={opts.get('num_neighbors', 4)} seed_search={opts.get('seed_search', 'probability')}",
            f"compactness_mean={opts.get('compactness_mean', 0.4)} compactness_range={opts.get('compactness_range', 0.05)}",
            f"discount_factor={opts.get('discount_factor', 1.0)} random_seed={opts.get('seed', 1)}",
            f"incentive_power={opts.get('incentive_power', 1.0)} output=final output_series=step"]
    if opts.get("potential_subregions"):
        args.append("subregions_potential=psub")
    if opts.get("weight"):
        args.append("potential_weight=weight")
    if opts.get("zoning"):
        args.append("zoning=zone" + (" zoning_effects=zoning.csv" if opts.get("zoning_effects") else ""))
    script = (heredocs + dp + "\n" + module + " " + " ".join(args) + "\n"
              + "r.out.gdal --quiet -c input=pressure output=$OUT/pressure.tif type=Float32\n"
              + "".join(f"r.out.gdal --quiet -c input=step_{k} output=$OUT/step_{k}.tif type=Int32\n"
                        for k in range(1, _FUT_STEPS + 1)))
    outputs = ["pressure.tif"] + [f"step_{k}.tif" for k in range(1, _FUT_STEPS + 1)]
    return {"rasters": rasters, "script": script, "outputs": outputs}


def futures_py(world, opts) -> dict:
    import pandas as pd

    S = _ls().simulate
    regions = opts.get("regions", [30, 10, 20])
    w = {b: world.sel(band=b).drop_vars("band") for b in world.band.values}
    sub = w["sub"] if regions != [1] else w["sub"].copy(data=np.ones(w["sub"].shape))
    pot_rows = _FUT_POT_PSUB if opts.get("potential_subregions") else _FUT_POT
    pot = S.DevelopmentPotential(pd.DataFrame(pot_rows, columns=["ID", "Intercept", "development_pressure", "slope",
                                                                  "road", "forest"]).set_index("ID"))
    years = list(range(2021, 2021 + _FUT_STEPS))
    demand = S.DevelopmentDemand(pd.DataFrame({r: _FUT_DEMAND[r] for r in regions}, index=years))
    patches = {r: _FUT_PATCHES[r] for r in regions} if opts.get("patch_columns") else \
        _all_patches()
    pr = opts.get("pressure", {"method": "gravity", "size": 8, "gamma": 1.5, "scaling_factor": 1.0})
    ze = opts.get("zoning_effects")
    zoning_effects = None
    if ze is not None:
        zoning_effects = pd.DataFrame({"stringency": [ze["stringency"][r] for r in regions],
                                       **{z: [ze["effects"][r][k] for r in regions] for k, z in enumerate(ze["zones"])}},
                                      index=regions)
    alloc = S.PatchGrowing(
        patches, subregions=sub, potential_subregions=w["psub"] if opts.get("potential_subregions") else None,
        neighborhood=pr["size"], approach=pr["method"], gamma=pr.get("gamma", 1.5),
        scaling_factor=pr.get("scaling_factor", 1.0), discount_factor=opts.get("discount_factor", 1.0),
        compactness_mean=opts.get("compactness_mean", 0.4), compactness_range=opts.get("compactness_range", 0.05),
        num_neighbors=opts.get("num_neighbors", 4), seed_search=opts.get("seed_search", "probability"),
        incentive_power=opts.get("incentive_power", 1.0), weight=w["weight"] if opts.get("weight") else None,
        zoning=w["zone"] if opts.get("zoning") else None, zoning_effects=zoning_effects)
    drivers = {k: w[k] for k in ("slope", "road", "forest")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = S.Model(pot, demand, alloc).run(w["developed"], drivers, seed=opts.get("seed", 1))
    pressure = S.development_pressure(w["developed"], pr["size"], method=pr["method"], gamma=pr.get("gamma"),
                                      scaling_factor=pr.get("scaling_factor"))
    out = {"pressure.tif": pressure.values.astype(np.float64)}
    for k in range(1, _FUT_STEPS + 1):
        out[f"step_{k}.tif"] = sim.development.values[k]
    return out


def _fut_case(opts, module, notes):
    return Case("futures", "grass", ("futures_world",),
                job=lambda futures_world: futures_job(futures_world, opts, module),
                py=lambda futures_world: futures_py(futures_world, opts), notes=notes)


_ZONING = {"zones": [130, 200, 101, 301],
           "stringency": {30: 1.3, 10: 0.7, 20: 1.0},
           "effects": {30: [-0.5, 0.2, -0.1, 0.4], 10: [-0.8, -0.3, 0.6, 0.1], 20: [0.0, -0.2, -0.9, 0.3]}}

CASES.update({
    "futures_one": _fut_case({"regions": [1]}, "r.futures.simulation",
                             "One subregion (patches cannot cross into another: the module as released)."),
    "futures_regions": _fut_case({"patch_columns": True, "num_neighbors": 8, "discount_factor": 0.7},
                                 "r.futures.simulation_fixed",
                                 "Three subregions, a patch library per subregion, 8 neighbours, discount factor; "
                                 "patches cross subregions (pressure correction)."),
    "futures_random": _fut_case({"seed_search": "random", "incentive_power": 2.0, "weight": True, "seed": 7,
                                 "pressure": {"method": "kernel", "size": 6, "gamma": 2.0, "scaling_factor": 3.0},
                                 "compactness_mean": 0.6, "compactness_range": 0.3},
                                "r.futures.simulation_fixed",
                                "Random seeds, infill incentive, potential weights, kernel pressure."),
    "futures_zoning": _fut_case({"zoning": True, "potential_subregions": True, "seed": 3,
                                 "pressure": {"method": "occurrence", "size": 5}},
                                "r.futures.simulation_fixed",
                                "Predefined zoning effects, potential subregions, occurrence pressure."),
    "futures_zoning_custom": _fut_case({"zoning": True, "zoning_effects": _ZONING, "seed": 11,
                                        "incentive_power": 0.5},
                                       "r.futures.simulation_fixed",
                                       "Zoning effects per subregion with stringency; sprawl incentive."),
})


def _fut_pressure_job(world, method, size, gamma, scaling):
    dev = world.sel(band="developed").drop_vars("band")
    extra = f" gamma={gamma} scaling_factor={scaling}" if method != "occurrence" else ""
    script = (f"r.futures.devpressure input=developed output=pressure method={method} size={size}{extra}\n"
              "r.out.gdal --quiet -c input=pressure output=$OUT/pressure.tif type=Float32\n")
    return {"rasters": {"developed": dev}, "script": script, "outputs": ["pressure.tif"]}


def _fut_pressure_py(world, method, size, gamma, scaling):
    dev = world.sel(band="developed").drop_vars("band")
    p = _ls().simulate.development_pressure(dev, size, method=method, gamma=gamma, scaling_factor=scaling, pad=False)
    return {"pressure.tif": p.values.astype(np.float64)}


def _fut_pressure_case(method, size, gamma=None, scaling=None):
    return Case("futures", "grass", ("futures_world",),
                job=lambda futures_world: _fut_pressure_job(futures_world, method, size, gamma, scaling),
                py=lambda futures_world: _fut_pressure_py(futures_world, method, size, gamma, scaling),
                notes=f"r.futures.devpressure without -n ({method}): edges keep their values, the lake gives NaN.")


def _fut_observed(world):
    """Four observed maps of development (the world's grown outwards) at 2000, 2005, 2010, 2015."""
    from scipy import ndimage

    dev = world.sel(band="developed").values
    lake = np.isnan(dev)
    maps = [dev]
    cur = np.nan_to_num(dev).astype(bool)
    for k in range(3):
        grown = ndimage.binary_dilation(cur, iterations=1)
        rng = np.random.default_rng(100 + k)
        cur = cur | (grown & (rng.random(cur.shape) < 0.35))
        maps.append(np.where(lake, np.nan, cur.astype(np.float64)))
    return maps


_FUT_OBS_POP = [(2000, 12000, 9000, 15000), (2005, 13500, 9600, 15800), (2010, 15600, 10100, 16900),
                (2015, 17000, 10900, 17500)]
_FUT_PROJ_POP = [(2020, 18800, 11500, 18600), (2025, 20500, 12300, 19300), (2030, 22900, 12900, 20800)]


def _fut_demand_job(world, methods):
    maps = _fut_observed(world)
    rasters = {f"dev{k}": world.sel(band="developed").drop_vars("band").copy(data=m) for k, m in enumerate(maps)}
    rasters["sub"] = world.sel(band="sub").drop_vars("band")
    header = "year,30,10,20"
    obs = header + "\n" + "\n".join(",".join(map(str, r)) for r in _FUT_OBS_POP) + "\n"
    proj = header + "\n" + "\n".join(",".join(map(str, r)) for r in _FUT_PROJ_POP) + "\n"
    times = ",".join(str(y) for y in range(2015, 2031))
    script = (f"cat > obs.csv <<'EOF'\n{obs}EOF\ncat > proj.csv <<'EOF'\n{proj}EOF\n"
              "r.mapcalc --quiet expression='subi = int(sub)'\n"
              f"r.futures.demand development={','.join(f'dev{k}' for k in range(len(maps)))} subregions=subi "
              f"observed_population=obs.csv projected_population=proj.csv simulation_times={times} "
              f"method={','.join(methods)} demand=$OUT/demand.csv population_demand=$OUT/population.csv "
              f"resulting_curves=$OUT/curves.txt separator=comma\n")
    return {"rasters": rasters, "script": script, "outputs": ["demand.csv", "population.csv", "curves.txt"]}


def _fut_demand_py(world, methods):
    import pandas as pd

    S = _ls().simulate
    maps = _fut_observed(world)
    obs = pd.DataFrame(_FUT_OBS_POP, columns=["year", 30, 10, 20]).set_index("year")
    proj = pd.DataFrame(_FUT_PROJ_POP, columns=["year", 30, 10, 20]).set_index("year")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = S.DevelopmentDemand.from_population(maps, world.sel(band="sub").values, obs, proj, range(2015, 2031),
                                                methods=methods)
    out = {"demand.csv/year": np.asarray(d.table.index, dtype=np.float64),
           "population.csv/year": np.asarray(d.population.index, dtype=np.float64)}
    for c in d.table.columns:
        out[f"demand.csv/{c}"] = d.table[c].to_numpy(dtype=np.float64)
        out[f"population.csv/{c}"] = d.population[c].to_numpy(dtype=np.float64)
    lines = ["subregion,method,coef,rmse"]
    for sid, row in d.curves.sort_index(key=lambda i: i.astype(str)).iterrows():
        lines.append(f"{sid},{row['method']},{';'.join(str(c) for c in row['coefficients'])},{row['rmse']}")
    out["curves.txt"] = lines
    return out


def _fut_demand_case(methods, notes):
    return Case("futures", "grass", ("futures_world",),
                job=lambda futures_world: _fut_demand_job(futures_world, methods),
                py=lambda futures_world: _fut_demand_py(futures_world, methods), notes=notes,
                extra={"text_tolerance": True})


CASES.update({
    "futures_pressure_gravity": _fut_pressure_case("gravity", 4, 1.5, 1.0),
    "futures_pressure_kernel": _fut_pressure_case("kernel", 3, 2.0, 5.0),
    "futures_pressure_occurrence": _fut_pressure_case("occurrence", 2),
    "futures_demand": _fut_demand_case(("linear", "logarithmic", "exponential"),
                                       "r.futures.demand: linear, logarithmic and exponential regressions."),
    "futures_demand_curves": _fut_demand_case(("exp_approach", "logarithmic2", "linear"),
                                              "r.futures.demand: scipy's curve fits (exp_approach, logarithmic2)."),
})


# ----------------------------------------------------------------------------- FUTURES: floods

_FLOOD_PROBS = (0.5, 0.2, 0.05, 0.01)
_FLOOD_POP = {30: [400, 380, 520, 300, 460], 10: [250, 310, 190, 360, 280], 20: [330, 210, 420, 170, 350]}
_REDIST = {30: (60.0, 20.0, 15.0, 5.0), 10: (10.0, 70.0, 15.0, 5.0), 20: (25.0, 25.0, 40.0, 10.0)}  # to 30, 10, 20, 99
_DDF = {"levels": (0.3, 0.6, 0.9, 1.5),
        "curves": {30: (5.0, 25.0, 45.0, 80.0), 10: (10.0, 30.0, 60.0, 90.0), 20: (2.0, 15.0, 35.0, 70.0)}}


def futures_flood_world():
    """Climate layers on futures_world's grid: catchments 101-104, adaptive capacity, HAND, a river, flood depths
    for four probabilities at steps 1 and 3 and flood probability maps at steps 1 and 4."""
    import xarray as xr

    rng = np.random.default_rng(23)
    ny, nx = 60, 80
    yy, xx = np.mgrid[0:ny, 0:nx]
    huc = np.where(yy + 0.3 * xx < 40, 101.0, np.where(xx < 45, 102.0, np.where(yy < 35, 103.0, 104.0)))
    ac = np.clip(np.sin(xx / 13.0) * np.cos(yy / 11.0) + 0.3 * rng.standard_normal((ny, nx)), -1, 1).round(3)
    river = np.abs(yy - (30 + 8 * np.sin(xx / 10.0)))
    hand = (river * 0.15 + 0.2 * rng.random((ny, nx))).round(3)
    layers = {"huc": huc, "ac": ac, "hand": hand}
    for st in (1, 3):
        for p in _FLOOD_PROBS:
            depth = np.clip((1.0 + 3 * (0.5 - p) + 0.4 * st) - hand, 0, None)
            layers[f"fd_{st}_{int(p * 1000):03d}"] = np.where(depth > 0, depth.round(3), np.nan)
    for st in (1, 4):
        fp = np.clip(0.6 * np.exp(-river / (6.0 + st)) - 0.02, 0, None).round(4)
        layers[f"fp_{st}"] = np.where(fp > 0, fp, np.nan)
    return xr.concat([grid(v) for v in layers.values()], dim="band").assign_coords(band=list(layers))


INPUTS.update({"futures_flood_world": futures_flood_world})


def _flood_files(mode, ddf_regions):
    regions = [30, 10, 20]
    years = list(range(2021, 2021 + _FUT_STEPS))
    files = {"population.csv": _fut_table([(y, *(_FLOOD_POP[r][k] for r in regions)) for k, y in enumerate(years)],
                                          ["year", *map(str, regions)])}
    files["matrix.csv"] = "ID,30,10,20,99\n" + "\n".join(f"{r}," + ",".join(repr(v) for v in _REDIST[r])
                                                          for r in regions) + "\n"
    if ddf_regions:
        rows = [(r, *_DDF["curves"][r]) for r in regions]
    else:
        rows = [(1, *_DDF["curves"][10])]
    files["ddf.csv"] = _fut_table(rows, ["ID", *map(repr, _DDF["levels"])])
    if mode == "depth":
        lines = ["step,probability,map"] + [f"{st},{p!r},fd_{st}_{int(p * 1000):03d}" for st in (1, 3)
                                            for p in _FLOOD_PROBS]
    else:
        lines = ["step,map"] + [f"{st},fp_{st}" for st in (1, 4)]
    files["floods.csv"] = "\n".join(lines) + "\n"
    return files


def futures_flood_job(world, flood, opts):
    regions = [30, 10, 20]
    rasters = {b: world.sel(band=b).drop_vars("band") for b in ("developed", "sub", "slope", "road", "forest")}
    for b in flood.band.values:
        rasters[str(b)] = flood.sel(band=b).drop_vars("band")
    files = {**_fut_files({"patch_columns": True, "regions": regions}), **_flood_files(opts["mode"], opts["ddf_regions"])}
    heredocs = "".join(f"cat > {name} <<'EOF'\n{text}EOF\n" for name, text in files.items())
    resp = ",".join(repr(v) for v in opts["response"])
    args = ["developed=developed subregions=sub predictors=slope,road,forest development_pressure=pressure",
            "n_dev_neighbourhood=8 development_pressure_approach=gravity gamma=1.5 scaling_factor=1.0",
            "devpot_params=potential.csv demand=demand.csv patch_sizes=patches.csv num_neighbors=4",
            f"seed_search=probability compactness_mean=0.4 compactness_range=0.05 discount_factor=1.0 random_seed={opts['seed']}",
            "output=final output_series=step population_demand=population.csv",
            "flood_maps_file=floods.csv huc=huc adaptive_capacity=ac depth_damage_functions=ddf.csv",
            f"response_func={resp} response_stddev={opts['stddev']!r} redistribution_matrix=matrix.csv",
            "redistribution_output=redist flood_logfile=$OUT/floods.txt output_adaptation=adapt"]
    if opts["ddf_regions"]:
        args.append("ddf_subregions=sub")
    if opts["mode"] != "depth":
        args.append(f"hand=hand hand_percentile={opts['percentile']}")
    script = (heredocs + "r.futures.devpressure -n input=developed output=pressure method=gravity size=8 gamma=1.5 "
              "scaling_factor=1.0\nr.futures.simulation_fixed " + " ".join(args) + "\n"
              + "".join(f"r.out.gdal --quiet -c input=step_{k} output=$OUT/step_{k}.tif type=Int32\n"
                        f"r.out.gdal --quiet -c input=adapt_{k} output=$OUT/adapt_{k}.tif type=Int32\n"
                        f"cp redist_{k}.csv $OUT/redist_{k}.csv\n" for k in range(1, _FUT_STEPS + 1)))
    outputs = (["floods.txt"] + [f"step_{k}.tif" for k in range(1, _FUT_STEPS + 1)]
               + [f"adapt_{k}.tif" for k in range(1, _FUT_STEPS + 1)]
               + [f"redist_{k}.csv" for k in range(1, _FUT_STEPS + 1)])
    return {"rasters": rasters, "script": script, "outputs": outputs}


def futures_flood_py(world, flood, opts) -> dict:
    import pandas as pd

    S = _ls().simulate
    regions = [30, 10, 20]
    w = {b: world.sel(band=b).drop_vars("band") for b in world.band.values}
    f = {str(b): flood.sel(band=b).drop_vars("band") for b in flood.band.values}
    years = list(range(2021, 2021 + _FUT_STEPS))
    demand = S.DevelopmentDemand(pd.DataFrame({r: _FUT_DEMAND[r] for r in regions}, index=years),
                                 population=pd.DataFrame({r: _FLOOD_POP[r] for r in regions}, index=years))
    pot = S.DevelopmentPotential(pd.DataFrame(_FUT_POT, columns=["ID", "Intercept", "development_pressure", "slope",
                                                                 "road", "forest"]).set_index("ID"))
    if opts["ddf_regions"]:
        damage = pd.DataFrame([_DDF["curves"][r] for r in regions], index=regions, columns=list(_DDF["levels"]))
    else:
        damage = pd.DataFrame([_DDF["curves"][10]], index=[1], columns=list(_DDF["levels"]))
    kw = {}
    if opts["mode"] == "depth":
        kw["depths"] = {st: {p: f[f"fd_{st}_{int(p * 1000):03d}"] for p in _FLOOD_PROBS} for st in (1, 3)}
    else:
        kw.update(probabilities={st: f[f"fp_{st}"] for st in (1, 4)}, hand=f["hand"], hand_percentile=opts["percentile"])
    flooding = S.Flooding(f["huc"], f["ac"], damage, opts["response"], response_stddev=opts["stddev"],
                          damage_regions="subregions" if opts["ddf_regions"] else None,
                          redistribution=pd.DataFrame(_REDIST, index=[30, 10, 20, 99]).T, **kw)
    alloc = S.PatchGrowing({r: _FUT_PATCHES[r] for r in regions}, subregions=w["sub"], neighborhood=8, climate=flooding)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = S.Model(pot, demand, alloc).run(w["developed"], {k: w[k] for k in ("slope", "road", "forest")},
                                              seed=opts["seed"])
    out = {"floods.txt": [f"{int(s) - 1},{int(h)},{float(p):.4f}" for s, h, p in sim.floods.itertuples(index=False)]}
    code = sim.development.values
    for k in range(1, _FUT_STEPS + 1):
        c = code[k].copy()
        c[c == -1] = np.nan  # with floods, the module writes undeveloped cells as null
        c[c < -1] += 1       # and abandoned ones as -(step)
        out[f"step_{k}.tif"] = c
        out[f"adapt_{k}.tif"] = sim.adaptation.values[k]
        mv = sim.redistribution.sel(step=k).values
        out[f"redist_{k}.csv/ID"] = np.array(regions, dtype=float)
        for j, to in enumerate([30, 10, 20, 99]):
            out[f"redist_{k}.csv/{to}"] = np.array([float(f"{float(v):f}") if v > 0 else float(int(v))
                                                    for v in mv[:, j]])
    return out


def _flood_case(opts, notes):
    return Case("futures", "grass", ("futures_world", "futures_flood_world"),
                job=lambda futures_world, futures_flood_world: futures_flood_job(futures_world, futures_flood_world,
                                                                                 opts),
                py=lambda futures_world, futures_flood_world: futures_flood_py(futures_world, futures_flood_world,
                                                                               opts), notes=notes)


CASES.update({
    "futures_flood_depth": _flood_case(
        {"mode": "depth", "ddf_regions": True, "response": (0.5, 0.35, 0.3, 0.5), "stddev": 0.1, "seed": 2},
        "Floods from depth maps (4 probabilities, steps 1 and 3), damage curves per subregion, Gaussian response, "
        "redistribution with population densities and a destination outside."),
    "futures_flood_hand": _flood_case(
        {"mode": "probability", "ddf_regions": False, "response": (0.4, 0.3, 0.2, 0.4), "stddev": 0.0,
         "percentile": 90, "seed": 9},
        "Floods from flood probability maps and HAND (90th percentile), one damage curve, deterministic response."),
})


# ----------------------------------------------------------------------------- FUTURES: calibration, potential surface

_INT_DEVS = ("r.mapcalc --quiet expression='dev0 = int(dev0f)'\n"
             "r.mapcalc --quiet expression='dev3 = int(dev3f)'\n")


def _calib_rasters(world, single):
    maps = _fut_observed(world)
    base = world.sel(band="developed").drop_vars("band")
    rasters = {"dev0f": base.copy(data=maps[0]), "dev3f": base.copy(data=maps[3])}
    sub = world.sel(band="sub").drop_vars("band")
    rasters["sub"] = sub.copy(data=np.ones(sub.shape)) if single else sub
    for b in ("slope", "road", "forest"):
        rasters[b] = world.sel(band=b).drop_vars("band")
    return rasters


def futures_library_job(world, per_subregion):
    rasters = _calib_rasters(world, False)
    flag = "-ls" if per_subregion else "-l"
    script = ("r.mapcalc --quiet expression='subi = int(sub)'\n" + _INT_DEVS +
              f"r.futures.calib {flag} development_start=dev0 development_end=dev3 subregions=subi "
              "patch_threshold=0 patch_sizes=$OUT/patches.txt nprocs=1\n")
    return {"rasters": rasters, "script": script, "outputs": ["patches.txt"]}


def futures_library_py(world, per_subregion):
    S = _ls().simulate
    r = _calib_rasters(world, False)
    lib = S.patch_library(r["dev0f"], r["dev3f"], subregions=r["sub"], per_subregion=per_subregion)
    if not isinstance(lib, dict):
        return {"patches.txt": [str(v) for v in lib]}
    keys = list(lib)
    n = max(len(v) for v in lib.values())
    lines = [",".join(str(k) for k in keys)]
    lines += [",".join(str(lib[k][i]) if i < len(lib[k]) else "" for k in keys) for i in range(n)]
    return {"patches.txt": lines}


_CALIB = {"means": (0.2, 0.5), "ranges": (0.05, 0.2), "discounts": (0.5, 1.0), "repeat": 2, "seed": 3}


def futures_calib_job(world):
    rasters = _calib_rasters(world, True)
    files = _fut_files({"regions": [1]})
    del files["patches.csv"]
    heredocs = "".join(f"cat > {name} <<'EOF'\n{text}EOF\n" for name, text in files.items())
    c = _CALIB
    script = (heredocs + "r.mapcalc --quiet expression='subi = int(sub)'\n" + _INT_DEVS +
              "r.futures.devpressure -n input=dev0 output=pressure method=gravity size=8 gamma=1.5 "
              "scaling_factor=1.0\n"
              "r.futures.calib development_start=dev0 development_end=dev3 subregions=subi patch_threshold=0 "
              "patch_sizes=patches.txt nprocs=1 "
              f"compactness_mean={','.join(map(str, c['means']))} compactness_range={','.join(map(str, c['ranges']))} "
              f"discount_factor={','.join(map(str, c['discounts']))} repeat={c['repeat']} random_seed={c['seed']} "
              "calibration_results=$OUT/calib.csv development_pressure=pressure predictors=slope,road,forest "
              "n_dev_neighbourhood=8 devpot_params=potential.csv num_neighbors=4 seed_search=probability "
              "development_pressure_approach=gravity gamma=1.5 scaling_factor=1.0 demand=demand.csv\n")
    return {"rasters": rasters, "script": script, "outputs": ["calib.csv"]}


def futures_calib_py(world):
    import pandas as pd

    S = _ls().simulate
    r = _calib_rasters(world, True)
    pot = S.DevelopmentPotential(pd.DataFrame(_FUT_POT, columns=["ID", "Intercept", "development_pressure", "slope",
                                                                 "road", "forest"]).set_index("ID"))
    years = list(range(2021, 2021 + _FUT_STEPS))
    demand = S.DevelopmentDemand(pd.DataFrame({1: _FUT_DEMAND[1]}, index=years))
    model = S.Model(pot, demand, S.PatchGrowing([1], subregions=r["sub"], neighborhood=8))
    c = _CALIB
    t = S.calibrate_patches(model, r["dev0f"], r["dev3f"], {k: r[k] for k in ("slope", "road", "forest")},
                            compactness_means=c["means"], compactness_ranges=c["ranges"],
                            discount_factors=c["discounts"], repeat=c["repeat"], seed=c["seed"])
    return {f"calib.csv/{col}": np.array([float(f"{v:.2f}") for v in t[col]]) for col in t.columns}


def futures_potsurface_job(world):
    rasters = _calib_rasters(world, False)
    rows = [r for r in _FUT_POT if r[0] != 1]
    pot = _fut_table(rows, ["ID", "Intercept", "pressure", "slope", "road", "forest"])
    script = (f"cat > potential.csv <<'EOF'\n{pot}EOF\n"
              "r.mapcalc --quiet expression='subi = int(sub)'\n" + _INT_DEVS +
              "r.futures.devpressure -n input=dev0 output=pressure method=gravity size=8 gamma=1.5 "
              "scaling_factor=1.0\n"
              "r.futures.potsurface input=potential.csv subregions=subi output=pot\n"
              "r.out.gdal --quiet -c input=pot output=$OUT/pot.tif type=Float64\n")
    return {"rasters": rasters, "script": script, "outputs": ["pot.tif"]}


def futures_potsurface_py(world):
    import pandas as pd

    S = _ls().simulate
    r = _calib_rasters(world, False)
    rows = [row for row in _FUT_POT if row[0] != 1]
    pot = S.DevelopmentPotential(pd.DataFrame(rows, columns=["ID", "Intercept", "development_pressure", "slope",
                                                             "road", "forest"]).set_index("ID"))
    pressure = S.development_pressure(r["dev0f"], 8)
    s = pot.surface({k: r[k] for k in ("slope", "road", "forest")}, pressure, r["sub"])
    return {"pot.tif": s.values}


CASES.update({
    "futures_library": Case("futures", "grass", ("futures_world",),
                            job=lambda futures_world: futures_library_job(futures_world, False),
                            py=lambda futures_world: futures_library_py(futures_world, False),
                            notes="r.futures.calib -l: the patch library of the observed development."),
    "futures_library_subregions": Case("futures", "grass", ("futures_world",),
                                       job=lambda futures_world: futures_library_job(futures_world, True),
                                       py=lambda futures_world: futures_library_py(futures_world, True),
                                       notes="r.futures.calib -ls: a patch library per subregion."),
    "futures_calib": Case("futures", "grass", ("futures_world",), job=lambda futures_world: futures_calib_job(futures_world),
                          py=lambda futures_world: futures_calib_py(futures_world),
                          notes="r.futures.calib: 8 combinations x 2 runs, one subregion (the module as released)."),
    "futures_potsurface": Case("futures", "grass", ("futures_world",), job=lambda futures_world: futures_potsurface_job(futures_world),
                               py=lambda futures_world: futures_potsurface_py(futures_world), notes="r.futures.potsurface: the potential surface."),
})


# ----------------------------------------------------------------------------- FUTURES: potential (r.futures.potential)

def futures_samples():
    """2500 sampled cells of 10 subregions with random intercepts and pressure slopes (seed 31)."""
    import pandas as pd

    rng = np.random.default_rng(31)
    G, n = 10, 250
    g = np.repeat(np.arange(1, G + 1) * 7, n)
    x1, x2, x3, pres = rng.normal(size=(4, G * n))
    a = rng.normal(0, 0.8, G)
    b = rng.normal(0, 0.4, G)
    eta = -1.0 + 0.8 * x1 - 0.5 * x2 + (0.6 + b[g // 7 - 1]) * pres + a[g // 7 - 1]
    y = (rng.random(G * n) < 1 / (1 + np.exp(-eta))).astype(np.float64)
    x3[5] = np.nan  # a missing value: the row is dropped
    return pd.DataFrame({"developed": y, "x1": x1.round(5), "x2": x2.round(5), "x3": x3.round(5),
                         "pressure": pres.round(5), "county": g.astype(np.float64)})


INPUTS.update({"futures_samples": futures_samples})

_POT_R = """
suppressMessages({library(lme4); library(MuMIn)})
d <- na.omit(futures_samples)
{body}
"""


def _pot_r(formula, kind, dredge=None):
    if kind == "glm":
        fit = f"m <- glm({formula}, family = binomial, data = d, na.action = 'na.fail')"
        coefs = "cf <- t(as.matrix(coef(m)))"
    else:
        fit = f"m <- glmer({formula}, family = binomial, data = d, na.action = 'na.fail')"
        coefs = "cf <- as.matrix(coef(m)[[1]])"
    body = fit + "\n"
    if dredge:
        body += (f"dd <- dredge(m, rank = 'AIC', fixed = {dredge['fixed']}, m.lim = c({dredge['min']}, {dredge['max']}),"
                 " trace = FALSE)\n"
                 "terms <- apply(dd[, setdiff(names(dd)[2:(ncol(dd) - 5)], '(Intercept)'), drop = FALSE], 1,\n"
                 "               function(r) paste(sort(names(r)[!is.na(r)]), collapse = '+'))\n"
                 "best <- get.models(dd, 1)[[1]]\n"
                 + ("m <- glmer(formula(best), family = binomial, data = d, na.action = 'na.fail')\n"
                    if kind != "glm" else "m <- glm(formula(best), family = binomial, data = d)\n"))
    body += coefs + "\n"
    body += ("cols <- c('(Intercept)', 'pressure', 'x1', 'x2', 'x3')\n"
             "out <- matrix(0, nrow(cf), length(cols)); for (k in seq_along(cols)) if (cols[k] %in% colnames(cf)) "
             "out[, k] <- cf[, cols[k]]\n"
             "res <- list(coef = out, m2ll = -2 * as.numeric(logLik(m)))\n")
    if dredge:
        body += "res$aic <- as.numeric(dd$AIC); res$terms <- unname(terms)\n"
    body += "res\n"
    return _POT_R.replace("{body}", body)


def _pot_py(samples, kind, dredge=None):
    S = _ls().simulate
    d = samples
    if kind == "glm":
        d = samples[samples.county == 7.0]
    kw = {"pressure": "pressure", "subregions": None if kind == "glm" else "county",
          "predictors": ["x1", "x2", "x3"] + ([] if kind == "slope" else ["pressure"])}
    if kind == "slope":
        kw["random"] = "pressure"
    if dredge:
        kw.update(dredge=True, fixed=["pressure"], min_variables=dredge["min"], max_variables=dredge["max"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pot = S.DevelopmentPotential.fit(d, **kw)
    t = pot.table
    out = {"coef": t.reindex(columns=["intercept", "devpressure", "x1", "x2", "x3"]).fillna(0.0).to_numpy(),
           "m2ll": np.array([pot.attrs["deviance"]])}
    if dredge:
        dd = pot.attrs["dredge"]
        out["aic"] = dd["aic"].to_numpy()
        out["terms"] = ["+".join(sorted(t.split("+"))) for t in dd["terms"]]
    return out


def _pot_case(kind, formula, notes, dredge=None):
    r = _pot_r(formula, kind, dredge)
    if kind == "glm":
        r = r.replace("d <- na.omit(futures_samples)", "d <- na.omit(futures_samples); d <- d[d$county == 7, ]")
    return Case("futures", "r", ("futures_samples",), r=r, py=lambda futures_samples: _pot_py(futures_samples, kind,
                                                                                               dredge),
                notes=notes, extra={"tolerance": kind != "glm"})


CASES.update({
    "potential_intercept": _pot_case("intercept", "developed ~ x1 + x2 + x3 + pressure + (1 | county)",
                                     "glmer with a random intercept per subregion."),
    "potential_slope": _pot_case("slope", "developed ~ x1 + x2 + x3 + (pressure | county)",
                                 "glmer with a random pressure slope (no fixed pressure), as r.futures.potential."),
    "potential_dredge": _pot_case("intercept", "developed ~ x1 + x2 + x3 + pressure + (1 | county)",
                                  "MuMIn's dredge by AIC, pressure fixed, 1 to 3 terms.",
                                  dredge={"fixed": "'pressure'", "min": 1, "max": 3}),
    "potential_single": _pot_case("glm", "developed ~ x1 + x2 + x3 + pressure",
                                  "One subregion: R's glm (bit for bit)."),
})


# ----------------------------------------------------------------------------- Expander / Patcher (Dinamica EGO)

GROUPS += ["patches"]

_PATCH_RUNS = 12


def patches_scattered():
    """120 x 120 cells: class 2 scattered at 45% (n_j from 0 to 8 around class-1 cells), a class-1 strip."""
    rng = np.random.default_rng(9)
    t0 = np.where(rng.random((120, 120)) < 0.45, 2.0, 1.0)
    t0[:, :20] = 1
    g = grid(t0)
    g.attrs["categorical"] = True
    return g


def patches_blobs():
    """120 x 120 cells of class 1 with four disks of class 2, and a probability of 0.95 within 5 cells of
    them (a band), 0.5 elsewhere."""
    from scipy import ndimage

    yy, xx = np.mgrid[0:120, 0:120]
    t0 = np.ones((120, 120))
    for a, b in [(30, 30), (80, 40), (50, 90), (95, 95)]:
        t0[(yy - a) ** 2 + (xx - b) ** 2 < 36] = 2
    dist = ndimage.distance_transform_edt(t0 != 2)
    prob = np.where(dist <= 5, 0.95, 0.5)
    g = grid(t0)
    g.attrs["categorical"] = True
    return {"t0": g, "prob": grid(prob)}


INPUTS.update({"patches_scattered": patches_scattered, "patches_blobs": lambda: patches_blobs()["t0"],
               "patches_blobs_prob": lambda: patches_blobs()["prob"]})


def _patch_job(t0, prob, functor, cells, mean_ha=0.09):
    script = ['Script {{', '  t0 := LoadCategoricalMap "t0.tif" .none .default 0;', '  prob := LoadMap "prob.tif";',
              '  changes := LoadTable "changes.csv";', '  params := LoadTable "params.csv";']
    for k in range(_PATCH_RUNS):
        script += [f"  ch{k} co{k} re{k} := {functor} t0 prob changes params;", f'  SaveMap ch{k} "out{k}.tif";']
    script.append("}};")
    heredoc = {"changes.csv": f'"From*","To*","Cells"\n1,2,{cells}\n',
               "params.csv": '"From*","To*","Mean_Patch_Size","Patch_Size_Variance","Patch_Isometry"\n'
                             f"1,2,{mean_ha},0,1\n"}
    return {"rasters": {"t0": t0, "prob": prob}, "files": heredoc, "script": "\n".join(script) + "\n",
            "outputs": [f"out{k}.tif" for k in range(_PATCH_RUNS)],
            "band_names": {"prob": "probability_1_to_2"}}


def _patch_py(t0, prob, functor, cells, runs=40):
    from landschaft import _core

    land = np.nan_to_num(t0.values, nan=-1).astype(np.int32)
    out = {}
    for k in range(runs):
        t = [{"from": 1, "to": 2, "probability": np.nan_to_num(prob.values).ravel().astype(np.float64),
              "expand": cells if functor == "Expander" else 0, "patch": 0 if functor == "Expander" else cells,
              "expander": (1, 0, 1), "patcher": (1, 0, 1)}]
        r = _core.sim_patches(land, t, 10.0, 3, 3, k)
        out[f"out{k}.tif"] = r["landscape"].astype(np.float64)
    return out


CASES.update({
    "patches_expander": Case(
        "patches", "dinamica", ("patches_scattered",),
        job=lambda patches_scattered: _patch_job(patches_scattered, grid(np.full(patches_scattered.shape, 0.9)),
                                                 "Expander", 3000),
        py=lambda patches_scattered: _patch_py(patches_scattered, grid(np.full(patches_scattered.shape, 0.9)),
                                               "Expander", 3000),
        notes="Dinamica EGO's Expander, one-cell patches, 12 calls: the acceptance by n_j (statistical)."),
    "patches_patcher": Case(
        "patches", "dinamica", ("patches_blobs", "patches_blobs_prob"),
        job=lambda patches_blobs, patches_blobs_prob: _patch_job(patches_blobs, patches_blobs_prob, "Patcher", 507),
        py=lambda patches_blobs, patches_blobs_prob: _patch_py(patches_blobs, patches_blobs_prob, "Patcher", 507),
        notes="Dinamica EGO's Patcher, one-cell patches, 12 calls: where the seeds fall (statistical)."),
})


# ============================================================================= phase 7b: MOLUSCE's WoE and simulator

GROUPS += ["molusce"]

_MOL_BINS = {"dist": [90.0, 300.0, 600.0], "slope": [5.0, 12.0]}


def molusce_world(holes: bool = True):
    """Two maps (classes 1, 2, 3; 45 x 60 cells) with change driven by the distance to class 2, a soil type
    (with holes) and a slope; a missing corner in both maps (numpy's generator, seed 23)."""
    import xarray as xr
    from scipy import ndimage

    rng = np.random.default_rng(23)
    ny, nx = 45, 60
    t0 = np.ones((ny, nx))
    t0[5:15, 8:20] = 2
    t0[30:40, 35:50] = 2
    t0[ndimage.gaussian_filter(rng.random((ny, nx)), 2) > 0.53] = 3
    dist = ndimage.distance_transform_edt(t0 != 2) * 30.0
    soil = np.floor(np.linspace(0, 3.999, nx))[None, :] + np.zeros((ny, 1))
    if holes:
        soil[20:24, 10:14] = np.nan
    g = ndimage.gaussian_filter(rng.random((ny, nx)), 3)
    slope = (g - g.min()) / (g.max() - g.min()) * 30
    p = np.exp(-dist / 200) * (0.5 + 0.2 * (soil == 1)) * (1 - slope / 40)
    t1 = t0.copy()
    t1[(t0 == 1) & (rng.random((ny, nx)) < np.nan_to_num(p))] = 2
    t1[(t0 == 3) & (rng.random((ny, nx)) < 0.08)] = 1
    t1[(t0 == 1) & (rng.random((ny, nx)) < 0.02)] = 3
    for a in (t0, t1):
        a[:3, :4] = np.nan
    bands = {"t0": t0, "t1": t1, "dist": dist, "soil": soil, "slope": slope}
    return xr.concat([grid(v) for v in bands.values()], dim="band").assign_coords(band=list(bands))


def molusce_job(world, iterations=3, seed=7) -> Callable:
    def job(import_module) -> dict:
        from numpy import ma

        dp = import_module("molusce.algorithms.dataprovider")
        aa = import_module("molusce.algorithms.models.area_analysis.manager")
        wm = import_module("molusce.algorithms.models.woe.manager")
        ct = import_module("molusce.algorithms.models.crosstabs.manager")
        sim = import_module("molusce.algorithms.models.simulator.sim")
        ny, nx = world.shape[1:]
        geo = {"xSize": nx, "ySize": ny, "proj": "", "transform": (0, 30, 0, 0, 0, -30), "units": "m"}

        def raster(a, dtype):
            r = dp.Raster()
            nan = np.isnan(a)
            r.create([ma.array(np.where(nan, 0, a).astype(dtype), mask=nan)], dict(geo))
            return r

        t0 = raster(world.sel(band="t0").values, np.uint8)
        t1 = raster(world.sel(band="t1").values, np.uint8)
        names = ["dist", "soil", "slope"]
        factors = [raster(world.sel(band=k).values, np.float64) for k in names]
        analyst = aa.AreaAnalyst(t0, t1)
        bins = {i: ([list(_MOL_BINS[k])] if k in _MOL_BINS else None) for i, k in enumerate(names)}
        model = wm.WoeManager(factors, analyst, bins=bins)
        model.train()
        cats = [float(c) for c in analyst.categories]
        m = len(cats)
        out = {}
        for code, per in model.weights.items():
            for k, w in per.items():
                out[f"w_{code}_{names[k]}"] = np.asarray(w[1], dtype=float)
        pred = model.getPrediction(t1).getBand(1)
        conf = model.getConfidence().getBand(1)
        p = ma.filled(pred.astype(float), np.nan)
        out["prediction"] = np.where((p == 255) | np.isnan(p), np.nan, np.asarray(cats)[np.nan_to_num(p).astype(int) % m])
        out["confidence"] = ma.filled(conf.astype(float), np.nan)
        cross = ct.CrossTableManager(t0, t1)
        s = sim.Simulator(t1, factors, model, cross)
        s.setIterationCount(1)
        np.random.seed(seed)
        for k in range(1, iterations + 1):
            s.simN()
            out[f"map{k}"] = ma.filled(s.getState().getBand(1).astype(float), np.nan)
        return out
    return job


def molusce_py(world, iterations=3, seed=7) -> dict:
    import pandas as pd

    import landschaft as ls

    S = ls.simulate
    t0, t1 = world.sel(band="t0"), world.sel(band="t1")
    drivers = {k: world.sel(band=k) for k in ("dist", "soil", "slope")}
    woe = S.MolusceWoE(bins=_MOL_BINS).fit(t0, t1, drivers)
    out = {}
    for code, per in woe.weights.items():
        for k, (_c, w) in per.items():
            out[f"w_{code}_{k}"] = w
    out["prediction"], out["confidence"] = woe.prediction(t1)
    a, b = t0.values.ravel(), t1.values.ravel()
    ok = ~(np.isnan(a) | np.isnan(b))
    cross = pd.crosstab(a[ok], b[ok])
    sim = S.Model(woe, cross, S.MolusceCA(iterations=iterations)).run(t1, seed=seed)
    for k in range(1, iterations + 1):
        out[f"map{k}"] = sim.maps.values[k]
    return out


INPUTS.update({"molusce_world": molusce_world, "molusce_world_full": lambda: molusce_world(holes=False)})

CASES.update({
    "molusce_fixed": Case(
        "molusce", "molusce_fixed", ("molusce_world",), job=lambda molusce_world: molusce_job(molusce_world),
        py=lambda molusce_world: molusce_py(molusce_world),
        notes="MOLUSCE's WoeManager (bins, a categorical factor with holes, persistence codes) and Simulator, "
              "3 iterations, numpy.random.seed(7); tools/molusce_fixed.patch (second largest weight; 255 for no "
              "transition, as numpy < 2 stored the -1)."),
    "molusce_original": Case(
        "molusce", "molusce", ("molusce_world_full",), job=lambda molusce_world_full: molusce_job(molusce_world_full),
        py=lambda molusce_world_full: molusce_py(molusce_world_full),
        notes="MOLUSCE as published, factors without holes (with numpy 2 it fails on a cell without weights): "
              "weights and predictions equal, its confidence (the largest weight before the largest) differs."),
})


# ============================================================================= phase 8a: three-map comparison (lulcc)

GROUPS += ["validation"]

_TMC_FACTORS = (1, 2, 3, 4, 8, 16)


def val_synth():
    """Reference t0, reference t1 and a simulated t1 (classes 1, 2, 3, 4; 37 x 45 cells, sizes that no factor
    divides), change around two fronts, the same missing corner in the three (numpy's generator, seed 41)."""
    import xarray as xr
    from scipy import ndimage

    rng = np.random.default_rng(41)
    ny, nx = 37, 45
    t0 = np.ones((ny, nx))
    t0[ndimage.gaussian_filter(rng.random((ny, nx)), 2) > 0.52] = 2
    t0[:, 30:] = np.where(rng.random((ny, 15)) < 0.7, 3, t0[:, 30:])
    t0[30:, :10] = 4
    t1 = t0.copy()
    yy, xx = np.mgrid[0:ny, 0:nx]
    front = (np.hypot(yy - 10, xx - 12) < 9) & (t0 == 1)
    t1[front & (rng.random((ny, nx)) < 0.6)] = 3
    t1[(t0 == 2) & (rng.random((ny, nx)) < 0.15)] = 1
    t1[(t0 == 3) & (rng.random((ny, nx)) < 0.1)] = 4
    sim = t0.copy()
    near = (np.hypot(yy - 13, xx - 15) < 9) & (t0 == 1)
    sim[near & (rng.random((ny, nx)) < 0.55)] = 3
    sim[(t0 == 2) & (rng.random((ny, nx)) < 0.12)] = 1
    sim[(t0 == 3) & (rng.random((ny, nx)) < 0.1)] = 2
    for a in (t0, t1, sim):
        a[:4, -6:] = np.nan
    return xr.concat([grid(v) for v in (t0, t1, sim)], dim="band").assign_coords(band=["t0", "t1", "sim"])


def val_pie():
    """Plum Island 1985 and 1999 as the references, 1991 standing in for the simulation."""
    p = pie()
    return p.isel(band=[0, 2, 1]).assign_coords(band=["t0", "t1", "sim"])


_TMC_R = """
x <- {inp}_raster
tmc <- ThreeMapComparison(x = x[[1]], x1 = x[[2]], y1 = x[[3]], factors = c({factors}),
                          categories = c({cats}), labels = c({labels}))
ab <- AgreementBudget(tmc)
fom <- FigureOfMerit(tmc)
res <- list()
for (f in seq_along(tmc@factors)) {{
  res[[paste0("table_", tmc@factors[f])]] <- tmc@tables[[f]]
  res[[paste0("fom_category_", tmc@factors[f])]] <- unname(fom@category[[f]])
  res[[paste0("fom_transition_", tmc@factors[f])]] <- unname(fom@transition[[f]])
}}
res$fom_overall <- unlist(fom@overall)
res$budget_overall <- as.matrix(ab@overall)
for (k in seq_along(ab@category)) res[[paste0("budget_category_", k)]] <- as.matrix(ab@category[[k]])
for (k in seq_along(ab@transition)) res[[paste0("budget_transition_", k)]] <- as.matrix(ab@transition[[k]])
res
"""


def _tmc_r(inp, cats, factors):
    return _TMC_R.format(inp=inp, factors=", ".join(map(str, factors)), cats=", ".join(map(str, cats)),
                         labels=", ".join(f'"c{c}"' for c in cats))


def _tmc_py(maps, cats, factors) -> dict:
    ls = _ls()
    tmc = ls.simulate.three_map_comparison(maps[0], maps[1], maps[2], factors=factors, categories=cats)
    fom = tmc.figure_of_merit()
    ab = tmc.agreement_budget()
    budget = ["misses", "hits", "wrong_hits", "false_alarms", "correct_rejections"]
    out = {}
    for f in factors:
        out[f"table_{f}"] = tmc.tables[f]
        out[f"fom_category_{f}"] = fom["category"].loc[f].to_numpy()
        out[f"fom_transition_{f}"] = fom["transition"][f].to_numpy()
    out["fom_overall"] = fom["overall"].to_numpy()
    out["budget_overall"] = ab[ab.level == "overall"][budget].to_numpy()
    for k, c in enumerate(cats, start=1):
        out[f"budget_category_{k}"] = ab[(ab.level == "category") & (ab["from"] == c)][budget].to_numpy()
    k = 0
    for c in cats:
        for d in cats:
            k += 1
            sub = ab[(ab.level == "transition") & (ab["from"] == c) & (ab["to"] == d)]
            out[f"budget_transition_{k}"] = sub[budget[:4]].to_numpy()
    return out


INPUTS.update({"val_synth": val_synth, "val_pie": val_pie})

CASES.update({
    "tmc_synth": Case("validation", "r", ("val_synth",), r=_tmc_r("val_synth", (1, 2, 3, 4), _TMC_FACTORS),
                      py=lambda val_synth: _tmc_py(val_synth, (1, 2, 3, 4), _TMC_FACTORS),
                      notes="lulcc's ThreeMapComparison, AgreementBudget and FigureOfMerit: 4 classes, 6 "
                            "resolutions, blocks cut at the edges, a missing corner."),
    "tmc_pie": Case("validation", "r", ("val_pie",), r=_tmc_r("val_pie", (1, 2, 3), (1, 2, 4, 8, 16, 32)),
                    py=lambda val_pie: _tmc_py(val_pie, (1, 2, 3), (1, 2, 4, 8, 16, 32)),
                    notes="Plum Island 1985 / 1999 with 1991 as the simulation."),
})


# ----------------------------------------------------------------------------- MOLUSCE: kappa, dependence, error budget


def molusce_validation_job(maps) -> Callable:
    def job(import_module) -> dict:
        from numpy import ma

        dp = import_module("molusce.algorithms.dataprovider")
        cm = import_module("molusce.algorithms.models.correlation.model")
        eb = import_module("molusce.algorithms.models.errorbudget.ebmodel")
        ref, sim = maps.sel(band="t1").values, maps.sel(band="sim").values
        ny, nx = ref.shape
        geo = {"xSize": nx, "ySize": ny, "proj": "", "transform": (0, 30, 0, 0, 0, -30), "units": "m"}

        def raster(a):
            r = dp.Raster()
            nan = np.isnan(a)
            r.create([ma.array(np.where(nan, 0, a).astype(np.uint8), mask=nan)], dict(geo))
            return r

        rr, sr = raster(ref), raster(sim)
        dep = cm.DependenceCoef(rr.getBand(1), sr.getBand(1), expand=True)
        k = dep.kappa(mode="all")
        plain = cm.DependenceCoef(rr.getBand(1), sr.getBand(1))
        out = {"coefficients": np.array([plain.correlation(), k["overal"], k["loc"], k["histo"],
                                         dep.correctness(), plain.cramer(), plain.jiu()], dtype=float)}
        stat = eb.EBudget(rr, sr).getStat(4, 2)
        out["budget"] = np.array([[stat[i][key] for key in ("NoNo", "NoMed", "MedMed", "MedPer", "PerPer")]
                                  for i in sorted(stat)], dtype=float)
        return out
    return job


def molusce_validation_py(maps) -> dict:
    ls = _ls()
    ref, sim = maps.sel(band="t1"), maps.sel(band="sim")
    c = ls.simulate.compare_maps(ref, sim)
    eb = ls.simulate.error_budget(ref, sim, iterations=4, scale=2)
    keys = ["correlation", "kappa", "kappa_location", "kappa_histogram", "correctness", "cramer", "jiu"]
    return {"coefficients": np.array([c[k] for k in keys]),
            "budget": eb[["no_no", "no_medium", "medium_medium", "medium_perfect", "perfect_perfect"]].to_numpy()}


CASES.update({
    "molusce_validation_synth": Case(
        "validation", "molusce", ("val_synth",), job=lambda val_synth: molusce_validation_job(val_synth),
        py=lambda val_synth: molusce_validation_py(val_synth),
        notes="MOLUSCE's DependenceCoef (correlation, the three kappas, correctness, Cramer, JIU) and EBudget "
              "(4 resolutions) of the synthetic reference and simulation."),
    "molusce_validation_pie": Case(
        "validation", "molusce", ("val_pie",), job=lambda val_pie: molusce_validation_job(val_pie),
        py=lambda val_pie: molusce_validation_py(val_pie),
        notes="The same on Plum Island (1999 against 1991)."),
})


# ----------------------------------------------------------------------------- r.futures.validation

GROUPS += ["futures_validation"]

_FUTVAL_KEYS = ["total_quantity", "total_allocation", "kappa", "kappasimulation", "misses", "hits",
                "false_alarms", "null_successes", "figure_of_merit", "producer", "user", "initially_developed"]

_FUTVAL_SCRIPT = r"""
for m in __MAPS__; do r.null map=$m setnull=-1; done
python3 - <<'PYEOF'
import builtins, csv, importlib.machinery, importlib.util, os
builtins.__dict__.setdefault("_", lambda s: s)
path = os.path.expanduser("~/.grass8/addons/scripts/r.futures.validation")
loader = importlib.machinery.SourceFileLoader("rfv", path)
spec = importlib.util.spec_from_loader("rfv", loader)
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)
cats, res = mod.compute("reference", "simulated", __ORIGINAL__)
rows = [("cat_" + str(c), float(c)) for c in cats]
for i, c in enumerate(cats):
    rows.append(("quantity_class_" + str(c), float(res["quantity"][i])))
    rows.append(("allocation_class_" + str(c), float(res["allocation"][i])))
for k in __KEYS__:
    v = res.get(k if k != "kappasimulation" else "kappasim")
    rows.append((k, float("nan") if v is None else float(v)))
with open(os.path.join(os.environ["OUT"], "results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["key", "value"])
    for k, v in rows:
        w.writerow([k, repr(v)])
PYEOF
"""


def futval_world(classes: int = 2):
    """Original, reference and simulated maps (0/1 developed, or classes 0-3), 50 x 60, holes (seed 51)."""
    import xarray as xr

    rng = np.random.default_rng(51 + classes)
    ny, nx = 50, 60
    if classes == 2:
        orig = (rng.random((ny, nx)) < 0.2).astype(float)
        ref = np.where(orig == 1, 1.0, (rng.random((ny, nx)) < 0.15).astype(float))
        sim = np.where(orig == 1, 1.0, (rng.random((ny, nx)) < 0.12).astype(float))
        sim[(ref == 1) & (orig == 0) & (rng.random((ny, nx)) < 0.4)] = 1
    else:
        orig = rng.integers(0, classes, (ny, nx)).astype(float)
        ref = np.where(rng.random((ny, nx)) < 0.2, rng.integers(0, classes, (ny, nx)), orig).astype(float)
        sim = np.where(rng.random((ny, nx)) < 0.2, rng.integers(0, classes, (ny, nx)), orig).astype(float)
    orig[:5, :5] = np.nan
    ref[40:43, 10:20] = np.nan
    sim[0, :] = np.nan
    return xr.concat([grid(v) for v in (orig, ref, sim)], dim="band").assign_coords(
        band=["original", "reference", "simulated"])


def _futval_job(world, use_original=True):
    import xarray as xr  # noqa: F401

    names = ["reference", "simulated"] + (["original"] if use_original else [])
    rasters = {n: world.sel(band=n).fillna(-1).astype("int32") for n in names}
    script = (_FUTVAL_SCRIPT.replace("__MAPS__", " ".join(names))
              .replace("__ORIGINAL__", '"original"' if use_original else "None")
              .replace("__KEYS__", repr(_FUTVAL_KEYS)))
    return {"rasters": rasters, "script": script, "outputs": ["results.csv"]}


def _futval_py(world, use_original=True) -> dict:
    ls = _ls()
    r = ls.simulate.futures_validation(world.sel(band="reference"), world.sel(band="simulated"),
                                       world.sel(band="original") if use_original else None)
    keys, values = [], []
    for c in r["classes"]:
        keys.append(f"cat_{c}")
        values.append(float(c))
    for i, c in enumerate(r["classes"]):
        keys += [f"quantity_class_{c}", f"allocation_class_{c}"]
        values += [float(r["quantity"][i]), float(r["allocation"][i])]
    for k in _FUTVAL_KEYS:
        v = r.get(k)
        keys.append(k)
        values.append(float("nan") if v is None else float(v))
    return {"results.csv/key": keys, "results.csv/value": np.array(values)}


INPUTS.update({"futval_binary": futval_world, "futval_classes": lambda: futval_world(4)})

CASES.update({
    "futval_binary": Case("futures_validation", "grass", ("futval_binary",),
                          job=lambda futval_binary: _futval_job(futval_binary),
                          py=lambda futval_binary: _futval_py(futval_binary),
                          notes="r.futures.validation's compute(): developed 0/1 with the original map (hits, "
                                "misses, figure of merit, kappa simulation), holes in each map."),
    "futval_classes": Case("futures_validation", "grass", ("futval_classes",),
                           job=lambda futval_classes: _futval_job(futval_classes),
                           py=lambda futval_classes: _futval_py(futval_classes),
                           notes="Four classes with the original: quantity, allocation, kappa, kappa simulation."),
    "futval_no_original": Case("futures_validation", "grass", ("futval_classes",),
                               job=lambda futval_classes: _futval_job(futval_classes, False),
                               py=lambda futval_classes: _futval_py(futval_classes, False),
                               notes="Without the original: quantity, allocation and kappa only."),
})


# ============================================================================= phase 8b: fuzzy similarity (Dinamica EGO)

GROUPS += ["similarity"]

_SIM_WINDOWS = (1, 3, 5, 7, 9, 11)


def sim_world():
    """t0, observed and simulated maps (classes 1-4, 60 x 80) with change in patches near each other and a
    missing block (numpy's generator, seed 61)."""
    import xarray as xr
    from scipy import ndimage

    rng = np.random.default_rng(61)
    ny, nx = 60, 80
    t0 = 1 + (ndimage.gaussian_filter(rng.random((ny, nx)), 3) * 40).astype(int) % 4
    t0 = t0.astype(float)
    obs, sim = t0.copy(), t0.copy()
    blobs = ndimage.gaussian_filter(rng.random((ny, nx)), 2)
    obs[(blobs > np.quantile(blobs, 0.9)) & (t0 != 2)] = 2
    shifted = np.roll(blobs, (2, 3), axis=(0, 1)) + 0.01 * rng.random((ny, nx))
    sim[(shifted > np.quantile(shifted, 0.88)) & (t0 != 2)] = 2
    obs[(t0 == 3) & (rng.random((ny, nx)) < 0.03)] = 4
    sim[(t0 == 3) & (rng.random((ny, nx)) < 0.03)] = 4
    for m in (t0, obs, sim):
        m[50:, :12] = np.nan
    return xr.concat([grid(v) for v in (t0, obs, sim)], dim="band").assign_coords(band=["t0", "observed",
                                                                                      "simulated"])


def _sim_job(world):
    rasters = {k: world.sel(band=k).fillna(255).astype("int32").rio.write_nodata(255)
               for k in ("t0", "observed", "simulated")}
    lines = ['  t0 := LoadCategoricalMap "t0.tif" 255 .default 0;',
             '  observed := LoadCategoricalMap "observed.tif" 255 .default 0;',
             '  simulated := LoadCategoricalMap "simulated.tif" 255 .default 0;']
    keys, outputs = [], []
    for decay in (".yes", ".no"):
        tag = "exp" if decay == ".yes" else "const"
        for w in _SIM_WINDOWS:
            lines.append(f"  f{tag}{w} mf{tag}{w} s{tag}{w} ms{tag}{w} := CalcReciprocalSimilarityMap observed "
                         f"simulated {w} {decay} .float32 -1;")
            lines.append(f'  SaveMap f{tag}{w} "first_{tag}_{w}.tif";')
            lines.append(f'  SaveMap s{tag}{w} "second_{tag}_{w}.tif";')
            lines.append(f"  d{tag}{w} v{tag}{w} := CalcSimilarityOfDifferences t0 observed simulated {decay} {w};")
            lines.append(f'  SaveMap d{tag}{w} "differences_{tag}_{w}.tif";')
            outputs += [f"first_{tag}_{w}.tif", f"second_{tag}_{w}.tif", f"differences_{tag}_{w}.tif"]
            keys += [f"mf{tag}{w}", f"ms{tag}{w}", f"v{tag}{w}"]
    lines.append("  t := CreateTable [\n\"Key*\",\"Value\"\n" + "\n".join(f"{k + 1},0" for k in range(len(keys)))
                 + "\n];")
    prev = "t"
    for k, name in enumerate(keys):
        lines.append(f"  t{k} := SetTableCellValue {prev} 2 [ {k + 1} ] {name};")
        prev = f"t{k}"
    lines.append(f'  SaveTable {prev} "means.csv";')
    script = "Script {{\n" + "\n".join(lines) + "\n}};\n"
    return {"rasters": rasters, "script": script, "outputs": [*outputs, "means.csv"]}


def _sim_py(world) -> dict:
    ls = _ls()
    t0, obs, sim = (world.sel(band=k) for k in ("t0", "observed", "simulated"))
    out, means = {}, []
    for decay in ("exponential", "constant"):
        tag = "exp" if decay == "exponential" else "const"
        table = ls.simulate.similarity_of_differences(t0, obs, sim, windows=_SIM_WINDOWS, decay=decay)
        for k, w in enumerate(_SIM_WINDOWS):
            s = ls.simulate.fuzzy_similarity(obs, sim, window=w, decay=decay)
            out[f"first_{tag}_{w}.tif"] = np.asarray(s.first)
            out[f"second_{tag}_{w}.tif"] = np.asarray(s.second)
            d = ls.simulate.fuzzy_similarity(ls.simulate.differences(t0, sim), ls.simulate.differences(t0, obs),
                                             window=w, decay=decay)
            row = table.iloc[k]
            out[f"differences_{tag}_{w}.tif"] = np.asarray(d.first if d.first_mean <= d.second_mean else d.second)
            means += [s.first_mean, s.second_mean, row["minimum"]]
    out["means.csv/Key*"] = np.arange(1.0, len(means) + 1)
    out["means.csv/Value"] = np.array(means)
    return out


INPUTS.update({"sim_world": sim_world})

CASES.update({
    "similarity_dinamica": Case(
        "similarity", "dinamica", ("sim_world",), job=lambda sim_world: _sim_job(sim_world),
        py=lambda sim_world: _sim_py(sim_world),
        notes="Dinamica EGO's CalcReciprocalSimilarityMap (observed against simulated) and "
              "CalcSimilarityOfDifferences, windows 1-11, exponential and constant decay, a missing block."),
})


# ============================================================================= Dinamica EGO's other functors

GROUPS += ["gravity"]

_GRAVITY_EXPONENTS = (2.0, 1.0, 1.5, 0.5, 3.0)


@cache
def gravity_world():
    """Masses on 300 cells of a 60 x 80 grid and a mask with a missing block."""
    import xarray as xr

    rng = np.random.default_rng(71)
    ny, nx = 60, 80
    src = np.full((ny, nx), np.nan)
    idx = rng.choice(ny * nx, 300, replace=False)
    src.flat[idx] = rng.integers(1, 1000, 300)
    mask = np.ones((ny, nx))
    mask[5:20, 30:50] = np.nan
    return xr.concat([grid(src), grid(mask)], dim="band").assign_coords(band=["sources", "mask"])


def _gravity_job(world):
    # NaN = no value: the harness writes float maps with -9999 as no data
    src, mask = world.sel(band="sources"), world.sel(band="mask")
    lines = ['  s := LoadMap "sources.tif";', '  m := LoadMap "mask.tif";']
    outputs = []
    for k, beta in enumerate(_GRAVITY_EXPONENTS):
        for tag, m in (("all", ".none"), ("masked", "m")):
            lines += [f"  p{tag}{k} := CalcInteractionPotentialMap s {m} {beta} .float32 .default;",
                      f'  SaveMap p{tag}{k} "{tag}_{beta:g}.tif";']
            outputs.append(f"{tag}_{beta:g}.tif")
    return {"rasters": {"sources": src, "mask": mask}, "script": "Script {{\n" + "\n".join(lines) + "\n}};\n",
            "outputs": outputs}


def _gravity_py(world) -> dict:
    ls = _ls()
    src, mask = world.sel(band="sources"), world.sel(band="mask")
    out = {}
    for beta in _GRAVITY_EXPONENTS:
        out[f"all_{beta:g}.tif"] = np.asarray(ls.surface.gravity(src, beta, method="direct"))
        out[f"masked_{beta:g}.tif"] = np.asarray(ls.surface.gravity(src, beta, mask=mask, method="direct"))
    return out


INPUTS.update({"gravity_world": gravity_world})

CASES.update({
    "gravity_dinamica": Case(
        "gravity", "dinamica", ("gravity_world",), job=lambda gravity_world: _gravity_job(gravity_world),
        py=lambda gravity_world: _gravity_py(gravity_world),
        notes="Dinamica EGO's CalcInteractionPotentialMap (float32 output), five exponents, with and without "
              "a mask; Dinamica sums float32 terms in float32."),
})


GROUPS += ["continuous_patcher"]

# (quota, prune factor): the ranked list kept whole or cut, met exactly, exceeded, left unmet
_PCS_DETERMINISTIC = ((95.0, 1.0), (45.0, 1.0), (2000.0, 10.0), (880.0, 1.5))
_PCS_DRAWS = 32


@cache
def pcs_world():
    """Stock (100 per hectare on 30 m cells: 9 per cell) and probabilities for the continuous Patcher."""
    import xarray as xr

    rng = np.random.default_rng(81)
    ny, nx = 20, 20
    stock = np.full((ny, nx), 100.0)
    stock[:, :15] = 0.0  # 100 cells with stock, 900 units
    prob = rng.random((ny, nx))
    prob[0, 19] = 0.0  # never a candidate
    draws = np.full((30, 40), 100.0)
    dprob = np.full(30 * 40, 0.001)  # below the 100 cells the 900 units of the pruned list hold
    dprob[rng.choice(30 * 40, 100, replace=False)] = np.linspace(0.01, 1.0, 100)
    dprob = dprob.reshape(30, 40)
    small = [grid(stock), grid(prob)]
    big = [grid(draws), grid(dprob)]
    return {"stock": small[0], "prob": small[1], "draw_stock": big[0], "draw_prob": big[1]}


def _pcs_job(world):
    lines = ['  s := LoadMap "stock.tif";', '  p := LoadMap "prob.tif";', '  ds := LoadMap "draw_stock.tif";',
             '  dp := LoadMap "draw_prob.tif";']
    outputs, names = [], []
    for k, (q, f) in enumerate(_PCS_DETERMINISTIC):
        lines += [f"  m{k} mp{k} r{k} := PatcherContinuousState s p {q} 0.09 0 1.5 3 3 {f};",
                  f'  SaveMap m{k} "det_{k}.tif";']
        outputs.append(f"det_{k}.tif")
        names.append(f"r{k}")
    for k in range(_PCS_DRAWS):
        lines += [f"  d{k} dq{k} dr{k} := PatcherContinuousState ds dp 180 0.09 0 1.5 3 3 5;",
                  f'  SaveMap d{k} "draw_{k}.tif";']
        outputs.append(f"draw_{k}.tif")
    lines.append("  t := CreateTable [\n\"Key*\",\"Value\"\n" + "\n".join(f"{k + 1},0" for k in range(len(names)))
                 + "\n];")
    prev = "t"
    for k, name in enumerate(names):
        lines.append(f"  t{k} := SetTableCellValue {prev} 2 [ {k + 1} ] {name};")
        prev = f"t{k}"
    lines.append(f'  SaveTable {prev} "remaining.csv";')
    rasters = {k: world[k] for k in ("stock", "prob", "draw_stock", "draw_prob")}
    return {"rasters": rasters, "script": "Script {{\n" + "\n".join(lines) + "\n}};\n",
            "outputs": [*outputs, "remaining.csv"]}


INPUTS.update({"pcs_world": pcs_world})

CASES.update({
    "continuous_patcher_dinamica": Case(
        "continuous_patcher", "dinamica", ("pcs_world",), job=lambda pcs_world: _pcs_job(pcs_world),
        notes="Dinamica EGO's PatcherContinuousState with one-cell patches (larger ones crash it): four quotas "
              "and prune factors whose outcome does not depend on the draw, and 32 draws of 20 cells among 100 "
              "of probability 0.01-1 for the seed statistics."),
})
