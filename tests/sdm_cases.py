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

GROUPS: list[str] = ["smoke", "cells", "thin", "clean", "predictors", "background", "folds", "evaluate", "extrapolation"]


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


def cc_synthetic() -> pd.DataFrame:
    """Records near each kind of reference CoordinateCleaner tests against, and controls."""
    from pyproj import Geod

    from landschaft.sdm._refdata import reference_data

    g = Geod(ellps="WGS84")
    cref = reference_data("countryref")
    cap = cref[cref["capital"] == "Brasilia"].iloc[0]
    cen = cref[(cref["type"] == "country") & (cref["iso3"] == "PER")].iloc[0]
    prov = cref[cref["type"] == "province"].iloc[100]
    inst = reference_data("institutions").dropna(subset=["decimalLongitude"]).iloc[50]

    def at(lon, lat, dist, az=37.0):
        x, y, _ = g.fwd(lon, lat, az, dist)
        return float(x), float(y)

    rows = []  # lon, lat, species, iso3
    for d in (500.0, 9_000.0, 11_000.0, 50_000.0):
        rows.append((*at(cap["capital.lon"], cap["capital.lat"], d), "cap", "BRA"))
    for d in (300.0, 1_500.0):
        rows.append((*at(cen["centroid.lon"], cen["centroid.lat"], d), "cen", "PER"))
        rows.append((*at(prov["centroid.lon"], prov["centroid.lat"], d), "cen", prov["iso3"]))
    for d in (200.0, 2_000.0):
        rows.append((*at(12.58, 55.67, d), "gbif", "DNK"))
    for d in (50.0, 150.0):
        rows.append((*at(inst["decimalLongitude"], inst["decimalLatitude"], d), "inst", "XXX"))
    rows += [(10.0, 10.0, "equal", "NGA"), (-7.5, 7.5, "equal", "CIV"), (0.3, 0.2, "zero", "XXX"),
             (0.0, 15.0, "zero", "MLI"), (-30.0, -20.0, "sea", "BRA"), (2.35, 48.85, "fra", "FRA"),
             (10.75, 59.91, "nor", "NOR"), (-46.63, -23.55, "urban", "BRA"), (-43.2, -22.9, "urban", "DEU")]
    rng = np.random.default_rng(42)
    for i in range(20):  # a species around Manaus and one record far away
        rows.append((-60.0 + rng.normal(0, 0.5), -3.1 + rng.normal(0, 0.5), "spA", "BRA"))
    rows.append((-40.0, 5.0, "spA", "BRA"))
    rows += [(-60.0 + rng.normal(0, 0.3), -10.0 + rng.normal(0, 0.3), "spB", "BRA") for _ in range(9)]
    rows += [rows[-1], rows[-1]]  # duplicates
    d = pd.DataFrame(rows, columns=["lon", "lat", "species", "iso3"])
    return d


def cc_edges() -> pd.DataFrame:
    """Records on the vertices of the circles, near the antimeridian and near the south pole."""
    from landschaft import _core
    from landschaft.sdm._refdata import reference_data

    cref = reference_data("countryref")
    cap = cref[cref["capital"] == "Brasilia"].iloc[0]
    cx, cy = float(cap["capital.lon"]), float(cap["capital.lat"])
    ring = _core.sdm_circles(np.array([cx]), np.array([cy]), 10_000.0, "geosphere")[0][0]
    terra = _core.sdm_circles(np.array([cx]), np.array([cy]), 10_000.0, "terra")[0][0]
    rows = [(cx, cy), tuple(ring[0]), tuple(ring[18]), tuple(ring[1]), tuple(terra[10]), tuple(terra[5]),
            ((ring[0][0] + ring[1][0]) / 2, (ring[0][1] + ring[1][1]) / 2),
            (179.99, -16.5), (-179.995, -16.5), (0.0, -16.5), (100.0, -16.52), (-100.0, -16.48),
            (0.0, -89.995), (45.0, -89.9999), (120.0, -89.5)]
    return pd.DataFrame(rows, columns=["lon", "lat"]).assign(species="sp")


def acaule_valid() -> pd.DataFrame:
    """The acaule records with coordinates (CoordinateCleaner stops on invalid ones)."""
    d = acaule()
    return d[d["lon"].notna() & d["lat"].notna()].reset_index(drop=True)


def vif_table() -> pd.DataFrame:
    """usdm's test table: set.seed(1); a, b = a + noise, c, d = b + c + noise (n = 200)."""
    from landschaft import _core

    r = _core.RRandom(1)
    n = 200
    a = r.rnorm(n, 0, 1)
    b = a + r.rnorm(n, 0, 0.3)
    c = r.rnorm(n, 0, 1)
    d = b + c + r.rnorm(n, 0, 0.1)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


def vif_collinear() -> pd.DataFrame:
    t = vif_table()
    return pd.DataFrame({"a": t["a"], "b": t["a"], "c": t["c"]})


def vif_big() -> pd.DataFrame:
    """8000 rows (sampled: 5000) with a missing value."""
    from landschaft import _core

    r = _core.RRandom(5)
    n = 8000
    x = r.rnorm(n, 0, 1)
    y = 0.5 * x + r.rnorm(n, 0, 1)
    z = r.runif(n, 0, 1)
    d = pd.DataFrame({"x": x, "y": y, "z": z})
    d.loc[17, "z"] = np.nan
    return d


def planar_stack(ny: int, nx: int, seed: int, na: int = 0):
    """Three correlated layers on a projected grid (UTM 23S), some cells missing."""
    import xarray as xr

    from landschaft import _core

    r = _core.RRandom(seed)
    n = ny * nx
    a = r.rnorm(n, 0, 1)
    b = a + r.rnorm(n, 0, 0.5)
    c = r.runif(n, 0, 1) + 0.2 * b
    v = np.stack([a, b, c]).reshape(3, ny, nx)
    if na:
        idx = r.sample_int(n, na, False, None) - 1
        v[0].reshape(-1)[idx] = np.nan
    da = xr.DataArray(v, dims=("band", "y", "x"),
                      coords={"band": ["a", "b", "c"], "y": 7_500_000.0 - (np.arange(ny) + 0.5) * 1000.0,
                              "x": 300_000.0 + (np.arange(nx) + 0.5) * 1000.0})
    return da.rio.write_crs("EPSG:32723")


def bradypus_area():
    """The accessible area of the Bradypus records: 500 km around them."""
    import landschaft as ls

    return ls.sdm.accessible_area(bradypus(), distance=500_000)


def bias_bio12():
    """A bias surface: annual precipitation (positive) on the bioclim grid."""
    return bioclim().sel(band="bio12").drop_vars("band")


def blk_points():
    """150 points in 10 x 7 km (UTM 22S), 40 % presences: blockCV's typical input."""
    import geopandas as gpd

    from landschaft import _core

    r = _core.RRandom(21)
    n = 150
    x = r.runif(n, 500_000.0, 510_000.0)
    y = r.runif(n, 7_400_000.0, 7_407_000.0)
    occ = (r.runif(n, 0.0, 1.0) < 0.4).astype(float)
    return gpd.GeoDataFrame({"occ": occ}, geometry=gpd.points_from_xy(x, y), crs="EPSG:32722")


def blk_grid():
    """Points on a 1 km lattice (0..10 x 0..8 km): many lie on block edges."""
    import geopandas as gpd

    xs, ys = np.meshgrid(np.arange(11.0), np.arange(9.0))
    x = 500_000.0 + 1000.0 * xs.ravel()
    y = 7_400_000.0 + 1000.0 * ys.ravel()
    occ = ((xs.ravel() + 2 * ys.ravel()) % 3 == 0).astype(float)
    return gpd.GeoDataFrame({"occ": occ}, geometry=gpd.points_from_xy(x, y), crs="EPSG:32722")


def _bradypus_data():
    import landschaft as ls

    env = bioclim()
    occ = bradypus()
    bg = ls.sdm.background(env, 600, occ=occ, seed=1)
    return ls.sdm.dataset(occ, bg, env)


def brad_occs() -> pd.DataFrame:
    d = _bradypus_data()
    g = d.coords[d.y == 1]
    return pd.DataFrame({"lon": g.geometry.x.to_numpy(), "lat": g.geometry.y.to_numpy()})


def brad_bg() -> pd.DataFrame:
    d = _bradypus_data()
    g = d.coords[d.y == 0]
    return pd.DataFrame({"lon": g.geometry.x.to_numpy(), "lat": g.geometry.y.to_numpy()})


def brad_points():
    """The Bradypus presences and background as points with occ (longitude / latitude)."""
    import geopandas as gpd

    d = _bradypus_data()
    return gpd.GeoDataFrame({"occ": d.y.astype(float)}, geometry=d.coords.geometry.to_numpy(), crs="EPSG:4326")


def eval_small() -> pd.DataFrame:
    """predicts' example of ties: 7 presences, 8 background values."""
    p = [0.9, 0.8, 0.8, 0.7, 0.6, 0.55, 0.3]
    a = [0.8, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1, 0.05]
    return pd.DataFrame({"pred": p + a, "y": [1.0] * len(p) + [0.0] * len(a)})


def eval_beta() -> pd.DataFrame:
    """200 uniform background predictions (3 decimals) and 30 presences skewed high (Beta(4, 1.5)-like)."""
    from landschaft import _core

    r = _core.RRandom(1)
    fit = np.round(r.runif(200, 0.0, 1.0), 3)
    u = r.runif(30, 0.0, 1.0)
    obs = np.round(1 - (1 - u) ** 0.6, 3)  # high values more likely
    return pd.DataFrame({"pred": np.concatenate([obs, fit]), "y": [1.0] * 30 + [0.0] * 200})


def eval_big() -> pd.DataFrame:
    """1500 presences and 1200 background predictions: quantile thresholds."""
    from landschaft import _core

    r = _core.RRandom(7)
    p = np.round(r.runif(1500, 0.2, 1.0), 4)
    a = np.round(r.runif(1200, 0.0, 0.8), 4)
    return pd.DataFrame({"pred": np.concatenate([p, a]), "y": [1.0] * 1500 + [0.0] * 1200})


def eval_scale1000() -> pd.DataFrame:
    d = eval_beta()
    d["pred"] = d["pred"] * 1000
    return d


def ref_x() -> pd.DataFrame:
    """The predictor values of the Bradypus presences and background (the reference conditions)."""
    return _bradypus_data().X.copy()


def env_warm():
    """bioclim + 30: 3 degrees warmer and 30 mm wetter (temperatures in tenths of a degree)."""
    return bioclim() + 30.0


def env_small():
    """A 40 x 40 window of bioclim."""
    return bioclim().isel(y=slice(80, 120), x=slice(80, 120))


def tg_small() -> pd.DataFrame:
    """The complete cells of env_small, three variables, as a table."""
    e = env_small().sel(band=["bio1", "bio12", "bio16"])
    v = e.values.reshape(3, -1).T
    d = pd.DataFrame(v, columns=["bio1", "bio12", "bio16"]).dropna().reset_index(drop=True)
    d.loc[0, "bio1"] = 400.0  # a cell outside the range of bio1
    return d


INPUTS: dict[str, Callable[[], object]] = {
    "ref_x": ref_x,
    "env_warm": env_warm,
    "env_small": env_small,
    "tg_small": tg_small,
    "eval_small": eval_small,
    "eval_beta": eval_beta,
    "eval_big": eval_big,
    "eval_scale1000": eval_scale1000,
    "cc_edges": cc_edges,
    "blk_points": blk_points,
    "blk_grid": blk_grid,
    "brad_occs": brad_occs,
    "brad_bg": brad_bg,
    "brad_points": brad_points,
    "bradypus_area": bradypus_area,
    "bias_bio12": bias_bio12,
    "vif_table": vif_table,
    "vif_collinear": vif_collinear,
    "vif_big": vif_big,
    "stack_80": lambda: planar_stack(80, 80, 3),
    "stack_200": lambda: planar_stack(200, 200, 4, na=3000),
    "cc_synthetic": cc_synthetic,
    "acaule_valid": acaule_valid,
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


# CoordinateCleaner, test by test: the original functions where they are right, tools/cc_fixed.R where
# they err (cc_zero's radius, cc_coun's alignment and code column, cc_outl's empty and raster cases).
_CC_R = """{{ suppressMessages(library(CoordinateCleaner)); source(file.path(TOOLS, "cc_fixed.R"))
x <- {p}; x$decimalLongitude <- x$lon; x$decimalLatitude <- x$lat
otl <- function(method = "quantile", mltpl = 5, tdi = 1000, thinning = FALSE, thinning_res = 0.5) {{
  tab <- table(x$species); tab <- tab[tab > 7]
  ot <- x[x$species %in% names(tab), c("species", "decimalLongitude", "decimalLatitude")]
  ids <- if (nrow(ot)) cc_outl_fixed(ot, method = method, mltpl = mltpl, tdi = tdi, thinning = thinning,
                                     thinning_res = thinning_res, value = "ids") else character(0)
  rownames(x) %in% ids }}
out <- list(
  equal = !cc_equ(x, test = "absolute", value = "flagged", verbose = FALSE),
  zeros = !cc_zero_fixed(x, buffer = 0.5, value = "flagged"),
  capitals = !cc_cap(x, buffer = 10000, value = "flagged", verbose = FALSE),
  centroids = !cc_cen(x, buffer = 1000, test = "both", value = "flagged", verbose = FALSE),
  seas = !cc_sea(x, scale = 50, value = "flagged", verbose = FALSE),
  outliers = otl(),
  outliers_mad = otl("mad"), outliers_distance = otl("distance", tdi = 500), outliers_thin = otl(thinning = TRUE),
  gbif = !cc_gbif(x, value = "flagged", verbose = FALSE),
  institutions = !cc_inst(x, buffer = 100, value = "flagged", verbose = FALSE){extra})
lapply(out, as.numeric) }}"""
_CC_EXTRA_SYNTH = """,
  urban = !cc_urb(x, value = "flagged", verbose = FALSE),
  countries = !cc_coun_fixed(x, iso3 = "iso3", ref_col = "iso_a3_eh", value = "flagged"),
  duplicates = !cc_dupl(x, species = "species", value = "flagged", verbose = FALSE),
  aohi = !cc_aohi_fixed(x, buffer = 10000, value = "flagged")"""


def _cc_py(points: str, synthetic: bool):
    def run(**kw):
        import landschaft as ls

        occ = kw[points]
        extra = dict(urban=True, countries="iso3", duplicates=True, aohi=True) if synthetic else {}
        f = ls.sdm.clean_occurrences(occ, action="flag", **extra)
        out = {k[5:]: f[k].to_numpy(float) for k in f.columns if k.startswith("flag_") and k != "flag_invalid"}
        for name, opts in (("outliers_mad", {"method": "mad"}), ("outliers_distance", {"method": "distance", "tdi": 500}),
                           ("outliers_thin", {"thinning": True})):
            g = ls.sdm.clean_occurrences(occ, action="flag", equal=False, zeros=False, capitals=False, centroids=False,
                                         seas=False, gbif=False, institutions=False, outliers=opts)
            out[name] = g["flag_outliers"].to_numpy(float)
        return out
    return run


CASES["clean_synthetic"] = Case("clean", r=_CC_R.format(p="cc_synthetic", extra=_CC_EXTRA_SYNTH),
                                inputs=("cc_synthetic",), py=_cc_py("cc_synthetic", True))
CASES["clean_acaule"] = Case("clean", r=_CC_R.format(p="acaule_valid", extra=""), inputs=("acaule_valid",),
                             py=_cc_py("acaule_valid", False))


_USDM_LIB = "suppressMessages(library(usdm)); "


def _usdm_vif_py(name: str, seed=None, size=5000):
    def run(**kw):
        import landschaft as ls

        v = ls.sdm.vif(kw[name], seed=seed, size=size)
        return {"vif": v["vif"].to_numpy(), "names": v["variable"].tolist()}
    return run


def _usdm_select_py(name: str, seed=None, **opts):
    def run(**kw):
        import landschaft as ls

        out = ls.sdm.select_predictors(kw[name], seed=seed, **opts)
        sel = out.attrs["selection"]
        return {"excluded": sel["excluded"], "kept": sel["kept"], "vif": list(sel["vif"].values())}
    return run


def _usdm_r_select(fn: str, data: str, args: str, seed=None) -> str:
    s = f"set.seed({seed}); " if seed is not None else ""
    return (f"{{ {_USDM_LIB}{s}v <- suppressWarnings({fn}({data}{args})); list(excluded = v@excluded,"
            " kept = v@results$Variables, vif = v@results$VIF) }")


for _name, _data, _seed in [("vif_table", "vif_table", None), ("vif_collinear", "vif_collinear", None),
                            ("vif_big", "vif_big", 42), ("vif_bioclim", "bioclim", 1),
                            ("vif_stack_80", "stack_80", None), ("vif_stack_200", "stack_200", 7)]:
    _s = f"set.seed({_seed}); " if _seed is not None else ""
    CASES[_name] = Case("predictors",
                        r=f"{{ {_USDM_LIB}{_s}v <- suppressWarnings(vif({_data})); list(vif = v$VIF,"
                          " names = v$Variables) }",
                        inputs=(_data,), py=_usdm_vif_py(_data, _seed))

for _name, _fn, _data, _args, _seed, _opts in [
    ("vifstep_table", "vifstep", "vif_table", "", None, {"vif": 10}),
    ("vifstep_table_keep", "vifstep", "vif_table", ", th = 3, keep = 'b'", None, {"vif": 3, "keep": "b"}),
    ("vifcor_pearson", "vifcor", "vif_table", ", th = 0.7", None, {"vif": None, "cor": 0.7}),
    ("vifcor_spearman", "vifcor", "vif_table", ", th = 0.7, method = 'spearman'", None,
     {"vif": None, "cor": 0.7, "method": "spearman"}),
    ("vifcor_kendall", "vifcor", "vif_table", ", th = 0.6, method = 'kendall'", None,
     {"vif": None, "cor": 0.6, "method": "kendall"}),
    ("vifcor_keep", "vifcor", "vif_table", ", th = 0.5, keep = 'a'", None, {"vif": None, "cor": 0.5, "keep": "a"}),
    ("vifcor_collinear", "vifcor", "vif_collinear", ", th = 0.9", None, {"vif": None, "cor": 0.9}),
    ("vifstep_collinear", "vifstep", "vif_collinear", ", th = 10", None, {"vif": 10}),
    ("vifstep_bioclim", "vifstep", "bioclim", ", th = 10", 1, {"vif": 10}),
    ("vifcor_bioclim", "vifcor", "bioclim", ", th = 0.7", 2, {"vif": None, "cor": 0.7}),
    ("vifstep_stack_200", "vifstep", "stack_200", ", th = 2", 3, {"vif": 2}),
]:
    CASES[_name] = Case("predictors", r=_usdm_r_select(_fn, _data, _args, _seed), inputs=(_data,),
                        py=_usdm_select_py(_data, _seed, **_opts))


# dismo::randomPoints on a mask with NA where some predictor is missing (and outside the area / the
# distance band / inside the envelope): the cells ls.sdm.background draws for the same seed.
_BG_MASK = "m <- ifel(any(is.na({env})), NA, 1); "


def _bg_r(env: str, n: int, seed: int, p: bool = True, area: bool = False, prob: str | None = None,
          distance=None, envelope=None) -> str:
    code = "{ suppressMessages({ library(dismo); library(terra) }); " + _BG_MASK.format(env=env)
    if area:
        code += "m <- mask(m, vect(bradypus_area)); "
    pts = "as.matrix(bradypus[, c('lon', 'lat')])"
    if distance is not None:
        lo, hi = distance
        code += (f"o <- rast(m); values(o) <- NA; o[cellFromXY(o, {pts})] <- 1; d <- distance(o); "
                 f"m[!(d > {lo} & d <= {hi})] <- NA; ")
    if envelope is not None:
        q = envelope
        code += (f"cl <- unique(na.omit(cellFromXY({env}, {pts}))); inside <- rast(m); values(inside) <- TRUE; "
                 f"for (k in 1:nlyr({env})) {{ v <- values({env}[[k]], mat = FALSE); "
                 f"qs <- quantile(v[cl], c({q}, 1 - {q}), na.rm = TRUE); "
                 f"inside <- inside & ({env}[[k]] >= qs[1] & {env}[[k]] <= qs[2]) }}; m[inside] <- NA; ")
    if prob is not None:
        code += f"m <- mask({prob}, m); "
    args = f", p = {pts}" if p else ""
    if prob is not None:
        args += ", prob = TRUE"
    code += (f"set.seed({seed}); cells <- suppressWarnings(randomPoints(raster::raster(m), {n}{args},"
             " cellnumbers = TRUE)); list(cell = cells) }")
    return code


def _bg_py(env: str, n: int, seed: int, p: bool = True, area: bool = False, prob: str | None = None,
           distance=None, envelope=None):
    def run(**kw):
        import warnings

        import landschaft as ls

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bg = ls.sdm.background(kw[env], n, occ=kw["bradypus"] if p or distance or envelope else None,
                                   exclude_occurrences=p, area=kw["bradypus_area"] if area else None,
                                   bias=kw[prob] if prob else None, distance=distance, envelope=envelope,
                                   seed=seed)
        return {"cell": bg["cell"].to_numpy() + 1}
    return run


for _name, _env, _n, _seed, _kw in [
    ("bg_lonlat", "bioclim", 1000, 1, {}),
    ("bg_no_presences", "bioclim", 500, 2, {"p": False}),
    ("bg_area", "bioclim", 1000, 3, {"area": True}),
    ("bg_shortfall", "bioclim", 1500, 6, {"area": True}),
    ("bg_planar", "stack_200", 400, 4, {"p": False}),
    ("bg_bias", "bioclim", 300, 5, {"prob": "bias_bio12"}),
    ("bg_distance", "bioclim", 400, 7, {"distance": (100_000, 800_000)}),
    ("bg_envelope", "bioclim", 400, 8, {"envelope": 0.025}),
]:
    _inputs = {_env, "bradypus"}
    if _kw.get("area"):
        _inputs.add("bradypus_area")
    if _kw.get("prob"):
        _inputs.add(_kw["prob"])
    CASES[_name] = Case("background", r=_bg_r(_env, _n, _seed, **_kw), inputs=tuple(sorted(_inputs)),
                        py=_bg_py(_env, _n, _seed, **_kw))


_CV = ("{{ suppressMessages({{ library(blockCV); library(sf) }}); sf_use_s2(FALSE); "
       "f <- suppressWarnings(cv_spatial({p}, column = 'occ'{args}, progress = FALSE, report = FALSE,"
       " plot = FALSE, biomod2 = FALSE)$folds_ids); f[is.na(f)] <- 0; list(folds = f) }}")


def _cv_py(points: str, **opts):
    def run(**kw):
        import landschaft as ls

        return {"folds": ls.sdm.spatial_blocks(kw[points], column="occ", **opts)}
    return run


for _name, _p, _args, _opts in [
    ("cv_square_random", "blk_points", ", size = 2000, k = 4, hexagon = FALSE, iteration = 20, seed = 3",
     dict(size=2000, k=4, hexagon=False, iteration=20, seed=3)),
    ("cv_square_edges", "blk_grid", ", rows_cols = c(4, 5), k = 3, hexagon = FALSE, iteration = 5, seed = 11",
     dict(rows_cols=(4, 5), k=3, hexagon=False, iteration=5, seed=11)),
    ("cv_square_systematic", "blk_points", ", size = 2000, k = 5, hexagon = FALSE, selection = 'systematic'",
     dict(size=2000, k=5, hexagon=False, selection="systematic", seed=1)),
    ("cv_square_checkerboard", "blk_points", ", size = 1500, hexagon = FALSE, selection = 'checkerboard'",
     dict(size=1500, hexagon=False, selection="checkerboard", seed=1)),
    ("cv_square_offset", "blk_points", ", size = 2000, k = 3, hexagon = FALSE, selection = 'systematic',"
     " offset = c(0.3, 0.6)", dict(size=2000, k=3, hexagon=False, selection="systematic", offset=(0.3, 0.6), seed=1)),
    ("cv_square_extend", "blk_points", ", size = 2000, k = 4, hexagon = FALSE, extend = 3, iteration = 10, seed = 8",
     dict(size=2000, k=4, hexagon=False, extend=3, iteration=10, seed=8)),
    ("cv_hex_random", "blk_points", ", size = 2500, k = 4, iteration = 10, seed = 5",
     dict(size=2500, k=4, iteration=10, seed=5)),
    ("cv_hex_systematic", "blk_points", ", size = 2500, k = 4, selection = 'systematic'",
     dict(size=2500, k=4, selection="systematic", seed=1)),
    ("cv_hex_flat", "blk_points", ", size = 2500, k = 3, flat_top = TRUE, iteration = 10, seed = 6",
     dict(size=2500, k=3, flat_top=True, iteration=10, seed=6)),
    ("cv_hex_rows", "blk_points", ", rows_cols = c(4, 9), k = 4, iteration = 10, seed = 7",
     dict(rows_cols=(4, 9), k=4, iteration=10, seed=7)),
    ("cv_hex_offset", "blk_points", ", size = 2500, k = 4, offset = c(0.3, 0.6), iteration = 5, seed = 9",
     dict(size=2500, k=4, offset=(0.3, 0.6), iteration=5, seed=9)),
    ("cv_lonlat_square", "brad_points", ", size = 500000, k = 5, hexagon = FALSE, iteration = 10, seed = 1",
     dict(size=500_000, k=5, hexagon=False, iteration=10, seed=1)),
    ("cv_lonlat_hex", "brad_points", ", size = 500000, k = 5, iteration = 10, seed = 2",
     dict(size=500_000, k=5, iteration=10, seed=2)),
]:
    CASES[_name] = Case("folds", r=_CV.format(p=_p, args=_args), inputs=(_p,), py=_cv_py(_p, **_opts))

_ENM = "{{ suppressMessages(library(ENMeval)); {seed}g <- {call}; list(occs = g$occs.grp, bg = g$bg.grp) }}"


def _enm_py(fn: str, **opts):
    def run(**kw):
        import landschaft as ls

        data = _bradypus_data()
        f = getattr(ls.sdm, fn)(data, **opts) if fn != "checkerboard_folds" else \
            ls.sdm.checkerboard_folds(data, kw["bioclim"], **opts)
        return {"occs": f.folds[data.y == 1], "bg": f.folds[data.y == 0]}
    return run


for _o in ("lat_lon", "lon_lat", "lon_lon", "lat_lat"):
    CASES[f"enm_block_{_o}"] = Case("folds", r=_ENM.format(seed="", call=f"get.block(brad_occs, brad_bg, '{_o}')"),
                                    inputs=("brad_occs", "brad_bg"), py=_enm_py("quadrant_folds", orientation=_o))
for _name, _af, _py in [("enm_checker_2", "2", 2), ("enm_checker_4", "4", 4), ("enm_checker_2_2", "c(2, 2)", (2, 2)),
                        ("enm_checker_3_2", "c(3, 2)", (3, 2))]:
    CASES[_name] = Case("folds", r=_ENM.format(seed="", call=f"get.checkerboard(brad_occs, bioclim, brad_bg, {_af})"),
                        inputs=("brad_occs", "brad_bg", "bioclim"), py=_enm_py("checkerboard_folds", aggregation=_py))
CASES["enm_randomkfold"] = Case("folds", r=_ENM.format(seed="set.seed(3); ", call="get.randomkfold(brad_occs, brad_bg, 4)"),
                                inputs=("brad_occs", "brad_bg"), py=_enm_py("random_folds", k=4, seed=3))
CASES["enm_randomkfold_7"] = Case("folds", r=_ENM.format(seed="set.seed(10); ",
                                                         call="get.randomkfold(brad_occs, brad_bg, 7)"),
                                  inputs=("brad_occs", "brad_bg"), py=_enm_py("random_folds", k=7, seed=10))


_CC_EDGES_R = """{ suppressMessages(library(CoordinateCleaner)); source(file.path(TOOLS, "cc_fixed.R"))
x <- cc_edges; x$decimalLongitude <- x$lon; x$decimalLatitude <- x$lat
dateline <- data.frame(capital.lon = 179.995, capital.lat = -16.5, capital = "X")
list(cap_geod = !cc_cap_fixed(x, buffer = 10000, value = "flagged"),
     cap_terra = !cc_cap_fixed(x, buffer = 10000, geod = FALSE, value = "flagged"),
     dateline_geod = !cc_cap_fixed(x, buffer = 10000, ref = dateline, value = "flagged"),
     dateline_terra = !cc_cap_fixed(x, buffer = 10000, geod = FALSE, ref = dateline, value = "flagged"),
     pole = !cc_cen_fixed(x, buffer = 1000, test = "country", value = "flagged"),
     pole_big = !cc_cen_fixed(x, buffer = 60000, test = "country", value = "flagged"),
     zeros = !cc_zero_fixed(x, buffer = 0, value = "flagged")) }"""


def _cc_edges_py(**kw):
    import landschaft as ls

    occ = kw["cc_edges"]
    off = dict(equal=False, zeros=False, capitals=False, centroids=False, seas=False, outliers=False, gbif=False,
               institutions=False)
    dateline = pd.DataFrame({"lon": [179.995], "lat": [-16.5]})

    def flag(test, **opts):
        f = ls.sdm.clean_occurrences(occ, action="flag", **{**off, **opts})
        return f[f"flag_{test}"].to_numpy(float)

    return {"cap_geod": flag("capitals", capitals=10_000),
            "cap_terra": flag("capitals", capitals={"buffer": 10_000, "geod": False}),
            "dateline_geod": flag("capitals", capitals={"ref": dateline}),
            "dateline_terra": flag("capitals", capitals={"ref": dateline, "geod": False}),
            "pole": flag("centroids", centroids={"buffer": 1000, "detail": "country"}),
            "pole_big": flag("centroids", centroids={"buffer": 60_000, "detail": "country"}),
            "zeros": flag("zeros", zeros=0)}


CASES["clean_edges"] = Case("clean", r=_CC_EDGES_R, inputs=("cc_edges",), py=_cc_edges_py)


_BOYCE_R = ("{{ source(file.path(TOOLS, 'ecospat_fixed.R')); d <- {d}; obs <- d$pred[d$y == 1]; "
            "fit <- d$pred[d$y == 0]; b <- ecospat_boyce_fixed(c(fit, obs), obs{args}); "
            "list(cor = b$cor, f = b$F.ratio, hs = b$HS, used = b$used) }}")


def _boyce_py(d: str, **opts):
    def run(**kw):
        import landschaft as ls

        x = kw[d]
        obs = x["pred"][x["y"] == 1].to_numpy()
        fit = x["pred"][x["y"] == 0].to_numpy()
        b = ls.sdm.boyce(obs, np.concatenate([fit, obs]), **opts)
        return {"cor": [b["cor"]], "f": b["f_ratio"], "hs": b["hs"], "used": b["used"]}
    return run


for _name, _d, _args, _opts in [
    ("boyce_default", "eval_beta", "", {}),
    ("boyce_scale1000", "eval_scale1000", "", {}),
    ("boyce_nclass", "eval_beta", ", nclass = 5", {"nclass": 5}),
    ("boyce_pearson", "eval_beta", ", method = 'pearson'", {"method": "pearson"}),
    ("boyce_kendall", "eval_beta", ", method = 'kendall'", {"method": "kendall"}),
    ("boyce_keep_duplicates", "eval_beta", ", rm.duplicate = FALSE", {"rm_duplicate": False}),
    ("boyce_window", "eval_beta", ", window.w = 0.25, res = 20", {"window": 0.25, "res": 20}),
    ("boyce_breaks", "eval_beta", ", nclass = c(-1, 0.2, 0.5, 0.7, 1.5)", {"nclass": [-1, 0.2, 0.5, 0.7, 1.5]}),
    ("boyce_small", "eval_small", "", {}),
]:
    CASES[_name] = Case("evaluate", r=_BOYCE_R.format(d=_d, args=_args), inputs=(_d,), py=_boyce_py(_d, **_opts))

_PA_R = ("{{ suppressMessages(library(predicts)); d <- {d}; e <- pa_evaluate(d$pred[d$y == 1], d$pred[d$y == 0]); "
         "s <- e@tr_stats; th <- e@thresholds; "
         "list(auc = e@stats$auc, cor = e@stats$cor, pcor = e@stats$pcor, tr = s$treshold, kappa = s$kappa, "
         "ccr = s$CCR, tpr = s$TPR, tnr = s$TNR, fpr = s$FPR, fnr = s$FNR, ppp = s$PPP, npp = s$NPP, mcr = s$MCR, "
         "or = s$OR, thresholds = c(th$max_kappa, th$max_spec_sens, th$no_omission, th$equal_prevalence, "
         "th$equal_sens_spec)) }}")


def _pa_py(d: str):
    def run(**kw):
        import landschaft as ls

        x = kw[d]
        e = ls.sdm.evaluate(x["pred"][x["y"] == 1].to_numpy(), x["pred"][x["y"] == 0].to_numpy())
        tb = e.table
        th = e.thresholds
        out = {"auc": [e.stats["auc"]], "cor": [e.stats["cor"]], "pcor": [e.stats["pcor"]],
               "tr": tb["threshold"].to_numpy(),
               "thresholds": [th[k] for k in ("max_kappa", "max_spec_sens", "no_omission", "equal_prevalence",
                                              "equal_sens_spec")]}
        for c in ("kappa", "ccr", "tpr", "tnr", "fpr", "fnr", "ppp", "npp", "mcr", "or"):
            out[c] = tb[c].to_numpy()
        return out
    return run


for _d in ("eval_small", "eval_beta", "eval_big"):
    CASES[f"pa_{_d}"] = Case("evaluate", r=_PA_R.format(d=_d), inputs=(_d,), py=_pa_py(_d))

_OR_R = ("{{ suppressMessages(library(ENMeval)); d <- {d}; p <- d$pred[d$y == 1]; out <- list(); "
         "for (n in c(1, 2, 5, 9, 10, 11, 29)) {{ tr <- p[seq_len(n)]; te <- p[-seq_len(n)]; "
         "out[[paste0('mtp', n)]] <- mean(te < min(tr)); "
         "out[[paste0('p10', n)]] <- mean(te < ENMeval:::calc.10p.trainThresh(tr)) }}; out }}")


def _or_py(d: str):
    def run(**kw):
        import landschaft as ls

        x = kw[d]
        p = x["pred"][x["y"] == 1].to_numpy()
        out = {}
        for n in (1, 2, 5, 9, 10, 11, 29):
            tr, te = p[:n], p[n:]
            out[f"mtp{n}"] = [ls.sdm.omission_rate(tr, te, "mtp")]
            out[f"p10{n}"] = [ls.sdm.omission_rate(tr, te, "p10")]
        return out
    return run


CASES["omission_beta"] = Case("evaluate", r=_OR_R.format(d="eval_beta"), inputs=("eval_beta",), py=_or_py("eval_beta"))


_MESS_R = ("{ suppressMessages(library(predicts)); m <- mess(env_warm, ref_x, full = TRUE); "
           "out <- lapply(seq_len(nlyr(m)), function(k) values(m[[k]], mat = FALSE)); names(out) <- names(m); out }")


def _mess_py(**kw):
    import landschaft as ls

    m = ls.sdm.mess(kw["ref_x"], kw["env_warm"], full=True)
    return {str(b): m.sel(band=b).values.ravel() for b in m.band.values}


CASES["mess_full"] = Case("extrapolation", r=_MESS_R, inputs=("ref_x", "env_warm"), py=_mess_py)

_MOP_R = ("{{ suppressMessages({{ library(mop); library(terra) }}); "
          "o <- mop(as.matrix(ref_x), {env}{args}, progress_bar = FALSE); "
          "v <- function(r) if (is.null(r)) rep(NA_real_, ncell({env})) else values(r, mat = FALSE); "
          "out <- list(mop = v(o$mop_distances), basic = v(o$mop_basic)); "
          "if (!is.null(o$mop_simple)) out$simple <- v(o$mop_simple); "
          "if (!is.null(o$mop_detailed$towards_low_combined)) {{ out$low_combined <- v(o$mop_detailed$towards_low_combined); "
          "out$high_combined <- v(o$mop_detailed$towards_high_combined) }}; out }}")


def _mop_py(env: str, **opts):
    def run(**kw):
        import landschaft as ls

        o = ls.sdm.mop(kw["ref_x"], kw[env], **opts)
        return {str(b): o.sel(band=b).values.ravel() for b in o.band.values}
    return run


CASES["mop_detailed"] = Case("extrapolation", r=_MOP_R.format(
    env="env_warm", args=", type = 'detailed', calculate_distance = TRUE, where_distance = 'all'"),
    inputs=("ref_x", "env_warm"), py=_mop_py("env_warm", type="detailed", where="all"))
CASES["mop_scaled"] = Case("extrapolation", r=_MOP_R.format(
    env="bioclim", args=", calculate_distance = TRUE, scale = TRUE, center = TRUE, percentage = 10,"
                        " rescale_distance = TRUE"),
    inputs=("ref_x", "bioclim"), py=_mop_py("bioclim", scale=True, center=True, percentage=10, rescale=True))

# Mahalanobis with the covariance of the reference (mop takes that of the cells tested)
_MOP_MAHAL_R = """{ suppressMessages(library(terra)); xp <- c("bio1", "bio12", "bio16")
m <- as.matrix(ref_x[, xp]); g <- values(env_small[[xp]])
ok <- complete.cases(g); k <- max(round(nrow(m) * 5 / 100), 1); cv <- cov(m)
d <- rep(NA_real_, nrow(g))
d[ok] <- apply(g[ok, , drop = FALSE], 1, function(gi) mean(sort(mahalanobis(m, gi, cv))[1:k]))
list(mop = d) }"""


def _mop_mahal_py(**kw):
    import landschaft as ls

    o = ls.sdm.mop(kw["ref_x"], kw["env_small"].sel(band=["bio1", "bio12", "bio16"]), distance="mahalanobis",
                   where="all", percentage=5)
    return {"mop": o.sel(band="mop").values.ravel()}


CASES["mop_mahalanobis"] = Case("extrapolation", r=_MOP_MAHAL_R, inputs=("ref_x", "env_small"), py=_mop_mahal_py,
                                notes="tolerance: R inverts the covariance with LAPACK")

# dsmextra's ExDet with its combinatorial MIC numbered the right way round
_EXDET_R = """{ .libPaths(.libPaths()); suppressMessages(library(dsmextra)); xp <- c("bio1", "bio12", "bio16")
ref <- na.omit(ref_x[, xp]); e <- dsmextra:::ExDet(ref, tg_small[, xp], xp = xp)
mic <- ifelse(!is.na(e$mic_combinatorial), length(xp) + 1 - e$mic_combinatorial, e$mic)
list(exdet = e$ExDet, mic = mic) }"""


def _exdet_py(**kw):
    import landschaft as ls

    e = ls.sdm.exdet(kw["ref_x"], kw["tg_small"])
    return {"exdet": e["exdet"].to_numpy(), "mic": e["mic"].to_numpy()}


CASES["exdet"] = Case("extrapolation", r=_EXDET_R, inputs=("ref_x", "tg_small"), py=_exdet_py,
                      notes="NT2 to tolerance (LAPACK solve)")


def case_ids(group: str) -> list[str]:
    return [k for k, c in CASES.items() if c.group == group]


def run_python(name: str) -> dict:
    case = CASES[name]
    return case.py(**{k: build_input(k) for k in case.inputs})


__all__ = ["CASES", "GROUPS", "INPUTS", "Case", "build_input", "case_ids", "np", "run_python"]
