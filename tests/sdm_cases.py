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

GROUPS: list[str] = ["smoke", "cells", "thin", "clean", "predictors", "background"]


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


INPUTS: dict[str, Callable[[], object]] = {
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


def case_ids(group: str) -> list[str]:
    return [k for k, c in CASES.items() if c.group == group]


def run_python(name: str) -> dict:
    case = CASES[name]
    return case.py(**{k: build_input(k) for k in case.inputs})


__all__ = ["CASES", "GROUPS", "INPUTS", "Case", "build_input", "case_ids", "np", "run_python"]
