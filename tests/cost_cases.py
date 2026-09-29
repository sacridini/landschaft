# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases shared by tools/make_reference_cost.py and the ls.cost tests.

Every raster is a float64 array (NaN = NA) with an extent and a CRS; R rebuilds
it with ``raster(nrows, ncols, xmn, xmx, ymn, ymx, crs)``. Every case runs R
code on it (``r`` is the case's raster, the others by name) giving a named
list, and a Python function giving a dict with the same keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

LONLAT = "+proj=longlat +datum=WGS84 +no_defs"
UTM = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"


@dataclass
class RasterSpec:
    values: np.ndarray  # (ny, nx) or (layers, ny, nx), NaN = NA
    extent: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    crs: str | None = None


def _gdistance_test_raster() -> np.ndarray:
    # gdistance's tests/testthat/helper-data.R: res (column-major, 9 rows) and its fourth mask.
    res = np.array([
        1, 1, 2, 2, 3, 3, 2, 2, 1,
        1, 2, 2, 3, 10, 10, 3, 2, 2,
        1, 2, 2, 3, 10, 10, 3, 2, 2,
        1, 1, 1, 3, 10, 10, 3, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 10, 10, 10, 10, 10, 10, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float).reshape(8, 9).T
    nan = np.nan
    mask = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, nan, nan, nan, 1, 1,
        1, 1, 1, 1, nan, 1, nan, 1, 1,
        1, 1, 1, 1, nan, nan, nan, 1, 1,
        1, nan, nan, nan, 1, 1, 1, 1, 1,
        1, nan, 1, nan, 1, 1, 1, 1, 1,
        1, nan, nan, nan, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(8, 9).T
    return res * mask


def _random(shape, seed, low=0.5, high=5.0, na=0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.uniform(low, high, shape)
    v[rng.random(shape) < na] = np.nan
    return v


def _rasters() -> dict[str, RasterSpec]:
    r = {}
    r["small"] = RasterSpec(_gdistance_test_raster(), (0.0, 8.0, 0.0, 9.0))
    r["rand"] = RasterSpec(_random((30, 40), 1), (500_000.0, 500_400.0, 4_000_000.0, 4_000_300.0), UTM)
    # Cells 30 x 20 m; values with ties (integers) for the asymmetric functions.
    r["rect"] = RasterSpec(np.floor(_random((25, 35), 2, 1, 9)), (1000.0, 2050.0, 0.0, 500.0), UTM)
    r["lonlat"] = RasterSpec(_random((20, 30), 3), (-20.0, 10.0, 30.0, 50.0), LONLAT)
    # Global: edges wrap around the 180th meridian.
    r["global"] = RasterSpec(_random((9, 18), 4, na=0.0), (-180.0, 180.0, -90.0, 90.0), LONLAT)
    r["global_na"] = RasterSpec(_random((12, 24), 5), (-180.0, 180.0, -90.0, 90.0), LONLAT)
    # Three layers for the Mahalanobis transition.
    b = np.stack([_random((15, 20), 6 + k, na=0.0) for k in range(3)])
    b[1] += 0.5 * b[0]
    b[:, 3, 4] = np.nan
    b[2, 7, 11] = np.nan
    r["brick"] = RasterSpec(b, (0.0, 200.0, 0.0, 150.0), UTM)
    r["maungawhau"] = _dataset("maungawhau.tif", "+proj=nzmg +lat_0=-41 +lon_0=173 +x_0=2510000 +y_0=6023150 "
                                                 "+ellps=intl +units=m +no_defs")
    r["europe"] = _dataset("europe.tif", "+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs")
    return r


DATA = Path(__file__).resolve().parents[1] / "src" / "landschaft" / "cost" / "data"


def _dataset(name: str, crs: str) -> RasterSpec:
    """A raster of gdistance's inst/external (exported to GeoTIFF), with R's extent."""
    import rasterio

    with rasterio.open(DATA / name) as src:
        values = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        b = src.bounds
    # 1/3-degree cells: the GeoTIFF's transform gives the right edge a few ulps off R's 45.
    return RasterSpec(values, tuple(round(v, 9) for v in (b.left, b.right, b.bottom, b.top)), crs)


RASTERS = _rasters()


@dataclass
class Case:
    raster: str | None
    r: str  # R code giving a named list
    py: Callable[[dict[str, RasterSpec]], dict]
    group: str = "transition"
    tolerance: float = 0.0  # 0: identical bits
    notes: str = field(default="")


def ls_cost():
    import landschaft as ls

    return ls.cost


def _tr(spec: RasterSpec, *args, **kwargs):
    import landschaft as ls

    return ls.cost.transition(spec.values, *args, extent=spec.extent, crs=spec.crs, **kwargs)


def _transition_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for name in ("small", "rand", "rect", "lonlat", "global", "global_na"):
        for d in (4, 8, 16):
            key = f"{name}_d{d}"
            cases[f"tr_{key}"] = Case(
                name,
                f"""list(
                    mean = transition(r, mean, {d}),
                    inv_mean = transition(r, function(x) 1 / mean(x), {d}),
                    diff = transition(r, function(x) x[2] - x[1], {d}, symm = FALSE),
                    minimum = transition(r, min, {d}),
                    ratio = transition(r, function(x) x[1] / x[2], {d}, symm = FALSE))""",
                lambda R, n=name, d=d: {
                    "mean": _tr(R[n], "mean", d),
                    "inv_mean": 1 / _tr(R[n], "mean", d),
                    "diff": _tr(R[n], lambda x: x[1] - x[0], d, symmetric=False),
                    "minimum": _tr(R[n], "min", d),
                    "ratio": _tr(R[n], lambda x: x[0] / x[1], d, symmetric=False),
                },
            )
            cases[f"geo_{key}"] = Case(
                name,
                f"""tr <- transition(r, function(x) 1 / mean(x), {d})
                    ta <- transition(r, function(x) x[2] - x[1], {d}, symm = FALSE)
                    list(
                      c = geoCorrection(tr, type = "c"),
                      c_scl = geoCorrection(tr, type = "c", scl = TRUE),
                      c_fac = geoCorrection(tr, type = "c", multpl = TRUE),
                      asym = geoCorrection(ta, type = "c"),
                      res = {{m <- tr; matrixValues(m) <- "resistance"; geoCorrection(m, type = "c")}},
                      r = if (isLonLat(r)) geoCorrection(tr, type = "r", scl = TRUE) else list(),
                      r_res = if (isLonLat(r)) {{m <- tr; matrixValues(m) <- "resistance"
                                                 geoCorrection(m, type = "r")}} else list())""",
                lambda R, n=name, d=d: _geo(R[n], d),
            )
    return cases


def _geo(spec: RasterSpec, d: int) -> dict:
    import landschaft as ls

    tr = 1 / _tr(spec, "mean", d)
    ta = _tr(spec, lambda x: x[1] - x[0], d, symmetric=False)
    res = tr.copy()
    res.matrix_values = "resistance"
    out = {
        "c": ls.cost.geo_correction(tr, "c"),
        "c_scl": ls.cost.geo_correction(tr, "c", scale=True),
        "c_fac": ls.cost.geo_correction(tr, "c", factors=True),
        "asym": ls.cost.geo_correction(ta, "c"),
        "res": ls.cost.geo_correction(res, "c"),
    }
    if tr.lonlat:
        out["r"] = ls.cost.geo_correction(tr, "r", scale=True)
        out["r_res"] = ls.cost.geo_correction(res, "r")
    return out


def _operation_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for name in ("small", "rand", "global_na"):
        cases[f"ops_{name}"] = Case(
            name,
            """t1 <- transition(r, mean, 8)
               t2 <- transition(r, function(x) x[1] + 2 * x[2], 8, symm = FALSE)
               t4 <- transition(r, max, 4)
               list(
                 prod = t1 * t2, sum = t1 + t2, diff = t2 - t1, quot = t1 / 3, scaled = t1 * 2.5,
                 sum4 = t1 + t4, prod4 = t1 * t4, sqrt = sqrt(t1),
                 row = normalize(t1, "row"), col = normalize(t2, "col"), symm = normalize(t1, "symm"),
                 recip = sumReciprocal(t1, t4),
                 r_nzcol = raster(t2), r_nzrow = raster(t2, "NZrowMeans"), r_colsums = raster(t1, "colSums"),
                 r_rowmeans = raster(t2, "rowMeans"),
                 adjacency = adjacencyFromTransition(t2))""",
            lambda R, n=name: _ops(R[n]),
        )
    # gdistance fails on an asymmetric matrix with empty rows (NA cells): no NA here.
    cases["ops_symm_asym"] = Case(
        "global",
        """list(symm_asym = normalize(transition(r, function(x) x[1] + 2 * x[2], 8, symm = FALSE), "symm"))""",
        lambda R: {"symm_asym": ls_cost().normalize(_tr(R["global"], lambda x: x[0] + 2 * x[1], 8, symmetric=False),
                                                     "symm")},
    )
    # gdistance's own method fails (it leaves transitionCells empty, an invalid object): its code
    # with that set.
    cases["mahal"] = Case(
        "brick",
        """mahal <- function(x, directions) {
             xy <- cbind(1:ncell(x), getValues(x))
             xy <- na.omit(xy)
             dataCells <- xy[, 1]
             adj <- adjacent(x, cells = dataCells, pairs = TRUE, target = dataCells, directions = directions)
             x.minus.y <- raster::getValues(x)[adj[, 1], ] - raster::getValues(x)[adj[, 2], ]
             cov.inv <- solve(cov(xy[, -1]))
             mahaldistance <- apply(x.minus.y, 1, function(x) { sqrt((x %*% cov.inv) %*% x) })
             mahaldistance <- mean(mahaldistance) / (mahaldistance + mean(mahaldistance))
             transitiondsC <- new("dsCMatrix", p = as.integer(rep(0, ncell(x) + 1)),
                                  Dim = as.integer(c(ncell(x), ncell(x))),
                                  Dimnames = list(as.character(1:ncell(x)), as.character(1:ncell(x))))
             transitiondsC[adj] <- mahaldistance
             new("TransitionLayer", transitionMatrix = transitiondsC, transitionCells = 1:ncell(x),
                 nrows = as.integer(nrow(x)), ncols = as.integer(ncol(x)), extent = extent(x),
                 crs = projection(x, asText = FALSE), matrixValues = "conductance")
           }
           list(mahal8 = mahal(r, 8), mahal4 = mahal(r, 4))""",
        lambda R: {"mahal8": _tr(R["brick"], "mahal", 8), "mahal4": _tr(R["brick"], "mahal", 4)},
    )
    # gdistance's vignette: Tobler's hiking function on the Maungawhau volcano.
    cases["tobler"] = Case(
        "maungawhau",
        """hd <- transition(r, function(x) x[2] - x[1], 8, symm = FALSE)
           slope <- geoCorrection(hd)
           adj <- adjacent(r, cells = 1:ncell(r), pairs = TRUE, directions = 8)
           speed <- slope
           speed[adj] <- 6 * exp(-3.5 * abs(slope[adj] + 0.05))
           list(slope = slope, speed = speed, conductance = geoCorrection(speed))""",
        _tobler,
        tolerance=4e-16,  # R's exp is mingw's x87 exp
    )
    cases["europe"] = Case(
        "europe",
        """tr <- transition(r, mean, 8)
           list(c = geoCorrection(tr, type = "c"), r = geoCorrection(tr, type = "r"))""",
        lambda R: {"c": _geo_plain(R["europe"], "c"), "r": _geo_plain(R["europe"], "r")},
    )
    return cases


def _geo_plain(spec: RasterSpec, type: str):
    import landschaft as ls

    return ls.cost.geo_correction(_tr(spec, "mean", 8), type)


def _tobler(R) -> dict:
    import landschaft as ls

    hd = _tr(R["maungawhau"], lambda x: x[1] - x[0], 8, symmetric=False)
    slope = ls.cost.geo_correction(hd)
    speed = slope.map(lambda s: 6 * np.exp(-3.5 * np.abs(s + 0.05)), pairs="all")
    return {"slope": slope, "speed": speed, "conductance": ls.cost.geo_correction(speed)}


def _ops(spec: RasterSpec) -> dict:
    import landschaft as ls

    t1 = _tr(spec, "mean", 8)
    t2 = _tr(spec, lambda x: x[0] + 2 * x[1], 8, symmetric=False)
    t4 = _tr(spec, "max", 4)
    adj = t2.adjacency()
    return {
        "prod": t1 * t2, "sum": t1 + t2, "diff": t2 - t1, "quot": t1 / 3, "scaled": t1 * 2.5,
        "sum4": t1 + t4, "prod4": t1 * t4, "sqrt": t1.map(np.sqrt),
        "row": ls.cost.normalize(t1, "row"), "col": ls.cost.normalize(t2, "col"),
        "symm": ls.cost.normalize(t1, "symm"),
        "recip": ls.cost.sum_reciprocal(t1, t4),
        "r_nzcol": t2.to_raster(), "r_nzrow": t2.to_raster("NZrowMeans"), "r_colsums": t1.to_raster("colSums"),
        "r_rowmeans": t2.to_raster("rowMeans"),
        "adjacency": np.column_stack(adj) + 1,
    }


def _cases() -> dict[str, Case]:
    cases = {}
    cases.update(_transition_cases())
    cases.update(_operation_cases())
    return cases


CASES = _cases()
# Rasters a group's R code uses by name besides the cases' own.
EXTRA_RASTERS: dict[str, tuple[str, ...]] = {}
GROUPS = sorted({c.group for c in CASES.values()})
