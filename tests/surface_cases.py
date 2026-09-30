# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases shared by tools/make_reference_surface.py and the ls.surface tests.

Every raster is a float64 array (NaN = NA) with an extent and a CRS; R rebuilds
it with ``terra::rast(nrows, ncols, xmin, xmax, ymin, ymax, crs)``. Every case
runs R code on it (``r`` is the case's raster, the others by name) giving a
named list, and a Python function giving a dict with the same keys. Call the
geodiv functions as ``geodiv::`` (``terra::sds`` masks ``geodiv::sds``), and
``fx$`` for geodiv with its errors corrected (tools/surface_fixed.R) where
ls.surface corrects them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

UTM = "+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs"


@dataclass
class RasterSpec:
    values: np.ndarray  # (ny, nx), NaN = NA
    extent: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    crs: str | None = None


DATA = Path(__file__).resolve().parents[1] / "src" / "landschaft" / "surface" / "data"


def _dataset(name: str) -> RasterSpec:
    """A geodiv dataset (exported to GeoTIFF by terra), with terra's extent."""
    import rasterio

    with rasterio.open(DATA / f"{name}.tif") as src:
        values = src.read(1).astype(np.float64)
        b = src.bounds
        crs = src.crs.to_proj4() if src.crs else None
    return RasterSpec(values, (b.left, b.right, b.bottom, b.top), crs)


def _normal(shape, seed, na=0.0, scale=1.0, trend=0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, scale, shape)
    if trend:
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        v += trend * (0.3 * xx - 0.2 * yy + 0.01 * xx * yy)
    if na:
        v[rng.random(shape) < na] = np.nan
    return v


def _dem(shape, seed, na=0.0) -> np.ndarray:
    """A smooth DEM (metres) with hills and valleys and some noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    v = (500 + 60 * np.sin(xx / 6.0) * np.cos(yy / 7.0) + 25 * np.sin((xx + 2 * yy) / 11.0) + 0.8 * xx
         + rng.normal(0, 1.5, shape))
    if na:
        v[rng.random(shape) < na] = np.nan
    return v


def _rasters() -> dict[str, RasterSpec]:
    r = {}
    r["rand"] = RasterSpec(_normal((20, 30), 1), (500_000.0, 500_300.0, 4_000_000.0, 4_000_200.0), UTM)
    r["rand_na"] = RasterSpec(_normal((24, 24), 2, na=0.08), (0.0, 240.0, 0.0, 240.0), UTM)
    # Cells 30 x 20 m, a trend to remove, some NA.
    r["rect"] = RasterSpec(_normal((25, 35), 3, na=0.03, trend=1.0), (1000.0, 2050.0, 0.0, 500.0), UTM)
    r["square"] = RasterSpec(_normal((32, 32), 4, scale=2.0, trend=0.5), (0.0, 320.0, 0.0, 320.0), UTM)
    # Integers: ties in sorting, quantiles and peaks.
    r["ties"] = RasterSpec(np.floor(_normal((15, 18), 5, scale=3.0)), (0.0, 180.0, 0.0, 150.0), UTM)
    r["positive"] = RasterSpec(np.abs(_normal((12, 16), 6)) + 1.0, (0.0, 160.0, 0.0, 120.0), UTM)
    r["const"] = RasterSpec(np.full((6, 7), 2.5), (0.0, 70.0, 0.0, 60.0), UTM)
    r["tiny"] = RasterSpec(np.array([[1.0, -2.0, 0.5], [3.0, 0.25, -1.0], [-0.5, 2.0, 1.5]]),
                           (0.0, 3.0, 0.0, 3.0), UTM)
    r["normforest"] = _dataset("normforest")
    r["orforest"] = _dataset("orforest")
    r["orelevation"] = _dataset("orelevation")
    # a 15 x 15 corner of normforest (longitude/latitude) for the windows
    nf = r["normforest"]
    xmin, xmax, ymin, ymax = nf.extent
    ny, nx = nf.values.shape
    xr, yr = (xmax - xmin) / nx, (ymax - ymin) / ny
    r["dem"] = RasterSpec(_dem((40, 50), 9), (400_000.0, 401_500.0, 5_000_000.0, 5_001_200.0), UTM)
    r["dem_rect"] = RasterSpec(_dem((30, 36), 10, na=0.01), (0.0, 36 * 25.0, 0.0, 30 * 40.0), UTM)
    el = r["orelevation"]
    exmin, exmax, eymin, eymax = el.extent
    eny, enx = el.values.shape
    exr, eyr = (exmax - exmin) / enx, (eymax - eymin) / eny
    r["elev_small"] = RasterSpec(el.values[100:140, 150:190].copy(),
                                 (exmin + 150 * exr, exmin + 190 * exr, eymax - 140 * eyr, eymax - 100 * eyr), el.crs)
    r["nf_small"] = RasterSpec(nf.values[100:115, 120:135].copy(),
                               (xmin + 120 * xr, xmin + 135 * xr, ymax - 115 * yr, ymax - 100 * yr), nf.crs)
    return r


RASTERS = _rasters()


@dataclass
class Case:
    raster: str | None
    r: str  # R code giving a named list
    py: Callable[[dict[str, RasterSpec]], dict]
    group: str
    tolerance: float | dict = 0.0  # 0: identical bits; or {key: (rtol, atol)}
    notes: str = field(default="")


def surface():
    import landschaft as ls

    return ls.surface


def _grid(spec: RasterSpec) -> dict:
    return {"extent": spec.extent, "crs": spec.crs}


def _amplitude(name: str) -> Callable[[dict], dict]:
    def run(rasters):
        s, spec = surface(), rasters[name]
        z, g = spec.values, _grid(spec)
        return {"sa": s.sa(z, **g), "sq": s.sq(z, **g), "ssk": s.ssk(z, **g), "ssk_unadj": s.ssk(z, adj=False, **g),
                "sku": s.sku(z, **g), "sku_raw": s.sku(z, excess=False, **g), "sv": s.sv(z, **g),
                "sph": s.sph(z, **g), "smean": s.smean(z, **g)}

    return run


AMPLITUDE_R = """list(sa = geodiv::sa(r), sq = fx$sq(r), ssk = fx$ssk(r),
     ssk_unadj = fx$ssk(r, adj = FALSE), sku = fx$sku(r), sku_raw = fx$sku(r, excess = FALSE),
     sv = geodiv::sv(r), sph = geodiv::sph(r), smean = geodiv::smean(r))"""


def _amplitude_cases() -> dict[str, Case]:
    return {f"amplitude_{name}": Case(name, AMPLITUDE_R, _amplitude(name), "amplitude")
            for name in ("rand", "rand_na", "rect", "ties", "positive", "const", "tiny", "normforest", "orforest",
                         "orelevation")}


def _plane(name: str, orders=(0, 1, 2, 3)) -> Callable[[dict], dict]:
    def run(rasters):
        s, spec = surface(), rasters[name]
        z, g = spec.values, _grid(spec)
        out = {f"fit{k}": s.fit_plane(z, k, **g).values for k in orders}
        best = s.best_fit_plane(z, **g)
        out["best"] = best.values
        out["order"] = best.attrs["order"]
        out["removed"] = s.remove_plane(z, **g).values
        return out

    return run


def _plane_r(orders=(0, 1, 2, 3)) -> str:
    fits = ", ".join(f"fit{k} = geodiv::fitplane(r, {k})" for k in orders)
    return (f"{{ b <- fx$bestfitplane(r)\n  list({fits}, best = b$fit, order = b$order, "
            f"removed = r - setValues(r, b$fit)) }}")


def _plane_cases() -> dict[str, Case]:
    # Not "tiny": with fewer cells than coefficients (order 3: 10) spatial's
    # householder reads past its matrix; ls.surface raises an error instead.
    return {f"plane_{name}": Case(name, _plane_r(), _plane(name), "plane")
            for name in ("rand", "rand_na", "rect", "square", "ties", "normforest", "orforest", "orelevation")}


def _hybrid(name: str) -> Callable[[dict], dict]:
    def run(rasters):
        s, spec = surface(), rasters[name]
        z, g = spec.values, _grid(spec)
        return {"sdq": s.sdq(z, **g), "sdq6": s.sdq6(z, **g), "surface_area": s.surface_area(z, **g),
                "flatsa": s.flat_area(z, **g), "sdr": s.sdr(z, **g)}

    return run


HYBRID_R = """list(sdq = fx$sdq(r), sdq6 = fx$sdq6(r), surface_area = geodiv::surface_area(r),
     flatsa = geodiv::flatsa(r), sdr = geodiv::sdr(r))"""


def _hybrid_cases() -> dict[str, Case]:
    return {f"hybrid_{name}": Case(name, HYBRID_R, _hybrid(name), "hybrid")
            for name in ("rand", "rand_na", "rect", "square", "ties", "positive", "const", "tiny", "normforest",
                         "orforest", "orelevation")}


def _geodiv_points(df, ny: int, nx: int) -> dict:
    """findpeaks' columns from ls.surface.find_peaks: 1-based rows and columns, and
    ``ind``, the number of the cell among the interior cells."""
    row, col = df["row"].to_numpy() + 1, df["col"].to_numpy() + 1
    return {"x": df["x"].to_numpy(), "y": df["y"].to_numpy(), "val": df["value"].to_numpy(),
            "ind": (row - 2) * (nx - 2) + (col - 1), "row": row, "col": col}


def _features(name: str) -> Callable[[dict], dict]:
    def run(rasters):
        s, spec = surface(), rasters[name]
        z, g = spec.values, _grid(spec)
        ny, nx = z.shape
        valleys = s.find_valleys(z, **g)
        return {"peaks": _geodiv_points(s.find_peaks(z, **g), ny, nx),
                "valleys": {"row": valleys["row"].to_numpy() + 1, "col": valleys["col"].to_numpy() + 1,
                            "val": valleys["value"].to_numpy()},
                "sds": s.sds(z, **g), "ssc": s.ssc(z, **g), "s10z": s.s10z(z, **g)}

    return run


FEATURES_R = """list(peaks = geodiv::findpeaks(r), valleys = fx$findvalleys(r), sds = geodiv::sds(r),
     ssc = fx$ssc(r), s10z = fx$s10z(r))"""


def _features_cases() -> dict[str, Case]:
    return {f"features_{name}": Case(name, FEATURES_R, _features(name), "features")
            for name in ("rand", "rand_na", "rect", "square", "ties", "tiny", "normforest", "orelevation")}


BEARING_PROBS = (0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0)


def _bearing(name: str) -> Callable[[dict], dict]:
    def run(rasters):
        s, spec = surface(), rasters[name]
        z, g = spec.values, _grid(spec)
        curve = s.bearing_area(z, **g)
        return {"curve_x": curve["bearing"].to_numpy(), "curve_y": curve["height"].to_numpy(),
                "height": s.height_ba(z, list(BEARING_PROBS), **g), "sdc": s.sdc(z, 0.1, 0.8, **g),
                "sbi": s.sbi(z, **g), "svi": s.svi(z, **g), "sci": s.sci(z, **g), "sk": s.sk(z, **g),
                "spk": s.spk(z, **g), "svk": s.svk(z, **g)}

    return run


BEARING_R = f"""{{
  z <- fx$vals(r)
  cv <- fx$curve(r)
  b <- fx$bearing(r)
  list(curve_x = cv$bearing, curve_y = cv$height,
       height = fx$height(z, c({", ".join(map(str, BEARING_PROBS))})),
       sdc = fx$height(z, 0.1) - fx$height(z, 0.8), sbi = b$sbi, svi = b$svi, sci = b$sci,
       sk = b$sk, spk = b$spk, svk = b$svk)
}}"""


def _bearing_cases() -> dict[str, Case]:
    return {f"bearing_{name}": Case(name, BEARING_R, _bearing(name), "bearing")
            for name in ("rand", "rand_na", "rect", "square", "ties", "positive", "tiny", "normforest",
                         "orelevation")}


def _spatial(name: str) -> Callable[[dict], dict]:
    def run(rasters):
        s, spec = surface(), rasters[name]
        z, g = spec.values, _grid(spec)
        return {"aacf": s.aacf(z, **g).values, "scl": s.scl(z, **g), "stxr": s.stxr(z, **g),
                "std": s.std(z, **g), "srw": s.srw(z, **g)}

    return run


SPATIAL_R = """list(aacf = geodiv::aacf(r), scl = fx$scl(r), stxr = fx$stxr(r), std = fx$std(r), srw = fx$srw(r))"""


def _spatial_cases() -> dict[str, Case]:
    return {f"spatial_{name}": Case(name, SPATIAL_R, _spatial(name), "spatial")
            for name in ("rand", "rand_na", "rect", "square", "ties", "positive", "normforest", "orelevation")}


def _fractal(name: str) -> Callable[[dict], dict]:
    def run(rasters):
        spec = rasters[name]
        return {"sfd": surface().sfd(spec.values, extent=spec.extent, crs=spec.crs)}

    return run


def _fractal_cases() -> dict[str, Case]:
    return {f"fractal_{name}": Case(name, "list(sfd = fx$sfd(r))", _fractal(name), "fractal")
            for name in ("rand", "square", "ties", "positive", "tiny", "normforest", "orforest", "orelevation")}


# window metrics: (R function of a window raster, names) per set
WINDOW_SETS = {
    "amplitude": ("function(w) c(geodiv::sa(w), fx$sq(w), fx$ssk(w), fx$sku(w), geodiv::sv(w), geodiv::sph(w), "
                  "geodiv::smean(w))", ("sa", "sq", "ssk", "sku", "sv", "sph", "smean")),
    "hybrid": ("function(w) c(fx$sdq(w), fx$sdq6(w), geodiv::sdr(w), geodiv::sds(w), fx$ssc(w), fx$s10z(w))",
               ("sdq", "sdq6", "sdr", "sds", "ssc", "s10z")),
    "bearing": ("function(w) { b <- fx$bearing(w); c(b$sbi, b$sci, b$svi, b$sk, b$spk, b$svk) }",
                ("sbi", "sci", "svi", "sk", "spk", "svk")),
    "spatial": ("function(w) { l <- fx$scl(w); c(l[1], l[1] / l[3], fx$std(w), fx$srw(w), fx$sfd(w)) }",
                ("sal", "str", "std", "stdi", "srw", "srwi", "shw", "sfd")),
}


def _window(name: str, which: str, size: int, shape: str) -> Callable[[dict], dict]:
    def run(rasters):
        spec = rasters[name]
        names = WINDOW_SETS[which][1]
        t = surface().window_metrics(spec.values, list(names), size, shape, extent=spec.extent, crs=spec.crs)
        return {n: t[n].values.ravel() for n in names}

    return run


def _window_cases() -> dict[str, Case]:
    specs = [("rand", "amplitude", 5, "square"), ("rand_na", "amplitude", 5, "circle"),
             ("rand", "hybrid", 7, "square"), ("rect", "hybrid", 7, "circle"),
             ("positive", "bearing", 5, "square"), ("positive", "spatial", 5, "square"),
             ("nf_small", "spatial", 7, "circle"), ("ties", "amplitude", 3, "square")]
    cases = {}
    for name, which, size, shape in specs:
        f, names = WINDOW_SETS[which]
        code = (f"fx$window(r, {size}, {'TRUE' if shape == 'circle' else 'FALSE'}, {f}, "
                f"c({', '.join(repr(n) for n in names)}))").replace("'", '"')
        cases[f"window_{which}_{name}_{size}_{shape}"] = Case(name, code, _window(name, which, size, shape), "window")
    return cases


TERRAIN_R = """{
  slp <- terra::terrain(r, v = "slope", unit = "degrees"); asp <- terra::terrain(r, v = "aspect", unit = "degrees")
  list(slope = slp, aspect = asp, slope4 = terra::terrain(r, v = "slope", neighbors = 4, unit = "radians"),
       aspect4 = terra::terrain(r, v = "aspect", neighbors = 4, unit = "radians"),
       tri = spatialEco::tri(r), tri5 = spatialEco::tri(r, 5), vrm = spatialEco::vrm(r), vrm5 = spatialEco::vrm(r, 5),
       srr = spatialEco::srr(r), dissection = spatialEco::dissection(r),
       spherical = spatialEco::spherical.sd(r, 3), spherical_var = spatialEco::spherical.sd(r, 5, variance = TRUE),
       tpi = fx$tpi(r, 5), tpi_norm = fx$tpi(r, 5, normalize = TRUE), tpi_circle = fx$tpi(r, CIRCLE, "circle"),
       planform = fx$curvature(r, "planform"), profile = fx$curvature(r, "profile"), total = fx$curvature(r, "total"),
       mcnab = fx$curvature(r, "mcnab"), bolstad = fx$curvature(r, "bolstad"),
       hsp = fx$hsp(r, 3, 11, 4), hsp_norm = fx$hsp(r, 3, 11, 4, normalize = TRUE),
       hsp_circle = fx$hsp(r, CIRCLE, CIRCLE2, CIRCLE, "circle"),
       trasp = fx$trasp(r), hli = fx$hli(asp, slp, 45), hli_direct = fx$hli(asp, slp, 45, direct = TRUE),
       hli_south = fx$hli(asp, slp, -30, hemisphere = "southern", equation = 2))
}"""
# vs spatialEco / terra's running sums, numpy's trigonometry: tolerances
TERRAIN_TOL = {k: (1e-11, 1e-12) for k in ("vrm", "vrm5", "srr", "spherical", "spherical_var", "trasp", "hli",
                                             "hli_direct", "hli_south")}
# focal means from compensated prefix sums, not window by window: ~1e-13 of the values' range
TERRAIN_TOL.update({k: (1e-12, 1e-12) for k in ("tpi", "tpi_norm", "tpi_circle", "hsp", "hsp_norm", "hsp_circle",
                                                "bolstad")})


def _terrain(name: str, circle: float) -> Callable[[dict], dict]:
    def run(rasters):
        s, spec = surface(), rasters[name]
        z, g = spec.values, _grid(spec)
        slp, asp = s.terrain(z, "slope", **g), s.terrain(z, "aspect", **g)
        out = {"slope": slp.values, "aspect": asp.values,
               "slope4": s.terrain(z, "slope", "radians", 4, **g).values,
               "aspect4": s.terrain(z, "aspect", "radians", 4, **g).values,
               "tri": s.tri(z, **g).values, "tri5": s.tri(z, 5, **g).values, "vrm": s.vrm(z, **g).values,
               "vrm5": s.vrm(z, 5, **g).values, "srr": s.srr(z, **g).values, "dissection": s.dissection(z, **g).values,
               "spherical": s.spherical_sd(z, 3, **g).values,
               "spherical_var": s.spherical_sd(z, 5, variance=True, **g).values,
               "tpi": s.tpi(z, 5, **g).values, "tpi_norm": s.tpi(z, 5, normalize=True, **g).values,
               "tpi_circle": s.tpi(z, circle, "circle", **g).values}
        for t in ("planform", "profile", "total", "mcnab", "bolstad"):
            out[t] = s.curvature(z, t, **g).values
        out["hsp"] = s.hsp(z, 3, 11, 4, **g).values
        out["hsp_norm"] = s.hsp(z, 3, 11, 4, normalize=True, **g).values
        out["hsp_circle"] = s.hsp(z, circle, 2 * circle, circle, "circle", **g).values
        out["trasp"] = s.trasp(z, **g).values
        out["hli"] = s.hli(asp, slp, 45).values
        out["hli_direct"] = s.hli(asp, slp, 45, direct=True).values
        out["hli_south"] = s.hli(asp, slp, -30, hemisphere="southern", equation=2).values
        return {k: v.ravel() for k, v in out.items()}

    return run


def _terrain_cases() -> dict[str, Case]:
    cases = {}
    for name, circle in (("dem", 75.0), ("dem_rect", 100.0), ("elev_small", 0.0007)):
        code = TERRAIN_R.replace("CIRCLE2", repr(2 * circle)).replace("CIRCLE", repr(circle))
        tol = dict(TERRAIN_TOL)
        if name == "elev_small":  # longitude/latitude: cell sizes from geodesics, which terra computes with the
            # UCRT's sin/cos (ls.cost's GeographicLib uses the x87 ones of the raster package): last bits
            tol.update({"slope": (1e-12, 1e-14), "slope4": (1e-12, 1e-14), "aspect": (1e-12, 1e-14, 360.0),
                        "aspect4": (1e-12, 1e-14, 2 * np.pi)})
        cases[f"terrain_{name}"] = Case(name, code, _terrain(name, circle), "terrain", tol)
    return cases


def _cases() -> dict[str, Case]:
    cases = {}
    cases.update(_amplitude_cases())
    cases.update(_plane_cases())
    cases.update(_hybrid_cases())
    cases.update(_features_cases())
    cases.update(_bearing_cases())
    cases.update(_spatial_cases())
    cases.update(_fractal_cases())
    cases.update(_window_cases())
    cases.update(_terrain_cases())
    return cases


CASES = _cases()
# Rasters a group's R code uses by name besides the cases' own.
EXTRA_RASTERS: dict[str, tuple[str, ...]] = {}
GROUPS = sorted({c.group for c in CASES.values()})
