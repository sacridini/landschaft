# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases shared by tools/make_reference_surface.py and the ls.surface tests.

Every raster is a float64 array (NaN = NA) with an extent and a CRS; R rebuilds
it with ``terra::rast(nrows, ncols, xmin, xmax, ymin, ymax, crs)``. Every case
runs R code on it (``r`` is the case's raster, the others by name) giving a
named list, and a Python function giving a dict with the same keys. Call the
geodiv functions as ``geodiv::`` (``terra::sds`` masks ``geodiv::sds``).
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
    return r


RASTERS = _rasters()


@dataclass
class Case:
    raster: str | None
    r: str  # R code giving a named list
    py: Callable[[dict[str, RasterSpec]], dict]
    group: str
    tolerance: float = 0.0  # 0: identical bits
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


AMPLITUDE_R = """list(sa = geodiv::sa(r), sq = geodiv::sq(r), ssk = geodiv::ssk(r),
     ssk_unadj = geodiv::ssk(r, adj = FALSE), sku = geodiv::sku(r), sku_raw = geodiv::sku(r, excess = FALSE),
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
        out["best"] = s.best_fit_plane(z, **g).values
        out["removed"] = s.remove_plane(z, **g).values
        return out

    return run


def _plane_r(orders=(0, 1, 2, 3)) -> str:
    fits = ", ".join(f"fit{k} = geodiv::fitplane(r, {k})" for k in orders)
    return f"list({fits}, best = geodiv::bestfitplane(r), removed = geodiv::remove_plane(r))"


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


HYBRID_R = """list(sdq = geodiv::sdq(r), sdq6 = geodiv::sdq6(r), surface_area = geodiv::surface_area(r),
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
        return {"peaks": _geodiv_points(s.find_peaks(z, **g), ny, nx),
                "valleys": _geodiv_points(s.find_valleys(z, **g), ny, nx),
                "sds": s.sds(z, **g), "ssc": s.ssc(z, **g), "s10z": s.s10z(z, **g)}

    return run


FEATURES_R = """list(peaks = geodiv::findpeaks(r), valleys = geodiv::findvalleys(r), sds = geodiv::sds(r),
     ssc = geodiv::ssc(r), s10z = geodiv::s10z(r))"""


def _features_cases() -> dict[str, Case]:
    return {f"features_{name}": Case(name, FEATURES_R, _features(name), "features")
            for name in ("rand", "rand_na", "rect", "square", "ties", "tiny", "normforest", "orelevation")}


def _cases() -> dict[str, Case]:
    cases = {}
    cases.update(_amplitude_cases())
    cases.update(_plane_cases())
    cases.update(_hybrid_cases())
    cases.update(_features_cases())
    return cases


CASES = _cases()
# Rasters a group's R code uses by name besides the cases' own.
EXTRA_RASTERS: dict[str, tuple[str, ...]] = {}
GROUPS = sorted({c.group for c in CASES.values()})
