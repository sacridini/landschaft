# SPDX-License-Identifier: GPL-3.0-or-later
"""What UDef-ARP (Clark Labs' tool of Verra's VT0007, https://github.com/Clark-Labs/UDef-ARP) gives, for landschaft's
tests (the same files as Terraflux's), written to tests/data/reference/redd/redd.npz and .json:

    python tools/make_reference_redd.py path/to/UDef-ARP

A jurisdiction of 140 x 160 cells of 30 m (four subdivisions, some cells outside), its forest at the start, the
deforestation of a calibration and of a confirmation period near the forest's edge, the distance to the edge and an
empirical vulnerability: UDef-ARP's own functions (vulnerability_map.py, allocation_tool.py) run on them, GDAL and
Qt replaced by stand-ins in memory: the negligible risk threshold, the benchmark and the alternative vulnerability
classes, the relative frequency table and the fitted density, and the allocation of the confirmation period (to
its deforestation) and of a validity period (to an expected area), with a subdivision without deforestation.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "data" / "reference" / "redd"

# -- stand-ins of GDAL and Qt, in memory ------------------------------------------------------------------------------

DATASETS: dict[str, Dataset] = {}
GT = (500000.0, 30.0, 0.0, 9000000.0, 0.0, -30.0)


class Band:
    def __init__(self, ds):
        self.ds = ds

    def ReadAsArray(self):
        return self.ds.array.copy()

    def GetMaximum(self):
        return None

    def WriteArray(self, a):
        self.ds.array = np.asarray(a).astype(self.ds.dtype)

    def SetNoDataValue(self, v):
        pass

    def FlushCache(self):
        pass


class Dataset:
    def __init__(self, array, dtype=None):
        self.array = array
        self.dtype = dtype or array.dtype
        self.RasterYSize, self.RasterXSize = array.shape

    def GetRasterBand(self, i):
        return Band(self)

    def GetGeoTransform(self):
        return GT

    def GetProjection(self):
        return ""

    def SetGeoTransform(self, gt):
        pass

    def SetProjection(self, p):
        pass

    def FlushCache(self):
        pass


class Driver:
    def Create(self, name, nx, ny, bands, dtype, options=None):
        ds = DATASETS[name] = Dataset(np.zeros((ny, nx)), dtype)
        return ds


gdal = types.ModuleType("osgeo.gdal")
gdal.UseExceptions = lambda: None
gdal.Open = lambda name, *a: DATASETS[name]
gdal.GetDriverByName = lambda name: Driver()
gdal.GDT_Int16, gdal.GDT_Float32 = np.int16, np.float32
osgeo = types.ModuleType("osgeo")
osgeo.gdal = gdal
qt = types.ModuleType("PyQt5.QtCore")


class QObject:
    def __init__(self, *a):
        pass


class Signal:
    def emit(self, *a):
        pass


qt.QObject = QObject
qt.pyqtSignal = lambda *a: Signal()
pyqt = types.ModuleType("PyQt5")
pyqt.QtCore = qt
sys.modules.update({"osgeo": osgeo, "osgeo.gdal": gdal, "PyQt5": pyqt, "PyQt5.QtCore": qt})


def put(name, a):
    DATASETS[name] = Dataset(np.asarray(a))
    return name


def main() -> None:
    udef = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(udef))
    from allocation_tool import AllocationTool
    from vulnerability_map import VulnerabilityMap

    ny, nx = 140, 160
    g = np.random.default_rng(17)
    field = ndimage.gaussian_filter(g.normal(size=(ny, nx)), 6)
    forest = (field > np.quantile(field, 0.45)).astype(np.int16)
    mask = np.ones((ny, nx), dtype=np.int16)
    mask[:12, :20] = 0
    mask[-8:, -30:] = 0
    rows, cols = np.mgrid[0:ny, 0:nx]
    regions = (1 + (rows >= 70) * 2 + (cols >= 80)).astype(np.int16)
    distance = ndimage.distance_transform_edt(forest) * 30.0  # metres to the nearest non-forest cell
    distance = (distance * forest).astype(np.float32)
    # deforestation near the edge, less in subdivision 4 (none at all in the confirmation)
    p = np.exp(-distance / 90.0) * 0.25 * forest * mask
    cal = (g.random((ny, nx)) < p).astype(np.int16)
    cnf = ((g.random((ny, nx)) < p) & (regions != 4) & (cal == 0)).astype(np.int16)
    vulnerability = (ndimage.gaussian_filter(p, 2) * forest).astype(np.float32)

    vm, at = VulnerabilityMap(), AllocationTool()
    put("distance", distance)
    put("cal", cal)
    put("mask", mask)
    put("forest", forest)
    put("regions", regions)
    put("cnf", cnf)
    put("vulnerability", vulnerability)
    nrt = vm.nrt_calculation("distance", "cal", "mask")
    benchmark = vm.geometric_classification("distance", nrt, 29, "mask").astype(np.int16)
    alternative = vm.geometric_classification_alternative("vulnerability", 29, "mask", "forest").astype(np.int16)
    put("risk", benchmark)

    # fit on the calibration
    bins = at.tabulation_bin_id_HRP("risk", "regions", "bins_fit")
    table = at.create_relative_frequency_table(bins, "cal", str(udef.parent / "redd_fit.csv"))
    at.create_fit_density_map("risk", bins.copy(), table.copy(), "density_fit")
    fit_density = DATASETS["density_fit"].array

    # the confirmation: subdivision 4 had no deforestation in the fit's table rows? (it had); a class of a region
    # missing from the fit is filled from the class's other regions
    csv = str(udef.parent / "redd_fit.csv")
    keep = table[~((table["ID"] // 1000 == 3) & (table["ID"] % 1000 == 4))]
    keep.to_csv(csv, index=False)
    at.tabulation_bin_id_VP("risk", "regions", "bins_cnf")
    ids, pre = at.check_modeling_region_ids(csv, "bins_cnf")
    csv_cnf = at.calculate_missing_bins_rf(ids, csv, pre) if ids.size else csv
    out = {}
    for name, target in [("cnf", "cnf"), ("vp", 400.0)]:
        bins_vp = at.tabulation_bin_id_VP("risk", "regions", f"bins_{name}")
        pd_arr = at.calculate_prediction_density_arr("risk", bins_vp, csv_cnf)
        ar = at.calculate_adjustment_ratio_cnf(pd_arr, target) if name == "cnf" else \
            at.calculate_adjustment_ratio(pd_arr, target)
        it, new = 0, None
        while ar > 1.00001 and it <= 50:
            new = at.adjusted_prediction_density_array(pd_arr, "risk", ar)
            ar = at.calculate_adjustment_ratio_cnf(new, target) if name == "cnf" else \
                at.calculate_adjustment_ratio(new, target)
            pd_arr = new
            it += 1
        at.adjusted_prediction_density_map(pd_arr, "risk", ar, f"density_{name}")
        out[name] = (DATASETS[f"density_{name}"].array, it, float(ar))

    # the assessment (model_evaluation.py's create_thiessen_polygon and create_plot, transcribed: shapely, geopandas
    # and rasterio in place of OGR; GDAL's rasterizer inside rasterio)
    import geopandas as gpd
    import rasterio.features as rfeat
    import shapely
    from affine import Affine
    from scipy.spatial import Voronoi

    grid_area = 81.0
    transform = Affine.from_gdal(*GT)
    shapes = [shapely.geometry.shape(s) for s, v in rfeat.shapes(mask, mask=mask == 1, transform=transform)]
    mask_df = gpd.GeoDataFrame(geometry=shapes)
    grid_size = int(np.sqrt(grid_area * 10000)) // int(GT[1])
    pts = [(GT[0] + x * GT[1], GT[3] + y * GT[5]) for y in range(-grid_size, ny + grid_size, grid_size)
           for x in range(-grid_size, nx + grid_size, grid_size)]
    vor = Voronoi(points=np.array(pts))
    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
    voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(shapely.ops.polygonize(lines)))
    cells = gpd.overlay(voronois, mask_df, how="intersection", keep_geom_type=False)
    cells["area"] = cells.area
    cells = cells[cells["area"] / cells["area"].max() > 0.999]
    density32 = out["cnf"][0].astype(np.float32)
    rows = []
    for geom in cells.geometry:
        sel = rfeat.rasterize([geom], out_shape=(ny, nx), transform=transform).astype(bool)
        actual = float(np.ma.MaskedArray(cnf, mask=(cnf == 0) | ~sel).sum() or 0) * (30 * 30 / 10000)
        predicted = float(np.ma.MaskedArray(density32, mask=(density32 == 0) | ~sel).sum() or 0)
        c = geom.centroid
        rows.append((c.x, c.y, actual, predicted))
    rows.sort(key=lambda r: (-r[1], r[0]))
    X = np.array([r[2] for r in rows], dtype=np.float32)
    Y = np.array([r[3] for r in rows], dtype=np.float32)
    dist = [abs(X[i] - Y[i]) for i in range(len(X))]
    from scipy import stats as sstats
    slope, intercept, _, _, _ = sstats.linregress(X, Y)
    ts_slope, ts_intercept, _, _ = sstats.theilslopes(Y, X)
    agree, union = np.sum([min(X[i], Y[i]) for i in range(len(X))]), np.sum([max(X[i], Y[i]) for i in range(len(X))])
    assess = {"grid_area": grid_area, "samples": len(X), "medae": float(np.median(dist)),
              "mae": float(np.sum(dist) / len(X)), "iou": float(0 if union == 0 else agree / union * 100),
              "agreement": float(agree), "difference": float(np.sum(dist)),
              "r2": float(np.corrcoef(X, Y)[0, 1] ** 2), "ols": [float(slope), float(intercept)],
              "theil_sen": [float(ts_slope), float(ts_intercept)]}

    arrays = {
        "assess/cells": np.array(rows, dtype=np.float64),
        "in/distance": distance.astype(np.float64), "in/cal": cal.astype(np.float64),
        "in/cnf": cnf.astype(np.float64), "in/mask": mask.astype(np.float64),
        "in/forest": forest.astype(np.float64), "in/regions": regions.astype(np.float64),
        "in/vulnerability": vulnerability.astype(np.float64),
        "benchmark": benchmark.astype(np.float64), "alternative": alternative.astype(np.float64),
        "fit/table": table.to_numpy(dtype=np.float64), "fit/density": fit_density.astype(np.float64),
        "cnf/density": out["cnf"][0].astype(np.float64), "vp/density": out["vp"][0].astype(np.float64),
        "missing": np.asarray(ids, dtype=np.float64),
    }
    meta = {"nrt": int(nrt), "classes": 29, "expected": 400.0, "max_iterations": 50,
            "iterations": {k: v[1] for k, v in out.items()}, "ar": {k: v[2] for k, v in out.items()},
            "dropped": [3004], "transform": list(GT), "assess": assess}
    REF.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF / "redd.npz", **arrays)
    (REF / "redd.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(REF / "redd.npz", meta)


if __name__ == "__main__":
    main()
