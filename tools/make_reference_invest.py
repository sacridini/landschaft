# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of InVEST (natcap.invest) for ls.services.

Usage::

    <python of an env with natcap.invest> tools/make_reference_invest.py [case ...]

Builds synthetic inputs (land use maps, carbon pools, threats, sensitivities, an
access layer), runs ``execute(args)`` of InVEST's ``carbon`` and ``habitat_quality``
models on them, and stores per case the inputs and the outputs in
``tests/data/reference/invest/<case>.npz`` and the settings, tables and versions in
``<case>.json``. The test-suite reads the stored files, so it runs anywhere without
InVEST. It does not import landschaft: run it with the python of an env that has
natcap.invest (``conda create -n invest-ref -c conda-forge python=3.12 natcap.invest``).

``hq_*`` cases marked ``patched`` run InVEST with its distance transform corrected: a
threat with no pixel at all is infinitely far from every cell (InVEST's transform gives
distances of about the raster's half perimeter there, which a long enough maximum
distance turns into degradation).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "reference" / "invest"
PIXEL = 30.0
ORIGIN = (500000.0, 8000000.0)
EPSG = 31983  # SIRGAS 2000 / UTM 23S


def _write_raster(path, array, nodata, gdal_type):
    from osgeo import gdal, osr

    ny, nx = array.shape
    ds = gdal.GetDriverByName("GTiff").Create(str(path), nx, ny, 1, gdal_type)
    ds.SetGeoTransform((ORIGIN[0], PIXEL, 0.0, ORIGIN[1], 0.0, -PIXEL))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array)
    ds = None


def _read(path):
    from osgeo import gdal

    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    a = band.ReadAsArray()
    return a, band.GetNoDataValue()


def _lulc(rng, ny, nx, codes, nodata, holes=0.03):
    a = rng.choice(np.asarray(codes), size=(ny, nx)).astype(np.int32)
    # blocks of one class, as land use maps have
    for _ in range(12):
        y, x = rng.integers(0, ny - 8), rng.integers(0, nx - 8)
        h, w = rng.integers(3, 12), rng.integers(3, 15)
        a[y:y + h, x:x + w] = rng.choice(codes)
    a[rng.random((ny, nx)) < holes] = nodata
    return a


def _csv(path, header, rows):
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")


# --------------------------------------------------------------------------- carbon

CARBON_CASES = {
    "carbon_storage": dict(seed=1, alt=False, valuation=None),
    "carbon_sequestration": dict(seed=2, alt=True, valuation=None),
    "carbon_valuation": dict(seed=3, alt=True, valuation=dict(price=43.0, discount=7.0, change=2.5,
                                                               years=(2000, 2030))),
    "carbon_valuation_flat": dict(seed=4, alt=True, valuation=dict(price=12.5, discount=0.0, change=0.0,
                                                                    years=(2010, 2023))),
    # changes of exactly -1 t/ha, InVEST's nodata of the change map: run with that nodata patched to NaN
    "carbon_minus_one_patched": dict(seed=5, alt=True, valuation=dict(price=20.0, discount=3.0, change=1.0,
                                                                       years=(2000, 2010)), minus_one=True),
}


def run_carbon(name, spec, tmp):
    from osgeo import gdal

    from natcap.invest.carbon import carbon

    rng = np.random.default_rng(spec["seed"])
    codes = [1, 2, 3, 5, 8, 13]
    ny, nx = 57, 83
    nodata = -1
    bas = _lulc(rng, ny, nx, codes, nodata)
    pools = np.round(rng.uniform(0, 300, size=(len(codes), 4)), 3)
    pools[2] = 0.0
    pools[4, 1] = 17.1  # decimals that float32 cannot hold exactly
    if spec.get("minus_one"):
        pools = np.round(pools)
        pools[1] = pools[0]
        pools[1, 3] = pools[0, 3] - 1.0  # class 2 stores 1 t/ha less than class 1
        carbon._CARBON_NODATA = float("nan")
    _csv(tmp / "pools.csv", ["lucode", "c_above", "c_below", "c_soil", "c_dead"],
         [[c, *p] for c, p in zip(codes, pools)])
    _write_raster(tmp / "bas.tif", bas, nodata, gdal.GDT_Int32)
    args = {"workspace_dir": str(tmp / "ws"), "results_suffix": "", "lulc_bas_path": str(tmp / "bas.tif"),
            "carbon_pools_path": str(tmp / "pools.csv"), "calc_sequestration": False, "do_valuation": False,
            "lulc_alt_path": "", "n_workers": -1}
    inputs = {"bas": bas, "pools": pools, "codes": np.asarray(codes)}
    if spec["alt"]:
        alt = bas.copy()
        change = rng.random((ny, nx)) < 0.3
        alt[change] = rng.choice(codes, size=int(change.sum()))
        alt[(bas == nodata)] = nodata
        alt[rng.random((ny, nx)) < 0.01] = nodata
        _write_raster(tmp / "alt.tif", alt, nodata, gdal.GDT_Int32)
        args.update(calc_sequestration=True, lulc_alt_path=str(tmp / "alt.tif"))
        inputs["alt"] = alt
    v = spec["valuation"]
    if v:
        args.update(do_valuation=True, price_per_metric_ton_of_c=v["price"], discount_rate=v["discount"],
                    rate_change=v["change"], lulc_bas_year=v["years"][0], lulc_alt_year=v["years"][1])
    registry = carbon.execute(args)
    outputs = {}
    for key, path in registry.items():
        if isinstance(path, str) and path.endswith(".tif") and Path(path).is_file():
            a, nd = _read(path)
            outputs[key] = np.where(a == nd, np.nan, a) if nd is not None else a
    summary = []
    with open(registry["summary_csv"]) as f:
        next(f)
        for line in f:
            label, total, units, filename = line.rstrip("\n").split(",")
            summary.append({"label": label, "total": float(total), "units": units, "filename": filename})
    meta = {"model": "carbon", "nodata": nodata, "pixel": PIXEL, "valuation": v, "summary": summary,
            "outputs": sorted(outputs), "patched": bool(spec.get("minus_one"))}
    carbon._CARBON_NODATA = -1.0
    return inputs, outputs, meta


# --------------------------------------------------------------------------- habitat quality

HQ_CASES = {
    "hq_current": dict(seed=11, fut=False, bas=False, access=False, empty=False, k=0.05),
    "hq_scenarios": dict(seed=12, fut=True, bas=True, access=False, empty=False, k=0.3),
    "hq_access": dict(seed=13, fut=False, bas=False, access=True, empty=False, k=0.1),
    # a threat without cells, with a maximum distance beyond the raster's half perimeter (64 + 77 cells)
    "hq_empty_threat": dict(seed=14, fut=False, bas=False, access=False, empty=True, k=0.05, urban_dist=6000.0),
    "hq_empty_threat_patched": dict(seed=14, fut=False, bas=False, access=False, empty=True, k=0.05,
                                    urban_dist=6000.0, patched=True),
}

THREATS = [("crops", 600.0, 0.7, "linear"), ("roads", 1000.0, 1.0, "exponential"),
           ("urban", 270.0, 0.35, "linear")]


def _threat_map(rng, ny, nx, density, nodata=None):
    t = np.zeros((ny, nx), dtype=np.float32)
    on = rng.random((ny, nx)) < density
    t[on] = np.round(rng.uniform(0.05, 1.0, size=int(on.sum())), 2)
    if nodata is not None:
        t[rng.random((ny, nx)) < 0.01] = nodata
    return t


def run_hq(name, spec, tmp):
    from osgeo import gdal, ogr, osr

    from natcap.invest.habitat_quality import habitat_quality

    rng = np.random.default_rng(spec["seed"])
    THREATS = [(t, spec.get("urban_dist", d) if t == "urban" else d, w, c) for t, d, w, c in globals()["THREATS"]]
    codes = [1, 2, 4, 7, 9, 12]
    ny, nx = 64, 77
    nodata = 255
    cur = _lulc(rng, ny, nx, codes, nodata).astype(np.uint8)
    _write_raster(tmp / "cur.tif", cur, nodata, gdal.GDT_Byte)
    habitat = np.round(rng.uniform(0, 1, size=len(codes)), 2)
    habitat[0] = 0.0
    habitat[3] = 1.0
    sens = np.round(rng.uniform(0, 1, size=(len(codes), len(THREATS))), 2)
    sens[0] = 0.0
    _csv(tmp / "sens.csv", ["lucode", "name", "habitat", *[t[0] for t in THREATS]],
         [[c, f"class{c}", h, *s] for c, h, s in zip(codes, habitat, sens)])
    inputs = {"cur": cur, "habitat": habitat, "sensitivity": sens, "codes": np.asarray(codes)}
    scenarios = ["cur"] + (["fut"] if spec["fut"] else []) + (["bas"] if spec["bas"] else [])
    rows = []
    for k, (tname, dist, weight, decay) in enumerate(THREATS):
        paths = {}
        for sc in [s for s in scenarios if s != "bas"]:
            density = [0.02, 0.05, 0.004][k] * (1.5 if sc == "fut" else 0.7 if sc == "bas" else 1.0)
            t = _threat_map(rng, ny, nx, density, nodata=-9.0 if k == 1 else None)
            if spec["empty"] and tname == "urban":
                t[:] = 0.0
            p = tmp / f"{tname}_{sc}.tif"
            _write_raster(p, t, -9.0 if k == 1 else None, gdal.GDT_Float32)
            paths[sc] = p.name
            inputs[f"threat_{tname}_{sc}"] = t
        # no base threats: with them InVEST 3.20.2 stops (KeyError 'deg_sum_b', an output it never registers)
        rows.append([tname, dist, weight, decay, paths["cur"], paths.get("fut", ""), ""])
    _csv(tmp / "threats.csv", ["threat", "max_dist", "weight", "decay", "cur_path", "fut_path", "base_path"],
         rows)
    args = {"workspace_dir": str(tmp / "ws"), "results_suffix": "", "lulc_cur_path": str(tmp / "cur.tif"),
            "threats_table_path": str(tmp / "threats.csv"), "sensitivity_table_path": str(tmp / "sens.csv"),
            "half_saturation_constant": spec["k"], "access_vector_path": "", "lulc_fut_path": "",
            "lulc_bas_path": "", "n_workers": -1}
    for sc in scenarios[1:]:
        lulc = cur.copy()
        change = rng.random((ny, nx)) < (0.25 if sc == "fut" else 0.15)
        lulc[change] = rng.choice(codes, size=int(change.sum()))
        if sc == "fut":
            lulc[(rng.random((ny, nx)) < 0.02)] = 3  # a class the baseline lacks (rarity 0)
        lulc[cur == nodata] = nodata
        if sc == "fut":
            inputs["sensitivity_extra"] = np.array([3, 0.6, 0.2, 0.4, 0.9])
        _write_raster(tmp / f"{sc}.tif", lulc, nodata, gdal.GDT_Byte)
        args[f"lulc_{sc}_path"] = str(tmp / f"{sc}.tif")
        inputs[sc] = lulc
    if "sensitivity_extra" in inputs:
        extra = inputs["sensitivity_extra"]
        with open(tmp / "sens.csv", "a", newline="") as f:
            f.write(",".join([str(int(extra[0])), "class3", *[str(v) for v in extra[1:]]]) + "\n")
    polygons = []
    if spec["access"]:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(EPSG)
        drv = ogr.GetDriverByName("GPKG")
        ds = drv.CreateDataSource(str(tmp / "access.gpkg"))
        layer = ds.CreateLayer("access", srs, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("ACCESS", ogr.OFTReal))
        x0, y0 = ORIGIN
        for (cx, cy, w, h, a) in [(10, 8, 25, 20, 0.0), (40, 30, 30, 25, 0.4), (5, 40, 18, 15, 0.75)]:
            xa, xb = x0 + cx * PIXEL + 7.0, x0 + (cx + w) * PIXEL - 4.0
            ya, yb = y0 - cy * PIXEL - 5.0, y0 - (cy + h) * PIXEL + 9.0
            ring = [(xa, ya), (xb, ya + 13.0), (xb, yb), (xa + 20.0, yb), (xa, ya)]
            polygons.append({"ring": ring, "access": a})
            g = ogr.Geometry(ogr.wkbLinearRing)
            for x, y in ring:
                g.AddPoint_2D(x, y)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(g)
            feat = ogr.Feature(layer.GetLayerDefn())
            feat.SetField("ACCESS", a)
            feat.SetGeometry(poly)
            layer.CreateFeature(feat)
        ds = None
        args["access_vector_path"] = str(tmp / "access.gpkg")
    if spec.get("patched"):
        _patch_distance_transform()
    registry = habitat_quality.execute(args)
    outputs = {}
    for key, path in registry.items():
        if isinstance(path, dict):
            for sub, p in path.items():
                if isinstance(p, str) and p.endswith(".tif") and key.startswith("degradation_"):
                    a, nd = _read(p)
                    outputs[f"{key}/{sub}"] = np.where(a == nd, np.nan, a) if nd is not None else a
            continue
        if isinstance(path, str) and path.endswith(".tif") and Path(path).is_file() and (
                key.startswith(("deg_sum", "quality", "rarity", "access_layer"))):
            a, nd = _read(path)
            outputs[key] = np.where(a == nd, np.nan, a) if nd is not None else a
    rarity = {}
    for key, path in registry.items():
        if isinstance(path, str) and key.startswith("rarity") and path.endswith(".csv"):
            with open(path) as f:
                next(f)
                rarity[key] = [[float(v) for v in line.strip().split(",")] for line in f if line.strip()]
    meta = {"model": "habitat_quality", "nodata": nodata, "pixel": PIXEL, "k": spec["k"],
            "threats": [{"threat": t, "max_dist": d, "weight": w, "decay": c} for t, d, w, c in THREATS],
            "threat_nodata": {"roads": -9.0}, "scenarios": scenarios, "polygons": polygons, "rarity": rarity,
            "patched": bool(spec.get("patched")), "outputs": sorted(outputs), "registry": sorted(
                k if not isinstance(v, dict) else f"{k}/*" for k, v in registry.items())}
    return inputs, outputs, meta


def _patch_distance_transform():
    """InVEST's transform with the correct distance of a threat without pixels: infinite."""
    import pygeoprocessing
    from osgeo import gdal

    from natcap.invest.habitat_quality import habitat_quality as hq

    original = pygeoprocessing.distance_transform_edt

    def fixed(base, target, *a, **kw):
        original(base, target, *a, **kw)
        arr, nd = _read(base[0])
        region = (arr != 0) if nd is None else (~np.isclose(arr, nd) & (arr != 0))
        if not region.any():
            ds = gdal.Open(str(target), gdal.GA_Update)
            band = ds.GetRasterBand(1)
            band.WriteArray(np.full(arr.shape, np.finfo(np.float32).max, dtype=np.float32))
            ds = None

    hq.pygeoprocessing.distance_transform_edt = fixed


# --------------------------------------------------------------------------- annual water yield

WY_CASES = {
    "wy_basic": dict(seed=21, dtype="float32", demand=False, sub=False, valuation=False),
    "wy_full": dict(seed=22, dtype="float32", demand=True, sub=True, valuation=True),
    "wy_int_inputs": dict(seed=23, dtype="int16", demand=True, sub=False, valuation=False),
    "wy_float64": dict(seed=24, dtype="float64", demand=True, sub=True, valuation=True),
}


def _boxes(ny, nx, parts, pad):
    """Polygons covering the raster (and ``pad`` metres beyond), cut in ``parts`` columns,
    with a slanted edge so that cells are split by the boundaries."""
    x0, y0 = ORIGIN
    x1, y1 = x0 + nx * PIXEL, y0 - ny * PIXEL
    xs = np.linspace(x0 - pad, x1 + pad, parts + 1)
    out = []
    for k in range(parts):
        xa, xb = xs[k], xs[k + 1]
        tilt = 37.0 if k else 0.0
        tilt_b = 37.0 if k < parts - 1 else 0.0
        out.append([(xa - tilt, y0 + pad), (xb - tilt_b, y0 + pad), (xb + tilt_b, y1 - pad), (xa + tilt, y1 - pad),
                    (xa - tilt, y0 + pad)])
    return out


def _write_polygons(path, rings, field, ids):
    from osgeo import ogr, osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(str(path))
    layer = ds.CreateLayer(Path(path).stem, srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn(field, ogr.OFTInteger))
    for ring, i in zip(rings, ids):
        g = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in ring:
            g.AddPoint_2D(x, y)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(g)
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField(field, int(i))
        feat.SetGeometry(poly)
        layer.CreateFeature(feat)
    ds = None


def _read_polygons(path):
    from osgeo import ogr

    ds = ogr.Open(str(path))
    layer = ds.GetLayer()
    defn = layer.GetLayerDefn()
    names = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
    rows = []
    for feat in layer:
        rows.append({n: feat.GetField(n) for n in names})
    return rows


def run_wy(name, spec, tmp):
    from osgeo import gdal

    from natcap.invest.annual_water_yield import annual_water_yield as awy

    rng = np.random.default_rng(spec["seed"])
    codes = [1, 2, 3, 5, 6, 8]
    ny, nx = 53, 71
    nodata = 0
    lulc = _lulc(rng, ny, nx, codes, nodata).astype(np.int16)
    dtype = np.dtype(spec["dtype"])
    gtype = {"float32": gdal.GDT_Float32, "float64": gdal.GDT_Float64, "int16": gdal.GDT_Int16}[dtype.name]
    nd = -9999

    def field(lo, hi, decimals, holes=0.01):
        a = rng.uniform(lo, hi, size=(ny, nx))
        a = np.round(a, decimals if dtype.kind == "f" else 0).astype(dtype)
        a[rng.random((ny, nx)) < holes] = nd
        return a

    precip = field(300, 2500, 2)
    precip[rng.random((ny, nx)) < 0.01] = 0  # InVEST leaves cells without rain out of the balance
    eto = field(600, 1800, 2)
    depth = field(200, 3000, 1)
    pawc = np.round(rng.uniform(0.05, 0.4, size=(ny, nx)), 3).astype(np.float32 if dtype.name != "float64"
                                                                     else np.float64)
    pawc[rng.random((ny, nx)) < 0.01] = nd
    _write_raster(tmp / "lulc.tif", lulc, nodata, gdal.GDT_Int16)
    for k, a in (("precip", precip), ("eto", eto), ("depth", depth)):
        _write_raster(tmp / f"{k}.tif", a, nd, gtype)
    _write_raster(tmp / "pawc.tif", pawc, nd, gdal.GDT_Float64 if pawc.dtype == np.float64 else gdal.GDT_Float32)
    veg = np.array([1, 1, 0, 1, 0, 1])
    root = np.round(rng.uniform(300, 7000, size=len(codes)), 1)
    kc = np.round(rng.uniform(0.2, 1.2, size=len(codes)), 3)
    kc[2] = 1.05  # open water
    _csv(tmp / "bio.csv", ["lucode", "lulc_veg", "root_depth", "kc"], [[c, v, r, k] for c, v, r, k in
                                                                       zip(codes, veg, root, kc)])
    ws_rings = _boxes(ny, nx, 2, 45.0)
    _write_polygons(tmp / "ws.shp", ws_rings, "ws_id", [1, 2])
    args = {"workspace_dir": str(tmp / "ws_out"), "results_suffix": "", "lulc_path": str(tmp / "lulc.tif"),
            "depth_to_root_rest_layer_path": str(tmp / "depth.tif"), "precipitation_path": str(tmp / "precip.tif"),
            "pawc_path": str(tmp / "pawc.tif"), "eto_path": str(tmp / "eto.tif"), "watersheds_path": str(tmp / "ws.shp"),
            "biophysical_table_path": str(tmp / "bio.csv"), "seasonality_constant": 7.5, "sub_watersheds_path": "",
            "demand_table_path": "", "valuation_table_path": "", "n_workers": -1}
    inputs = {"lulc": lulc, "precip": precip, "eto": eto, "depth": depth, "pawc": pawc, "codes": np.asarray(codes),
              "veg": veg, "root": root, "kc": kc}
    meta = {"model": "annual_water_yield", "nodata": nodata, "input_nodata": nd, "pixel": PIXEL, "z": 7.5,
            "watersheds": ws_rings}
    sub_rings = None
    if spec["sub"]:
        sub_rings = _boxes(ny, nx, 4, 45.0)
        _write_polygons(tmp / "sub.shp", sub_rings, "subws_id", [10, 11, 12, 13])
        args["sub_watersheds_path"] = str(tmp / "sub.shp")
        meta["subwatersheds"] = sub_rings
    if spec["demand"]:
        demand = np.round(rng.uniform(0, 3, size=len(codes)), 2)
        demand[0] = 0.0
        _csv(tmp / "demand.csv", ["lucode", "demand"], [[c, d] for c, d in zip(codes, demand)])
        args["demand_table_path"] = str(tmp / "demand.csv")
        inputs["demand"] = demand
    if spec["valuation"]:
        val = [[1, 0.8, 0.6, 200.0, 0.07, 1000.0, 100, 5.0],
               [2, 0.85, 0.7, 150.0, 0.05, 2000.0, 30, 0.0]]  # a discount rate of 0
        _csv(tmp / "val.csv", ["ws_id", "efficiency", "fraction", "height", "kw_price", "cost", "time_span",
                               "discount"], val)
        args["valuation_table_path"] = str(tmp / "val.csv")
        meta["valuation"] = val
    registry = awy.execute(args)
    outputs = {}
    for key in ("fractp", "aet", "wyield", "pet", "demand"):
        path = registry.get(key)
        if isinstance(path, str) and Path(path).is_file():
            a, ndv = _read(path)
            outputs[key] = np.where(a == ndv, np.nan, a) if ndv is not None else a
    meta["watershed_results"] = _read_polygons(registry["watershed_results_wyield"])
    if spec["sub"]:
        meta["subwatershed_results"] = _read_polygons(registry["subwatershed_results_wyield"])
    return inputs, outputs, meta


# --------------------------------------------------------------------------- sediment delivery ratio

SDR_CASES = {
    "sdr_mfd": dict(seed=31, flow="mfd", drainage=False, threshold=60),
    "sdr_d8": dict(seed=32, flow="d8", drainage=False, threshold=40),
    "sdr_drainage": dict(seed=33, flow="mfd", drainage=True, threshold=80),
}


def _dem(rng, ny, nx):
    """Hills and valleys with some noise, a few pits and flats (float32, metres)."""
    y, x = np.mgrid[:ny, :nx].astype(np.float64)
    z = (300 + 40 * np.sin(x / 9.0) * np.cos(y / 11.0) + 0.9 * x + 0.5 * y
         + 15 * np.exp(-((x - nx * 0.6) ** 2 + (y - ny * 0.4) ** 2) / 60.0))
    z += rng.normal(0, 0.8, size=(ny, nx))
    z[20:24, 30:35] = z[20:24, 30:35].min() - 2.0          # a pit
    z[40:44, 10:16] = np.round(z[40:44, 10:16].mean(), 1)   # a flat
    return np.round(z, 2).astype(np.float32)


def run_sdr(name, spec, tmp):
    from osgeo import gdal

    from natcap.invest.sdr import sdr

    rng = np.random.default_rng(spec["seed"])
    ny, nx = 61, 79
    codes = [1, 2, 3, 4, 7]
    dem = _dem(rng, ny, nx)
    nd = -9999.0
    dem[rng.random((ny, nx)) < 0.004] = nd
    lulc = _lulc(rng, ny, nx, codes, 0, holes=0.002).astype(np.int16)
    erosivity = np.round(rng.uniform(2000, 6000, size=(ny, nx)), 1).astype(np.float32)
    erodibility = np.round(rng.uniform(0.01, 0.05, size=(ny, nx)), 4).astype(np.float32)
    erosivity[rng.random((ny, nx)) < 0.003] = nd
    _write_raster(tmp / "dem.tif", dem, nd, gdal.GDT_Float32)
    _write_raster(tmp / "lulc.tif", lulc, 0, gdal.GDT_Int16)
    _write_raster(tmp / "erosivity.tif", erosivity, nd, gdal.GDT_Float32)
    _write_raster(tmp / "erodibility.tif", erodibility, nd, gdal.GDT_Float32)
    usle_c = np.round(rng.uniform(0.001, 0.4, size=len(codes)), 4)
    usle_c[0] = 0.0005  # below W's floor of 0.001
    usle_p = np.round(rng.uniform(0.3, 1.0, size=len(codes)), 3)
    _csv(tmp / "bio.csv", ["lucode", "usle_c", "usle_p"], [[c, a, b] for c, a, b in zip(codes, usle_c, usle_p)])
    # two watersheds covering the raster (and beyond), the first with a notch cut out: cells outside are masked
    x0, y0 = ORIGIN
    x1, y1 = x0 + nx * PIXEL, y0 - ny * PIXEL
    xm = x0 + 37 * PIXEL + 11.0
    ws_rings = [[(x0 - 40, y0 + 40), (xm, y0 + 40), (xm + 60, y1 - 40), (x0 - 40, y1 - 40),
                 (x0 - 40, y0 - 20 * PIXEL), (x0 + 9 * PIXEL, y0 - 15 * PIXEL), (x0 - 40, y0 - 10 * PIXEL),
                 (x0 - 40, y0 + 40)],
                [(xm, y0 + 40), (x1 + 40, y0 + 40), (x1 + 40, y1 - 40), (xm + 60, y1 - 40), (xm, y0 + 40)]]
    _write_polygons(tmp / "ws.shp", ws_rings, "ws_id", [1, 2])
    args = {"workspace_dir": str(tmp / "out"), "results_suffix": "", "dem_path": str(tmp / "dem.tif"),
            "erosivity_path": str(tmp / "erosivity.tif"), "erodibility_path": str(tmp / "erodibility.tif"),
            "lulc_path": str(tmp / "lulc.tif"), "watersheds_path": str(tmp / "ws.shp"),
            "biophysical_table_path": str(tmp / "bio.csv"), "threshold_flow_accumulation": spec["threshold"],
            "k_param": 2.0, "sdr_max": 0.8, "ic_0_param": 0.5, "l_max": 122, "drainage_path": "",
            "flow_dir_algorithm": spec["flow"], "n_workers": -1}
    inputs = {"dem": dem, "lulc": lulc, "erosivity": erosivity, "erodibility": erodibility,
              "codes": np.asarray(codes), "usle_c": usle_c, "usle_p": usle_p}
    if spec["drainage"]:
        drainage = np.zeros((ny, nx), np.uint8)
        drainage[:, 60:62] = 1
        drainage[50:52, :] = 1
        _write_raster(tmp / "drainage.tif", drainage, 255, gdal.GDT_Byte)
        args["drainage_path"] = str(tmp / "drainage.tif")
        inputs["drainage"] = drainage
    registry = sdr.execute(args)
    outputs = {}
    for key, path in registry.items():
        if isinstance(path, str) and path.endswith(".tif") and Path(path).is_file():
            a, ndv = _read(path)
            if a.dtype.kind == "f":
                outputs[key] = np.where(a == ndv, np.nan, a) if ndv is not None else a
            else:
                outputs[key] = a
    meta = {"model": "sdr", "nodata": 0, "input_nodata": nd, "pixel": PIXEL, "watersheds": ws_rings,
            "params": {k: args[k] for k in ("threshold_flow_accumulation", "k_param", "sdr_max", "ic_0_param",
                                            "l_max", "flow_dir_algorithm")},
            "watershed_results": _read_polygons(registry["watershed_results_sdr"]),
            "nodata_values": {key: _read(path)[1] for key, path in registry.items()
                              if isinstance(path, str) and path.endswith(".tif") and Path(path).is_file()}}
    return inputs, outputs, meta


def main(argv):
    import natcap.invest
    import numpy
    import pygeoprocessing

    OUT.mkdir(parents=True, exist_ok=True)
    cases = {**{k: ("carbon", v) for k, v in CARBON_CASES.items()}, **{k: ("hq", v) for k, v in HQ_CASES.items()},
             **{k: ("wy", v) for k, v in WY_CASES.items()}, **{k: ("sdr", v) for k, v in SDR_CASES.items()}}
    runners = {"carbon": run_carbon, "hq": run_hq, "wy": run_wy, "sdr": run_sdr}
    names = argv or list(cases)
    for name in names:
        kind, spec = cases[name]
        with tempfile.TemporaryDirectory(prefix=f"invest-{name}-", ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            inputs, outputs, meta = runners[kind](name, spec, tmp)
        meta["versions"] = {"natcap.invest": natcap.invest.__version__, "pygeoprocessing": pygeoprocessing.__version__,
                            "numpy": numpy.__version__}
        arrays = {f"in/{k}": v for k, v in inputs.items()}
        arrays.update({f"out/{k}": v for k, v in outputs.items()})
        np.savez_compressed(OUT / f"{name}.npz", **arrays)
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(name, "->", sorted(outputs))


if __name__ == "__main__":
    main(sys.argv[1:])
