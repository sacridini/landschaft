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


def main(argv):
    import natcap.invest
    import numpy
    import pygeoprocessing

    OUT.mkdir(parents=True, exist_ok=True)
    cases = {**{k: ("carbon", v) for k, v in CARBON_CASES.items()}, **{k: ("hq", v) for k, v in HQ_CASES.items()}}
    names = argv or list(cases)
    for name in names:
        kind, spec = cases[name]
        with tempfile.TemporaryDirectory(prefix=f"invest-{name}-") as tmp:
            tmp = Path(tmp)
            inputs, outputs, meta = (run_carbon if kind == "carbon" else run_hq)(name, spec, tmp)
        meta["versions"] = {"natcap.invest": natcap.invest.__version__, "pygeoprocessing": pygeoprocessing.__version__,
                            "numpy": numpy.__version__}
        arrays = {f"in/{k}": v for k, v in inputs.items()}
        arrays.update({f"out/{k}": v for k, v in outputs.items()})
        np.savez_compressed(OUT / f"{name}.npz", **arrays)
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(name, "->", sorted(outputs))


if __name__ == "__main__":
    main(sys.argv[1:])
