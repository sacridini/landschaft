# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of InVEST's seasonal water yield model for ls.services.

Usage::

    conda run --no-capture-output -n invest-ref python tools/make_reference_invest_swy.py [case ...]

Builds synthetic inputs (a DEM, land use, soil groups, twelve months of precipitation
and reference evapotranspiration, rain events or climate zones, a recharge map), runs
``execute(args)`` of natcap.invest's ``seasonal_water_yield`` on them and stores per
case the inputs and outputs in ``tests/data/reference/invest/<case>.npz`` and the
settings, tables, watershed results and versions in ``<case>.json``, as
``make_reference_invest.py`` does (its helpers are imported from there).

Cases marked ``patched`` run InVEST with its bugs corrected (``tools`` keeps InVEST's code,
with the fix):

* the cumulative upslope available recharge (``L_sum_avail``, eq. 103 of the User's
  Guide: ``sum_j p_ji (L_avail_j + L_sum_avail_j)``, ``p_ji`` the share of the flow of
  ``j`` that goes to ``i``) is, in ``swy.h``, the mean of the upslope terms weighted by
  the raw flow weights (``UpslopeNeighborsNoDivide`` and a division by their sum): a cell
  fed by three upslope cells gets their mean, not their sum. Fixed with
  ``UpslopeNeighbors`` (the shares) and without the division;
* the local recharge also visits a cell without elevation when a D8 direction points
  into it (pygeoprocessing drains to the holes): it gets a recharge (and enters the sum
  of Vri) and its own direction (128) indexes past the offset tables. Fixed by skipping
  the cells without a flow direction, as the seeds are;
* the baseflow counts the flow leaving the raster in the sum of the directions but adds
  nothing for it, so a cell draining off the edge has ``B_sum = 0`` instead of
  ``L_sum`` (eq. 108, the outlet), while one draining into a hole of the DEM (nodata in
  the stream map) gets ``L_sum``. Fixed by counting the flow off the raster as the flow
  into the holes;
* the baseflow reads the local and the available recharge of a cell without them
  (nodata: a recharge map with holes, or the cells at its edge once aligned to the DEM)
  as values (-9999, 3.4e38), which makes ``B`` and ``B_sum`` infinite or meaningless
  there and upslope. Fixed by counting no recharge there, as the accumulation of
  ``L_sum`` does;
* the curve numbers are written in the type of the land use and soil group rasters
  together, so with integer rasters (the usual case) fractional curve numbers of the
  table are truncated; here they are written as float64 then (the same S as InVEST's
  for integer curve numbers, which it divides in float64).

The first four are in InVEST's C++ (``swy.h``): its ``seasonal_water_yield_core.pyx`` is
rebuilt (Cython, MSVC) with the patched header and used in place of the installed one.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_reference_invest import (
    GRIDS,
    LOG,
    OUT,
    PIXEL,
    _boxes,
    _csv,
    _dem,
    _lulc,
    _read,
    _read_polygons,
    _set_grids,
    _write_polygons,
    _write_raster,
)

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

CASES = {
    # stock InVEST: the parts the bugs do not touch (curve numbers, quickflow, streams, routing)
    "swy_mfd": dict(seed=41, flow="mfd", dtype="float32", zones=False, monthly_alpha=False, user_l=False,
                    beta="1", gamma="1", alpha="1/12", threshold=50, frac_cn=False, patched=False),
    "swy_mfd_patched": dict(seed=41, flow="mfd", dtype="float32", zones=False, monthly_alpha=False, user_l=False,
                            beta="1", gamma="1", alpha="1/12", threshold=50, frac_cn=False, patched=True),
    "swy_d8_patched": dict(seed=42, flow="d8", dtype="float32", zones=True, monthly_alpha=True, user_l=False,
                           beta="0.7", gamma="0.4", alpha=None, threshold=35, frac_cn=False, patched=True),
    "swy_float64_patched": dict(seed=43, flow="mfd", dtype="float64", zones=False, monthly_alpha=False,
                                user_l=False, beta="0.9", gamma="0.8", alpha="0.1", threshold=60, frac_cn=True,
                                patched=True),
    "swy_user_recharge_patched": dict(seed=44, flow="mfd", dtype="float32", zones=False, monthly_alpha=False,
                                      user_l=True, beta="1", gamma="0.6", alpha="1/12", threshold=45,
                                      frac_cn=False, patched=True),
    "swy_user_recharge_d8_patched": dict(seed=45, flow="d8", dtype="float32", zones=False, monthly_alpha=False,
                                         user_l=True, beta="1", gamma="1", alpha="1/12", threshold=30,
                                         frac_cn=False, patched=True),
    # the land use at 45 m, the precipitation at 20 m over less, the evapotranspiration at 50 m, the soil
    # groups and climate zones shifted, the AOI 80 m inside the rasters
    "swy_grids_patched": dict(seed=46, flow="mfd", dtype="float32", zones=True, monthly_alpha=False, user_l=False,
                              beta="0.8", gamma="0.9", alpha="1/12", threshold=50, frac_cn=False, patched=True,
                              ws_pad=-80.0, grids={"lulc": (-29.0, 45.0, 33.0, 45.0), "soil": (4.0, 30.0, -6.0, 30.0),
                                                   "p*": (11.0, 20.0, -8.0, 20.0), "e*": (-50.0, 50.0, 50.0, 50.0),
                                                   "cz": (-13.0, 30.0, 9.0, 30.0)}),
    "swy_grids_recharge_patched": dict(seed=47, flow="d8", dtype="float32", zones=False, monthly_alpha=False,
                                       user_l=True, beta="1", gamma="1", alpha="1/12", threshold=40, frac_cn=False,
                                       patched=True, ws_pad=-60.0,
                                       grids={"l": (14.0, 20.0, -3.0, 20.0), "lulc": (8.0, 30.0, -5.0, 30.0)}),
}


# --------------------------------------------------------------------------- the patched extension

def _patched_core(build: Path):
    """InVEST's seasonal_water_yield_core rebuilt with the fixes of ``swy.h`` (see the module's doc)."""
    import natcap.invest.seasonal_water_yield as pkg

    src = Path(pkg.__file__).parent
    build.mkdir(parents=True, exist_ok=True)
    h = (src / "swy.h").read_text()
    fixes = [
        # L_sum_avail: the shares of the upslope flow, summed (eq. 103), not a weighted mean
        ("UpslopeNeighborsNoDivide<T>", "UpslopeNeighbors<T>", 2),
        (("              if (mfd_dir_sum > 0) {\n"
          "                l_sum_avail_i /= static_cast<float>(mfd_dir_sum);\n"
          "              }\n"), "", 1),
        # a cell without a flow direction reached by a D8 direction: not part of the model
        ("            work_queue.pop();\n",
         ("            work_queue.pop();\n"
          "            if (flow_dir_raster.get(xi, yi) == flow_dir_raster.nodata) {\n"
          "              continue;\n"
          "            }\n"), 1),
        # baseflow: a cell without local recharge (nodata) adds none, as in the accumulation of L_sum
        (("                l_j = l_raster.get(neighbor.x, neighbor.y);\n"
          "                l_avail_j = l_avail_raster.get(neighbor.x, neighbor.y);\n"),
         ("                l_j = l_raster.get(neighbor.x, neighbor.y);\n"
          "                l_avail_j = l_avail_raster.get(neighbor.x, neighbor.y);\n"
          "                if (is_close(l_j, l_raster.nodata)) {\n"
          "                  l_j = 0;\n"
          "                }\n"
          "                if (is_close(l_avail_j, l_avail_raster.nodata)) {\n"
          "                  l_avail_j = 0;\n"
          "                }\n"), 1),
        ("            l_i = l_raster.get(xi, yi);\n",
         ("            l_i = l_raster.get(xi, yi);\n"
          "            if (is_close(l_i, l_raster.nodata)) {\n"
          "              l_i = 0;\n"
          "            }\n"), 1),
        # baseflow: the flow off the raster as the flow into a hole (an outlet, eq. 108)
        (("                neighbor.y < 0 or neighbor.y >= flow_dir_raster.raster_y_size) {\n"
          "                continue;\n"),
         ("                neighbor.y < 0 or neighbor.y >= flow_dir_raster.raster_y_size) {\n"
          "                b_sum_i += neighbor.flow_proportion;\n"
          "                continue;\n"), 1),
    ]
    for old, new, count in fixes:
        assert h.count(old) == count, f"swy.h changed, review the patch: {old!r}"
        h = h.replace(old, new)
    (build / "swy.h").write_text(h)
    shutil.copy(src / "swy.pxd", build / "swy.pxd")
    pyx = (src / "seasonal_water_yield_core.pyx").read_text().replace("from .swy cimport", "from swy cimport")
    (build / "swy_core_patched.pyx").write_text(pyx)
    import numpy
    import pygeoprocessing

    pgp_ext = Path(pygeoprocessing.__file__).parent / "extensions"
    lib = Path(sys.prefix) / "Library"
    (build / "setup.py").write_text(f"""
from setuptools import setup, Extension
from Cython.Build import cythonize
ext = Extension("swy_core_patched", ["swy_core_patched.pyx"], language="c++",
                include_dirs=[{str(build)!r}, {str(pgp_ext)!r}, {numpy.get_include()!r}, {str(lib / 'include')!r}],
                library_dirs=[{str(lib / 'lib')!r}], libraries=["gdal"], extra_compile_args=["/std:c++20"])
setup(ext_modules=cythonize([ext], include_path=[{str(Path(pygeoprocessing.__file__).parent.parent)!r}],
                            compiler_directives={{"language_level": 3}}))
""")
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=build, check=True,
                   stdout=subprocess.DEVNULL)
    so = next(build.glob("swy_core_patched*.pyd"), None) or next(build.glob("swy_core_patched*.so"))
    spec = importlib.util.spec_from_file_location("swy_core_patched", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patched_cn(swy):
    """The curve numbers in float64 when the land use and soil groups are integer rasters."""
    import pygeoprocessing

    original_cn = swy._calculate_curve_number_raster

    def cn(lulc_path, soil_path, df, cn_path):
        original = pygeoprocessing.raster_map

        def raster_map(op, rasters, target_path, **kw):
            if np.issubdtype(pygeoprocessing.choose_dtype(*rasters), np.integer):
                kw["target_dtype"] = np.float64
            return original(op=op, rasters=rasters, target_path=target_path, **kw)

        pygeoprocessing.raster_map = raster_map
        try:
            original_cn(lulc_path, soil_path, df, cn_path)
        finally:
            pygeoprocessing.raster_map = original

    return cn


# --------------------------------------------------------------------------- a case

def run_swy(name, spec, tmp, core=None):
    from natcap.invest.seasonal_water_yield import seasonal_water_yield as swy
    from osgeo import gdal

    rng = np.random.default_rng(spec["seed"])
    ny, nx = 57, 73
    codes = [1, 2, 3, 4, 7]
    nd = -9999.0
    dem = _dem(rng, ny, nx)
    dem[rng.random((ny, nx)) < 0.003] = nd
    lulc = _lulc(rng, ny, nx, codes, 0, holes=0.004).astype(np.int16)
    soil = rng.integers(1, 5, size=(ny, nx)).astype(np.uint8)
    soil[rng.random((ny, nx)) < 0.004] = 255
    dtype = np.dtype(spec["dtype"])
    gtype = {"float32": gdal.GDT_Float32, "float64": gdal.GDT_Float64}[dtype.name]
    _write_raster(tmp / "dem.tif", dem, nd, gdal.GDT_Float32)
    _write_raster(tmp / "lulc.tif", lulc, 0, gdal.GDT_Int16)
    _write_raster(tmp / "soil.tif", soil, 255, gdal.GDT_Byte)
    inputs = {"dem": dem, "lulc": lulc, "soil": soil, "codes": np.asarray(codes)}
    # a dry and a wet season
    wet = 60 + 140 * (0.5 + 0.5 * np.cos(2 * np.pi * (np.arange(12) - 1) / 12))
    precip, et0 = [], []
    for m in range(12):
        p = np.round(rng.uniform(0.4, 1.6, size=(ny, nx)) * wet[m], 2).astype(dtype)
        p[rng.random((ny, nx)) < 0.02] = 0
        p[rng.random((ny, nx)) < 0.005] = nd
        e = np.round(rng.uniform(60, 160, size=(ny, nx)), 2).astype(dtype)
        e[rng.random((ny, nx)) < 0.005] = nd
        _write_raster(tmp / f"p{m + 1}.tif", p, nd, gtype)
        _write_raster(tmp / f"e{m + 1}.tif", e, nd, gtype)
        precip.append(p)
        et0.append(e)
    precip[6][:, :] = np.where(precip[6] == nd, nd, 0)  # a month without rain
    _write_raster(tmp / "p7.tif", precip[6], nd, gtype)
    inputs["precip"] = np.stack(precip)
    inputs["et0"] = np.stack(et0)
    _csv(tmp / "precip.csv", ["month", "path"], [[m + 1, str(tmp / f"p{m + 1}.tif")] for m in range(12)])
    _csv(tmp / "et0.csv", ["month", "path"], [[m + 1, str(tmp / f"e{m + 1}.tif")] for m in range(12)])
    cn = rng.integers(35, 98, size=(len(codes), 4)).astype(np.float64)
    if spec["frac_cn"]:
        cn = cn + np.round(rng.uniform(0, 1, size=cn.shape), 1)
    cn[2, 0] = 100.0  # S = 0: all the rain runs off
    kc = np.round(rng.uniform(0.3, 1.2, size=(len(codes), 12)), 3)
    header = ["lucode", "cn_a", "cn_b", "cn_c", "cn_d"] + [f"kc_{m}" for m in range(1, 13)]
    _csv(tmp / "bio.csv", header, [[c, *cn[i], *kc[i]] for i, c in enumerate(codes)])
    inputs["cn"] = cn
    inputs["kc"] = kc
    ws_rings = _boxes(ny, nx, 2, spec.get("ws_pad", 45.0))
    _write_polygons(tmp / "aoi.shp", ws_rings, "ws_id", [1, 2])
    args = {"workspace_dir": str(tmp / "out"), "results_suffix": "", "threshold_flow_accumulation": spec["threshold"],
            "et0_raster_table": str(tmp / "et0.csv"), "precip_raster_table": str(tmp / "precip.csv"),
            "dem_raster_path": str(tmp / "dem.tif"), "lulc_raster_path": str(tmp / "lulc.tif"),
            "soil_group_path": str(tmp / "soil.tif"), "aoi_path": str(tmp / "aoi.shp"),
            "biophysical_table_path": str(tmp / "bio.csv"), "beta_i": spec["beta"], "gamma": spec["gamma"],
            "user_defined_local_recharge": spec["user_l"], "user_defined_climate_zones": spec["zones"],
            "monthly_alpha": spec["monthly_alpha"], "flow_dir_algorithm": spec["flow"], "n_workers": -1}
    meta = {"model": "seasonal_water_yield", "input_nodata": nd, "pixel": PIXEL, "watersheds": ws_rings,
            "params": {k: spec[k] for k in ("flow", "beta", "gamma", "alpha", "threshold")}, "patched": spec["patched"]}
    if spec["alpha"] is not None:
        args["alpha_m"] = spec["alpha"]
    if spec["monthly_alpha"]:
        alpha = np.round(rng.uniform(0.02, 0.15, size=12), 4)
        _csv(tmp / "alpha.csv", ["month", "alpha"], [[m + 1, a] for m, a in enumerate(alpha)])
        args["monthly_alpha_path"] = str(tmp / "alpha.csv")
        inputs["alpha"] = alpha
    if spec["zones"]:
        cz = np.ones((ny, nx), np.int16)
        cz[:, nx // 3:] = 2
        cz[ny // 2:, 2 * nx // 3:] = 5
        cz[rng.random((ny, nx)) < 0.003] = -1
        _write_raster(tmp / "cz.tif", cz, -1, gdal.GDT_Int16)
        events = rng.integers(0, 16, size=(3, 12))
        events[:, 6] = 0
        _csv(tmp / "cz.csv", ["cz_id", *MONTHS], [[z, *events[i]] for i, z in enumerate([1, 2, 5])])
        args["climate_zone_raster_path"] = str(tmp / "cz.tif")
        args["climate_zone_table_path"] = str(tmp / "cz.csv")
        inputs["zones"] = cz
        inputs["zone_ids"] = np.array([1, 2, 5])
        inputs["zone_events"] = events
    else:
        events = rng.integers(1, 16, size=12)
        events[2] = 0
        _csv(tmp / "events.csv", ["month", "events"], [[m + 1, e] for m, e in enumerate(events)])
        args["rain_events_table_path"] = str(tmp / "events.csv")
        inputs["events"] = events
    if spec["user_l"]:
        rech = np.round(rng.uniform(-150, 600, size=(ny, nx)), 2).astype(np.float32)
        _write_raster(tmp / "l.tif", rech, nd, gdal.GDT_Float32)
        args["l_path"] = str(tmp / "l.tif")
        inputs["recharge"] = rech
    saved = swy.seasonal_water_yield_core, swy._calculate_curve_number_raster
    if spec["patched"]:
        swy.seasonal_water_yield_core = core
        swy._calculate_curve_number_raster = _patched_cn(swy)
    try:
        registry = swy.execute(args)
    finally:
        swy.seasonal_water_yield_core, swy._calculate_curve_number_raster = saved
    outputs, nodata_values = {}, {}

    def keep(key, path):
        if key.startswith(("prcp_a", "et0_a")) or key.endswith("_aligned"):
            return  # copies of the inputs
        a, ndv = _read(path)
        nodata_values[key] = ndv
        if a.dtype.kind == "f" and ndv is not None:
            a = np.where(np.isclose(a, ndv) if abs(ndv) > 1e30 else a == ndv, np.nan, a)
        outputs[key] = a

    for key, path in registry.items():
        if isinstance(path, dict):
            for m, p in sorted(path.items()):
                if isinstance(p, str) and p.endswith(".tif") and Path(p).is_file():
                    keep(f"{key.replace('[MONTH]', '')}{m}", p)
        elif isinstance(path, str) and path.endswith(".tif") and Path(path).is_file():
            keep(key, path)
    meta["nodata_values"] = nodata_values
    meta["watershed_results"] = _read_polygons(registry["aggregate_vector"])
    csv_path = registry.get("monthly_qf_table")
    if isinstance(csv_path, str) and Path(csv_path).is_file():
        import pandas as pd

        meta["monthly"] = pd.read_csv(csv_path).to_dict(orient="records")
    return inputs, outputs, meta


def main(argv):
    import natcap.invest
    import numpy
    import pygeoprocessing

    OUT.mkdir(parents=True, exist_ok=True)
    names = argv or list(CASES)
    with tempfile.TemporaryDirectory(prefix="invest-swy-build-", ignore_cleanup_errors=True) as build:
        core = _patched_core(Path(build)) if any(CASES[n]["patched"] for n in names) else None
        for name in names:
            spec = CASES[name]
            _set_grids(spec)
            with tempfile.TemporaryDirectory(prefix=f"invest-{name}-", ignore_cleanup_errors=True) as tmp:
                inputs, outputs, meta = run_swy(name, spec, Path(tmp), core)
            if GRIDS:
                meta.update(grids=dict(LOG["grids"]), out_gt=LOG["out_gt"])
            meta["versions"] = {"natcap.invest": natcap.invest.__version__,
                                "pygeoprocessing": pygeoprocessing.__version__, "numpy": numpy.__version__}
            arrays = {f"in/{k}": v for k, v in inputs.items()}
            arrays.update({f"out/{k}": v for k, v in outputs.items()})
            np.savez_compressed(OUT / f"{name}.npz", **arrays)
            (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
            print(name, "->", sorted(outputs))


if __name__ == "__main__":
    main(sys.argv[1:])
