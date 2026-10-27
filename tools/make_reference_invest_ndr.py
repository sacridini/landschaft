# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of InVEST's nutrient delivery ratio model (natcap.invest) for ls.services.ndr.

Usage::

    conda run -n invest-ref python tools/make_reference_invest_ndr.py [case ...]

Builds synthetic inputs (a DEM with hills, pits and flats, a land use map, a runoff
proxy, watersheds and a biophysical table), runs ``natcap.invest.ndr.ndr.execute`` on
them and stores per case the inputs and every raster InVEST writes in
``tests/data/reference/invest/<case>.npz``, with the settings, the watershed results and
the versions in ``<case>.json``, as ``tools/make_reference_invest.py`` does for the other
models (its helpers are imported from there).

Cases not marked ``pixels`` run InVEST with its subsurface NDR corrected: InVEST takes
the distance to the stream in cells (``dist_to_channel``) as if it were metres, against
the subsurface critical length in metres; the corrected form multiplies it by the cell
size. ``ndr_subsurface_pixels`` keeps InVEST's form, to show the difference.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_reference_invest import (
    ORIGIN,
    OUT,
    PIXEL,
    _csv,
    _dem,
    _lulc,
    _read,
    _read_polygons,
    _write_polygons,
    _write_raster,
)

CASES = {
    "ndr_mfd": dict(seed=41, shape=(61, 79), flow="mfd", n=True, p=True, threshold=60, k=2.0, av=None),
    "ndr_d8": dict(seed=42, shape=(61, 79), flow="d8", n=True, p=True, threshold=40, k=1.5, av=812.5),
    "ndr_p_only": dict(seed=43, shape=(53, 67), flow="mfd", n=False, p=True, threshold=50, k=2.0, av=None),
    "ndr_tall": dict(seed=44, shape=(283, 31), flow="mfd", n=True, p=False, threshold=45, k=2.0, av=None),
    "ndr_wide": dict(seed=45, shape=(29, 277), flow="d8", n=True, p=True, threshold=35, k=2.5, av=None),
    "ndr_subsurface_pixels": dict(seed=41, shape=(61, 79), flow="mfd", n=True, p=True, threshold=60, k=2.0,
                                  av=None, pixels=True),
}


def _patch_subsurface_ndr():
    """InVEST's subsurface NDR with the distance to the stream in metres (cells x cell size)."""
    import numpy
    import pygeoprocessing
    from natcap.invest.ndr import ndr

    def _calculate_sub_ndr(eff_sub, crit_len_sub, dist_to_channel_path, target_sub_ndr_path):
        cell = abs(pygeoprocessing.get_raster_info(dist_to_channel_path)["pixel_size"][0])
        pygeoprocessing.raster_map(
            op=lambda dist_to_channel: (1 - eff_sub * (1 - numpy.exp(-5 * (dist_to_channel * cell) / crit_len_sub))),
            rasters=[dist_to_channel_path], target_path=target_sub_ndr_path, target_nodata=ndr._TARGET_NODATA)

    original = ndr._calculate_sub_ndr
    ndr._calculate_sub_ndr = _calculate_sub_ndr
    return lambda: setattr(ndr, "_calculate_sub_ndr", original)


def run_ndr(name, spec, tmp):
    from natcap.invest.ndr import ndr
    from osgeo import gdal

    rng = np.random.default_rng(spec["seed"])
    ny, nx = spec["shape"]
    codes = [1, 2, 3, 4, 7, 9]
    dem = _dem(rng, ny, nx)
    nd = -9999.0
    dem[rng.random((ny, nx)) < 0.004] = nd
    lulc = _lulc(rng, ny, nx, codes, 0, holes=0.003).astype(np.int16)
    runoff = np.round(rng.uniform(300, 2200, size=(ny, nx)), 2).astype(np.float32)
    runoff[rng.random((ny, nx)) < 0.004] = nd
    _write_raster(tmp / "dem.tif", dem, nd, gdal.GDT_Float32)
    _write_raster(tmp / "lulc.tif", lulc, 0, gdal.GDT_Int16)
    _write_raster(tmp / "runoff.tif", runoff, nd, gdal.GDT_Float32)
    k = len(codes)
    table = {
        "load_n": np.round(rng.uniform(1, 40, size=k), 2),
        "eff_n": np.round(rng.uniform(0.05, 0.8, size=k), 3),
        "crit_len_n": np.round(rng.uniform(20, 300, size=k), 1),
        "proportion_subsurface_n": np.round(rng.uniform(0, 0.6, size=k), 3),
        "load_p": np.round(rng.uniform(0.1, 4, size=k), 3),
        "eff_p": np.round(rng.uniform(0.05, 0.8, size=k), 3),
        "crit_len_p": np.round(rng.uniform(20, 300, size=k), 1),
    }
    table["eff_n"][3] = 0.0           # an urban class retains nothing
    table["eff_p"][3] = 0.0
    table["crit_len_p"][2] = 0.0      # InVEST guards a critical length of 0
    table["proportion_subsurface_n"][0] = 0.0
    load_type_n = ["application-rate", "measured-runoff"] * (k // 2)
    load_type_p = ["measured-runoff", "application-rate", "application-rate"] * (k // 3)
    header = ["lucode", "load_n", "eff_n", "crit_len_n", "proportion_subsurface_n", "load_type_n", "load_p", "eff_p",
              "crit_len_p", "load_type_p"]
    rows = [[c, table["load_n"][i], table["eff_n"][i], table["crit_len_n"][i], table["proportion_subsurface_n"][i],
             load_type_n[i], table["load_p"][i], table["eff_p"][i], table["crit_len_p"][i], load_type_p[i]]
            for i, c in enumerate(codes)]
    _csv(tmp / "bio.csv", header, rows)
    # two watersheds covering the raster (and beyond), the first with a notch cut out: cells outside are masked
    x0, y0 = ORIGIN
    x1, y1 = x0 + nx * PIXEL, y0 - ny * PIXEL
    xm = x0 + (nx // 2) * PIXEL + 11.0
    ym = min(ny - 2, 20)
    ws_rings = [[(x0 - 40, y0 + 40), (xm, y0 + 40), (xm + 60, y1 - 40), (x0 - 40, y1 - 40),
                 (x0 - 40, y0 - ym * PIXEL), (x0 + min(9, nx // 3) * PIXEL, y0 - (ym - 5) * PIXEL),
                 (x0 - 40, y0 - (ym - 10) * PIXEL), (x0 - 40, y0 + 40)],
                [(xm, y0 + 40), (x1 + 40, y0 + 40), (x1 + 40, y1 - 40), (xm + 60, y1 - 40), (xm, y0 + 40)]]
    _write_polygons(tmp / "ws.shp", ws_rings, "ws_id", [1, 2])
    args = {"workspace_dir": str(tmp / "out"), "results_suffix": "", "dem_path": str(tmp / "dem.tif"),
            "lulc_path": str(tmp / "lulc.tif"), "runoff_proxy_path": str(tmp / "runoff.tif"),
            "watersheds_path": str(tmp / "ws.shp"), "biophysical_table_path": str(tmp / "bio.csv"),
            "calc_n": spec["n"], "calc_p": spec["p"], "threshold_flow_accumulation": spec["threshold"],
            "k_param": spec["k"], "flow_dir_algorithm": spec["flow"], "n_workers": -1}
    if spec["av"] is not None:
        args["runoff_proxy_av"] = spec["av"]
    if spec["n"]:
        args["subsurface_critical_length_n"] = 150.0
        args["subsurface_eff_n"] = 0.6
    restore = (lambda: None) if spec.get("pixels") else _patch_subsurface_ndr()
    try:
        registry = ndr.execute(args)
    finally:
        restore()
    inputs = {"dem": dem, "lulc": lulc, "runoff": runoff, "codes": np.asarray(codes),
              **{c: v for c, v in table.items()}}
    outputs = {}
    nodata_values = {}
    for key, path in registry.items():
        if key.startswith("aligned_") or key == "mask":  # copies of the inputs
            continue
        if isinstance(path, str) and path.endswith(".tif") and Path(path).is_file():
            a, ndv = _read(path)
            nodata_values[key] = ndv
            if a.dtype.kind == "f":
                outputs[key] = np.where(a == ndv, np.nan, a) if ndv is not None else a
            else:
                outputs[key] = a
    meta = {"model": "ndr", "nodata": 0, "input_nodata": nd, "pixel": PIXEL, "watersheds": ws_rings,
            "load_type_n": load_type_n, "load_type_p": load_type_p, "pixels": bool(spec.get("pixels")),
            "params": {k_: args.get(k_) for k_ in ("calc_n", "calc_p", "threshold_flow_accumulation", "k_param",
                                                    "flow_dir_algorithm", "runoff_proxy_av",
                                                    "subsurface_critical_length_n", "subsurface_eff_n")},
            "watershed_results": _read_polygons(registry["watershed_results_ndr"]), "nodata_values": nodata_values}
    return inputs, outputs, meta


def main(argv):
    import natcap.invest
    import numpy
    import pygeoprocessing

    OUT.mkdir(parents=True, exist_ok=True)
    for name in argv or list(CASES):
        with tempfile.TemporaryDirectory(prefix=f"invest-{name}-", ignore_cleanup_errors=True) as tmp:
            inputs, outputs, meta = run_ndr(name, CASES[name], Path(tmp))
        meta["versions"] = {"natcap.invest": natcap.invest.__version__, "pygeoprocessing": pygeoprocessing.__version__,
                            "numpy": numpy.__version__}
        arrays = {f"in/{k}": v for k, v in inputs.items()}
        arrays.update({f"out/{k}": v for k, v in outputs.items()})
        np.savez_compressed(OUT / f"{name}.npz", **arrays)
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(name, "->", sorted(outputs))


if __name__ == "__main__":
    main(sys.argv[1:])
