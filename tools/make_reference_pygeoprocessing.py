# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of pygeoprocessing's routing for ls.hydro (the InVEST models' routing).

Usage (with a Python that has natcap.invest / pygeoprocessing, not landschaft)::

    <invest python> tools/make_reference_pygeoprocessing.py

Reads the DEMs stored by ``tools/make_reference_hydro.py``
(``tests/data/reference/hydro/whitebox.npz``), runs ``fill_pits``, ``flow_dir_mfd``,
``flow_accumulation_mfd`` (plain and weighted), ``distance_to_channel_mfd`` (plain
and weighted), ``extract_streams_mfd`` (trace 0.7, as InVEST's SDR), and the D8 routing
(``flow_dir_d8``, ``flow_accumulation_d8``, ``distance_to_channel_d8``) and stores the outputs in ``tests/data/reference/hydro/pygeoprocessing.npz``
with the version in ``pygeoprocessing.json``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pygeoprocessing
import pygeoprocessing.routing as routing
from osgeo import gdal, osr

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "data" / "reference" / "hydro"
NODATA = -9999.0


def write(path: Path, values: np.ndarray, cell: float, nodata=NODATA, dtype=gdal.GDT_Float64) -> None:
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), values.shape[1], values.shape[0], 1, dtype, ["TILED=YES", "COMPRESS=DEFLATE"])
    ds.SetGeoTransform((500000.0, cell, 0.0, 4800000.0, 0.0, -cell))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32610)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
        values = np.where(np.isnan(values), nodata, values)
    band.WriteArray(values)
    ds = None


def read(path: Path, nodata=None) -> np.ndarray:
    ds = gdal.Open(str(path))
    a = ds.GetRasterBand(1).ReadAsArray()
    nd = ds.GetRasterBand(1).GetNoDataValue() if nodata is None else nodata
    ds = None
    if a.dtype.kind == "f" and nd is not None:
        a = a.astype(np.float64)
        a[a == nd] = np.nan
    return a


def main() -> None:
    src = np.load(REF / "whitebox.npz")
    names = sorted({k.split("/")[0] for k in src.files})
    out = {}
    with tempfile.TemporaryDirectory(prefix="ls-pgp-ref-") as tmp:
        wd = Path(tmp)
        for name in names:
            dem = src[f"{name}/dem"]
            cell = float(src[f"{name}/cell"][0])
            write(wd / "dem.tif", dem, cell)
            routing.fill_pits((str(wd / "dem.tif"), 1), str(wd / "filled.tif"), working_dir=str(wd))
            routing.flow_dir_mfd((str(wd / "filled.tif"), 1), str(wd / "dir.tif"), working_dir=str(wd))
            routing.flow_accumulation_mfd((str(wd / "dir.tif"), 1), str(wd / "acc.tif"))
            rng = np.random.default_rng(7)
            weights = rng.uniform(0.0, 2.0, dem.shape)
            write(wd / "w.tif", weights, cell)
            routing.flow_accumulation_mfd((str(wd / "dir.tif"), 1), str(wd / "accw.tif"),
                                          weight_raster_path_band=(str(wd / "w.tif"), 1))
            acc = read(wd / "acc.tif")
            channel = (np.nan_to_num(acc) >= 100).astype(np.uint8)
            write(wd / "channel.tif", channel.astype(np.float64), cell, nodata=None, dtype=gdal.GDT_Byte)
            routing.distance_to_channel_mfd((str(wd / "dir.tif"), 1), (str(wd / "channel.tif"), 1),
                                            str(wd / "dist.tif"))
            routing.distance_to_channel_mfd((str(wd / "dir.tif"), 1), (str(wd / "channel.tif"), 1),
                                            str(wd / "distw.tif"), weight_raster_path_band=(str(wd / "w.tif"), 1))
            routing.extract_streams_mfd((str(wd / "acc.tif"), 1), (str(wd / "dir.tif"), 1), 100.0,
                                        str(wd / "streams.tif"), trace_threshold_proportion=0.7)
            routing.flow_dir_d8((str(wd / "filled.tif"), 1), str(wd / "d8.tif"), working_dir=str(wd))
            routing.flow_accumulation_d8((str(wd / "d8.tif"), 1), str(wd / "d8acc.tif"))
            routing.flow_accumulation_d8((str(wd / "d8.tif"), 1), str(wd / "d8accw.tif"),
                                         weight_raster_path_band=(str(wd / "w.tif"), 1))
            routing.distance_to_channel_d8((str(wd / "d8.tif"), 1), (str(wd / "channel.tif"), 1),
                                           str(wd / "d8dist.tif"))
            routing.distance_to_channel_d8((str(wd / "d8.tif"), 1), (str(wd / "channel.tif"), 1),
                                           str(wd / "d8distw.tif"), weight_raster_path_band=(str(wd / "w.tif"), 1))
            out[f"{name}/streams"] = read(wd / "streams.tif").astype(np.uint8)
            out[f"{name}/d8"] = read(wd / "d8.tif").astype(np.uint8)
            out[f"{name}/d8acc"] = read(wd / "d8acc.tif")
            out[f"{name}/d8accw"] = read(wd / "d8accw.tif")
            out[f"{name}/d8dist"] = read(wd / "d8dist.tif", nodata=-1)
            out[f"{name}/d8distw"] = read(wd / "d8distw.tif", nodata=-1)
            out[f"{name}/filled"] = read(wd / "filled.tif")
            out[f"{name}/dir"] = read(wd / "dir.tif").astype(np.int32)
            out[f"{name}/acc"] = acc
            out[f"{name}/weights"] = weights
            out[f"{name}/accw"] = read(wd / "accw.tif")
            out[f"{name}/channel"] = channel
            out[f"{name}/dist"] = read(wd / "dist.tif", nodata=-1)
            out[f"{name}/distw"] = read(wd / "distw.tif", nodata=-1)
            print(name, "done")
    np.savez_compressed(REF / "pygeoprocessing.npz", **out)
    (REF / "pygeoprocessing.json").write_text(json.dumps({"pygeoprocessing": pygeoprocessing.__version__,
                                                          "cases": names}, indent=1))
    print("written", REF / "pygeoprocessing.npz")


if __name__ == "__main__":
    main()
