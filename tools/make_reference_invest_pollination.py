# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of InVEST's pollination model (natcap.invest) for ls.services.

Usage::

    <python of an env with natcap.invest> tools/make_reference_invest_pollination.py [case ...]

Builds synthetic inputs (a land use map, a guild table, a biophysical table and,
for some cases, farm polygons), runs ``natcap.invest.pollination.execute(args)`` on
them, and stores per case the inputs and the outputs in
``tests/data/reference/invest/<case>.npz`` and the tables, settings and versions in
``<case>.json`` (as ``make_reference_invest.py`` does for the other models, whose
helpers it uses). It does not import landschaft.

Cases marked ``patched`` run InVEST with the round-off clean-up of
pygeoprocessing's ``convolve_2d`` corrected: it sets to 0 the values within 1e-8 of
1e-8 (``numpy.isclose(result, set_tol_to_zero)``), not those within 1e-8 of 0 as it
documents.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_reference_invest import (
    EPSG,
    ORIGIN,
    OUT,
    PIXEL,
    _csv,
    _lulc,
    _read,
    _write_raster,
)

CASES = {
    # several 256 x 256 blocks of the signal (and a short last row of blocks)
    "pollination_blocks": dict(seed=41, ny=258, nx=300, species=["apis", "bombus"], seasons=["spring", "summer"],
                               alphas=[110.0, 55.0], farms=False, patched=True,
                               keep=["floral_resources_apis", "convolve_ps_bombus", "total_pollinator_abundance"]),
    # three guilds, two seasons and three substrates, every intermediate map
    "pollination_guilds": dict(seed=42, ny=67, nx=89, species=["andrena", "halictus", "osmia"],
                               seasons=["fall", "spring"], alphas=[90.0, 200.0, 31.0],
                               substrates=["cavity", "ground", "stem"], farms=False, patched=True),
    # farms: their floral resources and nesting over the land use map, half saturation by season
    "pollination_farms": dict(seed=43, ny=71, nx=83, species=["apis", "bombus"], seasons=["spring", "summer"],
                              alphas=[150.0, 60.0], farms=True, patched=True),
    # a kernel wider than 256 cells: 3 x 3 kernel blocks over a narrow raster (one block taller than 256 rows)
    "pollination_wide_kernel": dict(seed=44, ny=300, nx=61, species=["xylocopa"], seasons=["spring", "summer"],
                                    alphas=[1650.0], farms=False, patched=True,
                                    keep=["floral_resources", "convolve_ps", "pollinator_supply",
                                          "pollinator_abundance", "total_pollinator_abundance",
                                          "local_foraging_effectiveness", "habitat_nesting_index"]),
    # faint flowers (convolutions of about 1e-8 at the edges of the flowering patches), corrected and as InVEST is:
    # its clean-up of the convolutions zeroes values from -1e-13 to 2e-8
    "pollination_faint": dict(seed=45, ny=67, nx=89, species=["andrena", "osmia"], seasons=["fall", "spring"],
                              alphas=[45.0, 31.0], floral_scale=1e-4, farms=False, patched=True,
                              keep=["floral_resources", "convolve_ps", "total_pollinator_abundance"]),
    "pollination_faint_unpatched": dict(seed=45, ny=67, nx=89, species=["andrena", "osmia"],
                                        seasons=["fall", "spring"], alphas=[45.0, 31.0], floral_scale=1e-4,
                                        farms=False, patched=False,
                                        keep=["floral_resources", "convolve_ps", "total_pollinator_abundance"]),
}


def _patch_round_off():
    """pygeoprocessing's clean-up of the block convolutions as documented: |x| <= 1e-8 is 0."""
    import numpy

    original = numpy.isclose

    def isclose(a, b, *args, **kwargs):
        if not args and not kwargs and isinstance(b, float) and b == 1e-8:
            return numpy.abs(a) <= b
        return original(a, b, *args, **kwargs)

    numpy.isclose = isclose
    return original


def _farms(rng, ny, nx, seasons, substrates):
    """Farm polygons: slanted boxes, one over another, one beyond the raster, one tiny (no cell centre)."""
    x0, y0 = ORIGIN
    rows = []
    boxes = [(5, 8, 30, 25), (20, 40, 50, 70), (40, 15, 65, 45), (55, 60, 80, 95), (2, 70, 6, 74)]
    for k, (r0, c0, r1, c1) in enumerate(boxes):
        xa, xb = x0 + c0 * PIXEL + 7.0, x0 + c1 * PIXEL - 11.0
        ya, yb = y0 - r0 * PIXEL - 5.0, y0 - r1 * PIXEL + 13.0
        tilt = 41.0 if k % 2 else -23.0
        ring = [(xa, ya), (xb, ya + tilt), (xb, yb), (xa - tilt, yb), (xa, ya)]
        row = {"crop_type": ["apple", "coffee", "squash"][k % 3], "half_sat": round(float(rng.uniform(0.2, 0.8)), 3),
               "season": seasons[k % len(seasons)], "p_dep": round(float(rng.uniform(0.3, 1.0)), 3),
               "p_managed": round(float(rng.uniform(0.0, 0.4)), 3)}
        for s in seasons:
            row[f"fr_{s}"] = round(float(rng.uniform(0.0, 1.0)), 3)
        for s in substrates:
            row[f"n_{s}"] = round(float(rng.uniform(0.0, 1.0)), 3)
        rows.append((ring, row))
    # a polygon no cell centre falls in
    xa, ya = x0 + 10 * PIXEL + 2.0, y0 - 60 * PIXEL - 2.0
    tiny = [(xa, ya), (xa + 9.0, ya), (xa + 9.0, ya - 9.0), (xa, ya - 9.0), (xa, ya)]
    row = dict(rows[0][1], crop_type="tiny")
    rows.append((tiny, row))
    return rows


def _write_farms(path, farms, seasons, substrates):
    from osgeo import ogr, osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(str(path))
    layer = ds.CreateLayer(Path(path).stem, srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("crop_type", ogr.OFTString))
    layer.CreateField(ogr.FieldDefn("season", ogr.OFTString))
    names = ["half_sat", "p_dep", "p_managed", *[f"fr_{s}" for s in seasons], *[f"n_{s}" for s in substrates]]
    for n in names:
        layer.CreateField(ogr.FieldDefn(n, ogr.OFTReal))
    for ring, row in farms:
        g = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in ring:
            g.AddPoint_2D(x, y)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(g)
        feat = ogr.Feature(layer.GetLayerDefn())
        for k, v in row.items():
            feat.SetField(k, v)
        feat.SetGeometry(poly)
        layer.CreateFeature(feat)
    ds = None


def run(name, spec, tmp):
    from natcap.invest.pollination import pollination
    from osgeo import gdal, ogr

    rng = np.random.default_rng(spec["seed"])
    codes = [1, 2, 4, 7, 9]
    ny, nx = spec["ny"], spec["nx"]
    nodata = 255
    lulc = _lulc(rng, ny, nx, codes, nodata).astype(np.uint8)
    seasons, species = spec["seasons"], spec["species"]
    substrates = spec.get("substrates", ["cavity", "ground"])
    bio = {"lucode": codes}
    for s in substrates:
        v = np.round(rng.uniform(0, 1, len(codes)), 2)
        v[0] = 0.0
        bio[f"nesting_{s}_availability_index"] = v
    for s in seasons:
        v = np.round(rng.uniform(0, 1, len(codes)), 3)
        v[1] = 0.0  # a class without flowers: floral resources of 0 where it surrounds a cell
        if "floral_scale" in spec:
            v[2:] = 0.0
            v *= spec["floral_scale"]
            v[0] = spec["floral_scale"]
        bio[f"floral_resources_{s}_index"] = v
    guild = {"species": species}
    for s in substrates:
        guild[f"nesting_suitability_{s}_index"] = np.round(rng.uniform(0, 1, len(species)), 2)
    for s in seasons:
        guild[f"foraging_activity_{s}_index"] = np.round(rng.uniform(0.1, 1, len(species)), 2)
    guild["alpha"] = spec["alphas"]
    guild["relative_abundance"] = np.round(rng.uniform(0.2, 1, len(species)), 2)
    bio_cols, guild_cols = list(bio), list(guild)
    _csv(tmp / "bio.csv", bio_cols, [[bio[c][i] for c in bio_cols] for i in range(len(codes))])
    _csv(tmp / "guild.csv", guild_cols, [[guild[c][i] for c in guild_cols] for i in range(len(species))])
    _write_raster(tmp / "lulc.tif", lulc, nodata, gdal.GDT_Byte)
    args = {"workspace_dir": str(tmp / "ws"), "results_suffix": "", "landcover_raster_path": str(tmp / "lulc.tif"),
            "guild_table_path": str(tmp / "guild.csv"),
            "landcover_biophysical_table_path": str(tmp / "bio.csv"), "farm_vector_path": "", "n_workers": -1}
    farms = None
    if spec["farms"]:
        farms = _farms(rng, ny, nx, seasons, substrates)
        _write_farms(tmp / "farms.shp", farms, seasons, substrates)
        args["farm_vector_path"] = str(tmp / "farms.shp")
    original = _patch_round_off() if spec["patched"] else None
    try:
        pollination.execute(args)
    finally:
        if original is not None:
            import numpy

            numpy.isclose = original
    outputs, nodatas = {}, {}
    keep = spec.get("keep")
    for path in sorted((tmp / "ws").rglob("*.tif")):
        key = path.stem
        if key == "blank_raster" or (keep is not None and not any(key.startswith(k) for k in keep)):
            continue
        if key.startswith("kernel_") and spec["ny"] * spec["nx"] > 10000:
            continue
        a, nd = _read(path)
        nodatas[key] = nd
        outputs[key] = np.where(a == nd, np.nan, a) if nd is not None else a
    meta = {"model": "pollination", "nodata": nodata, "pixel": PIXEL, "patched": spec["patched"],
            "bio": {k: [float(x) for x in v] for k, v in bio.items()},
            "guild": {k: ([str(x) for x in v] if k == "species" else [float(x) for x in v]) for k, v in guild.items()},
            "outputs": sorted(outputs), "nodata_values": nodatas}
    if farms is not None:
        meta["farms"] = [{"ring": ring, **row} for ring, row in farms]
        ds = ogr.Open(str(tmp / "ws" / "farm_results.shp"))
        layer = ds.GetLayer()
        defn = layer.GetLayerDefn()
        fields = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
        meta["farm_results"] = [{f: feat.GetField(f) for f in fields} for feat in layer]
        ds = None
    return {"lulc": lulc}, outputs, meta


def main(argv):
    import natcap.invest
    import numpy
    import pygeoprocessing

    OUT.mkdir(parents=True, exist_ok=True)
    for name in argv or list(CASES):
        spec = CASES[name]
        with tempfile.TemporaryDirectory(prefix=f"invest-{name}-", ignore_cleanup_errors=True) as tmp:
            inputs, outputs, meta = run(name, spec, Path(tmp))
        meta["versions"] = {"natcap.invest": natcap.invest.__version__, "pygeoprocessing": pygeoprocessing.__version__,
                            "numpy": numpy.__version__}
        arrays = {f"in/{k}": v for k, v in inputs.items()}
        arrays.update({f"out/{k}": v for k, v in outputs.items()})
        np.savez_compressed(OUT / f"{name}.npz", **arrays)
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(name, "->", len(outputs), "maps")


if __name__ == "__main__":
    main(sys.argv[1:])
