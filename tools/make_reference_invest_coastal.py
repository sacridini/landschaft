# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of InVEST's coastal vulnerability model (natcap.invest) for ls.services.

Usage::

    <python of an env with natcap.invest> tools/make_reference_invest_coastal.py [case ...]

Builds a synthetic coast (a mainland with a bay and a peninsula that overlaps it,
islands, one with a lagoon, two that overlap; an area of interest that cuts the
coast), Wave Watch III points, bathymetry, a shelf contour, a DEM, habitats (rasters
and polygons), geomorphology lines, sea-level rise points and a population raster,
all in SIRGAS 2000 / UTM 23S, runs ``natcap.invest.coastal_vulnerability.execute``
on them, and stores per case the inputs and the outputs in
``tests/data/reference/invest/<case>.npz`` and the vectors (WKB), tables, settings
and versions in ``<case>.json`` (as ``make_reference_invest.py`` does for the other
models). It does not import landschaft.

The cases run InVEST with two bugs corrected (landschaft has only the correct form):

* shore lines: InVEST drops the edges of the landmass whose two ends are
  ``numpy.isclose(..., rtol=1e-7)``, meant for repeated vertices, but relative to
  the coordinates: at UTM northings of 8e6 that drops edges up to 0.8 m long (and
  5 cm across), which breaks the coastline there. Patched: only edges of length 0.
* bathymetry of a ray with no valid cell: InVEST widens a window around the ray's
  end until it holds one, but where the window meets the top or left edge of the
  raster it moves it instead of cutting it, so it widens to one side only. Patched:
  the window is the square around the end, cut at the edges.

``coastal_full`` also records how far InVEST unpatched is from the patched run
(``unpatched`` in its JSON).
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_reference_invest import EPSG, OUT, _csv

X0, Y0 = 500000.0, 8000000.0
LONLAT = 4674  # SIRGAS 2000, longitude / latitude
SECTORS = [int(i * 360 / 16) for i in range(16)]

CASES = {
    # every input, the optional ones too; float32 rasters with NoData
    "coastal_full": dict(seed=71, resolution=150.0, fetch=6000.0, dem_radius=1000.0, population=True,
                         geomorphology=True, slr=True, habitats=["seagrass", "reef", "mangrove", "dunes"],
                         bathy_dtype="float32", dem_dtype="float32", unpatched=True),
    # the required inputs only; a float64 bathymetry and an int16 DEM, other distances
    "coastal_required": dict(seed=72, resolution=230.0, fetch=4500.0, dem_radius=700.0, population=False,
                             geomorphology=False, slr=False, habitats=["reef", "seagrass"],
                             bathy_dtype="float64", dem_dtype="int16", unpatched=False),
    # the landmass, Wave Watch III, shelf, reefs, sea-level rise and the bathymetry in longitude / latitude
    "coastal_lonlat": dict(seed=73, resolution=200.0, fetch=5000.0, dem_radius=800.0, population=True,
                           geomorphology=True, slr=True, habitats=["reef", "seagrass"], bathy_dtype="float32",
                           dem_dtype="float32", unpatched=False, lonlat=True),
}
IN_LONLAT = ("landmass", "wwiii", "shelf", "reef", "slr")

HABITATS = {  # id: (kind, rank, protection distance)
    "seagrass": ("raster", 1, 500.0),
    "reef": ("vector", 1, 800.0),
    "mangrove": ("vector", 2, 200.0),
    "dunes": ("raster", 4, 150.0),
}


# --------------------------------------------------------------------------- the synthetic coast

def coast_x(yr):
    """Easting of the mainland's coast at northing ``Y0 + yr``: waves and a bay."""
    return (X0 + 5000.0 + 1100.0 * np.sin(2 * np.pi * yr / 5500.0) + 450.0 * np.sin(2 * np.pi * yr / 1850.0 + 1.3)
            - 1700.0 * np.exp(-((yr - 7000.0) / 600.0) ** 2))


def _star(rng, cx, cy, r, n, rough=0.18):
    a = np.sort(rng.uniform(0, 2 * np.pi, n))
    rr = r * (1 + rough * rng.uniform(-1, 1, n))
    return [(float(cx + q * np.cos(t)), float(cy + q * np.sin(t))) for t, q in zip(a, rr)]


def _with_micro_edges(pts, where):
    """Edges 3 cm across and 0.4 m long after the vertices ``where`` (InVEST's isclose drops them)."""
    out = []
    for i, p in enumerate(pts):
        out.append(p)
        if i in where:
            out.append((p[0] + 0.03, p[1] + 0.4))
    return out


def scene(rng):
    """The landmass, the area of interest and the other vectors (shapely), in metres."""
    from shapely.geometry import (
        LineString,
        MultiLineString,
        MultiPolygon,
        Point,
        Polygon,
    )

    yr = [-9000.0]
    while yr[-1] < 24000.0:
        yr.append(yr[-1] + float(rng.uniform(30.0, 130.0)))
    yr = np.asarray(yr)
    cx = coast_x(yr) + rng.uniform(-8, 8, yr.size)
    coast = [(float(x), float(Y0 + y)) for x, y in zip(cx, yr)]
    inside = [i for i, (x, y) in enumerate(coast) if Y0 + 2000 < y < Y0 + 10500]
    micro = {inside[len(inside) // 5], inside[len(inside) // 2], inside[4 * len(inside) // 5]}
    coast = _with_micro_edges(coast, micro)
    mainland = Polygon([(X0 - 9000.0, Y0 - 9000.0), *coast, (X0 - 9000.0, Y0 + 24000.0)])
    # a peninsula whose root lies inland: the union cuts it at the coast
    top = [(X0 + 2000.0 + t, Y0 + 3150.0 + 40 * np.sin(t / 300.0) + float(rng.uniform(-10, 10)))
           for t in np.arange(0.0, 5600.0, 170.0)]
    tip = [(X0 + 7600.0 + 160 * np.cos(a), Y0 + 2925.0 + 225 * np.sin(a)) for a in np.linspace(1.2, -1.2, 7)]
    bottom = [(X0 + 2000.0 + t, Y0 + 2700.0 + 35 * np.cos(t / 260.0) + float(rng.uniform(-10, 10)))
              for t in np.arange(5600.0, 0.0, -190.0)]
    peninsula = Polygon(top + tip + bottom)
    a_ring = _with_micro_edges(_star(rng, X0 + 8500, Y0 + 4500, 750, 36), {9})
    island_a = Polygon(a_ring)
    island_b = Polygon(_star(rng, X0 + 9800, Y0 + 9300, 1000, 44, 0.12),
                       [_star(rng, X0 + 9750, Y0 + 9250, 300, 16, 0.1)[::-1]])
    island_c = Polygon(_star(rng, X0 + 10600, Y0 + 9900, 450, 22))
    island_d = Polygon(_star(rng, X0 + 7200, Y0 + 11000, 150, 12))
    island_e = Polygon(_star(rng, X0 + 13000, Y0 + 1500, 600, 26))
    landmass = [mainland, peninsula, MultiPolygon([island_a, island_d]), island_b, island_c, island_e]
    aoi = Polygon([(X0 + 3000, Y0 + 1000), (X0 + 12000, Y0 + 1800), (X0 + 11500, Y0 + 12000),
                   (X0 + 2500, Y0 + 11200)])
    # Wave Watch III points: a jittered 3 km grid with the 5 x 16 fields
    wwiii, w_rows = [], []
    for j, gy in enumerate(np.arange(-6000.0, 21001.0, 3000.0)):
        for i, gx in enumerate(np.arange(-6000.0, 21001.0, 3000.0)):
            wwiii.append(Point(X0 + gx + rng.uniform(-400, 400), Y0 + gy + rng.uniform(-400, 400)))
            row = {"I": i, "J": j}
            pct = rng.dirichlet(np.ones(16)) * 100
            wpct = rng.dirichlet(np.ones(16) * 0.7) * 100
            for k, d in enumerate(SECTORS):
                row[f"REI_PCT{d}"] = float(pct[k])
                row[f"REI_V{d}"] = float(rng.uniform(3, 15))
                row[f"WavP_{d}"] = float(rng.uniform(0, 40))
                row[f"WavPPCT{d}"] = float(wpct[k])
                row[f"V10PCT_{d}"] = float(rng.uniform(0.4, 14))  # some below 1 m/s
            w_rows.append(row)
    # the shelf: two parts of one feature (sharing an end) and a far line
    sy = np.arange(-15000.0, 30001.0, 400.0)
    sx = X0 + 15500 + 700 * np.sin(2 * np.pi * sy / 7000) + rng.uniform(-30, 30, sy.size)
    k = int(np.searchsorted(sy, 8000.0))
    s1 = LineString([(float(x), float(Y0 + y)) for x, y in zip(sx[:k + 1], sy[:k + 1])])
    s2 = LineString([(float(x), float(Y0 + y)) for x, y in zip(sx[k:], sy[k:])])
    shelf = [MultiLineString([s2, s1]), LineString([(X0 + 40000, Y0 - 9000), (X0 + 41000, Y0 + 30000)])]
    # habitat polygons
    reefs = [Polygon(_star(rng, X0 + x, Y0 + y, r, 7, 0.3)) for x, y, r in
             [(6900, 1700, 220), (7400, 5600, 160), (11300, 4200, 300), (6100, 9900, 120), (8800, 7700, 260),
              (12400, 11000, 200), (9600, 2300, 140), (7700, 8600, 90)]]
    mangrove = []
    for y in (6300.0, 6900.0, 7600.0, 10050.0):
        x = float(coast_x(y)) - 60
        mangrove.append(Polygon(_star(rng, x, Y0 + y, 140, 9, 0.25)))
    mangrove = [MultiPolygon([mangrove[0]]), MultiPolygon(mangrove[1:3]), MultiPolygon([mangrove[3]])]
    # geomorphology: pieces of the coast (a few metres off) and two island rings
    geo, ranks = [], []
    for lo, hi, rank in ((1500.0, 5000.0, 2), (4500.0, 6000.0, 3), (6000.0, 9000.0, 4)):
        yy = np.arange(lo, hi, 90.0)
        geo.append(LineString([(float(coast_x(y) + rng.uniform(-15, 15)), float(Y0 + y)) for y in yy]))
        ranks.append(rank)
    geo.append(LineString(list(island_a.exterior.coords)))
    ranks.append(1)
    geo.append(LineString(list(island_b.exterior.coords)))
    ranks.append(5)
    # sea-level rise points
    slr, slr_rows = [], []
    for gy in np.arange(-4000.0, 18001.0, 4000.0):
        for gx in np.arange(-4000.0, 20001.0, 4000.0):
            slr.append(Point(X0 + gx + rng.uniform(-600, 600), Y0 + gy + rng.uniform(-600, 600)))
            slr_rows.append({"trend": float(rng.uniform(2, 6)), "other": float(rng.uniform(-1, 1))})
    return dict(landmass=landmass, aoi=[aoi], wwiii=wwiii, wwiii_rows=w_rows, shelf=shelf, reef=reefs,
                mangrove=mangrove, geomorphology=geo, geomorphology_rows=[{"RANK": r} for r in ranks],
                slr=slr, slr_rows=slr_rows, islands=[island_a, island_b, island_c, island_d, island_e])


def _smooth_noise(rng, ny, nx, scale):
    from numpy.fft import irfft2, rfft2

    n = rng.normal(size=(ny, nx))
    f = rfft2(n)
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.rfftfreq(nx)[None, :]
    f *= np.exp(-(ky ** 2 + kx ** 2) * (scale ** 2))
    out = irfft2(f, s=(ny, nx))
    return out / out.std()


def _centres(gt, ny, nx):
    x = gt[0] + (np.arange(nx) + 0.5) * gt[1]
    y = gt[3] + (np.arange(ny) + 0.5) * gt[5]
    return np.meshgrid(x, y)


def _island_height(xx, yy, islands):
    h = np.zeros(xx.shape)
    for isl in islands:
        c = isl.centroid
        r = math.sqrt(isl.area / math.pi)
        d = np.hypot(xx - c.x, yy - c.y) / r
        h = np.maximum(h, 12.0 * (1.1 - d))
    return h


def rasters(rng, sc, spec):
    """Bathymetry, DEM, population and habitat rasters: {name: (array, geotransform, nodata)}."""
    out = {}
    # bathymetry, 100 m: negative at sea, positive on land; a NoData disc and a NoData strip in the north
    gt = (X0 - 5037.3, 100.0, 0.0, Y0 + 20011.7, 0.0, -100.0)
    ny, nx = 265, 255
    xx, yy = _centres(gt, ny, nx)
    depth = -(2.0 + 0.006 * (xx - coast_x(yy - Y0))) + 1.5 * _smooth_noise(rng, ny, nx, 9.0)
    depth = np.maximum(depth, _island_height(xx, yy, sc["islands"]) - 3.0)
    depth = np.round(depth, 2)
    depth[np.hypot(xx - (X0 + 9000), yy - (Y0 + 6500)) < 500] = -9999.0
    depth[yy > Y0 + 11300] = -9999.0  # rays from the north of the AOI find no depth: window around their end
    out["bathymetry"] = (depth.astype(spec["bathy_dtype"]), gt, -9999.0, EPSG)
    if spec.get("lonlat"):
        # the same field on a grid of 0.0009 degrees
        x0, x1, y1, y0 = gt[0], gt[0] + nx * gt[1], gt[3], gt[3] + ny * gt[5]
        lons, lats = _lonlat_xy([x0, x1, x0, x1], [y0, y0, y1, y1])
        lo0, lo1, la0, la1 = lons.min(), lons.max(), lats.min(), lats.max()
        lgt = (lo0 - 0.00031, 0.0009, 0.0, la1 + 0.00017, 0.0, -0.0009)
        lnx, lny = int((lo1 - lo0) / 0.0009) + 2, int((la1 - la0) / 0.0009) + 2
        lon, lat = _centres(lgt, lny, lnx)
        ux, uy = _lonlat_xy(lon.ravel(), lat.ravel(), inverse=True)
        ux, uy = ux.reshape(lon.shape), uy.reshape(lon.shape)
        ld = -(2.0 + 0.006 * (ux - coast_x(uy - Y0))) + 1.5 * _smooth_noise(rng, lny, lnx, 9.0)
        ld = np.round(np.maximum(ld, _island_height(ux, uy, sc["islands"]) - 3.0), 2)
        ld[uy > Y0 + 11300] = -9999.0
        out["bathymetry"] = (ld.astype(np.float32), lgt, -9999.0, LONLAT)
    # DEM, 60 m: rises inland and on the islands
    gt = (X0 - 1021.7, 60.0, 0.0, Y0 + 14013.3, 0.0, -60.0)
    ny, nx = 240, 260
    xx, yy = _centres(gt, ny, nx)
    elev = (coast_x(yy - Y0) - xx) * 0.004 + 1.5 + 0.8 * _smooth_noise(rng, ny, nx, 6.0)
    elev = np.maximum(elev, _island_height(xx, yy, sc["islands"]))
    hole = rng.random((ny, nx)) < 0.02
    if spec["dem_dtype"] == "int16":
        dem, nd = np.round(elev * 10).astype(np.int16), -32768
    else:
        dem, nd = np.round(elev, 3).astype(np.float32), -9999.0
    dem[hole] = nd
    out["dem"] = (dem, gt, float(nd), EPSG)
    # population, 75 m: people on land, none at sea
    gt = (X0 - 1500.1, 75.0, 0.0, Y0 + 14200.9, 0.0, -75.0)
    ny, nx = 205, 210
    xx, yy = _centres(gt, ny, nx)
    land = (xx < coast_x(yy - Y0)) | (_island_height(xx, yy, sc["islands"]) > 1.2)
    pop = np.where(land, np.round(np.exp(rng.normal(2.0, 1.0, (ny, nx))), 2), 0.0).astype(np.float32)
    pop[rng.random((ny, nx)) < 0.02] = -1.0
    out["population"] = (pop, gt, -1.0, EPSG)
    # seagrass (uint8 presence, 40 m) and dunes (float32 cover, 60 m)
    gt = (X0 + 2011.3, 40.0, 0.0, Y0 + 12990.4, 0.0, -40.0)
    ny, nx = 300, 280
    xx, yy = _centres(gt, ny, nx)
    sea = xx > coast_x(yy - Y0) + 50
    grass = (sea & (_smooth_noise(rng, ny, nx, 12.0) > 0.9)).astype(np.uint8)
    grass[rng.random((ny, nx)) < 0.01] = 255
    out["seagrass"] = (grass, gt, 255.0, EPSG)
    gt = (X0 + 1530.9, 60.0, 0.0, Y0 + 13470.2, 0.0, -60.0)
    ny, nx = 230, 190
    xx, yy = _centres(gt, ny, nx)
    near = np.abs(xx - coast_x(yy - Y0)) < 120
    cover = np.where(near & (_smooth_noise(rng, ny, nx, 5.0) > 0.3), np.round(rng.uniform(0.1, 1, (ny, nx)), 3), 0)
    cover = cover.astype(np.float32)
    cover[rng.random((ny, nx)) < 0.01] = -1.0
    out["dunes"] = (cover, gt, -1.0, EPSG)
    return out


# --------------------------------------------------------------------------- files

def _srs(epsg=EPSG):
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _lonlat_xy(x, y, inverse=False):
    """UTM 23S to longitude / latitude (or back) by OGR's coordinate transformation."""
    from osgeo import osr

    a, b = (_srs(LONLAT), _srs()) if inverse else (_srs(), _srs(LONLAT))
    t = osr.CoordinateTransformation(a, b)
    out = np.asarray(t.TransformPoints(np.c_[np.asarray(x, float), np.asarray(y, float)].tolist()))
    return out[:, 0], out[:, 1]


def _to_lonlat(geoms):
    import shapely

    def fn(xy):
        lon, lat = _lonlat_xy(xy[:, 0], xy[:, 1])
        return np.c_[lon, lat]

    return [shapely.transform(g, fn) for g in geoms]


def _write_raster(path, array, gt, nodata, epsg=EPSG):
    from osgeo import gdal

    types = {"float32": gdal.GDT_Float32, "float64": gdal.GDT_Float64, "int16": gdal.GDT_Int16,
             "uint8": gdal.GDT_Byte}
    ny, nx = array.shape
    ds = gdal.GetDriverByName("GTiff").Create(str(path), nx, ny, 1, types[str(array.dtype)])
    ds.SetGeoTransform(gt)
    ds.SetProjection(_srs(epsg).ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array)
    ds = None


def _write_vector(path, geoms, rows=None, epsg=EPSG):
    from osgeo import ogr

    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    gtype = {"Point": ogr.wkbPoint, "LineString": ogr.wkbLineString, "MultiLineString": ogr.wkbMultiLineString,
             "Polygon": ogr.wkbPolygon, "MultiPolygon": ogr.wkbMultiPolygon}
    kinds = {g.geom_type for g in geoms}
    layer_type = gtype[next(iter(kinds))] if len(kinds) == 1 else ogr.wkbUnknown
    layer = ds.CreateLayer(Path(path).stem, _srs(epsg), layer_type)
    if rows:
        for k, v in rows[0].items():
            layer.CreateField(ogr.FieldDefn(k, ogr.OFTInteger if isinstance(v, int) else ogr.OFTReal))
    for i, g in enumerate(geoms):
        f = ogr.Feature(layer.GetLayerDefn())
        f.SetGeometry(ogr.CreateGeometryFromWkb(g.wkb))
        if rows:
            for k, v in rows[i].items():
                f.SetField(k, v)
        layer.CreateFeature(f)
    ds = None


def _vector_json(geoms, rows=None, epsg=EPSG):
    out = {"wkb": [g.wkb_hex for g in geoms], "epsg": epsg}
    if rows:
        out["columns"] = {k: [r[k] for r in rows] for k in rows[0]}
    return out


def _read_vector(path):
    """Geometry (points: x, y; lines: their vertices) and fields of a vector, by FID order."""
    from osgeo import gdal, ogr

    ds = gdal.OpenEx(str(path), gdal.OF_VECTOR)
    layer = ds.GetLayer()
    defn = layer.GetLayerDefn()
    names = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
    out = {n: [] for n in names}
    out["fid"], out["geom"] = [], []
    for f in layer:
        out["fid"].append(f.GetFID())
        g = f.GetGeometryRef()
        if g.GetGeometryType() == ogr.wkbPoint:
            out["geom"].append((g.GetX(), g.GetY()))
        else:
            out["geom"].append([g.GetPoint_2D(i) for i in range(g.GetPointCount())])
        for n in names:
            out[n].append(f.GetField(n))
    return out


def _floats(values):
    return np.asarray([np.nan if v is None else float(v) for v in values], np.float64)


def _read_raster(path):
    from osgeo import gdal

    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    return band.ReadAsArray(), ds.GetGeoTransform(), band.GetNoDataValue()


# --------------------------------------------------------------------------- the patches

def _patch(edges=True, window=True):
    """Correct InVEST's two bugs (see the module docstring); returns the undo."""
    import numpy
    import pygeoprocessing
    from natcap.invest.coastal_vulnerability import coastal_vulnerability as cv

    isclose = numpy.isclose

    def exact_isclose(a, b, *args, **kwargs):
        if kwargs.get("rtol") == 1e-7 and not args:  # the duplicate-vertex test of shore lines only
            return a == b
        return isclose(a, b, *args, **kwargs)

    original = cv.extract_bathymetry_along_ray

    def extract(ray_geometry, bathy_gt, bathy_nodata, bathy_band):
        values = original_samples(ray_geometry, bathy_gt, bathy_nodata, bathy_band)
        if values[0]:
            return values[0]
        ix, iy = values[1]
        n_rows, n_cols = bathy_band.YSize, bathy_band.XSize
        k = 0
        while True:
            x0, y0 = max(ix - k, 0), max(iy - k, 0)
            x1, y1 = min(ix + k + 1, n_cols), min(iy + k + 1, n_rows)
            if x0 == 0 and y0 == 0 and x1 == n_cols and y1 == n_rows:
                raise ValueError("searched entire bathymetry raster for valid values and found none")
            k += 1  # the square of side 2k + 1 around the end, cut at the edges of the raster
            x0, y0 = max(ix - k, 0), max(iy - k, 0)
            x1, y1 = min(ix + k + 1, n_cols), min(iy + k + 1, n_rows)
            win = bathy_band.ReadAsArray(xoff=x0, yoff=y0, win_xsize=x1 - x0, win_ysize=y1 - y0)
            valid = slice(None)
            if bathy_nodata is not None:
                valid = ~pygeoprocessing.array_equals_nodata(win, bathy_nodata)
            if numpy.any(valid):
                return [numpy.mean(win[valid])]

    def original_samples(ray_geometry, bathy_gt, bathy_nodata, bathy_band):
        """InVEST's samples along the ray (unchanged) and the cell of its end."""
        import shapely.wkb

        bathy_values = []
        n_pts = int(math.ceil(ray_geometry.Length() / bathy_gt[1]))
        shapely_line = shapely.wkb.loads(bytes(ray_geometry.ExportToWkb()))
        ix = iy = 0
        for i in range(n_pts + 1):
            point = shapely_line.interpolate(float(i) / n_pts, normalized=True)
            ix = int((point.x - bathy_gt[0]) / bathy_gt[1])
            iy = int((point.y - bathy_gt[3]) / bathy_gt[5])
            value = bathy_band.ReadAsArray(xoff=ix, yoff=iy, win_xsize=1, win_ysize=1)
            if bathy_nodata is None or not numpy.isclose(value[0][0], bathy_nodata, equal_nan=True):
                bathy_values.append(value)
        return bathy_values, (ix, iy)

    if edges:
        numpy.isclose = exact_isclose
    if window:
        cv.extract_bathymetry_along_ray = extract

    def undo():
        numpy.isclose = isclose
        cv.extract_bathymetry_along_ray = original

    return undo


# --------------------------------------------------------------------------- a run

def _execute(spec, tmp, patched, rng_seed, edges=True, window=True):
    from natcap.invest.coastal_vulnerability import coastal_vulnerability as cv

    rng = np.random.default_rng(rng_seed)
    sc = scene(rng)
    ras = rasters(rng, sc, spec)
    sc["epsg"] = {k: EPSG for k in ("landmass", "aoi", "shelf", "wwiii", "reef", "mangrove", "geomorphology", "slr")}
    if spec.get("lonlat"):
        for name in IN_LONLAT:
            sc[name], sc["epsg"][name] = _to_lonlat(sc[name]), LONLAT
    files = {}
    for name in ("landmass", "aoi", "shelf"):
        files[name] = tmp / f"{name}.gpkg"
        _write_vector(files[name], sc[name], epsg=sc["epsg"][name])
    files["wwiii"] = tmp / "wwiii.gpkg"
    _write_vector(files["wwiii"], sc["wwiii"], sc["wwiii_rows"], epsg=sc["epsg"]["wwiii"])
    for name in ("bathymetry", "dem", "population", "seagrass", "dunes"):
        files[name] = tmp / f"{name}.tif"
        _write_raster(files[name], *ras[name])
    rows = []
    for hid in spec["habitats"]:
        kind, rank, dist = HABITATS[hid]
        if kind == "vector":
            files[hid] = tmp / f"{hid}.gpkg"
            _write_vector(files[hid], sc[hid], epsg=sc["epsg"][hid])
        rows.append([hid, str(files[hid]), rank, dist])
    _csv(tmp / "habitats.csv", ["id", "path", "rank", "protection distance (m)"], rows)
    ws = "ws" if patched and edges and window else f"ws_{int(patched)}{int(edges)}{int(window)}"
    args = {"workspace_dir": str(tmp / ws), "results_suffix": "",
            "aoi_vector_path": str(files["aoi"]), "model_resolution": spec["resolution"],
            "landmass_vector_path": str(files["landmass"]), "wwiii_vector_path": str(files["wwiii"]),
            "max_fetch_distance": spec["fetch"], "bathymetry_raster_path": str(files["bathymetry"]),
            "shelf_contour_vector_path": str(files["shelf"]), "dem_path": str(files["dem"]),
            "dem_averaging_radius": spec["dem_radius"], "habitat_table_path": str(tmp / "habitats.csv"),
            "geomorphology_vector_path": "", "geomorphology_fill_value": "", "population_raster_path": "",
            "population_radius": "", "slr_vector_path": "", "slr_field": "", "n_workers": -1}
    if spec["geomorphology"]:
        files["geomorphology"] = tmp / "geomorphology.gpkg"
        _write_vector(files["geomorphology"], sc["geomorphology"], sc["geomorphology_rows"])
        args.update(geomorphology_vector_path=str(files["geomorphology"]), geomorphology_fill_value="3")
    if spec["population"]:
        args.update(population_raster_path=str(files["population"]), population_radius=1500.0)
    if spec["slr"]:
        files["slr"] = tmp / "slr.gpkg"
        _write_vector(files["slr"], sc["slr"], sc["slr_rows"], epsg=sc["epsg"]["slr"])
        args.update(slr_vector_path=str(files["slr"]), slr_field="trend")
    undo = _patch(edges, window) if patched else (lambda: None)
    try:
        registry = cv.execute(args)
    finally:
        undo()
    return sc, ras, args, registry


def _outputs(registry, spec):
    out, meta = {}, {}
    pts = _read_vector(registry["shore_points"])
    out["shore_xy"] = np.asarray(pts["geom"], np.float64).reshape(-1, 2)
    out["shore_id"] = np.asarray(pts["shore_id"], np.int64)
    fp = _read_vector(registry["fetch_points"])
    out["fdist"] = np.stack([_floats(fp[f"fdist_{d}"]) for d in SECTORS], axis=1)
    out["fdepth"] = np.stack([_floats(fp[f"fdepth_{d}"]) for d in SECTORS], axis=1)
    out["rei"] = _floats(fp["REI"])
    wfields = [k for k in fp if k not in ("fid", "geom", "shore_id", "REI") and not k.startswith(("fdist_", "fdepth_"))]
    out["wwiii_at_shore"] = np.stack([_floats(fp[k]) for k in wfields], axis=1)
    meta["wwiii_at_shore_fields"] = wfields
    rays = _read_vector(registry["fetch_rays"])
    out["ray_ends"] = np.asarray([[*g[0], *g[1]] for g in rays["geom"]], np.float64).reshape(-1, 4)
    out["ray_dist"] = _floats(rays["fetch_dist"])
    out["ray_direction"] = _floats(rays["direction"])
    we = _read_vector(registry["wave_energies"])
    for k in ("E_ocean", "E_local", "Eo_El_diff", "maxH_local", "minH_local", "maxT_local", "minT_local"):
        out[f"wave_{k}"] = _floats(we[k])
    meta["max_E_type"] = [v or "" for v in we["max_E_type"]]
    ie = _read_vector(registry["intermediate_exposure"])
    meta["intermediate_fields"] = [k for k in ie if k not in ("fid", "geom", "shore_id")]
    for k in meta["intermediate_fields"]:
        out[f"intermediate_{k}"] = _floats(ie[k])
    ce = _read_vector(registry["coastal_exposure"])
    meta["exposure_fields"] = [k for k in ce if k not in ("fid", "geom", "shore_id")]
    for k in meta["exposure_fields"]:
        out[f"exposure_{k}"] = _floats(ce[k])
    lines = Path(registry["habitat_protection"]).read_text().splitlines()
    meta["habitat_protection_header"] = lines[0].split(",")
    out["habitat_protection"] = np.asarray([[float(v) if v else np.nan for v in ln.split(",")] for ln in lines[1:]])
    for key in ("negative_bathymetry", "positive_dem", "clipped_projected_pop"):
        if key in registry and Path(registry[key]).is_file():
            a, gt, nd = _read_raster(registry[key])
            out[f"raster_{key}"] = a
            meta[f"raster_{key}"] = {"gt": list(gt), "nodata": nd}
    for hid in spec["habitats"]:
        p = Path(registry["intermediate_exposure"]).parent / "habitats" / f"clipped_projected_{hid}.tif"
        if p.is_file():
            a, gt, nd = _read_raster(p)
            out[f"raster_habitat_{hid}"] = a
            meta[f"raster_habitat_{hid}"] = {"gt": list(gt), "nodata": nd}
    if spec["geomorphology"]:
        miss = _read_vector(registry["shore_points_missing_geomorphology"])
        out["geomorphology_missing"] = np.asarray(miss["shore_id"], np.int64)
    return out, meta


def _differences(a, b):
    """How many shore points (and values) of two runs differ."""
    if a["shore_xy"].shape != b["shore_xy"].shape:
        return {"shore_points": [len(a["shore_xy"]), len(b["shore_xy"])]}
    d = {"shore_points_moved": int(np.sum(np.any(a["shore_xy"] != b["shore_xy"], axis=1)))}
    for k in ("fdepth", "exposure_exposure"):
        x, y = a[k], b[k]
        diff = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        d[k] = int(np.sum(np.any(diff.reshape(len(x), -1), axis=1)))
    return d


def run(name, spec, tmp):
    sc, ras, args, registry = _execute(spec, tmp, True, spec["seed"])
    outputs, meta = _outputs(registry, spec)
    inputs = {}
    meta["rasters"] = {}
    for rname, (arr, gt, nd, epsg) in ras.items():
        inputs[rname] = arr
        meta["rasters"][rname] = {"gt": list(gt), "nodata": nd, "dtype": str(arr.dtype), "epsg": epsg}
    meta["vectors"] = {k: _vector_json(sc[k], sc.get(f"{k}_rows"), sc["epsg"][k]) for k in sc["epsg"]}
    meta["epsg"] = EPSG
    meta["settings"] = {k: args[k] for k in ("model_resolution", "max_fetch_distance", "dem_averaging_radius",
                                             "geomorphology_fill_value", "population_radius", "slr_field")}
    meta["case"] = spec
    meta["habitats"] = [{"id": h, "kind": HABITATS[h][0], "rank": HABITATS[h][1], "distance": HABITATS[h][2]}
                        for h in spec["habitats"]]
    if spec.get("unpatched"):
        meta["unpatched"] = {}
        for what, edges, window in (("edges", False, True), ("window", True, False)):
            _, _, _, reg2 = _execute(spec, tmp, True, spec["seed"], edges, window)
            out2, _ = _outputs(reg2, spec)
            meta["unpatched"][what] = _differences(outputs, out2)
    return inputs, outputs, meta


def main(argv):
    import natcap.invest
    import numpy
    import pygeoprocessing
    import rtree
    import shapely
    from osgeo import gdal, ogr

    OUT.mkdir(parents=True, exist_ok=True)
    for name in argv or list(CASES):
        spec = CASES[name]
        with tempfile.TemporaryDirectory(prefix=f"invest-{name}-", ignore_cleanup_errors=True) as tmp:
            inputs, outputs, meta = run(name, spec, Path(tmp))
        meta["versions"] = {"natcap.invest": natcap.invest.__version__, "pygeoprocessing": pygeoprocessing.__version__,
                            "numpy": numpy.__version__, "gdal": gdal.__version__, "shapely": shapely.__version__,
                            "geos (shapely)": shapely.geos_version_string,
                            "geos (ogr)": ".".join(str(v) for v in (ogr.GetGEOSVersionMajor(), ogr.GetGEOSVersionMinor(),
                                                                    ogr.GetGEOSVersionMicro())),
                            "rtree": rtree.__version__}
        arrays = {f"in/{k}": v for k, v in inputs.items()}
        arrays.update({f"out/{k}": v for k, v in outputs.items()})
        np.savez_compressed(OUT / f"{name}.npz", **arrays)
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print(name, "->", len(outputs["shore_xy"]), "shore points", meta.get("unpatched", ""))


if __name__ == "__main__":
    main(sys.argv[1:])
