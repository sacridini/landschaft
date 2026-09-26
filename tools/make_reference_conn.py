# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden habitat-graph outputs with Makurhini (R) and Conefor Sensinode 2.2.

Usage::

    python tools/make_reference_conn.py [makurhini] [conefor] [spatial] [links] [deca] [protconn] [grids]
        [centrality] [betweenness] [focal] [commute] [samc]

``makurhini`` needs Rscript with Makurhini installed (``LANDSCHAFT_RSCRIPT`` as
in ``tests/lsm_reference.py``; the default R library is used). It exports
Makurhini's example ``habitat_nodes`` (centroid distances), builds three
synthetic scenes, stores them in ``tests/data/reference/conn/scenes.npz`` and
runs ``MK_dPCIIC`` (public, rounded output and the unrounded internal steps,
``tools/conn_makurhini_reference.R``) into ``makurhini.csv.gz``.

``spatial`` writes synthetic polygons (``tests/data/reference/conn/polygons.gpkg``)
and Makurhini's example raster ``habitat_nodes_raster``
(``habitat_nodes_raster.tif``), and stores Makurhini's ``nodesfile`` /
``distancefile`` (centroid and edge distances) on both and ``MK_dPCIIC`` run
on them end to end in ``tests/data/reference/conn/spatial`` (pairs up to 30 km
and every 17th of the others).

``links`` runs ``MK_dPCIIC_links`` (link removal and link change, with some
pairs brought closer and others moved apart) and ``MK_dPCIIC`` with
``restoration`` (the last five nodes to add, which Makurhini's code needs to
align its attributes) on the synthetic scenes (``makurhini_links.csv.gz``).

``deca`` runs ``MK_dECA`` on three times of the polygons and of the example
raster, some nodes removed at each time (``deca_steps.csv`` lists them;
``makurhini_deca.csv.gz``).

``protconn`` runs ``MK_ProtConn`` on synthetic protected areas and regions
(``protconn_areas.gpkg``, ``protconn_regions.gpkg``: densified boxes, which
mapshaper's simplification leaves as they are) with and without transboundary
zones and ProtConn[Bound], with its inner steps (``makurhini_protconn.csv.gz``).

``grids`` (after ``protconn`` and ``deca``) stores Makurhini's ``make_grid``
cells, ``MK_Connect_grid`` (ProtConn with its default, which cuts every link,
and with ``pij_min = 0``, and PC), ``MK_ProtConn_raster`` on a raster of the
protected areas (``protconn_areas.tif``, ids numbered in scan order) and
``MK_dECA_grid`` in ``tests/data/reference/conn/grid``.

``centrality`` (after ``spatial``) runs ``MK_RMCentrality`` (binary and
weighted graphs, centroid distances; Louvain communities left out) on the
polygons and the example raster (``makurhini_centrality.csv.gz``).

``betweenness`` runs the command line of Conefor 2.7.1 (``CONEFOR_EXE``, from
conefor.org; not run by default) for BC, BC(IIC) and BC(PC) on the synthetic
scenes (``conefor_betweenness.csv.gz``).

``focal`` (after ``spatial`` and ``protconn``) runs ``MK_Focal_nodes`` (IIC and
PC, one and two distances) on the polygons and the ProtConn areas
(``makurhini_focal.csv.gz``).

``commute`` (after ``spatial``) stores ``distancefile(type = "commute-time")``
(gdistance's ``commuteDistance``) on the polygons (8 and 4 directions) and the
example raster (every 7th pair) in ``tests/data/reference/conn/spatial``.

``samc`` (after ``spatial``) runs ``MK_dPC_SAMC`` on the polygons over
``polygons_resistance.tif`` with an absorption raster written from them
(``samc_absorption.tif``: 0 in the polygons, Fletcher et al.'s rate for 5 km
elsewhere), 4 and 8 directions, sum and mean, and stores its settlement
probabilities too (``makurhini_samc.csv.gz``).

``conefor`` builds ``tools/conefor/conefor_harness.cpp`` with the Conefor
Sensinode 2.2 sources (the last five nodes of each scene are Conefor's nodes
to add) (``LANDSCHAFT_CONEFOR_SRC``, default
``../Conefor Sensinode 2.2/src`` next to the repository) with g++ (Rtools:
``LANDSCHAFT_GXX_PATH``, default ``C:/rtools44``) and stores its indices and
node importances for the synthetic scenes in ``conefor.csv.gz``. The
test-suite compares landschaft against these files, so it runs without R or
Conefor.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import lsm_reference  # noqa: E402

OUT = ROOT / "tests" / "data" / "reference" / "conn"
R_SCRIPT = ROOT / "tools" / "conn_makurhini_reference.R"
CONEFOR_SRC = Path(os.environ.get("LANDSCHAFT_CONEFOR_SRC", ROOT.parent / "Conefor Sensinode 2.2" / "src"))
RTOOLS = Path(os.environ.get("LANDSCHAFT_GXX_PATH", "C:/rtools44"))


def _synthetic() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20260926)
    scenes = {}
    # Random points, lognormal areas, Euclidean distances.
    xy = rng.uniform(0, 50_000, (60, 2))
    area = rng.lognormal(0.0, 1.0, 60)
    scenes["points60"] = (np.arange(1, 61), area, np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1)))
    # Non-metric costs (least-cost-like): stepping stones beat direct links.
    n = 40
    c = rng.uniform(1_000, 20_000, (n, n))
    c = np.triu(c, 1)
    c = c + c.T
    scenes["cost40"] = (np.arange(101, 101 + n), rng.uniform(0.5, 30.0, n), c)
    # Four far-apart clusters: several components, cut nodes, unconnected pairs.
    centres = np.array([[0, 0], [40_000, 0], [0, 40_000], [40_000, 40_000]])
    pts = np.concatenate([centres[k] + rng.normal(0, 2_500, (12, 2)) for k in range(4)] + [rng.uniform(0, 40_000, (2, 2))])
    area = rng.gamma(2.0, 2.0, len(pts))
    scenes["clusters50"] = (np.arange(1, len(pts) + 1) * 3, area,
                            np.hypot(*(pts[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)))
    return scenes


# (metric, distance, probability, pij_min, la) per scene; "" = NULL in R.
JOBS = {
    "habitat": [("IIC", 10_000, "", "", ""), ("IIC", 20_000, "", "", ""), ("PC", 10_000, 0.5, 0.01, ""),
                ("PC", 40_000, 0.05, "", "LA")],
    "points60": [("IIC", 6_000, "", "", "LA"), ("IIC", 12_000, "", "", ""), ("PC", 8_000, 0.5, 0.01, "LA"),
                 ("PC", 15_000, 0.5, "", ""), ("PC", 5_000, 0.05, 0.001, "")],
    "cost40": [("IIC", 4_000, "", "", ""), ("IIC", 8_000, "", "", "LA"), ("PC", 5_000, 0.5, 0.01, ""),
               ("PC", 10_000, 0.2, "", "LA")],
    "clusters50": [("IIC", 4_000, "", "", ""), ("IIC", 30_000, "", "", ""), ("PC", 6_000, 0.5, 0.01, "LA"),
                   ("PC", 20_000, 0.5, "", "")],
}
# Conefor runs: (distance, probability, landscape area factor); binary links at
# the same distance.
CONEFOR_JOBS = {
    "points60": [(6_000, 0.5, 0), (12_000, 0.05, 3)],
    "cost40": [(4_000, 0.5, 0), (8_000, 0.2, 2)],
    "clusters50": [(4_000, 0.5, 2), (30_000, 0.5, 0)],
}


def landscape_area(attr: np.ndarray) -> float:
    """The LA of the jobs marked "LA": a round number above the total attribute."""
    return float(np.ceil(2.5 * attr.sum()))


def save_scenes(scenes) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, (ids, attr, dist) in scenes.items():
        i, j = np.triu_indices(len(ids), 1)
        arrays[f"{name}__ids"] = ids
        arrays[f"{name}__attr"] = attr
        arrays[f"{name}__upper"] = dist[i, j]
    np.savez_compressed(OUT / "scenes.npz", **arrays)


def load_scenes() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    data = np.load(OUT / "scenes.npz")
    names = sorted({k.split("__")[0] for k in data.files})
    out = {}
    for name in names:
        ids, attr, upper = data[f"{name}__ids"], data[f"{name}__attr"], data[f"{name}__upper"]
        n = len(ids)
        dist = np.zeros((n, n))
        i, j = np.triu_indices(n, 1)
        dist[i, j] = upper
        dist[j, i] = upper
        out[name] = (ids, attr, dist)
    return out


def synthetic_polygons():
    """Polygons of every kind Makurhini's centroids and edges meet."""
    import geopandas as gpd
    import shapely
    from shapely import affinity

    rng = np.random.default_rng(7)
    geoms = []
    for _ in range(18):  # boxes
        x, y = rng.uniform(0, 30_000, 2)
        w, h = rng.uniform(200, 2_000, 2)
        geoms.append(shapely.box(x, y, x + w, y + h))
    for _ in range(12):  # convex, rotated
        c = rng.uniform(0, 30_000, 2)
        pts = c + rng.normal(0, 600, (12, 2))
        geoms.append(affinity.rotate(shapely.MultiPoint(pts).convex_hull, rng.uniform(0, 90)))
    for _ in range(8):  # U shapes: the centroid falls outside
        x, y = rng.uniform(0, 30_000, 2)
        u = shapely.Polygon([(0, 0), (900, 0), (900, 900), (650, 900), (650, 250), (250, 250), (250, 900), (0, 900)])
        geoms.append(affinity.translate(affinity.rotate(u, rng.uniform(0, 360), origin=(450, 450)), x, y))
    for _ in range(5):  # with a hole
        x, y = rng.uniform(0, 30_000, 2)
        geoms.append(shapely.Polygon([(x, y), (x + 1500, y), (x + 1500, y + 1200), (x, y + 1200)],
                                     [[(x + 300, y + 300), (x + 900, y + 300), (x + 900, y + 800), (x + 300, y + 800)]]))
    for k in range(7):  # multipolygons, the last with two equal parts
        x, y = rng.uniform(0, 30_000, 2)
        parts = [shapely.box(x, y, x + 400 + 100 * k, y + 500), shapely.box(x + 1_500, y, x + 1_900, y + 700)]
        if k == 6:
            parts = [shapely.box(x, y, x + 500, y + 500), shapely.box(x + 2_000, y, x + 2_500, y + 500)]
        elif k % 2:
            parts.append(shapely.box(x, y + 2_000, x + 300, y + 2_300))
        geoms.append(shapely.MultiPolygon(parts))
    a = shapely.box(10_000, 10_000, 11_000, 11_000)  # touching and overlapping neighbours
    geoms += [a, shapely.box(11_000, 10_200, 11_800, 10_900), shapely.box(10_500, 10_900, 10_700, 11_600)]
    ids = np.arange(len(geoms)) * 5 + 11
    return gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs="EPSG:3857")


def synthetic_resistance(path: Path) -> None:
    """A smooth resistance (1-100) over the polygons, with NA walls that cut some of them off."""
    import rasterio
    from rasterio.transform import from_origin
    from scipy import ndimage

    rng = np.random.default_rng(11)
    ny, nx, res = 95, 95, 400.0  # covers every polygon (gdistance drops points outside, shifting its rows)
    values = ndimage.gaussian_filter(rng.random((ny, nx)), 3)
    values = 1 + 99 * (values - values.min()) / (values.max() - values.min())
    values[20:22, 5:60] = np.nan   # a wall
    values[40:70, 50:52] = np.nan  # another, closing a corner
    values[40:42, 50:85] = np.nan
    with rasterio.open(path, "w", driver="GTiff", width=nx, height=ny, count=1, dtype="float64",
                       crs="EPSG:3857", transform=from_origin(-2_000.0, 36_000.0, res, res), nodata=np.nan,
                       compress="deflate") as dst:
        dst.write(values, 1)


def run_spatial() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        raise SystemExit("Rscript not found")
    out = OUT / "spatial"
    out.mkdir(parents=True, exist_ok=True)
    polygons = OUT / "polygons.gpkg"
    synthetic_polygons().to_file(polygons, driver="GPKG")
    resistance = OUT / "polygons_resistance.tif"
    synthetic_resistance(resistance)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run([str(rscript), str(R_SCRIPT), "spatial", str(tmp), str(polygons), str(resistance)], check=True)
        for name in ("habitat_nodes_raster.tif", "resistance_matrix.tif"):
            (OUT / name).write_bytes((tmp / name).read_bytes())
        for f in sorted(tmp.glob("*.csv")):
            table = pd.read_csv(f, float_precision="round_trip")
            if {"From", "To", "Distance"} <= set(table.columns) and len(table) > 5_000:
                keep = (table["Distance"] <= 30_000) | (np.arange(len(table)) % 17 == 0)
                table = table[keep]
            table.to_csv(out / f"{f.stem}.csv.gz", index=False, float_format="%.17g")
            print(f.stem, len(table))


LINK_JOBS = {
    "points60": [("IIC", 8_000, "", ""), ("PC", 8_000, 0.5, 0.01)],
    "cost40": [("IIC", 6_000, "", ""), ("PC", 6_000, 0.3, 0.01)],
    "clusters50": [("IIC", 5_000, "", ""), ("PC", 10_000, 0.5, "")],
}


def changed_matrix(dist: np.ndarray, seed: int) -> np.ndarray:
    """40 pairs brought closer or moved apart (factors 0.3 to 1.6)."""
    rng = np.random.default_rng(seed)
    n = len(dist)
    i, j = np.triu_indices(n, 1)
    pick = rng.choice(len(i), 40, replace=False)
    out = dist.copy()
    factor = rng.uniform(0.3, 1.6, 40)
    out[i[pick], j[pick]] = dist[i[pick], j[pick]] * factor
    out[j[pick], i[pick]] = out[i[pick], j[pick]]
    return out


def run_links() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        raise SystemExit("Rscript not found")
    scenes = load_scenes()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jobs = []
        for k, (name, specs) in enumerate(LINK_JOBS.items()):
            ids, attr, dist = scenes[name]
            node_file, matrix_file, change_file = (tmp / f"{name}_{x}.csv" for x in ("nodes", "matrix", "change"))
            pd.DataFrame({"id": ids, "attribute": [repr(float(v)) for v in attr]}).to_csv(node_file, index=False)
            np.savetxt(matrix_file, dist, delimiter=",", fmt="%.17g")
            np.savetxt(change_file, changed_matrix(dist, 100 + k), delimiter=",", fmt="%.17g")
            for metric, d, p, pij in specs:
                jobs.append({"scene": name, "nodes": node_file.as_posix(), "matrix": matrix_file.as_posix(),
                             "change": change_file.as_posix(), "metric": metric, "distance": d, "probability": p,
                             "pij_min": pij, "restore": 5})
        pd.DataFrame(jobs).to_csv(tmp / "jobs.csv", index=False)
        subprocess.run([str(rscript), str(R_SCRIPT), "links", str(tmp / "jobs.csv"), str(tmp / "out.csv")],
                       check=True)
        out = pd.read_csv(tmp / "out.csv", float_precision="round_trip", keep_default_na=False,
                          dtype={"probability": str, "pij_min": str})
        out.to_csv(OUT / "makurhini_links.csv.gz", index=False)
        print(f"links: {len(out)} values")


DECA_LA_VECTOR = 400.0   # km2
DECA_LA_RASTER = 3.5e6   # ha


def deca_steps() -> pd.DataFrame:
    """The nodes kept at each of three times."""
    import geopandas as gpd
    import rasterio

    rows = []
    poly_ids = gpd.read_file(OUT / "polygons.gpkg")["id"].to_numpy()
    with rasterio.open(OUT / "habitat_nodes_raster.tif") as src:
        values = src.read(1)
        nodata = src.nodata
    raster_ids = np.unique(values[values != nodata]) if nodata is not None else np.unique(values)
    for kind, ids in (("vector", poly_ids), ("raster", raster_ids)):
        for time, drop in ((1, 0), (2, 5), (3, 3)):
            kept = ids if drop == 0 else ids[np.arange(len(ids)) % drop != 1]
            if time == 3:
                kept = kept[np.arange(len(kept)) % 4 != 2]
            rows += [(kind, time, int(i)) for i in kept]
    return pd.DataFrame(rows, columns=["kind", "time", "id"])


def run_deca() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        raise SystemExit("Rscript not found")
    steps = deca_steps()
    steps.to_csv(OUT / "deca_steps.csv", index=False)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run([str(rscript), str(R_SCRIPT), "deca", str(OUT / "polygons.gpkg"),
                        str(OUT / "habitat_nodes_raster.tif"), str(OUT / "deca_steps.csv"), repr(DECA_LA_VECTOR),
                        repr(DECA_LA_RASTER), str(tmp / "out.csv")], check=True)
        out = pd.read_csv(tmp / "out.csv", float_precision="round_trip")
        out.to_csv(OUT / "makurhini_deca.csv.gz", index=False, float_format="%.17g")
        print(f"deca: {len(out)} rows")


def densified_box(x0, y0, x1, y1, per_edge=40, angle=0.0):
    """A box with `per_edge` points on each side (mapshaper keeps its corners)."""
    import shapely
    from shapely import affinity

    t = np.linspace(0, 1, per_edge, endpoint=False)
    xs = np.concatenate([x0 + (x1 - x0) * t, np.full_like(t, x1), x1 - (x1 - x0) * t, np.full_like(t, x0)])
    ys = np.concatenate([np.full_like(t, y0), y0 + (y1 - y0) * t, np.full_like(t, y1), y1 - (y1 - y0) * t])
    poly = shapely.Polygon(np.column_stack([xs, ys]))
    return affinity.rotate(poly, angle, origin="centroid") if angle else poly


def protconn_geometry():
    import geopandas as gpd

    rng = np.random.default_rng(17)
    regions = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[densified_box(0, 0, 20_000, 15_000, 80),
                                                         densified_box(25_000, 0, 32_000, 6_000, 80)],
                               crs="EPSG:3857")
    boxes = []
    for _ in range(22):  # inside the first region (some touch or overlap)
        x, y = rng.uniform(500, 17_500), rng.uniform(500, 12_500)
        w, h = rng.uniform(300, 2_000, 2)
        boxes.append(densified_box(x, y, x + w, y + h))
    boxes.append(densified_box(18_500, 6_000, 21_500, 8_000))   # across the edge
    for _ in range(8):  # outside: transboundary candidates
        x, y = rng.uniform(-6_000, 26_000), rng.uniform(15_500, 19_000)
        w, h = rng.uniform(400, 1_800, 2)
        boxes.append(densified_box(x, y, x + w, y + h))
    for _ in range(5):  # in the second region
        x, y = rng.uniform(25_300, 30_500), rng.uniform(300, 4_800)
        w, h = rng.uniform(300, 900, 2)
        boxes.append(densified_box(x, y, x + w, y + h))
    pas = gpd.GeoDataFrame({"id": np.arange(1, len(boxes) + 1)}, geometry=boxes, crs="EPSG:3857")
    return pas, regions


PROTCONN_JOBS = [
    # case, region, distances, transboundary, type, bound
    ("plain", 1, "1000;5000", None, "nodes", False),
    ("trans_nodes", 1, "1000;5000", 3000.0, "nodes", False),
    ("trans_region", 1, "2000", 3000.0, "region", False),
    ("bound", 1, "2000", 3000.0, "nodes", True),
    ("second", 2, "1500", None, "nodes", True),
]


def run_protconn() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        raise SystemExit("Rscript not found")
    pas, regions = protconn_geometry()
    pas.to_file(OUT / "protconn_areas.gpkg", driver="GPKG")
    regions.to_file(OUT / "protconn_regions.gpkg", driver="GPKG")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pd.DataFrame(PROTCONN_JOBS, columns=["case", "region", "distance", "transboundary", "type", "bound"]).to_csv(
            tmp / "jobs.csv", index=False)
        subprocess.run([str(rscript), str(R_SCRIPT), "protconn", str(OUT / "protconn_areas.gpkg"),
                        str(OUT / "protconn_regions.gpkg"), str(tmp / "jobs.csv"), str(tmp / "out.csv")], check=True)
        out = pd.read_csv(tmp / "out.csv", float_precision="round_trip", keep_default_na=False)
        out.to_csv(OUT / "makurhini_protconn.csv.gz", index=False)
        print(f"protconn: {len(out)} values")


def protconn_raster_file() -> None:
    """The protected areas rasterized at 100 m, ids renumbered in the order they appear."""
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from rasterio.transform import from_origin

    pas = gpd.read_file(OUT / "protconn_areas.gpkg")
    transform = from_origin(-8000.0, 21000.0, 100.0, 100.0)
    lab = features.rasterize([(g, int(i)) for g, i in zip(pas.geometry, pas["id"])], out_shape=(240, 360),
                             transform=transform, fill=0, dtype="int32")
    flat = lab.ravel()
    _, first = np.unique(flat[flat > 0], return_index=True)
    order = np.unique(flat[flat > 0])[np.argsort(first)]
    remap = np.zeros(lab.max() + 1, dtype=np.int32)
    remap[order] = np.arange(1, len(order) + 1)
    with rasterio.open(OUT / "protconn_areas.tif", "w", driver="GTiff", width=360, height=240, count=1,
                       dtype="int32", crs=pas.crs, transform=transform, nodata=0, compress="deflate") as dst:
        dst.write(remap[lab], 1)


def run_grids() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        raise SystemExit("Rscript not found")
    out = OUT / "grid"
    out.mkdir(parents=True, exist_ok=True)
    protconn_raster_file()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run([str(rscript), str(R_SCRIPT), "grid", str(OUT / "protconn_areas.gpkg"),
                        str(OUT / "protconn_regions.gpkg"), str(tmp)], check=True)
        subprocess.run([str(rscript), str(R_SCRIPT), "protconn_raster", str(OUT / "protconn_areas.tif"),
                        str(OUT / "protconn_regions.gpkg"), str(tmp / "protconn_raster.csv")], check=True)
        subprocess.run([str(rscript), str(R_SCRIPT), "deca_grid", str(OUT / "polygons.gpkg"),
                        str(OUT / "deca_steps.csv"), str(tmp / "deca_grid.csv")], check=True)
        for f in sorted(tmp.glob("*.csv")):
            table = pd.read_csv(f, keep_default_na=False, dtype=str)
            table.to_csv(out / f"{f.stem}.csv.gz", index=False)
            print(f.stem, len(table))


BETWEENNESS_JOBS = {"points60": (8_000, 0.5), "cost40": (6_000, 0.3), "clusters50": (5_000, 0.5)}


def run_betweenness() -> None:
    """BC, BC(IIC) and BC(PC) of Conefor 2.7.1's command line (``CONEFOR_EXE``)."""
    exe = os.environ.get("CONEFOR_EXE")
    if not exe or not Path(exe).exists():
        raise SystemExit("set CONEFOR_EXE to Conefor's command-line executable (coneforWin64.exe, conefor.org)")
    scenes = load_scenes()
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, (d, p) in BETWEENNESS_JOBS.items():
            ids, attr, dist = scenes[name]
            (tmp / "nodes.txt").write_text("".join(f"{i}\t{float(v)!r}\n" for i, v in zip(ids, attr)))
            i, j = np.triu_indices(len(ids), 1)
            (tmp / "dist.txt").write_text("".join(f"{ids[x]}\t{ids[y]}\t{float(dist[x, y])!r}\n"
                                                  for x, y in zip(i, j)))
            subprocess.run([exe, "-nodeFile", "nodes.txt", "-conFile", "dist.txt", "-t", "dist", "all",
                            "-confAdj", str(d), "-IIC", "-BC", "-BCIIC", "-confProb", str(d), str(p), "-PC", "-BCPC",
                            "-prefix", name], cwd=tmp, check=True, capture_output=True)
            table = pd.read_csv(tmp / f"{name}_node_importances.txt", sep="\t", dtype=str).dropna(axis=1, how="all")
            table.insert(0, "scene", name)
            table.insert(1, "distance", d)
            table.insert(2, "probability", p)
            frames.append(table)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / "conefor_betweenness.csv.gz", index=False)
    print("betweenness", len(out))


def run_centrality() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        raise SystemExit("Rscript not found")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "centrality.csv"
        subprocess.run([str(rscript), str(R_SCRIPT), "centrality", str(OUT / "polygons.gpkg"),
                        str(OUT / "habitat_nodes_raster.tif"), str(out)], check=True)
        table = pd.read_csv(out, keep_default_na=False, dtype=str)
    table.to_csv(OUT / "makurhini_centrality.csv.gz", index=False)
    print("centrality", len(table))


def run_commute() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        raise SystemExit("Rscript not found")
    out = OUT / "spatial"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run([str(rscript), str(R_SCRIPT), "commute", str(tmp), str(OUT / "polygons.gpkg"),
                        str(OUT / "polygons_resistance.tif")], check=True)
        for f in sorted(tmp.glob("*.csv")):
            table = pd.read_csv(f, float_precision="round_trip")
            if len(table) > 5_000:
                table = table[np.arange(len(table)) % 7 == 0]
            table.to_csv(out / f"{f.stem}.csv.gz", index=False, float_format="%.17g")
            print(f.stem, len(table))


def samc_absorption(path: Path, dispersal: float = 5000.0) -> None:
    """Absorption for MK_dPC_SAMC: 0 on the polygons (cell centres inside), Fletcher et al.'s rate elsewhere."""
    import geopandas as gpd
    import rasterio
    from rasterio import features

    with rasterio.open(OUT / "polygons_resistance.tif") as src:
        res = src.read(1, masked=True)
        transform, crs, shape = src.transform, src.crs, src.shape
        cell = src.res[0]
    poly = gpd.read_file(OUT / "polygons.gpkg")
    inside = features.rasterize(((g, 1) for g in poly.geometry), out_shape=shape, transform=transform, fill=0,
                                all_touched=False).astype(bool)
    rate = np.exp(-1.247 - 1.788 * np.log(dispersal / cell))
    values = np.where(inside, 0.0, rate)
    values[np.ma.getmaskarray(res)] = np.nan
    with rasterio.open(path, "w", driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="float64",
                       crs=crs, transform=transform, nodata=np.nan, compress="deflate") as dst:
        dst.write(values, 1)


def run_samc() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        raise SystemExit("Rscript not found")
    samc_absorption(OUT / "samc_absorption.tif")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "samc.csv"
        subprocess.run([str(rscript), str(R_SCRIPT), "samc", str(OUT / "polygons.gpkg"),
                        str(OUT / "polygons_resistance.tif"), str(OUT / "samc_absorption.tif"), str(out)], check=True)
        table = pd.read_csv(out, keep_default_na=False, dtype=str)
    table.to_csv(OUT / "makurhini_samc.csv.gz", index=False)
    print("samc", len(table))


def run_focal() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        raise SystemExit("Rscript not found")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "focal.csv"
        subprocess.run([str(rscript), str(R_SCRIPT), "focal", str(OUT / "polygons.gpkg"),
                        str(OUT / "protconn_areas.gpkg"), str(out)], check=True)
        table = pd.read_csv(out, keep_default_na=False, dtype=str)
    table.to_csv(OUT / "makurhini_focal.csv.gz", index=False)
    print("focal", len(table))


def run_makurhini() -> None:
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        raise SystemExit("Rscript not found")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run([str(rscript), str(R_SCRIPT), "export", str(tmp)], check=True)
        nodes = pd.read_csv(tmp / "habitat_nodes.csv", float_precision="round_trip")
        pairs = pd.read_csv(tmp / "habitat_pairs.csv", float_precision="round_trip")
        ids = nodes["id"].to_numpy()
        index = pd.Index(ids)
        dist = np.zeros((len(ids), len(ids)))
        fi, fj = index.get_indexer(pairs["from"]), index.get_indexer(pairs["to"])
        dist[fi, fj] = pairs["distance"].to_numpy()
        dist[fj, fi] = pairs["distance"].to_numpy()
        scenes = {"habitat": (ids, nodes["attribute"].to_numpy(), dist), **_synthetic()}
        save_scenes(scenes)
        scenes = load_scenes()
        jobs = []
        for name, (ids, attr, dist) in scenes.items():
            node_file, matrix_file = tmp / f"{name}_nodes.csv", tmp / f"{name}_matrix.csv"
            pd.DataFrame({"id": ids, "attribute": [repr(float(v)) for v in attr]}).to_csv(node_file, index=False)
            np.savetxt(matrix_file, dist, delimiter=",", fmt="%.17g")
            for metric, d, p, pij, la in JOBS[name]:
                jobs.append({"scene": name, "nodes": node_file.as_posix(), "matrix": matrix_file.as_posix(),
                             "metric": metric, "distance": d, "probability": p, "pij_min": pij,
                             "la": repr(landscape_area(attr)) if la == "LA" else ""})
        pd.DataFrame(jobs).to_csv(tmp / "jobs.csv", index=False)
        subprocess.run([str(rscript), str(R_SCRIPT), "run", str(tmp / "jobs.csv"), str(tmp / "out.csv")], check=True)
        out = pd.read_csv(tmp / "out.csv", float_precision="round_trip", keep_default_na=False,
                          dtype={"probability": str, "pij_min": str, "la": str, "id": str})
        out.to_csv(OUT / "makurhini.csv.gz", index=False)
        print(f"makurhini: {len(out)} values")


def build_conefor(exe: Path) -> None:
    """Compile Conefor's landscape.cpp with tools/conefor/conefor_harness.cpp.

    Borland C++ bound temporaries to non-const references; two calls in
    landscape.cpp rely on it, so a copy of the file gets named temporaries.
    """
    work = exe.parent / "conefor_src"
    work.mkdir(exist_ok=True)
    for header in CONEFOR_SRC.glob("*.h"):
        (work / header.name).write_bytes(header.read_bytes())
    code = (CONEFOR_SRC / "landscape.cpp").read_bytes().decode("latin-1")
    patches = {
        "nodes[i].list_adjacency.inserir(j-1,ite1);":
            "{ int jm1 = j - 1; nodes[i].list_adjacency.inserir(jm1, ite1); }",
        "nodes[i].list_adjacency.inserirFi(nNodes-1);":
            "{ int last = nNodes - 1; nodes[i].list_adjacency.inserirFi(last); }",
    }
    for old, new in patches.items():
        if old not in code:
            raise SystemExit(f"unexpected Conefor source: {old!r} not found")
        code = code.replace(old, new)
    (work / "landscape.cpp").write_bytes(code.encode("latin-1"))
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(RTOOLS / "x86_64-w64-mingw32.static.posix" / "bin"), env["PATH"],
                                   str(RTOOLS / "usr" / "bin")])
    here = ROOT / "tools" / "conefor"
    gxx = RTOOLS / "x86_64-w64-mingw32.static.posix" / "bin" / "g++.exe"
    subprocess.run([str(gxx) if gxx.exists() else "g++", "-std=c++17", "-O2", "-fpermissive", "-w", "-I", str(here / "shim"), "-I", str(work),
                    str(here / "conefor_harness.cpp"), str(work / "landscape.cpp"), "-o", str(exe)],
                   check=True, env=env)


def run_conefor() -> None:
    scenes = load_scenes()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exe = tmp / "conefor_harness.exe"
        build_conefor(exe)
        for name, jobs in CONEFOR_JOBS.items():
            ids, attr, dist = scenes[name]
            for d, p, la_factor in jobs:
                la = int(np.ceil(la_factor * attr.sum())) if la_factor else 0
                f = tmp / "in.txt"
                # The last five nodes are the nodes to add (Conefor's "adding nodes").
                base, extra = slice(0, len(ids) - 5), slice(len(ids) - 5, len(ids))
                with open(f, "w") as out:
                    out.write(f"{len(ids) - 5} {d!r} {d!r} {p!r} 0 {la}\n")
                    for i, a in zip(ids[base], attr[base]):
                        out.write(f"{i} {float(a)!r}\n")
                    for row in dist[base, base]:
                        out.write(" ".join(repr(float(v)) for v in row) + "\n")
                    out.write("5\n")
                    for i, a in zip(ids[extra], attr[extra]):
                        out.write(f"{i} {float(a)!r}\n")
                    for row in dist[extra, base]:
                        out.write(" ".join(repr(float(v)) for v in row) + "\n")
                text = subprocess.run([str(exe), str(f)], check=True, capture_output=True, text=True).stdout
                header = None
                for line in text.splitlines():
                    parts = line.split(",")
                    if parts[0] == "overall":
                        rows.append((name, d, p, la, "overall", "", parts[1], parts[2]))
                    elif parts[1] == "label":
                        header = parts[2:]
                    else:
                        for key, value in zip(header, parts[2:]):
                            rows.append((name, d, p, la, parts[0], parts[1], key, value))
    frame = pd.DataFrame(rows, columns=["scene", "distance", "probability", "la", "level", "id", "name", "value"])
    frame.to_csv(OUT / "conefor.csv.gz", index=False)
    print(f"conefor: {len(frame)} values")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["makurhini", "conefor", "spatial", "links", "deca", "protconn", "grids", "centrality", "focal", "commute", "samc"]
    if "makurhini" in parts:
        run_makurhini()
    if "conefor" in parts:
        run_conefor()
    if "spatial" in parts:
        run_spatial()
    if "links" in parts:
        run_links()
    if "deca" in parts:
        run_deca()
    if "protconn" in parts:
        run_protconn()
    if "grids" in parts:
        run_grids()
    if "centrality" in parts:
        run_centrality()
    if "betweenness" in parts:
        run_betweenness()
    if "focal" in parts:
        run_focal()
    if "commute" in parts:
        run_commute()
    if "samc" in parts:
        run_samc()
