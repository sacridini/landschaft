# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden habitat-graph outputs with Makurhini (R) and Conefor Sensinode 2.2.

Usage::

    python tools/make_reference_conn.py [makurhini] [conefor] [spatial] [links]

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
    parts = sys.argv[1:] or ["makurhini", "conefor", "spatial", "links"]
    if "makurhini" in parts:
        run_makurhini()
    if "conefor" in parts:
        run_conefor()
    if "spatial" in parts:
        run_spatial()
    if "links" in parts:
        run_links()
