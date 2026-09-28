# SPDX-License-Identifier: GPL-3.0-or-later
"""Inputs of the Biodiverse reference runs (tools/make_reference_bio.py).

Each case gives the arguments of Biodiverse's import (``biodiverse``) and of the
landschaft call that must reproduce it (``landschaft``). File names are
relative to ``tests/data/bio``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "bio"
REFERENCE = Path(__file__).resolve().parent / "data" / "reference" / "bio"

SITE = "Example_site_data.csv"


def _bd(file, group_columns, label_columns, cell_sizes, cell_origins=None, **extra):
    biodiverse = {
        "file": file,
        "group_columns": group_columns,
        "label_columns": label_columns,
        "cell_sizes": cell_sizes,
        **extra,
    }
    if cell_origins is not None:
        biodiverse["cell_origins"] = cell_origins
    landschaft = {
        "group_columns": group_columns,
        "label_columns": label_columns,
        "cell_sizes": cell_sizes,
        "cell_origins": cell_origins,
    }
    rename = {"sample_count_columns": "sample_count_columns", "data_in_matrix_form": "matrix_form",
              "label_start_col": "label_start_col", "binarise_counts": "binarise_counts",
              "allow_empty_groups": "allow_empty_groups", "allow_empty_labels": "allow_empty_labels",
              "include_columns": "include_columns", "exclude_columns": "exclude_columns",
              "input_sep_char": "sep", "input_quote_char": "quote"}
    for k, v in extra.items():
        landschaft[rename[k]] = v
    return {"biodiverse": biodiverse, "landschaft": landschaft, "file": file}


BASEDATA_CASES = {
    "site_100k": _bd(SITE, [3, 4], [1, 2], [100000, 100000]),
    "site_origin": _bd(SITE, [3, 4], [1, 2], [200000, 150000], [50000, 25000]),
    "site_raw": _bd(SITE, [3, 4], [1, 2], [0, 0]),
    "site_text_axis": _bd(SITE, [2, 4], [1], [-1, 100000]),
    "site_counts": _bd(SITE, [3, 4], [2], [50000, 50000], sample_count_columns=[0]),
    "site_binarise": _bd(SITE, [3, 4], [1, 2], [300000, 300000], sample_count_columns=[0], binarise_counts=1),
    "site_one_axis": _bd(SITE, [3], [1, 2], [100000]),
    "site_matrix_form": _bd("Example_site_data_matrix_form.csv", [1, 2], [], [25000, 25000],
                            data_in_matrix_form=1, label_start_col=3),
    "decimal": _bd("synthetic_decimal.csv", [1, 2], [0], [0.1, 0.1], [0.03, -0.02], sample_count_columns=[3]),
    "decimal_empty": _bd("synthetic_decimal.csv", [1, 2], [0], [0.25, 0.25], sample_count_columns=[3],
                         allow_empty_groups=1, allow_empty_labels=1),
    "quoted": _bd("synthetic_quoted.csv", [2, 3], [0, 1], [-1, 1.5], exclude_columns=[4],
                  input_sep_char=",", input_quote_char='"'),
}

# Biodiverse's feature import (import_data_shapefile); landschaft: BaseData.from_vector
VECTOR_CASES = {
    "points": {"files": ["site_points.shp"], "labels": ["genus", "species"], "cell_sizes": [100000, 100000]},
    "points_counts": {"files": ["site_points.shp"], "labels": ["species"], "cell_sizes": [50000, 50000],
                      "counts": ["num"], "cell_origins": [10000, 20000]},
    "points_raw": {"files": ["site_points.shp"], "labels": ["species"], "cell_sizes": [0, 0]},
    "polygons_cells": {"files": ["polygon data.shp"], "labels": ["BINOMIAL"], "cell_sizes": [100000, 100000]},
    "polygons_cells_origin": {"files": ["polygon data.shp"], "labels": ["BINOMIAL"], "cell_sizes": [200000, 150000],
                              "cell_origins": [50000, 25000]},
    "polygons_area": {"files": ["polygon data.shp"], "labels": ["BINOMIAL"], "cell_sizes": [100000, 100000],
                      "measure": "area"},
    "lines_cells": {"files": ["polyline data.shp"], "labels": ["BINOMIAL"], "cell_sizes": [100000, 100000]},
    "lines_length": {"files": ["polyline data.shp"], "labels": ["BINOMIAL"], "cell_sizes": [100000, 100000],
                     "measure": "length"},
}

TREE_FILES = ["Example_tree.nex", "Example_tree_with_colours.nex", "tree_with_nonmatching_names.nex"]

NEWICK_CASES = {
    "simple": "((a:1,b:2):1,c:3);",
    "unnamed": "(((a,b),c),d);",
    "internal_names": "((a:0.1,b:0.2)ab:0.05,(c:0.3,d:0.4)cd:0.06)root:0;",
    "duplicates": "((a:1,a:2)x:1,(b:1,x:1):2);",
    "quoted": "(('a b':1,'c''d':2):1,'e:f':3);",
    "forest": "(a:1,b:2),(c:1,d:1)",
    "all_zero": "((a:0,b:0):0,c:0);",
    "exponent": "((a:1e-3,b:2.5E2):-0.5,c:.25);",
    "boot": "((a:1,b:2)[90]:1,c:3)[x];",
    "phylip_two": "(a:1,b:2);((a:1,b:1):1,c:1);",
}

MATRIX_CASES = {
    "full": {"file": "Example_matrix.txt", "params": {"ELEMENT_COLUMNS": [1, 2]}, "sparse": 0,
             "landschaft": {"element_columns": [1, 2]}},
    "sparse": {"file": "Example_matrix_sparse.txt", "params": {}, "sparse": 1,
               "args": {"label_row_columns": [0], "label_col_columns": [1], "value_column": 2},
               "landschaft": {"sparse": True, "row_columns": [0], "col_columns": [1], "value_column": 2}},
}

RASTER_CASES = {
    "bands_single": {"files": ["raster_single.tif"], "labels_as_bands": 1, "cell_sizes": [10, 10]},
    "bands_multi": {"files": ["raster_multi.tif"], "labels_as_bands": 1, "cell_sizes": [10, 10]},
    "bands_coarse": {"files": ["raster_multi.tif"], "labels_as_bands": 1, "cell_sizes": [30, 20], "cell_origins": [5, 3]},
    "values": {"files": ["raster_values.tif"], "labels_as_bands": 0, "cell_sizes": [10, 10]},
    "values_coarse": {"files": ["raster_values.tif"], "labels_as_bands": 0, "cell_sizes": [25, 25]},
    "two_files": {"files": ["raster_single.tif", "raster_multi.tif"], "labels_as_bands": 1, "cell_sizes": [10, 10]},
}


def write_synthetic_data() -> None:
    """Deterministic synthetic inputs (written once, committed)."""
    rng = np.random.default_rng(20260928)
    lines = ["label,x,y,count"]
    for i in range(400):
        lab = f"sp{rng.integers(0, 25)}"
        # coordinates near cell edges of 0.1 and 0.25 grids, both signs
        x = round(float(rng.integers(-30, 30)) * 0.05 + float(rng.choice([0.0, 1e-12, -1e-12, 0.0249999, 0.03])), 13)
        y = round(float(rng.integers(-30, 30)) * 0.05 + float(rng.choice([0.0, 0.02, -0.02, 0.07])), 13)
        c = rng.choice(["1", "2", "0.5", "2.25", "0", "3", "", "1"])
        lines.append(f"{lab},{x!r},{y!r},{c}")
    (DATA / "synthetic_decimal.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = ["genus,species,region,y,skip"]
    genera = ["Acacia", "Eucalyptus sp.", "Genus:odd", "Quote'd"]
    regions = ["north", "south", "east west", "region10", "region9", "Region2"]
    for i in range(200):
        g = genera[rng.integers(0, len(genera))]
        s = f"sp {rng.integers(0, 6)}" if i % 3 else f"sp{rng.integers(0, 6)}"
        r = regions[rng.integers(0, len(regions))]
        y = float(rng.integers(0, 40)) * 0.75
        skip = "1" if i % 17 == 0 else ("0" if i % 5 else "")
        gq = '"' + g + '"' if ("," in g or '"' in g or " " in g) else g
        rows.append(f"{gq},{s},{r},{y!r},{skip}")
    (DATA / "synthetic_quoted.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_points() -> None:
    """Point shapefile of the Biodiverse site data."""
    import geopandas as gpd
    import pandas as pd

    df = pd.read_csv(DATA / SITE, dtype={"x": float, "y": float})
    gdf = gpd.GeoDataFrame(df[["num", "genus", "species"]], geometry=gpd.points_from_xy(df["x"], df["y"]))
    gdf.to_file(DATA / "site_points.shp")


def write_rasters() -> None:
    """Small GeoTIFFs for the raster import."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(7)
    tf = from_origin(100.0, 250.0, 10.0, 10.0)
    z = rng.integers(0, 4, size=(12, 9)).astype(np.float32)
    z[0, 0] = -9999
    with rasterio.open(DATA / "raster_single.tif", "w", driver="GTiff", width=9, height=12, count=1,
                       dtype="float32", transform=tf, nodata=-9999) as dst:
        dst.write(z, 1)
    zz = rng.integers(0, 3, size=(3, 12, 9)).astype(np.int16)
    zz[1, 2, 3] = -1
    with rasterio.open(DATA / "raster_multi.tif", "w", driver="GTiff", width=9, height=12, count=3,
                       dtype="int16", transform=tf, nodata=-1) as dst:
        dst.write(zz)
        dst.set_band_description(1, "sp_a")
        dst.set_band_description(2, "sp_b")
    v = rng.choice([1, 2, 5, 7], size=(12, 9)).astype(np.int32)
    v[3, 3] = 0
    with rasterio.open(DATA / "raster_values.tif", "w", driver="GTiff", width=9, height=12, count=1,
                       dtype="int32", transform=tf, nodata=0) as dst:
        dst.write(v, 1)
