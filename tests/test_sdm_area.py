# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.sdm.accessible_area: geodesic buffers, convex hulls, regions, clipping."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

import landschaft as ls


def area_km2(gdf) -> float:
    return float(gdf.to_crs("EPSG:6933").area.iloc[0]) / 1e6


def test_buffer_is_a_geodesic_circle():
    occ = pd.DataFrame({"lon": [10.0], "lat": [0.0]})
    m = ls.sdm.accessible_area(occ, distance=100_000, resolution=256)
    assert m.crs.to_epsg() == 4326
    assert area_km2(m) == pytest.approx(np.pi * 100.0**2, rel=2e-3)
    # the same radius in metres at 60 degrees north: wider in longitude
    hi = ls.sdm.accessible_area(pd.DataFrame({"lon": [10.0], "lat": [60.0]}), distance=100_000)
    w0 = np.ptp(m.total_bounds[[0, 2]])
    w60 = np.ptp(hi.total_bounds[[0, 2]])
    assert w60 == pytest.approx(w0 * 2, rel=0.02)


def test_buffer_merges_and_accepts_any_crs():
    occ = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0.0, 0.5, 30.0], [0.0, 0.0, 0.0]), crs="EPSG:4326")
    m = ls.sdm.accessible_area(occ.to_crs("EPSG:3857"), distance=100_000)
    geom = m.geometry.iloc[0]
    assert geom.geom_type == "MultiPolygon" and len(geom.geoms) == 2
    assert geom.contains(shapely.Point(0.25, 0.0)) and not geom.contains(shapely.Point(15.0, 0.0))


def test_antimeridian_is_folded():
    m = ls.sdm.accessible_area(pd.DataFrame({"lon": [179.5, -179.5], "lat": [0.0, 0.0]}), distance=200_000)
    xmin, _, xmax, _ = m.total_bounds
    assert xmin == -180.0 and xmax == 180.0
    assert m.geometry.iloc[0].contains(shapely.Point(-179.9, 0.0))


def test_convex_hull_regions_and_within():
    occ = pd.DataFrame({"lon": [0.0, 2.0, 1.0, 1.0], "lat": [0.0, 0.0, 2.0, 0.5]})
    hull = ls.sdm.accessible_area(occ, method="convex_hull")
    assert hull.geometry.iloc[0].equals(shapely.Polygon([(0, 0), (2, 0), (1, 2)]))
    wider = ls.sdm.accessible_area(occ, method="convex_hull", distance=50_000)
    assert wider.geometry.iloc[0].contains(hull.geometry.iloc[0])
    regions = gpd.GeoDataFrame(geometry=[shapely.box(-1, -1, 1.5, 3), shapely.box(1.5, -1, 3, 3),
                                         shapely.box(5, 5, 6, 6)], crs="EPSG:4326")
    reg = ls.sdm.accessible_area(occ, method="regions", regions=regions)
    assert reg.geometry.iloc[0].equals(shapely.box(-1, -1, 3, 3).normalize()) or \
        reg.geometry.iloc[0].area == pytest.approx(16.0)
    cut = ls.sdm.accessible_area(occ, method="regions", regions=regions, within=shapely.box(0, 0, 10, 10))
    assert cut.geometry.iloc[0].area == pytest.approx(9.0)


def test_errors():
    occ = pd.DataFrame({"lon": [0.0, 1.0], "lat": [0.0, 0.0]})
    with pytest.raises(ValueError, match="needs a distance"):
        ls.sdm.accessible_area(occ)
    with pytest.raises(ValueError, match="three records"):
        ls.sdm.accessible_area(occ, method="convex_hull")
    with pytest.raises(ValueError, match="no record with coordinates"):
        ls.sdm.accessible_area(pd.DataFrame({"lon": [np.nan], "lat": [np.nan]}), distance=1000)
