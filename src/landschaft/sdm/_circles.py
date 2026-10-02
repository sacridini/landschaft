# SPDX-License-Identifier: GPL-3.0-or-later
"""Circles of a radius in metres around points, as CoordinateCleaner draws them.

Two polygonisations, as in R (CoordinateCleaner's ``geod`` switch), with R's
vertices to the last bit (C++, src/cpp/sdm/geocircle): 73 vertices at
bearings 0, 5, ..., 360 (``geosphere::destPoint``, GeographicLib 2.7) or
terra's point buffer, 40 vertices at bearings 0, 9, ..., 351 (GeographicLib C
1.52). Circles crossing the antimeridian are split, those holding a pole
closed through it (CoordinateCleaner's geosphere circles are not: a bug, fixed
here).
"""

from __future__ import annotations

import numpy as np

from .. import _core

__all__ = ["circles", "circle_extent", "planar_circle"]


def _polygon(rings):
    import shapely

    polys = [shapely.Polygon(r) for r in rings if len(r) >= 4]
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys)


def circles(lon, lat, d: float, kind: str = "geosphere"):
    """One (multi)polygon per point: the circle of radius ``d`` metres around it."""
    lon = np.ascontiguousarray(lon, dtype=float)
    lat = np.ascontiguousarray(lat, dtype=float)
    return [_polygon(r) for r in _core.sdm_circles(lon, lat, float(d), kind)]


def planar_circle(cx: float, cy: float, r: float):
    """GEOS's buffer of a planar point (quadrant segments 10), as terra::buffer without a CRS."""
    return _polygon(_core.sdm_circles(np.array([cx]), np.array([cy]), float(r), "planar")[0])


def circle_extent(lon, lat, d: float):
    """Bounding box of terra's 40-vertex circles around every point (CoordinateCleaner's crop).

    Only the points that can reach an edge of the box are drawn: the circle
    of radius d spans at most ``dlat`` degrees of latitude and ``dlon`` of
    longitude, so the extremes come from points within those margins of the
    extreme coordinates (and from circles crossing the antimeridian or
    holding a pole, which widen the box to the globe).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size == 0:
        return None
    dlat = d / 110_574.0 * 1.01 + 1e-9  # metres per degree of latitude (minimum, at the equator)
    top = min(90.0, np.abs(lat).max() + dlat)
    dlon = 360.0 if top >= 89.0 else d / (111_319.0 * np.cos(np.radians(top))) * 1.01 + 1e-9
    near = ((lon <= lon.min() + 2 * dlon) | (lon >= lon.max() - 2 * dlon) | (lat <= lat.min() + 2 * dlat)
            | (lat >= lat.max() - 2 * dlat) | (np.abs(lon) >= 180.0 - dlon) | (np.abs(lat) >= 90.0 - dlat))
    idx = np.flatnonzero(near)
    rings = _core.sdm_circles(np.ascontiguousarray(lon[idx]), np.ascontiguousarray(lat[idx]), float(d), "terra")
    coords = [r for parts in rings for r in parts]
    xy = np.concatenate(coords) if coords else np.empty((0, 2))
    return float(xy[:, 0].min()), float(xy[:, 0].max()), float(xy[:, 1].min()), float(xy[:, 1].max())
