# SPDX-License-Identifier: GPL-3.0-or-later
"""Circles of a radius in metres around points, as CoordinateCleaner draws them.

Two polygonisations, as in R (CoordinateCleaner's ``geod`` switch): 73
vertices at bearings 0, 5, ..., 360 (``geosphere::destPoint``) or terra's
point buffer, 40 vertices at bearings 0, 9, ..., 351. Both on the WGS 84
ellipsoid; circles crossing the antimeridian are split, those holding a pole
closed through it (CoordinateCleaner's geosphere circles are not: a bug, fixed
here).
"""

from __future__ import annotations

import numpy as np

__all__ = ["circles", "circle_extent"]

_GEOD = None


def _geod():
    global _GEOD
    if _GEOD is None:
        from pyproj import Geod

        _GEOD = Geod(ellps="WGS84")
    return _GEOD


def _vertices(lon: np.ndarray, lat: np.ndarray, d: float, kind: str):
    """(n, m) longitudes and latitudes of the vertices of each circle (not closed)."""
    if kind == "geosphere":
        bearings = np.arange(0.0, 360.0, 5.0)
    else:
        bearings = np.arange(40) * (360.0 / 40)
    m = bearings.size
    plon, plat, _ = _geod().fwd(np.repeat(lon, m), np.repeat(lat, m), np.tile(bearings, lon.size),
                                np.full(lon.size * m, float(d)))
    return plon.reshape(-1, m), plat.reshape(-1, m)


def _contains_pole(lon: float, lat: float, d: float) -> int:
    """+1 / -1 when the circle holds the north / south pole, else 0."""
    g = _geod()
    if g.inv(lon, lat, lon, 90.0)[2] < d:
        return 1
    if g.inv(lon, lat, lon, -90.0)[2] < d:
        return -1
    return 0


def circles(lon, lat, d: float, kind: str = "geosphere"):
    """One (multi)polygon per point: the circle of radius ``d`` metres around it."""
    import shapely

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    vx, vy = _vertices(lon, lat, d, kind)
    out = []
    for k in range(lon.size):
        x, y = vx[k], vy[k]
        pole = _contains_pole(lon[k], lat[k], d)
        if pole:
            order = np.argsort(x, kind="stable")
            x, y = x[order], y[order]
            ring = np.concatenate([[[-180.0, y[0]]], np.column_stack([x, y]),
                                   [[180.0, y[-1]], [180.0, 90.0 * pole], [-180.0, 90.0 * pole]]])
            out.append(shapely.Polygon(ring))
        elif np.ptp(x) > 180:
            dx = x - lon[k]
            x = np.where(dx > 180, x - 360, np.where(dx < -180, x + 360, x))
            shift = -360.0 if lon[k] > 0 else 360.0
            ring = np.column_stack([x, y])
            out.append(shapely.MultiPolygon([shapely.Polygon(ring), shapely.Polygon(ring + [shift, 0.0])]))
        else:
            out.append(shapely.Polygon(np.column_stack([x, y])))
    return out


def circle_extent(lon, lat, d: float):
    """Bounding box of terra's 40-vertex circles around every point (CoordinateCleaner's crop)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size == 0:
        return None
    vx, vy = _vertices(lon, lat, d, "terra")
    xmin, xmax, ymin, ymax = vx.min(), vx.max(), vy.min(), vy.max()
    if np.any(np.ptp(vx, axis=1) > 180):
        xmin, xmax = -180.0, 180.0
    for k in range(lon.size):
        pole = _contains_pole(lon[k], lat[k], d)
        if pole:
            xmin, xmax = -180.0, 180.0
            ymin, ymax = (ymin, 90.0) if pole > 0 else (-90.0, ymax)
    return xmin, xmax, ymin, ymax
