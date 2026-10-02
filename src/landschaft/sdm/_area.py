# SPDX-License-Identifier: GPL-3.0-or-later
"""The area accessible to the species (the "M" of the BAM diagram)."""

from __future__ import annotations

import numpy as np

from . import _points

__all__ = ["accessible_area"]

_METHODS = ("buffer", "convex_hull", "regions")


def _geodesic_circles(lon: np.ndarray, lat: np.ndarray, distance: float, n: int):
    """Polygons of the points at ``distance`` metres (on the WGS 84 ellipsoid) of each point."""
    import shapely
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    az = np.linspace(0.0, 360.0, n, endpoint=False)
    plon, plat, _ = geod.fwd(np.repeat(lon, n), np.repeat(lat, n), np.tile(az, lon.size),
                             np.full(lon.size * n, float(distance)))
    plon = plon.reshape(-1, n)
    plat = plat.reshape(-1, n)
    # rings that cross the antimeridian: unwrap around the centre (the union is clipped to +-180 below)
    plon = lon[:, None] + (plon - lon[:, None] + 180.0) % 360.0 - 180.0
    rings = np.stack([plon, plat], axis=-1)
    return shapely.polygons(np.concatenate([rings, rings[:, :1]], axis=1))


def _wrap(geom):
    """A lon/lat geometry cut at the antimeridian and folded back into [-180, 180]."""
    import shapely
    from shapely.affinity import translate

    world = shapely.box(-180.0, -90.0, 180.0, 90.0)
    parts = [shapely.intersection(geom, world)]
    for shift in (-360.0, 360.0):
        out = shapely.intersection(geom, shapely.box(-180.0 - shift, -90.0, 180.0 - shift, 90.0))
        if not out.is_empty:
            parts.append(translate(out, xoff=shift))
    return shapely.union_all(parts)


def accessible_area(occ, *, method: str = "buffer", distance: float | None = None, regions=None, within=None,
                    resolution: int = 64, x: str | None = None, y: str | None = None, crs=None):
    """The area the species could have reached: where its background is drawn.

    A model learns to tell the conditions where the species was found from
    those available to it. Drawing the background from areas the species
    never had access to (beyond a mountain range, on another continent) makes
    the model learn geography instead of ecology and inflates its scores
    (Barve et al. 2011, *Ecological Modelling* 222: 1810-1819). The
    accessible area bounds the background sampling, the predictor selection
    and the evaluation.

    Parameters
    ----------
    occ
        Occurrence records: GeoDataFrame of points, or a table with
        coordinate columns.
    method : {"buffer", "convex_hull", "regions"}
        ``"buffer"`` (default): the land within ``distance`` metres of a
        record (geodesic circles on the WGS 84 ellipsoid, merged);
        ``"convex_hull"``: the smallest convex polygon around the records,
        widened by ``distance`` metres if given; ``"regions"``: the polygons of
        ``regions`` (ecoregions, biomes, watersheds) that hold at least one
        record, widened by ``distance`` if given.
    distance
        Radius in metres. Required for ``"buffer"``.
    regions
        GeoDataFrame of polygons, for ``method="regions"``.
    within
        Polygons the result is cut to (a country, a continent, the land).
    resolution
        Vertices of each circle (``"buffer"``) or of the widening.
    x, y, crs
        Coordinate columns and CRS of a table (default: found by name;
        longitude / latitude in EPSG:4326).

    Returns
    -------
    geopandas.GeoDataFrame
        One (multi)polygon in longitude / latitude (EPSG:4326), with the
        method and distance as columns.

    Examples
    --------
    >>> import landschaft as ls
    >>> occ = ls.load_table(ls.sdm.example_data("bradypus"))
    >>> m = ls.sdm.accessible_area(occ, distance=300_000)
    >>> m.crs.to_epsg(), round(float(m.to_crs("EPSG:6933").area.iloc[0]) / 1e12, 1)    # million km2
    (4326, 7.5)
    """
    import geopandas as gpd
    import shapely

    if method not in _METHODS:
        raise ValueError(f"method must be one of {', '.join(_METHODS)}, not {method!r}")
    if distance is not None and not distance > 0:
        raise ValueError("distance must be positive (metres)")
    lon, lat = _points.lonlat(occ, x=x, y=y, crs=crs)
    ok = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[ok], lat[ok]
    if lon.size == 0:
        raise ValueError("no record with coordinates")

    if method == "buffer":
        if distance is None:
            raise ValueError("method='buffer' needs a distance (metres)")
        pts = np.unique(np.column_stack([lon, lat]), axis=0)
        area = shapely.union_all(_geodesic_circles(pts[:, 0], pts[:, 1], distance, resolution))
    else:
        if method == "convex_hull":
            area = shapely.convex_hull(shapely.multipoints(np.column_stack([lon, lat])))
            if area.geom_type != "Polygon":
                raise ValueError("the convex hull needs three records not on one line; use method='buffer'")
        else:
            if regions is None:
                raise ValueError("method='regions' needs regions (a GeoDataFrame of polygons)")
            reg = gpd.GeoDataFrame(geometry=gpd.GeoSeries(regions.geometry if hasattr(regions, "geometry")
                                                          else regions)).to_crs("EPSG:4326")
            hit = reg.sindex.query(shapely.points(lon, lat), predicate="within")[1]
            if hit.size == 0:
                raise ValueError("no record falls in any of the regions")
            area = shapely.union_all(reg.geometry.to_numpy()[np.unique(hit)])
        if distance is not None:
            area = _widen(area, distance, resolution)
    area = _wrap(shapely.make_valid(area))
    if within is not None:
        clip = gpd.GeoSeries(within.geometry if hasattr(within, "geometry") else within)
        if clip.crs is not None:
            clip = clip.to_crs("EPSG:4326")
        area = shapely.intersection(area, shapely.union_all(clip.to_numpy()))
    if area.is_empty:
        raise ValueError("the accessible area is empty")
    return gpd.GeoDataFrame({"method": [method], "distance": [distance]}, geometry=[area], crs="EPSG:4326")


def _widen(area, distance: float, resolution: int):
    """``area`` widened by ``distance`` metres: geodesic circles on the vertices of its outline, merged with it."""
    import shapely

    boundary = shapely.segmentize(shapely.boundary(area), max_segment_length=0.25)  # degrees
    xy = np.unique(shapely.get_coordinates(boundary), axis=0)
    circles = _geodesic_circles(xy[:, 0], xy[:, 1], distance, resolution)
    return shapely.union_all([area, *circles])
