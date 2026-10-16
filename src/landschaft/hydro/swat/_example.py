# SPDX-License-Identifier: GPL-3.0-or-later
"""The inputs of a small watershed model from the package's example data."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["example_landscape"]


def _soils() -> pd.DataFrame:
    """Two soils in SWAT+'s layout (one row per layer): a loam of hydrologic group B on
    the uplands and a clay loam of group C in the lower valleys."""
    rows = []
    for name, grp, texture, layers in (
        ("UPLAND_LOAM", "B", "L", [(200, 1.35, 0.15, 20.0, 2.5, 18, 42, 40, 0.28),
                                   (700, 1.45, 0.13, 12.0, 1.0, 22, 40, 38, 0.28),
                                   (1500, 1.55, 0.11, 6.0, 0.4, 25, 38, 37, 0.28)]),
        ("VALLEY_CLAY_LOAM", "C", "CL", [(250, 1.30, 0.17, 6.0, 2.0, 32, 36, 32, 0.30),
                                         (900, 1.40, 0.15, 3.0, 0.8, 36, 34, 30, 0.30),
                                         (1500, 1.50, 0.13, 1.5, 0.3, 38, 34, 28, 0.30)]),
    ):
        for dp, bd, awc, k, cbn, clay, silt, sand, usle_k in layers:
            rows.append({"name": name, "hyd_grp": grp, "dp_tot": float(layers[-1][0]), "anion_excl": 0.5,
                         "perc_crk": 0.5, "texture": texture, "dp": float(dp), "bd": bd, "awc": awc, "soil_k": k,
                         "carbon": cbn, "clay": float(clay), "silt": float(silt), "sand": float(sand), "rock": 5.0,
                         "alb": 0.12, "usle_k": usle_k, "ec": 0.0, "caco3": 0.0, "ph": 6.0})
    return pd.DataFrame(rows)


def example_landscape() -> dict:
    """The inputs of a small watershed model, from the package's example rasters.

    The terrain is the example DEM ``"orelevation"`` (11 km x 8 km of the Oregon
    Cascades, 433-1390 m) projected to UTM zone 10 at 30 m; the land use comes from
    the NDVI ``"orforest"`` (forest, ``FRSD``, where it exceeds 0.55, else pasture,
    ``PAST``); the soils are two made-up soils by elevation (a group B loam above 900
    m, a group C clay loam below); the weather is made up too: one station at the
    map's centre, daily rain and temperature for 2001-2004 from a seeded generator
    (wet winters, dry summers, about 1100 mm a year). Ready for
    :func:`watershed` / :func:`from_rasters` and :func:`climate`.

    Returns
    -------
    dict
        ``dem``, ``landuse``, ``soils`` (DataArrays on one grid), ``landuse_lookup``,
        ``soil_lookup``, ``soil_table`` (the arguments of :func:`watershed`),
        ``stations``, ``pcp``, ``tmax``, ``tmin`` and ``monthly`` (the arguments of
        :func:`climate`).

    Examples
    --------
    >>> import landschaft as ls
    >>> ex = ls.hydro.swat.example_landscape()
    >>> ex["dem"].shape, ex["dem"].rio.resolution()
    ((371, 273), (30.0, -30.0))
    >>> sorted(ex["landuse_lookup"].values()), ex["pcp"].shape
    (['FRSD', 'PAST'], (1461, 1))
    """
    from ... import load_raster, reproject
    from ...datasets import example_data

    geo = load_raster(example_data("orelevation"))
    dem = reproject(geo, "EPSG:32610", res=30.0)
    ndvi = reproject(load_raster(example_data("orforest")), like=dem)
    landuse = ((ndvi > 0.55) * 1.0 + (ndvi <= 0.55) * 2.0).where(np.isfinite(ndvi)).rename("landuse")
    soils = ((dem > 900) * 10.0 + (dem <= 900) * 20.0).where(np.isfinite(dem)).rename("soils")

    rng = np.random.default_rng(2001)
    days = pd.date_range("2001-01-01", "2004-12-31")
    season = np.sin(2 * np.pi * (days.dayofyear.to_numpy() - 110) / 365)  # 1 in late July
    wet = rng.random(len(days)) < 0.40 - 0.28 * season
    pcp = pd.DataFrame({"valley": np.where(wet, rng.gamma(0.9, 9.0, len(days)), 0.0).round(1)}, index=days)
    tmax = pd.DataFrame({"valley": (16 + 10 * season + rng.normal(0, 3, len(days))).round(1)}, index=days)
    tmin = (tmax - 9 - rng.random(tmax.shape).round(1) * 3).round(1)
    stations = pd.DataFrame({"lat": [float(geo.y.mean())], "lon": [float(geo.x.mean())], "elev": [float(geo.mean())]},
                            index=["valley"])
    monthly = {"pcp_hhr": 8.0, "slr_ave": [4, 7, 11, 16, 20, 23, 25, 21, 15, 9, 5, 3], "dew_ave": 4.0,
               "wnd_ave": 2.5}
    return {"dem": dem.rename("dem"), "landuse": landuse, "soils": soils,
            "landuse_lookup": {1: "FRSD", 2: "PAST"}, "soil_lookup": {10: "UPLAND_LOAM", 20: "VALLEY_CLAY_LOAM"},
            "soil_table": _soils(), "stations": stations, "pcp": pcp, "tmax": tmax, "tmin": tmin, "monthly": monthly}
