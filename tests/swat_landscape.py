# SPDX-License-Identifier: GPL-3.0-or-later
"""A small synthetic landscape for ls.hydro.swat.from_rasters (tests and
tools/make_reference_swat.py, case ``landscape``): a 1.5 x 1.8 km valley with a
tributary, three land uses, two soils, two weather stations, 2000-2001."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

LANDUSE = {1: "FRSD", 2: "PAST", 3: "RNGE"}
SOILS = {10: "LOAM1", 20: "CLAY1"}
OPTIONS = {"stream_threshold": 0.15, "hru_method": "filter", "thresholds": (5.0, 5.0, 5.0), "slope_bands": (5, 20)}


def rasters(ny: int = 60, nx: int = 50, res: float = 30.0):
    import rioxarray  # noqa: F401  (the .rio accessor)

    r, c = np.mgrid[0:ny, 0:nx].astype(float)
    z = 300 - 1.5 * r + 0.8 * np.abs(c - nx / 2) + 0.03 * (c - nx / 2) ** 2
    trib = np.exp(-((c - 0.75 * nx) - 0.5 * (r - 0.3 * ny)) ** 2 / 20.0) * (r < 0.6 * ny)
    z = z - 6 * trib + 0.3 * np.sin(r / 3.0) * np.cos(c / 4.0)
    xs = 600000.0 + (np.arange(nx) + 0.5) * res
    ys = 4800000.0 + ny * res - (np.arange(ny) + 0.5) * res

    def da(v, name):
        a = xr.DataArray(v, dims=("y", "x"), coords={"y": ys, "x": xs}, name=name)
        return a.rio.write_crs("EPSG:32617").rio.write_transform()

    lu = np.where(c < nx * 0.4, 1, np.where(r < ny * 0.5, 2, 3)).astype(float)
    soil = np.where(r + 0.5 * c < 45, 10, 20).astype(float)
    return da(z, "dem"), da(lu, "landuse"), da(soil, "soils")


def soil_table() -> pd.DataFrame:
    """Two soils in SWAT+'s layout (one row per layer)."""
    rows = []
    for name, grp, layers in (
        ("LOAM1", "B", [(150, 1.40, 0.13, 25.0, 1.2, 18, 40, 42), (600, 1.45, 0.12, 15.0, 0.6, 22, 38, 40),
                        (1500, 1.55, 0.10, 6.0, 0.3, 25, 35, 40)]),
        ("CLAY1", "D", [(200, 1.30, 0.15, 1.5, 1.7, 52, 30, 18), (1000, 1.38, 0.12, 0.8, 1.0, 55, 28, 17)]),
    ):
        for dp, bd, awc, k, cbn, clay, silt, sand in layers:
            rows.append({"name": name, "hyd_grp": grp, "dp_tot": float(layers[-1][0]), "anion_excl": 0.5,
                         "perc_crk": 0.5, "texture": "L" if grp == "B" else "C", "dp": float(dp), "bd": bd, "awc": awc,
                         "soil_k": k, "carbon": cbn, "clay": float(clay), "silt": float(silt), "sand": float(sand),
                         "rock": 2.0, "alb": 0.12, "usle_k": 0.3, "ec": 0.0, "caco3": 0.0, "ph": 6.5})
    return pd.DataFrame(rows)


def weather():
    """Two stations' daily rain and temperature, 2000-2001 (generated, rounded)."""
    rng = np.random.default_rng(7)
    days = pd.date_range("2000-01-01", "2001-12-31")
    stations = pd.DataFrame({"lat": [43.352, 43.340], "lon": [-79.761, -79.745], "elev": [250.0, 215.0]},
                            index=["north", "south"])
    season = 10 * np.sin(2 * np.pi * (days.dayofyear.to_numpy() - 110) / 365)
    wet = rng.random((len(days), 2)) < 0.3
    pcp = pd.DataFrame(np.where(wet, rng.gamma(0.8, 9.0, (len(days), 2)), 0.0).round(1), index=days,
                       columns=stations.index)
    tmax = pd.DataFrame((12 + season[:, None] + rng.normal(0, 3, (len(days), 2))).round(1), index=days,
                        columns=stations.index)
    tmin = (tmax - 9 - (rng.random(tmax.shape) * 3).round(1)).round(1)
    monthly = {"pcp_hhr": 12.0, "slr_ave": [5, 8, 12, 16, 19, 21, 22, 19, 14, 9, 5, 4], "dew_ave": 3.0, "wnd_ave": 3.5}
    return stations, pcp, tmax, tmin, monthly
