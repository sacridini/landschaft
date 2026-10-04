# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the example data of ls.prioritize (src/landschaft/prioritize/data): a hypothetical region of
40 x 60 cells of 1 km (UTM 23S), with a land cost, six species, protected areas and two threats.

    python tools/make_prioritize_examples.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

import landschaft as ls

OUT = Path(__file__).resolve().parents[1] / "src" / "landschaft" / "prioritize" / "data"
NY, NX, RES = 40, 60, 1000.0
SPECIES = ["forest_bird", "wetland_plant", "grassland_butterfly", "large_mammal", "amphibian", "endemic_snail"]


def field(rng, width=4):
    v = rng.random((NY + 2 * width, NX + 2 * width))
    k = np.ones(2 * width + 1) / (2 * width + 1)
    v = np.apply_along_axis(np.convolve, 1, v, k, "valid")
    v = np.apply_along_axis(np.convolve, 0, v, k, "valid")
    return (v - v.min()) / (v.max() - v.min())


def grid(values, names=None):
    coords = {"y": 7_500_000.0 + (np.arange(NY)[::-1] + 0.5) * RES, "x": 300_000.0 + (np.arange(NX) + 0.5) * RES}
    v = np.asarray(values, dtype=np.float32)
    if v.ndim == 3:
        da = xr.DataArray(v, dims=("band", "y", "x"), coords={"band": names, **coords})
    else:
        da = xr.DataArray(v, dims=("y", "x"), coords=coords)
    return da.rio.write_crs("EPSG:32723")


def main() -> None:
    rng = np.random.default_rng(2026)
    region = np.ones((NY, NX), dtype=bool)
    region[:8, :10] = False                      # outside the region (the sea)
    yy, xx = np.mgrid[:NY, :NX]
    region[(yy - 25) ** 2 + (xx - 40) ** 2 < 10] = False  # a reservoir
    cost = np.where(region, np.round(5 + 95 * field(rng) ** 1.5, 1), np.nan)
    species = []
    for k, share in enumerate((0.35, 0.12, 0.25, 0.5, 0.2, 0.06)):
        f = field(rng, 3 + k % 3)
        cut = np.quantile(f, 1 - share)
        species.append(np.where(region, np.clip((f - cut) / (1 - cut), 0, 1), np.nan).round(3))
    protected = np.zeros((NY, NX))
    protected[30:36, 5:12] = 1.0
    protected[12:16, 48:55] = 1.0
    protected = np.where(region, protected, np.nan)
    threats = []
    for k, share in enumerate((0.3, 0.4)):
        f = field(rng, 5)
        cut = np.quantile(f, 1 - share)
        threats.append(np.where(region, np.clip((f - cut) / (1 - cut) * 1.2, 0, 1), np.nan).round(2))
    OUT.mkdir(parents=True, exist_ok=True)
    ls.write_raster(grid(cost), OUT / "cost.tif")
    ls.write_raster(grid(np.stack(species), SPECIES), OUT / "features.tif")
    ls.write_raster(grid(protected), OUT / "protected.tif")
    ls.write_raster(grid(np.stack(threats), ["fire", "grazing"]), OUT / "threats.tif")
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
