# SPDX-License-Identifier: GPL-3.0-or-later
"""Write the example data of ls.prioritize (src/landschaft/prioritize/data): "Serra Verde", an imaginary
coastal region of 60 x 90 cells of 1 km (SIRGAS 2000 / UTM 23S) with mountains in the north-west, a
coastal plain with three cities, two rivers, roads, eight conservation features, protected areas,
habitat condition, regions, carbon and two threats.

The landscape is prioripy's tutorial data (docs/assets/tutorial_data.py of prioripy 0.11.11, by the
same author, MIT), generated with the same seed: the same maps.

    python tools/make_prioritize_examples.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

import landschaft as ls

OUT = Path(__file__).resolve().parents[1] / "src" / "landschaft" / "prioritize" / "data"
ROWS, COLS, CELL = 60, 90, 1000.0
CRS = "EPSG:31983"
ORIGIN_X, ORIGIN_Y = 600000.0, 7500000.0
SPECIES = ["jaguar", "lowland_tapir", "harpy_eagle", "maned_wolf", "golden_lion_tamarin", "neotropical_otter",
           "araucaria_forest", "mangrove"]


def _smooth_noise(rng, sigma, shape=(ROWS, COLS)):
    field = ndimage.gaussian_filter(rng.normal(size=shape), sigma=sigma, mode="reflect")
    field -= field.min()
    return field / field.max()


def _blob(cy, cx, sy, sx):
    y, x = np.mgrid[0:ROWS, 0:COLS]
    return np.exp(-(((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2) / 2.0)


def build_landscape(seed: int = 7) -> dict:
    """Every layer as arrays (prioripy's build_landscape, unchanged)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ROWS, 0:COLS]
    shore = 0.55 * (x / COLS) + 0.45 * (y / ROWS) + 0.22 * (_smooth_noise(rng, 5) - 0.5)
    land = shore < 0.78
    elevation = np.clip((0.78 - shore) / 0.78, 0, 1)
    elevation = 0.6 * elevation + 0.4 * _smooth_noise(rng, 6)
    elevation = (elevation - elevation[land].min()) / (elevation[land].max() - elevation[land].min())
    elevation = np.clip(elevation, 0, 1)
    city_intensity = _blob(43, 52, 3.0, 4.0) + 0.8 * _blob(28, 70, 2.5, 3.0) + 0.6 * _blob(52, 30, 2.5, 3.0)
    urban = (city_intensity > 0.45) & land
    river = np.zeros((ROWS, COLS), dtype=bool)
    for c0, slope, amp, period, phase in [(14, 0.55, 3.5, 6.0, 0.0), (50, 0.45, 2.5, 5.0, 1.3)]:
        for r in np.linspace(0, ROWS - 1, ROWS * 6):
            c = c0 + slope * r + amp * np.sin(r / period + phase) + 1.5 * np.sin(r / 2.3)
            ri, ci = int(round(r)), int(round(c))  # noqa: RUF046 (numpy floats)
            if 0 <= ci < COLS:
                if not land[ri, ci]:
                    break
                river[ri, ci] = True
    river_dist = ndimage.distance_transform_edt(~river)
    coast_dist = ndimage.distance_transform_edt(land)
    roads = np.zeros((ROWS, COLS), dtype=bool)
    roads[np.abs(coast_dist - 6) < 0.6] = True
    roads[(np.abs(x - (52 - 0.4 * (43 - y))) < 0.6) & (y < 43)] = True
    roads &= land & ~urban
    road_dist = ndimage.distance_transform_edt(~(roads | urban))
    cost = 15 + 55 * (1 - elevation) + 70 * city_intensity + 30 * np.exp(-road_dist / 3) + 30 * _smooth_noise(rng, 3)
    cost = np.round(cost, 1).astype(np.float32)
    cost[urban] = np.round(cost[urban] * 1.5, 1)
    cost[~land] = np.nan
    rng2 = np.random.default_rng(seed + 100)
    farmland = np.clip(1.5 * (1 - elevation) * _smooth_noise(rng2, 3) - 0.35, 0, 1)
    degradation = np.clip(1.3 * city_intensity + 0.7 * np.exp(-road_dist / 4) + 0.25 * _smooth_noise(rng, 3)
                          + 1.1 * farmland - 0.1, 0, 1)
    condition = np.round(np.clip(1 - degradation, 0.05, 1), 3).astype(np.float32)

    def habitat(suit, threshold):
        suit = np.clip(suit, 0, None)
        suit = suit / suit[land].max()
        return np.round(np.where((suit > threshold) & land & ~urban, suit, 0.0), 3).astype(np.float32)

    wide = _smooth_noise(rng, 5)
    species = [
        habitat(0.6 * elevation + 0.4 * wide - 0.5 * city_intensity, 0.45),
        habitat(np.exp(-((elevation - 0.45) / 0.2) ** 2) + 0.6 * np.exp(-river_dist / 4)
                + 0.2 * _smooth_noise(rng, 3), 0.55),
        habitat(elevation ** 1.5 * (1 - np.exp(-road_dist / 5)) + 0.25 * _smooth_noise(rng, 4), 0.6),
        habitat((x < 20 + 14 * _smooth_noise(rng, 6)) * (0.4 + 0.6 * _smooth_noise(rng, 3)), 0.45),
        habitat(_blob(35, 45, 3.5, 4.5) + 0.8 * _blob(47, 62, 2.5, 3) + 0.15 * _smooth_noise(rng, 2), 0.5),
        habitat(np.exp(-river_dist / 1.5), 0.45),
        habitat(np.clip(elevation - 0.62, 0, None) * (0.6 + 0.4 * _smooth_noise(rng, 2)), 0.25),
        habitat(np.exp(-coast_dist / 1.3) * (0.4 + 0.6 * _smooth_noise(rng, 3)), 0.35),
    ]
    protected = ((_blob(10, 18, 4, 6) > 0.5) | (_blob(14, 60, 3, 4) > 0.5) | (_blob(40, 66, 2, 2) > 0.5)) & land & ~urban
    regions = np.where(x < 30 + 4 * np.sin(y / 9), 1, np.where(x < 58 + 3 * np.cos(y / 7), 2, 3))
    deforestation = np.clip(np.exp(-road_dist / 6) * (0.5 + 0.5 * _smooth_noise(rng, 3)), 0, 1)
    deforestation = np.where((deforestation > 0.3) & land & ~urban, deforestation, 0)
    fire = np.clip(np.exp(-((x - 20) / 18) ** 2) * (0.4 + 0.6 * _smooth_noise(rng, 4)), 0, 1)
    fire = np.where((fire > 0.45) & land & ~urban, fire, 0)
    humidity = _smooth_noise(rng2, 5)
    carbon = np.round(30 + 190 * condition * (0.15 + 0.85 * humidity), 1)
    terminals = ((_blob(10, 18, 4, 6) > 0.5) | (_blob(14, 60, 3, 4) > 0.5)) & land
    return {"land": land, "cost": cost, "species": species, "protected": protected, "urban": urban,
            "condition": condition, "regions": regions, "deforestation": deforestation, "fire": fire,
            "carbon": carbon, "terminals": terminals, "elevation": elevation, "river": river, "roads": roads}


def _grid(values, land, names=None):
    v = np.asarray(values, dtype=np.float32)
    v = np.where(land, v, np.nan)
    coords = {"y": ORIGIN_Y - (np.arange(ROWS) + 0.5) * CELL, "x": ORIGIN_X + (np.arange(COLS) + 0.5) * CELL}
    if v.ndim == 3:
        da = xr.DataArray(v, dims=("band", "y", "x"), coords={"band": names, **coords})
    else:
        da = xr.DataArray(v, dims=("y", "x"), coords=coords)
    return da.rio.write_crs(CRS)


def main() -> None:
    L = build_landscape()
    land = L["land"]
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.tif"):
        old.unlink()
    write = {
        "cost": L["cost"], "protected": L["protected"], "urban": L["urban"], "condition": L["condition"],
        "regions": L["regions"], "carbon": L["carbon"], "terminals": L["terminals"],
        "elevation": np.round(L["elevation"], 3),
    }
    for name, arr in write.items():
        ls.write_raster(_grid(arr, land), OUT / f"{name}.tif")
    ls.write_raster(_grid(np.stack(L["species"]), land, SPECIES), OUT / "features.tif")
    ls.write_raster(_grid(np.stack([np.round(L["deforestation"], 3), np.round(L["fire"], 3)]), land,
                          ["deforestation", "fire"]), OUT / "threats.tif")
    # rivers and roads as rasters (1 = present) for the maps of the documentation
    ls.write_raster(_grid(np.stack([L["river"], L["roads"]]).astype(np.float32), land, ["rivers", "roads"]),
                    OUT / "lines.tif")
    print(f"written to {OUT}: {sorted(p.name for p in OUT.glob('*.tif'))}")


if __name__ == "__main__":
    main()
