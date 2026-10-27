# SPDX-License-Identifier: GPL-3.0-or-later
"""Habitat quality and rarity (InVEST's ``habitat_quality`` model)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ..surface._grid import read_surface
from ._common import LandUse, align_inputs, lookup_table, same_grid

__all__ = ["HabitatQuality", "habitat_quality"]

Z = 2.5  # the scaling parameter of the quality (InVEST's _SCALING_PARAM)


@dataclass
class HabitatQuality(Saveable):
    """Habitat degradation, quality and rarity (:func:`habitat_quality`).

    Maps are float32 (as InVEST writes them), NaN where the land use map has no
    value. ``degradation``, ``quality``, ``threats`` and ``rarity`` are dicts by
    scenario: ``"current"`` and, when given, ``"future"`` and ``"base"``.

    Attributes
    ----------
    degradation : dict
        Total degradation of each scenario: the sum over the threats.
    quality : dict
        Habitat quality: ``h (1 - D^z / (D^z + k^z))``, z = 2.5.
    threats : dict
        Each scenario's degradation by threat (a dict by threat name).
    rarity : dict
        Rarity of the current and future map against the base map: by cell
        (``rarity[scenario]``), and by class in ``rarity_table``.
    rarity_table : pandas.DataFrame
        ``scenario``, ``lucode``, ``rarity``: 1 - n_x / (n_base + n_x), the cell
        counts of the class in the scenario and the base map (0 for a class
        the base map lacks).
    """

    degradation: dict
    quality: dict
    threats: dict
    rarity: dict = field(default_factory=dict)
    rarity_table: pd.DataFrame | None = None

    def _save_items(self):
        items = {}
        for sc, da in self.degradation.items():
            items[f"deg_sum_{sc}"] = da
        for sc, da in self.quality.items():
            items[f"quality_{sc}"] = da
        for sc, d in self.threats.items():
            for t, da in d.items():
                items[f"degradation_{t}_{sc}"] = da
        for sc, da in self.rarity.items():
            items[f"rarity_{sc}"] = da
        if self.rarity_table is not None:
            items["rarity"] = self.rarity_table
        return items

    def __repr__(self) -> str:
        means = ", ".join(f"{sc}: {float(np.nanmean(q.values)):.4f}" for sc, q in self.quality.items())
        return f"HabitatQuality(mean quality {means})"


_SCENARIOS = ("current", "future", "base")


def _threat_maps(maps, names, shape, scenario: str):
    """The threat maps of a scenario by name: a dict, or a 3-D DataArray with a band per threat."""
    if isinstance(maps, xr.DataArray) and maps.ndim == 3:
        dim = maps.dims[0]
        labels = [str(v) for v in maps[dim].values]
        maps = {t: maps[labels.index(t)] for t in names if t in labels}
    if not isinstance(maps, dict):
        raise TypeError(f"the {scenario} threat maps are a dict {{threat: raster}} or a 3-D DataArray")
    out = {}
    for t in names:
        if t not in maps:
            raise ValueError(f"no {scenario} map of the threat {t!r}")
        s = read_surface(maps[t])
        if s.shape != shape:
            raise ValueError(f"the {scenario} map of {t!r} has {s.shape} cells, the land use map {shape}")
        v = s.values
        ok = ~np.isnan(v)
        if np.any((v[ok] < 0) | (v[ok] > 1)):
            raise ValueError(f"the {scenario} map of {t!r} must hold values from 0 to 1")
        out[t] = np.ascontiguousarray((ok & (v != 0)).astype(np.uint8))
    return out


def _by_threat(maps):
    """Threat maps as a dict ``{threat: raster}`` (a 3-D DataArray: a band per threat)."""
    if isinstance(maps, xr.DataArray) and maps.ndim == 3:
        dim = maps.dims[0]
        return {str(v): maps[i] for i, v in enumerate(maps[dim].values)}
    return maps


def _aligned(lulc, future, base, scenario_maps: dict, access, threads):
    """Every raster on the grid of the current land use map, as InVEST aligns them.

    InVEST's habitat_quality: the land use maps and the threat maps resampled to
    the nearest cell, with the smaller side of the current map's cells both ways,
    over its bounds and in its CRS. An access raster (InVEST takes polygons) too.
    """
    rasters = {"lulc": (lulc, "near"), "future": (future, "near"), "base": (base, "near")}
    for sc, maps in scenario_maps.items():
        if isinstance(maps, dict):
            for t, m in maps.items():
                rasters[f"{sc}/{t}"] = (m, "near")
    acc_raster = access is not None and not hasattr(access, "geometry")
    if acc_raster:
        rasters["access"] = (access, "near")
    out = align_inputs(rasters, reference="lulc", pixel="square", bbox="reference", n_threads=threads)
    maps = {}
    for sc, m in scenario_maps.items():
        maps[sc] = {t: out[f"{sc}/{t}"] for t in m} if isinstance(m, dict) else m
    return (out["lulc"], out.get("future"), out.get("base"), maps,
            out["access"] if acc_raster else access)


def _access(access, lu: LandUse):
    """The access raster (float32, 1 where not given), from a raster or polygons with an ``access`` column."""
    if access is None:
        return None
    try:
        import geopandas as gpd
    except ImportError:  # pragma: no cover
        gpd = None
    if gpd is not None and isinstance(access, gpd.GeoDataFrame):
        from .._vector import rasterize

        cols = {c.lower(): c for c in access.columns}
        if "access" not in cols:
            raise ValueError("the access polygons need an 'access' column (0: inaccessible ... 1: accessible)")
        like = lu.raster(np.zeros(lu.shape, np.float32), "access")
        if access.crs is not None and like.rio.crs is not None and access.crs != like.rio.crs:
            access = access.to_crs(like.rio.crs)
        burned = rasterize(access, like=like, field=cols["access"], fill=np.nan)
        a = np.asarray(burned.values, np.float64)
        a = np.where(np.isnan(a), 1.0, a)
    else:
        s = read_surface(access)
        if s.shape != lu.shape:
            raise ValueError(f"the access raster has {s.shape} cells, the land use map {lu.shape}")
        a = s.values
    return np.ascontiguousarray(a.astype(np.float32))


def habitat_quality(lulc, threats, sensitivity, threat_maps, *, half_saturation: float, future=None,
                    future_threat_maps=None, base=None, base_threat_maps=None, access=None, nodata=None,
                    n_threads: int | None = None) -> HabitatQuality:
    """Habitat degradation, quality and rarity (InVEST's habitat quality model).

    Each threat (crops, roads, towns...) degrades the habitat around it: its
    impact decays linearly or exponentially with the distance to the nearest
    cell of the threat, up to its maximum distance (exponentially: to 5 % there),
    and weighs by its relative weight, by the sensitivity of the cell's land use
    class to it and by the cell's access (legal protection). The degradation
    D of a cell is the sum over the threats; its quality is
    ``h (1 - D^z / (D^z + k^z))``, h the habitat suitability of its class, k the
    half-saturation constant and z = 2.5. With a base map, the rarity of each
    class of the current (and future) map: ``1 - n_x / (n_base + n_x)``, the
    cell counts in the two maps.

    The maps are InVEST's (natcap.invest 3.20, which takes the distance to the
    nearest threat cell since 3.12): the same distance transform in float32 and
    the same float32 expressions, bit for bit (the exponential decay through
    numpy's float32 ``exp``, as InVEST computes it).

    Parameters
    ----------
    lulc
        The current land use map (integer codes; projected, in metres).
    threats
        A DataFrame with a row per threat: ``threat`` (its name), ``max_dist``
        (metres), ``weight`` (relative) and ``decay`` (``"linear"`` or
        ``"exponential"``).
    sensitivity
        A DataFrame by land use class (``lucode``): ``habitat`` (0 to 1) and a
        column per threat with the sensitivity of the class to it (0 to 1).
    threat_maps
        The current map of each threat, ``{threat: raster}`` (or a 3-D DataArray
        with a band per threat): cells other than 0 and nodata are the threat
        (values from 0 to 1). Rasters on other grids (cells, extent, CRS) are put
        on the grid of ``lulc`` first, as InVEST aligns them: the nearest cell,
        square cells of the smaller side of ``lulc``'s, over its bounds; so are
        the future and base maps and an access raster.
    half_saturation
        The half-saturation constant k (often half the highest degradation).
    future, future_threat_maps
        A future land use map and its threat maps.
    base, base_threat_maps
        A base (reference) land use map, for the rarity, and its threat maps
        (optional: without them the base map has no degradation nor quality).
    access
        Accessibility to threats, from 0 (inaccessible) to 1: a raster, or
        polygons (GeoDataFrame) with an ``access`` column (1 outside them).
    nodata
        Value marking missing cells in the land use maps.

    Returns
    -------
    HabitatQuality

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from affine import Affine
    >>> import landschaft as ls
    >>> lulc = np.ones((40, 40), int)
    >>> lulc[:, 20:] = 2                                       # forest (1), pasture (2)
    >>> road = np.zeros((40, 40))
    >>> road[:, 30] = 1
    >>> lulc = ls.load_raster(ls.write_raster(lulc, "hq_lulc.tif", crs="EPSG:31983",
    ...                                       transform=Affine(30, 0, 500000, 0, -30, 8000000)))
    >>> threats = pd.DataFrame({"threat": ["road"], "max_dist": [600.0], "weight": [1.0],
    ...                         "decay": ["linear"]})
    >>> sens = pd.DataFrame({"lucode": [1, 2], "habitat": [1.0, 0.0], "road": [0.8, 0.0]})
    >>> hq = ls.services.habitat_quality(lulc, threats, sens, {"road": lulc.copy(data=road)},
    ...                                  half_saturation=0.1)
    >>> q = hq.quality["current"].values
    >>> q[0, 0] > q[0, 19], q[0, 25]                           # less quality near the road
    (np.True_, np.float32(0.0))
    """
    threads = resolve(n_threads)
    scenario_maps = {"current": _by_threat(threat_maps), "future": _by_threat(future_threat_maps),
                     "base": _by_threat(base_threat_maps)}
    lulc, future, base, scenario_maps, access = _aligned(lulc, future, base, scenario_maps, access, threads)
    threat_maps, future_threat_maps, base_threat_maps = (scenario_maps["current"], scenario_maps["future"],
                                                         scenario_maps["base"])
    lu = LandUse(lulc, nodata, "the land use map")
    if not isinstance(threats, pd.DataFrame):
        raise TypeError("threats is a DataFrame with threat, max_dist, weight and decay")
    tt = threats.copy()
    tt.columns = [str(c).strip().lower() for c in tt.columns]
    for c in ("threat", "max_dist", "weight", "decay"):
        if c not in tt.columns:
            raise ValueError(f"the threats table lacks the column {c!r}")
    names = [str(t) for t in tt["threat"]]
    if len(set(names)) != len(names):
        raise ValueError("the threats table lists a threat more than once")
    decays = [str(d).strip().lower() for d in tt["decay"]]
    if any(d not in ("linear", "exponential") for d in decays):
        raise ValueError("decay is 'linear' or 'exponential'")
    max_dist = tt["max_dist"].astype(np.float64).to_numpy()
    weights = tt["weight"].astype(np.float64).to_numpy()
    if np.any(max_dist <= 0):
        raise ValueError("max_dist must be positive")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("the weights are ratios from 0 to 1")
    weight_sum = weights.sum()  # numpy's (pairwise) sum, as pandas gives InVEST
    if not weight_sum > 0:
        raise ValueError("the weights sum to zero")
    if not half_saturation > 0:
        raise ValueError("half_saturation must be positive")
    codes, table = lookup_table(sensitivity, ["habitat", *names], "the sensitivity table")
    habitat = np.ascontiguousarray(table["habitat"].to_numpy().astype(np.float32))
    sens = {t: np.ascontiguousarray(table[t.lower()].to_numpy().astype(np.float32)) for t in names}
    px = lu.pixel()[0]
    ksq = float(half_saturation) ** Z
    acc = _access(access, lu)

    lands = {"current": (lu, threat_maps)}
    if future is not None:
        if future_threat_maps is None:
            raise ValueError("give the threat maps of the future (future_threat_maps)")
        fu = LandUse(future, nodata, "the future land use map")
        same_grid(lu, fu)
        lands["future"] = (fu, future_threat_maps)
    bu = None
    if base is not None:
        bu = LandUse(base, nodata, "the base land use map")
        same_grid(lu, bu)
        lands["base"] = (bu, base_threat_maps)
    elif base_threat_maps is not None:
        raise ValueError("base_threat_maps without the base map")

    out = HabitatQuality({}, {}, {})
    for sc, (land, maps) in lands.items():
        if maps is None:  # the base map without threats: only its rarity
            continue
        cls = land.classes(codes, threads)
        regions = _threat_maps(maps, names, land.shape, sc)
        total = np.zeros(land.shape, np.float32)
        per = {}
        for k, t in enumerate(names):
            dist = _core.serv_distance_transform(regions[t], threads)
            # weight / sum as numpy float64 scalars (pandas'), the float64 product InVEST makes
            w = float(np.float64(weights[k]) / weight_sum)
            max_px = float(max_dist[k] / px)
            given = None
            if decays[k] == "exponential":  # InVEST's numpy expression: numpy's float32 exp, to its bits
                given = np.where(dist > max_px, 0, np.exp((-dist * 2.99) / max_px)).astype(np.float32)
            deg = _core.serv_threat_degradation(dist, given, cls, sens[t], acc, max_px,
                                                decays[k] == "exponential", w, total, threads)
            per[t] = land.raster(deg, f"degradation_{t}_{sc}")
        out.threats[sc] = per
        out.degradation[sc] = land.raster(total, f"deg_sum_{sc}")
        hab = _core.serv_quality(total, cls, habitat, ksq, threads)
        out.quality[sc] = land.raster(hab, f"quality_{sc}")
    if bu is not None:
        rows = []
        base_counts = _counts(bu.codes[bu.valid])
        for sc in ("current", "future"):
            if sc not in lands:
                continue
            land = lands[sc][0]
            keep = land.valid & bu.valid  # the scenario trimmed to the base map's cells
            counts = _counts(land.codes[keep])
            index = {}
            area = px * lu.pixel()[1]  # InVEST's abs(dx) * abs(dy), the same for both maps once aligned
            for code, n in counts.items():
                if code in base_counts:
                    num = n * area
                    den = base_counts[code] * area
                    index[code] = 1.0 - (num / (den + num))
                else:
                    index[code] = 0.0
            r = np.full(land.shape, np.nan, np.float32)
            for code, v in index.items():
                r[keep & (land.codes == code)] = np.float32(v)
            out.rarity[sc] = land.raster(r, f"rarity_{sc}")
            rows += [(sc, code, index[code]) for code in sorted(index)]
        out.rarity_table = pd.DataFrame(rows, columns=["scenario", "lucode", "rarity"])
    return out


def _counts(values) -> dict:
    u, n = np.unique(values, return_counts=True)
    return {int(a): int(b) for a, b in zip(u, n)}
