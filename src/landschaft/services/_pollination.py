# SPDX-License-Identifier: GPL-3.0-or-later
"""Crop pollination by wild bees (InVEST's ``pollination`` model)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ._common import LandUse, lookup_table

__all__ = ["Pollination", "pollination"]

_SUBSTRATE = re.compile(r"nesting_suitability_([^_]+)_index$")
_SEASON = re.compile(r"foraging_activity_([^_]+)_index$")


@dataclass
class Pollination(Saveable):
    """Pollinator supply and abundance, and the yield of farms (:func:`pollination`).

    Maps are float32 as InVEST writes them (``floral_resources`` and
    ``convolved_supply``: float64), NaN where the land use map has no value.
    Dicts are by species (the guild table's ``species``) and season.

    Attributes
    ----------
    supply : dict
        Pollinator supply of each species: ``sa HN FR``, its relative abundance
        times its habitat nesting index times the floral resources within its
        flight range. Where the species can nest and find flowers.
    abundance : dict
        Pollinator abundance of each species in each season (``abundance[species]
        [season]``): the supply within its flight range, ``FF / FR conv(PS)``, where
        the flowers it forages on in the season are. Where it visits.
    total_abundance : dict
        Sum of the abundance over the species, by season.
    farm_pollinators, total_yield, wild_yield : xarray.DataArray or None
        With farms: the on-farm abundance of wild pollinators (summed over the
        seasons), the pollinator yield index ``min(managed + wild, 1)`` and its wild
        part ``max(total - managed, 0)`` (NaN outside the farms).
    farms : geopandas.GeoDataFrame or None
        The farms with ``p_abund`` (mean pollinator abundance in the farm's season),
        ``y_tot`` (yield index: ``1 - p_dep (1 - mean total yield)``), ``pdep_y_w``
        (mean wild pollinator yield) and ``y_wild`` (``p_dep pdep_y_w``); NaN for a
        farm over no cell.
    nesting, floral : dict
        (``intermediates=True``) Nesting substrate availability by substrate and
        floral resources by season, from the biophysical table (and the farms).
    habitat_nesting, foraging_effectiveness, floral_resources, convolved_supply : dict
        (``intermediates=True``) By species: ``HN = max_n N_n ns_n``, the local
        foraging effectiveness ``FE = sum_j FF_j``, its convolution ``FR`` with the
        species' kernel and the convolution of the supply.
    foraged_flowers : dict
        (``intermediates=True``) ``FF = RA fa``, by species and season.
    kernels : dict
        (``intermediates=True``) The exponential decay kernel of each species.
    half_saturation, farm_abundance : dict
        (``intermediates=True``, with farms) By season: the farms' half saturation
        and their on-farm wild pollinator abundance.
    managed : xarray.DataArray or None
        (``intermediates=True``, with farms) The farms' managed pollinators.
    """

    supply: dict
    abundance: dict
    total_abundance: dict
    farm_pollinators: xr.DataArray | None = None
    total_yield: xr.DataArray | None = None
    wild_yield: xr.DataArray | None = None
    farms: object = None
    nesting: dict = field(default_factory=dict)
    floral: dict = field(default_factory=dict)
    habitat_nesting: dict = field(default_factory=dict)
    foraging_effectiveness: dict = field(default_factory=dict)
    floral_resources: dict = field(default_factory=dict)
    convolved_supply: dict = field(default_factory=dict)
    foraged_flowers: dict = field(default_factory=dict)
    kernels: dict = field(default_factory=dict)
    half_saturation: dict = field(default_factory=dict)
    farm_abundance: dict = field(default_factory=dict)
    managed: xr.DataArray | None = None

    def _save_items(self):
        """The maps under InVEST's file names."""
        items = {}
        for sp, da in self.supply.items():
            items[f"pollinator_supply_{sp}"] = da
        for sp, d in self.abundance.items():
            for se, da in d.items():
                items[f"pollinator_abundance_{sp}_{se}"] = da
        for se, da in self.total_abundance.items():
            items[f"total_pollinator_abundance_{se}"] = da
        for k, v in (("farm_pollinators", self.farm_pollinators), ("total_pollinator_yield", self.total_yield),
                     ("wild_pollinator_yield", self.wild_yield), ("farm_results", self.farms),
                     ("managed_pollinators", self.managed)):
            if v is not None:
                items[k] = v
        for prefix, d in (("nesting_substrate_index", self.nesting), ("relative_floral_abundance_index", self.floral),
                          ("habitat_nesting_index", self.habitat_nesting),
                          ("local_foraging_effectiveness", self.foraging_effectiveness),
                          ("floral_resources", self.floral_resources), ("convolve_ps", self.convolved_supply),
                          ("half_saturation", self.half_saturation), ("farm_pollinator", self.farm_abundance)):
            for k, da in d.items():
                items[f"{prefix}_{k}"] = da
        for sp, d in self.foraged_flowers.items():
            for se, da in d.items():
                items[f"foraged_flowers_index_{sp}_{se}"] = da
        return items

    def __repr__(self) -> str:
        means = ", ".join(f"{se}: {float(np.nanmean(da.values)):.4g}" for se, da in self.total_abundance.items())
        return f"Pollination({len(self.supply)} species, mean total abundance {means})"


def _guilds(guilds):
    """The guild table: species sorted (as InVEST), seasons and substrates, the columns."""
    if not isinstance(guilds, pd.DataFrame):
        raise TypeError("the guild table is a DataFrame with a row per species")
    g = guilds.copy()
    g.columns = [str(c).strip().lower() for c in g.columns]
    if "species" in g.columns:
        g = g.set_index("species")
    g.index = [str(s) for s in g.index]
    if g.index.has_duplicates:
        raise ValueError("the guild table lists a species more than once")
    for c in ("alpha", "relative_abundance"):
        if c not in g.columns:
            raise ValueError(f"the guild table lacks the column {c!r}")
    seasons = sorted(m.group(1) for c in g.columns if (m := _SEASON.match(c)))
    substrates = sorted(m.group(1) for c in g.columns if (m := _SUBSTRATE.match(c)))
    if not seasons:
        raise ValueError("the guild table has no foraging_activity_<season>_index column")
    if not substrates:
        raise ValueError("the guild table has no nesting_suitability_<substrate>_index column")
    cols = ["alpha", "relative_abundance", *[f"foraging_activity_{s}_index" for s in seasons],
            *[f"nesting_suitability_{s}_index" for s in substrates]]
    g = g[cols].astype(np.float64)
    if g.isna().any().any():
        raise ValueError("the guild table has empty values")
    if np.any(g["alpha"].to_numpy() <= 0):
        raise ValueError("alpha (the mean flight distance, metres) must be positive")
    return g, seasons, substrates


def _check_names(found, wanted, kind, what):
    if sorted(found) != sorted(wanted):
        raise ValueError(f"the {what} must name the {kind} of the guild table, {sorted(wanted)}: got {sorted(found)}")


def _farms(farms, lu: LandUse, seasons, substrates):
    import geopandas as gpd

    if not isinstance(farms, gpd.GeoDataFrame):
        raise TypeError("the farms are a GeoDataFrame of polygons")
    f = farms.copy()
    lower = {str(c).lower(): c for c in f.columns if c != f.geometry.name}
    need = ["crop_type", "half_sat", "season", "p_dep", "p_managed", *[f"fr_{s}" for s in seasons],
            *[f"n_{s}" for s in substrates]]
    missing = [c for c in need if c not in lower]
    if missing:
        raise ValueError(f"the farms lack the column(s) {missing}")
    _check_names([m.group(1) for c in lower if (m := re.match(r"fr_([^_]+)$", c))], seasons, "seasons", "farms' fr_<season> columns")
    _check_names([m.group(1) for c in lower if (m := re.match(r"n_([^_]+)$", c))], substrates, "substrates", "farms' n_<substrate> columns")
    f = f.rename(columns={v: k for k, v in lower.items()})
    f["season"] = f["season"].astype(str)
    unknown = sorted(set(f["season"]) - set(seasons))
    if unknown:
        raise ValueError(f"the farms name season(s) {unknown} that the tables lack (seasons: {seasons})")
    if not np.all(f.geometry.geom_type.isin(["Polygon", "MultiPolygon"])):
        raise ValueError("the farms must be polygons")
    crs = lu.surface.crs
    if f.crs is not None and crs is not None and f.crs != crs:
        f = f.to_crs(crs)  # InVEST reprojects the farms to the land use map
    return f


def _burn(farms, column, lu: LandUse, base=None):
    """InVEST's _rasterize_vector_onto_base: the farms' values (float32) burnt over ``base`` (NaN: none)."""
    from .._vector import rasterize

    out = np.full(lu.shape, np.nan, np.float32) if base is None else base.copy()
    if len(farms):
        like = lu.raster(np.zeros(lu.shape, np.float32), "x")
        burned = rasterize(farms, like=like, field=column, fill=np.nan, dtype=np.float64)
        v = np.asarray(burned.values, np.float64)
        inside = ~np.isnan(v)
        out[inside] = v[inside].astype(np.float32)
    return out


def _farm_table(farms, pyt, pyw, total, like, n_threads):
    """InVEST's farm results: means over each farm of the yields and of the abundance in its season."""
    from .._zonal import zonal_stats

    names = ["pyt", "pyw", *[f"pa_{s}" for s in total]]
    maps = [pyt, pyw, *total.values()]
    stack = xr.concat([like.copy(data=np.asarray(m, np.float64)) for m in maps], dim="band")
    stack = stack.assign_coords(band=names)
    st = zonal_stats(stack, farms, ["sum", "count"], n_threads=n_threads)
    out = farms.copy()
    n = st["pyt_count"].to_numpy(np.float64)
    ok = n > 0
    nu = out["p_dep"].to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_t = st["pyt_sum"].to_numpy(np.float64) / n
        mean_w = st["pyw_sum"].to_numpy(np.float64) / st["pyw_count"].to_numpy(np.float64)
        mean_a = np.array([st[f"pa_{s}_sum"].iloc[i] / st[f"pa_{s}_count"].iloc[i]
                           for i, s in enumerate(out["season"])], np.float64)
    out["p_abund"] = np.where(ok, mean_a, np.nan)
    out["y_tot"] = np.where(ok, 1 - nu * (1 - mean_t), np.nan)
    out["pdep_y_w"] = np.where(ok, mean_w, np.nan)
    out["y_wild"] = np.where(ok, nu * mean_w, np.nan)
    return out


def pollination(lulc, guilds, biophysical, *, farms=None, nodata=None, intermediates: bool = False,
                n_threads: int | None = None) -> Pollination:
    """Pollinator abundance and crop pollination (InVEST's crop pollination model).

    Wild bees nest where the land offers their nesting substrates and fly to
    the flowers around (Lonsdorf et al. 2009). For each species ``s``: its
    habitat nesting index ``HN = max_n N_n ns_n`` (the availability of each
    substrate in the cell's land use class times the species' use of it); the
    flowers it forages on in each season, ``FF_j = RA_j fa_j`` (floral resources
    of the class times its relative activity in the season), summed over the
    seasons; the floral resources within its flight range ``FR``, that sum
    convolved with an exponential decay kernel of mean distance ``alpha`` (out
    to 5 alpha, normalized over the cells of the map); the supply
    ``PS = sa HN FR`` (``sa``: its relative abundance); and its abundance in each
    season, ``PA_j = FF_j / FR conv(PS)``. With farms, their own floral
    resources and nesting replace the land's, and each farm gets the on-farm
    wild pollinator abundance ``PAT (1 - h) / (h (1 - 2 PAT) + PAT)`` (``h`` its
    crop's half saturation, in its season), the yield index of all pollinators
    ``min(managed + wild, 1)`` and its wild part, averaged over the farm.

    The maps are InVEST's (natcap.invest 3.20), bit for bit: the kernels and
    the convolutions as pygeoprocessing makes them (256 x 256 blocks of the map
    and of the kernel, each pair by numpy's FFT, the same transforms, added up
    in float32), with one correction: pygeoprocessing zeroes the values of a
    block's convolution within 1e-8 of 1e-8 instead of within 1e-8 of 0, as it
    documents (here, of 0). The farm means agree to float32 precision (InVEST
    sums the cells in float32).

    Parameters
    ----------
    lulc
        Land use map (integer codes; projected, in metres).
    guilds
        A DataFrame with a row per species (``species``): ``alpha`` (mean flight
        distance, metres), ``relative_abundance``, a
        ``nesting_suitability_<substrate>_index`` column per nesting substrate and a
        ``foraging_activity_<season>_index`` column per season (0 to 1).
    biophysical
        A DataFrame by land use class (``lucode``): ``nesting_<substrate>_
        availability_index`` and ``floral_resources_<season>_index`` (0 to 1), for
        the substrates and seasons of the guild table.
    farms
        Optional farm polygons (GeoDataFrame): ``crop_type``, ``season`` (when the
        crop is pollinated), ``half_sat`` (the wild pollinator abundance that gives
        half the pollinator-dependent yield), ``p_dep`` (share of the yield that
        depends on pollinators), ``p_managed`` (share of pollination by managed
        pollinators), and the farm's own ``fr_<season>`` and ``n_<substrate>``.
    nodata
        Value marking missing cells in the land use map.
    intermediates
        Also keep the intermediate maps (the substrates and floral resources, the
        habitat nesting, foraged flowers, foraging effectiveness, floral resources
        and convolved supply of every species, the kernels, the farm rasters).

    Returns
    -------
    Pollination

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from affine import Affine
    >>> import landschaft as ls
    >>> codes = np.ones((60, 60), int)
    >>> codes[:, 30:] = 2                                      # forest (1), crops (2)
    >>> lulc = ls.load_raster(ls.write_raster(codes, "poll_lulc.tif", crs="EPSG:31983",
    ...                                       transform=Affine(30, 0, 500000, 0, -30, 8000000)))
    >>> guilds = pd.DataFrame({"species": ["bee"], "nesting_suitability_ground_index": [1.0],
    ...                        "foraging_activity_summer_index": [1.0], "alpha": [300.0],
    ...                        "relative_abundance": [1.0]})
    >>> bio = pd.DataFrame({"lucode": [1, 2], "nesting_ground_availability_index": [1.0, 0.1],
    ...                     "floral_resources_summer_index": [0.4, 0.9]})
    >>> p = ls.services.pollination(lulc, guilds, bio)
    >>> pa = p.abundance["bee"]["summer"].values
    >>> bool(pa[30, 31] > pa[30, 59])                          # more bees near the forest edge
    True
    """
    threads = resolve(n_threads)
    lu = LandUse(lulc, nodata, "the land use map")
    g, seasons, substrates = _guilds(guilds)
    species = sorted(g.index)
    cols = [*[f"nesting_{s}_availability_index" for s in substrates],
            *[f"floral_resources_{s}_index" for s in seasons]]
    bt = biophysical.copy() if isinstance(biophysical, pd.DataFrame) else biophysical
    if isinstance(bt, pd.DataFrame):
        lc = [str(c).strip().lower() for c in bt.columns]
        _check_names([m.group(1) for c in lc if (m := re.match(r"nesting_([^_]+)_availability_index$", c))],
                     substrates, "substrates", "biophysical table's nesting columns")
        _check_names([m.group(1) for c in lc if (m := re.match(r"floral_resources_([^_]+)_index$", c))],
                     seasons, "seasons", "biophysical table's floral columns")
    codes, table = lookup_table(biophysical, cols, "the biophysical table")
    cls = lu.classes(codes, threads)
    known = cls >= 0
    idx = np.maximum(cls, 0)

    def reclass(column):  # utils.reclassify_raster to float32
        v = table[column].to_numpy().astype(np.float32)
        return np.where(known, v[idx], np.float32(np.nan)).astype(np.float32)

    nesting = {s: reclass(f"nesting_{s}_availability_index") for s in substrates}
    floral = {s: reclass(f"floral_resources_{s}_index") for s in seasons}
    fv = None
    if farms is not None:
        fv = _farms(farms, lu, seasons, substrates)
        nesting = {s: _burn(fv, f"n_{s}", lu, nesting[s]) for s in substrates}
        floral = {s: _burn(fv, f"fr_{s}", lu, floral[s]) for s in seasons}

    # the scalars as InVEST computes them from its pandas tables (numpy float64)
    rel = g["relative_abundance"]
    total_rel = rel.sum()
    dx, dy = lu.pixel()
    pixel = dx if np.isclose(dx, dy) else min(dx, dy)

    out = Pollination({}, {}, {})
    if intermediates:
        out.nesting = {s: lu.raster(a, f"nesting_substrate_index_{s}") for s, a in nesting.items()}
        out.floral = {s: lu.raster(a, f"relative_floral_abundance_index_{s}") for s, a in floral.items()}
    total = {se: np.full(lu.shape, np.nan, np.float32) for se in seasons}
    nest_list = [np.ascontiguousarray(nesting[s]) for s in substrates]
    floral_list = [np.ascontiguousarray(floral[s]) for s in seasons]
    kernels = {}
    for sp in species:
        row = g.loc[sp]
        activity = np.array([row[f"foraging_activity_{s}_index"] for s in seasons], np.float64)
        fa = activity / np.sum([row[f"foraging_activity_{s}_index"] for s in seasons])
        ns = np.array([row[f"nesting_suitability_{s}_index"] for s in substrates], np.float64)
        hn, fe, ff = _core.serv_poll_local(nest_list, ns, floral_list, fa, True, threads)
        alpha = np.float64(row["alpha"]) / pixel
        # one kernel per alpha (InVEST shares a kernel between alphas equal to 6 decimals)
        kernel = kernels.get(float(alpha))
        if kernel is None:
            kernel = _core.serv_poll_kernel(float(alpha), float(alpha * 5))
            kernels[float(alpha)] = kernel
        fr = _core.serv_poll_convolve(fe, kernel, threads)
        sa = float(np.float64(rel[sp]) / total_rel)
        ps = _core.serv_poll_supply(hn, fr, sa, threads)
        cps = _core.serv_poll_convolve(ps, kernel, threads)
        out.supply[sp] = lu.raster(ps, f"pollinator_supply_{sp}")
        out.abundance[sp] = {}
        for j, se in enumerate(seasons):
            pa = _core.serv_poll_abundance(ff[j], fr, cps, total[se], True, threads)
            out.abundance[sp][se] = lu.raster(pa, f"pollinator_abundance_{sp}_{se}")
        if intermediates:
            out.habitat_nesting[sp] = lu.raster(hn, f"habitat_nesting_index_{sp}")
            out.foraging_effectiveness[sp] = lu.raster(fe, f"local_foraging_effectiveness_{sp}")
            out.floral_resources[sp] = lu.raster(fr, f"floral_resources_{sp}")
            out.convolved_supply[sp] = lu.raster(cps, f"convolve_ps_{sp}")
            out.foraged_flowers[sp] = {se: lu.raster(ff[j], f"foraged_flowers_index_{sp}_{se}")
                                       for j, se in enumerate(seasons)}
            out.kernels[sp] = kernel
    out.total_abundance = {se: lu.raster(t, f"total_pollinator_abundance_{se}") for se, t in total.items()}
    if fv is None:
        return out

    farm_total = np.full(lu.shape, np.nan, np.float32)
    for se in seasons:
        h = _burn(fv[fv["season"] == se], "half_sat", lu)
        fp = _core.serv_poll_farm(h, total[se], farm_total, intermediates, threads)
        if intermediates:
            out.half_saturation[se] = lu.raster(h, f"half_saturation_{se}")
            out.farm_abundance[se] = lu.raster(fp, f"farm_pollinator_{se}")
    mp = _burn(fv, "p_managed", lu)
    pyt, pyw = _core.serv_poll_yield(mp, farm_total, threads)
    out.farm_pollinators = lu.raster(farm_total, "farm_pollinators")
    out.total_yield = lu.raster(pyt, "total_pollinator_yield")
    out.wild_yield = lu.raster(pyw, "wild_pollinator_yield")
    if intermediates:
        out.managed = lu.raster(mp, "managed_pollinators")
    out.farms = _farm_table(fv, pyt, pyw, total, lu.raster(np.zeros(lu.shape), "x"), threads)
    return out
