# SPDX-License-Identifier: GPL-3.0-or-later
"""Carbon storage and sequestration (InVEST's ``carbon`` model)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._save import Saveable
from .._threads import resolve
from ._common import LandUse, lookup_table, same_grid

__all__ = ["Carbon", "carbon", "valuation_constant"]

POOLS = ("c_above", "c_below", "c_soil", "c_dead")


@dataclass
class Carbon(Saveable):
    """Carbon stored in a landscape, and its change (:func:`carbon`).

    Maps are densities in metric tons of carbon per hectare (float32, as InVEST
    writes them), NaN where the land use map has no value.

    Attributes
    ----------
    storage : xarray.DataArray
        Total carbon of the baseline map: the sum of the four pools.
    pools : dict
        Each pool's map (``c_above``, ``c_below``, ``c_soil``, ``c_dead``).
    storage_future, pools_future
        The same for the alternate (future) map, if given.
    change : xarray.DataArray or None
        Sequestration: future minus baseline storage.
    npv : xarray.DataArray or None
        Net present value of the change (currency per hectare), with valuation.
    totals : pandas.DataFrame
        The landscape totals (``label``, ``total``, ``units``): tons of carbon
        (densities times the cell areas) and currency.
    valuation : float or None
        The factor turning a change in carbon into its net present value.
    """

    storage: xr.DataArray
    pools: dict
    totals: pd.DataFrame
    storage_future: xr.DataArray | None = None
    pools_future: dict = field(default_factory=dict)
    change: xr.DataArray | None = None
    npv: xr.DataArray | None = None
    valuation: float | None = None

    def __repr__(self) -> str:
        t = dict(zip(self.totals["label"], self.totals["total"]))
        parts = ", ".join(f"{k}: {v:.6g}" for k, v in t.items())
        return f"Carbon({parts})"


def valuation_constant(price: float, discount_rate: float, rate_change: float, year: int, future_year: int) -> float:
    """The factor turning a change in carbon into its net present value (InVEST's).

    The change of storage between the two years, spread evenly over them, valued
    at ``price`` per ton rising by ``rate_change`` percent a year and discounted
    at ``discount_rate`` percent a year:
    ``price / n * sum_{t=0}^{n-1} 1 / ((1 + d/100) (1 + r/100))^t``, n the years.

    Examples
    --------
    >>> import landschaft as ls
    >>> round(ls.services.valuation_constant(43.0, 7.0, 0.0, 2000, 2030), 6)
    19.031333
    """
    n_years = future_year - year
    if n_years <= 0:
        raise ValueError(f"the future year ({future_year}) must come after the baseline year ({year})")
    # the arithmetic of InVEST's _calculate_valuation_constant, step by step
    ratio = 1 / ((1 + discount_rate / 100) * (1 + rate_change / 100))
    constant = price / n_years
    if ratio == 1:
        constant *= n_years
    else:
        constant *= (1 - ratio ** n_years) / (1 - ratio)
    return constant


def _maps(lu: LandUse, codes, pools32, threads):
    cls = lu.classes(codes, threads)
    above, below, soil, dead, tot = _core.serv_carbon_pools(cls, pools32, threads)
    return dict(zip(POOLS, (above, below, soil, dead))), tot


def carbon(lulc, pools, *, future=None, price: float | None = None, discount_rate: float | None = None,
           rate_change: float = 0.0, year: int | None = None, future_year: int | None = None, nodata=None,
           n_threads: int | None = None) -> Carbon:
    """Carbon storage of a landscape and its change between two maps (InVEST's carbon model).

    Each land use class holds a density of carbon (tons per hectare) in four
    pools: aboveground and belowground biomass, soil and dead organic matter.
    The storage of a map is the sum of its pools cell by cell; with a future
    map, the change is the sequestration (future minus baseline), and with a
    price, its net present value.

    The maps are those of InVEST's ``carbon`` model (natcap.invest 3.20), bit for
    bit: float32 densities, the pools summed in their order, the change and its
    value in float32; the landscape totals agree to the last digits.

    Parameters
    ----------
    lulc
        The baseline land use map (integer codes; a projected raster in metres).
    pools
        The densities by class: a DataFrame with ``lucode`` (or indexed by the
        codes) and ``c_above``, ``c_below``, ``c_soil``, ``c_dead`` (Mg C / ha),
        or a ``dict`` ``{code: {"c_above": ..., ...}}``. Every class of the maps
        must be in it.
    future
        The alternate (future) land use map, on the same grid.
    price, discount_rate, rate_change
        Valuation of the change (with ``future``): the price of a ton of carbon
        and the annual discount rate and change of the price, in percent.
    year, future_year
        The years of the two maps (needed with ``price``).
    nodata
        Value marking missing cells in the maps (besides their own nodata).

    Returns
    -------
    Carbon

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from affine import Affine
    >>> import landschaft as ls
    >>> lulc = ls.write_raster(np.array([[1, 1, 2], [2, 3, 3]]), "lulc.tif", crs="EPSG:31983",
    ...                        transform=Affine(30, 0, 500000, 0, -30, 8000000))   # 30 m cells
    >>> pools = pd.DataFrame({"lucode": [1, 2, 3], "c_above": [140, 35, 5], "c_below": [40, 10, 2],
    ...                       "c_soil": [90, 80, 60], "c_dead": [12, 3, 0]})
    >>> later = np.array([[1, 2, 2], [2, 3, 3]])
    >>> c = ls.services.carbon(ls.load_raster(lulc), pools, future=later)
    >>> c.storage.values[0, 0], c.change.values[0, 1]
    (np.float32(282.0), np.float32(-154.0))
    >>> c.totals
                          label  total units
    0   Baseline Carbon Storage  85.86     t
    1  Alternate Carbon Storage  72.00     t
    2  Change in Carbon Storage -13.86     t
    """
    threads = resolve(n_threads)
    lu = LandUse(lulc, nodata, "the land use map")
    codes, table = lookup_table(pools, POOLS, "the carbon pools")
    pools32 = np.ascontiguousarray(table.to_numpy(np.float64).astype(np.float32))
    dx, dy = lu.pixel()
    area_ha = dx * dy / 10000
    maps, storage = _maps(lu, codes, pools32, threads)
    out = Carbon(lu.raster(storage, "c_storage_bas"), {k: lu.raster(v, f"{k}_bas") for k, v in maps.items()},
                 None)
    rows = [("Baseline Carbon Storage", _core.serv_total(storage, threads) * area_ha, "t")]
    if price is not None and future is None:
        raise ValueError("the valuation values a change: give the future map")
    if future is not None:
        fu = LandUse(future, nodata, "the future land use map")
        same_grid(lu, fu)
        fmaps, fstorage = _maps(fu, codes, pools32, threads)
        out.storage_future = fu.raster(fstorage, "c_storage_alt")
        out.pools_future = {k: fu.raster(v, f"{k}_alt") for k, v in fmaps.items()}
        change = fstorage - storage  # float32, NaN where either has no value
        out.change = lu.raster(change, "c_change_bas_alt")
        rows += [("Alternate Carbon Storage", _core.serv_total(fstorage, threads) * area_ha, "t"),
                 ("Change in Carbon Storage", _core.serv_total(change, threads) * area_ha, "t")]
        if price is not None:
            if discount_rate is None or year is None or future_year is None:
                raise ValueError("the valuation needs price, discount_rate, year and future_year")
            k = valuation_constant(float(price), float(discount_rate), float(rate_change), int(year),
                                   int(future_year))
            out.valuation = k
            npv = change * np.float32(k)  # float32, as numpy multiplies a float32 map by a Python float
            out.npv = lu.raster(npv, "npv_alt")
            rows.append(("Net Present Value of Carbon Change", _core.serv_total(npv, threads) * area_ha,
                         "currency"))
    out.totals = pd.DataFrame(rows, columns=["label", "total", "units"])
    return out
