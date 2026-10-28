# SPDX-License-Identifier: GPL-3.0-or-later
"""Brazil's Forest Code (Lei 12.651/2012): the Áreas de Preservação Permanente and the Reserva Legal's balance.

The APPs of art. 4 (the margins of rivers by their width and of natural lakes
by their area, the springs, the veredas, the slopes above 45°, the hilltops of
:func:`landschaft.surface.hilltop`, the land above 1,800 m and protected
classes such as mangroves), with art. 61-A's narrower margins in the rural
areas consolidated by 22 July 2008 by the property's fiscal modules
(:func:`app`); and the balance of every rural property's Reserva Legal: art.
12's shares, art. 15's APP counted, art. 67's small properties
(:func:`reserve`). As Terraflux's ``forest_code.*`` operators.

Typical use::

    import landschaft as ls

    rivers = ls.load_raster("rivers.tif")              # the channels, non-zero
    a = ls.forest_code.app(rivers=rivers, river_width=ls.load_raster("width.tif"),
                           lakes=ls.load_raster("lakes.tif"), springs=ls.load_vector("springs.gpkg"),
                           slope=ls.surface.terrain(dem, "slope"), elevation=dem,
                           hilltops=ls.surface.hilltop(dem).app)
    a.app, a.table                                      # the kind of APP of every cell, the areas
    r = ls.forest_code.reserve(ls.load_raster("car.tif"), ls.load_raster("native.tif"), share=80, app=a.app)
    r.table, r.balance                                  # deficit and surplus of every property
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from ._save import Saveable

__all__ = ["APP", "APP_KINDS", "ReservaLegal", "app", "consolidated_lake", "consolidated_river", "reserve",
           "river_margin"]

#: The kinds of APP of :func:`app`, by their code (1..8).
APP_KINDS = {1: "river", 2: "lake", 3: "spring", 4: "vereda", 5: "slope", 6: "hilltop", 7: "altitude", 8: "class"}


def river_margin(width: float) -> float:
    """Art. 4, I: the margin of a river of a width (metres): 30 m below 10 m wide, 50 to 50, 100 to 200, 200 to 600, 500.

    Examples
    --------
    >>> import landschaft as ls
    >>> [ls.forest_code.river_margin(w) for w in (5, 10, 51, 600, 601)]
    [30.0, 50.0, 100.0, 200.0, 500.0]
    """
    if width < 10.0:
        return 30.0
    if width <= 50.0:
        return 50.0
    if width <= 200.0:
        return 100.0
    if width <= 600.0:
        return 200.0
    return 500.0


def consolidated_river(modules: float, over_four: float = 20.0) -> float:
    """Art. 61-A, §§ 1-4: a river's margin in a consolidated area by the property's fiscal modules.

    5, 8 and 15 m up to 1, 2 and 4 modules; above 4, the programme's width
    (``over_four``, 20 to 100 m).
    """
    if modules <= 1.0:
        return 5.0
    if modules <= 2.0:
        return 8.0
    if modules <= 4.0:
        return 15.0
    return float(over_four)


def consolidated_lake(modules: float) -> float:
    """Art. 61-A, § 6: a lake's margin in a consolidated area: 5, 8, 15 m up to 1, 2, 4 modules; 30 m beyond."""
    if modules <= 1.0:
        return 5.0
    if modules <= 2.0:
        return 8.0
    if modules <= 4.0:
        return 15.0
    return 30.0


def _on(v: np.ndarray) -> np.ndarray:
    return ~np.isnan(v) & (v != 0)


def _read(x, what: str):
    from .simulate._maps import read_map

    try:
        m = read_map(x, None)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what}: {e}") from None
    return np.asarray(m.as_float(), dtype=np.float64), m.template


def _grid(template) -> tuple[float, ...]:
    if template is None:
        raise ValueError("the maps need their grid (DataArrays from ls.load_raster), in metres")
    from .io import _transform

    t = _transform(template)
    return (t.c, t.a, t.b, t.f, t.d, t.e)


def _nearest(f: np.ndarray, dx: float, dy: float):
    """The exact distance of every cell's centre to the nearest cell of ``f`` and that cell's flat index."""
    from scipy import ndimage

    d, (ri, ci) = ndimage.distance_transform_edt(~f, sampling=(dy, dx), return_indices=True)
    return d, ri * f.shape[1] + ci


def _out(values: np.ndarray, template, name: str) -> xr.DataArray:
    out = template.copy(data=values).rename(name)
    out.attrs = {}
    out.encoding = {}
    try:
        out = out.rio.write_nodata(np.nan)
    except Exception:  # noqa: BLE001, S110  (no rioxarray georeference)
        pass
    return out


@dataclass
class APP(Saveable):
    """The Áreas de Preservação Permanente of a map (:func:`app`).

    Attributes
    ----------
    app : xarray.DataArray
        The APP of every cell: 1 river, 2 lake, 3 spring, 4 vereda, 5 slope,
        6 hilltop, 7 altitude, 8 protected class (the first that applies; see
        :data:`APP_KINDS`); 0 none.
    table : pandas.DataFrame
        A row per kind: ``kind`` (1-8), ``name``, ``cells``, ``area_ha``
        (each kind's cells, overlaps counted in each).
    """

    app: xr.DataArray
    table: pd.DataFrame

    def __repr__(self) -> str:
        a = np.asarray(self.app)
        return f"APP({int(np.sum(a > 0))} cells, {float(self.table['area_ha'].max()):.1f} ha in the largest kind)"


def app(*, rivers=None, river_width=5.0, lakes=None, urban=None, springs=None, veredas=None, slope=None,
        elevation=None, hilltops=None, landcover=None, protected_classes=(), properties=None, consolidated=None,
        fiscal_module: float | None = None, over_four_modules: float = 20.0) -> APP:
    """The Áreas de Preservação Permanente of Brazil's Forest Code (Lei 12.651/2012, art. 4).

    The margins of rivers from the channel's edge by their width (30 m below
    10 m wide, 50 m to 50, 100 m to 200, 200 m to 600, 500 m beyond), of
    natural lakes by their area (100 m; 50 m up to 20 ha; 30 m in urban
    zones), 50 m around springs and from veredas, slopes above 45°, hilltops,
    land above 1,800 m and protected classes (mangroves, restingas). In rural
    areas consolidated by 22 July 2008, art. 61-A's narrower margins by the
    property's fiscal modules: rivers and lakes 5, 8 and 15 m up to 1, 2 and 4
    modules, lakes 30 m and rivers the programme's width beyond; springs 15 m;
    veredas 30 or 50 m. Distances are exact (Euclidean, from cell centres; a
    margin from the channel's edge, half a cell from its centre). In maps of
    metres.

    Parameters
    ----------
    rivers
        The rivers' channels (non-zero).
    river_width
        The rivers' width, metres: a map (read at the nearest channel cell) or
        one number (default 5).
    lakes
        The natural lakes and lagoons (non-zero); a lake is a group of
        touching cells (8 neighbours).
    urban
        The urban zones (non-zero): lakes' margins of 30 m.
    springs
        The springs: a GeoDataFrame of points (reprojected to the maps' CRS)
        or an ``(n, 2)`` array of x, y.
    veredas
        The veredas' flooded areas (non-zero).
    slope
        The slope, degrees (APP above 45°).
    elevation
        The elevation, metres (APP above 1,800 m).
    hilltops
        The hilltops' APPs (:func:`landschaft.surface.hilltop`'s ``app``).
    landcover, protected_classes
        The land cover, and its classes that are APP.
    properties
        The rural properties (ids), for the consolidated areas' modules.
    consolidated
        The rural areas consolidated by 22 July 2008 (non-zero).
    fiscal_module
        The municipality's fiscal module, hectares (with ``consolidated``).
    over_four_modules
        The river margin of consolidated areas of properties above 4 modules,
        metres (default 20; the PRA sets 20 to 100).

    Returns
    -------
    APP

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> r = np.zeros((10, 20)); r[:, 0] = 1                   # a river 5 m wide down the first column
    >>> rivers = xr.DataArray(r, dims=("y", "x"), coords={"y": 95.0 - 10 * np.arange(10), "x": 5.0 + 10 * np.arange(20)})
    >>> ls.forest_code.app(rivers=rivers).app.values[0].tolist()[:6]     # 30 m from the channel's edge
    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    """
    named = {"rivers": rivers, "lakes": lakes, "urban": urban, "veredas": veredas, "slope": slope,
             "elevation": elevation, "hilltops": hilltops, "landcover": landcover, "properties": properties,
             "consolidated": consolidated}
    if isinstance(river_width, (xr.DataArray, np.ndarray)):
        named = {"rivers": rivers, "river_width": river_width, **{k: v for k, v in named.items() if k != "rivers"}}
    maps: dict[str, np.ndarray] = {}
    template = None
    for name, value in named.items():
        if value is None:
            continue
        v, t = _read(value, name)
        if maps and v.shape != next(iter(maps.values())).shape:
            raise ValueError(f"{name} and the other maps differ in size")
        if template is None:
            template = t
        maps[name] = v
    if not maps:
        raise ValueError("give a map at least")
    first = next(iter(maps.values()))
    t = _grid(template)
    ny, nx = first.shape
    dx, dy = abs(t[1]), abs(t[5])
    half = min(dx, dy) / 2.0
    cell_ha = dx * dy / 10_000.0
    cons = _on(maps["consolidated"]) if "consolidated" in maps else np.zeros(first.shape, bool)
    modules = np.full(first.shape, np.inf)
    if "consolidated" in maps:
        if fiscal_module is None or not fiscal_module > 0:
            raise ValueError("fiscal_module: the hectares of a fiscal module, with the consolidated areas")
        if "properties" in maps:
            p = maps["properties"]
            on = _on(p)
            ids, inv = np.unique(p[on], return_inverse=True)
            cells = np.bincount(inv, minlength=len(ids)).astype(np.float64)
            modules = np.full(first.shape, np.nan)
            modules[on] = cells[inv] * cell_ha / float(fiscal_module)
    kinds = np.zeros((8, ny, nx), bool)
    if "rivers" in maps:
        f = _on(maps["rivers"])
        if f.any():
            d, near = _nearest(f, dx, dy)
            if "river_width" in maps:
                w = maps["river_width"].ravel()[near]
                w = np.where(np.isnan(w), 5.0, w)
                margin = np.vectorize(river_margin, otypes=[float])(w)
            else:
                margin = np.full(first.shape, river_margin(float(river_width)))
            cm = np.vectorize(lambda m: consolidated_river(m, over_four_modules), otypes=[float])(modules)
            margin = np.where(cons, cm, margin)
            kinds[0] = ~f & (d - half <= margin)
    if "lakes" in maps:
        from scipy import ndimage

        f = _on(maps["lakes"])
        if f.any():
            label, _ = ndimage.label(f, structure=np.ones((3, 3), bool))
            sizes = np.bincount(label.ravel()).astype(np.float64)
            d, near = _nearest(f, dx, dy)
            area = sizes[label.ravel()[near]] * cell_ha
            margin = np.where(area <= 20.0, 50.0, 100.0)
            if "urban" in maps:
                margin = np.where(_on(maps["urban"]), 30.0, margin)
            margin = np.where(cons, np.vectorize(consolidated_lake, otypes=[float])(modules), margin)
            kinds[1] = ~f & (d - half <= margin)
    if springs is not None:
        f = np.zeros(first.shape, bool)
        if hasattr(springs, "geometry"):
            g = springs.geometry
            crs = getattr(template.rio, "crs", None) if template is not None else None
            if getattr(springs, "crs", None) is not None and crs is not None:
                g = g.to_crs(crs)
            if set(g.geom_type.dropna()) - {"Point"}:
                raise ValueError("springs: points")
            sx, sy = g.x.to_numpy(np.float64), g.y.to_numpy(np.float64)
        else:
            a = np.asarray(springs, dtype=np.float64).reshape(-1, 2)
            sx, sy = a[:, 0], a[:, 1]
        c = np.floor((sx - t[0]) / t[1])
        r = np.floor((sy - t[3]) / t[5])
        ok = (c >= 0) & (r >= 0) & (c < nx) & (r < ny)
        f[r[ok].astype(int), c[ok].astype(int)] = True
        if f.any():
            d, _ = _nearest(f, dx, dy)
            kinds[2] = d <= np.where(cons, 15.0, 50.0)
    if "veredas" in maps:
        f = _on(maps["veredas"])
        if f.any():
            d, _ = _nearest(f, dx, dy)
            margin = np.where(cons, np.where(modules <= 4.0, 30.0, 50.0), 50.0)
            kinds[3] = ~f & (d - half <= margin)
    if "slope" in maps:
        kinds[4] = maps["slope"] > 45.0
    if "hilltops" in maps:
        kinds[5] = _on(maps["hilltops"])
    if "elevation" in maps:
        kinds[6] = maps["elevation"] > 1800.0
    if "landcover" in maps:
        kinds[7] = np.isin(maps["landcover"], np.asarray(list(protected_classes), dtype=np.float64))
    anyk = kinds.any(axis=0)
    out = np.where(anyk, np.argmax(kinds, axis=0) + 1.0, np.where(np.isnan(first), np.nan, 0.0))
    counts = kinds.reshape(8, -1).sum(axis=1).astype(np.int64)
    table = pd.DataFrame({"kind": np.arange(1, 9), "name": list(APP_KINDS.values()), "cells": counts,
                          "area_ha": counts * cell_ha})
    return APP(_out(out, template, "app"), table)


@dataclass
class ReservaLegal(Saveable):
    """The Reserva Legal's balance of every rural property (:func:`reserve`).

    Attributes
    ----------
    table : pandas.DataFrame
        A row per property: ``property``, ``area_ha``, ``modules``,
        ``native_ha``, ``app_ha``, ``app_native_ha``, ``share`` (percent),
        ``required_ha``, ``counted_ha``, ``deficit_ha``, ``surplus_ha``.
    balance : xarray.DataArray
        Every property's surplus (positive) or deficit (negative), hectares, on
        its cells.
    """

    table: pd.DataFrame
    balance: xr.DataArray

    def __repr__(self) -> str:
        t = self.table
        return (f"ReservaLegal({len(t)} properties, deficit {t['deficit_ha'].sum():.1f} ha, "
                f"surplus {t['surplus_ha'].sum():.1f} ha)")


def reserve(properties, vegetation, *, share=20.0, app=None, vegetation_2008=None,
            fiscal_module: float | None = None, count_app: bool = True) -> ReservaLegal:
    """The balance of every rural property's Reserva Legal (Lei 12.651/2012).

    The share of its area a property must keep in native vegetation (art. 12:
    80 % in the Amazon's forests, 35 % in its cerrado, 20 % in its campos
    gerais and the rest of the country; a map of the share, or one for all),
    its native vegetation counted outside the APPs and, as art. 15 allows,
    inside them, and its deficit or surplus. The properties of up to 4 fiscal
    modules on 22 July 2008 keep only the vegetation they had then (art. 67).

    Parameters
    ----------
    properties
        The rural properties (ids; the CAR's).
    vegetation
        The native vegetation (non-zero).
    share
        The Reserva Legal's share, percent: a map, or one number (default 20).
    app
        The APPs (:func:`app`'s ``app``; non-zero).
    vegetation_2008
        The native vegetation of 22 July 2008 (art. 67, with
        ``fiscal_module``).
    fiscal_module
        The fiscal module, hectares (the ``modules`` column; art. 67).
    count_app
        Count the native vegetation of the APPs (art. 15; default True).

    Returns
    -------
    ReservaLegal

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> c = {"y": 450.0 - 100 * np.arange(5), "x": 50.0 + 100 * np.arange(10)}       # cells of 1 ha
    >>> props = xr.DataArray(np.repeat([[1.0] * 5 + [2.0] * 5], 5, axis=0), dims=("y", "x"), coords=c)
    >>> veg = props.copy(data=(np.arange(10) < 3) * np.ones((5, 1)))
    >>> ls.forest_code.reserve(props, veg).table[["property", "required_ha", "counted_ha", "deficit_ha", "surplus_ha"]]
       property  required_ha  counted_ha  deficit_ha  surplus_ha
    0       1.0          5.0        15.0         0.0        10.0
    1       2.0          5.0         0.0         5.0         0.0
    """
    p, template = _read(properties, "properties")
    t = _grid(template)
    veg, _ = _read(vegetation, "vegetation")

    def other(x, what):
        if x is None:
            return None
        v, _ = _read(x, what)
        if v.shape != p.shape:
            raise ValueError(f"{what} and the properties differ in size")
        return v

    if veg.shape != p.shape:
        raise ValueError("vegetation and the properties differ in size")
    app_v = other(app, "app")
    veg08 = other(vegetation_2008, "vegetation_2008")
    share_v = other(share, "share") if isinstance(share, (xr.DataArray, np.ndarray)) else None
    default_share = 20.0 if share_v is not None else float(share)
    ha = abs(t[1]) * abs(t[5]) / 10_000.0
    on = _on(p)
    ids, inv = np.unique(p[on], return_inverse=True)
    k = len(ids)

    def count(mask):
        return np.bincount(inv, weights=mask[on].astype(np.float64), minlength=k)

    in_app = _on(app_v) if app_v is not None else np.zeros(p.shape, bool)
    native = _on(veg)
    area = np.bincount(inv, minlength=k).astype(np.float64)
    sv = np.full(p.shape, default_share) if share_v is None else np.where(np.isnan(share_v), default_share, share_v)
    share_sum = np.zeros(k)
    np.add.at(share_sum, inv, sv[on])
    keep = native & (count_app | ~in_app)
    counted = count(keep) * ha
    counted08 = count(_on(veg08) & (count_app | ~in_app)) * ha if veg08 is not None else np.zeros(k)
    required = share_sum * ha / 100.0
    area_ha = area * ha
    modules = area_ha / fiscal_module if fiscal_module else np.full(k, np.nan)
    req = required.copy()
    if veg08 is not None:
        small = modules <= 4.0
        req[small] = np.minimum(req[small], counted08[small])
    deficit = np.maximum(req - counted, 0.0)
    surplus = np.maximum(counted - req, 0.0)
    table = pd.DataFrame({"property": ids, "area_ha": area_ha, "modules": modules, "native_ha": count(native) * ha,
                          "app_ha": count(in_app) * ha, "app_native_ha": count(in_app & native) * ha,
                          "share": required / area_ha * 100.0, "required_ha": req, "counted_ha": counted,
                          "deficit_ha": deficit, "surplus_ha": surplus})
    balance = np.full(p.shape, np.nan)
    balance[on] = (surplus - deficit)[inv]
    return ReservaLegal(table, _out(balance, template, "balance"))
