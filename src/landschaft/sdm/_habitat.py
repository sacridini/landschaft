# SPDX-License-Identifier: GPL-3.0-or-later
"""Area of habitat (Brooks et al. 2019): a species' range refined by its habitats and elevations.

After plangea's ``harmonize_bd`` (International Institute for Sustainability; GPL >= 3): the habitat
of a species is, over its range, the share of each pixel covered by the natural classes it lives in
times the pixel's area; its potential habitat the same with the original vegetation; and the habitat
it gains per unit of a pixel restored, the share of the restored vegetation it lives in.

References
----------
Brooks, T. M. et al. (2019). Measuring terrestrial Area of Habitat (AOH) and its utility for the
    IUCN Red List. *Trends in Ecology & Evolution* 34, 977-986. doi:10.1016/j.tree.2019.06.009
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

from .. import _core, _threads
from .._layers import like, named_layers
from .._save import Saveable

__all__ = ["HabitatAreas", "area_of_habitat", "cell_area_km2"]


def cell_area_km2(template: xr.DataArray) -> np.ndarray:
    """The area of every cell of a raster's grid in km²: the product of the resolutions in a
    projected coordinate system (in its linear unit), the cell on the ellipsoid in a geographic one."""
    try:
        crs = template.rio.crs
        rx, ry = template.rio.resolution()
    except Exception as e:  # noqa: BLE001 - no grid
        raise ValueError("the cell area needs a georeferenced raster, or give cell_area") from e
    if crs is None:
        raise ValueError("the raster has no coordinate system: give cell_area (km²)")
    from pyproj import CRS, Geod

    c = CRS.from_user_input(crs)
    ny, nx = template.shape[-2:]
    if c.is_geographic:
        geod = Geod(ellps="WGS84") if c.ellipsoid is None else Geod(a=c.ellipsoid.semi_major_metre,
                                                                      rf=c.ellipsoid.inverse_flattening or 0.0)
        ys = np.asarray(template["y"].values, dtype=np.float64)
        half = abs(float(ry)) / 2.0
        w = abs(float(rx))
        rows = np.array([abs(geod.polygon_area_perimeter([0.0, w, w, 0.0], [y - half, y - half, y + half, y + half])[0])
                         for y in ys]) / 1e6
        return np.repeat(rows[:, None], nx, axis=1)
    unit = c.axis_info[0].unit_conversion_factor if c.axis_info else 1.0
    return np.full((ny, nx), abs(float(rx) * float(ry)) * unit * unit / 1e6)


@dataclass(frozen=True)
class HabitatAreas(Saveable):
    """Result of :func:`area_of_habitat`.

    Attributes
    ----------
    table
        One row per species: ``species``, its ``current`` and ``potential``
        area of habitat (km², or the unit of ``cell_area``), the number of
        cells of its range (``range_cells``, after the elevation limits),
        its natural ``classes`` and whether it is ``valid`` (some suitable
        natural class, current and potential habitat above 0).
    gain
        The habitat each species gains per unit of a cell restored (km² per
        unit of the cell's area restored): a sparse matrix, species (in the
        order of ``table``) x cells of the grid (row-major), over the cells
        of its range with anthropic land. None without the original vegetation.
    """

    table: pd.DataFrame
    gain: sp.csr_matrix | None = field(default=None, repr=False)
    _template: xr.DataArray | None = field(default=None, repr=False)

    def gain_map(self, species: str) -> xr.DataArray:
        """The habitat a species gains per unit of each cell restored, as a raster (0 elsewhere)."""
        if self.gain is None:
            raise ValueError("no gain: area_of_habitat was given no original vegetation")
        rows = np.flatnonzero(self.table["species"].astype(str).to_numpy() == str(species))
        if rows.size == 0:
            raise KeyError(f"no species {species!r}")
        values = np.asarray(self.gain[rows[0]].todense()).reshape(self._template.shape)
        return like(self._template, values, f"habitat_gain_{species}")

    def _save_items(self) -> dict:
        return {"table": self.table}


def _suitability(table, species: list[str]) -> dict[str, list[str]]:
    if table is None:
        return {}
    if isinstance(table, Mapping):
        return {str(k): [str(c) for c in ([v] if isinstance(v, str) else v)] for k, v in table.items()}
    t = pd.DataFrame(table)
    if t.shape[1] < 2:
        raise ValueError("suitability: a table of two columns (species, class)")
    out: dict[str, list[str]] = {}
    for s, c in zip(t.iloc[:, 0].astype(str), t.iloc[:, 1].astype(str)):
        out.setdefault(s, []).append(c)
    return out


def area_of_habitat(
    ranges,
    land_cover,
    natural: Iterable[str],
    *,
    suitability=None,
    original=None,
    elevation=None,
    elevation_limits=None,
    include_unlisted: bool = True,
    cell_area=None,
    n_threads: int | None = None,
) -> HabitatAreas:
    """Area of habitat of species: their ranges refined by habitats and elevations.

    The area of habitat (AOH; Brooks et al. 2019) of a species is the part
    of its range covered by the land-cover classes it lives in, within its
    elevation limits. With land cover given as shares of classes per cell
    (as plangea's restoration model uses it), every cell of the range counts
    for its area times the share of the species' natural classes. The
    potential habitat is the same with the original vegetation, and the
    gain map says how much habitat a species wins per unit of a cell
    restored: what a restoration plan needs (plangea's ``harmonize_bd``).

    Parameters
    ----------
    ranges
        The species' ranges: a ``(band, y, x)`` raster whose band names are
        the species (values above 0 inside the range), or a mapping or list
        of named rasters.
    land_cover
        The current share of every class in each cell (as
        :func:`landschaft.original_vegetation` takes it); NaN counts as 0.
    natural
        The natural classes: a species' habitat is among them.
    suitability
        The classes each species lives in: a table of two columns (species,
        class), a mapping species -> classes. Classes that are not natural
        are left out.
    original
        The original vegetation: a :class:`~landschaft.OriginalVegetation`
        (its original cover gives the potential habitat, its restored
        proportions and anthropic share the gain) or the original share of
        every natural class (then restored land returns to those
        proportions). Without it the potential habitat is the current one.
    elevation
        An elevation raster on the same grid.
    elevation_limits
        A table with the species and their lower and upper elevations
        (columns ``species``, ``lower``, ``upper``, or the first three
        columns in that order); a missing limit is no limit. Cells without
        an elevation stay in the range.
    include_unlisted
        Species without a row in ``suitability`` live in every natural class
        (True) or are left out (False, ``valid`` False).
    cell_area
        The area of a cell: a number, a raster, or None: from the grid in km²
        (see :func:`cell_area_km2`).
    n_threads
        Threads (default: all cores but one).

    Returns
    -------
    HabitatAreas
        The areas by species and the gain matrix.

    Notes
    -----
    The sums are plangea's, bit for bit: the shares of a species' classes
    added in their order, times the cell's area, summed over its range in
    R's 80-bit ``sum()``.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import landschaft as ls
    >>> cover = xr.DataArray(np.array([[[0.6, 0.0]], [[0.2, 0.1]], [[0.2, 0.9]]]),
    ...                      dims=("band", "y", "x"), coords={"band": ["forest", "savanna", "crops"]})
    >>> ranges = xr.DataArray(np.array([[[1.0, 1.0]], [[0.0, 1.0]]]),
    ...                       dims=("band", "y", "x"), coords={"band": ["toucan", "rhea"]})
    >>> veg = ls.original_vegetation(cover, ["forest", "savanna"], ["crops"])
    >>> aoh = ls.sdm.area_of_habitat(ranges, cover, ["forest", "savanna"], original=veg, cell_area=1.0,
    ...                              suitability={"toucan": ["forest"], "rhea": ["savanna"]})
    >>> aoh.table[["species", "current", "potential"]].round(3)
      species  current  potential
    0  toucan      0.6       0.75
    1    rhea      0.1       1.00
    """
    natural = [str(c) for c in natural]
    sp_names, rvals, template = named_layers(ranges, "ranges")
    names, cover, t2 = named_layers(land_cover, "land_cover")
    shape = rvals.shape[1:]
    if cover.shape[1:] != shape:
        raise ValueError(f"land_cover's grid {cover.shape[1:]} is not the ranges' {shape}")
    pos = {n: i for i, n in enumerate(names)}
    missing = [c for c in natural if c not in pos]
    if missing:
        raise ValueError(f"land_cover has no layer for {missing}")
    k = len(natural)
    ncell = shape[0] * shape[1]
    current = np.ascontiguousarray(np.nan_to_num(cover[[pos[c] for c in natural]], nan=0.0).reshape(k, ncell))
    restore = domain = None
    if original is None:
        orig = current
    else:
        from .._vegetation import OriginalVegetation

        if isinstance(original, OriginalVegetation):
            on, ov, _ = named_layers(original.original, "original")
            rn, rv, _ = named_layers(original.restore, "restore")
            restore = np.ascontiguousarray(np.nan_to_num(rv[[rn.index(c) for c in natural]], nan=0.0).reshape(k, ncell))
            domain = (np.nan_to_num(np.asarray(original.anthropic.values, dtype=np.float64)).ravel() > 0).astype(np.uint8)
        else:
            on, ov, _ = named_layers(original, "original")
        if ov.shape[1:] != shape:
            raise ValueError("the original vegetation is not on the ranges' grid")
        lost = [c for c in natural if c not in on]
        if lost:
            raise ValueError(f"the original vegetation has no layer for {lost}")
        # the land outside the original cover (NaN) keeps its current cover, as plangea's oa_terr
        o = ov[[on.index(c) for c in natural]].reshape(k, ncell)
        orig = np.ascontiguousarray(np.where(np.isnan(o), current, o))
        if restore is None:
            with np.errstate(invalid="ignore", divide="ignore"):
                total = orig.sum(axis=0)
                restore = np.ascontiguousarray(np.where(total > 0, orig / total, 0.0))
    if cell_area is None:
        area = cell_area_km2(template)
    elif isinstance(cell_area, (int, float, np.floating, np.integer)):
        area = np.full(shape, float(cell_area))
    else:
        area = np.asarray(getattr(cell_area, "values", cell_area), dtype=np.float64)
        if area.shape != shape:
            raise ValueError(f"cell_area's grid {area.shape} is not the ranges' {shape}")
    area = np.ascontiguousarray(np.nan_to_num(area, nan=0.0).ravel())

    elev = None
    limits = {}
    if elevation is not None:
        elev = np.asarray(getattr(elevation, "values", elevation), dtype=np.float64).squeeze()
        if elev.shape != shape:
            raise ValueError(f"elevation's grid {elev.shape} is not the ranges' {shape}")
        elev = elev.ravel()
        if elevation_limits is not None:
            t = pd.DataFrame(elevation_limits)
            cols = ["species", "lower", "upper"]
            t = t[cols] if all(c in t.columns for c in cols) else t.iloc[:, :3].set_axis(cols, axis=1)
            for s, lo, hi in t.itertuples(index=False):
                limits[str(s)] = (float(lo) if pd.notna(lo) else -np.inf, float(hi) if pd.notna(hi) else np.inf)

    table_classes = _suitability(suitability, sp_names)
    ptr = [0]
    idx = []
    suit = np.zeros((len(sp_names), k), dtype=np.uint8)
    classes_col, listed = [], []
    for s, name in enumerate(sp_names):
        cells = np.flatnonzero(np.nan_to_num(rvals[s].ravel(), nan=0.0) > 0)
        if elev is not None and name in limits:
            lo, hi = limits[name]
            e = elev[cells]
            cells = cells[np.isnan(e) | ((e >= lo) & (e <= hi))]
        idx.append(cells)
        ptr.append(ptr[-1] + cells.size)
        if name in table_classes:
            cls = [c for c in natural if c in table_classes[name]]  # natural ones, in their order
            listed.append(True)
        else:
            cls = natural if include_unlisted else []
            listed.append(include_unlisted)
        suit[s, [natural.index(c) for c in cls]] = 1
        classes_col.append(cls)
    range_ptr = np.asarray(ptr, dtype=np.int64)
    range_idx = np.concatenate(idx).astype(np.int64) if idx else np.zeros(0, np.int64)
    threads = _threads.resolve(n_threads)
    now, pot = _core.rest_habitat(current, orig, area, range_ptr, range_idx, suit, threads)
    gain = None
    if restore is not None:
        indptr, index, values = _core.rest_habitat_gain(restore, area, range_ptr, range_idx, suit, domain, threads)
        gain = sp.csr_matrix((values, index, indptr), shape=(len(sp_names), ncell))
    table = pd.DataFrame({
        "species": sp_names, "current": now, "potential": pot,
        "range_cells": np.diff(range_ptr), "classes": [", ".join(c) for c in classes_col],
        "valid": (now > 0) & (pot > 0) & suit.any(axis=1) & np.asarray(listed),
    })
    return HabitatAreas(table=table, gain=gain, _template=template)
