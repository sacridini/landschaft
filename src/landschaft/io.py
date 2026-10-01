# SPDX-License-Identifier: GPL-3.0-or-later
"""Raster input/output built on rasterio and rioxarray."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.errors
import rasterio.windows
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from ._palettes import (
    CHANGE_PALETTE,
    CONTORTION_PALETTE,
    DIST_PALETTE,
    ENTROPY_PALETTE,
    FAD5_PALETTE,
    FAD_PALETTE,
    FOSCHANGE_PALETTE,
    LM_PALETTE,
    MSPA_ON_PALETTE,
    RESISTANCE_PALETTE,
)
from .validation import BACKGROUND, FOREGROUND, MISSING, InvalidInputError, validate_mspa_input

__all__ = [
    "load_raster",
    "write_raster",
    "reclassify",
    "mspa_colormap",
    "gtn_components_colormap",
    "gtn_importance_colormap",
    "GTN_COMPONENTS_COLORMAP",
    "GTN_IMPORTANCE_COLORMAP",
    "fad_colormap",
    "fad5_colormap",
    "dist_colormap",
    "entropy_colormap",
    "lm_colormap",
    "contortion_colormap",
    "resistance_colormap",
    "change_colormap",
    "foschange_colormap",
    "accounting_colormap",
    "FAD_COLORMAP",
    "DIST_COLORMAP",
    "ENTROPY_COLORMAP",
    "LM_COLORMAP",
    "CONTORTION_COLORMAP",
    "RESISTANCE_COLORMAP",
    "CHANGE_COLORMAP",
    "FOSCHANGE_COLORMAP",
    "ACCOUNTING_COLORMAP",
]

PathLike = str | os.PathLike


def load_raster(
    source,
    *,
    band: int | Iterable[int] | None = None,
    masked: bool | str = "auto",
    chunks: Any = None,
    validate: str | None = None,
    stack: bool = False,
    errors: str = "raise",
    recursive: bool = False,
):
    """Read one raster, or collect many.

    One function for every raster landschaft reads: habitat and land-cover
    maps, continuous surfaces, species stacks, costs. The result is a
    georeferenced ``xarray.DataArray`` (CRS, transform and NoData via
    ``.rio``) that :func:`write_raster` writes back as it was.

    Parameters
    ----------
    source
        One raster (path or GDAL URL), a directory (its ``.tif``/``.tiff``
        files), a glob pattern (``"tiles/*.tif"``) or a list of any of these.
    band
        Bands to read (1-based): ``None`` (default) reads them all, an integer
        one band, a list several. One band gives a ``(y, x)`` array, several a
        ``(band, y, x)`` array whose ``band`` coordinate holds the band
        descriptions when every band has a distinct one (else the band
        numbers). Several rasters: each file must have one band, or give
        ``band=N``.
    masked
        NoData handling. ``"auto"`` (default): in floating-point rasters NoData
        becomes NaN; integer rasters (class maps) keep their codes, with the
        NoData value in ``.rio.nodata``, so they are not turned into floats.
        ``True``: NoData becomes NaN in every raster (integers become
        ``float32``, or ``float64`` for 32- and 64-bit integers). ``False``: the
        values as stored.
    chunks
        Single raster only: ``None`` (default) loads the data into memory;
        anything else (e.g. ``"auto"``) returns a lazy dask-backed array.
    validate
        ``"mspa"`` checks that the map is an MSPA input: ``uint8`` coded
        0 = missing, 1 = background, 2 = foreground, with some foreground and
        some background. ``None`` (default) reads any raster; the
        habitat-pattern tools check their input themselves.
    stack
        Several rasters only: return a lazy ``(file, y, x)`` DataArray (one
        dask chunk per file) instead of a :class:`RasterCollection`. All
        rasters must share the same grid.
    errors
        Several rasters only: ``"raise"`` (default) if any file is unusable,
        ``"skip"`` to leave such files out (listed in ``.skipped``).
    recursive
        Several rasters only: also search sub-directories.

    Returns
    -------
    xarray.DataArray or RasterCollection
        For one raster, a georeferenced ``(y, x)`` or ``(band, y, x)`` array.
        For several, a :class:`RasterCollection` that reads only the file
        headers now; pass it to :func:`landschaft.mspa` to process every raster
        in parallel. With ``stack=True``, a lazy DataArray.

    Raises
    ------
    InvalidInputError
        If a file does not satisfy the requested checks.

    Examples
    --------
    A habitat map (0 missing, 1 background, 2 habitat) and a land-cover map:

    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> forest.shape, str(forest.dtype)
    ((1000, 1000), 'uint8')
    >>> landcover = ls.load_raster(ls.example_data("clc"))
    >>> landcover.rio.crs.to_epsg()
    3035

    Checked as an MSPA input on reading:

    >>> forest = ls.load_raster(ls.example_data("input"), validate="mspa")

    A multi-band stack keeps its band names; NoData of floats is NaN:

    >>> import numpy as np
    >>> import xarray as xr
    >>> habitats = xr.concat([landcover.isin([23, 24, 25]), landcover == 41], dim="band").astype("float32")
    >>> habitats = habitats.assign_coords(band=["forest", "water"]).where(landcover != 1)  # no data in towns
    >>> _ = ls.write_raster(habitats, "stacks/habitats.tif")   # creates the folder
    >>> stack = ls.load_raster("stacks/habitats.tif")
    >>> stack.dims, stack.band.values.tolist()
    (('band', 'y', 'x'), ['forest', 'water'])
    >>> bool(stack.sel(band="water").isnull().any()), ls.load_raster("stacks/habitats.tif", band=2).dims
    (True, ('y', 'x'))

    Many rasters at once (a folder, a pattern or a list) give a collection:

    >>> _ = ls.write_raster(forest[:500], "north.tif")
    >>> _ = ls.write_raster(forest[500:], "south.tif")
    >>> maps = ls.load_raster(["north.tif", "south.tif"])
    >>> len(maps)
    2
    """
    from .collection import RasterCollection, is_single_source

    if validate not in (None, "mspa"):
        raise ValueError(f"validate must be 'mspa' or None, got {validate!r}")
    if masked not in ("auto", True, False):
        raise ValueError(f"masked must be 'auto', True or False, got {masked!r}")
    if not is_single_source(source):
        if band is not None and not isinstance(band, (int, np.integer)):
            raise ValueError("several rasters: band must be one band number")
        collection = RasterCollection.from_sources(
            source, band=band, validate=validate, errors=errors, recursive=recursive, masked=masked
        )
        return collection.to_stack() if stack else collection
    return _load_single(source, band=band, masked=masked, chunks=chunks, validate=validate)


def _band_names(raw: xr.DataArray, n: int) -> list | None:
    """The band descriptions when every band has a distinct, non-empty one."""
    names = raw.attrs.get("long_name")
    if isinstance(names, str):
        names = (names,)
    if not isinstance(names, (tuple, list)) or len(names) != n:
        return None
    names = [str(v) for v in names]
    if any(not v.strip() for v in names) or len(set(names)) != n:
        return None
    return names


def _load_single(source: PathLike, *, band, masked, chunks: Any, validate: str | None) -> xr.DataArray:
    raw = rioxarray.open_rasterio(source, chunks=chunks, masked=False)
    if isinstance(raw, (list, xr.Dataset)):
        raise InvalidInputError(f"{source}: file contains several sub-datasets; open one of them explicitly.")
    try:
        if "band" not in raw.dims:
            raw = raw.expand_dims(band=[1])
        n_bands = raw.sizes["band"]
        names = _band_names(raw, n_bands)
        single = band is not None and isinstance(band, (int, np.integer))
        wanted = [int(band)] if single else list(range(1, n_bands + 1)) if band is None else [int(b) for b in band]
        for b in wanted:
            if not 1 <= b <= n_bands:
                raise InvalidInputError(f"{source}: band {b} not found (file has {n_bands} bands).")
        if single or (band is None and n_bands == 1):
            da = raw.isel(band=wanted[0] - 1, drop=True)
            da.attrs.pop("long_name", None)
            if names:
                da.attrs["long_name"] = names[wanted[0] - 1]
        else:
            da = raw.isel(band=[b - 1 for b in wanted])
            labels = [names[b - 1] for b in wanted] if names else wanted
            da = da.assign_coords(band=labels)
            if names:
                da.attrs["long_name"] = tuple(labels)
        if chunks is None:
            da = da.load()
    finally:
        if chunks is None:
            raw.close()  # release the file: on Windows an open handle locks it
    da = _mask_nodata(da, masked)
    da.name = Path(str(source)).stem
    da.attrs["source"] = str(source)

    if validate == "mspa":
        try:
            validate_mspa_input(da)
        except InvalidInputError as err:
            raise InvalidInputError(f"{source}: {err}") from None
    return da


def _mask_nodata(da: xr.DataArray, masked) -> xr.DataArray:
    """NoData to NaN as ``load_raster``'s ``masked`` says (``.rio`` metadata kept)."""
    if masked is False:
        return da
    is_float = np.issubdtype(da.dtype, np.floating)
    if not is_float and masked == "auto":
        return da
    nodata = da.rio.nodata
    attrs = dict(da.attrs)
    out = da
    if not is_float:
        out = da.astype(np.float32 if da.dtype.itemsize <= 2 else np.float64)
    if nodata is not None and not np.isnan(nodata):
        stored = np.array(nodata).astype(da.dtype)  # the NoData value as the band stores it
        out = out.where(da != stored)
    out.attrs = {k: v for k, v in attrs.items() if k not in ("_FillValue", "nodata")}
    out.encoding = {k: v for k, v in da.encoding.items() if k not in ("_FillValue", "dtype")}
    return out.rio.write_nodata(np.nan, encoded=False)


def _load_map(source, *, band: int | None = None, validate: str | None = None, **kwargs) -> xr.DataArray:
    """One 2-D map, for the tools that work on a single band (several bands: an error)."""
    da = load_raster(source, band=band, validate=validate, **kwargs)
    if isinstance(da, xr.DataArray) and da.ndim == 3:
        raise InvalidInputError(
            f"{source}: input map has {da.sizes['band']} bands, a single band is required. "
            "Select one with band=N."
        )
    return da


def reclassify(
    data,
    foreground: Iterable[int] | int | Callable[[np.ndarray], np.ndarray],
    *,
    missing: Iterable[int] | int | None = None,
    background: Iterable[int] | int | None = None,
):
    """Build a habitat map (0 = missing, 1 = background, 2 = habitat) from any raster.

    Parameters
    ----------
    data
        Categorical or continuous map: numpy array or DataArray, any integer or
        float type.
    foreground
        Value(s) that become habitat (2), or a function of the values returning
        a boolean mask (e.g. ``lambda a: a >= 30`` on a tree-cover percentage).
    missing
        Value(s) that become missing (0). NaN and the raster's NoData value
        (``data.rio.nodata``) are always missing.
    background
        Value(s) that become background (1). By default every value that is
        neither habitat nor missing; when given, the remaining values become
        missing.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Same type and georeferencing as ``data``, ``uint8`` coded 0/1/2, ready
        for :func:`mspa`, :func:`fad` and the other habitat-pattern tools.

    Examples
    --------
    Forests of a CORINE land-cover map (classes 23, 24 and 25):

    >>> import landschaft as ls
    >>> landcover = ls.load_raster(ls.example_data("clc"))
    >>> forest = ls.reclassify(landcover, foreground=[23, 24, 25])
    >>> sorted(int(v) for v in set(forest.values.ravel()))
    [1, 2]
    >>> morph = ls.mspa(forest)

    A rule on a continuous map, water (class 41) left out as missing:

    >>> tall = ls.reclassify(landcover, foreground=lambda a: (a >= 23) & (a <= 25), missing=[41])
    """
    values = np.asarray(data)

    def isin(spec) -> np.ndarray:
        if spec is None:
            return np.zeros(values.shape, dtype=bool)
        return np.isin(values, np.atleast_1d(np.asarray(list(spec) if isinstance(spec, Iterable) else spec)))

    fg = np.asarray(foreground(values), dtype=bool) if callable(foreground) else isin(foreground)
    miss = isin(missing)
    if np.issubdtype(values.dtype, np.floating):
        miss |= np.isnan(values)
    nodata = data.rio.nodata if isinstance(data, xr.DataArray) else None
    if nodata is not None:
        miss |= values == nodata

    out = np.full(values.shape, MISSING if background is not None else BACKGROUND, dtype=np.uint8)
    if background is not None:
        out[isin(background)] = BACKGROUND
    out[fg] = FOREGROUND
    out[miss] = MISSING

    if isinstance(data, xr.DataArray):
        result = data.copy(data=out)
        result.attrs = {k: v for k, v in data.attrs.items() if k not in ("_FillValue", "scale_factor", "add_offset")}
        result.encoding = {}
        return result.rio.write_nodata(None)
    return out


# --- colour tables ----------------------------------------------------------

# fsp.c colour LUT (16-bit) converted to 8-bit, i.e. GuidosToolbox MSPA colours.
_MSPA_COLORS = {
    0: (220, 220, 220),  # background
    1: (255, 140, 0),  # branch
    3: (0, 0, 0),  # edge
    5: (0, 0, 255),  # perforation
    9: (160, 60, 0),  # islet
    17: (0, 200, 0),  # core
    33: (255, 0, 0),  # bridge
    65: (255, 255, 0),  # loop
    100: (136, 136, 136),  # core-opening
    129: (255, 255, 255),  # missing
    220: (194, 194, 194),  # border-opening
}
_TRANSITION_ON = {35: (255, 0, 0), 37: (255, 0, 0), 67: (255, 255, 0), 69: (255, 255, 0)}
_TRANSITION_OFF = {35: (0, 0, 0), 37: (0, 0, 255), 67: (0, 0, 0), 69: (0, 0, 255)}

_GTN_COMPONENTS_COLORS = {
    0: (220, 220, 220),      # Background (light gray)
    1: (255, 140, 0),        # Network component color 1 (orange)
    5: (0, 0, 255),          # Network component color 2 (blue)
    9: (160, 60, 0),         # Network component color 3 (brown)
    17: (0, 200, 0),         # Network component color 4 (green)
    33: (255, 0, 0),         # Network component color 5 (red)
    65: (255, 255, 0),       # Network component color 6 (yellow)
    80: (255, 0, 255),       # Network component color 7 (magenta)
    103: (0, 0, 0),          # Isolated cores (black)
    129: (255, 255, 255),    # Missing (white)
    150: (150, 150, 150),    # Network component color 8 (gray)
    176: (191, 191, 255),    # Network component color 9 (light blue / lavender)
}

_GTN_IMPORTANCE_COLORS = {
    0: (220, 220, 220),      # Background (light gray)
    103: (0, 0, 0),          # Bridges baseline (black)
    129: (255, 255, 255),    # Missing (white)
    176: (191, 191, 255),    # Cores baseline (light blue / lavender)
    192: (170, 0, 0),        # Top 1% Links (dark red)
    194: (255, 101, 101),    # Top 5% Links (medium red)
    195: (255, 191, 191),    # Top 10% Links (light red)
    208: (0, 101, 0),        # Top 1% Nodes (dark green)
    210: (0, 200, 0),        # Top 5% Nodes (medium green)
    211: (170, 255, 170),    # Top 10% Nodes (light green)
}


def mspa_colormap(transition: bool = True) -> dict[int, tuple[int, int, int, int]]:
    """Colour table of MSPA class maps (:func:`mspa`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Parameters
    ----------
    transition
        Colour bridges and loops that cross an edge or a perforation in their own
        colours (default) or as the class they cross.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.mspa_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    colors = dict(_MSPA_COLORS)
    colors.update(_TRANSITION_ON if transition else _TRANSITION_OFF)
    for value in [v for v in colors if v not in (0, 100, 129, 220)]:
        colors[value + 100] = colors[value]  # internal classes share the colour
    return {i: (*colors.get(i, (0, 0, 0)), 255) for i in range(256)}


def gtn_components_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of network component maps (:func:`gtn_components`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.gtn_components_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*_GTN_COMPONENTS_COLORS.get(i, (0, 0, 0)), 255) for i in range(256)}


def gtn_importance_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of network importance maps (:func:`gtn_importance`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.gtn_importance_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*_GTN_IMPORTANCE_COLORS.get(i, (0, 0, 0)), 255) for i in range(256)}


GTN_COMPONENTS_COLORMAP: dict[int, tuple[int, int, int, int]] = gtn_components_colormap()
GTN_IMPORTANCE_COLORMAP: dict[int, tuple[int, int, int, int]] = gtn_importance_colormap()


def fad_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of density maps of :func:`fad`, :func:`fad_multiscale` and :func:`fos` (6 classes).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.fad_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*FAD_PALETTE[i], 255) for i in range(256)}


def fad5_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of 5-class fragmentation maps (:func:`fos` with ``classes=5``).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.fad5_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*FAD5_PALETTE[i], 255) for i in range(256)}


def dist_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of distance maps (:func:`euclidean_distance`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.dist_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*DIST_PALETTE[i], 255) for i in range(256)}


def entropy_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of spatial entropy maps (:func:`spatial_entropy_map`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.entropy_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*ENTROPY_PALETTE[i], 255) for i in range(256)}


def lm_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of landscape mosaic maps (:func:`landscape_mosaic`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.lm_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*LM_PALETTE[i], 255) for i in range(256)}


def contortion_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of contortion maps (:func:`contortion`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.contortion_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*CONTORTION_PALETTE[i], 255) for i in range(256)}


def resistance_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of resistance and isochrone maps (:func:`distance_to_resistance`, :func:`isochrone_map`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.resistance_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*RESISTANCE_PALETTE[i], 255) for i in range(256)}


def change_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of change maps (:func:`simple_change`, :func:`morphological_change`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.change_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*CHANGE_PALETTE[i], 255) for i in range(256)}


def foschange_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of fragmentation change maps (:func:`fos_change`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.foschange_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*FOSCHANGE_PALETTE[i], 255) for i in range(256)}


def accounting_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of object size-class maps (:func:`accounting`).

    The standard colours of the product, as embedded by :func:`write_raster`.

    Returns
    -------
    dict
        ``{value: (red, green, blue, alpha)}`` for the 256 values 0-255.

    Examples
    --------
    >>> import landschaft as ls
    >>> colours = ls.accounting_colormap()
    >>> len(colours), len(colours[0])
    (256, 4)
    """
    return {i: (*MSPA_ON_PALETTE[i], 255) for i in range(256)}


FAD_COLORMAP: dict[int, tuple[int, int, int, int]] = fad_colormap()
DIST_COLORMAP: dict[int, tuple[int, int, int, int]] = dist_colormap()
ENTROPY_COLORMAP: dict[int, tuple[int, int, int, int]] = entropy_colormap()
LM_COLORMAP: dict[int, tuple[int, int, int, int]] = lm_colormap()
CONTORTION_COLORMAP: dict[int, tuple[int, int, int, int]] = contortion_colormap()
RESISTANCE_COLORMAP: dict[int, tuple[int, int, int, int]] = resistance_colormap()
CHANGE_COLORMAP: dict[int, tuple[int, int, int, int]] = change_colormap()
FOSCHANGE_COLORMAP: dict[int, tuple[int, int, int, int]] = foschange_colormap()
ACCOUNTING_COLORMAP: dict[int, tuple[int, int, int, int]] = accounting_colormap()


def _resolve_colormap(colormap, data) -> dict | None:
    if colormap is None or colormap is False:
        return None
    if colormap == "auto":
        attrs = getattr(data, "attrs", {})
        product = attrs.get("landschaft_product")
        if product == "MSPA":
            return mspa_colormap(bool(attrs.get("mspa_transition", 1)))
        if product == "GTN_COMPONENTS":
            return gtn_components_colormap()
        if product == "GTN_IMPORTANCE":
            return gtn_importance_colormap()
        if product in ("FAD", "FAD_MULTISCALE"):
            return fad_colormap()
        if product == "FOS":
            return fad5_colormap() if attrs.get("fos_classes") == 5 else fad_colormap()
        if product == "EUCLIDEAN_DISTANCE":
            return dist_colormap()
        if product in ("SPATIAL_ENTROPY", "CONTAGION", "CLUSTERING", "ISOCHRONE_A", "ISOCHRONE_AB"):
            return entropy_colormap()
        if product == "LANDSCAPE_MOSAIC":
            return lm_colormap()
        if product == "CONTORTION":
            return contortion_colormap()
        if product in ("RESISTANCE", "COST_DISTANCE", "LEAST_COST_PATH"):
            return resistance_colormap()
        if product in ("SIMPLE_CHANGE", "MORPHOLOGICAL_CHANGE"):
            return change_colormap()
        if product == "FOS_CHANGE":
            return foschange_colormap()
        if product in ("ACCOUNTING", "INFLUENCE_ZONES", "PROXIMITY", "SPA2", "SPA3", "SPA5", "SPA6"):
            return accounting_colormap()  # GTB's mspacolorston table
        return None
    if colormap == "mspa":
        return mspa_colormap(True)
    if colormap == "gtn_components":
        return gtn_components_colormap()
    if colormap == "contortion":
        return contortion_colormap()
    if colormap == "resistance":
        return resistance_colormap()
    if colormap in ("change", "simple_change", "mcd"):
        return change_colormap()
    if colormap in ("foschange", "fos_change"):
        return foschange_colormap()
    if colormap == "gtn_importance":
        return gtn_importance_colormap()
    if colormap == "fad":
        return fad_colormap()
    if colormap in ("fad5", "fos5"):
        return fad5_colormap()
    if colormap == "dist":
        return dist_colormap()
    if colormap in ("entropy", "contagion", "clustering"):
        return entropy_colormap()
    if colormap == "lm":
        return lm_colormap()
    if colormap == "accounting":
        return accounting_colormap()
    return dict(colormap)


def _product_description(attrs: dict) -> str | None:
    """The TIFF description of a landschaft product (GuidosToolbox's wording)."""
    product = attrs.get("landschaft_product")
    if product == "MSPA":
        return (
            "GTB_MSPA (landschaft), MSPA parameters (FGconn_EdgeWidth_Transition_Intext): "
            f"{attrs['mspa_connectivity']}_{attrs['mspa_edge_width']}_"
            f"{attrs['mspa_transition']}_{attrs['mspa_internal']}"
        )
    if product in ("GTN_COMPONENTS", "GTN_COMPONENTS_DATA"):
        return (
            f"GTB_NW (landschaft), components: {attrs.get('num_components')}, "
            f"ECA: {attrs.get('eca', 0):.2f}, DOC: {attrs.get('doc', 0):.2f}%"
        )
    if product in ("GTN_IMPORTANCE", "GTN_IMPORTANCE_MAP"):
        return (
            f"GTB_CS (landschaft), Link_max: {attrs.get('max_link_dpc', 0):.2f}, "
            f"Node_max: {attrs.get('max_node_dpc', 0):.2f}"
        )
    if product in ("FAD", "FAD_MULTISCALE"):
        return (
            f"GTB_FAD (landschaft), window_size: {attrs.get('window_size', 'multi')}, "
            f"avg_fad: {attrs.get('average_fad', 0):.2f}%"
        )
    if product in ("EUCLIDEAN_DISTANCE", "EUCLIDEAN_DISTANCE_FLOAT"):
        return (
            f"GTB_EUCL (landschaft), mean_fg: {attrs.get('mean_fg_dist', 0):.2f}, "
            f"max_fg: {attrs.get('max_fg_dist', 0):.2f}"
        )
    if product == "SPATIAL_ENTROPY":
        return f"GTB_ENT (landschaft), window_size: {attrs.get('window_size', 'full')}"
    if product == "SIMPLE_CHANGE":
        return (
            f"GTB_CHANGE (landschaft), Gain: {attrs.get('gain', 0)}, "
            f"Loss: {attrs.get('loss', 0)}, Stable_FG: {attrs.get('stable_fg', 0)}"
        )
    if product == "MORPHOLOGICAL_CHANGE":
        return (
            f"GTB_MCD (landschaft), Gain: {attrs.get('mcd_gain', 0)}, "
            f"Loss: {attrs.get('mcd_loss', 0)}, Elasticity: {attrs.get('elasticity', 0):.4f}"
        )
    if product == "FOS_CHANGE":
        return (
            f"GTB_FOSCHANGE (landschaft), FragDec: {attrs.get('frag_decrease_pct', 0):.2f}%, "
            f"FragInc: {attrs.get('frag_increase_pct', 0):.2f}%"
        )
    if product == "ACCOUNTING":
        return (
            f"GTB_ACC (landschaft), {attrs.get('connectivity')}-conn. objects: "
            f"{attrs.get('n_objects')}, size classes: {attrs.get('thresholds')}"
        )
    if product == "GRAY_SPATIAL_CONVOLUTION":
        return (
            f"GTB_GSC (landschaft), metric: {attrs.get('spatial_metric', 'mean')}, "
            f"window_size: {attrs.get('window_size', 7)}"
        )
    return None


# drivers whose files are written whole at the end (no random access while writing)
_DRIVER_BY_SUFFIX = {".tif": "GTiff", ".tiff": "GTiff", ".asc": "AAIGrid", ".img": "HFA", ".nc": "netCDF",
                     ".vrt": "VRT", ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".bil": "EHdr",
                     ".rst": "RST", ".sdat": "SAGA", ".grd": "GSBG", ".kea": "KEA"}


def _driver_for(path: Path, driver: str | None) -> str:
    if driver:
        return driver
    suffix = path.suffix.lower()
    if suffix in _DRIVER_BY_SUFFIX:
        return _DRIVER_BY_SUFFIX[suffix]
    try:
        from rasterio.drivers import driver_from_extension

        return driver_from_extension(path)
    except (ValueError, ImportError):
        raise ValueError(f"cannot tell the raster format of {path.name!r}; pass driver=") from None


def _value_range(values) -> tuple[float, float]:
    if hasattr(values, "dask"):
        import dask

        lo, hi = dask.compute(values.min(), values.max())
        return float(lo), float(hi)
    return (float(values.min()), float(values.max())) if values.size else (0.0, 0.0)


def _output_dtype(values, dtype) -> np.dtype:
    """The type to write: the one asked for, else the data's, made storable."""
    if dtype is not None:
        return np.dtype(dtype)
    dt = np.dtype(values.dtype)
    if dt == np.bool_:
        return np.dtype(np.uint8)
    if dt == np.float16:
        return np.dtype(np.float32)
    if dt in (np.int64, np.uint64):  # most GIS software cannot read 64-bit integers
        lo, hi = _value_range(values)
        for small in ((np.uint8, np.uint16, np.uint32) if lo >= 0 else (np.int16, np.int32)):
            info = np.iinfo(small)
            if info.min <= lo and hi <= info.max:
                return np.dtype(small)
        return dt
    return dt


def _default_nodata(dt: np.dtype):
    """A NoData value for integer output that has to mark missing cells."""
    info = np.iinfo(dt)
    return info.max if info.min == 0 else info.min


def write_raster(
    data,
    path: PathLike,
    *,
    like=None,
    crs=None,
    transform=None,
    nodata=None,
    dtype=None,
    band_names=None,
    colormap: str | dict | None = "auto",
    compress: str = "lzw",
    description: str | None = None,
    driver: str | None = None,
) -> Path:
    """Write a raster: one band or several, any format GDAL writes.

    One function for every raster landschaft writes. It keeps what the data
    says about itself (georeferencing, NoData, band names, the colours and
    parameters of landschaft's products) and stores it so that GIS software
    and :func:`load_raster` read it back as it was:

    - floating-point data with NaN get NaN as NoData (or ``nodata``, which then
      replaces the NaN); integers keep the ``.rio.nodata`` of the data;
    - ``bool`` is written as ``uint8`` and 64-bit integers as the smallest
      integer type that holds them;
    - GeoTIFFs are compressed with the predictor that suits the type, tiled when
      larger than 512 x 512 and BigTIFF when needed; the format follows the file
      extension (``.tif``, ``.asc``, ``.img``, ``.nc``...) or ``driver``
      (``"COG"``: Cloud-Optimised GeoTIFF);
    - dask arrays are written block by block, never whole in memory;
    - missing folders of ``path`` are created.

    Parameters
    ----------
    data
        2-D ``(y, x)`` or 3-D ``(band, y, x)`` ``xarray.DataArray``
        (georeferencing taken from ``.rio``), numpy or dask array.
    path
        Output file.
    like
        A raster (path or DataArray) on the same grid whose CRS and transform
        georeference ``data``, e.g. a numpy result computed from an input.
    crs, transform
        Coordinate reference system and affine transform; they override those of
        ``data`` and ``like``.
    nodata
        NoData value; overrides the one of ``data``. NaN in the data are written
        as this value.
    dtype
        Data type to write (default: the data's, see above). Floats cast to
        integers get ``nodata`` (or the type's maximum, minimum if signed) where
        they are NaN.
    band_names
        Band descriptions (default: a ``band`` coordinate of strings, e.g. from
        :func:`load_raster`).
    colormap
        ``"auto"`` (default) embeds the colour table of the product when ``data``
        is a class map made by landschaft (MSPA, fragmentation, change...),
        ``"mspa"`` forces the MSPA colours, a ``{value: (r, g, b[, a])}`` mapping
        embeds your own and ``None`` none. One-band ``uint8``/``uint16`` only.
    compress
        GeoTIFF compression: ``"lzw"`` (default), ``"deflate"``, ``"zstd"``,
        ``None``...
    description
        TIFF image description (MSPA results record their parameters there).
    driver
        GDAL format (default: from the extension of ``path``).

    Returns
    -------
    pathlib.Path
        The written file.

    Examples
    --------
    A landschaft result keeps its colours and parameters:

    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> ls.write_raster(ls.mspa(forest), "forest_mspa.tif").name
    'forest_mspa.tif'

    A numpy array georeferenced like the input, with your own colours:

    >>> import numpy as np
    >>> mask = (forest.values == 2).astype("uint8")
    >>> path = ls.write_raster(mask, "mask.tif", like=forest,
    ...                        colormap={0: (255, 255, 255), 1: (21, 128, 61)})

    Floats with NaN: NaN is the NoData, or the value you choose:

    >>> share = forest.where(forest > 0) / 2.0
    >>> float(ls.load_raster(ls.write_raster(share, "share.tif")).rio.nodata)
    nan
    >>> ls.write_raster(share, "share.asc", nodata=-9999).name
    'share.asc'

    Several bands, named:

    >>> two = ls.write_raster(np.stack([mask, 1 - mask]), "stacks/two.tif", like=forest, band_names=["forest", "open"])
    >>> ls.load_raster(two).band.values.tolist()
    ['forest', 'open']
    """
    path = Path(path)
    is_da = isinstance(data, xr.DataArray)
    values = data.data if is_da else (data if hasattr(data, "dask") else np.asarray(data))
    if values.ndim not in (2, 3):
        raise ValueError(f"write_raster expects a 2-D (y, x) or 3-D (band, y, x) array, got shape {values.shape}")
    bands = values.shape[0] if values.ndim == 3 else 1
    height, width = values.shape[-2:]

    # georeferencing: like, then the data, then the arguments
    like_crs = like_transform = None
    if like is not None:
        ref = like
        if isinstance(like, (str, os.PathLike)):
            with rasterio.open(like) as src:
                like_crs, like_transform, shape = src.crs, src.transform, (src.height, src.width)
        else:
            like_crs, like_transform, shape = ref.rio.crs, ref.rio.transform(), ref.shape[-2:]
        if tuple(shape) != (height, width):
            raise ValueError(f"data has {height} x {width} cells, like has {shape[0]} x {shape[1]}")
    tags: dict[str, str] = {}
    data_crs = data_transform = data_nodata = None
    if is_da:
        data_crs = data.rio.crs
        if "x" in data.dims and "y" in data.dims and data.sizes["x"] > 1 and data.sizes["y"] > 1:
            data_transform = data.rio.transform()
        data_nodata = data.rio.nodata
        tags = {k: str(v) for k, v in data.attrs.items() if k.startswith(("mspa_", "landschaft_"))}
        if description is None:
            description = _product_description(data.attrs)
        if band_names is None and values.ndim == 3:
            coord = data[data.dims[0]].values if data.dims[0] in data.coords else None
            if coord is not None and all(isinstance(v, str) for v in coord.tolist()):
                band_names = [str(v) for v in coord.tolist()]
        if band_names is None and values.ndim == 2 and isinstance(data.attrs.get("long_name"), str):
            band_names = [data.attrs["long_name"]]
    crs = crs if crs is not None else (data_crs if data_crs is not None else like_crs)
    transform = transform if transform is not None else (data_transform if data_transform is not None
                                                         else like_transform)
    if band_names is not None:
        band_names = [str(v) for v in band_names]
        if len(band_names) != bands:
            raise ValueError(f"band_names: {len(band_names)} names for {bands} bands")

    # type and NoData
    out_dtype = _output_dtype(values, dtype)
    src_float = np.issubdtype(values.dtype, np.floating)
    out_float = np.issubdtype(out_dtype, np.floating)
    if nodata is None:
        nodata = data_nodata
    if nodata is not None and not out_float and np.isnan(nodata):
        nodata = None
    if nodata is None and src_float:
        has_nan = bool(np.isnan(values).any().compute() if hasattr(values, "dask") else np.isnan(values).any())
        if has_nan:
            nodata = np.nan if out_float else _default_nodata(out_dtype)
    fill = None if nodata is None or not src_float or (out_float and np.isnan(nodata)) else nodata

    def prepared(block: np.ndarray) -> np.ndarray:
        block = np.asarray(block)
        if fill is not None:
            block = np.where(np.isnan(block), fill, block)
        return block.astype(out_dtype, copy=False)

    # format
    drv = _driver_for(path, driver)
    profile: dict[str, Any] = {
        "driver": drv,
        "height": height,
        "width": width,
        "count": bands,
        "dtype": out_dtype.name,
        "crs": crs,
        "transform": transform if transform is not None else rasterio.transform.Affine.identity(),
    }
    if drv in ("GTiff", "COG"):
        profile["BIGTIFF"] = "IF_SAFER"
        if compress:
            profile["compress"] = compress
            if str(compress).lower() in ("lzw", "deflate", "zstd", "lzma"):
                profile["predictor"] = 3 if out_float else 2
        if drv == "GTiff" and height * width >= 512 * 512:
            profile.update(tiled=True, blockxsize=512, blockysize=512)
        if drv == "COG":
            profile["blocksize"] = 512
    if nodata is not None:
        profile["nodata"] = nodata

    cmap = _resolve_colormap(colormap, data)
    if cmap is not None and (bands != 1 or out_dtype not in (np.uint8, np.uint16)):
        if colormap == "auto":
            cmap = None
        else:
            raise ValueError("colour tables require a one-band uint8 or uint16 raster")

    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        if crs is None and transform is None:
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(path, "w", **profile) as dst:
            if hasattr(values, "dask"):
                rows = values.chunks[-2]
                y0 = 0
                for nr in rows:
                    block = prepared(values[..., y0:y0 + nr, :].compute())
                    window = rasterio.windows.Window(0, y0, width, nr)
                    dst.write(block if block.ndim == 3 else block[None], window=window)
                    y0 += nr
            else:
                block = prepared(values)
                dst.write(block if block.ndim == 3 else block[None])
            if cmap is not None:
                dst.write_colormap(1, cmap)
            if band_names:
                for i, name in enumerate(band_names, 1):
                    dst.set_band_description(i, name)
            if description:
                tags["TIFFTAG_IMAGEDESCRIPTION"] = description
            if tags:
                dst.update_tags(**tags)
    return path
