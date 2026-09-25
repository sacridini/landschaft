# SPDX-License-Identifier: GPL-3.0-or-later
"""Raster input/output built on rasterio and rioxarray."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from ._palettes import (
    CHANGE_PALETTE,
    CONTORTION_PALETTE,
    DIST_PALETTE,
    ENTROPY_PALETTE,
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
    band: int | None = None,
    chunks: Any = None,
    validate: str | None = "mspa",
    stack: bool = False,
    errors: str = "raise",
    recursive: bool = False,
):
    """Read one raster, or collect many, checking them for MSPA.

    Parameters
    ----------
    source
        One raster (path or GDAL URL), a directory (its ``.tif``/``.tiff``
        files), a glob pattern (``"tiles/*.tif"``) or a list of any of these.
    band
        Band number (1-based) to read from multi-band files. By default each
        file must have exactly one band.
    chunks
        Single raster only: ``None`` (default) loads the data into memory;
        anything else (e.g. ``"auto"``) returns a lazy dask-backed array.
        MSPA itself always needs the whole map in memory.
    validate
        ``"mspa"`` (default) checks that the map is an MSPA input: ``uint8``
        coded 0 = missing, 1 = background, 2 = foreground, with some
        foreground and some background. ``None`` skips the checks.
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
        For one raster, a georeferenced ``(y, x)`` array (CRS and transform via
        ``.rio``). For several, a :class:`RasterCollection` that reads only the
        file headers now; pass it to :func:`landschaft.mspa` to process every
        raster in parallel. With ``stack=True``, a lazy DataArray.

    Raises
    ------
    InvalidInputError
        If a file does not satisfy the requested checks.
    """
    from .collection import RasterCollection, is_single_source

    if validate not in (None, "mspa"):
        raise ValueError(f"validate must be 'mspa' or None, got {validate!r}")
    if not is_single_source(source):
        collection = RasterCollection.from_sources(
            source, band=band, validate=validate, errors=errors, recursive=recursive
        )
        return collection.to_stack() if stack else collection
    return _load_single(source, band=band, chunks=chunks, validate=validate)


def _load_single(source: PathLike, *, band: int | None, chunks: Any, validate: str | None) -> xr.DataArray:
    raw = rioxarray.open_rasterio(source, chunks=chunks, masked=False)
    if isinstance(raw, (list, xr.Dataset)):
        raise InvalidInputError(f"{source}: file contains several sub-datasets; open one of them explicitly.")
    try:
        n_bands = raw.sizes.get("band", 1)
        if band is None:
            if n_bands != 1:
                raise InvalidInputError(
                    f"{source}: input map has {n_bands} bands, a single band is required. "
                    "Select one with load_raster(..., band=N)."
                )
            da = raw.isel(band=0, drop=True)
        else:
            if band not in raw["band"].values:
                raise InvalidInputError(f"{source}: band {band} not found (file has {n_bands} bands).")
            da = raw.sel(band=band, drop=True)
        if chunks is None:
            da = da.load()
    finally:
        if chunks is None:
            raw.close()  # release the file: on Windows an open handle locks it
    da.name = Path(str(source)).stem
    da.attrs["source"] = str(source)

    if validate == "mspa":
        try:
            validate_mspa_input(da)
        except InvalidInputError as err:
            raise InvalidInputError(f"{source}: {err}") from None
    return da


def reclassify(
    data,
    foreground: Iterable[int] | int | Callable[[np.ndarray], np.ndarray],
    *,
    missing: Iterable[int] | int | None = None,
    background: Iterable[int] | int | None = None,
):
    """Build an MSPA input (0 = missing, 1 = background, 2 = foreground).

    Parameters
    ----------
    data
        Categorical map (numpy array or DataArray, any integer/float dtype).
    foreground
        Class value(s) that become foreground, or a function returning a
        boolean mask (e.g. ``lambda a: a >= 0.3`` on a tree-cover fraction).
    missing
        Class value(s) that become missing. NaN values and the raster's
        nodata value (``data.rio.nodata``) are always treated as missing.
    background
        Class value(s) that become background. By default every value that is
        neither foreground nor missing; if given, remaining values are missing.

    Returns
    -------
    Same type as ``data``, with dtype ``uint8``.
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
    """Colour table of GuidosToolbox for MSPA maps (256 entries, RGBA)."""
    colors = dict(_MSPA_COLORS)
    colors.update(_TRANSITION_ON if transition else _TRANSITION_OFF)
    for value in [v for v in colors if v not in (0, 100, 129, 220)]:
        colors[value + 100] = colors[value]  # internal classes share the colour
    return {i: (*colors.get(i, (0, 0, 0)), 255) for i in range(256)}


def gtn_components_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for GTN Components maps (256 entries, RGBA)."""
    return {i: (*_GTN_COMPONENTS_COLORS.get(i, (0, 0, 0)), 255) for i in range(256)}


def gtn_importance_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for GTN Importance maps (256 entries, RGBA)."""
    return {i: (*_GTN_IMPORTANCE_COLORS.get(i, (0, 0, 0)), 255) for i in range(256)}


GTN_COMPONENTS_COLORMAP: dict[int, tuple[int, int, int, int]] = gtn_components_colormap()
GTN_IMPORTANCE_COLORMAP: dict[int, tuple[int, int, int, int]] = gtn_importance_colormap()


def fad_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for FAD maps (256 entries, RGBA)."""
    return {i: (*FAD_PALETTE[i], 255) for i in range(256)}


def dist_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for Euclidean Distance maps (256 entries, RGBA)."""
    return {i: (*DIST_PALETTE[i], 255) for i in range(256)}


def entropy_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for Spatial Entropy maps (256 entries, RGBA)."""
    return {i: (*ENTROPY_PALETTE[i], 255) for i in range(256)}


def lm_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for Landscape Mosaic maps (256 entries, RGBA)."""
    return {i: (*LM_PALETTE[i], 255) for i in range(256)}


def contortion_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for Contortion maps (256 entries, RGBA)."""
    return {i: (*CONTORTION_PALETTE[i], 255) for i in range(256)}


def resistance_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for Resistance / Isochrone maps (256 entries, RGBA)."""
    return {i: (*RESISTANCE_PALETTE[i], 255) for i in range(256)}


def change_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for Simple Change and MCD maps (256 entries, RGBA)."""
    return {i: (*CHANGE_PALETTE[i], 255) for i in range(256)}


def foschange_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for FOS Change maps (256 entries, RGBA)."""
    return {i: (*FOSCHANGE_PALETTE[i], 255) for i in range(256)}


def accounting_colormap() -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for Accounting maps (256 entries, RGBA)."""
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
        if product == "EUCLIDEAN_DISTANCE":
            return dist_colormap()
        if product in ("SPATIAL_ENTROPY", "CONTAGION", "CLUSTERING"):
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
        if product == "ACCOUNTING":
            return accounting_colormap()
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
    if colormap == "dist":
        return dist_colormap()
    if colormap in ("entropy", "contagion", "clustering"):
        return entropy_colormap()
    if colormap == "lm":
        return lm_colormap()
    if colormap == "accounting":
        return accounting_colormap()
    return dict(colormap)


def write_raster(
    data,
    path: PathLike,
    *,
    crs=None,
    transform=None,
    nodata=None,
    colormap: str | dict | None = "auto",
    compress: str = "lzw",
    description: str | None = None,
) -> Path:
    """Write a 2-D array to a GeoTIFF.

    Parameters
    ----------
    data
        ``xarray.DataArray`` (georeferencing taken from ``.rio``) or numpy
        array (then pass ``crs`` and ``transform``). Dask-backed arrays are
        computed.
    path
        Output file.
    crs, transform, nodata
        Override/define the georeferencing and nodata value.
    colormap
        ``"auto"`` (default) writes the GuidosToolbox colour table for MSPA
        results, ``"mspa"`` forces it, a ``{value: (r, g, b[, a])}`` mapping
        writes a custom one and ``None`` writes none.
    compress
        GDAL compression (``"lzw"`` as GuidosToolbox, ``"deflate"``, ``"zstd"``...).
    description
        TIFF image description. MSPA results get the processing parameters.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    arr = np.asarray(data.values if isinstance(data, xr.DataArray) else data)
    if arr.ndim != 2:
        raise ValueError(f"write_raster expects a 2-D array, got shape {arr.shape}")

    tags: dict[str, str] = {}
    if isinstance(data, xr.DataArray):
        crs = crs if crs is not None else data.rio.crs
        if transform is None and "x" in data.dims and "y" in data.dims and data.sizes["x"] > 1 and data.sizes["y"] > 1:
            transform = data.rio.transform()
        if nodata is None:
            nodata = data.rio.nodata
        tags = {k: str(v) for k, v in data.attrs.items() if k.startswith(("mspa_", "landschaft_"))}
        if description is None and data.attrs.get("landschaft_product") == "MSPA":
            description = (
                "GTB_MSPA (landschaft), MSPA parameters (FGconn_EdgeWidth_Transition_Intext): "
                f"{data.attrs['mspa_connectivity']}_{data.attrs['mspa_edge_width']}_"
                f"{data.attrs['mspa_transition']}_{data.attrs['mspa_internal']}"
            )
        elif description is None and data.attrs.get("landschaft_product") in ("GTN_COMPONENTS", "GTN_COMPONENTS_DATA"):
            description = (
                f"GTB_NW (landschaft), components: {data.attrs.get('num_components')}, "
                f"ECA: {data.attrs.get('eca', 0):.2f}, DOC: {data.attrs.get('doc', 0):.2f}%"
            )
        elif description is None and data.attrs.get("landschaft_product") in ("GTN_IMPORTANCE", "GTN_IMPORTANCE_MAP"):
            description = (
                f"GTB_CS (landschaft), Link_max: {data.attrs.get('max_link_dpc', 0):.2f}, "
                f"Node_max: {data.attrs.get('max_node_dpc', 0):.2f}"
            )
        elif description is None and data.attrs.get("landschaft_product") in ("FAD", "FAD_MULTISCALE"):
            description = (
                f"GTB_FAD (landschaft), window_size: {data.attrs.get('window_size', 'multi')}, "
                f"avg_fad: {data.attrs.get('average_fad', 0):.2f}%"
            )
        elif description is None and data.attrs.get("landschaft_product") in ("EUCLIDEAN_DISTANCE", "EUCLIDEAN_DISTANCE_FLOAT"):
            description = (
                f"GTB_EUCL (landschaft), mean_fg: {data.attrs.get('mean_fg_dist', 0):.2f}, "
                f"max_fg: {data.attrs.get('max_fg_dist', 0):.2f}"
            )
        elif description is None and data.attrs.get("landschaft_product") == "SPATIAL_ENTROPY":
            description = f"GTB_ENT (landschaft), window_size: {data.attrs.get('window_size', 'full')}"
        elif description is None and data.attrs.get("landschaft_product") == "SIMPLE_CHANGE":
            description = (
                f"GTB_CHANGE (landschaft), Gain: {data.attrs.get('gain', 0)}, "
                f"Loss: {data.attrs.get('loss', 0)}, Stable_FG: {data.attrs.get('stable_fg', 0)}"
            )
        elif description is None and data.attrs.get("landschaft_product") == "MORPHOLOGICAL_CHANGE":
            description = (
                f"GTB_MCD (landschaft), Gain: {data.attrs.get('mcd_gain', 0)}, "
                f"Loss: {data.attrs.get('mcd_loss', 0)}, Elasticity: {data.attrs.get('elasticity', 0):.4f}"
            )
        elif description is None and data.attrs.get("landschaft_product") == "FOS_CHANGE":
            description = (
                f"GTB_FOSCHANGE (landschaft), FragDec: {data.attrs.get('frag_decrease_pct', 0):.2f}%, "
                f"FragInc: {data.attrs.get('frag_increase_pct', 0):.2f}%"
            )
        elif description is None and data.attrs.get("landschaft_product") == "ACCOUNTING":
            description = (
                f"GTB_ACC (landschaft), {data.attrs.get('connectivity')}-conn. objects: "
                f"{data.attrs.get('n_objects')}, size classes: {data.attrs.get('thresholds')}"
            )
        elif description is None and data.attrs.get("landschaft_product") == "GRAY_SPATIAL_CONVOLUTION":
            description = (
                f"GTB_GSC (landschaft), metric: {data.attrs.get('spatial_metric', 'mean')}, "
                f"window_size: {data.attrs.get('window_size', 7)}"
            )

    height, width = arr.shape
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": arr.dtype,
        "crs": crs,
        "transform": transform if transform is not None else rasterio.transform.Affine.identity(),
        "compress": compress,
        "BIGTIFF": "IF_SAFER",
    }
    if height * width >= 512 * 512:
        profile.update(tiled=True, blockxsize=512, blockysize=512)
    if nodata is not None:
        profile["nodata"] = nodata

    cmap = _resolve_colormap(colormap, data)
    if cmap is not None and arr.dtype not in (np.uint8, np.uint16):
        raise ValueError("colour tables require a uint8 or uint16 raster")

    path = Path(path)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
            if cmap is not None:
                dst.write_colormap(1, cmap)
            if description:
                tags["TIFFTAG_IMAGEDESCRIPTION"] = description
            if tags:
                dst.update_tags(**tags)
    return path
