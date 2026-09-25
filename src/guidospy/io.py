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

from .validation import BACKGROUND, FOREGROUND, MISSING, InvalidInputError, validate_mspa_input

__all__ = ["load_raster", "write_raster", "reclassify"]

PathLike = str | os.PathLike


def load_raster(
    source: PathLike,
    *,
    band: int | None = None,
    chunks: Any = None,
    validate: str | None = "mspa",
) -> xr.DataArray:
    """Read a single-band raster as a 2-D ``(y, x)`` DataArray and check it.

    Parameters
    ----------
    source
        Path (or GDAL URL) of the raster, e.g. a GeoTIFF.
    band
        Band number (1-based) to read from a multi-band file. By default the
        file must have exactly one band.
    chunks
        ``None`` (default) loads the data into memory. Anything else (e.g.
        ``"auto"`` or ``{"x": 4096, "y": 4096}``) returns a lazy dask-backed
        array; note that MSPA itself always needs the whole map in memory.
    validate
        ``"mspa"`` (default) checks that the map is an MSPA input:
        ``uint8`` coded 0 = missing, 1 = background, 2 = foreground, with some
        foreground and some background. ``None`` skips the checks.

    Returns
    -------
    xarray.DataArray
        Georeferenced array (CRS and transform available via ``.rio``).

    Raises
    ------
    InvalidInputError
        If the file does not satisfy the requested checks.
    """
    if validate not in (None, "mspa"):
        raise ValueError(f"validate must be 'mspa' or None, got {validate!r}")

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


def mspa_colormap(transition: bool = True) -> dict[int, tuple[int, int, int, int]]:
    """Colour table of GuidosToolbox for MSPA maps (256 entries, RGBA)."""
    colors = dict(_MSPA_COLORS)
    colors.update(_TRANSITION_ON if transition else _TRANSITION_OFF)
    for value in [v for v in colors if v not in (0, 100, 129, 220)]:
        colors[value + 100] = colors[value]  # internal classes share the colour
    return {i: (*colors.get(i, (0, 0, 0)), 255) for i in range(256)}


def _resolve_colormap(colormap, data) -> dict | None:
    if colormap is None or colormap is False:
        return None
    if colormap == "auto":
        attrs = getattr(data, "attrs", {})
        if attrs.get("guidospy_product") == "MSPA":
            return mspa_colormap(bool(attrs.get("mspa_transition", 1)))
        return None
    if colormap == "mspa":
        return mspa_colormap(True)
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
        tags = {k: str(v) for k, v in data.attrs.items() if k.startswith(("mspa_", "guidospy_"))}
        if description is None and data.attrs.get("guidospy_product") == "MSPA":
            description = (
                "GTB_MSPA (guidospy), MSPA parameters (FGconn_EdgeWidth_Transition_Intext): "
                f"{data.attrs['mspa_connectivity']}_{data.attrs['mspa_edge_width']}_"
                f"{data.attrs['mspa_transition']}_{data.attrs['mspa_internal']}"
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
