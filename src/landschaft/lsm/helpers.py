# SPDX-License-Identifier: GPL-3.0-or-later
"""landscapemetrics' public helpers: ``construct_buffer``, ``pad_raster``,
``unpad_raster``, ``raster_to_points`` and ``get_unique_values``."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .sampling import terra_grid

__all__ = ["construct_buffer", "get_unique_values", "pad_raster", "raster_to_points", "unpad_raster"]


# -- construct_buffer -------------------------------------------------------------


def construct_buffer(coords, shape: str, size: float, *, return_vec: bool = True, crs=None):
    """Square or circular sample plots around points, like landscapemetrics'
    ``construct_buffer``.

    Parameters
    ----------
    coords
        Points: an ``(n, 2)`` array of x/y coordinates or point geometries.
    shape
        ``"square"`` (half side ``size``) or ``"circle"`` (radius ``size``,
        100 vertices).
    size
        Buffer size, in map units.
    return_vec
        True: a ``geopandas.GeoDataFrame`` of polygons with ``plot_id``
        1..n (needs geopandas, the ``vector`` extra). False: landscapemetrics'
        vertex matrix, an ``(n * k, 3)`` array of ``x``, ``y`` and the plot
        number in the same row order (circles plot by plot, squares corner by
        corner), open rings.
    crs
        CRS of the polygons (``return_vec`` only; default: the CRS of
        ``coords`` if it has one).

    Notes
    -----
    These are the plots :func:`sample_metrics` builds around points.

    Examples
    --------
    >>> import landschaft as ls
    >>> ls.construct_buffer([[0.0, 0.0]], "square", 10, return_vec=False)
    array([[-10., -10.,   1.],
           [-10.,  10.,   1.],
           [ 10.,  10.,   1.],
           [ 10., -10.,   1.]])
    >>> plots = ls.construct_buffer([[0.0, 0.0], [100.0, 0.0]], "circle", 25)   # GeoDataFrame
    >>> len(plots)
    2
    """
    from .sampling import _CIRCLE_COS, _CIRCLE_SIN
    from .utils import point_coords

    xy = point_coords(coords)
    size = float(size)
    if shape not in ("square", "circle"):
        raise ValueError(f"shape must be 'square' or 'circle', got {shape!r}")
    if return_vec:
        try:
            import geopandas as gpd
            import shapely
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("construct_buffer(return_vec=True) needs geopandas (pip install landschaft[vector]); "
                              "use return_vec=False for the vertex matrix") from exc
        if crs is None:
            crs = getattr(coords, "crs", None)
        vertices = construct_buffer(xy, shape, size, return_vec=False)
        polys = []
        for k in range(len(xy)):
            ring = vertices[vertices[:, 2] == k + 1, :2]
            if not np.array_equal(ring[0], ring[-1]):  # terra closes open rings
                ring = np.vstack([ring, ring[:1]])
            polys.append(shapely.Polygon(ring))
        return gpd.GeoDataFrame({"plot_id": np.arange(1, len(xy) + 1)}, geometry=polys, crs=crs)
    n = len(xy)
    if shape == "circle":
        out = np.empty((n * 100, 3))
        out[:, 0] = (_CIRCLE_SIN[:, None] * size + xy[:, 0]).T.ravel()
        out[:, 1] = (_CIRCLE_COS[:, None] * size + xy[:, 1]).T.ravel()
        out[:, 2] = np.repeat(np.arange(1, n + 1), 100)
    else:
        # R: c(x - s, x - s, x + s, x + s) per corner, the points inside each corner
        out = np.empty((n * 4, 3))
        out[:, 0] = np.concatenate([xy[:, 0] - size, xy[:, 0] - size, xy[:, 0] + size, xy[:, 0] + size])
        out[:, 1] = np.concatenate([xy[:, 1] - size, xy[:, 1] + size, xy[:, 1] + size, xy[:, 1] - size])
        out[:, 2] = np.tile(np.arange(1, n + 1), 4)
    return out


# -- raster values ----------------------------------------------------------------------


def _layers(data):
    from .landscape import as_layers

    layers = as_layers(data)
    return (layers, False) if layers is not None else ([data], True)


def _raw(data, nodata, band):
    """Values (float64, NaN = NA), terra's extent (xmin, xmax, ymin, ymax),
    the CRS, and whether the input is georeferenced (a file or DataArray)."""
    from .landscape import _read
    from .utils import raster_extent

    if isinstance(data, (str, os.PathLike)):
        data = _read(data, band)
    crs = None
    if isinstance(data, xr.DataArray):
        da = data
        if "band" in da.dims:
            if da.sizes["band"] != 1 and band is None:
                raise ValueError("the raster has several bands; select one with band=")
            da = da.sel(band=band) if band is not None else da.isel(band=0)
        if da.ndim != 2:
            raise ValueError(f"expected a 2-D raster, got dimensions {da.dims}")
        if nodata is None:
            nodata = da.rio.nodata
        values = np.array(da.values, dtype=np.float64)
        try:
            transform = da.rio.transform()
            crs = da.rio.crs
            rx, ry = abs(transform.a), abs(transform.e)
        except Exception:
            transform, rx, ry = None, 1.0, 1.0
        extent = raster_extent(transform, values.shape, (rx, ry))
        georef = transform is not None
    else:
        if np.ma.isMaskedArray(data):
            values = np.ma.getdata(data).astype(np.float64)
            values[np.ma.getmaskarray(data)] = np.nan
        else:
            values = np.array(data, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"expected a 2-D raster, got shape {values.shape}")
        extent = raster_extent(None, values.shape, (1.0, 1.0))
        georef = False
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        values[values == nodata] = np.nan
    return values, extent, crs, georef




# -- pad_raster / unpad_raster -----------------------------------------------------------


def pad_raster(data, pad_raster_value: float = -999, pad_raster_cells: int = 1, global_: bool = False,
               return_raster: bool = True, *, nodata=None, band: int | None = None):
    """Add rows and columns of ``pad_raster_value`` around a raster, like
    landscapemetrics' ``pad_raster``.

    Parameters
    ----------
    data
        A raster (file, DataArray, numpy or masked array) or a list of them.
    pad_raster_value
        Value of the new cells.
    pad_raster_cells
        Number of rows / columns added on each side.
    global_
        Also replace the NA cells by ``pad_raster_value`` (R's ``global``).
    return_raster
        Return a georeferenced raster (DataArray, NaN = NA, extent grown by
        the new cells) for georeferenced inputs; numpy input always gives a
        numpy array.
    nodata, band
        Reading options, as in :class:`~landschaft.Landscape`.

    Returns
    -------
    xarray.DataArray or numpy.ndarray (float), or a list for several rasters.

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> ls.pad_raster(np.array([[1, 2], [3, 4]]), pad_raster_value=0)
    array([[0., 0., 0., 0.],
           [0., 1., 2., 0.],
           [0., 3., 4., 0.],
           [0., 0., 0., 0.]])
    """
    cells = int(pad_raster_cells)
    if cells < 0:
        raise ValueError(f"pad_raster_cells must be >= 0, got {pad_raster_cells}")
    layers, single = _layers(data)
    out = []
    for layer in layers:
        values, (xmin, xmax, ymin, ymax), crs, georef = _raw(layer, nodata, band)
        ny, nx = values.shape
        padded = np.full((ny + 2 * cells, nx + 2 * cells), float(pad_raster_value))
        padded[cells:cells + ny, cells:cells + nx] = values
        if global_:
            padded[np.isnan(padded)] = float(pad_raster_value)
        if return_raster and georef:
            # terra: ext(x) + res(x) * cells
            rx, ry = (xmax - xmin) / nx, (ymax - ymin) / ny
            padded = terra_grid(padded, (xmin - rx * cells, xmax + rx * cells, ymin - ry * cells, ymax + ry * cells),
                                crs)
        out.append(padded)
    return out[0] if single else out


def unpad_raster(data, unpad_raster_cells: int = 1, return_raster: bool = True, *, nodata=None,
                 band: int | None = None):
    """Remove ``unpad_raster_cells`` rows and columns on each side of a
        raster, like landscapemetrics' ``unpad_raster`` (the inverse of
        :func:`pad_raster`). Arguments and result as in :func:`pad_raster`.

        Parameters
        ----------
        data
            A raster or a list of rasters.
        unpad_raster_cells
            Rows and columns removed on each side.
        return_raster
            Return a georeferenced DataArray for georeferenced input.
        nodata, band
            Reading options.

        Returns
        -------
        xarray.DataArray or numpy.ndarray (float), or a list for several rasters.

        Examples
        --------
        >>> import landschaft as ls
        >>> padded = ls.pad_raster(ls.example_data("clc"), pad_raster_cells=2)
        >>> ls.unpad_raster(padded, 2).shape
        (2000, 2000)
    """
    cells = int(unpad_raster_cells)
    if cells < 1:
        raise ValueError(f"unpad_raster_cells must be >= 1, got {unpad_raster_cells}")
    layers, single = _layers(data)
    out = []
    for layer in layers:
        values, (xmin, xmax, ymin, ymax), crs, georef = _raw(layer, nodata, band)
        ny, nx = values.shape
        if ny <= 2 * cells or nx <= 2 * cells:
            raise ValueError(f"cannot remove {cells} cells from each side of a {ny} x {nx} raster")
        inner = values[cells:ny - cells, cells:nx - cells].copy()
        if return_raster and georef:
            rx, ry = (xmax - xmin) / nx, (ymax - ymin) / ny
            inner = terra_grid(inner, (xmin + rx * cells, xmax - rx * cells, ymin + ry * cells, ymax - ry * cells),
                               crs)
        out.append(inner)
    return out[0] if single else out


# -- raster_to_points ---------------------------------------------------------------------


def raster_to_points(data, return_NA: bool = True, *, nodata=None, band: int | None = None) -> pd.DataFrame:
    """Cell centres and values, like landscapemetrics' ``raster_to_points``.

    Parameters
    ----------
    data
        A raster or a list of rasters (see :func:`pad_raster`). Numpy input
        has terra's extent of a matrix (0..ncol, 0..nrow).
    return_NA
        Keep the NA cells (value NaN).

    Returns
    -------
    pandas.DataFrame
        ``layer``, ``x``, ``y``, ``z``, cell by cell in row order (the
        coordinates are terra's ``xyFromCell``).

    Examples
    --------
    >>> import numpy as np
    >>> import landschaft as ls
    >>> ls.raster_to_points(np.array([[1, 2], [np.nan, 4]]), return_NA=False)
       layer    x    y    z
    0      1  0.5  1.5  1.0
    1      1  1.5  1.5  2.0
    2      1  1.5  0.5  4.0
    """
    layers, _ = _layers(data)
    frames = []
    for k, layer in enumerate(layers, start=1):
        values, (xmin, xmax, ymin, ymax), _, _ = _raw(layer, nodata, band)
        ny, nx = values.shape
        rx, ry = (xmax - xmin) / nx, (ymax - ymin) / ny
        x = xmin + (np.arange(nx) + 0.5) * rx
        y = ymax - (np.arange(ny) + 0.5) * ry
        frame = pd.DataFrame({
            "layer": np.full(nx * ny, k, dtype=np.int64),
            "x": np.tile(x, ny),
            "y": np.repeat(y, nx),
            "z": values.ravel(),
        })
        if not return_NA:
            frame = frame[~np.isnan(frame["z"].to_numpy())]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].reset_index(drop=True)


# -- get_unique_values -------------------------------------------------------------------


def _unique(layer, nodata, band) -> np.ndarray:
    from .landscape import _prepare

    if not isinstance(layer, (str, os.PathLike, xr.DataArray)) and not np.ma.isMaskedArray(layer):
        arr = np.asarray(layer)
        if arr.ndim == 1:
            layer = arr.reshape(1, -1)
    values, mask, _, _ = _prepare(layer, None, nodata, band)
    return _core.lsm_unique_values(values, mask)


def get_unique_values(data, simplify: bool = False, *, nodata=None, band: int | None = None):
    """The distinct values of rasters (NA left out, doubles truncated to
        integers), sorted, like landscapemetrics' ``get_unique_values``.

        ``data``: a raster (file, DataArray, numpy array of any dimension up to 2,
        masked array) or a list of them. Returns ``{"layer_1": values, ...}``
        as landscapemetrics does, or only the values of a single raster with
        ``simplify=True``.

        Parameters
        ----------
        data
            A raster (file, DataArray, numpy or masked array, also 1-D) or a list
            of them.
        simplify
            Return the values of a single raster directly.
        nodata, band
            Reading options.

        Returns
        -------
        dict or numpy.ndarray

        Examples
        --------
        >>> import landschaft as ls
        >>> ls.get_unique_values(ls.example_data("input"), simplify=True).tolist()
        [0, 1, 2]
    """
    layers, single = _layers(data)
    result = {f"layer_{k}": _unique(layer, nodata, band) for k, layer in enumerate(layers, start=1)}
    if simplify:
        if len(result) == 1:
            return next(iter(result.values()))
        import warnings

        warnings.warn("Not able to simplify input with more than one layer.", stacklevel=2)
    return result
