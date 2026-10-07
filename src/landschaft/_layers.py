# SPDX-License-Identifier: GPL-3.0-or-later
"""Named stacks of rasters on one grid: land-cover fractions by class, species ranges, ..."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

__all__ = ["named_layers", "like"]


def _one(source) -> xr.DataArray:
    from .io import load_raster

    if isinstance(source, xr.DataArray):
        return source
    if isinstance(source, np.ndarray):
        return xr.DataArray(np.asarray(source), dims=("y", "x") if source.ndim == 2 else ("band", "y", "x"))
    return load_raster(source)


def _split(da: xr.DataArray, what: str) -> list[tuple[str, xr.DataArray]]:
    if da.ndim == 2:
        band = da.coords.get("band")
        name = da.name or (band.item() if band is not None and band.ndim == 0 else what)
        return [(str(name), da)]
    if da.ndim != 3:
        raise ValueError(f"{what}: a raster must be 2-D or (band, y, x)")
    dim = da.dims[0]
    names = da[dim].values if dim in da.coords else [f"{what}_{k + 1}" for k in range(da.shape[0])]
    return [(str(n), da[k]) for k, n in enumerate(names)]


def named_layers(data, what: str = "layer") -> tuple[list[str], np.ndarray, xr.DataArray]:
    """Layer names, their values as float64 ``(k, y, x)`` and a 2-D template of the grid.

    ``data``: a ``(band, y, x)`` DataArray (the band coordinate names the layers), a
    Dataset, a mapping name -> raster, a :class:`~landschaft.RasterCollection`
    or a list of rasters (files, DataArrays).
    """
    from .collection import RasterCollection

    if isinstance(data, RasterCollection):
        if not data.same_grid:
            raise ValueError(f"{what}: the rasters must share a grid")
        data = data.to_stack()
    if isinstance(data, xr.Dataset):
        pairs = [(str(k), v) for k, v in data.data_vars.items()]
    elif isinstance(data, Mapping):
        pairs = [(str(k), _one(v)) for k, v in data.items()]
    elif isinstance(data, (xr.DataArray, np.ndarray)):
        pairs = _split(_one(data), what)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        pairs = [p for item in data for p in _split(_one(item), what)]
    else:
        pairs = _split(_one(data), what)
    if not pairs:
        raise ValueError(f"{what}: no rasters")
    names = [n for n, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"{what}: the layer names repeat: {names}")
    shape = pairs[0][1].shape
    for n, v in pairs:
        if v.shape != shape:
            raise ValueError(f"{what}: layer {n!r} has shape {v.shape}, the first {shape}")
    values = np.stack([np.asarray(v.values, dtype=np.float64) for _, v in pairs])
    template = pairs[0][1]
    if "band" in template.dims:
        template = template.isel(band=0)
    template = template.drop_vars("band", errors="ignore")
    return names, values, template


def like(template: xr.DataArray, values: np.ndarray, name: str | None = None, bands=None) -> xr.DataArray:
    """A raster with ``values`` on the template's grid (2-D, or ``(band, y, x)`` with ``bands``)."""
    if values.ndim == 2:
        out = template.copy(data=values)
    else:
        out = xr.concat([template.copy(data=v) for v in values], dim="band")
        out = out.assign_coords(band=list(bands) if bands is not None else np.arange(1, values.shape[0] + 1))
    out.name = name
    out.attrs = {k: v for k, v in template.attrs.items() if k not in ("_FillValue", "long_name", "scale_factor",
                                                                      "add_offset", "units")}
    if hasattr(out, "rio"):
        try:
            out = out.rio.write_nodata(np.nan, encoded=False)
        except Exception:  # noqa: BLE001 - no rioxarray grid on plain arrays
            pass
    return out
