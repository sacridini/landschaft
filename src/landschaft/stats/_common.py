# SPDX-License-Identifier: GPL-3.0-or-later
"""Maps in and out of the statistics: float64 values with NaN, results on the input's grid."""

from __future__ import annotations

import numpy as np
import xarray as xr


def read_values(x, nodata=None, what: str = "the map"):
    """(float64 values with NaN where missing, template DataArray or None) of a 2-D map."""
    from ..simulate._maps import read_map

    try:
        m = read_map(x, nodata)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what}: {e}") from None
    return np.ascontiguousarray(m.as_float()), m.template


def wrap(values: np.ndarray, template, name: str):
    """``values`` on the template's grid (a DataArray named ``name``), or a plain (y, x) DataArray."""
    if template is None:
        return xr.DataArray(values, dims=("y", "x"), name=name)
    out = template.copy(data=values).rename(name)
    out.attrs = {}
    try:
        out = out.rio.write_nodata(np.nan)
    except Exception:  # noqa: BLE001, S110  (no rioxarray georeference)
        pass
    return out
