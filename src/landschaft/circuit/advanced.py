# SPDX-License-Identifier: GPL-3.0-or-later
"""Circuitscape's advanced mode on a raster (sources and grounds, one solve)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._backend import core


@dataclass
class AdvancedResult:
    """Node currents and voltages of an advanced-mode solve.

    ``current`` is the current through every cell (``max(inflow, outflow)``
    at its node, 0 where nothing was solved); ``voltage`` the node voltages (0
    in components without a source or a ground, NaN on cells without a node).
    DataArrays when the conductance was one.
    """

    current: object
    voltage: object


def _grid(values, name: str, *, band=None):
    """A raster (path, DataArray or array) as float64 values and the DataArray it came from, if any."""
    import xarray as xr

    if isinstance(values, (str, bytes)) or hasattr(values, "__fspath__"):
        from ..io import load_raster

        values = load_raster(values, band=band, validate=None)
    if isinstance(values, xr.DataArray):
        da = values.squeeze(drop=True) if "band" in values.dims else values
        v = np.array(da.values, dtype=np.float64)
        nodata = da.rio.nodata if hasattr(da, "rio") else None
        if nodata is not None and not np.isnan(nodata):
            v[v == nodata] = np.nan
        like = da
    else:
        v = np.array(np.ma.filled(np.ma.asarray(values, dtype=np.float64), np.nan), dtype=np.float64)
        like = None
    if v.ndim != 2:
        raise ValueError(f"the {name} must be a 2-D raster")
    return np.ascontiguousarray(v), like


def _like(values: np.ndarray, like, name: str):
    if like is None:
        return values
    out = like.copy(data=values)
    out.name = name
    if hasattr(out, "rio"):
        out = out.rio.write_nodata(np.nan, encoded=False)
    return out


def advanced(conductance, source, ground, *, four_neighbors: bool = False, n_threads: int = 0) -> AdvancedResult:
    """Currents and voltages of a raster circuit with sources and grounds.

    Circuitscape 6.1's ``compute_omniscape_current`` (advanced mode, the
    solve behind every Omniscape moving window): each cell with a
    conductance > 0 is a node joined to its 8 neighbours (4 with
    ``four_neighbors``) by the average conductance of the two cells, over √2
    on the diagonals. ``source`` injects current (amps) into its cells;
    ``ground`` connects cells to ground with the given conductance (``inf``:
    a direct ground at 0 V). A cell that is both loses its source. Every
    connected component with a source and a ground is solved exactly (sparse
    Cholesky in a nested-dissection order; the components in parallel), and
    a node's current is the larger of its inflow and outflow, leaving out
    branch currents under 1e-8 of the largest one, as Circuitscape does.

    Parameters
    ----------
    conductance, source, ground
        Rasters of the same shape (paths, DataArrays or arrays). NaN (or the
        nodata value) in ``conductance`` means no node, in ``source`` and
        ``ground`` nothing.
    four_neighbors
        Connect the four orthogonal neighbours only.
    n_threads
        OpenMP threads (0: all cores but one).

    Returns
    -------
    AdvancedResult
        ``current`` and ``voltage``.
    """
    g, like = _grid(conductance, "conductance")
    s, _ = _grid(source, "source")
    gr, _ = _grid(ground, "ground")
    if s.shape != g.shape or gr.shape != g.shape:
        raise ValueError("conductance, source and ground must have the same shape")
    if np.any(g == np.inf):
        raise ValueError("conductances must be finite")
    if np.any(gr[np.isfinite(gr)] < 0):
        raise ValueError("ground conductances cannot be negative")
    s = np.where(np.isnan(s), 0.0, s)
    gr = np.where(np.isnan(gr), 0.0, gr)
    r = core().circuit_advanced(np.where(np.isnan(g), 0.0, g), s, gr, bool(four_neighbors), int(n_threads))
    return AdvancedResult(_like(r["current"], like, "current"), _like(r["voltage"], like, "voltage"))
