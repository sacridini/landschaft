# SPDX-License-Identifier: GPL-3.0-or-later
"""Omniscape: omnidirectional circuit connectivity over moving windows."""

from __future__ import annotations

import warnings

import numpy as np

from .. import _core
from .advanced import _grid


def _to_int(value, name: str) -> int:
    # Omniscape parses Int64(round(parse(Float64, value))): halves go to even, as Python's round.
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return int(round(v))


def source_from_resistance(resistance, *, r_cutoff: float = np.inf, resistance_is_conductance: bool = False):
    """Source strengths from the resistance, as Omniscape's ``source_from_resistance``.

    The conductance (1 / resistance, or the values themselves with
    ``resistance_is_conductance``), set to 0 where it is below ``1 / r_cutoff``
    and on NoData.
    """
    r, like = _grid(resistance, "resistance")
    with np.errstate(divide="ignore"):
        s = r if resistance_is_conductance else 1 / r
    s = np.where(np.isnan(s) | (s < 1 / float(r_cutoff)), 0.0, s)
    if like is None:
        return s
    out = like.copy(data=s)
    out.name = "source"
    return out


def omniscape(resistance, radius, *, source=None, block_size=1, buffer=0, source_threshold: float = 0.0,
              r_cutoff: float = np.inf, resistance_is_conductance: bool = False, four_neighbors: bool = False,
              flow_potential: bool = False, normalized_current: bool = False, correct_artifacts: bool = True,
              mask_nodata: bool = True, n_threads: int = 0):
    """Omnidirectional connectivity with Omniscape's moving-window circuits.

    Omniscape 0.7 (``run_omniscape``; McRae et al. 2016, Landau et al. 2021):
    every ``block_size`` × ``block_size`` block of cells whose source
    strengths add up to more than 0 is a target. Its window is the circle of
    ``radius`` + ``buffer`` cells around the block's centre; the block's
    strength is injected into the sources within ``radius`` (the target block
    left out, shared in proportion to their strengths), the target cell is
    grounded, and the current through every cell of the window is added to the
    cumulative map. Windows are solved exactly (Circuitscape's circuit, see
    :func:`landschaft.circuit.advanced`) in parallel and summed in a fixed
    order, so the result does not depend on the number of threads.

    Omniscape's details are kept: windows and circles cut at the map edges,
    the target block zeroed at ``radius + buffer`` cells from the window's
    corner (so not at the target when the window is cut on the left or the
    top), and, with ``block_size`` > 1, its block-artifact correction.

    Parameters
    ----------
    resistance
        Resistance raster (path, DataArray or array); NaN or the nodata value
        is NoData (infinite resistance). Values must be > 0.
    radius
        Radius of the moving window, in cells.
    source
        Source strengths (NaN is 0; values below ``source_threshold`` are set
        to 0, as when Omniscape reads a source file). Default: from the
        resistance, as :func:`source_from_resistance` with ``r_cutoff``.
    block_size
        Side of the target blocks (made odd as Omniscape does: an even size
        gets + 1, with a warning).
    buffer
        Cells added to the window beyond ``radius`` (conductance only).
    source_threshold, r_cutoff
        Omniscape's options of the same names.
    resistance_is_conductance
        The raster holds conductances.
    four_neighbors
        ``connect_four_neighbors_only``.
    flow_potential, normalized_current
        Also compute the flow potential (the same circuits with conductance 1)
        and the current normalised by it (0 where both are 0).
    correct_artifacts
        Omniscape's correction of block artifacts (only with ``block_size`` > 1).
    mask_nodata
        NaN on the NoData cells of the resistance.
    n_threads
        OpenMP threads (0 = all).

    Returns
    -------
    xarray.Dataset
        ``cum_currmap`` and, when asked for, ``flow_potential`` and
        ``normalized_cum_currmap``, on the resistance's grid; ``attrs``
        records the options and the number of targets.
    """
    import xarray as xr

    r, like = _grid(resistance, "resistance")
    finite = r[~np.isnan(r)]
    if finite.size and finite.min() <= 0:
        raise ValueError("resistance (or conductance) values must be > 0")
    radius = _to_int(radius, "radius")
    buffer = _to_int(buffer, "buffer")
    block_size = _to_int(block_size, "block_size")
    if radius < 0 or buffer < 0 or block_size < 1:
        raise ValueError("radius and buffer must be >= 0 and block_size >= 1")
    if block_size % 2 == 0:
        warnings.warn("block_size is even, but must be odd. Using block_size + 1.", stacklevel=2)
        block_size += 1
    if source is None:
        s = source_from_resistance(r, r_cutoff=r_cutoff, resistance_is_conductance=resistance_is_conductance)
    else:
        s, _ = _grid(source, "source")
        if s.shape != r.shape:
            raise ValueError("the source raster must have the resistance's shape")
        s = np.where(np.isnan(s), 0.0, s)
        s = np.where(s < source_threshold, 0.0, s)
    s = np.ascontiguousarray(s, dtype=np.float64)
    compute_fp = bool(flow_potential or normalized_current)
    out = _core.circuit_omniscape(r, s, radius, buffer, block_size, bool(four_neighbors),
                                  bool(resistance_is_conductance), bool(correct_artifacts), compute_fp, int(n_threads))
    cum = out["current"]
    data = {"cum_currmap": cum}
    nodata = np.isnan(r)
    if compute_fp:
        fp = out["flow_potential"]
        if normalized_current:
            with np.errstate(divide="ignore", invalid="ignore"):
                norm = cum / fp
            norm[np.isnan(norm)] = 0.0
            data["normalized_cum_currmap"] = norm
        if flow_potential:
            data["flow_potential"] = fp
    if mask_nodata:
        for v in data.values():
            v[nodata] = np.nan
    attrs = {"radius": radius, "block_size": block_size, "buffer": buffer, "n_targets": int(out["n_targets"]),
             "four_neighbors": int(bool(four_neighbors)), "correct_artifacts": int(bool(correct_artifacts))}
    order = ["cum_currmap", "flow_potential", "normalized_cum_currmap"]
    if like is None:
        return xr.Dataset({k: (("y", "x"), data[k]) for k in order if k in data}, attrs=attrs)
    ds = xr.Dataset({k: like.copy(data=data[k]).rename(k) for k in order if k in data}, attrs=attrs)
    if hasattr(ds, "rio"):
        for k in ds.data_vars:
            ds[k] = ds[k].rio.write_nodata(np.nan, encoded=False)
    return ds
