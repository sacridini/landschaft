# SPDX-License-Identifier: GPL-3.0-or-later
"""Omniscape: omnidirectional circuit connectivity over moving windows."""

from __future__ import annotations

import warnings

import numpy as np

from ._backend import core
from .advanced import _grid

COMPARISONS = ("within", "equal")
COMPARE_TO_FUTURE = ("none", "1", "2", "both")
PRECISIONS = ("double", "single")


def physical_memory() -> int:
    """Bytes of physical memory (0 when unknown)."""
    import os
    import sys

    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                            ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                            ("avail_virtual", ctypes.c_ulonglong), ("avail_extended", ctypes.c_ulonglong)]

            st = Status()
            st.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.total)
            return 0
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 0


def _to_int(value, name: str) -> int:
    # Omniscape parses Int64(round(parse(Float64, value))): halves go to even, as Python's round.
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return int(round(v))


def read_reclass_table(path) -> np.ndarray:
    """An Omniscape reclass table: two whitespace-separated columns, ``missing`` for NoData."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"reclass table lines need two columns: {line.strip()!r}")
            rows.append([np.nan if p.strip('"') == "missing" else float(p) for p in parts])
    return np.array(rows, dtype=np.float64).reshape(-1, 2)


def _reclass_table(table) -> np.ndarray:
    if isinstance(table, (str, bytes)) or hasattr(table, "__fspath__"):
        return read_reclass_table(table)
    if isinstance(table, dict):
        table = [[k, np.nan if v is None else v] for k, v in table.items()]
    t = np.array(table, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != 2:
        raise ValueError("reclass_table must have two columns (value, new value)")
    return t


def reclassify(resistance: np.ndarray, table) -> np.ndarray:
    """Omniscape's ``reclassify_resistance!``: every row (value, new value) in turn, matched
    against the original values (NaN in the new column: NoData)."""
    t = _reclass_table(table)
    out = resistance.copy()
    for old, new in t:
        out[resistance == old] = new
    return out


def source_from_resistance(resistance, *, r_cutoff: float = np.inf, resistance_is_conductance: bool = False,
                           reclass_table=None):
    """Source strengths from the resistance, as Omniscape's ``source_from_resistance``.

    The conductance (1 / resistance, or the values themselves with
    ``resistance_is_conductance``) after the optional reclassification, set to
    0 where it is below ``1 / r_cutoff`` and on NoData.
    """
    r, like = _grid(resistance, "resistance")
    if reclass_table is not None:
        r = reclassify(r, reclass_table)
    with np.errstate(divide="ignore"):
        s = r if resistance_is_conductance else 1 / r
    s = np.where(np.isnan(s) | (s < 1 / float(r_cutoff)), 0.0, s)
    if like is None:
        return s
    out = like.copy(data=s)
    out.name = "source"
    return out


def _condition(value, shape, name, dtype):
    if value is None:
        return None
    v, _ = _grid(value, name)
    if v.shape != shape:
        raise ValueError(f"{name} must have the resistance's shape")
    if dtype is np.float32:
        v = v.astype(np.float32).astype(np.float64)
    return np.ascontiguousarray(v)


def omniscape(resistance, radius, *, source=None, block_size=1, buffer=0, source_threshold: float = 0.0,
              r_cutoff: float = np.inf, resistance_is_conductance: bool = False, four_neighbors: bool = False,
              flow_potential: bool = False, normalized_current: bool = False, correct_artifacts: bool = True,
              mask_nodata: bool = True, reclass_table=None, condition1=None, condition2=None,
              condition1_future=None, condition2_future=None, compare_to_future: str = "none",
              comparison1: str = "within", comparison2: str = "within", condition1_lower: float = 0.0,
              condition1_upper: float = 0.0, condition2_lower: float = 0.0, condition2_upper: float = 0.0,
              precision: str = "double", memory_limit: float | None = None, n_threads: int = 0):
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
    reclass_table
        Reclassify the resistance first (``reclassify_resistance``): a
        two-column table (value, new value; NaN makes NoData), a dict, or the
        path of an Omniscape reclass file. The default sources come from the
        reclassified resistance.
    condition1, condition2, condition1_future, condition2_future
        Conditional connectivity (``conditional``, ``n_conditions``): after a
        window's sources are scaled, those whose ``condition1`` (and
        ``condition2``) does not match the target block's value of the future
        condition are set to 0. The future conditions default to the present
        ones; ``compare_to_future`` (``"none"``, ``"1"``, ``"2"`` or
        ``"both"``) says which futures are used, as in Omniscape.
    comparison1, comparison2, condition1_lower, condition1_upper, condition2_lower, condition2_upper
        ``"within"``: kept when lower <= condition - median(block) <= upper;
        ``"equal"``: kept when condition == mode(block).
    precision
        ``"single"`` reads the rasters as float32 and returns float32 maps, as
        Omniscape's ``precision = single``; the arithmetic stays double.
    memory_limit
        Bytes the solves may use (default: half of the physical memory); with
        large windows fewer threads run when each thread's factorisation of a
        full window would not fit (``attrs["threads"]``).
    n_threads
        OpenMP threads (0: all cores but one).

    Returns
    -------
    xarray.Dataset
        ``cum_currmap`` and, when asked for, ``flow_potential`` and
        ``normalized_cum_currmap``, on the resistance's grid; ``attrs``
        records the options, the number of targets and of threads used.
    """
    import xarray as xr

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    dtype = np.float32 if precision == "single" else np.float64
    r, like = _grid(resistance, "resistance")
    if dtype is np.float32:
        r = r.astype(np.float32).astype(np.float64)
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
        s = source_from_resistance(r, r_cutoff=r_cutoff, resistance_is_conductance=resistance_is_conductance,
                                   reclass_table=reclass_table)
    else:
        s, _ = _grid(source, "source")
        if s.shape != r.shape:
            raise ValueError("the source raster must have the resistance's shape")
        if dtype is np.float32:
            s = s.astype(np.float32).astype(np.float64)
        s = np.where(np.isnan(s), 0.0, s)
        s = np.where(s < source_threshold, 0.0, s)
    s = np.ascontiguousarray(s, dtype=np.float64)
    if reclass_table is not None:
        r = reclassify(r, reclass_table)
        if dtype is np.float32:
            r = r.astype(np.float32).astype(np.float64)
        finite = r[~np.isnan(r)]
        if finite.size and finite.min() <= 0:
            raise ValueError("the reclassified resistance must be > 0")
    r = np.ascontiguousarray(r)

    conditions = {}
    if condition2 is not None and condition1 is None:
        raise ValueError("condition2 needs condition1")
    if condition1 is not None:
        if compare_to_future not in COMPARE_TO_FUTURE:
            raise ValueError(f"compare_to_future must be one of {COMPARE_TO_FUTURE}")
        for k, cmp in ((1, comparison1), (2, comparison2)):
            if cmp not in COMPARISONS:
                raise ValueError(f"comparison{k} must be one of {COMPARISONS}")
        future1 = condition1_future if compare_to_future in ("1", "both") else None
        future2 = condition2_future if compare_to_future in ("2", "both") else None
        if compare_to_future in ("1", "both") and condition1_future is None:
            raise ValueError("compare_to_future needs condition1_future")
        if compare_to_future in ("2", "both") and condition2 is not None and condition2_future is None:
            raise ValueError("compare_to_future needs condition2_future")
        conditions = {
            "condition1": _condition(condition1, r.shape, "condition1", dtype),
            "condition1_future": _condition(future1, r.shape, "condition1_future", dtype),
            "condition2": _condition(condition2, r.shape, "condition2", dtype),
            "condition2_future": _condition(future2, r.shape, "condition2_future", dtype),
            "comparison1": comparison1, "comparison2": comparison2,
            "condition1_lower": float(condition1_lower), "condition1_upper": float(condition1_upper),
            "condition2_lower": float(condition2_lower), "condition2_upper": float(condition2_upper),
        }
    elif any(c is not None for c in (condition1_future, condition2_future)):
        raise ValueError("future conditions need condition1")

    compute_fp = bool(flow_potential or normalized_current)
    out = core().circuit_omniscape(r, s, radius, buffer, block_size, bool(four_neighbors),
                                   bool(resistance_is_conductance), bool(correct_artifacts), compute_fp, int(n_threads),
                                   conditions, float(physical_memory() / 2 if memory_limit is None else memory_limit))
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
    if dtype is np.float32:
        data = {k: v.astype(np.float32) for k, v in data.items()}
    attrs = {"radius": radius, "block_size": block_size, "buffer": buffer, "n_targets": int(out["n_targets"]),
             "four_neighbors": int(bool(four_neighbors)), "correct_artifacts": int(bool(correct_artifacts)),
             "precision": precision, "conditional": int(condition1 is not None), "threads": int(out["threads"])}
    order = ["cum_currmap", "flow_potential", "normalized_cum_currmap"]
    if like is None:
        return xr.Dataset({k: (("y", "x"), data[k]) for k in order if k in data}, attrs=attrs)
    ds = xr.Dataset({k: like.copy(data=data[k]).rename(k) for k in order if k in data}, attrs=attrs)
    if hasattr(ds, "rio"):
        for k in ds.data_vars:
            ds[k] = ds[k].rio.write_nodata(np.nan, encoded=False)
    return ds
