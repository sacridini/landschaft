# SPDX-License-Identifier: GPL-3.0-or-later
"""Circuitscape 6.1's raster modes on landschaft's solver."""

from __future__ import annotations

import gzip
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._backend import core

POLICIES = ("keepall", "rmvsrc", "rmvgnd", "rmvall")
SCENARIOS = {"pairwise": "pairwise", "Pairwise": "pairwise", "advanced": "advanced", "Advanced": "advanced",
             "one-to-all": "one-to-all", "one_to_all": "one-to-all", "all-to-one": "all-to-one",
             "all_to_one": "all-to-one"}
TRUELIST = ("True", "true", "1")
FALSELIST = ("False", "false", "0")
NODATA = -9999.0


# --- reading, as Circuitscape's io.jl --------------------------------------------------

@dataclass
class RasterMeta:
    """Grid of the habitat raster: size, lower-left corner, cell size, GDAL transform and WKT."""

    nrows: int
    ncols: int
    xllcorner: float
    yllcorner: float
    cellsize: float
    transform: tuple = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    wkt: str = ""
    crs: object = None


def _open(path) -> object:
    p = os.fspath(path)
    return gzip.open(p, "rt") if p.lower().endswith("gz") else open(p, encoding="utf-8")


def file_type(path) -> str:
    """Circuitscape's _guess_file_type: ``geotiff``, ``npy``, ``aagrid``, ``pairs_aagrid``, ``pairs`` or ``list``."""
    p = os.fspath(path)
    raw = (gzip.open if p.lower().endswith("gz") else open)(p, "rb")
    with raw as f:
        head = f.read(4)
    if head[2:4] == b"\x2a\x00":
        return "geotiff"
    with _open(p) as f:
        line = f.readline()
    if line.startswith("\x93NUMPY"):
        return "npy"
    if line.lower().startswith("ncols"):
        return "aagrid"
    if line.startswith("min"):
        return "pairs_aagrid"
    if line.startswith("mode"):
        return "pairs"
    return "list"


_ASC_KEYS = ("ncols", "nrows", "xllcorner", "yllcorner", "xllcenter", "yllcenter", "cellsize", "dx", "dy",
             "nodata_value")


def _read_asc(path) -> tuple[np.ndarray, RasterMeta]:
    # read_asc: the header, then one row per line, parsed as double.
    header: dict[str, float] = {}
    rows = []
    with _open(path) as f:
        for line in f:
            toks = line.split()
            if not toks:
                continue
            if not rows and len(toks) == 2 and toks[0].lower() in _ASC_KEYS:
                header[toks[0].lower()] = float(toks[1])
                continue
            rows.append([float(t) for t in toks])
    nrows, ncols = int(header["nrows"]), int(header["ncols"])
    a = np.array(rows, dtype=np.float64)
    if a.shape != (nrows, ncols):
        raise ValueError(f"{path}: the ASCII grid does not hold {nrows} rows of {ncols} values")
    dx = header.get("dx", header.get("cellsize", math.nan))
    dy = header.get("dy", dx)
    xll = header["xllcorner"] if "xllcorner" in header else header["xllcenter"] - dx / 2
    yll = header["yllcorner"] if "yllcorner" in header else header["yllcenter"] - dy / 2
    nodata = header.get("nodata_value", NODATA)
    a[(a == nodata) | np.isnan(a)] = NODATA
    prj = Path(os.fspath(path).removesuffix(".gz")[:-4] + ".prj")
    wkt = prj.read_text().strip() if prj.exists() else ""
    return a, RasterMeta(nrows, ncols, xll, yll, dx, (xll, dx, 0.0, yll + nrows * dy, 0.0, -dy), wkt)


def read_raster(path) -> tuple[np.ndarray, RasterMeta]:
    """Circuitscape's read_raster: an ESRI ASCII grid (parsed in double) or any GDAL raster, with
    its nodata value and NaN set to -9999."""
    p = os.fspath(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f'the file "{p}" does not exist')
    if p.lower().removesuffix(".gz").endswith(".asc"):
        return _read_asc(p)
    import rasterio

    with rasterio.open(("/vsigzip/" + p) if p.endswith(".gz") else p) as ds:
        band = ds.read(1)
        nodata = ds.nodata
        t = ds.transform
        wkt = ds.crs.to_wkt() if ds.crs else ""
        crs = ds.crs
    a = band.astype(np.float64)
    if nodata is not None:
        kind = np.int64 if np.issubdtype(band.dtype, np.integer) else band.dtype
        a[a == np.float64(np.array(nodata).astype(kind))] = NODATA
    a[np.isnan(a)] = NODATA
    nrows, ncols = a.shape
    meta = RasterMeta(nrows, ncols, t.c, t.f - nrows * t.a, t.a, (t.c, t.a, t.b, t.f, t.d, t.e), wkt, crs)
    return a, meta


def _read_list(path) -> np.ndarray:
    rows = []
    with _open(path) as f:
        for line in f:
            toks = line.split()
            if toks:
                rows.append([float(t) for t in toks])
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def _list_cells(points: np.ndarray, meta: RasterMeta) -> tuple[np.ndarray, np.ndarray]:
    # 1-based row and column: ceil(nrows - (y - yll) / cellsize), ceil((x - xll) / cellsize).
    i = np.ceil(meta.nrows - (points[:, 2] - meta.yllcorner) / meta.cellsize).astype(np.int64)
    j = np.ceil((points[:, 1] - meta.xllcorner) / meta.cellsize).astype(np.int64)
    return i, j


def _check_meta(meta: RasterMeta, habitat: RasterMeta) -> None:
    for name in ("cellsize", "ncols", "nrows", "yllcorner", "xllcorner"):
        if getattr(meta, name) != getattr(habitat, name):
            warnings.warn(f"{name} is not the same", stacklevel=4)
            return


def read_polymap(path, habitat: RasterMeta, *, integer: bool = False, nodata_as=0) -> np.ndarray:
    """Circuitscape's read_polymap: a raster on the habitat's grid, NoData set to ``nodata_as``."""
    a, meta = read_raster(path)
    if integer:
        if not np.all(a == np.round(a)):
            raise ValueError(f"{path} contains non-integer values")
    if nodata_as is not None:
        a[a == NODATA] = nodata_as
    _check_meta(meta, habitat)
    return a


def read_reclass_table(path) -> dict[float, float]:
    """Circuitscape's reclass table (two columns, old and new value) as a lookup."""
    t = _read_list_n(path, 2)
    return {float(a): float(b) for a, b in t}


@dataclass
class FocalPoints:
    """Focal points as Circuitscape's read_point_map: 1-based rows and columns and ids, sorted by id."""

    rows: np.ndarray
    cols: np.ndarray
    ids: np.ndarray

    @property
    def cells(self) -> np.ndarray:
        return np.asarray(self.rows) - 1, np.asarray(self.cols) - 1

    @property
    def has_regions(self) -> bool:
        return len(self.ids) != len(np.unique(self.ids))


def read_point_map(path, habitat: RasterMeta) -> FocalPoints:
    """Circuitscape's read_point_map: a raster of ids (cells of one id: a focal region) or a
    text list of ``id x y``; negative ids dropped, sorted by id (stably)."""
    if file_type(path) == "list":
        pts = _read_list(path)
        i, j = _list_cells(pts, habitat)
        v = pts[:, 0]
    else:
        a = read_polymap(path, habitat, integer=True)
        jj, ii = np.nonzero(a.T)  # column-major order
        i, j, v = ii + 1, jj + 1, a[ii, jj]
    keep = v >= 0
    i, j, v = i[keep], j[keep], v[keep]
    order = np.argsort(v, kind="stable")
    i, j, v = i[order].astype(np.int64), j[order].astype(np.int64), v[order].astype(np.int64)
    if len(v) and (i.min() < 0 or j.min() < 0 or i.max() > habitat.nrows or j.max() > habitat.ncols):
        raise ValueError("At least one focal node location falls outside of habitat map")
    if len(np.unique(v)) < 2:
        raise ValueError("Less than two valid focal nodes found. Please check focal node location file.")
    return FocalPoints(i, j, v)


@dataclass
class IncludedPairs:
    """Circuitscape's include/exclude pairs: ``mode`` (include or exclude), the ids and the 0/1 matrix."""

    mode: str
    ids: np.ndarray
    matrix: np.ndarray

    def excluded(self) -> set[tuple[int, int]]:
        # exclude_pairs_from: include mode excludes the unmarked pairs, exclude mode the marked ones.
        target = 0 if self.mode == "include" else 1
        m = self.matrix
        n = len(self.ids)
        return {(int(self.ids[i]), int(self.ids[j])) for j in range(n) for i in range(n)
                if m[i, j] == target and m[j, i] == target}


def read_included_pairs(path) -> IncludedPairs:
    """Circuitscape's read_included_pairs: a ``min``/``max`` matrix or a ``mode`` list of pairs."""
    kind = file_type(path)
    with _open(path) as f:
        lines = [ln for ln in f.read().splitlines()]
    if kind == "pairs_aagrid":
        minval = float(lines[0].split()[1])
        maxval = float(lines[1].split()[1])
        rows = [[float(t) for t in ln.split()] for ln in lines[2:] if ln.split()]
        a = np.array(rows)
        ids = a[0, 1:].astype(np.int64)
        m = a[1:, 1:]
        m = np.where(m > maxval, 0, m)
        return IncludedPairs("include", ids, (m >= minval).astype(np.int64))
    if kind == "pairs":
        mode = lines[0].split()[1]
        pairs = np.array([[float(t) for t in ln.split()] for ln in lines[1:] if ln.split()]).reshape(-1, 2)
        ids = np.unique(pairs).astype(np.int64)
        ids = ids[ids != 0]
        pos = {int(v): k for k, v in enumerate(ids)}
        m = np.zeros((len(ids), len(ids)), dtype=np.int64)
        for a, b in pairs.astype(np.int64):
            if a in pos and b in pos:
                m[pos[a], pos[b]] = m[pos[b], pos[a]] = 1
        return IncludedPairs(mode, ids, m)
    raise ValueError("Error reading focal node include/exclude pairs file.")


def _read_list_n(path, ncol: int) -> np.ndarray:
    rows = []
    with _open(path) as f:
        for line in f:
            toks = line.split()
            if toks:
                rows.append([float(t) for t in toks])
    t = np.array(rows, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != ncol:
        raise ValueError(f"{path} must have exactly {ncol} columns")
    return t


# --- the habitat and the advanced inputs ---------------------------------------------

def habitat_conductance(cells: np.ndarray, *, is_resistance: bool = True, reclass: dict | None = None) -> np.ndarray:
    """Circuitscape's read_cellmap: optionally reclassified (a lookup on the values read), then
    1 / resistance (or the conductance), with the -9999 cells at 0."""
    c = np.array(cells, dtype=np.float64)
    if reclass:
        flat = c.ravel()
        for k, v in enumerate(flat):
            if v in reclass:
                flat[k] = reclass[v]
        c = flat.reshape(c.shape)
    nodata = c == NODATA
    if is_resistance:
        if np.any(c == 0):
            raise ValueError("zero resistance values are not supported for habitat maps; use a short-circuit "
                             "region file instead")
        g = 1 / c
    else:
        g = c.copy()
    g[nodata] = 0.0
    return g


def _array(value, name) -> np.ndarray:
    from .advanced import _grid

    a, _ = _grid(value, name)
    return np.where(np.isnan(a), NODATA, a)


def _cells_from_map(values, name, shape) -> np.ndarray:
    a = _array(values, name)
    if a.shape != shape:
        raise ValueError(f"the {name} must have the habitat's shape")
    return a


@dataclass
class CircuitscapeResult:
    """Circuitscape's maps: ``voltage`` and ``current`` (per cell; 0 on cells without a node,
    as Circuitscape writes them in advanced mode), the node map (-1: no node), and, by
    mode, ``resistances`` (the matrix Circuitscape returns and writes, ids in the first
    row and column), the cumulative and maximum current maps and the per-pair (or per
    focal node) maps."""

    voltage: np.ndarray | None = None
    current: np.ndarray | None = None
    nodemap: np.ndarray | None = None
    resistances: np.ndarray | None = None
    cum_current: np.ndarray | None = None
    max_current: np.ndarray | None = None
    maps: dict = field(default_factory=dict)


def circuitscape_advanced(habitat, source, ground, *, habitat_is_resistance: bool = True,
                          ground_is_resistance: bool = True, polygons=None, mask=None, four_neighbors: bool = False,
                          avg_resistances: bool = False, remove_src_or_gnd: str = "keepall",
                          unit_currents: bool = False, direct_grounds: bool = False, reclass: dict | None = None,
                          n_threads: int = 0) -> CircuitscapeResult:
    """Circuitscape's advanced mode on rasters (``scenario = advanced``).

    Every cell of ``habitat`` (resistance, or conductance with
    ``habitat_is_resistance=False``; NaN or -9999 is NoData) with a conductance
    > 0 is a node, joined to its 8 (or 4) neighbours by the average conductance
    (or, with ``avg_resistances``, the average resistance) of the two cells;
    the cells of a short-circuit region (``polygons``: ids > 0) that holds
    habitat are one node. ``source`` gives the current injected per cell,
    ``ground`` the resistance (or conductance) to ground per cell (NoData: no
    ground; 0 resistance: a direct ground); values of the cells of one node
    add up. ``remove_src_or_gnd`` resolves nodes that are both (Circuitscape's
    policies, with their quirks: ``rmvall`` only drops the source, and finite
    grounds dropped by ``rmvgnd`` stay on the matrix diagonal). Every connected
    component with a source and a ground is solved exactly.

    Returns
    -------
    CircuitscapeResult
        ``voltage`` and ``current`` maps (0 off the nodes and in components
        not solved), ``nodemap``.
    """
    if remove_src_or_gnd not in POLICIES:
        raise ValueError(f"remove_src_or_gnd must be one of {POLICIES}")
    cells = _array(habitat, "habitat")
    g = habitat_conductance(cells, is_resistance=habitat_is_resistance, reclass=reclass)
    poly = _polymap(polygons, g.shape)
    g = _mask(g, mask)
    src = _cells_from_map(source, "source", g.shape)
    src[src == NODATA] = 0.0
    gnd = _cells_from_map(ground, "ground", g.shape)
    nod = gnd == NODATA
    if ground_is_resistance:
        with np.errstate(divide="ignore"):
            gnd = 1 / gnd
    gnd[nod] = 0.0
    return _advanced(g, poly, src, gnd, four_neighbors, avg_resistances, remove_src_or_gnd, unit_currents,
                     direct_grounds, n_threads)


def _advanced(g, poly, src, gnd, four, avg, policy, unit, direct, n_threads) -> CircuitscapeResult:
    if unit:
        src = np.where(src != 0, 1.0, src)
    if direct:
        gnd = np.where(gnd != 0, np.inf, gnd)
    r = core().circuitscape_advanced(np.ascontiguousarray(g, dtype=np.float64), poly,
                                     np.ascontiguousarray(src, dtype=np.float64),
                                     np.ascontiguousarray(gnd, dtype=np.float64), bool(avg), bool(four), policy,
                                     int(n_threads))
    nm = r["nodemap"]
    on = nm >= 0
    volt = np.zeros(nm.shape)
    cur = np.zeros(nm.shape)
    volt[on] = r["voltage"][nm[on]]
    cur[on] = r["current"][nm[on]]
    res = CircuitscapeResult(voltage=volt, current=cur, nodemap=nm)
    res.maps["solved"] = bool(r["solved"])
    return res


def _pairwise(g, poly, points: FocalPoints, excluded_ids: set, four, avg, options: dict, emit, n_threads):
    rows, cols = points.cells
    cells = np.ascontiguousarray(rows * g.shape[1] + cols, dtype=np.int64)
    if points.has_regions:
        # pairwise_regions: a graph per pair of distinct ids.
        options = {**options, "region_ids": np.ascontiguousarray(points.ids, dtype=np.int64)}
        ids = np.unique(points.ids)
    else:
        ids = points.ids
    n = len(ids)
    ex = np.zeros((n, n), dtype=np.uint8)
    if excluded_ids:
        for a in range(n):
            for b in range(n):
                ex[a, b] = (int(ids[a]), int(ids[b])) in excluded_ids
    r = core().circuitscape_pairwise(np.ascontiguousarray(g, dtype=np.float64), poly, cells, ex, bool(avg), bool(four),
                                     options, emit, int(n_threads))
    res = np.zeros((n + 1, n + 1))
    res[0, 1:] = ids
    res[1:, 0] = ids
    res[1:, 1:] = r["resistances"]
    return res, r


def _pair_options(volt_maps, cur_maps, cum_only, max_maps, focal_zero, null_currents, null_voltages, log_transform):
    return {"volt_maps": bool(volt_maps), "cur_maps": bool(cur_maps and not cum_only),
            "cum": bool(cur_maps or cum_only), "max": bool(max_maps), "focal_zero": bool(focal_zero),
            "null_currents": bool(null_currents), "null_voltages": bool(null_voltages),
            "log_transform": bool(log_transform)}


def circuitscape_pairwise(habitat, points, *, habitat_is_resistance: bool = True, polygons=None, mask=None,
                          four_neighbors: bool = False, avg_resistances: bool = False, included_pairs=None,
                          voltage_maps: bool = False, current_maps: bool = False, cum_map: bool = False,
                          max_map: bool = False, set_focal_node_currents_to_zero: bool = False,
                          set_null_currents_to_nodata: bool = False, set_null_voltages_to_nodata: bool = False,
                          log_transform_maps: bool = False, reclass: dict | None = None,
                          n_threads: int = 0) -> CircuitscapeResult:
    """Circuitscape's pairwise mode on rasters (``scenario = pairwise``).

    Effective resistances between every pair of focal points (``points``: a
    raster of ids, cells of one id forming a focal region, or a
    :class:`FocalPoints`), on the graph of :func:`circuitscape_advanced`.
    Every component is factorised once and its pairs solved in batches
    (without maps or exclusions, Circuitscape's shortcut: only the pairs of the
    first focal point are solved). ``included_pairs`` (:class:`IncludedPairs`
    or a path) restricts the pairs as Circuitscape does. ``current_maps`` and
    ``voltage_maps`` keep every pair's maps in ``result.maps`` (keyed by the
    two ids); ``cum_map`` / ``max_map`` sum / take the maximum of the pairs'
    current maps.

    Returns
    -------
    CircuitscapeResult
        ``resistances`` (ids in the first row and column; -1: not connected),
        ``cum_current``, ``max_current``, ``maps``.
    """
    cells = _array(habitat, "habitat")
    g = habitat_conductance(cells, is_resistance=habitat_is_resistance, reclass=reclass)
    poly = _polymap(polygons, g.shape)
    g = _mask(g, mask)
    if not isinstance(points, FocalPoints):
        pm = _cells_from_map(points, "points", g.shape)
        pm[pm == NODATA] = 0
        meta = RasterMeta(g.shape[0], g.shape[1], 0.0, 0.0, 1.0)
        jj, ii = np.nonzero(pm.T)
        v = pm[ii, jj]
        keep = v >= 0
        order = np.argsort(v[keep], kind="stable")
        points = FocalPoints((ii[keep] + 1)[order], (jj[keep] + 1)[order], v[keep][order].astype(np.int64))
        del meta
    if included_pairs is not None and not isinstance(included_pairs, IncludedPairs):
        included_pairs = read_included_pairs(included_pairs)
    points, excluded = _apply_included_pairs(points, included_pairs)
    maps: dict = {}

    ids = np.unique(points.ids) if points.has_regions else points.ids

    def emit(ci, cj, volt, cur):
        key = (int(ids[ci]), int(ids[cj]))
        if volt is not None:
            maps[("voltage", *key)] = volt
        if cur is not None:
            maps[("current", *key)] = cur

    opts = _pair_options(voltage_maps, current_maps or cum_map, not current_maps and cum_map, max_map,
                         set_focal_node_currents_to_zero, set_null_currents_to_nodata, set_null_voltages_to_nodata,
                         log_transform_maps)
    res, r = _pairwise(g, poly, points, excluded, four_neighbors, avg_resistances, opts, emit, n_threads)
    return CircuitscapeResult(resistances=res, cum_current=r.get("cum"), max_current=r.get("max"), maps=maps)


def _polymap(polygons, shape):
    if polygons is None:
        return None
    pm = _cells_from_map(polygons, "polygons", shape)
    pm[pm == NODATA] = 0
    if not np.all(pm == np.round(pm)):
        raise ValueError("the polygon raster must hold integer ids")
    return np.ascontiguousarray(pm.astype(np.int64))


def _mask(g, mask):
    if mask is None:
        return g
    m = _cells_from_map(mask, "mask", g.shape)
    m[m == NODATA] = 0
    g = g * np.where(m > 0, 1.0, 0.0)
    if g.sum() == 0:
        raise ValueError("Mask file deleted everything!")
    return g


def _apply_included_pairs(points: FocalPoints, pairs: IncludedPairs | None):
    # generate_exclude_pairs: include mode prunes the focal points it does not name.
    if pairs is None:
        return points, set()
    if pairs.mode == "include":
        keep = np.isin(points.ids, pairs.ids)
        points = FocalPoints(points.rows[keep], points.cols[keep], points.ids[keep])
    return points, pairs.excluded()


def read_point_strengths(path) -> np.ndarray:
    """Circuitscape's read_point_strengths: two columns, id and strength (ids from 0 shifted to 1)."""
    t = _read_list_n(path, 2)
    if t[:, 0].min() == 0:
        t[:, 0] += 1
    return t


def _point_polymap(polymap, point_map: np.ndarray, ids: np.ndarray) -> np.ndarray:
    # create_new_polymap with a point map: focal points and regions become polygons.
    if polymap is None:
        return point_map.astype(np.int64)
    newpoly = polymap.copy()
    no_regions = len(ids) == len(np.unique(ids))
    cells = np.nonzero(point_map.T)  # column-major
    cells = list(zip(cells[1], cells[0]))
    if no_regions:
        k = polymap.max()
        for r, c in cells:
            if polymap[r, c] == 0:
                newpoly[r, c] = point_map[r, c] + k
        return newpoly
    k = max(polymap.max(), point_map.max())
    for r, c in cells:
        v1, v2 = point_map[r, c], newpoly[r, c]
        if v2 == 0:
            newpoly[r, c] = k + v1
            continue
        if v1 != v2:
            newpoly[newpoly == v2] = v1
    return newpoly


def _one_to_all(g, poly, points: FocalPoints, strengths, pairs: IncludedPairs | None, one_to_all: bool, four, avg,
                opts: dict, emit, n_threads) -> CircuitscapeResult:
    # run_onetoall
    ny, nx = g.shape
    if pairs is not None:
        keep = np.isin(points.ids, pairs.ids)
        points = FocalPoints(points.rows[keep], points.cols[keep], points.ids[keep])
        if strengths is not None:
            strengths = strengths[np.isin(strengths[:, 0], pairs.ids)]
    rows, cols = points.cells
    cells = rows * nx + cols
    ids = points.ids
    point_map = np.zeros((ny, nx), dtype=np.int64)
    point_map[rows, cols] = ids
    unique_ids, first = np.unique(ids, return_index=True)
    unique_map = np.zeros((ny, nx), dtype=np.int64)
    unique_map[rows[first], cols[first]] = unique_ids
    no_regions = len(ids) == len(unique_ids)
    mode = 0 if pairs is not None and pairs.mode == "include" else 1
    base_poly = _point_polymap(poly, point_map, ids)
    res = np.zeros(len(unique_ids))
    cum = np.zeros((ny, nx)) if opts["cum"] else None
    mx = np.full((ny, nx), NODATA) if opts["max"] else None
    policy = "rmvgnd" if one_to_all else "rmvsrc"
    maps: dict = {}

    def problem(i):
        n = int(unique_ids[i])
        pm = point_map.copy()
        newpoly = base_poly
        strength = float(strengths[i, 1]) if strengths is not None else 1.0
        if pairs is not None:
            row = int(np.nonzero(pairs.ids == n)[0][0])
            for j in range(len(pairs.ids)):
                if j != row and pairs.matrix[row, j] == mode:
                    pm[pm == pairs.ids[j]] = 0
            if not no_regions:
                newpoly = _point_polymap(poly, pm, ids)
        smap = None
        if strengths is not None:
            st = strengths[:, 1].copy()
            st[pm[rows, cols] == 0] = 1.0
            smap = np.zeros((ny, nx))
            smap[rows, cols] = st[: len(rows)]
        if pm.sum() == n:
            return None
        if one_to_all:
            src = np.where(unique_map == n, strength, 0.0)
            gnd = np.where(pm == n, 0, pm).astype(np.float64)
            gnd[gnd > 0] = np.inf
        else:
            if smap is not None:
                src = np.where(unique_map == n, 0.0, smap)
            else:
                src = np.where(unique_map != 0, 1.0, 0.0)
                src = np.where(pm == n, 0.0, src)
            gnd = np.where(pm == n, np.inf, 0.0)
        sc, gc = np.flatnonzero(src), np.flatnonzero(gnd)
        return {"n": n, "strength": strength, "poly": newpoly,
                "src_cells": sc.astype(np.int64), "src_vals": src.ravel()[sc], "gnd_cells": gc.astype(np.int64),
                "gnd_vals": gnd.ravel()[gc], "check_cell": int(cells[np.nonzero(ids == n)[0][0]])}

    threads = n_threads if n_threads > 0 else (os.cpu_count() or 1)
    batch = max(1, 4 * threads)
    for b0 in range(0, len(unique_ids), batch):
        probs = [(i, problem(i)) for i in range(b0, min(len(unique_ids), b0 + batch))]
        todo = [(i, q) for i, q in probs if q is not None]
        for i, q in probs:
            if q is None:
                res[i] = -1
        # One call per graph (focal regions with included pairs rebuild it per point).
        groups: list[tuple[object, list]] = []
        for i, q in todo:
            if groups and groups[-1][0] is q["poly"]:
                groups[-1][1].append((i, q))
            else:
                groups.append((q["poly"], [(i, q)]))
        solved: dict[int, tuple] = {}
        for pm_, items in groups:
            out = core().circuitscape_multi_advanced(np.ascontiguousarray(g, dtype=np.float64),
                                                     np.ascontiguousarray(pm_, dtype=np.int64), bool(avg), bool(four),
                                                     [q for _, q in items], policy, int(n_threads))
            for (i, _), r in zip(items, out["results"]):
                solved[i] = (out["nodemap"], r)
        for i, q in todo:
            nm, r = solved[i]
            n = q["n"]
            if not r["solved"]:
                res[i] = -1
            elif not one_to_all:
                res[i] = 0
            else:
                node = nm[unique_map == n][0]
                val = 0.0 if node < 0 else r["voltage"][node] / q["strength"]
                res[i] = -1 if val == 0 else val
            on = nm >= 0
            volt = np.zeros((ny, nx))
            volt[on] = r["voltage"][nm[on]]
            cur = np.zeros((ny, nx))
            cur[on] = r["current"][nm[on]]
            if opts["focal_zero"]:
                cur[np.isin(nm, r["active"])] = 0.0
            if cum is not None:
                cum += cur
            if mx is not None:
                np.maximum(mx, cur, out=mx)
            emit(n, volt if opts["volt_maps"] else None, cur if opts["cur_maps"] else None)
    if cum is not None:
        cum[cum < NODATA] = NODATA
    if mx is not None:
        mx[mx < NODATA] = NODATA
    result = np.column_stack([unique_ids.astype(np.float64), res])
    return CircuitscapeResult(resistances=result, cum_current=cum, max_current=mx, maps=maps)


def circuitscape_one_to_all(habitat, points, *, all_to_one: bool = False, habitat_is_resistance: bool = True,
                            polygons=None, mask=None, four_neighbors: bool = False, avg_resistances: bool = False,
                            included_pairs=None, strengths=None, voltage_maps: bool = False,
                            current_maps: bool = False, cum_map: bool = False, max_map: bool = False,
                            set_focal_node_currents_to_zero: bool = False, reclass: dict | None = None,
                            n_threads: int = 0) -> CircuitscapeResult:
    """Circuitscape's one-to-all (or, with ``all_to_one``, all-to-one) mode on rasters.

    For every focal point in turn (``points`` as in
    :func:`circuitscape_pairwise`; the focal points and regions are merged
    into the short-circuit regions): one-to-all injects 1 A (or its
    ``strengths`` value: a two-column table of id and strength) there with
    every other focal point a direct ground; all-to-one grounds it and injects
    1 A (or the strengths) at the others. Only the point's component is
    solved; the points run in parallel. Circuitscape's quirks are kept
    (points pruned by ``included_pairs`` in both of its modes, excluded points
    still sourcing in all-to-one, strengths indexed by position).

    Returns
    -------
    CircuitscapeResult
        ``resistances``: ids and the voltage over the strength at each point
        (one-to-all; -1 where nothing was solved) or 0 (all-to-one);
        ``cum_current``, ``max_current``, and the per-point maps in ``maps``.
    """
    cells = _array(habitat, "habitat")
    g = habitat_conductance(cells, is_resistance=habitat_is_resistance, reclass=reclass)
    poly = _polymap(polygons, g.shape)
    g = _mask(g, mask)
    points = _as_points(points, g.shape)
    if included_pairs is not None and not isinstance(included_pairs, IncludedPairs):
        included_pairs = read_included_pairs(included_pairs)
    if strengths is not None and isinstance(strengths, (str, os.PathLike)):
        strengths = read_point_strengths(strengths)
    maps: dict = {}

    def emit(n, volt, cur):
        if volt is not None:
            maps[("voltage", n)] = volt
        if cur is not None:
            maps[("current", n)] = cur

    opts = {"volt_maps": voltage_maps, "cur_maps": current_maps, "cum": current_maps or cum_map, "max": max_map,
            "focal_zero": set_focal_node_currents_to_zero}
    r = _one_to_all(g, poly, points, None if strengths is None else np.asarray(strengths, dtype=np.float64),
                    included_pairs, not all_to_one, four_neighbors, avg_resistances, opts, emit, n_threads)
    r.maps = maps
    return r


def _as_points(points, shape) -> FocalPoints:
    if isinstance(points, FocalPoints):
        return points
    pm = _cells_from_map(points, "points", shape)
    pm[pm == NODATA] = 0
    jj, ii = np.nonzero(pm.T)
    v = pm[ii, jj]
    keep = v >= 0
    order = np.argsort(v[keep], kind="stable")
    return FocalPoints((ii[keep] + 1)[order], (jj[keep] + 1)[order], v[keep][order].astype(np.int64))


# --- INI files --------------------------------------------------------------------------

DEFAULTS = {
    "version": "unknown", "connect_four_neighbors_only": "False", "connect_using_avg_resistances": "False",
    "use_polygons": "False", "polygon_file": "(Browse for a short-circuit region file)",
    "source_file": "(Browse for a current source file)", "ground_file": "(Browse for a ground point file)",
    "ground_file_is_resistances": "True", "use_unit_currents": "False", "use_direct_grounds": "False",
    "remove_src_or_gnd": "keepall", "use_mask": "False", "mask_file": "None", "data_type": "raster",
    "scenario": "not entered", "habitat_file": "(Browse for a resistance file)",
    "habitat_map_is_resistances": "True", "solver": "cg+amg", "parallelize": "False", "precision": "double",
    "use_variable_source_strengths": "False", "variable_source_file": "None", "use_included_pairs": "False",
    "included_pairs_file": "(Browse for a file with pairs to include or exclude)",
    "point_file": "(Browse for file with locations of focal points or regions)", "use_reclass_table": "False",
    "reclass_file": "(Browse for file with reclassification data)", "output_file": "(Choose a base name for output files)",
    "write_cur_maps": "False", "write_volt_maps": "False", "write_cum_cur_map_only": "False",
    "write_max_cur_maps": "False", "set_null_currents_to_nodata": "False", "set_null_voltages_to_nodata": "False",
    "set_focal_node_currents_to_zero": "False", "compress_grids": "False", "log_transform_maps": "False",
    "write_as_tif": "False",
}


def _unset(path: str) -> bool:
    return path == "" or path == "None" or path.startswith("(")


def _flag(cfg, key) -> bool:
    v = cfg[key]
    if v in TRUELIST:
        return True
    if v in FALSELIST:
        return False
    raise ValueError(f'{key} = "{v}": expected True or False')


def _resolve(name: str, base: Path) -> Path:
    # Circuitscape reads paths from the working directory; failing that, next to the INI file.
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    q = base / p
    return q if q.exists() else p


def _write_grid(prefix: str, values: np.ndarray, meta: RasterMeta, tif: bool) -> None:
    out = np.where(np.isnan(values), NODATA, values)
    if tif:
        import rasterio
        from rasterio.transform import Affine

        t = Affine(meta.transform[1], meta.transform[2], meta.transform[0], meta.transform[4], meta.transform[5],
                   meta.transform[3])
        with rasterio.open(prefix + ".tif", "w", driver="GTiff", width=out.shape[1], height=out.shape[0], count=1,
                           dtype="float64", nodata=NODATA, transform=t, crs=meta.crs or (meta.wkt or None),
                           compress="lzw") as ds:
            ds.write(out, 1)
        return
    with open(prefix + ".asc", "wb") as f:
        f.write(f"ncols         {meta.ncols}\nnrows         {meta.nrows}\n".encode())
        f.write(f"xllcorner     {meta.xllcorner!r}\nyllcorner     {meta.yllcorner!r}\n".encode())
        f.write(f"cellsize      {meta.cellsize!r}\nNODATA_value  -9999\n".encode())
        f.write(core().ascii_grid_rows(np.ascontiguousarray(out, dtype=np.float64)))
    if meta.wkt:
        Path(prefix + ".prj").write_text(meta.wkt)


def parse_config(path) -> dict[str, str]:
    """Circuitscape's parse_config: every line but ``[section]`` ones split at its first ``=``."""
    from .ini import parse_ini

    return parse_ini(path)


def circuitscape_ini(path, *, write_outputs: bool = True, n_threads: int = 0) -> CircuitscapeResult:
    """Run a Circuitscape INI file (raster data), as ``Circuitscape.compute(path)`` does.

    Paths are read from the working directory, or else next to the INI file.
    With ``write_outputs`` the maps go next to ``output_file`` as Circuitscape
    names them (``<base>_voltmap``, ``<base>_curmap``...; ASCII grids, or
    GeoTIFF with ``write_as_tif``) and ``output_file`` holds the configuration.
    ``solver``, ``precision`` and ``parallelize`` have no effect: every solve is
    exact, in double precision, with ``n_threads`` threads.
    """
    path = Path(path)
    base = path.resolve().parent
    cfg = {**DEFAULTS, **parse_config(path)}
    if cfg["data_type"] not in ("raster", "Raster"):
        raise NotImplementedError("only raster data are supported (data_type = raster)")
    scenario = SCENARIOS.get(cfg["scenario"])
    if scenario is None:
        raise ValueError(f'scenario = "{cfg["scenario"]}" is not one of {sorted(set(SCENARIOS.values()))}')
    for key in ("low_memory_mode", "preemptive_memory_release"):
        if cfg.get(key, "False") in TRUELIST:
            raise ValueError(f"{key} is not implemented in Circuitscape.jl; remove it or set it to False")

    def need(key):
        if _unset(cfg[key]):
            raise ValueError(f"{key} is not set")
        p = _resolve(cfg[key], base)
        if not p.is_file():
            raise FileNotFoundError(f'{key} = "{cfg[key]}" does not exist')
        return p

    habitat_path = need("habitat_file")
    reclass = read_reclass_table(need("reclass_file")) if _flag(cfg, "use_reclass_table") else None
    cells, meta = read_raster(habitat_path)
    g = habitat_conductance(cells, is_resistance=_flag(cfg, "habitat_map_is_resistances"), reclass=reclass)
    poly = None
    if _flag(cfg, "use_polygons"):
        pm = read_polymap(need("polygon_file"), meta, integer=True)
        poly = np.ascontiguousarray(pm.astype(np.int64))
    if _flag(cfg, "use_mask"):
        m = read_polymap(need("mask_file"), meta)
        g = g * np.where(m > 0, 1.0, 0.0)
        if g.sum() == 0:
            raise ValueError("Mask file deleted everything!")
    four = _flag(cfg, "connect_four_neighbors_only")
    avg = _flag(cfg, "connect_using_avg_resistances")
    policy = cfg["remove_src_or_gnd"]
    if policy == "not entered":  # Circuitscape 4's default
        policy = "keepall"
    if policy not in POLICIES:
        raise ValueError(f"remove_src_or_gnd must be one of {POLICIES}")

    if scenario == "advanced":
        src, gnd = _source_and_ground(need("source_file"), need("ground_file"), meta,
                                      _flag(cfg, "ground_file_is_resistances"))
        result = _advanced(g, poly, src, gnd, four, avg, policy, _flag(cfg, "use_unit_currents"),
                           _flag(cfg, "use_direct_grounds"), n_threads)
    elif scenario == "pairwise":
        points = read_point_map(need("point_file"), meta)
        pairs = read_included_pairs(need("included_pairs_file")) if _flag(cfg, "use_included_pairs") else None
        points, excluded = _apply_included_pairs(points, pairs)
        opts = _pair_options(_flag(cfg, "write_volt_maps"), _flag(cfg, "write_cur_maps"),
                             _flag(cfg, "write_cum_cur_map_only"), _flag(cfg, "write_max_cur_maps"),
                             _flag(cfg, "set_focal_node_currents_to_zero"), _flag(cfg, "set_null_currents_to_nodata"),
                             _flag(cfg, "set_null_voltages_to_nodata"), _flag(cfg, "log_transform_maps"))
        prefix, tif = _prefix(cfg) if write_outputs else (None, False)
        maps: dict = {}

        def emit(ci, cj, volt, cur):
            name = f"_{int(points.ids[ci])}_{int(points.ids[cj])}"
            for kind, grid in (("voltmap", volt), ("curmap", cur)):
                if grid is None:
                    continue
                if write_outputs:
                    _write_grid(f"{prefix}_{kind}{name}", grid, meta, tif)
                else:
                    maps[kind + name] = grid

        ids = np.unique(points.ids) if points.has_regions else points.ids

        def emit(ci, cj, volt, cur):  # noqa: F811 - map names from the ids solved
            name = f"_{int(ids[ci])}_{int(ids[cj])}"
            for kind, grid in (("voltmap", volt), ("curmap", cur)):
                if grid is None:
                    continue
                if write_outputs:
                    _write_grid(f"{prefix}_{kind}{name}", grid, meta, tif)
                else:
                    maps[kind + name] = grid

        res, r = _pairwise(g, poly, points, excluded, four, avg, opts, emit, n_threads)
        result = CircuitscapeResult(resistances=res, cum_current=r.get("cum"), max_current=r.get("max"), maps=maps)
    else:
        points = read_point_map(need("point_file"), meta)
        pairs = read_included_pairs(need("included_pairs_file")) if _flag(cfg, "use_included_pairs") else None
        strengths = (read_point_strengths(need("variable_source_file"))
                     if _flag(cfg, "use_variable_source_strengths") else None)
        cum_only = _flag(cfg, "write_cum_cur_map_only")
        opts = {"volt_maps": _flag(cfg, "write_volt_maps"),
                # advanced_kernel writes every point's current map with write_cum_cur_map_only too
                "cur_maps": _flag(cfg, "write_cur_maps") or cum_only,
                "cum": _flag(cfg, "write_cur_maps") or cum_only, "max": _flag(cfg, "write_max_cur_maps"),
                "focal_zero": _flag(cfg, "set_focal_node_currents_to_zero")}
        prefix, tif = _prefix(cfg) if write_outputs else (None, False)
        maps = {}

        def emit_point(n, volt, cur):
            for kind, grid in (("voltmap", volt), ("curmap", cur)):
                if grid is None:
                    continue
                if write_outputs:
                    _write_grid(f"{prefix}_{kind}_{n}", grid, meta, tif)
                else:
                    maps[f"{kind}_{n}"] = grid

        result = _one_to_all(g, poly, points, strengths, pairs, scenario == "one-to-all", four, avg, opts,
                             emit_point, n_threads)
        result.maps = maps

    if write_outputs:
        _write_outputs(cfg, scenario, result, meta, base)
    return result


def _prefix(cfg):
    out = cfg["output_file"]
    if _unset(out):
        raise ValueError("output_file is not set")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    return out.split(".out")[0], _flag(cfg, "write_as_tif")


def _write_matrix(path: str, a: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in np.atleast_2d(a):
            f.write(" ".join(repr(float(v)) for v in row) + "\n")


def _source_and_ground(source_path, ground_path, meta: RasterMeta, ground_is_res: bool):
    # read_source_and_ground_maps
    kind = file_type(ground_path)
    if kind in ("aagrid", "geotiff"):
        gnd = read_polymap(ground_path, meta, nodata_as=None)
    elif kind == "list":
        pts = _read_list(ground_path)
        gnd = np.full((meta.nrows, meta.ncols), NODATA)
        i, j = _list_cells(pts, meta)
        gnd[i - 1, j - 1] = pts[:, 0]
    else:
        raise ValueError("Cannot recognise file type.")
    kind = file_type(source_path)
    if kind in ("aagrid", "geotiff"):
        src = read_polymap(source_path, meta)
        src[src == NODATA] = 0.0
    elif kind == "list":
        pts = _read_list(source_path)
        src = np.zeros((meta.nrows, meta.ncols))
        i, j = _list_cells(pts, meta)
        src[i - 1, j - 1] = pts[:, 0]
    else:
        raise ValueError("Cannot recognize file type.")
    nod = gnd == NODATA
    if ground_is_res:
        with np.errstate(divide="ignore"):
            gnd = 1 / gnd
    gnd[nod] = 0.0
    return src, gnd


def _write_outputs(cfg, scenario, result: CircuitscapeResult, meta: RasterMeta, base: Path) -> None:
    out = cfg["output_file"]
    if _unset(out):
        raise ValueError("output_file is not set")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for k, v in cfg.items():
            f.write(f"{k} = {v}\n")
    prefix = out.split(".out")[0]
    tif = _flag(cfg, "write_as_tif")
    if scenario == "advanced":
        if _flag(cfg, "write_volt_maps"):
            _write_grid(prefix + "_voltmap", result.voltage, meta, tif)
        if _flag(cfg, "write_cur_maps") or _flag(cfg, "write_cum_cur_map_only"):
            _write_grid(prefix + "_curmap", result.current, meta, tif)
        return
    if result.cum_current is not None:
        _write_grid(prefix + "_cum_curmap", result.cum_current, meta, tif)
    if result.max_current is not None:
        _write_grid(prefix + "_max_curmap", result.max_current, meta, tif)
    if scenario == "pairwise":
        r = result.resistances
        _write_matrix(prefix + "_resistances.out", r)
        fp = r[1:, 0]
        three = [(fp[i], fp[j], r[j + 1, i + 1]) for i in range(len(fp)) for j in range(i + 1, len(fp))]
        _write_matrix(prefix + "_resistances_3columns.out", np.array(three).reshape(-1, 3))
