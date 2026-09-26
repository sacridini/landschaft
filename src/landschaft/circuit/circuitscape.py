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
    poly = None
    if polygons is not None:
        pm = _cells_from_map(polygons, "polygons", g.shape)
        pm[pm == NODATA] = 0
        if not np.all(pm == np.round(pm)):
            raise ValueError("the polygon raster must hold integer ids")
        poly = np.ascontiguousarray(pm.astype(np.int64))
    if mask is not None:
        m = _cells_from_map(mask, "mask", g.shape)
        m[m == NODATA] = 0
        g = g * np.where(m > 0, 1.0, 0.0)
        if g.sum() == 0:
            raise ValueError("Mask file deleted everything!")
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
    with open(prefix + ".asc", "w", encoding="utf-8") as f:
        f.write(f"ncols         {meta.ncols}\nnrows         {meta.nrows}\n")
        f.write(f"xllcorner     {meta.xllcorner!r}\nyllcorner     {meta.yllcorner!r}\n")
        f.write(f"cellsize      {meta.cellsize!r}\nNODATA_value  -9999\n")
        for row in out:
            f.write(" ".join(repr(float(v)) for v in row) + "\n")
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
    if policy not in POLICIES:
        raise ValueError(f"remove_src_or_gnd must be one of {POLICIES}")

    if scenario == "advanced":
        src, gnd = _source_and_ground(need("source_file"), need("ground_file"), meta,
                                      _flag(cfg, "ground_file_is_resistances"))
        result = _advanced(g, poly, src, gnd, four, avg, policy, _flag(cfg, "use_unit_currents"),
                           _flag(cfg, "use_direct_grounds"), n_threads)
    else:
        raise NotImplementedError(f"scenario = {scenario} is not implemented yet")

    if write_outputs:
        _write_outputs(cfg, scenario, result, meta, base)
    return result


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
