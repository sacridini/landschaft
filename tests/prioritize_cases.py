# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases shared by tools/make_reference_prioritize.py and the ls.prioritize tests.

Every raster is a float64 array (NaN = NoData) written as a one-strip GeoTIFF of
``dtype`` (NaN becomes ``nodata`` for integer types), so Zonation 5 reads it in
one block of rows and sums the layers in row order. A case is a Zonation 5 run:
the feature list (the ``filename`` column plus ``columns``), the settings file
(values naming a raster become its path), extra link files, the command-line
switches, the marginal loss rule (``--mode``) and the quantization (8 or 16
bits, i.e. ``z5.exe`` or ``z5_16bit.exe``). ``params`` are the arguments of
``ls.prioritize.zonation`` giving the same run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GROUPS = ["core", "transform", "connectivity", "post", "impact"]

EPSG = 32633


@dataclass
class RasterSpec:
    values: np.ndarray  # (ny, nx) float64, NaN = NoData
    dtype: str = "float32"
    nodata: float | None = float("nan")  # written for NaN cells (and declared as the band's NoData)
    cellsize: float = 100.0


@dataclass
class Case:
    group: str
    layers: list[str]  # raster names, one feature per entry
    mode: str = "CAZ2"
    columns: dict[str, list] = field(default_factory=dict)  # feature-list columns besides filename
    settings: dict[str, object] = field(default_factory=dict)  # settings file entries (raster names -> paths)
    files: dict[str, str] = field(default_factory=dict)  # extra link files (setting value "file:NAME")
    flags: tuple[str, ...] = ()
    bits: int = 8
    extract: dict[int, float] = field(default_factory=dict)  # layer index -> value extracted ("f.tif/3")
    load: bool = True  # also evaluate Zonation's final rank map in LOAD mode (curves of the final ranking)
    params: dict = field(default_factory=dict)  # ls.prioritize.zonation arguments (besides features)
    extra: tuple[str, ...] = ()  # other rasters the link files name ("<raster>.tif")


# ---------------------------------------------------------------------------- rasters

def _blobs(shape, seed, n=6, zero_below=0.0, scale=1.0, na=None) -> np.ndarray:
    """A species-like surface: a sum of Gaussian blobs, zero below a level, NaN where ``na``."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    v = np.zeros(shape)
    for _ in range(n):
        cy, cx = rng.uniform(0, ny), rng.uniform(0, nx)
        s = rng.uniform(0.06, 0.25) * max(ny, nx)
        v += rng.uniform(0.3, 1.0) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    v = v / v.max() * scale
    v[v < zero_below] = 0.0
    if na is not None:
        v[na] = np.nan
    return v


def _outline(shape, seed) -> np.ndarray:
    """NoData outside an irregular 'coastline' (True = NoData)."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    r = np.hypot((yy - ny / 2) / (ny / 2), (xx - nx / 2) / (nx / 2))
    wobble = 0.12 * np.sin(np.arctan2(yy - ny / 2, xx - nx / 2) * 5 + rng.uniform(0, 6))
    return r > 0.95 + wobble


def _rasters() -> dict[str, RasterSpec]:
    r: dict[str, RasterSpec] = {}
    shape = (40, 50)
    sea = _outline(shape, 1)
    for i in range(8):
        r[f"sp{i}"] = RasterSpec(_blobs(shape, 10 + i, n=3 + i % 4, zero_below=0.05 * (i % 3), na=sea))
    # presence/absence (many ties), integer counts with a numeric NoData, a layer with its own holes
    r["pa0"] = RasterSpec(np.where(sea, np.nan, (_blobs(shape, 30) > 0.35).astype(float)), "uint8", 255)
    r["pa1"] = RasterSpec(np.where(sea, np.nan, (_blobs(shape, 31) > 0.5).astype(float)), "uint8", 255)
    counts = np.round(_blobs(shape, 32, scale=20.0))
    counts[sea] = np.nan
    r["count"] = RasterSpec(counts, "int16", -9999)
    holes = _blobs(shape, 33, na=sea)
    holes[np.random.default_rng(34).random(shape) < 0.1] = np.nan
    r["holes"] = RasterSpec(holes)
    # a land-cover map (classes 1-4) for value extraction
    lc = np.floor(_blobs(shape, 35, n=8) * 3.999) + 1
    lc[sea] = np.nan
    r["landcover"] = RasterSpec(lc, "uint8", 0)
    # cost (positive, with some zero-cost cells), hierarchic mask (0/1/2), analysis areas (0 outside, 1-3)
    cost = 1.0 + 9.0 * _blobs(shape, 40, n=5)
    cost[_blobs(shape, 41) > 0.8] = 0.0
    cost[sea] = np.nan
    r["cost"] = RasterSpec(cost)
    hm = np.zeros(shape)
    hm[_blobs(shape, 42) > 0.55] = 1
    hm[_blobs(shape, 43) > 0.75] = 2
    r["hmask"] = RasterSpec(hm, "int32", None)
    areas = np.zeros(shape)
    areas[:, :18] = 1
    areas[:, 18:35] = 2
    areas[:, 35:] = 3
    areas[:4] = 0
    r["areas"] = RasterSpec(areas, "int32", None)
    # conditions (0-1, NoData outside the land), retention fractions and amounts
    for k, seed in (("cond1", 50), ("cond2", 51), ("retrel", 52)):
        v = 0.15 + 0.85 * _blobs(shape, seed, n=5)
        v[sea] = np.nan
        r[k] = RasterSpec(v)
    ra = 0.4 * _blobs(shape, 53, n=4)
    ra[sea] = np.nan
    r["retabs"] = RasterSpec(ra)
    # connectivity: a grid spanning several FFT blocks, and resource rasters
    cshape = (200, 300)
    csea = _outline(cshape, 3)
    for i in range(5):
        r[f"conn{i}"] = RasterSpec(_blobs(cshape, 200 + i, n=5 + i, zero_below=0.15 * (i % 2), na=csea))
    for k, seed in (("res1", 210), ("res2", 211)):
        v = _blobs(cshape, seed, n=6)
        v[csea] = np.nan
        r[k] = RasterSpec(v)
    # impact assessment: sources (roads, a mine, a town), subareas, loss rasters and pressures
    src = np.zeros(shape)
    src[20, 5:45] = 1          # a road
    src[5:30, 33] = 1
    src[10:14, 10:15] = 2      # a mine
    src[28:32, 20:26] = 3      # a town
    r["isrc"] = RasterSpec(src, "uint16", None)
    sub = np.zeros(shape)
    sub[:, :25] = 1
    sub[:22, 25:] = 2
    sub[30:, 25:] = 3
    r["subareas"] = RasterSpec(sub, "uint16", None)
    for k, seed in (("dloss1", 60), ("dloss2", 61)):
        v = _blobs(shape, seed, n=3)
        v[v < 0.4] = 0
        r[k] = RasterSpec(v)
    for k, seed, sc in (("press1", 62, 5.0), ("press2", 63, 1.0)):
        r[k] = RasterSpec(_blobs(shape, seed, n=4, scale=sc))
    # a larger stack for realistic numbers of cells and iterations
    big = (120, 160)
    bsea = _outline(big, 2)
    for i in range(12):
        r[f"big{i}"] = RasterSpec(_blobs(big, 100 + i, n=4 + i % 5, zero_below=0.1 * (i % 2), na=bsea))
    return r


RASTERS = _rasters()

SP = [f"sp{i}" for i in range(8)]
BIG = [f"big{i}" for i in range(12)]
W8 = [2.0, 1.0, 1.0, 3.0, 0.5, 1.0, 1.5, 1.0]


def _core() -> dict[str, Case]:
    c: dict[str, Case] = {}
    g = "core"
    for mode in ["CAZ1", "CAZ2", "CAZMAX", "ABF 0.25", "ABF 0.6", "CAZP 2", "CAZP 0.5", "CAZPW 3"]:
        name = mode.lower().replace(" ", "_").replace(".", "")
        rule, *p = mode.split()
        c[name] = Case(g, SP, mode, params={"rule": rule, **({"rule_param": float(p[0])} if p else {})})
    c["caz2_16bit"] = Case(g, SP, "CAZ2", bits=16, params={"rule": "CAZ2", "bits": 16})
    c["abf_16bit"] = Case(g, SP, "ABF 0.25", bits=16, params={"rule": "ABF", "bits": 16})
    c["caz2_weights"] = Case(g, SP, "CAZ2", {"weight": W8}, flags=("-w",), params={"weights": W8})
    wneg = [2.0, 1.0, -1.0, 3.0, 0.0, 1.0, -0.5, 1.0]
    for mode in ["CAZ1", "CAZ2", "CAZMAX", "ABF 0.25", "CAZPW 2"]:
        name = mode.lower().replace(" ", "_").replace(".", "")
        rule, *p = mode.split()
        c[f"{name}_negweights"] = Case(g, SP, mode, {"weight": wneg}, flags=("-w",),
                                       params={"rule": rule, "weights": wneg,
                                               **({"rule_param": float(p[0])} if p else {})})
    wgrp = [1, 1, 2, 2, 2, 3, -1, 3]
    c["caz2_weightgroups"] = Case(g, SP, "CAZ2", {"weight": W8, "wgrp": wgrp},
                                  settings={"weight groups file": "file:wgroups.txt"},
                                  files={"wgroups.txt": "1 4.0\n2 1.0\n3 2.5\n"}, flags=("-W",),
                                  params={"weights": W8, "weight_groups": wgrp,
                                          "group_weights": {1: 4.0, 2: 1.0, 3: 2.5}})
    grp = [1, 1, 2, 2, 2, 3, 3, 3]
    c["caz2_groups"] = Case(g, SP, "CAZ2", {"weight": W8, "group": grp}, flags=("-w", "-g"),
                            params={"weights": W8, "groups": grp})
    thr = [0.0, 0.2, 0.0, 0.5, 0.1, 0.0, 0.3, 0.0]
    c["caz2_threshold"] = Case(g, SP, "CAZ2", {"threshold": thr}, flags=("-t",), params={"thresholds": thr})
    abfz = [0.25, 0.5, 0.35, 0.1, 0.25, 0.8, 0.15, 0.3]
    c["abf_layer_z"] = Case(g, SP, "ABF 0.35", {"abfz": abfz},
                            params={"rule": "ABF", "rule_param": 0.35, "abf_z": abfz})
    c["caz2_zero_none"] = Case(g, SP, "CAZ2", settings={"zero mode": "none"}, params={"zero_mode": "none"})
    keep = [1, 0, 0, 1, 0, 0, 1, 0]
    c["caz2_zero_dynamic"] = Case(g, SP, "CAZ2", {"keep_zeros": keep}, settings={"zero mode": "dynamic"},
                                  params={"zero_mode": "dynamic", "keep_zeros": keep})
    mixed = ["sp0", "pa0", "count", "holes", "sp3", "pa1"]
    c["caz2_mixed"] = Case(g, mixed, "CAZ2")
    c["abf_mixed"] = Case(g, mixed, "ABF 0.25", params={"rule": "ABF"})
    c["caz1_presence"] = Case(g, ["pa0", "pa1"], "CAZ1", params={"rule": "CAZ1"})
    c["caz2_extract"] = Case(g, ["landcover", "landcover", "landcover", "sp1"], "CAZ2",
                             extract={0: 1.0, 1: 2.0, 2: 4.0}, params={"extract": [1.0, 2.0, 4.0, None]})
    c["caz2_cost_tracked"] = Case(g, SP, "CAZ2", settings={"cost layer": "cost"}, flags=("-x",),
                                  params={"cost": "cost", "cost_in_ranking": False})
    c["caz2_cost"] = Case(g, SP, "CAZ2", settings={"cost layer": "cost"}, flags=("-X",),
                          params={"cost": "cost"})
    c["abf_cost_negweights"] = Case(g, SP, "ABF 0.25", {"weight": wneg}, settings={"cost layer": "cost"},
                                    flags=("-w", "-X"),
                                    params={"rule": "ABF", "weights": wneg, "cost": "cost"})
    c["caz2_hmask"] = Case(g, SP, "CAZ2", settings={"hierarchic mask layer": "hmask"}, flags=("-h",),
                           params={"hierarchic_mask": "hmask"})
    c["caz2_hmask_cost"] = Case(g, SP, "CAZ2", {"weight": W8},
                                settings={"hierarchic mask layer": "hmask", "cost layer": "cost"},
                                flags=("-w", "-h", "-X"),
                                params={"weights": W8, "hierarchic_mask": "hmask", "cost": "cost"})
    c["caz2_area_all"] = Case(g, SP, "CAZ2", settings={"analysis area mask layer": "areas"}, flags=("-a",),
                              params={"analysis_area": "areas"})
    c["caz2_area_2"] = Case(g, SP, "CAZ2", settings={"analysis area mask layer": "areas"},
                            flags=("-a", "--area=2"), params={"analysis_area": "areas", "area": 2})
    c["caz2_smoothing"] = Case(g, SP, "CAZ2", settings={"smoothing factor": 0.5},
                               params={"smoothing_factor": 0.5})
    c["caz2_maxiter"] = Case(g, SP, "CAZ2", settings={"max iterations": 3}, params={"max_iterations": 3})
    c["caz2_error"] = Case(g, SP, "CAZ2", settings={"mean ranking error threshold": 0.2, "stop on plateau": 1},
                           params={"ranking_error": 0.2, "stop_on_plateau": True})
    c["caz2_curves"] = Case(g, SP, "CAZ2", settings={"curves output interval": 0.01},
                            params={"curves_interval": 0.01})
    c["load"] = Case(g, SP, "LOAD", settings={"external solution": "sp7"}, load=False,
                     params={"rule": "LOAD", "solution": "sp7"})
    c["big_caz2"] = Case(g, BIG, "CAZ2")
    c["big_abf"] = Case(g, BIG, "ABF 0.25", params={"rule": "ABF"})
    c["big_caz1_weights"] = Case(g, BIG, "CAZ1", {"weight": [1.0 + (i % 4) for i in range(12)]}, flags=("-w",),
                                 params={"rule": "CAZ1", "weights": [1.0 + (i % 4) for i in range(12)]})
    return c


def _transform() -> dict[str, Case]:
    c: dict[str, Case] = {}
    g = "transform"
    cond_links = "1 0 cond1.tif\n2 0 cond2.tif\n3 1 cond1.tif\n"
    cond_ids = [1, -1, 2, 1, 3, -1, 2, 3]
    cond_rasters = ["cond1", None, "cond2", "cond1", "cond1", None, "cond2", "cond1"]
    renorm = [False, False, False, False, True, False, False, True]
    c["caz2_condition"] = Case(g, SP, "CAZ2", {"condition": cond_ids},
                               settings={"condition link file": "file:conditions.txt"},
                               files={"conditions.txt": cond_links}, flags=("-c",), extra=("cond1", "cond2"),
                               params={"condition": cond_rasters, "renormalize": renorm})
    c["abf_condition_16bit"] = Case(g, SP, "ABF 0.25", {"condition": cond_ids},
                                    settings={"condition link file": "file:conditions.txt"},
                                    files={"conditions.txt": cond_links}, flags=("-c",), bits=16,
                                    extra=("cond1", "cond2"),
                                    params={"rule": "ABF", "bits": 16, "condition": cond_rasters,
                                            "renormalize": renorm})
    ret_links = "1 2 0.3\n2 1 retrel.tif\n3 0 retabs.tif\n4 2 0.75\n"
    ret_ids = [1, -1, 2, 3, 4, 2, -1, 3]
    ret = [0.3, None, "retrel", ("absolute", "retabs"), 0.75, "retrel", None, ("absolute", "retabs")]
    c["caz2_retention"] = Case(g, SP, "CAZ2", {"retention": ret_ids},
                               settings={"retention link file": "file:retention.txt"},
                               files={"retention.txt": ret_links}, flags=("-r",), extra=("retrel", "retabs"),
                               params={"retention": ret})
    c["abf_retention_weights"] = Case(g, SP, "ABF 0.25", {"weight": W8, "retention": ret_ids},
                                      settings={"retention link file": "file:retention.txt"},
                                      files={"retention.txt": ret_links}, flags=("-w", "-r"),
                                      extra=("retrel", "retabs"),
                                      params={"rule": "ABF", "weights": W8, "retention": ret})
    c["caz1_condition_retention_cost"] = Case(
        g, SP, "CAZ1", {"weight": W8, "condition": cond_ids, "retention": ret_ids},
        settings={"condition link file": "file:conditions.txt", "retention link file": "file:retention.txt",
                  "cost layer": "cost"},
        files={"conditions.txt": cond_links, "retention.txt": ret_links}, flags=("-w", "-c", "-r", "-X"),
        extra=("cond1", "cond2", "retrel", "retabs"),
        params={"rule": "CAZ1", "weights": W8, "condition": cond_rasters, "renormalize": renorm, "retention": ret,
                "cost": "cost"})
    wneg = [2.0, 1.0, -1.0, 3.0, 0.0, 1.0, -0.5, 1.0]
    grp = [1, 1, 2, 2, 2, 3, 3, 3]
    tr = [1, 0, 1, 1, 0, 0, 1, 0]
    c["caz2_preprocessing"] = Case(g, SP, "CAZ2", {"weight": wneg, "group": grp, "tr_out": tr},
                                   flags=("-w", "-g", "-b", "-f"),
                                   params={"weights": wneg, "groups": grp, "preprocessing": True,
                                           "transformed": [bool(x) for x in tr]})
    c["caz2_preprocessing_plain"] = Case(g, SP, "CAZ2", {"tr_out": [1] * 8}, flags=("-b", "-f"),
                                         params={"preprocessing": True, "transformed": True})
    c["caz2_preprocessing_all"] = Case(
        g, SP, "CAZ2", {"weight": wneg, "group": grp, "tr_out": tr, "condition": cond_ids, "retention": ret_ids},
        settings={"condition link file": "file:conditions.txt", "retention link file": "file:retention.txt"},
        files={"conditions.txt": cond_links, "retention.txt": ret_links},
        flags=("-w", "-g", "-b", "-f", "-c", "-r"), extra=("cond1", "cond2", "retrel", "retabs"),
        params={"weights": wneg, "groups": grp, "preprocessing": True, "transformed": [bool(x) for x in tr],
                "condition": cond_rasters, "renormalize": renorm, "retention": ret})
    c["big_abf_retention"] = Case(g, BIG, "ABF 0.25", {"retention": [1, -1, 1, -1, 4, -1, 1, -1, 4, -1, -1, 1]},
                                  settings={"retention link file": "file:retention.txt"},
                                  files={"retention.txt": "1 2 0.2\n4 2 0.5\n"}, flags=("-r",),
                                  params={"rule": "ABF", "retention": [0.2, None, 0.2, None, 0.5, None, 0.2, None,
                                                                       0.5, None, None, 0.2]})
    return c


CONN = [f"conn{i}" for i in range(5)]


def _connectivity() -> dict[str, Case]:
    from landschaft.prioritize import Interaction, Kernel

    c: dict[str, Case] = {}
    g = "connectivity"
    sf_links = "1 0 negexp(4, 3)|2 1 gaussian(3, 2)|3 0 uniform(5)|4 1 triangular(6)|5 0 negexp(40, 3)|"
    sf_links = sf_links.replace("|", chr(10))
    kern = {1: Kernel("negexp", 4, 3), 2: Kernel("gaussian", 3, 2), 3: Kernel("uniform", 5),
            4: Kernel("triangular", 6), 5: Kernel("negexp", 40, 3)}
    selfm = {1: False, 2: True, 3: False, 4: True, 5: False}
    sf = [1, 2, 3, 4, 5]
    sf_params = {"connectivity": [kern[i] for i in sf], "self_multiply": [selfm[i] for i in sf],
                 "transformed": True}
    c["conn_sf"] = Case(g, CONN, "CAZ2", {"sf_conn": sf, "tr_out": [1] * 5},
                        settings={"single-feature connectivity link file": "file:sf.txt"},
                        files={"sf.txt": sf_links}, flags=("-d", "-f"), params=sf_params)
    c["conn_sf_abf_16bit"] = Case(g, CONN, "ABF 0.25", {"sf_conn": sf, "tr_out": [1] * 5},
                                  settings={"single-feature connectivity link file": "file:sf.txt"},
                                  files={"sf.txt": sf_links}, flags=("-d", "-f"), bits=16,
                                  params={**sf_params, "rule": "ABF", "bits": 16})
    small = [2, -1, 1, 3, -1, 4, 1, -1]
    c["conn_sf_small"] = Case(g, SP, "CAZ2", {"sf_conn": small, "tr_out": [1] * 8},
                              settings={"single-feature connectivity link file": "file:sf.txt"},
                              files={"sf.txt": sf_links}, flags=("-d", "-f"),
                              params={"connectivity": [kern.get(i) for i in small],
                                      "self_multiply": [selfm.get(i, False) for i in small], "transformed": True})
    ia_links = ("1 1 1.0 1 negexp(3, 4) res1.tif|2 0 0.8 1 gaussian(2, 3) res2.tif|"
                "3 0 0.5 0 negexp(5, 2) res1.tif|").replace("|", chr(10))
    ia = {1: ("res1", Kernel("negexp", 3, 4), True, 1.0, True), 2: ("res2", Kernel("gaussian", 2, 3), False, 0.8, True),
          3: ("res1", Kernel("negexp", 5, 2), False, 0.5, False)}
    ia_ids = [1, -1, 2, 3, -1]
    c["conn_interaction"] = Case(g, CONN, "CAZ2", {"ia_conn1": ia_ids, "tr_out": [1] * 5},
                                 settings={"interaction connectivity link file": "file:ia.txt"},
                                 files={"ia.txt": ia_links}, flags=("-i", "-f"), extra=("res1", "res2"),
                                 params={"interaction": [IA(ia[i]) if i in ia else None for i in ia_ids],
                                         "transformed": True})
    ia2_ids = [-1, 2, 1, -1, 3]
    c["conn_sf_interaction2"] = Case(g, CONN, "CAZ1", {"sf_conn": sf, "ia_conn2": ia2_ids, "tr_out": [1] * 5},
                                     settings={"single-feature connectivity link file": "file:sf.txt",
                                               "interaction connectivity link file": "file:ia.txt"},
                                     files={"sf.txt": sf_links, "ia.txt": ia_links}, flags=("-d", "-I", "-f"),
                                     extra=("res1", "res2"),
                                     params={**sf_params, "rule": "CAZ1",
                                             "interaction2": [IA(ia[i]) if i in ia else None for i in ia2_ids]})
    m_links = "1 negexp(3, 4) matrix.txt|2 gaussian(2, 3) matrix.txt|".replace("|", chr(10))
    matrix = "0.8 0.1 0.3|0.2 0.7 0.0|0.0 0.2 1.0|".replace("|", chr(10))
    m_ids = [1, 1, -1, 2, -1]
    mk = {1: Kernel("negexp", 3, 4), 2: Kernel("gaussian", 2, 3)}
    c["conn_matrix"] = Case(g, CONN, "CAZ2", {"matrix_conn": m_ids, "tr_out": [1] * 5},
                            settings={"matrix connectivity link file": "file:m.txt"},
                            files={"m.txt": m_links, "matrix.txt": matrix}, flags=("-m", "-f"),
                            params={"matrix_connectivity": [(mk[i], "M") if i in mk else None for i in m_ids],
                                    "matrices": {"M": np.array([[0.8, 0.1, 0.3], [0.2, 0.7, 0.0], [0.0, 0.2, 1.0]])},
                                    "transformed": True})
    return c


def IA(t):
    """An Interaction with its resource as a raster name (the tests give its path)."""
    from landschaft.prioritize import Interaction

    res, kern, pos, beta, norm = t
    return Interaction(kern, res, pos, beta, norm)


PP_CALLS = [
    'area_statistics_external("areas.tif", "ext")',
    'area_statistics_rank_range(0.8, 1.0, "top")',
    'area_statistics_separate_areas(0.7, 1.0, 5, "sep")',
    'solution_comparison(0.8, 1.0, "sp7.tif", 0.5, 1.0, "cmp")',
    'histogram(0.8, 1.0, 10, "h")',
    'histogram(0.0, 1.0, 7, "")',
    'flexibility_map(2.0, true, true, true, "flex")',
    'flexibility_map(10.0, true, false, false, "wide")',
]


def _post() -> dict[str, Case]:
    c: dict[str, Case] = {}
    g = "post"
    pp = chr(10).join(PP_CALLS) + chr(10)
    base = {"post-processing file": "file:pp.txt"}
    wneg = [2.0, 1.0, -1.0, 3.0, 0.0, 1.0, -0.5, 1.0]
    grp = [1, 1, 2, 2, 2, 3, 3, 3]
    c["post_presence"] = Case(g, ["pa0", "pa1"], "CAZ1", settings=base, files={"pp.txt": pp}, flags=("-p",),
                              extra=("areas", "sp7"), params={"rule": "CAZ1"})
    c["post_caz2"] = Case(g, SP, "CAZ2", {"weight": wneg, "group": grp}, settings=base, files={"pp.txt": pp},
                          flags=("-w", "-g", "-p"), extra=("areas",), params={"weights": wneg, "groups": grp})
    c["post_abf_cost_hmask"] = Case(g, SP, "ABF 0.25", {"weight": W8},
                                    settings={**base, "cost layer": "cost", "hierarchic mask layer": "hmask"},
                                    files={"pp.txt": pp}, flags=("-w", "-X", "-h", "-p"), extra=("areas",),
                                    params={"rule": "ABF", "weights": W8, "cost": "cost", "hierarchic_mask": "hmask"})
    c["post_cazp"] = Case(g, SP, "CAZP 0.5", settings=base, files={"pp.txt": pp}, flags=("-p",), extra=("areas",),
                          params={"rule": "CAZP", "rule_param": 0.5})
    return c


def _impact() -> dict[str, Case]:
    c: dict[str, Case] = {}
    g = "impact"
    nl = chr(10)
    buffers = nl.join(["1 1 100 2 50 5 20", "1 2 80 3 30", "2 1 60 4 25", "2 3 90", ""])
    bins = nl.join(["1 0.1 0.5 1.0", "2 0.3", ""])
    scenario = nl.join(["project id = scen1", "area unit name = ha", "impact unit name = ind",
                        "area conversion factor = 0.25", "subareas raster file = subareas.tif",
                        "output intermediate raster maps = 1", "impact source raster file = isrc.tif", ""])
    cols = {"impact_grp": [1, 1, 2, -1, 2, 1, -1, 2], "impact_bins": [0, -1, 1, -1, 2, 0, -1, -1]}
    files = {"buffers.txt": buffers, "bins.txt": bins, "scenario.txt": scenario}
    settings = {"impact assessment scenario file": "file:scenario.txt", "impact buffers file": "file:buffers.txt",
                "binning link file": "file:bins.txt"}
    extra = ("isrc", "subareas")
    c["imp_source"] = Case(g, SP, "CAZ2", cols, settings=settings, files=files, flags=("-s",), extra=extra)
    c["imp_source_weights_abf"] = Case(g, SP, "ABF 0.25", {**cols, "weight": W8}, settings=settings, files=files,
                                       flags=("-w", "-s"), extra=extra)

    def cum_files(direct, cumul, agg):
        f = {
            "direct.txt": nl.join(["filename", "dloss1.tif", "dloss2.tif", ""]),
            "dmatrix.txt": nl.join(["1 0.5", "0 1", ""]),
            "pressures.txt": nl.join(["normalize offset divisor filename", "1 0 1 press1.tif", "0 0.1 2 press2.tif",
                                      ""]),
            "cmatrix.txt": nl.join(["1 0.5", "0.3 1", ""]),
            "pf1.txt": nl.join(["0 0", "0.5 0.2", "1 0.6", ""]),
            "pf2.txt": nl.join(["0.1 0", "0.4 0.5", "0.9 0.9", ""]),
        }
        lines = ["project id = cum1", "area unit name = sqkm", "impact unit name = kg", "area conversion factor = 2",
                 "subareas raster file = subareas.tif", "output intermediate raster maps = 1",
                 f"analyze direct habitat loss = {direct}", "direct habitat loss raster list = direct.txt",
                 "direct habitat loss link matrix = dmatrix.txt", f"analyze cumulative pressures = {cumul}",
                 "cumulative pressures raster list = pressures.txt", f"individual pressure aggregation = {agg}",
                 "cumulative pressures link matrix = cmatrix.txt", "", "[cumulative pressure functions]"]
        if cumul == 1:
            lines += ["group 1 = pf1.txt", "group 2 = pf2.txt"]
        else:
            lines += ["group 1 = pf1.txt", "group 1 pressure 2 = pf2.txt", "group 2 = pf2.txt"]
        f["cumulative.txt"] = nl.join(lines + [""])
        return f

    cset = {"cumulative impact assessment scenario file": "file:cumulative.txt"}
    cextra = ("subareas", "dloss1", "dloss2", "press1", "press2")
    c["imp_cumulative_v1_sum"] = Case(g, SP, "CAZ2", cols, settings=cset, files=cum_files(1, 1, 0), flags=("-e",),
                                      extra=cextra)
    c["imp_cumulative_v2_product_multiply"] = Case(g, SP, "ABF 0.25", cols, settings=cset,
                                                   files=cum_files(2, 2, 1), flags=("-e",), extra=cextra)
    c["imp_both_v2_max"] = Case(g, SP, "CAZ1", {**cols, "weight": W8}, settings={**settings, **cset},
                                files={**files, **cum_files(1, 2, 2)}, flags=("-w", "-s", "-e"),
                                extra=extra + cextra)
    return c


CASES: dict[str, Case] = {**_core(), **_transform(), **_connectivity(), **_post(), **_impact()}
