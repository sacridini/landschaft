# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleaning occurrence records: CoordinateCleaner's tests (Zizka et al. 2019)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core
from . import _circles, _points
from ._refdata import reference_data

__all__ = ["clean_occurrences"]

#: The tests in the order CoordinateCleaner runs them, and those run by default.
TESTS = ("invalid", "equal", "zeros", "capitals", "centroids", "seas", "urban", "countries", "outliers", "gbif",
         "institutions", "duplicates", "aohi")
DEFAULTS = {"equal": True, "zeros": 0.5, "capitals": 10_000, "centroids": 1_000, "seas": True, "urban": False,
            "countries": False, "outliers": True, "gbif": 1_000, "institutions": 100, "duplicates": False,
            "aohi": False}
_GBIF_HQ = (12.58, 55.67)


def _options(value, main: str, defaults: dict) -> dict | None:
    """A test's options from its argument: False / None (skip), True (defaults), a number (the main
    option), a dict (options) or an object (the reference data)."""
    if value is None or value is False:
        return None
    opts = dict(defaults)
    if value is True:
        return opts
    if isinstance(value, dict):
        unknown = set(value) - set(defaults)
        if unknown:
            raise ValueError(f"unknown options {sorted(unknown)}; expected some of {sorted(defaults)}")
        opts.update(value)
        return opts
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        opts[main] = float(value)
        return opts
    if "ref" in defaults:
        opts["ref"] = value
        return opts
    raise TypeError(f"cannot use {type(value).__name__} as options")


# ------------------------------------------------------------------------------------- point-in-polygon

def _covered(lon: np.ndarray, lat: np.ndarray, polygons) -> np.ndarray:
    """For each point, whether some polygon covers it (the boundary counts), as terra's extract."""
    import shapely

    out = np.zeros(lon.size, dtype=bool)
    polys = [p for p in polygons if p is not None and not p.is_empty]
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not polys or not ok.any():
        return out
    tree = shapely.STRtree(polys)
    idx = np.flatnonzero(ok)
    hit = tree.query(shapely.points(lon[idx], lat[idx]), predicate="covered_by")[0]
    out[idx[np.unique(hit)]] = True
    return out


def _covering_codes(lon, lat, polygons, codes) -> list[set]:
    import shapely

    tree = shapely.STRtree(list(polygons))
    sets = [set() for _ in range(lon.size)]
    ok = np.flatnonzero(np.isfinite(lon) & np.isfinite(lat))
    if ok.size:
        pi, gi = tree.query(shapely.points(lon[ok], lat[ok]), predicate="covered_by")
        for p, g in zip(ok[pi], gi):
            c = codes[g]
            if c is not None and not (isinstance(c, float) and np.isnan(c)):
                sets[p].add(str(c))
    return sets


def _near_points(lon, lat, ref_lon, ref_lat, buffer: float, geod: bool, crop: bool) -> np.ndarray:
    """Records within ``buffer`` metres of a reference point (cc_cap / cc_cen / cc_gbif / cc_inst / cc_aohi)."""
    ref_lon = np.asarray(ref_lon, dtype=float)
    ref_lat = np.asarray(ref_lat, dtype=float)
    keep = np.isfinite(ref_lon) & np.isfinite(ref_lat)
    ref_lon, ref_lat = ref_lon[keep], ref_lat[keep]
    if buffer == 0:
        buffer = 1e-11
    ok = np.isfinite(lon) & np.isfinite(lat)
    if crop and ok.any():
        ext = _circles.circle_extent(lon[ok], lat[ok], buffer)
        inside = (ref_lon >= ext[0]) & (ref_lon <= ext[1]) & (ref_lat >= ext[2]) & (ref_lat <= ext[3])
        ref_lon, ref_lat = ref_lon[inside], ref_lat[inside]
    if ref_lon.size == 0:
        return np.zeros(lon.size, dtype=bool)
    polys = _circles.circles(ref_lon, ref_lat, buffer, "geosphere" if geod else "terra")
    return _covered(lon, lat, polys)


def _verify(flag: np.ndarray, lon, lat, species) -> np.ndarray:
    """cc_cap / cc_cen / cc_gbif's ``verify``: keep a flag only where more flagged records (of any
    species) share the exact coordinates than its species has flagged records."""
    if not flag.any():
        return flag
    df = pd.DataFrame({"lon": lon, "lat": lat, "sp": species})
    ver = df[flag]
    coord = ver.groupby(["lon", "lat"], dropna=False).size().rename("coord")
    spc = ver.groupby("sp", dropna=False).size().rename("spc")
    m = df.join(coord, on=["lon", "lat"]).join(spc, on="sp").fillna({"coord": 0, "spc": 0})
    return flag & ~(m["coord"].to_numpy() <= m["spc"].to_numpy())


# ---------------------------------------------------------------------------------------------- tests

def _test_zeros(lon, lat, opts):
    import shapely

    r = opts["buffer"] if opts["buffer"] != 0 else 1e-14
    circle = shapely.buffer(shapely.Point(0.0, 0.0), r, quad_segs=10)  # planar degrees (GEOS)
    with np.errstate(invalid="ignore"):
        axis = (lon == 0) | (lat == 0)
    return axis | _covered(lon, lat, [circle])


def _test_capitals(lon, lat, opts, species):
    ref = opts["ref"]
    if ref is None:
        ref = reference_data("countryref")
        ref = ref[ref["capital"].notna()]
        rx, ry = ref["capital.lon"].to_numpy(float), ref["capital.lat"].to_numpy(float)
    else:
        rx, ry = _points.lonlat(ref)
    flag = _near_points(lon, lat, rx, ry, opts["buffer"], opts["geod"], crop=True)
    return _verify(flag, lon, lat, species) if opts["verify"] else flag


def _test_centroids(lon, lat, opts, species):
    ref = opts["ref"]
    detail = opts["detail"]
    if detail not in ("both", "country", "provinces"):
        raise ValueError("centroids detail must be 'both', 'country' or 'provinces'")
    if ref is None:
        ref = reference_data("countryref")
        if detail == "country":
            ref = ref[ref["type"] == "country"]
        elif detail == "provinces":
            ref = ref[ref["type"] == "province"]
        rx, ry = ref["centroid.lon"].to_numpy(float), ref["centroid.lat"].to_numpy(float)
    else:
        rx, ry = _points.lonlat(ref)
    flag = _near_points(lon, lat, rx, ry, opts["buffer"], opts["geod"], crop=True)
    return _verify(flag, lon, lat, species) if opts["verify"] else flag


def _test_gbif(lon, lat, opts, species):
    flag = _near_points(lon, lat, [_GBIF_HQ[0]], [_GBIF_HQ[1]], opts["buffer"], opts["geod"], crop=False)
    return _verify(flag, lon, lat, species) if opts["verify"] else flag


def _test_institutions(lon, lat, opts, species):
    ref = opts["ref"]
    if ref is None:
        ref = reference_data("institutions")
        rx, ry = ref["decimalLongitude"].to_numpy(float), ref["decimalLatitude"].to_numpy(float)
    else:
        rx, ry = _points.lonlat(ref)
    buffer = opts["buffer"] if opts["buffer"] != 0 else 1e-11
    flag = _near_points(lon, lat, rx, ry, buffer, opts["geod"], crop=False)
    if opts["verify"] and flag.any():
        idx = np.flatnonzero(flag)
        circles = _circles.circles(lon[idx], lat[idx], buffer * opts["verify_mltpl"],
                                   "geosphere" if opts["geod"] else "terra")
        sp = pd.Series(species)
        for c, i in zip(circles, idx):
            same = (sp == sp.iloc[i]).to_numpy()
            n_in = _covered(lon[same], lat[same], [c]).sum()
            n_same = (flag & same & (lon == lon[i]) & (lat == lat[i])).sum()
            flag[i] = not n_in > n_same
    return flag


def _test_aohi(lon, lat, opts, species):
    ref = reference_data("aohi")
    ref = ref[ref["taxa"].isin(list(opts["taxa"]))]
    return _near_points(lon, lat, ref["decimalLongitude"].to_numpy(float), ref["decimalLatitude"].to_numpy(float),
                        opts["buffer"], opts["geod"], crop=False)


def _test_seas(lon, lat, opts):
    ref = opts["ref"]
    if ref is None:
        ref = reference_data({110: "land110", 50: "land50", 10: "land10"}[int(opts["scale"])])
    polys = _polygons_lonlat(ref)
    # each distinct location once (CoordinateCleaner's speedup; the same result)
    xy = pd.DataFrame({"x": lon, "y": lat})
    first = ~xy.duplicated().to_numpy()
    land = np.zeros(lon.size, dtype=bool)
    land[first] = _covered(lon[first], lat[first], polys)
    land = pd.Series(land).groupby([xy.x, xy.y], dropna=False).transform("max").to_numpy(dtype=bool)
    ok = np.isfinite(lon) & np.isfinite(lat)
    return ok & ~land


def _test_urban(lon, lat, opts):
    ref = opts["ref"] if opts["ref"] is not None else reference_data("urban50")
    return _covered(lon, lat, _polygons_lonlat(ref))


def _test_countries(lon, lat, opts, table):
    col = opts["column"]
    if col is None or col not in table.columns:
        raise ValueError("the countries test needs the column of country codes: countries='countryCode' "
                         "or countries={'column': ...}")
    codes = table[col].astype("string").str.strip().str.upper().to_numpy(dtype=object, na_value=None)
    ref = opts["ref"] if opts["ref"] is not None else reference_data("countries50")
    ref_col = opts["ref_col"]
    if ref_col is None:  # two-letter codes (GBIF's countryCode) or three
        lengths = {len(c) for c in codes if c}
        ref_col = "iso_a2_eh" if lengths == {2} else "iso_a3_eh"
    if ref_col not in ref.columns:
        raise ValueError(f"the country polygons have no column {ref_col!r}")
    polys = _polygons_lonlat(ref)
    sets = _covering_codes(lon, lat, polys, ref[ref_col].to_numpy(dtype=object))
    return np.array([not (c is not None and c in s) for c, s in zip(codes, sets)], dtype=bool)


def _test_duplicates(lon, lat, species, table, opts):
    cols = {"lon": np.where(lon == 0, 0.0, lon), "lat": np.where(lat == 0, 0.0, lat), "sp": species}
    for c in opts["additions"] or ():
        cols[c] = table[c].to_numpy()
    return pd.DataFrame(cols).duplicated(keep="first").to_numpy()


def _r_quantile7(x: np.ndarray, probs) -> np.ndarray:
    """R's quantile(x, probs, type = 7): (1 - h) * x[lo] + h * x[hi]."""
    x = np.sort(x[~np.isnan(x)])
    n = x.size
    out = []
    for p in probs:
        index = 1 + max(n - 1, 0) * p
        lo, hi = int(np.floor(index)), int(np.ceil(index))
        q = x[lo - 1]
        if index > lo and x[hi - 1] != q:
            h = index - lo
            q = (1 - h) * q + h * x[hi - 1]
        out.append(q)
    return np.array(out)


def _r_median(x: np.ndarray) -> float:
    x = np.sort(x[~np.isnan(x)])
    n = x.size
    if n == 0:
        return np.nan
    half = (n + 1) // 2
    if n % 2 == 1:
        return float(x[half - 1])
    return _core.sdm_r_mean(np.ascontiguousarray(x[half - 1:half + 1]))


def _outlier_flags(lon, lat, stat_fn, opts) -> np.ndarray:
    """The records a species' statistic marks as outliers (cc_outl's thresholds)."""
    mins = stat_fn(opts["method"] == "distance")
    method = opts["method"]
    if method == "distance":
        return mins > opts["tdi"]
    if method == "quantile":
        q25, q75 = _r_quantile7(mins, (0.25, 0.75))
        return mins > (q75 + (q75 - q25) * opts["mltpl"])
    if method == "mad":
        med = _r_median(mins)
        mad = 1.4826 * _r_median(np.abs(mins - med))
        return mins > (med + mad * opts["mltpl"])
    raise ValueError("outliers method must be 'quantile', 'mad' or 'distance'")


def _raster_grid(lon_all, lat_all, res):
    """cc_outl's ras_create: extent of all records + 2 res (clamped to the globe), cells of about res."""
    ok = np.isfinite(lon_all) & np.isfinite(lat_all)
    xmin, xmax = lon_all[ok].min() - 2 * res, lon_all[ok].max() + 2 * res
    ymin, ymax = lat_all[ok].min() - 2 * res, lat_all[ok].max() + 2 * res
    xmin, xmax, ymin, ymax = max(xmin, -180.0), min(xmax, 180.0), max(ymin, -90.0), min(ymax, 90.0)

    def c_round(v):  # C's round(): half away from zero
        return np.floor(v + 0.5) if v >= 0 else np.ceil(v - 0.5)

    nc = max(1, int(c_round((xmax - xmin) / res)))
    nr = max(1, int(c_round((ymax - ymin) / res)))
    return xmin, xmax, ymin, ymax, nr, nc


def _test_outliers(lon, lat, species, opts):
    flag = np.zeros(lon.size, dtype=bool)
    ok = np.isfinite(lon) & np.isfinite(lat)
    sp = pd.Series(species, dtype=object)
    counts = sp[ok & sp.notna().to_numpy()].value_counts()
    groups = [s for s, c in counts.items() if c > opts["size"]]  # clean_coordinates: raw count > outliers_size
    grid = None
    for s in sorted(groups, key=str):
        idx = np.flatnonzero(ok & (sp == s).to_numpy())
        x, y = lon[idx], lat[idx]
        if len(pd.DataFrame({"x": x, "y": y}).drop_duplicates()) < opts["min_occs"]:
            continue
        raster = idx.size >= 10_000 or opts["thinning"]
        if not raster:
            fn = lambda use_min, x=x, y=y: _core.sdm_outlier_statistic(np.ascontiguousarray(x),  # noqa: E731
                                                                       np.ascontiguousarray(y), use_min)
            flag[idx] = _outlier_flags(x, y, fn, opts)
            continue
        if grid is None:
            grid = _raster_grid(lon, lat, opts["thinning_res"])
        xmin, xmax, ymin, ymax, nr, nc = grid
        col = np.where(x == xmax, nc - 1, np.floor((x - xmin) * (nc / (xmax - xmin)))).astype(np.int64)
        row = np.where(y == ymin, nr - 1, np.floor((ymax - y) * (nr / (ymax - ymin)))).astype(np.int64)
        cell = row * nc + col
        cells, first, inverse, cnt = np.unique(cell, return_index=True, return_inverse=True, return_counts=True)
        order = np.argsort(first, kind="stable")  # cells in order of first appearance
        cells, cnt = cells[order], cnt[order].astype(float)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        xres, yres = (xmax - xmin) / nc, (ymax - ymin) / nr
        cx = xmin + ((cells % nc) + 0.5) * xres
        cy = ymax - ((cells // nc) + 0.5) * yres
        if opts["thinning"]:
            fn = lambda use_min, cx=cx, cy=cy: _core.sdm_outlier_statistic(cx, cy, use_min)  # noqa: E731
        else:
            within = 100 * _core.sdm_r_mean(np.array([xres, yres])) / 2
            fn = lambda use_min, cx=cx, cy=cy, cnt=cnt, within=within: _core.sdm_outlier_statistic_weighted(  # noqa: E731
                cx, cy, cnt, within, use_min)
        cell_flag = _outlier_flags(cx, cy, fn, opts)
        flag[idx] = cell_flag[rank[inverse]]
    return flag


def _polygons_lonlat(ref):
    import geopandas as gpd

    if isinstance(ref, (gpd.GeoDataFrame, gpd.GeoSeries)):
        g = ref.geometry if isinstance(ref, gpd.GeoDataFrame) else ref
        if g.crs is not None and not g.crs.equals("EPSG:4326"):
            g = g.to_crs("EPSG:4326")
        return list(g.to_numpy())
    return list(ref)


# ------------------------------------------------------------------------------------------- the API

def clean_occurrences(occ, *, species: str | None = None, equal=True, zeros=0.5, capitals=10_000,
                      centroids=1_000, seas=True, urban=False, countries=False, outliers=True, gbif=1_000,
                      institutions=100, duplicates=False, aohi=False, action: str = "drop", x: str | None = None,
                      y: str | None = None, crs=None):
    """Flag or drop occurrence records with suspicious coordinates.

    The tests of CoordinateCleaner (Zizka et al. 2019, *Methods in Ecology
    and Evolution* 10: 744-751), the standard for GBIF data: records at
    country or province centroids and capitals (coordinates filled in from
    the name of a place), at biodiversity institutions (the address of a
    museum or herbarium) or GBIF's headquarters, at 0/0 or with equal
    longitude and latitude, in the sea, in towns, outside the country they
    claim, far from the other records of their species, or duplicated.
    Records without valid coordinates are always flagged (``invalid``).

    Each test takes ``False`` (skip it), ``True`` (its defaults), a number
    (its main option: a radius) or a dict of its options:

    ============== ========================================================================
    test           options (defaults)
    ============== ========================================================================
    equal          (none)
    zeros          ``buffer`` 0.5 (degrees around 0/0)
    capitals       ``buffer`` 10 000 m, ``geod`` True, ``ref`` None, ``verify`` False
    centroids      ``buffer`` 1 000 m, ``detail`` "both" / "country" / "provinces",
                   ``geod`` True, ``ref`` None, ``verify`` False
    seas           ``scale`` 50 (Natural Earth land, 110 / 50 / 10), ``ref`` None
    urban          ``ref`` None (Natural Earth urban areas)
    countries      ``column`` (the records' country codes, required), ``ref`` None,
                   ``ref_col`` None (``iso_a2_eh`` for two-letter codes, else ``iso_a3_eh``)
    outliers       ``method`` "quantile" / "mad" / "distance", ``mltpl`` 5, ``tdi`` 1000 km,
                   ``size`` 7 (species with more records are tested), ``min_occs`` 7
                   (distinct locations), ``thinning`` False, ``thinning_res`` 0.5
    gbif           ``buffer`` 1 000 m, ``geod`` True, ``verify`` False
    institutions   ``buffer`` 100 m, ``geod`` False, ``ref`` None, ``verify`` False,
                   ``verify_mltpl`` 10
    duplicates     ``additions`` None (more columns that must also be equal)
    aohi           ``buffer`` 10 000 m, ``geod`` True, ``taxa`` (Aves, Insecta, Mammalia, Plantae)
    ============== ========================================================================

    ``ref`` replaces the reference data (a table or GeoDataFrame of points, or
    of polygons for seas / urban / countries); passing it in place of the
    options also works (``seas=my_land``). The reference data are downloaded
    on first use and cached (see :func:`reference_data`).

    The results are CoordinateCleaner's (``clean_coordinates`` with the same
    options), except where it errs: the zeros radius is in degrees as
    documented (terra >= 1.8 made it 0.5 m), ``centroids`` "provinces"
    selects the provinces, circles that cross the antimeridian or hold a pole
    are drawn correctly, a record on a border between two countries is not
    misaligned, the country codes are matched against ``iso_a3_eh`` /
    ``iso_a2_eh`` (with ``iso_a3`` France and Norway are "-99" and all their
    records failed), species too small for the outlier test are not flagged,
    the outlier test also runs on species with 10 000 records or more, and
    institutions' ``verify`` works. Invalid coordinates are flagged instead of
    stopping.

    Parameters
    ----------
    occ
        Occurrence records: GeoDataFrame of points, or a table with
        coordinate columns (``decimalLongitude`` / ``decimalLatitude``,
        ``lon`` / ``lat``...).
    species
        Column with the species, for the outlier and duplicate tests and
        ``verify`` (default: ``species``, ``scientificName`` or ``label`` when
        present; else all records are one species).
    equal, zeros, capitals, centroids, seas, urban, countries, outliers, gbif, institutions, duplicates, aohi
        The tests and their options (see above).
    action : {"drop", "flag"}
        ``"drop"``: return the records that pass every test; ``"flag"``:
        return all of them with a ``flag_<test>`` column per test run (True =
        suspicious) and ``flagged`` (any).
    x, y, crs
        Coordinate columns and CRS of a table (default: found by name;
        longitude / latitude in EPSG:4326).

    Returns
    -------
    GeoDataFrame or DataFrame
        Same type and index as ``occ``. ``attrs["clean"]`` counts the records
        each test flagged.

    Examples
    --------
    >>> import landschaft as ls
    >>> occ = ls.load_table(ls.sdm.example_data("acaule"))
    >>> flags = ls.sdm.clean_occurrences(occ, action="flag", seas=False)
    >>> int(flags["flag_invalid"].sum()), len(ls.sdm.clean_occurrences(occ, seas=False))
    (284, 1050)
    """
    if action not in ("drop", "flag"):
        raise ValueError("action must be 'drop' or 'flag'")
    lon, lat = _points.lonlat(occ, x=x, y=y, crs=crs)
    lon = np.where(lon == 0, 0.0, lon)  # -0 is 0
    lat = np.where(lat == 0, 0.0, lat)
    with np.errstate(invalid="ignore"):
        invalid = ~(np.isfinite(lon) & np.isfinite(lat) & (lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90))
    vx = np.where(invalid, np.nan, lon)
    vy = np.where(invalid, np.nan, lat)
    table = occ if isinstance(occ, pd.DataFrame) else pd.DataFrame(occ)
    if species is None:
        species = next((c for c in ("species", "scientificName", "label") if c in table.columns), None)
    elif species not in table.columns:
        raise ValueError(f"no column {species!r}")
    sp = table[species].to_numpy(dtype=object) if species else np.full(len(table), "all", dtype=object)

    run = {
        "equal": _options(equal, "", {}),
        "zeros": _options(zeros, "buffer", {"buffer": 0.5}),
        "capitals": _options(capitals, "buffer", {"buffer": 10_000.0, "geod": True, "ref": None, "verify": False}),
        "centroids": _options(centroids, "buffer", {"buffer": 1_000.0, "detail": "both", "geod": True, "ref": None,
                                                    "verify": False}),
        "seas": _options(seas, "scale", {"scale": 50, "ref": None}),
        "urban": _options(urban, "", {"ref": None}),
        "countries": _options({"column": countries} if isinstance(countries, str) else countries, "",
                              {"column": None, "ref": None, "ref_col": None}),
        "outliers": _options(outliers, "mltpl", {"method": "quantile", "mltpl": 5.0, "tdi": 1000.0, "size": 7,
                                                 "min_occs": 7, "thinning": False, "thinning_res": 0.5}),
        "gbif": _options(gbif, "buffer", {"buffer": 1_000.0, "geod": True, "verify": False}),
        "institutions": _options(institutions, "buffer", {"buffer": 100.0, "geod": False, "ref": None,
                                                          "verify": False, "verify_mltpl": 10.0}),
        "duplicates": _options(duplicates, "", {"additions": None}),
        "aohi": _options(aohi, "buffer", {"buffer": 10_000.0, "geod": True,
                                          "taxa": ("Aves", "Insecta", "Mammalia", "Plantae")}),
    }
    flags = {"invalid": invalid}
    for name in TESTS[1:]:
        opts = run[name]
        if opts is None:
            continue
        if name == "equal":
            with np.errstate(invalid="ignore"):
                f = np.abs(vx) == np.abs(vy)
        elif name == "zeros":
            f = _test_zeros(vx, vy, opts)
        elif name == "capitals":
            f = _test_capitals(vx, vy, opts, sp)
        elif name == "centroids":
            f = _test_centroids(vx, vy, opts, sp)
        elif name == "seas":
            f = _test_seas(vx, vy, opts)
        elif name == "urban":
            f = _test_urban(vx, vy, opts)
        elif name == "countries":
            f = _test_countries(vx, vy, opts, table)
        elif name == "outliers":
            f = _test_outliers(vx, vy, sp, opts)
        elif name == "gbif":
            f = _test_gbif(vx, vy, opts, sp)
        elif name == "institutions":
            f = _test_institutions(vx, vy, opts, sp)
        elif name == "duplicates":
            f = _test_duplicates(vx, vy, sp, table, opts)
        else:
            f = _test_aohi(vx, vy, opts, sp)
        flags[name] = np.asarray(f, dtype=bool) & ~invalid
    any_flag = np.logical_or.reduce(list(flags.values()))
    counts = {k: int(v.sum()) for k, v in flags.items()}
    counts["total"] = int(any_flag.sum())
    if action == "flag":
        out = occ.copy()
        for k, v in flags.items():
            out[f"flag_{k}"] = v
        out["flagged"] = any_flag
    else:
        out = _points.subset(occ, ~any_flag)
    out.attrs["clean"] = counts
    return out
