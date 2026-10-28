# SPDX-License-Identifier: GPL-3.0-or-later
"""Coastal vulnerability (ls.services.coastal_vulnerability) against InVEST (tools/make_reference_invest_coastal.py).

The references are InVEST with two of its bugs corrected (the tool's doc). Everything is
compared bit for bit but the fetch: a ray that crosses two or more coast edges ends, in
InVEST, where its shortening in the R-tree's order leaves it (its last bits, ~1e-8 m),
here at the nearest edge; REI and the local waves carry that. The lon / lat case's fetch is
projected by GDAL's PROJ (rasterio's), the references made on Windows with PROJ 9.8: off
Windows (the system's libm) or with another PROJ (rasterio 1.4's wheels, on Python 3.10 and
3.11) its coordinates move by an ulp, so there ``conftest.assert_ucrt_close``'s tolerance.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
from landschaft.services._coastal import SECTORS

REF = Path(__file__).parent / "data" / "reference" / "invest"
CASES = ["coastal_full", "coastal_required", "coastal_lonlat"]


def _load(name):
    z = np.load(REF / f"{name}.npz")
    meta = json.loads((REF / f"{name}.json").read_text())
    return z, meta


def _vector(meta, name):
    import geopandas as gpd
    import shapely

    v = meta["vectors"][name]
    geoms = [shapely.from_wkb(bytes.fromhex(h)) for h in v["wkb"]]
    return gpd.GeoDataFrame(v.get("columns", {}), geometry=geoms, crs=f"EPSG:{v['epsg']}")


def _raster(z, meta, name):
    import xarray as xr
    from affine import Affine

    a = z[f"in/{name}"]
    info = meta["rasters"][name]
    gt, nd = info["gt"], info["nodata"]
    ny, nx = a.shape
    da = xr.DataArray(a, dims=("y", "x"), coords={"y": gt[3] + (np.arange(ny) + 0.5) * gt[5],
                                                  "x": gt[0] + (np.arange(nx) + 0.5) * gt[1]})
    t = Affine(gt[1], 0, gt[0], 0, gt[5], gt[3])
    crs = f"EPSG:{info['epsg']}"
    if a.dtype.kind == "f":  # as ls.load_raster reads it: NaN for NoData
        da = da.where(a != nd).astype(a.dtype)
        da = da.rio.write_transform(t).rio.write_crs(crs)
        return da.rio.write_nodata(nd, encoded=True)
    return da.rio.write_transform(t).rio.write_crs(crs).rio.write_nodata(nd)


def _inputs(name):
    z, meta = _load(name)
    spec, s = meta["case"], meta["settings"]
    habitats = {h["id"]: (_vector(meta, h["id"]) if h["kind"] == "vector" else _raster(z, meta, h["id"]),
                          h["rank"], h["distance"]) for h in meta["habitats"]}
    args = dict(aoi=_vector(meta, "aoi"), landmass=_vector(meta, "landmass"), wwiii=_vector(meta, "wwiii"),
                bathymetry=_raster(z, meta, "bathymetry"), shelf_contour=_vector(meta, "shelf"),
                dem=_raster(z, meta, "dem"), habitats=habitats, model_resolution=s["model_resolution"],
                max_fetch_distance=s["max_fetch_distance"], dem_averaging_radius=s["dem_averaging_radius"])
    if spec["geomorphology"]:
        args.update(geomorphology=_vector(meta, "geomorphology"),
                    geomorphology_fill_value=float(s["geomorphology_fill_value"]))
    if spec["population"]:
        args.update(population=_raster(z, meta, "population"), population_radius=float(s["population_radius"]))
    if spec["slr"]:
        args.update(slr=_vector(meta, "slr"), slr_field=s["slr_field"])
    return args


@cache
def _run(name, threads=4):
    return ls.services.coastal_vulnerability(**_inputs(name), n_threads=threads)


def _exact(ours, ref, what, exact=True):
    from conftest import UCRT_RTOL, assert_ucrt_close

    if exact:
        assert_ucrt_close(np.asarray(ours, np.float64), np.asarray(ref, np.float64), err_msg=what)
    else:
        np.testing.assert_allclose(np.asarray(ours, np.float64), np.asarray(ref, np.float64), rtol=UCRT_RTOL or 1e-5,
                                   atol=0, equal_nan=True, err_msg=what)


def _same_projection(name):
    """Whether the case's coordinates are projected as the references' were (module docstring)."""
    import rasterio
    from conftest import UCRT_LIBM

    proj = tuple(int(v) for v in rasterio.__proj_version__.split(".")[:2])
    return name != "coastal_lonlat" or (UCRT_LIBM and proj >= (9, 8))


def _table(df, cols):
    return np.c_[[df[c].to_numpy(np.float64) for c in cols]].T


def _nan(a, nodata):
    return np.where(a == nodata, np.nan, a) if a.dtype.kind == "f" else a


@pytest.mark.parametrize("name", CASES)
def test_exposure_bit_for_bit(name):
    import shapely

    r = _run(name)
    z, meta = _load(name)
    _exact(shapely.get_coordinates(r.exposure.geometry.values), z["out/shore_xy"], "shore_xy")
    np.testing.assert_array_equal(r.exposure["shore_id"], z["out/shore_id"])
    for k in meta["exposure_fields"]:
        _exact(r.exposure[k], z[f"out/exposure_{k}"], k)
    for k in ("surge", "relief", "slr", "population", "geomorphology"):
        if k in meta["intermediate_fields"]:
            _exact(r.intermediate[k], z[f"out/intermediate_{k}"], k)
    _exact(r.habitat_protection.to_numpy(np.float64), z["out/habitat_protection"], "habitat_protection")
    assert list(r.habitat_protection.columns) == meta["habitat_protection_header"]


@pytest.mark.parametrize("name", CASES)
def test_fetch_and_waves(name):
    r = _run(name)
    z, meta = _load(name)
    same = _same_projection(name)
    # (elsewhere the ulps the coordinates move by carry on along the rays and the waves)
    rtol, atol = (1e-9, 1e-7) if same else (1e-6, 1e-6)
    _exact(_table(r.fetch, [f"fdepth_{d}" for d in SECTORS]), z["out/fdepth"], "fdepth", same)
    _exact(_table(r.fetch, meta["wwiii_at_shore_fields"]), z["out/wwiii_at_shore"], "wwiii_at_shore", same)
    fdist = _table(r.fetch, [f"fdist_{d}" for d in SECTORS])
    ref = z["out/fdist"]
    np.testing.assert_allclose(fdist, ref, rtol=0, atol=atol)
    if same:
        assert np.mean(fdist == ref) > 0.9  # only rays across several edges differ
    np.testing.assert_allclose(r.fetch["REI"], z["out/rei"], rtol=rtol, atol=1e-9)
    import shapely

    ends = shapely.get_coordinates(r.fetch_rays.geometry.values).reshape(-1, 4)
    np.testing.assert_allclose(ends, z["out/ray_ends"], rtol=0, atol=atol)
    _exact(r.wave_energies["E_ocean"], z["out/wave_E_ocean"], "E_ocean", same)
    for k in ("E_local", "Eo_El_diff", "maxH_local", "minH_local", "maxT_local", "minT_local"):
        np.testing.assert_allclose(r.wave_energies[k], z[f"out/wave_{k}"], rtol=rtol, atol=1e-12, err_msg=k)
    assert list(r.wave_energies["max_E_type"].fillna("")) == meta["max_E_type"]
    np.testing.assert_allclose(r.intermediate["wind"], z["out/intermediate_wind"], rtol=rtol, atol=1e-9)
    np.testing.assert_allclose(r.intermediate["wave"], z["out/intermediate_wave"], rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("name", CASES)
def test_rasters_bit_for_bit(name):
    r = _run(name)
    z, meta = _load(name)
    m = meta["raster_negative_bathymetry"]
    _exact(r.bathymetry.values, _nan(z["out/raster_negative_bathymetry"], m["nodata"]), "bathymetry")
    assert tuple(r.bathymetry.rio.transform().to_gdal()) == tuple(m["gt"])
    m = meta["raster_positive_dem"]
    _exact(r.relief.values, _nan(z["out/raster_positive_dem"], m["nodata"]), "positive dem")
    for h in meta["habitats"]:
        if h["kind"] == "raster":
            m = meta[f"raster_habitat_{h['id']}"]
            _exact(r.habitat_rasters[h["id"]].values, _nan(z[f"out/raster_habitat_{h['id']}"], m["nodata"]), h["id"])
    if meta["case"]["population"]:
        m = meta["raster_clipped_projected_pop"]
        _exact(r.population.values, _nan(z["out/raster_clipped_projected_pop"], m["nodata"]), "population")
    else:
        assert r.population is None
    if "out/geomorphology_missing" in z.files:
        np.testing.assert_array_equal(r.missing_geomorphology["shore_id"], z["out/geomorphology_missing"])


def test_corrected_bugs_change_invest():
    """InVEST as released: the shore loses no point to its near-zero edge test but
    gains two from them, and its bathymetry search moves its window at the edges."""
    _, meta = _load("coastal_full")
    assert meta["unpatched"]["edges"]["shore_points"] == [267, 269]
    assert meta["unpatched"]["window"]["fdepth"] >= 1
    assert len(_run("coastal_full").exposure) == 267


def test_threads_do_not_change_results():
    a, b = _run("coastal_full", 1), _run("coastal_full", 4)
    for x, y in ((a.exposure, b.exposure), (a.fetch, b.fetch), (a.wave_energies, b.wave_energies)):
        pd.testing.assert_frame_equal(x.drop(columns="geometry"), y.drop(columns="geometry"), check_exact=True)


def test_files_and_habitat_table(tmp_path):
    """Inputs given as files, the habitats as InVEST's table of paths."""
    args = _inputs("coastal_required")
    for k in ("aoi", "landmass", "wwiii", "shelf_contour"):
        p = tmp_path / f"{k}.gpkg"
        args[k].to_file(p)
        args[k] = str(p)
    for k in ("bathymetry", "dem"):
        p = tmp_path / f"{k}.tif"
        ls.write_raster(args[k], p)
        args[k] = p
    rows = []
    for hid, (data, rank, dist) in args["habitats"].items():
        p = tmp_path / (f"{hid}.gpkg" if hasattr(data, "geometry") else f"{hid}.tif")
        data.to_file(p) if hasattr(data, "geometry") else ls.write_raster(data, p)
        rows.append({"id": hid, "path": str(p), "rank": rank, "protection distance (m)": dist})
    args["habitats"] = pd.DataFrame(rows)
    r = ls.services.coastal_vulnerability(**args, n_threads=2)
    ref = _run("coastal_required")
    pd.testing.assert_frame_equal(r.exposure.drop(columns="geometry"), ref.exposure.drop(columns="geometry"),
                                  check_exact=True)


def test_save(tmp_path):
    folder = _run("coastal_full").save(tmp_path / "cv")
    names = {p.stem for p in folder.iterdir()}
    assert {"coastal_exposure", "intermediate_exposure", "habitat_protection", "fetch_points", "fetch_rays",
            "wave_energies", "negative_bathymetry", "positive_dem", "clipped_projected_pop",
            "clipped_projected_seagrass"} <= names


def test_validation():
    args = _inputs("coastal_required")
    with pytest.raises(ValueError, match="population_radius"):
        ls.services.coastal_vulnerability(**args, population=args["dem"])
    with pytest.raises(ValueError, match="rank"):
        ls.services.coastal_vulnerability(**{**args, "habitats": {"reef": (args["habitats"]["reef"][0], 7, 100.0)}})
    aoi = args["aoi"].to_crs(4326)
    with pytest.raises(ValueError, match="projected"):
        ls.services.coastal_vulnerability(**{**args, "aoi": aoi})
    far = args["wwiii"].copy()
    far["geometry"] = far.geometry.translate(5e6, 0)
    with pytest.raises(ValueError, match="Wave Watch III"):
        ls.services.coastal_vulnerability(**{**args, "wwiii": far})


def test_ray_kernel():
    """A ray starts 1 m off the point (InVEST's): east from (0, 0) it meets the edge at
    x = 10 after 9 m; a ray that misses gets the full distance, one that starts on land 0."""
    from landschaft import _core

    pts = np.array([[0.0, 0.0]])
    deltas = np.array([[1.0, 0.0]] + [[0.0, 1.0]] * 15)
    seg = np.array([[10.0, -5.0, 10.0, 5.0]])
    fdist, ends = _core.serv_coastal_rays(pts, np.zeros((1, 16), np.uint8), deltas, 100.0, seg, 1)
    assert fdist.shape == (1, 16)
    assert fdist[0, 0] == 9.0
    np.testing.assert_allclose(ends[0, 0], [10.0, 0.0])
    assert fdist[0, 1] == 100.0
    fdist, _ = _core.serv_coastal_rays(pts, np.ones((1, 16), np.uint8), deltas, 100.0, seg, 1)
    assert np.all(fdist == 0)


def test_fewer_than_three_wave_watch_points():
    """InVEST takes 3 neighbours even when there are fewer; here the ones there are."""
    args = _inputs("coastal_required")
    w = args["wwiii"]
    near = w.distance(args["aoi"].geometry.iloc[0].centroid).sort_values().index[:2]
    r = ls.services.coastal_vulnerability(**{**args, "wwiii": w.loc[near]}, n_threads=2)
    assert np.isfinite(r.fetch["REI"]).all() and np.isfinite(r.exposure["exposure"]).all()
