# SPDX-License-Identifier: GPL-3.0-or-later
"""Influence Zones and Proximity, and the IDL routines they use.

HIST_EQUAL is compared with GDL's (``live_reference``). WATERSHED and
MORPH_DISTANCE are not available in GDL; the pipeline is compared with
outputs of the GuidosToolbox desktop application stored in
tests/data/reference/gtb_desktop (see tools/make_reference_gtb_desktop.md).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage as ndi
import xarray as xr

import landschaft as ls
import scenes
from conftest import mspa_input
from gtb_reference import find_gdl
from landschaft._core import detail as D


# --- IDL routines -------------------------------------------------------------------


def test_morph_distance_euclidean(rng):
    mask = (rng.random((40, 50)) < 0.05).astype(np.uint8)
    bg = D.morph_distance(mask, True, "euclidean")
    np.testing.assert_allclose(bg, ndi.distance_transform_edt(mask == 0), rtol=1e-6)
    fg = D.morph_distance(mask, False, "euclidean")
    assert not fg[mask == 0].any()
    np.testing.assert_allclose(fg[mask == 1], 1.0)


def test_morph_distance_quasi_euclidean():
    mask = np.zeros((9, 9), np.uint8)
    mask[4, 4] = 1
    d = D.morph_distance(mask, True, "quasi-euclidean")
    assert d[4, 7] == pytest.approx(3.0)
    assert d[1, 1] == pytest.approx(3 * np.sqrt(2), rel=1e-6)
    assert d[2, 7] == pytest.approx(1 + 2 * np.sqrt(2), rel=1e-6)


def test_watershed_two_basins():
    im = np.array([[0, 1, 2, 3, 2, 1, 0]] * 3, dtype=np.uint8)
    ws, n = D.watershed(im, 8)
    assert n == 2
    np.testing.assert_array_equal(ws[:, 3], 0)  # the ridge is the line
    assert (ws[:, :3] == 1).all() and (ws[:, 4:] == 2).all()


def test_watershed_plateau_and_numbering():
    im = np.full((5, 9), 5, np.uint8)
    im[:, 0] = 0
    im[:, 8] = 0
    ws, n = D.watershed(im, 8)
    assert n == 2
    np.testing.assert_array_equal(ws[:, 4], 0)  # plateau split in the middle
    assert ws[0, 0] == 1 and ws[0, 8] == 2  # minima numbered in raster order
    ws4, n4 = D.watershed(im, 4)
    assert n4 == 2 and (ws4[:, 4] == 0).all()


def test_watershed_single_minimum_has_no_line():
    im = np.arange(30, dtype=np.uint8).reshape(5, 6)
    ws, n = D.watershed(im, 8)
    assert n == 1 and (ws == 1).all()


def test_hist_equal_properties(rng):
    d = D.morph_distance((rng.random((60, 70)) < 0.02).astype(np.uint8), True, "euclidean")
    h = D.hist_equal(d)
    assert h.dtype == np.uint8 and h.max() == 255 and h[d == 0].max() == 0
    order = np.argsort(d.ravel(), kind="stable")
    assert (np.diff(h.ravel()[order].astype(int)) >= 0).all()  # monotone in the distance


# --- Influence Zones / Proximity --------------------------------------------------


def two_squares(gap: int = 6):
    a = mspa_input((30, 40), foreground=[np.s_[10:20, 5:15], np.s_[10:20, 15 + gap:25 + gap]])
    return a


def test_influence_two_objects():
    a = two_squares(6)
    r = ls.influence_zones(a, 50)
    assert r.n_objects == 2 and r.object_area.tolist() == [100, 100]
    cls = np.asarray(r.classes)
    # the watershed line runs midway between the squares
    assert (cls[:, 17:19] == 4).any()
    lines = np.nonzero((cls == 4).any(axis=0))[0]
    assert set(lines) <= {17, 18}
    # objects numbered from the bottom row (GTB labels its flipped image) and coloured by id
    ids = np.asarray(r.ids)
    assert set(np.unique(ids[10:20, 5:15])) == {1} and set(np.unique(ids[10:20, 21:31])) == {2}
    assert cls[12, 7] == 5 and cls[12, 25] == 9  # colours 5 (id 1) and 9 (id 2)
    zones = np.asarray(r.zones)
    assert set(np.unique(zones)) == {0, 1, 2}


def test_min_area_holes_missing_and_buffer():
    a = mspa_input((40, 40), foreground=[np.s_[5:25, 5:25], np.s_[35, 35]])
    a[10:12, 10:12] = 1  # a hole
    a[14, 14] = 0  # missing inside the object
    a[0, 39] = 0
    r0 = ls.influence_zones(a, 10)
    assert (np.asarray(r0.classes)[10:12, 10:12] == 0).all() and (np.asarray(r0.ids)[10:12, 10:12] == -1).all()
    r = ls.influence_zones(a, 10, buffer=3)
    cls, ids = np.asarray(r.classes), np.asarray(r.ids)
    assert r.n_objects == 1
    assert (cls[10:12, 10:12] == 160).all()  # as in GTB, the buffer also covers holes near the object
    assert cls[14, 14] == 129 and ids[14, 14] == -4 and cls[0, 39] == 129
    assert cls[35, 35] == 176 and ids[35, 35] == -2  # too small: not an object
    assert cls[27, 10] == 80 and ids[27, 10] == -5  # buffer boundary at distance 3
    assert cls[26, 10] == 160 and ids[26, 10] == -6  # inside the buffer
    assert r.object_area.tolist() == [400 - 4 - 1]
    with pytest.raises(ValueError, match="too large"):
        ls.influence_zones(a, 1000)


def test_core_buffer():
    a = mspa_input((40, 60), foreground=[np.s_[5:35, 5:35], np.s_[5:9, 40:55]])
    r = ls.influence_zones(a, 10, core_buffer=2)
    cls = np.asarray(r.classes)
    assert r.n_objects == 1  # the thin strip has no core
    assert (cls[5:7, 10] == 176).all()  # the object rim is not part of the core
    assert cls[20, 20] != 176


def test_proximity_two_objects():
    a = two_squares(6)
    r = ls.proximity(a, 50)
    assert r.n_objects == 2
    cag = int(round(np.sqrt(2 * 100 * 100)))
    assert (r.lines["cag"] == cag).all() and (r.lines["nbr"] == 2).all()
    # prox_min is round(2 * distance) at the closest watershed pixel (the gap is 6 pixels wide)
    lines = np.asarray(r.zones) == 0
    assert r.prox_min == int(np.asarray(r.distance)[lines].min()) and r.prox_min in (6, 7)
    assert r.prox_min_count > 0
    assert (np.asarray(r.classes)[np.asarray(r.zones) > 0] != 193).all()
    assert set(np.unique(np.asarray(r.classes))) <= {0, 1, 4, 12, 165, 193}
    text = r.csv_text("test.tif").splitlines()
    assert text[0] == f"test.tif - Min. FG-obj. size: 50 - 2 objects - prox_min: {r.prox_min}"
    assert text[1] == "WS-X,WS-Y, #nbr, clength, CAG, IDs(area),"
    assert text[2].endswith(f",2,{r.lines['clength'][0]},{cag},1-100  2-100")
    assert len(text) == 2 + len(r.lines)


def test_proximity_xarray_and_write(tmp_path):
    da = ls.load_raster(ls.example_data("input"))
    r = ls.proximity(da)
    assert isinstance(r.cag, xr.DataArray) and r.cag.rio.crs == da.rio.crs
    assert r.source.endswith("input.tif")
    path = ls.write_raster(r.classes, tmp_path / "prox.tif")
    assert path.exists()
    z = ls.influence_zones(da, buffer=10)
    assert z.classes.attrs["landschaft_product"] == "INFLUENCE_ZONES"
    ls.write_raster(z.classes, tmp_path / "influ.tif")


def test_invalid_input():
    with pytest.raises(ls.InvalidInputError):
        ls.influence_zones(np.full((5, 5), 3, np.uint8))
    with pytest.raises(ValueError):
        ls.proximity(two_squares(), 0)


# --- agreement with GDL (HIST_EQUAL) ---------------------------------------------


@pytest.mark.live_reference
@pytest.mark.parametrize("seed", range(3))
def test_hist_equal_matches_gdl(seed):
    gdl = find_gdl()
    if gdl is None:
        pytest.skip("GDL not found")
    a = scenes.blobs(seed, shape=(150, 170), scale=5 + 7 * seed)
    d = D.morph_distance((a == 2).astype(np.uint8), True, "euclidean")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        d.astype("<f4").tofile(tmp / "in.raw")
        p = str(tmp).replace("\\", "/")
        ny, nx = d.shape
        script = (f"x = fltarr({nx},{ny}) & openr, u, '{p}/in.raw', /get_lun & readu, u, x & free_lun, u\n"
                  f"h = hist_equal(x)\nopenw, u, '{p}/out.raw', /get_lun & writeu, u, h & free_lun, u\nexit\n")
        subprocess.run([str(gdl), "-quiet"], input=script, text=True, capture_output=True, timeout=120)
        ref = np.fromfile(tmp / "out.raw", dtype=np.uint8).reshape(ny, nx)
    np.testing.assert_array_equal(D.hist_equal(d), ref)
