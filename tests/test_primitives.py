# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests of the C++ morphological primitives against independent references."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from landschaft._core import detail as D

STRUCT = {4: ndi.generate_binary_structure(2, 1), 8: ndi.generate_binary_structure(2, 2)}


def random_binary(rng, shape=(47, 61), p=0.5) -> np.ndarray:
    return (rng.random(shape) < p).astype(np.uint8)


def edt_value(d2: np.ndarray) -> np.ndarray:
    """uint32_to_float followed by sqrt in double, rounded to float (as miallib)."""
    return np.sqrt(d2.astype(np.float32).astype(np.float64)).astype(np.float32)


# --- distances ----------------------------------------------------------------


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_sqedt_matches_scipy(rng, p):
    im = random_binary(rng, p=p)
    im[0, 0] = 0  # at least one zero pixel
    expected = np.round(ndi.distance_transform_edt(im) ** 2).astype(np.uint32)
    np.testing.assert_array_equal(D.sqedt(im), expected)


def test_sqedt_without_zero_pixels_reproduces_original_overflow():
    # With no zero pixel, efedt.c works with bigval = UINT32_MAX / 2 and the
    # 32-bit truncation makes the distance collapse to (y - 1)^2.
    im = np.ones((6, 9), np.uint8)
    y = np.arange(6)[:, None]
    np.testing.assert_array_equal(D.sqedt(im), np.broadcast_to((y - 1) ** 2, im.shape))


@pytest.mark.parametrize("s", [0.0, 1.0, np.float32(np.sqrt(2)), 1.98, 5.98, 10.5, 127.3, 360.0])
def test_squared_threshold_is_exact_boundary(s):
    s = np.float32(s)
    t = D.squared_threshold(float(s))
    assert edt_value(np.array([t], np.uint32))[0] <= s
    assert edt_value(np.array([t + 1], np.uint32))[0] > s


@pytest.mark.parametrize("t", [0, 1, 2, 8, 35, 36, 200])
def test_dilate_disk_matches_brute_force(rng, t):
    src = random_binary(rng, (41, 53), p=0.03)
    src[20, 26] = 1
    d2 = np.round(ndi.distance_transform_edt(src == 0) ** 2)
    np.testing.assert_array_equal(D.dilate_disk(src, t), (d2 <= t).astype(np.uint8))


def test_dilate_disk_empty_source():
    assert D.dilate_disk(np.zeros((5, 7), np.uint8), 10).sum() == 0


def test_ced_single_seed_equals_euclidean_distance():
    ref = np.zeros((25, 31), np.uint8)
    ref[12, 9] = 1
    mask = np.ones_like(ref)
    yy, xx = np.mgrid[: ref.shape[0], : ref.shape[1]]
    expected = np.sqrt((yy - 12.0) ** 2 + (xx - 9.0) ** 2).astype(np.float32)
    np.testing.assert_array_equal(D.ced(ref, mask), expected)


def test_ced_goes_around_obstacles():
    ref = np.zeros((21, 21), np.uint8)
    ref[10, 2] = 1
    mask = np.ones_like(ref)
    mask[3:18, 10] = 0  # wall
    d = D.ced(ref, mask)
    euclid = np.hypot(*np.mgrid[:21, :21] - np.array([10, 2])[:, None, None])
    assert d[10, 18] > euclid[10, 18] + 1  # must detour around the wall
    assert (d[mask == 0] == 0).all() and d[10, 2] == 0
    assert (d[mask == 1][1:] >= 0).all()


# --- reconstructions / connectivity -------------------------------------------


@pytest.mark.parametrize("graph", [4, 8])
def test_fill_holes_matches_scipy(rng, graph):
    im = random_binary(rng, p=0.6)
    expected = ndi.binary_fill_holes(im, structure=STRUCT[graph]).astype(np.uint8)
    np.testing.assert_array_equal(D.fill_holes(im, graph), expected)


@pytest.mark.parametrize("graph", [4, 8])
def test_reconstruct_dilation_matches_labelling(rng, graph):
    mask = random_binary(rng, p=0.55)
    marker = random_binary(rng, p=0.02) & mask
    labels, _ = ndi.label(mask, structure=STRUCT[graph])
    hit = np.unique(labels[marker > 0])
    expected = np.isin(labels, hit[hit > 0]).astype(np.uint8)
    np.testing.assert_array_equal(D.reconstruct_dilation(marker, mask, graph), expected)


@pytest.mark.parametrize("graph", [4, 8])
def test_label_components_raster_order_from_two(rng, graph):
    im = random_binary(rng, p=0.5).astype(np.uint32)
    inner = im.copy()
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = 0  # border is cleared first
    scipy_labels, _ = ndi.label(inner, structure=STRUCT[graph])
    expected = np.where(scipy_labels > 0, scipy_labels + 1, 0).astype(np.uint32)
    np.testing.assert_array_equal(D.label_components(im, graph), expected)


def test_wsfah_floods_from_both_ends():
    labels = np.zeros((5, 12), np.uint32)
    labels[2, 1] = 2
    labels[2, 10] = 3
    ref = np.full((5, 12), 255, np.uint8)
    ref[2, 1:11] = 1
    out = D.wsfah(labels, ref, 4, 254)
    np.testing.assert_array_equal(out[2, 1:11], [2] * 5 + [3] * 5)
    assert out[1, 3] == 0x7FFFFFFF  # blocked neighbours get LABEL_MAX
    assert (out[0] == 0).all() and (out[:, 0] == 0).all()  # frame reset to 0


# --- thinning -------------------------------------------------------------------


@pytest.mark.parametrize("stype,graph", [(0, 8), (1, 4)])
def test_thinning_is_homotopic_and_anchored(rng, stype, graph):
    im = np.zeros((40, 50), np.uint8)
    im[5:35, 5:45] = (rng.random((30, 40)) < 0.8).astype(np.uint8)
    im[15:25, 20:30] = 1
    im[18:22, 23:27] = 0  # a hole to preserve
    anchor = np.zeros_like(im)
    anchor[15, 20] = 1
    out = D.thin_anchored(im, stype, anchor)

    assert ((out == 1) <= (im == 1)).all()  # thinning only removes pixels
    assert out[15, 20] == 1  # anchors are kept
    bg_graph = 12 - graph  # background uses the complementary connectivity
    assert ndi.label(out, STRUCT[graph])[1] == ndi.label(im, STRUCT[graph])[1]
    assert ndi.label(out == 0, STRUCT[bg_graph])[1] == ndi.label(im == 0, STRUCT[bg_graph])[1]


@pytest.mark.parametrize("stype", [0, 1])
def test_thinning_reduces_thick_ring_to_thin_loop(stype):
    im = np.zeros((40, 50), np.uint8)
    im[5:35, 5:45] = 1
    im[15:25, 18:32] = 0
    out = D.thin_anchored(im, stype, np.zeros_like(im))
    assert out.sum() < im.sum() / 5
    assert ndi.label(out == 0, STRUCT[4])[1] == 2  # the hole survives
