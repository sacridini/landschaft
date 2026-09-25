# SPDX-License-Identifier: GPL-3.0-or-later
"""MSPA classes on hand-made patterns with an obvious expected answer."""

from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls
from conftest import mspa_input

S = np.s_


def classes(out) -> set[int]:
    return set(np.unique(out).tolist())


def test_square_is_core_with_edge():
    out = ls.mspa(mspa_input((40, 40), [S[10:30, 10:30]]))
    assert classes(out) == {0, 3, 17}
    assert out[20, 20] == 17 and out[10, 20] == 3 and out[0, 0] == 0


def test_isolated_pixel_is_islet():
    out = ls.mspa(mspa_input((15, 15), [S[7, 7]]))
    assert out[7, 7] == 9
    assert classes(out) == {0, 9}


def test_hole_gives_perforation_and_core_opening():
    a = mspa_input((40, 40), [S[5:35, 5:35]])
    a[15:25, 15:25] = 1
    out = ls.mspa(a)
    assert out[20, 20] == 100  # core-opening
    assert out[14, 20] == 105  # internal perforation
    ext = ls.mspa(a, internal=False)
    assert ext[20, 20] == 0 and ext[14, 20] == 5


def test_corridor_between_cores_is_bridge():
    a = mspa_input((30, 60), [S[5:25, 5:20], S[5:25, 40:55], S[15, 20:40]])
    out = ls.mspa(a)
    assert classes(out[15, 22:38]) == {33}


def test_corridor_back_to_same_core_is_loop():
    a = mspa_input((40, 40), [S[20:38, 5:35], S[5, 8:30], S[5:20, 8], S[5:20, 29]])
    out = ls.mspa(a)
    assert classes(out[5, 10:28]) == {65}
    assert out[10, 15] == 220  # background enclosed by the loop: border-opening


def test_dead_end_is_branch():
    out = ls.mspa(mspa_input((30, 40), [S[5:25, 5:20], S[15, 20:35]]))
    assert classes(out[15, 22:34]) == {1}


def test_connectivity_matters_for_diagonal_links():
    a = mspa_input((30, 60), [S[5:25, 5:20], S[5:25, 40:55]])
    for i in range(20):
        a[5 + i // 2, 20 + i] = 2  # staircase: 8-connected only
    assert 33 in classes(ls.mspa(a, connectivity=8))
    out4 = ls.mspa(a, connectivity=4)
    assert 33 not in classes(out4) and 9 in classes(out4)


def test_missing_pixels_are_129():
    a = mspa_input((40, 40), [S[10:30, 10:30]], missing=[S[0:5, :], S[20:22, 20:22]])
    out = ls.mspa(a)
    assert (out[a == 0] == 129).all()
    assert (out[a != 0] != 129).all()


def test_edge_width_widens_edges():
    a = mspa_input((60, 60), [S[5:55, 5:55]])
    n_edge = [int((ls.mspa(a, edge_width=w) == 3).sum()) for w in (1, 2, 4)]
    assert n_edge[0] < n_edge[1] < n_edge[2]


@pytest.mark.parametrize("internal", [True, False])
def test_output_values_are_known_classes(internal, rng):
    a = np.where(rng.random((80, 90)) < 0.55, 2, 1).astype(np.uint8)
    a[rng.random(a.shape) < 0.02] = 0
    out = ls.mspa(a, internal=internal)
    assert out.dtype == np.uint8 and out.shape == a.shape
    assert classes(out) <= set(ls.MSPA_CLASSES)
    if not internal:
        assert classes(out) <= {0, 1, 3, 5, 9, 17, 33, 35, 37, 65, 67, 69, 129}


def test_transition_only_affects_colours(rng):
    a = np.where(rng.random((60, 60)) < 0.6, 2, 1).astype(np.uint8)
    np.testing.assert_array_equal(ls.mspa(a, transition=True), ls.mspa(a, transition=False))


def test_threads_do_not_change_result(rng):
    a = np.where(rng.random((150, 170)) < 0.55, 2, 1).astype(np.uint8)
    np.testing.assert_array_equal(ls.mspa(a, n_threads=1), ls.mspa(a, n_threads=4))


@pytest.mark.parametrize("shape", [(1, 2), (1, 7), (6, 1), (2, 2), (3, 5)])
def test_tiny_images(shape):
    a = np.full(shape, 2, np.uint8)
    a.flat[0] = 1
    out = ls.mspa(a)
    assert out.shape == shape


def test_single_pixel_without_validation():
    assert ls.mspa(np.full((1, 1), 2, np.uint8), validate=False).shape == (1, 1)


@pytest.mark.parametrize("kw", [{"connectivity": 6}, {"edge_width": 0}, {"edge_width": 256}, {"edge_width": 1.5}])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError):
        ls.mspa(mspa_input((10, 10), [S[3:6, 3:6]]), **kw)
