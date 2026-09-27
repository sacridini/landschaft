# SPDX-License-Identifier: GPL-3.0-or-later
"""Threads of the parallel code: all cores but one by default, configurable per call."""

from __future__ import annotations

import inspect
import threading

import numpy as np
import pytest

import landschaft as ls
from landschaft import _core, _threads


@pytest.fixture(autouse=True)
def _reset_default():
    yield
    ls.set_default_threads(None)


def test_default_leaves_one_core():
    cores = _threads.available_cores()
    assert ls.default_threads() == max(1, cores - 1)
    if _core.openmp:
        assert _core.team_size() == ls.default_threads()


def test_limit_and_process_default():
    with _threads.limit(2):
        assert _core.team_size() == (2 if _core.openmp else 1)
        with _threads.limit(1):
            assert _core.team_size() == 1
        assert _core.team_size() == (2 if _core.openmp else 1)
    assert ls.set_default_threads(3) == 3
    assert ls.default_threads() == 3
    assert _threads.resolve(None) == _threads.resolve(0) == 3
    assert _threads.resolve(5) == 5
    if _core.openmp:
        assert _core.team_size() == 3
        # other Python threads follow the same rule
        seen = []
        t = threading.Thread(target=lambda: seen.append(_core.team_size()))
        t.start()
        t.join()
        assert seen == [3]
    ls.set_default_threads(None)
    assert ls.default_threads() == max(1, _threads.available_cores() - 1)
    with pytest.raises(ValueError):
        ls.set_default_threads(0)
    with pytest.raises(ValueError):
        _threads.resolve(-1)


def test_parallel_functions_take_n_threads():
    for fn in (ls.fad, ls.fad_multiscale, ls.landscape_mosaic, ls.spatial_entropy, ls.euclidean_distance,
               ls.parcellation, ls.morphological_change, ls.fos_change, ls.distance_to_resistance,
               ls.isochrone_map_ab, ls.raster_boundaries, ls.mspa, ls.accounting, ls.spa, ls.fos,
               ls.landscape_metrics, ls.sample_metrics, ls.window_metrics, ls.omniscape,
               ls.conn.protconn_raster, ls.conn.settlement_probabilities, ls.circuit.circuitscape_pairwise,
               ls.circuit.circuitscape_network_pairwise):
        assert "n_threads" in inspect.signature(fn).parameters, fn.__name__


def test_results_do_not_depend_on_the_threads():
    rng = np.random.default_rng(4)
    fg = rng.choice(np.array([1, 2], dtype=np.uint8), size=(120, 150), p=[0.4, 0.6])
    a = ls.fad(fg, n_threads=1)
    b = ls.fad(fg)
    c = ls.fad(fg, threads=3)  # the older name
    np.testing.assert_array_equal(np.asarray(a.density), np.asarray(b.density))
    np.testing.assert_array_equal(np.asarray(a.classes), np.asarray(c.classes))
    classes = rng.integers(1, 4, size=(80, 90))
    np.testing.assert_array_equal(ls.raster_boundaries(classes, n_threads=1), ls.raster_boundaries(classes))


def test_landscape_stores_a_count():
    from landschaft.lsm.landscape import core_options

    assert core_options({}, None)["n_threads"] == ls.default_threads()
    assert core_options({}, 2)["n_threads"] == 2


def test_batch_workers_default():
    from landschaft.batch import run_parallel

    src = inspect.getsource(run_parallel)
    assert "default_threads()" in src
