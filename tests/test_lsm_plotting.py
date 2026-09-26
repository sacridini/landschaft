# SPDX-License-Identifier: GPL-3.0-or-later
"""Plots of patches, cores, patch metrics and correlations."""

from __future__ import annotations

import numpy as np
import pytest

import landschaft as ls
import lsm_scenes

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = lsm_scenes.clustered((20, 24), [1, 2, 3], seed=2, na_fraction=0.05)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def images(fig):
    return [ax.get_images()[0].get_array() for ax in fig.axes if ax.get_images()]


def test_show_patches():
    lm = ls.Landscape(DATA)
    fig = lm.show_patches()
    (img,) = images(fig)
    labels = np.asarray(lm.labels())
    np.testing.assert_array_equal(np.ma.getmaskarray(img) | np.isnan(np.ma.getdata(img)), labels == 0)
    fig = ls.show_patches(DATA, "all", labels=True, directions=4)
    assert len(images(fig)) == 3
    assert any(ax.texts for ax in fig.axes)
    fig = lm.show_patches(classes=[2])
    (img,) = images(fig)
    assert np.all(np.isnan(np.ma.getdata(img))[np.ma.getdata(DATA) != 2])
    with pytest.raises(ValueError, match="not present"):
        lm.show_patches(classes=7)


def test_show_cores_marks_the_core_cells():
    lm = ls.Landscape(DATA, edge_depth=2)
    fig = lm.show_cores("global", edge_depth=2, consider_boundary=False, labels=True)
    (img,) = images(fig)
    img = np.ma.getdata(img)
    core_cells = np.sum(img > 0.5)
    assert core_cells * 1e-4 == pytest.approx(lm.landscape_metrics("tca")["tca"][0])


def test_show_lsm():
    lm = ls.Landscape(DATA, res=10)
    fig = lm.show_lsm("area", "all", labels=True)
    assert len(images(fig)) == 3
    area = np.asarray(lm.to_raster("area"))
    (first, *_) = images(fig)
    first = np.ma.getdata(first)
    keep = np.isfinite(first)
    np.testing.assert_array_equal(first[keep], area[keep])


def test_show_correlation():
    metrics = ls.Landscape(DATA).metrics(what=["lsm_p_area", "lsm_p_perim", "lsm_p_shape", "lsm_c_ca", "lsm_c_te"])
    fig = ls.show_correlation(metrics, labels=True)
    assert len(images(fig)) == 2
    cor = ls.calculate_correlation(metrics, method="spearman")
    fig = ls.show_correlation(cor["patch"])
    (img,) = images(fig)
    assert np.ma.getdata(img)[0, 0] == 1.0
