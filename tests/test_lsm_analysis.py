# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatio-temporal series and spatial signatures of landscape metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
import lsm_reference
import lsm_scenes

MAPS = [lsm_scenes.clustered((30, 34), [1, 2, 3, 4], seed=s, sigma=0.8 + 0.5 * (s % 4), na_fraction=0.03 * (s % 2))
        for s in range(10)]
# one class missing in the last map
MAPS[-1] = np.ma.MaskedArray(np.where(MAPS[-1].data == 4, 3, MAPS[-1].data), MAPS[-1].mask)


class TestSpatioTemporal:
    def test_landscape_and_class_series(self):
        dates = [2000 + i for i in range(len(MAPS))]
        sta = ls.SpatioTemporalAnalysis(MAPS, dates, directions=4)
        land = sta.landscape_metrics(["shdi", "np", "ed"])
        assert list(land.index) == dates
        for i, data in enumerate(MAPS):
            ref = ls.Landscape(data, directions=4).landscape_metrics(["shdi", "np", "ed"])
            np.testing.assert_array_equal(land.iloc[i][["shdi", "np", "ed"]].to_numpy(), ref.iloc[0].to_numpy())
        cls = sta.class_metrics(["pland", "np", "shape_mn"])
        assert cls.loc[(4, dates[-1]), "pland"] == 0 and cls.loc[(4, dates[-1]), "np"] == 0  # absent: 0
        assert np.isnan(cls.loc[(4, dates[-1]), "shape_mn"])
        raw = sta.class_metrics(["pland"], fillna=False)
        assert np.isnan(raw.loc[(4, dates[-1]), "pland"])
        long = sta.metrics(level="landscape", metric="ta")
        assert long["date"].tolist() == dates

    def test_zones_buffers_grid(self, tmp_path):
        paths = []
        for i, data in enumerate(MAPS[:3]):
            path = tmp_path / f"m{i}.tif"
            lsm_reference.write_landscape(path, data, (10.0, 10.0))
            paths.append(path)
        pts = np.array([[80.0, 200.0], [250.0, 120.0]])
        sta = ls.SpatioTemporalAnalysis(paths, ["a", "b", "c"], points=pts, size=60, shape="circle")
        land = sta.landscape_metrics(["ta", "shdi"])
        assert land.index.names == ["zone", "date"] and len(land) == 6
        for j, path in enumerate(paths):
            ref = ls.sample_metrics(path, pts, size=60, shape="circle", level="landscape", metric=["ta", "shdi"],
                                    verbose=False)
            for zone in (1, 2):
                r = ref[ref["plot_id"] == zone].set_index("metric")["value"]
                np.testing.assert_array_equal(land.loc[(zone, "abc"[j])].to_numpy(), r.loc[["shdi", "ta"]].to_numpy())
        zones = ls.grid_zones(paths[0], 170)
        assert len(zones) == 2 * 2
        grid = ls.SpatioTemporalAnalysis(paths, grid=170).landscape_metrics(["ta"])
        assert grid.loc[(1, 1), "ta"] == pytest.approx(17 * 17 * 100 / 10000)
        with pytest.raises(ValueError, match="several"):
            ls.SpatioTemporalAnalysis(paths, points=pts, grid=10)

    def test_plot_metric(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        sta = ls.SpatioTemporalAnalysis(MAPS[:4])
        fig = sta.plot_metric("shdi")
        assert len(fig.axes[0].lines) == 1
        fig = sta.plot_metric("pland", classes="all")
        assert len(fig.axes[0].lines) == 4
        import matplotlib.pyplot as plt

        plt.close("all")


class TestSpatialSignature:
    def signature(self):
        return ls.SpatialSignatureAnalysis(MAPS, landscape_metrics=["shdi", "np", "ed", "contag"],
                                           class_metrics=["np", "shape_mn"], classes=[1, 2])

    def test_table(self):
        sig = self.signature()
        assert list(sig.table.columns) == ["contag", "ed", "np", "shdi", "np_1", "np_2", "shape_mn_1", "shape_mn_2"]
        assert len(sig.table) == len(MAPS)
        np.testing.assert_array_equal(sig.table["np"].to_numpy(),
                                      [ls.Landscape(d).landscape_metrics("np")["np"][0] for d in MAPS])

    def test_pca_matches_scikit_learn(self):
        sklearn = pytest.importorskip("sklearn")  # noqa: F841
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        sig = self.signature()
        res = sig.pca()
        z = StandardScaler().fit_transform(sig.table.to_numpy())
        pca = PCA().fit(z)
        np.testing.assert_allclose(res.explained_variance_ratio, pca.explained_variance_ratio_, rtol=1e-10)
        ours = res.scores.to_numpy()
        theirs = pca.transform(z)
        for j in range(ours.shape[1]):
            if pca.explained_variance_ratio_[j] > 1e-10:
                assert np.allclose(ours[:, j], theirs[:, j], atol=1e-9) or np.allclose(ours[:, j], -theirs[:, j],
                                                                                        atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(res.loadings.to_numpy(), axis=0), 1.0)

    def test_kmeans_finds_separated_groups(self):
        rng = np.random.default_rng(3)
        table = pd.DataFrame(np.concatenate([rng.normal(0, 0.3, (15, 3)), rng.normal(5, 0.3, (12, 3)),
                                             rng.normal([0, 8, 0], 0.3, (10, 3))]), columns=["a", "b", "c"])
        labels = ls.SpatialSignatureAnalysis.from_table(table).cluster(3)
        expected = np.repeat([0, 1, 2], [15, 12, 10])
        pytest.importorskip("sklearn")
        from sklearn.metrics import adjusted_rand_score

        assert adjusted_rand_score(expected, labels) == 1.0
        assert labels.iloc[0] == 0
        with pytest.raises(ValueError, match="n_clusters"):
            ls.SpatialSignatureAnalysis.from_table(table).cluster(99)

    def test_zonal_signature_and_plots(self, tmp_path):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        path = tmp_path / "m.tif"
        lsm_reference.write_landscape(path, MAPS[0], (10.0, 10.0))
        sig = ls.SpatialSignatureAnalysis(path, grid=100, landscape_metrics=["shdi", "ed", "np"])
        assert sig.table.index.name == "zone" and len(sig.table) == 12
        clusters = sig.cluster(2, n_components=2)
        fig = sig.plot_pca(clusters, labels=True)
        assert fig.axes
        fig = sig.plot_cluster_metrics(clusters)
        assert len(fig.axes) >= 3
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_nan_rows_are_left_out(self):
        table = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 0.0, 1.0, 3.0]})
        with pytest.warns(RuntimeWarning, match="left out"):
            res = ls.SpatialSignatureAnalysis.from_table(table).pca()
        assert res.dropped == [2] and len(res.scores) == 3
