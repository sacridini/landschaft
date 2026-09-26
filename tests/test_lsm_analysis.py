# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatio-temporal series and spatial signatures of landscape metrics."""

from __future__ import annotations

import warnings

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


class TestSignatureFactorsAndClusters:
    TABLE = pd.DataFrame(np.random.default_rng(0).gamma(2, 1, (40, 6)), columns=list("abcdef"))

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_nmf_matches_scikit_learn(self, k):
        pytest.importorskip("sklearn")
        from sklearn.decomposition import NMF

        res = ls.SpatialSignatureAnalysis.from_table(self.TABLE).nmf(k)
        x = (self.TABLE - self.TABLE.min()) / (self.TABLE.max() - self.TABLE.min())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = NMF(k, init="nndsvda", solver="cd", max_iter=5000, tol=1e-10).fit(x.to_numpy())
        # the same local optimum from the same start, to the stopping tolerance
        assert res.reconstruction_error == pytest.approx(ref.reconstruction_err_, rel=1e-4)
        assert (res.scores.to_numpy() >= 0).all() and (res.components.to_numpy() >= 0).all()
        np.testing.assert_allclose(np.linalg.norm(x.to_numpy() - res.scores.to_numpy() @ res.components.to_numpy().T),
                                   res.reconstruction_error)
        assert list(res.components.index) == list("abcdef") and list(res.scores.columns)[0] == "NMF1"

    def test_nmf_recovers_non_negative_low_rank(self):
        rng = np.random.default_rng(1)
        w = rng.uniform(size=(30, 3))
        w[0] = 0  # a zero row: min-max scaling is a column scaling, rank 3 stays exact
        table = pd.DataFrame(w @ rng.uniform(size=(3, 8)))
        res = ls.SpatialSignatureAnalysis.from_table(table).nmf(3, max_iter=20000, tol=1e-14)
        assert res.reconstruction_error < 1e-10
        with pytest.raises(ValueError, match="n_components"):
            ls.SpatialSignatureAnalysis.from_table(table).nmf(9)

    @pytest.mark.parametrize("method", ["ward", "average", "complete", "single"])
    def test_hierarchical_matches_scipy(self, method):
        from scipy.cluster import hierarchy

        sig = ls.SpatialSignatureAnalysis.from_table(self.TABLE)
        labels = sig.cluster(4, method=method)
        z = (self.TABLE - self.TABLE.mean()) / self.TABLE.std(ddof=0)
        ref = hierarchy.fcluster(hierarchy.linkage(z.to_numpy(), method=method), 4, criterion="maxclust")
        # the same partition, numbered by first appearance
        mapping = dict(zip(ref, labels))
        assert [mapping[r] for r in ref] == labels.tolist()
        assert len(set(zip(ref, labels))) == len(set(ref)) and labels.iloc[0] == 0
        with pytest.raises(ValueError, match="method"):
            sig.cluster(3, method="spectral")

    def test_clustergram_matches_clustergram_package(self):
        clustergram = pytest.importorskip("clustergram")
        pytest.importorskip("sklearn")
        sig = ls.SpatialSignatureAnalysis.from_table(self.TABLE)
        ks = range(1, 7)
        res = sig.clustergram(ks, method="ward")
        z = (self.TABLE - self.TABLE.mean()) / self.TABLE.std(ddof=0)
        for k in ks:
            np.testing.assert_array_equal(res.labels[k], sig.cluster(k, method="ward"))
        # the package's means from our partitions (from_data) and its PCA weighting
        ref = clustergram.Clustergram.from_data(z, res.labels, method="mean")
        ref._compute_means(True, {"n_components": 1})
        theirs = ref.plot_data_pca[1]
        for k in ks:
            a, b = res.means[k].to_numpy(), theirs[k].to_numpy()
            assert np.allclose(a, b, atol=1e-12) or np.allclose(a, -b, atol=1e-12)  # PCA sign
        flat = sig.clustergram(ks, method="ward", pca_weighted=False)
        ref._compute_means(False, {})
        for k in ks:
            np.testing.assert_allclose(flat.means[k].to_numpy(), ref.plot_data[k].to_numpy(), atol=1e-12)
        km = sig.clustergram(range(2, 5))
        for k in range(2, 5):
            np.testing.assert_array_equal(km.labels[k], sig.cluster(k))
        with pytest.raises(ValueError, match="k_range"):
            sig.clustergram([0, 1])

    def test_plot_clustergram(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sig = ls.SpatialSignatureAnalysis.from_table(self.TABLE)
        fig = sig.plot_clustergram(k_range=range(1, 6), method="average")
        ax = fig.axes[0]
        assert len(ax.collections) == 5 and len(ax.lines) >= 4
        assert ax.get_xlabel() == "Number of clusters (k)"
        plt.close("all")
