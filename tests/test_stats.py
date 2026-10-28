# SPDX-License-Identifier: GPL-3.0-or-later
"""ls.stats against pymannkendall, PySAL's esda and scipy (tests/data/reference/stats, tools/make_reference_stats.py)."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import landschaft as ls
from landschaft import _core

REF = Path(__file__).parent / "data" / "reference" / "stats"


@pytest.fixture(scope="module")
def z():
    return dict(np.load(REF / "stats.npz"))


def close(got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    assert got.shape == want.shape
    assert np.all(np.abs(got - want) <= tol * np.maximum(np.abs(want), 1.0)), np.max(np.abs(got - want))


def test_ndtr_and_ndtri_are_scipys(z):
    from conftest import UCRT_LIBM
    from scipy.special import ndtr, ndtri

    # Cephes' erfc calls the C library's exp: the references carry the UCRT's (made on Windows),
    # elsewhere scipy's ndtr with the same libm
    x = z["ndtr/x"]
    np.testing.assert_array_equal(_core.stats_ndtr(x), z["ndtr/y"] if UCRT_LIBM else ndtr(x))
    q = np.concatenate([np.linspace(1e-12, 1 - 1e-12, 2001), [1e-300, 0.5, 0.025, 0.975]])
    close(_core.stats_ndtri(q), ndtri(q), 1e-15)


def test_mann_kendall_is_pymannkendalls_to_the_bit(z):
    series, out = z["mk/series"], z["mk/out"]
    names = ["s", "var_s", "z", "p", "tau", "slope", "intercept", "h", "trend"]
    for i, row in enumerate(series):
        r = ls.stats.mann_kendall(row)
        np.testing.assert_array_equal([r[k] for k in names], out[i], err_msg=f"series {i}")


@pytest.mark.parametrize("threads", [1, 2])
def test_the_trend_of_every_cell(z, threads):
    series, out = z["mk/series"], z["mk/out"]
    t = series.shape[1]
    maps = [xr.DataArray(series[:, d].reshape(15, 20), dims=("y", "x")) for d in range(t)]
    tr = ls.stats.trend(maps, n_threads=threads)
    for name, c in [("z", 2), ("p", 3), ("tau", 4), ("slope", 5), ("intercept", 6), ("trend", 8)]:
        np.testing.assert_array_equal(getattr(tr, name).values.ravel(), out[:, c], err_msg=name)


def test_a_cell_with_too_few_values_has_no_trend():
    a = np.array([[1.0, np.nan]])
    tr = ls.stats.trend([a, a + 1, a + 2])
    assert tr.slope.values[0, 0] == 1.0
    assert np.isnan(tr.slope.values[0, 1]) and np.isnan(tr.trend.values[0, 1])


@pytest.mark.parametrize("kind", ["rook", "queen"])
def test_moran_lisa_and_gi_are_esdas(z, kind):
    f = z["field"]
    m = ls.stats.moran(f, contiguity=kind, permutations=0)
    close([m.i, m.ei, m.vi_norm, m.z_norm, m.p_norm, m.vi_rand, m.z_rand, m.p_rand], z[f"moran/{kind}"], 1e-12)
    assert np.isnan(m.p_sim) and m.n == f.size

    lm = ls.stats.lisa(f, contiguity=kind, permutations=0)
    close(lm.i.values.ravel(), z[f"lisa/{kind}/I"], 1e-12)
    np.testing.assert_array_equal(lm.quadrant.values.ravel(), z[f"lisa/{kind}/q"])
    close(lm.ei.values.ravel(), z[f"lisa/{kind}/EI"], 1e-12)
    close(lm.vi.values.ravel(), z[f"lisa/{kind}/VI"], 1e-12)
    assert np.isnan(lm.p.values).all()

    gs = ls.stats.getis_ord(f, contiguity=kind)
    close(gs.g.values.ravel(), z[f"gi/{kind}/G"], 1e-12)
    close(gs.z.values.ravel(), z[f"gi/{kind}/Z"], 1e-10)
    close(gs.p.values.ravel(), z[f"gi/{kind}/p"], 1e-9)


def _neighbours(ny, nx, queen):
    out = []
    for r in range(ny):
        for c in range(nx):
            nb = [(r + a) * nx + c + b for a in (-1, 0, 1) for b in (-1, 0, 1)
                  if (a, b) != (0, 0) and (queen or a == 0 or b == 0) and 0 <= r + a < ny and 0 <= c + b < nx]
            out.append(nb)
    return out


def _pseudo_p(larger, perms):
    return (min(larger, perms - larger) + 1) / (perms + 1)


def test_the_permutations_are_numpys_default_rng(z):
    f = z["field"]
    y = f.ravel()
    nb = _neighbours(*f.shape, True)

    def moran_i(v):
        d = v - v.mean()
        return sum(d[k] * d[nb[k]].mean() for k in range(len(v))) / (d @ d)

    perms, seed = 49, 7
    observed = ls.stats.moran(f, permutations=0).i
    rng, s, larger = np.random.default_rng(seed), y.copy(), 0
    for _ in range(perms):
        rng.shuffle(s)
        larger += moran_i(s) >= observed
    assert ls.stats.moran(f, permutations=perms, seed=seed).p_sim == _pseudo_p(larger, perms)

    lm = ls.stats.lisa(f, permutations=perms, seed=seed)
    dev = y - y.mean()
    n, den = len(y), float(np.sum(dev * dev))
    want = np.empty(n)
    for k in range(n):
        c = len(nb[k])
        stat = lambda idx: (n - 1) * dev[k] * (sum(dev[j] for j in idx) / c) / den  # noqa: B023
        obs = stat(nb[k])
        g = np.random.default_rng((seed * 0x9E3779B97F4A7C15 + k) % 2**64)
        hits = 0
        for _ in range(perms):
            pick = g.choice(n - 1, c, replace=False)
            hits += stat(np.where(pick >= k, pick + 1, pick)) >= obs
        want[k] = _pseudo_p(hits, perms)
    np.testing.assert_array_equal(lm.p.values.ravel(), want)


@pytest.mark.parametrize(("seed", "pop", "size"),
                         [(1, 10, 3), (7, 100, 40), (3, 20000, 50), (5, 20000, 1000), (9, 12000, 12000),
                          (2, 5, 5), (4, 1, 1)])
def test_choice_without_replacement_is_numpys(seed, pop, size):
    got, after = _core.stats_numpy_choice(seed, pop, size, 3)
    g = np.random.default_rng(seed)
    np.testing.assert_array_equal(got, g.choice(pop, size, replace=False))
    np.testing.assert_array_equal(after, g.integers(1000, size=3))


def test_missing_cells_threads_and_the_map(z):
    f = z["field"].copy()
    f[0, 0] = np.nan
    da = xr.DataArray(f, dims=("y", "x"), coords={"y": np.arange(20.0)[::-1], "x": np.arange(25.0)})
    m = ls.stats.moran(da, permutations=99)
    assert m.n == 499 and m.i > 0.5 and m.p_sim == 0.01
    a = ls.stats.lisa(da, contiguity="rook", n_threads=1)
    b = ls.stats.lisa(da, contiguity="rook", n_threads=2)
    np.testing.assert_array_equal(a.p.values, b.p.values)
    assert np.isnan(a.p.values[0, 0]) and np.isnan(a.clusters.values[0, 0])
    assert (a.clusters.values == 1).sum() > 20 and (a.clusters.values == 3).sum() > 20
    assert a.i.dims == ("y", "x") and np.array_equal(a.i.y, da.y)
    s = ls.stats.getis_ord(da).spots.values
    assert (s >= 2).any() and (s <= -1).any()
    f2 = z["field"].copy()
    f2[3, 4] = -9999
    assert ls.stats.moran(f2, nodata=-9999, permutations=0).n == 499
    with pytest.raises(ValueError, match="queen or rook"):
        ls.stats.moran(f, contiguity="bishop")


# -- interpolation against R's gstat 2.1 (interpolation.npz, tools/make_reference_interpolation.py) -------------------

@pytest.fixture(scope="module")
def g():
    return dict(np.load(REF / "interpolation.npz"))


def _like(nan_at=None):
    v = np.zeros((20, 25))
    if nan_at is not None:
        v.flat[nan_at] = np.nan
    return xr.DataArray(v, dims=("y", "x"), coords={"y": 78 - 4 * np.arange(20.0), "x": 2 + 4 * np.arange(25.0)})


def test_the_sample_variogram_is_gstats(g):
    pts = g["points"]
    v = ls.stats.variogram(pts, model=None)
    for c in ("np", "dist", "gamma"):
        np.testing.assert_array_equal(v.table[c].to_numpy(float), g[f"vario/{c}"])
    assert v.model is None and v.table["model"].isna().all()
    v = ls.stats.variogram(pts, cutoff=50, width=5, model=None)
    for c in ("np", "dist", "gamma"):
        np.testing.assert_array_equal(v.table[c].to_numpy(float), g[f"vario50/{c}"])


def test_the_fits_are_gstats(g):
    pts, s2 = g["points"], g["s2"][0]
    for name, shape in [("Sph", "spherical"), ("Exp", "exponential"), ("Gau", "gaussian")]:
        if shape == "gaussian":
            # it does not converge in 200 iterations, nor gstat's: the last oscillations carry the libm's last bit
            with pytest.warns(RuntimeWarning, match="converge"):
                v = ls.stats.variogram(pts, model=shape, psill=s2, range=20, nugget=s2 / 10)
        else:
            v = ls.stats.variogram(pts, model=shape, psill=s2, range=20, nugget=s2 / 10)
        got = [v.model.nugget, v.model.psill, v.model.range]
        if shape == "gaussian":
            close(got, g[f"fit{name}/params"], 1e-5)
        else:
            np.testing.assert_array_equal(got, g[f"fit{name}/params"])
            assert v.converged
        np.testing.assert_array_equal(v.table["model"], v.model.gamma(v.table["dist"]))


@pytest.mark.parametrize("threads", [1, 2])
def test_idw_and_kriging_are_gstats(g, threads):
    pts, like = g["points"], _like()
    close(ls.stats.idw(pts, like, n_threads=threads).values.ravel(), g["idw2/pred"], 1e-12)
    close(ls.stats.idw(pts, like, power=2.5, nmax=8, n_threads=threads).values.ravel(), g["idw25/pred"], 1e-12)
    sph = ls.stats.VariogramModel("spherical", psill=1.2, range=30, nugget=0.2)
    for key, nmax in [("ok", None), ("ok10", 10)]:
        k = ls.stats.kriging(pts, like, sph, nmax=nmax, n_threads=threads)
        close(k.prediction.values.ravel(), g[f"{key}/pred"], 1e-10)
        close(k.variance.values.ravel(), g[f"{key}/var"], 1e-10)
    nug, ps, rg = g["fitExp/params"]
    k = ls.stats.kriging(pts, like, {"shape": "exponential", "psill": ps, "range": rg, "nugget": nug},
                         n_threads=threads)
    close(k.prediction.values.ravel(), g["okexp/pred"], 1e-10)
    close(k.variance.values.ravel(), g["okexp/var"], 1e-10)


def test_from_points_to_a_map(g):
    import geopandas as gpd

    p = g["points"]
    gdf = gpd.GeoDataFrame({"z": p[:, 2], "id": np.arange(len(p)).astype(str)},
                           geometry=gpd.points_from_xy(p[:, 0], p[:, 1]), crs="EPSG:32723")
    like = _like(7).rio.write_crs("EPSG:32723")
    v = ls.stats.variogram(gdf, model="exponential", range=20, psill=0.6, nugget=0.06)
    assert len(v.table) == len(g["vario/np"]) and v.converged
    k = ls.stats.kriging(gdf, like, v)
    assert np.isnan(k.prediction.values.flat[7]) and np.isnan(k.variance.values.flat[7])
    assert (k.variance.values[~np.isnan(k.variance.values)] > 0).all()
    assert k.prediction.rio.crs == like.rio.crs
    m = ls.stats.idw(gdf, like, "z").values.ravel()
    close(np.delete(m, 7), np.delete(g["idw2/pred"], 7), 1e-12)
    # the points are reprojected to the map's CRS
    m2 = ls.stats.idw(gdf.to_crs("EPSG:4326"), like, "z").values.ravel()
    close(np.delete(m2, 7), np.delete(g["idw2/pred"], 7), 1e-6)
    # a value missing leaves the point out; a point at a cell's centre gives its value there
    q = np.vstack([p, [[2.0, 78.0, np.nan]], [[6.0, 78.0, 9.0]]])
    assert ls.stats.idw(q, like).values[0, 1] == 9.0
    with pytest.raises(ValueError, match="field"):
        ls.stats.idw(gdf.assign(w=1.0), like)
    with pytest.raises(ValueError, match="no model"):
        ls.stats.kriging(gdf, like, ls.stats.variogram(gdf, model=None))
    with pytest.raises(ValueError, match="singular"):
        ls.stats.kriging(np.array([[1.0, 1, 1], [1, 1, 2], [5, 5, 3]]), like,
                         ls.stats.VariogramModel("spherical", psill=1, range=10))
    with pytest.raises(ValueError, match="coordinates"):
        ls.stats.idw(gdf, np.zeros((3, 3)))
