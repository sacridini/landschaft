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
    from scipy.special import ndtri

    np.testing.assert_array_equal(_core.stats_ndtr(z["ndtr/x"]), z["ndtr/y"])
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
