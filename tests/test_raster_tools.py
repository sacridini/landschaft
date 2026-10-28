# SPDX-License-Identifier: GPL-3.0-or-later
"""The raster tools from Terraflux: cell statistics of a stack, local correlation, viewshed, zonal maps, tabulate."""

import math

import numpy as np
import pytest
import xarray as xr

import landschaft as ls


@pytest.fixture(scope="module")
def stack():
    g = np.random.default_rng(3)
    s = np.round(g.normal(size=(7, 30, 40)) * 3, 1)
    s[g.random(s.shape) < 0.2] = np.nan
    s[:, 0, 0] = np.nan  # a cell without a value in any layer
    s[2, 1, 1] = s[5, 1, 1] = 100.0  # a tie of the largest
    return s


@pytest.mark.parametrize("threads", [1, 2])
def test_cell_stats_are_numpys(stack, threads):
    import warnings

    warnings.simplefilter("ignore", RuntimeWarning)
    k = stack.shape[0]
    want = {
        "mean": np.nanmean(stack, 0), "std": np.nanstd(stack, 0), "var": np.nanvar(stack, 0),
        "min": np.nanmin(stack, 0), "max": np.nanmax(stack, 0), "median": np.nanmedian(stack, 0),
        "p90": np.nanpercentile(stack, 90, axis=0), "p7.5": np.nanpercentile(stack, 7.5, axis=0),
        "count": np.sum(~np.isnan(stack), 0).astype(float), "nodata": np.sum(np.isnan(stack), 0).astype(float),
    }
    want["range"] = want["max"] - want["min"]
    want["sum"] = np.where(want["count"] > 0, np.nansum(stack, 0), np.nan)
    for stat, w in want.items():
        got = ls.stats.cell_stats(stack, stat, n_threads=threads).values
        np.testing.assert_allclose(got, w, rtol=1e-12, atol=1e-12, err_msg=stat)
    flat = stack.reshape(k, -1)
    arg = {"argmax": np.full(flat.shape[1], np.nan), "argmin": np.full(flat.shape[1], np.nan)}
    maj, mino, uni = (np.full(flat.shape[1], np.nan) for _ in range(3))
    uni[:] = 0
    for c in range(flat.shape[1]):
        v = flat[:, c]
        if np.isnan(v).all():
            continue
        arg["argmax"][c] = np.nanargmax(v) + 1
        arg["argmin"][c] = np.nanargmin(v) + 1
        vals, counts = np.unique(v[~np.isnan(v)], return_counts=True)
        maj[c], mino[c], uni[c] = vals[np.argmax(counts)], vals[np.argmin(counts)], len(vals)
    for stat, w in [*arg.items(), ("majority", maj), ("minority", mino), ("unique", uni)]:
        np.testing.assert_array_equal(ls.stats.cell_stats(stack, stat, n_threads=threads).values.ravel(), w,
                                      err_msg=stat)
    assert ls.stats.cell_stats(stack, "argmax").values[1, 1] == 3.0  # the first of a tie


def test_cell_stats_of_maps_keep_the_grid(stack):
    maps = [xr.DataArray(m, dims=("y", "x"), coords={"y": np.arange(30.0)[::-1], "x": np.arange(40.0)})
            .rio.write_crs("EPSG:32723") for m in stack]
    out = ls.stats.cell_stats(maps, "argmax")
    assert out.rio.crs == maps[0].rio.crs and out.name == "argmax"
    with pytest.raises(ValueError, match="unknown statistic"):
        ls.stats.cell_stats(maps, "mode")


def _local(x, y, r, circle, least):
    ny, nx = x.shape
    out = np.full((4, ny, nx), np.nan)
    for i in range(ny):
        for j in range(nx):
            if np.isnan(x[i, j]) or np.isnan(y[i, j]):
                continue
            a, b = [], []
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    ii, jj = i + di, j + dj
                    if (circle and di * di + dj * dj > r * r) or not (0 <= ii < ny and 0 <= jj < nx):
                        continue
                    if np.isnan(x[ii, jj]) or np.isnan(y[ii, jj]):
                        continue
                    a.append(x[ii, jj])
                    b.append(y[ii, jj])
            if len(a) < max(least, 2):
                continue
            a, b = np.array(a), np.array(b)
            vxx, vyy = np.sum((a - a.mean()) ** 2), np.sum((b - b.mean()) ** 2)
            vxy = np.sum((a - a.mean()) * (b - b.mean()))
            rho = vxy / math.sqrt(vxx * vyy)
            out[:, i, j] = rho, vxy / vxx, b.mean() - vxy / vxx * a.mean(), rho * rho
    return out


@pytest.mark.parametrize(("radius", "shape", "least"), [(1, "square", 3), (3, "circle", 5), (2, "square", 20)])
def test_local_correlation_is_the_window_regression(radius, shape, least):
    g = np.random.default_rng(5)
    x = g.normal(size=(25, 30)) * 10 + 1000
    y = 0.5 * x + g.normal(size=x.shape) * 3
    x[g.random(x.shape) < 0.1] = np.nan
    y[g.random(y.shape) < 0.1] = np.nan
    lc = ls.stats.local_correlation(x, y, radius=radius, shape=shape, min_cells=least, n_threads=2)
    want = _local(x, y, radius, shape == "circle", least)
    for k, name in enumerate(("correlation", "slope", "intercept", "r2")):
        got = getattr(lc, name).values
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want[k]), err_msg=name)
        np.testing.assert_allclose(got, want[k], rtol=1e-9, atol=1e-9, err_msg=name)
    a = ls.stats.local_correlation(x, y, radius=radius, shape=shape, min_cells=least, n_threads=1)
    np.testing.assert_array_equal(a.slope.values, lc.slope.values)


def _viewshed_one(z, orow, ocol, rw, rh, eye, target, reach, curvature):
    """Terraflux's raster.viewshed of one observer, transcribed."""
    h, w = z.shape
    drop = (lambda d: 0.85714 * d * d / (2.0 * 6_371_008.8)) if curvature else (lambda d: 0.0)
    seen = np.zeros(z.shape, bool)
    visited = np.zeros(z.shape, bool)
    if np.isnan(z[orow, ocol]):
        return seen
    z0 = z[orow, ocol] + eye
    kr, kc = int(min(np.ceil(reach / rh), h)), int(min(np.ceil(reach / rw), w))
    r0, r1, c0, c1 = max(orow - kr, 0), min(orow + kr, h - 1), max(ocol - kc, 0), min(ocol + kc, w - 1)
    seen[orow, ocol] = visited[orow, ocol] = True

    def rnd(v):  # Rust's round: half away from zero
        return math.floor(abs(v) + 0.5) * (1 if v >= 0 else -1)

    def dist(rr, cc):
        return math.sqrt(((rr - orow) * rh) ** 2 + ((cc - ocol) * rw) ** 2)

    ends = [e for c in range(c0, c1 + 1) for e in ((r0, c), (r1, c))]
    ends += [e for r in range(r0, r1 + 1) for e in ((r, c0), (r, c1))]
    for er, ec in ends:
        dr, dc = er - orow, ec - ocol
        steps = max(abs(dr), abs(dc))
        steepest = -math.inf
        for k in range(1, steps + 1):
            rr, cc = orow + rnd(dr * k / steps), ocol + rnd(dc * k / steps)
            v = z[rr, cc]
            if np.isnan(v):
                continue
            d = dist(rr, cc)
            if d > reach:
                break
            visited[rr, cc] = True
            v -= drop(d)
            if (v + target - z0) / d >= steepest:
                seen[rr, cc] = True
            steepest = max(steepest, (v - z0) / d)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            d = dist(rr, cc)
            if visited[rr, cc] or np.isnan(z[rr, cc]) or d > reach:
                continue
            dr, dc = rr - orow, cc - ocol
            steps = max(abs(dr), abs(dc))
            steepest = -math.inf
            for k in range(1, steps):
                r, c = orow + rnd(dr * k / steps), ocol + rnd(dc * k / steps)
                if not np.isnan(z[r, c]):
                    e = dist(r, c)
                    steepest = max(steepest, (z[r, c] - drop(e) - z0) / e)
            seen[rr, cc] = (z[rr, cc] - drop(d) + target - z0) / d >= steepest
    return seen


@pytest.mark.parametrize(("reach", "curvature", "target"), [(math.inf, False, 0.0), (400.0, True, 1.5)])
def test_viewshed_is_terrafluxs(reach, curvature, target):
    g = np.random.default_rng(11)
    yy, xx = np.mgrid[0:40, 0:50]
    z = 100 + 30 * np.sin(xx / 6) * np.cos(yy / 7) + g.normal(size=xx.shape)
    z[g.random(z.shape) < 0.03] = np.nan
    rw, rh = 20.0, 25.0
    extent = (0.0, 50 * rw, 0.0, 40 * rh)
    obs = [(3, 4), (20, 25), (39, 49), (10, 40)]
    xy = np.array([[(c + 0.5) * rw, extent[3] - (r + 0.5) * rh] for r, c in obs])
    got = ls.surface.viewshed(z, xy, extent=extent, radius=None if math.isinf(reach) else reach,
                              curvature=curvature, target_height=target, n_threads=2).values
    want = sum(_viewshed_one(z, r, c, rw, rh, 2.0, target, reach, curvature).astype(float) for r, c in obs)
    want = np.where(np.isnan(z), np.nan, want)
    np.testing.assert_array_equal(got, want)
    one = ls.surface.viewshed(z, xy[1], extent=extent, n_threads=1).values
    np.testing.assert_array_equal(one, np.where(np.isnan(z), np.nan, _viewshed_one(z, 20, 25, rw, rh, 2.0, 0.0,
                                                                                     math.inf, False)))


def test_viewshed_of_points():
    import geopandas as gpd

    dem = xr.DataArray(np.zeros((10, 10)), dims=("y", "x"),
                       coords={"y": 9.5 - np.arange(10.0), "x": np.arange(10.0) + 0.5}).rio.write_crs("EPSG:32723")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy([2.5, 7.5, 50.0], [2.5, 7.5, 50.0]), crs="EPSG:32723")
    v = ls.surface.viewshed(dem, pts)
    assert (v.values == 2).all() and v.rio.crs == dem.rio.crs  # flat: both see everything; the third is off the map
    near = ls.surface.viewshed(dem, (0.5, 9.5), radius=3)
    assert near.values[0, 3] == 1 and near.values[0, 4] == 0 and near.values[3, 3] == 0


def test_zonal_map_and_tabulate_area():
    g = np.random.default_rng(2)
    coords = {"y": 99.5 - np.arange(100.0) * 1, "x": np.arange(120.0) + 0.5}
    vals = xr.DataArray(g.normal(size=(100, 120)), dims=("y", "x"), coords=coords).rio.write_crs("EPSG:32723")
    zones = xr.DataArray(g.integers(1, 6, size=(100, 120)).astype(np.int32), dims=("y", "x"), coords=coords)
    zones = zones.rio.write_crs("EPSG:32723").rio.write_nodata(5)
    t = ls.zonal_stats(vals, zones, ["mean", "p90"])
    m = ls.zonal_map(vals, zones, "p90")
    z = zones.values
    for _, row in t.iterrows():
        assert np.all(m.values[z == row.zone] == row.p90)
    assert np.isnan(m.values[z == 5]).all() and 5 not in t.zone.to_numpy()
    st = xr.concat([vals, vals * 2], dim="band").assign_coords(band=[1, 2])
    ms = ls.zonal_map(st, zones, "mean")
    assert ms.shape == (2, 100, 120)
    np.testing.assert_allclose(ms.values[1], 2 * ls.zonal_map(vals, zones).values, rtol=1e-12)

    classes = xr.DataArray(g.integers(10, 13, size=(100, 120)).astype(np.int32), dims=("y", "x"), coords=coords)
    classes = classes.rio.write_crs("EPSG:32723")
    cells = ls.tabulate_area(zones, classes)
    want = ls.simulate.crosstab(zones, classes)
    assert list(cells.zone) == [1, 2, 3, 4] and list(cells.columns) == ["zone", "class_10", "class_11", "class_12",
                                                                      "total"]
    np.testing.assert_array_equal(cells.iloc[:, 1:4].to_numpy(), want.loc[[1, 2, 3, 4], [10, 11, 12]].to_numpy())
    assert (cells.total == cells.iloc[:, 1:4].sum(axis=1)).all()
    ha = ls.tabulate_area(zones, classes, units="hectares")
    np.testing.assert_allclose(ha.total, cells.total / 10_000)
    share = ls.tabulate_area(zones, classes, units="share")
    np.testing.assert_allclose(share.total, 1.0)
    with pytest.raises(ValueError, match="units"):
        ls.tabulate_area(zones, classes, units="acres")
