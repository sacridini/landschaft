# SPDX-License-Identifier: GPL-3.0-or-later
"""Landscape identification and comparison over any surface: on a Zonation rank, the same bits as the
Zonation post-processing already ported (area statistics of rank ranges and separate areas, solution
comparison); the rest against a direct implementation (scipy)."""

from __future__ import annotations

import numpy as np
import optimize_cases
import pytest
from scipy import ndimage

import landschaft as ls

P = ls.prioritize


@pytest.fixture(scope="module")
def ranking():
    x = np.random.default_rng(3).random((3, 30, 40))
    x[:, :4, :6] = np.nan  # cells without data
    return P.zonation(x, rule="CAZ2")


@pytest.fixture(scope="module")
def ranking_b():
    x = np.random.default_rng(3).random((3, 30, 40))
    x[:, :4, :6] = np.nan
    return P.zonation(x, rule="ABF")


def test_position_rank_is_the_zonation_rank(ranking):
    r = np.asarray(ranking.rank, dtype=np.float32)
    pr = P.position_rank(ranking.rank)
    np.testing.assert_array_equal(np.isnan(pr), ~np.isfinite(r))
    ok = np.isfinite(r)
    np.testing.assert_array_equal(pr[ok].view(np.uint32), r[ok].view(np.uint32))


@pytest.mark.parametrize(("fraction", "min_size"), [(0.1, 1), (0.2, 3), (0.35, 5), (1.0, 1)])
def test_identification_is_zonation(ranking, fraction, min_size):
    lsi = P.eval_landscape_identification(ranking, fraction, min_size=min_size)
    st = ranking.area_statistics_separate_areas(1 - fraction, 1.0, min_area_size=min_size)
    areas = np.asarray(lsi.areas)
    np.testing.assert_array_equal(np.nan_to_num(areas, nan=0).astype(np.int32), np.asarray(st.mask))
    pos = st.areas[st.areas["features"] == "pos"].reset_index(drop=True)
    np.testing.assert_array_equal(lsi.table["area"], pos["area"])
    np.testing.assert_array_equal(lsi.table["cells"], pos["size"])
    np.testing.assert_array_equal(lsi.table["mean_value"].to_numpy(), pos["avgrank"].to_numpy())
    top = ranking.area_statistics_rank_range(1 - fraction)
    t = np.asarray(lsi.top)
    np.testing.assert_array_equal(np.nan_to_num(t, nan=0).astype(np.uint8), np.nan_to_num(np.asarray(top.mask)))


def test_comparison_is_zonation(ranking, ranking_b):
    cmp = P.eval_landscape_comparison(ranking, ranking_b, fractions=[0.5, 0.25, 0.125, 0.1, 0.3])
    for row, band in zip(cmp.table.itertuples(), cmp.maps["band"].values):
        f = row.fraction
        if f in (0.5, 0.25, 0.125):  # exact in single precision: Zonation's float comparison agrees
            z = ranking.solution_comparison(ranking_b.rank, 1 - f, 1.0, 1 - f, 1.0)
            assert (row.both, row.only_a, row.only_b, row.neither) == (z["both"], z["only_this"], z["only_other"],
                                                                     z["neither"])
            np.testing.assert_array_equal(np.nan_to_num(cmp.maps.sel(band=band).values), z["map"])
        a = np.nan_to_num(np.asarray(ranking.area_statistics_rank_range(1 - f).mask)) == 1
        b = np.nan_to_num(np.asarray(ranking_b.area_statistics_rank_range(1 - f).mask)) == 1
        assert row.both == int((a & b).sum()) and row.only_a == int((a & ~b).sum())
        assert row.only_b == int((~a & b).sum())
        assert row.jaccard == pytest.approx((a & b).sum() / (a | b).sum(), rel=0, abs=0)


def _direct(values, fraction, min_size, merge, conn):
    """The rule by hand: a stable sort, the top cells, dilation, labels, sizes."""
    flat = values.ravel()
    ok = np.flatnonzero(np.isfinite(flat))
    order = ok[np.argsort(flat[ok], kind="stable")]
    rank = np.full(flat.size, -1.0)
    rank[order] = (np.arange(ok.size) / ok.size).astype(np.float32)
    top = (rank >= 1 - fraction).reshape(values.shape)
    s = np.ones((3, 3), bool) if conn == 8 else ndimage.generate_binary_structure(2, 1)
    w = ndimage.binary_dilation(top, s, iterations=merge) if merge else top
    lab, _ = ndimage.label(w, s)
    lab[~top] = 0
    sizes = np.bincount(lab.ravel())
    lab[sizes[lab] < min_size] = 0
    return top, lab


@pytest.mark.parametrize(("merge", "conn"), [(0, 4), (1, 8), (2, 4)])
def test_identification_rule_by_hand(merge, conn):
    rng = np.random.default_rng(5)
    v = ndimage.uniform_filter(rng.random((40, 50)), 5)
    v = np.round(v, 2)  # ties
    v[10:13, 20:30] = np.nan
    feats = rng.random((2, 40, 50))
    lsi = P.eval_landscape_identification(v, 0.15, min_size=4, merge_distance=merge, connectivity=conn,
                                          features=feats)
    top, lab = _direct(v, 0.15, 4, merge, conn)
    np.testing.assert_array_equal(np.nan_to_num(lsi.areas, nan=0).astype(int), lab)
    np.testing.assert_array_equal(np.nan_to_num(lsi.top, nan=0).astype(bool), top)
    ids = np.unique(lab[lab > 0])
    np.testing.assert_array_equal(lsi.table["area"], ids)
    for k, name in enumerate(("f1", "f2")):
        held = np.array([feats[k][lab == i].sum() for i in ids])
        np.testing.assert_allclose(lsi.table[name], held, rtol=1e-12)
        np.testing.assert_allclose(lsi.table[f"{name}_share"], held / feats[k].sum(), rtol=1e-12)
    np.testing.assert_allclose(lsi.table["mean_value"], [v[lab == i].mean() for i in ids], rtol=1e-12)


def test_ties_and_inputs():
    v = np.array([[1.0, 1.0, 1.0, 1.0]])
    lsi = P.eval_landscape_identification(v, 0.5)
    np.testing.assert_array_equal(lsi.top, [[0, 0, 1, 1]])  # ties: the later cells rank higher
    with pytest.raises(ValueError, match="within"):
        P.eval_landscape_identification(v, 0.0)
    with pytest.raises(ValueError, match="connectivity"):
        P.eval_landscape_identification(v, 0.5, connectivity=6)
    with pytest.raises(ValueError, match="same grid"):
        P.eval_landscape_comparison(v, np.ones((2, 2)))


def test_problem_surfaces():
    """A Ferrier total, a portfolio's frequency and a solution, with the problem's features and cost."""
    p = optimize_cases.CASES["minset_raster_relative"].py().add_highs_solver(gap=0)
    s = p.solve()
    fer = P.eval_ferrier_importance(p, s)
    lsi = P.eval_landscape_identification(fer, 0.3, problem=p)
    u = p.units
    lab = np.nan_to_num(lsi.areas.values, nan=0).ravel()[u.index]
    for i, row in lsi.table.iterrows():
        inside = lab == row["area"]
        assert row["cost"] == pytest.approx(u.cost[inside].sum(), rel=1e-12)
        assert row["f1"] == pytest.approx(u.rij[0].toarray().ravel()[inside].sum(), rel=1e-12)
        assert row["f1_share"] == pytest.approx(row["f1"] / u.totals[0], rel=1e-12)
    assert lsi.areas.rio.crs == optimize_cases.pu_cost().rio.crs
    pf = p.add_cuts_portfolio(3).solve()
    cmp = P.eval_landscape_comparison(pf, s, fractions=[0.2])
    assert cmp.table["cells_a"].iloc[0] == cmp.table["cells_b"].iloc[0]
    assert "Plot" in repr(ls.plot(lsi))
    assert "Plot" in repr(ls.plot(cmp))
