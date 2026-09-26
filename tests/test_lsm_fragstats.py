# SPDX-License-Identifier: GPL-3.0-or-later
"""FRAGSTATS metrics that landscapemetrics lacks (prox, simi, connect, econ,
cwed, teci) against brute-force implementations of FRAGSTATS' definitions,
and convention="fragstats" against pylandstats."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
import lsm_scenes

MAPS = {
    "clustered": (lsm_scenes.clustered((30, 36), [1, 2, 3, 4], seed=1, na_fraction=0.06), (10.0, 10.0), 8),
    "salt": (lsm_scenes.salt_and_pepper((18, 22), [1, 2, 3], seed=2, na_fraction=0.05), (5.0, 5.0), 4),
    "sparse": (lsm_scenes.sparse((40, 30), [2, 3], seed=3, n_blobs=14, background=1), (2.0, 3.0), 8),
}
CONTRAST = {(1, 2): 0.2, (1, 3): 0.7, (1, 4): 1.0, (2, 3): 0.5, (2, 4): 0.4, (3, 4): 0.9}
SIMILARITY = {(1, 2): 0.5, (1, 3): 0.1, (2, 3): 0.8, (1, 4): 0.0, (2, 4): 0.3, (3, 4): 0.6}


def weight(table, a, b, diagonal):
    if a == b:
        return diagonal
    return table.get((a, b), table.get((b, a), np.nan))


def patch_cells(lm):
    labels = np.asarray(lm.labels())
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs]
    return {i: np.column_stack([(xs[ids == i] + 1) * lm.resolution[0], (ys[ids == i] + 1) * lm.resolution[1]])
            for i in range(1, lm.n_patches + 1)}


def pair_distances(lm):
    cells = patch_cells(lm)
    out = {}
    for a, b in itertools.combinations(cells, 2):
        diff = cells[a][:, None, :] - cells[b][None, :, :]
        out[(a, b)] = out[(b, a)] = float(np.sqrt((diff**2).sum(axis=2).min()))
    return out


@pytest.mark.parametrize("case", sorted(MAPS))
def test_proximity_similarity_connectance(case):
    data, res, directions = MAPS[case]
    radius, threshold = 4.5 * res[0], 2.5 * res[1]
    lm = ls.Landscape(data, res=res, directions=directions, search_radius=radius, connect_threshold=threshold,
                      similarity=SIMILARITY)
    dist = pair_distances(lm)
    p = lm.patch_metrics(["area", "prox", "simi"])
    cls = dict(zip(p["id"], p["class"]))
    area_m2 = dict(zip(p["id"], p["area"] * 10000))
    prox = np.zeros(len(p))
    simi = np.zeros(len(p))
    for (a, b), h in dist.items():
        if h > radius:
            continue
        w = weight(SIMILARITY, cls[a], cls[b], 1.0)
        simi[a - 1] += area_m2[b] * w / h**2
        if cls[a] == cls[b]:
            prox[a - 1] += area_m2[b] / h**2
    np.testing.assert_allclose(p["prox"], prox, rtol=1e-12)
    np.testing.assert_allclose(p["simi"], simi, rtol=1e-12)
    # connectance per class and for the landscape
    c = lm.class_metrics("connect").set_index("class")["connect"]
    joined_all = pairs_all = 0
    for k in lm.classes:
        ids = [i for i in cls if cls[i] == k]
        n = len(ids)
        joined = sum(dist[(a, b)] <= threshold for a, b in itertools.combinations(ids, 2))
        joined_all += joined
        pairs_all += n * (n - 1) / 2
        expected = joined / (n * (n - 1) / 2) * 100 if n > 1 else np.nan
        np.testing.assert_allclose(c[k], expected, rtol=1e-13, equal_nan=True)
    np.testing.assert_allclose(lm.landscape_metrics("connect")["connect"][0], joined_all / pairs_all * 100, rtol=1e-13)


def sides(lm, data):
    """Per cell side: (patch id, own class, neighbour class or None, direction)."""
    labels = np.asarray(lm.labels())
    values = np.ma.getdata(data)
    mask = np.ma.getmaskarray(data)
    ny, nx = values.shape
    out = []
    for y in range(ny):
        for x in range(nx):
            if mask[y, x]:
                continue
            for dx, dy, kind in ((-1, 0, "h"), (1, 0, "h"), (0, -1, "v"), (0, 1, "v")):
                xx, yy = x + dx, y + dy
                nb = None if not (0 <= xx < nx and 0 <= yy < ny) or mask[yy, xx] else int(values[yy, xx])
                if nb == int(values[y, x]):
                    continue
                out.append((int(labels[y, x]), int(values[y, x]), nb, kind))
    return out


@pytest.mark.parametrize("case", sorted(MAPS))
@pytest.mark.parametrize("convention", ["landscapemetrics", "fragstats"])
@pytest.mark.parametrize("count_boundary", [False, True])
def test_contrast_metrics(case, convention, count_boundary):
    data, res, directions = MAPS[case]
    bc = 0.35
    lm = ls.Landscape(data, res=res, directions=directions, contrast=CONTRAST, boundary_contrast=bc,
                      convention=convention, count_boundary=count_boundary)
    rx, ry = res
    lh, lv = (ry, rx) if convention == "fragstats" else (rx, ry)
    econ_num = {}
    edge = {}  # (class, neighbour or None) -> length
    for pid, c, nb, kind in sides(lm, data):
        length = lh if kind == "h" else lv
        w = bc if nb is None else weight(CONTRAST, c, nb, 0.0)
        econ_num[pid] = econ_num.get(pid, 0.0) + w * length
        edge[(c, nb)] = edge.get((c, nb), 0.0) + length
    p = lm.patch_metrics(["perim", "econ"])
    expected = [econ_num.get(i, 0.0) / pr * 100 for i, pr in zip(p["id"], p["perim"])]
    np.testing.assert_allclose(p["econ"], expected, rtol=1e-12)
    ta = lm.landscape_metrics("ta")["ta"][0]
    cls = lm.class_metrics(["cwed", "teci"]).set_index("class")
    for k in lm.classes:
        inner = [(nb, e) for (c, nb), e in edge.items() if c == k and nb is not None]
        boundary = edge.get((k, None), 0.0)
        weighted = sum(weight(CONTRAST, k, nb, 0.0) * e for nb, e in inner)
        total = sum(e for _, e in inner)
        cwed = (weighted + (bc * boundary if count_boundary else 0)) / ta
        teci = (weighted + bc * boundary) / (total + boundary) * 100
        np.testing.assert_allclose(cls.loc[k, "cwed"], cwed, rtol=1e-12)
        np.testing.assert_allclose(cls.loc[k, "teci"], teci, rtol=1e-12)
    land = lm.landscape_metrics(["cwed", "teci"])
    pairs = [(c, nb, e) for (c, nb), e in edge.items() if nb is not None and c < nb]
    boundary = sum(e for (c, nb), e in edge.items() if nb is None)
    weighted = sum(weight(CONTRAST, c, nb, 0.0) * e for c, nb, e in pairs)
    total = sum(e for _, _, e in pairs)
    np.testing.assert_allclose(land["cwed"][0], (weighted + (bc * boundary if count_boundary else 0)) / ta, rtol=1e-12)
    np.testing.assert_allclose(land["teci"][0], (weighted + bc * boundary) / (total + boundary) * 100, rtol=1e-12)


def test_missing_parameters_give_nan():
    data, res, _ = MAPS["clustered"]
    lm = ls.Landscape(data, res=res)
    assert lm.patch_metrics(["prox", "simi"])[["prox", "simi"]].isna().all(axis=None)
    assert lm.class_metrics("connect")["connect"].isna().all()
    # no contrast table: NaN, as FRAGSTATS needs one
    assert lm.patch_metrics("econ")["econ"].isna().all()
    assert lm.class_metrics(["cwed", "teci"])[["cwed", "teci"]].isna().all(axis=None)
    partial = ls.Landscape(data, res=res, contrast={(1, 2): 1.0})
    econ = partial.patch_metrics("econ")
    assert econ["econ"].isna().any() and econ["econ"].notna().any()


def test_similarity_default_is_proximity():
    data, res, _ = MAPS["clustered"]
    p = ls.Landscape(data, res=res, search_radius=60).patch_metrics(["prox", "simi"])
    np.testing.assert_array_equal(p["prox"], p["simi"])


def test_weight_tables():
    data, res, _ = MAPS["clustered"]
    frame = pd.DataFrame(np.nan, index=[1, 2, 3, 4], columns=[1, 2, 3, 4])
    for (a, b), w in CONTRAST.items():
        frame.loc[a, b] = frame.loc[b, a] = w
    nested = {}
    for (a, b), w in CONTRAST.items():
        nested.setdefault(a, {})[b] = w
    a = ls.Landscape(data, res=res, contrast=CONTRAST).patch_metrics("econ")
    b = ls.Landscape(data, res=res, contrast=frame).patch_metrics("econ")
    c = ls.Landscape(data, res=res, contrast=nested).patch_metrics("econ")
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)
    with pytest.raises(ValueError, match="convention"):
        ls.Landscape(data, convention="other")


def test_extras_selection():
    full = ls.list_metrics(extras=True)
    for m, lv in [("prox", "patch"), ("econ", "patch"), ("connect", "class"), ("teci", "landscape"),
                  ("prox_am", "class"), ("econ_mn", "landscape")]:
        assert ((full["metric"] == m) & (full["level"] == lv)).any()
    assert "contrast metric" in set(full["type"])
    assert not ls.list_metrics()["metric"].isin(["prox", "cwed"]).any()
    lm = ls.Landscape(MAPS["clustered"][0], search_radius=40)
    assert lm.class_metrics(["proximity_index_mn"]).shape == (4, 2)


# -- convention = "fragstats" -------------------------------------------------------


@pytest.fixture(scope="module")
def pylandstats():
    return pytest.importorskip("pylandstats")


@pytest.mark.parametrize("res", [(10.0, 10.0), (30.0, 20.0), (2.0, 5.0)])
def test_fragstats_perimeter_matches_pylandstats(res, pylandstats):
    data = lsm_scenes.clustered((25, 30), [1, 2, 3], seed=5, sigma=1.5).data
    lm = ls.Landscape(data, res=res, convention="fragstats")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = pylandstats.Landscape(data, res=res, nodata=0, neighborhood_rule="8").compute_patch_metrics_df()
    np.testing.assert_allclose(np.sort(lm.patch_metrics("perim")["perim"]), np.sort(pp["perimeter"]), rtol=1e-12)
    # the landscapemetrics convention differs when the cells are not square
    lm_perim = np.sort(ls.Landscape(data, res=res).patch_metrics("perim")["perim"])
    assert np.allclose(lm_perim, np.sort(pp["perimeter"])) == (res[0] == res[1])


@pytest.mark.parametrize("seed", [6, 7])
def test_fragstats_dcore_matches_pylandstats(seed, pylandstats):
    data = lsm_scenes.clustered((40, 45), [1, 2, 3], seed=seed, sigma=2.5).data
    lm = ls.Landscape(data, res=10, convention="fragstats")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pl = pylandstats.Landscape(data, res=(10, 10), nodata=0, neighborhood_rule="8")
        for stat in ("mn", "am", "md", "ra"):
            theirs = pl.compute_class_metrics_df(metrics=[f"disjunct_core_area_{stat}"])
            ours = lm.class_metrics(f"dcore_{stat}").set_index("class")[f"dcore_{stat}"]
            np.testing.assert_allclose(ours.to_numpy(), theirs.loc[ours.index].to_numpy(dtype=float).ravel(),
                                       rtol=1e-12, err_msg=stat)
            # (pylandstats' landscape-level _am also averages its class column: use its core sizes)
            sizes = pl._disjunct_core_area()["core_area"].to_numpy()
            expected = {"mn": sizes.mean(), "am": np.average(sizes, weights=sizes), "md": np.median(sizes),
                        "ra": np.ptp(sizes)}[stat]
            np.testing.assert_allclose(lm.landscape_metrics(f"dcore_{stat}")[f"dcore_{stat}"][0], expected,
                                       rtol=1e-12, err_msg=stat)
    # landscapemetrics' dcore: the number of core areas per patch
    counts = ls.Landscape(data, res=10).class_metrics("dcore_mn")["dcore_mn"]
    ncore = ls.Landscape(data, res=10).patch_metrics("ncore")
    np.testing.assert_allclose(counts, ncore.groupby("class")["ncore"].mean().to_numpy())


def test_out_of_core(tmp_path):
    import lsm_reference

    data, res, _ = MAPS["sparse"]
    tif = tmp_path / "m.tif"
    lsm_reference.write_landscape(tif, data, res)
    opts = {"contrast": {(1, 2): 0.3, (1, 3): 0.6, (2, 3): 0.9}, "boundary_contrast": 0.2, "convention": "fragstats"}
    tiled = ls.Landscape(tif, block_rows=64, **opts)
    mem = ls.Landscape(tif, **opts)
    pd.testing.assert_frame_equal(tiled.class_metrics(["cwed", "teci", "te"]), mem.class_metrics(["cwed", "teci", "te"]))
    for metric in ("prox", "econ"):
        with pytest.raises(NotImplementedError):
            tiled.patch_metrics(metric)
    with pytest.raises(NotImplementedError):
        tiled.class_metrics("dcore_mn")
