# SPDX-License-Identifier: GPL-3.0-or-later
"""landscapemetrics' utility functions: centroids, nearest neighbours,
circumscribing circles, adjacencies, boundaries, patches, extract_lsm,
check_landscape and calculate_correlation.

``tests/data/reference/lsm_utils`` holds their results for the stored maps
(``make_reference_lsm.py utils``); all must be the same bits, except the
circles (landscapemetrics shuffles Welzl's points at random).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import landschaft as ls
import lsm_scenes
from conftest import DATA, same_bits
from test_lsm_reference import NOT_BITWISE, ULPS

REF_DIR = DATA / "reference" / "lsm_utils"
MANIFEST = json.loads((REF_DIR / "manifest.json").read_text())
CASES = sorted(MANIFEST["files"])


def read(case, name: str) -> pd.DataFrame | None:
    folder = REF_DIR / case if isinstance(case, str) else case
    path = folder / f"{name}.csv.gz"
    if not path.exists():
        path = folder / f"{name}.csv"
        if not path.exists():
            return None
    return pd.read_csv(path, na_values=["NA"], keep_default_na=False, float_precision="round_trip")


def matrix(case, name: str, shape) -> np.ndarray:
    r = read(case, name)
    out = np.full(shape, np.nan)
    out[r["row"] - 1, r["col"] - 1] = r["value"]
    return out


def assert_same(a, b, label: str) -> None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"{label}: shapes {a.shape} and {b.shape}"
    same = same_bits(a, b, ulps=ULPS)
    assert same.all(), f"{label}: {int((~same).sum())} values differ, first at {np.argwhere(~same)[:3].tolist()}"


def landscape(case: str) -> ls.Landscape:
    info = MANIFEST["files"][case]
    directions = info["options"].get("directions", 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ls.Landscape(DATA / "lsm" / info["input"], directions=directions)


def boundary_options(case: str) -> dict:
    o = MANIFEST["files"][case]["options"]
    return {"consider_boundary": o.get("consider_boundary", False), "edge_depth": o.get("edge_depth", 1)}


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("cell_center", [False, True])
def test_centroids(case, cell_center):
    ref = read(case, f"centroids_{str(cell_center).lower()}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = landscape(case).centroids(cell_center=cell_center)
    assert_same(ours[["class", "id", "x", "y"]], ref[["class", "id", "x", "y"]], f"{case} centroids")


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_nearest_neighbours(case):
    ref = read(case, "nearestneighbour")
    lm = landscape(case)
    ours = lm.nearest_neighbours(return_id=True)
    ours = ours[ours["dist"].notna()]
    if ref is None:
        assert ours.empty
        return
    key = ["id", "id_neighbour"]
    ours = ours.sort_values(key).reset_index(drop=True)
    ref = ref.sort_values(key).reset_index(drop=True)
    assert_same(ours[["class", "id", "dist", "id_neighbour"]].astype(float), ref[["class", "id", "dist", "id_neighbour"]],
                f"{case} nearest neighbours")
    # the distances are the enn metric
    dist = lm.nearest_neighbours().set_index("id")["dist"]
    np.testing.assert_allclose(dist.to_numpy(), lm.patch_metrics("enn")["enn"].to_numpy(), rtol=1e-15)


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("level", ["patch", "class"])
def test_circumscribing_circles(case, level):
    ref = read(case, f"circle_{level}")
    if ref is None:  # non-square cells: landscapemetrics refuses
        pytest.skip("landscapemetrics has no circles for non-square cells")
    ours = landscape(case).circumscribing_circles(level)
    np.testing.assert_array_equal(ours["class"], ref["class"])
    # landscapemetrics' random shuffle moves the last bit
    np.testing.assert_allclose(ours[["diameter", "center_x", "center_y"]].to_numpy(),
                               ref[["value", "center_x", "center_y"]].to_numpy(), rtol=1e-13)


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_adjacencies(case):
    lm = landscape(case)
    k = len(lm.classes)
    for nb in (4, 8):
        for what in ("full", "like", "unlike", "triangle"):
            for upper in (False, True):
                ours = lm.adjacencies(nb, what, upper)
                assert list(ours.index) == list(lm.classes) == list(ours.columns)
                assert_same(ours, matrix(case, f"adjacencies_{nb}_{what}_{str(upper).lower()}", (k, k)),
                            f"{case} {nb} {what} {upper}")
    custom = np.array(MANIFEST["neighbourhood"], dtype=float)
    assert_same(lm.adjacencies(custom), matrix(case, "adjacencies_custom", (k, k)), f"{case} custom")


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_boundaries(case):
    lm = landscape(case)
    opts = boundary_options(case)
    tif = DATA / "lsm" / MANIFEST["files"][case]["input"]
    for patch_id in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = ls.raster_boundaries(tif, patch_id=patch_id, **opts)
        assert_same(ours, matrix(case, f"boundaries_values_{str(patch_id).lower()}", lm.shape),
                    f"{case} raster values patch_id={patch_id}")
    assert_same(lm.boundaries(patch_id=True, **opts), matrix(case, "boundaries_patches", lm.shape), f"{case} patches")


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_extract(case):
    ref = read(case, "extract")
    points = pd.read_csv(REF_DIR / case / "points.csv", float_precision="round_trip").to_numpy()
    lm = landscape(case)
    ours = lm.extract_metrics(points)
    if ref is None or ref.empty:
        assert ours.empty
        return
    if (ref["metric"] == "circle").sum() == 0:
        ours = ours[ours["metric"] != "circle"].reset_index(drop=True)
    assert list(ours["metric"].astype(str)) == list(ref["metric"])
    assert list(ours["extract_id"]) == list(ref["extract_id"])
    assert_same(ours[["class", "id"]].astype(float), ref[["class", "id"]], f"{case} extract ids")
    loose = ours["metric"].astype(str).str.startswith(NOT_BITWISE).to_numpy()
    assert_same(ours.loc[~loose, "value"], ref.loc[~loose, "value"], f"{case} extract values")
    np.testing.assert_allclose(ours.loc[loose, "value"], ref.loc[loose, "value"], rtol=1e-10)


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
def test_check_landscape(case):
    ref = read(case, "check").iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = ls.check_landscape(DATA / "lsm" / MANIFEST["files"][case]["input"], verbose=False).iloc[0]
    for col in ("crs", "units", "class", "n_classes"):
        a, b = ours[col], ref[col]
        assert (pd.isna(a) and pd.isna(b)) or a == b, f"{case} {col}: {a!r} vs {b!r}"
    assert f"U+{ord(ours['OK']):04X}" == ref["OK"]


@pytest.mark.reference
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_correlation(case, method):
    lm = landscape(case)
    what = ["lsm_p_area", "lsm_p_perim", "lsm_p_shape", "lsm_p_frac", "lsm_p_contig", "lsm_p_enn",
            "lsm_c_ca", "lsm_c_np", "lsm_c_pland", "lsm_c_te", "lsm_c_lpi", "lsm_c_enn_mn"]
    metrics = lm.metrics(what=what)
    if case == "augusta":
        stack = [DATA / "lsm" / "augusta.tif"] + [DATA / "lsm" / f for f in MANIFEST["stack"]]
        land = ls.landscape_metrics(stack, what=["lsm_l_ta", "lsm_l_np", "lsm_l_shdi", "lsm_l_contag", "lsm_l_ed",
                                                 "lsm_l_area_mn"])
        metrics = pd.concat([metrics.assign(layer=1), land], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ls.calculate_correlation(metrics, method=method)
    for level, ours in result.items():
        ref = read(case, f"correlation_{method}_{level}")
        assert list(ours["metric_1"]) == list(ref["metric_1"]), level
        assert list(ours["metric_2"]) == list(ref["metric_2"]), level
        assert_same(ours["value"], ref["value"], f"{case} {method} {level}")


# -- live comparison -------------------------------------------------------------------


@pytest.fixture(scope="module")
def rscript():
    import lsm_reference

    exe = lsm_reference.find_lsm_rscript()
    if exe is None:
        pytest.skip("Rscript with terra and landscapemetrics not found (set LANDSCHAFT_RSCRIPT, LANDSCHAFT_R_LIBS)")
    return exe


@pytest.mark.live_reference
@pytest.mark.parametrize(
    "seed, maker, res, options",
    [
        (1, "clustered", (10.0, 10.0), {}),
        (2, "salt", (2.0, 2.0), {"directions": 4, "consider_boundary": True, "edge_depth": 2}),
        (3, "sparse", (5.0, 5.0), {"edge_depth": 3}),
    ],
)
def test_live_utilities(rscript, tmp_path, seed, maker, res, options):
    import lsm_reference

    make = {"clustered": lambda: lsm_scenes.clustered((30, 36), [1, 2, 3, 5], seed=seed, na_fraction=0.08),
            "salt": lambda: lsm_scenes.salt_and_pepper((20, 25), [1, 2, 3], seed=seed, na_fraction=0.05),
            "sparse": lambda: lsm_scenes.sparse((40, 30), [2, 3], seed=seed, n_blobs=12)}[maker]
    data = make()
    tif = tmp_path / "in.tif"
    lsm_reference.write_landscape(tif, data, res)
    rng = np.random.default_rng(seed)
    ny, nx = data.shape
    pts = pd.DataFrame({"x": rng.uniform(0, nx * res[0], 15), "y": rng.uniform(0, ny * res[1], 15)})
    pts.to_csv(tmp_path / "points.csv", index=False, float_format="%.17g")
    out = tmp_path / "out"
    lsm_reference.run_utils(rscript, tif, out, points=tmp_path / "points.csv", **options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm = ls.Landscape(tif, directions=options.get("directions", 8))
    for cc in (False, True):
        ref = read(out, f"centroids_{str(cc).lower()}")
        assert_same(lm.centroids(cc)[["class", "id", "x", "y"]], ref[["class", "id", "x", "y"]], f"centroids {cc}")
    ref = read(out, "nearestneighbour")
    if ref is not None:
        ours = lm.nearest_neighbours(True).dropna(subset=["dist"]).sort_values(["id", "id_neighbour"])
        ref = ref.sort_values(["id", "id_neighbour"])
        assert_same(ours[["id", "dist", "id_neighbour"]].astype(float), ref[["id", "dist", "id_neighbour"]], "nn")
    k = len(lm.classes)
    for nb in (4, 8):
        assert_same(lm.adjacencies(nb), matrix(out, f"adjacencies_{nb}_full_false", (k, k)), f"adjacencies {nb}")
    b_opts = {"consider_boundary": options.get("consider_boundary", False), "edge_depth": options.get("edge_depth", 1)}
    assert_same(lm.boundaries(patch_id=True, **b_opts), matrix(out, "boundaries_patches", lm.shape), "patches")
    for pid in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = ls.raster_boundaries(tif, patch_id=pid, **b_opts)
        assert_same(ours, matrix(out, f"boundaries_values_{str(pid).lower()}", lm.shape), f"values {pid}")
    ref = read(out, "extract")
    ours = lm.extract_metrics(pts.to_numpy())
    loose = ours["metric"].astype(str).str.startswith(NOT_BITWISE).to_numpy()
    assert list(ours["extract_id"]) == list(ref["extract_id"])
    assert_same(ours.loc[~loose, "value"], ref.loc[~loose, "value"], "extract")


# -- behaviour -------------------------------------------------------------------------


SMALL = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 2]])


class TestUtilities:
    def test_centroids_by_hand(self):
        c = ls.Landscape(SMALL, res=10).centroids()
        # terra's cell centres of a matrix: x = (col + 0.5) * 10, y from the top
        np.testing.assert_allclose(c["x"], [(5 + 15 + 5) / 3, (25 + 15 + 25 + 25) / 4, 10])
        np.testing.assert_allclose(c["y"], [(25 + 25 + 15) / 3, (25 + 15 + 15 + 5) / 4, 5])

    def test_patches(self):
        lm = ls.Landscape(SMALL)
        p = lm.patches()
        assert sorted(p) == [1, 2, 3]
        labels = lm.labels()
        np.testing.assert_array_equal(np.where(SMALL == 2, labels, np.nan), p[2])
        assert list(lm.patches(classes=3)) == [3]
        with pytest.raises(ValueError, match="not present"):
            lm.patches(classes=9)

    def test_boundaries_are_the_complement_of_core(self):
        data = lsm_scenes.clustered((40, 50), [1, 2, 3], seed=3, na_fraction=0.05)
        for depth in (1, 2, 3):
            lm = ls.Landscape(data, edge_depth=depth)
            b = np.asarray(lm.boundaries(edge_depth=depth))
            core_cells = np.nansum(b == 0) * 1e-4
            assert core_cells == pytest.approx(lm.landscape_metrics("tca")["tca"][0])

    def test_extract_on_edges_and_outside(self):
        lm = ls.Landscape(SMALL, res=1)  # extent 0..3 x 0..3
        pts = np.array([[0.0, 3.0], [3.0, 0.0], [1.0, 2.0], [-1.0, 1.0], [2.5, 0.5]])
        out = lm.extract_metrics(pts, what="lsm_p_area")
        # (0, 3): top-left cell; (3, 0): bottom-right (last column / row); (1, 2): right/down of the corner
        assert out["extract_id"].tolist() == [1, 2, 3, 5]
        assert out["id"].tolist() == [1, 2, 2, 2]

    def test_extract_ids_and_layers(self):
        pts = np.array([[0.5, 2.5], [2.5, 0.5]])
        out = ls.extract_metrics([SMALL, SMALL], pts, extract_id=["a", "b"], what=["lsm_p_area", "lsm_p_perim"])
        assert out["layer"].tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
        assert out["extract_id"].tolist() == ["a", "a", "b", "b"] * 2
        with pytest.raises(ValueError, match="patch level"):
            ls.extract_metrics(SMALL, pts, what="lsm_c_ca")

    def test_check_landscape(self):
        with pytest.warns(UserWarning):
            df = ls.check_landscape(np.array([[1.5, 2.0]]))
        assert df["class"][0] == "non-integer" and df["OK"][0] == "✖"
        with pytest.warns(UserWarning, match="not metric"):
            df = ls.check_landscape([SMALL, SMALL])
        assert df["layer"].tolist() == [1, 2] and df["OK"].tolist() == ["❓"] * 2
        assert df["n_classes"].tolist() == [3, 3]

    def test_correlation_errors(self):
        one = ls.Landscape(SMALL).metrics(what=["lsm_l_ta", "lsm_l_np"])
        with pytest.raises(ValueError, match="several landscapes"):
            ls.calculate_correlation(one)
        with pytest.raises(ValueError, match="more than one metric"):
            ls.calculate_correlation(ls.Landscape(SMALL).metrics(what=["lsm_c_ca"]))
        res = ls.calculate_correlation(ls.Landscape(SMALL).metrics(what=["lsm_c_ca", "lsm_c_te", "lsm_c_np"]),
                                       simplify=True)
        assert isinstance(res, pd.DataFrame)
        # np is constant: its correlations are NA and left out, its diagonal stays
        assert list(zip(res["metric_1"], res["metric_2"])) == [("ca", "ca"), ("te", "ca"), ("np", "np"), ("te", "te")]
        assert res["value"].tolist()[:2] == [1.0, pytest.approx(np.corrcoef([3, 4, 2], [4, 5, 3])[0, 1])]

    def test_out_of_core_not_available(self, tmp_path):
        import lsm_reference

        tif = tmp_path / "m.tif"
        lsm_reference.write_landscape(tif, SMALL)
        lm = ls.Landscape(tif, block_rows=64)
        for method in (lm.centroids, lm.nearest_neighbours, lm.boundaries, lm.patches):
            with pytest.raises(NotImplementedError):
                method()
